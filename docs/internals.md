# Internals

An MCP server that edits the open **Uncrashed: FPV Drone Simulator** track editor live (Steam app 1682970, Unreal Engine 4.27). An agent injected into the game calls the editor's own functions, so everything placed behaves and saves like a hand-placed object. The server exposes 30 tools with short replies by default.

## Run it

```bash
python3.10 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m uncrashed_mcp
```

Use Python 3.10 or 3.11. The current `mcp` package has no wheels for Python 3.14.

The server speaks MCP over stdio.

Saving waits for the game's deferred JSON write to complete. A first save also initializes the new map's editor directory, so subsequent saves target the same map.

Maps are written to the Proton profile the running game uses:

`.../compatdata/1682970/pfx/drive_c/users/steamuser/AppData/Roaming/Uncrashed/<SteamID>/MapEditor/User/<MapName>/`

Override that with `UNCRASHED_MAPS_DIR`. Override the creator id with `UNCRASHED_STEAM_ID` (default: the most recently used profile folder).

## What an agent can call

| tool | purpose |
| --- | --- |
| `status` | map folder, game running, agent loaded |
| `search_assets` / `get_assets` | the 804 editor rows (`NewRow_*`), with sizes, pivots and bounds |
| `list_gates` | race-gate rows |
| `list_maps` / `inspect_map` | saved maps and their object counts |
| `load_live` | show a saved map in the open editor (base level, objects, gate transforms/sizes, laps) |
| `save_live` | save the open level with the game's own SaveMap |
| `agent_batch` | many agent commands in one game-thread pass: the fast way to build (thousands per call) |
| `agent_exec` | one agent command |
| `camera_pose` / `move_camera` / `release_camera` / `capture_view` | native camera + offscreen scene images; no focus/desktop switch by default |
| `check_collisions` | placed-object clashes / support connectivity; writes a report page |
| `check_foundations` | footprint support coverage, unsupported corners, base-plane gaps / penetration |
| `check_environment` | landscape penetration, base / placed foliage collision candidates, visual canopy warnings |
| `get_object_bounds` | live world AABB, optional oriented mesh bounds, transform (metres) |
| `get_world_bounds` | loaded placed-object / terrain extents; explicit playable bounds or unknown |
| `sample_terrain` | bare landscape heights at up to 256 XY points; ignores trees and placed objects |
| `inspect_site` | pre-placement terrain grid, recognised base flora and placed-object inventory |
| `clearing_zones` | preview/apply/remove persistent, reversible base-foliage exclusions |
| `list_groups` / `edit_group` | group inventory, transform/clone/delete/regroup; preview first |
| `undo_edit` / `edit_history` | durable, map-scoped transaction journals and conflict-checked undo |
| `define_assembly` / `check_assembly` | saved group roles, measured foundation anchors and cantilever-aware connectivity |
| `agent_log` / `agent_unload` | agent log and DLL hot-swap |

Everything else goes through the agent's command language (centimetres and degrees). The full list is in the server instructions; the common ones:

```
spawn <row> x y z [pitch yaw roll] [sx sy sz]   place a catalogue row (pivot)
gate <row> x y z [pitch yaw roll] / laps <n>      race gates, STARTGATE first
actors x y z radius [limit]                       objects near a point (address, row, transform)
transform <actor> x y z pitch yaw roll sx sy sz   '-' keeps a value
replace <actor> <row> / destroy <actor>
destroyat x y z [tol] [count] / destroybox x0 y0 x1 y1 [z0] [z1] [row]
gates / mapinfo / save [folder] [title] / floating [gap]
collide <file> [x0 y0 x1 y1] [gap] / trace x y z_from z_to   collision facts / one vertical ray
drop <actor|last> [step_cm] [max_cm] [embed_cm] [dry]   rest an object exactly on what is under it (last = the last spawn)
```

Replies are cut to 40 lines unless `max_lines` asks for more (at most 400). `agent_batch` is quiet by default and reports only its summary and failures.

The capable-agent shortcut: the catalogue is `src/uncrashed_mcp/data/assets.json`. An agent with a code sandbox can filter it directly instead of calling `search_assets`; never read it whole (about 200 KB).

`BaseBoard` (Empty board) is flat at z = 0. Other bases, such as `LongRange1`, have real terrain: inspect `agent_exec("mapinfo")` before assuming a flat floor. `sample_terrain([[x, y], ...])` returns landscape heights in metres (null outside landscape). Sample a foundation's entire footprint, not just its centre. Native equivalent: `terrain x y z_from z_to` in centimetres. Unlike `trace`, this ignores foliage and placed structures; unlike `drop`, it measures bare ground, not the top of a platform. Restart an already-running MCP process after upgrading to expose the new tool.

Rows marked hidden are missing from the in-game picker and still spawn. Asphalt is one of them.

## Background capture, clearing zones, groups and assemblies

`capture_view` renders a fresh, UI-free image through the engine. Optional pose does **not** move the editor camera.
It never activates a window or falls back to foreground capture. Native camera control no longer uses an external
memory-writing helper. An explicit `backend="window", foreground=True` is the only capture mode allowed to focus/switch desktops.

New managed tools preview by default. Group changes journal before-state, support undo after handle changes, and refuse
intervening edits. Zone/assembly intent is saved in portable map sidecars; save-as copies intent, not undo history.
**Clearing zones require the resident DLL**; stock game alone ignores them. GPU-only landscape grass is reported
unavailable; supported-source-only clearing needs explicit `allow_partial=True`. Assembly checks are conservative
connectivity diagnostics, not load-bearing certification. See [workflow, limitations and verification](managed-editing.md).

## Collision check

`check_collisions` scans placed-object clashes and disconnected support groups, each with coordinates in metres and a camera pose
for `move_camera`. A clean result does **not** mean every building has full foundation support or clears terrain / trees.
Also run `check_foundations` and `check_environment`; see [bounds and environment audit details](environment-audits.md).

| kind | meaning |
| --- | --- |
| buried in rock | a rock or cliff cuts through a container, wagon, wall or prop |
| vehicle clipping | a vehicle inside a wall, barrier or another vehicle |
| inside each other | containers, buildings or big props occupying the same space |
| prop inside something | a smaller prop deep inside another object |
| vehicle sunk in | wheels below the surface the vehicle stands on (8 cm or more) |
| floating | not connected to the ground by anything (balloons are exempt) |
| slightly above support | under a metre of air, or a small wall-mounted item; off unless `include_minor` |

It writes `/tmp/uncrashed-collide/report.json` and `report.html`, a zoomable plan with every issue and a filterable
list. Default scan extents come from live placed-object bounds, not a fixed 6 km square. Scans are split into up to 100 passes
based on object count. Explicit XY regions remain supported. Missing result files are errors, never an empty clean report.

How it decides, so the list stays short:

- The agent's `collide` verb uses `KismetSystemLibrary` box and line traces against each object's real collision
  (complex collision, every object type except Pawn). Nothing comes from catalogue sizes.
- Intersections use the object's oriented box shrunk by 25 cm (or 35 %), and both objects must hit each other.
  Touching faces, a car on a car and a ladder on a wall therefore do not count. A half-size box marks deep overlaps
  as high severity. Plants, ground tiles, and walls, scaffolds or rocks built from overlapping parts are skipped.
- Floating is a graph question. 13 rays under each footprint, plus everything its box grown by 15 cm touches,
  link objects to what holds them. Whatever has no path to the landscape, the floor, a ground tile or a building
  floats, one issue per connected group, with the air gap measured below it. Balloons and catalogued plants hold nothing up.
  Objects below z=0 are no longer automatically grounded; native support rays ignore base foliage.
- Blueprint buildings (Kitbash `Bldg *`) report their bounds instead, because their upper floors have no query
  collision. A trace finds their roof at about 5 m, so the box decides.

`drop <actor>` fixes the commonest case, something floating above or sunk into what is under it. In a batch,
`spawn ...` followed by `drop last` places an object and rests it in one go; the server instructions tell agents to
do this for anything that stands on something. It samples the
object's footprint every 8 cm: a ray against the object alone finds its underside, a ray down from there finds the
surface below, and the smallest gap is how far it moves (up when negative). Things resting on it are ignored as
support. Stack bottom-up: drop the lowest object first. Objects placed or moved earlier in the same batch are
already visible to these rays.

By hand: run `collide <file> [x0 y0 x1 y1] [gap_cm]` (centimetres) through `agent_exec`, then
`python -m uncrashed_mcp.collide <files>` for the same JSON and page. `trace x y z_from z_to` shows what a
vertical ray hits at one spot, once per collision object type.

## How live editing works

`native/agent.c` builds `native/uncrashed_agent.dll`. The first live call injects it into `Uncrashed-Win64-Shipping.exe` with `native/inject.exe` (same Proton prefix, `LoadLibraryW`). It stays resident and polls `/tmp/uncrashed-agent-cmd.txt`; each command runs on the game thread under a fault guard, so a bad call is reported instead of crashing the game. `agent_unload` frees it, and the next call injects a rebuilt DLL without restarting the game.

The agent calls the track editor's own Blueprint functions through `ProcessEvent`: `Spawn Asset`, `SpawnTrackAsset`, `SaveMap`, `LoadLevel`, `UpdateGateListUI`, and engine functions such as `K2_SetActorLocationAndRotation` and `K2_DestroyActor`. Objects therefore behave and save exactly like ones placed by hand.

Game updates: no game address is a constant. `FNamePool` and `GUObjectArray` are found by their shape in `.data`, the GWorld slot by the live `UWorld` it holds, and `ProcessEvent` by the vtable slot that actually computes `Add_IntInt(2, 3) == 5`. Every Blueprint and engine field is looked up by name through reflection. `agent_exec scan` re-resolves and prints what it found. Only an engine upgrade past UE 4.27 would need code changes; the agent then reports "layout mismatch" instead of guessing. Camera control, render-target capture and instance masking use reflected UFunctions/fields on the game thread too.

Build the DLL (zig or mingw-w64):

```bash
x86_64-w64-mingw32-gcc -O2 -shared -static -o native/uncrashed_agent.dll native/agent.c
# or: zig cc -target x86_64-windows-gnu -Wno-date-time -O2 -shared -o native/uncrashed_agent.dll native/agent.c
```

## Where the numbers come from

The asset table, mesh bounds and save layout were read from the installed game, not copied from screenshots:

- Pak indexes and the live process confirm `DT_MapEditorAssets` and the `MapEditor/User` save folder under the Steam profile in `AppData/Roaming`.
- Workshop maps confirm the JSON keys, quaternion rotations, centimetre translations, gate rows and the `.info` file.
- Mesh bounds (origin and extent) decide the pivot correction. Several buildings have a pivot at a corner; placing that pivot on a lot centre drops the mesh tens of metres away.
- Ground materials share one 4 m tile and swap a triplanar material. Asphalt, pavements, dirt, gravel, sand and wood are different rows of that tile. Each material has its own height band so neighbouring surfaces do not z-fight.
- Concrete walls and floors are triplanar, so a scaled panel keeps a real texture scale.

`python -m unittest discover tests` checks rotation maths, reply parsing, terrain sampling, deferred saves,
audit validation / classification, negative-elevation grounding, foliage support exclusions, dynamic scan extents,
non-focusing capture policy, durable group undo, assembly intent and native zero-scale restoration guards.
The opt-in [live smoke script](../scripts/live_managed_smoke.py) verifies reversible changes without saving/clearing the existing map.
