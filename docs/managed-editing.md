# Background views and managed editing

Tool-level units are metres/degrees. Native commands use centimetres/degrees.
Restart an already-running MCP server after upgrading; rebuild/hot-load the DLL too.

## Camera and screenshots

`move_camera` now uses game-thread UFunctions and reflected editor state, not
`/proc/<pid>/mem` writes or a spinning helper process. `hold=True` temporarily
disables editor camera movement; `release_camera` restores its prior enable state.
A 120-second native watchdog releases abandoned holds. Pawn changes/DLL unload
release them too. The last pose stays.

`capture_view` defaults to `backend="engine", foreground=False`:

- A temporary, untagged `SceneCapture2D` actor renders into a dedicated sRGB render
  target. It is not an editor asset and is destroyed after capture.
- Optional XYZ/yaw/pitch specify an independent render pose. **Editor camera does
  not move**, including its hold state.
- No window lookup, activation, keyboard input or desktop switching. A render
  failure is an error, never an implicit foreground fallback.
- Several explicit renders warm view history. Small images use 2× spatial
  supersampling within the pixel cap to reduce foliage dither. Readback/export is a
  later game-thread job. A 30-second watchdog cleans abandoned captures.
- PNG defaults to 800×450. Width/height are configurable: 64–4096 per axis, at most
  4,194,304 pixels. UI is absent. Exposure/postprocessing/temporal rendering can
  differ from the editor viewport; blank/uniform frames are rejected. Capture-only
  exclusions hide editor gizmo actors, debug line batches and water collision-box
  components without changing world visibility/collision. Material-based editor
  guides (such as the stock water grid) can still appear.

`backend="window"` captures current editor pixels and includes UI. It never moves
or focuses the camera/window by default, but **occlusion, minimization and other
desktops can produce stale/obscured pixels**. Pixel variance cannot prove freshness.
Only explicit `backend="window", foreground=True` permits activation, with previous
focus restored afterward. That option can visibly switch desktops. No backend
falls back to it automatically.

The preview script `designs/cloudreach_expansion/previews.py` uses engine capture,
not its old independent ImageMagick/window-activation path.

## Inspect before placing

`inspect_site(bounds_min, bounds_max, spacing=5, min_foliage_height=2)` returns:

- Bare landscape height grid, including center axes, at most 225 samples. Actual
  spacing/elevation range/missing rays reported. No landscape is **not** z=0.
- Visual AABBs for recognised base flora: painted foliage instances, available
  landscape-generated instances and individual base static-mesh plants.
- Placed-object AABBs overlapping the XYZ box.
- Full report under `/tmp/uncrashed-managed/`; `top` limits returned object lists.

Plant classification uses recognised mesh paths, excluding known rock meshes;
unknown/custom plant sources are not universally discoverable. GPU-only landscape
grass can report zero CPU instances despite rendered data. Such sources are
reported unavailable, not counted as empty verified space. This is **not** a full
inventory of base rocks, cliffs, collision volumes or hidden unloaded geometry.

## Persistent clearing zones

Save the map first, then use `clearing_zones`:

```json
{"action":"upsert","name":"courtyard","center":[100,50,15],"size":[35,25,40],"yaw":30,"preview":true}
```

Review candidates, then repeat with `preview=false`. Defaults are previews.
`action="list"` reports saved intent/native state. `action="remove"` with
`preview=false` removes one exclusion and restores flora no longer covered by any
other zone.

Zones are yaw-oriented prisms tested conservatively against visual foliage AABBs.
A crown overlap hides the **whole instance/actor**, not just branches inside the
prism. Overlapping zones form a union. Terrain, placed assets and gates are not
removed or edited.

Instances are masked with the engine's `UpdateInstanceTransform` (zero scale),
retaining complete original local transforms. Individual recognised base plant
actors retain hidden/collision flags and are hidden with collision disabled.
`RemoveInstance` is never used, avoiding index churn. Restoration checks component
identity/serial, instance position and masked state before overwriting anything.
Zero-scale matrix readback loses quaternion; restoration uses the retained original
quaternion, not the readback value. Conflicts retain originals and refuse DLL unload
rather than silently abandoning restoration data.

GPU-only grass cannot be individually cleared through this API. Preview reports
unavailable sources. Apply refuses unless `allow_partial=true` explicitly accepts
supported-source-only clearing. Newly streamed unavailable sources remain reported;
never interpret a configured zone as proof that every rendered blade is absent.

Two portable files live next to the saved map JSON:

- `UncrashedMCP.json`: version/base, zones and assembly roles.
- `UncrashedMCP.zones`: validated native zone configuration.

Keep **both** when copying/exporting a map. The resident DLL polls configuration,
map/base/loaded-level changes and reapplies masks to fresh instance identities.
It restores masks on directory changes and before unload. No MCP call is needed
for subsequent reapplication while DLL remains resident. **The unmodified game
without this DLL ignores these sidecars**; they are not a stock-game foliage edit.
The configuration is durable; original runtime transforms are held by the DLL.
Restarting the game reloads original base flora, then the DLL reapplies intent.

`save_live` save-as copies portable intent, not handle-based undo journals. A target
that already has metadata requires an explicit merge instead of silently replacing
another map's intent. `load_live` explicitly refreshes exclusions after loading.

## Groups and undo

`list_groups` lists actual saved editor group tags; bounds describe actor pivots,
not mesh extents. Names use 1–96 ASCII letters/digits/`_.-` and are case-insensitive
like Unreal FNames. `None`, `Group`, `-` and `.` are reserved.

`edit_group` actions:

- `transform`: translate, quaternion-compose pitch/yaw/roll, uniformly scale.
- `clone`: fresh `target_group`, optional transform.
- `delete`: selected group/subset only.
- `assign`: regroup selected objects into `target_group`.

Catalogue placed assets only: gates and base actors are excluded. An explicit
group or actor list is mandatory; nothing defaults to the whole map.
`prefix=true` includes dot/underscore-separated descendants (`site.frame` and
`site_frame`, not `siteboat`). Clone preserves subgroup suffixes under its fresh
target branch.
An actor list further restricts selection and cannot escape a supplied group.
Pivot defaults to the center of selected actor pivots. Uniform scaling avoids
unrepresentable shear for rotated members. Up to 5,000 changed objects per edit.

Preview first, then `preview=false` to apply. Before mutation, a durable JSON journal
is written under `UncrashedMCP.history/<transaction>.json`. Native code validates
all handles/rows/groups/transforms before mutation. Creations happen first,
deletions last; explicit failures attempt rollback. **A native fault/timeout is not
an ACID transaction guarantee**: inspect `recovery_required` journal before retrying.

`undo_edit(transaction, preview=true)` checks complete affected groups against the
recorded after-state, using row/group/transform multisets rather than trusting old
handles. It works after handle changes on reload and refuses intervening edits.
Apply with `preview=false`. Successful undo itself has a transaction ID: undo that
inverse to redo. `edit_history` lists IDs/statuses. Histories exceeding 256 MiB must
be archived before more edits; they are not silently discarded.

Move/rotate/scale preserve actors. Clone and delete-undo regenerate catalogue assets:
**arbitrary Blueprint variables, material overrides, animation/random state are not
serialized**. These limitations are returned in tool results. Group edits do not
implicitly save geometry; use `save_live` afterward.

Protocol operations are serialized across Python threads/clients with a reentrant
session lock and `flock`; a batch file cannot be overwritten by another current
client mid-operation. Old MCP processes must be restarted to participate.

## Assembly-aware support

Split actual footings into a saved group with `edit_group(action="assign",...)` if
needed. Define intent using `define_assembly(name, roles, preview=false)`:

```json
{"name":"lab","roles":{"lab.footings":"foundation","lab.frame":"structure","lab.roof":"cantilever","lab.signs":"decoration"},"preview":false}
```

Omit `roles` from `define_assembly` to inspect a definition; omit `name` too to list
saved assemblies.

`check_assembly("lab")`:

1. Measures explicit foundations against bare landscape/base geometry, ignoring
   placed members/foliage. A 5×5 grid compares terrain with actual vertical collision
   spans when available; bounds-only/composite fallbacks are unknown anchors.
2. Requires configurable anchor coverage and at least three samples. Missing rays,
   partial coverage and buried samples remain visible in results.
3. Builds contact graph from **reciprocal OBB-versus-collision hits**.
4. Tests structural/cantilever paths to measured foundation anchors. Plants and
   decoration never carry structural support.

An intentional overhang needs an anchor path, not full ground-footprint coverage.
This is **not** exact mesh-mesh contact, a load/deflection/stability solver, or
engineering certification. Bounds can bridge empty mesh space. Hollow, tilted and
composite foundations need manual review; external assemblies are not inferred as
roots. Continue using `check_environment` for terrain/trees. Full graph/report JSON
is saved under `/tmp/uncrashed-managed/`.

## Verification

```bash
.venv/bin/python -m unittest discover tests
zig cc -target x86_64-windows-gnu -Wno-date-time -O2 -Wall -Wextra -Werror \
  -shared -o /tmp/uncrashed_agent.next.dll native/agent.c
# Explicit opt-in: temporary cubes/masks, sandboxed metadata; no existing map save.
.venv/bin/python scripts/live_managed_smoke.py --allow-temporary-edits
```

Live smoke checks transforms, clone/delete/regroup + undo, regenerated handles,
measured footing/cantilever connectivity, zone union, directory changes, DLL reload,
restoration (including collision-ray disappearance/reappearance) and camera-independent capture. It archives sandbox evidence under
`/tmp/uncrashed-managed/`, deletes only uniquely tagged test objects, restores save
directory, checks original map JSON hash and placed-object/gate counts. It does not
clear/reload the user's whole map. Actual minimized-window and base-stream reload
coverage must be tested separately; directory/DLL reload checks are not those tests.
