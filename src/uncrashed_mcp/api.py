"""Transport-neutral operations on the Uncrashed track editor: the single source of truth for every transport.

One plain function per operation, registered in OPERATIONS together with its CLI hints. The functions return
Python values and raise on error; cli.py and the optional MCP adapter (server.py) only convert arguments and
results. Nothing here may import `mcp`, and `live` (PIL) is imported inside the camera / capture operations.

Operation-level units are metres and degrees (Z up, yaw 0 = +X, 90 = +Y). The agent's own command
language (agent_exec, agent_batch) speaks centimetres, like the game. Asset rows are
DT_MapEditorAssets ids (NewRow_N) from search_assets.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple

from . import agent, assemblies, clearance, collide, consolidate, environment, groups, managed, performance, scenery
from .catalog import load_catalog
from .mapfile import list_user_maps, load_map, sanitize_name
from .paths import game_running, maps_dir, steam_id, workshop_dir

RELOAD = "Call load_live with a map name to show a saved map in the open editor."

OUTPUT_LINES = 40      # default reply length for agent_exec / agent_batch
OUTPUT_LINES_MAX = 400  # hard cap, whatever the caller asks for
COLLIDE_DIR = Path("/tmp/uncrashed-collide")

# native verb_batch reads each line with fgets into char[4096]: a longer line would be run as two commands
BATCH_LINE_MAX = 4095
BATCH_RETRY = "successful lines are applied; retry only the failed lines"
BATCH_UNKNOWN = "batch state UNKNOWN: may be partly applied; check with exec mapinfo / actors before retrying"
BATCH_FAILURES = 20    # failed batch lines listed in a reply
MISMATCH_LISTED = 50   # scenery.set_collision cuts its `mismatched` list to this many actors

VERBS = """
Agent commands (one per line; centimetres and degrees; <actor> is a hex address from a reply):
  spawn <row> x y z [pitch yaw roll] [sx sy sz] [group=tag]   place a catalogue row; x y z is the pivot
  gate <row> x y z [pitch yaw roll]      race gate; STARTGATE first. DefaultOneW normal is local +Y: yaw=travel heading-90, roll=90 for a downward gate
  laps <n>                               lap count
  gates                                  list race gates in order
  actors x y z radius [limit]            placed objects near a point: address, row, loc, rot, scale
  get <actor> | bounds <actor>           one object's transform | world AABB, transform, mesh OBB
  worldbounds placed|terrain|playable    loaded scene extents; playable requires explicit PlayableBounds tag
  transform <actor> x y z pitch yaw roll sx sy sz   '-' keeps a value; to pick by position, find it with actors first
  replace <actor> <row>                  same transform, other catalogue row
  destroy <actor>                        delete one object or gate
  destroyat x y z [tol] [count]          delete the nearest objects whose pivot is within tol (default 50)
  destroybox x0 y0 x1 y1 [z0|-] [z1|-] [row|-]   delete every object whose pivot is inside (optionally one row only)
  clear                                  delete everything placed (asks nothing: be sure)
  mapinfo                                open map folder, title, base, object and gate counts
  save [folder|-] [title]                save with the game's own SaveMap ('-' = the open folder)
  setmap <folder> | base <row>           retarget the save folder | reopen the editor on another base (clears the scene)
  floating [gap] [limit] [x0 y0 x1 y1]   objects hovering more than gap above anything
  collide <file> [x0 y0 x1 y1] [gap]     collision facts of every placed object into a file (check_collisions reads them)
  trace x y z_from z_to                  what a vertical ray hits, once per collision object type
  terrain x y z_from z_to                landscape height only, ignoring foliage and placed objects
  drop <actor|last> [step] [max] [embed] [dry]   rest it exactly on what is under it (lifts it if sunk in); last = the last spawn
  cells <cell_m> <rows|!rows|*> [zmax_m] occupancy grid (metres here), comma-separated row filter
  console <cmd> | cvar <name> | raise [off]   console command | read a console variable | game window on top
  collision query|on|off|flythrough|nophysics <sel> [preview]   sel: actor=<hex> group=<tag> row=<row> box=... (set_collision wraps it)
  collision status | channels            override counters | the drone's object channels (run while flying)
  perf <file> [x0 y0 x1 y1] | rowflags <file> | frametime   performance scan | row data flags | frame timing
  gatebox <file> [channel] | sweep <request> <output>       gates + collision components | swept craft volume
  debug: ping scan resolve names objects funcs func props rows fname read write call
Example: spawn NewRow_1 1500 -2000 0 0 90 0 1 1 1   -> a container at x=15 m, y=-20 m, turned 90 degrees.
""".strip()

INSTRUCTIONS = f"""
Edit Uncrashed: FPV Drone Simulator's track editor live. An agent injected into the game calls the editor's own
functions (Spawn Asset, SpawnTrackAsset, SaveMap); nothing is clicked and every change is visible at once.

Work like this: look up rows with search_assets / get_assets, write the commands, send many at once with
agent_batch (one game-thread pass, thousands of placements), look with capture_view, fix, then save with
save_live. agent_exec runs a single command. Check mapinfo first: only BaseBoard is a flat grid at z=0.
On landscape bases, sample_terrain measures bare ground through trees and structures (metres); sample the
whole footprint before choosing a foundation elevation. It does not measure platform or building roofs.

Anything that should stand on something (a car on the road, a crate on a shelf, a car on a car) goes down with
drop: spawn it a little high, then "drop last" on the next line of the same batch rests it exactly on what is under
it, using the real shapes of both. Stack bottom-up. Ground tiles sit 9 to 13 cm above z=0, so an object spawned at
z=0 on a tile is sunk in unless it is dropped. drop also lifts objects that are already sunk in or overlap.

Before saving a big build, run check_collisions for placed-object clashes / disconnected groups, then
check_foundations for unsupported footprint portions and check_environment for landscape / base-foliage
conflicts. A clean check_collisions is not full support or environment validation. All audits are read-only.
Use get_object_bounds for world AABB / optional mesh OBB and get_world_bounds for placed or terrain extents;
playable bounds are unknown unless explicitly tagged, never inferred from landscape size.

capture_view defaults to fresh engine render-target capture: no window activation, desktop switch, UI or editor
camera movement. Never silently use foreground fallback. Only explicit backend='window', foreground=True may focus.
move_camera is native, without external memory writes/helper; holds have a 120-second abandonment watchdog.
Before construction use inspect_site for bare-ground grid, recognised flora and placed-object inventory. GPU-only
landscape grass sources are reported unavailable. clearing_zones previews by default, persists portable intent and
masks/restores supported base flora while DLL is resident; stock game alone ignores sidecars. Partial source coverage
requires explicit allow_partial. Preserve both UncrashedMCP sidecars when copying maps.
Use list_groups/edit_group for explicit, preview-first transform/clone/delete/regroup and undo_edit/edit_history for
journaled inverses. Clone/delete undo does not preserve arbitrary Blueprint/material runtime state. No edit auto-saves.
For built-up structures define_assembly assigns foundation/structure/cantilever/decoration roles to groups;
check_assembly measures anchors and reciprocal contact paths, not load capacity. Intentional overhangs need a path,
not full ground-footprint coverage. Continue environment checks; do not call connectivity engineering certification.

Replies are short by default; ask for more lines only when needed. Pivot vs. bounds: spawn takes the pivot;
get_assets gives origin_cm / extent_cm so the bottom of a mesh can be put on the ground.

Non-colliding scenery: set_collision (preview first) switches placed objects selected by actors, group, row or box
to flythrough (drone/vehicle channels ignored, no rigid-body contact; traces and audits still see the object), off
(no collision at all), nophysics, or back to on (exact restore); physics_off/physics_on are live-only simulation
toggles. Gates only with force. persist=True writes UncrashedMCP.collision and the resident DLL re-applies it after
reloads while the editor is open, but the stock game without the DLL ignores it: for maps other people fly, prefer
non-colliding catalogue rows (search_assets collision='noncolliding', noncolliding_alternatives). collision_state
reports per-object and per-component collision facts. undo_collision reverts using the transaction id. Which
channel the drone collides on is unverified: run agent 'collision channels' while flying and pass channels=[...] if
it is not Pawn/PhysicsBody/Vehicle. Object rules follow catalogue row plus pivot (2 cm), so moving an object
detaches it from its rule; group rules survive moves. Keep UncrashedMCP.collision with the map.

For big maps run check_performance (offline with name=<saved map>, or live on the open editor; measure=True samples
frame time, optionally at camera poses). Its budgets are heuristic UE4 defaults, so override them. Live grading
counts placed objects only; *_base figures are informational. To cut object count, consolidate_scenery merges edge-
to-edge plain-box triplanar tiles (ground, Concrete Wall01/Floor01/Column...) into scaled copies. Preview first,
then apply with preview=False on a saved map (the largest merge goes first as a canary, and each replacement is
verified and auto-undone on mismatch). Undo with undo_edit per returned transaction, newest first. Non-triplanar
rows are never merged; thin them by hand.

For flyability use check_route_clearance. It sweeps the craft volume along a smooth line through the race gates
(source='live' or 'map:<name>') or through waypoints (source='points'). Profiles: 5inch, tiny_whoop, cinewhoop,
7inch, fixed_wing, or car/truck for roads; dims_m and margin_m customise them. It reports collisions, gate openings
too small (probed from each gate's centre), tight spots, low ceilings or ground, and road corridor width, each with
a move_camera pose. Overlap-only triggers and flythrough objects are not obstacles. The collision channel defaults
to Pawn, which is unverified for the drone; pass channel= (e.g. PhysicsBody, game3, '16') to change it. It is a
geometric check of a spline, not a flight test.

{VERBS}
""".strip()

GUIDE = """
Uncrashed track editor from the shell: an agent DLL in the game calls the editor's own functions (Spawn
Asset, SpawnTrackAsset, SaveMap). Nothing is clicked; changes show at once.

BASICS
Help: `./uc help` (commands: ! acts at once, ~ previews unless --apply), `./uc help <command>`,
`./uc help verbs` (agent commands), `./uc help output` (reply format). Commands use metres and degrees
(Z up, yaw 0 = +X, 90 = +Y); agent commands (exec = agent_exec, batch = agent_batch) use centimetres. Rows
are catalogue ids (NewRow_N). Replies are short: ask for more (--max_lines, --top, --limit) only when
needed. --json prints JSON, --pick a,b.c keeps fields.
Exit codes: 0 ok; 1 raised (a live change may be partly applied); 2 usage error (nothing ran); 3 ran but
reported failures. Every failure prints one stderr line `error: ...`; pipes need `set -o pipefail`.
No game needed: status, list_gates, list_maps, inspect_map, search_assets, get_assets,
noncolliding_alternatives; check_performance / consolidate_scenery with a saved map name.

BUILD
search_assets / get_assets (rows) -> write commands -> batch -> capture_view -> fix -> save_live.
`./uc exec mapinfo` first: only BaseBoard is a flat grid at z=0. On landscape bases sample_terrain measures
bare ground through trees and structures (not platform or building roofs): sample the whole footprint
before choosing a foundation elevation. Before building, inspect_site lists the bare-ground grid,
recognised flora and placed objects; GPU-only landscape grass is reported unavailable.
spawn takes the pivot; get_assets gives origin_cm / extent_cm to put the bottom of a mesh on the ground.
Batch: generate commands with a script kept in the project (not /tmp), then `./uc batch plan.txt` or
`python3 plan.py | ./uc batch -`: one game-thread pass, thousands of placements. Failed lines print as
`#<index> L<line> <error> << <command>`; all others ARE applied: resend only the failed ones.
drop: what stands on something (car on road, crate on shelf) is spawned a little high, then `drop last` on
the line directly after its spawn (same batch only) rests it exactly on what is under it, using the real
shapes of both. Stack bottom-up. Ground tiles sit 9 to 13 cm above z=0, so an object spawned at z=0 on a
tile is sunk in unless dropped; drop also lifts sunk-in or overlapping objects.
Quote an exec command containing * ! # ; | < > & as one argument.

LOOK
capture_view renders a fresh engine image to a PNG (--out file.png or dir/; view the file yourself): no
window activation, desktop switch, UI or editor camera movement. x y z [yaw pitch] set an independent pose.
No silent foreground fallback: only `--backend window --foreground` may focus the game.
move_camera / camera_pose / release_camera drive the editor camera natively; a hold ends after a 120-second
watchdog. With several agents, give capture_view an explicit pose and let one agent own the editor camera.
get_object_bounds: world AABB / optional mesh OBB. get_world_bounds: placed or terrain extents; playable
bounds are unknown unless explicitly tagged, never inferred from landscape size.

NO PREVIEW, NO UNDO
Agent verbs clear, destroy, destroyat, destroybox, replace, base (reopens the editor, clears the scene) and
setmap (retargets Save). load_live removes everything placed first; save_live overwrites the map. No edit
auto-saves.

SLOW COMMANDS
check_collisions (a minute per 27,000 objects), load_live, big batches, `consolidate_scenery --apply` and
check_route_clearance outlast default shell timeouts: raise the timeout or run them in the background.
NEVER re-send a command that timed out, failed or was interrupted: it was sent and may still run or be
partly applied (only an error containing `nothing was sent` is safe to retry). First check `./uc last`
(last finished reply), `./uc exec mapinfo`, `./uc edit_history`, `./uc log` (agent_log). stderr `waiting
for agent lock ...` or `agent is finishing an earlier job` means wait: do not re-send. After a batch error
without a summary line later commands may time out although they run, until agent_unload (`./uc unload`),
which restores masks/holds and unloads the DLL.

AUDITS (read-only)
Before saving a big build: check_collisions (placed-object clashes, disconnected groups), then
check_foundations (unsupported footprint portions) and check_environment (landscape / base-foliage
conflicts). A clean check_collisions is not full support or environment validation.
define_assembly gives groups foundation/structure/cantilever/decoration roles; check_assembly measures
anchors and reciprocal contact paths, not load capacity. Intentional overhangs need a path, not full
ground-footprint coverage; connectivity is not engineering certification. Keep running check_environment.
check_route_clearance sweeps the craft volume along a smooth line through the race gates (live, map:<name>)
or waypoints (points). Profiles 5inch, tiny_whoop, cinewhoop, 7inch, fixed_wing, car/truck (roads); --dims_m
and --margin_m customise. It reports collisions, gate openings too small (probed from the gate centre),
tight spots, low ceilings or ground and road corridor width, each with a move_camera pose. Overlap-only
triggers and flythrough objects are not obstacles. The channel defaults to Pawn, unverified for the drone
(--channel PhysicsBody, game3, 16). A geometric check of a spline, not a flight test.
check_performance: offline with a saved map name, or live (--measure samples frame time, optionally at
--poses). Budgets are heuristic UE4 defaults: override them. Live grading counts placed objects only;
*_base figures are informational.

EDITS (~ commands: preview first, then --apply)
list_groups / edit_group: transform, clone, delete, regroup. undo_edit / edit_history: journaled inverses;
clone/delete undo does not preserve arbitrary Blueprint/material runtime state. undo_edit of an interrupted
edit recovers its journal in the preview; that undo also reverts later changes to its groups.
clearing_zones persists portable intent and masks/restores supported base flora while the DLL is resident;
the stock game alone ignores sidecars. Partial source coverage needs --allow_partial. Preserve both
UncrashedMCP sidecars when copying maps.
consolidate_scenery merges edge-to-edge plain-box triplanar tiles (ground, Concrete Wall01/Floor01/Column...)
into fewer scaled copies. NAME only previews a saved map file. To apply: save your work, load_live the map,
run without NAME (preview, then --apply). The largest merge goes first as a canary; each replacement is
verified and auto-undone on mismatch. Undo with undo_edit per returned transaction, newest first.
Non-triplanar rows are never merged; thin them by hand.

COLLISION
set_collision (previews unless --apply) sets placed objects picked by --actors, --group, --row or
--box_min/--box_max to flythrough (drone/vehicle channels ignored, no rigid-body contact; traces and audits
still see them), off (no collision at all), nophysics or on (exact restore); physics_off / physics_on are
live-only simulation toggles. Gates only with --force. persist (default; --no-persist = live only) writes
UncrashedMCP.collision: the resident DLL re-applies it after reloads while the editor is open, the stock
game without the DLL ignores it. For maps others fly prefer non-colliding catalogue rows
(`search_assets --collision noncolliding`, noncolliding_alternatives). collision_state reports per-object
and per-component facts; undo_collision reverts by transaction id. An interrupted `set_collision --apply`
can leave a collision_prepared journal that undo_collision refuses: check collision_state, then re-apply
the wanted mode (on restores the originals). The drone's channel is unverified: run
`./uc exec collision channels` while flying and pass --channels if it is not Pawn/PhysicsBody/Vehicle.
Object rules follow catalogue row plus pivot (2 cm): moving an object detaches it from its rule; group
rules survive moves. Keep UncrashedMCP.collision with the map.
""".strip()


@dataclass(frozen=True)
class Text:
    """Plain-text result. ok is False when the agent replied `failed` or a batch reported failures;
    failure is then the one-line reason."""
    text: str
    ok: bool = True
    failure: str = ''


class Param(NamedTuple):
    name: str
    annotation: Any
    default: Any  # inspect.Parameter.empty when the parameter is required


@dataclass
class Op:
    name: str
    fn: Callable
    summary: str
    aliases: tuple[str, ...] = ()
    positional: tuple[str, ...] = ()
    variadic: str | None = None
    rest: str | None = None
    stdin: str | None = None
    live: str = 'yes'  # 'no' never needs the game | 'mixed' offline with a saved map name | 'yes'
    effect: str = 'read'  # 'read' | 'write'
    caution: str = ''
    examples: tuple[str, ...] = ()
    check: Callable[[dict], str | None] | None = None  # one-line reason when a returned payload reports a failure

    def params(self) -> list[Param]:
        """Public parameters with resolved annotations. `_`-prefixed parameters are hidden from every transport."""
        signature = inspect.signature(self.fn, eval_str=True)
        return [Param(p.name, p.annotation, p.default) for p in signature.parameters.values()
                if not p.name.startswith('_')]


OPERATIONS: dict[str, Op] = {}


def op(summary: str, **hints):
    def register(fn):
        OPERATIONS[fn.__name__] = Op(fn.__name__, fn, summary, **hints)
        return fn
    return register


def call(name: str, /, *args, **kwargs):
    """Run an operation. dict results come back as plain JSON types (tuples become lists), Text unchanged."""
    value = OPERATIONS[name].fn(*args, **kwargs)
    return value if isinstance(value, Text) else json.loads(json.dumps(value))


def _one_line(value) -> str:
    return ' '.join(str(value).split())


def _load_failed(result: dict) -> str | None:
    problems = []
    if result.get('failed_count'):
        problems.append(f"{result['failed_count']} map lines failed to load")
    for key in ('camera', 'clearing_zones'):
        part = result.get(key)
        if isinstance(part, dict) and part.get('error'):
            problems.append(f"{key}: {_one_line(part['error'])}")
    return '; '.join(problems) or None


def _save_failed(result: dict) -> str | None:
    if result.get('written') is False:
        return f"the game did not write {result.get('path') or 'the map file'}"
    return None


def _unload_failed(result: dict) -> str | None:
    if result.get('unloaded'):
        return None
    if result.get('error'):
        return f"agent not unloaded: {_one_line(result['error'])}"
    return 'agent is still loaded' if result.get('loaded') else 'agent was not loaded: nothing to unload'


def _collision_unverified(result: dict) -> str | None:
    if result.get('verify_error'):
        return _one_line(result['verify_error'])
    if result.get('verified') is False:
        listed = len(result.get('mismatched') or [])
        # the reply lists at most MISMATCH_LISTED of them: a full list is a lower bound, not the count
        wrong = result.get('mismatched_count', f"at least {listed}" if listed >= MISMATCH_LISTED else listed)
        return (f"collision change applied but not verified: {wrong} objects in another mode, "
                f"{result.get('missing_after', 0)} missing")
    return None


def _catalog():
    return load_catalog()


def _saved_folder(name: str) -> str:
    """The folder of a saved map, or FileNotFoundError naming where it was looked for."""
    folder = sanitize_name(name)
    path = maps_dir() / folder / f"{folder}.json"
    if not path.exists():
        raise FileNotFoundError(f"no local map {folder!r} in {maps_dir()}")
    return folder


def _require(name: str):
    return load_map(_saved_folder(name))


def _clip(lines: list[str], max_lines: int) -> list[str]:
    limit = max(1, min(int(max_lines), OUTPUT_LINES_MAX))
    if len(lines) <= limit:
        return lines
    return lines[:limit] + [f"... {len(lines) - limit} more lines (max_lines up to {OUTPUT_LINES_MAX})"]


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _capture_target(out: str) -> tuple[Path, bool]:
    """(absolute path, is a directory) for capture_view's out."""
    path = Path(os.path.abspath(os.path.expanduser(out)))
    if out.endswith('/') or path.is_dir():
        return path, True
    if path.suffix.lower() != '.png':
        raise ValueError(f"out must be a directory or a path ending in .png, got {out!r}")
    return path, False


def _capture_folder(folder: Path) -> None:
    """Create capture_view's target folder and check it takes files, before anything is rendered."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise type(exc)(f"out: cannot use {folder} as the target folder ({exc.strerror or exc}); "
                        "nothing was captured") from exc
    if not os.access(folder, os.W_OK | os.X_OK):
        raise PermissionError(f"out: no permission to write into {folder}; nothing was captured")


@op('Maps folder, game / agent state and catalogue size', live='no')
def status() -> dict:
    """Where maps are written, whether Uncrashed is running, and how many assets are known."""
    catalog = _catalog()
    workshop = workshop_dir()
    workshop_count = len(list(workshop.iterdir())) if workshop else 0
    return {
        "game_running": game_running(),
        "steam_id": steam_id(),
        "maps_dir": str(maps_dir()),
        "workshop_items": workshop_count,
        "assets": len(catalog.assets),
        "flat_base": "BaseBoard",
        "reload": RELOAD,
        "live_agent": agent.loaded() if game_running() else False,
        "units": "metres, yaw degrees, Z up",
    }


@op('Race-gate row names for the gate command', live='no')
def list_gates() -> dict:
    """Race-gate row names for the gate command. DefaultOneW is a normal gate; STARTGATE is a start arch."""
    return {"gates": _catalog().library["gates"]}


@op('Find catalogue rows by name, material, tag or id', positional=('query',), live='no',
    examples=('./uc search_assets container --category Big_assets --limit 5',
              './uc search_assets --collision noncolliding --category Decals'))
def search_assets(
    query: str = "",
    category: str = "",
    tag: str = "",
    limit: int = 20,
    include_hidden: bool = True,
    collision: str = "",
) -> dict:
    """Find editor assets by name, material, tag or id.

    Categories include Wall, Big structure, Foliage, Big_assets, Small_assets, Basic shape.
    Tags include building, house, tree, vehicle, ground, fence. Decals are omitted unless category is Decals.
    hidden=true means the picker hides the row; the JSON can still spawn it. size_m is the unscaled mesh.
    collision='noncolliding' keeps rows whose game table 'Collision?' flag is false (stock-reliable, no DLL
    needed): mostly neon signs; pass category='Decals' as well for decals. 'colliding' the opposite; empty = all.
    """
    catalog = _catalog()
    rows = catalog.search(
        query=query,
        category=category or None,
        tag=tag or None,
        limit=len(catalog.assets) if collision else max(1, min(limit, 50)),
        include_hidden=include_hidden,
        include_decals=category.lower() == "decals",
    )
    rows = scenery.collision_filter(rows, collision)[:max(1, min(limit, 50))]
    return {"count": len(rows), "assets": [row.public() for row in rows]}


@op('Full records (bounds, placement rule) for asset ids', positional=('ids',), variadic='ids', live='no',
    examples=('./uc get_assets NewRow_1 NewRow_788',))
def get_assets(ids: list[str]) -> dict:
    """Full records for asset ids, including bounds and the placement rule."""
    catalog = _catalog()
    found = []
    missing = []
    for row in ids:
        asset = catalog.assets.get(row)
        if asset is None:
            missing.append(row)
        else:
            found.append(asset.public(full=True))
    return {"assets": found, "missing": missing}


@op('Saved maps the editor can load, newest first', live='no')
def list_maps() -> dict:
    """Local maps the track editor can load, newest first."""
    return {"maps_dir": str(maps_dir()), "maps": list_user_maps()}


@op('Counts, bounds and weather of one saved map', positional=('name',), live='no')
def inspect_map(name: str) -> dict:
    """Instance counts, bounds and weather for one local map. Does not dump every transform."""
    doc = _require(name)
    return doc.summary(_catalog())


@op('Read the editor camera pose')
def camera_pose() -> dict:
    """Read the live editor camera. Metres and degrees. Requires Uncrashed to be open in the track editor."""
    from . import live
    return live.read_camera()


@op('Move the editor camera (native, no focus change)', positional=('x', 'y', 'z', 'yaw', 'pitch'),
    examples=('./uc move_camera 120 -40 35 200 -25', './uc move_camera 120 -40 35 200 --no-hold'))
def move_camera(x: float, y: float, z: float, yaw: float, pitch: float = -20, hold: bool = True) -> dict:
    """Move the live editor camera without clicking. Metres, degrees. Negative pitch looks down.

    Native game-thread transforms; no external memory writer or helper process.
    hold disables editor camera movement until release_camera or 120-second watchdog
    (previous enable state restored). No window activation, keys or mesh placement.
    """
    from . import live
    return live.move_camera(x, y, z, yaw, pitch, hold)


@op('Release a camera hold so mouse look works again')
def release_camera() -> dict:
    """Stop holding the editor look so mouse look works again. The last position stays."""
    from . import live
    live.release_camera()
    return {"held": False, **live.read_camera()}


@op('Render the scene to a PNG, optionally from an explicit pose', positional=('x', 'y', 'z', 'yaw', 'pitch'),
    examples=('./uc capture_view 120 -40 35 200 -25 --out shots/',
              './uc capture_view --width 1280 --height 720 --out overview.png'))
def capture_view(
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    yaw: float | None = None,
    pitch: float | None = None,
    backend: str = 'engine',
    foreground: bool = False,
    width: int = 800,
    height: int = 450,
    out: str = '',
) -> dict:
    """Fresh offscreen scene image. Default: no focus/desktop switch or editor-camera move.

    Pass x,y,z and optional yaw/pitch for independent render pose, not editor movement.
    Engine backend omits UI and never silently falls back. window backend captures
    current editor view; occluded/minimized pixels may be stale. Only explicit
    backend='window', foreground=True permits activation/desktop switching.
    At most 4,194,304 pixels. Inspect image yourself, not with another vision model.
    """
    from . import live
    target = _capture_target(out) if out else None
    if target is not None:
        _capture_folder(target[0] if target[1] else target[0].parent)
    path = live.capture_view(x, y, z, yaw, pitch, backend=backend, foreground=foreground, width=width, height=height)
    sidecar = path.with_suffix('.json')
    meta = json.loads(sidecar.read_text())
    if target is not None:
        dest = target[0] / path.name if target[1] else target[0]
        try:
            shutil.move(str(path), str(dest))
        except OSError as exc:  # the render is done: say where it is instead of losing it
            raise type(exc)(f"{exc}; the capture is at {path}") from exc
        meta['image'] = str(dest)
        try:
            managed.atomic_write(dest.with_suffix('.json'), json.dumps(meta))
        except OSError as exc:
            raise type(exc)(f"{exc}; the image is at {dest}, its metadata at {sidecar}") from exc
        if sidecar != dest.with_suffix('.json'):
            sidecar.unlink(missing_ok=True)
    return meta


@op('Replace the open editor scene with a saved map', positional=('name',), effect='write',
    caution='removes everything placed in the open editor before loading; save unsaved work first (save_live)',
    examples=('./uc load_live MyMap',), check=_load_failed)
def load_live(name: str, move_camera: bool = True) -> dict:
    """Show a saved map in the open editor without leaving it.

    Switches the base level if needed, removes what is placed, then places every object
    and gate of the map with the editor's own functions in one pass. Afterwards the
    editor's Save button (and save_live) write back to this map. move_camera puts the
    camera where the map's saved view is.
    """
    _saved_folder(name)  # a mistyped name fails here, before the lock and with the folder that was looked in
    return agent.load(name, move_camera)


@op("Save the open editor level with the game's own SaveMap", positional=('name',), effect='write',
    caution='overwrites the saved map (the open one unless a name is given)',
    examples=('./uc save_live', "./uc save_live MyMap_v2 --display_name 'My Map v2'"), check=_save_failed)
def save_live(name: str = "", display_name: str = "") -> dict:
    """Save the open editor level with the game's own SaveMap (no key presses).

    Everything placed live or by hand is written; deleted objects are gone. name is the
    map folder; empty overwrites the map that is open. display_name is the list title;
    empty keeps the current one. Writes the JSON and .info like the Save button.
    """
    return agent.save(name, display_name)


@op('Run one agent command (centimetres) and print its reply', aliases=('exec',), rest='command', effect='write',
    caution='no preview, no undo: clear, destroy, destroyat, destroybox, replace, base and setmap act at once',
    examples=('./uc exec mapinfo', './uc exec spawn NewRow_1 1500 -2000 0 0 90 0 1 1 1 group=yard',
              "./uc exec --max_lines 200 'cells 5 *'"))
def agent_exec(command: str, max_lines: int = OUTPUT_LINES, timeout: float = 30) -> Text:
    """Run one agent command in the open editor and return its reply as text.

    Centimetres and degrees. The full command list is in the server instructions; the common ones:
    spawn, gate, laps, gates, actors, get, transform, replace, destroy, destroyat, destroybox,
    mapinfo, save, floating. Example: "spawn NewRow_1 1500 -2000 0 0 90 0 1 1 1".
    The reply is cut to max_lines (default 40, at most 400).
    """
    reply = agent.execute(command, timeout)
    status = "ok" if reply["ok"] else "failed"
    text = "\n".join([f"{status} {reply['head']}"] + _clip(reply["lines"], max_lines))
    if reply["ok"]:
        return Text(text)
    return Text(text, False, f"the agent reported: {_one_line(agent._failure(reply))}")


@op('Run a file of agent commands in one game-thread pass', aliases=('batch',), stdin='commands', effect='write',
    caution='no preview, no undo; lines that succeed stay applied when others fail: never re-send the whole file',
    examples=('./uc batch plan.txt', './uc batch - --timeout 600'))
def agent_batch(commands: list[str], quiet: bool = True, max_lines: int = OUTPUT_LINES, timeout: float | None = None,
                *, _lines: list[int] | None = None) -> Text:
    """Run many agent commands in one game-thread pass (the fast way to build: thousands per call).

    Same command language as agent_exec (centimetres). quiet (default) reports only the summary and
    the failures; quiet=false also returns each command's reply, cut to max_lines.
    Empty entries and entries starting with # are not sent. The first reply line is the agent summary plus
    sent=N. Failures are listed as "#<index> [L<file line>] <error> << <command>" (index into the commands
    sent, from 0; L only for commands read from a file), at most 20 of them; all other commands are
    applied, so retry only the failed ones.
    """
    if _lines is not None and len(_lines) != len(commands):
        raise ValueError("_lines needs one source line number per command")
    sent: list[str] = []
    source: list[int] = []
    for i, entry in enumerate(commands):
        command = " ".join(entry.split())
        if not command or command.startswith("#"):
            continue  # the native batch skips these without counting them: its #<i> indexes what is left
        if len(command.encode()) > BATCH_LINE_MAX:
            where = f"line {_lines[i]}" if _lines is not None else f"entry {i}"
            raise ValueError(f"{where} is {len(command.encode())} bytes long; the agent reads at most "
                             f"{BATCH_LINE_MAX} per command")
        sent.append(command)
        source.append(_lines[i] if _lines is not None else i)
    if not sent:
        raise ValueError("no commands")
    try:
        results, summary = agent.batch(sent, timeout=timeout, quiet=quiet)
    except agent.AgentNotSent:
        raise  # nothing was written: nothing is applied
    except agent.AgentError as exc:
        # Sent: a timeout, a lost agent, or a job that answered `failed` (a fault inside one line). The lines
        # before it are applied either way, and an error line names one line, not how far the job got.
        text = str(exc) if exc.reply is None else f"batch job failed after {len(exc.reply['lines'])} reply lines: {exc}"
        unknown = type(exc)(f"{text}; {BATCH_UNKNOWN}")  # keeps the subclass (AgentTimeout) for the caller
        unknown.reply = exc.reply
        raise unknown from exc
    failed = [i for i, lines in sorted(results.items()) if any(l.startswith("error") for l in lines)]

    def describe(i: int) -> str:
        known = 0 <= i < len(sent)
        tag = f"#{i} L{source[i]}" if known and _lines is not None else f"#{i}"
        return f"{tag} {' '.join(results[i])} << {_shorten(sent[i], 120) if known else '?'}"

    out = [f"{summary or 'batch summary=missing'} sent={len(sent)}"]
    if failed:
        cap = max(1, min(int(max_lines), BATCH_FAILURES))
        out += ["failures:"] + [describe(i) for i in failed[:cap]]
        if len(failed) > cap:  # max_lines cannot lift this cap; the agent's own reply has every error line
            out.append(f"... {len(failed) - cap} more failed lines: run ./uc last now, before another agent "
                       "command replaces that reply, to list them all")
        out.append(BATCH_RETRY)
    if not quiet:
        out += _clip([f"#{i} {l}" for i, lines in sorted(results.items()) for l in lines], max_lines)
    counts = {key: int(value) for key, value in re.findall(r"\b(lines|ok|failed)=(\d+)", summary)}
    if not failed and counts.get("failed") == 0 and counts.get("lines") == len(sent):
        return Text("\n".join(out))
    if "lines" not in counts or "failed" not in counts:
        failure = f"batch sent {len(sent)} lines but the agent gave no summary; {BATCH_UNKNOWN}"
    elif counts["lines"] != len(sent):
        failure = f"batch ran {counts['lines']} of {len(sent)} lines sent; {BATCH_UNKNOWN}"
    else:
        shown = " ".join(f"#{i}" for i in failed[:10]) + (" ..." if len(failed) > 10 else "")
        failure = (f"batch applied {counts.get('ok', len(sent) - counts['failed'])}/{len(sent)} lines, "
                   f"{max(counts['failed'], len(failed))} failed{f' ({shown})' if shown else ''}; "
                   "do not re-run the whole file")
    return Text("\n".join(out), False, failure)


@op('Audit placed-object clashes and disconnected groups',
    examples=('./uc check_collisions', './uc check_collisions --region -100,-100,100,100 --top 30'))
def check_collisions(
    x0: float | None = None,
    y0: float | None = None,
    x1: float | None = None,
    y1: float | None = None,
    top: int = 15,
    include_minor: bool = False,
) -> dict:
    """Placed-object clashes and disconnected support groups. Read-only; engine collision.

    Reports objects buried in rock, vehicles clipping into walls or each other, containers and buildings
    inside each other, props deep inside other objects, vehicles sunk into the road, and anything not
    connected to the ground (balloons excepted). Touching parts, construction joints, plants and ground tiles
    are ignored. x0 y0 x1 y1 (metres) limit the scan to a region; default uses live placed-object extents,
    not a fixed +/-3 km box. This does NOT verify complete foundations or environment clearance:
    run check_foundations and check_environment too. No fixed z=0 ground assumption.

    Returns counts per kind and the worst `top` issues (include_minor adds things hovering less than a metre),
    each with its position and a camera pose for move_camera. Also writes report.json and report.html, a
    zoomable plan with every issue, to /tmp/uncrashed-collide/. A whole map of 27,000 objects takes about a
    minute; the editor pauses during each of the passes.
    """
    region = environment.validate_region(x0, y0, x1, y1)  # a usage error must not wait for the agent lock
    with agent.session(), tempfile.TemporaryDirectory(prefix='uncrashed-collide-', dir='/tmp',
                                                      ignore_cleanup_errors=True) as scan_dir:
        if region is None:
            bounds = environment.get_world_bounds('placed')
            if not bounds['available']:
                return {'objects_checked': 0, 'issues': {}, 'top': [], 'scope': 'no placed objects'}
            if bounds['truncated']:
                raise RuntimeError('loaded scene bounds are incomplete; pass an explicit scan region')
            lo, hi = bounds['aabb']['min'], bounds['aabb']['max']
            region = (lo[0]-5, lo[1]-5, hi[0]+5, hi[1]+5)
            grid = min(10, max(1, math.ceil(math.sqrt(bounds['actor_count']/400))))
        else:
            grid = 2
        commands, paths = collide.scan_commands(Path(scan_dir), agent._windows_path, region_m=region, grid=grid)
        for command in commands:
            agent.run(command, timeout=120)
        if not all(p.is_file() for p in paths):
            raise RuntimeError('collision scan incomplete: one or more result files are missing')
        report = collide.analyse(paths, collide.Rows.from_catalog(_catalog()))
        try:
            info = agent.map_info()
            title = info.get("display_name") or info.get("title") or info.get("folder") or "Map"
        except Exception:
            title = "Map"
        managed.atomic_write(COLLIDE_DIR / "report.json", json.dumps(report))
        managed.atomic_write(COLLIDE_DIR / "report.html", collide.render_html(report, title=f"{title} collision check"))
        out = collide.brief(report, top=max(1, min(int(top), 60)), include_minor=include_minor)
        out["report_html"] = str(COLLIDE_DIR / "report.html")
        out['region_m'] = region
        out['scope'] = 'Placed-object clashes and support connectivity only. Use check_foundations / check_environment for footprint and environment checks.'
        return out


@op('World AABB, mesh OBB and transform of one live actor', positional=('actor',))
def get_object_bounds(actor: str) -> dict:
    """Live world-space AABB, optional oriented mesh bounds and transform. Metres.

    Actor is a live hex address from agent_exec actors. AABB encloses owned actor
    components; child actors are not included. OBB needs a static-mesh root.
    Bounds enclose empty space too; they are not exact occupied geometry.
    """
    return environment.get_object_bounds(actor)


@op('Extents of placed objects, terrain or tagged playable area', positional=('scope',))
def get_world_bounds(scope: str = 'placed') -> dict:
    """Bounds of loaded placed objects, landscape, or explicit playable bounds. Metres.

    scope: placed | terrain | playable. Terrain extent is NOT a playable boundary.
    playable returns unavailable unless an actor is explicitly tagged PlayableBounds.
    Never guesses from arbitrary blocking volumes or assumes a fixed map size.
    """
    return environment.get_world_bounds(scope)


@op('Audit unsupported footprints and buried base planes',
    examples=('./uc check_foundations --region -100,-100,100,100', './uc check_foundations --actors 1A2B3C4D,5E6F7A8B'))
def check_foundations(
    actors: list[str] | None = None,
    x0: float | None = None, y0: float | None = None,
    x1: float | None = None, y1: float | None = None,
    spacing: float = .75, tolerance: float = .3, max_drop: float = 50,
    include_structural: bool = False, top: int = 15,
) -> dict:
    """Find partially unsupported footprints / buried base planes. Read-only; metres.

    Defaults to buildings, containers, vehicles and big props. Pass live actor hex
    addresses to target objects, or all four XY region bounds. include_structural
    also tests walls / beams / props, where intentional cantilevers create warnings.
    Samples a bounds base plane (oriented for upright meshes), NOT a structural load model. Reports
    support coverage, unsupported corners, maximum known gap and missing rays.
    Ignores base foliage and catalogued placed plants as support. No z=0 assumption.
    spacing grid capped at 32 intervals per axis; effective spacing is reported.
    Full results saved under /tmp/uncrashed-audit/. Does not move anything.
    """
    region = environment.validate_region(x0, y0, x1, y1)
    return environment.audit(_catalog(), 'foundation', actors=actors, region=region,
                            spacing=spacing, tolerance=tolerance, max_drop=max_drop,
                            include_canopies=False, include_structural=include_structural, top=top)


@op('Audit terrain penetration, foliage hits and canopies',
    examples=('./uc check_environment --region -100,-100,100,100 --no-include_canopies',))
def check_environment(
    actors: list[str] | None = None,
    x0: float | None = None, y0: float | None = None,
    x1: float | None = None, y1: float | None = None,
    spacing: float = .75, tolerance: float = .1,
    include_canopies: bool = True, include_structural: bool = False, top: int = 15,
) -> dict:
    """Landscape penetration, base/placed-foliage collision candidates and canopy warnings.

    Read-only; metres. Pass live actor addresses or four XY region bounds. By
    default skips plants, rocks, ground tiles and structural kit parts. Terrain
    samples use subject collision undersides when available, bounds otherwise.
    Foliage physics tests subject OBB vs base/placed plant collision; not exact mesh-vs-mesh.
    Optional canopy checks use visual per-instance / placed-plant AABBs, so leaves
    without collision are covered conservatively. Only loaded plants taller than 2 m.
    Full JSON report under /tmp/uncrashed-audit/; caps and unknowns are reported.
    """
    region = environment.validate_region(x0, y0, x1, y1)
    return environment.audit(_catalog(), 'environment', actors=actors, region=region,
                            spacing=spacing, tolerance=tolerance,
                            include_canopies=include_canopies, include_structural=include_structural, top=top)


@op('Bare-ground height at x,y points (metres)', positional=('points',), variadic='points',
    examples=('./uc sample_terrain 10,20 12.5,20 -15,20.5', "./uc sample_terrain --points '[[10,20],[12.5,20]]' --z_from 500"))
def sample_terrain(points: list[list[float]], z_from: float = 1000, z_to: float = -2000) -> dict:
    """Sample bare landscape at up to 256 [x, y] positions (metres).

    Ignores trees and placed structures, unlike trace/drop. Missing landscape returns z=null.
    Use a footprint grid to choose foundation elevations; does not move any objects.
    Requires a landscape base, not BaseBoard. Samples retain input order.
    """
    if not 1 <= len(points) <= 256:
        raise ValueError("provide 1 to 256 [x, y] points")
    if not all(len(p) == 2 and all(math.isfinite(v) for v in p) for p in points):
        raise ValueError("each point must contain two finite coordinates")
    if not math.isfinite(z_from) or not math.isfinite(z_to) or z_from <= z_to:
        raise ValueError("z_from must be finite and greater than z_to")
    commands = [f"terrain {x * 100:.3f} {y * 100:.3f} {z_from * 100:.3f} {z_to * 100:.3f}" for x, y in points]
    replies, summary = agent.batch(commands)
    samples = []
    for i, (x, y) in enumerate(points):
        lines = replies.get(i, [])
        hit = next((line for line in lines if line.startswith("terrain ")), None)
        if hit is None or any(line.startswith("error") for line in lines):
            raise RuntimeError(f"terrain sample {i} failed: {'; '.join(lines)}")
        fields = dict(token.split("=", 1) for token in hit.split() if "=" in token)
        samples.append({"x": x, "y": y, "z": float(fields["z"]) / 100 if "z" in fields else None})
    return {"samples": samples}


@op('Tail of the in-game agent log', aliases=('log',), positional=('lines',))
def agent_log(lines: int = 80) -> Text:
    """Tail of the in-game agent's log (load, unload, timing, caught faults)."""
    text = agent.log_tail(lines)
    if not text.strip():  # an empty reply would look like a command that printed nothing because it failed
        text = f"(agent log is {'empty' if agent.LOG.exists() else 'missing'}: {agent.LOG})"
    return Text(text)


@op('Restore masks / holds and unload the agent DLL', aliases=('unload',), effect='write',
    check=_unload_failed)
def agent_unload() -> dict:
    """Restore managed masks/holds and unload DLL; next call injects rebuilt DLL.

    Restoration conflicts refuse unload to retain original foliage snapshots.
    """
    return agent.unload()


@op('Terrain, foliage and placed-object inventory of an XYZ box', positional=('bounds_min', 'bounds_max'),
    examples=('./uc inspect_site -50,-50,0 50,50,40 --spacing 10',))
def inspect_site(bounds_min: list[float], bounds_max: list[float], spacing: float = 5,
                 min_foliage_height: float = 2, top: int = 30) -> dict:
    """Read-only pre-placement terrain/foliage/placed-object inventory in XYZ box, metres.

    Bare-ground grid includes center axes, capped at 225 points; effective spacing
    reported. Foliage visual AABBs include crowns, even without leaf collision.
    Set min_foliage_height=0 for groundcover too. Loaded geometry only, not complete
    base-rock obstacle inventory. Full JSON on disk; top limits returned objects.
    """
    return managed.inspect_site(bounds_min,bounds_max,spacing,min_foliage_height,top)


@op('List, add or remove persistent base-foliage clearings', positional=('action', 'name'), effect='write',
    examples=('./uc clearing_zones', './uc clearing_zones upsert yard --center 120,-40,5 --size 30,20,25 --yaw 15 --apply',
              './uc clearing_zones remove yard --apply'))
def clearing_zones(action: str = 'list', name: str = '', center: list[float] | None = None,
                   size: list[float] | None = None, yaw: float = 0, preview: bool = True,
                   allow_partial: bool = False) -> dict:
    """Persistent, reversible base-foliage exclusions: list | upsert | remove.

    Metres; center/size are XYZ triplets, yaw rotates clearing prism. Preview by
    default; preview=False writes portable map sidecars and applies live. Whole
    instances hidden by visual bounds, including collision; no RemoveInstance,
    no terrain/placed-object changes. Overlapping zones form union. Remove restores
    originals not covered by other zones. Saved map required. Resident DLL reapplies
    on map/base reload; sidecars must accompany map. Stock game alone ignores them.
    GPU-only landscape grass may lack CPU instance transforms. Unavailable sources
    reported; allow_partial=True explicitly permits supported-source-only clearing.
    """
    return managed.clearing_zones(action,name,center,size,yaw,preview,allow_partial)


@op('Editor group tags and object counts in the open scene', positional=('prefix',))
def list_groups(prefix: str = '', limit: int = 50) -> dict:
    """Live saved editor group tags/counts, optionally name prefix. Pivot bounds, not mesh bounds."""
    return groups.list_groups(prefix,limit)


@op('Transform, clone, delete or regroup a group (journaled)', positional=('group',), effect='write',
    caution='with --apply the edit happens at once; delete / clone undo regenerates catalogue assets only',
    examples=('./uc edit_group houses.a --delta 2,0,0 --rotation 0,90,0',
              './uc edit_group houses --action clone --target_group houses_b --delta 0,40,0 --prefix --apply',
              './uc edit_group --actors 1A2B3C4D,5E6F7A8B --action assign --target_group footings --apply'))
def edit_group(group: str = '', action: str = 'transform', delta: list[float] | None = None,
               rotation: list[float] | None = None, scale: float = 1, pivot: list[float] | None = None,
               target_group: str = '', prefix: bool = False, actors: list[str] | None = None,
               preview: bool = True) -> dict:
    """Preview/apply reversible transform | clone | delete | assign group edit.

    Catalogue assets only; gates/base actors excluded. Explicit group or actor
    handles required; never defaults to whole map. prefix
    includes dot/underscore-separated subgroups (clone preserves suffixes). delta/pivot XYZ metres; rotation delta is
    [pitch,yaw,roll] degrees, quaternion-composed. Uniform scale only (avoids shear).
    Default pivot is center of selected pivots. clone needs fresh target_group;
    assign regroups selected objects into target_group. Preview by default.
    preview=False journals before-state on disk, validates native preconditions,
    then edits. Up to 5000 objects. Undo via returned transaction id. Transforms
    preserve actors; clone/delete undo regenerates catalogue assets, not arbitrary
    Blueprint/material/animation state. Native faults may need journal recovery.
    """
    return groups.edit_group(group,action,delta,rotation,scale,pivot,target_group,prefix,actors,preview)


@op('Undo a journaled edit by transaction id', positional=('transaction',), effect='write',
    caution='a journal left prepared by an interrupted edit must be previewed first: that preview recovers it, and '
            'its undo also reverts later changes to these groups (reply fields recovered, deletes_not_in_before)')
def undo_edit(transaction: str, preview: bool = True) -> dict:
    """Preview/apply inverse of journaled edit. Explicit transaction id; saved map scoped.

    Refuses if affected groups changed since edit. Matches row/group/transform
    across handle changes on reload. Successful undo creates inverse transaction
    (undo that to redo). preview=False applies. Does not restore arbitrary runtime
    Blueprint/material state for deleted/respawned catalogue assets.
    """
    return groups.undo_edit(transaction,preview)


@op('Recent transaction ids and statuses of the open map')
def edit_history(limit: int = 20) -> dict:
    """Recent current-map MCP transaction ids/statuses, including recovery-required edits."""
    return groups.edit_history(limit)


@op('Inspect or save the group roles of a built-up assembly', positional=('name',), effect='write',
    examples=('./uc define_assembly', './uc define_assembly tower --roles base=foundation,frame=structure,deck=cantilever --apply'))
def define_assembly(name: str = '', roles: dict[str,str] | None = None, preview: bool = True) -> dict:
    """Inspect/save assembly intent: group -> foundation|structure|cantilever|decoration.

    Omit roles to inspect named assembly, or omit name too to list definitions.

    At least one explicit foundation group. Use edit_group assign to split footings
    from beams when necessary. Preview by default; preview=False writes map sidecar,
    not geometry. Group tags/roles survive save/reload without actor-handle IDs.
    """
    return assemblies.define_assembly(name,roles,preview)


@op('Support-connectivity check of a defined assembly', positional=('name',))
def check_assembly(name: str, tolerance: float = .05, foundation_tolerance: float = .3,
                   min_anchor_coverage: float = .5, top: int = 20) -> dict:
    """Read-only intent-aware support connectivity, NOT structural certification.

    Explicit foundation roots need measured landscape/base contact, not z=0.
    Reciprocal OBB-vs-collision hits build graph. Cantilevers need anchor path,
    not full footprint coverage; plants/decorations cannot support frame. Reports
    disconnected members, unknown anchors and conservative-contact limitations.
    Metres; loaded geometry only. Still run check_environment for terrain/trees.
    """
    return assemblies.check_assembly(_catalog(),name,tolerance,foundation_tolerance,min_anchor_coverage,top)


@op('Make placed objects flythrough / non-colliding, or restore them', positional=('mode',), effect='write',
    examples=('./uc set_collision flythrough --group fence_north',
              './uc set_collision off --box_min -10,-10,0 --box_max 10,10,30 --no-persist --apply',
              './uc set_collision on --row NewRow_412 --apply'), check=_collision_unverified)
def set_collision(mode: str = 'flythrough', actors: list[str] | None = None, group: str = '', row: str = '',
                  box_min: list[float] | None = None, box_max: list[float] | None = None,
                  channels: list[str] | None = None, keep_body: bool = False, force: bool = False,
                  preview: bool = True, persist: bool = True, top: int = 30) -> dict:
    """Make placed scenery non-colliding (live, resident DLL): on | nophysics | flythrough | off,
    plus live-only physics_off / physics_on (toggle simulation only, leave flythrough/off alone; persist=False).

    Exactly one selector: actors (hex handles, max 8192), group (editor group tag), row (catalogue id, every
    instance) or box_min/box_max (XYZ metres, object pivot inside); group/row/box have no size cap.
    Never base landscape/foliage. Race gates are checkpoints: skipped unless force=True. flythrough
    ignores channels (default pawn, physicsbody, vehicle; names, engineN, gameN or 0..31) and drops
    rigid-body collision (keep_body=True keeps it); query collision stays, so traces/audits/editor
    picking still see it. off disables all collision (traces miss it). nophysics only stops simulation.
    on restores the exact recorded originals. Preview by default. persist=True (saved map required)
    writes portable rules to UncrashedMCP.json + UncrashedMCP.collision and the resident DLL re-applies
    them after every map reload while the editor is open; the STOCK GAME WITHOUT THE DLL IGNORES THEM
    (use noncolliding_alternatives / search_assets(collision='noncolliding') for stock-reliable scenery).
    Returns a transaction id for undo_collision. Drone channel unverified: run agent 'collision channels'
    while flying.
    """
    return scenery.set_collision(mode, actors, group, row, box_min, box_max, channels, keep_body,
                                 force, preview, persist, top)


@op('Collision facts of the DLL or of selected objects',
    examples=('./uc collision_state', './uc collision_state --group fence_north --no-components'))
def collision_state(actors: list[str] | None = None, group: str = '', row: str = '',
                    box_min: list[float] | None = None, box_max: list[float] | None = None,
                    channels: list[str] | None = None, top: int = 30, components: bool = True) -> dict:
    """Read-only collision facts. No selector: DLL status (records, rules, rules matching nothing because
    objects moved, gone/replaced counters), persisted rules. With a selector: per object mode/owner,
    keep_body, actor collision, simulating, and per component CollisionEnabled, object channel, profile and
    responses (pawn, physicsbody, vehicle, visibility, camera, ...); 'blocks' lists queried channels it still blocks.
    """
    return scenery.collision_state(actors, group, row, box_min, box_max, channels, top, components)


@op('Undo a set_collision transaction', positional=('transaction',), effect='write')
def undo_collision(transaction: str, preview: bool = True) -> dict:
    """Preview/apply the inverse of a set_collision transaction (edit_history lists them).

    Restores the previous persisted rules (refuses if rules changed since) and replays the previous
    per-object modes (including keep_body) for the journaled objects that are still loaded; reports
    how many are missing. A live-only transaction whose objects are all gone is refused (nothing to undo).
    """
    return scenery.undo_collision(transaction, preview)


@op('Stock non-colliding catalogue rows resembling a row', positional=('row',), live='no')
def noncolliding_alternatives(row: str, limit: int = 5) -> dict:
    """Stock-reliable non-colliding catalogue rows resembling `row` (Collision? flag false in the game's
    table: works without the DLL and for every player). Often none exist: the catalogue's non-colliding
    rows are decals and neon signs; then use set_collision.
    """
    return scenery.lookalikes(_catalog(), row, limit)


@op('Performance budget check of a saved map or the open scene', positional=('name',), live='mixed',
    examples=('./uc check_performance MyMap --budgets placed_objects=6000,dynamic_lights.warn=8',
              """./uc check_performance --measure --poses '[{"x":0,"y":0,"z":30,"yaw":90,"pitch":-20}]'"""))
def check_performance(
    name: str = '',
    x0: float | None = None, y0: float | None = None,
    x1: float | None = None, y1: float | None = None,
    budgets: dict | None = None,
    measure: bool = False,
    poses: list[dict] | None = None,
    duration: float = 1.5,
    cell: float = 50,
    top: int = 10,
) -> dict:
    """Performance budget check for big maps. Read-only; metres.

    name = a saved map folder (or a .json path) scans the file offline: counts per row / category / kind, Blueprint
    objects, LOD0 triangle estimate, densest grid cells (cell m) with a move_camera pose each, top rows by count and
    by triangles, and how many objects consolidate_scenery could merge. Without name it scans the open editor live:
    every loaded actor, its mesh / instanced components, visible lights (movable, shadow casting), shadow casters,
    never-culled primitives, physics, collision, whether any placed object owns an instanced component, and the
    renderer's dynamic-instancing cvar. Live budgets grade placed objects only; *_base figures (landscape, foliage,
    base map) are informational. x0 y0 x1 y1 limit the scan to a region.
    budgets overrides heuristic defaults (NOT measured Uncrashed limits; listed in the reply): a number sets warn and
    high = 2.5 x warn, e.g. {"placed_objects": 6000}; a dict sets either or both, e.g. {"dynamic_lights": {"warn": 8,
    "high": 20}} (warn alone raises high to at least 2.5 x warn). cell_objects / cell_triangles are per 50 m cell and
    scale with cell area unless overridden.
    measure=True (live only) also samples frame time for `duration` seconds, at each of up to 8 poses
    [{x,y,z,yaw,pitch}] (camera released afterwards), and reports avg / p95 ms with VSync / MaxFPS / background notes.
    Returns warnings by severity with concrete advice; full JSON under /tmp/uncrashed-perf/.
    """
    region = environment.validate_region(x0, y0, x1, y1)
    report = performance.check_performance(_catalog(), name=name, region=region, budget=budgets, measure=measure,
                                           poses=poses, duration=duration, cell=cell, top=max(1, min(int(top), 50)))
    return report


@op('Merge runs of triplanar box tiles into fewer scaled copies', positional=('name',), live='mixed',
    effect='write', caution='--apply deletes the originals and spawns replacements (journaled; undo_edit, newest first)',
    examples=('./uc consolidate_scenery MyMap', './uc consolidate_scenery --rows NewRow_12,NewRow_40 --apply'))
def consolidate_scenery(
    name: str = '',
    x0: float | None = None, y0: float | None = None,
    x1: float | None = None, y1: float | None = None,
    rows: list[str] | None = None,
    preview: bool = True,
    tolerance: float = 0.02,
    max_size: float = 64,
    limit: int = 20,
) -> dict:
    """Merge edge-to-edge runs of the same plain-box triplanar row into fewer scaled copies. Preview by default.

    The save format has no instancing, so fewer bigger objects are the only persistent saving. Only rows that the
    catalogue marks triplanar (texture is world-projected, so scaling keeps its density) AND whose mesh hull is a
    plain box (ground tiles, Concrete Wall01 / Floor01 / Column, roof, grid net) are merged; slopes, corners, holed or
    windowed pieces and every non-triplanar row are never merged. Tiles must share row, group, rotation, scale and
    plane, and sit on one lattice within tolerance (m); merged copies are at most max_size m per side.
    name = saved map (preview only). Without name it plans on the open editor; preview=False deletes the originals
    and spawns the replacements through the journaled group edit: the largest-scale merge goes alone first as a
    canary, then one transaction per <=4000 commands. Every replacement's pivot/scale is checked against the request;
    a mismatch undoes that transaction and stops with an error. Returns transaction ids: undo with
    undo_edit(transaction, preview=False), newest first. Saved map required for apply (journal sidecar). Also lists
    high-count rows that cannot be merged, for manual thinning. No auto-save.
    """
    region = environment.validate_region(x0, y0, x1, y1)
    return consolidate.consolidate_scenery(_catalog(), name=name, region=region, rows=rows, preview=preview,
                                           tolerance=tolerance, max_size=max_size, limit=limit)


@op('Sweep a craft along the race line or waypoints for clearance', positional=('source',),
    examples=('./uc check_route_clearance map:MyMap --profile cinewhoop',
              "./uc check_route_clearance points --points '[[0,0,5],[40,10,8],[80,0,5]]' --dims_m 0.3,0.3,0.12"))
def check_route_clearance(
    source: str = "live",
    points: list[list[float]] | None = None,
    profile: str = "5inch",
    margin_m: float | None = None,
    spacing_m: float = 1.0,
    dims_m: list[float] | None = None,
    kind: str | None = None,
    channel: str = "Pawn",
    probe_m: float = 10.0,
    directions: int = 16,
    closed: bool | None = None,
    laps: int | None = None,
    align_gates: bool = True,
    ignore_start_m: float = 0.0,
    max_issues: int = 20,
) -> dict:
    """Route clearance for a craft along the race line or waypoints. Read-only; metres.

    source: 'live' (race gates in the open editor), 'map:<name>' (saved Race: start + checkpoints, laps),
    or 'points' with points=[[x,y,z],...] (ground mode accepts [x,y]: top-most non-foliage, non-gate surface).
    A centripetal Catmull-Rom spline through gate openings (trigger-volume centres; lead-in/out along
    each gate normal unless align_gates=False), sampled every spacing_m, closed when laps>1 (or closed=True).
    profile: tiny_whoop, cinewhoop, 5inch, 7inch, fixed_wing (air) or car, truck (ground; path draped on the
    surface at ride height); dims_m=[width,length,height] and margin_m override. Dimensions are echoed.
    The craft box is swept between samples against real collision; probes measure free space in
    `directions` horizontal directions + up/down up to probe_m. Gate openings are probed from each gate's exact centre.
    channel: the collision channel the craft is assumed to collide as: Pawn, PhysicsBody, Vehicle, ...,
    game1..game18, engine1..engine6 or an index '0'..'31'. Components that only overlap / ignore it (gate checkpoint
    triggers, flythrough objects) are not obstacles. The physical-craft model also skips QueryOnly components,
    even with channel='Visibility': it cannot validate visual openings in a QueryOnly/flythrough scene. Collision-
    disabled objects are invisible. Reports collisions, gate frame hits, gate openings smaller than craft + 2*margin,
    tight spots, low ceiling / ground, and for ground
    mode corridor width / narrow corridor / ground contact, each with route distance, obstacle row and a move_camera
    pose. Full JSON + HTML plan in /tmp/uncrashed-clearance/.
    """
    return clearance.check_route_clearance(
        source=source, points=points, profile=profile, margin_m=margin_m, spacing_m=spacing_m, dims_m=dims_m,
        kind=kind, channel=channel, probe_m=probe_m, directions=directions, closed=closed, laps=laps,
        align_gates=align_gates, ignore_start_m=ignore_start_m, max_issues=max_issues, catalog=_catalog())
