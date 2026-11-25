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

from . import (agent, assemblies, clearance, collide, consolidate, environment, groups, managed, opening, performance,
               reference, scenery, sheet, snapshots)
from .catalog import hay_of, load_catalog, word_match
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
Asset, SpawnTrackAsset, SaveMap).

BASICS
Help: `./uc help` (commands: ! acts at once, ~ previews unless --apply), `./uc help <command>`,
`./uc help verbs` (agent commands), `./uc help output` (reply format). Commands use metres and degrees
(Z up, yaw 0 = +X, 90 = +Y); agent commands (exec = agent_exec, batch = agent_batch) use cm.
Replies are short: ask more (--max_lines, --top, --limit). --json prints JSON, --pick a,b.c keeps fields.
Exit codes: 0 ok; 1 raised (a live change may be partly applied); 2 usage error (nothing ran); 3 ran but
reported failures. Every failure prints one stderr line `error: ...`; pipes need `set -o pipefail`.
No game needed: status, list_gates, list_maps, inspect_map, search_assets, get_assets,
noncolliding_alternatives; check_performance / consolidate_scenery with a saved map name.

BUILD
search_assets / get_assets (rows) -> write commands -> batch -> capture_view -> fix -> save_live.
`./uc exec mapinfo` first: only BaseBoard is a flat grid at z=0. On landscape bases sample_terrain measures
bare ground through trees and structures (not platform or building roofs): sample the whole footprint
before choosing a foundation elevation. inspect_site lists bare ground, known flora and placed
objects; GPU-only landscape grass is reported unavailable.
spawn takes the pivot; get_assets origin_cm / extent_cm locate a mesh's bottom.
Batch: generate commands with a script kept in the project (not /tmp), then `./uc batch plan.txt` or
`python3 plan.py | ./uc batch -`: one game-thread pass. Failed lines print as
`#<index> L<line> <error> << <command>`; all others ARE applied: resend only the failed ones.
drop: what stands on something is spawned a little high, then `drop last` on
the line directly after its spawn (same batch only) rests it on what is under it (real
shapes). Stack bottom-up. Ground tiles sit 9 to 13 cm above z=0, so an object spawned at z=0 on a
tile is sunk in unless dropped.
Quote an exec command containing * ! # ; | < > & as one argument.

LOOK
capture_view renders a fresh engine image to a PNG (--out file.png or dir/) without
focus, UI or editor camera movement; x y z [yaw pitch] set an independent pose. Only `--backend window
--foreground` may focus the game.
move_camera / camera_pose / release_camera drive the editor camera; holds end after a 120-second
watchdog. With several agents, use explicit poses; let one agent own the editor camera. capture_sheet
tiles up to 60 poses in one labelled PNG.
get_object_bounds: world AABB / mesh OBB. get_world_bounds: placed or terrain extents; playable bounds
need an explicit tag, never inferred from landscape size.

NO PREVIEW, NO UNDO
Agent verbs clear, destroy, destroyat, destroybox, replace, base (reopens the editor, clears the scene) and
setmap (retargets Save). load_live removes everything placed first; save_live overwrites the map. No edit
auto-saves.

SLOW COMMANDS
check_collisions (a minute per 27,000 objects), load_live, big batches, `consolidate_scenery --apply`,
check_route_clearance, check_opening and reference_load outlast shell timeouts: raise it or run in the background.
NEVER re-send a command that timed out, failed or was interrupted: it was sent and may still run or be
partly applied (only an error containing `nothing was sent` is safe to retry). First check `./uc last`
(last finished reply), `./uc exec mapinfo`, `./uc edit_history`, `./uc log` (agent_log). stderr `waiting
for agent lock ...` or `agent is finishing an earlier job` means wait: do not re-send. After a batch error
without a summary line later commands may time out although they run, until agent_unload (`./uc unload`).

AUDITS (read-only)
Before saving a big build: check_collisions (clashes, disconnected groups), check_foundations
(unsupported footprint portions), check_environment (landscape / base-foliage conflicts). A clean check_collisions is not full support or environment validation.
define_assembly gives groups foundation/structure/cantilever/decoration roles; check_assembly measures
anchors and reciprocal contact paths, not load capacity. Overhangs need a path, not full footprint
coverage; connectivity is not certification.
check_route_clearance sweeps the craft along a smooth line through the race gates (live, map:<name>)
or waypoints (points). Profiles 5inch, tiny_whoop, cinewhoop, 7inch, fixed_wing, car/truck (roads). Reports
collisions, small gate openings (probed from the gate centre), tight spots, low ceilings, road width, with
move_camera poses. Overlap-only
triggers and flythrough objects are not obstacles. The channel defaults to Pawn, unverified for the drone
(--channel game3). Spline geometry, not a flight test.
check_opening x,y,z ...: first obstruction on the path (live-collision, override-active or
unverified; no hit beside unverified objects is not clearance).
check_performance: saved map name offline, or live (--measure samples frame time, also at
--poses). Budgets are heuristic; live grading counts placed objects only (*_base: info).

EDITS (~ commands: preview first, then --apply)
list_groups / edit_group: transform, clone, delete, regroup. undo_edit / edit_history: journaled inverses;
clone/delete undo does not preserve arbitrary Blueprint/material runtime state. undo_edit of an interrupted
edit recovers its journal in the preview; that undo also reverts later changes to its groups.
Read-only scene_snapshot before, scene_diff after: change counts, gate verdicts.
clearing_zones persists portable intent and masks/restores supported base flora while the DLL is resident;
the stock game alone ignores sidecars. Partial source coverage needs --allow_partial. Preserve both
UncrashedMCP sidecars when copying maps.
consolidate_scenery merges plain-box triplanar tile runs into fewer scaled copies. NAME only previews a saved map file. To apply: save your work, load_live the map,
run without NAME (preview, then --apply). The largest merge is a canary; a mismatch is undone. Undo with undo_edit per returned transaction, newest first.
Non-triplanar rows are never merged; thin them by hand.

REFERENCE (textured OBJ: view only, unsaved, no collision)
reference_load OBJ --at x,y,z | --map_origin lat,lon [--center e,s --radius_m R]; reference_status,
reference_visibility [--no-hidden], reference_clear. load_live clears it; captures hide it unless --reference.

SCAN (textured OBJ, no game; `./uc help scan`)
scan_survey OBJ, scan_place, then scan_height, scan_profile, scan_at, scan_objects, scan_ground,
scan_building, scan_facade, scan_free, scan_grid, scan_view / scan_sheet (RAM), scan_status, scan_sources.
Rows: asset_profile, scan_match, scan_plan (a file), scan_compare. capture_topdown + scan_view --compare.

COLLISION
set_collision (previews unless --apply) sets placed objects picked by --actors/--group/--row or
--box_min/--box_max to flythrough (drone/vehicle channels ignored; traces and audits still see them), off, nophysics or on (exact restore); physics_off / physics_on are
live-only simulation toggles. Gates only with --force. persist (default; --no-persist = live only) writes
UncrashedMCP.collision: the resident DLL re-applies it after reloads while the editor is open, the stock
game without the DLL ignores it. For maps others fly prefer non-colliding catalogue rows
(`search_assets --collision noncolliding`, noncolliding_alternatives). collision_state reports per-object/component
facts; undo_collision reverts by transaction id. An interrupted `set_collision --apply`
can leave a collision_prepared journal that undo_collision refuses: check collision_state, then re-apply
the wanted mode (on restores the originals). The drone's channel is unverified: run
`./uc exec collision channels` while flying and pass --channels if it is not Pawn/PhysicsBody/Vehicle.
Object rules follow catalogue row plus pivot (2 cm): moving an object detaches it from its rule; group
rules survive moves.
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
    live: str = 'yes'  # 'no' never needs the game | 'mixed' offline for some inputs (saved map, file) | 'yes'
    effect: str = 'read'  # 'read' | 'write'
    caution: str = ''
    needs: str = ''  # CLI help: what a 'mixed' operation needs, when not the usual saved map NAME
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


SEARCH_CAP = 200   # search_assets rows per answer (a longer list is cut with a note, never silently)


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
        limit=len(catalog.assets),
        include_hidden=include_hidden,
        include_decals=category.lower() == "decals",
    )
    rows = scenery.collision_filter(rows, collision)
    cap = max(1, min(limit, SEARCH_CAP))
    out = {"count": min(len(rows), cap), "assets": [row.public() for row in rows[:cap]]}
    tokens = query.lower().split()
    if tokens and not any(word_match(hay_of(r), tokens) for r in rows):
        out['word_match'] = False
        out['word_note'] = (f"no row has '{query}' as a word" + (": these only contain it inside a word" if rows else '')
                       + '; ' + _related_hint(catalog, tokens))
    if len(rows) > cap:   # never cut silently: say how many matched and how to see more
        out.update(total=len(rows), truncated=True,
                   note=(f"limit {limit} is above the cap of {SEARCH_CAP}: " if limit > SEARCH_CAP else '') +
                        f"{len(rows) - cap} more rows match; raise limit (up to {SEARCH_CAP}) or narrow the query")
    elif limit > SEARCH_CAP:
        out['note'] = f'limit is capped at {SEARCH_CAP}'
    return out


SEARCH_RELATED = {'bin': 'trash', 'dustbin': 'trash', 'garbage': 'trash', 'waste': 'trash', 'litter': 'trash',
                  'lamp': 'light', 'streetlight': 'light', 'lantern': 'light', 'seat': 'bench', 'post': 'pole',
                  'auto': 'car', 'lorry': 'truck', 'shrub': 'bush'}


def _related_hint(catalog, tokens: list[str]) -> str:
    """Rows found with a related word ('bin' -> trash), or how to look further."""
    for t in tokens:
        alt = SEARCH_RELATED.get(t)
        if alt:
            hits = [r for r in catalog.search(alt, limit=200) if word_match(hay_of(r), [alt])][:5]
            if hits:
                return f"related ('{alt}'): " + ', '.join(f'{r.id} {r.name}' for r in hits)
    return 'try a broader word, a --tag or a --category'


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
    reference: bool = False,
    reference_only: bool = False,
) -> dict:
    """Fresh offscreen scene image. Default: no focus/desktop switch or editor-camera move.

    Pass x,y,z and optional yaw/pitch for independent render pose, not editor movement.
    Engine backend omits UI and never silently falls back. window backend captures
    current editor view; occluded/minimized pixels may be stale. Only explicit
    backend='window', foreground=True permits activation/desktop switching.
    At most 4,194,304 pixels. Inspect image yourself, not with another vision model.
    A loaded textured reference (reference_load) is hidden for the capture unless reference=True;
    reference_only renders nothing but the reference (a before picture for a pair; agent DLL with capture modes).
    """
    from . import live
    target = _capture_target(out) if out else None
    if target is not None:
        _capture_folder(target[0] if target[1] else target[0].parent)
    extra = {'include_reference': True} if reference else {}  # reference: keep a loaded textured reference in view
    if reference_only:
        extra['reference_only'] = True
    path = live.capture_view(x, y, z, yaw, pitch, backend=backend, foreground=foreground, width=width, height=height,
                             **extra)
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
    with agent.session():
        dropped = reference.drop_for_scene_change()  # the native reference belongs to the scene being replaced
        result = agent.load(name, move_camera)
    if dropped:
        result = {**result, 'reference': dropped}
    return result


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


def _sheet_incomplete(result: dict) -> str | None:
    if result.get('failed_count'):
        return f"{result['failed_count']} of {result.get('poses')} poses failed; see shots[].error"
    return None


@op('Capture many poses into one labelled contact sheet PNG', positional=('poses',), check=_sheet_incomplete,
    examples=("./uc capture_sheet 'start 0 -20 6 90 -10; gate1 30 0 8 0 -5' --out sheets/",
              "./uc capture_sheet 'a 0 0 20 0 -60, b 0 0 20 90 -60' --width 320 --height 180 --columns 2"))
def capture_sheet(poses: str, width: int = 480, height: int = 270, columns: int = 0, out: str = '',
                  reference: bool = False, reference_only: bool = False) -> dict:
    """Render up to 60 explicit poses offscreen and tile them into one labelled contact sheet.

    poses: items "label x y z yaw pitch" (metres, degrees; label optional) separated by
    newlines, ';' or ','; '#' starts a comment up to the end of the line; "@path" reads a text
    file with one pose per line. Each frame is width x height (64..1920 x 64..1080, default
    480x270); columns 0 = automatic. Each pose takes the agent lock on its own (about 1.5 s per
    pose) through the engine capture (editor camera FOV, roll 0; the editor camera is not moved).
    A failed pose is reported in shots[].error and drawn as a FAILED tile; the others continue.
    Writes sheet.png, one PNG per pose and index.json into out (a new or empty directory, created
    at the first frame; a non-empty one is refused; default a new folder under
    /tmp/uncrashed-shots/). Editor helper lines are
    hidden from captures; gates, gate numbers, the start arch and a race guide are not. A loaded
    textured reference (reference_load) is hidden for the sheet and shown again afterwards unless
    reference=True; reference_only renders nothing but the reference in every frame (the before half of pairs;
    agent DLL with capture modes).
    A sheet is evidence for looking, not proof of visual quality, clearance or a flyable line.
    """
    return sheet.capture_sheet(poses, width=width, height=height, columns=columns, out=out, include_reference=reference,
                               reference_only=reference_only)


@op('Stamped snapshot of placed objects and gates (read-only)',
    examples=('./uc scene_snapshot --name before_bridge', './uc scene_snapshot --group houses --box -50,-50,50,50'))
def scene_snapshot(group: str = '', box: list[float] | None = None, name: str = '') -> dict:
    """Read-only stamped snapshot of the open scene; changes nothing in the scene.

    One full group export (~1 s for 56k objects; the editor freezes meanwhile) plus the race gates. group limits it
    to a group and its group.* / group_* children, box = x0,y0,x1,y1 metres on pivots. name: 1..64 of letters,
    digits, _ . - (default snap-<date>-<time>); an existing name is refused. Writes gzip JSON (row, group, native
    transform, actor id, session) to UncrashedMCP.snapshots/<name>.json.gz beside the saved map (saved map required).
    Returns only the stamp: map folder, base, session token, object count, scope count, a content fingerprint
    (sorted rounded row/group/transform, no actor ids), gate count and order+transform fingerprint, start gate and
    the file. Gate order is live level order. Equal fingerprints mean the same rounded states, not a working build.
    """
    return snapshots.scene_snapshot(group=group, box=box, name=name)


@op('Compare the open scene with a scene_snapshot (read-only)', positional=('snapshot',),
    examples=('./uc scene_diff before_bridge', './uc scene_diff before_bridge --position_m 0.05 --examples 20'))
def scene_diff(snapshot: str, group: str | None = None, box: list[float] | None = None, position_m: float = 0.01,
               rotation_deg: float = 0.1, scale: float = 0.001, gate_position_m: float = 0.01,
               examples: int = 10) -> dict:
    """Read-only diff of the open scene against a scene_snapshot name or .json.gz file; changes nothing.

    Refuses a snapshot of another map folder or base, an unreadable or edited snapshot, and a scope other than the
    snapshot's own when the snapshot was scoped. group / box (x0,y0,x1,y1 metres, pivots) narrow a whole-scene
    snapshot. Same session: objects pair by actor id first; then (always) by same row nearest within position_m. When
    the session differs (game restart, other base or folder) actor ids are meaningless, so only row+position pairing is
    used and an object moved further than position_m counts as removed + added. Counts added, removed, moved,
    rotated, scaled, regrouped, row_changed, unchanged; tolerances position_m (metres), rotation_deg, scale (absolute),
    gate_position_m. Verdict lines: identity, gate order, gate transforms, start gate, objects. Gates are compared in
    live level order (not proven race order). Returns the first `examples` changes; the full diff is gzip JSON under
    UncrashedMCP.snapshots/diffs/. Geometry bookkeeping only: not visual, connectivity or flight evidence.
    """
    return snapshots.scene_diff(snapshot, group=group, box=box, position_m=position_m, rotation_deg=rotation_deg,
                                scale=scale, gate_position_m=gate_position_m, examples=examples)


@op('First obstruction on a short approach -> opening -> exit path', positional=('path',), variadic='path',
    examples=('./uc check_opening 0,-8,4 0,0,4 0,8,4', './uc check_opening 0,-8,4 0,0,4 0,8,5 --profile cinewhoop --margin_m 0.3'))
def check_opening(path: list[list[float]], profile: str = '5inch', margin_m: float | None = None,
                  channel: str = 'Pawn', spacing_m: float = 0.25, dims_m: list[float] | None = None,
                  near_m: float = 5.0, top: int = 10) -> dict:
    """Read-only opening check: sweep the craft body along 2..8 x,y,z points (approach -> opening -> exit,
    metres, up to 300 m) and report the FIRST obstruction by position along the path (sample, then hit time), not
    by severity, with its actor, row, name and class. Slow like check_route_clearance (one native sweep job plus
    collision queries). profile / dims_m=[width,length,height] / margin_m / channel as in check_route_clearance;
    spacing_m 0.1..2. The path is a spline through the points, so corners are rounded (spline_max_offset_m), and
    the swept box is the craft body only: margin only flags probe near-misses. Every finding carries an evidence
    class: live-collision (native hit, no override), override-active (the object has a DLL collision override per
    collision_state: the stock default was not what was tested; also hits passed only because of an override),
    bounds-estimate (bounds-only data; this check produces none) or unverified (unknown response, or a placed object
    whose pivot lies within near_m of the path box that no trace touched: Blueprint buildings can return no hits,
    so no hit beside them is NOT clearance; hits on objects whose override lookup failed are unverified and make the
    result incomplete). Probe near-misses include gate frames. Findings merge per run of consecutive samples
    (distance_m .. last_distance_m). The first line, verdict, always states the unverified, near-miss and
    override counts. top caps returned findings and objects. Full JSON in a new
    /tmp/uncrashed-clearance/opening-<time>-<id>.json per call. Geometry only: not a flight test.
    """
    return opening.check_opening(path, profile=profile, margin_m=margin_m, channel=channel, spacing_m=spacing_m,
                                 dims_m=dims_m, near_m=near_m, top=top, catalog=_catalog())


def _reference_partial(result: dict) -> str | None:
    if result.get('loaded_chunks', 0) < result.get('chunks', 0):
        return (f"reference loaded {result.get('loaded_chunks')}/{result.get('chunks')} chunks: "
                f"{result.get('stopped') or 'stopped early'}")
    return None


@op('Show a textured OBJ as a view-only reference (not saved)', positional=('obj',), effect='write',
    caution='adds the reference to the open editor at once; it is never saved and does not collide; '
            'reference_clear removes it',
    examples=('./uc reference_load /data/city/model.obj --at 0,0,0 --center 120,-40 --radius_m 150',
              './uc reference_load model.obj --map_origin 51.3397,12.3731 --yaw 0 --datum_m 112.5 --replace'),
    check=_reference_partial)
def reference_load(obj: str, at: list[float] | None = None, map_origin: list[float] | None = None, yaw: float = 0,
                   center: list[float] | None = None, radius_m: float = 0, datum_m: str = 'auto',
                   near_m: float = 160, far_scale: int = 2, axes: str = '', replace: bool = False,
                   min_free_gb: float = 3) -> dict:
    """Load a textured OBJ (+MTL, JPEG map_Kd) into game memory as a view-only reference. Metres, degrees.

    Exactly one of at=[x,y,z] (where the export origin, after center, goes in the map) or map_origin=[lat,lon(,z)]
    (map origin on WGS84; needs metadata.json with origin lat/lon next to the OBJ). yaw turns it about Z.
    center=[east,south] and radius_m pick a region of the export (export metres; radius 0 = everything).
    datum_m: source height put at z (at[2]); 'auto' = 2nd percentile of the region's vertex heights (ground),
    or a number. axes: what source x,y,z point to (default from metadata.json, else east,up,south). near_m: chunks
    within it keep full texture resolution, farther ones are downscaled by far_scale (1..8).
    Packs 96 m chunks (native/ucmesh, cached index; atlas pages only in /dev/shm, deleted after import) and imports
    them nearest first; stops before a chunk would leave less than min_free_gb GiB of host memory free (reported in
    stopped). Non-colliding, untagged, never saved, invisible to audits; one reference at a time (replace=True
    swaps it). load_live clears it. Captures hide it unless reference=True. Returns loaded/total chunks, triangles,
    est_game_gib, the datum used and timings. A failure midway says how many chunks stay loaded.
    """
    return reference.load(obj, at=at, map_origin=map_origin, yaw=yaw, datum_m=datum_m, center=center,
                          radius_m=radius_m, near_m=near_m, far_scale=far_scale, axes=axes, replace=replace,
                          min_free_gb=min_free_gb)


@op('Chunks, triangles and memory of the loaded reference')
def reference_status() -> dict:
    """Read-only state of the textured reference: loaded, hidden, chunks, triangles, texels, pose and estimated game
    memory. available=false means the loaded agent DLL predates the reference verb."""
    return {**reference.status(), 'leftover_packs': reference.leftover_packs()}


@op('Remove the textured reference and leftover pack files', effect='write')
def reference_clear() -> dict:
    """Remove the textured reference from the editor (game memory only; no map changes) and delete any pack folder
    a killed reference_load left in /dev/shm or the scratch cache."""
    return reference.clear()


@op('Hide or show the textured reference', effect='write',
    examples=('./uc reference_visibility', './uc reference_visibility --no-hidden'))
def reference_visibility(hidden: bool = True) -> dict:
    """Hide (default) or show the textured reference. Captures hide it by themselves; this is for working by eye."""
    return reference.set_hidden(hidden)


# ---------------------------------------------------------------------------- 3D scan (no game)


def _scan():
    from .scan import ops  # numpy is loaded only when a scan command runs
    return ops


@op('Survey a textured OBJ: ground, buildings, trees, cars, facades', positional=('obj',), live='no',
    examples=('./uc scan_survey /data/city/scene.obj',
              './uc scan_survey scene.obj --center 120,-40 --radius_m 150 --no-osm --no-vision'))
def scan_survey(obj: str, center: list[float] | None = None, radius_m: float = 0, res_m: float = 0.25, axes: str = '',
                osm: bool | None = None, vision: bool | None = None, facades: bool = True,
                refresh: bool = False, providers: str = 'auto', street_views: bool | None = None) -> dict:
    """Survey a textured photogrammetry OBJ (no game): rasterise it top-down (native/scanraster, cached ucmesh index)
    and derive a ground model, surface classes and objects (buildings with footprint, parts, roof type; trees;
    cars; low structures, shrubs, bridges) with a rule confidence and source each. Metres.

    center=[east,south] export metres and radius_m pick the region (radius 0 = metadata.json area, else the export);
    res_m analysis cell (0.25; the ortho image is twice as fine). osm (default: on when metadata.json has the origin
    lat/lon and the network answers): fetch OpenStreetMap for the bbox only (cached beside the survey) and use
    building footprints to split merged roofs, roads/parking/paths/water as ground classes, tree nodes; credit
    (c) OpenStreetMap contributors, ODbL. vision (default: on when .venv-vision exists; about 1.5 min on a GPU for
    500 x 500 m): aerial detectors (YOLO11-OBB vehicles and DOTA classes, Grounding DINO prompts, SAM 2 outlines) on
    the ortho image confirm or reject geometry cars (carports, sheds, skips), add missed vehicles, pools, boats,
    solar panels; src mesh+vision / vision, models per object. facades (default on, needs vision): every building
    wall is rendered straight on from the mesh and windows, doors, balconies, garage doors are detected and fitted
    to rows / storeys (scan_building, scan_facade). providers (auto: every open data set whose coverage holds the
    bbox, none, or names): Overture buildings, Copernicus GLO-30, ESA WorldCover and regional sets such as Bavarian
    DGM1 / LoD2 or the Dutch 3D BAG; a terrain model replaces the ground model, LoD2-type buildings override
    footprints, heights and roof planes, the rest cross-checks; every object lists agreeing / conflicting sources
    (evidence); only the bbox leaves the machine (scan_sources). street_views (default on with vision and OSM):
    cameras along the streets at 1.6 m and 8 m render the mesh; detections (doors, windows, signs, lamps, benches,
    bins, bollards, bicycles, fences) are lifted to 3D through the render depth and fused across views into objects.
    A survey of the same OBJ/region/parameters is reused unless
    refresh; running it again with vision available adds the vision stage to an older survey. The survey becomes
    current for the other scan_* commands. Disk: ~/.cache/uncrashed-mcp/scan/<key>/ (JSON, no pixels); rasters and
    images stay in /dev/shm (RAM) and are rebuilt on demand. Returns counts and confidence bands per class, class
    shares, timings, the vision report, what was skipped and why.
    """
    return _scan().scan_survey(obj, center=center, radius_m=radius_m, res_m=res_m, axes=axes, use_osm=osm,
                               use_vision=vision, facades=facades, refresh=refresh, providers=providers,
                               street_views=street_views)


@op('Scan surveys, placement, RAM rasters and vision models', live='no',
    examples=('./uc scan_status', './uc scan_status --drop_ram all'))
def scan_status(survey: str = '', drop_ram: str = '') -> dict:
    """Known scan surveys (current one marked, object count, OSM used, placed, rasters in RAM and their MB, stale
    when the OBJ changed since), the frame queries use (map via the placement, else scan), the current survey's
    newest images in RAM, and whether the vision environment and models exist. survey: key prefix filter.
    drop_ram: KEY (prefix) or all frees the RAM rasters and images of those surveys (/dev/shm); rasters are rebuilt
    on the next query, images are gone."""
    return _scan().scan_status(survey, drop_ram=drop_ram)


@op('Set or clear where a scan sits in the map (map frame)', live='mixed', effect='write',
    needs='no game, except --from_reference (reads the reference loaded in the open editor)',
    caution='writes the placement file at once (the reply gives the line that restores the previous one); every '
            'scan_* answer changes frame',
    examples=('./uc scan_place --at 0,0,0 --yaw 0 --datum_m 54.2', './uc scan_place --at 0,0,1',
              './uc scan_place --from_reference'))
def scan_place(at: list[float] | None = None, yaw: float | None = None, datum_m: float | None = None,
               from_reference: bool = False, clear: bool = False, survey: str = '') -> dict:
    """Placement of the survey's OBJ in the map, exactly as reference_load places it: at=[x,y,z] Unreal metres of
    the export origin, yaw degrees about Z, datum_m = source height put at z (yaw and datum_m left out keep the
    current placement's; the first placement needs datum_m, yaw defaults to 0). The reply lists the replaced
    placement and a restore line. from_reference copies the last
    reference_load of the same OBJ on this machine after checking that the reference loaded in the open editor
    has that pose (needs the game). clear removes it. Stored per OBJ path (every survey of it shares it). With a
    placement every scan_* query takes and returns map metres (x east, y south, z up); without, export metres
    (east, south, source up). --frame scan forces export metres."""
    return _scan().scan_place(at=at, yaw=yaw, datum_m=datum_m, from_reference=from_reference, clear=clear,
                              survey_key=survey)


@op('Ground, top and height above ground at x,y points', positional=('points',), variadic='points', live='no',
    examples=('./uc scan_height 12,-40 15.5,-38', './uc scan_height 12,-40,8 15.5,-38,8 --agl'))
def scan_height(points: list[list[float]], agl: bool = False, frame: str = 'auto', survey: str = '') -> dict:
    """Per x,y point (1..1024, metres in the result frame; scan_grid for whole boxes): ground_z (filtered ground
    model), top_z (highest mesh surface), height_m above ground, under_m (top minus lowest surface when > 1 m:
    overhang, deck, canopy), surface class and the object id/class there; under a roof, crown or low object also
    ground_cls, the class of the nearest visible ground and its distance. x,y,z points add z, agl_m and clearance_m
    above the top surface; agl: each z is metres above the scan ground. frame: auto (map when placed), map or scan."""
    return _scan().scan_height(points, agl=agl, frame=frame, survey_key=survey)


@op('Ground and top heights along a line', positional=('start', 'end'), live='no',
    examples=('./uc scan_profile 0,0 80,0 --step_m 2', './uc scan_profile 0,0,6 80,0,6 --agl'))
def scan_profile(start: list[float], end: list[float], step_m: float = 1, agl: bool = False, frame: str = 'auto',
                 survey: str = '') -> dict:
    """Samples every step_m (at most 201) from start x,y to end x,y: distance, ground_z, top_z, height_m, surface
    class and slope_pct to the next sample (only between two open-ground samples: roof and crown edges are not
    ground slopes); plus open-ground min/max, top max, steepest and mean ground slope. x,y,z ends add the line z and
    clearance_m per sample and the minimum; agl: the end z values are metres above the scan ground (the line then
    follows the ground). For roads, slopes, bridges, walls, flight lines."""
    return _scan().scan_profile(start, end, step_m=step_m, agl=agl, frame=frame, survey_key=survey)


@op('What is at x,y: surface, nearby objects with distances', positional=('point',), live='no',
    examples=('./uc scan_at 12,-40 --radius_m 15',))
def scan_at(point: list[float], radius_m: float = 10, limit: int = 15, frame: str = 'auto',
            survey: str = '') -> dict:
    """The point's heights and surface (as scan_height), surface class shares within radius_m (0..200), and the
    objects within radius_m by distance to their footprint or box (0 = inside), at most limit."""
    return _scan().scan_at(point, radius_m=radius_m, limit=limit, frame=frame, survey_key=survey)


@op('List scan objects by class, box or distance', live='no',
    examples=('./uc scan_objects --classes car --box 0,0,60,40', './uc scan_objects --classes tree,car --near 10,5 --radius_m 30',
              './uc scan_objects --classes building --sort windows --limit 5',
              './uc scan_objects --classes car --near 10,5 --nearest 3'))
def scan_objects(classes: list[str] | None = None, box: list[float] | None = None, near: list[float] | None = None,
                 radius_m: float = 50, min_conf: float = 0, limit: int = 30, sort: str = '', nearest: int = 0,
                 frame: str = 'auto', survey: str = '') -> dict:
    """Objects of the survey filtered by classes (building, building_part, tree, car, truck, bridge, low_structure,
    small_object, shrub, pool, ...), box x0,y0,x1,y1 or near x,y within radius_m, and min_conf. Rows: id, class,
    conf, centre, ground_z, height, length, width, yaw (long axis), src (mesh, mesh+osm, mesh+vision, vision),
    colour (a plain word and rgb hex of the median texture colour) and flag (a reason to doubt it: flat or short
    car, not a car (stack, clutter), weak vision-only, slender tree or one against a roof of its height, part of a
    shared OSM building, footprint overlapping another building, OSM footprint disagreeing with the heights).
    Buildings add area_m2, storeys, windows (facade survey) and roof. sort: near / distance (default with near),
    conf (default), height, area, storeys, or windows (the most windows on one facade; adds best_side,
    side_windows). At most limit (1..500) with the uncut total and per-class counts. nearest N (needs near): the N
    nearest matches with their distance, however far. No hit: summary names the nearest match outside the box or
    radius and its distance."""
    return _scan().scan_objects(classes=classes, box=box, near=near, radius_m=radius_m, min_conf=min_conf,
                                limit=limit, sort=sort, nearest=nearest, frame=frame, survey_key=survey)


@op('Surface class shares and outlines in a box', positional=('box',), live='no',
    examples=('./uc scan_ground 0,0,60,40',))
def scan_ground(box: list[float], min_area_m2: float = 20, limit: int = 20, simplify_m: float = 0.5,
                frame: str = 'auto', survey: str = '') -> dict:
    """Shares of paved, road, parking, path, grass, water, bare, building, vegetation and low_object cells in box
    x0,y0,x1,y1 (<= 300 x 300 m), and the largest class regions (>= min_area_m2) as outlines simplified by
    simplify_m, at most limit. Road / parking / path come from OSM when the survey used it."""
    return _scan().scan_ground(box, min_area_m2=min_area_m2, limit=limit, simplify_m=simplify_m, frame=frame,
                               survey_key=survey)


@op('Footprint, parts, heights, roof and facades of a building', positional=('building',), live='no',
    examples=('./uc scan_building 42', './uc scan_building 43 --no-survey_facades'))
def scan_building(building: int, min_facade_m: float = 2, survey_facades: bool = True, frame: str = 'auto',
                  survey: str = '') -> dict:
    """One building by object id: footprint polygon (roof outline), parts (each with footprint, heights and roof),
    ground_z / top_z, eave_m, height_m and ridge_m above ground, roof type with ridge direction and roof_shape
    (planes fitted to the roof heights: form, slope_pct per plane, pitch_deg, ridge / valley line with heights;
    low pitches the normals miss), colours (roof, facade tone), OSM tags (osm_conflict when the mapped footprint
    disagrees with the heights), objects on its roof, and walls >= min_facade_m (ends, length, outward direction;
    from OSM wall lines when mapped). A building_part id answers for that part (its footprint, heights, roof and
    walls) and names the parent. With a facade survey each wall has status, storeys, window count, columns and
    spacing, sill / head heights per window row above ground, doors, balconies, tone and check (window rows that
    need a look); the building has storeys and totals. A building the survey skipped (small, low) is surveyed now
    when vision is available (survey_facades, default on; about 10 s) and stored."""
    return _scan().scan_building(building, min_facade_m=min_facade_m, survey_facades=survey_facades, frame=frame,
                                 survey_key=survey)


@op('Facade image of a building wall with windows and doors drawn', positional=('building',), live='no',
    examples=('./uc scan_facade 42', './uc scan_facade 42 --side 1'))
def scan_facade(building: int, side: int | None = None, labels: bool = True, frame: str = 'auto',
                survey: str = '') -> dict:
    """Render a building's walls straight on from the scan mesh (one panel per wall, or only side; at most the six
    longest), with detected windows, doors, garage doors and balconies, window rows (sill / head lines), ground line
    and a metre grid along the wall and above ground. Works without a facade survey (render only). The PNG is
    written to /dev/shm (RAM only; derived from the scan: keep it on this machine). Returns its path and the per
    wall rows as scan_building; open the image to check detections before building from them."""
    return _scan().scan_facade(building, side=side, labels=labels, frame=frame, survey_key=survey)


@op('Is a path above the scan surface? first block, min clearance', positional=('points',), variadic='points',
    live='no', examples=('./uc scan_free 0,0,30 80,10,25', './uc scan_free 0,0,30 40,0,12 80,0,30 --radius_m 1',
                         './uc scan_free 0,0,8 100,-100,8 --agl'))
def scan_free(points: list[list[float]], radius_m: float = 0, margin_m: float = 0.5, agl: bool = False,
              frame: str = 'auto', survey: str = '') -> dict:
    """Straight segments through 2..64 x,y,z points tested against the scan's highest surface every half cell,
    widened by radius_m (0..20, craft half-width). clear when every sample is margin_m above it; else the first
    blocked sample (position, surface_z, object), blocked length and the minimum clearance. Scan only: placed
    objects are not seen (check_route_clearance / check_opening in the game); under bridges reads as solid.
    agl: each z is metres above the ground model and the line follows the ground between points (reported z
    values are absolute)."""
    return _scan().scan_free(points, radius_m=radius_m, margin_m=margin_m, agl=agl, frame=frame, survey_key=survey)


@op('Top-down scan image with metre grid and object ids', live='no',
    examples=('./uc scan_view --center 12,-40 --size_m 60', './uc scan_view --box 0,0,120,80 --layer classes'))
def scan_view(box: list[float] | None = None, center: list[float] | None = None, size_m: float = 60,
              layer: str = 'ortho', labels: bool = True, classes: list[str] | None = None, px: int = 900,
              frame: str = 'auto', survey: str = '', compare: str = '', deviations: int = 25) -> dict:
    """Render a top-down crop (box x0,y0,x1,y1, or center x,y with size_m; sides 1..600 m) of layer ortho (texture
    colour), ndsm (height above ground) or classes, about px pixels wide, with a labelled metre grid in the result
    frame (north up when unplaced), object outlines with ids (classes filters them) and a legend. compare: a
    capture_topdown PNG; then three panels on its box and pixel grid: scan layer | build | edge overlay (scan
    magenta, build green, white where both agree), with the worst `deviations` (0..500) of the last scan_compare
    labelled (never ground pieces; warnings when that report is stale or covers another box). The PNG
    is written to /dev/shm (RAM only; derived from the scan: keep it on this machine). Returns its path; open it."""
    return _scan().scan_view(box=box, center=center, size_m=size_m, layer=layer, labels=labels, classes=classes,
                             px=px, frame=frame, survey_key=survey, compare=compare, deviations=deviations)


def _scan_mapping():
    from .scan import mapping
    return mapping


@op('Contact sheet of scan_view crops around points or objects', positional=('points',), variadic='points',
    live='no', examples=('./uc scan_sheet 12,-40 60,-10', './uc scan_sheet --classes car --box 0,0,120,80 --limit 12',
                         './uc scan_sheet --objects 997,1014 --layer ndsm'))
def scan_sheet(points: list[list[float]] | None = None, objects: list[str] | None = None,
               classes: list[str] | None = None, box: list[float] | None = None, min_conf: float = 0, limit: int = 16,
               size_m: float = 0, layer: str = 'ortho', labels: bool = True, px: int = 360, columns: int = 0,
               frame: str = 'auto', survey: str = '') -> dict:
    """Several top-down scan crops (as scan_view: layer ortho, ndsm or classes; metre grid, object outlines and ids)
    tiled into one labelled PNG: one crop per x,y point (size_m, default 40 m), per object id in objects, or per
    object picked by classes / box / min_conf (highest conf first). size_m 0 fits each object (3 x its length,
    14..120 m). At most limit (1..36) crops of about px pixels; columns 0 = automatic. The PNG is written to /dev/shm
    (RAM only; derived from the scan: keep it on this machine). Returns its path and the crop list."""
    return _scan_mapping().scan_sheet(points=points, objects=objects, classes=classes, box=box, min_conf=min_conf,
                                      limit=limit, size_m=size_m, layer=layer, labels=labels, px=px, columns=columns,
                                      frame=frame, survey_key=survey)


@op('Best catalogue rows for scan objects: size, aspect, roof, scale', positional=('object_id',), live='no',
    examples=('./uc scan_match 997', './uc scan_match --classes car,tree --box 0,0,120,80 --limit 10'))
def scan_match(object_id: int | None = None, classes: list[str] | None = None, box: list[float] | None = None,
               min_conf: float = 0.5, top: int = 3, limit: int = 30, rows: list[str] | None = None,
               include_hidden: bool = False, frame: str = 'auto', survey: str = '') -> dict:
    """Catalogue stand-ins for scan objects (the rows search_assets / get_assets show). Per object class a candidate
    set (car / truck: vehicle meshes; boat; tree: foliage trees; shrub: bushes; building: building and house rows;
    pool: water; low_structure / small_object: props, barriers, industrial) is scored by box size after the
    suggested scale (vehicles 0.85..1.15 uniform, trees scale height and crown, buildings per axis 0.5..2), aspect,
    roof type and ridge direction (from the catalogue mesh hulls; Blueprint rows: roof unknown) and a colour word in
    the row name when there is one. object_id: the top candidates for one object with score, fit_err_pct,
    raw_err_pct (scale 1), scale x,y,z, spawn yaw_deg in the result frame and the scaled size. Otherwise every object
    picked by classes (default car, truck, tree, building), box x0,y0,x1,y1 and min_conf gets its best row (at most
    limit listed) and each class the rows used most. rows limits the candidates to given row ids; include_hidden
    adds rows the picker hides. Metres, degrees."""
    return _scan_mapping().scan_match(object_id=object_id, classes=classes, box=box, min_conf=min_conf, top=top,
                                      limit=limit, rows=rows, include_hidden=include_hidden, frame=frame,
                                      survey_key=survey)


@op('Write spawn + drop lines for scan objects to a file (not run)', live='no', effect='write',
    caution='writes a new plan .txt and .json at once (default ./scan-plans/); never sends anything to the game',
    examples=('./uc scan_plan --box 0,0,120,80', './uc scan_plan --classes car --min_conf 0.8 --out plans/cars.txt',
              './uc scan_plan --classes ground,car,tree --box -40,-40,40,40 --flat_tol_m 0.3',
              './uc scan_plan --near 0,0 --radius_m 40 --classes car,tree'))
def scan_plan(box: list[float] | None = None, classes: list[str] | None = None, min_conf: float = 0.6, out: str = '',
              buildings: str = 'hint', lift_m: float = 0.5, limit: int = 2000, include_hidden: bool = False,
              ground_cell_m: float = 2, flat_tol_m: float = 0.3, ground_yaw: float | None = None,
              max_tile_m: float = 16, near: list[float] | None = None, radius_m: float | None = None, weak: str = 'review',
              survey: str = '', ground_smooth_m: float = 4) -> dict:
    """Write agent commands that rebuild scan objects from catalogue rows, for review; nothing is sent to the game.
    Needs the scan placement (scan_place): every line is in map coordinates (agent centimetres). Per object picked
    by classes (default car, truck, tree, building), box x0,y0,x1,y1 map metres (or a circle: near x,y and
    radius_m, centres within it) and min_conf: its best scan_match
    row, a spawn whose mesh box centre sits on the object centre lift_m above the scan ground (pivot from
    origin_cm / extent_cm, so base, centre, top and offset pivots all land), yaw from the object's oriented box
    (buildings: ridge), scale, group=scan.<class>.<id>, then `drop last`. Cars and trucks need visible ground
    around them (on roofs, decks or water they are skipped); trees rotate through rows that fit nearly as well,
    with a varied yaw; 'trees' taller than 3 x a crown under 4 m are skipped (roof greenery, building edges). Buildings are not built: buildings=hint (default) writes a comment block (footprint, ground_z,
    eave / ridge heights, roof and ridge, walls with storeys, window rows, sill / head heights and doors from the
    facade survey) and the best stand-in as '#?' lines to uncomment by hand; standin spawns it; none leaves them
    out. out: a .txt path or folder (default ./scan-plans/<survey>-<time>.txt); a .json with the same data and the
    building hints is written beside it; existing files are never overwritten. limit caps planned objects (1..5000).
    The class ground (with box) paves the box with smooth fitted surfaces, never per-cell scan heights: one robust
    plane per connected paved area when it fits within flat_tol_m, else a stiff smooth surface; the kerbside one
    snapped kerb (8..20 cm or none) above the carriageway; large tilted tiles (up to max_tile_m) on a ground_cell_m
    lattice turned by ground_yaw (default: along the streets or the map axes, whichever needs fewer tiles) whose
    corners lie on that surface, so neighbours meet within a centimetre; ground_smooth_m is the bending length
    (larger: flatter, fewer tiles, further from the scan); a wall or embankment becomes a ramp tile, not a step; material by the surface class (road /
    parking asphalt, paved concrete, path pavement, grass and bare dirt; under crowns and parked cars the nearest
    open ground); cells under buildings are left out. The reply gives predicted_steps and the residual to the scan
    ground; objects in the same plan stand on the planned surface. A weak fit (size off
    by more than 40 % after scaling, or a non-foliage best score under 0.5: a pallet stack planned as a van) is
    written as '#?' lines and listed under review (weak=review, default); weak=keep spawns it with a WEAK FIT
    comment, weak=skip leaves it out. Run the file with agent_batch after reading it."""
    return _scan_mapping().scan_plan(box=box, classes=classes, min_conf=min_conf, out=out, buildings=buildings,
                                     lift_m=lift_m, limit=limit, include_hidden=include_hidden,
                                     ground_cell_m=ground_cell_m, flat_tol_m=flat_tol_m, ground_yaw=ground_yaw,
                                     max_tile_m=max_tile_m, ground_smooth_m=ground_smooth_m,
                                     near=near, radius_m=radius_m, weak=weak, survey_key=survey)


@op('Placed objects vs scan objects: position, yaw, size, height, fix', positional=('snapshot',), live='mixed',
    needs='no game with a scene_snapshot .json.gz file, otherwise the open track editor',
    examples=('./uc scan_compare', './uc scan_compare before_bridge --box 0,0,120,80 --classes car,tree'))
def scan_compare(snapshot: str = '', box: list[float] | None = None, group: str = '', classes: list[str] | None = None,
                 min_conf: float = 0.5, position_m: float = 1.0, height_m: float = 1.0, yaw_deg: float = 10,
                 size_tol: float = 0.15, cell_m: float = 5, top: int = 20, iou: float = 0.6, flat_tol_m: float = 0.3,
                 survey: str = '') -> dict:
    """Read-only check of placed objects against the scan; changes nothing. Placed objects come from the open
    editor (one read-only group export) or from a scene_snapshot name / .json.gz file (a file needs no game).
    Needs the scan placement. Scan objects picked by classes (default car, truck, tree, building), box x0,y0,x1,y1
    map metres and min_conf pair with placed objects by group tag first: a group named scan.<class>.<id>,
    <class>.<id> or b<id> with any suffix (b57nw) stands for that scan object with all its objects; then the
    nearest unpaired placed object of a fitting kind (vehicle, foliage); buildings also take untagged pieces standing
    on their cells. Buildings are compared as 0.5 m height rasters of the pieces' top surfaces (inside box when
    given): iou, coverage, missing / extra m2, median height difference, best_shift_m; status shape_off when iou is
    under iou, offset when a move within 3 m gains overlap (fix: edit_group delta), height_off beyond height_m.
    Point objects by their visible size (asset_profile): offset (centres further than position_m apart), yaw_off
    (long axes more than yaw_deg apart, modulo the row's symmetry), size_off (size_ratio off by more than size_tol;
    trees: crown), height_off (top beyond height_m; trees 20 %), facing_off or matched, with a transform fix line
    (centimetres, degrees, - keeps). Ground pieces (tiles, markings, decals, lawn foliage, slabs whose top is the
    ground) and gameplay rows are only counted; other placed objects nothing explains are listed, tallest first.
    ground_steps: height steps where neighbouring ground tiles meet (count over 1 cm, p90, max, worst places; a
    smooth ground has none except kerbs) and tiles whose top is more than flat_tol_m from the scan ground.
    height_cells compares scan and built height per cell_m cell. group narrows placed objects (group and its . / _
    children). At most top rows per list; the full report is JSON in the survey folder. Shape bookkeeping, not
    visual proof: look with capture_topdown and scan_view compare."""
    return _scan_mapping().scan_compare(snapshot=snapshot, box=box, group=group, classes=classes, min_conf=min_conf,
                                        position_m=position_m, height_m=height_m, yaw_deg=yaw_deg, size_tol=size_tol,
                                        cell_m=cell_m, top=top, iou=iou, flat_tol_m=flat_tol_m, survey_key=survey)


@op('Straight-down build image on the scan_view pixel grid',
    examples=('./uc capture_topdown --box 0,0,120,80', './uc capture_topdown --center 12,-40 --size_m 60 --reference only'))
def capture_topdown(box: list[float] | None = None, center: list[float] | None = None, size_m: float = 60,
                    px: int = 900, reference: str = 'hide', projection: str = 'auto', ground_z: float | None = None,
                    top_z: float | None = None, survey: str = '') -> dict:
    """Render the open editor straight down over box x0,y0,x1,y1 (or center x,y with size_m; map metres) on exactly
    the pixel grid scan_view uses for the same box and px, north up, without moving the editor camera.
    projection auto: orthographic (needs the agent DLL with capture modes), else a narrow perspective camera high
    above (the reply states projection and parallax); ortho / perspective force one. reference: hide (default:
    a loaded textured reference is hidden for the capture), include, or only (render nothing but the reference:
    a before picture at any time; needs the new DLL). ground_z / top_z place the camera; default from the scan
    (placement) in the box, else 0 / 60 m. The PNG and a .json sidecar go to /dev/shm (RAM); next: scan_view
    with compare set to the image path."""
    return _scan_topdown().capture_topdown(box=box, center=center, size_m=size_m, px=px, reference=reference,
                                           projection=projection, ground_z=ground_z, top_z=top_z, survey_key=survey)


def _scan_topdown():
    from .scan import topdown
    return topdown


@op('Visible size, symmetry, front, crown and tilt of catalogue rows', positional=('rows',), variadic='rows',
    live='no', examples=('./uc asset_profile NewRow_168 NewRow_52', './uc asset_profile NewRow_257 --refresh'))
def asset_profile(rows: list[str], refresh: bool = False) -> dict:
    """Measured profile per catalogue row, as a scan measures objects: visible_m (top silhouette extents along the
    row's x and y and its 95th percentile top height, metres), visible_frac of the bounding box, long_axis_deg of
    the silhouette in the row frame, symmetry and yaw_period_deg (a yaw error counts modulo it: 0 round, 90 square,
    180 symmetric, 360 none; front is +x by engine convention, not measured), crown_m / crown_frac / crown_base_m
    for foliage (a tree's bounding box reaches its outermost leaf cards: scale trees by crown, never by box),
    panel tilt and facing, footprint shape. From the game's cooked LOD0 when the paks are installed, else the
    simplified hulls in data/meshes.bin; Blueprints fall back to the box. Cached per catalogue version; refresh
    measures again. scan_match, scan_plan and scan_compare use these profiles."""
    from .scan import profile
    return profile.asset_profile(rows, refresh=refresh)


@op('Open data providers of a place or survey: coverage, licence, fusion', live='no',
    examples=('./uc scan_sources', './uc scan_sources --latlon 49.4539,11.0775 --radius_m 150'))
def scan_sources(latlon: list[float] | None = None, radius_m: float = 150, survey: str = '') -> dict:
    """Data providers scan_survey can use. With latlon=[lat, lon] (degrees) and radius_m: which providers cover
    that place (offline test of their published extents; regional tiles are confirmed when scan_survey fetches).
    Without: the current survey's providers (used, covering, why not, fetch seconds, licence), the attribution
    lines to credit, how each source fused, the declined sources with the reason (Mapillary), the fusion report
    (terrain offset and residuals, surface / land-cover checks, buildings overridden / split / added / flagged,
    agreeing and conflicting objects) and the street-view report. Only a bbox, tile names or byte ranges of public
    files ever leave the machine. survey: key prefix."""
    return _scan().scan_sources(latlon=latlon, radius_m=radius_m, survey_key=survey)


@op('Raster export of a box: ground, top, height, class, object id', positional=('box',), live='no',
    examples=('./uc scan_grid 0,0,120,80', './uc scan_grid -60,-60,60,60 --res_m 0.5 --layers top,class'))
def scan_grid(box: list[float], res_m: float = 1, layers: list[str] | None = None, out: str = '', frame: str = 'auto',
              survey: str = '') -> dict:
    """The scan sampled on a regular grid over box x0,y0,x1,y1 (result frame) every res_m (0.1..20; at most 1,000,000
    cells): per cell dtm (mean ground z), top (highest surface z), height above ground, class (code of the highest
    sub-sample; codes in the header) and object (scan object id, -1 none); layers picks some. Written as one
    compressed numpy .npz with a .json header (grid origin, cell size, layout, class codes) to /dev/shm, or to
    out (a new .npz path or a folder): derived from the scan, keep it on this machine. Returns the paths and a
    summary (z ranges, class shares, the objects with the most cells). For heights at a few points use
    scan_height; this replaces many calls when a whole area is needed (ground cells, height rasters, masks)."""
    return _scan().scan_grid(box, res_m=res_m, layers=layers, out=out, frame=frame, survey_key=survey)
