"""MCP server: edit the open Uncrashed track editor live, through an agent injected into the game.

Tool-level units are metres and degrees (Z up, yaw 0 = +X, 90 = +Y). The agent's own command
language (agent_exec, agent_batch) speaks centimetres, like the game. Asset rows are
DT_MapEditorAssets ids (NewRow_N) from search_assets.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.utilities.types import Image

from . import agent, assemblies, collide, environment, groups, managed
from .catalog import load_catalog
from .live import capture_view as live_capture_view
from .live import move_camera as live_move_camera
from .live import read_camera as live_read_camera
from .live import release_camera as live_release_camera
from .mapfile import list_user_maps, load_map, sanitize_name
from .paths import game_running, maps_dir, steam_id, workshop_dir

RELOAD = "Call load_live with a map name to show a saved map in the open editor."

OUTPUT_LINES = 40      # default reply length for agent_exec / agent_batch
OUTPUT_LINES_MAX = 400  # hard cap, whatever the caller asks for

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
  setmap <folder> | base <row>           retarget the save folder | switch the base level
  floating [gap] [limit] [x0 y0 x1 y1]   objects hovering more than gap above anything
  collide <file> [x0 y0 x1 y1] [gap]     collision facts of every placed object into a file (check_collisions reads them)
  trace x y z_from z_to                  what a vertical ray hits, once per collision object type
  terrain x y z_from z_to                landscape height only, ignoring foliage and placed objects
  drop <actor|last> [step] [max] [embed] [dry]   rest it exactly on what is under it (lifts it if sunk in); last = the last spawn
  cells <cell_m> <rows|!rows|*> [zmax_m] occupancy grid (metres here), comma-separated row filter
  console <cmd> | cvar <name> | raise [off]   console command | read a console variable | game window on top
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

{VERBS}
""".strip()

mcp = MCPServer("uncrashed", instructions=INSTRUCTIONS)


def _dump(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":"))


def _err(exc: Exception) -> str:
    return _dump({"error": str(exc)})


def _catalog():
    return load_catalog()


def _require(name: str):
    folder = sanitize_name(name)
    path = maps_dir() / folder / f"{folder}.json"
    if not path.exists():
        raise FileNotFoundError(f"no local map {folder!r} in {maps_dir()}")
    return load_map(folder)


def _clip(lines: list[str], max_lines: int) -> list[str]:
    limit = max(1, min(int(max_lines), OUTPUT_LINES_MAX))
    if len(lines) <= limit:
        return lines
    return lines[:limit] + [f"... {len(lines) - limit} more lines (max_lines up to {OUTPUT_LINES_MAX})"]


@mcp.tool()
def status() -> str:
    """Where maps are written, whether Uncrashed is running, and how many assets are known."""
    catalog = _catalog()
    workshop = workshop_dir()
    workshop_count = len(list(workshop.iterdir())) if workshop else 0
    return _dump(
        {
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
    )


@mcp.tool()
def list_gates() -> str:
    """Race-gate row names for the gate command. DefaultOneW is a normal gate; STARTGATE is a start arch."""
    return _dump({"gates": _catalog().library["gates"]})


@mcp.tool()
def search_assets(
    query: str = "",
    category: str = "",
    tag: str = "",
    limit: int = 20,
    include_hidden: bool = True,
) -> str:
    """Find editor assets by name, material, tag or id.

    Categories include Wall, Big structure, Foliage, Big_assets, Small_assets, Basic shape.
    Tags include building, house, tree, vehicle, ground, fence. Decals are omitted unless category is Decals.
    hidden=true means the picker hides the row; the JSON can still spawn it. size_m is the unscaled mesh.
    """
    catalog = _catalog()
    rows = catalog.search(
        query=query,
        category=category or None,
        tag=tag or None,
        limit=max(1, min(limit, 50)),
        include_hidden=include_hidden,
        include_decals=category.lower() == "decals",
    )
    return _dump({"count": len(rows), "assets": [row.public() for row in rows]})


@mcp.tool()
def get_assets(ids: list[str]) -> str:
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
    return _dump({"assets": found, "missing": missing})


@mcp.tool()
def list_maps() -> str:
    """Local maps the track editor can load, newest first."""
    return _dump({"maps_dir": str(maps_dir()), "maps": list_user_maps()})


@mcp.tool()
def inspect_map(name: str) -> str:
    """Instance counts, bounds and weather for one local map. Does not dump every transform."""
    try:
        doc = _require(name)
        return _dump(doc.summary(_catalog()))
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def camera_pose() -> str:
    """Read the live editor camera. Metres and degrees. Requires Uncrashed to be open in the track editor."""
    try:
        return _dump(live_read_camera())
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def move_camera(x: float, y: float, z: float, yaw: float, pitch: float = -20, hold: bool = True) -> str:
    """Move the live editor camera without clicking. Metres, degrees. Negative pitch looks down.

    Native game-thread transforms; no external memory writer or helper process.
    hold disables editor camera movement until release_camera or 120-second watchdog
    (previous enable state restored). No window activation, keys or mesh placement.
    """
    try:
        return _dump(live_move_camera(x, y, z, yaw, pitch, hold))
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def release_camera() -> str:
    """Stop holding the editor look so mouse look works again. The last position stays."""
    try:
        live_release_camera()
        return _dump({"held": False, **live_read_camera()})
    except Exception as exc:
        return _err(exc)


@mcp.tool(structured_output=False)
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
) -> list:
    """Fresh offscreen scene image. Default: no focus/desktop switch or editor-camera move.

    Pass x,y,z and optional yaw/pitch for independent render pose, not editor movement.
    Engine backend omits UI and never silently falls back. window backend captures
    current editor view; occluded/minimized pixels may be stale. Only explicit
    backend='window', foreground=True permits activation/desktop switching.
    At most 4,194,304 pixels. Inspect image yourself, not with another vision model.
    """
    try:
        path = live_capture_view(x, y, z, yaw, pitch, backend=backend, foreground=foreground, width=width, height=height)
        return [path.with_suffix('.json').read_text(), Image(path=path)]
    except Exception as exc:
        return [_err(exc)]


@mcp.tool()
def load_live(name: str, move_camera: bool = True) -> str:
    """Show a saved map in the open editor without leaving it.

    Switches the base level if needed, removes what is placed, then places every object
    and gate of the map with the editor's own functions in one pass. Afterwards the
    editor's Save button (and save_live) write back to this map. move_camera puts the
    camera where the map's saved view is.
    """
    try:
        return _dump(agent.load(name, move_camera))
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def save_live(name: str = "", display_name: str = "") -> str:
    """Save the open editor level with the game's own SaveMap (no key presses).

    Everything placed live or by hand is written; deleted objects are gone. name is the
    map folder; empty overwrites the map that is open. display_name is the list title;
    empty keeps the current one. Writes the JSON and .info like the Save button.
    """
    try:
        return _dump(agent.save(name, display_name))
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def agent_exec(command: str, max_lines: int = OUTPUT_LINES, timeout: float = 30) -> str:
    """Run one agent command in the open editor and return its reply as text.

    Centimetres and degrees. The full command list is in the server instructions; the common ones:
    spawn, gate, laps, gates, actors, get, transform, replace, destroy, destroyat, destroybox,
    mapinfo, save, floating. Example: "spawn NewRow_1 1500 -2000 0 0 90 0 1 1 1".
    The reply is cut to max_lines (default 40, at most 400).
    """
    try:
        reply = agent.execute(command, timeout)
        status = "ok" if reply["ok"] else "failed"
        return "\n".join([f"{status} {reply['head']}"] + _clip(reply["lines"], max_lines))
    except Exception as exc:
        return f"error {exc}"


@mcp.tool()
def agent_batch(commands: list[str], quiet: bool = True, max_lines: int = OUTPUT_LINES, timeout: float | None = None) -> str:
    """Run many agent commands in one game-thread pass (the fast way to build: thousands per call).

    Same command language as agent_exec (centimetres). quiet (default) reports only the summary and
    the failures; quiet=false also returns each command's reply, cut to max_lines.
    """
    try:
        if not commands:
            raise ValueError("no commands")
        results, summary = agent.batch(commands, timeout=timeout, quiet=quiet)
        failures = [f"#{i} {' '.join(lines)}" for i, lines in sorted(results.items()) if any(l.startswith("error") for l in lines)]
        out = [summary or f"batch lines={len(commands)}"]
        if failures:
            out += ["failures:"] + _clip(failures, min(max_lines, 20))
        if not quiet:
            out += _clip([f"#{i} {l}" for i, lines in sorted(results.items()) for l in lines], max_lines)
        return "\n".join(out)
    except Exception as exc:
        return f"error {exc}"


COLLIDE_DIR = Path("/tmp/uncrashed-collide")


@mcp.tool()
def check_collisions(
    x0: float | None = None,
    y0: float | None = None,
    x1: float | None = None,
    y1: float | None = None,
    top: int = 15,
    include_minor: bool = False,
) -> str:
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
    try:
        COLLIDE_DIR.mkdir(parents=True, exist_ok=True)
        for old in COLLIDE_DIR.glob("col_r*.txt"):
            old.unlink()
        region = environment.validate_region(x0, y0, x1, y1)
        if region is None:
            bounds = environment.get_world_bounds('placed')
            if not bounds['available']:
                return _dump({'objects_checked': 0, 'issues': {}, 'top': [], 'scope': 'no placed objects'})
            if bounds['truncated']:
                raise RuntimeError('loaded scene bounds are incomplete; pass an explicit scan region')
            lo, hi = bounds['aabb']['min'], bounds['aabb']['max']
            region = (lo[0]-5, lo[1]-5, hi[0]+5, hi[1]+5)
            grid = min(10, max(1, math.ceil(math.sqrt(bounds['actor_count']/400))))
        else:
            grid = 2
        commands, paths = collide.scan_commands(COLLIDE_DIR, agent._windows_path, region_m=region, grid=grid)
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
        (COLLIDE_DIR / "report.json").write_text(json.dumps(report))
        (COLLIDE_DIR / "report.html").write_text(collide.render_html(report, title=f"{title} collision check"))
        out = collide.brief(report, top=max(1, min(int(top), 60)), include_minor=include_minor)
        out["report_html"] = str(COLLIDE_DIR / "report.html")
        out['region_m'] = region
        out['scope'] = 'Placed-object clashes and support connectivity only. Use check_foundations / check_environment for footprint and environment checks.'
        return _dump(out)
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def get_object_bounds(actor: str) -> str:
    """Live world-space AABB, optional oriented mesh bounds and transform. Metres.

    Actor is a live hex address from agent_exec actors. AABB encloses owned actor
    components; child actors are not included. OBB needs a static-mesh root.
    Bounds enclose empty space too; they are not exact occupied geometry.
    """
    try:
        return _dump(environment.get_object_bounds(actor))
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def get_world_bounds(scope: str = 'placed') -> str:
    """Bounds of loaded placed objects, landscape, or explicit playable bounds. Metres.

    scope: placed | terrain | playable. Terrain extent is NOT a playable boundary.
    playable returns unavailable unless an actor is explicitly tagged PlayableBounds.
    Never guesses from arbitrary blocking volumes or assumes a fixed map size.
    """
    try:
        return _dump(environment.get_world_bounds(scope))
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def check_foundations(
    actors: list[str] | None = None,
    x0: float | None = None, y0: float | None = None,
    x1: float | None = None, y1: float | None = None,
    spacing: float = .75, tolerance: float = .3, max_drop: float = 50,
    include_structural: bool = False, top: int = 15,
) -> str:
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
    try:
        region = environment.validate_region(x0, y0, x1, y1)
        return _dump(environment.audit(_catalog(), 'foundation', actors=actors, region=region,
                                      spacing=spacing, tolerance=tolerance, max_drop=max_drop,
                                      include_canopies=False, include_structural=include_structural, top=top))
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def check_environment(
    actors: list[str] | None = None,
    x0: float | None = None, y0: float | None = None,
    x1: float | None = None, y1: float | None = None,
    spacing: float = .75, tolerance: float = .1,
    include_canopies: bool = True, include_structural: bool = False, top: int = 15,
) -> str:
    """Landscape penetration, base/placed-foliage collision candidates and canopy warnings.

    Read-only; metres. Pass live actor addresses or four XY region bounds. By
    default skips plants, rocks, ground tiles and structural kit parts. Terrain
    samples use subject collision undersides when available, bounds otherwise.
    Foliage physics tests subject OBB vs base/placed plant collision; not exact mesh-vs-mesh.
    Optional canopy checks use visual per-instance / placed-plant AABBs, so leaves
    without collision are covered conservatively. Only loaded plants taller than 2 m.
    Full JSON report under /tmp/uncrashed-audit/; caps and unknowns are reported.
    """
    try:
        region = environment.validate_region(x0, y0, x1, y1)
        return _dump(environment.audit(_catalog(), 'environment', actors=actors, region=region,
                                      spacing=spacing, tolerance=tolerance,
                                      include_canopies=include_canopies, include_structural=include_structural, top=top))
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def sample_terrain(points: list[list[float]], z_from: float = 1000, z_to: float = -2000) -> str:
    """Sample bare landscape at up to 256 [x, y] positions (metres).

    Ignores trees and placed structures, unlike trace/drop. Missing landscape returns z=null.
    Use a footprint grid to choose foundation elevations; does not move any objects.
    Requires a landscape base, not BaseBoard. Samples retain input order.
    """
    try:
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
        return _dump({"samples": samples})
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def agent_log(lines: int = 80) -> str:
    """Tail of the in-game agent's log (load, unload, timing, caught faults)."""
    try:
        return agent.log_tail(lines)
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def agent_unload() -> str:
    """Restore managed masks/holds and unload DLL; next call injects rebuilt DLL.

    Restoration conflicts refuse unload to retain original foliage snapshots.
    """
    try:
        return _dump(agent.unload())
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def inspect_site(bounds_min: list[float], bounds_max: list[float], spacing: float = 5,
                 min_foliage_height: float = 2, top: int = 30) -> str:
    """Read-only pre-placement terrain/foliage/placed-object inventory in XYZ box, metres.

    Bare-ground grid includes center axes, capped at 225 points; effective spacing
    reported. Foliage visual AABBs include crowns, even without leaf collision.
    Set min_foliage_height=0 for groundcover too. Loaded geometry only, not complete
    base-rock obstacle inventory. Full JSON on disk; top limits returned objects.
    """
    try:
        return _dump(managed.inspect_site(bounds_min,bounds_max,spacing,min_foliage_height,top))
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def clearing_zones(action: str = 'list', name: str = '', center: list[float] | None = None,
                   size: list[float] | None = None, yaw: float = 0, preview: bool = True,
                   allow_partial: bool = False) -> str:
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
    try:
        return _dump(managed.clearing_zones(action,name,center,size,yaw,preview,allow_partial))
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def list_groups(prefix: str = '', limit: int = 50) -> str:
    """Live saved editor group tags/counts, optionally name prefix. Pivot bounds, not mesh bounds."""
    try:
        return _dump(groups.list_groups(prefix,limit))
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def edit_group(group: str = '', action: str = 'transform', delta: list[float] | None = None,
               rotation: list[float] | None = None, scale: float = 1, pivot: list[float] | None = None,
               target_group: str = '', prefix: bool = False, actors: list[str] | None = None,
               preview: bool = True) -> str:
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
    try:
        return _dump(groups.edit_group(group,action,delta,rotation,scale,pivot,target_group,prefix,actors,preview))
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def undo_edit(transaction: str, preview: bool = True) -> str:
    """Preview/apply inverse of journaled edit. Explicit transaction id; saved map scoped.

    Refuses if affected groups changed since edit. Matches row/group/transform
    across handle changes on reload. Successful undo creates inverse transaction
    (undo that to redo). preview=False applies. Does not restore arbitrary runtime
    Blueprint/material state for deleted/respawned catalogue assets.
    """
    try:
        return _dump(groups.undo_edit(transaction,preview))
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def edit_history(limit: int = 20) -> str:
    """Recent current-map MCP transaction ids/statuses, including recovery-required edits."""
    try:
        return _dump(groups.edit_history(limit))
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def define_assembly(name: str = '', roles: dict[str,str] | None = None, preview: bool = True) -> str:
    """Inspect/save assembly intent: group -> foundation|structure|cantilever|decoration.

    Omit roles to inspect named assembly, or omit name too to list definitions.

    At least one explicit foundation group. Use edit_group assign to split footings
    from beams when necessary. Preview by default; preview=False writes map sidecar,
    not geometry. Group tags/roles survive save/reload without actor-handle IDs.
    """
    try:
        return _dump(assemblies.define_assembly(name,roles,preview))
    except Exception as exc:
        return _err(exc)


@mcp.tool()
def check_assembly(name: str, tolerance: float = .05, foundation_tolerance: float = .3,
                   min_anchor_coverage: float = .5, top: int = 20) -> str:
    """Read-only intent-aware support connectivity, NOT structural certification.

    Explicit foundation roots need measured landscape/base contact, not z=0.
    Reciprocal OBB-vs-collision hits build graph. Cantilevers need anchor path,
    not full footprint coverage; plants/decorations cannot support frame. Reports
    disconnected members, unknown anchors and conservative-contact limitations.
    Metres; loaded geometry only. Still run check_environment for terrain/trees.
    """
    try:
        return _dump(assemblies.check_assembly(_catalog(),name,tolerance,foundation_tolerance,min_anchor_coverage,top))
    except Exception as exc:
        return _err(exc)


def main() -> None:
    mcp.run(transport="stdio")
