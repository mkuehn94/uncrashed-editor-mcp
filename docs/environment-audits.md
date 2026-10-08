# Bounds, foundations and environment audits

All four tools are read-only. They do not move objects, trim foliage, change the
camera, clear the map or save it. MCP inputs / results use **metres** and degrees;
native commands use **centimetres**. Restart the Python MCP process after updating
so its tool list contains these additions. Rebuild / hot-reload the DLL too when
installing the source on another machine.

## `get_object_bounds(actor)`

`actor` is a live hexadecimal address from `agent_exec("actors ...")`, not a row
ID. Addresses become invalid after loading a map or restarting the game.

Returns:

- World-space AABB: `min`, `max`, `center`, `size`.
- Optional OBB: world `center`, scaled `half_extents`, rotation in degrees.
- Actor transform: position, rotation and scale.

AABB uses `GetActorBounds` for owned components, including non-colliding ones;
child actors are **not** included. OBB is available only for a static-mesh root.
A bounding box encloses empty space: it is not an exact mesh or a foundation
outline. A crane's box includes its jib; a house's box can include roof eaves.

Native: `bounds <actor>` (the original AABB line is preserved; transform and OBB
lines follow it).

## `get_world_bounds(scope="placed")`

| Scope | Meaning |
| --- | --- |
| `placed` | Aggregate bounds of loaded editor objects and race gates. |
| `terrain` | Aggregate bounds of loaded `LandscapeProxy` actors. |
| `playable` | Aggregate bounds of actors explicitly tagged `PlayableBounds`. |

Includes loaded streaming levels, not only the persistent level. Missing scope
geometry returns `available=false`, `aabb=null`. Enumeration limits are reported
as `truncated=true`.

**Terrain extent is not a playable boundary.** The tool does not infer a legal
flight area from landscape size, arbitrary blocking volumes, sky meshes or a
hard-coded +/-3 km square. If no authoritative `PlayableBounds` tag exists,
playable bounds remain unknown. Even a tagged aggregate AABB is not an exact
boundary shape / containment test.

Native: `worldbounds placed|terrain|playable`.

## `check_foundations(...)`

Parameters:

- `actors`: optional list of 1–256 live actor addresses. Explicit addresses
  override default category selection.
- `x0, y0, x1, y1`: optional region; supply all four. Selection uses overlapping
  world bounds, not just actor pivots. With both actors and region, their
  intersection is checked.
- `spacing=0.75`: requested sample spacing.
- `tolerance=0.3`: allowable base-plane / surface gap.
- `max_drop=50`: downward search distance.
- `include_structural=false`: by default checks buildings, containers, vehicles,
  big props. Enable to also check kit parts and small props. Plants, ground and
  balloons remain excluded from automatic selection.
- `top=15`: brief findings limit (1–60).

Samples a base plane under the bounds footprint. Upright static meshes use their
oriented footprint, so yaw is respected. Tilted meshes use the projected world
AABB and produce an explicit approximation warning. Blueprint / multi-component
actors fall back to their AABB. Grid edges are inset 1% to avoid numerical
edge-only hits. Both centre axes are always sampled, including on tiny round or
bevelled props. Grid intervals are capped at 32 per axis (1,089 points maximum);
`effective_spacing_m` makes coarsening visible.

Downward rays query actual support collision, ignoring base foliage and
catalogued placed plants. No fixed z=0 ground assumption. Results include:

- `support_coverage`, supported / unsupported / penetrating sample counts;
- four unsupported-corner flags (local footprint directions, not compass labels);
- `max_known_gap_m`, `missing` rays and base-plane penetration measurements;
- a few example coordinates, plus a full per-object JSON report.

A missing ray is **unknown beyond the search distance**, not a zero gap.
Penetration measurements from support rays can be clipped by their start height;
use `check_environment` for terrain penetration depth.

This is deliberately a **conservative footprint diagnostic, not structural
certification**. Roof overhangs, circular bases, crane jibs, stilted buildings,
bridges and engineered cantilevers can legitimately have unsupported bounds.
Review warnings against actual load-bearing geometry. A high support ratio does
not prove a safe structure, and sampling can miss narrow features.

Live validation on the overhanging Cloudreach house measured approximately 72%
support, two unsupported corners and a 6.7 m maximum known ground gap. No object
was moved during validation.

## `check_environment(...)`

Shares `actors`, XY region, `spacing`, `include_structural` and `top` selection
controls. Default selection omits plants, rocks, ground, attached lights / signs,
and structural kit parts. Explicit actor lists can target any placed object.

Additional controls: `tolerance=0.1` for terrain penetration,
`include_canopies=true` for conservative visual foliage overlap warnings.

Three distinct findings:

1. **`terrain_penetration`**: sample collision undersides against bare landscape
   heights. Traces ignore foliage / placed geometry when finding landscape.
   No underside collision at a sample is not evidence of penetration; actors
   without a mesh-root collision query fall back to bounds, explicitly labelled.
2. **`foliage_collision_candidate`**: subject OBB (AABB fallback) overlaps actual
   base / placed plant collision. Returns foliage actor, component, instance
   index and source. This is **box-versus-collision evidence**, not exact
   mesh-versus-mesh intersection; hollow subject interiors can cause candidates.
3. **`canopy_bounds_overlap_warning`**: visual AABB overlap with individual loaded
   foliage instances or placed plants taller than 2 m. Includes crowns / leaves
   without physics collision. Sparse crowns, rotated boxes and empty corners can
   create false positives. `include_canopies=false` disables these warnings.

Instanced foliage retains component + instance index instead of collapsing every
tree into its shared `InstancedFoliageActor`. Placed plants use instance `-1` for
visual candidates. Stored instances in loaded components are enumerated; renderer
density / visibility culling and material alpha are not evaluated. Unloaded
geometry is not measured. Foliage snapshot cap: 30,000 tall instances / plants.
Physical-hit and detail limits are reported; a truncated physics hit count is a
lower bound, not an exact total.

`terrain_available=false` means no loaded landscape, not a flat z=0 terrain
measurement. Use foundation support checks for `BaseBoard` floor geometry.
Objects with an available landscape but no usable terrain samples are reported
as `unknown`, never silently declared clear.

## Results and workflow

Brief responses contain counts, `sampling_complete`, truncation / unknown flags,
limited findings and a report path. Complete per-object summaries and up to 12
example points / foliage details per object are saved to:

- `/tmp/uncrashed-audit/foundation-report.json`
- `/tmp/uncrashed-audit/environment-report.json`

Each subsequent audit of the same kind replaces its report. This is not a dense
per-sample point-cloud export. No `no_findings` status proves absence of all
geometric or structural problems.

Recommended order:

1. Read `mapinfo`, `get_world_bounds("terrain")`, `sample_terrain(...)`.
2. Plan individual foundations using terrain samples and rotated object bounds.
3. Run `check_collisions` for placed-object clashes / support connectivity.
4. Run `check_foundations` for partially unsupported footprints.
5. Run `check_environment` for landscape / foliage conflicts.
6. Inspect warnings visually, then fix and repeat. Save only after review.

`check_collisions` now uses live placed-object extents by default. Negative world
Z no longer implies grounding, base foliage contacts are not ground seeds, and
catalogued plants do not propagate structural support in its connectivity graph.
Scan cells share inclusive boundaries (duplicate actor IDs are merged), avoiding
thin unscanned strips between cells. Missing scan files are errors, not a clean
empty result. Its original blueprint-building box fallback remains approximate. The older raw
`floating` command remains a legacy flat-board bounds heuristic; use these audits
for non-flat terrain.

## Native implementation

`native/environment.h` is included by `native/agent.c` and built into the same DLL.
It uses reflected `UWorld.Levels`, `GetActorBounds`, mesh bounds, per-instance world
transforms and Kismet collision queries. No landscape / actor addresses are
hard-coded. Native additions:

```text
worldbounds placed|terrain|playable
inventory <output_file>
audit <request_file> <output_file> foundation|environment <spacing_cm> <tolerance_cm> <max_drop_cm> <canopies_0_or_1>
```

Inventory / audit file paths use the Wine namespace, e.g. `Z:\tmp\...`. The MCP
wrapper creates temporary requests and validates actor addresses and numeric
inputs before querying the game. Requests contain `P <plant_actor_hex>` exclusions
and `A <subject_actor_hex>` subjects. A fresh inventory / foliage snapshot is used
for each audit; no actor-address cache survives between audits.
