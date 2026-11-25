# Asking a 3D scan: the `scan_*` commands

A builder agent turns a textured photogrammetry OBJ (Google Earth export, drone survey, ...) into placed catalogue
objects. The `scan_*` commands answer precise spatial questions about the scan without the game: where the ground
is, what stands where, how tall, which way it faces, whether a line is clear. They never place anything:
`scan_match` / `scan_plan` turn objects into catalogue rows and a reviewable command file, you run it with the usual
`batch` workflow, then check with `capture_view` and `scan_compare`.

`./uc help scan` is the one-screen version of this page; `./uc help scan_<name>` gives every parameter.

## Privacy

Scans are often personal-use-only data. No image or raster of the scan is written to disk or sent anywhere:
rasters (DSM, ground model, classes, ortho), the vision tiles and facade renders, and every image `scan_view` /
`scan_facade` / `scan_sheet` renders live in `/dev/shm/uncrashed-scan/` (RAM; worker tiles are deleted after each
run, each survey keeps its newest 50 images, `scan_status --drop_ram KEY|all` frees the rest). The detectors run on
this machine's GPU. Disk under `~/.cache/uncrashed-mcp/scan/<key>/` holds JSON: geometry facts (objects,
footprints, heights, walls, window rows) plus one median texture colour per object (`rgb`), which classes,
confidences and catalogue colour words are derived from. `scan_plan` files hold coordinates and catalogue rows
(still derived from the scan: keep them local; the default `./scan-plans/` is git-ignored). Look at the images
locally; never upload, attach or commit them.

Network: the OSM fetch and the other data providers (section 1b) send the survey's bounding box (four numbers), tile
names or byte ranges of public files, and nothing else; the first vision run downloads missing model weights. The
provider caches (open data, not scan pixels) sit beside the survey in `providers/<name>/`. Once the weights are on disk the worker runs with `YOLO_OFFLINE=1` and
`HF_HUB_OFFLINE=1` (no ultralytics usage events, no hub checks). `UNCRASHED_OSM=0` turns OSM off for every survey.

## 1. Survey

```
./uc scan_survey /data/city/scene.obj                      # region from metadata.json (area radius), else the export
./uc scan_survey scene.obj --center 120,-40 --radius_m 150  # export metres (east, south) around the origin
```

What happens (about 6 s for 500 x 500 m at 0.25 m on 32 cores, plus the OSM fetch and, with vision, about 90 s on an
RTX 3090: YOLO 5 s, Grounding DINO 75 s for 90 tiles and ~930 walls, SAM 1.5 s):

1. `native/ucmesh` indexes the OBJ once (cached), `native/scanraster` rasterises it top-down: highest surface
   (DSM), lowest surface, normals and texture colour at `--res_m` (0.25 m), plus an ortho image at half that.
   Coarse octree tiles covered by finer ones are skipped, as in `reference_load`.
2. A progressive morphological filter on a 1 m grid gives the ground model (DTM); height above ground = nDSM.
3. Cells are classified: building, vegetation (canopy), low object (0.6 to 2.2 m), and ground as paved, grass,
   water, bare, other. Raised cells the shape filters cut off (roof edges, wall tops) take the class of the nearest
   building / canopy cell and join a building within 1 m: they are never ground.
4. Objects: buildings (split at height steps, footprint polygon, parts, roof type, eave and ridge heights), trees
   (one per crown top), cars (single blobs, split at height dips, or tiled at car pitch), low structures, small
   objects, shrubs, bridges.
   - **Parts:** a footprint is cut along its box axes where both sides fill their own boxes far better (an L into
     two wings, a block into its wings; at least 15 m2 and 2.5 m wide each, one connected piece, no overlap). A
     footprint over 300 m2 filling under 60 % of its box that no straight cut helps (wings meeting at an angle: a V,
     a bent block, bars round a yard) is split into wings by the local long axis of the footprint (principal axis
     within 40 m; cells within 15 degrees of a frequent axis form a wing, corners join the nearest wing). Each
     part has its own footprint, eave / ridge heights and roof; `scan_building PART_ID` answers for the part.
   - **Eaves:** `eave_m` is the 30th percentile of the height above ground in the outer 1 m of the roof (where it
     meets the walls; `eave_from: edge`). Surveys before this stored the 15th percentile of the whole roof, about a
     metre high on tall gabled roofs with dormers; `scan_building` and `scan_plan` hints measure those again from the
     building's cells.
   - **Roofs:** the normals give flat / gabled / hipped / mono_pitch / complex; planes fitted to the roof heights
     (0.75 m in from the edge) add `roof_shape`: form, slope per plane in percent, pitch, and the ridge or valley line
     with its height and the edge heights above ground. A roof the normals call flat becomes `gabled` or `valley`
     when two planes meet at a line at least 0.2 m above / below the edges and slope 2 % or more (a 3.5 % valley
     hall, a 0.55 m gable on a narrow house); one plane is `mono_pitch` from 8 %, below that the roof stays `flat`
     and `roof_shape` keeps the fall (drained flat roofs and garages fall 3 to 6 %).
   - **Trees on roofs:** a crown candidate inside a mapped footprint, or a small or band-shaped one touching a roof
     of its own height (green roofs, planted roof edges, facade tops), is roof: its cells join the building. A crown
     partly inside a footprint or against a roof of its height stays with conf - 0.25 and a note (`flag`: against a
     roof of its height). The reply's `rules` counts both.
5. OSM (default on when `metadata.json` has the origin lat/lon and a server answers within a 3 s connect probe;
   `--no-osm` turns it off; a dead network costs about 3 s per server and the skip reason says how long):
   building footprints are aligned to the roofs (shift search up to 4 m), split merged roofs (terraces, blocks) and
   claim canopy-covered roof cells; roads, parking, paths and water become ground classes (roads first at their
   `width` tag, else 3.2 m per `lanes`, else a nominal width per highway type; then widened to the kerbs: profiles
   every 2 m look up to 8 m beyond that for a 6..35 cm step up within 0.75 m, cars and crowns skipped, and a side where
   at least 40 % of the profiles agree takes the median kerb distance, parking lanes included; the band beyond a kerb
   up to the next building or 4.5 m is sidewalk, class `path`; `osm.kerbs` in the survey reply counts the sides with
   a kerb); carport-like blobs inside
   mapped buildings are not cars; tree nodes confirm crowns. A footprint whose cells (foliage left out) stand less
   than 60 % above 1.8 m disagrees with the heights (moved, demolished, underground, cut across a roof): it is not
   used for splitting, claiming or walls, the building under it gets `osm_conflict` (cover, best local shift) and its
   footprint, parts and walls from the heights; the reply lists the disagreeing footprints. Answer cached as `osm.json`; data
   © OpenStreetMap contributors, ODbL.

6. Data providers (`--providers auto`, the default; `none`; or names, `auto,-overture` drops one): see 1b.
7. Vision (default on when `.venv-vision` exists; `--no-vision` turns it off) and facades (default on with vision;
   `--no-facades`): see below.
8. Street views (default on with vision and OSM; `--no-street_views`): see below.

Every object has `conf` (0..0.97) and `src`: `mesh`, `mesh+osm` (OSM agreed or shaped it), `mesh+vision` (a detector
confirmed or corrected it) or `vision` (found by a detector only); `models` lists detector scores such as
`yolo11l-obb:0.83`. The reply lists per class how many are high (>= 0.75), mid (>= 0.5) and low and how many carry
OSM / vision evidence, the vision report, what was skipped and why, and timings. A survey of the same OBJ, region and parameters is reused; `--refresh` redoes it
(and retries a failed OSM fetch). The newest survey is *current*; `--survey KEY` picks another; `scan_status` lists
them, whether their rasters are in RAM and how many MB (rebuilt automatically after a reboot, or when a file is
incomplete or truncated), the frame answers use and the newest images. When the OBJ changes after a survey (a
re-export), every query on that survey stops with `OBJ changed since survey KEY: run scan_survey OBJ again`
(`stale` in `scan_status`): old objects never mix with a raster of the new mesh.

### 1b. Data providers

`./uc scan_sources --latlon 49.4539,11.0775` says offline which providers cover a place; `./uc scan_sources` after a
survey lists what was used, why not, licences, the attribution lines to credit, how each source fused and the
fusion report. Providers are chosen by bbox coverage, never by country; one module per data set under
`scan/providers/` (interface: `coverage(bbox)`, `fetch(bbox, folder)`, `load(folder)`, licence, attribution, fuses).

| provider | data | licence | fuses |
| --- | --- | --- | --- |
| `osm` | OpenStreetMap (Overpass / map API) | ODbL | footprints, ground classes, tree nodes, street lines for the street views |
| `overture` | Overture Maps buildings (PMTiles, zoom 14, range requests) | ODbL | evidence only: footprint IoU, height, floors, roof shape |
| `glo30` | Copernicus DEM GLO-30 (DSM, 30 m) | Copernicus DEM licence | cross-check of the height datum and coarse relief |
| `worldcover` | ESA WorldCover 2021 (10 m) | CC BY 4.0 | class shares vs the scan; tree evidence |
| `by_dgm1` | Bavaria DGM1, 1 m terrain (airborne laser) | CC BY 4.0 | replaces the ground model |
| `by_lod2` | Bavaria LoD2 CityGML | CC BY 4.0 | overrides building footprints, heights, roof type and planes |
| `nl_3dbag` | 3D BAG, the Netherlands (LoD 2.2, bbox API) | CC BY 4.0 | as `by_lod2` |

The Bavarian licences were checked in the portal's own product list (geodaten.bayern.de/opengeodata,
`json/opengeodata_datensaetze.json`, 2026-11-25): DGM1, LoD2, DOM20, DOP20 and the laser points are CC BY 4.0,
ETRS89 / UTM 32N, DHHN2016 heights; tiles `download1.bayernwolke.de/a/dgm/dgm1/<E>_<N>.tif` (1 km) and
`.../lod2/citygml/<E>_<N>.gml` (2 km, even km; about 90 MB in a city: parsed, then only the bbox's buildings are
kept). A regional provider first tests its published extent offline, then confirms the tile exists (HEAD request):
no tile = not covered. Any failure of a provider (no answer, an HTML error page instead of CityGML, a cut-off or
garbled GeoTIFF / PMTiles) is recorded in its `why` and the survey goes on; it is raised only when that provider was
named in `--providers` (or OSM asked for with `--osm`). Mapillary was checked and declined (an access token is needed, the images are CC BY-SA 4.0
and the terms do not settle whether our own detections on them escape share-alike); the survey renders its own
street views instead. Not yet implemented: Bavarian DOM20 (open, but distributed through the polygon download
service), US 3DEP, NL AHN.

How the data fuses (`providers/fuse.py`):

- **Terrain** replaces the filtered ground model where it has data. The vertical offset (datum, geoid) is fitted as
  the median difference on open ground and reported with the residual spread before replacement, so the scan's own
  ground model is validated, not silently overwritten.
- **Buildings** are aligned to the scan roofs (shift search, as OSM) and paired with mesh buildings by overlap. An
  override provider (LoD2, 3D BAG) replaces footprint (`footprint_src`: the cadastral wall outline, not the eaves),
  height, eave, roof type, `roof_slope_deg`, `ridge_yaw_deg` and `roof_planes` (slope, `faces_deg`, z range, 3D ring
  per plane); the mesh values stay under `mesh`. Before that its roof planes are checked against the scan surface
  (`roof_fit`: share of the footprint within 1 m): a roof the scan does not show (rebuilt, demolished, misaligned),
  a footprint mostly beyond the scan, or footprints that overlap the mesh building by an IoU under 0.25 keep the
  mesh geometry and are flagged instead. A mesh building holding several provider buildings is split (terraces);
  a mesh piece mostly inside a provider building becomes its `building_part`; provider buildings hidden under
  canopy are added; mesh buildings the provider does not know lose conf and are flagged. Trees, cars and other small
  objects inside a provider footprint are flagged (roof greenery, carports) and their conf capped.
- **Evidence**: every object a source has something to say about lists `evidence` [{src, verdict agree | conflict |
  none, detail}], `conflicts` (count) and `flag_sources`; `flag` in query rows says `sources disagree (by_lod2)`.

Measured on a 150 m test export in Bavaria (fetch 50 s for OSM over Overpass, LoD2 tile
10 s, the rest about 3 s; fusion 2 s):

- DGM1 vs the scan's own ground model on open ground: offset -0.79 m, residual median 0.04 m, 90th percentile
  0.15 m, 2.4 % over 0.5 m. Under objects the ground moved by up to 0.5 m (90th percentile). GLO-30 agrees on the
  datum (+0.78 m, spread 2.0 m median: a 30 m surface model).
- LoD2: 183 buildings in the box, 43 mostly beyond the scan's round coverage. 67 mesh buildings overridden, 18 split
  (semi-detached and terraced houses: by eye correct in all four sampled crops), 50 added (split pieces, canopy),
  15 mesh fragments made parts, 7 kept against the provider, 26 mesh buildings flagged. Heights of the overridden
  buildings: LoD2 minus mesh median +0.10 m, 97 % within 1 m. Roof planes fit the scan surface on a median 83 % of
  their footprint. Roof type changed on 19 of 67 (21 in the run whose 16 crops were looked at): LoD2 clearly right on about 10 (e.g. a flat
  green roof the mesh called gabled, hipped houses called gabled), 4 undecidable at 0.07 m/px, 2 were buildings
  beyond the scan edge (since then never adopted).
- Flags checked by eye: 6 of 8 trees inside LoD2 footprints were roof greenery or roof edges, 1-2 real crowns over a
  shed; of 8 sampled mesh-only buildings, 6 were tree crowns or hedges, 2 real changes (a building site, a new
  building).
- Overture: 104 of 149 mesh buildings paired; many Overture heights here are 3-5 m below LoD2 (estimates), so its
  height conflicts mostly mean Overture is low.

### Street views

Virtual cameras every 10 m along the OSM streets and paths (deduplicated), eye 1.6 m above the ground (pitch 0) and
8 m (pitch -25), looking to both sides, 90 deg, 640 x 480, rendered from the textured mesh by `native/scanraster.cpp`
(`sr_views`: perspective z-buffer, near-plane clipping, perspective-correct texture). An eye under a crown or roof
moves sideways up to 3 m or is dropped; views whose median depth is under 4 m (a wall in the face) are not sent.
Grounding DINO runs two prompt groups (door, window | traffic sign, street lamp, bench, trash can, bollard, bicycle,
fence: in one joint prompt every box becomes a window). A detection is lifted to 3D by the render's depth buffer
(the first mesh surface along each pixel's ray): 25th percentile depth in the box's central third, the ray through
the box centre gives the position, the rays through top and bottom the height. Kept within 30 m, not cut by the
image edge, with a height in the class range, standing within 1.2 m of the ground (street furniture), doors within
2 m of a building outline. Points of different camera positions within a class radius fuse into one object (`src:
street`, `views`, `spread_m`, doors with `building` and `faces_deg`); one camera position needs a score of 0.5 and
gets conf <= 0.45. A geometry small object / shrub / low structure at the spot takes the class (`was`). Windows are
counted per building (`street_windows`). Renders exist in RAM only while the detector runs. A reused survey adds
the street views when they never ran (surveys made before them), and runs them again whenever the vision stage
rebuilt `objects.json` (e.g. facades switched on after `--no-facades`), so street objects are never dropped while
`meta.street` still reports them.

Measured on the same export: 1,308 views from 327 positions (render 4 s), 1,018 sent, DINO 175 s on the RTX 3090;
21,029 detections, of them 20,734 windows and 240 doors, but only 22 fence, 16 bench, 12 lamp and 5 sign boxes. Added:
13 doors (about 9 plausible in check renders: entrances and garage doors) and 1 pole (correct; 12.6 m, a lamp or
utility pole). Street furniture is mostly not in a Google Earth mesh (melted into the ground or walls), so recall
for signs, bins, bollards and benches is near zero on such scans; the stage pays off on finer meshes (drone or
terrestrial surveys).

### Vision and facades

The detectors run in `.venv-vision` (torch, ultralytics, transformers) as one subprocess per survey
(`scan/vision_worker.py`, JSON in and out), on the RTX 3090 when present (else the CUDA device with the most free
memory; `UNCRASHED_VISION_DEVICE=cuda:N` overrides). Each model is loaded once and freed before the next (peak about
13 GB). Weights live in `~/.cache/uncrashed-mcp/models` and are downloaded on first use.

- **Aerial:** the 0.125 m ortho is cut into 512 px tiles with 96 px overlap. YOLO11-OBB (DOTA; upsampled to
  1024 px) gives oriented vehicle boxes, plus pools, courts, fields, tanks, boats from 0.25 m tiles. Grounding DINO
  takes prompts (car, van, truck, bus, tree, solar panel, swimming pool, playground); SAM 2 outlines DINO vehicles
  and pools that no oriented box covers. Fusion: boxes from overlapping tiles merge; vehicles on roofs or over
  holes in the scan are dropped. A geometry car with a matching vehicle takes the detector's outline and yaw. A
  geometry car with no vehicle in the image becomes `low_structure` / `small_object` (`was: car`): skips, sheds,
  carports and water noise. Cars on the scan edge are the exception and are kept with lower conf. A car-class box
  whose visible heights are uneven (90th minus 30th percentile over 1.5 m, crown cells left out) and that is under 3 m
  long, vision-only or partly taller than a car is a stack, pallets, barrels or furniture: `small_object`, note
  `not a car: ...` (`cars_rejected_clutter` in the report). A vehicle on a
  small geometry blob relabels it; any other vehicle is added. Vehicles over 6.2 m are `truck`. DINO tree boxes
  raise crown conf. Solar panels get conf <= 0.5 (the boxes cover parts of arrays).
- **Facades:** walls come from the OSM footprint when mapped (wall lines), else the roof outline (cleaned to long
  edges). When one mapped building is split into several mesh buildings, each part keeps only the OSM walls that
  run along its own roof outline (older surveys stored every wall under each part; queries filter them now). Walls whose front is another building (party walls) are skipped. `native/scanraster` renders each wall
  straight on (0.08 m/px, the surface nearest to the viewer inside a slab 2.5 m behind to 1.6 m in front of the wall
  line, roofs and ground left out). A wall counts as a party wall when, at most of 9 positions along it, the first
cell 1 to 3 m out that is not the building's own (its eaves, dormers and balconies reach over an OSM wall line) is a
roof of another building; surveys before this rule could call a street facade a party wall because of the building's
own eaves: `scan_building` re-checks stored party walls and surveys the false ones now (vision available; else they
show `not_surveyed`). Grounding DINO marks window, door, balcony and garage door. Group boxes (twin
  windows, whole storeys) and duplicates are removed, and boxes over empty render are dropped. Windows are then
  clustered into rows with sill / head heights above ground, columns, spacing and storeys (raised ground floors
  count; basement light wells do not), and each wall's median render colour (`tone`). Stored as `facades.json`
  beside the survey (no pixels). Buildings under 25 m2 or 3 m are left out of the survey; `scan_building ID`
  surveys such a building's walls when asked (vision available, about 5 s) and stores them.

A survey made without vision gets the vision stage the next time `scan_survey` runs with vision available. The
detections of the last run stay in RAM, and an identical job reuses them, so re-fusing costs seconds.

## 2. Place

The scan has its own frame: export metres east, south, and source height up. To speak map coordinates, give the
same placement `reference_load` used:

```
./uc scan_place --at 0,0,0 --yaw 0 --datum_m 54.2   # at: map metres of the export origin; datum_m: source height at z
./uc scan_place --from_reference                    # copy the last reference_load of this OBJ (needs the game)
./uc scan_place --at 0,0,1                          # move only: yaw and datum_m stay those of the current placement
```

The first placement needs `--datum_m`; later ones keep yaw and datum when left out (`kept` says so). The placement
file changes at once: the reply lists the `previous` placement and a `restore` line (`./uc scan_place --at ... --yaw
... --datum_m ...`) that puts it back; `--clear` gives one too.

The reply's `site_ground` is the ground at the region centre in map z and relative to the datum (median of the
ground model within 30 m). `reference_load` reports the same (`site_ground`, from the 10th percentile of the vertex
heights within 30 m): an auto datum is the 2nd percentile of the whole region (a pit, a river bank), so the site can
sit a metre or more above map z; its note gives the `--at` z that puts the site ground where you want it.

`scan_status` prints the frame in use (`map: Unreal metres via placement at=... (reference_load)`): x,y,z of every
query and answer are then map coordinates, so "map 0,0" is the map origin, not the scan's.

From then on every query takes and returns Unreal metres (x east, y south, z up; yaw 0 = +X, 90 = +Y) and says
`frame: map`; `--frame scan` still answers in export metres. Placement is stored per OBJ, so all surveys of it share
it. `scan_place --clear` removes it.

## 3. Look

```
./uc scan_view --center -165,20 --size_m 60                 # ortho with metre grid, outlines and ids
./uc scan_view --box 0,0,120,80 --layer classes              # surface classes; --layer ndsm = height above ground
./uc scan_view --center 50,-110 --size_m 70 --classes car,building
./uc scan_sheet --classes car --box 60,-230,160,-150 --limit 12   # one labelled crop per object, one PNG
./uc scan_sheet 12,-40 60,-10 --size_m 50 --layer ndsm            # one crop per point
```

Open the PNG (path in the reply) and read object ids off it. Default overlays skip `building_part` and
`small_object`; `--classes` chooses. Trees are circles of their crown diameter, buildings their footprint, everything
else its oriented box. Labels are dropped for trees when more than 40 are in view.

Look before you trust a count: crops of 50 to 80 m show cars and doors-scale detail; 200 to 500 m show layout.

## 4. Query

| question | command |
| --- | --- |
| what is in this area? | `scan_objects --classes car,tree --box x0,y0,x1,y1` / `--near x,y --radius_m R` (`--min_conf 0.75`; no hit: `summary` names the nearest match and its distance) |
| the N nearest, however far | `scan_objects --classes car --near x,y --nearest 3` (with `dist_m`) |
| tallest, biggest, most storeys / windows | `scan_objects --classes building --sort height` / `area` / `storeys` / `windows` (building rows carry area_m2, storeys, windows, roof) |
| what is right here? | `scan_at x,y --radius_m 10` (surface shares + objects by distance, 0 = inside) |
| ground / roof height | `scan_height x,y x,y ...` (ground_z, top_z, height_m, under_m, surface, object id; under a roof or crown `ground_cls`, the nearest visible ground; up to 1024 points) |
| clearance at a height above ground | `scan_height x,y,z ... --agl` (z, agl_m, clearance_m above the top) |
| slope of a road, a wall, a bridge deck | `scan_profile x,y x,y --step_m 1` (`slope_pct` per segment between open-ground samples only; steepest and mean) |
| a flight line along a profile | `scan_profile x,y,z x,y,z [--agl]` (line z and `clearance_m` per sample, minimum) |
| many heights at once | `scan_grid x0,y0,x1,y1 --res_m 1` (dtm, top, height, class, object id per cell: one `.npz` + `.json` header in RAM; up to 1,000,000 cells) |
| ground cover for paving / grass / water | `scan_ground x0,y0,x1,y1` (shares + simplified outlines) |
| one building in detail | `scan_building ID` (footprint, parts, eave/ridge, roof and `roof_shape` planes / lines, colours, OSM tags or `osm_conflict`, walls with storeys, window rows, doors, tone); a part id answers for that part |
| what does this wall look like? | `scan_facade ID [--side N]` (straight-on render with windows / doors / rows drawn; PNG in RAM) |
| is a flight line above the scan? | `scan_free x,y,z x,y,z ... --radius_m 0.5` (`--agl`: z above ground, following it) |

All heights are metres; `ground_z` / `top_z` are absolute in the result frame, `height_m` / `eave_m` / `ridge_m`
are above ground. `yaw_deg` is the long axis (0..180); `faces_deg` of a facade is its outward direction (0..360).
`surface` is the class of the top surface: the ground cover under a roof or crown is not seen.

Before building from a row, read its `flag` (flat or short car, `not a car` stack, weak vision-only detection,
slender "tree" or one against a roof of its height, part of an OSM building split into several mesh buildings, a
footprint overlapping another building, an OSM footprint that disagrees with the heights) and a wall's `check`
(window rows below ground, with the head at or below the sill, or overlapping: look with `scan_facade`). Every
object row has `colour` (a plain word) and `rgb` (hex) of its median texture colour; `scan_building` adds the roof
colour and the facade tone. Shade tints photogrammetry blue: weak blues and mauves count as grey. `scan_view`
lists the class, conf and height of every id it draws; `--layer classes` and `ndsm` carry a legend in the image
and in the reply (`class_legend` with hex colours, `ndsm_legend`).
`scan_match` adds `warning` when the best row scores under 0.5 (trees often do: no catalogue tree fits well).

Shells: one call with several ids or points beats a loop (`scan_sheet --objects 12,40,77`, `scan_height a b c`,
`scan_grid` for areas). In zsh `for i in $ids` does not split a string into words; use an array
(`ids=(12 40 77); for i in $ids; do ./uc scan_building $i; done`) or `${=ids}`.

## 5. Match, plan, check

The scan becomes catalogue objects in three steps; nothing reaches the game until you run the plan file yourself.

```
./uc scan_match 997                                    # top rows for one object: score, errors, scale, spawn yaw
./uc scan_match --classes car,tree --box 60,-230,160,-150   # best row per object, rows used most per class
./uc scan_plan --box 60,-230,160,-150 --out plans/block1.txt # spawn + drop lines (+ plans/block1.json); not run
./uc batch plans/block1.txt                             # after reading it
./uc scan_compare --box 60,-230,160,-150               # placed objects vs scan objects (read-only), fix lines
./uc capture_topdown --box 60,-230,160,-150            # the build, straight down, on scan_view's pixel grid
./uc scan_view --compare /dev/shm/.../topdown-XXXX.png  # scan | build | edge overlay, deviations labelled
```

### Scan-based build loop

1. `scan_survey scene.obj`, then `scan_place` (placement in map metres); look at `scan_view` first.
2. `scan_plan --box ... --out plans/phaseN.txt`: read it. Phases are blocks of the box (ground, buildings, trees, cars), not the whole survey.
3. Build the phase with `batch`, then `save_live`. Save after every phase. Ground first: `scan_plan --classes
   ground,car,tree --box ...` (smooth fitted surfaces, never per-cell DTM heights; read `predicted_steps`: 0 seams over
   1 cm).
4. Compare, both ways: `scan_compare --box ...` (objects: offset, yaw modulo symmetry, size, missing/extra, height cells;
   `ground_steps`: seams over 1 cm between placed tiles must be 0 except kerbs, `far` tiles off the scan ground) and
   `capture_topdown --box ...` + `scan_view --compare IMG` (scan | build | edge overlay).
5. Give a cheap review agent only the compare sheets; it lists deviations (id, status, values), no fixes by eye.
6. Apply the `fix` lines (`transform ACTOR ...`), rebuild missing objects, save, compare again. Stop when nothing is over tolerance.
7. Foliage: size by visible size via `asset_profile` (crown to crown, height to height), never box to box. Yaw comes from the scan
   OBB modulo the row's `yaw_period_deg`; a 180 degree error on a symmetric row is not a deviation.

Remaining `size_off` on rows whose catalogue scale limits cannot match are catalogue limits, not errors: accept or pick another row.

Name groups so scan_compare can pair them: `scan.<class>.<id>` (what scan_plan writes), `<class>.<id>`, or for
buildings `b<id>` with any suffix (`b57nw`, `b57s`: several groups for one building are fine). Untagged building
pieces still pair by footprint.

**asset_profile ROW...** measures what a catalogue row looks like from above, the way the survey measures a scan
object: `visible_m` (top silhouette extents along the row's x / y and the 95th percentile top), `visible_frac` of
the bounding box, `long_axis_deg` (visible long axis in the row frame), `symmetry` / `yaw_period_deg` (round 0,
square 90, symmetric 180, none 360; front = +x by engine convention, not measured), foliage `crown_m` (equivalent
diameter of the silhouette cells with two or more leaf-card layers), `crown_frac`, `crown_base_m`, `panel` tilt and
facing for panel-like rows, footprint shape. Geometry: the cooked LOD0 from the installed game paks
(`meshextract.read_lod0`), else the simplified hull in `data/meshes.bin`, else (Blueprints) the box. Cached per
catalogue version in `~/.cache/uncrashed-mcp/asset-profiles/`. scan_match, scan_plan and scan_compare size and turn
vehicles, trees, shrubs, boats and props by these profiles (buildings and pools keep the box).

Why trees came out small in the first test build: its generator scaled each tree's bounding box to the scan's
`crown_m`. A game tree's box reaches its outermost leaf cards; the dense crown is 0.70 to 0.95 of it (Elm13 0.70,
Elm09 0.75, pine03 0.78, maple Tree A 0.88), and `crown_m` is an equivalent diameter (smaller than the crown's box).
Measured on the 2026-11-25 export (238 planned trees): that rule builds visible crowns at 0.78 of the scan crown
(median, 0.71 to 1.0); the previous scan_plan rule (box onto the scan's oriented box) 0.85; scaling crown to crown
1.00 (10th percentile 0.98). Trees also keep crown scale / height scale within 0.75..1.35 (`TREE_SPREAD`) and height
scale 0.4..2.5: wider squashing made 0.3-high willows stand in for wide low crowns; such crowns now report their
fit error instead. Alpha-masked leaves are not seen by geometry, so rendered crowns can still look sparser.


**scan_match** scores catalogue rows (the ones `search_assets` / `get_assets` show, hidden variants only with
`--include_hidden`) per class: cars and trucks against vehicle meshes (Gt86, Camaro, Kangoo, Old Van; bus stops,
spawners and wheels are excluded), boats, trees against foliage trees, shrubs against bushes, buildings against
building and house rows (arches, rigs, windmills, booths, roof pieces excluded), pools against water,
low structures / small objects against props, barriers and industrial rows. For each row it finds the yaw that lays
the row's long axis (buildings: its roof ridge) on the object, the scale within the class limits (vehicles 0.85..1.15
uniform, trees height and crown, buildings 0.5..2 per axis), and reports:

- `fit_err_pct`: size error left after that scale; `shape_err_pct`: proportion mismatch at one uniform scale;
  `raw_err_pct`: error at scale 1;
- `roof` of the row, read from its simplified mesh hull (`data/meshes.bin`: highest surface per 0.5 m cell, slope
  directions as in the survey). Blueprint rows have no hull: roof unknown, scored lower. A gabled row whose ridge
  cannot follow the scan ridge is marked `ridge turned`;
- `score` 0..1 combining these. Colour only counts for the few rows with a colour word in their name or material:
  the catalogue has no colours.

**Row choice by look and body.** Vehicles: a scan car 1.75 m or taller (or longer than 5.4 m) and every truck want
a van-bodied row (catalogue height 1.7 m or more: Kangoo, Old Van, trucks), lower cars a car-bodied row (Camaro, Gt86);
the other body scores 0.6 x (`notes`: `car body for a van`). Before: a 6.7 m lorry became a Camaro x1.3, a 2 m van a
Gt86. Trees: rows whose asset profile contradicts a tree are left out of the candidates (`rows_excluded` in the
scan_plan reply): no mesh geometry found (Maple08, renders as nothing) or foliage from the ground with no trunk and a
solid silhouette (crown base under 0.3 m, crown filling over 0.9 of the box: `yd1gfbjab`, renders as a rock).

**scan_plan** needs the placement (`scan_place`) and writes map coordinates in agent centimetres. Per object: a
comment line (class, id, conf, size, chosen row, fit, surface), `spawn ROW x y z 0 yaw 0 sx sy sz
group=scan.<class>.<id>` with the pivot computed from `origin_cm` / `extent_cm` so the mesh box centre sits on the
object centre and its bottom `--lift_m` (0.5) above the scan ground (with `ground` in the same plan: the planned
surface; base, centre, top and offset pivots all land), then `drop last`. Cars and trucks need visible ground around them (roof, deck, water: skipped with a reason).
Trees rotate through rows within 85 % of the best score and get a varied yaw; "trees" taller than 3 x a crown under
4 m are skipped (on the example scan nearly always roof greenery or a building edge; `scan_match ID` warns too).
Every skip is listed with its reason. `--near x,y --radius_m R` selects a circle instead of a box (object centres
within R). A weak fit (size off by more than 40 % after scaling, or a non-foliage best score under 0.5; foliage scores
low by nature) is held back: its spawn is written as `#?` lines under a `# WEAK FIT (reason)` comment and listed under
`review`, so it never reaches the batch unseen. Look at it (`scan_sheet X,Y`, `scan_match ID`), then uncomment, pick
another row or delete; `--weak keep` spawns weak fits with the comment, `--weak skip` leaves them out. On the
2026-11-25 export (40 m around the centre) this held back car 1008 (85 %: a red car under a tree, the crown above it
measured as its 7.3 m height) and car 704 (score 0.46).

**Ground** (`--classes ground[,car,tree...] --box x0,y0,x1,y1 [--flat_tol_m 0.3] [--ground_smooth_m 4]
[--ground_cell_m 2] [--max_tile_m 16] [--ground_yaw DEG]`): smooth fitted surfaces, never per-cell scan heights. A
photogrammetry / 1 m terrain ground model is noisy (5..15 cm), dented by cars and blurred at kerbs; a tile flat at
the height of its own cell turns a street into a staircase (iteration 2 of the live build: neighbour steps p90 12 cm
on asphalt, up to 74 cm on sidewalks). Instead (`scan/groundfit.py`, numpy only):

1. Material per lattice cell by surface class (rows below); classes that share a row are one material; pieces of a
   material under 12 m2 take the commonest material of the same level around them. Levels: carriageway (road,
   parking) and kerbside (everything else).
2. Islands of paved cells (touching at an edge or a corner). Per island one robust (Huber) fit to the open-ground
   samples (0.5 m apart; the 0.75 m strip at a level boundary, crowns, cars and objects left out): **one plane** when
   its robust residual (95th percentile of per-cell medians) is within `flat_tol_m` and no wall is inside; else a
   **thin-plate surface** on the lattice knots with bending length `ground_smooth_m`, stiffer on carriageway and in
   regions that alone fit a plane, twist (non-planar cells) ten times dearer. Neighbouring cells whose ground differs
   by over 0.35 m (a wall, an embankment) bend freely: a steep ramp tile, never a step or a 40 cm sag.
3. Kerb: the kerbside surface sits one kerb above the carriageway, measured across the level boundaries and snapped
   to 8..20 cm, or none under 4 cm.
4. Tiles: greedy rectangles of one material up to `max_tile_m` whose border knots lie within 4 mm of the tile plane
   (inner knots within 3 cm); a single cell whose corners leave one plane is split in four on its bilinear surface.
   Every tile corner is a knot of the one surface, so neighbours meet within a centimetre. `ground_yaw` default: the
   lattice along the street edges or the map axes, whichever needs fewer tiles.
5. Rotation: `_slab` turns the tile frame into an Unreal rotator; the tests read the spawn lines back through
   `geom.quat_rotator` (the engine's FRotator to quaternion) and fail on a sign error in pitch or roll.

Rows: road / parking asphalt (NewRow_182), paved and other ground Concrete Floor01 (NewRow_303, pivot at its top),
path pavement (NewRow_185), grass and bare dirt (NewRow_180; no lawn tile row: Grass Foliage NewRow_663 on top);
under crowns and parked cars the nearest open ground; cells under buildings and water are left out. Tile tops are on
the surface (the pivot offset of each row is taken out), 0.2 m thick; no `drop`; groups `scan.ground.<class>`.
Objects in the same plan (cars, trees, props) spawn `--lift_m` above the **planned** surface, then `drop last`.

The reply's `ground` gives tiles, islands (mode, residual, kerb, break cells), `residual` (|scan - planned top| on
open ground, carriageway and kerbside) and `predicted_steps` (largest step per neighbouring tile pair from the
written lines; same row = seams, between rows = kerbs). Larger `ground_smooth_m` is flatter with fewer tiles and a
larger residual; `flat_tol_m` is the residual a single plane may have.

Measured on the 2026-11-25 survey (1 m terrain model, no kerbs in it), 200 x 200 m box at the placement origin:
6,559 paved cells, 3,333 under buildings; 3,000 tiles (iteration 2: one per cell); predicted steps over 1 cm: 0 of
9,142 tile pairs, max 8 mm; residual carriageway median 4.6 cm / p90 12.9 cm, kerbside 4.6 / 19.7 cm (the terrain
bends more across gardens than a 4 m smoothing follows); kerbs found: none (measured 2.4 cm); 236 break cells along
an embankment; about 5 s for the whole plan including cars and trees. Look at it: a shaded relief of the ground model
next to the fitted surface shows smooth streets with no terraces.

Buildings are not built: there is no clean modular way to turn a footprint into catalogue walls automatically yet.
`--buildings hint` (default) writes a comment block per building for a builder: OSM tags, ground_z, eave / height /
ridge above ground, roof and ridge yaw, box, storeys, footprint polygon (map metres), every wall with ends, length,
outward direction and its window grid (storeys, windows, columns, spacing, sill / head per row, doors), then the best
stand-in as `#?` lines to uncomment by hand. `--buildings standin` spawns the stand-in (a block-out), `none` leaves
buildings out. The `.json` beside the plan has the same data (objects with row, scale, yaw, spawn; building hints
with walls). Existing files are never overwritten.

**scan_compare** pairs placed objects with scan objects: by group tag first (`scan.<class>.<id>`, `<class>.<id>`,
`b<id>` with any suffix; every object of the group stands for that scan object; a tag naming a scan object outside
the selected classes explains its objects: `tagged_outside_selection`), then by the nearest unpaired placed object of
a fitting kind; untagged pieces (not vehicles or foliage) standing on a selected building's cells join that building
(`by: footprint`; a group by majority of its pieces). Placed objects come from one read-only group export of the
open editor, or from a `scene_snapshot` name / `.json.gz` file (a file works without the game).

- **Ground pieces are counted, not compared**: ground and decal rows, lawn foliage, and any slab, tile or marking whose
  top is within 0.6 m of the scan ground (however deep it reaches down: foundation fills) - except small props and
  vehicles. The reply's `ground` gives the count per group; gameplay rows (spawners, gates) are counted apart.
- **ground_steps**: the placed ground tiles (ground-tagged meshes at least 1 m a side) as top faces under their full
  rotation; where two tiles' edges meet, the largest height step per pair: `edges_over_1cm`, `same_row` (seams: a
  smooth ground has none) and `between_rows` (may be intended kerbs) with count, median, p90, max, the `worst`
  places (x, y, rows, actor ids), and `far`: tiles whose top is more than `--flat_tol_m` (0.3) from the scan ground.
- **Buildings** (any number of pieces: walls, roof slabs, triangle gables) are compared as 0.5 m height rasters of
  the pieces' top surfaces (simplified mesh hull of each row, else its box, under the full rotation) against the
  scan building's cells (its footprint polygon when the classes left it few cells), clipped to `--box` when given:
  `iou` of the raised footprints, `coverage` (share of the scan building built), `precision`, `missing_m2` /
  `extra_m2`, `dz_median_m` (built minus scan top on shared cells), `dz_p90_m`, built and scan top, and
  `best_shift_m` (the whole-cell move within 3 m that overlaps best, with `iou_at_shift`). Status `offset` when that
  move gains 0.08 iou and is at least `--position_m` (fix: `edit_group GROUP --delta dx,dy,0`, preview first),
  `shape_off` when iou is under `--iou` (0.6), `height_off` when the median dz is beyond `--height_m`. The reply's
  `buildings` table lists them all.
- **Point objects** are measured by their visible size (asset profile); several objects in one tagged group (a hedge of
  bushes) are measured as one box along the scan axis. Status (the worst of `issues`): `missing`, `offset` (visible
  centres more than `--position_m` apart), `yaw_off` (long axes more than `--yaw_deg` apart, modulo the row's
  symmetry and at most 180: the scan box has no front), `size_off` (`size_ratio` placed / scan for length, width,
  height, trees crown and height, off by more than `--size_tol`, 0.15), `height_off` (placed top vs scan top beyond
  `--height_m`; trees 20 %), `facing_off` (the direction a tilted face looks to, e.g. solar panels rolled by hand, vs
  the scan's mean normal) or `matched`. A problem of a single object carries `fix`: an agent line `transform ACTOR x y z
  - yaw - sx sy sz` (cm, degrees, `-` keeps) with the visible centre on the scan object, the smallest yaw change onto
  the scan axis and the scan_match scale; z keeps the bottom (drop afterwards where the ground differs); no line when
  the row cannot do better (catalogue or scale limits: read `size_ratio`).
- `unmatched_placed`: other placed objects nothing explains, tallest first, with the scan surface, height and object
  under them. `height_cells`: per `--cell_m` cell (5 m) of the box (else around the placed objects), clipped to the
  survey extent, the scan height above ground vs the highest built top (95th percentiles; buildings and other solid
  pieces by their mesh top surfaces, foliage by crown discs), with the worst cells. At most 40,000 cells and 2 million
  lattice samples (about 0.5 m, coarser on big boxes); a box outside the survey gives no `height_cells`.

Problems are ranked by severity (status, then how far off). Full report: `compare.json` in the survey folder.

Measured live on a 100 x 100 m box of the second test build (3,177 objects: 22 building groups of 5 to 453 pieces,
1,225 ground tiles, 222 lawn foliage, 119 markings, 34 vehicles, 15 trees, props; 1.2 s): before, 0 of 17 buildings
paired (all `missing`) and 2,104 unmatched placed objects; now 15 of 17 `matched`, building 89 `offset` (iou 0.57,
a 1.6 m move would reach 0.69: a chamfered pentagon built as a rectangle) and 104 `shape_off` (iou 0.56, a 2.6 m wide
lean-to), the iou / dz values within a few hundredths of the builder's own raster check; 1,303 ground pieces counted,
7 unmatched (pallets and boxes standing on scan small objects outside the selected classes); `height_cells` median
|dz| 0.41 m (1.24 m when pitched roof slabs were measured by their unrotated boxes).
Measured on the 2026-11-25 export (whole survey, a plan of 192 cars and 238 trees turned into a fake snapshot with
injected faults; 0.2 s per compare): 8 / 8 cars turned 90 deg found as `yaw_off`, 8 / 8 turned 180 deg not flagged,
8 / 8 moved 2.5 m found, 8 / 8 deleted trees missing; 112 of 158 trees scaled by the test build's box rule flagged
`size_off` (the other 46 are rows whose crown fills 0.88+ of their box: within 15 %). Applying the 128 fix lines left
no offset / yaw problem; 17 `size_off` remained, all residuals of the exact plan too (catalogue vehicles 0.85..1.15,
squash limit on trees). On the exact plan 26 of 430 objects are `size_off` for the same reason.

**capture_topdown** renders the open editor straight down over the same box and pixel grid as `scan_view`
(`view.crop_frame`: metres per pixel = box side / px, north up), without moving the editor camera. With the agent DLL
built from this tree it is orthographic (`capture start ... ortho_cm [all|reference]` in `native/camera.h`:
`ProjectionType` Orthographic, `OrthoWidth`); an older injected DLL falls back to a 4 deg perspective camera about 14
box widths up, and the reply says so with its parallax (`agent_unload` makes the next call inject the new DLL).
`--reference only` renders nothing but the textured reference (show-only list: a before picture at any time; new DLL
only); `include` keeps it with the build; default `hide`. `capture_view --reference_only` and
`capture_sheet --reference_only` render only the reference for any pose (perspective, same DLL): the before half of
before/after pairs; a hidden reference is shown for the capture and hidden again. Camera height from the scan in the box (placement), else
`--ground_z` / `--top_z`. PNG and `.json` sidecar in `/dev/shm`. Not yet tested against the running game (the
orthographic and reference-only paths were built and compiled, not run live).

**scan_view --compare IMAGE** takes the capture's box and grid and writes three panels: the scan layer with ids,
the build with scan outlines, and an edge overlay (scan edges magenta, build green, white where both agree within a
pixel; `edge_agreement` is that share). The worst `--deviations` (25) of the survey's last `scan_compare` inside the
box are labelled on the build and overlay panels (`id status values`, buildings with their iou; then `extra ROW` for
unexplained placed objects, tallest first); ground pieces are never labelled, also from older reports
(`deviations_in_box` / `extras_in_box` give the totals). `warnings` (and the panel title) say when that report is
older than 10 minutes or covers a different box (run `scan_compare --box` with the image's box).

Then build one block, `capture_view` it next to a `scan_view` / `scan_sheet` of the same area, fix, replicate. For
buildings use `scan_building ID` and `scan_facade ID` (footprint = roof outline: walls are ~0.3 to 0.8 m inside;
eave_m for wall height; window rows for openings). For ground use `scan_ground` shares and outlines, `scan_profile`
for slopes.

## Catalogue hints for scan builds

What `search_assets` has (checked 2026-11-25; many rows are hidden variants: `--include_hidden`):

- Light asphalt: none. The only asphalt is `NewRow_182` (Ground, dark blue-grey). Light ground slabs: Concrete Floor01
  `NewRow_303` (triplanar concrete), its grey / white variants `NewRow_311` (Mi_GreySimplWall) and `NewRow_386`
  (MI_WhiteSimplWall); ground tiles Pavement 01 / 02 / 03 (`NewRow_197`, `NewRow_186`, `NewRow_185`), Flagstone
  `NewRow_376`.
- Roof tiles: no tiled roof row and no triangular tile row. Triangle Wall (`NewRow_237` brick, colour variants
  `NewRow_241`, `NewRow_249` white, `NewRow_567` grey, `NewRow_570` black, `NewRow_575` red, ...) for gables and hip
  faces in flat colours; roof slabs MEroof (`NewRow_781`..`786`, wood), Roof 8x8 `NewRow_336`, Garage Roof 4x4
  `NewRow_334`, Logcabin Roof `NewRow_337`; coloured floor slabs (Concrete Floor01 `NewRow_387`..`398`: black, beige,
  brown, red, ...) as pitched roof planes.
- Framed windows: none at facade scale. The only window row is Window 02 07 `NewRow_359` (0.5 x 0.6 m); frames have
  to be built from thin slabs (e.g. white Concrete Floor01 `NewRow_386` scaled flat) around a dark pane.

## Trust and failure modes

Measured by eye on the example Google Earth scan (500 x 500 m, 4.2 M triangles, a town centre).

| | geometry only | with vision |
| --- | --- | --- |
| cars, open lot (about 28) | precision 0.93, recall 0.87 | about 28 / 28, no false car |
| cars, street rows (about 10) | precision ~0.95, recall ~0.87 | 10 / 10 visible cars, no false car |
| construction yard / carports / sheds | precision ~0.7 (tiled false cars) | no false car in the yard; 28 of 30 sampled rejections correct |
| cars added by vision (39 sampled) | — | about 33 correct before the no-data filter (edge-of-scan boxes since dropped) |
| windows on clear walls (6 walls, ~190 windows) | not surveyed | precision ~0.95, recall ~0.9 |

Vision failure modes: cars under crowns, carport roofs or deep shadow are still missed (nothing to see). Shop
fronts and large glazing are not windows; glazed stairwells can read as garage doors. Balconies are rarely found,
and loggias whose railings the mesh lost show as holes. Dormers count as a window row (an attic storey). Walls
behind trees or seen at a grazing angle have no detections (`status: no_view`). Solar panel boxes cover parts of
arrays, and some skylights and glass roofs pass as panels or pools.

Mapping, measured on the same scan (whole survey, conf >= 0.6; 2026-11-25): `scan_match` of all 1,076 objects
takes 1.2 s, `scan_plan` 1.4 s (754 spawns: 591 trees, 161 cars, 2 trucks; 183 building hint blocks; 138 trees and
1 car skipped), `scan_compare` 0.2 s, a 16-crop `scan_sheet` 0.5 s. Planned mesh boxes land on the object centres
and on the lift height within 2 mm (recomputed from the written pivots). Cars: median fit error 7 % (Gt86 / Camaro /
Kangoo / Old Van); the 14 cars of a test lot drawn as their stand-in boxes sat on the cars in the image. Trees:
median size error 5 %; 16 / 16 sampled planned trees were real crowns (two of them merged groups). The plan skips
"slender" trees (taller than 3 x a crown under 4 m): 13 of 16 sampled were roof greenery or building edges, 3 small
real trees at the scan edge / riverbank. Buildings: median proportion error 21 %: the catalogue has few houses (four
gabled, mostly with the ridge across the long side) and many Blueprint blocks without a known roof, so stand-ins
are block-outs only. scan_compare on a fake snapshot made from a plan with injected faults found 46 / 46 moved
objects (offset), 52 / 52 deleted (missing), 39 / 40 re-scaled (height_off), paired 95 / 95 objects without group
tags by distance and listed 5 / 5 strays.

Rules from the live test builds, measured on the 2026-11-25 export (radius 150 m, survey with OSM and vision, no
other providers; images looked at): 14 crown candidates became roof, 12 to 13 of them by eye roof edges, facade tops
or planted roofs, one possibly a small garden tree; 25 crowns kept with lower conf beside a roof of their height.
4 OSM footprints disagree with the heights (a garage court mapped over open paving, two 11 m2 sheds that are not
there, one 3 m off). Raised cells with a ground class: 8,575 before, 0 after. Building parts: 13 parts on 6 buildings
(two houses, an L-shaped terrace, a block with its hall, wings), checked over the ortho; before, 10 parts of which
4 were slivers or a box around stray cells. Vision fusion rejected 3 car boxes as clutter (garden furniture or
stacks by eye); no real car among them. Roofs: 29 flat, 30 gabled, 22 hipped, 15 complex, 14 mono-pitch;
`roof_shape` finds 2 valleys and 44 two-plane ridges. `scan_building` of a building the survey skipped surveyed its
walls in 4.7 s; `scan_grid` of 200 x 200 m at 0.5 m took 0.2 s (1.3 MB).

Geometry failure modes, with or without vision: trees merge or split where crowns touch, and leafless or brown
crowns over roofs can read as roofs. Footprints are roof outlines and outer rings only (courtyards are not holes);
without OSM, terraced houses merge into one building. The DSM only sees the highest surface, so passages under
bridges, arches and dense canopy read as solid (`under_m` in `scan_height` hints at an overhang). Prefer high
`conf`, and look at crops (`scan_view`, `scan_facade`) before building from a count.

## Files

- `src/uncrashed_mcp/scan/` — `ops.py` (commands), `survey.py` (DTM, classes, objects), `osm.py` (fetch, parse,
  rasterise), `store.py` (keys, placement, frames), `view.py` (images), `grid.py` (numpy image helpers),
  `native.py` (ctypes binding), `vision.py` (vision stage: tiles, worker call, fusion, facade planning),
  `vision_worker.py` (runs in `.venv-vision`: YOLO11-OBB, Grounding DINO, SAM 2), `facade.py` (walls, facade frame,
  window-grid fit), `mapping.py` (scan_match, scan_plan, scan_compare, scan_sheet: catalogue candidates, mesh-hull
  roofs, fit and scale, plan writing, pairing, object comparison, fix lines, height cells), `composite.py` (group
  tags, ground / gameplay pieces, mesh top surfaces and the per-building height raster of scan_compare), `profile.py`
  (asset_profile), `topdown.py` (capture_topdown, scan_view --compare).
- `src/uncrashed_mcp/scan/providers/` — `__init__.py` (interface, registry, gather, Raster), `geo.py` (UTM, Web
  Mercator, scan frame), `net.py` (HTTP; `UNCRASHED_PROVIDERS=0` turns downloads off), `geotiff.py` (GeoTIFF / COG
  reader: LZW, Deflate, predictors, range requests), `osm_provider.py`, `overture.py` (PMTiles + vector tiles),
  `copernicus.py`, `worldcover.py`, `bavaria.py` (DGM1, LoD2 CityGML), `netherlands.py` (3D BAG, RD New), `fuse.py`.
  `streetview.py` — street-view cameras, lifting, multi-view fusion.
- `native/scanraster.cpp` — rasteriser, facade renderer, perspective views (`sr_views`), TIFF LZW, and helpers (labels, nearest, contours, polygon fill), built on first use into
  `native/libscanraster.so` (needs g++, OpenMP, libturbojpeg).
- Environment: `UNCRASHED_PROVIDERS=0` (no provider downloads), `UNCRASHED_SCAN_DIR` (disk), `UNCRASHED_SCAN_SHM` (RAM folder), `UNCRASHED_OSM=0` (never fetch OSM),
  `UNCRASHED_OVERPASS` (comma-separated Overpass endpoints; when set, the bbox goes only there),
  `UNCRASHED_OSM_MAP_API` (fallback map-API URL for small boxes; empty = none; default api.openstreetmap.org unless
  `UNCRASHED_OVERPASS` is set), `UNCRASHED_VISION=0` (never run vision), `UNCRASHED_VISION_VENV`, `UNCRASHED_MODELS` (weights
  folder), `UNCRASHED_VISION_DEVICE` (`cuda:N`), `UNCRASHED_VISION_CPU=1` (allow the CPU, slow),
  `UNCRASHED_VISION_TIMEOUT` (seconds, 1800), `UNCRASHED_VISION_REUSE=0` (never reuse RAM detections).
