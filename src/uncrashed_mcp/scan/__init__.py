"""Survey a textured photogrammetry OBJ and answer spatial questions about it (the scan_* operations).

This module stays import-light (no numpy): the CLI reads HELP for `./uc help scan`.
"""

HELP = """
SCAN: ask a 3D scan (textured photogrammetry OBJ) precise spatial questions; no game needed.
Full workflow: docs/scan.md. Every command: ./uc help scan_<name>.

1 SURVEY  scan_survey OBJ [--center e,s --radius_m R] [--res_m 0.25] [--no-osm] [--no-vision] [--no-facades]
          [--providers auto|none|names] [--no-street_views] [--refresh]
  Rasterises the mesh top-down (seconds), filters a ground model, classifies surfaces and finds buildings
  (footprint, parts, roof, eave/ridge), trees, cars, low structures, shrubs, bridges. With metadata.json
  (origin lat/lon) it fetches OpenStreetMap for the bbox once: footprints split merged roofs, roads / parking /
  paths / water become ground classes. With .venv-vision (GPU, ~1.5 min per 500 x 500 m) detectors look at the
  ortho image (cars confirmed / rejected / added, pools, boats, solar panels) and at a straight-on render of every
  building wall (windows, doors, balconies -> rows, storeys). Each object has conf and src (mesh / mesh+osm /
  mesh+vision / vision; models = detector scores). Open data providers chosen by bbox coverage (scan_sources):
  a regional terrain model (e.g. Bavarian DGM1) replaces the ground model, LoD2 / 3D BAG footprints and roof
  planes override building geometry (src ...+by_lod2, mesh values under `mesh`), Overture / GLO-30 / WorldCover
  cross-check; objects list `evidence` (agree / conflict per source); conflicts show in flag. Street views: the
  mesh rendered from the OSM streets at 1.6 m and 8 m, detections lifted to 3D (doors, signs, lamps, benches...).
  The survey becomes current; scan_status lists surveys (--survey KEY picks another). Running it again on an
  older survey adds the vision and street-view stages.
2 PLACE   scan_place --at x,y,z --yaw Y --datum_m D  (the reference_load numbers) or --from_reference.
  Then every query speaks map metres (x east, y south, z up); unplaced: export metres (east, south, up).
  Results always say which frame; scan_status prints it. A changed OBJ stops queries until scan_survey re-runs.
3 LOOK    scan_view --center x,y --size_m 60 [--layer ortho|ndsm|classes] [--classes car,tree]
          scan_sheet x,y x,y ... | --objects 12,40 | --classes car --box ...  (many crops, one labelled PNG)
  PNG in /dev/shm (RAM, derived from the scan: never upload or commit it). Open it, read ids off it.
4 QUERY   scan_objects --classes car --box x0,y0,x1,y1 | --near x,y [--radius_m R | --nearest N]
          [--sort height|area|storeys|windows]   scan_height x,y[,z] ... [--agl]   scan_profile x,y[,z] x,y[,z]
          scan_at x,y   scan_ground x0,y0,x1,y1   scan_grid x0,y0,x1,y1 --res_m 1 (one raster file: dtm, top,
          class, object)   scan_building ID|PART_ID (roof_shape, parts, colours)   scan_facade ID [--side N]
          scan_free x,y,z x,y,z ... [--radius_m 1] [--agl]
  Rows carry flag (a reason to doubt them), colour, and walls check (odd window rows): look before using those.
  One call with many ids / points beats a shell loop (zsh does not split $ids: use an array).
5 MATCH   scan_match ID (top catalogue rows: score, fit / shape error, scale, spawn yaw; buildings by roof and
          ridge from the catalogue meshes) | scan_match --classes car,tree --box ... (best row per object)
6 PLAN    scan_plan --box x0,y0,x1,y1 [--classes car,truck,tree,building,ground] [--out plans/x.txt]  needs scan_place.
  ground: seam-free tiles on the scan ground (shared corners, tilted on slopes, material by surface class).
  Writes spawn + `drop last` lines (map cm, mesh box centre on the object, group=scan.<class>.<id>) and a .json;
  never runs them. Cars need visible ground around them; trees vary rows and yaw (slender ones are skipped). Buildings: comment blocks
  (footprint, heights, roof, walls, window rows) with a '#?' stand-in (--buildings standin spawns it, none drops
  it). Read it, then `./uc batch FILE`; look with capture_view next to scan_view.
7 CHECK   scan_compare [SNAPSHOT] [--box ...]: placed objects (open editor, or a scene_snapshot name / file) vs
  scan objects: matched / offset / height_off / missing, and placed objects the scan does not explain.
Trust: vision on (measured on a Google Earth scan): cars precision ~0.95 and recall ~0.95 on lots and streets
(skips, sheds, carports, water noise rejected); missed under crowns / roofs. Windows ~0.95 / ~0.9 on clear walls;
shop fronts, glass, balconies weak. Without vision: rule-based cars, false hits beside carports. Trees: crowns merge
or split. Footprints are roof outlines (eaves). The DSM is the highest surface: under bridges reads as solid. Look
(scan_view, scan_facade, scan_sheet) before building from a count; prefer high conf. Stand-ins match box size,
proportions and roof only (the catalogue has no colours): look at them.
""".strip()
