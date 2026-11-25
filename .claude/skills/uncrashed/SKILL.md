---
name: uncrashed
description: Edit the Uncrashed FPV track editor live or audit saved maps through the ./uc command line (search assets, batch-place objects, capture views, run audits, save). Use for any map building, inspection or audit task in this repository.
---

# Uncrashed track editor CLI

Run from the repository root: `./uc <command> [positionals] [--option value]`.
Read `./uc help guide` once before a build (workflow and cautions, about 1.9k tokens). `./uc help` lists
every command, `./uc help <command>` its parameters and examples, `./uc help verbs` the agent command language.

- **Units:** command options are METRES and degrees; `exec` / `batch` agent commands are CENTIMETRES.
- **Exit codes:** 0 ok · 1 failed (a live change may be partly applied) · 2 usage error, nothing ran ·
  3 ran but reported failures. Every failure prints one stderr line `error: ...`.
- **No game needed:** `status`, `list_maps`, `inspect_map NAME`, `search_assets QUERY [--category --tag --limit]`,
  `get_assets ID...`, `list_gates`, `noncolliding_alternatives ROW`, `check_performance NAME`,
  `consolidate_scenery NAME` (preview only).
- **Live (track editor open):** `exec mapinfo` first; `batch FILE` or `... | ./uc batch -` (one agent command
  per line); `sample_terrain x,y x,y`; `inspect_site x,y,z x,y,z`; `capture_view [x y z yaw pitch] --out DIR/`
  then open the PNG yourself; `capture_sheet @poses.txt --out DIR/` (up to 60 `label x y z yaw pitch` lines into one
  labelled sheet.png; gates stay visible); `move_camera x y z yaw`; `save_live [NAME]`;
  `scene_snapshot --name before` then `scene_diff before` (read-only change counts and gate/start verdicts).
- **Textured OBJ reference (view only, never saved, no collision):** `reference_load model.obj --at x,y,z
  [--center e,s --radius_m R]` (ground datum auto), `reference_status`, `reference_visibility [--no-hidden]`,
  `reference_clear`. Captures hide it unless `--reference`; `load_live` clears it. See docs/obj-reference.md.
- **3D scan questions (no game):** `scan_survey scene.obj` (ground, buildings, trees, cars, OSM priors; with
  `.venv-vision` also aerial detectors and facade windows, ~1.5 min on the GPU), `scan_place --at x,y,z --yaw Y
  --datum_m D` (the reference_load numbers: answers in map metres), then `scan_view --center x,y --size_m 60` (PNG in
  /dev/shm: look locally, never upload), `scan_objects --classes car --box ...` (`--sort height|area|storeys|windows`,
  `--near x,y --nearest 3`; read `flag`), `scan_height` / `scan_profile` (x,y,z + `--agl`), `scan_grid BOX` (one
  raster file instead of many height calls),
  `scan_at`, `scan_ground`, `scan_building ID` (walls, storeys, window rows), `scan_facade ID` (wall image with
  detections), `scan_free` (`--agl`: z above ground), `scan_sheet --classes car --box ...` (many crops, one PNG). Look before trusting counts.
  To the game: `scan_match ID` (catalogue rows, scale, yaw), `scan_plan --box ... --out plans/x.txt` (`--classes
  ground`: seam-free ground tiles; spawn +
  `drop last` lines in map cm, building hint blocks; never run: read it, then `batch` it), `scan_compare` (live or
  a snapshot file: matched / offset / yaw / size / missing), `asset_profile ROW` (visible size, symmetry), `capture_topdown` +
  `scan_view --compare IMG`. Build in phases, save + compare after each: docs/scan.md "Scan-based build loop". `./uc help scan`.
- **Agent verbs:** `spawn ROW x y z [pitch yaw roll] [sx sy sz] [group=tag]`, then `drop last` on the next
  line to rest it on what is below; `transform ACTOR x y z p y r sx sy sz` (`-` keeps a value); `destroy ACTOR`;
  `actors x y z radius`.
- **Arguments:** lists are one comma-separated token (`--delta 1,0,0`), negative numbers are plain values,
  JSON goes in single quotes, `@file` reads a list from a file.
- **Previews:** group, collision, clearing, assembly, undo and consolidate edits (`~` in `./uc help`) preview
  until `--apply`; add it only after reading the preview. `exec`, `batch`, `load_live`, `save_live` and
  `unload` (`!`) act at once, no preview, no undo. Nothing auto-saves.
- **Batch results:** `batch lines=N ok=M failed=K sent=N`. Successful lines ARE applied: resend only the
  failed ones, never the whole file.
- **Never re-send** a command that timed out, was interrupted or failed with `may have run` / `UNKNOWN`;
  check `./uc last` (last finished reply) and `./uc exec mapinfo` first. Only an error saying `nothing was
  sent` is safe to retry. Give slow commands (`check_collisions`, `load_live`, big batches, `check_route_clearance`, `check_opening`) a long
  shell timeout or run them in the background.
- **Ask the user before:** `load_live` (replaces the scene), `exec` with `clear` / `destroybox` / `base` /
  `setmap` (`destroy`, `destroyat` and `replace` have no undo either), `--apply` on deletes or `consolidate_scenery`, and overwriting a saved map.
- **Before saving a build:** `check_collisions`, `check_foundations`, `check_environment` (and
  `check_route_clearance` for races; `check_opening x,y,z x,y,z x,y,z` for one opening: first obstruction by path
  position, unverified objects beside it are not clearance). A clean audit is not verification: look at captures.
- **Output** is compact text (`./uc help output`); `--json` gives JSON for `jq`, `--pick a,b.c` keeps fields.
  With pipes use `set -o pipefail`.

Keep generator scripts and command files in the project (not `/tmp`) so a build can be reproduced.
Mechanical runs (executing a prepared command file, running audits, collecting captures) can be handed to
the `uncrashed-operator` subagent; design decisions and judging images stay with the main agent.
