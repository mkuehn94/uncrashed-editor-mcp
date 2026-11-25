# Textured OBJ reference

A textured OBJ export (photogrammetry, a city model) shown in the open track editor as a **view-only reference**:
one untagged actor in game memory with one non-colliding `ProceduralMeshComponent` per chunk. It is not a catalogue
object, is never saved, does not collide, is invisible to every editor-object query and audit, and is gone when the
scene changes. Use it to trace a real place with stock assets; a shareable map still has to be built from the
catalogue.

## Commands

```sh
./uc reference_load /data/city/model.obj --at 0,0,0                          # whole export, origin at map 0,0,0
./uc reference_load model.obj --at 15,-40,0 --center 120,-40 --radius_m 150   # one region
./uc reference_load model.obj --map_origin 51.3397,12.3731 --replace          # georeferenced (metadata.json)
./uc reference_status          # chunks, triangles, texels, pose, hidden, estimated game memory
./uc reference_visibility      # hide; --no-hidden shows it again
./uc reference_clear           # remove it (and any pack folder a killed load left behind)
./uc exec reference move 2000 0 300 45   # centimetres, yaw degrees; native verb, no Python wrapper
```

`reference_load` acts at once (`!` in `./uc help`) and is slow for big exports: give it a long shell timeout. It
holds the agent lock for the whole load, so other clients wait.

Parameters (metres, degrees):

- `--at x,y,z` puts the export origin (after `--center`) at this map position; or `--map_origin lat,lon[,z]` gives
  the map origin on WGS84 and needs `metadata.json` with `origin` `lat`/`lon` next to the OBJ. `--yaw` turns about Z.
- `--center east,south --radius_m R` selects a region of the export (export metres, 0 = everything). Selection is
  per triangle centroid, after per-mesh bounds.
- `--datum_m auto|number` is the source height placed at `z`. `auto` (default) is the 2nd percentile of the
  selected region's vertex heights, read from the index (`verts.bin`, memory-mapped with numpy, a pure-Python
  sample without it): normally the ground, robust to a few deep spikes. The result reports `datum_m` and how it
  was chosen.
- `--axes` names what source x, y, z point to (`east,up,south`, the OBJ Y-up default; or from `metadata.json`
  text like `x=east y=north z=up`). Right-handed exports map onto Unreal's left-handed X east / Y south / Z up
  without a mirror image.
- `--near_m 160` keeps full texture resolution for chunks within it; farther chunks are downscaled by
  `--far_scale 2` (1..8).
- `--min_free_gb 3` is the host memory floor: before each chunk the estimated game memory of that chunk is
  compared with `MemAvailable`; the load stops before a chunk that would go below the floor. The result then
  says `stopped: memory floor: ...` and the CLI exits 3.
- `--replace` swaps an already loaded reference; without it a second load is refused.

The result reports `loaded_chunks` / `chunks`, `triangles` / `triangles_total`, `stopped`, `est_game_gib`
(loaded) and `est_game_gib_all`, `datum_m`, `atlas_mpx`, the pose, `index_seconds`, `pack_seconds`, whether the
pack was on tmpfs and deleted, and `warnings`. A chunk that fails after others were imported raises an error
saying how many chunks are loaded (from `reference status`) and that `reference_clear` removes them; the loaded
chunks stay until then.

## Pipeline and host disk

1. **Index** (`native/ucmesh index`, once per OBJ version): parses `v`, `vt`, `f`, `o`/`g`, `usemtl`, `mtllib`,
   `map_Kd` into `~/.cache/uncrashed-mcp/mesh/<path key>/<size>-<mtime>/` (`index.bin`, `verts.bin` = float x y z
   u v per vertex in source axes, `tris.bin`). Geometry and UVs only, no pixels. An index of another modification
   time of the same OBJ path is pruned when a new one is built.
2. **Pack** (`native/ucmesh pack`, every load): selects the region, drops parent triangles covered by finer
   octree tiles (digit tile names), groups meshes into 96 m chunks and writes per chunk one UCMESH2 file plus
   JPEG atlas pages (4096 px wide). The pack goes to tmpfs, `/dev/shm/uncrashed-reference/<key>-<pid>/`. When
   `/dev/shm` is missing or has less free space than the estimate (geometry plus source JPEG bytes, x1.5,
   +64 MiB), it goes to `~/.cache/uncrashed-mcp/reference-packs/` instead, with a warning.
3. **Import** (`reference add X Y Z YAW PATH`, nearest chunk first): the DLL reads the UCMESH2 file, imports each
   atlas page with `ImportFileAsTexture2D` and builds the mesh sections before the command returns; nothing reads
   the files again (`status`, `move`, `hide`, `show` and `clear` use memory only). Each chunk's files are deleted
   right after its import, and the pack folder is removed when the load ends, also after a failure or a memory
   stop. Source textures therefore exist only as the original files and, transiently, as tmpfs pages: no copy is
   kept on disk.

`reference_clear` also deletes pack folders a killed load left behind (a folder owned by another live process is
kept); `reference_status` counts them as `leftover_packs`.

## Captures and scene changes

- `capture_view` and `capture_sheet` hide a visible reference for the capture and show it again afterwards (one
  `reference status` per capture or sheet, not per pose). Pass `--reference` to keep it in the image. A failed
  show is reported in the result (`reference_error`), not raised.
- `load_live` clears a loaded reference before loading (it belongs to the editor world being replaced) and says
  so in its result (`reference`).
- An agent DLL built before `native/reference.h` answers `unknown verb reference`: captures and `load_live` treat
  that as no reference; `reference_load` refuses with a hint to rebuild and `./uc unload`.

## Limits

Measured on a full 250 m export: 4.2 million triangles in 32 chunks, about 350 Mpx of atlas, estimated about
1.7 GiB of game memory; index 1.6 s (first load of that OBJ version), pack 3.9 s.

- Textures: JPEG `map_Kd` only (convert PNG first); `map_Kd` options are refused; faces without `vt` in a textured
  mesh are an error; untextured geometry is skipped.
- Pack refuses more than 12 million triangles or 900 Mpx of atlas (shrink `--radius_m`, raise `--far_scale`).
- Per chunk at most 16 atlas pages, 6 million vertices, 24 million indices; at most 1024 chunks.
- Game memory estimate: ~100 bytes per vertex plus 4 bytes per texel (no mips). Diffuse texture and UVs only; no
  PBR, no baked lighting, no collision, no shadows.
- One reference at a time. Rights: the export's licence still applies to what you show or record.

## Not verified live

The Python side and `native/ucmesh` are covered by offline tests (`tests/test_reference.py`, with a faked agent).
Against the running game, not yet checked with this pipeline: that Proton's `Z:` drive sees `/dev/shm`, the
per-chunk deletion right after `reference add`, the memory estimate against real usage, and capture hiding.
