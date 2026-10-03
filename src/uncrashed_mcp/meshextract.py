"""Pull LOD0 triangles out of the Uncrashed paks and write data/meshes.bin.

The fit check used to treat every mesh as its bounding box, so a scaffold bay
and the space under an eave counted as solid. This reads the cooked position
and index buffers (UE 4.27, pak version 11, zlib) and keeps a simplified
triangle hull in centimetres.
"""

from __future__ import annotations

import json
import math
import struct
import zlib
from pathlib import Path

PAK_DIR = Path.home() / ".local/share/Steam/steamapps/common/Uncrashed FPV Drone Sim/Uncrashed/Content/Paks"
DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "meshes.bin"
MAX_TRIS = 8000


def _fstring(buf: bytes, pos: int) -> tuple[str, int]:
    n, = struct.unpack_from("<i", buf, pos)
    pos += 4
    if n == 0:
        return "", pos
    if n < 0:
        n = -n
        raw = buf[pos : pos + (n - 1) * 2]
        pos += n * 2
        return raw.decode("utf-16-le", "replace"), pos
    raw = buf[pos : pos + n - 1]
    pos += n
    return raw.decode("utf-8", "replace"), pos


class _Pak:
    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("rb")
        size = path.stat().st_size
        self.file.seek(max(0, size - 300))
        tail = self.file.read()
        magic = tail.rfind(bytes.fromhex("e1126f5a"))
        if magic < 0:
            raise RuntimeError(f"{path.name} has no pak footer")
        index_off, index_size = struct.unpack_from("<qq", tail, magic + 8)
        self.file.seek(index_off)
        index = self.file.read(index_size)
        pos = 0
        _mount, pos = _fstring(index, pos)
        pos += 12  # file count and path-hash seed
        has_hash, = struct.unpack_from("<I", index, pos)
        pos += 4
        if has_hash:
            pos += 16 + 20
        has_dir, = struct.unpack_from("<I", index, pos)
        pos += 4
        if not has_dir:
            self.encoded = b""
            self.files = {}
            return
        d_off, d_size = struct.unpack_from("<qq", index, pos)
        pos += 16 + 20
        enc_size, = struct.unpack_from("<I", index, pos)
        pos += 4
        self.encoded = index[pos : pos + enc_size]
        self.file.seek(d_off)
        directory = self.file.read(d_size)
        dpos = 0
        count, = struct.unpack_from("<I", directory, dpos)
        dpos += 4
        self.files: dict[str, int] = {}
        for _ in range(count):
            dirname, dpos = _fstring(directory, dpos)
            nfiles, = struct.unpack_from("<I", directory, dpos)
            dpos += 4
            for _ in range(nfiles):
                name, dpos = _fstring(directory, dpos)
                eindex, = struct.unpack_from("<I", directory, dpos)
                dpos += 4
                self.files[dirname + name] = eindex

    def extract(self, eindex: int) -> bytes:
        bitfield = struct.unpack_from("<I", self.encoded, eindex)[0]
        cursor = eindex + 4
        if (bitfield & 0x3F) == 0x3F:
            cursor += 4
        if bitfield & 0x80000000:
            offset = struct.unpack_from("<I", self.encoded, cursor)[0]
        else:
            offset = struct.unpack_from("<Q", self.encoded, cursor)[0]
        self.file.seek(offset)
        _base, _csize, uncompressed, method = struct.unpack("<qqqI", self.file.read(28))
        self.file.read(20)
        blocks: list[tuple[int, int]] = []
        if method:
            nblocks, = struct.unpack("<I", self.file.read(4))
            for _ in range(nblocks):
                start, end = struct.unpack("<qq", self.file.read(16))
                blocks.append((start, end))
        self.file.read(5)  # flag byte and compression block size
        if method == 0:
            self.file.seek(offset + 53)
            return self.file.read(uncompressed)
        parts = []
        for start, end in blocks:
            self.file.seek(offset + start)
            parts.append(zlib.decompress(self.file.read(end - start)))
        data = b"".join(parts)
        if len(data) != uncompressed:
            raise RuntimeError(f"decompressed {len(data)} bytes, expected {uncompressed}")
        return data


def _index_paks() -> dict[str, list[tuple[_Pak, int]]]:
    stems: dict[str, list[tuple[_Pak, int]]] = {}
    for path in sorted(PAK_DIR.glob("*.pak")):
        pak = _Pak(path)
        for name, eindex in pak.files.items():
            if not name.endswith(".uexp"):
                continue
            stem = Path(name).name[: -len(".uexp")]
            stems.setdefault(stem, []).append((pak, eindex))
    return stems


def _bounds_match(buf: bytes, at: int, count: int, origin, extent) -> float:
    mn = [1e9, 1e9, 1e9]
    mx = [-1e9, -1e9, -1e9]
    start = at + 8
    for k in range(count):
        x, y, z = struct.unpack_from("<fff", buf, start + k * 12)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return 1e9
        for axis, value in enumerate((x, y, z)):
            if value < mn[axis]:
                mn[axis] = value
            if value > mx[axis]:
                mx[axis] = value
    err = 0.0
    for axis in range(3):
        err = max(err, abs(mn[axis] - (origin[axis] - extent[axis])), abs(mx[axis] - (origin[axis] + extent[axis])))
    return err


def _find_positions(buf: bytes, origin, extent) -> tuple[int, int] | None:
    best: tuple[float, int, int, int] | None = None
    cursor = 0
    needle = b"\x0c\x00\x00\x00"
    while True:
        at = buf.find(needle, cursor)
        if at < 0:
            break
        cursor = at + 1
        if at + 8 > len(buf):
            break
        count = struct.unpack_from("<I", buf, at + 4)[0]
        if not (8 <= count <= 400000) or at + 8 + count * 12 > len(buf):
            continue
        # Reject the candidate from a handful of samples before measuring every vertex.
        step = max(1, count // 16)
        sample_ok = True
        for k in range(0, count, step):
            x, y, z = struct.unpack_from("<fff", buf, at + 8 + k * 12)
            if not (math.isfinite(x) and abs(x) < 200000 and abs(y) < 200000 and abs(z) < 200000):
                sample_ok = False
                break
        if not sample_ok:
            continue
        err = _bounds_match(buf, at, count, origin, extent)
        limit = max(3.0, 0.02 * max(extent))
        if err > limit:
            continue
        # LOD0 is the densest buffer that still matches the catalog box.
        if best is None or count > best[0] or (count == best[0] and err < best[1]):
            best = (count, err, at, count)
    if best is None:
        return None
    return best[2], best[3]


def _uv_end(buf: bytes, at: int, count: int) -> int | None:
    pos = at + 8 + count * 12
    if pos + 26 > len(buf):
        return None
    _strip = struct.unpack_from("<H", buf, pos)[0]
    pos += 2
    ntex, nverts, full_uv, high_tan = struct.unpack_from("<iiii", buf, pos)
    pos += 16
    if nverts != count or not (0 <= ntex <= 8) or full_uv not in (0, 1) or high_tan not in (0, 1):
        return None
    tstride, tcount = struct.unpack_from("<II", buf, pos)
    pos += 8
    if not (4 <= tstride <= 64) or tcount > count * 4 or pos + tcount * tstride > len(buf):
        return None
    pos += tcount * tstride
    if pos + 8 > len(buf):
        return None
    ustride, ucount = struct.unpack_from("<II", buf, pos)
    pos += 8
    if not (2 <= ustride <= 64) or pos + ucount * ustride > len(buf):
        return None
    return pos + ucount * ustride


def _skip_colors(buf: bytes, pos: int, count: int) -> int | None:
    if pos + 10 > len(buf):
        return None
    _strip = struct.unpack_from("<H", buf, pos)[0]
    pos += 2
    stride, num = struct.unpack_from("<ii", buf, pos)
    pos += 8
    if num == 0 and stride == 0:
        return pos
    if num != count or stride != 4 or pos + 8 > len(buf):
        return None
    stride2, num2 = struct.unpack_from("<ii", buf, pos)
    pos += 8
    if stride2 != stride or num2 != num or pos + num * stride > len(buf):
        return None
    return pos + num * stride


def _find_indices(buf: bytes, pos: int, count: int) -> tuple[int, ...] | None:
    limit = min(len(buf) - 8, pos + 64)
    for at in range(pos, limit):
        nbytes = struct.unpack_from("<I", buf, at)[0]
        if nbytes < 12 or nbytes % 6 != 0:
            continue
        nidx = nbytes // 2
        if nidx > count * 16 or at + 4 + nbytes > len(buf):
            continue
        indices = struct.unpack_from("<" + str(nidx) + "H", buf, at + 4)
        if any(v >= count for v in indices):
            continue
        return indices
    return None


def read_lod0(uexp: bytes, origin, extent) -> tuple[list[tuple[float, float, float]], tuple[int, ...]] | None:
    found = _find_positions(uexp, origin, extent)
    if found is None:
        return None
    at, count = found
    end = _uv_end(uexp, at, count)
    if end is None:
        return None
    after_color = _skip_colors(uexp, end, count)
    if after_color is None:
        return None
    indices = _find_indices(uexp, after_color, count)
    if not indices:
        return None
    verts = []
    base = at + 8
    for k in range(count):
        verts.append(struct.unpack_from("<fff", uexp, base + k * 12))
    return verts, indices


def _simplify(verts, indices, max_tris: int = MAX_TRIS):
    grid = 1
    chosen = None
    while grid <= 512:
        cells: dict[tuple[int, int, int], int] = {}
        reduced: list[tuple[int, int, int]] = []
        remap = [-1] * len(verts)
        for i, (x, y, z) in enumerate(verts):
            key = (int(round(x / grid)), int(round(y / grid)), int(round(z / grid)))
            slot = cells.get(key)
            if slot is None:
                slot = len(reduced)
                cells[key] = slot
                reduced.append((key[0] * grid, key[1] * grid, key[2] * grid))
            remap[i] = slot
        tris: set[tuple[int, int, int]] = set()
        for t in range(0, len(indices), 3):
            a, b, c = remap[indices[t]], remap[indices[t + 1]], remap[indices[t + 2]]
            if a == b or b == c or a == c:
                continue
            tris.add((a, b, c))
        chosen = (reduced, tris)
        if len(tris) <= max_tris and len(reduced) <= 65535:
            break
        grid *= 2
    assert chosen is not None
    reduced, tris = chosen
    if len(tris) > max_tris:
        # A huge shell can stay over the cap even at a half-metre grid. Keep a spread of faces.
        step = max(1, len(tris) // max_tris)
        kept = list(tris)[::step][:max_tris]
        tris = set(kept)
        used = sorted({v for tri in tris for v in tri})
        remap = {old: i for i, old in enumerate(used)}
        reduced = [reduced[old] for old in used]
        tris = {tuple(remap[v] for v in tri) for tri in tris}
    flat: list[int] = []
    for tri in tris:
        flat.extend(tri)
    return reduced, flat, grid


def build() -> dict:
    catalog = json.loads((DATA / "assets.json").read_text())["assets"]
    wanted: dict[str, tuple] = {}
    for asset in catalog:
        if asset["kind"] != "mesh" or asset.get("bounds_estimated") or not asset.get("source"):
            continue
        wanted.setdefault(asset["source"], (asset["origin_cm"], asset["extent_cm"]))
    print(f"indexing paks, {len(wanted)} meshes")
    stems = _index_paks()
    packed = []
    missing = []
    for source, (origin, extent) in sorted(wanted.items()):
        readers = stems.get(source) or []
        mesh = None
        for pak, eindex in readers:
            uexp = pak.extract(eindex)
            mesh = read_lod0(uexp, origin, extent)
            if mesh is not None:
                break
        if mesh is None:
            missing.append(source)
            continue
        verts, indices = mesh
        reduced, flat, grid = _simplify(verts, indices)
        if not flat:
            missing.append(source)
            continue
        for x, y, z in reduced:
            if not (-32768 <= x <= 32767 and -32768 <= y <= 32767 and -32768 <= z <= 32767):
                raise RuntimeError(f"{source} vertex {x, y, z} does not fit in int16 cm")
        packed.append((source, reduced, flat, grid, len(verts), len(indices) // 3))
        if len(packed) % 25 == 0:
            print(f"  {len(packed)} / {len(wanted)}")
    blob = bytearray()
    blob += b"UMSH"
    blob += struct.pack("<II", 1, len(packed))
    for source, reduced, flat, _grid, _nv, _nt in packed:
        name = source.encode("utf-8")
        blob += struct.pack("<H", len(name))
        blob += name
        blob += struct.pack("<II", len(reduced), len(flat))
        for x, y, z in reduced:
            blob += struct.pack("<hhh", x, y, z)
        blob += struct.pack("<" + str(len(flat)) + "H", *flat)
    OUT.write_bytes(blob)
    print(f"wrote {OUT} ({len(blob)} bytes), {len(packed)} meshes, missing {len(missing)}")
    if missing:
        print(" missing", ", ".join(missing[:12]))
    return {"meshes": len(packed), "missing": missing, "bytes": len(blob)}


if __name__ == "__main__":
    build()
