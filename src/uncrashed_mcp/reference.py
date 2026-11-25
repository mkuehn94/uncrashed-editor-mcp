"""Textured OBJ as a view-only reference in game memory.

Pipeline: native/ucmesh (host C++) indexes the OBJ once into a small cache (geometry + UVs, no pixels), then packs
the chosen region into chunks: one UCMESH2 file plus JPEG atlas pages each. The agent verb `reference add` turns
every chunk into a non-colliding ProceduralMeshComponent on one untagged actor; nothing is placed through the editor,
nothing is saved.

Host disk. Source textures are used as reference in RAM only, never kept as copies: the pack (atlas pages) goes to a
tmpfs folder (/dev/shm/uncrashed-reference/<key>), each chunk's files are deleted as soon as the DLL imported them
(`reference add` reads the UCMESH2 file and imports every atlas page with ImportFileAsTexture2D before it returns;
nothing reads them again), and the folder is removed when the load ends, whatever the outcome. Only the index cache
under ~/.cache/uncrashed-mcp/mesh stays; an index for another modification time of the same OBJ is pruned.

Frames. Unreal is left-handed: X east, Y SOUTH, Z up keeps the real-world picture. A right-handed export
(OBJ: x east, y up, z south; or x east, y north, z up) maps onto that without a mirror image. Map positions
given to `load` are Unreal metres; `map_origin` places a georeferenced export from latitude/longitude.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import mmap
import os
import re
import shutil
import struct
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from . import agent

REPO = Path(__file__).resolve().parents[2]
TOOL_SRC = REPO / 'native' / 'ucmesh.cpp'
TOOL = Path(os.environ.get('UNCRASHED_UCMESH', REPO / 'native' / 'ucmesh'))
CACHE = Path(os.environ.get('UNCRASHED_MESH_CACHE', Path.home() / '.cache' / 'uncrashed-mcp' / 'mesh'))
SHM = Path('/dev/shm')
PACKS = SHM / 'uncrashed-reference'                  # tmpfs: the atlas pages never reach the disk
SCRATCH_PACKS = CACHE.parent / 'reference-packs'      # fallback when /dev/shm is missing or too small
AXIS_WORDS = {'east', 'west', 'north', 'south', 'up', 'down'}
# per chunk in game memory: ~100 B per vertex (procedural mesh CPU + GPU copies), 4 B per texel (BGRA, no mips)
VERTEX_BYTES, TEXEL_BYTES = 100, 4
DATUM_PERCENTILE = 2.0      # auto datum: this percentile of the selected vertices' heights
DATUM_SAMPLES = 4_000_000   # vertices looked at for the auto datum (strided above that)
UNAVAILABLE = 'the loaded agent DLL has no reference verb (rebuild native/, then ./uc unload to reload it)'


class ReferenceError(ValueError):  # noqa: A001 - module-local name, raised for bad input and refused loads
    pass


class PartialLoad(RuntimeError):
    """A chunk failed after others were loaded: they stay in the editor until reference_clear."""


# ---------------------------------------------------------------------------- host tool


def tool() -> Path:
    """The ucmesh binary, rebuilt when its source is newer (needs g++ and libturbojpeg)."""
    if TOOL.is_file() and (not TOOL_SRC.is_file() or TOOL.stat().st_mtime >= TOOL_SRC.stat().st_mtime):
        return TOOL
    if not shutil.which('g++'):
        raise ReferenceError(f'{TOOL} is missing and g++ is not installed to build it from {TOOL_SRC}')
    cmd = ['g++', '-O2', '-std=c++17', str(TOOL_SRC), '-lturbojpeg', '-o', str(TOOL)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode:
        raise ReferenceError('building ucmesh failed (install libturbojpeg headers?): ' + done.stderr.strip()[-400:])
    return TOOL


def _run(*args: str) -> str:
    done = subprocess.run([str(tool()), *map(str, args)], capture_output=True, text=True)
    if done.returncode:
        text = (done.stderr or done.stdout).strip()
        raise ReferenceError(text.splitlines()[-1] if text else 'ucmesh failed')
    return done.stdout.strip()


def _key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------- source description


def metadata(obj: Path) -> dict:
    """Export metadata next to the OBJ (earth-inspect style metadata.json), or {}."""
    path = obj.with_name('metadata.json')
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def source_axes(obj: Path, axes: str = '') -> str:
    """Three words for source x, y, z. Default: metadata 'axes' text, else the OBJ Y-up convention."""
    if axes:
        words = axes.replace(' ', '').lower().split(',')
    else:
        text = str(metadata(obj).get('axes', '')).lower()
        found = dict(re.findall(r'\b([xyz])\s*=\s*(east|west|north|south|up|down)\b', text))
        words = [found.get(a, '') for a in 'xyz'] if len(found) == 3 else ['east', 'up', 'south']
    pairs = [{'east', 'west'}, {'north', 'south'}, {'up', 'down'}]
    if len(words) != 3 or not set(words) <= AXIS_WORDS or any(len(p & set(words)) != 1 for p in pairs):
        raise ReferenceError(f'axes must name east/west, north/south and up/down once each, got {axes or words}')
    return ','.join(words)


def _axis_map(axes: str) -> list[tuple[int, float]]:
    """(source index, sign) for Unreal east, south, up: the Python twin of ucmesh's parse_axes."""
    out: list[tuple[int, float] | None] = [None, None, None]
    target = {'east': (0, 1.0), 'west': (0, -1.0), 'south': (1, 1.0), 'north': (1, -1.0), 'up': (2, 1.0), 'down': (2, -1.0)}
    for i, word in enumerate(axes.split(',')):
        k, sign = target[word]
        out[k] = (i, sign)
    return out  # type: ignore[return-value]


def geo_offset(obj: Path, map_origin) -> tuple[float, float]:
    """Unreal metres (x east, y south) of the export origin relative to map_origin = (lat, lon) on WGS84."""
    origin = metadata(obj).get('origin') or {}
    try:
        lat, lon = float(origin['lat']), float(origin['lon'])
    except (KeyError, TypeError, ValueError):
        raise ReferenceError('map_origin needs metadata.json with origin lat/lon next to the OBJ; pass at instead') from None
    lat0, lon0 = float(map_origin[0]), float(map_origin[1])
    a, e2 = 6378137.0, 6.69437999014e-3
    phi = math.radians((lat + lat0) / 2)
    w = 1 - e2 * math.sin(phi) ** 2
    m, n = a * (1 - e2) / w ** 1.5, a / math.sqrt(w)
    east = math.radians(lon - lon0) * n * math.cos(phi)
    north = math.radians(lat - lat0) * m
    return east, -north


# ---------------------------------------------------------------------------- index cache


def _rmtree(path: Path) -> bool:
    if not path.exists():
        return False
    shutil.rmtree(path, ignore_errors=True)
    return not path.exists()


def ensure_index(obj: Path) -> Path:
    """Index cache of the OBJ: CACHE/<path key>/<size>-<mtime_ns>/. Builds it once (atomically) and prunes the
    indexes of other modification times of the same path. Holds geometry and UVs only, no pixels."""
    obj = obj.resolve()
    if not obj.is_file():
        raise ReferenceError(f'no such OBJ: {obj}')
    st = obj.stat()
    home = CACHE / _key(str(obj))
    cache = home / f'{st.st_size}-{st.st_mtime_ns}'
    if not (cache / 'index.bin').is_file():
        building = home / f'.building-{os.getpid()}'
        _rmtree(building)
        building.mkdir(parents=True)
        try:
            _run('index', obj, building)
            _rmtree(cache)
            building.rename(cache)
        finally:
            _rmtree(building)
        (home / 'source.txt').write_text(str(obj) + '\n')
    for other in home.iterdir():
        if other.is_dir() and other != cache and not other.name.startswith('.building'):
            _rmtree(other)  # an index of an older version of this OBJ
    for legacy in CACHE.iterdir():
        if (legacy / 'index.bin').is_file():
            _rmtree(legacy / 'packs')  # the first version kept atlas copies next to its index
    return cache


def read_index(cache: Path) -> tuple[list[str], list[dict]]:
    """(texture paths, meshes) of index.bin. Mesh: name, tex, v0, nv, i0, ni, lo, hi (source axes)."""
    data = (cache / 'index.bin').read_bytes()
    if data[:8] != b'UCMIDX1\0':
        raise ReferenceError(f'not a ucmesh index: {cache / "index.bin"}')
    off = 8
    ntex, nmesh = struct.unpack_from('<II', data, off)
    off += 8

    def text() -> str:
        nonlocal off
        (n,) = struct.unpack_from('<I', data, off)
        value = data[off + 4:off + 4 + n].decode('utf-8', 'replace')
        off += 4 + n
        return value

    try:
        textures = [text() for _ in range(ntex)]
        meshes = []
        for _ in range(nmesh):
            name = text()
            _level, tex, v0, nv, i0, ni = struct.unpack_from('<IIQQQQ', data, off)
            lo, hi = struct.unpack_from('<3f', data, off + 40), struct.unpack_from('<3f', data, off + 52)
            off += 64
            meshes.append({'name': name, 'tex': tex, 'v0': v0, 'nv': nv, 'i0': i0, 'ni': ni, 'lo': lo, 'hi': hi})
    except struct.error:
        raise ReferenceError(f'truncated ucmesh index: {cache / "index.bin"}') from None
    return textures, meshes


def _world(axes: list[tuple[int, float]], p) -> tuple[float, float, float]:
    return tuple(sign * p[src] for src, sign in axes)  # type: ignore[return-value]


def select(meshes: list[dict], axes: str, center=(0.0, 0.0), radius_m: float = 0.0) -> list[dict]:
    """Meshes whose horizontal bounds reach the region (radius 0 = all), like ucmesh pack's first test."""
    if radius_m <= 0:
        return list(meshes)
    amap, cx, cy, out = _axis_map(axes), float(center[0]), float(center[1]), []
    for m in meshes:
        a, b = _world(amap, m['lo']), _world(amap, m['hi'])
        dx = max(min(a[0], b[0]) - cx, 0.0, cx - max(a[0], b[0]))
        dy = max(min(a[1], b[1]) - cy, 0.0, cy - max(a[1], b[1]))
        if dx * dx + dy * dy <= radius_m * radius_m:
            out.append(m)
    return out


SITE_RADIUS_M = 30.0        # site ground: vertices within this radius of the centre ...
SITE_PERCENTILE = 10.0      # ... this percentile of their heights (matched the survey ground model on a city export)


def site_ground(cache: Path, meshes: list[dict], axes: str, center, radius_m: float, datum: float,
                at_z: float) -> dict | None:
    """Where the ground of the site centre lands relative to the datum: an auto datum is a low percentile of the
    whole region (a pit, a river bank, a basement ramp), so the centre's ground can sit metres above map z."""
    r = SITE_RADIUS_M if not radius_m else min(SITE_RADIUS_M, float(radius_m))
    try:
        g = auto_datum(cache, meshes, axes, center, r, percentile=SITE_PERCENTILE)
    except ReferenceError:
        return None
    up = round(g - float(datum), 2)
    return {'above_datum_m': up, 'map_z': round(float(at_z) + up, 2),
            'how': f'{SITE_PERCENTILE:g}th percentile of the vertex heights within {r:g} m of the centre',
            'note': (f'the site ground is at {up:+.2f} m from the datum (map z {at_z + up:.2f}); '
                     f'reference_load --at with z {at_z - up:.2f} puts it at map z {at_z:.2f}') if abs(up) >= 0.3 else
                    'the site ground is at the datum'}


def auto_datum(cache: Path, meshes: list[dict], axes: str, center=(0.0, 0.0), radius_m: float = 0.0,
               percentile: float = DATUM_PERCENTILE) -> float:
    """Ground height (source up metres) of the region: a low percentile of its vertices' heights.

    Reads verts.bin (float32 x y z u v per vertex, source axes) through a memory map; vertices outside the
    radius are left out, and above DATUM_SAMPLES vertices every n-th one is used."""
    total = sum(m['nv'] for m in meshes)
    if not total:
        raise ReferenceError('nothing selected for the datum (check center/radius_m and axes)')
    stride = max(1, math.ceil(total / DATUM_SAMPLES))
    (ei, es), (si, ss), (ui, us) = _axis_map(axes)
    cx, cy, r2 = float(center[0]), float(center[1]), float(radius_m) ** 2
    try:
        import numpy as np
    except ModuleNotFoundError:
        np = None
    if np is not None:
        verts = np.memmap(cache / 'verts.bin', dtype='<f4', mode='r').reshape(-1, 5)
        parts = []
        for m in meshes:
            block = np.asarray(verts[m['v0']:m['v0'] + m['nv']:stride])
            up = us * block[:, ui]
            if radius_m > 0:
                inside = (es * block[:, ei] - cx) ** 2 + (ss * block[:, si] - cy) ** 2 <= r2
                up = up[inside]
            parts.append(up)
        heights = np.concatenate(parts) if parts else np.empty(0, dtype='<f4')
        if not heights.size:
            raise ReferenceError('no vertex inside the region for the datum')
        return round(float(np.percentile(heights, percentile)), 3)
    heights = []
    with open(cache / 'verts.bin', 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
        floats = memoryview(view).cast('f')
        try:
            for m in meshes:
                for v in range(m['v0'], m['v0'] + m['nv'], stride):
                    base = v * 5
                    if radius_m > 0 and (es * floats[base + ei] - cx) ** 2 + (ss * floats[base + si] - cy) ** 2 > r2:
                        continue
                    heights.append(us * floats[base + ui])
        finally:
            floats.release()
    if not heights:
        raise ReferenceError('no vertex inside the region for the datum')
    heights.sort()
    rank = (len(heights) - 1) * percentile / 100
    low = int(rank)
    high = min(low + 1, len(heights) - 1)
    return round(heights[low] + (heights[high] - heights[low]) * (rank - low), 3)


# ---------------------------------------------------------------------------- pack (temporary)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_packs() -> int:
    """Delete leftover pack folders (a load that was killed). A folder of another live process is kept."""
    removed = 0
    for root in (PACKS, SCRATCH_PACKS):
        if not root.is_dir():
            continue
        for folder in root.iterdir():
            try:
                pid = int((folder / '.owner').read_text().split()[0])
            except (OSError, ValueError, IndexError):
                pid = 0
            if pid and pid != os.getpid() and _pid_alive(pid):
                continue
            removed += _rmtree(folder)
    return removed


def leftover_packs() -> int:
    """Pack folders present now (a running load's own, or ones a killed load left)."""
    return sum(1 for root in (PACKS, SCRATCH_PACKS) if root.is_dir() for folder in root.iterdir() if folder.is_dir())


def estimate_pack_bytes(textures: list[str], meshes: list[dict]) -> int:
    """Upper estimate of a pack's size: geometry (20 B per vertex, 4 B per index) plus the source JPEG bytes of
    the textures used (atlas pages are re-encoded, never larger in pixels), with a margin."""
    geometry = sum(m['nv'] * 20 + m['ni'] * 4 for m in meshes)
    pixels = 0
    for tex in {m['tex'] for m in meshes}:
        try:
            pixels += os.path.getsize(textures[tex])
        except (OSError, IndexError):
            pass
    return int((geometry + pixels) * 1.5) + (64 << 20)


def pack_root(need: int) -> tuple[Path, str]:
    """(folder for packs, warning). tmpfs when it exists and has room, the scratch cache otherwise."""
    try:
        if SHM.is_dir() and os.access(SHM, os.W_OK | os.X_OK) and shutil.disk_usage(SHM).free >= need:
            return PACKS, ''
        why = f'{SHM} has less than {need / 2 ** 20:.0f} MiB free' if SHM.is_dir() else f'{SHM} is missing'
    except OSError as exc:
        why = f'{SHM} is not usable ({exc.strerror or exc})'
    return SCRATCH_PACKS, f'{why}: the pack goes to {SCRATCH_PACKS} on disk (deleted after the load)'


def pack(cache: Path, out: Path, *, axes: str, center=(0.0, 0.0), radius_m: float = 0.0, datum_m: float = 0.0,
         near_m: float = 160.0, far_scale: int = 2, chunk_m: float = 96.0) -> dict:
    """Write a chunked UCMESH2 pack of the region into `out` (created, owned by this process). Returns the manifest."""
    args = ['--center', repr(float(center[0])), repr(float(center[1])), '--datum', repr(float(datum_m)),
            '--axes', axes, '--near', repr(float(near_m)), '--far-scale', str(int(far_scale)), '--chunk', repr(float(chunk_m))]
    args += ['--radius', repr(float(radius_m))] if radius_m > 0 else ['--all']
    if any(c.isspace() for c in str(out)):
        # the agent collapses runs of whitespace in a command line: the chunk path must not contain any
        raise ReferenceError(f'pack path must not contain spaces: {out} (set UNCRASHED_MESH_CACHE)')
    _rmtree(out)
    out.mkdir(parents=True)
    (out / '.owner').write_text(f'{os.getpid()}\n')
    _run('pack', cache, out, *args)
    return json.loads((out / 'manifest.json').read_text())


def plan(manifest: dict, directory: Path, at_m, yaw: float) -> list[tuple[dict, str]]:
    """(chunk, native `reference add` line), nearest chunks first."""
    x, y, z = (100.0 * float(v) for v in at_m)
    chunks = sorted(manifest['chunks'], key=lambda c: math.hypot(*c['centre_m']))
    return [(c, f'reference add {x:.3f} {y:.3f} {z:.3f} {float(yaw):.4f} {agent._windows_path(directory / c["file"])}')
            for c in chunks]


def _pages(chunk: dict, directory: Path) -> list[Path]:
    return sorted(directory.glob(Path(chunk['file']).stem + '_p*.jpg'))


def _available_gb() -> float:
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return math.inf


def chunk_gb(chunk: dict, directory: Path) -> float:
    """Estimated game memory of one chunk (GiB)."""
    texels = sum(_jpeg_texels(page) for page in _pages(chunk, directory))
    return (chunk['vertices'] * VERTEX_BYTES + texels * TEXEL_BYTES) / 2 ** 30


def _jpeg_texels(path: Path) -> int:
    data = path.read_bytes()
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker, size = data[i + 1], int.from_bytes(data[i + 2:i + 4], 'big')
        if marker in (0xC0, 0xC1, 0xC2):
            return int.from_bytes(data[i + 5:i + 7], 'big') * int.from_bytes(data[i + 7:i + 9], 'big')
        i += 2 + size
    raise ReferenceError(f'not a JPEG atlas: {path}')


# ---------------------------------------------------------------------------- live state


def _unknown_verb(exc: agent.AgentError) -> bool:
    """The agent answered that it does not know the verb: a DLL built before reference.h."""
    return not isinstance(exc, (agent.AgentNotSent, agent.AgentTimeout)) and 'unknown verb reference' in str(exc)


def status() -> dict:
    """Native `reference status`, parsed. available=False when the DLL predates the reference verb."""
    try:
        lines = agent.run('reference status')
    except agent.AgentError as exc:
        if _unknown_verb(exc):
            return {'available': False, 'loaded': False, 'hidden': False, 'note': UNAVAILABLE}
        raise
    line = next((l for l in lines if l.startswith('reference ')), '')
    if not line.startswith('reference loaded'):
        return {'available': True, 'loaded': False, 'hidden': False}
    fields = dict(re.findall(r'(\w+)=(\S+)', line))
    count = {k: int(fields[k]) for k in ('chunks', 'sections', 'triangles', 'vertices', 'textures', 'texels') if k in fields}
    pose = [float(v) for v in fields.get('pose', '').split(',') if v]
    out = {'available': True, 'loaded': True, 'hidden': fields.get('hidden') == '1', **count}
    if len(pose) == 4:
        out['pose_m'] = [round(v / 100, 3) for v in pose[:3]]
        out['yaw'] = pose[3]
    out['est_game_gib'] = round((count.get('vertices', 0) * VERTEX_BYTES + count.get('texels', 0) * TEXEL_BYTES) / 2 ** 30, 2)
    out['collision'], out['saved'] = 'off', False
    return out


def clear() -> dict:
    """Remove the reference from the editor and delete leftover pack folders."""
    packs = cleanup_packs()
    before = status()
    if not before['available']:
        return {'cleared': False, 'note': UNAVAILABLE, 'packs_removed': packs}
    if before['loaded']:
        agent.run('reference clear')
    return {'cleared': before['loaded'], 'chunks_removed': before.get('chunks', 0), 'packs_removed': packs}


def set_hidden(hidden: bool) -> dict:
    st = status()
    if not st['available']:
        raise ReferenceError(UNAVAILABLE)
    if not st['loaded']:
        raise ReferenceError('no reference loaded (reference_load first)')
    agent.run('reference hide' if hidden else 'reference show')
    return {'hidden': bool(hidden), 'chunks': st.get('chunks', 0)}


def drop_for_scene_change() -> str:
    """Clear a loaded reference before the scene is replaced; '' when there was none. The native reference
    belongs to the editor world that is open now."""
    st = status()
    if not st['loaded']:
        return ''
    agent.run('reference clear')
    cleanup_packs()
    return f"cleared the textured reference ({st.get('chunks', 0)} chunks) before loading: it belonged to the old scene"


@contextmanager
def capture_guard(include: bool = False):
    """Hide a visible reference for a capture and show it again afterwards. One status call.

    Yields a dict whose 'reference' entry says what happened: none | included | already hidden | hidden for
    capture (plus 'reference_error' when showing it again failed: it then stays hidden)."""
    state: dict = {'reference': 'included' if include else 'none'}
    if include:
        yield state
        return
    st = status()
    if not st['loaded']:
        yield state
        return
    if st['hidden']:
        state['reference'] = 'already hidden'
        yield state
        return
    agent.run('reference hide')
    state['reference'] = 'hidden for capture'
    try:
        yield state
    except BaseException:
        try:
            agent.run('reference show')
        except Exception:  # keep the capture's own error
            pass
        raise
    try:
        agent.run('reference show')
    except agent.AgentError as exc:
        state['reference_error'] = (f'the reference stays hidden: showing it again failed ({" ".join(str(exc).split())}); '
                                    'run reference_visibility --no-hidden')



@contextlib.contextmanager
def only_guard(state: dict):
    """For a reference-only capture: the reference must be loaded; a hidden one is shown for the capture and
    hidden again afterwards (state['reference'] says so)."""
    st = status()
    if not st.get('loaded'):
        raise ValueError('no textured reference loaded (reference_load first)')
    if not st.get('hidden'):
        yield state
        return
    agent.run('reference show')
    state['reference'] = 'shown for a reference-only capture, hidden again after'
    try:
        yield state
    finally:
        try:
            agent.run('reference hide')
        except Exception:  # keep the capture's own error
            pass


# ---------------------------------------------------------------------------- load


def _datum(value) -> float | None:
    """None for 'auto', else a finite number of metres."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
    else:
        text = str(value).strip().lower()
        if text in ('auto', ''):
            return None
        try:
            number = float(text)
        except ValueError:
            raise ReferenceError(f"datum_m is 'auto' or a number of metres, got {value!r}") from None
    if not math.isfinite(number):
        raise ReferenceError('datum_m must be finite')
    return number


def load(obj: str, *, at=None, map_origin=None, yaw: float = 0.0, datum_m='auto', center=None,
         radius_m: float = 0.0, near_m: float = 160.0, far_scale: int = 2, axes: str = '', replace: bool = False,
         min_free_gb: float = 3.0, _progress=None) -> dict:
    path = Path(obj).expanduser()
    if not path.is_file():
        raise ReferenceError(f'no such OBJ: {path}')
    if (at is None) == (map_origin is None):
        raise ReferenceError('give exactly one of at (x,y,z Unreal metres of the export origin) or map_origin (lat,lon[,z])')
    if at is not None:
        if len(at) != 3:
            raise ReferenceError('at is x,y,z')
        place = [float(v) for v in at]
    else:
        if len(map_origin) not in (2, 3):
            raise ReferenceError('map_origin is lat,lon[,z]')
        place = [*geo_offset(path, map_origin), float(map_origin[2]) if len(map_origin) == 3 else 0.0]
    center = tuple(float(v) for v in (center or (0.0, 0.0)))
    if len(center) != 2:
        raise ReferenceError('center is east,south metres in the export')
    datum = _datum(datum_m)
    if not all(math.isfinite(v) for v in (*place, *center, yaw, min_free_gb, radius_m, near_m)):
        raise ReferenceError('placement, center, radius_m, near_m and min_free_gb must be finite')
    if radius_m < 0 or near_m < 0 or min_free_gb < 0:
        raise ReferenceError('radius_m, near_m and min_free_gb must be >= 0')
    if not 1 <= int(far_scale) <= 8:
        raise ReferenceError('far_scale must be 1..8')
    axes = source_axes(path, axes)
    warnings: list[str] = []
    with agent.session():
        info = agent.map_info()
        if info.get('tag') != 'MapEditor':
            raise ReferenceError('open the track editor first')
        before = status()
        if not before['available']:
            raise ReferenceError(UNAVAILABLE)
        if before['loaded']:
            if not replace:
                raise ReferenceError(f"a reference is already loaded ({before.get('chunks', 0)} chunks): pass replace, "
                                     'or run reference_clear')
            agent.run('reference clear')
        cleanup_packs()
        started = time.time()
        cache = ensure_index(path)
        textures, meshes = read_index(cache)
        chosen = select(meshes, axes, center, radius_m)
        if not chosen:
            raise ReferenceError('nothing selected (check center/radius_m and axes)')
        indexed = time.time()
        datum_note = f'given {datum} m'
        if datum is None:
            datum = auto_datum(cache, chosen, axes, center, radius_m)
            datum_note = f'auto: {DATUM_PERCENTILE:g}th percentile of the region\'s vertex heights'
        site = site_ground(cache, chosen, axes, center, radius_m, datum, place[2])
        need = estimate_pack_bytes(textures, chosen)
        root, warning = pack_root(need)
        if warning:
            warnings.append(warning)
        directory = root / f'{_key(str(cache), axes, center, radius_m, datum, near_m, far_scale)}-{os.getpid()}'
        loaded, tris, game_gb, stopped = 0, 0, 0.0, ''
        try:
            manifest = pack(cache, directory, axes=axes, center=center, radius_m=radius_m, datum_m=datum,
                            near_m=near_m, far_scale=int(far_scale))
            packed = time.time()
            # the pack origin is `center`: move the actor there so the whole export keeps its source layout
            c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            pose = [place[0] + c * center[0] - s * center[1], place[1] + s * center[0] + c * center[1], place[2]]
            lines = plan(manifest, directory, pose, yaw)
            sizes = [chunk_gb(chunk, directory) for chunk, _ in lines]
            for (chunk, line), need_gb in zip(lines, sizes):
                free = _available_gb()
                if free - need_gb < min_free_gb:
                    stopped = (f'memory floor: stopped before chunk {loaded + 1}/{len(lines)}: {free:.1f} GiB free, '
                               f'chunk needs ~{need_gb:.2f} GiB, floor {min_free_gb:g} GiB')
                    break
                try:
                    agent.run(line, timeout=120)
                except agent.AgentError as exc:
                    raise PartialLoad(_partial_message(exc, loaded, len(lines))) from exc
                loaded += 1
                tris += chunk['triangles']
                game_gb += need_gb
                for done in [directory / chunk['file'], *_pages(chunk, directory)]:
                    done.unlink(missing_ok=True)  # imported: the DLL holds it in memory now
                if _progress:
                    _progress(loaded, len(lines))
        finally:
            _rmtree(directory)
            deleted = not directory.exists()
        final = status()
    if loaded:
        _record_load({'obj': str(path.resolve()), 'at': place, 'yaw': float(yaw), 'datum_m': datum, 'center': list(center),
                      'pose_m': [round(v, 3) for v in pose]})
    return {
        'loaded_chunks': loaded, 'chunks': len(lines), 'triangles': tris, 'triangles_total': manifest['triangles'],
        'stopped': stopped, 'est_game_gib': round(game_gb, 2), 'est_game_gib_all': round(sum(sizes), 2),
        'datum_m': datum, 'datum': datum_note, 'site_ground': site, 'atlas_mpx': round(float(manifest['atlas_mpx']), 1),
        'pose_m': [round(v, 3) for v in pose], 'yaw': yaw, 'axes': axes,
        'index_seconds': round(indexed - started, 1), 'pack_seconds': round(packed - indexed, 1),
        'seconds': round(time.time() - started, 1),
        'pack': 'tmpfs' if root == PACKS else 'disk', 'pack_deleted': deleted,
        'free_gib': round(_available_gb(), 1), 'status': final, 'warnings': warnings,
        'collision': 'off', 'saved': False,
    }


def _last_path() -> Path:
    return CACHE.parent / 'reference-last.json'


def last_load() -> dict | None:
    """Placement of the last reference_load on this machine (obj, at, yaw, datum_m, center, pose_m), or None.
    scan_place --from_reference reads it: reference_status alone has no datum."""
    try:
        value = json.loads(_last_path().read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and {'obj', 'at', 'yaw', 'datum_m'} <= set(value) else None


def _record_load(record: dict) -> None:
    try:
        path = _last_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))
    except OSError:
        pass  # a convenience record only


def _partial_message(exc: agent.AgentError, counted: int, total: int) -> str:
    loaded = counted
    try:
        live = status()
        loaded = live.get('chunks', 0) if live['loaded'] else 0
    except Exception:
        pass
    sent = ' (it was sent and may still be importing)' if isinstance(exc, agent.AgentTimeout) else ''
    held = (f'{loaded} of {total} chunks are loaded and stay in the editor (not saved); reference_clear removes them'
            if loaded else 'no chunk is loaded')
    return f"reference chunk {counted + 1}/{total} failed{sent}: {' '.join(str(exc).split())}; {held}"
