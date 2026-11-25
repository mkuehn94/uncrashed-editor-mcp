"""Where scan surveys live, and the map <-> scan frames.

Disk (no pixels): SCAN_HOME/<key>/meta.json, objects.json, providers/<name>/ (open data caches: OSM answer,
terrain / land-cover crops, building records); SCAN_HOME/placements.json
(placement per OBJ path); SCAN_HOME/current (key of the survey queries use by default).
RAM only: SHM/<key>/*.npy rasters (dsm, low, dtm, cls, objid, nrm, rgb, ortho) and SHM/<key>/views/*.png. They are
rebuilt from the OBJ index on demand (seconds) after a reboot.

Frames. scan: east, south, up metres of the export (after the axes mapping; up = source height). map: Unreal metres
(x east, y south, z up) once a placement {at, yaw, datum_m} is known: map_xy = at_xy + R(yaw) (e, s),
map_z = at_z + up - datum_m. This is exactly how reference_load places the export.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

SCAN_HOME = Path(os.environ.get('UNCRASHED_SCAN_DIR', Path.home() / '.cache' / 'uncrashed-mcp' / 'scan'))
SHM = Path(os.environ.get('UNCRASHED_SCAN_SHM', '/dev/shm/uncrashed-scan'))
RASTERS = ('dsm', 'low', 'dtm', 'cls', 'objid', 'nrm', 'rgb', 'ortho')
COMPLETE = 'complete.json'   # written last by a raster build: without it the RAM rasters are incomplete
KEEP_VIEWS = 50              # images kept per survey in SHM/<key>/views (oldest removed first)


class NoSurvey(ValueError):
    pass


def key(obj: Path, center, radius_m: float, res_m: float, params: dict) -> str:
    st = obj.stat()
    parts = [str(obj.resolve()), st.st_size, st.st_mtime_ns, [round(float(v), 3) for v in center], round(float(radius_m), 3),
             round(float(res_m), 4), params]
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]


def home(k: str) -> Path:
    return SCAN_HOME / k


def ram(k: str) -> Path:
    return SHM / k


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}')
    tmp.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False))
    tmp.replace(path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def surveys() -> list[dict]:
    out = []
    if SCAN_HOME.is_dir():
        for d in SCAN_HOME.iterdir():
            meta = read_json(d / 'meta.json')
            if isinstance(meta, dict) and meta.get('key') == d.name:
                out.append(meta)
    return sorted(out, key=lambda m: m.get('created', 0), reverse=True)


def current_key() -> str:
    try:
        return (SCAN_HOME / 'current').read_text().strip()
    except OSError:
        return ''


def set_current(k: str) -> None:
    SCAN_HOME.mkdir(parents=True, exist_ok=True)
    (SCAN_HOME / 'current').write_text(k + '\n')


def resolve(survey: str = '', fresh: bool = True) -> dict:
    """Meta of a survey key (or unique key prefix); '' = the current one. fresh: refuse a survey whose OBJ changed
    since (its objects and a rebuilt raster would describe two different scenes)."""
    k = (survey or '').strip() or current_key()
    if not k:
        raise NoSurvey('no scan survey yet: run scan_survey OBJ first')
    meta = read_json(home(k) / 'meta.json')
    if not isinstance(meta, dict):
        matches = [m for m in surveys() if m['key'].startswith(k)]
        if len(matches) != 1:
            raise NoSurvey(f'no scan survey {k!r}' + (' (ambiguous prefix)' if matches else '') + '; scan_status lists them')
        meta = matches[0]
    if fresh:
        why = stale(meta)
        if why:
            raise NoSurvey(f"{why} since survey {meta['key']}: run scan_survey {meta['obj']} again")
    return meta


def stale(meta: dict) -> str | None:
    """Why the survey no longer matches its OBJ ('OBJ changed' / 'OBJ missing'), or None."""
    try:
        st = Path(meta['obj']).stat()
    except OSError:
        return None   # moved or deleted: stored objects still answer; a raster rebuild says it needs the OBJ
    if 'obj_size' in meta and (st.st_size != meta['obj_size'] or st.st_mtime_ns != meta.get('obj_mtime_ns')):
        return 'OBJ changed'
    return None


def objects(meta: dict) -> list[dict]:
    value = read_json(home(meta['key']) / 'objects.json')
    if not isinstance(value, list):
        raise NoSurvey(f"survey {meta['key']} has no objects.json (re-run scan_survey)")
    return value


def rasters_present(k: str) -> bool:
    """A complete raster build is in RAM (its marker was written after every array)."""
    return (ram(k) / COMPLETE).is_file() and all((ram(k) / f'{name}.npy').is_file() for name in RASTERS)


def rasters_ok(k: str) -> bool:
    """Present and every array maps (a truncated or half-written file fails here, cheaply: headers only)."""
    if not rasters_present(k):
        return False
    import numpy as np
    try:
        for name in RASTERS:
            np.load(ram(k) / f'{name}.npy', mmap_mode='r')
    except (OSError, ValueError, EOFError):
        return False
    return True


def save_array(k: str, name: str, arr) -> None:
    """Write SHM/<key>/<name>.npy atomically: a reader mapping the old file keeps it; no reader sees half a file."""
    import numpy as np
    d = ram(k)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f'{name}.npy.{os.getpid()}.tmp'
    with open(tmp, 'wb') as f:
        np.save(f, arr)
    os.replace(tmp, d / f'{name}.npy')


def mark_complete(k: str, meta: dict) -> None:
    write_json(ram(k) / COMPLETE, {'key': k, 'obj_size': meta.get('obj_size'), 'obj_mtime_ns': meta.get('obj_mtime_ns'),
                                   'at': now()})


def unmark(k: str) -> None:
    (ram(k) / COMPLETE).unlink(missing_ok=True)
    for tmp in ram(k).glob('*.tmp') if ram(k).is_dir() else ():
        tmp.unlink(missing_ok=True)


_held: dict[str, int] = {}


@contextlib.contextmanager
def build_lock(k: str):
    """Serialises raster builds of one survey across processes (lock file beside the RAM folder); re-entrant
    within a process."""
    if _held.get(k):
        _held[k] += 1
        try:
            yield
        finally:
            _held[k] -= 1
        return
    SHM.mkdir(parents=True, exist_ok=True)
    with open(SHM / f'.{k}.lock', 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        _held[k] = 1
        try:
            yield
        finally:
            _held.pop(k, None)
            fcntl.flock(f, fcntl.LOCK_UN)


def drop_ram(k: str) -> None:
    shutil.rmtree(ram(k), ignore_errors=True)


def remove(k: str) -> None:
    """Delete a survey (disk and RAM)."""
    if k and '/' not in k:
        shutil.rmtree(home(k), ignore_errors=True)
        drop_ram(k)
        if current_key() == k:
            (SCAN_HOME / 'current').unlink(missing_ok=True)


def views_dir(k: str) -> Path:
    """Image folder of a survey in RAM; keeps the newest KEEP_VIEWS - 1 images so the caller can add one."""
    d = ram(k) / 'views'
    d.mkdir(parents=True, exist_ok=True)
    old = sorted(d.glob('*.png'), key=_mtime, reverse=True)[KEEP_VIEWS - 1:]
    for p in old:
        p.unlink(missing_ok=True)
    return d


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def views(k: str) -> list[Path]:
    """Images of a survey in RAM, newest first."""
    d = ram(k) / 'views'
    return sorted(d.glob('*.png'), key=_mtime, reverse=True) if d.is_dir() else []


def check_shm() -> None:
    parent = SHM.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise ValueError(f'{parent} is not a writable RAM folder: scan rasters and images stay in memory only '
                         '(set UNCRASHED_SCAN_SHM to a tmpfs folder)')


# ---------------------------------------------------------------------------- placement


def _placements() -> dict:
    value = read_json(SCAN_HOME / 'placements.json', {})
    return value if isinstance(value, dict) else {}


def placement(obj: str) -> dict | None:
    return _placements().get(str(obj))


def set_placement(obj: str, value: dict | None) -> None:
    data = _placements()
    if value is None:
        data.pop(str(obj), None)
    else:
        data[str(obj)] = value
    write_json(SCAN_HOME / 'placements.json', data)


class Frame:
    """Converts between the scan frame and the frame a caller uses ('map' when a placement exists, else 'scan')."""

    def __init__(self, meta: dict, frame: str = 'auto'):
        frame = (frame or 'auto').lower()
        if frame not in ('auto', 'map', 'scan'):
            raise ValueError("frame must be auto, map or scan")
        self.place = placement(meta['obj'])
        if frame == 'map' and not self.place:
            raise ValueError('no placement for this scan: run scan_place first, or pass --frame scan')
        self.name = 'map' if self.place and frame != 'scan' else 'scan'
        if self.name == 'map':
            self.at = [float(v) for v in self.place['at']]
            self.yaw = float(self.place['yaw'])
            self.datum = float(self.place['datum_m'])
            self.c, self.s = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))

    # points
    def to_scan(self, x, y):
        if self.name == 'scan':
            return x, y
        dx, dy = x - self.at[0], y - self.at[1]
        return self.c * dx + self.s * dy, -self.s * dx + self.c * dy

    def from_scan(self, e, s):
        if self.name == 'scan':
            return e, s
        return self.at[0] + self.c * e - self.s * s, self.at[1] + self.s * e + self.c * s

    # heights
    def z_out(self, up):
        return up if self.name == 'scan' else self.at[2] + up - self.datum

    def z_in(self, z):
        return z if self.name == 'scan' else z - self.at[2] + self.datum

    def yaw_out(self, yaw, period: float = 180.0):
        return yaw if self.name == 'scan' else (yaw + self.yaw) % period

    def describe(self) -> str:
        if self.name == 'scan':
            return 'scan: export east, south, up metres (no placement)'
        return (f"map: Unreal metres via placement at={self.at} yaw={self.yaw:g} datum_m={self.datum:g} "
                f"({self.place.get('source', 'given')})")


def now() -> float:
    return round(time.time(), 1)
