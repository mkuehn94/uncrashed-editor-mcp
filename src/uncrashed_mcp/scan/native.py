"""ctypes binding of native/scanraster.cpp (built on demand, like native/ucmesh)."""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / 'native' / 'scanraster.cpp'
LIB = Path(os.environ.get('UNCRASHED_SCANRASTER', REPO / 'native' / 'libscanraster.so'))
_lib = None


class NativeError(RuntimeError):
    pass


def library():
    """The loaded helper library; rebuilt when its source is newer (needs g++, OpenMP and libturbojpeg)."""
    global _lib
    if _lib is not None:
        return _lib
    if not (LIB.is_file() and (not SRC.is_file() or LIB.stat().st_mtime >= SRC.stat().st_mtime)):
        if not shutil.which('g++'):
            raise NativeError(f'{LIB} is missing and g++ is not installed to build it from {SRC}')
        tmp = LIB.with_name(f'.{LIB.name}.{os.getpid()}')
        cmd = ['g++', '-O3', '-fopenmp', '-std=c++17', '-shared', '-fPIC', str(SRC), '-lturbojpeg', '-o', str(tmp)]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode:
            tmp.unlink(missing_ok=True)
            raise NativeError('building scanraster failed (install g++ with OpenMP and libturbojpeg headers?): '
                              + done.stderr.strip()[-400:])
        tmp.replace(LIB)
    lib = ctypes.CDLL(str(LIB))
    P, I, D = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    lib.sr_raster.argtypes = [ctypes.c_char_p, P, P, D, D, D, I, I, I, P, P, P, P, P, ctypes.c_char_p, I]
    lib.sr_raster.restype = I
    lib.sr_label.argtypes = [P, I, I, I, P]
    lib.sr_label.restype = I
    lib.sr_nearest.argtypes = [P, I, I, P]
    lib.sr_nearest.restype = None
    lib.sr_contour.argtypes = [P, I, I, ctypes.c_int32, I, I, P, I]
    lib.sr_contour.restype = I
    lib.sr_fill_poly.argtypes = [P, I, I, P, P, I, ctypes.c_int32]
    lib.sr_fill_poly.restype = None
    lib.sr_facades.argtypes = [ctypes.c_char_p, P, P, I, P, P, P, P, P, P, D, P, ctypes.c_char_p, I]
    lib.sr_facades.restype = I
    lib.sr_views.argtypes = [ctypes.c_char_p, P, P, I, P, I, I, P, P, P, P, ctypes.c_char_p, I]
    lib.sr_views.restype = I
    lib.sr_lzw.argtypes = [P, ctypes.c_int64, P, ctypes.c_int64]
    lib.sr_lzw.restype = ctypes.c_int64
    _lib = lib
    return lib


def _p(a: np.ndarray):
    return a.ctypes.data_as(ctypes.c_void_p)


def raster(index: Path, axes: list[tuple[int, float]], x0: float, y0: float, res: float, w: int, h: int, *,
           normals: bool = True, low: bool = True, rgb: bool = True) -> dict:
    """Rasterise a ucmesh index onto the grid. Returns dsm (NaN where empty), and optionally low, nrm (h,w,3 east,
    south, up), rgb (h,w,3 uint8), plus stats."""
    lib = library()
    src = np.array([a[0] for a in axes], np.int32)
    sign = np.array([a[1] for a in axes], np.float32)
    out = {'dsm': np.empty((h, w), np.float32)}
    if low:
        out['low'] = np.empty((h, w), np.float32)
    if normals:
        out['nrm'] = np.empty((h, w, 3), np.float32)
    if rgb:
        out['rgb'] = np.empty((h, w, 3), np.uint8)
    stats = np.zeros(5, np.float64)
    err = ctypes.create_string_buffer(512)
    code = lib.sr_raster(str(index).encode(), _p(src), _p(sign), float(x0), float(y0), float(res), int(w), int(h), 1,
                         _p(out['dsm']), _p(out['low']) if low else None, _p(out['nrm']) if normals else None,
                         _p(out['rgb']) if rgb else None, _p(stats), err, len(err))
    if code:
        raise NativeError('scanraster: ' + err.value.decode(errors='replace'))
    out['stats'] = {'triangles': int(stats[0]), 'culled_finer': int(stats[1]), 'meshes': int(stats[2]),
                    'decode_s': round(float(stats[3]), 2), 'raster_s': round(float(stats[4]), 2)}
    return out


def label(mask: np.ndarray, conn8: bool = True) -> tuple[np.ndarray, int]:
    m = np.ascontiguousarray(mask, dtype=np.uint8)
    lab = np.empty(m.shape, np.int32)
    n = library().sr_label(_p(m), m.shape[1], m.shape[0], int(conn8), _p(lab))
    return lab, int(n)


def nearest(seed: np.ndarray) -> np.ndarray:
    """Flat index of a nearby seed cell for every cell (-1 when there is no seed)."""
    m = np.ascontiguousarray(seed, dtype=np.uint8)
    idx = np.empty(m.shape, np.int32)
    library().sr_nearest(_p(m), m.shape[1], m.shape[0], _p(idx))
    return idx


def contour(lab: np.ndarray, value: int, r0: int, c0: int, cap: int = 200000) -> np.ndarray:
    """Outer boundary cells (col,row) of the component `value` whose first raster cell is (r0, c0)."""
    lab = np.ascontiguousarray(lab, dtype=np.int32)
    out = np.empty((cap, 2), np.int32)
    n = library().sr_contour(_p(lab), lab.shape[1], lab.shape[0], int(value), int(r0), int(c0), _p(out), cap)
    return out[:n].copy()


def fill_poly(img: np.ndarray, xs, ys, value: int) -> None:
    """Even-odd fill in place (img int32, C-contiguous); xs/ys in pixel units, cell centres at integers."""
    assert img.dtype == np.int32 and img.flags['C_CONTIGUOUS']
    x = np.ascontiguousarray(xs, np.float64)
    y = np.ascontiguousarray(ys, np.float64)
    library().sr_fill_poly(_p(img), img.shape[1], img.shape[0], _p(x), _p(y), len(x), int(value))


def views(index: Path, axes: list[tuple[int, float]], cams: np.ndarray, w: int, h: int, *, rgb: bool = True,
          skip: np.ndarray | None = None) -> tuple[np.ndarray | None, np.ndarray, dict]:
    """Perspective renders (cams: N x 8 east, south, up, yaw, pitch, fov_h degrees, near, far; see sr_views).
    Returns rgb (N, h, w, 3) uint8 or None, depth (N, h, w) float32 view-axis metres (NaN = nothing), stats."""
    lib = library()
    cams = np.ascontiguousarray(cams, np.float64).reshape(-1, 8)
    n = len(cams)
    img = np.empty((max(n, 1), h, w, 3), np.uint8) if rgb else None
    dep = np.empty((max(n, 1), h, w), np.float32)
    src = np.array([a[0] for a in axes], np.int32)
    sign = np.array([a[1] for a in axes], np.float32)
    sk = np.ascontiguousarray(skip, np.uint8) if skip is not None else None
    stats = np.zeros(4, np.float64)
    err = ctypes.create_string_buffer(512)
    if n:
        code = lib.sr_views(str(index).encode(), _p(src), _p(sign), n, _p(cams), int(w), int(h),
                            _p(sk) if sk is not None else None, _p(img) if rgb else None, _p(dep), _p(stats), err, len(err))
        if code:
            raise NativeError('scanraster: ' + err.value.decode(errors='replace'))
    return (img[:n] if rgb else None), dep[:n], {'triangles': int(stats[0]), 'views': n,
                                                 'decode_s': round(float(stats[2]), 2), 'render_s': round(float(stats[3]), 2)}


def lzw(data: bytes, size: int) -> bytes:
    """Decode a TIFF LZW strip / tile of `size` decoded bytes."""
    src = np.frombuffer(data, np.uint8)
    out = np.empty(max(1, size), np.uint8)
    n = library().sr_lzw(_p(src), len(src), _p(out), size)
    if n < 0:
        raise NativeError('corrupt LZW data')
    return out[:n].tobytes()


def facades(index: Path, axes: list[tuple[int, float]], planes: np.ndarray, dims: np.ndarray, *,
            skip: np.ndarray | None = None, min_facing: float = 0.35) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    """Render wall planes (planes: N x 12, see sr_facades; dims: N x 2 W, H). Returns per facade an rgb (H, W, 3)
    uint8 image, a depth (H, W) float32 image (NaN = nothing drawn) and stats."""
    lib = library()
    planes = np.ascontiguousarray(planes, np.float64).reshape(-1, 12)
    dims = np.ascontiguousarray(dims, np.int32).reshape(-1, 2)
    n = len(planes)
    sizes = dims[:, 0].astype(np.int64) * dims[:, 1]
    offs = np.ascontiguousarray(np.r_[0, np.cumsum(sizes)[:-1]] if n else np.zeros(0), np.int64)
    total = int(sizes.sum()) if n else 0
    rgb = np.empty(max(total, 1) * 3, np.uint8)
    dep = np.empty(max(total, 1), np.float32)
    src = np.array([a[0] for a in axes], np.int32)
    sign = np.array([a[1] for a in axes], np.float32)
    sk = np.ascontiguousarray(skip, np.uint8) if skip is not None else None
    stats = np.zeros(4, np.float64)
    err = ctypes.create_string_buffer(512)
    if n:
        code = lib.sr_facades(str(index).encode(), _p(src), _p(sign), n, _p(planes), _p(dims), _p(offs), _p(rgb), _p(dep),
                              _p(sk) if sk is not None else None, float(min_facing), _p(stats), err, len(err))
        if code:
            raise NativeError('scanraster: ' + err.value.decode(errors='replace'))
    imgs, deps = [], []
    for k in range(n):
        w, h = int(dims[k, 0]), int(dims[k, 1])
        o = int(offs[k])
        imgs.append(rgb[o * 3:(o + w * h) * 3].reshape(h, w, 3))
        deps.append(dep[o:o + w * h].reshape(h, w))
    return imgs, deps, {'triangles': int(stats[0]), 'facades': n, 'decode_s': round(float(stats[2]), 2),
                        'render_s': round(float(stats[3]), 2)}
