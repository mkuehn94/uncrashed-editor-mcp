"""Measured catalogue asset profiles (asset_profile): what a row looks like from above, not its bounding box.

A catalogue row's bounding box is what spawn scaling works on, but it is not what a scan measures. A game tree's box
reaches the tips of its outermost leaf cards; the crown a scan sees is the dense part (and photogrammetry crowns are
solid). Scaling the box to a scan crown therefore builds trees that look clearly smaller than the scan. A profile
measures every row the way the survey measures a scan object:

- top silhouette at about 1/48 of the row's width: cells covered by at least one layer of geometry (foliage: two
  layers of leaf cards, so a single stray card does not count), the visible footprint;
- crown_m / visible size: equivalent diameter of that silhouette (trees, as the survey's crown_m), extents along the
  visible long axis and across (as the survey's oriented box), height as the 95th percentile of the top surface
  over the silhouette (as the survey's height_m) above the mesh bottom;
- long_axis_deg: direction of the visible long axis in the row's own frame (yaw convention, 0..180);
- symmetry: the top height field compared with itself turned by 90 / 180 degrees and mirrored. yaw_period_deg is
  what a yaw error is measured modulo (0 = round: any yaw fits, 90 = square, 180 = symmetric box, 360 = none);
- front: the engine convention (+X) for rows without symmetry; it is not measured;
- panel: tilt and the direction the dominant upward face looks to, for panel-like rows;
- crown_base_m for foliage: the height where the crown starts (the trunk below is narrow).

Geometry comes from the cooked LOD0 in the game's pak files when they are installed (meshextract.read_lod0), else
from the simplified hull in data/meshes.bin (snapped to a coarse grid for large meshes: rougher), else the row has no
mesh (Blueprint) and the profile says so. Profiles are cached as JSON per catalogue version.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..catalog import load_catalog
from . import store

VERSION = 3
PAKS: Path | None = None        # None: meshextract.PAK_DIR; tests point it to a missing folder
CELLS = 48                      # silhouette cells across the wider horizontal side
SYM_MATCH = 0.82                # share of matching cells for a symmetry
PANEL_SHARE = 0.6               # share of the top area facing one way for a panel


def cache_dir() -> Path:
    return store.SCAN_HOME.parent / 'asset-profiles'


def _pak_dir() -> Path:
    from .. import meshextract
    return PAKS if PAKS is not None else meshextract.PAK_DIR


@lru_cache(maxsize=1)
def catalogue_version() -> str:
    """Catalogue rows + hull file + installed paks (names, sizes, mtimes) + profile rules."""
    from .. import catalog, meshes
    h = hashlib.sha1(f'v{VERSION}'.encode())
    h.update((catalog.DATA / 'assets.json').read_bytes())
    if meshes.DATA.is_file():
        st = meshes.DATA.stat()
        h.update(f'{st.st_size}:{int(st.st_mtime)}'.encode())
    paks = _pak_dir()
    if paks.is_dir():
        for p in sorted(paks.glob('*.pak')):
            st = p.stat()
            h.update(f'{p.name}:{st.st_size}:{int(st.st_mtime)}'.encode())
    return h.hexdigest()[:12]


@lru_cache(maxsize=1)
def _paks():
    from .. import meshextract
    paks = _pak_dir()
    if not paks.is_dir() or not any(paks.glob('*.pak')):
        return None
    old = meshextract.PAK_DIR
    try:
        meshextract.PAK_DIR = paks
        return meshextract._index_paks()
    except (OSError, RuntimeError, ValueError):
        return None
    finally:
        meshextract.PAK_DIR = old


def _geometry(a) -> tuple[np.ndarray, np.ndarray, str] | None:
    """(vertices in metres relative to the pivot, triangles, source) or None."""
    from .. import meshes, meshextract
    if a.kind != 'mesh' or not a.source:
        return None
    stems = _paks()
    if stems is not None:
        for pak, entry in stems.get(a.source) or []:
            try:
                got = meshextract.read_lod0(pak.extract(entry), a.origin_cm, a.extent_cm)
            except (OSError, ValueError, RuntimeError, IndexError, struct.error):
                got = None
            if got:
                V = np.asarray(got[0], float) / 100.0
                T = np.asarray(got[1], np.int64).reshape(-1, 3)
                if len(T) and T.max() < len(V):
                    return V, T, 'lod0 (game paks)'
    raw = meshes.load().get(a.source)
    if raw is not None and raw.tris:
        return np.asarray(raw.verts, float), np.asarray(raw.tris, np.int64), 'hull (data/meshes.bin)'
    return None


def _samples(V: np.ndarray, T: np.ndarray, res: float):
    """Points on the triangles (about 8 per cell area) with the projected top area each point stands for."""
    p0, p1, p2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    area = np.linalg.norm(n, axis=1) / 2
    ok = area > 1e-9
    p0, p1, p2, n, area = p0[ok], p1[ok], p2[ok], n[ok], area[ok]
    nz = np.abs(n[:, 2]) / (2 * area)
    cell = res * res / 8
    k = np.clip(np.ceil(area / cell), 1, 4000).astype(np.int64)
    total = int(k.sum())
    if total > 3_000_000:   # very large meshes: fewer points, same coverage weights
        k = np.maximum(1, (k * 3_000_000 // total))
    idx = np.repeat(np.arange(len(area)), k)
    rng = np.random.default_rng(11)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1
    u, v = np.where(flip, 1 - u, u), np.where(flip, 1 - v, v)
    pts = p0[idx] + u[:, None] * (p1[idx] - p0[idx]) + v[:, None] * (p2[idx] - p0[idx])
    w = (area * nz / k)[idx]
    nrm = n[idx] / (2 * area[idx, None])
    nrm *= np.where(nrm[:, 2:3] < 0, -1, 1)   # look from above: winding of cooked / simplified meshes varies
    return pts, w, nrm, area[idx] / k[idx]


def _sym(hf: np.ndarray, occ: np.ndarray, tol: float, turn) -> float:
    """Share of the occupied cells (of either) whose height matches after the transform `turn`."""
    h2, o2 = turn(hf), turn(occ)
    if h2.shape != hf.shape:
        return 0.0
    both = occ & o2
    either = occ | o2
    if not either.any():
        return 0.0
    return float((both & (np.abs(hf - h2) <= tol)).sum() / either.sum())


def measure(a) -> dict:
    """Profile of one catalogue asset (uncached)."""
    out = {'row': a.id, 'name': a.name, 'aabb_m': [round(float(v), 3) for v in a.size_m], 'source': None}
    foliage = 'tree' in a.tags or 'bush' in a.tags or 'grass' in a.tags
    out['kind'] = 'foliage' if foliage else 'solid'
    geo = _geometry(a)
    if geo is None:
        out.update(source='none', note='Blueprint or no extracted mesh: visible size = bounding box, symmetry '
                                       'unknown', visible_m=out['aabb_m'], long_axis_deg=0.0 if a.size_m[0] >= a.size_m[1]
                   else 90.0, symmetry='unknown', yaw_period_deg=180.0 if a.size_m[0] != a.size_m[1] else 90.0,
                   front='+x (engine convention, not measured)', up='+z', visible_frac=None, crown_m=None,
                   crown_frac=None, crown_base_m=None, panel=None, footprint=None, visible_center_m=[0.0, 0.0],
                   visible_height_m=round(float(a.size_m[2]), 3), top_m=round(float(a.size_m[2]), 3))
        return out
    V, T, source = geo
    lo, hi = V.min(0), V.max(0)
    span = hi - lo
    res = float(min(max(max(span[0], span[1]) / CELLS, 0.03), 1.0))
    pts, w, nrm, sarea = _samples(V, T, res)
    allp = np.vstack([pts, V])
    c = np.floor((allp[:, :2] - lo[:2]) / res).astype(np.int64)
    nx, ny = int(c[:, 0].max()) + 1, int(c[:, 1].max()) + 1
    cov = np.zeros((nx, ny))
    np.add.at(cov, (c[:len(pts), 0], c[:len(pts), 1]), w)
    layers = cov / (res * res)
    hf = np.full((nx, ny), -np.inf)
    np.maximum.at(hf, (c[:, 0], c[:, 1]), allp[:, 2])
    occ = layers >= (1.5 if foliage else 0.25)
    if occ.sum() < 4:
        occ = layers > 0
    hf_occ = np.where(occ, hf - lo[2], 0.0)
    ci, cj = np.nonzero(occ)
    X = lo[0] + (ci + .5) * res
    Y = lo[1] + (cj + .5) * res
    A = float(occ.sum()) * res * res
    # visible long axis (PCA of the silhouette), extents along it and across
    P = np.c_[X, Y]
    ctr = P.mean(0)
    cov2 = np.cov((P - ctr).T) if len(P) > 2 else np.eye(2)
    evals, evecs = np.linalg.eigh(cov2)
    ax = evecs[:, 1]
    ang = math.degrees(math.atan2(ax[1], ax[0])) % 180
    if evals[1] < 1.15 * max(evals[0], 1e-12):   # square or round: no long axis, keep the row's own axes
        ang = 0.0 if span[0] >= span[1] else 90.0
    for snap in (0.0, 90.0, 180.0):   # a box-like row keeps its own axes unless the silhouette clearly turns
        if abs(ang - snap) < 8:
            ang = snap % 180
    t = math.radians(ang)
    u = (P - ctr) @ np.array([math.cos(t), math.sin(t)])
    v = (P - ctr) @ np.array([-math.sin(t), math.cos(t)])
    L = float(np.percentile(u, 99.5) - np.percentile(u, 0.5)) + res
    W = float(np.percentile(v, 99.5) - np.percentile(v, 0.5)) + res
    vx = float(np.percentile(X, 99.5) - np.percentile(X, 0.5)) + res
    vy = float(np.percentile(Y, 99.5) - np.percentile(Y, 0.5)) + res
    tops = hf_occ[occ]
    h95 = float(np.percentile(tops, 95))
    crown = 2 * math.sqrt(A / math.pi)
    bbox_mid = (lo[:2] + hi[:2]) / 2
    out.update(source=source, triangles=int(len(T)), cell_m=round(res, 3),
               visible_m=[round(vx, 3), round(vy, 3), round(h95, 3)],
               visible_oriented_m=[round(L, 3), round(W, 3)],
               long_axis_deg=round(ang, 1), visible_height_m=round(h95, 3), top_m=round(float(span[2]), 3),
               visible_frac=[round(float(vx / max(span[0], 1e-6)), 3), round(float(vy / max(span[1], 1e-6)), 3),
                             round(float(h95 / max(span[2], 1e-6)), 3)],
               visible_center_m=[round(float(ctr[0] - bbox_mid[0]), 3), round(float(ctr[1] - bbox_mid[1]), 3)],
               top_area_m2=round(A, 2), up='+z')
    # footprint shape
    fill = A / max(L * W, 1e-6)
    aspect = L / max(W, 1e-6)
    out['footprint'] = {'fill': round(fill, 2), 'aspect': round(aspect, 2),
                        'shape': 'rect' if fill > .85 else 'round' if fill > .68 and aspect < 1.25 else 'irregular'}
    # symmetry of the top height field about the silhouette centre
    hc = np.where(occ, hf_occ, 0.0)
    pad = max(nx, ny)
    big_h = np.zeros((2 * pad + 1, 2 * pad + 1))
    big_o = np.zeros_like(big_h, bool)
    oi, oj = int(round((ctr[0] - lo[0]) / res - .5)), int(round((ctr[1] - lo[1]) / res - .5))
    big_h[pad - oi:pad - oi + nx, pad - oj:pad - oj + ny] = hc
    big_o[pad - oi:pad - oi + nx, pad - oj:pad - oj + ny] = occ
    tol = max(0.12 * h95, 0.05)
    s180 = _sym(big_h, big_o, tol, lambda m: m[::-1, ::-1])
    s90 = _sym(big_h, big_o, tol, lambda m: np.rot90(m))
    mx = _sym(big_h, big_o, tol, lambda m: m[::-1, :])
    my = _sym(big_h, big_o, tol, lambda m: m[:, ::-1])
    roundish = fill > .68 and aspect < 1.25 and out['footprint']['shape'] != 'rect'
    if foliage or (roundish and s90 >= SYM_MATCH):
        sym, period = 'round', 0.0
    elif s90 >= SYM_MATCH and s180 >= SYM_MATCH:
        sym, period = 'square', 90.0
    elif s180 >= SYM_MATCH:
        sym, period = 'twofold', 180.0
    else:
        sym, period = 'none', 360.0
    out.update(symmetry=sym, yaw_period_deg=period,
               symmetry_scores={'turn180': round(s180, 2), 'turn90': round(s90, 2), 'mirror_x': round(mx, 2),
                                'mirror_y': round(my, 2)},
               front='+x (engine convention, not measured)' if period == 360 else None)
    # dominant upward face: a panel when most of the top area looks one way and is tilted
    up = nrm[:, 2] > 0.3
    panel = None
    if up.any():
        wt = sarea[up] * nrm[up, 2]
        mean = (nrm[up] * wt[:, None]).sum(0)
        mean /= np.linalg.norm(mean)
        agree = float(wt[(nrm[up] @ mean) > math.cos(math.radians(6))].sum() / wt.sum())
        tilt = math.degrees(math.acos(min(1.0, float(mean[2]))))
        if agree >= PANEL_SHARE and 2.0 <= tilt <= 75 and not foliage:
            panel = {'tilt_deg': round(tilt, 1), 'faces_deg': round(math.degrees(math.atan2(mean[1], mean[0])) % 360, 1),
                     'share': round(agree, 2)}
    out['panel'] = panel
    if foliage:
        out['crown_m'] = round(crown, 3)
        out['crown_frac'] = round(float(crown / max((span[0] + span[1]) / 2, 1e-6)), 3)
        z = pts[:, 2] - lo[2]
        nb = 40
        edges = np.linspace(0, span[2], nb + 1)
        spread = np.zeros(nb)
        for i in range(nb):
            s = pts[(z >= edges[i]) & (z < edges[i + 1])]
            if len(s) >= 12:
                q = np.percentile(s[:, :2], [5, 95], axis=0)
                spread[i] = float((q[1] - q[0]).mean())
        base = float(edges[int(np.argmax(spread > .5 * spread.max()))]) if spread.max() > 0 else 0.0
        out['crown_base_m'] = round(base, 2)
    else:
        out.update(crown_m=None, crown_frac=None, crown_base_m=None)
    return out


_MEM: dict[str, dict] = {}


def _cache_file() -> Path:
    return cache_dir() / f'{catalogue_version()}.json'


def _load_cache() -> dict:
    path = _cache_file()
    key = str(path)
    if key not in _MEM:
        try:
            _MEM[key] = json.loads(path.read_text())
        except (OSError, ValueError):
            _MEM[key] = {}
    return _MEM[key]


def _save_cache(data: dict) -> None:
    path = _cache_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.tmp{time.time_ns()}')
    tmp.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False))
    tmp.replace(path)


def get(row: str, refresh: bool = False) -> dict | None:
    """Cached profile of one row; None when the row is unknown."""
    a = load_catalog().assets.get(row)
    if a is None:
        return None
    data = _load_cache()
    if refresh or row not in data:
        data[row] = measure(a)
        _save_cache(data)
    return data[row]


def many(rows, refresh: bool = False) -> dict[str, dict]:
    cat = load_catalog().assets
    data = _load_cache()
    changed = False
    for r in rows:
        if r in cat and (refresh or r not in data):
            data[r] = measure(cat[r])
            changed = True
    if changed:
        _save_cache(data)
    return {r: data[r] for r in rows if r in data}


def asset_profile(rows, refresh: bool = False) -> dict:
    cat = load_catalog().assets
    wanted = [str(r).strip() for r in rows or [] if str(r).strip()]
    if not wanted:
        raise ValueError('give catalogue row ids (search_assets / get_assets list them)')
    unknown = [r for r in wanted if r not in cat]
    if unknown:
        raise ValueError(f"unknown catalogue rows: {', '.join(unknown)} (search_assets finds rows)")
    if len(wanted) > 200:
        raise ValueError('at most 200 rows per call')
    t0 = time.time()
    got = many(wanted, refresh)
    rows_out = []
    for r in wanted:
        p = got[r]
        rows_out.append({'row': r, 'name': p['name'], 'source': p['source'], 'aabb_m': p['aabb_m'],
                         'visible_m': p['visible_m'], 'visible_frac': p.get('visible_frac'),
                         'long_axis_deg': p['long_axis_deg'], 'symmetry': p['symmetry'],
                         'yaw_period_deg': p['yaw_period_deg'], 'crown_m': p.get('crown_m'),
                         'crown_frac': p.get('crown_frac'), 'crown_base_m': p.get('crown_base_m'),
                         'panel': (f"tilt {p['panel']['tilt_deg']:g} faces {p['panel']['faces_deg']:g}"
                                   if p.get('panel') else None),
                         'footprint': p['footprint']['shape'] if p.get('footprint') else None})
    return {'rows': rows_out, 'catalogue_version': catalogue_version(), 'cache': str(_cache_file()),
            'seconds': round(time.time() - t0, 2), 'units': 'metres',
            'meaning': 'row frame: x forward, y right (yaw 0 lays x on map +x), z up. visible_m: extents of the '
                       'top silhouette along x, y and the 95th percentile top height (as a scan measures objects); '
                       'visible_frac: visible / bounding box per axis; long_axis_deg: visible long axis in the row '
                       'frame; yaw_period_deg: a yaw error counts modulo this (0 round, 90 square, 180 symmetric, '
                       '360 none: front = +x by engine convention); crown_m: equivalent diameter of the top '
                       'silhouette (foliage; a scan crown_m means the same); crown_frac: crown_m / box width; '
                       'panel: tilt and the direction the dominant upward face looks to, row frame',
            'limitations': ['Geometry only: leaf cards and other alpha-masked textures count as covered, so foliage '
                            'crowns are an upper bound of what renders (two card layers are required per cell).',
                            'front is the engine convention (+X) and is not measured; vehicles and benches can be '
                            'modelled facing another way.',
                            'Rows without extracted geometry (Blueprints) fall back to the bounding box.']}
