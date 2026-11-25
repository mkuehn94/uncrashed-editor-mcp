"""Helpers for scan_compare on real builds: which placed objects belong together (group tags such as b104,
car.1002, scan.building.75, b57nw), which are ground cover (tiles, markings, decals, lawn foliage) or gameplay
pieces, and a height raster of a multi-object building (walls, roof slabs, triangle gables) compared with the
scan building's cells: footprint IoU, coverage, missing / extra area, median height difference.

Pure functions over placed-object dicts (scene_snapshot rows: actor, row, group, xf) and the survey rasters."""
from __future__ import annotations

import math
import re
from functools import lru_cache

import numpy as np

from .. import fit

# ---------------------------------------------------------------------------- group tags

# words a builder may put before a scan id in a group name -> the scan classes it may stand for (None: any)
TAG_WORDS = {
    'b': ('building',), 'bld': ('building',), 'bldg': ('building',), 'building': ('building',),
    'house': ('building',), 'hall': ('building',),
    'car': ('car', 'truck'), 'truck': ('car', 'truck'), 'van': ('car', 'truck'), 'vehicle': ('car', 'truck'),
    'veh': ('car', 'truck'),
    'tree': ('tree', 'shrub'), 'shrub': ('shrub', 'tree'), 'bush': ('shrub', 'tree'), 'hedge': ('shrub', 'tree'),
    'scan': None, 'prop': None, 'obj': None, 'object': None,
}
_TAG = re.compile(r'^(?:scan[._])?(?P<word>[a-z]+(?:_[a-z]+)*)[._-]?(?P<id>\d+)(?:[._-]?[a-z][a-z0-9_]*)?$', re.I)


def parse_tag(group: str, classes: set[str] | frozenset = frozenset()) -> tuple[str, int] | None:
    """(word, scan id) of a group name that names a scan object: scan.<class>.<id>, <class>.<id>, b<id>, b<id>nw,
    building_75, ... The word must be a known alias (TAG_WORDS) or a scan class name; None otherwise."""
    m = _TAG.match(group or '')
    if not m:
        return None
    word = m.group('word').lower()
    if word not in TAG_WORDS and word not in classes:
        return None
    return word, int(m.group('id'))


STRUCTURES = ('building', 'building_part', 'bridge', 'low_structure')   # what a builder may call a building


def tag_fits(word: str, cls: str) -> bool:
    allowed = TAG_WORDS.get(word, (word,))
    return allowed is None or cls in allowed or ('building' in allowed and cls in STRUCTURES)


# ---------------------------------------------------------------------------- kinds of placed objects

GROUND_TOP_M = 0.6      # ... top at most this far above the scan ground
COVER_H_M = 1.0         # low foliage (grass, ground cover) up to this height counts as ground


def kind_of(a, z_extent: float, top_above_ground: float | None) -> str:
    """'gameplay' (spawners, checkpoints, coins), 'ground' (tiles, road markings, decals, lawn foliage: thin and on
    the ground) or 'object'. a: catalogue asset (None: unknown row)."""
    if a is None:
        return 'object'
    role = fit.role_of(a)
    name = a.name.lower()
    if a.category == 'Interractive' or role == 'marker' or any(w in name for w in ('spawner', 'checkpoint', 'gate')):
        return 'gameplay'
    if role in ('surface', 'decal'):
        return 'ground'
    near = top_above_ground is not None and top_above_ground <= GROUND_TOP_M
    # a slab, tile or marking whose top is at the ground (however deep it reaches down: foundation fills) is ground;
    # small props and vehicles stay objects
    small = max(a.size_m[0], a.size_m[1]) < 1.5 and role in ('prop', 'industrial', 'street', 'barrier')
    if near and role not in ('vehicle', 'foliage', 'figure') and not small:
        return 'ground'
    if role == 'foliage' and 'grass' in name and top_above_ground is not None and top_above_ground <= COVER_H_M:
        return 'ground'
    return 'object'


# ---------------------------------------------------------------------------- world geometry


def rotation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Columns = local x, y, z axes in the world (Unreal FRotationMatrix, degrees)."""
    p, y, r = (math.radians(v) for v in (pitch, yaw, roll))
    cp, sp, cy, sy, cr, sr = math.cos(p), math.sin(p), math.cos(y), math.sin(y), math.cos(r), math.sin(r)
    X = np.array([cp * cy, cp * sy, sp])
    Y = np.array([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp])
    Z = np.array([-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp])
    return np.stack([X, Y, Z], axis=1)


_BOX_Q = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))


@lru_cache(maxsize=1024)
def _local_tris(row: str, source: str | None, origin: tuple, extent: tuple) -> np.ndarray:
    """Triangles of a row in its local frame (metres, before scale): the simplified hull from data/meshes.bin, else
    the catalogue box."""
    if source:
        from .. import meshes
        raw = meshes.load().get(source)
        if raw is not None and raw.tris:
            V = np.asarray(raw.verts, float)
            return V[np.asarray(raw.tris, int)]
    o, e = np.asarray(origin, float) / 100, np.asarray(extent, float) / 100
    V = np.array([o + e * (sx, sy, sz) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)])
    T = []
    for a, b, c, d in _BOX_Q:
        T += [(a, b, c), (a, c, d)]
    return V[np.asarray(T)]


def world_tris(o: dict, a) -> np.ndarray:
    """(n, 3, 3) world triangles (map metres) of a placed object."""
    xf = o['xf']
    T = _local_tris(a.id, a.source if a.kind == 'mesh' else None, tuple(a.origin_cm), tuple(a.extent_cm))
    S = np.asarray(xf[6:9], float)
    R = rotation(float(xf[3]), float(xf[4]), float(xf[5]))
    return (T * S) @ R.T + np.asarray(xf[:3], float) / 100


@lru_cache(maxsize=256)
def _lattice(n: int) -> np.ndarray:
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
    keep = (i + j) <= n
    return np.stack([i[keep], j[keep]], 1) / n


def top_points(tris: np.ndarray, step: float) -> np.ndarray:
    """Points (x, y, z) on the upward faces of a mesh, at most `step` apart: what a top-down height raster sees.
    Faces turned away from the mesh centre or steeper than ~78 degrees are left out (a thin wall's top face carries
    its top)."""
    if not len(tris):
        return np.zeros((0, 3))
    centre = tris.reshape(-1, 3).mean(0)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm = np.linalg.norm(n, axis=1)
    ok = norm > 1e-9
    n[ok] /= norm[ok, None]
    out_dir = tris.mean(1) - centre
    n *= np.where((n * out_dir).sum(1, keepdims=True) < 0, -1, 1)   # hull winding is not reliable: face outwards
    up = ok & (n[:, 2] > 0.2)
    if not up.any():
        up = ok & (np.abs(n[:, 2]) > 0.2)
    T = tris[up]
    edge = np.max(np.linalg.norm(T - np.roll(T, 1, axis=1), axis=2), axis=1)
    k = np.clip(np.ceil(edge / step), 1, 400).astype(int)
    pts = []
    for kk in np.unique(k):
        L = _lattice(int(kk))
        sel = T[k == kk]
        a, b, c = sel[:, 0], sel[:, 1], sel[:, 2]
        P = a[:, None] + L[None, :, :1] * (b - a)[:, None] + L[None, :, 1:] * (c - a)[:, None]
        pts.append(P.reshape(-1, 3))
    return np.concatenate(pts) if pts else np.zeros((0, 3))


# ---------------------------------------------------------------------------- building rasters

RES_M = 0.5          # building comparison lattice
RAISED_M = 1.5       # height above ground that counts as building (lower for low buildings, see raised_threshold)


def raised_threshold(scan_height: float) -> float:
    return max(0.6, min(RAISED_M, 0.5 * float(scan_height or RAISED_M * 2)))


def _in_poly(x: np.ndarray, y: np.ndarray, poly) -> np.ndarray:
    """Even-odd point-in-polygon test, vectorised over points."""
    inside = np.zeros(x.shape, bool)
    P = np.asarray(poly, float)
    for (x0, y0), (x1, y1) in zip(P, np.roll(P, -1, axis=0)):
        if y0 == y1:
            continue
        cross = (y0 > y) != (y1 > y)
        xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= cross & (x < xi)
    return inside


SHIFT_M = 3.0        # building shift search radius
SHIFT_GAIN = 0.08    # an IoU gain that makes a shift an offset


def _best_shift(have: np.ndarray, scan: np.ndarray, res: float) -> tuple[float, float, float]:
    """(dx, dy, iou) of the whole-cell shift of the built mask within SHIFT_M that overlaps the scan mask best."""
    k = int(round(SHIFT_M / res))
    best = (0.0, 0.0, -1.0)
    H, W = have.shape
    for dj in range(-k, k + 1):
        for di in range(-k, k + 1):
            sh = np.zeros_like(have)
            sh[max(dj, 0):H + min(dj, 0), max(di, 0):W + min(di, 0)] = \
                have[max(-dj, 0):H + min(-dj, 0), max(-di, 0):W + min(-di, 0)]
            u = (sh | scan).sum()
            v = float((sh & scan).sum() / u) if u else 0.0
            if v > best[2] + 1e-9 or (abs(v - best[2]) <= 1e-9 and math.hypot(di, dj) < math.hypot(*best[:2]) / res):
                best = (di * res, dj * res, v)
    return best


def compare_building(ras, fr, s: dict, pts: np.ndarray, res: float = RES_M, clip=None) -> dict:
    """Height-raster comparison of one scan building (s: map-frame record with 'o', 'x', 'y') with the top points of
    the placed objects that stand for it (map metres). clip (x0, y0, x1, y1): compare only inside this box (a
    partial build of a large building). Returns iou, coverage (share of the scan building the build covers),
    precision (share of the build over the scan building), missing / extra m2, signed and absolute median height
    difference on the shared cells, p90, built / scan top above ground, and best_shift_m: the move of the build
    within SHIFT_M that overlaps the scan best, with the iou it would reach."""
    o = s['o']
    reach = math.hypot(o['length_m'], o['width_m']) / 2 + 3
    x0, x1, y0, y1 = s['x'] - reach, s['x'] + reach, s['y'] - reach, s['y'] + reach
    if len(pts):
        x0, y0 = min(x0, float(pts[:, 0].min()) - 1), min(y0, float(pts[:, 1].min()) - 1)
        x1, y1 = max(x1, float(pts[:, 0].max()) + 1), max(y1, float(pts[:, 1].max()) + 1)
    # keep the lattice bounded (a stray piece far away must not make a huge raster)
    span = 4 * reach + 40
    x0, x1 = max(x0, s['x'] - span), min(x1, s['x'] + span)
    y0, y1 = max(y0, s['y'] - span), min(y1, s['y'] + span)
    nx, ny = max(1, int(math.ceil((x1 - x0) / res))), max(1, int(math.ceil((y1 - y0) / res)))
    fx, fy = x0 + (np.arange(nx) + .5) * res, y0 + (np.arange(ny) + .5) * res
    X, Y = np.meshgrid(fx, fy)
    e, sn = fr.to_scan(X.ravel(), Y.ravel())
    r, c, inside = ras.cell(e, sn)
    oid = np.asarray(ras.objid[r, c]).reshape(X.shape)
    dsm = np.asarray(ras.dsm[r, c], float).reshape(X.shape)
    dtm = np.asarray(ras.dtm[r, c], float).reshape(X.shape)
    inside = inside.reshape(X.shape)
    if clip is not None:
        inside &= (X >= clip[0]) & (X <= clip[2]) & (Y >= clip[1]) & (Y <= clip[3])
    gnd_map = fr.z_out(dtm)
    thr = raised_threshold(o.get('height_m'))
    with np.errstate(invalid='ignore'):
        raised = inside & (dsm - dtm > thr)
    scan = raised & (oid == o['id'])
    source = 'cells'
    fp = o.get('footprint')
    if fp and len(fp) >= 3:          # few labelled cells (a building the classes split off): its footprint
        P = np.asarray(fp, float)
        poly_area = 0.5 * abs(float(np.dot(P[:, 0], np.roll(P[:, 1], -1)) - np.dot(P[:, 1], np.roll(P[:, 0], -1))))
        if scan.sum() * res * res < 0.3 * poly_area:
            es, ss = e.reshape(X.shape), sn.reshape(X.shape)
            scan = raised & _in_poly(es, ss, P)
            source = 'footprint'
    built = np.full(X.shape, np.nan)
    if len(pts):
        i = np.floor((pts[:, 0] - x0) / res).astype(int)
        j = np.floor((pts[:, 1] - y0) / res).astype(int)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        flat = j[ok] * nx + i[ok]
        top = np.full(nx * ny, -np.inf)
        np.maximum.at(top, flat, pts[ok, 2])
        built = np.where(np.isfinite(top), top, np.nan).reshape(X.shape)
    with np.errstate(invalid='ignore'):
        have = np.isfinite(built) & (built - gnd_map > thr) & inside
    inter, union = have & scan, have | scan
    cell = res * res
    out = {'iou': round(float(inter.sum() / max(union.sum(), 1)), 3),
           'coverage': round(float(inter.sum() / max(scan.sum(), 1)), 3),
           'precision': round(float(inter.sum() / max(have.sum(), 1)), 3),
           'scan_m2': round(float(scan.sum() * cell), 1), 'built_m2': round(float(have.sum() * cell), 1),
           'missing_m2': round(float((scan & ~have).sum() * cell), 1),
           'extra_m2': round(float((have & ~scan).sum() * cell), 1),
           'dz_median_m': None, 'dz_abs_median_m': None, 'dz_p90_m': None, 'top_built_m': None, 'top_scan_m': None,
           'best_shift_m': None, 'iou_at_shift': None, 'scan_mask': source, 'clipped': clip is not None,
           'built_cells': int(have.sum())}
    scan_top = fr.z_out(dsm)
    if inter.any():
        dz = (built - scan_top)[inter]
        out.update(dz_median_m=round(float(np.median(dz)), 2), dz_abs_median_m=round(float(np.median(np.abs(dz))), 2),
                   dz_p90_m=round(float(np.percentile(np.abs(dz), 90)), 2))
    if have.any():
        out['top_built_m'] = round(float(np.percentile((built - gnd_map)[have], 98)), 2)
    if scan.any():
        out['top_scan_m'] = round(float(np.percentile((dsm - dtm)[scan], 98)), 2)
    if have.any() and scan.any():
        dx, dy, v = _best_shift(have, scan, res)
        out['best_shift_m'], out['iou_at_shift'] = [round(dx, 2), round(dy, 2)], round(v, 3)
    return out


def building_votes(ras, fr, p: dict, ids: set[int]) -> dict[int, int]:
    """How many sample points of a placed object's visible box fall on cells of each scan building in ids."""
    t = math.radians(p['yaw'])
    L, W = max(p['length'], .2), max(p['width'], .2)
    u, v = np.meshgrid(np.linspace(-L / 2, L / 2, 5), np.linspace(-W / 2, W / 2, 5))
    x = p['x'] + u * math.cos(t) - v * math.sin(t)
    y = p['y'] + u * math.sin(t) + v * math.cos(t)
    e, sn = fr.to_scan(x.ravel(), y.ravel())
    r, c, inside = ras.cell(e, sn)
    oid = np.asarray(ras.objid[r, c])[inside]
    out: dict[int, int] = {}
    for k in oid.tolist():
        if k in ids:
            out[k] = out.get(k, 0) + 1
    return out
