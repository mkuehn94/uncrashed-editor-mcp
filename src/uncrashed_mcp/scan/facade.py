"""Facades of surveyed buildings: wall lines, orthographic facade renders from the mesh, window grids.

Wall lines come from the OSM footprint when the building has one (mapped walls), else from the roof outline
(eaves: about 0.3-0.8 m outside the wall). Each wall is rendered straight on (native sr_facades): colour and depth of
the surface nearest to a viewer outside, inside a slab around the wall line, skipping roof / ground triangles. A
detector (vision_worker, Grounding DINO) marks windows, doors, balconies and garage doors on the render; fit()
turns the boxes into storeys, rows and columns with sill / head heights above the ground.

Facade frame: u metres along the wall from its start corner a to its end b (looking at the facade from outside,
u grows to the right), z metres up in the scan frame; image row 0 is the top.
"""
from __future__ import annotations

import math

import numpy as np

from . import grid

RES_M = 0.08            # facade image metres per pixel
DEPTH_IN, DEPTH_OUT = -2.5, 1.6   # slab around the wall line (m): recesses inside, balconies / eaves outside
MIN_WALL_M = 3.0
MAX_PX = 1400           # longest image side; longer walls get a coarser res
KINDS = ('window', 'door', 'balcony', 'garage_door')


def _merge_edges(ring: np.ndarray, angle_deg: float = 14.0, min_m: float = 0.8) -> np.ndarray:
    """Drop vertices between nearly collinear edges and tiny jogs, repeatedly."""
    p = ring.copy()
    changed = True
    while changed and len(p) > 3:
        changed = False
        n = len(p)
        for i in range(n):
            a, b, c = p[i - 1], p[i], p[(i + 1) % n]
            v1, v2 = b - a, c - b
            l1, l2 = float(np.hypot(*v1)), float(np.hypot(*v2))
            if l1 < 1e-6 or l2 < 1e-6:
                p = np.delete(p, i, 0)
                changed = True
                break
            ang = abs(math.degrees(math.atan2(v1[0] * v2[1] - v1[1] * v2[0], v1 @ v2)))
            if ang < angle_deg or (min(l1, l2) < min_m and ang < 60):
                p = np.delete(p, i, 0)
                changed = True
                break
    return p


def _inside(poly: np.ndarray, x: float, y: float) -> bool:
    xs, ys = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xs, 1), np.roll(ys, 1)
    cross = ((ys > y) != (yj > y)) & (x < (xj - xs) * (y - ys) / np.where(yj - ys == 0, 1e-12, yj - ys) + xs)
    return bool(cross.sum() % 2)


WALL_REACH_M = 3.0      # a wall whose midpoint is further from the building's roof outline belongs to another part


def belongs(footprint, a, b, reach_m: float = WALL_REACH_M) -> bool:
    """Is wall a-b part of this building? False for walls of an OSM outline that several mesh buildings share
    (one mapped building split into parts) when the wall runs along another part."""
    fp = np.asarray(footprint, float)
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    if len(fp) < 3 or _inside(fp, mx, my):
        return True
    q = np.roll(fp, -1, 0)
    ab = q - fp
    t = np.clip(((mx - fp[:, 0]) * ab[:, 0] + (my - fp[:, 1]) * ab[:, 1]) / np.maximum((ab ** 2).sum(1), 1e-12), 0, 1)
    return float(np.hypot(fp[:, 0] + t * ab[:, 0] - mx, fp[:, 1] + t * ab[:, 1] - my).min()) <= reach_m


def walls(ring, simplify_m: float = 0.5, min_m: float = MIN_WALL_M) -> list[dict]:
    """Wall segments of a footprint ring (scan e,s): start a, end b, outward normal, length. The ring is cleaned
    (Douglas-Peucker, collinear edges merged) so a jaggy roof outline gives few long walls."""
    p = np.asarray(ring, float)
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    if len(p) < 3:
        return []
    if len(p) > 4:
        p = grid.simplify(p, simplify_m)
    p = _merge_edges(p)
    if len(p) < 3:
        return []
    out = []
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        L = float(np.hypot(*(b - a)))
        if L < min_m:
            continue
        d = (b - a) / L
        n = np.array([d[1], -d[0]])
        if _inside(p, *((a + b) / 2 + 0.3 * n)):
            n = -n
        # looking at the wall from outside (against n), u must grow to the viewer's right: right = (-n_s, n_e)
        right = np.array([-n[1], n[0]])
        if d @ right < 0:
            a, b, d = b, a, -d
        out.append({'a': a, 'b': b, 'dir': d, 'n': n, 'length_m': L,
                    'faces_deg': math.degrees(math.atan2(n[1], n[0])) % 360})
    return out


def plane(w: dict, z0: float, z1: float, res: float = RES_M, pad_m: float = 0.6) -> tuple[list[float], tuple[int, int]]:
    """sr_facades parameters (12 doubles) and image size for wall w between heights z0 and z1 (scan up)."""
    L = w['length_m']
    span = max(L + 2 * pad_m, z1 - z0)
    res = max(res, span / MAX_PX)
    W = max(8, int(math.ceil((L + 2 * pad_m) / res)))
    H = max(8, int(math.ceil((z1 - z0) / res)))
    a, d, n = w['a'], w['dir'], w['n']
    return [float(a[0]), float(a[1]), float(d[0]), float(d[1]), float(n[0]), float(n[1]), -pad_m, float(z1), res,
            DEPTH_IN, DEPTH_OUT, 0.0], (W, H)


def to_px(p: list[float], u: float, z: float) -> tuple[float, float]:
    """Facade metres (u along the wall, z scan up) -> image pixel x, y."""
    return (u - p[6]) / p[8], (p[7] - z) / p[8]


def to_m(p: list[float], x: float, y: float) -> tuple[float, float]:
    return p[6] + x * p[8], p[7] - y * p[8]


FRONT_STEPS_M = (1.0, 1.5, 2.0, 2.5, 3.0)


def occluded_front(ras, w: dict, z_ground: float, eave_up: float, own_id: int | None = None) -> float:
    """Share of positions along the wall where another building stands right in front of it (a party wall).
    At each position the first cell 1 to 3 m out that is not this building's own (its eaves, dormers and balconies
    reach over an OSM wall line) decides: high and of class building = blocked."""
    t = np.linspace(0.1, 0.9, 9)
    base = w['a'][None, :] + t[:, None] * (w['b'] - w['a'])[None, :]
    found = np.zeros(len(t), bool)
    blocked = np.zeros(len(t), bool)
    for d in (FRONT_STEPS_M if own_id is not None else (1.5,)):
        pts = base + d * w['n'][None, :]
        r, c, inside = ras.cell(pts[:, 0], pts[:, 1])
        own = np.asarray(ras.objid[r, c]) == own_id if own_id is not None else np.zeros(len(t), bool)
        take = ~found & ~own
        top = np.asarray(ras.dsm[r, c], float)
        cls = np.asarray(ras.cls[r, c])
        high = inside & np.isfinite(top) & (top > z_ground + 0.6 * (eave_up - z_ground)) & (cls == 10)
        blocked[take] = high[take]
        found |= take
    return float(blocked.mean())


# ---------------------------------------------------------------------------- window grid


def _cluster(values: np.ndarray, tol: float) -> list[np.ndarray]:
    """1D single-link clusters (indices) of sorted values with gaps > tol."""
    if not len(values):
        return []
    order = np.argsort(values)
    groups, cur = [], [order[0]]
    for i, j in zip(order[:-1], order[1:]):
        if values[j] - values[i] > tol:
            groups.append(np.array(cur))
            cur = []
        cur.append(j)
    groups.append(np.array(cur))
    return groups


def fit(dets: list[dict], ground_up: float, eave_m: float | None, length_m: float) -> dict:
    """Window grid of one facade from detector boxes in facade metres ({'kind', 'score', 'u0','u1','z0','z1'}).

    Rows: windows clustered by centre height; per row sill and head heights (above ground), column centres and
    spacing; storeys = window rows above ground (a ground floor with only doors / garage doors counts too; a row of
    basement windows with heads below 1.2 m does not). Doors and garage
    doors are listed with their u range and width."""
    win = [d for d in dets if d['kind'] == 'window']
    rows = []
    if win:
        zc = np.array([(d['z0'] + d['z1']) / 2 for d in win])
        hts = np.array([d['z1'] - d['z0'] for d in win])
        tol = max(0.6, 0.45 * float(np.median(hts)))
        for idx in _cluster(zc, tol):
            g = [win[i] for i in idx]
            if len(g) < 1:
                continue
            uc = np.sort(np.array([(d['u0'] + d['u1']) / 2 for d in g]))
            gaps = np.diff(uc)
            rows.append({'z_m': round(float(np.median([(d['z0'] + d['z1']) / 2 for d in g])) - ground_up, 2),
                         'sill_m': round(float(np.median([d['z0'] for d in g])) - ground_up, 2),
                         'head_m': round(float(np.median([d['z1'] for d in g])) - ground_up, 2),
                         'count': len(g),
                         'width_m': round(float(np.median([d['u1'] - d['u0'] for d in g])), 2),
                         'spacing_m': round(float(np.median(gaps)), 2) if len(gaps) else None,
                         'u_m': [round(float(v), 2) for v in uc],
                         'score': round(float(np.mean([d['score'] for d in g])), 2)})
        rows.sort(key=lambda r: r['z_m'])
    doors = [{'kind': d['kind'], 'u0_m': round(d['u0'], 2), 'u1_m': round(d['u1'], 2),
              'width_m': round(d['u1'] - d['u0'], 2), 'head_m': round(d['z1'] - ground_up, 2),
              'score': round(d['score'], 2)} for d in dets if d['kind'] in ('door', 'garage_door')]
    balconies = [{'u0_m': round(d['u0'], 2), 'u1_m': round(d['u1'], 2), 'z0_m': round(d['z0'] - ground_up, 2),
                  'z1_m': round(d['z1'] - ground_up, 2), 'score': round(d['score'], 2)}
                 for d in dets if d['kind'] == 'balcony']
    above = [r for r in rows if r['head_m'] > 1.2]          # basement light-well windows are not a storey
    zs = [r['z_m'] for r in above]
    storey_h = float(np.median(np.diff(zs))) if len(zs) > 1 else None
    if storey_h is not None and not 2.3 <= storey_h <= 5.0:
        storey_h = None
    storeys = len(above)
    ground_floor = bool(above) and above[0]['sill_m'] < 2.5   # raised ground floors sit up to ~2 m
    if above and not ground_floor and doors:
        storeys += 1   # ground floor with doors only
    out = {'rows': rows, 'storeys': storeys or None, 'storey_height_m': round(storey_h, 2) if storey_h else None,
           'windows': len(win), 'doors': doors, 'balconies': balconies}
    if eave_m and storey_h:
        out['storeys_from_height'] = int(round(eave_m / storey_h))
    cols = [r['count'] for r in rows]
    out['columns'] = int(np.median(cols)) if cols else 0
    sp = [r['spacing_m'] for r in rows if r['spacing_m']]
    out['spacing_m'] = round(float(np.median(sp)), 2) if sp else None
    out['window_share'] = round(float(sum((d['u1'] - d['u0']) * (d['z1'] - d['z0']) for d in win)) /
                                max(1e-6, length_m * max(1.0, (eave_m or 3.0))), 3)
    return out
