"""Rule-based geometry survey of a top-down rasterised photogrammetry mesh (numpy + native/scanraster).

Input: DSM (highest surface), unit normals and texture colour on one grid (east, south, up metres). Output: DTM by a
progressive morphological filter, nDSM, a class raster, an object-id raster and objects (buildings with footprint,
parts and roof type; trees by crown tops; cars by blob / height-dip split / tiling; low structures, shrubs, bridges).
Optional OSM layers (rasterised on the same grid by osm.py) act as priors: building footprints split merged roofs
and claim canopy-covered roof cells, roads / parking / paths / water override ground classes, tree nodes confirm
crowns, and car candidates inside a mapped building (carports, garages) are rejected.

Every object records `src` (mesh, mesh+osm) and a rule confidence `conf`; nothing here is a learned detector.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import grid, native
from . import osm as osm_mod

VERSION = 2
# class raster codes
CLASSES = {0: 'none', 1: 'paved', 2: 'grass', 3: 'water', 4: 'bare', 5: 'other_ground', 6: 'road', 7: 'parking',
           8: 'path', 10: 'building', 11: 'vegetation', 12: 'low_object'}
CODE = {v: k for k, v in CLASSES.items()}
GROUND_CODES = (1, 2, 3, 4, 5, 6, 7, 8)
HIGH_M = 2.2          # nDSM above this: building or canopy
LOW_M = 0.6           # nDSM above this (and <= HIGH_M): low object (car, hedge, shelter)
CAR_L, CAR_W = (2.7, 6.0), (1.3, 2.6)   # accepted single-car blob size (cell-centre extents) on a melted mesh


class Grid:
    """Cell (r, c) centre = (x0 + (c + .5) res, y0 + (r + .5) res), east/south metres."""

    def __init__(self, x0: float, y0: float, res: float, w: int, h: int):
        self.x0, self.y0, self.res, self.w, self.h = float(x0), float(y0), float(res), int(w), int(h)

    def to_world(self, c, r):
        return self.x0 + (np.asarray(c, float) + .5) * self.res, self.y0 + (np.asarray(r, float) + .5) * self.res

    def to_cell(self, x, y):
        return (np.asarray(x, float) - self.x0) / self.res - .5, (np.asarray(y, float) - self.y0) / self.res - .5


def _fill_nearest(dsm: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if valid.all():
        return dsm.astype(np.float32)
    idx = native.nearest(valid)
    return dsm.ravel()[idx.ravel()].reshape(dsm.shape).astype(np.float32)


def dtm_pmf(dsmf: np.ndarray, res: float) -> np.ndarray:
    """Progressive morphological filter on a ~1 m grid, ground cells interpolated by normalised convolution."""
    H, W = dsmf.shape
    B = max(1, int(round(1.0 / res)))
    hp, wp = -(-H // B) * B, -(-W // B) * B
    pad = np.pad(dsmf, ((0, hp - H), (0, wp - W)), mode='edge')
    zc = pad.reshape(hp // B, B, wp // B, B).min(axis=(1, 3)).astype(np.float32)
    step = res * B
    ground = np.ones(zc.shape, bool)
    surf = zc.copy()
    for w_m, dh in ((3, 0.3), (5, 0.4), (9, 0.6), (17, 1.0), (33, 1.8), (65, 2.5)):
        k = max(3, int(round(w_m / step)) | 1)
        op = grid.grey_open(surf, k)
        ground &= (surf - op) <= dh
        surf = op
    val = np.where(ground, zc, 0).astype(np.float64)
    wgt = ground.astype(np.float64)
    dtm_c = np.full(zc.shape, np.nan)
    for s in (1, 2, 4, 8, 16, 32):
        num, den = grid.gauss(val, s / step), grid.gauss(wgt, s / step)
        est = num / np.maximum(den, 1e-9)
        fill = np.isnan(dtm_c) & (den > 0.05)
        dtm_c[fill] = est[fill]
    dtm_c = np.where(ground, zc, dtm_c)
    if np.isnan(dtm_c).any():
        dtm_c = np.nan_to_num(dtm_c, nan=float(np.nanmedian(dtm_c)) if np.isfinite(dtm_c).any() else 0.0)
    dtm_c = grid.gauss(dtm_c, 1.0 / step, mode='edge')
    dtm = grid.resize_linear(dtm_c, hp, wp)[:H, :W]
    return np.minimum(dtm, dsmf).astype(np.float32)


EAVE_BAND_M = 1.0     # the roof edge band whose heights give the eave
EAVE_PCT = 30         # percentile of the band: long eave sides, not the gable ends rising to the ridge


def edge_eave(rows: np.ndarray, cols: np.ndarray, nd: np.ndarray, res: float) -> float | None:
    """Eave height above ground: the 30th percentile of the nDSM in the outer EAVE_BAND_M of a roof (the cells of
    one building), i.e. where the roof meets the walls. The 15th percentile of the whole roof sits ~1 m higher on
    tall gabled roofs with dormers. None for fewer than 8 band cells."""
    if len(rows) < 8:
        return None
    r0, c0 = rows.min() - 1, cols.min() - 1
    m = np.zeros((rows.max() - r0 + 2, cols.max() - c0 + 2), bool)
    m[rows - r0, cols - c0] = True
    inner = m.copy()
    for _ in range(max(1, int(round(EAVE_BAND_M / res)))):
        e = inner.copy()
        e[1:] &= inner[:-1]
        e[:-1] &= inner[1:]
        e[:, 1:] &= inner[:, :-1]
        e[:, :-1] &= inner[:, 1:]
        inner = e
    band = ~inner[rows - r0, cols - c0]
    vals = np.asarray(nd, float)[band]
    vals = vals[np.isfinite(vals)]
    if len(vals) < 8:
        return None
    return float(np.percentile(vals, EAVE_PCT))


def _roof_type(nrm_cells: np.ndarray) -> tuple[str, float, float | None]:
    """(roof type, share supporting it, ridge yaw degrees or None) from the up-facing normals of a roof."""
    nx, ny, nz = nrm_cells[:, 0], nrm_cells[:, 1], nrm_cells[:, 2]
    tilt = np.degrees(np.arccos(np.clip(nz, -1, 1)))
    flat = float((tilt < 8).mean()) if len(tilt) else 1.0
    sloped = tilt > 15
    if flat > 0.6 or sloped.mean() < 0.45:
        return 'flat', flat, None
    asp = (np.degrees(np.arctan2(ny[sloped], nx[sloped])) + 360) % 360
    hist = np.bincount(((asp + 22.5) // 45).astype(int) % 8, minlength=8) / max(1, int(sloped.sum()))
    major = np.sort(hist)[::-1]
    peaks = [i for i in range(8) if hist[i] > 0.15]
    if len(peaks) == 2 and abs(peaks[0] - peaks[1]) == 4:
        # ridge runs perpendicular to the two opposite slope directions
        sel = ((asp + 22.5) // 45).astype(int) % 8 == peaks[0]
        a = float(np.degrees(np.arctan2(np.sin(np.radians(asp[sel])).mean(), np.cos(np.radians(asp[sel])).mean())))
        return 'gabled', float(major[:2].sum()), round((a + 90) % 180, 1)
    if len(peaks) >= 3 and major[:4].sum() > 0.75:
        return 'hipped', float(major[:4].sum()), None
    if len(peaks) == 1:
        a = peaks[0] * 45.0
        return 'mono_pitch', float(major[0]), round((a + 90) % 180, 1)
    return 'complex', float(major[:2].sum()), None


FLAT_PCT = 2.0        # roof planes below this slope (percent) count as flat
MONO_PCT = 8.0       # a single plane steeper than this is a mono-pitch roof, else flat with a fall
RIDGE_MIN_M = 0.2     # a ridge / valley must stand this far above / below the roof edges


def _plane(u, v, z, keep=None):
    """Robust least-squares plane z = a + b u + c v: (coef, rms of the kept points, kept mask)."""
    keep = np.ones(len(z), bool) if keep is None else keep.copy()
    coef = np.zeros(3)
    for _ in range(3):
        if keep.sum() < 6:
            return None
        A = np.stack([np.ones(int(keep.sum())), u[keep], v[keep]], 1)
        coef = np.linalg.lstsq(A, z[keep], rcond=None)[0]
        r = z - (coef[0] + coef[1] * u + coef[2] * v)
        mad = float(np.median(np.abs(r[keep]))) * 1.4826
        keep = keep & (np.abs(r) <= max(2.5 * mad, 0.08))
    r = z - (coef[0] + coef[1] * u + coef[2] * v)
    return coef, float(np.sqrt(np.mean(r[keep] ** 2))) if keep.any() else 0.0, keep


def roof_planes(e: np.ndarray, s: np.ndarray, z: np.ndarray, yaw_deg: float, ground: float) -> dict | None:
    """Fit one plane and the best pair of planes meeting at a line along either box axis to roof heights (metres;
    e, s positions, z absolute heights). Returns form flat / mono_pitch / gabled (ridge) / valley, slope per plane
    in percent, pitch, the ridge or valley line (ends and height above ground) and the fit residual. Low pitches
    (a 3 % valley roof, a 0.5 m gable on a narrow house) that the normals call flat show up here."""
    if len(z) < 30:
        return None
    a = math.radians(yaw_deg)
    cx, cy = float(np.mean(e)), float(np.mean(s))
    u = (e - cx) * math.cos(a) + (s - cy) * math.sin(a)
    v = -(e - cx) * math.sin(a) + (s - cy) * math.cos(a)
    one = _plane(u, v, z)
    if one is None:
        return None
    c1, rms1, keep1 = one
    best = None
    for name, t, w in (('across', v, u), ('along', u, v)):
        span = float(t.max() - t.min())
        if span < 3.0:
            continue
        for q in np.linspace(0.3, 0.7, 9):
            t0 = float(np.quantile(t, q))
            A = t < t0
            if min(A.sum(), (~A).sum()) < 0.2 * len(t):
                continue
            # each side: z = a + g (t - t0) + h w
            pa, pb = _plane(t[A] - t0, w[A], z[A]), _plane(t[~A] - t0, w[~A], z[~A])
            if pa is None or pb is None:
                continue
            n_a, n_b = int(pa[2].sum()), int(pb[2].sum())
            rms = math.sqrt((pa[1] ** 2 * n_a + pb[1] ** 2 * n_b) / max(1, n_a + n_b))
            if best is None or rms < best[0]:
                best = (rms, name, t0, pa[0], pb[0], t, w, A)
    out = {'rms_m': round(rms1, 2)}
    grad1 = math.hypot(c1[1], c1[2]) * 100
    form, planes, line = 'flat', [round(grad1, 1)], None
    if best is not None:
        rms, name, t0, ca, cb, t, w, A = best
        ga, gb = ca[1], cb[1]          # slope along t on each side (rising toward +t when positive)
        z_line = (ca[0] + cb[0]) / 2   # both planes at the line (w = 0)
        edge_a = ca[0] + ga * (float(t.min()) - t0)
        edge_b = cb[0] + gb * (float(t.max()) - t0)
        rise = min(z_line - edge_a, z_line - edge_b)
        fall = min(edge_a - z_line, edge_b - z_line)
        better = rms < 0.8 * rms1 and rms < rms1 - 0.02
        steep = min(abs(ga), abs(gb)) * 100 >= FLAT_PCT * 0.75
        if better and steep and ga > 0 > gb and rise >= RIDGE_MIN_M:
            form = 'gabled'
        elif better and steep and ga < 0 < gb and fall >= RIDGE_MIN_M:
            form = 'valley'
        if form != 'flat':
            planes = [round(math.hypot(ga, ca[2]) * 100, 1), round(math.hypot(gb, cb[2]) * 100, 1)]
            wl = w[np.abs(t - t0) < 1.0] if (np.abs(t - t0) < 1.0).any() else w
            w0, w1 = float(wl.min()), float(wl.max())
            if name == 'across':     # t = v, the line runs along u
                ends = [(w0, t0), (w1, t0)]
            else:
                ends = [(t0, w0), (t0, w1)]
            pts = [[round(cx + uu * math.cos(a) - vv * math.sin(a), 2), round(cy + uu * math.sin(a) + vv * math.cos(a), 2)]
                   for uu, vv in ends]
            line = {'kind': 'ridge' if form == 'gabled' else 'valley', 'a': pts[0], 'b': pts[1],
                    'height_m': round(float(z_line - ground), 2),
                    'edges_m': [round(float(edge_a - ground), 2), round(float(edge_b - ground), 2)],
                    'yaw_deg': round((yaw_deg + (0 if name == 'across' else 90)) % 180, 1)}
            out['rms_m'] = round(rms, 2)
    if form == 'flat' and grad1 >= FLAT_PCT:
        form = 'mono_pitch'
    out.update(form=form, slope_pct=planes, pitch_deg=round(math.degrees(math.atan(max(planes) / 100)), 1))
    if line:
        out['line'] = line
    return out


def _parts(xs, ys, res) -> list[tuple[np.ndarray, np.ndarray]]:
    """Near-rectangular parts of a footprint (cell coordinates): the greedy split, then every part made one
    connected piece of at least 15 m2 and 2.5 m width (slivers and stray cells join the neighbouring part).
    One entry when the footprint does not split into at least two such parts."""
    pieces = _rect_split(xs, ys, res)
    if len(pieces) < 2:
        pieces = _wings(xs, ys, res)
    if len(pieces) < 2:
        return [(xs, ys)]
    c0, r0 = int(xs.min()), int(ys.min())
    shape = (int(ys.max()) - r0 + 1, int(xs.max()) - c0 + 1)
    keep_lab = np.zeros(shape, np.int32)
    kept = []
    for pc, pr in pieces:
        m = np.zeros(shape, bool)
        m[pr - r0, pc - c0] = True
        lab, n = native.label(m, conn8=False)
        for rr, cc in grid.components(lab, n):
            if len(rr) * res * res < 15:
                continue
            _, _, _, wd, _ = grid.obb(cc, rr)
            if wd * res < 2.5:
                continue
            kept.append((rr, cc))
            keep_lab[rr, cc] = len(kept)
    if len(kept) < 2:
        return [(xs, ys)]
    idx = native.nearest(keep_lab > 0)
    owner = keep_lab.ravel()[idx[ys - r0, xs - c0]]
    return [(xs[owner == k], ys[owner == k]) for k in range(1, len(kept) + 1)]


WING_WINDOW_M = 40.0   # local axis of a wing: principal axis of the footprint cells within this window
WING_ANISO = 0.5       # cells this elongated locally belong to a wing (corners and squares are not)


def _wings(xs, ys, res) -> list[tuple[np.ndarray, np.ndarray]]:
    """Wings of a large footprint that fills its box poorly (a V, a bent block, three bars round a yard), where no
    straight cut on one box frame helps: each cell takes the local long axis of the footprint around it (principal
    axis within WING_WINDOW_M); cells of the same axis (within 15 degrees of a frequent axis) form a wing; corners
    join the nearest wing in _parts."""
    n = len(xs)
    if n * res * res < ANGLED_MIN_M2:
        return [(xs, ys)]
    c0, r0 = int(xs.min()), int(ys.min())
    m = np.zeros((int(ys.max()) - r0 + 1, int(xs.max()) - c0 + 1), np.float32)
    m[ys - r0, xs - c0] = 1
    fill = n / m.size
    if fill >= ANGLED_FILL:
        return [(xs, ys)]
    k = max(5, int(WING_WINDOW_M / res) | 1)
    R, C = np.mgrid[0:m.shape[0], 0:m.shape[1]].astype(np.float32)
    R -= R.mean()
    C -= C.mean()
    s0 = grid.box_blur(m, k)
    with np.errstate(all='ignore'):
        mx, my = grid.box_blur(m * C, k) / s0, grid.box_blur(m * R, k) / s0
        sxx = grid.box_blur(m * C * C, k) / s0 - mx * mx
        syy = grid.box_blur(m * R * R, k) / s0 - my * my
        sxy = grid.box_blur(m * C * R, k) / s0 - mx * my
        ang = (0.5 * np.degrees(np.arctan2(2 * sxy, sxx - syy))) % 180
        tr = sxx + syy
        det = np.sqrt(np.maximum((sxx - syy) ** 2 + 4 * sxy ** 2, 0))
        aniso = det / np.maximum(tr, 1e-6)
    a = ang[ys - r0, xs - c0]
    el = aniso[ys - r0, xs - c0] > WING_ANISO
    if el.sum() < 0.3 * n:
        return [(xs, ys)]
    hist = np.bincount((a[el] // 5).astype(int) % 36, minlength=36).astype(float)
    hist = (np.roll(hist, 1) + hist + np.roll(hist, -1)) / 3
    peaks = []
    for b in np.argsort(-hist):
        if hist[b] < 0.15 * el.sum() / 3:
            break
        centre = b * 5 + 2.5
        if all(min(abs(centre - q), 180 - abs(centre - q)) >= 20 for q in peaks):
            peaks.append(centre)
        if len(peaks) == 4:
            break
    if len(peaks) < 2:
        return [(xs, ys)]
    d = np.array([np.minimum(np.abs(a - q), 180 - np.abs(a - q)) for q in peaks])
    group = np.where(el & (d.min(0) <= 15), d.argmin(0), -1)
    out = []
    for g_ in range(len(peaks)):
        sel = group == g_
        if not sel.any():
            continue
        mm = np.zeros(m.shape, bool)
        mm[ys[sel] - r0, xs[sel] - c0] = True
        lab, nl = native.label(mm, conn8=False)
        for rr, cc in grid.components(lab, nl):
            if len(rr) * res * res >= 30:
                out.append((cc + c0, rr + r0))
    return out if len(out) >= 2 else [(xs, ys)]


PART_FILL = 0.9      # building parts: a cut that makes both sides fill their boxes this well is taken ...
PART_GAIN = 0.15     # ... and so is one that raises the fill this much (courtyard blocks, wings), at most 4 levels


ANGLED_MIN_M2 = 300  # a large footprint that fills its box poorly also tries cuts at other angles (wings
ANGLED_FILL = 0.6    # meeting at an angle: a V or a bent block whose wings no single box frame follows)


def _best_cut(xs, ys, res, yaw):
    """(fill, cells a, cells b, whole fill) of the best straight cut along the box axes at yaw."""
    n = len(xs)
    a = math.radians(yaw)
    u = np.round(xs * math.cos(a) + ys * math.sin(a))      # whole cells along the box axes
    v = np.round(-xs * math.sin(a) + ys * math.cos(a))
    whole = n / ((np.ptp(u) + 1) * (np.ptp(v) + 1))
    best = None
    min_cells = 15 / (res * res)
    min_w = 2.5 / res
    for q, p in ((u, v), (v, u)):
        order = np.argsort(q, kind='stable')
        qs, ps = q[order], p[order]
        lo_min, lo_max = np.minimum.accumulate(ps), np.maximum.accumulate(ps)
        hi_min = np.minimum.accumulate(ps[::-1])[::-1]
        hi_max = np.maximum.accumulate(ps[::-1])[::-1]
        cuts = np.flatnonzero(np.diff(qs) > 0.5) + 1      # split between distinct rows of cells
        for k in cuts:
            na, nb = k, n - k
            if na < min_cells or nb < min_cells:
                continue
            la, lb = qs[k - 1] - qs[0] + 1, qs[-1] - qs[k] + 1
            wa, wb = lo_max[k - 1] - lo_min[k - 1] + 1, hi_max[k] - hi_min[k] + 1
            if min(la, lb, wa, wb) < min_w:
                continue
            fill = n / (la * wa + lb * wb)
            if best is None or fill > best[0]:
                best = (fill, order[:k], order[k:])
    return best, whole


def _rect_split(xs, ys, res, depth=0, yaw=None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a footprint (cell coordinates) along its box axes where the two sides fill their own boxes best: an L
    becomes two rectangles, a courtyard block its wings. A cut is taken when it raises the fill of the boxes to
    PART_FILL (by 0.08) or by PART_GAIN, and leaves each side at least 15 m2 and 2.5 m wide; at most four levels.
    A large footprint filling its box poorly (wings at an angle) also tries cuts every 7.5 degrees; the pieces of
    such a cut then find their own box axes."""
    n = len(xs)
    if n * res * res < 40 or depth >= 4:
        return [(xs, ys)]
    if yaw is None:
        yaw = grid.obb(xs, ys)[4]
    best, whole = _best_cut(xs, ys, res, yaw)
    if whole >= PART_FILL:
        return [(xs, ys)]
    child_yaw = yaw
    ok = best is not None and best[0] >= whole + (0.08 if best[0] >= PART_FILL else PART_GAIN)
    if not ok and n * res * res >= ANGLED_MIN_M2 and whole < ANGLED_FILL:
        for cand in np.arange(0.0, 180.0, 7.5):
            b2, w2 = _best_cut(xs, ys, res, yaw + cand)
            if b2 is not None and b2[0] >= max(whole, w2) + PART_GAIN and (not ok or b2[0] > best[0]):
                best, ok, child_yaw = b2, True, None
    if not ok:
        return [(xs, ys)]
    _, ia, ib = best
    return (_rect_split(xs[ia], ys[ia], res, depth + 1, child_yaw) +
            _rect_split(xs[ib], ys[ib], res, depth + 1, child_yaw))


class Survey:
    def __init__(self, g: Grid, dsm, nrm, rgb, osm_features: dict | None = None, terrain=None):
        """terrain: (heights on this grid in the provider's datum, NaN = no data; provider name) or None: a
        terrain model that replaces the filtered ground model where it has data (providers/fuse.apply_terrain)."""
        self.g, self.dsm, self.nrm, self.rgb = g, dsm, nrm, rgb
        self.osm_features, self.osm, self.osm_info = osm_features, {}, None
        self.terrain, self.terrain_info = terrain, None
        self.osm_conflicts: dict[int, dict] = {}
        self.stats: dict[str, int] = {}
        self.edge_px = None
        self.objects: list[dict] = []
        self.timings: dict[str, float] = {}
        self.notes: list[str] = []

    # ------------------------------------------------------------------ helpers
    def _t(self, name, t0):
        self.timings[name] = round(time.time() - t0, 2)

    def _base(self, cls, conf, rows, cols, src='mesh', box=None) -> dict:
        g = self.g
        if box is None:
            cx, cy, L, Wd, yaw = grid.obb(cols, rows)
        else:
            cx, cy, L, Wd, yaw = box
        x, y = g.to_world(cx, cy)
        nd = self.ndsm[rows, cols]
        gz = float(np.median(self.dtm[rows, cols]))
        h = float(np.percentile(nd, 95))
        return {'id': len(self.objects), 'cls': cls, 'conf': round(float(conf), 2), 'src': src,
                'x': round(float(x), 2), 'y': round(float(y), 2), 'ground_z': round(gz, 2),
                'top_z': round(gz + h, 2), 'height_m': round(h, 2), 'length_m': round(L * g.res, 2),
                'width_m': round(Wd * g.res, 2), 'yaw_deg': round(yaw, 1),
                'area_m2': round(len(rows) * g.res * g.res, 1),
                'rgb': [int(v) for v in np.median(self.rgb[rows, cols], 0)]}

    def _add(self, obj: dict, rows=None, cols=None, poly=None) -> dict:
        obj['id'] = len(self.objects)
        self.objects.append(obj)
        if rows is not None:
            free = self.objid[rows, cols] < 0
            self.objid[rows[free], cols[free]] = obj['id']
        if poly is not None:
            native.fill_poly(self.objid, poly[:, 0], poly[:, 1], obj['id'])
        return obj

    def _footprint(self, lab, value, rows, cols) -> list[list[float]]:
        ring = grid.outline(lab, value, rows, cols)
        ring = grid.simplify(ring, 0.75 / self.g.res) if len(ring) > 3 else ring
        x, y = self.g.to_world(ring[:, 0], ring[:, 1])
        return [[round(float(a), 2), round(float(b), 2)] for a, b in zip(x, y)]

    # ------------------------------------------------------------------ stages
    def run(self) -> 'Survey':
        t0 = time.time()
        g = self.g
        self.valid = np.isfinite(self.dsm)
        if not self.valid.any():
            raise ValueError('the survey region holds no geometry (check center / radius_m)')
        self.dsmf = _fill_nearest(self.dsm, self.valid)
        self._t('fill', t0)
        t0 = time.time()
        self.dtm = dtm_pmf(self.dsmf, g.res)
        if self.terrain is not None and self.terrain[0] is not None:
            from .providers import fuse
            self.dtm, self.terrain_info = fuse.apply_terrain(self.dtm, self.dsmf, self.valid, *self.terrain)
        self.ndsm = np.where(self.valid, self.dsmf - self.dtm, 0).astype(np.float32)
        self._t('dtm', t0)
        t0 = time.time()
        self._features()
        self._t('features', t0)
        if self.osm_features:
            t0 = time.time()
            roofs = self.valid & (self.ndsm > HIGH_M) & (self.exg_s < 0.02)
            shift, before, after = osm_mod.align(self.osm_features, g, roofs)
            self.osm = osm_mod.rasterise(self.osm_features, g, shift)
            self.osm_info = {'shift_m': [round(shift[0], 2), round(shift[1], 2)],
                             'roof_overlap': [round(before, 2), round(after, 2)]}
            if self.osm_features.get('roads'):   # carriageway to the kerbs, sidewalks beyond them
                road, walk, krep = osm_mod.kerb_bands(self.osm_features, g, self.dsmf, self.ndsm, self.valid, shift)
                self.osm['road'] = self.osm['road'] | road
                self.osm['sidewalk'] = walk & ~self.osm['road']
                self.osm_info['kerbs'] = krep
            self._osm_fit()
            self._t('osm', t0)
        t0 = time.time()
        self._classify()
        self._t('classify', t0)
        self.objid = np.full(self.dsm.shape, -1, np.int32)
        t0 = time.time()
        self._buildings()
        self._claim_edges()
        self._t('buildings', t0)
        t0 = time.time()
        self._trees()
        self._t('trees', t0)
        t0 = time.time()
        self._cars()
        self._t('cars', t0)
        return self

    def _osm_fit(self, raised_m: float = 1.8, min_cover: float = 0.6):
        """Per mapped footprint: the share of its cells (with scan data) that stand raised_m above the ground. A
        footprint below min_cover disagrees with the heights (moved, demolished, underground, cut across a roof):
        it is taken out of the building layer (no split, no claimed cells, no OSM walls) and recorded in
        self.osm_conflicts {index: {...}} with the best local shift for the report."""
        self.osm_conflicts = {}
        bld = self.osm.get('building')
        if bld is None or not bld.any():
            return
        g = self.g
        # cells under foliage say nothing about the footprint: they count neither way
        seen = self.valid & ~((self.exg_s > 0.035) & (self.rough > 0.15))
        raised = seen & (self.ndsm > raised_m)
        ids = bld.ravel()
        n = np.bincount(ids, minlength=1)
        nv = np.bincount(ids, weights=seen.ravel(), minlength=len(n))
        nr = np.bincount(ids, weights=raised.ravel(), minlength=len(n))
        nd = np.bincount(ids, weights=self.valid.ravel(), minlength=len(n))
        tags = self.osm.get('building_tags', {})
        reach = int(round(3.0 / g.res))
        step = max(1, int(round(0.5 / g.res)))
        for i in range(1, len(n)):
            if n[i] * g.res ** 2 < 10 or nd[i] < 0.8 * n[i] or nv[i] < 0.3 * n[i]:
                continue
            cover = nr[i] / max(nv[i], 1)
            if cover >= min_cover:
                continue
            rr, cc = np.nonzero(bld == i)
            best = (cover, 0, 0)
            for dy in range(-reach, reach + 1, step):
                for dx in range(-reach, reach + 1, step):
                    r2, c2 = rr + dy, cc + dx
                    ok = (r2 >= 0) & (r2 < g.h) & (c2 >= 0) & (c2 < g.w)
                    if ok.sum() < 0.8 * len(rr):
                        continue
                    v = float(raised[r2[ok], c2[ok]].sum() / max(1, seen[r2[ok], c2[ok]].sum()))
                    if v > best[0] + 0.05:
                        best = (v, dx, dy)
            self.osm_conflicts[i] = {'osm': (tags.get(i) or {}).get('osm'), 'cover': round(float(cover), 2),
                                     'area_m2': round(float(n[i] * g.res ** 2), 1),
                                     'best_shift_m': [round(best[1] * g.res, 2), round(best[2] * g.res, 2)],
                                     'cover_shifted': round(float(best[0]), 2)}
        if self.osm_conflicts:
            self.osm['building_all'] = bld.copy()
            bld = bld.copy()
            bld[np.isin(bld, list(self.osm_conflicts))] = 0
            self.osm['building'] = bld
        self.osm_info['footprints_disagree'] = len(self.osm_conflicts)
        if self.osm_conflicts:
            self.osm_info['disagree'] = sorted(self.osm_conflicts.values(), key=lambda c: -c['area_m2'])[:20]

    def _features(self):
        f = self.rgb.astype(np.float32) / 255
        s = f.sum(2) + 1e-6
        r, gg, b = f[..., 0] / s, f[..., 1] / s, f[..., 2] / s
        self.exg = 2 * gg - r - b
        self.f = f
        self.bright = f.mean(2)
        self.sat = f.max(2) - f.min(2)
        nrm = self.nrm
        self.nup = nrm[..., 2]
        mn = np.stack([grid.box_blur(nrm[..., k], 5) for k in range(3)], -1)
        self.coh = np.linalg.norm(mn, axis=2)
        d = self.dsmf - grid.box_blur(self.dsmf, 5)
        self.rough = np.sqrt(np.maximum(grid.box_blur(d * d, 5), 0))
        self.slope = np.degrees(np.arccos(np.clip(self.nup, -1, 1)))
        self.exg_s = grid.box_blur(self.exg, 3)

    def _classify(self):
        valid, ndsm, res = self.valid, self.ndsm, self.g.res
        k = lambda m: max(3, int(round(m / res)) | 1)  # noqa: E731 - window in cells for a size in metres
        high = valid & (ndsm > HIGH_M)
        green_roof = grid.bin_open(high & (self.coh > 0.97) & (self.rough < 0.12) & (self.nup > math.cos(math.radians(8))),
                                   k(2.75))
        exg_5 = grid.box_blur(self.exg, 7)
        veg = high & (((self.exg_s > 0.035) & ~green_roof) | ((exg_5 > 0.012) & (self.coh < 0.975) & ~green_roof)
                      | ((self.coh < 0.80) & (self.rough > 0.25)))
        osm_bld = self.osm.get('building')
        self.osm_inner = None
        if osm_bld is not None:
            # mapped footprint (shrunk by ~1 m against misalignment): high cells there are roof unless clearly foliage
            self.osm_inner = grid.erode(osm_bld > 0, k(2.0))
            veg &= ~(self.osm_inner & ~((self.exg_s > 0.06) & (self.rough > 0.2)))
        bld = grid.bin_open(high & ~veg, 5)
        veg = grid.bin_open(high & ~bld, 3)
        self.bld_px, self.veg_px = bld, veg
        self.low_px = valid & (ndsm > LOW_M) & (ndsm <= HIGH_M) & ~grid.dilate(bld | veg, 3)
        gnd = valid & ~(bld | veg)
        f, sat, exg_s = self.f, self.sat, self.exg_s
        water_level = float(np.percentile(self.dtm[valid], 2))
        gc = np.zeros(ndsm.shape, np.uint8)
        gc[gnd] = 5
        gc[gnd & (sat < 0.12)] = 1
        gc[gnd & (exg_s > 0.03)] = 2
        gc[gnd & (f[..., 0] > 0.45) & (sat > 0.12) & (exg_s <= 0.03)] = 4
        gc[gnd & (self.bright < 0.30) & (self.slope < 8) & (self.dtm < water_level + 1.5) & (exg_s < 0.06)] = 3
        for name, code in (('sidewalk', 8), ('path', 8), ('road', 6), ('parking', 7), ('water', 3)):
            layer = self.osm.get(name)
            if layer is not None:
                where = gnd & layer
                if name != 'water':
                    where &= gc != 2  # mapped road widths are nominal: keep grass verges
                gc[where] = code
        self.gc = gc
        cls = gc.copy()
        cls[bld] = 10
        cls[veg] = 11
        cls[self.low_px] = 12
        # raised cells the openings cut off (roof edges, wall tops, thin crowns) are not ground: they take the
        # class of the nearest building / vegetation cell (a path blocked there names a roof edge, not paving)
        edge = valid & (ndsm > HIGH_M) & ~(bld | veg)
        if edge.any() and (bld | veg).any():
            idx = native.nearest(bld | veg)
            r, c = np.nonzero(edge)
            src = idx[r, c]
            ok = src >= 0
            cls[r[ok], c[ok]] = cls.ravel()[src[ok]]
        self.edge_px = edge
        self.cls = cls

    def _buildings(self):
        g, res = self.g, self.g.res
        gy, gx = np.gradient(self.dsmf)
        step = np.hypot(gx, gy) > 0.8 * res / 0.25   # wall between parts of different height
        lab, n = native.label(self.bld_px & ~step, conn8=False)
        water = self.gc == 3
        osm_bld = self.osm.get('building')
        tags = self.osm.get('building_tags', {})
        pad = int(round(3.0 / res))
        self.building_ids = []
        for i, (rows, cols) in enumerate(grid.components(lab, n), 1):
            a = len(rows) * res * res
            if a < 12:
                continue
            c_coh = float(self.coh[rows, cols].mean())
            h = float(np.percentile(self.ndsm[rows, cols], 95))
            if c_coh < 0.93 and a < 200 and not self._osm_share(osm_bld, rows, cols) > 0.5:
                o = self._base('tree', 0.45, rows, cols)
                o.update(crown_m=round(2 * math.sqrt(a / math.pi), 2), note='non-green crown')
                self._add(o, rows, cols)
                continue
            r0, r1 = max(0, rows.min() - pad), min(g.h, rows.max() + pad + 1)
            c0, c1 = max(0, cols.min() - pad), min(g.w, cols.max() + pad + 1)
            m2 = np.zeros((r1 - r0, c1 - c0), bool)
            m2[rows - r0, cols - c0] = True
            ring = grid.dilate(m2, 2 * pad + 1) & ~m2
            wfrac = float(water[r0:r1, c0:c1][ring].mean()) if ring.any() else 0.0
            if wfrac > 0.08 and h < 10:
                o = self._base('bridge', 0.6, rows, cols)
                o['water_ring'] = round(wfrac, 2)
                self._add(o, rows, cols)
                continue
            rt, rconf, _ = _roof_type(self.nrm[rows, cols])
            if float(self.exg[rows, cols].mean()) > 0.01 and rt != 'flat' and a < 150 \
                    and not self._osm_share(osm_bld, rows, cols) > 0.5:
                o = self._base('tree', 0.45, rows, cols)
                o.update(crown_m=round(2 * math.sqrt(a / math.pi), 2), note='smooth crown')
                self._add(o, rows, cols)
                continue
            pieces = self._osm_split(osm_bld, rows, cols)
            if len(pieces) > 1:
                for oid, pr, pc in pieces:  # footprint per piece: traced from the piece alone
                    self._building(pr, pc, oid, tags, split=True)
            else:
                self._building(rows, cols, pieces[0][0] if pieces else None, tags, lab=lab, value=i)

    def _claim_edges(self, reach_m: float = 1.0):
        """Roof-edge cells left out at height steps (walls) belong to the nearest building within reach_m."""
        seed = self.objid >= 0
        todo = (self.bld_px | (self.edge_px & (self.cls == 10))) & ~seed
        if not seed.any() or not todo.any():
            return
        idx = native.nearest(seed)
        r, c = np.nonzero(todo)
        src = idx[r, c]
        ok = src >= 0
        r, c, src = r[ok], c[ok], src[ok]
        sr, sc = src // self.g.w, src % self.g.w
        near = np.hypot(sr - r, sc - c) * self.g.res <= reach_m
        self.objid[r[near], c[near]] = self.objid[sr[near], sc[near]]

    def _osm_share(self, osm_bld, rows, cols) -> float:
        if osm_bld is None:
            return 0.0
        return float((osm_bld[rows, cols] > 0).mean())

    def _osm_split(self, osm_bld, rows, cols):
        """[(osm id or None, rows, cols)]: one entry unless the component covers several mapped footprints."""
        if osm_bld is None:
            return []
        ids = osm_bld[rows, cols]
        cnt = np.bincount(ids[ids > 0]) if (ids > 0).any() else np.zeros(1, int)
        area = cnt * self.g.res ** 2
        keep = [k for k in np.flatnonzero(area >= 15) if cnt[k] >= 0.08 * len(rows)]
        if not keep:
            return []
        if len(keep) == 1:
            return [(int(keep[0]), rows, cols)] if cnt[keep[0]] >= 0.3 * len(rows) else []
        # assign every cell to the nearest kept footprint cell inside the component's window
        r0, c0 = rows.min(), cols.min()
        win = np.zeros((rows.max() - r0 + 1, cols.max() - c0 + 1), bool)
        seed = np.isin(ids, keep)
        win[rows[seed] - r0, cols[seed] - c0] = True
        idx = native.nearest(win)
        flat = idx[rows - r0, cols - c0]
        sr, sc = flat // win.shape[1] + r0, flat % win.shape[1] + c0
        owner = osm_bld[sr, sc]
        return [(int(k), rows[owner == k], cols[owner == k]) for k in keep if (owner == k).sum() * self.g.res ** 2 >= 8]

    def _building(self, rows, cols, osm_id, tags, split=False, lab=None, value=None):
        res = self.g.res
        a = len(rows) * res * res
        c_coh = float(self.coh[rows, cols].mean())
        nd = self.ndsm[rows, cols]
        h = float(np.percentile(nd, 95))
        conf = 0.5 + 0.2 * (a > 40) + 0.2 * (c_coh > 0.95) + 0.1 * (h > 4) + 0.1 * (osm_id is not None)
        rt, rconf, ridge = _roof_type(self.nrm[rows, cols])
        if lab is None:
            r0, c0 = rows.min(), cols.min()
            lab = np.zeros((rows.max() - r0 + 3, cols.max() - c0 + 3), np.int32)
            lab[rows - r0 + 1, cols - c0 + 1] = 1
            lab, n = native.label(lab > 0, conn8=False)
            big = int(np.argmax(np.bincount(lab.ravel())[1:])) + 1 if n else 1
            rr, cc = np.nonzero(lab == big)
            ring = grid.outline(lab, big, rr, cc)
            ring[:, 0] += c0 - 1
            ring[:, 1] += r0 - 1
            ring = grid.simplify(ring, 0.75 / res) if len(ring) > 3 else ring
            x, y = self.g.to_world(ring[:, 0], ring[:, 1])
            fp = [[round(float(p), 2), round(float(q), 2)] for p, q in zip(x, y)]
        else:
            fp = self._footprint(lab, value, rows, cols)
        o = self._base('building', min(conf, 0.95), rows, cols, src='mesh+osm' if osm_id is not None else 'mesh')
        eave = edge_eave(rows, cols, nd, res)
        o.update(eave_m=round(eave if eave is not None else float(np.percentile(nd, 15)), 2),
                 eave_from='edge' if eave is not None else 'roof_p15', ridge_m=round(float(np.percentile(nd, 99)), 2),
                 footprint=fp, parts=[], osm=tags.get(osm_id) if osm_id is not None else None, split_by_osm=split)
        self._roof(o, rows, cols, rt, rconf, ridge)
        bad = self._osm_conflict(rows, cols)
        if bad:
            o['osm_conflict'] = bad
        self._add(o, rows, cols)
        self.building_ids.append(o['id'])
        parts = _parts(cols, rows, res)
        if len(parts) > 1:
            for pc, pr in parts:
                prt, prconf, pridge = _roof_type(self.nrm[pr, pc])
                p = self._base('building_part', min(conf, 0.95), pr, pc, src=o['src'])
                pnd = self.ndsm[pr, pc]
                peave = edge_eave(pr, pc, pnd, res)
                p.update(parent=o['id'], eave_m=round(peave if peave is not None else float(np.percentile(pnd, 15)), 2),
                         ridge_m=round(float(np.percentile(pnd, 99)), 2), footprint=self._cells_ring(pr, pc))
                self._roof(p, pr, pc, prt, prconf, pridge)
                self._add(p)
                o['parts'].append(p['id'])

    def _cells_ring(self, rows, cols) -> list[list[float]]:
        """Simplified outline (world metres) of the largest connected piece of some cells."""
        r0, c0 = rows.min(), cols.min()
        lab = np.zeros((rows.max() - r0 + 3, cols.max() - c0 + 3), np.int32)
        lab[rows - r0 + 1, cols - c0 + 1] = 1
        lab, n = native.label(lab > 0, conn8=False)
        big = int(np.argmax(np.bincount(lab.ravel())[1:])) + 1 if n else 1
        rr, cc = np.nonzero(lab == big)
        ring = grid.outline(lab, big, rr, cc)
        ring[:, 0] += c0 - 1
        ring[:, 1] += r0 - 1
        ring = grid.simplify(ring, 0.75 / self.g.res) if len(ring) > 3 else ring
        x, y = self.g.to_world(ring[:, 0], ring[:, 1])
        return [[round(float(p), 2), round(float(q), 2)] for p, q in zip(x, y)]

    def _roof(self, o: dict, rows, cols, rt: str, rconf: float, ridge):
        """Roof label from the normals, refined by plane fits on the heights: a roof the normals call flat or
        mono-pitch becomes gabled / valley when two planes meet at a ridge / valley and slope at least FLAT_PCT
        percent; one plane is mono_pitch from MONO_PCT, else flat (roof_shape keeps the measured fall). Adds
        roof_shape (form, slope per plane, pitch, ridge / valley line with heights above ground)."""
        res = self.g.res
        r0, c0 = rows.min(), cols.min()
        m = np.zeros((rows.max() - r0 + 1, cols.max() - c0 + 1), bool)
        m[rows - r0, cols - c0] = True
        k = max(3, int(round(1.5 / res)) | 1)       # 0.75 m in from the edge: no parapets, gutters, wall tops
        inner = grid.erode(m, k) if min(m.shape) > 2 * k else m
        ir, ic = np.nonzero(inner)
        if len(ir) < 30:
            ir, ic = rows - r0, cols - c0
        ir, ic = ir + r0, ic + c0
        tilt_ok = self.nup[ir, ic] > math.cos(math.radians(50))
        ir, ic = ir[tilt_ok], ic[tilt_ok]
        fit = None
        if len(ir) >= 30:
            e, s = self.g.to_world(ic, ir)
            fit = roof_planes(np.asarray(e), np.asarray(s), self.dsmf[ir, ic].astype(np.float64), o['yaw_deg'],
                              float(np.median(self.dtm[rows, cols])))
        if fit is not None and rt in ('flat', 'mono_pitch'):
            if fit['form'] in ('gabled', 'valley'):
                rt, ridge = fit['form'], fit['line']['yaw_deg']
            elif fit['form'] == 'mono_pitch' and fit['slope_pct'][0] >= MONO_PCT:
                rt = 'mono_pitch'
            elif rt == 'mono_pitch' and fit['slope_pct'][0] < MONO_PCT:
                rt, ridge = 'flat', None   # a fall of a few percent (drained flat roof, garage): build it flat
        o.update(roof=rt, roof_conf=round(rconf, 2), ridge_yaw_deg=ridge)
        if fit is not None:
            o['roof_shape'] = fit

    def _osm_conflict(self, rows, cols) -> dict | None:
        """The disagreeing mapped footprint (see _osm_fit) under most of these cells, if any."""
        allb = self.osm.get('building_all')
        if allb is None or not self.osm_conflicts:
            return None
        ids = allb[rows, cols]
        ids = ids[np.isin(ids, list(self.osm_conflicts))]
        if not len(ids):
            return None
        i = int(np.bincount(ids).argmax())
        if (ids == i).sum() < 0.1 * len(rows):
            return None
        c = self.osm_conflicts[i]
        return {**c, 'why': f"the mapped footprint {c['osm']} covers only {c['cover']:.0%} raised cells: footprint, "
                            'split and walls come from the heights'}

    def _trees(self):
        res = self.g.res
        veg, ndsm = self.veg_px, self.ndsm
        nds = grid.gauss(np.where(veg, ndsm, 0), 0.75 / res).astype(np.float32)
        mx = grid.max_filter(nds, max(3, int(3.0 / res)))
        tops = veg & (nds == mx) & (nds > 2.5)
        tl, nt = native.label(tops, conn8=True)
        vl, _ = native.label(veg, conn8=False)
        idx = native.nearest(tops).ravel()
        flat_t, flat_v = tl.ravel(), vl.ravel()
        safe = np.maximum(idx, 0)
        crown = np.where(veg.ravel() & (idx >= 0) & (flat_v == flat_v[safe]), flat_t[safe], 0).reshape(veg.shape)
        osm_trees = self.osm.get('trees_cells')
        for rows, cols in grid.components(crown, nt):
            if len(rows) * res * res < 3:
                continue
            a = len(rows) * res * res
            ex = float(self.exg[rows, cols].mean())
            h = float(np.percentile(ndsm[rows, cols], 95))
            conf = 0.4 + 0.3 * (ex > 0.04) + 0.2 * (float(self.rough[rows, cols].mean()) > 0.2) + 0.1 * (h > 4)
            src = 'mesh'
            if osm_trees is not None and len(osm_trees):
                cy, cx = rows.mean(), cols.mean()
                if (np.hypot(osm_trees[:, 0] - cx, osm_trees[:, 1] - cy) * res < 2.5).any():
                    conf, src = conf + 0.1, 'mesh+osm'
            crown_m = 2 * math.sqrt(a / math.pi)
            verdict, host, why = self._on_roof(rows, cols, h, crown_m)
            if verdict == 'roof':      # green roof / roof edge read as a crown: the cells belong to the roof
                self.stats['trees_on_roofs'] = self.stats.get('trees_on_roofs', 0) + 1
                if host is not None:
                    free = self.objid[rows, cols] < 0
                    self.objid[rows[free], cols[free]] = host
                    self.cls[rows, cols] = 10
                continue
            if verdict == 'beside':
                conf -= 0.25
                self.stats['trees_beside_roofs'] = self.stats.get('trees_beside_roofs', 0) + 1
            o = self._base('tree', max(0.2, min(conf, 0.95)), rows, cols, src=src)
            o['crown_m'] = round(crown_m, 2)
            if why:
                o['note'] = why
            self._add(o, rows, cols)

    def _on_roof(self, rows, cols, h: float, crown_m: float) -> tuple[str | None, int | None, str | None]:
        """Is a crown candidate really roof? ('roof', building id or None, why) when it lies inside a mapped
        footprint or is a small or band-shaped crown touching a roof of its own height (green roof, roof edge); ('beside', ...)
        for a crown partly inside a footprint or against a roof of its height (conf is lowered); (None, None, None)."""
        g, res = self.g, self.g.res
        pad = int(round(1.5 / res)) + 1
        r0, r1 = max(0, rows.min() - pad), min(g.h, rows.max() + pad + 1)
        c0, c1 = max(0, cols.min() - pad), min(g.w, cols.max() + pad + 1)
        m = np.zeros((r1 - r0, c1 - c0), bool)
        m[rows - r0, cols - c0] = True
        ring = grid.dilate(m, 2 * int(round(1.0 / res)) + 1) & ~m
        bwin = self.bld_px[r0:r1, c0:c1] & ring
        touch = float(bwin.sum() / max(1, ring.sum()))
        dh = h - float(np.median(self.ndsm[r0:r1, c0:c1][bwin])) if bwin.sum() >= 4 else math.inf
        inside = float(self.osm_inner[rows, cols].mean()) if self.osm_inner is not None else 0.0
        host = None
        if bwin.any():
            ids = self.objid[r0:r1, c0:c1][bwin]
            ids = ids[ids >= 0]
            ids = ids[[self.objects[i]['cls'] == 'building' for i in ids]] if len(ids) else ids
            host = int(np.bincount(ids).argmax()) if len(ids) else None
        level = abs(dh) < 1.6
        _, _, L, Wd, _ = grid.obb(cols, rows)
        strip = L >= 2.5 * Wd      # a band along the roof edge, not a round crown
        if (inside >= 0.5 and crown_m < 6) or (touch >= 0.35 and level and (crown_m < 3.5 or strip)):
            return 'roof', host, None
        if inside >= 0.25 or (touch >= 0.2 and level):
            return 'beside', host, (f'{inside:.0%} inside a mapped footprint' if inside >= 0.25 else
                                    f'against a roof of the same height ({dh:+.1f} m): roof greenery?')
        return None, None, None

    def _split_cars(self, xs, ys, depth=0):
        """[(cx, cy, length, width, yaw, how)] in cells (centre-to-centre sizes) for one low blob."""
        res = self.g.res
        cx, cy, L, Wd, yaw = grid.obb(xs, ys)
        L, Wd = L - 1, Wd - 1  # centre-based extents, as the size gates were tuned
        Lm, Wm = L * res, Wd * res
        if CAR_L[0] <= Lm <= CAR_L[1] and CAR_W[0] <= Wm <= CAR_W[1]:
            return [(cx, cy, L, Wd, yaw, 'blob' if depth == 0 else 'dip')]
        if depth > 5 or Lm < 2.0:
            return []
        a = math.radians(yaw)
        U, V = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        pts = np.stack([xs, ys], 1).astype(float)
        vals = self.ndsm[ys, xs]
        best = None
        for D in (U, V):
            q = np.round((pts - [cx, cy]) @ D).astype(int)
            q0 = q.min()
            q -= q0
            if q.max() < 12:
                continue
            prof = np.bincount(q, vals) / np.maximum(np.bincount(q), 1)
            prof = np.convolve(prof, np.ones(3) / 3, 'same')
            for i in range(6, len(prof) - 6):
                lm, rm = prof[max(0, i - 10):i].max(), prof[i + 1:i + 11].max()
                depth_ = min(lm, rm) - prof[i]
                if prof[i] < 0.8 * min(lm, rm) and prof[i] == prof[max(0, i - 3):i + 4].min():
                    if best is None or depth_ > best[0]:
                        best = (depth_, D, i + q0)
        if best is not None:
            _, D, cut = best
            m = np.round((pts - [cx, cy]) @ D) < cut
            return self._split_cars(xs[m], ys[m], depth + 1) + self._split_cars(xs[~m], ys[~m], depth + 1)
        opts = []
        for pl, pw, along in ((4.4, 2.1, True), (2.1, 4.4, False)):
            nL, nW = max(1, round(Lm / pl)), max(1, round(Wm / pw))
            err = max(abs(math.log((Lm / nL) / pl)), abs(math.log(max(Wm, 1e-3) / nW / pw))) + 0.12 * (nL > 1 and nW > 1)
            opts.append((err, nL, nW, along))
        err, nL, nW, along = min(opts)
        if err > 0.35 or nL * nW > 12:
            return []
        out = []
        for iL in range(nL):
            for iW in range(nW):
                du, dv = ((iL + 0.5) / nL - 0.5) * L, ((iW + 0.5) / nW - 0.5) * Wd
                c = (cx + du * U[0] + dv * V[0], cy + du * U[1] + dv * V[1])
                if along:
                    out.append((c[0], c[1], L / nL, Wd / nW, yaw, 'tiled'))
                else:
                    out.append((c[0], c[1], Wd / nW, L / nL, (yaw + 90) % 180, 'tiled'))
        return out

    def _cars(self):
        g, res = self.g, self.g.res
        lab, n = native.label(self.low_px, conn8=False)
        osm_bld = self.osm.get('building')
        pad = int(round(1.5 / res))
        for rows, cols in grid.components(lab, n):
            a = len(rows) * res * res
            if a < 1.5 or a > 90:
                continue
            _, _, L, Wd, _ = grid.obb(cols, rows)
            h = float(np.percentile(self.ndsm[rows, cols], 95))
            ex = float(self.exg[rows, cols].mean())
            r0, r1 = max(0, rows.min() - pad), min(g.h, rows.max() + pad + 1)
            c0, c1 = max(0, cols.min() - pad), min(g.w, cols.max() + pad + 1)
            m2 = np.zeros((r1 - r0, c1 - c0), bool)
            m2[rows - r0, cols - c0] = True
            ring = grid.dilate(m2, 9) & ~m2
            ring_low = float((self.ndsm[r0:r1, c0:c1][ring] < 0.4).mean()) if ring.any() else 0.0
            gcr = self.gc[r0:r1, c0:c1][ring]
            ring_paved = float(np.isin(gcr, (1, 6, 7)).mean()) if ring.any() else 0.0
            ring_carland = float(np.isin(gcr, (6, 7)).mean()) if ring.any() else 0.0
            fill = len(rows) / max(L * Wd, 1e-6)
            in_bld = self._osm_share(osm_bld, rows, cols) > 0.5
            ok_h = 0.8 <= h <= 2.3 and ex < 0.04 and fill > 0.5 and not in_bld
            pieces = self._split_cars(cols, rows) if ok_h else []
            if not pieces and ok_h and h >= 1.0 and fill > 0.6 and ring_paved > 0.5:
                # a car cut short by a crown or shadow edge: one partial blob on paving
                cx, cy, cl, cw, yaw = grid.obb(cols, rows)
                if 2.2 <= (cl - 1) * res <= CAR_L[1] and 1.0 <= (cw - 1) * res <= CAR_W[1]:
                    pieces = [(cx, cy, cl - 1, cw - 1, yaw, 'partial')]
            if pieces:
                base = 0.4 + 0.2 * (ring_low > 0.6) + 0.2 * (ring_paved > 0.4) + 0.1 * (ring_carland > 0.3)
                for cx, cy, cl, cw, yaw, how in pieces:
                    clm, cwm = (cl + 1) * res, (cw + 1) * res
                    good = 3.85 <= clm <= 5.65 and 1.75 <= cwm <= 2.55
                    conf = min(0.95, (base + 0.15 * good) * {'blob': 1.0, 'dip': 0.9, 'tiled': 0.75}.get(how, 0.6))
                    o = self._base('car', conf, rows, cols, src='mesh+osm' if ring_carland > 0.3 else 'mesh',
                                   box=(cx, cy, cl + 1, cw + 1, yaw))
                    o.update(split=how, ring_low=round(ring_low, 2), ring_paved=round(ring_paved, 2))
                    o['area_m2'] = round(clm * cwm, 1)
                    corners = grid.box_corners(cx, cy, cl + 1, cw + 1, yaw)
                    self._add(o, poly=corners)
            elif ex < 0.04:
                o = self._base('low_structure' if L * res > 6 or in_bld else 'small_object', 0.3, rows, cols)
                o['fill'] = round(fill, 2)
                if in_bld:
                    o['note'] = 'inside a mapped building (carport / shelter?)'
                self._add(o, rows, cols)
            else:
                self._add(self._base('shrub', 0.4, rows, cols), rows, cols)

    # ------------------------------------------------------------------ summary
    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for o in self.objects:
            counts[o['cls']] = counts.get(o['cls'], 0) + 1
        valid = max(1, int(self.valid.sum()))
        shares = {CLASSES[k]: round(float((self.cls == k).sum() / valid), 3) for k in CLASSES if k}
        return {'counts': counts, 'class_share': {k: v for k, v in shares.items() if v}, 'stats': dict(self.stats)}
