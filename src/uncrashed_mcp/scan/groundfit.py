"""Ground as continuous surfaces: fit smooth surfaces to a noisy scan ground model, pave them with large tilted tiles
whose shared corners come from the same surface, and audit the height steps between neighbouring placed tiles.

A photogrammetry ground model is noisy (5..15 cm), blurred at kerbs and dented by cars and crowns. Taking its height
per cell turns a street into a patchwork of small steps. Here the ground is split into islands (cells connected
edge to edge) and two levels: carriageway (road, parking) and kerbside (everything else); a region is a connected
piece of one level. Per island one continuous surface is fitted robustly (Huber) to the open-ground samples:
- one plane (with the kerb) when its robust residual (90th percentile) stays within flat_tol_m;
- else a thin-plate surface on the lattice knots (smoothing length smooth_m; bending across a cell, the twist that
  would make a tile's four corners leave one plane, costs TWIST times more), where every region that alone fits a
  plane within flat_tol_m is STIFF times stiffer, so a street stays a plane and the ground around it bends.
Tile corners are knot heights of that one surface, so neighbouring tiles meet (the twist left in a cell is a few
millimetres). The kerbside level sits one kerb above the carriageway: the fitted offset, 0 when it is under half
the smallest kerb, else clipped to KERB_M. Pure numpy; nothing here talks to the game.
"""
from __future__ import annotations

import contextlib
import ctypes
import glob
import math
import os

import numpy as np

from .. import geom

CARRIAGEWAY = ('road', 'parking')
KERB_M = (0.08, 0.20)          # a kerb between carriageway and kerbside: none, or this many metres
SAMPLE_M = 0.5                 # analysis grid of the ground model
SEAM_M = 0.004                 # merged tiles: lattice knots on their border within this of the tile plane
INNER_M = 0.03                 # ... and knots inside them (no neighbour there: only fidelity to the surface)
TWIST = 10.0                   # extra cost of twist (non-planar cells) over plain bending
STIFF = 10.0                   # extra bending cost inside regions that fit a plane
ROAD_STIFF = 3.0               # ... and on carriageway cells that do not (roads bend less than gardens)
BREAK_M = 0.35                 # neighbouring cells whose ground differs more: a wall or embankment, bent freely
SOFT = 0.05                    # bending cost on both sides of such a break (twist stays stiff: tiles are planes)
MAX_SLOPE = 0.4                # a plane steeper than this is a wall in the samples: level at the median instead


def level_of(name: str | None) -> int:
    return 0 if name in CARRIAGEWAY else 1


# ---------------------------------------------------------------------------- robust fits


@contextlib.contextmanager
def _one_blas_thread():
    """numpy's OpenBLAS on one thread: the many small (~300 x 300) solves of the thin-plate system run 10..40 times
    slower when every call wakes a pool of threads. No-op when the library or its symbols are not found."""
    setter = None
    try:
        base = os.path.dirname(np.__file__)
        for path in sorted(glob.glob(os.path.join(base, os.pardir, 'numpy.libs', '*openblas*')) +
                           glob.glob(os.path.join(base, '.dylibs', '*openblas*'))):
            lib = ctypes.CDLL(path)
            for pre in ('scipy_openblas_', 'openblas_'):
                for suf in ('64_', ''):
                    if hasattr(lib, f'{pre}set_num_threads{suf}'):
                        setter = getattr(lib, f'{pre}set_num_threads{suf}')
                        old = int(getattr(lib, f'{pre}get_num_threads{suf}')())
                        break
                if setter:
                    break
            if setter:
                break
    except (OSError, AttributeError):
        setter = None
    if setter is None:
        yield
        return
    setter(1)
    try:
        yield
    finally:
        setter(old)


def _huber(r: np.ndarray, floor: float = 0.03) -> np.ndarray:
    s = max(1.4826 * float(np.median(np.abs(r))) if r.size else floor, floor)
    k = 1.5 * s
    a = np.abs(r)
    return np.where(a <= k, 1.0, k / np.maximum(a, 1e-9))


def snap_kerb(o: float, kerb=KERB_M) -> float:
    lo, hi = kerb
    if not math.isfinite(o) or abs(o) < lo / 2:
        return 0.0
    return math.copysign(min(max(abs(o), lo), hi), o)


def _wls(A: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)
    return np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)[0]


def flatness(r: np.ndarray, cell: np.ndarray) -> float:
    """Robust residual of a fit: 95th percentile over lattice cells (3+ samples) of the median |residual| in the
    cell, so sample noise and a few dents do not count but a sunken lane does."""
    if not len(r):
        return 0.0
    order = np.argsort(cell, kind='stable')
    c, a = cell[order], np.abs(r[order])
    cuts = np.flatnonzero(np.diff(c)) + 1
    meds = [float(np.median(g)) for g in np.split(a, cuts) if len(g) >= 3]
    return float(np.percentile(meds, 95)) if meds else float(np.median(a))


def plane_fit(u, v, z, lev, kerb=KERB_M, it: int = 10, raw: float | None = None):
    """Robust plane + kerbside offset: (a, bu, bv), kerb (snapped), raw offset (fitted unless given), residuals,
    weights."""
    mixed = bool(lev.any() and (~lev).any())
    A = np.column_stack([np.ones_like(u), u, v] + ([lev.astype(float)] if mixed else []))
    w = np.ones_like(z)
    for _ in range(it):
        coef = _wls(A, z, w)
        w = _huber(z - A @ coef)
    if raw is None:
        raw = float(coef[3]) if mixed else 0.0
    k = snap_kerb(raw, kerb) if mixed else 0.0
    y = z - k * lev
    A3 = A[:, :3]
    for _ in range(4):
        c3 = _wls(A3, y, w)
        r = y - A3 @ c3
        w = _huber(r)
    return c3, k, raw, r, w


def _stencils(cells: np.ndarray, stiff_cell: np.ndarray):
    """Bending stencils on the knots of the given cells (H x W bool): (knot index arrays, coefficients, cell
    stiffness) for d2/du2, d2/dv2 (three knots in a row, all on the cells) and the twist of every cell."""
    H, W = cells.shape
    K = W + 1
    knot = np.zeros((H + 1, W + 1), bool)
    for dj in (0, 1):
        for di in (0, 1):
            knot[dj:dj + H, di:di + W] |= cells
    st_s = np.zeros((H + 1, W + 1))
    for dj in (0, 1):
        for di in (0, 1):
            st_s[dj:dj + H, di:di + W] = np.maximum(st_s[dj:dj + H, di:di + W], np.where(cells, stiff_cell, 0))
    soft = cells & (stiff_cell < 1)              # break cells: their knots bend freely (an embankment ramp)
    for dj in (0, 1):
        for di in (0, 1):
            st_s[dj:dj + H, di:di + W] = np.where(soft, np.minimum(st_s[dj:dj + H, di:di + W], stiff_cell),
                                                  st_s[dj:dj + H, di:di + W])
    rows = []
    jj, ii = np.nonzero(knot[:, 1:-1] & knot[:, :-2] & knot[:, 2:])
    ii = ii + 1
    rows.append((np.stack([jj * K + ii - 1, jj * K + ii, jj * K + ii + 1], 1), np.array([1., -2., 1.]), st_s[jj, ii], 1.0))
    jj, ii = np.nonzero(knot[1:-1, :] & knot[:-2, :] & knot[2:, :])
    jj = jj + 1
    rows.append((np.stack([(jj - 1) * K + ii, jj * K + ii, (jj + 1) * K + ii], 1), np.array([1., -2., 1.]), st_s[jj, ii], 1.0))
    jj, ii = np.nonzero(cells)
    rows.append((np.stack([jj * K + ii, jj * K + ii + 1, (jj + 1) * K + ii + 1, (jj + 1) * K + ii], 1),
                 np.array([1., -1., 1., -1.]), np.where(stiff_cell[jj, ii] < 1, STIFF, stiff_cell[jj, ii]), 2.0 * TWIST))
    return knot, rows


class _BlockSystem:
    """Symmetric system on a lattice of knots (nrows rows of K knots, id j * K + i) whose entries couple knot rows at
    most two apart: rows are paired into blocks of 2K unknowns, which makes it block-tridiagonal, solved exactly by
    block elimination (fast at any lattice size, no iteration). The fixed part (the bending penalty) is given here;
    solve adds the data part."""

    def __init__(self, K: int, nrows: int, r, c, v):
        self.B, self.nb = 2 * K, (nrows + 1) // 2
        self.D0, self.L0 = self._blocks(r, c, v)

    def _blocks(self, r, c, v):
        B, nb = self.B, self.nb
        br, bc = r // B, c // B
        m = br == bc
        D = np.bincount((br[m] * B + r[m] % B) * B + c[m] % B, v[m], nb * B * B).reshape(nb, B, B)
        m = br == bc + 1
        L = np.bincount((br[m] * B + r[m] % B) * B + c[m] % B, v[m], nb * B * B).reshape(nb, B, B)
        return D, L

    def solve(self, r, c, v, rhs, off) -> np.ndarray:
        B, nb = self.B, self.nb
        D, L = self._blocks(r, c, v)
        D += self.D0
        L += self.L0
        n = nb * B
        diag = np.zeros(n)
        diag[off[off < n]] = 1.0
        diag[len(rhs):] = 1.0                   # padding of an odd last row pair
        diag += 1e-6
        D[:, np.arange(B), np.arange(B)] += diag.reshape(nb, B)
        y = np.zeros(n)
        y[:len(rhs)] = rhs
        y = y.reshape(nb, B)
        S = [D[0]]
        for b in range(1, nb):                  # forward elimination
            M = np.linalg.solve(S[-1], L[b].T).T      # L_b S_{b-1}^-1 (S symmetric)
            S.append(D[b] - M @ L[b].T)
            y[b] = y[b] - M @ y[b - 1]
        x = np.zeros((nb, B))
        x[-1] = np.linalg.solve(S[-1], y[-1])
        for b in range(nb - 2, -1, -1):
            x[b] = np.linalg.solve(S[b], y[b] - L[b + 1].T @ x[b + 1])
        return x.ravel()[:len(rhs)]


def thin_plate(u, v, z, lev, cells, stiff_cell, gu, gv, smooth_m: float, k: float = 0.0, it: int = 5):
    """Robust thin-plate surface on the knots of `cells` (H x W bool on the lattice gu x gv) through samples u, v,
    z (lev True: kerbside, k above the surface). Returns (Z (H+1, W+1), NaN off the cells; residuals)."""
    H, W = cells.shape
    K = W + 1
    h = float(gu[1] - gu[0])
    knot, rows = _stencils(cells, stiff_cell)
    ids = np.flatnonzero(knot.ravel())
    pos = np.full(knot.size, -1)
    pos[ids] = np.arange(len(ids))
    n = len(ids)
    fi = np.clip((u - gu[0]) / h, 0, W - 1e-6)
    fj = np.clip((v - gv[0]) / h, 0, H - 1e-6)
    i, j = np.floor(fi).astype(int), np.floor(fj).astype(int)
    a, b = fi - i, fj - j
    idx = pos[np.stack([j * K + i, j * K + i + 1, (j + 1) * K + i + 1, (j + 1) * K + i], 1)]
    co = np.stack([(1 - a) * (1 - b), a * (1 - b), a * b, (1 - a) * b], 1)
    good = (idx >= 0).all(1)
    idx, co, z, lev = idx[good], co[good], z[good], lev[good]
    lam = 2.0 * smooth_m ** 4 / h ** 2          # ~2 samples per m2 of open ground; continuous length smooth_m
    gidx = ids[idx]                              # global knot ids (row j of knots: j * K .. j * K + K - 1)
    pr_, pc_, pv_ = [], [], []
    for r, c, s, f in rows:
        wt = lam * f * s
        for p in range(r.shape[1]):
            for q in range(r.shape[1]):
                pr_.append(r[:, p])
                pc_.append(r[:, q])
                pv_.append(wt * c[p] * c[q])
    pen = _BlockSystem(K, H + 1, np.concatenate(pr_), np.concatenate(pc_), np.concatenate(pv_))
    off = np.setdiff1d(np.arange(knot.size), ids)
    dr = np.concatenate([gidx[:, p] for p in range(4) for q in range(4)])
    dc = np.concatenate([gidx[:, q] for p in range(4) for q in range(4)])
    w = np.ones_like(z)
    x = np.zeros(knot.size)
    with _one_blas_thread():
        for _rep in range(it):
            y = z - k * lev
            rhs = np.bincount(gidx.ravel(), (co * (w * y)[:, None]).ravel(), knot.size)
            dv = np.concatenate([w * co[:, p] * co[:, q] for p in range(4) for q in range(4)])
            x = pen.solve(dr, dc, dv, rhs, off)
            xr = x[ids]
            res = y - (co * xr[idx]).sum(1)
            w = _huber(res)
    Z = np.full(knot.size, np.nan)
    Z[ids] = xr
    full = np.full(len(good), np.nan)
    full[good] = res
    return Z.reshape(H + 1, W + 1), full


def breaks(z, scell, cells, step_m: float | None = None) -> np.ndarray:
    """Cells (H x W bool) on either side of a wall or embankment: the median ground of two edge neighbours (3+
    samples each) differs by more than step_m (default BREAK_M; kerbs are lower and already taken out of z)."""
    step_m = BREAK_M if step_m is None else step_m
    H, W = cells.shape
    order = np.argsort(scell, kind='stable')
    c, zz = scell[order], z[order]
    cuts = np.flatnonzero(np.diff(c)) + 1
    med = np.full(H * W, np.nan)
    for g, idx in zip(np.split(zz, cuts), np.split(c, cuts)):
        if len(g) >= 3:
            med[idx[0]] = np.median(g)
    med = med.reshape(H, W)
    med[~cells] = np.nan
    out = np.zeros((H, W), bool)
    with np.errstate(invalid='ignore'):
        d = np.abs(med[:, 1:] - med[:, :-1]) > step_m
        out[:, 1:] |= d
        out[:, :-1] |= d
        d = np.abs(med[1:, :] - med[:-1, :]) > step_m
        out[1:, :] |= d
        out[:-1, :] |= d
    return out


def fit_island(u, v, z, lev, cell_lev, gu, gv, flat_tol_m: float = 0.3, smooth_m: float = 4.0,
               kerb=KERB_M, kerb_raw: float | None = None) -> dict:
    """Surface of one island. u, v, z, lev: its reliable samples (lev True = kerbside); cell_lev (H x W): level per
    cell of the island, -1 elsewhere. Z (H+1, W+1) is the carriageway level at the knots (the kerbside level where
    the island has no carriageway), kerb_m the kerbside offset: kerb_raw (measured across the level boundaries)
    snapped, or the plane's offset when not given."""
    u, v, z = (np.asarray(t, float) for t in (u, v, z))
    lev = np.asarray(lev, bool)
    has_road = bool((cell_lev == 0).any())
    if not has_road or not (~lev).any():
        lev = np.zeros_like(lev)
    H, W = cell_lev.shape
    h = float(gu[1] - gu[0])
    ci = np.clip(np.floor((u - gu[0]) / h).astype(int), 0, W - 1)
    cj = np.clip(np.floor((v - gv[0]) / h).astype(int), 0, H - 1)
    scell = cj * W + ci
    c, k, raw, r, w = plane_fit(u, v, z, lev, kerb, raw=kerb_raw if has_road else None)
    flat = flatness(r, scell)
    out = {'samples': int(len(z)), 'kerb_measured_m': round(raw, 3), 'plane_resid_m': round(flat, 3)}
    brk = breaks(z - k * lev, scell, cell_lev >= 0)
    if flat <= flat_tol_m and not brk.any():      # a wall or embankment inside: not one plane
        if math.hypot(c[1], c[2]) > MAX_SLOPE:
            c = np.array([float(np.median(z - k * lev)), 0.0, 0.0])
        Z = c[0] + c[1] * gu[None, :] + c[2] * gv[:, None]
        return {**out, 'mode': 'plane', 'Z': Z, 'kerb_m': round(k, 3), 'resid_m': round(flat, 3),
                'slope_pct': round(100 * math.hypot(c[1], c[2]), 2)}
    # regions that alone fit a plane are stiff, carriageway a little stiffer than the rest
    reg, n = label(cell_lev >= 0, cell_lev)
    sreg = reg[cj, ci]
    stiff = np.where(cell_lev == 0, ROAD_STIFF, 1.0)
    planar = 0
    regions = []
    for q in range(n):
        m = sreg == q
        row = {'level': int(cell_lev[reg == q][0]), 'cells': int((reg == q).sum()), 'samples': int(m.sum())}
        if m.sum() >= 12:
            *_, rq, _w = plane_fit(u[m], v[m], z[m], np.zeros(int(m.sum()), bool), kerb)
            row['plane_resid_m'] = round(flatness(rq, scell[m]), 3)
            if row['plane_resid_m'] <= flat_tol_m:
                stiff[reg == q] = STIFF
                planar += 1
                row['plane'] = True
        regions.append(row)
    stiff[brk] = SOFT
    Z, res = thin_plate(u, v, z, lev, cell_lev >= 0, stiff, gu, gv, smooth_m, k)
    return {**out, 'mode': 'smooth', 'Z': Z, 'kerb_m': round(k, 3), 'regions': n, 'planar_regions': planar,
            'break_cells': int(brk.sum()),
            'region_fits': sorted(regions, key=lambda r: -r['cells'])[:8],
            'resid_m': round(flatness(res[np.isfinite(res)], scell[np.isfinite(res)]), 3)}


# ---------------------------------------------------------------------------- lattice bookkeeping


def label(mask: np.ndarray, same=None, diagonal: bool = False) -> tuple[np.ndarray, int]:
    """4-connected (diagonal: 8-connected) components of mask (and of equal `same` values when given); -1
    outside."""
    lab = np.full(mask.shape, -1, int)
    n = 0
    H, W = mask.shape
    near = [(1, 0), (-1, 0), (0, 1), (0, -1)] + ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if diagonal else [])

    def steps(j, i):
        return [(j + a, i + b) for a, b in near]
    for j0 in range(H):
        for i0 in range(W):
            if not mask[j0, i0] or lab[j0, i0] >= 0:
                continue
            lab[j0, i0] = n
            stack = [(j0, i0)]
            while stack:
                j, i = stack.pop()
                for jj, ii in steps(j, i):
                    if 0 <= jj < H and 0 <= ii < W and mask[jj, ii] and lab[jj, ii] < 0 and \
                            (same is None or same[jj, ii] == same[j0, i0]):
                        lab[jj, ii] = n
                        stack.append((jj, ii))
            n += 1
    return lab, n


def clean_levels(key: np.ndarray, min_cells: int, level=level_of) -> int:
    """Pieces of one level smaller than min_cells take the commonest key around them (a 'paved' cell in the road
    would become a raised block). Changes key in place; returns the number of cells changed."""
    lev = np.vectorize(lambda k: -1 if k is None else level(k), otypes=[int])(key)
    lab, n = label(lev >= 0, lev)
    changed = 0
    H, W = key.shape
    for c in range(n):
        cells = np.argwhere(lab == c)
        if len(cells) >= min_cells:
            continue
        around: dict = {}
        for j, i in cells:
            for jj, ii in ((j + 1, i), (j - 1, i), (j, i + 1), (j, i - 1)):
                if 0 <= jj < H and 0 <= ii < W and lab[jj, ii] != c and key[jj, ii] is not None:
                    around[key[jj, ii]] = around.get(key[jj, ii], 0) + 1
        if around:
            k = max(around, key=around.get)
            for j, i in cells:
                key[j, i] = k
            changed += len(cells)
    return changed


def clean_keys(key: np.ndarray, min_cells: int, level=level_of) -> int:
    """Pieces of one material smaller than min_cells take the commonest material of the same level around them
    (fewer, larger tiles; a speck of path in a pavement is classification noise). In place; cells changed."""
    H, W = key.shape
    names = sorted({k for k in key.ravel() if k is not None})
    code = np.vectorize(lambda k: -1 if k is None else names.index(k), otypes=[int])(key)
    lab, n = label(code >= 0, code)
    changed = 0
    for c in range(n):
        cells = np.argwhere(lab == c)
        if len(cells) >= min_cells:
            continue
        own = level(key[cells[0][0], cells[0][1]])
        around: dict = {}
        for j, i in cells:
            for jj, ii in ((j + 1, i), (j - 1, i), (j, i + 1), (j, i - 1)):
                if 0 <= jj < H and 0 <= ii < W and lab[jj, ii] != c and key[jj, ii] is not None and \
                        level(key[jj, ii]) == own:
                    around[key[jj, ii]] = around.get(key[jj, ii], 0) + 1
        if around:
            k = max(around, key=around.get)
            for j, i in cells:
                key[j, i] = k
            changed += len(cells)
    return changed


def _planar(Z: np.ndarray, gu, gv, tol: float, inner: float | None = None) -> bool:
    """Do the knots of Z (a sub-lattice) lie on the plane through three of its corners: the border knots (shared
    with the neighbouring tiles: a seam) within tol, the inner ones (only the fidelity to the surface) within inner
    (default INNER_M)?"""
    if Z.shape[0] < 2 or Z.shape[1] < 2:
        return True
    du, dv = gu[-1] - gu[0], gv[-1] - gv[0]
    a = (Z[0, -1] - Z[0, 0]) / du
    b = (Z[-1, 0] - Z[0, 0]) / dv
    D = np.abs(Z[0, 0] + a * (gu[None, :] - gu[0]) + b * (gv[:, None] - gv[0]) - Z)
    rim = max(float(D[0].max()), float(D[-1].max()), float(D[:, 0].max()), float(D[:, -1].max()))
    return rim <= tol and float(D.max()) <= (INNER_M if inner is None else inner)


def rectangles(key: np.ndarray, island: np.ndarray, Z: dict, gu, gv, max_u: int, max_v: int,
               tol: float | None = None) -> list[tuple]:
    """Greedy maximal rectangles of equal key and island whose knots are coplanar (Z[island] knot heights):
    (i0, i1, j0, j1, key, island) with cells i0..i1-1, j0..j1-1."""
    H, W = key.shape
    tol = SEAM_M if tol is None else tol
    done = np.zeros((H, W), bool)
    out = []
    for j0 in range(H):
        for i0 in range(W):
            k = key[j0, i0]
            if k is None or done[j0, i0]:
                continue
            isl = island[j0, i0]
            Zi = Z[isl]

            def ok(i, j):
                return key[j, i] == k and island[j, i] == isl and not done[j, i]
            i1 = i0 + 1
            while i1 < W and i1 - i0 < max_u and ok(i1, j0) and \
                    _planar(Zi[j0:j0 + 2, i0:i1 + 2], gu[i0:i1 + 2], gv[j0:j0 + 2], tol):
                i1 += 1
            j1 = j0 + 1
            while j1 < H and j1 - j0 < max_v and all(ok(i, j1) for i in range(i0, i1)) and \
                    _planar(Zi[j0:j1 + 2, i0:i1 + 1], gu[i0:i1 + 1], gv[j0:j1 + 2], tol):
                j1 += 1
            done[j0:j1, i0:i1] = True
            out.append((i0, i1, j0, j1, k, isl))
    return out


def _kerb_across(lev: np.ndarray, isl: np.ndarray, use, scell, sz) -> float | None:
    """Median over carriageway / kerbside cell pairs of an island of (kerbside median - carriageway median) of the
    usable samples of the two cells (the blurred strip at the boundary is already left out); None: no pairs."""
    H, W = lev.shape
    order = np.argsort(scell[use], kind='stable')
    cells, zs = scell[use][order], sz[use][order]
    start = np.searchsorted(cells, np.arange(H * W))
    end = np.searchsorted(cells, np.arange(H * W), side='right')

    def med(j, i):
        a, b = start[j * W + i], end[j * W + i]
        return float(np.median(zs[a:b])) if b - a >= 3 else None
    d = []
    for j, i in np.argwhere(isl & (lev == 0)):
        for jj, ii in ((j + 1, i), (j - 1, i), (j, i + 1), (j, i - 1)):
            if 0 <= jj < H and 0 <= ii < W and isl[jj, ii] and lev[jj, ii] == 1:
                a, b = med(j, i), med(jj, ii)
                if a is not None and b is not None:
                    d.append(b - a)
    return float(np.median(d)) if len(d) >= 3 else None


def pave(key: np.ndarray, gu: np.ndarray, gv: np.ndarray, su, sv, sz, ok, flat_tol_m: float = 0.3,
         smooth_m: float = 4.0, kerb=KERB_M, max_tile_m: float = 24.0, min_region_m2: float = 24.0,
         blur_m: float = 0.75, level=level_of, min_piece_m2: float = 12.0) -> tuple[list[dict], list[dict]]:
    """Tiles for a lattice of cells. key (len(gv)-1, len(gu)-1): material key per cell (None: not paved; level by
    `level`: 0 carriageway, 1 kerbside). su, sv, sz, ok: ground model samples in the lattice frame and whether each
    is open ground. Pieces of one level smaller than min_region_m2 and of one material smaller than min_piece_m2 are
    relabelled first (key changes in place). Returns (tiles, islands): tile = {'u': (u0, u1), 'v': (v0, v1), 'key',
    'z': [z00, z10, z11, z01] corner tops}, island = fit summary."""
    cm = float(gu[1] - gu[0])
    H, W = key.shape
    clean_levels(key, max(1, int(math.ceil(min_region_m2 / (cm * cm)))), level)
    clean_keys(key, max(1, int(math.ceil(min_piece_m2 / (cm * cm)))), level)
    lev = np.vectorize(lambda k: -1 if k is None else level(k), otypes=[int])(key)
    island, n = label(lev >= 0, diagonal=True)     # cells touching at a corner share that knot: one surface
    su, sv, sz, ok = (np.asarray(a) for a in (su, sv, sz, ok))
    fi, fj = (su - gu[0]) / cm, (sv - gv[0]) / cm
    ci, cj = np.floor(fi).astype(int), np.floor(fj).astype(int)
    inside = (ci >= 0) & (ci < W) & (cj >= 0) & (cj < H)
    ci, cj = np.clip(ci, 0, W - 1), np.clip(cj, 0, H - 1)
    slev = np.where(inside, lev[cj, ci], -1)
    sisl = np.where(inside, island[cj, ci], -1)
    # kerb blur: samples within blur_m of an edge to a cell of the other level
    near = np.zeros(len(sz), bool)
    du, dv = (fi - np.floor(fi)) * cm, (fj - np.floor(fj)) * cm
    for di, dj, dist in ((1, 0, cm - du), (-1, 0, du), (0, 1, cm - dv), (0, -1, dv)):
        ni, nj = np.clip(ci + di, 0, W - 1), np.clip(cj + dj, 0, H - 1)
        other = lev[nj, ni]
        near |= (other >= 0) & (other != slev) & (dist < blur_m)
    use = inside & ok & (slev >= 0) & ~near & np.isfinite(sz)
    Z, islands, kerbs = {}, [], {}
    for c in range(n):
        m = use & (sisl == c)
        cells = int((island == c).sum())
        if m.sum() < 12:                       # hardly any open ground: every sample in the island, one plane
            m = inside & (sisl == c) & np.isfinite(sz)
        if m.sum() < 3:
            z0 = float(np.nanmedian(sz[inside])) if inside.any() else 0.0
            Z[c] = np.full((H + 1, W + 1), z0)
            kerbs[c] = 0.0
            islands.append({'island': c, 'cells': cells, 'samples': int(m.sum()), 'mode': 'constant'})
            continue
        f = fit_island(su[m], sv[m], sz[m], slev[m] == 1, np.where(island == c, lev, -1), gu, gv, flat_tol_m,
                       smooth_m, kerb, kerb_raw=_kerb_across(lev, island == c, use, cj * W + ci, sz))
        Z[c], kerbs[c] = f.pop('Z'), f['kerb_m'] if (lev[island == c] == 0).any() else 0.0
        islands.append({'island': c, 'cells': cells, **f})
    k = max(1, int(round(max_tile_m / cm)))
    tiles = []
    for i0, i1, j0, j1, kk, isl in rectangles(key, island, Z, gu, gv, k, k):
        off = kerbs[isl] if level(kk) == 1 else 0.0
        zz = Z[isl]
        tiles += _split({'u': (float(gu[i0]), float(gu[i1])), 'v': (float(gv[j0]), float(gv[j1])), 'key': kk,
                         'island': int(isl),
                         'z': [float(zz[j0, i0] + off), float(zz[j0, i1] + off), float(zz[j1, i1] + off),
                               float(zz[j1, i0] + off)]})
    return tiles, islands


def _split(t: dict, depth: int = 2) -> list[dict]:
    """A tile whose four corners leave one plane by more than SEAM_M (a twisted cell of a bent surface) as four
    tiles on its bilinear surface: each has a quarter of the twist, and the new edge midpoints lie on the straight
    edges the neighbours share, so they still meet."""
    z00, z10, z11, z01 = t['z']
    if depth == 0 or abs(z00 - z10 + z11 - z01) / 4 <= SEAM_M:
        return [t]
    (u0, u1), (v0, v1) = t['u'], t['v']
    um, vm = (u0 + u1) / 2, (v0 + v1) / 2
    zb, zr, zt, zl, zc = (z00 + z10) / 2, (z10 + z11) / 2, (z01 + z11) / 2, (z00 + z01) / 2, sum(t['z']) / 4
    out = []
    for uu, vv, zq in (((u0, um), (v0, vm), [z00, zb, zc, zl]), ((um, u1), (v0, vm), [zb, z10, zr, zc]),
                       ((um, u1), (vm, v1), [zc, zr, z11, zt]), ((u0, um), (vm, v1), [zl, zc, zt, z01])):
        out += _split({**t, 'u': uu, 'v': vv, 'z': zq}, depth - 1)
    return out


# ---------------------------------------------------------------------------- audit of placed tiles


def top_face(xf, asset) -> np.ndarray:
    """World corners (4 x 3, metres, around the face) of the top of a placed row's mesh bounds box. xf: native
    transform (x, y, z cm, pitch, yaw, roll, sx, sy, sz)."""
    q = geom.quat_rotator(float(xf[3]), float(xf[4]), float(xf[5]))
    o, e, s = np.array(asset.origin_cm, float), np.array(asset.extent_cm, float), np.array(xf[6:9], float)
    pts = []
    for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        loc = (o + np.array([su, sv, 1.0]) * e) * s
        pts.append((np.array(xf[:3], float) + np.array(geom.quat_rotate(q, loc))) / 100)
    P = np.array(pts)
    if np.cross(P[1] - P[0], P[3] - P[0])[2] < 0:        # mirrored or upside down: the other box face is on top
        pts = []
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            loc = (o + np.array([su, sv, -1.0]) * e) * s
            pts.append((np.array(xf[:3], float) + np.array(geom.quat_rotate(q, loc))) / 100)
        P = np.array(pts)
    return P


def is_tile(asset, xf, min_side_m: float = 1.0) -> bool:
    """A flat ground tile (not a marking strip, decal or foliage): a ground-tagged mesh whose two horizontal sides
    are at least min_side_m."""
    if asset is None or asset.kind != 'mesh' or 'ground' not in (asset.tags or ()):
        return False
    sx, sy = abs(float(xf[6])) * asset.extent_cm[0] / 50, abs(float(xf[7])) * asset.extent_cm[1] / 50
    return min(sx, sy) >= min_side_m


def _stats(v: list[float]) -> dict:
    if not v:
        return {'pairs': 0}
    a = np.abs(np.array(v))
    return {'pairs': int(a.size), 'over_1cm': int((a > 0.01).sum()), 'median_m': round(float(np.median(a)), 3),
            'p90_m': round(float(np.percentile(a, 90)), 3), 'max_m': round(float(a.max()), 3)}


def edge_steps(faces: list[dict], step_m: float = 0.5, near_m: float = 0.05, edge_m: float = 0.10,
               max_dz_m: float = 1.5, top: int = 10) -> dict:
    """Height steps between neighbouring tile tops. faces: {'P': 4 x 3 corners, 'row': ..., 'id': ...}. Points
    every step_m along each tile edge that lie on another tile's edge (within edge_m of its border, at most near_m
    outside it) give dz = this tile's height there minus the other's; per pair of tiles the largest |dz| counts.
    Steps over max_dz_m are different levels (a roof slab, a cellar), not neighbours."""
    F = []
    for f in faces:
        P = np.asarray(f['P'], float)
        a, b = P[1, :2] - P[0, :2], P[3, :2] - P[0, :2]
        A = np.column_stack([np.ones(4), P[:, 0], P[:, 1]])
        cz = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
        F.append({'P': P, 'a': a, 'b': b, 'la': float(np.linalg.norm(a)), 'lb': float(np.linalg.norm(b)),
                  'cz': cz, 'c': P[:, :2].mean(0), 'r': float(np.linalg.norm(P[:, :2] - P[:, :2].mean(0), axis=1).max()),
                  'row': f.get('row'), 'id': f.get('id')})
    if not F:
        return {'tiles': 0, 'same_row': {'pairs': 0}, 'between_rows': {'pairs': 0}, 'worst': []}
    C = np.array([f['c'] for f in F])
    R = np.array([f['r'] for f in F])
    cell = max(2 * float(R.max()), 1.0)
    grid: dict = {}
    for n, (x, y) in enumerate(C):
        grid.setdefault((int(x // cell), int(y // cell)), []).append(n)
    best: dict = {}
    for n, f in enumerate(F):
        P = f['P']
        pts = []
        for k in range(4):
            p0, p1 = P[k, :2], P[(k + 1) % 4, :2]
            L = float(np.linalg.norm(p1 - p0))
            m = max(2, int(math.ceil(L / step_m)) + 1)
            t = np.linspace(0.02 / max(L, 0.04), 1 - 0.02 / max(L, 0.04), m)
            pts.append(p0[None, :] + t[:, None] * (p1 - p0)[None, :])
        pts = np.concatenate(pts)
        za = f['cz'][0] + f['cz'][1] * pts[:, 0] + f['cz'][2] * pts[:, 1]
        gx, gy = int(f['c'][0] // cell), int(f['c'][1] // cell)
        cands = [m for dx in (-1, 0, 1) for dy in (-1, 0, 1) for m in grid.get((gx + dx, gy + dy), ()) if m != n]
        for m in cands:
            g = F[m]
            if np.hypot(*(g['c'] - f['c'])) > f['r'] + g['r'] + near_m:
                continue
            d = pts - g['P'][0, :2]
            s = d @ g['a'] / g['la']
            t = d @ g['b'] / g['lb']
            inside = (s >= -near_m) & (s <= g['la'] + near_m) & (t >= -near_m) & (t <= g['lb'] + near_m)
            border = np.minimum(np.minimum(s, g['la'] - s), np.minimum(t, g['lb'] - t)) <= edge_m
            hit = inside & border
            if not hit.any():
                continue
            zb = g['cz'][0] + g['cz'][1] * pts[hit, 0] + g['cz'][2] * pts[hit, 1]
            dz = za[hit] - zb
            j = int(np.argmax(np.abs(dz)))
            if abs(dz[j]) > max_dz_m:
                continue
            key = (min(n, m), max(n, m))
            if key not in best or abs(dz[j]) > abs(best[key][0]):
                best[key] = (float(dz[j]), pts[hit][j])
    same, cross, worst = [], [], []
    for (n, m), (dz, p) in best.items():
        (same if F[n]['row'] == F[m]['row'] else cross).append(dz)
        worst.append({'x': round(float(p[0]), 2), 'y': round(float(p[1]), 2), 'dz_m': round(abs(dz), 3),
                      'rows': [F[n]['row'], F[m]['row']], 'ids': [F[n]['id'], F[m]['id']]})
    worst.sort(key=lambda w: -w['dz_m'])
    return {'tiles': len(F), 'same_row': _stats(same), 'between_rows': _stats(cross), 'worst': worst[:top],
            'meaning': 'neighbouring ground tiles: height step where their edges meet (largest per pair); same_row '
                       'steps are seams (a continuous surface has none), between_rows may be intended kerbs'}
