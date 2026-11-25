"""Small numpy image operations for the scan survey (no scipy / OpenCV in the project venv)."""
from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import native


def box_blur(a: np.ndarray, k: int) -> np.ndarray:
    """Mean over a k x k window (k odd), edges replicated."""
    if k <= 1:
        return a.astype(np.float32)
    r = k // 2
    p = np.pad(a.astype(np.float64), r, mode='edge')
    c = np.cumsum(np.cumsum(p, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    h, w = a.shape
    s = c[k:k + h, k:k + w] - c[0:h, k:k + w] - c[k:k + h, 0:w] + c[0:h, 0:w]
    return (s / (k * k)).astype(np.float32)


def _kernel(sigma: float) -> np.ndarray:
    r = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _conv1(a: np.ndarray, k: np.ndarray, axis: int, mode: str) -> np.ndarray:
    r = len(k) // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode=mode)
    out = np.zeros(a.shape, np.float64)
    n = a.shape[axis]
    for i, w in enumerate(k):
        out += w * (p[i:i + n] if axis == 0 else p[:, i:i + n])
    return out


def gauss(a: np.ndarray, sigma: float, mode: str = 'reflect') -> np.ndarray:
    """Separable Gaussian filter (float64 result)."""
    if sigma <= 0:
        return a.astype(np.float64)
    k = _kernel(sigma)
    return _conv1(_conv1(a.astype(np.float64), k, 0, mode), k, 1, mode)


def _filt(a: np.ndarray, k: int, fn, fill) -> np.ndarray:
    if k <= 1:
        return a.copy()
    k |= 1  # odd windows keep the shape
    r = k // 2
    p = np.pad(a, ((r, r), (0, 0)), mode='constant', constant_values=fill)
    a = fn(sliding_window_view(p, k, axis=0), axis=-1)
    p = np.pad(a, ((0, 0), (r, r)), mode='constant', constant_values=fill)
    return fn(sliding_window_view(p, k, axis=1), axis=-1)


def min_filter(a: np.ndarray, k: int) -> np.ndarray:
    fill = np.inf if a.dtype.kind == 'f' else (True if a.dtype == bool else np.iinfo(a.dtype).max)
    return _filt(a, k, np.min, fill)


def max_filter(a: np.ndarray, k: int) -> np.ndarray:
    fill = -np.inf if a.dtype.kind == 'f' else (False if a.dtype == bool else np.iinfo(a.dtype).min)
    return _filt(a, k, np.max, fill)


def grey_open(a: np.ndarray, k: int) -> np.ndarray:
    return max_filter(min_filter(a, k), k)


def dilate(m: np.ndarray, k: int = 3) -> np.ndarray:
    return max_filter(m.astype(bool), k)


def erode(m: np.ndarray, k: int = 3) -> np.ndarray:
    return min_filter(m.astype(bool), k)


def bin_open(m: np.ndarray, k: int) -> np.ndarray:
    return dilate(erode(m, k), k)


def resize_linear(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear resize with pixel-centre alignment (like OpenCV INTER_LINEAR)."""
    sh, sw = a.shape
    y = np.clip((np.arange(h) + 0.5) * sh / h - 0.5, 0, sh - 1)
    x = np.clip((np.arange(w) + 0.5) * sw / w - 0.5, 0, sw - 1)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, sh - 1)
    x1 = np.minimum(x0 + 1, sw - 1)
    fy = (y - y0)[:, None]
    fx = (x - x0)[None, :]
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bot = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


def components(lab: np.ndarray, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(rows, cols) of each label 1..n, in label order."""
    flat = lab.ravel()
    nz = np.flatnonzero(flat)
    if not n or not nz.size:
        return [(np.empty(0, int), np.empty(0, int)) for _ in range(n)]
    order = nz[np.argsort(flat[nz], kind='stable')]
    counts = np.bincount(flat[order], minlength=n + 1)[1:]
    out, start, w = [], 0, lab.shape[1]
    for c in counts:
        idx = order[start:start + c]
        out.append((idx // w, idx % w))
        start += c
    return out


def obb(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float, float]:
    """Minimum-area oriented box of points (centres of unit cells): (cx, cy, length, width, yaw_deg) with the
    length along yaw (0..180, from +x toward +y). Sizes include the cell extent."""
    pts = np.stack([xs, ys], 1).astype(np.float64)
    if len(pts) > 4000:
        pts = _extreme(pts)
    best = None
    for step, lo, hi in ((2.0, 0.0, 90.0), (0.25, -2.0, 2.0)):
        base = 0.0 if best is None else best[0]
        ang = np.arange(lo, hi, step) + base
        rad = np.radians(ang)
        d = np.stack([np.cos(rad), np.sin(rad)])          # 2 x A
        u = pts @ d                                        # N x A
        v = pts @ np.stack([-d[1], d[0]])
        ext_u = u.max(0) - u.min(0) + 1
        ext_v = v.max(0) - v.min(0) + 1
        area = ext_u * ext_v
        i = int(np.argmin(area))
        best = (float(ang[i]), ext_u[i], ext_v[i])
    a, eu, ev = best
    rad = math.radians(a)
    du, dv = np.array([math.cos(rad), math.sin(rad)]), np.array([-math.sin(rad), math.cos(rad)])
    u, v = pts @ du, pts @ dv
    cu, cv = (u.max() + u.min()) / 2, (v.max() + v.min()) / 2
    c = cu * du + cv * dv
    if ev > eu:
        eu, ev, a = ev, eu, a + 90
    return float(c[0]), float(c[1]), float(eu), float(ev), float(a % 180)


def _extreme(pts: np.ndarray) -> np.ndarray:
    """Boundary-ish subset of many cell centres: per row the min and max column, per column the min and max row
    (exact for the hull)."""
    out = []
    for a, b in ((0, 1), (1, 0)):
        key = pts[:, a]
        order = np.lexsort((pts[:, b], key))
        k = key[order]
        first = np.r_[True, k[1:] != k[:-1]]
        last = np.r_[k[1:] != k[:-1], True]
        out += [pts[order][first], pts[order][last]]
    return np.unique(np.concatenate(out), axis=0)


def box_corners(cx: float, cy: float, length: float, width: float, yaw_deg: float) -> np.ndarray:
    a = math.radians(yaw_deg)
    du, dv = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    c = np.array([cx, cy])
    return np.array([c + su * length / 2 * du + sv * width / 2 * dv for su, sv in ((1, 1), (-1, 1), (-1, -1), (1, -1))])


def simplify(points: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker on a closed ring (no repeated end point)."""
    n = len(points)
    if n <= 4:
        return points
    # split the ring at the two farthest points
    d = np.linalg.norm(points - points[0], axis=1)
    j = int(np.argmax(d))
    a = _dp(points[:j + 1], tol)
    b = _dp(np.vstack([points[j:], points[:1]]), tol)
    return np.vstack([a[:-1], b[:-1]])


def _dp(p: np.ndarray, tol: float) -> np.ndarray:
    if len(p) <= 2:
        return p
    keep = np.zeros(len(p), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(p) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = p[i], p[j]
        ab = b - a
        L = math.hypot(*ab)
        seg = p[i + 1:j] - a
        if L < 1e-9:
            dist = np.linalg.norm(seg, axis=1)
        else:
            dist = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / L
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return p[keep]


def ring_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def outline(lab: np.ndarray, value: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Outer boundary (col,row cell centres) of one labelled component given its cells."""
    flat = rows * lab.shape[1] + cols
    k = int(np.argmin(flat))
    return native.contour(lab, value, int(rows[k]), int(cols[k])).astype(np.float64)
