"""Triangle hulls used by the fit check.

`meshes.bin` stores a simplified LOD0 for each static mesh, in centimetres.
Overlap ignores the empty space inside a bounding box: a scaffold bay, the
gap under an eave, the side of a wall. Faces that only touch (about 1.5cm)
are contact, not a hit. Faces that cross, or that lie on top of each other
with the same normal, are a hit.
"""

from __future__ import annotations

import math
import struct
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "meshes.bin"

# Contact closer than this is allowed. A real crossing is deeper.
# The hull is simplified on a 4–8cm grid, so a flush stack can stair-step by one cell.
SLOP_M = 0.05
# Simplified faces stair-step by about a cell. Two faces on the same plane
# have to overlap by more than that before they count as the same surface.
PLANE_SLOP_M = 0.2


class _Raw:
    __slots__ = ("verts", "tris", "shape")

    def __init__(self, verts, tris):
        self.verts = verts
        self.tris = tris
        self.shape = None

    def build(self) -> "Shape":
        if self.shape is None:
            self.shape = Shape(self.verts, self.tris)
        return self.shape


class Shape:
    __slots__ = ("verts", "tris", "normals", "aabbs", "bvh")

    def __init__(self, verts, tris):
        self.verts = verts
        self.tris = tris
        self.normals = []
        self.aabbs = []
        for a, b, c in tris:
            va, vb, vc = verts[a], verts[b], verts[c]
            ab = (vb[0] - va[0], vb[1] - va[1], vb[2] - va[2])
            ac = (vc[0] - va[0], vc[1] - va[1], vc[2] - va[2])
            nx = ab[1] * ac[2] - ab[2] * ac[1]
            ny = ab[2] * ac[0] - ab[0] * ac[2]
            nz = ab[0] * ac[1] - ab[1] * ac[0]
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            if length < 1e-8:
                self.normals.append(None)
            else:
                self.normals.append((nx / length, ny / length, nz / length))
            xs = (va[0], vb[0], vc[0])
            ys = (va[1], vb[1], vc[1])
            zs = (va[2], vb[2], vc[2])
            self.aabbs.append((min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)))
        self.bvh = _build_bvh(self.aabbs)


def _build_bvh(aabbs):
    indices = list(range(len(aabbs)))

    def build(ids):
        box = _union(aabbs, ids)
        if len(ids) <= 2:
            return (box, None, None, tuple(ids))
        axis = max(range(3), key=lambda i: box[i + 3] - box[i])
        ids.sort(key=lambda i: aabbs[i][axis] + aabbs[i][axis + 3])
        mid = len(ids) // 2
        return (box, build(ids[:mid]), build(ids[mid:]), None)

    if not indices:
        return None
    return build(indices)


def _union(aabbs, ids):
    mn = [1e9, 1e9, 1e9]
    mx = [-1e9, -1e9, -1e9]
    for i in ids:
        box = aabbs[i]
        for axis in range(3):
            if box[axis] < mn[axis]:
                mn[axis] = box[axis]
            if box[axis + 3] > mx[axis]:
                mx[axis] = box[axis + 3]
    return (mn[0], mn[1], mn[2], mx[0], mx[1], mx[2])


def _box_hit(a, b, pad: float) -> bool:
    return (
        a[0] <= b[3] + pad
        and b[0] <= a[3] + pad
        and a[1] <= b[4] + pad
        and b[1] <= a[4] + pad
        and a[2] <= b[5] + pad
        and b[2] <= a[5] + pad
    )


@lru_cache(maxsize=1)
def load() -> dict[str, _Raw]:
    if not DATA.is_file():
        return {}
    blob = DATA.read_bytes()
    if blob[:4] != b"UMSH":
        raise RuntimeError("meshes.bin is not a hull file")
    version, count = struct.unpack_from("<II", blob, 4)
    if version != 1:
        raise RuntimeError(f"meshes.bin version {version} is not supported")
    pos = 12
    shapes: dict[str, Shape] = {}
    for _ in range(count):
        nlen, = struct.unpack_from("<H", blob, pos)
        pos += 2
        name = blob[pos : pos + nlen].decode("utf-8")
        pos += nlen
        nv, ni = struct.unpack_from("<II", blob, pos)
        pos += 8
        raw = struct.unpack_from("<" + str(nv * 3) + "h", blob, pos)
        pos += nv * 6
        indices = struct.unpack_from("<" + str(ni) + "H", blob, pos)
        pos += ni * 2
        verts = tuple((raw[i] / 100.0, raw[i + 1] / 100.0, raw[i + 2] / 100.0) for i in range(0, len(raw), 3))
        tris = tuple((indices[i], indices[i + 1], indices[i + 2]) for i in range(0, len(indices), 3))
        shapes[name] = _Raw(verts, tris)
    return shapes


def shape_for(asset) -> Shape | None:
    if asset is None or asset.kind != "mesh" or asset.bounds_estimated or not asset.source:
        return None
    raw = load().get(asset.source)
    return None if raw is None else raw.build()


def _into(point, pivot, yaw: float, scale, target_pivot, target_yaw: float, target_scale):
    """Map a vertex from one mesh's local space into another's."""
    sx, sy, sz = scale
    c, s = math.cos(yaw), math.sin(yaw)
    lx, ly, lz = point[0] * sx, point[1] * sy, point[2] * sz
    wx = pivot[0] + lx * c - ly * s
    wy = pivot[1] + lx * s + ly * c
    wz = pivot[2] + lz
    dx = wx - target_pivot[0]
    dy = wy - target_pivot[1]
    dz = wz - target_pivot[2]
    tc, ts = math.cos(target_yaw), math.sin(target_yaw)
    tx, ty, tz = target_scale
    return ((dx * tc + dy * ts) / tx, (-dx * ts + dy * tc) / ty, dz / tz)


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b, t: float):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def _distances(normal, origin, pts):
    return tuple(_dot(normal, _sub(p, origin)) for p in pts)


def _in_tri(point, tri, normal) -> bool:
    a, b, c = tri
    area = _dot(_cross(_sub(b, a), _sub(c, a)), normal)
    if abs(area) < 1e-12:
        return False
    w0 = _dot(_cross(_sub(b, a), _sub(point, a)), normal) / area
    w1 = _dot(_cross(_sub(c, b), _sub(point, b)), normal) / area
    w2 = _dot(_cross(_sub(a, c), _sub(point, c)), normal) / area
    return w0 >= -0.05 and w1 >= -0.05 and w2 >= -0.05


def _overlap_2d(tri_a, tri_b, normal, slop: float) -> bool:
    drop = max(range(3), key=lambda i: abs(normal[i]))
    keep = [i for i in range(3) if i != drop]

    def proj(tri):
        return [(p[keep[0]], p[keep[1]]) for p in tri]

    pa, pb = proj(tri_a), proj(tri_b)
    for poly in (pa, pb):
        for i in range(3):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % 3]
            axis = (y0 - y1, x1 - x0)
            length = math.hypot(axis[0], axis[1])
            if length < 1e-12:
                continue
            axis = (axis[0] / length, axis[1] / length)
            avals = [axis[0] * p[0] + axis[1] * p[1] for p in pa]
            bvals = [axis[0] * p[0] + axis[1] * p[1] for p in pb]
            # Sharing an edge is contact. Stair-stepped faces need a wider margin.
            margin = max(slop, PLANE_SLOP_M)
            if max(avals) <= min(bvals) + margin or max(bvals) <= min(avals) + margin:
                return False
    return True


def _edge_crosses(p0, p1, d0, d1, tri, normal, slop: float) -> bool:
    if min(d0, d1) >= -slop or max(d0, d1) <= slop:
        return False
    denom = d0 - d1
    if abs(denom) < 1e-12:
        return False
    t = d0 / denom
    if t < 0.0 or t > 1.0:
        return False
    return _in_tri(_add(p0, p1, t), tri, normal)


def _pair_hit(tri_a, normal_a, tri_b, normal_b, slop: float) -> bool:
    if normal_a is None or normal_b is None:
        return False
    da = _distances(normal_a, tri_a[0], tri_b)
    db = _distances(normal_b, tri_b[0], tri_a)
    near_a = all(abs(d) <= slop for d in da)
    near_b = all(abs(d) <= slop for d in db)
    if near_a and near_b:
        # Lying on the same plane. Opposite normals are two faces touching.
        # The same normal means the surfaces occupy one place.
        if _dot(normal_a, normal_b) < 0.5:
            return False
        return _overlap_2d(tri_a, tri_b, normal_a, slop)
    if min(da) >= -slop or max(da) <= slop or min(db) >= -slop or max(db) <= slop:
        return False
    for i in range(3):
        j = (i + 1) % 3
        if _edge_crosses(tri_b[i], tri_b[j], da[i], da[j], tri_a, normal_a, slop):
            return True
        if _edge_crosses(tri_a[i], tri_a[j], db[i], db[j], tri_b, normal_b, slop):
            return True
    return False


def _query(node, box, tri, normal, shape: Shape, slop: float) -> bool:
    if node is None or not _box_hit(node[0], box, slop):
        return False
    leaves = node[3]
    if leaves is not None:
        for index in leaves:
            other_n = shape.normals[index]
            other = tuple(shape.verts[v] for v in shape.tris[index])
            if _pair_hit(other, other_n, tri, normal, slop):
                return True
        return False
    return _query(node[1], box, tri, normal, shape, slop) or _query(node[2], box, tri, normal, shape, slop)


def _tri_box(tri, pad: float):
    xs = [p[0] for p in tri]
    ys = [p[1] for p in tri]
    zs = [p[2] for p in tri]
    return (min(xs) - pad, min(ys) - pad, min(zs) - pad, max(xs) + pad, max(ys) + pad, max(zs) + pad)


def meshes_overlap(a: Shape, pivot_a, yaw_a: float, scale_a, b: Shape, pivot_b, yaw_b: float, scale_b, slop: float = SLOP_M) -> bool:
    """True when the two hulls cross, or lie on the same surface.

    Yaw is radians. Vertices of `b` are moved into `a`'s local space so `a`
    can keep the bounds tree it built at load.
    """
    if a.bvh is None or not b.tris:
        return False
    moved = [_into(v, pivot_b, yaw_b, scale_b, pivot_a, yaw_a, scale_a) for v in b.verts]
    for index, (i, j, k) in enumerate(b.tris):
        tri = (moved[i], moved[j], moved[k])
        ab = _sub(tri[1], tri[0])
        ac = _sub(tri[2], tri[0])
        normal = _cross(ab, ac)
        length = math.sqrt(_dot(normal, normal))
        if length < 1e-10:
            continue
        normal = (normal[0] / length, normal[1] / length, normal[2] / length)
        if _query(a.bvh, _tri_box(tri, slop), tri, normal, a, slop):
            return True
    return False
