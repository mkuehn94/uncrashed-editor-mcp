"""Placement math for Uncrashed map JSON.

Translations in the file are centimetres. Callers pass metres.
Yaw is radians about +Z. Yaw 0 points along +X, yaw 90° along +Y.
"""

from __future__ import annotations

import math
from typing import Iterable

Vec3 = tuple[float, float, float]


def quat_yaw(yaw: float) -> dict[str, float]:
    """Quaternion for a pure yaw. Matches Unreal's FRotator yaw about Z."""
    return {
        "X": 0.0,
        "Y": 0.0,
        "Z": round(math.sin(yaw / 2), 6),
        "W": round(math.cos(yaw / 2), 6),
    }


def quat_pitch_yaw(pitch_deg: float, yaw_deg: float) -> dict[str, float]:
    """Pitch then yaw, in degrees. Negative pitch looks down."""
    pitch = math.radians(pitch_deg) / 2
    yaw = math.radians(yaw_deg) / 2
    sp, cp, sy, cy = math.sin(pitch), math.cos(pitch), math.sin(yaw), math.cos(yaw)
    return {
        "X": round(sp * sy, 6),
        "Y": round(-sp * cy, 6),
        "Z": round(cp * sy, 6),
        "W": round(cp * cy, 6),
    }


def quat_rotator(pitch: float, yaw: float, roll: float) -> dict[str, float]:
    """Full Unreal FRotator -> unit FQuat; no precision-losing rounding."""
    p, y, r = (math.radians(v)/2 for v in (pitch, yaw, roll))
    sp, cp, sy, cy, sr, cr = math.sin(p), math.cos(p), math.sin(y), math.cos(y), math.sin(r), math.cos(r)
    return {'X': cr*sp*sy-sr*cp*cy, 'Y': -cr*sp*cy-sr*cp*sy,
            'Z': cr*cp*sy-sr*sp*cy, 'W': cr*cp*cy+sr*sp*sy}


def quat_multiply(a: dict, b: dict) -> dict[str, float]:
    x, y, z, w = (a[k] for k in ('X', 'Y', 'Z', 'W'))
    u, v, t, s = (b[k] for k in ('X', 'Y', 'Z', 'W'))
    q = (w*u+x*s+y*t-z*v, w*v-x*t+y*s+z*u, w*t+x*v-y*u+z*s, w*s-x*u-y*v-z*t)
    length = math.sqrt(sum(v*v for v in q))
    return dict(zip(('X', 'Y', 'Z', 'W'), (v/length for v in q)))


def quat_rotate(q: dict, value: Iterable[float]) -> Vec3:
    x, y, z = value
    a, b, c, w = (q[k] for k in ('X', 'Y', 'Z', 'W'))
    u, v, t = 2*(b*z-c*y), 2*(c*x-a*z), 2*(a*y-b*x)
    return x+w*u+b*t-c*v, y+w*v+c*u-a*t, z+w*t+a*v-b*u


def yaw_from_quat(q: dict) -> float:
    """Recover yaw from a pure-yaw quaternion written by quat_yaw."""
    return 2 * math.atan2(float(q["Z"]), float(q["W"]))


def l2w(cx: float, cy: float, yaw: float, u: float, v: float) -> tuple[float, float]:
    """Local offset to world. `u` runs along yaw, `v` to its left."""
    c, s = math.cos(yaw), math.sin(yaw)
    return cx + u * c - v * s, cy + u * s + v * c


def pivot_for_anchor(
    origin_cm: Iterable[float],
    extent_cm: Iterable[float],
    x: float,
    y: float,
    z: float,
    yaw: float,
    scale: Vec3,
    anchor: str,
) -> Vec3:
    """World pivot so the mesh bounds land on the requested anchor.

    bottom_center: bounding-box centre at (x, y), bottom at z.
    top_center: bounding-box centre at (x, y), top at z.
    pivot: (x, y, z) is the actor pivot, which is not the mesh centre for many rows.
    """
    if anchor == "pivot":
        return (x, y, z)
    sx, sy, sz = scale
    ox = origin_cm[0] / 100.0 * sx
    oy = origin_cm[1] / 100.0 * sy
    c, s = math.cos(yaw), math.sin(yaw)
    px = x - (ox * c - oy * s)
    py = y - (ox * s + oy * c)
    if anchor == "top_center":
        oz = (origin_cm[2] + extent_cm[2]) / 100.0 * sz
    elif anchor == "bottom_center":
        oz = (origin_cm[2] - extent_cm[2]) / 100.0 * sz
    else:
        raise ValueError(f"unknown anchor {anchor!r}")
    return (px, py, z - oz)


def world_aabb(
    origin_cm: Iterable[float],
    extent_cm: Iterable[float],
    pivot: Vec3,
    yaw: float,
    scale: Vec3,
) -> tuple[float, float, float, float, float, float]:
    """Axis-aligned bounds in metres of a yaw-rotated mesh."""
    sx, sy, szs = scale
    c, s = math.cos(yaw), math.sin(yaw)
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for dx in (-1.0, 1.0):
        for dy in (-1.0, 1.0):
            for dz in (-1.0, 1.0):
                lx = (origin_cm[0] + dx * extent_cm[0]) / 100.0 * sx
                ly = (origin_cm[1] + dy * extent_cm[1]) / 100.0 * sy
                lz = (origin_cm[2] + dz * extent_cm[2]) / 100.0 * szs
                xs.append(pivot[0] + lx * c - ly * s)
                ys.append(pivot[1] + lx * s + ly * c)
                zs.append(pivot[2] + lz)
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def yaw_for_long_axis(size_m: Iterable[float], heading: float) -> float:
    """Yaw that lays the mesh's longer horizontal axis along `heading`."""
    if size_m[1] > size_m[0]:
        return heading + math.pi / 2
    return heading


def rotator_from_quat(q: dict) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees for a map-file quaternion. Unreal's FQuat::Rotator."""
    x, y, z, w = (float(q[k]) for k in ("X", "Y", "Z", "W"))
    test = z * x - w * y
    yaw_y = 2.0 * (w * z + x * y)
    yaw_x = 1.0 - 2.0 * (y * y + z * z)
    threshold = 0.4999995
    yaw = math.degrees(math.atan2(yaw_y, yaw_x))
    if test < -threshold:
        pitch = -90.0
        roll = _wrap180(-yaw - math.degrees(2.0 * math.atan2(x, w)))
    elif test > threshold:
        pitch = 90.0
        roll = _wrap180(yaw - math.degrees(2.0 * math.atan2(x, w)))
    else:
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * test))))
        roll = math.degrees(math.atan2(-2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    return pitch, yaw, roll


def _wrap180(angle: float) -> float:
    angle = math.fmod(angle + 180.0, 360.0)
    if angle < 0:
        angle += 360.0
    return angle - 180.0
