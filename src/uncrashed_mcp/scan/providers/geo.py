"""Geodesy for the data providers: UTM (any zone, Krueger series), Web Mercator tiles, scan frame <-> lat/lon.

The scan frame is the export's local tangent plane (east, south metres from the metadata.json origin), the same
projection osm.py uses. UTM here is exact to well below a millimetre inside a zone (6th-order Krueger series).
"""
from __future__ import annotations

import math

import numpy as np

from .. import osm as _osm

A = 6378137.0
F = 1 / 298.257223563          # WGS84; GRS80 (ETRS89) differs by 0.1 mm in the semi-minor axis
K0 = 0.9996

_n = F / (2 - F)
_A = A / (1 + _n) * (1 + _n ** 2 / 4 + _n ** 4 / 64 + _n ** 6 / 256)
_ALPHA = (_n / 2 - 2 * _n ** 2 / 3 + 5 * _n ** 3 / 16 + 41 * _n ** 4 / 180 - 127 * _n ** 5 / 288 + 7891 * _n ** 6 / 37800,
          13 * _n ** 2 / 48 - 3 * _n ** 3 / 5 + 557 * _n ** 4 / 1440 + 281 * _n ** 5 / 630 - 1983433 * _n ** 6 / 1935360,
          61 * _n ** 3 / 240 - 103 * _n ** 4 / 140 + 15061 * _n ** 5 / 26880 + 167603 * _n ** 6 / 181440,
          49561 * _n ** 4 / 161280 - 179 * _n ** 5 / 168 + 6601661 * _n ** 6 / 7257600,
          34729 * _n ** 5 / 80640 - 3418889 * _n ** 6 / 1995840,
          212378941 * _n ** 6 / 319334400)
_BETA = (_n / 2 - 2 * _n ** 2 / 3 + 37 * _n ** 3 / 96 - _n ** 4 / 360 - 81 * _n ** 5 / 512 + 96199 * _n ** 6 / 604800,
         _n ** 2 / 48 + _n ** 3 / 15 - 437 * _n ** 4 / 1440 + 46 * _n ** 5 / 105 - 1118711 * _n ** 6 / 3870720,
         17 * _n ** 3 / 480 - 37 * _n ** 4 / 840 - 209 * _n ** 5 / 4480 + 5569 * _n ** 6 / 90720,
         4397 * _n ** 4 / 161280 - 11 * _n ** 5 / 504 - 830251 * _n ** 6 / 7257600,
         4583 * _n ** 5 / 161280 - 108847 * _n ** 6 / 3991680,
         20648693 * _n ** 6 / 638668800)
_E = math.sqrt(F * (2 - F))


def utm_zone(lon: float) -> int:
    return int((lon + 180) // 6) % 60 + 1


def utm_forward(lat, lon, zone: int):
    """(easting, northing) metres, northern hemisphere convention (no false northing)."""
    lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    lam = lon - math.radians(zone * 6 - 183)
    t = np.sinh(np.arctanh(np.sin(lat)) - _E * np.arctanh(_E * np.sin(lat)))
    xi_, eta_ = np.arctan2(t, np.cos(lam)), np.arctanh(np.sin(lam) / np.sqrt(1 + t * t))
    xi, eta = xi_.copy(), eta_.copy()
    for j, a in enumerate(_ALPHA, 1):
        xi = xi + a * np.sin(2 * j * xi_) * np.cosh(2 * j * eta_)
        eta = eta + a * np.cos(2 * j * xi_) * np.sinh(2 * j * eta_)
    return 500000 + K0 * _A * eta, K0 * _A * xi


def utm_inverse(e, n, zone: int):
    """(lat, lon) degrees of UTM metres (northern hemisphere)."""
    xi = np.asarray(n, float) / (K0 * _A)
    eta = (np.asarray(e, float) - 500000) / (K0 * _A)
    xi_, eta_ = xi.copy(), eta.copy()
    for j, b in enumerate(_BETA, 1):
        xi_ = xi_ - b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        eta_ = eta_ - b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    chi = np.arcsin(np.sin(xi_) / np.cosh(eta_))
    lam = np.arctan2(np.sinh(eta_), np.cos(xi_))
    lat = chi.copy()
    for _ in range(6):   # conformal -> geodetic latitude
        s = np.sin(lat)
        lat = np.arcsin(np.tanh(np.arctanh(np.sin(chi)) + _E * np.arctanh(_E * s)))
    return np.degrees(lat), np.degrees(lam) + zone * 6 - 183


def to_scan(lat, lon, lat0: float, lon0: float):
    return _osm.to_scan(lat, lon, lat0, lon0)


def to_latlon(e, s, lat0: float, lon0: float):
    return _osm.to_latlon(e, s, lat0, lon0)


def tile_xy(lat: float, lon: float, z: int) -> tuple[int, int]:
    """Web Mercator (XYZ) tile holding the point."""
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    la = math.radians(max(-85.05112878, min(85.05112878, lat)))
    y = int((1 - math.asinh(math.tan(la)) / math.pi) / 2 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def tile_lonlat(x, y, z: int):
    """(lon, lat) degrees of tile coordinates (fractional allowed: x + px / extent)."""
    n = 2 ** z
    lon = np.asarray(x, float) / n * 360 - 180
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * np.asarray(y, float) / n))))
    return lon, lat


def bbox_tiles(bbox: tuple[float, float, float, float], z: int) -> list[tuple[int, int]]:
    """XYZ tiles at zoom z covering bbox (south, west, north, east)."""
    s, w, n, e = bbox
    x0, y0 = tile_xy(n, w, z)
    x1, y1 = tile_xy(s, e, z)
    return [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


def intersects(bbox, other) -> bool:
    """Both (south, west, north, east)."""
    return not (bbox[2] < other[0] or bbox[0] > other[2] or bbox[3] < other[1] or bbox[1] > other[3])


def inside_polygon(lat: float, lon: float, ring: list[tuple[float, float]]) -> bool:
    """Even-odd test; ring of (lat, lon)."""
    inside = False
    for (a_la, a_lo), (b_la, b_lo) in zip(ring, ring[1:] + ring[:1]):
        if (a_la > lat) != (b_la > lat) and lon < a_lo + (lat - a_la) * (b_lo - a_lo) / (b_la - a_la):
            inside = not inside
    return inside
