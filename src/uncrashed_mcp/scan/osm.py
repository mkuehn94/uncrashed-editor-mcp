"""OpenStreetMap priors for a scan survey: fetch the survey bbox once from Overpass, project, rasterise.

Only the bounding box (four numbers) leaves the machine. The raw answer is cached beside the survey (osm.json).
Data © OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright).
"""
from __future__ import annotations

import json
import math
import os
import socket
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from . import grid as G
from . import native

ATTRIBUTION = '© OpenStreetMap contributors, ODbL'
OVERPASS = ('https://overpass-api.de/api/interpreter', 'https://overpass.private.coffee/api/interpreter')
MAP_API = 'https://api.openstreetmap.org/api/0.6/map'   # fallback for small boxes when Overpass is busy
MAP_API_MAX_DEG2 = 0.0025
TIMEOUT_S = 25
PROBE_S = 3.0             # quick TCP connect before each server's long request timeout
AGENT = {'User-Agent': 'uncrashed-mcp-scan/1'}
# nominal carriageway widths (metres) by highway type; `width` / `lanes` tags override
ROAD_W = {'motorway': 12, 'trunk': 10, 'primary': 9, 'secondary': 8, 'tertiary': 7, 'unclassified': 5.5,
          'residential': 5.5, 'living_street': 5, 'service': 3.5, 'road': 5, 'motorway_link': 6, 'trunk_link': 6,
          'primary_link': 6, 'secondary_link': 6, 'tertiary_link': 6, 'busway': 4}
PATH_W = {'footway': 2.0, 'path': 2.0, 'cycleway': 2.0, 'pedestrian': 4.0, 'steps': 2.0, 'track': 3.0, 'bridleway': 2.0}
QUERY = """[out:json][timeout:30];
(
  way["building"]({b}); relation["building"]({b});
  way["highway"]({b});
  way["amenity"="parking"]({b}); relation["amenity"="parking"]({b});
  way["natural"="water"]({b}); relation["natural"="water"]({b});
  way["waterway"="riverbank"]({b}); way["landuse"~"^(reservoir|basin)$"]({b});
  node["natural"="tree"]({b});
);
out body; >; out skel qt;"""


class OsmError(RuntimeError):
    pass


# ---------------------------------------------------------------------------- geodesy


def _radii(lat: float) -> tuple[float, float]:
    a, e2 = 6378137.0, 6.69437999014e-3
    w = 1 - e2 * math.sin(math.radians(lat)) ** 2
    return a * (1 - e2) / w ** 1.5, a / math.sqrt(w)


def to_scan(lat, lon, lat0: float, lon0: float):
    """East/south metres from the export origin (local tangent plane; fine within a few km)."""
    m, n = _radii(lat0)
    e = np.radians(np.asarray(lon, float) - lon0) * n * math.cos(math.radians(lat0))
    s = -np.radians(np.asarray(lat, float) - lat0) * m
    return e, s


def to_latlon(e, s, lat0: float, lon0: float):
    m, n = _radii(lat0)
    lat = lat0 + np.degrees(-np.asarray(s, float) / m)
    lon = lon0 + np.degrees(np.asarray(e, float) / (n * math.cos(math.radians(lat0))))
    return lat, lon


def bbox(x0: float, y0: float, x1: float, y1: float, lat0: float, lon0: float) -> tuple[float, float, float, float]:
    """(south, west, north, east) degrees of a scan-frame box."""
    lat, lon = to_latlon([x0, x1, x0, x1], [y0, y0, y1, y1], lat0, lon0)
    return float(lat.min()), float(lon.min()), float(lat.max()), float(lon.max())


# ---------------------------------------------------------------------------- fetch


def disabled() -> str | None:
    """Why OSM is switched off for every survey (UNCRASHED_OSM=0), or None."""
    if os.environ.get('UNCRASHED_OSM', '').strip().lower() in ('0', 'off', 'no', 'false'):
        return 'disabled by UNCRASHED_OSM=0'
    return None


def sources() -> tuple[list[str], str]:
    """(Overpass endpoints, map API URL or '') read at call time. UNCRASHED_OVERPASS (comma-separated) replaces the
    default Overpass list and, unless UNCRASHED_OSM_MAP_API names one, also drops the map-API fallback: the bbox
    goes only where the user said. UNCRASHED_OSM_MAP_API='' disables the fallback."""
    raw = os.environ.get('UNCRASHED_OVERPASS')
    overpass = [u.strip() for u in raw.split(',') if u.strip()] if raw is not None else list(OVERPASS)
    if 'UNCRASHED_OSM_MAP_API' in os.environ:
        map_api = os.environ['UNCRASHED_OSM_MAP_API'].strip()
    else:
        map_api = '' if raw is not None else MAP_API
    return overpass, map_api


def _unreachable(url: str) -> str | None:
    """A quick TCP connect to the server of url: an error text, or None when it accepts connections."""
    u = urllib.parse.urlparse(url)
    port = u.port or (443 if u.scheme == 'https' else 80)
    try:
        socket.create_connection((u.hostname, port), timeout=PROBE_S).close()
        return None
    except OSError as exc:
        return f'{u.netloc}: no connection within {PROBE_S:g} s ({" ".join(str(exc).split())[:60]})'


def fetch(path: Path, box: tuple[float, float, float, float], refresh: bool = False) -> dict:
    """OSM elements for the box (Overpass JSON shape), cached at path. Tries the Overpass endpoints, then the main
    API's map call for a small box (see sources). Each server is probed with a short connect first, so a dead
    network costs seconds, not minutes. Raises OsmError when every source fails (the message says how long it took)."""
    if path.is_file() and not refresh:
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    t0 = time.time()
    overpass, map_api = sources()
    errors = []
    b = ','.join(f'{v:.7f}' for v in box)
    answer = None
    for url in overpass:
        why = _unreachable(url)
        if why:
            errors.append(why)
            continue
        data = urllib.parse.urlencode({'data': QUERY.format(b=b)}).encode()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=AGENT), timeout=TIMEOUT_S) as r:
                answer = json.loads(r.read())
            if 'elements' in answer:
                answer['source'] = url
                break
            errors.append(f'{url}: no elements')
        except Exception as exc:  # network, HTTP, JSON: OSM is optional
            errors.append(f'{urllib.parse.urlparse(url).netloc}: {" ".join(str(exc).split())[:80]}')
        answer = None
    if answer is None and map_api and (box[2] - box[0]) * (box[3] - box[1]) <= MAP_API_MAX_DEG2:
        why = _unreachable(map_api)
        if why:
            errors.append(why)
        else:
            q = urllib.parse.urlencode({'bbox': f'{box[1]:.7f},{box[0]:.7f},{box[3]:.7f},{box[2]:.7f}'})
            try:
                with urllib.request.urlopen(urllib.request.Request(f'{map_api}?{q}', headers=AGENT), timeout=60) as r:
                    answer = from_xml(r.read())
                answer['source'] = map_api
            except Exception as exc:
                errors.append(f'{urllib.parse.urlparse(map_api).netloc}: {" ".join(str(exc).split())[:80]}')
                answer = None
    if answer is None:
        raise OsmError(f'no OSM source answered after {time.time() - t0:.1f} s (' + '; '.join(errors or ['no source configured']) + ')')
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(answer))
    tmp.replace(path)
    return answer


def from_xml(raw: bytes) -> dict:
    """OSM XML (API 0.6) -> the Overpass JSON element list."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(raw)
    out = []
    for el in root:
        tags = {t.get('k'): t.get('v') for t in el.findall('tag')}
        if el.tag == 'node':
            out.append({'type': 'node', 'id': int(el.get('id')), 'lat': float(el.get('lat')),
                        'lon': float(el.get('lon')), **({'tags': tags} if tags else {})})
        elif el.tag == 'way':
            out.append({'type': 'way', 'id': int(el.get('id')), 'nodes': [int(n.get('ref')) for n in el.findall('nd')],
                        'tags': tags})
        elif el.tag == 'relation':
            out.append({'type': 'relation', 'id': int(el.get('id')), 'tags': tags,
                        'members': [{'type': m.get('type'), 'ref': int(m.get('ref')), 'role': m.get('role', '')}
                                    for m in el.findall('member')]})
    return {'elements': out}


# ---------------------------------------------------------------------------- parse


def _rings(ways: list[list[int]]) -> list[list[int]]:
    """Join open node chains into closed rings (multipolygon members)."""
    rings, open_ = [], [list(w) for w in ways if len(w) >= 2]
    while open_:
        cur = open_.pop()
        changed = True
        while cur[0] != cur[-1] and changed:
            changed = False
            for i, w in enumerate(open_):
                if w[0] == cur[-1]:
                    cur += w[1:]
                elif w[-1] == cur[-1]:
                    cur += w[::-1][1:]
                elif w[-1] == cur[0]:
                    cur = w[:-1] + cur
                elif w[0] == cur[0]:
                    cur = w[::-1][:-1] + cur
                else:
                    continue
                open_.pop(i)
                changed = True
                break
        if cur[0] == cur[-1] and len(cur) >= 4:
            rings.append(cur)
    return rings


def parse(answer: dict, lat0: float, lon0: float) -> dict:
    """Features in scan metres: buildings, parking, water (polygons), roads, paths (lines + width), trees."""
    nodes, ways, rels = {}, {}, []
    for el in answer.get('elements', []):
        t = el.get('type')
        if t == 'node':
            nodes[el['id']] = (el['lat'], el['lon'], el.get('tags') or {})
        elif t == 'way':
            ways[el['id']] = (el.get('nodes') or [], el.get('tags') or {})
        elif t == 'relation':
            rels.append(el)
    ids = list(nodes)
    lat = np.array([nodes[i][0] for i in ids]) if ids else np.empty(0)
    lon = np.array([nodes[i][1] for i in ids]) if ids else np.empty(0)
    e, s = to_scan(lat, lon, lat0, lon0)
    xy = {i: (float(a), float(b)) for i, a, b in zip(ids, e, s)}

    def pts(chain):
        return [xy[n] for n in chain if n in xy]

    out = {'buildings': [], 'parking': [], 'water': [], 'roads': [], 'paths': [], 'trees': []}
    for wid, (chain, tags) in ways.items():
        closed = len(chain) >= 4 and chain[0] == chain[-1]
        if 'building' in tags and closed:
            out['buildings'].append({'osm': f'way/{wid}', 'rings': [pts(chain)], 'tags': _btags(tags)})
        elif tags.get('amenity') == 'parking' and closed:
            out['parking'].append({'rings': [pts(chain)]})
        elif closed and (tags.get('natural') == 'water' or tags.get('waterway') == 'riverbank'
                         or tags.get('landuse') in ('reservoir', 'basin')):
            out['water'].append({'rings': [pts(chain)]})
        elif 'highway' in tags:
            hw = tags['highway']
            width = _width(tags)
            if hw in PATH_W:
                if closed and tags.get('area') == 'yes':
                    out['parking'].append({'rings': [pts(chain)], 'path_area': True})
                else:
                    out['paths'].append({'line': pts(chain), 'width': width or PATH_W[hw]})
            elif hw in ROAD_W:
                if tags.get('area') == 'yes' and closed:
                    continue
                out['roads'].append({'line': pts(chain), 'width': width or ROAD_W[hw], 'highway': hw,
                                     'bridge': tags.get('bridge') == 'yes', 'tunnel': tags.get('tunnel') == 'yes'})
    for rel in rels:
        tags = rel.get('tags') or {}
        outers = [ways[m['ref']][0] for m in rel.get('members', [])
                  if m.get('type') == 'way' and m.get('role', 'outer') in ('outer', '') and m['ref'] in ways]
        rings = [pts(r) for r in _rings(outers)]
        if not rings:
            continue
        if 'building' in tags:
            out['buildings'].append({'osm': f"relation/{rel['id']}", 'rings': rings, 'tags': _btags(tags)})
        elif tags.get('amenity') == 'parking':
            out['parking'].append({'rings': rings})
        elif tags.get('natural') == 'water':
            out['water'].append({'rings': rings})
    for nid, (_la, _lo, tags) in nodes.items():
        if tags.get('natural') == 'tree' and nid in xy:
            out['trees'].append(xy[nid])
    return out


def _width(tags: dict) -> float | None:
    try:
        w = float(str(tags.get('width', '')).replace('m', '').strip())
        if 1 <= w <= 60:
            return w
    except ValueError:
        pass
    try:
        lanes = int(tags.get('lanes', ''))
        if 1 <= lanes <= 10:
            return 3.2 * lanes
    except ValueError:
        pass
    return None


def _btags(tags: dict) -> dict:
    keep = {'building': 'building', 'building:levels': 'levels', 'roof:shape': 'roof_shape', 'height': 'height',
            'roof:levels': 'roof_levels', 'name': 'name'}
    return {short: tags[k] for k, short in keep.items() if k in tags}


def counts(features: dict) -> dict:
    return {k: len(v) for k, v in features.items()}


# ---------------------------------------------------------------------------- rasterise


def _fill(img: np.ndarray, g, ring, value: int, dx: float, dy: float) -> None:
    if len(ring) < 3:
        return
    p = np.asarray(ring, float)
    c, r = g.to_cell(p[:, 0] + dx, p[:, 1] + dy)
    native.fill_poly(img, c, r, value)


def _lines(mask: np.ndarray, g, line, width: float, dx: float, dy: float) -> None:
    if len(line) < 2:
        return
    p = np.asarray(line, float) + [dx, dy]
    hw = width / 2
    for (ax, ay), (bx, by) in zip(p[:-1], p[1:]):
        c0, r0 = g.to_cell(min(ax, bx) - hw, min(ay, by) - hw)
        c1, r1 = g.to_cell(max(ax, bx) + hw, max(ay, by) + hw)
        c0, r0 = max(0, int(math.floor(c0))), max(0, int(math.floor(r0)))
        c1, r1 = min(g.w - 1, int(math.ceil(c1))), min(g.h - 1, int(math.ceil(r1)))
        if c0 > c1 or r0 > r1:
            continue
        cx, cy = g.to_world(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
        X, Y = np.meshgrid(cx, cy)
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy
        t = np.clip(((X - ax) * vx + (Y - ay) * vy) / L2, 0, 1) if L2 > 0 else np.zeros_like(X)
        d2 = (X - ax - t * vx) ** 2 + (Y - ay - t * vy) ** 2
        mask[r0:r1 + 1, c0:c1 + 1] |= d2 <= hw * hw


def rasterise(features: dict, g, shift=(0.0, 0.0)) -> dict:
    """Layers on the survey grid: building (int32 index+1), building_tags {index+1: tags}, road / path / parking /
    water (bool), trees_cells (N x 2 col,row)."""
    dx, dy = shift
    bld = np.zeros((g.h, g.w), np.int32)
    tags = {}
    for i, b in enumerate(features.get('buildings', []), 1):
        for ring in b['rings']:
            _fill(bld, g, ring, i, dx, dy)
        tags[i] = {'osm': b['osm'], **b.get('tags', {})}
    out = {'building': bld, 'building_tags': tags}
    for name, key in (('parking', 'parking'), ('water', 'water')):
        img = np.zeros((g.h, g.w), np.int32)
        for f in features.get(key, []):
            for ring in f['rings']:
                _fill(img, g, ring, 1, dx, dy)
        out[name] = img > 0
    for name, key in (('road', 'roads'), ('path', 'paths')):
        m = np.zeros((g.h, g.w), bool)
        for f in features.get(key, []):
            if f.get('tunnel'):
                continue
            _lines(m, g, f['line'], f['width'], dx, dy)
        out[name] = m
    trees = np.asarray(features.get('trees', []), float).reshape(-1, 2) + [dx, dy]
    if len(trees):
        c, r = g.to_cell(trees[:, 0], trees[:, 1])
        out['trees_cells'] = np.stack([c, r], 1)
    else:
        out['trees_cells'] = np.empty((0, 2))
    return out


# ---------------------------------------------------------------------------- kerbs: carriageway vs sidewalk

KERB_STEP_M = (0.06, 0.35)   # a kerb: the ground rises this much within KERB_RUN_M
KERB_RUN_M = 0.75
KERB_REACH_M = 8.0           # look for a kerb up to this far beyond the nominal half width
KERB_SHARE = 0.4             # share of profiles of a road that must find a kerb on a side
SIDEWALK_MAX_M = 4.5         # sidewalk band beyond a kerb, at most this wide (stops at buildings / raised cells)
PROFILE_STEP_M = 2.0


def _profile_side(dsm, ndsm, valid, g, p, n, hw: float, res: float):
    """(kerb distance or None, sidewalk end or None) on one side of a road at point p, normal n."""
    d = np.arange(res, hw + KERB_REACH_M + SIDEWALK_MAX_M, res)
    x, y = p[0] + d * n[0], p[1] + d * n[1]
    c, r = g.to_cell(x, y)
    c, r = np.rint(c).astype(int), np.rint(r).astype(int)
    inside = (c >= 0) & (c < g.w) & (r >= 0) & (r < g.h)
    c, r = np.clip(c, 0, g.w - 1), np.clip(r, 0, g.h - 1)
    z = np.where(inside & valid[r, c] & (ndsm[r, c] < 0.4), dsm[r, c], np.nan)
    wall = inside & valid[r, c] & (ndsm[r, c] > 2.2)
    stop = int(np.argmax(wall)) if wall.any() else len(d)
    k = max(1, int(round(KERB_RUN_M / res)))
    lim = min(stop, int((hw + KERB_REACH_M) / res))
    near = z[:max(2, int(min(hw, 2.5) / res))]
    if np.isfinite(near).sum() < 3:
        return None, None
    z0 = float(np.nanmedian(near))
    with np.errstate(all='ignore'):
        for i in range(max(k, int(0.75 * hw / res)), lim - 2 * k):   # kerbs widen the band, never halve it
            a, b, far = z[i - k:i], z[i:i + k], z[i:i + 2 * k]
            if np.isfinite(a).sum() < k // 2 + 1 or np.isfinite(b).sum() < k // 2 + 1:
                continue
            step = float(np.nanmedian(b) - np.nanmedian(a))
            if KERB_STEP_M[0] <= step <= KERB_STEP_M[1] and float(np.nanmedian(far)) - z0 >= KERB_STEP_M[0]:
                kerb = float(d[i])
                end = min(float(d[min(stop, len(d)) - 1]) if stop < len(d) else kerb + SIDEWALK_MAX_M,
                          kerb + SIDEWALK_MAX_M)
                return kerb, (end if end > kerb + 0.5 else None)
    return None, None


def _band(mask: np.ndarray, g, line, left: float, right: float, dx: float, dy: float, inner: float = 0.0) -> None:
    """Cells between inner and left (left of the line direction) / inner and right metres off a polyline."""
    if len(line) < 2:
        return
    p = np.asarray(line, float) + [dx, dy]
    reach = max(left, right)
    for (ax, ay), (bx, by) in zip(p[:-1], p[1:]):
        c0, r0 = g.to_cell(min(ax, bx) - reach, min(ay, by) - reach)
        c1, r1 = g.to_cell(max(ax, bx) + reach, max(ay, by) + reach)
        c0, r0 = max(0, int(math.floor(c0))), max(0, int(math.floor(r0)))
        c1, r1 = min(g.w - 1, int(math.ceil(c1))), min(g.h - 1, int(math.ceil(r1)))
        if c0 > c1 or r0 > r1:
            continue
        cx, cy = g.to_world(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
        X, Y = np.meshgrid(cx, cy)
        vx, vy = bx - ax, by - ay
        L = math.hypot(vx, vy)
        if L < 1e-6:
            continue
        t = ((X - ax) * vx + (Y - ay) * vy) / (L * L)
        sd = ((Y - ay) * vx - (X - ax) * vy) / L        # > 0: towards (-vy, vx), 'right' in kerb_bands
        on = (t >= 0) & (t <= 1)
        mask[r0:r1 + 1, c0:c1 + 1] |= on & (((sd >= inner) & (sd <= right)) | ((sd <= -inner) & (sd >= -left)))


def kerb_bands(features: dict, g, dsm: np.ndarray, ndsm: np.ndarray, valid: np.ndarray,
               shift=(0.0, 0.0)) -> tuple[np.ndarray, np.ndarray, dict]:
    """Carriageway and sidewalk layers from the mapped road lines and the ground heights: along each road, profiles
    every 2 m look on both sides for a kerb (the ground steps up 6..35 cm within 0.75 m, cars and crowns skipped);
    a side where most profiles agree takes the median kerb distance as the carriageway edge (parking lanes
    included), and the band beyond it up to the next building or 4.5 m is sidewalk. Sides without a kerb keep the
    tagged / nominal half width. Returns (road, sidewalk, report)."""
    dx, dy = shift
    res = g.res
    road = np.zeros((g.h, g.w), bool)
    walk = np.zeros((g.h, g.w), bool)
    rep = {'roads': 0, 'sides_with_kerb': 0, 'sides': 0, 'widths_m': []}
    for f in features.get('roads', []):
        if f.get('tunnel') or f.get('bridge') or len(f['line']) < 2:
            continue
        rep['roads'] += 1
        hw = f['width'] / 2
        line = np.asarray(f['line'], float) + [dx, dy]
        found = {1: [], -1: []}
        ends = {1: [], -1: []}
        n_prof = 0
        for (ax, ay), (bx, by) in zip(line[:-1], line[1:]):
            L = math.hypot(bx - ax, by - ay)
            if L < 1e-6:
                continue
            u = np.array([(bx - ax) / L, (by - ay) / L])
            nr = np.array([-u[1], u[0]])            # right of the direction in east/south axes is (-uy, ux)
            for t in np.arange(min(1.0, L / 2), L, PROFILE_STEP_M):
                pt = np.array([ax, ay]) + t * u
                n_prof += 1
                for side in (1, -1):
                    kerb, end = _profile_side(dsm, ndsm, valid, g, pt, side * nr, hw, res)
                    if kerb is not None:
                        found[side].append(kerb)
                        if end is not None:
                            ends[side].append(end)
        widths = {}
        for side in (1, -1):
            rep['sides'] += 1
            if n_prof and len(found[side]) >= max(2, KERB_SHARE * n_prof):
                widths[side] = float(np.median(found[side]))
                rep['sides_with_kerb'] += 1
            else:
                widths[side] = hw
        f['kerb_m'] = [round(widths[-1], 2), round(widths[1], 2)]   # left, right of the mapped direction
        rep['widths_m'].append(round(widths[1] + widths[-1], 1))
        _band(road, g, f['line'], widths[-1], widths[1], dx, dy)
        for side, key in ((1, 'right'), (-1, 'left')):
            if widths[side] != hw and ends[side]:
                end = float(np.median(ends[side]))
                kw = {'left': 0.0, 'right': 0.0}
                kw[key] = end
                _band(walk, g, f['line'], kw['left'], kw['right'], dx, dy, inner=widths[side])
    rep['median_width_m'] = float(np.median(rep['widths_m'])) if rep['widths_m'] else None
    rep.pop('widths_m')
    return road, walk & ~road, rep


def align(features: dict, g, roofs: np.ndarray, search_m: float = 4.0) -> tuple[tuple[float, float], float, float]:
    """Shift (dx, dy metres) of the mapped building footprints that best overlaps roof-like cells, on a 0.5 m search
    grid within +-search_m. Returns (shift, overlap at zero shift, overlap at the shift) as shares of footprint
    cells. A shift is only adopted when it improves the overlap clearly."""
    if not features.get('buildings'):
        return (0.0, 0.0), 0.0, 0.0
    base = rasterise({'buildings': features['buildings']}, g)['building'] > 0
    total = max(1, int(base.sum()))
    step = max(1, int(round(0.5 / g.res)))
    reach = int(round(search_m / g.res))
    best = (0, 0, int((base & roofs).sum()))
    zero = best[2]
    # coarse search on a decimated grid, then the chosen shift at full resolution
    sub = max(1, int(round(1.0 / g.res)))
    b_s, r_s = base[::sub, ::sub], roofs[::sub, ::sub]
    scores = {}
    for sy in range(-reach, reach + 1, step):
        for sx in range(-reach, reach + 1, step):
            ys, xs = sy // sub, sx // sub
            if (ys, xs) in scores:
                continue
            scores[(ys, xs)] = int((np.roll(b_s, (ys, xs), (0, 1)) & r_s).sum())
    (ys, xs), _ = max(scores.items(), key=lambda kv: kv[1])
    sy, sx = ys * sub, xs * sub
    val = int((np.roll(base, (sy, sx), (0, 1)) & roofs).sum())
    if val > best[2] * 1.03 + 0.01 * total:
        best = (sx, sy, val)
    return (best[0] * g.res, best[1] * g.res), zero / total, best[2] / total
