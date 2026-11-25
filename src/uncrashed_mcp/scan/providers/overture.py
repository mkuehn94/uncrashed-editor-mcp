"""Overture Maps buildings (global; footprints with height / floors / roof shape where known), read from the
release's PMTiles archive (tiles.overturemaps.org) by HTTP range requests: header, directories and the zoom-14 tiles
of the bbox. Only the byte ranges of those tiles leave the machine. Licence of the buildings theme: ODbL 1.0
(it contains OpenStreetMap data); attribution (c) OpenStreetMap contributors, Overture Maps Foundation.

The newest release is found in the Overture STAC catalogue. Geometry is the vector tile's (4096 units per tile:
about 0.1 m at zoom 14 in mid latitudes).
"""
from __future__ import annotations

import gzip
import json
import functools
import struct
import zlib
from pathlib import Path

import numpy as np

from . import Provider, cached, geo, net

STAC = 'https://stac.overturemaps.org/catalog.json'
TILES = 'https://tiles.overturemaps.org/{release}/buildings.pmtiles'
ZOOM = 14


# ---------------------------------------------------------------------------- PMTiles v3


def _damaged(fn):
    """A truncated or garbled archive / tile raises ValueError, never struct.error / IndexError / zlib.error."""
    @functools.wraps(fn)
    def run(*a, **k):
        try:
            return fn(*a, **k)
        except (struct.error, zlib.error, IndexError, KeyError, EOFError, OverflowError) as exc:
            raise ValueError(f'damaged Overture data in {fn.__name__} ({type(exc).__name__}: {exc})') from None
    return run


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    out = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        out |= (b & 0x7F) << shift
        if b < 0x80:
            return out, pos
        shift += 7


def zxy_to_tileid(z: int, x: int, y: int) -> int:
    acc = ((1 << (2 * z)) - 1) // 3
    for a in range(z - 1, -1, -1):
        s = 1 << a
        rx, ry = bool(x & s), bool(y & s)
        acc += s * s * ((3 * rx) ^ ry)
        if not ry:
            if rx:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
    return acc


@_damaged
def header(raw: bytes) -> dict:
    if raw[:7] != b'PMTiles' or raw[7] != 3:
        raise ValueError('not a PMTiles v3 archive')
    v = struct.unpack('<11Q', raw[8:96])
    keys = ('root_off', 'root_len', 'meta_off', 'meta_len', 'leaf_off', 'leaf_len', 'data_off', 'data_len')
    h = dict(zip(keys, v[:8]))
    h.update(internal=raw[97], tile_comp=raw[98], tile_type=raw[99], min_zoom=raw[100], max_zoom=raw[101])
    return h


@_damaged
def _inflate(raw: bytes, comp: int) -> bytes:
    if comp in (0, 1):
        return raw
    if comp == 2:
        return gzip.decompress(raw)
    raise ValueError(f'PMTiles compression {comp} is not supported')


@_damaged
def directory(raw: bytes) -> list[tuple[int, int, int, int]]:
    """[(tile_id, offset, length, run_length)]."""
    n, pos = _varint(raw, 0)
    ids, last = [], 0
    for _ in range(n):
        d, pos = _varint(raw, pos)
        last += d
        ids.append(last)
    runs = []
    for _ in range(n):
        v, pos = _varint(raw, pos)
        runs.append(v)
    lens = []
    for _ in range(n):
        v, pos = _varint(raw, pos)
        lens.append(v)
    offs = []
    for i in range(n):
        v, pos = _varint(raw, pos)
        offs.append(offs[-1] + lens[i - 1] if v == 0 and i > 0 else v - 1)
    return list(zip(ids, offs, lens, runs))


def find(entries, tid: int):
    lo, hi, best = 0, len(entries) - 1, None
    while lo <= hi:
        m = (lo + hi) // 2
        if entries[m][0] <= tid:
            best, lo = entries[m], m + 1
        else:
            hi = m - 1
    if best is None:
        return None
    t, off, ln, run = best
    if run == 0:
        return ('leaf', off, ln)
    if tid < t + run:
        return ('tile', off, ln)
    return None


class Archive:
    def __init__(self, url: str):
        self.url = url
        self.h = header(net.get_range(url, 0, 127))
        self.root = directory(_inflate(net.get_range(url, self.h['root_off'], self.h['root_len']), self.h['internal']))

    def tile(self, z: int, x: int, y: int) -> bytes | None:
        tid = zxy_to_tileid(z, x, y)
        entries = self.root
        for _ in range(4):
            hit = find(entries, tid)
            if hit is None:
                return None
            kind, off, ln = hit
            if kind == 'tile':
                return _inflate(net.get_range(self.url, self.h['data_off'] + off, ln), self.h['tile_comp'])
            entries = directory(_inflate(net.get_range(self.url, self.h['leaf_off'] + off, ln), self.h['internal']))
        return None


# ---------------------------------------------------------------------------- Mapbox vector tiles


def _fields(buf: bytes):
    pos, n = 0, len(buf)
    while pos < n:
        key, pos = _varint(buf, pos)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, pos = _varint(buf, pos)
        elif wt == 1:
            v, pos = buf[pos:pos + 8], pos + 8
        elif wt == 2:
            ln, pos = _varint(buf, pos)
            v, pos = buf[pos:pos + ln], pos + ln
        elif wt == 5:
            v, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError(f'protobuf wire type {wt}')
        yield f, wt, v


def _packed(buf: bytes) -> list[int]:
    out, pos = [], 0
    while pos < len(buf):
        v, pos = _varint(buf, pos)
        out.append(v)
    return out


def _value(buf: bytes):
    for f, _wt, v in _fields(buf):
        if f == 1:
            return v.decode('utf-8', 'replace')
        if f == 2:
            return struct.unpack('<f', v)[0]
        if f == 3:
            return struct.unpack('<d', v)[0]
        if f in (4, 5):
            return v
        if f == 6:
            return (v >> 1) ^ -(v & 1)
        if f == 7:
            return bool(v)
    return None


def _rings(cmds: list[int]) -> list[list[tuple[int, int]]]:
    rings, cur, x, y, i = [], [], 0, 0, 0
    while i < len(cmds):
        c = cmds[i]
        op, cnt = c & 7, c >> 3
        i += 1
        if op == 7:
            if len(cur) >= 3:
                rings.append(cur)
            cur = []
            continue
        for _ in range(cnt):
            dx, dy = cmds[i], cmds[i + 1]
            i += 2
            x += (dx >> 1) ^ -(dx & 1)
            y += (dy >> 1) ^ -(dy & 1)
            if op == 1:
                if len(cur) >= 3:
                    rings.append(cur)
                cur = [(x, y)]
            else:
                cur.append((x, y))
    return rings


@_damaged
def decode(tile: bytes, layer_name: str = 'building') -> tuple[int, list[dict]]:
    """(extent, [{'props', 'rings' (tile units, exterior rings only)}]) of polygon features in one layer."""
    for f, _wt, layer in _fields(tile):
        if f != 3:
            continue
        name, feats, keys, vals, extent = None, [], [], [], 4096
        for g, _w, v in _fields(layer):
            if g == 1:
                name = v.decode()
            elif g == 2:
                feats.append(v)
            elif g == 3:
                keys.append(v.decode())
            elif g == 4:
                vals.append(_value(v))
            elif g == 5:
                extent = v
        if name != layer_name:
            continue
        out = []
        for fb in feats:
            tags, gtype, geom = [], 0, []
            for g, _w, v in _fields(fb):
                if g == 2:
                    tags = _packed(v)
                elif g == 3:
                    gtype = v
                elif g == 4:
                    geom = _packed(v)
            if gtype != 3:
                continue
            props = {keys[tags[k]]: vals[tags[k + 1]] for k in range(0, len(tags) - 1, 2)}
            ext = [r for r in _rings(geom) if _signed(r) > 0]   # exterior: clockwise on screen (y down)
            out.append({'props': props, 'rings': ext})
        return extent, out
    return 4096, []


def _signed(r) -> float:
    a = np.asarray(r, float)
    return 0.5 * float(np.dot(a[:, 0], np.roll(a[:, 1], -1)) - np.dot(a[:, 1], np.roll(a[:, 0], -1)))


def clip_box(ring, lo: float, hi: float) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip of a ring to the square [lo, hi]^2 (tile extent without its buffer)."""
    pts = list(ring)
    for axis, bound, keep_le in ((0, lo, False), (0, hi, True), (1, lo, False), (1, hi, True)):
        if not pts:
            break
        out = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            cin = cur[axis] <= bound if keep_le else cur[axis] >= bound
            pin = prev[axis] <= bound if keep_le else prev[axis] >= bound
            if cin != pin:
                t = (bound - prev[axis]) / (cur[axis] - prev[axis])
                out.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
            if cin:
                out.append(cur)
        pts = out
    return pts


def records(tiles: list[tuple[int, int, int, int, list[dict]]]) -> list[dict]:
    """Decoded tiles (z, x, y, extent, features) -> building records; a building cut by tile edges keeps one ring per
    tile piece (the survey rasterises their union)."""
    by_id: dict[str, dict] = {}
    for z, x, y, extent, feats in tiles:
        for f in feats:
            p = f['props']
            rings = []
            for r in f['rings']:
                c = clip_box(r, 0, extent)
                if len(c) >= 3:
                    a = np.asarray(c, float)
                    lon, lat = geo.tile_lonlat(x + a[:, 0] / extent, y + a[:, 1] / extent, z)
                    rings.append([[round(float(la), 8), round(float(lo), 8)] for la, lo in zip(lat, lon)])
            if not rings:
                continue
            key = str(p.get('id') or f'{z}/{x}/{y}/{len(by_id)}')
            rec = by_id.get(key)
            if rec is None:
                h = p.get('height')
                rec = by_id[key] = {
                    'ref': key, 'rings': [], 'ground_h': None, 'top_h': None, 'eave_h': None,
                    'height_m': round(float(h), 2) if isinstance(h, (int, float)) else None,
                    'roof': {'flat': 'flat', 'gabled': 'gabled', 'hipped': 'hipped', 'skillion': 'mono_pitch',
                             'pyramidal': 'hipped', 'gambrel': 'complex', 'mansard': 'complex', 'dome': 'complex',
                             'round': 'complex', 'onion': 'complex', 'saltbox': 'complex'}.get(p.get('roof_shape')),
                    'storeys': int(p['num_floors']) if isinstance(p.get('num_floors'), (int, float)) else None,
                    'function': p.get('class') or p.get('subtype'), 'planes': [],
                    'roof_height_m': p.get('roof_height'), 'min_height_m': p.get('min_height'),
                    'geometry_source': p.get('@geometry_source'), 'height_source': p.get('@height_source')}
            rec['rings'] += rings
    return list(by_id.values())


def _touches(rec: dict, bbox, pad_deg: float = 0.0) -> bool:
    s, w, n, e = bbox
    for ring in rec['rings']:
        a = np.asarray(ring, float)
        if a[:, 0].max() >= s - pad_deg and a[:, 0].min() <= n + pad_deg and a[:, 1].max() >= w - pad_deg \
                and a[:, 1].min() <= e + pad_deg:
            return True
    return False


def latest_release() -> str:
    cat = json.loads(net.get(STAC, timeout=20))
    rel = cat.get('latest')
    if not rel:
        kids = [ln for ln in cat.get('links', []) if ln.get('rel') == 'child']
        rel = next((ln['title'].split()[0] for ln in kids if ln.get('latest')), None)
    if not rel:
        raise ValueError('no latest release in the Overture STAC catalogue')
    return rel


class Overture(Provider):
    def __init__(self):
        super().__init__(
            name='overture', title='Overture Maps buildings (PMTiles, zoom 14)', kind='buildings', scope='global',
            licence='ODbL 1.0 (buildings theme)',
            attribution='(c) OpenStreetMap contributors, Overture Maps Foundation',
            url='https://docs.overturemaps.org/attribution',
            fuses='evidence per building: footprint overlap, height, floors and roof shape agree or conflict with the '
                  'mesh (never overrides it: many heights are estimates)')

    def fetch(self, bbox, folder: Path, refresh: bool = False) -> dict:
        out = folder / 'overture.json'
        if cached(out, refresh):
            return {'covered': True, 'buildings': len(json.loads(out.read_text())), 'cached': True}
        release = latest_release()
        arch = Archive(TILES.format(release=release))
        z = min(ZOOM, arch.h['max_zoom'] or ZOOM)
        want = geo.bbox_tiles(bbox, z)
        if len(want) > 9:
            raise ValueError(f'{len(want)} Overture tiles: the bbox is too large (cap 9)')
        decoded = []
        for x, y in want:
            raw = arch.tile(z, x, y)
            if raw:
                extent, feats = decode(raw)
                decoded.append((z, x, y, extent, feats))
        recs = [r for r in records(decoded) if _touches(r, bbox, pad_deg=0.0005)]
        out.write_text(json.dumps(recs))
        with_h = sum(1 for r in recs if r['height_m'] is not None)
        return {'covered': True, 'release': release, 'tiles': len(decoded), 'buildings': len(recs),
                'with_height': with_h}

    def load(self, folder: Path):
        p = folder / 'overture.json'
        return json.loads(p.read_text()) if p.is_file() else None
