"""Offline tests of the scan data providers (readers, coverage, fusion), the street-view stage (render, lift,
fusion) and their survey integration. Never reaches the network or the game: provider downloads are replaced by
in-test data, and net.* is patched to fail if anything tries."""
import gzip
import json
import math
import struct
import tempfile
import types
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np

from uncrashed_mcp import api, reference
from uncrashed_mcp.scan import mapping, native, ops, store, streetview, survey, vision
from uncrashed_mcp.scan import providers as prov
from uncrashed_mcp.scan.providers import bavaria, fuse, geo, geotiff, net, netherlands, overture

from test_scan import NATIVE, Sandbox, make_scene

_guards = []
_REAL_NET = {'get': net.get, 'get_range': net.get_range}   # for tests that stub the socket layer


def setUpModule():
    for name in ('get', 'get_range', 'exists'):
        g = patch.object(net, name, side_effect=AssertionError(f'net.{name} reached from a test'))
        g.start()
        _guards.append(g)


def tearDownModule():
    while _guards:
        _guards.pop().stop()


# ---------------------------------------------------------------------------- tiny writers used by the tests


def lzw_encode(data: bytes) -> bytes:
    """TIFF LZW (MSB-first, early change) — the inverse of native sr_lzw, for test fixtures."""
    table = {bytes([i]): i for i in range(256)}
    nxt, width, out, acc, nbits = 258, 9, bytearray(), 0, 0

    def put(code):
        nonlocal acc, nbits
        acc = (acc << width) | code
        nbits += width
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)

    put(256)
    w = b''
    for b in data:
        wc = w + bytes([b])
        if wc in table:
            w = wc
            continue
        put(table[w])
        table[wc] = nxt
        nxt += 1
        if nxt >= (1 << width) and width < 12:   # the encoder runs one code ahead of the (early change) decoder
            width += 1
        if nxt >= 4094:
            put(256)
            table = {bytes([i]): i for i in range(256)}
            nxt, width = 258, 9
        w = bytes([b])
    if w:
        put(table[w])
    put(257)
    if nbits:
        out.append((acc << (8 - nbits)) & 0xFF)
    return bytes(out)


def write_tiff(arr: np.ndarray, *, compression=1, predictor=1, tile=None, scale=(1.0, 1.0),
               tie=(500.0, 6000.0), epsg=25832, pixel_is_point=False, nodata=None) -> bytes:
    """Little-endian classic single-band GeoTIFF."""
    arr = np.ascontiguousarray(arr)
    H, W = arr.shape
    bits = arr.dtype.itemsize * 8
    fmt = 3 if arr.dtype.kind == 'f' else 1

    def encode(block):
        raw = np.ascontiguousarray(block)
        if predictor == 2:
            raw = np.concatenate([raw[:, :1], np.diff(raw, axis=1)], 1).astype(arr.dtype)
            data = raw.tobytes()
        elif predictor == 3:
            nb = arr.dtype.itemsize
            b = raw.view(np.uint8).reshape(raw.shape[0], raw.shape[1], nb)[:, :, ::-1]   # big-endian planes
            planes = b.transpose(0, 2, 1).reshape(raw.shape[0], -1)
            d = np.concatenate([planes[:, :1], np.diff(planes.astype(np.int16), axis=1).astype(np.uint8)], 1)
            data = d.astype(np.uint8).tobytes()
        else:
            data = raw.tobytes()
        if compression == 8:
            return zlib.compress(data)
        if compression == 5:
            return lzw_encode(data)
        return data

    blocks = []
    if tile:
        tw, th = tile
        for ty in range(0, H, th):
            for tx in range(0, W, tw):
                b = np.zeros((th, tw), arr.dtype)
                part = arr[ty:ty + th, tx:tx + tw]
                b[:part.shape[0], :part.shape[1]] = part
                blocks.append(encode(b))
    else:
        rps = 3
        for y in range(0, H, rps):
            blocks.append(encode(arr[y:y + rps]))
    tags = [(256, 3, [W]), (257, 3, [H]), (258, 3, [bits]), (259, 3, [compression]), (262, 3, [1]),
            (277, 3, [1]), (317, 3, [predictor]), (339, 3, [fmt]),
            (33550, 12, [scale[0], scale[1], 0.0]), (33922, 12, [0.0, 0.0, 0.0, tie[0], tie[1], 0.0]),
            (34735, 3, [1, 1, 0, 2, 3072, 0, 1, epsg, 1025, 0, 1, 2 if pixel_is_point else 1])]
    if nodata is not None:
        tags.append((42113, 2, (str(nodata) + '\0').encode()))
    if tile:
        tags += [(322, 3, [tile[0]]), (323, 3, [tile[1]])]
    else:
        tags += [(278, 3, [3])]
    off_tag, cnt_tag = (324, 325) if tile else (273, 279)
    tags += [(off_tag, 4, [0] * len(blocks)), (cnt_tag, 4, [len(b) for b in blocks])]
    tags.sort()
    body = bytearray(b'II*\0' + struct.pack('<I', 8))
    ifd_size = 2 + 12 * len(tags) + 4
    extra_at = 8 + ifd_size
    extra = bytearray()
    data_at = None
    entries = []
    sizes = {3: 2, 4: 4, 12: 8, 2: 1}
    fmtc = {3: 'H', 4: 'I', 12: 'd'}
    for tag, typ, vals in tags:
        if typ == 2:
            payload = vals
            count = len(vals)
        else:
            payload = struct.pack('<' + fmtc[typ] * len(vals), *vals)
            count = len(vals)
        entries.append([tag, typ, count, payload])
    # block offsets are known only after the extra area: two passes
    for _pass in range(2):
        extra = bytearray()
        layout = []
        for tag, typ, count, payload in entries:
            if len(payload) <= 4:
                layout.append((tag, typ, count, payload.ljust(4, b'\0')))
            else:
                layout.append((tag, typ, count, struct.pack('<I', extra_at + len(extra))))
                extra += payload
                if len(extra) % 2:
                    extra += b'\0'
        data_at = extra_at + len(extra)
        offs, pos = [], data_at
        for b in blocks:
            offs.append(pos)
            pos += len(b)
        for e in entries:
            if e[0] == off_tag:
                e[3] = struct.pack('<' + 'I' * len(offs), *offs)
    out = bytearray(body)
    out += struct.pack('<H', len(layout))
    for tag, typ, count, val in layout:
        out += struct.pack('<HHI', tag, typ, count) + val
    out += struct.pack('<I', 0)
    out += extra
    for b in blocks:
        out += b
    return bytes(out)


def varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


def pb(field: int, wire: int, payload) -> bytes:
    key = varint(field << 3 | wire)
    if wire == 0:
        return key + varint(payload)
    return key + varint(len(payload)) + payload


def mvt_tile(features: list[tuple[dict, list[list[tuple[int, int]]]]], layer='building', extent=4096) -> bytes:
    keys, vals, feats = [], [], b''
    for props, rings in features:
        tags = []
        for k, v in props.items():
            if k not in keys:
                keys.append(k)
            if v not in vals:
                vals.append(v)
            tags += [keys.index(k), vals.index(v)]
        cmds, x, y = [], 0, 0
        for ring in rings:
            zz = lambda d: (d << 1) ^ (d >> 31)   # noqa: E731
            px, py = ring[0]
            cmds += [1 | (1 << 3), zz(px - x), zz(py - y)]
            x, y = px, py
            cmds.append(2 | ((len(ring) - 1) << 3))
            for px, py in ring[1:]:
                cmds += [zz(px - x), zz(py - y)]
                x, y = px, py
            cmds.append(7 | (1 << 3))
        f = pb(2, 2, b''.join(varint(t) for t in tags)) + pb(3, 0, 3) + pb(4, 2, b''.join(varint(c) for c in cmds))
        feats += pb(2, 2, f)
    def value(v):   # noqa: E306
        if isinstance(v, str):
            return pb(1, 2, v.encode())
        return b'\x19' + struct.pack('<d', float(v))   # field 3 (double), wire 1
    lay = pb(1, 2, layer.encode()) + feats + b''.join(pb(3, 2, k.encode()) for k in keys) \
        + b''.join(pb(4, 2, value(v)) for v in vals) + pb(5, 0, extent) + pb(15, 0, 2)
    return pb(3, 2, lay)


# ---------------------------------------------------------------------------- geodesy and readers


class GeodesyTests(unittest.TestCase):
    def test_utm_matches_an_independent_formula_and_round_trips(self):
        e, n = geo.utm_forward(49.4539, 11.0775, 32)
        # Snyder's series (computed separately for the scan origin): agree to about a millimetre
        self.assertAlmostEqual(float(e), 650562.9346, delta=0.003)
        self.assertAlmostEqual(float(n), 5479990.3176, delta=0.003)
        lat, lon = geo.utm_inverse(e, n, 32)
        self.assertAlmostEqual(float(lat), 49.4539, places=9)
        self.assertAlmostEqual(float(lon), 11.0775, places=9)
        self.assertEqual(geo.utm_zone(11.5), 32)

    def test_rd_new_origin_and_round_trip(self):
        x, y = netherlands.rd_forward(netherlands.LAT0, netherlands.LON0)
        self.assertAlmostEqual(float(x), 155000, places=3)
        self.assertAlmostEqual(float(y), 463000, places=3)
        x, y = netherlands.rd_forward(52.0116, 4.3571)          # Delft
        self.assertTrue(80000 < x < 90000 and 440000 < y < 452000)
        lat, lon = netherlands.rd_inverse(x, y)
        self.assertAlmostEqual(float(lat), 52.0116, places=6)
        self.assertAlmostEqual(float(lon), 4.3571, places=6)

    def test_web_mercator_tiles_cover_the_bbox(self):
        tiles = geo.bbox_tiles((49.4527, 11.0754, 49.4552, 11.0795), 14)
        self.assertIn(geo.tile_xy(49.4539, 11.0775, 14), tiles)
        lon, lat = geo.tile_lonlat(*geo.tile_xy(49.4539, 11.0775, 14), 14)
        self.assertLessEqual(float(lon), 11.0775)
        self.assertGreaterEqual(float(lat), 49.4539)


@unittest.skipUnless(NATIVE, 'needs g++ and libturbojpeg headers to build native/scanraster')
class GeoTiffTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.f = (500 + rng.normal(0, 3, (37, 45))).astype(np.float32)
        self.u = rng.integers(0, 200, (37, 45)).astype(np.uint8)

    def test_strips_lzw_deflate_and_none_read_back_exactly(self):
        for comp in (1, 5, 8):
            t = geotiff.open_bytes(write_tiff(self.f, compression=comp))
            np.testing.assert_array_equal(t.window(0, 0, 45, 37), self.f, err_msg=f'compression {comp}')
            np.testing.assert_array_equal(t.window(10, 5, 30, 33), self.f[5:33, 10:30])
        self.assertEqual(t.epsg, 25832)
        self.assertEqual(t.transform(), (500.0, 6000.0, 1.0, 1.0))

    def test_tiles_with_horizontal_and_floating_point_predictors(self):
        t = geotiff.open_bytes(write_tiff(self.u, compression=8, predictor=2, tile=(16, 16)))
        np.testing.assert_array_equal(t.window(3, 2, 40, 30), self.u[2:30, 3:40])
        t = geotiff.open_bytes(write_tiff(self.f, compression=8, predictor=3, tile=(16, 16)))
        np.testing.assert_array_equal(t.window(0, 0, 45, 37), self.f)

    def test_lzw_long_input_resets_the_table(self):
        data = np.random.default_rng(1).integers(0, 256, 60000).astype(np.uint8).tobytes()
        self.assertEqual(native.lzw(lzw_encode(data), len(data)), data)
        with self.assertRaises(native.NativeError):
            native.lzw(b'\x00\x7f\xff\xff\xff', 100)

    def test_pixel_is_point_and_nodata(self):
        t = geotiff.open_bytes(write_tiff(self.f, scale=(0.5, 0.5), tie=(11.0, 49.0), epsg=4326, pixel_is_point=True,
                                          nodata=-9999))
        self.assertEqual(t.transform(), (10.75, 49.25, 0.5, 0.5))
        self.assertEqual(t.nodata, -9999.0)
        self.assertEqual(t.epsg, 4326)


class RasterTests(unittest.TestCase):
    def test_bilinear_nearest_nodata_and_round_trip(self):
        a = np.arange(16, dtype=np.float32).reshape(4, 4)
        a[0, 0] = -1
        r = prov.Raster(a, 4326, (10.0, 50.0, 0.1, 0.1), 'dtm', -1)
        # centre of cell (1, 1) is lon 10.15, lat 49.85
        self.assertAlmostEqual(float(r.sample(49.85, 10.15)), 5.0)
        self.assertAlmostEqual(float(r.sample(49.80, 10.20)), (5 + 6 + 9 + 10) / 4)
        self.assertTrue(np.isnan(r.sample(49.95, 10.05)))     # touches the nodata cell
        self.assertEqual(float(r.sample(49.92, 10.18, nearest=True)), 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'r.npz'
            r.save(p)
            back = prov.Raster.read(p)
        self.assertEqual((back.epsg, back.kind, back.nodata, back.transform), (4326, 'dtm', -1.0, r.transform))

    def test_utm_raster_is_sampled_in_its_own_crs(self):
        e, n = geo.utm_forward(49.45, 11.07, 32)
        x0, y0 = math.floor(float(e)) - 5, math.floor(float(n)) + 5
        a = np.tile(np.arange(10, dtype=np.float32), (10, 1))    # value = column index
        r = prov.Raster(a, 25832, (x0, y0, 1.0, 1.0), 'dtm')
        self.assertAlmostEqual(float(r.sample(49.45, 11.07)), float(e) - x0 - 0.5, places=4)


class PmtilesMvtTests(unittest.TestCase):
    def test_hilbert_tile_ids_follow_the_spec(self):
        self.assertEqual(overture.zxy_to_tileid(0, 0, 0), 0)
        self.assertEqual([overture.zxy_to_tileid(1, x, y) for x, y in ((0, 0), (0, 1), (1, 1), (1, 0))], [1, 2, 3, 4])
        self.assertEqual(overture.zxy_to_tileid(2, 0, 0), 5)

    def test_directory_round_trip_and_lookup(self):
        entries = [(5, 0, 100, 1), (7, 100, 50, 2), (20, 0, 300, 0)]   # tile, tile run of 2, leaf
        raw = varint(len(entries))
        last = 0
        for t, *_ in entries:
            raw += varint(t - last)
            last = t
        raw += b''.join(varint(e[3]) for e in entries) + b''.join(varint(e[2]) for e in entries)
        raw += varint(0 + 1) + varint(0) + varint(0 + 1)      # offsets: 0, then contiguous, then 0 (leaf area)
        d = overture.directory(raw)
        self.assertEqual(d, [(5, 0, 100, 1), (7, 100, 50, 2), (20, 0, 300, 0)])
        self.assertEqual(overture.find(d, 8), ('tile', 100, 50))
        self.assertIsNone(overture.find(d, 9))
        self.assertEqual(overture.find(d, 25), ('leaf', 0, 300))
        self.assertIsNone(overture.find(d, 4))

    def test_vector_tile_buildings_become_lat_lon_records_clipped_to_the_tile(self):
        ring = [(100, 100), (300, 100), (300, 300), (100, 300)]     # clockwise on screen (y down): exterior
        out_ring = [(4000, 2000), (4300, 2000), (4300, 2200), (4000, 2200)]   # crosses the tile edge (buffer)
        tile = mvt_tile([({'id': 'b1', 'height': 12.5, 'num_floors': 4, 'roof_shape': 'gabled'}, [ring]),
                         ({'id': 'b2'}, [out_ring])])
        extent, feats = overture.decode(tile)
        self.assertEqual(extent, 4096)
        self.assertEqual(len(feats), 2)
        self.assertEqual(feats[0]['props']['height'], 12.5)
        z, x, y = 14, *geo.tile_xy(49.45, 11.07, 14)
        recs = overture.records([(z, x, y, extent, feats)])
        b1 = next(r for r in recs if r['ref'] == 'b1')
        self.assertEqual((b1['height_m'], b1['storeys'], b1['roof']), (12.5, 4, 'gabled'))
        lon0, lat0 = geo.tile_lonlat(x + 100 / 4096, y + 100 / 4096, z)
        self.assertAlmostEqual(b1['rings'][0][0][0], float(lat0), places=7)
        self.assertAlmostEqual(b1['rings'][0][0][1], float(lon0), places=7)
        b2 = next(r for r in recs if r['ref'] == 'b2')
        lon_edge, _ = geo.tile_lonlat(x + 1, y, z)
        self.assertLessEqual(max(p[1] for p in b2['rings'][0]), float(lon_edge) + 1e-7)   # clipped at the edge
        bbox = (float(lat0) - 0.0001, float(lon0) - 0.0001, float(lat0) + 0.0001, float(lon0) + 0.0001)
        self.assertTrue(overture._touches(b1, bbox))
        self.assertFalse(overture._touches(b2, bbox))


GML = """<?xml version="1.0" encoding="UTF-8"?>
<CityModel xmlns="http://www.opengis.net/citygml/1.0" xmlns:gml="http://www.opengis.net/gml"
 xmlns:bldg="http://www.opengis.net/citygml/building/1.0" xmlns:gen="http://www.opengis.net/citygml/generics/1.0">
<cityObjectMember><bldg:Building gml:id="DEBY_A">
<gen:stringAttribute name="HoeheGrund"><gen:value>506.5</gen:value></gen:stringAttribute>
<gen:stringAttribute name="HoeheDach"><gen:value>514.5</gen:value></gen:stringAttribute>
<bldg:function>31001_1000</bldg:function><bldg:roofType>3100</bldg:roofType>
<bldg:measuredHeight uom="m">8.0</bldg:measuredHeight><bldg:storeysAboveGround>2</bldg:storeysAboveGround>
<bldg:boundedBy><bldg:GroundSurface><bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember><gml:Polygon>
<gml:exterior><gml:LinearRing><gml:posList>650600 5479500 506.5 650610 5479500 506.5 650610 5479508 506.5 650600 5479508 506.5 650600 5479500 506.5</gml:posList></gml:LinearRing></gml:exterior>
</gml:Polygon></gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface></bldg:GroundSurface></bldg:boundedBy>
<bldg:boundedBy><bldg:RoofSurface><bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember><gml:Polygon>
<gml:exterior><gml:LinearRing><gml:posList>650600 5479500 512.5 650610 5479500 512.5 650610 5479504 514.5 650600 5479504 514.5 650600 5479500 512.5</gml:posList></gml:LinearRing></gml:exterior>
</gml:Polygon></gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface></bldg:RoofSurface></bldg:boundedBy>
<bldg:boundedBy><bldg:WallSurface/></bldg:boundedBy>
</bldg:Building></cityObjectMember>
<cityObjectMember><bldg:Building gml:id="DEBY_FAR"><bldg:roofType>1000</bldg:roofType>
<bldg:boundedBy><bldg:GroundSurface><bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember><gml:Polygon>
<gml:exterior><gml:LinearRing><gml:posList>654000 5479500 500 654010 5479500 500 654010 5479510 500 654000 5479500 500</gml:posList></gml:LinearRing></gml:exterior>
</gml:Polygon></gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface></bldg:GroundSurface></bldg:boundedBy>
</bldg:Building></cityObjectMember>
</CityModel>"""


class RegionalParserTests(unittest.TestCase):
    def test_citygml_buildings_in_the_box_with_roof_planes_and_heights(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 't.gml'
            p.write_text(GML)
            raw = bavaria.parse_citygml(p, (650500, 5479400, 650700, 5479600))
        self.assertEqual([b['ref'] for b in raw], ['DEBY_A'])
        recs = bavaria.to_records(raw)
        r = recs[0]
        self.assertEqual((r['roof'], r['storeys'], r['ground_h'], r['top_h'], r['eave_h'], r['height_m'], r['walls']),
                         ('gabled', 2, 506.5, 514.5, 512.5, 8.0, 1))
        self.assertEqual(len(r['planes']), 1)
        lat, lon = r['ring'][0]
        e, n = geo.utm_forward(lat, lon, 32)
        self.assertAlmostEqual(float(e), 650600, delta=0.01)
        self.assertAlmostEqual(float(n), 5479500, delta=0.01)

    def test_tile_names_of_a_utm_box(self):
        self.assertEqual(bavaria.tiles((650413, 5479591, 650713, 5479891), 1), ['650_5479'])
        self.assertEqual(bavaria.tiles((650900, 5479900, 651100, 5480100), 1),
                         ['650_5479', '651_5479', '650_5480', '651_5480'])

    def test_3dbag_cityjson_feature_records(self):
        feat = {'id': 'NL.P.1', 'vertices': [[0, 0, 0], [10000, 0, 0], [10000, 8000, 0], [0, 8000, 0],
                                             [0, 0, 6000], [10000, 0, 6000], [10000, 4000, 9000]],
                'CityObjects': {'NL.P.1': {'type': 'Building', 'attributes': {'b3_h_maaiveld': 0.5,
                                                                                'b3_h_dak_max': 9.5, 'b3_h_dak_min': 6.5,
                                                                                'b3_dak_type': 'slanted',
                                                                                'b3_bouwlagen': 3},
                                           'children': ['NL.P.1-0'],
                                           'geometry': [{'lod': '0', 'type': 'MultiSurface',
                                                         'boundaries': [[[0, 1, 2, 3]]]}]},
                                'NL.P.1-0': {'type': 'BuildingPart', 'geometry': [{
                                    'lod': '2.2', 'type': 'Solid', 'boundaries': [[[[4, 5, 6]], [[0, 1, 5, 4]]]],
                                    'semantics': {'surfaces': [{'type': 'RoofSurface'}, {'type': 'WallSurface'}],
                                                  'values': [[0, 1]]}}]}}}
        recs = netherlands.records([feat], {'scale': [0.001, 0.001, 0.001], 'translate': [85000, 447000, 0]})
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual((r['height_m'], r['storeys'], r['roof'], len(r['planes']), len(r['ring'])), (9.0, 3, 'complex', 1, 4))
        x, y = netherlands.rd_forward(*r['ring'][1])
        self.assertAlmostEqual(float(x), 85010, delta=0.01)


# ---------------------------------------------------------------------------- choosing and fetching


class FakeProvider(prov.Provider):
    def __init__(self, name, kind='terrain', envelope=None, data=None, fail=None, override=False):
        super().__init__(name=name, title=name, kind=kind, scope='regional' if envelope else 'global', licence='CC0',
                         attribution=f'{name} people', url='https://example.org', fuses='test', region='Testland',
                         envelope=envelope, override=override)
        self.data, self.fail, self.calls = data, fail, 0

    def fetch(self, bbox, folder, refresh=False):
        self.calls += 1
        if isinstance(self.fail, BaseException):
            raise self.fail
        if self.fail:
            raise net.NetError(self.fail)
        (folder / 'data.json').write_text('1')
        return {'covered': True}

    def load(self, folder):
        return self.data


class ChooseGatherTests(unittest.TestCase):
    def test_choose_auto_none_names_and_minus(self):
        self.assertEqual(prov.choose('auto'), (None, True))
        self.assertEqual(prov.choose('none'), ([], False))
        self.assertEqual(prov.choose('by_lod2,glo30'), (['by_lod2', 'glo30'], False))
        names, auto = prov.choose('auto,-overture')
        self.assertTrue(auto)
        self.assertNotIn('overture', names)
        self.assertIn('by_dgm1', names)
        with self.assertRaisesRegex(ValueError, "unknown provider 'nope'"):
            prov.choose('nope')

    def test_coverage_of_regional_providers(self):
        nuremberg = (49.452, 11.075, 49.455, 11.080)
        self.assertTrue(bavaria.DGM1().coverage(nuremberg)[0])
        self.assertFalse(bavaria.LoD2().coverage((50.94, 6.96, 50.95, 6.97))[0])   # outside Bavaria
        self.assertTrue(netherlands.Bag3D().coverage((52.0, 4.35, 52.01, 4.36))[0])
        self.assertFalse(netherlands.Bag3D().coverage(nuremberg)[0])

    def test_gather_records_coverage_failures_and_the_declined_source(self):
        inside = FakeProvider('inside', envelope=(40, 0, 60, 20))
        outside = FakeProvider('outside', envelope=(-10, -10, 0, 0))
        broken = FakeProvider('broken', fail='host down')
        osm = FakeProvider('osm', kind='map')
        with tempfile.TemporaryDirectory() as tmp, patch.object(prov, 'registry', lambda: [osm, inside, outside, broken]):
            recs = prov.gather(Path(tmp), (48, 11, 48.1, 11.1), osm=False)
            self.assertTrue(recs['inside']['used'])
            self.assertEqual(recs['outside'], {**outside.info(), 'used': False, 'covers': False,
                                               'why': 'outside Testland'})
            self.assertFalse(recs['broken']['used'])
            self.assertIn('host down', recs['broken']['why'])
            self.assertEqual(osm.calls, 0)
            self.assertTrue(recs['mapillary']['declined'])
            self.assertTrue((Path(tmp) / 'providers' / 'inside' / prov.SUMMARY).is_file())
            with self.assertRaisesRegex(ValueError, 'provider broken: fetch failed'):
                prov.gather(Path(tmp), (48, 11, 48.1, 11.1), want='broken', osm=False)
            recs = prov.gather(Path(tmp), (48, 11, 48.1, 11.1), want='none', osm=True)
            self.assertTrue(recs['osm']['used'])
            self.assertEqual(recs['inside']['why'], 'not chosen (--providers)')

    def test_auto_gather_records_parse_and_decoder_failures_and_keeps_going(self):
        import http.client
        import xml.etree.ElementTree as ET
        bad = [FakeProvider('xml', fail=ET.ParseError('no element found')),
               FakeProvider('tiff', fail=struct.error('unpack requires a buffer of 8 bytes')),
               FakeProvider('body', fail=http.client.IncompleteRead(b'x', 10)),
               FakeProvider('lzw', fail=native.NativeError('corrupt LZW'))]
        good = FakeProvider('good')
        with tempfile.TemporaryDirectory() as tmp, patch.object(prov, 'registry', lambda: [*bad, good]):
            recs = prov.gather(Path(tmp), (48, 11, 48.1, 11.1), want='auto', osm=False)
            self.assertTrue(recs['good']['used'])
            for p in bad:
                self.assertFalse(recs[p.name]['used'])
                self.assertIn('fetch failed', recs[p.name]['why'])
            self.assertIn('ParseError', recs['xml']['why'])
            with self.assertRaisesRegex(ValueError, 'provider xml: fetch failed'):
                prov.gather(Path(tmp), (48, 11, 48.1, 11.1), want='xml', osm=False)

    def test_damaged_downloads_raise_value_errors(self):
        good = write_tiff(np.arange(64, dtype=np.float32).reshape(8, 8), compression=8)
        for n in (10, 20, 40, 60):                      # truncated headers / directories
            with self.assertRaises(ValueError):
                geotiff.open_bytes(good[:n])
        t = geotiff.open_bytes(good)
        t.src = geotiff._Bytes(t.src.data[:t.offsets[0]] + b'\xff' * (len(good) - t.offsets[0]))  # garbled Deflate
        with self.assertRaisesRegex(geotiff.TiffError, 'damaged'):
            t.window(0, 0, 8, 8)
        with self.assertRaises(ValueError):
            overture.directory(varint(3) + varint(1))                     # truncated varints
        with self.assertRaises(ValueError):
            overture._inflate(b'\x1f\x8b\x08\x00garbage', 2)
        with self.assertRaises(ValueError):
            overture.header(b'PMTiles\x03' + b'\0' * 20)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'err.gml'
            p.write_text('<html><body>502 Bad Gateway')
            with self.assertRaisesRegex(ValueError, 'not valid CityGML'):
                bavaria.parse_citygml(p, (0, 0, 1, 1))

    def test_a_body_cut_off_mid_read_is_a_net_error(self):
        import http.client

        class Cut:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                raise http.client.IncompleteRead(b'abc', 100)

        with patch.object(net, '_check', lambda url: None), patch.object(net, '_open', lambda req, t: Cut()):
            with self.assertRaisesRegex(net.NetError, 'IncompleteRead'):
                _REAL_NET['get']('https://example.org/x')
            with self.assertRaisesRegex(net.NetError, 'IncompleteRead'):
                _REAL_NET['get_range']('https://example.org/x', 0, 10)

    def test_downloads_can_be_switched_off(self):
        with patch.dict('os.environ', {'UNCRASHED_PROVIDERS': '0'}):
            with self.assertRaisesRegex(net.NetError, 'UNCRASHED_PROVIDERS=0'):
                net._check('https://example.org/x')


# ---------------------------------------------------------------------------- fusion


class TerrainFusionTests(unittest.TestCase):
    def test_offset_is_fitted_on_open_ground_and_the_model_replaced(self):
        rng = np.random.default_rng(2)
        dtm = (100 + rng.normal(0, 0.03, (60, 60))).astype(np.float32)
        dsm = dtm.copy()
        dsm[20:30, 20:30] += 6                                    # a house: not open ground
        prior = (dtm - 2.0 + 0.5).astype(np.float32)               # provider datum: 1.5 m lower
        prior[20:30, 20:30] -= 0.4                                 # real ground under the house differs
        prior[:, :5] = np.nan                                      # no provider data there
        out, rep = fuse.apply_terrain(dtm, dsm, np.ones_like(dtm, bool), prior, 'dem')
        self.assertTrue(rep['used'])
        self.assertAlmostEqual(rep['offset_m'], 1.5, delta=0.02)
        self.assertLess(rep['residual_p90_m'], 0.1)
        np.testing.assert_allclose(out[25, 25], prior[25, 25] + rep['offset_m'], atol=1e-4)
        self.assertEqual(out[0, 0], dtm[0, 0])                    # kept where the provider has nothing
        self.assertAlmostEqual(rep['replaced_share'], 55 / 60, places=3)

    def test_too_little_overlap_keeps_the_scan_model(self):
        dtm = np.zeros((10, 10), np.float32)
        out, rep = fuse.apply_terrain(dtm, dtm, np.ones_like(dtm, bool), np.full_like(dtm, np.nan), 'dem')
        self.assertFalse(rep['used'])
        self.assertIs(out, dtm)


LAT0, LON0 = 49.45, 11.07


def rec_from_box(ref, x0, y0, x1, y1, ground, top, roof='flat', planes=True, storeys=None):
    """A provider building record over the scan box (east, south metres), flat roof plane at top (provider datum)."""
    xs, ys = [x0, x1, x1, x0], [y0, y0, y1, y1]
    lat, lon = geo.to_latlon(np.array(xs, float), np.array(ys, float), LAT0, LON0)
    ring = [[float(a), float(b)] for a, b in zip(lat, lon)]
    pl = [[[a, b, top] for a, b in ring]] if planes else []
    return {'ref': ref, 'ring': ring, 'ground_h': ground, 'top_h': top, 'eave_h': top,
            'height_m': None if top is None else top - ground,
            'roof': roof, 'storeys': storeys, 'function': None, 'planes': pl}


@unittest.skipUnless(NATIVE, 'needs g++ and libturbojpeg headers to build native/scanraster')
class BuildingFusionTests(unittest.TestCase):
    """A synthetic survey: 0.25 m grid, flat ground at 10 m, mesh buildings as boxes of cells in objid."""

    def make(self):
        g = survey.Grid(-40, -40, 0.25, 320, 320)
        dsm = np.full((320, 320), 10.0, np.float32)
        objid = np.full((320, 320), -1, np.int32)
        objects = []

        def cells(x0, y0, x1, y1):
            c0, r0 = int((x0 + 40) / 0.25), int((y0 + 40) / 0.25)
            c1, r1 = int((x1 + 40) / 0.25), int((y1 + 40) / 0.25)
            return slice(r0, r1), slice(c0, c1)

        def building(x0, y0, x1, y1, h, cls='building'):
            rs, cs = cells(x0, y0, x1, y1)
            dsm[rs, cs] = 10 + h
            o = {'id': len(objects), 'cls': cls, 'conf': 0.8, 'src': 'mesh', 'x': (x0 + x1) / 2, 'y': (y0 + y1) / 2,
                 'ground_z': 10.0, 'top_z': 10 + h, 'height_m': h, 'length_m': x1 - x0, 'width_m': y1 - y0,
                 'yaw_deg': 0.0, 'area_m2': (x1 - x0) * (y1 - y0), 'rgb': [1, 2, 3], 'roof': 'gabled',
                 'ridge_yaw_deg': None, 'eave_m': h, 'ridge_m': h, 'footprint': [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                 'parts': [], 'osm': None}
            objid[rs, cs] = o['id']
            objects.append(o)
            return o

        self.terrace = building(-30, -30, -10, -22, 7)      # two provider houses merged into one mesh building
        self.house = building(0, 0, 10, 8, 6)              # one provider house
        self.shed = building(20, 20, 24, 24, 3)            # no provider building: flagged
        self.rebuilt = building(-30, 10, -20, 18, 9)       # provider roof at another height: kept, flagged
        self.piece = building(10, 0, 12, 8, 4)             # a strip of the provider house the mesh cut off
        tree = {'id': len(objects), 'cls': 'tree', 'conf': 0.9, 'src': 'mesh', 'x': 5.0, 'y': 4.0, 'ground_z': 10,
                'top_z': 16, 'height_m': 6, 'length_m': 3, 'width_m': 3, 'yaw_deg': 0, 'area_m2': 7, 'rgb': [0, 0, 0]}
        objects.append(tree)
        self.tree = tree
        valid = np.ones_like(dsm, bool)
        valid[:, 300:] = False                              # the scan ends at east 35 m
        sv = types.SimpleNamespace(g=g, objects=objects, objid=objid, dtm=np.full_like(dsm, 10.0), dsmf=dsm,
                                   ndsm=dsm - 10.0, valid=valid, exg_s=np.zeros_like(dsm), veg_px=np.zeros_like(valid),
                                   rgb=np.zeros((320, 320, 3), np.uint8))
        # provider heights are 2 m lower (datum)
        recs = [rec_from_box('T1', -30, -30, -20, -22, 8, 15), rec_from_box('T2', -20, -30, -10, -22, 8, 15),
                rec_from_box('H', 0, 0, 12, 8, 8, 14, storeys=2), rec_from_box('R', -30, 10, -20, 18, 8, 25),
                rec_from_box('EDGE', 30, -10, 45, 0, 8, 14)]
        return sv, recs

    def test_override_split_fragment_and_conflicts(self):
        sv, recs = self.make()
        p = FakeProvider('lod2', kind='buildings', override=True)
        rep = fuse.BuildingFusion(sv, {'lat': LAT0, 'lon': LON0}, 'lod2', p, recs).run()
        self.assertTrue(rep['used'])
        self.assertAlmostEqual(rep['vertical_offset_m'], 2.0, places=2)
        objs = sv.objects
        t = self.terrace
        self.assertEqual((t['ref'], t['split_by'], t['geometry_src']), ('T1', 'lod2', 'lod2'))
        new = [o for o in objs if o.get('ref') == 'T2']
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0]['note'], f"split from mesh building {t['id']}")
        self.assertAlmostEqual(t['footprint'][0][0], -30, delta=0.05)
        self.assertAlmostEqual(max(p[0] for p in t['footprint']), -20, delta=0.05)
        self.assertAlmostEqual(t['height_m'], 7.0, delta=0.05)    # top 15 + 2 offset - ground 10
        self.assertEqual(t['mesh']['footprint'][1], [-10, -30])
        h = self.house
        self.assertEqual((h['ref'], h['storeys'], h['roof'], h['roof_slope_deg']), ('H', 2, 'flat', 0.0))
        self.assertEqual(h['roof_fit'], 1.0 if h['roof_fit'] == 1.0 else h['roof_fit'])
        self.assertIn('lod2', h['src'])
        self.assertEqual(self.piece['cls'], 'building_part')     # 100 % inside the provider house
        self.assertEqual(self.piece['parent'], h['id'])
        self.assertEqual(self.shed['evidence'][-1]['verdict'], 'conflict')
        self.assertLess(self.shed['conf'], 0.8)
        self.assertNotIn('ref', self.rebuilt)                    # roof 8 m off: mesh geometry kept
        self.assertIn('outdated or rebuilt', self.rebuilt['evidence'][-1]['detail'])
        self.assertFalse(any(o.get('ref') == 'EDGE' for o in objs))   # beyond the scan: never adopted
        self.assertEqual(self.tree['evidence'][-1]['verdict'], 'conflict')
        self.assertLessEqual(self.tree['conf'], 0.3)
        self.assertEqual(rep['small_objects_inside_footprints'], 1)
        # the merged mesh cells now carry the new building's id
        self.assertEqual(int(sv.objid[int((-26 + 40) / .25), int((-15 + 40) / .25)]), new[0]['id'])

    def test_a_far_off_footprint_does_not_replace_the_mesh_building(self):
        sv, recs = self.make()
        recs = [rec_from_box('H', 0, 0, 12, 8, 8, 14)]
        # the mesh building is a 2 x 2 m patch inside the 12 x 8 m provider house: paired, but hardly the same thing
        sv.objid[:, :] = -1
        sv.objid[int(42 / .25):int(44 / .25), int(42 / .25):int(44 / .25)] = self.house['id']
        p = FakeProvider('lod2', kind='buildings', override=True)
        fuse.BuildingFusion(sv, {'lat': LAT0, 'lon': LON0}, 'lod2', p, recs).run()
        self.assertNotIn('ref', self.house)
        self.assertIn('mesh geometry kept', self.house['evidence'][-1]['detail'])

    def test_evidence_provider_only_compares(self):
        sv, recs = self.make()
        recs = [rec_from_box('H', 0, 0, 12, 8, None, None)]
        recs[0]['height_m'] = 3.0
        recs[0]['planes'] = []
        p = FakeProvider('ov', kind='buildings', override=False)
        rep = fuse.BuildingFusion(sv, {'lat': LAT0, 'lon': LON0}, 'ov', p, recs).run()
        self.assertEqual(rep['conflict'], 1)
        ev = self.house['evidence'][-1]
        self.assertEqual((ev['src'], ev['verdict']), ('ov', 'conflict'))
        self.assertIn('height 6.0 vs 3.0 m', ev['detail'])
        self.assertNotIn('ref', self.house)
        self.assertEqual(self.terrace['evidence'][-1]['verdict'], 'none')

    def test_fuse_survey_flags_conflicts_for_queries(self):
        sv, recs = self.make()
        sv.terrain_info = {'used': True}
        sv.cls = np.zeros((320, 320), np.uint8)
        p = FakeProvider('lod2', kind='buildings', override=True)
        rep = fuse.fuse_survey(sv, {'lat': LAT0, 'lon': LON0}, {'lod2': (p, recs)})
        self.assertEqual(rep['terrain'], {'used': True})
        self.assertGreater(rep['objects_with_conflicts'], 0)
        self.assertIn('lod2:', self.shed['flag_sources'])
        self.assertIn('sources disagree (lod2)', ops._flag(self.shed))


# ---------------------------------------------------------------------------- street views


class StageReuseTests(Sandbox):
    def test_vision_rerun_on_reuse_brings_the_street_objects_back(self):
        """--no-facades then facades on: the reused survey re-runs vision, which rebuilds objects.json from the
        geometry; the street-view objects must be added again, not lost while meta still reports them."""
        meta = {'key': 'k1', 'skipped': [], 'osm': {'used': True}, 'georef': {'lat': 49.45, 'lon': 11.07}}
        home = store.home('k1')
        home.mkdir(parents=True)
        store.write_json(home / 'objects-geometry.json', [{'id': 1, 'cls': 'building'}])
        calls = []

        def vrun(meta, geometry, ras, **kw):
            return [dict(o) for o in geometry], {}, {'ran': True}

        def srun(meta, objs, ras, feats, shift):
            calls.append(1)
            return objs + [{'id': 900, 'cls': 'lamp'}], {'ran': True, 'added': 1}

        ras = types.SimpleNamespace(objid=np.zeros((2, 2), np.int32))
        with patch.object(vision, 'ready', return_value=(True, '')), patch.object(vision, 'run', vrun), \
                patch.object(vision, 'stamp', lambda *a: None), patch.object(ops, 'Rasters', lambda m: ras), \
                patch.object(store, 'save_array', lambda *a: None), patch.object(streetview, 'run', srun), \
                patch.object(ops, 'osm_answer', lambda m: {}), \
                patch.object(ops.osm, 'parse', lambda *a: {'buildings': [], 'roads': []}):
            ops._vision_stage(meta, None, facades=False, again=True)
            ops._street_stage(meta, None, None, again=True)
            self.assertIn('lamp', [o['cls'] for o in store.read_json(home / 'objects.json')])
            # reuse with facades switched on (the reuse path of scan_survey)
            self.assertTrue(ops._vision_stage(meta, None, facades=True, again=False))
            self.assertTrue(ops._street_stage(meta, None, None, again=False))
        self.assertEqual(len(calls), 2)
        self.assertIn('lamp', [o['cls'] for o in store.read_json(home / 'objects.json')])
        self.assertTrue(meta['street']['ran'])
        self.assertEqual(meta['counts'], {'building': 1, 'lamp': 1})


class LiftClusterTests(unittest.TestCase):
    CAM = {'eye': [0.0, 0.0, 11.6], 'yaw': 0.0, 'pitch': 0.0, 'pos': 0}

    def test_label_words(self):
        self.assertEqual(streetview.label_class('street lamp traffic sign'), 'sign')
        self.assertEqual(streetview.label_class('trash can'), 'bin')
        self.assertIsNone(streetview.label_class('tree'))

    def test_ray_point_and_lift_of_a_bollard(self):
        # a 1 m bollard 8 m east of the eye, ground at 10 m: rows of its top and bottom
        fx = (streetview.W / 2) / math.tan(math.radians(streetview.FOV) / 2)
        top = streetview.H / 2 - fx * (11.0 - 11.6) / 8
        bot = streetview.H / 2 - fx * (10.0 - 11.6) / 8
        dep = np.full((streetview.H, streetview.W), 30.0, np.float32)
        dep[int(top):int(bot) + 1, 315:326] = 8.0
        det = {'class': 'bollard', 'score': 0.6, 'box': [314, top, 326, bot]}
        p = streetview.lift(self.CAM, det, dep, lambda x, y: 10.0, lambda x, y: False)
        self.assertIsNotNone(p, det.get('drop'))
        self.assertAlmostEqual(p['x'], 8.0, delta=0.05)
        self.assertAlmostEqual(p['y'], 0.0, delta=0.1)
        self.assertAlmostEqual(p['height_m'], 1.0, delta=0.05)
        # standing 3 m above the ground (a texture on a wall behind): dropped
        q = streetview.lift(self.CAM, det, dep, lambda x, y: 7.0, lambda x, y: False)
        self.assertIsNone(q)
        self.assertEqual(det['drop'], 'not standing on the ground')
        edge = {'class': 'bollard', 'score': 0.6, 'box': [0, top, 20, bot]}
        self.assertIsNone(streetview.lift(self.CAM, edge, dep, lambda x, y: 10.0, lambda x, y: False))
        self.assertEqual(edge['drop'], 'cut by the image edge')

    def test_cluster_needs_two_camera_positions_or_a_strong_score(self):
        def pt(pos, x, score=0.3, cls='bin'):
            return {'cls': cls, 'score': score, 'x': x, 'y': 0.0, 'ground_z': 10.0, 'height_m': 1.0, 'width_m': 0.6,
                    'pos': pos, 'yaw_view': 0.0, 'eye': [0, 0, 11.6]}
        out = streetview.cluster([pt(0, 5.0), pt(1, 5.4), pt(2, 30.0), pt(2, 30.2), pt(3, 50.0, 0.6)])
        by_x = {round(c['x']): c for c in out}
        self.assertEqual(sorted(by_x), [5, 50])
        self.assertEqual(by_x[5]['views'], 2)
        self.assertLessEqual(by_x[50]['conf'], 0.45)

    def test_plan_keeps_eyes_out_of_crowns(self):
        g = survey.Grid(0, 0, 0.5, 100, 100)
        dsm = np.full((100, 100), 10.0, np.float32)
        dsm[:, 40:60] = 20.0                                  # a crown over the middle of the street
        ras = types.SimpleNamespace(g=g, dsm=dsm, dtm=np.full_like(dsm, 10.0))

        def cell(x, y):
            c, r = int(x / 0.5), int(y / 0.5)
            ok = 0 <= c < 100 and 0 <= r < 100
            return min(max(r, 0), 99), min(max(c, 0), 99), ok
        ras.cell = cell
        cams = streetview.plan({}, ras, {'roads': [{'line': [[2, 25], [48, 25]], 'width': 6}], 'paths': []}, step_m=5)
        self.assertTrue(cams)
        for c in cams:
            self.assertFalse(20 <= c['eye'][0] < 30, c)
        kinds = {(c['kind'], c['pitch']) for c in cams}
        self.assertEqual(kinds, {('street', 0.0), ('drone', -25.0)})
        self.assertEqual({round(c['eye'][2] - 10, 1) for c in cams}, {1.6, streetview.DRONE_M})


@unittest.skipUnless(NATIVE, 'needs g++ and libturbojpeg headers to build native/ucmesh and native/scanraster')
class ViewRenderTests(Sandbox):
    def test_depth_is_the_first_surface_and_the_ground_reaches_the_bottom_rows(self):
        obj = make_scene(self.root / 'src')
        cache = reference.ensure_index(obj)
        amap = reference._axis_map(reference.source_axes(obj, ''))
        # eye 1.6 m above the ground at east 0, south 0, looking north (yaw -90) at the flat-roof house:
        # its south wall is the plane south = -12, so 12 m away
        cams = np.array([[-15, 0, 1.6, -90, 0, 90, 0.3, 60], [0, 0, 1.6, 0, -60, 90, 0.3, 60]], float)
        img, dep, st = native.views(cache, amap, cams, 64, 48)
        self.assertEqual(st['views'], 2)
        self.assertAlmostEqual(float(dep[0, 20, 32]), 12.0, delta=0.05)
        self.assertTrue(np.isfinite(dep[0, -1]).all())          # ground clipped at the near plane, no hole
        self.assertTrue(np.isfinite(dep[1]).all())
        # looking down 60 deg from 1.6 m: the centre ray meets the ground 1.6 / sin 60 along the axis
        self.assertAlmostEqual(float(dep[1, 24, 32]), 1.6 / math.sin(math.radians(60)), delta=0.05)
        self.assertGreater(int(img[0].sum()), 0)


# ---------------------------------------------------------------------------- integration


class WorkerCommandTests(unittest.TestCase):
    def test_the_worker_runs_with_a_safe_path(self):
        seen = {}

        def run(cmd, **kw):
            seen['cmd'] = cmd
            Path(kw['cwd'], 'out.json').write_text(json.dumps({'errors': []}))
            return types.SimpleNamespace(returncode=0, stderr='')
        with tempfile.TemporaryDirectory() as tmp, patch.object(vision.subprocess, 'run', run):
            vision._worker({'x': 1}, Path(tmp))
        # without -P the worker's folder (scan/, holding profile.py) shadows stdlib `profile` and torch fails
        self.assertEqual(seen['cmd'][1], '-P')


class MappingStreetClassesTests(unittest.TestCase):
    def test_street_classes_match_named_rows_only(self):
        names = {a.name for a in mapping.candidates('bin')}
        self.assertNotIn('Logcabin Roof', names)
        for name in names:
            self.assertRegex(name.lower(), r'\b(trash|garbage|dumpster|waste|wheelie|bins?)\b')
        self.assertIn('door', mapping.NOT_MATCHABLE)


@unittest.skipUnless(NATIVE, 'needs g++ and libturbojpeg headers to build native/ucmesh and native/scanraster')
class SurveyIntegrationTests(Sandbox):
    """scan_survey with fake providers: terrain replaces the ground model, a building provider overrides the
    flat-roof house; the providers are recorded and scan_sources reports them."""

    def setUp(self):
        super().setUp()
        self.obj = make_scene(self.root / 'src')
        (self.root / 'src' / 'metadata.json').write_text(json.dumps({'origin': {'lat': LAT0, 'lon': LON0}}))
        # terrain 3 m below the scan datum, as a lat/lon grid over the scene
        lat, lon = geo.to_latlon(np.array([-40.0, 40.0]), np.array([-40.0, 40.0]), LAT0, LON0)
        sx = float(lon[1] - lon[0]) / 80
        sy = float(lat[0] - lat[1]) / 80
        dem = prov.Raster(np.full((80, 80), -3.0, np.float32), 4326, (float(lon[0]), float(lat[0]), sx, sy), 'dtm')
        house = rec_from_box('HOUSE', -20, -20, -10, -12, -3.0, 3.0, storeys=2)
        self.fakes = [FakeProvider('osm', kind='map'), FakeProvider('dem', data=dem),
                      FakeProvider('blds', kind='buildings', data=[house], override=True)]

    def test_providers_fuse_into_the_survey_and_are_reported(self):
        with patch.object(prov, 'registry', lambda: self.fakes):
            out = ops.scan_survey(str(self.obj), use_osm=False, use_vision=False)
            self.assertEqual([u['name'] for u in out['providers']['used']], ['dem', 'blds'])
            self.assertAlmostEqual(out['providers']['terrain']['offset_m'], 3.0, delta=0.05)
            self.assertIn('street views: disabled (needs vision (--no-vision))', out['skipped'])
            meta = store.resolve('')
            objs = store.objects(meta)
            h = next(o for o in objs if o.get('ref') == 'HOUSE')
            self.assertEqual((h['geometry_src'], h['storeys']), ('blds', 2))
            self.assertAlmostEqual(h['height_m'], 6.0, delta=0.1)
            self.assertEqual(h['evidence'][-1]['verdict'], 'agree')
            src = ops.scan_sources()
            used = {r['name'] for r in src['providers'] if r['used']}
            self.assertEqual(used, {'dem', 'blds'})
            self.assertIn('blds: blds people', src['attribution'])
            self.assertEqual(src['declined'][0]['name'], 'mapillary')
            # a rebuild of the RAM rasters fuses again from the cache: the same objects
            n = len(objs)
            store.drop_ram(meta['key'])
            r = ops._build(meta, rebuild=True)
            self.assertEqual(len(r['objects']), n)
            self.assertFalse((store.home(meta['key']) / 'rebuild-mismatch.json').exists())
            # --providers none is another survey without them
            out2 = ops.scan_survey(str(self.obj), use_osm=False, use_vision=False, providers='none')
            self.assertNotEqual(out2['survey'], out['survey'])
            self.assertNotIn('providers', out2)

    def test_place_coverage_query_is_offline(self):
        out = ops.scan_sources(latlon=[49.4539, 11.0775], radius_m=150)
        cov = {r['name']: r['covers'] for r in out['providers']}
        self.assertTrue(cov['by_lod2'] and cov['by_dgm1'] and cov['overture'])
        self.assertFalse(cov['nl_3dbag'])
        with self.assertRaisesRegex(ValueError, 'latlon'):
            ops.scan_sources(latlon=[95, 0])

    def test_api_forwards_the_new_survey_parameters(self):
        seen = {}
        with patch.object(ops, 'scan_survey', lambda *a, **k: seen.update(k) or {}), \
                patch.object(ops, 'scan_sources', lambda **k: seen.update(k) or {}):
            api.call('scan_survey', 'x.obj', providers='by_lod2', street_views=False)
            api.call('scan_sources', latlon=[1, 2], radius_m=50)
        self.assertEqual((seen['providers'], seen['street_views'], seen['latlon'], seen['radius_m']),
                         ('by_lod2', False, [1, 2], 50))


if __name__ == '__main__':
    unittest.main()
