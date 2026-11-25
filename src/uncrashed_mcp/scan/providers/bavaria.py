"""Bavaria (Germany): open geodata of the Bayerische Vermessungsverwaltung (geodaten.bayern.de/opengeodata).

Verified 2026-11-25 in the portal's product list (json/opengeodata_datensaetze.json): DGM1, LoD2, DOM20, DOP20 and
the laser points are CC BY 4.0; coordinates ETRS89 / UTM 32N (EPSG:25832), heights DHHN2016 (normal heights).

  by_dgm1  digital terrain model, 1 m grid, 1 x 1 km GeoTIFF tiles (LZW, float32) named <E km>_<N km>.tif
  by_lod2  3D building models LoD2, CityGML 1.0, 2 x 2 km tiles named <E km>_<N km>.gml (even kilometres);
           ground surface = cadastral wall outline, roof surfaces as planar polygons, roofType codes (AdV)

Coverage: the envelope of Bavaria first (offline), then the tile itself (HEAD request): a tile that does not
exist means the bbox is outside the state. Only the tile names leave the machine.
"""
from __future__ import annotations

import json
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from . import Provider, Raster, cached, geo, geotiff, net

DOWNLOAD = 'https://download1.bayernwolke.de/a'
ZONE = 32
EPSG = 25832
ATTRIBUTION = 'Bayerische Vermessungsverwaltung - www.geodaten.bayern.de (CC BY 4.0)'
ENVELOPE = (47.26, 8.97, 50.57, 13.84)
# AdV roofType codes (Dachform) -> survey roof words
ROOF = {'1000': 'flat', '2100': 'mono_pitch', '2200': 'mono_pitch', '3100': 'gabled', '3200': 'hipped',
        '3300': 'hipped', '3400': 'hipped', '3500': 'complex', '3600': 'complex', '3700': 'complex',
        '3800': 'complex', '3900': 'complex', '4000': 'complex', '5000': 'complex', '9999': 'complex'}
NS = {'bldg': 'http://www.opengis.net/citygml/building/1.0', 'gml': 'http://www.opengis.net/gml',
      'gen': 'http://www.opengis.net/citygml/generics/1.0'}


def utm_box(bbox, pad_m: float = 0.0) -> tuple[float, float, float, float]:
    s, w, n, e = bbox
    E, N = geo.utm_forward([s, s, n, n], [w, e, w, e], ZONE)
    return float(E.min()) - pad_m, float(N.min()) - pad_m, float(E.max()) + pad_m, float(N.max()) + pad_m


def tiles(box, km: int) -> list[str]:
    e0, n0, e1, n1 = box
    out = []
    for n in range(int(n0 // (km * 1000)) * km, int(n1 // (km * 1000)) * km + 1, km):
        for e in range(int(e0 // (km * 1000)) * km, int(e1 // (km * 1000)) * km + 1, km):
            out.append(f'{e}_{n}')
    return out


class DGM1(Provider):
    def __init__(self):
        super().__init__(
            name='by_dgm1', title='Bavaria DGM1 (terrain 1 m, airborne laser)', kind='terrain', scope='regional',
            licence='CC BY 4.0', attribution=ATTRIBUTION,
            url='https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=dgm1', region='Bavaria, Germany',
            envelope=ENVELOPE,
            fuses='replaces the scan ground model where it has data (vertical offset to the scan fitted on open '
                  'ground and reported); the agreement before replacement is reported')

    def fetch(self, bbox, folder: Path, refresh: bool = False) -> dict:
        out = folder / 'dgm1.npz'
        if cached(out, refresh):
            return {'covered': True, 'cached': True}
        box = utm_box(bbox, pad_m=4)
        names = tiles(box, 1)
        if len(names) > 16:
            raise ValueError(f'{len(names)} DGM1 tiles: the bbox is too large (cap 16 km2)')
        e0, n0 = int(box[0]) // 1 - 1, int(box[1]) - 1
        e1, n1 = int(math.ceil(box[2])) + 1, int(math.ceil(box[3])) + 1
        mosaic = np.full((n1 - n0, e1 - e0), np.nan, np.float32)
        got = []
        for name in names:
            url = f'{DOWNLOAD}/dgm/dgm1/{name}.tif'
            if not net.exists(url):
                continue
            t = geotiff.open_bytes(net.get(url))
            if t.epsg not in (None, EPSG):
                raise ValueError(f'DGM1 tile {name} is EPSG:{t.epsg}, expected {EPSG}')
            x0, y0, sx, sy = t.transform()
            c0, c1 = int(math.floor(e0 - x0)), int(math.ceil(e1 - x0))
            r0, r1 = int(math.floor(y0 - n1)), int(math.ceil(y0 - n0))
            a = t.window(c0, r0, c1, r1).astype(np.float32)
            if t.nodata is not None:
                a[a == t.nodata] = np.nan
            cc0, rr0 = max(0, c0), max(0, r0)
            # place: column cc0 of the tile is east x0 + cc0; mosaic column = that - e0
            mc, mr = int(x0 + cc0 - e0), int(n1 - (y0 - rr0))
            h, w = a.shape
            dst = mosaic[mr:mr + h, mc:mc + w]
            src = a[:dst.shape[0], :dst.shape[1]]
            np.copyto(dst, src, where=np.isfinite(src))
            got.append(name)
        if not got:
            return {'covered': False, 'why': 'no DGM1 tile for the bbox (outside Bavaria)'}
        Raster(mosaic, EPSG, (float(e0), float(n1), 1.0, 1.0), 'dtm', None).save(out)
        return {'covered': True, 'tiles': got, 'cells': list(mosaic.shape),
                'valid_share': round(float(np.isfinite(mosaic).mean()), 3)}

    def load(self, folder: Path):
        p = folder / 'dgm1.npz'
        return Raster.read(p) if p.is_file() else None


def _ring(pos: str) -> np.ndarray:
    v = np.array(pos.split(), float)
    return v.reshape(-1, 3)


def _attrs(el) -> dict:
    out = {}
    for a in el.findall('gen:stringAttribute', NS):
        v = a.find('gen:value', NS)
        if v is not None:
            out[a.get('name')] = v.text
    return out


def parse_citygml(path: Path, box) -> list[dict]:
    """Buildings whose ground outline touches box (UTM e0, n0, e1, n1). Rings in UTM metres + DHHN heights.
    A body that is not CityGML (an HTML error page, a cut-off download) raises ValueError."""
    try:
        return _parse_citygml(path, box)
    except ET.ParseError as exc:
        raise ValueError(f'LoD2 tile is not valid CityGML ({exc})') from None


def _parse_citygml(path: Path, box) -> list[dict]:
    e0, n0, e1, n1 = box
    out = []
    bld_tag = f"{{{NS['bldg']}}}Building"
    for _ev, el in ET.iterparse(str(path), events=('end',)):
        if el.tag != bld_tag:
            continue
        ground, roofs, walls = [], [], 0
        for kind, store in (('GroundSurface', ground), ('RoofSurface', roofs)):
            for surf in el.iter(f"{{{NS['bldg']}}}{kind}"):
                for pl in surf.iter(f"{{{NS['gml']}}}posList"):
                    store.append(_ring(pl.text or ''))
        walls = sum(1 for _ in el.iter(f"{{{NS['bldg']}}}WallSurface"))
        el_clear = el.clear
        if not ground:
            el_clear()
            continue
        allg = np.vstack(ground)
        if allg[:, 0].max() < e0 or allg[:, 0].min() > e1 or allg[:, 1].max() < n0 or allg[:, 1].min() > n1:
            el_clear()
            continue
        at = _attrs(el)

        def txt(tag):
            x = el.find(f'bldg:{tag}', NS)
            return x.text if x is not None else None

        out.append({'ref': el.get(f"{{{NS['gml']}}}id"), 'function': txt('function'), 'roof_code': txt('roofType'),
                    'measured_height_m': float(txt('measuredHeight')) if txt('measuredHeight') else None,
                    'storeys': int(txt('storeysAboveGround')) if txt('storeysAboveGround') else None,
                    'ground_h': float(at['HoeheGrund']) if at.get('HoeheGrund') else float(allg[:, 2].min()),
                    'roof_h': float(at['HoeheDach']) if at.get('HoeheDach') else None,
                    'ground': [g.tolist() for g in ground], 'roofs': [r.tolist() for r in roofs], 'walls': walls})
        el_clear()
    return out


def to_records(raw: list[dict]) -> list[dict]:
    """UTM rings -> lat/lon building records (provider heights kept)."""
    recs = []
    for b in raw:
        g = [np.asarray(r, float) for r in b['ground']]
        main = max(g, key=lambda r: abs(_area(r[:, :2])))
        la, lo = geo.utm_inverse(main[:, 0], main[:, 1], ZONE)
        planes = []
        for r in b['roofs']:
            r = np.asarray(r, float)
            pla, plo = geo.utm_inverse(r[:, 0], r[:, 1], ZONE)
            planes.append([[round(float(a), 8), round(float(o), 8), round(float(h), 3)] for a, o, h in zip(pla, plo, r[:, 2])])
        top = max((max(p[2] for p in pl) for pl in planes), default=None)
        roof_h = b['roof_h'] if b['roof_h'] is not None else top
        eave = None
        if planes:   # eave: the lowest roof edge height of the sloped planes, else the flat roof height
            eave = min(min(p[2] for p in pl) for pl in planes)
        recs.append({'ref': b['ref'], 'ring': [[round(float(a), 8), round(float(o), 8)] for a, o in zip(la, lo)],
                     'ground_h': b['ground_h'], 'top_h': roof_h, 'eave_h': eave,
                     'height_m': round(roof_h - b['ground_h'], 2) if roof_h is not None else b['measured_height_m'],
                     'roof': ROOF.get(str(b['roof_code']), 'complex') if b['roof_code'] else None,
                     'roof_code': b['roof_code'], 'storeys': b['storeys'], 'function': b['function'],
                     'planes': planes, 'walls': b['walls']})
    return recs


def _area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


class LoD2(Provider):
    def __init__(self):
        super().__init__(
            name='by_lod2', title='Bavaria LoD2 3D building models (CityGML)', kind='buildings', scope='regional',
            licence='CC BY 4.0', attribution=ATTRIBUTION,
            url='https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=lod2', region='Bavaria, Germany',
            envelope=ENVELOPE, override=True,
            fuses='cadastral footprints and roof planes override the mesh building geometry (footprint, eave, ridge, '
                  'roof type and planes); mesh buildings it does not know and trees / cars inside its footprints are '
                  'flagged as conflicts')

    def fetch(self, bbox, folder: Path, refresh: bool = False) -> dict:
        out = folder / 'lod2.json'
        if cached(out, refresh):
            recs = json.loads(out.read_text())
            return {'covered': True, 'buildings': len(recs), 'cached': True}
        box = utm_box(bbox, pad_m=20)
        names = tiles(box, 2)
        names = sorted({f'{int(n.split("_")[0]) // 2 * 2}_{int(n.split("_")[1]) // 2 * 2}' for n in names})
        if len(names) > 4:
            raise ValueError(f'{len(names)} LoD2 tiles: the bbox is too large (cap 4 tiles)')
        raw, got, size = [], [], 0
        for name in names:
            url = f'{DOWNLOAD}/lod2/citygml/{name}.gml'
            if not net.exists(url):
                continue
            body = net.get(url, timeout=300)
            size += len(body)
            with tempfile.NamedTemporaryFile(suffix='.gml', dir=folder, delete=True) as f:   # parsed, then removed
                f.write(body)
                f.flush()
                del body
                raw += parse_citygml(Path(f.name), box)
            got.append(name)
        if not got:
            return {'covered': False, 'why': 'no LoD2 tile for the bbox (outside Bavaria)'}
        recs = to_records(raw)
        out.write_text(json.dumps(recs))
        return {'covered': True, 'tiles': got, 'downloaded_mb': round(size / 1e6, 1), 'buildings': len(recs)}

    def load(self, folder: Path):
        p = folder / 'lod2.json'
        return json.loads(p.read_text()) if p.is_file() else None
