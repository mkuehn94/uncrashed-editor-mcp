"""The Netherlands: 3D BAG (3dbag.nl, TU Delft 3D geoinformation / 3DGI), CC BY 4.0, an OGC API Features service
returning CityJSON per building (BAG footprint, LoD 2.2 roof surfaces, heights from AHN, NAP).

This module is the pattern for a regional building provider with a bbox API: the request carries only the bbox (in
RD New, EPSG:28992); the answer is cached as building records (lat/lon rings, heights in NAP).

RD <-> WGS84 uses the published approximation formulas (Schreutelkamp & Strang van Hees, 2001; about 0.25 m),
which the survey's footprint alignment (shift search) absorbs.
"""
from __future__ import annotations

import json
import urllib.parse
from pathlib import Path

import numpy as np

from . import Provider, cached, net

API = 'https://api.3dbag.nl/collections/pand/items'
ENVELOPE = (50.70, 3.20, 53.70, 7.30)
MAX_BUILDINGS = 3000
ROOF = {'horizontal': 'flat', 'multiple horizontal': 'flat', 'slanted': 'complex', 'no points': None}

_R = ((0, 1, 190094.945), (1, 1, -11832.228), (2, 1, -114.221), (0, 3, -32.391), (1, 0, -0.705), (3, 1, -2.340),
      (1, 3, -0.608), (0, 2, -0.008), (2, 3, 0.148))
_S = ((1, 0, 309056.544), (0, 2, 3638.893), (2, 0, 73.077), (1, 2, -157.984), (3, 0, 59.788), (0, 1, 0.433),
      (2, 2, -6.439), (1, 1, -0.032), (0, 4, 0.092), (1, 4, -0.054))
_K = ((0, 1, 3235.65389), (2, 0, -32.58297), (0, 2, -0.24750), (2, 1, -0.84978), (0, 3, -0.06550), (2, 2, -0.01709),
      (1, 0, -0.00738), (4, 0, 0.00530), (2, 3, -0.00039), (4, 1, 0.00033), (1, 1, -0.00012))
_L = ((1, 0, 5260.52916), (1, 1, 105.94684), (1, 2, 2.45656), (3, 0, -0.81885), (1, 3, 0.05594), (3, 1, -0.05607),
      (0, 1, 0.01199), (3, 2, -0.00256), (1, 4, 0.00128), (0, 2, 0.00022), (2, 0, -0.00022), (5, 0, 0.00026))
LAT0, LON0, X0, Y0 = 52.15517440, 5.38720621, 155000.0, 463000.0


def rd_forward(lat, lon):
    """WGS84 degrees -> RD New metres (x east, y north)."""
    dp, dl = 0.36 * (np.asarray(lat, float) - LAT0), 0.36 * (np.asarray(lon, float) - LON0)
    x = X0 + sum(c * dp ** p * dl ** q for p, q, c in _R)
    y = Y0 + sum(c * dp ** p * dl ** q for p, q, c in _S)
    return x, y


def rd_inverse(x, y):
    dx, dy = (np.asarray(x, float) - X0) * 1e-5, (np.asarray(y, float) - Y0) * 1e-5
    lat = LAT0 + sum(c * dx ** p * dy ** q for p, q, c in _K) / 3600
    lon = LON0 + sum(c * dx ** p * dy ** q for p, q, c in _L) / 3600
    return lat, lon


def rd_box(bbox) -> tuple[float, float, float, float]:
    s, w, n, e = bbox
    x, y = rd_forward([s, s, n, n], [w, e, w, e])
    return float(x.min()), float(y.min()), float(x.max()), float(y.max())


def records(features: list[dict], transform: dict) -> list[dict]:
    """CityJSON features of the 3D BAG API -> building records."""
    sc = np.asarray(transform.get('scale', [1, 1, 1]), float)
    tr = np.asarray(transform.get('translate', [0, 0, 0]), float)
    out = []
    for f in features:
        verts = np.asarray(f.get('vertices') or [], float).reshape(-1, 3) * sc + tr
        objs = f.get('CityObjects') or {}
        parent = objs.get(f.get('id')) or next((o for o in objs.values() if o.get('type') == 'Building'), None)
        if parent is None or not len(verts):
            continue
        at = parent.get('attributes') or {}
        ring = None
        for geom in parent.get('geometry') or []:
            if str(geom.get('lod')) == '0' and geom.get('boundaries'):
                ring = verts[geom['boundaries'][0][0]]
        if ring is None:
            continue
        planes = []
        for child in parent.get('children') or []:
            for geom in (objs.get(child) or {}).get('geometry') or []:
                if str(geom.get('lod')) != '2.2' or geom.get('type') != 'Solid':
                    continue
                sem = geom.get('semantics') or {}
                surfaces, values = sem.get('surfaces') or [], (sem.get('values') or [[]])[0]
                for face, v in zip(geom['boundaries'][0], values):
                    if v is not None and surfaces[v].get('type') == 'RoofSurface':
                        planes.append(verts[face[0]])
        la, lo = rd_inverse(ring[:, 0], ring[:, 1])
        ground = at.get('b3_h_maaiveld')
        top = at.get('b3_h_dak_max')
        pls = []
        for p in planes:
            pla, plo = rd_inverse(p[:, 0], p[:, 1])
            pls.append([[round(float(a), 8), round(float(b), 8), round(float(h), 3)] for a, b, h in zip(pla, plo, p[:, 2])])
        roof = ROOF.get(at.get('b3_dak_type'), 'complex')
        out.append({'ref': f.get('id'), 'ring': [[round(float(a), 8), round(float(b), 8)] for a, b in zip(la, lo)],
                    'ground_h': ground, 'top_h': top, 'eave_h': at.get('b3_h_dak_min'),
                    'height_m': round(top - ground, 2) if top is not None and ground is not None else None,
                    'roof': roof, 'storeys': at.get('b3_bouwlagen'), 'function': None, 'planes': pls,
                    'year': at.get('oorspronkelijkbouwjaar')})
    return out


class Bag3D(Provider):
    def __init__(self):
        super().__init__(
            name='nl_3dbag', title='3D BAG (Netherlands, LoD 2.2 buildings)', kind='buildings', scope='regional',
            licence='CC BY 4.0', attribution='3DBAG by tudelft3d and 3DGI (CC BY 4.0), based on BAG and AHN',
            url='https://3dbag.nl', region='the Netherlands', envelope=ENVELOPE, override=True,
            fuses='BAG footprints and LoD 2.2 roof planes override the mesh building geometry, as by_lod2')

    def fetch(self, bbox, folder: Path, refresh: bool = False) -> dict:
        out = folder / '3dbag.json'
        if cached(out, refresh):
            return {'covered': True, 'buildings': len(json.loads(out.read_text())), 'cached': True}
        x0, y0, x1, y1 = rd_box(bbox)
        url = f'{API}?' + urllib.parse.urlencode({'bbox': f'{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}', 'limit': 100})
        recs, pages = [], 0
        while url and len(recs) < MAX_BUILDINGS:
            page = json.loads(net.get(url, timeout=60))
            pages += 1
            transform = (page.get('metadata') or {}).get('transform') or {}
            recs += records(page.get('features') or [], transform)
            url = next((ln['href'] for ln in page.get('links') or [] if ln.get('rel') == 'next'), None)
        if not recs:
            return {'covered': False, 'why': 'the 3D BAG has no building in the bbox (outside the Netherlands?)'}
        out.write_text(json.dumps(recs))
        return {'covered': True, 'buildings': len(recs), 'pages': pages, 'truncated': bool(url)}

    def load(self, folder: Path):
        p = folder / '3dbag.json'
        return json.loads(p.read_text()) if p.is_file() else None
