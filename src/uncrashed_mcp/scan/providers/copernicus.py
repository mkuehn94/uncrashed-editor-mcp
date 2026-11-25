"""Copernicus DEM GLO-30: global 1 arc-second (about 30 m) surface model, cloud-optimised GeoTIFFs on AWS Open Data.

It is a DSM (radar, X-band: roofs and canopy included), heights EGM2008. At 30 m it cannot replace the scan's
ground model; it cross-checks the scan's height datum and coarse relief (fuse.py: median offset and spread of the
scan surface averaged per 30 m cell against GLO-30).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import Provider, Raster, cached, geotiff, net

BASE = 'https://copernicus-dem-30m.s3.amazonaws.com'


def tile_name(lat: int, lon: int) -> str:
    ns, ew = ('N' if lat >= 0 else 'S'), ('E' if lon >= 0 else 'W')
    return f'Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM'


def window(t: geotiff.Tiff, bbox, pad: int = 2) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Pixels of a lat/lon GeoTIFF covering bbox (+pad pixels) and their transform."""
    x0, y0, sx, sy = t.transform()
    s, w, n, e = bbox
    c0, c1 = int(math.floor((w - x0) / sx)) - pad, int(math.ceil((e - x0) / sx)) + pad
    r0, r1 = int(math.floor((y0 - n) / sy)) - pad, int(math.ceil((y0 - s) / sy)) + pad
    c0, r0 = max(0, c0), max(0, r0)
    c1, r1 = min(t.width, c1), min(t.height, r1)
    return t.window(c0, r0, c1, r1), (x0 + c0 * sx, y0 - r0 * sy, sx, sy)


class Glo30(Provider):
    def __init__(self):
        super().__init__(
            name='glo30', title='Copernicus DEM GLO-30 (DSM, 30 m)', kind='surface', scope='global',
            licence='Copernicus DEM licence (free use with attribution)',
            attribution='Produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and '
                        'Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights '
                        'reserved',
            url='https://registry.opendata.aws/copernicus-dem/',
            fuses='cross-check only: median height offset (datum) and spread of the scan surface per 30 m cell; '
                  'never replaces the ground model (it is a surface model)')

    def fetch(self, bbox, folder: Path, refresh: bool = False) -> dict:
        out = folder / 'glo30.npz'
        if cached(out, refresh):
            r = Raster.read(out)
            return {'covered': True, 'cells': list(r.array.shape), 'cached': True}
        s, w, n, e = bbox
        tiles = [(la, lo) for la in range(math.floor(s), math.floor(n) + 1) for lo in range(math.floor(w), math.floor(e) + 1)]
        if len(tiles) != 1:
            raise ValueError('the bbox crosses a 1 degree tile edge: GLO-30 mosaics are not supported yet')
        name = tile_name(*tiles[0])
        url = f'{BASE}/{name}/{name}.tif'
        if not net.exists(url):
            return {'covered': False, 'why': f'no GLO-30 tile {name} (open sea or excluded area)'}
        t = geotiff.open_url(url)
        arr, tr = window(t, bbox)
        Raster(arr.astype(np.float32), 4326, tr, 'dsm', t.nodata).save(out)
        return {'covered': True, 'tile': name, 'cells': list(arr.shape), 'res_deg': tr[2]}

    def load(self, folder: Path):
        p = folder / 'glo30.npz'
        return Raster.read(p) if p.is_file() else None
