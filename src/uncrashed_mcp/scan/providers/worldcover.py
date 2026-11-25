"""ESA WorldCover 2021 (v200): global 10 m land cover, cloud-optimised GeoTIFFs on AWS Open Data (3 x 3 degree tiles).

Fused as a cross-check of the survey's surface classes (share of tree cover / built-up / grass / water in the
survey box against the scan classes) and as per-object evidence for trees and water (10 m cells: a street tree
often sits in a built-up cell, so built-up is never counted against a tree).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import Provider, Raster, cached, geotiff, net
from .copernicus import window

BASE = 'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map'
CLASSES = {10: 'tree_cover', 20: 'shrubland', 30: 'grassland', 40: 'cropland', 50: 'built_up', 60: 'bare',
           70: 'snow_ice', 80: 'water', 90: 'wetland', 95: 'mangroves', 100: 'moss_lichen'}


def tile_name(lat: float, lon: float) -> str:
    la, lo = int(math.floor(lat / 3) * 3), int(math.floor(lon / 3) * 3)
    return f"ESA_WorldCover_10m_2021_v200_{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}_Map"


class WorldCover(Provider):
    def __init__(self):
        super().__init__(
            name='worldcover', title='ESA WorldCover 2021 (10 m land cover)', kind='landcover', scope='global',
            licence='CC BY 4.0',
            attribution='(c) ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) '
                        'processed by ESA WorldCover consortium',
            url='https://esa-worldcover.org/', envelope=(-60.0, -180.0, 84.0, 180.0), region='land between 60 S and 84 N',
            fuses='cross-check of surface class shares; tree / water evidence per object (10 m: never against a '
                  'street tree in a built-up cell)')

    def fetch(self, bbox, folder: Path, refresh: bool = False) -> dict:
        out = folder / 'worldcover.npz'
        if cached(out, refresh):
            return {'covered': True, 'cached': True}
        s, w, n, e = bbox
        names = {tile_name(la, lo) for la in (s, n) for lo in (w, e)}
        if len(names) != 1:
            raise ValueError('the bbox crosses a 3 degree WorldCover tile edge: mosaics are not supported yet')
        name = names.pop()
        url = f'{BASE}/{name}.tif'
        if not net.exists(url):
            return {'covered': False, 'why': f'no WorldCover tile {name} (open sea)'}
        t = geotiff.open_url(url)
        arr, tr = window(t, bbox)
        Raster(arr.astype(np.uint8), 4326, tr, 'landcover', 0).save(out)
        vals, cnt = np.unique(arr, return_counts=True)
        share = {CLASSES.get(int(v), str(int(v))): round(float(c) / max(1, arr.size), 3) for v, c in zip(vals, cnt) if v}
        return {'covered': True, 'tile': name, 'cells': list(arr.shape), 'share': share}

    def load(self, folder: Path):
        p = folder / 'worldcover.npz'
        return Raster.read(p) if p.is_file() else None
