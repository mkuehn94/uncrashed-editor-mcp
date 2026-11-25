"""OpenStreetMap behind the provider interface (fetch / parse / rasterise live in scan/osm.py)."""
from __future__ import annotations

import json
from pathlib import Path

from .. import osm
from . import Provider


class OSM(Provider):
    def __init__(self):
        super().__init__(
            name='osm', title='OpenStreetMap (Overpass / map API)', kind='map', scope='global',
            licence='ODbL 1.0', attribution=osm.ATTRIBUTION, url='https://www.openstreetmap.org/copyright',
            fuses='building footprints split merged roofs and claim canopy-covered roof cells; roads, parking, paths '
                  'and water become ground classes; tree nodes confirm crowns; street lines carry the street views')

    def coverage(self, bbox):
        why = osm.disabled()
        return (False, why) if why else (True, 'global data set')

    def fetch(self, bbox, folder: Path, refresh: bool = False) -> dict:
        try:
            answer = osm.fetch(folder / 'osm.json', bbox, refresh=refresh)
        except osm.OsmError as exc:
            raise ValueError(str(exc)) from None
        return {'elements': len(answer.get('elements', [])), 'source': answer.get('source'),
                'bbox': [round(v, 6) for v in bbox]}

    def load(self, folder: Path):
        p = folder / 'osm.json'
        return json.loads(p.read_text()) if p.is_file() else None
