"""Location-agnostic data providers for scan_survey.

A provider is one open data set that can answer a bounding-box question: OpenStreetMap, Overture Maps buildings,
the Copernicus GLO-30 surface model, ESA WorldCover land cover, and regional high-resolution sets (one small
module per region: Bavarian DGM1 / LoD2, Dutch 3D BAG, ...). The survey picks providers by coverage of its bbox:

    coverage(bbox)          offline test of the provider's published extent -> (covers, why)
    fetch(bbox, folder)     downloads what the bbox needs (only the bbox, tile names or byte ranges of public files
                            leave the machine) into folder (the survey's providers/<name>/), returns a summary;
                            a regional provider confirms coverage here (no tile / empty answer = not covered)
    load(folder)            the cached data: a Raster (terrain, surface, land cover) or a list of Building records
    licence / attribution   recorded in the survey and shown by scan_sources
    fuses                   one sentence: what the data does in the survey (see fuse.py)

Coordinates: Raster has its own CRS (EPSG code and transform), Building records carry lat/lon rings and heights
in the provider's vertical datum; fuse.py maps both onto the scan grid and fits the vertical offset from data.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import geo, net

SUMMARY = 'summary.json'


@dataclass
class Raster:
    """A single-band grid. transform = (x of column 0 edge, y of row 0 edge, pixel w, pixel h); rows run south."""
    array: np.ndarray
    epsg: int
    transform: tuple[float, float, float, float]
    kind: str                       # dtm, dsm, landcover
    nodata: float | None = None

    def sample(self, lat, lon, nearest: bool = False) -> np.ndarray:
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        if self.epsg == 4326:
            x, y = lon, lat
        elif 32601 <= self.epsg <= 32660 or 25828 <= self.epsg <= 25838:
            zone = self.epsg - 32600 if self.epsg > 30000 else self.epsg - 25800
            x, y = geo.utm_forward(lat, lon, zone)
        elif self.epsg == 28992:
            from . import netherlands
            x, y = netherlands.rd_forward(lat, lon)
        else:
            raise ValueError(f'EPSG:{self.epsg} is not supported by the provider rasters')
        x0, y0, sx, sy = self.transform
        c = (x - x0) / sx - 0.5
        r = (y0 - y) / sy - 0.5
        a = self.array.astype(np.float64)
        if self.nodata is not None:
            a = np.where(a == self.nodata, np.nan, a)
        H, W = a.shape
        if nearest:
            ci, ri = np.rint(c).astype(int), np.rint(r).astype(int)
            ok = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
            out = np.full(c.shape, np.nan)
            out[ok] = a[ri[ok], ci[ok]]
            return out
        c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
        fc, fr = c - c0, r - r0
        ok = (c0 >= 0) & (c0 + 1 < W) & (r0 >= 0) & (r0 + 1 < H)
        out = np.full(c.shape, np.nan)
        c0, r0, fc, fr = c0[ok], r0[ok], fc[ok], fr[ok]
        out[ok] = ((a[r0, c0] * (1 - fc) + a[r0, c0 + 1] * fc) * (1 - fr)
                   + (a[r0 + 1, c0] * (1 - fc) + a[r0 + 1, c0 + 1] * fc) * fr)
        return out

    def save(self, path: Path) -> None:
        tmp = path.with_name(path.name + '.tmp.npz')
        np.savez_compressed(tmp, array=self.array, epsg=self.epsg, transform=np.array(self.transform),
                            kind=self.kind, nodata=np.nan if self.nodata is None else self.nodata)
        tmp.replace(path)

    @classmethod
    def read(cls, path: Path) -> 'Raster':
        with np.load(path) as z:
            nd = float(z['nodata'])
            return cls(z['array'], int(z['epsg']), tuple(float(v) for v in z['transform']), str(z['kind']),
                       None if np.isnan(nd) else nd)


@dataclass
class Provider:
    name: str
    title: str
    kind: str                      # map, buildings, terrain, surface, landcover
    scope: str                     # global or regional
    licence: str
    attribution: str
    url: str
    fuses: str
    region: str = 'world'
    envelope: tuple[float, float, float, float] | None = None   # (south, west, north, east) degrees
    outline: list[tuple[float, float]] | None = None             # (lat, lon) polygon refining the envelope
    override: bool = False         # buildings: its footprints and roofs replace the mesh geometry
    notes: list[str] = field(default_factory=list)

    def coverage(self, bbox) -> tuple[bool, str]:
        if self.envelope is None:
            return True, 'global data set'
        if not geo.intersects(bbox, self.envelope):
            return False, f'outside {self.region}'
        if self.outline:
            corners = [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[1]), (bbox[2], bbox[3]),
                       ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)]
            if not any(geo.inside_polygon(la, lo, self.outline) for la, lo in corners):
                return False, f'outside {self.region}'
        return True, f'inside the published extent of {self.region} (tiles confirmed on fetch)'

    def fetch(self, bbox, folder: Path, refresh: bool = False) -> dict:   # pragma: no cover - interface
        raise NotImplementedError

    def load(self, folder: Path):   # pragma: no cover - interface
        raise NotImplementedError

    def info(self) -> dict:
        return {'name': self.name, 'title': self.title, 'kind': self.kind, 'scope': self.scope, 'region': self.region,
                'licence': self.licence, 'attribution': self.attribution, 'fuses': self.fuses, 'url': self.url}


def registry() -> list[Provider]:
    from . import bavaria, copernicus, netherlands, osm_provider, overture, worldcover
    return [osm_provider.OSM(), overture.Overture(), copernicus.Glo30(), worldcover.WorldCover(),
            bavaria.DGM1(), bavaria.LoD2(), netherlands.Bag3D()]


# Checked and not used: the reason is recorded in every survey's providers report.
DECLINED = {
    'mapillary': ('Mapillary street-level images and map features: the API needs a personal access token, the '
                  'images are CC BY-SA 4.0 (share-alike would bind derived detections) and commercial use falls under '
                  'section 12 of the Mapillary terms; whether detections derived by our own models may be used '
                  'without share-alike is not settled in the terms (checked 2026-11-25). Skipped: the survey renders '
                  'its own street views from the scan instead.'),
}


def by_name(name: str) -> Provider:
    for p in registry():
        if p.name == name:
            return p
    raise ValueError(f"unknown provider '{name}'; known: {', '.join(p.name for p in registry())}")


def choose(want: str) -> tuple[list[str] | None, bool]:
    """want: 'auto' (every provider covering the bbox), 'none', or a comma list of names (optionally 'auto,-name'
    to drop one). Returns (names or None for auto, auto flag)."""
    w = (want or 'auto').strip().lower()
    if w in ('auto', ''):
        return None, True
    if w in ('none', 'off', '0'):
        return [], False
    names = [x.strip() for x in w.split(',') if x.strip()]
    known = {p.name for p in registry()}
    drop = {x[1:] for x in names if x.startswith('-')}
    add = [x for x in names if not x.startswith('-') and x != 'auto']
    for x in add + sorted(drop):
        if x not in known:
            raise ValueError(f"unknown provider '{x}'; known: {', '.join(sorted(known))} (or auto / none)")
    if 'auto' in names:
        return [p.name for p in registry() if p.name not in drop], True
    return add, False


def folder(home: Path, name: str) -> Path:
    return home / 'providers' / name


def gather(home: Path, bbox, want: str = 'auto', refresh: bool = False, osm: bool = True,
           osm_required: bool = False) -> dict:
    """Coverage test and fetch for every chosen provider. Returns {name: record} with used, covers, why, licence,
    attribution, summary and fetch_s. A failed download is recorded (not raised) unless the provider was named
    (or OSM was asked for explicitly: osm_required). OSM follows the osm flag, not want."""
    names, auto = choose(want)
    out = {}
    for p in registry():
        if p.name == 'osm':
            chosen = osm
        else:
            chosen = auto if names is None else p.name in names
        rec = {**p.info(), 'used': False}
        if not chosen:
            rec['why'] = 'not chosen (--providers)'
            out[p.name] = rec
            continue
        ok, why = p.coverage(bbox)
        rec['covers'] = ok
        if not ok:
            rec['why'] = why
            out[p.name] = rec
            continue
        d = folder(home, p.name)
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            summary = p.fetch(bbox, d, refresh=refresh)
        except Exception as exc:   # noqa: BLE001  any download or parse failure is recorded, not raised (auto mode)
            msg = str(exc) if isinstance(exc, (net.NetError, ValueError, OSError)) else f'{type(exc).__name__}: {exc}'
            rec['why'] = f'fetch failed after {time.time() - t0:.1f} s: {" ".join(msg.split())[:200]}'
            if (p.name == 'osm' and osm_required) or (p.name != 'osm' and not auto and names):
                raise ValueError(f'provider {p.name}: {rec["why"]}') from None
            out[p.name] = rec
            continue
        rec['fetch_s'] = round(time.time() - t0, 1)
        rec['summary'] = summary
        if summary.get('covered') is False:
            rec['covers'] = False
            rec['why'] = summary.get('why', 'no data for the bbox')
        else:
            rec['used'] = True
            rec['why'] = why
        (d / SUMMARY).write_text(json.dumps(summary))
        out[p.name] = rec
    for name, why in DECLINED.items():
        out[name] = {'name': name, 'used': False, 'declined': True, 'why': why}
    return out


def loaded(home: Path, record: dict) -> dict:
    """{name: (provider, data)} for the used providers whose cache is on disk (no network)."""
    out = {}
    for name, rec in (record or {}).items():
        if not rec.get('used'):
            continue
        try:
            p = by_name(name)
        except ValueError:
            continue
        d = folder(home, name)
        try:
            data = p.load(d)
        except Exception:   # noqa: BLE001  a damaged cache is skipped like a missing one
            continue
        if data is not None:
            out[name] = (p, data)
    return out


def cached(path: Path, refresh: bool) -> bool:
    return path.is_file() and not refresh


from . import fuse  # noqa: E402,F401  (ops reaches it as providers.fuse)
