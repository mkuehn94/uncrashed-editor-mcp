"""The scan_* operations: survey a textured OBJ, then answer spatial questions about it.

Called lazily from api.py (numpy is imported only when a scan command runs).
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np

from .. import reference
from . import facade, grid, native, osm, store, survey, vision
from . import providers as prov
from .store import Frame

MAX_CELLS = 36_000_000      # analysis grid cap (6000 x 6000)
VIEW_ROWS = 60              # objects listed with a scan_view image
DEFAULT_RADIUS_M = 250.0
LIMITATIONS = [
    'Rule-based geometry survey of a photogrammetry mesh: classes and confidences are heuristics, not detections.',
    'The DSM sees only the highest surface: passages under bridges, arches or canopies read as solid.',
    'Footprints follow roof outlines (eaves), not walls; heights are above the filtered ground model.',
]
LIMITATIONS_VISION = [
    'Geometry rules plus learned detectors (YOLO11-OBB, Grounding DINO, SAM 2) on the scan texture: conf combines '
    'both, models lists the detector scores; neither sees vehicles under roofs or dense crowns.',
    LIMITATIONS[1], LIMITATIONS[2],
    'Facade windows / doors are detections on blurred photogrammetry texture: check walls with scan_facade.',
]


# ---------------------------------------------------------------------------- helpers


def _r(v, n=2):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), n)


def _xy(points, what='points') -> np.ndarray:
    try:
        p = np.asarray(points, float)
    except (TypeError, ValueError):
        raise ValueError(f'{what}: give x,y numbers') from None
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if p.ndim != 2 or p.shape[1] < 2 or not np.isfinite(p).all():
        raise ValueError(f'{what}: give finite x,y points')
    return p


def _grid(meta: dict) -> survey.Grid:
    g = meta['grid']
    return survey.Grid(g['x0'], g['y0'], g['res'], g['w'], g['h'])


def _box4(box, what='box') -> tuple[float, float, float, float]:
    if box is None or len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError(f'{what} needs four finite numbers x0,y0,x1,y1 (metres)')
    x0, y0, x1, y1 = (float(v) for v in box)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


class Rasters:
    """Memory-mapped rasters of a survey in /dev/shm (rebuilt when missing, incomplete or unreadable)."""

    def __init__(self, meta: dict):
        self.meta, self.k = meta, meta['key']
        self.rebuilt = False
        self._d = store.ram(self.k)
        self.g = _grid(meta)
        if not store.rasters_ok(self.k):
            self._rebuild()

    def _rebuild(self):
        with store.build_lock(self.k):
            if not store.rasters_ok(self.k):   # another process may have finished it while we waited
                store.unmark(self.k)
                _build(self.meta, rebuild=True)
                self.rebuilt = True

    def __getattr__(self, name):
        if name not in store.RASTERS:
            raise AttributeError(name)
        try:
            value = np.load(self._d / f'{name}.npy', mmap_mode='r')
        except (OSError, ValueError, EOFError):   # truncated / removed since the check: rebuild once
            if self.__dict__.get('_retried'):
                raise
            self._retried = True
            store.unmark(self.k)
            self._rebuild()
            value = np.load(self._d / f'{name}.npy', mmap_mode='r')
        setattr(self, name, value)
        return value

    def cell(self, e, s):
        c, r = self.g.to_cell(e, s)
        c, r = np.rint(c).astype(int), np.rint(r).astype(int)
        inside = (c >= 0) & (c < self.g.w) & (r >= 0) & (r < self.g.h)
        return np.clip(r, 0, self.g.h - 1), np.clip(c, 0, self.g.w - 1), inside

    def ground(self, e, s):
        """Bilinear DTM."""
        g = self.g
        c, r = g.to_cell(e, s)
        c = np.clip(c, 0, g.w - 1.001)
        r = np.clip(r, 0, g.h - 1.001)
        c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
        fc, fr = c - c0, r - r0
        d = self.dtm
        return ((d[r0, c0] * (1 - fc) + d[r0, c0 + 1] * fc) * (1 - fr) + (d[r0 + 1, c0] * (1 - fc) + d[r0 + 1, c0 + 1] * fc) * fr)


def _context(survey_key: str, frame: str):
    meta = store.resolve(survey_key)
    return meta, Frame(meta, frame), Rasters(meta)


def _head(meta: dict, fr: Frame, ras: Rasters | None = None) -> dict:
    out = {'survey': meta['key'], 'frame': fr.name}
    if ras is not None and ras.rebuilt:
        out['note'] = 'rasters were rebuilt into RAM from the OBJ index'
    bad = store.read_json(store.home(meta['key']) / 'rebuild-mismatch.json')
    if isinstance(bad, dict):
        out['warning'] = (f"the RAM rasters were rebuilt with {bad.get('rebuilt')} objects, the survey stored "
                          f"{bad.get('stored')}: object ids at points may be wrong; run scan_survey --refresh")
    return out


# ---------------------------------------------------------------------------- survey


def _region(obj: Path, cache: Path, axes: str, center, radius_m: float) -> tuple[list[float], float]:
    center = [float(v) for v in (center or (0.0, 0.0))]
    if len(center) != 2 or not all(math.isfinite(v) for v in center):
        raise ValueError('center is east,south metres in the export')
    if radius_m and radius_m > 0:
        return center, float(radius_m)
    area = reference.metadata(obj).get('area') or {}
    try:
        r = float(area.get('radius'))
        if r > 0 and center == [0.0, 0.0]:
            return center, r
    except (TypeError, ValueError):
        pass
    _tex, meshes = reference.read_index(cache)
    amap = reference._axis_map(axes)
    lo = np.array([[reference._world(amap, m['lo'])[k] for k in (0, 1)] for m in meshes])
    hi = np.array([[reference._world(amap, m['hi'])[k] for k in (0, 1)] for m in meshes])
    a, b = np.minimum(lo, hi), np.maximum(lo, hi)
    reach = max(np.abs(a - center).max(), np.abs(b - center).max())
    return center, float(min(reach, DEFAULT_RADIUS_M * 4))


def _build(meta: dict, rebuild: bool = False) -> dict:
    """Rasterise + survey; writes RAM rasters (and objects/meta unless rebuild). Returns timing / stats dict."""
    store.check_shm()
    t_all = time.time()
    obj = Path(meta['obj'])
    if not obj.is_file():
        raise ValueError(f'no such OBJ: {obj} (the survey needs its source to rebuild the rasters)')
    why = store.stale(meta)
    if why:   # rasters of the new mesh would not match the stored objects
        raise store.NoSurvey(f"{why} since survey {meta['key']}: run scan_survey {obj} again")
    t0 = time.time()
    cache = reference.ensure_index(obj)
    t_index = time.time() - t0
    amap = reference._axis_map(meta['axes'])
    g = _grid(meta)
    t0 = time.time()
    lib_stats = {}
    out = native.raster(cache, amap, g.x0, g.y0, g.res, g.w, g.h, normals=True, low=True, rgb=True)
    lib_stats['analysis'] = out['stats']
    o = meta['ortho']
    orth = native.raster(cache, amap, g.x0, g.y0, o['res'], o['w'], o['h'], normals=False, low=False, rgb=True)
    lib_stats['ortho'] = orth['stats']
    t_raster = time.time() - t0
    feats = None
    k = meta['key']
    if meta.get('osm', {}).get('used'):
        answer = osm_answer(meta)
        geo = meta['georef']
        if isinstance(answer, dict):
            feats = osm.parse(answer, geo['lat'], geo['lon'])
    t0 = time.time()
    loaded = {n: v for n, v in prov.loaded(store.home(k), meta.get('providers')).items() if n != 'osm'}
    terrain = prov.fuse.terrain_prior(g, meta['georef'], loaded) if loaded and meta.get('georef') else (None, None)
    t_prior = time.time() - t0
    t0 = time.time()
    sv = survey.Survey(g, out['dsm'], out['nrm'], out['rgb'], feats, terrain=terrain).run()
    t_survey = time.time() - t0
    t0 = time.time()
    fused = prov.fuse.fuse_survey(sv, meta.get('georef'), loaded) if meta.get('georef') else {}
    t_fuse = time.time() - t0
    if rebuild:  # objects added by the vision stage live only in objects.json: put them back into the id raster
        vision.stamp(sv.objid, store.read_json(store.home(k) / 'objects.json', []), meta)
    arrays = {'dsm': out['dsm'], 'low': out['low'], 'dtm': sv.dtm, 'cls': sv.cls, 'objid': sv.objid,
              'nrm': out['nrm'].astype(np.float16), 'rgb': out['rgb'], 'ortho': orth['rgb']}
    with store.build_lock(k):
        store.unmark(k)
        for name, arr in arrays.items():
            store.save_array(k, name, arr)
        store.mark_complete(k, meta)
    result = {'objects': sv.objects, 'summary': sv.summary(), 'osm': sv.osm_info, 'raster_stats': lib_stats,
              'fusion': fused,
              'timings': {'index_s': round(t_index, 1), 'raster_s': round(t_raster, 1), 'survey_s': round(t_survey, 1),
                          **{f'{a}_s': b for a, b in sv.timings.items()},
                          **({'providers_prior_s': round(t_prior, 2), 'providers_fuse_s': round(t_fuse, 2)} if loaded else {}),
                          'total_s': round(time.time() - t_all, 1)}}
    if rebuild:
        old = store.read_json(store.home(k) / 'objects-geometry.json', None)
        if old is None:
            old = store.read_json(store.home(k) / 'objects.json', [])
        if len(old) != len(sv.objects):
            store.write_json(store.home(k) / 'rebuild-mismatch.json',
                             {'stored': len(old), 'rebuilt': len(sv.objects), 'at': store.now()})
        else:
            (store.home(k) / 'rebuild-mismatch.json').unlink(missing_ok=True)
    else:
        (store.home(k) / 'rebuild-mismatch.json').unlink(missing_ok=True)
    return result


def scan_survey(obj: str, center=None, radius_m: float = 0.0, res_m: float = 0.25, axes: str = '',
                use_osm: bool | None = None, use_vision: bool | None = None, facades: bool = True,
                refresh: bool = False, providers: str = 'auto', street_views: bool | None = None) -> dict:
    path = Path(obj).expanduser()
    if not path.is_file():
        raise ValueError(f'no such OBJ: {path}')
    path = path.resolve()
    if not 0.1 <= float(res_m) <= 2.0:
        raise ValueError('res_m must be 0.1..2 metres')
    if radius_m < 0 or not math.isfinite(radius_m):
        raise ValueError('radius_m must be >= 0 (0 = the export area)')
    chosen, auto = prov.choose(providers)   # validates the names before any work
    want_prov = 'auto' if auto and chosen is None else ('none' if not chosen else ','.join(sorted(chosen)))
    store.check_shm()
    axes = reference.source_axes(path, axes)
    t0 = time.time()
    cache = reference.ensure_index(path)
    t_index = round(time.time() - t0, 1)
    center, radius = _region(path, cache, axes, center, radius_m)
    res = float(res_m)
    n = int(math.ceil(2 * radius / res))
    if n * n > MAX_CELLS:
        raise ValueError(f'region of {2 * radius:.0f} m at {res:g} m is {n}x{n} cells (cap {MAX_CELLS:,}): '
                         'pass a smaller radius_m or a larger res_m')
    md = reference.metadata(path)
    origin = md.get('origin') or {}
    try:
        georef = {'lat': float(origin['lat']), 'lon': float(origin['lon'])}
    except (KeyError, TypeError, ValueError):
        georef = None
    skipped = []
    want_osm = use_osm is not False
    if not want_osm:
        skipped.append('osm: disabled (--no-osm)')
    elif osm.disabled():
        skipped.append(f'osm: {osm.disabled()}')
        want_osm = False
    elif not georef:
        skipped.append('osm: no metadata.json origin lat/lon next to the OBJ')
        want_osm = False
    params = {'v': survey.VERSION, 'axes': axes, 'osm': want_osm}
    if want_prov != 'none':
        params['providers'] = want_prov
    k = store.key(path, center, radius, res, params)
    home = store.home(k)
    old = store.read_json(home / 'meta.json')
    if isinstance(old, dict) and not refresh and (home / 'objects.json').is_file():
        store.set_current(k)
        Rasters(old)   # rebuilds missing, incomplete or truncated RAM rasters
        added = _vision_stage(old, use_vision, facades, again=False)
        added = _street_stage(old, use_vision, street_views, again=False) or added
        if added:
            store.write_json(home / 'meta.json', old)
        return _survey_reply(old, reused=True)
    for stale in store.surveys():  # surveys of this OBJ made by an older survey version
        if stale['obj'] == str(path) and stale.get('version') != survey.VERSION:
            store.remove(stale['key'])
    home.mkdir(parents=True, exist_ok=True)
    x0, y0 = center[0] - radius, center[1] - radius
    meta = {'key': k, 'version': survey.VERSION, 'obj': str(path), 'obj_size': path.stat().st_size,
            'obj_mtime_ns': path.stat().st_mtime_ns, 'axes': axes, 'center': center, 'radius_m': radius,
            'grid': {'x0': x0, 'y0': y0, 'res': res, 'w': n, 'h': n},
            'ortho': {'res': res / 2, 'w': 2 * n, 'h': 2 * n}, 'georef': georef, 'created': store.now(),
            'osm': {'used': False}, 'providers': {}, 'skipped': skipped}
    if georef and (want_osm or want_prov != 'none'):
        box = osm.bbox(x0, y0, x0 + n * res, y0 + n * res, georef['lat'], georef['lon'])
        t = time.time()
        recs = prov.gather(home, box, want=want_prov, refresh=refresh, osm=want_osm, osm_required=bool(use_osm))
        meta['providers'] = recs
        meta['providers_bbox'] = [round(v, 6) for v in box]
        o = recs.get('osm') or {}
        if o.get('used'):
            feats = osm.parse(prov.by_name('osm').load(prov.folder(home, 'osm')), georef['lat'], georef['lon'])
            meta['osm'] = {'used': True, 'bbox': [round(v, 6) for v in box], 'features': osm.counts(feats),
                           'attribution': osm.ATTRIBUTION, 'source': (o.get('summary') or {}).get('source'),
                           'fetch_s': o.get('fetch_s')}
        elif want_osm and o.get('why'):
            meta['skipped'].append(f"osm: {o['why'].replace('fetch failed after', 'no answer after')} "
                                   '(scan_survey --refresh retries)')
        for name, r in recs.items():
            if name != 'osm' and r.get('covers') and not r.get('used') and r.get('why'):
                meta['skipped'].append(f"provider {name}: {r['why']}")
        meta['providers_fetch_s'] = round(time.time() - t, 1)
    elif want_prov != 'none' and not georef:
        skipped.append('providers: no metadata.json origin lat/lon next to the OBJ')
    old_osm = prov.folder(home, 'osm') / 'osm.json'
    if old_osm.is_file() and not meta['osm']['used']:
        old_osm.unlink()
    result = _build(meta)
    result['timings']['index_s'] = t_index
    if meta.get('providers_fetch_s') is not None:
        result['timings']['providers_fetch_s'] = meta['providers_fetch_s']
    meta.update(counts=result['summary']['counts'], class_share=result['summary']['class_share'],
                survey_stats=result['summary'].get('stats') or {},
                timings=result['timings'], raster_stats=result['raster_stats'], vision={'ran': False})
    if result['osm']:
        meta['osm'].update(result['osm'])
        n_bld = sum(1 for o in result['objects'] if o['cls'] == 'building')
        meta['osm']['buildings_split_or_tagged'] = sum(1 for o in result['objects']
                                                       if o['cls'] == 'building' and o['src'] == 'mesh+osm')
        meta['osm']['buildings_mesh_only'] = n_bld - meta['osm']['buildings_split_or_tagged']
    if result.get('fusion'):
        meta['fusion'] = result['fusion']
    store.write_json(home / 'objects-geometry.json', result['objects'])
    store.write_json(home / 'objects.json', result['objects'])
    (home / 'facades.json').unlink(missing_ok=True)
    _vision_stage(meta, use_vision, facades, again=True)
    _street_stage(meta, use_vision, street_views, again=True)
    store.write_json(home / 'meta.json', meta)
    store.set_current(k)
    return _survey_reply(meta, reused=False)


def osm_answer(meta: dict):
    """The survey's cached OSM answer (providers/osm/osm.json; osm.json in surveys made before providers)."""
    home = store.home(meta['key'])
    for p in (prov.folder(home, 'osm') / 'osm.json', home / 'osm.json'):
        if p.is_file():
            return store.read_json(p)
    return None


def _osm_rings(meta: dict) -> tuple[dict, tuple[float, float]]:
    if not meta.get('osm', {}).get('used'):
        return {}, (0.0, 0.0)
    answer = osm_answer(meta)
    if not isinstance(answer, dict):
        return {}, (0.0, 0.0)
    feats = osm.parse(answer, meta['georef']['lat'], meta['georef']['lon'])
    shift = tuple(meta['osm'].get('shift_m') or (0.0, 0.0))
    return {b['osm']: b['rings'][0] for b in feats['buildings'] if b['rings']}, shift


def _vision_stage(meta: dict, use_vision: bool | None, facades: bool, again: bool) -> bool:
    """Run the vision stage when wanted and not done yet (or again). Updates meta (counts, vision, skipped),
    objects.json, facades.json and the RAM id raster. Returns True when it changed anything."""
    k = meta['key']
    home = store.home(k)
    skipped = [x for x in meta.get('skipped', []) if not x.startswith(('vision:', 'facades:'))]
    meta['skipped'] = skipped
    done = meta.get('vision') or {}
    if use_vision is False:
        if not done.get('ran'):
            skipped.append('vision: disabled (--no-vision): geometry and OSM evidence only')
        return False
    ok, why = vision.ready()
    if not ok:
        if use_vision:
            raise ValueError(f'vision requested but unavailable: {why}')
        if not done.get('ran'):
            skipped.append(f'vision: {why}')
        return False
    if done.get('ran') and not again and (done.get('facades') or not facades):
        return False
    geometry = store.read_json(home / 'objects-geometry.json')
    if not isinstance(geometry, list):  # survey made before the vision stage existed
        geometry = store.objects(meta)
        store.write_json(home / 'objects-geometry.json', geometry)
    ras = Rasters(meta)
    rings, shift = _osm_rings(meta)
    try:
        objs, record, rep = vision.run(meta, geometry, ras, aerial=True, facades=facades, osm_rings=rings, shift=shift)
    except vision.VisionError as exc:
        skipped.append(f'vision: {exc}')
        meta['vision'] = {'ran': False, 'reason': str(exc)}
        return True
    rep['facades_surveyed'] = bool(facades)
    meta['vision'] = {**rep, 'facades': bool(facades), 'at': store.now()}
    if not facades:
        skipped.append('facades: disabled (--no-facades): scan_building lists wall edges only')
    store.write_json(home / 'objects.json', objs)
    if (meta.get('street') or {}).get('ran'):
        # objects.json was rebuilt from objects-geometry.json without the street-view objects: the street stage
        # must run again (a reused survey with facades switched on re-runs vision, see scan_survey)
        meta['street'] = {'ran': False, 'reason': 'objects rebuilt by the vision stage: street views run again'}
    if facades:
        store.write_json(home / 'facades.json', record)
    counts: dict[str, int] = {}
    for o in objs:
        counts[o['cls']] = counts.get(o['cls'], 0) + 1
    meta['counts'] = counts
    objid = np.array(ras.objid)
    vision.stamp(objid, objs, meta)
    store.save_array(k, 'objid', objid)   # atomic: readers mapping the old file keep it
    return True


def _street_stage(meta: dict, use_vision: bool | None, street_views: bool | None, again: bool) -> bool:
    """Street views rendered from the scan (streetview.py) when wanted and not done yet (or again): detections along
    the OSM streets lifted to 3D objects. Needs vision and OSM streets. Updates meta, objects.json and the id raster.
    Returns True when it changed anything."""
    from . import streetview
    k = meta['key']
    home = store.home(k)
    meta['skipped'] = [x for x in meta.get('skipped', []) if not x.startswith('street views:')]
    done = meta.get('street') or {}
    if street_views is False or use_vision is False:
        if not done.get('ran'):
            why = '--no-street_views' if street_views is False else 'needs vision (--no-vision)'
            meta['skipped'].append(f'street views: disabled ({why})')
        return False
    ok, why = vision.ready()
    if not ok:
        if street_views:
            raise ValueError(f'street views need vision: {why}')
        if not done.get('ran'):
            meta['skipped'].append(f'street views: {why}')
        return False
    if done.get('ran') and not again:
        return False
    if not meta.get('osm', {}).get('used'):
        if street_views:
            raise ValueError('street views follow the OSM streets: run scan_survey with OSM (no --no-osm)')
        meta['skipped'].append('street views: no OSM streets (osm off or unavailable)')
        return False
    answer = osm_answer(meta)
    if not isinstance(answer, dict):
        meta['skipped'].append('street views: the cached OSM answer is missing (scan_survey --refresh)')
        return False
    feats = osm.parse(answer, meta['georef']['lat'], meta['georef']['lon'])
    shift = tuple(meta['osm'].get('shift_m') or (0.0, 0.0))
    objs = store.read_json(home / 'objects.json', [])
    ras = Rasters(meta)
    try:
        objs, rep = streetview.run(meta, objs, ras, feats, shift)
    except vision.VisionError as exc:
        meta['skipped'].append(f'street views: {exc}')
        meta['street'] = {'ran': False, 'reason': str(exc)}
        return True
    meta['street'] = {**rep, 'at': store.now()}
    if not rep.get('ran'):
        meta['skipped'].append(f"street views: {rep.get('reason')}")
        return True
    store.write_json(home / 'objects.json', objs)
    counts: dict[str, int] = {}
    for o in objs:
        counts[o['cls']] = counts.get(o['cls'], 0) + 1
    meta['counts'] = counts
    objid = np.array(ras.objid)
    vision.stamp(objid, objs, meta)
    store.save_array(k, 'objid', objid)
    return True


def _conf_summary(objs: list[dict]) -> list[dict]:
    rows = {}
    for o in objs:
        r = rows.setdefault(o['cls'], {'cls': o['cls'], 'count': 0, 'high': 0, 'mid': 0, 'low': 0, 'osm': 0,
                                       'vision': 0})
        r['count'] += 1
        r['high' if o['conf'] >= 0.75 else 'mid' if o['conf'] >= 0.5 else 'low'] += 1
        r['osm'] += o['src'] == 'mesh+osm' or bool(o.get('osm'))
        r['vision'] += 'vision' in o['src'] or bool(o.get('models'))
    return sorted(rows.values(), key=lambda r: -r['count'])


def _survey_reply(meta: dict, reused: bool) -> dict:
    objs = store.objects(meta)
    place = store.placement(meta['obj'])
    out = {'survey': meta['key'], 'reused': reused, 'obj': meta['obj'],
           'region_m': {'center': meta['center'], 'radius': meta['radius_m'], 'res': meta['grid']['res'],
                        'cells': f"{meta['grid']['w']}x{meta['grid']['h']}"},
           'placement': place, 'objects_total': len(objs), 'classes': _conf_summary(objs),
           'class_share': meta.get('class_share'), 'timings': meta.get('timings')}
    if meta.get('survey_stats'):
        out['rules'] = {**meta['survey_stats'], 'meaning': 'trees_on_roofs: crowns on roofs / roof edges dropped (cells '
                        'joined the roof); trees_beside_roofs: kept with lower conf and a note'}
    if meta.get('osm', {}).get('used'):
        o = meta['osm']
        out['osm'] = {'features': o.get('features'), 'source': o.get('source'), 'fetch_s': o.get('fetch_s'),
                      'shift_m': o.get('shift_m'), 'roof_overlap': o.get('roof_overlap'),
                      'footprints_disagree': o.get('footprints_disagree'), 'disagree': (o.get('disagree') or [])[:5],
                      'buildings_with_osm': o.get('buildings_split_or_tagged'), 'attribution': o.get('attribution')}
    vis = meta.get('vision') or {}
    if vis.get('ran'):
        w = vis.get('worker') or {}
        out['vision'] = {'gpu': w.get('gpu'), 'models': w.get('models'), 'timings': vis.get('timings'),
                         'worker_timings': w.get('timings'), 'vram_peak_gb': w.get('vram_peak_gb'),
                         'aerial': vis.get('aerial'), 'facades': vis.get('facade_survey')}
    if meta.get('providers'):
        out['providers'] = _providers_reply(meta)
    st = meta.get('street') or {}
    if st.get('ran'):
        out['street_views'] = {'cameras': st.get('cameras'), 'positions': st.get('camera_positions'),
                               'detections': st.get('detections'), 'lifted': st.get('lifted'), 'added': st.get('added'),
                               'relabelled': st.get('relabelled'), 'timings': st.get('timings')}
    out['skipped'] = meta.get('skipped', [])
    out['next'] = ('scan_place (map frame), then scan_view / scan_objects / scan_height' if not place else
                   'scan_view --center x,y; scan_objects --class car; scan_height x,y')
    if vis.get('facades'):
        out['next'] += '; scan_building ID, scan_facade ID'
    out['limitations'] = LIMITATIONS_VISION if vis.get('ran') else LIMITATIONS
    return out


def _providers_reply(meta: dict) -> dict:
    """Compact provider summary: used sources with attribution, fusion results, declined sources."""
    recs = meta.get('providers') or {}
    used = [{'name': n, 'kind': r.get('kind'), 'fetch_s': r.get('fetch_s'), 'attribution': r.get('attribution')}
            for n, r in recs.items() if r.get('used')]
    out: dict = {'used': used,
                 'not_covering': sorted(n for n, r in recs.items() if r.get('covers') is False),
                 'declined': sorted(n for n, r in recs.items() if r.get('declined'))}
    f = meta.get('fusion') or {}
    t = f.get('terrain')
    if t and t.get('used'):
        out['terrain'] = {k: t.get(k) for k in ('source', 'offset_m', 'residual_p50_m', 'residual_p90_m',
                                                'replaced_share')}
    for b in f.get('buildings') or []:
        keys = ('overridden', 'split', 'added', 'fragments_to_parts', 'mesh_only_flagged', 'agree', 'conflict',
                'roof_fit_p50', 'shift_m') if 'overridden' in b else ('paired', 'agree', 'conflict')
        out[b['source']] = {k: b.get(k) for k in keys if k in b}
    out['objects_with_conflicts'] = f.get('objects_with_conflicts')
    out['more'] = 'scan_sources'
    return out


# ---------------------------------------------------------------------------- state


def _ram_mb(k: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(store.ram(k)):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return round(total / 1e6, 1)


def scan_sources(latlon=None, radius_m: float = 150.0, survey_key: str = '') -> dict:
    """Data providers: for a place (latlon, offline coverage test only) or the survey (what was used, why not,
    licence, attribution, how it fused, and the fusion results)."""
    reg = prov.registry()
    if latlon is not None:
        p = _xy(latlon, 'latlon')[0]
        lat, lon = float(p[0]), float(p[1])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError('latlon: give latitude,longitude in degrees')
        if not 1 <= radius_m <= 5000:
            raise ValueError('radius_m must be 1..5000')
        dlat = radius_m / 111320.0
        dlon = radius_m / (111320.0 * max(0.05, math.cos(math.radians(lat))))
        bbox = (lat - dlat, lon - dlon, lat + dlat, lon + dlon)
        rows = []
        for pr in reg:
            ok, why = pr.coverage(bbox)
            rows.append({'name': pr.name, 'kind': pr.kind, 'scope': pr.scope, 'covers': ok, 'licence': pr.licence,
                         'why': why})
        return {'place': [lat, lon], 'radius_m': radius_m, 'providers': rows,
                'declined': [{'name': k, 'why': v} for k, v in prov.DECLINED.items()],
                'note': 'offline test of published extents; a regional provider confirms its tiles when scan_survey '
                        'fetches (only the bbox, tile names or byte ranges leave the machine)'}
    meta = store.resolve(survey_key)
    recs = meta.get('providers') or {}
    if not recs:
        return {'survey': meta['key'], 'providers': [],
                'note': 'this survey used no providers (made before them, --providers none, or no georeference)'}
    rows, attribution, fuses = [], [], {}
    for pr in reg:
        r = recs.get(pr.name) or {}
        rows.append({'name': pr.name, 'kind': pr.kind, 'scope': pr.scope, 'used': bool(r.get('used')),
                     'covers': r.get('covers'), 'fetch_s': r.get('fetch_s'), 'licence': pr.licence,
                     'why': (r.get('why') or '')[:90]})
        if r.get('used'):
            attribution.append(f'{pr.name}: {pr.attribution}')
            fuses[pr.name] = pr.fuses
    out = {'survey': meta['key'], 'bbox': meta.get('providers_bbox'), 'providers': rows, 'attribution': attribution,
           'how_fused': fuses, 'declined': [{'name': k, 'why': v} for k, v in prov.DECLINED.items()],
           'fusion': meta.get('fusion') or {}}
    st = meta.get('street') or {}
    if st:
        out['street_views'] = {k: st.get(k) for k in ('ran', 'reason', 'cameras', 'camera_positions', 'views_sent',
                                                      'detections', 'lifted', 'dropped', 'fused', 'added', 'relabelled',
                                                      'buildings_with_street_windows', 'timings') if k in st}
    out['limitations'] = [
        'Provider data and the scan were captured at different times: a conflict can be a real change.',
        'Vertical offsets between datums are fitted from the data (median), not from a geoid model.',
        'Street-view detections come from renders of the scan mesh: objects the mesh did not capture are not found.']
    return out


def scan_status(survey_key: str = '', drop_ram: str = '') -> dict:
    cur = store.current_key()
    dropped = []
    if drop_ram:
        want = drop_ram.strip()
        known = [m['key'] for m in store.surveys()]
        keys = [k for k in known if want == 'all' or k.startswith(want)]
        if want != 'all' and len(keys) != 1:
            raise ValueError(f'drop_ram: {want!r} matches {len(keys)} surveys; give one key (prefix) or all')
        for k in keys:
            if store.ram(k).is_dir():
                store.drop_ram(k)
                dropped.append(k)
    rows = []
    for m in store.surveys():
        if survey_key and not m['key'].startswith(survey_key):
            continue
        rows.append({'survey': m['key'], 'current': m['key'] == cur, 'obj': Path(m['obj']).name,
                     'center': m['center'], 'radius_m': m['radius_m'], 'res_m': m['grid']['res'],
                     'objects': sum((m.get('counts') or {}).values()), 'osm': bool(m.get('osm', {}).get('used')),
                     'placed': bool(store.placement(m['obj'])), 'in_ram': store.rasters_present(m['key']),
                     'ram_mb': _ram_mb(m['key']), 'stale': store.stale(m),
                     'created': time.strftime('%Y-%m-%d %H:%M', time.localtime(m.get('created', 0)))})
    out = {'surveys': rows, 'current': cur or None, 'disk': str(store.SCAN_HOME), 'ram': str(store.SHM)}
    if drop_ram:
        out['dropped_ram'] = dropped
    if cur:
        try:
            meta = store.resolve(cur, fresh=False)
            out['current_obj'] = meta['obj']
            out['placement'] = store.placement(meta['obj'])
            out['frame'] = Frame(meta).describe() + ('; x,y,z of every query and answer are in this frame' if
                                                     out['placement'] else '')
            imgs = store.views(cur)
            out['images'] = {'count': len(imgs), 'kept': store.KEEP_VIEWS, 'latest': [str(p) for p in imgs[:3]]}
        except store.NoSurvey:
            pass
    out['vision'] = vision.available()
    out['native'] = str(native.LIB) if native.LIB.is_file() else 'not built yet (built by the first scan_survey)'
    return out


def scan_place(at=None, yaw: float | None = None, datum_m: float | None = None, from_reference: bool = False,
               clear: bool = False, survey_key: str = '') -> dict:
    """yaw / datum_m left out keep the current placement's values (yaw 0 without one; datum_m is then required).
    The reply carries the replaced placement and the line that restores it."""
    meta = store.resolve(survey_key, fresh=False)
    obj = meta['obj']
    old = store.placement(obj)
    if clear:
        store.set_placement(obj, None)
        return {'survey': meta['key'], 'obj': obj, 'cleared': old is not None, 'frame': 'scan',
                **({'previous': old, 'restore': _restore_line(old)} if old else {})}
    kept = []
    if from_reference:
        if at is not None:
            raise ValueError('give either at/yaw/datum_m or from_reference')
        value = _from_reference(obj)
    else:
        if at is None or len(at) != 3:
            raise ValueError('at is x,y,z map metres where the export origin goes (as reference_load --at), '
                             'or pass from_reference')
        if datum_m is None:
            if not old:
                raise ValueError('the first placement needs datum_m, the source height put at z (reference_load '
                                 'reports datum_m; scan_status shows the survey ground)')
            datum_m = old['datum_m']
            kept.append('datum_m')
        if yaw is None:
            yaw = old['yaw'] if old else 0.0
            if old:
                kept.append('yaw')
        vals = [float(v) for v in at] + [float(yaw), float(datum_m)]
        if not all(math.isfinite(v) for v in vals):
            raise ValueError('at, yaw and datum_m must be finite')
        value = {'at': vals[:3], 'yaw': vals[3], 'datum_m': vals[4], 'source': 'given'}
    value['set'] = store.now()
    store.set_placement(obj, value)
    fr = Frame(meta, 'map')
    g = meta['grid']
    ce, cs = meta['center']
    cx, cy = fr.from_scan(ce, cs)
    out_site = _site_ground(meta, fr)
    out = {'survey': meta['key'], 'obj': obj, 'placement': value, 'frame': 'map',
           'scan_center_in_map': [round(cx, 2), round(cy, 2)], 'region_radius_m': meta['radius_m'],
           **({'site_ground': out_site} if out_site else {}),
           'note': f"queries now take and return map metres; --frame scan uses export metres (grid {g['w']}x{g['h']})"}
    if kept:
        out['kept'] = f"{' and '.join(kept)} of the previous placement"
    if old:
        same = all(old.get(k) == value.get(k) for k in ('at', 'yaw', 'datum_m'))
        out['previous'] = {k: old.get(k) for k in ('at', 'yaw', 'datum_m', 'source')}
        if not same:
            out['restore'] = _restore_line(old)
    return out


def _restore_line(p: dict) -> str:
    at = ','.join(f'{float(v):g}' for v in p['at'])
    return f"./uc scan_place --at {at} --yaw {float(p['yaw']):g} --datum_m {float(p['datum_m']):g}"


def _site_ground(meta: dict, fr: Frame, radius_m: float = 30.0) -> dict | None:
    """Ground of the site centre in the result frame from the survey's ground model (median within radius_m of the
    region centre) and where it sits relative to the datum; None without the rasters in RAM (no rebuild here)."""
    if not store.rasters_ok(meta['key']):
        return None
    try:
        ras = Rasters(meta)
        g = ras.g
        ce, cs = meta['center']
        k = int(radius_m / g.res)
        c, r = (int(round(float(v))) for v in g.to_cell(ce, cs))
        win = np.asarray(ras.dtm[max(0, r - k):r + k + 1, max(0, c - k):c + k + 1], float)
    except (store.NoSurvey, OSError, ValueError):
        return None
    if not win.size:
        return None
    up = float(np.median(win))
    z = fr.z_out(up)
    out = {'z': _r(z), 'how': f'median of the ground model within {radius_m:g} m of the region centre'}
    if fr.name == 'map':
        out['above_datum_m'] = _r(up - fr.datum)
        out['note'] = f"site ground at map z {z:.2f} ({up - fr.datum:+.2f} m from the datum)"
    return out


def _from_reference(obj: str) -> dict:
    last = reference.last_load()
    if not last:
        raise ValueError('no reference_load recorded on this machine: pass at, yaw and datum_m')
    if Path(last['obj']).resolve() != Path(obj).resolve():
        raise ValueError(f"the last reference_load was {last['obj']}, not this survey's OBJ: pass at, yaw and datum_m")
    st = reference.status()
    if not st.get('loaded'):
        raise ValueError('no reference is loaded in the editor now: pass at, yaw and datum_m (reference_load '
                         f"last used at={last['at']} yaw={last['yaw']} datum_m={last['datum_m']})")
    pose, want = st.get('pose_m') or [], last.get('pose_m') or []
    if len(pose) != 3 or any(abs(a - b) > 0.02 for a, b in zip(pose, want)) or abs(st.get('yaw', 0) - last['yaw']) > 0.01:
        raise ValueError('the loaded reference was not placed by the recorded reference_load: pass at, yaw and datum_m')
    return {'at': [float(v) for v in last['at']], 'yaw': float(last['yaw']), 'datum_m': float(last['datum_m']),
            'source': 'reference_load'}


# ---------------------------------------------------------------------------- point questions


def _obj_index(meta) -> dict[int, dict]:
    return {o['id']: o for o in store.objects(meta)}


MAX_POINTS = 1024


def _open_ground(ras: Rasters, r, c, reach_m: float = 25.0):
    """For cells under a roof or crown: the class of the nearest open-ground cell and its distance (metres) within
    reach_m, from the class raster (the ground under a roof is never seen; the nearest visible ground is the best
    guess of what continues under it)."""
    g = ras.g
    k = int(math.ceil(reach_m / g.res))
    out = []
    cls = ras.cls
    for ri, ci in zip(np.atleast_1d(r), np.atleast_1d(c)):
        r0, r1, c0, c1 = max(0, ri - k), min(g.h, ri + k + 1), max(0, ci - k), min(g.w, ci + k + 1)
        win = np.asarray(cls[r0:r1, c0:c1])
        m = np.isin(win, survey.GROUND_CODES)
        if not m.any():
            out.append((None, None))
            continue
        yy, xx = np.nonzero(m)
        d2 = (yy + r0 - ri) ** 2 + (xx + c0 - ci) ** 2
        j = int(np.argmin(d2))
        out.append((survey.CLASSES[int(win[yy[j], xx[j]])], round(math.sqrt(float(d2[j])) * g.res, 1)))
    return out


def scan_height(points, agl: bool = False, frame: str = 'auto', survey_key: str = '') -> dict:
    p = _xy(points)
    if len(p) > MAX_POINTS:
        raise ValueError(f'scan_height takes 1 to {MAX_POINTS} points (scan_grid samples a whole box)')
    has_z = p.shape[1] >= 3
    if agl and not has_z:
        raise ValueError('agl needs x,y,z points (z metres above the scan ground)')
    meta, fr, ras = _context(survey_key, frame)
    objs = _obj_index(meta)
    e, s = fr.to_scan(p[:, 0], p[:, 1])
    r, c, inside = ras.cell(e, s)
    ground = ras.ground(e, s)
    covered = np.isin(np.asarray(ras.cls[r, c]), (10, 11, 12))
    under = _open_ground(ras, r[covered], c[covered]) if covered.any() else []
    under_it = iter(under)
    rows = []
    for i in range(len(p)):
        top, low = float(ras.dsm[r[i], c[i]]), float(ras.low[r[i], c[i]])
        ok = bool(inside[i]) and math.isfinite(top)
        oid = int(ras.objid[r[i], c[i]]) if ok else -1
        o = objs.get(oid)
        gz = float(ground[i]) if ok else None
        row = {'x': _r(p[i, 0]), 'y': _r(p[i, 1]),
               'ground_z': _r(fr.z_out(gz)) if ok else None, 'top_z': _r(fr.z_out(top)) if ok else None,
               'height_m': _r(top - gz) if ok else None,
               'under_m': _r(top - low) if ok and top - low > 1.0 else None,
               'surface': survey.CLASSES.get(int(ras.cls[r[i], c[i]]), '?') if ok else 'no_data',
               'object': oid if o else None, 'cls': o['cls'] if o else None}
        if covered[i]:
            gcls, gd = next(under_it)
            if ok:
                row['ground_cls'] = gcls
                row['ground_cls_at_m'] = gd
        if has_z and ok:
            z_up = gz + float(p[i, 2]) if agl else float(fr.z_in(p[i, 2]))
            row['z'] = _r(fr.z_out(z_up))
            row['agl_m'] = _r(z_up - gz)
            row['clearance_m'] = _r(z_up - top)
        rows.append(row)
    out = {**_head(meta, fr, ras), 'points': rows, 'units': 'metres',
           'meaning': 'ground_z: filtered ground model (interpolated under roofs and crowns); top_z: highest mesh '
                      'surface; surface: class of that top surface; ground_cls: under a roof, crown or low object, the '
                      'class of the nearest visible ground (ground_cls_at_m away); under_m: top minus lowest surface '
                      'when > 1 m (overhang, bridge deck, canopy over mesh)'}
    if has_z:
        out['z_in'] = 'metres above the scan ground' if agl else f'{fr.name} z'
        out['meaning'] += '; with z: z (absolute), agl_m above ground and clearance_m above the top surface'
    return out


def scan_profile(a, b, step_m: float = 1.0, agl: bool = False, frame: str = 'auto', survey_key: str = '') -> dict:
    pa, pb = _xy(a, 'start')[0], _xy(b, 'end')[0]
    line_z = len(pa) >= 3 and len(pb) >= 3
    if agl and not line_z:
        raise ValueError('agl needs x,y,z start and end (z metres above the scan ground)')
    meta, fr, ras = _context(survey_key, frame)
    L = float(math.hypot(*(pb[:2] - pa[:2])))
    if L <= 0:
        raise ValueError('start and end are the same point')
    step = max(float(step_m), ras.g.res)
    note = None
    if L / step > 200:
        step = L / 200
        note = f'step_m raised to {step:.2f} to keep 201 samples'
    n = int(math.floor(L / step)) + 1
    t = np.linspace(0, (n - 1) * step / L, n)
    if t[-1] < 1:
        t = np.append(t, 1.0)
    x, y = pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])
    e, s = fr.to_scan(x, y)
    r, c, inside = ras.cell(e, s)
    gz = ras.ground(e, s)
    top = np.asarray(ras.dsm[r, c], float)
    cls = np.asarray(ras.cls[r, c])
    okm = inside & np.isfinite(top)
    open_ = np.isin(cls, survey.GROUND_CODES) & okm
    # slope to the next sample: only between two open-ground samples (roofs, crowns: the model is interpolated)
    d = t * L
    seg = np.full(len(t), np.nan)
    if len(t) > 1:
        both = open_[:-1] & open_[1:]
        g = np.diff(gz) / np.maximum(np.diff(d), 1e-6) * 100
        seg[:-1] = np.where(both, g, np.nan)
    if line_z:
        if agl:
            zl = gz + (pa[2] + t * (pb[2] - pa[2]))
        else:
            zl = fr.z_in(pa[2] + t * (pb[2] - pa[2]))
    rows = []
    for i in range(len(t)):
        ok = bool(okm[i])
        row = {'d_m': _r(d[i], 1), 'x': _r(x[i]), 'y': _r(y[i]),
               'ground_z': _r(fr.z_out(gz[i])) if ok else None, 'top_z': _r(fr.z_out(top[i])) if ok else None,
               'height_m': _r(top[i] - gz[i]) if ok else None,
               'surface': survey.CLASSES.get(int(cls[i]), '?') if ok else 'no_data',
               'slope_pct': _r(seg[i], 1)}
        if line_z:
            row['z'] = _r(fr.z_out(zl[i]))
            row['clearance_m'] = _r(zl[i] - top[i]) if ok else None
        rows.append(row)
    g_ok = gz[okm]
    g_open = gz[open_]
    steep = {}
    fin = np.isfinite(seg)
    if fin.any():
        j = int(np.nanargmax(np.abs(seg)))
        steep['max_ground_slope_pct'] = _r(abs(seg[j]), 1)
        steep['max_ground_slope_at'] = {'d_m': _r(d[j], 1), 'surface': rows[j]['surface']}
        steep['mean_ground_slope_pct'] = _r(float(np.nanmean(seg)), 1)
    else:
        steep['max_ground_slope_pct'] = None
    out = {**_head(meta, fr, ras), 'length_m': _r(L), 'step_m': _r(step),
           'ground_min_z': _r(fr.z_out(g_open.min())) if g_open.size else None,
           'ground_max_z': _r(fr.z_out(g_open.max())) if g_open.size else None,
           'top_max_z': _r(fr.z_out(top[okm].max())) if okm.any() else None,
           'open_ground_share': _r(float(open_.sum() / max(1, okm.sum())), 2),
           **steep, 'samples': rows, 'units': 'metres',
           'meaning': 'slope_pct: ground slope to the next sample (percent, + uphill along the line), only between '
                      'two samples on open ground (road, paved, grass...): under buildings and crowns the ground model '
                      'is interpolated, so their edges are not ground slopes; ground_min/max over open ground'}
    if line_z:
        cl = np.where(okm, zl - top, np.nan)
        if np.isfinite(cl).any():
            j = int(np.nanargmin(cl))
            out['min_clearance_m'] = _r(cl[j])
            out['min_clearance_at'] = {'d_m': _r(d[j], 1), 'surface': rows[j]['surface']}
        out['z_in'] = 'metres above the scan ground' if agl else f'{fr.name} z'
    if note:
        out['note'] = '; '.join(filter(None, (out.get('note'), note)))
    return out


def _dist_to_object(o: dict, e: float, s: float) -> float:
    fp = o.get('footprint')
    if fp and len(fp) >= 3:
        poly = np.asarray(fp, float)
        if _inside(poly, e, s):
            return 0.0
        return _poly_dist(poly, e, s)
    a = math.radians(o['yaw_deg'])
    dx, dy = e - o['x'], s - o['y']
    u, v = abs(dx * math.cos(a) + dy * math.sin(a)), abs(-dx * math.sin(a) + dy * math.cos(a))
    return math.hypot(max(0.0, u - o['length_m'] / 2), max(0.0, v - o['width_m'] / 2))


def _inside(poly: np.ndarray, x: float, y: float) -> bool:
    xs, ys = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xs, 1), np.roll(ys, 1)
    cross = ((ys > y) != (yj > y)) & (x < (xj - xs) * (y - ys) / np.where(yj - ys == 0, 1e-12, yj - ys) + xs)
    return bool(cross.sum() % 2)


def _poly_dist(poly: np.ndarray, x: float, y: float) -> float:
    a, b = poly, np.roll(poly, -1, 0)
    ab = b - a
    t = np.clip(((x - a[:, 0]) * ab[:, 0] + (y - a[:, 1]) * ab[:, 1]) / np.maximum((ab ** 2).sum(1), 1e-12), 0, 1)
    d = np.hypot(a[:, 0] + t * ab[:, 0] - x, a[:, 1] + t * ab[:, 1] - y)
    return float(d.min())


def colour_name(rgb) -> str | None:
    """A plain colour word for a median texture colour (photogrammetry: lighting and shade shift it)."""
    if not rgb or len(rgb) < 3:
        return None
    import colorsys
    r, g, b = (max(0, min(255, int(v))) / 255 for v in rgb[:3])
    h, light, sat = colorsys.rgb_to_hls(r, g, b)
    h *= 360
    chroma = max(r, g, b) - min(r, g, b)
    grey = 'black' if light < 0.16 else 'dark grey' if light < 0.38 else 'grey' if light < 0.68 else 'white'
    if sat < 0.15 or chroma < 0.1:
        return grey
    if (170 <= h < 260 and chroma < 0.25) or (260 <= h < 340 and chroma < 0.2):
        return grey   # shade and sky light tint grey surfaces blue or mauve in photogrammetry
    tone = 'dark ' if light < 0.3 else 'light ' if light > 0.72 else ''
    if h < 15 or h >= 340:
        name = 'red'
    elif h < 45:
        name = 'brown' if light < 0.45 else 'orange' if sat > 0.5 else 'beige'
        tone = '' if name in ('brown', 'beige') else tone
    elif h < 70:
        name = 'yellow' if sat > 0.4 and light > 0.35 else 'olive'
    elif h < 170:
        name = 'green'
    elif h < 260:
        name = 'blue'
    else:
        name = 'purple'
    return tone + name


def _colour(rgb) -> dict | None:
    if not rgb or len(rgb) < 3:
        return None
    return {'name': colour_name(rgb), 'hex': '#%02x%02x%02x' % tuple(max(0, min(255, int(v))) for v in rgb[:3])}


def _overlaps(objs, cell: float = 0.5) -> dict[int, tuple[int, float]]:
    """Building id -> (other building id, share of the smaller footprint) for footprints that overlap by more than
    a fifth of the smaller one: the same building found twice, or a part listed as its own building. Footprints
    are rasterised at cell metres (they are not convex)."""
    blds = [o for o in objs if o['cls'] == 'building' and o.get('footprint') and len(o['footprint']) >= 3]
    polys = [np.asarray(o['footprint'], float) for o in blds]
    boxes = np.array([(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()) for p in polys]).reshape(-1, 4)
    out: dict[int, tuple[int, float]] = {}
    for i in range(len(blds)):
        hit = np.flatnonzero((boxes[:, 0] < boxes[i, 2]) & (boxes[i, 0] < boxes[:, 2]) &
                             (boxes[:, 1] < boxes[i, 3]) & (boxes[i, 1] < boxes[:, 3]))
        for j in hit[hit > i]:
            x0, y0 = min(boxes[i, 0], boxes[j, 0]), min(boxes[i, 1], boxes[j, 1])
            w = int((max(boxes[i, 2], boxes[j, 2]) - x0) / cell) + 2
            h = int((max(boxes[i, 3], boxes[j, 3]) - y0) / cell) + 2
            masks = []
            for p in (polys[i], polys[j]):
                m = np.zeros((h, w), np.int32)
                native.fill_poly(m, (p[:, 0] - x0) / cell - .5, (p[:, 1] - y0) / cell - .5, 1)
                masks.append(m > 0)
            small = min(int(masks[0].sum()), int(masks[1].sum()))
            share = float((masks[0] & masks[1]).sum() / small) if small else 0.0
            if share > 0.2:
                for x, y in ((i, j), (j, i)):
                    if share > out.get(blds[x]['id'], (None, 0.0))[1]:
                        out[blds[x]['id']] = (blds[y]['id'], round(share, 2))
    return out


def _shared_osm(objs) -> set[str]:
    """OSM building ways claimed by more than one scan building (one mapped building in several mesh pieces)."""
    seen: dict[str, int] = {}
    for o in objs:
        ref = (o.get('osm') or {}).get('osm') if o['cls'] == 'building' else None
        if ref:
            seen[ref] = seen.get(ref, 0) + 1
    return {k for k, n in seen.items() if n > 1}


def _flag(o: dict, shared: set[str] = frozenset(), overlaps: dict | None = None) -> str | None:
    """Short reason to doubt an object before building from it, or None."""
    why = []
    if o.get('note', '').startswith('not a car'):
        why.append(o['note'])
    if o['cls'] in ('car', 'truck'):
        if o['height_m'] < 0.8:
            why.append(f"flat ({o['height_m']:g} m)")
        if o['length_m'] < 2.5:
            why.append('short')
    if o['cls'] == 'tree':
        crown = o.get('crown_m') or (o['length_m'] + o['width_m']) / 2
        if o['height_m'] > 3 * crown and min(crown, o['width_m']) < 4:
            why.append('slender: roof greenery or edge?')
    if o['src'] == 'vision' and o['conf'] < 0.5:
        why.append('weak, vision only')
    if o.get('conflicts'):
        why.append('sources disagree (' + ', '.join(sorted({e['src'] for e in o.get('evidence', [])
                                                           if e['verdict'] == 'conflict'})) + ')')
    ref = (o.get('osm') or {}).get('osm') if o['cls'] == 'building' else None
    if ref and ref in shared:
        why.append(f'part of {ref}')
    if overlaps and o['id'] in overlaps:
        other, share = overlaps[o['id']]
        why.append(f'overlaps building {other} ({share:.0%})')
    if o.get('osm_conflict'):
        why.append(f"OSM footprint disagrees ({o['osm_conflict']['cover']:.0%} raised)")
    if o['cls'] == 'tree' and 'roof' in (o.get('note') or ''):
        why.append('against a roof of its height')
    return '; '.join(why) or None


def _row(o: dict, fr: Frame, dist: float | None = None, shared: set[str] = frozenset(),
         overlaps: dict | None = None) -> dict:
    x, y = fr.from_scan(o['x'], o['y'])
    col = _colour(o.get('rgb'))
    row = {'id': o['id'], 'cls': o['cls'], 'conf': o['conf'], 'x': _r(x), 'y': _r(y),
           'ground_z': _r(fr.z_out(o['ground_z'])), 'height_m': o['height_m'], 'length_m': o['length_m'],
           'width_m': o['width_m'], 'yaw_deg': _r(fr.yaw_out(o['yaw_deg']), 1), 'src': o['src'],
           'colour': col['name'] if col else None, 'rgb': col['hex'] if col else None,
           'flag': _flag(o, shared, overlaps)}
    if dist is not None:
        row['dist_m'] = _r(dist, 1)
    return row


def scan_at(point, radius_m: float = 10.0, limit: int = 15, frame: str = 'auto', survey_key: str = '') -> dict:
    p = _xy(point, 'point')[0]
    if not 0 < radius_m <= 200:
        raise ValueError('radius_m must be 0..200')
    meta, fr, ras = _context(survey_key, frame)
    e, s = fr.to_scan(p[0], p[1])
    here = scan_height([p[:2].tolist()], frame=frame, survey_key=meta['key'])['points'][0]
    near = []
    every = store.objects(meta)
    shared = _shared_osm(every)
    over = _overlaps(every)
    for o in every:
        if o['cls'] == 'building_part':
            continue
        if math.hypot(o['x'] - e, o['y'] - s) > radius_m + max(o['length_m'], o['width_m']):
            continue
        d = _dist_to_object(o, float(e), float(s))
        if d <= radius_m:
            near.append((d, o))
    near.sort(key=lambda t: (t[0], -t[1]['conf']))
    lim = max(1, min(int(limit), 100))
    # surface shares of the cells within radius_m of the true point (cells off the grid do not count)
    g = ras.g
    cc, rr = g.to_cell(float(e), float(s))
    k = int(math.ceil(radius_m / g.res)) + 1
    r0, r1 = max(0, int(math.floor(rr)) - k), min(g.h, int(math.ceil(rr)) + k + 1)
    c0, c1 = max(0, int(math.floor(cc)) - k), min(g.w, int(math.ceil(cc)) + k + 1)
    shares, note = {}, None
    if r0 < r1 and c0 < c1:
        yy, xx = np.mgrid[r0:r1, c0:c1]
        disc = ((yy - rr) ** 2 + (xx - cc) ** 2) * g.res ** 2 <= radius_m ** 2
        cl = np.asarray(ras.cls[r0:r1, c0:c1])[disc]
        if cl.size:
            shares = {survey.CLASSES[int(v)]: round(float((cl == v).mean()), 2) for v in np.unique(cl)
                      if (cl == v).mean() >= 0.02}
    if not shares:
        note = 'point and radius lie outside the survey region: no surface shares'
    out = {**_head(meta, fr, ras), 'here': here, 'radius_m': radius_m,
           'surface_share': dict(sorted(shares.items(), key=lambda kv: -kv[1])),
           'objects_count': len(near), 'objects': [_row(o, fr, d, shared, over) for d, o in near[:lim]],
           'truncated': len(near) > lim, 'units': 'metres'}
    if note:
        out['note'] = note
    return out


SORTS = ('', 'near', 'distance', 'conf', 'height', 'area', 'windows', 'storeys')


def _windows(meta: dict, idx: dict[int, dict]) -> dict[int, tuple[int, int | None, int, int | None]]:
    """Building id -> (windows in total, side with the most, windows on it, storeys) from the facade survey."""
    out = {}
    for bid, walls in _facade_record(meta).items():
        o = idx.get(int(bid))
        if o is None:
            continue
        own = _own_walls(o, walls)
        per = [((w.get('grid') or {}).get('windows') or 0, w.get('side')) for w in own]
        best = max(per, default=(0, None))
        st = [(w.get('grid') or {}).get('storeys') for w in own if w.get('status') == 'surveyed']
        st = [v for v in st if v]
        out[int(bid)] = (sum(n for n, _ in per), best[1] if best[0] else None, best[0],
                         int(np.median(st)) if st else None)
    return out


def scan_objects(classes=None, box=None, near=None, radius_m: float = 50.0, min_conf: float = 0.0, limit: int = 30,
                 sort: str = '', nearest: int = 0, frame: str = 'auto', survey_key: str = '') -> dict:
    meta = store.resolve(survey_key)
    fr = Frame(meta, frame)
    want = {c.strip().lower() for c in (classes or []) if c.strip()}
    if box is not None and near is not None:
        raise ValueError('give box or near, not both')
    sort = (sort or '').strip().lower()
    if sort not in SORTS:
        raise ValueError('sort must be near (distance), conf, height, area, windows or storeys')
    sort = 'near' if sort == 'distance' else sort
    nearest = int(nearest or 0)
    if nearest < 0 or nearest > 500:
        raise ValueError('nearest must be 1..500 (0 = off)')
    if (sort == 'near' or nearest) and near is None:
        raise ValueError(('nearest' if nearest else 'sort near') + ' needs near x,y')
    b = _box4(box) if box is not None else None
    n = _xy(near, 'near')[0] if near is not None else None
    lim = max(1, min(int(limit), 500))
    if nearest:
        lim = nearest
    every = store.objects(meta)
    shared = _shared_osm(every)
    over = _overlaps(every)
    hits, outside, ref = [], None, None
    if b is not None:
        ref = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    elif n is not None:
        ref = (float(n[0]), float(n[1]))
    for o in every:
        if want and o['cls'] not in want:
            continue
        if o['conf'] < min_conf:
            continue
        x, y = fr.from_scan(o['x'], o['y'])
        d = math.hypot(x - ref[0], y - ref[1]) if ref is not None else None
        inside = not (b and not (b[0] <= x <= b[2] and b[1] <= y <= b[3])) and \
            not (n is not None and not nearest and d > radius_m)
        if not inside:
            if outside is None or d < outside[0]:
                outside = (d, o)
            continue
        hits.append((d if n is not None else None, o))
    win = _windows(meta, {o['id']: o for o in every})
    sort = sort or ('near' if n is not None else 'conf')
    keys = {'near': lambda t: (t[0], -t[1]['conf'], t[1]['id']),
            'conf': lambda t: (-t[1]['conf'], t[1]['id']),
            'height': lambda t: (-t[1]['height_m'], t[1]['id']),
            'area': lambda t: (-t[1].get('area_m2', 0), t[1]['id']),
            'windows': lambda t: (-win.get(t[1]['id'], (0, None, 0, None))[2], -win.get(t[1]['id'], (0,))[0],
                                  t[1]['id']),
            'storeys': lambda t: (-(win.get(t[1]['id'], (0, None, 0, None))[3] or t[1].get('storeys') or 0),
                                  -t[1]['height_m'], t[1]['id'])}
    hits.sort(key=keys[sort])
    by = {}
    for _, o in hits:
        by[o['cls']] = by.get(o['cls'], 0) + 1
    known = sorted({o['cls'] for o in every})
    unknown = sorted(want - set(known))
    rows = []
    for d, o in hits[:lim]:
        row = _row(o, fr, d, shared, over)
        if o['cls'] == 'building':
            total, side, most, storeys = win.get(o['id'], (None, None, None, None))
            row.update(area_m2=o.get('area_m2'), storeys=storeys or o.get('storeys'), windows=total,
                       roof=o.get('roof'))
            if sort == 'windows':
                row.update(best_side=side, side_windows=most)
        rows.append(row)
    out = {'survey': meta['key'], 'frame': fr.name, 'total': len(hits), 'by_class': by, 'sort': sort,
           'objects': rows, 'truncated': len(hits) > lim, 'units': 'metres'}
    if nearest:
        out['nearest'] = nearest
        out.pop('truncated')
    if not hits and outside is not None:
        d, o = outside
        x, y = fr.from_scan(o['x'], o['y'])
        out['nearest_outside'] = {'id': o['id'], 'cls': o['cls'], 'conf': o['conf'], 'x': _r(x), 'y': _r(y),
                                  'dist_m': _r(d, 1) if d is not None else None, 'flag': _flag(o, shared, over)}
    if not hits:
        what = ', '.join(sorted(want)) if want else 'object'
        where = 'in the box' if b else f'within {radius_m:g} m' if n is not None else 'in the survey'
        if min_conf:
            where += f' at conf >= {min_conf:g}'
        if outside is not None and outside[0] is not None:
            o = outside[1]
            frm = 'the box centre' if b else 'near'
            out['summary'] = f"no {what} {where}; nearest {o['cls']} {o['id']} is {outside[0]:.0f} m from {frm}"
        else:
            out['summary'] = f'no {what} {where}' + ('' if not unknown else ' (unknown class)')
    if sort in ('windows', 'storeys') and not win:
        out['note'] = 'no facade survey: windows and storeys are unknown (scan_survey with vision and facades)'
    if unknown:
        out['unknown_classes'] = unknown
        out['classes_known'] = known
    return out


def scan_ground(box, min_area_m2: float = 20.0, limit: int = 20, simplify_m: float = 0.5, frame: str = 'auto',
                survey_key: str = '') -> dict:
    x0, y0, x1, y1 = _box4(box)
    meta, fr, ras = _context(survey_key, frame)
    if (x1 - x0) * (y1 - y0) > 300 * 300:
        raise ValueError('box is larger than 300 x 300 m: query smaller boxes')
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    e, s = fr.to_scan(corners[:, 0], corners[:, 1])
    g = ras.g
    c, r = g.to_cell(np.asarray(e), np.asarray(s))
    c0, c1 = max(0, int(np.floor(c.min()))), min(g.w, int(np.ceil(c.max())) + 1)
    r0, r1 = max(0, int(np.floor(r.min()))), min(g.h, int(np.ceil(r.max())) + 1)
    if c0 >= c1 or r0 >= r1:
        raise ValueError('box lies outside the survey region')
    cls = np.array(ras.cls[r0:r1, c0:c1])
    # mask to the (possibly rotated) box
    yy, xx = np.mgrid[r0:r1, c0:c1]
    we, ws = g.to_world(xx, yy)
    fx, fy = fr.from_scan(we, ws)
    inbox = (fx >= x0) & (fx <= x1) & (fy >= y0) & (fy <= y1)
    cl = np.where(inbox, cls, 255)
    tot = max(1, int(inbox.sum()))
    shares = {survey.CLASSES[int(v)]: round(float((cl == v).sum() / tot), 3) for v in np.unique(cl[inbox])}
    polys = []
    for code in survey.GROUND_CODES + (10, 11):
        mask = cl == code
        if mask.sum() * g.res ** 2 < min_area_m2:
            continue
        lab, n = native.label(mask, conn8=False)
        for i, (rr, cc) in enumerate(grid.components(lab, n), 1):
            area = len(rr) * g.res ** 2
            if area < min_area_m2:
                continue
            ring = grid.outline(lab, i, rr, cc)
            ring = grid.simplify(ring, simplify_m / g.res) if len(ring) > 3 else ring
            pe, ps = g.to_world(ring[:, 0] + c0, ring[:, 1] + r0)
            px, py = fr.from_scan(pe, ps)
            polys.append({'cls': survey.CLASSES[code], 'area_m2': round(area, 1), 'n': len(ring),
                          'points': [[round(float(a), 1), round(float(b), 1)] for a, b in zip(px, py)]})
    polys.sort(key=lambda p: -p['area_m2'])
    lim = max(1, min(int(limit), 100))
    return {**_head(meta, fr, ras), 'box': [x0, y0, x1, y1],
            'share': dict(sorted(shares.items(), key=lambda kv: -kv[1])), 'polygons_count': len(polys),
            'polygons': polys[:lim], 'truncated': len(polys) > lim, 'units': 'metres',
            'note': 'outlines are cell-centre traces simplified by simplify_m; building/vegetation are included '
                    'for context'}


def _facade_record(meta: dict) -> dict:
    value = store.read_json(store.home(meta['key']) / 'facades.json', {})
    return value if isinstance(value, dict) else {}


def _building(meta: dict, ident: int, parts: bool = False) -> dict:
    """The building with this id; a building_part id selects its parent unless parts (then the part itself)."""
    objs = _obj_index(meta)
    o = objs.get(int(ident))
    if o is None:
        raise ValueError(f'no object {ident} in survey {meta["key"]}')
    if o['cls'] == 'building_part' and not parts:
        o = objs[o['parent']]
    if o['cls'] not in ('building', 'building_part'):
        raise ValueError(f"object {ident} is a {o['cls']}, not a building (scan_objects --classes building)")
    return o


def _own_walls(o: dict, walls: list[dict]) -> list[dict]:
    """Walls of a facade record that belong to this building (surveys before the shared-outline fix stored every
    wall of a shared OSM outline under each part)."""
    return [w for w in walls if facade.belongs(o['footprint'], w['a'], w['b'])]


def _rows_check(rows: list[dict]) -> str | None:
    """Doubts about a fitted window grid: rows below ground, rows overlapping the next one."""
    why = []
    low = sum(1 for r in rows if r['sill_m'] < -0.2)
    if low:
        why.append(f'{low} row(s) start below ground')
    inverted = sum(1 for r in rows if r['head_m'] <= r['sill_m'])
    if inverted:
        why.append(f'{inverted} row(s) with the head at or below the sill')
    over = sum(1 for a, b in zip(rows, rows[1:]) if b['sill_m'] < a['head_m'] - 0.1)
    if over:
        why.append(f'{over} row(s) overlap the next')
    return '; '.join(why) + ': check scan_facade' if why else None


def _wall_row(w: dict, fr: Frame, i: int) -> dict:
    ax, ay = fr.from_scan(*w['a'])
    bx, by = fr.from_scan(*w['b'])
    g = w.get('grid') or {}
    rows = g.get('rows') or []
    return {'side': w.get('side', i), 'from': w.get('wall_from', 'roof_outline'), 'x0': _r(ax), 'y0': _r(ay),
            'x1': _r(bx), 'y1': _r(by), 'length_m': _r(w['length_m']), 'faces_deg': _r(fr.yaw_out(w['faces_deg'], 360), 0),
            'status': w.get('status', 'not_surveyed'), 'storeys': g.get('storeys'), 'windows': g.get('windows'),
            'columns': g.get('columns'), 'spacing_m': g.get('spacing_m'),
            'sill_m': [r['sill_m'] for r in rows] or None, 'head_m': [r['head_m'] for r in rows] or None,
            'doors': len(g.get('doors') or []) if g else None, 'balconies': len(g.get('balconies') or []) if g else None,
            'tone': _colour(w.get('tone'))['name'] if w.get('tone') else None, 'check': _rows_check(rows)}


def _roof_out(o: dict, fr: Frame) -> dict | None:
    """roof_shape of a building or part in the result frame (line ends moved, yaw turned)."""
    f = o.get('roof_shape')
    if not f:
        return None
    f = dict(f)
    if f.get('line'):
        ln = dict(f['line'])
        for k in ('a', 'b'):
            x, y = fr.from_scan(*ln[k])
            ln[k] = [_r(x), _r(y)]
        ln['yaw_deg'] = _r(fr.yaw_out(ln['yaw_deg']), 1)
        f['line'] = ln
    return f


def _part_row(p: dict, fr: Frame) -> dict:
    row = _row(p, fr)
    fp = p.get('footprint')
    if fp:
        q = np.asarray(fp, float)
        qx, qy = fr.from_scan(q[:, 0], q[:, 1])
        row['footprint'] = [[_r(a, 2), _r(b, 2)] for a, b in zip(qx, qy)]
    row.update(area_m2=p.get('area_m2'), eave_m=p.get('eave_m'), ridge_m=p.get('ridge_m'), roof=p.get('roof'),
               ridge_yaw_deg=_r(fr.yaw_out(p['ridge_yaw_deg']), 1) if p.get('ridge_yaw_deg') is not None else None,
               roof_shape=_roof_out(p, fr))
    return row


def _facades_now(meta: dict, o: dict) -> tuple[list[dict] | None, str | None]:
    """Facade survey of one building the survey skipped, run now when vision is available and stored in
    facades.json: (walls, None) or (None, why not)."""
    ok, why = vision.ready()
    if not ok:
        return None, f'vision unavailable ({why})'
    ras = Rasters(meta)
    rings, shift = _osm_rings(meta)
    try:
        rec = vision.facades_on_demand(meta, store.objects(meta), ras, {o['id']}, rings, shift)
    except vision.VisionError as exc:
        return None, f'facade survey failed: {exc}'
    walls = rec.get(str(o['id'])) or []
    path = store.home(meta['key']) / 'facades.json'
    with store.build_lock(meta['key']):
        allrec = _facade_record(meta)
        allrec[str(o['id'])] = walls
        store.write_json(path, allrec)
    return walls, None


def eave_of(meta: dict, o: dict, ras: 'Rasters | None' = None) -> float | None:
    """Eave height above ground of a building: stored when the survey measured it at the roof edge
    (eave_from: edge); for older surveys (15th percentile of the whole roof, ~1 m high on tall gabled roofs)
    measured now from the roof edge band of the building's cells."""
    if o.get('eave_from') or o.get('cls') != 'building':
        return o.get('eave_m')
    ras = ras or Rasters(meta)
    g = ras.g
    c, r = g.to_cell(o['x'], o['y'])
    reach = int(math.hypot(o['length_m'], o['width_m']) / g.res) + 4
    r0, c0 = max(0, int(r) - reach), max(0, int(c) - reach)
    win = np.asarray(ras.objid[r0:int(r) + reach + 1, c0:int(c) + reach + 1]) == o['id']
    rows, cols = np.nonzero(win)
    if not len(rows):
        return o.get('eave_m')
    rr, cc = rows + r0, cols + c0
    nd = np.asarray(ras.dsm[rr, cc], float) - np.asarray(ras.dtm[rr, cc], float)
    e = survey.edge_eave(rows, cols, nd, g.res)
    return round(e, 2) if e is not None else o.get('eave_m')


def _false_party_walls(meta: dict, o: dict, walls: list[dict], ras: 'Rasters') -> list[int]:
    """Sides stored as party walls that only the building's own eaves stood in front of (surveys before the
    own-cell rule): they deserve a facade survey."""
    out = []
    eave = o['ground_z'] + (eave_of(meta, o, ras) or o['height_m'])
    for w in walls:
        if w.get('status') != 'party_wall':
            continue
        a, b = np.asarray(w['a'], float), np.asarray(w['b'], float)
        f = math.radians(w['faces_deg'])
        spec = {'a': a, 'b': b, 'n': np.array([math.cos(f), math.sin(f)])}
        if facade.occluded_front(ras, spec, w.get('ground_up', o['ground_z']), eave, own_id=o['id']) < 0.5:
            out.append(w['side'])
    return out


def scan_building(ident: int, min_facade_m: float = 2.0, survey_facades: bool = True, frame: str = 'auto',
                  survey_key: str = '') -> dict:
    meta = store.resolve(survey_key)
    fr = Frame(meta, frame)
    objs = _obj_index(meta)
    sel = _building(meta, ident, parts=True)
    part = sel if sel['cls'] == 'building_part' else None
    o = objs[sel['parent']] if part else sel
    fp = np.asarray(o['footprint'], float)
    px, py = fr.from_scan(fp[:, 0], fp[:, 1])
    poly_f = np.stack([px, py], 1)
    surveyed = _facade_record(meta).get(str(o['id']))
    on_demand = None
    ras = Rasters(meta)
    false_party = _false_party_walls(meta, o, _own_walls(o, surveyed), ras) if surveyed else []
    refreshed = False
    if (surveyed is None or false_party) and survey_facades:
        walls_now, on_demand = _facades_now(meta, o)
        if walls_now is not None:
            refreshed = True
            surveyed, on_demand = walls_now, (
                f"surveyed now: sides {', '.join(map(str, false_party))} were marked party walls by the building's "
                'own eaves' if false_party else 'surveyed now (the survey had skipped this building)')
    if surveyed and false_party and not refreshed:
        surveyed = [dict(w) for w in surveyed]
        for w in surveyed:   # not surveyed again: say so instead of calling them party walls
            if w.get('side') in false_party and w.get('status') == 'party_wall':
                w['status'] = 'not_surveyed'
        on_demand = (f"sides {', '.join(map(str, false_party))} were marked party walls by the building's own eaves: "
                     'not_surveyed now' + (f' ({on_demand})' if on_demand else ' (survey_facades off)'))
    elsewhere = 0
    if surveyed:
        own = _own_walls(o, surveyed)
        elsewhere = len(surveyed) - len(own)
        walls = [_wall_row(w, fr, i) for i, w in enumerate(own) if w['length_m'] >= min_facade_m]
    else:
        walls = [_wall_row({**w, 'a': w['a'].tolist(), 'b': w['b'].tolist(), 'side': i}, fr, i)
                 for i, w in enumerate(facade.walls(fp, min_m=min_facade_m))]
    if part and part.get('footprint'):   # the walls along this part's outline
        walls = [w for w in walls if facade.belongs(part['footprint'], fr.to_scan(w['x0'], w['y0']),
                                                    fr.to_scan(w['x1'], w['y1']))]
    parts = [_part_row(objs[pid], fr) for pid in o.get('parts', []) if pid in objs]
    on_roof = [{'id': x['id'], 'cls': x['cls'], 'conf': x['conf']} for x in objs.values() if x.get('on') == o['id']]
    x, y = fr.from_scan(o['x'], o['y'])
    ridge = o.get('ridge_yaw_deg')
    looked = [w for w in walls if w['status'] == 'surveyed']
    storeys = [w['storeys'] for w in looked if w['storeys']]
    tones = [w['tone'] for w in (own if surveyed else []) if w.get('tone') and w.get('status') == 'surveyed']
    limitations = ['Footprint traces the roof (eaves overhang the walls by ~0.3-0.8 m).']
    if surveyed:
        limitations += ['Windows / doors are Grounding DINO boxes on a render of the photogrammetry texture: blurred, '
                        'mirrored or tree-covered walls give few or false boxes; look with scan_facade.',
                        'Ground-floor shop fronts are often missed; storeys count window rows.']
    else:
        limitations.append('No facade survey for this building' + (f' ({on_demand})' if on_demand else '') +
                           ': walls listed from the roof outline; scan_facade still renders them.')
    out = {'survey': meta['key'], 'frame': fr.name, 'id': o['id'], 'conf': o['conf'], 'src': o['src'],
           'center': [_r(x), _r(y)], 'ground_z': _r(fr.z_out(o['ground_z'])), 'top_z': _r(fr.z_out(o['top_z'])),
           'height_m': o['height_m'], 'eave_m': eave_of(meta, o, ras), 'ridge_m': o.get('ridge_m'),
           'length_m': o['length_m'], 'width_m': o['width_m'], 'yaw_deg': _r(fr.yaw_out(o['yaw_deg']), 1),
           'area_m2': o['area_m2'], 'roof': o.get('roof'), 'roof_conf': o.get('roof_conf'),
           'ridge_yaw_deg': _r(fr.yaw_out(ridge), 1) if ridge is not None else None,
           'roof_shape': _roof_out(o, fr),
           'colours': {'roof': _colour(o.get('rgb')),
                       'facade': _colour(np.median(np.array(tones), 0).tolist()) if tones else None},
           'osm': o.get('osm'), 'split_by_osm': o.get('split_by_osm', False),
           'footprint': [[_r(a, 2), _r(b, 2)] for a, b in poly_f], 'parts': parts,
           'facades_surveyed': bool(surveyed), 'storeys': int(np.median(storeys)) if storeys else None,
           'windows': sum(w['windows'] or 0 for w in looked) if surveyed else None,
           'doors': sum(w['doors'] or 0 for w in looked) if surveyed else None,
           'facades': walls, 'walls_of_other_parts': elsewhere, 'on_roof': on_roof, 'units': 'metres',
           'meaning': 'heights above ground: eave_m where the roof meets the walls (30th percentile of the outer 1 m '
                      'of the roof), height_m 95th, ridge_m 99th percentile of the roof; '
                      'roof_shape: planes fitted to the roof heights (form, slope_pct per plane, pitch_deg, the ridge or '
                      'valley line with its height and the edge heights): a fall of a few percent stays roof flat; '
                      'facades: walls (from OSM wall lines or the roof outline), faces_deg outward (yaw convention), '
                      'status surveyed / no_view (too little mesh facing it) / party_wall (another building in front) / '
                      'not_surveyed; sill_m / head_m per window row above ground; tone: median wall colour; '
                      'scan_facade ID --side N shows one; a part id answers for that part (footprint, heights, '
                      'roof, its walls) with parent',
           'limitations': limitations}
    if on_demand and surveyed:
        out['facade_note'] = on_demand
    if o.get('osm_conflict'):
        out['osm_conflict'] = o['osm_conflict']
    if part:
        prow = _part_row(part, fr)
        out.update(id=part['id'], cls='building_part', parent=o['id'], part=prow, center=[prow['x'], prow['y']],
                   footprint=prow['footprint'], height_m=part['height_m'], eave_m=part.get('eave_m'),
                   ridge_m=part.get('ridge_m'), length_m=part['length_m'], width_m=part['width_m'],
                   yaw_deg=prow['yaw_deg'], area_m2=part.get('area_m2'), roof=part.get('roof'),
                   ridge_yaw_deg=prow['ridge_yaw_deg'], roof_shape=prow['roof_shape'])
        out['colours']['roof'] = _colour(part.get('rgb'))
    return out


def scan_facade(ident: int, side: int | None = None, labels: bool = True, frame: str = 'auto',
                survey_key: str = '') -> dict:
    from . import view
    meta = store.resolve(survey_key)
    fr = Frame(meta, frame)
    o = _building(meta, ident)
    stored = _facade_record(meta).get(str(o['id']))
    if stored:
        walls = [dict(w) for w in _own_walls(o, stored)]
        false_party = _false_party_walls(meta, o, walls, Rasters(meta))
        for w in walls:     # only the building's own eaves stood in front (older surveys): not a party wall
            if w['side'] in false_party:
                w['status'] = 'not_surveyed'
    if not stored or not walls:
        ras = Rasters(meta)
        rings, shift = _osm_rings(meta)
        walls = vision.plan_facades(meta, [o], ras, rings, shift, only={o['id']})
        for w in walls:
            w.update(status='not_surveyed', detections=[], grid=None)
    if not walls:
        raise ValueError(f"building {o['id']} has no wall of at least {facade.MIN_WALL_M:g} m")
    sides = [w['side'] for w in walls]
    if side is not None:
        if side not in sides:
            raise ValueError(f"building {o['id']} has sides {sides}; scan_building {o['id']} lists them")
        walls = [w for w in walls if w['side'] == side]
    elif len(walls) > 6:
        walls = sorted(walls, key=lambda w: -w['length_m'])[:6]
        walls.sort(key=lambda w: w['side'])
    imgs, _deps, _st = vision.render_facades(meta, walls)
    path = view.facades(meta, o, walls, imgs, labels)
    rows = [_wall_row(w, fr, i) for i, w in enumerate(walls)]
    return {'survey': meta['key'], 'frame': fr.name, 'building': o['id'], 'image': str(path), 'facades': rows,
            'sides_all': sides, 'units': 'metres',
            'meaning': 'one panel per wall, seen from outside (left to right along the wall); boxes: window cyan, '
                       'door yellow, garage_door orange, balcony magenta; lines: window rows with sill / head above '
                       'ground; grid every metre-step along the wall and above ground',
            'note': 'image in RAM (/dev/shm) only: look at it locally, never upload or commit it'}


def _hit(ras: Rasters, objs: dict, e: float, s: float, top: float, oid: int, code: int, radius_m: float) -> dict:
    """What a blocked path sample hits: the object and class of the highest cell under it. A raised cell without an
    object id (a roof edge or wall top the object outlines left out) names the nearest object within 1.5 m."""
    gz = float(ras.ground(np.array([e]), np.array([s]))[0])
    above = top - gz
    o = objs.get(oid)
    how = 'cell'
    if o is None and above > survey.LOW_M:
        g = ras.g
        c, r = g.to_cell(e, s)
        r, c = int(round(float(r))), int(round(float(c)))
        k = int(math.ceil((1.5 + radius_m) / g.res))
        win = np.asarray(ras.objid[max(0, r - k):r + k + 1, max(0, c - k):c + k + 1])
        ids = win[win >= 0]
        if ids.size:
            o = objs.get(int(np.bincount(ids).argmax()))
            how = 'nearest within 1.5 m'
    cls = survey.CLASSES.get(code, '?')
    if code in survey.GROUND_CODES and above > survey.HIGH_M:
        cls = f'raised {cls} cell (edge)'
    out = {'object': o['id'] if o else None, 'cls': o['cls'] if o else cls, 'surface': cls,
           'above_ground_m': _r(above)}
    if o is not None and how != 'cell':
        out['object_from'] = how
    return out


def scan_free(points, radius_m: float = 0.0, margin_m: float = 0.5, agl: bool = False, frame: str = 'auto',
              survey_key: str = '') -> dict:
    p = _xy(points)
    if p.shape[1] < 3 or not 2 <= len(p) <= 64:
        raise ValueError('scan_free takes 2 to 64 x,y,z points')
    if not 0 <= radius_m <= 20:
        raise ValueError('radius_m must be 0..20')
    meta, fr, ras = _context(survey_key, frame)
    objs = _obj_index(meta)
    g = ras.g
    k = int(math.ceil(radius_m / g.res))
    seg = np.linalg.norm(np.diff(p[:, :3], axis=0), axis=1)
    total = float(seg.sum())
    if total > 3000:
        raise ValueError('path longer than 3 km')
    n = max(2, int(math.ceil(total / (g.res / 2))) + 1)
    d = np.linspace(0, total, n)
    cum = np.r_[0, np.cumsum(seg)]
    pts = np.stack([np.interp(d, cum, p[:, j]) for j in range(3)], 1)
    e, s = fr.to_scan(pts[:, 0], pts[:, 1])
    if agl:   # z values are heights above the ground model: the line follows the ground between the points
        zs = np.asarray(ras.ground(e, s), float) + pts[:, 2]
        pts[:, 2] = fr.z_out(zs)
    else:
        zs = fr.z_in(pts[:, 2])
    c, r = g.to_cell(np.asarray(e), np.asarray(s))
    ci, ri = np.rint(c).astype(int), np.rint(r).astype(int)
    inside = (ci >= 0) & (ci < g.w) & (ri >= 0) & (ri < g.h)
    top = np.full(n, -np.inf)
    who = np.full(n, -1)
    wcls = np.zeros(n, np.uint8)
    offs = [(dy, dx) for dy in range(-k - 1, k + 2) for dx in range(-k - 1, k + 2)
            if math.hypot(dy, dx) * g.res <= radius_m + g.res * 0.75]
    for dy, dx in offs:
        rr, cc = ri + dy, ci + dx
        ok = inside & (rr >= 0) & (rr < g.h) & (cc >= 0) & (cc < g.w)
        rr, cc = np.clip(rr, 0, g.h - 1), np.clip(cc, 0, g.w - 1)
        v = np.where(ok, np.asarray(ras.dsm[rr, cc], float), -np.inf)
        v = np.where(np.isnan(v), -np.inf, v)
        better = v > top
        top = np.where(better, v, top)
        who = np.where(better, np.asarray(ras.objid[rr, cc]), who)
        wcls = np.where(better, np.asarray(ras.cls[rr, cc]), wcls)
    top[~np.isfinite(top)] = np.nan
    clear_m = zs - top
    known = np.isfinite(clear_m)
    blocked = known & (clear_m < margin_m)
    out = {**_head(meta, fr, ras), 'length_m': _r(total, 1), 'samples': n, 'radius_m': radius_m, 'margin_m': margin_m,
           'z_in': 'metres above the ground model (line follows the ground)' if agl else f'{fr.name} z',
           'clear': bool(known.all() and not blocked.any()), 'no_data_samples': int((~known).sum())}
    if known.any():
        j = int(np.nanargmin(np.where(known, clear_m, np.inf)))
        out['min_clearance_m'] = _r(clear_m[j])
        out['min_at'] = {'d_m': _r(d[j], 1), 'x': _r(pts[j, 0]), 'y': _r(pts[j, 1]), 'z': _r(pts[j, 2]),
                         'surface_z': _r(fr.z_out(top[j]))}
    if blocked.any():
        j = int(np.flatnonzero(blocked)[0])
        hit = _hit(ras, objs, e[j], s[j], top[j], int(who[j]), int(wcls[j]), radius_m)
        out['first_blocked'] = {'d_m': _r(d[j], 1), 'x': _r(pts[j, 0]), 'y': _r(pts[j, 1]), 'z': _r(pts[j, 2]),
                                'surface_z': _r(fr.z_out(top[j])), **hit}
        out['blocked_length_m'] = _r(blocked.sum() * total / max(n - 1, 1), 1)
    out['units'] = 'metres'
    out['limitations'] = ['Against the scan surface (DSM) only, not placed objects: use check_route_clearance / '
                          'check_opening in the game.',
                          'The DSM is the highest surface: a path under a bridge or through an arch reads as blocked.']
    return out


def scan_view(box=None, center=None, size_m: float = 60.0, layer: str = 'ortho', labels: bool = True,
              classes=None, px: int = 900, frame: str = 'auto', survey_key: str = '', compare: str = '',
              deviations: int = 25) -> dict:
    from . import view
    if compare:
        if box is not None or center is not None:
            raise ValueError('compare takes box and pixel grid from the capture_topdown image: drop box / center')
        from . import topdown
        return topdown.compare_view(compare, layer=layer.lower(), labels=labels, classes=classes, frame=frame,
                                    survey_key=survey_key, deviations=deviations)
    meta, fr, ras = _context(survey_key, frame)
    if box is not None and center is not None:
        raise ValueError('give box or center (+ size_m), not both')
    if box is not None:
        b = _box4(box)
    else:
        cxy = _xy(center if center is not None else fr.from_scan(*meta['center']), 'center')[0]
        if not 2 <= size_m <= 600:
            raise ValueError('size_m must be 2..600')
        b = (cxy[0] - size_m / 2, cxy[1] - size_m / 2, cxy[0] + size_m / 2, cxy[1] + size_m / 2)
    if b[2] - b[0] < 1 or b[3] - b[1] < 1 or max(b[2] - b[0], b[3] - b[1]) > 600:
        raise ValueError('box sides must be 1..600 m')
    layer = layer.lower()
    if layer not in view.LAYERS:
        raise ValueError(f"layer must be one of {', '.join(view.LAYERS)}")
    px = max(200, min(int(px), 1600))
    want = {c.strip().lower() for c in (classes or []) if c.strip()} or None
    objs = store.objects(meta)
    info: dict = {}
    path, drawn, mpp = view.render(meta, fr, ras, objs, b, layer, labels, want, px, info=info)
    shared, over = _shared_osm(objs), _overlaps(objs)
    legend = [{'id': o['id'], 'cls': o['cls'], 'conf': o['conf'], 'height_m': o['height_m'],
               'flag': _flag(o, shared, over)}
              for o in sorted(drawn, key=lambda o: o['id'])[:VIEW_ROWS]]
    return {**_head(meta, fr, ras), 'image': str(path), 'layer': layer, 'box': [_r(v) for v in b],
            'm_per_px': _r(mpp, 3), **info, 'objects_drawn': len(drawn), 'objects': legend,
            'objects_truncated': len(drawn) > VIEW_ROWS, 'units': 'metres',
            'note': 'image in RAM (/dev/shm) only: look at it locally, never upload or commit it; objects: the ids '
                    'drawn on it (scan_objects for positions)'}


# ---------------------------------------------------------------------------- raster export

GRID_LAYERS = ('dtm', 'top', 'height', 'class', 'object')
GRID_MAX_CELLS = 1_000_000


def scan_grid(box, res_m: float = 1.0, layers=None, out: str = '', frame: str = 'auto', survey_key: str = '') -> dict:
    """Raster export of a box on a regular grid in the result frame (row 0 = y0, column 0 = x0, cell centres at
    x0 + (i + .5) res_m): per cell ground z (mean), top z (max), height above ground, the class and object id of
    the highest sub-sample. Written as one compressed .npz (arrays named after the layers) plus a .json header;
    returns the paths and a summary (ranges, class shares, the objects with the most cells)."""
    x0, y0, x1, y1 = _box4(box)
    res = float(res_m)
    if not 0.1 <= res <= 20:
        raise ValueError('res_m must be 0.1..20 metres')
    W, H = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
    if W * H > GRID_MAX_CELLS:
        raise ValueError(f'{W} x {H} cells: at most {GRID_MAX_CELLS:,} (larger res_m or a smaller box)')
    want = [w.strip().lower() for w in (layers or GRID_LAYERS) if w.strip()]
    bad = sorted(set(want) - set(GRID_LAYERS))
    if bad:
        raise ValueError(f"unknown layers {', '.join(bad)}: choose from {', '.join(GRID_LAYERS)}")
    meta, fr, ras = _context(survey_key, frame)
    g = ras.g
    k = max(1, min(4, int(math.ceil(res / g.res)), int(math.sqrt(4e6 / (W * H)))))   # sub-samples per cell side
    off = (np.arange(k) + .5) / k
    cx = x0 + (np.arange(W)[:, None] + off[None, :]) * res     # W x k
    cy = y0 + (np.arange(H)[:, None] + off[None, :]) * res     # H x k
    X = np.broadcast_to(cx.reshape(1, W, 1, k), (H, W, k, k))
    Y = np.broadcast_to(cy.reshape(H, 1, k, 1), (H, W, k, k))
    e, s = fr.to_scan(X, Y)
    r, c, inside = ras.cell(e, s)
    top = np.where(inside, np.asarray(ras.dsm[r, c], np.float32), np.nan)
    gz = np.where(inside, np.asarray(ras.dtm[r, c], np.float32), np.nan)
    flat = (H, W, k * k)
    top, gz = top.reshape(flat), gz.reshape(flat)
    valid = np.isfinite(top)
    any_ok = valid.any(2)
    with np.errstate(invalid='ignore'):
        gmean = np.where(any_ok, np.nanmean(np.where(valid, gz, np.nan), 2), np.nan)
        tmax = np.where(any_ok, np.nanmax(np.where(valid, top, -np.inf), 2), np.nan)
    j = np.argmax(np.where(valid, top, -np.inf), 2)[..., None]
    cls = np.take_along_axis(np.asarray(ras.cls[r, c]).reshape(flat), j, 2)[..., 0]
    oid = np.take_along_axis(np.asarray(ras.objid[r, c]).reshape(flat), j, 2)[..., 0]
    cls = np.where(any_ok, cls, 0).astype(np.uint8)
    oid = np.where(any_ok, oid, -1).astype(np.int32)
    arrays = {'dtm': np.asarray(fr.z_out(gmean), np.float32), 'top': np.asarray(fr.z_out(tmax), np.float32),
              'height': (tmax - gmean).astype(np.float32), 'class': cls, 'object': oid}
    if out:
        path = Path(out).expanduser()
        if path.suffix != '.npz':
            path = path / f"grid-{time.strftime('%H%M%S')}.npz" if path.is_dir() or not path.suffix else \
                path.with_suffix('.npz')
        if path.exists():
            raise ValueError(f'{path} exists: give a new file name')
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        d = store.ram(meta['key']) / 'grids'
        d.mkdir(parents=True, exist_ok=True)
        for old in sorted(d.glob('*.npz'), key=store._mtime, reverse=True)[store.KEEP_VIEWS - 1:]:
            old.unlink(missing_ok=True)
            old.with_suffix('.json').unlink(missing_ok=True)
        path = d / f"grid-{time.strftime('%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.npz"
    np.savez_compressed(path, **{n: arrays[n] for n in want})
    header = {'survey': meta['key'], 'frame': fr.name, 'x0': x0, 'y0': y0, 'res_m': res, 'width': W, 'height': H,
              'layers': want, 'units': 'metres (z in the frame; height above the scan ground)',
              'layout': 'array[row, col]: row 0 at y0, col 0 at x0, cell centre x0 + (col + .5) res_m',
              'class_codes': {str(v): n for v, n in survey.CLASSES.items()},
              'object': 'scan object id (-1 none): scan_objects / scan_building answer for it',
              'no_data': 'NaN heights, class 0, object -1'}
    store.write_json(path.with_suffix('.json'), header)
    ok = np.isfinite(arrays['height'])
    shares = {}
    if ok.any():
        vals, cnt = np.unique(cls[ok], return_counts=True)
        shares = {survey.CLASSES.get(int(v), str(v)): round(float(n / ok.sum()), 3) for v, n in zip(vals, cnt)}
    objs = _obj_index(meta)
    ids, cnt = np.unique(oid[oid >= 0], return_counts=True)
    order = np.argsort(-cnt)[:15]
    tops = [{'id': int(ids[i]), 'cls': objs.get(int(ids[i]), {}).get('cls'), 'cells': int(cnt[i])} for i in order]

    def rng(a):
        a = a[np.isfinite(a)]
        return [_r(float(a.min())), _r(float(np.median(a))), _r(float(a.max()))] if a.size else None

    return {'survey': meta['key'], 'frame': fr.name, 'file': str(path), 'header': str(path.with_suffix('.json')),
            'grid': {'x0': x0, 'y0': y0, 'res_m': res, 'width': W, 'height': H, 'sub_samples': k * k},
            'layers': want, 'no_data_cells': int((~ok).sum()),
            'summary': {'ground_z_min_med_max': rng(arrays['dtm']), 'top_z_min_med_max': rng(arrays['top']),
                        'height_m_min_med_max': rng(arrays['height']),
                        'class_share': dict(sorted(shares.items(), key=lambda kv: -kv[1])),
                        'objects': len(ids), 'largest_objects': tops},
            'units': 'metres',
            'read': "numpy: d = np.load(file); d['top'][row, col]; the .json header has the grid and class codes",
            'note': ('in RAM (/dev/shm): derived from the scan, keep it on this machine' if not out else
                     'derived from the scan: keep it on this machine, do not commit it')}
