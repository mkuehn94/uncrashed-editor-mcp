"""From scan objects to the game: catalogue matches (scan_match), reviewable placement plans (scan_plan), a check of
a built scene against the scan (scan_compare) and contact sheets of scan crops (scan_sheet).

Nothing here talks to the game except scan_compare without a snapshot file, which reads the placed objects with the
read-only `groups` export. Plans are written to a file and never executed.

Catalogue facts come from catalog.load_catalog() (the rows search_assets / get_assets show): size_m, origin_cm /
extent_cm (mesh bounds relative to the pivot), pivot kind and long axis. The pivot of a spawn is computed with
geom.pivot_for_anchor(..., 'bottom_center') so the mesh box centre lands on the object centre and its bottom on the
ground, whatever the row's pivot rule (base, centre, top, offset). Roof shapes of catalogue buildings are read from the
simplified LOD0 hulls in data/meshes.bin (Blueprint rows have none: roof unknown).
"""
from __future__ import annotations

import json
import math
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import fit, geom
from ..catalog import load_catalog
from . import composite, ops, store, survey
from .store import Frame

# ---------------------------------------------------------------------------- what may stand in for what

PLAN_CLASSES = ('car', 'truck', 'tree', 'building')
MATCHABLE = ('car', 'truck', 'boat', 'tree', 'shrub', 'building', 'pool', 'low_structure', 'small_object')
NOT_MATCHABLE = {'bridge': 'no bridge rows in the catalogue: build it from walls / floors',
                 'solar_panel': 'no solar panel rows in the catalogue',
                 'building_part': 'parts belong to their building: match the parent',
                 'door': 'doors belong to their wall: see scan_building / scan_facade'}
# street-view classes -> words in catalogue row names
STREET_WORDS = {'street_lamp': ('lamp', 'streetlight', 'street light', 'lantern', 'electric pole'),
                'bench': ('bench',), 'bin': ('trash', 'garbage', 'dumpster', 'waste', 'wheelie', 'bin'),
                'bollard': ('bollard',), 'sign': ('sign',), 'fence': ('fence',), 'bicycle': ('bike', 'bicycle')}
BUILDING_SKIP = ('rooftop', 'dome', 'walls', 'crane', 'balloon', 'roof', 'rig', 'silo', 'bus stop', 'annex', 'arch',
                 'deck', 'tribune', 'windmill', 'turbine', 'food', 'bower', 'christmas')
VEHICLE_SKIP = ('bus stop', 'wheel', 'crane', 'carriage', 'boat', 'spawner')
CAR_NAMES = ('gt86',)  # cars the catalogue tags as props
# (mode, scale low, scale high, weights of the x / y / z size residuals)
SCALING = {'car': ('uniform', .85, 1.15, (1, 1, .5)), 'truck': ('uniform', .75, 1.3, (1, 1, .6)),
           'boat': ('uniform', .5, 3.0, (1, 1, .3)), 'tree': ('tree', .4, 2.5, (1, 1, 1.5)),
           'shrub': ('tree', .3, 3.0, (1, 1, 1)), 'building': ('free', .5, 2.0, (1, 1, 1)),
           'pool': ('flat', .2, 10.0, (1, 1, 0)), 'low_structure': ('free', .6, 1.6, (1, 1, 1)),
           'small_object': ('uniform', .6, 1.6, (1, 1, 1))}
COLOUR_WORDS = {'green': (70, 120, 60), 'yellow': (210, 180, 40), 'red': (170, 40, 35), 'blue': (40, 70, 150),
                'white': (220, 220, 220), 'black': (30, 30, 30), 'grey': (128, 128, 128), 'gray': (128, 128, 128),
                'orange': (220, 120, 30), 'brown': (110, 80, 50)}
GROUND_NAMES = ('paved', 'road', 'parking', 'path', 'bare', 'other_ground', 'grass')
LIMITATIONS_MATCH = [
    'Matches compare box sizes (length, width, height), aspect and roof shape only; the catalogue has no colours '
    'except colour words in a few row or material names, so colour rarely decides.',
    'A stand-in is the nearest catalogue shape, not the real object: look at it next to scan_view.',
    'Vehicle heading is known only up to 180 degrees (front / back look alike from above).',
]


def _role(asset) -> str:
    return fit.role_of(asset)


def _wanted(cls: str, a, role: str) -> bool:
    name, tags = a.name.lower(), set(a.tags)
    if a.category in ('Interractive', 'Decals'):
        return False
    if cls in ('car', 'truck'):
        if any(w in name for w in CAR_NAMES):
            return True
        if role != 'vehicle' or any(w in name for w in VEHICLE_SKIP):
            return False
        return cls == 'truck' or not any(w in name for w in ('train', 'tractor', 'fork'))
    if cls == 'boat':
        return 'boat' in name
    if cls == 'tree':
        return role == 'foliage' and 'tree' in tags and a.size_m[2] >= 3
    if cls == 'shrub':
        return role == 'foliage' and 'grass' not in name and ('bush' in tags or ('tree' in tags and a.size_m[2] < 3))
    if cls == 'building':
        return role in ('building', 'landmark') and not any(w in name for w in BUILDING_SKIP)
    if cls == 'pool':
        return 'water' in tags and 'water' in name and 'tank' not in name
    if cls in STREET_WORDS:
        words = re.findall(r'[a-z]+', name)
        words += [w[:-1] for w in words if w.endswith('s')]   # plural row names (signs, benches -> benche: no harm)
        return a.kind == 'mesh' and a.size_m[2] < 15 and any(
            (w in name) if ' ' in w else (w in words) for w in STREET_WORDS[cls])
    if cls in ('low_structure', 'small_object'):
        if 'bus stop' in name:
            return cls == 'low_structure'
        if any(w in name for w in CAR_NAMES):
            return False
        return role in ('industrial', 'prop', 'street', 'barrier') and a.size_m[2] < 4.5 and max(a.size_m[:2]) < 15
    return False


def look_conflict(cls: str, a) -> str | None:
    """Why a row's measured look (asset_profile) contradicts the class, or None. Trees: no geometry found (the row
    may render as nothing) or foliage down to the ground with no trunk and a solid silhouette (a rock or a hedge
    block, e.g. a 'tree' row that renders as a rock). Only rows measured from the game's LOD0 are judged."""
    if cls not in ('tree',) or not USE_PROFILES or a.kind != 'mesh':
        return None
    from . import profile
    p = profile.get(a.id)
    if not p:
        return None
    if p.get('source') == 'none':
        return 'no mesh geometry found (may render as nothing)'
    if not str(p.get('source', '')).startswith('lod0'):
        return None
    if p.get('crown_base_m') is not None and p['crown_base_m'] < TRUNK_MIN_M and (p.get('crown_frac') or 0) > .9:
        return f"solid from the ground (crown base {p['crown_base_m']:g} m, crown fills {p['crown_frac']:g} of its box)"
    return None


TRUNK_MIN_M = 0.3


def candidates(cls: str, include_hidden: bool = False, rows=None, excluded: list | None = None) -> list:
    """Catalogue rows that may stand for class cls; rows whose measured look contradicts the class are left out
    (listed in `excluded` as {row, name, why} when given)."""
    cat = load_catalog()
    if rows:
        missing = [r for r in rows if r not in cat.assets]
        if missing:
            raise ValueError(f"unknown catalogue rows: {', '.join(missing)} (search_assets finds rows)")
        return [cat.assets[r] for r in rows]
    out = []
    for a in cat.assets.values():
        if not ((include_hidden or not a.hidden) and min(a.size_m) > 0 and _wanted(cls, a, _role(a))):
            continue
        why = look_conflict(cls, a)
        if why:
            if excluded is not None:
                excluded.append({'row': a.id, 'name': a.name, 'cls': cls, 'why': why})
            continue
        out.append(a)
    return out


VAN_H_M = 1.75       # a scan vehicle this tall (or a truck) wants a van-bodied row
VAN_ROW_H_M = 1.7    # rows this tall are vans / trucks
BODY_PENALTY = .6


def body(o: dict) -> str | None:
    """'van' or 'car' for a scan vehicle from its class, height and length; None when the height is not a vehicle's
    (a crown or roof over it)."""
    if o['cls'] == 'truck':
        return 'van'
    h = float(o.get('height_m') or 0)
    if h > 4.0:
        return None
    return 'van' if h >= VAN_H_M or o['length_m'] >= 5.4 else 'car'


def row_body(a) -> str:
    return 'van' if a.size_m[2] >= VAN_ROW_H_M else 'car'


# ---------------------------------------------------------------------------- catalogue roofs


@lru_cache(maxsize=512)
def mesh_roof(row: str) -> dict | None:
    """Roof of a catalogue mesh from its simplified hull: type, share, ridge direction in the row's local frame
    (yaw convention, degrees mod 180). None for Blueprints / rows without a hull."""
    from .. import meshes
    a = load_catalog().assets.get(row)
    if a is None or a.kind != 'mesh' or not a.source:
        return None
    raw = meshes.load().get(a.source)
    if raw is None or not raw.tris:
        return None
    V = np.asarray(raw.verts, float)
    T = np.asarray(raw.tris, int)
    p0, p1, p2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    area = np.linalg.norm(n, axis=1) / 2
    ok = area > 1e-6
    n = n[ok] / (2 * area[ok, None])
    n *= np.where(n[:, 2:3] < 0, -1, 1)           # hull winding is not reliable: look from above
    p0, p1, p2, area = p0[ok], p1[ok], p2[ok], area[ok]
    # sample every triangle about every 0.35 m; keep the highest sample per 0.5 m cell = what is seen from above
    k = np.clip(np.ceil(area / 0.12), 1, 400).astype(int)
    idx = np.repeat(np.arange(len(area)), k)
    rng = np.random.default_rng(7)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1
    u, v = np.where(flip, 1 - u, u), np.where(flip, 1 - v, v)
    pts = p0[idx] + u[:, None] * (p1[idx] - p0[idx]) + v[:, None] * (p2[idx] - p0[idx])
    z0 = V[:, 2].min()
    cell = np.floor(pts[:, :2] / 0.5).astype(np.int64)
    key = (cell[:, 0] - cell[:, 0].min()) * 100000 + (cell[:, 1] - cell[:, 1].min())
    order = np.lexsort((-pts[:, 2], key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    top = order[first]
    top = top[pts[top, 2] > z0 + survey.HIGH_M]
    if len(top) < 6:
        return None
    kind, share, ridge = survey._roof_type(n[idx[top]])
    return {'roof': kind, 'share': round(share, 2), 'ridge_deg': ridge}


# ---------------------------------------------------------------------------- fitting


def _target(o: dict) -> tuple[float, float, float]:
    h = o.get('ridge_m') if o['cls'] == 'building' and o.get('ridge_m') else o['height_m']
    return max(o['length_m'], .05), max(o['width_m'], .05), max(float(h), .05)


def _colour_of(a) -> tuple | None:
    text = f'{a.name} {a.material or ""}'.lower()
    for word, rgb in COLOUR_WORDS.items():
        if re.search(rf'\b{word}\b|_{word}', text):
            return rgb
    return None


def _scales(mode: str, lo: float, hi: float, r: np.ndarray, w: np.ndarray) -> np.ndarray:
    lg = np.log(r)
    if mode == 'uniform':
        s = math.exp(float((w * lg).sum() / w.sum()))
        return np.full(3, min(max(s, lo), hi))
    if mode == 'tree':
        sz = min(max(float(r[2]), lo), hi)
        sxy = min(max(math.exp(float(lg[:2].mean())), lo, sz * TREE_SPREAD[0]), hi, sz * TREE_SPREAD[1])
        return np.array([sxy, sxy, sz])
    if mode == 'flat':
        return np.array([min(max(float(r[0]), lo), hi), min(max(float(r[1]), lo), hi), 1.0])
    return np.clip(r, lo, hi)


# crown scale / height scale of foliage: beyond this a row is visibly squashed or stretched (a 0.3-high willow
# 1.6 x as wide was the best 'tree' for wide low crowns on the example scan); the fit error then shows it
TREE_SPREAD = (.75, 1.35)
SOFT_PENALTY = {'palm': .6}  # exotic stand-ins: only when nothing else fits
SHAPE_WEIGHT = {'free': .6, 'tree': .3}  # per-axis scaling hides proportions: charge the shape error too


def _rms(e: np.ndarray, w: np.ndarray) -> float:
    """Weighted RMS of log ratios as a relative error (0.1 = 10 %)."""
    return math.expm1(math.sqrt(float((w * e * e).sum() / w.sum())))


VISIBLE_CLASSES = ('car', 'truck', 'boat', 'tree', 'shrub', 'low_structure', 'small_object')


def row_dims(a, cls: str) -> dict:
    """What the fit measures a row by. With an asset profile (VISIBLE_CLASSES): the visible size, measured the way the
    survey measures scan objects (top silhouette extents, 95th percentile top; foliage: crown = equivalent diameter
    of the silhouette), and the visible long axis. Else the bounding box. Scaling a tree's bounding box to a scan
    crown builds it too small: the box reaches the outermost leaf cards (crown_frac 0.7..0.95 of the box)."""
    sx, sy, sz = (float(v) for v in a.size_m)
    box = {'size': (sx, sy, sz), 'long_deg': 90.0 if sy > sx else 0.0, 'crown': None, 'period': None,
           'offset': (0.0, 0.0), 'by': 'box'}
    if cls not in VISIBLE_CLASSES or not USE_PROFILES:
        return box
    from . import profile
    p = profile.get(a.id)
    if not p or p.get('source') in (None, 'none'):
        return box
    vx, vy, vh = p['visible_m']
    long_deg = p['long_axis_deg'] if p['long_axis_deg'] in (0.0, 90.0) else box['long_deg']
    return {'size': (vx, vy, vh), 'long_deg': long_deg, 'crown': p.get('crown_m'), 'period': p.get('yaw_period_deg'),
            'offset': tuple(p.get('visible_center_m') or (0.0, 0.0)), 'by': 'visible'}


USE_PROFILES = True


def fit_one(o: dict, a, mode=None) -> dict:
    """Best yaw, scale and fit error of catalogue row `a` for scan object `o` (scan frame). Sizes are visible sizes
    (row_dims): trees match crown to crown and top to top, everything else silhouette to oriented box."""
    cls = o['cls']
    mode, lo, hi, w = SCALING.get(cls, ('uniform', .5, 2.0, (1, 1, 1))) if mode is None else mode
    w = np.asarray(w, float)
    L, W, H = _target(o)
    dims = row_dims(a, cls)
    sx, sy, sz = dims['size']
    if cls in ('tree', 'shrub') and dims['crown']:
        crown = float(o.get('crown_m') or (L + W) / 2)
        L = W = crown
        sx = sy = float(dims['crown'])
    long_yaw = (o['yaw_deg'] - dims['long_deg']) % 360
    yaws = [(long_yaw, 'long axis')]
    roof = mesh_roof(a.id) if cls == 'building' else None
    o_ridge = o.get('ridge_yaw_deg')
    if roof and roof['ridge_deg'] is not None and o_ridge is not None:
        yaws.insert(0, ((o_ridge - roof['ridge_deg']) % 360, 'ridge'))
    best = None
    k_shape = SHAPE_WEIGHT.get(mode, 0.0)
    for psi, how in yaws:
        d = (psi - o['yaw_deg']) % 180
        tx, ty = (L, W) if d < 45 or d > 135 else (W, L)
        r = np.array([tx / sx, ty / sy, H / sz])
        s = _scales(mode, lo, hi, r, w)
        err = _rms(np.log(r / s), w)
        shape = _rms(np.log(r) - float((w * np.log(r)).sum() / w.sum()), w)
        cost = err + k_shape * shape
        if best is None or cost < best['cost'] - 1e-9:
            best = {'yaw': psi, 'yaw_from': how, 's': s, 'err': err, 'shape': shape, 'cost': cost,
                    'raw': _rms(np.log(r), w)}
    notes = []
    factor = 1.0
    if cls == 'building':
        o_roof = o.get('roof')
        if o_roof == 'valley':      # a low valley roof: a flat-roof stand-in comes closest
            o_roof = 'flat'
        if roof is None:
            factor *= .8
            notes.append('roof unknown (Blueprint)' if a.kind == 'blueprint' else 'roof unknown')
        elif o_roof in ('flat', 'gabled', 'hipped', 'mono_pitch'):
            sloped = {'gabled', 'hipped', 'mono_pitch'}
            if roof['roof'] == o_roof:
                if o_roof == 'gabled' and best['yaw_from'] != 'ridge':
                    factor *= .85
                    notes.append('ridge turned')
            elif roof['roof'] in sloped and o_roof in sloped:
                factor *= .8
                notes.append(f"roof {roof['roof']}")
            else:
                factor *= .55
                notes.append(f"roof {roof['roof']}")
    if mode in ('free', 'tree'):
        factor *= math.exp(-abs(float(np.log(best['s']).mean())) / 2.5)
    rgb = _colour_of(a)
    if rgb is not None and o.get('rgb') and cls not in ('tree', 'shrub'):
        dist = float(np.linalg.norm(np.subtract(o['rgb'], rgb))) / 441.7
        factor *= 1 - .4 * dist
        notes.append('colour named')
    if cls in ('car', 'truck'):
        want = body(o)
        if want and row_body(a) != want:
            factor *= BODY_PENALTY
            notes.append(f'{row_body(a)} body for a {want}')
    for word, f in SOFT_PENALTY.items():
        if word in a.name.lower():
            factor *= f
            notes.append(word)
    if a.hidden:
        factor *= .97
    score = math.exp(-best['cost'] / .25) * factor
    scaled = [round(a.size_m[i] * float(best['s'][i]), 2) for i in range(3)]
    visible = [round(dims['size'][i] * float(best['s'][i]), 2) for i in range(3)]
    if cls in ('tree', 'shrub') and dims['crown']:
        visible[:2] = [round(float(dims['crown']) * float(best['s'][0]), 2)] * 2
    return {'row': a.id, 'name': a.name, 'score': round(score, 3), 'fit_err_pct': round(best['err'] * 100, 1),
            'shape_err_pct': round(best['shape'] * 100, 1), 'raw_err_pct': round(best['raw'] * 100, 1),
            'scale': [round(float(v), 3) for v in best['s']], 'yaw_scan': best['yaw'], 'yaw_from': best['yaw_from'], 'size_m': scaled,
            'visible_m': visible, 'sized_by': dims['by'], 'yaw_period_deg': dims['period'],
            'visible_offset_m': dims['offset'],
            'roof': roof['roof'] if roof else None, 'pivot': a.pivot, 'notes': ', '.join(notes) or None}


def ranked(o: dict, rows: list, top: int) -> list[dict]:
    out = [fit_one(o, a) for a in rows]
    out.sort(key=lambda c: (-c['score'], c['fit_err_pct'], c['row']))
    return out[:top]


def _cand_row(c: dict, fr: Frame, tree: bool = False) -> dict:
    return {'row': c['row'], 'name': c['name'], 'score': c['score'], 'fit_err_pct': c['fit_err_pct'],
            'shape_err_pct': c['shape_err_pct'], 'raw_err_pct': c['raw_err_pct'], 'scale': c['scale'],
            'yaw_deg': None if tree else round(fr.yaw_out(c['yaw_scan'], 360), 1),
            'size_m': c['size_m'], 'roof': c['roof'], 'notes': c['notes']}


# ---------------------------------------------------------------------------- selection


def _select(meta: dict, fr: Frame, classes, box, min_conf: float, default=None) -> tuple[list[dict], list[str]]:
    want = [c.strip().lower() for c in (classes or default or []) if c.strip()]
    b = ops._box4(box) if box is not None else None
    objs = store.objects(meta)
    known = {o['cls'] for o in objs}
    unknown = sorted(set(want) - known - set(MATCHABLE))
    if unknown:
        raise ValueError(f"unknown classes {', '.join(unknown)}; this survey has {', '.join(sorted(known))}")
    out = []
    for o in objs:
        if (want and o['cls'] not in want) or o['conf'] < min_conf:
            continue
        if b:
            x, y = fr.from_scan(o['x'], o['y'])
            if not (b[0] <= x <= b[2] and b[1] <= y <= b[3]):
                continue
        out.append(o)
    return out, want


def scan_match(object_id: int | None = None, classes=None, box=None, min_conf: float = 0.5, top: int = 3,
               limit: int = 30, rows=None, include_hidden: bool = False, frame: str = 'auto',
               survey_key: str = '') -> dict:
    meta = store.resolve(survey_key)
    fr = Frame(meta, frame)
    top = max(1, min(int(top), 10))
    lim = max(1, min(int(limit), 500))
    if object_id is not None:
        if classes or box is not None:
            raise ValueError('give an object id, or classes / box, not both')
        idx = ops._obj_index(meta)
        o = idx.get(int(object_id))
        if o is None:
            raise ValueError(f'no object {object_id} in survey {meta["key"]}')
        if o['cls'] == 'building_part':
            o = idx[o['parent']]
        if o['cls'] in NOT_MATCHABLE:
            raise ValueError(f"object {object_id} is a {o['cls']}: {NOT_MATCHABLE[o['cls']]}")
        pool = candidates(o['cls'], include_hidden, rows)
        best = ranked(o, pool, top)
        tree = o['cls'] in ('tree', 'shrub')
        row = ops._row(o, fr)
        if o['cls'] in ('tree', 'shrub') and slender(o):
            row['warning'] = 'slender: probably roof greenery or a building edge (scan_plan skips it)'
        if o['cls'] == 'building':
            row.update(roof=o.get('roof'), ridge_m=o.get('ridge_m'))
        out = {'survey': meta['key'], 'frame': fr.name, 'object': row, 'rows_considered': len(pool),
               'candidates': [_cand_row(c, fr, tree) for c in best]}
        if best and best[0]['score'] < WEAK_SCORE:
            out['warning'] = (f"weak match: the best row scores {best[0]['score']:g} (< {WEAK_SCORE:g}, shape error "
                              f"{best[0]['shape_err_pct']:g} %): no catalogue row has this size and shape; any "
                              'stand-in is approximate')
        return {**out, 'units': 'metres',
                'meaning': 'score 0..1 (size fit after the suggested scale, proportions, roof); fit_err_pct: size '
                           'error left after scale, shape_err_pct: proportion mismatch (at one uniform scale), '
                           'raw_err_pct: size error at scale 1; scale is x,y,z of the row; yaw_deg: spawn '
                           'yaw that lays the row on the object (trees: any); scan_plan writes the spawn line',
                'limitations': LIMITATIONS_MATCH}
    sel, _want = _select(meta, fr, classes, box, min_conf, default=PLAN_CLASSES)
    pools: dict[str, list] = {}
    by: dict[str, dict] = {}
    rows_out = []
    for o in sel:
        cls = o['cls']
        info = by.setdefault(cls, {'cls': cls, 'objects': 0, 'matched': 0, 'rows': {}, 'errs': [], 'shapes': [],
                                       'scores': [], 'reason': None})
        info['objects'] += 1
        if cls in NOT_MATCHABLE or cls not in SCALING:
            info['reason'] = NOT_MATCHABLE.get(cls, 'no rule for this class')
            continue
        if cls not in pools:
            pools[cls] = candidates(cls, include_hidden, rows)
        if not pools[cls]:
            info['reason'] = 'no catalogue rows for this class'
            continue
        c = ranked(o, pools[cls], 1)[0]
        info['matched'] += 1
        info['rows'][c['row']] = info['rows'].get(c['row'], 0) + 1
        info['errs'].append(c['fit_err_pct'])
        info['shapes'].append(c['shape_err_pct'])
        info['scores'].append(c['score'])
        x, y = fr.from_scan(o['x'], o['y'])
        rows_out.append({'id': o['id'], 'cls': cls, 'conf': o['conf'], 'x': ops._r(x), 'y': ops._r(y),
                         'size_m': [o['length_m'], o['width_m'], _target(o)[2]], 'row': c['row'], 'name': c['name'],
                         'score': c['score'], 'fit_err_pct': c['fit_err_pct'], 'shape_err_pct': c['shape_err_pct'],
                         'scale': c['scale'],
                         'yaw_deg': None if cls in ('tree', 'shrub') else round(fr.yaw_out(c['yaw_scan'], 360), 1)})
    summary = []
    for info in sorted(by.values(), key=lambda r: -r['objects']):
        cat = load_catalog()
        common = sorted(info['rows'].items(), key=lambda kv: -kv[1])[:4]
        summary.append({'cls': info['cls'], 'objects': info['objects'], 'matched': info['matched'],
                        'median_fit_err_pct': round(float(np.median(info['errs'])), 1) if info['errs'] else None,
                        'median_shape_err_pct': round(float(np.median(info['shapes'])), 1) if info['shapes'] else None,
                        'weak': sum(1 for sc in info['scores'] if sc < WEAK_SCORE),
                        'rows': [f'{r} {cat.assets[r].name} x{n}' for r, n in common], 'reason': info['reason']})
    rows_out.sort(key=lambda r: (r['cls'], -r['score'], r['id']))
    return {'survey': meta['key'], 'frame': fr.name, 'selected': len(sel), 'classes': summary,
            'objects': rows_out[:lim], 'truncated': len(rows_out) > lim, 'units': 'metres',
            'meaning': 'per object the best row (scan_match ID shows the next ones); size_m: length, width, height '
                       'of the scan object; fit_err_pct: size error left after scale; shape_err_pct: proportions; '
                       f'weak: objects whose best score is under {WEAK_SCORE:g} (no row fits well)',
            'limitations': LIMITATIONS_MATCH}


# ---------------------------------------------------------------------------- plan


def _ground_around(ras, o: dict) -> str | None:
    """Majority ground class in a ring box around an object (its own cells are low_object / building)."""
    pts = grid_ring(o)
    r, c, inside = ras.cell(pts[:, 0], pts[:, 1])
    cls = np.asarray(ras.cls[r[inside], c[inside]])
    names = [survey.CLASSES.get(int(v), '?') for v in cls]
    ground = [n for n in names if n in GROUND_NAMES or n == 'water']
    if len(ground) < max(3, len(names) // 5):
        return None
    return max(sorted(set(ground)), key=ground.count)


def grid_ring(o: dict) -> np.ndarray:
    a = math.radians(o['yaw_deg'])
    du, dv = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    L, W = o['length_m'] / 2 + 1.2, o['width_m'] / 2 + 1.2
    t = np.linspace(-1, 1, 9)
    edge = [(u, sv) for u in t for sv in (-1, 1)] + [(su, v) for v in t for su in (-1, 1)]
    return np.array([[o['x'] + u * L * du[0] + v * W * dv[0], o['y'] + u * L * du[1] + v * W * dv[1]] for u, v in edge])


WEAK_SCORE = 0.5        # best score under this: no catalogue row really fits (measured: trees 0.43-0.45)
WEAK_FIT_PCT = 40.0     # scan_plan: a pick whose size misses the scan object by more than this is a weak fit


def _weak_fit(cls: str, pick: dict) -> str | None:
    """Why a planned pick should be looked at before it is built (None: fine). Foliage scores low by nature (sparse
    meshes), so only its size error counts."""
    if pick['fit_err_pct'] > WEAK_FIT_PCT:
        return f"size off by {pick['fit_err_pct']:g} % after scaling"
    if cls not in ('tree', 'shrub') and pick['score'] < WEAK_SCORE:
        return f"best row scores {pick['score']:g} (< {WEAK_SCORE:g}): no row has this size and shape"
    return None


SLENDER_REASON = ('slender crown (taller than 3 x its crown, under 4 m wide): on the example scan these were roof '
                  'greenery or building edges, not trees; check with scan_view')


def slender(o: dict) -> bool:
    """A 'tree' far taller than its crown is wide: on the example scan always roof greenery or a building edge."""
    crown = o.get('crown_m') or (o['length_m'] + o['width_m']) / 2
    return o['height_m'] > 3 * crown and min(crown, o['width_m']) < 4


def _out_path(out: str, key: str) -> Path:
    stamp = time.strftime('%Y%m%d-%H%M%S')
    if not out:
        path = Path.cwd() / 'scan-plans' / f'{key}-{stamp}.txt'
    else:
        path = Path(out).expanduser()
        if out.endswith('/') or path.is_dir():
            path = path / f'{key}-{stamp}.txt'
    path = path.absolute()
    if path.suffix.lower() == '.json':
        raise ValueError('out is the plan text file (the .json beside it is written too); give a .txt path')
    for p in (path, path.with_suffix('.json')):
        if p.exists():
            raise ValueError(f'out: {p} exists; give a new file name (plans are never overwritten)')
    return path


def _spawn(row: str, asset, x: float, y: float, z: float, yaw: float, scale, group: str,
           offset=(0.0, 0.0)) -> str:
    """Spawn line whose mesh box bottom is at z and whose visible centre (box centre + offset, row frame metres
    before scale) is at x, y."""
    t = math.radians(yaw)
    ox, oy = offset[0] * scale[0], offset[1] * scale[1]
    x, y = x - (ox * math.cos(t) - oy * math.sin(t)), y - (ox * math.sin(t) + oy * math.cos(t))
    px, py, pz = geom.pivot_for_anchor(asset.origin_cm, asset.extent_cm, x, y, z, t, tuple(scale), 'bottom_center')
    sc = ' '.join(f'{v:.4g}' for v in scale)
    return f'spawn {row} {px * 100:.1f} {py * 100:.1f} {pz * 100:.1f} 0 {yaw % 360:.1f} 0 {sc} group={group}'


WALL_REACH_M = 3.0


def _hint(o: dict, fr: Frame, record: dict, idx: dict) -> tuple[dict, list[str]]:
    from . import facade
    fp = np.asarray(o['footprint'], float)
    px, py = fr.from_scan(fp[:, 0], fp[:, 1])
    walls = record.get(str(o['id']))
    elsewhere = 0
    if walls:
        rows = []
        for i, w in enumerate(walls):
            if w['length_m'] < 2:
                continue
            mx, my = (w['a'][0] + w['b'][0]) / 2, (w['a'][1] + w['b'][1]) / 2
            if not ops._inside(fp, mx, my) and ops._poly_dist(fp, mx, my) > WALL_REACH_M:
                elsewhere += 1  # an OSM outline shared by several split buildings: wall of another part
                continue
            rows.append(ops._wall_row(w, fr, i))
    else:
        rows = [ops._wall_row({**w, 'a': w['a'].tolist(), 'b': w['b'].tolist(), 'side': i}, fr, i)
                for i, w in enumerate(facade.walls(fp, min_m=2))]
    looked = [w for w in rows if w['status'] == 'surveyed' and w['storeys']]
    ridge = o.get('ridge_yaw_deg')
    gz = fr.z_out(o['ground_z'])
    hint = {'id': o['id'], 'conf': o['conf'], 'src': o['src'], 'osm': o.get('osm'),
            'footprint': [[round(float(a), 2), round(float(b), 2)] for a, b in zip(px, py)],
            'ground_z': round(gz, 2), 'eave_m': o.get('eave_m'), 'height_m': o['height_m'], 'ridge_m': o.get('ridge_m'),
            'roof': o.get('roof'), 'ridge_yaw_deg': round(fr.yaw_out(ridge), 1) if ridge is not None else None,
            'length_m': o['length_m'], 'width_m': o['width_m'], 'yaw_deg': round(fr.yaw_out(o['yaw_deg']), 1),
            'storeys': int(np.median([w['storeys'] for w in looked])) if looked else None,
            'parts': [{'id': p, 'roof': idx[p].get('roof'), 'height_m': idx[p]['height_m']} for p in o.get('parts', [])
                      if p in idx],
            'walls': rows, 'walls_of_other_parts': elsewhere}
    lines = [f"# BUILDING {o['id']} conf {o['conf']} src {o['src']}"
             + (f" osm {o['osm'].get('osm')} {o['osm'].get('building', '')} {o['osm'].get('name') or ''}".rstrip()
                if o.get('osm') else ''),
             f"#   ground_z {gz:.2f}  eave {o.get('eave_m')}  height {o['height_m']}  ridge {o.get('ridge_m')} m above "
             f"ground; roof {o.get('roof')}" + (f", ridge yaw {hint['ridge_yaw_deg']}" if ridge is not None else '')
             + f"; box {o['length_m']} x {o['width_m']} m yaw {hint['yaw_deg']}"
             + (f"; storeys {hint['storeys']}" if hint['storeys'] else ''),
             '#   footprint (map m): ' + ' '.join(f'{a:.1f},{b:.1f}' for a, b in hint['footprint'])]
    for w in rows:
        text = (f"#   wall {w['side']}: {w['x0']},{w['y0']} -> {w['x1']},{w['y1']}  {w['length_m']} m faces "
                f"{w['faces_deg']:.0f}  {w['status']}")
        if w['status'] == 'surveyed' and not w['windows']:
            text += ': no windows found'
        elif w['status'] == 'surveyed':
            text += (f"; {w['storeys']} storeys, {w['windows']} windows in {w['columns']} columns"
                     + (f" every {w['spacing_m']} m" if w['spacing_m'] else '')
                     + (f", sill {w['sill_m']} head {w['head_m']}" if w['sill_m'] else '')
                     + (f", doors {w['doors']}" if w['doors'] else ''))
        lines.append(text)
    return hint, lines


# ---------------------------------------------------------------------------- ground cells

# surface class -> catalogue tile (flat, triplanar or tiling material); classes left out are not paved over
GROUND_ROWS = {'road': 'NewRow_182', 'parking': 'NewRow_182', 'paved': 'NewRow_303', 'other_ground': 'NewRow_303',
               'bare': 'NewRow_180', 'path': 'NewRow_185', 'grass': 'NewRow_180'}
GROUND_NOTE = {'grass': 'dirt tile (no lawn tile row): add Grass Foliage NewRow_663 on top for a lawn',
               'water': 'no water surface row: left out'}
GROUND_THICK_M = 0.2
GROUND_SAME = {'parking': 'road', 'other_ground': 'paved', 'bare': 'grass'}   # same tile row: one material


def _rotator(X, Y, Z) -> tuple[float, float, float]:
    """Unreal (pitch, yaw, roll) degrees for local axes X, Y, Z given as map unit vectors."""
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, X[2]))))
    yaw = math.degrees(math.atan2(X[1], X[0]))
    roll = math.degrees(math.atan2(-Y[2], Z[2]))
    return pitch, yaw, roll


def _slab(row: str, asset, corners: np.ndarray, top: np.ndarray, group: str, thick: float = GROUND_THICK_M) -> str:
    """Spawn line of a tile whose top face is the plane fitted through four corner points (map metres, order
    p00 p10 p11 p01 of a cell along u then v); the tile hangs thick metres below its top."""
    P = np.column_stack([corners, top])
    a = (P[1] - P[0] + P[2] - P[3]) / 2          # along u
    b = (P[3] - P[0] + P[2] - P[1]) / 2          # along v
    la, lb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    X = a / la
    Z = np.cross(X, b / lb)
    if Z[2] < 0:
        Z = -Z
    Z /= np.linalg.norm(Z)
    Y = np.cross(Z, X)
    Y /= np.linalg.norm(Y)
    centre_top = P.mean(0)
    ox, oy, oz = (v / 100 for v in asset.origin_cm)
    ex, ey, ez = (max(v, 0.01) / 100 for v in asset.extent_cm)
    sx, sy, sz = la / (2 * ex), lb / (2 * ey), thick / (2 * ez)
    centre = centre_top - Z * thick / 2
    piv = centre - (X * ox * sx + Y * oy * sy + Z * oz * sz)
    p, y, r = _rotator(X, Y, Z)
    return (f'spawn {row} {piv[0] * 100:.1f} {piv[1] * 100:.1f} {piv[2] * 100:.1f} {p:.3f} {y % 360:.3f} {r:.3f} '
            f'{sx:.4f} {sy:.4f} {sz:.4f} group={group}')


def _codes_at(fr: Frame, ras, x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class code, scan ground (map z) and inside-the-survey flag at map points."""
    e, s = fr.to_scan(np.asarray(x, float), np.asarray(y, float))
    r, c, inside = ras.cell(e, s)
    codes = np.where(inside, np.asarray(ras.cls[r, c]), 0).astype(int)
    return codes, np.asarray(fr.z_out(ras.ground(e, s)), float), inside


def _blur(a: np.ndarray, passes: int = 2) -> np.ndarray:
    for _ in range(passes):
        for ax in (0, 1):
            p = np.pad(a, [(1, 1) if k == ax else (0, 0) for k in (0, 1)], mode='edge')
            a = (p[:-2] + p[1:-1] + p[2:]) / 3 if ax == 0 else (p[:, :-2] + p[:, 1:-1] + p[:, 2:]) / 3
    return a


def street_yaw(road: np.ndarray) -> tuple[float | None, float]:
    """Dominant direction of the edges of a road mask (rows along +y, columns along +x): the yaw 0..90 degrees of a
    lattice that runs along the streets, and its strength 0..1 (1: every road edge lies on that lattice)."""
    m = _blur(road.astype(float))
    gy, gx = np.gradient(m)
    w = np.hypot(gx, gy)
    tot = float(w.sum())
    if tot < 10:
        return None, 0.0
    th = 4 * np.arctan2(gy, gx)
    c, s = float((w * np.cos(th)).sum()), float((w * np.sin(th)).sum())
    return round(math.degrees(math.atan2(s, c) / 4) % 90, 1), round(math.hypot(c, s) / tot, 2)


class GroundSurface:
    """The built ground of a plan: per lattice cell the plane of the tile over it (map metres); NaN elsewhere."""

    def __init__(self, yaw: float, gu: np.ndarray, gv: np.ndarray, tiles: list[dict]):
        t = math.radians(yaw)
        self.U, self.V = np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])
        self.gu, self.gv = gu, gv
        self.of = np.full((len(gv) - 1, len(gu) - 1), -1, int)
        self.planes = np.zeros((max(len(tiles), 1), 5))     # uc, vc, z at the centre, dz/du, dz/dv
        for n, tl in enumerate(tiles):
            (u0, u1), (v0, v1) = tl['u'], tl['v']
            z00, z10, z11, z01 = tl['z']
            self.planes[n] = ((u0 + u1) / 2, (v0 + v1) / 2, (z00 + z10 + z11 + z01) / 4,
                              (z10 - z00 + z11 - z01) / 2 / (u1 - u0), (z01 - z00 + z11 - z10) / 2 / (v1 - v0))
            i0, i1 = np.searchsorted(gu, [u0 + 1e-6, u1 - 1e-6]) - 1
            j0, j1 = np.searchsorted(gv, [v0 + 1e-6, v1 - 1e-6]) - 1
            self.of[j0:j1 + 1, i0:i1 + 1] = n

    def z_at(self, x, y) -> np.ndarray:
        """Top of the planned ground at map points (NaN where no tile is planned)."""
        x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
        u, v = x * self.U[0] + y * self.U[1], x * self.V[0] + y * self.V[1]
        h = float(self.gu[1] - self.gu[0])
        i, j = np.floor((u - self.gu[0]) / h).astype(int), np.floor((v - self.gv[0]) / h).astype(int)
        H, W = self.of.shape
        ok = (i >= 0) & (i < W) & (j >= 0) & (j < H)
        n = np.where(ok, self.of[np.clip(j, 0, H - 1), np.clip(i, 0, W - 1)], -1)
        P = self.planes[np.maximum(n, 0)]
        z = P[:, 2] + P[:, 3] * (u - P[:, 0]) + P[:, 4] * (v - P[:, 1])
        return np.where(n >= 0, z, np.nan)


def _pave_box(fr: Frame, ras, box, cell_m: float, flat_tol_m: float, yaw: float, max_tile_m: float,
              smooth_m: float) -> dict:
    """Material key per lattice cell (turned by yaw) from the class raster, then groundfit.pave on the scan
    ground samples (SAMPLE_M apart)."""
    from . import groundfit
    x0, y0, x1, y1 = box
    t = math.radians(yaw)
    U, V = np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])
    cxy = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    us, vs = cxy @ U, cxy @ V
    u0, v0 = float(us.min()), float(vs.min())
    W, H = int(math.ceil((us.max() - u0) / cell_m)), int(math.ceil((vs.max() - v0) / cell_m))
    if W * H > 40000:
        raise ValueError(f'{W} x {H} ground cells: at most 40,000 (larger ground_cell_m or a smaller box)')
    gu, gv = u0 + np.arange(W + 1) * cell_m, v0 + np.arange(H + 1) * cell_m
    q = max(2, int(round(cell_m / groundfit.SAMPLE_M)))
    off = (np.arange(q) + 0.5) * cell_m / q
    su = (gu[:-1, None] + off[None, :]).ravel()
    sv = (gv[:-1, None] + off[None, :]).ravel()
    SU, SV = np.meshgrid(su, sv)
    su, sv = SU.ravel(), SV.ravel()
    codes, sz, inside = _codes_at(fr, ras, su * U[0] + sv * V[0], su * U[1] + sv * V[1])
    cell = (np.floor((sv - v0) / cell_m).astype(int).clip(0, H - 1) * W
            + np.floor((su - u0) / cell_m).astype(int).clip(0, W - 1))
    counts = np.zeros((H * W, 16), int)
    np.add.at(counts, (cell[inside], codes[inside]), 1)
    water, bld = survey.CODE['water'], survey.CODE['building']
    gcodes = [c for c in survey.GROUND_CODES if c != water]
    key = np.full((H, W), None, object)
    left: dict[str, int] = {}
    cu, cv = (gu[:-1] + gu[1:]) / 2, (gv[:-1] + gv[1:]) / 2
    CU, CV = np.meshgrid(cu, cv)
    cx, cy = CU * U[0] + CV * V[0], CU * U[1] + CV * V[1]
    inbox = (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
    lost = []
    for j in range(H):
        for i in range(W):
            if not inbox[j, i]:
                continue
            n = counts[j * W + i]
            tot = int(n.sum())
            why = None
            if tot == 0:
                why = 'outside the survey'
            elif n[bld] * 2 >= tot:
                why = 'under buildings'
            elif n[water] * 2 >= tot:
                why = GROUND_NOTE['water']
            if why:
                left[why] = left.get(why, 0) + 1
                continue
            g = n[gcodes]
            if g.sum() >= 2:
                name = survey.CLASSES[gcodes[int(np.argmax(g))]]
                key[j, i] = GROUND_SAME.get(name, name)
            else:
                lost.append((j, i))
    if lost:   # under crowns and parked cars: the nearest open ground
        e, s = fr.to_scan(np.array([cx[j, i] for j, i in lost]), np.array([cy[j, i] for j, i in lost]))
        r, c, _ok = ras.cell(e, s)
        for (j, i), (name, _d) in zip(lost, ops._open_ground(ras, r, c, reach_m=3 * cell_m)):
            if name in GROUND_ROWS:
                key[j, i] = GROUND_SAME.get(name, name)
            else:
                why = GROUND_NOTE.get(name, 'no ground class') if name else 'no ground class'
                left[why] = left.get(why, 0) + 1
    ok = inside & np.isin(codes, gcodes) & np.isfinite(sz)
    tiles, islands = groundfit.pave(key, gu, gv, su, sv, sz, ok, flat_tol_m=flat_tol_m, smooth_m=smooth_m,
                                    max_tile_m=max_tile_m)
    return {'yaw': yaw, 'U': U, 'V': V, 'gu': gu, 'gv': gv, 'key': key, 'tiles': tiles, 'islands': islands,
            'left': left, 'su': su, 'sv': sv, 'sz': sz, 'ok': ok}


def ground_cells(meta: dict, fr: Frame, ras, box, cell_m: float = 2.0, flat_tol_m: float = 0.3,
                 yaw: float | None = None, max_tile_m: float = 16.0, cat=None, smooth_m: float = 4.0,
                 thick_m: float = GROUND_THICK_M) -> tuple[list[str], dict, GroundSurface]:
    """Ground of box (map metres) as smooth fitted surfaces paved with large tilted tiles (groundfit): material by
    the surface class per cell_m cell, heights from one robust surface per connected paved area (a plane when it
    fits within flat_tol_m, else a stiff thin-plate surface), the kerbside one snapped kerb above the carriageway.
    yaw None: the lattice along the streets (road edges) or the map axes, whichever needs fewer tiles. Returns (agent
    lines, summary with predicted steps, the planned surface for objects standing on it)."""
    from . import groundfit
    cat = cat or load_catalog()
    yaws = [0.0]
    sy = strength = None
    if yaw is None:
        x0, y0, x1, y1 = box
        X, Y = np.meshgrid(np.arange(x0, x1, 0.5), np.arange(y0, y1, 0.5))
        codes, _z, _in = _codes_at(fr, ras, X, Y)
        sy, strength = street_yaw(np.isin(codes, (survey.CODE['road'], survey.CODE['parking'])))
        if sy is not None and min(sy, 90 - sy) > 2:
            yaws.append(sy)
    else:
        yaws = [float(yaw)]
    runs = [_pave_box(fr, ras, box, cell_m, flat_tol_m, y, max_tile_m, smooth_m) for y in yaws]
    best = min(runs, key=lambda r: len(r['tiles']))
    U, V = best['U'], best['V']
    lines, faces = [], []
    counts: dict[str, int] = {}
    for n, tl in enumerate(best['tiles']):
        (u0, u1), (v0, v1) = tl['u'], tl['v']
        uu, vv = np.array([u0, u1, u1, u0]), np.array([v0, v0, v1, v1])
        P = np.column_stack([uu * U[0] + vv * V[0], uu * U[1] + vv * V[1]])
        name = tl['key']
        row = GROUND_ROWS[name]
        ln = _slab(row, cat.assets[row], P, np.array(tl['z']), f'scan.ground.{name}', thick_m)
        lines.append(ln)
        counts[name] = counts.get(name, 0) + 1
        faces.append({'P': groundfit.top_face([float(w) for w in ln.split()[2:11]], cat.assets[row]), 'row': row,
                      'id': n})
    surf = GroundSurface(best['yaw'], best['gu'], best['gv'], best['tiles'])
    steps = groundfit.edge_steps(faces)
    zs = surf.z_at(best['su'] * U[0] + best['sv'] * V[0], best['su'] * U[1] + best['sv'] * V[1])
    res = best['sz'] - zs
    H, W = best['key'].shape
    h = cell_m
    ci = np.clip(((best['su'] - best['gu'][0]) // h).astype(int), 0, W - 1)
    cj = np.clip(((best['sv'] - best['gv'][0]) // h).astype(int), 0, H - 1)
    carriage = np.vectorize(lambda k: k in groundfit.CARRIAGEWAY, otypes=[bool])(best['key'])[cj, ci]
    use = best['ok'] & np.isfinite(res)

    def stats(m):
        a = np.abs(res[m])
        return ({'samples': int(a.size), 'median_m': round(float(np.median(a)), 3),
                 'p90_m': round(float(np.percentile(a, 90)), 3)} if a.size else {'samples': 0})
    kerbs = [i['kerb_m'] for i in best['islands'] if i.get('kerb_m')]
    summary = {'cell_m': cell_m, 'yaw_deg': best['yaw'], 'street_yaw_deg': sy, 'street_strength': strength,
               'tried_yaws': {f'{r["yaw"]:g}': len(r['tiles']) for r in runs}, 'flat_tol_m': flat_tol_m,
               'max_tile_m': max_tile_m, 'smooth_m': smooth_m, 'cells': int((best['key'] != None).sum()),  # noqa: E711
               'tiles': len(lines), 'by_class': counts,
               'rows': {n: GROUND_ROWS[n] for n in counts}, 'left_out': best['left'],
               'islands': [{k: v for k, v in i.items() if k != 'region_fits'} for i in best['islands']
                           if i['cells'] >= 4][:12],
               'island_count': len(best['islands']),
               'kerbs_m': sorted(set(kerbs)), 'max_kerb_m': max(kerbs) if kerbs else 0.0,
               'residual': {'carriageway': stats(use & carriage), 'kerbside': stats(use & ~carriage),
                            'meaning': '|scan ground - planned top| on open-ground samples (noise, dents and kerb '
                                       'blur included)'},
               'predicted_steps': {k: steps[k] for k in ('same_row', 'between_rows', 'worst')},
               'thick_m': thick_m,
               'notes': [GROUND_NOTE[n] for n in counts if n in GROUND_NOTE]}
    return lines, summary, surf


def scan_plan(box=None, classes=None, min_conf: float = 0.6, out: str = '', buildings: str = 'hint',
              lift_m: float = 0.5, limit: int = 2000, include_hidden: bool = False, ground_cell_m: float = 2.0,
              flat_tol_m: float = 0.3, ground_yaw: float | None = None, max_tile_m: float = 16.0, near=None,
              radius_m: float | None = None, weak: str = 'review', survey_key: str = '',
              ground_smooth_m: float = 4.0) -> dict:
    meta = store.resolve(survey_key)
    if weak not in ('review', 'keep', 'skip'):
        raise ValueError('weak must be review (weak fits as #? lines to check), keep (spawn them) or skip')
    if near is not None or radius_m is not None:
        if near is None or len(near) != 2 or radius_m is None or not 0 < float(radius_m) <= 5000:
            raise ValueError('a circle needs near x,y and radius_m 0..5000 (map metres)')
        if box is not None:
            raise ValueError('give box or near + radius_m, not both')
        cx, cy, rad = float(near[0]), float(near[1]), float(radius_m)
        if not all(math.isfinite(v) for v in (cx, cy)):
            raise ValueError('near needs two finite numbers x,y')
        box = [cx - rad, cy - rad, cx + rad, cy + rad]
    else:
        cx = cy = rad = None
    place = store.placement(meta['obj'])
    if not place:
        raise ValueError('scan_plan writes map coordinates: set the placement first (scan_place --at x,y,z --yaw Y '
                         '--datum_m D, or scan_place --from_reference)')
    if buildings not in ('hint', 'standin', 'none'):
        raise ValueError('buildings must be hint (comment block + commented stand-in), standin (spawn the stand-in) '
                         'or none')
    if not 0 <= lift_m <= 5:
        raise ValueError('lift_m must be 0..5 metres')
    lim = max(1, min(int(limit), 5000))
    fr = Frame(meta, 'map')
    want_ground = 'ground' in [c.strip().lower() for c in (classes or [])]
    if want_ground:
        if box is None:
            raise ValueError('ground cells need a box x0,y0,x1,y1 (map metres)')
        if not 0.5 <= ground_cell_m <= 20 or not 0.02 <= flat_tol_m <= 2:
            raise ValueError('ground_cell_m must be 0.5..20 and flat_tol_m 0.02..2 metres')
        if not ground_cell_m <= max_tile_m <= 100:
            raise ValueError('max_tile_m must be ground_cell_m..100 metres')
        if not 1 <= ground_smooth_m <= 30:
            raise ValueError('ground_smooth_m must be 1..30 metres')
        classes = [c for c in classes if c.strip().lower() != 'ground']
    sel, want = _select(meta, fr, classes, box, min_conf, default=None if want_ground else PLAN_CLASSES)
    if rad is not None:   # the circle: objects whose centre lies within radius_m of near
        def dist(o):
            mx, my = fr.from_scan(o['x'], o['y'])
            return math.hypot(mx - cx, my - cy)
        sel = [o for o in sel if dist(o) <= rad]
    if want_ground and not classes:
        sel, want = [], []
    path = _out_path(out, meta['key'])
    cat = load_catalog()
    idx = ops._obj_index(meta)
    record = ops._facade_record(meta) if 'building' in want or not want else {}
    ras = ops.Rasters(meta) if want_ground or any(o['cls'] in ('car', 'truck') for o in sel) else None
    ground = surf = None
    glines: list[str] = []
    if want_ground:   # first: objects standing on the ground take their height from the planned surface
        glines, ground, surf = ground_cells(meta, fr, ras, ops._box4(box), ground_cell_m, flat_tol_m, ground_yaw,
                                            max_tile_m, cat, smooth_m=ground_smooth_m)
    pools: dict[str, list] = {}
    excluded: list[dict] = []
    lines, objects, hints, skipped, review = [], [], [], [], []
    by: dict[str, dict] = {}
    sel.sort(key=lambda o: (o['cls'], o['id']))
    for o in sel:
        cls = o['cls']
        info = by.setdefault(cls, {'cls': cls, 'selected': 0, 'planned': 0, 'hints': 0, 'skipped': 0, 'rows': {}})
        info['selected'] += 1

        def skip(reason):
            info['skipped'] += 1
            skipped.append({'id': o['id'], 'cls': cls, 'reason': reason})

        if cls in NOT_MATCHABLE or cls not in SCALING:
            skip(NOT_MATCHABLE.get(cls, 'no catalogue rule for this class'))
            continue
        if len(objects) >= lim:
            skip(f'limit {lim} reached')
            continue
        on = None
        if cls in ('tree', 'shrub') and slender(o):
            skip(SLENDER_REASON)
            continue
        if cls in ('car', 'truck'):
            on = _ground_around(ras, o)
            if on is None:
                skip('no visible ground around it (roof, deck or under crowns): not placed')
                continue
            if on == 'water':
                skip('stands in water in the scan: not placed')
                continue
        if cls not in pools:
            pools[cls] = candidates(cls, include_hidden, excluded=excluded)
        if not pools[cls]:
            skip('no catalogue rows for this class')
            continue
        best = ranked(o, pools[cls], 5)
        pick = best[0]
        if cls in ('tree', 'shrub'):  # variety: rotate through rows that fit nearly as well
            near = [c for c in best if c['score'] >= .85 * pick['score']]
            pick = near[o['id'] % len(near)]
            yaw = (o['id'] * 137.508) % 360
        else:
            yaw = fr.yaw_out(pick['yaw_scan'], 360)
        a = cat.assets[pick['row']]
        x, y = fr.from_scan(o['x'], o['y'])
        gz = fr.z_out(o['ground_z'])
        if surf is not None and cls != 'building':   # on the planned ground, not the raw scan ground
            sz = float(surf.z_at(x, y)[0])
            gz = sz if math.isfinite(sz) else gz
        group = f'scan.{cls}.{o["id"]}'
        spawn = _spawn(pick['row'], a, x, y, gz + lift_m, yaw, pick['scale'], group, pick['visible_offset_m'])
        entry = {'id': o['id'], 'cls': cls, 'conf': o['conf'], 'x': round(x, 2), 'y': round(y, 2),
                 'ground_z': round(gz, 2), 'size_m': [o['length_m'], o['width_m'], _target(o)[2]],
                 'crown_m': o.get('crown_m') if cls in ('tree', 'shrub') else None,
                 'row': pick['row'], 'name': pick['name'], 'fit_err_pct': pick['fit_err_pct'],
                 'shape_err_pct': pick['shape_err_pct'], 'scale': pick['scale'], 'visible_m': pick['visible_m'],
                 'sized_by': pick['sized_by'], 'visible_offset_m': list(pick['visible_offset_m']),
                 'yaw_deg': round(yaw % 360, 1), 'group': group, 'on': on, 'spawn': spawn}
        head = (f"# {cls} {o['id']} conf {o['conf']} {o['src']} {o['length_m']}x{o['width_m']}x{_target(o)[2]} m"
                f" -> {pick['row']} {pick['name']} fit {pick['fit_err_pct']}% scale "
                f"{','.join(f'{v:g}' for v in pick['scale'])}" + (f' on {on}' if on else '')
                + (f" ({pick['notes']})" if pick['notes'] else ''))
        why_weak = _weak_fit(cls, pick)
        if why_weak and cls in ('car', 'truck') and _target(o)[2] > 4.0:
            why_weak += f'; scan height {_target(o)[2]:g} m: a crown or roof over it? (scan_sheet shows)'
        if why_weak and cls != 'building':
            entry['weak'] = why_weak
            if weak == 'skip':
                skip(f'weak fit: {why_weak}')
                continue
            if weak == 'review':
                review.append({'id': o['id'], 'cls': cls, 'row': pick['row'], 'why': why_weak})
                info['review'] = info.get('review', 0) + 1
                lines += [head, f'# WEAK FIT ({why_weak}): check scan_sheet / scan_match {o["id"]}, then uncomment',
                          f'#? {spawn}', '#? drop last']
                continue
            head += f'  WEAK FIT: {why_weak}'
        if cls == 'building':
            hint, block = _hint({**o, 'eave_m': ops.eave_of(meta, o)}, fr, record, idx)
            hint['standin'] = {k: entry[k] for k in ('row', 'name', 'fit_err_pct', 'shape_err_pct', 'scale',
                                                         'yaw_deg', 'spawn')}
            hints.append(hint)
            info['hints'] += 1
            if buildings == 'none':
                continue
            lines += block + [f"#   stand-in: {pick['row']} {pick['name']} fit {pick['fit_err_pct']}% shape "
                              f"{pick['shape_err_pct']}% scale "
                              f"{','.join(f'{v:g}' for v in pick['scale'])} yaw {yaw % 360:.1f}"
                              + (f" ({pick['notes']})" if pick['notes'] else '')]
            if buildings == 'hint':
                lines += [f'#? {spawn}', '#? drop last', '']
                continue
            lines += [spawn, 'drop last', '']
        else:
            lines += [head, spawn, 'drop last']
        objects.append(entry)
        info['planned'] += 1
        info['rows'][pick['row']] = info['rows'].get(pick['row'], 0) + 1
    if want_ground:
        st = ground['predicted_steps']
        lines = [f"# GROUND {ground['tiles']} tiles, {ground_cell_m:g} m lattice at yaw {ground['yaw_deg']:g}: smooth "
                 f"fitted surfaces (flat_tol_m {flat_tol_m:g}), kerbs {ground['kerbs_m'] or 'none'} m; predicted "
                 f"steps over 1 cm: {st['same_row'].get('over_1cm', 0)} seams, {st['between_rows'].get('over_1cm', 0)} "
                 f"between materials; no drop (exact heights)"] + glines + [''] + lines
    t = place
    header = [f"# scan_plan survey {meta['key']} of {meta['obj']}",
              f"# placement at={t['at']} yaw={t['yaw']} datum_m={t['datum_m']} ({t.get('source', 'given')}); "
              f"map metres -> agent centimetres",
              f"# written {time.strftime('%Y-%m-%d %H:%M:%S')}; NOT executed. Review it, then run it with agent_batch "
              "(./uc batch FILE).",
              f"# every object: spawn {lift_m:g} m above the {'planned ground' if want_ground else 'scan ground'}, then "
              'drop last rests it on the game surface; '
              'groups scan.<class>.<id> (scan_compare pairs by them)',
              f"# buildings: {'comment blocks; #? lines are a stand-in to uncomment by hand' if buildings == 'hint' else buildings}",
              '']
    path.parent.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(header + lines).rstrip() + '\n'
    path.write_text(text)
    jpath = path.with_suffix('.json')
    summary = [{'cls': i['cls'], 'selected': i['selected'], 'planned': i['planned'], 'building_hints': i['hints'],
                'skipped': i['skipped'], **({'review': i['review']} if i.get('review') else {}),
                'rows': [f'{r} x{n}' for r, n in sorted(i['rows'].items(), key=lambda kv: -kv[1])[:4]]}
               for i in sorted(by.values(), key=lambda r: -r['selected'])]
    data = {'survey': meta['key'], 'obj': meta['obj'], 'placement': place, 'frame': 'map', 'units': 'metres',
            'lift_m': lift_m, 'buildings_mode': buildings, 'objects': objects, 'buildings': hints, 'skipped': skipped,
            'review': review, 'region': ({'near': [cx, cy], 'radius_m': rad} if rad is not None else
                                         {'box': list(ops._box4(box))} if box is not None else None),
            'classes': summary, 'ground': ground, 'plan': str(path)}
    jpath.write_text(json.dumps(data, indent=1, allow_nan=False))
    spawns = sum(1 for ln in lines if ln.startswith('spawn '))
    return {'survey': meta['key'], 'frame': 'map', 'file': str(path), 'json': str(jpath), 'spawn_lines': spawns,
            'commands': spawns * 2 - (ground['tiles'] if ground else 0), 'classes': summary,
            **({'ground': ground} if ground else {}), 'skipped_count': len(skipped), 'skipped': skipped[:20],
            **({'rows_excluded': excluded} if excluded else {}),
            **({'review_count': len(review), 'review': review[:20],
                'review_note': "weak fits are '#?' lines (not run): look at each object (scan_sheet, scan_match ID), "
                               'then uncomment, pick another row or delete; --weak keep spawns them'}
               if review else {}),
            'executed': False,
            'next': f'read {path.name}; run it with agent_batch ({path}); then capture_view next to scan_view of '
                    'the same box, and scan_compare',
            'limitations': LIMITATIONS_MATCH[1:] + [
                'z: spawn at the scan ground (with class ground: the planned ground) + lift_m, then drop; where the '
                'game surface is far from it (other base, placement datum) drop rests the object on whatever the game '
                'has there.',
                'Buildings are not built: hints carry footprint, heights, roof, walls and window rows for a builder; '
                'a catalogue stand-in is only a block-out.']}


# ---------------------------------------------------------------------------- compare

COMPAT = {'car': {'vehicle'}, 'truck': {'vehicle'}, 'boat': {'vehicle'}, 'tree': {'foliage'}, 'shrub': {'foliage'},
          'building': {'building', 'landmark'}, 'pool': {'surface', 'prop', 'industrial'}}


def _compatible(cls: str, p: dict) -> bool:
    """Could placed object p stand for a scan object of class cls? Unknown rows always could."""
    if p['asset'] is None:
        return True
    roles = COMPAT.get(cls)
    return roles is None or p['role'] in roles or _wanted(cls, p['asset'], p['role'])


def _placed_objects(snapshot: str) -> tuple[list[dict], str]:
    from .. import agent, groups, managed, snapshots
    if not snapshot:
        with agent.session():
            snap = groups.snapshot()
        return snap['objects'], 'live groups export'
    p = Path(snapshot).expanduser()
    if p.is_file():
        data, p = snapshots.load_snapshot(str(p), p.parent)
    else:
        with agent.session():
            _info, root = managed.context()
        data, p = snapshots.load_snapshot(snapshot, root)
    return data['objects'], f'scene_snapshot {p}'


def _up_axis(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """World direction of a row's local +Z after an Unreal rotator (FRotationMatrix: roll about X, pitch about Y,
    yaw about Z)."""
    sp, cp = math.sin(math.radians(pitch)), math.cos(math.radians(pitch))
    sy, cy = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
    sr, cr = math.sin(math.radians(roll)), math.cos(math.radians(roll))
    return np.array([-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp])


def _local_dir(yaw: float, pitch: float, roll: float, v) -> np.ndarray:
    """World direction of a local vector under an Unreal rotator."""
    sp, cp = math.sin(math.radians(pitch)), math.cos(math.radians(pitch))
    sy, cy = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
    sr, cr = math.sin(math.radians(roll)), math.cos(math.radians(roll))
    X = np.array([cp * cy, cp * sy, sp])
    Y = np.array([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp])
    Z = _up_axis(pitch, yaw, roll)
    return v[0] * X + v[1] * Y + v[2] * Z


def _placed_box(o: dict, cat) -> dict:
    """A placed object as the scan would see it: visible centre, visible size along / across its visible long axis,
    visible top, long-axis yaw, yaw period (symmetry) and the facing of a tilted dominant face (from its asset
    profile when the class uses one, else the catalogue box)."""
    xf = o['xf']
    pivot = (xf[0] / 100, xf[1] / 100, xf[2] / 100)
    pitch, yaw, roll = float(xf[3]), float(xf[4]), float(xf[5])
    a = cat.assets.get(o['row'])
    if a is None:
        return {'row': o['row'], 'name': None, 'role': 'unknown', 'asset': None, 'x': pivot[0], 'y': pivot[1], 'bottom': pivot[2],
                'top': pivot[2], 'length': 0.0, 'width': 0.0, 'yaw': yaw % 180, 'period': None, 'xf': list(xf),
                'vis': (0.0, 0.0, 0.0), 'crown': None, 'facing': None, 'tilt': None, 'by': 'pivot'}
    scale = tuple(float(v) for v in xf[6:9])
    lo = geom.world_aabb(a.origin_cm, a.extent_cm, pivot, math.radians(yaw), scale)
    role = fit.role_of(a)
    cls = 'tree' if role == 'foliage' else 'car' if role == 'vehicle' else 'small_object'
    dims = row_dims(a, cls)
    vx, vy, vh = (dims['size'][i] * abs(scale[i]) for i in range(3))
    # visible centre: mesh box centre + profile offset, turned by yaw (pitch / roll ignored for the centre)
    cx_l = a.origin_cm[0] / 100 * scale[0] + dims['offset'][0] * scale[0]
    cy_l = a.origin_cm[1] / 100 * scale[1] + dims['offset'][1] * scale[1]
    t = math.radians(yaw)
    x = pivot[0] + cx_l * math.cos(t) - cy_l * math.sin(t)
    y = pivot[1] + cx_l * math.sin(t) + cy_l * math.cos(t)
    long_local = dims['long_deg']
    L, W = (vx, vy) if long_local == 0 else (vy, vx)
    crown = dims['crown'] * (abs(scale[0]) + abs(scale[1])) / 2 if dims['crown'] else None
    top = lo[2] + vh if dims['by'] == 'visible' else lo[5]
    up = _up_axis(pitch, yaw, roll)
    facing = tilt = None
    p = None
    if dims['by'] == 'visible':
        from . import profile
        p = profile.get(a.id)
    if p and p.get('panel'):
        pa = math.radians(p['panel']['faces_deg'])
        pt = math.radians(p['panel']['tilt_deg'])
        n = _local_dir(yaw, pitch, roll, (math.sin(pt) * math.cos(pa), math.sin(pt) * math.sin(pa), math.cos(pt)))
    else:
        n = up
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(n[2])))))
    if 2.0 <= tilt <= 80:
        facing = math.degrees(math.atan2(n[1], n[0])) % 360
    return {'row': o['row'], 'name': a.name, 'role': role, 'asset': a, 'x': x, 'y': y, 'bottom': lo[2], 'top': top,
            'length': max(L, W), 'width': min(L, W), 'height': vh, 'yaw': (yaw + long_local) % 180,
            'period': dims['period'] if dims['by'] == 'visible' else (180.0 if a.size_m[0] != a.size_m[1] else 90.0),
            'xf': list(xf), 'vis': (vx, vy, vh), 'crown': crown, 'facing': facing, 'tilt': tilt, 'by': dims['by']}


def _scan_facing(ras, o: dict) -> tuple[float | None, float | None]:
    """(tilt, facing in the scan frame) of the mean surface normal over an object's cells; None when flat / unknown."""
    oid = np.asarray(ras.objid)
    c, r = ras.g.to_cell(o['x'], o['y'])
    reach = int(max(o['length_m'], o['width_m']) / ras.g.res) + 2
    r0, c0 = max(0, int(r) - reach), max(0, int(c) - reach)
    win = oid[r0:int(r) + reach + 1, c0:int(c) + reach + 1] == o['id']
    if win.sum() < 4:
        return None, None
    rows, cols = np.nonzero(win)
    n = np.asarray(ras.nrm[rows + r0, cols + c0], float).mean(0)
    if not np.isfinite(n).all() or np.linalg.norm(n) < 1e-6:
        return None, None
    n /= np.linalg.norm(n)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(n[2])))))
    if tilt < 2.0:
        return round(tilt, 1), None
    return round(tilt, 1), math.degrees(math.atan2(n[1], n[0])) % 360


def _ang(d: float, period: float) -> float:
    """|d| folded into 0..period/2."""
    return abs((d + period / 2) % period - period / 2)


def _fix_line(s: dict, o: dict, p: dict, fr: Frame, statuses: set) -> str | None:
    """`transform ACTOR x y z - yaw - sx sy sz` (agent cm / degrees, '-' keeps) that puts placed object p's visible
    centre on scan object s, turns it by the smallest yaw change that lays it on the scan axis (within its symmetry)
    and rescales it as scan_match would; z keeps the bottom where it is."""
    a = p['asset']
    if a is None or not p.get('actor'):
        return None
    xf = p['xf']
    yaw, scale = float(xf[4]), [float(v) for v in xf[6:9]]
    cls = o['cls']
    new_yaw, new_scale = yaw, list(scale)
    if 'yaw_off' in statuses or 'size_off' in statuses:
        f = fit_one(o, a)
        if 'size_off' in statuses or 'height_off' in statuses:
            new_scale = [float(v) for v in f['scale']]
        if 'yaw_off' in statuses and p['period']:
            target = fr.yaw_out(f['yaw_scan'], 360)
            per = min(p['period'], 180.0)        # the scan box knows the axis, not the front
            k = round((yaw - target) / per)
            new_yaw = (target + k * per) % 360
    elif 'height_off' in statuses and cls in ('tree', 'shrub'):
        new_scale = [float(v) for v in fit_one(o, a)['scale']]
    dims = row_dims(a, 'tree' if p['role'] == 'foliage' else 'car' if p['role'] == 'vehicle' else 'small_object')
    t = math.radians(new_yaw)
    cx_l = a.origin_cm[0] / 100 * new_scale[0] + dims['offset'][0] * new_scale[0]
    cy_l = a.origin_cm[1] / 100 * new_scale[1] + dims['offset'][1] * new_scale[1]
    px = s['x'] - (cx_l * math.cos(t) - cy_l * math.sin(t))
    py = s['y'] - (cx_l * math.sin(t) + cy_l * math.cos(t))
    pz = p['bottom'] - (a.origin_cm[2] - a.extent_cm[2]) / 100 * new_scale[2]
    same_scale = all(abs(u - v) < 2e-3 * max(1.0, abs(v)) for u, v in zip(new_scale, scale))
    same_yaw = _ang(new_yaw - yaw, 360) < .05
    if same_scale and same_yaw and 'offset' not in statuses:
        return None     # nothing this row can do better (catalogue / scale limits): see size_ratio
    sc = '- - -' if same_scale else ' '.join(f'{v:.4g}' for v in new_scale)
    zs = '-' if same_scale else f'{pz * 100:.1f}'
    ys = '-' if same_yaw else f'{new_yaw:.1f}'
    return f"transform {p['actor']} {px * 100:.1f} {py * 100:.1f} {zs} - {ys} - {sc}"


STATUS_ORDER = {'missing': 0, 'offset': 1, 'shape_off': 2, 'yaw_off': 3, 'size_off': 4, 'height_off': 5,
                'facing_off': 6, 'matched': 7}


HEIGHT_CELLS_MAX = 40000          # cells of the height comparison raster
HEIGHT_SAMPLES_MAX = 2_000_000    # lattice samples over all cells (memory: about 100 bytes per sample)


def survey_extent(meta: dict, fr: Frame) -> tuple[float, float, float, float]:
    """Axis-aligned extent of the survey grid in the frame's x/y (the rotated square's corners)."""
    g = meta['grid']
    e = np.array([g['x0'], g['x0'] + g['w'] * g['res']] * 2, float)
    sn = np.array([g['y0']] * 2 + [g['y0'] + g['h'] * g['res']] * 2, float)
    x, y = fr.from_scan(e, sn)
    return float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))


def clip_region(region, extent):
    """region intersected with extent, or None when they do not overlap."""
    x0, y0 = max(region[0], extent[0]), max(region[1], extent[1])
    x1, y1 = min(region[2], extent[2]), min(region[3], extent[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _height_cells(ras, fr: Frame, placed: list[dict], region, cell_m: float, tol_m: float, top: int,
                  points=None) -> dict:
    """Per cell of cell_m: scan height above ground (95th percentile of the nDSM) vs the highest placed visible top
    above the scan ground. Placed footprints are their visible boxes (trees: crown discs); points (n x 3 map metres,
    the top surfaces of multi-object buildings) add their heights where they fall. The region should be clipped to
    the survey (clip_region); cells and lattice samples are capped either way."""
    x0, y0, x1, y1 = region
    nx, ny = max(1, int(math.ceil((x1 - x0) / cell_m))), max(1, int(math.ceil((y1 - y0) / cell_m)))
    if nx * ny > HEIGHT_CELLS_MAX:
        cell_m = math.sqrt((x1 - x0) * (y1 - y0) / HEIGHT_CELLS_MAX)
        nx, ny = int(math.ceil((x1 - x0) / cell_m)), int(math.ceil((y1 - y0) / cell_m))
    # scan: sample each cell on a lattice of about 0.5 m, coarser when the samples would exceed the cap
    k = max(2, min(int(round(cell_m / 0.5)), int(math.sqrt(HEIGHT_SAMPLES_MAX / (nx * ny)))))
    fx = x0 + (np.arange(nx * k) + .5) * cell_m / k
    fy = y0 + (np.arange(ny * k) + .5) * cell_m / k
    X, Y = np.meshgrid(fx, fy)
    e, sn = fr.to_scan(X, Y)
    r, c, inside = ras.cell(e.ravel(), sn.ravel())
    nd = (np.asarray(ras.dsm[r, c], float) - np.asarray(ras.dtm[r, c], float)).reshape(Y.shape)
    nd[~inside.reshape(Y.shape)] = np.nan
    nd = np.where(np.isfinite(nd), np.maximum(nd, 0), np.nan)
    blocks = nd.reshape(ny, k, nx, k).transpose(0, 2, 1, 3).reshape(ny, nx, k * k)
    with np.errstate(all='ignore'):
        valid = np.isfinite(blocks).mean(2) > .5
        scan_h = np.where(valid, np.nanpercentile(np.where(np.isfinite(blocks), blocks, -1), 95, axis=2), np.nan)
    gnd = np.asarray(ras.ground(e.ravel(), sn.ravel()), float).reshape(Y.shape)
    built = np.zeros(Y.shape)
    for p in placed:
        if p['asset'] is None:
            continue
        reach = max(p['length'], p['width'], p['crown'] or 0) / 2 + cell_m
        i0, i1 = np.searchsorted(fx, p['x'] - reach), np.searchsorted(fx, p['x'] + reach)
        j0, j1 = np.searchsorted(fy, p['y'] - reach), np.searchsorted(fy, p['y'] + reach)
        if i0 >= i1 or j0 >= j1:
            continue
        sx, sy = X[j0:j1, i0:i1] - p['x'], Y[j0:j1, i0:i1] - p['y']
        if p['crown']:
            inside_p = np.hypot(sx, sy) <= p['crown'] / 2
        else:
            t = math.radians(p['yaw'])
            u = sx * math.cos(t) + sy * math.sin(t)
            v = -sx * math.sin(t) + sy * math.cos(t)
            inside_p = (np.abs(u) <= p['length'] / 2) & (np.abs(v) <= p['width'] / 2)
        h = p['top'] - fr.z_out(gnd[j0:j1, i0:i1])
        sub = built[j0:j1, i0:i1]
        np.maximum(sub, np.where(inside_p, h, 0), out=sub)
    if points is not None and len(points):
        step = cell_m / k
        ii = np.floor((points[:, 0] - x0) / step).astype(int)
        jj = np.floor((points[:, 1] - y0) / step).astype(int)
        ok = (ii >= 0) & (ii < nx * k) & (jj >= 0) & (jj < ny * k)
        ii, jj = ii[ok], jj[ok]
        h = points[ok, 2] - fr.z_out(gnd[jj, ii])
        flat = built.reshape(-1)
        np.maximum.at(flat, jj * (nx * k) + ii, np.nan_to_num(h, nan=0.0))
    with np.errstate(all='ignore'):
        built_h = built.reshape(ny, k, nx, k).transpose(0, 2, 1, 3).reshape(ny, nx, k * k)
        built_h = np.percentile(built_h, 95, axis=2)
    either = valid & ((scan_h > tol_m) | (built_h > tol_m))
    diff = built_h - scan_h
    out = {'cell_m': round(cell_m, 2), 'cells': int(valid.sum()), 'cells_with_height': int(either.sum())}
    if not either.any():
        return {**out, 'median_abs_m': None, 'within_tol_share': None, 'too_low': 0, 'too_high': 0, 'worst': []}
    d = diff[either]
    worst = []
    order = np.argsort(-np.abs(np.where(either, diff, 0)), axis=None)
    for idx in order[:top]:
        j, i = divmod(int(idx), nx)
        if not either[j, i] or abs(diff[j, i]) <= tol_m:
            break
        worst.append({'x': round(x0 + (i + .5) * cell_m, 1), 'y': round(y0 + (j + .5) * cell_m, 1),
                      'scan_h_m': round(float(scan_h[j, i]), 1), 'built_h_m': round(float(built_h[j, i]), 1),
                      'diff_m': round(float(diff[j, i]), 1)})
    return {**out, 'median_abs_m': round(float(np.median(np.abs(d))), 2),
            'within_tol_share': round(float((np.abs(d) <= tol_m).mean()), 3),
            'too_low': int((d < -tol_m).sum()), 'too_high': int((d > tol_m).sum()), 'worst': worst}


def _z_range(o: dict, a) -> tuple[float, float]:
    """World bottom and top (map metres) of a placed object's catalogue box under its full rotation."""
    xf = o['xf']
    if a is None:
        return xf[2] / 100, xf[2] / 100
    R = composite.rotation(float(xf[3]), float(xf[4]), float(xf[5]))
    org = np.asarray(a.origin_cm, float) / 100 * np.asarray(xf[6:9], float)
    ext = np.abs(np.asarray(a.extent_cm, float) / 100 * np.asarray(xf[6:9], float))
    half = float(np.abs(R[2]) @ ext)
    cz = xf[2] / 100 + float(R[2] @ org)
    return cz - half, cz + half


def _merged_box(members: list[dict], s: dict, fr: Frame) -> dict:
    """Several placed objects standing for one scan object (a hedge of bushes, a car and its trailer): one visible
    box along the scan object's axis. No actor (no fix line), no yaw check."""
    t = math.radians(fr.yaw_out(s['o']['yaw_deg']))
    U, V = np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])
    pts = []
    for p in members:
        tp = math.radians(p['yaw'])
        cu, su = math.cos(tp), math.sin(tp)
        for a in (-1, 1):
            for b in (-1, 1):
                pts.append((p['x'] + a * p['length'] / 2 * cu - b * p['width'] / 2 * su,
                            p['y'] + a * p['length'] / 2 * su + b * p['width'] / 2 * cu))
    P = np.asarray(pts)
    pu, pv = P @ U, P @ V
    cu_, cv_ = (pu.max() + pu.min()) / 2, (pv.max() + pv.min()) / 2
    L, W = float(pu.max() - pu.min()), float(pv.max() - pv.min())
    main = max(members, key=lambda p: p['length'] * p['width'])
    foliage = all(p['role'] == 'foliage' for p in members)
    return {**main, 'x': float(cu_ * U[0] + cv_ * V[0]), 'y': float(cu_ * U[1] + cv_ * V[1]),
            'bottom': min(p['bottom'] for p in members), 'top': max(p['top'] for p in members),
            'length': max(L, W), 'width': min(L, W), 'height': max(p['top'] for p in members) - min(p['bottom'] for p in members),
            'yaw': fr.yaw_out(s['o']['yaw_deg']) % 180 if L >= W else (fr.yaw_out(s['o']['yaw_deg']) + 90) % 180,
            'period': None, 'crown': (L + W) / 2 if foliage else None, 'actor': None,
            'members': len(members)}


DEV_KEYS = ('missing', 'offset', 'shape_off', 'yaw_off', 'size_off', 'height_off', 'facing_off')


def severity(r: dict) -> float:
    """Rank of a compare row for labels and lists: status first, then how far off (bigger = worse)."""
    if r['status'] == 'matched':
        return 0.0
    base = {'missing': 6, 'offset': 5, 'shape_off': 4, 'yaw_off': 3.5, 'size_off': 3, 'height_off': 2.5,
            'facing_off': 2}.get(r['status'], 1)
    mag = 0.0
    if r['status'] == 'missing':
        mag = float(r.get('conf') or 0) + (min(r['raster']['scan_m2'], 500) / 500 if r.get('raster') else 0)
    else:
        mag = min(1.0, (r.get('offset_m') or 0) / 10) + min(1.0, abs(r.get('height_off_m') or 0) / 5)
        if r.get('raster'):
            mag += 1 - r['raster']['iou']
        if r.get('size_ratio'):
            mag += min(1.0, max(abs(v - 1) for v in r['size_ratio']))
    return base + mag / 4


def ground_steps(placed: list[dict], fr: Frame, ras, flat_tol_m: float = 0.3, top: int = 20) -> dict | None:
    """Seams of the placed ground tiles (groundfit.is_tile): height steps where neighbouring tile tops meet (largest
    per pair; between materials they may be intended kerbs) and tiles whose top is more than flat_tol_m from the scan
    ground (median over five points of the top face). None without tiles."""
    from . import groundfit
    faces = []
    for p in placed:
        xf = p['xf']
        if not groundfit.is_tile(p['asset'], xf):
            continue
        faces.append({'P': groundfit.top_face(xf, p['asset']), 'row': p['row'], 'id': p.get('actor')})
    if not faces:
        return None
    st = groundfit.edge_steps(faces, top=max(top, 10))
    P = np.array([f['P'] for f in faces])                       # n x 4 x 3
    c = P.mean(1, keepdims=True)
    pts = np.concatenate([c, c + (P - c) * 0.5], 1)              # centre and four points halfway to the corners
    e, sn = fr.to_scan(pts[..., 0].ravel(), pts[..., 1].ravel())
    _r, _c, inside = ras.cell(e, sn)
    g = np.where(inside, np.asarray(fr.z_out(ras.ground(e, sn)), float), np.nan).reshape(pts.shape[:2])
    with np.errstate(all='ignore'):
        dz = np.nanmedian(pts[..., 2] - g, 1)
    far = [{'actor': f['id'], 'row': f['row'], 'x': round(float(cc[0, 0]), 2), 'y': round(float(cc[0, 1]), 2),
            'top_minus_scan_m': round(float(d), 2)}
           for f, cc, d in zip(faces, c, dz) if math.isfinite(float(d)) and abs(float(d)) > flat_tol_m]
    far.sort(key=lambda r: -abs(r['top_minus_scan_m']))
    seams = st['same_row'].get('over_1cm', 0)
    return {'tiles': st['tiles'], 'edges_over_1cm': seams + st['between_rows'].get('over_1cm', 0),
            'same_row': st['same_row'], 'between_rows': st['between_rows'], 'worst': st['worst'],
            'far_from_scan': len(far), 'flat_tol_m': flat_tol_m, 'far': far[:top],
            'meaning': 'neighbouring ground tiles: height step where their top edges meet (largest per pair); '
                       'same_row steps are seams (a smooth ground has none over 1 cm), between_rows may be kerbs; '
                       'far: tiles whose top is more than flat_tol_m above (+) or below (-) the scan ground'}


def scan_compare(snapshot: str = '', box=None, group: str = '', classes=None, min_conf: float = 0.5,
                 position_m: float = 1.0, height_m: float = 1.0, yaw_deg: float = 10.0, size_tol: float = 0.15,
                 cell_m: float = 5.0, top: int = 20, iou: float = 0.6, flat_tol_m: float = 0.3,
                 survey_key: str = '') -> dict:
    from .. import snapshots
    meta = store.resolve(survey_key)
    if not store.placement(meta['obj']):
        raise ValueError('scan_compare needs the scan placement (scan_place) to put scan objects into the map')
    if not 0 < position_m <= 50 or not 0 < height_m <= 50:
        raise ValueError('position_m and height_m must be 0..50 metres')
    if not 0 < yaw_deg <= 90 or not 0 < size_tol <= 2:
        raise ValueError('yaw_deg must be 0..90 degrees and size_tol 0..2 (0.2 = 20 %)')
    if not 1 <= cell_m <= 50:
        raise ValueError('cell_m must be 1..50 metres')
    if not 0 < iou <= 1:
        raise ValueError('iou must be 0..1 (0.6: footprints sharing 60 % of their union)')
    if not 0.02 <= flat_tol_m <= 2:
        raise ValueError('flat_tol_m must be 0.02..2 metres')
    sc = snapshots.scope(group, box)
    sc_group = snapshots.scope(group, None)
    bx = sc['box_m']
    top = max(1, min(int(top), 200))
    fr = Frame(meta, 'map')
    scan_objs, want = _select(meta, fr, classes, box, min_conf, default=PLAN_CLASSES)
    ras = ops.Rasters(meta)
    placed_raw, source = _placed_objects(snapshot)
    cat = load_catalog()
    index = ops._obj_index(meta)
    known = {o['cls'] for o in index.values()}
    placed = []
    for o in placed_raw:
        if not snapshots.in_scope(o, sc_group):
            continue
        p = _placed_box(o, cat)
        p['actor'], p['group'] = o.get('actor'), o['group']
        p['raw'] = o
        p['zlo'], p['zhi'] = _z_range(o, p['asset'])
        x, y = o['xf'][0] / 100, o['xf'][1] / 100
        p['in_box'] = not bx or (bx[0] <= x <= bx[2] and bx[1] <= y <= bx[3])
        placed.append(p)
    if placed:      # scan ground under every placed object (one bilinear lookup)
        e, sn = fr.to_scan(np.array([p['x'] for p in placed]), np.array([p['y'] for p in placed]))
        _r, _c, inside = ras.cell(e, sn)
        gz = fr.z_out(np.asarray(ras.ground(e, sn), float))
        for p, g, ins in zip(placed, gz, inside):
            p['ground'] = float(g) if ins and math.isfinite(float(g)) else None
            above = p['zhi'] - p['ground'] if p['ground'] is not None else None
            p['kind'] = composite.kind_of(p['asset'], p['zhi'] - p['zlo'], above)
    # tags: a group that names a scan object (scan.car.12, car.12, b75, b57nw, building_75)
    for p in placed:
        p['tag'] = None
        t = composite.parse_tag(p['group'], known)
        if t and t[1] in index and composite.tag_fits(t[0], index[t[1]]['cls']):
            sid = t[1]
            if index[sid]['cls'] == 'building_part' and index[sid].get('parent') in index:
                sid = index[sid]['parent']
            p['tag'] = sid
    # scan objects in the map frame
    S = []
    for o in scan_objs:
        x, y = fr.from_scan(o['x'], o['y'])
        S.append({'o': o, 'x': x, 'y': y, 'ground': fr.z_out(o['ground_z']), 'top': fr.z_out(o['top_z'])})
    by_id = {s['o']['id']: i for i, s in enumerate(S)}
    members: dict[int, list[int]] = {}    # scan index -> placed indices
    how: dict[int, str] = {}
    used = set()
    tagged_other: dict[str, int] = {}
    for j, p in enumerate(placed):        # 1. group tags
        if p['tag'] is None or p['kind'] == 'gameplay':
            continue
        i = by_id.get(p['tag'])
        if i is None:
            if p['in_box']:
                key = f"{index[p['tag']]['cls']} {p['tag']}"
                tagged_other[key] = tagged_other.get(key, 0) + 1
            used.add(j)
            continue
        members.setdefault(i, []).append(j)
        how[i] = 'group'
        used.add(j)
    ground: dict[str, int] = {}
    gameplay = 0
    free = []
    for j, p in enumerate(placed):
        if j in used or not p['in_box']:
            continue
        if p['kind'] == 'ground':
            key = (p['group'] if p['group'] not in ('', '-') else p['name'] or p['row'])
            ground[key] = ground.get(key, 0) + 1
            used.add(j)
        elif p['kind'] == 'gameplay':
            gameplay += 1
            used.add(j)
        else:
            free.append(j)
    cand = []                              # 2. nearest compatible single object for point-like scan objects
    if free:
        P = np.array([[placed[j]['x'], placed[j]['y']] for j in free])
        for i, s in enumerate(S):
            o = s['o']
            if i in members or o['cls'] == 'building':
                continue
            reach = max(3.0, .75 * max(o['length_m'], o['width_m']), position_m)
            d = np.hypot(P[:, 0] - s['x'], P[:, 1] - s['y'])
            for k in np.flatnonzero(d <= reach):
                j = free[k]
                if _compatible(o['cls'], placed[j]):
                    cand.append((float(d[k]), i, j))
    for d, i, j in sorted(cand):
        if i in members or j in used:
            continue
        members[i] = [j]
        how[i] = 'nearest'
        used.add(j)
    # 3. buildings by footprint: untagged pieces standing on a scan building's cells (per group by majority)
    bids = {s['o']['id'] for s in S if s['o']['cls'] == 'building'}
    by_group: dict[str, list[int]] = {}
    for j in free:
        if j in used or placed[j]['role'] in ('vehicle', 'foliage', 'figure') or \
                any(w in (placed[j]['name'] or '').lower() for w in CAR_NAMES):
            continue
        by_group.setdefault(placed[j]['group'] if placed[j]['group'] not in ('', '-') else f'#{j}', []).append(j)
    if bids:
        for gname, js in by_group.items():
            votes = [composite.building_votes(ras, fr, placed[j], bids) for j in js]
            if gname.startswith('#'):
                for j, v in zip(js, votes):
                    if v and max(v.values()) >= 5:
                        b = max(v, key=v.get)
                        members.setdefault(by_id[b], []).append(j)
                        how.setdefault(by_id[b], 'footprint')
                        used.add(j)
                continue
            tally: dict[int, int] = {}
            for v in votes:
                if v and max(v.values()) >= 3:
                    b = max(v, key=v.get)
                    tally[b] = tally.get(b, 0) + 1
            if tally and max(tally.values()) * 2 >= len(js):
                b = max(tally, key=tally.get)
                members.setdefault(by_id[b], []).extend(js)
                how.setdefault(by_id[b], 'footprint')
                used.update(js)
    rows = []
    counts = {k: 0 for k in STATUS_ORDER}
    tops: dict[int, np.ndarray] = {}

    def top_of(j: int) -> np.ndarray:
        if j not in tops:
            tops[j] = composite.top_points(composite.world_tris(placed[j]['raw'], placed[j]['asset']),
                                           composite.RES_M / 2)
        return tops[j]
    for i, s in enumerate(S):
        o = s['o']
        base = {'id': o['id'], 'cls': o['cls'], 'conf': o['conf'], 'x': round(s['x'], 2), 'y': round(s['y'], 2)}
        empty = {'status': 'missing', 'issues': None, 'row': None, 'actor': None, 'by': None, 'members': 0,
                 'offset_m': None, 'height_off_m': None, 'base_off_m': None, 'yaw_off_deg': None, 'size_ratio': None,
                 'facing_off_deg': None, 'raster': None, 'fix': None}
        js = members.get(i) or []
        cls = o['cls']
        if cls == 'building':
            pts = [top_of(j) for j in js if placed[j]['asset'] is not None]
            pts = np.concatenate(pts) if pts else np.zeros((0, 3))
            rs = composite.compare_building(ras, fr, s, pts, clip=bx)
            if not js or rs['built_cells'] == 0:
                counts['missing'] += 1
                rows.append({**base, **empty, 'members': len(js), 'raster': rs,
                             'by': how.get(i) if js else None})
                continue
            issues = []
            shift = math.hypot(*rs['best_shift_m']) if rs['best_shift_m'] else 0.0
            if shift >= position_m and rs['iou_at_shift'] - rs['iou'] >= composite.SHIFT_GAIN:
                issues.append('offset')
            if rs['iou'] < iou:
                issues.append('shape_off')
            if rs['dz_median_m'] is not None and abs(rs['dz_median_m']) > height_m:
                issues.append('height_off')
            status = min(issues, key=STATUS_ORDER.get) if issues else 'matched'
            counts[status] += 1
            gs = sorted({placed[j]['group'] for j in js})
            fix = None
            if 'offset' in issues and all(placed[j]['tag'] is not None for j in js):
                dx, dy = rs['best_shift_m']
                fix = ' ; '.join(f"edit_group {g} --delta {dx:g},{dy:g},0" for g in gs)
            rows.append({**base, 'status': status, 'issues': ','.join(issues) or None,
                         'row': placed[js[0]]['row'] if len(js) == 1 else None,
                         'actor': placed[js[0]]['actor'] if len(js) == 1 else None, 'by': how.get(i),
                         'members': len(js), 'groups': gs[:6], 'offset_m': round(shift, 2),
                         'height_off_m': rs['dz_median_m'], 'base_off_m': None, 'yaw_off_deg': None,
                         'size_ratio': None, 'facing_off_deg': None, 'raster': rs, 'fix': fix})
            continue
        if not js:
            counts['missing'] += 1
            rows.append({**base, **empty})
            continue
        p = placed[js[0]] if len(js) == 1 else _merged_box([placed[j] for j in js], s, fr)
        tree = cls in ('tree', 'shrub')
        off = math.hypot(p['x'] - s['x'], p['y'] - s['y'])
        dh = p['top'] - s['top']
        tol_h = max(height_m, .2 * o['height_m']) if tree else height_m
        # size ratios, visible size against the scan's oriented box (trees: crown and height)
        if p['asset'] is None:
            ratio = None
        elif tree:
            crown = o.get('crown_m') or (o['length_m'] + o['width_m']) / 2
            pc = p['crown'] or (p['length'] + p['width']) / 2
            ratio = [round(pc / max(crown, .05), 2), round(p['height'] / max(o['height_m'], .05), 2)]
        else:
            ratio = [round(p['length'] / max(o['length_m'], .05), 2), round(p['width'] / max(o['width_m'], .05), 2),
                     round(p['height'] / max(_target(o)[2], .05), 2)]
        # yaw, modulo the row's symmetry and the scan box (which knows the axis, not the front)
        yaw_off = None
        period = p['period']
        if not tree and period and o['length_m'] > 1.15 * o['width_m'] and p['length'] > 1.1 * p['width']:
            yaw_off = round(_ang(p['yaw'] - fr.yaw_out(o['yaw_deg']), min(period, 180.0)), 1)
        # facing of a tilted face (solar panels, mono-pitch roofs) vs the scan's mean normal
        facing_off = None
        if cls in ('solar_panel', 'low_structure') and p['facing'] is not None:
            st, sf = _scan_facing(ras, o)
            if sf is not None and st is not None and st >= 3:
                facing_off = round(_ang(p['facing'] - fr.yaw_out(sf, 360), 360), 1)
        issues = []
        if off > position_m:
            issues.append('offset')
        if yaw_off is not None and yaw_off > yaw_deg:
            issues.append('yaw_off')
        if ratio and any(abs(r - 1) > size_tol for r in ratio[:2 if not tree else 1]):
            issues.append('size_off')
        if abs(dh) > tol_h:
            issues.append('height_off')
        if facing_off is not None and facing_off > max(30.0, 2 * yaw_deg):
            issues.append('facing_off')
        status = min(issues, key=STATUS_ORDER.get) if issues else 'matched'
        counts[status] += 1
        fix = _fix_line(s, o, p, fr, set(issues)) if issues and len(js) == 1 else None
        rows.append({**base, 'status': status, 'issues': ','.join(issues) or None, 'row': p['row'], 'actor': p['actor'],
                     'by': how.get(i), 'members': len(js), 'offset_m': round(off, 2), 'height_off_m': round(dh, 2),
                     'base_off_m': round(p['bottom'] - s['ground'], 2), 'yaw_off_deg': yaw_off, 'size_ratio': ratio,
                     'facing_off_deg': facing_off, 'raster': None, 'fix': fix})
    # placed objects nothing in the scan explains
    extra = []
    for j, p in enumerate(placed):
        if j in used or not p['in_box']:
            continue
        e, sn = fr.to_scan(p['x'], p['y'])
        r, c, inside = ras.cell(np.array([e]), np.array([sn]))
        if not inside[0]:
            continue  # outside the survey: nothing to say
        top_s = float(ras.dsm[r[0], c[0]])
        gnd = float(ras.ground(np.array([e]), np.array([sn]))[0])
        oid = int(ras.objid[r[0], c[0]])
        here = index.get(oid) if oid >= 0 else None
        extra.append({'actor': p['actor'], 'row': p['row'], 'name': p['name'], 'group': p['group'],
                      'x': round(p['x'], 2), 'y': round(p['y'], 2), 'top_z': round(p['top'], 2),
                      'height_m': round(p['zhi'] - p['zlo'], 2),
                      'scan_surface': survey.CLASSES.get(int(ras.cls[r[0], c[0]]), '?'),
                      'scan_height_m': round(top_s - gnd, 2) if math.isfinite(top_s) else None,
                      'scan_object': f"{here['cls']} {here['id']}" if here else None,
                      'base_above_scan_ground_m': round(p['bottom'] - fr.z_out(gnd), 2)})
    extra.sort(key=lambda e: -e['height_m'])
    # height raster over the box, else over the placed objects, clipped to the survey either way (outside it every
    # cell is NaN, and a build spread over kilometres would otherwise sample millions of points for nothing)
    # built heights: foliage by crown discs, everything else by the top surfaces of its mesh (pitched roof slabs,
    # gables) as the building comparison sees them
    solid = [p for p in placed if p['in_box'] and p.get('kind') == 'object' and p['role'] == 'foliage']
    surf = [top_of(j) for j, p in enumerate(placed)
            if p['in_box'] and p.get('kind') == 'object' and p['role'] != 'foliage' and p['asset'] is not None]
    if box is not None:
        region = ops._box4(box)
    elif any(p['in_box'] and p.get('kind') == 'object' for p in placed):
        xs = [p['x'] for p in placed if p['in_box'] and p.get('kind') == 'object']
        ys = [p['y'] for p in placed if p['in_box'] and p.get('kind') == 'object']
        region = (min(xs) - cell_m, min(ys) - cell_m, max(xs) + cell_m, max(ys) + cell_m)
    else:
        region = None
    if region is not None:
        region = clip_region(region, survey_extent(meta, fr))
    cells = _height_cells(ras, fr, solid, region, cell_m, height_m, top,
                          points=np.concatenate(surf) if surf else None) if region else None
    rows.sort(key=lambda r: (-severity(r), -r['conf'], r['id']))
    ground_total = sum(ground.values())
    ground_brief = {'objects': ground_total,
                    'by_group': dict(sorted(ground.items(), key=lambda kv: -kv[1])[:8]),
                    'meaning': 'tiles, road markings, decals and lawn foliage on the ground: not compared'}
    steps = ground_steps([p for p in placed if p['in_box']], fr, ras, flat_tol_m, top)
    report = {'survey': meta['key'], 'source': source, 'scope': sc, 'classes': want, 'min_conf': min_conf,
              'tolerances': {'position_m': position_m, 'height_m': height_m, 'yaw_deg': yaw_deg, 'size': size_tol,
                             'iou': iou},
              'counts': counts, 'objects': rows, 'unmatched_placed': extra, 'ground': {**ground_brief, 'by_group': ground},
              'gameplay': gameplay, 'tagged_outside_selection': tagged_other, 'height_cells': cells,
              'ground_steps': steps, 'compare_box': list(bx) if bx else None,
              'box': list(region) if region else None, 'units': 'metres', 'created': store.now()}
    path = store.home(meta['key']) / 'compare.json'
    store.write_json(path, report)
    by_cls: dict[str, dict] = {}
    for r in rows:
        b = by_cls.setdefault(r['cls'], {'cls': r['cls'], 'scan': 0, 'matched': 0, 'missing': 0, 'off': 0})
        b['scan'] += 1
        b['matched' if r['status'] == 'matched' else 'missing' if r['status'] == 'missing' else 'off'] += 1
    problems = [r for r in rows if r['status'] != 'matched']
    brief_cells = None
    if cells:
        brief_cells = {k: cells[k] for k in ('cell_m', 'cells_with_height', 'median_abs_m', 'within_tol_share',
                                             'too_low', 'too_high')}
        brief_cells['worst'] = cells['worst'][:min(top, 8)]
    buildings = [r for r in rows if r['cls'] == 'building' and r['raster']]
    return {'survey': meta['key'], 'frame': 'map', 'source': source, 'scan_objects': len(S),
            'placed_in_scope': sum(1 for p in placed if p['in_box']), 'counts': counts,
            'unmatched_placed_count': len(extra), 'ground': ground_brief, 'gameplay': gameplay,
            'ground_steps': steps and {**{k: v for k, v in steps.items() if k not in ('worst', 'far')},
                                       'worst': steps['worst'][:min(top, 8)], 'far': steps['far'][:min(top, 8)]},
            'tagged_outside_selection': sum(tagged_other.values()),
            'by_class': sorted(by_cls.values(), key=lambda r: -r['scan']),
            'buildings': [{'id': r['id'], 'status': r['status'], 'by': r['by'], 'members': r['members'],
                           **{k: r['raster'][k] for k in ('iou', 'coverage', 'missing_m2', 'extra_m2', 'dz_median_m',
                                                          'dz_p90_m', 'top_built_m', 'top_scan_m', 'best_shift_m')}}
                          for r in buildings][:top],
            'problems': problems[:top],
            'unmatched_placed': extra[:top], 'height_cells': brief_cells,
            'truncated': len(problems) > top or len(extra) > top or len(buildings) > top,
            'report_json': str(path), 'units': 'metres',
            'meaning': 'pairs by group tag first (scan.<class>.<id>, <class>.<id>, b<id>[suffix]: every object of '
                       'the group stands for that scan object), then the nearest placed object of a fitting kind; '
                       'buildings also take untagged pieces standing on their cells (by: footprint) and are compared '
                       'as height rasters (0.5 m): iou of the raised footprints, coverage of the scan building, '
                       'missing / extra m2, dz median (built - scan top on shared cells), best_shift_m (the move '
                       'within 3 m that overlaps best; offset when it gains iou), inside the box when one is given. Point objects are measured by their visible size (asset_profile). status: worst '
                       'issue (issues lists all); shape_off: iou under the tolerance; offset_m: visible centres; '
                       'height_off_m: placed top minus scan top; base_off_m: placed bottom minus scan ground; '
                       'yaw_off_deg: long axes, modulo the row symmetry; size_ratio: placed / scan for length, width, '
                       'height (trees: crown, height); fix: an agent transform line (cm, degrees, - keeps) or an '
                       'edit_group delta (preview first); ground: thin pieces on the ground, counted only; '
                       'unmatched_placed: other placed objects nothing explains, tallest first; height_cells: scan '
                       'height above ground vs built height per cell (95th percentiles)',
            'limitations': ['Read-only bookkeeping of shapes: it does not show that a stand-in looks right; look with '
                            'capture_topdown and scan_view --compare.',
                            'Building rasters use the simplified hull of each row (else its box) seen from above; '
                            'party walls and pieces between buildings count for the building they are tagged to.',
                            'fix lines keep z (the bottom); run drop on the actor afterwards when the ground differs.',
                            'Ground pieces are not compared one by one; ground_steps measures the seams between '
                            'ground tiles (a continuous surface has none over 1 cm except kerbs) and flags tiles far '
                            'from the scan ground.']}


# ---------------------------------------------------------------------------- contact sheet


def scan_sheet(points=None, objects=None, classes=None, box=None, min_conf: float = 0.0, limit: int = 16,
               size_m: float = 0.0, layer: str = 'ortho', labels: bool = True, px: int = 360, columns: int = 0,
               frame: str = 'auto', survey_key: str = '') -> dict:
    from . import view
    from .. import sheet
    layer = (layer or 'ortho').lower()
    if layer not in view.LAYERS:
        raise ValueError(f"layer must be one of {', '.join(view.LAYERS)}")
    if not (size_m == 0 or 4 <= size_m <= 300):
        raise ValueError('size_m must be 0 (fit each object) or 4..300 metres')
    lim = max(1, min(int(limit), 36))
    px = max(160, min(int(px), 800))
    if not 0 <= int(columns) <= 12:
        raise ValueError('columns must be 0 (automatic) to 12')
    meta, fr, ras = ops._context(survey_key, frame)
    idx = ops._obj_index(meta)
    tiles = []
    for p in (points or []):
        q = ops._xy([p], 'points')[0]
        tiles.append({'label': f'{q[0]:g},{q[1]:g}', 'x': float(q[0]), 'y': float(q[1]), 'size': size_m or 40.0,
                      'object': None})
    ids = []
    for v in objects or []:
        try:
            ids.append(int(str(v).strip()))
        except ValueError:
            raise ValueError(f'objects are survey object ids, got {v!r}') from None
    chosen = [idx[i] for i in ids if i in idx]
    missing = [i for i in ids if i not in idx]
    if missing:
        raise ValueError(f"no objects {', '.join(map(str, missing))} in survey {meta['key']}")
    if classes or (box is not None and not points and not objects):
        sel, _ = _select(meta, fr, classes, box, min_conf)
        chosen += sorted(sel, key=lambda o: (-o['conf'], o['id']))
    for o in chosen:
        x, y = fr.from_scan(o['x'], o['y'])
        auto = min(120.0, max(14.0, 3 * max(o['length_m'], o['width_m'])))
        tiles.append({'label': f"{o['cls']} {o['id']} c{o['conf']:g} {o['length_m']:g}x{o['width_m']:g}m",
                      'x': float(x), 'y': float(y), 'size': size_m or auto, 'object': o['id']})
    if not tiles:
        raise ValueError('give points x,y, objects ids, or classes / box to pick objects')
    total = len(tiles)
    tiles = tiles[:lim]
    objs = store.objects(meta)
    shots = []
    for t in tiles:
        h = t['size'] / 2
        path, _drawn, _mpp = view.render(meta, fr, ras, objs, (t['x'] - h, t['y'] - h, t['x'] + h, t['y'] + h), layer,
                                         labels, None, px)
        shots.append({**t, 'image': str(path)})
    from PIL import Image
    with Image.open(shots[0]['image']) as im:
        w, hgt = im.size
    cols = int(columns) or math.ceil(math.sqrt(len(shots)))
    out = store.views_dir(meta['key']) / f"sheet-{time.strftime('%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.png"
    sheet.compose(shots, w, hgt, min(cols, len(shots)), out,
                  caption=lambda i, s: f"{i + 1} {s['label']}  @{s['x']:.1f},{s['y']:.1f}  {s['size']:g} m")
    for s in shots:
        Path(s['image']).unlink(missing_ok=True)
    return {'survey': meta['key'], 'frame': fr.name, 'image': str(out), 'layer': layer, 'tiles': len(shots),
            'selected': total, 'truncated': total > lim,
            'crops': [{'n': i + 1, 'label': s['label'], 'object': s['object'], 'x': round(s['x'], 2),
                       'y': round(s['y'], 2), 'size_m': round(s['size'], 1)} for i, s in enumerate(shots)],
            'units': 'metres',
            'note': 'image in RAM (/dev/shm) only: look at it locally, never upload or commit it'}
