"""Vision stage of scan_survey: learned detectors on the ortho image and on facade renders, fused with geometry.

The detectors run in a separate environment (`.venv-vision`, torch + ultralytics + transformers) as one subprocess
per survey (scan/vision_worker.py, JSON in / out), so the CLI never imports torch and works without it. Images go
to /dev/shm only. Models: yolo11l-obb (DOTA aerial classes, oriented boxes), Grounding DINO base (text prompts,
aerial and facades), SAM 2.1 base (outlines of DINO-only vehicles / pools); weights in ~/.cache/uncrashed-mcp/models
(downloaded on first use).

Fusion (scan frame, metres):
- vehicles: OBB and DINO boxes from overlapping tiles are clustered; a cluster on a roof (inside a building, high
  above ground) or over a hole in the scan is dropped. A geometry car matched by a cluster takes its outline and
  yaw (src mesh+vision); an unmatched geometry car is re-labelled low_structure / small_object (skips, sheds,
  carports, kiosks, water noise) unless it lies on the edge of the image (kept, lower conf); a cluster on a small
  geometry blob relabels it; any other cluster becomes a new car / truck (src vision). Truck = longer than 6.2 m.
- other aerial classes (pool, courts, fields, tanks, boats, solar panels, playgrounds) are added as objects.
- trees: a DINO tree box around a geometry crown raises its conf.
- facades: per building wall (OSM wall line when mapped, else roof outline) a straight-on render; DINO boxes of
  windows / doors / balconies / garage doors become a window grid (facade.fit), stored in facades.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from . import facade, grid, native, store

REPO = Path(__file__).resolve().parents[3]
VENV = Path(os.environ.get('UNCRASHED_VISION_VENV', REPO / '.venv-vision'))
MODELS = Path(os.environ.get('UNCRASHED_MODELS', Path.home() / '.cache' / 'uncrashed-mcp' / 'models'))
WORKER = Path(__file__).with_name('vision_worker.py')
OBB_MODEL = 'yolo11l-obb.pt'
SAM_MODEL = 'sam2.1_b.pt'
DINO_MODEL = 'grounding-dino-base'
REQUIRED = (OBB_MODEL, DINO_MODEL, SAM_MODEL)
AERIAL_PROMPTS = ['car', 'van', 'truck', 'bus', 'tree', 'solar panel', 'swimming pool', 'playground']
FACADE_PROMPTS = ['window', 'door', 'balcony', 'garage door']
TILE_PX, TILE_OVERLAP = 512, 96          # fine ortho tiles (0.125 m/px; YOLO upsamples to 1024)
COARSE = 2                                # coarse tiles: ortho / 2 (0.25 m/px), 1024 px
TIMEOUT_S = float(os.environ.get('UNCRASHED_VISION_TIMEOUT', 1800))
MAX_FACADES = 2500
DOTA = {'swimming pool': 'pool', 'tennis court': 'tennis_court', 'basketball court': 'basketball_court',
        'soccer ball field': 'sports_field', 'ground track field': 'sports_field', 'baseball diamond': 'sports_field',
        'storage tank': 'storage_tank', 'roundabout': 'roundabout', 'ship': 'boat', 'plane': 'plane',
        'helicopter': 'helicopter'}
LARGE_CLASSES = {'tennis_court', 'basketball_court', 'sports_field', 'roundabout', 'storage_tank'}
MODEL_TAG = {'obb': 'yolo11l-obb', 'dino': 'gdino-base', 'sam': 'sam2.1-b'}


class VisionError(RuntimeError):
    pass


# ---------------------------------------------------------------------------- availability


def _model_present(name: str) -> bool:
    p = MODELS / name
    return (p / 'config.json').is_file() if not name.endswith('.pt') else p.is_file()


def available() -> dict:
    python = VENV / 'bin' / 'python'
    present = [m for m in REQUIRED if _model_present(m)]
    missing = [m for m in REQUIRED if m not in present]
    return {'venv': str(VENV) if python.is_file() else None, 'models_dir': str(MODELS), 'models': present,
            'missing': missing, 'ready': python.is_file(),
            'note': ('missing models are downloaded on the first vision run' if missing and python.is_file() else
                     None if python.is_file() else 'no .venv-vision: scan_survey runs geometry + OSM only')}


def ready() -> tuple[bool, str]:
    if os.environ.get('UNCRASHED_VISION', '').lower() in ('0', 'off', 'no'):
        return False, 'disabled by UNCRASHED_VISION'
    if not (VENV / 'bin' / 'python').is_file():
        return False, f'no vision environment at {VENV} (geometry and OSM evidence only)'
    return True, ''


# ---------------------------------------------------------------------------- geometry helpers


def _poly_area(p: np.ndarray) -> float:
    return grid.ring_area(np.asarray(p, float))


def _clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: convex subject clipped by convex clip polygon (both CCW or both CW)."""
    out = [tuple(p) for p in subject]
    sign = 1.0 if _signed(clip) > 0 else -1.0
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        inp, out = out, []
        if not inp:
            break
        def side(p):
            return sign * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))
        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return np.asarray(out, float).reshape(-1, 2)


def _signed(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def iou(p: np.ndarray, q: np.ndarray) -> float:
    inter = _clip(np.asarray(p, float), np.asarray(q, float))
    ia = _poly_area(inter) if len(inter) >= 3 else 0.0
    u = _poly_area(p) + _poly_area(q) - ia
    return ia / u if u > 0 else 0.0


def rect(corners: np.ndarray) -> tuple[float, float, float, float, float]:
    """(cx, cy, length, width, yaw_deg 0..180 along the long side) of a 4-corner box."""
    c = np.asarray(corners, float)
    e1, e2 = c[1] - c[0], c[2] - c[1]
    l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    long = e1 if l1 >= l2 else e2
    yaw = math.degrees(math.atan2(long[1], long[0])) % 180
    cx, cy = c.mean(0)
    return float(cx), float(cy), max(l1, l2), min(l1, l2), yaw


# ---------------------------------------------------------------------------- tiles


def _tiles(ortho: np.ndarray, out_dir: Path) -> tuple[list[dict], list[dict]]:
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = ortho.shape[:2]
    fine, coarse = [], []
    step = TILE_PX - TILE_OVERLAP
    for r0 in range(0, max(1, H - TILE_OVERLAP), step):
        for c0 in range(0, max(1, W - TILE_OVERLAP), step):
            t = np.asarray(ortho[r0:r0 + TILE_PX, c0:c0 + TILE_PX])
            if (t.max(2) > 0).mean() < 0.05:
                continue
            p = out_dir / f'f_{r0}_{c0}.png'
            Image.fromarray(np.ascontiguousarray(t)).save(p, compress_level=1)
            fine.append({'path': str(p), 'r0': r0, 'c0': c0, 'scale': 1, 'h': t.shape[0], 'w': t.shape[1]})
    small = np.asarray(Image.fromarray(np.asarray(ortho)).resize((W // COARSE, H // COARSE), Image.BILINEAR))
    h2, w2 = small.shape[:2]
    cs, ov = 1024, 128
    for r0 in range(0, max(1, h2 - ov), cs - ov):
        for c0 in range(0, max(1, w2 - ov), cs - ov):
            t = small[r0:r0 + cs, c0:c0 + cs]
            if (t.max(2) > 0).mean() < 0.05:
                continue
            p = out_dir / f'c_{r0}_{c0}.png'
            Image.fromarray(np.ascontiguousarray(t)).save(p, compress_level=1)
            coarse.append({'path': str(p), 'r0': r0 * COARSE, 'c0': c0 * COARSE, 'scale': COARSE, 'h': t.shape[0],
                           'w': t.shape[1]})
    return fine, coarse


def _px_to_scan(meta: dict, tile: dict, x, y):
    o = meta['ortho']
    g = meta['grid']
    res = o['res']
    return (g['x0'] + (tile['c0'] + np.asarray(x, float) * tile['scale']) * res,
            g['y0'] + (tile['r0'] + np.asarray(y, float) * tile['scale']) * res)


def _cls_of(label: str) -> str | None:
    s = label.lower()
    for key, name in (('solar', 'solar_panel'), ('pool', 'pool'), ('playground', 'playground'),
                      ('truck', 'truck'), ('bus', 'truck'), ('van', 'car'), ('car', 'car'), ('tree', 'tree')):
        if key in s:
            return name
    return None


def candidates(meta: dict, fine: list[dict], coarse: list[dict], res: dict) -> list[dict]:
    """Detections of every tile in scan metres: {cls, src, score, corners (4x2), oriented, label}."""
    out = []
    for tiles, dets_all, kind in ((fine, res.get('obb') or [], 'obb'), (coarse, res.get('coarse') or [], 'obb'),
                                  (fine, res.get('dino') or [], 'dino')):
        for tile, dets in zip(tiles, dets_all):
            margin = 6 / tile['scale']   # boxes cut by the tile edge are dropped where tiles overlap
            for d in dets:
                if kind == 'obb':
                    lab = d['class']
                    cls = 'car' if lab == 'small vehicle' else 'truck' if lab == 'large vehicle' else DOTA.get(lab)
                    if cls is None:
                        continue
                    if tile['scale'] > 1 and cls in ('car', 'truck'):
                        continue
                    if tile['scale'] == 1 and cls in LARGE_CLASSES:
                        continue
                    px = np.asarray(d['obb'], float)
                    oriented = True
                else:
                    lab = d['class']
                    cls = _cls_of(lab)
                    if cls is None:
                        continue
                    x0, y0, x1, y1 = d['box']
                    m = d.get('mask')
                    if m and m.get('rect'):
                        cx, cy, w, h, ang = m['rect']
                        a = math.radians(ang)
                        du, dv = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
                        px = np.array([[cx, cy] + su * w / 2 * du + sv * h / 2 * dv
                                       for su, sv in ((1, 1), (-1, 1), (-1, -1), (1, -1))])
                        oriented = True
                    else:
                        px = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
                        oriented = False
                xs, ys = px[:, 0], px[:, 1]
                edge = (xs.min() < margin and tile['c0'] > 0) or (ys.min() < margin and tile['r0'] > 0) or \
                       (xs.max() > tile['w'] - margin and tile['w'] == (TILE_PX if tile['scale'] == 1 else 1024)) or \
                       (ys.max() > tile['h'] - margin and tile['h'] == (TILE_PX if tile['scale'] == 1 else 1024))
                if edge:
                    continue
                e, s = _px_to_scan(meta, tile, xs, ys)
                out.append({'cls': cls, 'src': kind, 'score': float(d['score']), 'label': lab,
                            'corners': np.stack([e, s], 1), 'oriented': oriented,
                            'mask': bool(d.get('mask'))})
    return out


def cluster(cands: list[dict], classes: set[str], dist_m: float = 1.2) -> list[dict]:
    """Greedy merge of overlapping detections (oriented boxes first, then by score)."""
    pool = [c for c in cands if c['cls'] in classes]
    pool.sort(key=lambda c: (not c['oriented'], c['src'] != 'obb', -c['score']))
    groups: list[dict] = []
    for c in pool:
        cx, cy, L, Wd, yaw = rect(c['corners'])
        best = None
        for g in groups:
            d = math.hypot(cx - g['x'], cy - g['y'])
            if d < max(dist_m, 0.35 * min(L, g['length'])) or iou(c['corners'], g['corners']) > 0.3:
                if best is None or d < best[0]:
                    best = (d, g)
        if best is None:
            groups.append({'x': cx, 'y': cy, 'length': L, 'width': Wd, 'yaw': yaw, 'corners': c['corners'],
                           'cls': c['cls'], 'oriented': c['oriented'], 'scores': {c['src']: c['score']},
                           'sam': c['mask'], 'labels': {c['label']}})
        else:
            g = best[1]
            g['scores'][c['src']] = max(g['scores'].get(c['src'], 0.0), c['score'])
            g['labels'].add(c['label'])
            g['sam'] = g['sam'] or c['mask']
            if c['cls'] != g['cls']:
                g.setdefault('other', set()).add(c['cls'])
    for g in groups:   # YOLO calls many vans / estate cars 'large vehicle': size decides
        if g['cls'] in ('car', 'truck') and ('truck' in g.get('other', ()) or g['cls'] == 'truck'):
            g['cls'] = 'truck' if g['length'] > 6.2 else 'car'
    return groups


# ---------------------------------------------------------------------------- fusion


class _Ras:
    """The few raster reads fusion needs (analysis grid)."""

    def __init__(self, meta: dict, dsm, dtm, cls, objid, rgb):
        g = meta['grid']
        self.x0, self.y0, self.res, self.w, self.h = g['x0'], g['y0'], g['res'], g['w'], g['h']
        self.dsm, self.dtm, self.cls, self.objid, self.rgb = dsm, dtm, cls, objid, rgb

    def cells(self, corners: np.ndarray):
        c = (corners[:, 0] - self.x0) / self.res - .5
        r = (corners[:, 1] - self.y0) / self.res - .5
        c0, c1 = max(0, int(math.floor(c.min()))), min(self.w - 1, int(math.ceil(c.max())))
        r0, r1 = max(0, int(math.floor(r.min()))), min(self.h - 1, int(math.ceil(r.max())))
        if c0 > c1 or r0 > r1:
            return np.empty(0, int), np.empty(0, int)
        m = np.zeros((r1 - r0 + 1, c1 - c0 + 1), np.int32)
        native.fill_poly(m, c - c0, r - r0, 1)
        rr, cc = np.nonzero(m)
        return rr + r0, cc + c0

    def stats(self, corners: np.ndarray) -> dict | None:
        rr, cc = self.cells(corners)
        if not len(rr):
            return None
        top = np.asarray(self.dsm[rr, cc], float)
        gz = np.asarray(self.dtm[rr, cc], float)
        ok = np.isfinite(top)
        if ok.sum() < 2:
            return None
        nd = top[ok] - gz[ok]
        cl = np.asarray(self.cls[rr, cc])
        ids = np.asarray(self.objid[rr, cc])
        return {'valid': float(ok.mean()), 'ground_z': float(np.median(gz)), 'height': float(np.percentile(nd, 95)),
                'nd_median': float(np.median(nd)), 'building_share': float((cl == 10).mean()),
                'rgb': [int(v) for v in np.median(np.asarray(self.rgb[rr, cc]), 0)], 'ids': ids, 'rows': rr, 'cols': cc}


def _obj(oid: int, cls: str, conf: float, src: str, g: dict, st: dict, models: list[str]) -> dict:
    gz, h = st['ground_z'], max(0.0, st['height'])
    return {'id': oid, 'cls': cls, 'conf': round(min(0.97, conf), 2), 'src': src, 'x': round(g['x'], 2),
            'y': round(g['y'], 2), 'ground_z': round(gz, 2), 'top_z': round(gz + h, 2), 'height_m': round(h, 2),
            'length_m': round(g['length'], 2), 'width_m': round(g['width'], 2), 'yaw_deg': round(g['yaw'] % 180, 1),
            'area_m2': round(g['length'] * g['width'], 1), 'rgb': st['rgb'], 'models': models}


def _models(scores: dict) -> list[str]:
    return [f"{MODEL_TAG.get(k, k)}:{v:.2f}" for k, v in sorted(scores.items())]


CLUTTER_IRREGULAR_M = 1.5   # p90 - p30 of the body heights (foliage cells left out): a car body is even
CLUTTER_TALL = 0.2          # share of the box standing above a car (2.4 m) that is not foliage


def clutter(o: dict, ras: _Ras) -> str | None:
    """Why a car-class box is a stack of pallets, barrels, furniture or a kiosk rather than a car, or None: the
    heights inside the box (0.3 m in from its edge, foliage cells left out) are uneven where a car body is even,
    or part of it stands taller than a car without being foliage, together with a length a car rarely has or a
    vision-only source. Cars under crowns are judged on the cells the crown leaves visible."""
    L, W = o['length_m'], o['width_m']
    rr, cc = ras.cells(grid.box_corners(o['x'], o['y'], max(L - 0.6, 0.5), max(W - 0.6, 0.5), o['yaw_deg']))
    if len(rr) < 6:
        return None
    top = np.asarray(ras.dsm[rr, cc], float)
    nd = top - np.asarray(ras.dtm[rr, cc], float)
    keep = np.isfinite(nd) & (np.asarray(ras.cls[rr, cc]) != 11)
    if keep.sum() < 6:
        return None
    z = nd[keep]
    uneven = float(np.percentile(z, 90) - np.percentile(z, 30))
    tall = float((z > 2.4).mean())
    odd = L < 3.0 or o.get('src') == 'vision' or tall > CLUTTER_TALL
    if uneven > CLUTTER_IRREGULAR_M and odd:
        return f'uneven top ({uneven:.1f} m spread) ' + ('and ' + (f'{L:.1f} m long' if L < 3.0 else
                                                               'vision only' if o.get('src') == 'vision' else
                                                               f'{tall:.0%} taller than a car'))
    if uneven > 1.0 and L < 3.0:
        return f'uneven top ({uneven:.1f} m spread), {L:.1f} m long'
    return None


def fuse(objects: list[dict], cands: list[dict], ras: _Ras) -> tuple[list[dict], dict]:
    """New object list (geometry ids kept; added objects appended) and a fusion report."""
    objs = [dict(o) for o in objects]
    rep = {'vehicles_detected': 0, 'cars_confirmed': 0, 'cars_rejected': 0, 'cars_kept_unconfirmed': 0,
           'vehicles_added': 0, 'vehicles_dropped_on_roofs': 0, 'other_added': {}, 'trees_confirmed': 0}
    veh = cluster(cands, {'car', 'truck'})
    good = []
    for g in veh:
        dino_only = 'obb' not in g['scores']
        if dino_only:
            if not (2.0 <= g['length'] <= (14 if g['cls'] == 'truck' else 7.5) and 1.0 <= g['width'] <= 4.5):
                continue
            if g['scores']['dino'] < 0.3:
                continue
        st = ras.stats(g['corners'])
        if st is None:
            continue
        if st['valid'] < 0.85:   # on the edge of the scan / a hole: no reliable image there
            rep['vehicles_dropped_no_data'] = rep.get('vehicles_dropped_no_data', 0) + 1
            continue
        if st['building_share'] > 0.5 and st['nd_median'] > 2.5:
            rep['vehicles_dropped_on_roofs'] += 1
            continue
        g['st'] = st
        good.append(g)
    rep['vehicles_detected'] = len(good)
    geo = [o for o in objs if o['cls'] == 'car']
    used = set()
    pairs = []
    for o in geo:
        for k, g in enumerate(good):
            d = math.hypot(o['x'] - g['x'], o['y'] - g['y'])
            if d < max(1.6, 0.45 * max(o['length_m'], g['length'])):
                pairs.append((d, o['id'], k))
    pairs.sort()
    match: dict[int, int] = {}
    for d, oid, k in pairs:
        if oid in match or k in used:
            continue
        match[oid] = k
        used.add(k)
    by_id = {o['id']: o for o in objs}
    for o in geo:
        k = match.get(o['id'])
        if k is not None:
            g = good[k]
            s = max(g['scores'].get('obb', 0.0), min(1.0, 1.6 * g['scores'].get('dino', 0.0)))
            o['conf'] = round(min(0.97, max(o['conf'], 0.55 + 0.45 * s)), 2)
            o['src'] = 'mesh+vision'
            o['models'] = _models(g['scores'])
            if g['oriented']:
                o.update(x=round(g['x'], 2), y=round(g['y'], 2), yaw_deg=round(g['yaw'] % 180, 1),
                         length_m=round(g['length'], 2), width_m=round(g['width'], 2),
                         area_m2=round(g['length'] * g['width'], 1))
            if g['cls'] == 'truck' or o['length_m'] > 6.2:
                o['cls'] = 'truck' if o['length_m'] > 6.2 else 'car'
            rep['cars_confirmed'] += 1
            continue
        st = ras.stats(grid.box_corners(o['x'], o['y'], o['length_m'] + 2, o['width_m'] + 2, o['yaw_deg']))
        if st is None or st['valid'] >= 0.85:   # the image shows the spot, and no detector saw a vehicle there
            o['was'] = 'car'
            o['cls'] = 'low_structure' if o['length_m'] > 6 else 'small_object'
            o['conf'] = 0.3
            o['note'] = 'car-sized blob without a vehicle in the aerial image (carport, shed, kiosk, trailer?)'
            o['src'] = 'mesh+vision'
            rep['cars_rejected'] += 1
        else:
            o['conf'] = round(o['conf'] * 0.6, 2)
            o['note'] = 'no aerial vehicle detection; on the edge of the scan image'
            rep['cars_kept_unconfirmed'] += 1
    # a vehicle on a small geometry object (melted car read as a blob, shrub, small structure) relabels it
    small = [o for o in objs if o['cls'] in ('small_object', 'low_structure', 'shrub') and o['length_m'] <= 7.0
             and 'was' not in o]
    for k, g in enumerate(good):
        if k in used or not small:
            continue
        d, o = min(((math.hypot(o['x'] - g['x'], o['y'] - g['y']), o) for o in small), key=lambda t: t[0])
        if d > max(1.5, 0.4 * g['length']):
            continue
        s = max(g['scores'].get('obb', 0.0), min(1.0, 1.6 * g['scores'].get('dino', 0.0)))
        o.update(was=o['cls'], cls=g['cls'], conf=round(min(0.95, 0.45 + 0.45 * s), 2), src='mesh+vision',
                 models=_models(g['scores']), note='vehicle in the aerial image on a blob the geometry did not call a car')
        if g['oriented']:
            o.update(x=round(g['x'], 2), y=round(g['y'], 2), yaw_deg=round(g['yaw'] % 180, 1),
                     length_m=round(g['length'], 2), width_m=round(g['width'], 2),
                     area_m2=round(g['length'] * g['width'], 1))
        small.remove(o)
        used.add(k)
        rep['vehicles_relabelled'] = rep.get('vehicles_relabelled', 0) + 1
    nid = max([o['id'] for o in objs], default=-1) + 1
    for k, g in enumerate(good):
        if k in used:
            continue
        st = g['st']
        s_obb, s_dino = g['scores'].get('obb', 0.0), g['scores'].get('dino', 0.0)
        conf = 0.3 + 0.6 * s_obb if s_obb else 0.2 + 0.9 * s_dino
        if s_obb and s_dino:
            conf += 0.1
        if st['height'] < 0.5:
            conf *= 0.75
        o = _obj(nid, g['cls'], conf, 'vision', g, st, _models(g['scores']))
        o['yaw_from'] = 'box' if not g['oriented'] else ('mask' if g['sam'] and 'obb' not in g['scores'] else 'obb')
        objs.append(o)
        by_id[nid] = o
        nid += 1
        rep['vehicles_added'] += 1
    # stacks, pallets, barrels, outdoor furniture: detectors call some of them small vehicles
    for o in objs:
        if o['cls'] != 'car':
            continue
        why = clutter(o, ras)
        if why:
            o.update(was='car', cls='small_object', conf=0.35, note=f'not a car: {why}')
            rep['cars_rejected_clutter'] = rep.get('cars_rejected_clutter', 0) + 1
    # other aerial classes
    others = cluster(cands, set(DOTA.values()) | {'solar_panel', 'playground'}, dist_m=3.0)
    for g in others:
        s = max(g['scores'].values())
        if s < (0.35 if 'obb' in g['scores'] else 0.55 if g['cls'] == 'pool' else 0.3):
            continue
        if g['cls'] == 'playground' and not 5 <= g['length'] <= 60:
            continue
        if g['cls'] == 'solar_panel' and g['length'] > 40:
            continue
        st = ras.stats(g['corners'])
        if st is None:
            continue
        conf = 0.3 + 0.6 * s if g['cls'] != 'solar_panel' else min(0.5, 0.3 * s / 0.5)  # boxes cover parts of arrays
        o = _obj(nid, g['cls'], conf, 'vision', g, st, _models(g['scores']))
        ids = st['ids'][st['ids'] >= 0]
        if len(ids):
            host = by_id.get(int(np.bincount(ids).argmax()))
            if host is not None and host['cls'] in ('building', 'building_part'):
                o['on'] = host.get('parent', host['id']) if host['cls'] == 'building_part' else host['id']
        objs.append(o)
        by_id[nid] = o
        nid += 1
        rep['other_added'][g['cls']] = rep['other_added'].get(g['cls'], 0) + 1
    # tree confirmation
    trees = [c for c in cands if c['cls'] == 'tree' and c['score'] >= 0.3]
    if trees:
        boxes = np.array([[c['corners'][:, 0].min(), c['corners'][:, 1].min(), c['corners'][:, 0].max(),
                           c['corners'][:, 1].max()] for c in trees])
        small = (boxes[:, 2] - boxes[:, 0] <= 20) & (boxes[:, 3] - boxes[:, 1] <= 20)
        boxes = boxes[small]
        for o in objs:
            if o['cls'] != 'tree' or not len(boxes):
                continue
            inb = (boxes[:, 0] <= o['x']) & (o['x'] <= boxes[:, 2]) & (boxes[:, 1] <= o['y']) & (o['y'] <= boxes[:, 3])
            if inb.any():
                o['conf'] = round(min(0.95, o['conf'] + 0.1), 2)
                o['models'] = ['gdino-base:tree']
                rep['trees_confirmed'] += 1
    return objs, rep


def stamp(objid: np.ndarray, objects: list[dict], meta: dict) -> None:
    """Write vision- and street-view-added objects into the object-id raster (in place)."""
    g = meta['grid']
    for o in objects:
        if o.get('src') not in ('vision', 'street'):
            continue
        c = grid.box_corners(o['x'], o['y'], o['length_m'], o['width_m'], o['yaw_deg'])
        native.fill_poly(objid, (c[:, 0] - g['x0']) / g['res'] - .5, (c[:, 1] - g['y0']) / g['res'] - .5, o['id'])


# ---------------------------------------------------------------------------- facades


def wall_ring(o: dict, osm_rings: dict, shift) -> tuple[list, str]:
    ref = (o.get('osm') or {}).get('osm')
    if ref and ref in osm_rings:
        r = np.asarray(osm_rings[ref], float) + np.asarray(shift, float)
        return r.tolist(), 'osm'
    return o['footprint'], 'roof_outline'


def plan_facades(meta: dict, objects: list[dict], ras, osm_rings: dict, shift=(0.0, 0.0),
                 only: set[int] | None = None) -> list[dict]:
    """Facade specs (no pixels) of every building wall worth looking at."""
    specs = []
    for o in objects:
        if o['cls'] != 'building' or (only is not None and o['id'] not in only):
            continue
        if only is None and (o['area_m2'] < 25 or o['height_m'] < 3.0):
            continue
        ring, src = wall_ring(o, osm_rings, shift)
        for side, w in enumerate(facade.walls(ring)):
            if src == 'osm' and not facade.belongs(o['footprint'], w['a'], w['b']):
                continue   # the OSM outline covers several mesh buildings: this wall is another part's
            t = np.linspace(0.05, 0.95, 7)
            pts = w['a'][None, :] + t[:, None] * (w['b'] - w['a'])[None, :] + 1.0 * w['n'][None, :]
            gz = np.asarray(ras.ground(pts[:, 0], pts[:, 1]), float)
            ground = float(np.median(gz))
            z0 = float(gz.min()) - 0.4
            z1 = o['top_z'] + 0.5
            if z1 - z0 < 2.5:
                continue
            occ = facade.occluded_front(ras, w, ground, o['ground_z'] + (o.get('eave_m') or o['height_m']),
                                       own_id=o['id'])
            p, (W, H) = facade.plane(w, z0, z1)
            specs.append({'building': o['id'], 'side': side, 'wall_from': src,
                          'a': [round(float(v), 2) for v in w['a']], 'b': [round(float(v), 2) for v in w['b']],
                          'length_m': round(w['length_m'], 2), 'faces_deg': round(w['faces_deg'], 1),
                          'ground_up': round(ground, 2), 'eave_m': o.get('eave_m'), 'plane': p, 'size': [W, H],
                          'front_blocked': round(occ, 2)})
    return specs


def render_facades(meta: dict, specs: list[dict], out_dir: Path | None = None) -> tuple[list, list, dict]:
    from .. import reference
    cache = reference.ensure_index(Path(meta['obj']))
    amap = reference._axis_map(meta['axes'])
    imgs, deps, st = native.facades(cache, amap, np.array([s['plane'] for s in specs], float).reshape(-1, 12),
                                    np.array([s['size'] for s in specs], int).reshape(-1, 2), skip=_coarse_meshes(cache))
    if out_dir is not None:
        from PIL import Image
        out_dir.mkdir(parents=True, exist_ok=True)
        for s, im in zip(specs, imgs):
            s['image'] = str(out_dir / f"b{s['building']}_s{s['side']}.png")
            Image.fromarray(im).save(s['image'], compress_level=1)
    return imgs, deps, st


def _coarse_meshes(cache: Path) -> np.ndarray | None:
    """1 for meshes of an octree tile that has a finer descendant (coarser level of detail)."""
    from .. import reference
    _tex, meshes = reference.read_index(cache)
    tiles = [m['name'].split('_')[0] for m in meshes]
    if not tiles or not all(t and set(t) <= set('01234567') for t in tiles):
        return None
    prefixes = set()
    for t in tiles:
        for n in range(1, len(t)):
            prefixes.add(t[:n])
    return np.array([t in prefixes for t in tiles], np.uint8)


def facade_dets(spec: dict, dets: list[dict], depth: np.ndarray) -> list[dict]:
    """Detector boxes (pixels) -> cleaned boxes in facade metres."""
    p = spec['plane']
    H, W = depth.shape
    rows = []
    for d in dets:
        lab = d['class'].lower()
        kind = 'garage_door' if 'garage' in lab else 'balcony' if 'balcony' in lab else 'door' if 'door' in lab \
            else 'window' if 'window' in lab else None
        if kind is None:
            continue
        x0, y0, x1, y1 = d['box']
        if (x1 - x0) * (y1 - y0) > 0.25 * W * H:
            continue
        sub = depth[max(0, int(y0)):int(math.ceil(y1)), max(0, int(x0)):int(math.ceil(x1))]
        if not sub.size or np.isfinite(sub).mean() < 0.5:
            continue
        u0, z1 = facade.to_m(p, x0, y0)
        u1, z0 = facade.to_m(p, x1, y1)
        if u1 < -0.2 or u0 > spec['length_m'] + 0.2:
            continue
        rows.append({'kind': kind, 'score': float(d['score']), 'u0': u0, 'u1': u1, 'z0': z0, 'z1': z1, 'px': [x0, y0, x1, y1]})
    # a box holding two or more same-kind boxes is a group (twin window, whole storey): keep the parts
    keep = []
    for i, a in enumerate(rows):
        inner = sum(1 for j, b in enumerate(rows) if j != i and b['kind'] == a['kind'] and _inside_share(b, a) > 0.8)
        if inner < 2:
            keep.append(a)
    keep.sort(key=lambda r: -r['score'])
    out = []
    for a in keep:
        if any(b['kind'] == a['kind'] and (_iou1(a, b) > 0.45 or _inside_share(a, b) > 0.8) for b in out):
            continue
        out.append(a)
    for r in out:
        r.pop('px')
        for k in ('u0', 'u1', 'z0', 'z1', 'score'):
            r[k] = round(r[k], 2)
    return out


def _inter(a, b) -> float:
    return max(0.0, min(a['u1'], b['u1']) - max(a['u0'], b['u0'])) * max(0.0, min(a['z1'], b['z1']) - max(a['z0'], b['z0']))


def _area(a) -> float:
    return max(1e-9, (a['u1'] - a['u0']) * (a['z1'] - a['z0']))


def _iou1(a, b) -> float:
    i = _inter(a, b)
    return i / (_area(a) + _area(b) - i)


def _inside_share(a, b) -> float:
    """Share of a inside b."""
    return _inter(a, b) / _area(a)


# ---------------------------------------------------------------------------- the stage


def worker_tag() -> str:
    return hashlib.sha1(WORKER.read_bytes()).hexdigest()[:12]


def device_env() -> str:
    return os.environ.get('UNCRASHED_VISION_DEVICE', 'auto')


def allow_cpu() -> bool:
    return os.environ.get('UNCRASHED_VISION_CPU') == '1'


def worker_env() -> dict:
    """Environment of the worker. Once the weights are on disk the libraries are kept offline: ultralytics would
    otherwise send a usage event per run (settings sync) and the Hugging Face hub would check for updates."""
    env = dict(os.environ, YOLO_CONFIG_DIR=str(MODELS / 'ultralytics'), YOLO_VERBOSE='False',
               TRANSFORMERS_VERBOSITY='error', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               HF_HUB_DISABLE_PROGRESS_BARS='1', TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_TELEMETRY='1')
    if _model_present(OBB_MODEL) and _model_present(SAM_MODEL):
        env['YOLO_OFFLINE'] = '1'
    if _model_present(DINO_MODEL):
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env


def _worker(job: dict, work: Path) -> dict:
    jp, op = work / 'job.json', work / 'out.json'
    text = json.dumps(job)
    if op.is_file() and jp.is_file() and jp.read_text() == text and os.environ.get('UNCRASHED_VISION_REUSE') != '0':
        try:  # same tiles / facades as the last run in RAM: its detections still hold
            out = json.loads(op.read_text())
            if not out.get('errors'):
                out['reused'] = True
                return out
        except (OSError, json.JSONDecodeError):
            pass
    jp.write_text(text)
    op.unlink(missing_ok=True)
    env = worker_env()
    # -P: the worker's folder (scan/, which holds profile.py) must not shadow stdlib modules torch imports
    cmd = [str(VENV / 'bin' / 'python'), '-P', str(WORKER), str(jp), str(op)]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, env=env, cwd=str(work))
    except subprocess.TimeoutExpired:
        raise VisionError(f'vision worker exceeded {TIMEOUT_S:.0f} s (UNCRASHED_VISION_TIMEOUT)') from None
    if done.returncode or not op.is_file():
        tail = ' '.join(done.stderr.strip().splitlines()[-3:])[-400:]
        raise VisionError(f'vision worker failed (exit {done.returncode}): {tail}')
    return json.loads(op.read_text())


def tone(img, depth) -> list[int] | None:
    """Median colour of the wall pixels of a facade render (cells where the mesh was hit), or None."""
    try:
        im = np.asarray(img)[..., :3]
        m = np.isfinite(np.asarray(depth))
        if im.shape[:2] != m.shape or m.sum() < 50:
            return None
        return [int(v) for v in np.median(im[m], 0)]
    except (TypeError, ValueError, IndexError):
        return None


def _facade_record(specs: list[dict], look: list[int], depths: list, dets_all: list, party: list[dict]):
    """({building id: [wall rows]}, windows) from the rendered walls, their detections and the party walls."""
    if len(dets_all) != len(look):
        raise VisionError(f'facade detections for {len(dets_all)} images, expected {len(look)}')
    by_spec = dict(zip(look, dets_all))
    n_win = 0
    record: dict[str, list] = {}
    for i, s in enumerate(specs):
        row = {k2: s[k2] for k2 in ('side', 'wall_from', 'a', 'b', 'length_m', 'faces_deg', 'ground_up', 'plane',
                                    'size', 'coverage', 'front_blocked')}
        if s.get('tone'):
            row['tone'] = s['tone']
        if i in by_spec:
            dets = facade_dets(s, by_spec[i], depths[i])
            row['detections'] = dets
            row['grid'] = facade.fit(dets, s['ground_up'], s.get('eave_m'), s['length_m'])
            row['status'] = 'surveyed'
            n_win += row['grid']['windows']
        else:
            row['detections'] = []
            row['grid'] = None
            row['status'] = 'no_view'
        record.setdefault(str(s['building']), []).append(row)
    for s in party:   # another building stands in front: listed, not rendered
        row = {k2: s[k2] for k2 in ('side', 'wall_from', 'a', 'b', 'length_m', 'faces_deg', 'ground_up', 'plane',
                                    'size', 'front_blocked')}
        row.update(coverage=None, detections=[], grid=None, status='party_wall')
        record.setdefault(str(s['building']), []).append(row)
    for walls in record.values():
        walls.sort(key=lambda w: w['side'])
    return record, n_win


def facades_on_demand(meta: dict, objects: list[dict], ras, ids: set[int], osm_rings: dict,
                      shift=(0.0, 0.0)) -> dict:
    """Facade survey of a few buildings the survey skipped (small, low, over the cap): {building id: [walls]}
    (an empty list when no wall can be rendered). Same rendering, detector and fit as the survey; its own RAM
    work folder, so the survey's cached detections stay. Raises VisionError when the worker fails."""
    work = store.ram(meta['key']) / 'vision-facades'
    work.mkdir(parents=True, exist_ok=True)
    specs = plan_facades(meta, objects, ras, osm_rings, shift, only=set(ids))
    party = [s for s in specs if s['front_blocked'] >= 0.5]
    specs = [s for s in specs if s['front_blocked'] < 0.5][:MAX_FACADES]
    look, depths = [], []
    res: dict = {}
    if specs:
        imgs, depths, _st = render_facades(meta, specs, work / 'facades')
        for s, im, d in zip(specs, imgs, depths):
            s['coverage'] = round(float(np.isfinite(d).mean()), 2)
            s['tone'] = tone(im, d)
        look = [i for i, s in enumerate(specs) if s['coverage'] >= 0.2]
        job = {'worker': worker_tag(), 'models_dir': str(MODELS), 'device': device_env(), 'allow_cpu': allow_cpu(),
               'dino_model': DINO_MODEL,
               'facades': {'images': [specs[i]['image'] for i in look], 'prompts': FACADE_PROMPTS, 'thr': 0.25}}
        try:
            res = _worker(job, work) if look else {}
        finally:
            shutil.rmtree(work / 'facades', ignore_errors=True)
        if res.get('errors'):
            raise VisionError('; '.join(res['errors']))
    record, _ = _facade_record(specs, look, depths, res.get('facades') or [], party)
    return {str(i): record.get(str(i), []) for i in ids}


def run(meta: dict, geometry: list[dict], ras, *, aerial: bool = True, facades: bool = True,
        osm_rings: dict | None = None, shift=(0.0, 0.0)) -> tuple[list[dict], dict, dict]:
    """(fused objects, facades record {building id: [facade]}, report). Raises VisionError when the worker fails."""
    t_all = time.time()
    k = meta['key']
    work = store.ram(k) / 'vision'
    work.mkdir(parents=True, exist_ok=True)
    rep: dict = {'ran': True, 'timings': {}}
    job: dict = {'worker': worker_tag(), 'models_dir': str(MODELS), 'device': device_env(),
                 'allow_cpu': allow_cpu(), 'dino_model': DINO_MODEL}
    fine = coarse = []
    t0 = time.time()
    if aerial:
        fine, coarse = _tiles(ras.ortho, work / 'tiles')
        job['aerial'] = {'tiles': [t['path'] for t in fine], 'coarse': [t['path'] for t in coarse],
                         'obb': {'model': OBB_MODEL, 'imgsz': 1024, 'conf': 0.15},
                         'dino': {'prompts': AERIAL_PROMPTS, 'thr': 0.25},
                         'sam': {'model': SAM_MODEL, 'classes': ['car', 'van', 'truck', 'bus', 'swimming pool'],
                                 'thr': 0.3}}
    specs, depths, look, party = [], [], [], []
    if facades:
        specs = plan_facades(meta, geometry, ras, osm_rings or {}, shift)
        party = [s for s in specs if s['front_blocked'] >= 0.5]
        specs = [s for s in specs if s['front_blocked'] < 0.5][:MAX_FACADES]
        imgs, depths, st = render_facades(meta, specs, work / 'facades')
        cov = [float(np.isfinite(d).mean()) for d in depths]
        for s, c, im, d in zip(specs, cov, imgs, depths):
            s['coverage'] = round(c, 2)
            s['tone'] = tone(im, d)
        look = [i for i, c in enumerate(cov) if c >= 0.2]
        job['facades'] = {'images': [specs[i]['image'] for i in look], 'prompts': FACADE_PROMPTS, 'thr': 0.25}
        rep['facade_render'] = st
    rep['timings']['prepare_s'] = round(time.time() - t0, 2)
    t0 = time.time()
    try:
        res = _worker(job, work)
    finally:   # tiles and facade renders are only the worker's input: free the RAM (job/out.json stay for reuse)
        for sub in ('tiles', 'facades'):
            shutil.rmtree(work / sub, ignore_errors=True)
    rep['timings']['worker_s'] = round(time.time() - t0, 2)
    rep['worker'] = {'reused_detections': bool(res.get('reused')), 'device': res.get('device'), 'gpu': res.get('gpu'), 'timings': res.get('timings'),
                     'vram_peak_gb': res.get('vram_peak_gb'), 'models': res.get('models')}
    if res.get('errors'):
        raise VisionError('; '.join(res['errors']))
    t0 = time.time()
    objs = geometry
    if aerial:
        g = meta['grid']
        fr = _Ras(meta, ras.dsm, ras.dtm, ras.cls, ras.objid, ras.rgb)
        cands = candidates(meta, fine, coarse, res)
        objs, frep = fuse(geometry, cands, fr)
        rep['aerial'] = {'tiles': len(fine), 'coarse_tiles': len(coarse), 'detections': len(cands), **frep}
        del g
    record: dict[str, list] = {}
    if facades:
        record, n_win = _facade_record(specs, look, depths, res.get('facades') or [], party)
        rep['facade_survey'] = {'walls': len(specs), 'surveyed': len(look), 'buildings': len(record), 'windows': n_win,
                          'party_walls': len(party)}
    rep['timings']['fuse_s'] = round(time.time() - t0, 2)
    rep['timings']['total_s'] = round(time.time() - t_all, 2)
    return objs, record, rep
