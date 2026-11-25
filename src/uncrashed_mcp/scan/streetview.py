"""Street views rendered from the scan: virtual cameras along the OSM streets and paths, detectors on the renders,
detections lifted into 3D by the render's depth buffer (the first mesh surface along each pixel's ray), fused across
views into objects with 3D positions (signs, street lamps, benches, bins, bollards, bicycles, fences, doors;
windows are counted per building).

Cameras: every STEP_M along each street / path line (deduplicated), eye 1.6 m above the ground model (pitch 0) and
DRONE_M above it (pitch -25 deg), looking to both sides of the line (90 deg field of view, 640 x 480). An eye under a
roof or crown (surface more than 0.6 m above the ground there) moves sideways up to 3 m or is dropped. Renders live in
RAM (/dev/shm) only while the detector runs; depth is rendered again for the views with detections.

Lifting: the depth of a detection is the 25th percentile of the depth inside the box's central third (thin objects
in front of a wall), the 3D point is the ray through the box centre at that depth, height from the rays through the
box top and bottom at the same depth. Kept: within 30 m, not cut by the image edge, a height in the class range,
ground classes standing within 1.2 m of the ground model, doors within 2 m of a building outline, street furniture
outside building footprints. Fusion: per class, points of different camera positions within a class radius form
one object; an object needs two camera positions, or one detection scoring at least 0.5 (then conf <= 0.45).
A geometry small object / shrub / low structure at the spot takes the class instead of a new object.
"""
from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

import numpy as np

from . import grid, native, store, vision

STEP_M = 10.0
DRONE_M = 8.0
W, H, FOV = 640, 480, 90.0
FAR_M = 60.0
MAX_VIEWS = 1600
MAX_DEPTH_M = 30.0
# two prompt groups: facade openings dominate a joint prompt (every box becomes a window)
FACADE_PROMPTS = ['door', 'window']
FURNITURE_PROMPTS = ['traffic sign', 'street lamp', 'bench', 'trash can', 'bollard', 'bicycle', 'fence']
MIN_VIEW_DEPTH_M = 4.0      # a view whose median depth is closer looks into a wall or a crown: not sent
# label keyword -> class, in priority order (Grounding DINO may return joined phrases)
LABELS = (('traffic sign', 'sign'), ('sign', 'sign'), ('street lamp', 'street_lamp'), ('lamp', 'street_lamp'),
          ('bench', 'bench'), ('trash', 'bin'), ('bin', 'bin'), ('bollard', 'bollard'), ('bicycle', 'bicycle'),
          ('fence', 'fence'), ('door', 'door'), ('window', 'window'))
# class: (min height, max height, fusion radius m, stands on the ground)
CLASS = {'sign': (1.2, 5.0, 1.5, True), 'street_lamp': (2.5, 14.0, 2.0, True), 'bench': (0.35, 1.6, 1.5, True),
         'bin': (0.4, 1.9, 1.0, True), 'bollard': (0.3, 1.6, 0.8, True), 'bicycle': (0.5, 1.7, 1.5, True),
         'fence': (0.4, 3.0, 3.0, True), 'door': (1.5, 3.8, 1.0, False), 'window': (0.4, 3.0, 0.8, False)}
STREET_CLASSES = tuple(c for c in CLASS if c != 'window')


# ---------------------------------------------------------------------------- cameras


def _lines(features: dict, shift) -> list[tuple[np.ndarray, str]]:
    out = []
    for key in ('roads', 'paths'):
        for f in features.get(key, []):
            if f.get('tunnel') or len(f.get('line') or []) < 2:
                continue
            out.append((np.asarray(f['line'], float) + np.asarray(shift, float), key))
    return out


def plan(meta: dict, ras, features: dict, shift=(0.0, 0.0), step_m: float = STEP_M, max_views: int = MAX_VIEWS
         ) -> list[dict]:
    """Camera list: {'eye': [e, s, u], 'yaw', 'pitch', 'pos' (camera position index), 'kind' street | drone}."""
    g = ras.g
    x1, y1 = g.x0 + g.w * g.res, g.y0 + g.h * g.res
    pts = []
    for line, _kind in _lines(features, shift):
        seg = np.diff(line, axis=0)
        L = np.hypot(seg[:, 0], seg[:, 1])
        total = float(L.sum())
        if total < 1:
            continue
        for t in np.arange(step_m / 2, total, step_m):
            k = int(np.searchsorted(np.cumsum(L), t))
            k = min(k, len(seg) - 1)
            t0 = t - (np.cumsum(L)[k] - L[k])
            d = seg[k] / max(L[k], 1e-9)
            p = line[k] + d * t0
            if g.x0 + 2 < p[0] < x1 - 2 and g.y0 + 2 < p[1] < y1 - 2:
                pts.append((p, d))
    kept: list[tuple[np.ndarray, np.ndarray]] = []
    for p, d in pts:   # one camera position per 0.6 * step
        if all(np.hypot(*(p - q)) >= 0.6 * step_m for q, _ in kept):
            kept.append((p, d))
    cams = []
    for p, d in kept:
        n = np.array([-d[1], d[0]])
        eye = None
        for off in (0.0, 1.5, -1.5, 3.0, -3.0):
            q = p + off * n
            r, c, inside = ras.cell(q[0], q[1])
            if not inside or not np.isfinite(ras.dsm[r, c]):
                continue
            gz = float(ras.dtm[r, c])
            if float(ras.dsm[r, c]) - gz <= 0.6:
                eye = (q, gz)
                break
        if eye is None:
            continue
        q, gz = eye
        pos = len({tuple(c['eye'][:2]) for c in cams})
        side = math.degrees(math.atan2(n[1], n[0]))
        for yaw in (side, side + 180):
            cams.append({'eye': [float(q[0]), float(q[1]), gz + 1.6], 'yaw': yaw % 360, 'pitch': 0.0, 'pos': pos,
                         'kind': 'street'})
            cams.append({'eye': [float(q[0]), float(q[1]), gz + DRONE_M], 'yaw': yaw % 360, 'pitch': -25.0, 'pos': pos,
                         'kind': 'drone'})
    if len(cams) > max_views:
        keep_pos = sorted({c['pos'] for c in cams})
        every = math.ceil(len(cams) / max_views)
        chosen = set(keep_pos[::every])
        cams = [c for c in cams if c['pos'] in chosen]
    return cams


def _cam_array(cams: list[dict]) -> np.ndarray:
    return np.array([[*c['eye'], c['yaw'], c['pitch'], FOV, 0.3, FAR_M] for c in cams], float).reshape(-1, 8)


def render(meta: dict, cams: list[dict], rgb: bool = True, batch: int = 200, out_dir: Path | None = None):
    """Renders in batches; with out_dir writes PNGs (paths into cams[i]['image']). Returns (depths or None, stats)."""
    from .. import reference
    cache = reference.ensure_index(Path(meta['obj']))
    amap = reference._axis_map(meta['axes'])
    skip = vision._coarse_meshes(cache)
    stats = {'triangles': 0, 'views': 0, 'render_s': 0.0}
    depths = [] if not rgb else None
    for i in range(0, len(cams), batch):
        part = cams[i:i + batch]
        img, dep, st = native.views(cache, amap, _cam_array(part), W, H, rgb=rgb, skip=skip)
        stats['triangles'] += st['triangles']
        stats['views'] += st['views']
        stats['render_s'] = round(stats['render_s'] + st['render_s'] + st['decode_s'], 2)
        if rgb and out_dir is not None:
            from PIL import Image
            out_dir.mkdir(parents=True, exist_ok=True)
            for k, (c, im) in enumerate(zip(part, img)):
                c['image'] = str(out_dir / f'v{i + k:05d}.jpg')
                Image.fromarray(im).save(c['image'], quality=92)
                c['coverage'] = round(float(np.isfinite(dep[k]).mean()), 3)
                fin = dep[k][np.isfinite(dep[k])]
                c['median_depth_m'] = round(float(np.median(fin)), 2) if fin.size else None
        if depths is not None:
            depths.extend(dep)
    return depths, stats


# ---------------------------------------------------------------------------- lifting


def label_class(label: str) -> str | None:
    lab = label.lower()
    for key, cls in LABELS:
        if key in lab:
            return cls
    return None


def basis(cam: dict):
    y, p = math.radians(cam['yaw']), math.radians(cam['pitch'])
    f = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    r = np.array([-math.sin(y), math.cos(y), 0.0])
    u = np.array([-math.sin(p) * math.cos(y), -math.sin(p) * math.sin(y), math.cos(p)])
    return f, r, u


def ray_point(cam: dict, px: float, py: float, depth: float) -> np.ndarray:
    """World point (east, south, up) on the ray through pixel (px, py) (pixel centres at +0.5) at view depth."""
    f, r, u = basis(cam)
    fx = (W / 2) / math.tan(math.radians(FOV) / 2)
    xn, yn = (px - W / 2) / fx, -(py - H / 2) / fx
    return np.asarray(cam['eye'], float) + depth * (f + xn * r + yn * u)


def lift(cam: dict, det: dict, depth: np.ndarray, ground, inside_building) -> dict | None:
    """One detection -> 3D record or None (with the reason in det['drop'])."""
    cls = label_class(det['class'])
    if cls is None:
        return None
    x0, y0, x1, y1 = det['box']
    if x0 <= 1 or y0 <= 1 or x1 >= W - 2 or (y1 >= H - 2 and cls not in ('fence',)):
        det['drop'] = 'cut by the image edge'
        return None
    if (x1 - x0) * (y1 - y0) > 0.4 * W * H:
        det['drop'] = 'box covers most of the view'
        return None
    cw = (x1 - x0) / 3
    sub = depth[int(y0):int(math.ceil(y1)), int(x0 + cw):int(math.ceil(x1 - cw))]
    vals = sub[np.isfinite(sub)]
    if vals.size < 4:
        det['drop'] = 'no surface behind the box'
        return None
    z = float(np.percentile(vals, 25))
    if z > MAX_DEPTH_M:
        det['drop'] = 'farther than 30 m'
        return None
    cx = (x0 + x1) / 2
    p = ray_point(cam, cx, (y0 + y1) / 2, z)
    top, bot = ray_point(cam, cx, y0, z), ray_point(cam, cx, y1, z)
    gz = float(ground(p[0], p[1]))
    lo, hi, _r, on_ground = CLASS[cls]
    height = float(top[2] - (gz if on_ground else bot[2]))
    if on_ground and abs(float(bot[2]) - gz) > 1.2:
        det['drop'] = 'not standing on the ground'
        return None
    if not lo <= height <= hi:
        det['drop'] = f'height {height:.1f} m outside {lo}..{hi}'
        return None
    if cls in STREET_CLASSES and cls != 'door' and inside_building(p[0], p[1]):
        det['drop'] = 'inside a building footprint'
        return None
    fx = (W / 2) / math.tan(math.radians(FOV) / 2)
    width = (x1 - x0) / fx * z
    return {'cls': cls, 'score': float(det['score']), 'x': float(p[0]), 'y': float(p[1]), 'ground_z': gz,
            'top_z': float(top[2]), 'height_m': height, 'width_m': float(width), 'depth_m': z, 'pos': cam['pos'],
            'yaw_view': cam['yaw'], 'eye': cam['eye']}


def cluster(points: list[dict]) -> list[dict]:
    """Greedy multi-view fusion per class (highest score first)."""
    out = []
    for cls in sorted({p['cls'] for p in points}):
        rad = CLASS[cls][2]
        pts = sorted((p for p in points if p['cls'] == cls), key=lambda p: -p['score'])
        used = [False] * len(pts)
        for i, p in enumerate(pts):
            if used[i]:
                continue
            members = [p]
            used[i] = True
            for j in range(i + 1, len(pts)):
                if not used[j] and math.hypot(pts[j]['x'] - p['x'], pts[j]['y'] - p['y']) <= rad:
                    members.append(pts[j])
                    used[j] = True
            views = len({m['pos'] for m in members})
            best = max(m['score'] for m in members)
            if views < 2 and best < 0.5:
                continue
            mean = float(np.mean([m['score'] for m in members]))
            conf = min(0.9, 0.35 + 0.15 * (views - 1) + 0.3 * mean) if views >= 2 else min(0.45, 0.2 + 0.4 * best)
            xs = np.array([m['x'] for m in members])
            ys = np.array([m['y'] for m in members])
            out.append({'cls': cls, 'conf': round(conf, 2), 'views': views, 'detections': len(members),
                        'score': round(best, 2), 'x': float(np.median(xs)), 'y': float(np.median(ys)),
                        'spread_m': round(float(np.median(np.hypot(xs - np.median(xs), ys - np.median(ys)))), 2),
                        'ground_z': float(np.median([m['ground_z'] for m in members])),
                        'height_m': float(np.median([m['height_m'] for m in members])),
                        'width_m': float(np.median([m['width_m'] for m in members])),
                        'yaw_view': float(members[0]['yaw_view']), 'eye': members[0]['eye']})
    return out


# ---------------------------------------------------------------------------- objects


def _building_mask(objects: list[dict], g) -> np.ndarray:
    lab = np.zeros((g.h, g.w), np.int32)
    for o in objects:
        if o['cls'] == 'building' and len(o.get('footprint') or []) >= 3:
            fp = np.asarray(o['footprint'], float)
            c, r = g.to_cell(fp[:, 0], fp[:, 1])
            native.fill_poly(lab, c, r, o['id'] + 1)
    return lab


def to_objects(clusters: list[dict], objects: list[dict], g, ras) -> tuple[list[dict], dict]:
    """Add fused street objects (or relabel a geometry object at the spot). Returns (objects, counts)."""
    lab = _building_mask(objects, g)
    edge = grid.dilate(lab > 0, max(3, int(round(2.0 / g.res)) | 1)) & ~grid.erode(lab > 0, max(3, int(round(2.0 / g.res)) | 1))
    near_lab = native.nearest(lab > 0)
    added, relabelled, windows = {}, {}, {}
    small = [o for o in objects if o['cls'] in ('small_object', 'low_structure', 'shrub')]
    for cl in clusters:
        r, c, inside = ras.cell(cl['x'], cl['y'])
        if not inside:
            continue
        if cl['cls'] in ('window', 'door'):
            if not edge[r, c]:
                continue
            idx = near_lab[r, c]
            bid = int(lab.ravel()[idx]) - 1 if idx >= 0 else -1
            if cl['cls'] == 'window':
                if bid >= 0:
                    windows[bid] = windows.get(bid, 0) + 1
                continue
        else:
            bid = None
        hit = None
        if cl['cls'] in STREET_CLASSES and cl['cls'] != 'door':
            for o in small:
                if math.hypot(o['x'] - cl['x'], o['y'] - cl['y']) <= max(1.0, CLASS[cl['cls']][2]) and not o.get('street'):
                    hit = o
                    break
        model = f"gdino-base:{cl['score']:.2f}"
        ev = {'src': 'street_views', 'verdict': 'agree',
              'detail': f"{cl['views']} camera positions, {cl['detections']} detections, spread {cl['spread_m']} m"}
        if hit is not None:
            hit.update(street={'cls': cl['cls'], 'views': cl['views']}, was=hit['cls'], cls=cl['cls'],
                       conf=round(max(hit['conf'], cl['conf']), 2), src=hit['src'] + '+street')
            hit.setdefault('models', []).append(model)
            hit.setdefault('evidence', []).append(ev)
            relabelled[cl['cls']] = relabelled.get(cl['cls'], 0) + 1
            continue
        yaw = (cl['yaw_view'] + 90) % 180
        w = max(0.3, cl['width_m'])
        o = {'id': len(objects), 'cls': cl['cls'], 'conf': cl['conf'], 'src': 'street', 'x': round(cl['x'], 2),
             'y': round(cl['y'], 2), 'ground_z': round(cl['ground_z'], 2),
             'top_z': round(cl['ground_z'] + cl['height_m'], 2), 'height_m': round(cl['height_m'], 2),
             'length_m': round(w, 2), 'width_m': round(min(w, 0.5), 2), 'yaw_deg': round(yaw, 1),
             'area_m2': round(w * min(w, 0.5), 2), 'rgb': [int(v) for v in ras.rgb[r, c]], 'models': [model],
             'views': cl['views'], 'spread_m': cl['spread_m'], 'evidence': [ev]}
        if bid is not None and bid >= 0:
            o['building'] = bid
            e = np.asarray(cl['eye'], float)
            o['faces_deg'] = round((math.degrees(math.atan2(e[1] - cl['y'], e[0] - cl['x'])) + 360) % 360, 1)
        objects.append(o)
        added[cl['cls']] = added.get(cl['cls'], 0) + 1
    for bid, n in windows.items():
        if 0 <= bid < len(objects):
            objects[bid]['street_windows'] = n
    return objects, {'added': added, 'relabelled': relabelled, 'buildings_with_street_windows': len(windows)}


def run(meta: dict, objects: list[dict], ras, features: dict, shift=(0.0, 0.0)) -> tuple[list[dict], dict]:
    """The stage. Raises vision.VisionError when the worker fails."""
    t_all = time.time()
    k = meta['key']
    work = store.ram(k) / 'street'
    work.mkdir(parents=True, exist_ok=True)
    rep: dict = {'ran': True, 'timings': {}}
    t0 = time.time()
    cams = plan(meta, ras, features, shift)
    rep['cameras'] = len(cams)
    rep['camera_positions'] = len({c['pos'] for c in cams})
    rep['timings']['plan_s'] = round(time.time() - t0, 2)
    if not cams:
        rep['ran'] = False
        rep['reason'] = 'no street or path point with open sky inside the survey'
        return objects, rep
    t0 = time.time()
    _d, st = render(meta, cams, rgb=True, out_dir=work / 'views')
    rep['render'] = st
    rep['timings']['render_s'] = round(time.time() - t0, 2)
    look = [i for i, c in enumerate(cams) if c.get('coverage', 0) >= 0.3
            and (c.get('median_depth_m') or 0) >= MIN_VIEW_DEPTH_M]
    rep['views_sent'] = len(look)
    images = [cams[i]['image'] for i in look]
    job = {'worker': vision.worker_tag(), 'models_dir': str(vision.MODELS),
           'device': vision.device_env(), 'allow_cpu': vision.allow_cpu(), 'dino_model': vision.DINO_MODEL,
           'street': {'groups': [{'images': images, 'prompts': FACADE_PROMPTS, 'thr': 0.25},
                                 {'images': images, 'prompts': FURNITURE_PROMPTS, 'thr': 0.22}]}}
    t0 = time.time()
    try:
        res = vision._worker(job, work)
    finally:
        shutil.rmtree(work / 'views', ignore_errors=True)
    rep['timings']['worker_s'] = round(time.time() - t0, 2)
    rep['worker'] = {'reused_detections': bool(res.get('reused')), 'gpu': res.get('gpu'), 'timings': res.get('timings')}
    if res.get('errors'):
        raise vision.VisionError('; '.join(res['errors']))
    groups = res.get('street') or []
    if len(groups) != 2 or any(len(gr) != len(look) for gr in groups):
        raise vision.VisionError(f'street detections for {[len(gr) for gr in groups]} images, expected 2 x {len(look)}')
    dets = [a + b for a, b in zip(*groups)]
    t0 = time.time()
    with_dets = [(i, d) for i, d in zip(look, dets) if any(label_class(x['class']) for x in d)]
    depths, _st = render(meta, [cams[i] for i, _ in with_dets], rgb=False) if with_dets else ([], {})
    g = ras.g
    lab = _building_mask(objects, g)
    inner = grid.erode(lab > 0, max(3, int(round(1.0 / g.res)) | 1))

    def inside_building(x, y):
        r, c, ok = ras.cell(x, y)
        return bool(ok and inner[r, c])

    def ground(x, y):
        return float(ras.ground(np.array([x]), np.array([y]))[0])

    points, drops, raw = [], {}, 0
    for (i, d), dep in zip(with_dets, depths):
        for det in d:
            if label_class(det['class']) is None:
                continue
            raw += 1
            p = lift(cams[i], det, dep, ground, inside_building)
            if p is None:
                why = det.get('drop', 'unlifted')
                drops[why.split(' ')[0] if why.startswith('height') else why] = drops.get(
                    why.split(' ')[0] if why.startswith('height') else why, 0) + 1
                continue
            points.append(p)
    rep['timings']['lift_s'] = round(time.time() - t0, 2)
    clusters = cluster(points)
    by_cls: dict[str, list[int]] = {}
    for p in points:
        by_cls.setdefault(p['cls'], [0, 0])[0] += 1
    for c in clusters:
        by_cls.setdefault(c['cls'], [0, 0])[1] += 1
    objects, counts = to_objects(clusters, objects, g, ras)
    rep.update(detections=raw, lifted=len(points), dropped=drops, fused={k2: v[1] for k2, v in by_cls.items()},
               lifted_by_class={k2: v[0] for k2, v in by_cls.items()}, **counts)
    rep['timings']['total_s'] = round(time.time() - t_all, 2)
    return objects, rep
