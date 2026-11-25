"""Build vs scan, pixel for pixel: capture_topdown (a straight-down game image on scan_view's pixel grid) and
scan_view --compare (scan | build | edge overlay, with the deviations of the last scan_compare labelled).

The grid is view.crop_frame: a box x0,y0,x1,y1 and px give metres per pixel and the image size; pixel (i, j) covers
x0 + i * mpp east and y0 + j * mpp south. Both images are in map metres (placement), so a placed object and the scan
object it stands for sit on the same pixels when the build is right. Every image stays in RAM (/dev/shm).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import ops, store
from .store import Frame

TOPDOWN_DIR = 'uncrashed-topdown'


def _box(box, center, size_m: float) -> tuple[float, float, float, float]:
    if box is not None and center is not None:
        raise ValueError('give box or center (+ size_m), not both')
    if box is not None:
        b = ops._box4(box)
    elif center is not None:
        c = ops._xy(center, 'center')[0]
        if not 2 <= size_m <= 600:
            raise ValueError('size_m must be 2..600')
        b = (c[0] - size_m / 2, c[1] - size_m / 2, c[0] + size_m / 2, c[1] + size_m / 2)
    else:
        raise ValueError('give box x0,y0,x1,y1 or center x,y (+ size_m), map metres')
    if b[2] - b[0] < 1 or b[3] - b[1] < 1 or max(b[2] - b[0], b[3] - b[1]) > 600:
        raise ValueError('box sides must be 1..600 m')
    return b


def _scan_heights(survey_key: str, b) -> tuple[dict | None, float | None, float | None]:
    """(survey meta, median ground z, highest top z) of the scan in box b, map frame; Nones without a placed
    survey."""
    try:
        meta = store.resolve(survey_key)
    except ValueError:
        return None, None, None
    if not store.placement(meta['obj']):
        return meta, None, None
    fr = Frame(meta, 'map')
    ras = ops.Rasters(meta)
    xs = np.linspace(b[0], b[2], 40)
    ys = np.linspace(b[1], b[3], 40)
    X, Y = np.meshgrid(xs, ys)
    e, s = fr.to_scan(X.ravel(), Y.ravel())
    r, c, inside = ras.cell(e, s)
    if not inside.any():
        return meta, None, None
    gnd = np.asarray(ras.ground(e[inside], s[inside]), float)
    top = np.asarray(ras.dsm[r[inside], c[inside]], float)
    top = top[np.isfinite(top)]
    return meta, fr.z_out(float(np.median(gnd))), fr.z_out(float(top.max())) if len(top) else None


def out_dir(meta: dict | None) -> Path:
    if meta is not None:
        return store.views_dir(meta['key'])
    d = store.SHM.parent / TOPDOWN_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def capture_topdown(box=None, center=None, size_m: float = 60.0, px: int = 900, reference: str = 'hide',
                    projection: str = 'auto', ground_z: float | None = None, top_z: float | None = None,
                    survey_key: str = '') -> dict:
    from .. import live
    b = _box(box, center, size_m)
    px = max(200, min(int(px), 1600))
    meta, g, t = _scan_heights(survey_key, b)
    ground = ground_z if ground_z is not None else (g if g is not None else 0.0)
    top = top_z if top_z is not None else t
    res = live.capture_topdown(b, px, ground_z=ground, top_z=top, out_dir=out_dir(meta), reference_mode=reference,
                               projection=projection)
    res.update(px=px, survey=meta['key'] if meta else None,
               ground_z=round(ground, 2), top_z=round(top, 2) if top is not None else None,
               next=f"scan_view --compare {res['image']} (scan | build | overlay with the last scan_compare "
                    'deviations)',
               note='image in RAM (/dev/shm) only' + (' (it shows the textured scan: keep it on this machine)'
                                                      if reference != 'hide' else ''))
    Path(res['image']).with_suffix('.json').write_text(json.dumps(res))
    return res


# ---------------------------------------------------------------------------- compare sheet


def _edges(img: np.ndarray) -> np.ndarray:
    """Normalised gradient magnitude of the luminance (0..1)."""
    y = img[..., :3].astype(float) @ np.array([.299, .587, .114])
    gx = np.zeros_like(y)
    gy = np.zeros_like(y)
    gx[:, 1:-1] = y[:, 2:] - y[:, :-2]
    gy[1:-1] = y[2:] - y[:-2]
    m = np.hypot(gx, gy)
    q = np.percentile(m, 92) if m.any() else 1.0
    return np.clip(m / max(q, 1e-6), 0, 1)


def overlay(scan_rgb: np.ndarray, build_rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Edges of the scan (magenta) and of the build (green) over a dim grey blend; white where both agree. Also the
    share of strong scan edges that the build repeats within one pixel (a rough alignment score)."""
    es, eb = _edges(scan_rgb), _edges(build_rgb)
    grey = (scan_rgb[..., :3].astype(float).mean(2) * .5 + build_rgb[..., :3].astype(float).mean(2) * .5) * .35
    out = np.repeat(grey[..., None], 3, 2)
    s, bld = es > .5, eb > .5
    near_b = bld.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        near_b |= np.roll(np.roll(bld, dy, 0), dx, 1)
    both = s & near_b
    out[s & ~both] = (230, 40, 230)
    out[bld & ~s] = (40, 230, 60)
    out[both] = (255, 255, 255)
    agree = float(both.sum() / max(1, s.sum()))
    return out.clip(0, 255).astype(np.uint8), agree


DEVIATIONS = 25          # labels on a compare sheet: the worst deviations only
STALE_S = 600            # a compare report older than this is called stale
GROUND_ROLES = ('surface', 'decal')


def _label(r: dict) -> str:
    issues = r.get('issues') or ''
    bits = [f"{r['id']} {r['status']}"]
    if r.get('raster') and r['status'] != 'missing':
        bits.append(f"iou{r['raster']['iou']:g}")
    if r.get('offset_m') is not None and 'offset' in issues:
        bits.append(f"{r['offset_m']:g}m")
    if 'yaw_off' in issues:
        bits.append(f"{r['yaw_off_deg']:g}deg")
    if 'size_off' in issues and r.get('size_ratio'):
        bits.append('x' + '/'.join(f'{v:g}' for v in r['size_ratio']))
    if 'height_off' in issues and r.get('height_off_m') is not None:
        bits.append(f"h{r['height_off_m']:+g}")
    return ' '.join(bits)


def _is_ground_extra(e: dict) -> bool:
    """An unexplained placed object that is ground cover: reports made before scan_compare counted ground pieces
    apart listed tiles and markings as extras."""
    from .. import fit
    from ..catalog import load_catalog
    a = load_catalog().assets.get(e.get('row'))
    if a is None:
        return False
    role = fit.role_of(a)
    base = e.get('base_above_scan_ground_m')
    return role in GROUND_ROLES or 'grass' in a.name.lower() or (role == 'floor' and base is not None and base < .6)


def _marks(meta: dict, b, report: dict | None, limit: int = DEVIATIONS) -> tuple[list[tuple], dict]:
    """Labels for the worst `limit` deviations inside box b: scan objects off or missing (by severity), then
    unexplained placed objects (tallest first). Ground pieces are never labelled. Returns (marks, counts)."""
    if not report:
        return [], {'deviations': 0, 'labelled': 0}
    from .mapping import severity
    colours = {'missing': (255, 60, 60), 'offset': (255, 160, 0), 'shape_off': (255, 110, 40),
               'yaw_off': (255, 230, 0), 'size_off': (0, 220, 255), 'height_off': (120, 160, 255),
               'facing_off': (255, 120, 220)}
    inside = lambda x, y: b[0] <= x <= b[2] and b[1] <= y <= b[3]
    devs = [r for r in report.get('objects', []) if r['status'] != 'matched' and inside(r['x'], r['y'])]
    devs.sort(key=lambda r: (-severity(r), r['id']))
    extras = [e for e in report.get('unmatched_placed', []) if inside(e['x'], e['y']) and not _is_ground_extra(e)]
    extras.sort(key=lambda e: -(e.get('height_m') or 0))
    out = [(r['x'], r['y'], _label(r), colours.get(r['status'], (255, 255, 255))) for r in devs]
    out += [(e['x'], e['y'], f"extra {e['row']}", (200, 200, 200)) for e in extras]
    total = len(out)
    return out[:max(0, int(limit))], {'deviations': len(devs), 'extras': len(extras), 'labelled': min(total, max(0, int(limit)))}


def _report_warnings(report: dict | None, b, age: float | None) -> list[str]:
    out = []
    if not report:
        return out
    if age is not None and age > STALE_S:
        out.append(f'the scan_compare report is {age / 60:.0f} min old: run scan_compare again after changing the '
                   'build, else the labels describe an older scene')
    cb = report.get('compare_box')
    if cb is None and 'compare_box' not in report:
        cb = (report.get('scope') or {}).get('box_m')
    if cb is not None and not (cb[0] <= b[0] + 1e-6 and cb[1] <= b[1] + 1e-6 and b[2] <= cb[2] + 1e-6
                               and b[3] <= cb[3] + 1e-6):
        out.append('the scan_compare report covers box ' + ','.join(f'{v:g}' for v in cb) + ', not all of this '
                   'image: run scan_compare --box ' + ','.join(f'{v:g}' for v in b) + ' for complete labels')
    return out


def compare_view(image: str, layer: str = 'ortho', labels: bool = True, classes=None, frame: str = 'auto',
                 survey_key: str = '', deviations: int = DEVIATIONS) -> dict:
    """scan layer | build capture | edge overlay, on one grid, with the deviations of the survey's last
    scan_compare inside the box labelled on the build and overlay panels."""
    from PIL import Image
    from . import view
    path = Path(image).expanduser()
    side = path.with_suffix('.json')
    if not path.is_file() or not side.is_file():
        raise ValueError(f'compare: {path} is not a capture_topdown image (needs the PNG and its .json beside it)')
    cap = json.loads(side.read_text())
    meta, fr, ras = ops._context(survey_key, frame)
    if fr.name != 'map':
        raise ValueError('compare needs the scan placement (scan_place): the build image is in map metres')
    b = tuple(cap['box'])
    W, H = int(cap['width']), int(cap['height'])
    px = int(cap.get('px') or max(W, H))
    mpp, w2, h2 = view.crop_frame(b, px)
    if (w2, h2) != (W, H):
        px = max(W, H)
        mpp, w2, h2 = view.crop_frame(b, px)
    with Image.open(path) as im:
        build = np.asarray(im.convert('RGB'))
    if build.shape[:2] != (H, W):
        raise ValueError(f'compare: the image is {build.shape[1]} x {build.shape[0]} px, its sidecar says {W} x {H}')
    want = {c.strip().lower() for c in (classes or []) if c.strip()} or None
    objs = store.objects(meta)
    report_path = store.home(meta['key']) / 'compare.json'
    report = None
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text())
        except ValueError:
            report = None
    if not 0 <= int(deviations) <= 500:
        raise ValueError('deviations must be 0..500 (labels on the compare sheet)')
    marks, mcount = _marks(meta, b, report, deviations)
    age = round(time.time() - float(report.get('created', 0)), 0) if report else None
    warnings = _report_warnings(report, b, age)
    # scan panel: same grid; its raw pixels for the overlay
    from .view import LAYERS
    if layer not in LAYERS:
        raise ValueError(f"layer must be one of {', '.join(LAYERS)}")
    p1, drawn, _ = view.render(meta, fr, ras, objs, b, layer, labels, want, px,
                               title=f"scan {layer} | {mpp:.3f} m/px")
    raw_scan = _raw_layer(meta, fr, ras, b, px, layer)
    proj = cap.get('projection', '?')
    stale = f" | compare {age / 60:.0f} min old" if age is not None and age > STALE_S else ''
    p2, _, _ = view.render(meta, fr, ras, objs, b, layer, False, want, px, base=build, marks=marks,
                           title=f"build ({proj}) | worst {len(marks)} of scan_compare{stale}")
    ov, agree = overlay(raw_scan, build)
    p3, _, _ = view.render(meta, fr, ras, [], b, layer, False, want, px, base=ov, marks=marks,
                           title=f"edges: scan magenta, build green, both white ({agree:.0%} of scan edges)")
    panels = [Image.open(p) for p in (p1, p2, p3)]
    width = sum(p.width for p in panels) + 8 * (len(panels) - 1)
    sheet = Image.new('RGB', (width, max(p.height for p in panels)), (15, 15, 15))
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + 8
        p.close()
    out = store.views_dir(meta['key']) / f"compare-{time.strftime('%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.png"
    sheet.save(out)
    for p in (p1, p2, p3):
        Path(p).unlink(missing_ok=True)
    return {'survey': meta['key'], 'frame': fr.name, 'image': str(out), 'box': [ops._r(v) for v in b],
            'm_per_px': ops._r(mpp, 3), 'build_projection': proj, 'edge_agreement': round(agree, 3),
            'deviations_labelled': len(marks), 'deviations_in_box': mcount.get('deviations', 0),
            'extras_in_box': mcount.get('extras', 0), 'compare_report': str(report_path) if report else None,
            'compare_age_s': age, **({'warnings': warnings} if warnings else {}),
            'objects_drawn': len(drawn), 'units': 'metres',
            'meaning': 'panels: scan layer with ids | build capture | edges (scan magenta, build green, white where '
                       'both have an edge within a pixel); labels: the worst `deviations` scan_compare deviations '
                       '(id status values; extra = placed object nothing in the scan explains; ground pieces never); '
                       'edge_agreement: share of strong scan '
                       'edges the build repeats (rough: textures differ)',
            'note': 'image in RAM (/dev/shm) only: it shows the scan, keep it on this machine'
                    + ('' if report else '; no scan_compare report yet: run scan_compare for labels')}


def _raw_layer(meta: dict, fr: Frame, ras, b, px: int, layer: str) -> np.ndarray:
    """The scan layer's pixels on the crop grid, without grid or labels (for the overlay)."""
    from . import view
    mpp, W, H = view.crop_frame(b, px)
    fx = b[0] + (np.arange(W) + .5) * mpp
    fy = b[1] + (np.arange(H) + .5) * mpp
    X, Y = np.meshgrid(fx, fy)
    e, s = fr.to_scan(X, Y)
    g = ras.g
    if layer == 'ortho':
        o = meta['ortho']
        c = np.clip(np.rint((e - g.x0) / o['res'] - .5).astype(int), 0, o['w'] - 1)
        r = np.clip(np.rint((s - g.y0) / o['res'] - .5).astype(int), 0, o['h'] - 1)
        return np.asarray(ras.ortho[r, c])[..., :3].astype(np.uint8)
    c = np.clip(np.rint((e - g.x0) / g.res - .5).astype(int), 0, g.w - 1)
    r = np.clip(np.rint((s - g.y0) / g.res - .5).astype(int), 0, g.h - 1)
    nd = np.nan_to_num(np.asarray(ras.dsm[r, c], float) - np.asarray(ras.dtm[r, c], float), nan=0)
    v = np.clip(nd / max(5.0, float(np.percentile(nd, 99))), 0, 1)
    return (np.repeat(v[..., None], 3, 2) * 255).astype(np.uint8)
