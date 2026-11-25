"""Top-down crops of a survey for looking (RAM only): ortho / nDSM / classes with a metre grid and object labels."""
from __future__ import annotations

import math
import time

import numpy as np

from . import grid, store, survey

LAYERS = ('ortho', 'ndsm', 'classes')
COLOURS = {'building': (255, 70, 70), 'building_part': (255, 200, 0), 'car': (0, 240, 255), 'tree': (40, 220, 40),
           'bridge': (60, 90, 255), 'low_structure': (255, 0, 255), 'small_object': (255, 160, 0),
           'shrub': (170, 255, 170)}
PALETTE = {0: (30, 30, 30), 1: (150, 150, 150), 2: (110, 200, 90), 3: (40, 90, 200), 4: (210, 180, 120),
           5: (120, 90, 80), 6: (90, 90, 100), 7: (180, 170, 210), 8: (220, 210, 170), 10: (200, 60, 60),
           11: (20, 110, 30), 12: (240, 140, 0)}
DEFAULT_SKIP = {'building_part', 'small_object'}
STEPS = (1, 2, 5, 10, 20, 25, 50, 100, 200)


def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _ramp(v: np.ndarray) -> np.ndarray:
    """0..1 -> dark blue, cyan, green, yellow, red."""
    stops = np.array([[20, 20, 80], [0, 160, 200], [40, 200, 60], [240, 230, 40], [230, 40, 30]], float)
    x = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.minimum(np.floor(x).astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def crop_frame(box, px: int) -> tuple[float, int, int]:
    """(metres per pixel, width, height) of a top-down crop of box x0,y0,x1,y1 about px pixels wide. Pixel (i, j)
    covers x0 + i * mpp .. x0 + (i + 1) * mpp east and y0 + j * mpp .. south (north up). scan_view and
    capture_topdown share it, so their images line up pixel for pixel."""
    x0, y0, x1, y1 = box
    mpp = max(x1 - x0, y1 - y0) / px
    return mpp, max(1, round((x1 - x0) / mpp)), max(1, round((y1 - y0) / mpp))


def render(meta: dict, fr, ras, objs: list[dict], box, layer: str, labels: bool, want, px: int, base=None,
           marks=None, title: str = '', info: dict | None = None):
    """Top-down crop with grid, outlines and ids. base: an HxW RGB image (a build capture) drawn instead of the
    scan layer; marks: [(x, y, text, rgb)] labels in map metres drawn with a ring. info (a dict) receives the
    layer legend: classes {name: '#rrggbb'} present in the crop, or the ndsm ramp top."""
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = box
    mpp, W, H = crop_frame(box, px)
    fx = x0 + (np.arange(W) + .5) * mpp
    fy = y0 + (np.arange(H) + .5) * mpp
    X, Y = np.meshgrid(fx, fy)
    e, s = fr.to_scan(X, Y)
    g = ras.g
    legend = None
    if base is not None:
        img = np.asarray(base, np.uint8)[..., :3]
        if img.shape[:2] != (H, W):
            raise ValueError(f'base image is {img.shape[1]} x {img.shape[0]} px, the crop {W} x {H}')
        ok = np.ones((H, W), bool)
    elif layer == 'ortho':
        o = meta['ortho']
        c = np.rint((e - g.x0) / o['res'] - .5).astype(int)
        r = np.rint((s - g.y0) / o['res'] - .5).astype(int)
        ok = (c >= 0) & (c < o['w']) & (r >= 0) & (r < o['h'])
        img = np.asarray(ras.ortho[np.clip(r, 0, o['h'] - 1), np.clip(c, 0, o['w'] - 1)])
    else:
        c = np.rint((e - g.x0) / g.res - .5).astype(int)
        r = np.rint((s - g.y0) / g.res - .5).astype(int)
        ok = (c >= 0) & (c < g.w) & (r >= 0) & (r < g.h)
        rc, cc = np.clip(r, 0, g.h - 1), np.clip(c, 0, g.w - 1)
        if layer == 'ndsm':
            nd = np.asarray(ras.dsm[rc, cc], float) - np.asarray(ras.dtm[rc, cc], float)
            ok &= np.isfinite(nd)
            top = max(5.0, float(np.nanpercentile(nd[ok], 99))) if ok.any() else 20.0
            img = _ramp(np.nan_to_num(nd, nan=0) / top)
            legend = ('ndsm', top)
        else:
            lut = np.zeros((256, 3), np.uint8)
            for k, col in PALETTE.items():
                lut[k] = col
            codes = np.asarray(ras.cls[rc, cc])
            img = lut[codes]
            legend = ('classes', [int(k) for k in np.unique(codes[ok]) if int(k) in PALETTE])
    img = np.where(ok[..., None], img, np.uint8(25)).astype(np.uint8)
    pad_top, pad_left = 34, 46
    extra = 18 if layer in ('classes', 'ndsm') else 0   # same height for every panel of a compare sheet
    canvas = Image.new('RGB', (W + pad_left, H + pad_top + 22 + extra), (15, 15, 15))
    tile = Image.fromarray(img)
    m = ImageDraw.Draw(tile)          # map layer: drawing is clipped to the crop
    d = ImageDraw.Draw(canvas)        # margins: axis labels, title, legend
    font, small = _font(13), _font(11)

    def to_px(x, y):
        return (np.asarray(x) - x0) / mpp, (np.asarray(y) - y0) / mpp

    # metre grid in the caller's frame
    span = max(x1 - x0, y1 - y0)
    step = next((st for st in STEPS if span / st <= 8), STEPS[-1])
    for v in np.arange(math.ceil(x0 / step) * step, x1 + 1e-9, step):
        p = float(to_px(v, y0)[0])
        m.line([(p, 0), (p, H)], fill=(255, 255, 255), width=1)
        d.text((pad_left + p + 2, pad_top - 14), f'{v:g}', fill=(255, 255, 255), font=small)
    for v in np.arange(math.ceil(y0 / step) * step, y1 + 1e-9, step):
        p = float(to_px(x0, v)[1])
        m.line([(0, p), (W, p)], fill=(255, 255, 255), width=1)
        d.text((2, pad_top + p - 6), f'{v:g}', fill=(255, 255, 255), font=small)
    # objects
    drawn = []
    for o in objs:
        cls = o['cls']
        if (want and cls not in want) or (not want and cls in DEFAULT_SKIP):
            continue
        cx, cy = fr.from_scan(o['x'], o['y'])
        reach = max(o['length_m'], o['width_m'])
        if cx < x0 - reach or cx > x1 + reach or cy < y0 - reach or cy > y1 + reach:
            continue
        if o.get('footprint') and cls == 'building':
            pts = np.asarray(o['footprint'], float)
        else:
            pts = grid.box_corners(o['x'], o['y'], o['length_m'], o['width_m'], o['yaw_deg'])
        ex, ey = fr.from_scan(pts[:, 0], pts[:, 1])
        qx, qy = to_px(ex, ey)
        col = COLOURS.get(cls, (255, 255, 255))
        if cls == 'tree':
            rad = o.get('crown_m', o['length_m']) / 2 / mpp
            pcx, pcy = to_px(cx, cy)
            m.ellipse([pcx - rad, pcy - rad, pcx + rad, pcy + rad], outline=col, width=1)
        else:
            m.polygon(list(zip(qx.tolist(), qy.tolist())), outline=col, width=2 if cls in ('car', 'building') else 1)
        drawn.append((o, cx, cy, col))
    if labels:
        n_trees = sum(1 for o, *_ in drawn if o['cls'] == 'tree')
        for o, cx, cy, col in drawn:
            if o['cls'] in ('tree', 'shrub') and n_trees > 40:
                continue
            p, q = to_px(cx, cy)
            text = str(o['id'])
            m.text((float(p) - 3 * len(text) + 1, float(q) - 5), text, fill=(0, 0, 0), font=small)
            m.text((float(p) - 3 * len(text), float(q) - 6), text, fill=(255, 255, 255), font=small)
    for mx_, my_, text, col in marks or []:
        p, q = to_px(mx_, my_)
        p, q = float(p), float(q)
        m.ellipse([p - 9, q - 9, p + 9, q + 9], outline=col, width=2)
        m.text((p + 11, q - 6), text, fill=(0, 0, 0), font=small)
        m.text((p + 10, q - 7), text, fill=col, font=small)
    canvas.paste(tile, (pad_left, pad_top))
    # title, scale bar, legend
    d.text((pad_left, 2), title or f"{meta['key']} {layer} | frame {fr.name} | grid {step:g} m | {mpp:.3f} m/px",
           fill=(255, 255, 255), font=font)
    bar = 10 if span <= 150 else 50
    yb = pad_top + H + 6
    d.line([(pad_left, yb + 6), (pad_left + bar / mpp, yb + 6)], fill=(255, 255, 255), width=3)
    d.text((pad_left + bar / mpp + 6, yb), f'{bar} m', fill=(255, 255, 255), font=small)
    xl = pad_left + bar / mpp + 50
    shown = sorted({o['cls'] for o, *_ in drawn})
    for cls in shown:
        d.rectangle([xl, yb + 2, xl + 9, yb + 11], fill=COLOURS.get(cls, (255, 255, 255)))
        d.text((xl + 12, yb), cls, fill=(255, 255, 255), font=small)
        xl += 16 + 7 * len(cls)
    if legend:      # what the layer colours mean
        yl, xl = yb + 18, pad_left
        if legend[0] == 'classes':
            for k in legend[1]:
                name = survey.CLASSES.get(k, str(k))
                d.rectangle([xl, yl + 2, xl + 9, yl + 11], fill=PALETTE[k], outline=(255, 255, 255))
                d.text((xl + 12, yl), name, fill=(255, 255, 255), font=small)
                xl += 18 + 7 * len(name)
            if info is not None:
                info['class_legend'] = {survey.CLASSES.get(k, str(k)): '#%02x%02x%02x' % PALETTE[k] for k in legend[1]}
        else:
            ramp = _ramp(np.linspace(0, 1, 120))
            for i, col in enumerate(ramp):
                d.line([(xl + i, yl + 2), (xl + i, yl + 11)], fill=tuple(int(v) for v in col))
            d.text((xl + 126, yl), f'0 .. {legend[1]:.0f} m above ground', fill=(255, 255, 255), font=small)
            if info is not None:
                info['ndsm_legend'] = f'dark blue 0 m, cyan, green, yellow, red {legend[1]:.0f} m above ground'
    out = store.views_dir(meta['key']) / f"view-{time.strftime('%H%M%S')}-{layer}-{int(time.time() * 1000) % 1000:03d}.png"
    canvas.save(out)
    return out, [o for o, *_ in drawn], mpp


KIND_COLOURS = {'window': (0, 230, 255), 'door': (255, 230, 0), 'garage_door': (255, 140, 0),
                'balcony': (255, 0, 220)}


def facades(meta: dict, o: dict, walls: list[dict], imgs: list, labels: bool = True):
    """Panels of facade renders (one per wall, stacked) with detections, window rows and a metre grid."""
    from PIL import Image, ImageDraw
    from . import facade
    font, small = _font(13), _font(11)
    panels = []
    for w, im in zip(walls, imgs):
        p = w['plane']
        H0, W0 = im.shape[:2]
        k = max(1.0, min(3.0, 700 / max(W0, H0)))
        tile = Image.fromarray(np.where(im.any(2, keepdims=True), im, np.uint8(35))).resize(
            (max(1, round(W0 * k)), max(1, round(H0 * k))), Image.NEAREST)
        m = ImageDraw.Draw(tile)
        g0 = w['ground_up']

        def px(u, z):
            x, y = facade.to_px(p, u, z)
            return x * k, y * k
        # grid: every 1/2/5 m along the wall and above ground
        span = max(w['length_m'], p[7] - g0)
        step = next((st for st in STEPS if span / st <= 10), STEPS[-1])
        for v in np.arange(0, w['length_m'] + 1e-6, step):
            x, _ = px(v, 0)
            m.line([(x, 0), (x, tile.height)], fill=(90, 90, 90), width=1)
        for v in np.arange(0, p[7] - g0 + 1e-6, step):
            _, y = px(0, g0 + v)
            m.line([(0, y), (tile.width, y)], fill=(90, 90, 90), width=1)
        _, yg = px(0, g0)
        m.line([(0, yg), (tile.width, yg)], fill=(120, 255, 120), width=1)
        for d in w.get('detections') or []:
            x0, y0 = px(d['u0'], d['z1'])
            x1, y1 = px(d['u1'], d['z0'])
            m.rectangle([x0, y0, x1, y1], outline=KIND_COLOURS.get(d['kind'], (255, 255, 255)), width=2)
        grid_ = w.get('grid') or {}
        for r in grid_.get('rows') or []:
            for key, col in (('sill_m', (0, 140, 255)), ('head_m', (0, 255, 160))):
                _, y = px(0, g0 + r[key])
                m.line([(0, y), (tile.width, y)], fill=col, width=1)
            if labels:
                _, y = px(0, g0 + r['head_m'])
                m.text((3, y - 13), f"{r['count']} x  sill {r['sill_m']:g}  head {r['head_m']:g}", fill=(255, 255, 255),
                       font=small)
        head = (f"building {o['id']} side {w['side']}  faces {w['faces_deg']:.0f} deg  {w['length_m']:.1f} m  "
                f"wall from {w.get('wall_from', '?')}  {w.get('status', '')}")
        if grid_:
            head += f"  storeys {grid_.get('storeys')}  windows {grid_.get('windows')}  doors {len(grid_.get('doors') or [])}"
        panel = Image.new('RGB', (tile.width + 40, tile.height + 40), (15, 15, 15))
        d = ImageDraw.Draw(panel)
        d.text((40, 2), head, fill=(255, 255, 255), font=font)
        panel.paste(tile, (40, 20))
        for v in np.arange(0, w['length_m'] + 1e-6, step):
            x, _ = px(v, 0)
            d.text((40 + x - 4, 20 + tile.height + 3), f'{v:g}', fill=(255, 255, 255), font=small)
        for v in np.arange(0, p[7] - g0 + 1e-6, step):
            _, y = px(0, g0 + v)
            d.text((2, 20 + y - 6), f'{v:g}', fill=(255, 255, 255), font=small)
        panels.append(panel)
    W = max(pn.width for pn in panels)
    canvas = Image.new('RGB', (W, sum(pn.height for pn in panels) + 20), (15, 15, 15))
    y = 0
    for pn in panels:
        canvas.paste(pn, (0, y))
        y += pn.height
    ImageDraw.Draw(canvas).text((4, y + 3), 'window cyan, door yellow, garage door orange, balcony magenta; '
                                'rows: sill blue / head green; heights above ground (green line)',
                                fill=(200, 200, 200), font=small)
    out = store.views_dir(meta['key']) / f"facade-b{o['id']}-{time.strftime('%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.png"
    canvas.save(out)
    return out
