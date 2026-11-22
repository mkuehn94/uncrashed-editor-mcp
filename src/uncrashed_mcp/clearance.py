"""Route clearance: swept craft volumes and radial free-space probes along a flight or driving route.

The route is a centripetal Catmull-Rom spline through race-gate openings (live gates, a saved map's
Race) or explicit waypoints, sampled every `spacing` metres. The agent's `sweep` verb sweeps the craft
box between consecutive samples (BoxTraceMultiForObjects, complex collision) and fires horizontal / up /
down probes from every sample. Every hit component is asked its response to the craft's collision
channel, so overlap-only triggers (gate checkpoint volumes, flythrough objects) are not obstacles.
Objects with collision switched off are invisible to the traces. Native facts are cm; this module is metres.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile

from . import agent

REPORT_DIR = Path('/tmp/uncrashed-clearance')
CHANNELS = {'WorldStatic': 0, 'WorldDynamic': 1, 'Pawn': 2, 'Visibility': 3, 'Camera': 4,
            'PhysicsBody': 5, 'Vehicle': 6, 'Destructible': 7}
CHANNELS |= {f'engine{n}': 7 + n for n in range(1, 7)} | {f'game{n}': 13 + n for n in range(1, 19)}  # ECC_*TraceChannelN
RESPONSES = {0: 'ignore', 1: 'overlap', 2: 'block', -1: 'unknown'}
GROUND_WINDOW = 3     # ground mode: a box hit on the drape surface of a sample this close is ground contact
CHUNK = 1000          # samples per agent job (one box sweep + ndirs+2 line traces each)
MAX_SAMPLES = 40000

# Full outer dimensions in metres (props / wings included). length is along travel, width across it.
PROFILES = {
    'tiny_whoop': {'kind': 'air', 'width': .09, 'length': .09, 'height': .045, 'margin': .15,
                   'about': '65-75 mm brushless whoop, ducted 31-40 mm props'},
    'cinewhoop': {'kind': 'air', 'width': .16, 'length': .16, 'height': .07, 'margin': .25,
                  'about': '3 inch ducted cinewhoop'},
    '5inch': {'kind': 'air', 'width': .30, 'length': .30, 'height': .10, 'margin': .50,
              'about': '5 inch freestyle/racing quad, ~220 mm wheelbase, props tip to tip'},
    '7inch': {'kind': 'air', 'width': .42, 'length': .42, 'height': .13, 'margin': .75,
              'about': '7 inch long-range quad, ~300 mm wheelbase'},
    'fixed_wing': {'kind': 'air', 'width': 1.2, 'length': .9, 'height': .25, 'margin': 1.5,
                   'about': '1.2 m wingspan FPV plane, wings level (banking not modelled)'},
    'car': {'kind': 'ground', 'width': 1.9, 'length': 4.5, 'height': 1.5, 'margin': .3, 'ground_clearance': .15,
            'about': 'passenger car'},
    'truck': {'kind': 'ground', 'width': 2.55, 'length': 12.0, 'height': 4.0, 'margin': .5, 'ground_clearance': .3,
              'about': 'articulated truck, EU legal width and height'},
}
ALIASES = {'whoop': 'tiny_whoop', '5': '5inch', '5in': '5inch', '7': '7inch', '7in': '7inch',
           'wing': 'fixed_wing', 'plane': 'fixed_wing', 'cine': 'cinewhoop'}


def resolve_profile(profile: str = '5inch', dims_m=None, margin_m=None, kind=None) -> dict:
    """Craft dimensions with half extents. dims_m = [width, length, height] overrides a preset."""
    name = ALIASES.get(str(profile).lower(), str(profile).lower())
    if name not in PROFILES and dims_m is None:
        raise ValueError(f'unknown profile {profile!r}; presets: {", ".join(PROFILES)} or pass dims_m')
    p = dict(PROFILES.get(name, {'kind': 'air', 'margin': .5, 'about': 'custom'}))
    p['name'] = name if name in PROFILES else 'custom'
    if dims_m is not None:
        if len(dims_m) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 < v <= 50 for v in dims_m):
            raise ValueError('dims_m must be [width, length, height], each 0 < v <= 50 m')
        p['width'], p['length'], p['height'] = (float(v) for v in dims_m)
        p['about'] = p.get('about', '') + ' (custom dimensions)'
    if kind is not None:
        if kind not in ('air', 'ground'):
            raise ValueError("kind must be 'air' or 'ground'")
        p['kind'] = kind
    if margin_m is not None:
        if not math.isfinite(margin_m) or not 0 <= margin_m <= 20:
            raise ValueError('margin_m must be 0..20')
        p['margin'] = float(margin_m)
    if p['kind'] == 'ground':
        p.setdefault('ground_clearance', .15)
    p['half_extents_m'] = [round(p['length'] / 2, 4), round(p['width'] / 2, 4), round(p['height'] / 2, 4)]
    p['half_extents_axes'] = 'x along travel, y across, z up'
    return p


# ---------------------------------------------------------------------------- vector helpers

def _sub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def _add(a, b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def _mul(a, s): return (a[0]*s, a[1]*s, a[2]*s)
def _dot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def _len(a): return math.sqrt(_dot(a, a))


def _unit(a):
    n = _len(a)
    return _mul(a, 1/n) if n > 1e-12 else (0.0, 0.0, 0.0)


def rot_axes(pitch: float, yaw: float, roll: float = 0.0):
    """Unreal FRotator -> world X (forward), Y (right), Z (up) axes. Same as agent.c rot_axes."""
    p, y, r = (math.radians(v) for v in (pitch, yaw, roll))
    sp, cp, sy, cy, sr, cr = math.sin(p), math.cos(p), math.sin(y), math.cos(y), math.sin(r), math.cos(r)
    return ((cp*cy, cp*sy, sp),
            (sr*sp*cy - cr*sy, sr*sp*sy + cr*cy, -sr*cp),
            (-(cr*sp*cy + sr*sy), cy*sr - cr*sp*sy, cr*cp))


def heading(a, b) -> tuple[float, float]:
    """(pitch, yaw) in degrees of the direction a -> b; a box with this rotation has +X along it."""
    d = _sub(b, a)
    return math.degrees(math.atan2(d[2], math.hypot(d[0], d[1]))), math.degrees(math.atan2(d[1], d[0]))


def support(half, axes, u) -> float:
    """Distance from a box centre to its surface along unit direction u (box support function)."""
    return sum(h * abs(_dot(ax, u)) for h, ax in zip(half, axes))


def box_gap(d: float, u, half, axes) -> float:
    """Gap between the craft box (centred on the probe origin) and the probe's hit point d along u: the exact
    point-to-box distance, negative inside. Along a box axis this is d - half extent; diagonally it is not
    d - support(u), which would charge a long vehicle its full length for a wall beside it."""
    p = [d * _dot(u, ax) for ax in axes]
    out = [max(abs(c) - h, 0.0) for c, h in zip(p, half)]
    if any(out):
        return math.sqrt(sum(v * v for v in out))
    return -min(h - abs(c) for c, h in zip(p, half))


def probe_dirs(yaw: float, ndirs: int):
    """World unit vectors of the native probes: ndirs horizontal (k=0 forward, ndirs/4 right), up, down."""
    out = []
    for k in range(ndirs):
        a = math.radians(yaw) + 2*math.pi*k/ndirs
        out.append((math.cos(a), math.sin(a), 0.0))
    return out + [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]


# ---------------------------------------------------------------------------- route geometry

def _dedupe(points, closed):
    out = []
    for p in points:
        p = tuple(float(v) for v in p)
        if not out or _len(_sub(p, out[-1])) > 1e-6:
            out.append(p)
    if closed and len(out) > 2 and _len(_sub(out[0], out[-1])) <= 1e-6:
        out.pop()
    return out


def catmull_rom(points, closed=False, alpha=.5, step=.1):
    """Dense polyline of a centripetal Catmull-Rom spline that passes exactly through every point.
    Open ends are extrapolated; a closed curve returns to its first point."""
    P = _dedupe(points, closed)
    if len(P) < 2:
        raise ValueError('a route needs at least two distinct points')
    n = len(P)
    if closed:
        ext = [P[-1]] + P + [P[0], P[1 % n]]
    else:
        ext = [_sub(_mul(P[0], 2), P[1])] + P + [_sub(_mul(P[-1], 2), P[-2])]
    out = [P[0]]
    for i in range(n if closed else n - 1):
        p0, p1, p2, p3 = ext[i], ext[i+1], ext[i+2], ext[i+3]
        t0 = 0.0
        t1 = t0 + max(_len(_sub(p1, p0)), 1e-9) ** alpha
        t2 = t1 + max(_len(_sub(p2, p1)), 1e-9) ** alpha
        t3 = t2 + max(_len(_sub(p3, p2)), 1e-9) ** alpha
        steps = max(4, math.ceil(_len(_sub(p2, p1)) / step))
        for s in range(1, steps + 1):
            if s == steps:
                out.append(p2)
                continue
            t = t1 + (t2 - t1) * s / steps
            lerp = lambda a, b, ta, tb: _add(_mul(a, (tb - t)/(tb - ta)), _mul(b, (t - ta)/(tb - ta)))
            a1, a2, a3 = lerp(p0, p1, t0, t1), lerp(p1, p2, t1, t2), lerp(p2, p3, t2, t3)
            b1, b2 = lerp(a1, a2, t0, t2), lerp(a2, a3, t1, t3)
            out.append(lerp(b1, b2, t1, t2))
    return out


def resample(polyline, spacing):
    """Points every `spacing` metres of arc length plus the end point: [(point, distance)]."""
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError('spacing must be positive')
    out, walked, target = [(polyline[0], 0.0)], 0.0, spacing
    for a, b in zip(polyline, polyline[1:]):
        seg = _len(_sub(b, a))
        while seg > 0 and walked + seg >= target - 1e-9:
            f = (target - walked) / seg
            out.append((_add(a, _mul(_sub(b, a), f)), target))
            target += spacing
        walked += seg
    if walked - out[-1][1] > spacing * .05:
        out.append((polyline[-1], walked))
    elif out[-1][1] < walked:
        out[-1] = (polyline[-1], walked)
    return out


def samples_from(points_with_distance):
    """Sample records with the orientation of the segment to the next sample (last: previous segment)."""
    pts = [p for p, _ in points_with_distance]
    out = []
    for i, (p, d) in enumerate(points_with_distance):
        if i + 1 < len(pts):
            pitch, yaw = heading(p, pts[i+1])
        elif i:
            pitch, yaw = heading(pts[i-1], p)
        else:
            pitch, yaw = 0.0, 0.0
        out.append({'i': i, 'p': [round(v, 4) for v in p], 'd': round(d, 3), 'pitch': round(pitch, 3), 'yaw': round(yaw, 3)})
    return out


def gate_controls(gates, closed, lead_m=2.0):
    """Control points through gate centres, with lead-in/out points along each gate normal so the path
    crosses the opening square to it. A normal is flipped to agree with the direction of travel."""
    n = len(gates)
    if n < 2:
        raise ValueError('a gate route needs at least two gates (start + one checkpoint)')
    out = []
    for i, g in enumerate(gates):
        c = tuple(g['center'])
        prev = gates[(i - 1) % n]['center'] if (i or closed) else None
        nxt = gates[(i + 1) % n]['center'] if (i < n - 1 or closed) else None
        normal = g.get('normal')
        if not normal or lead_m <= 0:
            out.append(c)
            continue
        travel = _sub(nxt or c, prev or c)
        if _dot(normal, travel) < 0:
            normal = _mul(normal, -1)
        g['normal'] = [round(v, 6) for v in normal]
        near = [_len(_sub(c, q)) for q in (prev, nxt) if q is not None]
        lead = min([lead_m] + [.3 * v for v in near])
        if prev is not None and lead > .05:
            out.append(_sub(c, _mul(normal, lead)))
        out.append(c)
        if nxt is not None and lead > .05:
            out.append(_add(c, _mul(normal, lead)))
    if closed:  # a loop starts in the start gate, its lead-in closes the loop
        first = out.index(tuple(gates[0]['center']))
        out = out[first:] + out[:first]
    return out


def build_route(controls, spacing=1.0, closed=False):
    if not 0.1 <= spacing <= 20:
        raise ValueError('spacing_m must be 0.1..20')
    poly = catmull_rom(controls, closed=closed, step=min(.1, spacing / 4))
    pts = resample(poly, spacing)
    if len(pts) > MAX_SAMPLES:
        raise ValueError(f'route has {len(pts)} samples; raise spacing_m (limit {MAX_SAMPLES})')
    return samples_from(pts)


def nearest_sample(samples, point):
    best = min(samples, key=lambda s: _len(_sub(s['p'], point)))
    return best, _len(_sub(best['p'], point))


# ---------------------------------------------------------------------------- gates

def _floats(tokens):
    v = [float(t) for t in tokens]
    if not all(math.isfinite(x) for x in v):
        raise ValueError('non-finite native value')
    return v


def parse_gatebox(text: str) -> dict:
    gates, order, summary = {}, [], None
    for line in text.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == 'G' and len(p) in (17, 19):
            v = _floats(p[2:17])
            gates[p[1]] = {'actor': p[1], 'pivot': [round(x/100, 4) for x in v[0:3]], 'rotation': v[3:6], 'scale': v[6:9],
                           'aabb': {'min': [x/100 for x in v[9:12]], 'max': [x/100 for x in v[12:15]]}, 'components': [],
                           'row': None if len(p) < 19 or p[17] == '-' else p[17], 'start': len(p) == 19 and p[18] == '1'}
            order.append(p[1])
        elif p[0] == 'GC' and len(p) == 22:
            if p[1] not in gates:
                raise ValueError('gate component before its gate')
            v = _floats(p[7:])
            gates[p[1]]['components'].append({
                'component': p[2], 'class': p[3], 'response': int(p[4]), 'object_type': int(p[5]), 'collision_enabled': int(p[6]),
                'location': [x/100 for x in v[0:3]], 'rotation': v[3:6],
                'box_half': [x/100 for x in v[6:9]] if v[6] >= 0 else None,
                'aabb': {'min': [x/100 for x in v[9:12]], 'max': [x/100 for x in v[12:15]]}})
        elif p[0] == 'gatebox':
            summary = {k: int(v) for k, v in (t.split('=', 1) for t in p[1:])}
        else:
            raise ValueError(f'unknown gatebox row: {line}')
    if summary is None or summary['gates'] != len(order):
        raise ValueError('incomplete gatebox report')
    return {'gates': [gates[a] for a in order], 'summary': summary}


def gate_opening(gate: dict) -> dict:
    """Centre, passage normal and trigger size of a gate. The checkpoint trigger is the overlap-only,
    query-enabled box component (largest when several); its thinnest axis is the passage direction.
    Without one: bounds centre and the actor's local +Y (DefaultOneW convention)."""
    # the passable part: overlap-only trigger boxes, or (Uncrashed's gates) a QueryOnly opening plane mesh
    triggers = [c for c in gate.get('components', []) if c['box_half'] and min(c['box_half']) >= 0
                and ((c['response'] in (0, 1) and c['collision_enabled'] in (1, 3)) or c['collision_enabled'] == 1)]
    out = {'actor': gate.get('actor'), 'row': gate.get('row'), 'start': bool(gate.get('start'))}
    if triggers:
        t = max(triggers, key=lambda c: c['box_half'][0] * c['box_half'][1] * c['box_half'][2])
        axes = rot_axes(*t['rotation'])
        thin = min(range(3), key=lambda k: t['box_half'][k])
        others = [k for k in range(3) if k != thin]
        up = max(others, key=lambda k: abs(axes[k][2]))
        side = others[0] if others[1] == up else others[1]
        out.update(center=[round(v, 4) for v in t['location']], normal=[round(v, 6) for v in axes[thin]],
                   trigger_size_m={'width': round(2*t['box_half'][side], 3), 'height': round(2*t['box_half'][up], 3),
                                   'depth': round(2*t['box_half'][thin], 3)},
                   basis='trigger_volume', trigger_component=t['component'],
                   side_axis=[round(v, 6) for v in axes[side]], up_axis=[round(v, 6) for v in axes[up]])
    else:
        box = gate['aabb']
        out.update(center=[round((a + b) / 2, 4) for a, b in zip(box['min'], box['max'])],
                   normal=[round(v, 6) for v in rot_axes(*gate['rotation'])[1]], trigger_size_m=None,
                   basis='bounds_centre_local_y')
    return out


def saved_race(doc) -> tuple[list[dict], int]:
    """StartTransform then CheckPoints of a saved map: pivots, rotations and rows (no geometry)."""
    from .geom import rotator_from_quat
    race = doc.race or {}
    entries = []
    if race.get('StartTransform'):
        entries.append(('STARTGATE', race['StartTransform']))
    entries += [(cp.get('RowName') or 'DefaultOneW', cp.get('Transform') or {}) for cp in race.get('CheckPoints') or []]
    out = []
    for row, t in entries:
        tr = t.get('Translation') or {}
        pivot = [float(tr.get(k, 0)) / 100 for k in ('X', 'Y', 'Z')]
        rot = list(rotator_from_quat(t.get('Rotation') or {'X': 0, 'Y': 0, 'Z': 0, 'W': 1}))
        out.append({'row': row, 'pivot': pivot, 'rotation': rot})
    return out, int(race.get('Laps') or 1)


def match_saved(saved, live_gates, tolerance=.5):
    """Refine saved gates with the live gate at the same pivot (measured opening). Returns gates + unmatched count."""
    out, unmatched, used = [], 0, set()
    for s in saved:
        best = None
        for g in live_gates:
            d = _len(_sub(g['pivot'], s['pivot']))
            if g['actor'] not in used and d <= tolerance and (best is None or d < best[0]):
                best = (d, g)
        if best:
            used.add(best[1]['actor'])
            opening = gate_opening(best[1])
            opening['row'] = s['row']
        else:
            unmatched += 1
            opening = {'actor': None, 'row': s['row'], 'center': s['pivot'], 'trigger_size_m': None,
                       'normal': [round(v, 6) for v in rot_axes(*s['rotation'])[1]], 'basis': 'saved_pivot_unmeasured'}
        out.append(opening)
    return out, unmatched


# ---------------------------------------------------------------------------- native request / reply

def request_text(samples, profile, mode='sweep', channel=2, probe_m=10.0, ndirs=16, drape=(2.0, 50.0), first=0, last=None,
                 ignore=(), probe_only=False, drape_skip=True):
    """One native request for samples[first..last] (list positions, inclusive); every sample is probed once,
    and segment i is swept in the chunk that holds both i and i+1. probe_only: no sweeps (gate-centre probes).
    drape_skip: the drape passes over foliage and race gates (cars do not drive on trees or gate arches)."""
    last = len(samples) - 1 if last is None else last
    h = profile['half_extents_m']
    lines = [f'B {h[0]*100:.2f} {h[1]*100:.2f} {h[2]*100:.2f}', f'R {probe_m*100:.1f} {ndirs}', f'C {channel}']
    lines.append(f'M drape {drape[0]*100:.1f} {drape[1]*100:.1f} {int(drape_skip)}' if mode == 'drape' else 'M sweep')
    lines += [f'I {a}' for a in ignore]
    for pos in range(first, last + 1):
        s = samples[pos]
        sweep = int(pos < last and not probe_only)
        probe = int(pos < last or last == len(samples) - 1 or probe_only)
        x, y, z = (v * 100 for v in s['p'])
        lines.append(f'S {s["i"]} {x:.2f} {y:.2f} {z:.2f} {s["pitch"]:.4f} {s["yaw"]:.4f} {sweep} {probe}')
    return '\n'.join(lines) + '\n'


def chunks(n, size=CHUNK):
    """Inclusive [first, last] ranges overlapping by one sample so every segment is swept once."""
    if n < 1:
        return []
    out, first = [], 0
    while True:
        last = min(n - 1, first + size)
        out.append((first, last))
        if last == n - 1:
            return out
        first = last


def parse_sweep(text: str, ndirs: int) -> dict:
    out = {'hits': [], 'probes': {}, 'drape': {}, 'components': {}, 'actors': {}, 'summary': None}
    for line in text.splitlines():
        p = line.split()
        if not p:
            continue
        k = p[0]
        if k == 'H' and len(p) == 9:
            v = _floats(p[5:9])
            out['hits'].append({'seg': int(p[1]), 'actor': p[2], 'component': p[3], 'start_penetrating': p[4] == '1',
                                'impact': [round(x/100, 4) for x in v[:3]], 'time': v[3]})
        elif k == 'P' and len(p) == 2 + 3 * (ndirs + 2):
            probes = []
            for j in range(ndirs + 2):
                d = float(p[2 + 3*j])
                probes.append((None if d < 0 else d / 100, p[3 + 3*j], p[4 + 3*j]))
            out['probes'][int(p[1])] = probes
        elif k == 'D' and len(p) in (3, 5):
            out['drape'][int(p[1])] = None if p[2] == 'miss' else (float(p[2]) / 100, p[3], p[4])
        elif k == 'K' and len(p) == 7:
            out['components'][p[1]] = {'actor': p[2], 'class': p[3], 'response': int(p[4]), 'object_type': int(p[5]),
                                       'collision_enabled': int(p[6])}
        elif k == 'A' and len(p) == 4:
            out['actors'][p[1]] = {'row': None if p[2] == '-' else p[2], 'class': p[3]}
        elif k == 'sweep':
            out['summary'] = {a: (b if a == 'mode' else int(b)) for a, b in (t.split('=', 1) for t in p[1:])}
        else:
            raise ValueError(f'unknown sweep row: {line[:80]}')
    if out['summary'] is None:
        raise ValueError('incomplete sweep report')
    return out


def merge_parsed(parts):
    total = {'hits': [], 'probes': {}, 'drape': {}, 'components': {}, 'actors': {}, 'summaries': []}
    for part in parts:
        total['hits'] += part['hits']
        for key in ('probes', 'drape', 'components', 'actors'):
            total[key].update(part[key])
        total['summaries'] += part['summaries'] if 'summaries' in part else [part['summary']]  # parsed or merged
    return total


# ---------------------------------------------------------------------------- classification

def obstacle(actor, component, parsed, rows=None) -> dict | None:
    if not actor or actor == '0':
        return None
    a = parsed['actors'].get(actor, {})
    c = parsed['components'].get(component, {})
    cls, row = a.get('class', '?'), a.get('row')
    if cls == 'BP_Gate_C':
        kind = 'gate'
    elif row:
        kind = 'placed'
    elif 'Landscape' in cls:
        kind = 'terrain'
    elif cls == 'InstancedFoliageActor' or 'Foliage' in c.get('class', ''):
        kind = 'foliage'
    else:
        kind = 'base_level'
    out = {'actor': actor, 'kind': kind, 'class': cls, 'component_class': c.get('class'),
           'response': RESPONSES.get(c.get('response', -1), 'unknown')}
    if row:
        out['row'] = row
        if rows is not None:
            out['name'] = rows.name(row)
    return out


def blocking(component, parsed) -> bool:
    """Block (or unknown) for the craft's channel AND physics collision (PhysicsOnly / QueryAndPhysics or unknown).
    QueryOnly geometry is passed through by a simulated body: gate opening planes are QueryOnly and block Pawn."""
    c = parsed['components'].get(component, {})
    return c.get('response', -1) in (2, -1) and c.get('collision_enabled', -1) not in (0, 1)


def view_for(sample, profile) -> dict:
    """move_camera pose behind the sample along the route, looking at it."""
    size = max(profile['width'], profile['length'], profile['height'])
    back = max(3.0, min(25.0, size * 4 + 2))
    h = back * .35
    yaw = math.radians(sample['yaw'])
    x, y, z = sample['p']
    return {'x': round(x - back*math.cos(yaw), 2), 'y': round(y - back*math.sin(yaw), 2), 'z': round(z + h, 2),
            'yaw': round(sample['yaw'], 1), 'pitch': round(-math.degrees(math.atan2(h, back)), 1)}


def _severity(clearance, margin):
    return 3 if clearance < 0 else 2 if clearance < margin / 2 else 1


def sample_findings(samples, parsed, profile, ndirs, ignore_start_m=0.0, rows=None, judge_gates=False):
    """Per-sample findings (before merging) and per-sample minimum clearance.

    judge_gates=False drops tight/ceiling_low/ground_low findings on gates (route clearance judges each gate
    opening separately); check_opening passes True to keep them."""
    skip_gate = lambda ob: not judge_gates and ob and ob['kind'] == 'gate'  # noqa: E731
    ground = profile['kind'] == 'ground'
    half, margin = profile['half_extents_m'], profile['margin']
    total = samples[-1]['d'] if samples else 0
    found, clear = [], {}
    by_i = {s['i']: s for s in samples}
    drape_actor = {i: v[1] for i, v in parsed['drape'].items() if v}
    surface = lambda i: {drape_actor.get(j) for j in range(i - GROUND_WINDOW, i + GROUND_WINDOW + 2)} - {None}
    overlap = {}
    for h in parsed['hits']:
        s = by_i.get(h['seg'])
        if s is None:
            continue
        if not blocking(h['component'], parsed):
            cls = parsed['components'].get(h['component'], {}).get('class', '?')
            overlap[cls] = overlap.get(cls, 0) + 1
            continue
        ob = obstacle(h['actor'], h['component'], parsed, rows)
        if ob is None:
            ob = {'actor': None, 'kind': 'unknown'}
        if ground and (ob['kind'] == 'terrain' or h['actor'] in surface(s['i'])):  # the driving surface itself
            kind, sev = 'ground_contact', 1
        elif ob['kind'] == 'gate':
            kind, sev = 'gate_frame_hit', 3
        else:
            kind, sev = 'collision', 3
        found.append({'kind': kind, 'sample': s['i'], 'severity': sev, 'measure_m': None, 'obstacle': ob,
                      'impact_m': h['impact'], 'start_penetrating': h['start_penetrating']})
    for s in samples:
        probes = parsed['probes'].get(s['i'])
        if not probes:
            continue
        axes = rot_axes(s['pitch'], s['yaw'])
        dirs = probe_dirs(s['yaw'], ndirs)
        cl = [None if d is None else round(box_gap(d, u, half, axes), 4) for (d, _, _), u in zip(probes, dirs)]
        horiz = [(c, k) for k, c in enumerate(cl[:ndirs]) if c is not None]
        rec = {'min': min([c for c, _ in horiz] + [c for c in cl[ndirs:ndirs + (1 if ground else 2)] if c is not None], default=None)}
        right, left = ndirs // 4, 3 * ndirs // 4
        if ground:
            rec['left'], rec['right'] = cl[left], cl[right]
            d_l, d_r = probes[left][0], probes[right][0]
            rec['corridor_m'] = round(d_l + d_r, 3) if d_l is not None and d_r is not None else None
            for side, k in (('left', left), ('right', right)):
                c = cl[k]
                if c is not None and c < margin:
                    found.append({'kind': 'narrow_corridor', 'sample': s['i'], 'severity': _severity(c, margin), 'measure_m': c,
                                  'side': side, 'corridor_m': rec['corridor_m'],
                                  'obstacle': obstacle(probes[k][1], probes[k][2], parsed, rows)})
        elif horiz:
            c, k = min(horiz)
            ob = obstacle(probes[k][1], probes[k][2], parsed, rows)
            if c < margin and not skip_gate(ob):  # gate openings are judged per gate
                found.append({'kind': 'tight', 'sample': s['i'], 'severity': _severity(c, margin), 'measure_m': c,
                              'direction_deg': round(360 * k / ndirs, 1), 'obstacle': ob})
        up, down = cl[ndirs], cl[ndirs + 1]
        if up is not None and up < margin:
            ob = obstacle(probes[ndirs][1], probes[ndirs][2], parsed, rows)
            if not skip_gate(ob):  # a gate's top bar is judged per gate (gate_height_small)
                found.append({'kind': 'overhead_low' if ground else 'ceiling_low', 'sample': s['i'],
                              'severity': _severity(up, margin), 'measure_m': up, 'obstacle': ob})
        near_end = s['d'] < ignore_start_m or total - s['d'] < ignore_start_m
        if not ground and down is not None and down < margin and not near_end:
            ob = obstacle(probes[ndirs + 1][1], probes[ndirs + 1][2], parsed, rows)
            if not skip_gate(ob):
                found.append({'kind': 'ground_low', 'sample': s['i'], 'severity': _severity(down, margin), 'measure_m': down,
                              'obstacle': ob})
        rec['up'], rec['down'] = up, down
        clear[s['i']] = rec
    return found, clear, overlap


def gate_probe_samples(gates, base):
    """Probe-only samples exactly at each gate centre, facing along its normal: indices base + gate number."""
    out = []
    for gi, g in enumerate(gates):
        n = g.get('normal') or (0.0, 1.0, 0.0)
        pitch, yaw = heading((0, 0, 0), n)
        out.append({'i': base + gi, 'gate': gi, 'p': [round(v, 4) for v in g['center']], 'd': 0.0,
                    'pitch': round(pitch, 3), 'yaw': round(yaw, 3)})
    return out


def inside_opening(hit, gate) -> bool:
    """A swept hit on a gate's own body that lies inside its detection opening is not a frame hit: gate meshes
    can fill the opening (DefaultOneW has a pane that complex traces hit), and gates are flown through by design.
    Clipping a bar puts the impact on the bar, i.e. outside the opening."""
    if not gate or not hit.get('impact'):
        return False
    rel = _sub(hit['impact'], gate['center'])
    side = abs(sum(a*b for a, b in zip(rel, gate['side_axis'])))
    up = abs(sum(a*b for a, b in zip(rel, gate['up_axis'])))
    size = gate['trigger_size_m']
    return side < size['width'] / 2 - .01 and up < size['height'] / 2 - .01


def gate_checks(gates, samples, parsed, profile, ndirs, spacing, rows=None, probe_base=None):
    """Opening size measured by probes fired from the exact gate centre (samples from gate_probe_samples
    at probe_base + gate number), else the trigger size; plus swept hits on the gate's own frame."""
    out, margin = [], profile['margin']
    probe_base = len(samples) if probe_base is None else probe_base
    frame_hits = {}
    openings = {g.get('actor'): g for g in gates if g.get('trigger_size_m') and g.get('side_axis')}
    for h in parsed['hits']:
        if blocking(h['component'], parsed) and not inside_opening(h, openings.get(h['actor'])):
            frame_hits.setdefault(h['actor'], []).append(h['seg'])
    for gi, g in enumerate(gates):
        s, miss = nearest_sample(samples, g['center'])
        rec = {'gate': gi, 'actor': g.get('actor'), 'row': g.get('row'), 'center_m': g['center'], 'basis': g.get('basis'),
               'sample': s['i'], 'distance_m': s['d'], 'path_offset_m': round(miss, 3), 'trigger_size_m': g.get('trigger_size_m'),
               'probe_opening_m': None, 'required_m': {'width': round(profile['width'] + 2*margin, 3),
                                                      'height': round(profile['height'] + 2*margin, 3)},
               'frame_hits': len(frame_hits.get(g.get('actor'), [])), 'issues': []}
        if g.get('start'):
            # the start gate is where the craft spawns (often a pad / banner, not a fly-through opening);
            # its own body is ignored by the sweeps, so only report where the route leaves it
            rec.update(basis='start_gate', frame_hits=0)
            out.append(rec)
            continue
        if miss > max(2 * spacing, 1.0):
            rec['issues'].append('route_misses_gate_centre')
        probes = parsed['probes'].get(probe_base + gi)
        if probes:
            fwd, right, back, left = 0, ndirs // 4, ndirs // 2, 3 * ndirs // 4
            pair = lambda a, b: round(probes[a][0] + probes[b][0], 3) if probes[a][0] is not None and probes[b][0] is not None else None
            steep = abs(heading((0, 0, 0), g.get('normal') or (0, 1, 0))[0]) > 45   # flat gate: flown through vertically
            rec['probe_opening_m'] = {'width': pair(right, left), 'height': pair(fwd, back) if steep else pair(ndirs, ndirs + 1),
                                      'axes': 'horizontal pairs (flat gate)' if steep else 'left/right, up/down',
                                      'from': 'gate_centre'}
            if all(v is None for k, v in rec['probe_opening_m'].items() if k in ('width', 'height')):
                rec['probe_opening_m'] = None
        measured = rec['probe_opening_m'] or {}
        trig = rec['trigger_size_m'] or {}
        for dim, need, size in (('width', rec['required_m']['width'], profile['width']),
                                ('height', rec['required_m']['height'], profile['height'])):
            # the game's own detection opening wins; probes from the centre can start inside a gate pane
            have, basis = (trig.get(dim), 'trigger_volume') if trig.get(dim) is not None else (measured.get(dim), 'probes')
            if have is not None and have < need:
                rec['issues'].append({'kind': f'gate_{dim}_small', 'opening_m': have, 'required_m': need, 'basis': basis,
                                      'severity': 3 if have < size else 2})
        if rec['frame_hits']:
            rec['issues'].append({'kind': 'gate_frame_hit', 'segments': frame_hits[g.get('actor')][:8], 'severity': 3})
        out.append(rec)
    return out


def merge_findings(found, samples, profile, max_gap=2):
    """Join consecutive samples with the same finding and obstacle into one issue (worst sample kept)."""
    by_i = {s['i']: s for s in samples}
    found = sorted(found, key=lambda f: (f['kind'], (f.get('obstacle') or {}).get('actor') or '', f['sample']))
    issues = []
    for f in found:
        key = (f['kind'], (f.get('obstacle') or {}).get('actor'))
        last = issues[-1] if issues else None
        if last and last['_key'] == key and f['sample'] - last['_end'] <= max_gap:
            last['_end'] = f['sample']
            last['samples'] += 1
            worse = (f['measure_m'] is not None and (last['measure_m'] is None or f['measure_m'] < last['measure_m']))
            if f['severity'] > last['severity'] or worse:
                last.update({k: v for k, v in f.items() if k != 'sample'}, sample=f['sample'], severity=max(f['severity'], last['severity']))
            continue
        issues.append(dict(f, _key=key, _start=f['sample'], _end=f['sample'], samples=1))
    out = []
    for it in issues:
        s = by_i[it['sample']]
        it['at_m'] = s['p']
        it['distance_m'] = s['d']
        it['route_span_m'] = [by_i[it.pop('_start')]['d'], by_i[it.pop('_end')]['d']]
        it.pop('_key')
        it['view'] = view_for(s, profile)
        out.append(it)
    out.sort(key=lambda i: (-i['severity'], i['measure_m'] if i['measure_m'] is not None else -1e9, i['distance_m']))
    for n, it in enumerate(out, 1):
        it['id'] = n
    return out


def analyse(samples, parsed, profile, gates=(), ndirs=16, spacing=1.0, ignore_start_m=0.0, rows=None, probe_base=None) -> dict:
    openings = {g.get('actor'): g for g in gates if g.get('trigger_size_m') and g.get('side_axis')}
    passed = [h for h in parsed['hits'] if inside_opening(h, openings.get(h['actor']))]
    if passed:  # flying through a gate's own opening (its pane / detection plane) is the point of a gate
        parsed = dict(parsed, hits=[h for h in parsed['hits'] if not inside_opening(h, openings.get(h['actor']))])
    found, clear, overlap = sample_findings(samples, parsed, profile, ndirs, ignore_start_m, rows)
    gate_rows = gate_checks(list(gates), samples, parsed, profile, ndirs, spacing, rows, probe_base) if gates else []
    by_i = {s['i']: s for s in samples}
    for g in gate_rows:
        for issue in g['issues']:
            if isinstance(issue, dict) and issue['kind'] != 'gate_frame_hit':
                found.append({'kind': issue['kind'], 'sample': g['sample'], 'severity': issue['severity'],
                              'measure_m': issue['opening_m'], 'required_m': issue['required_m'], 'basis': issue['basis'],
                              'gate': g['gate'], 'obstacle': {'actor': g['actor'], 'kind': 'gate', 'row': g['row']}})
    issues = merge_findings(found, samples, profile)
    mins = [(r['min'], i) for i, r in clear.items() if r['min'] is not None]
    worst = min(mins) if mins else None
    summary = {}
    for it in issues:
        summary[it['kind']] = summary.get(it['kind'], 0) + 1
    result = {'summary': summary, 'issues': issues, 'gates': gate_rows, 'overlap_only_ignored': overlap,
              'gate_opening_hits_ignored': len(passed),
              'min_clearance_m': None if worst is None else {'clearance_m': worst[0], 'sample': worst[1],
                                                               'at_m': by_i[worst[1]]['p'], 'distance_m': by_i[worst[1]]['d']},
              'probed_samples': len(clear)}
    if profile['kind'] == 'ground':
        widths = [r['corridor_m'] for r in clear.values() if r.get('corridor_m') is not None]
        result['corridor_m'] = {'min': min(widths), 'median': sorted(widths)[len(widths)//2]} if widths else None
    result['clearance_profile'] = [[by_i[i]['d'], r['min']] for i, r in sorted(clear.items())]
    return result


# ---------------------------------------------------------------------------- live orchestration

def _channel(value) -> int:
    """ECollisionChannel index: a name (Pawn, PhysicsBody, ...), gameN / engineN (ECC_GameTraceChannelN = 13+N,
    ECC_EngineTraceChannelN = 7+N) or an index 0..31, also as a numeric string."""
    if isinstance(value, bool):
        value = None
    if isinstance(value, str):
        v = value.strip()
        low = v.lower().removeprefix('ecc_').replace('tracechannel', '')
        named = {k.lower(): i for k, i in CHANNELS.items()}
        if low in named:
            return named[low]
        if v.isdigit():
            value = int(v)
    if isinstance(value, int) and 0 <= value < 32:
        return value
    raise ValueError(f'channel must be one of {", ".join(k for k in CHANNELS if not k[-1].isdigit())}, game1..game18, '
                     'engine1..engine6 or an index 0..31')


def channel_name(index: int) -> str:
    return next((k for k, v in CHANNELS.items() if v == index), str(index))


def _run_native(command, timeout=120):
    try:
        return agent.run(command, timeout)
    except agent.AgentError as exc:
        if 'unknown verb' in str(exc):
            raise RuntimeError(f'{exc}: rebuild/reload the DLL (native/clearance.h)') from exc
        raise


def live_gates(tmp: Path, channel: int) -> dict:
    out = tmp / 'gatebox.txt'
    _run_native(f'gatebox {agent._windows_path(out)} {channel}', 60)
    return parse_gatebox(out.read_text())


def route_source(source, points, tmp, channel, closed=None, laps=None):
    """Gates (with measured openings) or waypoints, the order basis and the loop decision."""
    info = {'source': source}
    if source == 'points':
        if not points or len(points) < 2:
            raise ValueError("source='points' needs at least two [x, y, z] points (metres)")
        if not all(len(p) in (2, 3) and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p) for p in points):
            raise ValueError('each point must be [x, y] or [x, y, z] finite metres')
        info.update(laps=laps or 1, closed=bool(closed) if closed is not None else (laps or 1) > 1, gates=[])
        info['points'] = [list(map(float, p)) for p in points]
        return info
    from .mapfile import load_map, sanitize_name
    live = live_gates(tmp, channel)['gates']
    if source == 'live':
        gates, saved_laps, basis = [gate_opening(g) for g in live], None, 'level_order'
        try:
            folder = agent.map_info().get('folder')
            saved, saved_laps = saved_race(load_map(sanitize_name(folder))) if folder else ([], None)
            matched, unmatched = match_saved(saved, live)
            if saved and not unmatched and len(saved) == len(live):
                gates, basis = matched, 'saved_race_order'
        except Exception:  # unsaved map: level order is all there is
            saved_laps = None
    elif source.startswith('map:'):
        name = sanitize_name(source[4:])
        saved, saved_laps = saved_race(load_map(name))
        gates, unmatched = match_saved(saved, live)
        basis = 'saved_race_order'
        info['unmeasured_gates'] = unmatched
        info['loaded_map'] = agent.map_info().get('folder')
        if info['loaded_map'] != name:
            info['warning'] = f'route from saved map {name} checked against the loaded world {info["loaded_map"]}'
    else:
        raise ValueError("source must be 'live', 'map:<name>' or 'points'")
    if len(gates) < 2:
        raise ValueError(f'{len(gates)} race gate(s) found; a gate route needs a start and at least one checkpoint')
    laps = laps or saved_laps or 1
    info.update(gates=gates, order_basis=basis, laps=laps, closed=bool(closed) if closed is not None else laps > 1)
    return info


def drape_samples(samples, profile, tmp, channel, up_m=2.0, down_m=50.0):
    """Ground mode: put each sample on the highest blocking surface below it (foliage and gates passed over),
    at ride height. Returns the draped samples, the parsed drape replies and the samples without a surface."""
    parts = []
    for first, last in chunks(len(samples)):
        req, out = tmp / 'drape-request.txt', tmp / 'drape.txt'
        req.write_text(request_text(samples, profile, 'drape', channel, 0, 0, (up_m, down_m), first, last, drape_skip=True))
        _run_native(f'sweep {agent._windows_path(req)} {agent._windows_path(out)}')
        parts.append(parse_sweep(out.read_text(), 0))
    parsed = merge_parsed(parts)
    lift = profile['ground_clearance'] + profile['half_extents_m'][2]
    pts, missing = [], []
    for s in samples:
        hit = parsed['drape'].get(s['i'])
        z = hit[0] + lift if hit else s['p'][2]
        if not hit:
            missing.append(s['i'])
        pts.append(((s['p'][0], s['p'][1], z), s['d']))
    return samples_from(pts), parsed, missing


def check_route_clearance(source='live', points=None, profile='5inch', margin_m=None, spacing_m=1.0, dims_m=None,
                          kind=None, channel='Pawn', probe_m=10.0, directions=16, closed=None, laps=None,
                          align_gates=True, gate_lead_m=2.0, ignore_start_m=0.0, max_issues=20, html=True,
                          catalog=None) -> dict:
    prof = resolve_profile(profile, dims_m, margin_m, kind)
    ch = _channel(channel)
    if directions not in (4, 8, 12, 16, 24, 32):
        raise ValueError('directions must be 4, 8, 12, 16, 24 or 32')
    if not 1 <= probe_m <= 200 or not 1 <= max_issues <= 200:
        raise ValueError('probe_m must be 1..200 and max_issues 1..200')
    if laps is not None and not (isinstance(laps, int) and 1 <= laps <= 99):
        raise ValueError('laps must be 1..99')
    if not math.isfinite(ignore_start_m) or ignore_start_m < 0:
        raise ValueError('ignore_start_m must be >= 0')
    rows = None
    if catalog is not None:
        from .collide import Rows
        rows = Rows.from_catalog(catalog)
    ground = prof['kind'] == 'ground'
    with agent.session(), tempfile.TemporaryDirectory(prefix='uncrashed-clearance-', dir='/tmp') as tmp:
        tmp = Path(tmp)
        info = route_source(source, points, tmp, ch, closed, laps)
        if info['gates']:
            controls = gate_controls(info['gates'], info['closed'], gate_lead_m if align_gates else 0)
        else:
            if not ground and any(len(p) != 3 for p in info['points']):
                raise ValueError('air routes need [x, y, z] points')
            controls = [tuple(p) if len(p) == 3 else (p[0], p[1], 0.0) for p in info['points']]
        samples = build_route(controls, spacing_m, info['closed'])
        drape_missing, drape_up = [], 2.0
        if ground:
            if info.get('points') and any(len(p) == 2 for p in info['points']):
                drape_up = 1000.0  # no z given: the top-most surface
            samples, drape_parsed, drape_missing = drape_samples(samples, prof, tmp, ch, drape_up)
        parts = []
        for first, last in chunks(len(samples)):
            req, out = tmp / 'sweep-request.txt', tmp / 'sweep.txt'
            req.write_text(request_text(samples, prof, 'sweep', ch, probe_m, directions, first=first, last=last,
                                        ignore=[g['actor'] for g in info.get('gates', []) if g.get('start') and g.get('actor')]))
            _run_native(f'sweep {agent._windows_path(req)} {agent._windows_path(out)}')
            parts.append(parse_sweep(out.read_text(), directions))
        probe_base = len(samples)
        if info['gates']:
            gate_samples = gate_probe_samples(info['gates'], probe_base)
            req, out = tmp / 'gate-request.txt', tmp / 'gate-probes.txt'
            req.write_text(request_text(gate_samples, prof, 'sweep', ch, probe_m, directions, probe_only=True))
            _run_native(f'sweep {agent._windows_path(req)} {agent._windows_path(out)}')
            parts.append(parse_sweep(out.read_text(), directions))
        if ground:
            parts.append(drape_parsed)  # drape surfaces (ground_contact) and the drape jobs' completeness
        parsed = merge_parsed(parts)
    result = analyse(samples, parsed, prof, info['gates'], directions, spacing_m, ignore_start_m, rows, probe_base)
    if drape_missing:
        result['summary']['no_surface'] = len(drape_missing)
    native = parsed['summaries']
    truncated = any(s.get('truncated') or s.get('seen_full') for s in native)
    failed = sum(s.get('failed', 0) for s in native)
    report = {
        'units': 'metres', 'mode': 'ground' if ground else 'air', 'profile': prof,
        'collision_channel': {'name': channel_name(ch), 'index': ch},
        'route': {k: v for k, v in info.items() if k not in ('gates', 'points')} | {
            'length_m': samples[-1]['d'], 'samples': len(samples), 'spacing_m': spacing_m,
            'align_gates': align_gates and bool(info['gates']), 'drape_missing_samples': drape_missing[:50]} | (
            {'drape_from_m': drape_up, 'drape_skipped_hits': sum(s.get('skipped', 0) for s in native if s.get('mode') == 'drape')}
            if ground else {}),
        'probe': {'length_m': probe_m, 'horizontal_directions': directions},
        'complete': not truncated and not failed, 'native': {'jobs': len(native), 'truncated': truncated, 'failed_traces': failed,
                                                             'ms': sum(s.get('ms', 0) for s in native)},
        **result,
        'limitations': [
            'Swept volume is the craft box oriented along the route (no banking, no roll, no prop wash or flight dynamics).',
            'The route is a smooth spline through gate centres, not a recorded flight line; real lines cut corners.',
            f'Obstacles are components blocking the {channel_name(ch)} channel; which channel the drone really collides as '
            'is not verified. Unknown responses count as blocking.',
            "A real block needs both sides: the obstacle blocking the craft's channel (checked) and the craft blocking the "
            "obstacle's object type (the craft's own collision settings are unknown, assumed to block everything traced).",
            'Objects whose collision is disabled are invisible to the traces; Pawn object types are not traced.',
            'Probe clearance is measured from the craft box surface along 2 + horizontal directions, not in every direction.',
            'Gate openings come from probes fired from the exact gate centre (trigger centre), else from the trigger size.',
            *(['Ground mode: corridor probes run at ride height (box centre); kerbs and barriers lower than that only show up '
               'as sweep hits, not in corridor_m.',
               'Ground mode: the drape takes the highest non-foliage, non-gate blocking surface from drape_from_m above the '
               'waypoint; with [x, y] waypoints a bridge deck or roof above the road wins, so give z there.']
              if ground else []),
            'Only loaded geometry is tested.'],
        'path': _path(samples),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / 'report.json'
    path.write_text(json.dumps(report, indent=1))
    reply = {k: report[k] for k in ('mode', 'complete', 'summary', 'min_clearance_m', 'overlap_only_ignored',
                                     'gate_opening_hits_ignored', 'collision_channel')}
    reply['profile'] = {k: prof[k] for k in ('name', 'kind', 'width', 'length', 'height', 'margin', 'half_extents_m')
                        } | ({'ground_clearance': prof['ground_clearance']} if ground else {})
    reply['route'] = report['route']
    if ground:
        reply['corridor_m'] = result.get('corridor_m')
    reply['gates'] = [{k: g[k] for k in ('gate', 'row', 'basis', 'probe_opening_m', 'trigger_size_m', 'frame_hits')} | {
        'issues': [i if isinstance(i, str) else i['kind'] for i in g['issues']]} for g in result['gates'] if g['issues']][:max_issues]
    reply['issues'] = [{k: v for k, v in it.items() if k in ('id', 'kind', 'severity', 'at_m', 'distance_m', 'route_span_m',
                                                             'measure_m', 'required_m', 'obstacle', 'view', 'side', 'gate')}
                       for it in result['issues'][:max_issues]]
    reply['issue_count'] = len(result['issues'])
    reply['report_json'] = str(path)
    if html:
        page = REPORT_DIR / 'report.html'
        page.write_text(render_html(report))
        reply['report_html'] = str(page)
    return reply


# ---------------------------------------------------------------------------- report page

def _path(samples, limit=2000):
    """Route polyline for the plan: [x, y, z, distance] at most ~limit points, always ending at the last sample."""
    picked = samples[::max(1, len(samples) // limit)]
    if picked and picked[-1] is not samples[-1]:
        picked.append(samples[-1])
    return [list(s['p']) + [s['d']] for s in picked]


_PAGE_PATH = Path(__file__).parent / 'data' / 'clearance_page.html'


def render_html(report: dict, title: str = 'Route clearance') -> str:
    data = {k: report.get(k) for k in ('mode', 'profile', 'route', 'summary', 'issues', 'gates', 'min_clearance_m',
                                       'collision_channel', 'limitations', 'path', 'clearance_profile', 'corridor_m')}
    payload = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
    page = _PAGE_PATH.read_text() if _PAGE_PATH.exists() else '<pre id="raw"></pre><script id="report" type="application/json">__DATA__</script>'
    body = page.replace('__TITLE__', title.replace('&', '&amp;').replace('<', '&lt;')).replace('__DATA__', payload)
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '</head><body>' + body + '</body></html>')
