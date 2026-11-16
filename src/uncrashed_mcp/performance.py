"""Performance diagnostics for big maps: counts, density, lights, culling, measured frame time.

Works offline on a saved map JSON (counts straight from the file + catalogue) and live
(native `perf` walks every loaded actor and its components). Budgets are HEURISTIC
defaults for a UE4 game on a mid-range PC, not measured limits of Uncrashed; override
them. Every placed editor object is its own actor with its own components (unless the
live scan reports InstancedStaticMeshComponents), so object count drives per-object CPU
cost (visibility, culling, transforms, collision bodies); UE 4.22+ may still merge draw
calls of identical mesh+material pairs in the renderer (dynamic instancing).
Units: metres here; native speaks cm.
"""
from __future__ import annotations

import json
import math
import re
import struct
import tempfile
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from . import agent, consolidate, meshes
from .geom import quat_rotate, quat_rotator

REPORT_DIR = Path('/tmp/uncrashed-perf')
ROWFLAGS_CACHE = REPORT_DIR / 'rowflags.txt'
SEVERITY = ('ok', 'info', 'warn', 'high')

# name: (warn, high, label). Heuristic, overridable per call.
BUDGETS = {
    'placed_objects': (4000, 10000, 'placed editor objects (each is an actor)'),
    'mesh_components': (5000, 12000, 'visible mesh components of placed objects (per-component CPU visibility/culling/'
                                     'transform cost, plus draw cost unless the renderer auto-instances)'),
    'blueprint_objects': (300, 1000, 'placed Blueprint-row objects (construction/tick/extra components)'),
    'triangles': (8e6, 25e6, 'LOD0 triangles of everything placed (estimate, before culling)'),
    'shadow_casters': (4000, 10000, 'shadow-casting primitives of placed objects'),
    'never_culled': (2000, 6000, 'primitives of placed objects with no max draw distance'),
    'dynamic_lights': (16, 48, 'visible movable/stationary lights on placed objects'),
    'shadow_lights': (4, 12, 'visible shadow-casting lights on placed objects'),
    'physics_objects': (50, 200, 'primitives of placed objects simulating physics'),
    'cell_objects': (300, 800, 'placed objects per 50 m x 50 m cell (scaled by area for other cell sizes)'),
    'cell_triangles': (1.5e6, 4e6, 'LOD0 triangles per 50 m x 50 m cell (scaled by area for other cell sizes)'),
    'row_count': (500, 2000, 'placed copies of one row'),
    'frame_ms_avg': (16.7, 33.3, 'measured average frame time, ms'),
    'frame_ms_p95': (25.0, 50.0, 'measured 95th percentile frame time, ms'),
}
BASE_CELL_M = 50.0
SCALAR_HIGH = 2.5  # a scalar override sets warn; high = 2.5 x warn
LIGHT_WORDS = ('light', 'lamp', 'neon')
RENDER_CVARS = ('r.MeshDrawCommands.DynamicInstancing', 'r.MeshDrawCommands.UseCachedCommands')
FRAME_CVARS = ('t.MaxFPS', 'r.VSync', 't.IdleWhenNotForeground')
LIMITATIONS = [
    'Budgets are heuristic UE4 defaults (override them), not limits measured for Uncrashed or your GPU.',
    'Triangle estimates come from the extracted LOD0 hull: exact for meshes under ~8000 triangles, capped near 8000 '
    'for detailed ones, unknown (0) for Blueprint rows unless the live engine exposes a count. LODs are not modelled.',
    'Offline mode cannot see lights, shadows, culling or physics unless row flags were cached by a live run.',
    'Frame time is sampled once per agent job (~50-100 ms apart) in the editor, not in flight; it is a coarse sample. '
    'VSync, t.MaxFPS, the engine smoothed frame-rate range and an unfocused/background window can all raise it '
    'above the real cost.',
    'Live shadow/cull/physics/mesh-component budgets grade placed objects only; *_base figures (landscape, foliage, '
    'base-map actors) are informational because the map editor cannot change them.',
]


# ---------------------------------------------------------------------------
# budgets


def budgets(overrides: dict | None = None) -> dict[str, dict]:
    out = {k: {'warn': w, 'high': h, 'label': label} for k, (w, h, label) in BUDGETS.items()}
    for key, value in (overrides or {}).items():
        if key not in out:
            raise ValueError(f'unknown budget {key!r}; known: {", ".join(sorted(BUDGETS))}')
        if isinstance(value, dict):
            if not value or set(value) - {'warn', 'high'}:
                raise ValueError(f'budget {key} dict takes "warn" and/or "high"')
            warn = float(value['warn']) if 'warn' in value else None
            high = float(value['high']) if 'high' in value else None
            if high is None:  # raise high with warn, as the scalar form does
                high = max(out[key]['high'], warn * SCALAR_HIGH) if math.isfinite(warn) else warn
            if warn is None:
                warn = min(out[key]['warn'], high)
        else:
            warn = float(value)
            high = max(warn * SCALAR_HIGH, warn)
        if not (math.isfinite(warn) and math.isfinite(high) and 0 <= warn <= high):
            raise ValueError(f'budget {key} needs finite 0 <= warn <= high')
        out[key].update(warn=warn, high=high, overridden=True)
    return out


def cell_limits(limits: dict, cell: float) -> dict:
    """Per-cell budgets are defined for 50 m cells; scale defaults by area for other cell sizes."""
    out = dict(limits)
    f = (cell / BASE_CELL_M) ** 2
    for key in ('cell_objects', 'cell_triangles'):
        if not out[key].get('overridden') and f != 1:
            out[key] = {**out[key], 'warn': round(out[key]['warn'] * f, 1), 'high': round(out[key]['high'] * f, 1),
                        'label': f'{out[key]["label"]}; scaled x{f:g} for {cell:g} m cells'}
    return out


def grade(value: float, budget: dict) -> str:
    if value is None:
        return 'ok'
    if value > budget['high']:
        return 'high'
    if value > budget['warn']:
        return 'warn'
    return 'ok'


# ---------------------------------------------------------------------------
# data


@lru_cache(maxsize=1)
def hull_triangles() -> dict[str, int]:
    """Triangle count per source mesh from meshes.bin headers (no geometry decode)."""
    path = meshes.DATA
    if not path.is_file():
        return {}
    blob = path.read_bytes()
    if blob[:4] != b'UMSH':
        return {}
    version, count = struct.unpack_from('<II', blob, 4)
    if version != 1:
        raise RuntimeError(f'meshes.bin version {version} is not supported')
    pos, out = 12, {}
    for _ in range(count):
        nlen, = struct.unpack_from('<H', blob, pos)
        name = blob[pos + 2:pos + 2 + nlen].decode('utf-8')
        pos += 2 + nlen
        nv, ni = struct.unpack_from('<II', blob, pos)
        pos += 8 + nv * 6 + ni * 2
        out[name] = ni // 3
    return out


def row_triangles(catalog, row: str) -> int:
    asset = catalog.assets.get(row)
    if asset is None or not asset.source:
        return 0
    return hull_triangles().get(asset.source, 0)


_SUFFIX = re.compile(r'_\d+_[0-9A-Fa-f]{32}$')


def _key(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', _SUFFIX.sub('', name).lower())


def parse_rowflags(text: str) -> dict[str, dict]:
    """R\\trow\\tfield\\ttype\\tvalue... -> {row: {normalised field: value}} (BP GUID suffixes stripped)."""
    rows = {}
    for line in text.splitlines():
        parts = line.split('\t')
        if not parts or parts[0] != 'R' or len(parts) < 2 or (len(parts) - 2) % 3:
            continue
        fields = {}
        for i in range(2, len(parts), 3):
            name, kind, value = parts[i:i + 3]
            if kind == 'BoolProperty':
                fields[_key(name)] = value == '1'
            elif kind in ('FloatProperty', 'IntProperty', 'ByteProperty', 'EnumProperty'):
                try:
                    fields[_key(name)] = float(value)
                except ValueError:
                    pass
            else:
                fields[_key(name)] = value
        rows[parts[1]] = fields
    return rows


def flag(flags: dict | None, *names, default=None):
    if not flags:
        return default
    for name in names:
        if _key(name) in flags:
            return flags[_key(name)]
    return default


def parse_perf(text: str) -> dict:
    actors, instanced, lights, end = [], [], [], None
    for line in text.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == 'A' and len(p) == 28:
            nums = [float(v) for v in p[5:11]]
            ints = [int(v) for v in p[11:]]
            names = ('prim', 'sm', 'ism', 'inst', 'shadow', 'coll', 'phys', 'lights', 'dynlights', 'shadowlights',
                     'nevercull', 'hidden', 'skel', 'fx', 'tris', 'movable', 'actorhidden')
            rec = {'actor': p[1], 'editor': p[2] == '1', 'row': None if p[3] == '-' else p[3], 'cls': p[4],
                   'center': [v / 100 for v in nums[:3]], 'half': [v / 100 for v in nums[3:]]}
            rec.update(zip(names, ints))
            actors.append(rec)
        elif p[0] == 'I' and len(p) == 5:
            instanced.append({'actor': p[1], 'cls': p[2], 'mesh': p[3], 'instances': int(p[4])})
        elif p[0] == 'L' and len(p) == 9:
            lights.append({'actor': p[1], 'cls': p[2], 'mobility': int(p[3]), 'shadows': int(p[4]),
                           'intensity': float(p[5]), 'at_m': [round(float(v) / 100, 2) for v in p[6:9]]})
        elif p[0] == 'end':
            end = {k: int(v) for k, v in re.findall(r'(\w+)=(-?\d+)', line)}
        else:
            raise ValueError(f'unexpected perf line: {line[:80]}')
    if end is None or end.get('actors') != len(actors):
        raise ValueError('incomplete perf scan')
    return {'actors': actors, 'instanced': instanced, 'lights': lights, 'end': end}


def offline_records(objects: list[dict], catalog, rowflags: dict | None) -> list[dict]:
    """Saved-map objects -> per-object records shaped like live A lines (flags from row data when known)."""
    out = []
    for o in objects:
        asset = catalog.assets.get(o['row'])
        flags = (rowflags or {}).get(o['row'])
        q = quat_rotator(*o['xf'][3:6])
        if asset:
            local = [asset.origin_cm[i] * o['xf'][6 + i] for i in range(3)]
            c = quat_rotate(q, local)
            half = [0.0, 0.0, 0.0]
            for axis in range(3):
                e = [0.0, 0.0, 0.0]
                e[axis] = asset.extent_cm[axis] * abs(o['xf'][6 + axis])
                w = quat_rotate(q, e)
                half = [half[i] + abs(w[i]) for i in range(3)]
            center = [(o['xf'][i] + c[i]) / 100 for i in range(3)]
            half = [v / 100 for v in half]
        else:
            center, half = [o['xf'][i] / 100 for i in range(3)], [0.0, 0.0, 0.0]
        end_cull = flag(flags, 'Endculldistance')
        out.append({'actor': o['actor'], 'editor': True, 'row': o['row'], 'cls': 'Actor', 'center': center, 'half': half,
                    'shadow': None if flags is None else int(bool(flag(flags, 'CastdynamicShadow', default=True))),
                    'phys': None if flags is None else int(bool(flag(flags, 'PhysicsEnabled?', default=False))),
                    'coll': int(bool(flag(flags, 'Collision?', default=asset.collision if asset else True))),
                    'nevercull': None if end_cull is None else int(end_cull <= 0)})
    return out


# ---------------------------------------------------------------------------
# analysis


def _view(x, y, z, size):
    d = max(12.0, min(60.0, size * 0.9 + 10.0))
    h = max(6.0, d * 0.6)
    a = math.radians(225)
    cx, cy, cz = x + d * math.cos(a), y + d * math.sin(a), z + h
    return {'x': round(cx, 1), 'y': round(cy, 1), 'z': round(cz, 1),
            'yaw': round(math.degrees(math.atan2(y - cy, x - cx)), 1), 'pitch': round(-math.degrees(math.atan2(h, d)), 1)}


def _sum(records, key):
    vals = [r.get(key) for r in records]
    known = [v for v in vals if v is not None]
    return sum(known) if known and len(known) == len(vals) else (sum(known) if known else None)


def analyse(records: list[dict], catalog, limits: dict, cell: float = 50.0, top: int = 10,
            lights: list | None = None, instanced: list | None = None, live: bool = False,
            consolidation: dict | None = None, render_cvars: dict | None = None) -> dict:
    if not 5 <= cell <= 1000:
        raise ValueError('cell must be 5..1000 m')
    limits = cell_limits(limits, cell)
    placed = [r for r in records if r['editor'] and r['cls'] != 'BP_Gate_C']
    gates = [r for r in records if r['cls'] == 'BP_Gate_C']
    base = [r for r in records if not r['editor']]
    tris_live = live and any((r.get('tris') or 0) > 0 for r in placed)
    by_row = defaultdict(list)
    for r in placed:
        r['tri_est'] = r['tris'] if tris_live and r.get('tris', -1) >= 0 else row_triangles(catalog, r['row'] or '')
        by_row[r['row'] or '-'].append(r)
    cats, kinds = defaultdict(int), defaultdict(int)
    for row, items in by_row.items():
        asset = catalog.assets.get(row)
        cats[asset.category if asset else 'unknown'] += len(items)
        kinds[asset.kind if asset else 'unknown'] += len(items)
    total_tris = sum(r['tri_est'] for r in placed)
    rows_ranked = sorted(by_row.items(), key=lambda kv: -len(kv[1]))
    top_rows = []
    for row, items in rows_ranked[:top]:
        asset = catalog.assets.get(row)
        each = row_triangles(catalog, row)
        top_rows.append({'row': row, 'name': asset.name if asset else row, 'category': asset.category if asset else None,
                         'kind': asset.kind if asset else None, 'count': len(items), 'triplanar': bool(asset and asset.triplanar),
                         'triangles_each': each, 'triangles_total': sum(r['tri_est'] for r in items)})
    grid = defaultdict(lambda: {'objects': 0, 'triangles': 0, 'zs': [], 'rows': defaultdict(int)})
    for r in placed:
        g = grid[(math.floor(r['center'][0] / cell), math.floor(r['center'][1] / cell))]
        g['objects'] += 1
        g['triangles'] += r['tri_est']
        g['zs'].append(r['center'][2])
        g['rows'][r['row'] or '-'] += 1

    def cell_info(key, g):
        zs = sorted(g['zs'])
        x, y, z = (key[0] + .5) * cell, (key[1] + .5) * cell, zs[len(zs) // 2]
        rows = sorted(g['rows'].items(), key=lambda kv: -kv[1])[:3]
        return {'center_m': [round(x, 1), round(y, 1), round(z, 1)], 'objects': g['objects'], 'triangles': g['triangles'],
                'top_rows': [{'row': r, 'name': catalog.assets[r].name if r in catalog.assets else r, 'count': n} for r, n in rows],
                'camera': _view(x, y, z, cell)}
    by_tris = sorted(by_row.items(), key=lambda kv: -sum(r['tri_est'] for r in kv[1]))
    top_tris = [{'row': row, 'name': catalog.assets[row].name if row in catalog.assets else row, 'count': len(items),
                 'triangles_total': sum(r['tri_est'] for r in items)} for row, items in by_tris[:max(3, min(top, 10))]
                if sum(r['tri_est'] for r in items) > 0]
    densest = [cell_info(k, g) for k, g in sorted(grid.items(), key=lambda kv: -kv[1]['objects'])[:top]]
    heaviest = [cell_info(k, g) for k, g in sorted(grid.items(), key=lambda kv: -kv[1]['triangles'])[:min(top, 5)]]
    metrics = {
        'placed_objects': len(placed), 'gates': len(gates), 'level_actors': len(records) if live else None,
        'non_editor_actors': len(records) - len(placed) - len(gates) if live else None,
        'blueprint_objects': kinds.get('blueprint', 0), 'triangles': total_tris,
        'triangles_source': 'engine LOD0 counts' if tris_live else 'catalogue hull estimate',
        'shadow_casters': _sum(placed, 'shadow'),
        'never_culled': _sum(placed, 'nevercull'),
        'physics_objects': _sum(placed, 'phys'),
        'collision_objects': _sum(placed, 'coll'),
        'mesh_components': (sum(r.get('sm', 0) + r.get('ism', 0) for r in placed) if live else None),
        'grid_cells': len(grid), 'cell_m': cell,
    }
    if live:  # base level / other actors: informational, not graded
        for key, field in (('shadow_casters', 'shadow'), ('never_culled', 'nevercull'), ('physics_objects', 'phys')):
            metrics[key + '_base'] = _sum(base, field) or 0
        metrics['mesh_components_base'] = sum(r.get('sm', 0) + r.get('ism', 0) for r in base)
        lights = lights or []
        mine = {r['actor'] for r in placed}
        own = [l for l in lights if l['actor'] in mine]
        metrics['lights'] = len(own)
        metrics['dynamic_lights'] = sum(1 for l in own if l['mobility'] >= 1)
        metrics['shadow_lights'] = sum(1 for l in own if l['shadows'] == 1)
        metrics['lights_base'] = len(lights) - len(own)
        metrics['dynamic_lights_base'] = sum(1 for l in lights if l['actor'] not in mine and l['mobility'] >= 1)
        metrics['shadow_lights_base'] = sum(1 for l in lights if l['actor'] not in mine and l['shadows'] == 1)
    else:
        metrics['possible_light_objects'] = sum(len(v) for k, v in by_row.items() if k in catalog.assets and any(
            w in catalog.assets[k].name.lower() or w in catalog.assets[k].tags for w in LIGHT_WORDS))
        metrics['dynamic_lights'] = metrics['shadow_lights'] = None
    eng = None
    if instanced is not None:
        ed = {r['actor'] for r in placed}
        own = [i for i in instanced if i['actor'] in ed]
        eng = {'editor_ism_components': len(own), 'editor_instances': sum(max(0, i['instances']) for i in own),
               'other_ism_components': len(instanced) - len(own),
               'other_instances': sum(max(0, i['instances']) for i in instanced if i['actor'] not in ed),
               'renderer_cvars': render_cvars or {}}
        dyn = (render_cvars or {}).get('r.MeshDrawCommands.DynamicInstancing')
        renderer = ('the renderer may still merge draw calls of identical mesh+material pairs (UE 4.22+ dynamic '
                    'instancing' + ('' if dyn is None else f', r.MeshDrawCommands.DynamicInstancing={dyn:g}') + ')')
        eng['meaning'] = ('placed objects own instanced (ISM/HISM) components: the game instances some editor objects'
                          if own else 'no placed object owns an ISM component: each placed object is its own actor + '
                          'component with its own CPU cost (visibility, culling, transform, collision body); ' + renderer)
        if dyn == 0:
            eng['meaning'] += '; dynamic instancing is OFF, so each mesh component is its own draw'
    warnings = []

    def warn(metric, value, message, advice, budget_key=None):
        b = limits[budget_key or metric]
        sev = grade(value, b)
        if sev != 'ok':
            warnings.append({'severity': sev, 'metric': metric, 'value': value, 'warn_at': b['warn'], 'high_at': b['high'],
                             'budget': b['label'], 'message': message, 'advice': advice})

    cons = consolidation or {}
    merge_hint = (f'consolidate_scenery can merge {cons.get("objects_removed", 0)} triplanar box tiles into '
                  f'{cons.get("objects_added", 0)} scaled copies (saves {cons.get("saved", 0)}). ' if cons.get('saved') else '')
    big_rows = ', '.join(f'{t["name"]} ({t["row"]}) x{t["count"]}' for t in top_rows[:4])
    warn('placed_objects', len(placed), f'{len(placed)} placed objects',
         merge_hint + f'Biggest rows: {big_rows}. Thin out repeated props, prefer larger variants.')
    if metrics['mesh_components'] is not None:
        warn('mesh_components', metrics['mesh_components'], f'{metrics["mesh_components"]} visible mesh components',
             merge_hint + 'Every component costs CPU each frame (visibility, culling, transform) and draw work unless the '
             'renderer auto-instances identical mesh+material pairs; fewer, larger objects cut both.')
    warn('blueprint_objects', metrics['blueprint_objects'], f'{metrics["blueprint_objects"]} Blueprint objects',
         'Blueprint rows (decals, interactive and animated props) cost more than meshes; keep them where they matter.')
    warn('triangles', total_tris, f'about {total_tris:,} LOD0 triangles placed',
         'Heaviest rows by triangles: ' + ', '.join(f'{t["name"]} ({t["row"]}) {t["triangles_total"]:,}' for t in top_tris[:3])
         + '. Use fewer of them or lighter variants.')
    for key, msg, adv in (
            ('shadow_casters', 'shadow-casting primitives', 'Shadow depth passes redraw every caster; thin dense small props.'),
            ('never_culled', 'primitives never distance-culled', 'Rows with no cull distance render at any range; '
             'avoid scattering thousands of small never-culled props over the whole map.'),
            ('physics_objects', 'physics-simulating primitives', 'Each simulating body costs CPU every frame.'),
            ('dynamic_lights', 'movable/stationary lights', 'Each dynamic light re-shades what it reaches; keep neons/lamps sparse.'),
            ('shadow_lights', 'shadow-casting lights', 'Shadowed dynamic lights re-render shadow depth for objects in range.')):
        if metrics.get(key) is not None:
            warn(key, metrics[key], f'{metrics[key]} {msg}', adv)
    for c in densest[:3]:
        warn('cell_objects', c['objects'], f'{c["objects"]} objects in the {cell:g} m cell at {c["center_m"][:2]}',
             f'Mostly {", ".join(r["name"] for r in c["top_rows"])}; look with move_camera {c["camera"]}.')
    for c in heaviest[:2]:
        warn('cell_triangles', c['triangles'], f'{c["triangles"]:,} triangles in the {cell:g} m cell at {c["center_m"][:2]}',
             f'Mostly {", ".join(r["name"] for r in c["top_rows"])}; look with move_camera {c["camera"]}.')
    for t in top_rows[:5]:
        advice = ('triplanar: consolidate_scenery may merge edge-to-edge runs' if t['triplanar'] else
                  'not triplanar: cannot be merged by scaling; thin out or replace with fewer larger pieces')
        warn('row_count', t['count'], f'{t["count"]} x {t["name"]} ({t["row"]})', advice)
    warnings.sort(key=lambda w: (-SEVERITY.index(w['severity']), -float(w['value'] or 0) / max(1e-9, w['high_at'])))
    return {'metrics': metrics, 'per_category': dict(sorted(cats.items(), key=lambda kv: -kv[1])), 'per_kind': dict(kinds),
            'top_rows': top_rows, 'top_rows_by_triangles': top_tris, 'densest_cells': densest, 'heaviest_cells': heaviest, 'engine_instancing': eng,
            'warnings': warnings, 'status': warnings[0]['severity'] if warnings else 'ok',
            'cell_budgets': {k: {'warn': limits[k]['warn'], 'high': limits[k]['high']} for k in ('cell_objects', 'cell_triangles')}}


# ---------------------------------------------------------------------------
# frame time


def parse_frame(line: str) -> dict | None:
    m = dict(re.findall(r'(\w+)=(-?[\d.eE+-]+)', line))
    if not line.startswith('frame ') or 'delta' not in m:
        return None
    return {'delta': float(m['delta']), 'real': float(m['real']), 'frames': int(float(m['frames'])), 'clock': float(m['clock'])}


def frame_stats(samples: list[dict]) -> dict:
    """Percentiles from world delta seconds; average also from the frame counter over the QPC clock when exposed."""
    deltas = sorted(s['delta'] * 1000 for s in samples if s['delta'] > 0)
    counted = None
    if len(samples) >= 2:
        first, last = samples[0], samples[-1]
        frames = last['frames'] - first['frames']
        span, clock = last['clock'] - first['clock'], 'QPC clock'
        if span <= 0:
            span, clock = last['real'] - first['real'], 'world real time'
        if first['frames'] >= 0 and frames > 0 and span > 0:
            counted = {'avg_ms_counted': round(span * 1000 / frames, 2), 'frames_counted': frames, 'clock': clock}
    if not deltas and counted is None:
        raise ValueError('no frame samples (neither world delta nor frame counter available)')
    out = {'samples': len(samples)}
    if deltas:
        p95 = deltas[min(len(deltas) - 1, math.ceil(0.95 * len(deltas)) - 1)]
        out.update(avg_ms=round(sum(deltas) / len(deltas), 2), p50_ms=round(deltas[len(deltas) // 2], 2),
                   p95_ms=round(p95, 2), max_ms=round(deltas[-1], 2), basis='sampled world delta seconds')
    else:
        out.update(avg_ms=counted['avg_ms_counted'], p50_ms=None, p95_ms=None, max_ms=None,
                   basis='frame counter only (world delta unavailable; no percentiles)')
    if counted:
        out.update(counted)
        if deltas:
            out['basis'] = f'frame counter / {counted["clock"]} (avg_ms_counted); world deltas for percentiles'
    return out


def read_cvars(names) -> dict:
    out = {}
    for name in names:
        try:
            for line in agent.run(f'cvar {name}'):
                m = re.search(r'(float|int)=(\S+)', line)
                if m:
                    out[name] = float(m.group(2))
        except agent.AgentError:
            pass
    return out


def frame_notes(stats: dict, cvars: dict) -> list[str]:
    notes, avg = [], stats.get('avg_ms_counted', stats.get('avg_ms'))
    if cvars.get('t.MaxFPS', 0) > 0 and abs(avg - 1000 / cvars['t.MaxFPS']) < 1.0:
        notes.append('frame time sits at the t.MaxFPS cap: real cost is lower than measured')
    if cvars.get('r.VSync', 0) >= 1:
        notes.append('r.VSync is on: frame time may be locked to the display refresh (e.g. 16.7 ms at 60 Hz)')
    if cvars.get('t.IdleWhenNotForeground', 0) >= 1:
        notes.append('t.IdleWhenNotForeground is on: an unfocused game window is throttled; focus it for a real figure')
    return notes


def measure_frames(duration: float = 1.5, poses: list[dict] | None = None, settle: float = 0.5,
                   grace: float = 5.0) -> list[dict]:
    from . import live
    if not 0.3 <= duration <= 10:
        raise ValueError('duration must be 0.3..10 s')
    if poses and len(poses) > 8:
        raise ValueError('at most 8 camera poses')
    results = []
    cvars = read_cvars(FRAME_CVARS)
    moved = False
    try:
        for pose in poses or [None]:
            if pose is not None:
                live.move_camera(float(pose['x']), float(pose['y']), float(pose['z']), float(pose.get('yaw', 0)),
                                 float(pose.get('pitch', -30)), hold=True)
                moved = True
                time.sleep(settle)
            samples, attempts, end = [], 0, time.time() + duration
            while time.time() < end or len(samples) < 3:
                if len(samples) >= 3 and attempts >= 400:
                    break
                if time.time() > end + grace or attempts >= 400:
                    raise RuntimeError(f'frametime gave {len(samples)} parseable samples in {attempts} calls; '
                                       'is the frametime verb in the injected DLL?')
                attempts += 1
                s = next((parse_frame(l) for l in agent.run('frametime', 10) if l.startswith('frame ')), None)
                if s:
                    samples.append(s)
            stats = frame_stats(samples)
            stats.update(pose=pose, cvars=cvars)
            notes = frame_notes(stats, cvars)
            if notes:
                stats['note'] = '; '.join(notes)
            results.append(stats)
    finally:
        if moved:
            live.release_camera()
    return results


# ---------------------------------------------------------------------------
# entry point


def _region(region):
    if region is None:
        return None
    if len(region) != 4 or not all(math.isfinite(float(v)) for v in region):
        raise ValueError('region needs four finite numbers [x0, y0, x1, y1] in metres')
    x0, y0, x1, y1 = (float(v) for v in region)
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _in(region, center) -> bool:
    return region is None or (region[0] <= center[0] <= region[2] and region[1] <= center[1] <= region[3])


def _cached_rowflags() -> dict | None:
    try:
        return parse_rowflags(ROWFLAGS_CACHE.read_text())
    except OSError:
        return None


def check_performance(catalog, name: str = '', region=None, budget: dict | None = None, measure: bool = False,
                      poses: list[dict] | None = None, duration: float = 1.5, cell: float = 50.0, top: int = 10) -> dict:
    if not 1 <= top <= 50:
        raise ValueError('top must be 1..50')
    limits = budgets(budget)
    region = _region(region)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if name:
        if measure:
            raise ValueError('measure needs the live editor; omit name to scan the open map')
        data, path = consolidate.load_saved(name)
        objects = consolidate.objects_from_map(data)
        flags = _cached_rowflags()
        records = [r for r in offline_records(objects, catalog, flags) if _in(region, r['center'])]
        cons = consolidate.plan(objects, catalog, region=region)
        report = analyse(records, catalog, limits, cell, top, live=False, consolidation=cons)
        report.update(source=str(path), mode='offline', row_flags='cached from a live run' if flags else 'unavailable')
    else:
        with agent.session():
            with tempfile.TemporaryDirectory(prefix='uncrashed-perf-', dir='/tmp') as tmp:
                out = Path(tmp) / 'perf.txt'
                box = '' if region is None else ' ' + ' '.join(f'{v * 100:.1f}' for v in region)
                agent.run(f'perf {agent._windows_path(out)}{box}', 180)
                scan = parse_perf(out.read_text())
                rf = Path(tmp) / 'rowflags.txt'
                try:
                    agent.run(f'rowflags {agent._windows_path(rf)}', 60)
                    ROWFLAGS_CACHE.write_text(rf.read_text())
                except (agent.AgentError, OSError):
                    pass
            try:
                from . import groups
                cons = consolidate.plan(groups.snapshot()['objects'], catalog, region=region)
            except Exception:  # consolidation hint is optional
                cons = None
            report = analyse(scan['actors'], catalog, limits, cell, top, lights=scan['lights'],
                             instanced=scan['instanced'], live=True, consolidation=cons,
                             render_cvars=read_cvars(RENDER_CVARS))
            report.update(mode='live', scan=scan['end'])
            if scan['end'].get('truncated'):
                report['warnings'].insert(0, {'severity': 'info', 'metric': 'scan', 'message': 'scan truncated at native caps'})
            if measure:
                frames = measure_frames(duration, poses)
                report['frame_time'] = frames
                for f in frames:
                    for key, metric in (('avg_ms_counted' if 'avg_ms_counted' in f else 'avg_ms', 'frame_ms_avg'),
                                        ('p95_ms', 'frame_ms_p95')):
                        b = limits[metric]
                        sev = grade(f.get(key), b)
                        if sev != 'ok':
                            report['warnings'].insert(0, {'severity': sev, 'metric': metric, 'value': f[key], 'warn_at': b['warn'],
                                                          'high_at': b['high'], 'budget': b['label'], 'pose': f['pose'],
                                                          'message': f'{f[key]} ms {metric}',
                                                          'advice': 'see densest_cells / heaviest_cells for what is in view'
                                                          + (f' ({f["note"]})' if f.get('note') else '')})
    report['status'] = max((w['severity'] for w in report['warnings']), key=SEVERITY.index, default='ok')
    if cons:
        report['consolidation'] = {k: cons[k] for k in ('objects_removed', 'objects_added', 'saved')}
    report['region_m'] = region
    report['budgets'] = {k: {'warn': v['warn'], 'high': v['high']} for k, v in limits.items()}
    report['budgets'].update(report.pop('cell_budgets'))
    report['limitations'] = LIMITATIONS
    path = REPORT_DIR / 'performance-report.json'
    path.write_text(json.dumps(report, indent=1, default=str))
    report['report_json'] = str(path)
    return report
