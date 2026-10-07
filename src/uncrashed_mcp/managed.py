"""Portable map-scoped metadata, site inventories and persistent foliage zones."""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import uuid
from pathlib import Path

from . import agent, environment
from .mapfile import sanitize_name

STATE_NAME = 'UncrashedMCP.json'
ZONES_NAME = 'UncrashedMCP.zones'
REPORTS = Path('/tmp/uncrashed-managed')
TAG = re.compile(r'^[A-Za-z0-9_.-]{1,96}$')


def tag(value: str) -> str:
    if not TAG.fullmatch(value) or value.casefold() in ('none', 'group', '-', '.'):
        raise ValueError('name must be 1..96 ASCII letters/digits/_.-; None, Group, -, . reserved')
    return value


def vec(value, name: str, *, positive: bool = False) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f'{name} needs three finite numbers')
    result = [float(v) for v in value]
    if not all(math.isfinite(v) and (not positive or v > 0) for v in result):
        raise ValueError(f'{name} needs three finite {"positive " if positive else ""}numbers')
    return result


def directory_path(info: dict) -> Path:
    directory = info.get('directory', '')
    if directory[:2].lower() == 'z:':
        root = Path(directory[2:].replace('\\', '/'))
    elif directory[:2].lower() == 'c:':
        root = agent._wine()[1] / 'drive_c' / directory[3:].replace('\\', '/')
    else:
        raise ValueError('current map directory is unavailable or unsupported')
    return root


def context() -> tuple[dict, Path]:
    info = agent.map_info()
    folder = info.get('folder', '')
    if not folder or sanitize_name(folder) != folder:
        raise ValueError('save current map before using map-scoped editing metadata')
    root = directory_path(info)
    if not root.is_absolute() or root.name != folder or not (root / f'{folder}.json').is_file():
        raise ValueError('save current map before writing metadata (map file missing)')
    return info, root


def read_state(info: dict, root: Path) -> dict:
    path = root / STATE_NAME
    if not path.exists():
        return {'version': 1, 'base': info['base'], 'zones': {}, 'assemblies': {}}
    data = json.loads(path.read_text())
    if data.get('version') != 1 or data.get('base') != info['base']:
        raise ValueError('MCP metadata version/base mismatch; do not apply to another terrain')
    if not isinstance(data.get('zones'), dict) or not isinstance(data.get('assemblies'), dict):
        raise ValueError('invalid MCP metadata')
    return data


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('x') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def write_state(root: Path, state: dict) -> None:
    atomic_write(root / STATE_NAME, json.dumps(state, indent=2, allow_nan=False) + '\n')


def zones_text(state: dict) -> str:
    base = state['base']
    if not re.fullmatch(r'[A-Za-z0-9_]+', base):
        raise ValueError('invalid terrain base name')
    if len(state['zones']) > 128:
        raise ValueError('at most 128 clearing zones per map')
    lines = [f'UNCRASHED_ZONES_V1 {base}']
    for name, z in sorted(state['zones'].items()):
        tag(name)
        center, size = vec(z['center'], 'center'), vec(z['size'], 'size', positive=True)
        yaw = float(z['yaw'])
        if not math.isfinite(yaw) or max(size) > 20000 or max(map(abs, center)) > 1000000:
            raise ValueError('zone dimensions/position out of range')
        values = [v*100 for v in center] + [v*50 for v in size] + [yaw]
        lines.append('Z ' + name + ' ' + ' '.join(f'{v:.9g}' for v in values))
    return '\n'.join(lines) + '\n'


def zone_aabb(zone: dict) -> tuple[list[float], list[float]]:
    c = vec(zone['center'], 'center')
    h = [v/2 for v in vec(zone['size'], 'size', positive=True)]
    co, si = abs(math.cos(math.radians(zone['yaw']))), abs(math.sin(math.radians(zone['yaw'])))
    e = [co*h[0]+si*h[1], si*h[0]+co*h[1], h[2]]
    return [c[i]-e[i] for i in range(3)], [c[i]+e[i] for i in range(3)]


def zone_overlap(zone: dict, lo: list[float], hi: list[float]) -> bool:
    c, h = zone['center'], [v/2 for v in zone['size']]
    co, si = math.cos(math.radians(zone['yaw'])), math.sin(math.radians(zone['yaw']))
    dx, dy = (lo[0]+hi[0])/2-c[0], (lo[1]+hi[1])/2-c[1]
    ex, ey = (hi[0]-lo[0])/2, (hi[1]-lo[1])/2
    return (lo[2] <= c[2]+h[2] and hi[2] >= c[2]-h[2] and
            abs(dx) <= ex+abs(co)*h[0]+abs(si)*h[1] and abs(dy) <= ey+abs(si)*h[0]+abs(co)*h[1] and
            abs(co*dx+si*dy) <= h[0]+abs(co)*ex+abs(si)*ey and
            abs(-si*dx+co*dy) <= h[1]+abs(si)*ex+abs(co)*ey)


def parse_site(text: str) -> dict:
    records, session, end = [], None, None
    for line in text.splitlines():
        p = line.split()
        if p[0] == 'session' and len(p) == 2:
            session = p[1]
        elif p[0] == 'T' and len(p) == 10:
            lo, hi = [float(v)/100 for v in p[4:7]], [float(v)/100 for v in p[7:10]]
            if any(not math.isfinite(v) for v in lo+hi) or any(a > b for a, b in zip(lo, hi)):
                raise ValueError('invalid foliage bounds')
            source = 'base_actor' if int(p[2]) == -2 else 'base_instance'
            records.append({'component': environment.actor_id(p[1]) if source == 'base_instance' else None,
                            'actor': environment.actor_id(p[1]) if source == 'base_actor' else None,
                            'source': source, 'instance': int(p[2]),
                            'masked': bool(int(p[3])), 'aabb': {'min': lo, 'max': hi}})
        elif p[0] == 'end':
            end = environment.fields(line)
        else:
            raise ValueError('invalid site inventory')
    if session is None or end is None or int(end['count']) != len(records):
        raise ValueError('incomplete site inventory')
    return {'session': session, 'foliage': records, 'truncated': bool(int(end['truncated'])),
            'unavailable_sources': int(end.get('unavailable_sources', 0))}


def foliage_inventory(lo: list, hi: list, min_height: float = 2) -> dict:
    with tempfile.TemporaryDirectory(prefix='uncrashed-site-', dir='/tmp') as tmp:
        path = Path(tmp) / 'foliage.txt'
        args = ' '.join(f'{v*100:.7f}' for v in lo+hi+[min_height])
        agent.run(f'site {agent._windows_path(path)} {args}', 120)
        return parse_site(path.read_text())


def terrain_grid(lo, hi, spacing: float) -> dict:
    if not math.isfinite(spacing) or spacing < .5:
        raise ValueError('terrain spacing must be finite and >= 0.5 m')
    intervals = [min(14, max(2, math.ceil((hi[i]-lo[i])/spacing))) for i in range(2)]
    intervals = [v + v % 2 for v in intervals]  # center axes always included
    bounds = environment.get_world_bounds('terrain')
    if bounds.get('truncated'):
        raise RuntimeError('terrain bounds incomplete')
    if not bounds.get('available'):
        return {'available': False, 'samples': [], 'note': 'no landscape; do not assume z=0'}
    bottom, top = bounds['aabb']['min'][2]-10, bounds['aabb']['max'][2]+10
    points = [[lo[0]+(hi[0]-lo[0])*i/intervals[0], lo[1]+(hi[1]-lo[1])*j/intervals[1]]
              for i in range(intervals[0]+1) for j in range(intervals[1]+1)]
    replies, _ = agent.batch([f'terrain {x*100:.5f} {y*100:.5f} {top*100:.5f} {bottom*100:.5f}' for x, y in points])
    samples = []
    for i, (x, y) in enumerate(points):
        rows = replies.get(i, [])
        line = next((line for line in rows if line.startswith('terrain ')), None)
        if line is None or agent._errors(rows):
            raise RuntimeError(f'terrain grid incomplete at sample {i}')
        f = environment.fields(line)
        samples.append({'x': x, 'y': y, 'z': float(f['z'])/100 if 'z' in f else None})
    heights = [s['z'] for s in samples if s['z'] is not None]
    return {'available': True, 'samples': samples, 'missing': len(samples)-len(heights),
            'elevation_range_m': [min(heights), max(heights)] if heights else None,
            'effective_spacing_m': [(hi[i]-lo[i])/intervals[i] for i in range(2)]}


def inspect_site(bounds_min: list, bounds_max: list, spacing: float = 5,
                 min_foliage_height: float = 2, top: int = 30) -> dict:
    lo, hi = vec(bounds_min, 'bounds_min'), vec(bounds_max, 'bounds_max')
    if any(a >= b for a, b in zip(lo, hi)) or not math.isfinite(min_foliage_height) or min_foliage_height < 0:
        raise ValueError('strictly ordered bounds and nonnegative foliage height required')
    if not 1 <= top <= 100:
        raise ValueError('top must be 1..100')
    with agent.session(), tempfile.TemporaryDirectory(prefix='uncrashed-site-', dir='/tmp') as tmp:
        foliage = foliage_inventory(lo, hi, min_foliage_height)
        terrain = terrain_grid(lo, hi, spacing)
        path = Path(tmp)/'placed.txt'
        agent.run(f'inventory {agent._windows_path(path)}', 60)
        placed = [o for o in environment.parse_inventory(path.read_text())
                  if all(o['aabb']['min'][i] <= hi[i] and o['aabb']['max'][i] >= lo[i] for i in range(3))]
        report = {'units': 'metres', 'bounds': {'min': lo, 'max': hi}, 'foliage': foliage,
                  'terrain': terrain, 'placed': placed,
                  'limitations': ['Loaded geometry only; visual foliage AABBs include empty space.',
                                  'Terrain grid is sampled, not exhaustive; rock/base-mesh obstacles are not a full scene inventory.',
                                  'Base flora recognised by plant mesh paths. GPU-only landscape grass sources cannot expose individual transforms; reported unavailable.']}
        REPORTS.mkdir(parents=True, exist_ok=True)
        output = REPORTS/f'site-{uuid.uuid4().hex}.json'
        atomic_write(output, json.dumps(report, allow_nan=False))
        return {'units': 'metres', 'report_json': str(output), 'foliage_count': len(foliage['foliage']),
                'foliage_truncated': foliage['truncated'], 'foliage_unavailable_sources': foliage['unavailable_sources'],
                'foliage': foliage['foliage'][:top],
                'placed_count': len(placed), 'placed': placed[:top], 'terrain': terrain,
                'limitations': report['limitations']}


def clearing_zones(action: str = 'list', name: str = '', center: list | None = None,
                   size: list | None = None, yaw: float = 0, preview: bool = True, allow_partial: bool = False) -> dict:
    if action not in ('list', 'upsert', 'remove'):
        raise ValueError('action must be list, upsert or remove')
    if action != 'list':
        tag(name)
    with agent.session():
        info, root = context()
        state = read_state(info, root)
        if action == 'list':
            return {'zones': state['zones'], 'native': agent.run('zones status'), 'sidecar': str(root/STATE_NAME)}
        old = json.loads(json.dumps(state))
        if action == 'upsert':
            z = {'center': vec(center, 'center'), 'size': vec(size, 'size', positive=True), 'yaw': float(yaw),
                 'allow_partial': allow_partial}
            if not math.isfinite(z['yaw']):
                raise ValueError('yaw must be finite')
            z['yaw'] %= 360
            state['zones'][name] = z
        else:
            if name not in state['zones']:
                raise ValueError(f'unknown clearing zone {name!r}')
            z = state['zones'].pop(name)
        text = zones_text(state)  # validate before any write or live mutation
        lo, hi = zone_aabb(z)
        inv = foliage_inventory(lo, hi, 0)
        if inv['truncated']:
            raise RuntimeError('foliage preview incomplete; refusing to change zones')
        candidates = [t for t in inv['foliage'] if zone_overlap(z, t['aabb']['min'], t['aabb']['max'])]
        result = {'action': action, 'name': name, 'preview': preview, 'zone': z,
                  'candidate_instances': len(candidates), 'currently_masked_candidates': sum(t['masked'] for t in candidates),
                  'unavailable_sources': inv.get('unavailable_sources', 0),
                  'limitations': ['Whole base-foliage instances hidden by conservative visual bounds; not branch trimming.',
                                  'Overlapping zones form a union. Removing one zone may leave instances masked by others.',
                                  'Portable sidecars must accompany map. Resident DLL reapplies after reload; stock game alone does not.',
                                  'GPU-only landscape grass cannot be individually cleared; allow_partial explicitly accepts skipping unavailable sources.']}
        if preview:
            return result
        if action == 'upsert' and inv.get('unavailable_sources', 0) and not allow_partial:
            raise RuntimeError('GPU-only foliage sources unavailable; preview first and pass allow_partial=True to clear supported sources only')
        # Keep recoverable previous config if either disk/runtime step fails.
        write_state(root, state)
        try:
            atomic_write(root/ZONES_NAME, text)
            result['native'] = agent.run('zones refresh', 120)
        except Exception as exc:
            write_state(root, old)
            atomic_write(root/ZONES_NAME, zones_text(old))
            try:
                agent.run('zones refresh', 120)
            except Exception as rollback:
                raise RuntimeError(f'zone update failed: {exc}; restore also failed: {rollback}; originals retained by DLL') from exc
            raise
        result.update(persistent=True, sidecar=str(root/STATE_NAME), runtime_config=str(root/ZONES_NAME))
        return result


def copy_metadata(source: Path, destination: Path, base: str) -> None:
    """Save-as copies portable intent, never actor-handle undo journals."""
    if source == destination or not (source/STATE_NAME).exists():
        return
    if (destination/STATE_NAME).exists() or (destination/ZONES_NAME).exists():
        raise ValueError('save-as target already has MCP metadata; explicit merge required')
    state = read_state({'base': base}, source)
    write_state(destination, state)
    atomic_write(destination/ZONES_NAME, zones_text(state))
