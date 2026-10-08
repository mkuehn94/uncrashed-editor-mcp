"""Read-only live bounds, footprint support and environment diagnostics.

Native facts use centimetres; this module exposes metres. Reports deliberately
separate collision evidence from conservative bounds / canopy candidates.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import tempfile

from . import agent
from .collide import Rows

REPORT_DIR = Path('/tmp/uncrashed-audit')
CORNERS = ('u_min_v_min', 'u_max_v_min', 'u_min_v_max', 'u_max_v_max')
DEFAULT_FOUNDATIONS = {'BUILDING', 'CONTAINER', 'VEHICLE', 'BIGPROP'}
EXCLUDED_ENVIRONMENT = {'PLANT', 'GROUND', 'BALLOON', 'KIT', 'ATTACH', 'SHAPE', 'ROCK'}


def actor_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'(?:0x)?[0-9a-fA-F]{1,16}', value):
        raise ValueError('actor must be a hexadecimal live address')
    return f'{int(value, 16):x}'


def fields(line: str) -> dict[str, str]:
    return dict(token.split('=', 1) for token in line.split()[1:] if '=' in token)


def vector(value: str, *, cm: bool = True) -> list[float]:
    values = [float(v) for v in value.split(',')]
    if len(values) not in (2, 3) or not all(math.isfinite(v) for v in values):
        raise ValueError(f'invalid native vector: {value}')
    return [round(v / 100 if cm else v, 6) for v in values]


def aabb(lo, hi):
    if len(lo) != 3 or len(hi) != 3 or any(a > b for a, b in zip(lo, hi)):
        raise ValueError('invalid native bounds')
    return {'min': lo, 'max': hi,
            'center': [round((a+b)/2, 6) for a, b in zip(lo, hi)],
            'size': [round(b-a, 6) for a, b in zip(lo, hi)]}


def get_object_bounds(actor: str) -> dict:
    actor = actor_id(actor)
    lines = agent.run(f'bounds {actor}')
    line = next((l for l in lines if l.startswith('bounds ')), None)
    if line is None:
        raise RuntimeError('bounds response missing; rebuild/reload the DLL')
    f = fields(line)
    result = {'actor': actor, 'units': 'metres', 'aabb': aabb(vector(f['lo']), vector(f['hi'])), 'obb': None, 'transform': None}
    for line in lines:
        if line.startswith('actor '):
            result['transform'] = agent.parse_actor(line)
        if line.startswith('obb '):
            f = fields(line)
            result['obb'] = {'center': vector(f['center']), 'half_extents': vector(f['half']),
                             'rotation_deg': vector(f['rotation'], cm=False)}
    result['limitations'] = 'AABB encloses owned components, not child actors; OBB available only for a static-mesh root. Bounds are not occupied volume.'
    return result


def get_world_bounds(scope: str = 'placed') -> dict:
    if scope not in ('placed', 'terrain', 'playable'):
        raise ValueError('scope must be placed, terrain or playable')
    lines = agent.run(f'worldbounds {scope}')
    header = next((fields(l) for l in lines if l.startswith('worldbounds ')), None)
    if header is None:
        raise RuntimeError('worldbounds response missing; rebuild/reload the DLL')
    available = header['available'] == '1'
    bounds_line = next((fields(l) for l in lines if l.startswith('aabb ')), None)
    if available and bounds_line is None:
        raise RuntimeError('worldbounds response lacks its AABB')
    return {'scope': scope, 'units': 'metres', 'available': available,
            'actor_count': int(header['actors']), 'truncated': header['truncated'] == '1',
            'aabb': aabb(vector(bounds_line['min']), vector(bounds_line['max'])) if available else None,
            'meaning': {'placed': 'Aggregate bounds of placed editor objects and gates in loaded levels.',
                        'terrain': 'Aggregate loaded LandscapeProxy bounds, not a playable boundary.',
                        'playable': 'Only actors explicitly tagged PlayableBounds; unknown when none are tagged. Aggregate AABB, not the exact boundary shape.'}[scope]}


def validate_region(x0=None, y0=None, x1=None, y1=None):
    values = (x0, y0, x1, y1)
    if all(v is None for v in values):
        return None
    if any(v is None for v in values) or not all(math.isfinite(v) for v in values):
        raise ValueError('region needs four finite coordinates: x0, y0, x1, y1')
    if x0 >= x1 or y0 >= y1:
        raise ValueError('region must have x0 < x1 and y0 < y1')
    return values


def parse_inventory(text: str) -> list[dict]:
    result = []
    for line in text.splitlines():
        p = line.split()
        if len(p) != 9 or p[0] != 'O':
            raise ValueError(f'invalid inventory row: {line}')
        lo = [float(v)/100 for v in p[3:6]]
        hi = [float(v)/100 for v in p[6:9]]
        if not all(math.isfinite(v) for v in lo+hi):
            raise ValueError('non-finite inventory bounds')
        result.append({'actor': actor_id(p[1]), 'row': p[2], 'aabb': aabb(lo, hi)})
    return result


def select_subjects(inventory, rows, mode, actors=None, region=None, include_structural=False):
    if actors is not None:
        if not 1 <= len(actors) <= 256:
            raise ValueError('actors must contain 1 to 256 addresses')
        wanted = list(dict.fromkeys(actor_id(a) for a in actors))
        found = {obj['actor']: obj for obj in inventory}
        missing = [a for a in wanted if a not in found]
        if missing:
            raise ValueError(f'actors are not current placed objects: {", ".join(missing)}')
        candidates = [found[a] for a in wanted]
    else:
        candidates = []
        for obj in inventory:
            category = rows.cat(obj['row'])
            if include_structural:
                keep = category not in {'PLANT', 'GROUND', 'BALLOON'}
            else:
                keep = category in DEFAULT_FOUNDATIONS if mode == 'foundation' else category not in EXCLUDED_ENVIRONMENT
            if keep:
                candidates.append(obj)
    if region:
        x0, y0, x1, y1 = region
        candidates = [obj for obj in candidates if obj['aabb']['max'][0] >= x0 and obj['aabb']['min'][0] <= x1
                      and obj['aabb']['max'][1] >= y0 and obj['aabb']['min'][1] <= y1]
    return candidates


def parse_audit(text: str, mode: str) -> dict:
    records, errors, current, summary = [], [], None, None
    for line in text.splitlines():
        p = line.split()
        if not p:
            continue
        kind, f = p[0], fields(line)
        if kind == 'actor':
            if current is not None:
                raise ValueError('unterminated actor audit')
            current = {'actor': actor_id(f['id']), 'row': f['row'], 'bounds_basis': f['basis'],
                       'aabb': aabb(vector(f['min']), vector(f['max'])),
                       'bounds_rotation_deg': vector(f['rotation'], cm=False), 'footprint_basis': f.get('footprint', 'unknown'), 'points': [], 'tree_collisions': [], 'canopy_candidates': []}
        elif kind == 'error':
            errors.append({'actor': actor_id(f['actor']), 'reason': f['reason']})
        elif kind == 'scan':
            summary = {k: int(v) for k, v in f.items()}
        elif kind == 'end':
            if current is None or actor_id(p[1]) != current['actor'] or mode not in current:
                raise ValueError('incomplete actor audit')
            records.append(current)
            current = None
        elif current is None:
            raise ValueError(f'unexpected audit row: {line}')
        elif kind == 'foundation':
            total = int(f['samples'])
            current['foundation'] = {k: int(f[k]) for k in ('samples', 'supported', 'unsupported', 'penetrating', 'missing')}
            current['foundation'].update(support_coverage=round(int(f['supported'])/total, 4) if total else None,
                max_known_gap_m=float(f['max_gap'])/100 if float(f['max_gap']) >= 0 else None,
                max_penetration_m=float(f['max_penetration'])/100,
                unsupported_corners=[corner for i, corner in enumerate(CORNERS) if int(f['corner_mask']) & (1 << i)],
                effective_spacing_m=vector(f['spacing']))
        elif kind == 'environment':
            current['environment'] = {k: int(f[k]) for k in ('terrain_samples', 'collision_samples', 'penetration_samples', 'tree_hits', 'canopy_candidates')}
            current['environment'].update(terrain_available=f['terrain_available'] == '1', terrain_basis=f['terrain_basis'],
                max_terrain_penetration_m=float(f['max_depth'])/100, effective_spacing_m=vector(f['spacing']))
        elif kind in ('support_point', 'terrain_point'):
            point = {k: round(float(f[k])/100, 6) for k in ('x', 'y', 'bottom')}
            if kind == 'support_point':
                point.update(gap_m=float(f['gap'])/100 if float(f['gap']) > -1e29 else None, surface=f['surface'])
            else:
                point.update(terrain_z=float(f['terrain'])/100, penetration_m=float(f['depth'])/100)
            current['points'].append(point)
        elif kind == 'tree_collision':
            current['tree_collisions'].append({'actor': f['actor'], 'component': f['component'], 'instance': int(f['instance']),
                                               'source': f.get('source', 'unknown')})
        elif kind == 'canopy':
            current['canopy_candidates'].append({'actor': f.get('actor'), 'component': f['component'], 'instance': int(f['instance']),
                                                'source': f.get('source', 'unknown'), 'aabb': aabb(vector(f['min']), vector(f['max']))})
        else:
            raise ValueError(f'unknown audit row: {line}')
    if current is not None or summary is None or summary['checked'] != len(records) or summary['failed'] != len(errors):
        raise ValueError('incomplete audit report')
    return {'records': records, 'errors': errors, 'scan': summary}


def assess(record, mode):
    issues = []
    if mode == 'foundation':
        f = record['foundation']
        if f['unsupported_corners'] or f['unsupported']:
            issues.append('incomplete_footprint_support')
        if f['penetrating']:
            issues.append('base_plane_penetration')
        rotation = record.get('bounds_rotation_deg', [0, 0, 0])
        if abs(rotation[0]) > 5 or abs(rotation[2]) > 5:
            issues.append('tilted_footprint_approximation')
        unknown = f['samples'] == 0
        score = (1-(f['support_coverage'] or 0))*100 + f['max_penetration_m']
    else:
        f = record['environment']
        if f['penetration_samples']:
            issues.append('terrain_penetration')
        if f['tree_hits']:
            issues.append('foliage_collision_candidate')
        if f['canopy_candidates']:
            issues.append('canopy_bounds_overlap_warning')
        unknown = f['terrain_available'] and f['terrain_samples'] == 0
        score = f['max_terrain_penetration_m']*100 + f['tree_hits']*10 + f['canopy_candidates']
    record.update(status='review' if issues else 'unknown' if unknown else 'no_findings', issues=issues)
    return score


def audit(catalog, mode: str, actors=None, region=None, spacing=.75, tolerance=.3,
          max_drop=50., include_canopies=True, include_structural=False, top=15) -> dict:
    if mode not in ('foundation', 'environment'):
        raise ValueError('invalid audit mode')
    if not all(math.isfinite(v) for v in (spacing, tolerance, max_drop)) or not .05 <= spacing <= 20 or tolerance < 0 or not tolerance < max_drop <= 10000:
        raise ValueError('spacing must be 0.05..20 m; 0 <= tolerance < max_drop <= 10000 m')
    if not 1 <= top <= 60:
        raise ValueError('top must be 1..60')
    # Validate caller-controlled addresses before any game command.
    if actors is not None:
        if not 1 <= len(actors) <= 256:
            raise ValueError('actors must contain 1 to 256 addresses')
        actors = [actor_id(a) for a in actors]
    if region is not None:
        if len(region) != 4:
            raise ValueError('region needs four coordinates')
        region = validate_region(*region)
    rows = Rows.from_catalog(catalog)
    all_records, errors, scans = [], [], []
    with tempfile.TemporaryDirectory(prefix='uncrashed-audit-', dir='/tmp') as tmp:
        tmp = Path(tmp)
        inv = tmp/'inventory.txt'
        agent.run(f'inventory {agent._windows_path(inv)}', 60)
        inventory = parse_inventory(inv.read_text())
        subjects = select_subjects(inventory, rows, mode, actors, region, include_structural)
        plant_lines = [f'P {obj["actor"]}' for obj in inventory if rows.cat(obj['row']) == 'PLANT']
        for start in range(0, len(subjects), 128):
            chunk = subjects[start:start+128]
            request, output = tmp/'request.txt', tmp/'report.txt'
            request.write_text('\n'.join(plant_lines + [f'A {obj["actor"]}' for obj in chunk])+'\n')
            agent.run(f'audit {agent._windows_path(request)} {agent._windows_path(output)} {mode} '
                      f'{spacing*100:.3f} {tolerance*100:.3f} {max_drop*100:.3f} {int(include_canopies)}', 120)
            parsed = parse_audit(output.read_text(), mode)
            expected = {obj['actor'] for obj in chunk}
            actual = [r['actor'] for r in parsed['records']] + [r['actor'] for r in parsed['errors']]
            if len(actual) != len(expected) or set(actual) != expected:
                raise RuntimeError('audit response does not match requested actors')
            all_records.extend(parsed['records']); errors.extend(parsed['errors']); scans.append(parsed['scan'])
    ranked = []
    for record in all_records:
        record['name'] = rows.name(record['row'])
        score = assess(record, mode)
        if record['status'] != 'no_findings':
            ranked.append((score, record))
    ranked.sort(key=lambda item: -item[0])
    limitations = [
        'Read-only heuristic diagnostics, not structural certification or a flight test.',
        'Foundation coverage uses the bounds base plane: eaves, arches, stilts and intentional cantilevers need manual review.',
        'Physics foliage findings test a subject OBB against base/placed-foliage collision, not exact mesh-versus-mesh intersection.',
        'Canopy warnings compare visual AABBs of loaded base instances and placed plants taller than 2 m; sparse crowns may not intersect actual geometry.',
        'Only loaded levels/instances are measured; detail lists capped at 12 per actor. Grid intervals capped at 32 per axis; effective spacing reported.',
        'Base flora recognised by plant mesh paths. GPU-only landscape grass cannot expose instance transforms; unavailable-source counts make canopy coverage incomplete.',
    ]
    if scans and not any(s['terrain_available'] for s in scans):
        limitations.append('No loaded landscape: terrain penetration not evaluated. Use foundation support checks for flat base geometry.')
    truncated = any(s['foliage_truncated'] or s['query_truncated'] or s.get('foliage_unavailable_sources', 0) for s in scans)
    counts = {}
    for record in all_records:
        for issue in record['issues']:
            counts[issue] = counts.get(issue, 0)+1
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR/f'{mode}-report.json'
    report = {'mode': mode, 'units': 'metres', 'selected': len(subjects), 'checked': len(all_records), 'errors': errors,
              'findings': counts, 'objects_with_findings': sum(bool(r['issues']) for r in all_records),
              'unknown_objects': sum(r['status'] == 'unknown' for r in all_records),
              'sampling_complete': not truncated and not errors and all(r['status'] != 'unknown' for r in all_records),
              'truncated': truncated, 'scans': scans, 'parameters': {'region': region, 'spacing_m': spacing,
              'tolerance_m': tolerance, 'max_drop_m': max_drop, 'include_canopies': include_canopies,
              'include_structural': include_structural}, 'limitations': limitations, 'records': all_records}
    path.write_text(json.dumps(report, indent=2))
    brief = []
    for _, record in ranked[:top]:
        item = {k: record[k] for k in ('actor', 'row', 'name', 'status', 'issues', mode)}
        box = record['aabb']
        item['at_m'] = [box['center'][0], box['center'][1], box['min'][2]]
        item['bounds_basis'] = record['bounds_basis']
        item['footprint_basis'] = record['footprint_basis']
        item['examples'] = record['points'][:3]
        if mode == 'environment':
            item['tree_collisions'] = record['tree_collisions'][:3]
            item['canopy_candidates'] = [{k: v for k, v in c.items() if k != 'aabb'} for c in record['canopy_candidates'][:3]]
        brief.append(item)
    return {k: v for k, v in report.items() if k not in ('records', 'scans')} | {
        'top': brief, 'report_json': str(path)}
