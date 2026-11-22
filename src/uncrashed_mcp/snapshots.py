"""Stamped scene snapshots of the open editor and read-only diffs against them.

A snapshot is the native `groups` export (row, group, transform of every placed editor object, with the
scene session token) plus the race gates from `gatebox` (level order, transform, Start flag), written as
gzip JSON beside the map's UncrashedMCP sidecars. Nothing here changes the scene.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from . import agent, clearance, environment, groups, managed
from .geom import quat_rotator

DIRECTORY = 'UncrashedMCP.snapshots'
SUFFIX = '.json.gz'
VERSION = 1
NAME = re.compile(r'^[A-Za-z0-9_.-]{1,64}$')
LIMITATIONS = [
    'Snapshots hold row, group and transform of placed editor objects and gate transforms; not Blueprint, '
    'material, collision or weather state.',
    'Gate order is the live level order (gatebox), not proven to be the saved race order; spawn height is not read.',
    'Box and group scopes test pivots; an object that left the scope is matched against the whole scene.',
    'A diff is geometry bookkeeping: it does not show visual quality, connectivity or that a line is flyable.',
]


def _hash(lines) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode())
        digest.update(b'\n')
    return digest.hexdigest()[:16]


def object_key(o: dict) -> str:
    """Rounded state of one object without its session-local actor id (1 mm, 0.01 deg, 1e-5 scale)."""
    xf = o['xf']
    return '|'.join([o['row'].casefold(), o['group'].casefold(), *(f'{v:.1f}' for v in xf[:3]),
                     *(f'{_angle(v):.2f}' for v in xf[3:6]), *(f'{v:.5f}' for v in xf[6:])])


def _angle(value: float) -> float:
    value = (value + 180) % 360 - 180
    return 0.0 if abs(value) < .005 or abs(abs(value) - 180) < .005 and value < 0 else value


def fingerprint(objects: list[dict]) -> str:
    return _hash(sorted(object_key(o) for o in objects))


def gate_key(g: dict) -> str:
    return '|'.join([str(g.get('row')), '1' if g.get('start') else '0', *(f'{v:.3f}' for v in g['pivot_m']),
                     *(f'{_angle(v):.2f}' for v in g['rotation']), *(f'{v:.4f}' for v in g['scale'])])


def gate_fingerprint(gates: list[dict]) -> str:
    return _hash(gate_key(g) for g in gates)  # order matters


def _gates(raw: list[dict]) -> list[dict]:
    return [{'actor': g['actor'], 'row': g.get('row'), 'start': bool(g.get('start')), 'pivot_m': list(g['pivot']),
             'rotation': [float(v) for v in g['rotation']], 'scale': [float(v) for v in g['scale']]} for g in raw]


def start_gate(gates: list[dict]) -> dict | None:
    for i, g in enumerate(gates):
        if g['start']:
            return {'index': i, 'row': g['row'], 'pivot_m': g['pivot_m']}
    return None


def scope(group: str = '', box: list[float] | None = None) -> dict:
    """Validated scope: group prefix (case-insensitive, matches the group and group.* / group_* children) and/or
    an x0,y0,x1,y1 box in metres tested on pivots."""
    if group and not re.fullmatch(r'[A-Za-z0-9_.-]{1,96}', group):
        raise ValueError('group must be 1..96 of letters, digits, _ . -')
    if box is not None:
        if len(box) != 4:
            raise ValueError('box needs four finite coordinates: x0, y0, x1, y1 (metres)')
        box = list(environment.validate_region(*map(float, box)))
    return {'group': group or None, 'box_m': box}


def in_scope(o: dict, sc: dict) -> bool:
    group, box = sc.get('group'), sc.get('box_m')
    if group:
        g, want = o['group'].casefold(), group.casefold()
        if g != want and not any(g.startswith(want + sep) for sep in '._'):
            return False
    if box:
        x, y = o['xf'][0] / 100, o['xf'][1] / 100
        if not (box[0] <= x <= box[2] and box[1] <= y <= box[3]):
            return False
    return True


def _scope_label(sc: dict) -> str:
    parts = []
    if sc.get('group'):
        parts.append(f"group {sc['group']}")
    if sc.get('box_m'):
        parts.append('box ' + ','.join(f'{v:g}' for v in sc['box_m']))
    return ' + '.join(parts) or 'whole scene'


def _read_scene() -> tuple[dict, Path, dict, list[dict]]:
    """Map identity, map folder, groups export and gates read under one agent session."""
    info, root = managed.context()
    snap = groups.snapshot()
    with tempfile.TemporaryDirectory(prefix='uncrashed-snapshot-', dir='/tmp') as tmp:
        gates = _gates(clearance.live_gates(Path(tmp), clearance.CHANNELS['Pawn'])['gates'])
    after = agent.map_info()
    if (after.get('folder'), after.get('base')) != (info.get('folder'), info.get('base')):
        raise RuntimeError('the open map changed while reading the scene; nothing was compared or written')
    return info, root, snap, gates


def _write_gz(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    with gzip.open(tmp, 'wt', compresslevel=3) as f:
        json.dump(data, f, separators=(',', ':'), allow_nan=False)
    os.replace(tmp, path)


def _warnings(info: dict) -> list[str]:
    if info.get('folder') == 'Autosave':
        return ['the open map is the Autosave folder: save it under a name before relying on this snapshot']
    return []


def scene_snapshot(group: str = '', box: list[float] | None = None, name: str = '') -> dict:
    sc = scope(group, box)
    name = name or time.strftime('snap-%Y%m%d-%H%M%S')
    if not NAME.match(name):
        raise ValueError('name must be 1..64 of letters, digits, _ . -')
    with agent.session():
        info, root, snap, gates = _read_scene()
        path = root / DIRECTORY / f'{name}{SUFFIX}'
        if path.exists():
            raise ValueError(f'snapshot {name!r} already exists; choose another name')
        chosen = [o for o in snap['objects'] if in_scope(o, sc)]
        stamp = {
            'name': name, 'map': info['folder'], 'base': info.get('base'), 'session': snap['session'],
            'objects': len(snap['objects']), 'scope': _scope_label(sc), 'scope_objects': len(chosen),
            'fingerprint': fingerprint(chosen), 'gates': len(gates), 'gate_fingerprint': gate_fingerprint(gates),
            'start_gate': start_gate(gates), 'file': str(path),
        }
        data = {'version': VERSION, 'kind': 'scene_snapshot', 'created': time.time(), 'stamp': stamp,
                'scope': sc, 'units': 'objects xf: centimetres, degrees (pitch,yaw,roll), scale; gates: metres',
                'objects': chosen, 'gates': gates, 'limitations': LIMITATIONS,
                # every actor id present at snapshot time: a scoped diff must not match a deleted in-scope
                # object to an object that already existed outside the scope
                'all_actors': sorted(o['actor'] for o in snap['objects'])}
        _write_gz(path, data)
    return {**stamp, 'warnings': _warnings(info), 'limitations': LIMITATIONS}


def load_snapshot(name_or_file: str, root: Path) -> tuple[dict, Path]:
    if not name_or_file:
        raise ValueError('give a snapshot name or file')
    bare = '/' not in name_or_file and os.sep not in name_or_file and not name_or_file.startswith('~')
    if bare and NAME.match(name_or_file):  # a name, or 'name.json.gz': the map's snapshot folder first
        stem = name_or_file[:-len(SUFFIX)] if name_or_file.endswith(SUFFIX) else name_or_file
        path = root / DIRECTORY / f'{stem}{SUFFIX}'
        if not path.is_file() and name_or_file.endswith(SUFFIX) and Path(name_or_file).is_file():
            path = Path(name_or_file)
    else:
        path = Path(name_or_file).expanduser()
    if not path.is_file():
        raise ValueError(f'no snapshot {name_or_file!r} (looked for {path})')
    try:
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(f'snapshot {path} is unreadable: {exc}') from None
    stamp = data.get('stamp') if isinstance(data, dict) else None
    if data.get('version') != VERSION or data.get('kind') != 'scene_snapshot' or not isinstance(stamp, dict) or \
            not isinstance(data.get('objects'), list) or not isinstance(data.get('gates'), list):
        raise ValueError(f'{path} is not a version {VERSION} scene snapshot; refusing to compare')
    if len(data['objects']) != stamp.get('scope_objects') or fingerprint(data['objects']) != stamp.get('fingerprint') \
            or len(data['gates']) != stamp.get('gates') or gate_fingerprint(data['gates']) != stamp.get('gate_fingerprint'):
        raise ValueError(f'snapshot {path} contents do not match its stamp (edited or mixed); refusing to compare')
    return data, path


def turn_deg(a: list[float], b: list[float]) -> float:
    qa, qb = quat_rotator(*a), quat_rotator(*b)
    dot = min(1.0, abs(sum(qa[k] * qb[k] for k in qa)))
    return math.degrees(2 * math.acos(dot))


def _compare(old: dict, now: dict, tol: dict) -> dict:
    d = [(now['xf'][i] - old['xf'][i]) / 100 for i in range(3)]
    turn = turn_deg(old['xf'][3:6], now['xf'][3:6])
    return {'moved': math.dist(old['xf'][:3], now['xf'][:3]) / 100 > tol['position_m'],
            'rotated': turn > tol['rotation_deg'],
            'scaled': any(abs(now['xf'][i] - old['xf'][i]) > tol['scale'] for i in range(6, 9)),
            'regrouped': old['group'].casefold() != now['group'].casefold(),
            'row_changed': old['row'].casefold() != now['row'].casefold(),
            'delta_m': [round(v, 4) for v in d], 'turn_deg': round(turn, 3)}


def match(old: list[dict], current: list[dict], by_actor: bool, tol: dict,
          exclude: set | frozenset = frozenset()) -> tuple[list, list, list, dict]:
    """Pairs (old, now), unmatched old, unmatched current, and how many matched by actor id / by row+position.
    Same session: actor id first. Then (any session) the nearest unused current object of the same row within
    the position tolerance, preferring the same group and the same rotation (multiset; duplicates pair up).
    exclude: current actor ids that may not be position candidates (objects that already existed out of scope)."""
    pairs, used, counts = [], set(), {'actor': 0, 'position': 0}
    rest = old
    if by_actor:
        ids = {o['actor']: o for o in current}
        rest = []
        for o in old:
            now = ids.get(o['actor'])
            if now is not None and now['actor'] not in used:
                pairs.append((o, now))
                used.add(now['actor'])
                counts['actor'] += 1
            else:
                rest.append(o)
    cell = max(tol['position_m'] * 100, 1.0)
    grid = defaultdict(list)
    for o in current:
        if o['actor'] not in used and o['actor'] not in exclude:
            grid[(o['row'].casefold(), *(math.floor(v / cell) for v in o['xf'][:3]))].append(o)
    unmatched_old = []
    for o in rest:
        row, c = o['row'].casefold(), [math.floor(v / cell) for v in o['xf'][:3]]
        best, best_score = None, None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for cand in grid.get((row, c[0] + dx, c[1] + dy, c[2] + dz), ()):
                        if cand['actor'] in used:
                            continue
                        dist = math.dist(o['xf'][:3], cand['xf'][:3]) / 100
                        if dist > tol['position_m']:
                            continue
                        score = (cand['group'].casefold() != o['group'].casefold(), not groups.same_object(o, cand), dist)
                        if best_score is None or score < best_score:
                            best, best_score = cand, score
        if best is None:
            unmatched_old.append(o)
        else:
            used.add(best['actor'])
            pairs.append((o, best))
            counts['position'] += 1
    return pairs, unmatched_old, [o for o in current if o['actor'] not in used], counts


def _example(change: str, o: dict, now: dict | None = None, cmp: dict | None = None) -> dict:
    return {'change': change, 'actor': (now or o)['actor'], 'row': o['row'], 'group': o['group'],
            'new_group': now['group'] if now is not None and cmp and cmp['regrouped'] else None,
            'pivot_m': [round(v / 100, 3) for v in o['xf'][:3]],
            'delta_m': cmp['delta_m'] if cmp and cmp['moved'] else None,
            'turn_deg': cmp['turn_deg'] if cmp and cmp['rotated'] else None}


def compare_gates(old: list[dict], now: list[dict], same_session: bool, tol: dict) -> dict:
    """Order, transform and start-gate verdicts between two level-ordered gate lists."""
    order = len(old) == len(now) and all(
        (a['actor'] == b['actor']) if same_session else (a['row'] == b['row'] and a['start'] == b['start'] and
                                                         math.dist(a['pivot_m'], b['pivot_m']) <= tol['gate_match_m'])
        for a, b in zip(old, now))
    changed = []
    for i, (a, b) in enumerate(zip(old, now)):
        if math.dist(a['pivot_m'], b['pivot_m']) > tol['gate_position_m'] or \
                turn_deg(a['rotation'], b['rotation']) > tol['rotation_deg'] or \
                any(abs(x - y) > tol['scale'] for x, y in zip(a['scale'], b['scale'])):
            changed.append(i)
    transforms = len(old) == len(now) and not changed
    s_old, s_now = start_gate(old), start_gate(now)
    start = s_old is not None and s_now is not None and s_old['index'] == s_now['index'] and \
        s_old['row'] == s_now['row'] and s_now['index'] not in changed
    return {'before': len(old), 'now': len(now), 'order_unchanged': order, 'transforms_unchanged': transforms,
            'changed_indices': changed[:20], 'start_unchanged': start, 'start_before': s_old, 'start_now': s_now}


def scene_diff(name_or_file: str, group: str | None = None, box: list[float] | None = None,
               position_m: float = .01, rotation_deg: float = .1, scale: float = .001, gate_position_m: float = .01,
               examples: int = 10) -> dict:
    for label, value, hi in (('position_m', position_m, 100), ('rotation_deg', rotation_deg, 180),
                             ('scale', scale, 100), ('gate_position_m', gate_position_m, 100)):
        if not math.isfinite(value) or not 0 <= value <= hi:
            raise ValueError(f'{label} must be 0..{hi}')
    if not 0 <= examples <= 100:
        raise ValueError('examples must be 0..100')
    override = scope(group or '', box) if group is not None or box is not None else None
    tol = {'position_m': position_m, 'rotation_deg': rotation_deg, 'scale': scale,
           'gate_position_m': gate_position_m, 'gate_match_m': max(gate_position_m, .5)}
    with agent.session():
        info, root = managed.context()
        data, path = load_snapshot(name_or_file, root)
        stamp = data['stamp']
        if stamp.get('map') != info.get('folder') or stamp.get('base') != info.get('base'):
            raise ValueError(f"snapshot is of map {stamp.get('map')} on base {stamp.get('base')}, the open map is "
                             f"{info.get('folder')} on base {info.get('base')}; refusing to compare different scenes")
        info, root, snap, gates = _read_scene()
    sc = override or data['scope']
    old = [o for o in data['objects'] if in_scope(o, sc)]
    if override and override != data['scope'] and data['scope'] != {'group': None, 'box_m': None}:
        raise ValueError(f"snapshot covers only {_scope_label(data['scope'])}; compare with that scope "
                         'or take a whole-scene snapshot')
    same_session = stamp['session'] == snap['session']
    exclude = frozenset()
    if same_session:  # objects present at snapshot time but outside the compared scope are not candidates
        before = set(data.get('all_actors') or (o['actor'] for o in data['objects']))
        exclude = frozenset(before - {o['actor'] for o in old})
    pairs, removed, rest, counts = match(old, snap['objects'], same_session, tol, exclude)
    added = [o for o in rest if in_scope(o, sc)]
    totals = dict.fromkeys(('moved', 'rotated', 'scaled', 'regrouped', 'row_changed'), 0)
    changed, rows = 0, []
    for o, now in pairs:
        cmp = _compare(o, now, tol)
        flags = [k for k in totals if cmp[k]]
        for k in flags:
            totals[k] += 1
        if flags:
            changed += 1
            rows.append(_example('+'.join(flags), o, now, cmp))
    full = rows + [_example('removed', o) for o in removed] + [_example('added', o) for o in added]
    gate = compare_gates(data['gates'], gates, same_session, tol)
    warnings = _warnings(info)
    if not same_session:
        warnings.append('session differs (actor ids are session-local): objects matched by row + position only; '
                        f'an object moved more than {position_m} m counts as removed + added')
    verdicts = [
        f"identity: same map folder {info['folder']} and base; "
        + ('same session, matched by actor id first' if same_session else 'different session, matched by row+position'),
        f"gate order: {'unchanged' if gate['order_unchanged'] else 'CHANGED'} ({gate['before']} -> {gate['now']} gates)",
        f"gate transforms: {'unchanged within' if gate['transforms_unchanged'] else 'CHANGED beyond'} {gate_position_m} m / "
        f"{rotation_deg} deg / {scale} scale",
        f"start gate: {'unchanged' if gate['start_unchanged'] else 'CHANGED or missing'}",
        f"objects ({_scope_label(sc)}): {changed} changed, {len(added)} added, {len(removed)} removed",
    ]
    report_dir = root / DIRECTORY / 'diffs'
    report = report_dir / f"{stamp['name']}-{time.strftime('%Y%m%d-%H%M%S')}{SUFFIX}"
    unchanged = len(pairs) - changed
    result = {
        'snapshot': stamp['name'], 'map': info['folder'], 'same_session': same_session, 'scope': _scope_label(sc),
        'objects_before': len(old), 'objects_now': sum(1 for o in snap['objects'] if in_scope(o, sc)),
        'unchanged': unchanged, 'changed': changed, 'added': len(added), 'removed': len(removed), **totals,
        'matched_by': counts,
        'tolerance': {'position_m': position_m, 'rotation_deg': rotation_deg, 'scale': scale,
                      'gate_position_m': gate_position_m},
        'gate_order_unchanged': gate['order_unchanged'], 'gate_transforms_unchanged': gate['transforms_unchanged'],
        'start_gate_unchanged': gate['start_unchanged'], 'gate_changed_indices': gate['changed_indices'],
        'verdicts': verdicts, 'examples': full[:examples], 'examples_truncated': len(full) > examples,
        'diff_file': str(report), 'warnings': warnings, 'limitations': LIMITATIONS,
    }
    _write_gz(report, {'version': VERSION, 'kind': 'scene_diff', 'created': time.time(), 'snapshot_file': str(path),
                       'session_now': snap['session'], 'summary': {k: v for k, v in result.items() if k != 'examples'},
                       'changes': full, 'gates': gate})
    return result
