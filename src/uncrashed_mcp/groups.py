"""Map-scoped group edits, durable snapshots and conflict-checked undo.

Transforms preserve existing actors. Clone/delete undo regenerates catalogue
assets: arbitrary Blueprint/material/animation state is NOT serialized.
"""
from __future__ import annotations

import copy
import json
import math
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path

from . import agent, environment, managed
from .geom import quat_multiply, quat_rotate, quat_rotator, rotator_from_quat

MAX_EDIT = 5000
HISTORY = 'UncrashedMCP.history'
LIMITATIONS = [
    'Move/rotate/uniform-scale preserve actors; clone/delete undo respawns catalogue rows.',
    'Snapshots cover row, group and transform, not arbitrary Blueprint/material/animation state.',
    'Explicit failures attempt rollback; native faults/timeouts may require journal recovery. Never retry blindly.',
    'Undo checks whole affected groups for intervening changes; no silent overwrite.',
]


def parse_snapshot(text: str) -> dict:
    session, objects, end = None, [], None
    for line in text.splitlines():
        p = line.split()
        if p[0] == 'session' and len(p) == 2:
            session = p[1]
        elif p[0] == 'O' and len(p) == 13:
            xf = [float(v) for v in p[4:]]
            if not all(math.isfinite(v) for v in xf) or min(xf[6:]) <= 0:
                raise ValueError('invalid group transform')
            objects.append({'actor': environment.actor_id(p[1]), 'row': p[2], 'group': p[3], 'xf': xf})
        elif p[0] == 'end':
            end = environment.fields(line)
        else:
            raise ValueError('invalid group inventory')
    if not session or end is None or int(end['objects']) != len(objects) or int(end['failed']) or int(end['truncated']):
        raise ValueError('incomplete group inventory; refusing edits')
    if len({o['actor'] for o in objects}) != len(objects):
        raise ValueError('duplicate actors in group inventory')
    return {'session': session, 'objects': objects}


def snapshot() -> dict:
    with tempfile.TemporaryDirectory(prefix='uncrashed-groups-', dir='/tmp') as tmp:
        path = Path(tmp)/'objects.txt'
        agent.run(f'groups {agent._windows_path(path)}', 120)
        return parse_snapshot(path.read_text())


def list_groups(prefix: str = '', limit: int = 50) -> dict:
    if not 1 <= limit <= 200:
        raise ValueError('limit must be 1..200')
    buckets = defaultdict(list)
    for o in snapshot()['objects']:
        if not prefix or o['group'].casefold().startswith(prefix.casefold()):
            buckets[o['group']].append(o)
    groups = []
    for name, members in sorted(buckets.items()):
        lo = [min(o['xf'][i] for o in members)/100 for i in range(3)]
        hi = [max(o['xf'][i] for o in members)/100 for i in range(3)]
        groups.append({'name': name, 'objects': len(members), 'pivot_bounds_m': {'min': lo, 'max': hi}})
    return {'group_count': len(groups), 'groups': groups[:limit], 'truncated': len(groups)>limit}


def select(objects: list[dict], group: str, prefix: bool = False, actors: list[str] | None = None) -> list[dict]:
    if group:
        managed.tag(group)
    if not group and not actors:
        raise ValueError('provide a group or explicit actors; never defaults to whole map')
    if actors is not None:
        if not actors or len(actors) != len(set(actors)) or len(actors)>MAX_EDIT:
            raise ValueError('actors must contain 1..5000 unique handles')
        wanted = {environment.actor_id(a) for a in actors}
    else:
        wanted = None
    result = [o for o in objects if (wanted is None or o['actor'] in wanted) and
              (not group or o['group'].casefold()==group.casefold() or
               prefix and any(o['group'].casefold().startswith(group.casefold()+sep) for sep in '._'))]
    if wanted is not None and {o['actor'] for o in result} != wanted:
        raise ValueError('actor missing, outside group, or not an editable placed object')
    if not result:
        raise ValueError(f'group {group!r} has no matching objects')
    if len(result)>MAX_EDIT:
        raise ValueError('edit limited to 5000 objects; split into subgroups')
    return result


def transform_object(o: dict, pivot_cm: list[float], delta_cm: list[float], rotation: list[float], scale: float) -> dict:
    out = copy.deepcopy(o)
    q = quat_rotator(*rotation)
    relative = [(o['xf'][i]-pivot_cm[i])*scale for i in range(3)]
    rotated = quat_rotate(q, relative)
    out['xf'][:3] = [pivot_cm[i]+rotated[i]+delta_cm[i] for i in range(3)]
    out['xf'][3:6] = rotator_from_quat(quat_multiply(q, quat_rotator(*o['xf'][3:6])))
    out['xf'][6:] = [v*scale for v in o['xf'][6:]]
    if not all(math.isfinite(v) for v in out['xf']) or min(out['xf'][6:])<=0 or max(out['xf'][6:])>10000 or max(map(abs,out['xf'][:3]))>1e8:
        raise ValueError('transformed position/scale out of native range')
    return out


def same_object(a: dict, b: dict) -> bool:
    if a['row'].casefold()!=b['row'].casefold() or a['group'].casefold()!=b['group'].casefold():
        return False
    if any(abs(a['xf'][i]-b['xf'][i])>.2 for i in range(3)) or any(abs(a['xf'][i]-b['xf'][i])>.00002 for i in range(6,9)):
        return False
    qa, qb = quat_rotator(*a['xf'][3:6]), quat_rotator(*b['xf'][3:6])
    return abs(sum(qa[k]*qb[k] for k in qa))>.9999998


def match_states(expected: list[dict], current: list[dict]) -> dict[str, dict]:
    """Multiset match including duplicate geometry; handles may change on reload."""
    if len(expected)!=len(current):
        raise ValueError('undo conflict: affected-group object count changed')
    buckets = defaultdict(list)
    for o in current:
        buckets[(o['row'].casefold(),o['group'].casefold())].append(o)
    matched = {}
    for old in expected:
        bucket = buckets[(old['row'].casefold(),old['group'].casefold())]
        index = next((i for i,o in enumerate(bucket) if o['actor']==old['actor'] and same_object(old,o)), None)
        if index is None:
            index = next((i for i,o in enumerate(bucket) if same_object(old,o)), None)
        if index is None:
            raise ValueError('undo conflict: an affected object moved, changed row/group, or disappeared')
        matched[old['actor']] = bucket.pop(index)
    return matched


def scoped(snap: dict, groups: list[str]) -> dict:
    names = {g.casefold() for g in groups}
    return {'session': snap['session'], 'objects': [o for o in snap['objects'] if o['group'].casefold() in names]}


def _xf(o: dict) -> str:
    return ' '.join(f'{float(v):.9g}' for v in o['xf'])


def edit_line(kind: str, before: dict | None = None, after: dict | None = None) -> str:
    if kind == 'C':
        return f'C {after["row"]} {after["group"]} {_xf(after)}'
    line = f'{kind} {before["actor"]} {before["row"]} {before["group"]} {_xf(before)}'
    if kind == 'T':
        line += f' {after["group"]} {_xf(after)}'
    return line


def _write_journal(path: Path, data: dict) -> None:
    managed.atomic_write(path, json.dumps(data, indent=2, allow_nan=False)+'\n')


def _apply(info: dict, root: Path, before: dict, affected: list[str], commands: list[str], label: str) -> dict:
    if not 1<=len(commands)<=MAX_EDIT:
        raise ValueError('operation must change 1..5000 objects')
    history = root/HISTORY
    if history.exists() and sum(p.stat().st_size for p in history.glob('*.json'))>256*1024*1024:
        raise ValueError('undo history exceeds 256 MiB; archive journals before more edits')
    transaction = uuid.uuid4().hex
    path = history/f'{transaction}.json'
    journal = {'version': 1, 'id': transaction, 'map': info['folder'], 'base': info['base'], 'label': label,
               'created': time.time(), 'status': 'prepared', 'groups': affected, 'before': before,
               'commands': commands, 'limitations': LIMITATIONS}
    _write_journal(path, journal)  # durable BEFORE any mutation
    with tempfile.TemporaryDirectory(prefix='uncrashed-edit-', dir='/tmp') as tmp:
        request, result = Path(tmp)/'request.txt', Path(tmp)/'result.txt'
        request.write_text('session '+before['session']+'\n'+'\n'.join(commands)+'\n')
        try:
            agent.run(f'edit {agent._windows_path(request)} {agent._windows_path(result)}', max(120,len(commands)*.08))
            text = result.read_text()
            if f'end changed={len(commands)} complete=1' not in text.splitlines():
                raise RuntimeError('native edit result incomplete')
            after = scoped(snapshot(), affected)
            journal.update(status='applied', after=after, native_result=text)
            _write_journal(path,journal)
        except Exception as exc:
            journal.update(status='recovery_required', error=str(exc),
                           native_result=result.read_text() if result.exists() else None)
            try:
                journal['after'] = scoped(snapshot(),affected)
                match_states(before['objects'],journal['after']['objects'])
                journal['status'] = 'rolled_back'
            except Exception:
                pass
            _write_journal(path,journal)
            raise RuntimeError(f'edit failed; status={journal["status"]}; journal={path}; inspect before undo/retry: {exc}') from exc
    return {'transaction': transaction, 'journal': str(path), 'changed': len(commands), 'preview': False,
            'undo': 'undo_edit with this transaction id', 'limitations': LIMITATIONS}


def edit_group(group: str = '', action: str = 'transform', delta: list | None = None,
               rotation: list | None = None, scale: float = 1, pivot: list | None = None,
               target_group: str = '', prefix: bool = False, actors: list[str] | None = None,
               preview: bool = True) -> dict:
    if action not in ('transform','clone','delete','assign'):
        raise ValueError('action must be transform, clone, delete or assign')
    delta, rotation = managed.vec([0,0,0] if delta is None else delta,'delta'), managed.vec([0,0,0] if rotation is None else rotation,'rotation')
    if not math.isfinite(scale) or not 0<scale<=1000:
        raise ValueError('uniform scale must be finite and in (0,1000]')
    if action in ('clone','assign'):
        managed.tag(target_group)
    elif target_group:
        raise ValueError('target_group only applies to clone/assign')
    if action in ('delete','assign') and (any(delta) or any(rotation) or scale!=1 or pivot is not None):
        raise ValueError('delete/assign do not accept transform parameters')
    with agent.session():
        info, root = managed.context()
        all_objects = snapshot()
        chosen = select(all_objects['objects'],group,prefix,actors)
        targets = {}
        if action in ('clone','assign'):
            for o in chosen:
                suffix = o['group'][len(group):] if action=='clone' and prefix and group else ''
                targets[o['actor']] = managed.tag(target_group+suffix)
        affected = sorted({o['group'] for o in chosen} | set(targets.values()))
        if action=='clone' and any(o['group'].casefold()==target_group.casefold() or
                                  any(o['group'].casefold().startswith(target_group.casefold()+sep) for sep in '._')
                                  for o in all_objects['objects']):
            raise ValueError('clone target group/branch already exists; choose a fresh name')
        pivot_cm = [v*100 for v in managed.vec(pivot,'pivot')] if pivot is not None else [
            (min(o['xf'][i] for o in chosen)+max(o['xf'][i] for o in chosen))/2 for i in range(3)]
        commands=[]
        for o in chosen:
            if action=='delete':
                commands.append(edit_line('D',o))
                continue
            new = transform_object(o,pivot_cm,[v*100 for v in delta],rotation,scale)
            if action in ('clone','assign'):
                new['group']=targets[o['actor']]
            commands.append(edit_line('C' if action=='clone' else 'T',o,new))
        result = {'action': action, 'objects': len(chosen), 'groups': affected, 'preview': preview,
                  'pivot_m': [v/100 for v in pivot_cm], 'delta_m': delta, 'rotation_pitch_yaw_roll': rotation,
                  'uniform_scale': scale, 'limitations': LIMITATIONS}
        if preview:
            return result
        result.update(_apply(info,root,scoped(all_objects,affected),affected,commands,f'{action} {group or "actors"}'))
        return result


def undo_edit(transaction: str, preview: bool = True) -> dict:
    if not re_transaction(transaction):
        raise ValueError('transaction must be a 32-character lowercase hex id')
    with agent.session():
        info, root = managed.context()
        path = root/HISTORY/f'{transaction}.json'
        journal = json.loads(path.read_text())
        if journal.get('version')!=1 or journal.get('map')!=info['folder'] or journal.get('base')!=info['base']:
            raise ValueError('journal map/base/version mismatch')
        if journal.get('status') not in ('applied','recovery_required') or 'after' not in journal:
            raise ValueError('transaction is not undoable; inspect journal status')
        # Undo only last unsuperseded state, not last entry blindly.
        current = scoped(snapshot(),journal['groups'])
        matched = match_states(journal['after']['objects'],current['objects'])
        before_ids = {o['actor']:o for o in journal['before']['objects']}
        after_ids = {o['actor']:o for o in journal['after']['objects']}
        commands=[]
        for old_id,now in matched.items():
            wanted = before_ids.get(old_id)
            if wanted is None:
                commands.append(edit_line('D',now))
            elif not same_object(now,wanted):
                commands.append(edit_line('T',now,wanted))
        for old_id,wanted in before_ids.items():
            if old_id not in after_ids:
                commands.append(edit_line('C',after=wanted))
        if not commands:
            raise ValueError('transaction already matches original state')
        if preview:
            return {'transaction': transaction, 'preview': True, 'changes': len(commands), 'groups': journal['groups'], 'limitations': LIMITATIONS}
        result = _apply(info,root,current,journal['groups'],commands,f'undo {transaction}')
        journal.update(status='undone',undo_transaction=result['transaction'])
        _write_journal(path,journal)
        return result


def re_transaction(value: str) -> bool:
    return len(value)==32 and all(c in '0123456789abcdef' for c in value)


def edit_history(limit: int = 20) -> dict:
    if not 1<=limit<=100:
        raise ValueError('limit must be 1..100')
    _,root = managed.context()
    paths = sorted((root/HISTORY).glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    entries=[]
    for path in paths[:limit]:
        data=json.loads(path.read_text())
        entries.append({k:data.get(k) for k in ('id','label','status','created','groups','undo_transaction')})
    return {'transactions': entries, 'total': len(paths), 'directory': str(root/HISTORY)}
