"""Collision / physics toggles for placed scenery (native/collision.h) and stock-safe
non-colliding catalogue rows.

Tool units are metres; the native verb speaks centimetres. Live toggles exist only
while the injected DLL is resident: the stock game ignores UncrashedMCP.collision,
so a shared map collides normally for everyone else. Catalogue rows whose
DT_MapEditorAssets 'Collision?' flag is false (decals, neons, ...) are the only
stock-reliable non-colliding scenery.
"""
from __future__ import annotations

import json
import math
import re
import tempfile
import time
import uuid
from pathlib import Path

from . import agent, environment, managed

MODES = ('on', 'nophysics', 'flythrough', 'off')
PHYSICS_MODES = {'physics_off': 'physics off', 'physics_on': 'physics on'}  # live-only toggles
CHANNELS = {'worldstatic': 0, 'worlddynamic': 1, 'pawn': 2, 'visibility': 3, 'camera': 4,
            'physicsbody': 5, 'vehicle': 6, 'destructible': 7}
CHANNELS.update({f'engine{i}': 7+i for i in range(1, 7)})
CHANNELS.update({f'game{i}': 13+i for i in range(1, 19)})
NAMES = {v: k for k, v in CHANNELS.items()}
DEFAULT_CHANNELS = ('pawn', 'physicsbody', 'vehicle')
TOLERANCE_CM = 2.0
MAX_RULES = 20000
MAX_LIST = 8192
HEADER = 'UNCRASHED_COLLISION_V1'
KIND = {'object': 'O', 'group': 'G', 'row': 'R'}
ROW = re.compile(r'^NewRow[A-Za-z0-9_.-]{0,90}$')
WARNINGS = [
    'Live-only: the stock game without the resident DLL ignores UncrashedMCP.collision; everyone else (and a '
    'game started without the agent) gets normal collision. Use non-colliding catalogue rows for stock-reliable scenery.',
    'Which channel the drone / vehicles collide on is not verified: default Pawn+PhysicsBody+Vehicle. Run agent '
    '"collision channels" while flying and pass channels=[...] if the drone uses another object channel.',
    'Object rules match catalogue row + pivot within 2 cm: moving an object detaches it from its rule '
    '(collision_state reports unmatched rules). Group rules survive moves.',
    'flythrough keeps query collision (object type, Visibility, Camera), so traces, audits and editor picking still '
    'see the object; off removes it from every trace too.',
]


# ---------------------------------------------------------------------------
# pure helpers (unit tested)
# ---------------------------------------------------------------------------

def channel_ids(channels) -> list[int]:
    if channels is None:
        channels = DEFAULT_CHANNELS
    if isinstance(channels, (str, int)):
        channels = [channels]
    ids = []
    for c in channels:
        if isinstance(c, bool):
            raise ValueError('channel must be a name or 0..31')
        if isinstance(c, int):
            value = c
        elif isinstance(c, str) and c.strip().isdigit():
            value = int(c)
        elif isinstance(c, str) and c.strip().casefold() in CHANNELS:
            value = CHANNELS[c.strip().casefold()]
        else:
            raise ValueError(f'unknown collision channel {c!r}; use {", ".join(list(CHANNELS)[:8])}, engineN, gameN or 0..31')
        if not 0 <= value <= 31:
            raise ValueError('collision channel must be 0..31')
        ids.append(value)
    if not ids:
        raise ValueError('at least one channel required')
    return sorted(set(ids))


def mask_of(ids) -> int:
    return sum(1 << i for i in set(ids))


def channel_names(mask: int) -> list[str]:
    return [NAMES.get(i, str(i)) for i in range(32) if mask >> i & 1]


def mode_name(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f'mode must be one of {", ".join(MODES)}')
    return mode


def make_rule(kind: str, key: str, mode: str, channels=None, force: bool = False, keep_body: bool = False,
              location_m=None) -> dict:
    if kind not in KIND:
        raise ValueError('rule kind must be object, group or row')
    mode_name(mode)
    if kind == 'group':
        managed.tag(key)
    elif not ROW.fullmatch(key):
        raise ValueError(f'invalid catalogue row {key!r}')
    rule = {'kind': kind, 'key': key, 'mode': mode,
            'channels': channel_ids(channels) if mode == 'flythrough' else [],
            'force': bool(force), 'keep_body': bool(keep_body) if mode == 'flythrough' else False}
    if kind == 'object':
        loc = managed.vec(location_m, 'location')
        if max(map(abs, loc)) > 1e6:
            raise ValueError('object location out of range')
        rule['location_m'] = [round(v, 4) for v in loc]
    return rule


def validate_rules(rules) -> list[dict]:
    if not isinstance(rules, list) or len(rules) > MAX_RULES:
        raise ValueError(f'collision rules must be a list of at most {MAX_RULES}')
    return [make_rule(r['kind'], r['key'], r['mode'], r.get('channels') or None, r.get('force', False),
                      r.get('keep_body', False), r.get('location_m')) for r in rules]


def rule_line(rule: dict) -> str:
    flags = ('F' if rule['force'] else '') + ('K' if rule['keep_body'] else '') or '-'
    head = f'{KIND[rule["kind"]]} {rule["mode"]} {mask_of(rule["channels"]):x} {flags} {rule["key"]}'
    if rule['kind'] == 'object':
        # fixed 0.01 cm, like the native C lines: '%g' lost the 2 cm tolerance beyond 10 km
        return head + ' ' + ' '.join(f'{v*100:.2f}' for v in rule['location_m']) + f' {TOLERANCE_CM:g}'
    return head


def collision_text(base: str, rules) -> str:
    if not re.fullmatch(r'[A-Za-z0-9_]+', base or ''):
        raise ValueError('invalid terrain base name')
    return '\n'.join([f'{HEADER} {base}'] + [rule_line(r) for r in validate_rules(rules)]) + '\n'


def rules_of(state: dict) -> list[dict]:
    block = state.get('collision', {'rules': []})
    if not isinstance(block, dict):
        raise ValueError('invalid collision metadata')
    return validate_rules(block.get('rules', []))


def _matches(rule: dict, obj: dict) -> float | None:
    """distance-like score (lower wins) or None. Mirrors native coll_match."""
    if obj.get('gate') and not rule['force']:
        return None
    if rule['kind'] == 'object':
        if rule['key'].casefold() != obj['row'].casefold() or obj.get('location_m') is None:
            return None
        d = math.dist(rule['location_m'], obj['location_m']) * 100
        return d if d <= TOLERANCE_CM + 1e-6 else None
    if rule['kind'] == 'group':
        return 0.0 if obj.get('group', '-') != '-' and rule['key'].casefold() == obj['group'].casefold() else None
    return 0.0 if rule['key'].casefold() == obj['row'].casefold() else None


def governing(rules: list[dict], obj: dict) -> dict | None:
    """Rule the native poll applies: nearest object rule, then group, then row; later duplicates win."""
    for kind in ('object', 'group', 'row'):
        best, best_d = None, None
        for rule in rules:
            if rule['kind'] != kind:
                continue
            d = _matches(rule, obj)
            if d is not None and (best_d is None or d <= best_d):
                best, best_d = rule, d
        if best is not None:
            return best
    return None


def _same_effect(rule: dict | None, mode: str, mask: int, keep_body: bool) -> bool:
    if rule is None or rule['mode'] == 'on':
        return mode == 'on'
    if rule['mode'] != mode:
        return False
    return mode != 'flythrough' or (mask_of(rule['channels']) == mask and rule['keep_body'] == keep_body)


def update_rules(rules: list[dict], selector: tuple[str, str], objects: list[dict], mode: str,
                 channels=None, force: bool = False, keep_body: bool = False) -> list[dict]:
    """New rule list so every selected object (gates only with force) ends in `mode`.

    selector is ('group', tag), ('row', row) or ('objects', ''). Object rules of the
    selection are replaced; mode 'on' becomes an explicit exception only where a
    broader rule would still apply."""
    mode_name(mode)
    kind, key = selector
    mask = mask_of(channel_ids(channels)) if mode == 'flythrough' else 0
    chosen = [o for o in objects if force or not o.get('gate')]
    out = []
    for rule in rules:
        if kind in ('group', 'row') and rule['kind'] == kind and rule['key'].casefold() == key.casefold():
            continue
        if rule['kind'] == 'object' and any(_matches(dict(rule, force=True), o) is not None for o in chosen):
            continue
        out.append(rule)
    if kind in ('group', 'row') and mode != 'on':
        out.append(make_rule(kind, key, mode, channels, force, keep_body))
    for o in chosen:
        if not _same_effect(governing(out, o), mode, mask, keep_body):
            out.append(make_rule('object', o['row'], mode, channels, force or bool(o.get('gate')), keep_body, o['location_m']))
    if len(out) > MAX_RULES:
        raise ValueError(f'more than {MAX_RULES} collision rules; prefer group/row rules over object selections')
    return out


def _flag(value: str) -> bool:
    return value not in ('0', '')


def parse_lines(text: str) -> dict:
    """Native C/P lines and the summary line of a collision verb out= file."""
    objects, by_actor, summary = [], {}, None
    for line in text.splitlines():
        if line.startswith('C '):
            p = line.split()
            f = environment.fields(line)
            loc = [float(v)/100 for v in f['loc'].split(',')]
            o = {'actor': environment.actor_id(p[1]), 'row': f['row'], 'group': f['group'],
                 'location_m': loc if all(math.isfinite(v) for v in loc) else None,
                 'gate': _flag(f['gate']), 'mode': f['mode'], 'owner': f.get('owner', '-'),
                 'channels': channel_names(int(f.get('rmask', '0'), 16))}
            for k in ('to', 'skipped'):
                if k in f:
                    o[k] = f[k]
            for k in ('change', 'ok', 'actor_collision', 'sim'):
                if k in f:
                    o[k] = _flag(f[k])
            o['keep_body'] = _flag(f.get('kb', '0'))
            if 'blocks' in f:
                o['blocks'] = channel_names(int(f['blocks'], 16))
                o['components'] = []
            for k in ('refused', 'partial', 'restore_failed'):
                if k in p:
                    o['problem'] = k
            objects.append(o)
            by_actor[o['actor']] = o
        elif line.startswith('P '):
            p = line.split()
            f = environment.fields(line)
            owner = by_actor.get(environment.actor_id(p[1]))
            if owner is None:
                raise ValueError('component line before its actor')
            owner['components'].append({
                'component': environment.actor_id(f['comp']), 'class': f['class'], 'enabled': int(f['enabled']),
                'object_channel': NAMES.get(int(f['object']), f['object']), 'profile': f['profile'],
                'simulating': _flag(f['sim']), 'blocks': channel_names(int(f['blocks'], 16)),
                'responses': {NAMES[i]: ('ignore', 'overlap', 'block')[int(f[f'r{i}'])] if f[f'r{i}'] in '012' else f[f'r{i}']
                              for i in range(7)}})
        elif line.startswith('collision '):
            summary = environment.fields(line)
    if summary is None:
        raise ValueError('collision result incomplete')
    return {'objects': objects, 'summary': summary}


ENABLED = ('NoCollision', 'QueryOnly', 'PhysicsOnly', 'QueryAndPhysics')


def passable(o: dict) -> bool:
    """A fully queried object that blocks none of the queried channels (drone passes through).
    Brief query lines carry no 'blocks' and are never passable by this test."""
    return 'blocks' in o and not o['blocks']


def passable_actors(query: dict) -> set[str]:
    """Actor ids audits should not treat as physical obstacles / clash partners."""
    return {o['actor'] for o in query['objects'] if passable(o) or o['mode'] in ('flythrough', 'off')}


# ---------------------------------------------------------------------------
# catalogue: stock-reliable non-colliding rows
# ---------------------------------------------------------------------------

def collision_filter(assets, collision: str):
    """'' | any: all; colliding; noncolliding (DT_MapEditorAssets Collision? flag, stock-reliable)."""
    c = (collision or 'any').casefold()
    if c == 'any':
        return list(assets)
    if c not in ('colliding', 'noncolliding'):
        raise ValueError('collision filter must be any, colliding or noncolliding')
    return [a for a in assets if a.collision == (c == 'colliding')]


def _tokens(asset) -> set[str]:
    return {t for t in re.split(r'[^a-z0-9]+', f'{asset.name} {asset.source or ""}'.lower()) if len(t) > 2 and not t.isdigit()} - {'sm', 'merged'}


def lookalikes(catalog, row: str, limit: int = 5) -> dict:
    """Non-colliding catalogue rows resembling `row` (same mesh, category, tags, size)."""
    target = catalog.get(row)
    result = {'row': row, 'name': target.name, 'collision': target.collision, 'suggestions': []}
    if not target.collision:
        result['note'] = 'row is already non-colliding in the stock game'
        return result
    scored = []
    for a in catalog.assets.values():
        if a.collision or a.id == row or a.category == 'Decals' and target.category != 'Decals':
            continue
        score = 0.0
        if a.source and a.source == target.source:
            score += 10
        if a.category == target.category:
            score += 2
        score += 2*len(set(a.tags) & set(target.tags)) + len(_tokens(a) & _tokens(target))
        ratio = [min(x, y)/max(x, y) if max(x, y) > 0 else 1 for x, y in zip(a.size_m, target.size_m)]
        score += 3*sum(ratio)/3
        if score >= 4:
            scored.append((score, a))
    scored.sort(key=lambda item: (-item[0], item[1].id))
    result['suggestions'] = [dict(a.public(), score=round(s, 2)) for s, a in scored[:max(1, min(limit, 20))]]
    if not result['suggestions']:
        result['note'] = ('no stock non-colliding look-alike exists; the catalogue only has non-colliding decals and '
                          'neon signs. Use set_collision(mode="flythrough") (live with the DLL only).')
    return result


# ---------------------------------------------------------------------------
# live
# ---------------------------------------------------------------------------

def _selector(tmp: Path, actors, group: str, row: str, box_min, box_max, allow_all: bool = False) -> tuple[str, tuple[str, str]]:
    """Native selector string and the rule scope. group/row/box go to the DLL as such (no handle
    list, so no 8192 cap); only an explicit actors list becomes a list= file."""
    given = sum(bool(x) for x in (actors, group, row, box_min is not None or box_max is not None))
    if given != 1:
        if given == 0 and allow_all:
            return 'all', ('objects', '')
        raise ValueError('choose exactly one selector: actors, group, row or box_min+box_max; never the whole map')
    if actors:
        ids = [environment.actor_id(a) for a in actors]
        if len(ids) > MAX_LIST or len(set(ids)) != len(ids):
            raise ValueError(f'actors must hold 1..{MAX_LIST} unique handles')
        return _list_file(tmp, ids, 'actors.txt'), ('objects', '')
    if group:
        return f'group={managed.tag(group)}', ('group', group)
    if row:
        if not ROW.fullmatch(row):
            raise ValueError(f'invalid catalogue row {row!r}')
        return f'row={row}', ('row', row)
    lo, hi = managed.vec(box_min, 'box_min'), managed.vec(box_max, 'box_max')
    if any(a > b for a, b in zip(lo, hi)):
        raise ValueError('box_min must not exceed box_max')
    return 'box=' + ','.join(f'{v*100:.3f}' for v in lo+hi), ('objects', '')


def _native(tmp: Path, verb: str, selector: str, options: str = '', name: str = 'out.txt') -> dict:
    path = tmp/name
    agent.run(f'collision {verb} {selector} {options} limit=0 out={agent._windows_path(path)}', 120)
    return parse_lines(path.read_text())


def _list_file(tmp: Path, actors: list[str], name: str) -> str:
    path = tmp/name
    path.write_text('\n'.join(actors)+'\n')
    return f'list={agent._windows_path(path)}'


def _chunks(ids: list[str]):
    for i in range(0, len(ids), MAX_LIST):
        yield ids[i:i+MAX_LIST]


def _run_lists(tmp: Path, command: str, ids: list[str], options: str = '') -> None:
    """command for every handle, in native-sized list files (native lists hold at most 8192)."""
    for i, chunk in enumerate(_chunks(ids)):
        agent.run(f'{command} {_list_file(tmp, chunk, f"chunk{i}.txt")} {options}'.rstrip(), 120)


def _options(channels, keep_body: bool, force: bool) -> str:
    return ' '.join(x for x in (f'channels={",".join(map(str, channel_ids(channels)))}',
                                'keepbody' if keep_body else '', 'force' if force else '') if x)


def _journal(root: Path, data: dict) -> Path:
    path = root/'UncrashedMCP.history'/f'{data["id"]}.json'
    managed.atomic_write(path, json.dumps(data, indent=2, allow_nan=False)+'\n')
    return path


def _write_rules(info: dict, root: Path, state: dict, rules: list[dict]) -> None:
    text = collision_text(info['base'], rules)  # validate before any write
    state['collision'] = {'version': 1, 'rules': rules}
    managed.write_state(root, state)
    managed.atomic_write(root/managed.COLLISION_NAME, text)


def _expected(mode: str, current: str) -> bool:
    if mode == 'physics_off':
        return current in ('nophysics', 'flythrough', 'off')
    if mode == 'physics_on':
        return current != 'nophysics'
    return current == mode


def set_collision(mode: str = 'flythrough', actors: list[str] | None = None, group: str = '', row: str = '',
                  box_min: list | None = None, box_max: list | None = None, channels: list | None = None,
                  keep_body: bool = False, force: bool = False, preview: bool = True, persist: bool = True,
                  top: int = 30) -> dict:
    physics = PHYSICS_MODES.get(mode)
    if physics is None:
        mode_name(mode)
    elif persist:
        raise ValueError(f'{mode} is a live-only toggle (it leaves flythrough/off alone); pass persist=False, '
                         'or persist mode="nophysics" / mode="on"')
    if mode == 'flythrough' or channels is not None:
        channel_ids(channels)
    if not 0 <= top <= 500:
        raise ValueError('top must be 0..500')
    verb = physics or mode
    with agent.session(), tempfile.TemporaryDirectory(prefix='uncrashed-collision-', dir='/tmp') as d:
        tmp = Path(d)
        selector, key = _selector(tmp, actors, group, row, box_min, box_max)
        options = _options(channels, keep_body, force)
        plan = _native(tmp, verb, selector, options + ' preview', 'preview.txt')
        objects = plan['objects']
        gates = [o for o in objects if o['gate']]
        result = {'mode': mode, 'preview': preview, 'persist': persist, 'matched': len(objects),
                  'would_change': sum(o.get('change', False) for o in objects), 'gates': len(gates),
                  'gates_skipped': 0 if force else len(gates),
                  'channels': channel_names(mask_of(channel_ids(channels))) if mode == 'flythrough' else [],
                  'objects': objects[:top], 'warnings': WARNINGS}
        if gates and not force:
            result['warnings'] = WARNINGS + [f'{len(gates)} race gates are checkpoints and were left alone; pass force=True to include them.']
        info = root = state = None
        if persist:
            info, root = managed.context()
            state = managed.read_state(info, root)
            before_rules = rules_of(state)
            after_rules = update_rules(before_rules, key, objects, mode, channels, force, keep_body)
            collision_text(info['base'], after_rules)
            result.update(rules_before=len(before_rules), rules_after=len(after_rules),
                          sidecar=str(root/managed.STATE_NAME), runtime_config=str(root/managed.COLLISION_NAME))
        if preview:
            return result
        chosen_objects = [o for o in objects if force or not o['gate']]
        if not chosen_objects:
            raise ValueError('nothing to change (only gates matched; pass force=True)')
        chosen = {o['actor'] for o in chosen_objects}
        try:
            info_j, root_j = (info, root) if persist else managed.context()
        except ValueError:
            info_j = root_j = None  # unsaved map: live-only change, no journal possible
        journal = {'version': 1, 'kind': 'collision', 'id': uuid.uuid4().hex, 'created': time.time(),
                   'label': f'collision {mode} {key[0]} {key[1]}'.strip(), 'status': 'collision_prepared',
                   'persist': persist, 'groups': [key[1]] if key[0] == 'group' else [],
                   'selector': selector if not selector.startswith('list=') else None,
                   'before': [{k: o[k] for k in ('actor', 'row', 'group', 'mode', 'owner', 'channels', 'keep_body', 'gate')}
                              for o in chosen_objects]}
        if info_j is not None:
            journal.update(map=info_j['folder'], base=info_j['base'])
        if persist:
            journal['rules_before'], journal['rules_after'] = before_rules, after_rules
        path = _journal(root_j, journal) if root_j is not None else None
        try:
            if persist:
                _write_rules(info, root, state, after_rules)
                # adopt only this selection: unrelated manual overrides elsewhere stay manual
                result['native'] = agent.run(f'collision refresh {selector}', 120)
                if mode == 'on':  # manual records the new rules do not cover
                    agent.run(f'collision on {selector} {"force" if force else ""}'.rstrip(), 120)
            else:
                agent.run(f'collision {verb} {selector} {options}', 120)
        except Exception as exc:
            note = 'originals retained by DLL'
            if persist:
                note = 'previous rules restored; ' + note
                _write_rules(info, root, state, before_rules)
                try:
                    agent.run(f'collision refresh {selector}', 120)
                except Exception as again:
                    note = f'restoring previous rules also failed: {again}; ' + note
            journal.update(status='collision_failed', error=str(exc))
            if path:
                _journal(root_j, journal)
            raise RuntimeError(f'collision change failed; {note}: {exc}') from exc
        # The change is in effect: make it undoable before verifying.
        journal['status'] = 'collision_applied'
        if path:
            _journal(root_j, journal)
            result.update(transaction=journal['id'], journal=str(path), undo='undo_collision with this transaction id')
        else:
            result['warnings'] = result['warnings'] + ['map not saved: no journal; use set_collision(mode="on") to restore']
        result['changed'] = True
        try:
            check = [o for o in _native(tmp, 'query', selector, 'brief', 'check.txt')['objects'] if o['actor'] in chosen]
        except Exception as exc:
            result.update(verified=None, verify_error=f'applied, but the verification query failed: {exc}')
            return result
        wrong = [o['actor'] for o in check if not _expected(mode, o['mode'])]
        missing = len(chosen) - len(check)
        result.update(verified=not wrong and not missing, mismatched=wrong[:50], missing_after=missing)
        if wrong or missing:
            journal.update(status='collision_partial', mismatched=wrong)
            if path:
                _journal(root_j, journal)
        return result


def undo_collision(transaction: str, preview: bool = True) -> dict:
    if not (len(transaction) == 32 and all(c in '0123456789abcdef' for c in transaction)):
        raise ValueError('transaction must be a 32-character lowercase hex id')
    with agent.session(), tempfile.TemporaryDirectory(prefix='uncrashed-collision-', dir='/tmp') as d:
        tmp = Path(d)
        info, root = managed.context()
        path = root/'UncrashedMCP.history'/f'{transaction}.json'
        journal = json.loads(path.read_text())
        if journal.get('kind') != 'collision' or journal.get('map') != info['folder'] or journal.get('base') != info['base']:
            raise ValueError('not a collision transaction of this map/base')
        if journal.get('status') not in ('collision_applied', 'collision_partial'):
            raise ValueError(f'transaction status {journal.get("status")!r} is not undoable')
        state = managed.read_state(info, root)
        if journal['persist'] and rules_of(state) != journal['rules_after']:
            raise ValueError('undo conflict: collision rules changed since this transaction; undo later transactions first')
        before = journal['before']
        # one map-wide probe: a single missing handle must not hide the live ones
        loaded = {o['actor'] for o in _native(tmp, 'query', 'all', 'brief', 'probe.txt')['objects']}
        live = [o for o in before if o['actor'] in loaded]
        plan = {'transaction': transaction, 'preview': preview, 'restores_rules': journal['persist'],
                'actors': len(before), 'actors_live': len(live), 'actors_missing': len(before) - len(live)}
        if len(live) < len(before):
            plan['note'] = (f'{len(before) - len(live)} journaled handles are no longer loaded (map reloaded or objects '
                            'deleted): ' + ('only the persisted rules apply to them' if journal['persist'] else
                                            'their live-only change is already gone'))
        if not live and not journal['persist']:
            plan.update(undoable=False, note='none of the journaled objects is loaded any more; a live-only change '
                                             'does not survive a reload, so there is nothing to undo')
            if preview:
                return plan
            raise ValueError(plan['note'])
        if preview:
            return plan
        if journal['persist']:
            _write_rules(info, root, state, journal['rules_before'])
            agent.run('collision refresh', 120)  # re-read; manual records elsewhere stay manual
        ids = [o['actor'] for o in live]
        _run_lists(tmp, 'collision refresh', ids)  # previously rule-owned objects go back to the rules
        plain = [o for o in live if o['owner'] == '-']
        if plain:
            _run_lists(tmp, 'collision on', [o['actor'] for o in plain], 'force' if any(o['gate'] for o in plain) else '')
        replay: dict[tuple, list[dict]] = {}
        for o in live:
            if o['owner'] == 'manual':
                replay.setdefault((o['mode'], tuple(o['channels']), bool(o.get('keep_body'))), []).append(o)
        for (mode, chans, keep), objs in replay.items():
            opts = _options(list(chans) or None, keep, any(o['gate'] for o in objs))
            _run_lists(tmp, f'collision {mode}', [o['actor'] for o in objs], opts)
        journal.update(status='collision_undone', undone=time.time(), undone_live=len(live))
        _journal(root, journal)
        plan['undone'] = True
        return plan


def collision_state(actors: list[str] | None = None, group: str = '', row: str = '', box_min: list | None = None,
                    box_max: list | None = None, channels: list | None = None, top: int = 30,
                    components: bool = True) -> dict:
    if not 0 <= top <= 500:
        raise ValueError('top must be 0..500')
    with agent.session(), tempfile.TemporaryDirectory(prefix='uncrashed-collision-', dir='/tmp') as d:
        tmp = Path(d)
        status = agent.run('collision status unmatched')
        result = {'native_status': status, 'warnings': WARNINGS}
        try:
            info, root = managed.context()
            rules = rules_of(managed.read_state(info, root))
            result.update(rules=len(rules), rules_by_kind={k: sum(r['kind'] == k for r in rules) for k in KIND},
                          sidecar=str(root/managed.STATE_NAME), runtime_config=str(root/managed.COLLISION_NAME))
        except ValueError as exc:
            result['rules_note'] = str(exc)
        if not (actors or group or row or box_min is not None or box_max is not None):
            return result
        selector, _ = _selector(tmp, actors, group, row, box_min, box_max)
        data = _native(tmp, 'query', selector, f'channels={",".join(map(str, channel_ids(channels)))}')
        objects = data['objects']
        if not components:
            objects = [{k: v for k, v in o.items() if k != 'components'} for o in objects]
        result.update(units='metres', channels=channel_names(mask_of(channel_ids(channels))),
                      matched=len(objects), blocking=sum(bool(o['blocks']) for o in data['objects']),
                      passable=sum(passable(o) for o in data['objects']),
                      modes={m: sum(o['mode'] == m for o in objects) for m in MODES},
                      simulating=sum(o.get('sim', False) for o in objects), objects=objects[:top],
                      enabled_legend=list(ENABLED))
        return result
