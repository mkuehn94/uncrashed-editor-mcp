"""Opening check: sweep the craft body along a short approach -> opening -> exit path and report the FIRST
obstruction along it, with an evidence class for every finding.

Built on the route-clearance machinery (clearance.py, source='points'): the same spline, native `sweep`
request/reply and probe classification. On top it keeps the hit time, orders by position along the path
(sample index, then hit time), and asks `collision query` (scenery.collision_state) which nearby placed objects
carry a DLL collision override, so a live result is never presented as stock behaviour.

Evidence classes:
  live-collision   a native sweep/probe hit on a component with a known blocking response, no override active
  override-active  the object has a DLL collision override (mode != on): the stock default was not what was tested
  bounds-estimate  bounds-only data (not produced by this check: the sweep has no bounds-only path; kept in the legend)
  unverified       no usable data: unknown response, or a placed object near the path that no trace touched
                   (Blueprint buildings can lack query collision; collision 'off' objects are invisible to traces)
"""
from __future__ import annotations

import json
import math
import tempfile
import time
import uuid
from pathlib import Path

from . import agent, clearance, scenery

REPORT_DIR = clearance.REPORT_DIR
MAX_POINTS = 8
MAX_LENGTH_M = 300.0
DIRECTIONS = 16
PROBE_M = 10.0
EVIDENCE = ('live-collision', 'override-active', 'bounds-estimate', 'unverified')
NEAR_MISS = ('tight', 'ceiling_low', 'ground_low')
LIMITATIONS = [
    'The path is a centripetal Catmull-Rom spline through your points: corners are rounded (spline_max_offset_m), '
    'it is not a recorded flight line.',
    'The swept box is the craft body only (no margin, no banking or roll, no prop wash or flight dynamics); '
    'margin_m is applied to probe clearances (near-miss) only.',
    'Evidence is the LIVE collision state of the open editor with the resident DLL; stock Uncrashed ignores MCP '
    'collision overrides. override-active findings were not tested with stock collision.',
    'Which channel the drone really collides as is not verified (default Pawn); unknown responses count as blocking.',
    'Objects with collision off are invisible to traces; Blueprint buildings can have floors without query '
    'collision, so no hit next to an unverified object is not clearance.',
    'Nearby objects are found by pivot inside the path box widened by near_m: large objects with a far pivot '
    'can be missed.',
    'Probes run in 16 horizontal directions plus up/down from each sample, not in every direction.',
    'Geometry check only: not a flight test, visual check or proof that the line is flyable.',
]


def _seg_dist(p, a, b):
    ab, ap = clearance._sub(b, a), clearance._sub(p, a)
    n = clearance._dot(ab, ab)
    t = 0.0 if n < 1e-12 else max(0.0, min(1.0, clearance._dot(ap, ab) / n))
    return clearance._len(clearance._sub(p, clearance._add(a, clearance._mul(ab, t))))


def validate_path(path) -> list[tuple[float, float, float]]:
    if not isinstance(path, (list, tuple)) or not 2 <= len(path) <= MAX_POINTS:
        raise ValueError(f'path needs 2..{MAX_POINTS} x,y,z points (approach -> opening -> exit, metres)')
    pts = []
    for p in path:
        if (not isinstance(p, (list, tuple)) or len(p) != 3
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in p)):
            raise ValueError('each path point must be x,y,z: three finite numbers in metres')
        pts.append(tuple(float(v) for v in p))
    for a, b in zip(pts, pts[1:]):
        if clearance._len(clearance._sub(b, a)) < 0.05:
            raise ValueError('consecutive path points must be at least 0.05 m apart')
    length = sum(clearance._len(clearance._sub(b, a)) for a, b in zip(pts, pts[1:]))
    if length > MAX_LENGTH_M:
        raise ValueError(f'path is {length:.0f} m long; check_opening takes up to {MAX_LENGTH_M:.0f} m '
                         '(use check_route_clearance for whole routes)')
    return pts


def _leg(distance, marks):
    """1-based leg number: leg k runs from point k to point k+1."""
    leg = 1
    for k, d in enumerate(marks[1:-1], 2):
        if distance >= d:
            leg = k
    return leg


def _who(ob):
    ob = ob or {}
    return {'actor': ob.get('actor'), 'row': ob.get('row'), 'name': ob.get('name'), 'class': ob.get('class'),
            'component_class': ob.get('component_class')}


def classify(samples, parsed, prof, controls, overrides, rows=None, unchecked=()):
    """Ordered findings with evidence classes. overrides: {actor: mode} for placed objects with mode != on;
    unchecked: actors whose override state could not be looked up (their hits are 'unverified')."""
    unchecked = set(unchecked)
    by_i = {s['i']: s for s in samples}
    marks = [clearance.nearest_sample(samples, c)[0]['d'] for c in controls]
    found = []

    def row(kind, s, t, ob, evidence, why, measure=None, extra=None):
        d = s['d'] if t is None or s['i'] + 1 not in by_i else s['d'] + t * (by_i[s['i'] + 1]['d'] - s['d'])
        rec = {'distance_m': round(d, 3), 'leg': _leg(d, marks), 'sample': s['i'],
               'time': None if t is None else round(t, 4), 'kind': kind, 'evidence': evidence,
               'measure_m': measure, **_who(ob), 'why': why}
        rec['_extra'] = extra or {}
        found.append(rec)

    for h in parsed['hits']:
        s = by_i.get(h['seg'])
        if s is None:
            continue
        comp = parsed['components'].get(h['component'], {})
        ob = clearance.obstacle(h['actor'], h['component'], parsed, rows) or {'actor': None, 'kind': 'unknown'}
        mode = overrides.get(h['actor'])
        extra = {'impact_m': h['impact'], 'start_penetrating': h['start_penetrating'], 'at_m': s['p'],
                 'view': clearance.view_for(s, prof)}
        if clearance.blocking(h['component'], parsed):
            if h['actor'] in unchecked:
                row('blocked', s, h['time'], ob, 'unverified', 'override state not looked up; counted as blocking',
                    extra=extra)
            elif mode:
                row('blocked', s, h['time'], ob, 'override-active', f'DLL override mode={mode}; stock not tested',
                    extra=extra)
            elif comp.get('response', -1) == -1 or comp.get('collision_enabled', -1) == -1:
                row('blocked', s, h['time'], ob, 'unverified', 'collision response unknown; counted as blocking',
                    extra=extra)
            else:
                row('blocked', s, h['time'], ob, 'live-collision', 'sweep hit, blocking response', extra=extra)
        elif mode:  # passed only because of the override: may block in stock
            row('passed_by_override', s, h['time'], ob, 'override-active',
                f'not blocking live because of DLL override mode={mode}; stock may block', extra=extra)
        elif h['actor'] in unchecked:  # may be passing only because of an override nobody checked
            row('passed_unchecked', s, h['time'], ob, 'unverified',
                'not blocking live; override state not looked up, stock may block', extra=extra)
    near, _clear, _overlap = clearance.sample_findings(samples, parsed, prof, DIRECTIONS, rows=rows, judge_gates=True)
    for f in near:
        if f['kind'] not in NEAR_MISS:
            continue
        s = by_i[f['sample']]
        ob = f.get('obstacle')
        actor = (ob or {}).get('actor')
        if actor in unchecked:
            ev, why = 'unverified', 'probe within margin; override state not looked up'
        elif actor in overrides:
            ev, why = 'override-active', f'probe within margin; DLL override mode={overrides[actor]}'
        elif ob and ob.get('response') == 'unknown':
            ev, why = 'unverified', 'probe within margin; collision response unknown'
        else:
            ev, why = 'live-collision', f'probe {f["measure_m"]} m from the body, margin {prof["margin"]} m'
        row('near_miss_' + f['kind'], s, None, ob, ev, why, f['measure_m'], {'at_m': s['p'], 'view': clearance.view_for(s, prof)})
    found.sort(key=lambda r: (r['sample'], -1 if r['time'] is None else r['time'], r['distance_m']))
    # one row per run of consecutive samples with the same (kind, actor): the first along the path, with the hit
    # count and the distance where the run ends; a gap of more than one sample starts a new run
    merged, seen = [], {}
    for r in found:
        key = (r['kind'], r['actor'])
        first = seen.get(key)
        if first is not None and r['sample'] - first['_last'] <= 1:
            first['hits'] += 1
            first['_last'] = r['sample']
            first['last_distance_m'] = r['distance_m']
            if r['measure_m'] is not None and (first['measure_m'] is None or r['measure_m'] < first['measure_m']):
                first['measure_m'] = r['measure_m']
            continue
        r['hits'], r['_last'], r['last_distance_m'] = 1, r['sample'], r['distance_m']
        seen[key] = r
        merged.append(r)
    return merged, marks


def _ordered(rec):
    return {k: rec[k] for k in ('distance_m', 'last_distance_m', 'leg', 'sample', 'time', 'kind', 'evidence', 'measure_m', 'hits',
                                'actor', 'row', 'name', 'class', 'component_class', 'why')}


def verdict(first, complete, counts, unverified_objects, near_miss) -> str:
    tail = (f'{unverified_objects + counts["unverified"]} unverified, {near_miss} near-miss, '
            f'{counts["override-active"]} override-active')
    if not complete:
        return f'INCOMPLETE (truncated/failed traces or capped override lookup): not evidence; {tail}'
    if first:
        who = first['name'] or first['row'] or first['class'] or 'unknown object'
        return (f'BLOCKED at {first["distance_m"]} m, leg {first["leg"]} (sample {first["sample"]}, '
                f't={first["time"]}) by {who} [{first["evidence"]}]; {tail}')
    return f'no blocking sweep hit for the craft body; NOT proven clear: {tail}; geometry only, not a flight test'


PAGE = 500          # collision_state answers at most this many objects per call
MAX_SPLIT = 8       # box halvings when one box holds more than a page (2^8 boxes at most)


def _nearby(lo, hi, depth: int = 0) -> tuple[list[dict], int, int]:
    """Collision state of every placed object whose pivot lies in the box lo..hi, paged: a box holding more than
    PAGE objects is halved along its longer horizontal side until each half fits. Returns (objects, matched,
    calls); matched > len(objects) only when MAX_SPLIT halvings were not enough."""
    state = scenery.collision_state(box_min=list(lo), box_max=list(hi), top=PAGE, components=False)
    objs = list(state.get('objects', []))
    matched = int(state.get('matched', len(objs)))
    if matched <= len(objs) or depth >= MAX_SPLIT:
        return objs, matched, 1
    k = 0 if hi[0] - lo[0] >= hi[1] - lo[1] else 1
    mid = (lo[k] + hi[k]) / 2
    hi_a, lo_b = list(hi), list(lo)
    hi_a[k], lo_b[k] = mid, mid
    out: dict[str, dict] = {}
    total, calls = 0, 1
    for a, b in ((lo, hi_a), (lo_b, hi)):
        o, m, c = _nearby(a, b, depth + 1)
        calls += c
        total += m - len(o)
        for x in o:
            out.setdefault(x['actor'], x)
    return list(out.values()), len(out) + total, calls


def check_opening(path, profile='5inch', margin_m=None, channel='Pawn', spacing_m=0.25, dims_m=None,
                  near_m=5.0, top=10, catalog=None) -> dict:
    controls = validate_path(path)
    prof = clearance.resolve_profile(profile, dims_m, margin_m)
    if prof['kind'] != 'air':
        raise ValueError(f"profile {prof['name']} is a ground vehicle; check_opening takes air profiles "
                         '(tiny_whoop, cinewhoop, 5inch, 7inch, fixed_wing)')
    ch = clearance._channel(channel)
    if not 0.1 <= spacing_m <= 2:
        raise ValueError('spacing_m must be 0.1..2')
    if not math.isfinite(near_m) or not 0 <= near_m <= 50:
        raise ValueError('near_m must be 0..50')
    if not 1 <= top <= 100:
        raise ValueError('top must be 1..100')
    rows = None
    if catalog is not None:
        from .collide import Rows
        rows = Rows.from_catalog(catalog)
    with agent.session(), tempfile.TemporaryDirectory(prefix='uncrashed-opening-', dir='/tmp') as tmp:
        tmp = Path(tmp)
        info = clearance.route_source('points', [list(p) for p in controls], tmp, ch, closed=False)
        samples = clearance.build_route([tuple(p) for p in info['points']], spacing_m, False)
        parts = []
        for first, last in clearance.chunks(len(samples)):
            req, out = tmp / 'sweep-request.txt', tmp / 'sweep.txt'
            req.write_text(clearance.request_text(samples, prof, 'sweep', ch, PROBE_M, DIRECTIONS, first=first, last=last))
            clearance._run_native(f'sweep {agent._windows_path(req)} {agent._windows_path(out)}')
            parts.append(clearance.parse_sweep(out.read_text(), DIRECTIONS))
        parsed = clearance.merge_parsed(parts)
        lo = [min(s['p'][k] for s in samples) - near_m for k in range(3)]
        hi = [max(s['p'][k] for s in samples) + near_m for k in range(3)]
        notes = []
        nearby, matched, pages = _nearby(lo, hi)
        capped = matched > len(nearby)
        known = {o['actor'] for o in nearby}
        traced = {a for a, v in parsed['actors'].items() if v.get('row')} - known
        if capped:
            notes.append(f'{matched} placed objects near the path; only {len(nearby)} were checked for overrides '
                         f'({pages} box pages, split limit reached)')
        elif pages > 1:
            notes.append(f'{len(nearby)} placed objects near the path, looked up in {pages} box pages')
        unchecked = set()
        if len(traced) > scenery.MAX_LIST:
            unchecked = traced
            notes.append(f'{len(traced)} hit object(s) outside the box: too many to look up overrides; '
                         'their hits are unverified')
        elif traced:
            try:
                extra = {'objects': []}
                names = sorted(traced)
                for k in range(0, len(names), PAGE):     # by handle, one page of PAGE at a time
                    extra['objects'] += scenery.collision_state(actors=names[k:k + PAGE], top=PAGE,
                                                                components=False).get('objects', [])
                nearby += extra.get('objects', [])
                unchecked = traced - {o['actor'] for o in extra.get('objects', [])}
                if unchecked:
                    notes.append(f'override state of {len(unchecked)} hit object(s) not returned; their hits are unverified')
            except (ValueError, agent.AgentError) as exc:
                if isinstance(exc, (agent.AgentNotSent, agent.AgentTimeout)):
                    raise
                unchecked = traced
                notes.append(f'override state of {len(traced)} hit object(s) outside the box unknown '
                             f'(hits on them are unverified): {exc}')
        override_lookup_failed = bool(unchecked)
    overrides = {o['actor']: o['mode'] for o in nearby if o.get('mode') not in (None, 'on')}
    findings, marks = classify(samples, parsed, prof, controls, overrides, rows, unchecked)
    touched = set(parsed['actors'])
    unverified_objects = []
    for o in nearby:
        if o['actor'] in touched:
            continue
        reason = ('collision off override: invisible to traces' if o.get('mode') == 'off'
                  else 'no trace data: may lack query collision (Blueprint) or be out of reach')
        unverified_objects.append({'actor': o['actor'], 'row': o.get('row'),
                                   'name': rows.name(o['row']) if rows and o.get('row') else None,
                                   'mode': o.get('mode'), 'location_m': o.get('location_m'), 'reason': reason})
    native = parsed['summaries']
    truncated = any(s.get('truncated') or s.get('seen_full') for s in native)
    failed = sum(s.get('failed', 0) for s in native)
    complete = not truncated and not failed and not capped and not override_lookup_failed
    counts = {e: sum(f['evidence'] == e for f in findings) for e in EVIDENCE}
    near_miss = sum(f['kind'].startswith('near_miss_') for f in findings)
    first = next((f for f in findings if f['kind'] == 'blocked'), None)
    line = verdict(first, complete, counts, len(unverified_objects), near_miss)
    offset = max(min(_seg_dist(s['p'], a, b) for a, b in zip(controls, controls[1:])) for s in samples)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    report_path = REPORT_DIR / f'opening-{stamp}-{uuid.uuid4().hex[:8]}.json'
    report = {
        'verdict': line, 'units': 'metres', 'complete': complete,
        'profile': prof, 'collision_channel': {'name': clearance.channel_name(ch), 'index': ch},
        'path_points_m': [list(p) for p in controls], 'point_distance_m': marks,
        'route': {'length_m': samples[-1]['d'], 'samples': len(samples), 'spacing_m': spacing_m,
                  'spline_max_offset_m': round(offset, 3)},
        'override_lookup_failed': override_lookup_failed,
        'native': {'jobs': len(native), 'truncated': truncated, 'failed_traces': failed,
                   'ms': sum(s.get('ms', 0) for s in native)},
        'evidence_counts': counts,
        'first_obstruction': None if first is None else _ordered(first) | first['_extra'],
        'findings': [_ordered(f) | f['_extra'] for f in findings],
        'unverified_objects': unverified_objects, 'overrides_near_path': overrides, 'notes': notes,
        'evidence_legend': {e: d for e, d in zip(EVIDENCE, (
            'native sweep/probe hit on a known blocking response, no DLL override',
            'object has a DLL collision override: stock default was not what was tested',
            'bounds-only data (none produced by this check)',
            'no usable data: unknown response or a nearby object no trace touched'))},
        'limitations': LIMITATIONS,
        'path': clearance._path(samples),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1))
    reply = {
        'verdict': line, 'units': 'metres', 'complete': complete, 'override_lookup_failed': override_lookup_failed,
        'profile': {k: prof[k] for k in ('name', 'width', 'length', 'height', 'margin')},
        'collision_channel': clearance.channel_name(ch),
        'route': report['route'] | {'point_distance_m': marks},
        'evidence_counts': counts,
        'first_obstruction': report['first_obstruction'],
        'findings': [_ordered(f) for f in findings[:top]], 'finding_count': len(findings),
        'unverified_objects': [{k: o[k] for k in ('actor', 'row', 'name', 'mode', 'reason')} for o in unverified_objects[:top]],
        'unverified_object_count': len(unverified_objects),
        'notes': notes + ['the spline rounds the path through your points (see spline_max_offset_m)',
                          'the swept box is the craft body only; margin applies to probe near-misses'],
        'report_json': str(report_path),
        'limitations': LIMITATIONS,
    }
    return reply
