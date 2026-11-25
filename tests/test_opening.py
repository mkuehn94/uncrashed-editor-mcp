"""check_opening: first obstruction by path position, evidence classes and the verdict line (no game)."""
import contextlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from uncrashed_mcp import api, opening

PATH = [[0, -8, 4], [0, 0, 4], [0, 8, 4]]


def win_to_posix(p):
    return Path(p.replace('Z:', '').replace('\\', '/'))


def probes(i, near=None):
    """16 horizontal + up + down probes, all free except one horizontal at `near` metres."""
    cells = ['-1.0 0 0'] * 18
    if near is not None:
        cells[4] = f'{near * 100:.1f} n1 nc'
    return f'P {i} ' + ' '.join(cells)


class FakeSweep:
    """Answers `sweep` with synthetic hits keyed on the sample's y (cm)."""

    def __init__(self, hits=True, near=True, truncated=0, near_zones=((500, 525),), near_class='StaticMeshActor'):
        self.hits, self.near, self.truncated, self.commands = hits, near, truncated, []
        self.near_zones, self.near_class = near_zones, near_class

    def run(self, command, timeout=30):
        self.commands.append(command)
        p = command.split()
        assert p[0] == 'sweep', command
        req = win_to_posix(p[1]).read_text().splitlines()
        out = []
        for line in req:
            if not line.startswith('S '):
                continue
            q = line.split()
            i, y = int(q[1]), float(q[3])
            if self.hits and q[7] == '1':
                if -300 <= y < -275:
                    out.append(f'H {i} f1 fc 0 0.0 {y:.1f} 400.0 0.5')     # flythrough override: passes live
                if -25 <= y < 0:
                    out.append(f'H {i} a1 ac 0 0.0 {y:.1f} 400.0 0.7')     # later in the segment
                    out.append(f'H {i} b2 bc 0 0.0 {y:.1f} 400.0 0.2')     # earlier in the same segment
                if 300 <= y < 325:
                    out.append(f'H {i} u1 uc 0 0.0 {y:.1f} 400.0 0.1')     # unknown response
            if q[8] == '1':
                out.append(probes(i, .4 if self.near and any(lo <= y < hi for lo, hi in self.near_zones) else None))
        if self.hits or self.near:   # native lists only components / actors a trace touched
            out += ['K fc f1 StaticMeshComponent 0 0 1', 'K ac a1 StaticMeshComponent 2 0 3',
                    'K bc b2 StaticMeshComponent 2 0 3', 'K uc u1 StaticMeshComponent -1 0 -1',
                    'K nc n1 StaticMeshComponent 2 0 3',
                    'A f1 NewRow_3 StaticMeshActor', 'A a1 NewRow_7 StaticMeshActor', 'A b2 NewRow_8 StaticMeshActor',
                    'A u1 NewRow_9 StaticMeshActor', f'A n1 NewRow_5 {self.near_class}']
        out += [f'sweep mode=sweep samples=1 segments=1 hits=1 overlap_only=0 truncated={self.truncated} seen_full=0 '
                'failed=0 channel=2 ms=3']
        win_to_posix(p[2]).write_text('\n'.join(out) + '\n')
        return ['sweep ok']


def state(objects, matched=None, lookup_error=None):
    def fake(**kw):
        if 'actors' in kw:  # a lookup by handle answers for every handle (default mode on)
            if lookup_error:
                raise lookup_error
            known = {o['actor']: o for o in objects}
            return {'objects': [known.get(a, {'actor': a, 'row': None, 'mode': 'on'}) for a in kw['actors']],
                    'matched': len(kw['actors'])}
        return {'objects': objects, 'matched': len(objects) if matched is None else matched}
    return fake


NEARBY = [{'actor': 'f1', 'row': 'NewRow_3', 'mode': 'flythrough', 'location_m': [0, -3, 4]},
          {'actor': 'a1', 'row': 'NewRow_7', 'mode': 'on', 'location_m': [0, 0, 4]},
          {'actor': 'b2', 'row': 'NewRow_8', 'mode': 'on', 'location_m': [0, 0, 4]},
          {'actor': 'bp', 'row': 'NewRow_1', 'mode': 'on', 'location_m': [3, 0, 0]},     # never traced
          {'actor': 'off1', 'row': 'NewRow_2', 'mode': 'off', 'location_m': [-2, 1, 4]}]


class OpeningTests(unittest.TestCase):
    def check(self, world=None, objects=NEARBY, matched=None, lookup_error=None, **kw):
        world = world or FakeSweep()
        with TemporaryDirectory() as tmp, patch.object(opening, 'REPORT_DIR', Path(tmp)), \
                patch.object(opening.agent, 'run', world.run), patch.object(opening.agent, 'session', contextlib.nullcontext), \
                patch.object(opening.scenery, 'collision_state', state(objects, matched, lookup_error)):
            reply = opening.check_opening(PATH, **kw)
            report = json.loads(Path(reply['report_json']).read_text())
            self.assertTrue(Path(reply['report_json']).name.startswith('opening-'))
        return world, reply, report

    def test_first_obstruction_is_ordered_by_position_then_hit_time(self):
        _, reply, report = self.check()
        first = reply['first_obstruction']
        self.assertEqual((first['actor'], first['row'], first['evidence'], first['kind']),
                         ('b2', 'NewRow_8', 'live-collision', 'blocked'))
        self.assertEqual(first['time'], .2)
        self.assertEqual(first['leg'], 1)
        self.assertTrue(reply['verdict'].startswith('BLOCKED at '))
        self.assertIn('NewRow_8', reply['verdict'])
        distances = [f['distance_m'] for f in report['findings']]
        self.assertEqual(distances, sorted(distances))
        a1 = next(f for f in reply['findings'] if f['actor'] == 'a1')
        self.assertGreater(a1['distance_m'], first['distance_m'])
        self.assertEqual(a1['sample'], first['sample'])

    def test_evidence_classes(self):
        _, reply, report = self.check()
        by_actor = {f['actor']: f for f in report['findings']}
        self.assertEqual(by_actor['f1']['evidence'], 'override-active')
        self.assertEqual(by_actor['f1']['kind'], 'passed_by_override')
        self.assertEqual(by_actor['u1']['evidence'], 'unverified')
        self.assertEqual(by_actor['n1']['kind'], 'near_miss_tight')
        self.assertEqual(by_actor['n1']['evidence'], 'live-collision')
        self.assertEqual(reply['evidence_counts'], {'live-collision': 3, 'override-active': 1, 'bounds-estimate': 0,
                                                   'unverified': 1})
        untouched = {o['actor']: o for o in reply['unverified_objects']}
        self.assertEqual(set(untouched), {'bp', 'off1'})
        self.assertIn('invisible', untouched['off1']['reason'])
        self.assertIn('unverified', reply['verdict'])
        self.assertTrue(reply['notes'])
        self.assertGreater(reply['route']['spline_max_offset_m'], -1)

    def test_clean_sweep_never_says_clear_without_caveats(self):
        _, reply, _ = self.check(FakeSweep(hits=False, near=False))
        self.assertIsNone(reply['first_obstruction'])
        self.assertTrue(reply['verdict'].startswith('no blocking sweep hit'))
        self.assertIn('NOT proven clear', reply['verdict'])
        self.assertIn('unverified', reply['verdict'])
        self.assertIn('near-miss', reply['verdict'])
        self.assertEqual(reply['unverified_object_count'], 5)

    def test_incomplete_native_or_capped_lookup(self):
        _, reply, _ = self.check(FakeSweep(truncated=1))
        self.assertFalse(reply['complete'])
        self.assertTrue(reply['verdict'].startswith('INCOMPLETE'))
        _, reply, _ = self.check(matched=900)
        self.assertFalse(reply['complete'])

    def test_many_objects_near_the_path_are_paged_not_cut_at_500(self):
        many = [{'actor': f'w{i}', 'row': 'NewRow_502', 'mode': 'on', 'location_m': [i * 0.006 - 3.6, 0, 4]}
                for i in range(1200)] + NEARBY

        def fake(**kw):
            if 'actors' in kw:
                return {'objects': [{'actor': a, 'row': None, 'mode': 'on'} for a in kw['actors']],
                        'matched': len(kw['actors'])}
            lo, hi = kw['box_min'], kw['box_max']
            inside = [o for o in many if all(lo[k] <= o['location_m'][k] <= hi[k] for k in range(3))]
            return {'objects': inside[:kw['top']], 'matched': len(inside)}
        world = FakeSweep(hits=False, near=False)
        with TemporaryDirectory() as tmp, patch.object(opening, 'REPORT_DIR', Path(tmp)), \
                patch.object(opening.agent, 'run', world.run), \
                patch.object(opening.agent, 'session', contextlib.nullcontext), \
                patch.object(opening.scenery, 'collision_state', fake):
            reply = opening.check_opening(PATH, near_m=5)
        self.assertTrue(reply['complete'], reply['notes'])
        self.assertEqual(reply['unverified_object_count'], len(many))
        self.assertTrue(any('box pages' in n for n in reply['notes']), reply['notes'])

    def test_failed_override_lookup_is_incomplete_and_unverified(self):
        _, reply, report = self.check(lookup_error=ValueError('lookup refused'))
        self.assertFalse(reply['complete'])
        self.assertTrue(reply['override_lookup_failed'])
        self.assertTrue(reply['verdict'].startswith('INCOMPLETE'))
        n1 = next(f for f in report['findings'] if f['actor'] == 'n1')
        self.assertEqual((n1['kind'], n1['evidence']), ('near_miss_tight', 'unverified'))
        self.assertNotIn('live-collision', {f['evidence'] for f in report['findings'] if f['actor'] in ('n1', 'u1')})

    def test_too_many_traced_handles_is_incomplete(self):
        with patch.object(opening.scenery, 'MAX_LIST', 1):
            _, reply, report = self.check()
        self.assertFalse(reply['complete'])
        n1 = next(f for f in report['findings'] if f['actor'] == 'n1')
        self.assertEqual(n1['evidence'], 'unverified')

    def test_gate_inside_the_margin_is_a_near_miss(self):
        _, reply, report = self.check(FakeSweep(hits=False, near_class='BP_Gate_C'))
        gate = [f for f in report['findings'] if f['actor'] == 'n1']
        self.assertEqual([(f['kind'], f['class']) for f in gate], [('near_miss_tight', 'BP_Gate_C')])

    def test_near_miss_carries_the_catalogue_name(self):
        class Rows:
            def name(self, row):
                return f'name of {row}'
        with patch('uncrashed_mcp.collide.Rows.from_catalog', lambda catalog: Rows()):
            _, reply, report = self.check(catalog='CAT')
        n1 = next(f for f in report['findings'] if f['actor'] == 'n1')
        self.assertEqual(n1['name'], 'name of NewRow_5')

    def test_separate_runs_of_the_same_obstacle_stay_separate(self):
        _, reply, report = self.check(FakeSweep(hits=False, near_zones=((500, 551), (650, 676))))
        runs = [f for f in report['findings'] if f['actor'] == 'n1']
        self.assertEqual(len(runs), 2)
        self.assertTrue(all(f['hits'] >= 2 for f in runs))
        for f in runs:
            self.assertLess(f['distance_m'], f['last_distance_m'])
        self.assertLess(runs[0]['last_distance_m'] + 0.5, runs[1]['distance_m'])

    def test_each_call_writes_its_own_report(self):
        _, one, _ = self.check()
        _, two, _ = self.check()
        self.assertNotEqual(one['report_json'], two['report_json'])

    def test_validation_before_any_agent_call(self):
        def refuse(*a, **k):
            raise AssertionError('agent reached')
        with patch.object(opening.agent, 'run', refuse), patch.object(opening.agent, 'session', refuse):
            for path in ([[0, 0, 0]], [[0, 0, i] for i in range(9)], [[0, 0], [1, 1, 1]],
                         [[0, 0, 0], [0, 0, 0.01]], [[0, 0, 0], [400, 0, 0]], [[0, 0, 0], [1, float('nan'), 0]]):
                with self.assertRaises(ValueError):
                    opening.check_opening(path)
            for kw in ({'spacing_m': 5}, {'near_m': -1}, {'top': 0}, {'channel': 'Water'}, {'profile': 'car'}):
                with self.assertRaises(ValueError):
                    opening.check_opening(PATH, **kw)

    def test_api_forwards_by_name(self):
        echo = lambda *a, **k: {'args': list(a), 'kwargs': k}
        with patch.object(api.opening, 'check_opening', echo), patch.object(api, '_catalog', lambda: 'CAT'):
            self.assertEqual(api.call('check_opening', PATH, margin_m=.3), {'args': [PATH], 'kwargs': {
                'profile': '5inch', 'margin_m': .3, 'channel': 'Pawn', 'spacing_m': .25, 'dims_m': None,
                'near_m': 5.0, 'top': 10, 'catalog': 'CAT'}})


if __name__ == '__main__':
    unittest.main()
