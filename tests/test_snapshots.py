import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uncrashed_mcp import groups, snapshots


def obj(actor, group='house', xyz=(100, 0, 0), rot=(0, 0, 0), row='NewRow_1', scale=1):
    return {'actor': actor, 'row': row, 'group': group, 'xf': list(xyz) + list(rot) + [scale] * 3}


def gate(actor, xyz, start=False, row='DefaultOneW', rot=(0, 90, 0)):
    return {'actor': actor, 'row': 'STARTGATE' if start else row, 'start': start, 'pivot': list(xyz),
            'rotation': list(rot), 'scale': [1.0, 1.0, 1.0], 'aabb': None, 'components': []}


SCENE = [obj('a1', 'house.a', (100, 0, 0)), obj('a2', 'house.b', (300, 0, 0)), obj('a3', 'tree', (5000, 0, 0), row='NewRow_9'),
         obj('a4', 'house.a', (100, 0, 0))]  # a4 duplicates a1's geometry
GATES = [gate('g0', (0, 0, 1), start=True), gate('g1', (10, 0, 2)), gate('g2', (20, 5, 3))]


def setUpModule():
    tmp = tempfile.TemporaryDirectory()
    unittest.addModuleCleanup(tmp.cleanup)
    lock = patch.object(snapshots.agent, 'LOCK', Path(tmp.name) / 'agent.lock')
    lock.start()
    unittest.addModuleCleanup(lock.stop)


class Scene:
    """A fake open editor: managed.context, groups export, gatebox and mapinfo."""

    def __init__(self, test, root, objects=SCENE, gates=GATES, session='s1', folder='Map_A', base='Base_1'):
        self.objects, self.gates, self.session, self.folder, self.base = objects, gates, session, folder, base
        info = lambda: {'folder': self.folder, 'base': self.base}  # noqa: E731
        self.patches = [patch.object(snapshots.managed, 'context', lambda: (snapshots.agent.map_info(), root)),
                        patch.object(snapshots.groups, 'snapshot',
                                     lambda: {'session': self.session, 'objects': copy.deepcopy(self.objects)}),
                        patch.object(snapshots.clearance, 'live_gates', lambda tmp, ch: {'gates': copy.deepcopy(self.gates)}),
                        patch.object(snapshots.agent, 'map_info', info)]
        for p in self.patches:
            p.start()
            test.addCleanup(p.stop)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / 'Map_A'
        self.root.mkdir()
        self.scene = Scene(self, self.root)

    def test_fingerprint_ignores_order_and_actor_ids_but_not_moves(self):
        renamed = [dict(o, actor='b' + o['actor']) for o in reversed(SCENE)]
        self.assertEqual(snapshots.fingerprint(SCENE), snapshots.fingerprint(renamed))
        moved = copy.deepcopy(SCENE)
        moved[0]['xf'][0] += 1
        self.assertNotEqual(snapshots.fingerprint(SCENE), snapshots.fingerprint(moved))
        swapped = [GATES[1], GATES[0], GATES[2]]
        self.assertNotEqual(snapshots.gate_fingerprint(snapshots._gates(GATES)), snapshots.gate_fingerprint(snapshots._gates(swapped)))

    def test_snapshot_writes_beside_the_map_and_returns_a_stamp(self):
        stamp = snapshots.scene_snapshot(name='before')
        path = self.root / 'UncrashedMCP.snapshots' / 'before.json.gz'
        self.assertEqual(stamp['file'], str(path))
        self.assertEqual((stamp['map'], stamp['base'], stamp['session'], stamp['objects'], stamp['scope_objects'], stamp['gates']),
                         ('Map_A', 'Base_1', 's1', 4, 4, 3))
        self.assertEqual(stamp['start_gate'], {'index': 0, 'row': 'STARTGATE', 'pivot_m': [0, 0, 1]})
        self.assertEqual(stamp['scope'], 'whole scene')
        data = json.loads(gzip.decompress(path.read_bytes()))
        self.assertEqual(len(data['objects']), 4)
        self.assertNotIn('objects_list', stamp)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            snapshots.scene_snapshot(name='before')
        scoped = snapshots.scene_snapshot(group='house', box=[0, -1, 2, 1], name='houses')
        self.assertEqual((scoped['objects'], scoped['scope_objects'], scoped['scope']), (4, 2, 'group house + box 0,-1,2,1'))

    def test_validation_happens_before_the_scene_is_read(self):
        with patch.object(snapshots.groups, 'snapshot', side_effect=AssertionError('read')):
            for kwargs, message in ((dict(name='a b'), 'name must be'), (dict(box=[1, 0, 0, 1]), 'x0 < x1'),
                                    (dict(box=[0, 1]), 'four finite'), (dict(group='a b'), 'group must be')):
                with self.assertRaisesRegex(ValueError, message):
                    snapshots.scene_snapshot(**kwargs)
            with self.assertRaisesRegex(ValueError, 'position_m must be 0..100'):
                snapshots.scene_diff('x', position_m=-1)
            with self.assertRaisesRegex(ValueError, 'examples must be'):
                snapshots.scene_diff('x', examples=1000)

    def test_unchanged_scene_gives_clean_verdicts(self):
        snapshots.scene_snapshot(name='s')
        diff = snapshots.scene_diff('s')
        self.assertEqual((diff['unchanged'], diff['changed'], diff['added'], diff['removed']), (4, 0, 0, 0))
        self.assertTrue(diff['same_session'] and diff['gate_order_unchanged'] and diff['gate_transforms_unchanged']
                        and diff['start_gate_unchanged'])
        self.assertEqual(diff['matched_by'], {'actor': 4, 'position': 0})
        self.assertEqual(diff['verdicts'][1:4], ['gate order: unchanged (3 -> 3 gates)',
                                                 'gate transforms: unchanged within 0.01 m / 0.1 deg / 0.001 scale',
                                                 'start gate: unchanged'])
        self.assertTrue(Path(diff['diff_file']).is_file())
        self.assertEqual(diff['warnings'], [])

    def test_same_session_counts_every_kind_of_change(self):
        snapshots.scene_snapshot(name='s')
        now = copy.deepcopy(SCENE)
        now[0]['xf'][0] += 250              # moved 2.5 m
        now[1]['xf'][4] = 30                # rotated
        now[1]['group'] = 'shed'            # and regrouped
        now[2]['xf'][6:] = [2, 2, 2]        # scaled
        del now[3]                          # removed
        now.append(obj('a9', 'house.c', (700, 0, 0)))  # added
        self.scene.objects = now
        diff = snapshots.scene_diff('s', examples=2)
        counts = {k: diff[k] for k in ('unchanged', 'changed', 'added', 'removed', 'moved', 'rotated', 'scaled',
                                       'regrouped', 'row_changed')}
        self.assertEqual(counts, {'unchanged': 0, 'changed': 3, 'added': 1, 'removed': 1, 'moved': 1, 'rotated': 1,
                                  'scaled': 1, 'regrouped': 1, 'row_changed': 0})
        self.assertEqual(len(diff['examples']), 2)
        self.assertTrue(diff['examples_truncated'])
        self.assertEqual(diff['examples'][0]['delta_m'], [2.5, 0, 0])
        self.assertEqual(diff['examples'][1]['change'], 'rotated+regrouped')
        self.assertEqual(diff['examples'][1]['new_group'], 'shed')
        full = json.loads(gzip.decompress(Path(diff['diff_file']).read_bytes()))
        self.assertEqual(len(full['changes']), 5)

    def test_other_session_matches_by_row_and_position_and_warns(self):
        snapshots.scene_snapshot(name='s')
        now = [dict(copy.deepcopy(o), actor='f' + o['actor']) for o in SCENE]
        now[2]['xf'][0] += 500  # moved beyond tolerance: removed + added without actor ids
        self.scene.objects, self.scene.session = now, 's2'
        self.scene.gates = [dict(g, actor='x' + g['actor']) for g in GATES]
        diff = snapshots.scene_diff('s')
        self.assertFalse(diff['same_session'])
        self.assertEqual(diff['matched_by'], {'actor': 0, 'position': 3})
        self.assertEqual((diff['unchanged'], diff['added'], diff['removed']), (3, 1, 1))
        self.assertIn('session differs', diff['warnings'][0])
        self.assertTrue(diff['gate_order_unchanged'])  # by row + position, not actor id

    def test_gate_changes_are_named(self):
        snapshots.scene_snapshot(name='s')
        moved = copy.deepcopy(GATES)
        moved[2]['pivot'][2] += 1
        self.scene.gates = moved
        diff = snapshots.scene_diff('s')
        self.assertTrue(diff['gate_order_unchanged'])
        self.assertFalse(diff['gate_transforms_unchanged'])
        self.assertEqual(diff['gate_changed_indices'], [2])
        self.assertTrue(diff['start_gate_unchanged'])
        self.scene.gates = [GATES[1], GATES[0], GATES[2]]
        diff = snapshots.scene_diff('s')
        self.assertFalse(diff['gate_order_unchanged'])
        self.assertFalse(diff['start_gate_unchanged'])
        self.assertIn('gate order: CHANGED', diff['verdicts'][1])

    def test_scope_moves_out_of_the_box_are_moves_not_removals(self):
        snapshots.scene_snapshot(name='s')
        now = copy.deepcopy(SCENE)
        now[0]['xf'][0] += 10000
        self.scene.objects = now
        diff = snapshots.scene_diff('s', box=[0, -1, 4, 1])
        self.assertEqual((diff['objects_before'], diff['moved'], diff['removed'], diff['added']), (3, 1, 0, 0))

    def test_scoped_delete_is_removed_not_matched_to_a_pre_existing_out_of_scope_twin(self):
        twin = obj('z1', 'shed', (300, 0, 0))  # same row and position as a2, outside the 'house' scope
        self.scene.objects = SCENE + [twin]
        snapshots.scene_snapshot(group='house', name='houses')
        self.scene.objects = [o for o in SCENE if o['actor'] != 'a2'] + [twin]
        diff = snapshots.scene_diff('houses')
        self.assertEqual((diff['removed'], diff['regrouped'], diff['changed'], diff['added']), (1, 0, 0, 0))
        self.assertEqual(diff['examples'][0]['change'], 'removed')

    def test_file_name_without_a_folder_resolves_in_the_snapshot_dir(self):
        snapshots.scene_snapshot(name='s')
        diff = snapshots.scene_diff('s.json.gz')
        self.assertEqual(diff['snapshot'], 's')

    def test_refuses_other_map_tampered_or_mismatched_scope(self):
        snapshots.scene_snapshot(name='s')
        snapshots.scene_snapshot(group='tree', name='trees')
        self.scene.folder = 'Map_B'
        with self.assertRaisesRegex(ValueError, 'snapshot is of map Map_A .* refusing to compare'):
            snapshots.scene_diff(str(self.root / 'UncrashedMCP.snapshots' / 's.json.gz'))
        self.scene.folder = 'Map_A'
        with self.assertRaisesRegex(ValueError, 'covers only group tree'):
            snapshots.scene_diff('trees', group='house')
        path = self.root / 'UncrashedMCP.snapshots' / 's.json.gz'
        data = json.loads(gzip.decompress(path.read_bytes()))
        data['objects'][0]['xf'][0] += 1
        path.write_bytes(gzip.compress(json.dumps(data).encode()))
        with self.assertRaisesRegex(ValueError, 'do not match its stamp'):
            snapshots.scene_diff('s')
        with self.assertRaisesRegex(ValueError, "no snapshot 'missing'"):
            snapshots.scene_diff('missing')

    def test_map_change_during_the_read_is_refused(self):
        calls = iter([{'folder': 'Map_A', 'base': 'Base_1'}, {'folder': 'Map_B', 'base': 'Base_1'}])
        with patch.object(snapshots.agent, 'map_info', lambda: next(calls)):
            with self.assertRaisesRegex(RuntimeError, 'open map changed'):
                snapshots.scene_snapshot(name='x')
        self.assertFalse((self.root / 'UncrashedMCP.snapshots' / 'x.json.gz').exists())


class RollbackTests(unittest.TestCase):
    def test_rollback_line_is_parsed(self):
        self.assertIsNone(groups.rollback_complete({}))
        self.assertIsNone(groups.rollback_complete({'native_result': 'C 0 1a\nend changed=1 complete=1\n'}))
        self.assertTrue(groups.rollback_complete({'native_result': 'T 0 1a\nrollback complete=1\n'}))
        self.assertFalse(groups.rollback_complete({'native_result': 'T 0 1a\nrollback complete=0\n'}))

    def test_edit_history_reports_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / groups.HISTORY
            history.mkdir()
            (history / ('a' * 32 + '.json')).write_text(json.dumps({'id': 'a' * 32, 'status': 'rolled_back',
                                                                    'native_result': 'rollback complete=1\n'}))
            with patch.object(groups.managed, 'context', return_value=({}, Path(tmp))):
                entry = groups.edit_history()['transactions'][0]
        self.assertIs(entry['rollback_complete'], True)


if __name__ == '__main__':
    unittest.main()
