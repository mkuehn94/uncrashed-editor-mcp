import ctypes
import json
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uncrashed_mcp import managed, scenery
from uncrashed_mcp.catalog import load_catalog

NATIVE = Path(__file__).resolve().parents[1]/'native'


def obj(actor, row, group='-', loc=(0, 0, 0), gate=False):
    return {'actor': actor, 'row': row, 'group': group, 'location_m': list(loc), 'gate': gate}


class PureRuleTests(unittest.TestCase):
    def test_channels_names_numbers_and_custom(self):
        self.assertEqual(scenery.channel_ids(None), [2, 5, 6])
        self.assertEqual(scenery.channel_ids(['Pawn', 'game3', 'engine1', '4']), [2, 4, 8, 16])
        self.assertEqual(scenery.channel_names(scenery.mask_of([2, 16])), ['pawn', 'game3'])
        for bad in [['nope'], [32], [-1], [], [True]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                scenery.channel_ids(bad)

    def test_rule_text_units_flags_and_validation(self):
        rules = [scenery.make_rule('object', 'NewRow_5', 'flythrough', None, False, True, [1.5, -2, 0.25]),
                 scenery.make_rule('group', 'bridge', 'off', force=True),
                 scenery.make_rule('row', 'NewRow_9', 'nophysics')]
        self.assertEqual(scenery.collision_text('LongRange1', rules),
                         'UNCRASHED_COLLISION_V1 LongRange1\n'
                         'O flythrough 64 K NewRow_5 150.00 -200.00 25.00 2\n'
                         'G off 0 F bridge\n'
                         'R nophysics 0 - NewRow_9\n')
        for args in [('group', 'None', 'off'), ('row', 'Tree', 'off'), ('group', 'x y', 'off'), ('row', 'NewRow_1', 'fly')]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                scenery.make_rule(*args)
        with self.assertRaises(ValueError):
            scenery.make_rule('object', 'NewRow_1', 'off', location_m=[float('nan'), 0, 0])
        with self.assertRaises(ValueError):
            scenery.collision_text('../x', [])

    def test_precedence_object_group_row_and_gates(self):
        rules = [scenery.make_rule('row', 'NewRow_1', 'off'),
                 scenery.make_rule('group', 'g', 'flythrough'),
                 scenery.make_rule('object', 'NewRow_1', 'on', location_m=[1, 1, 1])]
        self.assertEqual(scenery.governing(rules, obj('a', 'NewRow_1', 'g', (1, 1, 1.01)))['kind'], 'object')
        self.assertEqual(scenery.governing(rules, obj('a', 'NewRow_1', 'g', (1, 1, 1.05)))['kind'], 'group')
        self.assertEqual(scenery.governing(rules, obj('a', 'newrow_1', '-', (9, 9, 9)))['kind'], 'row')
        self.assertIsNone(scenery.governing(rules, obj('a', 'NewRow_1', 'g', (9, 9, 9), gate=True)))
        rules.append(scenery.make_rule('row', 'NewRow_1', 'nophysics'))
        self.assertEqual(scenery.governing(rules, obj('a', 'NewRow_1'))['mode'], 'nophysics')  # later duplicate wins

    def test_group_flythrough_replaces_member_object_rules(self):
        members = [obj('a', 'NewRow_1', 'g', (0, 0, 0)), obj('b', 'NewRow_2', 'g', (5, 0, 0))]
        rules = [scenery.make_rule('object', 'NewRow_1', 'off', location_m=[0, 0, 0]),
                 scenery.make_rule('object', 'NewRow_3', 'off', location_m=[0, 0, 0])]
        out = scenery.update_rules(rules, ('group', 'g'), members, 'flythrough')
        self.assertEqual([(r['kind'], r['key'], r['mode']) for r in out],
                         [('object', 'NewRow_3', 'off'), ('group', 'g', 'flythrough')])
        for m in members:
            self.assertEqual(scenery.governing(out, m)['mode'], 'flythrough')

    def test_on_is_exception_only_where_broader_rule_applies(self):
        rules = [scenery.make_rule('row', 'NewRow_1', 'off'), scenery.make_rule('group', 'g', 'flythrough')]
        members = [obj('a', 'NewRow_1', 'g', (0, 0, 0)), obj('b', 'NewRow_2', 'g', (5, 0, 0))]
        out = scenery.update_rules(rules, ('group', 'g'), members, 'on')
        self.assertEqual([(r['kind'], r['key'], r['mode']) for r in out],
                         [('row', 'NewRow_1', 'off'), ('object', 'NewRow_1', 'on')])
        self.assertEqual(scenery.update_rules(out, ('objects', ''), [members[0]], 'on'),
                         [out[0], out[1]])  # still needed: row rule would apply
        self.assertEqual(scenery.update_rules([out[1]], ('objects', ''), [members[0]], 'on'), [])

    def test_object_selection_skips_gates_unless_forced(self):
        sel = [obj('a', 'NewRow_1', loc=(1, 2, 3)), obj('g', 'NewRow_7', loc=(4, 5, 6), gate=True)]
        out = scenery.update_rules([], ('objects', ''), sel, 'flythrough', ['pawn'])
        self.assertEqual([(r['key'], r['force'], r['channels']) for r in out], [('NewRow_1', False, [2])])
        out = scenery.update_rules([], ('objects', ''), sel, 'off', force=True)
        self.assertEqual([(r['key'], r['force']) for r in out], [('NewRow_1', True), ('NewRow_7', True)])

    def test_flythrough_channel_change_is_a_change(self):
        a = obj('a', 'NewRow_1', 'g')
        out = scenery.update_rules([], ('group', 'g'), [a], 'flythrough')
        out2 = scenery.update_rules(out, ('group', 'g'), [a], 'flythrough', ['pawn', 'game2'])
        self.assertEqual(out2[-1]['channels'], [2, 15])
        self.assertEqual(len(out2), 1)

    def test_parse_native_lines(self):
        text = ('C 1a2b row=NewRow_1 group=g loc=100.00,200.00,-50.00 gate=0 mode=flythrough owner=manual rmask=64 '
                'actor_collision=1 comps=1 truncated=0 sim=0 blocks=0\n'
                'P 1a2b comp=ff class=StaticMeshComponent enabled=1 object=0 profile=Custom sim=0 '
                'r0=2 r1=2 r2=0 r3=2 r4=2 r5=0 r6=0 blocks=0\n'
                'C 3c row=NewRow_2 group=- loc=nan,nan,nan gate=1 mode=on owner=- rmask=0 actor_collision=1 comps=1 '
                'truncated=0 sim=0 blocks=24\n'
                'collision query matched=2 gates=1 blocking=1 actor_collision_off=0 simulating=0 mask=64 shown=2\n')
        data = scenery.parse_lines(text)
        a, b = data['objects']
        self.assertEqual(a['location_m'], [1, 2, -.5])
        self.assertEqual(a['channels'], ['pawn', 'physicsbody', 'vehicle'])
        self.assertEqual(a['components'][0]['responses']['pawn'], 'ignore')
        self.assertEqual(a['components'][0]['responses']['visibility'], 'block')
        self.assertIsNone(b['location_m'])
        self.assertEqual(b['blocks'], ['pawn', 'physicsbody'])
        self.assertEqual(scenery.passable_actors(data), {'1a2b'})
        self.assertEqual(data['summary']['matched'], '2')
        with self.assertRaises(ValueError):
            scenery.parse_lines('C 1 row=x group=- loc=0,0,0 gate=0 mode=on\n')


class CatalogueTests(unittest.TestCase):
    def test_filter_and_lookalikes_stay_noncolliding(self):
        cat = load_catalog()
        non = scenery.collision_filter(cat.assets.values(), 'noncolliding')
        self.assertTrue(non and all(not a.collision for a in non))
        self.assertEqual(len(scenery.collision_filter(cat.assets.values(), 'any')), len(cat.assets))
        with self.assertRaises(ValueError):
            scenery.collision_filter([], 'maybe')
        colliding = next(a for a in cat.assets.values() if a.collision and a.category != 'Decals')
        result = scenery.lookalikes(cat, colliding.id)
        self.assertTrue(all(not cat.get(s['id']).collision for s in result['suggestions']))
        self.assertTrue(result['suggestions'] or 'flythrough' in result['note'])
        self.assertIn('already', scenery.lookalikes(cat, non[0].id)['note'])


@unittest.skipUnless(shutil.which('cc'), 'host C compiler unavailable')
class NativeAgreementTests(unittest.TestCase):
    """Python rule text parsed and matched by the exact native COLL_PURE code."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root/'w.c').write_text(
            '#define COLL_PURE_ONLY\n#include "collision.h"\n'
            'static struct coll_rule R[COLL_RULE_CAP]; static int B[COLL_BUCKETS], N;\n'
            'void reset(void){N=0;coll_index(R,0,B);}\n'
            'int add(const char*l){if(!coll_rule_parse(l,&R[N]))return 0;N++;coll_index(R,N,B);return 1;}\n'
            'int match(const char*row,const char*g,float x,float y,float z,int has,int gate){float l[3]={x,y,z};'
            'return coll_match(R,B,row,g,has?l:0,gate);}\n'
            'int channels(const char*t,unsigned*m){return coll_channels_parse(t,m);}\n')
        subprocess.run(['cc', '-shared', '-fPIC', '-O2', '-Wall', '-Werror', '-I', str(NATIVE), str(root/'w.c'),
                        '-o', str(root/'w.so'), '-lm'], check=True, capture_output=True)
        lib = cls.lib = ctypes.CDLL(str(root/'w.so'))
        lib.add.argtypes = [ctypes.c_char_p]
        lib.match.argtypes = [ctypes.c_char_p, ctypes.c_char_p] + [ctypes.c_float]*3 + [ctypes.c_int]*2
        lib.channels.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint)]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def load(self, rules):
        self.lib.reset()
        for line in scenery.collision_text('B', rules).splitlines()[1:]:
            self.assertEqual(self.lib.add(line.encode()), 1, line)

    def test_native_rejects_malformed_lines(self):
        self.lib.reset()
        for line in ['O off 0 - NewRow_1 0 0 0 99', 'G fly 0 - g', 'R off 0 - Tree', 'G off 0 X g', 'G off 0 - None',
                     'O flythrough 0 - NewRow_1 0 0 0 2', 'R off 0 - NewRow_1 extra', 'Z off 0 - g']:
            with self.subTest(line=line):
                self.assertEqual(self.lib.add(line.encode()), 0)

    def test_channel_parser_agrees(self):
        m = ctypes.c_uint()
        for text, ids in [('2,5,6', [2, 5, 6]), ('pawn,game3,engine1', [2, 16, 8]), ('Vehicle', [6])]:
            self.assertEqual(self.lib.channels(text.encode(), ctypes.byref(m)), 1)
            self.assertEqual(m.value, scenery.mask_of(ids))
        for bad in ['', 'x', '32', 'game19', '2,,5']:
            self.assertEqual(self.lib.channels(bad.encode(), ctypes.byref(m)), 0, bad)

    def test_far_object_rules_keep_two_cm_precision(self):
        loc = [15234.567, -98765.432, 1234.5]  # beyond 10 km: '%g' would be 10 cm off
        rule = scenery.make_rule('object', 'NewRow_1', 'off', location_m=loc)
        self.assertIn('1523456.70 -9876543.20 123450.00', scenery.rule_line(rule))
        self.load([rule])
        cm = [v*100 for v in loc]
        self.assertEqual(self.lib.match(b'NewRow_1', b'-', *cm, 1, 0), 0)
        self.assertEqual(self.lib.match(b'NewRow_1', b'-', cm[0]+1.5, cm[1], cm[2], 1, 0), 0)
        self.assertEqual(self.lib.match(b'NewRow_1', b'-', cm[0]+3, cm[1], cm[2], 1, 0), -1)

    def test_random_rules_match_like_python(self):
        rnd = random.Random(7)
        rows, groups = [f'NewRow_{i}' for i in range(4)], ['g1', 'g2', '-']
        for trial in range(30):
            rules = []
            for _ in range(rnd.randint(0, 12)):
                kind = rnd.choice(['object', 'group', 'row'])
                key = rnd.choice(groups[:2]) if kind == 'group' else rnd.choice(rows)
                rules.append(scenery.make_rule(kind, key, rnd.choice(scenery.MODES), force=rnd.random() < .3,
                                               location_m=[rnd.choice([0, .995, 1, 2.99]), rnd.choice([0, 1]), 0]))
            self.load(rules)
            for _ in range(40):
                o = obj('x', rnd.choice(rows), rnd.choice(groups),
                        (rnd.choice([0, .99, 1, 1.012, 3]), rnd.choice([0, 1, .985]), rnd.choice([0, .01])),
                        rnd.random() < .2)
                index = self.lib.match(o['row'].encode(), o['group'].encode(), *[v*100 for v in o['location_m']], 1, int(o['gate']))
                expect = scenery.governing(rules, o)
                with self.subTest(trial=trial, obj=o):
                    self.assertEqual(rules[index] if index >= 0 else None, expect)


class LiveFlowTests(unittest.TestCase):
    PREVIEW = ('C a1 row=NewRow_1 group=g loc=100,0,0 gate=0 mode=on owner=- rmask=0 to=flythrough change=1 ok=1\n'
               'C a2 row=NewRow_7 group=g loc=0,0,0 gate=1 mode=on owner=- rmask=0 to=on change=0 ok=1 skipped=gate\n'
               'collision mode=flythrough preview=1 matched=2 changed=1 unchanged=0 gates=1 gates_forced=0 failed=0 '
               'physics_bound=0 mask=64 keepbody=0 records=0\n')
    QUERY = ('C a1 row=NewRow_1 group=g loc=100,0,0 gate=0 mode=flythrough owner=rule rmask=64 actor_collision=1 comps=0 '
             'truncated=0 sim=0 blocks=0\ncollision query matched=1 gates=0 blocking=0 actor_collision_off=0 simulating=0 mask=64 shown=1\n')

    def fake(self, calls, fail_refresh=False, preview=None, query=None, probe=None, fail_query=False):
        def run(command, timeout=30):
            calls.append(command)
            if command.startswith('collision refresh') and fail_refresh and len([c for c in calls if 'refresh' in c]) == 1:
                raise scenery.agent.AgentError('engine refused')
            out = next((t[4:] for t in command.split() if t.startswith('out=')), None)
            if out:
                if ' query all ' in command and probe is not None:
                    text = probe
                elif ' query ' in command:
                    if fail_query:
                        raise scenery.agent.AgentError('query failed')
                    text = query or self.QUERY
                else:
                    text = preview or self.PREVIEW
                Path(out).write_text(text)
            return ['ok']
        return run

    @staticmethod
    def lines(n, mode='on', owner='-', to='flythrough', kb=0, start=0):
        rows = [f'C {start+i+1:x} row=NewRow_1 group=g loc={i},0,0 gate=0 mode={mode} owner={owner} rmask=64 kb={kb} '
                f'to={to} change=1 ok=1' for i in range(n)]
        return '\n'.join(rows) + '\ncollision mode=x preview=1 matched=%d\n' % n

    @staticmethod
    def brief(ids, mode):
        return ''.join(f'C {a} row=NewRow_1 group=g loc=0,0,0 gate=0 mode={mode} owner=rule rmask=64 kb=0 actor_collision=1\n'
                       for a in ids) + 'collision query matched=%d brief=1\n' % len(ids)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root/'m.json').write_text('{}')
        self.info = {'base': 'LongRange1', 'folder': 'm'}
        self.patches = [patch.object(scenery.agent, '_windows_path', side_effect=str),
                        patch.object(managed, 'context', return_value=(self.info, self.root))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_preview_writes_nothing_and_only_previews(self):
        calls = []
        with patch.object(scenery.agent, 'run', side_effect=self.fake(calls)):
            result = scenery.set_collision('flythrough', group='g')
        self.assertTrue(result['preview'])
        self.assertEqual(result['rules_after'], 1)
        self.assertEqual(result['gates_skipped'], 1)
        self.assertEqual(len(calls), 1)
        self.assertIn(' preview', calls[0])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['m.json'])

    def test_apply_persists_journals_and_undo_restores(self):
        calls = []
        with patch.object(scenery.agent, 'run', side_effect=self.fake(calls)):
            result = scenery.set_collision('flythrough', group='g', preview=False)
            self.assertTrue(result['verified'])
            self.assertEqual((self.root/managed.COLLISION_NAME).read_text(),
                             'UNCRASHED_COLLISION_V1 LongRange1\nG flythrough 64 - g\n')
            state = json.loads((self.root/managed.STATE_NAME).read_text())
            self.assertEqual(state['collision']['rules'][0]['key'], 'g')
            self.assertIn('collision refresh group=g', calls)  # adopts only this selection
            self.assertFalse(any('list=' in c for c in calls))  # group goes to native as a selector
            journal = json.loads(Path(result['journal']).read_text())
            self.assertEqual(journal['status'], 'collision_applied')
            self.assertEqual([o['actor'] for o in journal['before']], ['a1'])  # gate not touched
            self.assertTrue(scenery.undo_collision(result['transaction'])['preview'])
            scenery.undo_collision(result['transaction'], preview=False)
        self.assertEqual((self.root/managed.COLLISION_NAME).read_text(), 'UNCRASHED_COLLISION_V1 LongRange1\n')
        self.assertTrue(any(c.startswith('collision on list=') for c in calls))
        with self.assertRaisesRegex(ValueError, 'not undoable'):
            scenery.undo_collision(result['transaction'])

    def test_failed_refresh_restores_previous_rules(self):
        calls = []
        with patch.object(scenery.agent, 'run', side_effect=self.fake(calls, fail_refresh=True)):
            with self.assertRaisesRegex(RuntimeError, 'previous rules restored'):
                scenery.set_collision('flythrough', group='g', preview=False)
        self.assertEqual((self.root/managed.COLLISION_NAME).read_text(), 'UNCRASHED_COLLISION_V1 LongRange1\n')
        self.assertEqual(json.loads((self.root/managed.STATE_NAME).read_text())['collision']['rules'], [])

    def test_huge_group_uses_selector_not_handle_lists(self):
        calls, n = [], 9000
        ids = [f'{i+1:x}' for i in range(n)]
        run = self.fake(calls, preview=self.lines(n), query=self.brief(ids, 'flythrough'))
        with patch.object(scenery.agent, 'run', side_effect=run):
            result = scenery.set_collision('flythrough', group='g', preview=False, persist=False)
        self.assertTrue(result['verified'])
        self.assertEqual(result['missing_after'], 0)
        self.assertFalse(any('list=' in c for c in calls))
        self.assertTrue(any(c.startswith('collision flythrough group=g channels=2,5,6') for c in calls))
        journal = json.loads(Path(result['journal']).read_text())
        self.assertEqual((journal['status'], journal['selector'], len(journal['before'])), ('collision_applied', 'group=g', n))

    def test_failed_verification_keeps_change_undoable(self):
        calls = []
        with patch.object(scenery.agent, 'run', side_effect=self.fake(calls, fail_query=True)):
            result = scenery.set_collision('flythrough', group='g', preview=False)
        self.assertIsNone(result['verified'])
        self.assertIn('verification query failed', result['verify_error'])
        self.assertEqual(json.loads(Path(result['journal']).read_text())['status'], 'collision_applied')

    def test_live_only_error_does_not_claim_rule_restore(self):
        def run(command, timeout=30):
            out = next((t[4:] for t in command.split() if t.startswith('out=')), None)
            if out:
                Path(out).write_text(self.PREVIEW)
                return ['ok']
            raise scenery.agent.AgentError('engine refused')
        with patch.object(scenery.agent, 'run', side_effect=run):
            with self.assertRaises(RuntimeError) as cm:
                scenery.set_collision('off', group='g', preview=False, persist=False)
        self.assertNotIn('previous rules', str(cm.exception))
        self.assertIn('originals retained', str(cm.exception))

    def test_undo_replays_live_subset_with_keepbody(self):
        calls = []
        before = self.lines(3, mode='flythrough', owner='manual', to='off', kb=1)
        with patch.object(scenery.agent, 'run', side_effect=self.fake(calls, preview=before, query=self.brief(['1', '2', '3'], 'off'))):
            result = scenery.set_collision('off', group='g', preview=False, persist=False)
        journal = json.loads(Path(result['journal']).read_text())
        self.assertTrue(all(o['keep_body'] for o in journal['before']))
        calls.clear()
        with patch.object(scenery.agent, 'run', side_effect=self.fake(calls, probe=self.brief(['1', '3'], 'off'))):
            plan = scenery.undo_collision(result['transaction'])
            self.assertEqual((plan['actors_live'], plan['actors_missing']), (2, 1))
            scenery.undo_collision(result['transaction'], preview=False)
        replay = [c for c in calls if c.startswith('collision flythrough list=')]
        self.assertEqual(len(replay), 1)
        self.assertIn('keepbody', replay[0])
        self.assertTrue(any(c.startswith('collision refresh list=') for c in calls))
        self.assertNotIn('collision refresh', calls)  # live-only: rules untouched, no map-wide re-read

    def test_undo_live_only_with_nothing_loaded_refuses(self):
        calls = []
        with patch.object(scenery.agent, 'run', side_effect=self.fake(calls)):
            result = scenery.set_collision('flythrough', group='g', preview=False, persist=False)
        with patch.object(scenery.agent, 'run', side_effect=self.fake(calls, probe='collision query matched=0 brief=1\n')):
            self.assertFalse(scenery.undo_collision(result['transaction'])['undoable'])
            with self.assertRaisesRegex(ValueError, 'nothing to undo'):
                scenery.undo_collision(result['transaction'], preview=False)
        self.assertEqual(json.loads(Path(result['journal']).read_text())['status'], 'collision_applied')

    def test_physics_toggles_are_live_only(self):
        with self.assertRaisesRegex(ValueError, 'live-only'):
            scenery.set_collision('physics_on', group='g')
        calls = []
        with patch.object(scenery.agent, 'run', side_effect=self.fake(calls, query=self.brief(['a1'], 'nophysics'))):
            result = scenery.set_collision('physics_off', group='g', persist=False, preview=False)
        self.assertTrue(result['verified'])
        self.assertTrue(calls[0].startswith('collision physics off group=g'))

    def test_brief_lines_are_not_passable(self):
        data = scenery.parse_lines(self.brief(['a1'], 'on'))
        self.assertEqual(scenery.passable_actors(data), set())
        self.assertFalse(data['objects'][0]['keep_body'])

    def test_selector_rules(self):
        with self.assertRaisesRegex(ValueError, 'exactly one selector'):
            scenery.set_collision('off')
        with self.assertRaisesRegex(ValueError, 'exactly one selector'):
            scenery.set_collision('off', group='g', row='NewRow_1')
        with self.assertRaises(ValueError):
            scenery.set_collision('sideways', group='g')

    def test_save_as_carries_collision_rules(self):
        source, target = self.root/'a', self.root/'b'
        state = {'version': 1, 'base': 'LongRange1', 'zones': {}, 'assemblies': {},
                 'collision': {'version': 1, 'rules': [scenery.make_rule('group', 'g', 'off')]}}
        managed.write_state(source, state)
        managed.copy_metadata(source, target, 'LongRange1')
        self.assertEqual((target/managed.COLLISION_NAME).read_text(), 'UNCRASHED_COLLISION_V1 LongRange1\nG off 0 - g\n')
        (target/managed.STATE_NAME).unlink()
        (target/managed.ZONES_NAME).unlink()
        with self.assertRaisesRegex(ValueError, 'already has'):
            managed.copy_metadata(source, target, 'LongRange1')


if __name__ == '__main__':
    unittest.main()


@unittest.skipUnless(shutil.which('cc'), 'host C compiler unavailable')
class RecordTableTerminates(unittest.TestCase):
    """Regression: the static record index starts zeroed (looks full). Inserting before it was rebuilt
    spun the game thread forever (fresh DLL + persisted rule, first apply from the background poll)."""

    def test_zeroed_table_insert_returns_instead_of_spinning(self):
        src = (NATIVE / 'collision.h').read_text()
        block = src[src.index('/* ---- COLL_TABLE begin'):src.index('/* ---- COLL_TABLE end')]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 't.c').write_text(
                '#include <stdint.h>\n#define COLL_REC_CAP 64\n'
                'struct coll_rec { uint8_t *actor; };\n'
                'static struct coll_rec coll_recs[COLL_REC_CAP];\n'
                'static int coll_nrec, coll_table[COLL_REC_CAP * 2 + 1];\nstatic int coll_table_ready;\n'
                + block +
                'int main(void) {\n'
                '    uint8_t *a = (uint8_t *)0x123450;\n'
                '    if (coll_table_put(a, 0)) return 2;          /* zeroed = full: must refuse, not spin */\n'
                '    coll_recs[0].actor = a; coll_nrec = 1; coll_rebuild();\n'
                '    if (!coll_table_ready) return 3;\n'
                '    if (!coll_table_put((uint8_t *)0x223450, 1)) return 4;\n'
                '    return 0;\n}\n')
            subprocess.run(['cc', '-O2', '-Wall', '-Wno-unused-function', '-o', str(root / 't'), str(root / 't.c')],
                           check=True, capture_output=True, text=True)
            result = subprocess.run([str(root / 't')], timeout=10)
            self.assertEqual(result.returncode, 0)
