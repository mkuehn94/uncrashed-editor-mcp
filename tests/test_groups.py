import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uncrashed_mcp import groups, managed
from uncrashed_mcp.geom import quat_rotate, quat_rotator


def obj(actor='a',group='house',xyz=(100,0,0),rot=(0,0,0)):
    return {'actor':actor,'row':'NewRow_1','group':group,'xf':list(xyz)+list(rot)+[1,1,1]}


class GroupTests(unittest.TestCase):
    def test_full_rotation_composition_preserves_pitch_roll(self):
        original=obj(rot=(23,42,17))
        out=groups.transform_object(original,[0,0,0],[0,0,0],[10,35,-12],2)
        # Rotated local axes match delta * original axes, not Euler addition.
        for axis in ((1,0,0),(0,1,0),(0,0,1)):
            expected=quat_rotate(quat_rotator(10,35,-12),quat_rotate(quat_rotator(23,42,17),axis))
            actual=quat_rotate(quat_rotator(*out['xf'][3:6]),axis)
            for a,b in zip(actual,expected):
                self.assertAlmostEqual(a,b,places=6)
        self.assertEqual(out['xf'][6:],[2,2,2])
        self.assertEqual(original['xf'][6:],[1,1,1])

    def test_rotation_around_pivot_then_translation(self):
        out=groups.transform_object(obj(xyz=(200,100,0)),[100,100,0],[1000,0,50],[0,90,0],1)
        for actual,expected in zip(out['xf'][:3],[1100,200,50]):
            self.assertAlmostEqual(actual,expected)

    def test_snapshot_strict_complete_and_full_float_precision(self):
        s=groups.parse_snapshot('session test\nO abc NewRow_1 house 123.12345 2 3 0 90 10 1 2 3\nend objects=1 failed=0 truncated=0\n')
        self.assertEqual(s['objects'][0]['xf'][0],123.12345)
        for suffix in ('failed=1 truncated=0','failed=0 truncated=1'):
            with self.assertRaisesRegex(ValueError,'incomplete'):
                groups.parse_snapshot('session test\nend objects=0 '+suffix+'\n')

    def test_never_default_to_entire_map(self):
        with self.assertRaisesRegex(ValueError,'never defaults'):
            groups.select([obj()], '')

    def test_prefix_uses_dot_boundary(self):
        objects=[obj('a','house'),obj('b','house.frame'),obj('c','houseboat')]
        self.assertEqual([o['actor'] for o in groups.select(objects,'house',True)],['a','b'])
        self.assertEqual([o['actor'] for o in groups.select(objects,'house')],['a'])

    def test_explicit_subset_cannot_escape_group(self):
        with self.assertRaisesRegex(ValueError,'outside group'):
            groups.select([obj('a'),obj('b','other')],'house',actors=['b'])

    def test_multiset_reload_handles_and_duplicate_geometry(self):
        expected=[obj('a'),obj('b')];current=[obj('c'),obj('d')]
        matched=groups.match_states(expected,current)
        self.assertEqual(set(matched),{'a','b'})
        self.assertEqual({o['actor'] for o in matched.values()},{'c','d'})

    def test_changed_member_refuses_undo(self):
        changed=obj();changed['xf'][0]+=1
        with self.assertRaisesRegex(ValueError,'conflict'):
            groups.match_states([obj()],[changed])

    def test_quaternion_equivalent_rotations_match(self):
        a,b=obj(),obj(rot=(0,360,0))
        self.assertTrue(groups.same_object(a,b))

    def test_preview_never_journals_or_mutates(self):
        with tempfile.TemporaryDirectory() as d,patch.object(managed,'context',return_value=({'folder':'Demo','base':'BaseBoard'},Path(d))),patch.object(groups,'snapshot',return_value={'session':'s','objects':[obj()]}),patch.object(groups,'_apply') as apply:
            result=groups.edit_group('house',delta=[1,2,3])
            self.assertTrue(result['preview']);apply.assert_not_called();self.assertEqual(list(Path(d).iterdir()),[])

    def test_clone_cannot_overwrite_existing_group(self):
        with tempfile.TemporaryDirectory() as d,patch.object(managed,'context',return_value=({'folder':'Demo','base':'BaseBoard'},Path(d))),patch.object(groups,'snapshot',return_value={'session':'s','objects':[obj(),obj('b','copy')]}):
            with self.assertRaisesRegex(ValueError,'already exists'):
                groups.edit_group('house',action='clone',target_group='copy',preview=False)

    def test_snapshot_durable_before_native_edit_and_fault_marked(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);s={'session':'s','objects':[obj()]}
            def run(command,*args):
                files=list((root/groups.HISTORY).glob('*.json'))
                self.assertEqual(len(files),1)
                self.assertEqual(json.loads(files[0].read_text())['status'],'prepared')
                raise RuntimeError('native fault')
            with patch.object(groups.agent,'run',side_effect=run),patch.object(groups,'snapshot',return_value=s):
                with self.assertRaisesRegex(RuntimeError,'rolled_back'):
                    groups._apply({'folder':'Demo','base':'BaseBoard'},root,s,['house'],[groups.edit_line('D',obj())],'delete')
            j=json.loads(next((root/groups.HISTORY).glob('*.json')).read_text())
            self.assertEqual(j['before'],s);self.assertEqual(j['status'],'rolled_back')

    def test_undo_relocates_current_handle_after_reload(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);transaction='a'*32
            before=obj('old',xyz=(100,0,0));after=obj('old',xyz=(200,0,0));now=obj('new',xyz=(200,0,0))
            # Native handles are hex. Replace illustrative names with hex.
            for o,address in [(before,'ab'),(after,'ab'),(now,'cd')]:o['actor']=address
            journal={'version':1,'map':'Demo','base':'BaseBoard','status':'applied','groups':['house'],
                     'before':{'session':'s','objects':[before]},'after':{'session':'s','objects':[after]}}
            managed.atomic_write(root/groups.HISTORY/f'{transaction}.json',json.dumps(journal))
            with patch.object(managed,'context',return_value=({'folder':'Demo','base':'BaseBoard'},root)),patch.object(groups,'snapshot',return_value={'session':'new','objects':[now]}),patch.object(groups,'_apply',return_value={'transaction':'b'*32}) as apply:
                groups.undo_edit(transaction,preview=False)
                commands=apply.call_args.args[4]
                self.assertTrue(commands[0].startswith('T cd NewRow_1 house '))
                self.assertIn('house 100 0 0',commands[0])

    def test_undo_delete_regenerates_catalogue_row(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);transaction='a'*32
            journal={'version':1,'map':'Demo','base':'BaseBoard','status':'applied','groups':['house'],
                     'before':{'session':'s','objects':[obj()]},'after':{'session':'s','objects':[]}}
            managed.atomic_write(root/groups.HISTORY/f'{transaction}.json',json.dumps(journal))
            with patch.object(managed,'context',return_value=({'folder':'Demo','base':'BaseBoard'},root)),patch.object(groups,'snapshot',return_value={'session':'new','objects':[]}),patch.object(groups,'_apply',return_value={'transaction':'b'*32}) as apply:
                groups.undo_edit(transaction,preview=False)
                self.assertTrue(apply.call_args.args[4][0].startswith('C NewRow_1 house '))

    def test_clone_preserves_dot_and_legacy_underscore_subgroups(self):
        objects=[obj('a','site.foot'),obj('b','site_frame'),obj('c','siteboat')]
        with tempfile.TemporaryDirectory() as d,patch.object(managed,'context',return_value=({'folder':'Demo','base':'BaseBoard'},Path(d))),patch.object(groups,'snapshot',return_value={'session':'s','objects':objects}),patch.object(groups,'_apply',return_value={}) as apply:
            result=groups.edit_group('site',action='clone',prefix=True,target_group='copy',preview=False)
            self.assertEqual(result['objects'],2)
            commands=apply.call_args.args[4]
            self.assertIn('copy.foot',commands[0]);self.assertIn('copy_frame',commands[1])
            self.assertEqual(set(apply.call_args.args[3]),{'site.foot','site_frame','copy.foot','copy_frame'})

    def test_invalid_transform_rejected(self):
        for kwargs in (dict(scale=-1),dict(delta=[]),dict(rotation=[0,float('nan'),0]),dict(action='delete',delta=[1,0,0])):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                groups.edit_group('house',**kwargs)
