import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uncrashed_mcp import managed


class ManagedTests(unittest.TestCase):
    def test_zone_sat_rejects_rotated_aabb_false_positive(self):
        z={'center':[0,0,0],'size':[10,1,2],'yaw':45}
        self.assertTrue(managed.zone_overlap(z,[-.1,-.1,-.1],[.1,.1,.1]))
        self.assertFalse(managed.zone_overlap(z,[-3.5,3.4,-.1],[-3.4,3.5,.1]))
        self.assertFalse(managed.zone_overlap(z,[0,0,10],[1,1,11]))

    def test_zone_text_units_and_validation(self):
        state={'base':'LongRange1','zones':{'court':{'center':[1,2,3],'size':[4,6,8],'yaw':30}}}
        self.assertEqual(managed.zones_text(state),'UNCRASHED_ZONES_V1 LongRange1\nZ court 100 200 300 200 300 400 30\n')
        state['zones']['court']['size'][0]=float('nan')
        with self.assertRaises(ValueError):
            managed.zones_text(state)

    def test_names_reject_injection_reserved_and_nonascii(self):
        for name in ['None','none','Group','-', '.', 'hello world','x\nclear','../x','ä']:
            with self.subTest(name=name),self.assertRaises(ValueError):
                managed.tag(name)

    def test_state_base_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            managed.write_state(root,{'version':1,'base':'A','zones':{},'assemblies':{}})
            with self.assertRaisesRegex(ValueError,'base mismatch'):
                managed.read_state({'base':'B'},root)

    def test_save_as_copies_intent_not_undo_handles(self):
        with tempfile.TemporaryDirectory() as d:
            source,target=Path(d)/'a',Path(d)/'b'
            state={'version':1,'base':'LongRange1','zones':{},'assemblies':{'house':{'roles':{'foot':'foundation'}}}}
            managed.write_state(source,state)
            (source/'UncrashedMCP.history').mkdir();(source/'UncrashedMCP.history'/'old.json').write_text('{}')
            managed.copy_metadata(source,target,'LongRange1')
            self.assertEqual(managed.read_state({'base':'LongRange1'},target),state)
            self.assertTrue((target/managed.ZONES_NAME).is_file())
            self.assertFalse((target/'UncrashedMCP.history').exists())
            with self.assertRaisesRegex(ValueError,'already has'):
                managed.copy_metadata(source,target,'LongRange1')

    def test_zone_preview_has_no_writes_or_native_mutations(self):
        with tempfile.TemporaryDirectory() as d,patch.object(managed,'context',return_value=({'base':'LongRange1'},Path(d))),patch.object(managed,'foliage_inventory',return_value={'foliage':[],'truncated':False}),patch.object(managed.agent,'run') as run:
            result=managed.clearing_zones('upsert','court',[1,2,3],[4,5,6])
            self.assertTrue(result['preview']);self.assertEqual(list(Path(d).iterdir()),[]);run.assert_not_called()

    def test_zone_apply_failure_restores_previous_configuration(self):
        with tempfile.TemporaryDirectory() as d,patch.object(managed,'context',return_value=({'base':'LongRange1'},Path(d))),patch.object(managed,'foliage_inventory',return_value={'foliage':[],'truncated':False}),patch.object(managed.agent,'run',side_effect=[RuntimeError('partial mask'),['zones count=0 masked=0']]):
            with self.assertRaisesRegex(RuntimeError,'partial mask'):
                managed.clearing_zones('upsert','court',[1,2,3],[4,5,6],preview=False)
            self.assertEqual(json.loads((Path(d)/managed.STATE_NAME).read_text())['zones'],{})
            self.assertEqual((Path(d)/managed.ZONES_NAME).read_text(),'UNCRASHED_ZONES_V1 LongRange1\n')

    def test_zone_union_is_persisted_and_remove_keeps_other_zone(self):
        with tempfile.TemporaryDirectory() as d,patch.object(managed,'context',return_value=({'base':'LongRange1'},Path(d))),patch.object(managed,'foliage_inventory',return_value={'foliage':[],'truncated':False}),patch.object(managed.agent,'run',return_value=[]):
            managed.clearing_zones('upsert','a',[0,0,0],[10,10,10],preview=False)
            managed.clearing_zones('upsert','b',[2,0,0],[10,10,10],preview=False)
            managed.clearing_zones('remove','a',preview=False)
            text=(Path(d)/managed.ZONES_NAME).read_text()
            self.assertNotIn('Z a ',text);self.assertIn('Z b ',text)

    def test_truncated_preview_cannot_apply(self):
        with tempfile.TemporaryDirectory() as d,patch.object(managed,'context',return_value=({'base':'LongRange1'},Path(d))),patch.object(managed,'foliage_inventory',return_value={'foliage':[],'truncated':True}),patch.object(managed,'write_state') as write:
            with self.assertRaisesRegex(RuntimeError,'preview incomplete'):
                managed.clearing_zones('upsert','a',[0,0,0],[10,10,10],preview=False)
            write.assert_not_called()

    def test_site_parser_retains_masked_state(self):
        text='session token\nT abc 2 1 100 200 300 400 500 600\nend count=1 truncated=0\n'
        result=managed.parse_site(text)
        self.assertEqual(result['foliage'][0]['aabb']['min'],[1,2,3])
        self.assertTrue(result['foliage'][0]['masked'])
        with self.assertRaisesRegex(ValueError,'incomplete'):
            managed.parse_site(text.replace('count=1','count=2'))

    def test_context_requires_existing_saved_file(self):
        with tempfile.TemporaryDirectory() as d,patch.object(managed.agent,'map_info',return_value={'folder':'Demo','base':'BaseBoard','directory':f'Z:{d}/Demo'}):
            with self.assertRaisesRegex(ValueError,'map file missing'):
                managed.context()
            root=Path(d)/'Demo';root.mkdir();(root/'Demo.json').write_text('{}')
            self.assertEqual(managed.context()[1],root)

    def test_atomic_write_preserves_previous_file_on_serialization_error(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);managed.write_state(root,{'ok':1})
            with self.assertRaises(ValueError):
                managed.write_state(root,{'bad':float('nan')})
            self.assertEqual(json.loads((root/managed.STATE_NAME).read_text()),{'ok':1})

    def test_gpu_only_sources_need_explicit_partial_apply(self):
        with tempfile.TemporaryDirectory() as d,patch.object(managed,'context',return_value=({'base':'LongRange1'},Path(d))),patch.object(managed,'foliage_inventory',return_value={'foliage':[],'truncated':False,'unavailable_sources':2}),patch.object(managed,'write_state') as write:
            preview=managed.clearing_zones('upsert','a',[0,0,0],[10,10,10])
            self.assertEqual(preview['unavailable_sources'],2)
            with self.assertRaisesRegex(RuntimeError,'allow_partial'):
                managed.clearing_zones('upsert','a',[0,0,0],[10,10,10],preview=False)
            write.assert_not_called()

    def test_baseboard_grid_not_fake_zero(self):
        with patch.object(managed.environment,'get_world_bounds',return_value={'available':False}),patch.object(managed.agent,'batch') as batch:
            result=managed.terrain_grid([0,0,0],[10,10,10],2)
            self.assertFalse(result['available']);self.assertEqual(result['samples'],[]);batch.assert_not_called()
