"""Pure parsing, selection, validation and assessment of native audit facts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from uncrashed_mcp import environment as env
from uncrashed_mcp.collide import Rows

ROWS = Rows({
    'House': ('House', 'Big structure', ('building',)),
    'Tree': ('pine Tree', 'Foliage', ('tree',)),
    'Beam': ('Beam01', 'Wall', ('structure',)),
    'Car': ('Camaro', 'Big_assets', ('vehicle',)),
})

HEADER = 'actor id=a row=House basis=mesh_obb min=-100,-200,-500 max=100,200,500 rotation=0,37,0\n'
FOUNDATION = ('foundation samples=20 supported=12 unsupported=8 penetrating=0 missing=2 '
              'max_gap=675 max_penetration=0 corner_mask=12 spacing=75,50\n')
ENVIRONMENT = ('environment terrain_available=1 terrain_samples=20 collision_samples=20 terrain_basis=collision_underside '
               'penetration_samples=3 max_depth=125 tree_hits=1 canopy_candidates=2 spacing=75,50\n')
SCAN = 'scan checked=1 failed=0 terrain_available=1 foliage_instances=706 foliage_truncated=0 query_truncated=0\n'


class BoundsTests(unittest.TestCase):
    def test_actor_address_validation(self):
        self.assertEqual(env.actor_id('0x00Ab'), 'ab')
        for bad in ('abc\nclear', 'a b', 'xyz', '', '-1', 'a'*17):
            with self.assertRaises(ValueError):
                env.actor_id(bad)

    @patch.object(env.agent, 'run')
    def test_typed_bounds_rotated_mesh(self, run):
        run.return_value = ['bounds a lo=-100,-200,-500 hi=100,200,500',
                            'actor a row=House class=Actor loc=0,0,0 rot=0,37,0 scale=1,1,1',
                            'obb center=0,0,0 half=100,200,500 rotation=0,37,0']
        result = env.get_object_bounds('a')
        self.assertEqual(result['aabb']['size'], [2,4,10])
        self.assertEqual(result['obb']['rotation_deg'], [0,37,0])
        self.assertEqual(result['obb']['half_extents'], [1,2,5])
        self.assertEqual(result['transform']['yaw'], 37)

    @patch.object(env.agent, 'run', return_value=['bounds a lo=0,0,0 hi=100,100,100'])
    def test_blueprint_has_no_invented_obb(self, run):
        self.assertIsNone(env.get_object_bounds('a')['obb'])

    @patch.object(env.agent, 'run', return_value=['worldbounds scope=playable available=0 actors=0 truncated=0'])
    def test_unknown_playable_boundary(self, run):
        result = env.get_world_bounds('playable')
        self.assertFalse(result['available'])
        self.assertIsNone(result['aabb'])

    @patch.object(env.agent, 'run', return_value=['worldbounds scope=terrain available=1 actors=1 truncated=0',
                                                'aabb min=-1000,-500,-200 max=2000,800,300'])
    def test_terrain_bounds_are_not_playable_bounds(self, run):
        result = env.get_world_bounds('terrain')
        self.assertEqual(result['aabb']['min'], [-10,-5,-2])
        self.assertIn('not a playable boundary', result['meaning'])

    def test_region_validation(self):
        self.assertIsNone(env.validate_region())
        self.assertEqual(env.validate_region(-1,-2,3,4), (-1,-2,3,4))
        for region in ((1,None,2,3), (2,1,1,3), (1,1,float('nan'),2), (1,1,1,2)):
            with self.assertRaises(ValueError):
                env.validate_region(*region)


class AuditTests(unittest.TestCase):
    def test_partial_foundation_below_zero(self):
        text = HEADER + 'support_point x=100 y=200 bottom=-500 gap=675 surface=landscape\n' + FOUNDATION + 'end a\n' + SCAN
        record = env.parse_audit(text, 'foundation')['records'][0]
        env.assess(record, 'foundation')
        self.assertEqual(record['foundation']['support_coverage'], .6)
        self.assertEqual(record['foundation']['max_known_gap_m'], 6.75)
        self.assertEqual(record['foundation']['missing'], 2)
        self.assertEqual(len(record['foundation']['unsupported_corners']), 2)
        self.assertIn('incomplete_footprint_support', record['issues'])
        self.assertEqual(record['points'][0]['bottom'], -5)

    def test_unknown_support_is_not_zero_gap(self):
        text = HEADER + 'support_point x=0 y=0 bottom=-500 gap=-1e30 surface=unknown\n' + FOUNDATION.replace('max_gap=675', 'max_gap=-1') + 'end a\n' + SCAN
        record = env.parse_audit(text, 'foundation')['records'][0]
        self.assertIsNone(record['points'][0]['gap_m'])
        self.assertIsNone(record['foundation']['max_known_gap_m'])

    def test_environment_keeps_collision_and_canopy_distinct(self):
        text = HEADER + 'tree_collision actor=b component=c instance=58\n' + \
               'canopy component=c instance=58 min=-100,-100,-300 max=100,100,900\n' + \
               'terrain_point x=0 y=0 bottom=-500 terrain=-375 depth=125\n' + ENVIRONMENT + 'end a\n' + SCAN
        record = env.parse_audit(text, 'environment')['records'][0]
        env.assess(record, 'environment')
        self.assertEqual(record['environment']['max_terrain_penetration_m'], 1.25)
        self.assertEqual(record['tree_collisions'][0]['instance'], 58)
        self.assertEqual(set(record['issues']), {'terrain_penetration','foliage_collision_candidate','canopy_bounds_overlap_warning'})

    def test_placed_plant_collision_identity(self):
        text = HEADER + 'tree_collision actor=b component=c instance=-1 source=placed\n' + ENVIRONMENT + 'end a\n' + SCAN
        record = env.parse_audit(text, 'environment')['records'][0]
        self.assertEqual(record['tree_collisions'][0]['source'], 'placed')
        self.assertEqual(record['tree_collisions'][0]['instance'], -1)

    def test_tilted_footprint_is_explicitly_approximate(self):
        text = HEADER.replace('rotation=0,37,0', 'rotation=25,37,0') + FOUNDATION + 'end a\n' + SCAN
        record = env.parse_audit(text, 'foundation')['records'][0]
        env.assess(record, 'foundation')
        self.assertIn('tilted_footprint_approximation', record['issues'])

    def test_unsampled_terrain_is_unknown(self):
        record = env.parse_audit(HEADER + ENVIRONMENT.replace('terrain_samples=20','terrain_samples=0').replace('penetration_samples=3','penetration_samples=0').replace('tree_hits=1','tree_hits=0').replace('canopy_candidates=2','canopy_candidates=0') + 'end a\n' + SCAN, 'environment')['records'][0]
        env.assess(record, 'environment')
        self.assertEqual(record['status'], 'unknown')

    def test_rejects_incomplete_and_mismatched_native_reports(self):
        for text in (HEADER+FOUNDATION+SCAN, HEADER+'end a\n'+SCAN,
                     HEADER+FOUNDATION+'end b\n'+SCAN,
                     HEADER+FOUNDATION+'end a\n'+SCAN.replace('checked=1','checked=2')):
            with self.assertRaises(ValueError):
                env.parse_audit(text, 'foundation')

    def test_inventory_and_selection_use_world_extents_not_pivots(self):
        inventory = env.parse_inventory('O a House -100 -100 -500 100 100 500\n'
                                        'O b Tree 0 0 0 100 100 900\n'
                                        'O c Beam 1000 0 0 1100 100 100\n')
        self.assertEqual(len(env.select_subjects(inventory,ROWS,'foundation')), 1)
        self.assertEqual(len(env.select_subjects(inventory,ROWS,'foundation',include_structural=True)), 2)
        # House centre outside region, footprint intersects: still included.
        self.assertEqual(len(env.select_subjects(inventory,ROWS,'foundation',region=(.5,.5,2,2))), 1)
        self.assertEqual(env.select_subjects(inventory,ROWS,'foundation',actors=['c'])[0]['actor'], 'c')
        with self.assertRaises(ValueError):
            env.select_subjects(inventory,ROWS,'foundation',actors=['dead'])

    @patch.object(env.agent, 'run')
    def test_invalid_parameters_do_not_contact_game(self, run):
        for kwargs in ({'actors':['a\nclear']}, {'spacing':0}, {'tolerance':-1}, {'max_drop':.1}, {'top':0}, {'actors':[]}):
            with self.assertRaises(ValueError):
                env.audit(None,'foundation',**kwargs)
        run.assert_not_called()

    @patch.object(env.Rows, 'from_catalog', return_value=ROWS)
    def test_audit_driver_supplies_plant_exclusions_and_persists_report(self, rows):
        def run(command, *args):
            p = command.split()
            def unix(path):
                return Path(path[2:].replace('\\','/'))
            if p[0] == 'inventory':
                unix(p[1]).write_text('O a House -100 -200 -500 100 200 500\nO b Tree 0 0 0 100 100 1000\n')
            elif p[0] == 'audit':
                self.assertEqual(unix(p[1]).read_text(), 'P b\nA a\n')
                unix(p[2]).write_text(HEADER+FOUNDATION+'end a\n'+SCAN)
            return []
        with TemporaryDirectory() as tmp, patch.object(env,'REPORT_DIR',Path(tmp)), patch.object(env.agent,'run',side_effect=run):
            result = env.audit(None,'foundation')
            saved = json.loads(Path(result['report_json']).read_text())
        self.assertEqual(result['checked'],1)
        self.assertEqual(saved['records'][0]['foundation']['support_coverage'],.6)
        self.assertNotIn('records',result)


if __name__ == '__main__':
    unittest.main()
