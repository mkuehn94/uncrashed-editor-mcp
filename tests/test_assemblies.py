import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uncrashed_mcp import assemblies, groups, managed


def obj(a,g):
    return {'actor':a,'group':g,'row':'NewRow_1','xf':[0,0,0,0,0,0,1,1,1]}


def anchor(a='a',basis='collision_vertical_span',anchored=25):
    return {'actor':a,'samples':25,'anchored':anchored,'missing':0,'buried':0,'basis':basis}


class AssemblyTests(unittest.TestCase):
    def test_cantilever_needs_path_not_full_ground_footprint(self):
        objects=[obj('a','foot'),obj('b','frame'),obj('c','roof')]
        roles={'foot':'foundation','frame':'structure','roof':'cantilever'}
        facts=assemblies.analyse(objects,roles,{'a':{'b'},'b':{'a','c'},'c':{'b'}},[anchor()],.5)
        self.assertEqual(facts['disconnected'],[])
        self.assertEqual(facts['anchored_cantilevers'],['c'])

    def test_one_way_bounds_overlap_is_not_contact_edge(self):
        objects=[obj('a','foot'),obj('b','beam')]
        facts=assemblies.analyse(objects,{'foot':'foundation','beam':'cantilever'},{'a':{'b'},'b':set()},[anchor()],.5)
        self.assertEqual(facts['disconnected'],['b'])
        self.assertEqual(facts['edges'],[])

    def test_decoration_cannot_bridge_structure(self):
        objects=[obj('a','foot'),obj('b','plant'),obj('c','beam')]
        contacts={'a':{'b'},'b':{'a','c'},'c':{'b'}}
        facts=assemblies.analyse(objects,{'foot':'foundation','plant':'decoration','beam':'structure'},contacts,[anchor()],.5)
        self.assertEqual(facts['disconnected'],['c'])

    def test_catalogued_plant_cannot_support_even_if_declared_structure(self):
        objects=[obj('a','foot'),obj('b','plant'),obj('c','beam')]
        contacts={'a':{'b'},'b':{'a','c'},'c':{'b'}}
        facts=assemblies.analyse(objects,{'foot':'foundation','plant':'structure','beam':'structure'},contacts,[anchor()],.5,{'b'})
        self.assertEqual(facts['disconnected'],['c'])

    def test_bounds_only_anchor_is_unknown_not_certified_root(self):
        facts=assemblies.analyse([obj('a','foot')],{'foot':'foundation'},{'a':set()},[anchor(basis='bounds_span')],.5)
        self.assertEqual(facts['roots'],[])
        self.assertEqual(facts['unknown_foundations'],['a'])

    def test_inadequate_coverage_no_ground_seed(self):
        facts=assemblies.analyse([obj('a','foot')],{'foot':'foundation'},{'a':set()},[anchor(anchored=10)],.5)
        self.assertEqual(facts['roots'],[])
        self.assertEqual(facts['foundation_warnings'][0]['coverage'],.4)

    def test_foundation_group_requires_explicit_declaration(self):
        with self.assertRaisesRegex(ValueError,'foundation'):
            assemblies.define_assembly('house',{'frame':'structure'})

    def test_definition_preview_does_not_write(self):
        with tempfile.TemporaryDirectory() as d,patch.object(managed,'context',return_value=({'base':'BaseBoard'},Path(d))),patch.object(groups,'snapshot',return_value={'objects':[obj('a','foot')]}):
            result=assemblies.define_assembly('house',{'foot':'foundation'})
            self.assertTrue(result['preview']);self.assertEqual(list(Path(d).iterdir()),[])

    def test_contact_parser_rejects_missing_actor(self):
        with self.assertRaisesRegex(ValueError,'incomplete'):
            assemblies.parse_contacts('A a 0 b\nend count=1\n',{'a','b'})
        contacts,truncated,errors=assemblies.parse_contacts('A a 1 b\nA b 0 a\nend count=2\n',{'a','b'})
        self.assertEqual(truncated,{'a'});self.assertEqual(errors,[])

    def test_saved_definitions_are_inspectable(self):
        with tempfile.TemporaryDirectory() as d,patch.object(managed,'context',return_value=({'base':'BaseBoard'},Path(d))):
            managed.write_state(Path(d),{'version':1,'base':'BaseBoard','zones':{},'assemblies':{'house':{'roles':{'foot':'foundation'}}}})
            self.assertIn('house',assemblies.define_assembly()['assemblies'])
            self.assertEqual(assemblies.define_assembly('house')['roles'],{'foot':'foundation'})

    def test_anchor_parser_counts_missing_basis(self):
        records,truncated=assemblies.parse_anchors('F a samples=25 anchored=25 missing=0 buried=0 basis=collision_vertical_span\nend count=1 truncated=0\n',{'a'})
        self.assertEqual(records[0]['anchored'],25);self.assertFalse(truncated)
