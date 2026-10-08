"""The collision report's logic on hand-written agent output (no game needed)."""
import tempfile
import unittest
from pathlib import Path

from uncrashed_mcp.collide import Rows, analyse, brief, render_html, scan_commands

ROWS = Rows({
    "NewRow_1": ("container green", "Big_assets", ()),
    "NewRow_71": ("Rock 05", "Big structure", ()),
    "NewRow_33": ("Beam01", "Wall", ()),
    "NewRow_157": ("Scafold 01a", "Wall", ()),
    "NewRow_165": ("Air Balloon Random E", "Big_assets", ()),
    "NewRow_103": ("Camaro", "Big_assets", ("vehicle",)),
    "NewRow_180": ("Base Ground 0-1", "Wall", ("ground",)),
    "NewRow_12": ("Bldg SM B", "Big structure", ("building",)),
    "NewRow_222": ("LED SCREEN", "Interractive", ()),
    "NewRow_279": ("pine03", "Foliage", ("tree",)),
})


def report(text: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "col_r1.txt"
        p.write_text(text)
        return analyse([p], ROWS)


class CollideReport(unittest.TestCase):
    def test_rock_through_container_needs_both_sides(self):
        both = report("O a NewRow_1 0 0 0 1200 250 260\nO b NewRow_71 500 0 0 900 400 500\nR a 0\nR b 0\n"
                      "I a b NewRow_71 2\nI b a NewRow_1 2\n")
        self.assertEqual(both["summary"]["rock"], 1)
        one = report("O a NewRow_1 0 0 0 1200 250 260\nO b NewRow_71 500 0 0 900 400 500\nR a 0\nR b 0\n"
                     "I a b NewRow_71 2\n")
        self.assertEqual(one["summary"]["rock"], 0)

    def test_floating_group_is_one_issue(self):
        r = report("O a NewRow_33 0 0 800 2800 30 830\nO b NewRow_222 1000 0 820 1200 20 900\n"
                   "T a b\nT b a\nF a NewRow_33 1400 15 800 790 0\n")
        self.assertEqual(r["summary"]["float"], 1)
        self.assertEqual(r["clusters"][0]["measure"], 7.9)

    def test_chain_to_the_ground_is_held_up(self):
        r = report("O s NewRow_157 0 0 0 240 240 418\nO t NewRow_157 0 0 400 240 240 818\n"
                   "O c NewRow_1 -400 0 818 800 250 1120\nR s 0\nR t s\nT c t\n")
        self.assertEqual(sum(r["summary"].values()), 0)

    def test_balloon_floats_and_holds_nothing(self):
        r = report("O b NewRow_165 0 0 3000 2000 2000 5000\nO x NewRow_33 500 500 5000 900 600 5040\nT x b\n"
                   "F b NewRow_165 1000 1000 3000 3000 0\nF x NewRow_33 700 550 5000 5000 0\n")
        self.assertEqual(r["summary"]["float"], 1)
        self.assertNotIn("Air Balloon Random E", r["clusters"][0]["objects"])

    def test_resting_on_a_blueprint_building(self):
        r = report("B h NewRow_12 0 0 0 5000 5000 2280\nO s NewRow_157 1000 1000 2300 1140 1140 2720\n"
                   "F s NewRow_157 1070 1070 2300 1674 0\n")
        self.assertEqual(sum(r["summary"].values()), 0)

    def test_negative_world_height_is_not_automatic_ground(self):
        r = report("O a NewRow_33 0 0 -1000 1000 30 -970\nF a NewRow_33 500 15 -1000 500 0\n")
        self.assertEqual(r['summary']['float'], 1)

    def test_negative_world_height_with_measured_ground_is_supported(self):
        r = report("O a NewRow_33 0 0 -1000 1000 30 -970\nR a 0\n")
        self.assertEqual(sum(r['summary'].values()), 0)

    def test_placed_plant_does_not_hold_up_a_prop(self):
        r = report("O t NewRow_279 0 0 -1000 500 500 1000\nR t 0\n"
                   "O a NewRow_33 0 0 1000 1000 30 1030\nR a t\n"
                   "F a NewRow_33 500 15 1000 2000 0\n")
        self.assertEqual(r['summary']['float'], 1)

    def test_car_sunk_in_a_ground_tile(self):
        r = report("O a NewRow_103 0 0 0 480 200 140\nS a 13.2 NewRow_180 StaticMeshActor 240 100\nR a 0\n")
        self.assertEqual(r["summary"]["sunk"], 1)
        self.assertEqual(r["clusters"][0]["measure"], 13.2)

    def test_scan_regions_share_boundaries_without_gaps(self):
        cmds, paths = scan_commands(Path("/tmp/x"), lambda p: "Z:" + str(p).replace("/", "\\"), region_m=(-100, -100, 100, 100), grid=2)
        self.assertEqual(len(cmds), 4)
        self.assertEqual(cmds[0].split()[2:], ["-10000", "-10000", "0", "0", "30"])
        self.assertEqual(cmds[3].split()[2:], ["0", "0", "10000", "10000", "30"])
        self.assertTrue(cmds[0].split()[1].startswith("Z:\\tmp\\x\\col_r1"))

    def test_duplicate_border_objects_do_not_duplicate_findings(self):
        text = "O a NewRow_33 0 0 800 2800 30 830\nF a NewRow_33 1400 15 800 790 0\n"
        self.assertEqual(report(text+text)['summary']['float'], 1)

    def test_catalogue_rock_tag_prevents_treating_it_as_a_plant(self):
        rows = Rows({'rock': ('opaque_mesh_name', 'Foliage', ('rock',))})
        self.assertEqual(rows.cat('rock'), 'ROCK')

    def test_bush_is_not_a_vehicle_despite_inferred_bus_tag(self):
        rows = Rows({'bush': ('Bush5 01', 'Foliage', ('bush', 'vehicle'))})
        self.assertEqual(rows.cat('bush'), 'PLANT')

    def test_brief_and_page(self):
        r = report("O a NewRow_33 0 0 800 2800 30 830\nF a NewRow_33 1400 15 800 790 0\n")
        self.assertEqual(brief(r)["issues"], {"float": 1})
        self.assertIn("Beam01", render_html(r, title="Test"))


if __name__ == "__main__":
    unittest.main()
