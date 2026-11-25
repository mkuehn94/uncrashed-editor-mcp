"""Survey rules and query details found in live test builds (brief 2, sections 6 and 7): roof greenery read as trees,
OSM footprints that disagree with the heights, low-pitch roofs, building parts, raised edge cells, stacks read as
cars, and the query surfaces (nearest, AGL heights, ground under roofs, profile slopes, raster export, part ids,
colours, legends, ground tiles, reference-only captures). Synthetic rasters and meshes; never reaches the game."""
import contextlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from uncrashed_mcp import agent, api, geom, live, managed, reference, sheet
from uncrashed_mcp.catalog import load_catalog
from uncrashed_mcp.scan import mapping, ops, profile, store, survey, vision

from test_scan import NATIVE, Sandbox, make_scene

RES = 0.25
_guards = []
_tmp = tempfile.TemporaryDirectory()


def setUpModule():
    for name in ('ensure_agent', '_execute'):
        guard = patch.object(agent, name, side_effect=AssertionError(f'agent.{name} reached from a test'))
        guard.start()
        _guards.append(guard)
    for target, name, value in ((agent, 'session', contextlib.nullcontext),
                                (profile, 'PAKS', Path(_tmp.name) / 'no-paks'),
                                (profile, 'cache_dir', lambda: Path(_tmp.name) / 'profiles')):
        guard = patch.object(target, name, value)
        guard.start()
        _guards.append(guard)
    profile._MEM.clear()
    profile.catalogue_version.cache_clear()
    profile._paks.cache_clear()


def tearDownModule():
    while _guards:
        _guards.pop().stop()
    profile._MEM.clear()
    profile.catalogue_version.cache_clear()
    profile._paks.cache_clear()


# ---------------------------------------------------------------------------- synthetic rasters


def normals(dsm: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(dsm.astype(np.float64), RES)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    return (n / np.linalg.norm(n, axis=2, keepdims=True)).astype(np.float32)


def blank(n: int = 240, seed: int = 4):
    rng = np.random.default_rng(seed)
    dsm = np.zeros((n, n), np.float32)
    rgb = np.clip(128 + rng.normal(0, 5, (n, n, 3)), 0, 255).astype(np.uint8)
    return dsm, rgb, rng, survey.Grid(-n * RES / 2, -n * RES / 2, RES, n, n)


def paint_green(rgb, rr, cc, rng):
    rgb[rr, cc] = np.clip(np.array([60, 140, 50]) + rng.normal(0, 20, (len(rr), 3)), 0, 255)


def crown(dsm, rgb, rng, r0, c0, radius_m, top_m):
    yy, xx = np.mgrid[0:dsm.shape[0], 0:dsm.shape[1]]
    d = np.hypot(yy - r0, xx - c0) * RES
    m = d < radius_m
    h = (top_m - 3) * np.sqrt(np.clip(1 - (d[m] / radius_m) ** 2, 0, 1)) + 3 + rng.normal(0, 0.4, m.sum())
    dsm[m] = np.maximum(dsm[m], h.astype(np.float32))
    paint_green(rgb, *np.nonzero(m), rng)


@unittest.skipUnless(NATIVE, 'needs g++ and libturbojpeg headers to build native/scanraster')
class SurveyRuleTests(unittest.TestCase):
    def run_survey(self, dsm, rgb, g, feats=None):
        return survey.Survey(g, dsm, normals(dsm), rgb, feats).run()

    def test_greenery_on_a_roof_edge_is_roof_not_a_tree(self):
        dsm, rgb, rng, g = blank()
        dsm[40:120, 40:80] = 12.0                      # a 20 x 10 m block, 12 m tall
        rr, cc = (a.ravel() for a in np.mgrid[64:88, 80:85])   # a 1.25 x 6 m green band at roof height beside it
        dsm[rr, cc] = 12 + rng.normal(0, 0.5, len(rr))
        paint_green(rgb, rr, cc, rng)
        crown(dsm, rgb, rng, 180, 180, 3.2, 12)        # a real tree in the open
        sv = self.run_survey(dsm, rgb, g)
        trees = [o for o in sv.objects if o['cls'] == 'tree']
        self.assertEqual(len(trees), 1, trees)
        self.assertAlmostEqual(trees[0]['x'], 15.0, delta=0.5)
        self.assertEqual(sv.stats.get('trees_on_roofs'), 1)
        building = next(o for o in sv.objects if o['cls'] == 'building')
        self.assertTrue((sv.cls[rr, cc] == 10).all())            # the band is roof now ...
        self.assertTrue((sv.objid[rr, cc] == building['id']).all())   # ... of that building

    def test_a_crown_inside_a_mapped_footprint_is_dropped(self):
        dsm, rgb, rng, g = blank()
        dsm[40:80, 40:80] = 7.0                         # a mapped 10 x 10 m house ...
        crown(dsm, rgb, rng, 60, 60, 1.6, 9)            # ... with a green, rough planting on its roof
        crown(dsm, rgb, rng, 180, 180, 3.2, 12)
        feats = {'buildings': [{'osm': 'way/9', 'rings': [[(-20, -20), (-10, -20), (-10, -10), (-20, -10)]],
                                'tags': {'building': 'yes'}}], 'roads': [], 'trees': []}
        sv = self.run_survey(dsm, rgb, g, feats)
        self.assertEqual([round(o['x']) for o in sv.objects if o['cls'] == 'tree'], [15])

    def test_osm_footprints_that_disagree_with_the_heights_are_flagged_not_used(self):
        dsm, rgb, rng, g = blank()
        dsm[40:80, 40:80] = 7.0         # house A, 10 x 10 m at -20..-10
        dsm[140:180, 140:180] = 7.0     # house B at 5..15
        feats = {'buildings': [
            {'osm': 'way/1', 'rings': [[(-20, -20), (-10, -20), (-10, -10), (-20, -10)]], 'tags': {'building': 'house'}},
            {'osm': 'way/2', 'rings': [[(-20, 5), (-10, 5), (-10, 15), (-20, 15)]], 'tags': {'building': 'garage'}},
            {'osm': 'way/3', 'rings': [[(10, 5), (20, 5), (20, 15), (10, 15)]], 'tags': {'building': 'house'}}],
            'roads': [], 'trees': []}
        sv = self.run_survey(dsm, rgb, g, feats)
        self.assertEqual(sv.osm_info['footprints_disagree'], 2)    # way/2 over bare ground, way/3 half off house B
        bad = {c['osm']: c for c in sv.osm_info['disagree']}
        self.assertEqual(set(bad), {'way/2', 'way/3'})
        self.assertEqual(bad['way/2']['cover'], 0.0)
        self.assertAlmostEqual(bad['way/3']['cover'], 0.5, delta=0.1)
        self.assertEqual(bad['way/3']['best_shift_m'][0], -3.0)    # moving it 3 m west is as far as the search goes
        a, b = sorted((o for o in sv.objects if o['cls'] == 'building'), key=lambda o: o['x'])
        self.assertEqual(a['osm']['osm'], 'way/1')
        self.assertIsNone(b['osm'])                                 # footprint, split and walls from the heights
        self.assertEqual(b['osm_conflict']['osm'], 'way/3')
        self.assertAlmostEqual(b['area_m2'], a['area_m2'], delta=2)    # the whole house, not the mapped half
        self.assertEqual(vision.wall_ring(b, {'way/3': [[10, 5], [20, 5], [20, 15], [10, 15]]}, (0, 0))[1],
                         'roof_outline')

    def test_low_pitch_roofs_report_planes_lines_and_heights(self):
        dsm, rgb, rng, g = blank(320)
        yy, xx = np.mgrid[0:320, 0:320]
        hall = (yy >= 20) & (yy < 140) & (xx >= 20) & (xx < 100)        # 30 x 20 m, valley along its length
        dsm[hall] = (8 + 0.035 * np.abs(xx[hall] - 59.5) * RES).astype(np.float32)
        house = (yy >= 180) & (yy < 228) & (xx >= 40) & (xx < 60)       # 12 x 5 m, 0.55 m gable along y
        dsm[house] = (6.55 - 0.55 * np.abs(xx[house] - 49.5) / 10).astype(np.float32)
        shed = (yy >= 180) & (yy < 220) & (xx >= 200) & (xx < 240)      # 10 x 10 m, flat with a 5 % fall
        dsm[shed] = (5 + 0.05 * (xx[shed] - 200) * RES).astype(np.float32)
        sv = self.run_survey(dsm, rgb, g)
        b = {round(o['x']): o for o in sv.objects if o['cls'] == 'building'}
        hall_o, house_o, shed_o = b[-25], b[-28], b[15]
        self.assertEqual(hall_o['roof'], 'valley')
        fit = hall_o['roof_shape']
        self.assertEqual(fit['line']['kind'], 'valley')
        for v in fit['slope_pct']:
            self.assertAlmostEqual(v, 3.5, delta=0.6)
        self.assertAlmostEqual(fit['line']['height_m'], 8.0, delta=0.15)
        self.assertGreater(min(fit['line']['edges_m']), fit['line']['height_m'] + 0.2)
        self.assertAlmostEqual(fit['line']['yaw_deg'] % 180, 90, delta=3)        # runs south (+y), the hall's length
        self.assertEqual(house_o['roof'], 'gabled')
        self.assertAlmostEqual(house_o['roof_shape']['line']['height_m'], 6.55, delta=0.12)
        for v in house_o['roof_shape']['line']['edges_m']:
            self.assertAlmostEqual(v, 6.1, delta=0.2)
        self.assertEqual(shed_o['roof'], 'flat')                                  # a fall of 5 % is still flat ...
        self.assertEqual(shed_o['roof_shape']['form'], 'mono_pitch')               # ... and says so
        self.assertAlmostEqual(shed_o['roof_shape']['slope_pct'][0], 5.0, delta=0.6)

    def test_parts_are_whole_rectangles_without_slivers(self):
        m = np.zeros((120, 120), bool)
        m[0:40, 0:100] = True       # 25 x 10 m wing
        m[40:100, 0:40] = True      # 10 x 15 m wing (an L)
        m[40:44, 40:100] = True     # a 1 m wide strip along the inner corner
        rows, cols = np.nonzero(m)
        parts = survey._parts(cols, rows, RES)      # the greedy split made two 1.75 m slivers and an L of it
        self.assertEqual(sorted(round(len(c) * RES * RES) for c, _ in parts), [165, 250])
        self.assertEqual(sum(len(c) for c, _ in parts), len(cols))
        seen = set()
        for c, r in parts:
            self.assertGreaterEqual(len(c) * RES * RES, 15)
            self.assertGreaterEqual(grid_width(c, r), 2.5)
            cells = set(zip(c.tolist(), r.tolist()))
            self.assertFalse(cells & seen)
            seen |= cells

    def test_raised_edge_cells_are_not_ground_and_belong_to_the_building(self):
        dsm, rgb, rng, g = blank()
        dsm[60:120, 60:120] = 9.0
        dsm[90, 120:126] = 9.0      # a wall top one cell wide, 1.5 m long: the openings cut it off the roof
        sv = self.run_survey(dsm, rgb, g)
        raised = sv.valid & (sv.ndsm > survey.HIGH_M)
        self.assertFalse(np.isin(sv.cls[raised], survey.GROUND_CODES).any())
        bid = next(o['id'] for o in sv.objects if o['cls'] == 'building')
        self.assertTrue((sv.objid[90, 118:120] == bid).all())   # within reach of the roof: claimed by it


def grid_width(c, r) -> float:
    from uncrashed_mcp.scan import grid
    return grid.obb(c, r)[3] * RES


class ClutterTests(unittest.TestCase):
    def ras(self, heights: np.ndarray, cls_code: int = 1):
        n = heights.shape[0]
        meta = {'grid': {'x0': 0.0, 'y0': 0.0, 'res': RES, 'w': n, 'h': n}}
        return vision._Ras(meta, heights.astype(np.float32), np.zeros_like(heights, np.float32),
                           np.full(heights.shape, cls_code, np.uint8), np.full(heights.shape, -1, np.int32),
                           np.zeros(heights.shape + (3,), np.uint8))

    def car(self, length=4.4, src='mesh+vision'):
        return {'x': 5.0, 'y': 5.0, 'length_m': length, 'width_m': 1.9, 'yaw_deg': 0.0, 'src': src}

    def test_stacks_and_clutter_are_not_cars(self):
        rng = np.random.default_rng(2)
        even = np.zeros((40, 40))
        even[14:26, 11:29] = 1.45 + rng.normal(0, 0.05, (12, 18))     # a car body: even top
        self.assertIsNone(vision.clutter(self.car(), self.ras(even)))
        stacks = np.zeros((40, 40))
        stacks[14:26, 11:29] = rng.choice([0.3, 1.2, 2.8], (12, 18))    # pallets, barrels, a stack: uneven
        self.assertIn('vision only', vision.clutter(self.car(src='vision'), self.ras(stacks)))
        self.assertIn('2.6 m long', vision.clutter(self.car(2.6), self.ras(stacks)))
        self.assertIn('taller than a car', vision.clutter(self.car(), self.ras(stacks)))
        under_tree = self.ras(stacks, cls_code=11)     # a crown over the car: those cells say nothing
        self.assertIsNone(vision.clutter(self.car(), under_tree))

    def test_fusion_turns_a_detected_stack_into_a_small_object(self):
        rng = np.random.default_rng(3)
        heights = np.zeros((40, 40))
        heights[14:26, 11:29] = rng.choice([0.3, 1.2, 2.8], (12, 18))
        ras = self.ras(heights)
        geo = [{'id': 0, 'cls': 'car', 'conf': 0.6, 'src': 'mesh', 'x': 5.0, 'y': 5.0, 'length_m': 4.4,
                'width_m': 1.9, 'yaw_deg': 0.0, 'height_m': 2.8, 'ground_z': 0.0, 'top_z': 2.8, 'area_m2': 8.4}]
        corners = np.array([[2.8, 4.05], [7.2, 4.05], [7.2, 5.95], [2.8, 5.95]])
        cand = {'cls': 'car', 'src': 'obb', 'score': 0.9, 'label': 'small vehicle', 'corners': corners,
                'oriented': True, 'mask': False}
        objs, rep = vision.fuse(geo, [cand], ras)
        self.assertEqual(objs[0]['cls'], 'small_object')
        self.assertTrue(objs[0]['note'].startswith('not a car'))
        self.assertEqual(rep['cars_rejected_clutter'], 1)
        self.assertTrue(ops._flag(objs[0]).startswith('not a car'))


class HelperTests(unittest.TestCase):
    def test_colour_words_ignore_the_blue_cast_of_shade(self):
        self.assertEqual(ops.colour_name([65, 81, 117]), 'dark grey')    # a shaded grey car
        self.assertEqual(ops.colour_name([40, 60, 170]), 'blue')
        self.assertEqual(ops.colour_name([180, 60, 40]), 'red')
        self.assertEqual(ops.colour_name([240, 240, 238]), 'white')
        self.assertEqual(ops.colour_name([20, 20, 22]), 'black')
        self.assertEqual(ops.colour_name([60, 140, 50]), 'green')
        self.assertEqual(ops._colour([255, 0, 0])['hex'], '#ff0000')
        self.assertIsNone(ops.colour_name(None))

    def test_overlapping_footprints_flag_each_other(self):
        sq = lambda x, y, s: [[x, y], [x + s, y], [x + s, y + s], [x, y + s]]   # noqa: E731
        objs = [{'id': 1, 'cls': 'building', 'footprint': sq(0, 0, 10)},
                {'id': 2, 'cls': 'building', 'footprint': sq(5, 0, 10)},      # half over 1
                {'id': 3, 'cls': 'building', 'footprint': sq(30, 0, 10)},
                {'id': 4, 'cls': 'building', 'footprint': sq(39.5, 0, 10)}]   # touches 3 by 5 %
        over = ops._overlaps(objs)
        self.assertEqual(set(over), {1, 2})
        self.assertEqual(over[1][0], 2)
        self.assertAlmostEqual(over[1][1], 0.5, delta=0.06)
        row = {'id': 1, 'cls': 'building', 'height_m': 5, 'length_m': 10, 'src': 'mesh', 'conf': 0.9}
        self.assertIn('overlaps building 2', ops._flag(row, overlaps=over))

    def test_window_rows_with_the_head_under_the_sill_are_flagged(self):
        self.assertIn('head at or below the sill', ops._rows_check([{'sill_m': 3.0, 'head_m': 2.5}]))
        self.assertIsNone(ops._rows_check([{'sill_m': 0.9, 'head_m': 2.2}, {'sill_m': 3.9, 'head_m': 5.2}]))

    def test_roof_planes_find_a_shallow_valley(self):
        e, s = np.meshgrid(np.arange(-15, 15, 0.5), np.arange(-10, 10, 0.5))
        z = 100 + 0.03 * np.abs(s)                                 # valley along e, 3 % to each side
        fit = survey.roof_planes(e.ravel(), s.ravel(), z.ravel(), 0.0, 92.0)
        self.assertEqual((fit['form'], fit['line']['kind']), ('valley', 'valley'))
        self.assertAlmostEqual(fit['line']['height_m'], 8.0, delta=0.02)
        flat = survey.roof_planes(e.ravel(), s.ravel(), np.full(e.size, 100.0), 0.0, 92.0)
        self.assertEqual((flat['form'], flat['slope_pct']), ('flat', [0.0]))

    def test_edit_errors_say_to_save_first(self):
        with patch.object(agent, 'map_info', return_value={'folder': '', 'directory': ''}):
            with self.assertRaisesRegex(ValueError, r'no saved name yet.*Fix: save it first \(./uc save_live NAME'):
                managed.context()

    def test_site_ground_is_reported_against_the_datum(self):
        with patch.object(reference, 'auto_datum', return_value=57.9):
            site = reference.site_ground(Path('/x'), [], 'x,z,-y', (0, 0), 150, 54.1, 1.0)
        self.assertEqual((site['above_datum_m'], site['map_z']), (3.8, 4.8))
        self.assertIn('+3.80 m from the datum', site['note'])
        self.assertIn('with z -2.80 puts it at map z 1.00', site['note'])


class GroundTileTests(unittest.TestCase):
    def test_a_tilted_tile_top_passes_through_its_corners(self):
        cat = load_catalog()
        corners = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
        top = np.array([10.0, 10.1, 10.25, 10.15])          # 5 % along x, 7.5 % along y
        for row in ('NewRow_182', 'NewRow_303'):            # a base pivot and a top pivot
            line = mapping._slab(row, cat.assets[row], corners, top, 'g')
            w = line.split()
            px, py, pz, p, y, r, sx, sy, sz = map(float, w[2:11])
            a = cat.assets[row]
            q = geom.quat_rotator(p, y, r)
            got = []
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                loc = (np.array(a.origin_cm) + np.array([su, sv, 1]) * np.array(a.extent_cm)) * [sx, sy, sz]
                got.append((np.array([px, py, pz]) + np.array(geom.quat_rotate(q, loc))) / 100)
            got = np.array(got)
            for k in range(4):   # every corner of the top face sits on one of the given corners
                d = np.linalg.norm(got - np.r_[corners[k], top[k]], axis=1).min()
                self.assertLess(d, 0.02, (row, k, got))


# ---------------------------------------------------------------------------- query surfaces on a surveyed OBJ


@unittest.skipUnless(NATIVE, 'needs g++ and libturbojpeg headers to build native/ucmesh and native/scanraster')
class QueryTests(Sandbox):
    @classmethod
    def setUpClass(cls):
        reference.tool()
        from uncrashed_mcp.scan import native
        native.library()

    def setUp(self):
        super().setUp()
        self.obj = make_scene(self.root / 'src')
        ops.scan_survey(str(self.obj), use_osm=False)
        self.meta = store.resolve('')

    def objs(self, cls):
        return [o for o in store.objects(self.meta) if o['cls'] == cls]

    def test_objects_nearest_sorts_colours_and_say_where_the_nearest_miss_is(self):
        near = ops.scan_objects(classes=['car', 'tree'], near=[25, 25], nearest=2, radius_m=1)
        self.assertEqual(len(near['objects']), 2)                 # however far, the two nearest
        self.assertEqual(near['objects'][0]['cls'], 'tree')
        self.assertLess(near['objects'][0]['dist_m'], near['objects'][1]['dist_m'])
        self.assertEqual(near['nearest'], 2)
        cars = ops.scan_objects(classes=['car'])['objects']
        self.assertEqual({r['colour'] for r in cars}, {'blue'})    # painted (40, 60, 160)
        self.assertTrue(all(r['rgb'].startswith('#') for r in cars))
        big = ops.scan_objects(sort='area', limit=2)['objects']
        self.assertEqual([r['cls'] for r in big], ['building', 'building'])
        self.assertEqual(ops.scan_objects(sort='distance', near=[0, 0])['sort'], 'near')
        blds = ops.scan_objects(classes=['building'], sort='storeys')['objects']
        self.assertTrue(all({'area_m2', 'storeys', 'windows', 'roof'} <= set(r) for r in blds))
        none = ops.scan_objects(classes=['car'], box=[20, -28, 28, -20])
        self.assertEqual(none['total'], 0)
        self.assertRegex(none['summary'], r'^no car in the box; nearest car \d+ is \d+ m from the box centre$')
        with self.assertRaisesRegex(ValueError, 'nearest needs near'):
            ops.scan_objects(nearest=3)

    def test_heights_above_ground_and_the_ground_under_a_roof(self):
        h = ops.scan_height([[-15, -16, 8], [0, 0, 8]], agl=True)
        roof, open_ = h['points']
        self.assertAlmostEqual(roof['z'], 8.0, delta=0.1)
        self.assertAlmostEqual(roof['clearance_m'], 2.0, delta=0.15)   # 8 m above ground, 6 m roof
        self.assertAlmostEqual(open_['agl_m'], 8.0, delta=0.01)
        self.assertEqual(roof['surface'], 'building')
        self.assertIn(roof['ground_cls'], {'paved', 'other_ground', 'grass', 'bare'})
        self.assertGreater(roof['ground_cls_at_m'], 1.0)
        self.assertNotIn('ground_cls', open_)
        self.assertEqual(len(ops.scan_height([[0, 0]] * 1024)['points']), 1024)
        with self.assertRaisesRegex(ValueError, '1 to 1024 points'):
            ops.scan_height([[0, 0]] * 1025)
        with self.assertRaisesRegex(ValueError, 'agl needs x,y,z'):
            ops.scan_height([[0, 0]], agl=True)

    def test_profile_with_a_flight_line_and_slopes_per_segment(self):
        prof = ops.scan_profile([-25, -16, 7], [-5, -16, 7], step_m=1)
        self.assertAlmostEqual(prof['min_clearance_m'], 1.0, delta=0.15)       # 7 m over the 6 m roof
        self.assertEqual(prof['min_clearance_at']['surface'], 'building')
        self.assertTrue(all('slope_pct' in r and 'clearance_m' in r for r in prof['samples']))
        agl = ops.scan_profile([-25, -16, 2], [-5, -16, 2], agl=True)
        self.assertLess(agl['min_clearance_m'], 0)

    def test_free_names_the_object_of_a_raised_cell_without_an_id(self):
        g = types.SimpleNamespace(res=0.25, to_cell=lambda e, s: (e / 0.25, s / 0.25))
        objid = np.full((20, 20), -1, np.int32)
        objid[8:12, 12:16] = 7
        fake = types.SimpleNamespace(g=g, objid=objid, ground=lambda e, s: np.zeros(len(e)))
        hit = ops._hit(fake, {7: {'id': 7, 'cls': 'building'}}, 2.5, 2.5, 9.0, -1, 1, 0.0)
        self.assertEqual((hit['object'], hit['cls'], hit['object_from']), (7, 'building', 'nearest within 1.5 m'))
        self.assertEqual(hit['surface'], 'raised paved cell (edge)')
        self.assertEqual(hit['above_ground_m'], 9.0)
        low = ops._hit(fake, {}, 0.5, 0.5, 0.05, -1, 1, 0.0)
        self.assertEqual((low['object'], low['cls']), (None, 'paved'))

    def test_grid_exports_a_box_as_one_file(self):
        out = ops.scan_grid([-24, -24, 24, 24], res_m=1)
        data = np.load(out['file'])
        self.assertEqual(data['top'].shape, (48, 48))
        self.assertEqual(set(data.files), set(ops.GRID_LAYERS))
        header = json.loads(Path(out['header']).read_text())
        self.assertEqual(header['class_codes']['10'], 'building')
        # the flat-roof house (-20..-10 east, -20..-12 south) is 6 m above ground in its cells
        self.assertAlmostEqual(float(data['height'][24 - 16, 24 - 15]), 6.0, delta=0.2)
        self.assertEqual(int(data['class'][24 - 16, 24 - 15]), 10)
        self.assertGreaterEqual(int(data['object'][24 - 16, 24 - 15]), 0)
        self.assertTrue(out['file'].startswith(str(store.SHM)))
        self.assertIn('building', out['summary']['class_share'])
        some = ops.scan_grid([-4, -4, 4, 4], res_m=0.5, layers=['top'], out=str(self.root / 'g.npz'))
        self.assertEqual(np.load(some['file']).files, ['top'])
        with self.assertRaisesRegex(ValueError, 'at most 1,000,000'):
            ops.scan_grid([-400, -400, 400, 400], res_m=0.5)
        with self.assertRaisesRegex(ValueError, 'unknown layers lidar'):
            ops.scan_grid([-4, -4, 4, 4], layers=['lidar'])

    def test_classes_and_ndsm_views_carry_a_legend(self):
        v = ops.scan_view(center=[0, 0], size_m=50, layer='classes', px=300)
        self.assertIn('building', v['class_legend'])
        self.assertRegex(v['class_legend']['building'], r'^#[0-9a-f]{6}$')
        self.assertIn('above ground', ops.scan_view(center=[0, 0], size_m=50, layer='ndsm', px=300)['ndsm_legend'])
        self.assertNotIn('class_legend', ops.scan_view(center=[0, 0], size_m=50, px=300))

    def test_a_part_id_answers_for_the_part(self):
        objs = store.objects(self.meta)
        house = next(o for o in objs if o['cls'] == 'building' and o['roof'] == 'flat')
        part = {**house, 'id': len(objs), 'cls': 'building_part', 'parent': house['id'], 'length_m': 5.0,
                'footprint': [[-20, -20], [-15, -20], [-15, -12], [-20, -12]], 'x': -17.5, 'area_m2': 40.0}
        house = {**house, 'parts': [part['id']]}
        store.write_json(store.home(self.meta['key']) / 'objects.json',
                         [house if o['id'] == house['id'] else o for o in objs] + [part])
        b = ops.scan_building(part['id'])
        self.assertEqual((b['id'], b['cls'], b['parent']), (part['id'], 'building_part', house['id']))
        self.assertAlmostEqual(b['center'][0], -17.5, delta=0.01)
        self.assertEqual(len(b['footprint']), 4)
        self.assertLessEqual(len(b['facades']), 3)              # the walls along this half, not all four
        whole = ops.scan_building(house['id'])
        self.assertEqual(whole['parts'][0]['id'], part['id'])
        self.assertIn('footprint', whole['parts'][0])
        self.assertIn('roof', whole['colours'])

    def test_a_building_the_survey_skipped_gets_its_facades_now(self):
        bid = self.objs('building')[0]['id']
        b = ops.scan_building(bid)          # tests have no vision env: says why
        self.assertIn('vision unavailable', ' '.join(b['limitations']))
        wall = {'side': 0, 'a': [-20, -20], 'b': [-10, -20], 'length_m': 10.0, 'faces_deg': 270.0, 'status': 'surveyed',
                'tone': [200, 190, 170], 'grid': {'storeys': 2, 'windows': 6, 'columns': 3, 'spacing_m': 3.0,
                                                   'rows': [{'sill_m': 0.9, 'head_m': 2.2}, {'sill_m': 3.9, 'head_m': 5.2}],
                                                   'doors': [], 'balconies': []}}
        with patch.object(vision, 'ready', return_value=(True, '')), \
                patch.object(vision, 'facades_on_demand', return_value={str(bid): [wall]}) as run:
            b = ops.scan_building(bid)
            again = ops.scan_building(bid)
        self.assertEqual(run.call_count, 1)                       # stored: the second answer reads it
        self.assertTrue(b['facades_surveyed'] and again['facades_surveyed'])
        self.assertEqual(b['storeys'], 2)
        self.assertEqual(b['colours']['facade']['name'], 'beige')
        self.assertIn('surveyed now', b['facade_note'])
        self.assertIn(str(bid), json.loads((store.home(self.meta['key']) / 'facades.json').read_text()))
        self.assertFalse(ops.scan_building(self.objs('building')[1]['id'], survey_facades=False)['facades_surveyed'])

    def test_placement_keeps_datum_and_yaw_and_says_how_to_restore_the_old_one(self):
        with self.assertRaisesRegex(ValueError, 'first placement needs datum_m'):
            ops.scan_place(at=[0, 0, 0])
        ops.scan_place(at=[0, 0, 1.0], yaw=30, datum_m=-2.0)
        out = ops.scan_place(at=[0, 0, 0])                                   # only --at: keeps yaw and datum
        self.assertEqual((out['placement']['yaw'], out['placement']['datum_m']), (30.0, -2.0))
        self.assertIn('datum_m', out['kept'])
        self.assertEqual(out['previous']['at'], [0.0, 0.0, 1.0])
        self.assertEqual(out['restore'], './uc scan_place --at 0,0,1 --yaw 30 --datum_m -2')
        self.assertNotIn('restore', ops.scan_place(at=[0, 0, 0]))           # nothing changed
        cleared = ops.scan_place(clear=True)
        self.assertTrue(cleared['cleared'])
        self.assertEqual(cleared['restore'], './uc scan_place --at 0,0,0 --yaw 30 --datum_m -2')

    def test_plan_takes_a_circle_and_holds_weak_fits_back_for_review(self):
        ops.scan_place(at=[0, 0, 0], yaw=0, datum_m=0.0)
        c1, c2 = sorted(self.objs('car'), key=lambda o: o['x'])
        self.assertAlmostEqual(c2['x'] - c1['x'], 6.0, delta=0.5)
        near = [c1['x'] + 2, c1['y'] + 4.5]          # car 2 lies in the 10 m square but not in the 5 m circle
        sq = mapping.scan_plan(box=[near[0] - 5, near[1] - 5, near[0] + 5, near[1] + 5], classes=['car'], min_conf=0,
                               out=str(self.root / 'sq.txt'))
        ci = mapping.scan_plan(near=near, radius_m=5, classes=['car'], min_conf=0, out=str(self.root / 'ci.txt'))
        self.assertEqual((sq['classes'][0]['planned'], ci['classes'][0]['planned']), (2, 1))
        self.assertEqual(json.loads(Path(ci['json']).read_text())['region'], {'near': near, 'radius_m': 5.0})
        with self.assertRaisesRegex(ValueError, 'not both'):
            mapping.scan_plan(box=[0, 0, 1, 1], near=near, radius_m=5, out=str(self.root / 'x.txt'))
        # weak fits: a pallet stack scored 0.03 / 85 % as a van must not reach the batch unseen
        self.assertIn('85', mapping._weak_fit('car', {'fit_err_pct': 85.0, 'score': 0.03}))
        self.assertIn('0.46', mapping._weak_fit('car', {'fit_err_pct': 19.0, 'score': 0.46}))
        self.assertIsNone(mapping._weak_fit('tree', {'fit_err_pct': 10.0, 'score': 0.44}))   # sparse foliage meshes
        self.assertIsNone(mapping._weak_fit('car', {'fit_err_pct': 5.7, 'score': 0.9}))
        with patch.object(mapping, 'WEAK_FIT_PCT', -1.0):                    # every pick is weak now
            rv = mapping.scan_plan(classes=['car'], min_conf=0, out=str(self.root / 'rv.txt'))
            keep = mapping.scan_plan(classes=['car'], min_conf=0, weak='keep', out=str(self.root / 'kp.txt'))
            skip = mapping.scan_plan(classes=['car'], min_conf=0, weak='skip', out=str(self.root / 'sk.txt'))
        self.assertEqual((rv['spawn_lines'], rv['review_count'], rv['classes'][0]['review']), (0, 2, 2))
        text = Path(rv['file']).read_text()
        self.assertEqual(text.count('#? spawn '), 2)
        self.assertIn('WEAK FIT', text)
        self.assertEqual(keep['spawn_lines'], 2)
        self.assertIn('WEAK FIT', Path(keep['file']).read_text())
        self.assertEqual((skip['spawn_lines'], skip['skipped_count']), (0, 2))

    def test_placement_reports_the_site_ground(self):
        out = ops.scan_place(at=[0, 0, 1.0], yaw=0, datum_m=-2.0)
        self.assertAlmostEqual(out['site_ground']['z'], 3.0, delta=0.05)   # ground 0 in the source, datum -2, at z 1
        self.assertAlmostEqual(out['site_ground']['above_datum_m'], 2.0, delta=0.05)

    def test_plan_paves_the_ground_with_matching_tiles(self):
        ops.scan_place(at=[0, 0, 0], yaw=0, datum_m=0.0)
        out = mapping.scan_plan(box=[-28, -28, 28, 28], classes=['ground'], out=str(self.root / 'g.txt'))
        gr = out['ground']
        self.assertGreater(gr['tiles'], 30)
        self.assertGreater(gr['left_out'].get('under buildings', 0), 0)        # nothing under the houses
        self.assertEqual(gr['predicted_steps']['same_row'].get('over_1cm', 0), 0)   # one smooth surface, no seams
        self.assertLess(gr['tiles'], gr['cells'])                                # tiles merge across cells
        lines = [ln for ln in Path(out['file']).read_text().splitlines() if ln.startswith('spawn')]
        self.assertEqual(len(lines), gr['tiles'])
        self.assertNotIn('drop last', Path(out['file']).read_text().splitlines())
        self.assertEqual(out['commands'], gr['tiles'])
        self.assertTrue(all(ln.split()[1] in mapping.GROUND_ROWS.values() for ln in lines))
        with self.assertRaisesRegex(ValueError, 'need a box'):
            mapping.scan_plan(classes=['ground'], out=str(self.root / 'h.txt'))


class SearchAndCaptureTests(unittest.TestCase):
    def test_reference_only_capture_passes_the_mode_and_shows_a_hidden_reference(self):
        calls, runs = [], []
        status = {'available': True, 'loaded': True, 'hidden': True}

        def engine(path, pose, w, h, ortho_cm=0, only=''):
            calls.append((ortho_cm, only))
            Image.new('RGB', (w, h)).save(path)

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(live, 'SHOTS', Path(tmp)), patch.object(live, '_engine_capture', engine), \
                patch.object(live, 'read_camera', return_value={'x': 0, 'y': 0, 'z': 10, 'yaw': 0, 'pitch': -20,
                                                                'roll': 0, 'fov': 90}), \
                patch.object(reference, 'status', return_value=status), \
                patch.object(agent, 'run', side_effect=lambda c, **k: runs.append(c) or []):
            path = live.capture_view(0, 0, 10, width=64, height=64, reference_only=True)
            meta = json.loads(path.with_suffix('.json').read_text())
        self.assertEqual(calls, [(0, 'reference')])
        self.assertEqual(meta['only'], 'reference')
        self.assertEqual(runs, ['reference show', 'reference hide'])   # hidden by the user: shown for it, hidden again
        with self.assertRaisesRegex(live.LiveError, 'engine backend'):
            live.capture_view(backend='window', reference_only=True)

    def test_a_reference_only_sheet_renders_every_pose_in_that_mode(self):
        calls = []

        def engine(path, pose, w, h, ortho_cm=0, only=''):
            calls.append(only)
            Image.new('RGB', (w, h)).save(path)

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(sheet, 'SHEETS', Path(tmp) / 'sheets'), patch.object(live, '_engine_capture', engine), \
                patch.object(live, 'read_camera', return_value={'fov': 90}), \
                patch.object(reference, 'status', return_value={'available': True, 'loaded': True, 'hidden': False}):
            out = sheet.capture_sheet('a 0 0 5 0 -10; b 1 0 5 0 -10', width=64, height=64,
                                      out=str(Path(tmp) / 'o'), reference_only=True)
        self.assertEqual(calls, ['reference', 'reference'])
        self.assertEqual(out['only'], 'reference')

    def test_api_forwards_the_new_options(self):
        seen = {}
        with patch.object(live, 'capture_view', side_effect=lambda *a, **k: seen.update(k) or (_ for _ in ()).throw(
                RuntimeError('stop'))):
            with self.assertRaisesRegex(RuntimeError, 'stop'):
                api.capture_view(reference_only=True)
        self.assertTrue(seen['reference_only'])
        calls = []

        def echo(*a, **k):
            calls.append((a, k))
            return {}

        with patch.object(ops, 'scan_grid', echo), patch.object(ops, 'scan_objects', echo), \
                patch.object(mapping, 'scan_plan', echo):
            api.call('scan_grid', [0, 0, 10, 10], res_m=0.5)
            api.call('scan_objects', near=[1, 2], nearest=4)
            api.call('scan_plan', box=[0, 0, 4, 4], classes=['ground'], ground_cell_m=1)
        self.assertEqual(calls[0], (([0, 0, 10, 10],), {'res_m': 0.5, 'layers': None, 'out': '', 'frame': 'auto',
                                                         'survey_key': ''}))
        self.assertEqual(calls[1][1]['nearest'], 4)
        self.assertEqual((calls[2][1]['ground_cell_m'], calls[2][1]['flat_tol_m'], calls[2][1]['ground_yaw']), (1, 0.3, None))



# ---------------------------------------------------------------------------- iteration 2: facades and eaves


class FacadeEaveTests(unittest.TestCase):
    def ras(self, objid, dsm, cls):
        g = survey.Grid(0, 0, RES, objid.shape[1], objid.shape[0])

        def cell(e, s):
            c, r = g.to_cell(e, s)
            c, r = np.rint(c).astype(int), np.rint(r).astype(int)
            inside = (c >= 0) & (c < g.w) & (r >= 0) & (r < g.h)
            return np.clip(r, 0, g.h - 1), np.clip(c, 0, g.w - 1), inside
        return types.SimpleNamespace(g=g, objid=objid, dsm=dsm, cls=cls, cell=cell)

    def test_the_buildings_own_eaves_in_front_of_a_wall_are_not_a_party_wall(self):
        from uncrashed_mcp.scan import facade
        n = 80                                          # 20 x 20 m; wall along y = 5 m facing +y (south)
        objid = np.full((n, n), -1, np.int32)
        dsm = np.zeros((n, n))
        cls = np.full((n, n), 1, np.uint8)
        objid[:28, :], dsm[:28, :], cls[:28, :] = 5, 15.0, 10      # the roof reaches 2 m over the wall line
        wall = {'a': np.array([2.0, 5.0]), 'b': np.array([18.0, 5.0]), 'n': np.array([0.0, 1.0])}
        ras = self.ras(objid, dsm, cls)
        self.assertEqual(facade.occluded_front(ras, wall, 0.0, 12.0), 1.0)              # the old test: blocked
        self.assertEqual(facade.occluded_front(ras, wall, 0.0, 12.0, own_id=5), 0.0)    # own eaves: open street
        objid[28:, :], dsm[28:, :], cls[28:, :] = 6, 14.0, 10      # a neighbour right behind the eaves
        self.assertEqual(facade.occluded_front(ras, wall, 0.0, 12.0, own_id=5), 1.0)

    def test_eave_is_read_at_the_roof_edge_not_over_the_whole_roof(self):
        rows, cols = np.mgrid[0:40, 0:60]                 # 10 x 15 m gabled roof, ridge along the long axis
        rows, cols = rows.ravel(), cols.ravel()
        v = np.abs(rows * RES + RES / 2 - 5.0) / 5.0      # 0 at the ridge, 1 at the eaves
        nd = 10.0 - 4.0 * v                               # eave 6 m, ridge 10 m (39 degrees)
        dormer = (np.abs(rows - 10) < 4) & (cols % 15 < 6)
        nd[dormer] = 9.5
        edge = survey.edge_eave(rows, cols, nd, RES)
        self.assertAlmostEqual(edge, 6.0, delta=0.4)
        self.assertGreater(float(np.percentile(nd, 15)) - edge, 0.3)        # the old rule sits higher
        self.assertIsNone(survey.edge_eave(rows[:5], cols[:5], nd[:5], RES))



class KerbTests(unittest.TestCase):
    def test_carriageway_reaches_the_kerbs_and_sidewalks_lie_beyond(self):
        from uncrashed_mcp.scan import osm
        g = survey.Grid(-30.0, -15.0, RES, 240, 120)          # 60 x 30 m, road along s = 0 (east-west)
        e, s_ = g.to_world(np.arange(g.w), np.arange(g.h))
        E, S = np.meshgrid(e, s_)
        dtm = np.zeros(E.shape)
        dsm = np.where(np.abs(S) >= 6.0, 0.12, 0.0) + 0.01 * np.sin(E)      # 12 cm kerbs 6 m out
        dsm[np.abs(S) >= 9.0] = 10.0                                         # house fronts 9 m out
        cars = (np.abs(S - 4.8) < 0.9) & ((E % 10) < 4.5)                    # parked cars along one kerb
        dsm[cars] = 1.4
        valid = np.ones(E.shape, bool)
        feats = {'roads': [{'line': [[-28.0, 0.0], [28.0, 0.0]], 'width': 5.5, 'highway': 'residential'}]}
        road, walk, rep = osm.kerb_bands(feats, g, dsm, dsm - dtm, valid)
        self.assertEqual(rep['sides_with_kerb'], 2, rep)
        left, right = feats['roads'][0]['kerb_m']
        self.assertAlmostEqual(left, 6.0, delta=0.4)
        self.assertAlmostEqual(right, 6.0, delta=0.4)
        mid = g.w // 2
        col = lambda m: [abs(v) for v, k in zip(s_, m[:, mid]) if k]
        self.assertAlmostEqual(max(col(road)), 6.0, delta=0.4)              # nominal: 2.75
        self.assertAlmostEqual(min(col(walk)), 6.0, delta=0.4)
        self.assertLessEqual(max(col(walk)), 9.2)
        flat = osm.kerb_bands(feats, g, np.zeros(E.shape), np.zeros(E.shape), valid)
        self.assertEqual(flat[2]['sides_with_kerb'], 0)                       # no kerb: nominal width stays
        self.assertEqual(feats['roads'][0]['kerb_m'], [2.75, 2.75])



class WingSplitTests(unittest.TestCase):
    def test_wings_meeting_at_an_angle_become_parts(self):
        # two 60 x 12 m wings meeting at 55 degrees (a V block no single box frame follows), 0.25 m cells
        cells = set()
        for ang in (0.0, 55.0):
            a = np.radians(ang)
            u, v = np.meshgrid(np.arange(0, 60, RES), np.arange(0, 12, RES))
            x = u * np.cos(a) - v * np.sin(a)
            y = u * np.sin(a) + v * np.cos(a)
            cells |= set(zip(np.round(x / RES).astype(int).ravel() + 400, np.round(y / RES).astype(int).ravel() + 400))
        xs, ys = (np.array(t) for t in zip(*sorted(cells)))
        self.assertEqual(len(survey._rect_split(xs, ys, RES)), 1)          # straight cuts do not help
        parts = survey._parts(xs, ys, RES)
        self.assertGreaterEqual(len(parts), 2)
        self.assertEqual(sum(len(p[0]) for p in parts), len(xs))
        big = sorted(len(p[0]) * RES * RES for p in parts)[-2:]
        self.assertTrue(all(a > 400 for a in big), big)                   # both wings, not slivers
        # a plain rectangle stays whole
        u, v = np.meshgrid(np.arange(0, 160), np.arange(0, 60))
        self.assertEqual(len(survey._parts(u.ravel(), v.ravel(), RES)), 1)


if __name__ == '__main__':
    unittest.main()
