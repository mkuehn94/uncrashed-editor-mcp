"""Offline tests of the build-vs-scan loop: asset profiles (visible size, symmetry, crown, panel tilt), sizing and
turning catalogue rows by visible size, object-level scan_compare with fix lines and the height raster,
capture_topdown (pixel grid, orthographic / reference-only capture words, old-DLL fallback) and scan_view --compare.
Never reaches the game: agent calls are patched or refused."""
import contextlib
import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from uncrashed_mcp import agent, live, reference
from uncrashed_mcp.catalog import load_catalog
from uncrashed_mcp.scan import composite, mapping, ops, profile, store, survey, topdown, view

from test_scan import NATIVE, Sandbox, make_scene, snapshot_file, spawns

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


# ---------------------------------------------------------------------------- synthetic meshes


def box_mesh(x0, y0, z0, x1, y1, z1, start=0):
    V = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    T = []
    for a, b, c, d in q:
        T += [(a + start, b + start, c + start), (a + start, c + start, d + start)]
    return V, T


def join(*parts):
    V, T = [], []
    for pv, pt in parts:
        off = len(V)
        V += pv
        T += [(a + off, b + off, c + off) for a, b, c in pt]
    return np.asarray(V, float), np.asarray(T, np.int64)


def tree_mesh(core=2.0, fringe=4.0, top=10.0, seed=2):
    """Many leaf cards in a 2 m crown, a few single cards out to 4 m (as game trees: the box reaches the outermost
    card), a thin trunk."""
    rng = np.random.default_rng(seed)
    parts = [box_mesh(-.15, -.15, 0, .15, .15, 5)]
    def card(cx, cy, cz):
        s = .35
        return ([(cx - s, cy - s, cz), (cx + s, cy - s, cz), (cx + s, cy + s, cz), (cx - s, cy + s, cz)],
                [(0, 1, 2), (0, 2, 3)])
    for _ in range(900):
        r, a = core * math.sqrt(rng.random()), rng.random() * 2 * math.pi
        parts.append(card(r * math.cos(a), r * math.sin(a), 5 + rng.random() * (top - 5)))
    for k in range(10):
        a = k * 2 * math.pi / 10
        r = core + (fringe - core) * rng.random()
        parts.append(card(r * math.cos(a), r * math.sin(a), 6 + 3 * rng.random()))
    return join(*parts)


def with_geometry(geo):
    return patch.object(profile, '_geometry', return_value=(*geo, 'test mesh'))


def fake_asset(row='T', name='test', tags=('prop',), size=(4, 2, 1), origin=(0, 0, 50), kind='mesh'):
    ext = tuple(v * 50 for v in size)
    return types.SimpleNamespace(id=row, name=name, tags=tags, size_m=size, origin_cm=origin, extent_cm=ext, kind=kind,
                                 source='SM_test', hidden=False, pivot='base', category='Props', material=None)


class ProfileTests(unittest.TestCase):
    def test_a_tree_crown_is_the_dense_silhouette_not_the_bounding_box(self):
        V, T = tree_mesh()
        span = V.max(0) - V.min(0)
        a = fake_asset(tags=('tree',), size=tuple(span))
        with with_geometry((V, T)):
            p = profile.measure(a)
        self.assertEqual((p['kind'], p['symmetry'], p['yaw_period_deg']), ('foliage', 'round', 0.0))
        self.assertTrue(3.8 < p['crown_m'] < 5.2, p['crown_m'])          # the 2 m dense core (+ card), not the fringe
        self.assertLess(p['crown_frac'], 0.65)
        self.assertAlmostEqual(p['crown_base_m'], 5.0, delta=0.8)
        self.assertGreater(p['visible_height_m'], 8.5)

    def test_symmetry_long_axis_and_footprint_of_solid_rows(self):
        with with_geometry(join(box_mesh(-2, -1, 0, 2, 1, 1))):
            plain = profile.measure(fake_asset())
        self.assertEqual((plain['symmetry'], plain['yaw_period_deg'], plain['long_axis_deg']), ('twofold', 180.0, 0.0))
        self.assertEqual(plain['footprint']['shape'], 'rect')
        self.assertAlmostEqual(plain['visible_m'][0], 4.0, delta=0.15)
        with with_geometry(join(box_mesh(-1, -1, 0, 1, 1, 1))):
            square = profile.measure(fake_asset(size=(2, 2, 1)))
        self.assertEqual((square['symmetry'], square['yaw_period_deg']), ('square', 90.0))
        # a car-like row: a cabin at the back makes it one-way (front by convention, period 360)
        with with_geometry(join(box_mesh(-2, -.9, 0, 2, .9, 1), box_mesh(-2, -.8, 1, 0, .8, 1.5))):
            car = profile.measure(fake_asset(size=(4, 1.8, 1.5)))
        self.assertEqual((car['symmetry'], car['yaw_period_deg']), ('none', 360.0))
        self.assertIn('+x', car['front'])
        # long along y although the (wrong) catalogue box says x: the silhouette decides
        with with_geometry(join(box_mesh(-.4, -1.5, 0, .4, 1.5, 1))):
            bin_ = profile.measure(fake_asset(size=(3.2, 3.0, 1)))
        self.assertEqual(bin_['long_axis_deg'], 90.0)

    def test_a_tilted_slab_is_a_panel_with_tilt_and_facing(self):
        t = math.radians(25)
        V = [(x, y * math.cos(t), y * math.sin(t) + 1) for x, y in ((-2, -.8), (2, -.8), (2, .8), (-2, .8))]
        with with_geometry(join((V, [(0, 1, 2), (0, 2, 3)]))):
            p = profile.measure(fake_asset(size=(4, 1.45, .7)))
        self.assertAlmostEqual(p['panel']['tilt_deg'], 25, delta=1.5)
        self.assertAlmostEqual(p['panel']['faces_deg'], 270, delta=2)    # the slab rises towards +y: looks to -y

    def test_profiles_are_cached_per_catalogue_version_and_refreshed_on_request(self):
        calls = []
        real = profile.measure

        def counting(a):
            calls.append(a.id)
            return real(a)
        with patch.object(profile, 'measure', counting):
            first = profile.asset_profile(['NewRow_52'])
            again = profile.asset_profile(['NewRow_52'])
            self.assertEqual(calls, ['NewRow_52'])
            profile.asset_profile(['NewRow_52'], refresh=True)
            self.assertEqual(calls, ['NewRow_52', 'NewRow_52'])
        self.assertEqual(first['rows'], again['rows'])
        self.assertTrue(Path(first['cache']).is_file())
        self.assertIn(profile.catalogue_version(), first['cache'])
        self.assertEqual(first['rows'][0]['source'], 'hull (data/meshes.bin)')   # paks patched away
        blueprint = next(a.id for a in load_catalog().assets.values() if a.kind == 'blueprint')
        self.assertEqual(profile.asset_profile([blueprint])['rows'][0]['source'], 'none')
        with self.assertRaisesRegex(ValueError, 'unknown catalogue rows'):
            profile.asset_profile(['NewRow_nope'])


# ---------------------------------------------------------------------------- sizing and turning by visible size


def tree_obj(**kw):
    base = {'id': 7, 'cls': 'tree', 'conf': .9, 'x': 0.0, 'y': 0.0, 'ground_z': 0.0, 'top_z': 10.0, 'height_m': 10.0,
            'length_m': 8.4, 'width_m': 7.6, 'crown_m': 7.0, 'yaw_deg': 0.0, 'rgb': [60, 120, 50]}
    return {**base, **kw}


def fake_profile(crown=None, visible=(4, 4, 4), long_deg=0.0, period=180.0, offset=(0.0, 0.0)):
    return {'source': 'test', 'visible_m': list(visible), 'long_axis_deg': long_deg, 'crown_m': crown,
            'yaw_period_deg': period, 'visible_center_m': list(offset)}


class VisibleSizingTests(unittest.TestCase):
    def test_trees_are_scaled_crown_to_crown_not_box_to_crown(self):
        a = load_catalog().assets['NewRow_169']          # Elm13, box 14.5 x 15.8 x 14.2
        prof = fake_profile(crown=10.0, visible=(13, 14, 13.0), period=0.0)
        with patch.object(profile, 'get', return_value=prof):
            f = mapping.fit_one(tree_obj(), a)
        self.assertEqual(f['sized_by'], 'visible')
        self.assertAlmostEqual(f['visible_m'][0], 7.0, delta=0.05)      # the visible crown is the scan crown
        self.assertAlmostEqual(f['visible_m'][2], 10.0, delta=0.05)
        self.assertGreater(f['size_m'][0], 7.0 * 1.3)                    # so the box is wider than the crown
        with patch.object(mapping, 'USE_PROFILES', False):              # the old rule: box onto the scan box
            old = mapping.fit_one(tree_obj(), a)
        self.assertLess(old['size_m'][0] * 10.0 / 14.47, 6.0)            # its visible crown: ~5.3 of 7 m

    def test_wide_low_crowns_do_not_squash_a_tree(self):
        a = load_catalog().assets['NewRow_169']
        with patch.object(profile, 'get', return_value=fake_profile(crown=10.0, visible=(13, 14, 13.0), period=0.0)):
            f = mapping.fit_one(tree_obj(crown_m=9.0, height_m=3.0), a)
        sxy, sz = f['scale'][0], f['scale'][2]
        self.assertLessEqual(sxy / sz, mapping.TREE_SPREAD[1] + 1e-6)
        self.assertGreaterEqual(sz, mapping.SCALING['tree'][1])
        self.assertGreater(f['fit_err_pct'], 20)                         # and the fit says it does not fit

    def test_yaw_follows_the_visible_long_axis(self):
        a = load_catalog().assets['NewRow_52']           # trash box: box long on x
        o = {'id': 3, 'cls': 'small_object', 'conf': .9, 'x': 0, 'y': 0, 'ground_z': 0, 'top_z': 1, 'height_m': 1.1,
             'length_m': 1.9, 'width_m': 1.1, 'yaw_deg': 30.0}
        with patch.object(profile, 'get', return_value=fake_profile(visible=(1.1, 1.9, 1.1), long_deg=90.0)):
            f = mapping.fit_one(o, a)
        self.assertAlmostEqual(f['yaw_scan'], 300.0)                     # its visible long axis is y: turned -90
        self.assertLess(f['fit_err_pct'], 3)

    def test_the_visible_centre_lands_on_the_object(self):
        a = load_catalog().assets['NewRow_52']
        line = mapping._spawn('NewRow_52', a, 10.0, 20.0, 0.0, 90.0, (1, 1, 1), 'g', offset=(0.5, 0.0))
        w = line.split()
        px, py = float(w[2]) / 100, float(w[3]) / 100
        cx = px + (a.origin_cm[0] / 100 + .5) * math.cos(math.pi / 2) - a.origin_cm[1] / 100 * math.sin(math.pi / 2)
        cy = py + (a.origin_cm[0] / 100 + .5) * math.sin(math.pi / 2) + a.origin_cm[1] / 100 * math.cos(math.pi / 2)
        self.assertAlmostEqual(cx, 10.0, delta=1e-3)
        self.assertAlmostEqual(cy, 20.0, delta=1e-3)


class RowChoiceTests(unittest.TestCase):
    def veh(self, cls, L, W, H):
        return {'id': 1, 'cls': cls, 'conf': .9, 'x': 0, 'y': 0, 'ground_z': 0, 'top_z': H, 'height_m': H,
                'length_m': L, 'width_m': W, 'yaw_deg': 0.0}

    def best(self, o):
        # catalogue boxes (as live LOD0 profiles measure these rows): the case where a Camaro x1.3 won the lorry
        with patch.object(mapping, 'USE_PROFILES', False):
            return mapping.ranked(o, mapping.candidates(o['cls']), 1)[0]

    def test_vans_and_trucks_get_van_bodied_rows_and_sedans_cars(self):
        cat = load_catalog().assets
        truck = self.best(self.veh('truck', 6.68, 2.41, 2.11))      # a lorry was a Camaro x1.3
        self.assertEqual(mapping.row_body(cat[truck['row']]), 'van', truck)
        van = self.best(self.veh('car', 5.13, 2.07, 2.01))           # a 2 m tall van was a Gt86
        self.assertEqual(mapping.row_body(cat[van['row']]), 'van', van)
        sedan = self.best(self.veh('car', 4.4, 1.9, 1.3))
        self.assertEqual(mapping.row_body(cat[sedan['row']]), 'car', sedan)
        self.assertIsNone(mapping.body(self.veh('car', 4.4, 1.9, 7.3)))   # a crown over it: height says nothing

    def test_tree_rows_that_look_like_rocks_or_nothing_are_left_out(self):
        real = profile.get

        def fake(row, refresh=False):
            if row == 'NewRow_179':
                return {'source': 'lod0 (game paks)', 'crown_base_m': 0.0, 'crown_frac': 0.95}
            if row == 'NewRow_267':
                return {'source': 'none'}
            return real(row)
        with patch.object(profile, 'get', fake):
            out = []
            rows = {a.id for a in mapping.candidates('tree', excluded=out)}
        self.assertNotIn('NewRow_179', rows)
        self.assertNotIn('NewRow_267', rows)
        self.assertIn('NewRow_169', rows)
        self.assertEqual({e['row'] for e in out}, {'NewRow_179', 'NewRow_267'})


class FacingTests(unittest.TestCase):
    def test_unreal_rotator_turns_the_up_axis(self):
        np.testing.assert_allclose(mapping._up_axis(0, 0, 0), [0, 0, 1], atol=1e-9)
        n = mapping._up_axis(0, -18.5, 12)        # roll leans the top towards local +y = yaw + 90
        self.assertAlmostEqual(math.degrees(math.atan2(n[1], n[0])) % 360, 71.5, delta=.01)
        self.assertAlmostEqual(math.degrees(math.acos(n[2])), 12, delta=.01)
        n = mapping._up_axis(10, 0, 0)            # pitch (nose up) leans it backwards
        self.assertLess(n[0], 0)

    def test_scan_facing_is_the_mean_normal_of_the_object_cells(self):
        g = survey.Grid(0, 0, .25, 40, 40)
        objid = np.full((40, 40), -1, np.int32)
        objid[10:20, 10:30] = 5
        t = math.radians(20)
        nrm = np.zeros((40, 40, 3), np.float32)
        nrm[...] = (0, math.sin(t), math.cos(t))                  # looks south (+y in the scan frame)
        ras = types.SimpleNamespace(g=g, objid=objid, nrm=nrm)
        tilt, faces = mapping._scan_facing(ras, {'id': 5, 'x': 5.0, 'y': 3.75, 'length_m': 5, 'width_m': 2.5})
        self.assertAlmostEqual(tilt, 20, delta=.2)
        self.assertAlmostEqual(faces, 90, delta=.2)
        nrm[...] = (0, 0, 1)
        self.assertIsNone(mapping._scan_facing(ras, {'id': 5, 'x': 5.0, 'y': 3.75, 'length_m': 5, 'width_m': 2.5})[1])


# ---------------------------------------------------------------------------- compare on the synthetic scene


def placed_from_plan(plan: Path) -> list[dict]:
    out = []
    for i, w in enumerate(spawns(plan)):
        xf = [float(v) for v in w[2:5]] + [0.0, float(w[6]), 0.0] + [float(v) for v in w[8:11]]
        out.append({'actor': f'{0x200 + i:x}', 'row': w[1], 'group': w[-1][6:], 'xf': xf})
    return out


def apply_fix(placed: list[dict], line: str) -> None:
    w = line.split()
    assert w[0] == 'transform', line
    p = next(p for p in placed if p['actor'] == w[1])
    for k, v in enumerate(w[2:11]):
        if v != '-':
            p['xf'][k] = float(v)


@unittest.skipUnless(NATIVE, 'needs g++ and libturbojpeg headers to build native/ucmesh and native/scanraster')
class CompareTests(Sandbox):
    @classmethod
    def setUpClass(cls):
        reference.tool()
        from uncrashed_mcp.scan import native
        native.library()

    def setUp(self):
        super().setUp()
        self.obj = make_scene(self.root / 'src')
        ops.scan_survey(str(self.obj), use_osm=False)
        ops.scan_place(at=[100, 200, 50], yaw=90, datum_m=0.0)
        self.objs = store.objects(store.resolve(''))
        plan = self.root / 'plan.txt'
        mapping.scan_plan(out=str(plan), classes=['car', 'tree'], min_conf=0)
        self.placed = placed_from_plan(plan)

    def of(self, cls):
        return sorted((o for o in self.objs if o['cls'] == cls), key=lambda o: o['id'])

    def compare(self, placed, **kw):
        snap = snapshot_file(self.root / f'snap{len(list(self.root.glob("snap*")))}.json.gz', placed)
        return mapping.scan_compare(str(snap), classes=['car', 'tree'], min_conf=0, **kw)

    def by_id(self, res):
        return {r['id']: r for r in json.loads(Path(res['report_json']).read_text())['objects']}

    def test_a_plan_built_as_written_matches(self):
        res = self.compare(self.placed)
        self.assertEqual(res['counts']['matched'], len(self.of('car')) + len(self.of('tree')), res['problems'])
        self.assertEqual(res['problems'], [])
        rows = self.by_id(res)
        tree = rows[self.of('tree')[0]['id']]
        self.assertAlmostEqual(tree['size_ratio'][0], 1.0, delta=0.05)    # visible crown vs scan crown

    def test_yaw_counts_modulo_symmetry_and_a_fix_line_repairs_turn_move_and_scale(self):
        car_a, car_b = self.of('car')[:2]
        tree = self.of('tree')[0]
        placed = json.loads(json.dumps(self.placed))
        pa = next(p for p in placed if p['group'] == f"scan.car.{car_a['id']}")
        pb = next(p for p in placed if p['group'] == f"scan.car.{car_b['id']}")
        pt = next(p for p in placed if p['group'] == f"scan.tree.{tree['id']}")
        pa['xf'][4] += 180                     # front / back: the scan box cannot tell, so no deviation
        pb['xf'][4] += 35
        pb['xf'][0] += 150
        a = load_catalog().assets[pt['row']]   # the test build's rule: the tree's bounding box = the scan crown
        pt['xf'][6:9] = [tree['crown_m'] / a.size_m[0], tree['crown_m'] / a.size_m[1], tree['height_m'] / a.size_m[2]]
        res = self.compare(placed)
        rows = self.by_id(res)
        self.assertEqual(rows[car_a['id']]['status'], 'matched')
        b = rows[car_b['id']]
        self.assertEqual(b['status'], 'offset')
        self.assertIn('yaw_off', b['issues'])
        self.assertAlmostEqual(b['yaw_off_deg'], 35, delta=2)
        t = rows[tree['id']]
        self.assertIn('size_off', t['issues'])
        self.assertLess(t['size_ratio'][0], 0.85)
        for r in (b, t):
            self.assertTrue(r['fix'].startswith(f"transform {r['actor']} "), r['fix'])
            apply_fix(placed, r['fix'])
        again = self.by_id(self.compare(placed))
        self.assertEqual(again[car_b['id']]['status'], 'matched', again[car_b['id']])
        self.assertEqual(again[tree['id']]['status'], 'matched', again[tree['id']])
        self.assertIsNone(again[car_a['id']]['fix'])

    def test_ground_steps_measure_the_seams_of_placed_tiles(self):
        plan = self.root / 'ground.txt'
        mapping.scan_plan(box=[72, 172, 128, 228], classes=['ground'], out=str(plan))
        tiles = [{'actor': f'{0x900 + i:x}', 'row': w[1], 'group': w[-1][6:], 'xf': [float(v) for v in w[2:11]]}
                 for i, w in enumerate(ln.split() for ln in plan.read_text().splitlines() if ln.startswith('spawn '))]
        snap = snapshot_file(self.root / 'g0.json.gz', tiles)
        st = mapping.scan_compare(str(snap), classes=['car'], min_conf=0)['ground_steps']
        self.assertEqual(st['tiles'], len(tiles))
        self.assertEqual(st['edges_over_1cm'], 0, st['worst'][:3])             # as planned: one smooth surface
        tiles[len(tiles) // 2]['xf'][2] += 5.0                                # one tile 5 cm too high
        snap = snapshot_file(self.root / 'g1.json.gz', tiles)
        st = mapping.scan_compare(str(snap), classes=['car'], min_conf=0)['ground_steps']
        self.assertGreater(st['edges_over_1cm'], 0)
        self.assertAlmostEqual(st['worst'][0]['dz_m'], 0.05, delta=0.01)
        self.assertIn(tiles[len(tiles) // 2]['actor'], st['worst'][0]['ids'])

    def test_height_cells_show_where_the_build_is_missing_height(self):
        tree = self.of('tree')[0]
        box = [70, 170, 130, 230]                                         # the whole scene, map metres
        full = self.compare(self.placed, box=box)['height_cells']
        placed = [p for p in self.placed if p['group'] != f"scan.tree.{tree['id']}"]
        res = self.compare(placed, box=box, top=60)
        cells = res['height_cells']
        self.assertGreater(cells['too_low'], full['too_low'])
        fr = store.Frame(store.resolve(''), 'map')
        tx, ty = fr.from_scan(tree['x'], tree['y'])
        report = json.loads(Path(res['report_json']).read_text())['height_cells']['worst']
        near = [c for c in report if math.hypot(c['x'] - tx, c['y'] - ty) < 5]   # (the unbuilt houses rank too)
        self.assertTrue(near, report)
        self.assertLess(min(c['diff_m'] for c in near), -3)
        self.assertEqual(self.by_id(res)[tree['id']]['status'], 'missing')

    def test_height_raster_is_clipped_to_the_survey_and_its_samples_capped(self):
        meta = store.resolve('')
        ext = mapping.survey_extent(meta, store.Frame(meta, 'map'))
        far = json.loads(json.dumps(self.placed))
        stray = json.loads(json.dumps(far[0]))
        stray['xf'][0] += 300000                                           # 3 km east, in cm
        stray['actor'] = 'stray_1'
        res = self.compare(far + [stray])
        box = json.loads(Path(res['report_json']).read_text())['box']
        self.assertTrue(ext[0] - 1e-6 <= box[0] and box[2] <= ext[2] + 1e-6, (box, ext))
        res = self.compare(self.placed, box=[-1e5, -1e5, 1e5, 1e5])        # a huge box: clipped, not sampled
        report = json.loads(Path(res['report_json']).read_text())
        self.assertEqual([round(v, 3) for v in report['box']], [round(v, 3) for v in ext])
        self.assertEqual(report['height_cells']['cell_m'], 5.0)
        res = self.compare(self.placed, box=[5000, 5000, 5100, 5100])      # no overlap with the scan
        self.assertIsNone(json.loads(Path(res['report_json']).read_text())['height_cells'])
        ras, fr, seen = ops.Rasters(meta), store.Frame(meta, 'map'), []
        cell = ras.cell
        ras.cell = lambda e, s: seen.append(len(e)) or cell(e, s)
        with patch.object(mapping, 'HEIGHT_SAMPLES_MAX', 10000, create=True):
            mapping._height_cells(ras, fr, [], (0, 0, 100, 100), 5.0, 1.0, 5)   # 400 cells
        self.assertLessEqual(seen[0], 10000)

    def test_compare_view_lays_scan_build_and_overlay_on_one_grid(self):
        res = self.compare(self.placed)
        box = [80, 170, 140, 230]
        mpp, W, H = view.crop_frame(box, 300)
        img = self.root / 'shm' / 'topdown-x.png'
        Image.fromarray(np.full((H, W, 3), 90, np.uint8)).save(img)
        img.with_suffix('.json').write_text(json.dumps({'box': box, 'width': W, 'height': H, 'px': 300,
                                                         'projection': 'orthographic'}))
        out = ops.scan_view(compare=str(img))
        self.assertEqual(out['build_projection'], 'orthographic')
        self.assertTrue(out['image'].startswith(str(store.SHM)))
        with Image.open(out['image']) as im:
            self.assertGreater(im.width, 3 * W)
        self.assertEqual(out['compare_report'], res['report_json'])
        with self.assertRaisesRegex(ValueError, 'drop box'):
            ops.scan_view(box=box, compare=str(img))
        Image.fromarray(np.zeros((H + 3, W, 3), np.uint8)).save(img)
        with self.assertRaisesRegex(ValueError, 'sidecar says'):
            ops.scan_view(compare=str(img))

    def test_overlay_marks_shared_edges(self):
        a = np.zeros((60, 60, 3), np.uint8)
        a[20:40, 20:40] = 255
        same, agree = topdown.overlay(a, a)
        self.assertGreater(agree, 0.95)
        b = np.zeros_like(a)
        b[5:15, 40:55] = 255
        _, apart = topdown.overlay(a, b)
        self.assertLess(apart, 0.1)


# ---------------------------------------------------------------------------- buildings made of many pieces


def piece(row, cx, cy, cz, size, yaw=0.0, group='g', actor='a'):
    """A placed object whose catalogue box is centred on cx, cy, cz (map metres) with size (x, y, z) metres."""
    a = load_catalog().assets[row]
    S = [size[i] / a.size_m[i] for i in range(3)]
    t = math.radians(yaw)
    ox, oy, oz = (a.origin_cm[i] / 100 * S[i] for i in range(3))
    px, py = cx - (ox * math.cos(t) - oy * math.sin(t)), cy - (ox * math.sin(t) + oy * math.cos(t))
    return {'actor': actor, 'row': row, 'group': group,
            'xf': [px * 100, py * 100, (cz - oz) * 100, 0.0, yaw, 0.0, *S]}


def house_pieces(x0, y0, x1, y1, ground, h, group, start=0):
    """Four walls (Concrete Wall01, long on local y) and a roof slab (Concrete Floor01), map metres."""
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    walls = [('NewRow_502', x0 + .15, cy, (0.3, y1 - y0, h), 0.0), ('NewRow_502', x1 - .15, cy, (0.3, y1 - y0, h), 0.0),
             ('NewRow_502', cx, y0 + .15, (0.3, x1 - x0, h), 90.0), ('NewRow_502', cx, y1 - .15, (0.3, x1 - x0, h), 90.0)]
    out = [piece(r, x, y, ground + h / 2, size, yaw, group, f'{start + k:x}') for k, (r, x, y, size, yaw) in enumerate(walls)]
    out.append(piece('NewRow_303', cx, cy, ground + h - .15, (x1 - x0, y1 - y0, 0.3), 0.0, group, f'{start + 9:x}'))
    return out


class TagAndKindTests(unittest.TestCase):
    def test_group_names_that_name_a_scan_object(self):
        known = {'building', 'car', 'tree', 'small_object'}
        cases = {'scan.building.75': ('building', 75), 'b104': ('b', 104), 'b57nw': ('b', 57), 'car.1002': ('car', 1002),
                 'scan.car.12': ('car', 12), 'building_75': ('building', 75), 'small_object.9': ('small_object', 9),
                 'Ground': None, 'marking': None, 'prop.extra': None, 'block2': None, 'b': None}
        for name, want in cases.items():
            self.assertEqual(composite.parse_tag(name, known), want, name)
        self.assertTrue(composite.tag_fits('b', 'bridge'))          # a hall the survey called a bridge
        self.assertFalse(composite.tag_fits('car', 'building'))
        self.assertTrue(composite.tag_fits('prop', 'small_object'))

    def test_ground_pieces_and_gameplay_rows_are_told_apart(self):
        cat = load_catalog()
        self.assertEqual(composite.kind_of(cat.assets['NewRow_17'], 2.0, 1.0), 'gameplay')        # ME Spawner
        self.assertEqual(composite.kind_of(cat.assets['NewRow_182'], 0.1, 0.05), 'ground')         # asphalt tile
        # a floor slab scaled 2 m deep (a foundation fill) whose top is at the ground is ground too
        self.assertEqual(composite.kind_of(cat.assets['NewRow_303'], 2.1, 0.02), 'ground')
        self.assertEqual(composite.kind_of(cat.assets['NewRow_303'], 0.3, 6.0), 'object')          # a roof slab
        self.assertEqual(composite.kind_of(cat.assets['NewRow_502'], 4.0, 4.0), 'object')          # a wall
        self.assertEqual(composite.kind_of(cat.assets['NewRow_663'], 0.4, 0.4), 'ground')          # lawn foliage

    def test_best_shift_finds_the_move_that_overlaps_best(self):
        scan = np.zeros((40, 40), bool)
        scan[10:30, 12:28] = True
        built = np.roll(scan, 4, axis=1)                    # 2 m east at 0.5 m cells
        dx, dy, v = composite._best_shift(built, scan, 0.5)
        self.assertEqual((dx, dy), (-2.0, 0.0))
        self.assertAlmostEqual(v, 1.0)


@unittest.skipUnless(NATIVE, 'needs g++ and libturbojpeg headers to build native/ucmesh and native/scanraster')
class BuildingCompareTests(Sandbox):
    """The flat house of the synthetic scene (scan e -20..-10, s -20..-12, 6 m) placed at 100,200,50 yaw 90 lies at
    map x 112..120, y 180..190, ground z 50."""
    @classmethod
    def setUpClass(cls):
        reference.tool()
        from uncrashed_mcp.scan import native
        native.library()

    def setUp(self):
        super().setUp()
        ops.scan_survey(str(make_scene(self.root / 'src')), use_osm=False)
        ops.scan_place(at=[100, 200, 50], yaw=90, datum_m=0.0)
        objs = store.objects(store.resolve(''))
        self.flat = min((o for o in objs if o['cls'] == 'building'), key=lambda o: o['x'])
        tiles = [piece('NewRow_182', 100 + 4 * i, 170, 50.05, (4, 4, 0.1), group='Ground', actor=f't{i}') for i in range(5)]
        tiles += [piece('NewRow_303', 104 + 4 * i, 196, 49.0, (4, 4, 2.0), group='Ground', actor=f'f{i}')
                  for i in range(3)]                       # deep fills whose top is the ground
        self.tiles = tiles

    def compare(self, placed, **kw):
        snap = snapshot_file(self.root / f'snap{len(list(self.root.glob("snap*")))}.json.gz', placed)
        return mapping.scan_compare(str(snap), classes=['building'], min_conf=0, **kw)

    def row(self, res, sid):
        return next(r for r in json.loads(Path(res['report_json']).read_text())['objects'] if r['id'] == sid)

    def test_a_house_of_walls_and_a_roof_pairs_by_its_group_and_matches(self):
        placed = house_pieces(112, 180, 120, 190, 50, 6, f"b{self.flat['id']}nw") + self.tiles
        res = self.compare(placed)
        r = self.row(res, self.flat['id'])
        self.assertEqual((r['status'], r['by'], r['members']), ('matched', 'group', 5), r)
        self.assertGreater(r['raster']['iou'], 0.8)
        self.assertLess(abs(r['raster']['dz_median_m']), 0.5)
        self.assertEqual(res['unmatched_placed'], [])                    # tiles are ground, counted only
        self.assertEqual(res['ground']['objects'], len(self.tiles))
        # the same pieces without a tag pair by footprint
        for p in placed[:5]:
            p['group'] = 'house_a'
        r = self.row(self.compare(placed), self.flat['id'])
        self.assertEqual((r['status'], r['by']), ('matched', 'footprint'), r)

    def test_a_shifted_house_is_an_offset_with_a_group_move(self):
        group = f"b{self.flat['id']}"
        placed = house_pieces(114, 180, 122, 190, 50, 6, group)          # 2 m too far east
        res = self.compare(placed)
        r = self.row(res, self.flat['id'])
        self.assertEqual(r['status'], 'offset', r)
        self.assertEqual(r['raster']['best_shift_m'], [-2.0, 0.0])
        self.assertEqual(r['fix'], f'edit_group {group} --delta -2,0,0')
        low = house_pieces(112, 180, 120, 190, 50, 4, group)              # 2 m too low
        r = self.row(self.compare(low), self.flat['id'])
        self.assertEqual(r['status'], 'height_off', r)
        self.assertAlmostEqual(r['height_off_m'], -2.0, delta=0.3)


class CompareLabelTests(unittest.TestCase):
    def report(self, n_extra=3000):
        objs = [{'id': i, 'cls': 'car', 'conf': .9, 'x': float(i), 'y': 0.0, 'status': st, 'issues': st,
                 'offset_m': off, 'height_off_m': 0.0, 'size_ratio': None, 'raster': None}
                for i, (st, off) in enumerate((('offset', 4.0), ('offset', 1.2), ('missing', None), ('matched', 0.1)))]
        extras = [{'row': 'NewRow_303', 'x': 1.0, 'y': 1.0, 'top_z': .1, 'base_above_scan_ground_m': -0.1}] * n_extra
        extras += [{'row': 'NewRow_182', 'x': 2.0, 'y': 2.0, 'top_z': .1, 'base_above_scan_ground_m': 0.0}]
        extras += [{'row': 'NewRow_0', 'x': 3.0, 'y': 3.0, 'top_z': 1.0, 'height_m': 0.8,
                    'base_above_scan_ground_m': 0.0}]
        return {'objects': objs, 'unmatched_placed': extras, 'created': 0, 'compare_box': [0, -10, 10, 10]}

    def test_only_the_worst_deviations_are_labelled_and_never_ground(self):
        marks, counts = topdown._marks({}, (-5, -5, 20, 20), self.report(), limit=25)
        texts = [m[2] for m in marks]
        self.assertEqual(texts[0].split()[:2], ['2', 'missing'])        # missing first, then the bigger offset
        self.assertEqual(texts[1].split()[:2], ['0', 'offset'])
        self.assertEqual(texts[-1], 'extra NewRow_0')                   # floor tiles and asphalt are ground
        self.assertEqual((len(marks), counts['deviations'], counts['extras']), (4, 3, 1))
        marks, _ = topdown._marks({}, (-5, -5, 20, 20), self.report(), limit=2)
        self.assertEqual(len(marks), 2)

    def test_stale_or_other_box_reports_warn(self):
        rep = self.report(0)
        self.assertEqual(topdown._report_warnings(rep, (0, 0, 10, 10), 30), [])
        w = topdown._report_warnings(rep, (0, 0, 10, 10), 6581)
        self.assertIn('110 min old', w[0])
        w = topdown._report_warnings(rep, (-50, -50, 50, 50), 30)
        self.assertIn('scan_compare --box -50,-50,50,50', w[0])


# ---------------------------------------------------------------------------- capture_topdown


class TopdownCaptureTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.sent = []

    @contextlib.contextmanager
    def guard(self, include=False):
        yield {'reference': 'included' if include else 'none'}

    def fake_run(self, old_dll=False):
        def run(command, timeout=None):
            self.sent.append(command)
            if command.startswith('capture start'):
                if old_dll and len(command.split()) > 11:
                    raise agent.AgentError('capture token expired or unknown')
                w, h = int(command.split()[2]), int(command.split()[3])
                self.size = (w, h)
                return ['capture token=7 width=%d height=%d backend=engine' % (w, h)]
            if command.startswith('capture export'):
                w = command.split()
                Image.fromarray(np.random.default_rng(1).integers(0, 255, (self.size[1], self.size[0], 3),
                                                                  dtype=np.uint8)).save(self.dir / w[-1])
            return []
        return run

    def patches(self, old_dll=False):
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(agent, 'run', side_effect=self.fake_run(old_dll)))
        stack.enter_context(patch.object(reference, 'capture_guard', self.guard))
        stack.enter_context(patch.object(reference, 'status', return_value={'loaded': True, 'hidden': False}))
        stack.enter_context(patch.object(live.time, 'sleep'))
        stack.enter_context(patch.object(agent, '_windows_path', lambda p: str(p)))
        return stack

    def test_the_capture_is_orthographic_on_the_scan_view_grid(self):
        box = (10.0, -40.0, 70.0, 0.0)
        with self.patches():
            res = live.capture_topdown(box, 600, ground_z=2.0, top_z=20.0, out_dir=self.dir)
        mpp, W, H = view.crop_frame(box, 600)
        self.assertEqual((res['width'], res['height'], res['m_per_px']), (W, H, mpp))
        self.assertEqual(res['projection'], 'orthographic')
        start = next(c for c in self.sent if c.startswith('capture start')).split()
        self.assertEqual(len(start), 12)                                 # + ortho width; mode all by default
        self.assertAlmostEqual(float(start[11]), W * mpp * 100, places=1)       # ortho width = the crop, cm
        self.assertEqual((float(start[7]), float(start[8])), (-90.0, -90.0))    # straight down, north up
        self.assertAlmostEqual(float(start[4]) / 100, box[0] + W * mpp / 2, places=3)
        self.assertGreater(float(start[6]) / 100, 20.0)
        with Image.open(res['image']) as im:
            self.assertEqual(im.size, (W, H))
        self.assertIsNone(res['parallax_m_per_10m_at_corner'])

    def test_reference_only_asks_the_native_show_only_list(self):
        with self.patches():
            res = live.capture_topdown((0, 0, 40, 40), 400, out_dir=self.dir, reference_mode='only')
        start = next(c for c in self.sent if c.startswith('capture start')).split()
        self.assertEqual(start[12], 'reference')
        self.assertEqual(res['projection'], 'orthographic')

    def test_an_old_agent_falls_back_to_a_narrow_perspective_and_says_so(self):
        with self.patches(old_dll=True):
            res = live.capture_topdown((0, 0, 60, 60), 600, ground_z=0.0, out_dir=self.dir)
        self.assertEqual(res['projection'], 'perspective')
        self.assertIn('agent DLL', res['note'])
        starts = [c.split() for c in self.sent if c.startswith('capture start')]
        self.assertEqual(len(starts[-1]), 11)                                 # the old command form
        self.assertAlmostEqual(float(starts[-1][10]), live.TOPDOWN_FOV)
        self.assertGreater(res['parallax_m_per_10m_at_corner'], 0)
        with self.patches(old_dll=True), self.assertRaises(live.CaptureModeUnsupported):
            live.capture_topdown((0, 0, 60, 60), 600, out_dir=self.dir, reference_mode='only')

    def test_ordinary_captures_keep_the_old_command(self):
        with self.patches():
            live._engine_capture_locked(self.dir / 'a.png', {'x': 1, 'y': 2, 'z': 3, 'pitch': -20, 'yaw': 0}, 64, 64)
        self.assertEqual(len(next(c for c in self.sent if c.startswith('capture start')).split()), 11)

    def test_bad_requests_are_refused_before_anything_is_sent(self):
        with self.patches():
            for kw, msg in (({'reference_mode': 'some'}, 'reference_mode'), ({'projection': 'fisheye'}, 'projection'),
                            ({'reference_mode': 'only', 'projection': 'perspective'}, 'reference-only')):
                with self.assertRaisesRegex(live.LiveError, msg):
                    live.capture_topdown((0, 0, 40, 40), 400, out_dir=self.dir, **kw)
        self.assertEqual(self.sent, [])


if __name__ == '__main__':
    unittest.main()
