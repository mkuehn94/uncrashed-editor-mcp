"""Offline tests of the scan_* survey on a synthetic textured OBJ (flat ground, a flat-roof house, a gabled house,
two cars, a tree) and fake OSM data. Never reaches the game or the network."""
import contextlib
import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from uncrashed_mcp import agent, api, groups, reference, snapshots
from uncrashed_mcp.catalog import load_catalog
from uncrashed_mcp.geom import world_aabb
from uncrashed_mcp.scan import facade, grid, mapping, native, ops, osm, store, survey, vision

NATIVE = bool(shutil.which('g++')) and Path('/usr/include/turbojpeg.h').is_file()
_guards = []


def setUpModule():
    for name in ('ensure_agent', '_execute'):
        guard = patch.object(agent, name, side_effect=AssertionError(f'agent.{name} reached from a test'))
        guard.start()
        _guards.append(guard)
    guard = patch.object(agent, 'session', contextlib.nullcontext)
    guard.start()
    _guards.append(guard)
    # asset profiles from data/meshes.bin only (installed game paks would change the numbers), cached in a temp dir
    from uncrashed_mcp.scan import profile
    folder = tempfile.mkdtemp(prefix='uncrashed-profiles-')
    for name, value in (('PAKS', Path(folder) / 'no-paks'), ('cache_dir', lambda: Path(folder) / 'profiles')):
        guard = patch.object(profile, name, value)
        guard.start()
        _guards.append(guard)
    profile._MEM.clear()
    profile.catalogue_version.cache_clear()
    profile._paks.cache_clear()


def tearDownModule():
    while _guards:
        _guards.pop().stop()


HALF = 30.0  # scene spans -30..30 m east and south


class Scene:
    """Triangles in (east, up, south) metres; texture colour by position."""

    def __init__(self):
        self.tris = []

    def quad(self, a, b, c, d):
        self.tris += [(a, b, c), (a, c, d)]

    def box(self, x0, z0, x1, z1, h, walls=False):
        self.quad((x0, h, z0), (x1, h, z0), (x1, h, z1), (x0, h, z1))
        if walls:
            for (a0, b0), (a1, b1) in (((x0, z0), (x1, z0)), ((x1, z0), (x1, z1)), ((x1, z1), (x0, z1)),
                                       ((x0, z1), (x0, z0))):
                self.quad((a0, 0, b0), (a1, 0, b1), (a1, h, b1), (a0, h, b0))

    def gable(self, x0, z0, x1, z1, eave, ridge):
        zm = (z0 + z1) / 2
        self.quad((x0, eave, z0), (x1, eave, z0), (x1, ridge, zm), (x0, ridge, zm))
        self.quad((x0, ridge, zm), (x1, ridge, zm), (x1, eave, z1), (x0, eave, z1))

    def ground(self, step=2.0):
        n = int(2 * HALF / step)
        for j in range(n):
            for i in range(n):
                x, z = -HALF + i * step, -HALF + j * step
                self.quad((x, 0, z), (x + step, 0, z), (x + step, 0, z + step), (x, 0, z + step))

    def tree(self, cx, cz, radius, top, base, seed=3):
        rng = np.random.default_rng(seed)
        rings, segs = 8, 24
        pts = {}
        for k in range(rings + 1):
            rr = radius * math.sin(math.pi / 2 * k / rings)
            for s in range(segs):
                a = 2 * math.pi * s / segs
                h = base + (top - base) * math.cos(math.pi / 2 * k / rings) + (rng.normal(0, 0.35) if k < rings else 0)
                pts[k, s] = (cx + rr * math.cos(a), h, cz + rr * math.sin(a))
        for k in range(rings):
            for s in range(segs):
                t = (s + 1) % segs
                self.quad(pts[k, s], pts[k, t], pts[k + 1, t], pts[k + 1, s])

    def write(self, folder: Path) -> Path:
        size = 512
        img = np.full((size, size, 3), 125, np.uint8)
        rng = np.random.default_rng(1)
        img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)

        def paint(x0, z0, x1, z1, rgb, noise=4):
            c0, c1 = int((x0 + HALF) / (2 * HALF) * size), int((x1 + HALF) / (2 * HALF) * size)
            r0, r1 = int((z0 + HALF) / (2 * HALF) * size), int((z1 + HALF) / (2 * HALF) * size)
            patch_ = np.array(rgb, float) + rng.normal(0, noise, (r1 - r0, c1 - c0, 3))
            img[r0:r1, c0:c1] = np.clip(patch_, 0, 255)

        paint(-20, -20, -10, -12, (85, 85, 90))       # flat roof
        paint(9, -19, 21, -11, (170, 70, 50))         # gabled roof
        paint(-13, 13, -2, 18, (40, 60, 160))          # cars
        paint(11, 11, 19, 19, (60, 150, 50), 18)       # tree crown
        Image.fromarray(img).save(folder / 'paint.jpg', quality=92)
        (folder / 'scene.mtl').write_text('newmtl paint\nmap_Kd paint.jpg\n')
        lines = ['mtllib scene.mtl', 'o scene']
        for tri in self.tris:
            for x, y, z in tri:
                lines.append(f'v {x:.4f} {y:.4f} {z:.4f}')
                lines.append(f'vt {(x + HALF) / (2 * HALF):.5f} {1 - (z + HALF) / (2 * HALF):.5f}')
        lines.append('usemtl paint')
        for i in range(len(self.tris)):
            a = 3 * i + 1
            lines.append(f'f {a}/{a} {a + 1}/{a + 1} {a + 2}/{a + 2}')
        path = folder / 'scene.obj'
        path.write_text('\n'.join(lines) + '\n')
        return path


def make_scene(folder: Path) -> Path:
    s = Scene()
    s.ground()
    s.box(-20, -20, -10, -12, 6.0, walls=True)  # flat-roof house 10 x 8, 6 m, with walls
    s.gable(9, -19, 21, -11, 5.0, 8.0)          # gabled house 12 x 8, eave 5, ridge 8 along east
    s.box(-12.5, 14.0, -8.0, 15.9, 1.5)         # car 4.5 x 1.9
    s.box(-6.5, 14.0, -2.0, 15.9, 1.5)          # car
    s.tree(15, 15, 3.5, 8.0, 3.0)
    return s.write(folder)


class Sandbox(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for target, name, value in ((store, 'SCAN_HOME', self.root / 'scan'), (store, 'SHM', self.root / 'shm' / 'scan'),
                                    (reference, 'CACHE', self.root / 'mesh'), (vision, 'VENV', self.root / 'no-venv')):
            p = patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)
        (self.root / 'shm').mkdir()
        (self.root / 'src').mkdir()


@unittest.skipUnless(NATIVE, 'needs g++ and libturbojpeg headers to build native/ucmesh and native/scanraster')
class SurveyTests(Sandbox):
    @classmethod
    def setUpClass(cls):
        reference.tool()
        native.library()

    def survey(self, **kw):
        self.obj = make_scene(self.root / 'src')
        return ops.scan_survey(str(self.obj), use_osm=False, **kw)

    def objs(self, cls):
        meta = store.resolve('')
        return [o for o in store.objects(meta) if o['cls'] == cls]

    def test_survey_finds_the_scene_and_stores_no_pixels_on_disk(self):
        out = self.survey()
        counts = {r['cls']: r['count'] for r in out['classes']}
        self.assertEqual(counts.get('building'), 2, out)
        self.assertEqual(counts.get('car'), 2, out)
        self.assertGreaterEqual(counts.get('tree', 0), 1, out)
        self.assertIn('osm: disabled (--no-osm)', out['skipped'])
        self.assertTrue(any(s.startswith('vision:') for s in out['skipped']))
        self.assertEqual(out['region_m']['radius'], 30.0)
        home = store.home(out['survey'])
        self.assertEqual(sorted(p.name for p in home.iterdir()), ['meta.json', 'objects-geometry.json', 'objects.json'])
        self.assertTrue(store.rasters_present(out['survey']))
        flat, gabled = sorted(self.objs('building'), key=lambda o: o['x'])
        self.assertEqual(flat['roof'], 'flat')
        self.assertAlmostEqual(flat['height_m'], 6.0, delta=0.3)
        self.assertAlmostEqual(flat['length_m'], 10.0, delta=0.6)
        self.assertAlmostEqual(flat['width_m'], 8.0, delta=0.6)
        self.assertEqual(gabled['roof'], 'gabled')
        self.assertAlmostEqual(gabled['ridge_yaw_deg'] % 180, 0, delta=5)
        self.assertAlmostEqual(gabled['ridge_m'], 8.0, delta=0.4)
        self.assertAlmostEqual(gabled['eave_m'], 5.4, delta=0.6)
        for car in self.objs('car'):
            self.assertAlmostEqual(car['length_m'], 4.5, delta=0.5)
            self.assertAlmostEqual(car['yaw_deg'] % 180, 0, delta=5)
        # the same survey is reused, and rebuilt into RAM when the rasters are gone
        again = ops.scan_survey(str(self.obj), use_osm=False)
        self.assertTrue(again['reused'])
        store.drop_ram(out['survey'])
        h = ops.scan_height([[-15, -16]])
        self.assertIn('rebuilt', h['note'])
        self.assertAlmostEqual(h['points'][0]['height_m'], 6.0, delta=0.2)

    def test_point_line_and_area_questions(self):
        self.survey()
        h = ops.scan_height([[-15, -16], [0, 0], [100, 100]])
        house, ground, outside = h['points']
        self.assertEqual(h['frame'], 'scan')
        self.assertEqual((house['surface'], house['cls']), ('building', 'building'))
        self.assertAlmostEqual(house['top_z'], 6.0, delta=0.1)
        self.assertAlmostEqual(ground['height_m'], 0.0, delta=0.1)
        self.assertEqual(outside['surface'], 'no_data')
        prof = ops.scan_profile([-25, -16], [-5, -16], step_m=1)
        self.assertEqual(len(prof['samples']), 21)
        self.assertAlmostEqual(prof['top_max_z'], 6.0, delta=0.1)
        free = ops.scan_free([[-25, -16, 10], [-5, -16, 10]])
        self.assertTrue(free['clear'])
        self.assertAlmostEqual(free['min_clearance_m'], 4.0, delta=0.15)
        blocked = ops.scan_free([[-25, -16, 4], [-5, -16, 4]], radius_m=0.5)
        self.assertFalse(blocked['clear'])
        self.assertEqual(blocked['first_blocked']['cls'], 'building')
        self.assertAlmostEqual(blocked['first_blocked']['x'], -20.5, delta=0.6)
        cars = ops.scan_objects(classes=['car'], near=[-7, 15], radius_m=10)
        self.assertEqual(cars['total'], 2)
        self.assertEqual([r['cls'] for r in cars['objects']], ['car', 'car'])
        self.assertEqual(ops.scan_objects(classes=['boat'])['unknown_classes'], ['boat'])
        at = ops.scan_at([-10, 15], radius_m=3)
        self.assertEqual({o['cls'] for o in at['objects']}, {'car'})
        self.assertEqual(at['objects'][0]['dist_m'], 0.0)
        gnd = ops.scan_ground([-25, -25, 25, 25], min_area_m2=30)
        self.assertAlmostEqual(sum(gnd['share'].values()), 1.0, delta=0.01)
        self.assertIn('building', gnd['share'])
        bid = next(o['id'] for o in self.objs('building') if o['roof'] == 'flat')
        b = ops.scan_building(bid)
        self.assertEqual(len(b['facades']), 4, b['facades'])
        faces = sorted(f['faces_deg'] for f in b['facades'])
        for got, want in zip(faces, (0, 90, 180, 270)):
            self.assertAlmostEqual(got, want, delta=4)
        self.assertFalse(b['facades_surveyed'])
        with self.assertRaisesRegex(ValueError, 'not a building'):
            ops.scan_building(self.objs('car')[0]['id'])

    def test_placement_moves_every_answer_into_the_map_frame(self):
        self.survey()
        before = ops.scan_objects(classes=['building'])['objects']
        placed = ops.scan_place(at=[100, 200, 50], yaw=90, datum_m=0.0)
        self.assertEqual(placed['frame'], 'map')
        after = ops.scan_objects(classes=['building'])['objects']
        for a, b in zip(sorted(before, key=lambda r: r['id']), sorted(after, key=lambda r: r['id'])):
            # yaw 90: east -> +y (south), south -> -x
            self.assertAlmostEqual(b['x'], 100 - a['y'], delta=0.01)
            self.assertAlmostEqual(b['y'], 200 + a['x'], delta=0.01)
            self.assertAlmostEqual(b['ground_z'], 50 + a['ground_z'], delta=0.01)
            self.assertAlmostEqual(b['yaw_deg'], (a['yaw_deg'] + 90) % 180, delta=0.11)
        # map point of the flat house centre (-15, -16) -> (116, 185)
        h = ops.scan_height([[116, 185]])
        self.assertEqual(h['frame'], 'map')
        self.assertAlmostEqual(h['points'][0]['top_z'], 56.0, delta=0.1)
        self.assertEqual(ops.scan_height([[-15, -16]], frame='scan')['points'][0]['cls'], 'building')
        self.assertTrue(ops.scan_place(clear=True)['cleared'])
        with self.assertRaisesRegex(ValueError, 'no placement'):
            ops.scan_height([[0, 0]], frame='map')

    def test_view_writes_a_png_to_ram_only(self):
        out = self.survey()
        for layer in ('ortho', 'ndsm', 'classes'):
            v = ops.scan_view(center=[0, 0], size_m=50, layer=layer, px=400)
            path = Path(v['image'])
            self.assertTrue(path.is_file())
            self.assertTrue(str(path).startswith(str(store.SHM)))
            with Image.open(path) as im:
                self.assertGreaterEqual(im.width, 400)
            self.assertGreaterEqual(v['objects_drawn'], 4)
        self.assertFalse(list(store.home(out['survey']).glob('*.png')))
        with self.assertRaisesRegex(ValueError, 'layer must be'):
            ops.scan_view(layer='lidar')

    def test_a_changed_obj_is_refused_not_mixed_with_the_old_objects(self):
        out = self.survey()
        n = sum(1 for line in self.obj.read_text().splitlines() if line.startswith('v '))
        with open(self.obj, 'a') as f:   # a re-export: a 10 m slab over the origin
            for x, z in ((-2, -2), (2, -2), (2, 2), (-2, 2)):
                f.write(f'v {x} 10 {z}\nvt 0.5 0.5\n')
            f.write(''.join(f'f {n + a}/{n + a} {n + b}/{n + b} {n + c}/{n + c}\n' for a, b, c in ((1, 2, 3), (1, 3, 4))))
        store.drop_ram(out['survey'])
        for ask in (lambda: ops.scan_height([[0, 0]]), lambda: ops.scan_objects(), lambda: ops.scan_at([0, 0])):
            with self.assertRaisesRegex(ValueError, f"OBJ changed since survey {out['survey']}: run scan_survey "):
                ask()
        self.assertFalse(store.rasters_present(out['survey']))   # nothing was rebuilt from the new mesh
        with self.assertRaisesRegex(ValueError, 'OBJ changed'):
            ops._build(store.resolve('', fresh=False), rebuild=True)
        row = ops.scan_status()['surveys'][0]
        self.assertEqual((row['survey'], row['stale']), (out['survey'], 'OBJ changed'))
        again = ops.scan_survey(str(self.obj), use_osm=False)     # a new survey of the new file answers again
        self.assertNotEqual(again['survey'], out['survey'])
        self.assertAlmostEqual(ops.scan_height([[0, 0]])['points'][0]['top_z'], 10.0, delta=0.1)

    def test_broken_ram_rasters_are_rebuilt_not_trusted(self):
        out = self.survey()
        k = out['survey']
        dsm = store.ram(k) / 'dsm.npy'
        with open(dsm, 'r+b') as f:   # an interrupted write / full tmpfs
            f.truncate(dsm.stat().st_size // 2)
        self.assertTrue(store.rasters_present(k))
        self.assertFalse(store.rasters_ok(k))
        h = ops.scan_height([[-15, -16]])
        self.assertIn('rebuilt', h['note'])
        self.assertAlmostEqual(h['points'][0]['top_z'], 6.0, delta=0.1)
        self.assertTrue(store.rasters_ok(k))
        # a build without its completion marker counts as absent; survey reuse repairs it
        (store.ram(k) / store.COMPLETE).unlink()
        self.assertFalse(ops.scan_status()['surveys'][0]['in_ram'])
        self.assertTrue(ops.scan_survey(str(self.obj), use_osm=False)['reused'])
        self.assertTrue(store.rasters_ok(k))
        self.assertEqual(sorted(p.name for p in store.ram(k).glob('*.tmp')), [])
        # a mapped reader keeps its file when a writer replaces it
        ras = ops.Rasters(store.resolve(''))
        before = float(ras.dsm[0, 0])
        store.save_array(k, 'dsm', np.zeros((2, 2), np.float32))
        self.assertEqual(float(ras.dsm[0, 0]), before)

    def test_a_rebuild_that_disagrees_is_reported(self):
        out = self.survey()
        store.write_json(store.home(out['survey']) / 'rebuild-mismatch.json', {'stored': 7, 'rebuilt': 6, 'at': 0})
        self.assertIn('rebuilt with 6 objects, the survey stored 7', ops.scan_height([[0, 0]])['warning'])
        store.drop_ram(out['survey'])
        h = ops.scan_height([[0, 0]])           # a rebuild that agrees clears it
        self.assertIn('rebuilt', h['note'])
        self.assertNotIn('warning', h)

    def test_shares_only_count_cells_near_the_true_point(self):
        self.survey()
        far = ops.scan_at([1000, 1000], radius_m=5)
        self.assertEqual((far['here']['surface'], far['surface_share']), ('no_data', {}))
        self.assertIn('outside the survey region', far['note'])
        edge = ops.scan_at([29.5, 0], radius_m=2)   # half the disc is off the grid: shares of the half on it
        self.assertAlmostEqual(sum(edge['surface_share'].values()), 1.0, delta=0.03)
        self.assertNotIn('note', edge)

    def test_status_frees_ram_and_lists_images(self):
        out = self.survey()
        ops.scan_view(center=[0, 0], size_m=20, px=200)
        st = ops.scan_status()
        self.assertEqual(st['images']['count'], 1)
        self.assertGreater(st['surveys'][0]['ram_mb'], 0)
        self.assertIn('scan: export east, south, up', st['frame'])
        self.assertEqual(ops.scan_status(drop_ram='all')['dropped_ram'], [out['survey']])
        self.assertFalse(store.ram(out['survey']).exists())
        with self.assertRaisesRegex(ValueError, 'matches 0 surveys'):
            ops.scan_status(drop_ram='zzz')

    def test_objects_sort_flag_and_point_to_the_nearest_miss(self):
        self.survey()
        tall = ops.scan_objects(sort='height', limit=3)['objects']
        self.assertEqual([r['height_m'] for r in tall], sorted((r['height_m'] for r in tall), reverse=True))
        self.assertGreater(tall[0]['height_m'], 7.5)   # the tree or the gabled ridge, not the default conf order
        miss = ops.scan_objects(classes=['car'], near=[25, -25], radius_m=5)
        self.assertEqual(miss['total'], 0)
        self.assertEqual(miss['nearest_outside']['cls'], 'car')
        self.assertGreater(miss['nearest_outside']['dist_m'], 5)
        self.assertIn('flag', miss['nearest_outside'])
        cars = ops.scan_objects(classes=['car'])['objects']
        self.assertEqual([r['flag'] for r in cars], [None, None])
        flat = {**store.objects(store.resolve(''))[0], 'cls': 'car', 'height_m': 0.03, 'src': 'vision', 'conf': 0.33}
        self.assertEqual(ops._flag(flat), 'flat (0.03 m); weak, vision only')
        windows = ops.scan_objects(classes=['building'], sort='windows')
        self.assertIn('no facade survey', windows['note'])
        self.assertEqual(windows['objects'][0]['windows'], None)
        with self.assertRaisesRegex(ValueError, 'sort near needs near'):
            ops.scan_objects(sort='near')

    def test_free_above_ground_and_located_slopes(self):
        self.survey()
        free = ops.scan_free([[-28, -16, 8], [-2, -16, 8]], agl=True)   # 8 m above ground: over the 6 m roof
        self.assertTrue(free['clear'])
        self.assertAlmostEqual(free['min_clearance_m'], 2.0, delta=0.2)
        self.assertIn('above the ground', free['z_in'])
        self.assertFalse(ops.scan_free([[-28, -16, 5], [-2, -16, 5]], agl=True)['clear'])
        prof = ops.scan_profile([-25, -16], [-5, -16], step_m=1)
        # only open ground counts: the 6 m roof edges are not ground slopes
        self.assertLess(prof['max_ground_slope_pct'], 5)
        self.assertNotIn(prof['max_ground_slope_at']['surface'], ('building', 'vegetation'))
        on_roof = [r for r in prof['samples'] if r['surface'] == 'building']
        self.assertTrue(on_roof and all(r['slope_pct'] is None for r in on_roof))

    def test_view_lists_what_its_labels_are(self):
        self.survey()
        v = ops.scan_view(center=[0, 0], size_m=60, px=300)
        self.assertEqual(len(v['objects']), v['objects_drawn'])
        self.assertEqual(set(v['objects'][0]), {'id', 'cls', 'conf', 'height_m', 'flag'})
        self.assertIn('car', {r['cls'] for r in v['objects']})

    def test_status_lists_the_survey(self):
        out = self.survey()
        st = ops.scan_status()
        self.assertEqual(st['current'], out['survey'])
        self.assertEqual(st['surveys'][0]['in_ram'], True)
        self.assertIn('vision', st)


@unittest.skipUnless(NATIVE, 'needs g++ and libturbojpeg headers to build native/scanraster')
class OsmPriorTests(unittest.TestCase):
    """Two touching flat roofs of the same height merge into one building; mapped footprints split them."""

    def rasters(self):
        res, n = 0.25, 160
        dsm = np.zeros((n, n), np.float32)
        dsm[40:80, 40:120] = 7.0     # terrace: two 10 m houses side by side (cells 40..80 = 10 m)
        nrm = np.zeros((n, n, 3), np.float32)
        nrm[..., 2] = 1
        rgb = np.full((n, n, 3), 120, np.uint8)
        return survey.Grid(-20, -20, res, n, n), dsm, nrm, rgb

    def test_footprints_split_a_terrace(self):
        g, dsm, nrm, rgb = self.rasters()
        alone = survey.Survey(g, dsm, nrm, rgb).run()
        self.assertEqual(sum(o['cls'] == 'building' for o in alone.objects), 1)
        feats = {'buildings': [
            {'osm': 'way/1', 'rings': [[(-10, -10), (0, -10), (0, 0), (-10, 0)]], 'tags': {'building': 'house'}},
            {'osm': 'way/2', 'rings': [[(0, -10), (10, -10), (10, 0), (0, 0)]], 'tags': {'building': 'house', 'levels': '2'}}],
            'roads': [{'line': [(-20, 10), (20, 10)], 'width': 6.0}], 'trees': []}
        sv = survey.Survey(g, dsm, nrm, rgb, feats).run()
        blds = sorted((o for o in sv.objects if o['cls'] == 'building'), key=lambda o: o['x'])
        self.assertEqual(len(blds), 2)
        self.assertEqual([b['osm']['osm'] for b in blds], ['way/1', 'way/2'])
        self.assertTrue(all(b['src'] == 'mesh+osm' and b['split_by_osm'] for b in blds))
        self.assertAlmostEqual(blds[0]['length_m'], 10.0, delta=0.6)
        self.assertEqual(sv.osm_info['shift_m'], [0.0, 0.0])
        road = sv.cls[int((10 + 20) / 0.25), 80]
        self.assertEqual(survey.CLASSES[int(road)], 'road')

    def test_parse_projects_and_joins_multipolygons(self):
        lat0, lon0 = 52.0, 9.0
        lat = lambda s: lat0 - s / 111_260.0  # noqa: E731 - metres south -> degrees (approx.)
        lon = lambda e: lon0 + e / (111_320.0 * math.cos(math.radians(lat0)))  # noqa: E731
        nodes = [(1, 0, 0), (2, 10, 0), (3, 10, 10), (4, 0, 10), (5, 50, 50)]
        elements = [{'type': 'node', 'id': i, 'lat': lat(s), 'lon': lon(e)} for i, e, s in nodes]
        elements[-1]['tags'] = {'natural': 'tree'}
        elements += [{'type': 'way', 'id': 10, 'nodes': [1, 2, 3], 'tags': {}},
                     {'type': 'way', 'id': 11, 'nodes': [3, 4, 1], 'tags': {}},
                     {'type': 'relation', 'id': 20, 'tags': {'building': 'yes', 'building:levels': '3'},
                      'members': [{'type': 'way', 'ref': 10, 'role': 'outer'}, {'type': 'way', 'ref': 11, 'role': 'outer'}]},
                     {'type': 'way', 'id': 12, 'nodes': [1, 2], 'tags': {'highway': 'residential', 'lanes': '2'}}]
        feats = osm.parse({'elements': elements}, lat0, lon0)
        self.assertEqual(osm.counts(feats), {'buildings': 1, 'parking': 0, 'water': 0, 'roads': 1, 'paths': 0, 'trees': 1})
        ring = np.array(feats['buildings'][0]['rings'][0])
        self.assertEqual(len(ring), 5)
        self.assertAlmostEqual(ring[:, 0].max(), 10, delta=0.1)
        self.assertAlmostEqual(ring[:, 1].max(), 10, delta=0.1)
        self.assertEqual(feats['buildings'][0]['tags'], {'building': 'yes', 'levels': '3'})
        self.assertEqual(feats['roads'][0]['width'], 6.4)
        self.assertAlmostEqual(feats['trees'][0][0], 50, delta=0.2)
        box = osm.bbox(-10, -10, 10, 10, lat0, lon0)
        self.assertLess(box[0], lat0)
        self.assertGreater(box[2], lat0)

    def test_fetch_uses_the_cache_and_never_the_network_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'osm.json'
            path.write_text(json.dumps({'elements': [], 'source': 'cached'}))
            with patch('urllib.request.urlopen', side_effect=AssertionError('network')), \
                    patch('socket.create_connection', side_effect=AssertionError('network')):
                self.assertEqual(osm.fetch(path, (52, 9, 52.01, 9.01))['source'], 'cached')

    def test_a_dead_network_fails_after_the_short_probe(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, clear=False) as env:
            env.pop('UNCRASHED_OVERPASS', None)
            env.pop('UNCRASHED_OSM_MAP_API', None)
            with patch('urllib.request.urlopen', side_effect=AssertionError('long request')), \
                    patch('socket.create_connection', side_effect=OSError('unreachable')) as probe:
                with self.assertRaisesRegex(osm.OsmError, r'no OSM source answered after \d+\.\d s .*no connection within 3 s'):
                    osm.fetch(Path(tmp) / 'new.json', (52, 9, 52.01, 9.01))
            self.assertEqual([c.args[0][0] for c in probe.call_args_list],
                             ['overpass-api.de', 'overpass.private.coffee', 'api.openstreetmap.org'])
            self.assertEqual(probe.call_args_list[0].kwargs['timeout'], osm.PROBE_S)

    def test_a_configured_overpass_is_the_only_place_the_bbox_goes(self):
        sent = []

        def urlopen(request, timeout=None):
            sent.append(request.full_url)
            raise OSError('refused')

        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {'UNCRASHED_OVERPASS': 'http://127.0.0.1:9/api/interpreter'}) as env, \
                patch('urllib.request.urlopen', urlopen), patch.object(osm, '_unreachable', return_value=None):
            env.pop('UNCRASHED_OSM_MAP_API', None)
            self.assertEqual(osm.sources(), (['http://127.0.0.1:9/api/interpreter'], ''))
            with self.assertRaises(osm.OsmError):
                osm.fetch(Path(tmp) / 'new.json', (52, 9, 52.001, 9.001))
            self.assertEqual(sent, ['http://127.0.0.1:9/api/interpreter'])   # no api.openstreetmap.org fallback
            env['UNCRASHED_OSM_MAP_API'] = 'http://127.0.0.1:9/map'
            self.assertEqual(osm.sources()[1], 'http://127.0.0.1:9/map')
        with patch.dict(os.environ, {'UNCRASHED_OSM_MAP_API': ''}) as env:
            env.pop('UNCRASHED_OVERPASS', None)
            self.assertEqual(osm.sources(), (list(osm.OVERPASS), ''))
        with patch.dict(os.environ, {'UNCRASHED_OSM': '0'}):
            self.assertEqual(osm.disabled(), 'disabled by UNCRASHED_OSM=0')

    def test_xml_answers_become_elements(self):
        xml = (b'<osm><node id="1" lat="52.0" lon="9.0"><tag k="natural" v="tree"/></node>'
               b'<way id="2"><nd ref="1"/><nd ref="1"/><tag k="highway" v="footway"/></way>'
               b'<relation id="3"><member type="way" ref="2" role="outer"/><tag k="building" v="yes"/></relation></osm>')
        els = osm.from_xml(xml)['elements']
        self.assertEqual([e['type'] for e in els], ['node', 'way', 'relation'])
        self.assertEqual(els[1]['nodes'], [1, 1])
        self.assertEqual(els[2]['members'][0], {'type': 'way', 'ref': 2, 'role': 'outer'})


def _px(e, s):
    """Scan metres -> pixel of the single fine ortho tile of the synthetic scene (origin -30, -30; 0.125 m)."""
    return (e + HALF) / 0.125, (s + HALF) / 0.125


def _car_px(cx, cs, length=4.5, width=1.9):
    return [list(_px(cx + dx, cs + dy)) for dx, dy in ((length / 2, width / 2), (-length / 2, width / 2),
                                                        (-length / 2, -width / 2), (length / 2, -width / 2))]


def fake_worker(job, work):
    """Stands in for .venv-vision: vehicles on the first car, on open ground and on the flat roof; on every facade
    two rows of two windows and a door."""
    out = {'device': 'cuda:1', 'gpu': 'fake', 'timings': {}, 'models': {}, 'errors': []}
    tiles = job['aerial']['tiles']
    first = [{'class': 'small vehicle', 'score': 0.9, 'obb': _car_px(-10.25, 14.95)},
             {'class': 'small vehicle', 'score': 0.8, 'obb': _car_px(0.0, 0.0)},
             {'class': 'small vehicle', 'score': 0.7, 'obb': _car_px(-15.0, -16.0)},
             {'class': 'swimming pool', 'score': 0.8, 'obb': _car_px(5.0, 22.0, 6.0, 3.0)}]
    out['obb'] = [first] + [[] for _ in tiles[1:]]
    out['coarse'] = [[] for _ in job['aerial']['coarse']]
    out['dino'] = [[] for _ in tiles]
    fac = []
    for path in job['facades']['images']:
        with Image.open(path) as im:
            W, H = im.size
        boxes = [[f * W, g * H, (f + 0.1) * W, (g + 0.12) * H] for f in (0.25, 0.6) for g in (0.25, 0.55)]
        fac.append([{'class': 'window', 'score': 0.6, 'box': b} for b in boxes] +
                   [{'class': 'door', 'score': 0.5, 'box': [0.45 * W, 0.78 * H, 0.52 * W, 0.93 * H]}])
    out['facades'] = fac
    return out


@unittest.skipUnless(NATIVE, 'needs g++ and libturbojpeg headers to build native/ucmesh and native/scanraster')
class VisionStageTests(Sandbox):
    @classmethod
    def setUpClass(cls):
        reference.tool()
        native.library()

    def setUp(self):
        super().setUp()
        (self.root / 'no-venv' / 'bin').mkdir(parents=True)
        (self.root / 'no-venv' / 'bin' / 'python').write_text('')  # present, but never run: _worker is faked
        p = patch.object(vision, '_worker', side_effect=fake_worker)
        self.worker = p.start()
        self.addCleanup(p.stop)
        self.obj = make_scene(self.root / 'src')

    def test_detections_confirm_reject_and_add_vehicles(self):
        out = ops.scan_survey(str(self.obj), use_osm=False)
        self.assertFalse(any(x.startswith('vision:') for x in out['skipped']), out['skipped'])
        self.assertEqual(out['vision']['aerial']['cars_confirmed'], 1)
        objs = store.objects(store.resolve(''))
        cars = [o for o in objs if o['cls'] == 'car']
        confirmed = [o for o in cars if o['src'] == 'mesh+vision']
        added = [o for o in cars if o['src'] == 'vision']
        self.assertEqual(len(confirmed), 1)
        self.assertAlmostEqual(confirmed[0]['x'], -10.25, delta=0.3)
        self.assertGreaterEqual(confirmed[0]['conf'], 0.9)
        self.assertEqual(confirmed[0]['models'], ['yolo11l-obb:0.90'])
        self.assertEqual(len(added), 1)
        self.assertAlmostEqual(added[0]['x'], 0.0, delta=0.2)
        self.assertAlmostEqual(added[0]['length_m'], 4.5, delta=0.1)
        rejected = [o for o in objs if o.get('was') == 'car']
        self.assertEqual(len(rejected), 1)                      # the second car: no vehicle in the image
        self.assertEqual(out['vision']['aerial']['vehicles_dropped_on_roofs'], 1)
        pool = [o for o in objs if o['cls'] == 'pool']
        self.assertEqual(len(pool), 1)
        # the id raster knows the added car; running the survey again does not repeat the stage
        h = ops.scan_height([[0.0, 0.0]])
        self.assertEqual(h['points'][0]['object'], added[0]['id'])
        ops.scan_survey(str(self.obj), use_osm=False)
        self.assertEqual(self.worker.call_count, 1)
        store.drop_ram(out['survey'])
        self.assertEqual(ops.scan_height([[0.0, 0.0]])['points'][0]['object'], added[0]['id'])

    def test_facades_give_window_rows_and_storeys(self):
        ops.scan_survey(str(self.obj), use_osm=False)
        bid = next(o['id'] for o in store.objects(store.resolve('')) if o['cls'] == 'building' and o['roof'] == 'flat')
        b = ops.scan_building(bid)
        self.assertTrue(b['facades_surveyed'])
        looked = [f for f in b['facades'] if f['status'] == 'surveyed']
        self.assertEqual(len(looked), 4, b['facades'])
        for f in looked:
            self.assertEqual((f['windows'], f['columns'], f['doors'], f['storeys']), (4, 2, 1, 2), f)
            self.assertLess(f['sill_m'][0], f['head_m'][0])
            self.assertLess(f['head_m'][0], f['sill_m'][1])
        self.assertEqual(b['storeys'], 2)
        v = ops.scan_facade(bid, side=looked[0]['side'])
        self.assertTrue(Path(v['image']).is_file() and str(v['image']).startswith(str(store.SHM)))
        self.assertEqual(len(v['facades']), 1)
        with self.assertRaisesRegex(ValueError, 'has sides'):
            ops.scan_facade(bid, side=99)
        self.assertFalse(list(store.home(store.resolve('')['key']).glob('*.png')))

    def test_worker_images_leave_ram_after_the_run(self):
        out = ops.scan_survey(str(self.obj), use_osm=False)
        work = store.ram(out['survey']) / 'vision'
        self.assertEqual(self.worker.call_count, 1)
        self.assertFalse((work / 'tiles').exists() or (work / 'facades').exists())
        self.assertFalse(list(store.ram(out['survey']).rglob('*.png')))

    def test_a_worker_failure_keeps_the_geometry(self):
        self.worker.side_effect = vision.VisionError('vision worker failed (exit 1): CUDA out of memory')
        out = ops.scan_survey(str(self.obj), use_osm=False)
        self.assertIn('vision: vision worker failed (exit 1): CUDA out of memory', out['skipped'])
        self.assertEqual({r['cls']: r['count'] for r in out['classes']}.get('car'), 2)
        with self.assertRaisesRegex(ValueError, 'vision requested but unavailable'):
            with patch.object(vision, 'VENV', self.root / 'missing'):
                ops.scan_survey(str(self.obj), use_osm=False, use_vision=True, refresh=True)


class FacadeHelperTests(unittest.TestCase):
    def test_walls_merge_a_jaggy_outline_and_face_outwards(self):
        ring = [(0, 0), (5, 0.2), (10, 0), (10, 6), (5, 6.1), (0, 6)]   # a 10 x 6 box with wobbly long sides
        walls = facade.walls(ring)
        self.assertEqual(len(walls), 4)
        faces = sorted(round(w['faces_deg']) % 360 for w in walls)
        for got, want in zip(faces, (0, 90, 180, 270)):
            self.assertAlmostEqual(got, want, delta=3)
        for w in walls:  # seen from outside, u runs to the viewer's right
            right = np.array([-w['n'][1], w['n'][0]])
            self.assertGreater(float(w['dir'] @ right), 0)

    def test_fit_counts_rows_columns_and_a_door_only_ground_floor(self):
        dets = [{'kind': 'window', 'score': .6, 'u0': u, 'u1': u + 1.2, 'z0': 100 + z, 'z1': 101.5 + z}
                for u in (1, 4, 7) for z in (4.0, 7.0)]
        dets.append({'kind': 'door', 'score': .5, 'u0': 3, 'u1': 4.1, 'z0': 100, 'z1': 102.2})
        g = facade.fit(dets, ground_up=100.0, eave_m=9.0, length_m=10.0)
        self.assertEqual([r['count'] for r in g['rows']], [3, 3])
        self.assertEqual((g['columns'], g['spacing_m'], g['storeys']), (3, 3.0, 3))   # two rows + door floor
        self.assertEqual(g['rows'][0]['sill_m'], 4.0)
        self.assertEqual(g['storey_height_m'], 3.0)
        self.assertEqual(g['doors'][0]['width_m'], 1.1)
        basement = dets + [{'kind': 'window', 'score': .6, 'u0': 1, 'u1': 2, 'z0': 99.6, 'z1': 100.4}]
        self.assertEqual(facade.fit(basement, 100.0, 9.0, 10.0)['storeys'], 3)

    def test_detections_drop_group_boxes_and_duplicates(self):
        spec = {'plane': [0, 0, 1, 0, 0, 1, 0.0, 10.0, 0.1, -2, 2, 0], 'length_m': 10.0}
        depth = np.ones((100, 100), np.float32)
        twin = [{'class': 'window', 'score': .7, 'box': [10, 10, 20, 30]},
                {'class': 'window', 'score': .6, 'box': [22, 10, 32, 30]},
                {'class': 'window', 'score': .8, 'box': [9, 9, 33, 31]},      # the pair as one box
                {'class': 'window', 'score': .5, 'box': [10.5, 10, 20.5, 30]},  # duplicate
                {'class': 'window', 'score': .9, 'box': [60, 60, 70, 80]}]
        depth[55:85, 55:75] = np.nan                                            # nothing rendered there
        got = vision.facade_dets(spec, twin, depth)
        self.assertEqual(sorted((d['u0'], d['z1']) for d in got), [(1.0, 9.0), (2.2, 9.0)])

    def test_box_helpers(self):
        sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
        self.assertAlmostEqual(vision.iou(sq, sq + [1, 0]), 1 / 3, places=6)
        self.assertEqual(vision.iou(sq, sq + [5, 0]), 0.0)
        cx, cy, L, W, yaw = vision.rect(np.array([[0, 0], [0, 4], [-2, 4], [-2, 0]], float))
        self.assertEqual((cx, cy, L, W, yaw), (-1.0, 2.0, 4.0, 2.0, 90.0))


class RamAndPrivacyTests(unittest.TestCase):
    def test_worker_runs_offline_once_the_weights_are_here(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(vision, 'MODELS', Path(tmp)):
            env = vision.worker_env()
            self.assertNotIn('YOLO_OFFLINE', env)   # first run: downloads allowed
            self.assertNotIn('HF_HUB_OFFLINE', env)
            for name in (vision.OBB_MODEL, vision.SAM_MODEL):
                (Path(tmp) / name).write_bytes(b'w')
            (Path(tmp) / vision.DINO_MODEL).mkdir()
            (Path(tmp) / vision.DINO_MODEL / 'config.json').write_text('{}')
            env = vision.worker_env()
            self.assertEqual((env['YOLO_OFFLINE'], env['HF_HUB_OFFLINE'], env['TRANSFORMERS_OFFLINE']), ('1', '1', '1'))

    def test_views_keep_only_the_newest_images(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(store, 'SHM', Path(tmp)):
            d = store.views_dir('k')
            for i in range(store.KEEP_VIEWS + 10):
                (d / f'v{i:03d}.png').write_bytes(b'x')
                os.utime(d / f'v{i:03d}.png', (i, i))
            store.views_dir('k')
            kept = store.views('k')
            self.assertEqual(len(kept), store.KEEP_VIEWS - 1)
            self.assertEqual(kept[0].name, f'v{store.KEEP_VIEWS + 9:03d}.png')

    def test_walls_of_a_shared_osm_outline_go_to_the_part_they_bound(self):
        ring = [(0, 0), (40, 0), (40, 10), (0, 10)]            # one mapped building, two mesh parts
        left = [(0, 0), (19, 0), (19, 10), (0, 10)]
        walls = facade.walls(ring)
        own = [w for w in walls if facade.belongs(left, w['a'], w['b'])]
        self.assertEqual(sorted(round(w['faces_deg']) for w in own), [90, 180, 270])   # not the far east wall
        o = {'footprint': left}
        stored = [{'a': [40, 0], 'b': [40, 10]}, {'a': [0, 10], 'b': [0, 0]}]
        self.assertEqual(ops._own_walls(o, stored), [stored[1]])

    def test_window_rows_that_need_a_look_are_flagged(self):
        rows = [{'sill_m': -0.6, 'head_m': 0.9}, {'sill_m': 0.5, 'head_m': 2.5}, {'sill_m': 3.0, 'head_m': 4.5}]
        self.assertEqual(ops._rows_check(rows), '1 row(s) start below ground; 1 row(s) overlap the next: check scan_facade')
        self.assertIsNone(ops._rows_check(rows[2:]))


class HelperTests(unittest.TestCase):
    def test_obb_of_a_rotated_rectangle(self):
        yy, xx = np.mgrid[-40:41, -40:41]
        a = math.radians(30)
        u, v = xx * math.cos(a) + yy * math.sin(a), -xx * math.sin(a) + yy * math.cos(a)
        m = (np.abs(u) <= 20) & (np.abs(v) <= 8)
        cx, cy, L, W, yaw = grid.obb(xx[m].astype(float), yy[m].astype(float))
        self.assertAlmostEqual(yaw, 30, delta=1)
        self.assertAlmostEqual(L, 41, delta=1.5)
        self.assertAlmostEqual(W, 17, delta=1.5)
        self.assertAlmostEqual(cx, 0, delta=0.5)

    def test_simplify_keeps_corners(self):
        sq = [(i, 0) for i in range(10)] + [(10, j) for j in range(10)] + [(10 - i, 10) for i in range(10)] + \
             [(0, 10 - j) for j in range(10)]
        out = grid.simplify(np.array(sq, float), 0.5)
        self.assertEqual(len(out), 4)

    def test_filters_keep_shapes(self):
        a = np.random.default_rng(0).random((37, 41)).astype(np.float32)
        for k in (2, 3, 8):
            self.assertEqual(grid.max_filter(a, k).shape, a.shape)
            self.assertEqual(grid.min_filter(a, k).shape, a.shape)
        self.assertTrue(np.allclose(grid.box_blur(np.ones((5, 7)), 3), 1))
        self.assertAlmostEqual(float(grid.gauss(np.ones((9, 9)), 2).mean()), 1.0, places=6)
        self.assertEqual(grid.resize_linear(np.ones((4, 4), np.float32), 8, 6).shape, (8, 6))

    def test_frame_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(store, 'SCAN_HOME', Path(tmp)):
            store.set_placement('/x.obj', {'at': [10, 20, 30], 'yaw': 33, 'datum_m': 5})
            fr = store.Frame({'obj': '/x.obj'})
            self.assertEqual(fr.name, 'map')
            x, y = fr.from_scan(3.0, -4.0)
            e, s = fr.to_scan(x, y)
            self.assertAlmostEqual(e, 3.0)
            self.assertAlmostEqual(s, -4.0)
            self.assertAlmostEqual(fr.z_out(5.0), 30.0)
            self.assertEqual(store.Frame({'obj': '/x.obj'}, 'scan').name, 'scan')
            self.assertEqual(store.Frame({'obj': '/other.obj'}).name, 'scan')
            with self.assertRaisesRegex(ValueError, 'frame must be'):
                store.Frame({'obj': '/x.obj'}, 'world')

    def test_no_survey_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(store, 'SCAN_HOME', Path(tmp)):
            with self.assertRaisesRegex(ValueError, '^no scan survey yet: run scan_survey OBJ first$'):
                ops.scan_objects()


def spawns(path: Path) -> list[list[str]]:
    """Spawn lines of a plan file, each checked to be followed by `drop last`."""
    lines = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith('#')]
    out = []
    for i, ln in enumerate(lines):
        if ln.startswith('spawn '):
            assert lines[i + 1] == 'drop last', lines[i:i + 2]
            out.append(ln.split())
    return out


def mesh_box(words: list[str]) -> tuple:
    """World AABB (metres) of a plan spawn line: spawn row x y z pitch yaw roll sx sy sz group=..."""
    a = load_catalog().assets[words[1]]
    pivot = tuple(float(v) / 100 for v in words[2:5])
    return world_aabb(a.origin_cm, a.extent_cm, pivot, math.radians(float(words[6])), tuple(map(float, words[8:11])))


def snapshot_file(path: Path, objects: list[dict]) -> Path:
    """A scene_snapshot .json.gz that load_snapshot accepts."""
    import gzip
    stamp = {'name': 'fake', 'map': 'Map_A', 'base': 'BaseBoard', 'session': 's', 'objects': len(objects),
             'scope': 'whole scene', 'scope_objects': len(objects), 'fingerprint': snapshots.fingerprint(objects),
             'gates': 0, 'gate_fingerprint': snapshots.gate_fingerprint([]), 'start_gate': None, 'file': str(path)}
    with gzip.open(path, 'wt') as f:
        json.dump({'version': 1, 'kind': 'scene_snapshot', 'stamp': stamp, 'objects': objects, 'gates': []}, f)
    return path


@unittest.skipUnless(NATIVE, 'needs g++ and libturbojpeg headers to build native/ucmesh and native/scanraster')
class MappingTests(Sandbox):
    """scan_match / scan_plan / scan_compare / scan_sheet on the synthetic scene (placed at 100,200,50 yaw 90)."""

    @classmethod
    def setUpClass(cls):
        reference.tool()
        native.library()

    def setUp(self):
        super().setUp()
        self.obj = make_scene(self.root / 'src')
        ops.scan_survey(str(self.obj), use_osm=False)
        self.objs = store.objects(store.resolve(''))

    def of(self, cls):
        return [o for o in self.objs if o['cls'] == cls]

    def test_match_cars_and_a_gabled_house(self):
        car = self.of('car')[0]
        m = mapping.scan_match(car['id'])
        self.assertEqual(m['frame'], 'scan')
        rows = [c['row'] for c in m['candidates']]
        self.assertTrue(set(rows) <= {'NewRow_103', 'NewRow_257', 'NewRow_775', 'NewRow_93'}, rows)
        best = m['candidates'][0]
        self.assertLess(best['fit_err_pct'], 15)
        a = load_catalog().assets[best['row']]
        long_yaw = (best['yaw_deg'] + (90 if a.size_m[1] > a.size_m[0] else 0)) % 180
        self.assertAlmostEqual(min(long_yaw, 180 - long_yaw), car['yaw_deg'] % 180, delta=6)
        self.assertTrue(all(0.85 <= s <= 1.15 for s in best['scale']))
        gabled = next(o for o in self.of('building') if o['roof'] == 'gabled')
        g = mapping.scan_match(gabled['id'], top=5)
        self.assertIn(g['candidates'][0]['roof'], ('gabled', 'hipped'), g['candidates'])
        # a gabled row turned by its ridge (House 02's ridge runs across its long side, so it is 'ridge turned')
        aligned = next(c for c in g['candidates'] if c['roof'] == 'gabled' and 'ridge turned' not in (c['notes'] or ''))
        ridge = mapping.mesh_roof(aligned['row'])['ridge_deg']
        self.assertAlmostEqual(((aligned['yaw_deg'] + ridge) - gabled['ridge_yaw_deg'] + 90) % 180 - 90, 0, delta=1)
        flat = next(o for o in self.of('building') if o['roof'] == 'flat')
        self.assertNotEqual(mapping.scan_match(flat['id'])['candidates'][0]['roof'], 'gabled')
        many = mapping.scan_match(classes=['car', 'building'], min_conf=0)
        self.assertEqual({r['cls']: r['matched'] for r in many['classes']}, {'car': 2, 'building': 2})
        with self.assertRaisesRegex(ValueError, 'not both'):
            mapping.scan_match(car['id'], classes=['car'])
        with self.assertRaisesRegex(ValueError, 'unknown catalogue rows'):
            mapping.scan_match(car['id'], rows=['NewRow_nope'])

    def test_plan_needs_a_placement_and_lands_mesh_boxes_on_the_objects(self):
        out = self.root / 'plans' / 'p.txt'
        with self.assertRaisesRegex(ValueError, 'set the placement first'):
            mapping.scan_plan(out=str(out))
        ops.scan_place(at=[100, 200, 50], yaw=90, datum_m=0.0)
        res = mapping.scan_plan(out=str(out), min_conf=0)
        self.assertFalse(res['executed'])
        self.assertTrue(out.is_file() and out.with_suffix('.json').is_file())
        by = {r['cls']: r for r in res['classes']}
        self.assertEqual(by['car']['planned'], 2)
        self.assertEqual(by['building']['building_hints'], 2)
        self.assertEqual(by['building']['planned'], 0)
        fr = store.Frame(store.resolve(''), 'map')
        lines = spawns(out)
        self.assertEqual(len(lines), res['spawn_lines'])
        planned = {int(w[-1].split('.')[-1]): w for w in lines}
        offsets = {e['id']: e['visible_offset_m'] for e in json.loads(out.with_suffix('.json').read_text())['objects']}
        for o in self.of('car') + self.of('tree'):
            w = planned[o['id']]
            self.assertEqual(w[-1], f"group=scan.{o['cls']}.{o['id']}")
            lo = mesh_box(w)
            x, y = fr.from_scan(o['x'], o['y'])
            # the visible centre (box centre + profile offset, turned and scaled) sits on the object
            t, (ox, oy) = math.radians(float(w[6])), offsets[o['id']]
            ox, oy = ox * float(w[8]), oy * float(w[9])
            self.assertAlmostEqual((lo[0] + lo[3]) / 2 + ox * math.cos(t) - oy * math.sin(t), x, delta=0.01)
            self.assertAlmostEqual((lo[1] + lo[4]) / 2 + ox * math.sin(t) + oy * math.cos(t), y, delta=0.01)
            self.assertAlmostEqual(lo[2], fr.z_out(o['ground_z']) + 0.5, delta=0.01)  # lift_m above the scan ground
        text = out.read_text()
        self.assertIn('# BUILDING', text)
        self.assertIn('#? spawn ', text)
        self.assertIn('footprint (map m)', text)
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            mapping.scan_plan(out=str(out))
        standin = mapping.scan_plan(out=str(self.root / 'plans' / 's.txt'), buildings='standin', classes=['building'],
                                    min_conf=0)
        self.assertEqual(standin['spawn_lines'], 2)
        with self.assertRaisesRegex(ValueError, 'buildings must be'):
            mapping.scan_plan(out=str(self.root / 'x.txt'), buildings='all')

    def test_compare_finds_moved_missing_and_unexplained_objects(self):
        ops.scan_place(at=[100, 200, 50], yaw=90, datum_m=0.0)
        plan = self.root / 'plan.txt'
        mapping.scan_plan(out=str(plan), classes=['car', 'tree'], min_conf=0)
        placed = []
        for i, w in enumerate(spawns(plan)):
            xf = [float(v) for v in w[2:5]] + [0.0, float(w[6]), 0.0] + [float(v) for v in w[8:11]]
            placed.append({'actor': f'{0x100 + i:x}', 'row': w[1], 'group': w[-1][6:], 'xf': xf})
        cars = sorted(self.of('car'), key=lambda o: o['id'])
        moved = next(p for p in placed if p['group'] == f"scan.car.{cars[0]['id']}")
        moved['xf'][0] += 300                                         # 3 m along x
        placed = [p for p in placed if p['group'] != f"scan.car.{cars[1]['id']}"]   # the second car was never built
        placed.append({'actor': 'f00', 'row': 'NewRow_1', 'group': 'stray', 'xf': [10000, 20000, 5000, 0, 0, 0, 1, 1, 1]})
        snap = snapshot_file(self.root / 'snap.json.gz', placed)
        res = mapping.scan_compare(str(snap), classes=['car', 'tree'], min_conf=0)
        status = {r['id']: r['status'] for r in res['problems']}
        self.assertEqual(status.get(cars[0]['id']), 'offset')
        self.assertEqual(status.get(cars[1]['id']), 'missing')
        self.assertEqual(res['counts']['missing'], 1)
        self.assertEqual([p['group'] for p in res['unmatched_placed']], ['stray'])
        self.assertTrue(Path(res['report_json']).is_file())
        # live: one read-only group export (no group tags: nearest pairing)
        for p in placed:
            p['group'] = 'misc'
        with patch.object(groups, 'snapshot', return_value={'session': 's', 'objects': placed}) as snapshot:
            live = mapping.scan_compare(classes=['car'], min_conf=0)
        snapshot.assert_called_once_with()
        self.assertEqual(live['source'], 'live groups export')
        self.assertEqual({k: v for k, v in live['counts'].items() if v}, {'offset': 1, 'missing': 1})
        ops.scan_place(clear=True)
        with self.assertRaisesRegex(ValueError, 'needs the scan placement'):
            mapping.scan_compare(str(snap))

    def test_sheet_tiles_crops_into_one_ram_png(self):
        out = mapping.scan_sheet(points=[[0, 0]], classes=['car'], size_m=0, px=200, min_conf=0)
        self.assertEqual(out['tiles'], 3)
        self.assertEqual([c['object'] for c in out['crops']][0], None)
        path = Path(out['image'])
        self.assertTrue(path.is_file() and str(path).startswith(str(store.SHM)))
        self.assertEqual(sorted(p.name for p in path.parent.iterdir()), [path.name])  # tile files removed
        with Image.open(path) as im:
            self.assertGreater(im.width, 400)
        with self.assertRaisesRegex(ValueError, 'give points'):
            mapping.scan_sheet()
        with self.assertRaisesRegex(ValueError, 'no objects 9999'):
            mapping.scan_sheet(objects=['9999'])


class CatalogueMatchTests(unittest.TestCase):
    def obj(self, **kw):
        base = {'id': 1, 'cls': 'car', 'conf': .9, 'x': 0.0, 'y': 0.0, 'ground_z': 0.0, 'top_z': 1.5, 'height_m': 1.45,
                'length_m': 4.3, 'width_m': 2.0, 'yaw_deg': 30.0, 'rgb': [90, 90, 90]}
        return {**base, **kw}

    def test_candidate_sets_follow_the_class(self):
        cars = {a.id for a in mapping.candidates('car')}
        self.assertIn('NewRow_257', cars)              # Gt86: tagged prop in the catalogue, still a car
        self.assertNotIn('NewRow_29', cars)            # bus stop: tagged vehicle, not a car
        self.assertNotIn('NewRow_101', cars)           # interactive spawner
        self.assertNotIn('NewRow_622', cars)           # hidden variant unless asked
        self.assertIn('NewRow_622', {a.id for a in mapping.candidates('car', include_hidden=True)})
        self.assertEqual({a.id for a in mapping.candidates('pool')}, {'NewRow_183'})
        self.assertTrue(all('tree' in a.tags for a in mapping.candidates('tree')))
        self.assertIn('NewRow_788', {a.id for a in mapping.candidates('building')})

    def test_slender_trees_are_roof_greenery(self):
        # measured on the example scan: 'trees' taller than 3 x a narrow crown were roof plants or building edges
        self.assertTrue(mapping.slender(self.obj(cls='tree', height_m=16.6, crown_m=2.7, width_m=3.0)))
        self.assertFalse(mapping.slender(self.obj(cls='tree', height_m=23.3, crown_m=5.9, width_m=5.2)))  # tall, wide
        self.assertFalse(mapping.slender(self.obj(cls='tree', height_m=6.0, crown_m=2.2, width_m=2.0)))   # young tree

    def test_roofs_of_catalogue_meshes(self):
        self.assertEqual(mapping.mesh_roof('NewRow_788')['roof'], 'gabled')
        self.assertEqual(mapping.mesh_roof('NewRow_9')['roof'], 'flat')
        self.assertIsNone(mapping.mesh_roof('NewRow_10'))   # Blueprint: no hull

    def test_fit_scales_and_turns(self):
        cat = load_catalog().assets
        f = mapping.fit_one(self.obj(), cat['NewRow_257'])  # Gt86 4.25 x 1.99 x 1.44, long on x
        self.assertLess(f['fit_err_pct'], 6)                 # measured by its visible size (mirrors left out)
        self.assertAlmostEqual(f['yaw_scan'], 30.0)
        van = mapping.fit_one(self.obj(), cat['NewRow_93'])  # long on y: turned by -90
        self.assertAlmostEqual(van['yaw_scan'], 300.0)
        self.assertEqual(van['scale'], [0.85] * 3)           # vehicles stay within 0.85..1.15
        tree = mapping.fit_one(self.obj(cls='tree', length_m=8, width_m=7.5, height_m=12, yaw_deg=0), cat['NewRow_168'])
        self.assertAlmostEqual(tree['visible_m'][2], 12, delta=0.05)   # visible top, not the box top
        house = self.obj(cls='building', length_m=14.5, width_m=9.1, height_m=8, ridge_m=8.3, roof='gabled',
                         ridge_yaw_deg=20.0, yaw_deg=110.0)
        h = mapping.fit_one(house, cat['NewRow_788'])
        self.assertEqual((h['roof'], h['yaw_from']), ('gabled', 'ridge'))
        self.assertLess(h['shape_err_pct'], 5)
        flat = mapping.fit_one(house, cat['NewRow_9'])
        self.assertLess(flat['score'], h['score'])
        self.assertIn('roof flat', flat['notes'])


class ApiTests(unittest.TestCase):
    def test_api_forwards_to_the_scan_module(self):
        calls = []

        def echo(name):
            return lambda *a, **k: calls.append((name, a, k)) or {'ok': name}

        with patch.multiple(ops, scan_height=echo('h'), scan_view=echo('v'), scan_survey=echo('s')):
            self.assertEqual(api.call('scan_height', [[1, 2]], frame='scan'), {'ok': 'h'})
            api.call('scan_view', center=[1, 2], layer='ndsm')
            api.call('scan_survey', 'x.obj', osm=False, vision=False)
        self.assertEqual(calls[0], ('h', ([[1, 2]],), {'agl': False, 'frame': 'scan', 'survey_key': ''}))
        self.assertEqual(calls[1][2]['layer'], 'ndsm')
        self.assertEqual((calls[2][2]['use_osm'], calls[2][2]['use_vision']), (False, False))

    def test_api_forwards_to_the_mapping_module(self):
        calls = []

        def echo(name):
            return lambda *a, **k: calls.append((name, a, k)) or {'ok': name}

        with patch.multiple(mapping, scan_match=echo('m'), scan_plan=echo('p'), scan_compare=echo('c'),
                            scan_sheet=echo('s')):
            self.assertEqual(api.call('scan_match', 12), {'ok': 'm'})
            api.call('scan_plan', box=[0, 0, 10, 10], out='x.txt')
            api.call('scan_compare', 'snap', min_conf=.8)
            api.call('scan_sheet', [[1, 2]], layer='ndsm')
        self.assertEqual(calls[0][2]['object_id'], 12)
        self.assertEqual((calls[1][2]['box'], calls[1][2]['out'], calls[1][2]['buildings']), ([0, 0, 10, 10], 'x.txt', 'hint'))
        self.assertEqual((calls[2][2]['snapshot'], calls[2][2]['min_conf'], calls[2][2]['survey_key']), ('snap', .8, ''))
        self.assertEqual((calls[3][2]['points'], calls[3][2]['layer']), ([[1, 2]], 'ndsm'))

    def test_from_reference_accepts_only_the_recorded_pose_of_the_same_obj(self):
        last = {'obj': '/data/a.obj', 'at': [1, 2, 3], 'yaw': 30.0, 'datum_m': 4.0, 'pose_m': [1, 2, 3]}
        shown = {'loaded': True, 'pose_m': [1.0, 2.0, 3.0], 'yaw': 30.0}
        cases = [(None, shown, '/data/a.obj', 'no reference_load recorded'),
                 (last, shown, '/data/b.obj', 'the last reference_load was /data/a.obj'),
                 (last, {'loaded': False}, '/data/a.obj', 'no reference is loaded in the editor now'),
                 (last, {**shown, 'pose_m': [1.0, 2.5, 3.0]}, '/data/a.obj', 'not placed by the recorded'),
                 (last, {**shown, 'yaw': 31.0}, '/data/a.obj', 'not placed by the recorded')]
        for record, status, obj, error in cases:
            with patch.object(reference, 'last_load', return_value=record), \
                    patch.object(reference, 'status', return_value=status):
                with self.assertRaisesRegex(ValueError, error):
                    ops._from_reference(obj)
        with patch.object(reference, 'last_load', return_value=last), patch.object(reference, 'status', return_value=shown):
            self.assertEqual(ops._from_reference('/data/a.obj'),
                             {'at': [1.0, 2.0, 3.0], 'yaw': 30.0, 'datum_m': 4.0, 'source': 'reference_load'})

    def test_help_says_what_the_mixed_and_writing_scan_commands_need(self):
        from uncrashed_mcp import cli
        place = cli._command_help(api.OPERATIONS['scan_place'])
        self.assertIn('needs: no game, except --from_reference (reads the reference loaded in the open editor); '
                      'changes the scene or files at once (no preview)', place)
        self.assertIn('; changes the scene or files at once', cli._command_help(api.OPERATIONS['scan_plan']))
        self.assertIn('needs: no game with a scene_snapshot .json.gz file', cli._command_help(api.OPERATIONS['scan_compare']))

if __name__ == '__main__':
    unittest.main()
