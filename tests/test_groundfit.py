"""Ground planner: smooth fitted surfaces paved with tilted tiles, on synthetic scan grounds (no survey, no game)."""
import math
import types
import unittest

import numpy as np

from uncrashed_mcp import geom
from uncrashed_mcp.catalog import load_catalog
from uncrashed_mcp.scan import groundfit, mapping, ops, survey

RES = 0.25
C = survey.CODE


class FakeRas:
    """The parts of ops.Rasters the ground planner reads: a class raster and a ground model (DTM) on one grid."""

    def __init__(self, cls: np.ndarray, dtm: np.ndarray):
        self.g = survey.Grid(0, 0, RES, cls.shape[1], cls.shape[0])
        self.cls, self.dtm = cls, dtm.astype(np.float32)

    def cell(self, e, s):
        return ops.Rasters.cell(self, e, s)

    def ground(self, e, s):
        return ops.Rasters.ground(self, e, s)


FRAME = types.SimpleNamespace(to_scan=lambda x, y: (np.asarray(x, float), np.asarray(y, float)),
                              z_out=lambda z: z, name='map')


def street(size=(40, 30), z=lambda x, y: 1.0 + 0 * x, noise=0.06, kerb=0.0, seed=1):
    """A road along x (y 10..20) between two paved sidewalks; ground noise in 1 m blocks (as the 1 m DTM filter
    leaves it) plus fine noise; the sidewalks kerb metres above the road."""
    W, H = int(size[0] / RES), int(size[1] / RES)
    X, Y = survey.Grid(0, 0, RES, W, H).to_world(*np.meshgrid(np.arange(W), np.arange(H)))
    cls = np.full((H, W), C['paved'], np.uint8)
    road = (Y >= 10) & (Y < 20)
    cls[road] = C['road']
    rng = np.random.default_rng(seed)
    block = rng.normal(0, noise, (H // 4 + 1, W // 4 + 1))
    dtm = z(X, Y) + np.kron(block, np.ones((4, 4)))[:H, :W] + rng.normal(0, noise / 3, (H, W))
    dtm[~road] += kerb
    return cls, dtm, X, Y


def plan(cls, dtm, box, **kw):
    lines, summ, surf = mapping.ground_cells({}, FRAME, FakeRas(cls, dtm), box, **kw)
    return lines, summ, surf


def faces_of(lines):
    cat = load_catalog()
    out = []
    for n, ln in enumerate(lines):
        w = ln.split()
        out.append({'P': groundfit.top_face([float(v) for v in w[2:11]], cat.assets[w[1]]), 'row': w[1], 'id': n})
    return out


class GroundPlannerTests(unittest.TestCase):
    def test_noisy_flat_road_is_one_plane_without_steps(self):
        cls, dtm, *_ = street()
        lines, summ, surf = plan(cls, dtm, (0, 0, 40, 30), yaw=0.0)
        tops = np.concatenate([f['P'][:, 2] for f in faces_of(lines)])
        self.assertLess(np.ptp(tops), 0.03)                       # one plane, not the noise
        self.assertLess(np.abs(tops - 1.0).max(), 0.03)           # tops on the ground (pivot TOP of NewRow_303 too)
        self.assertEqual({f['row'] for f in faces_of(lines)}, {'NewRow_182', 'NewRow_303'})
        self.assertEqual(summ['predicted_steps']['same_row']['over_1cm'], 0)
        self.assertEqual(summ['predicted_steps']['between_rows']['over_1cm'], 0)
        self.assertLessEqual(len(lines), 20)                      # big tiles: 300 cells, not 300 tiles
        self.assertLess(abs(float(np.nanmean(surf.z_at([5, 20, 35], [15, 15, 15]))) - 1.0), 0.03)

    def test_sloped_street_follows_the_slope(self):
        cls, dtm, *_ = street(z=lambda x, y: 1.0 + 0.04 * x)
        lines, summ, surf = plan(cls, dtm, (0, 0, 40, 30), yaw=0.0)
        x = np.linspace(2, 38, 10)
        got = surf.z_at(x, np.full(10, 15.0))
        self.assertLess(np.abs(got - (1.0 + 0.04 * x)).max(), 0.03)
        self.assertEqual(summ['predicted_steps']['same_row']['over_1cm'], 0)
        for f in faces_of(lines):                                 # every tile tilted 4 % along x
            P = f['P']
            self.assertAlmostEqual((P[1, 2] - P[0, 2]) / (P[1, 0] - P[0, 0]), 0.04, delta=0.01)

    def test_a_12_cm_kerb_survives_as_one_step(self):
        cls, dtm, *_ = street(kerb=0.12)
        lines, summ, surf = plan(cls, dtm, (0, 0, 40, 30), yaw=0.0)
        self.assertEqual(len(summ['kerbs_m']), 1)
        self.assertAlmostEqual(summ['kerbs_m'][0], 0.12, delta=0.015)
        road = surf.z_at(np.full(5, 20.0), np.linspace(11, 19, 5))
        side = surf.z_at(np.full(4, 20.0), [2, 7, 23, 28])
        self.assertLess(np.ptp(road), 0.02)
        self.assertLess(np.ptp(side), 0.02)
        self.assertAlmostEqual(float(side.mean() - road.mean()), 0.12, delta=0.02)
        st = summ['predicted_steps']
        self.assertEqual(st['same_row']['over_1cm'], 0)           # within a material: no steps
        steps = [w['dz_m'] for w in st['worst'] if w['dz_m'] > 0.01]
        self.assertTrue(steps and all(abs(d - 0.12) < 0.02 for d in steps), st['worst'])

    def test_car_dents_and_crown_blobs_are_ignored(self):
        cls, dtm, X, Y = street(noise=0.03)
        car = (abs(X - 12) < 2.2) & (abs(Y - 13) < 1)
        cls[car], dtm[car] = C['low_object'], dtm[car] + 0.5    # a melted car on the road
        crown = (X - 28) ** 2 + (Y - 16) ** 2 < 9
        cls[crown], dtm[crown] = C['vegetation'], dtm[crown] + 0.8
        dent = (abs(X - 20) < 1) & (abs(Y - 17) < 1)
        dtm[dent] -= 0.3                                          # an unexplained hole, still classed road
        lines, summ, surf = plan(cls, dtm, (0, 0, 40, 30), yaw=0.0)
        got = surf.z_at([12, 28, 20, 5], [13, 16, 17, 15])
        self.assertLess(np.abs(got - 1.0).max(), 0.03, got)
        self.assertEqual(summ['by_class'].keys() - {'road', 'paved'}, set())   # under the car / crown: road

    def test_tile_corners_meet_after_the_rotation_maths(self):
        """A bent, sloped ground on a turned lattice: the spawn lines, read back through the engine's rotator
        (geom.quat_rotator), give tile tops whose edges meet within a centimetre."""
        cls, dtm, *_ = street(size=(48, 40), z=lambda x, y: 1 + 0.03 * x - 0.02 * y + 0.001 * (x - 24) ** 2,
                              noise=0.04)
        lines, summ, surf = plan(cls, dtm, (4, 4, 44, 36), yaw=30.0, flat_tol_m=0.05)
        self.assertEqual(summ['yaw_deg'], 30.0)
        st = groundfit.edge_steps(faces_of(lines))
        self.assertGreater(st['same_row']['pairs'], 20)
        self.assertLess(max(st['same_row']['max_m'], st['between_rows']['max_m']), 0.01, st['worst'][:3])
        x = np.linspace(12, 36, 9)                                # the bend is followed (the paving edge sags a little)
        y = np.full(9, 15.0)
        self.assertLess(np.abs(surf.z_at(x, y) - (1 + 0.03 * x - 0.02 * y + 0.001 * (x - 24) ** 2)).max(), 0.03)

    def test_an_embankment_becomes_a_ramp_not_a_sag(self):
        """Two terraces 0.6 m apart (a retaining wall across the paving): each stays flat away from the wall, and
        the tiles still meet (a steep ramp tile at the wall, no step)."""
        cls, dtm, X, Y = street(z=lambda x, y: 1.0 + 0.6 * (x >= 20), noise=0.03)
        lines, summ, surf = plan(cls, dtm, (0, 0, 40, 30), yaw=0.0)
        got = surf.z_at([4, 12, 28, 36, 4, 36], [15, 15, 15, 15, 5, 25])
        np.testing.assert_allclose(got, [1.0, 1.0, 1.6, 1.6, 1.0, 1.6], atol=0.04)
        st = summ['predicted_steps']
        self.assertEqual(st['same_row']['over_1cm'] + st['between_rows']['over_1cm'], 0, st['worst'][:3])

    def test_rotator_round_trip_matches_the_engine_axes(self):
        for X in ([0.99, 0.0, 0.14], [0.6, 0.79, -0.1], [-0.7, 0.7, 0.05]):
            X = np.array(X) / np.linalg.norm(X)
            Z = np.cross(X, [0.1, 1.0, 0.0])
            Z = Z if Z[2] > 0 else -Z
            Z /= np.linalg.norm(Z)
            Y = np.cross(Z, X)
            q = geom.quat_rotator(*mapping._rotator(X, Y, Z))
            for axis, want in ((1, 0, 0), X), ((0, 1, 0), Y), ((0, 0, 1), Z):
                np.testing.assert_allclose(geom.quat_rotate(q, axis), want, atol=1e-9)

    def test_street_yaw_reads_the_road_direction(self):
        H = W = 160
        yy, xx = np.mgrid[0:H, 0:W].astype(float)
        t = math.radians(30)
        road = np.abs(-(xx - 80) * math.sin(t) + (yy - 80) * math.cos(t)) < 12
        yaw, strength = mapping.street_yaw(road)
        self.assertAlmostEqual(yaw, 30, delta=3)
        self.assertGreater(strength, 0.5)


if __name__ == '__main__':
    unittest.main()
