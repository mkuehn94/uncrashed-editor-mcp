"""Route clearance: spline, sampling, orientation math, native parsing and classification (no game)."""
import contextlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from uncrashed_mcp import clearance as cl


def dist(a, b):
    return math.dist(a, b)


SQUARE = [{'center': (0, 0, 2), 'normal': (1, 0, 0)}, {'center': (30, 0, 3), 'normal': (0, -1, 0)},
          {'center': (30, 30, 2), 'normal': (1, 0, 0)}, {'center': (0, 30, 2), 'normal': (0, 1, 0)}]


class ProfileTests(unittest.TestCase):
    def test_presets_are_explicit(self):
        for name in cl.PROFILES:
            p = cl.resolve_profile(name)
            self.assertEqual(p['half_extents_m'], [round(p['length']/2, 4), round(p['width']/2, 4), round(p['height']/2, 4)])
            self.assertIn(p['kind'], ('air', 'ground'))
        self.assertEqual(cl.resolve_profile('truck')['ground_clearance'], .3)
        self.assertEqual(cl.resolve_profile('whoop')['name'], 'tiny_whoop')

    def test_custom_and_overrides(self):
        p = cl.resolve_profile('custom', dims_m=[2, 1, .5], margin_m=.2, kind='ground')
        self.assertEqual((p['name'], p['kind'], p['half_extents_m'], p['margin']), ('custom', 'ground', [.5, 1, .25], .2))
        self.assertIn('ground_clearance', p)
        for bad in (dict(profile='nope'), dict(dims_m=[1, 2]), dict(dims_m=[1, 2, -1]), dict(margin_m=-1), dict(kind='sea')):
            with self.assertRaises(ValueError):
                cl.resolve_profile(**({'profile': '5inch'} | bad))


class GeometryTests(unittest.TestCase):
    def test_spline_passes_through_every_point(self):
        pts = [(0, 0, 0), (10, 2, 1), (20, -3, 4), (25, 10, 2)]
        poly = cl.catmull_rom(pts)
        for p in pts:
            self.assertLess(min(dist(p, q) for q in poly), 1e-9)
        self.assertEqual(poly[0], pts[0])
        self.assertEqual(poly[-1], pts[-1])

    def test_closed_loop_returns_to_start(self):
        pts = [s['center'] for s in SQUARE]
        poly = cl.catmull_rom(pts, closed=True)
        self.assertEqual(poly[-1], poly[0])
        for p in pts:
            self.assertLess(min(dist(p, q) for q in poly), 1e-9)
        # smooth: no reversal between consecutive steps
        for a, b, c in zip(poly, poly[1:], poly[2:]):
            u, v = cl._sub(b, a), cl._sub(c, b)
            if cl._len(u) > 1e-6 and cl._len(v) > 1e-6:
                self.assertGreater(cl._dot(cl._unit(u), cl._unit(v)), .5)

    def test_duplicate_points_do_not_break_the_spline(self):
        poly = cl.catmull_rom([(0, 0, 0), (0, 0, 0), (5, 0, 0), (5, 0, 0), (9, 1, 0)])
        self.assertTrue(all(all(math.isfinite(v) for v in p) for p in poly))
        with self.assertRaises(ValueError):
            cl.catmull_rom([(1, 1, 1), (1, 1, 1)])

    def test_resample_spacing(self):
        out = cl.resample([(0, 0, 0), (10.5, 0, 0)], 1.0)
        self.assertEqual(len(out), 12)
        for (a, da), (b, db) in zip(out, out[1:-1]):
            self.assertAlmostEqual(dist(a, b), 1.0)
            self.assertAlmostEqual(db - da, 1.0)
        self.assertEqual(out[-1], ((10.5, 0, 0), 10.5))
        # curved: arc-length spacing measured along the dense polyline stays close to the request
        samples = cl.build_route([s['center'] for s in SQUARE], 1.0, closed=True)
        gaps = [dist(a['p'], b['p']) for a, b in zip(samples, samples[1:-1])]
        self.assertTrue(all(.95 < g <= 1.0001 for g in gaps), (min(gaps), max(gaps)))
        self.assertLess(dist(samples[0]['p'], samples[-1]['p']), 1e-6)
        self.assertEqual([s['i'] for s in samples], list(range(len(samples))))

    def test_heading_and_box_orientation(self):
        for d in [(1, 0, 0), (0, 1, 0), (-3, 2, 1), (1, -1, -2), (0.2, 0.1, 5)]:
            pitch, yaw = cl.heading((0, 0, 0), d)
            x, y, z = cl.rot_axes(pitch, yaw)
            u = cl._unit(d)
            for a, b in zip(x, u):
                self.assertAlmostEqual(a, b, places=9)
            self.assertAlmostEqual(y[2], 0, places=9)          # sideways axis stays horizontal without roll
            self.assertAlmostEqual(cl._dot(x, y), 0, places=9)
            self.assertAlmostEqual(cl._dot(x, z), 0, places=9)
            self.assertGreater(z[2], 0)
        x, y, z = cl.rot_axes(0, 0)
        self.assertEqual(tuple(round(v, 9) for v in y), (0, 1, 0))  # Unreal: +Y is right of +X

    def test_support_and_probe_directions(self):
        axes = cl.rot_axes(0, 90)
        half = (2, 1, .5)
        self.assertAlmostEqual(cl.support(half, axes, (0, 1, 0)), 2)   # forward is +Y now
        self.assertAlmostEqual(cl.support(half, axes, (1, 0, 0)), 1)
        self.assertAlmostEqual(cl.support(half, axes, (0, 0, -1)), .5)
        diag = cl._unit((1, 1, 0))
        self.assertAlmostEqual(cl.support(half, axes, diag), (2 + 1) / math.sqrt(2))
        dirs = cl.probe_dirs(30, 16)
        self.assertEqual(len(dirs), 18)
        right = cl.rot_axes(0, 30)[1]
        for a, b in zip(dirs[4], right):
            self.assertAlmostEqual(a, b)
        self.assertEqual(dirs[16], (0, 0, 1))
        self.assertEqual(dirs[17], (0, 0, -1))

    def test_gate_route_crosses_square_to_normals(self):
        gates = [dict(g) for g in SQUARE]
        controls = cl.gate_controls(gates, closed=True, lead_m=2.0)
        self.assertEqual(controls[0], gates[0]['center'])           # loop starts in the start gate
        samples = cl.build_route(controls, .5, closed=True)
        for g in gates:
            s, miss = cl.nearest_sample(samples, g['center'])
            self.assertLessEqual(miss, .26)
            fwd = cl.rot_axes(s['pitch'], s['yaw'])[0]
            self.assertGreater(cl._dot(fwd, g['normal']), .97)        # normal flipped to the direction of travel
        self.assertEqual(gates[1]['normal'], [0, 1, 0])

    def test_open_route_has_no_lead_in_at_the_start(self):
        controls = cl.gate_controls([dict(g) for g in SQUARE[:2]], closed=False, lead_m=2)
        self.assertEqual(controls[0], SQUARE[0]['center'])
        self.assertEqual(controls[-1], SQUARE[1]['center'])
        self.assertEqual(len(controls), 4)
        with self.assertRaises(ValueError):
            cl.gate_controls([SQUARE[0]], closed=False)


class RequestTests(unittest.TestCase):
    def test_chunks_sweep_every_segment_and_probe_every_sample_once(self):
        samples = cl.build_route([(0, 0, 1), (60, 0, 1)], 1.0)
        prof = cl.resolve_profile('5inch')
        swept, probed = [], []
        for first, last in cl.chunks(len(samples), 25):
            text = cl.request_text(samples, prof, 'sweep', 2, 10, 16, first=first, last=last)
            lines = text.splitlines()
            self.assertEqual(lines[:4], ['B 15.00 15.00 5.00', 'R 1000.0 16', 'C 2', 'M sweep'])
            for line in lines[4:]:
                p = line.split()
                if p[7] == '1':
                    swept.append(int(p[1]))
                if p[8] == '1':
                    probed.append(int(p[1]))
        self.assertEqual(sorted(swept), list(range(len(samples) - 1)))
        self.assertEqual(sorted(probed), list(range(len(samples))))
        self.assertEqual(cl.chunks(1), [(0, 0)])
        self.assertEqual(cl.chunks(0), [])

    def test_drape_header(self):
        samples = cl.build_route([(0, 0, 0), (3, 0, 0)], 1.0)
        text = cl.request_text(samples, cl.resolve_profile('car'), 'drape', 6, 0, 0, (2, 50))
        self.assertIn('M drape 200.0 5000.0 1', text)
        self.assertIn('C 6', text)

    def test_gate_centre_probe_request(self):
        gates = [{'center': (10, 0, 2), 'normal': (1, 0, 0)}, {'center': (0, 5, 8), 'normal': (0, 0, -1)}]
        gs = cl.gate_probe_samples(gates, 50)
        self.assertEqual([g['i'] for g in gs], [50, 51])
        self.assertEqual(gs[0]['p'], [10, 0, 2])                    # exactly on the gate centre
        self.assertEqual((gs[0]['pitch'], gs[0]['yaw']), (0, 0))
        self.assertEqual(gs[1]['pitch'], -90)
        lines = cl.request_text(gs, cl.resolve_profile('5inch'), 'sweep', 2, 10, 16, probe_only=True).splitlines()[4:]
        self.assertEqual([l.split()[1] for l in lines], ['50', '51'])
        self.assertTrue(all(l.split()[7:] == ['0', '1'] for l in lines))  # probe only, never swept

    def test_channel_names(self):
        self.assertEqual(cl._channel('Pawn'), 2)
        self.assertEqual(cl._channel('physicsbody'), 5)
        self.assertEqual(cl._channel('16'), 16)
        self.assertEqual(cl._channel('game1'), 14)
        self.assertEqual(cl._channel('ECC_GameTraceChannel18'), 31)
        self.assertEqual(cl._channel('engine1'), 8)
        self.assertEqual(cl.channel_name(16), 'game3')
        for bad in ('Water', '32', 'game19', -1, True):
            with self.assertRaises(ValueError):
                cl._channel(bad)


GATEBOX = '''G 1a 0.00 0.00 0.00 0.000 0.000 0.000 1.0000 1.0000 1.0000 -150.00 -20.00 0.00 150.00 20.00 300.00
GC 1a 2b StaticMeshComponent 2 0 3 0.00 0.00 0.00 0.000 0.000 0.000 -1.00 -1.00 -1.00 -150.00 -20.00 0.00 150.00 20.00 300.00
GC 1a 3c BoxComponent 1 1 1 0.00 0.00 150.00 0.000 0.000 0.000 120.00 5.00 100.00 -120.00 -5.00 50.00 120.00 5.00 250.00
G 4d 2000.00 0.00 0.00 0.000 90.000 0.000 1.0000 1.0000 1.0000 1900.00 -150.00 0.00 2100.00 150.00 300.00
gatebox gates=2 components=2 channel=2 truncated=0
'''


class GateTests(unittest.TestCase):
    def test_parse_and_trigger_opening(self):
        parsed = cl.parse_gatebox(GATEBOX)
        self.assertEqual([g['actor'] for g in parsed['gates']], ['1a', '4d'])
        a = cl.gate_opening(parsed['gates'][0])
        self.assertEqual(a['basis'], 'trigger_volume')
        self.assertEqual(a['center'], [0, 0, 1.5])
        self.assertEqual([round(v, 6) for v in a['normal']], [0, 1, 0])
        self.assertEqual(a['trigger_size_m'], {'width': 2.4, 'height': 2.0, 'depth': .1})
        b = cl.gate_opening(parsed['gates'][1])                     # no trigger: bounds centre, local +Y
        self.assertEqual(b['basis'], 'bounds_centre_local_y')
        self.assertEqual(b['center'], [20, 0, 1.5])
        self.assertEqual([round(v, 6) for v in b['normal']], [-1, 0, 0])

    def test_incomplete_gatebox_rejected(self):
        with self.assertRaises(ValueError):
            cl.parse_gatebox(GATEBOX.splitlines()[0])
        with self.assertRaises(ValueError):
            cl.parse_gatebox('GC 9 1 BoxComponent 1 1 1 ' + '0 ' * 15 + '\ngatebox gates=0 components=1 channel=2 truncated=0')

    def test_saved_race_matched_to_live_gates(self):
        class Doc:
            race = {'StartTransform': {'Translation': {'X': 0, 'Y': 0, 'Z': 0}, 'Rotation': {'X': 0, 'Y': 0, 'Z': 0, 'W': 1}},
                    'CheckPoints': [{'RowName': 'DefaultOneW', 'Transform': {'Translation': {'X': 2000, 'Y': 10, 'Z': 0}}},
                                    {'RowName': 'SideNeon', 'Transform': {'Translation': {'X': 9000, 'Y': 0, 'Z': 0}}}],
                    'Laps': 3}
        saved, laps = cl.saved_race(Doc)
        self.assertEqual(laps, 3)
        self.assertEqual([s['row'] for s in saved], ['STARTGATE', 'DefaultOneW', 'SideNeon'])
        gates, unmatched = cl.match_saved(saved, cl.parse_gatebox(GATEBOX)['gates'])
        self.assertEqual(unmatched, 1)
        self.assertEqual([g['basis'] for g in gates], ['trigger_volume', 'bounds_centre_local_y', 'saved_pivot_unmeasured'])
        self.assertEqual(gates[1]['row'], 'DefaultOneW')


def probes_line(i, n, overrides=None):
    """P line, every probe free unless overridden {k: (cm, actor, comp)}."""
    vals = [(-1, '0', '0')] * (n + 2)
    for k, v in (overrides or {}).items():
        vals[k] = v
    return f'P {i} ' + ' '.join(f'{d:.1f} {a} {c}' for d, a, c in vals)


class ClassifyTests(unittest.TestCase):
    def setUp(self):
        self.prof = cl.resolve_profile('5inch')            # half 0.15/0.15/0.05, margin 0.5
        self.samples = cl.build_route([(0, 0, 2), (20, 0, 2)], 1.0)

    def parsed(self, body):
        text = body + '\nK c1 w1 StaticMeshComponent 2 0 3\nK ct g1 BoxComponent 1 1 1\nK cf g1 StaticMeshComponent 2 0 3' \
                      '\nK cl ls LandscapeHeightfieldCollisionComponent 2 0 3\nK cu u1 StaticMeshComponent -1 -1 -1' \
                      '\nA w1 NewRow_7 Actor\nA g1 - BP_Gate_C\nA ls - Landscape\nA u1 - StaticMeshActor' \
                      '\nsweep mode=sweep samples=21 segments=20 hits=3 overlap_only=1 truncated=0 seen_full=0 failed=0 channel=2 ms=5\n'
        return cl.parse_sweep(text, 16)

    def test_trigger_ignored_frame_and_wall_reported(self):
        parsed = self.parsed('\n'.join([
            'H 3 g1 ct 0 300.0 0.0 200.0 0.5',          # checkpoint trigger: overlap only
            'H 4 g1 cf 0 400.0 10.0 210.0 0.3',         # gate frame blocks
            'H 9 w1 c1 1 900.0 0.0 200.0 0.0',          # placed wall, start penetrating
            'H 10 w1 c1 0 1000.0 0.0 200.0 0.1',
            'H 12 u1 cu 0 1200.0 0.0 200.0 0.1',        # unknown response counts as blocking
        ]))
        res = cl.analyse(self.samples, parsed, self.prof, ndirs=16)
        self.assertEqual(res['overlap_only_ignored'], {'BoxComponent': 1})
        self.assertEqual(res['summary'], {'gate_frame_hit': 1, 'collision': 2})
        wall = next(i for i in res['issues'] if i['obstacle']['actor'] == 'w1')
        self.assertEqual((wall['samples'], wall['route_span_m'], wall['obstacle']['row']), (2, [9.0, 10.0], 'NewRow_7'))
        self.assertTrue(all(i['severity'] == 3 for i in res['issues']))
        self.assertEqual([i['id'] for i in res['issues']], [1, 2, 3])

    def test_probe_findings(self):
        lines = [probes_line(s['i'], 16) for s in self.samples]
        lines[5] = probes_line(5, 16, {4: (40.0, 'w1', 'c1')})     # right wall 0.40 m from centre -> 0.25 clear
        lines[6] = probes_line(6, 16, {16: (45.0, 'w1', 'c1')})    # ceiling 0.45 -> 0.40 clear
        lines[7] = probes_line(7, 16, {17: (20.0, 'ls', 'cl')})    # ground 0.20 -> 0.15 clear
        lines[4] = probes_line(4, 16, {12: (10.0, 'w1', 'c1')})    # inside the craft box, next to sample 5
        res = cl.analyse(self.samples, self.parsed('\n'.join(lines)), self.prof, ndirs=16)
        kinds = {i['kind']: i for i in res['issues']}
        self.assertEqual(set(kinds), {'tight', 'ceiling_low', 'ground_low'})
        tight = [i for i in res['issues'] if i['kind'] == 'tight']
        self.assertEqual(len(tight), 1)                      # same wall on adjacent samples merges
        self.assertEqual(tight[0]['measure_m'], -.05)
        self.assertEqual(tight[0]['severity'], 3)
        self.assertEqual(kinds['ground_low']['obstacle']['kind'], 'terrain')
        self.assertAlmostEqual(kinds['ceiling_low']['measure_m'], .4)
        self.assertEqual(res['min_clearance_m']['sample'], 4)
        self.assertEqual(tight[0]['route_span_m'], [4.0, 5.0])
        v = kinds['ground_low']['view']
        self.assertLess(v['x'], kinds['ground_low']['at_m'][0])   # camera behind the sample, looking along the route
        self.assertLess(v['pitch'], 0)
        # ignore_start_m suppresses ground proximity near the ends
        res = cl.analyse(self.samples, self.parsed('\n'.join(lines)), self.prof, ndirs=16, ignore_start_m=8)
        self.assertNotIn('ground_low', res['summary'])

    def test_gate_opening_small(self):
        # no detection plane known: the gate-centre probes decide
        gate = {'actor': 'g1', 'row': 'DefaultOneW', 'center': [10, 0, 2], 'normal': [1, 0, 0], 'basis': 'bounds_centre_local_y',
                'trigger_size_m': None}
        lines = [probes_line(s['i'], 16) for s in self.samples]
        frame = {4: (30.0, 'g1', 'cf'), 12: (30.0, 'g1', 'cf'), 16: (40.0, 'g1', 'cf'), 17: (90.0, 'g1', 'cf')}
        lines[10] = probes_line(10, 16, frame)                    # route sample next to the gate sees the frame too
        lines.append(probes_line(len(self.samples), 16, frame))   # gate-centre probe (index base + 0)
        res = cl.analyse(self.samples, self.parsed('\n'.join(lines)), self.prof, [gate], ndirs=16)
        self.assertNotIn('tight', res['summary'])          # the gate's own frame is judged per gate
        self.assertNotIn('ceiling_low', res['summary'])    # its top bar too
        g = res['gates'][0]
        self.assertEqual(g['probe_opening_m']['width'], .6)
        self.assertEqual(g['probe_opening_m']['from'], 'gate_centre')
        self.assertEqual(g['probe_opening_m']['height'], 1.3)
        self.assertEqual(g['required_m'], {'width': 1.3, 'height': 1.1})
        kinds = [i['kind'] for i in g['issues']]
        self.assertEqual(kinds, ['gate_width_small'])
        self.assertEqual(g['issues'][0]['severity'], 2)    # 0.6 m fits a 0.3 m quad, below the margin
        self.assertEqual(res['summary'], {'gate_width_small': 1})
        # with the game's detection plane known, the plane size wins over probes (which can start inside a pane)
        planed = dict(gate, basis='trigger_volume', trigger_size_m={'width': 3, 'height': 3, 'depth': .1},
                      side_axis=[0, 1, 0], up_axis=[0, 0, 1])
        res = cl.analyse(self.samples, self.parsed('\n'.join(lines)), self.prof, [planed], ndirs=16)
        self.assertEqual(res['gates'][0]['issues'], [])
        # without a gate-centre probe: the trigger size, never the nearest route sample
        res = cl.analyse(self.samples, self.parsed('\n'.join(lines[:-1])), self.prof, [planed], ndirs=16)
        self.assertIsNone(res['gates'][0]['probe_opening_m'])
        self.assertEqual(res['gates'][0]['issues'], [])

    def test_box_gap_is_point_to_box_distance(self):
        axes = cl.rot_axes(0, 0)
        half = (6.0, 1.275, 2.0)                                   # a 12 m truck
        self.assertAlmostEqual(cl.box_gap(2.7, (0, 1, 0), half, axes), 1.425)       # wall beside it
        u = (math.cos(math.radians(45)), math.sin(math.radians(45)), 0)             # hits 2.7 m ahead: beside it
        self.assertAlmostEqual(cl.box_gap(2.7 / u[1], u, half, axes), 1.425, 3)     # same wall, diagonal probe
        self.assertAlmostEqual(cl.box_gap(5.0, (1, 0, 0), half, axes), -1.0)        # obstacle inside the box

    def test_gate_hits_inside_opening_are_not_frame_hits(self):
        gate = {'actor': 'g1', 'center': [10, 0, 2], 'trigger_size_m': {'width': 1.7, 'height': 1.7, 'depth': .001},
                'side_axis': [0, 1, 0], 'up_axis': [0, 0, 1]}
        self.assertTrue(cl.inside_opening({'impact': [10, .05, 2]}, gate))       # pane at the centre
        self.assertFalse(cl.inside_opening({'impact': [10, .9, 2]}, gate))       # side bar
        self.assertFalse(cl.inside_opening({'impact': [10, 0, 2.9]}, gate))      # top bar
        self.assertFalse(cl.inside_opening({'impact': [10, 0, 2]}, None))        # no opening known

    def test_ground_corridor_and_contact(self):
        prof = cl.resolve_profile('car')                   # half width 0.95, margin 0.3
        samples = cl.build_route([(0, 0, 1), (10, 0, 1)], 1.0)
        lines = [probes_line(s['i'], 8, {2: (120.0, 'w1', 'c1'), 6: (300.0, 'w1', 'c1')}) for s in samples]
        lines += ['H 2 ls cl 0 200.0 0.0 0.0 0.2', 'D 2 0.00 ls cl']
        text = '\n'.join(lines) + '\nK c1 w1 StaticMeshComponent 2 0 3\nK cl ls LandscapeHeightfieldCollisionComponent 2 0 3' \
                                  '\nA w1 NewRow_7 Actor\nA ls - Landscape\nsweep mode=sweep samples=11 segments=10 hits=1\n'
        res = cl.analyse(samples, cl.parse_sweep(text, 8), prof, ndirs=8)
        self.assertEqual(res['corridor_m'], {'min': 4.2, 'median': 4.2})
        self.assertEqual(res['summary'], {'narrow_corridor': 1, 'ground_contact': 1})
        narrow = next(i for i in res['issues'] if i['kind'] == 'narrow_corridor')
        self.assertEqual((narrow['side'], narrow['measure_m'], narrow['samples']), ('right', .25, 11))

    def test_parse_rejects_garbage(self):
        with self.assertRaises(ValueError):
            cl.parse_sweep('H 1 2\nsweep samples=1', 16)
        with self.assertRaises(ValueError):
            cl.parse_sweep('P 0 -1 0 0', 16)


def win_to_posix(p):
    return Path(p.replace('Z:', '').replace('\\', '/'))


class FakeWorld:
    """Stands in for the agent: answers gatebox/sweep/mapinfo by writing synthetic native files."""

    def __init__(self, surface=('ls', 'cl'), road_hits=False):
        self.commands, self.requests = [], []
        self.surface, self.road_hits = surface, road_hits

    def run(self, command, timeout=30):
        self.commands.append(command)
        p = command.split()
        if p[0] == 'gatebox':
            win_to_posix(p[1]).write_text(GATEBOX)
            return ['gatebox gates=2 components=2']
        if p[0] == 'mapinfo':
            return ['folder Unsaved', 'gates 2']
        assert p[0] == 'sweep', command
        req = win_to_posix(p[1]).read_text().splitlines()
        self.requests.append(req)
        ndirs = int(next(l for l in req if l.startswith('R ')).split()[2])
        drape = any(l.startswith('M drape') for l in req)
        out = []
        for line in req:
            if not line.startswith('S '):
                continue
            q = line.split()
            i, x, y = int(q[1]), float(q[2]), float(q[3])
            if drape:
                out.append(f'D {i} 0.00 {self.surface[0]} {self.surface[1]}')
                continue
            if q[7] == '1' and self.road_hits and 1000 <= x <= 1400:
                out.append(f'H {i} {self.surface[0]} {self.surface[1]} 0 {x:.1f} {y:.1f} 5.0 0.4')  # scrapes the road tile
            if q[7] == '1' and 900 <= x <= 1000:
                out.append(f'H {i} w1 c1 0 {x:.1f} {y:.1f} 200.0 0.5')
            if q[7] == '1' and 0 <= x <= 100:
                out.append(f'H {i} 1a 3c 1 {x:.1f} {y:.1f} 150.0 0.0')   # start gate trigger
            if q[8] == '1':
                out.append(probes_line(i, ndirs))
        out += ['K c1 w1 StaticMeshComponent 2 0 3', 'K 3c 1a BoxComponent 1 1 1', 'K cl ls Landscape 2 0 3',
                'K rc rd StaticMeshComponent 2 0 3', 'A rd NewRow_42 Actor',
                'A w1 NewRow_7 Actor', 'A 1a - BP_Gate_C', 'A ls - Landscape',
                f'sweep mode={"drape" if drape else "sweep"} samples=1 segments=1 hits=1 overlap_only=0 truncated=0 seen_full=0 failed=0 channel=2 ms=3']
        win_to_posix(p[2]).write_text('\n'.join(out) + '\n')
        return ['sweep ok']


class PipelineTests(unittest.TestCase):
    def run_check(self, world=None, **kw):
        world = world or FakeWorld()
        with TemporaryDirectory() as tmp, patch.object(cl, 'REPORT_DIR', Path(tmp)), \
                patch.object(cl.agent, 'run', world.run), patch.object(cl.agent, 'session', contextlib.nullcontext), \
                patch.object(cl.agent, 'map_info', lambda: {'folder': 'Unsaved'}):
            reply = cl.check_route_clearance(**kw)
            report = json.loads(Path(reply['report_json']).read_text())
            html = Path(reply['report_html']).read_text()
        return world, reply, report, html

    def test_live_gate_route(self):
        world, reply, report, html = self.run_check(source='live', profile='5inch', spacing_m=1.0, laps=2)
        self.assertEqual(world.commands[0].split()[0], 'gatebox')
        self.assertTrue(reply['route']['closed'])
        self.assertEqual(reply['route']['order_basis'], 'level_order')
        self.assertEqual(reply['summary'], {'collision': 2})   # out and back through the same wall: two passes
        spans = sorted(i['route_span_m'][0] for i in reply['issues'])
        self.assertGreater(spans[1] - spans[0], 10)
        issue = reply['issues'][0]
        self.assertEqual(issue['obstacle']['row'], 'NewRow_7')
        self.assertEqual(set(issue['view']), {'x', 'y', 'z', 'yaw', 'pitch'})
        # the start gate's trigger is hit inside its opening: dropped before classification, not an obstacle
        self.assertGreaterEqual(reply['gate_opening_hits_ignored'], 1)
        self.assertNotIn('gate_frame_hit', reply['summary'])
        for key in ('profile', 'route', 'probe', 'issues', 'gates', 'limitations', 'path', 'clearance_profile', 'collision_channel'):
            self.assertIn(key, report)
        self.assertEqual(report['profile']['half_extents_m'], [.15, .15, .05])
        self.assertEqual(len(report['gates']), 2)
        self.assertIn('Route clearance', html)
        self.assertNotIn('__DATA__', html)

    def test_points_ground_mode_drapes(self):
        world, reply, report, _ = self.run_check(source='points', points=[[0, 0], [30, 0]], profile='car', spacing_m=2.0)
        self.assertEqual(reply['profile']['kind'], 'ground')
        self.assertEqual([c.split()[0] for c in world.commands], ['sweep', 'sweep'])
        ride = .15 + .75
        self.assertTrue(all(abs(p[2] - ride) < 1e-6 for p in report['path']))
        self.assertEqual(report['route']['drape_missing_samples'], [])
        self.assertIn('M drape 100000.0 5000.0 1', world.requests[0])   # [x, y]: from far above, foliage/gates skipped
        self.assertEqual(report['native']['jobs'], 2)      # the drape job counts towards complete
        self.assertEqual(report['path'][-1][3], report['route']['length_m'])  # plan path ends at the last sample
        self.assertIsNone(reply['corridor_m'])             # nothing to the sides within the probe length

    def test_ground_contact_on_placed_road_tiles(self):
        world = FakeWorld(surface=('rd', 'rc'), road_hits=True)
        _, reply, _, _ = self.run_check(world=world, source='points', points=[[0, 0, 0], [30, 0, 0]], profile='car', spacing_m=2.0)
        self.assertEqual(reply['summary'], {'collision': 1, 'ground_contact': 1})   # the wall w1, and the road tile
        by_kind = {i['kind']: i for i in reply['issues']}
        self.assertEqual(by_kind['collision']['obstacle']['row'], 'NewRow_7')
        self.assertEqual(by_kind['ground_contact']['obstacle']['row'], 'NewRow_42')   # placed tile, not a collision
        self.assertEqual(by_kind['ground_contact']['severity'], 1)

    def test_gate_probes_fired_from_gate_centres(self):
        world, _, report, _ = self.run_check(source='live', profile='5inch', spacing_m=1.0)
        gate_req = world.requests[-1]
        s_lines = [l.split() for l in gate_req if l.startswith('S ')]
        self.assertEqual(len(s_lines), 2)
        self.assertEqual([float(v) for v in s_lines[0][2:5]], [0, 0, 150])   # trigger centre, cm
        self.assertTrue(all(l[7:] == ['0', '1'] for l in s_lines))
        self.assertIsNone(report['gates'][0]['probe_opening_m'])           # fake probes all free: trigger size used

    def test_points_validation(self):
        for kw in (dict(source='points', points=[[0, 0, 0]]), dict(source='points', points=[[0, 0], [5, 5]]),
                   dict(source='bogus'), dict(source='points', points=[[0, 0, 0], [1, 1, 1]], directions=7),
                   dict(source='points', points=[[0, 0, 0], [1, 1, 1]], channel='Water')):
            with self.assertRaises(ValueError):
                self.run_check(**kw)


if __name__ == '__main__':
    unittest.main()
