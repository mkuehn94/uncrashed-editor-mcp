"""Budgets, native perf/rowflags/frametime parsing, offline and live analysis."""
import contextlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uncrashed_mcp import performance as perf
from uncrashed_mcp.catalog import Asset, Catalog


def asset(row, kind='mesh', category='Wall', name=None, triplanar=False, tags=()):
    return Asset(id=row, name=name or row, category=category, kind=kind, tags=tuple(tags), hidden=False, size_m=(1, 1, 1),
                 pivot='centre', triplanar=triplanar, collision=True, material=None, source=None if kind != 'mesh' else 'SM_' + row,
                 origin_cm=(0.0, 0.0, 50.0), extent_cm=(50.0, 50.0, 50.0), bounds_estimated=False)


CAT = Catalog({'Box': asset('Box', triplanar=True), 'Lamp': asset('Lamp', name='Street Lamp', tags=('lamp',)),
               'Deco': asset('Deco', kind='blueprint', category='Decals')}, {})
TRIS = {'SM_Box': 12, 'SM_Lamp': 500}
REAL_HULL = perf.hull_triangles


def a_line(addr, editor, row, cls, x, y, z, **kw):
    names = ('prim', 'sm', 'ism', 'inst', 'shadow', 'coll', 'phys', 'lights', 'dynlights', 'shadowlights', 'nevercull',
             'hidden', 'skel', 'fx', 'tris', 'movable', 'actorhidden')
    defaults = {'prim': 1, 'sm': 1, 'shadow': 1, 'coll': 1, 'nevercull': 1}
    vals = [str(kw.get(n, defaults.get(n, 0))) for n in names]
    return f'A {addr} {editor} {row} {cls} {x} {y} {z} 50 50 50 ' + ' '.join(vals)


def obj(i, row, x, y, z=0):
    return {'actor': f'file:{i}', 'row': row, 'group': '-', 'xf': [x, y, z, 0, 0, 0, 1, 1, 1]}


@patch.object(perf, 'hull_triangles', lambda: TRIS)
class AnalysisTests(unittest.TestCase):
    def test_budget_overrides(self):
        b = perf.budgets({'placed_objects': 10, 'dynamic_lights': {'warn': 1, 'high': 2}})
        self.assertEqual((b['placed_objects']['warn'], b['placed_objects']['high']), (10, 25))
        self.assertEqual(b['dynamic_lights']['high'], 2)
        self.assertEqual(perf.grade(11, b['placed_objects']), 'warn')
        self.assertEqual(perf.grade(30, b['placed_objects']), 'high')
        b = perf.budgets({'dynamic_lights': {'warn': 60}, 'shadow_lights': {'high': 2}})
        self.assertEqual((b['dynamic_lights']['warn'], b['dynamic_lights']['high']), (60, 150))  # high follows warn x2.5
        self.assertEqual((b['shadow_lights']['warn'], b['shadow_lights']['high']), (2, 2))
        for bad in ({'nope': 1}, {'placed_objects': {'warn': 5, 'high': 1}}, {'triangles': float('nan')},
                    {'triangles': {'warn': float('nan')}}, {'triangles': {}}, {'triangles': {'max': 1}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                perf.budgets(bad)

    def test_rowflags_strip_blueprint_suffixes(self):
        text = ('R\tNewRow_1\tCollision?_12_8F3C11A24B5A6F1D2E3C4B5A6F7E8D9C\tBoolProperty\t1'
                '\tEndculldistance_7_0123456789ABCDEF0123456789ABCDEF\tFloatProperty\t0'
                '\tMesh_2_0123456789ABCDEF0123456789ABCDEF\tObjectProperty\tSM_X\n'
                'garbage line\n')
        rows = perf.parse_rowflags(text)
        self.assertEqual(perf.flag(rows['NewRow_1'], 'Collision?'), True)
        self.assertEqual(perf.flag(rows['NewRow_1'], 'Endculldistance'), 0.0)
        self.assertEqual(perf.flag(rows['NewRow_1'], 'mesh'), 'SM_X')
        self.assertIsNone(perf.flag(rows['NewRow_1'], 'PhysicsEnabled?'))

    def test_parse_perf_strict(self):
        text = '\n'.join([a_line('ab', 1, 'Box', 'Actor', 100, 200, 0), a_line('cd', 0, '-', 'InstancedFoliageActor', 0, 0, 0, ism=1, inst=900),
                          'I cd HierarchicalInstancedStaticMeshComponent SM_Pine 900',
                          'L ab PointLightComponent 2 1 5000 100.0 200.0 300.0',
                          'end actors=2 components=4 truncated=0 trisfn=0 instancefn=1'])
        scan = perf.parse_perf(text)
        self.assertEqual(scan['actors'][0]['center'], [1, 2, 0])
        self.assertIsNone(scan['actors'][1]['row'])
        self.assertEqual(scan['instanced'][0]['instances'], 900)
        self.assertEqual(scan['lights'][0]['at_m'], [1, 2, 3])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            perf.parse_perf(text.replace('actors=2', 'actors=3'))
        with self.assertRaises(ValueError):
            perf.parse_perf('Z what\nend actors=0')

    def test_offline_counts_cells_and_warnings(self):
        objects = [obj(i, 'Box', (i % 10) * 100, (i // 10) * 100) for i in range(60)]
        objects += [obj(100 + i, 'Lamp', 5000 + i * 100, 0) for i in range(5)] + [obj(200, 'Deco', 0, 0)]
        records = perf.offline_records(objects, CAT, None)
        self.assertIsNone(records[0]['shadow'])
        self.assertAlmostEqual(records[0]['center'][2], 0.5)
        limits = perf.budgets({'placed_objects': 20, 'cell_objects': 20, 'blueprint_objects': {'warn': 0, 'high': 5}})
        report = perf.analyse(records, CAT, limits, cell=5, top=5, consolidation={'objects_removed': 60, 'objects_added': 2, 'saved': 58})
        m = report['metrics']
        self.assertEqual((m['placed_objects'], m['blueprint_objects'], m['possible_light_objects']), (66, 1, 5))
        self.assertEqual(m['triangles'], 60 * 12 + 5 * 500)
        self.assertIsNone(m['dynamic_lights'])
        self.assertEqual(report['top_rows'][0]['row'], 'Box')
        self.assertEqual(report['densest_cells'][0]['objects'], 26)  # the 0..5 m cell holds 5 x 5 boxes + the decal
        self.assertIn('camera', report['densest_cells'][0])
        metrics = {w['metric']: w for w in report['warnings']}
        self.assertEqual(metrics['placed_objects']['severity'], 'high')
        self.assertIn('consolidate_scenery can merge 60', metrics['placed_objects']['advice'])
        self.assertEqual(metrics['blueprint_objects']['severity'], 'warn')
        self.assertEqual(report['status'], 'high')

    def test_offline_uses_cached_row_flags(self):
        flags = {'Box': {perf._key('CastdynamicShadow'): False, perf._key('Endculldistance'): 0.0, perf._key('PhysicsEnabled?'): True}}
        records = perf.offline_records([obj(0, 'Box', 0, 0), obj(1, 'Lamp', 0, 0)], CAT, flags)
        self.assertEqual((records[0]['shadow'], records[0]['nevercull'], records[0]['phys']), (0, 1, 1))
        self.assertIsNone(records[1]['nevercull'])

    def test_live_lights_instancing_and_engine_triangles(self):
        scan = perf.parse_perf('\n'.join([
            a_line('a1', 1, 'Box', 'Actor', 0, 0, 0, tris=24), a_line('a2', 1, 'Lamp', 'Actor', 0, 0, 0, lights=1, dynlights=1, tris=480),
            a_line('g', 1, '-', 'BP_Gate_C', 0, 0, 0), a_line('f', 0, '-', 'InstancedFoliageActor', 0, 0, 0, sm=0, ism=2, inst=50),
            'I f HierarchicalInstancedStaticMeshComponent SM_Tree 50',
            'L a2 PointLightComponent 2 1 5000 0 0 300', 'L x DirectionalLightComponent 1 1 10 0 0 0',
            'end actors=4 components=7 truncated=0 trisfn=1 instancefn=1']))
        report = perf.analyse(scan['actors'], CAT, perf.budgets({'shadow_lights': 1}), lights=scan['lights'],
                              instanced=scan['instanced'], live=True)
        m = report['metrics']
        self.assertEqual((m['placed_objects'], m['gates'], m['level_actors'], m['non_editor_actors']), (2, 1, 4, 1))
        # placed objects graded; base-level light / foliage components informational
        self.assertEqual((m['dynamic_lights'], m['shadow_lights'], m['mesh_components']), (1, 1, 2))
        self.assertEqual((m['dynamic_lights_base'], m['shadow_lights_base'], m['mesh_components_base']), (1, 1, 2))
        self.assertEqual((m['shadow_casters'], m['shadow_casters_base'], m['never_culled'], m['never_culled_base']), (2, 1, 2, 1))
        self.assertEqual(m['triangles'], 504)
        self.assertEqual(m['triangles_source'], 'engine LOD0 counts')
        self.assertEqual(report['engine_instancing']['editor_ism_components'], 0)
        self.assertIn('no placed object owns an ISM', report['engine_instancing']['meaning'])
        self.assertIn('renderer may still merge', report['engine_instancing']['meaning'])
        self.assertFalse(any(w['metric'] == 'shadow_lights' for w in report['warnings']))  # 1 placed, budget warn > 1
        report = perf.analyse(scan['actors'], CAT, perf.budgets({'shadow_lights': {'warn': 0, 'high': 5}}), lights=scan['lights'],
                              instanced=scan['instanced'], live=True, render_cvars={'r.MeshDrawCommands.DynamicInstancing': 0})
        self.assertEqual(next(w for w in report['warnings'] if w['metric'] == 'shadow_lights')['severity'], 'warn')
        self.assertIn('DynamicInstancing=0', report['engine_instancing']['meaning'])
        self.assertIn('OFF', report['engine_instancing']['meaning'])

    def test_cell_budgets_scale_with_cell_size_unless_overridden(self):
        limits = perf.budgets()
        self.assertEqual(perf.cell_limits(limits, 100)['cell_objects']['warn'], 1200)
        self.assertEqual(perf.cell_limits(limits, 25)['cell_objects']['high'], 200)
        self.assertEqual(perf.cell_limits(limits, 50)['cell_objects']['warn'], 300)
        self.assertEqual(perf.cell_limits(perf.budgets({'cell_objects': 10}), 100)['cell_objects']['warn'], 10)
        records = perf.offline_records([obj(i, 'Box', i, 0) for i in range(400)], CAT, None)
        self.assertFalse(any(w['metric'] == 'cell_objects' for w in perf.analyse(records, CAT, limits, cell=100)['warnings']))
        report = perf.analyse(records, CAT, limits, cell=50)
        self.assertTrue(any(w['metric'] == 'cell_objects' for w in report['warnings']))
        self.assertEqual(report['cell_budgets']['cell_objects']['warn'], 300)

    def test_triangle_advice_names_heaviest_rows_not_most_numerous(self):
        objects = [obj(i, 'Box', i * 100, 0) for i in range(30)] + [obj(100, 'Lamp', 0, 500)]  # 360 vs 500 triangles
        records = perf.offline_records(objects, CAT, None)
        report = perf.analyse(records, CAT, perf.budgets({'triangles': 10}), top=1)
        self.assertEqual([t['row'] for t in report['top_rows']], ['Box'])
        self.assertEqual([t['row'] for t in report['top_rows_by_triangles']], ['Lamp', 'Box'])
        w = next(w for w in report['warnings'] if w['metric'] == 'triangles')
        self.assertIn('Street Lamp (Lamp) 500', w['advice'])


class FrameTests(unittest.TestCase):
    def test_frame_stats_counted_and_percentiles(self):
        samples = [perf.parse_frame(f'frame delta={d} real={r} frames={f} clock=1.0')
                   for d, r, f in ((0.016, 10.0, 100), (0.017, 10.5, 130), (0.050, 11.0, 160))]
        s = perf.frame_stats(samples)
        self.assertEqual(s['samples'], 3)
        self.assertEqual(s['p95_ms'], 50.0)
        self.assertAlmostEqual(s['avg_ms_counted'], 1000 / 60, places=1)
        self.assertIsNone(perf.parse_frame('camera x=1'))
        with self.assertRaises(ValueError):
            perf.frame_stats([])

    def test_frame_stats_counted_uses_qpc_clock_and_survives_missing_delta(self):
        samples = [perf.parse_frame(f'frame delta=-1 real=-1 frames={f} clock={c}') for f, c in ((10, 5.0), (70, 6.0))]
        s = perf.frame_stats(samples)
        self.assertAlmostEqual(s['avg_ms'], 16.67, places=2)
        self.assertIsNone(s['p95_ms'])
        self.assertEqual(s['clock'], 'QPC clock')
        with self.assertRaisesRegex(ValueError, 'no frame samples'):
            perf.frame_stats([perf.parse_frame('frame delta=-1 real=1 frames=-1 clock=1')] * 3)

    def test_measure_stops_at_hard_deadline_and_releases(self):
        with patch.object(perf.agent, 'run', return_value=['garbage']), \
             patch('uncrashed_mcp.live.move_camera'), patch('uncrashed_mcp.live.release_camera') as release:
            with self.assertRaisesRegex(RuntimeError, 'parseable samples'):
                perf.measure_frames(0.3, [{'x': 1, 'y': 2, 'z': 3}], settle=0, grace=0.1)
        release.assert_called_once()

    def test_frame_notes_vsync_and_background(self):
        notes = perf.frame_notes({'avg_ms': 16.7}, {'r.VSync': 1, 't.IdleWhenNotForeground': 1, 't.MaxFPS': 0})
        self.assertTrue(any('VSync' in n for n in notes) and any('unfocused' in n for n in notes))
        self.assertEqual(perf.frame_notes({'avg_ms': 40}, {'t.MaxFPS': 60, 'r.VSync': 0}), [])

    def test_frame_stats_without_frame_counter(self):
        s = perf.frame_stats([perf.parse_frame('frame delta=0.02 real=1 frames=-1 clock=1')] * 4)
        self.assertNotIn('avg_ms_counted', s)
        self.assertEqual(s['avg_ms'], 20.0)

    def test_measure_moves_camera_and_always_releases(self):
        calls = []

        def run(cmd, timeout=30):
            calls.append(cmd)
            if cmd.startswith('cvar'):
                return [f'{cmd} float={60 if "MaxFPS" in cmd else 0}']
            return [f'frame delta=0.0166 real={len(calls) * 0.166} frames={len(calls) * 10} clock=0']
        with patch.object(perf.agent, 'run', side_effect=run), \
             patch('uncrashed_mcp.live.move_camera') as move, patch('uncrashed_mcp.live.release_camera') as release:
            out = perf.measure_frames(0.3, [{'x': 1, 'y': 2, 'z': 3, 'yaw': 0}], settle=0)
        move.assert_called_once()
        release.assert_called_once()
        self.assertEqual(out[0]['cvars']['t.MaxFPS'], 60)
        self.assertIn('t.MaxFPS cap', out[0]['note'])
        with patch.object(perf.agent, 'run', side_effect=RuntimeError('boom')), \
             patch('uncrashed_mcp.live.move_camera'), patch('uncrashed_mcp.live.release_camera') as release:
            with self.assertRaises(RuntimeError):
                perf.measure_frames(0.3, [{'x': 1, 'y': 2, 'z': 3}], settle=0)


@patch.object(perf, 'hull_triangles', lambda: TRIS)
class EntryTests(unittest.TestCase):
    def test_offline_saved_map_file(self):
        data = {'S_AssetsTransform': [{'Rname': 'Box', 'Transform': [
            {'Rotation': {'X': 0, 'Y': 0, 'Z': 0, 'W': 1}, 'Translation': {'X': 100 * i, 'Y': 0, 'Z': 0}, 'Scale3D': {'X': 1, 'Y': 1, 'Z': 1}}
            for i in range(4)]}]}
        with tempfile.TemporaryDirectory() as tmp, patch.object(perf, 'REPORT_DIR', Path(tmp)), \
             patch.object(perf, 'ROWFLAGS_CACHE', Path(tmp) / 'none.txt'), \
             patch.object(perf.consolidate, 'row_shapes', lambda c, rows: {r: (True, 'box') for r in rows}), \
             patch.object(perf.consolidate, 'REPORT_DIR', Path(tmp)):
            path = Path(tmp) / 'm.json'
            path.write_text(json.dumps(data))
            r = perf.check_performance(CAT, name=str(path))
            self.assertEqual(r['mode'], 'offline')
            self.assertEqual(r['metrics']['placed_objects'], 4)
            self.assertEqual(r['consolidation']['saved'], 3)
            self.assertTrue(Path(r['report_json']).is_file())
            self.assertEqual(perf.check_performance(CAT, name=str(path), region=[-1, -1, 1.5, 1])['metrics']['placed_objects'], 2)
            with self.assertRaisesRegex(ValueError, 'live'):
                perf.check_performance(CAT, name=str(path), measure=True)

    def test_live_runs_native_scan_with_region_in_cm(self):
        commands = []

        def run(cmd, timeout=30):
            commands.append(cmd)
            path = cmd.split()[1]
            local = Path(path[2:].replace('\\', '/')) if path[:2].lower() == 'z:' else Path(path)
            if cmd.startswith('perf'):
                local.write_text(a_line('a1', 1, 'Box', 'Actor', 0, 0, 0) + '\nend actors=1 components=1 truncated=0 trisfn=0 instancefn=1\n')
            elif cmd.startswith('rowflags'):
                local.write_text('R\tBox\tCastdynamicShadow\tBoolProperty\t0\n')
            return []
        with tempfile.TemporaryDirectory() as tmp, patch.object(perf, 'REPORT_DIR', Path(tmp)), \
             patch.object(perf, 'ROWFLAGS_CACHE', Path(tmp) / 'rowflags.txt'), \
             patch.object(perf.agent, 'session', contextlib.nullcontext), patch.object(perf.agent, 'run', side_effect=run), \
             patch('uncrashed_mcp.groups.snapshot', return_value={'session': 's', 'objects': []}), \
             patch.object(perf.consolidate, 'REPORT_DIR', Path(tmp)):
            r = perf.check_performance(CAT, region=[10, -2, 0, 2])
            self.assertTrue((Path(tmp) / 'rowflags.txt').read_text().startswith('R\tBox'))
        self.assertTrue(re.fullmatch(r'perf \S+ 0\.0 -200\.0 1000\.0 200\.0', commands[0]), commands[0])
        self.assertEqual(r['mode'], 'live')
        self.assertEqual(r['metrics']['placed_objects'], 1)
        self.assertEqual(r['region_m'], [0, -2, 10, 2])

    def _live(self, perf_text, extra=()):
        def run(cmd, timeout=30):
            path = cmd.split()[1] if len(cmd.split()) > 1 else ''
            local = Path(path[2:].replace('\\', '/')) if path[:2].lower() == 'z:' else Path(path)
            if cmd.startswith('perf'):
                local.write_text(perf_text)
            elif cmd.startswith('rowflags'):
                local.write_text('')
            return []
        tmp = tempfile.mkdtemp()
        stack = contextlib.ExitStack()
        for p in (patch.object(perf, 'REPORT_DIR', Path(tmp)), patch.object(perf, 'ROWFLAGS_CACHE', Path(tmp) / 'rf.txt'),
                  patch.object(perf.agent, 'session', contextlib.nullcontext), patch.object(perf.agent, 'run', side_effect=run),
                  patch('uncrashed_mcp.groups.snapshot', return_value={'session': 's', 'objects': []}),
                  patch.object(perf.consolidate, 'REPORT_DIR', Path(tmp)), *extra):
            stack.enter_context(p)
        return stack

    def test_measure_slow_frames_warns_and_raises_status(self):
        text = a_line('a1', 1, 'Box', 'Actor', 0, 0, 0) + '\nend actors=1 components=1 truncated=0 trisfn=0 instancefn=1\n'
        slow = [{'samples': 10, 'avg_ms': 45.0, 'avg_ms_counted': 45.0, 'p50_ms': 44.0, 'p95_ms': 60.0, 'max_ms': 70.0,
                 'pose': {'x': 0, 'y': 0, 'z': 10}, 'cvars': {}}]
        with self._live(text, [patch.object(perf, 'measure_frames', return_value=slow)]):
            r = perf.check_performance(CAT, measure=True)
        self.assertEqual(r['status'], 'high')
        metrics = {w['metric']: w for w in r['warnings']}
        self.assertEqual(metrics['frame_ms_avg']['severity'], 'high')
        self.assertEqual(metrics['frame_ms_p95']['severity'], 'high')
        self.assertEqual(r['frame_time'], slow)

    def test_truncated_scan_is_reported(self):
        text = a_line('a1', 1, 'Box', 'Actor', 0, 0, 0) + '\nend actors=1 components=1 truncated=1 trisfn=0 instancefn=1\n'
        with self._live(text):
            r = perf.check_performance(CAT)
        self.assertEqual(r['warnings'][0]['metric'], 'scan')
        self.assertEqual(r['status'], 'info')

    def test_hull_header_version_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'meshes.bin'
            path.write_bytes(b'UMSH' + (2).to_bytes(4, 'little') + (0).to_bytes(4, 'little'))
            REAL_HULL.cache_clear()
            try:
                with patch.object(perf.meshes, 'DATA', path), self.assertRaisesRegex(RuntimeError, 'version 2'):
                    REAL_HULL()
            finally:
                REAL_HULL.cache_clear()

    def test_real_hull_triangle_header(self):
        counts = REAL_HULL()
        if not counts:
            self.skipTest('meshes.bin not available')
        self.assertEqual(counts.get('SM_BaseGround_0-1'), 12)
        self.assertTrue(all(isinstance(v, int) and v >= 0 for v in counts.values()))


if __name__ == '__main__':
    unittest.main()
