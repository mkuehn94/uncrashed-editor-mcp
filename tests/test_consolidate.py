"""Pure lattice planning, shape test and journaled apply of consolidate_scenery."""
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uncrashed_mcp import consolidate as con
from uncrashed_mcp.catalog import Asset, Catalog, load_catalog
from uncrashed_mcp.geom import quat_rotate, quat_rotator


def asset(row='Tile', triplanar=True, origin=(0, 0, 5), extent=(200, 200, 5)):
    return Asset(id=row, name=row, category='Ground', kind='mesh', tags=(), hidden=False, size_m=(4, 4, .1), pivot='base',
                 triplanar=triplanar, collision=True, material=None, source='SM_' + row, origin_cm=tuple(map(float, origin)),
                 extent_cm=tuple(map(float, extent)), bounds_estimated=False)


CAT = Catalog({'Tile': asset(), 'Prop': asset('Prop', triplanar=False), 'Slope': asset('Slope')}, {})
SHAPES = {'Tile': (True, 'box'), 'Prop': (False, 'not triplanar'), 'Slope': (False, 'sloped face')}


def tile(i, x, y, z=0, rot=(0, 0, 0), scale=(1, 1, 1), row='Tile', group='-'):
    return {'actor': f'{i:x}', 'row': row, 'group': group, 'xf': [x, y, z, *rot, *scale]}


def corners(o, a):
    q = quat_rotator(*o['xf'][3:6])
    pts = []
    for dx in (-1, 1):
        for dy in (-1, 1):
            for dz in (-1, 1):
                local = [(a.origin_cm[i] + (dx, dy, dz)[i] * a.extent_cm[i]) * o['xf'][6 + i] for i in range(3)]
                w = quat_rotate(q, local)
                pts.append([o['xf'][i] + w[i] for i in range(3)])
    return pts


def box_mesh(top_slope=False):
    v = [(x, y, z) for x in (0, 1) for y in (0, 2) for z in (0, .5)]
    if top_slope:
        v[7] = (1, 2, .3)
    idx = lambda x, y, z: x * 4 + y * 2 + z
    quads = [(idx(0, 0, 0), idx(0, 1, 0), idx(0, 1, 1), idx(0, 0, 1)), (idx(1, 0, 0), idx(1, 1, 0), idx(1, 1, 1), idx(1, 0, 1)),
             (idx(0, 0, 0), idx(1, 0, 0), idx(1, 0, 1), idx(0, 0, 1)), (idx(0, 1, 0), idx(1, 1, 0), idx(1, 1, 1), idx(0, 1, 1)),
             (idx(0, 0, 0), idx(1, 0, 0), idx(1, 1, 0), idx(0, 1, 0)), (idx(0, 0, 1), idx(1, 0, 1), idx(1, 1, 1), idx(0, 1, 1))]
    tris = []
    for a, b, c, d in quads:
        tris += [(a, b, c), (a, c, d)]
    return v, tris


@patch.object(con, 'row_shapes', lambda catalog, rows: {r: SHAPES[r] for r in rows})
class PlanTests(unittest.TestCase):
    def test_grid_merges_into_one_scaled_copy_covering_the_union(self):
        objs = [tile(i, 400 * (i % 3), 400 * (i // 3)) for i in range(6)]
        p = con.plan(objs, CAT)
        self.assertEqual((p['objects_removed'], p['objects_added'], p['saved']), (6, 1, 5))
        m = p['merges'][0]
        self.assertEqual(sorted(m['grid']), [2, 3])
        merged = {'xf': m['xf']}
        lo = [min(c[i] for o in objs for c in corners(o, CAT.assets['Tile'])) for i in range(3)]
        hi = [max(c[i] for o in objs for c in corners(o, CAT.assets['Tile'])) for i in range(3)]
        got = corners(merged, CAT.assets['Tile'])
        for i in range(3):
            self.assertAlmostEqual(min(c[i] for c in got), lo[i], places=3)
            self.assertAlmostEqual(max(c[i] for c in got), hi[i], places=3)

    def test_rotated_lattice_respects_rotation(self):
        rot = (10, 30, 0)
        q = quat_rotator(*rot)
        objs = []
        for i in range(4):
            off = quat_rotate(q, [400 * i, 0, 0])
            objs.append(tile(i, 1000 + off[0], 50 + off[1], 20 + off[2], rot=rot))
        p = con.plan(objs, CAT)
        self.assertEqual(p['objects_added'], 1)
        m = p['merges'][0]
        self.assertAlmostEqual(m['xf'][6], 4)
        self.assertAlmostEqual(m['xf'][7], 1)
        got = corners({'xf': m['xf']}, CAT.assets['Tile'])
        want = [c for o in objs for c in corners(o, CAT.assets['Tile'])]
        for i in range(3):
            self.assertAlmostEqual(min(c[i] for c in got), min(c[i] for c in want), places=2)
            self.assertAlmostEqual(max(c[i] for c in got), max(c[i] for c in want), places=2)

    def test_gaps_heights_rotations_groups_and_scales_split(self):
        objs = [tile(0, 0, 0), tile(1, 410, 0),                       # 10 cm gap: not on lattice
                tile(2, 0, 2000, z=30), tile(3, 400, 2000),           # different height
                tile(4, 0, 4000, rot=(0, 90, 0)), tile(5, 400, 4000),  # different rotation
                tile(6, 0, 6000, group='a'), tile(7, 400, 6000),       # different group
                tile(8, 0, 8000, scale=(1, 1, 2)), tile(9, 400, 8000)]
        self.assertEqual(con.plan(objs, CAT)['merges'], [])

    def test_never_merges_non_triplanar_or_non_box(self):
        objs = [tile(i, 400 * i, 0, row='Prop') for i in range(3)] + [tile(9 + i, 400 * i, 0, row='Slope') for i in range(3)]
        p = con.plan(objs, CAT)
        self.assertEqual(p['merges'], [])
        self.assertEqual(p['skipped_rows']['Slope']['count'], 3)
        self.assertIn('Prop', [c['row'] for c in p['manual_candidates']])
        with self.assertRaisesRegex(ValueError, 'not triplanar'):
            con.plan(objs, CAT, rows=['Prop'])

    def test_duplicates_tolerance_region_and_max_size(self):
        objs = [tile(i, 400 * i + (1 if i == 2 else 0), 0) for i in range(5)] + [tile(99, 0, 0)]
        p = con.plan(objs, CAT, tolerance=0.02, max_size=8)
        self.assertEqual(len(p['duplicates']), 1)
        self.assertEqual([m['tiles'] for m in p['merges']], [2, 2])
        strict = con.plan(objs, CAT, tolerance=0.005)  # 1 cm off: that tile leaves the run, both sides merge
        self.assertEqual([m['tiles'] for m in strict['merges']], [2, 2])
        self.assertNotIn('2', [o['actor'] for m in strict['merges'] for o in m['members']])
        self.assertEqual(con.plan(objs, CAT, region=[-1, -1, 5, 1])['considered'], 3)

    def test_largest_rectangle_and_l_shape(self):
        cells = {(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)}
        self.assertEqual(con.largest_rectangle(cells, 9, 9)[2:], (3, 1))
        objs = [tile(i, 400 * u, 400 * v) for i, (u, v) in enumerate(sorted(cells))]
        p = con.plan(objs, CAT)
        self.assertEqual(sorted(m['tiles'] for m in p['merges']), [2, 3])

    def test_commands_and_chunks(self):
        p = con.plan([tile(i, 400 * i, 0) for i in range(3)], CAT)
        lines = con._commands(p['merges'])
        self.assertTrue(lines[0].startswith('C Tile - '))
        self.assertEqual([l[0] for l in lines], ['C', 'D', 'D', 'D'])
        merges = [{'tiles': 3}, {'tiles': 3}, {'tiles': 5}]
        self.assertEqual([len(c) for c in con._chunks(merges, 8)], [2, 1])

    def _apply_env(self, snaps, after=lambda commands: None, fail_on=None):
        """Mocks for live apply: snapshots in order (last repeats), _apply writes a journal whose 'after' is
        the merged result unless after() returns something else."""
        tmp = Path(tempfile.mkdtemp())
        applied, undone = [], []
        snaps = list(snaps)

        def snapshot():
            return snaps.pop(0) if len(snaps) > 1 else snaps[0]

        def fake_apply(info, root, before, affected, commands, label):
            applied.append((info, root, before, affected, commands, label))
            if fail_on is not None and len(applied) == fail_on:
                raise RuntimeError('native edit failed')
            tx = f'{len(applied):032x}'
            objs = after(commands)
            if objs is None:
                creates = [c.split() for c in commands if c.startswith('C ')]
                objs = [{'actor': f'n{i}', 'row': c[1], 'group': c[2], 'xf': [float(v) for v in c[3:]]} for i, c in enumerate(creates)]
            path = tmp / f'{tx}.json'
            path.write_text(json.dumps({'after': {'session': 's', 'objects': objs}}))
            return {'transaction': tx, 'journal': str(path)}
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(con.agent, 'session', contextlib.nullcontext))
        snap_mock = stack.enter_context(patch.object(con.groups, 'snapshot', side_effect=snapshot))
        stack.enter_context(patch.object(con.managed, 'context', return_value=({'folder': 'm', 'base': 'b'}, tmp)))
        stack.enter_context(patch.object(con.groups, '_apply', side_effect=fake_apply))
        stack.enter_context(patch.object(con.groups, 'undo_edit', side_effect=lambda tx, preview: undone.append(tx)))
        stack.enter_context(patch.object(con, 'REPORT_DIR', tmp))
        return stack, applied, undone, snap_mock

    def test_apply_uses_journaled_group_edit_and_verifies(self):
        objs = [tile(i, 400 * i, 0) for i in range(3)]
        snap = {'session': 's', 'objects': objs}
        stack, applied, undone, _ = self._apply_env([snap])
        with stack:
            preview = con.consolidate_scenery(CAT)
            self.assertTrue(preview['preview'])
            self.assertEqual(applied, [])
            done = con.consolidate_scenery(CAT, preview=False)
        self.assertEqual(done['transactions'], ['0' * 31 + '1'])
        self.assertEqual(done['verified'], 1)
        self.assertEqual(done['undo'], 'undo_edit <transaction> with apply (preview off) for each id, newest first')
        info, root, before, affected, commands, label = applied[0]
        self.assertEqual(affected, ['-'])
        self.assertEqual(len(commands), 4)
        self.assertEqual(before['objects'], objs)
        self.assertEqual(undone, [])

    def test_wrong_replacement_scale_is_undone_and_reported(self):
        objs = [tile(i, 400 * i, 0) for i in range(3)]

        def clamped(commands):
            c = next(c.split() for c in commands if c.startswith('C '))
            xf = [float(v) for v in c[3:]]
            xf[6] = 1.0  # Spawn Asset ignored the scale
            return [{'actor': 'n0', 'row': c[1], 'group': c[2], 'xf': xf}]
        stack, applied, undone, _ = self._apply_env([{'session': 's', 'objects': objs}], after=clamped)
        with stack, self.assertRaisesRegex(RuntimeError, r'did not come out as requested.*requested_scale.*was undone'):
            con.consolidate_scenery(CAT, preview=False)
        self.assertEqual(undone, ['0' * 31 + '1'])

    def test_a_failed_auto_undo_names_the_transaction_without_call_syntax(self):
        objs = [tile(i, 400 * i, 0) for i in range(3)]
        stack, applied, undone, _ = self._apply_env([{'session': 's', 'objects': objs}], after=lambda commands: [])
        tx = '0' * 31 + '1'
        with stack, patch.object(con.groups, 'undo_edit', side_effect=RuntimeError('agent gone')), \
                self.assertRaises(RuntimeError) as caught:
            con.consolidate_scenery(CAT, preview=False)
        self.assertIn(f'AUTO-UNDO FAILED (agent gone); run undo_edit {tx} with apply (preview off);', str(caught.exception))
        self.assertNotIn('preview=False', str(caught.exception))

    def test_canary_then_chunks_resnapshot_and_partial_failure(self):
        objs = [tile(3 * r + i, 400 * i, 2000 * r, scale=(1, 1, 1)) for r in range(3) for i in range(3)]
        objs += [tile(50 + i, 800 * i, 9000, scale=(2, 1, 1)) for i in range(3)]  # biggest scale -> canary
        snap = {'session': 's', 'objects': objs}
        stack, applied, undone, snap_mock = self._apply_env([snap])
        with stack, patch.object(con, 'CHUNK', 8):
            done = con.consolidate_scenery(CAT, preview=False)
        self.assertEqual(len(done['transactions']), 3)  # canary + 2 chunks of <= 8 commands
        self.assertTrue(applied[0][4][0].startswith('C Tile - '))
        self.assertAlmostEqual(float(applied[0][4][0].split()[9]), 6.0)  # canary is the 2x-scaled run (3 x 2)
        self.assertTrue(all(len(a[4]) <= 8 for a in applied))
        self.assertGreaterEqual(snap_mock.call_count, 3)  # plan + re-snapshot before each later chunk
        stack, applied, undone, _ = self._apply_env([snap], fail_on=2)
        with stack, patch.object(con, 'CHUNK', 8), self.assertRaisesRegex(RuntimeError, r'native edit failed; already applied transactions=\[\'0+1\'\]'):
            con.consolidate_scenery(CAT, preview=False)

    def test_member_changed_between_chunks_lists_applied(self):
        objs = [tile(3 * r + i, 400 * i, 2000 * r) for r in range(2) for i in range(3)]
        moved = [dict(o, xf=[o['xf'][0] + 50] + o['xf'][1:]) for o in objs]
        stack, applied, undone, _ = self._apply_env([{'session': 's', 'objects': objs}, {'session': 's', 'objects': moved}])
        with stack, self.assertRaisesRegex(RuntimeError, r'changed during consolidation; applied=\[\'0+1\'\]'):
            con.consolidate_scenery(CAT, preview=False)
        self.assertEqual(len(applied), 1)

    def test_yaw_180_and_minus_180_and_plane_boundary_share_buckets(self):
        self.assertEqual(con._qkey(quat_rotator(0, 180, 0)), con._qkey(quat_rotator(0, -180, 0)))
        self.assertEqual(con._qkey(quat_rotator(0, 90, 0)), con._qkey({k: -v for k, v in quat_rotator(0, 90, 0).items()}))
        objs = [tile(0, 0, 0, rot=(0, 180, 0)), tile(1, -400, 0, rot=(0, -180, 0))]
        self.assertEqual(con.plan(objs, CAT)['saved'], 1)
        # local z centres 4.9 / 5.1 cm straddle a round(z / 2 cm) boundary; within tolerance they are one plane
        objs = [tile(0, 0, 0, z=-0.1), tile(1, 400, 0, z=0.1)]
        self.assertEqual(con.plan(objs, CAT)['saved'], 1)
        objs = [tile(0, 0, 0, z=0), tile(1, 400, 0, z=3)]
        self.assertEqual(con.plan(objs, CAT)['saved'], 0)

    def test_manual_candidate_reasons(self):
        objs = [tile(i, 0, 0, row='Prop') for i in range(3)] + [tile(9, 0, 0, row='Ghost')]
        reasons = {c['row']: c['not_merged_because'] for c in con.plan(objs, CAT)['manual_candidates']}
        self.assertIn('not triplanar', reasons['Prop'])
        self.assertEqual(reasons['Ghost'], 'unknown row (not in catalogue)')
        objs = [tile(0, 0, 0), tile(1, 0, 0, row='Slope')]
        reasons = {c['row']: c['not_merged_because'] for c in con.plan(objs, CAT, rows=['Slope'])['manual_candidates']}
        self.assertEqual(reasons['Tile'], 'excluded by rows filter')


class ShapeTests(unittest.TestCase):
    def test_box_hull_accepted_any_winding(self):
        v, t = box_mesh()
        ok, why = con.hull_is_box(v, t, (50, 100, 25), (50, 100, 25))
        self.assertTrue(ok, why)
        ok, _ = con.hull_is_box(v, [(a, c, b) for a, b, c in t], (50, 100, 25), (50, 100, 25))
        self.assertTrue(ok)

    def test_slope_open_face_and_bounds_mismatch_rejected(self):
        v, t = box_mesh(top_slope=True)
        self.assertFalse(con.hull_is_box(v, t, (50, 100, 25), (50, 100, 25))[0])
        v, t = box_mesh()
        self.assertFalse(con.hull_is_box(v, t[:-2], (50, 100, 25), (50, 100, 25))[0])
        self.assertFalse(con.hull_is_box(v, t, (50, 100, 30), (50, 100, 25))[0])

    def test_real_catalogue_ground_tiles_are_boxes_and_slopes_are_not(self):
        cat = load_catalog()
        shapes = con.row_shapes(cat, ['NewRow_182', 'NewRow_305', 'NewRow_238', 'NewRow_192', 'NewRow_1'])
        if not shapes['NewRow_182'][0] and shapes['NewRow_182'][1] == 'no measured mesh hull':
            self.skipTest('meshes.bin not available')
        self.assertTrue(shapes['NewRow_182'][0])   # Asphalt ground tile
        self.assertTrue(shapes['NewRow_305'][0])   # Concrete Wall01
        self.assertFalse(shapes['NewRow_238'][0])  # 3Coin
        self.assertFalse(shapes['NewRow_192'][0])  # Concrete Wall02 (openings)
        self.assertFalse(shapes['NewRow_1'][0])    # container, not triplanar


class SourceTests(unittest.TestCase):
    def test_objects_from_saved_map(self):
        data = {'S_AssetsTransform': [{'Rname': 'Tile', 'UniqueGroupTag': ['yard', 'None'], 'Transform': [
            {'Rotation': {'X': 0, 'Y': 0, 'Z': 0.7071068, 'W': 0.7071068}, 'Translation': {'X': 100, 'Y': 2, 'Z': 3},
             'Scale3D': {'X': 2, 'Y': 1, 'Z': 1}},
            {'Rotation': {'X': 0, 'Y': 0, 'Z': 0, 'W': 1}, 'Translation': {'X': 0, 'Y': 0, 'Z': 0}, 'Scale3D': {'X': 1, 'Y': 1, 'Z': 1}}]}]}
        objs = con.objects_from_map(data)
        self.assertEqual([o['group'] for o in objs], ['yard', '-'])
        self.assertAlmostEqual(objs[0]['xf'][4], 90, places=4)
        self.assertEqual(objs[0]['xf'][6], 2)

    def test_saved_map_is_preview_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm.json'
            path.write_text(json.dumps({'S_AssetsTransform': []}))
            with self.assertRaisesRegex(ValueError, '^a saved map file is preview-only; load_live it, then apply on '
                                                    'the open editor without a name$'):
                con.consolidate_scenery(CAT, name=str(path), preview=False)
            with patch.object(con, 'REPORT_DIR', Path(tmp)):
                self.assertEqual(con.consolidate_scenery(CAT, name=str(path))['considered'], 0)


if __name__ == '__main__':
    unittest.main()
