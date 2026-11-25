"""Offline tests of the textured OBJ reference: native/ucmesh index + pack, auto datum, tmpfs pack lifetime,
memory floor, partial loads, capture hiding and old-DLL tolerance. The agent is always faked."""
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from uncrashed_mcp import agent, api, live, reference as r, sheet

BINARY = r.TOOL


def native_ok() -> bool:
    if not (BINARY.is_file() and os.access(BINARY, os.X_OK)):
        return False
    done = subprocess.run([str(BINARY)], capture_output=True, text=True)
    return 'usage: ucmesh' in done.stderr


NATIVE = native_ok()
_guards = []


def setUpModule():
    # Nothing here may reach the game: the agent is faked per test, and these fail loudly otherwise.
    for name in ('ensure_agent', '_execute'):
        guard = patch.object(agent, name, side_effect=AssertionError(f'agent.{name} reached from a test'))
        guard.start()
        _guards.append(guard)
    guard = patch.object(agent, 'session', contextlib.nullcontext)
    guard.start()
    _guards.append(guard)


def tearDownModule():
    while _guards:
        _guards.pop().stop()


def grid(lines, name, x0, z0, size, n, height, base):
    """An n x n vertex grid at source height `height` (y up); returns the next vertex number."""
    lines.append(f'o {name}')
    for j in range(n):
        for i in range(n):
            lines.append(f'v {x0 + size * i / (n - 1)} {height} {z0 + size * j / (n - 1)}')
            lines.append(f'vt {i / (n - 1)} {j / (n - 1)}')
    lines.append('usemtl paint')
    for j in range(n - 1):
        for i in range(n - 1):
            a = base + j * n + i
            b, c, d = a + 1, a + n, a + n + 1
            lines.append(f'f {a}/{a} {c}/{c} {b}/{b}')
            lines.append(f'f {b}/{b} {c}/{c} {d}/{d}')
    return base + n * n


def write_scene(root: Path) -> Path:
    """Ground at source y=10 (with a house on it and one deep spike), a second patch at y=12 300 m east."""
    Image.new('RGB', (64, 32), (200, 40, 40)).save(root / 'paint.jpg', quality=90)
    (root / 'scene.mtl').write_text('newmtl paint\nmap_Kd paint.jpg\n')
    lines = ['mtllib scene.mtl']
    nxt = grid(lines, 'ground', 0, 0, 50, 6, 10, 1)
    lines.append('o house')
    house = [(20, 10, 20), (30, 10, 20), (30, 10, 30), (20, 10, 30), (20, 30, 20), (30, 30, 20), (30, 30, 30), (20, 30, 30)]
    for x, y, z in house:
        lines += [f'v {x} {y} {z}', 'vt 0.5 0.5']
    lines.append('usemtl paint')
    for a, b, c in ((0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7), (4, 5, 6), (4, 6, 7)):
        lines.append(f'f {nxt + a}/{nxt + a} {nxt + b}/{nxt + b} {nxt + c}/{nxt + c}')
    nxt += 8
    lines += ['o spike', 'v 5 -40 5', 'v 6 10 5', 'v 5 10 6', 'vt 0 0', 'vt 1 0', 'vt 0 1', 'usemtl paint',
              f'f {nxt}/{nxt} {nxt + 2}/{nxt + 2} {nxt + 1}/{nxt + 1}']
    nxt += 3
    grid(lines, 'far', 300, 0, 20, 3, 12, nxt)
    obj = root / 'scene.obj'
    obj.write_text('\n'.join(lines) + '\n')
    return obj


class FakeAgent:
    """agent.run stand-in that keeps a reference state like native/reference.h and records every command."""

    def __init__(self, test, fail_add_at=None, verb=True, packs=None):
        self.test, self.fail_add_at, self.verb, self.packs = test, fail_add_at, verb, packs
        self.calls, self.chunks, self.hidden, self.seen = [], 0, False, []

    def status_line(self):
        if not self.chunks:
            return 'reference empty'
        return (f'reference loaded chunks={self.chunks} sections={self.chunks} triangles={100 * self.chunks} '
                f'vertices={60 * self.chunks} textures={self.chunks} texels={4096 * self.chunks} hidden={int(self.hidden)} '
                'pose=100.0,200.0,300.0,0.000 collision=off saved=no')

    def __call__(self, command, timeout=30):
        self.calls.append(command)
        if not self.verb and command.startswith('reference'):
            error = agent.AgentError('unknown verb reference')
            error.reply = {'ok': False}
            raise error
        if command == 'reference status':
            return [self.status_line()]
        if command == 'reference clear':
            self.chunks, self.hidden = 0, False
            return ['reference cleared']
        if command in ('reference hide', 'reference show'):
            self.hidden = command.endswith('hide')
            return ['reference ' + ('hidden' if self.hidden else 'shown')]
        if command.startswith('reference add '):
            words = command.split(' ', 6)
            chunk = Path(words[6][2:].replace('\\', '/'))
            # the DLL reads the chunk and its atlas pages now: they must exist, in the pack folder
            self.test.assertTrue(chunk.is_file(), chunk)
            pages = sorted(chunk.parent.glob(chunk.stem + '_p*.jpg'))
            self.test.assertTrue(pages)
            if self.packs is not None:
                self.test.assertEqual(chunk.parent.parent, self.packs)
            self.seen.append((chunk, pages, [float(v) for v in words[2:6]]))
            if self.fail_add_at is not None and len(self.seen) == self.fail_add_at:
                error = agent.AgentError('mesh section 0 failed; chunk rolled back')
                error.reply = {'ok': False}
                raise error
            self.chunks += 1
            return [f'reference added sections=1 triangles=100 texels=4096 total_chunks={self.chunks}']
        raise AssertionError(f'unexpected agent command {command!r}')


class Sandbox(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.shm = self.root / 'shm'
        self.shm.mkdir()
        for name, value in (('CACHE', self.root / 'cache'), ('SHM', self.shm), ('PACKS', self.shm / 'uncrashed-reference'),
                            ('SCRATCH_PACKS', self.root / 'scratch')):
            p = patch.object(r, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(r, 'tool', return_value=BINARY)
        p.start()
        self.addCleanup(p.stop)
        (self.root / 'src').mkdir()
        self.obj = write_scene(self.root / 'src')

    def fake(self, **kwargs):
        fake = FakeAgent(self, **kwargs)
        for target, value in (('run', fake), ('map_info', lambda: {'tag': 'MapEditor'})):
            p = patch.object(agent, target, value)
            p.start()
            self.addCleanup(p.stop)
        return fake


@unittest.skipUnless(NATIVE, 'native/ucmesh is not built (g++ -O2 -std=c++17 native/ucmesh.cpp -lturbojpeg -o native/ucmesh)')
class NativePipelineTests(Sandbox):
    def test_index_holds_no_pixels_and_prunes_older_versions(self):
        cache = r.ensure_index(self.obj)
        self.assertEqual(sorted(p.name for p in cache.iterdir()), ['index.bin', 'tris.bin', 'verts.bin'])
        textures, meshes = r.read_index(cache)
        self.assertEqual([m['name'] for m in meshes], ['ground', 'house', 'spike', 'far'])
        self.assertEqual(textures, [str((self.root / 'src' / 'paint.jpg').resolve())])
        self.assertEqual((cache / 'verts.bin').stat().st_size, 20 * sum(m['nv'] for m in meshes))
        built = (cache / 'index.bin').stat().st_mtime_ns
        self.assertEqual(r.ensure_index(self.obj), cache)
        self.assertEqual((cache / 'index.bin').stat().st_mtime_ns, built)  # cached, not rebuilt
        os.utime(self.obj, ns=(1, 1))
        newer = r.ensure_index(self.obj)
        self.assertNotEqual(newer, cache)
        self.assertFalse(cache.exists())
        self.assertEqual([p for p in newer.parent.iterdir() if p.is_dir()], [newer])

    def test_auto_datum_is_the_ground_of_the_region(self):
        cache = r.ensure_index(self.obj)
        _, meshes = r.read_index(cache)
        axes = r.source_axes(self.obj)
        self.assertEqual(axes, 'east,up,south')
        self.assertEqual(r.auto_datum(cache, r.select(meshes, axes), axes), 10.0)  # the -40 m spike is an outlier
        far = r.select(meshes, axes, (310, 10), 30)
        self.assertEqual([m['name'] for m in far], ['far'])
        self.assertEqual(r.auto_datum(cache, far, axes, (310, 10), 30), 12.0)
        with patch.dict(sys.modules, {'numpy': None}):  # the fallback without numpy agrees
            self.assertEqual(r.auto_datum(cache, r.select(meshes, axes), axes), 10.0)
            self.assertEqual(r.auto_datum(cache, far, axes, (310, 10), 30), 12.0)

    def test_load_packs_in_tmpfs_and_deletes_every_file_after_import(self):
        fake = self.fake(packs=self.shm / 'uncrashed-reference')
        result = api.call('reference_load', str(self.obj), at=[1, 2, 3], yaw=0)
        self.assertEqual((result['loaded_chunks'], result['chunks'], result['stopped']), (2, 2, ''))
        self.assertIsNone(api.OPERATIONS['reference_load'].check(result))
        self.assertEqual((result['datum_m'], result['pack'], result['pack_deleted']), (10.0, 'tmpfs', True))
        self.assertIn('auto', result['datum'])
        self.assertEqual(result['triangles'], result['triangles_total'])
        self.assertEqual(result['est_game_gib'], result['est_game_gib_all'])  # all loaded (tiny: rounds to 0)
        self.assertEqual(result['warnings'], [])
        self.assertTrue(result['status']['loaded'])
        # nearest chunk first, all at the export origin pose in centimetres
        self.assertEqual([pose for _, _, pose in fake.seen], [[100.0, 200.0, 300.0, 0.0]] * 2)
        self.assertIn('+000_+000', fake.seen[0][0].name)
        for chunk, pages, _ in fake.seen:
            self.assertFalse(chunk.exists())
            self.assertFalse(any(p.exists() for p in pages))
        self.assertEqual(list((self.shm / 'uncrashed-reference').iterdir()), [])
        self.assertFalse((self.root / 'scratch').exists())
        # nothing with pixels stays on disk: the cache holds the index only
        self.assertEqual([p for p in (self.root / 'cache').rglob('*') if p.suffix == '.jpg'], [])
        # the load is recorded for scan_place --from_reference, which accepts it while the editor shows that pose
        last = r.last_load()
        self.assertEqual({k: last[k] for k in ('obj', 'at', 'yaw', 'datum_m', 'pose_m')},
                         {'obj': str(self.obj.resolve()), 'at': [1.0, 2.0, 3.0], 'yaw': 0.0, 'datum_m': 10.0,
                          'pose_m': [1.0, 2.0, 3.0]})
        from uncrashed_mcp.scan import ops
        self.assertEqual(ops._from_reference(str(self.obj)),
                         {'at': [1.0, 2.0, 3.0], 'yaw': 0.0, 'datum_m': 10.0, 'source': 'reference_load'})

    def test_datum_given_as_a_number_lands_at_z(self):
        self.fake()
        result = r.load(str(self.obj), at=[0, 0, 0], datum_m='12.5', center=[310, 10], radius_m=30)
        self.assertEqual((result['datum_m'], result['datum'], result['chunks']), (12.5, 'given 12.5 m', 1))

    def test_full_tmpfs_falls_back_to_the_scratch_cache_with_a_warning(self):
        with patch.object(r, 'SHM', self.root / 'no-shm'):
            self.fake(packs=self.root / 'scratch')
            result = r.load(str(self.obj), at=[0, 0, 0])
        self.assertEqual(result['pack'], 'disk')
        self.assertIn('is missing', result['warnings'][0])
        self.assertEqual(list((self.root / 'scratch').iterdir()), [])

    def test_memory_floor_stops_before_a_chunk_and_reports_it(self):
        fake = self.fake()
        with patch.object(r, '_available_gb', return_value=3.0):
            result = api.call('reference_load', str(self.obj), at=[0, 0, 0])
        self.assertEqual((result['loaded_chunks'], result['chunks'], fake.seen), (0, 2, []))
        self.assertIn('memory floor: stopped before chunk 1/2', result['stopped'])
        self.assertEqual(api.OPERATIONS['reference_load'].check(result)[:31], 'reference loaded 0/2 chunks: me')
        self.assertTrue(result['pack_deleted'])
        self.assertEqual(list((self.shm / 'uncrashed-reference').iterdir()), [])

    def test_failure_midway_names_the_loaded_chunks_and_the_cleanup(self):
        self.fake(fail_add_at=2)
        with self.assertRaises(r.PartialLoad) as caught:
            r.load(str(self.obj), at=[0, 0, 0])
        message = str(caught.exception)
        self.assertIn('reference chunk 2/2 failed: mesh section 0 failed; chunk rolled back', message)
        self.assertIn('1 of 2 chunks are loaded and stay in the editor', message)
        self.assertIn('reference_clear removes them', message)
        self.assertEqual(list((self.shm / 'uncrashed-reference').iterdir()), [])

    def test_second_load_needs_replace(self):
        fake = self.fake()
        r.load(str(self.obj), at=[0, 0, 0])
        with self.assertRaisesRegex(r.ReferenceError, 'already loaded .2 chunks.: pass replace'):
            r.load(str(self.obj), at=[0, 0, 0])
        r.load(str(self.obj), at=[0, 0, 0], replace=True)
        self.assertEqual(fake.calls.count('reference clear'), 1)
        self.assertEqual(fake.chunks, 2)


class ValidationTests(Sandbox):
    def test_bad_input_is_refused_before_the_agent(self):
        with patch.object(agent, 'run', side_effect=AssertionError('agent reached')):
            for kwargs, needle in ((dict(), 'exactly one of at'), (dict(at=[0, 0]), 'at is x,y,z'),
                                   (dict(at=[0, 0, 0], datum_m='ground'), "datum_m is 'auto' or a number"),
                                   (dict(at=[0, 0, 0], far_scale=9), 'far_scale'),
                                   (dict(at=[0, 0, 0], radius_m=-1), '>= 0'),
                                   (dict(map_origin=[51, 12]), 'metadata.json'),
                                   (dict(at=[0, 0, 0], axes='east,east,up'), 'axes must name')):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(r.ReferenceError, needle):
                    r.load(str(self.obj), **kwargs)
            with self.assertRaisesRegex(r.ReferenceError, 'no such OBJ'):
                r.load(str(self.root / 'missing.obj'), at=[0, 0, 0])

    def test_clear_removes_leftover_packs_but_not_a_live_load(self):
        fake = self.fake()
        fake.chunks = 3
        packs = self.shm / 'uncrashed-reference'
        dead, alive, scratch = packs / 'dead', packs / 'alive', self.root / 'scratch' / 'old'
        for folder, pid in ((dead, 999_999_999), (alive, os.getppid()), (scratch, 0)):
            folder.mkdir(parents=True)
            (folder / 'chunk_+000_+000_p0.jpg').write_bytes(b'x')
            (folder / '.owner').write_text(f'{pid}\n')
        self.assertEqual(api.call('reference_status')['leftover_packs'], 3)
        fake.calls.clear()
        result = api.call('reference_clear')
        self.assertEqual(result, {'cleared': True, 'chunks_removed': 3, 'packs_removed': 2})
        self.assertEqual((dead.exists(), alive.exists(), scratch.exists()), (False, True, False))
        self.assertEqual(fake.calls, ['reference status', 'reference clear'])

    def test_status_and_visibility(self):
        fake = self.fake()
        self.assertEqual(r.status(), {'available': True, 'loaded': False, 'hidden': False})
        with self.assertRaisesRegex(r.ReferenceError, 'no reference loaded'):
            api.call('reference_visibility')
        fake.chunks = 2
        status = r.status()
        self.assertEqual((status['chunks'], status['triangles'], status['pose_m'], status['hidden']), (2, 200, [1.0, 2.0, 3.0], False))
        self.assertEqual(api.call('reference_visibility'), {'hidden': True, 'chunks': 2})
        self.assertTrue(r.status()['hidden'])
        self.assertEqual(api.call('reference_visibility', hidden=False), {'hidden': False, 'chunks': 2})


class OldDllTests(Sandbox):
    def test_a_dll_without_the_verb_means_no_reference(self):
        fake = self.fake(verb=False)
        self.assertFalse(r.status()['available'])
        self.assertEqual(r.drop_for_scene_change(), '')
        with r.capture_guard() as shown:
            pass
        self.assertEqual(shown, {'reference': 'none'})
        self.assertEqual(api.call('reference_clear')['cleared'], False)
        with self.assertRaisesRegex(r.ReferenceError, 'no reference verb'):
            r.load(str(self.obj), at=[0, 0, 0])
        self.assertTrue(all(c == 'reference status' for c in fake.calls))

    def test_other_agent_errors_are_not_swallowed(self):
        with patch.object(agent, 'run', side_effect=agent.AgentNotSent('offline')):
            with self.assertRaises(agent.AgentNotSent):
                r.status()
        with patch.object(agent, 'run', side_effect=agent.AgentTimeout('unknown verb reference?')):
            with self.assertRaises(agent.AgentTimeout):
                r.status()


class CaptureTests(Sandbox):
    def setUp(self):
        super().setUp()
        self.events = []
        self.fake_agent = self.fake()
        self.fake_agent.calls = self.events  # one ordered record for agent commands and captures
        self.fail = False

    def capture(self, path, pose, width, height):
        self.events.append('capture')
        if self.fail:
            raise RuntimeError('render unavailable')
        Image.new('RGB', (width, height), 'white').save(path)

    def run_capture(self, **kwargs):
        self.fail = kwargs.pop('fail', False)
        with patch.object(live, 'SHOTS', self.root / 'shots'), \
                patch.object(live, 'read_camera', return_value=dict(x=0, y=0, z=0, pitch=0, yaw=0, roll=0, fov=90, held=False)), \
                patch.object(live, '_engine_capture', self.capture):
            path = live.capture_view(**kwargs)
        return json.loads(path.with_suffix('.json').read_text())

    def test_a_visible_reference_is_hidden_for_the_capture_only(self):
        self.fake_agent.chunks = 2
        meta = self.run_capture()
        self.assertEqual(self.events, ['reference status', 'reference hide', 'capture', 'reference show'])
        self.assertEqual(meta['reference'], 'hidden for capture')
        self.assertFalse(self.fake_agent.hidden)

    def test_include_reference_skips_even_the_status_call(self):
        self.fake_agent.chunks = 2
        self.assertEqual(self.run_capture(include_reference=True)['reference'], 'included')
        self.assertEqual(self.events, ['capture'])

    def test_no_or_hidden_reference_is_left_alone(self):
        self.assertEqual(self.run_capture()['reference'], 'none')
        self.fake_agent.chunks, self.fake_agent.hidden = 1, True
        self.assertEqual(self.run_capture()['reference'], 'already hidden')
        self.assertEqual(self.events, ['reference status', 'capture', 'reference status', 'capture'])
        self.assertTrue(self.fake_agent.hidden)

    def test_a_failed_capture_still_shows_the_reference_again(self):
        self.fake_agent.chunks = 1
        with self.assertRaisesRegex(RuntimeError, 'render unavailable'):
            self.run_capture(fail=True)
        self.assertEqual(self.events, ['reference status', 'reference hide', 'capture', 'reference show'])
        self.assertFalse(self.fake_agent.hidden)

    def test_a_failed_show_is_reported_not_raised(self):
        self.fake_agent.chunks = 1
        real = self.fake_agent.__call__

        def run(command, timeout=30):
            if command == 'reference show':
                raise agent.AgentError('no reference loaded')
            return real(command, timeout)
        with patch.object(agent, 'run', run):
            meta = self.run_capture()
        self.assertIn('stays hidden', meta['reference_error'])
        self.assertIn('reference_visibility --no-hidden', meta['reference_error'])

    def test_a_sheet_checks_once_and_hides_around_every_pose(self):
        self.fake_agent.chunks = 1
        self.fail = False
        with patch.object(sheet, 'SHEETS', self.root / 'sheets'), \
                patch.object(live, 'read_camera', return_value=dict(x=0, y=0, z=0, pitch=0, yaw=0, roll=0, fov=90, held=False)), \
                patch.object(live, '_engine_capture', self.capture):
            result = api.call('capture_sheet', 'a 0 0 5 0 0; b 1 0 5 0 0; c 2 0 5 0 0', width=64, height=64,
                              out=str(self.root / 'out'))
        self.assertEqual(self.events, ['reference status', 'reference hide', 'capture', 'capture', 'capture', 'reference show'])
        self.assertEqual(result['reference'], 'hidden for capture')


class SceneChangeTests(Sandbox):
    def test_load_live_clears_a_loaded_reference_first(self):
        fake = self.fake()
        fake.chunks = 4
        with tempfile.TemporaryDirectory() as tmp, patch.object(api, 'maps_dir', return_value=Path(tmp)), \
                patch.object(agent, 'load', lambda name, move: fake.calls.append('load') or {'loaded': name}):
            (Path(tmp) / 'M').mkdir()
            (Path(tmp) / 'M' / 'M.json').write_text('{}')
            result = api.call('load_live', 'M')
            self.assertEqual(fake.calls, ['reference status', 'reference clear', 'load'])
            self.assertEqual(result['loaded'], 'M')
            self.assertIn('cleared the textured reference (4 chunks)', result['reference'])
            fake.calls.clear()
            self.assertNotIn('reference', api.call('load_live', 'M'))
            self.assertEqual(fake.calls, ['reference status', 'load'])


if __name__ == '__main__':
    unittest.main()
