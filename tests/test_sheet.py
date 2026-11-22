import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from uncrashed_mcp import agent, api, live, sheet

_guards = []


def setUpModule():
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


class ParseTests(unittest.TestCase):
    def test_items_labels_and_separators(self):
        poses = sheet.parse_poses('start 0 -20 6 90 -10; gate/1 30 0 8 0 -5\n 1 2 3 4 5 # unlabelled\nb 1 1 1 0 0')
        self.assertEqual([p['label'] for p in poses], ['start', 'gate_1', 'p3', 'b'])
        self.assertEqual(poses[0], {'label': 'start', 'x': 0.0, 'y': -20.0, 'z': 6.0, 'yaw': 90.0, 'pitch': -10.0})

    def test_comments_are_stripped_per_line_before_splitting(self):
        poses = sheet.parse_poses('a 0 0 5 0 0 # note, with a comma; and 1 2 3\nb 1 1 1 0 0')
        self.assertEqual([p['label'] for p in poses], ['a', 'b'])

    def test_file_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'poses.txt'
            path.write_text('# route\na 0 0 5 0 0\na 1 0 5 0 0\n\n')
            self.assertEqual([p['label'] for p in sheet.parse_poses(f'@{path}')], ['a', 'a_2'])
            with self.assertRaisesRegex(ValueError, r"^poses: cannot read '.*missing\.txt'"):
                sheet.parse_poses(f'@{tmp}/missing.txt')

    def test_refusals(self):
        for text, message in (('', 'at least one'), ('a 1 2 3', 'must be "label x y z yaw pitch"'),
                              ('a 1 2 x 4 5', 'non-numeric'), ('a 1 2 nan 4 5', 'finite'),
                              (';'.join(['1 2 3 4 5'] * 61), 'at most 60 poses')):
            with self.assertRaisesRegex(ValueError, message):
                sheet.parse_poses(text)

    def test_sizes(self):
        self.assertEqual(sheet._validate(480, 270, 0, 5), 3)
        self.assertEqual(sheet._validate(480, 270, 8, 5), 5)
        for args in ((32, 270, 0, 1), (480, 2000, 0, 1), (480, 270, 13, 1), (480, 270, -1, 1)):
            with self.assertRaises(ValueError):
                sheet._validate(*args)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.out = root / 'out'
        patch.object(sheet, 'SHEETS', root / 'shots').start()
        patch.object(live, 'read_camera', return_value={'x': 0, 'y': 0, 'z': 0, 'pitch': 0, 'yaw': 0, 'roll': 0,
                                                       'fov': 90.0, 'held': False}).start()
        self.addCleanup(patch.stopall)
        self.calls = []

    def fake_capture(self, fail=()):
        def capture(path, pose, width, height):
            self.calls.append((pose, width, height))
            if pose['label'] in fail:
                raise live.LiveError('capture is uniform/blank;\n rendering may be suspended')
            Image.new('RGB', (width, height), (len(self.calls) * 40 % 256, 90, 160)).save(path)
        return capture

    def test_every_pose_in_one_sheet_with_originals_and_index(self):
        with patch.object(live, '_engine_capture', side_effect=self.fake_capture()):
            result = api.call('capture_sheet', 'a 0 0 5 90 -10; b 10 0 5 0 -20; c 20 0 5 0 0', width=96, height=64,
                              out=str(self.out))
        self.assertEqual((result['poses'], result['captured'], result['failed_count'], result['columns']), (3, 3, 0, 2))
        self.assertEqual([c[0]['fov'] for c in self.calls], [90.0] * 3)
        self.assertEqual(self.calls[1][0], {'label': 'b', 'x': 10.0, 'y': 0.0, 'z': 5.0, 'yaw': 0.0, 'pitch': -20.0,
                                            'roll': 0.0, 'fov': 90.0})
        self.assertEqual([s['image'] for s in result['shots']], ['01-a.png', '02-b.png', '03-c.png'])
        for name in ('01-a.png', '02-b.png', '03-c.png'):
            self.assertTrue((self.out / name).is_file())
        with Image.open(result['sheet']) as image:
            self.assertEqual(image.size, (2 * 96, 2 * (64 + sheet._strip(64))))
            # the caption strip of tile 1 carries drawn label pixels
            strip = image.crop((0, 0, 96, sheet._strip(64))).convert('L')
            self.assertGreater(strip.getextrema()[1], 200)
        index = json.loads(Path(result['index_json']).read_text())
        self.assertEqual(index['shots'][0]['image'], str(self.out / '01-a.png'))
        self.assertIsNone(api.OPERATIONS['capture_sheet'].check(result))
        self.assertFalse(list(sheet.SHEETS.glob('*.png')))  # nothing left behind in the shots folder

    def test_a_failed_pose_does_not_stop_the_others(self):
        with patch.object(live, '_engine_capture', side_effect=self.fake_capture(fail={'b'})):
            result = api.call('capture_sheet', 'a 0 0 5 0 0, b 1 0 5 0 0, c 2 0 5 0 0', width=64, height=64,
                              out=str(self.out))
        self.assertEqual((result['captured'], result['failed_count']), (2, 1))
        failed = result['shots'][1]
        self.assertEqual((failed['image'], failed['error']), (None, 'capture is uniform/blank; rendering may be suspended'))
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(api.OPERATIONS['capture_sheet'].check(result), '1 of 3 poses failed; see shots[].error')
        self.assertTrue(Path(result['sheet']).is_file())

    def test_not_sent_before_any_capture_is_raised_unchanged(self):
        error = agent.AgentNotSent('lock busy; nothing was sent')
        with patch.object(live, 'read_camera', side_effect=error), self.assertRaises(agent.AgentNotSent) as caught:
            api.call('capture_sheet', 'a 0 0 5 0 0', out=str(self.out))
        self.assertIs(caught.exception, error)
        self.assertFalse(self.out.exists())  # no empty folder when nothing was captured

    def test_not_sent_on_the_first_capture_leaves_no_folder(self):
        with patch.object(live, '_engine_capture', side_effect=agent.AgentNotSent('lock busy')), \
                self.assertRaises(agent.AgentNotSent):
            api.call('capture_sheet', 'a 0 0 5 0 0', out=str(self.out))
        self.assertFalse(self.out.exists())

    def test_non_empty_out_is_refused_before_the_agent(self):
        self.out.mkdir()
        (self.out / '01-old.png').write_bytes(b'stale')
        with patch.object(live, 'read_camera', side_effect=AssertionError('agent reached')), \
                self.assertRaisesRegex(ValueError, 'not empty'):
            api.call('capture_sheet', 'a 0 0 5 0 0', out=str(self.out))
        self.assertEqual([p.name for p in self.out.iterdir()], ['01-old.png'])

    def test_empty_out_folder_is_accepted(self):
        self.out.mkdir()
        with patch.object(live, '_engine_capture', side_effect=self.fake_capture()):
            result = api.call('capture_sheet', 'a 0 0 5 0 0', width=64, height=64, out=str(self.out))
        self.assertEqual(result['captured'], 1)

    def test_agent_lock_is_taken_per_pose(self):
        holds = []

        @contextlib.contextmanager
        def session():
            holds.append(len(self.calls))
            yield
        with patch.object(agent, 'session', session), \
                patch.object(live, '_engine_capture', side_effect=self.fake_capture()):
            api.call('capture_sheet', 'a 0 0 5 0 0; b 1 0 5 0 0; c 2 0 5 0 0', width=64, height=64, out=str(self.out))
        self.assertEqual(holds, [0, 0, 1, 2])  # one short hold for the FOV read, then one per pose

    def test_validation_runs_before_the_agent(self):
        with patch.object(live, 'read_camera', side_effect=AssertionError('agent reached')):
            with self.assertRaisesRegex(ValueError, 'width must be'):
                api.call('capture_sheet', 'a 0 0 5 0 0', width=5000)
            with self.assertRaisesRegex(ValueError, 'poses:'):
                api.call('capture_sheet', 'nonsense')
