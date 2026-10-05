import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from uncrashed_mcp import live

POSE = dict(x=1,y=2,z=3,pitch=-20,yaw=90,roll=0,fov=90,held=False)


def image(path, width, height):
    im=Image.new('RGB',(width,height),'white')
    im.paste('black',(0,0,width//2,height))
    im.save(path)


class LiveTests(unittest.TestCase):
    def test_native_camera_units_no_process_or_memory(self):
        with patch.object(live.agent,'run') as run,patch.object(live,'read_camera',return_value=POSE),patch.object(live.subprocess,'Popen') as popen:
            self.assertEqual(live.move_camera(1,2,3,40,-10),POSE)
            self.assertEqual(run.call_args.args[0],'camera set 100.00000 200.00000 300.00000 -10.00000 40.00000 1')
            popen.assert_not_called()

    def test_read_native(self):
        with patch.object(live.agent,'run',return_value=['camera loc=100,200,300 rot=-20,90,0 fov=80 held=1']):
            p=live.read_camera()
        self.assertEqual((p['x'],p['pitch'],p['fov'],p['held']),(1,-20,80,True))

    def test_release_native(self):
        with patch.object(live.agent,'run') as run:
            live.release_camera()
        run.assert_called_once_with('camera release')

    def test_engine_capture_never_touches_window_or_editor_pose(self):
        with tempfile.TemporaryDirectory() as d,patch.object(live,'SHOTS',Path(d)),patch.object(live,'read_camera',return_value=POSE.copy()),patch.object(live,'_engine_capture',side_effect=image_capture) as capture,patch.object(live,'_xdo',side_effect=AssertionError('focus query forbidden')),patch.object(live,'move_camera',side_effect=AssertionError('editor movement forbidden')):
            path=live.capture_view(10,20,30,45,-15)
            self.assertTrue(path.is_file())
            self.assertEqual(capture.call_args.args[1]['yaw'],45)
            self.assertEqual(capture.call_args.args[1]['x'],10)
            self.assertIn('"editor_camera_moved": false',path.with_suffix('.json').read_text())

    def test_engine_failure_never_foreground_fallback(self):
        with tempfile.TemporaryDirectory() as d,patch.object(live,'SHOTS',Path(d)),patch.object(live,'read_camera',return_value=POSE.copy()),patch.object(live,'_engine_capture',side_effect=RuntimeError('render unavailable')),patch.object(live,'_window_capture') as window,patch.object(live,'_xdo') as xdo:
            with self.assertRaisesRegex(RuntimeError,'render unavailable'):
                live.capture_view()
            window.assert_not_called();xdo.assert_not_called()

    def test_export_failure_releases_own_capture(self):
        def run(command,**kwargs):
            if command.startswith('capture start'):
                return ['capture token=42 width=800 height=450 backend=engine']
            if command.startswith('capture export'):
                raise RuntimeError('readback failed')
            return []
        with patch.object(live.agent,'run',side_effect=run) as call,patch.object(live.time,'sleep'):
            with self.assertRaisesRegex(RuntimeError,'readback failed'):
                live._engine_capture(Path('/tmp/test.png'),POSE,800,450)
            self.assertEqual(call.call_args.args[0],'capture stop 42')

    def test_busy_capture_does_not_stop_someone_elses_job(self):
        with patch.object(live.agent,'run',side_effect=RuntimeError('capture busy')) as call:
            with self.assertRaisesRegex(RuntimeError,'busy'):
                live._engine_capture(Path('/tmp/test.png'),POSE,800,450)
            self.assertEqual(call.call_count,1)

    def test_window_background_never_activates_or_queries_active_window(self):
        with tempfile.TemporaryDirectory() as d,patch.object(live,'game_window',return_value='42'),patch.object(live,'_xdo',side_effect=AssertionError('xdotool not allowed')),patch.object(live.subprocess,'run',side_effect=lambda argv,**kwargs:image(Path(argv[-1]),100,100)):
            live._window_capture(Path(d)/'s.png',100,100,False)

    def test_foreground_requires_explicit_opt_in_and_restores_on_error(self):
        with patch.object(live,'game_window',return_value='42'),patch.object(live,'_xdo',side_effect=['old','','']) as xdo,patch.object(live.subprocess,'run',side_effect=RuntimeError('failed')),patch.object(live.time,'sleep'):
            with self.assertRaisesRegex(RuntimeError,'failed'):
                live._window_capture(Path('/tmp/no.png'),100,100,True)
            self.assertEqual([c.args for c in xdo.call_args_list],[('getactivewindow',),('windowactivate','--sync','42'),('windowactivate','old')])

    def test_capture_argument_validation_before_native_calls(self):
        for kwargs in [dict(x=1),dict(yaw=5),dict(backend='bad'),dict(foreground=True),dict(backend='window',x=1,y=2,z=3),dict(width=5000),dict(width=4096,height=4096)]:
            with self.subTest(kwargs=kwargs),patch.object(live,'read_camera') as read:
                with self.assertRaises(live.LiveError):
                    live.capture_view(**kwargs)
                read.assert_not_called()

    def test_supersampling_respects_native_pixel_limit_and_cleans_raw(self):
        with tempfile.TemporaryDirectory() as d,patch.object(live,'_engine_capture_locked',side_effect=image_capture) as capture:
            path=Path(d)/'view.png';live._engine_capture(path,POSE,100,100)
            self.assertEqual(capture.call_args.args[2:],(200,200))
            with Image.open(path) as im:self.assertEqual(im.size,(100,100))
            self.assertEqual(list(Path(d).iterdir()),[path])
        self.assertEqual(live._render_size(4096,1024),(4096,1024))

    def test_reject_blank_frame(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'blank.png';Image.new('RGB',(100,100),'black').save(p)
            with self.assertRaisesRegex(live.LiveError,'uniform/blank'):
                live._validate_image(p,100,100,True)


def image_capture(path, pose, width, height):
    image(path,width,height)
