import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from uncrashed_mcp import live

POSE = dict(x=1,y=2,z=3,pitch=-20,yaw=90,roll=0,fov=90,held=False)


def setUpModule():
    # agent.session() must not take the production lock from a test.
    tmp = tempfile.TemporaryDirectory()
    unittest.addModuleCleanup(tmp.cleanup)
    lock = patch.object(live.agent, 'LOCK', Path(tmp.name)/'agent.lock')
    lock.start()
    unittest.addModuleCleanup(lock.stop)
    # no textured reference loaded: capture_view's one status call answers without the agent
    none = patch.object(live.reference, 'status', return_value={'available': True, 'loaded': False, 'hidden': False})
    none.start()
    unittest.addModuleCleanup(none.stop)


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

    def test_read_rounds_float_noise(self):
        self.assertNotEqual(float('1234.57')/100,12.3457)
        with patch.object(live.agent,'run',return_value=['camera loc=1234.57000,-27000.30000,0.12345 rot=-20.12346,90.00004,0.00000 fov=80.0000 held=0']):
            p=live.read_camera()
        self.assertEqual([p[k] for k in ('x','y','z','pitch','yaw','roll')],[12.3457,-270.003,.0012,-20.1235,90.0,0.0])

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

    def test_failing_stop_does_not_replace_the_error_already_on_its_way_out(self):
        sent=live.agent.AgentTimeout("the agent did not answer 'capture render 42' within 30s; the command was sent and may still run")
        for stop in (live.agent.AgentNotSent('agent busy: an earlier job is still running; nothing was sent'),RuntimeError('stop broke')):
            def run(command,**kwargs):
                if command.startswith('capture start'):
                    return ['capture token=42 width=64 height=64 backend=engine']
                if command.startswith('capture render'):
                    raise sent
                raise stop
            with self.subTest(stop=stop),patch.object(live.agent,'run',side_effect=run) as call,patch.object(live.time,'sleep'):
                with self.assertRaises(live.agent.AgentError) as caught:
                    live._engine_capture_locked(Path('/tmp/test.png'),POSE,64,64)
                self.assertIs(caught.exception,sent)
                self.assertEqual([c.args[0] for c in call.call_args_list][1:],['capture render 42','capture stop 42'])

    def test_failing_stop_after_a_good_capture_is_still_reported(self):
        def run(command,**kwargs):
            if command.startswith('capture start'):
                return ['capture token=42 width=64 height=64 backend=engine']
            if command.startswith('capture stop'):
                raise live.agent.AgentError('no capture job 42')
            return []
        with patch.object(live.agent,'run',side_effect=run),patch.object(live.time,'sleep'),patch.object(live,'_validate_image'):
            with self.assertRaisesRegex(live.agent.AgentError,'no capture job 42'):
                live._engine_capture_locked(Path('/tmp/test.png'),POSE,64,64)

    def test_busy_capture_retries_start_and_never_stops_someone_elses_job(self):
        clock=[0.]
        def sleep(seconds):clock[0]+=seconds
        busy=live.agent.AgentError('capture busy; finish or stop existing job')
        with patch.object(live.agent,'run',side_effect=busy) as call,patch.object(live.time,'sleep',side_effect=sleep),patch.object(live.time,'monotonic',side_effect=lambda:clock[0]):
            with self.assertRaisesRegex(live.agent.AgentError,'busy'):
                live._engine_capture(Path('/tmp/test.png'),POSE,800,450)
            self.assertEqual(clock[0],32)
            self.assertEqual({c.args[0].split()[1] for c in call.call_args_list},{'start'})

    def test_busy_capture_starts_once_the_native_watchdog_freed_the_slot(self):
        replies=[live.agent.AgentError('capture busy; finish or stop existing job')]*3+[['capture token=7 width=64 height=64 backend=engine']]
        def run(command,**kwargs):
            if command.startswith('capture start'):
                reply=replies.pop(0)
                if isinstance(reply,Exception):raise reply
                return reply
            return []
        with patch.object(live.agent,'run',side_effect=run) as call,patch.object(live.time,'sleep'),patch.object(live,'_validate_image'):
            live._engine_capture_locked(Path('/tmp/test.png'),POSE,64,64)
        self.assertEqual(replies,[])
        self.assertEqual(call.call_args.args[0],'capture stop 7')

    def test_other_capture_start_errors_are_not_retried(self):
        with patch.object(live.agent,'run',side_effect=live.agent.AgentError('no editor world')) as call,patch.object(live.time,'sleep') as sleep:
            with self.assertRaisesRegex(live.agent.AgentError,'no editor world'):
                live._engine_capture_locked(Path('/tmp/test.png'),POSE,64,64)
            self.assertEqual(call.call_count,1);sleep.assert_not_called()

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
