"""Process-level hardening of the agent client: kill-switch, lock, protocol files, busy guard.

Everything runs on a temp lock and temp protocol files. The injection path is stubbed in
every test that lifts the offline switch, so no test can reach a running game.
"""

from __future__ import annotations

import contextlib
import fcntl
import io
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from uncrashed_mcp import agent, groups, managed

SRC = Path(agent.__file__).resolve().parents[1]
ENSURE = agent.ensure_agent  # the real one; sandbox() replaces it

NEW_HOLDER = """
import sys
from uncrashed_mcp import agent
with agent.session():
    print('held', flush=True)
    sys.stdin.read()
"""

# What a server process running the pre-port agent module does.
OLD_HOLDER = """
import fcntl, sys
lock = open(sys.argv[1], 'a')
print('ready', flush=True)
fcntl.flock(lock, fcntl.LOCK_EX)
print('held', flush=True)
sys.stdin.read()
"""


@contextlib.contextmanager
def sandbox(offline: bool = False, **env: str):
    """Temp lock and protocol files; a loaded agent is simulated, injection is impossible."""
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        root = Path(tmp)
        for name in ('CMD', 'OUT', 'SEQ', 'BATCH', 'LOCK'):
            stack.enter_context(patch.object(agent, name, root / f'{name.lower()}.txt'))
        stack.enter_context(patch.dict(os.environ, env))
        if offline:
            os.environ['UNCRASHED_OFFLINE'] = '1'
        else:
            os.environ.pop('UNCRASHED_OFFLINE', None)
        stack.enter_context(patch.object(agent.subprocess, 'run', side_effect=AssertionError('injection attempted')))
        stack.enter_context(patch.object(agent, 'game_pid', return_value=4242))
        stack.enter_context(patch.object(agent, 'loaded', return_value=True))
        stack.enter_context(patch.object(agent, '_game_started', return_value=0.0))
        ensure = stack.enter_context(patch.object(agent, 'ensure_agent'))
        yield root, ensure


def spawn_holder(lock: Path, code: str = NEW_HOLDER, wait: str = 'held') -> subprocess.Popen:
    env = dict(os.environ, UNCRASHED_LOCK=str(lock), UNCRASHED_OFFLINE='1', PYTHONPATH=str(SRC))
    child = subprocess.Popen([sys.executable, '-c', code, str(lock), 'marker'], env=env,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    if child.stdout.readline().strip() != wait:
        child.kill()
        child.wait()
        raise AssertionError('lock holder did not start')
    return child


def stop(child: subprocess.Popen) -> None:
    if child.poll() is None:
        child.kill()
    child.wait()
    child.stdin.close()
    child.stdout.close()


def fake_agent(before=None, body: str = 'pong\nend {seq} ok\n') -> threading.Thread:
    """Answer the next command written to CMD the way the DLL does (reply renamed onto OUT)."""
    cmd, out = agent.CMD, agent.OUT

    def work() -> None:
        if before:
            before()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                seq, command = cmd.read_text().split(None, 1)
            except (OSError, ValueError):
                time.sleep(.005)
                continue
            reply = out.with_name('reply')
            reply.write_text(f'{seq} ok {command.split()[0]}\n' + body.format(seq=seq))
            reply.replace(out)
            return

    thread = threading.Thread(target=work)
    thread.start()
    return thread


class KillSwitchTests(unittest.TestCase):
    def test_ensure_agent_refuses_before_looking_for_the_game(self):
        with patch.dict(os.environ, {'UNCRASHED_OFFLINE': '1'}), \
                patch.object(agent, 'game_pid', side_effect=AssertionError('looked for the game')), \
                patch.object(agent.subprocess, 'run', side_effect=AssertionError('injection attempted')):
            with self.assertRaisesRegex(agent.AgentError, r'^offline mode \(UNCRASHED_OFFLINE set\): live agent calls are disabled$'):
                agent.ensure_agent()

    def test_execute_refuses_before_anything_is_written(self):
        with sandbox(offline=True) as (root, ensure):
            calls = [lambda: agent._execute('mapinfo'), lambda: agent._execute(''), lambda: agent.execute('mapinfo'),
                     lambda: agent.run('mapinfo'), lambda: agent.batch(['spawn A 0 0 0 0 0 0 1 1 1'])]
            # Not even a batch file that is removed again: a file in /tmp is shared with live clients.
            with patch.object(agent, '_write_batch', side_effect=AssertionError('batch file written offline')):
                for call in calls:
                    with self.assertRaisesRegex(agent.AgentNotSent, 'offline mode'):
                        call()
            ensure.assert_not_called()
            # No command, sequence number, reply or batch file.
            self.assertEqual([p.name for p in root.iterdir() if p != agent.LOCK], [])
            self.assertEqual(agent.batch_files(), [])

    def test_only_a_non_empty_value_switches_off(self):
        with patch.dict(os.environ, {'UNCRASHED_OFFLINE': ''}):
            agent._refuse_offline()
        with patch.dict(os.environ, {'UNCRASHED_OFFLINE': '0'}), self.assertRaises(agent.AgentError):
            agent._refuse_offline()

    def test_unload_reports_instead_of_contacting_the_agent(self):
        with sandbox(offline=True):
            self.assertEqual(agent.unload()['unloaded'], False)
            self.assertIn('offline mode', agent.unload()['error'])


class LockTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.lock = Path(tmp.name) / 'agent.lock'
        for item in (patch.object(agent, 'LOCK', self.lock), patch.dict(os.environ, {'UNCRASHED_LOCK_TIMEOUT': '0.3'})):
            item.start()
            self.addCleanup(item.stop)

    def hold(self) -> int:
        """Hold the lock on a second open file description, like another process would."""
        fd = os.open(self.lock, os.O_RDWR | os.O_CREAT, 0o666)
        fcntl.flock(fd, fcntl.LOCK_EX)
        self.addCleanup(os.close, fd)
        return fd

    def test_holder_info_is_written_in_place_and_cleared(self):
        self.lock.write_text('left over from a killed process')
        inode = self.lock.stat().st_ino
        with agent.session():
            first = agent.lock_holder()
            with agent.session():
                self.assertEqual(agent._lock_depth, 2)
                self.assertEqual(agent.lock_holder(), first)
            self.assertEqual(json.loads(self.lock.read_text()), first)
        self.assertEqual((first['pid'], first['argv']), (os.getpid(), sys.argv))
        self.assertAlmostEqual(first['since'], time.time(), delta=5)
        self.assertEqual((agent._lock_depth, self.lock.read_text(), self.lock.stat().st_ino), (0, '', inode))
        self.assertIsNone(agent.lock_holder())

    def test_holder_info_tolerates_any_content(self):
        gone = subprocess.Popen([sys.executable, '-c', 'pass'])
        gone.wait()
        self.assertIsNone(agent.lock_holder())  # no lock file yet
        for content in (b'', b'{', b'{"pid": ', b'null', b'[]', b'7', b'{"argv": ["x"]}', b'{"pid": "x"}', b'{"pid": null}',
                        b'{"pid": -1}', b'{"pid": 0}', b'\xff\xfe\x00', b'{"pid": %d, "argv": ["x"]}' % gone.pid,
                        b'{"pid": Infinity}', b'{"pid": 1e999}', b'{"pid": NaN}', b'[' * 200000, b'{"a":' * 50000,
                        b'{"pid": true}', b'{"pid": 1.9}', b'{"pid": 1.0}', b'{"pid": "%d"}' % os.getpid(),
                        b'{"pid": %s}' % (b'9' * 5000), b'{"pid": %d, "argv": ["%s"]}' % (os.getpid(), b'x' * 70000)):
            self.lock.write_bytes(content)
            with self.subTest(content=content[:40]):
                self.assertIsNone(agent.lock_holder())
                with agent.session():  # never refused or broken because of the content
                    self.assertEqual(agent.lock_holder()['pid'], os.getpid())
        for content in ('{"pid": %d}', '{"pid": %d, "argv": "x", "since": "y"}', '{"pid": %d, "since": true}'):
            self.lock.write_text(content % os.getpid())
            self.assertEqual(agent.lock_holder(), {'pid': os.getpid(), 'argv': [], 'since': None})

    def test_garbage_in_the_lock_file_does_not_stop_a_waiter(self):
        fd = self.hold()
        for content in (b'{"pid": Infinity}', b'[' * 200000):
            os.ftruncate(fd, 0)
            os.pwrite(fd, content, 0)
            with self.subTest(content=content[:20]):
                with self.assertRaisesRegex(agent.AgentNotSent, '^agent busy: lock held by another process for 0s; nothing was sent'):
                    with agent.session():
                        self.fail('entered a held lock')
                self.assertEqual(agent._lock_depth, 0)

    def test_holder_record_carries_the_extra_fields_of_the_client(self):
        with patch.object(agent, 'holder_extra', {'digest': 'abc', 'pid': 1, 'argv': ['spoof'], 'n': [1, 2]}):
            with agent.session():
                holder = agent.lock_holder()
                self.assertEqual(json.loads(self.lock.read_text()), holder)
        self.assertEqual((holder['pid'], holder['argv'], holder['digest'], holder['n']), (os.getpid(), sys.argv, 'abc', [1, 2]))
        self.assertEqual(sorted(holder), ['argv', 'digest', 'n', 'pid', 'since'])
        self.assertEqual(agent.holder_extra, {})
        with agent.session():
            self.assertEqual(sorted(agent.lock_holder()), ['argv', 'pid', 'since'])
        loop: dict = {}
        loop['self'] = loop
        with patch.object(agent, 'holder_extra', loop), agent.session():  # unusable extras never cost the record
            self.assertEqual(sorted(agent.lock_holder()), ['argv', 'pid', 'since'])

    def test_holder_is_shown_as_uc_when_started_through_the_cli_module(self):
        for argv, shown in ((['/repo/src/uncrashed_mcp/cli.py', 'save_live', 'City'], 'uc save_live City'),
                            (['/repo/src/uncrashed_mcp/__main__.py', 'batch', '-'], 'uc batch -'),
                            (['/opt/tools/uc', 'log'], 'uc log'), (['/x/mycli.py', 'a'], 'mycli.py a'), ([], '?')):
            self.assertEqual(agent._holder_text({'pid': 7, 'argv': argv, 'since': None}), f'pid 7 ({shown})')

    def test_second_process_is_refused_then_served(self):
        child = spawn_holder(self.lock)
        try:
            holder = agent.lock_holder()
            self.assertEqual((holder['pid'], holder['argv']), (child.pid, ['-c', str(self.lock), 'marker']))
            started = time.monotonic()
            with self.assertRaisesRegex(agent.AgentNotSent, rf'^agent busy: lock held by pid {child.pid} \(-c .* marker\) for \d+s; nothing was sent; '
                                                            rf'retry when that process has finished \(the lock frees itself when its holder exits: '
                                                            rf'never delete {self.lock}\)$'):
                with agent.session():
                    self.fail('entered a session while another process held the lock')
            self.assertLess(time.monotonic() - started, 5)
            self.assertEqual(agent._lock_depth, 0)
            child.stdin.close()  # the holder leaves its session and exits
            self.assertEqual(child.wait(10), 0)
            with agent.session():
                self.assertEqual(agent.lock_holder()['pid'], os.getpid())
        finally:
            stop(child)

    def test_killed_holder_frees_the_lock_and_its_stale_info_is_ignored(self):
        child = spawn_holder(self.lock)
        try:
            os.kill(child.pid, signal.SIGKILL)
            child.wait(10)
            self.assertEqual(json.loads(self.lock.read_text())['pid'], child.pid)
            self.assertIsNone(agent.lock_holder())
            with agent.session():
                self.assertEqual(agent.lock_holder()['pid'], os.getpid())
        finally:
            stop(child)

    def test_older_server_holding_the_lock_blocks_without_holder_info(self):
        child = spawn_holder(self.lock, OLD_HOLDER, wait='ready')
        try:
            self.assertEqual(child.stdout.readline().strip(), 'held')
            with self.assertRaisesRegex(agent.AgentNotSent, r'^agent busy: lock held by another process for \d+s; nothing was sent; retry when '):
                with agent.session():
                    self.fail('entered a session while an older process held the lock')
        finally:
            stop(child)

    def test_older_server_waiting_on_the_lock_is_kept_out(self):
        with agent.session():
            child = spawn_holder(self.lock, OLD_HOLDER, wait='ready')
            try:
                self.assertEqual(select.select([child.stdout], [], [], .3)[0], [])
            except BaseException:
                stop(child)
                raise
        try:
            self.assertEqual(select.select([child.stdout], [], [], 10)[0], [child.stdout])
            self.assertEqual(child.stdout.readline().strip(), 'held')
        finally:
            stop(child)

    def test_waiting_is_announced_after_two_seconds_then_every_thirty(self):
        fd = self.hold()
        os.write(fd, json.dumps({'pid': os.getpid(), 'argv': ['/opt/tools/uc', 'batch', 'plan.txt'], 'since': time.time() - 100}).encode())
        clock = [0.]

        def sleep(seconds):
            clock[0] += 1

        stderr = io.StringIO()
        with patch.dict(os.environ, {'UNCRASHED_LOCK_TIMEOUT': '65'}), patch.object(agent.time, 'sleep', side_effect=sleep), \
                patch.object(agent.time, 'monotonic', side_effect=lambda: clock[0]), contextlib.redirect_stderr(stderr):
            with self.assertRaisesRegex(agent.AgentNotSent, rf'^agent busy: lock held by pid {os.getpid()} \(uc batch plan\.txt\) for 1\d\ds; nothing was sent; '):
                with agent.session():
                    self.fail('entered a held lock')
        self.assertEqual(clock[0], 65)
        self.assertEqual(stderr.getvalue().splitlines(), [f'waiting for agent lock held by pid {os.getpid()} (uc batch plan.txt)'] * 3)

    def test_short_waits_stay_silent(self):
        fd = self.hold()
        stderr = io.StringIO()
        timer = threading.Timer(.15, fcntl.flock, (fd, fcntl.LOCK_UN))
        timer.start()
        with patch.dict(os.environ, {'UNCRASHED_LOCK_TIMEOUT': '10'}), contextlib.redirect_stderr(stderr):
            with agent.session():
                pass
        timer.join()
        self.assertEqual(stderr.getvalue(), '')

    def test_wait_hook_sees_the_holder_and_can_abort(self):
        fd = self.hold()
        os.write(fd, json.dumps({'pid': os.getpid(), 'argv': ['uc', 'batch', 'plan.txt'], 'since': 1.5}).encode())
        seen = []

        def hook(holder, waited):
            seen.append((holder, waited))
            if len(seen) == 2:
                raise RuntimeError('identical command already running')

        with patch.object(agent, 'on_lock_wait', hook), patch.dict(os.environ, {'UNCRASHED_LOCK_TIMEOUT': '10'}):
            with self.assertRaisesRegex(RuntimeError, 'identical command'):
                with agent.session():
                    self.fail('entered a held lock')
        self.assertEqual([holder for holder, _ in seen], [{'pid': os.getpid(), 'argv': ['uc', 'batch', 'plan.txt'], 'since': 1.5}] * 2)
        self.assertTrue(0 <= seen[0][1] <= seen[1][1] < 5)
        self.assertEqual(agent._lock_depth, 0)
        fcntl.flock(fd, fcntl.LOCK_UN)
        with agent.session():  # the aborted wait left nothing locked in this process
            pass

    def test_a_wait_given_up_by_the_hook_leaves_no_lock_or_descriptor(self):
        fd = self.hold()

        class Stop(BaseException):  # what a signal handler raises
            pass

        def descriptors():
            return sorted(os.listdir('/proc/self/fd'))

        before = descriptors()
        for error in (agent.AgentNotSent('interrupted while waiting; nothing was sent'), Stop()):
            with patch.object(agent, 'on_lock_wait', side_effect=error), patch.dict(os.environ, {'UNCRASHED_LOCK_TIMEOUT': '10'}):
                with self.assertRaises(type(error)):
                    with agent.session():
                        self.fail('entered a held lock')
            self.assertEqual((agent._lock_depth, descriptors()), (0, before))
            self.assertTrue(agent._protocol_lock.acquire(blocking=False))  # not left held for other threads either
            agent._protocol_lock.release()
        fcntl.flock(fd, fcntl.LOCK_UN)  # with the other holder gone the lock is free: no descriptor of ours kept it
        probe = os.open(self.lock, os.O_RDWR)
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(probe)

    def test_closed_stderr_does_not_abort_the_wait(self):
        class Gone(io.StringIO):
            def write(self, text):
                raise BrokenPipeError(32, 'Broken pipe')

        self.hold()
        clock = [0.]

        def sleep(seconds):
            clock[0] += 1

        with patch.dict(os.environ, {'UNCRASHED_LOCK_TIMEOUT': '40'}), patch.object(agent.time, 'sleep', side_effect=sleep), \
                patch.object(agent.time, 'monotonic', side_effect=lambda: clock[0]), contextlib.redirect_stderr(Gone()):
            with self.assertRaisesRegex(agent.AgentNotSent, '^agent busy: lock held by'):  # not BrokenPipeError
                with agent.session():
                    self.fail('entered a held lock')
        self.assertEqual(clock[0], 40)

    def test_a_note_on_a_closed_stderr_pipe_keeps_the_exit_code(self):
        code = ('import sys\nfrom uncrashed_mcp import agent\nsys.stdin.readline()\n'
                'agent._note("waiting for agent lock held by pid 1 (x)")\nagent._note("again")\nsys.exit(7)\n')
        env = dict(os.environ, UNCRASHED_LOCK=str(self.lock), UNCRASHED_OFFLINE='1', PYTHONPATH=str(SRC))
        child = subprocess.Popen([sys.executable, '-c', code], env=env, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        child.stderr.close()  # the reader of `2>&1 | head -1` has left
        child.stdin.write('go\n')
        child.stdin.close()
        self.assertEqual(child.wait(20), 7)  # 120 when the unwritten note fails the exit-time flush

    def test_a_note_without_a_stderr_stream_does_not_go_to_stdout(self):
        # Started with fd 2 closed, sys.stderr is None and print(file=None) would write into the result.
        out = io.StringIO()
        with patch.object(sys, 'stderr', None), contextlib.redirect_stdout(out):
            agent._note('waiting for agent lock held by pid 1 (x)')
        self.assertEqual(out.getvalue(), '')

    def test_wait_hook_is_not_called_for_a_free_lock(self):
        with patch.object(agent, 'on_lock_wait', side_effect=AssertionError('no wait happened')):
            with agent.session(), agent.session():
                pass

    def test_other_thread_of_this_process_times_out_too(self):
        entered, leave = threading.Event(), threading.Event()

        def work():
            with agent.session():
                entered.set()
                leave.wait(10)

        thread = threading.Thread(target=work)
        thread.start()
        try:
            self.assertTrue(entered.wait(10))
            with self.assertRaisesRegex(agent.AgentNotSent, 'agent busy: lock held by another thread of this process for 0s; nothing was sent'):
                with agent.session():
                    self.fail('entered a session held by another thread')
        finally:
            leave.set()
            thread.join()
        with agent.session():
            pass

    def test_timeout_setting(self):
        for value, expected in (('', 60), ('abc', 60), ('-1', 60), ('nan', 60), ('5', 5), ('0', 0), ('0.5', .5), ('inf', 86400)):
            with self.subTest(value=value), patch.dict(os.environ, {'UNCRASHED_LOCK_TIMEOUT': value}):
                self.assertEqual(agent._lock_timeout(), expected)

    def test_lock_path_comes_from_the_environment(self):
        code = 'from uncrashed_mcp import agent; print(agent.LOCK)'
        for value, expected in ((str(self.lock), str(self.lock)), ('', '/tmp/uncrashed-agent.lock')):
            env = dict(os.environ, UNCRASHED_LOCK=value, UNCRASHED_OFFLINE='1', PYTHONPATH=str(SRC))
            self.assertEqual(subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True).stdout.strip(), expected)


class ProtocolFileTests(unittest.TestCase):
    REPLY = ['#0 spawned 1a row=A class=B', 'batch lines=1 ok=1 failed=0']

    def test_every_batch_gets_its_own_file_named_in_the_command_and_removed_after_the_reply(self):
        seen = []

        def run(command, timeout):
            verb, windows, *rest = command.split()
            path = Path(windows[2:].replace('\\', '/'))
            seen.append((verb, path, rest, path.read_text(), agent.batch_files()))
            return self.REPLY

        with sandbox() as (root, _), patch.object(agent, 'run', side_effect=run):
            self.assertEqual(agent.batch(['spawn  A 0 0 0']), ({0: ['spawned 1a row=A class=B']}, 'batch lines=1 ok=1 failed=0'))
            agent.batch(['destroybox 1 2 3 4', 'clear'], quiet=True)
            self.assertEqual([p.name for p in root.iterdir()], ['lock.txt'])  # both files are gone again
        (verb, first, rest, text, present), (_, second, quiet, text2, present2) = seen
        self.assertEqual((verb, rest, text, present), ('batch', [], 'spawn A 0 0 0\n', [first]))
        self.assertEqual((quiet, text2, present2), (['quiet'], 'destroybox 1 2 3 4\nclear\n', [second]))
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, root)
        self.assertRegex(first.name, rf'^batch\.{os.getpid()}\.\d+\.[0-9a-f]{{8}}\.txt$')
        self.assertNotIn(agent.BATCH, (first, second))

    def test_a_job_that_was_sent_but_never_answered_keeps_its_own_commands(self):
        # The DLL opens the named file only when the job starts: a client that gave up, then the next client.
        gone = agent.AgentError('the agent is not loaded (unloaded, or the game closed)')
        for error in (agent.AgentTimeout('did not answer'), gone, KeyboardInterrupt()):
            def sent_then(command, timeout, error=error):
                agent.sent_count += 1  # what agent._execute does just before it writes the command
                raise error

            with self.subTest(error=error), sandbox() as (root, _):
                with patch.object(agent, 'run', side_effect=sent_then) as run, self.assertRaises(type(error)):
                    agent.batch(['spawn A1 0 0 0', 'spawn A2 0 0 0'])
                pending = Path(run.call_args.args[0].split()[1][2:].replace('\\', '/'))
                self.assertEqual(agent.batch_files(), [pending])
                with patch.object(agent, 'run', return_value=self.REPLY) as run:
                    agent.batch(['spawn B1 0 0 0'])
                self.assertNotEqual(run.call_args.args[0].split()[1], agent._windows_path(pending))
                self.assertEqual(pending.read_text(), 'spawn A1 0 0 0\nspawn A2 0 0 0\n')  # what the late job will read
                self.assertEqual(agent.batch_files(), [pending])

    def test_batch_file_is_removed_when_the_wait_is_interrupted_before_sending(self):
        # The CLI ends a signalled wait by raising from on_lock_wait: not an AgentError, and nothing was sent.
        for error in (KeyboardInterrupt(), SystemExit(143)):
            with self.subTest(error=error), sandbox(), patch.object(agent, 'run', side_effect=error):
                with self.assertRaises(type(error)):
                    agent.batch(['spawn A 0 0 0'])
                self.assertEqual(agent.batch_files(), [])

    def test_batch_file_is_removed_when_nothing_was_sent_or_the_job_answered_failed(self):
        failed = agent.AgentError('agent job faulted: 1001 fault c0000005 rip=1 rva=0')
        failed.reply = {'seq': 1001, 'ok': False, 'head': 'ok batch', 'lines': [], 'ended': True}
        for error in (agent.AgentNotSent('agent busy: an earlier job is still running; nothing was sent'), failed):
            with self.subTest(error=error), sandbox(), patch.object(agent, 'run', side_effect=error):
                with self.assertRaises(agent.AgentError) as caught:
                    agent.batch(['spawn A 0 0 0'])
                self.assertIs(caught.exception, error)
                self.assertEqual(agent.batch_files(), [])

    def test_batch_files_older_than_a_day_are_swept_on_the_next_call(self):
        with sandbox() as (root, _), patch.object(agent, 'run', return_value=self.REPLY):
            old, recent, other = root / 'batch.1.1.aaaaaaaa.txt', root / 'batch.1.2.bbbbbbbb.txt', root / 'batchx.1.txt'
            for path, age in ((old, 86400 + 60), (recent, 86400 - 60), (other, 10 * 86400), (agent.BATCH, 10 * 86400)):
                path.write_text('spawn X 0 0 0\n')
                os.utime(path, (time.time() - age, time.time() - age))
            self.assertEqual(agent.batch([]), ({}, 'batch lines=0 ok=0 failed=0'))
            self.assertTrue(old.exists())  # an empty batch touches nothing
            agent.batch(['spawn A 0 0 0'])
            self.assertEqual(sorted(p.name for p in root.iterdir()), ['batch.1.2.bbbbbbbb.txt', 'batch.txt', 'batchx.1.txt', 'lock.txt'])

    def test_batch_through_the_protocol_names_and_removes_its_file(self):
        with sandbox() as (root, _):
            thread = fake_agent(body='#0 spawned 1a row=A class=B\nbatch lines=1 ok=1 failed=0\nend {seq} ok\n')
            try:
                self.assertEqual(agent.batch(['spawn A 0 0 0'], timeout=5)[1], 'batch lines=1 ok=1 failed=0')
            finally:
                thread.join()
            seq, verb, windows = agent.CMD.read_text().split()
            self.assertEqual((seq, verb), ('1001', 'batch'))
            self.assertRegex(windows, r'^Z:\\.*\\batch\.\d+\.\d+\.[0-9a-f]{8}\.txt$')
            self.assertEqual(agent.batch_files(), [])

    def test_empty_batch_sends_nothing(self):
        with sandbox() as (root, _), patch.object(agent, 'run', side_effect=AssertionError('sent')):
            self.assertEqual(agent.batch([]), ({}, 'batch lines=0 ok=0 failed=0'))
            self.assertEqual(agent.batch_files(), [])
        self.assertFalse(hasattr(agent, 'BATCH_WINDOWS'))  # one fixed path for every client is what went wrong

    def test_sequence_file_is_replaced_not_truncated(self):
        with sandbox() as (root, _):
            self.assertEqual(agent._next_seq(), 1001)
            first = agent.SEQ.stat().st_ino
            self.assertEqual(agent._next_seq(), 1002)
            self.assertNotEqual(agent.SEQ.stat().st_ino, first)
            self.assertEqual((agent.SEQ.read_text(), [p.name for p in root.iterdir()]), ('1002', ['seq.txt']))


class BusyGuardTests(unittest.TestCase):
    def test_waits_for_an_earlier_job_before_sending(self):
        with sandbox() as (root, ensure):
            marker = agent.OUT.with_name(agent.OUT.name + '.tmp')
            marker.write_text('1000 ok batch\n')
            sent_early = []

            def earlier_job_ends():
                time.sleep(.3)
                sent_early.append(agent.CMD.exists())
                marker.unlink()

            stderr = io.StringIO()
            thread = fake_agent(before=earlier_job_ends)
            try:
                with contextlib.redirect_stderr(stderr):
                    reply = agent.execute('ping', timeout=5)
            finally:
                marker.unlink(missing_ok=True)
                thread.join()
            self.assertEqual(sent_early, [False])
            self.assertEqual(reply, {'seq': 1001, 'ok': True, 'head': 'ok ping', 'lines': ['pong'], 'ended': True})
            self.assertEqual(stderr.getvalue(), 'agent is finishing an earlier job\n')
            ensure.assert_called_once_with()

    def test_gives_up_without_sending(self):
        with sandbox(UNCRASHED_LOCK_TIMEOUT='0.2') as (root, _):
            marker = agent.OUT.with_name(agent.OUT.name + '.tmp')
            marker.write_text('')
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(agent.AgentNotSent, '^agent busy: an earlier job is still running; nothing was sent; ') as caught:
                    agent.execute('spawn A 0 0 0')
            self.assertFalse(agent.CMD.exists() or agent.SEQ.exists())
            # The hint ties deleting the marker to evidence that the job has ended, and names no job it cannot know.
            self.assertEqual(str(caught.exception).split('; ', 2)[2],
                             "if it stays busy read './uc log': a line 'job <n> <verb> <n>ms' near its end means the job has "
                             f'ended and {marker} is a leftover that may be deleted; without that line the job still runs or '
                             'hangs (wait, or restart the game) and the file must stay')
            agent.SEQ.write_text('1041')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(agent.AgentNotSent, "a line 'job 1041 <verb> <n>ms' near"):
                agent.execute('spawn A 0 0 0')
            self.assertEqual(agent.SEQ.read_text(), '1041')

    def test_stops_waiting_when_the_game_is_gone(self):
        with sandbox() as (root, _), patch.object(agent, 'loaded', return_value=False):
            agent.OUT.with_name(agent.OUT.name + '.tmp').write_text('')
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(agent.AgentNotSent, r'^the agent is not loaded \(unloaded, or the game closed\); nothing was sent$'):
                    agent.execute('ping')
            self.assertFalse(agent.CMD.exists())

    def test_agent_gone_after_the_command_was_written_is_not_a_not_sent_error(self):
        with sandbox() as (root, _), patch.object(agent, 'loaded', return_value=False):
            with self.assertRaisesRegex(agent.AgentError, r"^the agent is not loaded \(unloaded, or the game closed\); 'spawn A 0 0 0' "
                                                          'was sent before that and may have run$') as caught:
                agent.execute('spawn A 0 0 0')
            self.assertIs(type(caught.exception), agent.AgentError)
            self.assertIsNone(caught.exception.reply)
            self.assertEqual(agent.CMD.read_text(), '1001 spawn A 0 0 0\n')

    def test_the_wait_for_an_earlier_job_can_be_given_up_by_the_hook(self):
        with sandbox(UNCRASHED_LOCK_TIMEOUT='10') as (root, _):
            agent.OUT.with_name(agent.OUT.name + '.tmp').write_text('')
            seen = []

            def hook(holder, waited):
                seen.append((holder, agent._lock_depth, agent.sent_count))
                raise agent.AgentNotSent('interrupted while waiting; nothing was sent')

            sent = agent.sent_count
            with patch.object(agent, 'on_lock_wait', hook), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(agent.AgentNotSent, 'interrupted while waiting'):
                    agent.execute('clear')
            self.assertEqual(seen, [(None, 1, sent)])  # no holder: this process has the lock, the game is busy
            self.assertEqual((agent._lock_depth, agent.sent_count), (0, sent))
            self.assertFalse(agent.CMD.exists() or agent.SEQ.exists())

    def test_closed_stderr_does_not_stop_a_command_from_being_sent(self):
        class Gone(io.StringIO):
            def write(self, text):
                raise BrokenPipeError(32, 'Broken pipe')

        with sandbox() as (root, _):
            marker = agent.OUT.with_name(agent.OUT.name + '.tmp')
            marker.write_text('')
            thread = fake_agent(before=lambda: (time.sleep(.2), marker.unlink()))
            try:
                with contextlib.redirect_stderr(Gone()):
                    self.assertTrue(agent.execute('ping', timeout=5)['ok'])
            finally:
                marker.unlink(missing_ok=True)
                thread.join()

    def test_marker_older_than_the_game_process_is_ignored(self):
        with sandbox() as (root, _), patch.object(agent, '_game_started', return_value=time.time() + 60):
            agent.OUT.with_name(agent.OUT.name + '.tmp').write_text('left by a crashed game')
            thread = fake_agent()
            try:
                self.assertTrue(agent.execute('ping', timeout=5)['ok'])
            finally:
                thread.join()

    def test_unknown_game_start_falls_back_to_ten_minutes(self):
        with sandbox():
            marker = agent.OUT.with_name(agent.OUT.name + '.tmp')
            self.assertFalse(agent._job_running(None))
            marker.write_text('')
            self.assertTrue(agent._job_running(None))
            os.utime(marker, (time.time() - 590, time.time() - 590))
            self.assertTrue(agent._job_running(None))
            os.utime(marker, (time.time() - 610, time.time() - 610))
            self.assertFalse(agent._job_running(None))
            self.assertTrue(agent._job_running(time.time() - 3600))  # a long job in a game whose start is known

    def test_process_start_time_from_proc(self):
        started = time.time()
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE)
        try:
            self.assertAlmostEqual(agent._game_started(child.pid), started, delta=3)
        finally:
            child.stdin.close()
            child.wait()
        self.assertIsNone(agent._game_started(child.pid))

    def test_reply_timeout_says_the_command_was_sent(self):
        with sandbox() as (root, _):
            with self.assertRaisesRegex(agent.AgentTimeout, r"^the agent did not answer 'spawn A 0 0 0' within 0s; the command was sent and may still run$") as caught:
                agent.execute('spawn  A 0 0 0', timeout=.05)
            self.assertNotIsInstance(caught.exception, agent.AgentNotSent)
            self.assertEqual(agent.CMD.read_text(), '1001 spawn A 0 0 0\n')


class NotSentTests(unittest.TestCase):
    """Callers tell 'nothing was sent' from 'sent' by type and by sent_count, never by message text."""

    def refused(self, call, text: str) -> None:
        sent = agent.sent_count
        with self.assertRaisesRegex(agent.AgentNotSent, text):
            call()
        self.assertEqual(agent.sent_count, sent)
        self.assertFalse(agent.CMD.exists() or agent.SEQ.exists())

    def test_every_refusal_before_the_command_file_is_written(self):
        with sandbox(offline=True):
            self.refused(lambda: agent.execute('mapinfo'), 'offline mode')
        with sandbox():
            self.refused(lambda: agent.execute('  '), '^empty command$')
        with sandbox(UNCRASHED_LOCK_TIMEOUT='0'):
            fd = os.open(agent.LOCK, os.O_RDWR | os.O_CREAT, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                self.refused(lambda: agent.execute('mapinfo'), '^agent busy: lock held by another process')
            finally:
                os.close(fd)

    def test_injection_refusals(self):
        with tempfile.TemporaryDirectory() as tmp, sandbox(UNCRASHED_AGENT_DLL=str(Path(tmp) / 'agent.dll'),
                                                           UNCRASHED_WINE=str(Path(tmp) / 'wine')):
            with patch.object(agent, 'ensure_agent', ENSURE), patch.object(agent, 'loaded', return_value=False), patch.object(agent.time, 'sleep'):
                with patch.object(agent, 'game_pid', return_value=None):
                    self.refused(lambda: agent.execute('mapinfo'), '^Uncrashed is not running$')
                self.refused(lambda: agent.execute('mapinfo'), '^missing .*agent\\.dll$')
                for name in ('agent.dll', 'inject.exe', 'wine'):
                    (Path(tmp) / name).write_text('')
                failed = subprocess.CalledProcessError(1, ['wine'], stderr='err: could not open\n  process 4242\n')
                with patch.object(agent.subprocess, 'run', side_effect=failed):
                    self.refused(lambda: agent.execute('mapinfo'), '^injecting the agent failed: err: could not open process 4242$')
                with patch.object(agent.subprocess, 'run', side_effect=OSError(8, 'Exec format error')):
                    self.refused(lambda: agent.execute('mapinfo'), '^injecting the agent failed: .*Exec format error')
                clock = [0.]
                with patch.object(agent.subprocess, 'run'), patch.object(agent.time, 'time', side_effect=lambda: clock.__setitem__(0, clock[0] + 1) or clock[0]):
                    self.refused(lambda: agent.execute('mapinfo'), '^the agent did not load into the game$')

    def test_sent_count_rises_immediately_before_the_command_file_appears(self):
        with sandbox() as (root, _):
            sent, seen = agent.sent_count, []
            rename = Path.replace

            def replace(self, target):
                if Path(target) == agent.CMD:
                    seen.append((agent.sent_count - sent, agent.CMD.exists()))
                return rename(self, target)

            thread = fake_agent()
            try:
                with patch.object(Path, 'replace', replace):
                    agent.execute('ping', timeout=5)
            finally:
                thread.join()
            self.assertEqual(seen, [(1, False)])
            with self.assertRaises(agent.AgentTimeout):
                agent.execute('spawn A 0 0 0', timeout=.05)
            self.assertEqual(agent.sent_count - sent, 2)  # in flight, whatever became of the reply

    def test_failed_reply_is_a_plain_agent_error_carrying_the_reply(self):
        with sandbox() as (root, _):
            thread = fake_agent(body='#0 spawned 1a row=A class=B\n{seq} fault c0000005 rip=1234 rva=0\nend {seq} failed\n')
            try:
                with self.assertRaises(agent.AgentError) as caught:
                    agent.run('batch Z:\\tmp\\x.txt', timeout=5)
            finally:
                thread.join()
            error = caught.exception
            self.assertIs(type(error), agent.AgentError)
            self.assertEqual(str(error), 'agent job faulted: 1001 fault c0000005 rip=1234 rva=0')
            self.assertEqual(error.reply, {'seq': 1001, 'ok': False, 'head': 'ok batch', 'ended': True,
                                           'lines': ['#0 spawned 1a row=A class=B', '1001 fault c0000005 rip=1234 rva=0']})

    def test_reply_without_an_end_marker(self):
        with sandbox() as (root, _):
            thread = fake_agent(body='')  # a fault inside a quiet batch line swallows everything after the head
            try:
                with self.assertRaisesRegex(agent.AgentError, r'^agent job ended without an end marker \(fault\?\)$') as caught:
                    agent.batch(['spawn A 0 0 0', 'crash', 'spawn C 0 0 0'], timeout=5, quiet=True)
            finally:
                thread.join()
            self.assertEqual(caught.exception.reply, {'seq': 1001, 'ok': False, 'head': 'ok batch', 'lines': [], 'ended': False})
            self.assertNotIsInstance(caught.exception, (agent.AgentNotSent, agent.AgentTimeout))


def obj(actor: str = 'ab', x: float = 100) -> dict:
    return {'actor': actor, 'row': 'NewRow_1', 'group': 'house', 'xf': [x, 0, 0, 0, 0, 0, 1, 1, 1]}


class PreparedJournalTests(unittest.TestCase):
    """A process that died mid-apply leaves status 'prepared' and no 'after' snapshot."""
    TRANSACTION = 'a' * 32

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.path = self.root / groups.HISTORY / f'{self.TRANSACTION}.json'
        journal = {'version': 1, 'id': self.TRANSACTION, 'map': 'Demo', 'base': 'BaseBoard', 'status': 'prepared',
                   'groups': ['house'], 'before': {'session': 's', 'objects': [obj()]}, 'commands': ['T ...']}
        managed.atomic_write(self.path, json.dumps(journal))
        for item in (patch.object(agent, 'LOCK', self.root / 'agent.lock'),
                     patch.object(managed, 'context', return_value=({'folder': 'Demo', 'base': 'BaseBoard'}, self.root))):
            item.start()
            self.addCleanup(item.stop)

    def journal(self) -> dict:
        return json.loads(self.path.read_text())

    def scene(self, *objects: dict):
        # An unrelated group shows that only the journal's groups are compared.
        other = dict(obj('ff'), group='shed')
        return patch.object(groups, 'snapshot', return_value={'session': 'now', 'objects': [*objects, other]})

    def test_unchanged_scene_marks_rolled_back(self):
        with self.scene(obj('cd')), patch.object(groups, '_apply') as apply:  # handles may differ after a reload
            with self.assertRaisesRegex(ValueError, 'changed nothing; journal marked rolled_back'):
                groups.undo_edit(self.TRANSACTION)
            for preview in (True, False):
                with self.assertRaisesRegex(ValueError, 'not undoable'):
                    groups.undo_edit(self.TRANSACTION, preview=preview)
            apply.assert_not_called()
        journal = self.journal()
        self.assertEqual((journal['status'], journal['after']), ('rolled_back', {'session': 'now', 'objects': [obj('cd')]}))

    def test_changed_scene_becomes_recovery_required_then_previews_normally(self):
        with self.scene(obj(x=200)), patch.object(groups, '_apply', return_value={'transaction': 'b' * 32}) as apply:
            result = groups.undo_edit(self.TRANSACTION)
            journal = self.journal()
            self.assertEqual((result['preview'], result['changes'], result['groups']), (True, 1, ['house']))
            self.assertEqual((result['recovered'], result['deletes_not_in_before']), (groups.RECOVERED, 0))
            self.assertEqual((journal['status'], journal['after']), ('recovery_required', {'session': 'now', 'objects': [obj(x=200)]}))
            self.assertEqual((journal['recovered'], journal['error']), (True, 'apply was interrupted before its outcome was recorded'))
            apply.assert_not_called()

            # The note does not depend on the status the first preview has already rewritten.
            self.assertEqual(groups.undo_edit(self.TRANSACTION)['recovered'], groups.RECOVERED)
            self.assertEqual(groups.undo_edit(self.TRANSACTION, preview=False), {'transaction': 'b' * 32, 'recovered': groups.RECOVERED})
            commands = apply.call_args.args[4]
            self.assertEqual(len(commands), 1)
            self.assertTrue(commands[0].startswith('T ab NewRow_1 house 200 '))
            self.assertIn(' house 100 0 0 ', commands[0])
        journal = self.journal()
        self.assertEqual((journal['status'], journal['undo_transaction'], journal['recovered']), ('undone', 'b' * 32, True))

    def test_apply_is_refused_until_the_recovery_was_previewed(self):
        later = [dict(obj('e1'), row='Tree_7'), dict(obj('e2'), row='Tree_7')]  # added to the group after the interrupted edit
        with self.scene(obj(x=200), *later) as snapshot, patch.object(groups, '_apply', return_value={'transaction': 'b' * 32}) as apply:
            for _ in range(2):  # refusing changes nothing, so it cannot be worn down by repeating
                with self.assertRaisesRegex(ValueError, '^journal was interrupted; run the preview first'):
                    groups.undo_edit(self.TRANSACTION, preview=False)
            snapshot.assert_not_called()
            self.assertEqual((self.journal()['status'], 'after' in self.journal()), ('prepared', False))

            result = groups.undo_edit(self.TRANSACTION)
            self.assertEqual((result['changes'], result['deletes_not_in_before'], result['recovered']), (3, 2, groups.RECOVERED))
            apply.assert_not_called()

            self.assertEqual(groups.undo_edit(self.TRANSACTION, preview=False)['recovered'], groups.RECOVERED)
            self.assertEqual(sorted(command.split()[0] + ' ' + command.split()[1] for command in apply.call_args.args[4]),
                             ['D e1', 'D e2', 'T ab'])
        self.assertEqual(self.journal()['status'], 'undone')

    def test_ordinary_journal_has_no_recovery_note(self):
        journal = dict(self.journal(), status='applied', after={'session': 's', 'objects': [obj(x=200), obj('c1', x=300)]})
        managed.atomic_write(self.path, json.dumps(journal))
        with self.scene(obj(x=200), obj('c1', x=300)), patch.object(groups, '_apply', return_value={'transaction': 'b' * 32}):
            result = groups.undo_edit(self.TRANSACTION)
            self.assertEqual((result['changes'], result['deletes_not_in_before'], 'recovered' in result), (2, 1, False))
            self.assertEqual(groups.undo_edit(self.TRANSACTION, preview=False), {'transaction': 'b' * 32})

    def test_failed_snapshot_leaves_the_journal_prepared(self):
        with patch.object(groups, 'snapshot', side_effect=agent.AgentError('agent busy')):
            with self.assertRaisesRegex(agent.AgentError, 'busy'):
                groups.undo_edit(self.TRANSACTION)
        self.assertEqual(self.journal()['status'], 'prepared')
        self.assertNotIn('after', self.journal())


if __name__ == '__main__':
    unittest.main()
