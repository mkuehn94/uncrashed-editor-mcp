import os

os.environ['UNCRASHED_OFFLINE'] = '1'  # second guard: nothing in this module may reach the game

import contextlib
import functools
import inspect
import io
import json
import shlex
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import uncrashed_mcp
from uncrashed_mcp import agent, api, cli
from uncrashed_mcp.render import parse

ROOT = Path(__file__).resolve().parents[1]
UC = str(ROOT / 'uc')
EMPTY = inspect.Parameter.empty

_guards = []
_scratch = None


def setUpModule():
    global _scratch
    _scratch = tempfile.TemporaryDirectory()
    _guards.append(patch.object(agent, 'LOCK', Path(_scratch.name) / 'agent.lock'))
    for name in ('ensure_agent', '_execute'):
        _guards.append(patch.object(agent, name, side_effect=AssertionError(f'agent.{name} reached from a test')))
    for guard in _guards:
        guard.start()


def tearDownModule():
    while _guards:
        _guards.pop().stop()
    _scratch.cleanup()


def environment(**extra):
    """The caller's environment without its own CLI settings, offline."""
    kept = {key: value for key, value in os.environ.items()
            if key not in ('UNCRASHED_FORMAT', 'UNCRASHED_DEBUG', 'UNCRASHED_LOCK_TIMEOUT')}
    return {**kept, 'UNCRASHED_OFFLINE': '1', **extra}


def run(*argv, stdin='', env=None):
    """cli.main in-process -> (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with patch.dict(os.environ, environment(**(env or {})), clear=True), \
            patch.object(sys, 'stdin', io.StringIO(stdin)), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


@contextlib.contextmanager
def fake(name, result=None, body=None):
    """Replace an operation's function, keeping its signature; yields the keyword arguments of every call."""
    op = api.OPERATIONS[name]
    calls = []

    @functools.wraps(op.fn)
    def stub(*args, **kwargs):
        assert not args, 'the CLI passes keyword arguments only'
        calls.append(kwargs)
        if body is not None:
            return body(**kwargs)
        if isinstance(result, BaseException):
            raise result
        return {} if result is None else result

    with patch.object(op, 'fn', stub):
        yield calls


def uc(*argv, stdin=None, env=None, cwd=None):
    """The real launcher in a subprocess, offline and on a private lock file."""
    full = environment(UNCRASHED_LOCK=str(Path(_scratch.name) / 'uc.lock'), **(env or {}))
    return subprocess.run([UC, *argv], input=stdin, capture_output=True, text=True, env=full, cwd=cwd, timeout=60)


class Case(unittest.TestCase):
    def bound(self, name, *argv, stdin=''):
        """Keyword arguments an operation receives for a command line that must be accepted."""
        with fake(name) as calls, patch.object(api.OPERATIONS[name], 'check', None):
            code, out, err = run(*argv, stdin=stdin)
        self.assertEqual((code, err), (0, ''), argv)
        self.assertEqual(len(calls), 1, argv)
        return calls[0]

    def refused(self, name, *argv, stdin='', needle=''):
        """A command line that must be a usage error: exit 2, one error line, a usage line, nothing executed."""
        with fake(name) as calls:
            code, out, err = run(*argv, stdin=stdin)
        self.assertEqual(code, 2, (argv, err))
        self.assertEqual(calls, [], argv)
        self.assertEqual(out, '')
        lines = err.splitlines()
        self.assertEqual(len([line for line in lines if line.startswith('error: ')]), 1, err)
        self.assertTrue(lines[0].startswith('error: '), err)
        self.assertEqual(lines[1:], [f'usage: {cli._usage(api.OPERATIONS[name])}'], err)
        self.assertIn(needle, lines[0])
        return lines[0]


# ---------------------------------------------------------------------------
# binding
# ---------------------------------------------------------------------------


def sample(kind, param, tmp):
    """(tokens when given as an option value, expected Python value) for one parameter."""
    if kind == 'str':
        return [f'v_{param.name}'], f'v_{param.name}'
    if kind == 'int':
        return ['-7'], -7
    if kind == 'float':
        return ['-2.5'], -2.5
    if kind == 'strs':
        return ['a1,b2'], ['a1', 'b2']
    if kind == 'floats':
        return ['-1,0.5,2'], [-1.0, 0.5, 2.0]
    if kind == 'points':
        return ['[[1,2],[-3,4.5]]'], [[1.0, 2.0], [-3.0, 4.5]]
    if kind == 'strmap':
        return ['a=b,c.d=e'], {'a': 'b', 'c.d': 'e'}
    if kind == 'map':
        return ['a=1,b.c=2.5,d=text'], {'a': 1, 'b': {'c': 2.5}, 'd': 'text'}
    if kind == 'maps':
        return ['[{"x":1},{"y":[2]}]'], [{'x': 1}, {'y': [2]}]
    raise AssertionError(kind)


def as_option(op, param, kind, tmp):
    """(argv tokens, expected value) binding one parameter by its option."""
    if kind == 'bool':
        want = param.default is not True
        return [f'--{param.name}' if want else f'--no-{param.name}'], want
    if param.name == op.stdin:
        return [f'--{param.name}', 'spawn NewRow_1 0 0 0', f'--{param.name}', 'cells 5 a,b'], \
            ['spawn NewRow_1 0 0 0', 'cells 5 a,b']
    tokens, expected = sample(kind, param, tmp)
    return [f'--{param.name}', *tokens], expected


def as_positional(op, param, kind, tmp):
    if param.name == op.variadic and kind == 'points':
        return ['1,2', '-3,4.5'], [[1.0, 2.0], [-3.0, 4.5]]
    if param.name == op.variadic:
        return ['a1', 'b2,c3'], ['a1', 'b2', 'c3']
    return sample(kind, param, tmp)


class EveryParameterTests(Case):
    def test_registry_only_uses_types_the_cli_can_bind(self):
        for op in api.OPERATIONS.values():
            for param in op.params():
                self.assertIn(cli._kind(param.annotation), cli.LABELS | {'bool': ''}, f'{op.name}.{param.name}')

    def test_hints_name_real_parameters_and_nothing_collides_with_cli_words(self):
        reserved = {*cli.GLOBAL_FLAGS, 'apply', 'region'}
        commands = set()
        for op in api.OPERATIONS.values():
            names = [p.name for p in op.params()]
            for hint in (*op.positional, op.variadic, op.rest, op.stdin):
                self.assertIn(hint, [*names, None], op.name)
            self.assertFalse(reserved & set(names), op.name)
            self.assertFalse([n for n in names if n.startswith('no_') or '-' in n], op.name)
            if op.variadic:
                self.assertEqual(op.positional[-1], op.variadic, op.name)
            for word in (op.name, *op.aliases):
                self.assertNotIn(word, commands)
                self.assertNotIn(word, cli.BUILTINS + cli.TOPICS)
                commands.add(word)
            # every required parameter can be given without an option
            for param in op.params():
                if param.default is EMPTY:
                    self.assertIn(param.name, (*op.positional, op.rest, op.stdin), f'{op.name}.{param.name}')

    def test_every_public_parameter_of_every_operation_binds_in_one_command_line(self):
        for op in api.OPERATIONS.values():
            with self.subTest(op=op.name), tempfile.TemporaryDirectory() as tmp:
                params = {param.name: param for param in op.params()}
                front, options, tail, expected = [], [], [], {}
                for name in op.positional:
                    tokens, expected[name] = as_positional(op, params[name], cli._kind(params[name].annotation), tmp)
                    front += tokens
                for name, param in params.items():
                    if name == op.rest:
                        tail = ['spawn', 'NewRow_1', '-5', '0', '0', 'group=g']
                        expected[name] = 'spawn NewRow_1 -5 0 0 group=g'
                    elif name == op.stdin:
                        path = Path(tmp) / 'plan.txt'
                        path.write_text('# plan\nspawn NewRow_1 0 0 0\n\ncells 5 a,b\n')
                        front.append(str(path))
                        expected[name] = ['# plan', 'spawn NewRow_1 0 0 0', '', 'cells 5 a,b']
                        expected['_lines'] = [1, 2, 3, 4]
                    elif name not in op.positional:
                        tokens, expected[name] = as_option(op, param, cli._kind(param.annotation), tmp)
                        options += tokens
                self.assertEqual(set(expected) - {'_lines'}, set(params))
                self.assertEqual(self.bound(op.name, op.name, *front, *options, *tail), expected)
                if not op.rest:  # options may also come first
                    self.assertEqual(self.bound(op.name, op.name, *options, *front), expected)

    def test_every_public_parameter_binds_as_an_option(self):
        for op in api.OPERATIONS.values():
            params = op.params()
            for param in params:
                with self.subTest(op=op.name, param=param.name), tempfile.TemporaryDirectory() as tmp:
                    kind = cli._kind(param.annotation)
                    tokens, expected = as_option(op, param, kind, tmp)
                    others = []
                    for other in params:
                        if other.default is EMPTY and other.name != param.name:
                            others += as_option(op, other, cli._kind(other.annotation), tmp)[0]
                    got = self.bound(op.name, op.name, *others, *tokens)
                    self.assertEqual(got[param.name], expected)
                    self.assertEqual(type(got[param.name]), type(expected))

    def test_every_registry_example_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'plan.txt').write_text('mapinfo\n')
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                for op in api.OPERATIONS.values():
                    for example in op.examples:
                        with self.subTest(example=example):
                            words = shlex.split(example)
                            self.assertEqual(words[0], './uc')
                            result = api.Text('ok') if op.name in ('agent_exec', 'agent_batch', 'agent_log') else {}
                            with fake(op.name, result) as calls, patch.object(op, 'check', None):
                                code, out, err = run(*words[1:], stdin='mapinfo\n')
                            self.assertEqual((code, err, len(calls)), (0, '', 1), example)
            finally:
                os.chdir(previous)


class NumberTests(Case):
    def test_negative_values_in_every_position(self):
        self.assertEqual(self.bound('move_camera', 'move_camera', '1', '-2', '3', '-90', '--pitch', '-30'),
                         {'x': 1.0, 'y': -2.0, 'z': 3.0, 'yaw': -90.0, 'pitch': -30.0})
        self.assertEqual(self.bound('move_camera', 'move_camera', '-1', '-2', '-3', '-90', '-30'),
                         {'x': -1.0, 'y': -2.0, 'z': -3.0, 'yaw': -90.0, 'pitch': -30.0})
        self.assertEqual(self.bound('edit_group', 'edit_group', 'g', '--delta', '-1,0,0'),
                         {'group': 'g', 'delta': [-1.0, 0.0, 0.0]})
        self.assertEqual(self.bound('sample_terrain', 'sample_terrain', '-12.5,30', '4,-5'),
                         {'points': [[-12.5, 30.0], [4.0, -5.0]]})
        self.assertEqual(self.bound('inspect_site', 'inspect_site', '-50,-50,0', '50,50,40', '--spacing', '-.5'),
                         {'bounds_min': [-50.0, -50.0, 0.0], 'bounds_max': [50.0, 50.0, 40.0], 'spacing': -0.5})
        self.assertEqual(self.bound('agent_log', 'log', '-5'), {'lines': -5})
        self.assertEqual(self.bound('check_assembly', 'check_assembly', 'a', '--tolerance=-1e3', '--top', '-1'),
                         {'name': 'a', 'tolerance': -1000.0, 'top': -1})
        self.assertEqual(self.bound('check_collisions', 'check_collisions', '--region', '-100,-100,100,100'),
                         {'x0': -100.0, 'y0': -100.0, 'x1': 100.0, 'y1': 100.0})

    def test_numbers_are_strict_decimal_and_finite(self):
        for bad in ('3.0', '0x10', '1_000', '1e2', ' 3', '', 'three', '٣'):
            self.refused('search_assets', 'search_assets', '--limit', bad, needle='--limit expects an integer')
        for bad in ('inf', '-inf', 'nan', '1e999', '0x1p3', '1_0.5', '1,5', '', '1.5m'):
            self.refused('move_camera', 'move_camera', '1', '2', '3', f'--yaw={bad}', needle='finite decimal')
        self.refused('move_camera', 'move_camera', '1', '2', 'up', '4', needle='Z expects')
        for good, value in (('7', 7.0), ('+7', 7.0), ('.5', 0.5), ('5.', 5.0), ('1e3', 1000.0), ('-2.5E-1', -0.25)):
            got = self.bound('move_camera', 'move_camera', '0', '0', '0', good)
            self.assertEqual(got['yaw'], value)
            self.assertIs(type(got['yaw']), float)

    def test_typing_follows_the_annotation_never_the_value(self):
        got = self.bound('save_live', 'save_live', '42', '--display_name', 'true')
        self.assertEqual(got, {'name': '42', 'display_name': 'true'})
        self.assertEqual(self.bound('search_assets', 'search_assets', '-5'), {'query': '-5'})
        self.assertEqual(self.bound('search_assets', 'search_assets', '-'), {'query': '-'})
        self.assertEqual(self.bound('search_assets', 'search_assets', ''), {'query': ''})


class ParsePhaseTests(Case):
    """Inputs that ended in a traceback and exit 1 although nothing had run."""

    def test_an_integer_with_more_digits_than_python_converts(self):
        line = self.refused('search_assets', 'search_assets', '--limit', '9' * 5000)
        self.assertEqual(line, 'error: --limit expects an integer, got one of 5000 characters')
        self.refused('agent_log', 'log', '-' + '9' * 5000, needle='LINES expects an integer')
        self.assertEqual(self.bound('search_assets', 'search_assets', '--limit', '9' * 30), {'limit': int('9' * 30)})

    def test_json_nested_too_deeply(self):
        for depth in (cli.JSON_DEPTH + 1, 5000, 100000):
            for name, argv in (('get_assets', ['[' * depth]), ('get_assets', ['[' * depth + ']' * depth]),
                               ('check_performance', ['--poses', '[' * depth + ']' * depth]),
                               ('check_performance', ['--budgets', '{"a":' * depth + '1' + '}' * depth]),
                               ('check_performance', ['--budgets', 'a=' + '[' * depth + ']' * depth])):
                with fake(name) as calls:
                    code, out, err = run(name, *argv)
                self.assertEqual((code, out, calls), (2, '', []), (name, depth))
                self.assertNotIn('Traceback', err)
                self.assertTrue(err.startswith('error: '), err[:200])
        line = self.refused('check_performance', 'check_performance', '--poses', '[' * 5000 + ']' * 5000)
        self.assertEqual(line, f'error: --poses: JSON nested more than {cli.JSON_DEPTH} levels deep')
        deep = '[' * (cli.JSON_DEPTH - 1) + '{"x":1}' + ']' * (cli.JSON_DEPTH - 1)  # the limit itself is fine
        self.refused('check_performance', 'check_performance', '--poses', deep, needle='JSON list of objects')

    def test_stdin_named_but_closed(self):
        for name, argv in (('agent_batch', ['batch', '-']), ('agent_batch', ['batch', '--commands', '@-']),
                           ('get_assets', ['get_assets', '@-']), ('sample_terrain', ['sample_terrain', '@-'])):
            out, err = io.StringIO(), io.StringIO()
            with fake(name) as calls, patch.object(sys, 'stdin', None), contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(err):
                self.assertEqual(cli.main(argv), 2, argv)
            self.assertEqual((calls, out.getvalue()), ([], ''))
            self.assertIn('stdin (-) is closed', err.getvalue().splitlines()[0])
            self.assertNotIn('Traceback', err.getvalue())

    def test_a_programming_error_in_the_parse_phase_is_not_dressed_up_as_a_usage_error(self):
        with patch.object(cli, '_kind', side_effect=TypeError('no command-line form')):
            with self.assertRaises(TypeError):
                run('search_assets', 'house')


class BoolTests(Case):
    def test_flag_forms(self):
        base = ('move_camera', '1', '2', '3', '4')
        for tokens, want in ((['--hold'], True), (['--no-hold'], False), (['--hold=false'], False),
                             (['--hold=YES'], True), (['--hold=0'], False), (['--hold=No'], False),
                             (['--hold', 'false'], False), (['--hold', 'TRUE'], True), (['--no_hold'], False)):
            self.assertIs(self.bound('move_camera', *base, *tokens)['hold'], want, tokens)

    def test_bool_consumes_only_a_literal_true_or_false(self):
        got = self.bound('edit_group', 'edit_group', '--prefix', 'houses')
        self.assertEqual(got, {'prefix': True, 'group': 'houses'})
        got = self.bound('edit_group', 'edit_group', '--prefix', 'false', 'houses')
        self.assertEqual(got, {'prefix': False, 'group': 'houses'})

    def test_a_flag_followed_by_a_word_that_reads_as_its_value_is_refused(self):
        # each of these bound the flag true and moved the word into a positional, at exit 0
        cases = ((['set_collision', '--group', 'fence', '--persist', 'off', '--apply'], '--persist off is ambiguous'),
                 (['set_collision', '--group', 'fence', '--force', 'on', '--apply'], '--force on is ambiguous'),
                 (['move_camera', '1', '2', '3', '4', '--hold', '0'], '--hold 0 is ambiguous'),
                 (['search_assets', '--include_hidden', 'no'], '--include_hidden no is ambiguous'),
                 (['capture_view', '--foreground', '0'], '--foreground 0 is ambiguous'),
                 (['edit_group', '--actors', 'A,B', '--action', 'assign', '--target_group', 'f', '--prefix', 'NO',
                   '--apply'], '--prefix NO is ambiguous'),
                 (['check_route_clearance', '--closed', '1'], '--closed 1 is ambiguous'),
                 (['search_assets', '--include_hidden', 'yes'], '--include_hidden yes is ambiguous'),
                 (['set_collision', '--group', 'fence', '--apply', '0'], "--apply takes no value but '0' follows"),
                 (['set_collision', '--group', 'fence', '--apply', 'off'], "--apply takes no value but 'off' follows"),
                 (['set_collision', '--group', 'fence', '--no-persist', 'on'], "--no-persist takes no value but 'on' follows"))
        for argv, needle in cases:
            line = self.refused(argv[0], *argv, needle=needle)
            self.assertIn('put a positional', line)
            self.assertIn('=true', line)
            self.assertNotIn('=VALUE', line)  # --flag=on is not a spelling
            self.assertNotIn('after --', line)  # `--flag -- on --apply` would make --apply a positional
        line = self.refused('move_camera', 'move_camera', '1', '2', '3', '4', '--hold', '0')
        self.assertEqual(line, "error: --hold 0 is ambiguous; write --hold=true, --hold=false or --no-hold, or put a "
                               "positional '0' before --hold")
        # the unambiguous spellings of the same lines
        self.assertEqual(self.bound('set_collision', 'set_collision', 'off', '--group', 'fence', '--no-persist', '--apply'),
                         {'mode': 'off', 'group': 'fence', 'persist': False, 'preview': False})
        self.assertEqual(self.bound('set_collision', 'set_collision', 'on', '--group', 'fence', '--force'),
                         {'mode': 'on', 'group': 'fence', 'force': True})
        self.assertEqual(self.bound('move_camera', 'move_camera', '1', '2', '3', '4', '0', '--hold'),
                         {'x': 1.0, 'y': 2.0, 'z': 3.0, 'yaw': 4.0, 'pitch': 0.0, 'hold': True})
        self.assertEqual(self.bound('search_assets', 'search_assets', '--include_hidden=no'), {'include_hidden': False})
        self.assertEqual(self.bound('search_assets', 'search_assets', '--include_hidden', 'none'),
                         {'include_hidden': True, 'query': 'none'})  # any other word is still a positional

    def test_optional_bool(self):
        self.assertIs(self.bound('check_route_clearance', 'check_route_clearance', '--closed')['closed'], True)
        self.assertIs(self.bound('check_route_clearance', 'check_route_clearance', '--no-closed')['closed'], False)
        self.assertNotIn('closed', self.bound('check_route_clearance', 'check_route_clearance'))

    def test_bad_bool_forms(self):
        base = ('move_camera', '1', '2', '3', '4')
        self.refused('move_camera', *base, '--hold=maybe', needle='expects true or false')
        self.refused('move_camera', *base, '--no-hold=true', needle='--no-hold takes no value')
        self.refused('move_camera', *base, '--no-hold', 'true', needle='--no-hold takes no value')
        self.refused('move_camera', *base, '--hold', '--no-hold', needle='both set hold')
        self.refused('move_camera', *base, '--hold', '--hold', needle='--hold given twice')

    def test_apply_is_no_preview(self):
        self.assertEqual(self.bound('undo_edit', 'undo_edit', 't1', '--apply'), {'transaction': 't1', 'preview': False})
        self.assertEqual(self.bound('undo_edit', 'undo_edit', 't1', '--no-preview')['preview'], False)
        self.assertEqual(self.bound('undo_edit', 'undo_edit', 't1', '--preview')['preview'], True)
        self.assertNotIn('preview', self.bound('undo_edit', 'undo_edit', 't1'))
        for tokens in (['--apply', '--preview'], ['--preview', '--apply'], ['--preview=true', '--apply']):
            self.refused('undo_edit', 'undo_edit', 't1', *tokens, needle='--apply and --preview contradict')
        self.refused('undo_edit', 'undo_edit', 't1', '--apply', '--apply', needle='--apply given twice')
        self.refused('undo_edit', 'undo_edit', 't1', '--apply', '--no-preview', needle='both set preview')
        self.refused('undo_edit', 'undo_edit', 't1', '--apply=true', needle='--apply takes no value')
        self.refused('undo_edit', 'undo_edit', 't1', '--apply', 'false', needle='--apply takes no value')
        self.refused('list_groups', 'list_groups', '--apply', needle='unknown option --apply')


class ListTests(Case):
    def test_one_token_per_list_option(self):
        got = self.bound('edit_group', 'edit_group', '--delta', '1,0,0', 'houses')
        self.assertEqual(got, {'delta': [1.0, 0.0, 0.0], 'group': 'houses'})
        self.refused('edit_group', 'edit_group', '--delta', '1', '0', '0', 'houses', needle='at most 1 (GROUP)')
        got = self.bound('edit_group', 'edit_group', '--actors', '1A2B,3C4D', '--actors', '5E6F')
        self.assertEqual(got, {'actors': ['1A2B', '3C4D', '5E6F']})
        got = self.bound('edit_group', 'edit_group', '--delta', '[1, 2.5, -3]', '--actors', '["a,b", "12"]')
        self.assertEqual(got, {'delta': [1.0, 2.5, -3.0], 'actors': ['a,b', '12']})
        self.assertEqual(self.bound('set_collision', 'set_collision', '--actors', '007,1e5'),
                         {'actors': ['007', '1e5']})

    def test_at_file_and_stdin(self):
        with tempfile.TemporaryDirectory() as tmp:
            ids = Path(tmp) / 'ids.txt'
            ids.write_text('NewRow_1, NewRow_2\nNewRow_3\tNewRow_4\n\n')
            self.assertEqual(self.bound('consolidate_scenery', 'consolidate_scenery', '--rows', f'@{ids}'),
                             {'rows': ['NewRow_1', 'NewRow_2', 'NewRow_3', 'NewRow_4']})
            self.assertEqual(self.bound('get_assets', 'get_assets', f'@{ids}', 'NewRow_9')['ids'],
                             ['NewRow_1', 'NewRow_2', 'NewRow_3', 'NewRow_4', 'NewRow_9'])
            nums = Path(tmp) / 'nums.json'
            nums.write_text('[1, 2, 3.5]')
            self.assertEqual(self.bound('edit_group', 'edit_group', 'g', '--pivot', f'@{nums}')['pivot'], [1.0, 2.0, 3.5])
            empty = Path(tmp) / 'empty.txt'
            empty.write_text(' \n')
            self.refused('consolidate_scenery', 'consolidate_scenery', '--rows', f'@{empty}', needle='holds no values')
            (Path(tmp) / 'none.json').write_text('[]')
            self.refused('consolidate_scenery', 'consolidate_scenery', '--rows', f"@{Path(tmp) / 'none.json'}",
                         needle='holds no values')
            self.refused('consolidate_scenery', 'consolidate_scenery', '--rows', f"@{Path(tmp) / 'absent'}",
                         needle='cannot read')
        self.assertEqual(self.bound('consolidate_scenery', 'consolidate_scenery', '--rows', '@-', stdin='r1\nr2\n'),
                         {'rows': ['r1', 'r2']})
        self.refused('set_collision', 'set_collision', '--actors', '@-', '--channels', '@-', stdin='a\n',
                     needle='already read')

    def test_an_empty_list_is_refused_unless_written_as_brackets(self):
        self.refused('consolidate_scenery', 'consolidate_scenery', '--rows', '', '--apply', needle='empty value')
        self.refused('consolidate_scenery', 'consolidate_scenery', '--rows=', '--apply', needle='empty value')
        self.refused('consolidate_scenery', 'consolidate_scenery', '--rows', 'a,,b', needle='empty value')
        self.refused('consolidate_scenery', 'consolidate_scenery', '--rows', 'a,', needle='empty value')
        self.refused('get_assets', 'get_assets', '', needle='empty value')
        self.refused('set_collision', 'set_collision', 'off', '--actors', '', '--apply', needle='--actors: empty value')
        self.refused('set_collision', 'set_collision', 'off', '--actors', needle='--actors needs a value')
        self.assertEqual(self.bound('consolidate_scenery', 'consolidate_scenery', '--rows=[]'), {'rows': []})
        self.assertEqual(self.bound('consolidate_scenery', 'consolidate_scenery', '--rows', '[]'), {'rows': []})

    def test_json_lists_are_checked_against_the_annotation(self):
        self.refused('edit_group', 'edit_group', '--delta', '["a"]', needle='finite decimal')
        self.refused('edit_group', 'edit_group', '--delta', '[true]', needle='expects finite numbers')
        self.refused('edit_group', 'edit_group', '--delta', '[[1,2]]', needle='expects finite numbers')
        self.refused('edit_group', 'edit_group', '--actors', '[1]', needle='expects strings')
        self.refused('edit_group', 'edit_group', '--actors', '[1', needle='invalid JSON')
        self.refused('edit_group', 'edit_group', '--delta', 'a,b,c', needle='finite decimal')

    def test_positional_lists_take_one_token_each(self):
        self.assertEqual(self.bound('inspect_site', 'inspect_site', '0,0,0', '[10, 10, 5]'),
                         {'bounds_min': [0.0, 0.0, 0.0], 'bounds_max': [10.0, 10.0, 5.0]})
        self.refused('inspect_site', 'inspect_site', '0', '0', '0', '10', '10', '5', needle='at most 2 (BOUNDS_MIN')
        self.refused('inspect_site', 'inspect_site', '0,0,0', needle='missing BOUNDS_MAX')
        self.refused('inspect_site', 'inspect_site', '0,0,0', '1,1,1', '--bounds_min', '2,2,2',
                     needle='BOUNDS_MIN and --bounds_min both set')

    def test_variadic_takes_all_remaining_positionals(self):
        self.assertEqual(self.bound('get_assets', 'get_assets', 'NewRow_1', 'NewRow_2', 'NewRow_3'),
                         {'ids': ['NewRow_1', 'NewRow_2', 'NewRow_3']})
        self.assertEqual(self.bound('get_assets', 'get_assets', 'NewRow_1,NewRow_2'), {'ids': ['NewRow_1', 'NewRow_2']})
        self.refused('get_assets', 'get_assets', needle='missing IDS')
        self.refused('get_assets', 'get_assets', 'NewRow_1', '--ids', 'NewRow_2', needle='IDS and --ids both set')


class PointTests(Case):
    def test_points_as_variadic_positionals(self):
        self.assertEqual(self.bound('sample_terrain', 'sample_terrain', '10,20', '12.5,20', '-15,20.5', '--z_from', '500'),
                         {'points': [[10.0, 20.0], [12.5, 20.0], [-15.0, 20.5]], 'z_from': 500.0})
        self.assertEqual(self.bound('sample_terrain', 'sample_terrain', '[[1,2],[3,4]]', '5,6')['points'],
                         [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def test_check_opening_path_as_variadic_points(self):
        got = self.bound('check_opening', 'check_opening', '0,-8,4', '0,0,4', '0,8,5', '--profile', 'cinewhoop',
                         '--margin_m', '0.3', '--channel', 'game3')
        self.assertEqual(got, {'path': [[0.0, -8.0, 4.0], [0.0, 0.0, 4.0], [0.0, 8.0, 5.0]], 'profile': 'cinewhoop',
                               'margin_m': .3, 'channel': 'game3'})
        self.refused('check_opening', 'check_opening', needle='missing PATH')

    def test_points_option_takes_one_token_and_repeats(self):
        got = self.bound('check_route_clearance', 'check_route_clearance', 'points', '--points', '[[0,0,5],[40,10,8]]')
        self.assertEqual(got, {'source': 'points', 'points': [[0.0, 0.0, 5.0], [40.0, 10.0, 8.0]]})
        got = self.bound('check_route_clearance', 'check_route_clearance', '--points', '0,0,5', '--points', '-4,1,8',
                         '--points', '[7, 8]')
        self.assertEqual(got['points'], [[0.0, 0.0, 5.0], [-4.0, 1.0, 8.0], [7.0, 8.0]])
        self.assertEqual(self.bound('check_route_clearance', 'check_route_clearance', '--points=[]')['points'], [])

    def test_points_from_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines = Path(tmp) / 'pts.txt'
            lines.write_text('1,2\n-3 4.5\n\n5, 6, 7\n')
            self.assertEqual(self.bound('sample_terrain', 'sample_terrain', f'@{lines}')['points'],
                             [[1.0, 2.0], [-3.0, 4.5], [5.0, 6.0, 7.0]])
            doc = Path(tmp) / 'pts.json'
            doc.write_text('[[1, 2], [3, 4]]')
            got = self.bound('check_route_clearance', 'check_route_clearance', '--points', f'@{doc}')
            self.assertEqual(got['points'], [[1.0, 2.0], [3.0, 4.0]])
            (Path(tmp) / 'none.txt').write_text('\n')
            self.refused('sample_terrain', 'sample_terrain', f"@{Path(tmp) / 'none.txt'}", needle='holds no points')
        self.assertEqual(self.bound('sample_terrain', 'sample_terrain', '@-', stdin='1,2\n3,4\n')['points'],
                         [[1.0, 2.0], [3.0, 4.0]])

    def test_bad_points(self):
        self.refused('sample_terrain', 'sample_terrain', '10', '20', needle='a point is x,y or x,y,z')
        self.refused('sample_terrain', 'sample_terrain', '1,2,3,4', needle='a point is x,y or x,y,z')
        self.refused('sample_terrain', 'sample_terrain', '1,north', needle='finite decimal')
        self.refused('sample_terrain', 'sample_terrain', '[[1,2],3]', needle='JSON list of')
        self.refused('sample_terrain', 'sample_terrain', '{"x":1}', needle='a point is')


class DictTests(Case):
    def test_text_dict(self):
        want = {'base': 'foundation', 'frame': 'structure'}
        for tokens in (['--roles', 'base=foundation', '--roles', 'frame=structure'],
                       ['--roles', 'base=foundation,frame=structure'],
                       ['--roles', '{"base": "foundation", "frame": "structure"}']):
            self.assertEqual(self.bound('define_assembly', 'define_assembly', 'tower', *tokens),
                             {'name': 'tower', 'roles': want}, tokens)
        got = self.bound('define_assembly', 'define_assembly', '--roles', 'houses.a=structure,n=12')
        self.assertEqual(got['roles'], {'houses.a': 'structure', 'n': '12'})  # no nesting, no value typing
        self.assertEqual(self.bound('define_assembly', 'define_assembly', '--roles={}')['roles'], {})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'roles.json'
            path.write_text(json.dumps(want))
            self.assertEqual(self.bound('define_assembly', 'define_assembly', '--roles', f'@{path}')['roles'], want)

    def test_bad_text_dict(self):
        self.refused('define_assembly', 'define_assembly', '--roles', 'base', needle='expects key=value')
        self.refused('define_assembly', 'define_assembly', '--roles', '', needle='expects key=value')
        self.refused('define_assembly', 'define_assembly', '--roles', '=x', needle='expects key=value')
        self.refused('define_assembly', 'define_assembly', '--roles', 'a=x,a=y', needle='a is given twice')
        self.refused('define_assembly', 'define_assembly', '--roles', 'a=x', '--roles', 'a=y', needle='a is given twice')
        self.refused('define_assembly', 'define_assembly', '--roles', '{"a": 1}', needle='expects text values')
        self.refused('define_assembly', 'define_assembly', '--roles', '["a"]', needle='expects key=value')
        self.refused('define_assembly', 'define_assembly', '--roles', '{"a": ', needle='invalid JSON')

    def test_free_dict_reads_json_values_and_nests_dotted_keys(self):
        got = self.bound('check_performance', 'check_performance', 'MyMap', '--budgets',
                         'placed_objects=6000,dynamic_lights.warn=8')
        self.assertEqual(got, {'name': 'MyMap', 'budgets': {'placed_objects': 6000, 'dynamic_lights': {'warn': 8}}})
        got = self.bound('check_performance', 'check_performance', '--budgets', 'dynamic_lights.warn=8', '--budgets',
                         'dynamic_lights.high=20.5', '--budgets', 'note=x y,flag=true,none=null')
        self.assertEqual(got['budgets'], {'dynamic_lights': {'warn': 8, 'high': 20.5}, 'note': 'x y', 'flag': True,
                                          'none': None})
        got = self.bound('check_performance', 'check_performance', '--budgets', '{"dynamic_lights": {"warn": 8}}')
        self.assertEqual(got['budgets'], {'dynamic_lights': {'warn': 8}})
        self.refused('check_performance', 'check_performance', '--budgets', 'a=1,a.b=2', needle='a is given twice')
        self.refused('check_performance', 'check_performance', '--budgets', 'a.b=1', '--budgets', 'a.b=2',
                     needle='a.b is given twice')
        self.refused('check_performance', 'check_performance', '--budgets', 'a..b=1', needle='expects key=value')
        self.refused('check_performance', 'check_performance', '--budgets', '[1]', needle='expects key=value')

    def test_list_of_dicts_is_json(self):
        pose = {'x': 0, 'y': 0, 'z': 30, 'yaw': 90, 'pitch': -20}
        got = self.bound('check_performance', 'check_performance', '--measure', '--poses', json.dumps([pose, pose]))
        self.assertEqual(got, {'measure': True, 'poses': [pose, pose]})
        got = self.bound('check_performance', 'check_performance', '--poses', json.dumps(pose), '--poses', json.dumps(pose))
        self.assertEqual(got['poses'], [pose, pose])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'poses.json'
            path.write_text(json.dumps([pose]))
            self.assertEqual(self.bound('check_performance', 'check_performance', '--poses', f'@{path}')['poses'], [pose])
        self.refused('check_performance', 'check_performance', '--poses', '[1, 2]', needle='JSON list of objects')
        self.refused('check_performance', 'check_performance', '--poses', 'x=1', needle='invalid JSON')


class GrammarTests(Case):
    def test_command_names_accept_dashes_and_aliases(self):
        self.assertEqual(self.bound('search_assets', 'search-assets', 'house'), {'query': 'house'})
        self.assertEqual(self.bound('agent_log', 'log'), {})
        self.assertEqual(self.bound('agent_unload', 'unload'), {})
        self.assertEqual(self.bound('agent_exec', 'exec', 'mapinfo'), {'command': 'mapinfo'})
        self.assertEqual(self.bound('agent_exec', 'agent-exec', 'mapinfo'), {'command': 'mapinfo'})

    def test_unknown_command_names_the_nearest(self):
        for argv, hint in ((['serch_assets', 'x'], 'did you mean search_assets?'), (['ex', 'mapinfo'], ''),
                           (['get-asset', 'x'], 'did you mean get_assets?'), (['bogus'], '')):
            code, out, err = run(*argv)
            self.assertEqual((code, out), (2, ''), argv)
            self.assertEqual(len(err.splitlines()), 1, err)
            self.assertTrue(err.startswith(f'error: unknown command {argv[0]!r}'), err)
            self.assertIn(hint, err)
            self.assertIn("'./uc help' lists the commands", err)
        self.assertNotIn('did you mean', run('bogus')[2])

    def test_option_names_are_exact(self):
        line = self.refused('search_assets', 'search_assets', '--limt', '3', needle='unknown option --limt')
        self.assertIn('did you mean --limit?', line)
        self.refused('search_assets', 'search_assets', '--lim', '3', needle='unknown option --lim')
        self.refused('search_assets', 'search_assets', '--Limit', '3', needle='unknown option --Limit')
        self.refused('search_assets', 'search_assets', '--display_name', 'x', needle='unknown option')
        self.assertEqual(self.bound('agent_exec', 'exec', '--max-lines', '9', 'mapinfo')['max_lines'], 9)

    def test_only_double_dash_letter_is_an_option(self):
        self.assertEqual(self.bound('list_groups', 'list_groups', '--', '--weird'), {'prefix': '--weird'})
        self.assertEqual(self.bound('save_live', 'save_live', '--name=--x', '--display_name=-- y'),
                         {'name': '--x', 'display_name': '-- y'})
        self.assertEqual(self.bound('list_groups', 'list_groups', '--prefix', '-x'), {'prefix': '-x'})
        self.assertEqual(self.bound('list_groups', 'list_groups', '--', '-x', ), {'prefix': '-x'})
        self.assertEqual(self.bound('list_groups', 'list_groups', '--', 'limit=3'), {'prefix': 'limit=3'})
        self.assertEqual(self.bound('list_groups', 'list_groups', '--limit', '5', '--', 'a'), {'limit': 5, 'prefix': 'a'})
        self.refused('list_groups', 'list_groups', '-limit', '3', needle="'-limit' looks like an option")
        self.refused('list_groups', 'list_groups', '-x', needle='looks like an option')
        self.refused('list_groups', 'list_groups', '--', 'a', 'b', needle='at most 1 (PREFIX)')

    def test_a_value_option_never_swallows_an_option(self):
        self.refused('search_assets', 'search_assets', '--category', '--limit', '3', needle='--category needs a value')
        self.refused('search_assets', 'search_assets', '--category', '--', 'x', needle='--category needs a value')
        self.refused('search_assets', 'search_assets', 'x', '--category', needle='--category needs a value')
        line = self.refused('consolidate_scenery', 'consolidate_scenery', '--rows', '--apply')
        self.assertIn('--rows needs a value but --apply follows', line)

    def test_duplicates_are_refused(self):
        self.refused('search_assets', 'search_assets', '--limit', '1', '--limit', '2', needle='--limit given twice')
        self.refused('search_assets', 'search_assets', 'house', '--query', 'barn', needle='QUERY and --query both set')
        self.refused('search_assets', 'search_assets', '--limit=1', '--limit', '1', needle='--limit given twice')

    def test_surplus_positionals_are_never_joined_or_dropped(self):
        self.refused('status', 'status', 'now', needle='status takes no positional arguments')
        self.refused('search_assets', 'search_assets', 'big', 'house', needle="unexpected 'house'")
        self.refused('load_live', 'load_live', 'My', 'Map', needle="unexpected 'Map'")
        self.refused('inspect_map', 'inspect_map', needle='missing NAME')

    def test_region_expands_to_the_four_bounds(self):
        want = {'x0': -1.0, 'y0': 2.5, 'x1': 3.0, 'y1': 4.0}
        for name in ('check_collisions', 'check_foundations', 'check_environment', 'check_performance',
                     'consolidate_scenery'):
            self.assertEqual(self.bound(name, name, '--region', '-1,2.5,3,4'), want, name)
            self.assertEqual(self.bound(name, name, '--x0', '-1', '--y0', '2.5', '--x1', '3', '--y1', '4'), want)
        self.refused('check_collisions', 'check_collisions', '--region', '1,2,3', needle='expects x0,y0,x1,y1')
        self.refused('check_collisions', 'check_collisions', '--region', '1,2,3,4', '--x1', '9', needle='both set x1')
        self.refused('check_collisions', 'check_collisions', '--region', '1,2,3,4', '--region', '1,2,3,4',
                     needle='--region given twice')
        self.refused('list_groups', 'list_groups', '--region', '1,2,3,4', needle='unknown option --region')
        self.assertEqual({op.name for op in api.OPERATIONS.values() if '--region' in cli._usage(op)},
                         {'check_collisions', 'check_foundations', 'check_environment', 'check_performance',
                          'consolidate_scenery'})


class MisbindingTests(Case):
    """The command lines that bound silently to the wrong thing in the first design."""

    def test_save_live_my_map(self):
        line = self.refused('save_live', 'save_live', 'My', 'Map', needle='save_live takes at most 1 (NAME)')
        self.assertIn("unexpected 'Map'", line)
        self.assertEqual(self.bound('save_live', 'save_live', 'My Map'), {'name': 'My Map'})
        self.assertEqual(self.bound('save_live', 'save_live', 'My', '--display_name', 'Map'),
                         {'name': 'My', 'display_name': 'Map'})

    def test_save_live_name_equals_city(self):
        line = self.refused('save_live', 'save_live', 'name=city')
        self.assertIn("'name=city' is not an argument of save_live", line)
        self.assertIn('write --name city', line)
        self.assertIn('write --display_name My Map', self.refused('save_live', 'save_live', 'display_name=My Map'))
        self.assertIn('write --preview=false', self.refused('undo_edit', 'undo_edit', 't1', 'preview=false'))
        self.assertIn('write --display_name x', self.refused('save_live', 'save_live', 'display-name=x'))
        # a misspelt parameter must not become the map name either
        line = self.refused('save_live', 'save_live', 'nam=city', needle="save_live has no parameter 'nam'")
        self.assertIn('did you mean name?', line)
        self.refused('save_live', 'save_live', 'limit=3', needle="no parameter 'limit'")
        self.assertEqual(self.bound('save_live', 'save_live', '--', 'a=b'), {'name': 'a=b'})
        self.assertEqual(self.bound('save_live', 'save_live', '--name', 'a=b'), {'name': 'a=b'})
        self.assertEqual(self.bound('check_route_clearance', 'check_route_clearance', 'map:City=2'),
                         {'source': 'map:City=2'})

    def test_options_typed_after_an_exec_command(self):
        for tokens in (['mapinfo', '--timeout', '5'], ['cells', '5', '*', '--max_lines', '200'],
                       ['mapinfo', '--max-lines=9'], ['mapinfo', '--pick', 'x'], ['mapinfo', '-h'],
                       ['mapinfo', '--help'], ['spawn', 'NewRow_1', '0', '0', '0', '--version']):
            line = self.refused('agent_exec', 'exec', *tokens, needle='put options before the command')
            self.assertIn(tokens[-1] if tokens[-1].startswith('-') else tokens[-2], line)
        with fake('agent_exec') as calls:
            code, out, err = run('exec', 'mapinfo', '--json')
        self.assertEqual((code, calls), (2, []))
        self.assertIn("put options before the command: --json came after 'mapinfo'", json.loads(out)['error'])
        self.assertTrue(err.startswith('error: put options before the command: --json'))
        self.assertEqual(self.bound('agent_exec', 'exec', '--timeout', '5', 'mapinfo'),
                         {'timeout': 5.0, 'command': 'mapinfo'})

    def test_preview_false_as_two_tokens_is_not_a_name(self):
        self.assertEqual(self.bound('consolidate_scenery', 'consolidate_scenery', '--preview', 'false'),
                         {'preview': False})
        self.assertEqual(self.bound('consolidate_scenery', 'consolidate_scenery', '--preview=False'), {'preview': False})


class RestTests(Case):
    def test_the_command_is_everything_after_the_options_joined_verbatim(self):
        got = self.bound('agent_exec', 'exec', 'spawn', 'NewRow_1', '1500', '-2000', '0', '0', '90', '0', '1', '1',
                         '1', 'group=yard')
        self.assertEqual(got, {'command': 'spawn NewRow_1 1500 -2000 0 0 90 0 1 1 1 group=yard'})
        self.assertEqual(self.bound('agent_exec', 'exec', '--max_lines', '200', 'cells 5 *'),
                         {'max_lines': 200, 'command': 'cells 5 *'})
        self.assertEqual(self.bound('agent_exec', 'exec', 'transform', '1A2B', '-', '-', '-5', '-', '-', '-'),
                         {'command': 'transform 1A2B - - -5 - - -'})
        self.assertEqual(self.bound('agent_exec', 'exec', 'max_lines=3', 'x  y'), {'command': 'max_lines=3 x  y'})
        self.assertEqual(self.bound('agent_exec', 'exec', 'console', 'stat', '--', '-flag', '--', '--9'),
                         {'command': 'console stat -- -flag -- --9'})  # only --<letter> is an option
        self.assertEqual(self.bound('agent_exec', 'exec', 'spawn X 0 0 0 --foo'), {'command': 'spawn X 0 0 0 --foo'})

    def test_an_unknown_option_after_the_command_is_not_sent_to_the_game(self):
        for tokens, typed, hint in ((['actors', '--max_line', '200'], '--max_line', '; did you mean --max_lines?'),
                                    (['spawn', 'NewRow_1', '0', '0', '0', '--time-out', '5'], '--time-out',
                                     '; did you mean --timeout?'),
                                    (['mapinfo', '--apply'], '--apply', None),
                                    (['destroy', 'last', '--preview'], '--preview', None),
                                    (['console', 'r.foo', '--bar=1'], '--bar', ''),
                                    (['console', 'stat', '--', '--unknownflag'], '--unknownflag', '')):
            line = self.refused('agent_exec', 'exec', *tokens)
            late = next(token for token in tokens if token.startswith(typed))
            start = (f"error: put options before the command: {late} came after {tokens[0]!r} (a command that itself "
                     f"contains {late} goes after --); agent_exec has no option {typed}")
            self.assertTrue(line.startswith(start), line)
            if hint is not None:
                self.assertEqual(line, start + hint)
        self.assertEqual(self.bound('agent_exec', 'exec', '--', 'actors', '--max_line', '200'),
                         {'command': 'actors --max_line 200'})
        self.assertEqual(self.bound('agent_exec', 'exec', '--max_lines', '200', 'actors'),
                         {'max_lines': 200, 'command': 'actors'})

    def test_double_dash_opens_a_verbatim_remainder(self):
        self.assertEqual(self.bound('agent_exec', 'exec', '--', 'console', 'x', '--json', '--timeout', '5', '-h'),
                         {'command': 'console x --json --timeout 5 -h'})
        self.assertEqual(self.bound('agent_exec', 'exec', '--timeout', '9', '--', '--odd'),
                         {'timeout': 9.0, 'command': '--odd'})

    def test_bad_rest_lines(self):
        self.refused('agent_exec', 'exec', 'destroy 1A\ndestroy 2B', needle='one command per exec; use batch')
        self.refused('agent_exec', 'exec', 'mapinfo', 'a\rb', needle='one command per exec; use batch')
        self.refused('agent_exec', 'exec', needle='missing COMMAND')
        self.refused('agent_exec', 'exec', '--', needle='missing COMMAND')
        self.refused('agent_exec', 'exec', '--timeout', '5', needle='missing COMMAND')
        self.refused('agent_exec', 'exec', ' ', needle='the command is empty')
        self.refused('agent_exec', 'exec', '--bogus', 'mapinfo', needle='unknown option --bogus')
        self.refused('agent_exec', 'exec', '--command', 'mapinfo', 'gates', needle='not both')
        self.refused('agent_exec', 'exec', '--timeout', 'soon', 'mapinfo', needle='--timeout expects')
        self.assertEqual(self.bound('agent_exec', 'exec', '--command', 'mapinfo'), {'command': 'mapinfo'})


class BatchTests(Case):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.plan = Path(self.tmp.name) / 'plan.txt'

    def test_file_lines_are_passed_unsplit_with_their_line_numbers(self):
        self.plan.write_text('# yard\nspawn NewRow_1 0 0 0\n\n  cells 5 a,b  \r\ndrop last\n')
        got = self.bound('agent_batch', 'batch', str(self.plan), '--timeout', '600')
        self.assertEqual(got, {'commands': ['# yard', 'spawn NewRow_1 0 0 0', '', '  cells 5 a,b  ', 'drop last'],
                               '_lines': [1, 2, 3, 4, 5], 'timeout': 600.0})
        self.plan.write_text('mapinfo')  # no final newline
        self.assertEqual(self.bound('agent_batch', 'agent_batch', '--no-quiet', str(self.plan)),
                         {'commands': ['mapinfo'], '_lines': [1], 'quiet': False})

    def test_stdin_only_with_an_explicit_dash(self):
        got = self.bound('agent_batch', 'batch', '-', stdin='spawn NewRow_1 0 0 0\ndrop last\n')
        self.assertEqual(got, {'commands': ['spawn NewRow_1 0 0 0', 'drop last'], '_lines': [1, 2]})
        got = self.bound('agent_batch', 'batch', '-', '--timeout', '600', stdin='mapinfo\n')
        self.assertEqual(got['timeout'], 600.0)

        class Unread(io.StringIO):
            def read(self, *args):
                raise AssertionError('stdin was read without -')

        with fake('agent_batch') as calls, patch.object(sys, 'stdin', Unread('mapinfo\n')), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(cli.main(['batch']), 2)
        self.assertEqual(calls, [])
        self.assertIn('no command file: give FILE, or - to read stdin', err.getvalue())
        self.assertIn(cli.FILE_HINT, err.getvalue())

    def test_commands_option(self):
        got = self.bound('agent_batch', 'batch', '--commands', 'spawn NewRow_1 0 0 0', '--commands', 'cells 5 a,b')
        self.assertEqual(got, {'commands': ['spawn NewRow_1 0 0 0', 'cells 5 a,b']})
        self.plan.write_text('mapinfo\n# c\ngates\n')
        got = self.bound('agent_batch', 'batch', '--commands', f'@{self.plan}')
        self.assertEqual(got, {'commands': ['mapinfo', '# c', 'gates'], '_lines': [1, 2, 3]})
        self.refused('agent_batch', 'batch', '--commands', 'a\nb', needle='line break')

    def test_commands_dash_is_not_sent_as_a_command(self):
        line = self.refused('agent_batch', 'batch', '--commands', '-', stdin='spawn A 0 0 0\n')
        self.assertEqual(line, "error: --commands - would send the command '-'; to read stdin write ./uc batch - or "
                               "--commands @-")
        self.refused('agent_batch', 'batch', '--commands', 'mapinfo', '--commands=-', needle="would send the command '-'")
        got = self.bound('agent_batch', 'batch', '--commands', '@-', stdin='spawn A 0 0 0\n')
        self.assertEqual(got, {'commands': ['spawn A 0 0 0'], '_lines': [1]})

    def test_wrong_batch_lines(self):
        self.plan.write_text('mapinfo\n')
        self.refused('agent_batch', 'batch', needle='no command file')
        self.refused('agent_batch', 'batch', str(self.plan), '--commands', 'gates', needle='not both')
        line = self.refused('agent_batch', 'batch', 'mapinfo', needle="cannot read 'mapinfo'")
        self.assertTrue(line.endswith('batch takes a command file; use ./uc exec for one command'), line)
        line = self.refused('agent_batch', 'batch', 'spawn', 'NewRow_1', '0', '0', '0', needle='expected one command file')
        self.assertTrue(line.endswith('batch takes a command file; use ./uc exec for one command'), line)
        self.plan.write_text('# nothing\n\n')
        self.refused('agent_batch', 'batch', str(self.plan), needle='holds no commands')
        self.plan.write_text('')
        self.refused('agent_batch', 'batch', str(self.plan), needle='holds no commands')
        self.plan.write_bytes(b'spawn \xff\n')
        self.refused('agent_batch', 'batch', str(self.plan), needle='not UTF-8')
        self.plan.write_text('a\rb\n')
        self.refused('agent_batch', 'batch', str(self.plan), needle='line break')

    def test_failed_lines_name_the_source_line_and_exit_3(self):
        self.plan.write_text('# yard\nspawn NewRow_1 0 0 0\n\nspawn Nope 1 2 3\n')
        replies = ({1: ['error unknown row Nope']}, 'batch lines=2 ok=1 failed=1')
        with patch.object(agent, 'batch', return_value=replies) as sent:
            code, out, err = run('batch', str(self.plan))
        self.assertEqual(sent.call_args.args[0], ['spawn NewRow_1 0 0 0', 'spawn Nope 1 2 3'])
        self.assertEqual(code, 3)
        self.assertEqual(out.splitlines(), ['batch lines=2 ok=1 failed=1 sent=2', 'failures:',
                                            '#1 L4 error unknown row Nope << spawn Nope 1 2 3', api.BATCH_RETRY])
        self.assertEqual(err, 'error: batch applied 1/2 lines, 1 failed (#1); do not re-run the whole file\n')

    def test_batch_timeout_says_the_state_is_unknown(self):
        self.plan.write_text('mapinfo\n')
        timeout = agent.AgentTimeout("the agent did not answer 'batch' within 5s; the command was sent and may still run")
        with patch.object(agent, 'batch', side_effect=timeout):
            code, out, err = run('batch', str(self.plan))
        self.assertEqual((code, out), (1, ''))
        self.assertEqual(len(err.splitlines()), 1)
        self.assertIn('batch state UNKNOWN', err)
        self.assertEqual(err.count('before retrying'), 1)

    def test_a_batch_job_that_faulted_says_the_state_is_unknown(self):
        self.plan.write_text('spawn A 0 0 0\ncrash\nspawn C 0 0 0\n')
        reply = {'seq': 1002, 'ok': False, 'head': 'ok batch', 'lines': [], 'ended': False}  # a fault in a quiet batch
        failed = agent.AgentError(agent._failure(reply))
        failed.reply = reply
        with patch.object(agent, 'batch', side_effect=failed):
            code, out, err = run('batch', str(self.plan))
        self.assertEqual((code, out), (1, ''))
        self.assertEqual(err, 'error: batch job failed after 0 reply lines: agent job ended without an end marker '
                              f'(fault?); {api.BATCH_UNKNOWN}\n')
        with patch.object(agent, 'batch', side_effect=agent.AgentNotSent('Uncrashed is not running')):
            self.assertEqual(run('batch', str(self.plan))[2], 'error: Uncrashed is not running\n')


# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------


class HelpTests(Case):
    def test_overview(self):
        code, out, err = run()
        self.assertEqual((code, err), (0, ''))
        self.assertLess(len(out), 4500)
        for argv in (['help'], ['--help'], ['-h'], ['help', 'commands']):
            self.assertEqual(run(*argv), (0, out, ''), argv)
        lines = out.splitlines()
        self.assertEqual(lines[0], f'Uncrashed track editor CLI {uncrashed_mcp.__version__}')
        self.assertEqual(lines[1], 'usage: ./uc [--json] [--pick PATHS] <command> [positionals] [--param value ...]')
        head = '\n'.join(lines[:lines.index('')])
        for needle in ('metres and degrees', 'centimetres', '--json', '--pick', '0 ok', '2 usage error, nothing ran',
                       '3 ran but reported failures', '"error: ..."', './uc help <command>', './uc help guide',
                       './uc help verbs', './uc help output', './uc last', './uc mcp', '@file', '--no-flag',
                       '! = changes the scene or files at once: no preview.',
                       '~ = changes them only with --apply, previews otherwise.'):
            self.assertIn(needle, head)
        self.assertNotIn('edits preview unless --apply', head)  # untrue for exec, batch, load_live, save_live, unload
        headings = ['No game needed:', 'A saved map NAME without the game, otherwise the open editor:',
                    'Game open in the track editor:']
        self.assertEqual([line for line in lines if line.endswith(':') and not line.startswith(('!', ' '))], headings)
        group = {'No game needed:': 'no', headings[1]: 'mixed', headings[2]: 'yes'}
        listed, current = {}, None
        for line in lines[lines.index('') + 1:]:
            if line in group:
                current = group[line]
                continue
            self.assertRegex(line, r'^[!~ ] [a-z_|]+( \S+)*  \S')
            name = line[2:].split()[0].split('|')[0]
            listed[name] = (line[0], current, line)
        self.assertEqual(list(listed), sorted(api.OPERATIONS, key=lambda n: ['no', 'mixed', 'yes'].index(api.OPERATIONS[n].live)))
        for name, (mark, live, line) in listed.items():
            op = api.OPERATIONS[name]
            previews = any(p.name == 'preview' for p in op.params())
            self.assertEqual(mark, ' ' if op.effect != 'write' else '~' if previews else '!', name)
            self.assertEqual(live, op.live, name)
            self.assertTrue(line.endswith(f'  {op.summary}'), line)
            # the mark is true: ~ commands take --apply, ! commands refuse it and act at once
            self.assertEqual('[--apply]' in cli._usage(op), mark == '~', name)
        marks = {name: mark for name, (mark, _live, _line) in listed.items()}
        self.assertEqual({name for name, mark in marks.items() if mark == '!'},
                         {'load_live', 'save_live', 'agent_exec', 'agent_batch', 'agent_unload'})
        self.assertEqual({name for name, mark in marks.items() if mark == '~'},
                         {'clearing_zones', 'edit_group', 'undo_edit', 'define_assembly', 'set_collision',
                          'undo_collision', 'consolidate_scenery'})

    def test_command_help_structure(self):
        for name, op in api.OPERATIONS.items():
            with self.subTest(op=name):
                code, out, err = run('help', name)
                self.assertEqual((code, err), (0, ''))
                lines = out.splitlines()
                self.assertEqual(lines[0], f'{name}: {op.summary}')
                self.assertEqual(lines[1], f'usage: {cli._usage(op)}')
                for param in op.params():
                    if param.name in (*op.positional, op.rest):
                        self.assertRegex(lines[1], rf'(?<![\w-]){param.name.upper()}(?!\w)')
                    elif param.name in cli.REGION:
                        self.assertIn('[--region x0,y0,x1,y1]', lines[1])
                    elif param.name == 'preview':
                        self.assertIn('[--apply]', lines[1])
                    else:
                        self.assertRegex(lines[1], rf'[\[ |<]--(no-)?{param.name}(?!\w)')
                self.assertEqual(f'alias: {", ".join(op.aliases)}' in lines, bool(op.aliases))
                self.assertEqual(f'caution: {op.caution}' in lines, bool(op.caution))
                self.assertEqual(sum(line.startswith('needs: ') for line in lines), 1)
                needs = next(line for line in lines if line.startswith('needs: '))
                self.assertEqual(needs.endswith('; changes the scene or files only with --apply, previews otherwise'),
                                 '[--apply]' in lines[1])
                self.assertEqual(needs.endswith('; changes the scene or files at once (no preview)'),
                                 op.effect == 'write' and '[--apply]' not in lines[1])
                self.assertEqual(needs.endswith('; read-only'), op.effect == 'read')
                for example in op.examples:
                    self.assertIn(f'  {example}', lines)
                legend = lines.index(f'({cli.LEGEND})')
                self.assertEqual(lines[legend + 1], '')
                doc = inspect.cleandoc(op.fn.__doc__).replace('server instructions', './uc help verbs')
                self.assertEqual('\n'.join(lines[legend + 2:]), doc)
                self.assertNotIn('server instructions', out)

    def test_usage_lines(self):
        usage = {name: cli._usage(op) for name, op in api.OPERATIONS.items()}
        self.assertEqual(usage['status'], './uc status')
        self.assertEqual(usage['save_live'], './uc save_live [NAME:str] [--display_name str]')
        self.assertEqual(usage['move_camera'], './uc move_camera X:num Y:num Z:num YAW:num [PITCH:num=-20] [--no-hold]')
        for line in usage.values():
            self.assertNotIn('\n', line)

    def test_help_flag_after_a_command_runs_nothing(self):
        with fake('save_live') as calls:
            expected = run('help', 'save_live')
            for argv in (['save_live', '--help'], ['save_live', '-h'], ['--help', 'save_live'], ['save-live', 'x', '-h'],
                         ['save_live', 'My', 'Map', '--help'], ['help', 'save-live']):
                self.assertEqual(run(*argv), expected, argv)
            self.assertEqual(run('exec', '--help'), run('help', 'agent_exec'))
            self.assertEqual(run('help', 'exec'), run('help', 'agent_exec'))
        self.assertEqual(calls, [])
        self.assertIn('overwrites the saved map', expected[1])

    def test_topics(self):
        self.assertEqual(run('help', 'verbs'), (0, api.VERBS + '\n', ''))
        self.assertEqual(run('help', 'guide'), (0, api.GUIDE + '\n', ''))
        self.assertEqual(run('help', 'output'), (0, cli.OUTPUT + '\n', ''))
        self.assertEqual(run('help', '--json', 'verbs')[1], api.VERBS + '\n')  # help is never JSON
        self.assertIn('last finished agent reply', run('help', 'last')[1])
        self.assertIn('MCP stdio server', run('mcp', '--help')[1])

    def test_the_output_topic_describes_what_the_renderer_writes(self):
        from uncrashed_mcp import render
        text = cli.OUTPUT
        rows = [{'id': f'NewRow_{i}', 'name': f'House {i}', 'kind': 'mesh', 'category': 'Big structure'}
                for i in range(render.HOIST_ROWS)]
        shown = render.render({'assets': rows}).splitlines()
        self.assertEqual(shown[:2], [f'assets[{len(rows)}]{{id|name}}:',
                                     f'  {render.HOIST} kind=mesh category="Big structure"'])  # the form the help shows
        self.assertNotIn(render.HOIST, render.render({'assets': rows[1:]}))  # one row fewer: plain columns
        self.assertIn(f'\n    {render.HOIST} kind=mesh   fields with the same value in every row', text)
        self.assertIn(f'tables of {render.HOIST_ROWS} or more rows only', text)
        wide = [dict(row, note=str(i) + 'x' * 90) for i, row in enumerate(rows)]  # a cell too long for a table
        shown = render.render({'assets': wide}).splitlines()
        self.assertEqual(shown[1], f'  {render.HOIST} kind=mesh category="Big structure"')  # the list form
        self.assertTrue(shown[2].startswith('  - {id=NewRow_0 '), shown[2])
        self.assertIn(f'\n  {render.HOIST} k=v      before - items: fields every item has, left out of each item', text)
        self.assertIn('key: value lines indented under it are more fields of that same item', text)
        self.assertIn('nothing is rounded or dropped; only the order of fields inside rows under an `(every row also '
                      'has)` line may differ', ' '.join(text.split()))
        self.assertIn("exec / batch / log print plain text (the agent's reply), not fields: a failure shows as exit "
                      "code 3 plus the stderr `error:` line.", ' '.join(text.split()))
        self.assertIn('`(empty reply)`', text)
        self.assertIn('NaN / Infinity become null', text)

    def test_unknown_topic(self):
        code, out, err = run('help', 'serch_assets')
        self.assertEqual((code, out), (2, ''))
        self.assertEqual(len(err.splitlines()), 1)
        self.assertIn("no help topic 'serch_assets'; did you mean search_assets?", err)
        self.assertEqual(run('help', 'a', 'b')[0], 2)

    def test_help_does_not_load_the_catalogue(self):
        with patch.object(api, '_catalog', side_effect=AssertionError('catalogue loaded for help')), \
                patch.object(api, 'load_catalog', side_effect=AssertionError('catalogue loaded for help')):
            for argv in ([], ['help', 'search_assets'], ['help', 'guide'], ['search_assets', '--help'], ['--version']):
                self.assertEqual(run(*argv)[0], 0, argv)

    def test_version(self):
        want = (0, f'uncrashed {uncrashed_mcp.__version__}\n', '')
        with fake('status') as calls:
            self.assertEqual(run('--version'), want)
            self.assertEqual(run('status', '--version'), want)
            self.assertEqual(run('--version', 'status'), want)
        self.assertEqual(calls, [])

    def test_global_flags_before_the_command(self):
        self.assertEqual(run('--limit', '3', 'search_assets')[0], 2)
        self.assertIn('before the command', run('--limit', '3', 'search_assets')[2])
        self.assertEqual(run('--', 'status')[0], 2)
        self.assertEqual(run('--json=1', 'status')[0], 2)
        self.assertEqual(run('--pick')[0], 2)


# ---------------------------------------------------------------------------
# results, exit codes, output
# ---------------------------------------------------------------------------


ASSETS = {'count': 2, 'assets': [{'id': 'NewRow_1', 'name': 'a|b', 'size': {'x': 1.0, 'y': 2.0}},
                                {'id': 'NewRow_2', 'name': '12', 'size': {'x': 3.0, 'y': 4.0}}],
          'scope': 'catalogue only', 'missing': []}


class ExitCodeTests(Case):
    def test_exit_0_prints_the_rendered_value_once(self):
        with fake('search_assets', ASSETS):
            code, out, err = run('search_assets', 'x')
        self.assertEqual((code, err), (0, ''))
        self.assertEqual(parse(out), ASSETS)
        self.assertTrue(out.endswith('\n') and not out.endswith('\n\n'))
        self.assertIn('assets[2]{id|name|size.x|size.y}:', out)

    def test_values_are_normalised_through_json(self):
        with fake('status', {'pair': (1, 2), 'path': 'x'}):
            code, out, err = run('status')
        self.assertEqual(out, 'pair: [1,2]\npath: x\n')

    def test_exit_1_when_the_operation_raises(self):
        for error, line in ((ValueError('provide 1 to 256 [x, y] points'), 'provide 1 to 256 [x, y] points'),
                            (FileNotFoundError("no local map 'x' in /maps"), "no local map 'x' in /maps"),
                            (KeyError('aabb'), "KeyError: 'aabb'"),
                            (TypeError('bad operand'), 'TypeError: bad operand'),
                            (IndexError('list index out of range'), 'IndexError: list index out of range'),
                            (AttributeError("no attribute 'x'"), "AttributeError: no attribute 'x'"),
                            (RuntimeError('first line\n  second line\n'), 'first line second line'),
                            (RuntimeError(''), 'RuntimeError'),
                            (agent.AgentError('Uncrashed is not running'), 'Uncrashed is not running')):
            with fake('inspect_map', error) as calls:
                code, out, err = run('inspect_map', 'x')
            self.assertEqual((code, out, err), (1, '', f'error: {line}\n'), error)
            self.assertEqual(len(calls), 1)

    def test_a_reply_timeout_says_the_command_may_still_run(self):
        sent = agent.AgentTimeout("the agent did not answer 'load x' within 30s; the command was sent and may still run")
        with fake('load_live', sent):
            code, out, err = run('load_live', 'x')
        self.assertEqual(code, 1)
        self.assertEqual(err, "error: the agent did not answer 'load x' within 30s; the command was sent and may still "
                              "run: check './uc last' / './uc exec mapinfo' before retrying\n")
        with fake('load_live', agent.AgentTimeout('no reply')):
            err = run('load_live', 'x')[2]
        self.assertEqual(err, "error: no reply; the command was sent and may still run: check './uc last' / "
                              "'./uc exec mapinfo' before retrying\n")
        for kind in (agent.AgentError, agent.AgentNotSent):
            with fake('load_live', kind('agent busy: lock held by pid 5 (uc) for 60s; nothing was sent')):
                err = run('load_live', 'x')[2]
            self.assertEqual(err, 'error: agent busy: lock held by pid 5 (uc) for 60s; nothing was sent\n')

    def test_an_agent_lost_after_sending_gets_the_recovery_hint_and_nothing_sent_does_not(self):
        lost = "the agent is not loaded (unloaded, or the game closed); 'load x' was sent before that and may have run"
        with fake('load_live', agent.AgentError(lost)):
            err = run('load_live', 'x')[2]
        self.assertEqual(err, f"error: {lost}: check './uc last' / './uc exec mapinfo' before retrying\n")
        for text in ('the agent is not loaded (unloaded, or the game closed); nothing was sent',
                     'odd wording that ends in may have run'):
            with fake('load_live', agent.AgentNotSent(text)):
                self.assertEqual(run('load_live', 'x')[2], f'error: {text}\n')
        answered = agent.AgentError('no row X; it may have run')  # the agent answered: its reply is the state
        answered.reply = {'seq': 1, 'ok': False, 'head': 'ok spawn', 'lines': [], 'ended': True}
        with fake('load_live', answered):
            self.assertEqual(run('load_live', 'x')[2], 'error: no row X; it may have run\n')

    def test_no_traceback_unless_debugging(self):
        with fake('status', RuntimeError('boom')):
            self.assertEqual(run('status')[2], 'error: boom\n')
            self.assertEqual(run('status', env={'UNCRASHED_DEBUG': '0'})[2], 'error: boom\n')
            code, out, err = run('status', env={'UNCRASHED_DEBUG': '1'})
        self.assertEqual(code, 1)
        self.assertIn('Traceback (most recent call last)', err)
        self.assertTrue(err.endswith('error: boom\n'))

    def test_exit_2_prints_one_error_line_and_the_usage_line(self):
        line = self.refused('search_assets', 'search_assets', '--limit', 'x')
        self.assertEqual(line, "error: --limit expects an integer, got 'x'")

    def test_exit_3_when_a_text_reply_reports_failure(self):
        reply = api.Text('failed spawn\nerror unknown row Nope', False, 'the agent reported: error unknown row Nope')
        with fake('agent_exec', reply):
            code, out, err = run('exec', 'spawn', 'Nope', '0', '0', '0')
        self.assertEqual((code, out), (3, 'failed spawn\nerror unknown row Nope\n'))
        self.assertEqual(err, 'error: the agent reported: error unknown row Nope\n')
        with fake('agent_exec', api.Text('failed x', False)):
            self.assertEqual(run('exec', 'x')[2], 'error: the agent reported a failure\n')
        with fake('agent_exec', api.Text('ok mapinfo\nfolder=City')):
            self.assertEqual(run('exec', 'mapinfo'), (0, 'ok mapinfo\nfolder=City\n', ''))

    def test_an_empty_text_reply_is_told_apart_from_a_failure(self):
        with fake('agent_log', api.Text('')):
            self.assertEqual(run('log'), (0, '', '(empty reply)\n'))
            self.assertEqual(run('log', '--json'), (0, '{"ok":true,"text":""}\n', ''))  # stdout is not empty here
        with fake('agent_exec', api.Text('', False, 'the agent reported: x')):
            self.assertEqual(run('exec', 'x'), (3, '', '(empty reply)\nerror: the agent reported: x\n'))
        with tempfile.TemporaryDirectory() as tmp, patch.object(agent, 'LOG', Path(tmp) / 'agent.log'):
            self.assertEqual(run('log'), (0, f"(agent log is missing: {Path(tmp) / 'agent.log'})\n", ''))

    def test_exit_3_through_the_real_exec_operation(self):
        reply = {'seq': 7, 'ok': False, 'head': 'spawn', 'lines': ['error unknown row Nope'], 'ended': True}
        with patch.object(agent, 'execute', return_value=reply) as sent:
            code, out, err = run('exec', '--timeout', '5', 'spawn', 'Nope', '0', '0', '0')
        sent.assert_called_once_with('spawn Nope 0 0 0', 5.0)
        self.assertEqual((code, out), (3, 'failed spawn\nerror unknown row Nope\n'))
        self.assertEqual(err, 'error: the agent reported: unknown row Nope\n')
        faulted = {'seq': 8, 'ok': False, 'head': 'ok batch', 'lines': [], 'ended': False}
        with patch.object(agent, 'execute', return_value=faulted):
            code, out, err = run('exec', 'batch', 'Z:\\tmp\\b.txt', 'quiet')
        self.assertEqual((code, out), (3, 'failed ok batch\n'))
        self.assertEqual(err, 'error: the agent reported: agent job ended without an end marker (fault?)\n')

    def test_exit_3_for_soft_failures_in_a_normal_reply(self):
        cases = (
            ('load_live', ['load_live', 'City'], {'folder': 'City', 'failed_count': 2, 'failed': ['#3 error x']},
             '2 map lines failed to load'),
            ('load_live', ['load_live', 'City'], {'folder': 'City', 'failed_count': 0, 'camera': {'error': 'no pose'}},
             'camera: no pose'),
            ('save_live', ['save_live'], {'written': False, 'path': '/maps/City/City.json'},
             'the game did not write /maps/City/City.json'),
            ('agent_unload', ['unload'], {'unloaded': False, 'loaded': True, 'error': 'restore conflict'},
             'agent not unloaded: restore conflict'),
            ('set_collision', ['set_collision', 'off', '--group', 'g', '--apply'],
             {'preview': False, 'verified': False, 'mismatched': ['1A'], 'missing_after': 2},
             'collision change applied but not verified: 1 objects in another mode, 2 missing'),
            ('set_collision', ['set_collision', 'off', '--group', 'g', '--apply'],
             {'preview': False, 'verified': False, 'mismatched': [f'{i:X}' for i in range(50)], 'missing_after': 0},
             'collision change applied but not verified: at least 50 objects in another mode, 0 missing'),
        )
        for name, argv, result, reason in cases:
            with self.subTest(name=name, reason=reason), fake(name, result):
                code, out, err = run(*argv)
                self.assertEqual(code, 3)
                self.assertEqual(parse(out), result)  # the reply is still printed in full
                self.assertEqual(err, f'error: {reason}\n')
        for name, argv, result in (('load_live', ['load_live', 'City'], {'folder': 'City', 'failed_count': 0}),
                                   ('save_live', ['save_live'], {'written': True, 'path': 'p'}),
                                   ('agent_unload', ['unload'], {'unloaded': True, 'loaded': False})):
            with fake(name, result):
                self.assertEqual(run(*argv)[0], 0, name)

    def test_a_result_that_cannot_be_printed_says_the_operation_finished(self):
        with fake('status', {'x': 1}), patch.object(cli, '_present', side_effect=TypeError('not JSON')):
            code, out, err = run('status')
        self.assertEqual((code, out), (1, ''))
        self.assertEqual(err, 'error: status finished but its result could not be printed (TypeError: not JSON)\n')

    def test_a_broken_soft_failure_check_is_itself_a_reported_failure(self):
        with fake('save_live', {'written': True}), patch.object(api.OPERATIONS['save_live'], 'check',
                                                                side_effect=KeyError('path')):
            code, out, err = run('save_live')
        self.assertEqual((code, out), (3, 'written: true\n'))
        self.assertIn('could not check the reply for failures', err)

    def test_preview_note(self):
        with fake('edit_group', {'preview': True, 'changes': 3}):
            code, out, err = run('edit_group', 'houses', '--delta', '1,0,0')
        self.assertEqual((code, out), (0, 'preview: true\nchanges: 3\n'))
        self.assertEqual(err, 'preview only: nothing changed (re-run with --apply)\n')
        with fake('edit_group', {'preview': False, 'changes': 3}):
            self.assertEqual(run('edit_group', 'houses', '--delta', '1,0,0', '--apply')[2], '')
        with fake('edit_group', {'preview': True}):
            self.assertEqual(run('--json', 'edit_group', 'g')[2], 'preview only: nothing changed (re-run with --apply)\n')


class JsonTests(Case):
    def test_json_prints_exactly_one_document(self):
        with fake('search_assets', ASSETS):
            for argv in (['--json', 'search_assets', 'x'], ['search_assets', '--json', 'x'], ['search_assets', 'x', '--json']):
                code, out, err = run(*argv)
                self.assertEqual((code, err), (0, ''))
                self.assertEqual(out, json.dumps(ASSETS, separators=(',', ':')) + '\n')
            code, out, err = run('search_assets', 'x', env={'UNCRASHED_FORMAT': 'json'})
            self.assertEqual(json.loads(out), ASSETS)
            self.assertEqual(parse(run('search_assets', 'x', env={'UNCRASHED_FORMAT': 'text'})[1]), ASSETS)

    def test_text_replies_as_json(self):
        with fake('agent_exec', api.Text('ok mapinfo\nfolder=City')):
            self.assertEqual(run('exec', '--json', 'mapinfo'), (0, '{"ok":true,"text":"ok mapinfo\\nfolder=City"}\n', ''))
        with fake('agent_exec', api.Text('failed x', False, 'the agent reported: error y')):
            code, out, err = run('--json', 'exec', 'x')
        self.assertEqual((code, json.loads(out)), (3, {'ok': False, 'text': 'failed x'}))
        self.assertEqual(err, 'error: the agent reported: error y\n')

    def test_failures_as_json(self):
        with fake('inspect_map', FileNotFoundError('no local map')):
            code, out, err = run('--json', 'inspect_map', 'x')
        self.assertEqual((code, out, err), (1, '{"error":"no local map"}\n', 'error: no local map\n'))
        with fake('save_live', {'written': False, 'path': 'p'}):
            code, out, err = run('--json', 'save_live')
        self.assertEqual((code, json.loads(out)), (3, {'written': False, 'path': 'p'}))
        self.assertEqual(err, 'error: the game did not write p\n')

    def test_usage_errors_as_json(self):
        for argv in (['--json', 'search_assets', '--limit', 'x'], ['search_assets', '--limit', 'x', '--json'],
                     ['bogus', '--json']):
            with fake('search_assets') as calls:
                code, out, err = run(*argv)
            self.assertEqual(code, 2)
            self.assertEqual(list(json.loads(out)), ['error'])
            self.assertEqual(err.splitlines()[0], f"error: {json.loads(out)['error']}")
            self.assertEqual(calls, [])
        code, out, err = run('bogus', env={'UNCRASHED_FORMAT': 'json'})
        self.assertEqual((code, list(json.loads(out))), (2, ['error']))
        for argv in (['status', '--json=true'], ['--json=1', 'status']):  # a wrong --json still asked for JSON
            code, out, err = run(*argv)
            self.assertEqual((code, json.loads(out)), (2, {'error': '--json takes no value'}), argv)
            self.assertEqual(err.splitlines()[0], 'error: --json takes no value')
        self.assertEqual(self.bound('list_groups', 'list_groups', '--', '--json=true'),
                         {'prefix': '--json=true'})  # after -- it is a value

    def test_non_finite_numbers_are_null_in_json_and_the_result_is_kept(self):
        strict = functools.partial(json.loads, parse_constant=lambda name: self.fail(f'bare {name} in --json output'))
        value = {'t': [1, 2.5], 'bad': float('nan'), 'deep': [{'hi': float('inf'), 'lo': float('-inf')}], 'ok': 1e300}
        with fake('get_world_bounds', value):
            code, out, err = run('get_world_bounds', '--json')
            self.assertEqual(code, 0)  # the operation ran: its data is not thrown away
            self.assertEqual(strict(out), {'t': [1, 2.5], 'bad': None, 'deep': [{'hi': None, 'lo': None}], 'ok': 1e300})
            self.assertEqual(err, 'warning: 3 non-finite numbers (NaN / Infinity) are null in --json; the text output '
                                  'keeps them\n')
            code, out, err = run('get_world_bounds')
            self.assertEqual((code, err), (0, ''))
            self.assertIn('bad: NaN', out)
        with fake('save_live', {'written': False, 'path': 'p', 'ms': float('nan')}):
            code, out, err = run('--json', 'save_live')
        self.assertEqual((code, strict(out)), (3, {'written': False, 'path': 'p', 'ms': None}))
        self.assertEqual(err.splitlines()[-1], 'error: the game did not write p')

    def test_non_finite_numbers_on_the_command_line_are_usage_errors(self):
        for tokens in (['--poses', '[{"x":NaN}]'], ['--poses', '[{"x":1e999}]'], ['--poses', '{"x":[-Infinity]}'],
                       ['--budgets', '{"a":NaN}'], ['--budgets', '{"a":{"warn":Infinity}}'], ['--budgets', 'a=NaN'],
                       ['--budgets', 'a=Infinity'], ['--budgets', 'a.warn=-Infinity'], ['--budgets', 'a=1e999'],
                       ['--budgets', 'a=[NaN]']):
            line = self.refused('check_performance', 'check_performance', *tokens, needle='NaN and Infinity are not')
            self.assertIn(tokens[0], line)
        self.assertEqual(self.bound('check_performance', 'check_performance', '--budgets', 'a=1e300,note=Nan'),
                         {'budgets': {'a': 1e300, 'note': 'Nan'}})
        self.refused('define_assembly', 'define_assembly', '--roles', '{"a":NaN}', needle='NaN and Infinity')


class PickTests(Case):
    def test_pick_keeps_paths_and_maps_over_lists(self):
        with fake('search_assets', ASSETS):
            self.assertEqual(run('search_assets', '--pick', 'count')[1], 'count: 2\n')
            code, out, err = run('--pick', 'assets.id,count', '--json', 'search_assets')
            self.assertEqual((code, err), (0, ''))
            self.assertEqual(json.loads(out), {'count': 2, 'assets': [{'id': 'NewRow_1'}, {'id': 'NewRow_2'}]})
            self.assertEqual(list(json.loads(out)), ['count', 'assets'])  # the reply's own order
            out = run('search_assets', '--pick=assets.size.x, assets.id', '--json')[1]
            self.assertEqual(json.loads(out), {'assets': [{'id': 'NewRow_1', 'size': {'x': 1.0}},
                                                          {'id': 'NewRow_2', 'size': {'x': 3.0}}]})
            self.assertEqual(json.loads(run('search_assets', '--pick', 'assets,assets.id', '--json')[1]),
                             {'assets': ASSETS['assets']})
            self.assertEqual(run('search_assets', '--pick', 'missing')[1], 'missing: []\n')

    def test_a_path_that_matches_nothing_prints_the_full_result(self):
        with fake('search_assets', ASSETS):
            for spec in ('nope', 'count,assets.nope', 'count.x', 'assets.id.x'):
                code, out, err = run('search_assets', '--pick', spec)
                self.assertEqual(code, 0, spec)  # never a usage error: the operation already ran
                self.assertEqual(parse(out), ASSETS, spec)
                self.assertEqual(len(err.splitlines()), 1)
                self.assertIn('matched nothing (top-level keys: count, assets, scope, missing); printing the full result',
                              err)
                self.assertIn(spec.split(',')[-1], err)

    def test_keys_missing_in_some_elements_and_empty_lists_are_fine(self):
        mixed = {'top': [{'id': 1, 'view': {'x': 1}}, {'id': 2}], 'issues': []}
        with fake('check_collisions', mixed):
            code, out, err = run('check_collisions', '--pick', 'top.view,issues.kind', '--json')
        self.assertEqual((code, err), (0, ''))
        self.assertEqual(json.loads(out), {'top': [{'view': {'x': 1}}, {}], 'issues': []})

    def test_pick_runs_after_the_soft_failure_check_and_the_preview_note(self):
        with fake('save_live', {'written': False, 'path': 'p'}):
            code, out, err = run('save_live', '--pick', 'path')
        self.assertEqual((code, out, err), (3, 'path: p\n', 'error: the game did not write p\n'))
        with fake('edit_group', {'preview': True, 'changes': 3}):
            code, out, err = run('edit_group', 'g', '--pick', 'changes')
        self.assertEqual((out, err), ('changes: 3\n', 'preview only: nothing changed (re-run with --apply)\n'))

    def test_bad_pick_syntax_is_a_usage_error_before_anything_runs(self):
        for spec in ('', 'a,,b', 'a..b', '.a'):
            self.refused('status', 'status', '--pick', spec, needle='--pick expects field paths')
        self.refused('status', 'status', '--pick', 'a', '--pick', 'b', needle='--pick given twice')
        self.refused('status', 'status', '--pick', needle='--pick needs a value')

    def test_pick_does_not_apply_to_text(self):
        with fake('agent_exec', api.Text('ok mapinfo')):
            code, out, err = run('exec', '--pick', 'x', 'mapinfo')
        self.assertEqual((code, out), (0, 'ok mapinfo\n'))
        self.assertIn('--pick does not apply to a text reply', err)


class OutputTests(Case):
    def test_a_closed_pipe_keeps_the_exit_code(self):
        class Closed(io.StringIO):
            def write(self, text):
                raise BrokenPipeError(32, 'Broken pipe')

        for result, want in (({'written': True}, 0), ({'written': False, 'path': 'p'}, 3)):
            err = io.StringIO()
            with fake('save_live', result), contextlib.redirect_stdout(Closed()), contextlib.redirect_stderr(err):
                self.assertEqual(cli.main(['save_live']), want)
            self.assertNotIn('Traceback', err.getvalue())
            self.assertEqual(err.getvalue(), 'error: the game did not write p\n' if want else '')
        with contextlib.redirect_stdout(Closed()), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(cli.main(['help', 'guide']), 0)
        self.assertEqual(err.getvalue(), '')

    def test_an_unwritable_stdout_after_the_operation_ran_says_that_it_ran(self):
        class Full(io.StringIO):
            def write(self, text):
                raise OSError(28, 'No space left on device')

        class Shut(io.StringIO):
            def write(self, text):
                raise ValueError('I/O operation on closed file.')

        for stream, reason in ((Full(), 'No space left on device'), (Shut(), 'stdout is closed'),
                               (None, 'stdout is closed')):
            for result, code, tail in (({'written': True}, 1, ''),
                                       ({'written': False, 'path': 'p'}, 3, '; it reported: the game did not write p')):
                err = io.StringIO()
                with fake('save_live', result) as calls, patch.object(sys, 'stdout', stream), \
                        contextlib.redirect_stderr(err):
                    self.assertEqual(cli.main(['save_live']), code)
                self.assertEqual(len(calls), 1)
                self.assertEqual(err.getvalue(), 'error: save_live finished but its output could not be written '
                                                 f'({reason}){tail}\n')
            err = io.StringIO()
            with patch.object(sys, 'stdout', stream), contextlib.redirect_stderr(err):
                self.assertEqual(cli.main(['help', 'verbs']), 1)
                self.assertEqual(cli.main(['--version']), 1)
            self.assertEqual(err.getvalue(), f'error: the output could not be written ({reason})\n' * 2)

    def test_a_dead_stderr_changes_neither_the_exit_code_nor_the_run(self):
        class Gone(io.StringIO):
            def write(self, text):
                raise BrokenPipeError(32, 'Broken pipe')

        for stream in (Gone(), None):
            out = io.StringIO()
            with fake('edit_group', {'preview': True, 'changes': 3}), patch.object(sys, 'stderr', stream), \
                    contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(['edit_group', 'g']), 0)  # the preview note has no reader
                self.assertEqual(out.getvalue(), 'preview: true\nchanges: 3\n')
                self.assertEqual(cli.main(['search_assets', '--bogus']), 2)
                self.assertEqual(cli.main(['bogus']), 2)
            with fake('save_live', {'written': False, 'path': 'p'}), patch.object(sys, 'stderr', stream), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(['save_live']), 3)
            self.assertEqual(out.getvalue().count('error'), 0)  # nothing meant for stderr lands on stdout

    def test_usage_errors_stay_on_one_line_whatever_a_token_holds(self):
        for argv in (['status', '--x\ny'], ['search_assets', '--limit', '--x\ny'], ['search_assets', '--limt\n3'],
                     ['help', 'a\nb'], ['bo\ngus']):
            code, out, err = run(*argv)
            self.assertEqual((code, out), (2, ''), argv)
            self.assertEqual(len([line for line in err.splitlines() if line.startswith('error: ')]), 1, err)
            self.assertLessEqual(len(err.splitlines()), 2, err)  # the error line and, for a command, its usage line
        self.assertEqual(run('status', '--x\ny')[2].splitlines()[0], 'error: unknown option --x y for status')
        code, out, err = run('--json', 'status', '--x\ny')
        self.assertEqual(json.loads(out), {'error': 'unknown option --x y for status'})

    def test_output_is_written_once(self):
        class Counting(io.StringIO):
            writes = 0

            def write(self, text):
                Counting.writes += 1
                return super().write(text)

        out = Counting()
        with fake('search_assets', ASSETS), contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(['search_assets']), 0)
        self.assertEqual(Counting.writes, 1)
        self.assertEqual(parse(out.getvalue()), ASSETS)

    def test_streams_are_reconfigured_to_utf8_with_backslashreplace(self):
        raw_out, raw_err = io.BytesIO(), io.BytesIO()
        out = io.TextIOWrapper(raw_out, encoding='ascii', errors='strict')
        err = io.TextIOWrapper(raw_err, encoding='ascii', errors='strict')
        with fake('status', body=lambda: (_ for _ in ()).throw(RuntimeError('kaputt: ä'))), \
                patch.object(sys, 'stdout', out), patch.object(sys, 'stderr', err):
            self.assertEqual(cli.main(['status']), 1)
            err.flush()
        self.assertEqual(raw_err.getvalue().decode('utf-8'), 'error: kaputt: ä\n')
        raw_out, out = io.BytesIO(), None
        out = io.TextIOWrapper(raw_out, encoding='ascii', errors='strict')
        with fake('agent_exec', api.Text('Straße \udc80')), patch.object(sys, 'stdout', out):
            self.assertEqual(cli.main(['exec', 'mapinfo']), 0)
            out.flush()
        self.assertEqual(raw_out.getvalue().decode('utf-8'), 'Straße \\udc80\n')


# ---------------------------------------------------------------------------
# signals, the agent lock, built-ins
# ---------------------------------------------------------------------------


class SignalTests(Case):
    def setUp(self):
        self.before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        self.addCleanup(lambda: [signal.signal(s, h) for s, h in self.before.items()])
        # a runner started in the background or under nohup hands down ignored signals, which the CLI respects
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGHUP, signal.SIG_DFL)

    def interrupt(self, name, argv, signals, result=None, sent=True):
        """Run a command whose operation gets the signals; sent: it has written a command to the agent by then."""
        trace = []

        def body(**kwargs):
            try:
                if sent:
                    agent.sent_count += 1  # what agent._execute does just before it writes the command
                for signum in signals:
                    signal.raise_signal(signum)
                    trace.append(f'after {signal.Signals(signum).name}')
            finally:
                trace.append('finally')
            return {'done': True} if result is None else result

        installed = {s: signal.getsignal(s) for s in self.before}
        with fake(name, body=body):
            code, out, err = run(*argv)
        self.assertEqual({s: signal.getsignal(s) for s in self.before}, installed)  # handlers are put back
        return code, out, err, trace

    def test_a_reading_operation_is_interrupted_and_its_finally_blocks_run(self):
        for signum, code in ((signal.SIGTERM, 143), (signal.SIGHUP, 129), (signal.SIGINT, 130)):
            got, out, err, trace = self.interrupt('check_collisions', ['check_collisions'], [signum])
            self.assertEqual((got, out, trace), (code, '', ['finally']))
            self.assertEqual(err, "error: interrupted during check_collisions; live state may be partial: check "
                                  "'./uc exec mapinfo', './uc edit_history', './uc log'\n")

    def test_an_offline_operation_is_interrupted_without_the_live_state_hint(self):
        got, out, err, trace = self.interrupt('list_maps', ['list_maps'], [signal.SIGTERM], sent=False)
        self.assertEqual((got, out, err, trace), (143, '', 'error: interrupted during list_maps\n', ['finally']))
        got, out, err, trace = self.interrupt('list_maps', ['--json', 'list_maps'], [signal.SIGTERM], sent=False)
        self.assertEqual((got, json.loads(out)), (143, {'error': 'interrupted during list_maps'}))

    def test_the_live_state_hint_needs_a_command_that_was_sent(self):
        # check_performance NAME reads a file; a live read that has not sent anything cannot have changed anything
        for name, argv in (('check_performance', ['check_performance', 'Example_Map']),
                           ('check_collisions', ['check_collisions'])):
            got, out, err, trace = self.interrupt(name, argv, [signal.SIGTERM], sent=False)
            self.assertEqual((got, out, trace), (143, '', ['finally']))
            self.assertEqual(err, f'error: interrupted during {name}; nothing was sent\n')

    def test_a_changing_operation_that_has_sent_nothing_stops_at_the_first_signal(self):
        for signum, code in ((signal.SIGTERM, 143), (signal.SIGHUP, 129), (signal.SIGINT, 130)):
            got, out, err, trace = self.interrupt('save_live', ['save_live'], [signum], sent=False)
            self.assertEqual((got, out, trace), (code, '', ['finally']))  # not deferred: the body did not go on
            self.assertEqual(err, 'error: interrupted during save_live; nothing was sent\n')
        self.assertNotIn('finishing', err)
        self.assertNotIn('partial', err)

    def test_a_signal_during_a_wait_that_precedes_sending_says_so(self):
        seen = []

        def body(**kwargs):
            agent.on_lock_wait(None, 0.1)  # agent.py polls this in the lock wait and in the earlier-job wait
            seen.append('waited')
            signal.raise_signal(signal.SIGTERM)
            seen.append('went on')

        for name in ('save_live', 'camera_pose'):
            with fake(name, body=body):
                code, out, err = run(name)
            self.assertEqual((code, out, err), (143, '', 'error: interrupted while waiting; nothing was sent\n'))
        self.assertEqual(seen, ['waited', 'waited'])

    def test_deferral_starts_with_the_first_command_sent_and_lasts_through_later_waits(self):
        seen = []

        def body(**kwargs):
            agent.on_lock_wait(None, 0.1)
            agent.sent_count += 1
            agent.on_lock_wait(None, 0.1)  # a second session of the same operation waits again
            signal.raise_signal(signal.SIGTERM)
            seen.append('went on')
            return {'written': True, 'path': 'p'}

        with fake('save_live', body=body):
            code, out, err = run('save_live')
        self.assertEqual((code, out, seen), (143, 'written: true\npath: p\n', ['went on']))
        self.assertTrue(err.startswith('SIGTERM received: finishing save_live first'), err)

    def test_a_second_signal_while_the_result_is_printed_ends_the_command(self):
        class Stalled(io.StringIO):
            def write(self, text):  # a reader that does not read: the handler runs inside the write
                signal.raise_signal(signal.SIGTERM)
                signal.raise_signal(signal.SIGHUP)
                raise AssertionError('the second signal did not end the write')

        for name, tail in (('save_live', '; do not re-run it'), ('camera_pose', '')):
            err = io.StringIO()
            with fake(name, {'done': True}), contextlib.redirect_stdout(Stalled()), contextlib.redirect_stderr(err):
                self.assertEqual(cli.main([name]), 143)  # the code of the first signal
            self.assertEqual(err.getvalue(), f'error: SIGHUP received while printing the result of {name}: the '
                                             f'operation finished, its output is incomplete{tail}\n')

    def test_one_signal_while_the_result_is_printed_lets_the_print_finish(self):
        class Slow(io.StringIO):
            def write(self, text):
                signal.raise_signal(signal.SIGTERM)
                return super().write(text)

        out, err = Slow(), io.StringIO()
        with fake('camera_pose', {'done': True}), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(cli.main(['camera_pose']), 143)
        self.assertEqual(out.getvalue(), 'done: true\n')
        self.assertEqual(err.getvalue(), 'error: SIGTERM received during camera_pose: the operation finished and its '
                                         'result is printed\n')

    def test_a_deferred_signal_does_not_claim_a_result_nobody_read(self):
        class Closed(io.StringIO):
            def write(self, text):
                raise BrokenPipeError(32, 'Broken pipe')

        def body(**kwargs):
            agent.sent_count += 1
            signal.raise_signal(signal.SIGTERM)
            return {'written': True, 'path': 'p'}

        err = io.StringIO()
        with fake('save_live', body=body), contextlib.redirect_stdout(Closed()), contextlib.redirect_stderr(err):
            self.assertEqual(cli.main(['save_live']), 143)
        self.assertEqual(err.getvalue().splitlines()[-1], 'error: SIGTERM received during save_live: the operation '
                                                          'finished; its output was not read; do not re-run it')

    def test_a_changing_operation_finishes_after_the_first_signal(self):
        for signum, code in ((signal.SIGTERM, 143), (signal.SIGHUP, 129), (signal.SIGINT, 130)):
            name = signal.Signals(signum).name
            got, out, err, trace = self.interrupt('save_live', ['save_live'], [signum], {'written': True, 'path': 'p'})
            self.assertEqual((got, out), (code, 'written: true\npath: p\n'))
            self.assertEqual(trace, [f'after {name}', 'finally'])
            lines = err.splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(lines[0].startswith(f'{name} received: finishing save_live first'), err)
            self.assertEqual(lines[1], f'error: {name} received during save_live: the operation finished and its '
                                       'result is printed; do not re-run it')

    def test_a_deferred_signal_keeps_the_operations_own_error_line(self):
        got, out, err, trace = self.interrupt('save_live', ['save_live'], [signal.SIGTERM], {'written': False, 'path': 'p'})
        self.assertEqual((got, out), (143, 'written: false\npath: p\n'))
        self.assertEqual([line for line in err.splitlines() if line.startswith('error: ')],
                         ['error: the game did not write p'])

        def body(**kwargs):
            agent.sent_count += 1
            signal.raise_signal(signal.SIGTERM)
            raise agent.AgentError('row missing')

        with fake('save_live', body=body):
            code, out, err = run('save_live')
        self.assertEqual((code, out), (143, ''))
        self.assertEqual([line for line in err.splitlines() if line.startswith('error: ')], ['error: row missing'])

    def test_a_second_signal_aborts_a_changing_operation(self):
        got, out, err, trace = self.interrupt('save_live', ['save_live'], [signal.SIGTERM, signal.SIGINT])
        self.assertEqual((got, out, trace), (130, '', ['after SIGTERM', 'finally']))
        self.assertEqual(err.splitlines()[-1], "error: interrupted during save_live; live state may be partial: check "
                                               "'./uc exec mapinfo', './uc edit_history', './uc log'")

    def test_an_ignored_signal_stays_ignored(self):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        got, out, err, trace = self.interrupt('check_collisions', ['check_collisions'], [signal.SIGHUP])
        self.assertEqual((got, trace), (0, ['after SIGHUP', 'finally']))
        signal.signal(signal.SIGHUP, self.before[signal.SIGHUP])

    def test_the_agent_session_is_released_on_interrupt(self):
        def body(**kwargs):
            with agent.session():
                signal.raise_signal(signal.SIGTERM)

        for name in ('check_collisions', 'save_live'):  # a changing operation that has sent nothing stops as well
            with fake(name, body=body):
                self.assertEqual(run(name)[0], 143)
            self.assertEqual(agent._lock_depth, 0)
            with agent.session():  # not left locked
                pass
        self.assertEqual(agent.holder_extra, {})


HOLDER = textwrap.dedent('''
    import fcntl, json, os, sys, time
    fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)
    os.ftruncate(fd, 0)
    record = {"pid": os.getpid(), "argv": json.loads(sys.argv[2]), "since": time.time(), **json.loads(sys.argv[3])}
    os.pwrite(fd, json.dumps(record).encode(), 0)
    print("held", flush=True)
    time.sleep(60)
''')


def digest(operation, /, **kwargs):
    return cli._digest(api.OPERATIONS[operation], kwargs)


class SameCommandGuardTests(Case):
    def hold(self, argv, **extra):
        holder = subprocess.Popen([sys.executable, '-c', HOLDER, str(agent.LOCK), json.dumps(argv), json.dumps(extra)],
                                  stdout=subprocess.PIPE, text=True)
        self.addCleanup(holder.wait)
        self.addCleanup(holder.kill)
        self.assertEqual(holder.stdout.readline(), 'held\n')
        holder.stdout.close()
        return holder

    def attempt(self, name, argv, stdin=''):
        reached = []

        def body(**kwargs):
            with agent.session():
                reached.append(kwargs)
            return {}

        started = time.monotonic()
        with fake(name, body=body):
            code, out, err = run(*argv, stdin=stdin, env={'UNCRASHED_LOCK_TIMEOUT': '0.5'})
        return code, out, err, reached, time.monotonic() - started

    def test_the_same_changing_command_fails_at_once(self):
        holder = self.hold(['/somewhere/uncrashed_mcp/cli.py', 'save_live', 'City'], digest=digest('save_live', name='City'))
        for argv in (['save_live', 'City'], ['save_live', '--name', 'City'], ['save-live', '--name=City']):
            code, out, err, reached, took = self.attempt('save_live', argv)  # every spelling of the same request
            self.assertEqual((code, out, reached), (1, '', []), argv)
            self.assertEqual(err, f'error: the same command is already running as pid {holder.pid}; nothing was sent: '
                                  'wait for it instead of re-sending\n')
            self.assertLess(took, 0.4)
        self.assertIsNone(agent.on_lock_wait)
        self.assertEqual(agent.holder_extra, {})

    def test_the_same_words_with_other_input_wait_for_the_lock(self):
        # `batch -` and `batch plan.txt` are the same argv for different work: only what is sent counts
        argv = ['/somewhere/uncrashed_mcp/cli.py', 'batch', '-']
        holder = self.hold(argv, digest=digest('agent_batch', commands=['spawn A 0 0 0'], _lines=[1]))
        code, out, err, reached, took = self.attempt('agent_batch', ['batch', '-'], stdin='spawn B 9 9 9\n')
        self.assertEqual((code, reached), (1, []))
        self.assertIn(f'agent busy: lock held by pid {holder.pid} (uc batch -)', err)
        self.assertNotIn('the same command', err)
        self.assertGreaterEqual(took, 0.5)
        code, out, err, reached, took = self.attempt('agent_batch', ['batch', '-'], stdin='spawn A 0 0 0\n')
        self.assertIn('the same command is already running', err)
        self.assertLess(took, 0.4)

    def test_a_holder_without_a_digest_never_matches(self):
        holder = self.hold(['/somewhere/uncrashed_mcp/cli.py', 'save_live', 'City'])  # an older ./uc, the MCP server
        code, out, err, reached, took = self.attempt('save_live', ['save_live', 'City'])
        self.assertEqual((code, reached), (1, []))
        self.assertIn(f'agent busy: lock held by pid {holder.pid} (uc save_live City)', err)
        self.assertIn('nothing was sent', err)
        self.assertGreaterEqual(took, 0.5)

    def test_a_different_command_waits_for_the_lock(self):
        holder = self.hold(['uc', 'save_live', 'City'], digest=digest('save_live', name='City'))
        code, out, err, reached, took = self.attempt('save_live', ['save_live', 'Town'])
        self.assertEqual((code, reached), (1, []))
        self.assertIn(f'agent busy: lock held by pid {holder.pid} (uc save_live City)', err)
        self.assertIn('nothing was sent', err)
        self.assertNotIn('before retrying', err)  # nothing was sent: no "may still run" recovery text
        self.assertGreaterEqual(took, 0.5)

    def test_an_identical_apply_is_refused(self):
        self.hold(['uc', 'x'], digest=digest('edit_group', group='g', delta=[1.0, 0.0, 0.0], preview=False))
        code, out, err, reached, took = self.attempt('edit_group', ['edit_group', 'g', '--delta', '1,0,0', '--apply'])
        self.assertEqual((code, reached), (1, []))
        self.assertIn('the same command is already running', err)

    def test_the_hook_ignores_holders_it_cannot_identify(self):
        signals = cli._Signals('save_live', defer=True)
        hook = cli._wait_hook('abc', signals)
        self.assertFalse(signals.waiting)
        hook(None, 0.0)  # also how the wait for an earlier job calls it
        self.assertTrue(signals.waiting)
        hook({'pid': 1, 'argv': [], 'since': None}, 0.0)
        hook({'pid': 1, 'argv': ['uncrashed-mcp'], 'since': None}, 0.0)
        hook({'pid': 1, 'argv': ['uc', 'save_live', 'City'], 'since': None, 'digest': 'other'}, 0.0)
        hook({'pid': 1, 'argv': ['uc', 'save_live', 'City'], 'since': None, 'digest': None}, 0.0)
        hook({'pid': os.getpid(), 'argv': ['uc', 'save_live', 'City'], 'since': None, 'digest': 'abc'}, 0.0)
        with self.assertRaisesRegex(agent.AgentNotSent, 'the same command is already running as pid 1;'):
            hook({'pid': 1, 'argv': ['anything'], 'since': None, 'digest': 'abc'}, 0.0)
        cli._wait_hook(None, signals)({'pid': 1, 'argv': [], 'since': None, 'digest': None}, 0.0)  # unguarded

    def test_the_digest_covers_the_operation_and_its_bound_arguments(self):
        self.assertEqual(digest('save_live', name='City', display_name='C'), digest('save_live', display_name='C', name='City'))
        self.assertNotEqual(digest('save_live', name='City'), digest('save_live', name='Town'))
        self.assertNotEqual(digest('save_live'), digest('agent_unload'))
        self.assertNotEqual(digest('agent_batch', commands=['a', 'b']), digest('agent_batch', commands=['a b']))
        self.assertRegex(digest('save_live'), r'^[0-9a-f]{64}$')

    def test_hook_and_digest_are_set_only_while_an_operation_runs(self):
        seen = []

        def body(**kwargs):
            seen.append((agent.on_lock_wait, dict(agent.holder_extra)))
            return {}

        with fake('save_live', body=body):
            run('save_live', 'City')
        with fake('camera_pose', body=body):
            run('camera_pose')
        with fake('edit_group', body=body):
            run('edit_group', 'g')
            run('edit_group', 'g', '--apply')
        with fake('save_live', agent.AgentError('boom')):
            run('save_live', 'City')
        self.assertTrue(all(callable(hook) for hook, _extra in seen))
        self.assertEqual([extra for _hook, extra in seen],
                         [{'digest': digest('save_live', name='City')}, {}, {}, {'digest': digest('edit_group', group='g', preview=False)}])
        self.assertIsNone(agent.on_lock_wait)
        self.assertEqual(agent.holder_extra, {})

    def test_the_holder_record_of_a_running_command_carries_its_digest(self):
        seen = []

        def body(**kwargs):
            with agent.session():
                seen.append(agent.lock_holder())
            return {}

        with fake('save_live', body=body):
            self.assertEqual(run('save_live', 'City')[0], 0)
        self.assertEqual(seen[0]['digest'], digest('save_live', name='City'))
        self.assertEqual(seen[0]['pid'], os.getpid())


class BuiltinTests(Case):
    def test_last_prints_the_previous_agent_reply(self):
        with tempfile.TemporaryDirectory() as tmp:
            reply = Path(tmp) / 'out.txt'
            with patch.object(agent, 'OUT', reply):
                code, out, err = run('last')
                self.assertEqual((code, out), (1, ''))
                self.assertEqual(len(err.splitlines()), 1)
                self.assertIn(f'error: no previous agent reply: cannot read {reply}', err)
                reply.write_text('1042 batch\nbatch lines=3 ok=3 failed=0\nend 1042 ok\n')
                self.assertEqual(run('last'), (0, reply.read_text(), ''))
                self.assertEqual(json.loads(run('last', '--json')[1]), {'ok': True, 'text': reply.read_text()})
                self.assertEqual(run('last', 'extra')[0], 2)

    def test_last_says_when_a_job_is_still_running(self):
        note = ('note: an agent job is still running; this is the reply of the job before it (the first line starts '
                'with its sequence number)\n')
        with tempfile.TemporaryDirectory() as tmp:
            reply = Path(tmp) / 'out.txt'
            marker = Path(tmp) / 'out.txt.tmp'  # the DLL writes here and renames it onto the reply when the job ends
            with patch.object(agent, 'OUT', reply), patch.object(agent, 'game_pid', return_value=None):
                reply.write_text('41 ok mapinfo\nfolder City\nend 41 ok\n')
                marker.write_text('42 ok batch\n')
                self.assertEqual(run('last'), (0, reply.read_text(), note))
                code, out, err = run('last', '--json')
                self.assertEqual((code, json.loads(out), err), (0, {'ok': True, 'text': reply.read_text()}, note))
                reply.unlink()
                code, out, err = run('last')  # the first job since the files were cleared
                self.assertEqual((code, out), (1, ''))
                self.assertEqual(err.splitlines()[0], note.strip())
                self.assertIn('error: no previous agent reply', err.splitlines()[1])
                reply.write_text('41 ok mapinfo\nend 41 ok\n')
                os.utime(marker, (time.time() - 3600, time.time() - 3600))  # a leftover, not a running job
                self.assertEqual(run('last'), (0, reply.read_text(), ''))
                marker.unlink()
                self.assertEqual(run('last'), (0, reply.read_text(), ''))

    def test_mcp_without_the_package_is_one_line(self):
        absent = {name: None for name in list(sys.modules) if name == 'mcp' or name.startswith('mcp.')}
        with patch.dict(sys.modules, {**absent, 'mcp': None}):
            sys.modules.pop('uncrashed_mcp.server', None)
            code, out, err = run('mcp')
        self.assertEqual((code, out), (1, ''))
        self.assertEqual(err, "error: the MCP server needs the 'mcp' package: pip install 'uncrashed-mcp[mcp]'\n")

    def test_mcp_runs_the_server(self):
        try:
            from uncrashed_mcp import server
        except ModuleNotFoundError:
            self.skipTest('mcp is not installed')
        with patch.object(server, 'main') as serve:
            self.assertEqual(run('mcp'), (0, '', ''))
        serve.assert_called_once_with()
        self.assertEqual(run('mcp', 'now')[0], 2)


# ---------------------------------------------------------------------------
# the launcher
# ---------------------------------------------------------------------------


# sitecustomize for a ./uc subprocess: a slow save_live that never reaches the agent
SLOW_WRITE = textwrap.dedent('''
    import contextlib, os, time
    if os.environ.get("UC_TEST_MARKER"):
        from uncrashed_mcp import agent, api

        def save_live(name: str = "", display_name: str = "") -> dict:
            with agent.session() if os.environ.get("UC_TEST_SESSION") else contextlib.nullcontext():
                if os.environ.get("UC_TEST_SENT"):
                    agent.sent_count += 1  # as agent._execute does just before it writes a command
                open(os.environ["UC_TEST_MARKER"], "w").close()
                deadline = time.time() + float(os.environ.get("UC_TEST_HOLD", "1.5"))
                while time.time() < deadline:
                    time.sleep(0.02)
            return {"written": True, "path": name}

        api.OPERATIONS["save_live"].fn = save_live
''')


class LauncherTests(unittest.TestCase):
    def test_status(self):
        done = uc('status')
        self.assertEqual((done.returncode, done.stderr), (0, ''))
        status = parse(done.stdout)
        self.assertEqual(status['flat_base'], 'BaseBoard')
        self.assertGreater(status['assets'], 100)
        picked = uc('status', '--pick', 'assets,flat_base')
        self.assertEqual(picked.stdout, f"assets: {status['assets']}\nflat_base: BaseBoard\n")

    def test_search_assets(self):
        done = uc('search_assets', 'house', '--limit', '2')
        self.assertEqual((done.returncode, done.stderr), (0, ''))
        found = parse(done.stdout)
        self.assertEqual(found['count'], 2)
        self.assertEqual(len(found['assets']), 2)
        self.assertRegex(done.stdout, r'(?m)^assets:\n  - \{id=NewRow_')  # 11 columns, too few rows to hoist any
        more = uc('search_assets', 'house', '--limit', '3')
        self.assertRegex(more.stdout, r'(?m)^assets\[3\]\{id\|name.*\}:\n  \(every row also has\) ')
        self.assertEqual(len(parse(more.stdout)['assets']), 3)
        as_json = uc('--json', 'search_assets', 'house', '--limit', '2')
        self.assertEqual(json.loads(as_json.stdout), found)
        self.assertEqual(as_json.stdout.count('\n'), 1)
        self.assertEqual(json.loads(uc('search-assets', 'house', '--limit=2', env={'UNCRASHED_FORMAT': 'json'}).stdout),
                         found)

    def test_get_assets(self):
        done = uc('get_assets', 'NewRow_1', 'NewRow_2', 'NoSuchRow')
        self.assertEqual((done.returncode, done.stderr), (0, ''))
        found = parse(done.stdout)
        self.assertEqual([asset['id'] for asset in found['assets']], ['NewRow_1', 'NewRow_2'])
        self.assertEqual(found['missing'], ['NoSuchRow'])

    def test_usage_error(self):
        done = uc('search_assets', 'big', 'house')
        self.assertEqual((done.returncode, done.stdout), (2, ''))
        lines = done.stderr.splitlines()
        self.assertTrue(lines[0].startswith('error: search_assets takes at most 1 (QUERY) positional arguments'))
        self.assertTrue(lines[1].startswith('usage: ./uc search_assets [QUERY:str] [--category str]'))
        self.assertEqual(len(lines), 2)
        done = uc('bogus')
        self.assertEqual((done.returncode, done.stdout), (2, ''))
        self.assertEqual(len(done.stderr.splitlines()), 1)

    def test_a_live_command_is_refused_offline_and_sends_nothing(self):
        def state():
            return ([(path.stat().st_mtime_ns if path.exists() else None) for path in (agent.CMD, agent.SEQ)],
                    agent.batch_files())

        before = state()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'plan.txt').write_text('spawn NewRow_1 0 0 0\n')
            done = uc('batch', str(Path(tmp) / 'plan.txt'))  # a batch writes its command file before it sends
            self.assertEqual((done.returncode, done.stdout), (1, ''))
            self.assertEqual(done.stderr, 'error: offline mode (UNCRASHED_OFFLINE set): live agent calls are disabled\n')
        for argv in (['exec', 'mapinfo'], ['camera_pose'], ['list_groups'], ['save_live', 'City']):
            done = uc(*argv)
            self.assertEqual((done.returncode, done.stdout), (1, ''), argv)
            self.assertEqual(done.stderr, 'error: offline mode (UNCRASHED_OFFLINE set): live agent calls are disabled\n')
        done = uc('--json', 'exec', 'mapinfo')
        self.assertEqual(done.returncode, 1)
        self.assertEqual(json.loads(done.stdout), {'error': 'offline mode (UNCRASHED_OFFLINE set): live agent calls '
                                                            'are disabled'})
        self.assertEqual(state(), before)

    def test_a_mistyped_map_name_is_named_before_the_agent_is_tried(self):
        done = uc('load_live', 'no such map 7f3a')
        self.assertEqual((done.returncode, done.stdout), (1, ''))
        self.assertRegex(done.stderr, r"^error: no local map 'no_such_map_7f3a' in /.*\n$")
        self.assertNotIn('Errno', done.stderr)

    def test_every_registry_example_runs_or_is_refused_offline(self):
        offline = 'error: offline mode (UNCRASHED_OFFLINE set): live agent calls are disabled\n'
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'plan.txt').write_text('mapinfo\n')
            for op in api.OPERATIONS.values():
                for example in op.examples:
                    words = shlex.split(example)
                    done = uc(*words[1:], stdin='mapinfo\n', cwd=tmp)
                    self.assertIn(done.returncode, (0, 1), (example, done.stderr))  # never a usage error
                    if done.returncode == 1 and done.stderr != offline:  # the examples' map does not exist
                        self.assertRegex(done.stderr, r"^error: no (local map 'MyMap' in |saved map at \S*MyMap\.json\n)",
                                         example)
                    if done.returncode == 0:
                        self.assertEqual(op.live, 'no', example)

    def test_help_and_version(self):
        done = uc()
        self.assertEqual((done.returncode, done.stderr), (0, ''))
        self.assertLess(len(done.stdout), 4500)
        self.assertEqual(uc('help').stdout, done.stdout)
        self.assertEqual(uc('--version').stdout, f'uncrashed {uncrashed_mcp.__version__}\n')
        self.assertEqual(uc('help', 'verbs').stdout, api.VERBS + '\n')

    def test_the_launcher_works_from_another_directory_and_through_a_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp) / 'uc-link'
            link.symlink_to(UC)
            done = subprocess.run([str(link), '--version'], capture_output=True, text=True, cwd=tmp,
                                  env=environment(PYTHONPATH=tmp))
            self.assertEqual((done.returncode, done.stdout, done.stderr),
                             (0, f'uncrashed {uncrashed_mcp.__version__}\n', ''))
        self.assertTrue(os.access(UC, os.X_OK))
        self.assertTrue(Path(UC).read_text().startswith('#!/bin/sh\n'))

    def test_python_m_package_runs_the_cli(self):
        done = subprocess.run([sys.executable, '-m', 'uncrashed_mcp', '--version'], capture_output=True, text=True,
                              env=environment(PYTHONPATH=str(ROOT / 'src')))
        self.assertEqual((done.returncode, done.stdout), (0, f'uncrashed {uncrashed_mcp.__version__}\n'))

    def test_startup_imports_stay_light(self):
        code = ('import sys; from uncrashed_mcp import cli; cli.main(["help"]); cli.main(["help", "search_assets"]);'
                'heavy = [m for m in sys.modules if m.split(".")[0] in ("mcp", "PIL", "pydantic", "argparse") '
                'or m in ("uncrashed_mcp.live", "uncrashed_mcp.server")]; print("HEAVY", heavy, file=sys.stderr)')
        done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                              env=environment(PYTHONPATH=str(ROOT / 'src')))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stderr.strip(), 'HEAVY []')

    def test_a_closed_stderr_pipe_keeps_the_exit_code(self):
        # `2>&1 | head -1` under pipefail: the unwritten line made the interpreter's exit-time flush fail (120)
        for argv, code in ((['search_assets', '--bogus'], 2), (['exec', 'mapinfo'], 1),
                           (['search_assets', 'house', '--limit', '2', '--pick', 'nope'], 0)):
            process = subprocess.Popen([UC, *argv], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=environment())
            process.stderr.close()  # the reader is gone before the first write
            self.assertEqual(process.wait(timeout=30), code, argv)

    def test_closed_standard_streams_are_not_tracebacks(self):
        def closing(*fds):
            return lambda: [os.close(fd) for fd in fds]

        done = subprocess.run([UC, 'status'], stderr=subprocess.PIPE, text=True, env=environment(), preexec_fn=closing(1))
        self.assertEqual(done.returncode, 1)
        self.assertEqual(done.stderr, 'error: status finished but its output could not be written (stdout is closed)\n')
        done = subprocess.run([UC, 'bogus'], stdout=subprocess.PIPE, text=True, env=environment(), preexec_fn=closing(2))
        self.assertEqual((done.returncode, done.stdout), (2, ''))
        if os.path.exists('/dev/full'):
            with open('/dev/full', 'w') as full:
                done = subprocess.run([UC, 'help'], stdout=full, stderr=subprocess.PIPE, text=True, env=environment())
            self.assertEqual(done.returncode, 1)
            self.assertEqual(done.stderr, 'error: the output could not be written (No space left on device)\n')
        done = subprocess.run([UC, 'batch', '-'], capture_output=True, text=True, env=environment(),
                              preexec_fn=closing(0))
        self.assertEqual((done.returncode, done.stdout), (2, ''))
        self.assertTrue(done.stderr.startswith('error: command file: stdin (-) is closed'), done.stderr)
        self.assertNotIn('Traceback', done.stderr)

    def test_a_closed_pipe_is_not_an_error(self):
        process = subprocess.Popen([UC, 'help', 'guide'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env=environment())
        process.stdout.close()  # the reader is gone before the first write
        self.assertEqual(process.wait(timeout=30), 0)
        self.assertEqual(process.stderr.read(), b'')
        process.stderr.close()

    def start_slow(self, tmp, marker, *argv, **env):
        env = environment(PYTHONPATH=tmp, UC_TEST_MARKER=str(marker), **env)
        process = subprocess.Popen([UC, *argv], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        self.addCleanup(process.wait)
        self.addCleanup(process.kill)
        self.addCleanup(process.stderr.close)
        self.addCleanup(process.stdout.close)
        deadline = time.time() + 20
        while not marker.exists() and time.time() < deadline and process.poll() is None:
            time.sleep(0.01)
        self.assertTrue(marker.exists(), 'the slow operation did not start')
        return process

    def test_an_identical_changing_command_is_refused_while_the_first_holds_the_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'sitecustomize.py').write_text(SLOW_WRITE)
            lock = {'UNCRASHED_LOCK': str(Path(tmp) / 'agent.lock'), 'UC_TEST_SESSION': '1', 'UC_TEST_HOLD': '20'}
            first = self.start_slow(tmp, Path(tmp) / 'first', 'save_live', 'City', **lock)
            started = time.time()
            second = subprocess.run([UC, 'save_live', 'City'], capture_output=True, text=True,
                                    env=environment(PYTHONPATH=tmp, UC_TEST_MARKER=str(Path(tmp) / 'second'), **lock))
            self.assertLess(time.time() - started, 1.0)
            self.assertEqual((second.returncode, second.stdout), (1, ''))
            self.assertEqual(second.stderr, f'error: the same command is already running as pid {first.pid}; nothing was '
                                            'sent: wait for it instead of re-sending\n')
            other = subprocess.run([UC, 'save_live', 'Town'], capture_output=True, text=True,
                                   env=environment(PYTHONPATH=tmp, UC_TEST_MARKER=str(Path(tmp) / 'second'),
                                                   UNCRASHED_LOCK_TIMEOUT='0.2', **lock))
            self.assertEqual((other.returncode, other.stdout), (1, ''))
            self.assertRegex(other.stderr, rf'^error: agent busy: lock held by pid {first.pid} \(uc save_live City\) '
                                           r'for \d+s; nothing was sent; retry when that process has finished \(the '
                                           r'lock frees itself when its holder exits: never delete \S+agent\.lock\)\n$')
            self.assertFalse((Path(tmp) / 'second').exists())
            self.assertIsNone(first.poll())  # still holding the agent; the cleanup kills it

    def test_sigterm_lets_a_changing_operation_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'sitecustomize.py').write_text(SLOW_WRITE)
            process = self.start_slow(tmp, Path(tmp) / 'started', 'save_live', 'City', UC_TEST_SENT='1')
            process.send_signal(signal.SIGTERM)
            out, err = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 143)
        self.assertEqual(out, 'written: true\npath: City\n')
        lines = err.splitlines()
        self.assertEqual(len(lines), 2, err)
        self.assertTrue(lines[0].startswith('SIGTERM received: finishing save_live first'))
        self.assertTrue(lines[1].startswith('error: SIGTERM received during save_live: the operation finished'))

    def test_sigterm_stops_a_changing_operation_that_still_waits_for_the_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'sitecustomize.py').write_text(SLOW_WRITE)
            lock = {'UNCRASHED_LOCK': str(Path(tmp) / 'agent.lock'), 'UC_TEST_SESSION': '1'}
            first = self.start_slow(tmp, Path(tmp) / 'first', 'save_live', 'City', UC_TEST_HOLD='20', **lock)
            env = environment(PYTHONPATH=tmp, UC_TEST_MARKER=str(Path(tmp) / 'second'), UC_TEST_HOLD='0', **lock)
            waiting = subprocess.Popen([UC, 'save_live', 'Town'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, env=env)
            self.addCleanup(waiting.kill)
            deadline = time.time() + 20
            while time.time() < deadline:  # until it polls the lock: its descriptor is open then
                links = [os.readlink(entry) for entry in Path(f'/proc/{waiting.pid}/fd').iterdir()
                         if os.path.lexists(entry)] if waiting.poll() is None else []
                if lock['UNCRASHED_LOCK'] in links:
                    break
                time.sleep(0.02)
            time.sleep(0.3)  # several polls (every 50 ms) of the wait have passed
            started = time.time()
            waiting.send_signal(signal.SIGTERM)
            out, err = waiting.communicate(timeout=30)
            self.assertLess(time.time() - started, 2.0)  # not after the 20 s the holder keeps the lock
            self.assertEqual((waiting.returncode, out), (143, ''))
            self.assertEqual(err, 'error: interrupted while waiting; nothing was sent\n')
            self.assertFalse((Path(tmp) / 'second').exists())  # the cancelled save never started
            self.assertIsNone(first.poll())

    def test_sigterm_interrupts_a_reading_operation(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'sitecustomize.py').write_text(SLOW_WRITE.replace('save_live', 'list_maps')
                                                        .replace('name: str = "", display_name: str = ""', '')
                                                        .replace('"path": name', '"path": ""'))
            process = self.start_slow(tmp, Path(tmp) / 'started', 'list_maps')
            started = time.time()
            process.send_signal(signal.SIGTERM)
            out, err = process.communicate(timeout=30)
        self.assertEqual((process.returncode, out, err), (143, '', 'error: interrupted during list_maps\n'))
        self.assertLess(time.time() - started, 1.0)


if __name__ == '__main__':
    unittest.main()
