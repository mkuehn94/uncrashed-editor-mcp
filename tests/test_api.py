import contextlib
import dataclasses
import inspect
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import uncrashed_mcp
from uncrashed_mcp import agent, api

BASELINE = json.loads((Path(__file__).parent / 'data' / 'mcp_tools_baseline.json').read_text())
TOOLS = {tool['name']: tool for tool in BASELINE['tools']}
EMPTY = inspect.Parameter.empty

# The only places where an operation's public surface may differ from the pre-port MCP tools.
ADDED_PARAMS = {'capture_view': {'out': '', 'reference': False, 'reference_only': False}}
CHANGED_DOCS = {'agent_batch', 'capture_view'}
# Operations added after the port, appended after the baseline tools in this order (their surface is pinned below).
SCAN_OPS = ('scan_survey', 'scan_status', 'scan_place', 'scan_height', 'scan_profile', 'scan_at', 'scan_objects',
            'scan_ground', 'scan_building', 'scan_facade', 'scan_free', 'scan_view', 'scan_sheet', 'scan_match',
            'scan_plan')
ADDED_OPS = ('capture_sheet', 'scene_snapshot', 'scene_diff', 'check_opening', 'reference_load', 'reference_status',
             'reference_clear', 'reference_visibility', *SCAN_OPS, 'scan_compare', 'capture_topdown', 'asset_profile',
             'scan_sources', 'scan_grid')

_guards = []


def setUpModule():
    # No test in this module may reach the game: the agent is mocked per test, and these fail loudly otherwise.
    for name in ('ensure_agent', '_execute'):
        guard = patch.object(agent, name, side_effect=AssertionError(f'agent.{name} reached from a test'))
        guard.start()
        _guards.append(guard)
    guard = patch.object(agent, 'session', contextlib.nullcontext)  # keep tests off the shared agent lock
    guard.start()
    _guards.append(guard)


def tearDownModule():
    while _guards:
        _guards.pop().stop()


def echo(*args, **kwargs):
    return {'args': list(args), 'kwargs': kwargs}


class RegistryTests(unittest.TestCase):
    def test_operations_are_the_baseline_tools_in_order(self):
        self.assertEqual(list(api.OPERATIONS), [tool['name'] for tool in BASELINE['tools']] + list(ADDED_OPS))
        for name, op in api.OPERATIONS.items():
            self.assertEqual(op.name, name)
            self.assertIs(getattr(api, name), op.fn)

    def test_instructions_and_docstrings_are_the_baseline_text(self):
        self.assertEqual(api.INSTRUCTIONS, BASELINE['instructions'])
        self.assertTrue(api.INSTRUCTIONS.endswith(api.VERBS))
        for name, op in api.OPERATIONS.items():
            if name in ADDED_OPS:
                continue
            if name in CHANGED_DOCS:
                self.assertTrue(op.fn.__doc__.startswith(TOOLS[name]['description'].rstrip()), name)
                if name == 'capture_view':  # the declared addition: the reference parameter
                    self.assertIn('hidden for the capture unless reference=True', op.fn.__doc__)
                    continue
                # the declared addition: the failure format, with the source line a file-based caller gets
                self.assertIn('"#<index> [L<file line>] <error> << <command>"', op.fn.__doc__)
                self.assertIn('at most 20 of them', op.fn.__doc__)
            else:
                self.assertEqual(op.fn.__doc__, TOOLS[name]['description'], name)

    def test_parameters_and_defaults_match_the_baseline_schema(self):
        for name, op in api.OPERATIONS.items():
            if name in ADDED_OPS:
                continue
            schema = TOOLS[name]['input_schema']
            params = {p.name: p for p in op.params()}
            added = ADDED_PARAMS.get(name, {})
            self.assertEqual(set(params), set(schema['properties']) | set(added), name)
            self.assertEqual({n for n, p in params.items() if p.default is EMPTY}, set(schema.get('required', [])), name)
            for pname, param in params.items():
                if pname in added:
                    self.assertEqual(param.default, added[pname])
                elif param.default is not EMPTY:
                    self.assertEqual(param.default, schema['properties'][pname].get('default'), f'{name}.{pname}')

    def test_added_operations_surface(self):
        op = api.OPERATIONS['capture_sheet']
        self.assertEqual([(p.name, p.annotation, p.default) for p in op.params()],
                         [('poses', str, EMPTY), ('width', int, 480), ('height', int, 270), ('columns', int, 0),
                          ('out', str, ''), ('reference', bool, False), ('reference_only', bool, False)])
        self.assertEqual((op.live, op.effect, op.positional), ('yes', 'read', ('poses',)))

    def test_scene_snapshot_and_diff_surface(self):
        snap, diff = api.OPERATIONS['scene_snapshot'], api.OPERATIONS['scene_diff']
        self.assertEqual([(p.name, p.annotation, p.default) for p in snap.params()],
                         [('group', str, ''), ('box', list[float] | None, None), ('name', str, '')])
        self.assertEqual([(p.name, p.default) for p in diff.params()],
                         [('snapshot', EMPTY), ('group', None), ('box', None), ('position_m', .01),
                          ('rotation_deg', .1), ('scale', .001), ('gate_position_m', .01), ('examples', 10)])
        self.assertEqual((snap.live, snap.effect, snap.positional), ('yes', 'read', ()))
        self.assertEqual((diff.live, diff.effect, diff.positional), ('yes', 'read', ('snapshot',)))

    def test_reference_surface(self):
        load = api.OPERATIONS['reference_load']
        self.assertEqual([(p.name, p.default) for p in load.params()],
                         [('obj', EMPTY), ('at', None), ('map_origin', None), ('yaw', 0), ('center', None),
                          ('radius_m', 0), ('datum_m', 'auto'), ('near_m', 160), ('far_scale', 2), ('axes', ''),
                          ('replace', False), ('min_free_gb', 3)])
        self.assertEqual((load.effect, load.positional), ('write', ('obj',)))
        self.assertEqual(load.check({'loaded_chunks': 3, 'chunks': 3}), None)
        self.assertEqual(load.check({'loaded_chunks': 1, 'chunks': 3, 'stopped': 'memory floor: x'}),
                         'reference loaded 1/3 chunks: memory floor: x')
        self.assertEqual([p.name for p in api.OPERATIONS['reference_status'].params()], [])
        self.assertEqual(api.OPERATIONS['reference_status'].effect, 'read')
        self.assertEqual([(p.name, p.default) for p in api.OPERATIONS['reference_visibility'].params()], [('hidden', True)])
        self.assertEqual(api.OPERATIONS['reference_clear'].effect, 'write')

    def test_check_opening_surface(self):
        op = api.OPERATIONS['check_opening']
        self.assertEqual([(p.name, p.annotation, p.default) for p in op.params()],
                         [('path', list[list[float]], EMPTY), ('profile', str, '5inch'), ('margin_m', float | None, None),
                          ('channel', str, 'Pawn'), ('spacing_m', float, .25), ('dims_m', list[float] | None, None),
                          ('near_m', float, 5.0), ('top', int, 10)])
        self.assertEqual((op.live, op.effect, op.positional, op.variadic), ('yes', 'read', ('path',), 'path'))

    def test_params_are_resolved_and_hide_private_ones(self):
        params = api.OPERATIONS['agent_batch'].params()
        self.assertEqual([p.name for p in params], ['commands', 'quiet', 'max_lines', 'timeout'])
        self.assertEqual(params[0], ('commands', list[str], EMPTY))
        self.assertIn('_lines', inspect.signature(api.agent_batch).parameters)
        for op in api.OPERATIONS.values():
            for name, annotation, _default in op.params():
                self.assertNotIsInstance(annotation, str, f'{op.name}.{name}')
        self.assertEqual(dict((p.name, p.annotation) for p in api.OPERATIONS['check_performance'].params())['budgets'],
                         dict | None)

    def test_summaries_are_one_short_line(self):
        for op in api.OPERATIONS.values():
            self.assertTrue(op.summary and op.summary == op.summary.strip(), op.name)
            self.assertNotIn('\n', op.summary)
            self.assertLessEqual(len(op.summary), 70, op.name)
            self.assertNotIn('\n', op.caution)

    def test_live_and_effect_classification(self):
        by_live = {kind: {n for n, op in api.OPERATIONS.items() if op.live == kind} for kind in ('no', 'mixed', 'yes')}
        self.assertEqual(by_live['no'], {'status', 'list_gates', 'search_assets', 'get_assets', 'list_maps', 'inspect_map',
                                         'noncolliding_alternatives', *SCAN_OPS, 'asset_profile', 'scan_sources',
                                         'scan_grid'}
                         - {'scan_place'})
        # scan_place --from_reference reads the open editor; scan_compare does unless given a snapshot file
        self.assertEqual(by_live['mixed'], {'check_performance', 'consolidate_scenery', 'scan_place', 'scan_compare'})
        self.assertEqual(len(by_live['yes']), 37)   # 36 + capture_topdown
        self.assertEqual({n for n, op in api.OPERATIONS.items() if op.effect == 'write'},
                         {'load_live', 'save_live', 'agent_exec', 'agent_batch', 'agent_unload', 'clearing_zones',
                          'edit_group', 'undo_edit', 'undo_collision', 'define_assembly', 'set_collision',
                          'consolidate_scenery', 'reference_load', 'reference_clear', 'reference_visibility',
                          'scan_place', 'scan_plan'})  # placement file; plan .txt / .json in the project
        for name in ('scan_place', 'scan_compare'):  # a mixed op that is not about saved maps says what it needs
            self.assertTrue(api.OPERATIONS[name].needs, name)
        self.assertEqual({op.effect for op in api.OPERATIONS.values()}, {'read', 'write'})
        for name in ('load_live', 'save_live', 'agent_exec', 'agent_batch'):
            self.assertTrue(api.OPERATIONS[name].caution, name)
        # the MCP description of undo_edit is pinned to the baseline; what it leaves out is a CLI caution
        caution = api.OPERATIONS['undo_edit'].caution
        for needle in ('interrupted', 'previewed first', 'later changes', 'recovered', 'deletes_not_in_before'):
            self.assertIn(needle, caution)

    def test_cli_hints_name_real_parameters(self):
        expected_positional = {
            'search_assets': ('query',), 'get_assets': ('ids',), 'inspect_map': ('name',),
            'noncolliding_alternatives': ('row',), 'load_live': ('name',), 'save_live': ('name',),
            'move_camera': ('x', 'y', 'z', 'yaw', 'pitch'), 'capture_view': ('x', 'y', 'z', 'yaw', 'pitch'),
            'get_object_bounds': ('actor',), 'get_world_bounds': ('scope',), 'sample_terrain': ('points',),
            'agent_log': ('lines',), 'inspect_site': ('bounds_min', 'bounds_max'), 'clearing_zones': ('action', 'name'),
            'list_groups': ('prefix',), 'edit_group': ('group',), 'undo_edit': ('transaction',),
            'undo_collision': ('transaction',), 'define_assembly': ('name',), 'check_assembly': ('name',),
            'set_collision': ('mode',), 'check_performance': ('name',), 'consolidate_scenery': ('name',),
            'check_route_clearance': ('source',), 'capture_sheet': ('poses',), 'scene_diff': ('snapshot',),
            'check_opening': ('path',), 'reference_load': ('obj',), 'scan_survey': ('obj',),
            'scan_height': ('points',), 'scan_profile': ('start', 'end'), 'scan_at': ('point',), 'scan_ground': ('box',),
            'scan_building': ('building',), 'scan_facade': ('building',), 'scan_free': ('points',),
            'scan_sheet': ('points',), 'scan_match': ('object_id',), 'scan_compare': ('snapshot',),
            'asset_profile': ('rows',), 'scan_grid': ('box',),
        }
        aliases = {}
        for name, op in api.OPERATIONS.items():
            params = {p.name: p for p in op.params()}
            self.assertEqual(op.positional, expected_positional.get(name, ()), name)
            for hint in (op.variadic, op.rest, op.stdin):
                self.assertTrue(hint is None or hint in params, name)
            if op.variadic:
                self.assertEqual(op.positional[-1], op.variadic)
            for alias in op.aliases:
                self.assertNotIn(alias, api.OPERATIONS)
                self.assertNotIn(alias, aliases)
                aliases[alias] = name
        self.assertEqual(aliases, {'exec': 'agent_exec', 'batch': 'agent_batch', 'log': 'agent_log', 'unload': 'agent_unload'})
        self.assertEqual({n: op.variadic for n, op in api.OPERATIONS.items() if op.variadic},
                         {'get_assets': 'ids', 'sample_terrain': 'points', 'check_opening': 'path',
                          'scan_height': 'points', 'scan_free': 'points', 'scan_sheet': 'points',
                          'asset_profile': 'rows'})
        self.assertEqual({n: op.rest for n, op in api.OPERATIONS.items() if op.rest}, {'agent_exec': 'command'})
        self.assertEqual({n: op.stdin for n, op in api.OPERATIONS.items() if op.stdin}, {'agent_batch': 'commands'})

    def test_examples_use_real_commands_and_options(self):
        for name, op in api.OPERATIONS.items():
            params = {p.name: p for p in op.params()}
            allowed = {f'--{p}' for p in params}
            allowed |= {f'--no-{p}' for p, param in params.items() if param.annotation in (bool, bool | None)}
            if 'preview' in params:
                allowed.add('--apply')
            if {'x0', 'y0', 'x1', 'y1'} <= set(params):
                allowed.add('--region')
            for example in op.examples:
                words = example.split()
                self.assertEqual(words[0], './uc', example)
                self.assertIn(words[1], (name, *op.aliases), example)
                self.assertNotIn('\n', example)
                if op.rest:
                    continue
                for word in words[2:]:
                    if re.match(r'--[A-Za-z]', word):
                        self.assertIn(word.split('=')[0], allowed, example)
        for name in ('sample_terrain', 'inspect_site', 'clearing_zones', 'edit_group', 'set_collision', 'define_assembly',
                     'check_performance', 'check_route_clearance', 'capture_view', 'agent_exec', 'agent_batch'):
            self.assertTrue(api.OPERATIONS[name].examples, name)

    def test_importing_api_loads_neither_mcp_nor_pil(self):
        code = ("import sys, uncrashed_mcp.api\n"
                "bad = [m for m in sys.modules if m.split('.')[0] in ('mcp', 'PIL', 'pydantic')"
                " or m in ('uncrashed_mcp.live', 'uncrashed_mcp.server')]\n"
                "print(bad)")
        env = dict(os.environ, UNCRASHED_OFFLINE='1',
                   PYTHONPATH=os.pathsep.join([str(Path(uncrashed_mcp.__file__).parents[1]), os.environ.get('PYTHONPATH', '')]))
        out = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, check=True).stdout
        self.assertEqual(out.strip(), '[]')


class PackagingTests(unittest.TestCase):
    def test_version_and_pyproject(self):
        text = (Path(__file__).parents[1] / 'pyproject.toml').read_text()
        self.assertEqual(re.search(r'^version = "([^"]+)"', text, re.M).group(1), uncrashed_mcp.__version__)
        for line in ('dependencies = ["pillow>=10"]', 'mcp = ["mcp>=2,<3"]', 'uncrashed = "uncrashed_mcp.cli:main"',
                     'uncrashed-mcp = "uncrashed_mcp.__main__:serve"'):
            self.assertIn(line, text.splitlines())

    def run_main(self, argv, tty, modules=None, entry_point='main'):
        from uncrashed_mcp import __main__ as entry
        cli, server = types.ModuleType('uncrashed_mcp.cli'), types.ModuleType('uncrashed_mcp.server')
        cli.main, server.main = Mock(return_value=7), Mock(return_value=None)
        stdin = None
        if tty is not None:  # None: the launcher closed fd 0, so Python has no sys.stdin
            stdin = Mock()
            stdin.isatty.return_value = tty
        out, err = io.StringIO(), io.StringIO()
        fakes = {'uncrashed_mcp.cli': cli, 'uncrashed_mcp.server': server} if modules is None else modules
        with patch.dict(sys.modules, fakes), patch.object(sys, 'stdin', stdin), \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            if modules is not None:
                sys.modules.pop('uncrashed_mcp.server', None)
            code = entry.main(argv) if entry_point == 'main' else entry.serve()
        return code, cli.main, server.main, out.getvalue(), err.getvalue()

    def test_module_entry_point_dispatch(self):
        code, cli, server, _out, _err = self.run_main(['status', '--json'], tty=False)
        self.assertEqual(code, 7)
        cli.assert_called_once_with(['status', '--json'])
        server.assert_not_called()

        code, cli, server, out, _err = self.run_main([], tty=True)
        self.assertEqual(code, 7)
        cli.assert_called_once_with(['help'])
        server.assert_not_called()
        self.assertIn('run `uncrashed mcp` for the MCP server', out)

        code, cli, server, _out, _err = self.run_main([], tty=False)  # an MCP client's `python -m uncrashed_mcp`
        self.assertEqual(code, 0)
        server.assert_called_once_with()
        cli.assert_not_called()

    def test_module_entry_point_without_the_mcp_package(self):
        hidden = {name: None for name in list(sys.modules) if name == 'mcp' or name.startswith('mcp.')}
        code, _cli, _server, out, err = self.run_main([], tty=False, modules={**hidden, 'mcp': None})
        self.assertEqual((code, out), (1, ''))
        self.assertRegex(err, r"^error: the MCP server needs the 'mcp' package.*\n$")

    def test_module_entry_point_with_stdin_closed_is_one_error_line(self):
        code, cli, server, out, err = self.run_main([], tty=None)
        self.assertEqual((code, out), (1, ''))
        self.assertRegex(err, r'^error: stdin is closed: the MCP server needs it.*\n$')
        self.assertEqual(len(err.splitlines()), 1)
        cli.assert_not_called()
        server.assert_not_called()
        code, cli, server, _out, _err = self.run_main(['status'], tty=None)  # with arguments stdin is not looked at
        self.assertEqual(code, 7)
        cli.assert_called_once_with(['status'])

    def test_the_mcp_script_serves_or_names_the_missing_extra(self):
        code, cli, server, out, err = self.run_main([], tty=True, entry_point='serve')  # serves on a terminal too
        self.assertEqual((code, out, err), (0, '', ''))
        server.assert_called_once_with()
        cli.assert_not_called()
        hidden = {name: None for name in list(sys.modules) if name == 'mcp' or name.startswith('mcp.')}
        code, _cli, _server, out, err = self.run_main([], tty=False, modules={**hidden, 'mcp': None},
                                                      entry_point='serve')
        self.assertEqual((code, out), (1, ''))
        self.assertEqual(err, "error: the MCP server needs the 'mcp' package: pip install 'uncrashed-mcp[mcp]'\n")

    def test_the_mcp_script_does_not_hide_other_import_errors(self):
        from uncrashed_mcp import __main__ as entry
        err = io.StringIO()
        with patch.dict(sys.modules, {'uncrashed_mcp.server': None}), contextlib.redirect_stderr(err):
            with self.assertRaises(ModuleNotFoundError) as caught:  # a missing module that is not the mcp package
                entry.serve()
        self.assertEqual((caught.exception.name, err.getvalue()), ('uncrashed_mcp.server', ''))


class GuideTests(unittest.TestCase):
    # Cautions of INSTRUCTIONS that the CLI guide must keep (whitespace-insensitive, lower case).
    KEYWORDS = (
        'BaseBoard', 'whole footprint', 'building roofs', 'drop last', 'bottom-up', '9 to 13 cm', 'read-only',
        'not full support or environment validation', 'never inferred from landscape size', 'foreground', '120-second',
        'GPU-only', 'allow_partial', 'sidecars', 'Blueprint', 'auto-saves', 'load capacity', 'certification',
        'origin_cm', 'extent_cm', 'flythrough', 'nophysics', 'physics_off', 'physics_on', 'force',
        'UncrashedMCP.collision', 'stock game', 'noncolliding', 'collision channels', 'unverified', '2 cm',
        'group rules survive moves', 'heuristic', '*_base', 'placed objects only', 'triplanar', 'canary', 'newest first',
        'thin them by hand', 'tiny_whoop', 'cinewhoop', 'fixed_wing', 'Overlap-only', 'not a flight test', 'mapinfo',
        'Spawn Asset', 'transaction', 'Pawn/PhysicsBody/Vehicle', 'game3',
    )
    # What the brief adds for shell use.
    CLI_TOPICS = (
        'Exit codes', 'pipefail', './uc batch plan.txt', '| ./uc batch -', 'not /tmp', './uc last', './uc exec mapinfo',
        './uc edit_history', './uc help verbs', 'explicit pose', 'own the editor camera', 'collision_prepared',
        'clear, destroy, destroyat, destroybox, replace, base', 'setmap', 'directly after its spawn', '--apply',
        'never re-send', 'partly applied', 'background', '--json', '--pick',
        # review additions: what the waits and busy errors mean, and the recoveries nothing else names
        './uc help output', '! acts at once, ~ previews unless --apply', 'waiting for agent lock',
        'agent is finishing an earlier job', 'means wait: do not re-send',
        'only an error containing `nothing was sent` is safe to retry', 'last finished reply',
        'batch error without a summary line', './uc unload', 'interrupted edit recovers its journal in the preview',
        'also reverts later changes', 'preview first, then --apply', 'set_collision (previews unless --apply)',
        'probed from the gate centre', 'NAME only previews a saved map file', 'save your work, load_live the map',
        'run without NAME',
    )

    @staticmethod
    def flat(text):
        return ' '.join(text.split()).lower()

    def test_guide_keeps_every_tool_name_and_caution_of_the_instructions(self):
        guide, instructions = self.flat(api.GUIDE), self.flat(api.INSTRUCTIONS)
        for name in api.OPERATIONS:
            self.assertIn(name, api.GUIDE, name)
        for keyword in self.KEYWORDS:
            self.assertIn(self.flat(keyword), instructions, f'{keyword!r} is not in INSTRUCTIONS any more: update KEYWORDS')
            self.assertIn(self.flat(keyword), guide, f'GUIDE lost {keyword!r}')

    def test_guide_covers_the_shell_topics(self):
        guide = self.flat(api.GUIDE)
        for topic in self.CLI_TOPICS:
            self.assertIn(self.flat(topic), guide, topic)
        for slow in ('check_collisions', 'load_live', 'consolidate_scenery --apply', 'check_route_clearance',
                     'check_opening'):
            self.assertIn(slow, self.flat(api.GUIDE.split('SLOW COMMANDS')[1].split('AUDITS')[0]))

    def test_guide_speaks_cli_and_stays_compact(self):
        self.assertLessEqual(len(api.GUIDE), 8000)
        for mcp_idiom in ('preview=False', 'persist=True', "backend='window'", 'foreground=True', 'server instructions'):
            self.assertNotIn(mcp_idiom, api.GUIDE)
        self.assertNotIn('On a saved map --apply', api.GUIDE)  # `consolidate_scenery NAME --apply` is refused
        commands = set(api.OPERATIONS) | {a for op in api.OPERATIONS.values() for a in op.aliases} | {'help', 'last'}
        for command in re.findall(r'\./uc (\w+)', api.GUIDE):
            self.assertIn(command, commands)
        for name, option in re.findall(r'`(\w+) (--[a-z_-]+)', api.GUIDE):
            if name in api.OPERATIONS:
                params = {p.name for p in api.OPERATIONS[name].params()}
                self.assertTrue(option == '--apply' or option[2:] in params or option[5:] in params, (name, option))


class CallTests(unittest.TestCase):
    def test_call_normalises_dicts_to_json_types(self):
        with patch.object(api.OPERATIONS['list_gates'], 'fn', return_value={'region': (1, 2), 3: 'int key'}):
            self.assertEqual(api.call('list_gates'), {'region': [1, 2], '3': 'int key'})
        with patch.object(api.OPERATIONS['list_gates'], 'fn', return_value={'path': Path('/x')}):
            with self.assertRaises(TypeError):
                api.call('list_gates')

    def test_call_passes_text_through_and_accepts_a_name_keyword(self):
        with patch.object(agent, 'log_tail', return_value='a\nb'):
            self.assertEqual(api.call('agent_log', lines=2), api.Text('a\nb'))
        with self.assertRaises(FileNotFoundError):
            api.call('inspect_map', name='no such map 7f3a')
        with self.assertRaises(KeyError):
            api.call('nope')

    def test_text_is_a_frozen_dataclass_not_a_str(self):
        text = api.Text('ok mapinfo')
        self.assertFalse(isinstance(text, str))
        self.assertEqual((text.text, text.ok, text.failure), ('ok mapinfo', True, ''))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            text.ok = False


class OfflineOperationTests(unittest.TestCase):
    def test_status_and_gates(self):
        with patch.object(api, 'game_running', return_value=False), patch.object(api, 'steam_id', return_value='1'), \
             patch.object(api, 'maps_dir', return_value=Path('/maps')), patch.object(api, 'workshop_dir', return_value=None):
            status = api.status()
        self.assertEqual(status['maps_dir'], '/maps')
        self.assertEqual((status['game_running'], status['live_agent'], status['workshop_items']), (False, False, 0))
        self.assertEqual((status['flat_base'], status['units']), ('BaseBoard', 'metres, yaw degrees, Z up'))
        self.assertGreater(status['assets'], 100)
        self.assertIn('STARTGATE', json.dumps(api.list_gates()))

    def test_search_ranks_whole_words_and_says_when_a_word_only_hides_inside_others(self):
        import dataclasses
        from uncrashed_mcp.catalog import Catalog
        real = api._catalog()
        base = real.assets['NewRow_52']
        rows = {r.id: r for r in (dataclasses.replace(base, id='R1', name='Logcabin Roof', tags=()),
                                  dataclasses.replace(base, id='R2', name='Zz Bin 01', tags=()))}
        cat = Catalog(rows, real.library)
        self.assertEqual([r.id for r in cat.search('bin')], ['R2', 'R1'])   # word before inner substring
        bin_ = api.search_assets('bin')                                      # the real catalogue: no bin row
        self.assertFalse(bin_['word_match'])
        self.assertIn('inside a word', bin_['word_note'])
        self.assertIn('NewRow_52', bin_['word_note'])                        # related: trash box
        self.assertNotIn('word_match', api.search_assets('bench'))

    def test_search_assets_clamps_limit_before_and_after_the_collision_filter(self):
        self.assertEqual(api.search_assets(limit=0)['count'], 1)
        self.assertEqual(api.search_assets(limit=-5)['count'], 1)
        capped = api.search_assets(limit=500)
        self.assertEqual(capped['count'], api.SEARCH_CAP)
        # a cut list says so: how many matched and that the limit was above the cap
        self.assertTrue(capped['truncated'])
        self.assertGreater(capped['total'], api.SEARCH_CAP)
        self.assertIn('above the cap of 200', capped['note'])
        few = api.search_assets(limit=7)
        self.assertEqual((few['count'], few['truncated']), (7, True))
        self.assertIn('more rows match', few['note'])
        found = api.search_assets(collision='noncolliding', limit=500)
        self.assertEqual(found['count'], len(found['assets']))
        self.assertTrue(0 < found['count'] <= api.SEARCH_CAP)
        self.assertNotIn('truncated', api.search_assets('NewRow_1', limit=5000))
        catalog = api._catalog()
        rows = Mock(wraps=catalog.search)
        with patch.object(catalog, 'search', rows), patch.object(api, '_catalog', return_value=catalog):
            api.search_assets('wall', category='Decals', tag='x', limit=3, include_hidden=False, collision='colliding')
            self.assertEqual(rows.call_args.kwargs, {'query': 'wall', 'category': 'Decals', 'tag': 'x',
                                                     'limit': len(catalog.assets), 'include_hidden': False,
                                                     'include_decals': True})
            api.search_assets(limit=99)
            self.assertEqual(rows.call_args.kwargs['limit'], len(catalog.assets))   # counted, then cut
            self.assertEqual((rows.call_args.kwargs['category'], rows.call_args.kwargs['tag']), (None, None))

    def test_get_assets_reports_missing_ids(self):
        result = api.get_assets(['NewRow_1', 'nope'])
        self.assertEqual(result['missing'], ['nope'])
        self.assertEqual([a['id'] for a in result['assets']], ['NewRow_1'])
        self.assertEqual(api.get_assets([]), {'assets': [], 'missing': []})

    def test_maps(self):
        with patch.object(api, 'maps_dir', return_value=Path('/maps')), patch.object(api, 'list_user_maps', return_value=[{'folder': 'A'}]):
            self.assertEqual(api.list_maps(), {'maps_dir': '/maps', 'maps': [{'folder': 'A'}]})
        with tempfile.TemporaryDirectory() as tmp, patch.object(api, 'maps_dir', return_value=Path(tmp)):
            with self.assertRaisesRegex(FileNotFoundError, r"no local map 'no_such' in "):
                api.inspect_map('no such')
        doc = Mock()
        doc.summary.return_value = {'instances': 3}
        with patch.object(api, '_require', return_value=doc) as require, patch.object(api, '_catalog', return_value='CAT'):
            self.assertEqual(api.inspect_map('Demo'), {'instances': 3})
        require.assert_called_once_with('Demo')
        doc.summary.assert_called_once_with('CAT')


class DelegationTests(unittest.TestCase):
    """Argument order and renames of the operations that only forward to a module function."""

    def setUp(self):
        catalog = patch.object(api, '_catalog', return_value='CAT')
        catalog.start()
        self.addCleanup(catalog.stop)

    def test_positional_forwarding(self):
        cases = [
            (api.managed, 'inspect_site', 'inspect_site', dict(bounds_min=[0, 0, 0], bounds_max=[1, 1, 1], spacing=2, min_foliage_height=0, top=3),
             [[0, 0, 0], [1, 1, 1], 2, 0, 3]),
            (api.managed, 'clearing_zones', 'clearing_zones', dict(action='upsert', name='yard', center=[1, 2, 3], size=[4, 5, 6], yaw=15, preview=False, allow_partial=True),
             ['upsert', 'yard', [1, 2, 3], [4, 5, 6], 15, False, True]),
            (api.managed, 'clearing_zones', 'clearing_zones', {}, ['list', '', None, None, 0, True, False]),
            (api.groups, 'list_groups', 'list_groups', dict(prefix='h', limit=5), ['h', 5]),
            (api.groups, 'edit_group', 'edit_group', dict(group='g', action='clone', delta=[1, 0, 0], rotation=[0, 90, 0], scale=2, pivot=[0, 0, 0],
                                                         target_group='h', prefix=True, actors=['1A'], preview=False),
             ['g', 'clone', [1, 0, 0], [0, 90, 0], 2, [0, 0, 0], 'h', True, ['1A'], False]),
            (api.groups, 'edit_group', 'edit_group', {}, ['', 'transform', None, None, 1, None, '', False, None, True]),
            (api.groups, 'undo_edit', 'undo_edit', dict(transaction='t'), ['t', True]),
            (api.groups, 'edit_history', 'edit_history', {}, [20]),
            (api.assemblies, 'define_assembly', 'define_assembly', dict(name='tower', roles={'a': 'foundation'}, preview=False),
             ['tower', {'a': 'foundation'}, False]),
            (api.assemblies, 'check_assembly', 'check_assembly', dict(name='tower'), ['CAT', 'tower', .05, .3, .5, 20]),
            (api.scenery, 'set_collision', 'set_collision', dict(mode='off', actors=['1A'], group='g', row='r', box_min=[0, 0, 0], box_max=[1, 1, 1],
                                                                channels=['pawn'], keep_body=True, force=True, preview=False, persist=False, top=5),
             ['off', ['1A'], 'g', 'r', [0, 0, 0], [1, 1, 1], ['pawn'], True, True, False, False, 5]),
            (api.scenery, 'set_collision', 'set_collision', {}, ['flythrough', None, '', '', None, None, None, False, False, True, True, 30]),
            (api.scenery, 'collision_state', 'collision_state', dict(group='g', components=False), [None, 'g', '', None, None, None, 30, False]),
            (api.scenery, 'undo_collision', 'undo_collision', dict(transaction='t', preview=False), ['t', False]),
            (api.scenery, 'lookalikes', 'noncolliding_alternatives', dict(row='NewRow_4'), ['CAT', 'NewRow_4', 5]),
            (api.environment, 'get_object_bounds', 'get_object_bounds', dict(actor='1A'), ['1A']),
            (api.environment, 'get_world_bounds', 'get_world_bounds', {}, ['placed']),
            (agent, 'load', 'load_live', dict(name='M', move_camera=False), ['M', False]),
            (agent, 'save', 'save_live', dict(name='M', display_name='T'), ['M', 'T']),
            (agent, 'save', 'save_live', {}, ['', '']),
            (agent, 'unload', 'agent_unload', {}, []),
        ]
        with tempfile.TemporaryDirectory() as tmp, patch.object(api, 'maps_dir', return_value=Path(tmp)), \
                patch.object(api.reference, 'drop_for_scene_change', return_value=''):
            (Path(tmp) / 'M').mkdir()
            (Path(tmp) / 'M' / 'M.json').write_text('{}')  # load_live looks for the map before it takes the lock
            for module, target, name, kwargs, expected in cases:
                with patch.object(module, target, echo):
                    self.assertEqual(api.call(name, **kwargs), {'args': expected, 'kwargs': {}}, (name, kwargs))

    def test_load_live_names_a_map_that_does_not_exist_before_reaching_the_agent(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(api, 'maps_dir', return_value=Path(tmp)), \
                patch.object(agent, 'load') as load:
            for name, folder in (('X', 'X'), ('Some Display Title', 'Some_Display_Title')):
                with self.assertRaises(FileNotFoundError) as caught:
                    api.load_live(name)
                self.assertEqual(str(caught.exception), f'no local map {folder!r} in {tmp}')
            load.assert_not_called()

    def test_audits_validate_the_region_and_forward_keywords(self):
        with patch.object(api.environment, 'audit', echo):
            self.assertEqual(api.call('check_foundations', x0=0, y0=1, x1=2, y1=3, actors=['1A'], top=4), {
                'args': ['CAT', 'foundation'],
                'kwargs': {'actors': ['1A'], 'region': [0, 1, 2, 3], 'spacing': .75, 'tolerance': .3, 'max_drop': 50,
                           'include_canopies': False, 'include_structural': False, 'top': 4}})
            self.assertEqual(api.call('check_environment', include_canopies=False), {
                'args': ['CAT', 'environment'],
                'kwargs': {'actors': None, 'region': None, 'spacing': .75, 'tolerance': .1, 'include_canopies': False,
                           'include_structural': False, 'top': 15}})

    def test_performance_and_consolidate(self):
        with patch.object(api.performance, 'check_performance', echo):
            for top, clamped in ((0, 1), (-3, 1), (10, 10), (999, 50)):
                result = api.call('check_performance', name='M', budgets={'placed_objects': 6000}, top=top)
                self.assertEqual(result['args'], ['CAT'])
                self.assertEqual(result['kwargs'], {'name': 'M', 'region': None, 'budget': {'placed_objects': 6000},
                                                    'measure': False, 'poses': None, 'duration': 1.5, 'cell': 50,
                                                    'top': clamped})
        with patch.object(api.consolidate, 'consolidate_scenery', echo):
            self.assertEqual(api.call('consolidate_scenery', rows=['NewRow_1'], preview=False, x0=0, y0=0, x1=5, y1=5), {
                'args': ['CAT'], 'kwargs': {'name': '', 'region': [0, 0, 5, 5], 'rows': ['NewRow_1'], 'preview': False,
                                            'tolerance': .02, 'max_size': 64, 'limit': 20}})

    def test_scene_snapshot_and_diff_forward_by_name(self):
        with patch.object(api.snapshots, 'scene_snapshot', echo), patch.object(api.snapshots, 'scene_diff', echo):
            self.assertEqual(api.call('scene_snapshot', box=[0, 0, 1, 1], name='n'),
                             {'args': [], 'kwargs': {'group': '', 'box': [0, 0, 1, 1], 'name': 'n'}})
            self.assertEqual(api.call('scene_diff', 's', examples=3), {'args': ['s'], 'kwargs': {
                'group': None, 'box': None, 'position_m': .01, 'rotation_deg': .1, 'scale': .001,
                'gate_position_m': .01, 'examples': 3}})

    def test_route_clearance_forwards_every_parameter_by_name(self):
        with patch.object(api.clearance, 'check_route_clearance', echo):
            result = api.call('check_route_clearance', source='points', points=[[0, 0, 5]], profile='car')
        expected = {p.name: p.default for p in api.OPERATIONS['check_route_clearance'].params()}
        expected.update(source='points', points=[[0, 0, 5]], profile='car', catalog='CAT')
        self.assertEqual(result, {'args': [], 'kwargs': expected})

    def test_region_validation_errors_raise_before_any_work(self):
        for name, module, target in (('check_foundations', api.environment, 'audit'), ('check_environment', api.environment, 'audit'),
                                     ('check_performance', api.performance, 'check_performance'),
                                     ('consolidate_scenery', api.consolidate, 'consolidate_scenery')):
            with patch.object(module, target) as work:
                for region in ({'x0': 1}, {'x0': 0, 'y0': 0, 'x1': 10}, {'x0': 10, 'y0': 0, 'x1': 0, 'y1': 10}):
                    with self.assertRaises(ValueError, msg=(name, region)):
                        api.call(name, **region)
                work.assert_not_called()

    def test_errors_of_the_delegate_are_raised_unchanged(self):
        error = agent.AgentError('Uncrashed is not running')
        with patch.object(api.groups, 'list_groups', side_effect=error):
            with self.assertRaises(agent.AgentError) as caught:
                api.call('list_groups')
        self.assertIs(caught.exception, error)


class CameraTests(unittest.TestCase):
    POSE = {'x': 1.0, 'y': 2.0, 'z': 3.0, 'pitch': -20.0, 'yaw': 90.0, 'roll': 0.0, 'fov': 90.0, 'held': True, 'backend': 'native'}

    def test_camera_pose_and_move(self):
        from uncrashed_mcp import live
        with patch.object(live, 'read_camera', return_value=dict(self.POSE)), patch.object(live, 'move_camera', echo):
            self.assertEqual(api.camera_pose(), self.POSE)
            self.assertEqual(api.move_camera(1, 2, 3, 45), {'args': [1, 2, 3, 45, -20, True], 'kwargs': {}})
            self.assertEqual(api.move_camera(1, 2, 3, 45, pitch=-60, hold=False)['args'], [1, 2, 3, 45, -60, False])

    def test_release_reads_the_pose_after_releasing(self):
        from uncrashed_mcp import live
        calls = []
        with patch.object(live, 'release_camera', side_effect=lambda: calls.append('release')), \
             patch.object(live, 'read_camera', side_effect=lambda: calls.append('read') or dict(self.POSE, held=False)):
            self.assertEqual(api.release_camera(), dict(self.POSE, held=False))
        self.assertEqual(calls, ['release', 'read'])
        with patch.object(live, 'release_camera'), patch.object(live, 'read_camera', return_value=dict(self.POSE)):
            self.assertIs(api.release_camera()['held'], True)  # the camera's own answer wins, as before the port
        with patch.object(live, 'release_camera', side_effect=live.LiveError('no camera')), patch.object(live, 'read_camera') as read:
            with self.assertRaises(live.LiveError):
                api.release_camera()
            read.assert_not_called()


class CaptureViewTests(unittest.TestCase):
    def setUp(self):
        from uncrashed_mcp import live
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        self.shots = self.tmp / 'shots'
        self.shots.mkdir()
        self.captures = []

        def capture(x=None, y=None, z=None, yaw=None, pitch=None, **kwargs):
            self.captures.append(((x, y, z, yaw, pitch), kwargs))
            path = self.shots / f'view-{len(self.captures)}.png'
            path.write_bytes(b'PNG%d' % len(self.captures))
            path.with_suffix('.json').write_text(json.dumps({'image': str(path), 'camera': {'x': x}, 'render_size': (1600, 900)}))
            return path

        patcher = patch.object(live, 'capture_view', capture)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_without_out_the_image_stays_and_the_sidecar_is_returned(self):
        meta = api.call('capture_view', x=1, y=2, z=3, yaw=4, pitch=-5, backend='window', foreground=True, width=320, height=200)
        self.assertEqual(meta, {'image': str(self.shots / 'view-1.png'), 'camera': {'x': 1}, 'render_size': [1600, 900]})
        self.assertEqual(self.captures, [((1, 2, 3, 4, -5), {'backend': 'window', 'foreground': True, 'width': 320, 'height': 200})])
        self.assertTrue((self.shots / 'view-1.json').exists())

    def test_out_file_moves_the_image_and_rewrites_the_sidecar(self):
        target = self.tmp / 'new' / 'dir' / 'overview.png'
        meta = api.capture_view(out=str(target))
        self.assertEqual(meta['image'], str(target))
        self.assertEqual(target.read_bytes(), b'PNG1')
        self.assertEqual(json.loads(target.with_suffix('.json').read_text()), meta)
        self.assertEqual(list(self.shots.iterdir()), [])
        api.capture_view(out=str(target))  # an existing file is overwritten
        self.assertEqual(target.read_bytes(), b'PNG2')
        self.assertEqual(sorted(p.name for p in target.parent.iterdir()), ['overview.json', 'overview.png'])

    def test_out_directory_keeps_the_generated_name(self):
        existing = self.tmp / 'existing'
        existing.mkdir()
        self.assertEqual(api.capture_view(out=str(existing))['image'], str(existing / 'view-1.png'))
        fresh = self.tmp / 'fresh'
        meta = api.capture_view(out=str(fresh) + '/')
        self.assertEqual(meta['image'], str(fresh / 'view-2.png'))
        self.assertEqual(json.loads((fresh / 'view-2.json').read_text())['image'], str(fresh / 'view-2.png'))
        self.assertEqual(api.capture_view(out=str(self.shots))['image'], str(self.shots / 'view-3.png'))
        self.assertEqual(json.loads((self.shots / 'view-3.json').read_text())['image'], str(self.shots / 'view-3.png'))

    def test_out_is_made_absolute(self):
        before = os.getcwd()
        os.chdir(self.tmp)
        try:
            self.assertEqual(api.capture_view(out='rel/shot.png')['image'], str(self.tmp / 'rel' / 'shot.png'))
        finally:
            os.chdir(before)
        with patch.dict(os.environ, {'HOME': str(self.tmp)}):
            self.assertEqual(api.capture_view(out='~/home.png')['image'], str(self.tmp / 'home.png'))

    def test_bad_out_is_rejected_before_anything_is_captured(self):
        for bad in (str(self.tmp / 'shot.jpg'), str(self.tmp / 'noext')):
            with self.assertRaisesRegex(ValueError, r'\.png'):
                api.capture_view(out=bad)
        self.assertEqual(self.captures, [])

    def test_an_unusable_target_folder_is_rejected_before_anything_is_captured(self):
        taken = self.tmp / 'taken.png'
        taken.write_bytes(b'old')
        for out in (str(taken) + '/', str(taken / 'sub' / 'x.png')):  # a file where a folder is needed
            with self.assertRaises(OSError) as caught:
                api.capture_view(out=out)
            self.assertIn('nothing was captured', str(caught.exception))
            self.assertIn(str(taken), str(caught.exception))
        self.assertIsInstance(caught.exception, (NotADirectoryError, FileExistsError))
        self.assertEqual(taken.read_bytes(), b'old')
        self.assertEqual(self.captures, [])

    @unittest.skipIf(os.geteuid() == 0, 'root may write anywhere')
    def test_a_read_only_target_folder_is_rejected_before_anything_is_captured(self):
        locked = self.tmp / 'locked'
        locked.mkdir()
        locked.chmod(0o500)
        self.addCleanup(locked.chmod, 0o700)
        for out in (str(locked / 'x.png'), str(locked) + '/', str(locked / 'sub') + '/'):
            with self.assertRaises(PermissionError) as caught:
                api.capture_view(out=out)
            self.assertIn('nothing was captured', str(caught.exception))
            self.assertIn(str(locked), str(caught.exception))
        self.assertEqual(self.captures, [])

    def test_a_failed_move_says_where_the_capture_is(self):
        target = self.tmp / 'full' / 'overview.png'
        with patch.object(api.shutil, 'move', side_effect=OSError(28, 'No space left on device')):
            with self.assertRaises(OSError) as caught:
                api.capture_view(out=str(target))
        self.assertEqual(str(caught.exception),
                         f"[Errno 28] No space left on device; the capture is at {self.shots / 'view-1.png'}")
        self.assertTrue((self.shots / 'view-1.png').exists() and (self.shots / 'view-1.json').exists())
        with patch.object(api.shutil, 'move', side_effect=PermissionError(13, 'Permission denied')):
            with self.assertRaises(PermissionError) as caught:  # the class survives for callers that test it
                api.capture_view(out=str(target))
        self.assertIn(f"the capture is at {self.shots / 'view-2.png'}", str(caught.exception))
        with patch.object(api.managed, 'atomic_write', side_effect=OSError(28, 'No space left on device')):
            with self.assertRaises(OSError) as caught:
                api.capture_view(out=str(target))
        self.assertIn(f"the image is at {target}, its metadata at {self.shots / 'view-3.json'}", str(caught.exception))
        self.assertEqual(target.read_bytes(), b'PNG3')


class AgentExecTests(unittest.TestCase):
    def reply(self, ok=True, head='ok mapinfo', lines=('folder Demo', 'objects 3')):
        return {'seq': 7, 'ok': ok, 'head': head, 'lines': list(lines)}

    def test_ok_reply(self):
        with patch.object(agent, 'execute', return_value=self.reply()) as execute:
            result = api.agent_exec('mapinfo', timeout=5)
        execute.assert_called_once_with('mapinfo', 5)
        self.assertEqual(result, api.Text('ok ok mapinfo\nfolder Demo\nobjects 3'))

    def test_failed_reply_is_returned_not_raised(self):
        with patch.object(agent, 'execute', return_value=self.reply(False, 'ok spawn', ['error no row NewRow_9999'])):
            result = api.agent_exec('spawn NewRow_9999 0 0 0')
        self.assertEqual(result.text, 'failed ok spawn\nerror no row NewRow_9999')
        self.assertFalse(result.ok)
        self.assertIn('no row NewRow_9999', result.failure)
        self.assertNotIn('\n', result.failure)
        with patch.object(agent, 'execute', return_value=self.reply(False, 'fault in spawn', [])):
            self.assertIn('fault in spawn', api.agent_exec('spawn x').failure)

    def test_the_failure_reason_is_never_the_ok_head(self):
        cases = (
            (dict(self.reply(False, 'ok batch', []), ended=False),
             'the agent reported: agent job ended without an end marker (fault?)'),
            (dict(self.reply(False, 'ok batch', ['#0 spawned 7f00 row=A', '1006 fault c0000005 rip=1234 rva=0']), ended=True),
             'the agent reported: agent job faulted: 1006 fault c0000005 rip=1234 rva=0'),
            (dict(self.reply(False, 'ok spawn', ['error no row X', 'error bad args']), ended=True),
             'the agent reported: no row X; bad args'),
            (dict(self.reply(False, 'ok spawn', ['odd line']), ended=True),
             'the agent reported: agent job failed without an error line; odd line'),
        )
        for reply, failure in cases:
            with patch.object(agent, 'execute', return_value=reply):
                result = api.agent_exec('x')
            self.assertEqual((result.ok, result.failure), (False, failure))
            self.assertTrue(result.text.startswith(f"failed {reply['head']}"))

    def test_reply_is_clipped_between_one_and_the_hard_cap(self):
        lines = [f'actor {i}' for i in range(1000)]
        with patch.object(agent, 'execute', return_value=self.reply(lines=lines)):
            self.assertEqual(len(api.agent_exec('actors').text.split('\n')), 1 + 40 + 1)
            self.assertEqual(api.agent_exec('actors', max_lines=5).text.split('\n')[-1], '... 995 more lines (max_lines up to 400)')
            self.assertEqual(len(api.agent_exec('actors', max_lines=0).text.split('\n')), 1 + 1 + 1)
            self.assertEqual(len(api.agent_exec('actors', max_lines=5000).text.split('\n')), 1 + 400 + 1)
        with patch.object(agent, 'execute', return_value=self.reply(lines=lines[:400])):
            self.assertEqual(len(api.agent_exec('actors', max_lines=400).text.split('\n')), 401)

    def test_transport_errors_raise(self):
        with patch.object(agent, 'execute', side_effect=agent.AgentError('Uncrashed is not running')):
            with self.assertRaisesRegex(agent.AgentError, 'not running'):
                api.agent_exec('mapinfo')

    def test_agent_log(self):
        with patch.object(agent, 'log_tail', return_value='one\ntwo') as tail:
            self.assertEqual(api.agent_log(2), api.Text('one\ntwo'))
        tail.assert_called_once_with(2)

    def test_an_empty_or_missing_agent_log_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'agent.log'
            with patch.object(agent, 'LOG', log):
                self.assertEqual(api.agent_log(), api.Text(f'(agent log is missing: {log})'))
                log.write_text('')
                self.assertEqual(api.agent_log(), api.Text(f'(agent log is empty: {log})'))
                log.write_text('\n \n')
                self.assertEqual(api.agent_log(), api.Text(f'(agent log is empty: {log})'))
                log.write_text('job 7 mapinfo 2ms\n')
                self.assertEqual(api.agent_log(), api.Text('job 7 mapinfo 2ms'))


class AgentBatchTests(unittest.TestCase):
    def run_batch(self, commands, results=None, summary=None, **kwargs):
        sent = []

        def batch(lines, timeout=None, quiet=False):
            sent.append((list(lines), timeout, quiet))
            return (results or {}), (f'batch lines={len(lines)} ok={len(lines)} failed=0' if summary is None else summary)

        with patch.object(agent, 'batch', batch):
            return api.agent_batch(commands, **kwargs), sent

    def test_entries_are_normalised_and_comments_dropped_like_the_native_batch(self):
        result, sent = self.run_batch(['  spawn   NewRow_1\t0 0 0 ', '', '   ', '# a comment', '#spawn disabled', ' # indented',
                                       'drop  last', 'console r.Foo #1'])
        self.assertEqual(sent, [(['spawn NewRow_1 0 0 0', 'drop last', 'console r.Foo #1'], None, True)])
        self.assertEqual(result, api.Text('batch lines=3 ok=3 failed=0 sent=3'))

    def test_nothing_left_to_send(self):
        for commands in ([], ['', '  '], ['# only', '#comments']):
            with patch.object(agent, 'batch') as batch:
                with self.assertRaisesRegex(ValueError, '^no commands$'):
                    api.agent_batch(commands)
            batch.assert_not_called()

    def test_lines_the_native_reader_would_split_are_rejected(self):
        ok = 'spawn ' + 'x' * (4095 - 6)
        self.assertEqual(len(ok), 4095)
        result, sent = self.run_batch([ok])
        self.assertTrue(result.ok)
        for too_long, where in ((ok + 'y', 'entry 1'), ('ä' * 2048, 'entry 1')):
            with patch.object(agent, 'batch') as batch:
                with self.assertRaisesRegex(ValueError, where):
                    api.agent_batch(['mapinfo', too_long])
            batch.assert_not_called()
        with patch.object(agent, 'batch') as batch:
            with self.assertRaisesRegex(ValueError, 'line 12 is 4096 bytes'):
                api.agent_batch(['mapinfo', ok + 'y'], _lines=[11, 12])

    def test_options_reach_the_agent(self):
        _result, sent = self.run_batch(['mapinfo'], quiet=False, timeout=600.0)
        self.assertEqual(sent, [(['mapinfo'], 600.0, False)])

    def test_failures_are_listed_with_their_command_and_the_retry_rule(self):
        long_command = 'spawn NewRow_1 ' + '1 ' * 100
        commands = ['spawn NewRow_1 0 0 0', '# skipped', 'spawn NewRow_9999 100 200 0', long_command, 'destroy 1a2b']
        results = {1: ['error no row NewRow_9999 in DT_MapEditorAssets'], 2: ['error bad', 'error args'], 3: ['destroyed 1a2b']}
        result, _sent = self.run_batch(commands, results, 'batch lines=4 ok=2 failed=2')
        clipped = ' '.join(long_command.split())[:117] + '...'
        self.assertEqual(result.text.split('\n'), [
            'batch lines=4 ok=2 failed=2 sent=4',
            'failures:',
            '#1 error no row NewRow_9999 in DT_MapEditorAssets << spawn NewRow_9999 100 200 0',
            f'#2 error bad error args << {clipped}',
            'successful lines are applied; retry only the failed lines',
        ])
        self.assertEqual(len(clipped), 120)
        self.assertFalse(result.ok)
        self.assertEqual(result.failure, 'batch applied 2/4 lines, 2 failed (#1 #2); do not re-run the whole file')

    def test_source_line_numbers(self):
        commands = ['spawn A 0 0 0', '', '# note', 'spawn B 0 0 0', 'spawn C 0 0 0']
        result, sent = self.run_batch(commands, {1: ['error no row B']}, 'batch lines=3 ok=2 failed=1', _lines=[10, 11, 12, 13, 14])
        self.assertEqual(sent[0][0], ['spawn A 0 0 0', 'spawn B 0 0 0', 'spawn C 0 0 0'])
        self.assertEqual(result.text.split('\n')[2], '#1 L13 error no row B << spawn B 0 0 0')
        with self.assertRaises(ValueError):
            api.agent_batch(['a', 'b'], _lines=[1])

    def test_failure_list_is_capped_and_verbose_replies_follow(self):
        commands = [f'bad {i}' for i in range(30)]
        results = {i: [f'error {i}'] for i in range(30)}
        summary = 'batch lines=30 ok=0 failed=30'
        lines = self.run_batch(commands, results, summary)[0].text.split('\n')
        self.assertEqual(len(lines), 1 + 1 + 20 + 1 + 1)
        tail = 'more failed lines: run ./uc last now, before another agent command replaces that reply, to list them all'
        self.assertEqual(lines[22:], [f'... 10 {tail}', api.BATCH_RETRY])
        lines = self.run_batch(commands, results, summary, max_lines=400)[0].text.split('\n')
        self.assertEqual(lines[22:], [f'... 10 {tail}', api.BATCH_RETRY])  # max_lines never lifts the cap of 20
        self.assertNotIn('max_lines up to', '\n'.join(lines))
        lines = self.run_batch(commands, results, summary, max_lines=3)[0].text.split('\n')
        self.assertEqual(lines[5:], [f'... 27 {tail}', api.BATCH_RETRY])
        lines = self.run_batch(commands[:20], {i: [f'error {i}'] for i in range(20)}, 'batch lines=20 ok=0 failed=20')[0]
        self.assertEqual(len(lines.text.split('\n')), 1 + 1 + 20 + 1)  # exactly the cap: no tail
        result = self.run_batch(commands, results, summary, quiet=False, max_lines=2)[0]
        self.assertEqual(result.text.split('\n')[6:], ['#0 error 0', '#1 error 1', '... 28 more lines (max_lines up to 400)'])
        self.assertIn('30 failed (#0 #1 #2 #3 #4 #5 #6 #7 #8 #9 ...)', result.failure)
        verbose = self.run_batch(['a', 'b'], {0: ['spawn 1A', 'more'], 1: ['spawn 2B']}, quiet=False)[0]
        self.assertEqual(verbose, api.Text('batch lines=2 ok=2 failed=0 sent=2\n#0 spawn 1A\n#0 more\n#1 spawn 2B'))

    def test_ok_needs_a_clean_summary_that_matches_what_was_sent(self):
        self.assertTrue(self.run_batch(['a', 'b'])[0].ok)
        missing = self.run_batch(['a', 'b'], summary='')[0]
        self.assertFalse(missing.ok)
        self.assertEqual(missing.text, 'batch summary=missing sent=2')
        self.assertIn('no summary', missing.failure)
        self.assertIn('UNKNOWN', missing.failure)
        short = self.run_batch(['a', 'b', 'c'], summary='batch lines=2 ok=2 failed=0')[0]
        self.assertFalse(short.ok)
        self.assertIn('ran 2 of 3 lines', short.failure)
        counted = self.run_batch(['a', 'b'], summary='batch lines=2 ok=1 failed=1')[0]  # failure line lost, count kept
        self.assertFalse(counted.ok)
        self.assertEqual(counted.failure, 'batch applied 1/2 lines, 1 failed; do not re-run the whole file')
        unlisted = self.run_batch(['a', 'b'], {1: ['error late']}, 'batch lines=2 ok=2 failed=0')[0]
        self.assertFalse(unlisted.ok)
        for result in (missing, short, counted, unlisted):
            self.assertNotIn('\n', result.failure)

    UNKNOWN = 'batch state UNKNOWN: may be partly applied; check with exec mapinfo / actors before retrying'

    def raised(self, error):
        with patch.object(agent, 'batch', side_effect=error):
            with self.assertRaises(agent.AgentError) as caught:
                api.agent_batch(['spawn A 0 0 0'])
        return caught.exception

    def test_reply_timeout_says_the_state_is_unknown(self):
        timeout = agent.AgentTimeout("the agent did not answer 'batch Z:\\\\tmp\\\\x' within 60s; the command was sent "
                                     "and may still run")
        got = self.raised(timeout)
        self.assertIs(type(got), agent.AgentTimeout)  # the caller still sees a timeout
        self.assertEqual(str(got), f'{timeout}; {self.UNKNOWN}')
        self.assertIs(got.__cause__, timeout)
        self.assertIsNone(got.reply)

    def test_nothing_sent_passes_through_unchanged(self):
        for message in ('Uncrashed is not running', 'agent busy: lock held by pid 1 (x) for 60s; nothing was sent',
                        'offline mode (UNCRASHED_OFFLINE set): live agent calls are disabled',
                        # worded like a sent command, but the type decides
                        'the agent did not answer the door; may still run'):
            error = agent.AgentNotSent(message)
            self.assertIs(self.raised(error), error)

    def test_a_job_that_answered_failed_is_unknown_too(self):
        # a fault inside one line: the lines before it are applied, and the message alone reads like a clean failure
        reply = {'seq': 9, 'ok': False, 'head': 'ok batch', 'lines': ['#0 spawned 7f00 row=A', '#1 error no row in bad 2'],
                 'ended': False}
        failed = agent.AgentError(agent._failure(reply))
        failed.reply = reply
        got = self.raised(failed)
        self.assertIs(type(got), agent.AgentError)
        self.assertEqual(str(got), 'batch job failed after 2 reply lines: agent job ended without an end marker (fault?); '
                                   f'no row in bad 2; {self.UNKNOWN}')
        self.assertIs(got.reply, reply)  # still attached for the caller
        self.assertIs(got.__cause__, failed)
        self.assertNotIn('ok batch', str(got))

    def test_an_agent_lost_after_sending_is_unknown_too(self):
        lost = agent.AgentError("the agent is not loaded (unloaded, or the game closed); 'batch Z:\\\\tmp\\\\b.txt quiet' "
                                "was sent before that and may have run")
        got = self.raised(lost)
        self.assertEqual(str(got), f'{lost}; {self.UNKNOWN}')
        self.assertIsNone(got.reply)


class SampleTerrainTests(unittest.TestCase):
    def test_reply_errors_raise(self):
        with patch.object(agent, 'batch', return_value=({0: ['error no landscape']}, 'ok')):
            with self.assertRaisesRegex(RuntimeError, 'terrain sample 0 failed: error no landscape'):
                api.sample_terrain([[1, 2]])
        with patch.object(agent, 'batch', return_value=({0: ['terrain x=1 y=2 z=5'], 1: ['terrain x=1 y=2 z=5', 'error late']}, 'ok')):
            with self.assertRaisesRegex(RuntimeError, 'terrain sample 1 failed'):
                api.sample_terrain([[1, 2], [3, 4]])

    def test_each_validation_error(self):
        nan, inf = float('nan'), float('inf')
        with patch.object(agent, 'batch') as batch:
            for kwargs, message in (
                ({'points': []}, 'provide 1 to 256'), ({'points': [[1, 2]] * 257}, 'provide 1 to 256'),
                ({'points': [[1]]}, 'two finite'), ({'points': [[1, 2, 3]]}, 'two finite'), ({'points': [[1, nan]]}, 'two finite'),
                ({'points': [[inf, 2]]}, 'two finite'), ({'points': [[1, 2]], 'z_from': 0, 'z_to': 1}, 'z_from must be finite'),
                ({'points': [[1, 2]], 'z_from': 5, 'z_to': 5}, 'z_from must be finite'),
                ({'points': [[1, 2]], 'z_from': nan}, 'z_from must be finite'), ({'points': [[1, 2]], 'z_to': -inf}, 'z_from must be finite'),
            ):
                with self.assertRaisesRegex(ValueError, message, msg=kwargs):
                    api.sample_terrain(**kwargs)
            batch.assert_not_called()


class CheckCollisionsTests(unittest.TestCase):
    BOUNDS = {'available': True, 'truncated': False, 'actor_count': 1700, 'aabb': {'min': [100, -80, -10], 'max': [110, -70, 10]}}

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.reports = Path(tmp.name) / 'reports'
        self.scan_dirs = []
        self.entered = []

        def run(command, timeout=30.0):
            path = Path(command.split()[1][2:].replace('\\', '/'))
            self.scan_dirs.append(path.parent)
            self.assertEqual((self.entered[-1], timeout), ('session', 120))  # every scan pass runs inside the session
            path.write_text('O a NewRow_33 1000000 -800000 -1000 1000100 -799900 -900\nR a 0\n')
            return []

        @contextlib.contextmanager
        def session():
            self.entered.append('session')
            yield
            self.entered.append('left')

        self.run_mock = self.start(patch.object(agent, 'run', side_effect=run))
        self.start(patch.object(api, 'COLLIDE_DIR', self.reports))
        self.start(patch.object(agent, 'session', session))
        self.start(patch.object(api, '_catalog', return_value=None))
        self.start(patch.object(api.collide.Rows, 'from_catalog',
                                return_value=api.collide.Rows({'NewRow_33': ('Beam01', 'Wall', ())})))

    def start(self, patcher):
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_scan_files_are_per_call_and_reports_stay_in_collide_dir(self):
        with patch.object(agent, 'map_info', return_value={'display_name': 'Demo Map'}):
            result = api.call('check_collisions', x0=0, y0=0, x1=10, y1=10, top=0)
        self.assertEqual(self.run_mock.call_count, 4)  # an explicit region is scanned as a 2 x 2 grid
        self.assertEqual(self.entered, ['session', 'left'])
        self.assertEqual(len(set(self.scan_dirs)), 1)
        scan_dir = self.scan_dirs[0]
        self.assertTrue(scan_dir.name.startswith('uncrashed-collide-') and scan_dir.parent == Path('/tmp'))
        self.assertFalse(scan_dir.exists())
        self.assertEqual(sorted(p.name for p in self.reports.iterdir()), ['report.html', 'report.json'])
        self.assertIn('Demo Map collision check', (self.reports / 'report.html').read_text())
        json.loads((self.reports / 'report.json').read_text())
        self.assertEqual(result['report_html'], str(self.reports / 'report.html'))
        self.assertEqual(result['region_m'], [0, 0, 10, 10])
        self.assertIn('check_foundations / check_environment', result['scope'])

    def test_default_region_and_grid_come_from_the_placed_bounds(self):
        with patch.object(api.environment, 'get_world_bounds', return_value=self.BOUNDS) as bounds, \
             patch.object(agent, 'map_info', side_effect=agent.AgentError('no editor')):
            result = api.check_collisions()
        bounds.assert_called_once_with('placed')
        self.assertEqual(result['region_m'], (95, -85, 115, -65))
        self.assertEqual(self.run_mock.call_count, 9)  # ceil(sqrt(1700 / 400)) = 3 -> 3 x 3
        self.assertIn('Map collision check', (self.reports / 'report.html').read_text())  # title lookup failed: default

    def test_no_placed_objects_returns_early_without_scanning(self):
        with patch.object(api.environment, 'get_world_bounds', return_value={'available': False}):
            self.assertEqual(api.check_collisions(), {'objects_checked': 0, 'issues': {}, 'top': [], 'scope': 'no placed objects'})
        self.run_mock.assert_not_called()
        self.assertFalse(self.reports.exists())

    def test_truncated_bounds_and_bad_regions_raise(self):
        with patch.object(api.environment, 'get_world_bounds', return_value=dict(self.BOUNDS, truncated=True)):
            with self.assertRaisesRegex(RuntimeError, 'pass an explicit scan region'):
                api.check_collisions()
        self.entered.clear()
        with self.assertRaises(ValueError):
            api.check_collisions(x0=1)
        self.assertEqual(self.entered, [])  # a usage error does not wait for the agent lock
        self.run_mock.assert_not_called()

    def test_top_is_clamped(self):
        with patch.object(agent, 'map_info', return_value={}), patch.object(api.collide, 'brief', return_value={}) as brief:
            for top, clamped in ((0, 1), (15, 15), (999, 60)):
                api.check_collisions(0, 0, 10, 10, top=top, include_minor=True)
                self.assertEqual(brief.call_args.kwargs, {'top': clamped, 'include_minor': True})


class SoftFailureTests(unittest.TestCase):
    def check(self, name, payload):
        reason = api.OPERATIONS[name].check(payload)
        if reason is not None:
            self.assertIsInstance(reason, str)
            self.assertTrue(reason and '\n' not in reason)
        return reason

    def test_only_these_operations_have_a_predicate(self):
        self.assertEqual({n for n, op in api.OPERATIONS.items() if op.check},
                         {'load_live', 'save_live', 'agent_unload', 'set_collision', 'capture_sheet', 'reference_load'})

    def test_capture_sheet(self):
        self.assertIsNone(self.check('capture_sheet', {'poses': 3, 'failed_count': 0}))
        self.assertEqual(self.check('capture_sheet', {'poses': 3, 'failed_count': 1}),
                         '1 of 3 poses failed; see shots[].error')

    def test_load_live(self):
        ok = {'failed': [], 'failed_count': 0, 'camera': {'x': 1.0}, 'clearing_zones': {'native': ['zones 0']}}
        self.assertIsNone(self.check('load_live', ok))
        self.assertIsNone(self.check('load_live', dict(ok, camera=None)))
        self.assertIn('3 map lines failed', self.check('load_live', dict(ok, failed_count=3)))
        self.assertIn('camera: no pawn', self.check('load_live', dict(ok, camera={'error': 'no pawn'})))
        zones = self.check('load_live', dict(ok, failed_count=1, clearing_zones={'error': 'zones\nfailed', 'note': 'geometry loaded'}))
        self.assertIn('1 map lines failed', zones)
        self.assertIn('clearing_zones: zones failed', zones)

    def test_save_live(self):
        self.assertIsNone(self.check('save_live', {'written': True, 'path': '/maps/A/A.json'}))
        self.assertIn('/maps/A/A.json', self.check('save_live', {'written': False, 'path': '/maps/A/A.json'}))

    def test_agent_unload(self):
        self.assertIsNone(self.check('agent_unload', {'unloaded': True, 'loaded': False}))
        self.assertIn('not loaded', self.check('agent_unload', {'unloaded': False, 'loaded': False}))
        self.assertIn('still loaded', self.check('agent_unload', {'unloaded': False, 'loaded': True}))
        self.assertIn('restore conflict', self.check('agent_unload', {'unloaded': False, 'loaded': True, 'error': 'restore conflict'}))

    def test_set_collision(self):
        self.assertIsNone(self.check('set_collision', {'preview': True, 'matched': 4}))
        self.assertIsNone(self.check('set_collision', {'changed': True, 'verified': True, 'mismatched': [], 'missing_after': 0}))
        self.assertIn('1 objects in another mode, 2 missing',
                      self.check('set_collision', {'changed': True, 'verified': False, 'mismatched': ['1A'], 'missing_after': 2}))
        actors = [f'{i:X}' for i in range(120)]
        cut = {'changed': True, 'verified': False, 'mismatched': actors[:50], 'missing_after': 0}  # as scenery cuts it
        self.assertEqual(self.check('set_collision', cut),
                         'collision change applied but not verified: at least 50 objects in another mode, 0 missing')
        self.assertIn(': 120 objects in another mode', self.check('set_collision', dict(cut, mismatched_count=120)))
        self.assertIn(': 49 objects in another mode', self.check('set_collision', dict(cut, mismatched=actors[:49])))
        self.assertIn('verification query failed',
                      self.check('set_collision', {'changed': True, 'verified': None,
                                                   'verify_error': 'applied, but the verification query failed: timeout'}))


if __name__ == '__main__':
    unittest.main()
