import asyncio
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uncrashed_mcp import agent, api

try:
    from uncrashed_mcp import server
except ModuleNotFoundError as exc:  # the MCP adapter is optional
    if exc.name != 'mcp':
        raise
    server = None

BASELINE = json.loads((Path(__file__).parent / 'data' / 'mcp_tools_baseline.json').read_text())

_guards = []


def setUpModule():
    for name in ('ensure_agent', '_execute'):
        _guards.append(patch.object(agent, name, side_effect=AssertionError(f'agent.{name} reached from a test')))
    _guards.append(patch.object(agent, 'session', contextlib.nullcontext))
    for guard in _guards:
        guard.start()


def tearDownModule():
    while _guards:
        _guards.pop().stop()


def call_tool(name, arguments):
    return asyncio.run(server.mcp.call_tool(name, arguments))


@unittest.skipIf(server is None, 'the mcp package is not installed')
class ToolListTests(unittest.TestCase):
    def test_tool_list_equals_the_baseline_except_declared_differences(self):
        listed = [tool.model_dump(mode='json', exclude_none=True) for tool in asyncio.run(server.mcp.list_tools())]
        expected = json.loads(json.dumps(BASELINE['tools']))
        by_name = {tool['name']: tool for tool in expected}
        # declared: capture_view gains `out`; agent_batch documents the new failure format (with the L<file line>
        # a file-based caller gets, and the cap of 20 listed failures)
        by_name['capture_view']['input_schema']['properties']['out'] = {'default': '', 'title': 'Out', 'type': 'string'}
        # declared: capture_view gains `reference` (keep a loaded textured reference in the image) and says so
        by_name['capture_view']['input_schema']['properties']['reference'] = {'default': False, 'title': 'Reference',
                                                                              'type': 'boolean'}
        # declared: and `reference_only` (render nothing but the reference: the before half of a pair)
        by_name['capture_view']['input_schema']['properties']['reference_only'] = {
            'default': False, 'title': 'Reference Only', 'type': 'boolean'}
        view = next(tool for tool in listed if tool['name'] == 'capture_view')
        self.assertEqual(view['description'], api.capture_view.__doc__)
        self.assertTrue(view['description'].startswith(by_name['capture_view']['description'].rstrip()))
        by_name['capture_view']['description'] = view['description']
        batch = next(tool for tool in listed if tool['name'] == 'agent_batch')
        self.assertTrue(batch['description'].startswith(by_name['agent_batch']['description'].rstrip()))
        self.assertNotEqual(batch['description'], by_name['agent_batch']['description'])
        self.assertIn('"#<index> [L<file line>] <error> << <command>"', batch['description'])
        by_name['agent_batch']['description'] = batch['description']
        # declared: operations added after the port follow the baseline tools, described by their own docstrings
        added = listed[len(expected):]
        self.assertEqual([tool['name'] for tool in added], ['capture_sheet', 'scene_snapshot', 'scene_diff', 'check_opening',
                                                            'reference_load', 'reference_status', 'reference_clear',
                                                            'reference_visibility', 'scan_survey', 'scan_status',
                                                            'scan_place', 'scan_height', 'scan_profile', 'scan_at',
                                                            'scan_objects', 'scan_ground', 'scan_building', 'scan_facade',
                                                            'scan_free', 'scan_view', 'scan_sheet', 'scan_match',
                                                            'scan_plan', 'scan_compare', 'capture_topdown',
                                                            'asset_profile', 'scan_sources', 'scan_grid'])
        load = added[4]
        self.assertEqual(load['input_schema']['required'], ['obj'])
        self.assertEqual(load['input_schema']['properties']['datum_m'].get('default'), 'auto')
        opening = added[3]
        self.assertEqual(opening['input_schema']['required'], ['path'])
        self.assertEqual({k: v.get('default') for k, v in opening['input_schema']['properties'].items()},
                         {'path': None, 'profile': '5inch', 'margin_m': None, 'channel': 'Pawn', 'spacing_m': .25,
                          'dims_m': None, 'near_m': 5.0, 'top': 10})
        for tool in added[1:]:
            self.assertEqual(tool['description'], getattr(api, tool['name']).__doc__)
        self.assertNotIn('required', added[1]['input_schema'])
        self.assertEqual(added[2]['input_schema']['required'], ['snapshot'])
        self.assertEqual({k: v.get('default') for k, v in added[2]['input_schema']['properties'].items()},
                         {'snapshot': None, 'group': None, 'box': None, 'position_m': .01, 'rotation_deg': .1,
                          'scale': .001, 'gate_position_m': .01, 'examples': 10})
        sheet = added[0]
        self.assertEqual(sheet['description'], api.capture_sheet.__doc__)
        self.assertEqual(sheet['input_schema']['required'], ['poses'])
        self.assertEqual({k: v.get('default') for k, v in sheet['input_schema']['properties'].items()},
                         {'poses': None, 'width': 480, 'height': 270, 'columns': 0, 'out': '', 'reference': False,
                          'reference_only': False})
        listed = listed[:len(expected)]
        self.assertEqual([tool['name'] for tool in listed], [tool['name'] for tool in expected])
        for got, want in zip(listed, expected):
            self.assertEqual(got, want, got['name'])
        self.assertNotIn('output_schema', by_name['capture_view'])

    def test_instructions_and_module_attributes(self):
        self.assertEqual(server.mcp.instructions, BASELINE['instructions'])
        self.assertEqual(server.INSTRUCTIONS, BASELINE['instructions'])
        self.assertIs(server.VERBS, api.VERBS)
        self.assertTrue(callable(server.main))
        for name, op in api.OPERATIONS.items():
            tool = getattr(server, name)
            self.assertEqual((tool.__name__, tool.__doc__), (name, op.fn.__doc__))
            self.assertIsNot(tool, op.fn)

    def test_stale_patch_targets_are_gone(self):
        # tests that still patch the old server internals must fail loudly instead of patching nothing
        for name in ('COLLIDE_DIR', '_catalog', 'agent', 'environment', 'collide', 'groups', 'managed', 'scenery',
                     'performance', 'consolidate', 'clearance', 'assemblies', 'live_capture_view'):
            self.assertFalse(hasattr(server, name), name)


@unittest.skipIf(server is None, 'the mcp package is not installed')
class ConversionTests(unittest.TestCase):
    def test_dict_results_are_compact_json(self):
        with patch.object(agent, 'batch', return_value=({0: ['terrain x=100.00 y=200.00 z=-951.56']}, 'ok')):
            self.assertEqual(server.sample_terrain([[1, 2]]), '{"samples":[{"x":1,"y":2,"z":-9.5156}]}')
            result = call_tool('sample_terrain', {'points': [[1, 2]]})
        self.assertFalse(result.is_error)
        self.assertEqual(json.loads(result.content[0].text), {'samples': [{'x': 1, 'y': 2, 'z': -9.5156}]})
        self.assertNotIn(', ', result.content[0].text)

    def test_tuples_are_serialised_like_before(self):
        with patch.object(api.environment, 'audit', lambda *a, **k: {'region': k['region']}), \
             patch.object(api, '_catalog', return_value=None):
            self.assertEqual(server.check_foundations(x0=0, y0=1, x1=2, y1=3), '{"region":[0,1,2,3]}')

    def test_json_tools_report_errors_as_json(self):
        with patch.object(api, 'maps_dir', return_value=Path('/nowhere')):
            self.assertEqual(json.loads(server.inspect_map('no such')), {'error': "no local map 'no_such' in /nowhere"})
        self.assertEqual(set(json.loads(server.check_collisions(x0=1))), {'error'})
        self.assertEqual(set(json.loads(server.sample_terrain([]))), {'error'})
        with patch.object(agent, 'log_tail', side_effect=OSError('denied')):
            self.assertEqual(server.agent_log(), '{"error":"denied"}')
        result = call_tool('inspect_map', {'name': 'no such map 7f3a'})
        self.assertFalse(result.is_error)
        self.assertIn('error', json.loads(result.content[0].text))

    def test_formerly_unguarded_tools_now_report_errors_as_json(self):
        with patch.object(api, '_catalog', side_effect=RuntimeError('catalogue unreadable')):
            for name, args in (('status', ()), ('list_gates', ()), ('search_assets', ()), ('get_assets', (['NewRow_1'],))):
                self.assertEqual(getattr(server, name)(*args), '{"error":"catalogue unreadable"}', name)
        with patch.object(api, 'list_user_maps', side_effect=RuntimeError('maps unreadable')):
            self.assertEqual(server.list_maps(), '{"error":"maps unreadable"}')

    def test_text_tools(self):
        reply = {'seq': 1, 'ok': True, 'head': 'ok mapinfo', 'lines': ['folder Demo']}
        with patch.object(agent, 'execute', return_value=reply):
            self.assertEqual(server.agent_exec('mapinfo'), 'ok ok mapinfo\nfolder Demo')
            self.assertEqual(call_tool('agent_exec', {'command': 'mapinfo'}).content[0].text, 'ok ok mapinfo\nfolder Demo')
        with patch.object(agent, 'execute', return_value=dict(reply, ok=False, lines=['error no row X'])):
            self.assertEqual(server.agent_exec('spawn X 0 0 0'), 'failed ok mapinfo\nerror no row X')
        with patch.object(agent, 'execute', side_effect=agent.AgentError('Uncrashed is not running')):
            self.assertEqual(server.agent_exec('mapinfo'), 'error Uncrashed is not running')
        with patch.object(agent, 'batch', return_value=({1: ['error no row X']}, 'batch lines=2 ok=1 failed=1')):
            self.assertEqual(server.agent_batch(['spawn A 0 0 0', 'spawn X 0 0 0']).split('\n'), [
                'batch lines=2 ok=1 failed=1 sent=2', 'failures:', '#1 error no row X << spawn X 0 0 0',
                'successful lines are applied; retry only the failed lines'])
        self.assertEqual(server.agent_batch([]), 'error no commands')
        with patch.object(agent, 'batch', side_effect=agent.AgentNotSent('Uncrashed is not running')):
            self.assertEqual(server.agent_batch(['mapinfo']), 'error Uncrashed is not running')
        lost = agent.AgentError("the agent is not loaded (unloaded, or the game closed); 'batch x' was sent before that "
                                "and may have run")
        with patch.object(agent, 'batch', side_effect=lost):  # sent: an MCP client must not re-send the commands either
            self.assertEqual(server.agent_batch(['mapinfo']), f'error {lost}; {api.BATCH_UNKNOWN}')
        with tempfile.TemporaryDirectory() as tmp, patch.object(agent, 'LOG', Path(tmp) / 'agent.log'):
            self.assertEqual(server.agent_log(), f"(agent log is missing: {Path(tmp) / 'agent.log'})")
        with patch.object(agent, 'log_tail', return_value='one\ntwo'):
            self.assertEqual(server.agent_log(2), 'one\ntwo')

    def test_capture_view_returns_metadata_and_image(self):
        from uncrashed_mcp import live
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / 'view-1.png'
            png.write_bytes(b'\x89PNG\r\n\x1a\n')
            png.with_suffix('.json').write_text(json.dumps({'image': str(png), 'backend': 'engine', 'width': 800}))
            with patch.object(live, 'capture_view', return_value=png):
                direct = server.capture_view()
                self.assertEqual(direct[0], json.dumps({'image': str(png), 'backend': 'engine', 'width': 800}, separators=(',', ':')))
                self.assertEqual(Path(direct[1].path), png)
                result = call_tool('capture_view', {})
                self.assertEqual([part.type for part in result.content], ['text', 'image'])
                self.assertEqual(result.content[1].mime_type, 'image/png')
                self.assertIsNone(result.structured_content)
                moved = json.loads(server.capture_view(out=str(Path(tmp) / 'out') + '/')[0])
                self.assertEqual(moved['image'], str(Path(tmp) / 'out' / 'view-1.png'))
        with patch.object(live, 'capture_view', side_effect=live.LiveError('pass x, y and z together')):
            self.assertEqual(server.capture_view(x=1), ['{"error":"pass x, y and z together"}'])
            result = call_tool('capture_view', {'x': 1})
        self.assertEqual([part.type for part in result.content], ['text'])
        self.assertEqual(json.loads(result.content[0].text), {'error': 'pass x, y and z together'})


if __name__ == '__main__':
    unittest.main()
