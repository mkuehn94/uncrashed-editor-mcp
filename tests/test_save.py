import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from uncrashed_mcp import agent


class SaveTests(unittest.TestCase):
    def test_waits_for_complete_deferred_write(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'map.json'
            path.write_text('{')
            with patch.object(agent.time, 'sleep', side_effect=lambda _: path.write_text('{}')) as sleep:
                self.assertTrue(agent._wait_for_save(path, None))
            sleep.assert_called_once()

    def test_stale_existing_file_is_not_success(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'map.json'
            path.write_text('{}')
            self.assertFalse(agent._wait_for_save(path, path.stat().st_mtime_ns, timeout=.001))

    def test_first_save_initializes_editor_directory(self):
        with TemporaryDirectory(prefix='map test ') as directory:
            root = Path(directory)
            def run(command, *args):
                if command.startswith('save '):
                    path = root / 'NewMap' / 'NewMap.json'
                    path.parent.mkdir()
                    path.write_text(json.dumps({'assets': []}))
                return []
            with patch.object(agent, 'maps_dir', return_value=root), \
                 patch.object(agent, 'map_info', return_value={'folder': '', 'objects': 1, 'gates': 0}), \
                 patch.object(agent, 'run', side_effect=run) as calls:
                result = agent.save('NewMap', 'New Map')
            self.assertTrue(result['written'])
            self.assertEqual(calls.call_args_list[-1].args[0],
                             f'setmap {agent._windows_path(root / "NewMap")}')

    def test_save_as_copies_portable_metadata(self):
        from uncrashed_mcp import managed
        with TemporaryDirectory() as directory:
            root=Path(directory); source=root/'Old';source.mkdir()
            state={'version':1,'base':'LongRange1','zones':{},'assemblies':{}}
            managed.write_state(source,state)
            def run(command,*args):
                if command.startswith('save '):
                    target=root/'New';target.mkdir();(target/'New.json').write_text('{}')
                return []
            info={'folder':'Old','directory':agent._windows_path(source),'base':'LongRange1','objects':1,'gates':0}
            with patch.object(agent,'maps_dir',return_value=root),patch.object(agent,'map_info',return_value=info),patch.object(agent,'run',side_effect=run):
                result=agent.save('New')
            self.assertTrue(result['written'])
            self.assertEqual(json.loads((root/'New'/managed.STATE_NAME).read_text()),state)
            self.assertTrue((root/'New'/managed.ZONES_NAME).exists())

    def test_save_as_metadata_conflict_refuses_before_game_save(self):
        from uncrashed_mcp import managed
        with TemporaryDirectory() as directory:
            root=Path(directory);source=root/'Old';target=root/'New'
            state={'version':1,'base':'LongRange1','zones':{},'assemblies':{}}
            managed.write_state(source,state);managed.write_state(target,state)
            info={'folder':'Old','directory':agent._windows_path(source),'base':'LongRange1'}
            with patch.object(agent,'maps_dir',return_value=root),patch.object(agent,'map_info',return_value=info),patch.object(agent,'run') as run:
                with self.assertRaisesRegex(agent.AgentError,'explicit merge'):
                    agent.save('New')
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
