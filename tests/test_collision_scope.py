import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from uncrashed_mcp import server
from uncrashed_mcp.collide import Rows


class CollisionScopeTests(unittest.TestCase):
    def test_default_scan_uses_live_bounds_outside_old_three_km_limit(self):
        commands = []
        def run(command, **kwargs):
            commands.append(command)
            path = Path(command.split()[1][2:].replace('\\', '/'))
            path.write_text('O a NewRow_33 1000000 -800000 -1000 1000100 -799900 -900\nR a 0\n')
            return []
        with TemporaryDirectory() as tmp, \
             patch.object(server, 'COLLIDE_DIR', Path(tmp)), \
             patch.object(server.environment, 'get_world_bounds', return_value={
                 'available': True, 'truncated': False, 'actor_count': 1,
                 'aabb': {'min': [10000,-8000,-10], 'max': [10010,-7990,10]}}), \
             patch.object(server.agent, 'run', side_effect=run), \
             patch.object(server.agent, 'map_info', return_value={}), \
             patch.object(server, '_catalog', return_value=None), \
             patch.object(server.collide.Rows, 'from_catalog', return_value=Rows({'NewRow_33': ('Beam01','Wall',())})):
            result = json.loads(server.check_collisions())
        self.assertNotIn('error', result)
        self.assertEqual(result['region_m'], [9995,-8005,10015,-7985])
        self.assertEqual(commands[0].split()[2:6], ['999500','-800500','1001500','-798500'])

    def test_missing_scan_file_is_error_not_clean_report(self):
        with TemporaryDirectory() as tmp, patch.object(server, 'COLLIDE_DIR', Path(tmp)), \
             patch.object(server.agent, 'run', return_value=[]):
            result = json.loads(server.check_collisions(0,0,10,10))
        self.assertIn('incomplete', result['error'])

    def test_partial_region_is_rejected_without_game_commands(self):
        with TemporaryDirectory() as tmp, patch.object(server, 'COLLIDE_DIR', Path(tmp)), \
             patch.object(server.agent, 'run') as run:
            self.assertIn('error', json.loads(server.check_collisions(x0=1)))
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
