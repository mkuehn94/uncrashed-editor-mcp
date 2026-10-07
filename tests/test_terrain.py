import json
import unittest
from unittest.mock import patch

from uncrashed_mcp.server import sample_terrain


class TerrainTests(unittest.TestCase):
    @patch("uncrashed_mcp.server.agent.batch")
    def test_units_and_missing_landscape(self, batch):
        batch.return_value = ({0: ["terrain x=100.00 y=200.00 z=-951.56"],
                               1: ["terrain miss x=300.00 y=400.00"]}, "ok")
        result = json.loads(sample_terrain([[1, 2], [3, 4]]))
        self.assertEqual(result["samples"], [{"x": 1, "y": 2, "z": -9.5156},
                                             {"x": 3, "y": 4, "z": None}])
        self.assertEqual(batch.call_args.args[0][0], "terrain 100.000 200.000 100000.000 -200000.000")

    @patch("uncrashed_mcp.server.agent.batch")
    def test_validation(self, batch):
        for points in ([], [[1]], [[1, float("nan")]], [[1, 2]] * 257):
            self.assertIn("error", json.loads(sample_terrain(points)))
        self.assertIn("error", json.loads(sample_terrain([[1, 2]], z_from=0, z_to=1)))
        batch.assert_not_called()

    @patch("uncrashed_mcp.server.agent.batch", return_value=({}, "ok"))
    def test_missing_reply_is_not_silent_miss(self, batch):
        self.assertIn("error", json.loads(sample_terrain([[1, 2]])))


if __name__ == "__main__":
    unittest.main()
