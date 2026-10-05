"""Pure parts of the live agent client: rotations and reply parsing. No game needed."""

from __future__ import annotations

import math
import random
import unittest

from uncrashed_mcp import agent
from uncrashed_mcp.geom import quat_yaw, rotator_from_quat


def rotator_quat(pitch: float, yaw: float, roll: float) -> dict:
    """Unreal's FRotator::Quaternion, the same formula agent.c uses."""
    p, y, r = (math.radians(v) / 2 for v in (pitch, yaw, roll))
    sp, cp, sy, cy, sr, cr = math.sin(p), math.cos(p), math.sin(y), math.cos(y), math.sin(r), math.cos(r)
    return {
        "X": cr * sp * sy - sr * cp * cy,
        "Y": -cr * sp * cy - sr * cp * sy,
        "Z": cr * cp * sy - sr * sp * cy,
        "W": cr * cp * cy + sr * sp * sy,
    }


class RotatorTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        rng = random.Random(7)
        for _ in range(500):
            want = (rng.uniform(-89, 89), rng.uniform(-179, 179), rng.uniform(-179, 179))
            got = rotator_from_quat(rotator_quat(*want))
            for a, b in zip(got, want):
                self.assertAlmostEqual(a, b, places=6)

    def test_map_file_yaw(self) -> None:
        pitch, yaw, roll = rotator_from_quat(quat_yaw(math.radians(90)))
        self.assertAlmostEqual(pitch, 0, places=4)
        self.assertAlmostEqual(yaw, 90, places=3)
        self.assertAlmostEqual(roll, 0, places=4)

    def test_straight_down(self) -> None:
        pitch, _yaw, _roll = rotator_from_quat(rotator_quat(-90, 0, 0))
        self.assertAlmostEqual(pitch, -90, places=4)


class ParseTest(unittest.TestCase):
    def test_actor_line(self) -> None:
        row = agent.parse_actor(
            "actor 17e63100 row=NewRow_188 class=Actor loc=-5500.00,-27000.00,10.00 rot=0.000,45.000,0.000 "
            "scale=2.0000,2.0000,2.0000 d=12.50 tags=MapEditor,NewRow_188,ME_StaticMesh"
        )
        self.assertEqual(row["actor"], "17e63100")
        self.assertEqual(row["asset"], "NewRow_188")
        self.assertEqual((row["x"], row["y"], row["z"]), (-55.0, -270.0, 0.1))
        self.assertEqual(row["yaw"], 45.0)
        self.assertEqual(row["scale"], [2.0, 2.0, 2.0])
        self.assertEqual(row["distance_m"], 0.125)
        self.assertIn("MapEditor", row["tags"])

    def test_gate_line_has_no_row(self) -> None:
        row = agent.parse_actor("gate 4249e1c0 row=- class=BP_Gate_C loc=0.00,0.00,300.00 rot=0.000,90.000,0.000 scale=1.0000,1.0000,1.0000")
        self.assertIsNone(row["asset"])
        self.assertEqual(row["class"], "BP_Gate_C")
        self.assertEqual(row["z"], 3.0)

    def test_spawn_command_is_centimetres(self) -> None:
        cmd = agent._spawn_command("NewRow_1", 1.5, -2, 0.25, 0, 90, 0, (1, 1, 2), "Block7")
        self.assertEqual(cmd, "spawn NewRow_1 150.00 -200.00 25.00 0.0000 90.0000 0.0000 1.00000 1.00000 2.00000 group=Block7")
        self.assertNotIn("group=", agent._spawn_command("NewRow_1", 0, 0, 0, 0, 0, 0, (1, 1, 1), "None"))

    def test_spawn_command_rejects_bad_scale(self) -> None:
        with self.assertRaises(agent.AgentError):
            agent._spawn_command("NewRow_1", 0, 0, 0, 0, 0, 0, (1, 0, 1))

    def test_errors_from_batch_lines(self) -> None:
        self.assertEqual(agent._errors(["#3 error no row X in DT", "spawned 1 row=A class=B"]), ["no row X in DT"])


if __name__ == "__main__":
    unittest.main()
