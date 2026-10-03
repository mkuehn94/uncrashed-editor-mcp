"""Geometry and city layout. No game process required."""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("UNCRASHED_MAPS_DIR", tempfile.mkdtemp(prefix="uncrashed-maps-"))

from uncrashed_mcp.build import _city_layout, build_city, build_district, build_structure
from uncrashed_mcp.catalog import load_catalog
from uncrashed_mcp.geom import world_aabb
from uncrashed_mcp.mapfile import load_map, new_map


def _center(aabb):
    return ((aabb[0] + aabb[3]) / 2, (aabb[1] + aabb[4]) / 2, (aabb[2] + aabb[5]) / 2)


class PlacementTests(unittest.TestCase):
    def test_offset_house_sits_on_the_requested_point(self):
        catalog = load_catalog()
        house = catalog.get("NewRow_788")
        doc = new_map("House", root=Path(os.environ["UNCRASHED_MAPS_DIR"]))
        doc.add(house, 10, 20, 0.5, yaw_deg=35, anchor="bottom_center")
        placed = doc.iter_placed(catalog)[0]
        aabb = placed["aabb_m"]
        cx, cy, _ = _center(aabb)
        self.assertAlmostEqual(cx, 10, delta=0.05)
        self.assertAlmostEqual(cy, 20, delta=0.05)
        self.assertAlmostEqual(aabb[2], 0.5, delta=0.05)

    def test_asphalt_top_is_the_requested_height(self):
        catalog = load_catalog()
        row = catalog.get("NewRow_182")
        doc = new_map("Road", root=Path(os.environ["UNCRASHED_MAPS_DIR"]))
        doc.add(row, 0, 0, 0.10, scale=(3, 2, 1), anchor="top_center")
        aabb = doc.iter_placed(catalog)[0]["aabb_m"]
        self.assertAlmostEqual(aabb[5], 0.10, delta=0.02)
        self.assertAlmostEqual(aabb[3] - aabb[0], row.size_m[0] * 3, delta=0.15)

    def test_roundtrip_schema(self):
        catalog = load_catalog()
        root = Path(os.environ["UNCRASHED_MAPS_DIR"])
        doc = new_map("Round", root=root)
        build_structure(doc, catalog, 0, 0, 12, 10, 8, ground_z=0)
        doc.save(catalog)
        raw = json.loads((root / "Round" / "Round.json").read_text())
        self.assertEqual(
            set(raw),
            {"S_AssetsTransform", "BaseMapRow", "WeatherIndex", "Race", "CamLoc", "CustomWeather"},
        )
        group = raw["S_AssetsTransform"][0]
        self.assertIn("UniqueGroupTag", group)
        self.assertEqual(len(group["UniqueGroupTag"]), len(group["Transform"]))
        transform = group["Transform"][0]
        self.assertEqual(set(transform), {"Rotation", "Translation", "Scale3D"})
        loaded = load_map("Round", root=root)
        self.assertEqual(loaded.instance_count(), doc.instance_count())
        info = json.loads((root / "Round" / "Round.info").read_text())
        self.assertEqual(info["BaseMapRow"], "BaseBoard")
        self.assertIn("Name", info)


class CityTests(unittest.TestCase):
    def _build(self, style, folder):
        catalog = load_catalog()
        root = Path(os.environ["UNCRASHED_MAPS_DIR"])
        doc = new_map(folder, root=root)
        info = build_city(doc, catalog, style=style, blocks_x=3, blocks_y=3, block_m=36, road_m=12, seed=3)
        doc.save(catalog)
        return catalog, doc, info

    def test_downtown_keeps_roads_buildings_and_spawn_apart(self):
        catalog, doc, info = self._build("downtown", "Down")
        report = doc.validate(catalog)
        self.assertTrue(report["ok"], report["problems"])
        self.assertGreater(info["instances"], 20)
        self.assertLess(info["instances"], 8000)
        placed = doc.iter_placed(catalog)
        ids = {item["id"] for item in placed}
        self.assertNotIn("NewRow_109", ids)
        self.assertNotIn("NewRow_318", ids)
        self.assertTrue(ids & {"NewRow_9", "NewRow_788", "NewRow_794", "NewRow_796", "NewRow_228"})
        from uncrashed_mcp.build import _street_zones

        zone = _street_zones(3, 3, 36, 12)
        roads = zone["roads"]
        buildings = []
        cars = []
        for item in placed:
            aabb = item["aabb_m"]
            if item["id"] in {"NewRow_775", "NewRow_93", "NewRow_103"}:
                cars.append(aabb)
            if item["id"] in {"NewRow_9", "NewRow_788", "NewRow_789", "NewRow_790", "NewRow_791", "NewRow_793", "NewRow_794", "NewRow_795", "NewRow_796", "NewRow_228", "NewRow_95", "NewRow_134"}:
                buildings.append(aabb)
        self.assertGreaterEqual(len(buildings), 4)

        def overlap(a, rect, pad=0.0):
            rx, ry, rw, rh = rect
            return a[0] < rx + rw - pad and rx < a[3] - pad and a[1] < ry + rh - pad and ry < a[4] - pad

        for aabb in buildings:
            self.assertFalse(any(overlap(aabb, road, 0.2) for road in roads), aabb)
        self.assertGreaterEqual(len(cars), 2)
        for aabb in cars:
            self.assertTrue(
                any(
                    aabb[0] >= road[0] - 0.05
                    and aabb[1] >= road[1] - 0.05
                    and aabb[3] <= road[0] + road[2] + 0.05
                    and aabb[4] <= road[1] + road[3] + 0.05
                    for road in roads
                ),
                aabb,
            )
            for other in buildings:
                self.assertFalse(
                    aabb[0] < other[3] and other[0] < aabb[3] and aabb[1] < other[4] and other[1] < aabb[4]
                )
        sx, sy, _ = report["spawn_m"]
        for item in placed:
            aabb = item["aabb_m"]
            if aabb[5] < 1:
                continue
            nearest_x = min(max(sx, aabb[0]), aabb[3])
            nearest_y = min(max(sy, aabb[1]), aabb[4])
            if math.hypot(nearest_x - sx, nearest_y - sy) < 1.2 and aabb[5] > 1.5 and aabb[2] < 3:
                self.fail(f"{item['name']} intersects spawn")

    def test_suburb_and_industrial_validate(self):
        for style, folder in (("suburb", "Sub"), ("industrial", "Ind")):
            catalog, doc, _ = self._build(style, folder)
            report = doc.validate(catalog)
            self.assertTrue(report["ok"], (style, report["problems"]))

    def test_road_rectangles_meet_without_overlapping(self):
        horizontal, vertical, blocks = _city_layout(2, 2, 40, 10)
        def overlap(a, b):
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            return ax < bx + bw - 0.01 and bx < ax + aw - 0.01 and ay < by + bh - 0.01 and by < ay + ah - 0.01
        for road in vertical:
            for other in horizontal:
                self.assertFalse(overlap(road, other))
        for block in blocks:
            for road in horizontal + vertical:
                self.assertFalse(overlap(block, road))


class DistrictTests(unittest.TestCase):
    def test_highline_is_zoned_and_tall(self):
        catalog = load_catalog()
        root = Path(os.environ["UNCRASHED_MAPS_DIR"])
        doc = new_map("Line", root=root)
        info = build_district(doc, catalog, seed=4)
        report = doc.validate(catalog)
        self.assertTrue(report["ok"], report["problems"])
        self.assertGreater(info["instances"], 1600)
        self.assertGreaterEqual(info["checkpoints"], 6)
        self.assertGreater(info["size_m"][0], 140)
        self.assertLess(info["size_m"][0], 220)
        placed = doc.iter_placed(catalog)
        counts: dict[str, int] = {}
        top = 0.0
        for item in placed:
            counts[item["id"]] = counts.get(item["id"], 0) + 1
            top = max(top, item["aabb_m"][5])
        self.assertGreater(top, 40)
        self.assertLess(top, 120)
        self.assertGreaterEqual(counts.get("NewRow_11", 0), 1)
        self.assertGreaterEqual(counts.get("NewRow_122", 0), 40)
        self.assertGreaterEqual(counts.get("NewRow_1", 0), 8)
        self.assertGreaterEqual(counts.get("NewRow_322", 0), 1)
        self.assertGreaterEqual(counts.get("NewRow_157", 0), 4)
        self.assertEqual(counts.get("NewRow_121", 0), 0)
        self.assertEqual(counts.get("NewRow_335", 0), 0)
        sx, sy, sz = info["shaft_m"]
        self.assertGreater(sz, 1)
        for item in placed:
            if item["id"] != "NewRow_303":
                continue
            box = item["aabb_m"]
            self.assertFalse(
                box[0] < sx < box[3] and box[1] < sy < box[4] and box[2] < sz < box[5],
                "floor slab closes the shaft",
            )
        from uncrashed_mcp.build import DISTRICT_ROADS, STREET_BUILDINGS

        roads_hit = 0
        for item in placed:
            if item["id"] not in STREET_BUILDINGS:
                continue
            box = item["aabb_m"]
            for rx, ry, rw, rh in DISTRICT_ROADS:
                if box[0] < rx + rw - 0.3 and rx < box[3] - 0.3 and box[1] < ry + rh - 0.3 and ry < box[4] - 0.3:
                    roads_hit += 1
        self.assertEqual(roads_hit, 0)


class BoundsDirect(unittest.TestCase):
    def test_rotated_bottom(self):
        catalog = load_catalog()
        wall = catalog.get("NewRow_318")
        from uncrashed_mcp.geom import pivot_for_anchor

        yaw = math.radians(-90)
        scale = (1, 4, 2)
        pivot = pivot_for_anchor(wall.origin_cm, wall.extent_cm, 3, 4, 1, yaw, scale, "bottom_center")
        aabb = world_aabb(wall.origin_cm, wall.extent_cm, pivot, yaw, scale)
        self.assertAlmostEqual((aabb[0] + aabb[3]) / 2, 3, delta=0.05)
        self.assertAlmostEqual((aabb[1] + aabb[4]) / 2, 4, delta=0.05)
        self.assertAlmostEqual(aabb[2], 1, delta=0.05)
        # Long axis of the wall is local Y. Yaw -90 maps local +Y onto +X, so the long side is world X.
        self.assertGreater(aabb[3] - aabb[0], aabb[4] - aabb[1])


class FitKnowledge(unittest.TestCase):
    def test_every_asset_has_a_scene_role(self):
        from uncrashed_mcp.fit import conflicts, describe, knowledge

        catalog = load_catalog()
        info = knowledge(catalog)
        self.assertEqual(sum(info["counts"].values()), len(catalog.assets))
        wall = describe(catalog.get("NewRow_318"))
        self.assertEqual(wall["role"], "wall")
        self.assertIn("lane", wall["avoid"])
        self.assertIn("yaw 90", wall["sits"])
        floor = describe(catalog.get("NewRow_303"))
        self.assertEqual(floor["role"], "floor")
        self.assertIn("raw_pivot_write", floor["avoid"])
        car = describe(catalog.get("NewRow_775"))
        self.assertEqual(car["role"], "vehicle")
        self.assertIn("north_south_road", car["avoid"])
        cone = describe(catalog.get("NewRow_202"))
        self.assertFalse(cone["blocks_flight"])

        doc = new_map("Fit", "Fit")
        wall = catalog.get("NewRow_318")
        doc.add(wall, -4, -78, 0.1, yaw_deg=90, scale=(1, 2, 1), anchor="pivot")
        blocked = conflicts(
            catalog,
            doc.iter_placed(catalog),
            wall,
            -4,
            -78,
            0.1,
            yaw_deg=90,
            scale=(1, 2, 1),
            anchor="pivot",
        )
        self.assertFalse(blocked["clear"])
        extension = conflicts(
            catalog,
            doc.iter_placed(catalog),
            catalog.get("NewRow_182"),
            -6,
            -86,
            0.05,
            anchor="pivot",
        )
        self.assertTrue(extension["clear"], extension["hits"])

    def test_variety_policy_rejects_repeated_cars_and_bushes_on_gravel(self):
        from uncrashed_mcp.fit import variety_problems

        catalog = load_catalog()
        cars = [{"asset": "NewRow_775", "x": i * 6, "y": 0, "z": 0.22} for i in range(5)]
        self.assertTrue(any("cap" in problem for problem in variety_problems(cars, catalog)))
        self.assertEqual(variety_problems(cars[:4], catalog), [])
        on_gravel = [{"asset": "NewRow_164", "x": 0, "y": 0, "z": 0.34, "surface": "gravel"}]
        self.assertTrue(variety_problems(on_gravel, catalog))
        on_dirt = [{"asset": "NewRow_164", "x": 0, "y": 0, "z": 0.34, "surface": "dirt"}]
        self.assertEqual(variety_problems(on_dirt, catalog), [])
        tiles = [{"asset": "NewRow_181", "x": i * 4, "y": 0, "z": 0.02} for i in range(30)]
        self.assertEqual(variety_problems(tiles, catalog), [])

    def test_vertex_hull_keeps_the_scaffold_bay_open(self):
        from uncrashed_mcp.fit import conflicts
        from uncrashed_mcp.geom import world_aabb
        from uncrashed_mcp.meshes import shape_for

        catalog = load_catalog()
        scaffold = catalog.get("NewRow_157")
        cone = catalog.get("NewRow_202")
        pivot = (0.0, 0.0, 0.07)
        box = world_aabb(scaffold.origin_cm, scaffold.extent_cm, pivot, 0.0, (1.0, 1.0, 1.0))
        placed = [
            {
                "id": scaffold.id,
                "name": scaffold.name,
                "category": scaffold.category,
                "pivot_m": pivot,
                "yaw_deg": 0.0,
                "scale": (1.0, 1.0, 1.0),
                "aabb_m": box,
            }
        ]
        cx, cy, cz = (c / 100.0 for c in scaffold.origin_cm)
        centre = conflicts(catalog, placed, cone, cx, cy, 0.07 + cz, anchor="pivot")
        self.assertEqual(centre["tested"], "vertices")
        self.assertTrue(centre["clear"], centre["hits"])
        # The cone still sits inside the scaffold's box. The old box test called that a hit.
        self.assertLess(cx, box[3])
        self.assertGreater(cx, box[0])
        shape = shape_for(scaffold)
        far = max(shape.verts, key=lambda v: v[0] * v[0] + v[1] * v[1])
        ox, oy, oz = (c / 100.0 for c in cone.origin_cm)
        on_tube = conflicts(
            catalog,
            placed,
            cone,
            far[0] - ox,
            far[1] - oy,
            0.07 + far[2] - oz,
            anchor="pivot",
        )
        self.assertFalse(on_tube["clear"])
        container = catalog.get("NewRow_1")
        lower = {
            "id": container.id,
            "name": container.name,
            "category": container.category,
            "pivot_m": (0.0, 0.0, 0.07),
            "yaw_deg": 0.0,
            "scale": (1.0, 1.0, 1.0),
            "aabb_m": world_aabb(container.origin_cm, container.extent_cm, (0.0, 0.0, 0.07), 0.0, (1.0, 1.0, 1.0)),
        }
        stacked = conflicts(catalog, [lower], container, 0.0, 0.0, 0.07 + container.size_m[2], anchor="pivot")
        self.assertTrue(stacked["clear"], stacked["hits"])
        sunk = conflicts(catalog, [lower], container, 0.0, 0.0, 0.07 + container.size_m[2] - 0.4, anchor="pivot")
        self.assertFalse(sunk["clear"])


if __name__ == "__main__":
    unittest.main()
