"""How each catalog mesh sits, what it overlaps, and which part of a scene it belongs in.

Sizes and pivots come from the measured catalog. The notes below are the cases
that were checked in the open editor: a raw pivot write is not the same as
bottom_center, and a clone keeps the donor yaw.
"""

from __future__ import annotations

import math

from .geom import pivot_for_anchor, world_aabb
from .meshes import SLOP_M, meshes_overlap, shape_for

# A drone body is about this tall. Anything solid above it blocks the line.
FLIGHT_M = 1.5

# How many copies of one mesh a single spawn batch may contain.
# Surfaces tile, so they are not capped. Anything else repeated past the cap
# is the same object pasted across the yard.
REPEAT_CAP = {
    "vehicle": 4,
    "foliage": 4,
    "building": 1,
    "landmark": 1,
    "industrial": 6,
    "prop": 8,
    "barrier": 12,
    "street": 6,
    "column": 4,
    "wall": 8,
    "floor": 4,
    "primitive": 4,
    "figure": 2,
    "marker": 6,
    "decal": 8,
}
SOFT_SURFACE = ("dirt", "sand", "grass", "soil", "garden", "planter")
HARD_SURFACE = ("asphalt", "pavement", "gravel", "concrete", "flagstone", "slab")
# Keep this radius around spawn free of meshes that rise above the body.
SPAWN_CLEAR_M = 1.2

# Rows whose live behaviour was checked, not inferred from the bounds alone.
SEEN = {
    "NewRow_318": (
        "Concrete wall. At yaw 0 the 4m run is along Y; at yaw 90 it is along X. "
        "Scale (1, 2, 1) makes an 8m run, 0.3m thick and 4m tall. Courses stack every 4m "
        "(the perimeter uses z 0.1, 4.1 and 8.1). A run across a road is a dead end. "
        "Do not use these as a slalom."
    ),
    "NewRow_303": (
        "Concrete floor. The pivot is the top face, 0.29m above the bottom. "
        "A live move writes that pivot, so copying a ground z leaves the slab in the air. "
        "place_instances with bottom_center drops it onto the surface. Not a roof prop."
    ),
    "NewRow_122": (
        "0.6m column, 4m tall. It reads as a missing building, not as a gate or a slalom pole."
    ),
    "NewRow_100": (
        "Road barrier, 1.91m along X and 0.51m along Y at yaw 0. Every donor faces that way, "
        "and a live clone cannot turn it. Use it on an east-west curb. On a north-south curb it sits sideways."
    ),
    "NewRow_775": "Kangoo, long on X (4.21m). Park on an east-west curb, z matching the donor (0.22m).",
    "NewRow_103": "Camaro, long on X (4.78m). Same curb rule as the Kangoo. Donors at yaw 0 and 180.",
    "NewRow_93": "Van, long on X even at yaw 90 and 270. There is no donor whose long axis follows Y.",
    "NewRow_202": "Traffic cone, 0.28m wide and 0.5m tall. Round, so donor yaw does not matter. Sits at z 0.1.",
    "NewRow_164": (
        "Bush. Plant it on dirt, sand, or grass. On asphalt, pavement, gravel, or concrete it reads as a prop dropped on a slab. "
        "Donors that look planted use z 0.34."
    ),
    "NewRow_1": (
        "Container, 12.2m along Y at yaw 0 and along X at yaw 90. Yard only. 3m tall, not a gate. "
        "On a scaffold, the bottom has to clear the top rails. With courses 4.0m apart that is about "
        "4.6m above the top course pivot. Lower, and the container end sits inside the tower."
    ),
    "NewRow_157": (
        "Scaffold, 2.4m by 2.4m and 4.18m tall, base z 0.07. MCP_Highline stacks the next bay at +4.20m "
        "(0.07, 4.27, 8.47, …). A 4.00m step overlaps the bay above by 0.18m. One column per tower, "
        "spaced well apart. Highline spans them with Beam01, not with a container."
    ),
    "NewRow_33": (
        "Beam, 6.66m along X, 0.41m by 0.47m. Highline scales X between about 1.2 and 2.1 so it reaches "
        "the next support, and sets it at z 11.07 between scaffold courses. Yaw 0. Not in the Curb level, "
        "so a live clone needs that map loaded."
    ),
    "NewRow_788": (
        "House 02. The mesh hangs off the pivot: the body runs about 14m south of the pivot at yaw 0. "
        "Fit the mesh box, not the pivot, or a tower ends up inside the roof."
    ),
    "NewRow_26": "Container, same footprint as NewRow_1. Match the yaw of the row it continues.",
    "NewRow_27": "Container, same footprint as NewRow_1.",
    "NewRow_28": "Container, same footprint as NewRow_1.",
    "NewRow_48": "Horizontal tank, long axis follows yaw 90 in the yard. About 7m by 3m by 3m.",
    "NewRow_49": "Vertical tank, 5.2m across and 13m tall. One per yard, not on a sidewalk.",
    "NewRow_251": "Silo, 3.2m across and 8.7m tall. Base pivot, stands on the gravel.",
    "NewRow_196": "Forklift, 10m long. It belongs in the yard aisle, not on the road.",
    "NewRow_201": "Tower crane. Offset pivot; bottom_center is required or the mast misses the pad.",
    "NewRow_54": "Pallet, 1.34m by 1.3m and 0.17m tall. Stack the next one at z + 0.17. Donor sits at z 0.07.",
    "NewRow_182": "Asphalt tile, 4m by 4m. Centre on the 4m grid, pivot z 0.05. Edges should touch, not overlap.",
    "NewRow_185": "Pavement tile, 4m, pivot z 0.17. Sidewalks. Do not share a cell with asphalt.",
    "NewRow_181": "Gravel tile. Yard surface. Same 4m grid, lower than the sidewalk.",
    "NewRow_3": "Street lamp, about 10m tall. Curb line every 16m. The head overhangs; leave the lane centre clear.",
    "NewRow_11": "Office, 49m by 46m and 86m tall. One already fills the north-east block. Do not clone it.",
    "NewRow_187": (
        "Barrel, 0.56m across and 0.91m tall, base pivot. The props panel can also drop SM_Barrel03, "
        "which is the same family and is not a separate catalog row."
    ),
}


def _bottom_m(asset) -> float:
    """Mesh bottom relative to the actor pivot, in metres. Negative means the bottom is below the pivot."""
    return (asset.origin_cm[2] - asset.extent_cm[2]) / 100.0


def role_of(asset) -> str:
    """One of: surface, decal, wall, floor, column, barrier, vehicle, foliage, building, industrial, street, prop, primitive, figure, landmark."""
    tags = set(asset.tags)
    name = asset.name.lower()
    sx, sy, sz = asset.size_m
    foot = max(sx, sy)
    if asset.category == "Decals" or "decal" in tags:
        return "decal"
    if asset.category == "Ground" or asset.id in {
        "NewRow_182",
        "NewRow_185",
        "NewRow_197",
        "NewRow_186",
        "NewRow_376",
        "NewRow_180",
        "NewRow_181",
        "NewRow_662",
        "NewRow_184",
    }:
        return "surface"
    if "person" in tags:
        return "figure"
    if asset.category == "Foliage" or "tree" in tags or "bush" in tags:
        return "foliage"
    if "container" in tags or "crane" in tags or any(
        word in name for word in ("tank", "silo", "scaffold", "scafold", "fork", "container")
    ):
        return "industrial"
    if "vehicle" in tags or any(word in name for word in ("van", "camaro", "kangoo", "bus stop", "tractor")):
        return "vehicle"
    if "coin" in name:
        return "marker"
    if "column" in name or (asset.category == "Wall" and sx < 1.0 and sy < 1.0 and sz >= 2.0):
        return "column"
    if asset.category == "Wall" and sz <= 0.45 and foot >= 2.0:
        return "floor"
    if asset.pivot == "top" and sz <= 0.5 and foot >= 2.0:
        return "floor"
    if asset.category == "Wall" and sz >= 2.0 and min(sx, sy) <= 0.6:
        return "wall"
    if "fence" in tags or "barrier" in tags or "wall" in name:
        return "barrier" if sz < 2.2 else "wall"
    if asset.category == "Big structure" or "building" in tags or "house" in tags:
        if foot > 40.0 or sz > 40.0:
            return "landmark"
        return "building"
    if "lamp" in tags or "sign" in tags or "neon" in tags:
        return "street"
    if asset.category == "Basic shape":
        return "primitive"
    if foot < 2.0 and sz < 2.5:
        return "prop"
    if foot >= 15.0 or sz >= 20.0:
        return "landmark"
    return "industrial" if "industrial" in tags else "prop"


def _scenes(asset, role: str) -> list[str]:
    sx, sy, _sz = asset.size_m
    foot = max(sx, sy)
    if role == "surface":
        name = asset.name.lower()
        if "asphalt" in name:
            return ["road"]
        if "gravel" in name or "dirt" in name or "sand" in name:
            return ["yard"]
        if "wood" in name:
            return ["interior"]
        return ["sidewalk"]
    if role == "decal":
        return ["road", "sidewalk", "wall_face"]
    if role == "wall":
        return ["perimeter", "building_shell"]
    if role == "floor":
        return ["building_shell"]
    if role == "column":
        return ["building_shell"]
    if role == "barrier":
        return ["curb", "yard"]
    if role == "vehicle":
        if asset.long_axis == "x":
            return ["east_west_curb"]
        return ["north_south_curb"]
    if role == "foliage":
        return ["dirt", "sand", "grass"]
    if role == "building":
        return ["house_front"] if foot < 20 else ["block"]
    if role == "landmark":
        return []
    if role == "industrial":
        return ["yard", "hall"]
    if role == "street":
        return ["curb"]
    if role == "marker":
        return ["race_line"]
    if role == "figure":
        return ["sidewalk"]
    if role == "primitive":
        return []
    return ["sidewalk", "yard"]


def _avoid(asset, role: str) -> list[str]:
    out = []
    sx, sy, sz = asset.size_m
    if role in {"wall", "floor", "column", "building", "landmark", "industrial"} and sz > FLIGHT_M:
        out.append("lane")
    if role == "foliage":
        out.extend(["asphalt", "pavement", "gravel", "concrete"])
    if role == "vehicle" and asset.long_axis == "x":
        out.append("north_south_road")
    if role == "barrier" and asset.long_axis == "x" and max(sx, sy) > 1.2:
        out.append("north_south_curb")
    if asset.pivot in {"offset", "top"}:
        out.append("raw_pivot_write")
    if role == "landmark" or max(sx, sy) > 30 or sz > 40:
        out.append("this_board")
    if asset.hidden and role == "primitive":
        out.append("stand_in")
    if not asset.collision and role not in {"decal", "surface"}:
        out.append("expects_collision")
    return out


def _sits(asset, role: str) -> str:
    bottom = _bottom_m(asset)
    sx, sy, sz = (round(v, 2) for v in asset.size_m)
    if asset.id in SEEN:
        return SEEN[asset.id]
    if role == "surface":
        return f"{sx:.2f}m by {sy:.2f}m tile. Centre it on the 4m grid so edges meet. A second surface in the same cell z-fights."
    if role == "decal":
        return "Flat. It needs a surface under it and does not take space in the air."
    if asset.pivot == "top":
        return (
            f"Pivot is the top. The mesh hangs {abs(bottom):.2f}m below the pivot. "
            "Live moves write the pivot; bottom_center is what puts the underside on the ground."
        )
    if asset.pivot == "offset" or abs(bottom) > 0.15:
        return (
            f"Pivot is {abs(bottom):.2f}m off the mesh bottom. "
            "bottom_center lands it. A live clone copies the donor, so match that donor's z."
        )
    if asset.pivot == "centre":
        return f"Bounds are centred on the pivot. bottom_center lowers it by about {sz / 2:.2f}m."
    yaw = "along X" if asset.long_axis == "x" else "along Y"
    return (
        f"Base pivot, {sx:.2f}m by {sy:.2f}m, {sz:.2f}m tall, long axis {yaw} at yaw 0. "
        "The pivot z is the ground contact."
    )


def _stack(asset, role: str) -> str:
    _sx, _sy, sz = asset.size_m
    if role == "surface":
        return "Do not stack. Tiles share an edge, not a volume."
    if role == "decal":
        return "Lies on one face."
    if role in {"wall", "column"} and asset.pivot == "base":
        return f"Next course at z + {sz:.2f}m times scale Z."
    if role == "prop" and sz < 0.4 and asset.pivot == "base":
        return f"Stack at z + {sz:.2f}m."
    if asset.id == "NewRow_157":
        return "Next bay at z + 4.20m. The mesh is 4.18m, so a 4.00m step overlaps the bay above."
    if asset.pivot != "base":
        return "Do not stack. The pivot is not the underside."
    return "Sits once, on a surface."


def describe(asset) -> dict:
    """Fit record for one catalog asset."""
    role = role_of(asset)
    sx, sy, sz = asset.size_m
    foot = max(sx, sy)
    solid = asset.collision and role not in {"decal", "surface"}
    return {
        "id": asset.id,
        "name": asset.name,
        "role": role,
        "size_m": [round(sx, 3), round(sy, 3), round(sz, 3)],
        "long_axis": asset.long_axis,
        "yaw0_runs": "x" if asset.long_axis == "x" else "y",
        "yaw90_runs": "y" if asset.long_axis == "x" else "x",
        "pivot": asset.pivot,
        "bottom_below_pivot_m": round(-_bottom_m(asset), 3),
        "height_m": round(sz, 3),
        "footprint_m": round(foot, 3),
        "clearance_m": 0.0 if role in {"surface", "decal"} else SLOP_M,
        "blocks_flight": bool(solid and sz > FLIGHT_M),
        "scenes": _scenes(asset, role),
        "avoid": _avoid(asset, role),
        "stack": _stack(asset, role),
        "sits": _sits(asset, role),
        "hidden": asset.hidden,
        "collision": asset.collision,
        "repeat_cap": None if role == "surface" else REPEAT_CAP.get(role, 8),
    }


def variety_problems(placements: list[dict], catalog) -> list[str]:
    """Reasons a batch breaks the variety rules. Empty means it may be spawned.

    Surfaces may repeat. Every other asset has a per-batch cap, vehicles the tightest.
    Foliage is refused unless surface names dirt, sand, or grass, and it is refused on
    asphalt, pavement, gravel, and concrete.
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    foliage_bad = 0
    unknown = []
    for item in placements:
        asset_id = str(item.get("asset", ""))
        asset = catalog.assets.get(asset_id)
        if asset is None:
            unknown.append(asset_id)
            continue
        role = role_of(asset)
        if role != "surface":
            counts[asset_id] += 1
        if role != "foliage":
            continue
        surface = str(item.get("surface") or "").lower()
        soft = any(word in surface for word in SOFT_SURFACE)
        hard = any(word in surface for word in HARD_SURFACE)
        if hard or not soft:
            foliage_bad += 1
    problems = []
    if unknown:
        problems.append("unknown assets: " + ", ".join(sorted(set(unknown))[:6]))
    if foliage_bad:
        problems.append(
            f"{foliage_bad} foliage placements need surface dirt, sand, or grass, and cannot sit on asphalt, pavement, gravel, or concrete"
        )
    for asset_id, count in sorted(counts.items()):
        asset = catalog.assets[asset_id]
        role = role_of(asset)
        cap = REPEAT_CAP.get(role, 8)
        if count > cap:
            problems.append(f"{asset_id} ({asset.name}) ×{count} exceeds the {role} cap of {cap}")
    return problems


def knowledge(catalog) -> dict:
    """The whole catalog, grouped so a caller can see what fits together."""
    groups: dict[str, list[dict]] = {}
    for asset in catalog.assets.values():
        row = describe(asset)
        groups.setdefault(row["role"], []).append(
            {
                "id": row["id"],
                "name": row["name"],
                "size_m": row["size_m"],
                "pivot": row["pivot"],
                "scenes": row["scenes"],
                "avoid": row["avoid"],
                "blocks_flight": row["blocks_flight"],
            }
        )
    for rows in groups.values():
        rows.sort(key=lambda item: item["id"])
    return {
        "assets": len(catalog.assets),
        "counts": {role: len(rows) for role, rows in sorted(groups.items())},
        "rules": list(RULES),
        "construction": [dict(item) for item in CONSTRUCTION],
        "groups": groups,
    }


RULES = (
    "Live spawn and move write the actor pivot. place_instances with bottom_center writes the bounding-box bottom. They are not the same z for offset, top, or centre pivots.",
    "A live clone keeps the donor yaw. If every donor is long on X, the clone cannot face along Y.",
    "Ground tiles are 4m. Place the centre on the 4m grid. Sharing a cell z-fights; meeting at the edge is correct.",
    "Asphalt is the road (pivot z 0.05), pavement the sidewalk (0.17), gravel the yard. Do not mix them in one cell.",
    "Concrete walls are shells and the map edge. Across a road they end the map. They are not a slalom, and neither are columns or concrete floors.",
    "Cars on this board are long on X. Park them on an east-west curb. A north-south road gets cones, not those cars or the yaw-0 barriers.",
    "Anything solid taller than 1.5m blocks the drone. Cones, pallets and low barriers are flyable. Leave 1.2m around spawn clear of those tall meshes.",
    "Variety: one batch may contain at most 4 of each vehicle and 4 bushes, and foliage only on dirt, sand, or grass. Asphalt, pavement, gravel, and concrete are hard surfaces. Ground tiles are the exception and may repeat.",
    "Buildings wider than about 25m, and the 86m office, do not get cloned into a block that is already full.",
    "Containers, tanks, the crane and the forklift stay in the yard. Houses stay on the house front. Lamps stay on the curb, 16m apart. Coin rows are race markers, not floors.",
    "Overlap is tested on the mesh triangles. The bounding box is only a broad phase, so a bay, an eave, or the side of a wall is empty. Contact within clearance_m is allowed. Surfaces are the exception and should meet at an edge.",
    "Scaffold bays stack at +4.20m. House meshes are often offset from the pivot; fit the triangle hull, not the box. See construction.",
)


# Measured from the saved maps, not from the catalog bounds alone.
# asset_fit with no arguments returns this list.
CONSTRUCTION = (
    {
        "topic": "scaffold tower",
        "source": "MCP_Highline",
        "notes": (
            "NewRow_157 starts at z 0.07 and the next bay is +4.20m "
            "(0.07, 4.27, 8.47, 12.67, 16.87, 21.07, and so on up to 11 bays). "
            "The mesh is 4.18m tall, so a 4.00m step drives 0.18m into the bay above. "
            "Each tower is one column. Highline spaces those columns tens of metres apart "
            "and varies the height. It does not build an 8m lattice of identical towers."
        ),
    },
    {
        "topic": "tower span and deck",
        "source": "MCP_Highline",
        "notes": (
            "The span is Beam01 (NewRow_33), yaw 0, scaled on X from about 1.2 to 2.1 "
            "so the 6.66m beam reaches the next post. Those beams sit at z 11.07, between courses, "
            "not inside a bay. A plank (NewRow_158) is a thin top, scaled on X. "
            "Decks are Concrete Floor01 (NewRow_303). Its pivot is the top face, placed at z 8.1 and 16.1, "
            "and scaled in X and Y (about 1.5 to 3.5) to cover the bay. Floors are decks, not track obstacles. "
            "Concrete columns (NewRow_99) are a separate frame and stack every 4.0m. "
            "Beam01, the plank, and the pipes are in Highline, not in the open Curb level, so they cannot be cloned until that map is loaded."
        ),
    },
    {
        "topic": "container on a scaffold",
        "source": "live editor",
        "notes": (
            "A container is 2.6m wide and the scaffold is 2.4m, so the end covers the tower top. "
            "With courses 4.0m apart, the container bottom needs to be about 4.6m above the top course pivot "
            "or the end stays inside the top rails. Highline does not use a container as the span."
        ),
    },
    {
        "topic": "terrain",
        "source": "MCP_Highline, MCP_Works, MCP_Curb, MCP_DemoCity",
        "notes": (
            "The tiled maps use 4m squares at scale 1, centred on the 4m grid, one material per cell. "
            "Asphalt pivot z is 0.05, pavement 0.17, gravel 0.02, dirt 0.00. Edges meet. Overlapping cells z-fight. "
            "DemoCity changes the ground by scaling the tile: asphalt scale (36, 3) is a 144m by 12m road, "
            "and the sidewalks are pavement scaled to the strip beside it. Keep each material's z. "
            "A scaled tile still has to meet its neighbour on the edge. "
            "A concrete wall at scale (1, 2, 1) is an 8m run; courses are z 0.1, 4.1, 8.1."
        ),
    },
    {
        "topic": "street detail",
        "source": "MCP_DemoCity",
        "notes": (
            "Each house type faces the street, yaw 0 or 270, and is not repeated down the block. "
            "Several houses are offset from their pivot. House 02's body runs about 14m south of the pivot at yaw 0, "
            "so placing by the pivot puts the roof on whatever is south of that point. "
            "Lamps sit on both curbs, following the street, about 26m apart. "
            "Parked cars are mixed models, a few per block, yaw along the curb. "
            "The small commercial building uses a centre pivot at z 4.56, which puts its base on the ground."
        ),
    },
)


def placement_bounds(asset, x: float, y: float, z: float, yaw_deg: float, scale, anchor: str):
    """World AABB for a placement. (x, y, z) uses the same anchor as place_instances."""
    yaw = math.radians(yaw_deg)
    pivot = pivot_for_anchor(asset.origin_cm, asset.extent_cm, x, y, z, yaw, scale, anchor)
    return pivot, world_aabb(asset.origin_cm, asset.extent_cm, pivot, yaw, scale)


def _hits(a, b, gap: float) -> bool:
    return (
        a[0] < b[3] + gap
        and b[0] < a[3] + gap
        and a[1] < b[4] + gap
        and b[1] < a[4] + gap
        and a[2] < b[5] + gap
        and b[2] < a[5] + gap
    )


def _mesh_hit(catalog, asset, pivot, yaw_deg: float, scale, item) -> bool | None:
    """Triangle test. None when either side has no hull, so the caller keeps the box result."""
    ours = shape_for(asset)
    other = shape_for(catalog.assets.get(item.get("id")))
    if ours is None or other is None:
        return None
    other_yaw = math.radians(float(item.get("yaw_deg") or 0.0))
    other_scale = item.get("scale") or (1.0, 1.0, 1.0)
    return meshes_overlap(
        ours,
        pivot,
        math.radians(yaw_deg),
        scale,
        other,
        item["pivot_m"],
        other_yaw,
        other_scale,
    )


def conflicts(catalog, placed: list[dict], asset, x: float, y: float, z: float, yaw_deg: float = 0.0, scale=(1.0, 1.0, 1.0), anchor: str = "bottom_center", limit: int = 8) -> dict:
    """What a proposed placement would intersect on an already-built map."""
    info = describe(asset)
    ours = shape_for(asset)
    # Without a hull the box still has to be tight. 1.5cm is contact, not a keep-out.
    gap = 0.0 if info["role"] in {"surface", "decal"} or ours is not None else SLOP_M
    pivot, box = placement_bounds(asset, x, y, z, yaw_deg, scale, anchor)
    hits = []
    tested = "vertices" if ours is not None else "bounds"
    for item in placed:
        other = item.get("aabb_m")
        if not other:
            continue
        use_gap = 0.0 if item.get("category") == "Ground" and info["role"] == "surface" else gap
        # Surfaces z-fight only when they occupy the same slab, not when a prop stands on them.
        if info["role"] == "surface" and item.get("category") == "Ground":
            same_slab = abs(box[2] - other[2]) < 0.05 and abs(box[5] - other[5]) < 0.05
            xy = box[0] < other[3] - 0.05 and other[0] < box[3] - 0.05 and box[1] < other[4] - 0.05 and other[1] < box[4] - 0.05
            if not (same_slab and xy):
                continue
        elif not _hits(box, other, use_gap):
            continue
        if info["role"] != "surface" and item.get("category") == "Ground":
            # Standing on a tile is contact, not a clash, when the mesh bottom is within 5cm of the tile top.
            if box[2] >= other[5] - 0.05:
                continue
        if ours is not None and info["role"] != "surface" and item.get("category") != "Ground":
            verdict = _mesh_hit(catalog, asset, pivot, yaw_deg, scale, item)
            if verdict is False:
                continue
        hits.append(
            {
                "id": item["id"],
                "name": item["name"],
                "at_m": [round(v, 2) for v in item["pivot_m"]],
            }
        )
        if len(hits) >= limit:
            break
    return {
        "asset": info["id"],
        "role": info["role"],
        "aabb_m": [round(v, 3) for v in box],
        "clearance_m": SLOP_M if ours is not None else gap,
        "tested": tested,
        "blocks_flight": info["blocks_flight"],
        "avoid": info["avoid"],
        "hits": hits,
        "clear": not hits,
    }
