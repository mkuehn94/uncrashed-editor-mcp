"""City, road and structure builders.

Lengths are metres. A generated city is centred on the origin. Roads, sidewalks
and lots use separate height bands so coplanar materials do not z-fight.
Mesh pivots are compensated from the catalogue bounds.
"""

from __future__ import annotations

import math
import random

from .catalog import Asset, Catalog
from .geom import l2w, pivot_for_anchor, world_aabb, yaw_for_long_axis
from .mapfile import MapFile

WALLS = (
    "NewRow_318",
    "NewRow_207",
    "NewRow_214",
    "NewRow_224",
    "NewRow_321",
    "NewRow_348",
)
FLOOR = "NewRow_303"
PARAPET = "NewRow_347"
LAMP = "NewRow_3"
BENCH = "NewRow_204"
CARS = ("NewRow_775", "NewRow_93", "NewRow_103")
FENCE = "NewRow_780"
MAX_INSTANCES = 25000

Rect = tuple[float, float, float, float]


def _layer(catalog: Catalog, name: str, ground_z: float) -> float:
    return ground_z + float(catalog.library["layers_m"][name])


def _ground(catalog: Catalog, name: str) -> str:
    return catalog.library["grounds"][name]


def add_plane(
    doc: MapFile,
    catalog: Catalog,
    row: str,
    cx: float,
    cy: float,
    length: float,
    width: float,
    z_top: float,
    yaw: float = 0.0,
) -> None:
    """One triplanar ground tile scaled to a rectangle. Local +X runs along `yaw`."""
    asset = catalog.get(row)
    sx = max(length, 0.4) / asset.size_m[0]
    sy = max(width, 0.4) / asset.size_m[1]
    doc.add(asset, cx, cy, z_top, yaw_deg=math.degrees(yaw), scale=(sx, sy, 1.0), anchor="top_center")


def add_rect(doc: MapFile, catalog: Catalog, material: str, rect: Rect, z_top: float) -> None:
    x, y, w, h = rect
    if w < 1 or h < 1:
        return
    add_plane(doc, catalog, _ground(catalog, material), x + w / 2, y + h / 2, w, h, z_top, 0.0)


def _city_layout(blocks_x: int, blocks_y: int, block: float, road: float) -> tuple[list[Rect], list[Rect], list[Rect]]:
    """Horizontal roads include the junctions. Vertical roads are the gaps between them."""
    pitch = block + road
    total_x = blocks_x * block + (blocks_x + 1) * road
    total_y = blocks_y * block + (blocks_y + 1) * road
    x0 = -total_x / 2
    y0 = -total_y / 2
    horizontal: list[Rect] = []
    vertical: list[Rect] = []
    blocks: list[Rect] = []
    for j in range(blocks_y + 1):
        horizontal.append((x0, y0 + j * pitch, total_x, road))
    for i in range(blocks_x + 1):
        for j in range(blocks_y):
            vertical.append((x0 + i * pitch, y0 + road + j * pitch, road, block))
    for j in range(blocks_y):
        for i in range(blocks_x):
            blocks.append((x0 + road + i * pitch, y0 + road + j * pitch, block, block))
    return horizontal, vertical, blocks


def _inset(rect: Rect, margin: float) -> Rect:
    x, y, w, h = rect
    return (x + margin, y + margin, w - 2 * margin, h - 2 * margin)


def _wall(
    doc: MapFile,
    catalog: Catalog,
    row: str,
    cx: float,
    cy: float,
    length: float,
    height: float,
    heading: float,
    z_bottom: float,
) -> None:
    asset = catalog.get(row)
    sy = length / asset.size_m[1]
    sz = height / asset.size_m[2]
    # Concrete wall sections are 4 m along local Y.
    doc.add(
        asset,
        cx,
        cy,
        z_bottom,
        yaw_deg=math.degrees(heading - math.pi / 2),
        scale=(1.0, sy, sz),
        anchor="bottom_center",
    )


def build_structure(
    doc: MapFile,
    catalog: Catalog,
    cx: float,
    cy: float,
    length: float,
    width: float,
    height: float,
    yaw_deg: float = 0.0,
    ground_z: float = 0.0,
    wall: str | None = None,
    door: bool = True,
    rng: random.Random | None = None,
) -> dict:
    """A panel building. `length` runs along yaw, `width` to the left. Bottom sits on ground_z."""
    rng = rng or random.Random(0)
    floors = max(1, int(round(height / 4)))
    height = floors * 4
    if length < 3 or width < 3:
        raise ValueError("structure footprint must be at least 3 m by 3 m")
    row = wall or rng.choice(WALLS)
    if row not in catalog.assets:
        row = WALLS[0]
    yaw = math.radians(yaw_deg)
    sides = (
        (0.0, -width / 2, length, yaw, yaw - math.pi / 2, True),
        (0.0, width / 2, length, yaw, yaw + math.pi / 2, False),
        (length / 2, 0.0, width, yaw + math.pi / 2, yaw, False),
        (-length / 2, 0.0, width, yaw + math.pi / 2, yaw + math.pi, False),
    )
    for u, v, side_len, heading, outward, is_front in sides:
        wx, wy = l2w(cx, cy, yaw, u, v)
        _wall(doc, catalog, row, wx, wy, side_len, height, heading, ground_z)
        parapet = catalog.assets.get(PARAPET)
        if parapet is not None:
            doc.add(
                parapet,
                wx,
                wy,
                ground_z + height,
                yaw_deg=math.degrees(heading - math.pi / 2),
                scale=(1.0, side_len / parapet.size_m[1], 1.0),
                anchor="bottom_center",
            )
    floor = catalog.get(FLOOR)
    doc.add(
        floor,
        cx,
        cy,
        ground_z + height,
        yaw_deg=yaw_deg,
        scale=(length / floor.size_m[0], width / floor.size_m[1], 1.0),
        anchor="top_center",
    )
    return {"floors": floors, "height_m": height, "wall": row, "footprint_m": [length, width]}


def _fit_mesh(
    doc: MapFile,
    asset: Asset,
    cx: float,
    cy: float,
    lx: float,
    ly: float,
    z: float,
    yaw: float,
    min_scale: float,
    max_scale: float,
) -> float | None:
    """Uniform-scale a mesh so its footprint fits lx along `yaw` and ly to the left."""
    best: tuple[float, float] | None = None
    for extra, along, across in ((0.0, lx, ly), (math.pi / 2, ly, lx)):
        scale = min(along / asset.size_m[0], across / asset.size_m[1], max_scale)
        if scale >= min_scale and (best is None or scale > best[0]):
            best = (scale, yaw + extra)
    if best is None:
        return None
    scale, use_yaw = best
    doc.add(
        asset,
        cx,
        cy,
        z,
        yaw_deg=math.degrees(use_yaw),
        scale=(scale, scale, scale),
        anchor="bottom_center",
    )
    return scale


def build_road(
    doc: MapFile,
    catalog: Catalog,
    points: list[dict],
    width: float = 12.0,
    ground_z: float = 0.0,
    sidewalk: bool = True,
    cars: bool = True,
    lamps: bool = True,
    seed: int = 1,
) -> dict:
    """Pave a polyline. Points are {x, y} in metres. Width includes the carriageway only."""
    if len(points) < 2:
        raise ValueError("a road needs at least two points")
    rng = random.Random(seed)
    z_road = _layer(catalog, "asphalt", ground_z)
    z_walk = _layer(catalog, "sidewalk", ground_z)
    asphalt = _ground(catalog, "asphalt")
    pavement = _ground(catalog, "pavement")
    car_assets = [catalog.assets[row] for row in CARS if row in catalog.assets]
    lamp = catalog.assets.get(LAMP)
    segments = 0
    for index, (a, b) in enumerate(zip(points, points[1:])):
        x0, y0, x1, y1 = float(a["x"]), float(a["y"]), float(b["x"]), float(b["y"])
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1:
            continue
        heading = math.atan2(dy, dx)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        # Crossings of a later segment sit 4 mm higher so the joint does not flicker.
        add_plane(doc, catalog, asphalt, cx, cy, length + 0.3, width, z_road + 0.004 * (index % 2), heading)
        if sidewalk:
            for side in (-1, 1):
                ox = math.sin(heading) * side * (width / 2 + 2)
                oy = -math.cos(heading) * side * (width / 2 + 2)
                add_plane(
                    doc,
                    catalog,
                    pavement,
                    cx + ox,
                    cy + oy,
                    length,
                    4,
                    z_walk,
                    heading,
                )
        if cars and car_assets:
            step = 18.0
            count = max(0, int((length - 12) / step))
            for n in range(count):
                t = (n + 0.5) / count
                px = x0 + dx * t
                py = y0 + dy * t
                lane = 2.3 if n % 2 == 0 else -2.3
                face = heading if lane > 0 else heading + math.pi
                vx = px + math.sin(heading) * lane
                vy = py - math.cos(heading) * lane
                car = car_assets[rng.randrange(len(car_assets))]
                doc.add(
                    car,
                    vx,
                    vy,
                    z_road,
                    yaw_deg=math.degrees(yaw_for_long_axis(car.size_m, face)),
                    anchor="bottom_center",
                )
        if lamps and lamp is not None:
            step = 24.0
            count = max(1, int(length / step))
            for n in range(count):
                t = (n + 0.5) / count
                px = x0 + dx * t
                py = y0 + dy * t
                for side in (-1, 1):
                    dist = width / 2 + 1.4
                    lx = px + math.sin(heading) * side * dist
                    ly = py - math.cos(heading) * side * dist
                    doc.add(lamp, lx, ly, z_walk, yaw_deg=math.degrees(heading), anchor="bottom_center")
        segments += 1
    return {"segments": segments, "width_m": width}


def _cars_on_rect(
    doc: MapFile,
    catalog: Catalog,
    rect: Rect,
    along_x: bool,
    z: float,
    spacing: float,
    inset: float,
    rng: random.Random,
) -> None:
    cars = [catalog.assets[row] for row in CARS if row in catalog.assets]
    if not cars:
        return
    x, y, w, h = rect
    if along_x:
        start, end = x + inset, x + w - inset
        if end - start < 8:
            return
        n = max(1, int((end - start) / spacing))
        for i in range(n):
            px = start + (i + 0.5) * (end - start) / n
            lane = 2.4 if i % 2 == 0 else -2.4
            heading = 0.0 if lane < 0 else math.pi
            car = cars[rng.randrange(len(cars))]
            doc.add(
                car,
                px,
                y + h / 2 + lane,
                z,
                yaw_deg=math.degrees(yaw_for_long_axis(car.size_m, heading)),
                anchor="bottom_center",
            )
    else:
        start, end = y + inset, y + h - inset
        if end - start < 8:
            return
        n = max(1, int((end - start) / spacing))
        for i in range(n):
            py = start + (i + 0.5) * (end - start) / n
            lane = 2.4 if i % 2 == 0 else -2.4
            heading = math.pi / 2 if lane < 0 else -math.pi / 2
            car = cars[rng.randrange(len(cars))]
            doc.add(
                car,
                x + w / 2 + lane,
                py,
                z,
                yaw_deg=math.degrees(yaw_for_long_axis(car.size_m, heading)),
                anchor="bottom_center",
            )


def _lamps_around(doc: MapFile, catalog: Catalog, rect: Rect, z: float, spacing: float = 26.0) -> None:
    lamp = catalog.assets.get(LAMP)
    if lamp is None:
        return
    x, y, w, h = rect
    inset = 1.6
    edges = (
        (x + 6, x + w - 6, y + inset, True),
        (x + 6, x + w - 6, y + h - inset, True),
        (y + 6, y + h - 6, x + inset, False),
        (y + 6, y + h - 6, x + w - inset, False),
    )
    for a, b, fixed, horizontal in edges:
        if b - a < 8:
            continue
        n = max(1, int((b - a) / spacing))
        for i in range(n):
            t = a + (i + 0.5) * (b - a) / n
            if horizontal:
                doc.add(lamp, t, fixed, z, anchor="bottom_center")
            else:
                doc.add(lamp, fixed, t, z, anchor="bottom_center")


def _pick(catalog: Catalog, predicate, limit: int = 12) -> list[Asset]:
    found = [asset for asset in catalog.assets.values() if predicate(asset) and not asset.bounds_estimated]
    found.sort(key=lambda asset: asset.id)
    return found[:limit]


def _houses(catalog: Catalog) -> list[Asset]:
    return _pick(
        catalog,
        lambda a: "house" in a.tags
        and not a.hidden
        and 6 <= min(a.size_m[0], a.size_m[1])
        and max(a.size_m[0], a.size_m[1]) <= 22
        and 4 < a.size_m[2] < 16,
    )


def _trees(catalog: Catalog) -> list[Asset]:
    return _pick(
        catalog,
        lambda a: "tree" in a.tags
        and not a.hidden
        and 4 <= a.size_m[2] <= 16
        and max(a.size_m[0], a.size_m[1]) <= 12,
    )


def _industrial_props(catalog: Catalog) -> list[Asset]:
    def ok(asset: Asset) -> bool:
        if asset.hidden or asset.bounds_estimated:
            return False
        footprint = max(asset.size_m[0], asset.size_m[1])
        if footprint > 28 or asset.size_m[2] > 30:
            return False
        blob = f"{asset.name} {' '.join(asset.tags)}".lower()
        return any(word in blob for word in ("container", "silo", "barn", "crane", "tank"))

    return _pick(catalog, ok, 16)


def _footprint(world_x: float, world_y: float, yaw: float) -> tuple[float, float]:
    """Length along `yaw` and width to its left, for an axis-aligned world footprint."""
    if abs(math.cos(yaw)) > 0.5:
        return world_x, world_y
    return world_y, world_x


def _lots(inner: Rect) -> list[tuple[Rect, float]]:
    """Four lots and the heading of the street they face, in radians."""
    x, y, w, h = inner
    gap = 3.0
    lw, lh = (w - gap) / 2, (h - gap) / 2
    if lw < 8 or lh < 8:
        return [(inner, -math.pi / 2)]
    specs = (
        ((x, y, lw, lh), -math.pi / 2),
        ((x + lw + gap, y, lw, lh), 0.0),
        ((x, y + lh + gap, lw, lh), math.pi),
        ((x + lw + gap, y + lh + gap, lw, lh), math.pi / 2),
    )
    return list(specs)


def _fill_block(
    doc: MapFile,
    catalog: Catalog,
    rect: Rect,
    style: str,
    ground_z: float,
    sidewalk: float,
    rng: random.Random,
) -> None:
    z_walk = _layer(catalog, "sidewalk", ground_z)
    z_dirt = _layer(catalog, "dirt", ground_z)
    z_gravel = _layer(catalog, "gravel", ground_z)
    if style == "downtown":
        add_rect(doc, catalog, "pavement2", rect, z_walk)
        surface_z = z_walk
    elif style == "industrial":
        add_rect(doc, catalog, "gravel", rect, z_gravel)
        surface_z = z_gravel
    else:
        x, y, w, h = rect
        band = min(sidewalk, w / 5, h / 5)
        inner = _inset(rect, band)
        add_rect(doc, catalog, "dirt", (inner[0], inner[1], inner[2], inner[3]), z_dirt + 0.01)
        add_rect(doc, catalog, "pavement", (x, y, w, band), z_walk)
        add_rect(doc, catalog, "pavement", (x, y + h - band, w, band), z_walk)
        add_rect(doc, catalog, "pavement", (x, y + band, band, h - 2 * band), z_walk)
        add_rect(doc, catalog, "pavement", (x + w - band, y + band, band, h - 2 * band), z_walk)
        surface_z = z_dirt + 0.01
    _lamps_around(doc, catalog, rect, z_walk)
    inner = _inset(rect, sidewalk)
    if inner[2] < 10 or inner[3] < 10:
        return
    if style == "downtown":
        _downtown_lots(doc, catalog, inner, surface_z, rng)
    elif style == "suburb":
        _suburb_lots(doc, catalog, rect, inner, surface_z, ground_z, rng)
    else:
        _industrial_lots(doc, catalog, inner, surface_z, rng)


def _downtown_lots(doc: MapFile, catalog: Catalog, inner: Rect, z: float, rng: random.Random) -> None:
    for lot, face in _lots(inner):
        margin = 0.8
        lx = lot[2] - margin
        ly = lot[3] - margin
        cx, cy = lot[0] + lot[2] / 2, lot[1] + lot[3] / 2
        # length runs along yaw; yaw is chosen so the front (-v) points along `face`.
        yaw = face + math.pi / 2
        length, width = _footprint(lx, ly, yaw)
        height = rng.choice((8, 12, 12, 16, 20, 24))
        build_structure(
            doc,
            catalog,
            cx,
            cy,
            length,
            width,
            height,
            yaw_deg=math.degrees(yaw),
            ground_z=z,
            wall=rng.choice(WALLS),
            rng=rng,
        )


def _suburb_lots(
    doc: MapFile, catalog: Catalog, block: Rect, inner: Rect, z: float, ground_z: float, rng: random.Random
) -> None:
    houses = _houses(catalog)
    trees = _trees(catalog)
    for lot, face in _lots(inner):
        cx, cy = lot[0] + lot[2] / 2, lot[1] + lot[3] / 2
        yaw = face + math.pi / 2
        placed = False
        rng.shuffle(houses)
        for house in houses:
            scale = _fit_mesh(doc, house, cx, cy, lot[2] * 0.92, lot[3] * 0.92, z, yaw, 0.62, 1.0)
            if scale is not None:
                placed = True
                depth = min(house.size_m[0], house.size_m[1]) * scale
                _driveway(doc, catalog, block, cx, cy, face, depth, ground_z)
                break
        if not placed:
            build_structure(
                doc, catalog, cx, cy, lot[2] * 0.7, lot[3] * 0.7, 8, math.degrees(yaw), z, rng=rng
            )
        if trees:
            for _ in range(rng.randint(1, 3)):
                tree = trees[rng.randrange(len(trees))]
                tx = cx + rng.uniform(-lot[2] * 0.35, lot[2] * 0.35)
                ty = cy + rng.uniform(-lot[3] * 0.35, lot[3] * 0.35)
                if math.hypot(tx - cx, ty - cy) < 3:
                    continue
                doc.add(tree, tx, ty, z, yaw_deg=rng.uniform(0, 360), anchor="bottom_center")


def _driveway(
    doc: MapFile,
    catalog: Catalog,
    block: Rect,
    cx: float,
    cy: float,
    face: float,
    depth: float,
    ground_z: float,
) -> None:
    bx, by, bw, bh = block
    half = depth / 2 + 0.4
    # Sit just above the sidewalk band so the apron is visible across the curb.
    z_top = _layer(catalog, "sidewalk", ground_z) + 0.02
    if abs(face + math.pi / 2) < 0.1:  # south
        add_rect(doc, catalog, "asphalt", (cx - 1.8, by, 3.6, max(1.0, cy - half - by)), z_top)
    elif abs(face - math.pi / 2) < 0.1:  # north
        add_rect(doc, catalog, "asphalt", (cx - 1.8, cy + half, 3.6, max(1.0, by + bh - (cy + half))), z_top)
    elif abs(face) < 0.1:  # east
        add_rect(doc, catalog, "asphalt", (cx + half, cy - 1.8, max(1.0, bx + bw - (cx + half)), 3.6), z_top)
    elif abs(abs(face) - math.pi) < 0.1:  # west
        add_rect(doc, catalog, "asphalt", (bx, cy - 1.8, max(1.0, cx - half - bx), 3.6), z_top)


def _industrial_lots(doc: MapFile, catalog: Catalog, inner: Rect, z: float, rng: random.Random) -> None:
    props = _industrial_props(catalog)
    fence = catalog.assets.get(FENCE)
    x, y, w, h = inner
    if fence is not None:
        for cx, cy, length, heading in (
            (x + w / 2, y, w, 0.0),
            (x + w / 2, y + h, w, 0.0),
            (x, y + h / 2, h, math.pi / 2),
            (x + w, y + h / 2, h, math.pi / 2),
        ):
            _wall(doc, catalog, fence.id, cx, cy, length, fence.size_m[2], heading, z)
    if not props:
        return
    for _ in range(rng.randint(2, 4)):
        asset = props[rng.randrange(len(props))]
        px = x + rng.uniform(w * 0.2, w * 0.8)
        py = y + rng.uniform(h * 0.2, h * 0.8)
        _fit_mesh(doc, asset, px, py, w * 0.42, h * 0.42, z, rng.uniform(0, math.tau), 0.45, 1.0)


def _plaza(doc: MapFile, catalog: Catalog, rect: Rect, ground_z: float, rng: random.Random) -> tuple[float, float, float]:
    z = _layer(catalog, "flagstone", ground_z)
    add_rect(doc, catalog, "flagstone", rect, z)
    _lamps_around(doc, catalog, rect, _layer(catalog, "sidewalk", ground_z))
    cx, cy = rect[0] + rect[2] / 2, rect[1] + rect[3] / 2
    bench = catalog.assets.get(BENCH)
    radius = min(rect[2], rect[3]) * 0.22
    if bench is not None:
        for yaw_face, ox, oy in ((-90, 0, radius), (90, 0, -radius), (180, radius, 0), (0, -radius, 0)):
            doc.add(bench, cx + ox, cy + oy, z, yaw_deg=yaw_face, anchor="bottom_center")
    trees = _trees(catalog)
    if trees:
        for ox, oy in ((radius * 1.6, radius * 1.6), (-radius * 1.6, radius * 1.6), (radius * 1.6, -radius * 1.6), (-radius * 1.6, -radius * 1.6)):
            tree = trees[rng.randrange(len(trees))]
            doc.add(tree, cx + ox, cy + oy, z, yaw_deg=rng.uniform(0, 360), anchor="bottom_center")
    return cx, cy, z + 1.3


# Local axis that faces the street. Towers use the long glazed side (+X when Y is longer).
# Houses use +Y, the pivot end, which is the porch in the editor.
ENTRANCE = {
    "NewRow_9": "+x",
    "NewRow_95": "+y",
    "NewRow_97": "+x",
    "NewRow_134": "+y",
    "NewRow_220": "+x",
    "NewRow_228": "+x",
    "NewRow_788": "+y",
    "NewRow_789": "+y",
    "NewRow_790": "+y",
    "NewRow_791": "+y",
    "NewRow_793": "+y",
    "NewRow_794": "+x",
    "NewRow_795": "+x",
    "NewRow_796": "+x",
    "NewRow_797": "+x",
}
# Natural-scale meshes. Order is a preference list; each row starts at a different offset.
STREET_BUILDINGS = (
    "NewRow_796",
    "NewRow_9",
    "NewRow_228",
    "NewRow_788",
    "NewRow_794",
    "NewRow_789",
    "NewRow_795",
    "NewRow_793",
    "NewRow_790",
    "NewRow_95",
    "NewRow_791",
    "NewRow_134",
    "NewRow_97",
)


def _yaw_entrance(entrance: str, face: float) -> float:
    """Yaw that points `entrance` (a local axis) along world heading `face`."""
    if entrance == "+x":
        return face
    if entrance == "-x":
        return face + math.pi
    if entrance == "+y":
        return face - math.pi / 2
    if entrance == "-y":
        return face + math.pi / 2
    raise ValueError(f"unknown entrance {entrance!r}")


def _box(asset: Asset, x: float, y: float, z: float, yaw: float, scale: tuple[float, float, float] = (1, 1, 1)):
    pivot = pivot_for_anchor(asset.origin_cm, asset.extent_cm, x, y, z, yaw, scale, "bottom_center")
    return world_aabb(asset.origin_cm, asset.extent_cm, pivot, yaw, scale)


def _hits(box, blocked: list[tuple[float, float, float, float]], pad: float = 0.3) -> bool:
    x0, y0, _, x1, y1, _ = box
    for rx0, ry0, rx1, ry1 in blocked:
        if x0 < rx1 + pad and rx0 < x1 + pad and y0 < ry1 + pad and ry0 < y1 + pad:
            return True
    return False


def _inside_rect(box, rect: Rect, pad: float = 0.15) -> bool:
    x0, y0, _, x1, y1, _ = box
    rx, ry, rw, rh = rect
    return x0 >= rx + pad and y0 >= ry + pad and x1 <= rx + rw - pad and y1 <= ry + rh - pad


def _street_zones(blocks_x: int, blocks_y: int, block_m: float, road_m: float) -> dict:
    hx = blocks_x * block_m / 2
    hy = blocks_y * block_m / 2
    hr = road_m / 2
    walk = 4.0
    front = hr + walk
    lot = 34.0
    roads = [
        (-hx, -hr, 2 * hx, road_m),
        (-hr, -hy, road_m, hy - hr),
        (-hr, hr, road_m, hy - hr),
    ]
    return {
        "half_x": hx,
        "half_y": hy,
        "half_road": hr,
        "walk": walk,
        "front": front,
        "lot": lot,
        "roads": roads,
    }


def _build_street(
    doc: MapFile,
    catalog: Catalog,
    blocks_x: int,
    blocks_y: int,
    block_m: float,
    road_m: float,
    seed: int,
    ground_z: float,
) -> dict:
    """One crossing: real buildings at natural size, parked cars only on asphalt."""
    zone = _street_zones(blocks_x, blocks_y, block_m, road_m)
    hx, hy = zone["half_x"], zone["half_y"]
    hr, front, lot = zone["half_road"], zone["front"], zone["lot"]
    rng = random.Random(seed)
    z_road = _layer(catalog, "asphalt", ground_z)
    z_walk = _layer(catalog, "sidewalk", ground_z)
    z_lot = z_walk - 0.004
    # Carriageway. The north-south arms stop at the crossing so the same asphalt does not z-fight.
    add_rect(doc, catalog, "asphalt", (-hx, -hr, 2 * hx, road_m), z_road)
    add_rect(doc, catalog, "asphalt", (-hr, -hy, road_m, hy - hr), z_road)
    add_rect(doc, catalog, "asphalt", (-hr, hr, road_m, hy - hr), z_road)
    # Sidewalks stop at the crossing. Lot bands sit 4 mm lower and tuck 15 cm under the curb.
    for x0, x1 in ((-hx, -front), (front, hx)):
        add_rect(doc, catalog, "pavement", (x0, hr, x1 - x0, front - hr), z_walk)
        add_rect(doc, catalog, "pavement", (x0, -front, x1 - x0, front - hr), z_walk)
        add_rect(doc, catalog, "pavement3", (x0, front - 0.15, x1 - x0, lot + 0.15), z_lot)
        add_rect(doc, catalog, "pavement3", (x0, -(front + lot), x1 - x0, lot + 0.15), z_lot)
    for y0, y1 in ((-hy, -(front + lot)), (front + lot, hy)):
        add_rect(doc, catalog, "pavement", (-front, y0, front - hr, y1 - y0), z_walk)
        add_rect(doc, catalog, "pavement", (hr, y0, front - hr, y1 - y0), z_walk)
        add_rect(doc, catalog, "pavement3", (-(front + lot), y0, lot + 0.15, y1 - y0), z_lot)
        add_rect(doc, catalog, "pavement3", (front - 0.15, y0, lot + 0.15, y1 - y0), z_lot)

    assets = [catalog.assets[row] for row in STREET_BUILDINGS if row in catalog.assets]
    if not assets:
        raise RuntimeError("catalogue has none of the street buildings")
    # Carriageway and sidewalks. Buildings test this and are not allowed to step into it.
    reserved: list[tuple[float, float, float, float]] = [(-hx, -front, hx, front), (-front, -hy, front, hy)]
    solids: list[tuple[float, float, float, float]] = []
    frontage: list[tuple[float, float]] = []

    def pack(origin: float, limit: float, fixed: float, along_x: bool, inward: float, face: float, cursor: int) -> int:
        gap = 4.0
        cursor_pos = origin
        tries = 0
        while cursor_pos < limit - 6 and tries < len(assets) * 3:
            asset = assets[cursor % len(assets)]
            cursor += 1
            tries += 1
            entrance = ENTRANCE.get(asset.id, "+y")
            yaw = _yaw_entrance(entrance, face)
            probe = _box(asset, 0, 0, z_walk, yaw)
            span_x = probe[3] - probe[0]
            span_y = probe[4] - probe[1]
            along = span_x if along_x else span_y
            depth = span_y if along_x else span_x
            if depth > lot - 1 or cursor_pos + along > limit:
                continue
            if along_x:
                cx = cursor_pos + along / 2
                cy = fixed + inward * depth / 2
            else:
                cx = fixed + inward * depth / 2
                cy = cursor_pos + along / 2
            box = _box(asset, cx, cy, z_walk, yaw)
            if _hits(box, reserved, 0.0) or _hits(box, solids, 0.5):
                continue
            doc.add(asset, cx, cy, z_walk, yaw_deg=math.degrees(yaw) % 360, anchor="bottom_center")
            solids.append((box[0], box[1], box[3], box[4]))
            if abs(face - math.pi / 2) < 0.1:
                frontage.append((cx, box[4]))
            cursor_pos += along + gap
            tries = 0
        return cursor

    edge = front + 0.4
    start = rng.randrange(len(assets))
    # face is the heading the entrance points, toward the carriageway.
    start = pack(-hx + 2, -edge - 1, -edge, True, -1, math.pi / 2, start)
    start = pack(edge + 1, hx - 2, -edge, True, -1, math.pi / 2, start + 1)
    start = pack(-hx + 2, -edge - 1, edge, True, 1, -math.pi / 2, start + 1)
    start = pack(edge + 1, hx - 2, edge, True, 1, -math.pi / 2, start + 1)
    start = pack(-hy + 2, -(front + lot) - 1, -edge, False, -1, 0.0, start + 1)
    start = pack(front + lot + 1, hy - 2, -edge, False, -1, 0.0, start + 1)
    start = pack(-hy + 2, -(front + lot) - 1, edge, False, 1, math.pi, start + 1)
    pack(front + lot + 1, hy - 2, edge, False, 1, math.pi, start + 1)

    cars = [catalog.assets[row] for row in CARS if row in catalog.assets]
    lamp = catalog.assets.get(LAMP)
    trees = _trees(catalog)
    keep = front + 8

    def park(x: float, y: float, heading: float) -> None:
        if not cars:
            return
        if abs(x) < keep and abs(y) < keep:
            return
        car = cars[rng.randrange(len(cars))]
        yaw = yaw_for_long_axis(car.size_m, heading)
        box = _box(car, x, y, z_road, yaw)
        on_road = any(_inside_rect(box, rect, 0.05) for rect in zone["roads"])
        if not on_road or _hits(box, solids, 0.3):
            return
        doc.add(car, x, y, z_road, yaw_deg=math.degrees(yaw) % 360, anchor="bottom_center")
        solids.append((box[0], box[1], box[3], box[4]))

    curb = max(1.4, hr - 1.7)
    x = -hx + 10
    step = 0
    while x < hx - 10:
        if abs(x) > keep:
            park(x, -curb, 0.0 if step % 2 == 0 else math.pi)
            park(x + 7, curb, math.pi if step % 2 == 0 else 0.0)
        x += 18
        step += 1
    y = -hy + 10
    step = 0
    while y < hy - 10:
        if abs(y) > keep:
            park(-curb, y, math.pi / 2 if step % 2 == 0 else -math.pi / 2)
            park(curb, y + 7, -math.pi / 2 if step % 2 == 0 else math.pi / 2)
        y += 18
        step += 1

    if lamp is not None:
        x = -hx + 8
        while x < hx - 8:
            if abs(x) > front + 1:
                for side in (-1, 1):
                    lx, ly = x, side * (hr + 1.3)
                    box = _box(lamp, lx, ly, z_walk, 0.0)
                    if not _hits(box, solids, 0.3):
                        doc.add(lamp, lx, ly, z_walk, anchor="bottom_center")
                        solids.append((box[0], box[1], box[3], box[4]))
            x += 26
        y = -hy + 8
        while y < hy - 8:
            if abs(y) > front + lot:
                for side in (-1, 1):
                    lx, ly = side * (hr + 1.3), y
                    box = _box(lamp, lx, ly, z_walk, math.pi / 2)
                    if not _hits(box, solids, 0.3):
                        doc.add(lamp, lx, ly, z_walk, yaw_deg=90, anchor="bottom_center")
                        solids.append((box[0], box[1], box[3], box[4]))
            y += 26

    if trees:
        corners = (
            (-(front + 6), -(front + 6)),
            (front + 6, -(front + 6)),
            (-(front + 6), front + 6),
            (front + 6, front + 6),
        )
        for tx, ty in corners:
            tree = trees[rng.randrange(len(trees))]
            box = _box(tree, tx, ty, z_lot, 0.0)
            if _hits(box, solids, 1.0) or _hits(box, reserved, 0.2):
                continue
            doc.add(tree, tx, ty, z_lot, yaw_deg=rng.uniform(0, 360), anchor="bottom_center")
            solids.append((box[0], box[1], box[3], box[4]))

    doc.set_spawn(0, 0, z_road + 1.35, 0)
    if frontage:
        bx, facade_y = frontage[len(frontage) // 2]
    else:
        bx, facade_y = 0.0, -edge
    # In the roadway, looking straight at a south facade.
    doc.set_camera(bx, min(-1.5, facade_y + 16), 2.4, -90, -2)
    doc.set_weather(16.2, 0.28, 0)
    doc.track = False
    return {
        "style": "downtown",
        "blocks": [blocks_x, blocks_y],
        "block_m": block_m,
        "road_m": road_m,
        "instances": doc.instance_count(),
        "spawn_m": [0.0, 0.0, round(z_road + 1.35, 2)],
        "size_m": [round(2 * hx, 1), round(2 * hy, 1)],
    }


def build_city(
    doc: MapFile,
    catalog: Catalog,
    style: str = "downtown",
    blocks_x: int = 3,
    blocks_y: int = 3,
    block_m: float = 44.0,
    road_m: float = 14.0,
    seed: int = 1,
    ground_z: float = 0.0,
) -> dict:
    """Replace the map with streets and buildings. Centred on the origin.

    Downtown is one crossing lined with the game's building meshes at natural size.
    Suburb and industrial keep a block grid.
    """
    styles = {"downtown", "suburb", "industrial"}
    if style not in styles:
        raise ValueError(f"style must be one of {sorted(styles)}")
    if not (1 <= blocks_x <= 8 and 1 <= blocks_y <= 8):
        raise ValueError("blocks_x and blocks_y must be from 1 to 8")
    if not (20 <= block_m <= 80 and 8 <= road_m <= 24):
        raise ValueError("block_m must be 20–80 and road_m 8–24")
    if style == "downtown":
        doc.clear_instances()
        info = _build_street(doc, catalog, blocks_x, blocks_y, block_m, road_m, seed, ground_z)
        if doc.instance_count() > MAX_INSTANCES:
            raise RuntimeError(f"city has {doc.instance_count()} instances; lower blocks or block_m (limit {MAX_INSTANCES})")
        return info
    sidewalk = 4.0
    rng = random.Random(seed)
    doc.clear_instances()
    horizontal, vertical, blocks = _city_layout(blocks_x, blocks_y, block_m, road_m)
    z_road = _layer(catalog, "asphalt", ground_z)
    for rect in horizontal:
        add_rect(doc, catalog, "asphalt", rect, z_road)
    for rect in vertical:
        add_rect(doc, catalog, "asphalt", rect, z_road + 0.004)
    plaza_index = (len(blocks) // 2) if (blocks_x % 2 and blocks_y % 2) else 0
    spawn = (0.0, 0.0, ground_z + 1.3)
    for index, rect in enumerate(blocks):
        if index == plaza_index:
            spawn = _plaza(doc, catalog, rect, ground_z, rng)
            continue
        _fill_block(doc, catalog, rect, style, ground_z, sidewalk, rng)
    spacing = {"downtown": 18.0, "suburb": 34.0, "industrial": 26.0}[style]
    for rect in horizontal:
        _cars_on_rect(doc, catalog, rect, True, z_road, spacing, road_m + 4, rng)
    for rect in vertical:
        _cars_on_rect(doc, catalog, rect, False, z_road, spacing, 6, rng)
    if doc.instance_count() > MAX_INSTANCES:
        raise RuntimeError(f"city has {doc.instance_count()} instances; lower blocks or block_m (limit {MAX_INSTANCES})")
    sx, sy, sz = spawn
    doc.set_spawn(sx, sy, sz, 90)
    total_y = blocks_y * block_m + (blocks_y + 1) * road_m
    doc.set_camera(0, -total_y / 2 - 30, max(36, total_y * 0.42), 90, -34)
    doc.set_weather(16.5 if style != "industrial" else 15.0, 0.35 if style != "industrial" else 0.55, 0)
    doc.track = False
    return {
        "style": style,
        "blocks": [blocks_x, blocks_y],
        "block_m": block_m,
        "road_m": road_m,
        "instances": doc.instance_count(),
        "spawn_m": [round(sx, 2), round(sy, 2), round(sz, 2)],
        "size_m": [
            round(blocks_x * block_m + (blocks_x + 1) * road_m, 1),
            round(total_y, 1),
        ],
    }



# One crossing. Four blocks, each filled with one family of meshes:
# houses, a concrete hall, a container yard, one office building.
_TILE = 4.0
_SIDEWALK = 4.0

# (x, y, w, h) metres. The two roads cross at the origin.
DISTRICT_ROADS: tuple[Rect, ...] = (
    (-80.0, -8.0, 160.0, 16.0),
    (-8.0, -80.0, 16.0, 160.0),
)

# Houses and workshops of one height. The deep and mid-rise meshes are left out
# so a row reads as a street instead of a catalogue.
_HOUSES = (
    "NewRow_228",
    "NewRow_788",
    "NewRow_789",
    "NewRow_790",
    "NewRow_95",
    "NewRow_791",
    "NewRow_134",
    "NewRow_793",
    "NewRow_9",
    "NewRow_97",
)


def _rect_has(px: float, py: float, rect: Rect) -> bool:
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h


def _dist_rect(px: float, py: float, rect: Rect) -> float:
    x, y, w, h = rect
    dx = 0.0 if x <= px <= x + w else min(abs(px - x), abs(px - (x + w)))
    dy = 0.0 if y <= py <= y + h else min(abs(py - y), abs(py - (y + h)))
    return math.hypot(dx, dy)


def build_district(
    doc: MapFile,
    catalog: Catalog,
    blocks: int = 4,
    block_m: float = 64.0,
    road_m: float = 16.0,
    seed: int = 4,
    ground_z: float = 0.0,
) -> dict:
    """A filled crossing: house rows, a concrete hall, a container yard, one office.

    `blocks`, `block_m` and `road_m` are accepted so older calls still run.
    Every prop sits on the surface under it. Slabs sit on the columns that hold them.
    """
    del blocks, block_m, road_m, seed
    doc.clear_instances()
    z_road = _layer(catalog, "asphalt", ground_z)
    z_walk = _layer(catalog, "sidewalk", ground_z)
    z_yard = _layer(catalog, "gravel", ground_z)
    roads = DISTRICT_ROADS
    road_boxes = [(x, y, x + w, y + h) for x, y, w, h in roads]
    xmin, ymin, width, height = -80.0, -80.0, 160.0, 160.0
    nx, ny = int(round(width / _TILE)), int(round(height / _TILE))

    def surface(px: float, py: float) -> tuple[str, float]:
        # Gravel only where the hall and the yard stand. The other two blocks are pavement.
        if px <= -12.0 and py >= 12.0:
            return "gravel", z_yard
        if px >= 12.0 and py <= -12.0:
            return "gravel", z_yard
        return "pavement", z_walk

    for ix in range(nx):
        for iy in range(ny):
            cx = xmin + _TILE / 2 + ix * _TILE
            cy = ymin + _TILE / 2 + iy * _TILE
            if any(_rect_has(cx, cy, rect) for rect in roads):
                kind, top = "asphalt", z_road
            elif any(_dist_rect(cx, cy, rect) <= _SIDEWALK + 0.01 for rect in roads):
                kind, top = "pavement", z_walk
            else:
                kind, top = surface(cx, cy)
            add_plane(doc, catalog, _ground(catalog, kind), cx, cy, _TILE, _TILE, top)

    footprints: list[tuple[float, float, float, float]] = []

    def hits_road(box, pad: float = 0.3) -> bool:
        return _hits(box, road_boxes, pad)

    def try_add(asset: Asset, x: float, y: float, z: float, yaw_deg: float = 0.0, scale: tuple[float, float, float] = (1, 1, 1), gap: float = 0.6, keep_off_roads: bool = True):
        yaw = math.radians(yaw_deg)
        box = _box(asset, x, y, z, yaw, scale)
        if keep_off_roads and hits_road(box, 0.4):
            return None
        if _hits(box, footprints, gap):
            return None
        doc.add(asset, x, y, z, yaw_deg=yaw_deg % 360, scale=scale, anchor="bottom_center")
        footprints.append((box[0], box[1], box[3], box[4]))
        return box

    def put(row: str, x: float, y: float, z: float, yaw_deg: float = 0.0, scale: tuple[float, float, float] = (1, 1, 1), gap: float = 0.6, keep_off_roads: bool = True):
        asset = catalog.assets.get(row)
        if asset is None:
            return None
        return try_add(asset, x, y, z, yaw_deg, scale, gap, keep_off_roads)

    # South-west block: house rows filling the block, fronts toward a 4 m alley.
    def pack_houses(y_line: float, face: float, x0: float, x1: float, y_limit: float, offset: int) -> float | None:
        cursor = x0
        far = y_line
        placed = False
        guard = 0
        while cursor < x1 - 6 and guard < 30:
            guard += 1
            row = _HOUSES[(offset + guard) % len(_HOUSES)]
            asset = catalog.assets.get(row)
            if asset is None:
                continue
            yaw = _yaw_entrance(ENTRANCE.get(row, "+y"), face)
            rel = _box(asset, 0.0, 0.0, z_walk, yaw)
            depth = rel[4] - rel[1]
            width_m = rel[3] - rel[0]
            if cursor + width_m > x1 or y_line - depth < y_limit:
                continue
            place_y = y_line - rel[4]
            box = try_add(asset, cursor - rel[0], place_y, z_walk, math.degrees(yaw), gap=0.4)
            if box is None:
                cursor += 4.0
                continue
            cursor = box[3] + 0.4
            far = min(box[1], far) if face > 0 else max(box[4], far)
            placed = True
        return far if placed else None

    y_line = -14.0
    for row_index in range(6):
        far = pack_houses(y_line, math.pi / 2, -74.0, -16.0, -76.0, row_index * 3)
        if far is None:
            break
        y_line = far - 3.5
        if y_line < -70.0:
            break

    # North-west block: one concrete hall. 8 m bays. The slab bottom is the column top.
    column = catalog.get("NewRow_122")
    floor = catalog.get(FLOOR)
    wall = catalog.get("NewRow_318")
    scaffold = catalog.get("NewRow_157")
    xs = [-72.0 + 8.0 * i for i in range(8)]
    ys = [16.0 + 8.0 * j for j in range(8)]
    for j, y in enumerate(ys):
        for x in xs:
            doc.add(column, x, y, z_yard, anchor="bottom_center")
            if j >= 3:
                doc.add(column, x, y, z_yard + 4.0, anchor="bottom_center")
    for j in range(7):
        for i in range(7):
            if i == 3 and j == 1:
                continue
            level = 8.0 if j >= 3 else 4.0
            cx = (xs[i] + xs[i + 1]) / 2
            cy = (ys[j] + ys[j + 1]) / 2
            doc.add(floor, cx, cy, z_yard + level + 0.3, scale=(2.0, 2.0, 1.0), anchor="top_center")
    for j in range(7):
        cy = (ys[j] + ys[j + 1]) / 2
        tall = j >= 3
        for edge_x in (xs[0], xs[-1]):
            if edge_x == xs[-1] and j in (0, 1):
                continue
            doc.add(wall, edge_x, cy, z_yard, scale=(1.0, 2.0, 1.0), anchor="bottom_center")
            if tall:
                doc.add(wall, edge_x, cy, z_yard + 4.0, scale=(1.0, 2.0, 1.0), anchor="bottom_center")
        if tall:
            doc.add(scaffold, xs[0] - 2.0, cy, z_yard, anchor="bottom_center")
            doc.add(scaffold, xs[0] - 2.0, cy, z_yard + 4.0, anchor="bottom_center")
        else:
            doc.add(scaffold, xs[0] - 2.0, cy, z_yard, anchor="bottom_center")
    for i in range(7):
        cx = (xs[i] + xs[i + 1]) / 2
        doc.add(wall, cx, ys[-1], z_yard, yaw_deg=90, scale=(1.0, 2.0, 1.0), anchor="bottom_center")
        doc.add(wall, cx, ys[-1], z_yard + 4.0, yaw_deg=90, scale=(1.0, 2.0, 1.0), anchor="bottom_center")
    slot_cx = (xs[3] + xs[4]) / 2
    slot_cy = (ys[1] + ys[2]) / 2
    # Goods along the side bays. The centre aisle and the south entrance stay open.
    stored = catalog.get("NewRow_1")
    for cy in (56.0, 66.0):
        doc.add(stored, -60.0, cy, z_yard, anchor="bottom_center")
        doc.add(stored, -36.0, cy, z_yard, anchor="bottom_center")

    # South-east block: containers in rows, then a silo cluster. Same family, no gaps big enough to fly a block through.
    containers = ("NewRow_1", "NewRow_26", "NewRow_27", "NewRow_28")
    for row_i, cy in enumerate(-66.0 + 3.2 * n for n in range(8)):
        for col_i, cx in enumerate(24.0 + 12.6 * n for n in range(4)):
            asset = catalog.assets.get(containers[(row_i + col_i) % 4])
            if asset is not None:
                doc.add(asset, cx, cy, z_yard, yaw_deg=90, anchor="bottom_center")
    put("NewRow_322", 24.0, -28.0, z_yard, gap=0.4)
    put("NewRow_322", 34.0, -30.0, z_yard, gap=0.4)
    put("NewRow_251", 30.0, -22.0, z_yard, gap=0.2)
    put("NewRow_251", 38.0, -22.0, z_yard, gap=0.2)
    put("NewRow_49", 50.0, -26.0, z_yard, gap=0.4)
    put("NewRow_48", 58.0, -24.0, z_yard, yaw_deg=90, gap=0.4)
    put("NewRow_201", 68.0, -28.0, z_yard, yaw_deg=-20, gap=0.2)
    put("NewRow_196", 46.0, -36.0, z_yard, yaw_deg=20, gap=0.2)
    for x in range(16, 72, 8):
        if 36 <= x <= 52:
            continue
        doc.add(wall, x + 4.0, -14.0, z_yard, yaw_deg=90, scale=(1.0, 2.0, 1.0), anchor="bottom_center")
    for y in range(-72, -16, 8):
        doc.add(wall, 14.0, y + 4.0, z_yard, scale=(1.0, 2.0, 1.0), anchor="bottom_center")

    # North-east block: one office, filling the block. Shops would not fit beside it.
    put("NewRow_11", 46.0, 48.0, z_walk, gap=1.0)
    bush = catalog.assets.get("NewRow_164")
    if bush is not None:
        for x in range(28, 68, 4):
            doc.add(bush, float(x), 18.0, z_walk, yaw_deg=(x * 17) % 360, anchor="bottom_center")

    # Compound wall on the road height so it meets the asphalt and closes the board.
    for index in range(20):
        along = -76.0 + index * 8.0
        for level in range(3):
            z = z_road + level * 4.0
            doc.add(wall, along, 78.0, z, yaw_deg=90, scale=(1.0, 2.0, 1.0), anchor="bottom_center")
            doc.add(wall, along, -78.0, z, yaw_deg=90, scale=(1.0, 2.0, 1.0), anchor="bottom_center")
            doc.add(wall, 78.0, along, z, scale=(1.0, 2.0, 1.0), anchor="bottom_center")
            doc.add(wall, -78.0, along, z, scale=(1.0, 2.0, 1.0), anchor="bottom_center")

    lamp = catalog.assets.get(LAMP)
    if lamp is not None:
        for x in range(-64, 72, 16):
            if abs(x) < 14:
                continue
            put(lamp.id, float(x), 10.0, z_walk, gap=0.2)
            put(lamp.id, float(x), -10.0, z_walk, gap=0.2)
        for y in range(-64, 72, 16):
            if abs(y) < 14:
                continue
            put(lamp.id, 10.0, float(y), z_walk, gap=0.2)
            put(lamp.id, -10.0, float(y), z_walk, gap=0.2)

    cars = [catalog.assets[row] for row in CARS if row in catalog.assets]
    spawn = (-58.0, 0.0, z_road + 0.45)
    # Long axis along the road, sitting on the sidewalk so the centre lane stays open.
    lamp_x = {float(x) for x in range(-64, 72, 16) if abs(x) >= 14}
    if cars:
        parked = 0
        for x in (-54.0, -36.0, -18.0, 22.0, 40.0, 58.0):
            if any(abs(x - lamp) < 3.5 for lamp in lamp_x):
                continue
            for side in (1.0, -1.0):
                asset = cars[parked % len(cars)]
                face_east = side > 0
                long_on_y = asset.size_m[1] > asset.size_m[0]
                yaw = (90.0 if face_east else -90.0) if long_on_y else (0.0 if face_east else 180.0)
                rel = _box(asset, 0.0, 0.0, z_walk, math.radians(yaw))
                half = (rel[4] - rel[1]) / 2
                put(asset.id, x, side * (8.6 + half), z_walk, yaw_deg=yaw, gap=0.2, keep_off_roads=False)
                parked += 1

    # Barriers channel the turn into the hall. The opening around x=-44 stays empty.
    barrier = catalog.assets.get("NewRow_100")
    cone = catalog.assets.get("NewRow_202")
    if barrier is not None:
        for x in (-68.0, -66.0, -64.0, -62.0, -60.0, -58.0, -36.0, -34.0, -32.0, -30.0, -28.0, -26.0):
            doc.add(barrier, x, 7.2, z_road, anchor="bottom_center")
    if cone is not None:
        for x, y in ((-56.5, 7.2), (-37.5, 7.2)):
            doc.add(cone, x, y, z_road, anchor="bottom_center")
    pallet = catalog.assets.get("NewRow_54")
    if pallet is not None:
        doc.add(pallet, -66.0, 18.0, z_yard, yaw_deg=15, anchor="bottom_center")

    doc.set_spawn(*spawn, 0)
    lap = [
        {"x": -40.0, "y": 0.0, "z": z_road + 1.35, "yaw": 0, "gate": "DefaultOneW"},
        {"x": slot_cx, "y": 22.0, "z": z_yard + 1.4, "yaw": 90, "gate": "SideNeon"},
        {"x": slot_cx, "y": 44.0, "z": z_yard + 1.4, "yaw": 90, "gate": "DefaultOneW"},
        {"x": -20.0, "y": 44.0, "z": z_yard + 1.4, "yaw": 0, "gate": "DefaultOneW"},
        {"x": -20.0, "y": 20.0, "z": z_yard + 1.4, "yaw": -90, "gate": "DefaultOneW"},
        {"x": 0.0, "y": 20.0, "z": z_road + 1.35, "yaw": 0, "gate": "DefaultOneW"},
        {"x": 0.0, "y": 0.0, "z": z_road + 1.35, "yaw": -90, "gate": "SideNeon"},
        {"x": 40.0, "y": 0.0, "z": z_road + 1.35, "yaw": 180, "gate": "InflSimpl01"},
    ]
    doc.set_checkpoints(lap, laps=2)
    doc.set_camera(0.0, -26.0, 7.0, 90, -16)
    doc.set_weather(17.5, 0.42, 18)
    if doc.instance_count() > MAX_INSTANCES:
        raise RuntimeError(f"district has {doc.instance_count()} instances; limit is {MAX_INSTANCES}")
    return {
        "style": "highline",
        "blocks": [2, 2],
        "block_m": 72.0,
        "road_m": 16.0,
        "instances": doc.instance_count(),
        "spawn_m": [round(v, 2) for v in spawn],
        "shaft_m": [round(slot_cx, 2), round(slot_cy, 2), round(z_yard + 6.0, 2)],
        "size_m": [round(width, 1), round(height, 1)],
        "checkpoints": len(lap),
        "skyline_top_m": round(z_walk + 86.1, 1),
    }
