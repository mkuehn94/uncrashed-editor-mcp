"""Top-down JPEG the track editor shows on the map tile."""

from __future__ import annotations

from pathlib import Path

COLORS = {
    "ground": (70, 74, 78),
    "asphalt": (48, 50, 54),
    "pavement": (150, 148, 140),
    "dirt": (120, 96, 64),
    "gravel": (110, 108, 100),
    "flagstone": (168, 164, 150),
    "foliage": (46, 120, 58),
    "vehicle": (176, 48, 42),
    "building": (92, 104, 118),
    "roof": (186, 190, 194),
    "wall": (74, 82, 92),
    "spawn": (240, 208, 64),
}


def _color(item: dict) -> tuple[int, int, int]:
    name = (item.get("name") or "").lower()
    category = (item.get("category") or "").lower()
    if "asphalt" in name:
        return COLORS["asphalt"]
    if "dirt" in name:
        return COLORS["dirt"]
    if "gravel" in name:
        return COLORS["gravel"]
    if "flagstone" in name:
        return COLORS["flagstone"]
    if "pavement" in name or "wood" in name:
        return COLORS["pavement"]
    if category == "foliage":
        return COLORS["foliage"]
    if "car" in name or "van" in name or "kangoo" in name or "camaro" in name:
        return COLORS["vehicle"]
    if "floor" in name:
        return COLORS["roof"]
    if category == "wall":
        return COLORS["wall"]
    if category in ("big structure", "big_assets"):
        return COLORS["building"]
    return (130, 136, 142)


def write_thumbnail(path: Path, map_file, catalog, size: int = 640) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    placed = map_file.iter_placed(catalog)
    boxes = [item for item in placed if item["aabb_m"]]
    if not boxes:
        image = Image.new("RGB", (size, size), (32, 36, 40))
        image.save(path, quality=85)
        return
    minx = min(item["aabb_m"][0] for item in boxes)
    miny = min(item["aabb_m"][1] for item in boxes)
    maxx = max(item["aabb_m"][3] for item in boxes)
    maxy = max(item["aabb_m"][4] for item in boxes)
    span = max(maxx - minx, maxy - miny, 1.0)
    pad = span * 0.06
    minx -= pad
    miny -= pad
    span += pad * 2
    image = Image.new("RGB", (size, size), (28, 32, 36))
    draw = ImageDraw.Draw(image)

    def xy(x: float, y: float) -> tuple[float, float]:
        px = (x - minx) / span * (size - 1)
        py = (1 - (y - miny) / span) * (size - 1)
        return px, py

    ordered = sorted(boxes, key=lambda item: (item["aabb_m"][3] - item["aabb_m"][0]) * (item["aabb_m"][4] - item["aabb_m"][1]), reverse=True)
    for item in ordered:
        x0, y0, _, x1, y1, _ = item["aabb_m"]
        ax, ay = xy(x0, y0)
        bx, by = xy(x1, y1)
        draw.rectangle([min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)], fill=_color(item))
    spawn = map_file.race.get("StartTransform", {}).get("Translation", {})
    if spawn:
        sx, sy = xy(float(spawn["X"]) / 100.0, float(spawn["Y"]) / 100.0)
        r = 5
        draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=COLORS["spawn"])
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=85)
