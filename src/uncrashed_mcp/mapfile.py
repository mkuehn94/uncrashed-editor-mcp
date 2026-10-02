"""Read and write a track-editor map folder."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import Asset
from .geom import pivot_for_anchor, quat_pitch_yaw, quat_yaw, world_aabb, yaw_from_quat
from .paths import maps_dir, steam_id
from .thumb import write_thumbnail

BOARD_M = 3000.0


def _transform(pivot, yaw: float, scale) -> dict:
    return {
        "Rotation": quat_yaw(yaw),
        "Translation": {
            "X": round(pivot[0] * 100, 2),
            "Y": round(pivot[1] * 100, 2),
            "Z": round(pivot[2] * 100, 2),
        },
        "Scale3D": {
            "X": round(scale[0], 4),
            "Y": round(scale[1], 4),
            "Z": round(scale[2], 4),
        },
    }


def _cam(x: float, y: float, z: float, yaw_deg: float, pitch_deg: float) -> dict:
    return {
        "Rotation": quat_pitch_yaw(pitch_deg, yaw_deg),
        "Translation": {"X": round(x * 100, 2), "Y": round(y * 100, 2), "Z": round(z * 100, 2)},
        "Scale3D": {"X": 1, "Y": 1, "Z": 1},
    }


def _start(x: float, y: float, z: float, yaw_deg: float) -> dict:
    yaw = math.radians(yaw_deg)
    return {
        "Rotation": quat_yaw(yaw),
        "Translation": {"X": round(x * 100, 2), "Y": round(y * 100, 2), "Z": round(z * 100, 2)},
        "Scale3D": {"X": 1, "Y": 1, "Z": 1},
    }


@dataclass
class Group:
    rname: str
    transforms: list[dict] = field(default_factory=list)
    parallel: dict[str, list] = field(default_factory=dict)


@dataclass
class MapFile:
    folder_name: str
    display_name: str
    base: str = "BaseBoard"
    weather_index: int = 0
    weather: dict = field(default_factory=lambda: {"timeOfDay": 16.0, "northyaw": 0.0, "Cloud": 0.35})
    race: dict = field(default_factory=dict)
    cam: dict = field(default_factory=dict)
    track: bool = False
    created: int = 0
    groups: dict[str, Group] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    root: Path | None = None

    def __post_init__(self) -> None:
        if not self.race:
            self.race = {
                "CheckPoints": [],
                "StartTransform": _start(0, 0, 1.2, 90),
                "Laps": 1,
                "BestTime": -1,
                "Spoints": [],
            }
        if not self.cam:
            self.cam = _cam(0, -80, 40, 90, -35)
        if not self.created:
            self.created = int(time.time())

    @property
    def directory(self) -> Path:
        return (self.root or maps_dir()) / self.folder_name

    def clear_instances(self) -> None:
        self.groups.clear()
        self.order.clear()

    def instance_count(self) -> int:
        return sum(len(group.transforms) for group in self.groups.values())

    def add(
        self,
        asset: Asset,
        x: float,
        y: float,
        z: float,
        yaw_deg: float = 0.0,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        anchor: str = "bottom_center",
    ) -> None:
        if any(component <= 0 for component in scale):
            raise ValueError(f"scale must be positive, got {scale}")
        yaw = math.radians(yaw_deg)
        pivot = pivot_for_anchor(
            asset.origin_cm, asset.extent_cm, x, y, z, yaw, scale, anchor
        )
        group = self.groups.get(asset.id)
        if group is None:
            group = Group(asset.id, [], {"UniqueGroupTag": []})
            self.groups[asset.id] = group
            self.order.append(asset.id)
        group.transforms.append(_transform(pivot, yaw, scale))
        for key, values in group.parallel.items():
            if isinstance(values, list) and len(values) == len(group.transforms) - 1:
                values.append("None")

    def iter_placed(self, catalog) -> list[dict]:
        """Decoded instances with world AABB in metres."""
        out = []
        for rname in self.order:
            group = self.groups[rname]
            asset = catalog.assets.get(rname)
            for transform in group.transforms:
                scale = (
                    float(transform["Scale3D"]["X"]),
                    float(transform["Scale3D"]["Y"]),
                    float(transform["Scale3D"]["Z"]),
                )
                translation = transform["Translation"]
                pivot = (
                    float(translation["X"]) / 100.0,
                    float(translation["Y"]) / 100.0,
                    float(translation["Z"]) / 100.0,
                )
                yaw = yaw_from_quat(transform["Rotation"])
                aabb = None
                if asset is not None:
                    aabb = world_aabb(asset.origin_cm, asset.extent_cm, pivot, yaw, scale)
                out.append(
                    {
                        "id": rname,
                        "name": asset.name if asset else rname,
                        "category": asset.category if asset else "",
                        "pivot_m": pivot,
                        "yaw_deg": round(math.degrees(yaw), 2),
                        "scale": scale,
                        "aabb_m": aabb,
                    }
                )
        return out

    def remove(
        self,
        asset_id: str | None = None,
        x0: float | None = None,
        y0: float | None = None,
        x1: float | None = None,
        y1: float | None = None,
    ) -> int:
        removed = 0
        for rname in list(self.order):
            if asset_id and rname != asset_id:
                continue
            group = self.groups[rname]
            keep_tf = []
            keep_parallel = {key: [] for key in group.parallel}
            for index, transform in enumerate(group.transforms):
                x = float(transform["Translation"]["X"]) / 100.0
                y = float(transform["Translation"]["Y"]) / 100.0
                inside = True
                if x0 is not None and x < x0:
                    inside = False
                if y0 is not None and y < y0:
                    inside = False
                if x1 is not None and x > x1:
                    inside = False
                if y1 is not None and y > y1:
                    inside = False
                # A region filter with no asset removes pivots inside the rect.
                # No filter at all would wipe the map; require at least one bound or an asset.
                drop = inside if (x0 is not None or y0 is not None or x1 is not None or y1 is not None) else True
                if asset_id and x0 is None and y0 is None and x1 is None and y1 is None:
                    drop = True
                if drop:
                    removed += 1
                    continue
                keep_tf.append(transform)
                for key, values in group.parallel.items():
                    if isinstance(values, list) and index < len(values):
                        keep_parallel[key].append(values[index])
            group.transforms = keep_tf
            group.parallel = keep_parallel
            if not group.transforms:
                del self.groups[rname]
                self.order.remove(rname)
        return removed

    def set_spawn(self, x: float, y: float, z: float, yaw_deg: float) -> None:
        self.race["StartTransform"] = _start(x, y, z, yaw_deg)

    def set_checkpoints(self, gates: list[dict], laps: int = 1) -> None:
        points = []
        for gate in gates:
            yaw = math.radians(float(gate.get("yaw", 0)))
            points.append(
                {
                    "Transform": _transform(
                        (float(gate["x"]), float(gate["y"]), float(gate["z"])),
                        yaw,
                        (1, 1, 1),
                    ),
                    "RowName": gate.get("gate", "DefaultOneW"),
                }
            )
        self.race["CheckPoints"] = points
        self.race["Laps"] = int(laps)
        self.track = bool(points)

    def set_camera(self, x: float, y: float, z: float, yaw_deg: float, pitch_deg: float) -> None:
        self.cam = _cam(x, y, z, yaw_deg, pitch_deg)

    def set_weather(self, time_of_day: float, clouds: float, north_yaw: float = 0.0) -> None:
        self.weather = {
            "timeOfDay": float(time_of_day),
            "northyaw": float(north_yaw),
            "Cloud": float(clouds),
        }

    def to_dict(self) -> dict:
        assets = []
        for rname in self.order:
            group = self.groups[rname]
            if not group.transforms:
                continue
            entry: dict = {"Rname": rname, "Transform": group.transforms}
            tags = group.parallel.get("UniqueGroupTag")
            if not isinstance(tags, list) or len(tags) != len(group.transforms):
                tags = ["None"] * len(group.transforms)
            entry["UniqueGroupTag"] = tags
            for key, values in group.parallel.items():
                if key == "UniqueGroupTag":
                    continue
                if isinstance(values, list) and len(values) == len(group.transforms):
                    entry[key] = values
            assets.append(entry)
        return {
            "S_AssetsTransform": assets,
            "BaseMapRow": self.base,
            "WeatherIndex": self.weather_index,
            "Race": self.race,
            "CamLoc": self.cam,
            "CustomWeather": self.weather,
        }

    def save(self, catalog=None) -> Path:
        folder = self.directory
        folder.mkdir(parents=True, exist_ok=True)
        self.created = int(time.time())
        payload = self.to_dict()
        text = json.dumps(payload, separators=(",", ":"))
        json_path = folder / f"{self.folder_name}.json"
        tmp = json_path.with_suffix(".json.tmp")
        tmp.write_text(text)
        tmp.replace(json_path)
        info = {
            "BaseMapRow": self.base,
            "Name": self.display_name,
            "CreatorSteamID": steam_id(),
            "CreationTimeUnix": self.created,
            "Track": bool(self.track),
        }
        (folder / f"{self.folder_name}.info").write_text(json.dumps(info))
        if catalog is not None:
            write_thumbnail(folder / "Thumbnail.jpg", self, catalog)
        return json_path

    def validate(self, catalog) -> dict:
        problems: list[str] = []
        placed = self.iter_placed(catalog)
        if not placed:
            problems.append("map has no instances")
        spawn = self.race.get("StartTransform", {}).get("Translation", {})
        sx = float(spawn.get("X", 0)) / 100.0
        sy = float(spawn.get("Y", 0)) / 100.0
        sz = float(spawn.get("Z", 0)) / 100.0
        blocked = 0
        unknown = 0
        for item in placed:
            if item["id"] not in catalog.assets:
                unknown += 1
            aabb = item["aabb_m"]
            if aabb is None:
                continue
            if max(abs(aabb[0]), abs(aabb[3]), abs(aabb[1]), abs(aabb[4])) > BOARD_M:
                problems.append(f"{item['id']} is outside the {BOARD_M:.0f} m board")
            for component in item["scale"]:
                if component <= 0 or component > 80:
                    problems.append(f"{item['id']} has extreme scale {item['scale']}")
                    break
            # Spawn capsule roughly 0.2–2.2 m above the pawn origin, 1 m radius.
            nearest_x = min(max(sx, aabb[0]), aabb[3])
            nearest_y = min(max(sy, aabb[1]), aabb[4])
            horizontal = math.hypot(nearest_x - sx, nearest_y - sy)
            vertical = aabb[2] < sz + 2.2 and aabb[5] > sz + 0.2
            if horizontal < 1.2 and vertical:
                blocked += 1
        if unknown:
            problems.append(f"{unknown} instances use rows missing from the catalogue")
        if blocked:
            problems.append(f"{blocked} instances intersect the spawn capsule")
        counts: dict[str, int] = {}
        for item in placed:
            counts[item["id"]] = counts.get(item["id"], 0) + 1
        return {
            "ok": not problems,
            "problems": problems[:30],
            "instances": len(placed),
            "assets": len(counts),
            "spawn_m": [round(sx, 2), round(sy, 2), round(sz, 2)],
        }

    def summary(self, catalog) -> dict:
        placed = self.iter_placed(catalog)
        by_cat: dict[str, int] = {}
        minx = miny = minz = 0.0
        maxx = maxy = maxz = 0.0
        if placed:
            boxes = [item["aabb_m"] for item in placed if item["aabb_m"]]
            if boxes:
                minx = min(box[0] for box in boxes)
                miny = min(box[1] for box in boxes)
                minz = min(box[2] for box in boxes)
                maxx = max(box[3] for box in boxes)
                maxy = max(box[4] for box in boxes)
                maxz = max(box[5] for box in boxes)
        for item in placed:
            by_cat[item["category"] or "?"] = by_cat.get(item["category"] or "?", 0) + 1
        top = sorted(by_cat.items(), key=lambda pair: -pair[1])[:12]
        return {
            "folder": self.folder_name,
            "name": self.display_name,
            "base": self.base,
            "track": self.track,
            "instances": len(placed),
            "by_category": top,
            "bounds_m": [round(v, 1) for v in (minx, miny, minz, maxx, maxy, maxz)],
            "checkpoints": len(self.race.get("CheckPoints") or []),
            "laps": self.race.get("Laps", 1),
            "weather": self.weather,
            "path": str(self.directory / f"{self.folder_name}.json"),
        }


def _load_groups(rows: list) -> tuple[dict[str, Group], list[str]]:
    groups: dict[str, Group] = {}
    order: list[str] = []
    for row in rows:
        rname = row["Rname"]
        parallel = {}
        transforms = row.get("Transform") or []
        for key, value in row.items():
            if key in ("Rname", "Transform"):
                continue
            if isinstance(value, list):
                parallel[key] = list(value)
        if "UniqueGroupTag" not in parallel:
            parallel["UniqueGroupTag"] = ["None"] * len(transforms)
        group = groups.get(rname)
        if group is None:
            groups[rname] = Group(rname, list(transforms), parallel)
            order.append(rname)
        else:
            group.transforms.extend(transforms)
            for key, value in parallel.items():
                group.parallel.setdefault(key, [])
                if isinstance(value, list):
                    group.parallel[key].extend(value)
    return groups, order


def load_map(folder_name: str, root: Path | None = None) -> MapFile:
    root = root or maps_dir()
    folder = root / folder_name
    data = json.loads((folder / f"{folder_name}.json").read_text())
    info_path = folder / f"{folder_name}.info"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    groups, order = _load_groups(data.get("S_AssetsTransform") or [])
    return MapFile(
        folder_name=folder_name,
        display_name=info.get("Name") or folder_name,
        base=data.get("BaseMapRow") or info.get("BaseMapRow") or "BaseBoard",
        weather_index=int(data.get("WeatherIndex") or 0),
        weather=data.get("CustomWeather") or {"timeOfDay": 16.0, "northyaw": 0.0, "Cloud": 0.35},
        race=data.get("Race") or {},
        cam=data.get("CamLoc") or {},
        track=bool(info.get("Track")),
        created=int(info.get("CreationTimeUnix") or 0),
        groups=groups,
        order=order,
        root=root,
    )


def sanitize_name(folder_name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in folder_name).strip("_")
    if not safe:
        raise ValueError("map name is empty")
    return safe


def new_map(
    folder_name: str,
    display_name: str | None = None,
    base: str = "BaseBoard",
    root: Path | None = None,
) -> MapFile:
    safe = sanitize_name(folder_name)
    return MapFile(
        folder_name=safe,
        display_name=display_name or folder_name,
        base=base,
        root=root or maps_dir(),
    )


def list_user_maps(root: Path | None = None) -> list[dict]:
    root = root or maps_dir()
    found = []
    if not root.is_dir():
        return found
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        info = folder / f"{folder.name}.info"
        js = folder / f"{folder.name}.json"
        if not js.exists():
            continue
        meta = {}
        if info.exists():
            try:
                meta = json.loads(info.read_text())
            except json.JSONDecodeError:
                meta = {}
        found.append(
            {
                "folder": folder.name,
                "name": meta.get("Name") or folder.name,
                "base": meta.get("BaseMapRow"),
                "track": meta.get("Track"),
                "created": meta.get("CreationTimeUnix"),
                "bytes": js.stat().st_size,
            }
        )
    found.sort(key=lambda row: row.get("created") or 0, reverse=True)
    return found
