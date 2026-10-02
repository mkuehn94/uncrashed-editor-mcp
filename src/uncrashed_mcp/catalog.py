"""Asset catalogue measured from the game's DT_MapEditorAssets and static-mesh bounds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

# These rows share one mesh and swap the material. The table stores them as "Base Ground".
GROUND_LABELS = {
    "NewRow_182": "Asphalt",
    "NewRow_185": "Pavement 03",
    "NewRow_197": "Pavement 01",
    "NewRow_186": "Pavement 02",
    "NewRow_376": "Flagstone",
    "NewRow_180": "Dirt",
    "NewRow_181": "Gravel",
    "NewRow_662": "Sand",
    "NewRow_184": "Wood planks",
}


@dataclass(frozen=True)
class Asset:
    id: str
    name: str
    category: str
    kind: str
    tags: tuple[str, ...]
    hidden: bool
    size_m: tuple[float, float, float]
    pivot: str
    triplanar: bool
    collision: bool
    material: str | None
    source: str | None
    origin_cm: tuple[float, float, float]
    extent_cm: tuple[float, float, float]
    bounds_estimated: bool

    @property
    def long_axis(self) -> str:
        return "y" if self.size_m[1] > self.size_m[0] else "x"

    def public(self, full: bool = False) -> dict:
        out = {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "kind": self.kind,
            "tags": list(self.tags),
            "hidden": self.hidden,
            "size_m": list(self.size_m),
            "long_axis": self.long_axis,
            "pivot": self.pivot,
            "triplanar": self.triplanar,
            "material": self.material,
        }
        if full:
            out.update(
                {
                    "collision": self.collision,
                    "source": self.source,
                    "origin_cm": list(self.origin_cm),
                    "extent_cm": list(self.extent_cm),
                    "bounds_estimated": self.bounds_estimated,
                    "placement": (
                        "anchor bottom_center puts the bounding-box centre on x,y and the bottom on z. "
                        "The actor pivot is often a corner or an end; do not place offset meshes by the pivot."
                    ),
                }
            )
        return out


class Catalog:
    def __init__(self, assets: dict[str, Asset], library: dict):
        self.assets = assets
        self.library = library

    def get(self, row: str) -> Asset:
        try:
            return self.assets[row]
        except KeyError as exc:
            raise KeyError(f"unknown asset {row!r}") from exc

    def search(
        self,
        query: str = "",
        category: str | None = None,
        tag: str | None = None,
        limit: int = 20,
        include_hidden: bool = True,
        include_decals: bool = False,
    ) -> list[Asset]:
        tokens = [t for t in query.lower().split() if t]
        hits: list[tuple[int, Asset]] = []
        for asset in self.assets.values():
            if category and asset.category.lower() != category.lower():
                continue
            if tag and tag.lower() not in asset.tags:
                continue
            if not include_hidden and asset.hidden:
                continue
            if asset.category == "Decals" and not include_decals and (not category or category.lower() != "decals"):
                continue
            hay = " ".join(
                [
                    asset.id,
                    asset.name,
                    asset.category,
                    asset.material or "",
                    asset.source or "",
                    " ".join(asset.tags),
                ]
            ).lower()
            if tokens and not all(t in hay for t in tokens):
                continue
            score = 0
            if query and asset.id.lower() == query.lower():
                score += 100
            if tokens and asset.name.lower().startswith(tokens[0]):
                score += 20
            if not asset.hidden:
                score += 5
            if not asset.bounds_estimated:
                score += 1
            hits.append((score, asset))
        hits.sort(key=lambda item: (-item[0], item[1].name, item[1].id))
        return [asset for _, asset in hits[:limit]]


def _tup3(values) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


@lru_cache(maxsize=1)
def load_catalog() -> Catalog:
    raw = json.loads((DATA / "assets.json").read_text())
    assets: dict[str, Asset] = {}
    for row in raw["assets"]:
        name = GROUND_LABELS.get(row["id"], row["name"])
        category = "Ground" if row["id"] in GROUND_LABELS else (row.get("category") or "")
        if category == "Decals" and row.get("material"):
            # Every decal row is the same blueprint. The material is the only thing that differs.
            name = row["material"]
        asset = Asset(
            id=row["id"],
            name=name,
            category=category,
            kind=row.get("kind") or "",
            tags=tuple(row.get("tags") or []),
            hidden=bool(row.get("hidden")),
            size_m=_tup3(row["size_m"]),
            pivot=row.get("pivot") or "centre",
            triplanar=bool(row.get("triplanar")),
            collision=bool(row.get("collision")),
            material=row.get("material"),
            source=row.get("source"),
            origin_cm=_tup3(row["origin_cm"]),
            extent_cm=_tup3(row["extent_cm"]),
            bounds_estimated=bool(row.get("bounds_estimated")),
        )
        assets[asset.id] = asset
    library = json.loads((DATA / "library.json").read_text())
    return Catalog(assets, library)
