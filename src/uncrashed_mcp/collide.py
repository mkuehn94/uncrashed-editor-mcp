"""Collision and floating check for the map open in the editor.

The agent's `collide` verb asks the engine itself (KismetSystemLibrary box and line traces against the real
collision of every placed object) and writes raw facts to a text file:

  O a row box     every object tested (address, catalogue row, world bounds in cm)
  B a row box     objects without a single static mesh (blueprint buildings): bounds only
  I a b row lvl   a's oriented box, shrunk (lvl 1: by 25 cm or 35 %, lvl 2: to half), intersects b
  S a d row cls   a stands d cm deep in what is under its centre
  R a b           a rests on b (0 = landscape / base level)
  T a b ...       everything a's box grown by 15 cm touches (0 = landscape / base level)
  F a row x y bot gap side   nothing under a's footprint within 30 cm

This module turns those facts into a short list of real problems: things buried in rock, vehicles clipping,
containers and buildings inside each other, vehicles sunk into the road, and objects that are not connected to
the ground at all (balloons excepted). Touching parts, construction joints, plants and ground tiles are ignored.
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path

KINDS = {
    # key: (label, colour, what to do)
    "rock": ("Buried in rock", "#d64545", "A rock or cliff cuts through it. Move the rock or the object."),
    "vehicle": ("Vehicle clipping", "#e8772e", "A vehicle sits inside a wall, barrier or another vehicle."),
    "clash": ("Inside each other", "#9b5de5", "Containers, buildings or big props occupy the same space."),
    "prop": ("Prop inside something", "#c99a1e", "A smaller prop is deep inside another object."),
    "sunk": ("Vehicle sunk in", "#a0785a", "The vehicle stands below the surface it is on (wheels in the road)."),
    "float": ("Floating", "#2f86d6", "Not connected to the ground by anything; hangs in the air."),
    "floatside": ("Hanging beside something", "#4fa3e0", "Touches nothing, but something is within 60 cm sideways."),
    "floatlite": ("Floating light or screen", "#22a6b3", "A light, neon or screen in the air with nothing holding it."),
    "hover": ("Slightly above support", "#7fa8a3", "Less than about 1 m above what is under it, or a small wall-mounted item."),
}
ORDER = list(KINDS)

GROUND_ROWS = {"NewRow_180", "NewRow_181", "NewRow_182", "NewRow_184", "NewRow_185", "NewRow_186", "NewRow_197",
               "NewRow_376", "NewRow_377", "NewRow_662", "NewRow_663", "NewRow_183"}
WATER_ROWS = {"NewRow_183"}
GRASS_ROWS = {"NewRow_663"}
BALLOON_ROWS = {"NewRow_165", "NewRow_104", "NewRow_105", "NewRow_107"}
CONTAINER_ROWS = {"NewRow_1", "NewRow_26", "NewRow_27", "NewRow_28", "NewRow_190", "NewRow_195", "NewRow_48"}
ROCK_WORDS = ("Rock", "Cliff", "vd3ffgp")
VEHICLE_WORDS = ("Camaro", "Gt86", "Kangoo", "Old Van", "Tractor", "Boat", "Train Box Car", "Carriage Vehicle", "Fork Lift",
                 "Train Metal", "Sport Cars", "Rally Car", "Barrow", "Concrete Mixer")
KIT_WORDS = ("Concrete Column", "Concrete Floor", "Concrete Wall", "Coin", "Corner Angle", "Half Corner", "3Coin",
             "Triangle Wall", "Scafold", "Scaffolding", "Metal Construction", "Beam", "Handrail", "Steel Catwalks", "Stairs",
             "Stair A", "Roof", "MEroof", "Wall 8x8", "Wall Broken", "Chunk", "Base 01a", "Wallkit", "Pipe 01 02",
             "Storage Pipe", "Tribune Metal", "Plank", "Wall 01 1", "Wall 02 01", "parapet", "Concrete Fence",
             "concrete fence", "Echelle", "Construction Pale", "Gate 01", "Grid Net", "Fence04", "Race Track Fence",
             "Pit Concrete", "Hesco", "Chain", "Window", "Ground Concrete", "Metal Rusty Wall", "Metal Support",
             "Wood Fence", "Road Barrier", "Concrete Pipe", "Deck Floor", "Building Walls", "Race Track Tire Wall",
             "Wood Ladder", "copmensator", "compensator", "gaspipes", "Pipe 8")
# parts that are meant to sit in or against other things
ATTACHED_WORDS = ("Echelle", "Wood Ladder", "Handrail", "Chain", "Pipe 01 02", "Steel Catwalks", "Grid Net", "parapet",
                  "Wheel Tire")

CELL = 20.0  # m, spatial hash


class Rows:
    """Catalogue lookups by row id: name, editor category, tags."""

    def __init__(self, table: dict[str, tuple[str, str, tuple[str, ...]]]):
        self.table = table
        self._cat: dict[str, str] = {}

    @classmethod
    def from_catalog(cls, catalog) -> "Rows":
        return cls({a.id: (a.name, a.category, tuple(a.tags)) for a in catalog.assets.values()})

    @classmethod
    def from_assets_json(cls, path: str | Path) -> "Rows":
        doc = json.loads(Path(path).read_text())
        return cls({a["id"]: (a["name"], a["category"], tuple(a.get("tags", ()))) for a in doc["assets"]})

    def name(self, row: str) -> str:
        return self.table.get(row, (row, "", ()))[0]

    def tags(self, row: str) -> tuple[str, ...]:
        return self.table.get(row, (row, "", ()))[2]

    def cat(self, row: str) -> str:
        if row in self._cat:
            return self._cat[row]
        self._cat[row] = c = self._classify(row)
        return c

    def _classify(self, row: str) -> str:
        if row not in self.table:
            return "OTHER"
        n, c, tags = self.table[row]
        if row in GROUND_ROWS:
            return "GROUND"
        if row in BALLOON_ROWS:
            return "BALLOON"
        if row in CONTAINER_ROWS:
            return "CONTAINER"
        if 'rock' in tags or any(w in n for w in ROCK_WORDS):
            return "ROCK"
        # Catalogue inference may attach 'vehicle' to 'Bush' through 'bus'.
        # Explicit foliage category takes precedence (rocks handled above).
        if c == "Foliage":
            return "PLANT"
        if 'vehicle' in tags or any(w in n for w in VEHICLE_WORDS):
            return "VEHICLE"
        if c == "Basic shape":
            return "SHAPE"
        if c == "Interractive" or "Sign" in n or "LED" in n or "Neon" in n or "Lights" in n:
            return "ATTACH"
        if any(w in n for w in KIT_WORDS):
            return "KIT"
        if c == "Big structure":
            return "BUILDING"
        if c == "Big_assets":
            return "BIGPROP"
        if c == "Small_assets":
            return "PROP"
        if c == "Wall":
            return "KIT"
        return "OTHER"

    def attached(self, row: str) -> bool:
        n = self.name(row)
        return any(w in n for w in ATTACHED_WORDS)


# ---------------------------------------------------------------------------- scanning (needs the game)

def scan_commands(out_dir: Path, windows_path, gap_cm: float = 30.0, region_m=None, grid: int = 10,
                  extent_m: float = 3000.0) -> tuple[list[str], list[Path]]:
    """The agent commands for a scan split into grid x grid regions (each a short game-thread pass)."""
    if region_m:
        x0, y0, x1, y1 = (v * 100 for v in region_m)
    else:
        x0 = y0 = -extent_m * 100
        x1 = y1 = extent_m * 100
    cmds, paths = [], []
    xs = [x0 + (x1 - x0) * i / grid for i in range(grid + 1)]
    ys = [y0 + (y1 - y0) * j / grid for j in range(grid + 1)]
    k = 0
    for j in range(grid):
        for i in range(grid):
            k += 1
            p = out_dir / f"col_r{k}.txt"
            paths.append(p)
            # Inclusive shared boundaries: duplicate IDs are merged by analyse.
            # Subtracting 1 cm used to leave thin unscanned strips between cells.
            cmds.append(f"collide {windows_path(p)} {xs[i]:.0f} {ys[j]:.0f} {xs[i + 1]:.0f} "
                        f"{ys[j + 1]:.0f} {gap_cm:.0f}")
    return cmds, paths


# ---------------------------------------------------------------------------- analysis

def _load(paths):
    O, B, I, S, F = {}, {}, [], {}, {}
    E, seed = collections.defaultdict(set), set()
    for fn in paths:
        for line in Path(fn).read_text(errors="replace").splitlines():
            p = line.split()
            if not p:
                continue
            t = p[0]
            if t == "O":
                O[p[1]] = (p[2],) + tuple(float(v) / 100 for v in p[3:9])
            elif t == "B":
                B[p[1]] = (p[2],) + tuple(float(v) / 100 for v in p[3:9])
            elif t == "I":
                I.append((p[1], p[2], int(p[4])))
            elif t == "S":
                S[p[1]] = (p[1], float(p[2]), p[3], float(p[5]) / 100, float(p[6]) / 100)
            elif t == "R":
                if p[2] == "0":
                    seed.add(p[1])
                else:
                    E[p[1]].add(p[2])
                    E[p[2]].add(p[1])
            elif t == "T":
                for b in p[2:]:
                    if b == "0":
                        seed.add(p[1])
                    else:
                        E[p[1]].add(b)
                        E[b].add(p[1])
            elif t == "F":
                F[p[1]] = (p[2], float(p[3]) / 100, float(p[4]) / 100, float(p[5]) / 100, float(p[6]) / 100,
                           int(p[7]) if len(p) > 7 else 0)
    return O, B, I, list(S.values()), F, E, seed


class _Grid:
    def __init__(self):
        self.cells = collections.defaultdict(list)

    def add(self, key, b):
        for cx in range(int(b[1] // CELL), int(b[4] // CELL) + 1):
            for cy in range(int(b[2] // CELL), int(b[5] // CELL) + 1):
                self.cells[(cx, cy)].append((key, b))

    def near(self, b):
        seen = set()
        for cx in range(int(b[1] // CELL), int(b[4] // CELL) + 1):
            for cy in range(int(b[2] // CELL), int(b[5] // CELL) + 1):
                for key, o in self.cells.get((cx, cy), ()):
                    if key not in seen:
                        seen.add(key)
                        yield key, o


def _pair_kind(rows: Rows, ra: str, rb: str, lvl: int):
    ca, cb = rows.cat(ra), rows.cat(rb)
    s = {ca, cb}
    if s & {"PLANT", "GROUND", "BALLOON"}:
        return None
    if ca == cb and ca in ("KIT", "ROCK", "SHAPE", "ATTACH"):
        return None  # walls, scaffolds, rock piles and shapes are built from overlapping parts
    if "ROCK" in s:
        other = cb if ca == "ROCK" else ca
        if rows.attached(ra) or rows.attached(rb) or (other in ("KIT", "ATTACH") and lvl < 2):
            return None
        return "rock"
    if "VEHICLE" in s:
        if (ca == cb == "VEHICLE" and lvl < 2) or rows.attached(ra) or rows.attached(rb):
            return None  # stacked cars touch a little
        return "vehicle"
    if rows.attached(ra) or rows.attached(rb) or "ATTACH" in s:
        return None
    big = {"CONTAINER", "BUILDING"}
    if s & big:
        if s <= big | {"BIGPROP"}:
            return "clash"
        if lvl >= 2 and s & {"KIT", "PROP"}:
            return "clash"
        return None
    if lvl >= 2 and s & {"PROP", "BIGPROP"} and s <= {"PROP", "BIGPROP", "KIT", "SHAPE"} and not s <= {"KIT", "SHAPE"}:
        return "prop"
    return None


def _view(x, y, z, size):
    """A camera pose that frames a spot: from the south-west, a little above, looking at it."""
    d = max(12.0, min(45.0, size * 1.4 + 10.0))
    h = max(4.0, d * 0.45)
    a = math.radians(225)
    cx, cy, cz = x + d * math.cos(a), y + d * math.sin(a), z + h
    yaw = math.degrees(math.atan2(y - cy, x - cx))
    pitch = -math.degrees(math.atan2(h, d))
    return {"x": round(cx, 1), "y": round(cy, 1), "z": round(cz, 1), "yaw": round(yaw, 1), "pitch": round(pitch, 1)}


def analyse(paths, rows: Rows, gap: float = 0.3, cluster_m: float = 9.0) -> dict:
    O, B, I, S, F, E, seed = _load(paths)
    items = []  # (kind, x, y, z, size, rows, severity, measure)

    # --- intersections: keep pairs both sides confirm
    lv = {}
    for a, b, lvl in I:
        lv[(a, b)] = max(lv.get((a, b), 0), lvl)
    for (a, b), lvl in lv.items():
        if a not in O or b not in O or (b, a) not in lv or a > b:
            continue
        lvl = max(lvl, lv[(b, a)])
        k = _pair_kind(rows, O[a][0], O[b][0], lvl)
        if not k:
            continue
        va = (O[a][4] - O[a][1]) * (O[a][5] - O[a][2])
        vb = (O[b][4] - O[b][1]) * (O[b][5] - O[b][2])
        s = O[a] if va <= vb else O[b]  # the marker goes on the smaller object
        items.append((k, (s[1] + s[4]) / 2, (s[2] + s[5]) / 2, s[3], max(s[4] - s[1], s[5] - s[2]),
                      [O[a][0], O[b][0]], lvl, None))

    # --- vehicles sunk into the surface under them
    sunk_minor = collections.Counter()
    for a, d, srow, x, y in S:
        if a not in O:
            continue
        ca = rows.cat(O[a][0])
        surf = rows.cat(srow) if srow != "-" else "FLOOR"
        if surf not in ("GROUND", "FLOOR") or srow in GRASS_ROWS or srow in WATER_ROWS or "Boat" in rows.name(O[a][0]):
            continue
        if ca == "VEHICLE" and d >= 8:
            items.append(("sunk", x, y, O[a][3], max(O[a][4] - O[a][1], O[a][5] - O[a][2]), [O[a][0], srow],
                          1 if d < 20 else 2, round(d, 1)))
        elif ca in ("PROP", "BIGPROP", "CONTAINER", "SHAPE") and d >= 5:
            sunk_minor[ca] += 1

    # --- floating: whatever is not connected to the ground through rest-on and touch edges
    def nob(a):
        obj = O.get(a, B.get(a))
        return obj is None or rows.cat(obj[0]) not in ("BALLOON", "PLANT")

    # Grounding requires measured contact, never a global z=0 assumption.
    # Vegetation is not a structural load path, even when planted / grounded.
    grounded = {a for a in seed if nob(a)}
    grounded |= {b for a in list(E) for b in E[a] if b not in O and nob(b)}  # gates and blueprint actors: solid

    def spread(start):
        todo = list(start)
        while todo:
            a = todo.pop()
            for b in E.get(a, ()):
                if b not in grounded and nob(b):
                    grounded.add(b)
                    todo.append(b)

    spread(grounded)
    # blueprint buildings report only their bounds (parts of them have no query collision): resting on or
    # against the box of a building counts as supported
    bgrid = _Grid()
    for key, b in B.items():
        if any(t in rows.tags(b[0]) for t in ("building", "water")):
            bgrid.add(key, b)

    def on_building(b, m=0.3):
        for _, o in bgrid.near(b):
            if (o[1] - m < b[4] and b[1] < o[4] + m and o[2] - m < b[5] and b[2] < o[5] + m
                    and o[3] - m < b[6] and b[3] < o[6] + m):
                return True
        return False

    extra = [a for a in O if a not in grounded and nob(a) and on_building(O[a])]
    grounded |= set(extra)
    spread(extra)

    ogrid = _Grid()
    for key, b in O.items():
        if rows.cat(b[0]) not in ("BALLOON", "PLANT"):
            ogrid.add(key, b)

    def gap_below(a):
        """Down to the highest bounding-box top under the footprint. Rays fall through lattices, boxes do not."""
        b, best = O[a], None
        for key, o in ogrid.near(b):
            if key == a:
                continue
            if (o[1] < b[4] - 0.05 and b[1] + 0.05 < o[4] and o[2] < b[5] - 0.05 and b[2] + 0.05 < o[5]
                    and o[6] <= b[3] + 0.05 and (best is None or o[6] > best)):
                best = o[6]
        return None if best is None else b[3] - best

    float_counts = collections.Counter()
    seen = set()
    for a0 in O:
        if a0 in grounded or a0 in seen or not nob(a0):
            continue
        comp, todo = [], [a0]
        seen.add(a0)
        while todo:
            a = todo.pop()
            comp.append(a)
            for b in E.get(a, ()):
                if b in O and b not in seen and b not in grounded and nob(b):
                    seen.add(b)
                    todo.append(b)
        crow = [O[a][0] for a in comp]
        if all(rows.cat(r) == "PLANT" for r in crow):
            continue
        gaps = []
        for a in comp:
            g = F[a][4] if a in F else None
            ga = gap_below(a)
            if ga is not None and (g is None or g < 0 or ga < g):
                g = max(ga, 0.0)
            if g is not None and g >= 0:
                gaps.append(g)
        g = min(gaps) if gaps else -1.0
        side = any(F[a][5] for a in comp if a in F)
        x0 = min(O[a][1] for a in comp); y0 = min(O[a][2] for a in comp); z0 = min(O[a][3] for a in comp)
        x1 = max(O[a][4] for a in comp); y1 = max(O[a][5] for a in comp); z1 = max(O[a][6] for a in comp)
        dim = max(x1 - x0, y1 - y0, z1 - z0)
        if 0 <= g < 1.0:
            k = "hover"
        elif dim < 2.5 and 0 <= g < 1.6:
            k = "hover"  # TV, fuse box, sign at hand height: wall-mounted
        elif all(rows.cat(r) == "ATTACH" for r in crow):
            k = "floatlite"
        elif side and len(comp) == 1:
            k = "floatside"
        else:
            k = "float"
        float_counts[k] += len(comp)
        items.append((k, (x0 + x1) / 2, (y0 + y1) / 2, z0, max(x1 - x0, y1 - y0), crow,
                      2 if (g < 0 or g >= 3.0) else 1, round(g, 2) if g >= 0 else None))

    # --- cluster per kind within cluster_m in plan
    clusters = []
    by_kind = collections.defaultdict(list)
    for it in items:
        by_kind[it[0]].append(it)
    for k, its in by_kind.items():
        parent = list(range(len(its)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        cell = collections.defaultdict(list)
        for i, it in enumerate(its):
            cell[(int(it[1] // cluster_m), int(it[2] // cluster_m))].append(i)
        for i, it in enumerate(its):
            cx, cy = int(it[1] // cluster_m), int(it[2] // cluster_m)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in cell.get((cx + dx, cy + dy), ()):
                        if j > i and math.hypot(its[j][1] - it[1], its[j][2] - it[2]) < cluster_m:
                            parent[find(j)] = find(i)
        groups = collections.defaultdict(list)
        for i in range(len(its)):
            groups[find(i)].append(its[i])
        for g in groups.values():
            names = collections.Counter(rows.name(r) for it in g for r in it[5])
            x = sum(i[1] for i in g) / len(g)
            y = sum(i[2] for i in g) / len(g)
            z = min(i[3] for i in g)
            size = max(max(i[4] for i in g), max(abs(i[1] - x) for i in g) * 2, max(abs(i[2] - y) for i in g) * 2)
            measures = [i[7] for i in g if i[7] is not None]
            clusters.append({
                "kind": k, "x": round(x, 1), "y": round(y, 1), "z": round(z, 1),
                "z_top": round(max(i[3] for i in g), 1), "n": len(g), "severity": max(i[6] for i in g),
                "measure": (max(measures) if k == "sunk" else min(measures)) if measures else None,
                "objects": [f"{n} x{c}" if c > 1 else n for n, c in names.most_common(6)],
                "view": _view(x, y, z, size),
            })
    clusters.sort(key=lambda c: (-c["severity"], ORDER.index(c["kind"]), -c["n"]))
    for i, c in enumerate(clusters, 1):
        c["id"] = i

    # --- a plan of everything for the map: [x0, y0, x1, y1] in decimetres, by class
    classes = {"GROUND": "ground", "PLANT": "plant", "ROCK": "rock", "CONTAINER": "box", "BUILDING": "build",
               "VEHICLE": "veh"}
    plan = collections.defaultdict(list)
    for b in list(O.values()) + list(B.values()):
        r = b[0]
        if r in WATER_ROWS:
            cls = "water"
        elif rows.cat(r) == "BALLOON" or "decal" in rows.tags(r):
            continue
        else:
            cls = classes.get(rows.cat(r), "other")
            if r in B and "building" in rows.tags(r):
                cls = "build"
        plan[cls].append([round(b[1] * 10), round(b[2] * 10), round(b[4] * 10), round(b[5] * 10)])

    return {
        "objects_checked": len(O),
        "blueprints": len(B),
        "kinds": {k: {"label": v[0], "colour": v[1], "hint": v[2]} for k, v in KINDS.items()},
        "summary": {k: sum(1 for c in clusters if c["kind"] == k) for k in KINDS},
        "clusters": clusters,
        "sunk_minor": dict(sunk_minor),
        "float_objects": dict(float_counts),
        "plan": dict(plan),
    }


def brief(report: dict, top: int = 15, include_minor: bool = False) -> dict:
    """A compact version for a tool reply: counts and the worst issues, no plan."""
    keep = [c for c in report["clusters"] if include_minor or c["kind"] != "hover"]
    return {
        "objects_checked": report["objects_checked"],
        "issues": {k: n for k, n in report["summary"].items() if n},
        "minor_sinking_props": sum(report["sunk_minor"].values()),
        "top": [{"id": c["id"], "kind": c["kind"], "at_m": [c["x"], c["y"], c["z"]], "severity": c["severity"],
                 "measure": c["measure"], "objects": c["objects"][:3], "view": c["view"]} for c in keep[:top]],
    }


# ---------------------------------------------------------------------------- report page

def render_html(report: dict, title: str = "Collision check", subtitle: str = "", eyebrow: str = "",
                shots: list[dict] | None = None, notes: list[dict] | None = None, standalone: bool = True) -> str:
    """One self-contained page: a plan of the map with every issue marked, a filterable list, optional
    screenshots ({"src": data-uri, "caption": str, "issue": id}) and notes ({"title": str, "html": str})."""
    data = dict(report)
    data["title"] = title
    data["subtitle"] = subtitle
    data["eyebrow"] = eyebrow or "Uncrashed track editor"
    data["shots"] = shots or []
    data["notes"] = notes if notes is not None else DEFAULT_NOTES
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    body = _PAGE.replace("__TITLE__", _esc(title)).replace("__DATA__", payload)
    if standalone:
        return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">"
                "</head><body>" + body + "</body></html>")
    return body


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


DEFAULT_NOTES = [
    {"title": "How the check works", "html": (
        "<ul><li>The agent's <code>collide</code> command asks the game's own physics: box and ray traces against the real "
        "collision of every placed object, not catalogue sizes.</li>"
        "<li>Intersections: each object's box is shrunk by 25 cm (or 35 %) first, and both objects must hit each other, so "
        "things that merely touch are not reported. A second, half-size box marks deep overlaps as high severity.</li>"
        "<li>Floating: an object counts as held up when it rests on or touches the ground, a building, or anything that is "
        "itself held up. Whatever is left over is floating, as one group per connected cluster. Balloons are exempt.</li>"
        "<li>Ignored on purpose: plants, ground tiles, ladders, handrails, chains and nets against what they hang on, and "
        "walls, scaffolds or rocks built from overlapping parts.</li></ul>")},
    {"title": "Run it again", "html": (
        "<ul><li>With the MCP: <code>check_collisions()</code> scans the open map and writes this page to "
        "<code>/tmp/uncrashed-collide/report.html</code>.</li>"
        "<li>By hand: <code>collide &lt;file&gt; [x0 y0 x1 y1] [gap_cm]</code> in centimetres, then "
        "<code>python -m uncrashed_mcp.collide &lt;files&gt;</code>.</li>"
        "<li><code>trace x y z_from z_to</code> shows what a vertical ray hits, for checking one spot.</li></ul>")},
]

_PAGE_PATH = Path(__file__).parent / "data" / "collide_page.html"
_PAGE = _PAGE_PATH.read_text() if _PAGE_PATH.exists() else ""


def main(argv=None) -> None:
    """python -m uncrashed_mcp.collide col_r1.txt [...] : write report.json and report.html next to the first file."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--assets", default=str(Path(__file__).parent / "data" / "assets.json"))
    ap.add_argument("--title", default="Collision check")
    args = ap.parse_args(argv)
    report = analyse(args.files, Rows.from_assets_json(args.assets))
    out = Path(args.files[0]).parent
    (out / "report.json").write_text(json.dumps(report))
    (out / "report.html").write_text(render_html(report, title=args.title))
    print(json.dumps(brief(report), indent=1))


if __name__ == "__main__":
    main()
