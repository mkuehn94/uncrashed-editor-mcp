"""The in-game agent: injection, the command protocol, and every live editor operation.

native/agent.c is injected once per game process. It finds the engine's globals by
shape and calls the track editor's own Blueprint functions through ProcessEvent:
"Spawn Asset", "SpawnTrackAsset", "SaveMap", "LoadLevel", K2_DestroyActor and the
K2 actor transform setters. Nothing here presses keys or pokes render memory.

This module speaks metres and degrees. The agent speaks centimetres.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import threading
from contextlib import contextmanager
import re
import subprocess
import time
from pathlib import Path

from .geom import quat_rotator, rotator_from_quat
from .mapfile import list_user_maps, load_map, sanitize_name
from .paths import game_pid, maps_dir

CMD = Path("/tmp/uncrashed-agent-cmd.txt")
OUT = Path("/tmp/uncrashed-agent-out.txt")
LOG = Path("/tmp/uncrashed-agent.log")
SEQ = Path("/tmp/uncrashed-agent-seq.txt")
BATCH = Path("/tmp/uncrashed-agent-batch.txt")
BATCH_WINDOWS = "Z:\\tmp\\uncrashed-agent-batch.txt"
DLL_NAME = "uncrashed_agent.dll"
START_GATE = "STARTGATE"


class AgentError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# injection
# ---------------------------------------------------------------------------


def _pid() -> int:
    pid = game_pid()
    if pid is None:
        raise AgentError("Uncrashed is not running")
    return pid


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def dll_path() -> Path:
    override = os.environ.get("UNCRASHED_AGENT_DLL")
    return Path(override) if override else _repo_root() / "native" / DLL_NAME


def _wine() -> tuple[Path, Path]:
    home = Path.home()
    wine = Path(os.environ.get("UNCRASHED_WINE", home / ".local/share/Steam/steamapps/common/Proton - Experimental/files/bin/wine"))
    prefix = Path(os.environ.get("UNCRASHED_WINEPREFIX", home / ".local/share/Steam/steamapps/compatdata/1682970/pfx"))
    return wine, prefix


def _windows_path(path: Path) -> str:
    return "Z:" + str(path).replace("/", "\\")


def loaded(pid: int | None = None) -> bool:
    try:
        maps = Path(f"/proc/{pid or _pid()}/maps").read_text(errors="replace")
    except (OSError, AgentError):
        return False
    return DLL_NAME in maps


def ensure_agent() -> None:
    """Inject the agent into this game process if it is not there yet."""
    pid = _pid()
    if loaded(pid):
        return
    dll = dll_path()
    inject = dll.parent / "inject.exe"
    wine, prefix = _wine()
    for needed in (dll, inject, wine):
        if not needed.is_file():
            raise AgentError(f"missing {needed}")
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")
    env["WINEPREFIX"] = str(prefix)
    subprocess.run([str(wine), _windows_path(inject), _windows_path(dll)], env=env, check=True, capture_output=True, text=True)
    deadline = time.time() + 10
    while time.time() < deadline:
        if loaded(pid):
            return
        time.sleep(0.1)
    raise AgentError("the agent did not load into the game")


def unload() -> dict:
    """Free the agent so a rebuilt DLL is injected by the next call."""
    if not loaded():
        return {"unloaded": False, "loaded": False}
    try:
        reply = execute("unload", timeout=5)
        if not reply['ok']:
            return {'unloaded': False, 'loaded': loaded(),
                    'error': '; '.join(_errors(reply['lines']) or [reply['head']])}
    except AgentError as exc:
        if loaded():
            return {'unloaded': False, 'loaded': True, 'error': str(exc)}
    deadline = time.time() + 5
    while time.time() < deadline and loaded():
        time.sleep(0.1)
    return {"unloaded": not loaded(), "loaded": loaded()}


def log_tail(lines: int = 80) -> str:
    if not LOG.exists():
        return ""
    return "\n".join(LOG.read_text(errors="replace").splitlines()[-max(1, lines):])


# ---------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------


def _next_seq() -> int:
    try:
        current = int(SEQ.read_text().strip())
    except (OSError, ValueError):
        current = 1000
    current += 1
    SEQ.write_text(str(current))
    return current


_protocol_lock = threading.RLock()
_lock_depth = 0


@contextmanager
def session():
    """Serialize complete multi-command operations across MCP/CLI clients."""
    global _lock_depth
    with _protocol_lock:
        outer = _lock_depth == 0
        lock = None
        if outer:
            lock = open('/tmp/uncrashed-agent.lock', 'a')
            fcntl.flock(lock, fcntl.LOCK_EX)
        _lock_depth += 1
        try:
            yield
        finally:
            _lock_depth -= 1
            if lock is not None:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()


def execute(command: str, timeout: float = 30.0) -> dict:
    with session():
        return _execute(command, timeout)


def _execute(command: str, timeout: float = 30.0) -> dict:
    """Run one agent command on the game thread.

    Returns {seq, ok, head, lines}. ok is False when the agent reported an error or
    a caught fault; lines are the reply between the header and the end marker.
    """
    command = " ".join(command.split())
    if not command:
        raise AgentError("empty command")
    ensure_agent()
    seq = _next_seq()
    tmp = CMD.with_suffix(".tmp")
    tmp.write_text(f"{seq} {command}\n")
    tmp.replace(CMD)
    prefix = f"{seq} "
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            text = OUT.read_text(errors="replace")
        except OSError:
            text = ""
        if text.startswith(prefix):
            rows = text.rstrip("\n").split("\n")
            ok = rows[-1].strip() == f"end {seq} ok"
            body = rows[1:-1] if rows[-1].startswith("end ") else rows[1:]
            return {"seq": seq, "ok": ok, "head": rows[0][len(prefix):], "lines": body}
        if not loaded():
            raise AgentError("the agent is not loaded (unloaded, or the game closed)")
        time.sleep(0.03)
    raise AgentError(f"the agent did not answer {command[:60]!r} within {timeout:.0f}s")


def _errors(lines: list[str]) -> list[str]:
    return [line.split("error ", 1)[1] for line in lines if "error " in line and line.lstrip("#0123456789 ").startswith("error ")]


def run(command: str, timeout: float = 30.0) -> list[str]:
    """execute(), raising AgentError with the agent's own message on failure."""
    reply = execute(command, timeout)
    if not reply["ok"]:
        problems = _errors(reply["lines"]) or [reply["head"]] + reply["lines"][-2:]
        raise AgentError("; ".join(problems))
    return reply["lines"]


def batch(commands: list[str], timeout: float | None = None, quiet: bool = False) -> tuple[dict[int, list[str]], str]:
    with session():
        return _batch(commands, timeout, quiet)


def _batch(commands: list[str], timeout: float | None = None, quiet: bool = False) -> tuple[dict[int, list[str]], str]:
    """Run many agent commands in one game-thread job. Returns (lines by index, summary).

    quiet makes the agent report only failures, for large jobs whose results are not needed.
    """
    if not commands:
        return {}, "batch lines=0 ok=0 failed=0"
    BATCH.write_text("\n".join(" ".join(c.split()) for c in commands) + "\n")
    lines = run(f"batch {BATCH_WINDOWS}{' quiet' if quiet else ''}", timeout or max(60.0, len(commands) * 0.02))
    by_index: dict[int, list[str]] = {}
    summary = ""
    for line in lines:
        m = re.match(r"^#(\d+) (.*)$", line)
        if m:
            by_index.setdefault(int(m.group(1)), []).append(m.group(2))
        elif line.startswith("batch "):
            summary = line
    return by_index, summary


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

_ACTOR = re.compile(
    r"^(?P<verb>\w+) (?P<addr>[0-9a-fA-F]+) row=(?P<row>\S+) class=(?P<cls>\S+)"
    r"(?: loc=(?P<loc>\S+) rot=(?P<rot>\S+) scale=(?P<scale>\S+))?"
    r"(?: d=(?P<d>\S+))?(?: tags=(?P<tags>.*))?$"
)


def _triple(text: str | None) -> list[float]:
    return [float(v) for v in text.split(",")] if text else []


def parse_actor(line: str) -> dict | None:
    m = _ACTOR.match(line.strip())
    if not m:
        return None
    out: dict = {"actor": m["addr"], "asset": None if m["row"] == "-" else m["row"], "class": m["cls"]}
    if m["loc"]:
        loc, rot, scale = _triple(m["loc"]), _triple(m["rot"]), _triple(m["scale"])
        out.update(
            {
                "x": round(loc[0] / 100, 3), "y": round(loc[1] / 100, 3), "z": round(loc[2] / 100, 3),
                "pitch": round(rot[0], 3), "yaw": round(rot[1], 3), "roll": round(rot[2], 3),
                "scale": [round(v, 4) for v in scale],
            }
        )
    if m["d"]:
        out["distance_m"] = round(float(m["d"]) / 100, 3)
    if m["tags"] is not None:
        out["tags"] = [t for t in m["tags"].split(",") if t]
    return out


def _first_actor(lines: list[str]) -> dict:
    for line in lines:
        actor = parse_actor(line)
        if actor:
            return actor
    raise AgentError("the agent did not report an actor: " + " | ".join(lines[:3]))


def _cm(v: float) -> str:
    return f"{v * 100:.2f}"


def _spawn_command(asset: str, x: float, y: float, z: float, pitch: float, yaw: float, roll: float, scale, group: str = "") -> str:
    sx, sy, sz = scale
    if min(sx, sy, sz) <= 0:
        raise AgentError(f"scale must be positive, got {scale}")
    cmd = f"spawn {asset} {_cm(x)} {_cm(y)} {_cm(z)} {pitch:.4f} {yaw:.4f} {roll:.4f} {sx:.5f} {sy:.5f} {sz:.5f}"
    if group and group != "None":
        cmd += f" group={group}"
    return cmd


# ---------------------------------------------------------------------------
# editor operations
# ---------------------------------------------------------------------------


def map_info() -> dict:
    """The map open in the editor: folder, base level, title, object and gate counts."""
    info: dict = {}
    for line in run("mapinfo", 20):
        key, _, value = line.partition(" ")
        info[key] = int(value) if key in ("objects", "gates", "level_actors") else value
    for row in list_user_maps():
        if row["folder"] == info.get("folder"):
            info["display_name"] = row["name"]
    return info


def resolve() -> dict[str, int]:
    """Addresses and offsets the agent found for this game build (camera tools use them)."""
    values = {}
    for line in run("resolve", 30):
        key, _, value = line.partition("=")
        try:
            values[key.strip()] = int(value.strip(), 16) if not key.endswith("_slot") else int(value)
        except ValueError:
            continue
    return values


def spawn(asset: str, x: float, y: float, z: float, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0, scale=(1.0, 1.0, 1.0)) -> dict:
    """Place any catalogue row through the editor's own "Spawn Asset". x, y, z is the pivot."""
    return _first_actor(run(_spawn_command(asset, x, y, z, pitch, yaw, roll, scale)))


def spawn_many(items: list[dict]) -> dict:
    """Place many rows in one game-thread job. Items: asset, x, y, z, optional yaw, pitch, roll, sx, sy, sz."""
    commands = [
        _spawn_command(
            str(item["asset"]), float(item["x"]), float(item["y"]), float(item["z"]),
            float(item.get("pitch", 0)), float(item.get("yaw", item.get("yaw_deg", 0))), float(item.get("roll", 0)),
            (float(item.get("sx", 1)), float(item.get("sy", 1)), float(item.get("sz", 1))),
        )
        for item in items
    ]
    results, summary = batch(commands)
    placed, failed = [], []
    for index in range(len(commands)):
        lines = results.get(index, [])
        actor = next((a for a in (parse_actor(l) for l in lines) if a), None)
        if actor:
            placed.append(actor)
        else:
            failed.append({"index": index, "asset": items[index]["asset"], "error": " ".join(_errors(lines)) or "no reply"})
    return {"placed": len(placed), "failed": failed[:20], "failed_count": len(failed), "actors": placed[:50], "summary": summary}


def gate(row: str, x: float, y: float, z: float, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> dict:
    """Place a race gate (STARTGATE, DefaultOneW, ...) through the editor's "SpawnTrackAsset"."""
    return _first_actor(run(f"gate {row} {_cm(x)} {_cm(y)} {_cm(z)} {pitch:.4f} {yaw:.4f} {roll:.4f}"))


def find(x: float, y: float, z: float, radius: float = 10.0, limit: int = 50) -> list[dict]:
    """Placed objects and gates whose pivot is within radius metres, nearest first."""
    limit = max(1, min(int(limit), 2000))
    lines = run(f"actors {_cm(x)} {_cm(y)} {_cm(z)} {_cm(radius)} {limit}", 60)
    actors = [a for a in (parse_actor(l) for l in lines) if a]
    actors.sort(key=lambda a: a.get("distance_m", 0))
    return actors


def _target(actor: str = "", x: float | None = None, y: float | None = None, z: float | None = None, tolerance: float = 0.5) -> str:
    if actor:
        return actor
    if x is None or y is None or z is None:
        raise AgentError("pass an actor address, or x, y and z of its pivot")
    near = find(x, y, z, tolerance, 1)
    if not near:
        raise AgentError(f"no placed object within {tolerance} m of {[x, y, z]}")
    return near[0]["actor"]


def get(actor: str) -> dict:
    return _first_actor(run(f"get {actor}"))


def transform(
    actor: str = "",
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    new_x: float | None = None,
    new_y: float | None = None,
    new_z: float | None = None,
    pitch: float | None = None,
    yaw: float | None = None,
    roll: float | None = None,
    sx: float | None = None,
    sy: float | None = None,
    sz: float | None = None,
    tolerance: float = 0.5,
) -> dict:
    """Move, turn or scale one placed object. Omitted values stay as they are."""
    target = _target(actor, x, y, z, tolerance)
    fields = [
        "-" if new_x is None else _cm(new_x), "-" if new_y is None else _cm(new_y), "-" if new_z is None else _cm(new_z),
        "-" if pitch is None else f"{pitch:.4f}", "-" if yaw is None else f"{yaw:.4f}", "-" if roll is None else f"{roll:.4f}",
        "-" if sx is None else f"{sx:.5f}", "-" if sy is None else f"{sy:.5f}", "-" if sz is None else f"{sz:.5f}",
    ]
    return _first_actor(run(f"transform {target} " + " ".join(fields)))


def replace(asset: str, actor: str = "", x: float | None = None, y: float | None = None, z: float | None = None, tolerance: float = 0.5) -> dict:
    """Swap a placed object for another catalogue row with the same transform."""
    return _first_actor(run(f"replace {_target(actor, x, y, z, tolerance)} {asset}"))


def delete(
    actor: str = "",
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    tolerance: float = 0.5,
    count: int = 1,
    box: tuple[float, float, float, float] | None = None,
    asset: str = "",
    z_range: tuple[float, float] | None = None,
) -> dict:
    """Delete placed objects: one address, the nearest count to a point, or everything in a box."""
    if actor:
        lines = run(f"destroy {actor}")
    elif box is not None:
        z0, z1 = ("-", "-") if z_range is None else (_cm(min(z_range)), _cm(max(z_range)))
        lines = run(f"destroybox {_cm(box[0])} {_cm(box[1])} {_cm(box[2])} {_cm(box[3])} {z0} {z1} {asset or '-'}", 120)
    else:
        if x is None or y is None or z is None:
            raise AgentError("pass an actor, x/y/z, or a box")
        lines = run(f"destroyat {_cm(x)} {_cm(y)} {_cm(z)} {_cm(tolerance)} {max(1, min(int(count), 2000))}", 120)
    removed = []
    total = None
    for line in lines:
        m = re.match(r"^destroyed ([0-9a-fA-F]+) row=(\S+)", line)
        if m:
            removed.append({"actor": m.group(1), "asset": None if m.group(2) == "-" else m.group(2), "gate": line.endswith(" gate")})
        c = re.match(r"^count (\d+)$", line)
        if c:
            total = int(c.group(1))  # the agent lists only the first 50 of a large deletion
    return {"destroyed": total if total is not None else len(removed), "removed": removed[:50]}


def clear() -> dict:
    """Remove every placed object and gate from the open level (not the base level)."""
    lines = run("clear", 120)
    m = re.search(r"cleared objects=(\d+) gates=(\d+)", " ".join(lines))
    return {"objects": int(m.group(1)), "gates": int(m.group(2))} if m else {"log": lines}


def _wait_for_save(path: Path, previous_mtime: int | None, timeout: float = 10.0) -> bool:
    """SaveMap can defer its file write until a later game tick."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if path.stat().st_mtime_ns != previous_mtime:
                json.loads(path.read_text())  # Do not report a partial write as saved.
                return True
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    return False


def save(name: str = "", display_name: str = "") -> dict:
    with session():
        return _save(name, display_name)


def _save(name: str = "", display_name: str = "") -> dict:
    """Write the open level with the editor's own SaveMap.

    name is the folder (empty = the map that is open); display_name is the title
    (empty = keep the open map's title when saving over it, else the folder name).
    """
    info = map_info()
    folder = sanitize_name(name) if name else info.get("folder", "")
    if not folder:
        raise AgentError("no map is open and no name was given")
    title = display_name or (info.get("display_name") if folder == info.get("folder") else "") or folder
    path = maps_dir() / folder / f"{folder}.json"
    from . import managed
    try:
        source = managed.directory_path(info)
    except ValueError:
        source = maps_dir() / info.get('folder', '')
    if source != path.parent and (source/managed.STATE_NAME).exists():
        if (path.parent/managed.STATE_NAME).exists() or (path.parent/managed.ZONES_NAME).exists():
            raise AgentError('save-as target already has MCP metadata; explicit merge required before overwriting')
        managed.read_state({'base': info.get('base')}, source)  # reject wrong-base metadata before saving
    previous_mtime = path.stat().st_mtime_ns if path.exists() else None
    run(f"save {folder} {' '.join(title.split())}", 120)
    written = _wait_for_save(path, previous_mtime)
    if written:
        # SaveMap does not initialize LoadedMapDirectory for a new blank map.
        # Retarget Save / subsequent save_live calls without loading or clearing it.
        managed.copy_metadata(source, path.parent, info.get('base', ''))
        run(f"setmap {_windows_path(path.parent)}")
    return {
        "folder": folder,
        "display_name": title,
        "path": str(path),
        "written": written,
        "bytes": path.stat().st_size if written else 0,
        "objects": info.get("objects"),
        "gates": info.get("gates"),
        "overwrote_open_map": folder == info.get("folder"),
    }


def _wait_for_base(base: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if map_info().get("base") == base:
            return
        time.sleep(0.5)
    raise AgentError(f"the editor did not switch to base level {base}")


def load(name: str, move_camera: bool = True) -> dict:
    with session():
        return _load(name, move_camera)


def _gate_scale_commands(gates: list, lines: list[str]) -> list[str]:
    """Restore saved gate sizes after SpawnTrackAsset's unit-scale spawn.

    Validate the complete freshly spawned gate sequence before changing any actor.
    Handles come from the current inventory, never the saved map.
    """
    actors = [parse_actor(line) for line in lines if line.startswith("gate ")]
    if len(actors) != len(gates):
        raise AgentError("gate scale restore refused: gate count changed")
    commands = []
    for (_, transform), actor in zip(gates, actors):
        tr = transform.get("Translation") or {"X": 0, "Y": 0, "Z": 0}
        expected = [float(tr[k]) / 100 for k in ("X", "Y", "Z")]
        if actor.get("class") != "BP_Gate_C" or any(
            not math.isfinite(value) or not math.isfinite(actor[k]) or abs(actor[k] - value) > .001
            for k, value in zip(("x", "y", "z"), expected)
        ):
            raise AgentError("gate scale restore refused: gate sequence changed")
        want_q = transform.get("Rotation") or {"X": 0, "Y": 0, "Z": 0, "W": 1}
        actual_q = quat_rotator(actor["pitch"], actor["yaw"], actor["roll"])
        norm = math.sqrt(sum(float(want_q[k]) ** 2 for k in ("X", "Y", "Z", "W")))
        dot = abs(sum(float(want_q[k]) * actual_q[k] for k in ("X", "Y", "Z", "W"))) / norm if norm else 0
        if not math.isfinite(dot) or dot < .999999995 or any(not math.isfinite(v) or abs(v - 1) > 1e-5 for v in actor["scale"]):
            raise AgentError("gate scale restore refused: gate transform changed")
        scale = [float((transform.get("Scale3D") or {}).get(k, 1)) for k in ("X", "Y", "Z")]
        if not all(math.isfinite(value) and value > 0 for value in scale):
            raise AgentError("gate scale restore refused: invalid saved scale")
        if any(abs(value - 1) > 1e-6 for value in scale):
            commands.append(f"transform {actor['actor']} - - - - - - {' '.join(f'{value:.8g}' for value in scale)}")
    return commands


def _load(name: str, move_camera: bool = True) -> dict:
    """Replace what the open editor shows with a saved map, without leaving the editor.

    Uses the editor's own pieces: LoadLevel for the base, Spawn Asset for every
    object, SpawnTrackAsset for the start and checkpoints. The editor's Save button
    then writes back to this map's folder.
    """
    folder = sanitize_name(name)
    doc = load_map(folder)
    started = time.time()
    info = map_info()
    switched = False
    if doc.base and doc.base != info.get("base"):
        run(f"base {doc.base}")
        _wait_for_base(doc.base)
        time.sleep(1.0)  # the base level streams in; anything it brought along is cleared next
        switched = True
    cleared = clear()

    commands: list[str] = []
    for rname in doc.order:
        group = doc.groups[rname]
        tags = group.parallel.get("UniqueGroupTag") or []
        for index, t in enumerate(group.transforms):
            tr, sc = t["Translation"], t["Scale3D"]
            pitch, yaw, roll = rotator_from_quat(t["Rotation"])
            commands.append(
                _spawn_command(
                    rname, float(tr["X"]) / 100, float(tr["Y"]) / 100, float(tr["Z"]) / 100, pitch, yaw, roll,
                    (float(sc["X"]), float(sc["Y"]), float(sc["Z"])),
                    str(tags[index]) if index < len(tags) else "",
                )
            )
    gates = []
    checkpoints = doc.race.get("CheckPoints") or []
    if checkpoints:
        gates.append((START_GATE, doc.race.get("StartTransform") or {}))
        gates.extend((cp.get("RowName") or "DefaultOneW", cp.get("Transform") or {}) for cp in checkpoints)
    for row, t in gates:
        tr = t.get("Translation") or {"X": 0, "Y": 0, "Z": 0}
        pitch, yaw, roll = rotator_from_quat(t.get("Rotation") or {"X": 0, "Y": 0, "Z": 0, "W": 1})
        commands.append(f"gate {row} {float(tr['X']):.2f} {float(tr['Y']):.2f} {float(tr['Z']):.2f} {pitch:.4f} {yaw:.4f} {roll:.4f}")
    if checkpoints:
        commands.append(f"laps {int(doc.race.get('Laps') or 1)}")
    commands.append(f"setmap {folder}")

    results, summary = batch(commands, timeout=max(120.0, len(commands) * 0.03), quiet=True)
    failures = [
        {"line": index, "command": commands[index][:80], "error": " ".join(_errors(lines))}
        for index, lines in results.items() if _errors(lines)
    ]
    if not failures and any(
        any(abs(float((t.get("Scale3D") or {}).get(k, 1)) - 1) > 1e-6 for k in ("X", "Y", "Z"))
        for _, t in gates
    ):
        try:
            scaling = _gate_scale_commands(gates, run("gates"))
            scaled, scale_summary = batch(scaling, timeout=max(60., len(scaling) * .1), quiet=True)
            failures.extend(
                {"line": index, "command": scaling[index][:80], "error": " ".join(_errors(lines))}
                for index, lines in scaled.items() if _errors(lines)
            )
            summary += "; gate sizes: " + scale_summary
        except Exception as exc:
            failures.append({"command": "restore gate scales", "error": str(exc)})
    camera = None
    if move_camera and doc.cam.get("Translation"):
        from .live import move_camera as _move_camera  # camera module is optional for loading

        tr = doc.cam["Translation"]
        pitch, yaw, _roll = rotator_from_quat(doc.cam.get("Rotation") or {"X": 0, "Y": 0, "Z": 0, "W": 1})
        try:
            camera = _move_camera(float(tr["X"]) / 100, float(tr["Y"]) / 100, float(tr["Z"]) / 100, yaw, pitch, hold=False)
        except Exception as exc:  # the map is loaded either way
            camera = {"error": str(exc)}
    after = map_info()
    try:
        zones = {'native': run('zones refresh', 120)}
    except Exception as exc:
        zones = {'error': str(exc), 'note': 'geometry loaded; foliage exclusions were not fully applied'}
    return {
        "folder": folder,
        "display_name": doc.display_name,
        "base": doc.base,
        "base_switched": switched,
        "cleared": cleared,
        "objects_in_file": doc.instance_count(),
        "gates_in_file": len(gates),
        "objects_now": after.get("objects"),
        "gates_now": after.get("gates"),
        "failed": failures[:20],
        "failed_count": len(failures),
        "seconds": round(time.time() - started, 1),
        "summary": summary,
        "camera": camera,
        "clearing_zones": zones,
    }
