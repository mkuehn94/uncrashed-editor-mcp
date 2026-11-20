"""The in-game agent: injection, the command protocol, and every live editor operation.

native/agent.c is injected once per game process. It finds the engine's globals by
shape and calls the track editor's own Blueprint functions through ProcessEvent:
"Spawn Asset", "SpawnTrackAsset", "SaveMap", the menu's ?MapEditor=<base> level open, K2_DestroyActor and the
K2 actor transform setters. Nothing here presses keys or pokes render memory.

This module speaks metres and degrees. The agent speaks centimetres.
"""

from __future__ import annotations

import fcntl
import glob
import itertools
import json
import math
import os
import secrets
import threading
from contextlib import contextmanager
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from .geom import quat_rotator, rotator_from_quat
from .mapfile import list_user_maps, load_map, sanitize_name
from .paths import game_pid, maps_dir

CMD = Path("/tmp/uncrashed-agent-cmd.txt")
OUT = Path("/tmp/uncrashed-agent-out.txt")
LOG = Path("/tmp/uncrashed-agent.log")
SEQ = Path("/tmp/uncrashed-agent-seq.txt")
# Name pattern only: every batch call writes its own <stem>.<pid>.<n>.<token><suffix> next to it (batch_files()).
BATCH = Path("/tmp/uncrashed-agent-batch.txt")
BATCH_KEEP = 86400.0  # seconds a batch file whose job may still be pending is left alone
# Older server processes flock this same file; never unlink or rename it.
LOCK = Path(os.environ.get("UNCRASHED_LOCK") or "/tmp/uncrashed-agent.lock")
DLL_NAME = "uncrashed_agent.dll"
START_GATE = "STARTGATE"


class AgentError(RuntimeError):
    """A live agent call failed. Tell the cases apart by type, never by message text:

    AgentNotSent: the failing call wrote nothing to the agent.
    AgentTimeout: the command was sent and no reply arrived in time; it may still run.
    AgentError itself with reply set: the agent answered and reported a failure or a caught fault.
    AgentError itself with reply None: sent, but no reply can come any more (agent unloaded, game
    closed), or a check of this module failed around the commands it sent.
    """

    reply: dict | None = None  # the _execute() reply when the agent itself answered with a failure


class AgentNotSent(AgentError):
    """The failing call sent nothing: offline mode, lock or agent busy, game not running, injection failed,
    empty command, or the wait was given up (on_lock_wait). An operation of several commands may have sent
    earlier ones: sent_count tells."""


class AgentTimeout(AgentError):
    """The command was sent but no reply arrived in time: it may still run."""


def _refuse_offline() -> None:
    if os.environ.get("UNCRASHED_OFFLINE"):
        raise AgentNotSent("offline mode (UNCRASHED_OFFLINE set): live agent calls are disabled")


def _note(text: str) -> None:
    """A stderr line of this module. A closed stderr must never abort a wait or change the exit code."""
    if sys.stderr is None:  # started with fd 2 closed: print(file=None) would write to stdout, into the result
        return
    try:
        print(text, file=sys.stderr, flush=True)
    except OSError:
        try:  # the reader is gone: keep the interpreter's exit-time flush of the unwritten line from failing too
            null = os.open(os.devnull, os.O_WRONLY)
            os.dup2(null, sys.stderr.fileno())
            os.close(null)
        except (OSError, ValueError, AttributeError):
            pass
    except (ValueError, RuntimeError, AttributeError):  # closed or missing stream; re-entered from a signal handler
        pass


# ---------------------------------------------------------------------------
# injection
# ---------------------------------------------------------------------------


def _pid() -> int:
    pid = game_pid()
    if pid is None:
        raise AgentNotSent("Uncrashed is not running")
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
    _refuse_offline()
    pid = _pid()
    if loaded(pid):
        return
    dll = dll_path()
    inject = dll.parent / "inject.exe"
    wine, prefix = _wine()
    for needed in (dll, inject, wine):
        if not needed.is_file():
            raise AgentNotSent(f"missing {needed}")
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")
    env["WINEPREFIX"] = str(prefix)
    try:
        subprocess.run([str(wine), _windows_path(inject), _windows_path(dll)], env=env, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = " ".join(str(getattr(exc, "stderr", "") or exc).split())[-300:]
        raise AgentNotSent(f"injecting the agent failed: {detail}") from exc
    deadline = time.time() + 10
    while time.time() < deadline:
        if loaded(pid):
            return
        time.sleep(0.1)
    raise AgentNotSent("the agent did not load into the game")


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
    tmp = SEQ.with_name(f"{SEQ.name}.{os.getpid()}.tmp")
    tmp.write_text(str(current))
    os.replace(tmp, SEQ)
    return current


_protocol_lock = threading.RLock()
_lock_depth = 0

# Called on every poll of a wait that comes before sending, with the seconds waited so far: while another
# process holds the lock (first argument lock_holder(), None when unknown) and while an earlier job still runs
# in the game (first argument None; this process holds the lock then). It may raise to give up, preferably
# AgentNotSent: that wait has sent nothing, and the raise leaves no lock or descriptor behind.
on_lock_wait: Callable[[dict | None, float], None] | None = None

# Commands this process has written to CMD, counted immediately before each write: 0 means nothing was
# sent yet, so giving up is free; anything else means a command is or was in flight.
sent_count = 0

# Extra JSON-serializable fields a client adds to its holder record (the CLI's command digest).
# lock_holder() returns them next to pid, argv and since; those three cannot be overridden.
holder_extra: dict = {}


def _lock_timeout() -> float:
    try:
        value = float(os.environ.get("UNCRASHED_LOCK_TIMEOUT") or 60)
    except ValueError:
        return 60.0
    return min(value, 86400.0) if value >= 0 else 60.0


def lock_holder() -> dict | None:
    """What the lock file says about its holder: {pid, argv, since, plus its holder_extra fields}, or None
    when unknown.

    Diagnostic only and never raises. The file may be empty, partial, foreign or left by a killed process,
    and processes running older code hold the lock without writing anything.
    """
    try:
        with open(LOCK, "rb") as file:
            info = json.loads(file.read(65536).decode(errors="replace"))
        pid = info["pid"]
        if type(pid) is not int or pid <= 0 or not Path(f"/proc/{pid}").exists():  # bool and float are no pids
            return None
        argv, since = info.get("argv"), info.get("since")
        return {
            **info,
            "pid": pid,
            "argv": [str(a) for a in argv] if isinstance(argv, list) else [],
            "since": since if isinstance(since, (int, float)) and not isinstance(since, bool) else None,
        }
    except Exception:  # anything a foreign file can provoke (OverflowError, RecursionError, ...)
        return None


def _holder_text(holder: dict | None) -> str:
    if holder is None:
        return "another process"
    argv = holder["argv"]
    if not argv:
        return f"pid {holder['pid']} (?)"
    name = Path(argv[0]).name
    if name in ("cli.py", "__main__.py"):  # `./uc` runs `python -m uncrashed_mcp.cli`: argv[0] is the module file
        name = "uc"
    return f"pid {holder['pid']} ({' '.join([name, *argv[1:]])[:120]})"


def _holder_record() -> str:
    record = {"pid": os.getpid(), "argv": sys.argv, "since": round(time.time(), 3)}
    try:
        return json.dumps({**holder_extra, **record}, default=str)
    except (TypeError, ValueError, RecursionError):
        return json.dumps(record, default=str)


def _set_holder(fd: int, text: str) -> None:
    # In place on the locked descriptor: replacing the file would hand out a second lock.
    try:
        os.ftruncate(fd, 0)
        os.pwrite(fd, text.encode(), 0)
    except OSError:
        pass


def _acquire(started: float, timeout: float) -> int:
    try:
        fd = os.open(LOCK, os.O_RDWR | os.O_CREAT, 0o666)
    except PermissionError:
        fd = os.open(LOCK, os.O_RDONLY)
    try:
        note_at = 2.0
        # Polling, so the wait can report, time out and be given up. The price: a poller is never queued in the
        # kernel, so a holder that re-enters sessions back to back (gaps well under the 50 ms poll; no operation
        # here does that, an older server looping over single commands can) may keep it out until the timeout.
        # That failure is loud and sends nothing.
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                pass
            waited = time.monotonic() - started
            holder = lock_holder()
            if on_lock_wait is not None:
                on_lock_wait(holder, waited)
            if waited >= timeout:
                held = waited if holder is None or holder["since"] is None else max(waited, time.time() - holder["since"])
                raise AgentNotSent(f"agent busy: lock held by {_holder_text(holder)} for {held:.0f}s; nothing was sent; "
                                   f"retry when that process has finished (the lock frees itself when its holder "
                                   f"exits: never delete {LOCK})")
            if waited >= note_at:
                _note(f"waiting for agent lock held by {_holder_text(holder)}")
                note_at += 30.0
            time.sleep(0.05)
        _set_holder(fd, _holder_record())
        return fd
    except BaseException:  # also what on_lock_wait raises: closing the descriptor drops a lock taken meanwhile
        os.close(fd)
        raise


@contextmanager
def session():
    """Serialize complete multi-command operations across MCP/CLI clients."""
    global _lock_depth
    started, timeout = time.monotonic(), _lock_timeout()
    if not _protocol_lock.acquire(timeout=timeout):
        raise AgentNotSent(f"agent busy: lock held by another thread of this process for {timeout:.0f}s; nothing was sent; "
                           "retry when its operation has finished")
    try:
        fd = _acquire(started, timeout) if _lock_depth == 0 else None
        _lock_depth += 1
        try:
            yield
        finally:
            _lock_depth -= 1
            if fd is not None:
                _set_holder(fd, "")
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
    finally:
        _protocol_lock.release()


def _game_started(pid: int) -> float | None:
    """Wall-clock start of a process: /proc starttime (ticks since boot) plus the boot time."""
    try:
        ticks = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        boot = next(int(row.split()[1]) for row in Path("/proc/stat").read_text().splitlines() if row.startswith("btime "))
        return boot + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def _job_running(game_started: float | None) -> bool:
    """The DLL creates OUT + '.tmp' when a job starts and renames it onto OUT when the job ends."""
    try:
        written = OUT.with_name(OUT.name + ".tmp").stat().st_mtime
    except OSError:
        return False
    # Older than the game process (or, start unknown, than 10 minutes): left by a game that crashed mid-job.
    return written >= (time.time() - 600 if game_started is None else game_started)


def _busy_hint() -> str:
    """What to do about a job marker that does not go away. Deleting the marker under a job that still runs
    loses that job's reply (the DLL renames the marker onto OUT), so the advice hangs on the agent log."""
    try:
        last = str(int(SEQ.read_text().strip()))
    except (OSError, ValueError):
        last = "<n>"
    marker = OUT.with_name(OUT.name + ".tmp")
    return (f"if it stays busy read './uc log': a line 'job {last} <verb> <n>ms' near its end means the job has ended "
            f"and {marker} is a leftover that may be deleted; without that line the job still runs or hangs "
            "(wait, or restart the game) and the file must stay")


def _wait_idle() -> None:
    """A job orphaned by a timed-out or killed client still owns the game thread: do not queue behind it."""
    if not OUT.with_name(OUT.name + ".tmp").exists():
        return
    pid = game_pid()
    game_started = _game_started(pid) if pid is not None else None
    started, timeout, noted = time.monotonic(), _lock_timeout(), False
    while _job_running(game_started):
        waited = time.monotonic() - started
        if on_lock_wait is not None:
            on_lock_wait(None, waited)
        if waited >= timeout:
            raise AgentNotSent(f"agent busy: an earlier job is still running; nothing was sent; {_busy_hint()}")
        if not loaded():
            raise AgentNotSent("the agent is not loaded (unloaded, or the game closed); nothing was sent")
        if not noted:
            _note("agent is finishing an earlier job")
            noted = True
        time.sleep(0.05)


def execute(command: str, timeout: float = 30.0) -> dict:
    with session():
        return _execute(command, timeout)


def _execute(command: str, timeout: float = 30.0) -> dict:
    """Run one agent command on the game thread.

    Returns {seq, ok, head, lines, ended}. ok is False when the agent reported an error or
    a caught fault; lines are the reply between the header and the end marker; ended is False
    when the reply has no "end <seq> ..." line at all.

    Raises AgentNotSent while nothing was written, AgentTimeout when no reply came in time, and a
    plain AgentError when the agent went away after the command was written.
    """
    global sent_count
    _refuse_offline()
    command = " ".join(command.split())
    if not command:
        raise AgentNotSent("empty command")
    ensure_agent()
    _wait_idle()
    seq = _next_seq()
    tmp = CMD.with_suffix(".tmp")
    tmp.write_text(f"{seq} {command}\n")
    sent_count += 1  # before the rename: from here on the command counts as in flight
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
            ended = rows[-1].startswith("end ")
            return {"seq": seq, "ok": rows[-1].strip() == f"end {seq} ok", "head": rows[0][len(prefix):],
                    "lines": rows[1:-1] if ended else rows[1:], "ended": ended}
        if not loaded():
            raise AgentError(f"the agent is not loaded (unloaded, or the game closed); {command[:60]!r} was sent "
                             "before that and may have run")
        time.sleep(0.03)
    raise AgentTimeout(f"the agent did not answer {command[:60]!r} within {timeout:.0f}s; the command was sent and may still run")


def _errors(lines: list[str]) -> list[str]:
    return [line.split("error ", 1)[1] for line in lines if "error " in line and line.lstrip("#0123456789 ").startswith("error ")]


def _failure(reply: dict) -> str:
    """Text for a reply that did not end with "end <seq> ok": the agent's error lines, and never the bare
    "ok <verb>" head, which reads like a success."""
    head, lines = reply["head"], reply["lines"]
    errors = ([head[6:]] if head.startswith("error ") else []) + _errors(lines)
    faults = [line for line in lines if re.match(r"\d+ fault ", line)]
    if not reply.get("ended", True):
        # e.g. a fault inside a quiet batch line: the fault and end lines are swallowed with the rest
        cause = ["agent job ended without an end marker (fault?)"] + faults[-1:]
    elif faults:
        cause = [f"agent job faulted: {faults[-1]}"]
    elif errors:
        return "; ".join(errors)
    else:
        return "; ".join(["agent job failed without an error line"] + ([] if head.startswith("ok ") else [head]) + lines[-2:])
    more = [f"{len(errors) - 5} more error lines"] if len(errors) > 5 else []
    return "; ".join(cause + errors[:5] + more)


def run(command: str, timeout: float = 30.0) -> list[str]:
    """execute(), raising AgentError (reply attached) with the agent's own message on failure."""
    reply = execute(command, timeout)
    if not reply["ok"]:
        error = AgentError(_failure(reply))
        error.reply = reply
        raise error
    return reply["lines"]


_batch_serial = itertools.count(1)


def batch_files() -> list[Path]:
    """Every per-call batch file present, of any process."""
    pattern = Path(glob.escape(str(BATCH.parent))) / f"{glob.escape(BATCH.stem)}.*{glob.escape(BATCH.suffix)}"
    return sorted(Path(found) for found in glob.glob(str(pattern)))


def _write_batch(commands: list[str]) -> Path:
    """A file of its own for one batch. The DLL opens the path named in the command only when the job starts,
    which can be long after the command was latched (stalled game thread) and after this client gave up: a
    shared path would by then hold the next client's commands."""
    for old in batch_files():  # left by jobs that never answered; a day covers any job still pending
        try:
            if old.stat().st_mtime < time.time() - BATCH_KEEP:
                old.unlink()
        except OSError:
            pass
    # pid and a per-process counter name the call; the token covers pid reuse against files not yet swept
    path = BATCH.with_name(f"{BATCH.stem}.{os.getpid()}.{next(_batch_serial)}.{secrets.token_hex(4)}{BATCH.suffix}")
    with open(path, "x") as file:
        file.write("\n".join(" ".join(c.split()) for c in commands) + "\n")
    return path


def batch(commands: list[str], timeout: float | None = None, quiet: bool = False) -> tuple[dict[int, list[str]], str]:
    with session():
        return _batch(commands, timeout, quiet)


def _batch(commands: list[str], timeout: float | None = None, quiet: bool = False) -> tuple[dict[int, list[str]], str]:
    """Run many agent commands in one game-thread job. Returns (lines by index, summary).

    quiet makes the agent report only failures, for large jobs whose results are not needed.
    """
    if not commands:
        return {}, "batch lines=0 ok=0 failed=0"
    _refuse_offline()  # before any file is written
    path = _write_batch(commands)
    sent_before = sent_count
    try:
        lines = run(f"batch {_windows_path(path)}{' quiet' if quiet else ''}", timeout or max(60.0, len(commands) * 0.02))
    except AgentNotSent:
        path.unlink(missing_ok=True)  # no job will ever name it
        raise
    except AgentError as exc:
        if exc.reply is not None:
            path.unlink(missing_ok=True)  # the job answered, so it has ended
        raise  # otherwise sent without a reply: the job may still be pending or reading, so the file stays
    except BaseException:  # interrupted (signal, KeyboardInterrupt)
        if sent_count == sent_before:
            path.unlink(missing_ok=True)  # nothing was sent, so no job will ever name it
        raise
    path.unlink(missing_ok=True)
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
        try:
            if map_info().get("base") == base:
                return
        except AgentError:
            pass  # mid-travel: no editor world / pawn for a moment
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

    Uses the editor's own pieces: the menu's ?MapEditor=<base> level open for a different base, Spawn Asset for every
    object, SpawnTrackAsset for the start and checkpoints. The editor's Save button
    then writes back to this map's folder.
    """
    folder = sanitize_name(name)
    doc = load_map(folder)
    started = time.time()
    info = map_info()
    switched = False
    if doc.base and doc.base != info.get("base"):
        # The base is chosen like the game's menu does it: the editor level reopens with ?MapEditor=<row>
        # (native "base"). The map folder is set first; the game instance keeps it across the travel.
        run(f"setmap {_windows_path(maps_dir() / folder)}")
        run(f"base {doc.base}")
        _wait_for_base(doc.base)
        time.sleep(2.0)  # fresh editor world: let BeginPlay and the base streaming settle before clearing
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
    # Absolute, like save: a relative folder resolves against the editor's current directory, which after a
    # game restart is MapEditor/Autosave -> MapEditor/<folder> (no User/): sidecars, journal and Save miss the map.
    commands.append(f"setmap {_windows_path(maps_dir() / folder)}")

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
