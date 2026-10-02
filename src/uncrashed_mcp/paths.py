"""Where Uncrashed reads local track-editor maps on this machine."""

from __future__ import annotations

import os
from pathlib import Path

APP_ID = "1682970"


def _profile_roots() -> list[Path]:
    home = Path.home()
    roaming = "pfx/drive_c/users/steamuser/AppData/Roaming/Uncrashed"
    return [
        home / ".local/share/Steam/steamapps/compatdata" / APP_ID / roaming,
        home / ".steam/debian-installation/steamapps/compatdata" / APP_ID / roaming,
        home / "Library/Application Support/Uncrashed",
        home / "AppData/Roaming/Uncrashed",
    ]


def steam_id() -> str:
    """UNCRASHED_STEAM_ID, else the most recently used numeric profile folder the game created."""
    override = os.environ.get("UNCRASHED_STEAM_ID")
    if override:
        return override
    profiles = [
        entry
        for root in _profile_roots()
        if root.is_dir()
        for entry in root.iterdir()
        if entry.is_dir() and entry.name.isdigit()
    ]
    if not profiles:
        return "0"
    return max(profiles, key=lambda entry: entry.stat().st_mtime).name


def maps_dir() -> Path:
    """Folder of <MapName>/<MapName>.json the track editor lists as local maps.

    The running game builds paths under the Steam profile as MapEditor/User.
    On this Proton install that profile is AppData/Roaming/Uncrashed/<steamid>.
    """
    override = os.environ.get("UNCRASHED_MAPS_DIR")
    if override:
        return Path(override).expanduser()
    sid = steam_id()
    candidates = [root / sid / "MapEditor/User" for root in _profile_roots()]
    for path in candidates:
        if path.exists() or path.parent.parent.exists():
            return path
    return candidates[0]


def game_pid() -> int | None:
    proc = Path("/proc")
    if not proc.exists():
        return None
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        # Launchers mention Uncrashed.exe in their arguments. The game itself is the shipping binary.
        if b"Uncrashed-Win64-Shipping.exe" in cmd:
            return int(entry.name)
    return None


def game_running() -> bool:
    return game_pid() is not None


def workshop_dir() -> Path | None:
    path = (
        Path.home()
        / ".local/share/Steam/steamapps/workshop/content"
        / APP_ID
    )
    return path if path.is_dir() else None
