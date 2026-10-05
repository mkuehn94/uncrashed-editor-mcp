"""Game-thread camera control and offscreen images. Never silently activates windows."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

from PIL import Image, ImageStat

from . import agent
from .paths import game_pid

SHOTS = Path('/tmp/uncrashed-shots')


class LiveError(RuntimeError):
    pass


def _finite(*values: float) -> None:
    if not all(math.isfinite(v) for v in values):
        raise LiveError('pose must contain finite numbers')


def read_camera() -> dict:
    """Current reflected component pose, metres/degrees; no /proc memory writes."""
    for line in agent.run('camera get'):
        if not line.startswith('camera '):
            continue
        fields = dict(re.findall(r'(\w+)=(\S+)', line))
        xyz = [float(v) / 100 for v in fields['loc'].split(',')]
        rot = [float(v) for v in fields['rot'].split(',')]
        return dict(zip(('x', 'y', 'z', 'pitch', 'yaw', 'roll'), xyz + rot),
                    fov=float(fields['fov']), held=fields['held'] == '1', backend='native')
    raise LiveError('native camera reply missing; rebuild/reload native agent')


def move_camera(x: float, y: float, z: float, yaw: float, pitch: float, hold: bool = True) -> dict:
    """Set actual root/SpringArm/component transforms on game thread. Never focuses."""
    _finite(x, y, z, yaw, pitch)
    agent.run(f'camera set {x*100:.5f} {y*100:.5f} {z*100:.5f} {pitch:.5f} {yaw:.5f} {int(hold)}')
    return read_camera()


def release_camera() -> None:
    """Restore editor's previous movement-enable state. No helper process exists."""
    agent.run('camera release')


def _display_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault('DISPLAY', ':0')
    env.setdefault('XAUTHORITY', str(Path.home() / '.Xauthority'))
    return env


def _xdo(*args: str) -> str:
    try:
        return subprocess.check_output(['xdotool', *args], env=_display_env(), text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ''


def game_window() -> str:
    """Find game's own X window, excluding terminals/Proton helper windows."""
    candidates: list[str] = []
    pid = game_pid()
    if pid is not None:
        candidates += _xdo('search', '--pid', str(pid)).split()
    candidates += _xdo('search', '--class', 'steam_app_1682970').split()
    candidates += _xdo('search', '--name', '^Uncrashed').split()
    best, best_area = None, 0
    for wid in dict.fromkeys(candidates):
        name = _xdo('getwindowname', wid).lower()
        if any(s in name for s in ('python', 'uncrashed-mcp', '/', 'terminal')):
            continue
        size = [p for p in _xdo('getwindowgeometry', wid).split() if 'x' in p and p[0].isdigit()]
        if not size:
            continue
        w, h = (int(v) for v in size[-1].split('x'))
        if w >= 200 and h >= 200 and w*h > best_area:
            best, best_area = wid, w*h
    if best is None:
        raise LiveError('Uncrashed game window was not found')
    return best


def _validate_image(path: Path, width: int, height: int, engine: bool) -> None:
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != (width, height) and engine:
                raise LiveError(f'render target size mismatch: {image.size}')
            # Uniform frames normally mean rendering is suspended. Do not label
            # black/clear frames successful. Nonuniformity cannot prove freshness
            # for window capture; its limitations remain explicit in metadata.
            stats = ImageStat.Stat(image.convert('RGB'))
            if max(stats.stddev) < .1:
                raise LiveError('capture is uniform/blank; rendering may be suspended')
    except (OSError, ValueError) as exc:
        raise LiveError('capture missing or invalid; no foreground fallback attempted') from exc


def _render_size(width: int, height: int) -> tuple[int, int]:
    # Scene-capture temporal AA differs from viewport. Spatial supersampling
    # reduces foliage dither/edge noise without changing global game settings.
    scale = 2 if width*height*4 <= 4194304 and max(width,height)*2 <= 4096 else 1
    return width*scale, height*scale


def _engine_capture(path: Path, pose: dict, width: int, height: int) -> None:
    with agent.session():
        rw,rh = _render_size(width,height)
        if (rw,rh) == (width,height):
            _engine_capture_locked(path, pose, width, height)
            return
        raw = path.with_name(path.stem+'-render.png')
        try:
            _engine_capture_locked(raw, pose, rw, rh)
            with Image.open(raw) as image:
                image.resize((width,height),Image.Resampling.LANCZOS).save(path)
        finally:
            raw.unlink(missing_ok=True)


def _engine_capture_locked(path: Path, pose: dict, width: int, height: int) -> None:
    command = (f'capture start {width} {height} '
               f'{pose["x"]*100:.5f} {pose["y"]*100:.5f} {pose["z"]*100:.5f} '
               f'{pose["pitch"]:.5f} {pose["yaw"]:.5f} {pose.get("roll", 0):.5f} {pose.get("fov", 90):.5f}')
    lines = agent.run(command)
    token = next((m[1] for line in lines if (m := re.search(r'^capture token=(\d+)\b', line))), None)
    if token is None:
        raise LiveError('capture job token missing; native agent update needed')
    try:
        # Independent view needs several frames for temporal AA/exposure history.
        # Explicit renders work even if the viewport does not redraw in background.
        for _ in range(3):
            time.sleep(.12)
            agent.run(f'capture render {token}')
        time.sleep(.15)
        agent.run(f'capture export {token} {agent._windows_path(path.parent)} {path.name}', timeout=30)
        _validate_image(path, width, height, True)
    finally:
        agent.run(f'capture stop {token}')


def _window_capture(path: Path, width: int, height: int, foreground: bool) -> None:
    wid = game_window()
    previous = _xdo('getactivewindow') if foreground else ''
    try:
        if foreground and previous != wid:
            _xdo('windowactivate', '--sync', wid)
            time.sleep(.35)
        subprocess.run(['import', '-window', wid, '-resize', f'{width}x{height}!', str(path)],
                       env=_display_env(), check=True, capture_output=True)
    finally:
        if foreground and previous and previous != wid:
            _xdo('windowactivate', previous)
    _validate_image(path, width, height, False)


def capture_view(x: float | None = None, y: float | None = None, z: float | None = None,
                 yaw: float | None = None, pitch: float | None = None, *,
                 backend: str = 'engine', foreground: bool = False,
                 width: int = 800, height: int = 450) -> Path:
    """Capture fresh scene, no UI/focus/camera changes by default.

    Engine backend uses independent render target. Explicit window backend can
    include UI, but occlusion/minimization/other desktops may yield stale pixels.
    Only foreground=True + backend='window' permits activation. No fallback.
    """
    supplied = (x is not None, y is not None, z is not None)
    if any(supplied) and not all(supplied):
        raise LiveError('pass x, y and z together')
    if (yaw is not None or pitch is not None) and not all(supplied):
        raise LiveError('pass x, y and z when setting the look')
    if backend not in ('engine', 'window'):
        raise LiveError('backend must be engine or window')
    if foreground and backend != 'window':
        raise LiveError('foreground is only supported by explicit window backend')
    if backend == 'window' and any(supplied):
        raise LiveError('window backend captures current editor view; use move_camera first')
    if (not isinstance(width, int) or not isinstance(height, int) or
            not 64 <= width <= 4096 or not 64 <= height <= 4096 or width*height > 4194304):
        raise LiveError('capture size must be 64..4096 per axis and at most 4,194,304 pixels')
    pose = read_camera()
    if all(supplied):
        pose.update(x=x, y=y, z=z, yaw=90.0 if yaw is None else yaw,
                    pitch=-20.0 if pitch is None else pitch, roll=0.0)
    _finite(*(pose[k] for k in ('x', 'y', 'z', 'yaw', 'pitch', 'roll', 'fov')))
    SHOTS.mkdir(parents=True, exist_ok=True)
    path = SHOTS / f'view-{uuid.uuid4().hex}.png'
    if backend == 'engine':
        _engine_capture(path, pose, width, height)
    else:
        _window_capture(path, width, height, foreground)
    path.with_suffix('.json').write_text(json.dumps({
        'image': str(path), 'camera': pose, 'backend': backend, 'foreground': foreground,
        'editor_camera_moved': False, 'width': width, 'height': height,
        'render_size': _render_size(width,height) if backend == 'engine' else [width,height],
        'freshness': 'new dedicated render target' if backend == 'engine' else 'unverified window pixels',
    }))
    return path
