"""Game-thread camera control and offscreen images. Never silently activates windows."""
from __future__ import annotations

import contextlib
import json
import math
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

from PIL import Image, ImageStat

from . import agent, reference
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
        xyz = [round(float(v) / 100, 4) for v in fields['loc'].split(',')]
        rot = [round(float(v), 4) for v in fields['rot'].split(',')]
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


def _engine_capture(path: Path, pose: dict, width: int, height: int, ortho_cm: float = 0, only: str = '') -> None:
    with agent.session():
        rw,rh = _render_size(width,height)
        mode = (ortho_cm, only) if ortho_cm or only else ()   # ordinary captures: the old call, the old command
        if (rw,rh) == (width,height):
            _engine_capture_locked(path, pose, width, height, *mode)
            return
        raw = path.with_name(path.stem+'-render.png')
        try:
            _engine_capture_locked(raw, pose, rw, rh, *mode)
            with Image.open(raw) as image:
                image.resize((width,height),Image.Resampling.LANCZOS).save(path)
        finally:
            raw.unlink(missing_ok=True)


class CaptureModeUnsupported(LiveError):
    """The loaded agent DLL predates `capture start ... ortho_cm mode`: nothing was rendered."""


def _engine_capture_locked(path: Path, pose: dict, width: int, height: int, ortho_cm: float = 0,
                           only: str = '') -> None:
    command = (f'capture start {width} {height} '
               f'{pose["x"]*100:.5f} {pose["y"]*100:.5f} {pose["z"]*100:.5f} '
               f'{pose["pitch"]:.5f} {pose["yaw"]:.5f} {pose.get("roll", 0):.5f} {pose.get("fov", 90):.5f}')
    if ortho_cm or only:   # older agents know only the 11-word form: keep it for every ordinary capture
        command += f' {ortho_cm:.3f}' + (f' {only}' if only else '')
    deadline = time.monotonic() + 32
    while True:
        try:
            lines = agent.run(command)
            break
        except agent.AgentError as exc:
            if (ortho_cm or only) and ('token expired or unknown' in str(exc) or 'usage: capture' in str(exc)):
                # an agent built before the projection / mode words reads them as a token command
                raise CaptureModeUnsupported('the loaded agent DLL has no orthographic / reference-only capture '
                                             '(native/camera.h): agent_unload, then the next call injects the '
                                             'rebuilt DLL') from exc
            # A capture whose client died keeps the single native slot until the 30 s watchdog frees it.
            if 'capture busy' not in str(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(1)
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
    except BaseException:
        # Keep the error that is on its way out (a reply timeout says the render was sent and may still run):
        # a stop that fails as well must not replace it. The native watchdog frees the slot after 30 s.
        try:
            agent.run(f'capture stop {token}')
        except Exception:
            pass
        raise
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
                 width: int = 800, height: int = 450, include_reference: bool = False,
                 reference_only: bool = False) -> Path:
    """Capture fresh scene, no UI/focus/camera changes by default.

    Engine backend uses independent render target. Explicit window backend can
    include UI, but occlusion/minimization/other desktops may yield stale pixels.
    Only foreground=True + backend='window' permits activation. No fallback.
    A visible textured reference is hidden for the capture unless include_reference. reference_only renders
    nothing but the loaded reference (engine backend, agent DLL with capture modes): a before picture.
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
    if reference_only and backend != 'engine':
        raise LiveError('a reference-only capture needs the engine backend')
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
    with agent.session(), reference.capture_guard(include_reference or reference_only) as shown, \
            (reference.only_guard(shown) if reference_only else contextlib.nullcontext()):
        if backend == 'engine':
            _engine_capture(path, pose, width, height, *((0, 'reference') if reference_only else ()))
        else:
            _window_capture(path, width, height, foreground)
    path.with_suffix('.json').write_text(json.dumps({
        **shown,
        'image': str(path), 'camera': pose, 'backend': backend, 'foreground': foreground,
        'editor_camera_moved': False, 'width': width, 'height': height,
        **({'only': 'reference'} if reference_only else {}),
        'render_size': _render_size(width,height) if backend == 'engine' else [width,height],
        'freshness': 'new dedicated render target' if backend == 'engine' else 'unverified window pixels',
    }))
    return path


TOPDOWN_FOV = 4.0          # perspective fallback: degrees across the box (camera about 14 x the box width up)


def capture_topdown(box, px: int = 900, ground_z: float = 0.0, top_z: float | None = None, out_dir: Path | None = None,
                    reference_mode: str = 'hide', projection: str = 'auto') -> dict:
    """Straight-down image of box x0,y0,x1,y1 (map metres) on the pixel grid of scan_view (crop_frame): pixel (i, j)
    covers x0 + i * mpp .. east and y0 + j * mpp .. south, north up. projection auto: orthographic (agent DLL with
    capture modes), else a narrow perspective camera far above (stated, with its parallax). reference_mode: hide
    (the textured reference is hidden for the capture), include, only (only the reference is rendered: needs the
    ortho-capable agent DLL)."""
    from .scan.view import crop_frame
    x0, y0, x1, y1 = (float(v) for v in box)
    if not (x1 > x0 and y1 > y0) or max(x1 - x0, y1 - y0) > 2000:
        raise LiveError('box must be x0,y0,x1,y1 with x1 > x0, y1 > y0 and sides up to 2000 m')
    if reference_mode not in ('hide', 'include', 'only'):
        raise LiveError('reference_mode must be hide, include or only')
    if projection not in ('auto', 'ortho', 'perspective'):
        raise LiveError('projection must be auto, ortho or perspective')
    if reference_mode == 'only' and projection == 'perspective':
        raise LiveError('a reference-only capture needs the agent DLL with capture modes (projection ortho or auto)')
    mpp, W, H = crop_frame((x0, y0, x1, y1), int(px))
    if W < 64 or H < 64 or W > 4096 or H > 4096 or W * H > 4194304:
        raise LiveError(f'the crop is {W} x {H} px: give px so that both sides are 64..4096 (box aspect)')
    span_x, span_y = W * mpp, H * mpp
    cx, cy = x0 + span_x / 2, y0 + span_y / 2
    top = ground_z + 60.0 if top_z is None else top_z
    pose_o = {'x': cx, 'y': cy, 'z': top + 40.0, 'pitch': -90.0, 'yaw': -90.0, 'roll': 0.0, 'fov': 90.0}
    height = (span_x / 2) / math.tan(math.radians(TOPDOWN_FOV / 2))
    pose_p = {'x': cx, 'y': cy, 'z': ground_z + height, 'pitch': -90.0, 'yaw': -90.0, 'roll': 0.0, 'fov': TOPDOWN_FOV}
    folder = out_dir or SHOTS
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'topdown-{uuid.uuid4().hex[:12]}.png'
    used, note = None, None
    only = 'reference' if reference_mode == 'only' else ''
    with agent.session(), reference.capture_guard(reference_mode != 'hide') as shown:
        if reference_mode == 'only':
            st = reference.status()
            if not st.get('loaded'):
                raise LiveError('no textured reference loaded (reference_load first)')
            if st.get('hidden'):
                agent.run('reference show')
                shown['reference'] = 'shown for a reference-only capture, hidden again after'
        try:
            if projection in ('auto', 'ortho'):
                try:
                    _engine_capture(path, pose_o, W, H, ortho_cm=span_x * 100, only=only)
                    used = 'orthographic'
                except CaptureModeUnsupported as exc:
                    if projection == 'ortho' or only:
                        raise
                    note = str(exc)
            if used is None:
                _engine_capture(path, pose_p, W, H)
                used = 'perspective'
        finally:
            if reference_mode == 'only' and shown.get('reference', '').startswith('shown for'):
                agent.run('reference hide')
    pose = pose_o if used == 'orthographic' else pose_p
    parallax = None if used == 'orthographic' else round(10.0 * (math.hypot(span_x, span_y) / 2) / height, 3)
    meta = {'image': str(path), 'box': [x0, y0, x0 + span_x, y0 + span_y], 'm_per_px': mpp, 'width': W, 'height': H,
            'projection': used, 'camera': pose, 'reference': shown.get('reference'), 'note': note,
            'parallax_m_per_10m_at_corner': parallax}
    path.with_suffix('.json').write_text(json.dumps(meta))
    return meta
