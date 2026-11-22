"""Contact sheet: many engine captures in one agent session, tiled into one labelled PNG.

Imports PIL and live lazily so api stays light. Units: metres and degrees.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
import uuid
from pathlib import Path

from . import agent, managed

SHEETS = Path('/tmp/uncrashed-shots')
MAX_POSES = 60
MAX_COLUMNS = 12
SHEET_PIXELS = 40_000_000  # the assembled sheet; tiles are small by default
LIMITATIONS = [
    'Images only: a sheet does not establish visual quality, clearance, connectivity or a flyable line.',
    'Editor line batchers, the editor pawn and ruler/axis helpers are hidden from captures natively; '
    'gates, gate numbers, the start arch and any race guide stay visible (hiding them needs a native change).',
    'Each pose is a separate render-target job (start, 3 renders, export, stop) at the editor camera FOV, roll 0.',
    'Runtime is roughly 1.5 s per pose (a 60-pose sheet takes about 90 s); the agent lock is taken per pose, so '
    'other clients can interleave and the scene may change between frames.',
]
_LABEL = re.compile(r'[^A-Za-z0-9_.-]+')


def parse_poses(text: str) -> list[dict]:
    """Poses from 'label x y z yaw pitch' items (label optional), separated by newlines, ';' or ','.

    '@path' reads the items from a text file (one pose per line, '#' comments)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('poses: give at least one "label x y z yaw pitch" pose')
    text = text.strip()
    if text.startswith('@'):
        name = os.path.expanduser(text[1:])
        try:
            text = Path(name).read_text(encoding='utf-8-sig')
        except OSError as exc:
            raise ValueError(f'poses: cannot read {name!r} ({exc.strerror or exc})') from None
        except UnicodeDecodeError:
            raise ValueError(f'poses: {name!r} is not UTF-8 text') from None
    text = '\n'.join(line.split('#', 1)[0] for line in text.splitlines())  # comments end at the line end
    items = [item.strip() for item in re.split(r'[\n;,]', text)]
    items = [item for item in items if item]
    if not items:
        raise ValueError('poses: no poses found')
    if len(items) > MAX_POSES:
        raise ValueError(f'poses: at most {MAX_POSES} poses per sheet, got {len(items)}')
    poses, seen = [], set()
    for index, item in enumerate(items, 1):
        words = item.split()
        label = f'p{index}'
        if len(words) == 6:
            label, words = words[0], words[1:]
        if len(words) != 5:
            raise ValueError(f'poses: item {index} {item!r} must be "label x y z yaw pitch" (label optional)')
        try:
            x, y, z, yaw, pitch = (float(w) for w in words)
        except ValueError:
            raise ValueError(f'poses: item {index} {item!r} has a non-numeric coordinate') from None
        if not all(math.isfinite(v) and abs(v) <= 1e6 for v in (x, y, z, yaw, pitch)):
            raise ValueError(f'poses: item {index} {item!r} needs finite values within 1e6')
        safe = _LABEL.sub('_', label)[:40] or f'p{index}'
        if safe in seen:
            safe = f'{safe}_{index}'
        seen.add(safe)
        poses.append({'label': safe, 'x': x, 'y': y, 'z': z, 'yaw': yaw, 'pitch': pitch})
    return poses


def _validate(width: int, height: int, columns: int, count: int) -> int:
    if not isinstance(width, int) or not isinstance(height, int) or not 64 <= width <= 1920 or not 64 <= height <= 1080:
        raise ValueError('width must be 64..1920 and height 64..1080 per frame')
    if not isinstance(columns, int) or not 0 <= columns <= MAX_COLUMNS:
        raise ValueError(f'columns must be 0 (automatic) to {MAX_COLUMNS}')
    columns = min(columns or math.ceil(math.sqrt(count)), count)
    rows = math.ceil(count / columns)
    if columns * width * rows * (height + _strip(height)) > SHEET_PIXELS:
        raise ValueError(f'sheet would exceed {SHEET_PIXELS:,} pixels; use smaller frames or fewer poses')
    return columns


def _strip(height: int) -> int:
    return max(14, height // 14)


def _target(out: str) -> Path:
    """The target folder, refused before any agent call if it is a file or already holds anything
    (reusing a folder would mix stale and fresh frames). Not created here."""
    folder = Path(os.path.abspath(os.path.expanduser(out))) if out else SHEETS / f'sheet-{uuid.uuid4().hex[:12]}'
    if folder.exists() and not folder.is_dir():
        raise ValueError(f'out must be a directory, got the file {folder}')
    if folder.is_dir() and any(folder.iterdir()):
        raise ValueError(f'out: {folder} is not empty; give a new or empty folder so old and new frames '
                         'cannot mix (nothing was captured)')
    return folder


def _folder(folder: Path) -> Path:
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise type(exc)(f'out: cannot use {folder} as the target folder ({exc.strerror or exc}); '
                        'nothing was captured') from exc
    if not os.access(folder, os.W_OK | os.X_OK):
        raise PermissionError(f'out: no permission to write into {folder}; nothing was captured')
    return folder


def _compose(shots: list[dict], width: int, height: int, columns: int, path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    strip = _strip(height)
    rows = math.ceil(len(shots) / columns)
    sheet = Image.new('RGB', (columns * width, rows * (height + strip)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, shot in enumerate(shots):
        left, top = (index % columns) * width, (index // columns) * (height + strip)
        if shot['image']:
            with Image.open(shot['image']) as tile:
                sheet.paste(tile.convert('RGB').resize((width, height)), (left, top + strip))
        else:
            draw.rectangle((left, top + strip, left + width - 1, top + strip + height - 1), fill=(70, 20, 20))
            draw.text((left + 6, top + strip + 6), 'FAILED', fill=(255, 200, 200), font=font)
        caption = f"{index + 1} {shot['label']}  {shot['x']:g},{shot['y']:g},{shot['z']:g} y{shot['yaw']:g} p{shot['pitch']:g}"
        draw.text((left + 4, top + 2), caption, fill=(255, 255, 255), font=font)
        draw.rectangle((left, top, left + width - 1, top + strip + height - 1), outline=(90, 90, 90))
    sheet.save(path)


def capture_sheet(poses: str, width: int = 480, height: int = 270, columns: int = 0, out: str = '') -> dict:
    """Validate, capture every pose inside one agent session, tile, write originals + index.json."""
    from . import live
    parsed = parse_poses(poses)
    columns = _validate(width, height, columns, len(parsed))
    folder = _target(out)
    started = time.monotonic()
    shots = []
    with agent.session():  # short: other clients are not starved during a long sheet
        camera = live.read_camera()  # FOV only; the editor camera is not moved
    fov = float(camera['fov'])
    SHEETS.mkdir(parents=True, exist_ok=True)
    for index, pose in enumerate(parsed, 1):
        shot = {**pose, 'image': None, 'seconds': None, 'error': None}
        began = time.monotonic()
        raw = SHEETS / f'sheet-{uuid.uuid4().hex}.png'
        try:
            with agent.session():  # one lock hold per pose
                live._engine_capture(raw, {**pose, 'roll': 0.0, 'fov': fov}, width, height)
        except agent.AgentNotSent:
            raw.unlink(missing_ok=True)
            if not any(s['image'] for s in shots):
                raise  # nothing captured and nothing sent: the plain transport error is the clearer one
            shot['error'] = 'not sent: agent unavailable or lock busy'
        except Exception as exc:  # one bad pose must not cost the others
            raw.unlink(missing_ok=True)
            shot['error'] = ' '.join(str(exc).split())[:200] or type(exc).__name__
        else:
            _folder(folder)  # lazily, just before the first move: a refused call leaves no empty folder
            dest = folder / f'{index:02d}-{pose["label"]}.png'
            shutil.move(str(raw), str(dest))
            shot['image'] = str(dest)
        shot['seconds'] = round(time.monotonic() - began, 2)
        shots.append(shot)
    capture_seconds = round(time.monotonic() - started, 2)
    _folder(folder)
    sheet_path = folder / 'sheet.png'
    _compose(shots, width, height, columns, sheet_path)
    failed = [s for s in shots if s['error']]
    index_path = folder / 'index.json'
    result = {
        'sheet': str(sheet_path), 'directory': str(folder), 'index_json': str(index_path),
        'units': 'metres', 'poses': len(shots), 'captured': len(shots) - len(failed), 'failed_count': len(failed),
        'width': width, 'height': height, 'columns': columns, 'fov': fov, 'roll': 0.0,
        'capture_seconds': capture_seconds, 'seconds': round(time.monotonic() - started, 2),
        'shots': [{'label': s['label'], 'x': s['x'], 'y': s['y'], 'z': s['z'], 'yaw': s['yaw'], 'pitch': s['pitch'],
                   'image': s['image'] and os.path.basename(s['image']), 'seconds': s['seconds'], 'error': s['error']}
                  for s in shots],
        'editor_camera_moved': False,
        'limitations': LIMITATIONS,
    }
    full = dict(result, shots=shots)
    managed.atomic_write(index_path, json.dumps(full, indent=1))
    return result
