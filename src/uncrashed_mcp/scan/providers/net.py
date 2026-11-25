"""HTTP for the data providers. Requests carry a generic User-Agent and only what a public server needs to answer
a bounding-box question: a bbox, a tile name or a byte range of a public file. Nothing about the scan is sent.

UNCRASHED_PROVIDERS=0 switches every download off (cached answers are still used).
"""
from __future__ import annotations

import http.client
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

AGENT = {'User-Agent': 'uncrashed-mcp-scan/1'}
PROBE_S = 3.0
PROBE_TRIES_S = (PROBE_S, 10.0, 20.0)   # connect timeouts of successive probes
TIMEOUT_S = 60
LOG: list[dict] = []     # every request of this process: url, bytes, seconds (for timings and tests)


class NetError(RuntimeError):
    pass


def disabled() -> str | None:
    if os.environ.get('UNCRASHED_PROVIDERS', '').strip().lower() in ('0', 'off', 'no', 'false'):
        return 'downloads disabled by UNCRASHED_PROVIDERS=0'
    return None


def _check(url: str) -> None:
    why = disabled()
    if why:
        raise NetError(why)
    u = urllib.parse.urlparse(url)
    last = None
    for timeout in PROBE_TRIES_S:          # one slow answer must not drop a whole data set
        try:
            socket.create_connection((u.hostname, u.port or (443 if u.scheme == 'https' else 80)), timeout=timeout).close()
            return
        except OSError as exc:
            last = exc
    raise NetError(f'{u.netloc}: no connection in {len(PROBE_TRIES_S)} tries (last waited {PROBE_TRIES_S[-1]:g} s; '
                   f'{" ".join(str(last).split())[:60]})')


def _open(req: urllib.request.Request, timeout: float):
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise NetError(f'{urllib.parse.urlparse(req.full_url).netloc}: HTTP {exc.code}') from None
    except (urllib.error.URLError, OSError) as exc:
        raise NetError(f'{urllib.parse.urlparse(req.full_url).netloc}: {" ".join(str(exc).split())[:80]}') from None


def _read(r, url: str) -> bytes:
    """The body; a connection that breaks mid-body (IncompleteRead, reset, timeout) is a NetError."""
    try:
        return r.read()
    except (http.client.HTTPException, OSError) as exc:
        raise NetError(f'{urllib.parse.urlparse(url).netloc}: body read failed '
                       f'({type(exc).__name__}: {" ".join(str(exc).split())[:60]})') from None


def get(url: str, timeout: float = TIMEOUT_S, data: bytes | None = None, headers: dict | None = None) -> bytes:
    _check(url)
    t0 = time.time()
    with _open(urllib.request.Request(url, data=data, headers={**AGENT, **(headers or {})}), timeout) as r:
        body = _read(r, url)
    LOG.append({'url': url, 'bytes': len(body), 's': round(time.time() - t0, 2)})
    return body


def get_range(url: str, off: int, n: int, allow_short: bool = False, timeout: float = TIMEOUT_S) -> bytes:
    _check(url)
    t0 = time.time()
    req = urllib.request.Request(url, headers={**AGENT, 'Range': f'bytes={off}-{off + n - 1}'})
    with _open(req, timeout) as r:
        body = _read(r, url)
        status = r.status
    if status == 200 and off:      # the server ignored the range
        body = body[off:off + n]
    if len(body) < n and not allow_short:
        raise NetError(f'{urllib.parse.urlparse(url).netloc}: short read ({len(body)} of {n} bytes)')
    LOG.append({'url': url, 'bytes': len(body), 's': round(time.time() - t0, 2), 'range': [off, n]})
    return body


def exists(url: str, timeout: float = 20) -> bool:
    """HEAD request: True for 200, False for 403/404 (public buckets answer 403 for missing keys)."""
    _check(url)
    req = urllib.request.Request(url, method='HEAD', headers=AGENT)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ok = r.status == 200
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            ok = False
        else:
            raise NetError(f'{urllib.parse.urlparse(url).netloc}: HTTP {exc.code}') from None
    except (urllib.error.URLError, OSError) as exc:
        raise NetError(f'{urllib.parse.urlparse(url).netloc}: {" ".join(str(exc).split())[:80]}') from None
    LOG.append({'url': url, 'bytes': 0, 's': 0.0, 'head': ok})
    return ok
