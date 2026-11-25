"""A small GeoTIFF reader: classic and BigTIFF, strips or tiles, no / LZW / Deflate compression, horizontal and
floating-point predictors, single band. Reads a pixel window from a local file or over HTTP range requests (a
cloud-optimised GeoTIFF is fetched tile by tile: only the tiles of the window are requested).
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

from .. import native
from . import net

_TYPES = {1: 'B', 2: 's', 3: 'H', 4: 'I', 5: 'II', 6: 'b', 7: 'B', 8: 'h', 9: 'i', 10: 'ii', 11: 'f', 12: 'd',
          16: 'Q', 17: 'q', 18: 'Q'}
_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8, 17: 8, 18: 8}


class TiffError(ValueError):
    pass


# what a truncated or garbled file raises inside the decoders (struct, zlib, LZW, indexing)
_BROKEN = (struct.error, zlib.error, IndexError, EOFError, OverflowError, native.NativeError)


class _Bytes:
    def __init__(self, data: bytes):
        self.data = data

    def read(self, off: int, n: int) -> bytes:
        return self.data[off:off + n]


class _Http:
    """Range reads with a small block cache (the header and IFD arrive in one request)."""

    def __init__(self, url: str, block: int = 65536):
        self.url, self.block, self.cache = url, block, {}

    def read(self, off: int, n: int) -> bytes:
        if n > self.block:
            return net.get_range(self.url, off, n)
        b0 = off // self.block
        out = b''
        for b in range(b0, (off + n - 1) // self.block + 1):
            if b not in self.cache:
                self.cache[b] = net.get_range(self.url, b * self.block, self.block, allow_short=True)
            out += self.cache[b]
        start = off - b0 * self.block
        return out[start:start + n]


@dataclass
class Tiff:
    src: object
    width: int
    height: int
    bits: int
    fmt: int                      # SampleFormat: 1 uint, 2 int, 3 float
    compression: int
    predictor: int
    tile: tuple[int, int] | None  # (w, h) or None for strips
    rows_per_strip: int
    offsets: list[int]
    counts: list[int]
    scale: tuple[float, float] | None = None
    tie: tuple[float, float, float, float] | None = None   # i, j, x, y
    epsg: int | None = None
    pixel_is_point: bool = False
    nodata: float | None = None
    endian: str = '<'
    extra: dict = field(default_factory=dict)

    @property
    def dtype(self) -> np.dtype:
        kind = {1: 'u', 2: 'i', 3: 'f'}.get(self.fmt, 'u')
        return np.dtype(f'{self.endian}{kind}{self.bits // 8}')

    def transform(self) -> tuple[float, float, float, float]:
        """(x of column 0 edge, y of row 0 edge, pixel width, pixel height) in CRS units; rows run south."""
        if not self.scale or not self.tie:
            raise TiffError('no georeference (ModelPixelScale / ModelTiepoint) in the GeoTIFF')
        i, j, x, y = self.tie
        sx, sy = self.scale
        x0, y0 = x - i * sx, y + j * sy
        if self.pixel_is_point:   # the tie point is the centre of pixel (i, j)
            x0, y0 = x0 - sx / 2, y0 + sy / 2
        return x0, y0, sx, sy

    def window(self, c0: int, r0: int, c1: int, r1: int) -> np.ndarray:
        """Pixels [r0:r1, c0:c1] (clipped to the image), decoded."""
        c0, r0 = max(0, c0), max(0, r0)
        c1, r1 = min(self.width, c1), min(self.height, r1)
        if c1 <= c0 or r1 <= r0:
            return np.zeros((0, 0), self.dtype)
        out = np.zeros((r1 - r0, c1 - c0), self.dtype)
        if self.tile:
            tw, th = self.tile
            across = -(-self.width // tw)
            for ty in range(r0 // th, (r1 - 1) // th + 1):
                for tx in range(c0 // tw, (c1 - 1) // tw + 1):
                    k = ty * across + tx
                    block = self._block(k, tw, th)
                    y0, x0 = ty * th, tx * tw
                    ys, xs = slice(max(r0, y0), min(r1, y0 + th)), slice(max(c0, x0), min(c1, x0 + tw))
                    out[ys.start - r0:ys.stop - r0, xs.start - c0:xs.stop - c0] = \
                        block[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
        else:
            rps = self.rows_per_strip
            for k in range(r0 // rps, (r1 - 1) // rps + 1):
                rows = min(rps, self.height - k * rps)
                block = self._block(k, self.width, rows)
                y0 = k * rps
                ys = slice(max(r0, y0), min(r1, y0 + rows))
                out[ys.start - r0:ys.stop - r0] = block[ys.start - y0:ys.stop - y0, c0:c1]
        return out

    def _block(self, k: int, w: int, h: int) -> np.ndarray:
        try:
            return self._decode(k, w, h)
        except _BROKEN as exc:
            raise TiffError(f'block {k} is damaged ({type(exc).__name__}: {exc})') from None

    def _decode(self, k: int, w: int, h: int) -> np.ndarray:
        if k >= len(self.offsets):
            raise TiffError(f'block {k} outside the image')
        size = w * h * self.bits // 8
        if self.counts[k] == 0:
            return np.zeros((h, w), self.dtype)
        raw = self.src.read(self.offsets[k], self.counts[k])
        if self.compression == 1:
            data = raw
        elif self.compression == 5:
            data = native.lzw(raw, size)
        elif self.compression in (8, 32946):
            data = zlib.decompress(raw)
        else:
            raise TiffError(f'unsupported TIFF compression {self.compression}')
        if len(data) < size:
            data += b'\0' * (size - len(data))
        a = np.frombuffer(data[:size], np.uint8)
        if self.predictor == 3:   # floating point: byte planes, differenced
            nb = self.bits // 8
            a = np.cumsum(a.reshape(h, w * nb), axis=1, dtype=np.uint8)
            a = a.reshape(h, nb, w).transpose(0, 2, 1)[:, :, ::-1].copy()   # big-endian planes -> little-endian
            return a.reshape(h, w * nb).view(self.dtype.newbyteorder('<')).reshape(h, w).astype(self.dtype)
        arr = a.view(self.dtype).reshape(h, w)
        if self.predictor == 2:
            arr = np.cumsum(arr, axis=1, dtype=self.dtype)
        return arr


def open_bytes(data: bytes) -> Tiff:
    return _parse(_Bytes(data))


def open_url(url: str) -> Tiff:
    return _parse(_Http(url))


def _parse(src) -> Tiff:
    try:
        return _parse_raw(src)
    except _BROKEN as exc:
        raise TiffError(f'damaged TIFF header ({type(exc).__name__}: {exc})') from None


def _parse_raw(src) -> Tiff:
    head = src.read(0, 16)
    if head[:2] == b'II':
        e = '<'
    elif head[:2] == b'MM':
        e = '>'
    else:
        raise TiffError('not a TIFF file')
    magic = struct.unpack(e + 'H', head[2:4])[0]
    big = magic == 43
    if magic not in (42, 43):
        raise TiffError('not a TIFF file')
    ifd = struct.unpack(e + 'Q', head[8:16])[0] if big else struct.unpack(e + 'I', head[4:8])[0]
    cnt_fmt, ent, cnt_n = (e + 'Q', 20, 8) if big else (e + 'H', 12, 2)
    n = struct.unpack(cnt_fmt, src.read(ifd, cnt_n))[0]
    body = src.read(ifd + cnt_n, n * ent)
    tags = {}
    for i in range(n):
        b = body[i * ent:(i + 1) * ent]
        tag, typ = struct.unpack(e + 'HH', b[:4])
        count = struct.unpack(e + ('Q' if big else 'I'), b[4:12] if big else b[4:8])[0]
        inline = b[12:20] if big else b[8:12]
        size = _SIZE.get(typ, 1) * count
        raw = inline[:size] if size <= len(inline) else src.read(
            struct.unpack(e + ('Q' if big else 'I'), inline)[0], size)
        if typ == 2:
            tags[tag] = raw.rstrip(b'\0').decode('latin-1')
        elif typ in _TYPES:
            code = _TYPES[typ]
            k = len(code)
            vals = struct.unpack(e + code * count, raw[:size])
            if typ in (5, 10):
                vals = tuple(vals[j] / vals[j + 1] if vals[j + 1] else 0.0 for j in range(0, len(vals), 2))
            del k
            tags[tag] = vals
    def one(t, d=None):  # noqa: E306
        v = tags.get(t)
        return v[0] if isinstance(v, tuple) and v else (v if v is not None else d)
    spp = one(277, 1)
    if spp != 1:
        raise TiffError(f'{spp} samples per pixel: only single-band GeoTIFFs are read')
    tiled = 322 in tags
    t = Tiff(src=src, width=int(one(256)), height=int(one(257)), bits=int(one(258, 8)), fmt=int(one(339, 1)),
             compression=int(one(259, 1)), predictor=int(one(317, 1)),
             tile=(int(one(322)), int(one(323))) if tiled else None,
             rows_per_strip=int(one(278, one(257))),
             offsets=list(tags.get(324 if tiled else 273, ())), counts=list(tags.get(325 if tiled else 279, ())),
             endian=e)
    if 33550 in tags:
        t.scale = (float(tags[33550][0]), float(tags[33550][1]))
    if 33922 in tags:
        tp = tags[33922]
        t.tie = (float(tp[0]), float(tp[1]), float(tp[3]), float(tp[4]))
    keys = tags.get(34735)
    if keys:
        for k in range(4, 4 + 4 * int(keys[3]), 4):
            kid, loc, _cnt, val = keys[k:k + 4]
            if loc != 0:
                continue
            if kid == 3072 and val not in (0, 32767):
                t.epsg = int(val)
            elif kid == 2048 and val not in (0, 32767) and t.epsg is None:
                t.epsg = int(val)
            if kid == 1025:
                t.pixel_is_point = val == 2
    nd = tags.get(42113)
    if isinstance(nd, str) and nd.strip():
        try:
            t.nodata = float(nd)
        except ValueError:
            pass
    return t
