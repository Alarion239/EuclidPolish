"""Euclid Q1 MER cutout server for the in-browser super-resolution viewer.

A browser page can read the public Q1 bucket (``nasa-irsa-euclid-q1`` on S3
sends ``Access-Control-Allow-Origin: *`` and honours ``Range``), but a MER
mosaic is one 19200×19200 float32 image stored row by row, so a small
cutout is hundreds of separate 1–4 KB pieces per band. Over HTTP/1.1 from
a browser (6 connections per host) that is ~15 s per 4-band cutout. This
server does those row reads next to the bucket and returns the four bands
of one cutout in a single response.

Endpoints
---------
``GET /health``
    Liveness check.
``GET /v1/tile/{tile}``
    The four band files of a MER tile with their image size, ``MAGZERO``,
    ADU/s → electrons factor and WCS.
``GET /v1/cutout?tile=…&x0=…&y0=…&w=…&h=…``  (or ``size=`` for a square)
    The ``w×h`` pixel window starting at 0-based column ``x0`` and row
    ``y0`` of the tile, in all four bands (VIS, Y_E, J_E, H_E), as the
    binary frame described below. Pixels outside the tile are NaN, so a
    window may hang over the tile edge (the viewer's tile margins do).

Cutout frame (``application/x-euclid-cutout``)::

    b"EPCT"                     4 bytes  magic
    uint32 little-endian        4 bytes  length n of the JSON metadata
    JSON metadata               n bytes  UTF-8
    zero padding                to an 8-byte boundary
    float32 little-endian       4·h·w values, band-major: [band][row][col]

Pixel values are the archive's ADU/s, untouched. Row 0 of a band is FITS
row ``y0 + 1`` (numpy order of the file: the first stored row is the
bottom of the image). The metadata's per-band ``to_electrons`` factor
converts to electrons over the stack the way ``euclid_polish.photometry``
does, ``10^((sim_zeropoint_e − MAGZERO)/2.5)``; ``SIM_ZEROPOINT_E`` below is
pinned to ``euclid_polish.config`` by ``tests/test_cutout_service.py``.

Configuration (environment)
---------------------------
``PORT``                port to listen on (Railway sets it; default 8080)
``EUCLID_BUCKET_URL``   bucket base URL (default: the public Q1 bucket)
``ALLOWED_ORIGINS``     comma-separated CORS origins (default ``*``)
``S3_CONCURRENCY``      simultaneous range requests to the bucket (default 64)
``MAX_CUTOUT``          largest allowed window side in pixels (default 1024)
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import aiohttp
import numpy as np
from aiohttp import web

DEFAULT_BUCKET_URL = "https://nasa-irsa-euclid-q1.s3.us-east-1.amazonaws.com"
RELEASE_PREFIX = "q1/MER"
FITS_BLOCK = 2880
CARD = 80
MAGIC = b"EPCT"
CONTENT_TYPE = "application/x-euclid-cutout"

# (band name, instrument directory, product token in the file name)
BANDS: tuple[tuple[str, str, str], ...] = (
    ("VIS", "VIS", "BGSUB-MOSAIC-VIS"),
    ("Y_E", "NISP", "BGSUB-MOSAIC-NIR-Y"),
    ("J_E", "NISP", "BGSUB-MOSAIC-NIR-J"),
    ("H_E", "NISP", "BGSUB-MOSAIC-NIR-H"),
)

# Magnitude of one electron over the stack, per band: ``BandConfig.sim_zeropoint_e``
# (zeropoint e⁻/s + 2.5·log10 of the stack exposure time). Pinned by a test.
SIM_ZEROPOINT_E: dict[str, float] = {
    "VIS": 34.29662776165843,
    "Y_E": 31.396441190651323,
    "J_E": 31.616441190651326,
    "H_E": 31.566441190651325,
}

WCS_KEYS = ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
            "CD1_1", "CD1_2", "CD2_1", "CD2_2", "RADESYS", "EQUINOX")
# The viewer (static/): the page, its script, the tile index and the ONNX model.
STATIC_DIR = Path(__file__).resolve().parent / "static"
TILE_RE = re.compile(r"^\d{6,12}$")
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


class UpstreamError(Exception):
    """The bucket answered unexpectedly (missing file, bad FITS, HTTP error)."""


@dataclass(frozen=True)
class BandImage:
    """One band file of a tile: where its pixels start and how to read them."""

    band: str
    key: str
    width: int
    height: int
    data_offset: int
    bscale: float
    bzero: float
    magzero: float
    wcs: dict

    @property
    def to_electrons(self) -> float:
        return 10.0 ** ((SIM_ZEROPOINT_E[self.band] - self.magzero) / 2.5)

    def describe(self) -> dict:
        return {"band": self.band, "key": self.key, "width": self.width, "height": self.height,
                "magzero": self.magzero, "to_electrons": self.to_electrons, "wcs": self.wcs}


def parse_header_cards(raw: bytes) -> dict | None:
    """The keyword → value map of a FITS header, or ``None`` if ``raw`` does
    not yet reach the ``END`` card. Only the simple scalar forms the MER
    mosaics use are parsed (logical, integer, float, quoted string)."""
    cards: dict = {}
    for i in range(0, len(raw) - CARD + 1, CARD):
        card = raw[i:i + CARD].decode("ascii", errors="replace")
        key = card[:8].strip()
        if key == "END":
            return cards
        if card[8:10] != "= " or not key:
            continue
        value = card[10:]
        if value.lstrip().startswith("'"):
            text = value.lstrip()[1:]
            end = text.find("'")
            while end != -1 and text[end + 1:end + 2] == "'":
                end = text.find("'", end + 2)
            cards[key] = text[:end].replace("''", "'").rstrip()
            continue
        token = value.split("/", 1)[0].strip()
        if token in ("T", "F"):
            cards[key] = token == "T"
            continue
        try:
            cards[key] = int(token)
        except ValueError:
            try:
                cards[key] = float(token.replace("D", "E"))
            except ValueError:
                cards[key] = token
    return None


class Bucket:
    """Range reads, listings and per-file header cache against the bucket."""

    def __init__(self, session: aiohttp.ClientSession, base_url: str, concurrency: int = 64):
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.limit = asyncio.Semaphore(concurrency)
        self._tiles: OrderedDict[str, dict[str, BandImage]] = OrderedDict()
        self._tile_locks: dict[str, asyncio.Lock] = {}

    async def read(self, key: str, start: int, length: int) -> bytes:
        headers = {"Range": f"bytes={start}-{start + length - 1}"}
        async with self.limit, self.session.get(f"{self.base_url}/{key}", headers=headers) as resp:
            if resp.status not in (200, 206):
                raise UpstreamError(f"GET {key} bytes {start}+{length}: HTTP {resp.status}")
            body = await resp.read()
        if resp.status == 200:          # server ignored Range: take our slice
            body = body[start:start + length]
        if len(body) != length:
            raise UpstreamError(f"GET {key}: wanted {length} bytes, got {len(body)}")
        return body

    async def list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        token = None
        while True:
            params = {"list-type": "2", "prefix": prefix}
            if token:
                params["continuation-token"] = token
            async with self.limit, self.session.get(f"{self.base_url}/", params=params) as resp:
                if resp.status != 200:
                    raise UpstreamError(f"list {prefix}: HTTP {resp.status}")
                root = ElementTree.fromstring(await resp.read())
            keys += [el.text for el in root.iter(f"{S3_NS}Key") if el.text]
            truncated = root.findtext(f"{S3_NS}IsTruncated") == "true"
            token = root.findtext(f"{S3_NS}NextContinuationToken")
            if not truncated or not token:
                return keys

    async def header(self, band: str, key: str) -> BandImage:
        raw = b""
        cards = None
        while cards is None:
            if len(raw) >= 40 * FITS_BLOCK:
                raise UpstreamError(f"{key}: no END card in the first {len(raw)} bytes")
            raw += await self.read(key, len(raw), 2 * FITS_BLOCK)
            cards = parse_header_cards(raw)
        end = next(i for i in range(0, len(raw), CARD) if raw[i:i + 8] == b"END     ")
        data_offset = (end // FITS_BLOCK + 1) * FITS_BLOCK
        if cards.get("NAXIS") != 2 or cards.get("BITPIX") != -32:
            raise UpstreamError(f"{key}: expected a 2-D float32 primary image, got "
                                f"NAXIS={cards.get('NAXIS')} BITPIX={cards.get('BITPIX')}")
        magzero = cards.get("MAGZERO")
        if not isinstance(magzero, (int, float)) or not math.isfinite(magzero):
            raise UpstreamError(f"{key}: no finite MAGZERO")
        return BandImage(
            band=band, key=key, width=int(cards["NAXIS1"]), height=int(cards["NAXIS2"]),
            data_offset=data_offset, bscale=float(cards.get("BSCALE", 1.0)),
            bzero=float(cards.get("BZERO", 0.0)), magzero=float(magzero),
            wcs={k: cards[k] for k in WCS_KEYS if k in cards},
        )

    async def tile(self, tile: str) -> dict[str, BandImage]:
        """The four band images of ``tile`` (cached; 352 tiles fit easily)."""
        if tile in self._tiles:
            self._tiles.move_to_end(tile)
            return self._tiles[tile]
        lock = self._tile_locks.setdefault(tile, asyncio.Lock())
        async with lock:
            if tile in self._tiles:
                return self._tiles[tile]
            listings = {}
            for inst in sorted({inst for _, inst, _ in BANDS}):
                listings[inst] = await self.list_keys(f"{RELEASE_PREFIX}/{tile}/{inst}/")
            keys = {}
            for band, inst, token in BANDS:
                match = [k for k in listings[inst]
                         if k.rsplit("/", 1)[-1].startswith(f"EUC_MER_{token}_TILE{tile}-")
                         and k.endswith(".fits")]
                if len(match) != 1:
                    raise LookupError(f"tile {tile}: {len(match)} {token} files")
                keys[band] = match[0]
            images = await asyncio.gather(*(self.header(b, k) for b, k in keys.items()))
            shapes = {(im.width, im.height) for im in images}
            if len(shapes) != 1:
                raise UpstreamError(f"tile {tile}: bands differ in shape {shapes}")
            self._tiles[tile] = {im.band: im for im in images}
            while len(self._tiles) > 512:
                self._tiles.popitem(last=False)
            return self._tiles[tile]

    async def window(self, image: BandImage, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        """``image[y0:y0+h, x0:x0+w]`` as float32 with NaN outside the image."""
        out = np.full((h, w), np.nan, dtype=np.float32)
        xa, xb = max(x0, 0), min(x0 + w, image.width)
        ya, yb = max(y0, 0), min(y0 + h, image.height)
        if xa >= xb or ya >= yb:
            return out
        n = xb - xa

        async def row(y: int) -> None:
            start = image.data_offset + 4 * (y * image.width + xa)
            values = np.frombuffer(await self.read(image.key, start, 4 * n), dtype=">f4")
            out[y - y0, xa - x0:xb - x0] = values

        await asyncio.gather(*(row(y) for y in range(ya, yb)))
        if image.bscale != 1.0 or image.bzero != 0.0:
            out = out * np.float32(image.bscale) + np.float32(image.bzero)
        return out


BUCKET = web.AppKey("bucket", Bucket)
MAX_CUTOUT = web.AppKey("max_cutout", int)


def encode_frame(meta: dict, cube: np.ndarray) -> bytes:
    head = json.dumps(meta, separators=(",", ":")).encode()
    pad = (-(8 + len(head))) % 8
    return MAGIC + struct.pack("<I", len(head)) + head + b"\0" * pad + cube.astype("<f4").tobytes()


def decode_frame(body: bytes) -> tuple[dict, np.ndarray]:
    """Inverse of :func:`encode_frame` (used by tests and Python clients)."""
    if body[:4] != MAGIC:
        raise ValueError("not a cutout frame")
    (n,) = struct.unpack("<I", body[4:8])
    meta = json.loads(body[8:8 + n])
    start = 8 + n + (-(8 + n)) % 8
    cube = np.frombuffer(body[start:], dtype="<f4").reshape(len(meta["bands"]), meta["h"], meta["w"])
    return meta, cube


def _int_param(request: web.Request, name: str, default: int | None = None) -> int:
    raw = request.query.get(name)
    if raw is None:
        if default is None:
            raise web.HTTPBadRequest(text=f"missing parameter {name}")
        return default
    try:
        return int(raw)
    except ValueError:
        raise web.HTTPBadRequest(text=f"{name} must be an integer") from None


def _tile_param(value: str) -> str:
    if not TILE_RE.match(value or ""):
        raise web.HTTPBadRequest(text="tile must be a MER tile id (digits)")
    return value


async def _tile_images(request: web.Request, tile: str) -> dict[str, BandImage]:
    try:
        return await request.app[BUCKET].tile(tile)
    except LookupError as exc:
        raise web.HTTPNotFound(text=str(exc)) from None
    except (TimeoutError, UpstreamError, aiohttp.ClientError) as exc:
        raise web.HTTPBadGateway(text=f"bucket error: {exc}") from None


async def health(_request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def tile_info(request: web.Request) -> web.Response:
    tile = _tile_param(request.match_info["tile"])
    images = await _tile_images(request, tile)
    return web.json_response(
        {"tile": tile, "bands": [images[b].describe() for b, _, _ in BANDS]},
        headers={"Cache-Control": "public, max-age=86400"},
    )


async def cutout(request: web.Request) -> web.Response:
    tile = _tile_param(request.query.get("tile", ""))
    size = request.query.get("size")
    x0, y0 = _int_param(request, "x0"), _int_param(request, "y0")
    w = _int_param(request, "w", int(size) if size and size.isdigit() else None)
    h = _int_param(request, "h", w)
    limit = request.app[MAX_CUTOUT]
    if not (1 <= w <= limit and 1 <= h <= limit):
        raise web.HTTPBadRequest(text=f"w and h must be 1..{limit}")
    images = await _tile_images(request, tile)
    first = images[BANDS[0][0]]
    if not (-w < x0 < first.width and -h < y0 < first.height):
        raise web.HTTPBadRequest(text="window does not overlap the tile")
    t0 = time.perf_counter()
    try:
        planes = await asyncio.gather(*(request.app[BUCKET].window(images[b], x0, y0, w, h)
                                        for b, _, _ in BANDS))
    except (TimeoutError, UpstreamError, aiohttp.ClientError) as exc:
        raise web.HTTPBadGateway(text=f"bucket error: {exc}") from None
    fetch_ms = round(1000 * (time.perf_counter() - t0), 1)
    meta = {
        "format": 1, "tile": tile, "x0": x0, "y0": y0, "w": w, "h": h,
        "order": "band,row,col", "units": "ADU/s", "fill": "NaN outside the tile",
        "bands": [{"band": images[b].band, "key": images[b].key, "magzero": images[b].magzero,
                   "to_electrons": images[b].to_electrons} for b, _, _ in BANDS],
        # WCS of the cutout itself: the tile's with CRPIX moved to the window.
        "wcs": {**first.wcs, "CRPIX1": first.wcs.get("CRPIX1", 0) - x0,
                "CRPIX2": first.wcs.get("CRPIX2", 0) - y0},
        "fetch_ms": fetch_ms,
    }
    return web.Response(
        body=encode_frame(meta, np.stack(planes)), content_type=CONTENT_TYPE,
        headers={"Cache-Control": "public, max-age=31536000, immutable",
                 "Server-Timing": f"fetch;dur={fetch_ms}"},
    )


def cors_middleware(origins: list[str]):
    def add_cors(request: web.Request, headers) -> None:
        origin = request.headers.get("Origin")
        if "*" in origins:
            headers["Access-Control-Allow-Origin"] = "*"
        elif origin in origins:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Vary"] = "Origin"
        headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        headers["Access-Control-Expose-Headers"] = "Server-Timing"

    @web.middleware
    async def middleware(request: web.Request, handler):
        if request.method == "OPTIONS":
            response = web.Response(status=204)
        else:
            try:
                response = await handler(request)
            except web.HTTPException as exc:
                add_cors(request, exc.headers)   # browsers can read the error text too
                raise
        add_cors(request, response.headers)
        return response

    return middleware


def create_app(bucket_url: str | None = None, *, origins: list[str] | None = None,
               concurrency: int | None = None, max_cutout: int | None = None) -> web.Application:
    bucket_url = bucket_url or os.environ.get("EUCLID_BUCKET_URL", DEFAULT_BUCKET_URL)
    if origins is None:
        origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
    concurrency = concurrency or int(os.environ.get("S3_CONCURRENCY", "64"))
    app = web.Application(middlewares=[isolation_middleware, cors_middleware(origins)])
    app[MAX_CUTOUT] = max_cutout or int(os.environ.get("MAX_CUTOUT", "1024"))

    async def lifecycle(app: web.Application):
        connector = aiohttp.TCPConnector(limit=concurrency, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=60, sock_read=20)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            app[BUCKET] = Bucket(session, bucket_url, concurrency)
            yield

    app.cleanup_ctx.append(lifecycle)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/tile/{tile}", tile_info)
    app.router.add_get("/v1/cutout", cutout)
    if STATIC_DIR.is_dir():
        app.router.add_get("/", viewer_page)
        app.router.add_static("/static/", STATIC_DIR)
    return app


async def viewer_page(_request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "index.html")


@web.middleware
async def isolation_middleware(request: web.Request, handler):
    """Cross-origin isolation for the viewer: multi-threaded WebAssembly (the
    model's CPU fallback) needs SharedArrayBuffer, which needs COOP + COEP.
    ``credentialless`` rather than ``require-corp``: the map (Aladin Lite)
    loads HiPS tiles and assets from CDS/ESA servers that send CORS but not
    CORP. Browsers without credentialless (Safari) are simply not isolated and
    the model's CPU fallback runs single-threaded there."""
    response = await handler(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    if request.path == "/" or request.path.startswith("/static/"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


if __name__ == "__main__":
    web.run_app(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")),
                access_log_format='%a "%r" %s %b %Tf')
