"""Cutout server (services/cutout) against a fake bucket of synthetic MER tiles."""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pytest

aiohttp = pytest.importorskip("aiohttp")
fits = pytest.importorskip("astropy.io.fits")
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "cutout"))
import cutout_server as cs  # noqa: E402

from euclid_polish.config import Config  # noqa: E402

TILE = "102158892"
W, H = 300, 200                       # non-square, so swapped axes would show
MAGZERO = {"VIS": 24.6, "Y_E": 29.8, "J_E": 29.9, "H_E": 30.0}


def _fits_bytes(data: np.ndarray, band: str) -> bytes:
    hdr = fits.Header()
    hdr["EQUINOX"] = 2000.0
    hdr["RADESYS"] = ("ICRS", "Astrometric system")
    hdr["CTYPE1"], hdr["CTYPE2"] = "RA---TAN", "DEC--TAN"
    hdr["CRVAL1"], hdr["CRVAL2"] = 270.93008130, 65.5
    hdr["CRPIX1"], hdr["CRPIX2"] = 150, 100
    hdr["CD1_1"], hdr["CD1_2"], hdr["CD2_1"], hdr["CD2_2"] = -2.7777777778e-05, 0.0, 0.0, 2.7777777778e-05
    hdr["MAGZERO"] = (MAGZERO[band], "AB zeropoint")
    hdr["FILTER"] = ("it's " + band, "quote escape")
    for i in range(60):               # push the header past one 2880-byte block
        hdr[f"PAD{i:03d}"] = (i, "padding card")
    buf = io.BytesIO()
    fits.PrimaryHDU(data.astype(">f4"), header=hdr).writeto(buf)
    return buf.getvalue()


@pytest.fixture(scope="module")
def tile_files():
    rng = np.random.default_rng(3)
    files, arrays = {}, {}
    for band, inst, token in cs.BANDS:
        arrays[band] = rng.normal(0, 1, (H, W)).astype(np.float32)
        key = f"q1/MER/{TILE}/{inst}/EUC_MER_{token}_TILE{TILE}-ABC123_20241025T000000Z_00.00.fits"
        files[key] = _fits_bytes(arrays[band], band)
        decoy = f"q1/MER/{TILE}/{inst}/EUC_MER_{token}-RMS_TILE{TILE}-DEF456_20241025T000000Z_00.00.fits"
        files[decoy] = b"not this one"
    return files, arrays


def _fake_bucket(files: dict[str, bytes]) -> web.Application:
    """S3 stand-in: ListObjectsV2 (2 keys per page, so paging is exercised) and Range GET."""
    async def listing(request):
        prefix = request.query.get("prefix", "")
        keys = sorted(k for k in files if k.startswith(prefix))
        start = int(request.query.get("continuation-token", "0"))
        page = keys[start:start + 2]
        more = start + 2 < len(keys)
        xml = ('<?xml version="1.0" encoding="UTF-8"?>'
               '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
               + "".join(f"<Contents><Key>{escape(k)}</Key></Contents>" for k in page)
               + f"<IsTruncated>{'true' if more else 'false'}</IsTruncated>"
               + (f"<NextContinuationToken>{start + 2}</NextContinuationToken>" if more else "")
               + "</ListBucketResult>")
        return web.Response(text=xml, content_type="application/xml")

    async def get(request):
        body = files.get(request.match_info["key"])
        if body is None:
            return web.Response(status=404)
        lo, hi = request.headers["Range"].removeprefix("bytes=").split("-")
        part = body[int(lo):int(hi) + 1]
        return web.Response(status=206, body=part,
                            headers={"Content-Range": f"bytes {lo}-{int(lo) + len(part) - 1}/{len(body)}"})

    app = web.Application()
    app.router.add_get("/", listing)
    app.router.add_get("/{key:.+}", get)
    return app


def _with_client(tile_files, check):
    files, arrays = tile_files

    async def main():
        async with TestServer(_fake_bucket(files)) as bucket:
            app = cs.create_app(str(bucket.make_url("")), origins=["*"], concurrency=8, max_cutout=128)
            async with TestClient(TestServer(app)) as client:
                await check(client, arrays)

    asyncio.run(main())


def test_interior_cutout_matches_the_fits_pixels(tile_files):
    async def check(client, arrays):
        resp = await client.get("/v1/cutout", params={"tile": TILE, "x0": 37, "y0": 11, "w": 64, "h": 40})
        assert resp.status == 200
        assert resp.headers["Access-Control-Allow-Origin"] == "*"
        assert resp.headers["Content-Type"] == cs.CONTENT_TYPE
        meta, cube = cs.decode_frame(await resp.read())
        assert cube.shape == (4, 40, 64)
        for i, (band, _, _) in enumerate(cs.BANDS):
            np.testing.assert_array_equal(cube[i], arrays[band][11:51, 37:101])
            entry = meta["bands"][i]
            assert entry["band"] == band and entry["magzero"] == MAGZERO[band]
            assert entry["to_electrons"] == pytest.approx(
                10 ** ((Config.get_band(band).sim_zeropoint_e - MAGZERO[band]) / 2.5), rel=1e-12)
        assert (meta["wcs"]["CRPIX1"], meta["wcs"]["CRPIX2"]) == (150 - 37, 100 - 11)
        assert meta["wcs"]["CTYPE1"] == "RA---TAN"

    _with_client(tile_files, check)


def test_window_over_the_tile_edge_is_nan_outside(tile_files):
    async def check(client, arrays):
        resp = await client.get("/v1/cutout", params={"tile": TILE, "x0": W - 20, "y0": -10, "size": 32})
        meta, cube = cs.decode_frame(await resp.read())
        assert (meta["w"], meta["h"]) == (32, 32)
        assert np.isnan(cube[:, :10, :]).all() and np.isnan(cube[:, :, 20:]).all()
        np.testing.assert_array_equal(cube[2, 10:, :20], arrays["J_E"][0:22, W - 20:W])

    _with_client(tile_files, check)


def test_tile_info_lists_the_four_bands(tile_files):
    async def check(client, _arrays):
        resp = await client.get(f"/v1/tile/{TILE}")
        info = await resp.json()
        assert [b["band"] for b in info["bands"]] == ["VIS", "Y_E", "J_E", "H_E"]
        assert all((b["width"], b["height"]) == (W, H) for b in info["bands"])
        assert all("-RMS_" not in b["key"] for b in info["bands"])

    _with_client(tile_files, check)


@pytest.mark.parametrize("params, status", [
    ({"tile": "../etc", "x0": 0, "y0": 0, "size": 8}, 400),
    ({"tile": TILE, "x0": 0, "y0": 0, "size": 129}, 400),        # over max_cutout
    ({"tile": TILE, "x0": "a", "y0": 0, "size": 8}, 400),
    ({"tile": TILE, "x0": W, "y0": 0, "size": 8}, 400),          # no overlap
    ({"tile": "999999999", "x0": 0, "y0": 0, "size": 8}, 404),
])
def test_bad_requests(tile_files, params, status):
    async def check(client, _arrays):
        resp = await client.get("/v1/cutout", params=params)
        assert resp.status == status
        assert resp.headers["Access-Control-Allow-Origin"] == "*"

    _with_client(tile_files, check)


def test_header_parser_reads_astropy_headers(tile_files):
    files, _ = tile_files
    raw = next(v for k, v in files.items() if "BGSUB-MOSAIC-VIS_" in k)
    cards = cs.parse_header_cards(raw)
    assert cards["NAXIS1"] == W and cards["BITPIX"] == -32 and cards["MAGZERO"] == 24.6
    assert cards["FILTER"] == "it's VIS"
    assert cs.parse_header_cards(raw[:2880]) is None      # END is in block 2


def test_server_zeropoints_match_the_band_config():
    assert {b.name: b.sim_zeropoint_e for b in Config.BANDS} == pytest.approx(cs.SIM_ZEROPOINT_E, abs=1e-12)
