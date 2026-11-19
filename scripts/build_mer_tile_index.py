"""Build the viewer's MER tile index for one Euclid release from the IRSA bucket.

    python scripts/build_mer_tile_index.py dr1     # → services/cutout/index/dr1.json
    python scripts/build_mer_tile_index.py q1      # → services/cutout/index/q1.json

For every MER tile it reads the FITS header of the VIS background-subtracted
mosaic (two 2880-byte blocks by HTTP range request) and records the tile's
exact TAN geometry: CRVAL (the tile centre), image size and CRPIX. The viewer
and the server use it to find the tile that holds a sky position deepest
(the tile arrays overlap) and to outline tiles on the map.

Output::

    {"release": "dr1", "dataset": "DR1_R1", "built": "…", "bucket": "…",
     "cd": [CD1_1, CD1_2, CD2_1, CD2_2],          # identical for every tile
     "columns": ["tile", "ra", "dec", "naxis1", "naxis2", "crpix1", "crpix2", "deep"],
     "tiles": [[…], …]}

``deep`` is 1 where the served mosaic comes from a deep-field processing plan
(``cutout_server.DEEP_PLAN``): DR1 deep-field tiles with no Wide Survey run.
Where a tile has both, the Wide Survey run is served.

The script fails if tiles disagree on the CD matrix or projection, or if a
tile has no usable VIS mosaic, rather than writing a partial index.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "cutout"))
import cutout_server as cs  # noqa: E402


async def tile_ids(bucket: cs.Bucket) -> list[str]:
    ids: list[str] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": f"{bucket.prefix}/", "delimiter": "/"}
        if token:
            params["continuation-token"] = token
        async with bucket.session.get(f"{bucket.base_url}/", params=params) as resp:
            resp.raise_for_status()
            root = cs.ElementTree.fromstring(await resp.read())
        for el in root.iter(f"{cs.S3_NS}Prefix"):
            tile = (el.text or "").rstrip("/").rsplit("/", 1)[-1]
            if tile.isdigit():
                ids.append(tile)
        token = root.findtext(f"{cs.S3_NS}NextContinuationToken")
        if root.findtext(f"{cs.S3_NS}IsTruncated") != "true" or not token:
            return sorted(set(ids))


async def vis_header(bucket: cs.Bucket, tile: str) -> list:
    keys = cs.mosaic_keys(await bucket.list_keys(f"{bucket.prefix}/{tile}/VIS/"), "BGSUB-MOSAIC-VIS", tile)
    if not keys:
        raise RuntimeError(f"tile {tile}: no VIS mosaic")
    image = await bucket.choose("VIS", keys)      # the wide-depth one when there are two
    w = image.wcs
    if (w.get("CTYPE1"), w.get("CTYPE2")) != ("RA---TAN", "DEC--TAN"):
        raise RuntimeError(f"tile {tile}: projection {w.get('CTYPE1')}/{w.get('CTYPE2')}")
    return [tile, w["CRVAL1"], w["CRVAL2"], image.width, image.height, w["CRPIX1"], w["CRPIX2"],
            int(image.deep),
            (w.get("CD1_1"), w.get("CD1_2", 0.0), w.get("CD2_1", 0.0), w.get("CD2_2")), image.dataset]


async def build(release: str, concurrency: int) -> dict:
    conf = cs.RELEASES[release]
    timeout = aiohttp.ClientTimeout(total=120, sock_read=60)
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        bucket = cs.Bucket(session, conf["bucket"], conf["prefix"], concurrency)
        ids = await tile_ids(bucket)
        print(f"{release}: {len(ids)} tiles", file=sys.stderr)
        done = 0

        async def one(tile):
            nonlocal done
            row = await vis_header(bucket, tile)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(ids)}", file=sys.stderr)
            return row

        rows = await asyncio.gather(*(one(t) for t in ids))
    cds = {tuple(round(v, 15) for v in r[8]) for r in rows}
    if len(cds) != 1:
        raise RuntimeError(f"{release}: tiles disagree on the CD matrix: {sorted(cds)[:4]}")
    datasets = {r[9] for r in rows}
    return {
        "release": release, "dataset": "/".join(sorted(d for d in datasets if d)),
        "built": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "bucket": conf["bucket"],
        "cd": list(next(iter(cds))),
        "columns": ["tile", "ra", "dec", "naxis1", "naxis2", "crpix1", "crpix2", "deep"],
        "tiles": [r[:8] for r in rows],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("release", choices=sorted(cs.RELEASES))
    parser.add_argument("--concurrency", type=int, default=64)
    args = parser.parse_args()
    index = asyncio.run(build(args.release, args.concurrency))
    out = ROOT / "services" / "cutout" / "index" / f"{args.release}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(index, separators=(",", ":")))
    print(f"wrote {out} ({len(index['tiles'])} tiles, {out.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
