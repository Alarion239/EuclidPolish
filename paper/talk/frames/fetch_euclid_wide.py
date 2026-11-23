"""A wide, coarse Euclid Q1 view around the HUDF spiral for the proportional-area ladder: every 8th row of
each MER mosaic (VIS, Y, J, H) that touches the box, binned 8 columns at a time (0.8" pixels), read with
HTTP range requests from IRSA's public bucket through services/cutout's Bucket. One FITS per tile and band
in data/hudf_multi/euwide_<tile>_<band>.fits, archive ADU/s, with the WCS of the binned grid.

usage: fetch_euclid_wide.py [box_arcsec]"""
import asyncio
import json
import os
import sys

import aiohttp
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT = "/Users/alarion239/Desktop/EuclidPolish"
sys.path.insert(0, os.path.join(ROOT, "services", "cutout"))
from cutout_server import RELEASES, Bucket  # noqa: E402

OUT = os.path.join(ROOT, "paper", "talk", "data", "hudf_multi")
GRA, GDEC = 53.17468, -27.799422            # the HUDF spiral
BOX = float(sys.argv[1]) if len(sys.argv) > 1 else 1932.0
B = 8                                       # binning / row stride


def wcs_of(d):
    w = WCS(naxis=2)
    w.wcs.ctype = [d["CTYPE1"], d["CTYPE2"]]
    w.wcs.crval = [d["CRVAL1"], d["CRVAL2"]]
    w.wcs.crpix = [d["CRPIX1"], d["CRPIX2"]]
    w.wcs.cd = [[d["CD1_1"], d.get("CD1_2", 0.0)], [d.get("CD2_1", 0.0), d["CD2_2"]]]
    return w


async def main():
    idx = json.load(open(os.path.join(ROOT, "services", "cutout", "index", "q1.json")))
    cosd = np.cos(np.radians(GDEC))
    near = [t for t in idx["tiles"] if np.hypot((t[1] - GRA) * cosd, t[2] - GDEC) < 0.8]
    print("tiles", [t[0] for t in near], flush=True)
    half = BOX / 2 / 3600 * 1.05
    corners_ra = GRA + np.array([-1, 1, 1, -1]) * half / cosd
    corners_de = GDEC + np.array([-1, -1, 1, 1]) * half
    async with aiohttp.ClientSession() as s:
        bucket = Bucket(s, RELEASES["q1"]["bucket"], RELEASES["q1"]["prefix"], concurrency=48)
        for t in near:
            tile = str(t[0])
            imgs = await bucket.tile(tile)
            for band, im in imgs.items():
                path = os.path.join(OUT, f"euwide_{tile}_{band}.fits")
                if os.path.exists(path):
                    print("have", path); continue
                w = wcs_of(im.wcs)
                x, y = w.world_to_pixel_values(corners_ra, corners_de)
                xa, xb = int(max(0, np.floor(x.min()))), int(min(im.width, np.ceil(x.max())))
                ya, yb = int(max(0, np.floor(y.min()))), int(min(im.height, np.ceil(y.max())))
                if xa >= xb or ya >= yb:
                    print(tile, band, "outside"); break
                nx = (xb - xa) // B
                xb = xa + nx * B
                rows = list(range(ya, yb, B))
                out = np.full((len(rows), nx), np.nan, np.float32)

                async def row(i, yy):
                    start = im.data_offset + 4 * (yy * im.width + xa)
                    v = np.frombuffer(await bucket.read(im.key, start, 4 * (xb - xa)), ">f4").astype(np.float32)
                    v = v.reshape(nx, B)
                    v[v == 0] = np.nan
                    out[i] = np.nanmean(v, 1)

                for c in range(0, len(rows), 256):
                    await asyncio.gather(*(row(i, rows[i]) for i in range(c, min(c + 256, len(rows)))))
                    print(f"  {tile} {band} {min(c + 256, len(rows))}/{len(rows)}", flush=True)
                out = out * im.bscale + im.bzero
                wb = w.deepcopy()       # binned pixel (j, i) = original (xa + B j + (B-1)/2, ya + B i)
                wb.wcs.cd = w.wcs.cd * np.array([[B, B], [B, B]])
                wb.wcs.crpix = [(w.wcs.crpix[0] - 1 - xa - (B - 1) / 2) / B + 1, (w.wcs.crpix[1] - 1 - ya) / B + 1]
                h = wb.to_header()
                h["MAGZERO"] = im.magzero
                h["BAND"] = band
                fits.writeto(path, out, h, overwrite=True)
                print("wrote", path, out.shape, flush=True)


asyncio.run(main())
