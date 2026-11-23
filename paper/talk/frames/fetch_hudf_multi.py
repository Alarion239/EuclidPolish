"""Same 102.4" field around the Hubble Ultra Deep Field from the official archives, for the talk's
"wide or sharp" ladder: DSS2 (STScI), Pan-STARRS1 3pi stacks (STScI), DESI Legacy Surveys DR10 (DECam),
Hubble XDF (MAST HLSP, 30 mas), JWST JADES DR2 GOODS-S deep (MAST HLSP, 30 mas). Euclid Q1 comes from
data/euclid_inference/talk/hudf (the field is centred on that cutout). Byte-range reads for the big mosaics."""
import os
import subprocess
import sys

import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hudf_multi")
os.makedirs(OUT, exist_ok=True)
EU = "/Users/alarion239/Desktop/EuclidPolish/data/euclid_inference/talk/hudf/original_stack.fits"
we = WCS(fits.getheader(EU)).celestial
RA, DEC = (float(v) for v in we.pixel_to_world_values(511.5, 511.5))
SIZE = 102.4  # arcsec
print("centre", RA, DEC, flush=True)


def get(url, path):
    if not os.path.exists(path):
        subprocess.run(["curl", "-s", "-m", "600", "-o", path, url], check=True)
    return path


def section(url, out, band):
    path = os.path.join(OUT, out)
    if os.path.exists(path):
        print("have", out); return
    with fits.open(url, use_fsspec=True, lazy_load_hdus=True, fsspec_kwargs={"block_size": 8 * 2**20}) as h:
        ext = 1 if len(h) > 1 and h[1].header.get("NAXIS", 0) == 2 else 0
        hd = h[ext].header
        w = WCS(hd).celestial
        x, y = w.world_to_pixel_values(RA, DEC)
        pix = abs(hd.get("CD1_1") or hd.get("CDELT1")) * 3600
        n = int(round(SIZE / pix))
        x0, y0 = int(round(x - n / 2)), int(round(y - n / 2))
        data = np.asarray(h[ext].section[y0:y0 + n, x0:x0 + n], np.float32)
        wc = w.slice((slice(y0, y0 + n), slice(x0, x0 + n)))
    fits.writeto(path, data, wc.to_header(), overwrite=True)
    print("wrote", out, data.shape, f"{pix:.3f}\"", flush=True)


# DSS2 blue / red / IR (arcmin box)
for v in ("poss2ukstu_blue", "poss2ukstu_red", "poss2ukstu_ir"):
    get(f"https://archive.stsci.edu/cgi-bin/dss_search?v={v}&r={RA}&d={DEC}&e=J2000&h=1.8&w=1.8&f=fits",
        os.path.join(OUT, f"dss2_{v.split('_')[1]}.fits"))
print("dss ok", flush=True)
# Pan-STARRS1 stacks g r i z (0.25"/px)
lst = subprocess.run(["curl", "-s", "-m", "120", f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={RA}&dec={DEC}&filters=grizy"],
                     capture_output=True, text=True).stdout.splitlines()[1:]
for line in lst:
    p = line.split(); band, fname = p[4], p[7]
    get(f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?ra={RA}&dec={DEC}&size={int(SIZE / 0.25)}&format=fits&red={fname}",
        os.path.join(OUT, f"ps1_{band}.fits"))
print("ps1 ok", flush=True)
# Legacy Surveys DR10 (DECam) g r i z
get(f"https://www.legacysurvey.org/viewer/fits-cutout?ra={RA}&dec={DEC}&layer=ls-dr10&pixscale=0.262&size={int(SIZE / 0.262)}&bands=griz",
    os.path.join(OUT, "lsdr10_griz.fits"))
print("ls ok", flush=True)
# Hubble XDF 30 mas
for b in ("f435w", "f606w", "f775w", "f850lp"):
    section(f"https://archive.stsci.edu/hlsps/xdf/hlsp_xdf_hst_acswfc-30mas_hudf_{b}_v1_sci.fits", f"xdf_{b}.fits", b)
# JWST JADES DR2 GOODS-S deep 30 mas
for b in ("f090w", "f115w", "f150w", "f200w"):
    section(f"https://archive.stsci.edu/hlsps/jades/dr2/goods-s/images/hlsp_jades_jwst_nircam_goods-s-deep_{b}_v2.0_drz.fits",
            f"jades_{b}.fits", b)
print("done", flush=True)
