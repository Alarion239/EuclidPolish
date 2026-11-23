"""The same 102.4" Hubble Ultra Deep Field patch, telescope by telescope (fetch_hudf_multi.py), and one
galaxy in it: DSS2 photographic plates -> Pan-STARRS1 -> DECam (Legacy Surveys DR10) -> Euclid Q1 ->
Hubble XDF -> JWST JADES. Every image is resampled onto one north-up grid at (close to) its native pixel
scale, then shown as an identically framed square, so in Prezi each can fade in over the previous one.
Colour: three bands per instrument, each divided by its own bright-galaxy level, then one asinh stretch."""
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import save_panel  # noqa: E402

TALK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(TALK, "data", "hudf_multi")
EU = "/Users/alarion239/Desktop/EuclidPolish/data/euclid_inference/talk/hudf/original_stack.fits"
SR = "/Users/alarion239/Desktop/EuclidPolish/data/euclid_inference/talk/hudf/SR.fits"
we = WCS(fits.getheader(EU)).celestial
RA0, DEC0 = (float(v) for v in we.pixel_to_world_values(511.5, 511.5))
FIELD = 102.4


def grid_wcs(ra, dec, scale, n):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    w.wcs.cdelt = [-scale / 3600, scale / 3600]
    return w


def resample(img, w_in, ra, dec, scale, size, order=1):
    n = int(round(size / scale))
    wo = grid_wcs(ra, dec, scale, n)
    yy, xx = np.mgrid[0:n, 0:n]
    r, d = wo.pixel_to_world_values(xx, yy)
    x, y = w_in.world_to_pixel_values(r, d)
    return map_coordinates(np.nan_to_num(img), [y, x], order=order, mode="constant", cval=0.0)


def load(name, ext=0, plane=None):
    with fits.open(os.path.join(D, name)) as h:
        a = np.asarray(h[ext].data, np.float64)
        w = WCS(h[ext].header).celestial
    return (a[plane] if plane is not None else a), w


def bands(inst):
    if inst == "dss":
        r = load("dss2_red.fits")
        return [r, r, r]                                     # one plate, grey, as it looks
    if inst == "ps1":
        return [load(f"ps1_{b}.fits") for b in ("i", "r", "g")]
    if inst == "ls":
        a, w = load("lsdr10_griz.fits")
        return [(a[3], w), (a[1], w), (a[0], w)]          # z, r, g
    if inst == "euclid":
        a, w = load(EU)
        return [(a[3], we), (0.5 * (a[1] + a[2]), we), (a[0], we)]   # H, Y+J, VIS
    if inst == "polished" and GAL_SR["on"]:
        with fits.open(os.path.join(D, "euclid_galaxy_sr_tta8.fits")) as h:   # hudf_galaxy_sr.py, TTA x8
            a = np.asarray(h[0].data, np.float64); ws = WCS(h[0].header).celestial
        return [(a[3], ws), (0.5 * (a[1] + a[2]), ws), (a[0], ws)]
    if inst == "polished":
        a = np.asarray(fits.getdata(SR), np.float64)
        ws = grid_wcs(RA0, DEC0, 0.05, a.shape[-1])                   # SR grid = Euclid grid x2
        return [(a[3], ws), (0.5 * (a[1] + a[2]), ws), (a[0], ws)]
    if inst == "hst":
        b = {k: load(f"xdf_{k}.fits") for k in ("f435w", "f606w", "f775w", "f850lp")}
        return [b["f850lp"], (0.5 * (b["f606w"][0] + b["f775w"][0]), b["f606w"][1]), b["f435w"]]
    if inst == "jwst":
        b = {k: load(f"jades_{k}.fits") for k in ("f090w", "f115w", "f150w", "f200w")}
        return [b["f200w"], (0.5 * (b["f115w"][0] + b["f150w"][0]), b["f115w"][1]), b["f090w"]]


FLOOR = {"dss": 25, "ps1": 12}   # white >= this many noise sigma (shallow surveys)
INST = {  # name: (label, field pixel scale, zoom pixel scale, stretch Q, interp)
    "dss": ("DSS2 · photographic plates · ~2″", 1.0, 1.0, 6, "nearest"),
    "ps1": ("Pan-STARRS1 · 1.1″", 0.25, 0.25, 8, "nearest"),
    "ls": ("DECam (DES / Legacy Surveys) · 0.9″", 0.262, 0.262, 8, "nearest"),
    "euclid": ("Euclid · 0.16″", 0.1, 0.1, 10, "nearest"),
    "polished": ("POLISHed Euclid", 0.05, 0.05, 10, "lanczos"),
    "hst": ("Hubble · XDF · 0.09″", 0.06, 0.03, 12, "lanczos"),
    "jwst": ("JWST · JADES · 0.06″", 0.06, 0.03, 12, "lanczos"),
}


def rgb(planes, white, Q):
    out = []
    for p, wv in zip(planes, white):
        _, med, _ = sigma_clipped_stats(p, sigma=3, maxiters=5)
        out.append(np.clip(p - med, 0, None) / wv)
    I = (out[0] + out[1] + out[2]) / 3 + 1e-12
    f = np.arcsinh(Q * I) / np.arcsinh(Q) / I
    return np.clip(np.stack([o * f for o in out], -1), 0, 1)


def render(inst, ra, dec, size, scale, tag, white=None, bar=None):
    label, _, _, Q, interp = INST[inst]
    planes = [resample(a, w, ra, dec, scale, size) for a, w in bands(inst)]
    if white is None:
        white = []
        for p in planes:
            _, med, sd = sigma_clipped_stats(p, sigma=3, maxiters=5)
            white.append(max(np.percentile(p - med, 99.7), FLOOR.get(inst, 0) * sd))
    img = rgb(planes, white, Q)[::-1]          # map_coordinates grid is y-up; save_panel uses origin=lower
    save_panel(img[::-1], f"HUDF_{tag}_{inst}.png", label=label, interp=interp, bar_arcsec=bar,
               pix_arcsec=scale)
    return white


WHITE_EU = {}
GAL_SR = {"on": False}
if __name__ == "__main__":
    which = sys.argv[1:] or list(INST)
    gal = (float(os.environ.get("GRA", "nan")), float(os.environ.get("GDEC", "nan")))
    for inst in which:
        if inst in ("hst", "jwst") and not os.path.exists(os.path.join(D, f"{'xdf_f850lp' if inst == 'hst' else 'jades_f200w'}.fits")):
            print("skip", inst); continue
        wf = None
        if inst == "polished":
            wf = [v / 4 for v in WHITE_EU["field"]]
        w = render(inst, RA0, DEC0, FIELD, INST[inst][1], "field", white=wf, bar=10.0)
        if inst == "euclid":
            WHITE_EU["field"] = w
        if np.isfinite(gal[0]):
            GAL_SR["on"] = True
            wg = render(inst, gal[0], gal[1], float(os.environ.get("GSIZE", "8")), INST[inst][2], "galaxy",
                        white=[v / 4 for v in WHITE_EU["galaxy"]] if inst == "polished" else None, bar=1.0)
            if inst == "euclid":
                WHITE_EU["galaxy"] = wg
            GAL_SR["on"] = False
        print(inst, "ok", flush=True)
