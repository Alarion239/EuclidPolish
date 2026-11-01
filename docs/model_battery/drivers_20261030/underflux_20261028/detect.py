"""Same-pipeline source detection on real (NEXUS) and synthetic LR VIS, so r/r_Kron means the same in both.

python detect.py OUT_DIR   → sources_real.npz, sources_syn.npz (one row per source: tile, x, y, kron_px,
ellipticity, theta (rad), mag, star) and zp.json (VIS electrons → AB, from the real MER cross-match).
"""
import json
import sys

import numpy as np
from astropy.convolution import convolve
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from photutils.segmentation import SourceCatalog, deblend_sources, detect_sources, make_2dgaussian_kernel

from features import catalogue

out_dir = sys.argv[1]
KERNEL = make_2dgaussian_kernel(2.0, size=5)


def detect(vis):
    hole = (vis == 0) | ~np.isfinite(vis)
    img = np.where(hole, 0.0, vis)
    sm = convolve(img, KERNEL, mask=hole, normalize_kernel=True)
    _, med, std = sigma_clipped_stats(sm[~hole], sigma=3.0)
    seg = detect_sources(sm - med, 2.5 * std, npixels=10, mask=hole)
    if seg is None:
        return None, None
    seg = deblend_sources(sm - med, seg, npixels=10, nlevels=32, contrast=0.005, progress_bar=False)
    cat = SourceCatalog(img - med, seg, convolved_data=sm - med, mask=hole)
    return cat, seg


def rows(cat):
    kr = np.asarray(cat.kron_radius.value, float) * np.asarray(cat.semimajor_sigma.value, float)
    flux = np.asarray(cat.kron_flux, float)
    flux = np.where(np.isfinite(flux) & (flux > 0), flux, np.asarray(cat.segment_flux, float))
    return dict(x=np.asarray(cat.xcentroid, float), y=np.asarray(cat.ycentroid, float),
                kron_px=np.where(np.isfinite(kr) & (kr > 0), kr, 1.5),
                ellipticity=np.nan_to_num(np.asarray(cat.ellipticity.value, float)),
                theta=np.deg2rad(np.nan_to_num(np.asarray(cat.orientation.value, float))),
                flux=flux, label=np.asarray(cat.labels))


def collect(acc, tile, r, star):
    for k, v in r.items():
        acc.setdefault(k, []).append(v)
    acc.setdefault("tile", []).append(np.full(len(r["x"]), tile))
    acc.setdefault("star", []).append(star)


# real: stars + zeropoint from the MER cross-match
mer = catalogue()
lr_real = np.load(f"{out_dir}/lr.npy", mmap_mode="r")
tiles = json.load(open(f"{out_dir}/tiles.json"))
real, zp_terms = {}, []
for i in range(lr_real.shape[0]):
    cat, _ = detect(np.asarray(lr_real[i, ..., 0], np.float64))
    if cat is None:
        continue
    r = rows(cat)
    mx, my = WCS(fits.Header.fromstring(tiles["headers"][i])).world_to_pixel_values(
        mer["right_ascension"], mer["declination"])
    inside = np.flatnonzero((mx > -3) & (mx < 258) & (my > -3) & (my < 258))
    star = np.zeros(len(r["x"]), bool)
    for s in range(len(r["x"])):
        d = np.hypot(mx[inside] - r["x"][s], my[inside] - r["y"][s])
        if d.size and d.min() < 2.0:
            j = inside[np.argmin(d)]
            star[s] = mer["point_like_prob"][j] > 0.5
            if 18 < mer["mag"][j] < 21 and r["flux"][s] > 0:
                zp_terms.append(mer["mag"][j] + 2.5 * np.log10(r["flux"][s]))
    collect(real, i, r, star)
    if i % 100 == 0:
        print("real", i, flush=True)
zp = float(np.median(zp_terms))
print("ZP (VIS e- -> AB)", zp, "from", len(zp_terms), "matches, MAD",
      float(np.median(np.abs(np.array(zp_terms) - zp))), flush=True)
real = {k: np.concatenate(v) for k, v in real.items()}
real["mag"] = zp - 2.5 * np.log10(np.clip(real["flux"], 1e-3, None))
np.savez(f"{out_dir}/sources_real.npz", **real)

# synthetic: star = the segment's truth light is mostly the star component
lr_syn = np.load(f"{out_dir}/lr_syn.npy", mmap_mode="r")
RT = np.load(f"{out_dir}/RT_syn.npy", mmap_mode="r")
RC = np.load(f"{out_dir}/RC_syn.npy", mmap_mode="r")
syn = {}
for i in range(lr_syn.shape[0]):
    cat, seg = detect(np.asarray(lr_syn[i, ..., 0], np.float64))
    if cat is None:
        continue
    r = rows(cat)
    t, c = np.asarray(RT[i, ..., 0], np.float64), np.asarray(RC[i, ..., 0], np.float64)
    labels = seg.data
    tot = np.bincount(labels.ravel(), weights=t.ravel(), minlength=labels.max() + 1)
    stars = np.bincount(labels.ravel(), weights=(t - c).ravel(), minlength=labels.max() + 1)
    star = stars[r["label"]] > 0.5 * np.maximum(tot[r["label"]], 1e-9)
    collect(syn, i, r, star)
syn = {k: np.concatenate(v) for k, v in syn.items()}
syn["mag"] = zp - 2.5 * np.log10(np.clip(syn["flux"], 1e-3, None))
np.savez(f"{out_dir}/sources_syn.npz", **syn)
json.dump({"zp": zp}, open(f"{out_dir}/zp.json", "w"))
for name, s in (("real", real), ("syn", syn)):
    n_tiles = len(np.unique(s["tile"]))
    g = ~s["star"]
    print(name, "sources/tile", len(s["x"]) / n_tiles, "stars/tile", s["star"].sum() / n_tiles,
          "galaxy mag pct 10/50/90", np.percentile(s["mag"][g], [10, 50, 90]).round(2),
          "kron_px median", float(np.median(s["kron_px"][g])), flush=True)
print("done")
