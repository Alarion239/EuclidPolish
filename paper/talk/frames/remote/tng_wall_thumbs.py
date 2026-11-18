"""Run on FASRC (repo root on sys.path): colour thumbnails of TNG50-SKIRT atlas galaxies for the talk's
atlas wall. Picks N donors evenly in stellar mass from the usable atlas (dust-box and box-cut galaxies
excluded by TNGGalaxy.discover), orientation 1, R=H, G=(Y+J)/2, B=VIS, asinh stretch, 360 px PNG."""
import csv, json, os, sys
import numpy as np
from astropy.io import fits
from PIL import Image
sys.path.insert(0, os.getcwd())
from euclid_polish.tng.atlas import TNGGalaxy

root, params, out, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
os.makedirs(out, exist_ok=True)
gals = {g.subhalo_id.lstrip("0") or "0": g for g in TNGGalaxy.discover(root)}
rows = {}
for r in csv.DictReader(open(params)):
    if r["orientation"] == "1" and r["subhalo_id"] in gals:
        rows[r["subhalo_id"]] = r
ids = sorted(rows, key=lambda s: float(rows[s]["logmass_stars"]))
pick = [ids[int(i)] for i in np.linspace(0, len(ids) - 1, n)]
meta, crops = [], []
for sid in pick:
    g, r = gals[sid], rows[sid]
    planes = [np.asarray(fits.getdata(g.fits_path(1, b)), np.float64) for b in ("VIS", "Y", "J", "H")]
    h, w = planes[0].shape
    hw = int(np.clip(4.0 * float(r["native_re_px"]), 100, min(h, w) // 2))
    cy, cx = h // 2, w // 2
    cube = np.stack([p[cy - hw:cy + hw, cx - hw:cx + hw] for p in planes], -1)
    small = np.stack([np.asarray(Image.fromarray(cube[..., k].astype(np.float32)).resize((360, 360), Image.BOX))
                      for k in range(4)], -1)
    crops.append(small)
    sfr = float(r["sfr_msun_yr"]); m = float(r["mass_stars_msun"])
    meta.append(dict(id=sid, logm=float(r["logmass_stars"]), ssfr=float(np.log10(max(sfr, 1e-6) / m)),
                     re_kpc=float(r["native_re_kpc"]), fov_kpc=2 * hw * 0.1))
# white balance: the median galaxy of the wall comes out neutral (R=H, G=(Y+J)/2, B=VIS)
tot = np.array([[c[..., k].sum() for k in range(4)] for c in crops])
rgb_tot = np.stack([tot[:, 3], 0.5 * (tot[:, 1] + tot[:, 2]), tot[:, 0]], -1)
wb = 1.0 / np.median(rgb_tot / rgb_tot.mean(-1, keepdims=True), 0)
print("white balance", wb)
for sid, c in zip(pick, crops):
    R, G, B = (wb[0] * c[..., 3], wb[1] * 0.5 * (c[..., 1] + c[..., 2]), wb[2] * c[..., 0])
    I = (R + G + B) / 3
    white = np.percentile(I, 99.7); soft = white / 40
    f = np.arcsinh(np.maximum(I, 0) / soft) / np.arcsinh(white / soft)
    sc = np.where(I > 0, f / np.maximum(I, 1e-30), 0)
    rgb = np.clip(np.stack([R, G, B], -1) * sc[..., None], 0, None)
    rgb = rgb / np.maximum(rgb.max(-1, keepdims=True), 1)
    lum = rgb.mean(-1, keepdims=True); rgb = np.clip(lum + 1.3 * (rgb - lum), 0, 1)
    Image.fromarray((rgb * 255).astype(np.uint8)[::-1]).save(os.path.join(out, f"tng_{int(sid):06d}.png"))
json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
print(len(ids), "usable;", len(pick), "rendered")
