"""Re-POLISH the HUDF ladder galaxy from a 25.6" Euclid crop centred on it (production 257, dihedral TTA x8),
so the galaxy sits in the middle of the network's input instead of near a tile seam of the full-field run."""
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT = "/Users/alarion239/Desktop/EuclidPolish"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "output", "lens_dr1_dspl"))
os.chdir(ROOT)
EU = os.path.join(ROOT, "data/euclid_inference/talk/hudf/original_stack.fits")
OUT = os.path.join(ROOT, "paper/talk/data/hudf_multi/euclid_galaxy_sr_tta8.fits")
GRA, GDEC, N = 53.17468, -27.799422, 256

from euclid_polish.config import Config  # noqa: E402
from euclid_polish.eval.ensemble_infer import load_eval_ensemble, sr_from_model  # noqa: E402

h = fits.open(EU)[0]
w = WCS(h.header).celestial
x, y = w.world_to_pixel_values(GRA, GDEC)
x0, y0 = int(round(float(x))) - N // 2, int(round(float(y))) - N // 2
x0 = max(0, min(x0, h.data.shape[2] - N)); y0 = max(0, min(y0, h.data.shape[1] - N))
print('crop origin', x0, y0, 'galaxy at', float(x) - x0, float(y) - y0)
lr = np.moveaxis(np.asarray(h.data[:, y0:y0 + N, x0:x0 + N], np.float32), 0, -1)


def fwd(a, t):
    a = np.swapaxes(a, 0, 1) if t & 4 else a
    return np.rot90(a, t & 3, axes=(0, 1))


def inv(a, t):
    a = np.rot90(a, -(t & 3), axes=(0, 1))
    return np.swapaxes(a, 0, 1) if t & 4 else a


model = load_eval_ensemble(None, Config.DEFAULT_NUM_RES_BLOCKS, log=print)
acc = 0
for t in range(8):
    _, sr, _ = sr_from_model(model, np.ascontiguousarray(fwd(lr, t)))
    sr = np.asarray(sr, np.float64)
    sr = sr[0] if sr.ndim == 4 else sr
    if t == 0:
        print('sr shape', sr.shape, flush=True)
    acc = acc + inv(sr, t)
sr = (acc / 8).astype(np.float32)
wl = w.slice((slice(y0, y0 + N), slice(x0, x0 + N)))
ws = wl.deepcopy()
ws.wcs.crpix = [(c - 0.5) * 2 + 0.5 for c in wl.wcs.crpix]
if ws.wcs.has_cd():
    ws.wcs.cd = wl.wcs.cd / 2
else:
    ws.wcs.cdelt = wl.wcs.cdelt / 2
fits.HDUList([fits.PrimaryHDU(np.moveaxis(sr, -1, 0), ws.to_header()),
              fits.ImageHDU(np.moveaxis(lr, -1, 0), wl.to_header(), name="LR")]).writeto(OUT, overwrite=True)
print("wrote", OUT, sr.shape)
