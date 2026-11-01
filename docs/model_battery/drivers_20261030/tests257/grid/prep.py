"""Inputs shared by every model run: tile/field lists, noise levels, window
classes (from the LR VIS only, so identical for every model) and the figure
candidates. Writes prep.npz + prep.json. No inference.

    python prep.py
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from euclid_polish.web.helpers import universal_models as um  # noqa: E402
from euclid_polish.web.helpers.ensemble_viz import universal_score_identity  # noqa: E402

SMOOTH = 2.0          # Gaussian sigma of the LR VIS smoothing, LR px (0.2")
NAMED = ["f200w-0141", "f200w-0038", "f200w-0106", "f200w-0362"]


def window_classes(lr_vis, sig_s, origins):
    """Per SR window (origins in SR px): fraction of the footprint's LR pixels
    with the smoothed VIS in [1, 10] sigma_s, the footprint's max and mean
    (sigma_s units), and the mean LR VIS per SR pixel (e-)."""
    S = ndimage.gaussian_filter(lr_vis.astype(np.float64), SMOOTH) / sig_s
    out = []
    for y, x in origins:
        win = S[y // 2:(y + C.W) // 2, x // 2:(x + C.W) // 2]
        raw = lr_vis[y // 2:(y + C.W) // 2, x // 2:(x + C.W) // 2]
        out.append(((win >= 1) & (win <= 10)).mean())
        out.append(win.max())
        out.append(win.mean())
        out.append(raw.mean() / 4.0)
    return np.asarray(out, np.float32).reshape(len(origins), 4)


def main():
    os.makedirs(C.STATS, exist_ok=True)
    # ---- real
    tiles = C.clean_tiles()
    lrs = {t: C.tile(t)[0] for t in tiles}
    sig_r = np.array([C.noise_sigma(np.concatenate([lrs[t][..., b].ravel()[::7] for t in tiles]))
                      for b in range(4)])
    sm = np.concatenate([ndimage.gaussian_filter(lrs[t][..., 0].astype(np.float64), SMOOTH).ravel()[::7]
                         for t in tiles])
    sigs_r = C.noise_sigma(sm)
    origins = C.stride_origins(510, 510, lo=0)
    cls_r = np.stack([window_classes(lrs[t][..., 0], sigs_r, origins) for t in tiles])   # (T, N, 4)
    # ---- synthetic
    ident = universal_score_identity()
    fwhm = float(ident["target_psf_fwhm_arcsec"])
    syn_lr, fields = [], []
    for idx, lr, _rec in um._test_pairs(C.RECORDS, ident["indices"]):
        syn_lr.append(lr)
        fields.append(int(idx))
    sig_s = np.array([C.noise_sigma(np.concatenate([x[..., b].ravel()[::7] for x in syn_lr])) for b in range(4)])
    sm = np.concatenate([ndimage.gaussian_filter(x[..., 0].astype(np.float64), SMOOTH).ravel()[::7]
                         for x in syn_lr])
    sigs_s = C.noise_sigma(sm)
    cls_s = np.stack([window_classes(x[..., 0], sigs_s, origins) for x in syn_lr])
    # ---- outskirt windows: >= 50 % of the footprint in [1, 10] sigma_s, max <= 30 sigma_s
    inner = np.array([(32 <= y <= 510 - 32 - C.W) and (32 <= x <= 510 - 32 - C.W) for y, x in origins])
    out_r = (cls_r[..., 0] >= 0.5) & (cls_r[..., 1] <= 30) & inner[None]
    out_s = (cls_s[..., 0] >= 0.5) & (cls_s[..., 1] <= 30) & inner[None]
    # figure candidates: the named tiles + the 16 tiles with the most outskirt windows
    order = np.argsort(-out_r.sum(1), kind="stable")
    cand = list(NAMED) + [tiles[i] for i in order if tiles[i] not in NAMED][:16]
    cand_syn = [fields[i] for i in np.argsort(-out_s.sum(1), kind="stable")[:4]]
    np.savez(os.path.join(C.HERE, "prep.npz"), tiles=np.array(tiles), fields=np.array(fields),
             origins=np.array(origins), cls_real=cls_r, cls_syn=cls_s, out_real=out_r, out_syn=out_s,
             inner=inner)
    meta = {"tiles": tiles, "fields": fields, "sigma_lr_real": sig_r.tolist(),
            "sigma_lr_syn": sig_s.tolist(), "sigma_smooth_vis_real": sigs_r, "sigma_smooth_vis_syn": sigs_s,
            "smooth_lr_px": SMOOTH, "fwhm": fwhm, "cand_real": cand, "cand_syn": cand_syn,
            "n_outskirt_real": int(out_r.sum()), "n_outskirt_syn": int(out_s.sum())}
    with open(os.path.join(C.HERE, "prep.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("tiles", "fields")}, indent=1))


if __name__ == "__main__":
    main()
