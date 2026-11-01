"""Which pixels does the SR under-predict vs Euclid within its noise?  python analyze.py OUT_DIR"""
import json
import sys

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

from common import BANDS, block_factors
from features import CLASSES, catalogue, tile_features

out_dir = sys.argv[1]
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "A"
MODELS = ("member_238", "member_249")
BAND = [b.name for b in BANDS]
rng = np.random.default_rng(0)
lr_all = np.load(f"{out_dir}/lr.npy", mmap_mode="r")
tiles = json.load(open(f"{out_dir}/tiles.json"))
cat = catalogue()
factors = block_factors()
KEEP = ("z1", "z4", "diff", "lr", "snr", "snr_smooth", "curv", "colour", "cls", "rnorm", "dist", "mag",
        "crowd", "near_hole", "hole")
SAMPLE = 0.05
report = {}

for label in MODELS:
    a_all = np.load(f"{out_dir}/{PREFIX}_{label}.npy", mmap_mode="r")
    # exact per-class sums over every pixel: count, Σdiff, ΣLR, # z4<-5, # z4>+5
    sums = np.zeros((len(CLASSES), 4, 5))
    snr_edges = np.array([-np.inf, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 1e3, 1e4, np.inf])
    snr_sums = np.zeros((len(snr_edges) - 1, 4, 3))
    rows = {k: [] for k in KEEP}
    weights = []
    for i in range(len(tiles["ids"])):
        f = tile_features(np.asarray(lr_all[i]), np.asarray(a_all[i]), tiles["headers"][i], cat, factors)
        ok = np.isfinite(f["z4"])
        for c in range(len(CLASSES)):
            for b in range(4):
                m = ok[:, b] & (f["cls"][:, b] == c)
                lrp = np.where(f["hole"][m, b], 0, f["lr"][m, b])
                sums[c, b] += [m.sum(), f["diff"][m, b].sum(), lrp.sum(),
                               (f["z4"][m, b] < -5).sum(), (f["z4"][m, b] > 5).sum()]
        for b in range(4):
            m = ok[:, b] & ~f["hole"][:, b]
            k = np.digitize(f["snr_smooth"][m, b], snr_edges) - 1
            np.add.at(snr_sums[:, b, 0], k, 1)
            np.add.at(snr_sums[:, b, 1], k, f["diff"][m, b])
            np.add.at(snr_sums[:, b, 2], k, f["lr"][m, b])
        # sample: every significant deficit + 5 % of the rest (weight 20)
        tail = ok & (f["z4"] < -5)
        rest = ok & ~tail & (rng.random(ok.shape) < SAMPLE)
        for sel, w in ((tail, 1.0), (rest, 1 / SAMPLE)):
            pix, band = np.nonzero(sel)
            for k in KEEP:
                rows[k].append(f[k][pix, band])
            rows.setdefault("band", []).append(band)
            weights.append(np.full(pix.size, w))
        if i % 50 == 0:
            print(label, i, flush=True)
    data = {k: np.concatenate(v) for k, v in rows.items()}
    data["w"] = np.concatenate(weights)
    np.savez_compressed(f"{out_dir}/sample_{PREFIX}_{label}.npz", **data)
    np.save(f"{out_dir}/classsums_{PREFIX}_{label}.npy", sums)
    np.save(f"{out_dir}/snrsums_{PREFIX}_{label}.npy", snr_sums)

    rep = {"classes": {}, "snr": {}, "tree": {}}
    for b in range(4):
        tot = sums[:, b]
        deficit = -tot[:, 1]
        for c, name in enumerate(CLASSES):
            rep["classes"].setdefault(name, {})[BAND[b]] = {
                "pixel_share": tot[c, 0] / tot[:, 0].sum(),
                "share_of_z4_below_-5": tot[c, 3] / max(tot[:, 3].sum(), 1),
                "frac_z4_below_-5": tot[c, 3] / max(tot[c, 0], 1),
                "frac_z4_above_+5": tot[c, 4] / max(tot[c, 0], 1),
                "net_flux_ratio_A_over_LR": (tot[c, 2] + tot[c, 1]) / tot[c, 2] if tot[c, 2] > 0 else None,
                "share_of_net_deficit": deficit[c] / deficit.sum() if deficit.sum() else None,
            }
        s = snr_sums[:, b]
        rep["snr"][BAND[b]] = [{"snr_lo": float(snr_edges[j]), "n": float(s[j, 0]),
                                "ratio": float((s[j, 2] + s[j, 1]) / s[j, 2]) if s[j, 2] > 0 else None}
                               for j in range(len(s))]
        # decision tree: what predicts a significant deficit (z4 < -5)?
        m = (data["band"] == b) & ~data["hole"]
        y = data["z4"][m] < -5
        X = np.stack([data["snr_smooth"][m], data["curv"][m], data["colour"][m], data["rnorm"][m],
                      data["dist"][m], np.nan_to_num(data["mag"][m], nan=30.0), data["crowd"][m],
                      data["near_hole"][m]], 1)
        names = ["snr_smooth", "curv", "colour", "rnorm", "dist", "mag", "crowd", "near_hole"]
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, class_weight="balanced",
                                      random_state=0).fit(X, y, sample_weight=data["w"][m])
        rep["tree"][BAND[b]] = {"rules": export_text(tree, feature_names=names, decimals=2),
                                "importance": dict(zip(names, tree.feature_importances_.round(3).tolist()))}
    report[label] = rep
json.dump(report, open(f"{out_dir}/report_{PREFIX}.json", "w"), indent=1, default=float)
print("done")
