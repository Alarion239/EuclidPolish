"""member_257 @ 15M on the synthetic test fields (v5 test records, the ones it was scored on).

    cd EuclidPolish && python render_synthetic.py OUT_DIR

Rows: test fields picked by their PSNR gain over production 238 (best, 75th
percentile, median, 25th percentile, worst). Columns: Euclid input (nearest,
flux per SR pixel), production 238, 257 @ 15M, truth (HR blurred to the target
PSF, the scoring truth). RGB = H / J / VIS, asinh with the knee at the field's
input noise per SR pixel; one stretch per row, shared by its columns.
Writes zooms.png (8" crops on the brightest galaxy) and full_<field>.png.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import jwst_resolution as J  # noqa: E402
from euclid_polish.ensemble import EnsembleModel  # noqa: E402
from euclid_polish.web.helpers import ensemble_viz as ev  # noqa: E402
from euclid_polish.web.helpers import milestones, model_catalog  # noqa: E402
from euclid_polish.web.helpers import universal_models as um  # noqa: E402

OUT = sys.argv[1]
CACHE = os.path.join(OUT, "sr_cache")
CROP = int(os.environ.get("CROP", 160))
RGB = (3, 2, 0)                       # H, J, VIS channel indices of the 4-band arrays
_MODELS = {}


def model(spec):
    if spec not in _MODELS:
        if spec.startswith("milestone:"):
            _MODELS[spec] = milestones.load(*milestones.parse_spec(spec))
        else:
            item = model_catalog.resolve_spec(spec)
            _MODELS[spec] = EnsembleModel(model_catalog.ensemble_dir(), labels=[item.reads[0]],
                                          include_universal=True, starless=False)
    return _MODELS[spec]


def sr(spec, index, lr):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{spec.replace(':', '_').replace('@', '_')}__{index}.npy")
    if os.path.isfile(path):
        return np.load(path)
    out = np.asarray(model(spec).member_arrays(np.ascontiguousarray(lr, np.float32))[0], np.float32)
    np.save(path, out)
    return out


def brightest_galaxy(index, shape):
    rows = []
    with open(os.path.join(ev._sky_records_local_dir(), "sources_test.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            if int(r["field_index"]) == index and r["type"] == "galaxy" and r["off_field"] == "0":
                rows.append((float(r["flux_vis_e"]), float(r["x_pix"]), float(r["y_pix"])))
    _f, x, y = max(rows)
    h, w = shape
    return int(np.clip(round(y) - CROP // 2, 0, h - CROP)), int(np.clip(round(x) - CROP // 2, 0, w - CROP))


def rgb(chans, sigs, tops):
    out = np.stack([np.clip((np.arcsinh(c / s) + 0.3) / (t + 0.3), 0, 1)
                    for c, s, t in zip(chans, sigs, tops)], -1)
    return np.nan_to_num(out)


def main():
    pf = np.load(os.path.join(OUT, "per_field.npz"))
    fields, p238, p257 = pf["fields"], pf["p238"].mean(1), pf["p257"].mean(1)
    order = np.argsort(p257 - p238)
    n = len(order) - 1
    picks = [int(fields[order[i]]) for i in (n, round(0.75 * n), round(0.5 * n), round(0.25 * n), 0)]
    tags = ["largest gain", "75th pct gain", "median gain", "25th pct gain", "smallest gain"]
    at = {int(f): i for i, f in enumerate(fields)}

    identity = ev.universal_score_identity()
    fwhm = float(identity["target_psf_fwhm_arcsec"])
    pairs = {i: (lr, t) for i, lr, t in um._test_pairs(ev._sky_records_local_dir(), picks)}
    specs = ["universal:member_238", "milestone:member_257@15000000"]
    cols = ["Euclid input", "production 238", "257 @ 15M", "truth"]

    rows = []
    for index in picks:
        lr, target = pairs[index]
        truth = um._field_truth(target, fwhm)
        lr_up = np.repeat(np.repeat(np.nan_to_num(lr), 2, 0), 2, 1) / 4.0
        sigs = [J._noise_sigma([lr[..., b]]) / 4.0 for b in RGB]
        rows.append((index, [lr_up, *[sr(s, index, lr) for s in specs], truth], sigs))
        print(index, "sigma per SR px", np.round(sigs, 3), "lr median", np.round(np.median(lr, (0, 1)), 3))

    fig, axes = plt.subplots(len(rows), 4, figsize=(2.7 * 4, 2.7 * len(rows) + 0.5), squeeze=False)
    for r, ((index, imgs, sigs), tag) in enumerate(zip(rows, tags)):
        y0, x0 = brightest_galaxy(index, imgs[-1].shape[:2])
        cut = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
        tops = [np.percentile(np.arcsinh(imgs[-1][cut][..., b] / s), 99.7) for b, s in zip(RGB, sigs)]
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(rgb([img[cut][..., b] for b in RGB], sigs, tops), origin="lower", interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
        i = at[index]
        axes[r, 0].set_ylabel(f"field {index} · {tag}\n238 {p238[i]:.2f} → 257 {p257[i]:.2f} dB",
                              fontsize=8)
    fig.suptitle(f"257 @ 15M on synthetic test fields: {CROP * 0.05:.0f}″ crops on each field's brightest "
                 "galaxy\nRGB = H / J / VIS, asinh knee at the input noise per SR pixel; "
                 "dB = knee-integrated PSNR, mean of 4 bands", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "zooms.png"), dpi=130)
    plt.close(fig)

    index, imgs, sigs = rows[2]                       # the median-gain field, whole
    tops = [np.percentile(np.arcsinh(imgs[-1][..., b] / s), 99.7) for b, s in zip(RGB, sigs)]
    h = imgs[-1].shape[0]
    fig, axes = plt.subplots(1, 4, figsize=(4 * h / 128, h / 128 + 0.6))
    for c, img in enumerate(imgs):
        axes[c].imshow(rgb([img[..., b] for b in RGB], sigs, tops), origin="lower", interpolation="nearest")
        axes[c].set_xticks([]), axes[c].set_yticks([])
        axes[c].set_title(cols[c], fontsize=12)
    i = at[index]
    fig.suptitle(f"Whole test field {index} ({h * 0.05:.1f}″): median gain, "
                 f"238 {p238[i]:.2f} → 257 @ 15M {p257[i]:.2f} dB", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"full_{index}.png"), dpi=128)
    print("wrote", OUT, picks)


if __name__ == "__main__":
    main()
