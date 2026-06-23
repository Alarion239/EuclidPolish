#!/usr/bin/env python
"""Evaluate the lens-finder: TPR-vs-Einstein-radius for LR vs SR vs HR.

Reads the stamp catalog and the per-reconstruction ``predictions.csv`` written
by ``scripts/lensfinder_train.py``, then for each reconstruction computes the
POLISH++ headline: a single global score threshold at a fixed false-positive
rate, the true-positive rate per Einstein-radius bin, and ROC/AUC. Emits a
metrics CSV plus two plots — the 3-curve TPR-vs-θ_E comparison and a ROC
overlay. Pure numpy + matplotlib (no torch), runs in the main env.

Usage (EuclidPolishEnv)::

    python scripts/lensfinder_evaluate.py \
        --catalog data/lensfinder/stamps/catalog.csv \
        --heads-dir data/lensfinder/heads --out-dir data/lensfinder/eval
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.lensfinder import catalog as lf_catalog
from euclid_polish.lensfinder import metrics as lf_metrics
from euclid_polish.lensfinder import predictions as lf_pred

#: Plot colour + label per reconstruction.
RECON_STYLE = {
    "lr": ("#888888", "LR (raw input)"),
    "sr": ("#2a5db0", "SR (super-resolved)"),
    "hr": ("#1a1a1a", "HR (truth)"),
}


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", required=True, help="stamp catalog.csv")
    p.add_argument("--heads-dir", required=True,
                   help="train out-dir with lr/sr/hr/predictions.csv")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--target-fpr", type=float, default=1e-3)
    p.add_argument("--n-bins", type=int, default=8)
    p.add_argument("--theta-max", type=float, default=0.0,
                   help="max Einstein radius for binning (0 = from data)")
    return p.parse_args(argv)


def _evaluate_recon(rows, pred_csv, *, target_fpr, bins):
    """Per-recon metrics dict, or None when the predictions file is absent."""
    import numpy as np

    if not os.path.isfile(pred_csv):
        return None
    scores = lf_pred.read_scores(pred_csv)
    joined = lf_pred.join_scores(rows, scores, split="test")
    if not joined:
        return None
    s, y, te = lf_pred.scored_arrays(joined)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return None
    fpr, tpr, _ = lf_metrics.roc_curve(s, y)
    thr = lf_metrics.threshold_at_fpr(s, y, target_fpr)
    return {
        "auc": lf_metrics.auc(fpr, tpr),
        "threshold": thr,
        "tpr_at_fpr": lf_metrics.tpr_at_threshold(s, y, thr),
        "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
        "roc": (fpr, tpr),
        "theta": lf_metrics.tpr_vs_theta_e(s, y, te, target_fpr=target_fpr,
                                           bins=bins),
    }


def _write_metrics_csv(path, per_recon):
    rows = []
    for recon, r in per_recon.items():
        for b in r["theta"]["bins"]:
            rows.append({
                "recon": recon, "auc": r["auc"], "threshold": r["threshold"],
                "tpr_at_fpr": r["tpr_at_fpr"], "n_pos": r["n_pos"],
                "n_neg": r["n_neg"], "theta_lo": b["theta_lo"],
                "theta_hi": b["theta_hi"], "bin_n_pos": b["n_pos"],
                "bin_tpr": b["tpr"],
            })
    cols = ["recon", "auc", "threshold", "tpr_at_fpr", "n_pos", "n_neg",
            "theta_lo", "theta_hi", "bin_n_pos", "bin_tpr"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _plots(per_recon, out_dir, target_fpr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # TPR vs Einstein radius.
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for recon, r in per_recon.items():
        color, label = RECON_STYLE.get(recon, ("#444", recon))
        bins = r["theta"]["bins"]
        x = [0.5 * (b["theta_lo"] + b["theta_hi"]) for b in bins]
        ytpr = [b["tpr"] for b in bins]
        ax.plot(x, ytpr, "-o", color=color, label=f"{label} (AUC {r['auc']:.3f})")
    ax.set_xlabel("Einstein radius θ_E (arcsec)")
    ax.set_ylabel(f"TPR at FPR = {target_fpr:g}")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Lens recovery vs Einstein radius")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "tpr_vs_theta_e.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    # ROC overlay.
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for recon, r in per_recon.items():
        color, label = RECON_STYLE.get(recon, ("#444", recon))
        fpr, tpr = r["roc"]
        ax.plot(fpr, tpr, color=color, label=f"{label} (AUC {r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="#bbb")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC by reconstruction")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "roc.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    return p1, p2


def main(argv=None) -> int:
    args = _parse_args(argv)
    import numpy as np

    rows = lf_catalog.read_catalog(args.catalog)
    # Einstein-radius range from the test-split positives.
    te = [float(r["theta_E_arcsec"]) for r in rows
          if r.get("is_lens") == "1" and r.get("split") == "test"
          and r.get("theta_E_arcsec")]
    theta_max = args.theta_max or (max(te) if te else 3.0)
    bins = lf_metrics.theta_e_bins(0.0, theta_max, args.n_bins)

    os.makedirs(args.out_dir, exist_ok=True)
    per_recon = {}
    for recon in lf_catalog.RECONS:
        r = _evaluate_recon(
            rows, os.path.join(args.heads_dir, recon, "predictions.csv"),
            target_fpr=args.target_fpr, bins=bins)
        if r is not None:
            per_recon[recon] = r
            print(f"  {recon}: AUC {r['auc']:.3f}, TPR@FPR={args.target_fpr:g} "
                  f"= {r['tpr_at_fpr']:.3f}  ({r['n_pos']}+/{r['n_neg']}-)")
    if not per_recon:
        print("no usable predictions found (need test-split positives+negatives)")
        return 1

    csv_path = os.path.join(args.out_dir, "metrics.csv")
    _write_metrics_csv(csv_path, per_recon)
    p1, p2 = _plots(per_recon, args.out_dir, args.target_fpr)
    print(f"\n✓ {len(per_recon)} recon(s) → {csv_path}\n  {p1}\n  {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
