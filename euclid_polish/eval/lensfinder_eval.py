"""Consume ``lens_scores.csv`` on the /evaluation page (pure main env, no torch).

``scripts/lensfinder_score_eval.py`` (torch env) writes one row per object with
``P(lens)`` for the LR / SR / HR renders. Here we read those back to drive two
things: the PCA marker opacity (``per_object_plens``) and the lens-identification
analysis figure (``render_lensfinder_summary``) — SR-vs-LR shift, synthetic
ROC/AUC, score histograms by group, and P(lens) vs expert grade.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Any

GROUPS = ("A", "B", "C", "gal", "syn-lens", "syn-gal")
#: which render's score a morphology point uses, by its ``view``.
VIEW_TO_RECON = {"before": "lr", "after": "sr", "hr": "hr"}


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def read_lens_scores(run_dir: str) -> list[dict[str, Any]]:
    """Read ``lens_scores.csv`` → ``[{id, grade, lr, sr, hr}]`` (empty if absent)."""
    p = os.path.join(run_dir, "lens_scores.csv")
    if not os.path.isfile(p):
        return []
    out: list[dict[str, Any]] = []
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            out.append({
                "id": r.get("id", ""), "grade": r.get("grade", ""),
                "lr": _f(r.get("p_lens_lr")), "sr": _f(r.get("p_lens_sr")),
                "hr": _f(r.get("p_lens_hr")),
            })
    return out


def per_object_plens(run_dir: str) -> dict[str, dict[str, float]]:
    """``{object_id: {"lr": P, "sr": P, "hr": P}}`` for the embedding payload."""
    out: dict[str, dict[str, float]] = {}
    for r in read_lens_scores(run_dir):
        out[r["id"]] = {k: r[k] for k in ("lr", "sr", "hr")}
    return out


def _finite_pairs(rows, key, labels=None):
    """(values[, labels]) keeping only finite ``key`` entries."""
    vals, labs = [], []
    for i, r in enumerate(rows):
        v = r[key]
        if math.isfinite(v):
            vals.append(v)
            if labels is not None:
                labs.append(labels[i])
    return (vals, labs) if labels is not None else vals


def render_lensfinder_summary(run_dir: str, out_png: str) -> str | None:
    """Render the 6-panel lens-identification figure → ``out_png`` (or None).

    Top row: (1) SR-vs-LR P(lens) shift coloured by group, (2) synthetic ROC+AUC
    for LR/SR/HR, (2b) real ROC (A/B/C lenses vs real field galaxies, LR/SR).
    Bottom row: (3) SR score ridgeline by group, (4) LR score ridgeline by group
    (the same view before SR — read the SR shift off (3) vs (4)), (4b) real
    lens-vs-galaxy SR score histograms. Returns ``None`` when ``lens_scores.csv``
    is absent.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from euclid_polish.eval.zoobot_morph import _group_color
    from euclid_polish.lensfinder import metrics as lm

    rows = read_lens_scores(run_dir)
    if not rows:
        return None

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.0))

    # (1) SR vs LR shift — points above the diagonal mean SR raised P(lens).
    a = ax[0, 0]
    for g in GROUPS:
        gr = [r for r in rows if r["grade"] == g
              and math.isfinite(r["lr"]) and math.isfinite(r["sr"])]
        if gr:
            a.scatter([r["lr"] for r in gr], [r["sr"] for r in gr], s=14,
                      alpha=0.6, color=_group_color(g), label=g, edgecolors="none")
    a.plot([0, 1], [0, 1], ":", color="#999")
    a.set_xlabel("P(lens) — LR (before)")
    a.set_ylabel("P(lens) — SR (after)")
    a.set_title("Does SR raise lens identifiability?")
    a.set_xlim(-0.02, 1.02); a.set_ylim(-0.02, 1.02)
    a.legend(fontsize=8, loc="lower right"); a.grid(alpha=0.2)

    # (2) Synthetic ROC + AUC (syn-lens = positive, syn-gal = negative).
    # HR is the ground-truth high-res upper bound — it exists only for the
    # synthetic objects (real A/B/C have no HR), but this panel is synthetic-only
    # so HR is fully available and worth showing as the ceiling LR→SR climbs to.
    a = ax[0, 1]
    syn = [r for r in rows if r["grade"] in ("syn-lens", "syn-gal")]
    if syn:
        for recon, color in (("lr", "#888888"), ("sr", "#2a5db0"), ("hr", "#1a9850")):
            pairs = [(r[recon], 1 if r["grade"] == "syn-lens" else 0)
                     for r in syn if math.isfinite(r[recon])]
            if len(pairs) >= 2 and len({p[1] for p in pairs}) == 2:
                s = [p[0] for p in pairs]; y = [p[1] for p in pairs]
                fpr, tpr, _ = lm.roc_curve(s, y)
                a.plot(fpr, tpr, color=color, lw=1.8,
                       label=f"{recon.upper()} (AUC {lm.auc(fpr, tpr):.3f})")
        a.plot([0, 1], [0, 1], ":", color="#bbb")
    a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate")
    a.set_title("Synthetic lens vs galaxy — ROC")
    a.legend(fontsize=9, loc="lower right"); a.grid(alpha=0.2)

    # (2b) Real ROC — A/B/C lenses (positive) vs real field galaxies (negative).
    # No HR: real objects have no ground truth. This is the payoff of having real
    # negatives — does SR improve real lens-vs-galaxy separability?
    a = ax[0, 2]
    real = [r for r in rows if r["grade"] in ("A", "B", "C", "gal")]
    def is_lens(r):
        return 1 if r["grade"] in ("A", "B", "C") else 0
    if real and any(is_lens(r) for r in real) and any(not is_lens(r) for r in real):
        for recon, color in (("lr", "#888888"), ("sr", "#2a5db0")):
            pairs = [(r[recon], is_lens(r)) for r in real if math.isfinite(r[recon])]
            if len(pairs) >= 2 and len({p[1] for p in pairs}) == 2:
                s = [p[0] for p in pairs]; y = [p[1] for p in pairs]
                fpr, tpr, _ = lm.roc_curve(s, y)
                a.plot(fpr, tpr, color=color, lw=1.8,
                       label=f"{recon.upper()} (AUC {lm.auc(fpr, tpr):.3f})")
        a.plot([0, 1], [0, 1], ":", color="#bbb")
    a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate")
    a.set_title("Real lens (A/B/C) vs galaxy — ROC")
    if a.get_legend_handles_labels()[0]:           # only when a run has gal negatives
        a.legend(fontsize=9, loc="lower right")
    a.grid(alpha=0.2)

    # (3)/(4) P(lens) score distribution by group — ridgelines (one band per
    # group). Overlaid step-histograms were unreadable (many curves on one
    # axis); here each group gets its own baseline, its histogram filled and
    # peak-normalized so the *shape* (where the P(lens) mass sits) is comparable
    # across groups regardless of count. Bands overlap slightly, front-to-back,
    # for the classic joyplot look; the median is marked with a short tick. The
    # SR (after) and LR (before) panels sit side by side so the SR shift in each
    # group's score mass is read off directly.
    def _ridgeline(a, key, title, xlabel):
        bins = np.linspace(0, 1, 26)
        centers = 0.5 * (bins[:-1] + bins[1:])
        order = list(GROUPS)                    # A, B, C, gal, syn-lens, syn-gal
        n_g = len(order)
        spacing, band_h = 1.0, 1.5              # band_h > spacing → gentle overlap
        yticks, yticklabels = [], []
        for i, g in enumerate(order):
            base = (n_g - 1 - i) * spacing       # A on top, syn-gal at the bottom
            vals = _finite_pairs([r for r in rows if r["grade"] == g], key)
            yticks.append(base)
            yticklabels.append(f"{g}  (n={len(vals)})")
            a.axhline(base, color="#e0e0e0", lw=0.7, zorder=i * 3)
            if not vals:
                continue
            counts, _ = np.histogram(vals, bins=bins)
            h = counts / counts.max() * band_h   # peak-normalize each group
            col = _group_color(g)
            # Lower groups draw on top so their bands overlap the ones above.
            a.fill_between(centers, base, base + h, color=col, alpha=0.80,
                           linewidth=0.0, zorder=i * 3 + 1)
            a.plot(centers, base + h, color="#ffffff", lw=1.4, zorder=i * 3 + 2)
            a.plot(centers, base + h, color=col, lw=1.1, zorder=i * 3 + 2)
            med = float(np.median(vals))
            a.plot([med, med], [base, base + band_h * 0.30], color="#222",
                   lw=1.4, zorder=i * 3 + 3)
        a.set_yticks(yticks); a.set_yticklabels(yticklabels, fontsize=9)
        a.set_ylim(-0.25, (n_g - 1) * spacing + band_h + 0.1)
        a.set_xlim(0, 1)
        a.set_xlabel(xlabel); a.set_title(title)
        a.grid(axis="x", alpha=0.2)

    # (3) SR (after) score distribution by group — includes the real galaxies.
    _ridgeline(ax[1, 0], "sr",
               "Score distribution by group — SR (peak-normalized)",
               "P(lens) — SR")

    # (4) LR (before) score distribution by group — same view pre-SR, so the SR
    # shift in each group's score mass is the (3)−(4) difference.
    _ridgeline(ax[1, 1], "lr",
               "Score distribution by group — LR (peak-normalized)",
               "P(lens) — LR")

    # (4b) Real binary score separation — A/B/C lenses vs real galaxies (SR P(lens)).
    a = ax[1, 2]
    bins = np.linspace(0, 1, 26)
    lens_vals = _finite_pairs([r for r in rows if r["grade"] in ("A", "B", "C")], "sr")
    gal_vals = _finite_pairs([r for r in rows if r["grade"] == "gal"], "sr")
    if lens_vals:
        a.hist(lens_vals, bins=bins, color="#2a5db0", alpha=0.55, density=True,
               label=f"lenses A/B/C (n={len(lens_vals)})")
    if gal_vals:
        a.hist(gal_vals, bins=bins, color=_group_color("gal"), alpha=0.55, density=True,
               label=f"galaxies (n={len(gal_vals)})")
    a.set_xlabel("P(lens) — SR"); a.set_ylabel("density")
    a.set_title("Real: lens vs galaxy score")
    if a.get_legend_handles_labels()[0]:
        a.legend(fontsize=8)
    a.grid(alpha=0.2)

    fig.suptitle("Lens identification — trained finder on the eval objects",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png
