"""Lens-finder evaluation metrics (pure numpy, no sklearn).

The POLISH++ construction (Wu+ 2026): choose one global score threshold at a
fixed false-positive rate on the negatives, then measure the true-positive rate
as a function of Einstein radius on the positives. Plus ROC / AUC. numpy-only so
this imports in the main env without torch or TensorFlow.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def roc_curve(scores: Sequence[float], labels: Sequence[int]
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(fpr, tpr, thresholds)`` for binary labels (1 = positive).

    A point is the rate when predicting positive for ``score >= threshold``.
    Tied scores collapse to a single step; the curve starts at (0, 0).
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if s.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    P = max(int((y == 1).sum()), 1)
    N = max(int((y == 0).sum()), 1)
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    distinct = np.r_[np.diff(s) != 0, True]      # keep last index of each run
    tpr = np.r_[0.0, tp[distinct] / P]
    fpr = np.r_[0.0, fp[distinct] / N]
    thr = np.r_[np.inf, s[distinct]]
    return fpr, tpr, thr


def auc(fpr: Sequence[float], tpr: Sequence[float]) -> float:
    """Area under the ROC curve by the trapezoid rule (``fpr`` increasing).

    Hand-rolled (not ``np.trapz``, which is removed in NumPy 2.0) so this stays
    version-agnostic.
    """
    x = np.asarray(fpr, dtype=float)
    y = np.asarray(tpr, dtype=float)
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0))


def threshold_at_fpr(scores: Sequence[float], labels: Sequence[int],
                     target_fpr: float) -> float:
    """Lowest ``>=`` threshold whose FPR on negatives stays ≤ ``target_fpr``.

    Predict positive when ``score >= threshold``. Maximizes admitted positives
    subject to the FPR cap. Returns ``+inf`` when there are no negatives.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    neg = np.sort(s[y == 0])[::-1]               # negatives, high → low
    N = neg.size
    if N == 0:
        return float("inf")
    k = int(np.floor(target_fpr * N))            # negatives we may admit
    if k <= 0:
        return float(np.nextafter(neg[0], np.inf))   # admit none
    if k >= N:
        return float("-inf")                     # admit all
    return float(np.nextafter(neg[k], np.inf))   # just above (k+1)-th negative


def tpr_at_threshold(scores: Sequence[float], labels: Sequence[int],
                     thr: float) -> float:
    """Fraction of positives with ``score >= thr`` (NaN if no positives)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos = s[y == 1]
    if pos.size == 0:
        return float("nan")
    return float((pos >= thr).mean())


def theta_e_bins(lo: float, hi: float, n: int) -> np.ndarray:
    """``n`` equal Einstein-radius bins over ``[lo, hi]`` → ``n+1`` edges."""
    return np.linspace(lo, hi, n + 1)


def tpr_vs_theta_e(
    scores: Sequence[float], labels: Sequence[int], theta_e: Sequence[float], *,
    target_fpr: float, bins: Sequence[float],
) -> Dict[str, object]:
    """TPR per Einstein-radius bin at a single global FPR threshold.

    The threshold is computed once on *all* negatives (fixed FPR across the test
    set); TPR is then measured per θ_E bin on the positives that fall in it.
    Negatives have no θ_E and only set the threshold. Returns
    ``{"threshold", "target_fpr", "bins": [{theta_lo, theta_hi, n_pos, tpr}]}``.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    te = np.asarray(theta_e, dtype=float)
    thr = threshold_at_fpr(s, y, target_fpr)
    edges = np.asarray(bins, dtype=float)
    pos = y == 1
    out: List[Dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = pos & (te >= lo) & (te < hi)
        n = int(m.sum())
        tpr = float((s[m] >= thr).mean()) if n > 0 else float("nan")
        out.append({"theta_lo": float(lo), "theta_hi": float(hi),
                    "n_pos": n, "tpr": tpr})
    return {"threshold": float(thr), "target_fpr": float(target_fpr),
            "bins": out}


def bootstrap_tpr_ci(
    scores: Sequence[float], labels: Sequence[int], *,
    target_fpr: float, n_boot: int = 500, seed: int = 0,
    alpha: float = 0.16,
) -> Dict[str, float]:
    """Bootstrap CI for overall TPR@FPR (resampling rows). ~1σ by default."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    n = s.size
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        thr = threshold_at_fpr(s[idx], y[idx], target_fpr)
        vals.append(tpr_at_threshold(s[idx], y[idx], thr))
    vals = np.asarray([v for v in vals if not np.isnan(v)], float)
    if vals.size == 0:
        return {"lo": float("nan"), "hi": float("nan")}
    return {"lo": float(np.quantile(vals, alpha)),
            "hi": float(np.quantile(vals, 1 - alpha))}
