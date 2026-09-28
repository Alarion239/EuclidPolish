"""Statistics of a study: paired bootstrap intervals and group bands.

A difference between two models is stated as the mean over fields of the
per-field difference with a 95 % paired bootstrap interval: the field indices
are resampled with replacement (``n`` times, ``numpy.random.default_rng(seed)``)
and the interval is the 2.5–97.5 percentile of the resampled means. Pairing
removes the (large) field-to-field spread that both models share.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from euclid_polish.eval.knee_psnr import integrated_psnr

#: Resamples of the paired bootstrap (the default everywhere in the studies).
BOOTSTRAP_RESAMPLES = 2000
#: Two-sided confidence level of the interval.
CONFIDENCE = 0.95
#: The spread band drawn around a group's median across members.
BAND_PERCENTILES = (16.0, 84.0)
#: Recipe fields a study can group / colour members by.
GROUP_FIELDS = ("loss", "training_knee", "asinh_knee", "output_knee", "knee_loss", "blocks",
                "bootstrap", "noise_aug", "icnr", "status", "op")
#: The group name of a member whose recipe lacks the grouping field.
MISSING = "—"


def paired_bootstrap(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray, *,
                     n: int = BOOTSTRAP_RESAMPLES, seed: int = 0) -> dict[str, Any]:
    """Mean of ``a − b`` over paired fields with its 95 % bootstrap interval:
    ``{mean, lo, hi, n_fields, n_resamples, seed, confidence}``. Fields where
    either value is not finite are dropped; fewer than two pairs raise."""
    x = np.asarray(a, np.float64)
    y = np.asarray(b, np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"paired values must be two 1-D arrays of one length "
                         f"({x.shape} vs {y.shape})")
    keep = np.isfinite(x) & np.isfinite(y)
    diff = (x - y)[keep]
    if diff.size < 2:
        raise ValueError("a paired interval needs at least two fields with both values")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, diff.size, size=(int(n), diff.size))
    means = diff[draws].mean(axis=1)
    tail = 100.0 * (1.0 - CONFIDENCE) / 2.0
    lo, hi = np.percentile(means, [tail, 100.0 - tail])
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi),
            "n_fields": int(diff.size), "n_resamples": int(n), "seed": int(seed),
            "confidence": CONFIDENCE}


def nan_mean(values: Sequence[Any] | np.ndarray, axis: int | None = None) -> np.ndarray:
    """Mean ignoring NaN (an all-NaN slice is NaN, without a warning) — the
    one averaging rule of a study's numbers and charts."""
    array = np.asarray(values, np.float64)
    finite = np.isfinite(array)
    count = finite.sum(axis=axis)
    total = np.where(finite, array, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def field_integrated(psnr: Sequence[Any] | np.ndarray, knees: Sequence[float]) -> np.ndarray:
    """Knee-integrated PSNR of every per-field curve: ``(models, fields,
    knees, bands)`` → ``(models, fields, bands)`` (NaN where a curve has a
    NaN). A model's integrated PSNR is :func:`nan_mean` of these over fields."""
    array = np.asarray(psnr, np.float64)
    if array.ndim != 4:
        raise ValueError(f"expected (models, fields, knees, bands), got {array.shape}")
    return integrated_psnr(np.moveaxis(array, 2, 0), knees)


def group_band(values: Sequence[Any] | np.ndarray) -> dict[str, Any]:
    """Median and p16–p84 across members (axis 0) of ``(members, …)``."""
    array = np.asarray(values, np.float64)
    if array.ndim == 0 or array.shape[0] == 0:
        raise ValueError("a group band needs at least one member")
    lo, hi = np.nanpercentile(array, BAND_PERCENTILES, axis=0)
    return {"median": np.nanmedian(array, axis=0), "lo": lo, "hi": hi,
            "n": int(array.shape[0])}


def group_key(value: Any) -> str:
    """A recipe value as a group name (lists → ``a+b``)."""
    if value is None or value == "":
        return MISSING
    if isinstance(value, list | tuple):
        return "+".join(f"{v:g}" if isinstance(v, int | float) else str(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def groups(rows: Sequence[Mapping[str, Any]], field: str | None) -> dict[str, list[str]]:
    """Member labels grouped by the recipe ``field`` (first-seen order);
    ``None`` puts every member in its own group."""
    out: dict[str, list[str]] = {}
    for row in rows:
        label = str(row.get("label"))
        key = label if not field else group_key(row.get(field))
        out.setdefault(key, []).append(label)
    return out


__all__ = [
    "BAND_PERCENTILES",
    "BOOTSTRAP_RESAMPLES",
    "CONFIDENCE",
    "GROUP_FIELDS",
    "MISSING",
    "field_integrated",
    "group_band",
    "group_key",
    "groups",
    "nan_mean",
    "paired_bootstrap",
]
