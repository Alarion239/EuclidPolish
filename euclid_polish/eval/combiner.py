"""Asinh-space all-inference ensemble combiner.

The production fit streams every validation pixel through bounded minibatches.
The older in-memory fitter remains available for small tests and callers that
already provide a compact array.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from euclid_polish.config import Config

_BAND_SCALE = {name: float(Config.get_band(name).asinh_stretch_scale_e)
               for name in Config.HR_TARGET_BAND_NAMES}
GATE_LEVEL_RANGE = (-1.0, 13.0)
DEFAULT_N_KERNELS = 128
ASINH_L1_SMOOTH_DELTA = 1e-4
ASINH_L1_GRADIENT_TOL = 1e-7
ASINH_L1_FTOL = 1e-10
DEFAULT_WITHIN_STAGE_MIN_SEPARATION = 0.35
CROSS_STAGE_MIN_SEPARATION = 0.1
MAX_MINIBATCH_RBF_SIGMA = 8.0
RAW_INCREMENTAL_MINMEANMAX_RBF_KIND = "raw_incremental_minmeanmax_rbf"
BAND_NAMES = tuple(Config.HR_TARGET_BAND_NAMES)


@dataclass(frozen=True)
class CombinerModelSpec:
    kind: str
    label: str
    artifact_dir: str
    payload_name: str
    cube_prefix: str
    feature_names: tuple[str, ...]
    default_kernels: int = DEFAULT_N_KERNELS
    default_min_usage: float = 0.0


_RAW_FEATURE_NAMES = (
    "all_member_inferences_asinh_v9_staged_global_logit_convex_output",
)
COMBINER_MODELS = {
    RAW_INCREMENTAL_MINMEANMAX_RBF_KIND: CombinerModelSpec(
        RAW_INCREMENTAL_MINMEANMAX_RBF_KIND,
        "minibatched convex all-asinh RBF",
        "raw_incremental_minmeanmax_rbf_combiner",
        "raw_incremental_minmeanmax_rbf_combiner_evals.json",
        "comb_raw_incremental_minmeanmax_rbf",
        _RAW_FEATURE_NAMES,
    ),
}
ACTIVE_COMBINER_KINDS = (RAW_INCREMENTAL_MINMEANMAX_RBF_KIND,)


def _band_scale(name: str) -> float:
    return _BAND_SCALE[name]


def _stretched_psnr_from_mse(mse: float) -> float:
    value = float(mse)
    if value <= 0.0:
        return float("inf")
    peak = float(Config.PSNR_PEAK_STRETCHED)
    return float(10.0 * np.log10(peak * peak / value))


def combiner_model_spec(kind: str | None = None) -> CombinerModelSpec:
    return COMBINER_MODELS[normalize_model_kind(kind)]


def normalize_model_kind(kind: str | None) -> str:
    key = str(kind or RAW_INCREMENTAL_MINMEANMAX_RBF_KIND).strip().lower()
    if key in {
        RAW_INCREMENTAL_MINMEANMAX_RBF_KIND,
        "incremental_raw_minmeanmax_rbf",
        "raw_minmeanmax_rbf",
        "residual_rbf",
    }:
        return RAW_INCREMENTAL_MINMEANMAX_RBF_KIND
    raise ValueError(f"unsupported combiner model kind: {kind!r}")


def _all_inference_features(member_pixels: np.ndarray) -> np.ndarray:
    values = np.asarray(member_pixels, np.float64)
    if values.ndim != 3:
        raise ValueError(f"expected (N,M,C) member pixels, got {values.shape}")
    if values.shape[2] != len(BAND_NAMES):
        raise ValueError(
            f"expected {len(BAND_NAMES)} bands, got {values.shape[2]}")
    scales = np.asarray([_band_scale(name) for name in BAND_NAMES], np.float64)
    return np.arcsinh(
        values / scales[None, None, :]
    ).reshape(len(values), -1)


def _member_interval_error_floor(member_values: np.ndarray,
                                 target_values: np.ndarray) -> np.ndarray:
    """Smallest per-band error attainable inside the member interval.

    Both inputs are expected in the space where error is measured.  During
    fitting that is asinh space, so the returned floor uses exactly the same
    units as the optimization objective.
    """
    members = np.asarray(member_values, np.float64)
    targets = np.asarray(target_values, np.float64)
    if members.ndim != 3 or targets.shape != (len(members), members.shape[2]):
        raise ValueError(
            "member-interval error expects (N,M,C) members and (N,C) targets")
    lower = np.min(members, axis=1)
    upper = np.max(members, axis=1)
    return np.maximum(
        np.maximum(lower - targets, targets - upper), 0.0)


def _recoverable_error(current_values: np.ndarray,
                       target_values: np.ndarray,
                       error_floor: np.ndarray) -> np.ndarray:
    """Error remaining after subtracting the member-interval error floor."""
    current = np.asarray(current_values, np.float64)
    targets = np.asarray(target_values, np.float64)
    floor = np.asarray(error_floor, np.float64)
    if current.shape != targets.shape or floor.shape != targets.shape:
        raise ValueError("recoverable error inputs must have matching shapes")
    return np.maximum(np.abs(current - targets) - floor, 0.0)


def _best_member_achievable_l1_gain(
    member_values: np.ndarray,
    target_values: np.ndarray,
    current_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return attainable shared-band improvement and its best-member floor."""
    members = np.asarray(member_values, np.float64)
    targets = np.asarray(target_values, np.float64)
    current = np.asarray(current_values, np.float64)
    if (members.ndim != 3
            or targets.shape != (len(members), members.shape[2])
            or current.shape != targets.shape):
        raise ValueError("achievable gain inputs have incompatible shapes")
    member_l1 = np.mean(
        np.abs(members - targets[:, None, :]), axis=2)
    floor = np.min(member_l1, axis=1)
    current_l1 = np.mean(np.abs(current - targets), axis=1)
    return np.maximum(current_l1 - floor, 0.0), floor


def _unit_rank(values: np.ndarray, *, higher_is_better: bool) -> np.ndarray:
    """Scale-free [0,1] ranks with equal values receiving equal scores."""
    raw = np.asarray(values, np.float64).reshape(-1)
    out = np.full(len(raw), np.nan, np.float64)
    finite = np.isfinite(raw)
    unique = np.unique(raw[finite])
    if not len(unique):
        return out
    if len(unique) == 1:
        out[finite] = 1.0
        return out
    ranks = np.searchsorted(unique, raw[finite]).astype(np.float64)
    ranks /= float(len(unique) - 1)
    out[finite] = ranks if higher_is_better else 1.0 - ranks
    return out

class FitBufferAccumulator:
    """Bounded, aligned four-band pixels for fitting the incremental combiner.

    Sampling is stratified by the brightest target band, but a selected pixel
    always retains every member and every band.  This alignment is what makes
    one shared asinh-space model possible.
    """

    def __init__(self, band_names, *, max_rows: int = 200_000,
                 n_bright_bins: int = 8, per_bin_per_field: int = 250,
                 level_range: tuple[float, float] = (-1.0, 12.0), seed: int = 0):
        self.band_names = tuple(band_names)
        self.max_rows = int(max_rows)
        self.n_bright_bins = int(n_bright_bins)
        self.per_bin_per_field = int(per_bin_per_field)
        self.edges = np.linspace(level_range[0], level_range[1],
                                 self.n_bright_bins + 1)
        self._rng = np.random.default_rng(seed)
        self._X: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        self._n = 0
        self._member_vis_psnr_sum: np.ndarray | None = None
        self._member_asinh_l1_sum: np.ndarray | None = None
        self._n_psnr_fields = 0
        self._coherence_accumulator = None

    def add(self, preds: np.ndarray, hr: np.ndarray) -> None:
        raw = np.asarray(preds, np.float32)
        target = np.asarray(hr, np.float32)
        if raw.ndim != 4 or target.ndim != 3:
            raise ValueError(f"expected (M,H,W,C)/(H,W,C), got {raw.shape}/{target.shape}")
        m, _, _, c = raw.shape
        if c != len(self.band_names) or target.shape[-1] != c:
            raise ValueError(f"expected bands {self.band_names}, got {c}")
        vis_index = self.band_names.index("VIS") if "VIS" in self.band_names else 0
        knee = float(Config.STRETCH_SCALE_E)
        truth_vis = np.arcsinh(
            np.asarray(target[..., vis_index], np.float64) / knee)
        member_psnr = np.empty(m, np.float64)
        for mi in range(m):
            pred_vis = np.arcsinh(
                np.asarray(raw[mi, ..., vis_index], np.float64) / knee)
            mse = float(np.mean((pred_vis - truth_vis) ** 2))
            member_psnr[mi] = _stretched_psnr_from_mse(mse)
        if self._member_vis_psnr_sum is None:
            self._member_vis_psnr_sum = np.zeros(m, np.float64)
        if len(self._member_vis_psnr_sum) != m:
            raise ValueError("fit-buffer member count changed between fields")
        self._member_vis_psnr_sum += member_psnr
        band_scales = np.asarray(
            [_band_scale(name) for name in self.band_names], np.float64)
        members_asinh = np.arcsinh(
            np.asarray(raw, np.float64) / band_scales[None, None, None, :])
        target_asinh = np.arcsinh(
            np.asarray(target, np.float64) / band_scales[None, None, :])
        member_l1 = np.mean(
            np.abs(members_asinh - target_asinh[None, ...]), axis=(1, 2, 3))
        if self._member_asinh_l1_sum is None:
            self._member_asinh_l1_sum = np.zeros(m, np.float64)
        self._member_asinh_l1_sum += member_l1
        self._n_psnr_fields += 1
        if raw.shape[1] == raw.shape[2]:
            from euclid_polish.eval.power_spectrum import EnsembleSpectrumAccumulator

            if self._coherence_accumulator is None:
                self._coherence_accumulator = EnsembleSpectrumAccumulator(
                    int(raw.shape[1]), float(Config.DEFAULT_PIXEL_SCALE),
                    collect_pairwise=False)
            self._coherence_accumulator.add(
                target[..., vis_index], np.mean(raw[..., vis_index], axis=0),
                raw[..., vis_index])
        if self._n >= self.max_rows:
            return
        pixels = raw.reshape(m, -1, c).transpose(1, 0, 2)
        targets = target.reshape(-1, c)
        scales = np.asarray([_band_scale(name) for name in self.band_names])
        target_asinh = np.arcsinh(targets / scales[None, :])
        level = np.max(target_asinh, axis=1)
        bin_idx = np.clip(np.digitize(level, self.edges) - 1,
                          0, self.n_bright_bins - 1)
        for bin_i in range(self.n_bright_bins):
            if self._n >= self.max_rows:
                break
            selected = np.where(bin_idx == bin_i)[0]
            if selected.size == 0:
                continue
            take = int(min(self.per_bin_per_field, selected.size,
                           self.max_rows - self._n))
            pick = self._rng.choice(selected, size=take, replace=False)
            self._X.append(pixels[pick])
            self._y.append(targets[pick])
            self._n += take

    def buffer(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._X:
            return (np.zeros((0, 0, len(self.band_names)), np.float32),
                    np.zeros((0, len(self.band_names)), np.float32))
        return (np.concatenate(self._X).astype(np.float32),
                np.concatenate(self._y).astype(np.float32))

    def member_validation_psnr(self) -> np.ndarray | None:
        if self._member_vis_psnr_sum is None or self._n_psnr_fields <= 0:
            return None
        return self._member_vis_psnr_sum.copy() / self._n_psnr_fields

    def member_validation_metrics(self) -> dict[str, np.ndarray]:
        metrics: dict[str, np.ndarray] = {}
        if self._n_psnr_fields > 0:
            if self._member_vis_psnr_sum is not None:
                metrics["vis_asinh_psnr"] = (
                    self._member_vis_psnr_sum.copy() / self._n_psnr_fields)
            if self._member_asinh_l1_sum is not None:
                metrics["asinh_l1"] = (
                    self._member_asinh_l1_sum.copy() / self._n_psnr_fields)
        if self._coherence_accumulator is not None:
            scores = self._coherence_accumulator.coherence_scores().get("scores", [])
            members = [row for row in scores
                       if str(row.get("id", "")).startswith("member_")]
            if members:
                members.sort(key=lambda row: int(str(row["id"]).split("_", 1)[1]))
                metrics["coherence_overall"] = np.asarray(
                    [row.get("overall", np.nan) for row in members], np.float64)
                metrics["coherence_sr"] = np.asarray(
                    [row.get("sr", np.nan) for row in members], np.float64)
        return metrics


SharedFitBufferAccumulator = FitBufferAccumulator

def _weighted_kmeans(rows: np.ndarray, sample_weight: np.ndarray,
                     n_clusters: int, *, seed: int, max_iter: int = 40,
                     tol: float = 1e-5,
                     existing_centers: np.ndarray | None = None,
                     min_separation: float = 0.0,
                     existing_min_separation: float | None = None,
                     ) -> np.ndarray:
    """Deterministic weighted Lloyd K-means with weighted K-means++ seeds.

    ``sample_weight`` affects both seed probability and centroid means.  This
    small NumPy implementation avoids making scikit-learn a runtime dependency
    of inference artifacts while preserving its standard weighted-K-means
    semantics.
    """
    points = np.asarray(rows, np.float64)
    weights = np.asarray(sample_weight, np.float64).reshape(-1)
    if points.ndim != 2 or len(points) != len(weights):
        raise ValueError("weighted K-means expects (N,D) rows and N weights")
    finite = np.all(np.isfinite(points), axis=1) & np.isfinite(weights)
    points = points[finite]
    weights = np.maximum(weights[finite], 0.0)
    if not len(points):
        return np.empty((0, rows.shape[1]), np.float64)
    if not np.any(weights > 0):
        weights = np.ones(len(points), np.float64)
    requested_k = min(max(1, int(n_clusters)), len(points))
    anchors = np.asarray(
        existing_centers if existing_centers is not None
        else np.empty((0, points.shape[1])), np.float64)
    if anchors.ndim != 2 or anchors.shape[1] != points.shape[1]:
        raise ValueError("existing weighted K-means centers have wrong shape")
    anchor_separation = (
        float(min_separation) if existing_min_separation is None
        else float(existing_min_separation))
    # Request extra candidates when a hard separation filter is active. This
    # lets K-means++ find a complete batch without accepting near-duplicates.
    separation_active = min_separation > 0 or anchor_separation > 0
    k = min(len(points), requested_k * 2 if separation_active else requested_k)
    rng = np.random.default_rng(seed)
    centers = np.empty((k, points.shape[1]), np.float64)
    chosen: list[int] = []
    def distance_matrix(left: np.ndarray, right: np.ndarray,
                        *, chunk_rows: int = 2048) -> np.ndarray:
        result = np.empty((len(left), len(right)), np.float64)
        for start in range(0, len(left), max(1, int(chunk_rows))):
            stop = min(len(left), start + max(1, int(chunk_rows)))
            delta = left[start:stop, None, :] - right[None, :, :]
            result[start:stop] = np.sum(delta * delta, axis=2)
        return result

    if len(anchors):
        anchor_distance = np.min(distance_matrix(points, anchors), axis=1)
        first_score = weights * anchor_distance
    else:
        first_score = weights
    if float(first_score.sum()) <= 0:
        first_score = weights
    first = int(rng.choice(len(points), p=first_score / first_score.sum()))
    centers[0] = points[first]
    chosen.append(first)
    nearest = np.sum((points - centers[0]) ** 2, axis=1)
    if len(anchors):
        nearest = np.minimum(nearest, anchor_distance)
    for ci in range(1, k):
        score = weights * nearest
        score[chosen] = 0.0
        if float(score.sum()) > 0:
            pick = int(rng.choice(len(points), p=score / score.sum()))
        else:
            available = np.ones(len(points), bool)
            available[chosen] = False
            candidates = np.flatnonzero(available)
            pick = int(candidates[np.argmax(weights[candidates])])
        centers[ci] = points[pick]
        chosen.append(pick)
        nearest = np.minimum(
            nearest, np.sum((points - centers[ci]) ** 2, axis=1))

    for _ in range(max(1, int(max_iter))):
        distance = distance_matrix(points, centers)
        labels = np.argmin(distance, axis=1)
        updated = centers.copy()
        nearest = distance[np.arange(len(points)), labels]
        for ci in range(k):
            hit = labels == ci
            mass = float(weights[hit].sum())
            if mass > 0:
                updated[ci] = np.average(
                    points[hit], axis=0, weights=weights[hit])
            else:
                updated[ci] = points[int(np.argmax(weights * nearest))]
        shift = np.sqrt(np.sum((updated - centers) ** 2, axis=1)).max()
        centers = updated
        if float(shift) <= float(tol):
            break
    if not separation_active:
        return centers[:requested_k]
    distance = distance_matrix(points, centers)
    labels = np.argmin(distance, axis=1)
    mass = np.asarray(
        [weights[labels == ci].sum() for ci in range(len(centers))])
    accepted: list[np.ndarray] = []
    floor2 = float(max(0.0, min_separation)) ** 2
    anchor_floor2 = float(max(0.0, anchor_separation)) ** 2
    for ci in np.argsort(mass)[::-1]:
        candidate = centers[int(ci)]
        anchor_ok = (not len(anchors) or float(np.min(np.sum(
            (anchors - candidate[None, :]) ** 2, axis=1))) >= anchor_floor2)
        accepted_ok = (not accepted or float(np.min(np.sum(
            (np.asarray(accepted) - candidate[None, :]) ** 2,
            axis=1))) >= floor2)
        if anchor_ok and accepted_ok:
            accepted.append(candidate)
            if len(accepted) >= requested_k:
                break
    return np.asarray(accepted, np.float64).reshape(-1, points.shape[1])

def _rbf_sigma_from_centers(centers: np.ndarray) -> float:
    centers = np.asarray(centers, np.float64)
    if len(centers) <= 1:
        return 1.0
    separation = np.sqrt(np.sum(
        (centers[:, None, :] - centers[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(separation, np.inf)
    nearest = np.min(separation, axis=1)
    positive = nearest[np.isfinite(nearest) & (nearest > 1e-8)]
    return max(0.5, float(np.median(positive)) * 1.25
               if len(positive) else 1.0)


def _local_capped_rbf_sigmas(centers: np.ndarray, *,
                             maximum: float = MAX_MINIBATCH_RBF_SIGMA,
                             chunk_rows: int = 256) -> np.ndarray:
    """Local center widths with an explicit normalized support ceiling."""
    points = np.asarray(centers, np.float64)
    if len(points) <= 1:
        return np.ones(len(points), np.float64)
    nearest = np.full(len(points), np.inf, np.float64)
    norms = np.sum(points * points, axis=1)
    chunk = max(1, int(chunk_rows))
    for start in range(0, len(points), chunk):
        stop = min(len(points), start + chunk)
        distance2 = (
            norms[start:stop, None] + norms[None, :]
            - 2.0 * points[start:stop] @ points.T
        )
        np.maximum(distance2, 0.0, out=distance2)
        rows = np.arange(stop - start)
        distance2[rows, start + rows] = np.inf
        nearest[start:stop] = np.sqrt(np.min(distance2, axis=1))
    return np.clip(0.75 * nearest, 0.5, float(maximum))


def _weighted_separated_center_indices(
    points: np.ndarray,
    sample_weight: np.ndarray,
    n_centers: int,
    *,
    min_separation: float,
    existing_centers: np.ndarray | None = None,
    existing_min_separation: float | None = None,
    seed: int,
) -> np.ndarray:
    """Weighted K-means++ candidates with stage-aware separation floors."""
    values = np.asarray(points, np.float64)
    weights = np.maximum(np.asarray(sample_weight, np.float64).reshape(-1), 0.0)
    if values.ndim != 2 or len(values) != len(weights):
        raise ValueError("separated centers expect (N,D) points and N weights")
    if not len(values) or int(n_centers) <= 0:
        return np.empty((0,), np.int64)
    if not np.any(weights > 0):
        weights = np.ones(len(values), np.float64)
    rng = np.random.default_rng(int(seed))
    anchors = (np.asarray(existing_centers, np.float64)
               if existing_centers is not None
               else np.empty((0, values.shape[1]), np.float64))
    if anchors.ndim != 2 or anchors.shape[1] != values.shape[1]:
        raise ValueError("existing separated centers have wrong shape")
    selected: list[int] = []
    available = np.ones(len(values), bool)
    if len(anchors):
        anchor_distance2 = np.full(len(values), np.inf, np.float64)
        for anchor in anchors:
            delta = values - anchor
            anchor_distance2 = np.minimum(
                anchor_distance2, np.sum(delta * delta, axis=1))
    else:
        anchor_distance2 = np.full(len(values), np.inf, np.float64)
    selected_distance2 = np.full(len(values), np.inf, np.float64)
    floor2 = float(max(0.0, min_separation)) ** 2
    anchor_floor2 = float(max(
        0.0,
        min_separation if existing_min_separation is None
        else existing_min_separation,
    )) ** 2
    nearest2 = np.minimum(anchor_distance2, selected_distance2)
    score = (weights * np.maximum(nearest2, 1e-12)
             if len(anchors) else weights.copy())
    for _ in range(min(int(n_centers), len(values))):
        eligible = available & (anchor_distance2 >= anchor_floor2)
        if selected:
            eligible &= selected_distance2 >= floor2
        score = np.where(eligible, score, 0.0)
        mass = float(np.sum(score))
        if mass <= 0.0:
            break
        picked = int(rng.choice(len(values), p=score / mass))
        selected.append(picked)
        available[picked] = False
        delta = values - values[picked]
        distance2 = np.sum(delta * delta, axis=1)
        selected_distance2 = np.minimum(selected_distance2, distance2)
        nearest2 = np.minimum(anchor_distance2, selected_distance2)
        score = weights * np.maximum(nearest2, 1e-12)
    return np.asarray(selected, np.int64)


def _rbf_basis_matrix(features: np.ndarray, centers: np.ndarray,
                      scales: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    values = np.asarray(features, np.float64)
    kernels = np.asarray(centers, np.float64)
    if not len(kernels):
        return np.zeros((len(values), 0), np.float32)
    safe_scales = np.maximum(np.asarray(scales, np.float64), 1e-8)
    normalized = values / safe_scales[None, :]
    normalized_centers = kernels / safe_scales[None, :]
    distance2 = (np.sum(normalized * normalized, axis=1)[:, None]
                 + np.sum(normalized_centers * normalized_centers, axis=1)[None, :]
                 - 2.0 * normalized @ normalized_centers.T)
    np.maximum(distance2, 0.0, out=distance2)
    widths = np.maximum(np.asarray(sigmas, np.float64), 1e-6)
    return np.exp(
        -0.5 * distance2 / (widths[None, :] * widths[None, :])
    ).astype(np.float32)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, np.float64)
    if values.ndim != 2 or not values.shape[1]:
        raise ValueError("softmax expects a nonempty (N,M) matrix")
    shifted = values - np.max(values, axis=1, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / np.sum(numerator, axis=1, keepdims=True)


def _best_psnr_initial_logits(
    member_psnr: np.ndarray,
    *,
    best_weight: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Finite softmax logits for a near-one-hot best-PSNR initialization."""
    scores = np.asarray(member_psnr, np.float64).reshape(-1)
    if not len(scores) or not np.any(np.isfinite(scores)):
        raise ValueError("best-PSNR initialization needs a finite member score")
    best = int(np.nanargmax(scores))
    if len(scores) == 1:
        probabilities = np.ones(1, np.float64)
    else:
        selected = float(np.clip(best_weight, 0.5, 1.0 - 1e-6))
        probabilities = np.full(
            len(scores), (1.0 - selected) / (len(scores) - 1), np.float64)
        probabilities[best] = selected
    logits = np.log(np.maximum(probabilities, 1e-12))
    logits -= np.mean(logits)
    return logits, probabilities, best

@dataclass
class RawIncrementalMinMeanMaxRBFCombiner:
    """Shared-band convex member gate in all-inference asinh space.

    A learned global logit plus localized RBF residual logits produce one
    shared softmax member-weight vector. All bands are mixed in asinh space, so
    every transformed output stays inside the member convex hull before the
    final per-band sinh transform.
    The historical class name is retained for artifact/API compatibility.
    """

    member_labels: list[str]
    n_kernels: int
    coefficients: np.ndarray       # (K, M), member-logit coefficients
    centers: np.ndarray            # (K, M*C), asinh member coordinates
    scales: np.ndarray             # (M*C,), distance normalization only
    sigmas: np.ndarray             # (K,), fixed per-increment widths
    increment_ids: np.ndarray      # (K,), allocation batch for diagnostics
    reference_features: np.ndarray # (M*C,), validation median
    output_floors: np.ndarray      # retained artifact field; unused by gate
    baseline_member_index: int | None = None
    global_logits: np.ndarray | None = None  # (M,), learned global prior
    band_names: tuple[str, ...] = BAND_NAMES
    level_range: tuple[float, float] = GATE_LEVEL_RANGE
    records_fp: str | None = None
    starfull: bool = True
    val_l1: float | None = None
    kind: str = RAW_INCREMENTAL_MINMEANMAX_RBF_KIND
    fit_meta: dict = field(default_factory=dict)
    member_weight_peaks: dict[str, list[float]] = field(default_factory=dict)
    member_weight_integrals: dict[str, list[float]] = field(default_factory=dict)
    sigma_scale: float = 1.0
    min_usage: float = 0.0
    max_prune_regret: float = 0.0
    min_peak_weight: float = 0.0
    member_importance: dict[str, list[float]] = field(default_factory=dict)
    member_ablation: dict = field(default_factory=dict)

    @property
    def bands(self) -> dict:
        return {}

    @property
    def weight_labels(self) -> list[str]:
        return list(self.member_labels)

    def features_from_electrons(self, pixels: np.ndarray) -> np.ndarray:
        raw = np.asarray(pixels, np.float64)
        if raw.ndim != 3 or raw.shape[1] != len(self.member_labels):
            raise ValueError(
                f"expected (N,{len(self.member_labels)},C) member pixels, got {raw.shape}")
        if raw.shape[2] != len(self.band_names):
            raise ValueError(f"expected {len(self.band_names)} bands, got {raw.shape[2]}")
        return _all_inference_features(raw)

    def _basis_from_features(self, features: np.ndarray) -> np.ndarray:
        return _rbf_basis_matrix(
            features, self.centers, self.scales, self.sigmas)

    def weights_from_electrons(self, pixels: np.ndarray, *,
                               chunk_rows: int = 4096) -> np.ndarray:
        """Return one shared convex member-weight vector per pixel."""
        features = self.features_from_electrons(pixels)
        coefficients = np.asarray(self.coefficients, np.float64)
        if coefficients.shape != (self.n_kernels, len(self.member_labels)):
            raise ValueError(
                "convex RBF coefficients must have shape (kernels, members)")
        global_logits = (
            np.zeros(len(self.member_labels), np.float64)
            if self.global_logits is None
            else np.asarray(self.global_logits, np.float64))
        if global_logits.shape != (len(self.member_labels),):
            raise ValueError("convex RBF global logits must match members")
        out = np.empty((len(features), len(self.member_labels)), np.float64)
        chunk = max(1, int(chunk_rows))
        for start in range(0, len(features), chunk):
            stop = min(len(features), start + chunk)
            logits = (global_logits[None, :]
                      + self._basis_from_features(features[start:stop])
                      @ coefficients)
            out[start:stop] = _softmax_rows(logits)
        return out

    def predict_pixels(self, pixels: np.ndarray) -> np.ndarray:
        raw = np.asarray(pixels, np.float64)
        scales = np.asarray(
            [_band_scale(name) for name in self.band_names], np.float64)
        members_asinh = np.arcsinh(raw / scales[None, None, :])
        weights = self.weights_from_electrons(raw)
        prediction_asinh = np.einsum(
            "nm,nmc->nc", weights, members_asinh, optimize=True)
        return np.sinh(prediction_asinh) * scales[None, :]

    def apply_field(self, preds: np.ndarray,
                    band_names: tuple[str, ...] | None = None) -> np.ndarray:
        raw = np.asarray(preds, np.float32)
        if raw.ndim != 4:
            raise ValueError(f"expected (M,H,W,C) member stack, got {raw.shape}")
        m, h, w, c = raw.shape
        names = tuple(band_names) if band_names is not None else self.band_names
        if (m != len(self.member_labels) or c != len(self.band_names)
                or tuple(names) != tuple(self.band_names)):
            raise ValueError(
                f"raw incremental RBF expects {len(self.member_labels)} members "
                f"and bands {self.band_names}, got {m} members and {names}")
        pixels = raw.reshape(m, h * w, c).transpose(1, 0, 2)
        return self.predict_pixels(pixels).reshape(h, w, c).astype(np.float32)

    def needed_member_indices(self) -> list[int]:
        return list(range(len(self.member_labels)))

    def member_pruned(self, index: int) -> bool:
        return False

    def without_member(self, index: int):
        raise ValueError("a raw incremental RBF must be refitted after membership changes")

    def surviving_members(self) -> dict[str, list[bool]]:
        return {"source": [True] * len(self.member_labels)}

    def upsample(self, ens, lr_array: np.ndarray) -> np.ndarray:
        return self.apply_field(ens.member_arrays(lr_array))

    def pca_weight_surface(self, pixels: np.ndarray, *, n_pc1: int = 31,
                           n_pc2: int = 31) -> dict:
        """PCA surface of shared convex member weights."""
        features = self.features_from_electrons(pixels)
        finite = np.all(np.isfinite(features), axis=1)
        features = features[finite]
        if len(features) < 2 or features.shape[1] < 2:
            return {"n_pixels": int(len(features)), "available": False}
        scales = np.maximum(np.asarray(self.scales, np.float64), 1e-8)
        feature_mean = np.mean(features, axis=0)
        normalized = (features - feature_mean[None, :]) / scales[None, :]
        covariance = normalized.T @ normalized / max(1, len(normalized) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order[:2]], 0.0)
        components = eigenvectors[:, order[:2]].T
        for component in components:
            pivot = int(np.argmax(np.abs(component)))
            if component[pivot] < 0:
                component *= -1.0
        scores = normalized @ components.T
        center_geometry = ((np.asarray(self.centers, np.float64)
                            - feature_mean[None, :]) / scales[None, :])
        center_scores = (center_geometry @ components.T if len(center_geometry)
                         else np.empty((0, 2), np.float64))
        bounds = []
        for ci in range(2):
            lo, hi = np.quantile(scores[:, ci], (0.005, 0.995))
            if len(center_scores):
                lo = min(float(lo), float(np.min(center_scores[:, ci])))
                hi = max(float(hi), float(np.max(center_scores[:, ci])))
            span = max(float(hi - lo), 1e-3)
            bounds.append((float(min(lo, 0.0) - 0.05 * span),
                           float(max(hi, 0.0) + 0.05 * span)))
        pc1 = np.linspace(*bounds[0], max(3, int(n_pc1)))
        pc2 = np.linspace(*bounds[1], max(3, int(n_pc2)))
        xx, yy = np.meshgrid(pc1, pc2, indexing="xy")
        path = (feature_mean[None, :]
                + (xx.reshape(-1, 1) * components[0][None, :]
                   + yy.reshape(-1, 1) * components[1][None, :]) * scales[None, :])
        basis = self._basis_from_features(path)
        global_logits = (
            np.zeros(len(self.member_labels), np.float64)
            if self.global_logits is None
            else np.asarray(self.global_logits, np.float64))
        weights = _softmax_rows(
            global_logits[None, :]
            + basis @ np.asarray(self.coefficients, np.float64))
        feature_names = [
            f"{label}:{band} asinh" for label in self.member_labels
            for band in self.band_names
        ]
        total_variance = max(float(np.trace(covariance)), 1e-12)
        return {
            "available": True,
            "n_pixels": int(len(features)),
            "feature_space": "scale-normalized all-member asinh inferences",
            "conditioning_note": (
                "The surface shows the shared convex member weights along PC1 "
                "and PC2; all remaining PCs stay at their validation mean."),
            "pc1": pc1, "pc2": pc2,
            "center_pc1": center_scores[:, 0],
            "center_pc2": center_scores[:, 1],
            "weights": weights.reshape(len(pc2), len(pc1), -1),
            "explained_variance_ratio": eigenvalues / total_variance,
            "feature_names": feature_names,
            "loadings": components.copy(),
            "z_label": "shared convex member weight [0-1]",
            "surface_labels": list(self.member_labels),
        }

def _fit_raw_incremental_minmeanmax_rbf(
    Xtr: np.ndarray, ytr: np.ndarray, Xval: np.ndarray, yval: np.ndarray,
    labels: list[str], names: tuple[str, ...], *, n_kernels: int,
    seed: int, residual_abort: float = 1e-3,
    increment_size: int = 16, within_increment_separation: float = 0.35,
    ridge: float = 1e-5, max_optimizer_iterations: int = 500,
    member_validation_psnr: np.ndarray | None = None,
    member_validation_metrics: dict[str, np.ndarray] | None = None,
) -> RawIncrementalMinMeanMaxRBFCombiner:
    """Fit shared convex member-logit RBF blocks from a uniform ensemble.

    Center exclusion applies across every increment, so no two kernels can
    create a near-discontinuous transition in normalized asinh feature space.
    After every increment, all accumulated member-logit coefficients are
    jointly re-optimized with deterministic full-batch L-BFGS. The previous
    optimum plus zero-valued new logits is only the warm start.
    """
    del member_validation_psnr, member_validation_metrics
    rng = np.random.default_rng(seed)
    train_features = _all_inference_features(Xtr)
    val_features = _all_inference_features(Xval)
    reference = np.median(train_features, axis=0)
    q_lo, q_hi = np.quantile(train_features, (0.005, 0.995), axis=0)
    scales = np.maximum((q_hi - q_lo) / 4.0, 1e-3)
    normalized = (train_features - reference[None, :]) / scales[None, :]

    target_k = min(max(1, int(n_kernels)), len(normalized))
    geometry_cap = min(len(normalized), max(50_000, target_k * 1024))
    geometry_idx = (np.arange(len(normalized)) if len(normalized) <= geometry_cap
                    else np.sort(rng.choice(
                        len(normalized), geometry_cap, replace=False)))
    geometry = normalized[geometry_idx]

    band_scales = np.asarray([_band_scale(name) for name in names], np.float64)
    train_targets_asinh = np.arcsinh(
        np.asarray(ytr, np.float64) / band_scales[None, :])
    val_targets_asinh = np.arcsinh(
        np.asarray(yval, np.float64) / band_scales[None, :])
    train_members_asinh = np.arcsinh(
        np.asarray(Xtr, np.float64) / band_scales[None, None, :])
    val_members_asinh = np.arcsinh(
        np.asarray(Xval, np.float64) / band_scales[None, None, :])
    train_error_floor = _member_interval_error_floor(
        train_members_asinh, train_targets_asinh)
    train_pred_asinh = np.mean(train_members_asinh, axis=1)
    val_pred_asinh = np.mean(val_members_asinh, axis=1)
    initial_val_l1 = float(np.mean(np.abs(val_pred_asinh - val_targets_asinh)))
    initial_val_vis_mse = float(np.mean(
        (val_pred_asinh[:, 0] - val_targets_asinh[:, 0]) ** 2))
    output_floors = []
    for ci in range(len(names)):
        positive = np.abs(train_targets_asinh[:, ci])
        positive = positive[positive > 1e-8]
        output_floors.append(max(
            1e-3,
            float(np.quantile(positive, 0.10)) if len(positive) else 1e-3))
    output_floors = np.asarray(output_floors, np.float64)

    all_centers: list[np.ndarray] = []
    all_sigmas: list[np.ndarray] = []
    all_increment_ids: list[np.ndarray] = []
    train_basis = np.empty((len(Xtr), 0), np.float32)
    val_basis = np.empty((len(Xval), 0), np.float32)
    coefficients = np.empty((0, len(labels)), np.float64)
    history: list[dict[str, float | int | str | bool]] = []
    best_val_l1 = initial_val_l1
    best_val_vis_mse = initial_val_vis_mse
    best_k = 0
    best_centers = np.empty((0, train_features.shape[1]), np.float64)
    best_sigmas = np.empty((0,), np.float64)
    best_coefficients = np.empty((0, len(labels)), np.float64)
    best_increment_ids = np.empty((0,), np.int32)
    abort_reason = "kernel_limit"
    stage = 0

    def basis(features: np.ndarray, centers: np.ndarray,
              sigmas: np.ndarray, *, chunk_rows: int = 8192) -> np.ndarray:
        out = np.empty((len(features), len(centers)), np.float32)
        widths = np.maximum(np.asarray(sigmas, np.float64), 1e-6)
        normalized_centers = centers / scales[None, :]
        center_norm2 = np.sum(normalized_centers * normalized_centers, axis=1)
        for start in range(0, len(features), max(1, int(chunk_rows))):
            stop = min(len(features), start + max(1, int(chunk_rows)))
            normalized = features[start:stop] / scales[None, :]
            distance2 = (np.sum(normalized * normalized, axis=1)[:, None]
                         + center_norm2[None, :]
                         - 2.0 * normalized @ normalized_centers.T)
            np.maximum(distance2, 0.0, out=distance2)
            out[start:stop] = np.exp(
                -0.5 * distance2
                / (widths[None, :] * widths[None, :])).astype(np.float32)
        return out

    def jointly_optimize(phi: np.ndarray, initial: np.ndarray):
        from scipy.optimize import minimize

        k = phi.shape[1]
        members = len(labels)
        normalization = float(max(1, len(phi) * len(names)))

        def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
            current = np.asarray(flat, np.float64).reshape(k, members)
            weights = _softmax_rows(phi @ current)
            prediction_asinh = np.einsum(
                "nm,nmc->nc", weights, train_members_asinh, optimize=True)
            error = prediction_asinh - train_targets_asinh
            smooth = np.sqrt(error * error + ASINH_L1_SMOOTH_DELTA ** 2)
            data_loss = float(np.sum(
                smooth - ASINH_L1_SMOOTH_DELTA) / normalization)
            regularization = 0.5 * float(ridge) * float(np.sum(current ** 2))
            prediction_gradient = (error / smooth) / normalization
            weight_gradient = np.einsum(
                "nc,nmc->nm", prediction_gradient,
                train_members_asinh, optimize=True)
            logit_gradient = weights * (
                weight_gradient
                - np.sum(weight_gradient * weights, axis=1, keepdims=True))
            gradient = phi.T @ logit_gradient + float(ridge) * current
            return data_loss + regularization, gradient.reshape(-1)

        x = np.asarray(initial, np.float64).reshape(-1)
        total_iterations = 0
        result = None
        for _attempt in range(2):
            result = minimize(
                objective, x, method="L-BFGS-B", jac=True,
                options={"maxiter": max(1, int(max_optimizer_iterations)),
                         "maxls": 40, "ftol": ASINH_L1_FTOL,
                         "gtol": ASINH_L1_GRADIENT_TOL})
            total_iterations += int(result.nit)
            x = np.asarray(result.x, np.float64)
            if bool(result.success) or int(result.status) != 1:
                break
        final_loss, final_gradient = objective(x)
        return (
            x.reshape(k, members), bool(result.success), [total_iterations],
            float(np.max(np.abs(final_gradient))), float(final_loss),
            str(result.message),
        )

    while sum(len(batch) for batch in all_centers) < target_k:
        placement_recoverable_error = _recoverable_error(
            train_pred_asinh, train_targets_asinh, train_error_floor)
        row_placement_weight = np.mean(placement_recoverable_error, axis=1)
        max_placement_weight = float(np.max(row_placement_weight))
        if max_placement_weight <= float(residual_abort):
            abort_reason = "all_train_recoverable_errors_below_threshold"
            break
        stage += 1
        used = sum(len(batch) for batch in all_centers)
        add_count = min(max(1, int(increment_size)), target_k - used)
        center_weight = row_placement_weight[geometry_idx]
        existing_norm = (
            np.concatenate([
                (batch - reference[None, :]) / scales[None, :]
                for batch in all_centers
            ], axis=0)
            if all_centers else None
        )
        new_norm = _weighted_kmeans(
            geometry, center_weight, add_count, seed=seed + stage,
            existing_centers=existing_norm,
            min_separation=float(within_increment_separation),
            existing_min_separation=float(CROSS_STAGE_MIN_SEPARATION))
        if not len(new_norm):
            abort_reason = "global_center_separation_exhausted"
            break
        new_sigma = _rbf_sigma_from_centers(new_norm)
        new_sigmas = np.full(len(new_norm), new_sigma, np.float64)
        new_centers = new_norm * scales[None, :] + reference[None, :]
        phi_train = basis(train_features, new_centers, new_sigmas)
        phi_val = basis(val_features, new_centers, new_sigmas)
        train_basis = np.concatenate((train_basis, phi_train), axis=1)
        val_basis = np.concatenate((val_basis, phi_val), axis=1)
        coefficients = np.concatenate(
            (coefficients, np.zeros((len(new_centers), len(labels)), np.float64)),
            axis=0)
        initial_coefficients = coefficients.copy()
        (coefficients, optimizer_converged, optimizer_iterations_by_band,
         optimizer_gradient_inf, optimizer_loss,
         optimizer_message) = jointly_optimize(train_basis, coefficients)
        parameter_delta_norm = float(np.linalg.norm(
            coefficients - initial_coefficients))
        new_block_norm = float(np.linalg.norm(coefficients[used:]))
        optimizer_progress = bool(
            sum(optimizer_iterations_by_band) > 0
            and parameter_delta_norm > 1e-10)
        train_weights = _softmax_rows(train_basis @ coefficients)
        train_pred_asinh = np.einsum(
            "nm,nmc->nc", train_weights, train_members_asinh, optimize=True)
        train_recoverable_error = _recoverable_error(
            train_pred_asinh, train_targets_asinh, train_error_floor)
        val_weights = _softmax_rows(val_basis @ coefficients)
        val_pred_asinh = np.einsum(
            "nm,nmc->nc", val_weights, val_members_asinh, optimize=True)

        all_centers.append(new_centers)
        all_sigmas.append(new_sigmas)
        all_increment_ids.append(np.full(len(new_norm), stage, np.int32))
        total_k = sum(len(batch) for batch in all_centers)
        train_l1 = float(np.mean(np.abs(train_pred_asinh - train_targets_asinh)))
        val_l1 = float(np.mean(np.abs(val_pred_asinh - val_targets_asinh)))
        val_vis_mse = float(np.mean(
            (val_pred_asinh[:, 0] - val_targets_asinh[:, 0]) ** 2))
        selected = bool(
            optimizer_progress and np.isfinite(val_vis_mse) and np.isfinite(val_l1)
            and val_vis_mse < best_val_vis_mse - 1e-12
            and val_l1 < best_val_l1 - 1e-9)
        if selected:
            best_val_l1 = val_l1
            best_val_vis_mse = val_vis_mse
            best_k = total_k
            best_centers = np.concatenate(all_centers, axis=0).copy()
            best_sigmas = np.concatenate(all_sigmas).copy()
            best_coefficients = coefficients.copy()
            best_increment_ids = np.concatenate(all_increment_ids).copy()
        history.append({
            "stage": stage,
            "n_centers": int(total_k),
            "added_centers": int(len(new_norm)),
            "sigma": float(new_sigma),
            "train_mean_residual": train_l1,
            "train_max_residual": float(np.max(np.mean(
                np.abs(train_targets_asinh - train_pred_asinh), axis=1))),
            "train_mean_minimum_possible_l1": float(np.mean(train_error_floor)),
            "center_weight_mean_recoverable_l1": float(
                np.mean(placement_recoverable_error)),
            "center_weight_max_recoverable_l1": max_placement_weight,
            "train_mean_recoverable_l1": float(
                np.mean(train_recoverable_error)),
            "train_max_recoverable_l1": float(np.max(np.mean(
                train_recoverable_error, axis=1))),
            "val_l1": val_l1,
            "val_vis_asinh_mse": val_vis_mse,
            "val_vis_asinh_psnr": _stretched_psnr_from_mse(val_vis_mse),
            "val_improvement_from_uniform_asinh_mean": initial_val_l1 - val_l1,
            "selected_by_validation": selected,
            "optimizer_converged": bool(optimizer_converged),
            "optimizer_iterations": int(sum(optimizer_iterations_by_band)),
            "optimizer_iterations_by_band": [
                int(value) for value in optimizer_iterations_by_band],
            "optimizer_gradient_inf": float(optimizer_gradient_inf),
            "optimizer_objective": float(optimizer_loss),
            "optimizer_message": optimizer_message,
            "optimizer_progress": optimizer_progress,
            "parameter_delta_norm": parameter_delta_norm,
            "new_block_norm": new_block_norm,
        })
        if not optimizer_progress:
            abort_reason = "joint_optimizer_stalled_after_increment"
            break

    # Validation chooses a prefix of complete increments, including the exact
    # uniform convex average when every learned block generalizes poorly.
    centers = best_centers
    sigmas = best_sigmas
    coefficients = best_coefficients
    increment_ids = best_increment_ids
    return RawIncrementalMinMeanMaxRBFCombiner(
        member_labels=labels, n_kernels=int(best_k),
        coefficients=coefficients.astype(np.float32),
        centers=centers.astype(np.float32), scales=scales.astype(np.float32),
        sigmas=sigmas.astype(np.float32), increment_ids=increment_ids,
        reference_features=reference.astype(np.float32),
        output_floors=output_floors.astype(np.float32), band_names=names,
        baseline_member_index=None,
        val_l1=float(best_val_l1),
        fit_meta={
            "shared_across_bands": True,
            "features": "all member asinh inferences",
            "feature_schema": _RAW_FEATURE_NAMES[0],
            "feature_dimension": int(train_features.shape[1]),
            "input_members": int(Xtr.shape[1]),
            "input_bands": int(Xtr.shape[2]),
            "initial_prediction": "uniform_member_average_in_asinh_space",
            "baseline_member_index": None,
            "baseline_selection_metric": "not_applicable_convex_member_gate",
            "input_space": "per_band_asinh",
            "output": "shared_weight_convex_member_average_in_asinh_space",
            "output_space": "per_band_asinh_then_electrons",
            "output_activation": "member_softmax_then_asinh_average_then_sinh",
            "signed_sky_subtracted_output": True,
            "loss": "smooth_asinh_l1_plus_ridge",
            "coefficient_parameterization": "rbf_member_logits",
            "ridge_normalized_coefficient_l2": float(ridge),
            "optimizer": "joint_full_batch_lbfgs_after_every_increment",
            "optimizer_acceptance": (
                "joint validation equal-band asinh-L1 and VIS asinh-MSE "
                "improvement plus nonzero parameter progress; "
                "solver convergence is diagnostic"),
            "validation_prefix_metric": "joint_asinh_L1_and_VIS_asinh_MSE",
            "optimizer_warm_start": "previous_joint_optimum_plus_zero_new_block",
            "optimizer_max_iterations_per_call": int(max_optimizer_iterations),
            "optimizer_max_continuations": 2,
            "requested_kernels": int(target_k),
            "selected_kernels": int(best_k),
            "learned_parameter_count": int(best_k * len(labels)),
            "stored_parameter_count": int(best_k * (train_features.shape[1]
                                                       + len(labels))),
            "increment_size": int(increment_size),
            "within_increment_min_separation_normalized": float(
                within_increment_separation),
            "cross_increment_min_separation_normalized": float(
                CROSS_STAGE_MIN_SEPARATION),
            "minimum_center_separation_normalized": float(min(
                within_increment_separation, CROSS_STAGE_MIN_SEPARATION)),
            "kernel_width_rule": "per_increment_median_nearest_distance_x1.25",
            "residual_weight": "current_equal_band_recoverable_asinh_l1",
            "center_weight_rule": (
                "mean(max(abs(current-target)-distance(target,"
                "[member_min,member_max]),0)) across bands in asinh space"),
            "minimum_possible_error": (
                "per-band distance from target to the member-prediction interval"),
            "asinh_scales_e": band_scales.tolist(),
            "asinh_l1_smooth_delta": float(ASINH_L1_SMOOTH_DELTA),
            "asinh_l1_gradient_tolerance": float(ASINH_L1_GRADIENT_TOL),
            "asinh_l1_function_tolerance": float(ASINH_L1_FTOL),
            "residual_abort_threshold_asinh": float(residual_abort),
            "initial_val_l1": float(initial_val_l1),
            "initial_val_vis_asinh_mse": float(initial_val_vis_mse),
            "initial_val_vis_asinh_psnr": _stretched_psnr_from_mse(
                initial_val_vis_mse),
            "selected_val_vis_asinh_mse": float(best_val_vis_mse),
            "selected_val_vis_asinh_psnr": _stretched_psnr_from_mse(
                best_val_vis_mse),
            "center_history": history,
            "center_abort_reason": abort_reason,
        })

def fit_combiner_minibatched(
    field_factory: Callable[
        [Sequence[int]],
        Iterable[tuple[int, np.ndarray, np.ndarray]],
    ],
    field_indices: Sequence[int],
    member_labels: Sequence[str],
    *,
    band_names: Sequence[str] = BAND_NAMES,
    n_kernels: int = DEFAULT_N_KERNELS,
    seed: int = 0,
    holdout_fields: float = 0.1,
    batch_rows: int = 8192,
    epochs: int = 1,
    learning_rate: float = 0.01,
    ridge: float = 1e-5,
    normalizer_rows: int = 100_000,
    candidate_rows: int | None = None,
    increment_size: int = 16,
    initial_best_weight: float = 0.99,
    member_validation_psnr: np.ndarray | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> RawIncrementalMinMeanMaxRBFCombiner:
    """Fit staged shared-band RBF logits from every streamed field pixel.

    Each stage rescans the training fields for improvement that is definitely
    attainable by selecting one shared four-band member, adds at most 16
    weighted K-means++ centers, and jointly refits every accumulated RBF
    coefficient plus the global member logits with minibatch Adam.
    """
    labels = [str(value) for value in member_labels]
    names = tuple(str(value) for value in band_names)
    indices = np.asarray(sorted({int(value) for value in field_indices}), np.int64)
    if not len(indices):
        raise ValueError("minibatched combiner needs at least one field")
    if not labels or not names:
        raise ValueError("minibatched combiner needs members and bands")

    split_rng = np.random.default_rng(int(seed))
    shuffled_fields = split_rng.permutation(indices)
    if len(indices) == 1:
        train_fields = val_fields = indices.copy()
    else:
        n_val_fields = min(
            len(indices) - 1,
            max(1, int(round(len(indices) * float(holdout_fields)))),
        )
        val_fields = np.sort(shuffled_fields[:n_val_fields])
        train_fields = np.sort(shuffled_fields[n_val_fields:])

    def tick(position: int, total: int, label: str) -> None:
        if progress is not None:
            progress(int(position), int(total), str(label))

    # Pass 1: collect a bounded reservoir for robust feature normalization and
    # compute a pixel-weighted training PSNR fallback for the global prior.
    normalization_member_rows: list[np.ndarray] = []
    normalization_targets: list[np.ndarray] = []
    normalization_per_field = max(
        256, int(np.ceil(max(1, int(normalizer_rows)) / len(train_fields))))
    normalizer_rng = np.random.default_rng(int(seed) + 101)
    train_pixel_count = 0
    vis_index = names.index("VIS") if "VIS" in names else 0
    vis_scale = float(_band_scale(names[vis_index]))
    member_vis_squared = np.zeros(len(labels), np.float64)
    member_vis_pixels = 0
    for position, (field_index, raw0, target0) in enumerate(
            field_factory(train_fields.tolist()), 1):
        raw = np.asarray(raw0, np.float32)
        target = np.asarray(target0, np.float32)
        if raw.ndim != 4 or target.ndim != 3:
            raise ValueError(
                f"expected (M,H,W,C)/(H,W,C), got {raw.shape}/{target.shape}")
        if raw.shape[0] != len(labels) or raw.shape[-1] != len(names):
            raise ValueError("streamed field does not match members/bands")
        pixels = raw.reshape(len(labels), -1, len(names)).transpose(1, 0, 2)
        targets = target.reshape(-1, len(names))
        train_pixel_count += len(pixels)
        members_vis_asinh = np.arcsinh(
            np.asarray(raw[..., vis_index], np.float64) / vis_scale)
        target_vis_asinh = np.arcsinh(
            np.asarray(target[..., vis_index], np.float64) / vis_scale)
        member_vis_squared += np.sum(
            (members_vis_asinh - target_vis_asinh[None, ...]) ** 2,
            axis=(1, 2),
        )
        member_vis_pixels += int(target_vis_asinh.size)
        take = min(len(pixels), normalization_per_field)
        picked = normalizer_rng.choice(len(pixels), size=take, replace=False)
        normalization_member_rows.append(pixels[picked])
        normalization_targets.append(targets[picked])
        tick(position, len(train_fields),
             f"feature normalization field {field_index}")
    if not normalization_member_rows:
        raise ValueError("streamed training fields produced no pixels")

    normalization_pixels = np.concatenate(normalization_member_rows)
    normalization_features = _all_inference_features(normalization_pixels)
    reference = np.median(normalization_features, axis=0)
    q_lo, q_hi = np.quantile(
        normalization_features, (0.005, 0.995), axis=0)
    feature_scales = np.maximum((q_hi - q_lo) / 4.0, 1e-3)
    band_scales = np.asarray([_band_scale(name) for name in names], np.float64)

    supplied_psnr = (None if member_validation_psnr is None
                     else np.asarray(member_validation_psnr, np.float64))
    if (supplied_psnr is not None
            and supplied_psnr.shape == (len(labels),)
            and np.any(np.isfinite(supplied_psnr))):
        initialization_psnr = supplied_psnr.copy()
        psnr_source = "supplied member evaluation PSNR"
    else:
        initialization_psnr = np.asarray([
            _stretched_psnr_from_mse(value / max(1, member_vis_pixels))
            for value in member_vis_squared
        ], np.float64)
        psnr_source = "streamed training-field VIS asinh PSNR"
    initial_global_logits, initial_probabilities, best_member_index = (
        _best_psnr_initial_logits(
            initialization_psnr, best_weight=float(initial_best_weight)))

    target_k = max(1, int(n_kernels))
    candidate_cap = max(target_k, int(candidate_rows or max(
        16_384, min(131_072, target_k * 64))))
    per_field_candidates = max(
        64, int(np.ceil(candidate_cap / len(train_fields))))
    basis_element_budget = 4_000_000

    positive_targets = np.abs(np.arcsinh(
        np.asarray(np.concatenate(normalization_targets), np.float64)
        / band_scales[None, :]))
    output_floors = []
    for channel in range(len(names)):
        positive = positive_targets[:, channel]
        positive = positive[positive > 1e-8]
        output_floors.append(max(
            1e-3,
            float(np.quantile(positive, 0.10)) if len(positive) else 1e-3,
        ))
    output_floors = np.asarray(output_floors, np.float64)

    def basis_rows(batch: np.ndarray, centers: np.ndarray,
                   sigmas: np.ndarray) -> np.ndarray:
        return _rbf_basis_matrix(
            _all_inference_features(batch), centers, feature_scales, sigmas)

    def predict_asinh(batch: np.ndarray, centers: np.ndarray,
                      sigmas: np.ndarray, theta: np.ndarray,
                      global_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        members_asinh = np.arcsinh(
            np.asarray(batch, np.float64) / band_scales[None, None, :])
        logits = global_logits[None, :]
        if len(centers):
            logits = logits + basis_rows(batch, centers, sigmas) @ theta
        weights = _softmax_rows(logits)
        prediction = np.einsum(
            "nm,nmc->nc", weights, members_asinh, optimize=True)
        return prediction, members_asinh

    def effective_rows(center_count: int) -> int:
        return max(1, min(
            max(1, int(batch_rows)),
            basis_element_budget // max(1, int(center_count)),
        ))

    def evaluate(centers: np.ndarray, sigmas: np.ndarray, theta: np.ndarray,
                 global_logits: np.ndarray, selected_fields: np.ndarray,
                 label: str) -> tuple[float, float, int]:
        l1_total = 0.0
        vis_squared_total = 0.0
        pixel_total = 0
        rows = effective_rows(len(centers))
        for position, (field_index, raw0, target0) in enumerate(
                field_factory(selected_fields.tolist()), 1):
            raw = np.asarray(raw0, np.float32)
            target = np.asarray(target0, np.float32)
            pixels = raw.reshape(
                len(labels), -1, len(names)).transpose(1, 0, 2)
            targets = target.reshape(-1, len(names))
            for start in range(0, len(pixels), rows):
                stop = min(len(pixels), start + rows)
                batch = pixels[start:stop]
                prediction_asinh, _ = predict_asinh(
                    batch, centers, sigmas, theta, global_logits)
                target_asinh = np.arcsinh(
                    np.asarray(targets[start:stop], np.float64)
                    / band_scales[None, :])
                error = prediction_asinh - target_asinh
                l1_total += float(np.sum(np.abs(error)))
                vis_squared_total += float(np.sum(error[:, 0] ** 2))
                pixel_total += len(error)
            tick(position, len(selected_fields),
                 f"{label} field {field_index}")
        if pixel_total <= 0:
            raise ValueError("streamed validation fields produced no pixels")
        return (
            l1_total / (pixel_total * len(names)),
            vis_squared_total / pixel_total,
            pixel_total,
        )

    candidate_rng = np.random.default_rng(int(seed) + 211)

    def collect_candidates(stage: int, centers: np.ndarray,
                           sigmas: np.ndarray, theta: np.ndarray,
                           global_logits: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray, float, float]:
        candidate_member_rows: list[np.ndarray] = []
        candidate_weights: list[np.ndarray] = []
        gain_sum = 0.0
        gain_count = 0
        gain_max = 0.0
        rows = effective_rows(len(centers))
        for position, (field_index, raw0, target0) in enumerate(
                field_factory(train_fields.tolist()), 1):
            raw = np.asarray(raw0, np.float32)
            target = np.asarray(target0, np.float32)
            pixels = raw.reshape(
                len(labels), -1, len(names)).transpose(1, 0, 2)
            targets = target.reshape(-1, len(names))
            gains = np.empty(len(pixels), np.float32)
            for start in range(0, len(pixels), rows):
                stop = min(len(pixels), start + rows)
                prediction, members_asinh = predict_asinh(
                    pixels[start:stop], centers, sigmas, theta, global_logits)
                target_asinh = np.arcsinh(
                    np.asarray(targets[start:stop], np.float64)
                    / band_scales[None, :])
                gain, _ = _best_member_achievable_l1_gain(
                    members_asinh, target_asinh, prediction)
                gains[start:stop] = gain.astype(np.float32)
            gain_sum += float(np.sum(gains, dtype=np.float64))
            gain_count += len(gains)
            gain_max = max(gain_max, float(np.max(gains, initial=0.0)))
            hard_take = min(
                len(pixels), max(1, int(per_field_candidates * 0.8)))
            hard = (np.arange(len(pixels)) if hard_take == len(pixels)
                    else np.argpartition(
                        gains, len(gains) - hard_take)[-hard_take:])
            remaining_take = min(
                len(pixels) - hard_take,
                max(0, per_field_candidates - hard_take),
            )
            if remaining_take:
                available = np.ones(len(pixels), bool)
                available[hard] = False
                random_rows = candidate_rng.choice(
                    np.flatnonzero(available),
                    size=remaining_take,
                    replace=False,
                )
                chosen = np.concatenate((hard, random_rows))
            else:
                chosen = hard
            candidate_member_rows.append(pixels[chosen])
            candidate_weights.append(np.maximum(gains[chosen], 1e-12))
            tick(position, len(train_fields),
                 f"stage {stage} achievable-gain scan field {field_index}")
        candidates = np.concatenate(candidate_member_rows)
        placement = np.concatenate(candidate_weights).astype(np.float64)
        if len(candidates) > candidate_cap:
            keep = np.argpartition(placement, -candidate_cap)[-candidate_cap:]
            candidates = candidates[keep]
            placement = placement[keep]
        return (
            candidates,
            placement,
            gain_sum / max(1, gain_count),
            gain_max,
        )

    centers = np.empty((0, len(labels) * len(names)), np.float64)
    sigmas = np.empty((0,), np.float64)
    increment_ids = np.empty((0,), np.int32)
    theta = np.empty((0, len(labels)), np.float64)
    global_logits = initial_global_logits.copy()
    initial_val_l1, initial_val_vis_mse, val_pixel_count = evaluate(
        centers, sigmas, theta, global_logits, val_fields,
        "best-PSNR initialization validation")
    best_val_l1 = initial_val_l1
    best_val_vis_mse = initial_val_vis_mse
    best_centers = centers.copy()
    best_sigmas = sigmas.copy()
    best_increment_ids = increment_ids.copy()
    best_theta = theta.copy()
    best_global_logits = global_logits.copy()
    best_stage = 0
    history: list[dict] = []
    optimizer_rng = np.random.default_rng(int(seed) + 307)
    abort_reason = "kernel_limit"
    stage = 0
    last_candidate_count = 0

    while len(centers) < target_k:
        stage += 1
        candidates, placement_weights, mean_gain, max_gain = collect_candidates(
            stage, centers, sigmas, theta, global_logits)
        last_candidate_count = len(candidates)
        if max_gain <= 1e-12 or not np.any(placement_weights > 1e-12):
            abort_reason = "no_remaining_best_member_achievable_gain"
            break
        candidate_features = _all_inference_features(candidates)
        normalized_candidates = (
            candidate_features - reference[None, :]) / feature_scales[None, :]
        existing_normalized = (
            (centers - reference[None, :]) / feature_scales[None, :]
            if len(centers) else None)
        add_count = min(max(1, int(increment_size)), target_k - len(centers))
        center_rows = _weighted_separated_center_indices(
            normalized_candidates,
            placement_weights,
            add_count,
            min_separation=DEFAULT_WITHIN_STAGE_MIN_SEPARATION,
            existing_centers=existing_normalized,
            existing_min_separation=CROSS_STAGE_MIN_SEPARATION,
            seed=int(seed) + 223 + stage,
        )
        if not len(center_rows):
            abort_reason = "global_center_separation_limited"
            break
        new_centers = candidate_features[center_rows].astype(np.float64)
        new_normalized = normalized_candidates[center_rows]
        combined_normalized = (
            new_normalized if existing_normalized is None
            else np.concatenate((existing_normalized, new_normalized), axis=0))
        new_sigmas = _local_capped_rbf_sigmas(combined_normalized)[-len(new_centers):]
        used = len(centers)
        centers = np.concatenate((centers, new_centers), axis=0)
        sigmas = np.concatenate((sigmas, new_sigmas))
        increment_ids = np.concatenate((
            increment_ids,
            np.full(len(new_centers), stage, np.int32),
        ))
        theta = np.concatenate((
            theta,
            np.zeros((len(new_centers), len(labels)), np.float64),
        ), axis=0)

        # Reset Adam moments at each stage while warm-starting the actual
        # parameters. This makes every stage a genuine joint refit.
        theta_first = np.zeros_like(theta)
        theta_second = np.zeros_like(theta)
        global_first = np.zeros_like(global_logits)
        global_second = np.zeros_like(global_logits)
        adam_step = 0
        stage_l1_total = 0.0
        stage_pixel_count = 0
        batches = 0
        rows = effective_rows(len(centers))
        for epoch in range(1, max(1, int(epochs)) + 1):
            for position, (field_index, raw0, target0) in enumerate(
                    field_factory(train_fields.tolist()), 1):
                raw = np.asarray(raw0, np.float32)
                target = np.asarray(target0, np.float32)
                pixels = raw.reshape(
                    len(labels), -1, len(names)).transpose(1, 0, 2)
                targets = target.reshape(-1, len(names))
                order = optimizer_rng.permutation(len(pixels))
                for start in range(0, len(order), rows):
                    chosen = order[start:start + rows]
                    batch = pixels[chosen]
                    target_batch = targets[chosen]
                    phi = basis_rows(batch, centers, sigmas)
                    members_asinh = np.arcsinh(
                        np.asarray(batch, np.float64)
                        / band_scales[None, None, :])
                    member_weights = _softmax_rows(
                        global_logits[None, :] + phi @ theta)
                    prediction_asinh = np.einsum(
                        "nm,nmc->nc", member_weights, members_asinh,
                        optimize=True)
                    target_asinh = np.arcsinh(
                        np.asarray(target_batch, np.float64)
                        / band_scales[None, :])
                    error = prediction_asinh - target_asinh
                    smooth = np.sqrt(
                        error * error + ASINH_L1_SMOOTH_DELTA ** 2)
                    prediction_gradient = (
                        error / smooth) / (len(batch) * len(names))
                    weight_gradient = np.einsum(
                        "nc,nmc->nm", prediction_gradient,
                        members_asinh, optimize=True)
                    logit_gradient = member_weights * (
                        weight_gradient - np.sum(
                            weight_gradient * member_weights,
                            axis=1, keepdims=True))
                    theta_gradient = (
                        phi.T @ logit_gradient + float(ridge) * theta)
                    global_gradient = np.sum(logit_gradient, axis=0)
                    adam_step += 1
                    theta_first = 0.9 * theta_first + 0.1 * theta_gradient
                    theta_second = (
                        0.999 * theta_second + 0.001 * theta_gradient ** 2)
                    global_first = 0.9 * global_first + 0.1 * global_gradient
                    global_second = (
                        0.999 * global_second + 0.001 * global_gradient ** 2)
                    correction1 = 1.0 - 0.9 ** adam_step
                    correction2 = 1.0 - 0.999 ** adam_step
                    theta -= float(learning_rate) * (theta_first / correction1) / (
                        np.sqrt(theta_second / correction2) + 1e-8)
                    global_logits -= float(learning_rate) * (
                        global_first / correction1) / (
                            np.sqrt(global_second / correction2) + 1e-8)
                    global_logits -= np.mean(global_logits)
                    stage_l1_total += float(np.sum(np.abs(error)))
                    stage_pixel_count += len(batch)
                    batches += 1
                tick(position, len(train_fields),
                     f"stage {stage} refit {epoch}/{epochs} field {field_index}")

        val_l1, val_vis_mse, _ = evaluate(
            centers, sigmas, theta, global_logits, val_fields,
            f"stage {stage} validation")
        selected = bool(
            np.isfinite(val_l1) and val_l1 < best_val_l1 - 1e-9)
        if selected:
            best_val_l1 = val_l1
            best_val_vis_mse = val_vis_mse
            best_centers = centers.copy()
            best_sigmas = sigmas.copy()
            best_increment_ids = increment_ids.copy()
            best_theta = theta.copy()
            best_global_logits = global_logits.copy()
            best_stage = stage
        history.append({
            "stage": int(stage),
            "epoch": int(max(1, int(epochs))),
            "n_centers": int(len(centers)),
            "added_centers": int(len(new_centers)),
            "train_pixels": int(stage_pixel_count),
            "train_l1": float(
                stage_l1_total / max(1, stage_pixel_count * len(names))),
            "candidate_pixels": int(len(candidates)),
            "candidate_mean_achievable_gain": float(mean_gain),
            "candidate_max_achievable_gain": float(max_gain),
            "val_pixels": int(val_pixel_count),
            "val_l1": float(val_l1),
            "val_vis_asinh_mse": float(val_vis_mse),
            "val_vis_asinh_psnr": _stretched_psnr_from_mse(val_vis_mse),
            "selected_by_validation": selected,
            "optimizer_iterations": int(batches),
            "optimizer_progress": bool(batches > 0),
            "new_block_norm": float(np.linalg.norm(theta[used:])),
        })

    selected_k = len(best_centers)

    return RawIncrementalMinMeanMaxRBFCombiner(
        member_labels=labels,
        n_kernels=int(selected_k),
        coefficients=best_theta.astype(np.float32),
        centers=best_centers.astype(np.float32),
        scales=feature_scales.astype(np.float32),
        sigmas=best_sigmas.astype(np.float32),
        increment_ids=best_increment_ids,
        reference_features=reference.astype(np.float32),
        output_floors=output_floors.astype(np.float32),
        band_names=names,
        baseline_member_index=None,
        global_logits=best_global_logits.astype(np.float32),
        val_l1=float(best_val_l1),
        fit_meta={
            "shared_across_bands": True,
            "features": "all member asinh inferences",
            "feature_schema": _RAW_FEATURE_NAMES[0],
            "feature_dimension": int(len(labels) * len(names)),
            "input_members": int(len(labels)),
            "input_bands": int(len(names)),
            "initial_prediction": "near_one_hot_best_PSNR_member",
            "initial_best_member_index": int(best_member_index),
            "initial_best_member_label": labels[best_member_index],
            "initial_member_probabilities": initial_probabilities.tolist(),
            "initialization_member_psnr": initialization_psnr.tolist(),
            "initialization_psnr_source": psnr_source,
            "baseline_member_index": None,
            "baseline_selection_metric": "PSNR_initialization_only",
            "input_space": "per_band_asinh",
            "output": "shared_weight_convex_member_average_in_asinh_space",
            "output_space": "per_band_asinh_then_electrons",
            "output_activation": "member_softmax_then_asinh_average_then_sinh",
            "signed_sky_subtracted_output": True,
            "loss": "minibatch_smooth_asinh_l1_plus_ridge",
            "coefficient_parameterization": "global_plus_rbf_member_logits",
            "ridge_normalized_coefficient_l2": float(ridge),
            "optimizer": "streaming_minibatch_adam",
            "optimizer_acceptance": "lowest whole-field holdout asinh-L1",
            "validation_prefix_metric": "asinh_L1_primary",
            "training_mode": "all_validation_pixels_minibatch",
            "field_split": "deterministic_disjoint_fields",
            "training_field_indices": train_fields.tolist(),
            "validation_field_indices": val_fields.tolist(),
            "training_fields": int(len(train_fields)),
            "validation_fields": int(len(val_fields)),
            "training_pixels_per_epoch": int(train_pixel_count),
            "validation_pixels_per_epoch": int(val_pixel_count),
            "requested_batch_rows": int(batch_rows),
            "batch_rows": int(effective_rows(max(1, selected_k))),
            "basis_element_budget": int(basis_element_budget),
            "requested_epochs": int(max(1, int(epochs))),
            "selected_epoch": int(max(1, int(epochs)) if best_stage else 0),
            "selected_stage": int(best_stage),
            "requested_kernels": int(target_k),
            "selected_kernels": int(selected_k),
            "learned_parameter_count": int(
                len(labels) + selected_k * len(labels)),
            "stored_parameter_count": int(
                selected_k * (len(labels) * len(names) + len(labels))),
            "center_candidates": int(last_candidate_count),
            "center_candidate_rule": (
                "rescanned after every stage; field-balanced hard attainable-"
                "gain plus uniform coverage; P(center) proportional to linear "
                "best-member achievable asinh-L1 gain times nearest distance^2"),
            "achievable_gain_floor": (
                "best single member equal-band asinh-L1 with one member shared "
                "across all four bands"),
            "increment_size": int(max(1, int(increment_size))),
            "within_increment_min_separation_normalized": float(
                DEFAULT_WITHIN_STAGE_MIN_SEPARATION),
            "cross_increment_min_separation_normalized": float(
                CROSS_STAGE_MIN_SEPARATION),
            "global_center_min_separation_normalized": float(
                CROSS_STAGE_MIN_SEPARATION),
            "normalization_rows": int(len(normalization_pixels)),
            "kernel_width_rule": (
                "0.75x local nearest-center distance capped in normalized space"),
            "max_kernel_sigma_normalized": float(MAX_MINIBATCH_RBF_SIGMA),
            "asinh_scales_e": band_scales.tolist(),
            "asinh_l1_smooth_delta": float(ASINH_L1_SMOOTH_DELTA),
            "initial_val_l1": float(initial_val_l1),
            "initial_val_vis_asinh_mse": float(initial_val_vis_mse),
            "initial_val_vis_asinh_psnr": _stretched_psnr_from_mse(
                initial_val_vis_mse),
            "selected_val_vis_asinh_mse": float(best_val_vis_mse),
            "selected_val_vis_asinh_psnr": _stretched_psnr_from_mse(
                best_val_vis_mse),
            "center_history": history,
            "center_abort_reason": abort_reason,
        },
    )


def fit_combiner(
    buffer,
    member_labels,
    *,
    band_names=BAND_NAMES,
    n_kernels: int = DEFAULT_N_KERNELS,
    seed: int = 0,
    holdout: float = 0.1,
    model_kind: str = RAW_INCREMENTAL_MINMEANMAX_RBF_KIND,
    member_validation_psnr: np.ndarray | None = None,
    member_validation_metrics: dict[str, np.ndarray] | None = None,
    **_unused,
) -> RawIncrementalMinMeanMaxRBFCombiner:
    """Fit the asinh-space combiner from aligned electron-domain inputs."""
    normalize_model_kind(model_kind)
    X, y = buffer
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.float32)
    labels = [str(label) for label in member_labels]
    names = tuple(band_names)
    if X.ndim != 3 or y.ndim != 2 or len(X) != len(y):
        raise ValueError(f"expected (N,M,C)/(N,C), got {X.shape}/{y.shape}")
    if not len(X) or X.shape[1] != len(labels) or X.shape[2] != len(names):
        raise ValueError("combiner fit buffer does not match members/bands")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(X))
    if len(X) == 1:
        train_idx = val_idx = order
    else:
        n_val = min(len(X) - 1, max(1, int(round(len(X) * float(holdout)))))
        val_idx, train_idx = order[:n_val], order[n_val:]
    return _fit_raw_incremental_minmeanmax_rbf(
        X[train_idx], y[train_idx], X[val_idx], y[val_idx], labels, names,
        n_kernels=int(n_kernels), seed=int(seed),
        within_increment_separation=DEFAULT_WITHIN_STAGE_MIN_SEPARATION,
        member_validation_psnr=member_validation_psnr,
        member_validation_metrics=member_validation_metrics,
    )


fit_shared_combiner = fit_combiner


def build_fit_buffers_from_fields(field_iter, band_names=BAND_NAMES, **kwargs):
    accumulator = FitBufferAccumulator(band_names, **kwargs)
    for predictions, target in field_iter:
        accumulator.add(predictions, target)
    return accumulator.buffer()


def combiner_region_ids(comb: RawIncrementalMinMeanMaxRBFCombiner,
                        pixels: np.ndarray, band: str | None = None) -> np.ndarray:
    """Return the nearest active-kernel id for aligned four-band pixels."""
    del band
    features = comb.features_from_electrons(np.asarray(pixels, np.float64))
    if not comb.n_kernels:
        return np.full(len(features), -1, np.int32)
    scales = np.maximum(np.asarray(comb.scales, np.float64), 1e-8)
    distance = ((features[:, None, :] - np.asarray(comb.centers)[None, :, :])
                / scales[None, None, :])
    return np.argmin(np.sum(distance * distance, axis=2), axis=1).astype(np.int32)


def _combiner_dir(base_dir: str, artifact_dir: str | None = None) -> str:
    return os.path.join(base_dir, artifact_dir or combiner_model_spec().artifact_dir)


def combiner_artifact_fingerprint(base_dir: str, artifact_dir: str) -> str | None:
    digest = hashlib.sha256()
    for filename in ("combiner.json", "combiner.npz"):
        path = os.path.join(_combiner_dir(base_dir, artifact_dir), filename)
        try:
            with open(path, "rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        except OSError:
            return None
        digest.update(filename.encode())
    return digest.hexdigest()


def save_combiner(comb: RawIncrementalMinMeanMaxRBFCombiner, base_dir: str, *,
                  artifact_dir: str | None = None) -> None:
    if not isinstance(comb, RawIncrementalMinMeanMaxRBFCombiner):
        raise TypeError("only the all-inference RBF combiner is supported")
    directory = _combiner_dir(base_dir, artifact_dir)
    os.makedirs(directory, exist_ok=True)
    np.savez_compressed(
        os.path.join(directory, "combiner.npz"),
        coefficients=np.asarray(comb.coefficients, np.float32),
        global_logits=np.asarray(
            np.zeros(len(comb.member_labels), np.float32)
            if comb.global_logits is None else comb.global_logits,
            np.float32),
        centers=np.asarray(comb.centers, np.float32),
        scales=np.asarray(comb.scales, np.float32),
        sigmas=np.asarray(comb.sigmas, np.float32),
        increment_ids=np.asarray(comb.increment_ids, np.int32),
        reference_features=np.asarray(comb.reference_features, np.float32),
        output_floors=np.asarray(comb.output_floors, np.float32),
    )
    manifest = {
        "schema": 9,
        "kind": RAW_INCREMENTAL_MINMEANMAX_RBF_KIND,
        "feature_names": list(_RAW_FEATURE_NAMES),
        "member_labels": list(comb.member_labels),
        "n_kernels": int(comb.n_kernels),
        "band_names": list(comb.band_names),
        "level_range": list(comb.level_range),
        "records_fp": comb.records_fp,
        "starfull": bool(comb.starfull),
        "val_l1": comb.val_l1,
        "baseline_member_index": comb.baseline_member_index,
        "fit_meta": comb.fit_meta,
    }
    with open(os.path.join(directory, "combiner.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)


def load_combiner(base_dir: str, *, member_labels: list[str] | None = None,
                  artifact_dir: str | None = None
                  ) -> RawIncrementalMinMeanMaxRBFCombiner | None:
    """Load the sole supported artifact; all former formats are rejected."""
    directory = _combiner_dir(base_dir, artifact_dir)
    manifest_path = os.path.join(directory, "combiner.json")
    arrays_path = os.path.join(directory, "combiner.npz")
    if not (os.path.isfile(manifest_path) and os.path.isfile(arrays_path)):
        return None
    try:
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        if manifest.get("kind") != RAW_INCREMENTAL_MINMEANMAX_RBF_KIND:
            return None
        if manifest.get("feature_names") != list(_RAW_FEATURE_NAMES):
            return None
        labels = [str(value) for value in manifest["member_labels"]]
        if member_labels is not None and labels != [str(value) for value in member_labels]:
            return None
        arrays = np.load(arrays_path)
        return RawIncrementalMinMeanMaxRBFCombiner(
            member_labels=labels,
            n_kernels=int(manifest.get("n_kernels", 0)),
            coefficients=arrays["coefficients"],
            global_logits=arrays["global_logits"],
            centers=arrays["centers"],
            scales=arrays["scales"],
            sigmas=arrays["sigmas"],
            increment_ids=arrays["increment_ids"],
            reference_features=arrays["reference_features"],
            output_floors=arrays["output_floors"],
            baseline_member_index=manifest.get("baseline_member_index"),
            band_names=tuple(manifest.get("band_names", BAND_NAMES)),
            level_range=tuple(manifest.get("level_range", GATE_LEVEL_RANGE)),
            records_fp=manifest.get("records_fp"),
            starfull=bool(manifest.get("starfull", True)),
            val_l1=manifest.get("val_l1"),
            fit_meta=manifest.get("fit_meta", {}),
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
