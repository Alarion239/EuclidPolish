"""Incremental raw-electron min/mean/max ensemble combiner.

The combiner starts at the exact ensemble mean and adds blocks of localized
RBF residual corrections in the twelve raw-electron min/mean/max coordinates.
Kernels within one stage repel each other. Kernels from different stages have
no minimum-distance constraint, allowing later stages to refine the same part
of feature space.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from euclid_polish.config import Config

_BAND_SCALE = {name: float(Config.get_band(name).asinh_stretch_scale_e)
               for name in Config.HR_TARGET_BAND_NAMES}
GATE_LEVEL_RANGE = (-1.0, 13.0)
DEFAULT_N_KERNELS = 128
DEFAULT_WITHIN_STAGE_MIN_SEPARATION = 0.35
CROSS_STAGE_MIN_SEPARATION = 0.0
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


_RAW_FEATURE_NAMES = tuple(
    f"{band}_{stat}_e" for band in BAND_NAMES for stat in ("min", "mean", "max")
) + (
    "raw_electrons_mean_residual_boosting_v1",
    "within_stage_separation_0.35_cross_stage_separation_0_v2",
)
COMBINER_MODELS = {
    RAW_INCREMENTAL_MINMEANMAX_RBF_KIND: CombinerModelSpec(
        RAW_INCREMENTAL_MINMEANMAX_RBF_KIND,
        "incremental raw min/mean/max RBF",
        "raw_incremental_minmeanmax_rbf_combiner",
        "raw_incremental_minmeanmax_rbf_combiner_evals.json",
        "comb_raw_incremental_minmeanmax_rbf",
        _RAW_FEATURE_NAMES,
    ),
}
ACTIVE_COMBINER_KINDS = (RAW_INCREMENTAL_MINMEANMAX_RBF_KIND,)


def _band_scale(name: str) -> float:
    return _BAND_SCALE[name]


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


def _raw_minmeanmax_features(member_pixels: np.ndarray) -> np.ndarray:
    values = np.asarray(member_pixels, np.float64)
    if values.ndim != 3:
        raise ValueError(f"expected (N,M,C) member pixels, got {values.shape}")
    return np.stack((np.min(values, axis=1), np.mean(values, axis=1),
                     np.max(values, axis=1)), axis=2).reshape(len(values), -1)

class FitBufferAccumulator:
    """Bounded, aligned four-band pixels for fitting the incremental combiner.

    Sampling is stratified by the brightest target band, but a selected pixel
    always retains every member and every band.  This alignment is what makes
    one shared raw-electron model possible.
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

    def add(self, preds: np.ndarray, hr: np.ndarray) -> None:
        if self._n >= self.max_rows:
            return
        raw = np.asarray(preds, np.float32)
        target = np.asarray(hr, np.float32)
        if raw.ndim != 4 or target.ndim != 3:
            raise ValueError(f"expected (M,H,W,C)/(H,W,C), got {raw.shape}/{target.shape}")
        m, _, _, c = raw.shape
        if c != len(self.band_names) or target.shape[-1] != c:
            raise ValueError(f"expected bands {self.band_names}, got {c}")
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


SharedFitBufferAccumulator = FitBufferAccumulator

def _weighted_kmeans(rows: np.ndarray, sample_weight: np.ndarray,
                     n_clusters: int, *, seed: int, max_iter: int = 40,
                     tol: float = 1e-5,
                     existing_centers: np.ndarray | None = None,
                     min_separation: float = 0.0) -> np.ndarray:
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
    # Request extra candidates when a hard separation filter is active. This
    # lets K-means++ find a complete batch without accepting near-duplicates.
    k = min(len(points), requested_k * 2 if min_separation > 0 else requested_k)
    rng = np.random.default_rng(seed)
    centers = np.empty((k, points.shape[1]), np.float64)
    chosen: list[int] = []
    if len(anchors):
        anchor_distance = np.min(np.sum(
            (points[:, None, :] - anchors[None, :, :]) ** 2, axis=2), axis=1)
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
        distance = np.sum(
            (points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
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
    if min_separation <= 0:
        return centers[:requested_k]
    distance = np.sum(
        (points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(distance, axis=1)
    mass = np.asarray(
        [weights[labels == ci].sum() for ci in range(len(centers))])
    accepted: list[np.ndarray] = []
    floor2 = float(min_separation) ** 2
    for ci in np.argsort(mass)[::-1]:
        candidate = centers[int(ci)]
        comparison = (anchors if not accepted else
                      np.concatenate((anchors, np.asarray(accepted)), axis=0))
        if (not len(comparison) or float(np.min(np.sum(
                (comparison - candidate[None, :]) ** 2, axis=1))) >= floor2):
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

@dataclass
class RawIncrementalMinMeanMaxRBFCombiner:
    """Raw min/mean/max RBF boosting around an exact ensemble-mean default.

    Fixed centers describe residual regions in twelve raw-electron coordinates.
    Each kernel contributes one signed correction per output band; the final
    prediction is ``relu(member_mean + rbf @ correction)``.  With no nearby
    kernel the correction tends smoothly to zero, so inference returns the
    ordinary ensemble mean rather than a learned global bias.
    """

    member_labels: list[str]
    n_kernels: int
    coefficients: np.ndarray       # (K, C), signed raw-electron corrections
    centers: np.ndarray            # (K, 3*C), raw min/mean/max coordinates
    scales: np.ndarray             # (3*C,), distance normalization only
    sigmas: np.ndarray             # (K,), fixed per-increment widths
    increment_ids: np.ndarray      # (K,), allocation batch for diagnostics
    reference_features: np.ndarray # (3*C,), validation median
    output_floors: np.ndarray      # (C,), PCA suppression denominator floors
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
        # These label the four PCA suppression surfaces.  This model has no
        # member weights; its source membership is retained separately.
        return list(self.band_names)

    def features_from_electrons(self, pixels: np.ndarray) -> np.ndarray:
        raw = np.asarray(pixels, np.float64)
        if raw.ndim != 3 or raw.shape[1] != len(self.member_labels):
            raise ValueError(
                f"expected (N,{len(self.member_labels)},C) member pixels, got {raw.shape}")
        if raw.shape[2] != len(self.band_names):
            raise ValueError(f"expected {len(self.band_names)} bands, got {raw.shape[2]}")
        return _raw_minmeanmax_features(raw)

    def _basis_from_features(self, features: np.ndarray) -> np.ndarray:
        z = np.asarray(features, np.float64)
        if not self.n_kernels:
            return np.zeros((len(z), 0), np.float64)
        scales = np.maximum(np.asarray(self.scales, np.float64), 1e-8)
        normalized = z / scales[None, :]
        centers = np.asarray(self.centers, np.float64) / scales[None, :]
        distance2 = (np.sum(normalized * normalized, axis=1)[:, None]
                     + np.sum(centers * centers, axis=1)[None, :]
                     - 2.0 * normalized @ centers.T)
        np.maximum(distance2, 0.0, out=distance2)
        sigma = np.maximum(np.asarray(self.sigmas, np.float64), 1e-6)
        return np.exp(-0.5 * distance2
                      / (sigma[None, :] * sigma[None, :]))

    def correction_from_electrons(self, pixels: np.ndarray, *,
                                  chunk_rows: int = 4096) -> np.ndarray:
        features = self.features_from_electrons(pixels)
        coefficients = np.asarray(self.coefficients, np.float64)
        out = np.empty((len(features), len(self.band_names)), np.float64)
        chunk = max(1, int(chunk_rows))
        for start in range(0, len(features), chunk):
            stop = min(len(features), start + chunk)
            out[start:stop] = (self._basis_from_features(features[start:stop])
                               @ coefficients)
        return out

    def predict_pixels(self, pixels: np.ndarray) -> np.ndarray:
        raw = np.asarray(pixels, np.float64)
        out = np.mean(raw, axis=1) + self.correction_from_electrons(raw)
        return np.maximum(out, 0.0)

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
        """PCA surface of the fraction removed from the ensemble mean.

        The existing WebUI surface contract calls the final axis ``weights``.
        Here it is explicitly a diagnostic suppression fraction in [0, 1], one
        surface per output band, not a member-routing weight.
        """
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
        correction = basis @ np.asarray(self.coefficients, np.float64)
        mean_idx = np.arange(len(self.band_names)) * 3 + 1
        mean = path[:, mean_idx]
        pred = np.maximum(mean + correction, 0.0)
        denominator = np.maximum(
            np.abs(mean), np.asarray(self.output_floors, np.float64)[None, :])
        suppression = np.clip((mean - pred) / denominator, 0.0, 1.0)
        feature_names = [
            f"{band}:{stat} raw e-" for band in self.band_names
            for stat in ("min", "mean", "max")
        ]
        total_variance = max(float(np.trace(covariance)), 1e-12)
        return {
            "available": True,
            "n_pixels": int(len(features)),
            "feature_space": "scale-normalized raw-electron min/mean/max",
            "conditioning_note": (
                "The surface shows the fraction suppressed from the ensemble "
                "mean along PC1 and PC2; all remaining PCs stay at their "
                "validation mean."),
            "pc1": pc1, "pc2": pc2,
            "center_pc1": center_scores[:, 0],
            "center_pc2": center_scores[:, 1],
            "weights": suppression.reshape(len(pc2), len(pc1), -1),
            "explained_variance_ratio": eigenvalues / total_variance,
            "feature_names": feature_names,
            "loadings": components.copy(),
            "z_label": "mean suppression fraction [0-1]",
            "surface_labels": list(self.band_names),
        }

def _fit_raw_incremental_minmeanmax_rbf(
    Xtr: np.ndarray, ytr: np.ndarray, Xval: np.ndarray, yval: np.ndarray,
    labels: list[str], names: tuple[str, ...], *, n_kernels: int,
    seed: int, residual_abort: float = 1e-3,
    increment_size: int = 16, within_increment_separation: float = 0.35,
    ridge: float = 1e-5, max_optimizer_iterations: int = 500,
) -> RawIncrementalMinMeanMaxRBFCombiner:
    """Fit residual RBF blocks around an exact ensemble-mean baseline.

    Center exclusion deliberately applies only inside the newly allocated
    increment.  Previous centers are not supplied to weighted K-means++, so a
    later residual block may refine an already represented part of feature
    space.  After every increment, all accumulated coefficients are jointly
    re-optimized with deterministic full-batch L-BFGS.  The previous optimum
    plus zero-valued new coefficients is only the warm start; no old block is
    frozen.
    """
    rng = np.random.default_rng(seed)
    train_features = _raw_minmeanmax_features(Xtr)
    val_features = _raw_minmeanmax_features(Xval)
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

    train_mean = np.mean(np.asarray(Xtr, np.float64), axis=1)
    val_mean = np.mean(np.asarray(Xval, np.float64), axis=1)
    train_linear = train_mean.copy()
    val_linear = val_mean.copy()
    train_pred = np.maximum(train_linear, 0.0)
    val_pred = np.maximum(val_linear, 0.0)
    baseline_val_l1 = float(np.mean(np.abs(val_pred - yval)))
    output_floors = []
    for ci in range(len(names)):
        positive = np.abs(np.asarray(ytr[:, ci], np.float64))
        positive = positive[positive > 1e-6]
        output_floors.append(max(
            0.1, float(np.quantile(positive, 0.10)) if len(positive) else 0.1))
    output_floors = np.asarray(output_floors, np.float64)

    all_centers: list[np.ndarray] = []
    all_sigmas: list[np.ndarray] = []
    all_increment_ids: list[np.ndarray] = []
    train_basis = np.empty((len(Xtr), 0), np.float32)
    val_basis = np.empty((len(Xval), 0), np.float32)
    coefficients = np.empty((0, len(names)), np.float64)
    history: list[dict[str, float | int | str | bool]] = []
    best_val_l1 = baseline_val_l1
    best_k = 0
    best_centers = np.empty((0, train_features.shape[1]), np.float64)
    best_sigmas = np.empty((0,), np.float64)
    best_coefficients = np.empty((0, len(names)), np.float64)
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
        channels = len(names)
        deltas = np.maximum(0.05, 0.05 * output_floors)
        targets = np.asarray(ytr, np.float64)
        normalization = float(max(1, len(phi)))
        fitted = np.empty((k, channels), np.float64)
        converged: list[bool] = []
        iterations: list[int] = []
        gradient_infinity: list[float] = []
        final_losses: list[float] = []
        messages: list[str] = []
        # The four-band objective is separable once centers are fixed. Solving
        # each K-vector independently reaches the same joint optimum, while
        # avoiding a poorly conditioned interleaved 4K L-BFGS history.
        for ci in range(channels):
            def objective(current: np.ndarray, *, channel=ci
                          ) -> tuple[float, np.ndarray]:
                error = (train_mean[:, channel] + phi @ current
                         - targets[:, channel])
                smooth = np.sqrt(
                    error * error + deltas[channel] * deltas[channel])
                data_loss = float(
                    np.sum(smooth - deltas[channel]) / normalization)
                regularization = (0.5 * float(ridge)
                                  * float(np.sum(current * current)))
                gradient = phi.T @ ((error / smooth) / normalization)
                gradient += float(ridge) * current
                return data_loss + regularization, gradient

            x = np.asarray(initial[:, ci], np.float64)
            total_iterations = 0
            result = None
            for _attempt in range(2):
                result = minimize(
                    objective, x, method="L-BFGS-B", jac=True,
                    options={"maxiter": max(1, int(max_optimizer_iterations)),
                             "maxls": 40, "ftol": 1e-9, "gtol": 1e-5})
                total_iterations += int(result.nit)
                x = np.asarray(result.x, np.float64)
                if bool(result.success) or int(result.status) != 1:
                    break
            final_loss, final_gradient = objective(x)
            fitted[:, ci] = x
            converged.append(bool(result.success))
            iterations.append(total_iterations)
            gradient_infinity.append(float(np.max(np.abs(final_gradient))))
            final_losses.append(float(final_loss))
            messages.append(str(result.message))
        return (fitted, bool(all(converged)), iterations,
                float(max(gradient_infinity, default=0.0)),
                float(np.mean(final_losses)), " | ".join(messages))

    while sum(len(batch) for batch in all_centers) < target_k:
        train_residual = np.asarray(ytr, np.float64) - train_pred
        row_residual = np.mean(np.abs(train_residual), axis=1)
        max_residual = float(np.max(row_residual))
        if max_residual <= float(residual_abort):
            abort_reason = "all_train_residuals_below_threshold"
            break
        stage += 1
        used = sum(len(batch) for batch in all_centers)
        add_count = min(max(1, int(increment_size)), target_k - used)
        center_weight = np.maximum(row_residual[geometry_idx], 0.0)
        new_norm = _weighted_kmeans(
            geometry, center_weight, add_count, seed=seed + stage,
            # Intentionally no existing_centers: only this increment repels
            # itself; later increments may land beside earlier kernels.
            existing_centers=None,
            min_separation=float(within_increment_separation))
        if not len(new_norm):
            abort_reason = "within_increment_separation_exhausted"
            break
        new_sigma = _rbf_sigma_from_centers(new_norm)
        new_sigmas = np.full(len(new_norm), new_sigma, np.float64)
        new_centers = new_norm * scales[None, :] + reference[None, :]
        phi_train = basis(train_features, new_centers, new_sigmas)
        phi_val = basis(val_features, new_centers, new_sigmas)
        train_basis = np.concatenate((train_basis, phi_train), axis=1)
        val_basis = np.concatenate((val_basis, phi_val), axis=1)
        coefficients = np.concatenate(
            (coefficients, np.zeros((len(new_centers), len(names)), np.float64)),
            axis=0)
        (coefficients, optimizer_converged, optimizer_iterations_by_band,
         optimizer_gradient_inf, optimizer_loss,
         optimizer_message) = jointly_optimize(train_basis, coefficients)
        train_linear = train_mean + train_basis @ coefficients
        train_pred = np.maximum(train_linear, 0.0)
        val_linear = val_mean + val_basis @ coefficients
        val_pred = np.maximum(val_linear, 0.0)

        all_centers.append(new_centers)
        all_sigmas.append(new_sigmas)
        all_increment_ids.append(np.full(len(new_norm), stage, np.int32))
        total_k = sum(len(batch) for batch in all_centers)
        train_l1 = float(np.mean(np.abs(train_pred - ytr)))
        val_l1 = float(np.mean(np.abs(val_pred - yval)))
        selected = bool(optimizer_converged and val_l1 < best_val_l1 - 1e-9)
        if selected:
            best_val_l1 = val_l1
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
                np.abs(np.asarray(ytr, np.float64) - train_pred), axis=1))),
            "val_l1": val_l1,
            "val_improvement_from_mean": baseline_val_l1 - val_l1,
            "selected_by_validation": selected,
            "optimizer_converged": bool(optimizer_converged),
            "optimizer_iterations": int(sum(optimizer_iterations_by_band)),
            "optimizer_iterations_by_band": [
                int(value) for value in optimizer_iterations_by_band],
            "optimizer_gradient_inf": float(optimizer_gradient_inf),
            "optimizer_objective": float(optimizer_loss),
            "optimizer_message": optimizer_message,
        })
        if not optimizer_converged:
            abort_reason = "joint_optimizer_did_not_converge"
            break

    # Validation chooses a prefix of complete increments, including the exact
    # zero-correction baseline when every learned block generalizes poorly.
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
        val_l1=float(best_val_l1),
        fit_meta={
            "shared_across_bands": True,
            "features": "raw electron min/mean/max per band",
            "initial_prediction": "ensemble_mean",
            "output": "four_signed_raw_electron_residuals",
            "output_activation": "relu_after_mean_plus_correction",
            "loss": "smooth_raw_electron_l1_plus_ridge",
            "optimizer": "joint_full_batch_lbfgs_after_every_increment",
            "optimizer_warm_start": "previous_joint_optimum_plus_zero_new_block",
            "optimizer_max_iterations_per_call": int(max_optimizer_iterations),
            "optimizer_max_continuations": 2,
            "requested_kernels": int(target_k),
            "selected_kernels": int(best_k),
            "learned_parameter_count": int(best_k * len(names)),
            "stored_parameter_count": int(best_k * (train_features.shape[1]
                                                       + len(names))),
            "increment_size": int(increment_size),
            "within_increment_min_separation_normalized": float(
                within_increment_separation),
            "cross_increment_min_separation_normalized": 0.0,
            "kernel_width_rule": "per_increment_median_nearest_distance_x1.25",
            "residual_weight": "current_equal_band_raw_l1",
            "residual_abort_threshold_e": float(residual_abort),
            "baseline_val_l1": float(baseline_val_l1),
            "center_history": history,
            "center_abort_reason": abort_reason,
        })

def fit_combiner(
    buffer,
    member_labels,
    *,
    band_names=BAND_NAMES,
    n_kernels: int = DEFAULT_N_KERNELS,
    seed: int = 0,
    holdout: float = 0.1,
    model_kind: str = RAW_INCREMENTAL_MINMEANMAX_RBF_KIND,
    **_unused,
) -> RawIncrementalMinMeanMaxRBFCombiner:
    """Fit the sole supported combiner on aligned raw-electron pixels."""
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
        raise TypeError("only the incremental raw min/mean/max combiner is supported")
    directory = _combiner_dir(base_dir, artifact_dir)
    os.makedirs(directory, exist_ok=True)
    np.savez_compressed(
        os.path.join(directory, "combiner.npz"),
        coefficients=np.asarray(comb.coefficients, np.float32),
        centers=np.asarray(comb.centers, np.float32),
        scales=np.asarray(comb.scales, np.float32),
        sigmas=np.asarray(comb.sigmas, np.float32),
        increment_ids=np.asarray(comb.increment_ids, np.int32),
        reference_features=np.asarray(comb.reference_features, np.float32),
        output_floors=np.asarray(comb.output_floors, np.float32),
    )
    manifest = {
        "schema": 2,
        "kind": RAW_INCREMENTAL_MINMEANMAX_RBF_KIND,
        "feature_names": list(_RAW_FEATURE_NAMES),
        "member_labels": list(comb.member_labels),
        "n_kernels": int(comb.n_kernels),
        "band_names": list(comb.band_names),
        "level_range": list(comb.level_range),
        "records_fp": comb.records_fp,
        "starfull": bool(comb.starfull),
        "val_l1": comb.val_l1,
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
            centers=arrays["centers"],
            scales=arrays["scales"],
            sigmas=arrays["sigmas"],
            increment_ids=arrays["increment_ids"],
            reference_features=arrays["reference_features"],
            output_floors=arrays["output_floors"],
            band_names=tuple(manifest.get("band_names", BAND_NAMES)),
            level_range=tuple(manifest.get("level_range", GATE_LEVEL_RANGE)),
            records_fp=manifest.get("records_fp"),
            starfull=bool(manifest.get("starfull", True)),
            val_l1=manifest.get("val_l1"),
            fit_meta=manifest.get("fit_meta", {}),
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
