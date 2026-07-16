"""Deep-ensemble **combiner** — tiny per-band convex mixtures.

For the STARFULL regime the reconstruction is normally the naive per-pixel mean
of the ``M`` ensemble members. That discards their complementary strengths
(L1-trained members render faint galaxies cleanly but erase star cores;
L2-trained members keep the point sources but leave a speckle floor). This
module learns a tiny **per-band** model that fuses the members — leaning on the
right member at the right brightness.

Design (RBF brightness gate — replaced the earlier skip-MLP, which couldn't
route faint pixels to the L1 members: near zero it collapsed to a fixed linear
blend that inherited the L2 floor, and its tanh gate modulated strongest at
faint, i.e. the wrong direction):

* **Per-band gated convex mixture.** One model per output band (VIS, Y_E, J_E,
  H_E). At each pixel a **brightness scalar** ``b`` (the max over members, asinh)
  is expanded in ``K`` fixed **RBF kernels**, mapped to per-member logits and
  soft-maxed to weights ``w(b)`` (≥0, sum to 1); the output is the convex
  combination ``Σ wₘ(b)·xₘ``. Localized kernels give an ARBITRARY weight-vs-
  brightness profile — faint bins can put ~all weight on the L1 members (killing
  the L2 floor exactly, since it's a convex combination) while bright bins pick
  the star-reproducing members. The weight curves ARE the model (see
  :meth:`Combiner.effective_weights`). No saturation issue: a convex combo of
  members never has to extrapolate.
* **asinh space.** ``arcsinh(electrons / STRETCH_SCALE_E)`` in; output inverse-
  stretched with ``sinh(·)·scale`` after the same ±clip the inference path uses.
* **L1 loss** (asinh-space ``mean|·|``), fit LOCALLY on the ``validate`` split.
* **Mean/std RBF gate.** The second RBF regime sees the full-stack ``mean``
  and ``std``, so it can react to disagreement without making a sparse,
  redundant three-dimensional geometry.
* **Diagnostic-only member weights.** Every fit retains every member.  Peak
  gate share and distribution-integrated (mean) gate share are measured on the
  brightness-stratified validation rows, independently for every band, and
  persisted for comparison in the UI.
* **Per regime.** Starfull learns against ``hr_``; starless learns against
  ``clean_`` and remains a separately persisted model.

Persistence: ``<dir>/combiner/combiner.npz`` + ``combiner.json``.
:func:`load_combiner` returns ``None`` when the saved member labels no longer
match the active ensemble (stale) or the format is incompatible.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace

import numpy as np

from euclid_polish.config import Config

#: Per-band asinh knee (electrons). All HR bands use STRETCH_SCALE_E today, but
#: honour the per-band config so a future change tracks automatically.
_BAND_SCALE = {name: float(Config.get_band(name).asinh_stretch_scale_e)
               for name in Config.HR_TARGET_BAND_NAMES}

#: Clip on the asinh output before ``sinh`` (matches training/inference.py).
SINH_CLIP = float(getattr(Config, "SINH_STRETCH_CLIP", 20.0))

#: Brightness (asinh) sweep the RBF kernels tile — faint sky (~0) to a bright
#: star core (asinh of a very bright source).
GATE_LEVEL_RANGE = (-1.0, 13.0)
DEFAULT_N_KERNELS = 12
DEFAULT_STATS_RBF_N_KERNELS = 32
DEFAULT_STATS_RBF_MIN_USAGE = 0.0
DEFAULT_STATS_RBF_MIN_CENTER_SEPARATION = 0.35
DEFAULT_SIGMA_SCALE = 1.0

RBF_GATE_KIND = "rbf_gate"
STATS_RBF_GATE_KIND = "stats_rbf_gate"
MINMAX_RBF_GATE_KIND = "minmax_rbf_gate"


@dataclass(frozen=True)
class CombinerModelSpec:
    kind: str
    label: str
    artifact_dir: str
    payload_name: str
    cube_prefix: str
    feature_names: tuple[str, ...] | None = None
    default_kernels: int = DEFAULT_N_KERNELS
    default_min_usage: float = 0.0


COMBINER_MODELS = {
    RBF_GATE_KIND: CombinerModelSpec(RBF_GATE_KIND, "max RBF", "combiner",
                                     "combiner_evals.json", "comb", None),
    STATS_RBF_GATE_KIND: CombinerModelSpec(
        STATS_RBF_GATE_KIND, "mean + std RBF", "stats_rbf_combiner",
        "stats_rbf_combiner_evals.json", "comb_stats_rbf",
        ("mean", "log_std", "repelled_hybrid_reservoirs_v2"),
        DEFAULT_STATS_RBF_N_KERNELS, DEFAULT_STATS_RBF_MIN_USAGE),
    MINMAX_RBF_GATE_KIND: CombinerModelSpec(
        MINMAX_RBF_GATE_KIND, "min + max RBF", "minmax_rbf_combiner",
        "minmax_rbf_combiner_evals.json", "comb_minmax_rbf",
        ("min", "max", "repelled_hybrid_reservoirs_v2"),
        DEFAULT_STATS_RBF_N_KERNELS, DEFAULT_STATS_RBF_MIN_USAGE),
}


def combiner_model_spec(kind: str | None) -> CombinerModelSpec:
    return COMBINER_MODELS[normalize_model_kind(kind)]

BAND_NAMES = tuple(Config.HR_TARGET_BAND_NAMES)


def _band_scale(name: str) -> float:
    return _BAND_SCALE.get(name, float(Config.STRETCH_SCALE_E))


def _brightness(X: np.ndarray) -> np.ndarray:
    """Per-pixel brightness scalar from the member vector (asinh): the max over
    members — 'is there a bright source here'. Sharpens star (all-high) vs floor
    (one member's small speckle) far better than the mean."""
    return np.max(np.asarray(X, np.float64), axis=1)


def _rbf(b: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    z = (np.asarray(b, np.float64)[:, None] - np.asarray(centers)[None, :]) / sigma
    return np.exp(-0.5 * z * z)


def _stats_features(X: np.ndarray, kind: str = STATS_RBF_GATE_KIND) -> np.ndarray:
    """Permutation-invariant full-stack summary: consensus and spread."""
    X = np.asarray(X, np.float64)
    if kind == MINMAX_RBF_GATE_KIND:
        return np.stack((np.min(X, axis=1), np.max(X, axis=1)), axis=1)
    return np.stack((np.mean(X, axis=1), np.std(X, axis=1)), axis=1)


def _stats_geometry_features(features: np.ndarray, std_floor: float,
                             kind: str = STATS_RBF_GATE_KIND) -> np.ndarray:
    """Mean + logarithmic disagreement coordinates used by the stats RBF.

    Member predictions are already in asinh brightness space.  Their spread is
    non-negative; a log coordinate sharply resolves quiet stacks without
    allowing a rare high-disagreement tail to dominate RBF distances.
    """
    raw = np.asarray(features, np.float64)
    if kind == MINMAX_RBF_GATE_KIND:
        return raw
    floor = max(float(std_floor), 1e-8)
    return np.stack((raw[:, 0], np.log(np.maximum(raw[:, 1], 0.0) + floor)),
                    axis=1)


def _std_floor(features: np.ndarray) -> float:
    """Small robust offset that makes ``log(std + floor)`` finite at zero."""
    spread = np.asarray(features, np.float64)[:, 1]
    positive = spread[np.isfinite(spread) & (spread > 1e-6)]
    return max(0.005, float(np.quantile(positive, 0.05)) if len(positive) else 0.005)


def _softmax_masked(logits: np.ndarray, surviving: np.ndarray) -> np.ndarray:
    logits = np.where(surviving[None, :], logits, -1e9)
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def normalize_model_kind(kind: str | None) -> str:
    """Normalize public model names to the persisted model-kind names."""
    key = (kind or RBF_GATE_KIND).strip().lower().replace("-", "_")
    if key in {"rbf", "rbf_gate", "rbf_max", "max", "max_conditioned"}:
        return RBF_GATE_KIND
    if key in {"stats_rbf", "stats_rbf_gate", "max_mean_std_rbf", "max_mean_std"}:
        return STATS_RBF_GATE_KIND
    if key in {"minmax_rbf", "minmax_rbf_gate", "min_max_rbf", "minmax"}:
        return MINMAX_RBF_GATE_KIND
    raise ValueError(f"unknown combiner model kind: {kind!r}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class BandCombiner:
    """One band's RBF brightness gate: ``(N, M) asinh members → (N,) asinh``.

    ``b = max_m x`` → ``φ = rbf(b)`` (K kernels) → ``w = softmax(φ·V + a)`` →
    ``y = Σ wₘ·xₘ`` (convex combination).
    """

    V: np.ndarray                  # (K, M)
    a: np.ndarray                  # (M,)
    centers: np.ndarray            # (K,)
    sigma: float
    surviving: np.ndarray          # (M,) bool

    def weights(self, X: np.ndarray) -> np.ndarray:
        """Per-pixel member weights ``(N, M)`` for a member stack ``(N, M)``.

        The brightness scalar is the max over the SURVIVING members only, so
        pruned members influence neither the gate nor the convex sum (weight 0):
        their SR is genuinely unused and need never be computed. With no pruning
        (all surviving) this is identical to the max over all members."""
        X = np.asarray(X, np.float64)
        surv = np.asarray(self.surviving, bool)
        b = np.max(X[:, surv] if surv.any() else X, axis=1)
        logits = _rbf(b, self.centers, self.sigma) @ self.V + self.a
        return _softmax_masked(logits, self.surviving)

    def forward_asinh(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float64)
        return np.sum(self.weights(X) * X, axis=1)

    def weights_at(self, b_levels: np.ndarray) -> np.ndarray:
        """Gate weights ``(L, M)`` at given brightness levels (asinh)."""
        logits = _rbf(b_levels, self.centers, self.sigma) @ self.V + self.a
        return _softmax_masked(logits, self.surviving)


@dataclass
class StatsRBFBandCombiner:
    """A compact mean/std RBF gate.

    ``x(M) → [mean(x), std(x)] → RBF(32) → softmax(M) → Σ w·x``.
    The std coordinate is ``log(std + std_floor)``. Centers are fixed after a
    small deterministic k-means pass over validation features; only the
    kernel-to-member logits are supervised. This retains RBF smoothness with
    deliberately high resolution for quiet, low-disagreement stacks.
    """

    V: np.ndarray                  # (K, M)
    a: np.ndarray                  # (M,)
    centers: np.ndarray            # (K, 2), mean / log(std + std_floor)
    scales: np.ndarray             # (2,), geometry-coordinate scales
    sigma: float                   # isotropic width in normalized space
    surviving: np.ndarray          # (M,) bool
    std_floor: float               # raw asinh-space std offset for log geometry
    feature_kind: str = STATS_RBF_GATE_KIND

    def _raw_features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float64)
        surv = np.asarray(self.surviving, bool)
        return _stats_features(X[:, surv] if surv.any() else X, self.feature_kind)

    def _basis_from_raw_features(self, z: np.ndarray) -> np.ndarray:
        geometry = _stats_geometry_features(z, self.std_floor, self.feature_kind)
        d = (geometry[:, None, :] - self.centers[None, :, :])
        d = d / np.maximum(np.asarray(self.scales, np.float64)[None, None, :], 1e-6)
        return np.exp(-0.5 * np.sum(d * d, axis=2) / max(float(self.sigma) ** 2, 1e-6))

    def weights(self, X: np.ndarray) -> np.ndarray:
        logits = self._basis_from_raw_features(self._raw_features(X)) @ self.V + self.a
        return _softmax_masked(logits, self.surviving)

    def forward_asinh(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float64)
        return np.sum(self.weights(X) * X, axis=1)

    def weights_at(self, b_levels: np.ndarray) -> np.ndarray:
        """Equal-input, zero-disagreement diagnostic slice for the UI."""
        levels = np.asarray(b_levels, np.float64).reshape(-1)
        z = np.stack((levels, np.zeros_like(levels)), axis=1)
        logits = self._basis_from_raw_features(z) @ self.V + self.a
        return _softmax_masked(logits, self.surviving)

    def weight_surface(self, *, n_mean: int = 31, n_std: int = 31) -> dict:
        """A deliberately padded 2-D feature grid for the rotatable WebUI view."""
        centers = np.asarray(self.centers, np.float64)
        scales = np.maximum(np.asarray(self.scales, np.float64), 1e-6)
        width = max(float(self.sigma), 0.5)
        if self.feature_kind == MINMAX_RBF_GATE_KIND:
            # The min/max features are in raw asinh space and can extend well
            # beyond the old brightness sweep.  Do not clip them to the
            # brightness-gate range: that could cut off fitted centres and
            # make the surface look truncated.  Keep a margin around the
            # actual fitted geometry instead.
            pad = scales * width * 1.25
            finite_centers = np.isfinite(centers)
            lo = np.where(finite_centers, centers, np.inf).min(axis=0) - pad
            hi = np.where(finite_centers, centers, -np.inf).max(axis=0) + pad
            # Degenerate/sparse fitted geometry should still produce a useful
            # axis rather than a zero-width linspace.
            span = np.maximum(pad * 2.0, 1e-3)
            lo = np.where(np.isfinite(lo), lo, -span / 2.0)
            hi = np.where(np.isfinite(hi), hi, span / 2.0)
            hi = np.maximum(hi, lo + span)
            minimum = np.linspace(lo[0], hi[0], int(n_mean))
            maximum = np.linspace(lo[1], hi[1], int(n_std))
            mm, xx = np.meshgrid(minimum, maximum, indexing="xy")
            z = np.stack((mm.reshape(-1), xx.reshape(-1)), axis=1)
            logits = self._basis_from_raw_features(z) @ self.V + self.a
            weights = _softmax_masked(logits, self.surviving)
            return {"mean_asinh": minimum, "std_asinh": maximum,
                    "std_log": maximum,
                    "center_mean_asinh": centers[:, 0],
                    "center_std_asinh": centers[:, 1],
                    "center_std_log": centers[:, 1],
                    "x_label": "min", "y_label": "max", "y_is_log": False,
                    "weights": weights.reshape(len(maximum), len(minimum), -1)}
        # Mean only needs a conventional local RBF margin.  For std, show the
        # full zero-disagreement boundary plus three kernel-widths above the
        # outermost centre: the action is often near the low-std edge, and a
        # symmetric narrow crop made that structure visually misleading.
        pad = scales * width * 1.25
        lo, hi = centers.min(axis=0) - pad, centers.max(axis=0) + pad
        lo[0], hi[0] = max(-1.0, lo[0]), min(13.0, hi[0])
        # Keep zero disagreement visible, then extend high std by three RBF
        # widths in *log-like* geometry before returning to raw std units.
        std_hi_geometry = max(0.0, float(centers[:, 1].max())
                              + float(scales[1] * width * 3.0))
        lo[1] = 0.0
        hi[1] = max(float(np.exp(std_hi_geometry) - self.std_floor), lo[1] + 1e-4)
        mean = np.linspace(lo[0], hi[0], int(n_mean))
        # Uniform rows in the model's log coordinate preserve low-std detail.
        std_log = np.linspace(np.log(self.std_floor), std_hi_geometry, int(n_std))
        std = np.maximum(0.0, np.exp(std_log) - self.std_floor)
        mm, ss = np.meshgrid(mean, std, indexing="xy")
        z = np.stack((mm.reshape(-1), ss.reshape(-1)), axis=1)
        logits = self._basis_from_raw_features(z) @ self.V + self.a
        weights = _softmax_masked(logits, self.surviving)
        return {"mean_asinh": mean, "std_asinh": std,
                "std_log": std_log,
                "center_mean_asinh": centers[:, 0],
                "center_std_asinh": np.maximum(0.0, np.exp(centers[:, 1]) - self.std_floor),
                "center_std_log": centers[:, 1],
                "x_label": "mean", "y_label": "std", "y_is_log": True,
                "weights": weights.reshape(len(std), len(mean), -1)}


@dataclass
class Combiner:
    """The 4-band combiner + metadata to apply/persist/validate it."""

    member_labels: list[str]
    n_kernels: int
    sigma_scale: float
    min_usage: float
    bands: dict[str, BandCombiner | StatsRBFBandCombiner]
    band_names: tuple[str, ...] = BAND_NAMES
    level_range: tuple[float, float] = GATE_LEVEL_RANGE
    records_fp: str | None = None
    starfull: bool = True
    val_l1: float | None = None
    kind: str = RBF_GATE_KIND
    fit_meta: dict = field(default_factory=dict)
    member_importance: dict[str, list[float]] = field(default_factory=dict)
    member_weight_peaks: dict[str, list[float]] = field(default_factory=dict)
    member_weight_integrals: dict[str, list[float]] = field(default_factory=dict)
    max_prune_regret: float = 0.0
    min_peak_weight: float = 0.0
    member_ablation: dict = field(default_factory=dict)

    # -- inference -- #
    def apply_field(self, preds: np.ndarray,
                    band_names: tuple[str, ...] | None = None) -> np.ndarray:
        """Combine a member stack ``(M,H,W,C)`` (electrons) → ``(H,W,C)``."""
        preds = np.asarray(preds, np.float32)
        if preds.ndim != 4:
            raise ValueError(f"expected (M,H,W,C) member stack, got {preds.shape}")
        m, h, w, c = preds.shape
        names = tuple(band_names) if band_names is not None else self.band_names
        out = np.empty((h, w, c), np.float32)
        for ci in range(c):
            name = names[ci]
            bc = self.bands[name]
            scale = _band_scale(name)
            x = np.arcsinh(preds[..., ci].reshape(m, h * w).T / scale)   # (HW, M)
            y = np.clip(bc.forward_asinh(x), -SINH_CLIP, SINH_CLIP)
            out[..., ci] = (np.sinh(y) * scale).reshape(h, w).astype(np.float32)
        return out

    def needed_member_indices(self) -> list[int]:
        """Indices of the members actually used by the gate (surviving in any
        band). Pruned members contribute nothing and don't need to be run."""
        m = len(self.member_labels)
        mask = np.zeros(m, bool)
        for bc in self.bands.values():
            s = np.asarray(bc.surviving, bool)
            mask[:len(s)] |= s
        return [int(i) for i in np.where(mask)[0]]

    def member_pruned(self, index: int) -> bool:
        """True when the member at ``index`` is dropped in EVERY band — its gate
        weight is 0 everywhere, so the fused output does not depend on it (and it
        can be removed without changing anything)."""
        return 0 <= index < len(self.member_labels) \
            and index not in set(self.needed_member_indices())

    def without_member(self, index: int) -> Combiner:
        """A copy with the member at ``index`` removed from ``member_labels`` and
        every band's weights (``V`` column, bias ``a``, ``surviving`` mask),
        reindexed contiguous. **Exact only for a PRUNED member** (see
        :meth:`member_pruned`): dropping a weight-0, masked-off column leaves the
        max-over-surviving brightness and the softmax over the surviving members
        unchanged, so every kept member's weight — and the fused output — is
        identical. Lets the combiner survive archiving a member it never used."""
        keep = [i for i in range(len(self.member_labels)) if i != index]
        bands = {}
        for name, bc in self.bands.items():
            if self.kind in {STATS_RBF_GATE_KIND, MINMAX_RBF_GATE_KIND}:
                bands[name] = StatsRBFBandCombiner(
                    V=np.asarray(bc.V)[:, keep], a=np.asarray(bc.a)[keep],
                    centers=np.asarray(bc.centers), scales=np.asarray(bc.scales),
                    sigma=bc.sigma, surviving=np.asarray(bc.surviving, bool)[keep],
                    std_floor=bc.std_floor, feature_kind=bc.feature_kind)
            else:
                bands[name] = BandCombiner(
                    V=np.asarray(bc.V)[:, keep], a=np.asarray(bc.a)[keep],
                    centers=bc.centers, sigma=bc.sigma,
                    surviving=np.asarray(bc.surviving, bool)[keep])
        importance = {
            name: [float(values[i]) for i in keep]
            for name, values in self.member_importance.items()
            if len(values) == len(self.member_labels)
        }
        peaks = {
            name: [float(values[i]) for i in keep]
            for name, values in self.member_weight_peaks.items()
            if len(values) == len(self.member_labels)
        }
        integrals = {
            name: [float(values[i]) for i in keep]
            for name, values in self.member_weight_integrals.items()
            if len(values) == len(self.member_labels)
        }
        return replace(self, member_labels=[self.member_labels[i] for i in keep],
                       bands=bands, member_importance=importance,
                       member_weight_peaks=peaks,
                       member_weight_integrals=integrals)

    def upsample(self, ens, lr_array: np.ndarray) -> np.ndarray:
        """Combine one LR field, running the members required by this model.

        RBF gates can skip members pruned from every band.
        """
        keep = self.needed_member_indices()
        if not keep:
            raise RuntimeError("combiner has no surviving members")
        m = len(self.member_labels)
        preds = None
        for i in keep:
            sr = np.asarray(ens._models[i].upsample_array(lr_array), np.float32)
            if preds is None:
                preds = np.zeros((m, *sr.shape), np.float32)   # pruned slots stay 0
            preds[i] = sr
        return self.apply_field(preds)

    # -- interpretability -- #
    def effective_weights(self, band: str, *, n_levels: int = 25,
                          level_range: tuple[float, float] | None = None) -> dict:
        """A one-dimensional gate view for the UI.

        Max-RBF is exact on this brightness sweep. Mean/std RBF uses the
        equal-input, zero-disagreement slice; its HR-conditioned diagnostic
        remains the faithful view of real input stacks.
        """
        bc = self.bands[band]
        lr = level_range or self.level_range
        levels = np.linspace(lr[0], lr[1], int(n_levels))
        scale = _band_scale(band)
        return {"brightness_asinh": levels,
                "brightness_e": np.sinh(levels) * scale,
                "jacobian": bc.weights_at(levels)}       # (L, M), rows sum to 1

    def surviving_members(self) -> dict[str, list[bool]]:
        return {b: self.bands[b].surviving.astype(bool).tolist()
                for b in self.bands}


# ---------------------------------------------------------------------------
# Fit-data assembly (unchanged — streaming, brightness-stratified)
# ---------------------------------------------------------------------------

class FitBufferAccumulator:
    """Streaming per-band fit-buffer builder. ``add(preds, hr)`` one field at a
    time — so the caller never retains the full (multi-GB) member stack — pixel-
    subsampling **stratified by brightness** (equal quota per asinh-brightness
    bin) so faint structure isn't drowned by the dominant sky. ``buffers()``
    returns ``{band: (X(N,M), y(N,))}`` in asinh space."""

    def __init__(self, band_names, *, max_rows: int = 3_000_000,
                 n_bright_bins: int = 8, per_bin_per_field: int = 2000,
                 level_range: tuple[float, float] = (-1.0, 12.0), seed: int = 0):
        self.band_names = tuple(band_names)
        self.max_rows = int(max_rows)
        self.n_bright_bins = int(n_bright_bins)
        self.per_bin_per_field = int(per_bin_per_field)
        self.edges = np.linspace(level_range[0], level_range[1],
                                 int(n_bright_bins) + 1)
        self._rng = np.random.default_rng(seed)
        self._X = {b: [] for b in self.band_names}
        self._y = {b: [] for b in self.band_names}
        self._n = dict.fromkeys(self.band_names, 0)

    def add(self, preds: np.ndarray, hr: np.ndarray) -> None:
        preds = np.asarray(preds, np.float32)
        hr = np.asarray(hr, np.float32)
        m = preds.shape[0]
        for ci, name in enumerate(self.band_names):
            if self._n[name] >= self.max_rows:
                continue
            scale = _band_scale(name)
            xs = np.arcsinh(preds[..., ci].reshape(m, -1).T / scale)   # (P, M)
            ys = np.arcsinh(hr[..., ci].reshape(-1) / scale)           # (P,)
            bin_idx = np.clip(np.digitize(ys, self.edges) - 1,
                              0, self.n_bright_bins - 1)
            for b in range(self.n_bright_bins):
                if self._n[name] >= self.max_rows:
                    break
                sel = np.where(bin_idx == b)[0]
                if sel.size == 0:
                    continue
                take = int(min(self.per_bin_per_field, sel.size,
                               self.max_rows - self._n[name]))
                pick = self._rng.choice(sel, size=take, replace=False)
                self._X[name].append(xs[pick])
                self._y[name].append(ys[pick])
                self._n[name] += take

    def buffers(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in self.band_names:
            if self._X[name]:
                out[name] = (np.concatenate(self._X[name]).astype(np.float32),
                             np.concatenate(self._y[name]).astype(np.float32))
            else:
                out[name] = (np.zeros((0, 0), np.float32), np.zeros((0,), np.float32))
        return out


def build_fit_buffers_from_fields(field_iter, band_names, **kw
                                  ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Convenience: drive a :class:`FitBufferAccumulator` from an iterator of
    ``(preds(M,H,W,C), hr(H,W,C))`` fields (electrons)."""
    acc = FitBufferAccumulator(band_names, **kw)
    for preds, hr in field_iter:
        acc.add(preds, hr)
    return acc.buffers()


def build_fit_buffers(base_dir: str, records_dir: str, *, subset: str = "validate",
                      num_images: int = 100, ens=None, on_progress=None,
                      **kw) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    """Records-backed convenience: run each ``validate`` field through the
    STARFULL ensemble and assemble the per-band fit buffers. Returns
    ``(buffers, member_labels)``."""
    from euclid_polish.ensemble import EnsembleModel
    from euclid_polish.image.collection import ImageSet
    from euclid_polish.image.tfio import tfrecord_path

    ens = ens or EnsembleModel(base_dir, starless=False)
    labels = list(ens.member_labels)
    lr_list = list(ImageSet.read(tfrecord_path(records_dir, f"dirty_{subset}"),
                                 num_images=num_images))
    hr_by = {h.index: h for h in ImageSet.read(
        tfrecord_path(records_dir, f"hr_{subset}"), num_images=num_images)}

    def _iter():
        n = len(lr_list)
        for i, lr in enumerate(lr_list):
            hr = hr_by.get(lr.index)
            if hr is None:
                continue
            preds = ens.member_arrays(lr.data)
            if on_progress is not None:
                on_progress(i + 1, n, f"field {lr.index}")
            yield preds, np.asarray(hr.data, np.float32)

    buffers = build_fit_buffers_from_fields(_iter(), BAND_NAMES, **kw)
    return buffers, labels


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def _fit_one_band(X: np.ndarray, y: np.ndarray, *, n_kernels: int,
                  sigma_scale: float, level_range: tuple[float, float],
                  steps: int, lr: float, batch: int, seed: int, holdout: float
                  ) -> tuple[BandCombiner, np.ndarray, np.ndarray]:
    import tensorflow as tf

    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.float32)
    n, m = X.shape
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    X, y = X[order], y[order]
    n_val = int(n * holdout)
    Xtr, ytr = (X[n_val:], y[n_val:]) if n_val > 0 else (X, y)
    Xval, yval = (X[:n_val], y[:n_val]) if n_val > 0 else (X, y)

    centers = np.linspace(level_range[0], level_range[1],
                          int(n_kernels)).astype(np.float32)
    sigma = float((level_range[1] - level_range[0])
                  / max(int(n_kernels) - 1, 1) * float(sigma_scale))

    tf.random.set_seed(seed)
    V = tf.Variable(tf.zeros((int(n_kernels), m), tf.float32))   # start uniform
    a = tf.Variable(tf.zeros((m,), tf.float32))
    cen = tf.constant(centers)
    opt = tf.keras.optimizers.Adam(lr)

    def _weights(Xb):
        b = tf.reduce_max(Xb, axis=1, keepdims=True)            # (n, 1)
        phi = tf.exp(-0.5 * ((b - cen) / sigma) ** 2)           # (n, K)
        return tf.nn.softmax(tf.matmul(phi, V) + a, axis=1)     # (n, M)

    def _forward(Xb):
        return tf.reduce_sum(_weights(Xb) * Xb, axis=1)

    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.abs(_forward(xb) - yb))
        grads = tape.gradient(loss, [V, a])
        opt.apply_gradients(zip(grads, [V, a], strict=True))

    bs = int(min(batch, max(1, len(Xtr))))
    ds = (tf.data.Dataset.from_tensor_slices((Xtr, ytr))
          .shuffle(min(len(Xtr), 100_000), seed=seed, reshuffle_each_iteration=True)
          .batch(bs).repeat())
    it = iter(ds)

    def _val_l1():
        return float(np.mean(np.abs(_forward(tf.constant(Xval)).numpy() - yval)))

    eval_every = max(1, int(steps) // 20)
    patience = 5
    best = np.inf
    best_w = None
    stale = 0
    for s in range(int(steps)):
        xb, yb = next(it)
        train_step(xb, yb)
        if (s + 1) % eval_every == 0 or s == int(steps) - 1:
            v = _val_l1()
            if v < best - 1e-6:
                best, stale = v, 0
                best_w = [V.numpy().copy(), a.numpy().copy()]
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_w is None:
        best_w = [V.numpy().copy(), a.numpy().copy()]
    Vn, an = best_w
    # No pruning here. Spatial conditional ablation is a separate bounded
    # post-fit stage; return the holdout for the fitted gate's validation loss.
    bc = BandCombiner(V=Vn.astype(np.float32), a=an.astype(np.float32),
                      centers=centers, sigma=sigma, surviving=np.ones(m, bool))
    return bc, Xval, yval


def _spread_rbf_centers(rows: np.ndarray, seeds: np.ndarray, count: int, *,
                        min_separation: float = DEFAULT_STATS_RBF_MIN_CENTER_SEPARATION
                        ) -> np.ndarray:
    """Keep useful anchors apart, then cover the remaining occupied manifold.

    Both inputs are in normalized gate coordinates.  Close seed centroids are
    redundant when their distance is small compared with the RBF width, so the
    greedy seed pass retains an anchor only when it clears ``min_separation``.
    Missing anchors are selected from real feature rows by max-min (farthest-
    point) coverage.  Thus separation does not push centers into unsupported
    parts of the 2-D plane.  If the data itself has fewer distinct locations
    than requested, max-min selection degrades gracefully and still returns the
    requested number of centers.
    """
    rows = np.asarray(rows, np.float64).reshape(-1, 2)
    seeds = np.asarray(seeds, np.float64).reshape(-1, 2)
    rows = rows[np.all(np.isfinite(rows), axis=1)]
    seeds = seeds[np.all(np.isfinite(seeds), axis=1)]
    count = min(max(1, int(count)), len(rows))
    if count <= 0:
        return np.empty((0, 2), np.float64)

    chosen: list[np.ndarray] = []
    row_distance = np.full(len(rows), np.inf, np.float64)

    def add(center: np.ndarray) -> None:
        chosen.append(np.asarray(center, np.float64).copy())
        delta = rows - chosen[-1]
        np.minimum(row_distance, np.sqrt(np.sum(delta * delta, axis=1)),
                   out=row_distance)

    floor = max(0.0, float(min_separation))
    for center in seeds:
        if len(chosen) >= count:
            break
        if not chosen:
            add(center)
            continue
        delta = np.asarray(chosen) - center
        if float(np.sqrt(np.sum(delta * delta, axis=1)).min()) >= floor:
            add(center)

    while len(chosen) < count:
        idx = int(np.argmax(row_distance))
        add(rows[idx])

    return np.asarray(chosen, np.float64)


def _stats_rbf_geometry(X: np.ndarray, y: np.ndarray, n_kernels: int, *, seed: int,
                        kind: str = STATS_RBF_GATE_KIND
                        ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Hybrid data/tail/hard-example RBF geometry from validation features.

    Sixty-four kernels over two summary features cost only 128 distance terms
    per pixel; clustering a bounded validation sample keeps fitting cheap
    even though the buffers can contain millions of examples.
    """
    raw = _stats_features(np.asarray(X, np.float32), kind)
    rng = np.random.default_rng(seed)
    cap = min(len(raw), max(20_000, int(n_kernels) * 512))
    sample_raw = (raw if len(raw) <= cap
                  else raw[rng.choice(len(raw), cap, replace=False)])
    std_floor = _std_floor(sample_raw) if kind == STATS_RBF_GATE_KIND else 0.0
    sample = _stats_geometry_features(sample_raw, std_floor, kind)
    floors = np.array([0.25, 0.02], np.float64)
    scales = np.maximum(np.std(sample, axis=0), floors)
    zn = sample / scales
    k = min(max(1, int(n_kernels)), len(zn))

    def cluster(rows, count):
        count = min(int(count), len(rows))
        if count <= 0:
            return np.empty((0, 2), np.float64)
        centers = rows[rng.choice(len(rows), count, replace=False)].copy()
        for _ in range(8):
            dist2 = np.sum((rows[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dist2, axis=1)
            for j in range(count):
                hit = rows[labels == j]
                centers[j] = hit.mean(axis=0) if len(hit) else rows[rng.integers(len(rows))]
        return centers

    # Scale allocations with K: most centres follow density, while tail and
    # residual reservoirs guarantee support for rare occupied feature regions.
    n_tail = min(max(2, int(np.ceil(k * 0.25))), k)
    n_hard = min(max(2, int(np.ceil(k * 0.125))), k - n_tail)
    n_base = k - n_tail - n_hard
    q_mean, q_std = np.quantile(sample[:, 0], 0.99), np.quantile(sample[:, 1], 0.99)
    tail_rows = zn[(sample[:, 0] >= q_mean) | (sample[:, 1] >= q_std)]
    # Hard examples are selected by the naive ensemble-mean error, then
    # clustered in feature space so duplicate star-core pixels share support.
    residual = np.abs(np.mean(np.asarray(X, np.float64), axis=1) - np.asarray(y, np.float64))
    n_hard_rows = max(1, int(np.ceil(len(residual) * 0.01)))
    hard_idx = np.argpartition(residual, -n_hard_rows)[-n_hard_rows:]
    hard_raw = raw[hard_idx]
    hard_rows = _stats_geometry_features(hard_raw, std_floor, kind) / scales
    # Rare and difficult regions get first choice of anchors.  A global
    # separation pass then removes cross-reservoir duplicates and fills their
    # slots by max-min coverage of the occupied feature manifold.
    tail_centers = cluster(tail_rows, n_tail)
    hard_centers = cluster(hard_rows, n_hard)
    base_centers = cluster(zn, n_base)
    seeds = np.concatenate((tail_centers, hard_centers, base_centers), axis=0)
    coverage_rows = np.concatenate((zn, tail_rows, hard_rows), axis=0)
    centers = _spread_rbf_centers(coverage_rows, seeds, k)
    if k == 1:
        sigma = 1.0
    else:
        sep = np.sqrt(np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(sep, np.inf)
        sigma = max(0.5, float(np.median(np.min(sep, axis=1))) * 1.25)
    return (centers * scales).astype(np.float32), scales.astype(np.float32), sigma, std_floor


def _fit_one_band_stats_rbf(X: np.ndarray, y: np.ndarray, *, n_kernels: int,
                            steps: int, lr: float, batch: int, seed: int,
                            holdout: float, kind: str = STATS_RBF_GATE_KIND
                            ) -> tuple[StatsRBFBandCombiner, np.ndarray, np.ndarray]:
    """Fit the compact mean/std RBF gate with frozen data-covering centers.

    This deliberately uses a NumPy clipped-Adam loop rather than TensorFlow's
    input pipeline. There are only ``K × M + M`` learned values; matching the
    max-RBF step, learning-rate, and batch budgets makes feature comparisons
    meaningful without carrying a heavyweight graph/dataset.
    """
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.float32)
    n, m = X.shape
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    X, y = X[order], y[order]
    n_val = int(n * holdout)
    Xtr, ytr = (X[n_val:], y[n_val:]) if n_val > 0 else (X, y)
    Xval, yval = (X[:n_val], y[:n_val]) if n_val > 0 else (X, y)
    centers, scales, sigma, std_floor = _stats_rbf_geometry(Xtr, ytr, n_kernels, seed=seed, kind=kind)

    ztr = _stats_geometry_features(_stats_features(Xtr, kind), std_floor, kind)
    zval = _stats_geometry_features(_stats_features(Xval, kind), std_floor, kind)
    centers = centers.astype(np.float64)
    scales = np.maximum(scales.astype(np.float64), 1e-6)

    def _basis(z):
        d = (z[:, None, :] - centers[None, :, :]) / scales[None, None, :]
        return np.exp(-0.5 * np.sum(d * d, axis=2) / (sigma * sigma))

    def _weights(z, V, a):
        logits = _basis(z) @ V + a
        logits = np.nan_to_num(logits, nan=0.0, posinf=80.0, neginf=-80.0)
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    # Use the same requested optimizer budget as max-RBF. The public defaults
    # are 3,000 steps, lr=0.01, and batch=16,384 for every RBF regime.
    n_steps = max(1, int(steps))
    bs = min(max(1, int(batch)), max(1, len(Xtr)))
    rate = float(lr)
    V = np.zeros((len(centers), m), np.float64)  # uniform softmax start
    a = np.zeros((m,), np.float64)
    mV = np.zeros_like(V); vV = np.zeros_like(V)
    ma = np.zeros_like(a); va = np.zeros_like(a)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best, best_w, stale = np.inf, None, 0
    eval_every = max(10, n_steps // 20)
    for step in range(1, n_steps + 1):
        pick = (np.arange(len(Xtr)) if bs == len(Xtr)
                else rng.integers(0, len(Xtr), size=bs))
        xb, yb, zb = Xtr[pick], ytr[pick], ztr[pick]
        phi = _basis(zb)
        w = _weights(zb, V, a)
        pred = np.sum(w * xb, axis=1)
        # d |Σ w·x − y| / d logits. This is the softmax Jacobian contracted
        # with the convex mixture, expressed without an M×M allocation.
        grad_logits = (np.sign(pred - yb)[:, None] * w
                       * (xb - pred[:, None]) / float(len(xb)))
        grad_V = phi.T @ grad_logits
        grad_a = grad_logits.sum(axis=0)
        # A few rare saturated star pixels can otherwise make L1's subgradient
        # jump sharply. Clip the *global* update while retaining its direction.
        grad_norm = float(np.sqrt(np.sum(grad_V * grad_V) + np.sum(grad_a * grad_a)))
        if grad_norm > 5.0:
            grad_V *= 5.0 / grad_norm
            grad_a *= 5.0 / grad_norm
        mV = beta1 * mV + (1.0 - beta1) * grad_V
        vV = beta2 * vV + (1.0 - beta2) * (grad_V * grad_V)
        ma = beta1 * ma + (1.0 - beta1) * grad_a
        va = beta2 * va + (1.0 - beta2) * (grad_a * grad_a)
        corr1, corr2 = 1.0 - beta1 ** step, 1.0 - beta2 ** step
        V -= rate * (mV / corr1) / (np.sqrt(vV / corr2) + eps)
        a -= rate * (ma / corr1) / (np.sqrt(va / corr2) + eps)
        np.clip(V, -20.0, 20.0, out=V)
        np.clip(a, -20.0, 20.0, out=a)
        if step % eval_every == 0 or step == n_steps:
            val_w = _weights(zval, V, a)
            value = float(np.mean(np.abs(np.sum(val_w * Xval, axis=1) - yval)))
            if value < best - 1e-6:
                best, stale = value, 0
                best_w = [V.copy(), a.copy()]
            else:
                stale += 1
                if stale >= 5:
                    break
    if best_w is None:
        best_w = [V.copy(), a.copy()]
    return (StatsRBFBandCombiner(
                V=best_w[0].astype(np.float32), a=best_w[1].astype(np.float32),
                centers=centers, scales=scales, sigma=float(sigma),
                surviving=np.ones(m, bool), std_floor=float(std_floor), feature_kind=kind), Xval, yval)


def _fit_stats_rbf_combiner(buffers, member_labels, *, n_kernels: int,
                            min_usage: float, steps: int, lr: float,
                            batch: int, seed: int, holdout: float,
                            level_range: tuple[float, float],
                            kind: str = STATS_RBF_GATE_KIND) -> Combiner:
    bands: dict[str, StatsRBFBandCombiner] = {}
    holdouts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (X, y) in buffers.items():
        if np.asarray(X).size == 0:
            continue
        bc, Xval, yval = _fit_one_band_stats_rbf(
            X, y, n_kernels=int(n_kernels), steps=int(steps), lr=float(lr),
            batch=int(batch), seed=int(seed), holdout=float(holdout), kind=kind)
        bands[name] = bc
        holdouts[name] = (Xval, yval)

    vals = [float(np.mean(np.abs(bc.forward_asinh(holdouts[name][0]) - holdouts[name][1])))
            for name, bc in bands.items()]
    return Combiner(member_labels=list(member_labels), n_kernels=int(n_kernels),
                    sigma_scale=1.0, min_usage=float(min_usage), bands=bands,
                    band_names=tuple(buffers.keys()), level_range=tuple(level_range),
                    val_l1=(float(np.mean(vals)) if vals else None),
                    kind=kind)


def fit_combiner(buffers, member_labels, *, n_kernels: int = DEFAULT_N_KERNELS,
                 sigma_scale: float = DEFAULT_SIGMA_SCALE,
                 min_usage: float | None = None,
                 level_range: tuple[float, float] = GATE_LEVEL_RANGE,
                 steps: int = 3000, lr: float = 1e-2, batch: int = 16384,
                 seed: int = 0, holdout: float = 0.1,
                 model_kind: str = RBF_GATE_KIND) -> Combiner:
    """Fit one per-band RBF brightness gate (convex mixture, L1 loss) for each
    band in ``buffers`` (``{band: (X(N,M), y(N,))}`` asinh space).

    ``min_usage`` is retained for artifact/API compatibility but does not
    remove members. Every fitted member remains active."""
    kind = normalize_model_kind(model_kind)
    if min_usage is None:
        min_usage = combiner_model_spec(kind).default_min_usage
    if kind in {STATS_RBF_GATE_KIND, MINMAX_RBF_GATE_KIND}:
        if int(n_kernels) == DEFAULT_N_KERNELS:
            n_kernels = DEFAULT_STATS_RBF_N_KERNELS
        return _fit_stats_rbf_combiner(
            buffers, member_labels, n_kernels=int(n_kernels),
            min_usage=float(min_usage), steps=int(steps), lr=float(lr),
            batch=int(batch), seed=int(seed), holdout=float(holdout),
            level_range=tuple(level_range), kind=kind)

    bands: dict[str, BandCombiner] = {}
    holdouts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (X, y) in buffers.items():
        if np.asarray(X).size == 0:
            continue
        bc, Xval, yval = _fit_one_band(X, y, n_kernels=int(n_kernels),
                                       sigma_scale=float(sigma_scale),
                                       level_range=level_range, steps=int(steps),
                                       lr=float(lr), batch=int(batch),
                                       seed=int(seed), holdout=float(holdout))
        bands[name] = bc
        holdouts[name] = (Xval, yval)

    vals = [float(np.mean(np.abs(bc.forward_asinh(holdouts[name][0]) - holdouts[name][1])))
            for name, bc in bands.items()]
    return Combiner(member_labels=list(member_labels), n_kernels=int(n_kernels),
                    sigma_scale=float(sigma_scale), min_usage=float(min_usage),
                    bands=bands, band_names=tuple(buffers.keys()),
                    level_range=tuple(level_range),
                    val_l1=(float(np.mean(vals)) if vals else None),
                    kind=RBF_GATE_KIND)


# ---------------------------------------------------------------------------
# Bounded rare-aware post-fit ablation
# ---------------------------------------------------------------------------

def combiner_region_ids(comb: Combiner, X: np.ndarray,
                        band: str = "VIS") -> np.ndarray:
    """Return the strongest RBF region for asinh member rows ``X(N,M)``.

    This is a geometric diagnostic only: it does not integrate gate weight or
    use population frequency, so a rare occupied region is represented on the
    same footing as a common one.
    """
    bc = comb.bands[band]
    X = np.asarray(X, np.float64)
    if isinstance(bc, StatsRBFBandCombiner):
        basis = bc._basis_from_raw_features(bc._raw_features(X))
    else:
        surv = np.asarray(bc.surviving, bool)
        brightness = np.max(X[:, surv] if surv.any() else X, axis=1)
        basis = _rbf(brightness, bc.centers, bc.sigma)
    return np.argmax(basis, axis=1).astype(np.int32)


def _set_member_surviving(comb: Combiner, index: int, value: bool) -> None:
    for bc in comb.bands.values():
        bc.surviving[index] = bool(value)


def _apply_combiner_points(comb: Combiner, stacks: np.ndarray) -> np.ndarray:
    """Apply a combiner to ``(N,M,C)`` point stacks without image allocation."""
    stacks = np.asarray(stacks, np.float32)
    n, _m, c = stacks.shape
    out = np.empty((n, c), np.float64)
    for ci, name in enumerate(comb.band_names[:c]):
        scale = _band_scale(name)
        x = np.arcsinh(stacks[..., ci] / scale)
        y = np.clip(comb.bands[name].forward_asinh(x), -SINH_CLIP, SINH_CLIP)
        out[:, ci] = np.sinh(y) * scale
    return out


def _apply_combiner_patches(comb: Combiner, patches: np.ndarray) -> np.ndarray:
    """Apply once to many square patches by packing them into one strip."""
    patches = np.asarray(patches, np.float32)
    p, m, size, width, c = patches.shape
    if size != width:
        raise ValueError("ablation patches must be square")
    strip = patches.transpose(1, 0, 2, 3, 4).reshape(m, p * size, size, c)
    return comb.apply_field(strip).reshape(p, size, size, c)


def _log_domain_mean(k: np.ndarray, values: np.ndarray, *,
                     k_min: float, k_max: float) -> float:
    if not (k_max > k_min > 0):
        return float("nan")
    k = np.asarray(k, float)
    values = np.asarray(values, float)
    keep = (np.isfinite(k) & np.isfinite(values) & (k >= k_min)
            & (k <= k_max) & (k > 0))
    if int(keep.sum()) < 2:
        return float("nan")
    order = np.argsort(k[keep])
    x = np.log(k[keep][order])
    y = values[keep][order]
    integral = (np.trapezoid(y, x) if hasattr(np, "trapezoid")
                else np.trapz(y, x))
    return float(integral / np.log(k_max / k_min))


def _patch_spectral_scores(truth: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-patch VIS coherence and |log transfer| error in mid/SR domains."""
    from euclid_polish.eval.power_spectrum import (
        LR_NYQUIST_CYC_ARCSEC,
        bin_powers,
        log_k_edges,
        normalized_log_scale_mean,
        ratios_from_powers,
        tukey_window_2d,
    )

    truth = np.asarray(truth, np.float64)
    pred = np.asarray(pred, np.float64)
    size = int(truth.shape[1])
    pixel_scale = float(Config.DEFAULT_PIXEL_SCALE)
    k_min_available = 1.0 / (size * pixel_scale)
    edges = log_k_edges(pixel_scale, kmin=k_min_available, nbins=12)
    k = np.sqrt(edges[:-1] * edges[1:])
    window = tukey_window_2d(size)
    stretch = float(Config.STRETCH_SCALE_E)
    coherence = np.full((len(truth), 2), np.nan)
    transfer_error = np.full((len(truth), 2), np.nan)
    domains = ((max(k_min_available, 1.0), LR_NYQUIST_CYC_ARCSEC),
               (LR_NYQUIST_CYC_ARCSEC, 1.0 / (2.0 * pixel_scale)))
    for i in range(len(truth)):
        ah = np.arcsinh(truth[i] / stretch)
        ap = np.arcsinh(pred[i] / stretch)
        if float(np.std(ah)) < 1e-6:
            continue
        bh, bs, bx, counts = bin_powers(ah, ap, pixel_scale, edges, window)
        transfer, corr = ratios_from_powers(bh, bs, bx, counts)
        transfer_penalty = np.abs(np.log(np.clip(transfer, 1e-6, 1e6)))
        for d, (lo, hi) in enumerate(domains):
            coherence[i, d] = normalized_log_scale_mean(
                k, corr, k_min=float(lo), k_max=float(hi))
            transfer_error[i, d] = _log_domain_mean(
                k, transfer_penalty, k_min=float(lo), k_max=float(hi))
    return coherence, transfer_error


def member_weight_diagnostics(
    comb: Combiner, buffers, *, chunk_rows: int = 32_768,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Peak and distribution-integrated gate share per member and band.

    The integral is the mean weight over the brightness-stratified validation
    rows. Rows are evaluated in bounded chunks, so neither diagnostic
    materializes a full ``N x M`` weight matrix.
    """
    m = len(comb.member_labels)
    peaks: dict[str, list[float]] = {}
    integrals: dict[str, list[float]] = {}
    for name, (X, _y) in buffers.items():
        X = np.asarray(X, np.float32)
        if name not in comb.bands or not len(X):
            peaks[name] = [0.0] * m
            integrals[name] = [0.0] * m
            continue
        if X.ndim != 2 or X.shape[1] != m:
            raise ValueError(
                f"{name} fit buffer has {X.shape[1] if X.ndim == 2 else '?'} "
                f"members, expected {m}")
        peak = np.zeros(m, np.float64)
        total = np.zeros(m, np.float64)
        count = 0
        for start in range(0, len(X), max(1, int(chunk_rows))):
            weights = comb.bands[name].weights(
                X[start:start + max(1, int(chunk_rows))])
            peak = np.maximum(peak, np.max(weights, axis=0))
            total += np.sum(weights, axis=0, dtype=np.float64)
            count += len(weights)
        peaks[name] = peak.astype(float).tolist()
        integrals[name] = (total / max(count, 1)).astype(float).tolist()
    return peaks, integrals


def peak_member_weights(comb: Combiner, buffers, *,
                        chunk_rows: int = 32_768) -> dict[str, list[float]]:
    """Backward-compatible peak-only diagnostic helper."""
    peaks, _integrals = member_weight_diagnostics(
        comb, buffers, chunk_rows=chunk_rows)
    return peaks


def _expand_refit_combiner(comb: Combiner, all_labels: list[str],
                           keep: list[int]) -> Combiner:
    """Restore a reduced refit to the original cache/member indexing."""
    if len(comb.member_labels) != len(keep):
        raise ValueError("refit member labels do not match kept columns")
    m = len(all_labels)
    bands: dict[str, BandCombiner | StatsRBFBandCombiner] = {}
    for name, bc in comb.bands.items():
        V = np.zeros((len(bc.V), m), np.float32)
        a = np.zeros(m, np.float32)
        surviving = np.zeros(m, bool)
        V[:, keep] = np.asarray(bc.V, np.float32)
        a[keep] = np.asarray(bc.a, np.float32)
        surviving[keep] = np.asarray(bc.surviving, bool)
        if isinstance(bc, StatsRBFBandCombiner):
            bands[name] = StatsRBFBandCombiner(
                V=V, a=a, centers=np.asarray(bc.centers, np.float32),
                scales=np.asarray(bc.scales, np.float32), sigma=float(bc.sigma),
                surviving=surviving, std_floor=float(bc.std_floor),
                feature_kind=bc.feature_kind)
        else:
            bands[name] = BandCombiner(
                V=V, a=a, centers=np.asarray(bc.centers, np.float32),
                sigma=float(bc.sigma), surviving=surviving)
    return replace(comb, member_labels=list(all_labels), bands=bands)


def _refit_patch_comparison(before: Combiner, after: Combiner,
                            patch_stacks: np.ndarray,
                            patch_truth: np.ndarray) -> dict[str, float | int]:
    """Compare a reduced refit to its predecessor on rare-aware patches."""
    patches = np.asarray(patch_stacks, np.float32)
    truth = np.asarray(patch_truth, np.float32)
    before_val = float(before.val_l1) if before.val_l1 is not None else float("inf")
    after_val = float(after.val_l1) if after.val_l1 is not None else float("inf")
    val_regret = ((after_val - before_val) / max(before_val, 0.01)
                  if np.isfinite(before_val) and np.isfinite(after_val)
                  else float("inf"))
    if (patches.ndim != 5 or truth.ndim != 4 or len(patches) != len(truth)
            or len(patches) < 4):
        return {
            "val_l1_regret": float(val_regret),
            "l1_region_max": float(val_regret),
            "coherence_drop_max": 0.0,
            "transfer_worsening_max": 0.0,
            "n_patches": int(len(patches) if patches.ndim else 0),
        }

    size = int(patches.shape[2])
    centre = size // 2
    point_stacks = patches[:, :, centre, centre, :]
    point_truth = truth[:, centre, centre, :]
    before_points = _apply_combiner_points(before, point_stacks)
    after_points = _apply_combiner_points(after, point_stacks)
    scale = np.asarray([_band_scale(b) for b in before.band_names], float)
    truth_asinh = np.arcsinh(point_truth / scale)
    before_err = np.abs(np.arcsinh(before_points / scale) - truth_asinh)
    after_err = np.abs(np.arcsinh(after_points / scale) - truth_asinh)

    region_regrets: list[float] = []
    for ci, name in enumerate(before.band_names[:point_stacks.shape[-1]]):
        X = np.arcsinh(point_stacks[..., ci] / scale[ci])
        regions = combiner_region_ids(before, X, band=name)
        for region in np.unique(regions):
            selected = regions == region
            base = float(np.mean(before_err[selected, ci]))
            candidate = float(np.mean(after_err[selected, ci]))
            region_regrets.append((candidate - base) / max(base, 0.01))

    before_patches = _apply_combiner_patches(before, patches)
    after_patches = _apply_combiner_patches(after, patches)
    before_coh, before_transfer = _patch_spectral_scores(
        truth[..., 0], before_patches[..., 0])
    after_coh, after_transfer = _patch_spectral_scores(
        truth[..., 0], after_patches[..., 0])
    coherence_delta = before_coh - after_coh
    transfer_delta = after_transfer - before_transfer
    finite_coh = coherence_delta[np.isfinite(coherence_delta)]
    finite_transfer = transfer_delta[np.isfinite(transfer_delta)]
    return {
        "val_l1_regret": float(val_regret),
        "l1_region_max": max(region_regrets, default=float("inf")),
        "coherence_drop_max": (float(np.max(finite_coh))
                                if finite_coh.size else 0.0),
        "transfer_worsening_max": (float(np.max(finite_transfer))
                                    if finite_transfer.size else 0.0),
        "n_patches": int(len(patches)),
    }


def iterative_peak_weight_refit(
    comb: Combiner,
    buffers,
    patch_stacks: np.ndarray,
    patch_truth: np.ndarray,
    *,
    min_peak_weight: float = 0.05,
    max_regret: float = 0.01,
    max_coherence_drop: float = 0.01,
    max_transfer_worsening: float = 0.02,
    min_members: int = 2,
    steps: int = 3000,
    lr: float = 1e-2,
    batch: int = 16_384,
    seed: int = 0,
    holdout: float = 0.1,
    on_refit=None,
) -> Combiner:
    """Iteratively refit without members whose peak gate share is too small.

    Each round first tries all sub-threshold members as one batch.  A failed
    rare-aware comparison retries the lower-peak half, providing binary
    backoff without assuming that validation quality is monotone in arbitrary
    member subsets.  Accepted refits are expanded back to the original member
    indexing with removed columns hard-masked, then peak weights are recomputed.
    """
    all_labels = list(comb.member_labels)
    m = len(all_labels)
    threshold = max(0.0, min(1.0, float(min_peak_weight)))
    tolerance = max(0.0, float(max_regret))
    working = comb
    active = working.needed_member_indices()
    blocked: set[int] = set()
    removed: list[int] = []
    attempts: list[dict] = []
    rows = {
        i: {"index": i, "label": label, "kept": True,
            "status": "not_evaluated", "peak_weight_max": 0.0,
            "peak_weights": {}}
        for i, label in enumerate(all_labels)
    }
    attempt_number = 0
    accepted_round = 0

    while threshold > 0.0 and len(active) > max(1, int(min_members)):
        importance = peak_member_weights(working, buffers)
        working.member_importance = importance
        aggregate = np.max(np.asarray(list(importance.values()), float), axis=0)
        for i in active:
            rows[i]["peak_weight_max"] = float(aggregate[i])
            rows[i]["peak_weights"] = {
                band: float(values[i]) for band, values in importance.items()}

        capacity = len(active) - max(1, int(min_members))
        pool = sorted(
            (i for i in active if i not in blocked and aggregate[i] < threshold),
            key=lambda i: (aggregate[i], i))[:capacity]
        if not pool:
            break

        trial = list(pool)
        accepted = False
        tried: set[tuple[int, ...]] = set()
        while trial:
            signature = tuple(trial)
            if signature in tried:
                break
            tried.add(signature)
            attempt_number += 1
            keep = [i for i in active if i not in set(trial)]
            if on_refit is not None:
                on_refit(attempt_number, [all_labels[i] for i in trial])
            reduced_buffers = {
                name: (np.asarray(X)[:, keep], y)
                for name, (X, y) in buffers.items()
            }
            reduced = fit_combiner(
                reduced_buffers, [all_labels[i] for i in keep],
                n_kernels=int(working.n_kernels),
                sigma_scale=float(working.sigma_scale), min_usage=0.0,
                level_range=tuple(working.level_range), steps=int(steps),
                lr=float(lr), batch=int(batch), seed=int(seed),
                holdout=float(holdout), model_kind=working.kind)
            candidate = _expand_refit_combiner(reduced, all_labels, keep)
            comparison = _refit_patch_comparison(
                working, candidate, patch_stacks, patch_truth)
            safe = (
                comparison["val_l1_regret"] <= tolerance
                and comparison["l1_region_max"] <= tolerance
                and comparison["coherence_drop_max"] <= float(max_coherence_drop)
                and comparison["transfer_worsening_max"] <= float(max_transfer_worsening)
            )
            attempts.append({
                "attempt": int(attempt_number),
                "round": int(accepted_round + 1),
                "removed": [all_labels[i] for i in trial],
                "accepted": bool(safe),
                "before_val_l1": working.val_l1,
                "after_val_l1": candidate.val_l1,
                **comparison,
            })
            if safe:
                accepted_round += 1
                for i in trial:
                    rows[i].update({
                        "kept": False, "status": "pruned_after_refit",
                        "prune_round": int(accepted_round),
                        "l1_region_max": comparison["l1_region_max"],
                        "coherence_drop_max": comparison["coherence_drop_max"],
                        "transfer_worsening_max": comparison["transfer_worsening_max"],
                    })
                removed.extend(trial)
                working = candidate
                active = keep
                blocked.clear()
                accepted = True
                break

            if len(trial) > 1:
                trial = trial[:max(1, len(trial) // 2)]
                continue
            rejected = trial[0]
            blocked.add(rejected)
            rows[rejected].update({
                "status": "retained_by_refit_veto",
                "l1_region_max": comparison["l1_region_max"],
                "coherence_drop_max": comparison["coherence_drop_max"],
                "transfer_worsening_max": comparison["transfer_worsening_max"],
            })
            remaining = [i for i in pool if i not in blocked]
            trial = remaining[:capacity]
        if not accepted:
            break

    final_importance = peak_member_weights(working, buffers)
    working.member_importance = final_importance
    final_aggregate = np.max(np.asarray(list(final_importance.values()), float), axis=0)
    active_set = set(working.needed_member_indices())
    for i in active_set:
        rows[i]["kept"] = True
        rows[i]["peak_weight_max"] = float(final_aggregate[i])
        rows[i]["peak_weights"] = {
            band: float(values[i]) for band, values in final_importance.items()}
        if threshold <= 0.0:
            rows[i]["status"] = "retained_threshold_disabled"
        elif final_aggregate[i] >= threshold:
            rows[i]["status"] = "retained_peak_above_threshold"
        elif rows[i]["status"] != "retained_by_refit_veto":
            rows[i]["status"] = "retained_no_safe_refit"

    report = {
        "metric": "iterative peak gate weight refit + patch spectrum",
        "peak_weight": "maximum gate share on represented validation pixels in any band",
        "min_peak_weight": float(threshold),
        "max_regret": float(tolerance),
        "max_coherence_drop": float(max_coherence_drop),
        "max_transfer_worsening": float(max_transfer_worsening),
        "binary_backoff": True,
        "n_patches": int(len(np.asarray(patch_stacks))),
        "attempts": attempts,
        "pruned": [all_labels[i] for i in sorted(set(removed))],
        "members": [rows[i] for i in range(m)],
    }
    working.min_peak_weight = float(threshold)
    working.max_prune_regret = float(tolerance)
    working.member_ablation = report
    return working


def bounded_ablation_prune(
    comb: Combiner,
    patch_stacks: np.ndarray,
    patch_truth: np.ndarray,
    patch_regions: np.ndarray,
    *,
    max_regret: float = 0.01,
    max_rounds: int = 4,
    shortlist: int = 6,
    max_coherence_drop: float = 0.01,
    max_transfer_worsening: float = 0.02,
) -> dict:
    """Conservatively prune with exact, sequential, bounded ablations.

    ``patch_stacks`` is ``(P,M,S,S,C)`` and is intentionally capped by the
    caller. Regional L1 is evaluated at patch centres for every member; only a
    small L1-safe shortlist receives the more expensive patch FFT evaluation.
    At most one member is removed per round and at most ``max_rounds`` members
    can be removed. Explicit ``max_regret=0`` computes diagnostics but removes
    nothing.
    """
    patches = np.asarray(patch_stacks, np.float32)
    truth = np.asarray(patch_truth, np.float32)
    regions = np.asarray(patch_regions, np.int32)
    if patches.ndim != 5 or truth.ndim != 4 or len(patches) != len(truth):
        raise ValueError("invalid bounded ablation patch arrays")
    if regions.ndim == 1:
        regions = np.repeat(regions[:, None], truth.shape[-1], axis=1)
    if regions.ndim != 2 or regions.shape != (len(patches), truth.shape[-1]):
        raise ValueError("patch regions must be (P,) or (P,C)")
    if len(patches) < 4:
        report = {"metric": "conditional ablation regret + patch spectrum",
                  "status": "insufficient_patches", "n_patches": int(len(patches)),
                  "members": []}
        comb.max_prune_regret = float(max_regret)
        comb.member_ablation = report
        return report

    size = int(patches.shape[2])
    centre = size // 2
    point_stacks = patches[:, :, centre, centre, :]
    point_truth = truth[:, centre, centre, :]
    labels = list(comb.member_labels)
    report_rows: dict[int, dict] = {}
    pruned: list[int] = []
    rounds = 1 if float(max_regret) <= 0 else max(1, int(max_rounds))

    for round_index in range(rounds):
        survivors = comb.needed_member_indices()
        if len(survivors) <= 2:
            break
        full_points = _apply_combiner_points(comb, point_stacks)
        scale = np.asarray([_band_scale(b) for b in comb.band_names], float)
        full_err = np.abs(np.arcsinh(full_points / scale)
                          - np.arcsinh(point_truth / scale))
        candidates: list[tuple[float, int, dict]] = []
        for member in survivors:
            _set_member_surviving(comb, member, False)
            try:
                ablated_points = _apply_combiner_points(comb, point_stacks)
            finally:
                _set_member_surviving(comb, member, True)
            ablated_err = np.abs(np.arcsinh(ablated_points / scale)
                                 - np.arcsinh(point_truth / scale))
            regret = (ablated_err - full_err) / np.maximum(full_err, 0.01)
            region_values = [
                float(np.max(regret[regions[:, ci] == region, ci]))
                for ci in range(regions.shape[1])
                for region in np.unique(regions[:, ci])
                if np.any(regions[:, ci] == region)
            ]
            worst = max(region_values, default=float("inf"))
            row = {
                "index": int(member), "label": labels[member], "kept": True,
                "status": "retained", "l1_region_max": float(worst),
                "coherence_drop_max": None, "transfer_worsening_max": None,
                "round": int(round_index + 1),
            }
            report_rows[member] = row
            candidates.append((worst, member, row))

        candidates.sort(key=lambda item: item[0])
        spectral_candidates = candidates[:max(1, min(int(shortlist), len(candidates)))]
        full_patches = _apply_combiner_patches(comb, patches)
        full_coh, full_transfer = _patch_spectral_scores(
            truth[..., 0], full_patches[..., 0])
        safe: list[tuple[float, int, dict]] = []
        for l1_regret, member, row in spectral_candidates:
            _set_member_surviving(comb, member, False)
            try:
                ablated_patches = _apply_combiner_patches(comb, patches)
            finally:
                _set_member_surviving(comb, member, True)
            coh, transfer = _patch_spectral_scores(truth[..., 0], ablated_patches[..., 0])
            coh_delta = full_coh - coh
            transfer_delta = transfer - full_transfer
            finite_coh = coh_delta[np.isfinite(coh_delta)]
            finite_transfer = transfer_delta[np.isfinite(transfer_delta)]
            coherence_drop = (float(np.max(finite_coh)) if finite_coh.size
                              else float("inf"))
            transfer_worsening = (float(np.max(finite_transfer))
                                  if finite_transfer.size else float("inf"))
            row["coherence_drop_max"] = coherence_drop if np.isfinite(coherence_drop) else None
            row["transfer_worsening_max"] = (transfer_worsening
                                               if np.isfinite(transfer_worsening) else None)
            row["spectral_patches"] = int(np.sum(np.any(np.isfinite(coh_delta), axis=1)))
            is_safe = (float(max_regret) > 0.0
                       and l1_regret <= float(max_regret)
                       and coherence_drop <= float(max_coherence_drop)
                       and transfer_worsening <= float(max_transfer_worsening))
            if is_safe:
                score = max(l1_regret / max(float(max_regret), 1e-9),
                            coherence_drop / max(float(max_coherence_drop), 1e-9),
                            transfer_worsening / max(float(max_transfer_worsening), 1e-9))
                safe.append((float(score), member, row))
            else:
                row["status"] = "retained_by_veto"

        if float(max_regret) <= 0.0 or not safe:
            break
        _score, member, row = min(safe, key=lambda item: item[0])
        _set_member_surviving(comb, member, False)
        row["kept"] = False
        row["status"] = "pruned"
        row["prune_round"] = int(round_index + 1)
        pruned.append(member)

    kept = set(comb.needed_member_indices())
    for member, row in report_rows.items():
        if member in kept and row.get("status") == "retained":
            row["status"] = "retained_not_shortlisted"
        row["kept"] = member in kept
    report = {
        "metric": "conditional ablation regret + patch spectrum",
        "regret": "max relative asinh-L1 increase within represented RBF regions",
        "spectrum": "max VIS coherence drop and |log T| worsening over 1-5 and 5-10 cyc/arcsec",
        "n_patches": int(len(patches)), "patch_size": int(size),
        "n_regions": int(sum(len(np.unique(regions[:, ci]))
                             for ci in range(regions.shape[1]))),
        "regions_per_band": {
            name: int(len(np.unique(regions[:, ci])))
            for ci, name in enumerate(comb.band_names[:regions.shape[1]])
        },
        "max_rounds": int(max_rounds),
        "shortlist": int(shortlist), "max_regret": float(max_regret),
        "max_coherence_drop": float(max_coherence_drop),
        "max_transfer_worsening": float(max_transfer_worsening),
        "pruned": [labels[i] for i in pruned],
        "members": [report_rows[i] for i in sorted(report_rows)],
    }
    comb.max_prune_regret = float(max_regret)
    comb.member_ablation = report
    return report


def recompute_holdout_l1(comb: Combiner, buffers, *, holdout: float = 0.1,
                         seed: int = 0, chunk_rows: int = 32_768) -> float | None:
    """Re-evaluate the original deterministic holdout after survivor changes.

    Rows are gathered and evaluated in bounded chunks; unlike fitting, this
    never copies or materialises the full shuffled buffer.
    """
    losses: list[float] = []
    for name, (X, y) in buffers.items():
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float32)
        if not len(X) or name not in comb.bands:
            continue
        n_val = int(len(X) * float(holdout))
        if n_val <= 0:
            indices = np.arange(len(X), dtype=np.int64)
        else:
            indices = np.random.default_rng(seed).permutation(len(X))[:n_val]
        total = 0.0
        count = 0
        for start in range(0, len(indices), max(1, int(chunk_rows))):
            idx = indices[start:start + int(chunk_rows)]
            pred = comb.bands[name].forward_asinh(X[idx])
            total += float(np.sum(np.abs(pred - y[idx]), dtype=np.float64))
            count += len(idx)
        if count:
            losses.append(total / count)
    comb.val_l1 = float(np.mean(losses)) if losses else None
    return comb.val_l1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _combiner_dir(base_dir: str, artifact_dir: str | None = None) -> str:
    """Return a combiner artifact directory without changing the default path."""
    return os.path.join(base_dir, artifact_dir or "combiner")


def save_combiner(comb: Combiner, base_dir: str, *,
                  artifact_dir: str | None = None) -> None:
    d = _combiner_dir(base_dir, artifact_dir)
    os.makedirs(d, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for name, bc in comb.bands.items():
        if comb.kind == RBF_GATE_KIND:
            if not isinstance(bc, BandCombiner):
                raise TypeError("RBF combiner has a non-RBF band")
            arrays[f"{name}__V"] = np.asarray(bc.V, np.float32)
            arrays[f"{name}__a"] = np.asarray(bc.a, np.float32)
            arrays[f"{name}__centers"] = np.asarray(bc.centers, np.float32)
            arrays[f"{name}__sigma"] = np.asarray([bc.sigma], np.float32)
        elif comb.kind in {STATS_RBF_GATE_KIND, MINMAX_RBF_GATE_KIND}:
            if not isinstance(bc, StatsRBFBandCombiner):
                raise TypeError("stats RBF combiner has a non-stats-RBF band")
            arrays[f"{name}__V"] = np.asarray(bc.V, np.float32)
            arrays[f"{name}__a"] = np.asarray(bc.a, np.float32)
            arrays[f"{name}__centers"] = np.asarray(bc.centers, np.float32)
            arrays[f"{name}__scales"] = np.asarray(bc.scales, np.float32)
            arrays[f"{name}__sigma"] = np.asarray([bc.sigma], np.float32)
            arrays[f"{name}__std_floor"] = np.asarray([bc.std_floor], np.float32)
        else:
            raise ValueError(f"unsupported combiner kind: {comb.kind}")
        arrays[f"{name}__mask"] = np.asarray(bc.surviving, bool)
    np.savez_compressed(os.path.join(d, "combiner.npz"), **arrays)
    manifest = {
        "kind": comb.kind,
        "member_labels": list(comb.member_labels),
        "n_kernels": int(comb.n_kernels),
        "sigma_scale": float(comb.sigma_scale),
        "min_usage": float(comb.min_usage),
        "max_prune_regret": float(comb.max_prune_regret),
        "min_peak_weight": float(comb.min_peak_weight),
        "level_range": list(comb.level_range),
        "band_names": list(comb.band_names),
        "stretch_e": {b: _band_scale(b) for b in comb.band_names},
        "records_fp": comb.records_fp,
        "starfull": bool(comb.starfull),
        "val_l1": comb.val_l1,
        "surviving": comb.surviving_members(),
        "member_importance": comb.member_importance,
        "member_weight_peaks": comb.member_weight_peaks,
        "member_weight_integrals": comb.member_weight_integrals,
        "member_ablation": comb.member_ablation,
        "fit_meta": comb.fit_meta,
    }
    if comb.kind in {STATS_RBF_GATE_KIND, MINMAX_RBF_GATE_KIND}:
        manifest["feature_names"] = list(combiner_model_spec(comb.kind).feature_names or ())
    with open(os.path.join(d, "combiner.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def load_combiner(base_dir: str, *, member_labels: list[str] | None = None,
                  artifact_dir: str | None = None
                  ) -> Combiner | None:
    """Load a persisted combiner, or ``None`` if absent, **stale** (its saved
    member labels no longer match ``member_labels``), or an incompatible/old
    format (e.g. a pre-RBF combiner)."""
    d = _combiner_dir(base_dir, artifact_dir)
    jp, npzp = os.path.join(d, "combiner.json"), os.path.join(d, "combiner.npz")
    if not (os.path.exists(jp) and os.path.exists(npzp)):
        return None
    try:
        with open(jp) as f:
            man = json.load(f)
        kind = normalize_model_kind(man.get("kind"))
        if kind not in COMBINER_MODELS:
            return None
        # Old stats-RBF artifacts have incompatible feature geometry. Treat
        # them as unavailable rather than silently applying raw-std centres.
        if (kind != RBF_GATE_KIND and man.get("feature_names")
                != list(combiner_model_spec(kind).feature_names or ())):
            return None
        if member_labels is not None and list(man["member_labels"]) != list(member_labels):
            return None
        z = np.load(npzp)
        bands: dict[str, BandCombiner | StatsRBFBandCombiner] = {}
        for name in man["band_names"]:
            if kind == RBF_GATE_KIND:
                bands[name] = BandCombiner(
                    V=z[f"{name}__V"], a=z[f"{name}__a"],
                    centers=z[f"{name}__centers"].reshape(-1),
                    sigma=float(z[f"{name}__sigma"][0]),
                    surviving=z[f"{name}__mask"])
            elif kind in {STATS_RBF_GATE_KIND, MINMAX_RBF_GATE_KIND}:
                bands[name] = StatsRBFBandCombiner(
                    V=z[f"{name}__V"], a=z[f"{name}__a"],
                    centers=z[f"{name}__centers"], scales=z[f"{name}__scales"],
                    sigma=float(z[f"{name}__sigma"][0]),
                    surviving=z[f"{name}__mask"],
                    std_floor=float(z[f"{name}__std_floor"][0]), feature_kind=kind)
        return Combiner(
            member_labels=list(man["member_labels"]),
            n_kernels=int(man.get("n_kernels", DEFAULT_N_KERNELS)),
            sigma_scale=float(man.get("sigma_scale", DEFAULT_SIGMA_SCALE)),
            min_usage=float(man.get("min_usage", 0.0)), bands=bands,
            band_names=tuple(man["band_names"]),
            level_range=tuple(man.get("level_range", GATE_LEVEL_RANGE)),
            records_fp=man.get("records_fp"), starfull=bool(man.get("starfull", True)),
            val_l1=man.get("val_l1"), fit_meta=man.get("fit_meta", {}),
            kind=kind,
            max_prune_regret=float(man.get("max_prune_regret", 0.0)),
            min_peak_weight=float(man.get("min_peak_weight", 0.0)),
            member_ablation=dict(man.get("member_ablation") or {}),
            member_importance={
                str(name): [float(v) for v in values]
                for name, values in (man.get("member_importance") or {}).items()
            },
            member_weight_peaks={
                str(name): [float(v) for v in values]
                for name, values in (man.get("member_weight_peaks") or {}).items()
            },
            member_weight_integrals={
                str(name): [float(v) for v in values]
                for name, values in (man.get("member_weight_integrals") or {}).items()
            })
    except (OSError, ValueError, KeyError, TypeError):
        return None
