"""Knee attribution: which asinh knees a multi-knee model relies on, for which pixels.

A multi-knee member reads its LR bands stretched at several knees,
``s_{k,c} = asinh(x_c / q_k)``, stacked knee-major (channel ``k·C + c``, see
:func:`~euclid_polish.training.augmentation.asinh_stretch_multi_knee`). For
one predicted pixel ``y`` (an HR position and output band, in the electrons
the member serves) one backward pass gives ``g = ∂y/∂s``, and the chain rule
splits the sensitivity to the LR electrons exactly into knee terms::

    ∂y/∂x_c(u) = Σ_k t_{k,c}(u),    t_{k,c}(u) = g_{k,c}(u) / √(q_k² + x_c(u)²)

A term is per input electron, and the bands' electrons differ in scale (the
VIS sky noise is ~20 e⁻, NISP's ~2 e⁻ per pixel), so each band's terms are weighted
by the band's noise σ_c (:func:`noise_sigma`): ``σ_c·|t_{k,c}|`` is the change
of ``y`` for a 1σ change of band ``c`` through knee ``k``, comparable across
bands. :func:`attribute` reports, per sampled pixel, the share of
``Σ σ|t|`` each knee carries (``knee_share``) and each input band carries
(``band_share``), each input band's own split over the knees
(``knee_share_by_input``, independent of σ), the total (``sensitivity``, in
output electrons per input σ), and an ablation check (``ablation_share``: each
knee's channels replaced by their window mean, the change of ``y`` per knee).

The sums run over the whole receptive field, which is mostly sky around any
pixel, so they say what the prediction depends on overall. The ``*_local``
shares sum only the 3×3 LR pixels under the sample (:data:`LOCAL_RADIUS`):
how the pixel's own neighbourhood is read.

Pixels are classed per HR pixel and band (:func:`predicted_classes` on any
field, :func:`truth_classes` on synthetic fields, :func:`real_classes` on real
tiles), sampled per class (:func:`sample_pixels`) and summarised over fields
by class and by brightness (:func:`aggregate`). Pure numerics + TF, no I/O:
the job, the store and the API are
:mod:`euclid_polish.web.helpers.knee_attribution_runs`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.eval.spatial_gate_compare import BRIGHTNESS_EDGES, BRIGHTNESS_NAMES
from euclid_polish.training.augmentation import asinh_stretch_multi_knee
from euclid_polish.training.inference import default_head_knee
from euclid_polish.training.models.output_knee import SINH_CLIP

#: Predicted-image bins: the combiner diagnostic's brightness classes.
PRED_CLASSES: tuple[str, ...] = tuple(BRIGHTNESS_NAMES)
#: Synthetic truth classes, in priority order (first match wins).
TRUTH_CLASSES: tuple[str, ...] = ("saturated", "star core", "star wing", "galaxy core",
                                  "galaxy outskirts", "faint", "sky")
#: Real-tile classes: the predicted bins plus LR pixels that are 0 or non-finite.
REAL_CLASSES: tuple[str, ...] = PRED_CLASSES + ("blank",)
#: The predicted-bin edges in electrons, ≈ (2.0, 10.0, 52.1, 362.7): the
#: combiner diagnostic's edges are in ``asinh(x / 100)`` units.
BRIGHTNESS_EDGES_E: tuple[float, ...] = tuple(
    float(Config.STRETCH_SCALE_E * math.sinh(edge)) for edge in BRIGHTNESS_EDGES)
#: Truth above this (and up to the first edge) is "faint", below it "sky".
FAINT_FLOOR_E = 0.01
#: Brightness-curve bins: half-decade edges from 1e-3 to 1e7 e⁻ (21 edges).
LOG_EDGES_E: np.ndarray = np.logspace(-3.0, 7.0, 21)
#: What the network's output is: electrons (a learned output knee), one image
#: stretched at a fixed output knee, or one block per knee (multi-head).
OUTPUTS: tuple[str, ...] = ("electrons", "stretched", "heads")
#: Smallest LR window side (the ablation's context).
MIN_WINDOW_SIDE = 64
#: The local shares' reach in LR pixels around the sample's LR pixel (1: 3×3).
LOCAL_RADIUS = 1
#: Lag (LR pixels) of the pixel differences :func:`noise_sigma` reads. NISP
#: is resampled from 0.3″ to the 0.1″ grid, so its noise is correlated over
#: neighbours (ρ ≈ 0.86): lag-1 differences see only ~0.38 of its pixel σ.
#: From lag 4 on the noise is independent (synthetic fields: lag-4 σ within
#: 2 % of the empty-sky pixel σ in all four bands, sources included).
NOISE_LAG = 4
#: The gradient splits: over the receptive field, and over the local block.
_SPLITS: tuple[tuple[str, str], ...] = (("", "the receptive field"), ("_local", "the local block"))

_BLANK = REAL_CLASSES.index("blank")
#: Relative size of float32 rounding noise (a few ulps): ablation changes
#: below it are not resolved and count as 0.
_ROUNDING = 8.0 * float(np.finfo(np.float32).eps)


@dataclass(frozen=True)
class KneeNet:
    """A multi-knee network and how to read the image its member serves.

    ``model`` maps stretched ``(N, h, w, K·C)`` to ``(N, h·scale, w·scale,
    Cout)``: electrons (``output="electrons"``, a learned output knee), one
    image stretched at ``output_knee`` (``"stretched"``) or ``K`` stretched
    blocks of which the member serves ``head_index`` (``"heads"``).
    """

    model: Any
    knees: tuple[float, ...]
    depth: int
    output: str
    output_knee: float | None = None
    head_index: int | None = None
    scale: int = 2
    n_bands: int = 4
    # Compiled attribution passes, built on first use.
    _fns: dict = field(default_factory=dict, init=False, repr=False, compare=False, hash=False)

    def __post_init__(self):
        object.__setattr__(self, "knees", tuple(float(q) for q in self.knees))
        if not self.knees:
            raise ValueError("a KneeNet needs at least one knee")
        if self.output not in OUTPUTS:
            raise ValueError(f"output must be one of {OUTPUTS}, got {self.output!r}")
        if self.output == "stretched" and self.output_knee is None:
            raise ValueError("a 'stretched' output needs its output_knee")
        if self.output == "heads" and (self.head_index is None
                                       or not 0 <= int(self.head_index) < len(self.knees)):
            raise ValueError(f"a 'heads' output needs a head_index in [0, {len(self.knees)}), "
                             f"got {self.head_index!r}")

    @classmethod
    def from_member(cls, member) -> KneeNet:
        """The network of a multi-knee :class:`~euclid_polish.model.Model`,
        read the way ``member.upsample_array`` serves it. A member reading a
        single knee has nothing to attribute: ``ValueError``."""
        knees = getattr(member, "_asinh_knees", None)
        if not knees or len(knees) < 2:
            knee = getattr(member, "_asinh_knee", None)
            which = "the default per-band knee" if knee is None else f"one knee, {float(knee):g} e⁻"
            raise ValueError(f"not a multi-knee model: it reads {which}, so there are no knees "
                             "to attribute")
        knees = tuple(float(q) for q in knees)
        model = member._tf_model
        common = {"model": model, "knees": knees, "depth": int(member._num_res_blocks),
                  "scale": int(member._scale),
                  "n_bands": int(model.inputs[0].shape[-1]) // len(knees)}
        if getattr(member, "_learn_output_knee", False):
            return cls(output="electrons", **common)
        output_knee = getattr(member, "_output_knee", None)
        if output_knee is not None:
            return cls(output="stretched", output_knee=float(output_knee), **common)
        head = knees.index(float(default_head_knee(knees)))
        return cls(output="heads", output_knee=knees[head], head_index=head, **common)

    def predict_electrons(self, stretched: tf.Tensor) -> tf.Tensor:
        """Stretched ``(N, h, w, K·C)`` → the served image ``(N, H, W, C)``
        in electrons, as :func:`~euclid_polish.training.inference.reconstruct`
        computes it (stretched outputs clipped at ±20 before ``sinh``)."""
        out = self.model(stretched, training=False)
        if self.output == "electrons":
            return out
        head = int(self.head_index or 0)
        if self.output == "heads":
            c = int(self.n_bands)
            out = out[..., head * c:(head + 1) * c]
        knee = float(self.output_knee if self.output_knee is not None else self.knees[head])
        return tf.sinh(tf.clip_by_value(out, -SINH_CLIP, SINH_CLIP)) * knee


def window_side(depth: int) -> int:
    """LR window side for a trunk of ``depth`` residual blocks: at least
    ``MIN_WINDOW_SIDE``, and ``2·(depth + 4)`` so the window covers the
    receptive field (``depth`` 3×3 blocks, the entry and tail 3×3 and the
    5×5 skip reach ``depth + 2`` LR pixels); even."""
    side = max(MIN_WINDOW_SIDE, 2 * (int(depth) + 4))
    return side + side % 2


def _support_side(depth: int) -> int:
    """LR side of the crop the network actually runs on: the receptive field
    plus a margin. A sample's value and gradient only depend on the
    ``depth + 2`` LR pixels around it, so this crop gives the same numbers as
    the whole window at a fraction of the cost."""
    return 2 * (int(depth) + 4)


def stretch(x_e: np.ndarray, knees: Sequence[float]) -> np.ndarray:
    """``asinh(x / q)`` at every knee, knee-major on the last axis: NumPy in
    and out (see :func:`asinh_stretch_multi_knee`)."""
    knees = tuple(float(q) for q in knees)
    return asinh_stretch_multi_knee(tf.constant(np.asarray(x_e, np.float32)), knees).numpy()


# --------------------------------------------------------------------------- #
# pixel classes
# --------------------------------------------------------------------------- #

def noise_sigma(lr_e: np.ndarray, lag: int = NOISE_LAG) -> np.ndarray:
    """Per-band noise σ (electrons) of an ``(h, w, C)`` LR field: the robust
    spread of pixel differences ``lag`` pixels apart, ``1.4826·MAD(Δ)/√2``
    over both axes, leaving out pairs touching a zero (a blanked hole) or a
    non-finite pixel. The lag (:data:`NOISE_LAG`) steps past the noise's
    correlation length, so the result is the per-pixel σ even for resampled
    bands; structure barely moves it (most pixels are sky). A band without a
    usable σ takes the median of the others (1 when none has one)."""
    lr = np.asarray(lr_e, np.float64)
    if lr.ndim != 3:
        raise ValueError(f"the LR field must be (h, w, C), got {lr.shape}")
    sigma = np.full(lr.shape[2], np.nan)
    for c in range(lr.shape[2]):
        plane = lr[..., c]
        good = np.isfinite(plane) & (plane != 0)
        k = max(1, min(int(lag), plane.shape[0] - 1, plane.shape[1] - 1))
        diffs = [(plane[:, k:] - plane[:, :-k])[good[:, k:] & good[:, :-k]],
                 (plane[k:, :] - plane[:-k, :])[good[k:, :] & good[:-k, :]]]
        d = np.concatenate(diffs)
        if d.size >= 16:
            sigma[c] = 1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2.0)
    ok = np.isfinite(sigma) & (sigma > 0)
    fallback = float(np.median(sigma[ok])) if ok.any() else 1.0
    return np.where(ok, sigma, fallback).astype(np.float32)


def _lr_under(lr_e: np.ndarray, shape: tuple[int, ...], scale: int) -> np.ndarray:
    """The LR value under every HR pixel of an ``(H, W, C)`` image."""
    lr = np.asarray(lr_e, np.float32)
    h, w, c = shape
    if lr.ndim != 3 or lr.shape[2] != c:
        raise ValueError(f"the LR field {lr.shape} does not have the HR image's {c} bands")
    rows, cols = np.arange(h) // int(scale), np.arange(w) // int(scale)
    if (h and rows[-1] >= lr.shape[0]) or (w and cols[-1] >= lr.shape[1]):
        raise ValueError(f"the LR field {lr.shape} does not cover the HR image {shape} "
                         f"at scale {scale}")
    return lr[rows[:, None], cols[None, :], :]


def predicted_classes(sr_e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bin an ``(H, W, C)`` SR image (electrons) into :data:`PRED_CLASSES`:
    ``(codes int8, value_e float32)``, the value being the SR itself.
    Non-finite pixels count as sky."""
    sr = np.asarray(sr_e, np.float32)
    if sr.ndim != 3:
        raise ValueError(f"the SR image must be (H, W, C), got {sr.shape}")
    codes = np.digitize(sr, BRIGHTNESS_EDGES_E).astype(np.int8)
    codes[~np.isfinite(sr)] = 0
    return codes, sr


def truth_classes(galaxy_e: np.ndarray, star_e: np.ndarray, lr_e: np.ndarray,
                  scale: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Class every HR pixel and band of a synthetic field into
    :data:`TRUTH_CLASSES` from its truth components (``(H, W, C)`` electrons,
    already blurred to the target PSF) and the LR field ``(H/scale, W/scale,
    C)``: ``(codes int8, value_e float32)``, the value being the truth
    ``galaxy + star``. First match wins:

    1. saturated: the LR pixel under it is exactly 0 and truth > 52 e⁻;
    2. star core: star > ½ truth and truth > 52;
    3. star wing: star > ½ truth and 2 < truth ≤ 52;
    4. galaxy core: truth > 52;
    5. galaxy outskirts: 2 < truth ≤ 52;
    6. faint: 0.01 < truth ≤ 2;
    7. sky.

    (2 and 52 are the predicted-bin edges, :data:`BRIGHTNESS_EDGES_E`.)
    """
    galaxy = np.asarray(galaxy_e, np.float32)
    star = np.asarray(star_e, np.float32)
    if galaxy.ndim != 3 or galaxy.shape != star.shape:
        raise ValueError(f"galaxy {galaxy.shape} and star {star.shape} must be one (H, W, C) shape")
    truth = galaxy + star
    lr = _lr_under(lr_e, truth.shape, scale)
    low, high = BRIGHTNESS_EDGES_E[0], BRIGHTNESS_EDGES_E[2]
    bright = truth > high
    mid = (truth > low) & ~bright
    starry = star > 0.5 * truth
    conditions = [bright & (lr == 0), starry & bright, starry & mid, bright, mid,
                  truth > FAINT_FLOOR_E]
    codes = np.select(conditions, list(range(len(conditions))), default=len(conditions))
    return codes.astype(np.int8), truth


def real_classes(sr_e: np.ndarray, lr_e: np.ndarray, scale: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Class a real tile's SR into :data:`REAL_CLASSES`: the predicted bins,
    and "blank" where the LR pixel under it is 0 or non-finite in that band.
    ``(codes int8, value_e float32)``, the value being the SR."""
    codes, value = predicted_classes(sr_e)
    lr = _lr_under(lr_e, codes.shape, scale)
    codes[(lr == 0) | ~np.isfinite(lr)] = _BLANK
    return codes, value


def _picked(codes: np.ndarray, bands: list, picks: list, weights: list) -> dict[str, np.ndarray]:
    flat_index = np.concatenate(picks) if picks else np.zeros(0, np.int64)
    y, x = np.divmod(flat_index, codes.shape[1])
    band = (np.concatenate(bands) if bands else np.zeros(0)).astype(np.int16)
    return {"band": band, "y": y.astype(np.int32), "x": x.astype(np.int32),
            "code": codes[y, x, band].astype(np.int16) if band.size else np.zeros(0, np.int16),
            "weight": (np.concatenate(weights) if weights else np.zeros(0)).astype(np.float64)}


def sample_pixels(codes: np.ndarray, per_class: int,
                  rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Stratified sample of an ``(H, W, C)`` class map: per band, up to
    ``per_class`` distinct random pixels of every class present.
    ``{"band": int16, "y": int32, "x": int32, "code": int16, "weight":
    float64}``, band-major, classes ascending within a band; ``weight`` is
    the number of the band's pixels each sample stands for (its class's
    pixels over the class's samples), so weighted sums count pixels as a
    uniform sample would."""
    codes = np.asarray(codes)
    if codes.ndim != 3:
        raise ValueError(f"the class map must be (H, W, C), got {codes.shape}")
    if int(per_class) < 1:
        raise ValueError(f"per_class must be at least 1, got {per_class!r}")
    bands, picks, weights = [], [], []
    for band in range(codes.shape[2]):
        flat = codes[..., band].ravel()
        for code in np.unique(flat):
            where = np.flatnonzero(flat == code)
            take = where[rng.choice(where.size, size=min(int(per_class), where.size), replace=False)]
            picks.append(take)
            bands.append(np.full(take.size, band))
            weights.append(np.full(take.size, where.size / take.size))
    return _picked(codes, bands, picks, weights)


def sample_uniform(codes: np.ndarray, per_band: int,
                   rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Uniform sample of an ``(H, W, C)`` class map: per band, ``per_band``
    distinct random pixels (all of them when the band has fewer), whatever
    their class, so the classes come in as often as they cover the field.
    Same keys as :func:`sample_pixels`; ``weight`` is ``H·W / per_band``."""
    codes = np.asarray(codes)
    if codes.ndim != 3:
        raise ValueError(f"the class map must be (H, W, C), got {codes.shape}")
    if int(per_band) < 1:
        raise ValueError(f"per_band must be at least 1, got {per_band!r}")
    size = codes.shape[0] * codes.shape[1]
    take = min(int(per_band), size)
    bands, picks, weights = [], [], []
    for band in range(codes.shape[2]):
        picks.append(np.sort(rng.choice(size, size=take, replace=False)))
        bands.append(np.full(take, band))
        weights.append(np.full(take, size / max(take, 1)))
    return _picked(codes, bands, picks, weights)


def sample_mixed(codes: np.ndarray, per_class: int, per_band: int,
                 rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Both designs pooled: :func:`sample_uniform` (``per_band``) then
    :func:`sample_pixels` (``per_class``), each weight halved. Either design
    alone counts the pixels without bias, so their mean does too, while the
    stratified half keeps rare classes (cores, saturated pixels) sampled. A
    pixel may come twice, once per design."""
    uniform = sample_uniform(codes, per_band, rng)
    stratified = sample_pixels(codes, per_class, rng)
    out = {key: np.concatenate([uniform[key], stratified[key]]) for key in uniform}
    out["weight"] = 0.5 * out["weight"]
    return out


# --------------------------------------------------------------------------- #
# attribution
# --------------------------------------------------------------------------- #

def _knee_terms(g: tf.Tensor, x_e: tf.Tensor, knees: Sequence[float], n_bands: int) -> tf.Tensor:
    """``t_{k,c} = g_{k,c} / √(q_k² + x_c²)``: the stretched-input gradient
    ``(B, h, w, K·C)`` and the electrons ``(B, h, w, C)`` → ``(B, h, w, K, C)``."""
    shape = tf.shape(x_e)
    g = tf.reshape(g, [shape[0], shape[1], shape[2], len(knees), n_bands])
    q = tf.constant(knees, tf.float32)[:, None]
    return g / tf.sqrt(q * q + tf.square(x_e)[:, :, :, None, :])


def _values_and_terms(net: KneeNet, x_e: tf.Tensor, idx: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """One tape over a batch of windows: ``(values (B,), t (B, h, w, K, C))``.
    ``idx`` rows are ``[window, y, x, band]``; the objective is the sum of each
    window's own sample, and the windows never mix, so each window's gradient
    is its sample's alone."""
    s = asinh_stretch_multi_knee(x_e, net.knees)
    with tf.GradientTape() as tape:
        tape.watch(s)
        values = tf.gather_nd(net.predict_electrons(s), idx)
        total = tf.reduce_sum(values)
    g = tf.convert_to_tensor(tape.gradient(total, s))
    return values, _knee_terms(g, x_e, net.knees, net.n_bands)


def _sample_index(y, x, band) -> np.ndarray:
    y, x, band = (np.asarray(v, np.int64).reshape(-1) for v in (y, x, band))
    return np.stack([np.arange(y.size), y, x, band], axis=1).astype(np.int32)


def _terms(net: KneeNet, windows_e, y, x, band) -> tuple[tf.Tensor, tf.Tensor]:
    """The knee terms of one sample per window (eager): ``windows_e`` ``(B, h,
    w, C)`` electrons, ``y, x`` the window-local HR pixel and ``band`` the
    output band of each → ``(values (B,), t (B, h, w, K, C))``."""
    return _values_and_terms(net, tf.convert_to_tensor(windows_e, tf.float32),
                             tf.constant(_sample_index(y, x, band)))


def _compiled(net: KneeNet) -> tuple[Any, Any]:
    """``net``'s compiled passes: the gradient pass ``(x_e, idx, centre) →
    (values, Σ_u |t| (B, K, C), the same Σ over the LR pixels within
    :data:`LOCAL_RADIUS` of ``centre`` (B, 2: the sample's crop-local LR
    pixel))`` and the ablation pass ``(x_e, means, idx, k) → values`` with
    knee ``k``'s channels set to ``means`` (B, K·C)."""
    if "passes" not in net._fns:
        n_channels = len(net.knees) * net.n_bands
        x_spec = tf.TensorSpec([None, None, None, net.n_bands], tf.float32)
        idx_spec = tf.TensorSpec([None, 4], tf.int32)

        @tf.function(input_signature=[x_spec, idx_spec, tf.TensorSpec([None, 2], tf.int32)])
        def gradient_pass(x_e, idx, centre):
            values, t = _values_and_terms(net, x_e, idx)
            size = tf.abs(t)
            shape = tf.shape(x_e)
            near_y = tf.abs(tf.range(shape[1])[None, :] - centre[:, :1]) <= LOCAL_RADIUS
            near_x = tf.abs(tf.range(shape[2])[None, :] - centre[:, 1:]) <= LOCAL_RADIUS
            near = tf.cast(near_y[:, :, None] & near_x[:, None, :], size.dtype)   # (B, h, w)
            return (values, tf.reduce_sum(size, axis=[1, 2]),
                    tf.reduce_sum(size * near[:, :, :, None, None], axis=[1, 2]))

        @tf.function(input_signature=[x_spec, tf.TensorSpec([None, n_channels], tf.float32), idx_spec,
                                      tf.TensorSpec([], tf.int32)])
        def ablation_pass(x_e, means, idx, k):
            s = asinh_stretch_multi_knee(x_e, net.knees)
            knee_of_channel = tf.range(n_channels) // net.n_bands
            ablated = tf.where(tf.equal(knee_of_channel, k), means[:, None, None, :], s)
            return tf.gather_nd(net.predict_electrons(ablated), idx)

        net._fns["passes"] = (gradient_pass, ablation_pass)
    return net._fns["passes"]


def _window_starts(centre: np.ndarray, side: int, size: int) -> tuple[np.ndarray, int]:
    """Start of a ``side`` window centred on ``centre`` and moved inside
    ``[0, size)`` (the whole axis when it is shorter): ``(starts, side)``."""
    side = min(int(side), int(size))
    return np.clip(centre - side // 2, 0, size - side), side


def _crops(lr: np.ndarray, y0: np.ndarray, x0: np.ndarray, h: int, w: int) -> np.ndarray:
    rows = y0[:, None] + np.arange(h)
    cols = x0[:, None] + np.arange(w)
    return lr[rows[:, :, None], cols[:, None, :]]


def _shares(parts: np.ndarray, total: np.ndarray) -> np.ndarray:
    """``parts / total`` per row; 0s where the total is 0 or not finite."""
    ok = np.isfinite(total) & (total > 0)
    return np.where(ok[:, None], parts / np.where(ok, total, 1.0)[:, None], 0.0).astype(np.float32)


def _split_into(out: dict[str, np.ndarray], suffix: str, rows: slice, sums: np.ndarray) -> np.ndarray:
    """Write one batch's knee, band and per-input shares of the σ-weighted
    ``sums`` (B, K, C) under ``suffix``; returns the totals (B,)."""
    n_knees = sums.shape[1]
    total = sums.sum(axis=(1, 2))
    out["knee_share" + suffix][rows] = _shares(sums.sum(axis=2), total)
    out["band_share" + suffix][rows] = _shares(sums.sum(axis=1), total)
    per_input = sums.transpose(0, 2, 1)                                  # (B, C, K)
    out["knee_share_by_input" + suffix][rows] = _shares(
        per_input.reshape(-1, n_knees), per_input.sum(axis=2).reshape(-1)).reshape(per_input.shape)
    return total


def attribute(net: KneeNet, lr_e: np.ndarray, samples: Mapping[str, np.ndarray], *,
              batch: int = 64, ablation: bool = True,
              sigma_e: Sequence[float] | None = None) -> dict[str, np.ndarray]:
    """Knee and band shares of every sampled pixel of one field.

    ``lr_e`` is the LR field ``(h, w, C)`` in electrons (non-finite pixels
    are read as 0); ``samples`` holds HR ``y``, ``x`` and the output
    ``band`` (as :func:`sample_pixels` returns); ``sigma_e`` is each input
    band's noise σ, the weight of its terms (default: the field's own,
    :func:`noise_sigma`). Returns float32 ``knee_share (N, K)``,
    ``band_share (N, C)``, ``knee_share_by_input (N, C, K)`` (each input
    band's own knee split), ``sensitivity (N,)`` (output electrons per input
    σ) and, with ``ablation``, ``ablation_share (N, K)``; shares sum to 1
    (all 0 when the gradient, or every ablation change, is 0). The gradient
    splits come twice: over the receptive field (the keys above) and over
    the local block (``knee_share_local``, ``band_share_local``,
    ``knee_share_by_input_local``).

    Each sample is computed on LR crops around its LR pixel, kept inside the
    field: a crop edge on the field edge is the full field's own ``same``
    padding, and every other edge is beyond the receptive field, so values
    and gradients equal the full field's everywhere. The ablation's window
    mean is over the :func:`window_side` window. Samples run ``batch``
    windows at a time: one gradient pass, then one pass per knee.
    """
    lr = np.asarray(lr_e, np.float32)
    if lr.ndim != 3 or lr.shape[2] != net.n_bands:
        raise ValueError(f"the LR field must be (h, w, {net.n_bands}) bands, got {lr.shape}")
    lr = np.where(np.isfinite(lr), lr, np.float32(0.0))
    sigma = np.asarray(noise_sigma(lr) if sigma_e is None else sigma_e, np.float64).reshape(-1)
    if sigma.size != net.n_bands:
        raise ValueError(f"sigma_e needs one σ per input band ({net.n_bands}), got {sigma.size}")
    band, y, x = (np.asarray(samples[key], np.int64).reshape(-1) for key in ("band", "y", "x"))
    n, n_knees = band.size, len(net.knees)
    out = {"sensitivity": np.zeros(n, np.float32)}
    for suffix, _ in _SPLITS:
        out["knee_share" + suffix] = np.zeros((n, n_knees), np.float32)
        out["band_share" + suffix] = np.zeros((n, net.n_bands), np.float32)
        out["knee_share_by_input" + suffix] = np.zeros((n, net.n_bands, n_knees), np.float32)
    if ablation:
        out["ablation_share"] = np.zeros((n, n_knees), np.float32)
    if n == 0:
        return out
    h, w, scale = lr.shape[0], lr.shape[1], int(net.scale)
    if ((y < 0) | (y >= h * scale) | (x < 0) | (x >= w * scale)
            | (band < 0) | (band >= net.n_bands)).any():
        raise ValueError(f"a sample lies outside the {h * scale}×{w * scale}×{net.n_bands} SR field")
    cy, cx = y // scale, x // scale
    support = _support_side(net.depth)
    sy0, sh = _window_starts(cy, support, h)
    sx0, sw = _window_starts(cx, support, w)
    context = window_side(net.depth)
    wy0, wh = _window_starts(cy, context, h)
    wx0, ww = _window_starts(cx, context, w)
    gradient_pass, ablation_pass = _compiled(net)
    for start in range(0, n, max(1, int(batch))):
        rows = slice(start, start + max(1, int(batch)))
        x_e = tf.constant(_crops(lr, sy0[rows], sx0[rows], sh, sw))
        idx = tf.constant(_sample_index(y[rows] - scale * sy0[rows], x[rows] - scale * sx0[rows],
                                        band[rows]))
        centre = tf.constant(np.stack([cy[rows] - sy0[rows], cx[rows] - sx0[rows]], axis=1), tf.int32)
        values, *sums = gradient_pass(x_e, idx, centre)
        for (suffix, _), block in zip(_SPLITS, sums, strict=True):
            block = block.numpy().astype(np.float64) * sigma[None, None, :]   # (B, K, C), per input σ
            total = _split_into(out, suffix, rows, block)
            if not suffix:
                out["sensitivity"][rows] = np.where(np.isfinite(total), total, 0.0)
        if not ablation:
            continue
        window = asinh_stretch_multi_knee(tf.constant(_crops(lr, wy0[rows], wx0[rows], wh, ww)),
                                          net.knees)
        # float64 mean: a constant block's mean is then exactly the constant.
        means = tf.cast(tf.reduce_mean(tf.cast(window, tf.float64), axis=[1, 2]), tf.float32)
        base = values.numpy().astype(np.float64)
        deltas = np.stack([np.abs(base - ablation_pass(x_e, means, idx, tf.constant(k)).numpy())
                           for k in range(n_knees)], axis=1)
        # A change within float32 rounding of the value is no change.
        deltas[deltas <= _ROUNDING * np.abs(base)[:, None]] = 0.0
        out["ablation_share"][rows] = _shares(deltas, deltas.sum(axis=1))
    return out


# --------------------------------------------------------------------------- #
# aggregates
# --------------------------------------------------------------------------- #

def _column(parts: Sequence[Mapping[str, Any]], key: str, width: int | None) -> np.ndarray:
    arrays = [np.asarray(p[key], np.float64 if width is not None or key == "value_e" else np.int64)
              for p in parts]
    if width is not None:
        arrays = [a.reshape(-1, width) for a in arrays]
    if arrays:
        return np.concatenate(arrays)
    return np.zeros((0, width) if width is not None else 0)


def _mean(rows: np.ndarray) -> list[float]:
    return [float(v) for v in rows.mean(axis=0)]


def _matrix(block: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in block]


def aggregate(parts: Sequence[Mapping[str, Any]], class_names: Sequence[str], n_bands: int,
              knees: Sequence[float]) -> dict[str, Any]:
    """Summarise per-field samples (each part: ``band``, ``code``,
    ``value_e``, ``knee_share``, ``band_share``, ``ablation_share`` and
    optionally ``knee_share_by_input`` and the local splits ``*_local``, one
    row per sample) into a JSON-able record, per output band ``"0"``…:

    * ``by_class[band][class]``: mean ``knee_share``, ``ablation_share``
      (``None`` when no part carries it), ``band_share``,
      ``knee_share_by_input`` (C × K, ``None`` when no part carries it), the
      same three ``*_local`` (``None`` when not carried) and ``n``; classes
      without samples are left out;
    * ``by_brightness``: the :data:`LOG_EDGES_E` edges and, per band, the
      mean ``knee_share`` (and ``knee_share_by_input``, C × K,
      ``knee_share_local`` and ``knee_share_by_input_local``, when carried)
      and ``n`` per half-decade bin of ``value_e`` (``None`` and 0 for an
      empty bin; values below the first edge go to the first bin, values
      from the last edge on to the last, non-finite values nowhere);
    * ``cross_band`` (and ``cross_band_local`` when carried): per output
      band the mean ``band_share`` over input bands (``None`` for a band
      without samples);
    * ``n_samples``.
    """
    n_knees = len(knees)
    parts = [p for p in parts if len(p["band"])]
    band = _column(parts, "band", None)
    code = _column(parts, "code", None)
    value = _column(parts, "value_e", None)
    shapes = {"knee_share": (n_knees,), "ablation_share": (n_knees,), "band_share": (n_bands,),
              "knee_share_by_input": (n_bands, n_knees), "knee_share_local": (n_knees,),
              "band_share_local": (n_bands,), "knee_share_by_input_local": (n_bands, n_knees)}
    # Every key every part carries (knee_share and band_share always).
    rows = {key: _column(parts, key, int(np.prod(shape))).reshape((-1, *shape))
            for key, shape in shapes.items()
            if key in ("knee_share", "band_share") or (parts and all(key in p for p in parts))}

    def mean(key: str, sel: np.ndarray):
        if key not in rows:
            return None
        block = rows[key][sel].mean(axis=0)
        return _matrix(block) if block.ndim == 2 else [float(v) for v in block]

    by_class: dict[str, dict[str, Any]] = {}
    for b in range(n_bands):
        entry: dict[str, Any] = {}
        for c, name in enumerate(class_names):
            sel = (band == b) & (code == c)
            if not sel.any():
                continue
            entry[name] = {key: mean(key, sel) for key in shapes} | {"n": int(sel.sum())}
        by_class[str(b)] = entry

    n_bins = len(LOG_EDGES_E) - 1
    finite = np.isfinite(value)
    bins = np.clip(np.searchsorted(LOG_EDGES_E, np.where(finite, value, 0.0), side="right") - 1,
                   0, n_bins - 1)
    bands: dict[str, Any] = {}
    for b in range(n_bands):
        sel = (band == b) & finite
        counts = np.bincount(bins[sel], minlength=n_bins)
        bands[str(b)] = {}
        for key in ("knee_share", "knee_share_by_input", "knee_share_local", "knee_share_by_input_local"):
            if key not in rows:
                continue
            sums = np.zeros((n_bins, *shapes[key]))
            np.add.at(sums, bins[sel], rows[key][sel])
            bands[str(b)][key] = [(_matrix(sums[i] / counts[i]) if sums.ndim == 3
                                   else [float(v) for v in sums[i] / counts[i]]) if counts[i] else None
                                  for i in range(n_bins)]
        bands[str(b)]["n"] = [int(v) for v in counts]

    out = {"by_class": by_class,
           "by_brightness": {"edges_e": [float(v) for v in LOG_EDGES_E], "bands": bands}}
    for suffix, _ in _SPLITS:
        if "band_share" + suffix in rows:
            cross = rows["band_share" + suffix]
            out["cross_band" + suffix] = [_mean(cross[band == b]) if (band == b).any() else None
                                          for b in range(n_bands)]
    out["n_samples"] = int(band.size)
    return out
