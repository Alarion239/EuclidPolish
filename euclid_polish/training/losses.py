"""Training losses: the Lp reconstruction-norm family, MSE, and reverse-Huber.

``lp_loss(p)`` returns ``(mean |a − b|^p)^(1/p)`` — the p-NORM, not the raw
p-th-power mean. The root keeps every p on the same scale/units as the L1
default (asinh residuals), so the LR schedule, gradient-spike threshold and
plateau guard need no per-p retuning, and members trained with different p
log comparable loss curves. p=1 is exactly MAE; p=2 is RMSE; p=3 weights the
worst residuals hardest (a sharper penalty on rare large errors — bright
structure and hallucinated features).

``mse_loss`` returns ``mean((a − b)²)`` without the outer square root. Unlike
RMSE, it is a linear mean over pixels and examples, so regrouping the same
residual pixels into full fields or cutouts does not change the scalar
objective when every pixel receives the same weight.

``berhu_loss`` is the reverse Huber (BerHu): L1 on residuals below a threshold
δ, L2 above it — the mirror image of Huber. That is exactly the estimator the
L1-galaxy / L2-star split calls for: robust (median-like) in the extended,
noise-dominated flux where L2 rings, but quadratic on the bright point-source
peaks where L1 leaves stars soft. δ is set adaptively per batch to
``c · max|residual|`` (c=0.2, the Laina+16 / Zwald+12 default), so the loss is
scale-adaptive, parameter-free and sits on the same L1 magnitude scale — no
LR/threshold retuning, like the p-norms.

``build_loss(name)`` is the single dispatcher over all of the above; it is what
the trainer wires up from a member's ``loss_norm`` knob.
"""

from __future__ import annotations

import math

import tensorflow as tf

from euclid_polish.training.augmentation import asinh_stretch_multi_knee, expand_to_knees
from euclid_polish.training.loss_names import (  # noqa: F401  (re-exported)
    BERHU_DEFAULT_C,
    KNEE_LOSS_MODES,
    LOSS_NAMES,
    LOSS_NORMS,
    MSE_NAME,
)


def lp_loss(norm: str = "l1"):
    """The loss callable for a ``LOSS_NORMS`` key; signature ``loss(a, b)``."""
    p = LOSS_NORMS.get(str(norm).lower())
    if p is None:
        raise ValueError(f"unknown loss norm {norm!r}; use one of "
                         f"{sorted(LOSS_NORMS)}")
    if p == 1:
        def _l1(a, b):
            return tf.reduce_mean(tf.abs(a - b))
        return _l1

    def _lp(a, b):
        return tf.reduce_mean(tf.abs(a - b) ** p) ** (1.0 / p)
    return _lp


def mse_loss():
    """Mean squared error callable; signature ``loss(a, b)``."""
    def _mse(a, b):
        return tf.reduce_mean(tf.square(a - b))
    return _mse


def berhu_loss(c: float = BERHU_DEFAULT_C):
    """Reverse-Huber (BerHu) loss; signature ``loss(a, b)``.

    ``B(x) = |x|`` for ``|x| ≤ δ`` and ``(x² + δ²) / 2δ`` for ``|x| > δ``,
    averaged over the batch. δ is a per-batch constant ``c · max|residual|``
    (``stop_gradient`` — δ is a threshold, not a learned quantity), so the
    L1↔L2 crossover tracks the batch's own dynamic range. The two branches
    meet at ``|x| = δ`` in both value and slope, so B is C¹.
    """
    c = float(c)

    def _berhu(a, b):
        d = tf.abs(a - b)
        # δ from the batch max; floored so an all-zero (perfect) batch, where
        # d ≤ δ holds trivially, takes the L1 branch instead of dividing by 0.
        delta = tf.maximum(c * tf.stop_gradient(tf.reduce_max(d)), 1e-8)
        quad = (d * d + delta * delta) / (2.0 * delta)
        return tf.reduce_mean(tf.where(d <= delta, d, quad))
    return _berhu


def build_loss(name: str = "l1"):
    """Reconstruction loss for a ``loss_norm`` knob value."""
    key = str(name).lower()
    if key == "berhu":
        return berhu_loss()
    if key == MSE_NAME:
        return mse_loss()
    if key in LOSS_NORMS:
        return lp_loss(key)
    raise ValueError(f"unknown loss {name!r}; use one of {sorted(LOSS_NAMES)}")


def knee_tilt_weights(knees, tilt: float, n_bands: int) -> list[float]:
    """Per-channel weights of a knee-major tensor (channel ``k·n_bands + c``
    = band ``c`` at knee ``k``) tilting the balanced loss across knees: knee
    ``k`` weighs ``(k / k_ref)**tilt``, ``k_ref`` the geometric mean of the
    knees, normalised to a mean of 1 — so ``tilt`` < 0 puts more weight on
    the small knees (faint light), > 0 on the large ones (bright cores), and
    0 is the plain balanced loss."""
    logs = [math.log(float(q)) for q in knees]
    centre = sum(logs) / len(logs)
    per_knee = [math.exp(float(tilt) * (v - centre)) for v in logs]
    mean = sum(per_knee) / len(per_knee)
    return [w / mean for w in per_knee for _ in range(int(n_bands))]


def knee_channel_weights(knee_weights, n_bands: int) -> list[float]:
    """Per-channel weights of a knee-major tensor (channel ``k·n_bands + c``)
    from one weight per scored knee, repeated for every band and normalised
    to a mean of 1 — e.g. under a loss floor, where every knee far below the
    floor scores the same relative error, ``[3, 1, …]`` on the lowest knee
    stands for the three identical sub-noise channels of 1e-3/1e-2/0.1 e⁻."""
    per_knee = [float(w) for w in knee_weights]
    if not per_knee or min(per_knee) <= 0.0 or not all(math.isfinite(w) for w in per_knee):
        raise ValueError(f"loss knee weights must be positive and finite, got {per_knee}")
    mean = sum(per_knee) / len(per_knee)
    return [w / mean for w in per_knee for _ in range(int(n_bands))]


def channel_balanced_loss(name: str, weights=None):
    """``name``'s loss per output channel, combined by the geometric mean over
    channels (weighted by ``weights``, one per channel, when given — see
    :func:`knee_tilt_weights`); signature ``loss(a, b)``.

    Minimising the geometric mean minimises the mean log error over channels,
    i.e. maximises the mean PSNR over every band and knee. It is the
    parameter-free form of learned uncertainty weighting (Kendall, Gal &
    Cipolla 2018), whose per-output weights converge to 1 / loss. The plain
    loss over all channels is instead dominated by the channels with the
    largest stretched residuals: the lowest knees by orders of magnitude, and
    VIS among the bands. The result keeps the per-channel losses' units, so
    the LR schedule, clipping and spike guard need no retuning.
    """
    base = build_loss(name)
    w = None if weights is None else tf.constant([float(v) for v in weights], tf.float32)

    def _balanced(a, b):
        per_channel = tf.stack([base(a[..., i:i + 1], b[..., i:i + 1])
                                for i in range(int(a.shape[-1]))])
        logs = tf.math.log(tf.maximum(per_channel, 1e-12))
        if w is None:
            return tf.exp(tf.reduce_mean(logs))
        weight = tf.cast(w, logs.dtype)
        if int(weight.shape[0]) != int(per_channel.shape[0]):
            raise ValueError(f"{int(weight.shape[0])} weights for "
                             f"{int(per_channel.shape[0])} channels")
        return tf.exp(tf.reduce_sum(weight * logs) / tf.reduce_sum(weight))
    return _balanced


def knee_expanded_loss(loss, output_knee: float, knees):
    """Loss of a single-image multi-knee member: its one output (stretched at
    ``output_knee``) is re-stretched at every knee and compared, channel for
    channel, with the knee-major multi-knee target by ``loss`` (plain or
    channel-balanced); signature ``loss(sr, hr)``."""
    def _loss(sr, hr):
        return loss(expand_to_knees(sr, output_knee, knees), hr)
    return _loss


def knee_channel_indices(knees, subset, n_bands: int) -> list[int]:
    """Channels of a knee-major multi-knee tensor (channel ``k·n_bands + c``
    = band ``c`` at knee ``k``) that hold the knees in ``subset``."""
    knees = [float(q) for q in knees]
    missing = [float(q) for q in subset if float(q) not in knees]
    if missing:
        raise ValueError(f"loss knees {missing} are not among the input knees {knees}")
    return [knees.index(float(q)) * n_bands + c for q in subset for c in range(n_bands)]


def knee_subset_loss(loss, knees, subset, n_bands: int):
    """``loss(sr, hr)`` on the ``subset`` knees only of a knee-major target
    stretched at every knee in ``knees``: a member can READ all its knees
    and be SCORED at fewer (e.g. 8 input knees, loss at 0.1–10⁴ e⁻)."""
    index = tf.constant(knee_channel_indices(knees, subset, n_bands), tf.int32)

    def _loss(sr, hr):
        return loss(sr, tf.gather(hr, index, axis=-1))
    return _loss


def knee_floored_loss(loss, knees, floor_e, n_bands: int):
    """``loss`` on ``asinh((x + f_b) / k)`` at every knee for a member that
    outputs ELECTRONS, with a per-band floor ``f_b`` (e⁻) inside the stretch,
    applied identically to prediction and target — so it is exactly unbiased
    (a noisy target is not: averaging it in this log-like stretch drags faint
    light toward zero). Below ``f_b`` the stretch is linear, so the loss stops
    rewarding sub-noise structure that tiny knees amplify. The target arrives
    stretched at every knee (``asinh(hr / k)``, knee-major); each knee block's
    electrons are recovered exactly before the floored stretch."""
    floor = [float(v) for v in floor_e]
    if len(floor) != int(n_bands):
        raise ValueError(f"floor_e needs one value per band ({n_bands}), got {floor}")
    if min(floor) < 0.0:
        raise ValueError(f"floor_e must be non-negative, got {floor}")
    floor_t = tf.constant(floor, tf.float32)

    def _loss(sr, hr):
        f = tf.cast(floor_t, sr.dtype)
        pred, target = [], []
        for j, q in enumerate(knees):
            k = tf.cast(float(q), sr.dtype)
            hr_e = k * tf.sinh(hr[..., j * n_bands:(j + 1) * n_bands])
            target.append(tf.asinh((hr_e + f) / k))
            pred.append(tf.asinh((sr + f) / k))
        return loss(tf.concat(pred, axis=-1), tf.concat(target, axis=-1))
    return _loss


def lupton_loss(loss, knees, softening_e, n_bands: int):
    """``loss`` on Lupton's asinh magnitudes (Lupton, Gunn & Szalay 1999):
    ``asinh(x / s_b)`` per band, the softening ``s_b`` (e⁻) at the band's
    per-pixel sky scatter, for a member that outputs ELECTRONS. One stretch
    for every brightness and no knee grid: linear below the noise, where the
    estimate stays unbiased (the posterior mean) and errors count in noise
    units; logarithmic above it, where every surface brightness counts by its
    relative error. All bands are in noise units, so one ``loss`` over them
    needs no balancing. The target arrives stretched at every knee
    (knee-major); its electrons are recovered exactly from the block of the
    knee nearest 1 e⁻."""
    soft = [float(v) for v in softening_e]
    if len(soft) != int(n_bands):
        raise ValueError(f"softening_e needs one value per band ({n_bands}), got {soft}")
    if min(soft) <= 0.0:
        raise ValueError(f"softening_e must be positive, got {soft}")
    if not knees:
        raise ValueError("lupton_loss needs the target's knees to recover its electrons")
    j = min(range(len(knees)), key=lambda i: abs(math.log(float(knees[i]))))
    knee = float(knees[j])
    soft_t = tf.constant(soft, tf.float32)

    def _loss(sr, hr):
        s = tf.cast(soft_t, sr.dtype)
        hr_e = tf.cast(knee, sr.dtype) * tf.sinh(hr[..., j * n_bands:(j + 1) * n_bands])
        return loss(tf.asinh(sr / s), tf.asinh(hr_e / s))
    return _loss


def knee_stretched_loss(loss, knees):
    """Loss of a member that outputs ELECTRONS (a learned output knee): its
    output is stretched at every knee and compared, channel for channel, with
    the knee-major multi-knee target by ``loss`` (plain or channel-balanced);
    signature ``loss(sr, hr)``. With the head's knee at ``k`` this is
    :func:`knee_expanded_loss` at output knee ``k``."""
    def _loss(sr, hr):
        return loss(asinh_stretch_multi_knee(sr, knees), hr)
    return _loss
