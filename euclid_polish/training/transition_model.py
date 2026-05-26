"""Tiny learned HST → Euclid PSF-transition model A_θ.

Replaces the analytic Wiener-deconvolution kernel from
:mod:`euclid_polish.sky.differential_kernel` with a small CNN trained to
satisfy ``A_θ(scene ⊛ PSF_HST) ≈ scene ⊛ PSF_Euclid`` on synthetic
clean scenes.

Why a CNN instead of the analytic kernel
----------------------------------------

The Wiener inverse ``Ê · conj(Ĥ) / (|Ĥ|² + reg²)`` only sees the two
PSF images. The angular spike mismatch between HST's 4-vane and
Euclid's 6-vane diffraction pattern is a physical incompatibility no
linear kernel can resolve cleanly — it can only pick a Fourier-domain
compromise that leaks into image space as a checkerboard or rings.

A learned CNN is trained on real image distributions instead, picking a
compromise that's image-space-minimal. The same fundamental mismatch
still exists, but its residual gets distributed across the image
without the oscillatory artefacts.

Architecture (≤5k parameters)
-----------------------------

Plain stack of 3×3 convolutions with ReLU, residual wrap:

    A_θ(x) = x + f(x)
    where f = Conv3×3·1→C → ReLU → Conv3×3·C→C → ReLU → ⋯ → Conv3×3·C→1

For ``C=12`` and 5 layers the param count is (no biases — see
``use_bias=False`` rationale in ``__init__``)::

    (3·3·1·C) + 3·(3·3·C·C) + (3·3·C·1)
    = 9C + 27C² + 9C
    = 27C² + 18C
    = 4104  (for C=12)
    = 4797  (for C=13)
    = 5544  (for C=14 — exceeds 5k budget)

Residual structure means at init ``A_θ(x) = x`` (since ``f`` weights
are sampled from N(0, small)). Training pushes ``f`` to learn the
*difference* between HST-PSF-blurred and Euclid-PSF-blurred — that
difference is small for our case (both PSFs ~0.1″ FWHM order, mainly
differing in the wings), so the residual framing is well-conditioned.

No batch norm — breaks translation equivariance, which we need
(PSF convolution is translation-equivariant by construction).

No padding tricks beyond ``padding='same'`` with zeros — the receptive
field is small (5 × 2 = 11 px ≈ 0.55″ at 0.05″/pix) and PSF supports
are 0.1–0.2″ FWHM, so a pixel near the array boundary can still see
enough of the PSF core to do the transition.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import tensorflow as tf

from euclid_polish.config import Config


# Default model location — mirrors the analytic ``diff_kernel_VIS.fits``
# slot so callers can swap one for the other without path gymnastics.
DEFAULT_MODEL_PATH = os.path.join(
    Config.DATA_DIR, "hst_psf", "transition_model.weights.h5",
)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class HSTtoEuclidTransition(tf.keras.Model):
    """Learned A_θ: HST-PSF-blurred → Euclid-PSF-blurred (single band).

    Operates per-channel on inputs of shape ``[B, H, W, 1]`` (linear
    electron units; **no** asinh stretch — this model deals with image
    formation, not perceptual scaling). Output has the same shape and
    units.

    Receptive field grows by 2 per 3×3 conv. With 5 layers the central
    pixel sees a 11×11 window — wider than the PSF cores (which sit
    inside ~7 HR pixels) at the HR pixel scale.

    Parameters
    ----------
    channels
        Hidden channel width ``C``. Default 12 → 4153 params, well
        inside the 5k budget. ``C=13`` is 4850 (also fits); ``C=14``
        spills over.
    n_inner_layers
        Number of full ``Conv(C→C)`` middle layers between the
        first ``Conv(1→C)`` and the final ``Conv(C→1)``. Default 3
        gives 5 total convs and the 11-pixel receptive field; bumping
        this widens the receptive field at quadratic param cost.
    kernel_size
        Spatial kernel side. Stays at 3 unless you're deliberately
        widening the receptive field for one of the wide-band tests.
    name
        Keras model name; defaults to ``"hst_to_euclid_transition"``.

    Notes
    -----
    The model is deterministic at inference (no dropout, no batchnorm).
    Training adds no noise to either inputs or targets — see the
    accompanying ``scripts/fasrc_train_transition_model.py`` for the
    "clean → clean" objective rationale.
    """

    def __init__(
        self,
        channels: int = 12,
        n_inner_layers: int = 3,
        kernel_size: int = 3,
        baseline_kernel: Optional[np.ndarray] = None,
        name: str = "hst_to_euclid_transition",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        if channels < 1:
            raise ValueError(f"channels must be ≥ 1, got {channels}")
        if n_inner_layers < 0:
            raise ValueError(
                f"n_inner_layers must be ≥ 0, got {n_inner_layers}"
            )
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError(
                f"kernel_size must be odd and ≥ 1, got {kernel_size}"
            )
        self._channels       = int(channels)
        self._n_inner_layers = int(n_inner_layers)
        self._kernel_size    = int(kernel_size)

        # ── Optional analytic-baseline residual. ──────────────────────
        #
        # When ``baseline_kernel`` is provided, the forward pass becomes
        # ``A_θ(x) = (baseline_kernel ⊛ x) + f(x)`` instead of
        # ``A_θ(x) = x + f(x)``. The kernel is pinned as a NON-TRAINABLE
        # tf.Variable so it serialises with model.save_weights but
        # doesn't move under SGD. Typical use: pass the
        # ``diff_kernel_VIS.fits`` (analytic Wiener inverse) — A_θ
        # then starts at "approximately correct, modulo known Wiener
        # artifacts" and only has to learn the artifact-correction
        # residual, which is small + local + well-suited to a tiny CNN.
        #
        # Centre-crop the kernel BEFORE constructing the model: tf.nn.
        # conv2d scales linearly with kernel area and a 511² baseline
        # makes every forward pass painfully slow. Most of the kernel's
        # energy is concentrated in the central few pixels; 31-51 px is
        # usually plenty.
        self._has_baseline = baseline_kernel is not None
        if self._has_baseline:
            if baseline_kernel.ndim != 2:
                raise ValueError(
                    f"baseline_kernel must be 2-D, got "
                    f"{baseline_kernel.shape}"
                )
            if (baseline_kernel.shape[0] != baseline_kernel.shape[1]
                    or baseline_kernel.shape[0] % 2 == 0):
                raise ValueError(
                    f"baseline_kernel must be square + odd-sided; got "
                    f"{baseline_kernel.shape}"
                )
            # Flip for tf.nn.conv2d (cross-correlation → true convolution),
            # matching the scipy.signal.fftconvolve convention used at
            # pair-gen + analytic-kernel-validation time.
            k = baseline_kernel[::-1, ::-1].astype(np.float32, copy=True)
            kernel = k[:, :, np.newaxis, np.newaxis]
            self._baseline_kernel = self.add_weight(
                name="baseline_kernel",
                shape=kernel.shape,
                initializer=tf.constant_initializer(kernel),
                trainable=False,
                dtype=tf.float32,
            )
            self._baseline_kernel_side = int(baseline_kernel.shape[0])
        else:
            self._baseline_kernel = None
            self._baseline_kernel_side = 0

        # Small init so residual ≈ identity at start. He/Glorot would
        # also work but pushes f away from zero, breaking the "tiny
        # correction" framing the residual gives us.
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)

        # ── ``use_bias=False`` is load-bearing for scale invariance. ──
        #
        # A PSF transition operator MUST satisfy ``A_θ(α·x) = α·A_θ(x)``
        # for any scalar α ≥ 0: convolution is linear, and the training
        # objective ``clean ⊛ PSF_HST → clean ⊛ PSF_Euclid`` is
        # homogeneous in the scene amplitude. The synthetic clean scenes
        # live at electron scale (~100–1000s of e⁻ per pixel); the
        # PSF-identity diagnostic feeds a unit-flux PSF (peak ~10⁻⁴).
        # If we let the Conv2D layers learn biases, SGD drives them
        # away from zero to fit the electron-scale training distribution
        # — biases then dominate the response on small-scale inputs
        # like the PSF, and ``A_θ(PSF_HST)`` comes out at electron scale
        # (10³× larger than ``PSF_Euclid``). Empirically: ``PSF_id_err``
        # explodes from 0.35 (at init, A_θ ≈ identity) to 1300+ within
        # 500 training steps while ``val_L1`` stays small — the model
        # is fitting the data distribution at the cost of the
        # operator-theoretic correctness we care about.
        #
        # Without biases, the conv stack is *purely linear* (ReLU is
        # the identity on positive inputs, and clean⊛PSF ≥ 0 always).
        # ``A_θ(α·x) = α·A_θ(x)`` holds by construction — the PSF
        # identity probe is then bounded by the training L1 loss
        # itself, not by some unrelated bias drift.
        common_kwargs = dict(
            padding="same",
            kernel_initializer=init,
            use_bias=False,
        )
        self._first = tf.keras.layers.Conv2D(
            filters=self._channels, kernel_size=self._kernel_size,
            activation="relu", name="conv_in", **common_kwargs,
        )
        self._inner = [
            tf.keras.layers.Conv2D(
                filters=self._channels, kernel_size=self._kernel_size,
                activation="relu", name=f"conv_inner_{i}",
                **common_kwargs,
            )
            for i in range(self._n_inner_layers)
        ]
        # Final layer: no activation. Same use_bias=False — the residual
        # ``f(x)`` is then a pure linear function of x (positive-input
        # regime), so ``A_θ(x) = x + f(x)`` stays homogeneous: doubling
        # the input doubles the output.
        self._last = tf.keras.layers.Conv2D(
            filters=1, kernel_size=self._kernel_size,
            activation=None, name="conv_out", **common_kwargs,
        )

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def receptive_field(self) -> int:
        """RF side length at the HR grid (= 2 · n_total_layers + 1)."""
        n_layers = 1 + self._n_inner_layers + 1
        # Each kernel_size=k conv adds (k-1) to the RF.
        return 1 + n_layers * (self._kernel_size - 1)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            ``[B, H, W, 1]`` float32 in linear electron units.
        training
            Unused — model is deterministic at both train and eval
            (no dropout / no batchnorm).

        Returns
        -------
        ``[B, H, W, 1]`` Euclid-PSF-blurred (same shape, same units).
        """
        # Two residual modes:
        #
        #   * ``baseline_kernel=None`` (default): the conv stack ``f``
        #     sees the raw input ``x`` and produces a correction added
        #     back to ``x``:
        #         A_θ(x) = x + f(x)
        #     At init ``f ≈ 0`` so ``A_θ ≈ Identity``. Model learns
        #     the entire PSF transition from scratch.
        #
        #   * ``baseline_kernel`` given (e.g. the analytic Wiener
        #     differential kernel): first apply the kernel to get a
        #     "rough draft" of the Euclid-blurred output (which has
        #     known Wiener artifacts — checkerboard, FFT-boundary
        #     rings, spike-mismatch residuals), then feed THAT into
        #     the CNN ``f``. The CNN's job becomes "clean up the
        #     analytic kernel's artifacts", which it can do well
        #     because the artifacts are visible in its input.
        #         baseline = A_analytic ⊛ x
        #         A_θ(x) = (1 + f)(baseline) = baseline + f(baseline)
        #     At init ``f ≈ 0`` so ``A_θ(x) ≈ A_analytic ⊛ x`` — the
        #     model starts at "approximately correct" and only learns
        #     the correction.
        if self._has_baseline:
            baseline = tf.nn.conv2d(
                x, self._baseline_kernel, strides=1, padding="SAME",
            )
            cnn_input = baseline
        else:
            baseline = x
            cnn_input = x

        h = self._first(cnn_input)
        for layer in self._inner:
            h = layer(h)
        residual = self._last(h)
        return baseline + residual

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "channels":       self._channels,
            "n_inner_layers": self._n_inner_layers,
            "kernel_size":    self._kernel_size,
        })
        return cfg


# ---------------------------------------------------------------------------
# Helpers — counting params, saving / loading weights
# ---------------------------------------------------------------------------

def total_parameter_count(model: HSTtoEuclidTransition) -> int:
    """**Trainable** parameter count, building the model if needed.

    Keras lazy-builds variables on first call. We materialise them by
    feeding a small dummy input the first time so the count is real.

    Counts only ``trainable_variables`` so the optional non-trainable
    ``baseline_kernel`` (which can be 31² or larger) doesn't blow the
    user-facing param budget. ``model.count_params()`` includes both
    trainable and non-trainable and would falsely report a 31² baseline
    as 961 params on top of the actual learnable ~5k.
    """
    _build_if_needed(model)
    return int(sum(int(np.prod(v.shape)) for v in model.trainable_variables))


def save_model_weights(
    model: HSTtoEuclidTransition,
    path: str = DEFAULT_MODEL_PATH,
) -> str:
    """Save trained weights to a Keras ``.weights.h5`` file.

    Picked the weights-only format (vs full ``model.save``) so the file
    is small (a few KB for a 5k-param model) and the loader doesn't
    have to recompile graph state. The architecture is reconstructed
    by the caller via :class:`HSTtoEuclidTransition`'s constructor.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _build_if_needed(model)
    model.save_weights(path)
    return path


def load_model_weights(
    model: HSTtoEuclidTransition,
    path: str = DEFAULT_MODEL_PATH,
) -> HSTtoEuclidTransition:
    """Load weights into a fresh model instance.

    Caller is responsible for instantiating ``HSTtoEuclidTransition``
    with the *same* hyperparameters used at training time — there's no
    schema header to cross-check against. The Keras loader will raise
    on shape mismatch, so a wrong width gets caught at load time.
    """
    _build_if_needed(model)
    model.load_weights(path)
    return model


def _build_if_needed(model: HSTtoEuclidTransition) -> None:
    """Force Keras to materialise variables.

    Uses a dummy input larger than any baseline kernel so SAME-padding
    convolutions are well-defined even for big baselines. Centralised
    here so save/load/count_params all use the same sizing rule.
    """
    if model.built:
        return
    side = max(8, getattr(model, "_baseline_kernel_side", 0) + 1)
    if side % 2:
        side += 1
    _ = model(tf.zeros((1, side, side, 1), dtype=tf.float32))


# ---------------------------------------------------------------------------
# Pure-numpy inference helper (used by the offline TFRecord generator)
# ---------------------------------------------------------------------------

def apply_transition_numpy(
    model: HSTtoEuclidTransition,
    image: np.ndarray,
    *,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """Run ``model`` on a single ``(H, W)`` or ``(H, W, 1)`` numpy image.

    Convenience wrapper for offline use — the HST→Euclid TFRecord
    generator processes one cutout at a time and doesn't want to build
    a tf.data pipeline. Returns a numpy float32 with the input shape.

    ``batch_size`` is ignored when a single image is passed; reserved
    for a future batched variant.
    """
    if image.ndim == 2:
        squeezed = True
        x = image[np.newaxis, ..., np.newaxis]
    elif image.ndim == 3 and image.shape[-1] == 1:
        squeezed = False
        x = image[np.newaxis, ...]
    else:
        raise ValueError(
            f"apply_transition_numpy expects (H, W) or (H, W, 1); "
            f"got shape {image.shape}"
        )
    x = x.astype(np.float32)
    y = model(x, training=False).numpy()
    if squeezed:
        return y[0, :, :, 0].astype(image.dtype)
    return y[0].astype(image.dtype)
