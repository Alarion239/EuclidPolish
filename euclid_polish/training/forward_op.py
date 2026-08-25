"""TF-graph forward operator that maps the model's deconvolved sky → an image.

The model emits a single deconvolved VIS sky ``SR``; each non-synthetic
training lane scores ``SR`` by passing it through its instrument's forward
operator *inside* the gradient tape, so the operator has to be a TF-graph op
differentiable with respect to its input. This module provides that as a Keras
layer:

  * Constant (non-trainable) PSF loaded from FITS. ``EuclidVISForwardOp`` uses
    the VIS ePSF (``$DATA_DIR/euclid_psf/euclid_psf_VIS.fits``); the
    :class:`HSTForwardOp` subclass uses the HST F814W PSF with ``rebin_factor=1``.
  * PSF convolution evaluated in the **Fourier domain** (``tf.signal.rfft2d`` /
    ``irfft2d``), matching ``scipy.signal.fftconvolve(..., mode='same')`` — far
    cheaper than a spatial ``conv2d`` for the ~1023² PSF.
  * Stride-N sum-rebin via ``avg_pool2d × N²``, mirroring
    :meth:`ObservationSimulator.sum_rebin`.
  * **No noise.** Noise is a non-differentiable sampling step; the operator is
    defined on the deterministic image formation only.

In the current trainer the operator applied in training is :class:`HSTForwardOp`
(the HST lane, ``H ⊛ SR``). :class:`EuclidVISForwardOp` is retained for the
``/inference`` "forward(SR)" diagnostic panel.

The numerical contract with the numpy path is tested in
``tests/test_forward_op.py``: applying this layer to a known HR input must match
:meth:`ObservationSimulator._process_one_band` for VIS to within float32 round-off
across the bulk of the array (the test crops a sentinel border to stay robust to
boundary effects).
"""

from __future__ import annotations

import os
from typing import cast

import numpy as np
import tensorflow as tf
from astropy.io import fits

from euclid_polish.config import Config


def _default_vis_psf_path() -> str:
    """Same VIS PSF FITS the synthetic forward (and the diff kernel) consume."""
    return os.path.join(Config.EUCLID_PSF_DIR, Config.BAND_VIS.psf_fits_filename)


def _default_hst_psf_path() -> str:
    """HST F814W ePSF FITS — the same file the differential kernel consumes."""
    return Config.HST_PSF_PATH


def _next_pow2(n: tf.Tensor) -> tf.Tensor:
    """Smallest power of two ≥ ``n`` (scalar int tensor → scalar int32).

    FFT throughput is dominated by the transform length's prime
    factorisation; a linear-conv pad of ``H + K − 1`` often lands on an
    ugly size (e.g. 1278 = 2·3²·71). Rounding the transform length up to
    a power of two — cuFFT's fastest case — costs a little extra memory
    but is far faster, and is exact for linear convolution (the extra
    samples are trailing zeros). The ``-1e-6`` keeps an exact power of
    two from rounding up to the next one due to float log round-off.
    """
    n_f = tf.cast(n, tf.float32)
    exp = tf.math.ceil(tf.math.log(n_f) / tf.math.log(2.0) - 1e-6)
    return tf.cast(tf.pow(2.0, exp), tf.int32)


def _load_vis_psf_kernel(
    psf_fits_path: str | None = None, *, crop_half_side: int | None = None,
) -> np.ndarray:
    """Load + unit-normalise the VIS PSF as a 2-D float32 kernel.

    ``crop_half_side`` optionally restricts the kernel to a
    ``(2 * crop_half_side + 1)``-side central crop — a meaningful speed
    win for ``tf.nn.conv2d``, which scales linearly with kernel area
    (a 511×511 kernel on a 256² HR patch is ~256² × 511² ≈ 1.7e10
    multiplications per call, vs ~256² × 65² ≈ 2.8e8 for a 65-side
    crop). Default ``None`` keeps the full saved kernel to stay
    bit-equivalent to the numpy synthetic forward.
    """

    if psf_fits_path is None:
        psf_fits_path = _default_vis_psf_path()
    with fits.open(psf_fits_path) as hdul:
        primary = cast(fits.PrimaryHDU, hdul[0])
        psf = np.asarray(primary.data, dtype=np.float32)
    if psf.ndim != 2:
        raise ValueError(
            f"VIS PSF FITS must be 2-D; got shape {psf.shape}"
        )
    if crop_half_side is not None:
        ch, cw = psf.shape[0] // 2, psf.shape[1] // 2
        psf = psf[ch - crop_half_side : ch + crop_half_side + 1,
                  cw - crop_half_side : cw + crop_half_side + 1]
    s = float(psf.sum())
    if s <= 0:
        raise ValueError(
            f"VIS PSF at {psf_fits_path} sums to {s}; cannot normalise"
        )
    return (psf / s).astype(np.float32)


class EuclidVISForwardOp(tf.keras.layers.Layer):
    """Deterministic Euclid VIS image-formation forward op.

    ``call(hr_vis_linear)`` → ``lr_vis_linear``:

    1. PSF convolution on the HR grid (Fourier-domain ``tf.signal.rfft2d``
       with the VIS empirical PSF as a fixed kernel, ``mode='same'`` so the
       output keeps the HR shape).
    2. ``rebin_factor × rebin_factor`` sum-rebin via average-pool×area.

    Units in = units out: pixels in linear electrons (no asinh stretch
    inside this layer). Callers are responsible for any input/output
    stretching.

    Notes
    -----
    Use ``crop_half_side`` to shrink the convolution kernel for speed
    during training; the default (full saved kernel) keeps numerical
    parity with :class:`ObservationSimulator._process_one_band`.

    The PSF kernel is registered as a non-trainable Keras weight so
    ``model.save`` / ``model.load`` carry it without needing the FITS
    file alongside the checkpoint.
    """

    def __init__(
        self,
        psf_fits_path: str | None = None,
        *,
        rebin_factor: int = Config.DEFAULT_REBIN_FACTOR,
        crop_half_side: int | None = None,
        name: str = "euclid_vis_forward_op",
        **kwargs,
    ) -> None:
        super().__init__(name=name, trainable=False, **kwargs)
        if rebin_factor < 1:
            raise ValueError(f"rebin_factor must be ≥ 1, got {rebin_factor}")
        self.rebin_factor = int(rebin_factor)
        self._psf_fits_path = psf_fits_path or _default_vis_psf_path()
        self._crop_half_side = crop_half_side

        psf = _load_vis_psf_kernel(
            self._psf_fits_path, crop_half_side=self._crop_half_side,
        )
        # Stored UN-flipped: the FFT path below computes true convolution
        # directly (irfft(rfft(img)·rfft(psf))), matching
        # ``scipy.signal.fftconvolve(img, psf)`` — no kernel flip needed
        # (a flip would turn it into correlation). Shape ``[Kh, Kw]``.
        self._psf_kernel = self.add_weight(
            name="vis_psf_kernel",
            shape=psf.shape,
            initializer=tf.constant_initializer(psf),
            trainable=False,
            dtype=tf.float32,
        )

    @property
    def psf_kernel_size(self) -> int:
        """Side length of the active PSF kernel (after optional crop)."""
        size = self._psf_kernel.shape[0]
        if size is None:
            raise RuntimeError("PSF kernel has no statically known size")
        return int(size)

    def call(self, hr_vis_linear: tf.Tensor) -> tf.Tensor:
        """Apply PSF + sum-rebin to a batch of HR VIS images.

        The PSF convolution is done in the Fourier domain
        (``tf.signal.rfft2d``) rather than ``tf.nn.conv2d``. The Euclid
        VIS PSF is ~1023×1023 on the HR grid; a direct spatial conv2d
        with a kernel that large forces cuDNN onto an FFT-conv path that
        is pathologically slow / hangs on GPU. An explicit FFT is
        O(N log N) regardless of kernel size, keeps the full PSF wings
        (no cropping needed), is differentiable, and reproduces
        ``scipy.signal.fftconvolve(..., mode='same')`` — the exact
        operator the offline synthetic forward uses.

        Parameters
        ----------
        hr_vis_linear
            ``[B, H, W, 1]`` float32 tensor in linear electron units.

        Returns
        -------
        ``[B, H // rebin_factor, W // rebin_factor, 1]`` float32 tensor
        in linear electron units (sum-rebin preserves photometric flux).
        """
        psf = self._psf_kernel                       # [Kh, Kw], unit-sum
        kh_raw, kw_raw = psf.shape[0], psf.shape[1]
        if kh_raw is None or kw_raw is None:
            raise RuntimeError("PSF kernel has no statically known spatial shape")
        kh, kw = int(kh_raw), int(kw_raw)

        img = tf.gather(hr_vis_linear, 0, axis=-1)   # [B, H, W]
        h = tf.shape(img)[1]
        w = tf.shape(img)[2]
        # Linear (not circular) convolution: zero-pad both operands to at
        # least the full-conv size (H+Kh-1, W+Kw-1). We round the FFT
        # length up to a power of two — exact for linear conv (extra
        # trailing zeros), but a much faster transform than the ugly
        # H+K-1 length. The 'same' crop below is unaffected (it indexes
        # the central H×W, well inside the valid region).
        fh = _next_pow2(h + (kh - 1))
        fw = _next_pow2(w + (kw - 1))
        fft_len = tf.stack([fh, fw])

        img_f = tf.signal.rfft2d(img, fft_length=fft_len)        # [B, fh, fw//2+1]
        psf_f = tf.signal.rfft2d(psf, fft_length=fft_len)        # [fh, fw//2+1]
        full  = tf.signal.irfft2d(img_f * psf_f, fft_length=fft_len)  # [B, fh, fw]

        # ``mode='same'`` crop: scipy keeps the central H×W of the full
        # linear conv, offset by ``(K-1)//2`` per axis.
        sh = (kh - 1) // 2
        sw = (kw - 1) // 2
        same = full[:, sh:sh + h, sw:sw + w]         # [B, H, W]
        x = tf.expand_dims(same, axis=-1)             # [B, H, W, 1]

        # Sum-rebin via avg_pool × area. ``padding='VALID'`` trims
        # trailing pixels that don't fit a whole bin — matches
        # ``ObservationSimulator.sum_rebin``'s ``[:h_t, :w_t]``.
        if self.rebin_factor > 1:
            x = tf.nn.avg_pool2d(
                x, ksize=self.rebin_factor, strides=self.rebin_factor,
                padding="VALID",
            ) * (self.rebin_factor ** 2)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "psf_fits_path":  self._psf_fits_path,
            "rebin_factor":   self.rebin_factor,
            "crop_half_side": self._crop_half_side,
        })
        return cfg


class HSTForwardOp(EuclidVISForwardOp):
    """HST F814W image-formation forward op: ``H ⊛ SR``, no rebin.

    Same FFT-conv machinery as :class:`EuclidVISForwardOp` but with the
    HST PSF and ``rebin_factor=1`` — the HST image shares SR's 0.05″/pix
    HR grid, so there is no downsample. Used by the ``SR = sky``
    objective: the HST supervised loss compares ``H ⊛ SR`` to the
    observed HST image, so the model learns the deconvolved sky (matching
    what the synthetic + round-trip paths target) rather than the
    HST-PSF-blurred image.
    """

    def __init__(
        self,
        psf_fits_path: str | None = None,
        *,
        crop_half_side: int | None = None,
        name: str = "hst_forward_op",
        **kwargs,
    ) -> None:
        super().__init__(
            psf_fits_path or _default_hst_psf_path(),
            rebin_factor=1,
            crop_half_side=crop_half_side,
            name=name,
            **kwargs,
        )
