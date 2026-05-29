"""TF-graph Euclid VIS forward operator for round-trip training.

The numpy/scipy forward model in
:class:`euclid_polish.sky.multiband_forward.MultiBandForward` runs as an
*offline* preprocessing step — it turns clean HR scenes into the dirty
LR records the dataloader streams. That's fine for supervised training
where M never sees its own output passed through it.

The round-trip loss path needs ``Conv(M(LR))`` evaluated *inside* the
gradient tape, so ``Conv`` has to be a TF-graph op whose output is
differentiable with respect to its input. This module provides that as
a Keras layer:

  * Constant (non-trainable) VIS PSF loaded from the same FITS the
    synthetic forward already uses (``$DATA_DIR/euclid_psf/euclid_psf_VIS.fits``).
  * ``tf.nn.conv2d`` with the PSF as a fixed kernel (flipped so it acts
    as a true convolution, matching ``scipy.signal.fftconvolve``;
    for the near-symmetric Euclid PSF the flip is a precaution, but it
    keeps the layer correct if a future PSF develops asymmetry).
  * Stride-N sum-rebin via ``avg_pool2d × N²``, mirroring
    :meth:`MultiBandForward.sum_rebin`.
  * **No noise.** Noise is a non-differentiable sampling step that
    would inject stochastic gradient noise unrelated to the model's
    output; the round-trip loss is defined on the deterministic image
    formation only.

The numerical contract with the numpy path is tested in
``tests/test_forward_op.py``: applying this layer to a known HR input
must match :meth:`MultiBandForward._process_one_band` for VIS to
within float32 round-off across the bulk of the array (boundary pixels
diverge slightly because ``tf.nn.conv2d`` uses zero-padding while
``scipy.signal.fftconvolve(mode='same')`` does the same — they agree
to round-off; the test crops a sentinel border to make the comparison
robust to the kernel size).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import tensorflow as tf

from euclid_polish.config import Config

from astropy.io import fits


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
    psf_fits_path: Optional[str] = None, *, crop_half_side: Optional[int] = None,
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
        psf = np.asarray(hdul[0].data, dtype=np.float32)
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

    1. PSF convolution on the HR grid (``tf.nn.conv2d`` with the VIS
       empirical PSF as a fixed kernel, ``padding='SAME'`` so the
       output keeps the HR shape).
    2. ``rebin_factor × rebin_factor`` sum-rebin via average-pool×area.

    Units in = units out: pixels in linear electrons (no asinh stretch
    inside this layer). Callers are responsible for any input/output
    stretching.

    Notes
    -----
    Use ``crop_half_side`` to shrink the convolution kernel for speed
    during training; the default (full saved kernel) keeps numerical
    parity with :class:`MultiBandForward._process_one_band`.

    The PSF kernel is registered as a non-trainable Keras weight so
    ``model.save`` / ``model.load`` carry it without needing the FITS
    file alongside the checkpoint.
    """

    def __init__(
        self,
        psf_fits_path: Optional[str] = None,
        *,
        rebin_factor: int = Config.DEFAULT_REBIN_FACTOR,
        crop_half_side: Optional[int] = None,
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
        return int(self._psf_kernel.shape[0])

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
        kh = int(psf.shape[0])
        kw = int(psf.shape[1])

        img = hr_vis_linear[..., 0]                  # [B, H, W]
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
        x = same[..., tf.newaxis]                    # [B, H, W, 1]

        # Sum-rebin via avg_pool × area. ``padding='VALID'`` trims
        # trailing pixels that don't fit a whole bin — matches
        # ``MultiBandForward.sum_rebin``'s ``[:h_t, :w_t]``.
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
        psf_fits_path: Optional[str] = None,
        *,
        crop_half_side: Optional[int] = None,
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
