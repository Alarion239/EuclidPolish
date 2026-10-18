"""A differentiable detector of pixel-grid-aligned artifacts (the "grid").

Long-trained members put faint lines along the pixel rows and columns into
low-signal light of real Euclid images (2026-10-18). The sky has no reason
to prefer the detector's axes, so the signature is power concentrated on the
frequency axes. Per square window of the high-passed image (image minus its
Gaussian blur, mirror-padded so tile edges add nothing), tapered by a
circular Hann window, the FFT power on the axes (``|kx|`` or ``|ky|`` within
``half_width`` bins) of every integer radius ring (periods ``period_min`` to
``period_max`` pixels) is compared with what the rest of the same ring
predicts for it. Summed over the rings this gives, per window and band:

* :meth:`AxisExcess.score`: on-axis power over its prediction (1 = isotropic),
* :meth:`AxisExcess.zscore`: the excess in units of its fluctuation for an
  isotropic image,
* :meth:`AxisExcess.amplitude`: the excess as a signed RMS amplitude in image
  units (Parseval).

The excess is signed, so an isotropic image scores 0 on average whatever its
power spectrum; the fluctuation is exact for white noise and within ~20 %
for steep spectra. It needs no truth, so it serves both as a metric on real
images and as a training penalty (:meth:`AxisExcess.penalty`).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def _gaussian_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray:
    radius = int(truncate * sigma + 0.5)          # as scipy.ndimage.gaussian_filter
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def _white_noise_fluctuation(on: np.ndarray, off: np.ndarray, taper: np.ndarray) -> float:
    """``Var(excess) / sum_r pred_r**2 / n_on_r`` for white Gaussian noise.

    The tapered window correlates neighbouring FFT bins: for unit white noise
    ``Cov(|F_k|², |F_k'|²) = G(k - k') + G(k + k')`` with ``G = |FFT(taper²)|²``.
    The excess is ``sum_k a_k |F_k|²`` (a = +1 on the axes, minus the ring's
    on/off bin ratio off them), and ``a`` is symmetric under ``k -> -k``.
    """
    a = (on - off * (on.sum(axis=(1, 2)) / off.sum(axis=(1, 2)))[:, None, None]).sum(axis=0)
    g = np.abs(np.fft.fft2(taper.astype(np.float64) ** 2)) ** 2
    var = 2.0 * float((a * np.real(np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(g)))).sum())
    mean_bin = float((taper.astype(np.float64) ** 2).sum())
    return var / float(on.sum() * mean_bin ** 2)


class AxisExcess:
    """The grid detector for one window size (see the module docstring)."""

    def __init__(self, window: int = 64, period_min: float = 3.0, period_max: float = 24.0,
                 highpass_sigma: float = 3.0, half_width: float = 1.5):
        w = int(window)
        if w < 16:
            raise ValueError(f"window must be >= 16 pixels, got {window}")
        self.window = w
        self.highpass_sigma = float(highpass_sigma)
        self._kernel = _gaussian_kernel(self.highpass_sigma)
        ky, kx = np.meshgrid(np.fft.fftfreq(w), np.fft.fftfreq(w), indexing="ij")
        kr = np.hypot(kx, ky)
        annulus = (kr >= 1.0 / float(period_max)) & (kr <= 1.0 / float(period_min))
        on = annulus & ((np.abs(kx) <= half_width / w) | (np.abs(ky) <= half_width / w))
        off = annulus & ~on
        rings = np.floor(kr * w + 0.5).astype(int)
        on_masks, off_masks = [], []
        for r in sorted(set(rings[on].tolist())):
            mo, mf = on & (rings == r), off & (rings == r)
            if mo.any() and mf.any():
                on_masks.append(mo)
                off_masks.append(mf)
        on_r = np.stack(on_masks).astype(np.float32)                    # (R, w, w)
        off_r = np.stack(off_masks).astype(np.float32)
        self._on = on_r
        self._off = off_r
        self._on_n = on_r.sum(axis=(1, 2))                              # (R,)
        self._ring_ratio = self._on_n / off_r.sum(axis=(1, 2))
        yy, xx = np.mgrid[0:w, 0:w]
        rr = np.hypot(yy - (w - 1) / 2.0, xx - (w - 1) / 2.0)
        hann = 0.5 * (1.0 + np.cos(np.pi * rr / (w / 2.0)))
        self._taper = np.where(rr < w / 2.0, hann, 0.0).astype(np.float32)
        self._norm = float(w * w * (self._taper.astype(np.float64) ** 2).sum())
        self._fluctuation = _white_noise_fluctuation(on_r, off_r, self._taper)

    # -- building blocks -------------------------------------------------------

    def highpass(self, img: tf.Tensor) -> tf.Tensor:
        """``img - blur(img)`` for ``(B, H, W, C)``, mirror-padded."""
        img = tf.convert_to_tensor(img, tf.float32)
        k = tf.constant(self._kernel)
        pad = (len(self._kernel) - 1) // 2
        channels = int(img.shape[-1])
        padded = tf.pad(img, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode="SYMMETRIC")
        kh = tf.tile(tf.reshape(k, (-1, 1, 1, 1)), (1, 1, channels, 1))
        kw = tf.tile(tf.reshape(k, (1, -1, 1, 1)), (1, 1, channels, 1))
        blur = tf.nn.depthwise_conv2d(padded, kh, strides=(1, 1, 1, 1), padding="VALID")
        blur = tf.nn.depthwise_conv2d(blur, kw, strides=(1, 1, 1, 1), padding="VALID")
        return img - blur

    def windows(self, img: tf.Tensor) -> tf.Tensor:
        """Non-overlapping windows of ``(B, H, W, C)`` → ``(B, nh, nw, C, w, w)``
        (trailing rows/columns that do not fill a window are dropped)."""
        w = self.window
        img = tf.convert_to_tensor(img, tf.float32)
        if (img.shape[1] is not None and img.shape[1] < w) or (img.shape[2] is not None and img.shape[2] < w):
            raise ValueError(f"image {tuple(img.shape[1:3])} is smaller than one {w}-pixel window")
        b, h, wd, c = (tf.shape(img)[i] for i in range(4))
        nh, nw = h // w, wd // w
        x = img[:, :nh * w, :nw * w, :]
        x = tf.reshape(x, (b, nh, w, nw, w, c))
        return tf.transpose(x, (0, 1, 3, 5, 2, 4))

    def ring_power(self, win: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """On-axis power per ring of high-passed windows ``(..., w, w)`` and
        the isotropic prediction for it from the rest of the ring → two
        ``(..., R)`` tensors."""
        win = tf.convert_to_tensor(win, tf.float32)
        spec = tf.signal.fft2d(tf.complex(win * self._taper, tf.zeros_like(win)))
        power = tf.math.real(spec) ** 2 + tf.math.imag(spec) ** 2
        on = tf.einsum("...ij,rij->...r", power, self._on)
        predicted = tf.einsum("...ij,rij->...r", power, self._off) * self._ring_ratio
        return on, predicted

    def _excess_and_spread(self, on: tf.Tensor, predicted: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        excess = tf.reduce_sum(on - predicted, axis=-1)
        spread = tf.sqrt(self._fluctuation * tf.reduce_sum(predicted ** 2 / self._on_n, axis=-1) + 1e-30)
        return excess, spread

    def _ring_power_of(self, img: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return self.ring_power(self.windows(self.highpass(img)))

    # -- public ----------------------------------------------------------------

    def score(self, img: tf.Tensor) -> tf.Tensor:
        """On-axis power over its isotropic prediction per window and band of
        ``(B, H, W, C)`` → ``(B, nh, nw, C)``; 1 for an isotropic image."""
        on, predicted = self._ring_power_of(img)
        return tf.reduce_sum(on, axis=-1) / (tf.reduce_sum(predicted, axis=-1) + 1e-30)

    def zscore(self, img: tf.Tensor) -> tf.Tensor:
        """The axis excess in units of its isotropic fluctuation → ``(B, nh, nw, C)``."""
        excess, spread = self._excess_and_spread(*self._ring_power_of(img))
        return excess / spread

    def amplitude(self, img: tf.Tensor) -> tf.Tensor:
        """The axis excess as a signed RMS amplitude in image units → ``(B, nh, nw, C)``."""
        on, predicted = self._ring_power_of(img)
        excess = tf.reduce_sum(on - predicted, axis=-1) / self._norm
        return tf.sign(excess) * tf.sqrt(tf.abs(excess))

    def penalty(self, img: tf.Tensor, scale, gate: float = 2.0, floor: float = 0.01) -> tf.Tensor:
        """Mean over windows and bands of the axis excess beyond ``gate``
        isotropic fluctuations, in units of that fluctuation (``relu(z -
        gate)``), so a grid in nearly empty sky counts as much as one in
        bright light: what shows under a faint display stretch.

        The fluctuation is floored at the power of a pattern ``floor`` times
        ``scale`` RMS (per band; with each band's per-pixel noise, 1 % of the
        noise), which keeps the gradient bounded where the image is ~0.

        The prediction from the rest of each ring (and so the fluctuation) is
        held constant: the gradient only removes on-axis power, and adding
        isotropic texture to hide a grid earns nothing. Within ``gate``
        fluctuations of isotropic the penalty is 0.
        """
        on, predicted = self._ring_power_of(img)
        predicted = tf.stop_gradient(predicted)
        excess, spread = self._excess_and_spread(on, predicted)
        scale = tf.reshape(tf.convert_to_tensor(scale, tf.float32), (1, 1, 1, -1))
        floor_power = self._norm * (float(floor) * scale) ** 2
        return tf.reduce_mean(tf.nn.relu(excess - float(gate) * spread) / (spread + floor_power))


def default_detector() -> AxisExcess:
    """The detector used in the 2026-10-18 measurements (64-px windows,
    periods 3-24 px, 3-px high-pass)."""
    return AxisExcess(window=64, period_min=3.0, period_max=24.0, highpass_sigma=3.0)


__all__ = ["AxisExcess", "default_detector"]
