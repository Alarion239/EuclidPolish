"""A training penalty that keeps the super-resolved sky as bright as Euclid saw it.

Re-observed the way Euclid observes, the model's sky must reproduce what
Euclid would record within Euclid's own error bars. On synthetic scenes the
truth is known, so the comparison needs no noise draw: the error
``SR − truth`` (electrons on the 0.05" grid) is blurred by each band's mean
Euclid PSF and sum-rebinned 2× onto the 0.10" archive grid — the
simulator's own operator
(:class:`~euclid_polish.sky.observation.observation_simulator.ObservationSimulator`).
Both images pass through the same operator, so a representative PSF (the
member's mean ePSF, cut to its core and renormalised) measures brightness at
Euclid's resolution; a scene's own warped PSF would change nothing here. On
real sky the PSF is unknown and would have to be fitted.

Units are Euclid's noise. The archive noise model
(:func:`~euclid_polish.sky.observation.noise.apply_archive_noise`) sets a
pixel's level to ``sqrt(sky² + signal)`` — sky plus the source's photon
noise — times a correlated unit field, so the noise of the mean over an
``s × s`` block is that level times the unit field's block-mean scatter
``c_s``, measured once from the noise model (VIS is nearly white beyond 2 px;
NISP, resampled from 0.3" pixels, is not).

The penalty is what lies beyond 1σ at every scale::

    z_s = blockmean_s(A(SR − truth)) / (c_s · level_s)
    e_s = relu(|z_s| − 1)
    penalty = mean over s of mean(huber(e_s)),  huber(e) = e²/2 up to 1, e − 1/2 beyond

Inside Euclid's error bars it is exactly zero, so it never pulls a model off
the truth and stops pulling on a source once its brightness is right to
within the noise; a bright core is measured to ~1 %, so it is held to ~1 %.
Beyond 2σ it grows linearly: a fresh network's outputs, or a blown-out star,
sit thousands of σ off, and a squared excess would hand the whole gradient
(~10¹⁸ at the first step) to those few pixels.
The single-pixel band alone would pass a faint galaxy missing 10 % of its
light (each pixel only ~0.1σ low); the block means at 2, 4 and 8 archive
pixels, where the noise averages down, catch it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tensorflow as tf

from euclid_polish.config import BandConfig
from euclid_polish.sky.observation.noise import dithered_unit_noise

#: Block sizes (archive pixels) at which the brightness must agree.
BRIGHTNESS_SCALES = (1, 2, 4, 8)
#: Side (HR pixels) of the PSF core kept for re-observing: 2.05", which holds
#: ~94 % of a VIS ePSF and all but ~0.1 % of a NISP one.
BRIGHTNESS_KERNEL_PX = 41
#: The error bar, in units of Euclid's noise, inside which nothing is charged.
BRIGHTNESS_TOLERANCE = 1.0
#: Excess (σ beyond the error bar) where the charge turns from quadratic to linear.
BRIGHTNESS_HUBER_DELTA = 1.0


def block_noise_factors(band: BandConfig, scales: Sequence[int] = BRIGHTNESS_SCALES, *,
                        size: int = 512, seed: int = 0) -> tuple[float, ...]:
    """Scatter of the band's unit (dithered, resampled) noise field averaged
    over ``s × s`` archive-pixel blocks, for every ``s`` in ``scales``: the
    factor ``c_s`` that turns a pixel's noise level into the noise of a block
    mean. ``s = 1`` is :func:`~euclid_polish.sky.observation.noise.pixel_scatter_factor`."""
    unit = dithered_unit_noise((int(size), int(size)), band, np.random.default_rng(seed))
    out = []
    for s in scales:
        s = int(s)
        h = (int(size) // s) * s
        blocks = unit[:h, :h].reshape(h // s, s, h // s, s).mean(axis=(1, 3))
        out.append(float(np.std(blocks)))
    return tuple(out)


def core_kernel(psf: np.ndarray, size: int = BRIGHTNESS_KERNEL_PX) -> np.ndarray:
    """The central ``size × size`` of a PSF stamp (``size`` odd), renormalised
    to unit sum so re-observing conserves flux."""
    data = np.asarray(psf, np.float64)
    size = int(size)
    if size % 2 == 0:
        raise ValueError(f"the kernel side must be odd, got {size}")
    if data.ndim != 2 or min(data.shape) < size:
        raise ValueError(f"PSF stamp {data.shape} is smaller than the {size}-px core")
    cy, cx = data.shape[0] // 2, data.shape[1] // 2
    half = size // 2
    core = data[cy - half:cy + half + 1, cx - half:cx + half + 1]
    total = float(core.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("the PSF core has no positive flux")
    return core / total


class BrightnessConsistency:
    """The re-observed brightness penalty (see the module docstring).

    ``kernels``: one odd, square PSF core per band (HR pixels, same size);
    ``sky_rms_e``: each band's sky level on the archive grid (e⁻ per 0.10"
    pixel, the noise model's ``mer_rms_e``); ``block_noise``: per band, the
    :func:`block_noise_factors` at ``scales``.
    """

    def __init__(self, kernels: Sequence[np.ndarray], sky_rms_e: Sequence[float],
                 block_noise: Sequence[Sequence[float]], *,
                 scales: Sequence[int] = BRIGHTNESS_SCALES, rebin: int = 2,
                 tolerance: float = BRIGHTNESS_TOLERANCE,
                 huber_delta: float = BRIGHTNESS_HUBER_DELTA):
        kernels = [np.asarray(k, np.float64) for k in kernels]
        n_bands = len(kernels)
        if n_bands == 0 or len(sky_rms_e) != n_bands or len(block_noise) != n_bands:
            raise ValueError("kernels, sky_rms_e and block_noise need one entry per band")
        shapes = {k.shape for k in kernels}
        side = kernels[0].shape[0]
        if len(shapes) != 1 or kernels[0].ndim != 2 or kernels[0].shape[1] != side or side % 2 == 0:
            raise ValueError(f"kernels must share one odd square shape, got {sorted(shapes)}")
        self.scales = tuple(int(s) for s in scales)
        if not self.scales or min(self.scales) < 1:
            raise ValueError(f"scales must be positive integers, got {scales!r}")
        factors = np.asarray(block_noise, np.float64)
        if factors.shape != (n_bands, len(self.scales)) or not np.all(factors > 0.0):
            raise ValueError(f"block_noise needs {n_bands} × {len(self.scales)} positive factors")
        sky = np.asarray(sky_rms_e, np.float64)
        if not np.all(np.isfinite(sky)) or not np.all(sky > 0.0):
            raise ValueError(f"sky_rms_e must be positive, got {list(sky_rms_e)}")
        if int(rebin) < 1 or not float(tolerance) >= 0.0 or not float(huber_delta) > 0.0:
            raise ValueError("rebin must be >= 1, tolerance >= 0 and huber_delta > 0")
        stacked = np.stack([k / k.sum() for k in kernels], axis=-1)[..., None]
        self._kernel = tf.constant(stacked, tf.float32)
        self._sky_var = tf.constant(np.square(sky), tf.float32)
        self._block_var = [tf.constant(np.square(factors[:, i]), tf.float32)
                           for i in range(len(self.scales))]
        self.n_bands = n_bands
        self.kernel_px = int(side)
        self.rebin = int(rebin)
        self.tolerance = float(tolerance)
        self.huber_delta = float(huber_delta)

    def describe(self) -> str:
        return (f"{self.kernel_px}-px PSF core, {self.rebin}× sum-rebin, "
                f"±{self.tolerance:g}σ at blocks of "
                + "/".join(str(s) for s in self.scales) + " archive px, linear beyond "
                f"{self.tolerance + self.huber_delta:g}σ")

    def reobserve(self, image: tf.Tensor) -> tf.Tensor:
        """``A(image)``: blur by each band's PSF core and sum-rebin onto the
        archive grid (``VALID``: no zero padding at the crop edges)."""
        image = tf.cast(image, tf.float32)
        blurred = tf.nn.depthwise_conv2d(image, self._kernel, strides=[1, 1, 1, 1],
                                         padding="VALID")
        if self.rebin == 1:
            return blurred
        r = self.rebin
        return tf.nn.avg_pool2d(blurred, r, r, "VALID") * float(r * r)

    def z_scores(self, sr: tf.Tensor, hr: tf.Tensor) -> list[tf.Tensor]:
        """Per scale, the re-observed error of the block means in units of
        Euclid's noise there."""
        diff = self.reobserve(tf.cast(sr, tf.float32) - tf.cast(hr, tf.float32))
        level2 = self._sky_var + tf.nn.relu(self.reobserve(hr))
        out = []
        for s, block_var in zip(self.scales, self._block_var, strict=True):
            if s == 1:
                err, var = diff, level2
            else:
                err = tf.nn.avg_pool2d(diff, s, s, "VALID")
                var = tf.nn.avg_pool2d(level2, s, s, "VALID")
            out.append(err / tf.sqrt(var * block_var))
        return out

    def penalty(self, sr: tf.Tensor, hr: tf.Tensor) -> tf.Tensor:
        """Mean over scales of the mean Huber charge on ``relu(|z_s| −
        tolerance)``: zero while the re-observed sky agrees with the truth
        within Euclid's noise, quadratic just beyond, linear far beyond."""
        d = self.huber_delta
        terms = []
        for z in self.z_scores(sr, hr):
            excess = tf.nn.relu(tf.abs(z) - self.tolerance)
            charge = tf.where(excess <= d, 0.5 * tf.square(excess), d * (excess - 0.5 * d))
            terms.append(tf.reduce_mean(charge))
        return tf.add_n(terms) / float(len(terms))
