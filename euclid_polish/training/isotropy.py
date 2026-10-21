"""A training penalty that keeps a member's error free of any preferred direction.

Members trained at sub-noise knees paint a pixel-axis grid into faint light
(2026-10-18): NISP noise, resampled from 0.3" detector pixels onto the 0.1"
grid, is correlated along the pixel axes, and the low knees reward
reproducing sub-noise structure, so the network reports noise streaks as
light. The targets have no such pattern, and neither should the error.
Averaged over many windows, an error with no preferred direction has a power
spectrum that depends only on the frequency's length |k|.

Per square window of the residual ``r = asinh(sr/k) - asinh(hr/k)``
(electrons; ``k`` = ``knee_e``, 0.01 e⁻ by default, nearly logarithmic above
it, so faint light counts as much as bright), minus its mean and tapered by
a round Hann window, the FFT power of every bin with a period of
``period_min``-``period_max`` pixels (down to the Nyquist period, 2 px: beyond
it only the square grid's corners remain, which no rotation reaches) is
divided by the window's own isotropic expectation at that |k|: its ring
means, interpolated log-log to the bin's exact |k|. A plain ring mean,
constant over a ring one bin wide, would not do: on the square lattice the
bins of one ring sit at radii that depend on their direction, so a steep but
isotropic spectrum looks 4-fold anisotropic by up to ~50 % (2026-10-20). A
window's normalised spectrum is ~1 at every bin when its error is isotropic,
and a faint empty-sky window counts as much as a bright halo.

The batch mean ``B(k)`` of the normalised spectra feeds an exponential moving
average ``M(k)`` (``decay``; state checkpointed with the model), so isotropy
is enforced on average, not batch by batch. The excess
``E(k) = relu(M(k) / ringmean(M) - 1 - tolerance)`` marks the directions that
carry more error than the rest of their ring, and the penalty is
``sum_k E(k) B(k) / sum_k B(k)``, averaged over the bands. ``E``, the window
references and the denominator are constants, so the gradient only removes
error power where it concentrates (adding isotropic error elsewhere earns
nothing), and the penalty vanishes once the error is isotropic. Within a step
it is a weighted squared error of the residual, which the ideal estimator
(the posterior mean) minimises for any weights: it never pulls a perfect
model off the truth, it only changes what an imperfect one fixes first, and
an anisotropy no estimator can remove costs no gradient at the optimum.

:class:`MultiScaleIsotropy` scores several scales, each a window size and
period range with its own moving average and weight: a 64-px window resolves
too few directions at long periods (its rings there are a few bins on the
square grid; it reads round 128-px spectra as ~10 % axis-heavy at 32-43 px,
2026-10-20). Periods of 2-3 px carry far more gradient per unit of anisotropy
than the rest (~400x on member_235): a period-2 pattern along the axes is what
the 2x upsampler's four sub-pixel channels make when slightly unbalanced, and
one last-layer weight moves every other pixel at once. That band is weighted
down to the same leverage per unit of anisotropy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tensorflow as tf

#: The residual's asinh knee (e⁻ per HR pixel).
DEFAULT_KNEE_E = 0.01
#: (window px, shortest period px, longest period px, weight) per scale:
#: 64-px windows for 3-21 px, the same for 2-3 px weighted 0.003 (equal
#: leverage per unit of anisotropy, see above), 128-px windows for 21-42 px.
#: On 2026-10-20 the largest anisotropy left (2.6-3.5x on the axes in every
#: member) was at 2-3 px, outside the first members' range.
DEFAULT_SCALES: tuple[tuple[float, float, float, float], ...] = (
    (64.0, 3.0, 21.0, 1.0), (64.0, 2.0, 3.0, 0.003), (128.0, 21.0, 42.0, 1.0))
#: The range of the first members with the penalty (233-235).
LEGACY_SCALES: tuple[tuple[float, float, float, float], ...] = ((64.0, 3.0, 21.0, 1.0),)


def canonical_scales(scales: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float, float], ...]:
    """Scales as (window, shortest, longest period, weight) floats; a
    three-number scale has weight 1."""
    out = []
    for scale in scales:
        values = tuple(float(v) for v in scale)
        if len(values) not in (3, 4):
            raise ValueError(f"a scale is (window, shortest, longest[, weight]), got {scale!r}")
        out.append(values if len(values) == 4 else (*values, 1.0))
    return tuple(out)


def _round_hann(window: int) -> np.ndarray:
    yy, xx = np.mgrid[0:window, 0:window]
    rr = np.hypot(yy - (window - 1) / 2.0, xx - (window - 1) / 2.0)
    hann = 0.5 * (1.0 + np.cos(np.pi * rr / (window / 2.0)))
    return np.where(rr < window / 2.0, hann, 0.0)


class ResidualIsotropy(tf.Module):
    """The penalty and its moving average for ``n_bands`` bands (see the
    module docstring)."""

    def __init__(self, knee_e: float = DEFAULT_KNEE_E, n_bands: int = 4, window: int = 64,
                 period_min: float = 3.0, period_max: float = 21.0, decay: float = 0.99,
                 tolerance: float = 0.05, floor_e: float = 1e-4, name: str = "residual_isotropy"):
        super().__init__(name=name)
        w = int(window)
        if not knee_e > 0:
            raise ValueError(f"knee_e must be positive, got {knee_e!r}")
        if not 2.0 <= period_min < period_max:
            raise ValueError(f"need 2 <= period_min < period_max, got {period_min!r}, {period_max!r}")
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay!r}")
        lo, hi = w / float(period_max), w / float(period_min)
        if lo < 2.0:
            raise ValueError(f"period_max {period_max!r} is too long for a {w}-pixel window")
        k_y, k_x = np.meshgrid(np.fft.fftfreq(w) * w, np.fft.fftfreq(w) * w, indexing="ij")
        radius = np.hypot(k_x, k_y).ravel()                 # |k| in cycles per window
        rings = np.rint(radius).astype(int)
        # Whole rings one beyond the annulus on each side, so every annulus bin
        # lies between two ring centres - at the top no further than the last
        # ring the window holds whole: the few bins past its centre, up to the
        # Nyquist circle, take the last two rings' slope.
        first = max(1, int(np.floor(lo)) - 1)
        last = min(int(np.ceil(hi)) + 1, int(np.floor(w / 2.0 - 0.5)))
        hi = min(hi, w / 2.0)
        ring_ids = np.arange(first, last + 1)
        member = (rings[None, :] == ring_ids[:, None]).astype(np.float64)       # (R, w²)
        log_centre = np.log((member @ radius) / member.sum(axis=1))
        annulus = np.flatnonzero((radius >= lo) & (radius <= hi))
        log_k = np.log(radius[annulus])
        j = np.clip(np.searchsorted(log_centre, log_k) - 1, 0, ring_ids.size - 2)
        t = (log_k - log_centre[j]) / (log_centre[j + 1] - log_centre[j])
        interp = np.zeros((annulus.size, ring_ids.size))
        interp[np.arange(annulus.size), j] = 1.0 - t
        interp[np.arange(annulus.size), j + 1] = t
        # The annulus bins grouped by ring: the reference of the moving average.
        ann_rings = rings[annulus]
        onehot = (ann_rings[None, :] == np.unique(ann_rings)[:, None]).astype(np.float64)
        taper = _round_hann(w)

        self.knee_e = float(knee_e)
        self.n_bands = int(n_bands)
        self.window = w
        self.period_min = float(period_min)
        self.period_max = float(period_max)
        self.decay = float(decay)
        self.tolerance = float(tolerance)
        self.n_bins = int(annulus.size)
        self._taper = tf.constant(taper, tf.float32)
        self._taper_sum = float(taper.sum())
        self._ring_mean = tf.constant(member / member.sum(axis=1, keepdims=True), tf.float32)
        self._interp = tf.constant(interp, tf.float32)
        self._annulus = tf.constant(annulus, tf.int32)
        self._ann_mean = tf.constant(onehot / onehot.sum(axis=1, keepdims=True), tf.float32)
        self._ann_onehot = tf.constant(onehot, tf.float32)
        # The power of a white pattern of ``floor_e`` e⁻ RMS near zero flux:
        # keeps a window whose residual is ~0 from dividing by ~0.
        self._floor_power = float((floor_e / knee_e) ** 2 * (taper ** 2).sum())
        self.mean = tf.Variable(tf.ones((self.n_bands, self.n_bins)), trainable=False,
                                name="normalised_spectrum_mean")

    def normalised_spectra(self, residual: tf.Tensor) -> tf.Tensor:
        """``(B, H, W, C)`` residual (stretched) → ``(N, C, n_bins)``: every
        window's annulus power over its isotropic expectation. Trailing rows
        and columns that do not fill a window are dropped."""
        w = self.window
        residual = tf.convert_to_tensor(residual, tf.float32)
        if any(n is not None and n < w for n in residual.shape[1:3]):
            raise ValueError(f"image {tuple(residual.shape[1:3])} is smaller than one {w}-pixel window")
        b, h, wd, c = (tf.shape(residual)[i] for i in range(4))
        nh, nw = h // w, wd // w
        x = tf.reshape(residual[:, :nh * w, :nw * w, :], (b, nh, w, nw, w, c))
        x = tf.reshape(tf.transpose(x, (0, 1, 3, 5, 2, 4)), (-1, c, w, w))
        x = x - tf.reduce_sum(x * self._taper, axis=(-2, -1), keepdims=True) / self._taper_sum
        spec = tf.signal.fft2d(tf.complex(x * self._taper, tf.zeros_like(x)))
        power = tf.reshape(tf.math.real(spec) ** 2 + tf.math.imag(spec) ** 2, (-1, c, w * w))
        ring_mean = tf.stop_gradient(tf.einsum("ncp,rp->ncr", power, self._ring_mean))
        expected = tf.exp(tf.einsum("ncr,ar->nca", tf.math.log(ring_mean + self._floor_power),
                                    self._interp))
        return tf.gather(power, self._annulus, axis=-1) / expected

    def _excess(self, mean: tf.Tensor) -> tf.Tensor:
        """``relu(M / ringmean(M) - 1 - tolerance)`` per band and bin."""
        ring = tf.matmul(tf.matmul(mean, self._ann_mean, transpose_b=True), self._ann_onehot)
        return tf.nn.relu(mean / ring - 1.0 - self.tolerance)

    def residual(self, sr: tf.Tensor, hr: tf.Tensor) -> tf.Tensor:
        """``asinh(sr/k) - asinh(hr/k)`` of electrons, ``(B, H, W, C)``."""
        sr = tf.cast(sr, tf.float32)
        hr = tf.cast(hr, tf.float32)
        return tf.asinh(sr / self.knee_e) - tf.asinh(hr / self.knee_e)

    def penalty(self, sr: tf.Tensor, hr: tf.Tensor) -> tf.Tensor:
        """The penalty of a batch of predictions ``sr`` against targets
        ``hr``, both ``(B, H, W, n_bands)`` electrons; folds the batch into
        the moving average."""
        return self.penalty_of_residual(self.residual(sr, hr))

    def penalty_of_residual(self, residual: tf.Tensor) -> tf.Tensor:
        """:meth:`penalty` of a :meth:`residual`."""
        batch = tf.reduce_mean(self.normalised_spectra(residual), axis=0)          # (C, n_bins)
        mean = self.decay * self.mean + (1.0 - self.decay) * tf.stop_gradient(batch)
        self.mean.assign(mean)
        excess = tf.stop_gradient(self._excess(mean))
        per_band = tf.reduce_sum(excess * batch, axis=-1) / tf.stop_gradient(
            tf.reduce_sum(batch, axis=-1))
        return tf.reduce_mean(per_band)

    def excess(self) -> np.ndarray:
        """Per band, the penalty the moving average implies on its own
        (``sum E M / sum M``): how anisotropic the recent error has been."""
        mean = tf.convert_to_tensor(self.mean)
        return (tf.reduce_sum(self._excess(mean) * mean, axis=-1)
                / tf.reduce_sum(mean, axis=-1)).numpy()

    def describe(self) -> str:
        return f"{self.window}-px windows, periods {self.period_min:g}-{self.period_max:g} px"

    def summary(self) -> str:
        """The moving-average excess per band."""
        return " ".join(f"band{b} {float(e):.4f}" for b, e in enumerate(self.excess()))


class MultiScaleIsotropy(tf.Module):
    """The penalty over several scales - ``(window, period_min, period_max[,
    weight])`` each, with its own moving average - as a weighted mean; one
    residual feeds them all."""

    def __init__(self, scales: Sequence[Sequence[float]] = DEFAULT_SCALES,
                 knee_e: float = DEFAULT_KNEE_E, n_bands: int = 4, decay: float = 0.99,
                 tolerance: float = 0.05, floor_e: float = 1e-4, name: str = "multi_scale_isotropy"):
        super().__init__(name=name)
        if not scales:
            raise ValueError("need at least one scale")
        scales = canonical_scales(scales)
        if not all(weight > 0 for *_, weight in scales):
            raise ValueError(f"scale weights must be positive, got {scales!r}")
        self.scales = [ResidualIsotropy(knee_e=knee_e, n_bands=n_bands, window=int(w), period_min=lo,
                                        period_max=hi, decay=decay, tolerance=tolerance, floor_e=floor_e,
                                        name=f"isotropy_{int(w)}px_{i}")
                       for i, (w, lo, hi, _) in enumerate(scales)]
        self.weights = tuple(weight for *_, weight in scales)
        self.knee_e = float(knee_e)
        self.decay = float(decay)
        self.tolerance = float(tolerance)

    @property
    def window(self) -> int:
        """The largest window: training crops must hold one."""
        return max(scale.window for scale in self.scales)

    def penalty(self, sr: tf.Tensor, hr: tf.Tensor) -> tf.Tensor:
        residual = self.scales[0].residual(sr, hr)
        terms = [weight * scale.penalty_of_residual(residual)
                 for scale, weight in zip(self.scales, self.weights, strict=True)]
        return tf.add_n(terms) / sum(self.weights)

    def excess(self) -> np.ndarray:
        """Per band, the scales' :meth:`ResidualIsotropy.excess`, weighted as
        the penalty weights them."""
        return np.average([scale.excess() for scale in self.scales], axis=0, weights=self.weights)

    def describe(self) -> str:
        return "; ".join(f"{scale.describe()} x{weight:g}" for scale, weight in zip(self.scales, self.weights,
                                                                                   strict=True))

    def summary(self) -> str:
        """Each scale's moving-average excess, mean over bands."""
        return ", ".join(f"{scale.window}px {scale.period_min:g}-{scale.period_max:g} "
                         f"{float(np.mean(scale.excess())):.4f}" for scale in self.scales)


__all__ = ["DEFAULT_KNEE_E", "DEFAULT_SCALES", "LEGACY_SCALES", "MultiScaleIsotropy", "ResidualIsotropy",
           "canonical_scales"]
