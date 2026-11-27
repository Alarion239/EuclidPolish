"""Far-field light of bright stars beyond the ePSF stamp.

The empirical ePSF stamps end at 12.75″ (VIS, 511 px at 0.05″) and 25.6″
(NISP, 1023 px). Real DR1 stars keep a smooth halo and six diffraction-spike
ridges far beyond that: at VIS 12 the spikes stay above the sky noise to about
a minute of arc, at VIS 8 to several. The NISP stamps also hold a flat
pedestal of 7-9% of the flux between 10″ and 26″ that real H stars do not
have (about 0.9%), which draws a fake disc around every bright NISP star.

This module carries the measured far field (``bright_star_params.json``,
fitted on DR1 Wide only, provenance in the file):

  * halo   ``H(r)`` per 0.1″ pixel, as a fraction of the flux: the measured
    median profile (3-76″), log-log interpolated, continued beyond the last
    point as ``r^-γ_h``;
  * spikes ``S(a) rel_k exp(-p² / 2σ²)`` along each ridge, ``a`` the distance
    along the ridge, ``p`` across it, ``σ`` from the measured ridge FWHM;
    ``S`` is the fitted ``A_s (a / r0)^-γ_s`` beyond 9″ (the ridge medians are
    too noisy to tabulate there) and the measured medians inside. The ridge
    angles and relative heights ``rel_k`` are read off the scene's own
    (rolled, warped) kernel, so the far spikes continue the stamp's spikes.

The ePSF stamps themselves run too bright beyond ~7″: at 9″ the VIS kernel's
halo is 3.5× the real one and NISP's 2×, at 14″ NISP's 5× (the pedestal). So
the default join sits at 4.5-6.5″, where kernel and real stars agree.

The kernel and the far field meet in a join annulus ``[r_in, r_out]``: the
kernel is multiplied by a cosine taper ``w(r)`` (1 inside ``r_in``, 0 beyond
``r_out``) and the far field by ``1 - w(r)``, so their sum is continuous.
For star flux to be conserved the star kernel is scaled to ``1 - f_far``,
``f_far`` being the expected far-field flux (:func:`far_field_flux`).

Everything is analytic and rendered on the band's LR grid before noise, so the
far light gets photon noise and counts towards the saturation trigger like any
other starlight.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks

PARAMS_PATH = os.path.join(os.path.dirname(__file__), "bright_star_params.json")

#: Annulus (arcsec) of the kernel in which the spike ridges are located.
SPIKE_ANNULUS_ARCSEC = (5.0, 8.0)
#: Angular samples of the kernel polar unwrap (0.25° each).
_N_THETA = 1440
#: Ridges weaker than this many standard deviations of the angular profile
#: are not trusted (a Gaussian fallback kernel has none): evenly spaced
#: ridges of equal height are used instead.
_MIN_RIDGE_SNR = 2.0
#: Relative ridge heights read off a kernel are clipped to this range.
_REL_CLIP = (0.5, 2.0)


def _loglog(r: np.ndarray, r_tab: np.ndarray, v_tab: np.ndarray,
            tail_gamma: float) -> np.ndarray:
    """Log-log interpolation of ``v_tab(r_tab)``: the first segment's slope
    inside the table, ``r^-tail_gamma`` beyond its last point."""
    lr, lt, lv = np.log(np.asarray(r, np.float64)), np.log(r_tab), np.log(v_tab)
    out = np.interp(lr, lt, lv)
    first = (lv[1] - lv[0]) / (lt[1] - lt[0])
    out = np.where(lr < lt[0], lv[0] + first * (lr - lt[0]), out)
    out = np.where(lr > lt[-1], lv[-1] - tail_gamma * (lr - lt[-1]), out)
    return np.exp(out)


@dataclass(frozen=True)
class FarFieldBand:
    """Measured far field of one band (values per reference pixel, as a
    fraction of the star's flux): tabulated halo and ridge profiles with
    power-law tails."""

    r_arcsec: tuple[float, ...]
    halo: tuple[float, ...]
    ridge: tuple[float, ...]
    halo_gamma: float
    spike_gamma: float
    spike_fwhm_arcsec: float

    def __post_init__(self) -> None:
        r = np.asarray(self.r_arcsec)
        if (len(r) < 2 or len(self.halo) != len(r) or len(self.ridge) != len(r)
                or np.any(np.diff(r) <= 0) or min(self.halo) <= 0 or min(self.ridge) <= 0):
            raise ValueError("far-field tables need >= 2 increasing radii and "
                             "positive halo/ridge values of the same length")

    @property
    def spike_sigma_arcsec(self) -> float:
        return self.spike_fwhm_arcsec / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    def halo_at(self, r: np.ndarray) -> np.ndarray:
        return _loglog(r, np.asarray(self.r_arcsec), np.asarray(self.halo), self.halo_gamma)

    def ridge_at(self, r: np.ndarray) -> np.ndarray:
        return _loglog(r, np.asarray(self.r_arcsec), np.asarray(self.ridge), self.spike_gamma)


@dataclass(frozen=True)
class FarFieldParams:
    """The far-field model: per-band laws plus the shared geometry."""

    pixel_scale_arcsec: float
    r_max_arcsec: float
    n_spikes: int
    halo_jitter_ln: float
    spike_jitter_ln: float
    bands: dict[str, FarFieldBand]

    @classmethod
    def from_payload(cls, payload: dict) -> FarFieldParams:
        ff = payload["far_field"]
        return cls(
            pixel_scale_arcsec=float(ff["pixel_scale_arcsec"]),
            r_max_arcsec=float(ff["r_max_arcsec"]),
            n_spikes=int(ff["n_spikes"]),
            halo_jitter_ln=float(ff["halo_jitter_ln"]),
            spike_jitter_ln=float(ff["spike_jitter_ln"]),
            bands={
                name: FarFieldBand(
                    r_arcsec=tuple(float(v) for v in b["profile"]["r_arcsec"]),
                    halo=tuple(float(v) for v in b["profile"]["halo"]),
                    ridge=tuple(float(v) for v in b["profile"]["ridge"]),
                    halo_gamma=float(b["halo_gamma"]),
                    spike_gamma=float(b["spike_gamma"]),
                    spike_fwhm_arcsec=float(b["spike_fwhm_arcsec"]),
                )
                for name, b in ff["bands"].items()
            },
        )


@lru_cache(maxsize=4)
def load_far_field_params(path: str = PARAMS_PATH) -> FarFieldParams:
    """The committed far-field parameters (cached per path)."""
    with open(path, encoding="utf-8") as fh:
        return FarFieldParams.from_payload(json.load(fh))


def join_weight(r: np.ndarray, r_in: float, r_out: float) -> np.ndarray:
    """Kernel weight of the join: 1 inside ``r_in``, cosine to 0 at ``r_out``.

    The far field takes ``1 - join_weight``.
    """
    t = np.clip((np.asarray(r, np.float64) - r_in) / max(r_out - r_in, 1e-9), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * t))


def join_radii(kernel_half_arcsec: float, requested: tuple[float, float],
               taper_inner_frac: float) -> tuple[float, float]:
    """Join annulus inside the kernel's own edge taper.

    The join must end where the stamp's own taper (``taper_inner_frac`` of the
    half-side) begins, or the kernel would be tapered twice there. ``requested``
    is ``(r_in, r_out)``; it is shifted inwards, keeping its width, when the
    stamp is smaller.
    """
    r_in, r_out = float(requested[0]), float(requested[1])
    if not 0.0 < r_in < r_out:
        raise ValueError(f"join radii must satisfy 0 < r_in < r_out, got {requested!r}")
    limit = taper_inner_frac * kernel_half_arcsec
    if r_out > limit:
        r_in, r_out = r_in - (r_out - limit), limit
    if r_in <= 0.0:
        raise ValueError(
            f"kernel half-side {kernel_half_arcsec:.2f}″ too small for a join of "
            f"width {requested[1] - requested[0]:.2f}″")
    return r_in, r_out


def spike_ridges(kernel: np.ndarray, pixel_scale_arcsec: float, n_spikes: int,
                 annulus_arcsec: tuple[float, float] = SPIKE_ANNULUS_ARCSEC,
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Angles (radians, array frame: ``atan2(row, col)``) and relative heights
    (mean 1) of the ``n_spikes`` strongest ridges of ``kernel``.

    The kernel is unwrapped to polar coordinates in ``annulus_arcsec``, its
    azimuthal median removed per radius, and the r²-weighted residual averaged
    over radius; the highest peaks (at least 15° apart) are the ridges. A
    kernel without trustworthy ridges gets evenly spaced ones of equal height.
    """
    k = np.asarray(kernel, np.float64)
    c = (np.asarray(k.shape, np.float64) - 1.0) / 2.0
    half = min(c) * pixel_scale_arcsec
    lo = min(annulus_arcsec[0], 0.5 * half)
    hi = min(annulus_arcsec[1], 0.9 * half)
    r = np.arange(lo / pixel_scale_arcsec, hi / pixel_scale_arcsec, 1.0)
    th = np.linspace(0.0, 2.0 * np.pi, _N_THETA, endpoint=False)
    yy = c[0] + r[:, None] * np.sin(th)[None, :]
    xx = c[1] + r[:, None] * np.cos(th)[None, :]
    P = map_coordinates(k, [yy, xx], order=1)
    res = P - np.median(P, axis=1, keepdims=True)
    prof = (res * (r[:, None] ** 2)).mean(axis=0)
    ext = np.concatenate([prof, prof[: _N_THETA // 12]])
    pk, _ = find_peaks(ext, distance=_N_THETA // 24)
    pk = np.unique(pk % _N_THETA)
    sd = float(np.std(prof))
    if pk.size >= n_spikes and sd > 0:
        h = prof[pk]
        order = np.argsort(-h)[:n_spikes]
        top, heights = pk[order], h[order]
        if heights.min() > 0 and heights.mean() / sd >= _MIN_RIDGE_SNR:
            rel = np.clip(heights / heights.mean(), *_REL_CLIP)
            srt = np.argsort(top)
            return th[top[srt]], (rel / rel.mean())[srt]
    angles = np.arange(n_spikes) * (2.0 * np.pi / n_spikes)
    return angles, np.ones(n_spikes)


def far_field_flux(band: FarFieldBand, params: FarFieldParams, r_in: float,
                   r_out: float, *, r_max: float | None = None) -> float:
    """Expected far-field flux (fraction of the star's flux) of one star.

    Halo plus ``n_spikes`` ridges of mean relative height 1, weighted by the
    join complement ``1 - w(r)`` and truncated at ``r_max`` (default
    ``params.r_max_arcsec``). The ridge integral runs along its length with
    the Gaussian cross-section ``σ√(2π)``.
    """
    r_max = float(params.r_max_arcsec if r_max is None else r_max)
    if r_max <= r_in:
        return 0.0
    r = np.geomspace(r_in, r_max, 20001)
    wc = 1.0 - join_weight(r, r_in, r_out)
    per_area = 1.0 / params.pixel_scale_arcsec ** 2      # per pixel -> per arcsec²
    halo = band.halo_at(r) * 2.0 * np.pi * r
    ridge = (params.n_spikes * band.ridge_at(r)
             * band.spike_sigma_arcsec * np.sqrt(2.0 * np.pi))
    return float(np.trapezoid(wc * (halo + ridge), r) * per_area)


def draw_jitter(n: int, params: FarFieldParams, rng: np.random.Generator,
                ) -> tuple[np.ndarray, np.ndarray]:
    """Mean-one log-normal halo and spike amplitude factors for ``n`` stars."""
    sh, ss = params.halo_jitter_ln, params.spike_jitter_ln
    halo = np.exp(sh * rng.standard_normal(n) - 0.5 * sh * sh)
    spike = np.exp(ss * rng.standard_normal(n) - 0.5 * ss * ss)
    return halo, spike


def _ray_entry(y0: float, x0: float, dy: float, dx: float, h_max: float,
               w_max: float, pad: float) -> float | None:
    """Distance along the ray ``(y0, x0) + t (dy, dx)``, t >= 0, at which it
    enters the box ``[-pad, h_max + pad] x [-pad, w_max + pad]``; ``None``
    when it misses (slab method)."""
    t_lo, t_hi = 0.0, np.inf
    for p0, d, hi in ((y0, dy, h_max), (x0, dx, w_max)):
        lo_b, hi_b = -pad, hi + pad
        if abs(d) < 1e-12:
            if not lo_b <= p0 <= hi_b:
                return None
            continue
        t1, t2 = (lo_b - p0) / d, (hi_b - p0) / d
        t_lo, t_hi = max(t_lo, min(t1, t2)), min(t_hi, max(t1, t2))
    return t_lo if t_lo <= t_hi else None


def render_far_field(
    shape: tuple[int, int],
    ys: np.ndarray,
    xs: np.ndarray,
    fluxes: np.ndarray,
    band: FarFieldBand,
    params: FarFieldParams,
    *,
    pixel_scale_arcsec: float,
    angles: np.ndarray,
    rel: np.ndarray,
    r_in: float,
    r_out: float,
    halo_jitter: np.ndarray | None = None,
    spike_jitter: np.ndarray | None = None,
    min_peak_e: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Far-field light (electrons per pixel) of stars on a ``shape`` grid.

    ``ys, xs`` are star centres in pixel coordinates of this grid (pixel
    centres at integers), ``fluxes`` their total electrons in this band.
    Stars whose brightest far-field pixel stays below ``min_peak_e`` (one
    value, or one per star) are skipped, and each star is drawn only where
    it reaches that level: its halo inside the radius where the halo falls
    below it, each ridge along a strip (±4σ) out to where that ridge does.
    """
    out = np.zeros(shape, np.float64)
    n = len(fluxes)
    if n == 0:
        return out.astype(np.float32)
    hj = np.ones(n) if halo_jitter is None else np.asarray(halo_jitter, np.float64)
    sj = np.ones(n) if spike_jitter is None else np.asarray(spike_jitter, np.float64)
    ps = float(pixel_scale_arcsec)
    area = (ps / params.pixel_scale_arcsec) ** 2
    sig = band.spike_sigma_arcsec
    r_max = params.r_max_arcsec
    rg = np.geomspace(r_in, r_max, 256)
    halo_g, ridge_g = band.halo_at(rg), band.ridge_at(rg)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    rel = np.asarray(rel, np.float64)
    h_max, w_max = (shape[0] - 1) * ps, (shape[1] - 1) * ps
    floor = np.broadcast_to(np.asarray(min_peak_e, np.float64), (n,))

    def reach(curve: np.ndarray, amp: float, lo: float) -> float:
        above = amp * curve >= lo
        return float(rg[above].max()) if above.any() else 0.0

    def box(y0: float, y1: float, x0: float, x1: float):
        i0, i1 = max(0, int(np.floor(y0 / ps))), min(shape[0], int(np.ceil(y1 / ps)) + 1)
        j0, j1 = max(0, int(np.floor(x0 / ps))), min(shape[1], int(np.ceil(x1 / ps)) + 1)
        if i1 <= i0 or j1 <= j0:
            return None
        return (slice(i0, i1), slice(j0, j1),
                (np.arange(i0, i1, dtype=np.float64) * ps)[:, None],
                (np.arange(j0, j1, dtype=np.float64) * ps)[None, :])

    for y, x, f, h, s, lo in zip(ys, xs, fluxes, hj, sj, floor, strict=True):
        scale = float(f) * area
        yo, xo = float(y) * ps, float(x) * ps
        # Brightest far-field pixel: at r_in, or at the nearest grid pixel of
        # a star off the grid (its halo there, or a ridge where its ray
        # enters the grid, widened by 3σ).
        near = np.hypot(max(0.0, -yo, yo - h_max), max(0.0, -xo, xo - w_max))
        if near > r_max:
            continue
        if near > r_in:
            peak = scale * float(band.halo_at(np.array([near]))[0]) * h
            for c, sn, rk in zip(cos_a, sin_a, rel, strict=True):
                t = _ray_entry(yo, xo, sn, c, h_max, w_max, 3.0 * sig)
                if t is not None:
                    peak = max(peak, scale * float(band.ridge_at(np.array([max(t, r_in)]))[0]) * s * rk)
        else:
            peak = scale * max(halo_g[0] * h, ridge_g[0] * s * rel.max())
        if not peak > lo:
            continue
        # Halo, inside the radius where it still reaches ``lo``.
        rh = reach(halo_g, scale * h, lo)
        b = box(yo - rh, yo + rh, xo - rh, xo + rh) if rh > 0.0 else None
        if b is not None:
            sy_, sx_, gy, gx = b
            r = np.hypot(gy - yo, gx - xo)
            m = (r >= r_in) & (r <= rh)
            if m.any():
                halo = scale * h * band.halo_at(np.maximum(r, r_in))
                out[sy_, sx_] += np.where(m, halo * (1.0 - join_weight(r, r_in, r_out)), 0.0)
        # Ridges, each along its own strip.
        for c, sn, rk in zip(cos_a, sin_a, rel, strict=True):
            rr = reach(ridge_g, scale * s * rk, lo)
            if rr <= 0.0:
                continue
            pad = 4.0 * sig
            ya, yb = yo + r_in * sn, yo + rr * sn
            xa, xb = xo + r_in * c, xo + rr * c
            b = box(min(ya, yb) - pad, max(ya, yb) + pad, min(xa, xb) - pad, max(xa, xb) + pad)
            if b is None:
                continue
            sy_, sx_, gy, gx = b
            dy, dx = gy - yo, gx - xo
            along = dx * c + dy * sn
            perp = dy * c - dx * sn
            r = np.hypot(dy, dx)
            m = (along > 0.0) & (along <= rr) & (np.abs(perp) <= pad) & (r >= r_in) & (r <= r_max)
            if m.any():
                out[sy_, sx_] += np.where(
                    m, scale * s * rk * band.ridge_at(np.maximum(along, r_in))
                    * np.exp(-0.5 * (perp / sig) ** 2) * (1.0 - join_weight(r, r_in, r_out)), 0.0)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Off-field stars
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OffFieldPrior:
    """Bright stars around the field: DR1 Wide counts per magnitude bin and
    measured NISP/VIS electron ratios (``bright_star_params.json``)."""

    mag_lo: np.ndarray          # bin edges (VIS)
    mag_hi: np.ndarray
    density_deg2: np.ndarray    # stars per deg² in each bin
    colour_bands: tuple[str, ...]
    colour_ratios: np.ndarray   # (n_stars, len(colour_bands)) electrons / VIS electrons

    @classmethod
    def from_payload(cls, payload: dict) -> OffFieldPrior:
        off, counts = payload["off_field"], payload["star_counts"]
        edges = np.asarray(counts["mag_edges"], np.float64)
        dens = np.asarray(counts["density_deg2_mag"], np.float64) * np.diff(edges)
        use = (edges[:-1] >= float(off["mag_min"]) - 1e-9) & (edges[1:] <= float(off["mag_max"]) + 1e-9)
        return cls(mag_lo=edges[:-1][use], mag_hi=edges[1:][use], density_deg2=dens[use],
                   colour_bands=tuple(off["colour_ratio_bands"]),
                   colour_ratios=np.asarray(off["colour_ratios"], np.float64))


@lru_cache(maxsize=4)
def load_off_field_prior(path: str = PARAMS_PATH) -> OffFieldPrior:
    with open(path, encoding="utf-8") as fh:
        return OffFieldPrior.from_payload(json.load(fh))


def ridge_reach_arcsec(band: FarFieldBand, params: FarFieldParams, flux_e: float,
                       threshold_e: float) -> float:
    """Largest radius at which the brightest possible ridge of a ``flux_e``
    star (relative height 2, a +2σ amplitude draw) still reaches
    ``threshold_e`` per pixel; at most ``params.r_max_arcsec``."""
    r = np.geomspace(0.5, params.r_max_arcsec, 2000)
    boost = 2.0 * np.exp(2.0 * params.spike_jitter_ln)
    above = flux_e * boost * band.ridge_at(r) >= threshold_e
    return float(r[above].max()) if above.any() else 0.0


def draw_off_field_stars(
    rng: np.random.Generator,
    prior: OffFieldPrior,
    reach_arcsec: np.ndarray,
    field_shape: tuple[int, int],
    pixel_scale_arcsec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stars outside a ``field_shape`` grid that can reach into it.

    Per magnitude bin, stars are Poisson-drawn at the DR1 density over the
    square ring between the field and ``reach_arcsec[bin]`` beyond its edges,
    uniform in position and magnitude. Returns grid coordinates ``ys, xs``
    (pixel centres at integers, outside the grid), VIS magnitudes and one
    row index into ``prior.colour_ratios`` per star.
    """
    H, W = field_shape
    hh, hw = 0.5 * H * pixel_scale_arcsec, 0.5 * W * pixel_scale_arcsec
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    ys, xs, mags = [], [], []
    for lo, hi, dens, reach in zip(prior.mag_lo, prior.mag_hi, prior.density_deg2,
                                   reach_arcsec, strict=True):
        if reach <= 0.0 or dens <= 0.0:
            continue
        oy, ox = hh + reach, hw + reach
        ring_deg2 = (4.0 * oy * ox - 4.0 * hh * hw) / 3600.0 ** 2
        n = int(rng.poisson(dens * ring_deg2))
        got = 0
        while got < n:
            py = rng.uniform(-oy, oy, size=2 * (n - got) + 8)
            px = rng.uniform(-ox, ox, size=py.size)
            ok = (np.abs(py) >= hh) | (np.abs(px) >= hw)
            take = np.flatnonzero(ok)[: n - got]
            ys.extend(cy + py[take] / pixel_scale_arcsec)
            xs.extend(cx + px[take] / pixel_scale_arcsec)
            got += take.size
        mags.extend(rng.uniform(lo, hi, size=n))
    k = len(mags)
    colours = rng.integers(0, len(prior.colour_ratios), size=k)
    return (np.asarray(ys, np.float64), np.asarray(xs, np.float64),
            np.asarray(mags, np.float64), colours)


def draw_infield_bright_magnitudes(rng: np.random.Generator, prior: OffFieldPrior,
                                   mag_min: float, mag_max: float,
                                   area_deg2: float) -> np.ndarray:
    """VIS magnitudes of the stars brighter than the regular stellar law
    (``mag_min`` <= m < ``mag_max``) that fall inside a field of
    ``area_deg2``: Poisson per DR1 count bin, uniform within the bin."""
    if not mag_min < mag_max:
        return np.zeros(0)
    mags = []
    for lo, hi, dens in zip(prior.mag_lo, prior.mag_hi, prior.density_deg2, strict=True):
        a, b = max(lo, mag_min), min(hi, mag_max)
        if b <= a or dens <= 0.0:
            continue
        n = int(rng.poisson(dens * (b - a) / (hi - lo) * area_deg2))
        mags.extend(rng.uniform(a, b, size=n))
    return np.asarray(mags, np.float64)


# --------------------------------------------------------------------------- #
# Blank trails of bright stars
# --------------------------------------------------------------------------- #

def _pair(v) -> tuple[float, float]:
    return float(v[0]), float(v[1])


@dataclass(frozen=True)
class TrailParams:
    """VIS bleed and NISP persistence trails (``bright_star_params.json``)."""

    axis_offset_deg: float
    vis_mag: np.ndarray
    vis_length: np.ndarray
    vis_width: np.ndarray
    vis_sd_ln: np.ndarray
    vis_split: tuple[float, float]
    nisp_mag: np.ndarray
    nisp_prob: np.ndarray
    nisp_length: tuple[float, float]
    nisp_split: tuple[float, float]
    nisp_width: dict[str, tuple[float, float]]
    nisp_bow_at_50: tuple[float, float]
    nisp_axis_scatter_deg: float

    @classmethod
    def from_payload(cls, payload: dict) -> TrailParams:
        t = payload["trails"]
        vis = t["vis"]["table"]
        nisp = t["nisp"]
        return cls(
            axis_offset_deg=float(t["axis_offset_deg"]),
            vis_mag=np.array([r["mag"] for r in vis], np.float64),
            vis_length=np.array([r["length_arcsec"] for r in vis], np.float64),
            vis_width=np.array([r["width_arcsec"] for r in vis], np.float64),
            vis_sd_ln=np.clip(np.array([r["length_sd_ln"] for r in vis], np.float64), 0.2, 0.6),
            vis_split=_pair(t["vis"]["split_fraction"]),
            nisp_mag=np.array([r["mag"] for r in nisp["probability_table"]], np.float64),
            nisp_prob=np.array([r["probability"] for r in nisp["probability_table"]], np.float64),
            nisp_length=_pair(nisp["length_arcsec"]),
            nisp_split=_pair(nisp["split_fraction"]),
            nisp_width={b: _pair(v) for b, v in nisp["width_arcsec"].items()},
            nisp_bow_at_50=_pair(nisp["bow_arcsec_at_50"]),
            nisp_axis_scatter_deg=float(nisp["axis_scatter_deg"]),
        )

    @property
    def vis_mag_max(self) -> float:
        return float(self.vis_mag[-1] + 0.5)

    @property
    def nisp_mag_max(self) -> float:
        nz = self.nisp_mag[self.nisp_prob > 0]
        return float(nz.max() + 0.25) if nz.size else -np.inf

    def vis_length_at(self, mag: float) -> float:
        return float(np.exp(np.interp(mag, self.vis_mag, np.log(self.vis_length))))

    def nisp_probability_at(self, mag: float) -> float:
        if mag > self.nisp_mag[-1] + 0.25:
            return 0.0
        return float(np.interp(mag, self.nisp_mag, self.nisp_prob))


@lru_cache(maxsize=4)
def load_trail_params(path: str = PARAMS_PATH) -> TrailParams:
    with open(path, encoding="utf-8") as fh:
        return TrailParams.from_payload(json.load(fh))


def trail_axis(angles: np.ndarray, rel: np.ndarray, offset_deg: float) -> float:
    """Detector-column (trail) axis of a scene: its kernel's strongest spike
    axis (the pair of opposite ridges with the largest summed height) plus
    the measured offset. Radians, array frame, modulo π."""
    axes = np.mod(np.asarray(angles, np.float64), np.pi)
    rel = np.asarray(rel, np.float64)
    group = np.round(((axes - axes[0]) % np.pi) / (np.pi / 3.0)).astype(int) % 3
    heights = np.bincount(group, weights=rel, minlength=3)
    members = np.flatnonzero(group == int(np.argmax(heights)))
    strongest = axes[members[np.argmax(rel[members])]]
    return float((strongest + np.radians(offset_deg)) % np.pi)


def strip_mask(shape: tuple[int, int], y: float, x: float, axis: float,
               back: float, ahead: float, width: float, *, bow_at_50: float = 0.0,
               pixel_scale_arcsec: float) -> np.ndarray:
    """Pixels of a strip through ``(y, x)`` (grid pixels) along ``axis``,
    from ``back`` arcsec behind to ``ahead`` arcsec in front of the star,
    ``width`` arcsec wide, bowed sideways by ``bow_at_50 (a / 50″)²``."""
    ps = pixel_scale_arcsec
    gy = (np.arange(shape[0], dtype=np.float64)[:, None] - y) * ps
    gx = (np.arange(shape[1], dtype=np.float64)[None, :] - x) * ps
    c, s = np.cos(axis), np.sin(axis)
    along = gx * c + gy * s
    perp = gy * c - gx * s - bow_at_50 * (along / 50.0) ** 2
    return (along >= -back) & (along <= ahead) & (np.abs(perp) <= 0.5 * width)


def draw_trail_masks(
    rng: np.random.Generator,
    params: TrailParams,
    shape: tuple[int, int],
    ys: np.ndarray,
    xs: np.ndarray,
    vis_mags: np.ndarray,
    axis: float,
    *,
    band_names: tuple[str, ...],
    pixel_scale_arcsec: float,
) -> np.ndarray:
    """Blank trails ``(H, W, n_bands)`` of stars at grid positions ``ys, xs``
    (may lie off the grid) with VIS magnitudes ``vis_mags``.

    VIS: a straight bleed trail of the measured length (log-normal scatter)
    and width, the star at a random split along it. NISP: with the measured
    magnitude-dependent probability, a ~150″ persistence trail through the
    star, bowed (one bow sign per scene), the same strip in Y, J and H with
    each band's width. Both run along the scene's detector-column ``axis``.
    """
    out = np.zeros((*shape, len(band_names)), bool)
    k_vis = band_names.index("VIS")
    nisp = [(band_names.index(b), b) for b in params.nisp_width]
    ps = pixel_scale_arcsec
    half_diag = 0.5 * np.hypot(*shape) * ps
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    bow_sign = 1.0 if rng.random() < 0.5 else -1.0
    order = np.argsort(vis_mags, kind="stable")
    for i in order:
        m, y, x = float(vis_mags[i]), float(ys[i]), float(xs[i])
        if m >= max(params.vis_mag_max, params.nisp_mag_max):
            break
        dist = np.hypot(y - cy, x - cx) * ps
        if m < params.vis_mag_max:
            sd = float(np.interp(m, params.vis_mag, params.vis_sd_ln))
            length = params.vis_length_at(m) * float(np.exp(sd * rng.standard_normal()))
            f = float(rng.uniform(*params.vis_split))
            width = float(np.interp(m, params.vis_mag, params.vis_width))
            if dist - half_diag <= length:
                out[..., k_vis] |= strip_mask(shape, y, x, axis, f * length, (1.0 - f) * length,
                                              width, pixel_scale_arcsec=ps)
        if rng.random() < params.nisp_probability_at(m):
            length = float(rng.uniform(*params.nisp_length))
            f = float(rng.uniform(*params.nisp_split))
            bow = bow_sign * float(rng.uniform(*params.nisp_bow_at_50))
            ax = axis + np.radians(params.nisp_axis_scatter_deg) * float(rng.standard_normal())
            widths = [float(rng.uniform(*params.nisp_width[b])) for _, b in nisp]
            if dist - half_diag <= length:
                for (k, _), w in zip(nisp, widths, strict=True):
                    out[..., k] |= strip_mask(shape, y, x, ax, f * length, (1.0 - f) * length, w,
                                              bow_at_50=bow, pixel_scale_arcsec=ps)
    return out


# --------------------------------------------------------------------------- #
# Recipe string (training / generation switches)
# --------------------------------------------------------------------------- #

#: Pieces of a ``bright_stars`` recipe, in canonical order.
RECIPE_PIECES = ("far", "nisp_taper", "off_field", "trails", "infield")


def parse_bright_star_recipe(recipe: str | None) -> dict:
    """Forward-model switches of a ``bright_stars`` recipe.

    ``"off"`` (or empty) is today's forward model. Otherwise a comma list of
    ``far`` (far field), ``nisp_taper``, ``off_field`` (needs ``far``),
    ``trails`` (needs blank pixels; the caller checks) and ``infield`` or
    ``infield=<VIS cap>``. Returns the keyword arguments of
    :class:`~euclid_polish.training.forward_onthefly.OnTheFlyForward`:
    ``bright_star_far_field``, ``nisp_psf_taper``, ``off_field_stars``,
    ``bright_star_trails``, ``infield_bright_stars``, ``infield_bright_mag_min``.
    """
    out = {"bright_star_far_field": False, "nisp_psf_taper": False, "off_field_stars": False,
           "bright_star_trails": False, "infield_bright_stars": False,
           "infield_bright_mag_min": None}
    text = (recipe or "off").strip().lower()
    if text in ("", "off", "none"):
        return out
    for raw in text.split(","):
        item = raw.strip()
        name, _, value = item.partition("=")
        if name not in RECIPE_PIECES or (value and name != "infield"):
            raise ValueError(f"bright_stars: unknown piece {item!r} (pieces: "
                             f"{', '.join(RECIPE_PIECES)}, infield=<VIS cap>, or 'off')")
        if name == "far":
            out["bright_star_far_field"] = True
        elif name == "nisp_taper":
            out["nisp_psf_taper"] = True
        elif name == "off_field":
            out["off_field_stars"] = True
        elif name == "trails":
            out["bright_star_trails"] = True
        else:
            out["infield_bright_stars"] = True
            if value:
                try:
                    cap = float(value)
                except ValueError as exc:
                    raise ValueError("bright_stars: infield cap must be a VIS magnitude, "
                                     f"got {value!r}") from exc
                if not 5.0 <= cap < 12.0:
                    raise ValueError(f"bright_stars: infield cap must be in [5, 12), got {cap:g}")
                out["infield_bright_mag_min"] = cap
    if out["off_field_stars"] and not out["bright_star_far_field"]:
        raise ValueError("bright_stars: off_field needs far")
    return out


def canonical_bright_star_recipe(recipe: str | None, *, default_cap: float) -> str:
    """The recipe in canonical form (``"off"`` or pieces in fixed order, the
    infield cap written out), for origin.json and identity checks."""
    kw = parse_bright_star_recipe(recipe)
    names = [("far", "bright_star_far_field"), ("nisp_taper", "nisp_psf_taper"),
             ("off_field", "off_field_stars"), ("trails", "bright_star_trails")]
    parts = [p for p, k in names if kw[k]]
    if kw["infield_bright_stars"]:
        cap = kw["infield_bright_mag_min"]
        parts.append(f"infield={default_cap if cap is None else cap:g}")
    return ",".join(parts) if parts else "off"
