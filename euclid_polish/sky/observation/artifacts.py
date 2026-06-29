"""Detector artifact injection: cosmic rays + hot pixels + masked-trail streaks.

Real Euclid LR exposures contain features that pure Poisson + read-noise
does not reproduce. Three classes are modelled here:

* **Cosmic rays** — galactic-cosmic-ray (GCR) muons and protons hit the
  detector at ~5 hits/cm²/s at L2. Each hit deposits charge in one or a
  few neighbouring pixels; the full track geometry is approximated as
  short oriented streaks of length 1–4 native pixels. Most are removed
  by the MER cross-dither median; this models the post-rejection survivors.
* **Hot pixels** — ~0.1% of detector pixels show anomalously high dark
  current and effectively saturate over a typical exposure.
* **Dead pixels** — a rarer set of unresponsive pixels collect ~no charge,
  reading at the background floor (≈0) regardless of incident light; their
  real signal is destroyed so the network must inpaint from neighbours.
* **Long faint streaks** — the MER pipeline (Q1 paper, Romelli+ 2025
  arXiv:2503.15305) masks pixels affected by long oblique CR trails,
  satellite/asteroid trails, and NISP persistence, then interpolates
  across the mask. The interpolated values are smooth (no shot noise),
  so the masked stripe appears as a sub-σ but visually distinct ridge
  in the final image. Roughly 30–50% of VIS Q1 cutouts show at least
  one such streak.

All three are injected *after* the Poisson shot-noise stage but *before*
read noise, matching the physical readout order (charge integrates →
pixels are read → read noise added).

Rates are quoted per native detector pixel (12 µm VIS, 18 µm NISP). The
caller passes the band, which sets the pixel area to scale to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from euclid_polish.config import BandConfig, Config


@dataclass
class ArtifactConfig:
    """Per-frame artifact injection knobs.

    Defaults come from :class:`Config` and follow flight measurements
    (Schirmer+ 2025 NISP Instrument paper; Holmes+ 2012 SREM CR rate).
    Disable a class of artifacts by setting its ``add_*`` flag to False.
    """

    add_cosmic_rays:        bool  = True
    cr_rate_per_s_per_cm2:  float = Config.CR_RATE_PER_S_PER_CM2
    cr_charge_median_e:     float = Config.CR_CHARGE_MEDIAN_E
    # Track length is drawn from Exp(mean=cr_track_length_mean) and
    # rounded to int (≥1). Most CR hits are short (1-2 native pixels),
    # but oblique-incidence GCR muons leave long tracks — the exponential
    # tail captures both regimes. Capped at cr_max_track_length to
    # bound the per-hit work.
    cr_track_length_mean:   float = 3.0             # native pixels
    cr_max_track_length:    int   = 25              # native pixels

    add_hot_pixels:         bool  = True
    hot_pixel_fraction:     float = Config.HOT_PIXEL_FRACTION
    hot_pixel_charge_mean_e: float = Config.HOT_PIXEL_CHARGE_MEAN_E

    # Dead / unresponsive pixels (see :func:`inject_dead_pixels`).
    add_dead_pixels:         bool  = True
    dead_pixel_fraction:     float = Config.DEAD_PIXEL_FRACTION
    dead_pixel_jitter_sigma: float = Config.DEAD_PIXEL_JITTER_SIGMA

    # Long faint masked-trail streaks (see module docstring).
    add_streaks:               bool  = True
    streak_rate_per_kpix2:     float = Config.STREAK_RATE_PER_KPIX2
    streak_length_mean_pix:    float = Config.STREAK_LENGTH_MEAN_PIX
    streak_length_min_pix:     int   = Config.STREAK_LENGTH_MIN_PIX
    streak_length_max_pix:     int   = Config.STREAK_LENGTH_MAX_PIX
    streak_width_pix_min:      int   = Config.STREAK_WIDTH_PIX_MIN
    streak_width_pix_max:      int   = Config.STREAK_WIDTH_PIX_MAX
    streak_amp_sigma_min:      float = Config.STREAK_AMP_SIGMA_MIN
    streak_amp_sigma_max:      float = Config.STREAK_AMP_SIGMA_MAX

    def __post_init__(self) -> None:
        if self.cr_rate_per_s_per_cm2 < 0:
            raise ValueError("cr_rate_per_s_per_cm2 must be ≥ 0")
        if self.cr_charge_median_e <= 0:
            raise ValueError("cr_charge_median_e must be > 0")
        if self.cr_track_length_mean <= 0:
            raise ValueError("cr_track_length_mean must be > 0")
        if not (0.0 <= self.hot_pixel_fraction <= 1.0):
            raise ValueError("hot_pixel_fraction must be in [0, 1]")
        if self.hot_pixel_charge_mean_e <= 0:
            raise ValueError("hot_pixel_charge_mean_e must be > 0")
        if not (0.0 <= self.dead_pixel_fraction <= 1.0):
            raise ValueError("dead_pixel_fraction must be in [0, 1]")
        if self.dead_pixel_jitter_sigma < 0:
            raise ValueError("dead_pixel_jitter_sigma must be ≥ 0")
        if self.cr_max_track_length < 1:
            raise ValueError("cr_max_track_length must be ≥ 1")
        if self.streak_rate_per_kpix2 < 0:
            raise ValueError("streak_rate_per_kpix2 must be ≥ 0")
        if not (1 <= self.streak_length_min_pix <= self.streak_length_max_pix):
            raise ValueError("require 1 ≤ streak_length_min_pix ≤ streak_length_max_pix")
        if self.streak_length_mean_pix < self.streak_length_min_pix:
            raise ValueError("streak_length_mean_pix must be ≥ streak_length_min_pix")
        if not (1 <= self.streak_width_pix_min <= self.streak_width_pix_max):
            raise ValueError("require 1 ≤ streak_width_pix_min ≤ streak_width_pix_max")
        if not (0.0 <= self.streak_amp_sigma_min <= self.streak_amp_sigma_max):
            raise ValueError("require 0 ≤ streak_amp_sigma_min ≤ streak_amp_sigma_max")


# ---------------------------------------------------------------------------
# Cosmic-ray injection
# ---------------------------------------------------------------------------

def _native_pix_arcsec(band: BandConfig) -> float:
    """Native detector pixel scale in arcsec for this band's instrument.

    VIS CCD = 0.10″/pix, NISP H2RG = 0.30″/pix. Single source of truth is
    :attr:`BandConfig.native_detector_scale_arcsec`.
    """
    return band.native_detector_scale_arcsec


def expected_cosmic_ray_count(
    image_shape: tuple[int, int],
    band: BandConfig,
    cfg: ArtifactConfig,
) -> float:
    """Expected number of CR hits for one frame at this band's exposure.

    ``image_shape`` is in LR archive pixels (typically 0.10″/pix). VIS 1:1,
    NISP 1:9.
    """
    if not cfg.add_cosmic_rays or cfg.cr_rate_per_s_per_cm2 == 0:
        return 0.0
    H, W = image_shape
    # Detector pixel area in cm² (1 µm = 1e-4 cm).
    det_area_cm2 = (band.detector_pixel_um * 1e-4) ** 2
    # Number of *native* detector pixels covered by the frame.
    pix_ratio   = _native_pix_arcsec(band) / band.pixel_scale_lr_arcsec
    n_native_pixels = (H * W) / (pix_ratio ** 2)
    t_total = band.exposure_time_s * band.n_exposures
    # Per-band post-rejection factor: VIS has aggressive cross-dither CR
    # rejection (~98% killed → factor 0.02); NISP keeps most hits.
    rejection = float(getattr(band, "cr_rate_factor", 1.0))
    return cfg.cr_rate_per_s_per_cm2 * det_area_cm2 * n_native_pixels * t_total * rejection


def inject_cosmic_rays(
    image_e: np.ndarray,
    band: BandConfig,
    rng: np.random.Generator,
    cfg: ArtifactConfig,
) -> np.ndarray:
    """Add CR hits to ``image_e`` (in-place safe, returns array).

    Hits are placed at uniform-random pixel locations on the LR grid
    with the count drawn from a Poisson distribution. Each hit's
    deposited charge is exponential with scale ``cr_charge_median_e``
    and is spread along a randomly oriented short track of length
    1–``cr_max_track_length`` native pixels (clipped to the image).
    """
    if not cfg.add_cosmic_rays:
        return image_e
    H, W = image_e.shape
    mean_n = expected_cosmic_ray_count((H, W), band, cfg)
    if mean_n <= 0:
        return image_e
    n_hits = int(rng.poisson(mean_n))
    if n_hits == 0:
        return image_e

    out = image_e.astype(np.float64, copy=True)
    # Random starting pixels.
    xs = rng.integers(0, W, size=n_hits)
    ys = rng.integers(0, H, size=n_hits)
    charges = rng.exponential(scale=cfg.cr_charge_median_e, size=n_hits)
    # Random orientations (radians); track length ~ Exp(mean) clamped to
    # [1, max_track_length]. The exponential gives a heavy tail so most
    # hits are 1-3 px (perpendicular incidence) but a few reach 15-25 px
    # (oblique muons traversing the depleted layer at shallow angles).
    thetas  = rng.uniform(0.0, np.pi, size=n_hits)
    raw_len = rng.exponential(scale=cfg.cr_track_length_mean, size=n_hits)
    lengths = np.clip(np.round(raw_len).astype(int),
                      1, cfg.cr_max_track_length)

    for x0, y0, q, theta, L in zip(xs, ys, charges, thetas, lengths, strict=False):
        # Distribute charge equally along the track for L > 1.
        per_step = q / max(L, 1)
        dx = np.cos(theta)
        dy = np.sin(theta)
        for k in range(L):
            xi = int(round(x0 + k * dx))
            yi = int(round(y0 + k * dy))
            if 0 <= xi < W and 0 <= yi < H:
                out[yi, xi] += per_step
    return out.astype(image_e.dtype, copy=False)


# ---------------------------------------------------------------------------
# Hot pixels
# ---------------------------------------------------------------------------

def inject_hot_pixels(
    image_e: np.ndarray,
    rng: np.random.Generator,
    cfg: ArtifactConfig,
) -> np.ndarray:
    """Add hot pixels at random locations to ``image_e``.

    Each pixel independently becomes hot with probability
    ``cfg.hot_pixel_fraction``. Hot pixels get an additive charge drawn
    from ``Exponential(mean=hot_pixel_charge_mean_e)`` — this models the
    cumulative dark current of a defective pixel filling toward
    saturation over the integration.

    Note: a real detector has a *fixed* hot-pixel mask; here the positions
    are randomized per frame so the network learns position-independent
    robustness instead of memorising the mask.
    """
    if not cfg.add_hot_pixels or cfg.hot_pixel_fraction <= 0:
        return image_e
    mask = rng.random(image_e.shape) < cfg.hot_pixel_fraction
    if not mask.any():
        return image_e
    n = int(mask.sum())
    charges = rng.exponential(scale=cfg.hot_pixel_charge_mean_e, size=n).astype(image_e.dtype)
    out = image_e.copy()
    out[mask] = out[mask] + charges
    return out


# ---------------------------------------------------------------------------
# Dead / unresponsive pixels
# ---------------------------------------------------------------------------

def inject_dead_pixels(
    image_e: np.ndarray,
    rng: np.random.Generator,
    cfg: ArtifactConfig,
    local_sigma_e: float = 0.0,
) -> np.ndarray:
    """Zero out random pixels, modelling dead/unresponsive detector pixels.

    Each pixel independently goes dead with probability
    ``cfg.dead_pixel_fraction``. A dead pixel collects ~no charge, so it reads
    at the background floor (≈0 after sky/dark subtraction) no matter what
    light fell on it — its real signal is **destroyed** (replacement, not
    additive), forcing the network to inpaint the value from its neighbours
    (and, as a side effect, to learn the local pixel correlations).

    The replacement value is drawn from ``N(0, dead_pixel_jitter_sigma ·
    local_sigma_e)`` rather than being exactly zero, so the model keys on the
    spatial anomaly instead of memorising a magic constant. (Read noise, added
    downstream, jitters it further.) ``local_sigma_e <= 0`` → exactly zero.

    Positions are randomized per frame (like the hot-pixel mask) so the network
    learns position-independent robustness, not a fixed dead-pixel map.
    """
    if not cfg.add_dead_pixels or cfg.dead_pixel_fraction <= 0:
        return image_e
    mask = rng.random(image_e.shape) < cfg.dead_pixel_fraction
    if not mask.any():
        return image_e
    n = int(mask.sum())
    sigma = max(0.0, cfg.dead_pixel_jitter_sigma * float(local_sigma_e))
    vals = (rng.normal(0.0, sigma, size=n) if sigma > 0
            else np.zeros(n)).astype(image_e.dtype)
    out = image_e.copy()
    out[mask] = vals
    return out


# ---------------------------------------------------------------------------
# Long faint masked-trail streaks
# ---------------------------------------------------------------------------

def expected_streak_count(
    image_shape: tuple[int, int],
    cfg: ArtifactConfig,
) -> float:
    """Mean number of streaks per frame for this image area.

    ``streak_rate_per_kpix2`` is normalised to a (1000×1000)-pixel area so
    the same setting applies regardless of the actual LR frame size.
    """
    if not cfg.add_streaks or cfg.streak_rate_per_kpix2 == 0:
        return 0.0
    H, W = image_shape
    return cfg.streak_rate_per_kpix2 * (H * W) / 1.0e6


def inject_streaks(
    image_e: np.ndarray,
    rng: np.random.Generator,
    cfg: ArtifactConfig,
    local_sigma_e: float,
) -> np.ndarray:
    """Add long faint additive ridges that mimic MER-masked trail interpolation.

    Each streak has:
      - random midpoint anywhere in the frame
      - random orientation in [0, π) (no preferred direction — these come
        from cosmic rays, satellites, asteroids, persistence — all unrelated
        to the readout axis)
      - exponentially distributed length, clamped to [min, max]
      - integer width drawn uniform on [w_min, w_max]
      - signed amplitude ``A`` drawn from
        ``Uniform([sigma_min, sigma_max]) · sign · local_sigma_e``
      - Gaussian cross-section across the spine (full width ≈ width_pix)

    ``local_sigma_e`` is the per-pixel noise RMS at this point in the
    pipeline (sky + dark + read shot noise quadrature). The caller computes
    it once per band; this function does not re-derive it from the image,
    since the image at this stage is sky/dark-subtracted and a per-pixel
    estimate would be biased by sources.

    The amplitude is intentionally sub-σ (typically 0.3–0.8 σ): the
    streaks are invisible at normal stretches and only emerge under tight
    background clipping, matching the real Q1 cutouts.
    """
    if not cfg.add_streaks:
        return image_e
    H, W = image_e.shape
    mean_n = expected_streak_count((H, W), cfg)
    if mean_n <= 0:
        return image_e
    n_streaks = int(rng.poisson(mean_n))
    if n_streaks == 0:
        return image_e

    out = image_e.astype(np.float64, copy=True)
    for _ in range(n_streaks):
        # Geometry: midpoint anywhere, angle anywhere, length exponential.
        cy = rng.uniform(0.0, H)
        cx = rng.uniform(0.0, W)
        theta = rng.uniform(0.0, np.pi)
        raw_len = rng.exponential(scale=cfg.streak_length_mean_pix)
        L = float(np.clip(raw_len,
                          cfg.streak_length_min_pix,
                          cfg.streak_length_max_pix))
        width = int(rng.integers(cfg.streak_width_pix_min,
                                 cfg.streak_width_pix_max + 1))
        # Amplitude: uniform in [min, max] σ, sign ±1.
        amp_sigma = rng.uniform(cfg.streak_amp_sigma_min, cfg.streak_amp_sigma_max)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        peak_e = sign * amp_sigma * local_sigma_e

        # Walk the spine in one-pixel-per-step increments along the
        # dominant axis. Stepping ``1/max(|dx|,|dy|)`` along the spine
        # yields ~one new integer pixel per step regardless of angle, so
        # consecutive steps do not land in the same integer cell (cos/sin
        # steps along a 45° spine would revisit every other cell).
        dx = np.cos(theta)
        dy = np.sin(theta)
        n_unit = max(abs(dx), abs(dy), 1e-9)
        n_steps = int(np.ceil(L * n_unit))
        step_scale = 1.0 / n_unit
        sigma_cross = max(width / 2.355, 0.6)   # FWHM ≈ width
        half_cross = max(1, int(np.ceil(3.0 * sigma_cross)))
        px, py = -dy, dx
        for k in range(n_steps):
            t = (k - n_steps / 2.0) * step_scale
            yc = cy + t * dy
            xc = cx + t * dx
            # Within one spine step, deposit at most once per integer
            # pixel; otherwise a narrow Gaussian cross-section (sigma_cross
            # < 1) puts all j-offsets into the same int cell, stacking
            # depositions to exceed peak_e.
            seen: set[tuple[int, int]] = set()
            for j in range(-half_cross, half_cross + 1):
                yi = int(round(yc + j * py))
                xi = int(round(xc + j * px))
                if (yi, xi) in seen:
                    continue
                seen.add((yi, xi))
                if 0 <= xi < W and 0 <= yi < H:
                    w = np.exp(-0.5 * (j / sigma_cross) ** 2)
                    out[yi, xi] += peak_e * w
    return out.astype(image_e.dtype, copy=False)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def inject_artifacts(
    image_e: np.ndarray,
    band: BandConfig,
    rng: np.random.Generator,
    cfg: ArtifactConfig | None = None,
    local_sigma_e: float = 0.0,
) -> np.ndarray:
    """Apply the full artifact stack (CR + hot pixels + streaks) to one band frame.

    ``local_sigma_e`` is the per-pixel noise RMS for the current band's
    LR grid (Poisson sky + dark + read quadrature); needed by
    :func:`inject_streaks` to calibrate its sub-σ amplitude. Pass 0.0 to
    disable streaks even if ``cfg.add_streaks`` is True.
    """
    cfg = cfg or ArtifactConfig()
    if cfg.add_streaks and local_sigma_e <= 0:
        raise ValueError(
            "inject_artifacts: cfg.add_streaks is True but local_sigma_e <= 0; "
            "provide a positive per-pixel noise RMS or set add_streaks=False."
        )
    out = inject_cosmic_rays(image_e, band, rng, cfg)
    out = inject_hot_pixels(out, rng, cfg)
    out = inject_streaks(out, rng, cfg, local_sigma_e)
    # Dead pixels LAST: an unresponsive pixel reads ~0 regardless of any CR /
    # hot / streak charge that "landed" on it, so it overwrites them.
    out = inject_dead_pixels(out, rng, cfg, local_sigma_e)
    return out
