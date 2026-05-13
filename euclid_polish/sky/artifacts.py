"""Detector artifact injection: cosmic rays + hot pixels.

Real Euclid LR exposures contain features that pure Poisson + read-noise
does not reproduce. The two most-impactful ones for VIS/NISP wide-survey
imaging are:

* **Cosmic rays** — galactic-cosmic-ray (GCR) muons and protons hit the
  detector at ~5 hits/cm²/s at L2. Each hit deposits charge in one or a
  few neighbouring pixels (full track geometry is approximated here as
  short oriented streaks of length 1–4 native pixels).
* **Hot pixels** — ~0.1% of detector pixels show anomalously high dark
  current and effectively saturate over a typical exposure.

Both are injected *after* the Poisson shot-noise stage but *before* read
noise — that matches the physical readout order (charge integrates →
pixels are read → read noise added).

Rates are quoted per native detector pixel (12 µm VIS, 18 µm NISP). The
caller passes the band so we scale to the correct pixel area.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from euclid_polish.config import BandConfig, Config


@dataclass
class ArtifactConfig:
    """Per-frame artifact injection knobs.

    Defaults come from :class:`Config` and match flight measurements
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
        if self.cr_max_track_length < 1:
            raise ValueError("cr_max_track_length must be ≥ 1")


# ---------------------------------------------------------------------------
# Cosmic-ray injection
# ---------------------------------------------------------------------------

def _native_pix_arcsec(band: BandConfig) -> float:
    """Native detector pixel scale in arcsec for this band's instrument.

    VIS CCD = 0.10″/pix, NISP H2RG = 0.30″/pix — distinguished by the
    18 µm pitch of the H2RGs versus the 12 µm VIS pixels.
    """
    return 0.30 if band.detector_pixel_um >= 17.0 else 0.10


def expected_cosmic_ray_count(
    image_shape: tuple[int, int],
    band: BandConfig,
    cfg: ArtifactConfig,
) -> float:
    """Expected number of CR hits for one frame at this band's exposure.

    ``image_shape`` is in *LR archive pixels* (typically 0.10″/pix). We
    convert to the equivalent native detector area: VIS is 1:1 with its
    12 µm pixel, NISP is 1:9 (each archive 0.10″ pixel covers 1/9 of a
    native 18 µm 0.30″ pixel).
    """
    if not cfg.add_cosmic_rays or cfg.cr_rate_per_s_per_cm2 == 0:
        return 0.0
    H, W = image_shape
    # Detector pixel area in cm² (1 µm = 1e-4 cm).
    det_area_cm2 = (band.detector_pixel_um * 1e-4) ** 2
    # Number of *native* detector pixels covered by the LR frame.
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

    for x0, y0, q, theta, L in zip(xs, ys, charges, thetas, lengths):
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

    Note: a real detector has a *fixed* hot-pixel mask; for training we
    randomize per frame so the network learns position-independent
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
# Public entry point
# ---------------------------------------------------------------------------

def inject_artifacts(
    image_e: np.ndarray,
    band: BandConfig,
    rng: np.random.Generator,
    cfg: Optional[ArtifactConfig] = None,
) -> np.ndarray:
    """Apply the full artifact stack (CR + hot pixels) to one band frame."""
    cfg = cfg or ArtifactConfig()
    out = inject_cosmic_rays(image_e, band, rng, cfg)
    out = inject_hot_pixels(out, rng, cfg)
    return out
