"""Central home for image-space noise models.

Two noise generators live here so :mod:`euclid_polish.sky.types`
can import them at module scope (used to live in
``sky.multiband_forward`` and ``training.transition_augmentations``,
both of which import ``MultiBandSkyImage``, forming a triangular
circular dependency — see the note on the deleted deferred-import
sites for the historical context).

Both functions are re-exported from their original locations so
existing imports continue to work unchanged.

  * :func:`apply_band_noise` — Euclid VIS/NISP per-band Poisson
    photon + sky + dark + (optional artifacts) + Gaussian read.
  * :func:`add_hlsp_noise` — HLSP F814W Gaussian-approximated
    signal-dependent shot + sky/read floor.
  * :func:`sample_hlsp_noise_params` — per-image (α, σ_floor) draw
    helper used by the denoiser training loop.
"""

from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from euclid_polish.config import BandConfig

from euclid_polish.sky.artifacts import inject_artifacts, ArtifactConfig

if TYPE_CHECKING:
    # Only needed for type hints — actual ArtifactConfig is fetched
    # lazily through the artifacts module to keep this file free of
    # the heavy artifacts machinery when noise is the only thing
    # the caller wants.
    from euclid_polish.sky.artifacts import ArtifactConfig


# ---------------------------------------------------------------------------
# Euclid per-band noise (was: sky/multiband_forward.py:apply_band_noise)
# ---------------------------------------------------------------------------

def apply_band_noise(
    signal_e: np.ndarray,
    band: BandConfig,
    rng: np.random.Generator,
    *,
    add_artifacts: bool = False,
    artifact_config: Optional["ArtifactConfig"] = None,
) -> np.ndarray:
    """Per-band Poisson + (optional) detector artifacts + Gaussian read noise.

    Order matches the physical readout chain: photons + sky + dark
    accumulate → cosmic rays / hot pixels / interpolation residuals
    deposit charge → ramp is read with Gaussian read noise →
    sky-subtracted on the ground.

    Module-level so non-class callers (e.g. the HST→Euclid TFRecord
    generator at ``scripts/fasrc_generate_hst_tfrecords.py``, the
    :class:`MultiBandForward` per-band pipeline, the
    :meth:`MultiBandSkyImage.with_band_noise` method) can use the
    exact same noise model.
    """

    t_total = band.t_total_s
    pixel_area = band.pixel_scale_lr_arcsec ** 2
    sky_e  = band.sky_e_per_s_per_arcsec2 * pixel_area * t_total
    dark_e = band.dark_e_per_s_per_pix * t_total

    lam = np.clip(
        signal_e.astype(np.float64) + sky_e + dark_e, 0.0, None,
    )
    observed = rng.poisson(lam).astype(np.float64) - (sky_e + dark_e)

    if add_artifacts:
        acfg = artifact_config or ArtifactConfig()
        sigma_floor_e = float(np.sqrt(
            sky_e + dark_e + band.n_exposures * band.read_noise_e ** 2
        ))
        observed = inject_artifacts(
            observed, band, rng, acfg, local_sigma_e=sigma_floor_e,
        ).astype(np.float64)

    read_sigma = band.read_noise_e * np.sqrt(band.n_exposures)
    read = rng.normal(0.0, read_sigma, size=signal_e.shape)
    return (observed + read).astype(np.float32)


# ---------------------------------------------------------------------------
# HLSP F814W noise (was: training/transition_augmentations.py:add_hlsp_noise)
# ---------------------------------------------------------------------------

def add_hlsp_noise(
    clean: np.ndarray,
    *,
    alpha: float,
    sigma_floor: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add HLSP F814W-style noise to a clean HR image (Euclid-scaled e⁻).

    Models the two noise sources that dominate HLSP COSMOS F814W
    rate maps after the pair-gen's ``rate × C`` rescaling to Euclid
    electrons (where ``C = HST→VIS rate ratio × t_VIS ≈ 1514``).

    Components
    ----------

    **Photon shot noise (Poisson on original HST counts).** In
    Euclid-scaled units, the shot variance per HR pixel is
    ``α · signal_e_HR`` where ``α = C / t_eff`` and ``t_eff`` is the
    effective HLSP integration time at that pixel (which varies
    across the tile — COSMOS-deep ≈ 2000s → α ≈ 0.76, COSMOS-DASH
    ≈ 700s → α ≈ 2.2). The Gaussian approximation is exact in the
    limit of large counts and indistinguishable from Poisson at the
    25+ e⁻ pixel scale where any structure exists; it also stays
    well-defined for sky-subtracted negative pixels (where literal
    ``np.random.poisson`` would refuse to draw).

    **Sky / read-noise floor.** Signal-independent Gaussian floor
    matching the HLSP depth: ``σ_floor`` for COSMOS-deep is
    ~10-20 e⁻ on the Euclid HR grid (5σ point-source depth
    ~26.5-27 AB).

    Total per-pixel noise variance:

        σ² = α · max(clean, 0) + σ_floor²
        ε  ∼ N(0, σ²)
    """
    if alpha < 0:
        raise ValueError(f"alpha must be ≥ 0, got {alpha}")
    if sigma_floor < 0:
        raise ValueError(f"sigma_floor must be ≥ 0, got {sigma_floor}")
    if rng is None:
        rng = np.random.default_rng()
    clean_f = np.asarray(clean, dtype=np.float32)
    shot_var = np.float32(alpha) * np.maximum(clean_f, 0.0)
    total_var = shot_var + np.float32(sigma_floor) ** 2
    sigma_per_pix = np.sqrt(total_var)
    eps = rng.standard_normal(clean_f.shape).astype(np.float32) * sigma_per_pix
    return clean_f + eps


def sample_hlsp_noise_params(
    rng: np.random.Generator,
    *,
    alpha_min: float = 0.5,
    alpha_max: float = 1.0,
    sigma_floor_min: float = 8.0,
    sigma_floor_max: float = 22.0,
) -> Tuple[float, float]:
    """Sample one ``(alpha, sigma_floor)`` pair for HLSP noise.

    The ranges span the spatial variation of effective exposure time
    + sky depth across a real HLSP COSMOS F814W tile. Drawing a fresh
    pair per training image teaches the denoiser to be robust across
    the full distribution of tile positions, not just the centre.
    """
    alpha       = float(rng.uniform(alpha_min,       alpha_max))
    sigma_floor = float(rng.uniform(sigma_floor_min, sigma_floor_max))
    return alpha, sigma_floor
