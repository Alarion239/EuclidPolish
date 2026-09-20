"""Image-space noise models.

:func:`apply_archive_noise` is the delivered-MER noise used by generated
records and on-the-fly training: Euclid's own MER noise level with the pixel
correlation of a dithered, bilinearly resampled stack.
:func:`apply_band_noise` is the older detector-grid Poisson/read model, still
used by :meth:`Image.with_band_noise`.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from euclid_polish.config import BandConfig
from euclid_polish.sky.observation.artifacts import (
    ArtifactConfig,
    inject_artifacts,
)

# ---------------------------------------------------------------------------
# Euclid per-band noise
# ---------------------------------------------------------------------------

def background_sigma_e(band: BandConfig) -> float:
    """Expected blank-sky detector RMS per pixel of :func:`apply_band_noise`.

    Sky and dark current contribute Poisson variance over the full stack
    integration; every exposure adds one read-noise variance. The value is in
    stack electrons on ``band.pixel_scale_lr_arcsec`` pixels and does not
    depend on any scene, which makes it the reference level for calibrations
    and artifact thresholds.
    """
    t_total = band.t_total_s
    pixel_area = band.pixel_scale_lr_arcsec ** 2
    sky_e = band.sky_e_per_s_per_arcsec2 * pixel_area * t_total
    dark_e = band.dark_e_per_s_per_pix * t_total
    return float(np.sqrt(
        sky_e + dark_e + band.n_exposures * band.read_noise_e ** 2
    ))


def apply_band_noise(
    signal_e: np.ndarray,
    band: BandConfig,
    rng: np.random.Generator,
    *,
    add_artifacts: bool = False,
    artifact_config: ArtifactConfig | None = None,
) -> np.ndarray:
    """Per-band Poisson + (optional) detector artifacts + Gaussian read noise.

    Order follows the physical readout chain: photons + sky + dark
    accumulate → cosmic rays / hot pixels / interpolation residuals
    deposit charge → ramp is read with Gaussian read noise →
    sky-subtracted on the ground.

    Module-level so non-class callers (the :class:`ObservationSimulator`
    per-band pipeline, the :meth:`Image.with_band_noise` method) share
    one noise model.
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
        observed = inject_artifacts(
            observed, band, rng, acfg, local_sigma_e=background_sigma_e(band),
        ).astype(np.float64)

    read_sigma = band.read_noise_e * np.sqrt(band.n_exposures)
    read = rng.normal(0.0, read_sigma, size=signal_e.shape)
    return (observed + read).astype(np.float32)


# ---------------------------------------------------------------------------
# Delivered-MER noise
# ---------------------------------------------------------------------------
#
# Level: Euclid's MER RMS maps (``BandConfig.mer_rms_e``) combined with the
# photon noise of the sources. The maps quote per-pixel noise as if pixels
# were independent, which is the noise that adds up over an aperture.
#
# Texture: the MER pipeline interpolates every exposure onto the 0.10" grid
# with bilinear weights at some sub-pixel offset, so neighbouring pixels share
# noise. :func:`dithered_unit_noise` rebuilds exactly that, which gives the
# lower single-pixel scatter and the neighbour correlation of a real stack
# without fitting any correlation parameter.


def _detector_factor(band: BandConfig) -> int:
    """Integer ratio of detector pixel size to archive pixel size."""
    ratio = band.native_detector_scale_arcsec / band.pixel_scale_lr_arcsec
    factor = int(round(ratio))
    if factor < 1 or not math.isclose(ratio, factor, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            "native/archive pixel-scale ratio must be a positive integer; "
            f"got {band.native_detector_scale_arcsec:g}/"
            f"{band.pixel_scale_lr_arcsec:g}={ratio:g} for {band.name}"
        )
    return factor


def _bilinear_axis(
    n_out: int, factor: int, offset: float, pad: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Lower detector index and upper-neighbour weight for one output axis."""
    position = (
        (np.arange(n_out, dtype=np.float64) + 0.5) / factor - 0.5
        + offset + pad
    )
    lower = np.floor(position).astype(np.int64)
    return lower, position - lower


def _shifted_bilinear(
    detector: np.ndarray,
    factor: int,
    offset_y: float,
    offset_x: float,
    shape: tuple[int, int],
    pad: int,
) -> np.ndarray:
    """Resample a detector-grid image onto the archive grid at a sub-pixel offset.

    Archive pixel ``j`` samples detector coordinate
    ``(j + 0.5) / factor - 0.5 + offset``. Each output pixel's weights sum to
    one, so every detector pixel spreads a total weight of ``factor**2`` over
    the archive grid. ``pad`` detector pixels of margin keep all samples
    inside the array.
    """
    y0, wy = _bilinear_axis(shape[0], factor, offset_y, pad)
    x0, wx = _bilinear_axis(shape[1], factor, offset_x, pad)
    rows = (
        detector[y0] * (1.0 - wy)[:, None]
        + detector[y0 + 1] * wy[:, None]
    )
    return rows[:, x0] * (1.0 - wx) + rows[:, x0 + 1] * wx


def dithered_unit_noise(
    shape: tuple[int, int],
    band: BandConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Noise with unit variance on large scales and the texture of a MER stack.

    ``band.n_exposures`` exposures of independent unit white noise are drawn
    on the detector grid (0.10" VIS, 0.30" NISP). Each is shifted by its own
    sub-pixel offset, resampled bilinearly onto the 0.10" archive grid, and
    the exposures are averaged. Offsets are stratified along each axis so the
    exposures spread over the detector pixel; a draw with all offsets at one
    phase would leave a fixed period-``factor`` variance pattern.

    Dividing by ``factor * sqrt(n_exposures)`` makes the variance of a sum
    over a large area equal to its pixel count. Single pixels have lower
    variance, about (2/3)**2 in VIS and (2/9)**2 in NISP, because
    interpolation shares each detector pixel's noise with its neighbours.
    """
    height, width = int(shape[0]), int(shape[1])
    if height < 1 or width < 1:
        raise ValueError(f"shape must be positive, got {shape}")
    n_exposures = int(band.n_exposures)
    if n_exposures < 1:
        raise ValueError(f"{band.name} needs at least one exposure")
    factor = _detector_factor(band)
    pad = 2
    detector_shape = (
        math.ceil(height / factor) + 2 * pad,
        math.ceil(width / factor) + 2 * pad,
    )
    offsets_y = (rng.permutation(n_exposures) + rng.random(n_exposures)) / n_exposures
    offsets_x = (rng.permutation(n_exposures) + rng.random(n_exposures)) / n_exposures
    total = np.zeros((height, width), dtype=np.float64)
    for offset_y, offset_x in zip(offsets_y, offsets_x, strict=True):
        white = rng.standard_normal(detector_shape)
        total += _shifted_bilinear(
            white, factor, float(offset_y), float(offset_x),
            (height, width), pad,
        )
    return total / (factor * math.sqrt(n_exposures))


def apply_archive_noise(
    signal_e: np.ndarray,
    band: BandConfig,
    rng: np.random.Generator,
    *,
    add_artifacts: bool = False,
    artifact_config: ArtifactConfig | None = None,
    noise_scale_map: np.ndarray | None = None,
    sky_rms_e: float | None = None,
) -> np.ndarray:
    """Add noise as it appears in the delivered 0.10" MER mosaic.

    The local noise level is Euclid's sky level combined with the photon
    noise of the source, ``sqrt(sky_rms_e**2 + signal)``, multiplied by
    ``noise_scale_map`` when given (field depth and pointing overlaps). The
    level multiplies a :func:`dithered_unit_noise` field, so apertures see
    that level while single pixels show the lower, correlated scatter of a
    resampled stack. The signal itself is untouched: empirical MER ePSFs
    already contain detector sampling and mosaic interpolation.

    ``sky_rms_e`` is the sky level for this scene, normally one draw from the
    measured Q1 distribution; ``None`` uses the band's median,
    ``band.mer_rms_e``.

    Sparse artifacts are survivors of the pipeline's rejection, so they are
    injected last on the archive grid and are never rescaled.
    """
    signal = np.asarray(signal_e, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError(f"signal_e must be 2-D, got shape {signal.shape}")
    sky_rms = float(band.mer_rms_e if sky_rms_e is None else sky_rms_e)
    if not math.isfinite(sky_rms) or sky_rms <= 0.0:
        raise ValueError(
            f"{band.name} needs a positive sky noise level "
            f"(sky_rms_e or mer_rms_e), got {sky_rms!r}"
        )
    noise_scale = None
    if noise_scale_map is not None:
        noise_scale = np.asarray(noise_scale_map, dtype=np.float32)
        if noise_scale.shape != signal.shape:
            raise ValueError(
                f"noise_scale_map shape {noise_scale.shape} must match "
                f"signal shape {signal.shape}"
            )
        if not np.all(np.isfinite(noise_scale)) or np.any(noise_scale <= 0.0):
            raise ValueError("noise_scale_map must contain finite positive values")

    unit = dithered_unit_noise(signal.shape, band, rng)
    level = np.sqrt(
        sky_rms * sky_rms + np.clip(signal.astype(np.float64), 0.0, None)
    )
    if noise_scale is not None:
        level = level * noise_scale
    observed = (signal + level * unit).astype(np.float32)

    if add_artifacts:
        cfg = artifact_config or ArtifactConfig()
        area_ratio = float(_detector_factor(band) ** 2)
        # A surviving hit is one detector pixel's charge spread over the
        # factor**2 archive pixels that cover it; keep it a single sparse
        # archive pixel at that per-pixel charge.
        archive_cfg = replace(
            cfg,
            cr_charge_median_e=cfg.cr_charge_median_e / area_ratio,
            hot_pixel_charge_mean_e=cfg.hot_pixel_charge_mean_e / area_ratio,
        )
        depth = 1.0 if noise_scale is None else float(np.median(noise_scale))
        sky_pixel_sigma_e = sky_rms * depth * float(np.std(unit))
        observed = inject_artifacts(
            observed, band, rng, archive_cfg, local_sigma_e=sky_pixel_sigma_e,
        )
    return np.asarray(observed, dtype=np.float32)
