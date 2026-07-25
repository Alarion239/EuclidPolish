"""Image-space noise models.

:func:`apply_band_noise` is the low-level detector-grid Poisson/read/artifact
model. :func:`apply_archive_noise` wraps it with the native NISP exposure and
MER resampling path used by generated records and on-the-fly training.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from euclid_polish.config import BandConfig
from euclid_polish.sky.observation.artifacts import (
    ArtifactConfig,
    inject_artifacts,
)
from euclid_polish.sky.observation.resample import upsample

# ---------------------------------------------------------------------------
# Euclid per-band noise
# ---------------------------------------------------------------------------

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

    Module-level so non-class callers (e.g. the HST→Euclid TFRecord
    generator at ``scripts/fasrc_generate_hst_tfrecords.py``, the
    :class:`ObservationSimulator` per-band pipeline, the
    :meth:`Image.with_band_noise` method) share one noise model.
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
# Delivered-MER noise path
# ---------------------------------------------------------------------------

_NISP_DITHER_PHASES_4 = ((0, 1), (1, 0), (1, 1), (2, 2))


def _sum_rebin_2d(image: np.ndarray, factor: int) -> np.ndarray:
    """Sum adjacent ``factor x factor`` cells, padding only the far edges."""
    height, width = image.shape
    pad_y = (-height) % factor
    pad_x = (-width) % factor
    if pad_y or pad_x:
        image = np.pad(image, ((0, pad_y), (0, pad_x)), mode="constant")
    out_h, out_w = image.shape[0] // factor, image.shape[1] // factor
    return image.reshape(out_h, factor, out_w, factor).sum(axis=(1, 3))


def _shift_without_wrap(
    image: np.ndarray, dy: int, dx: int,
) -> np.ndarray:
    """Integer-shift an image using reflected edge samples, never wraparound."""
    if dy == 0 and dx == 0:
        return image
    pad = max(abs(int(dy)), abs(int(dx)))
    padded = np.pad(image, pad, mode="reflect")
    height, width = image.shape
    y0 = pad - int(dy)
    x0 = pad - int(dx)
    return padded[y0:y0 + height, x0:x0 + width]


def _dither_phases(
    n_exposures: int,
    factor: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Return reproducible detector-to-MER subpixel phases for one stack.

    The four NISP phases are variance-balanced over the 3x3 output phases.
    A random dihedral transform plus global translation prevents the small
    residual phase pattern from occupying fixed array coordinates in every
    training example.
    """
    if factor == 3 and n_exposures == 4:
        phases = np.asarray(_NISP_DITHER_PHASES_4, dtype=np.int64)
        if bool(rng.integers(0, 2)):
            phases[:, 0] *= -1
        if bool(rng.integers(0, 2)):
            phases[:, 1] *= -1
        if bool(rng.integers(0, 2)):
            phases = phases[:, ::-1]
        offset = rng.integers(0, factor, size=2)
        phases = (phases + offset) % factor
        return [(int(y), int(x)) for y, x in phases]

    all_phases = np.asarray(
        [(y, x) for y in range(factor) for x in range(factor)],
        dtype=np.int64,
    )
    order = rng.permutation(len(all_phases))
    return [
        tuple(int(v) for v in all_phases[order[i % len(order)]])
        for i in range(n_exposures)
    ]


def _robust_sigma(image: np.ndarray) -> float:
    """Robust background RMS used only to scale faint MER streaks."""
    arr = np.asarray(image, dtype=np.float64)
    if min(arr.shape) > 24:
        arr = arr[6:-6, 6:-6]
    median = float(np.median(arr))
    return 1.4826 * float(np.median(np.abs(arr - median)))


def apply_archive_noise(
    signal_e: np.ndarray,
    band: BandConfig,
    rng: np.random.Generator,
    *,
    add_artifacts: bool = False,
    artifact_config: ArtifactConfig | None = None,
    resample_kernel: Literal["lanczos3", "cubic"] = "lanczos3",
    noise_scale_map: np.ndarray | None = None,
) -> np.ndarray:
    """Add detector noise as it appears in the delivered 0.10" MER mosaic.

    VIS is native at the archive scale and therefore uses
    :func:`apply_band_noise` unchanged. NISP Y/J/H are different: four
    independent 0.30" H2RG exposures are sky/dark/read-noised on their native
    detector cells, Lanczos-resampled at their dither phases, converted from
    native-cell integrated electrons to 0.10"-cell electrons (``/ 3**2``),
    and co-added. This reproduces the strong short-range covariance and much
    lower per-output-pixel RMS of real MER NISP mosaics. Sparse artifacts are
    post-rejection MER residuals, so they are injected only after resampling;
    they must not acquire Lanczos lobes that resemble an optical PSF.

    ``signal_e`` remains on the delivered archive grid. Empirical MER ePSFs
    already contain detector sampling and resampling, so reprocessing the
    deterministic signal through the native grid would blur it twice. Only
    the stochastic detector residual follows the native path.

    ``noise_scale_map`` optionally scales only the stochastic residual on the
    archive grid.  The observation simulator uses one shared map for all four
    bands to represent a field's depth plus a noisier pointing intersection.
    """
    signal = np.asarray(signal_e, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError(f"signal_e must be 2-D, got shape {signal.shape}")
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

    ratio = band.native_detector_scale_arcsec / band.pixel_scale_lr_arcsec
    factor = int(round(ratio))
    if factor < 1 or not np.isclose(ratio, factor, rtol=0.0, atol=1e-6):
        raise ValueError(
            "native/archive pixel-scale ratio must be a positive integer; "
            f"got {band.native_detector_scale_arcsec:g}/"
            f"{band.pixel_scale_lr_arcsec:g}={ratio:g} for {band.name}"
        )
    if factor == 1:
        observed = apply_band_noise(
            signal, band, rng,
            add_artifacts=add_artifacts,
            artifact_config=artifact_config,
        )
        if noise_scale is not None:
            observed = signal + (observed - signal) * noise_scale
        return observed.astype(np.float32, copy=False)

    # The input is the full-stack source expectation. A native detector cell
    # covers factor^2 archive pixels; each dither receives 1/N of that stack.
    native_signal_stack = _sum_rebin_2d(signal, factor).astype(np.float32)
    native_signal_one = native_signal_stack / float(band.n_exposures)
    native_band = replace(
        band,
        pixel_scale_lr_arcsec=band.native_detector_scale_arcsec,
        n_exposures=1,
    )

    cfg = artifact_config or ArtifactConfig()
    phases = _dither_phases(band.n_exposures, factor, rng)
    output_residual = np.zeros(
        (native_signal_stack.shape[0] * factor,
         native_signal_stack.shape[1] * factor),
        dtype=np.float32,
    )
    area_ratio = float(factor * factor)
    for phase_y, phase_x in phases:
        native_observed = apply_band_noise(
            native_signal_one,
            native_band,
            rng,
            # Detector masks/ramp fitting/dither rejection happen before the
            # delivered MER mosaic.  We model only the sparse survivors below
            # on the final grid, after the native noise has been resampled.
            add_artifacts=False,
        )
        native_residual = native_observed - native_signal_one
        archive_residual = upsample(
            native_residual,
            factor=factor,
            kernel=resample_kernel,
        ) / area_ratio
        output_residual += _shift_without_wrap(
            archive_residual, phase_y, phase_x,
        )

    height, width = signal.shape
    residual = output_residual[:height, :width]
    if noise_scale is not None:
        residual = residual * noise_scale
    observed = (signal + residual).astype(
        np.float32, copy=False)
    if add_artifacts:
        sigma_e = _robust_sigma(output_residual[:height, :width])
        if sigma_e > 0.0:
            # A native single-pixel charge was previously spread over roughly
            # factor² MER pixels. Keep the residual peak scale while making
            # the surviving artifact genuinely sparse on the delivered grid.
            archive_cfg = replace(
                cfg,
                cr_charge_median_e=cfg.cr_charge_median_e / area_ratio,
                hot_pixel_charge_mean_e=(
                    cfg.hot_pixel_charge_mean_e / area_ratio),
            )
            observed = inject_artifacts(
                observed, band, rng, archive_cfg, local_sigma_e=sigma_e,
            )
    return observed.astype(np.float32, copy=False)
