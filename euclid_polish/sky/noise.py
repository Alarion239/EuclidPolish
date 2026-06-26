"""Central home for image-space noise models.

:func:`apply_band_noise` is Euclid VIS/NISP per-band Poisson photon +
sky + dark + (optional artifacts) + Gaussian read. It lives here so
:mod:`euclid_polish.image` can import it at module scope without
re-introducing the triangular import cycle through
:mod:`euclid_polish.sky.multiband_forward` (which itself imports
``Image``).
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

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
    :meth:`Image.with_band_noise` method) can use the
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
