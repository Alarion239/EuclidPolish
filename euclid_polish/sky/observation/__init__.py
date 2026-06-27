"""
Observation sub-package: forward model.

Re-exports the observation-side public API from the flat ``euclid_polish.sky``
modules. All ``from euclid_polish.sky.<module> import X`` paths continue to
work unchanged; this namespace just adds ``euclid_polish.sky.observation.X``
as an alternative.
"""

from euclid_polish.sky.observation_simulator import (
    ObservationSimulator, ObservationSimulatorConfig,
)
from euclid_polish.sky.noise import apply_band_noise
from euclid_polish.sky.saturation import StarSaturationModel
from euclid_polish.sky.resample import upsample, lanczos3_upsample, cubic_upsample
from euclid_polish.sky.artifacts import inject_artifacts, inject_cosmic_rays, inject_hot_pixels
from euclid_polish.sky.differential_kernel import (
    DifferentialKernel, compute_differential_kernel,
)

__all__ = [
    "ObservationSimulator", "ObservationSimulatorConfig",
    "apply_band_noise",
    "StarSaturationModel",
    "upsample", "lanczos3_upsample", "cubic_upsample",
    "inject_artifacts", "inject_cosmic_rays", "inject_hot_pixels",
    "DifferentialKernel", "compute_differential_kernel",
]
