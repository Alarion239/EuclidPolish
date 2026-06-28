"""
Observation sub-package: forward model.
"""

from euclid_polish.sky.observation.artifacts import (
    inject_artifacts,
    inject_cosmic_rays,
    inject_hot_pixels,
)
from euclid_polish.sky.observation.differential_kernel import (
    DifferentialKernel,
    compute_differential_kernel,
)
from euclid_polish.sky.observation.noise import apply_band_noise
from euclid_polish.sky.observation.observation_simulator import (
    ObservationSimulator,
    ObservationSimulatorConfig,
)
from euclid_polish.sky.observation.resample import cubic_upsample, lanczos3_upsample, upsample
from euclid_polish.sky.observation.saturation import StarSaturationModel

__all__ = [
    "ObservationSimulator", "ObservationSimulatorConfig",
    "apply_band_noise",
    "StarSaturationModel",
    "upsample", "lanczos3_upsample", "cubic_upsample",
    "inject_artifacts", "inject_cosmic_rays", "inject_hot_pixels",
    "DifferentialKernel", "compute_differential_kernel",
]
