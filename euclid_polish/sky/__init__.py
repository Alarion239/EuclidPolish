"""
Sky Generation module for EuclidPolish.

This module provides classes for generating clean and dirty sky images,
including PSF convolution and downsampling operations.
"""

# PSFConvolution is lightweight (numpy + scipy only) — import eagerly.
from euclid_polish.sky.psf_convolution import PSFConvolution

# CleanSkyGenerator / CleanGalaxySimulator depend on galsim and tensorflow,
# which are slow to import.  They are loaded lazily on first access so that
# importing this package (e.g. to get PSFConvolution) doesn't stall startup.
__all__ = [
    "CleanSkyGenerator",
    "CleanGalaxySimulator",
    "PSFConvolution",
]


def __getattr__(name: str):
    if name in ("CleanSkyGenerator", "CleanGalaxySimulator"):
        from euclid_polish.sky.clean_generator import (
            CleanSkyGenerator,
            CleanGalaxySimulator,
        )
        # Cache in module globals so subsequent accesses are instant.
        globals()["CleanSkyGenerator"] = CleanSkyGenerator
        globals()["CleanGalaxySimulator"] = CleanGalaxySimulator
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
