"""Test-only minimal catalog fixture.

The production pipeline requires the real COSMOS2025 master FITS file
(:class:`euclid_polish.sky.cosmos2025.Cosmos2025Catalog`). This module
provides a tiny synthetic in-memory catalog used by unit / benchmark
tests that exercise downstream pipeline pieces without needing the
10-GB FITS file on disk. It is intentionally NOT exported from the
production package.
"""

from __future__ import annotations

import math

import numpy as np

from euclid_polish.config import Config
from euclid_polish.sky.generation.cosmos2025 import CosmosCatalog, GalaxyParams


def _mag_to_electrons_per_stack(mag: np.ndarray, band) -> np.ndarray:
    """Convert AB mag → electrons over the full exposure stack."""
    sim_zp = band.zeropoint_ab_e_per_s + 2.5 * math.log10(
        band.exposure_time_s * band.n_exposures
    )
    return 10.0 ** (-0.4 * (np.asarray(mag) - sim_zp))


class TinyCosmosCatalog(CosmosCatalog):
    """A small synthetic catalog for tests. Same API as :class:`CosmosCatalog`."""

    def __init__(self, n_galaxies: int = 2_000, seed: int = 0):
        rng = np.random.default_rng(seed)
        n = int(n_galaxies)

        self.catalog_id      = np.arange(n, dtype=np.int64)
        self.ra_deg          = rng.uniform(149.5, 150.5, n)
        self.dec_deg         = rng.uniform( 1.5,   2.5,  n)
        self.z_phot          = np.clip(rng.lognormal(mean=-0.3, sigma=0.7, size=n), 0.05, 4.0)

        self.bulge_re_arcsec = rng.uniform(0.05, 0.30, n)
        self.disk_re_arcsec  = rng.uniform(0.15, 0.80, n)
        self.bulge_q         = rng.uniform(0.7,  1.0,  n)
        self.disk_q          = rng.uniform(0.3,  0.95, n)
        self.angle_rad       = rng.uniform(0.0,  math.pi, n)

        mag_vis        = rng.uniform(20.0, 25.0, n)
        bt             = rng.beta(2.0, 3.0, n)
        rel_offset_mag = np.array([0.0, 0.4, 0.7, 0.9])  # VIS, Y_E, J_E, H_E

        bd_total_flux_per_band = np.zeros((n, 4), dtype=np.float64)
        for k, name in enumerate(Config.LR_INPUT_BAND_NAMES):
            band = Config.get_band(name)
            band_mag = mag_vis + rel_offset_mag[k]
            bd_total_flux_per_band[:, k] = _mag_to_electrons_per_stack(band_mag, band)

        self.bulge_flux_e = bd_total_flux_per_band * bt[:, None]
        self.disk_flux_e  = bd_total_flux_per_band * (1.0 - bt[:, None])

    def __len__(self) -> int:
        return int(self.catalog_id.size)

    def _row_to_params(self, i: int) -> GalaxyParams:
        return GalaxyParams(
            catalog_id        = int(self.catalog_id[i]),
            ra_deg            = float(self.ra_deg[i]),
            dec_deg           = float(self.dec_deg[i]),
            z_phot            = float(self.z_phot[i]),
            angle_rad         = float(self.angle_rad[i]),
            bulge_r_e_arcsec  = float(self.bulge_re_arcsec[i]),
            bulge_axis_ratio  = float(self.bulge_q[i]),
            disk_r_e_arcsec   = float(self.disk_re_arcsec[i]),
            disk_axis_ratio   = float(self.disk_q[i]),
            bulge_flux_e      = tuple(float(v) for v in self.bulge_flux_e[i]),
            disk_flux_e       = tuple(float(v) for v in self.disk_flux_e[i]),
        )

    def sample_galaxy(self, rng: np.random.Generator) -> GalaxyParams:
        return self._row_to_params(int(rng.integers(0, len(self))))

    # ``sample_lens_galaxy`` / ``sample_source_galaxy`` (incl. the source
    # physical-size cut) are inherited from :class:`CosmosCatalog`.
