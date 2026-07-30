"""Joint COSMOS2025 population prior for TNG morphology draws.

COSMOS supplies the observable and physical parameters of each synthetic
galaxy.  The TNG atlas supplies only the resolved morphology.  Missing NISP
photometry or size measurements are imputed from a nearby COSMOS galaxy in
the joint (VIS magnitude, redshift) plane, rather than dropping faint rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from euclid_polish.config import Config
from euclid_polish.photometry import ab_mag_to_electrons


@dataclass(frozen=True)
class CosmosTngDraw:
    catalog_id: str
    magnitudes: tuple[float, float, float, float]
    flux_e_per_band: tuple[float, float, float, float]
    z: float
    logmass: float
    re_arcsec: float
    imputed_photometry: bool
    imputed_size: bool


class CosmosTngPrior:
    """Memory-resident sampler of the fitted COSMOS2025 latent population."""

    def __init__(self, path: str | Path, *, mag_min: float = 18.0,
                 mag_max: float = 28.0):
        self.path = str(path)
        with np.load(self.path, allow_pickle=False) as data:
            keys = set(data.files)

            def take(*names: str) -> np.ndarray:
                for name in names:
                    if name in keys:
                        return np.asarray(data[name])
                raise KeyError(f"{self.path} has none of {names!r}")

            catalog_id = take("catalog_id")
            vis = take("mag_VIS", "mag_vis")
            y = take("mag_Y_E", "mag_y_e")
            j = take("mag_J_E", "mag_j_e")
            h = take("mag_H_E", "mag_h_e")
            z = take("z_phot")
            mass = take("logmass_lephare", "logmass")
            re = take("re_combined_arcsec", "disk_re_arcsec")

        valid = (
            np.isfinite(vis) & (vis >= mag_min) & (vis < mag_max)
            & np.isfinite(z) & (z > 0.01) & (z < 6.0)
            & np.isfinite(mass) & (mass > 4.0) & (mass < 13.0)
        )
        if not np.any(valid):
            raise ValueError(f"No usable joint COSMOS rows in {self.path}")
        self.catalog_id = catalog_id[valid].astype(str)
        self.vis = vis[valid].astype(np.float32)
        self.z = z[valid].astype(np.float32)
        self.mass = mass[valid].astype(np.float32)
        self.re = re[valid].astype(np.float32)
        self.nisp = np.stack((y[valid], j[valid], h[valid]), axis=1).astype(np.float32)
        self._phot_donors = np.flatnonzero(
            np.all(np.isfinite(self.nisp) & (self.nisp < 90.0), axis=1)
        )
        self._size_donors = np.flatnonzero(
            np.isfinite(self.re) & (self.re > 0.01) & (self.re < 20.0)
        )
        if not len(self._phot_donors) or not len(self._size_donors):
            raise ValueError(f"COSMOS prior lacks photometry or size donors: {self.path}")

    def __len__(self) -> int:
        return len(self.vis)

    def _nearby_donor(self, rng: np.random.Generator, candidates: np.ndarray,
                      index: int) -> int:
        # A small random candidate pool is fast and preserves conditional
        # scatter; the standardized distance keeps both magnitude and z local.
        pool = candidates[rng.integers(0, len(candidates), size=min(128, len(candidates)))]
        distance = (
            ((self.vis[pool] - self.vis[index]) / 0.5) ** 2
            + ((self.z[pool] - self.z[index]) / 0.35) ** 2
        )
        return int(pool[int(np.argmin(distance))])

    def sample(self, rng: np.random.Generator) -> CosmosTngDraw:
        i = int(rng.integers(0, len(self)))
        nisp = self.nisp[i]
        imputed_photometry = not np.all(np.isfinite(nisp) & (nisp < 90.0))
        if imputed_photometry:
            donor = self._nearby_donor(rng, self._phot_donors, i)
            # Transfer donor colours, while keeping the selected row's VIS.
            nisp = self.vis[i] + (self.nisp[donor] - self.vis[donor])
        re = float(self.re[i])
        imputed_size = not np.isfinite(re) or not 0.01 < re < 20.0
        if imputed_size:
            donor = self._nearby_donor(rng, self._size_donors, i)
            re = float(self.re[donor])
        magnitudes = (float(self.vis[i]), *(float(x) for x in nisp))
        fluxes = tuple(
            float(ab_mag_to_electrons(mag, Config.get_band(band)))
            for mag, band in zip(
                magnitudes, Config.LR_INPUT_BAND_NAMES, strict=True
            )
        )
        return CosmosTngDraw(
            catalog_id=str(self.catalog_id[i]),
            magnitudes=magnitudes,
            flux_e_per_band=fluxes,
            z=float(self.z[i]),
            logmass=float(self.mass[i]),
            re_arcsec=re,
            imputed_photometry=imputed_photometry,
            imputed_size=imputed_size,
        )
