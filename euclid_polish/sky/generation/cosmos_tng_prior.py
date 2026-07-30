"""COSMOS2025 physical prior for TNG morphology draws.

COSMOS supplies redshift, stellar mass, apparent size, and an HST/ACS F814W
brightness anchor. TNG supplies the Euclid VIS/NISP morphology and colours.
The F814W anchor is mapped to VIS by the fitted observation transfer, then one
shared scalar normalizes all four TNG channels so their ratios are preserved.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from euclid_polish.config import Config
from euclid_polish.photometry import ab_mag_to_electrons


@dataclass(frozen=True)
class CosmosTngDraw:
    catalog_id: str
    mag_hst_f814w: float
    target_vis_mag: float
    target_vis_flux_e: float
    z: float
    logmass: float
    re_arcsec: float
    imputed_size: bool
    brightness_transfer: str


@dataclass(frozen=True)
class F814WToVisTransfer:
    offset_mag: float = 0.0
    magnitude_slope: float = 1.0
    scatter_mag: float = 0.0
    source: str = "identity_fallback"

    def sample_vis_mag(
        self, mag_hst_f814w: float, rng: np.random.Generator,
    ) -> float:
        pivot = 24.0
        mean = (
            pivot
            + self.magnitude_slope * (float(mag_hst_f814w) - pivot)
            + self.offset_mag
        )
        return float(mean + rng.normal(0.0, self.scatter_mag))


def _load_brightness_transfer(path: str | Path) -> F814WToVisTransfer:
    fit_path = Path(path)
    try:
        payload = json.loads(fit_path.read_text())
    except (OSError, json.JSONDecodeError):
        return F814WToVisTransfer()
    cone_count = int((payload.get("inputs") or {}).get("euclid_cone_count", 1))
    key = (
        "local_normalization_sensitivity_fit"
        if cone_count >= 3
        else "fit"
    )
    fit = payload.get(key) or {}
    try:
        return F814WToVisTransfer(
            offset_mag=float(fit["vis_minus_f814w_mag"]),
            magnitude_slope=float(fit["magnitude_slope"]),
            scatter_mag=max(0.0, float(fit["scatter_mag"])),
            source=f"{key}:{fit_path}",
        )
    except (KeyError, TypeError, ValueError):
        return F814WToVisTransfer()


class CosmosTngPrior:
    """Memory-resident COSMOS physical/brightness sampler."""

    def __init__(
        self,
        path: str | Path,
        *,
        photometric_fit_path: str | Path = Config.COSMOS_EUCLID_FIT_PATH,
        mag_min: float = 18.0,
        mag_max: float = 28.0,
    ):
        self.path = str(path)
        with np.load(self.path, allow_pickle=False) as data:
            keys = set(data.files)

            def take(*names: str) -> np.ndarray:
                for name in names:
                    if name in keys:
                        return np.asarray(data[name])
                raise KeyError(f"{self.path} has none of {names!r}")

            catalog_id = take("catalog_id")
            # Old artifacts are accepted for replay, but all newly extracted
            # files use the physical filter name.
            f814w = take("mag_hst_f814w", "mag_vis", "mag_VIS")
            z = take("z_phot")
            mass = take("logmass_lephare", "logmass")
            re = take("re_combined_arcsec", "disk_re_arcsec")

        valid = (
            np.isfinite(f814w) & (f814w >= mag_min) & (f814w < mag_max)
            & np.isfinite(z) & (z > 0.01) & (z < 6.0)
            & np.isfinite(mass) & (mass > 4.0) & (mass < 13.0)
        )
        if not np.any(valid):
            raise ValueError(f"No usable COSMOS physical rows in {self.path}")
        self.catalog_id = catalog_id[valid].astype(str)
        self.f814w = f814w[valid].astype(np.float32)
        self.z = z[valid].astype(np.float32)
        self.mass = mass[valid].astype(np.float32)
        self.re = re[valid].astype(np.float32)
        self._size_donors = np.flatnonzero(
            np.isfinite(self.re) & (self.re > 0.01) & (self.re < 20.0)
        )
        if not len(self._size_donors):
            raise ValueError(f"COSMOS prior lacks size donors: {self.path}")
        self.brightness_transfer = _load_brightness_transfer(
            photometric_fit_path
        )

    def __len__(self) -> int:
        return len(self.f814w)

    def _nearby_size_donor(
        self, rng: np.random.Generator, index: int,
    ) -> int:
        candidates = self._size_donors
        pool = candidates[
            rng.integers(0, len(candidates), size=min(128, len(candidates)))
        ]
        distance = (
            ((self.f814w[pool] - self.f814w[index]) / 0.5) ** 2
            + ((self.z[pool] - self.z[index]) / 0.35) ** 2
        )
        return int(pool[int(np.argmin(distance))])

    def sample(self, rng: np.random.Generator) -> CosmosTngDraw:
        i = int(rng.integers(0, len(self)))
        re = float(self.re[i])
        imputed_size = not np.isfinite(re) or not 0.01 < re < 20.0
        if imputed_size:
            re = float(self.re[self._nearby_size_donor(rng, i)])
        mag_hst_f814w = float(self.f814w[i])
        target_vis_mag = self.brightness_transfer.sample_vis_mag(
            mag_hst_f814w, rng
        )
        return CosmosTngDraw(
            catalog_id=str(self.catalog_id[i]),
            mag_hst_f814w=mag_hst_f814w,
            target_vis_mag=target_vis_mag,
            target_vis_flux_e=float(ab_mag_to_electrons(
                target_vis_mag, Config.get_band("VIS")
            )),
            z=float(self.z[i]),
            logmass=float(self.mass[i]),
            re_arcsec=re,
            imputed_size=imputed_size,
            brightness_transfer=self.brightness_transfer.source,
        )
