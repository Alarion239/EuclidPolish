"""
COSMOS2025 master-catalog wrapper.

The simulator draws galaxies from the COSMOS-Web ``v1.1`` master catalog
(Shuntov+ 2025, arXiv:2506.03243). Each galaxy is described by:

* a single-band photometric anchor (``mag_auto_hst-f814w`` — our VIS proxy)
* a bulge+disk decomposition from HDU 6 (shared centroid + PA, independent
  radii + axis ratios per component) with per-band bulge and disk fluxes
* a LePHARE photo-z and galaxy/star/QSO type from HDU 2

The four bands we surface match :attr:`Config.COSMOS2025_BAND_TO_CATALOG_COLUMN`,
i.e. HST F814W ↦ VIS_E (the VIS proxy already used by the simulator) and
UltraVISTA Y/J/H ↦ Euclid NISP Y_E/J_E/H_E (close-bandpass proxies).

:class:`Cosmos2025Catalog` reads the real FITS file, filters to galaxies
with viable B+D fits and finite per-band magnitudes, and indexes the catalog
by redshift for fast lens/source sampling. The catalog is mandatory — the
pipeline does not run without it.
"""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits

from euclid_polish.config import Config


# ---------------------------------------------------------------------------
# Public dataclass returned by sampling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GalaxyParams:
    """All per-galaxy parameters needed to rasterise a B+D source.

    Geometry is shared between the bulge and disk (centroid + PA); the two
    components have independent radii, axis ratios, and per-band fluxes.

    Coordinates / sizes are in **arcsec** unless noted. Fluxes are in e⁻
    referenced to each band's full Wide-Survey stack integration; the band
    order matches :attr:`Config.LR_INPUT_BAND_NAMES`.
    """

    # Identity / sky position (None for stub-generated entries).
    catalog_id:        Optional[int]   = None
    ra_deg:            Optional[float] = None
    dec_deg:           Optional[float] = None
    z_phot:            float           = 0.5

    # Shared geometry
    angle_rad:         float           = 0.0    # major-axis PA, radians CCW from +x

    # Bulge (n=4 fixed)
    bulge_r_e_arcsec:  float           = 0.10
    bulge_axis_ratio:  float           = 0.80

    # Disk (n=1 fixed)
    disk_r_e_arcsec:   float           = 0.30
    disk_axis_ratio:   float           = 0.60

    # Per-band electron fluxes for each component.
    # Length-4 tuples ordered per Config.LR_INPUT_BAND_NAMES.
    bulge_flux_e:      Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    disk_flux_e:       Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def total_flux_e(self, band_index: int) -> float:
        """Total bulge+disk flux in band ``band_index`` (0–3, VIS..H_E)."""
        return float(self.bulge_flux_e[band_index] + self.disk_flux_e[band_index])


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CosmosCatalog(ABC):
    """Sampling interface used by the multi-band scene generator."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of usable galaxies after filtering."""

    @abstractmethod
    def sample_galaxy(self, rng: np.random.Generator) -> GalaxyParams:
        """Return parameters for one randomly drawn foreground galaxy."""

    @abstractmethod
    def sample_lens_galaxy(
        self, rng: np.random.Generator, z_lens_range: Tuple[float, float],
    ) -> GalaxyParams:
        """Return parameters for a galaxy plausibly acting as a strong lens.

        Filters to ``z_phot`` in ``z_lens_range`` (typically Collett's
        ``LENS_Z_LENS_*`` window).
        """

    @abstractmethod
    def sample_source_galaxy(
        self, rng: np.random.Generator, z_lens: float,
    ) -> GalaxyParams:
        """Return parameters for a background source at ``z_phot > z_lens + offset``."""


# ---------------------------------------------------------------------------
# Photometric helpers
# ---------------------------------------------------------------------------

def _mag_to_electrons_per_stack(mag_ab: np.ndarray, band: "BandConfig") -> np.ndarray:
    """Convert AB magnitude → total electrons over the band's stack integration.

    ``mag = ZP_E - 2.5 log10(electrons_total)`` with ``ZP_E`` = the band's
    stack zeropoint.
    """
    return 10.0 ** (-0.4 * (mag_ab - band.sim_zeropoint_e))


# ---------------------------------------------------------------------------
# Real catalog
# ---------------------------------------------------------------------------

class Cosmos2025Catalog(CosmosCatalog):
    """Reads the real COSMOS2025 v1.1 master FITS into in-memory numpy arrays.

    Only the columns the simulator uses are loaded — RA/Dec, B+D geometry,
    per-band bulge & disk magnitudes, and LePHARE redshift/type. The full
    787k-row catalog (10 GB) is read once at construction; usable subsets
    are filtered down to roughly ~few × 10⁵ galaxies depending on cuts.
    """

    # Columns we pull from each HDU.
    _PHOTO_COLS = ("id", "ra", "dec", "flag_star", "flag_blend", "warn_flag")
    _LEPHARE_COLS = ("zfinal", "type", "mod_minchi2_phys")
    _BD_GEOM_COLS = (
        "ra_detec_bd", "dec_detec_bd",
        "disk_radius_deg",  "disk_axratio",
        "bulge_radius_deg", "bulge_axratio",
        "angle_bd",
        "fmf_b+d_chi2",
    )

    def __init__(
        self,
        path: str = Config.COSMOS2025_CATALOG_PATH,
        *,
        max_bd_chi2: float = 10.0,
        max_mag: float = 25.0,
        verbose: bool = True,
    ):
        """COSMOS2025 master catalog reader.

        Parameters
        ----------
        max_mag : float, default 25.0
            Faint-end cutoff on **total** VIS (HST F814W) magnitude
            (``-2.5·log10(bulge_flux + disk_flux)``). Defaults to Euclid's
            VIS wide-survey 5σ depth (~25.6 spec for point sources; we
            anchor at 25.0 for typical extended galaxies). Galaxies
            fainter than this are dropped — they would be below Euclid's
            noise floor and contribute no useful signal to the simulated
            LR scenes. Loosen to 26.0–26.5 to include marginal detections;
            tighten further (e.g. 24.0) to keep only secure detections.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"COSMOS2025 catalog not found at {path}. "
                f"Set Config.COSMOS2025_CATALOG_PATH to its location."
            )
        self.path = path
        self.max_bd_chi2 = float(max_bd_chi2)
        self.max_mag = float(max_mag)

        if verbose:
            print(f"[cosmos2025] loading {path} ...")

        with fits.open(path, memmap=True) as hdul:
            photo  = hdul[Config.COSMOS2025_HDU_PHOTOMETRY].data
            leph   = hdul[Config.COSMOS2025_HDU_LEPHARE].data
            bd     = hdul[Config.COSMOS2025_HDU_BD].data

            # Photo (sky position + flags)
            ids       = np.asarray(photo["id"], dtype=np.int64)
            ra_deg    = np.asarray(photo["ra"], dtype=np.float64)
            dec_deg   = np.asarray(photo["dec"], dtype=np.float64)
            flag_star = np.asarray(photo["flag_star"], dtype=np.int32)
            flag_blend= np.asarray(photo["flag_blend"], dtype=np.int32)
            warn_flag = np.asarray(photo["warn_flag"], dtype=np.int32)

            # LePHARE
            z_phot    = np.asarray(leph["zfinal"], dtype=np.float64)
            obj_type  = np.asarray(leph["type"], dtype=np.int32)

            # B+D geometry (radii in degrees → convert to arcsec)
            disk_re_arcsec  = np.asarray(bd["disk_radius_deg"],  dtype=np.float64) * 3600.0
            bulge_re_arcsec = np.asarray(bd["bulge_radius_deg"], dtype=np.float64) * 3600.0
            disk_q          = np.asarray(bd["disk_axratio"],     dtype=np.float64)
            bulge_q         = np.asarray(bd["bulge_axratio"],    dtype=np.float64)
            angle_bd_deg    = np.asarray(bd["angle_bd"],         dtype=np.float64)
            bd_chi2         = np.asarray(bd["fmf_b+d_chi2"],     dtype=np.float64)

            # Per-band magnitudes (B+D model — separate bulge / disk)
            bands = Config.LR_INPUT_BAND_NAMES
            mag_bulge = np.empty((bd.shape[0], len(bands)), dtype=np.float64)
            mag_disk  = np.empty((bd.shape[0], len(bands)), dtype=np.float64)
            for k, band_name in enumerate(bands):
                col = Config.COSMOS2025_BAND_TO_CATALOG_COLUMN[band_name]
                mag_bulge[:, k] = np.asarray(bd[f"mag_model_bulge_{col}"], dtype=np.float64)
                mag_disk[:,  k] = np.asarray(bd[f"mag_model_disk_{col}"],  dtype=np.float64)

        # ---- Quality + viability mask ----
        finite_mag = np.isfinite(mag_bulge).all(axis=1) & np.isfinite(mag_disk).all(axis=1)
        finite_geom = (
            np.isfinite(disk_re_arcsec) & np.isfinite(bulge_re_arcsec)
            & np.isfinite(disk_q) & np.isfinite(bulge_q) & np.isfinite(angle_bd_deg)
            & (disk_re_arcsec > 0) & (bulge_re_arcsec > 0)
            & (disk_q > 0) & (disk_q <= 1.0) & (bulge_q > 0) & (bulge_q <= 1.0)
        )
        good_fit = np.isfinite(bd_chi2) & (bd_chi2 < self.max_bd_chi2)

        # Total VIS flux = bulge + disk in the HST F814W (= VIS proxy) channel.
        # Using the *total* flux rather than each component independently is
        # the honest detectability cut: a galaxy is "Euclid-visible" iff its
        # combined bulge + disk flux clears the VIS magnitude limit.
        vis_idx = Config.LR_INPUT_BAND_NAMES.index("VIS")
        vis_band = Config.BAND_VIS
        # Need finite per-component VIS mags to compute the total mag safely.
        finite_vis = np.isfinite(mag_bulge[:, vis_idx]) & np.isfinite(mag_disk[:, vis_idx])
        with np.errstate(invalid="ignore"):
            flux_bulge_vis = _mag_to_electrons_per_stack(mag_bulge[:, vis_idx], vis_band)
            flux_disk_vis  = _mag_to_electrons_per_stack(mag_disk[:,  vis_idx], vis_band)
            total_flux_vis = np.where(finite_vis,
                                      np.nan_to_num(flux_bulge_vis, nan=0.0)
                                      + np.nan_to_num(flux_disk_vis, nan=0.0),
                                      0.0)
            mag_total_vis = (vis_band.zeropoint_ab_e_per_s
                             + 2.5 * math.log10(vis_band.exposure_time_s * vis_band.n_exposures)
                             - 2.5 * np.log10(np.maximum(total_flux_vis, 1e-30)))
        bright_enough = finite_vis & (mag_total_vis < self.max_mag)
        is_galaxy = (obj_type == 0) & np.isfinite(z_phot) & (z_phot > 0.0)
        clean = (flag_star == 0) & (flag_blend == 0) & (warn_flag == 0)

        mask = finite_mag & finite_geom & good_fit & bright_enough & is_galaxy & clean

        # ---- Cache filtered arrays ----
        self.catalog_id      = ids[mask]
        self.ra_deg          = ra_deg[mask]
        self.dec_deg         = dec_deg[mask]
        self.z_phot          = z_phot[mask]
        self.disk_re_arcsec  = disk_re_arcsec[mask]
        self.bulge_re_arcsec = bulge_re_arcsec[mask]
        self.disk_q          = disk_q[mask]
        self.bulge_q         = bulge_q[mask]
        # Catalog stores degrees, world coords; for rasterisation we just need
        # an in-plane PA (the simulator's HR canvas has no celestial WCS).
        # Use ``angle_bd`` directly as the in-image PA (radians).
        self.angle_rad       = np.deg2rad(angle_bd_deg[mask])

        # Convert per-component magnitudes → per-component electron fluxes per band
        self.bulge_flux_e = np.empty_like(mag_bulge[mask])
        self.disk_flux_e  = np.empty_like(mag_disk[mask])
        for k, band_name in enumerate(Config.LR_INPUT_BAND_NAMES):
            band = Config.get_band(band_name)
            self.bulge_flux_e[:, k] = _mag_to_electrons_per_stack(mag_bulge[mask][:, k], band)
            self.disk_flux_e[:,  k] = _mag_to_electrons_per_stack(mag_disk[mask][:,  k], band)

        if verbose:
            n_total = int(mask.size)
            n_kept  = int(mask.sum())
            print(f"[cosmos2025] {n_kept}/{n_total} galaxies kept after quality cuts "
                  f"({100.0*n_kept/n_total:.1f}%)")

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
        idx = int(rng.integers(0, len(self)))
        return self._row_to_params(idx)

    def sample_lens_galaxy(
        self, rng: np.random.Generator, z_lens_range: Tuple[float, float],
    ) -> GalaxyParams:
        lo, hi = z_lens_range
        mask = (self.z_phot >= lo) & (self.z_phot <= hi)
        cand = np.flatnonzero(mask)
        if cand.size == 0:
            raise RuntimeError(f"No galaxies in z range {z_lens_range}")
        return self._row_to_params(int(rng.choice(cand)))

    def sample_source_galaxy(
        self, rng: np.random.Generator, z_lens: float,
    ) -> GalaxyParams:
        z_min = z_lens + Config.LENS_Z_SOURCE_OFFSET
        z_max = Config.LENS_Z_SOURCE_MAX
        mask = (self.z_phot >= z_min) & (self.z_phot <= z_max)
        cand = np.flatnonzero(mask)
        if cand.size == 0:
            raise RuntimeError(f"No source galaxies with z > {z_min}")
        return self._row_to_params(int(rng.choice(cand)))


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def open_cosmos2025(
    path: Optional[str] = None,
    **kwargs,
) -> CosmosCatalog:
    """Open the COSMOS2025 catalog. The catalog is mandatory.

    If ``path`` (or :attr:`Config.COSMOS2025_CATALOG_PATH`) does not exist
    on disk, this raises :class:`FileNotFoundError` rather than falling
    back to a synthetic catalog — the pipeline relies on real photometry
    and morphology and must not be run without it.
    """
    p = path or Config.COSMOS2025_CATALOG_PATH
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"COSMOS2025 catalog not found at {p}. Place the master "
            f"catalog there (Shuntov+ 2025, arXiv:2506.03243) or set "
            f"Config.COSMOS2025_CATALOG_PATH to its location."
        )
    return Cosmos2025Catalog(path=p, **kwargs)
