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
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.cosmology import Planck15 as _COSMO
from astropy.io import fits
from scipy.special import gammainc, gammaincinv

from euclid_polish.config import Config

#: Radians subtended by 1 arcsec — for the arcsec → proper-kpc conversion.
_RAD_PER_ARCSEC = np.pi / 180.0 / 3600.0

#: Filtered per-galaxy arrays a :class:`Cosmos2025Catalog` needs to sample —
#: everything ``_row_to_params`` / the lens+source samplers read. Persisting just
#: these (post-quality-cut) gives a few-MB ``.npz`` that loads much faster than
#: re-parsing the 10 GB master FITS, which matters when every parallel worker
#: rebuilds the catalog.
_FILTERED_ARRAYS = (
    "catalog_id", "ra_deg", "dec_deg", "z_phot", "angle_rad",
    "disk_re_arcsec", "bulge_re_arcsec", "disk_q", "bulge_q",
    "bulge_flux_e", "disk_flux_e",
)


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
    catalog_id:        int | None   = None
    ra_deg:            float | None = None
    dec_deg:           float | None = None
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
    bulge_flux_e:      tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    disk_flux_e:       tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def total_flux_e(self, band_index: int) -> float:
        """Total bulge+disk flux in band ``band_index`` (0–3, VIS..H_E)."""
        return float(self.bulge_flux_e[band_index] + self.disk_flux_e[band_index])


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CosmosCatalog(ABC):
    """Sampling interface used by the multi-band scene generator.

    Concrete subclasses must expose ``bulge_flux_e`` and ``disk_flux_e``
    as ``(N, NUM_LR_CHANNELS)`` float arrays so the shared
    :meth:`typical_band_electron_ratios` accessor works on both the
    real :class:`Cosmos2025Catalog` and any synthetic stand-in.
    """

    # Concrete subclasses populate these — declared here only so type
    # checkers don't trip over ``self.bulge_flux_e`` in the base method.
    bulge_flux_e: np.ndarray
    disk_flux_e:  np.ndarray
    z_phot:       np.ndarray

    @abstractmethod
    def __len__(self) -> int:
        """Number of usable galaxies after filtering."""

    @abstractmethod
    def sample_galaxy(self, rng: np.random.Generator) -> GalaxyParams:
        """Return parameters for one randomly drawn foreground galaxy."""

    @abstractmethod
    def _row_to_params(self, i: int) -> GalaxyParams:
        """Build :class:`GalaxyParams` from filtered-array row ``i``."""

    def sample_lens_galaxy(
        self, rng: np.random.Generator, z_lens_range: tuple[float, float],
    ) -> GalaxyParams:
        """Draw a foreground lens galaxy uniformly within ``z_lens_range``."""
        lo, hi = z_lens_range
        mask = (self.z_phot >= lo) & (self.z_phot <= hi)
        cand = np.flatnonzero(mask)
        if cand.size == 0:
            raise RuntimeError(f"No galaxies in z range {z_lens_range}")
        return self._row_to_params(int(rng.choice(cand)))

    def sample_source_galaxy(
        self, rng: np.random.Generator, z_lens: float,
    ) -> GalaxyParams:
        """Draw a background source galaxy for a lens at ``z_lens``.

        Candidates must sit behind the lens (``z ≥ z_lens + offset``), within
        ``LENS_Z_SOURCE_MAX``, and be **physically small** (a real lensed
        source is a compact background galaxy). Capping the *physical*
        half-light radius (``LENS_SOURCE_MAX_PHYS_RE_KPC``) rather than the
        angular one means the rendered angular size ``r_phys / d_A(z)`` shrinks
        with distance, so a more distant source appears smaller."""
        z_min = z_lens + Config.LENS_Z_SOURCE_OFFSET
        z_max = Config.LENS_Z_SOURCE_MAX
        mask = (self.z_phot >= z_min) & (self.z_phot <= z_max)
        re_cap_kpc = float(Config.LENS_SOURCE_MAX_PHYS_RE_KPC)
        if re_cap_kpc > 0.0:
            mask &= self.effective_re_kpc <= re_cap_kpc
        cand = np.flatnonzero(mask)
        if cand.size == 0:
            raise RuntimeError(
                f"No source galaxies with z > {z_min} and "
                f"physical R_e ≤ {re_cap_kpc} kpc"
            )
        return self._row_to_params(int(rng.choice(cand)))

    @property
    def effective_re_arcsec(self) -> np.ndarray:
        """Per-galaxy circularized combined bulge+disk half-light radius (arcsec),
        in VIS — the realistic on-sky size of each catalog galaxy. Computed once
        and cached. Used to size injected TNG stamps so they match the catalog's
        own size distribution. Requires the geometry/flux arrays every concrete
        catalog exposes (``bulge_re_arcsec``/``disk_re_arcsec``/``*_q``/``*_flux_e``)."""
        cached = getattr(self, "_effective_re_arcsec", None)
        if cached is None:
            cached = circularized_effective_radius_arcsec(
                self.bulge_re_arcsec, self.bulge_q, self.bulge_flux_e[:, 0],
                self.disk_re_arcsec,  self.disk_q,  self.disk_flux_e[:, 0])
            self._effective_re_arcsec = cached
        return cached

    @property
    def effective_re_kpc(self) -> np.ndarray:
        """Per-galaxy **physical** circularized half-light radius (proper kpc).

        ``r_phys = θ_e · d_A(z)`` — the angular size :attr:`effective_re_arcsec`
        times the proper kpc subtended by 1 arcsec at the galaxy's redshift
        (Planck15 angular-diameter distance, the same cosmology the apparent-size
        model uses). This is the redshift-independent intrinsic size, used to cap
        the lensed-source population to physically small galaxies. Computed once
        and cached."""
        cached = getattr(self, "_effective_re_kpc", None)
        if cached is None:
            kpc_per_arcsec = (
                _COSMO.angular_diameter_distance(self.z_phot).to(u.kpc).value
                * _RAD_PER_ARCSEC
            )
            cached = self.effective_re_arcsec * kpc_per_arcsec
            self._effective_re_kpc = cached
        return cached

    def typical_band_electron_ratios(self) -> np.ndarray:
        """Median per-source ``e_band / e_VIS`` after the catalog's
        quality cuts.

        Returns a length-:attr:`Config.NUM_LR_CHANNELS` float32 array.
        Entry 0 (VIS) is exactly 1.0 by construction; NISP entries
        express the "typical galaxy" Euclid-stack electron count in
        that band as a fraction of its VIS electron count.

        Used by the HST-derived TFRecord generator
        (:mod:`scripts.fasrc_generate_hst_tfrecords`) to paint NISP HR
        channels from the single-band HST F814W cutout: each pixel's
        NISP brightness is ``VIS_pixel × typical_ratio[band]``. That's
        a per-pixel global colour — every source in the cutout treated
        as a typical-colour galaxy — wrong per-source but the right
        order of magnitude for the noise statistics the network sees.
        """
        total_e = np.asarray(self.bulge_flux_e + self.disk_flux_e,
                             dtype=np.float64)             # (N, 4)
        # Per-source ratio; guard against the rare ``e_VIS = 0`` edge
        # case (shouldn't happen post-filter but cheap to be robust).
        vis = total_e[:, 0:1]
        safe_vis = np.where(vis > 0, vis, np.nan)
        per_source_ratio = total_e / safe_vis              # (N, 4)
        finite_rows = np.isfinite(per_source_ratio).all(axis=1)
        # Median is robust to the heavy-tailed flux distribution; mean
        # would be dragged up by a handful of unusually red sources.
        med = np.nanmedian(per_source_ratio[finite_rows], axis=0)
        med[0] = 1.0       # snap VIS-vs-itself to exactly 1
        return med.astype(np.float32)


# ---------------------------------------------------------------------------
# Photometric helpers
# ---------------------------------------------------------------------------

def _mag_to_electrons_per_stack(mag_ab: np.ndarray, band: BandConfig) -> np.ndarray:
    """Convert AB magnitude → total electrons over the band's stack integration.

    ``mag = ZP_E - 2.5 log10(electrons_total)`` with ``ZP_E`` = the band's
    stack zeropoint.
    """
    return 10.0 ** (-0.4 * (mag_ab - band.sim_zeropoint_e))


def circularized_effective_radius_arcsec(
    bulge_re: np.ndarray, bulge_q: np.ndarray, bulge_flux: np.ndarray,
    disk_re: np.ndarray, disk_q: np.ndarray, disk_flux: np.ndarray,
    *, n_bulge: float = 4.0, n_disk: float = 1.0, n_iter: int = 60,
) -> np.ndarray:
    """Per-galaxy **circularized** half-light radius (arcsec) of the combined
    bulge(n=4)+disk(n=1) profile — the size at which the network actually *sees*
    a COSMOS galaxy, and the like-for-like match to the circular curve-of-growth
    radius we measure on TNG stamps.

    It is smaller than either component's catalog radius for two reasons:
    (1) the catalog stores **major-axis** half-light radii — the circularized
    radius (circle with the half-light ellipse's area) is ``re·√q``; and
    (2) the **combined** half-light radius of a compact bulge + extended disk is
    pulled inward by the bulge, well below the disk radius for bulge-dominated
    galaxies. The combined curve of growth

        f_b·P(2n_b, b_{n_b}·(R/re_b)^{1/n_b}) + f_d·P(2n_d, b_{n_d}·(R/re_d)^{1/n_d})

    (``P`` the regularised lower incomplete gamma; ``re`` circularised) is solved
    for the ``R`` enclosing half the total bulge+disk flux by vectorised
    bisection. Fluxes are the per-band electron counts (pass VIS for the SR band).
    """
    reb = np.asarray(bulge_re, float) * np.sqrt(np.clip(np.asarray(bulge_q, float), 1e-3, 1.0))
    red = np.asarray(disk_re,  float) * np.sqrt(np.clip(np.asarray(disk_q,  float), 1e-3, 1.0))
    reb = np.where(reb > 0, reb, 1e-6)
    red = np.where(red > 0, red, 1e-6)
    fb = np.clip(np.asarray(bulge_flux, float), 0.0, None)
    fd = np.clip(np.asarray(disk_flux,  float), 0.0, None)
    ftot = fb + fd
    ftot = np.where(ftot > 0, ftot, 1.0)
    bnb = gammaincinv(2.0 * n_bulge, 0.5)
    bnd = gammaincinv(2.0 * n_disk, 0.5)

    def enclosed(R: np.ndarray) -> np.ndarray:
        pb = gammainc(2.0 * n_bulge, bnb * (R / reb) ** (1.0 / n_bulge))
        pd = gammainc(2.0 * n_disk,  bnd * (R / red) ** (1.0 / n_disk))
        return (fb * pb + fd * pd) / ftot

    lo = np.full(reb.shape, 1e-4, dtype=float)
    hi = np.full(reb.shape, 60.0, dtype=float)
    for _ in range(int(n_iter)):
        mid = 0.5 * (lo + hi)
        over = enclosed(mid) > 0.5
        hi = np.where(over, mid, hi)
        lo = np.where(over, lo, mid)
    return 0.5 * (lo + hi)


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
        # The detectability cut uses the *total* flux: a galaxy is
        # "Euclid-visible" iff its combined bulge + disk flux clears the VIS
        # magnitude limit.
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

    # ``sample_lens_galaxy`` / ``sample_source_galaxy`` are inherited from
    # :class:`CosmosCatalog` so the real and stub catalogs share one
    # implementation (and the same source size cut).

    # ------------------------------------------------------------------ #
    # Fast (de)serialisation of the *filtered* catalog
    # ------------------------------------------------------------------ #
    def save_filtered_npz(self, path: str) -> None:
        """Persist the post-quality-cut arrays to a compact ``.npz`` so workers
        can reload the catalog without re-parsing the 10 GB master FITS."""
        np.savez(
            path,
            max_mag=np.float64(self.max_mag),
            max_bd_chi2=np.float64(self.max_bd_chi2),
            **{k: getattr(self, k) for k in _FILTERED_ARRAYS},
        )

    @classmethod
    def from_filtered_npz(cls, path: str, *, verbose: bool = True
                          ) -> Cosmos2025Catalog:
        """Reconstruct a catalog from a :meth:`save_filtered_npz` file — no
        FITS read, no quality cuts (already applied). Bypasses ``__init__``."""
        obj = cls.__new__(cls)        # skip the heavy FITS-reading __init__
        with np.load(path) as data:
            for k in _FILTERED_ARRAYS:
                setattr(obj, k, data[k])
            obj.max_mag = float(data["max_mag"])
            obj.max_bd_chi2 = float(data["max_bd_chi2"])
        obj.path = path
        if verbose:
            print(f"[cosmos2025] loaded {len(obj)} pre-filtered galaxies "
                  f"from {path}")
        return obj


# ---------------------------------------------------------------------------
# Pre-filtered catalog cache (built once, read by every worker)
# ---------------------------------------------------------------------------

def _filtered_cache_path(fits_path: str, max_mag: float, max_bd_chi2: float,
                         cache_dir: str | None = None) -> str:
    base = cache_dir or os.path.dirname(os.path.abspath(fits_path)) or "."
    stem = os.path.splitext(os.path.basename(fits_path))[0]
    return os.path.join(base, f"{stem}.filtered_mag{max_mag:g}_chi{max_bd_chi2:g}.npz")


def ensure_prefiltered_catalog(
    fits_path: str | None = None, *,
    max_mag: float = 25.0, max_bd_chi2: float = 10.0,
    cache_dir: str | None = None, verbose: bool = True,
) -> str:
    """Return a path to a small ``.npz`` holding the *filtered* COSMOS2025
    catalog, building it from the master FITS only when no fresh cache exists.

    The full 10 GB FITS is read **once** (here, in the parent) instead of once
    per parallel worker per subset; workers then reload the few-MB ``.npz`` in
    milliseconds. The cache is keyed by the cut parameters and invalidated when
    the source FITS is newer. If ``fits_path`` is already a ``.npz`` it is
    returned unchanged.
    """
    p = fits_path or Config.COSMOS2025_CATALOG_PATH
    if str(p).endswith(".npz"):
        return p
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"COSMOS2025 catalog not found at {p}. Place the master catalog "
            f"there (Shuntov+ 2025) or set Config.COSMOS2025_CATALOG_PATH.")
    npz = _filtered_cache_path(p, max_mag, max_bd_chi2, cache_dir)
    if os.path.isfile(npz) and os.path.getmtime(npz) >= os.path.getmtime(p):
        return npz                                  # fresh cache → reuse
    cat = Cosmos2025Catalog(path=p, max_mag=max_mag,
                            max_bd_chi2=max_bd_chi2, verbose=verbose)
    try:
        npz = _atomic_save_filtered(cat, npz)
    except OSError:                                  # cache dir not writable
        npz = _atomic_save_filtered(
            cat, _filtered_cache_path(p, max_mag, max_bd_chi2,
                                      tempfile.gettempdir()))
    if verbose:
        print(f"[cosmos2025] pre-filtered {len(cat)} galaxies → {npz}")
    return npz


def _atomic_save_filtered(cat: Cosmos2025Catalog, npz: str) -> str:
    os.makedirs(os.path.dirname(npz) or ".", exist_ok=True)
    tmp = npz + ".tmp.npz"
    cat.save_filtered_npz(tmp)
    os.replace(tmp, npz)
    return npz


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def open_cosmos2025(
    path: str | None = None,
    **kwargs,
) -> CosmosCatalog:
    """Open the COSMOS2025 catalog. The catalog is mandatory.

    If ``path`` (or :attr:`Config.COSMOS2025_CATALOG_PATH`) does not exist
    on disk, this raises :class:`FileNotFoundError` rather than falling
    back to a synthetic catalog — the pipeline relies on real photometry
    and morphology and must not be run without it.

    A ``.npz`` path is read as a pre-filtered catalog
    (:meth:`Cosmos2025Catalog.from_filtered_npz`) — the fast path workers use.
    """
    p = path or Config.COSMOS2025_CATALOG_PATH
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"COSMOS2025 catalog not found at {p}. Place the master "
            f"catalog there (Shuntov+ 2025, arXiv:2506.03243) or set "
            f"Config.COSMOS2025_CATALOG_PATH to its location."
        )
    if str(p).endswith(".npz"):
        return Cosmos2025Catalog.from_filtered_npz(
            p, verbose=bool(kwargs.get("verbose", True)))
    return Cosmos2025Catalog(path=p, **kwargs)
