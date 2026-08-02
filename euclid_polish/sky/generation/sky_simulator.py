"""
Multi-band clean-HR scene generator.

Every field galaxy uses a resolved TNG50 SKIRT morphology and its native
VIS/NISP proportions. A COSMOS2025 row supplies photometric redshift, stellar
mass, apparent half-light radius, and an F814W brightness anchor. One shared
brightness scale is applied to all TNG bands after the fitted F814W→VIS transfer.

The output of :meth:`SkySimulator.simulate_field` is a single :class:`Image`
with ``data`` of shape ``(H, W, 4)`` in **raw electrons** on the 0.05″ HR
grid, one channel per band ordered as :attr:`Config.LR_INPUT_BAND_NAMES`
(``VIS, Y_E, J_E, H_E``).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, replace

import numpy as np

from euclid_polish.config import Config
from euclid_polish.image import Image, Role
from euclid_polish.photometry import ab_mag_to_electrons
from euclid_polish.provenance.defaults import mint_id
from euclid_polish.provenance.records import Stamp
from euclid_polish.skirt.image import composite_stamp
from euclid_polish.sky.generation.cosmos_tng_prior import CosmosTngPrior
from euclid_polish.sky.generation.lens_population import (
    render_lens_to_multiband_canvas,
    sample_lens_geometry,
)
from euclid_polish.sky.generation.redshift_model import (
    TNG_NATIVE_PC_PER_PIXEL,
    compactness_factor,
    load_tng_properties,
    physical_pc_to_arcsec,
    sigma_v_from_stellar_mass,
)
from euclid_polish.sky.generation.stellar_sed import (
    EmpiricalStellarPrior,
    sample_stellar_sed,
)
from euclid_polish.sky.generation.tng_galaxy import (
    N_ORIENTATIONS,
    list_tng_galaxies,
    native_halflight_px,
    predict_vis_flux_e,
    predict_visible_radius_arcsec,
    sample_tng_stamp,
    tng_stamp_at_redshift,
)
from euclid_polish.sky.generation.tng_radius_manifest import (
    load_manifest, radius_lookup, validate_manifest,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SkySimulatorConfig:
    """Field-level config for the multi-band simulator.

    ``galaxy_density_arcmin2`` controls a single COSMOS-conditioned TNG
    population. There is no analytic Sérsic field-galaxy branch.
    """
    image_size:               int   = Config.DEFAULT_IMAGE_SIZE
    pixel_scale:              float = Config.DEFAULT_PIXEL_SCALE     # arcsec/pix
    galaxy_density_arcmin2:   float = Config.GALAXY_DENSITY_ARCMIN2
    # Calibration-only master density. With a shared field seed, lower-density
    # runs are exact nested thinnings of the same master source proposals.
    galaxy_thinning_max_density_arcmin2: float | None = None
    cosmos_prior_path:        str   = Config.COSMOS_TNG_PRIOR_PATH
    tng_galaxy_dir:           str   = Config.TNG_SKIRT_DIR
    tng_properties_csv:       str   = ""
    tng_radius_manifest_path: str   = os.path.join(
        Config.DATA_DIR, "_tng_infographics", "tng_radius_manifest.json"
    )
    strict_population_artifacts: bool = False
    # Stars
    star_density_arcmin2:     float = Config.DEFAULT_STAR_DENSITY_ARCMIN2
    star_mag_slope:           float = Config.STAR_MAG_SLOPE
    star_mag_bright:          float = Config.STAR_MAG_BRIGHT
    star_mag_faint:           float = Config.STAR_MAG_FAINT
    star_prior_payload:       dict | None = None
    # Lenses
    lens_density_arcmin2:     float = Config.LENS_DENSITY_ARCMIN2
    # Keep the foreground lens-galaxy light compact: cap its effective radius at
    # this multiple of the Einstein radius θ_E so arcs are not buried.
    lens_light_re_factor:     float = 0.7
    lens_sigma_v_min_kms:     float = Config.LENS_SIGMA_V_MIN_KMS
    lens_sigma_v_max_kms:     float = Config.LENS_SIGMA_V_MAX_KMS
    lens_theta_e_min_re_ratio: float = Config.LENS_THETA_E_MIN_RE_RATIO
    lens_require_showable:    bool  = False

    def validate(self) -> tuple[bool, str | None]:
        if self.image_size <= 0:
            return False, "image_size must be positive"
        if self.pixel_scale <= 0:
            return False, "pixel_scale must be positive"
        if min(self.galaxy_density_arcmin2,
               self.star_density_arcmin2, self.lens_density_arcmin2) < 0:
            return False, "densities must be non-negative"
        if (
            self.galaxy_thinning_max_density_arcmin2 is not None
            and self.galaxy_thinning_max_density_arcmin2
            < self.galaxy_density_arcmin2
        ):
            return False, (
                "galaxy_thinning_max_density_arcmin2 must be >= "
                "galaxy_density_arcmin2"
            )
        if self.lens_light_re_factor <= 0.0:
            return False, "lens_light_re_factor must be > 0"
        if not (0.0 < self.lens_sigma_v_min_kms < self.lens_sigma_v_max_kms):
            return False, ("lens_sigma_v_min_kms must be in "
                           "(0, lens_sigma_v_max_kms)")
        if self.star_mag_bright >= self.star_mag_faint:
            return False, "star_mag_bright must be < star_mag_faint"
        if self.lens_theta_e_min_re_ratio <= 0.0:
            return False, "lens_theta_e_min_re_ratio must be > 0"
        return True, None


# ---------------------------------------------------------------------------
# Stars (point sources on a correlated stellar-colour locus)
# ---------------------------------------------------------------------------

def _sample_star_mag(
    rng: np.random.Generator, *,
    slope: float, m_bright: float, m_faint: float,
    stellar_prior: EmpiricalStellarPrior | None = None,
) -> float:
    """Sample one VIS magnitude from the differential stellar number-count law
    ``dN/dm ∝ 10^(slope · m)`` over ``[m_bright, m_faint]``, by inverse-CDF.
    """
    if stellar_prior is None:
        raise ValueError("an active empirical stellar prior is required")
    return stellar_prior.sample_magnitude(
        rng, slope=slope, m_bright=m_bright, m_faint=m_faint,
    )


_STAR_MAG_KEYS = {
    "VIS": "mag_vis",
    "Y_E": "mag_y_e",
    "J_E": "mag_j_e",
    "H_E": "mag_h_e",
}


def _sample_star_band_magnitudes(
    rng: np.random.Generator, mag_vis: float,
) -> dict[str, float]:
    """Compatibility wrapper returning a temperature-driven four-band SED."""
    return sample_stellar_sed(rng, mag_vis).magnitudes


def _cross_validated_mass_bandwidth(logm: np.ndarray) -> float:
    """Choose a Gaussian morphology-kernel bandwidth by leave-one-out CV."""
    values = np.asarray(logm, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.25
    spread = float(np.std(values)) or 0.25
    scale = float(np.median(np.abs(values - np.median(values)))) * 1.4826
    scale = max(scale, spread / max(values.size ** 0.2, 1.0), 0.05)
    grid = np.geomspace(max(0.03, scale / 4.0),
                        min(1.0, max(0.08, scale * 4.0)), 24)
    best_h, best_score = float(grid[0]), -float("inf")
    for h in grid:
        diff = (values[:, None] - values[None, :]) / h
        kernels = np.exp(-0.5 * diff * diff)
        np.fill_diagonal(kernels, 0.0)
        denom = kernels.sum(axis=1)
        score = float(np.log(np.maximum(denom / max(values.size - 1, 1), 1e-300)).sum())
        if score > best_score:
            best_h, best_score = float(h), score
    return best_h


def star_band_magnitudes_from_record(star: dict) -> dict[str, float]:
    """Read persisted star magnitudes, with old-catalog compatibility."""
    mag_vis = float(star["mag_vis"])
    mags = {"VIS": mag_vis}
    for band_name in Config.LR_INPUT_BAND_NAMES[1:]:
        value = star.get(_STAR_MAG_KEYS[band_name])
        mags[band_name] = (
            float(value) if value is not None
            else mag_vis + Config.STAR_BAND_OFFSETS_MAG[band_name]
        )
    return mags


def _deposit_star(
    canvas_4ch: np.ndarray,
    x_pix: float,
    y_pix: float,
    mag_vis: float,
    *,
    band_magnitudes: dict[str, float] | None = None,
) -> None:
    """Drop a point source at the nearest HR pixel in all four bands."""
    H, W, C = canvas_4ch.shape
    ix = int(round(x_pix))
    iy = int(round(y_pix))
    if not (0 <= ix < W and 0 <= iy < H):
        return
    for k, band_name in enumerate(Config.LR_INPUT_BAND_NAMES):
        band = Config.get_band(band_name)
        mag_k = (
            float(band_magnitudes[band_name])
            if band_magnitudes is not None and band_name in band_magnitudes
            else mag_vis + Config.STAR_BAND_OFFSETS_MAG[band_name]
        )
        canvas_4ch[iy, ix, k] += np.float32(ab_mag_to_electrons(mag_k, band))


def inject_random_stars(
    canvas_4ch: np.ndarray, rng: np.random.Generator, *,
    n_stars: int, mag_slope: float, mag_bright: float, mag_faint: float,
    stellar_prior: EmpiricalStellarPrior | None = None,
) -> list[dict]:
    """Draw ``n_stars`` random point sources and DEPOSIT them onto ``canvas_4ch``.

    The shared star primitive: generation uses it to place a field's fixed
    stars (validate/test), and the on-the-fly forward calls it to inject a
    FRESH star realization per visit — stars are HR deltas added *before* the
    PSF/rebin, so the forward gives them realistic shape and the model learns
    to erase them (the target stays starless). Returns the per-star metadata
    (position + four-band magnitudes), so a caller can persist it to the
    source CSV.
    """
    N = canvas_4ch.shape[0]
    stars: list[dict] = []
    for _ in range(int(n_stars)):
        x_pix = float(rng.uniform(0.0, N - 1))
        y_pix = float(rng.uniform(0.0, N - 1))
        mag = _sample_star_mag(rng, slope=mag_slope,
                               m_bright=mag_bright, m_faint=mag_faint,
                               stellar_prior=stellar_prior)
        sed = sample_stellar_sed(rng, mag, stellar_prior)
        band_mags = sed.magnitudes
        _deposit_star(
            canvas_4ch, x_pix, y_pix, mag, band_magnitudes=band_mags,
        )
        stars.append({
            "type": "star", "x_pix": x_pix, "y_pix": y_pix,
            **{_STAR_MAG_KEYS[name]: value for name, value in band_mags.items()},
            "temperature_k": sed.temperature_k,
            "extinction_av": sed.extinction_av,
        })
    return stars


# ---------------------------------------------------------------------------
# Multi-band simulator
# ---------------------------------------------------------------------------

class SkySimulator:
    """Generates ``(H, W, 4)`` HR clean fields in electrons.

    COSMOS supplies the joint population draw; TNG supplies morphology.
    """

    def __init__(
        self,
        population_prior: CosmosTngPrior | None,
        config: SkySimulatorConfig | None = None,
    ):
        self.population_prior = population_prior
        self.config  = config or SkySimulatorConfig()
        ok, why = self.config.validate()
        if not ok:
            raise ValueError(f"Invalid generator config: {why}")
        self.stellar_prior = (
            EmpiricalStellarPrior.from_payload(self.config.star_prior_payload)
            if self.config.star_prior_payload else None
        )
        self._radius_lookup: dict[tuple[str, int], float] | None = None
        self._radius_manifest_fingerprint = ""

        if self.config.strict_population_artifacts:
            if self.config.star_density_arcmin2 > 0.0 and self.stellar_prior is None:
                raise ValueError(
                    "strict population generation requires an active empirical "
                    "stellar prior"
                )
            if (self.config.galaxy_density_arcmin2 > 0.0
                    or self.config.lens_density_arcmin2 > 0.0):
                if population_prior is None:
                    raise ValueError(
                        "strict population generation requires a COSMOS prior"
                    )
                status = validate_manifest(
                    self.config.tng_galaxy_dir,
                    properties_path=self.config.tng_properties_csv or None,
                    manifest_path_value=(
                        self.config.tng_radius_manifest_path or None
                    ),
                )
                if not status.get("valid"):
                    raise ValueError(
                        "TNG radius manifest is not submit-ready: "
                        + "; ".join(status.get("reasons", []))
                    )
                manifest_path = (
                    self.config.tng_radius_manifest_path
                    or os.path.join(self.config.tng_galaxy_dir,
                                    "tng_radius_manifest.json")
                )
                payload = load_manifest(manifest_path)
                if payload is None:
                    raise ValueError("validated TNG radius manifest disappeared")
                self._radius_lookup = radius_lookup(payload)
                self._radius_manifest_fingerprint = str(
                    payload.get("manifest_fingerprint", "")
                )

        if (
            population_prior is None
            and self.config.galaxy_density_arcmin2 > 0.0
        ):
            raise ValueError(
                "population_prior=None requires galaxy_density_arcmin2=0")

        # Load TNG galaxies when the TNG population is enabled OR when TNG
        # stamps may be used for lens/source light.
        needs_tng = (population_prior is not None
                     or self.config.galaxy_density_arcmin2 > 0.0
                     or self.config.lens_density_arcmin2 > 0.0)
        self.tng_galaxies: list[tuple[str, str]] = (
            list_tng_galaxies(self.config.tng_galaxy_dir)
            if needs_tng else []
        )
        if self.config.galaxy_density_arcmin2 > 0.0 and not self.tng_galaxies:
            # HARD failure, not a warning: with a TNG population requested and
            # zero usable stamps, every field silently renders star-only — a
            # buried stderr line shipped 200 galaxy-free validate/test fields
            # when the netscratch purge deleted the SKIRT atlas (2026-07-06).
            raise RuntimeError(
                f"galaxy_density_arcmin2={self.config.galaxy_density_arcmin2:g} but "
                f"ZERO usable TNG galaxies under "
                f"{self.config.tng_galaxy_dir!r} (a galaxy needs its .done "
                "marker + VIS O1 frame — an empty dir usually means the "
                "atlas was purged from netscratch). Re-download it via the "
                "TNG atlas page's download step, or set "
                "galaxy_density_arcmin2=0 for star-only fields.")

        # TNG properties for mass → σ_v mapping (redshift mode).
        self.tng_properties: dict = {}
        self._atlas_logm: np.ndarray | None = None
        if self.tng_galaxies:
            self.tng_properties = load_tng_properties(
                self.config.tng_properties_csv or None)
            m = np.array([
                self.tng_properties.get(str(gid), {}).get(
                    "mass_stars", float("nan")
                ) for _, gid in self.tng_galaxies
            ])
            with np.errstate(invalid="ignore", divide="ignore"):
                self._atlas_logm = np.where(m > 0, np.log10(m), np.nan)
            finite_mass = self._atlas_logm[np.isfinite(self._atlas_logm)]
            self._mass_kernel_bandwidth = _cross_validated_mass_bandwidth(
                finite_mass
            ) if finite_mass.size else None
            if self.config.strict_population_artifacts and (
                not self.tng_properties or not np.isfinite(self._atlas_logm).all()
            ):
                raise ValueError(
                    "strict population generation requires finite mass_stars "
                    "properties for every TNG galaxy"
                )
            if not np.isfinite(self._atlas_logm).any():
                self._atlas_logm = None
        else:
            self._mass_kernel_bandwidth = None

    # ------------------------------------------------------------------ #
    def _field_area_arcmin2(self) -> float:
        side_arcmin = self.config.image_size * self.config.pixel_scale / 60.0
        return side_arcmin ** 2

    def _random_pix(self, rng: np.random.Generator) -> tuple[float, float]:
        N = self.config.image_size
        return float(rng.uniform(0.0, N - 1)), float(rng.uniform(0.0, N - 1))

    # ------------------------------------------------------------------ #
    def _pick_field_galaxy(
        self, rng: np.random.Generator, target_logmass: float,
    ) -> list[tuple[str, str]]:
        """Choose morphology near the COSMOS row's stellar mass."""
        if self._atlas_logm is None:
            if self.config.strict_population_artifacts:
                raise ValueError("TNG morphology mass model has no valid support")
            return self.tng_galaxies
        lm = self._atlas_logm
        finite = np.flatnonzero(np.isfinite(lm))
        if self.config.strict_population_artifacts:
            if not np.isfinite(target_logmass):
                raise ValueError("COSMOS morphology target mass is not finite")
            bandwidth = float(self._mass_kernel_bandwidth or 0.25)
            support = (float(np.min(lm[finite])), float(np.max(lm[finite])))
            if target_logmass < support[0] or target_logmass > support[1]:
                raise ValueError(
                    f"COSMOS mass {target_logmass:.3f} lies outside TNG "
                    f"morphology support [{support[0]:.3f}, {support[1]:.3f}]"
                )
            distance = (lm[finite] - target_logmass) / bandwidth
            weights = np.exp(-0.5 * distance * distance)
            if not np.isfinite(weights).any() or float(weights.sum()) <= 0.0:
                raise ValueError("COSMOS mass lies outside TNG morphology support")
            weights = weights / weights.sum()
            selected = int(rng.choice(finite, p=weights))
            return [self.tng_galaxies[selected]]
        distance = np.abs(lm[finite] - target_logmass)
        nearest = finite[np.argsort(distance)[:min(12, len(finite))]]
        return [self.tng_galaxies[int(rng.choice(nearest))]]

    # ------------------------------------------------------------------ #
    def _add_tng_galaxy(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator,
    ) -> dict | None:
        """Inject one TNG SED conditioned on COSMOS z/mass/size/brightness."""
        if self.population_prior is None:
            return None
        draw = self.population_prior.sample(rng)
        galaxies = self._pick_field_galaxy(rng, draw.logmass)
        res = sample_tng_stamp(
                               galaxies, rng,
                               pixel_scale_arcsec=self.config.pixel_scale,
                               target_re_arcsec=draw.re_arcsec, z=draw.z,
                               target_vis_flux_e=draw.target_vis_flux_e,
                               radius_lookup_map=self._radius_lookup,
                               radius_manifest_fingerprint=(
                                   self._radius_manifest_fingerprint
                               ))
        if res is None:
            raise RuntimeError("TNG population returned no stamp")
        stamp, tmeta = res
        x_pix, y_pix = self._random_pix(rng)
        composite_stamp(canvas_4ch, stamp, x_pix, y_pix)
        return {
            "type": "galaxy",
            "render": "tng",
            "x_pix": float(x_pix),
            "y_pix": float(y_pix),
            "subhalo_id":   tmeta["subhalo_id"],
            "orientation":  tmeta["orientation"],
            "rebin_factor": tmeta["rebin_factor"],
            "rot_k":        tmeta["rot_k"],
            "catalog_id":   draw.catalog_id,
            "z":            draw.z,
            "mass_scale":   1.0,
            "galaxy_density_arcmin2": float(self.config.galaxy_density_arcmin2),
            "population_prior": "cosmos2025_joint",
            "mag_hst_f814w": draw.mag_hst_f814w,
            "target_vis_mag": draw.target_vis_mag,
            "brightness_transfer": draw.brightness_transfer,
            "brightness_scale": float(
                tmeta.get("brightness_scale", float("nan"))
            ),
            "drift_eps":    float(tmeta.get("drift_eps", float("nan"))),
            "target_re_arcsec":   float(tmeta.get("target_re_arcsec", float("nan"))),
            "apparent_re_arcsec": float(tmeta.get("apparent_re_arcsec", float("nan"))),
            # Unified half-light radius + log stellar mass persisted to the
            # source catalog for later analysis.
            "re_arcsec":    draw.re_arcsec,
            "logmass":      draw.logmass,
            "imputed_size": draw.imputed_size,
            "flux_e_per_band": [float(tmeta["flux_e_per_band"][b])
                                for b in Config.LR_INPUT_BAND_NAMES],
        }

    def _draw_star(self, rng: np.random.Generator) -> dict:
        """Draw one star's position + four-band colour — WITHOUT depositing it.

        Deposition is deferred: at generation the base scene stays starless
        (stars are re-added in the forward op — fresh per visit on-the-fly, or
        from this recorded metadata for the fixed validate/test fields)."""
        x_pix, y_pix = self._random_pix(rng)
        cfg = self.config
        mag = _sample_star_mag(
            rng, slope=cfg.star_mag_slope,
            m_bright=cfg.star_mag_bright, m_faint=cfg.star_mag_faint,
            stellar_prior=self.stellar_prior)
        sed = sample_stellar_sed(rng, mag, self.stellar_prior)
        band_mags = sed.magnitudes
        return {
            "type": "star",
            "x_pix": float(x_pix),
            "y_pix": float(y_pix),
            **{_STAR_MAG_KEYS[name]: value for name, value in band_mags.items()},
            "temperature_k": sed.temperature_k,
            "extinction_av": sed.extinction_av,
        }

    def _add_lens_pure(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator,
        *, max_tries: int = 32,
    ) -> dict | None:
        """Catalog-free lens system: TNG deflector + TNG source.

        Used when no COSMOS catalog is available (TNG-only mode). Per try:
        pick a subhalo, derive σ_v from its stellar mass, draw geometry via
        :func:`sample_lens_geometry`, accept when θ_E ≥ κ × apparent R_e.
        """
        cfg = self.config
        kappa = cfg.lens_theta_e_min_re_ratio
        for _ in range(max_tries):
            gdir, gid = self.tng_galaxies[
                int(rng.integers(0, len(self.tng_galaxies)))]
            orientation = int(rng.integers(1, N_ORIENTATIONS + 1))
            props = self.tng_properties.get(str(gid), {})
            mstar = float(props.get("mass_stars", float("nan")))
            sigma_v = sigma_v_from_stellar_mass(mstar, rng)
            if not math.isfinite(sigma_v):
                if cfg.strict_population_artifacts:
                    raise ValueError(
                        f"TNG stellar mass is invalid for strict lens {gid}"
                    )
                sigma_v = float(rng.uniform(cfg.lens_sigma_v_min_kms,
                                            cfg.lens_sigma_v_max_kms))
            lp = sample_lens_geometry(rng, sigma_v)
            if lp is None:
                continue
            re_px = native_halflight_px(gdir, gid, orientation)
            if not (np.isfinite(re_px) and re_px > 0.0):
                continue
            re_app = physical_pc_to_arcsec(
                re_px * TNG_NATIVE_PC_PER_PIXEL,
                lp.z_lens) / compactness_factor(lp.z_lens)
            if lp.theta_E_arcsec < kappa * re_app:
                continue
            sgdir, sgid = self.tng_galaxies[
                int(rng.integers(0, len(self.tng_galaxies)))]
            sori = int(rng.integers(1, N_ORIENTATIONS + 1))
            if cfg.lens_require_showable:
                r_vis = predict_visible_radius_arcsec(
                    gdir, gid, orientation, lp.z_lens,
                    pixel_scale_arcsec=cfg.pixel_scale)
                if (lp.theta_E_arcsec
                        < Config.LENS_SHOWABLE_THETA_E_FRAC * r_vis):
                    continue
                if (predict_vis_flux_e(sgdir, sgid, sori, lp.z_source,
                                       pixel_scale_arcsec=cfg.pixel_scale)
                        < Config.LENS_SHOWABLE_MIN_SRC_VIS_E):
                    continue
            try:
                lens_light_stamp, _ = tng_stamp_at_redshift(
                    gdir, gid, orientation, lp.z_lens, rng,
                    pixel_scale_arcsec=cfg.pixel_scale,
                    native_re_px=(self._radius_lookup or {}).get(
                        (str(gid), orientation)),
                    radius_manifest_fingerprint=(
                        self._radius_manifest_fingerprint
                    ))
                src = tng_stamp_at_redshift(
                    sgdir, sgid, sori, lp.z_source, rng,
                    pixel_scale_arcsec=cfg.pixel_scale,
                    native_re_px=(self._radius_lookup or {}).get(
                        (str(sgid), sori)),
                    radius_manifest_fingerprint=(
                        self._radius_manifest_fingerprint
                    ))
            except Exception:
                continue
            x_pix, y_pix = self._random_pix(rng)
            lp = replace(lp, centre_x_pix=x_pix, centre_y_pix=y_pix)
            render_lens_to_multiband_canvas(
                canvas_4ch, params=lp, pixel_scale=cfg.pixel_scale,
                lens_light_stamp=lens_light_stamp, source_stamp=src[0],
            )
            return {
                "type": "lens",
                "x_pix": float(x_pix),
                "y_pix": float(y_pix),
                "z_lens": float(lp.z_lens),
                "z_source": float(lp.z_source),
                "theta_E_arcsec": float(lp.theta_E_arcsec),
                "sie_axis_ratio": float(lp.lens_q),
                "sigma_v_kms": float(sigma_v),
                "lens_mstar_msun": float(mstar),
                "lens_apparent_re_arcsec": float(re_app),
                "lens_light_render": "tng",
                "lens_light_re_arcsec": float(re_app),
                "source_render": "tng",
                "lens_subhalo_id": str(gid),
                "lens_visible_r_arcsec": float(
                    lens_light_stamp.shape[0] * cfg.pixel_scale / 2.0),
                "source_flux_vis_e": float(src[0][..., 0].sum()),
            }
        return None

    def _add_lens(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator,
    ) -> dict | None:
        """Render one pure-TNG strong-lens system onto the canvas."""
        if not self.tng_galaxies:
            return None
        return self._add_lens_pure(canvas_4ch, rng)

    # ------------------------------------------------------------------ #
    def generate(self, rng=None, *, store=None, **kwargs) -> Image:
        """Generate one clean HR field as a stamped :class:`Image` (role ``'hr'``)."""
        if rng is None:
            rng = np.random.default_rng()
        sky, _meta = self.simulate_field(rng, **kwargs)
        return sky.with_role(Role.HR).with_stamp(
            Stamp(id=mint_id(store), schema_version=3))

    def simulate_field(
        self,
        rng: np.random.Generator,
        *,
        n_galaxies: int | None = None,
        n_stars:   int | None = None,
        n_lenses:  int | None = None,
        deposit_stars: bool = False,
    ) -> tuple[Image, dict]:
        """Render one clean HR field in 4 bands.

        The rendered scene is STARLESS by default (galaxies + lenses only) —
        the network's target. Stars are still DRAWN and returned in
        ``metadata["stars"]`` (positions + four-band magnitudes) so the forward op can
        re-inject them: fresh per visit for on-the-fly training, or from this
        record for the fixed validate/test fields. Pass ``deposit_stars=True``
        to also stamp them onto the canvas (the with-stars scene, e.g. for
        inspection); ``n_stars=0`` draws none (the training split).

        Returns
        -------
        sky_image : :class:`Image` with shape ``(image_size, image_size, 4)``
                    in raw electrons (starless unless ``deposit_stars``).
        metadata  : dict with per-source parameter records.
        """
        cfg  = self.config
        N    = cfg.image_size
        area = self._field_area_arcmin2()

        # Component seeds are consumed once per field. Galaxy-count changes can
        # therefore never shift the star, lens, or forward-model RNG streams.
        component_seeds = rng.integers(
            0, np.iinfo(np.uint64).max, size=3, dtype=np.uint64,
        )
        galaxy_rng = np.random.default_rng(component_seeds[0])
        star_rng = np.random.default_rng(component_seeds[1])
        lens_rng = np.random.default_rng(component_seeds[2])

        galaxy_seeds: list[int]
        if n_galaxies is None:
            master_density = float(
                cfg.galaxy_thinning_max_density_arcmin2
                if cfg.galaxy_thinning_max_density_arcmin2 is not None
                else cfg.galaxy_density_arcmin2
            )
            master_count = int(galaxy_rng.poisson(master_density * area))
            keep_probability = (
                cfg.galaxy_density_arcmin2 / master_density
                if master_density > 0 else 0.0
            )
            keep = galaxy_rng.random(master_count) < keep_probability
            proposals = galaxy_rng.integers(
                0, np.iinfo(np.uint64).max,
                size=master_count, dtype=np.uint64,
            )
            galaxy_seeds = [
                int(seed) for seed, retained
                in zip(proposals, keep, strict=True) if retained
            ]
            n_galaxies = len(galaxy_seeds)
        else:
            galaxy_seeds = [
                int(value) for value in galaxy_rng.integers(
                    0, np.iinfo(np.uint64).max,
                    size=int(n_galaxies), dtype=np.uint64,
                )
            ]
        if n_stars is None:
            n_stars = int(star_rng.poisson(cfg.star_density_arcmin2 * area))
        if n_lenses is None:
            n_lenses = int(lens_rng.poisson(cfg.lens_density_arcmin2 * area))

        canvas = np.zeros((N, N, Config.NUM_LR_CHANNELS), dtype=np.float32)
        galaxies, stars, lenses = [], [], []

        for source_seed in galaxy_seeds:
            rec = self._add_tng_galaxy(
                canvas, np.random.default_rng(source_seed)
            )
            if rec is not None:
                galaxies.append(rec)

        # Stars are DRAWN (recorded) but not deposited — the base stays
        # starless; the forward op re-injects them (see inject_random_stars).
        for _ in range(n_stars):
            star = self._draw_star(star_rng)
            if deposit_stars:
                _deposit_star(
                    canvas,
                    star["x_pix"],
                    star["y_pix"],
                    star["mag_vis"],
                    band_magnitudes=star_band_magnitudes_from_record(star),
                )
            stars.append(star)

        for _ in range(n_lenses):
            rec = self._add_lens(canvas, lens_rng)
            if rec is not None:
                lenses.append(rec)

        meta = {
            "field_area_arcmin2":      float(area),
            "galaxy_density_arcmin2":  float(cfg.galaxy_density_arcmin2),
            "galaxy_thinning_max_density_arcmin2": (
                float(cfg.galaxy_thinning_max_density_arcmin2)
                if cfg.galaxy_thinning_max_density_arcmin2 is not None
                else None
            ),
            "population_prior": "cosmos2025_joint",
            "star_density_arcmin2":    float(cfg.star_density_arcmin2),
            "star_population_fingerprint": (
                str(cfg.star_prior_payload.get("fingerprint", ""))
                if cfg.star_prior_payload else "legacy"
            ),
            "lens_density_arcmin2":    float(cfg.lens_density_arcmin2),
            "n_galaxies": len(galaxies),
            "n_stars":    len(stars),
            "n_lenses":   len(lenses),
            "galaxies":   galaxies,
            "stars":      stars,
            "lenses":     lenses,
        }
        sky = Image(
            data=canvas,
            pixel_scale_arcsec=cfg.pixel_scale,
            band_names=Config.LR_INPUT_BAND_NAMES,
            is_clean=True,
            metadata=meta,
        )
        return sky, meta
