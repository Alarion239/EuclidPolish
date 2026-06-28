"""
Multi-band clean-HR scene generator.

Two independent source populations are rendered onto each field:

  * **Sersic** — analytic bulge+disk galaxies drawn from the COSMOS2025 catalog
    (:mod:`euclid_polish.sky.cosmos2025`). Typically the small/compact end of
    the galaxy size distribution; density set by ``sersic_density_arcmin2``.

  * **TNG** — real TNG50 SKIRT stamps (:mod:`euclid_polish.sky.tng_galaxy`).
    Covers the resolved/extended population; size drawn log-uniformly over
    ``tng_re_arcsec_range`` (or from D_A(z) when ``tng_redshift_mode`` is on).
    Density set by ``tng_density_arcmin2``. Set to 0.0 (default) to disable.

The two populations together produce a merged galaxy size distribution with
a Sersic-dominated compact peak and a TNG-populated extended tail. Stars and
strong lenses are rendered independently of both densities.

The output of :meth:`SkySimulator.simulate_field` is a single :class:`Image`
with ``data`` of shape ``(H, W, 4)`` in **raw electrons** on the 0.05″ HR
grid, one channel per band ordered as :attr:`Config.LR_INPUT_BAND_NAMES`
(``VIS, Y_E, J_E, H_E``).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np

from euclid_polish.config import Config
from euclid_polish.sky.generation.cosmos2025 import CosmosCatalog
from euclid_polish.sky.generation.lens_population import (
    LensPopulation, render_lens_to_multiband_canvas, sample_lens_geometry,
)
from euclid_polish.sky.generation.cosmos2025 import circularized_effective_radius_arcsec
from euclid_polish.sky.generation.profiles import add_sersic_to_bands
from euclid_polish.sky.generation.redshift_model import (
    TNG_NATIVE_PC_PER_PIXEL,
    compactness_factor,
    load_tng_properties,
    physical_pc_to_arcsec,
    predicted_vis_mag,
    sample_galaxy_redshift,
    sample_target_logmass,
    sigma_v_from_stellar_mass,
)
from euclid_polish.sky.generation.tng_galaxy import (
    N_ORIENTATIONS, composite_stamp, list_tng_galaxies, native_halflight_px,
    predict_vis_flux_e, predict_visible_radius_arcsec, sample_tng_stamp,
    tng_stamp_at_redshift,
)
from euclid_polish.image import Image, Role
from euclid_polish.provenance.defaults import mint_id
from euclid_polish.provenance.records import Stamp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SkySimulatorConfig:
    """Field-level config for the multi-band simulator.

    Two independent galaxy populations contribute to each field:
    ``sersic_density_arcmin2`` controls the compact COSMOS-catalog population
    (analytic bulge+disk); ``tng_density_arcmin2`` controls the resolved
    TNG-stamp population (log-uniform R_e over ``tng_re_arcsec_range``).
    Set ``tng_density_arcmin2=0.0`` (default) to disable TNG injection
    entirely — the field is then all-Sersic, which also requires no downloaded
    TNG data.
    """
    image_size:               int   = Config.DEFAULT_IMAGE_SIZE
    pixel_scale:              float = Config.DEFAULT_PIXEL_SCALE     # arcsec/pix
    # Sersic galaxy population (compact, from COSMOS catalog)
    sersic_density_arcmin2:   float = Config.DEFAULT_GAL_DENSITY_ARCMIN2
    # TNG stamp population (resolved, log-uniform size)
    tng_density_arcmin2:      float = 0.0
    tng_re_arcsec_range:      Tuple[float, float] = (0.3, 4.0)
    tng_galaxy_dir:           str   = Config.TNG_SKIRT_DIR
    # Physical-redshift mode for TNG: z from n(z) → D_A sizing + Tolman dimming
    tng_redshift_mode:        bool  = False
    tng_properties_csv:       str   = ""
    # Stars
    star_density_arcmin2:     float = Config.DEFAULT_STAR_DENSITY_ARCMIN2
    star_mag_slope:           float = Config.STAR_MAG_SLOPE
    star_mag_bright:          float = Config.STAR_MAG_BRIGHT
    star_mag_faint:           float = Config.STAR_MAG_FAINT
    # Lenses
    lens_density_arcmin2:     float = Config.LENS_DENSITY_ARCMIN2
    # Keep the foreground lens-galaxy light compact: cap its effective radius at
    # this multiple of the Einstein radius θ_E so arcs are not buried.
    lens_light_re_factor:     float = 0.7
    lens_sigma_v_min_kms:     float = Config.LENS_SIGMA_V_MIN_KMS
    lens_sigma_v_max_kms:     float = Config.LENS_SIGMA_V_MAX_KMS
    lens_theta_e_min_re_ratio: float = Config.LENS_THETA_E_MIN_RE_RATIO
    lens_require_showable:    bool  = False

    def validate(self) -> Tuple[bool, Optional[str]]:
        if self.image_size <= 0:
            return False, "image_size must be positive"
        if self.pixel_scale <= 0:
            return False, "pixel_scale must be positive"
        if min(self.sersic_density_arcmin2, self.tng_density_arcmin2,
               self.star_density_arcmin2, self.lens_density_arcmin2) < 0:
            return False, "densities must be non-negative"
        lo, hi = self.tng_re_arcsec_range
        if not (0.0 < lo <= hi):
            return False, "tng_re_arcsec_range must be (lo, hi) with 0 < lo ≤ hi"
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
# Stars (point sources with fixed colour)
# ---------------------------------------------------------------------------

def _sample_star_mag(
    rng: np.random.Generator, *,
    slope: float, m_bright: float, m_faint: float,
) -> float:
    """Sample one VIS magnitude from the differential stellar number-count law
    ``dN/dm ∝ 10^(slope · m)`` over ``[m_bright, m_faint]``, by inverse-CDF.
    """
    span = float(m_faint) - float(m_bright)
    if span <= 0.0:
        return float(m_bright)
    beta = float(slope) * math.log(10.0)
    u = rng.random()
    if abs(beta) < 1e-9:
        t = u * span
    else:
        t = math.log1p(u * math.expm1(beta * span)) / beta
    return float(m_bright + t)


def _deposit_star(
    canvas_4ch: np.ndarray, x_pix: float, y_pix: float, mag_vis: float,
) -> None:
    """Drop a point source at the nearest HR pixel, replicating across bands."""
    H, W, C = canvas_4ch.shape
    ix = int(round(x_pix))
    iy = int(round(y_pix))
    if not (0 <= ix < W and 0 <= iy < H):
        return
    for k, band_name in enumerate(Config.LR_INPUT_BAND_NAMES):
        band = Config.get_band(band_name)
        mag_k = mag_vis + Config.STAR_BAND_OFFSETS_MAG[band_name]
        flux_k = 10.0 ** (-0.4 * (mag_k - band.sim_zeropoint_e))
        canvas_4ch[iy, ix, k] += np.float32(flux_k)


# ---------------------------------------------------------------------------
# Multi-band simulator
# ---------------------------------------------------------------------------

class SkySimulator:
    """Generates ``(H, W, 4)`` HR clean fields in electrons.

    Two independent galaxy populations are mixed on each field:
    Sersic (compact, from COSMOS) and TNG (resolved, real stamps).
    Set ``tng_density_arcmin2=0.0`` (the default) for the all-Sersic baseline.
    """

    def __init__(
        self,
        catalog: Optional[CosmosCatalog],
        config: Optional[SkySimulatorConfig] = None,
        *,
        lens_population: Optional[LensPopulation] = None,
    ):
        self.catalog = catalog
        self.config  = config or SkySimulatorConfig()
        ok, why = self.config.validate()
        if not ok:
            raise ValueError(f"Invalid generator config: {why}")

        if catalog is None and self.config.sersic_density_arcmin2 > 0.0:
            raise ValueError(
                "catalog=None requires sersic_density_arcmin2=0.0 "
                "(Sersic galaxies need COSMOS rows)")

        # Load TNG galaxies when the TNG population is enabled OR when TNG
        # stamps may be used for lens/source light.
        needs_tng = (self.config.tng_density_arcmin2 > 0.0
                     or self.config.lens_density_arcmin2 > 0.0)
        self.tng_galaxies: List[Tuple[str, str]] = (
            list_tng_galaxies(self.config.tng_galaxy_dir)
            if needs_tng else []
        )
        if self.config.tng_density_arcmin2 > 0.0 and not self.tng_galaxies:
            sys.stderr.write(
                f"[generator] tng_density_arcmin2={self.config.tng_density_arcmin2} "
                f"but no downloaded galaxies under {self.config.tng_galaxy_dir} — "
                "TNG population will be empty.\n")

        # Catalog-backed lens priors (needed when catalog is available).
        self.lens_population: Optional[LensPopulation] = (
            lens_population or LensPopulation(
                catalog,
                sigma_v_min_kms=self.config.lens_sigma_v_min_kms,
                sigma_v_max_kms=self.config.lens_sigma_v_max_kms)
        ) if catalog is not None else None

        # TNG properties for mass → σ_v mapping (redshift mode).
        self.tng_properties: dict = {}
        self._atlas_logm: Optional[np.ndarray] = None
        if self.config.tng_redshift_mode and self.tng_galaxies:
            self.tng_properties = load_tng_properties(
                self.config.tng_properties_csv or None)
            if not self.tng_properties:
                sys.stderr.write(
                    "[generator] tng_redshift_mode: no usable "
                    "tng_properties.csv — lens σ_v falls back to the "
                    "uniform prior; field rescaling to log-uniform.\n")
            else:
                m = np.array([
                    self.tng_properties.get(str(gid), {}).get(
                        "mass_stars", float("nan"))
                    for _, gid in self.tng_galaxies])
                with np.errstate(invalid="ignore"):
                    self._atlas_logm = np.where(m > 0, np.log10(m), np.nan)
                if not np.isfinite(self._atlas_logm).any():
                    self._atlas_logm = None

    # ------------------------------------------------------------------ #
    def _field_area_arcmin2(self) -> float:
        side_arcmin = self.config.image_size * self.config.pixel_scale / 60.0
        return side_arcmin ** 2

    def _random_pix(self, rng: np.random.Generator) -> Tuple[float, float]:
        N = self.config.image_size
        return float(rng.uniform(0.0, N - 1)), float(rng.uniform(0.0, N - 1))

    # ------------------------------------------------------------------ #
    def _pick_field_galaxy(
        self, rng: np.random.Generator,
    ) -> Tuple[List[Tuple[str, str]], float, Optional[float]]:
        """Mass-function-weighted pick for one TNG field stamp (redshift mode).

        Draws the target mass from the Schechter MF, finds atlas galaxies in
        the matching mass window, and returns ``(candidate_list, mass_scale,
        target_logM)``. Falls back to log-uniform rescale when masses are
        missing.
        """
        if self._atlas_logm is None:
            s_min = Config.TNG_MASS_RESCALE_MIN
            if 0.0 < s_min < 1.0:
                return self.tng_galaxies, float(
                    10.0 ** rng.uniform(np.log10(s_min), 0.0)), None
            return self.tng_galaxies, 1.0, None
        lm_t = sample_target_logmass(rng)
        lm = self._atlas_logm
        window = (lm >= lm_t) & (lm <= lm_t + np.log10(Config.TNG_MASS_WINDOW))
        if not window.any():
            window = lm >= lm_t
        if not window.any():
            idx = int(np.nanargmax(lm))
            return [self.tng_galaxies[idx]], 1.0, float(lm[idx])
        idx = int(rng.choice(np.nonzero(window)[0]))
        mass_scale = min(1.0, 10.0 ** (lm_t - lm[idx]))
        return [self.tng_galaxies[idx]], float(mass_scale), float(lm_t)

    # ------------------------------------------------------------------ #
    def _add_tng_galaxy(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator,
        *, z: Optional[float] = None,
    ) -> Optional[dict]:
        """Inject one TNG stamp at a random field position.

        **Non-redshift mode**: R_e drawn log-uniformly over ``tng_re_arcsec_range``.
        **Redshift mode**: z drawn from survey n(z); D_A(z) sets the downsample
        factor with Tolman dimming and spectral drift.
        """
        if self.config.tng_redshift_mode:
            if z is None:
                z = sample_galaxy_redshift(rng)
            galaxies, mass_scale, lm_eff = self._pick_field_galaxy(rng)
            if lm_eff is not None:
                cut = Config.TNG_FAINT_SKIP_MAG_VIS
                if cut > 0 and predicted_vis_mag(lm_eff, z) > cut:
                    return None
            target_re = None
        else:
            galaxies = self.tng_galaxies
            mass_scale = 1.0
            lo, hi = self.config.tng_re_arcsec_range
            target_re = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

        res = sample_tng_stamp(galaxies, rng,
                               pixel_scale_arcsec=self.config.pixel_scale,
                               target_re_arcsec=target_re, z=z,
                               mass_scale=mass_scale)
        if res is None:
            return None
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
            "z":            float(tmeta.get("z", float("nan"))),
            "mass_scale":   float(tmeta.get("mass_scale", float("nan"))),
            "drift_eps":    float(tmeta.get("drift_eps", float("nan"))),
            "target_re_arcsec":   float(tmeta.get("target_re_arcsec", float("nan"))),
            "apparent_re_arcsec": float(tmeta.get("apparent_re_arcsec", float("nan"))),
            "flux_e_per_band": [float(tmeta["flux_e_per_band"][b])
                                for b in Config.LR_INPUT_BAND_NAMES],
        }

    def _render_sersic_galaxy(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator,
    ) -> dict:
        """Rasterise one COSMOS B+D Sersic galaxy at a random field position."""
        g = self.catalog.sample_galaxy(rng)
        x_pix, y_pix = self._random_pix(rng)
        add_sersic_to_bands(
            canvas_4ch, flux_per_band=g.bulge_flux_e, n=4.0,
            r_e=g.bulge_r_e_arcsec, q=g.bulge_axis_ratio,
            theta_rad=g.angle_rad, x0=x_pix, y0=y_pix,
            pixel_scale=self.config.pixel_scale,
        )
        add_sersic_to_bands(
            canvas_4ch, flux_per_band=g.disk_flux_e, n=1.0,
            r_e=g.disk_r_e_arcsec, q=g.disk_axis_ratio,
            theta_rad=g.angle_rad, x0=x_pix, y0=y_pix,
            pixel_scale=self.config.pixel_scale,
        )
        return {
            "type": "galaxy",
            "render": "sersic",
            "catalog_id": g.catalog_id,
            "x_pix": float(x_pix),
            "y_pix": float(y_pix),
            "z_phot": float(g.z_phot),
            "bulge_re_arcsec": float(g.bulge_r_e_arcsec),
            "disk_re_arcsec":  float(g.disk_r_e_arcsec),
            "flux_e_per_band": list(map(float, [g.total_flux_e(k) for k in range(4)])),
        }

    def _add_star(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator,
    ) -> dict:
        x_pix, y_pix = self._random_pix(rng)
        cfg = self.config
        mag = _sample_star_mag(
            rng, slope=cfg.star_mag_slope,
            m_bright=cfg.star_mag_bright, m_faint=cfg.star_mag_faint)
        _deposit_star(canvas_4ch, x_pix, y_pix, mag)
        return {
            "type": "star",
            "x_pix": float(x_pix),
            "y_pix": float(y_pix),
            "mag_vis": float(mag),
        }

    @staticmethod
    def _galaxy_effective_re(g) -> float:
        """Galaxy ``g``'s circularized combined bulge+disk half-light radius."""
        return float(circularized_effective_radius_arcsec(
            np.array([g.bulge_r_e_arcsec]), np.array([g.bulge_axis_ratio]),
            np.array([g.bulge_flux_e[0]]),
            np.array([g.disk_r_e_arcsec]), np.array([g.disk_axis_ratio]),
            np.array([g.disk_flux_e[0]]))[0])

    def _tng_stamp_for_galaxy(
        self, g, rng: np.random.Generator,
        *, target_re_arcsec: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """A TNG stamp sized to match galaxy ``g``'s effective radius (or
        ``target_re_arcsec`` when given). None if TNG is unavailable."""
        if not self.tng_galaxies:
            return None
        target = (target_re_arcsec if target_re_arcsec is not None
                  else self._galaxy_effective_re(g))
        res = sample_tng_stamp(
            self.tng_galaxies, rng, pixel_scale_arcsec=self.config.pixel_scale,
            target_re_arcsec=target)
        return res[0] if res is not None else None

    def _add_lens_pure(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator,
        *, max_tries: int = 32,
    ) -> Optional[dict]:
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
                    pixel_scale_arcsec=cfg.pixel_scale)
                src = tng_stamp_at_redshift(
                    sgdir, sgid, sori, lp.z_source, rng,
                    pixel_scale_arcsec=cfg.pixel_scale)
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
                "sigma_v_proxy_q": float(lp.lens_q),
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
    ) -> Optional[dict]:
        """Render one strong-lens system onto the canvas.

        Routes to :meth:`_add_lens_pure` (catalog-free, TNG-only) when no
        COSMOS catalog is available. Otherwise samples geometry from
        :attr:`lens_population` and overlays TNG stamps for the lens-galaxy
        light and lensed source when TNG galaxies are downloaded.
        """
        cfg = self.config
        if self.lens_population is None:
            # No catalog: pure-TNG geometry path.
            if not self.tng_galaxies:
                return None
            return self._add_lens_pure(canvas_4ch, rng)

        try:
            lp = self.lens_population.sample(rng)
        except RuntimeError:
            return None

        x_pix, y_pix = self._random_pix(rng)
        lp = replace(lp, centre_x_pix=x_pix, centre_y_pix=y_pix)

        # Cap lens light to stay compact inside the arcs.
        lens_eff = self._galaxy_effective_re(lp.lens_galaxy)
        lens_cap = cfg.lens_light_re_factor * float(lp.theta_E_arcsec)
        lens_re = min(lens_eff, lens_cap) if lens_eff > 0 else lens_cap
        lens_scale = (lens_re / lens_eff) if lens_eff > 0 else 1.0

        lp_render = lp
        if lens_scale < 1.0:
            lg = lp.lens_galaxy
            lp_render = replace(lp, lens_galaxy=replace(
                lg, bulge_r_e_arcsec=lg.bulge_r_e_arcsec * lens_scale,
                disk_r_e_arcsec=lg.disk_r_e_arcsec * lens_scale))

        # Use TNG stamps for realistic lens/source morphology when available.
        lens_light_stamp = source_stamp = None
        lens_render = source_render = "sersic"
        if self.tng_galaxies:
            lens_light_stamp = self._tng_stamp_for_galaxy(
                lp.lens_galaxy, rng, target_re_arcsec=lens_re)
            if lens_light_stamp is not None:
                lens_render = "tng"
            source_stamp = self._tng_stamp_for_galaxy(lp.source_galaxy, rng)
            if source_stamp is not None:
                source_render = "tng"

        render_lens_to_multiband_canvas(
            canvas_4ch, params=lp_render, pixel_scale=cfg.pixel_scale,
            lens_light_stamp=lens_light_stamp, source_stamp=source_stamp,
        )
        return {
            "type": "lens",
            "x_pix": float(x_pix),
            "y_pix": float(y_pix),
            "z_lens": float(lp.z_lens),
            "z_source": float(lp.z_source),
            "theta_E_arcsec": float(lp.theta_E_arcsec),
            "sigma_v_proxy_q": float(lp.lens_q),
            "lens_light_render": lens_render,
            "lens_light_re_arcsec": float(lens_re),
            "source_render": source_render,
        }

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
        n_sersic:  Optional[int] = None,
        n_tng:     Optional[int] = None,
        n_stars:   Optional[int] = None,
        n_lenses:  Optional[int] = None,
    ) -> Tuple[Image, dict]:
        """Render one clean HR field in 4 bands.

        Returns
        -------
        sky_image : :class:`Image` with shape ``(image_size, image_size, 4)``
                    in raw electrons.
        metadata  : dict with per-source parameter records.
        """
        cfg  = self.config
        N    = cfg.image_size
        area = self._field_area_arcmin2()

        if n_sersic is None:
            n_sersic = int(rng.poisson(cfg.sersic_density_arcmin2 * area))
        if n_tng is None:
            n_tng = int(rng.poisson(cfg.tng_density_arcmin2 * area))
        if n_stars is None:
            n_stars = int(rng.poisson(cfg.star_density_arcmin2 * area))
        if n_lenses is None:
            n_lenses = int(rng.poisson(cfg.lens_density_arcmin2 * area))

        canvas = np.zeros((N, N, Config.NUM_LR_CHANNELS), dtype=np.float32)
        galaxies, stars, lenses = [], [], []

        for _ in range(n_sersic):
            galaxies.append(self._render_sersic_galaxy(canvas, rng))

        for _ in range(n_tng):
            rec = self._add_tng_galaxy(canvas, rng)
            if rec is not None:
                galaxies.append(rec)

        for _ in range(n_stars):
            stars.append(self._add_star(canvas, rng))

        for _ in range(n_lenses):
            rec = self._add_lens(canvas, rng)
            if rec is not None:
                lenses.append(rec)

        meta = {
            "field_area_arcmin2":      float(area),
            "sersic_density_arcmin2":  float(cfg.sersic_density_arcmin2),
            "tng_density_arcmin2":     float(cfg.tng_density_arcmin2),
            "star_density_arcmin2":    float(cfg.star_density_arcmin2),
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
