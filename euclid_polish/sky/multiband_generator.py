"""
Multi-band clean-HR scene generator.

A self-contained renderer (no GalSim) that uses:

  * :mod:`euclid_polish.sky.profiles` for Sersic rasterisation (galaxies,
    bulge+disk, lens galaxy light, lensed source light)
  * :mod:`euclid_polish.sky.cosmos2025` as the parametric galaxy/source
    catalog
  * :mod:`euclid_polish.sky.lens_population` for the lens-population priors
    and lensed-source ray-shooting

The default path is fully analytic; when ``tng_fraction > 0`` a fraction of
galaxies (and lens/source light) is replaced by real TNG50 SKIRT stamps via
:mod:`euclid_polish.sky.tng_galaxy`. ``tng_fraction == 1`` is **pure-TNG
mode**: every source is a redshift-realistic stamp
(:mod:`euclid_polish.sky.redshift_model`), nothing Sersic is rendered, and
the COSMOS catalog is optional.

The output of :meth:`MultiBandSimulator.simulate_field` is a single
:class:`MultiBandSkyImage` with ``data`` of shape ``(H, W, 4)`` in **raw
electrons** on the 0.05″ HR grid, one channel per band ordered as
:attr:`Config.LR_INPUT_BAND_NAMES` (``VIS, Y_E, J_E, H_E``).

Stars are drawn with a single fixed G-type SED — the per-band magnitude
offsets live in :attr:`Config.STAR_BAND_OFFSETS_MAG`.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from euclid_polish.config import Config
from euclid_polish.sky.cosmos2025 import CosmosCatalog
from euclid_polish.sky.lens_population import (
    LensPopulation, render_lens_to_multiband_canvas, sample_lens_geometry,
)
from euclid_polish.sky.apparent_size import CosmosSizeSampler
from euclid_polish.sky.cosmos2025 import circularized_effective_radius_arcsec
from euclid_polish.sky.profiles import add_sersic_to_bands, draw_sersic
from euclid_polish.sky.redshift_model import (
    TNG_NATIVE_PC_PER_PIXEL,
    compactness_factor,
    load_tng_properties,
    physical_pc_to_arcsec,
    sample_galaxy_redshift,
    sigma_v_from_stellar_mass,
)
from euclid_polish.sky.tng_galaxy import (
    N_ORIENTATIONS, composite_stamp, list_tng_galaxies, native_halflight_px,
    sample_tng_stamp, tng_stamp_at_redshift,
)
from euclid_polish.sky.types import MultiBandSkyImage

from dataclasses import replace


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultiBandGeneratorConfig:
    """Field-level config for the multi-band simulator.

    The synthetic generator is fully analytic: all galaxies render via
    Sersic B+D from the COSMOS catalog. Real-HST-morphology training
    data lives in a separate stream (``fasrc_generate_hst_tfrecords``)
    and round-trip data lives in another
    (``fasrc_generate_euclid_roundtrip_tfrecords``); both are mixed at
    the dataloader level, not inside the simulator.
    """
    image_size:               int   = Config.DEFAULT_IMAGE_SIZE
    pixel_scale:              float = Config.DEFAULT_PIXEL_SCALE     # arcsec/pix
    gal_density_arcmin2:      float = Config.DEFAULT_GAL_DENSITY_ARCMIN2
    star_density_arcmin2:     float = Config.DEFAULT_STAR_DENSITY_ARCMIN2
    # Smooth stellar magnitude distribution: dN/dm ∝ 10^(slope·m) over
    # [bright, faint] — the high-Galactic-latitude star-count law.
    star_mag_slope:           float = Config.STAR_MAG_SLOPE
    star_mag_bright:          float = Config.STAR_MAG_BRIGHT
    star_mag_faint:           float = Config.STAR_MAG_FAINT
    lens_density_arcmin2:     float = Config.LENS_DENSITY_ARCMIN2
    # Keep the foreground lens-galaxy light compact: cap its effective radius at
    # this multiple of the Einstein radius θ_E. Real lens ellipticals have
    # R_e ~ θ_E, but the on-sky footprint runs several × R_e (extended wings), so
    # an uncapped lens sprawls over (and dwarfs) the lensed source arcs. 0.7 keeps
    # the lens core comfortably inside θ_E; lower it for even more compact lenses.
    # Applies to both the TNG-stamp and the Sersic lens light.
    lens_light_re_factor:     float = 0.7
    # Lens velocity-dispersion range (km/s) — uniform σ_v draw that sets the
    # Einstein radius via the SIS law θ_E ∝ σ_v² (D_ls/D_s). [150,350] gives
    # θ_E ~ 0.3-2.0". Widening/raising this shifts the θ_E distribution.
    lens_sigma_v_min_kms:     float = Config.LENS_SIGMA_V_MIN_KMS
    lens_sigma_v_max_kms:     float = Config.LENS_SIGMA_V_MAX_KMS
    # Fraction of galaxies drawn as real TNG50 SKIRT stamps instead of analytic
    # Sersic profiles (per galaxy). 0 keeps generation exactly as before.
    tng_fraction:             float = 0.0
    tng_galaxy_dir:           str   = Config.TNG_SKIRT_DIR
    # When True, each injected TNG galaxy is downsampled so its apparent angular
    # half-light radius is drawn from the COSMOS catalog's own effective-radius
    # distribution (:class:`CosmosSizeSampler`) — so TNG galaxies match the
    # Sersic population by construction. When False, the legacy flat
    # ×1/×2/×3/×4 draw is used.
    tng_realistic_sizes:      bool  = True
    # Genuinely big galaxies are their OWN population at a fixed sky surface
    # density — INDEPENDENT of tng_fraction (which only governs the small
    # field-galaxy TNG/Sersic mix). Legacy sizing path only: redshift mode has
    # no separate big population. They are always rendered as real TNG stamps
    # at a large size (R_e log-uniform over ``tng_big_re_arcsec``). The default
    # density ≈ 1 big galaxy per 15 stamps of 512² @ 0.05″/px (0.182 arcmin²);
    # the count per field is Poisson(density · area). Set to 0 to disable.
    # NOTE these are big in *half-light radius*; because a bright compact bulge +
    # extended disk has an on-sky FOOTPRINT several × R_e (de Vaucouleurs wings),
    # even a ~1-4" R_e galaxy can fill a 25" stamp — so they are kept rare.
    big_galaxy_density_arcmin2: float = 0.37
    tng_big_re_arcsec:        Tuple[float, float] = (1.0, 4.0)
    # Physical-redshift mode for TNG injection: one z draw per stamp sets
    # its downsample factor (via D_A), Tolman dimming, and a randomized
    # spectral drift — replacing the COSMOS-matched target-size draw (see
    # sky/redshift_model.py). TNG-lit lens galaxies take σ_v from the
    # subhalo's stellar mass and must satisfy θ_E ≥
    # lens_theta_e_min_re_ratio × apparent half-light radius. False keeps
    # generation byte-identical to before. Implied by tng_fraction == 1
    # (pure-TNG mode, see the class docstring).
    tng_redshift_mode:        bool  = False
    # Property catalog for the mass→σ_v mapping; "" → the local cache
    # written by the TNG-infographic render.
    tng_properties_csv:       str   = ""
    lens_theta_e_min_re_ratio: float = Config.LENS_THETA_E_MIN_RE_RATIO
    # Field-galaxy density in PURE-TNG mode: the real sky density of
    # atlas-like massive galaxies (see Config.TNG_GAL_DENSITY_ARCMIN2),
    # NOT the full COSMOS density — the atlas has no faint dwarfs, so the
    # full count would fill every field with giants.
    tng_gal_density_arcmin2:  float = Config.TNG_GAL_DENSITY_ARCMIN2
    # Pure-TNG dwarf backfill: COSMOS Sersic rows (R_e ≤ the cut) supply the
    # faint small population TNG cannot. Needs a catalog; ≤ 0 disables.
    tng_dwarf_density_arcmin2: float = Config.TNG_DWARF_SERSIC_DENSITY_ARCMIN2
    tng_dwarf_max_re_arcsec:   float = Config.TNG_DWARF_MAX_RE_ARCSEC

    def validate(self) -> Tuple[bool, Optional[str]]:
        if self.image_size <= 0:
            return False, "image_size must be positive"
        if self.pixel_scale <= 0:
            return False, "pixel_scale must be positive"
        if min(self.gal_density_arcmin2, self.star_density_arcmin2,
               self.lens_density_arcmin2, self.tng_gal_density_arcmin2) < 0:
            return False, "densities must be non-negative"
        if not (0.0 <= self.tng_fraction <= 1.0):
            return False, "tng_fraction must be in [0, 1]"
        if self.big_galaxy_density_arcmin2 < 0.0:
            return False, "big_galaxy_density_arcmin2 must be ≥ 0"
        if self.lens_light_re_factor <= 0.0:
            return False, "lens_light_re_factor must be > 0"
        if not (0.0 < self.lens_sigma_v_min_kms < self.lens_sigma_v_max_kms):
            return False, ("lens_sigma_v_min_kms must be in "
                           "(0, lens_sigma_v_max_kms)")
        if self.star_mag_bright >= self.star_mag_faint:
            return False, "star_mag_bright must be < star_mag_faint"
        lo, hi = self.tng_big_re_arcsec
        if not (0.0 < lo <= hi):
            return False, "tng_big_re_arcsec must be (lo, hi) with 0 < lo ≤ hi"
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

    A single smooth, monotonic distribution (replacing the old 3-bin prior):
    most stars sit near the faint limit, with a thin tail to the bright cap.
    ``slope`` is the high-Galactic-latitude star-count slope ``d log N / dm``
    (~0.14–0.35 in the optical/NIR; Euclid observes away from the plane). The
    inverse-CDF uses ``log1p``/``expm1`` for numerical stability; ``slope→0``
    degenerates to uniform.
    """
    span = float(m_faint) - float(m_bright)
    if span <= 0.0:
        return float(m_bright)
    beta = float(slope) * math.log(10.0)
    u = rng.random()
    if abs(beta) < 1e-9:
        t = u * span                                  # flat counts → uniform
    else:
        t = math.log1p(u * math.expm1(beta * span)) / beta
    return float(m_bright + t)


def _deposit_star(
    canvas_4ch: np.ndarray, x_pix: float, y_pix: float, mag_vis: float,
) -> None:
    """Drop a point source at the nearest HR pixel, replicating across bands.

    The per-band magnitude is ``mag_vis + STAR_BAND_OFFSETS_MAG[band]`` and
    each band's flux is computed in its own per-stack zeropoint.
    """
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

class MultiBandSimulator:
    """Generates ``(H, W, 4)`` HR clean fields in electrons.

    Each channel is the clean image of the same scene seen in band
    ``Config.LR_INPUT_BAND_NAMES[k]``. Geometry of every source is
    band-independent; per-band flux normalisations come from the catalog
    (galaxies/lenses) or from the fixed stellar colour (stars).

    ``tng_fraction == 1`` (with downloaded TNG galaxies) is **pure-TNG
    mode**: redshift mode is forced on, lens systems are sampled
    catalog-free (:func:`sample_lens_geometry`, σ_v from the subhalo
    mass), and the big-galaxy population is dropped. TNG renders the
    massive population (``tng_gal_density_arcmin2``); small COSMOS Sersic
    rows backfill the faint dwarfs (``tng_dwarf_density_arcmin2``) when a
    catalog is supplied — ``catalog=None`` is allowed and renders TNG
    only.
    """

    def __init__(
        self,
        catalog: Optional[CosmosCatalog],
        config: Optional[MultiBandGeneratorConfig] = None,
        *,
        lens_population: Optional[LensPopulation] = None,
    ):
        self.catalog = catalog
        self.config  = config or MultiBandGeneratorConfig()
        ok, why = self.config.validate()
        if not ok:
            raise ValueError(f"Invalid generator config: {why}")
        # Load the injectable TNG galaxy list once (only when enabled, so the
        # default path does no filesystem work).
        self.tng_galaxies: List[Tuple[str, str]] = (
            list_tng_galaxies(self.config.tng_galaxy_dir)
            if self.config.tng_fraction > 0.0 else []
        )
        if self.config.tng_fraction > 0.0 and not self.tng_galaxies:
            sys.stderr.write(
                f"[generator] tng_fraction={self.config.tng_fraction} but no "
                f"downloaded galaxies under {self.config.tng_galaxy_dir} — "
                "falling back to all-Sersic.\n")
        # Pure-TNG mode: every galaxy / lens light / lensed source is a real
        # stamp, so redshift mode is mandatory (a stamp's size and photometry
        # need a z) and the COSMOS catalog is never consulted.
        self.pure_tng: bool = (self.config.tng_fraction >= 1.0
                               and bool(self.tng_galaxies))
        if self.pure_tng and not self.config.tng_redshift_mode:
            self.config = replace(self.config, tng_redshift_mode=True)
        if catalog is None and not self.pure_tng:
            raise ValueError(
                "catalog=None requires pure-TNG mode: tng_fraction == 1 with "
                "downloaded TNG galaxies (anything Sersic needs COSMOS rows)")
        # Catalog-backed lens priors only outside pure mode — the pure path
        # samples geometry directly (sample_lens_geometry) with σ_v from the
        # subhalo's stellar mass.
        self.lens_population = None if self.pure_tng else (
            lens_population or LensPopulation(
                catalog,
                sigma_v_min_kms=self.config.lens_sigma_v_min_kms,
                sigma_v_max_kms=self.config.lens_sigma_v_max_kms))
        # Realistic apparent-size sampler for TNG injection: drawn from the
        # COSMOS catalog's own effective-radius distribution (+ rare big tail),
        # so TNG galaxies match the Sersic population by construction.
        # Redshift mode replaces it with the D_A(z) sizing.
        self.tng_size_model: Optional[CosmosSizeSampler] = None
        if (self.catalog is not None and self.config.tng_fraction > 0.0
                and self.config.tng_realistic_sizes and self.tng_galaxies):
            # Small field galaxies: pure COSMOS-anchored sizes (no big tail —
            # big galaxies are a separate fixed-density population, below).
            self.tng_size_model = CosmosSizeSampler(
                self.catalog.effective_re_arcsec, big_fraction=0.0,
            )
        # Redshift mode: subhalo properties (stellar mass → σ_v for TNG-lit
        # lenses). Missing CSV / missing rows degrade gracefully to the
        # uniform σ_v prior.
        self.tng_properties: dict = {}
        if self.config.tng_redshift_mode and self.tng_galaxies:
            self.tng_properties = load_tng_properties(
                self.config.tng_properties_csv or None)
            if not self.tng_properties:
                sys.stderr.write(
                    "[generator] tng_redshift_mode: no usable "
                    "tng_properties.csv — lens σ_v falls back to the "
                    "uniform prior.\n")

    # ------------------------------------------------------------------ #
    def _field_area_arcmin2(self) -> float:
        side_arcmin = self.config.image_size * self.config.pixel_scale / 60.0
        return side_arcmin ** 2

    def _random_pix(self, rng: np.random.Generator) -> Tuple[float, float]:
        N = self.config.image_size
        return float(rng.uniform(0.0, N - 1)), float(rng.uniform(0.0, N - 1))

    # ------------------------------------------------------------------ #
    def _add_tng_galaxy(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator,
        *, target_re_arcsec: Optional[float] = None,
        z: Optional[float] = None,
    ) -> Optional[dict]:
        """Inject a random downloaded TNG galaxy (random orientation +
        realistic-size downsample + quarter-rotation), centred at a random field
        position and clipped to the canvas. Returns None if the stamp can't be
        loaded.

        Sizing: an explicit ``z`` (or ``tng_redshift_mode``, which draws one
        from the survey n(z)) routes through the full redshift treatment —
        downsample from D_A(z), Tolman dimming, randomized spectral drift.
        Otherwise ``target_re_arcsec`` overrides the apparent half-light
        radius; when None it is drawn from the COSMOS catalog's own size
        distribution (:class:`CosmosSizeSampler`), matching the Sersic field
        population."""
        target_re = target_re_arcsec
        if z is None and self.config.tng_redshift_mode:
            z = sample_galaxy_redshift(rng)
        if z is None and target_re is None and self.tng_size_model is not None:
            target_re = self.tng_size_model.sample(rng)
        res = sample_tng_stamp(self.tng_galaxies, rng,
                               pixel_scale_arcsec=self.config.pixel_scale,
                               target_re_arcsec=target_re, z=z)
        if res is None:
            return None
        stamp, tmeta = res
        x_pix, y_pix = self._random_pix(rng)
        composite_stamp(canvas_4ch, stamp, x_pix, y_pix)
        return {
            "type": "galaxy",
            "render": "tng",
            "big": False,
            "x_pix": float(x_pix),
            "y_pix": float(y_pix),
            "subhalo_id":   tmeta["subhalo_id"],
            "orientation":  tmeta["orientation"],
            "rebin_factor": tmeta["rebin_factor"],
            "rot_k":        tmeta["rot_k"],
            "z":            float(tmeta.get("z", float("nan"))),
            "drift_eps":    float(tmeta.get("drift_eps", float("nan"))),
            "target_re_arcsec":   float(tmeta.get("target_re_arcsec", float("nan"))),
            "apparent_re_arcsec": float(tmeta.get("apparent_re_arcsec", float("nan"))),
            "flux_e_per_band": [float(tmeta["flux_e_per_band"][b])
                                for b in Config.LR_INPUT_BAND_NAMES],
        }

    def _add_big_galaxy(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator,
    ) -> Optional[dict]:
        """Inject one genuinely big galaxy — always a real TNG stamp, sized to
        a large apparent half-light radius drawn log-uniformly over
        ``tng_big_re_arcsec``. Legacy path only: redshift mode has no separate
        big population (see :meth:`simulate_field`)."""
        lo, hi = self.config.tng_big_re_arcsec
        target = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        rec = self._add_tng_galaxy(canvas_4ch, rng, target_re_arcsec=target)
        if rec is not None:
            rec["big"] = True
        return rec

    def _add_galaxy(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator,
    ) -> Optional[dict]:
        """Render one galaxy. With probability ``config.tng_fraction`` it's a
        real TNG50 stamp; otherwise a Sersic B+D from the COSMOS catalog row.
        In pure-TNG mode there is no Sersic fallback (no catalog): a failed
        stamp load is retried on other draws, then the slot is dropped (None).

        Geometry band-independent; per-band fluxes from the catalog
        drive the photometry. Each Sersic component is rendered once and
        broadcast-scaled into every channel by its per-band flux — cuts
        Sersic evaluations by ``NUM_LR_CHANNELS=4×`` without changing
        the result.
        """
        cfg = self.config
        if self.pure_tng:
            for _ in range(3):
                rec = self._add_tng_galaxy(canvas_4ch, rng)
                if rec is not None:
                    return rec
            return None
        # Short-circuit on tng_fraction==0 so the default path consumes no extra
        # RNG and stays byte-identical to the all-Sersic generator.
        if (cfg.tng_fraction > 0.0 and self.tng_galaxies
                and rng.random() < cfg.tng_fraction):
            rec = self._add_tng_galaxy(canvas_4ch, rng)
            if rec is not None:
                return rec
            # TNG load failed → don't waste the slot, fall through to Sersic.
        return self._render_sersic_galaxy(
            canvas_4ch, rng, self.catalog.sample_galaxy(rng))

    def _render_sersic_galaxy(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator, g,
    ) -> dict:
        """Rasterise one COSMOS B+D Sersic galaxy at a random field position."""
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
            "big": False,
            "catalog_id": g.catalog_id,
            "x_pix": float(x_pix),
            "y_pix": float(y_pix),
            "z_phot": float(g.z_phot),
            "bulge_re_arcsec": float(g.bulge_r_e_arcsec),
            "disk_re_arcsec":  float(g.disk_r_e_arcsec),
            "flux_e_per_band": list(map(float, [g.total_flux_e(k) for k in range(4)])),
        }

    def _add_dwarf_galaxy(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator,
        *, max_tries: int = 50,
    ) -> dict:
        """Pure-TNG dwarf backfill: a *small* COSMOS Sersic galaxy
        (circularized R_e ≤ ``tng_dwarf_max_re_arcsec`` — bigger rows are
        TNG's job). At these sizes the profile is unresolvable after the
        PSF, so the analytic render is observationally exact. Settles for
        the last draw if no small row shows up."""
        cut = self.config.tng_dwarf_max_re_arcsec
        for _ in range(max_tries):
            g = self.catalog.sample_galaxy(rng)
            if cut <= 0.0 or self._galaxy_effective_re(g) <= cut:
                break
        rec = self._render_sersic_galaxy(canvas_4ch, rng, g)
        rec["dwarf"] = True
        return rec

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
        """A TNG stamp sized to ``target_re_arcsec`` (default: galaxy ``g``'s
        circularized effective radius). None if TNG is unavailable / can't load."""
        if self.tng_size_model is None or not self.tng_galaxies:
            return None
        target = (target_re_arcsec if target_re_arcsec is not None
                  else self._galaxy_effective_re(g))
        res = sample_tng_stamp(
            self.tng_galaxies, rng, pixel_scale_arcsec=self.config.pixel_scale,
            target_re_arcsec=target)
        return res[0] if res is not None else None

    def _sample_tng_lens_system(
        self, rng: np.random.Generator, *, max_tries: int = 32,
    ) -> Optional[tuple]:
        """Pick a TNG subhalo as the deflector of one lens system.

        σ_v from the subhalo's stellar mass (Faber–Jackson; uniform prior if
        the catalog row is missing), θ_E from the SIS law at the drawn
        (z_lens, z_source). Rejected until θ_E ≥ lens_theta_e_min_re_ratio ×
        the galaxy's apparent half-light radius at z_lens, so the arcs clear
        the foreground light.

        Returns ``(lp, gdir, gid, orientation, sigma_v, mstar, re_app)`` or
        None when no visible configuration is found.
        """
        kappa = self.config.lens_theta_e_min_re_ratio
        for _ in range(max_tries):
            gdir, gid = self.tng_galaxies[
                int(rng.integers(0, len(self.tng_galaxies)))]
            orientation = int(rng.integers(1, N_ORIENTATIONS + 1))
            props = self.tng_properties.get(str(gid), {})
            mstar = float(props.get("mass_stars", float("nan")))
            sigma_v = sigma_v_from_stellar_mass(mstar, rng)
            try:
                lp = self.lens_population.sample(
                    rng, sigma_v_kms=(sigma_v if math.isfinite(sigma_v)
                                      else None))
            except RuntimeError:
                return None
            re_px = native_halflight_px(gdir, gid, orientation)
            if not (np.isfinite(re_px) and re_px > 0.0):
                continue
            re_app = physical_pc_to_arcsec(
                re_px * TNG_NATIVE_PC_PER_PIXEL,
                lp.z_lens) / compactness_factor(lp.z_lens)
            if lp.theta_E_arcsec >= kappa * re_app:
                return lp, gdir, gid, orientation, sigma_v, mstar, re_app
        return None

    def _add_lens_pure(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator,
        *, max_tries: int = 32,
    ) -> Optional[dict]:
        """Catalog-free lens system: TNG deflector + TNG source.

        Per try: pick a subhalo (σ_v from its stellar mass, uniform prior if
        unknown), draw the geometry via :func:`sample_lens_geometry`, and
        accept only when θ_E ≥ κ × the deflector's apparent half-light radius
        at z_lens. Both lights are real stamps prepared at their own
        redshifts. Returns None when no visible system materialises."""
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
            try:
                lens_light_stamp, _ = tng_stamp_at_redshift(
                    gdir, gid, orientation, lp.z_lens, rng,
                    pixel_scale_arcsec=cfg.pixel_scale)
            except Exception:
                continue
            src = sample_tng_stamp(self.tng_galaxies, rng,
                                   pixel_scale_arcsec=cfg.pixel_scale,
                                   z=lp.z_source)
            if src is None:
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
                # Arc-prominence diagnostics: the deflector's VISIBLE radius
                # (the μ-truncated stamp's half-size — several × the
                # half-light radius) and the surviving source flux. A lens
                # is eye-visible roughly when θ_E clears half the visible
                # radius AND the source kept enough flux after dimming —
                # most honest draws don't, just like most real lenses.
                "lens_visible_r_arcsec": float(
                    lens_light_stamp.shape[0] * cfg.pixel_scale / 2.0),
                "source_flux_vis_e": float(src[0][..., 0].sum()),
            }
        return None

    def _add_lens(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator,
    ) -> Optional[dict]:
        cfg = self.config
        if self.pure_tng:
            return self._add_lens_pure(canvas_4ch, rng)
        # Redshift mode: with probability tng_fraction the deflector is a TNG
        # subhalo — mass-derived σ_v, visibility-constrained θ_E. The gate
        # draws no RNG when the mode is off, keeping the legacy path
        # byte-identical.
        z_mode = (cfg.tng_redshift_mode and cfg.tng_fraction > 0.0
                  and bool(self.tng_galaxies))
        tng_lens_pick = None
        if z_mode and rng.random() < cfg.tng_fraction:
            tng_lens_pick = self._sample_tng_lens_system(rng)
            if tng_lens_pick is None:
                return None
        if tng_lens_pick is None:
            try:
                lp = self.lens_population.sample(rng)
            except RuntimeError:
                return None
        else:
            lp = tng_lens_pick[0]
        x_pix, y_pix = self._random_pix(rng)
        lp = replace(lp, centre_x_pix=x_pix, centre_y_pix=y_pix)

        # Keep the lens light compact: cap its effective radius at
        # lens_light_re_factor × θ_E so it doesn't sprawl over the source arcs.
        lens_eff = self._galaxy_effective_re(lp.lens_galaxy)
        lens_cap = cfg.lens_light_re_factor * float(lp.theta_E_arcsec)
        lens_re = min(lens_eff, lens_cap) if lens_eff > 0 else lens_cap
        lens_scale = (lens_re / lens_eff) if lens_eff > 0 else 1.0
        # For the Sersic lens light, shrink the rendered component radii to match.
        lp_render = lp
        if lens_scale < 1.0:
            lg = lp.lens_galaxy
            lp_render = replace(lp, lens_galaxy=replace(
                lg, bulge_r_e_arcsec=lg.bulge_r_e_arcsec * lens_scale,
                disk_r_e_arcsec=lg.disk_r_e_arcsec * lens_scale))

        # Lens light and lensed source are each, independently, a real TNG stamp
        # with probability tng_fraction (same proportion as field galaxies);
        # otherwise an analytic B+D Sersic. tng_fraction==0 draws no extra RNG.
        lens_light_stamp = source_stamp = None
        lens_render = source_render = "sersic"
        sigma_v_kms = mstar = re_app = float("nan")
        use_tng = (cfg.tng_fraction > 0.0 and self.tng_size_model is not None
                   and bool(self.tng_galaxies))
        if tng_lens_pick is not None:
            # Redshift mode: the picked subhalo's light, prepared at z_lens
            # (no resizing/shrinking — its visibility was enforced by the
            # θ_E ≥ κ·R_e rejection above). Load failure falls back to the
            # capped Sersic light.
            _, gdir, gid, orientation, sigma_v_kms, mstar, re_app = \
                tng_lens_pick
            try:
                lens_light_stamp, _ = tng_stamp_at_redshift(
                    gdir, gid, orientation, lp.z_lens, rng,
                    pixel_scale_arcsec=cfg.pixel_scale)
                lens_render = "tng"
                lens_re = re_app
            except Exception:
                lens_light_stamp = None
        elif not z_mode and use_tng and rng.random() < cfg.tng_fraction:
            lens_light_stamp = self._tng_stamp_for_galaxy(
                lp.lens_galaxy, rng, target_re_arcsec=lens_re)
            if lens_light_stamp is not None:
                lens_render = "tng"
        if z_mode:
            # TNG source rendered as it would appear at z_source: D_A sizing
            # + dimming + drift (high-z sources are compact and red-drifted).
            if rng.random() < cfg.tng_fraction:
                res = sample_tng_stamp(
                    self.tng_galaxies, rng,
                    pixel_scale_arcsec=cfg.pixel_scale, z=lp.z_source)
                if res is not None:
                    source_stamp = res[0]
                    source_render = "tng"
        elif use_tng and rng.random() < cfg.tng_fraction:
            source_stamp = self._tng_stamp_for_galaxy(lp.source_galaxy, rng)
            if source_stamp is not None:
                source_render = "tng"
        # Fast path: render the lens once into the 4-channel canvas (geometry
        # is band-independent; only per-band fluxes differ).
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
            "sigma_v_kms": float(sigma_v_kms),
            "lens_mstar_msun": float(mstar),
            "lens_apparent_re_arcsec": float(re_app),
            "lens_light_render": lens_render,
            "lens_light_re_arcsec": float(lens_re),
            "source_render": source_render,
        }

    # ------------------------------------------------------------------ #
    def simulate_field(
        self,
        rng: np.random.Generator,
        *,
        n_galaxies: Optional[int] = None,
        n_stars:    Optional[int] = None,
        n_lenses:   Optional[int] = None,
        n_big:      Optional[int] = None,
        n_dwarfs:   Optional[int] = None,
    ) -> Tuple[MultiBandSkyImage, dict]:
        """Render one clean HR field in 4 bands.

        Returns
        -------
        sky_image : :class:`MultiBandSkyImage` with ``data`` shape
                    ``(image_size, image_size, 4)`` in raw electrons.
        metadata  : dict with per-source parameter records.
        """
        cfg = self.config
        N   = cfg.image_size

        area = self._field_area_arcmin2()
        if n_galaxies is None:
            # Pure-TNG mode: every draw is an atlas (massive) galaxy, so the
            # count follows the massive-galaxy sky density, not the full
            # COSMOS density.
            density = (cfg.tng_gal_density_arcmin2 if self.pure_tng
                       else cfg.gal_density_arcmin2)
            n_galaxies = int(rng.poisson(density * area))
        if n_stars is None:
            n_stars    = int(rng.poisson(cfg.star_density_arcmin2 * area))
        if n_lenses is None:
            n_lenses   = int(rng.poisson(cfg.lens_density_arcmin2 * area))
        # Big galaxies: a fixed-density population, independent of tng_fraction.
        # Gated on TNG being enabled so tng_fraction==0 stays the pure-Sersic
        # baseline (no extra RNG drawn → byte-identical). Redshift mode has no
        # separate big population at all — the realistic n(z) already yields
        # big nearby galaxies at the rate the sky does.
        if cfg.tng_redshift_mode:
            n_big = 0
        big_enabled = (cfg.tng_fraction > 0.0 and bool(self.tng_galaxies)
                       and cfg.big_galaxy_density_arcmin2 > 0.0
                       and not cfg.tng_redshift_mode)
        if n_big is None:
            n_big = (int(rng.poisson(cfg.big_galaxy_density_arcmin2 * area))
                     if big_enabled else 0)
        # Pure-TNG dwarf backfill: small COSMOS Sersic galaxies supply the
        # faint population the massive-only atlas cannot.
        dwarf_enabled = (self.pure_tng and self.catalog is not None
                         and cfg.tng_dwarf_density_arcmin2 > 0.0)
        if n_dwarfs is None:
            n_dwarfs = (int(rng.poisson(cfg.tng_dwarf_density_arcmin2 * area))
                        if dwarf_enabled else 0)
        elif not dwarf_enabled:
            n_dwarfs = 0

        canvas = np.zeros((N, N, Config.NUM_LR_CHANNELS), dtype=np.float32)

        galaxies, stars, lenses = [], [], []
        for _ in range(n_galaxies):
            rec = self._add_galaxy(canvas, rng)
            if rec is not None:      # pure-TNG: a failed stamp drops the slot
                galaxies.append(rec)
        for _ in range(n_big):
            rec = self._add_big_galaxy(canvas, rng)
            if rec is not None:
                galaxies.append(rec)
        for _ in range(n_dwarfs):
            galaxies.append(self._add_dwarf_galaxy(canvas, rng))
        for _ in range(n_stars):
            stars.append(self._add_star(canvas, rng))
        for _ in range(n_lenses):
            rec = self._add_lens(canvas, rng)
            if rec is not None:
                lenses.append(rec)

        n_big_rendered = sum(1 for g in galaxies if g.get("big"))
        meta = {
            "field_area_arcmin2":     float(area),
            "galaxy_density_arcmin2": float(cfg.tng_gal_density_arcmin2
                                            if self.pure_tng
                                            else cfg.gal_density_arcmin2),
            "star_density_arcmin2":   float(cfg.star_density_arcmin2),
            "lens_density_arcmin2":   float(cfg.lens_density_arcmin2),
            "big_galaxy_density_arcmin2": float(cfg.big_galaxy_density_arcmin2),
            "n_galaxies": len(galaxies),
            "n_big_galaxies": int(n_big_rendered),
            "n_dwarf_galaxies": sum(1 for g in galaxies if g.get("dwarf")),
            "n_stars":    len(stars),
            "n_lenses":   len(lenses),
            "galaxies":   galaxies,
            "stars":      stars,
            "lenses":     lenses,
        }
        sky = MultiBandSkyImage(
            data=canvas,
            pixel_scale_arcsec=cfg.pixel_scale,
            band_names=Config.LR_INPUT_BAND_NAMES,
            is_clean=True,
            metadata=meta,
        )
        return sky, meta
