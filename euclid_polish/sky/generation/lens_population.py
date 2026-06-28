"""
Strong-lens population sampling and rasterisation.

Implements the Collett 2015 (arXiv:1507.02657) galaxy-galaxy lens population:
each lens is a Singular Isothermal Ellipsoid (SIE) lens galaxy plus a small
external shear, lensing a Sersic source. ``lenstronomy`` provides *only* the
ray-tracing (``LensModel(['SIE','SHEAR']).ray_shooting``); the rasterisation
of light (both the lens-galaxy's own light and the lensed source) uses the
project's custom Sersic implementation in :mod:`euclid_polish.sky.profiles`.
A single Sersic implementation is shared across galaxies and lenses.

Coordinate conventions:
  * The HR canvas is in pixels at ``Config.DEFAULT_PIXEL_SCALE`` (0.05″/pix).
  * Lens / source positions are in arcsec relative to the *image centre*;
    angles are in radians (CCW from +x).
  * ``theta_E`` is the SIE Einstein radius in arcsec.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from scipy.ndimage import map_coordinates

from euclid_polish.config import Config
from euclid_polish.sky.generation.cosmos2025 import CosmosCatalog, GalaxyParams
from euclid_polish.sky.generation.profiles import (
    add_sersic_to_bands,
    evaluate_sersic_at_coords,
)
from euclid_polish.sky.generation.redshift_model import (
    angular_diameter_distance,
    comoving_distance_mpc,
)
from euclid_polish.sky.generation.tng_galaxy import composite_stamp

# ---------------------------------------------------------------------------
# Per-lens parameter dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LensParams:
    """All physical parameters describing one realised lens system.

    The lens-galaxy light and lensed-source light are both rasterised; both
    use the per-band fluxes stored in the underlying :class:`GalaxyParams`.

    ``centre_x_pix`` and ``centre_y_pix`` are the lens-galaxy centroid in
    HR pixel coordinates on the simulator canvas.
    """

    # Geometry of the lensing mass
    z_lens:           float
    z_source:         float
    theta_E_arcsec:   float
    lens_q:           float           # axis ratio of the SIE
    lens_pa_rad:      float           # major-axis PA of the SIE
    shear_gamma1:     float
    shear_gamma2:     float

    # Source position relative to the lens centre (arcsec, lens-frame)
    src_dx_arcsec:    float
    src_dy_arcsec:    float

    # Lens-galaxy and source-galaxy parametric descriptions. None in the
    # pure-TNG (catalog-free) path, where both lights are real stamps and
    # there is no Sersic fallback.
    lens_galaxy:      GalaxyParams | None
    source_galaxy:    GalaxyParams | None

    # Placement on the simulator canvas (HR pixel coords).
    # None means "use the canvas centre" (the default before placement).
    centre_x_pix:     float | None = None
    centre_y_pix:     float | None = None


# ---------------------------------------------------------------------------
# Cosmological distance helpers (flat ΛCDM, Collett-2015 cosmology)
# ---------------------------------------------------------------------------

# The distance helpers live in :mod:`euclid_polish.sky.redshift_model`
# (shared with the TNG redshift model); these private aliases keep them
# importable from here.
_comoving_distance_mpc = comoving_distance_mpc
_angular_diameter_distance = angular_diameter_distance

#: Observable Einstein-radius window (arcsec): smaller and the arcs are
#: unresolved at Euclid resolution; larger is rarer than the simulated sky.
#: Both samplers rejection-sample θ_E into this window.
THETA_E_RANGE_ARCSEC = (0.10, 3.5)


def einstein_radius_sis(sigma_v_kms: float, z_lens: float, z_source: float) -> float:
    """Einstein radius (arcsec) of a Singular Isothermal Sphere.

    θ_E = 4π σ_v² / c² · D_ls / D_s   (radians) → arcsec.
    """
    c_kms = 299_792.458
    D_s  = _angular_diameter_distance(z_source)
    D_ls = _angular_diameter_distance(z_lens, z_source)
    if D_ls <= 0 or D_s <= 0:
        return 0.0
    theta_E_rad = 4.0 * math.pi * (sigma_v_kms / c_kms) ** 2 * D_ls / D_s
    return float(np.degrees(theta_E_rad) * 3600.0)


# ---------------------------------------------------------------------------
# Module-level sampling helpers (shared by LensPopulation and
# sample_lens_geometry so the logic lives in exactly one place)
# ---------------------------------------------------------------------------

def _sample_shear(rng: np.random.Generator) -> tuple[float, float]:
    """Draw (γ1, γ2) from a zero-mean Gaussian with σ = LENS_EXT_SHEAR_SIGMA."""
    s = Config.LENS_EXT_SHEAR_SIGMA
    return float(rng.normal(0.0, s)), float(rng.normal(0.0, s))


def _sample_disk_offset(
    rng: np.random.Generator, theta_E: float,
) -> tuple[float, float]:
    """Uniform-in-disk source offset within LENS_SOURCE_OFFSET_FRAC × θ_E.

    Returns (dx, dy) in arcsec relative to the lens centre. Sampling
    uniformly in the disk preferentially yields strong-lensing geometries
    (caustic crossings, fold images) over pure weak shear.
    """
    r_max = Config.LENS_SOURCE_OFFSET_FRAC * theta_E
    r = math.sqrt(rng.uniform()) * r_max
    phi = rng.uniform(0.0, 2.0 * math.pi)
    return r * math.cos(phi), r * math.sin(phi)


# ---------------------------------------------------------------------------
# Lens-population sampler
# ---------------------------------------------------------------------------

class LensPopulation:
    """Draws strong-lens realisations from the Collett 2015 priors.

    Each call to :meth:`sample` returns a :class:`LensParams` whose lens galaxy
    and source galaxy come from the supplied :class:`CosmosCatalog`. Strict
    failure modes (no source available at z > z_lens + offset, etc.) raise
    ``RuntimeError`` — the caller can retry a few times.
    """

    def __init__(
        self, catalog: CosmosCatalog, *,
        sigma_v_min_kms: float = Config.LENS_SIGMA_V_MIN_KMS,
        sigma_v_max_kms: float = Config.LENS_SIGMA_V_MAX_KMS,
    ):
        self.catalog = catalog
        self.sigma_v_min_kms = float(sigma_v_min_kms)
        self.sigma_v_max_kms = float(sigma_v_max_kms)

    def _sample_sigma_v(self, rng: np.random.Generator) -> float:
        """Velocity dispersion — uniform in σ_v over the truncation range.

        Uniform σ_v ∈ [min, max] km/s (default [150, 350]) gives θ_E ∈
        roughly [0.3″, 2.0″] at typical lens redshifts. σ_v² sets θ_E via
        the SIS law, so this is the knob on the θ_E spread.
        """
        return float(rng.uniform(self.sigma_v_min_kms, self.sigma_v_max_kms))

    def _sample_shear(self, rng: np.random.Generator) -> tuple[float, float]:
        return _sample_shear(rng)

    def _sample_source_offset(
        self, rng: np.random.Generator, theta_E: float,
    ) -> tuple[float, float]:
        return _sample_disk_offset(rng, theta_E)

    def sample(
        self,
        rng: np.random.Generator,
        *,
        max_retries: int = 16,
        sigma_v_kms: float | None = None,
    ) -> LensParams:
        """Sample one fully populated :class:`LensParams`.

        ``sigma_v_kms`` overrides the uniform σ_v prior — used when the lens
        galaxy is a TNG stamp whose σ_v is derived from the subhalo's stellar
        mass, so the deflector strength matches the light on the canvas.
        """
        last_error: Exception | None = None
        for _ in range(max_retries):
            try:
                lens_gal = self.catalog.sample_lens_galaxy(
                    rng, (Config.LENS_Z_LENS_MIN, Config.LENS_Z_LENS_MAX),
                )
                src_gal  = self.catalog.sample_source_galaxy(rng, lens_gal.z_phot)

                sigma_v = (sigma_v_kms if sigma_v_kms is not None
                           else self._sample_sigma_v(rng))
                theta_E = einstein_radius_sis(
                    sigma_v_kms=sigma_v,
                    z_lens=lens_gal.z_phot,
                    z_source=src_gal.z_phot,
                )
                if not (THETA_E_RANGE_ARCSEC[0] < theta_E
                        < THETA_E_RANGE_ARCSEC[1]):
                    continue   # outside the regime where lensing is observable

                # Lens-galaxy axis ratio: prefer the catalog's bulge_q (the
                # bulge dominates the central mass) clipped to Collett's range.
                q = max(
                    Config.LENS_AXIS_RATIO_MIN,
                    min(Config.LENS_AXIS_RATIO_MAX, lens_gal.bulge_axis_ratio),
                )
                pa = lens_gal.angle_rad
                g1, g2 = self._sample_shear(rng)
                dx, dy = self._sample_source_offset(rng, theta_E)
                return LensParams(
                    z_lens         = lens_gal.z_phot,
                    z_source       = src_gal.z_phot,
                    theta_E_arcsec = theta_E,
                    lens_q         = q,
                    lens_pa_rad    = pa,
                    shear_gamma1   = g1,
                    shear_gamma2   = g2,
                    src_dx_arcsec  = dx,
                    src_dy_arcsec  = dy,
                    lens_galaxy    = lens_gal,
                    source_galaxy  = src_gal,
                )
            except RuntimeError as e:
                last_error = e
                continue
        raise RuntimeError(
            f"LensPopulation.sample exhausted {max_retries} retries; "
            f"last error: {last_error}"
        )


def sample_lens_geometry(
    rng: np.random.Generator,
    sigma_v_kms: float,
    *,
    max_retries: int = 16,
) -> LensParams | None:
    """Catalog-free lens-system geometry from the Collett-2015 priors.

    The pure-TNG path: both lights are real stamps, so no COSMOS rows are
    needed — only the geometry. Redshifts and axis ratio come straight from
    the configured priors, θ_E from the SIS law at ``sigma_v_kms``, PA
    uniform. ``lens_galaxy``/``source_galaxy`` are None — the caller must
    supply stamps for both lights. Returns None if no draw lands in the
    observable θ_E window.
    """
    for _ in range(max_retries):
        z_lens = float(rng.uniform(
            Config.LENS_Z_LENS_MIN, Config.LENS_Z_LENS_MAX))
        z_source = float(rng.uniform(
            z_lens + Config.LENS_Z_SOURCE_OFFSET, Config.LENS_Z_SOURCE_MAX))
        theta_E = einstein_radius_sis(
            sigma_v_kms=sigma_v_kms, z_lens=z_lens, z_source=z_source)
        if not (THETA_E_RANGE_ARCSEC[0] < theta_E < THETA_E_RANGE_ARCSEC[1]):
            continue
        q = float(rng.uniform(
            Config.LENS_AXIS_RATIO_MIN, Config.LENS_AXIS_RATIO_MAX))
        pa = float(rng.uniform(0.0, math.pi))
        g1, g2 = _sample_shear(rng)
        dx, dy = _sample_disk_offset(rng, theta_E)
        return LensParams(
            z_lens         = z_lens,
            z_source       = z_source,
            theta_E_arcsec = theta_E,
            lens_q         = q,
            lens_pa_rad    = pa,
            shear_gamma1   = g1,
            shear_gamma2   = g2,
            src_dx_arcsec  = dx,
            src_dy_arcsec  = dy,
            lens_galaxy    = None,
            source_galaxy  = None,
        )
    return None


# ---------------------------------------------------------------------------
# Lens / source rasterisation
# ---------------------------------------------------------------------------

def _build_lenstronomy_lens(params: LensParams):
    """Construct a ``lenstronomy.LensModel.LensModel`` for ray-shooting.

    Returns ``(lens_model, kwargs_list)``. Heavy import is deferred to here
    so the module is cheap to import when lensing is disabled.
    """
    lens_model = LensModel(lens_model_list=["SIE", "SHEAR"])

    # Convert SIE axis ratio + PA to lenstronomy's eccentricity convention.
    # lenstronomy uses ``e1, e2`` with q = (1-|e|)/(1+|e|) and PA = 0.5 atan2(e2, e1).
    e = (1.0 - params.lens_q) / (1.0 + params.lens_q)
    e1 = e * math.cos(2.0 * params.lens_pa_rad)
    e2 = e * math.sin(2.0 * params.lens_pa_rad)

    kwargs_sie = {
        "theta_E": params.theta_E_arcsec,
        "e1": e1, "e2": e2,
        "center_x": 0.0, "center_y": 0.0,
    }
    kwargs_shear = {
        "gamma1": params.shear_gamma1,
        "gamma2": params.shear_gamma2,
        "ra_0": 0.0, "dec_0": 0.0,
    }
    return lens_model, [kwargs_sie, kwargs_shear]


def _lensed_source_from_stamp(
    stamp: np.ndarray, dx: np.ndarray, dy: np.ndarray, pixel_scale: float,
) -> np.ndarray:
    """Lensed image of a TNG **source stamp**: bilinear-sample the ``(Hs,Ws,4)``
    source-plane stamp (centred at the source, ``pixel_scale`` arcsec/px) at the
    ray-shot source-plane coords ``(dx, dy)`` [arcsec] for every image pixel.

    The stamp is electrons-per-pixel at the same pixel scale as the image, so
    sampling conserves surface brightness and the magnification falls out of the
    geometry (many image pixels mapping to one source region) automatically."""
    Hs, Ws = stamp.shape[:2]
    col = dx / pixel_scale + (Ws - 1) / 2.0
    row = dy / pixel_scale + (Hs - 1) / 2.0
    coords = np.stack([row.ravel(), col.ravel()])
    out = np.empty(dx.shape + (stamp.shape[2],), dtype=np.float32)
    for c in range(stamp.shape[2]):
        out[..., c] = map_coordinates(
            np.asarray(stamp[..., c], dtype=np.float32), coords,
            order=1, mode="constant", cval=0.0).reshape(dx.shape)
    return out


def render_lens_to_multiband_canvas(
    canvas_4ch: np.ndarray,
    *,
    params: LensParams,
    pixel_scale: float = Config.DEFAULT_PIXEL_SCALE,
    lens_light_stamp: np.ndarray | None = None,
    source_stamp: np.ndarray | None = None,
) -> np.ndarray:
    """Add one lens system to a 4-channel canvas in a single pass.

    Renders the morphology *once* (band-independent geometry) and scales
    it into every band by the corresponding per-band flux. This is the
    fast path used by :class:`SkySimulator`; cuts cost from
    ``4 × (2 Sersic + ray-shoot + 2 source-evals)`` down to
    ``2 Sersic + 1 ray-shoot + 2 source-evals`` per lens system.

    Parameters
    ----------
    canvas_4ch  : ``(H, W, 4)`` float32 array, modified in place.
    params      : :class:`LensParams` instance (already placed with
                  ``centre_x_pix`` / ``centre_y_pix`` if non-zero).
    pixel_scale : arcsec/pixel of ``canvas_4ch``.
    lens_light_stamp : optional ``(Hs,Ws,4)`` TNG stamp — when given, the
                  foreground lens-galaxy light is this real-morphology stamp
                  (composited at the lens centre) instead of the analytic B+D
                  Sersic.
    source_stamp : optional ``(Hs,Ws,4)`` TNG stamp — when given, the lensed
                  background source is this real-morphology stamp (ray-shot +
                  bilinear-sampled) instead of the analytic B+D Sersic.

    Returns the updated canvas.
    """
    H, W, C = canvas_4ch.shape
    cx_pix = params.centre_x_pix if params.centre_x_pix is not None else W / 2.0
    cy_pix = params.centre_y_pix if params.centre_y_pix is not None else H / 2.0

    # --- 1. Lens galaxy's own light: real TNG stamp or analytic B+D Sersic ---
    lg = params.lens_galaxy
    if lens_light_stamp is not None:
        composite_stamp(canvas_4ch, lens_light_stamp, cx_pix, cy_pix)
    elif lg is not None:
        add_sersic_to_bands(
            canvas_4ch, flux_per_band=lg.bulge_flux_e, n=4.0,
            r_e=lg.bulge_r_e_arcsec, q=lg.bulge_axis_ratio,
            theta_rad=lg.angle_rad, x0=cx_pix, y0=cy_pix,
            pixel_scale=pixel_scale,
        )
        add_sersic_to_bands(
            canvas_4ch, flux_per_band=lg.disk_flux_e, n=1.0,
            r_e=lg.disk_r_e_arcsec, q=lg.disk_axis_ratio,
            theta_rad=lg.angle_rad, x0=cx_pix, y0=cy_pix,
            pixel_scale=pixel_scale,
        )

    # --- 2. Ray-shooting (band-independent) ---
    lens_model, kw = _build_lenstronomy_lens(params)
    yy_pix, xx_pix = np.indices((H, W), dtype=np.float64)
    x_img_arcsec = (xx_pix - cx_pix) * pixel_scale
    y_img_arcsec = (yy_pix - cy_pix) * pixel_scale
    src_x_flat, src_y_flat = lens_model.ray_shooting(
        x_img_arcsec.ravel(), y_img_arcsec.ravel(), kw,
    )
    src_x = src_x_flat.reshape(H, W)
    src_y = src_y_flat.reshape(H, W)

    # --- 3. Lensed source: real TNG stamp (ray-shot + sampled) or B+D Sersic
    #        evaluated at the ray-shot coords. Morphology band-independent for
    #        Sersic; per-band for the TNG stamp. ---
    sg = params.source_galaxy
    dx = src_x - params.src_dx_arcsec
    dy = src_y - params.src_dy_arcsec

    if source_stamp is not None:
        canvas_4ch += _lensed_source_from_stamp(source_stamp, dx, dy, pixel_scale)
        return canvas_4ch
    if sg is None:
        # Catalog-free geometry with no source stamp: nothing to lens.
        return canvas_4ch

    bulge_unit = evaluate_sersic_at_coords(
        dx, dy, flux=1.0, n=4.0,
        r_e_arcsec=sg.bulge_r_e_arcsec, q=sg.bulge_axis_ratio,
        theta_rad=sg.angle_rad, pixel_scale=pixel_scale,
    ).astype(np.float32)
    disk_unit = evaluate_sersic_at_coords(
        dx, dy, flux=1.0, n=1.0,
        r_e_arcsec=sg.disk_r_e_arcsec, q=sg.disk_axis_ratio,
        theta_rad=sg.angle_rad, pixel_scale=pixel_scale,
    ).astype(np.float32)

    bulge_flux = np.asarray(sg.bulge_flux_e, dtype=np.float32)
    disk_flux  = np.asarray(sg.disk_flux_e,  dtype=np.float32)
    # Broadcast: (H, W) × (4,) → (H, W, 4)
    canvas_4ch += bulge_unit[:, :, None] * bulge_flux[None, None, :]
    canvas_4ch += disk_unit [:, :, None] * disk_flux [None, None, :]
    return canvas_4ch


def render_lens_to_canvas(
    canvas: np.ndarray,
    *,
    params: LensParams,
    band_index: int,
    pixel_scale: float = Config.DEFAULT_PIXEL_SCALE,
) -> np.ndarray:
    """Single-band wrapper for tests & inspection.

    Production code uses :func:`render_lens_to_multiband_canvas` —
    rendering 4 bands at once is ~4× faster because the morphology is
    band-independent. This wrapper synthesises a 4-channel canvas, calls
    the multi-band path, then copies the requested channel back.
    """
    H, W = canvas.shape
    tmp = np.zeros((H, W, Config.NUM_LR_CHANNELS), dtype=np.float32)
    render_lens_to_multiband_canvas(tmp, params=params, pixel_scale=pixel_scale)
    canvas += tmp[..., band_index]
    return canvas
