"""
Strong-lens population sampling and rasterisation.

Implements the Collett 2015 (arXiv:1507.02657) galaxy-galaxy lens population:
each lens is a Singular Isothermal Ellipsoid (SIE) lens galaxy plus a small
external shear, lensing a Sersic source. We use ``lenstronomy`` *only* for the
ray-tracing (``LensModel(['SIE','SHEAR']).ray_shooting``); the rasterisation
of light (both the lens-galaxy's own light and the lensed source) uses the
project's custom Sersic implementation in :mod:`euclid_polish.sky.profiles`.

Why the split:
  * Lenstronomy's ``ImageModel`` works fine but couples PSF + pixel-response
    into the renderer. We already have a tested empirical PSF pipeline and a
    fast custom Sersic — we just want the deflection field.
  * Keeping one Sersic implementation across galaxies and lenses means there
    is exactly one place to fix if the photometry is ever wrong.

Coordinate conventions:
  * The HR canvas is in pixels at ``Config.DEFAULT_PIXEL_SCALE`` (0.05″/pix).
  * Lens / source positions are in arcsec relative to the *image centre*;
    angles are in radians (CCW from +x).
  * ``theta_E`` is the SIE Einstein radius in arcsec.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from euclid_polish.config import Config
from euclid_polish.sky.cosmos2025 import CosmosCatalog, GalaxyParams
from euclid_polish.sky.profiles import (
    add_sersic_to_bands,
    draw_sersic,
    evaluate_sersic_at_coords,
)
from euclid_polish.sky.tng_galaxy import composite_stamp

from scipy.integrate import trapezoid
from scipy.ndimage import map_coordinates
from lenstronomy.LensModel.lens_model import LensModel


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

    # Lens-galaxy and source-galaxy parametric descriptions
    lens_galaxy:      GalaxyParams
    source_galaxy:    GalaxyParams

    # Placement on the simulator canvas (HR pixel coords)
    centre_x_pix:     float = 0.0
    centre_y_pix:     float = 0.0


# ---------------------------------------------------------------------------
# Cosmological distance helpers (flat ΛCDM, Collett-2015 cosmology)
# ---------------------------------------------------------------------------

def _comoving_distance_mpc(
    z: float,
    *,
    H0: float = Config.LENS_COSMOLOGY_H0,
    Omega_m: float = Config.LENS_COSMOLOGY_OMEGA_M,
    Omega_L: float = Config.LENS_COSMOLOGY_OMEGA_L,
    n_int: int = 1024,
) -> float:
    """Line-of-sight comoving distance (Mpc) via Simpson rule.

    No external cosmology dependency — we only need distance ratios for θ_E.
    """
    c_kms = 299_792.458
    DH = c_kms / H0       # Hubble distance, Mpc
    zs = np.linspace(0.0, z, n_int + 1)
    E = np.sqrt(Omega_m * (1.0 + zs) ** 3 + Omega_L)
    integrand = 1.0 / E
    # ``np.trapz`` was removed in NumPy 2.0; use ``scipy.integrate.trapezoid``
    # for forward compatibility.
    return DH * float(trapezoid(integrand, zs))


def _angular_diameter_distance(z1: float, z2: Optional[float] = None) -> float:
    """Angular-diameter distance D_A (Mpc).

    If ``z2`` is ``None``: from observer (z=0) to z1. Otherwise from z1 to z2
    in a flat cosmology, using the identity D_A(z1,z2) = (D_C(z2) - D_C(z1)) / (1+z2).
    """
    if z2 is None:
        return _comoving_distance_mpc(z1) / (1.0 + z1)
    return (_comoving_distance_mpc(z2) - _comoving_distance_mpc(z1)) / (1.0 + z2)


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

        Collett 2015 uses a Schechter-like distribution; for our simulation
        the precise functional form doesn't matter as long as the *range*
        of θ_E we produce is realistic. Uniform σ_v ∈ [min, max] km/s (default
        [150, 350]) gives θ_E ∈ roughly [0.3″, 2.0″] at typical lens redshifts
        — σ_v² sets θ_E via the SIS law, so this is the knob on the θ_E spread.
        """
        return float(rng.uniform(self.sigma_v_min_kms, self.sigma_v_max_kms))

    def _sample_shear(self, rng: np.random.Generator) -> Tuple[float, float]:
        s = Config.LENS_EXT_SHEAR_SIGMA
        return float(rng.normal(0.0, s)), float(rng.normal(0.0, s))

    def _sample_source_offset(
        self, rng: np.random.Generator, theta_E: float,
    ) -> Tuple[float, float]:
        """Pick a source position inside ``LENS_SOURCE_OFFSET_FRAC × θ_E``.

        Uniform in the disk — this preferentially yields strong-lensing
        configurations (caustic crossings, fold images) rather than pure
        weak shear. Returns (dx, dy) in arcsec relative to the lens centre.
        """
        r_max = Config.LENS_SOURCE_OFFSET_FRAC * theta_E
        # Uniform-in-disk sample
        r = math.sqrt(rng.uniform()) * r_max
        phi = rng.uniform(0.0, 2.0 * math.pi)
        return r * math.cos(phi), r * math.sin(phi)

    def sample(
        self,
        rng: np.random.Generator,
        *,
        max_retries: int = 16,
    ) -> LensParams:
        """Sample one fully populated :class:`LensParams`."""
        last_error: Optional[Exception] = None
        for _ in range(max_retries):
            try:
                lens_gal = self.catalog.sample_lens_galaxy(
                    rng, (Config.LENS_Z_LENS_MIN, Config.LENS_Z_LENS_MAX),
                )
                src_gal  = self.catalog.sample_source_galaxy(rng, lens_gal.z_phot)

                sigma_v = self._sample_sigma_v(rng)
                theta_E = einstein_radius_sis(
                    sigma_v_kms=sigma_v,
                    z_lens=lens_gal.z_phot,
                    z_source=src_gal.z_phot,
                )
                if not (0.10 < theta_E < 3.5):
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
    lens_light_stamp: Optional[np.ndarray] = None,
    source_stamp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Add one lens system to a 4-channel canvas in a single pass.

    Renders the morphology *once* (band-independent geometry) and scales
    it into every band by the corresponding per-band flux. This is the
    fast path used by :class:`MultiBandSimulator`; cuts cost from
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
    cx_pix = params.centre_x_pix if params.centre_x_pix != 0 else W / 2.0
    cy_pix = params.centre_y_pix if params.centre_y_pix != 0 else H / 2.0

    # --- 1. Lens galaxy's own light: real TNG stamp or analytic B+D Sersic ---
    lg = params.lens_galaxy
    if lens_light_stamp is not None:
        composite_stamp(canvas_4ch, lens_light_stamp, cx_pix, cy_pix)
    else:
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
    """Single-band wrapper kept for tests & inspection.

    Production code should use :func:`render_lens_to_multiband_canvas` —
    rendering 4 bands at once is ~4× faster because the morphology is
    band-independent. This wrapper synthesises a 4-channel canvas, calls
    the multi-band path, then copies the requested channel back.
    """
    H, W = canvas.shape
    tmp = np.zeros((H, W, Config.NUM_LR_CHANNELS), dtype=np.float32)
    render_lens_to_multiband_canvas(tmp, params=params, pixel_scale=pixel_scale)
    canvas += tmp[..., band_index]
    return canvas
