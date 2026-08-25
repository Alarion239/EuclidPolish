"""
Strong-lens geometry sampling and rasterisation.

Implements the Collett 2015 (arXiv:1507.02657) galaxy-galaxy lens population:
each lens is a Singular Isothermal Ellipsoid (SIE) plus a small external
shear, lensing a real TNG galaxy stamp. ``lenstronomy`` provides the
ray-tracing (``LensModel(['SIE','SHEAR']).ray_shooting``); the deflector and
source light both come from the current SKIRT/TNG atlas.

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
from scipy.ndimage import map_coordinates

from euclid_polish.config import Config
from euclid_polish.image.cube import AngularGrid, CubeLike, PixelUnit
from euclid_polish.skirt.image import composite_stamp
from euclid_polish.sky.generation.redshift_model import angular_diameter_distance
from euclid_polish.sky.generation.tng_types import RenderedTNG

# ---------------------------------------------------------------------------
# Per-lens parameter dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LensParams:
    """Physical geometry for one realised TNG-backed lens system.

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

    # Placement on the simulator canvas (HR pixel coords).
    # None means "use the canvas centre" (the default before placement).
    centre_x_pix:     float | None = None
    centre_y_pix:     float | None = None


# ---------------------------------------------------------------------------
# Cosmological distance helpers (flat ΛCDM, Collett-2015 cosmology)
# ---------------------------------------------------------------------------

#: Observable Einstein-radius window (arcsec): smaller and the arcs are
#: unresolved at Euclid resolution; larger is rarer than the simulated sky.
#: Both samplers rejection-sample θ_E into this window.
THETA_E_RANGE_ARCSEC = (0.10, 3.5)


def einstein_radius_sis(sigma_v_kms: float, z_lens: float, z_source: float) -> float:
    """Einstein radius (arcsec) of a Singular Isothermal Sphere.

    θ_E = 4π σ_v² / c² · D_ls / D_s   (radians) → arcsec.
    """
    c_kms = 299_792.458
    D_s = angular_diameter_distance(z_source)
    D_ls = angular_diameter_distance(z_lens, z_source)
    if D_ls <= 0 or D_s <= 0:
        return 0.0
    theta_E_rad = 4.0 * math.pi * (sigma_v_kms / c_kms) ** 2 * D_ls / D_s
    return float(np.degrees(theta_E_rad) * 3600.0)


# ---------------------------------------------------------------------------
# Module-level sampling helpers
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


def sample_lens_geometry(
    rng: np.random.Generator,
    sigma_v_kms: float,
    *,
    max_retries: int = 16,
) -> LensParams | None:
    """Catalog-free lens-system geometry from the Collett-2015 priors.

    Both lights are real stamps, so no COSMOS rows are needed. Redshifts and
    axis ratio come from the configured priors, θ_E from the SIS law at
    ``sigma_v_kms``, and PA is uniform. Returns None if no draw lands in the
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
    from lenstronomy.LensModel.lens_model import LensModel

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


LensStamp = CubeLike | RenderedTNG


def _validated_stamp_data(
    stamp: LensStamp,
    *,
    name: str,
    pixel_scale: float,
) -> np.ndarray:
    """Return one validated rendered-stamp array for the lensing kernel."""
    cube: CubeLike
    if isinstance(stamp, RenderedTNG):
        cube = stamp.cube
    elif isinstance(stamp, CubeLike):
        cube = stamp
    else:
        raise TypeError(
            f"{name} must be a CubeLike image or RenderedTNG, "
            f"got {type(stamp).__name__}"
        )

    if cube.unit is not PixelUnit.ELECTRONS_PER_PIXEL:
        raise ValueError(
            f"{name} must contain electrons/pixel, got {cube.unit.value!r}"
        )
    if not isinstance(cube.grid, AngularGrid):
        raise ValueError(f"{name} must use an angular grid")
    if not np.isclose(
        cube.grid.pixel_scale_arcsec,
        pixel_scale,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{name} pixel scale {cube.grid.pixel_scale_arcsec!r} arcsec/pixel "
            f"does not match canvas pixel scale {pixel_scale!r} arcsec/pixel"
        )
    expected_bands = tuple(Config.LR_INPUT_BAND_NAMES)
    if cube.bands != expected_bands:
        raise ValueError(
            f"{name} bands must be {expected_bands!r}, got {cube.bands!r}"
        )

    data = np.asarray(cube.data)
    if data.ndim != 3 or data.shape[-1] != len(expected_bands):
        raise ValueError(
            f"{name} must have shape (H, W, {len(expected_bands)}), "
            f"got {data.shape!r}"
        )
    return data


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
    lens_light_stamp: LensStamp,
    source_stamp: LensStamp,
) -> np.ndarray:
    """Add one lens system to a 4-channel canvas in a single pass.

    This is the only lens renderer used by :class:`SkySimulator`: it places
    the foreground TNG stamp, ray-shoots once, and samples the four-band
    source stamp at the mapped source-plane coordinates.

    Parameters
    ----------
    canvas_4ch  : ``(H, W, 4)`` float32 array, modified in place.
    params      : :class:`LensParams` instance (already placed with
                  ``centre_x_pix`` / ``centre_y_pix`` if non-zero).
    pixel_scale : arcsec/pixel of ``canvas_4ch``.
    lens_light_stamp : foreground electron :class:`~euclid_polish.image.CubeLike`
                       or :class:`~euclid_polish.sky.generation.tng_types.RenderedTNG`.
    source_stamp : background electron cube or rendered TNG stamp. Both stamps
                   must use the canvas angular grid and canonical band order.

    Returns the updated canvas.
    """
    H, W, _ = canvas_4ch.shape
    lens_light_data = _validated_stamp_data(
        lens_light_stamp,
        name="lens_light_stamp",
        pixel_scale=pixel_scale,
    )
    source_data = _validated_stamp_data(
        source_stamp,
        name="source_stamp",
        pixel_scale=pixel_scale,
    )
    cx_pix = params.centre_x_pix if params.centre_x_pix is not None else W / 2.0
    cy_pix = params.centre_y_pix if params.centre_y_pix is not None else H / 2.0

    # --- 1. Foreground lens-galaxy light ---
    composite_stamp(canvas_4ch, lens_light_data, cx_pix, cy_pix)

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

    # --- 3. Lensed source stamp (ray-shot + sampled) ---
    dx = src_x - params.src_dx_arcsec
    dy = src_y - params.src_dy_arcsec
    canvas_4ch += _lensed_source_from_stamp(source_data, dx, dy, pixel_scale)
    return canvas_4ch
