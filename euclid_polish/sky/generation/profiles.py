"""
Vectorised Sersic-profile rasterisation.

Galaxies are rendered as one or two analytic Sersic profiles (bulge n=4,
disk n=1 — both fixed; a joint Sersic with free n is supported for
single-component fits):

* ``sersic_b_n(n)``           — Ciotti & Bertin 1999 inverse-Gamma constant b_n.
* ``sersic_amp_from_flux(...)``— closed-form amplitude given the total flux.
* ``draw_sersic(...)``        — rasterise one elliptical Sersic onto a grid.
* ``draw_bulge_disk(...)``    — convenience: sum of n=4 bulge + n=1 disk.

The amplitude convention is the standard intensity-at-half-light-radius
definition of Sersic:

    I(r) = amp · exp(-b_n · ((r / R_e)^(1/n) - 1))

with the total flux through an elliptical aperture being

    F = 2π · n · Γ(2n) · exp(b_n) / b_n^(2n) · q · R_e² · amp

(see Graham & Driver 2005, eq. 7). The factor of ``q`` accounts for the
intensity being defined along the major axis while ``R_e`` is also the
major-axis half-light radius.

The same total ``F`` is used to set ``amp`` and analytically conserves flux
when ``I(r)`` is integrated to infinity; integration over a finite stamp
captures whatever fraction of the analytic profile falls on it.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.special import gammaincinv

from euclid_polish.config import Config

# ---------------------------------------------------------------------------
# Sersic geometry helpers
# ---------------------------------------------------------------------------

def sersic_b_n(n: float) -> float:
    """Sersic constant b_n, defined by γ(2n, b_n) = Γ(2n) / 2.

    Computed exactly via ``scipy.special.gammaincinv``.
    """
    if n <= 0:
        raise ValueError(f"Sersic n must be positive, got {n}")
    # gammaincinv(a, y) solves P(a, x) = y where P is the regularised
    # lower incomplete gamma function. b_n satisfies P(2n, b_n) = 0.5.
    return float(gammaincinv(2.0 * n, 0.5))


def sersic_amp_from_flux(
    flux: float, n: float, r_e: float, q: float,
) -> float:
    """Closed-form amplitude that makes the analytic Sersic integrate to ``flux``.

    Parameters
    ----------
    flux : total flux through an elliptical aperture (any units; e.g. e⁻).
    n    : Sersic index.
    r_e  : major-axis half-light radius (same units as the grid the
           profile will be evaluated on — typically arcsec or HR pixels).
    q    : axis ratio b/a in (0, 1].

    Returns the central-amplitude at r=0 in *intensity* units (flux / area).
    """
    if r_e <= 0:
        raise ValueError(f"r_e must be positive, got {r_e}")
    if not (0 < q <= 1):
        raise ValueError(f"q must be in (0, 1], got {q}")
    b_n = sersic_b_n(n)
    # F = 2π · n · Γ(2n) · exp(b_n) / b_n^(2n) · q · R_e² · amp
    denom = (2.0 * math.pi * n * float(gamma_fn(2.0 * n))
             * math.exp(b_n) / (b_n ** (2.0 * n)) * q * (r_e ** 2))
    return float(flux) / denom


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------

def _ellipse_radius_grid(
    x_grid: np.ndarray, y_grid: np.ndarray,
    x0: float, y0: float, q: float, theta_rad: float,
) -> np.ndarray:
    """Return the elliptical isophote radius for each grid point.

    A point (x, y) on an ellipse with axis ratio q and major-axis position
    angle ``theta_rad`` (CCW from +x) has elliptical radius

        r_ell = sqrt( (x'/1)^2 + (y'/q)^2 )

    where x', y' are the source-frame coordinates after translating by
    (x0, y0) and rotating by -theta. ``r_ell == r_e`` is the half-light
    isophote.
    """
    dx = x_grid - x0
    dy = y_grid - y0
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    xp =  c * dx + s * dy
    yp = -s * dx + c * dy
    return np.sqrt(xp * xp + (yp / q) ** 2)


def _default_csub(n: float, r_e_pix: float) -> int:
    """Auto-select sub-pixel sampling factor from :class:`Config` knobs.

    Looks up ``base`` from ``Config.SERSIC_CSUB_BASE_BY_N`` (a tuple of
    ``(n_max, csub)`` rules — first match wins), then applies a
    compactness bump from ``Config.SERSIC_CSUB_COMPACTNESS_BUMPS`` if the
    source is compact + high-n enough that a single pixel near the
    centroid spans many e-folds of intensity. The final csub is forced
    odd (so the centre sub-pixel coincides with the pixel centre) and
    capped at ``Config.SERSIC_CSUB_MAX_CAP``.

    Accuracy is better than 1% for n ≤ 4 and r_e_pix ≥ 2 — the regime of
    physical relevance for COSMOS galaxies on the 0.05″ HR grid.
    """
    # 1. Base csub by n.
    base = None
    for n_max, csub_base in Config.SERSIC_CSUB_BASE_BY_N:
        if n <= n_max:
            base = int(csub_base)
            break
    if base is None:
        base = int(Config.SERSIC_CSUB_BASE_BY_N[-1][1])

    # 2. Compactness bumps. Rules are ordered most-compact-first; the
    #    first matching rule wins.
    for r_e_pix_max, n_min, bump in Config.SERSIC_CSUB_COMPACTNESS_BUMPS:
        if r_e_pix < r_e_pix_max and n > n_min:
            if bump == "x2_plus_1":
                base = base * 2 + 1
            else:
                base = base + int(bump)
            break

    base = min(int(Config.SERSIC_CSUB_MAX_CAP), max(1, base))
    return base if base % 2 == 1 else base + 1


def _default_stamp_radius(r_e_pix: float, n: float) -> float:
    """Half-side, in pixels, beyond which the Sersic is negligible.

    Multipliers in :class:`Config` (``SERSIC_STAMP_RADIUS_R_E_*``).
    Returned value is clamped to at least ``SERSIC_STAMP_RADIUS_MIN_PIX``.
    """
    if n <= 1.5:
        mult = Config.SERSIC_STAMP_RADIUS_R_E_LOW_N
    elif n <= 3.0:
        mult = Config.SERSIC_STAMP_RADIUS_R_E_MID_N
    else:
        mult = Config.SERSIC_STAMP_RADIUS_R_E_HIGH_N
    return max(float(Config.SERSIC_STAMP_RADIUS_MIN_PIX), mult * r_e_pix)


def draw_sersic(
    shape: tuple[int, int],
    *,
    flux: float,
    n: float,
    r_e: float,
    q: float,
    theta_rad: float,
    x0: float, y0: float,
    pixel_scale: float = 1.0,
    csub: int | None = None,
    out: np.ndarray | None = None,
    add: bool = True,
) -> np.ndarray:
    """Rasterise one elliptical Sersic profile onto a 2-D grid.

    The grid spans pixel indices [0, H) × [0, W); centroid ``(x0, y0)``
    is in **pixel units**, ``r_e`` is in **arcsec** (converted internally
    via ``pixel_scale``), and ``flux`` is the total source flux in image
    units (e.g. electrons).

    Parameters
    ----------
    shape          : (H, W) of the output grid.
    flux           : total flux of the source (e⁻).
    n              : Sersic index.
    r_e            : major-axis half-light radius (arcsec).
    q              : axis ratio b/a in (0, 1].
    theta_rad      : major-axis position angle (CCW from +x), in radians.
    x0, y0         : centroid in pixel coords.
    pixel_scale    : arcsec/pixel.
    csub           : odd integer sub-pixel sampling factor. ``None`` → auto.
                     For n>1 the Sersic varies rapidly near the centroid,
                     and a single pixel-centre evaluation under-integrates
                     the core; ``csub`` sub-pixels per pixel cures this at
                     cost ``csub²``. ``csub=1`` disables oversampling.
    out            : pre-allocated output array; created zeros if None.
    add            : if True, ADD into ``out`` (default); if False, OVERWRITE.

    Returns the output array.
    """
    if out is None:
        out = np.zeros(shape, dtype=np.float32)
    elif not add:
        out.fill(0)

    # Render at unit flux once, then scale (scalar broadcast).
    stamp, (i_lo, i_hi, j_lo, j_hi) = compute_sersic_stamp(
        shape, n=n, r_e=r_e, q=q, theta_rad=theta_rad,
        x0=x0, y0=y0, pixel_scale=pixel_scale, csub=csub,
    )
    if stamp.size == 0:
        return out
    out[i_lo:i_hi, j_lo:j_hi] += stamp * np.float32(flux)
    return out


def _eval_sersic_region(
    *,
    i_lo: int, i_hi: int, j_lo: int, j_hi: int,
    csub: int,
    x0: float, y0: float,
    q: float, theta_rad: float, r_e_pix: float, n: float,
    amp: float, b_n: float,
) -> np.ndarray:
    """Evaluate a Sersic on pixels ``[i_lo:i_hi, j_lo:j_hi]`` with ``csub`` oversampling.

    Returns a ``(H_s, W_s)`` float32 array of *unit-flux* integrated pixel
    values. Used internally by :func:`compute_sersic_stamp`.
    """
    H_s, W_s = i_hi - i_lo, j_hi - j_lo
    if csub == 1:
        # Single-pixel evaluation — much faster path when oversampling is
        # not needed (smooth profile / large r_e / low n).
        y_grid, x_grid = np.indices((H_s, W_s), dtype=np.float64)
        x_grid = x_grid + j_lo
        y_grid = y_grid + i_lo
        r_ell = _ellipse_radius_grid(x_grid, y_grid, x0, y0, q, theta_rad)
        r_over_re = np.maximum(r_ell / r_e_pix, 1e-12)
        return (amp * np.exp(-b_n * (r_over_re ** (1.0 / n) - 1.0))).astype(np.float32)

    offsets = (np.arange(csub, dtype=np.float64) + 0.5) / csub - 0.5
    y_sub, x_sub = np.indices((H_s * csub, W_s * csub), dtype=np.float64)
    x_grid = (x_sub // csub) + offsets[(x_sub % csub).astype(np.int64)] + j_lo
    y_grid = (y_sub // csub) + offsets[(y_sub % csub).astype(np.int64)] + i_lo
    r_ell = _ellipse_radius_grid(x_grid, y_grid, x0, y0, q, theta_rad)
    r_over_re = np.maximum(r_ell / r_e_pix, 1e-12)
    sub = amp * np.exp(-b_n * (r_over_re ** (1.0 / n) - 1.0))
    return sub.reshape(H_s, csub, W_s, csub).mean(axis=(1, 3)).astype(np.float32)


def compute_sersic_stamp(
    canvas_shape: tuple[int, int],
    *,
    n: float,
    r_e: float,
    q: float,
    theta_rad: float,
    x0: float, y0: float,
    pixel_scale: float = 1.0,
    csub: int | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Compute a unit-flux Sersic stamp and its location on the canvas.

    Uses a **core+wings** adaptive sampler: the central few R_e where the
    Sersic profile varies steeply gets sampled at the full auto-csub level,
    while the wings (which are smooth and contribute little to per-pixel
    integration error) get csub=1.

    Returns
    -------
    stamp  : 2-D float32 array of shape ``(H_stamp, W_stamp)``, integrated
             to total flux = 1.0 to within the discrete-stamp accuracy.
    bounds : ``(i_lo, i_hi, j_lo, j_hi)`` indices on the canvas of size
             ``canvas_shape``. ``stamp`` is meant to be **added** at
             ``out[i_lo:i_hi, j_lo:j_hi]``.

    Typical caller pattern (multi-band broadcasting):

        stamp, (i_lo, i_hi, j_lo, j_hi) = compute_sersic_stamp(...)
        for k in range(n_bands):
            canvas[i_lo:i_hi, j_lo:j_hi, k] += stamp * flux_per_band[k]

    Empty stamp / out-of-bounds source yields ``stamp.shape == (0, 0)`` and
    the caller's loop is a no-op.
    """
    H, W = canvas_shape
    r_e_pix = r_e / pixel_scale
    if csub is None:
        csub = _default_csub(n, r_e_pix)
    if csub < 1 or csub % 2 == 0:
        raise ValueError(f"csub must be a positive odd integer (got {csub})")

    # Outer stamp: full source extent.
    half = _default_stamp_radius(r_e_pix, n)
    i_lo = max(0, int(math.floor(y0 - half)))
    i_hi = min(H, int(math.ceil (y0 + half + 1)))
    j_lo = max(0, int(math.floor(x0 - half)))
    j_hi = min(W, int(math.ceil (x0 + half + 1)))
    if i_hi <= i_lo or j_hi <= j_lo:
        return np.zeros((0, 0), dtype=np.float32), (i_lo, i_hi, j_lo, j_hi)

    # Unit-flux amplitude (analytic, independent of grid).
    amp = sersic_amp_from_flux(flux=1.0, n=n, r_e=r_e_pix, q=q)
    b_n = sersic_b_n(n)

    common = dict(
        x0=x0, y0=y0, q=q, theta_rad=theta_rad,
        r_e_pix=r_e_pix, n=n, amp=amp, b_n=b_n,
    )

    # Fast path: low requested csub → uniform single-pass evaluation.
    if csub <= int(Config.SERSIC_CSUB_FAST_PATH_THRESHOLD):
        stamp = _eval_sersic_region(
            i_lo=i_lo, i_hi=i_hi, j_lo=j_lo, j_hi=j_hi, csub=csub, **common,
        )
        return stamp, (i_lo, i_hi, j_lo, j_hi)

    # Two-tier path: csub=1 over the full stamp + high csub over a small
    # core. The core radius covers the strongly non-linear region (where
    # the pixel-centre sample disagrees with the pixel-integrated value).
    core_radius_pix = max(4.0, float(Config.SERSIC_CORE_RADIUS_R_E) * r_e_pix)
    core_i_lo = max(i_lo, int(math.floor(y0 - core_radius_pix)))
    core_i_hi = min(i_hi, int(math.ceil (y0 + core_radius_pix + 1)))
    core_j_lo = max(j_lo, int(math.floor(x0 - core_radius_pix)))
    core_j_hi = min(j_hi, int(math.ceil (x0 + core_radius_pix + 1)))

    # 1. Cheap pass: pixel-centre evaluation over the whole stamp.
    stamp = _eval_sersic_region(
        i_lo=i_lo, i_hi=i_hi, j_lo=j_lo, j_hi=j_hi, csub=1, **common,
    )

    # 2. Accurate pass: high-csub evaluation just in the core; overwrite
    #    the corresponding pixels in the cheap stamp.
    if core_i_hi > core_i_lo and core_j_hi > core_j_lo:
        core_stamp = _eval_sersic_region(
            i_lo=core_i_lo, i_hi=core_i_hi,
            j_lo=core_j_lo, j_hi=core_j_hi,
            csub=csub, **common,
        )
        stamp[
            core_i_lo - i_lo : core_i_hi - i_lo,
            core_j_lo - j_lo : core_j_hi - j_lo,
        ] = core_stamp

    return stamp, (i_lo, i_hi, j_lo, j_hi)


def add_sersic_to_bands(
    canvas_multi: np.ndarray,
    flux_per_band: np.ndarray | tuple[float, ...],
    *,
    n: float,
    r_e: float,
    q: float,
    theta_rad: float,
    x0: float, y0: float,
    pixel_scale: float = 1.0,
    csub: int | None = None,
) -> None:
    """Render a Sersic at unit flux ONCE, then scatter-add into every band.

    For a galaxy whose only band-to-band difference is the total flux (i.e.
    same morphology), this renders the morphology a single time. The canvas
    shape is ``(H, W, n_bands)`` and ``flux_per_band`` is a length-``n_bands``
    vector of per-channel electrons.
    """
    if canvas_multi.ndim != 3:
        raise ValueError(f"canvas_multi must be (H, W, C); got {canvas_multi.shape}")
    H, W, C = canvas_multi.shape
    flux = np.asarray(flux_per_band, dtype=np.float32)
    if flux.shape != (C,):
        raise ValueError(
            f"flux_per_band length {flux.shape} != canvas channels {C}"
        )

    stamp, (i_lo, i_hi, j_lo, j_hi) = compute_sersic_stamp(
        (H, W), n=n, r_e=r_e, q=q, theta_rad=theta_rad,
        x0=x0, y0=y0, pixel_scale=pixel_scale, csub=csub,
    )
    if stamp.size == 0:
        return
    # Broadcast: (H_s, W_s) * (C,) → (H_s, W_s, C) without explicit reshape.
    canvas_multi[i_lo:i_hi, j_lo:j_hi, :] += stamp[:, :, None] * flux[None, None, :]


def evaluate_sersic_at_coords(
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    *,
    flux: float,
    n: float,
    r_e_arcsec: float,
    q: float,
    theta_rad: float,
    pixel_scale: float,
) -> np.ndarray:
    """Evaluate a Sersic profile at arbitrary (x, y) arcsec coordinates.

    Unlike :func:`draw_sersic` which rasterises onto a regular pixel grid,
    this evaluates the analytic profile at user-supplied coordinates
    (typically the ray-shot source-plane positions produced by lenstronomy).
    The flux normalisation matches :func:`draw_sersic`: passing identical
    ``flux, n, r_e, q, theta`` and evaluating on a regular grid yields the
    same per-pixel values as ``draw_sersic`` with ``csub=1``.

    Inputs ``x_arcsec, y_arcsec`` are in **arcsec** relative to the source
    centroid; ``pixel_scale`` converts to pixel units for the amplitude
    formula (whose closed form is derived in pixel² area units).
    """
    r_e_pix = r_e_arcsec / pixel_scale
    xp_pix = ( np.cos(theta_rad) * x_arcsec + np.sin(theta_rad) * y_arcsec) / pixel_scale
    yp_pix = (-np.sin(theta_rad) * x_arcsec + np.cos(theta_rad) * y_arcsec) / pixel_scale
    r_ell = np.sqrt(xp_pix * xp_pix + (yp_pix / q) ** 2)

    amp = sersic_amp_from_flux(flux, n=n, r_e=r_e_pix, q=q)
    b_n = sersic_b_n(n)
    r_over_re = np.maximum(r_ell / r_e_pix, 1e-12)
    return amp * np.exp(-b_n * (r_over_re ** (1.0 / n) - 1.0))


def draw_bulge_disk(
    shape: tuple[int, int],
    *,
    flux_bulge: float,
    r_e_bulge: float,
    q_bulge: float,
    flux_disk: float,
    r_e_disk: float,
    q_disk: float,
    theta_rad: float,
    x0: float, y0: float,
    pixel_scale: float = 1.0,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Convenience: bulge (n=4) + disk (n=1) at a shared centroid + PA.

    The bulge and disk share the same centroid and position angle but have
    independent half-light radii, axis ratios, and fluxes. This matches the
    COSMOS2025 B+D decomposition (HDU 6 of the master catalog) where
    ``angle_bd`` is shared but the per-component radii and axis ratios differ.
    """
    if out is None:
        out = np.zeros(shape, dtype=np.float32)
    draw_sersic(shape, flux=flux_bulge, n=4.0, r_e=r_e_bulge, q=q_bulge,
                theta_rad=theta_rad, x0=x0, y0=y0, pixel_scale=pixel_scale,
                out=out, add=True)
    draw_sersic(shape, flux=flux_disk, n=1.0, r_e=r_e_disk, q=q_disk,
                theta_rad=theta_rad, x0=x0, y0=y0, pixel_scale=pixel_scale,
                out=out, add=True)
    return out
