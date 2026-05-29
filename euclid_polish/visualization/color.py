"""4-band → RGB color rendering for Euclid LR cubes.

The naive (R, G, B) = (H_E, J_E, VIS) stack would make everything look
blue, because the per-band electron count is dominated by which band
has the longest exposure (VIS: 4 × 565 s vs NISP: 4 × 112 s) and the
highest zeropoint — not by the source's actual SED.

The fix is a two-step calibration before mixing:

1. **AB flux normalization.** Divide each band by
   ``t_total_s · 10^(0.4 · zp_AB_per_s)``. After this an AB-flat (flat
   in f_ν, "gray") source produces equal pixel values in every band.
2. **Reference SED whitening.** Multiply each band by a fixed per-band
   factor so that a chosen reference SED renders neutral white. We
   support:

      * ``"ab_flat"``: no extra scaling — AB-flat objects are white.
      * ``"solar"``: divide each band's flux by the sun's flux in that
        band, so sun-like (G2V) stars look white. Most galaxies are
        redder than the sun → render warm; hot stars render blue.

After calibration, three bands feed the Lupton 2004 asinh stretch
(:func:`lupton_rgb`) and the result is clipped to ``[0, 1]``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from euclid_polish.config import Config


# Calibration constants (solar AB mags + default RGB band picks) now live
# in Config.Color — see Config.Color.SOLAR_AB_MAG / Config.Color.RGB_SCHEMES.


# ---------------------------------------------------------------------------
# Per-band flux calibration
# ---------------------------------------------------------------------------

def _ab_flux_norm(band_name: str) -> float:
    """Multiplier that takes e⁻-over-stack → proportional AB flux density.

    Specifically: ``norm = 1 / (t_total_s · 10^(0.4 · zp_AB))``. Pixel
    values multiplied by this constant are in units where an AB-flat
    source has the same value in every band.
    """
    band = Config.get_band(band_name)
    return 1.0 / (band.t_total_s * 10 ** (0.4 * band.zeropoint_ab_e_per_s))


def _solar_balance(band_name: str) -> float:
    """Per-band factor that whitens a solar (G2V) SED on top of
    :func:`_ab_flux_norm`.

    Sun is brighter in VIS than in H_E in absolute f_ν terms (peaks in
    visible), so its AB-flux is ``10^(-0.4 · M_AB_sun_band)``. Dividing
    by that flux gives "in units of solar flux" — sun is (1, 1, 1, 1)
    by construction and any redder/bluer source's color is measured
    relative to sun.
    """
    return 1.0 / (10 ** (-0.4 * Config.Color.SOLAR_AB_MAG[band_name]))


def calibrate(
    cube: np.ndarray,
    band_names: Tuple[str, ...] = Config.LR_INPUT_BAND_NAMES,
    reference: str = "solar",
) -> np.ndarray:
    """Apply per-band flux calibration to a ``(H, W, C)`` LR cube.

    Returns an array with the same shape, where each channel is scaled
    by an ``ab_flux × (solar_balance)`` constant. After this, a
    ``reference``-SED source has equal values across all bands.
    """
    if reference not in ("ab_flat", "solar"):
        raise ValueError(
            f"reference must be 'ab_flat' or 'solar', got {reference!r}"
        )
    out = cube.astype(np.float32, copy=True)
    for k, name in enumerate(band_names):
        factor = _ab_flux_norm(name)
        if reference == "solar":
            factor *= _solar_balance(name)
        out[..., k] *= factor
    return out


# ---------------------------------------------------------------------------
# Lupton 2004 asinh RGB stretch
# ---------------------------------------------------------------------------

def lupton_rgb_from_channels(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    *,
    Q: float = 8.0,
    stretch: float = 1.0,
    minimum: float = 0.0,
) -> np.ndarray:
    """Map three calibrated channels to an ``(H, W, 3)`` RGB image in
    ``[0, 1]`` using the Lupton+ 2004 asinh stretch.

    The asinh transform preserves color ratios (so a source's *hue* is
    independent of its brightness) while compressing the bright tail
    enough that point sources don't blow out and faint extended sources
    remain visible.

    Parameters
    ----------
    r, g, b : ndarray
        Same-shape 2D arrays, already in the same calibrated unit (e.g.
        post-``calibrate``). Negative values are clipped at 0 *after* the
        intensity computation, so a band that's slightly negative from
        sky subtraction doesn't pull the hue out of gamut.
    Q : float
        Asinh "compression" knob. Higher → tighter compression of bright
        pixels; lower → near-linear behaviour. Lupton recommends 5–10
        for typical optical/NIR data.
    stretch : float
        The intensity ``stretch`` is the linear/log knee — pixels with
        intensity ≪ ``stretch`` are linear; ≫ stretch are log-like.
        Defaults to ``1.0`` because :func:`calibrate` already roughly
        normalises to ~unit intensity.
    minimum : float
        Subtracted from each channel before stretching (sky-floor
        offset). Default 0.
    """
    if not (r.shape == g.shape == b.shape):
        raise ValueError(
            f"r/g/b shapes must match; got {r.shape}, {g.shape}, {b.shape}"
        )
    r0 = (r - minimum).astype(np.float64)
    g0 = (g - minimum).astype(np.float64)
    b0 = (b - minimum).astype(np.float64)
    intensity = (r0 + g0 + b0) / 3.0
    # Asinh-stretched intensity. The ratio ``stretched / intensity`` is
    # the per-pixel rescale; multiplying each channel by it preserves the
    # color ratio while remapping the brightness.
    eps = 1e-12
    stretched = np.arcsinh(Q * np.maximum(intensity, 0.0) / stretch) / Q
    # Where intensity is 0 (or negative), set the rescale to 0 to avoid
    # nan / division by zero.
    rescale = np.where(intensity > eps, stretched / np.maximum(intensity, eps), 0.0)
    rgb = np.stack([r0 * rescale, g0 * rescale, b0 * rescale], axis=-1)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def calibrated_rgb_panel(
    cube: np.ndarray,
    band_names: Tuple[str, ...] = Config.LR_INPUT_BAND_NAMES,
    scheme: str = "vis_nisp",
    reference: str = "solar",
    stretch: str = "asinh",
    asinh_scale_e: float = 1000.0,
) -> np.ndarray:
    """4-band cube → RGB with the same ``stretch`` semantics as the
    grayscale ``add_scale_panel``: per-channel ``[p1, p99.5]`` clipping
    on top of the chosen stretch.

    Why this and not ``lupton_rgb``: Lupton applies *one* asinh on the
    summed intensity and rescales channels by a shared factor — great
    for a single composite image, but the asinh "scale" knob has no
    natural per-band analogue. Here we want to drop in as a colour
    replacement for the existing grayscale linear / asinh panels, so we
    do the per-channel thing: calibrate each band, apply the same
    stretch the grayscale panel would, then independently normalise to
    [0, 1] via the [p1, p99.5] window. The colour bar in the title
    ("linear [p1, p99.5]" / "asinh (scale=…)") therefore carries the
    same meaning as in the grayscale panel.
    """
    if stretch not in ("linear", "asinh"):
        raise ValueError(f"stretch must be 'linear' or 'asinh'; got {stretch!r}")
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3-D; got shape {cube.shape}")
    if scheme not in Config.Color.RGB_SCHEMES:
        raise ValueError(f"scheme must be one of {list(Config.Color.RGB_SCHEMES)}; got {scheme!r}")
    idx = {name: i for i, name in enumerate(band_names)}
    rgb_names = Config.Color.RGB_SCHEMES[scheme]
    try:
        chans = [cube[..., idx[n]].astype(np.float64, copy=True) for n in rgb_names]
    except KeyError as exc:
        raise ValueError(
            f"scheme {scheme!r} needs band {exc.args[0]!r}, which is "
            f"not present in band_names={band_names}"
        ) from exc

    # 1. Per-band flux calibration. Same factors as `calibrate()` but
    # applied to the picked R/G/B channels only.
    for i, name in enumerate(rgb_names):
        chans[i] *= _ab_flux_norm(name)
        if reference == "solar":
            chans[i] *= _solar_balance(name)
        elif reference != "ab_flat":
            raise ValueError(f"unknown reference {reference!r}")

    # 2. Apply the requested stretch. For asinh, the knee in calibrated
    # units is what ``asinh_scale_e`` electrons would calibrate to under
    # VIS's normalisation — that keeps the visual "knee" tied to the
    # same physical brightness across linear / asinh panels.
    if stretch == "asinh":
        knee = float(asinh_scale_e) * _ab_flux_norm("VIS")
        if reference == "solar":
            knee *= _solar_balance("VIS")
        if knee <= 0:
            knee = 1.0
        chans = [np.arcsinh(c / knee) for c in chans]

    # 3. Per-channel [p1, p99.5] normalisation so each channel uses its
    # full dynamic range — same convention as the grayscale panels.
    out = []
    for c in chans:
        finite = c[np.isfinite(c)]
        if finite.size == 0:
            out.append(np.zeros_like(c, dtype=np.float32))
            continue
        lo, hi = np.percentile(finite, [1.0, 99.5])
        if hi <= lo:
            hi = lo + 1.0
        out.append(np.clip((c - lo) / (hi - lo), 0.0, 1.0).astype(np.float32))
    return np.stack(out, axis=-1)


def lupton_rgb(
    cube: np.ndarray,
    band_names: Tuple[str, ...] = Config.LR_INPUT_BAND_NAMES,
    scheme: str = "vis_nisp",
    reference: str = "solar",
    *,
    Q: float = 8.0,
    stretch: float = 1.0,
    minimum: float = 0.0,
) -> np.ndarray:
    """High-level convenience: calibrate the cube, pick 3 bands, stretch.

    Parameters
    ----------
    cube : ndarray of shape ``(H, W, C)``
        Multi-band LR data in raw electrons over the stack.
    band_names : sequence of band names matching the channel order of
        ``cube``. Defaults to the project's LR input order
        ``(VIS, Y_E, J_E, H_E)``.
    scheme : ``"vis_nisp" | "nisp_only" | "h_y_vis"`` — which 3 bands
        feed R, G, B.
    reference : ``"ab_flat" | "solar"`` — see :func:`calibrate`.
    Q, stretch, minimum : Lupton stretch parameters; see
        :func:`lupton_rgb_from_channels`.
    """
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3-D (H, W, C); got shape {cube.shape}")
    if len(band_names) != cube.shape[-1]:
        raise ValueError(
            f"band_names length ({len(band_names)}) doesn't match cube "
            f"channel count ({cube.shape[-1]})"
        )
    if scheme not in Config.Color.RGB_SCHEMES:
        raise ValueError(
            f"scheme must be one of {list(Config.Color.RGB_SCHEMES)}; got {scheme!r}"
        )
    calibrated = calibrate(cube, band_names=band_names, reference=reference)
    rgb_bands = Config.Color.RGB_SCHEMES[scheme]
    idx = {name: i for i, name in enumerate(band_names)}
    try:
        r = calibrated[..., idx[rgb_bands[0]]]
        g = calibrated[..., idx[rgb_bands[1]]]
        b = calibrated[..., idx[rgb_bands[2]]]
    except KeyError as exc:
        raise ValueError(
            f"scheme {scheme!r} needs band {exc.args[0]!r}, which is "
            f"not present in band_names={band_names}"
        ) from exc
    return lupton_rgb_from_channels(r, g, b, Q=Q, stretch=stretch, minimum=minimum)
