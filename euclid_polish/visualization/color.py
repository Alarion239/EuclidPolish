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

A third renderer, :func:`eye_rgb`, is the PHYSICAL mode: per-pixel
blackbody color-temperature fit → CIE Planckian-locus chromaticity →
sRGB, with an absolute (image-independent) luminance transfer — every
pixel renders with the hue the dark-adapted eye would assign to a star
of that SED temperature, and the same source renders identically in
every image. :func:`planck_color_strip` provides the hue ↔ T legend.
"""

from __future__ import annotations

import numpy as np

from euclid_polish.config import Config
from euclid_polish.photometry import ab_mag_to_electrons

# Calibration constants (solar AB mags + default RGB band picks) now live
# in Config.Color — see Config.Color.SOLAR_AB_MAG / Config.Color.RGB_SCHEMES.


# ---------------------------------------------------------------------------
# Per-band flux calibration
# ---------------------------------------------------------------------------

def _ab_flux_norm(band_name: str) -> float:
    """Multiplier that takes e⁻-over-stack → proportional AB flux density.

    ``norm = 1 / ab_mag_to_electrons(0, band)`` — one over the electrons an
    AB = 0 source deposits over the band's stack, so pixel values multiplied
    by this constant are the flux relative to an AB-0 source and an AB-flat
    SED has the same value in every band. (Algebraically identical to the
    historical ``1 / (t_total_s · 10^(0.4·zp_rate))`` — same anchor, now
    routed through the canonical conversion.)
    """
    return 1.0 / float(ab_mag_to_electrons(0.0, Config.get_band(band_name)))


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
    band_names: tuple[str, ...] = Config.LR_INPUT_BAND_NAMES,
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
    band_names: tuple[str, ...] = Config.LR_INPUT_BAND_NAMES,
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


# ---------------------------------------------------------------------------
# "Eye" physical color mode — what a (very NIR-sensitive) eye would see
# ---------------------------------------------------------------------------
#
# The Euclid bands span 0.55–2.0 μm, mostly invisible to a human eye, so
# "true color" cannot be read off the pixels directly. What CAN be made
# rigorous is the chain
#
#     band fluxes → SED shape → blackbody color temperature T
#                 → CIE Planckian-locus chromaticity → sRGB hue,
#
# i.e. render every pixel with the color a blackbody of the *fitted*
# temperature would have to the dark-adapted eye: the Sun (~5800 K)
# renders near-white (slightly warm vs the D65 display white point),
# cool/red SEDs render orange like Betelgeuse, hot ones blue-white like
# Rigel — the actual night-sky experience.
#
# Crucially the mapping is ABSOLUTE: a pixel's RGB is a pure function of
# its physical (SED, surface brightness), with no per-image percentile
# windows — the same source renders the same color in every cutout, every
# panel, every training run, so an SR-vs-HR hue difference is evidence of
# a reconstruction error, never a rendering artifact.

# Planckian-locus chromaticity approximation (Kim et al. 2002 / CIE),
# valid for 1667 K ≤ T ≤ 25000 K — clamp outside.
_EYE_T_MIN = 1667.0
_EYE_T_MAX = 25000.0


def _planck_fnu(wavelength_um: np.ndarray, T: float) -> np.ndarray:
    """Blackbody f_ν (arbitrary normalisation) at ``wavelength_um`` for ``T``.

    ``B_ν ∝ ν³ / (exp(hν/kT) − 1)``; with ν = c/λ the exponent is
    ``h·c / (λ·k·T) = 14387.77 μm·K / (λ·T)`` (second radiation constant).
    """
    lam = np.asarray(wavelength_um, dtype=np.float64)
    x = 14387.77 / (lam * float(T))
    # expm1 keeps precision for the long-wavelength (small-x) limit.
    return (1.0 / lam) ** 3 / np.expm1(x)


def _planckian_xy(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CIE 1931 (x, y) chromaticity of a blackbody at ``T`` (K).

    Cubic-in-1/T fits to the Planckian locus (Kim et al. 2002), the
    standard approximation; inputs are clamped to [1667, 25000] K.
    """
    T = np.clip(np.asarray(T, dtype=np.float64), _EYE_T_MIN, _EYE_T_MAX)
    u = 1e3 / T          # kK⁻¹
    x = np.where(
        T <= 4000.0,
        -0.2661239 * u ** 3 - 0.2343589 * u ** 2 + 0.8776956 * u + 0.179910,
        -3.0258469 * u ** 3 + 2.1070379 * u ** 2 + 0.2226347 * u + 0.240390,
    )
    y_low  = -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683
    y_mid  = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
    y_high =  3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483
    y = np.where(T <= 2222.0, y_low, np.where(T <= 4000.0, y_mid, y_high))
    return x, y


def _xy_to_linear_srgb(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(x, y) chromaticity → linear sRGB (D65), normalised so max channel = 1.

    The max-channel normalisation keeps only the *hue/saturation* of the
    chromaticity — luminance is applied separately by the caller — and
    doubles as the hue-preserving gamut clip (negative out-of-gamut
    channels are clipped to 0 before normalising).
    """
    Y = np.ones_like(x)
    X = x / y
    Z = (1.0 - x - y) / y
    r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, None)
    peak = np.max(rgb, axis=-1, keepdims=True)
    return rgb / np.maximum(peak, 1e-12)


def _srgb_gamma_encode(c: np.ndarray) -> np.ndarray:
    """Linear-light → sRGB-encoded, the standard piecewise transfer."""
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308,
                    12.92 * c,
                    1.055 * np.power(c, 1.0 / 2.4) - 0.055).astype(np.float32)


def _eye_t_grid(n: int = 96) -> np.ndarray:
    """Log-spaced color-temperature grid spanning the locus fit's validity."""
    return np.geomspace(_EYE_T_MIN, _EYE_T_MAX, int(n))


def fit_color_temperature(
    cube_calibrated: np.ndarray,
    band_names: tuple[str, ...] = Config.LR_INPUT_BAND_NAMES,
    t_grid_n: int = 96,
) -> np.ndarray:
    """Per-pixel blackbody color temperature (K) from AB-calibrated fluxes.

    For each grid temperature the model SED is the Planck f_ν sampled at
    the bands' pivot wavelengths (``Config.Color.PIVOT_WAVELENGTH_UM``),
    unit-normalised; the per-pixel best T maximises the projection
    ``f · P̂(T)`` (equivalent to least squares over a free positive
    amplitude). Noise-dominated pixels with no positive projection get
    6500 K — display-white, and their luminance is ~0 anyway.

    Runs the grid as a Python loop over ~``t_grid_n`` temperatures with
    two (H, W) running maps, so memory stays flat for large cutouts.
    """
    cube = np.asarray(cube_calibrated, dtype=np.float64)
    lam = np.array([Config.Color.PIVOT_WAVELENGTH_UM[n] for n in band_names])
    ts = _eye_t_grid(t_grid_n)
    best_score = np.full(cube.shape[:-1], -np.inf)
    best_t = np.full(cube.shape[:-1], 6500.0)
    for T in ts:
        p = _planck_fnu(lam, T)
        p = p / np.linalg.norm(p)
        score = cube @ p
        sel = score > best_score
        best_score = np.where(sel, score, best_score)
        best_t = np.where(sel, T, best_t)
    return np.where(best_score > 0.0, best_t, 6500.0)


def eye_rgb(
    cube: np.ndarray,
    band_names: tuple[str, ...] = Config.LR_INPUT_BAND_NAMES,
    *,
    stretch: str = "asinh",
    asinh_scale_e: float = 1000.0,
    white_e: float | None = None,
    t_grid_n: int = 96,
) -> np.ndarray:
    """Physical "eye" rendering of a 4-band cube → sRGB in ``[0, 1]``.

    Per pixel:

      1. AB-calibrate the bands (:func:`_ab_flux_norm` — instrument out).
      2. Fit a blackbody color temperature to the calibrated SED
         (:func:`fit_color_temperature`).
      3. Chromaticity = the CIE Planckian locus at that T, rendered in
         sRGB/D65 — the hue the dark-adapted eye assigns to a star of
         that temperature (Sun ≈ near-white, M star orange, B star blue).
      4. Luminance = the broadband calibrated intensity through an
         ABSOLUTE transfer (no per-image normalisation):

           * ``stretch="asinh"``  → ``asinh(I/knee) / asinh(white/knee)``
           * ``stretch="linear"`` → ``I / white``

         with ``knee`` ↔ ``asinh_scale_e`` electrons and ``white`` ↔
         ``white_e`` electrons, both VIS-equivalent over the stack
         (``white_e`` defaults to ``30 × asinh_scale_e`` — a typical
         bright-source level, so ordinary flux renders bright rather
         than near-black). Pixels at ``white_e`` render at full
         brightness; brighter pixels clip hue-preservingly.

    Same (SED, surface brightness) → same RGB, in every image: colors
    are directly comparable across LR/SR/HR panels, scenes, and runs.
    """
    if stretch not in ("linear", "asinh"):
        raise ValueError(f"stretch must be 'linear' or 'asinh'; got {stretch!r}")
    if cube.ndim != 3 or cube.shape[-1] != len(band_names):
        raise ValueError(
            f"cube must be (H, W, {len(band_names)}); got shape {cube.shape}"
        )
    calibrated = np.stack(
        [cube[..., k].astype(np.float64) * _ab_flux_norm(name)
         for k, name in enumerate(band_names)], axis=-1)

    # Chromaticity from the fitted color temperature.
    t_map = fit_color_temperature(calibrated, band_names, t_grid_n=t_grid_n)
    x, y = _planckian_xy(t_map)
    hue = _xy_to_linear_srgb(x, y)                      # (H, W, 3), max=1

    # Absolute luminance transfer in VIS-equivalent electrons. The white
    # point ``white_e`` (electrons → full brightness) defaults to 30× the
    # asinh knee: with the knee at the faint/noise level, ~30× sits at a
    # typical bright-source surface brightness, so ordinary galaxy flux
    # renders at a natural brightness rather than crushed near black.
    # (The old 1000× default put the white point so high that everything
    # but the brightest cores looked dim.) Brighter cores still clip
    # hue-preservingly. Override ``white_e`` to re-anchor the absolute
    # scale; it stays image-independent so colours remain comparable.
    knee_cal  = float(asinh_scale_e) * _ab_flux_norm("VIS")
    white_e   = float(white_e) if white_e is not None else 30.0 * float(asinh_scale_e)
    white_cal = white_e * _ab_flux_norm("VIS")
    intensity = np.maximum(calibrated.mean(axis=-1), 0.0)
    if stretch == "asinh":
        lum = np.arcsinh(intensity / knee_cal) / np.arcsinh(white_cal / knee_cal)
    else:
        lum = intensity / white_cal
    lum = np.clip(lum, 0.0, 1.0)[..., None]

    return _srgb_gamma_encode(hue * lum)


def planck_color_strip(
    n: int = 256,
    t_min: float = 2500.0,
    t_max: float = 20000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Legend strip for the eye mode: ``(strip, temps)``.

    ``strip`` is a ``(1, n, 3)`` sRGB image of the Planckian-locus hues
    at full luminance, rendered through the SAME chromaticity pipeline
    the eye mode uses; ``temps`` are the n corresponding temperatures
    (K, log-spaced) for axis ticks. Read it as "a pixel of this hue has
    an SED like a blackbody of this temperature".
    """
    temps = np.geomspace(float(t_min), float(t_max), int(n))
    x, y = _planckian_xy(temps)
    hue = _xy_to_linear_srgb(x, y)
    return _srgb_gamma_encode(hue)[None, :, :], temps


def lupton_rgb(
    cube: np.ndarray,
    band_names: tuple[str, ...] = Config.LR_INPUT_BAND_NAMES,
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
