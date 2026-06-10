"""Turn an IllustrisTNG TNG50-1 SKIRT-atlas mock into an injectable,
electron-calibrated galaxy stamp on the simulation's HR clean-sky grid.

The atlas (downloaded by ``scripts/fasrc_download_tng_skirt_atlas.py``) renders
each subhalo as dusty Euclid frames ``TNG<id>_O<k>_Euclid_<band>.fits`` for
band ∈ {VIS, Y, J, H} and orientation k ∈ {1..5}. Each frame is a 1600×1600
**surface-brightness** image (``BUNIT = MJy/sr``) on a *physical* grid
(``CDELT = 100 pc/pixel``) — there is no instrument PSF, noise, or angular WCS;
these are idealized, dust-attenuated intrinsic images.

To place a galaxy into a synthetic field we run three steps (in this order):

1. **Rebin** by an integer factor (2 or 4) with a *block-mean*, which keeps the
   surface brightness (MJy/sr is intensive) and matches the ``_800``/``_400``
   prototypes already on disk. Because the output pixel is then mapped to a
   fixed angular scale (the 0.05″ HR grid), the rebin factor behaves as an
   effective distance knob: a coarser rebin makes the galaxy both smaller (in
   pixels) and fainter (total flux ∝ pixel solid angle), as a more distant
   galaxy would appear.
2. **Rotate** by a quarter turn (0/90/180/270°) — exact ``np.rot90``, a free
   orientation augmentation on top of the atlas's 5 physical viewpoints.
3. **Convert to electrons** over the Euclid stack via
   :func:`~euclid_polish.euclid.photometry.mjy_per_sr_to_electrons`, using the
   assigned HR pixel scale to turn MJy/sr into electrons-per-pixel. The result
   is a clean, pre-PSF source ready to drop onto the HR sky; the existing
   forward model supplies the Euclid PSF, noise, and LR rebin.

The returned stamp is ``(H, W, 4)`` in ``Config.LR_INPUT_BAND_NAMES`` order
(VIS, Y_E, J_E, H_E), float32 electrons.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits

from euclid_polish.config import BandConfig, Config
from euclid_polish.euclid.photometry import (
    mjy_per_sr_to_electrons,
    mjy_per_sr_to_electrons_factor,
)
from euclid_polish.sky.redshift_model import (
    TNG_NATIVE_PC_PER_PIXEL,
    band_drift_factors,
    physical_pc_to_arcsec,
    rebin_factor_for_redshift,
)

# FITS band token (in the filename) → simulation BandConfig. The atlas frames
# are named with the short Euclid band labels; the model's bands carry the
# ``_E`` suffix for the NISP filters.
TNG_FITS_BANDS: Tuple[str, ...] = ("VIS", "Y", "J", "H")
_FITS_BAND_TO_CONFIG: Dict[str, str] = {
    "VIS": "VIS", "Y": "Y_E", "J": "J_E", "H": "H_E",
}


def tng_fits_path(galaxy_dir: str, subhalo_id: int | str,
                  orientation: int, fits_band: str) -> str:
    """Path of one atlas frame, e.g. ``…/167396/TNG167396_O4_Euclid_VIS.fits``."""
    name = f"TNG{subhalo_id}_O{orientation}_Euclid_{fits_band}.fits"
    return os.path.join(galaxy_dir, name)


def load_tng_frame(path: str) -> np.ndarray:
    """Read a SKIRT mock FITS as a native-endian float32 MJy/sr array.

    SKIRT writes big-endian ``>f4``; we byte-swap to the host order so
    downstream numpy/TF ops don't choke. Non-finite pixels (rare edge NaNs)
    are zeroed — they are sky, i.e. no flux.
    """
    with fits.open(path) as hdul:
        data = hdul[0].data
    if data is None:
        raise ValueError(f"empty primary HDU: {path}")
    arr = np.asarray(data, dtype=np.float32)   # also normalises endianness
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def block_mean(arr: np.ndarray, factor: int, *,
               trim_remainder: bool = True) -> np.ndarray:
    """Surface-brightness-preserving rebin: average each ``factor × factor``
    block. ``factor == 1`` is a no-op copy.

    Unlike :meth:`MultiBandSkyImage.rebin_array` (a flux-conserving *sum*),
    this keeps the intensive MJy/sr brightness unchanged, so a rebinned frame
    is still a valid surface-brightness image (matching the atlas's own
    ``2x2 block-mean … (SB-preserving, MJy/sr)`` downsamples).
    """
    if factor < 1:
        raise ValueError(f"factor must be ≥ 1, got {factor}")
    a = np.asarray(arr, dtype=np.float32)
    if factor == 1:
        return a.copy()
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {a.shape}")
    H, W = a.shape
    if H % factor != 0 or W % factor != 0:
        if not trim_remainder:
            raise ValueError(
                f"spatial dims {(H, W)} not divisible by factor={factor}")
        H, W = (H // factor) * factor, (W // factor) * factor
        a = a[:H, :W]
    Hn, Wn = H // factor, W // factor
    return a.reshape(Hn, factor, Wn, factor).mean(axis=(1, 3))


def rotate_quarter(arr: np.ndarray, k: int) -> np.ndarray:
    """Rotate by ``k`` quarter-turns (k·90° CCW) via exact ``np.rot90``.

    ``k`` is taken mod 4, so 0/1/2/3 → 0/90/180/270°; no interpolation, so
    flux is bit-exactly conserved.
    """
    return np.rot90(np.asarray(arr), k=int(k) % 4)


# ---------------------------------------------------------------------------
# Half-light radius (for sizing a stamp to a target apparent angular size)
# ---------------------------------------------------------------------------

#: Cache of the integer radius-from-centre grid, keyed by frame shape — the
#: atlas frames are all the same size, so this is computed once per shape.
_RADIUS_INT_GRID: Dict[Tuple[int, int], np.ndarray] = {}


def _radius_int_grid(shape: Tuple[int, int]) -> np.ndarray:
    grid = _RADIUS_INT_GRID.get(shape)
    if grid is None:
        H, W = shape
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        yy = np.arange(H, dtype=np.float64)[:, None] - cy
        xx = np.arange(W, dtype=np.float64)[None, :] - cx
        grid = np.sqrt(yy * yy + xx * xx).astype(np.int64)
        _RADIUS_INT_GRID[shape] = grid
    return grid


def measure_halflight_radius_px(frame: np.ndarray, *, frac: float = 0.5) -> float:
    """Half-light radius (pixels) of ``frame`` via a curve of growth about the
    frame centre — SKIRT centres each subhalo, so the geometric centre is the
    light centre. Only positive flux counts; an empty/negative frame → NaN.

    Returns the (sub-pixel, linearly interpolated) radius enclosing ``frac`` of
    the total positive flux.
    """
    a = np.asarray(frame, dtype=np.float64)
    a = np.where(np.isfinite(a) & (a > 0.0), a, 0.0)
    total = float(a.sum())
    if total <= 0.0:
        return float("nan")
    rint = _radius_int_grid(a.shape)
    prof = np.bincount(rint.ravel(), weights=a.ravel())
    cum = np.cumsum(prof)
    target = frac * total
    idx = int(np.searchsorted(cum, target))
    if idx <= 0:
        return 0.5
    c0, c1 = cum[idx - 1], cum[idx]
    sub = (target - c0) / (c1 - c0) if c1 > c0 else 0.0
    return float(idx - 1 + sub)


def rebin_for_target_size(
    re_native_px: float,
    target_re_arcsec: float,
    hr_pixel_scale: float,
    *,
    rng: Optional[np.random.Generator] = None,
    f_max: int = 64,
) -> int:
    """Integer block-mean factor that places a stamp of native half-light radius
    ``re_native_px`` at apparent half-light radius ``target_re_arcsec`` on the
    ``hr_pixel_scale`` grid.

    Geometry: after a block-mean of ``F`` the half-light radius is
    ``re_native_px / F`` *output* pixels, each spanning ``hr_pixel_scale``
    arcsec, so ``apparent = hr_pixel_scale · re_native_px / F`` →
    ``F = hr_pixel_scale · re_native_px / target``. Clipped to ``[1, f_max]``
    (can't upsample below native; cap keeps the stamp from collapsing to a few
    pixels). With an ``rng`` the non-integer factor is **stochastically rounded**
    so the *mean* apparent size is unbiased even where integer granularity is
    coarse (the rare big galaxies, F≈2–4).
    """
    if not (re_native_px > 0.0) or not (target_re_arcsec > 0.0):
        return 1
    f = hr_pixel_scale * re_native_px / target_re_arcsec
    f = min(max(f, 1.0), float(f_max))
    lo = int(np.floor(f))
    rem = f - lo
    if rng is not None and rem > 0.0:
        f_int = lo + (1 if rng.random() < rem else 0)
    else:
        f_int = int(round(f))
    return max(1, f_int)


def surface_brightness_to_electrons(arr_mjy_sr: np.ndarray, band: BandConfig,
                                    pixel_scale_arcsec: float) -> np.ndarray:
    """MJy/sr → electrons-per-pixel over ``band``'s stack at the HR pixel
    scale. Thin wrapper over the photometry primitive, kept here so the
    three-step recipe reads top-to-bottom."""
    return mjy_per_sr_to_electrons(arr_mjy_sr, band, pixel_scale_arcsec)


def prepare_tng_galaxy(
    galaxy_dir: str,
    subhalo_id: int | str,
    orientation: int,
    *,
    rebin_factor: int = 2,
    rot_k: int = 0,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
    fits_bands: Tuple[str, ...] = TNG_FITS_BANDS,
) -> Tuple[np.ndarray, dict]:
    """Build a 4-band, electron-calibrated TNG stamp ready to inject.

    Loads the four Euclid frames for ``(subhalo_id, orientation)``, rebins each
    by ``rebin_factor`` (block-mean, SB-preserving), rotates by ``rot_k``
    quarter-turns, and converts MJy/sr → electrons at ``pixel_scale_arcsec``
    (the HR clean-sky grid, 0.05″ by default).

    Returns
    -------
    stamp : ndarray
        ``(H, W, 4)`` float32 electrons, channels in
        ``Config.LR_INPUT_BAND_NAMES`` order (VIS, Y_E, J_E, H_E).
    meta : dict
        Provenance + per-band integrated electron counts.
    """
    if rebin_factor < 1:
        raise ValueError(f"rebin_factor must be ≥ 1, got {rebin_factor}")
    # Assemble in canonical model-band order so channel c is LR band c.
    config_to_fits = {v: k for k, v in _FITS_BAND_TO_CONFIG.items()}
    channels: List[np.ndarray] = []
    flux_e: Dict[str, float] = {}
    for cfg_name in Config.LR_INPUT_BAND_NAMES:
        fband = config_to_fits[cfg_name]
        if fband not in fits_bands:
            raise ValueError(f"band {fband} not in requested fits_bands={fits_bands}")
        band = Config.get_band(cfg_name)
        path = tng_fits_path(galaxy_dir, subhalo_id, orientation, fband)
        sb = load_tng_frame(path)                       # MJy/sr, 1600²
        sb = block_mean(sb, rebin_factor)               # still MJy/sr
        sb = rotate_quarter(sb, rot_k)
        e = surface_brightness_to_electrons(sb, band, pixel_scale_arcsec)
        channels.append(e)
        flux_e[cfg_name] = float(e.sum())
    stamp = np.stack(channels, axis=-1).astype(np.float32)
    meta = {
        "subhalo_id": str(subhalo_id),
        "orientation": int(orientation),
        "rebin_factor": int(rebin_factor),
        "rot_k": int(rot_k) % 4,
        "pixel_scale_arcsec": float(pixel_scale_arcsec),
        "shape": tuple(stamp.shape),
        "flux_e_per_band": flux_e,
        "bands": tuple(Config.LR_INPUT_BAND_NAMES),
    }
    return stamp, meta


# ---------------------------------------------------------------------------
# Enumeration + random sampling (for injection into synthetic scenes)
# ---------------------------------------------------------------------------

N_ORIENTATIONS = 5                              # SKIRT viewpoints O1..O5
DOWNSAMPLE_CHOICES: Tuple[int, ...] = (1, 2, 3, 4)


def list_tng_galaxies(tng_dir: str) -> List[Tuple[str, str]]:
    """List ``(galaxy_dir, subhalo_id)`` for downloaded galaxies ready to inject.

    A galaxy qualifies if its folder holds a ``.done`` marker AND the VIS O1
    frame exists, so :func:`prepare_tng_galaxy` won't choke on a partial dir."""
    if not os.path.isdir(tng_dir):
        return []
    out: List[Tuple[str, str]] = []
    for gid in os.listdir(tng_dir):
        gdir = os.path.join(tng_dir, gid)
        if (os.path.isfile(os.path.join(gdir, Config.Tng.DONE_MARKER))
                and os.path.isfile(tng_fits_path(gdir, gid, 1, "VIS"))):
            out.append((gdir, gid))
    try:
        return sorted(out, key=lambda t: int(t[1]))
    except ValueError:
        return sorted(out)


#: Per-(dir, galaxy, orientation) native VIS half-light radius cache (pixels).
#: Filled lazily the first time a galaxy is sized; one entry per orientation, so
#: repeat draws skip the FITS read + curve-of-growth entirely. Keyed on the
#: directory too, so distinct galaxy sets (e.g. test fixtures reusing ids) never
#: alias.
_HALFLIGHT_PX_CACHE: Dict[Tuple[str, str, int], float] = {}


def composite_stamp(canvas_4ch: np.ndarray, stamp: np.ndarray,
                    x0: float, y0: float) -> None:
    """Add a ``(Hs,Ws,C)`` stamp centred at ``(x0,y0)`` onto the canvas, clipped
    to the canvas bounds. The stamp may be larger than the field — only the
    overlapping region is added. Shared by field-galaxy injection and TNG lens
    light."""
    H, W = canvas_4ch.shape[:2]
    Hs, Ws = stamp.shape[:2]
    i0 = int(round(y0)) - Hs // 2
    j0 = int(round(x0)) - Ws // 2
    ci_lo, ci_hi = max(0, i0), min(H, i0 + Hs)
    cj_lo, cj_hi = max(0, j0), min(W, j0 + Ws)
    if ci_lo >= ci_hi or cj_lo >= cj_hi:
        return
    si_lo, sj_lo = ci_lo - i0, cj_lo - j0
    canvas_4ch[ci_lo:ci_hi, cj_lo:cj_hi, :] += \
        stamp[si_lo:si_lo + (ci_hi - ci_lo), sj_lo:sj_lo + (cj_hi - cj_lo), :]


def native_halflight_px(
    galaxy_dir: str, subhalo_id: int | str, orientation: int,
    *, fits_band: str = "VIS",
) -> float:
    """Cached native half-light radius (px) of one galaxy/orientation's VIS
    frame. NaN if the frame can't be read or carries no flux."""
    key = (str(galaxy_dir), str(subhalo_id), int(orientation))
    cached = _HALFLIGHT_PX_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        frame = load_tng_frame(
            tng_fits_path(galaxy_dir, subhalo_id, orientation, fits_band))
        re_px = measure_halflight_radius_px(frame)
    except Exception:
        re_px = float("nan")
    _HALFLIGHT_PX_CACHE[key] = re_px
    return re_px


def tng_stamp_at_redshift(
    galaxy_dir: str,
    subhalo_id: int | str,
    orientation: int,
    z: float,
    rng: Optional[np.random.Generator] = None,
    *,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
    rot_k: Optional[int] = None,
    f_max: int = 64,
) -> Tuple[np.ndarray, dict]:
    """Build one TNG stamp **as it would appear at redshift ``z``**.

    A single ``z`` drives all three observables (see
    :mod:`euclid_polish.sky.redshift_model`):

    * the block-mean factor comes from the angular size of the 100 pc native
      pixel at D_A(z) (stochastically rounded with ``rng`` so the mean
      apparent size is unbiased);
    * Tolman (1+z)⁻³ surface-brightness dimming;
    * a randomized spectral drift across the four bands, anchored on the
      stamp's own 4-point SED (``rng=None`` → deterministic drift only).

    Raises on unreadable frames, like :func:`prepare_tng_galaxy`.
    """
    f_cont = min(float(f_max), rebin_factor_for_redshift(
        z, pixel_scale_arcsec=pixel_scale_arcsec))
    lo = int(np.floor(f_cont))
    rem = f_cont - lo
    if rng is not None and rem > 0.0:
        rebin = lo + (1 if rng.random() < rem else 0)
    else:
        rebin = int(round(f_cont))
    rebin = max(1, rebin)
    if rot_k is None:
        rot_k = int(rng.integers(0, 4)) if rng is not None else 0

    stamp, meta = prepare_tng_galaxy(
        galaxy_dir, subhalo_id, orientation,
        rebin_factor=rebin, rot_k=rot_k,
        pixel_scale_arcsec=pixel_scale_arcsec,
    )
    # The stamp's own rest-frame 4-point SED in relative f_ν: undo each
    # band's linear MJy/sr → electrons factor so the zeropoints/integration
    # don't masquerade as colour.
    sed_fnu = [
        meta["flux_e_per_band"][b]
        / mjy_per_sr_to_electrons_factor(Config.get_band(b), pixel_scale_arcsec)
        for b in Config.LR_INPUT_BAND_NAMES
    ]
    factors, dmeta = band_drift_factors(sed_fnu, z, rng)
    stamp *= np.asarray(factors, dtype=np.float32)[None, None, :]
    meta["flux_e_per_band"] = {
        b: float(meta["flux_e_per_band"][b] * factors[k])
        for k, b in enumerate(Config.LR_INPUT_BAND_NAMES)
    }
    meta["z"] = float(z)
    meta["rebin_factor_continuous"] = float(f_cont)
    meta["redshift_band_factors"] = [float(f) for f in factors]
    meta.update(dmeta)
    re_px = native_halflight_px(galaxy_dir, subhalo_id, orientation)
    if np.isfinite(re_px) and re_px > 0.0:
        meta["native_halflight_px"] = float(re_px)
        meta["apparent_re_arcsec"] = physical_pc_to_arcsec(
            re_px * TNG_NATIVE_PC_PER_PIXEL, z)
    return stamp, meta


def sample_tng_stamp(
    galaxies: List[Tuple[str, str]],
    rng: np.random.Generator,
    *,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
    downsample_choices: Tuple[int, ...] = DOWNSAMPLE_CHOICES,
    target_re_arcsec: Optional[float] = None,
    z: Optional[float] = None,
    f_max: int = 64,
) -> Optional[Tuple[np.ndarray, dict]]:
    """Pick a random galaxy / orientation / downsample / quarter-rotation and
    return its injectable ``(H,W,4)`` electron stamp + meta (None if it can't
    load).

    Sizing (first match wins):

    * ``z`` given → full redshift treatment via :func:`tng_stamp_at_redshift`:
      the downsample follows from D_A(z), and (1+z)⁻³ dimming + the randomized
      spectral drift are applied. ``meta`` carries ``z`` and the per-band
      factors.
    * ``target_re_arcsec`` given → the galaxy is downsampled so its **apparent
      half-light radius** matches that target (via :func:`rebin_for_target_size`
      on the measured native size), giving a realistic, mostly-small angular
      size with occasional big resolved galaxies. ``meta`` then also carries
      ``target_re_arcsec`` / ``native_halflight_px`` / ``apparent_re_arcsec``.
    * otherwise → legacy uniform draw from ``downsample_choices`` (×1/×2/×3/×4);
      coarser = smaller and fainter, like a more distant galaxy.
    """
    if not galaxies:
        return None
    gdir, gid = galaxies[int(rng.integers(0, len(galaxies)))]
    orientation = int(rng.integers(1, N_ORIENTATIONS + 1))      # O1..O5

    if z is not None:
        try:
            return tng_stamp_at_redshift(
                gdir, gid, orientation, z, rng,
                pixel_scale_arcsec=pixel_scale_arcsec, f_max=f_max)
        except Exception:
            return None

    rot_k = int(rng.integers(0, 4))

    re_px: Optional[float] = None
    if target_re_arcsec is not None and target_re_arcsec > 0.0:
        re_px = native_halflight_px(gdir, gid, orientation)
        if re_px is not None and np.isfinite(re_px) and re_px > 0.0:
            rebin = rebin_for_target_size(
                re_px, target_re_arcsec, pixel_scale_arcsec,
                rng=rng, f_max=f_max)
        else:                                       # unmeasurable → safe default
            rebin = int(downsample_choices[
                int(rng.integers(0, len(downsample_choices)))])
    else:
        rebin = int(downsample_choices[
            int(rng.integers(0, len(downsample_choices)))])

    try:
        stamp, meta = prepare_tng_galaxy(
            gdir, gid, orientation,
            rebin_factor=rebin, rot_k=rot_k,
            pixel_scale_arcsec=pixel_scale_arcsec,
        )
    except Exception:
        return None
    if target_re_arcsec is not None and re_px is not None and np.isfinite(re_px):
        meta["target_re_arcsec"] = float(target_re_arcsec)
        meta["native_halflight_px"] = float(re_px)
        meta["apparent_re_arcsec"] = float(
            pixel_scale_arcsec * re_px / rebin)
    return stamp, meta
