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
from euclid_polish.euclid.photometry import mjy_per_sr_to_electrons

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


def sample_tng_stamp(
    galaxies: List[Tuple[str, str]],
    rng: np.random.Generator,
    *,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
    downsample_choices: Tuple[int, ...] = DOWNSAMPLE_CHOICES,
) -> Optional[Tuple[np.ndarray, dict]]:
    """Pick a random galaxy / orientation / downsample / quarter-rotation and
    return its injectable ``(H,W,4)`` electron stamp + meta (None if it can't
    load).

    The downsample factor is drawn from ``downsample_choices`` (default
    ×1/×2/×3/×4); coarser = smaller and fainter, like a more distant galaxy.
    """
    if not galaxies:
        return None
    gdir, gid = galaxies[int(rng.integers(0, len(galaxies)))]
    orientation = int(rng.integers(1, N_ORIENTATIONS + 1))      # O1..O5
    rebin = int(downsample_choices[int(rng.integers(0, len(downsample_choices)))])
    rot_k = int(rng.integers(0, 4))
    try:
        return prepare_tng_galaxy(
            gdir, gid, orientation,
            rebin_factor=rebin, rot_k=rot_k,
            pixel_scale_arcsec=pixel_scale_arcsec,
        )
    except Exception:
        return None
