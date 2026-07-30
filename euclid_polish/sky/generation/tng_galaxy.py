"""Turn an IllustrisTNG TNG50-1 SKIRT-atlas mock into an injectable,
electron-calibrated galaxy stamp on the simulation's HR clean-sky grid.

The atlas (downloaded by ``scripts/fasrc_download_tng_skirt_atlas.py``) renders
each subhalo as dusty Euclid frames ``TNG<id>_O<k>_Euclid_<band>.fits`` for
band ∈ {VIS, Y, J, H} and orientation k ∈ {1..5}. Each frame is a 1600×1600
**surface-brightness** image (``BUNIT = MJy/sr``) on a *physical* grid
(``CDELT = 100 pc/pixel``) — there is no instrument PSF, noise, or angular WCS;
these are idealized, dust-attenuated intrinsic images.

To place a galaxy into a synthetic field we run three steps (in this order):

1. **Rebin** by an integer factor with a *block-mean*, which keeps the
   surface brightness (MJy/sr is intensive). The output pixel maps to a fixed
   angular scale (the 0.05″ HR grid), so the rebin factor acts as a distance
   knob: a coarser rebin makes the galaxy both smaller and fainter, as a more
   distant galaxy would appear. In redshift mode
   (:func:`tng_stamp_at_redshift`) the factor is computed from
   D_A(z) — see :mod:`euclid_polish.sky.generation.redshift_model`.
2. **Rotate** for orientation augmentation on top of the atlas's 5 physical
   viewpoints. When the rebin factor is ≥ ``ARBITRARY_ROTATION_MIN_REBIN`` (4)
   an *arbitrary* 0–360° cubic-spline rotation is applied at native resolution
   *before* the rebin — the ≥4× downsample averages out the interpolation blur
   (validated by ``scripts/check_tng_rotation_downsample.py``), multiplying the
   effective galaxy set by continuous orientation. Below the threshold an exact
   ``np.rot90`` quarter-turn is used after the rebin (lossless, artefact-free).
3. **Convert to electrons** over the Euclid stack via
   :func:`~euclid_polish.photometry.mjy_per_sr_to_electrons`, using the
   assigned HR pixel scale to turn MJy/sr into electrons-per-pixel. The result
   is a clean, pre-PSF source ready to drop onto the HR sky; the existing
   forward model supplies the Euclid PSF, noise, and LR rebin.

The returned stamp is ``(H, W, 4)`` in ``Config.LR_INPUT_BAND_NAMES`` order
(VIS, Y_E, J_E, H_E), float32 electrons.
"""

from __future__ import annotations

import os

import numpy as np

from euclid_polish.config import BandConfig, Config
from euclid_polish.photometry import (
    ab_mag_to_electrons,
    mjy_per_sr_to_electrons,
    mjy_per_sr_to_electrons_factor,
)
from euclid_polish.skirt.image import (
    block_mean,
    centered_rotation_crop_slices,
    load_skirt_frame,
    measure_halflight_radius_px,
    radius_int_grid,
    rebin_for_target_size,
    rotate_arbitrary,
    rotate_quarter,
    stochastic_round_factor,
)
from euclid_polish.skirt.image import (
    composite_stamp as _composite_stamp,
)
from euclid_polish.sky.generation.redshift_model import (
    TNG_NATIVE_PC_PER_PIXEL,
    band_drift_factors,
    compactness_factor,
    physical_pc_to_arcsec,
    rebin_factor_for_redshift,
)

# FITS band token (in the filename) → simulation BandConfig. The atlas frames
# are named with the short Euclid band labels; the model's bands carry the
# ``_E`` suffix on the NISP filters.
TNG_FITS_BANDS: tuple[str, ...] = ("VIS", "Y", "J", "H")
_FITS_BAND_TO_CONFIG: dict[str, str] = {
    "VIS": "VIS", "Y": "Y_E", "J": "J_E", "H": "H_E",
}

# Backward-compatible import for callers that historically obtained this
# generic operation from ``tng_galaxy``.
composite_stamp = _composite_stamp


def tng_fits_path(galaxy_dir: str, subhalo_id: int | str,
                  orientation: int, fits_band: str) -> str:
    """Path of one atlas frame, e.g. ``…/167396/TNG167396_O4_Euclid_VIS.fits``."""
    name = f"TNG{subhalo_id}_O{orientation}_Euclid_{fits_band}.fits"
    return os.path.join(galaxy_dir, name)


def load_tng_frame(path: str) -> np.ndarray:
    """Read a TNG-SKIRT FITS as a native-endian float32 MJy/sr array.

    This compatibility wrapper keeps the established TNG API while the generic
    FITS mechanics live in :mod:`euclid_polish.skirt.image`.
    """
    return load_skirt_frame(path)


#: TNG-policy threshold at/above which an arbitrary-angle rotation may precede
#: the block mean. The generic rotation primitive itself lives in
#: :mod:`euclid_polish.skirt.image`; this threshold is specific to our validation
#: on the TNG atlas and Euclid HR grid.
ARBITRARY_ROTATION_MIN_REBIN = 4
TNG_ROTATION_CROP_ENCLOSED_FRACTION = 0.99
TNG_ROTATION_CROP_PADDING = 1.05
TNG_MAX_REBIN_FACTOR = 64


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
    rot_angle: float | None = None,
    min_rebin_for_angle: int = ARBITRARY_ROTATION_MIN_REBIN,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
    fits_bands: tuple[str, ...] = TNG_FITS_BANDS,
) -> tuple[np.ndarray, dict]:
    """Build a 4-band, electron-calibrated TNG stamp ready to inject.

    Loads the four Euclid frames for ``(subhalo_id, orientation)`` and converts
    MJy/sr → electrons at ``pixel_scale_arcsec`` (the HR clean-sky grid, 0.05″ by
    default). Rotation depends on the downsample:

    * ``rot_angle`` given **and** ``rebin_factor >= min_rebin_for_angle`` →
      arbitrary-angle cubic-spline rotation at native resolution *before* the
      block-mean (the ≥K× averaging washes out the interpolation blur — see
      :func:`rotate_arbitrary`). This is the orientation-augmentation path.
    * otherwise → block-mean first, then an exact ``rot_k`` quarter-turn (the
      lossless fallback for low-downsample stamps / when no angle is requested).

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
    # Arbitrary-angle rotation only when requested AND enough downsample follows
    # to wash out the spline blur; else the exact quarter-turn.
    use_angle = (rot_angle is not None
                 and rebin_factor >= int(min_rebin_for_angle))
    # Assemble in canonical model-band order so channel c is LR band c.
    config_to_fits = {v: k for k, v in _FITS_BAND_TO_CONFIG.items()}
    channels: list[np.ndarray] = []
    flux_e: dict[str, float] = {}
    rot_crop = None     # centred crop slices, computed once (from VIS) for all bands
    for cfg_name in Config.LR_INPUT_BAND_NAMES:
        fband = config_to_fits[cfg_name]
        if fband not in fits_bands:
            raise ValueError(f"band {fband} not in requested fits_bands={fits_bands}")
        band = Config.get_band(cfg_name)
        path = tng_fits_path(galaxy_dir, subhalo_id, orientation, fband)
        sb = load_tng_frame(path)                       # MJy/sr, 1600²
        if use_angle:
            if rot_crop is None:                        # size from the first (VIS) frame
                rot_crop = centered_rotation_crop_slices(
                    sb,
                    rebin_factor,
                    enclosed_fraction=TNG_ROTATION_CROP_ENCLOSED_FRACTION,
                    padding=TNG_ROTATION_CROP_PADDING,
                )
            # Crop to the galaxy core, then spline-rotate (cheap) at native res.
            sb = rotate_arbitrary(sb[rot_crop], rot_angle)
        sb = block_mean(sb, rebin_factor)               # still MJy/sr
        if not use_angle:
            sb = rotate_quarter(sb, rot_k)              # exact 90° fallback
        e = surface_brightness_to_electrons(sb, band, pixel_scale_arcsec)
        channels.append(e)
        flux_e[cfg_name] = float(e.sum())
    stamp = np.stack(channels, axis=-1).astype(np.float32)
    meta = {
        "subhalo_id": str(subhalo_id),
        "orientation": int(orientation),
        "rebin_factor": int(rebin_factor),
        "rot_k": int(rot_k) % 4,
        "rot_angle": float(rot_angle) % 360.0 if use_angle else None,
        "arbitrary_rotation": bool(use_angle),
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
DOWNSAMPLE_CHOICES: tuple[int, ...] = (1, 2, 3, 4)


def list_tng_galaxies(tng_dir: str) -> list[tuple[str, str]]:
    """List ``(galaxy_dir, subhalo_id)`` for downloaded galaxies ready to inject.

    A galaxy qualifies if its folder holds a ``.done`` marker AND the VIS O1
    frame exists, so :func:`prepare_tng_galaxy` won't choke on a partial dir."""
    if not os.path.isdir(tng_dir):
        return []
    out: list[tuple[str, str]] = []
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
_HALFLIGHT_PX_CACHE: dict[tuple[str, str, int], float] = {}


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


def truncate_below_sb(
    stamp: np.ndarray,
    pixel_scale_arcsec: float,
    sb_cut_mag_arcsec2: float,
) -> np.ndarray:
    """Zero pixels fainter than ``sb_cut_mag_arcsec2`` (AB, per band) and
    crop to the surviving footprint, kept square and centred so the galaxy
    stays at the stamp centre.

    The SKIRT box is 160 kpc with nonzero light everywhere, with outskirts
    far below Euclid's detection limit. Modifies per-band channels in place
    and returns the (possibly smaller) cropped view.
    """
    if not (sb_cut_mag_arcsec2 > 0.0):
        return stamp
    pix2 = pixel_scale_arcsec ** 2
    keep = np.zeros(stamp.shape[:2], dtype=bool)
    for k, name in enumerate(Config.LR_INPUT_BAND_NAMES):
        # SB cut (mag/arcsec²) → e⁻/arcsec² → e⁻/pixel via the pixel area.
        thr = ab_mag_to_electrons(sb_cut_mag_arcsec2,
                                  Config.get_band(name)) * pix2
        ch = stamp[..., k]
        ch[ch < thr] = 0.0
        keep |= ch > 0.0
    if not keep.any():
        return stamp
    # Crop at the radius enclosing 99.5% of the surviving (band-summed)
    # flux, so a faint outlying satellite blob doesn't hold the whole
    # 160 kpc box open.
    total = stamp.sum(axis=2, dtype=np.float64)
    H, W = total.shape
    rint = radius_int_grid((H, W))
    prof = np.bincount(rint.ravel(), weights=total.ravel())
    cum = np.cumsum(prof)
    r = int(np.searchsorted(cum, 0.995 * cum[-1])) + 4
    cy, cx = int(round((H - 1) / 2.0)), int(round((W - 1) / 2.0))
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    return stamp[y0:y1, x0:x1]


def tng_stamp_at_redshift(
    galaxy_dir: str,
    subhalo_id: int | str,
    orientation: int,
    z: float,
    rng: np.random.Generator | None = None,
    *,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
    rot_k: int | None = None,
    f_max: int = TNG_MAX_REBIN_FACTOR,
    sb_cut_mag_arcsec2: float = Config.TNG_SB_TRUNCATE_MAG_ARCSEC2,
    mass_scale: float = 1.0,
    target_re_arcsec: float | None = None,
    target_flux_e_per_band: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, dict]:
    """Build one TNG stamp **as it would appear at redshift ``z``**.

    ``mass_scale`` < 1 re-uses the stamp as a *smaller galaxy of similar
    morphology*: flux × s (L ∝ M) and an extra size squeeze s^-α along the
    observed mass-size relation R ∝ M^α — so its surface brightness drops
    as s^(1-2α), the observed trend. Distinct from the fixed-mass
    compactness correction, which conserves flux.

    A single ``z`` drives all three observables (see
    :mod:`euclid_polish.sky.generation.redshift_model`):

    * the block-mean factor comes from the angular size of the 100 pc native
      pixel at D_A(z) (stochastically rounded with ``rng`` so the mean
      apparent size is unbiased), times the flux-conserving compactness
      correction C(z) for the z = 0 atlas morphologies
      (:func:`~euclid_polish.sky.generation.redshift_model.compactness_factor`);
    * Tolman (1+z)⁻³ surface-brightness dimming;
    * a randomized spectral drift across the four bands, anchored on the
      stamp's own 4-point SED (``rng=None`` → deterministic drift only);
    * outskirts fainter than ``sb_cut_mag_arcsec2`` are truncated and the
      stamp cropped (:func:`truncate_below_sb`).

    Raises on unreadable frames, like :func:`prepare_tng_galaxy`.
    """
    compact = compactness_factor(z)
    if not (0.0 < mass_scale <= 1.0):
        raise ValueError(f"mass_scale must be in (0, 1], got {mass_scale}")
    squeeze = compact * mass_scale ** -Config.TNG_MASS_SIZE_ALPHA
    f_geo = rebin_factor_for_redshift(z, pixel_scale_arcsec=pixel_scale_arcsec)
    re_px = native_halflight_px(galaxy_dir, subhalo_id, orientation)
    if (
        target_re_arcsec is not None and target_re_arcsec > 0.0
        and np.isfinite(re_px) and re_px > 0.0
    ):
        f_cont = min(
            float(f_max),
            max(1.0, float(re_px) * pixel_scale_arcsec / target_re_arcsec),
        )
    else:
        f_cont = min(float(f_max), f_geo * squeeze)
    rebin = stochastic_round_factor(f_cont, rng)
    if rot_k is None:
        rot_k = int(rng.integers(0, 4)) if rng is not None else 0
    # Arbitrary orientation for the dataset-multiplying augmentation (applied by
    # prepare_tng_galaxy only when rebin ≥ ARBITRARY_ROTATION_MIN_REBIN); rng=None
    # → no angle → exact quarter-turn, preserving the deterministic path.
    rot_angle = float(rng.uniform(0.0, 360.0)) if rng is not None else None

    stamp, meta = prepare_tng_galaxy(
        galaxy_dir, subhalo_id, orientation,
        rebin_factor=rebin, rot_k=rot_k, rot_angle=rot_angle,
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
    # Flux: the block-mean keeps surface brightness while the pixel count
    # shrinks, so (rebin/f_geo)² pins the total to the *continuous
    # geometric* prediction (covering the compactness squeeze and the
    # integer rounding); mass_scale then dims the rescaled galaxy (L ∝ M).
    stamp *= (np.asarray(factors, dtype=np.float32)[None, None, :]
              * np.float32((rebin / f_geo) ** 2 * mass_scale))
    if target_flux_e_per_band is not None:
        for k, target in enumerate(target_flux_e_per_band):
            current = float(stamp[..., k].sum(dtype=np.float64))
            if current > 0.0 and target >= 0.0:
                stamp[..., k] *= np.float32(float(target) / current)
    stamp = truncate_below_sb(stamp, pixel_scale_arcsec, sb_cut_mag_arcsec2)
    # The COSMOS magnitude is the total-flux contract. Restore flux removed by
    # the surface-brightness crop so size and flux are independently controlled.
    if target_flux_e_per_band is not None:
        for k, target in enumerate(target_flux_e_per_band):
            current = float(stamp[..., k].sum(dtype=np.float64))
            if current > 0.0 and target >= 0.0:
                stamp[..., k] *= np.float32(float(target) / current)
    meta["flux_e_per_band"] = {
        b: float(stamp[..., k].sum())
        for k, b in enumerate(Config.LR_INPUT_BAND_NAMES)
    }
    meta["shape"] = tuple(stamp.shape)
    meta["sb_cut_mag_arcsec2"] = float(sb_cut_mag_arcsec2)
    meta["z"] = float(z)
    meta["rebin_factor_continuous"] = float(f_cont)
    meta["compactness"] = float(compact)
    meta["mass_scale"] = float(mass_scale)
    meta["redshift_band_factors"] = [float(f) for f in factors]
    meta.update(dmeta)
    if np.isfinite(re_px) and re_px > 0.0:
        meta["native_halflight_px"] = float(re_px)
        meta["apparent_re_arcsec"] = float(
            pixel_scale_arcsec * re_px / rebin
            if target_re_arcsec is not None
            else physical_pc_to_arcsec(
                re_px * TNG_NATIVE_PC_PER_PIXEL, z) / squeeze
        )
    if target_re_arcsec is not None:
        meta["target_re_arcsec"] = float(target_re_arcsec)
    if target_flux_e_per_band is not None:
        meta["target_flux_e_per_band"] = [
            float(value) for value in target_flux_e_per_band
        ]
    return stamp, meta


#: Per-(dir, galaxy, orientation) native photometry cache: the VIS frame's
#: mean-SB radial profile (MJy/sr per native-pixel radius) + each band's
#: total MJy/sr sum. One 4-frame read each, then dict lookups — powers the
#: ANALYTIC lens-showability predictors (no rendering).
_NATIVE_PHOTOM_CACHE: dict[tuple[str, str, int],
                           tuple[np.ndarray, np.ndarray]] = {}


def native_photometry(galaxy_dir: str, subhalo_id: int | str,
                      orientation: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached ``(vis_sb_profile, band_sums)`` for one atlas frame set."""
    key = (str(galaxy_dir), str(subhalo_id), int(orientation))
    cached = _NATIVE_PHOTOM_CACHE.get(key)
    if cached is not None:
        return cached
    sums = []
    profile = None
    for fband in TNG_FITS_BANDS:
        frame = load_tng_frame(
            tng_fits_path(galaxy_dir, subhalo_id, orientation, fband))
        sums.append(float(frame.sum()))
        if fband == "VIS":
            rint = radius_int_grid(frame.shape)
            flux = np.bincount(rint.ravel(), weights=frame.ravel())
            cnt = np.bincount(rint.ravel()).astype(np.float64)
            profile = (flux / np.maximum(cnt, 1.0)).astype(np.float64)
    out = (profile, np.asarray(sums, dtype=np.float64))
    _NATIVE_PHOTOM_CACHE[key] = out
    return out


def predict_visible_radius_arcsec(
    galaxy_dir: str, subhalo_id: int | str, orientation: int, z: float,
    *,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
    sb_cut_mag_arcsec2: float = Config.TNG_SB_TRUNCATE_MAG_ARCSEC2,
) -> float:
    """Analytic prediction of a stamp's visible radius at redshift ``z`` —
    where the dimmed/drifted/compactness-boosted mean VIS surface brightness
    crosses the truncation threshold. Approximate (mean profile, VIS only,
    deterministic drift) but cheap: scalars on the cached native profile,
    no stamp load. Used to reject unshowable lens systems before rendering;
    a post-render check remains the backstop.
    """
    profile, sums = native_photometry(galaxy_dir, subhalo_id, orientation)
    band = Config.get_band("VIS")
    factors, _ = band_drift_factors(sums, z, None)        # dimming + drift
    compact = compactness_factor(z)
    fac = mjy_per_sr_to_electrons_factor(band, pixel_scale_arcsec)
    sb_e = profile * fac * factors[0] * compact ** 2      # e⁻/HR-pixel
    thr = (ab_mag_to_electrons(sb_cut_mag_arcsec2, band)
           * pixel_scale_arcsec ** 2)
    above = np.nonzero(sb_e >= thr)[0]
    if above.size == 0:
        return 0.0
    return physical_pc_to_arcsec(
        float(above.max()) * TNG_NATIVE_PC_PER_PIXEL, z) / compact


def predict_vis_flux_e(
    galaxy_dir: str, subhalo_id: int | str, orientation: int, z: float,
    *,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
    mass_scale: float = 1.0,
) -> float:
    """Analytic prediction of a stamp's total VIS flux at redshift ``z``
    (electrons; truncation losses ignored). The flux-conservation boost makes
    the total independent of the integer rebin: native sum × conversion ×
    (dimming · drift) / f_geo² × mass_scale.
    """
    _, sums = native_photometry(galaxy_dir, subhalo_id, orientation)
    factors, _ = band_drift_factors(sums, z, None)
    f_geo = rebin_factor_for_redshift(z, pixel_scale_arcsec=pixel_scale_arcsec)
    fac = mjy_per_sr_to_electrons_factor(
        Config.get_band("VIS"), pixel_scale_arcsec)
    return float(sums[0] * fac * factors[0] / f_geo ** 2 * mass_scale)


def sample_tng_stamp(
    galaxies: list[tuple[str, str]],
    rng: np.random.Generator,
    *,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
    downsample_choices: tuple[int, ...] = DOWNSAMPLE_CHOICES,
    target_re_arcsec: float | None = None,
    z: float | None = None,
    mass_scale: float = 1.0,
    f_max: int = TNG_MAX_REBIN_FACTOR,
    target_flux_e_per_band: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, dict] | None:
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
    * otherwise → uniform draw from ``downsample_choices`` (×1/×2/×3/×4);
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
                pixel_scale_arcsec=pixel_scale_arcsec, f_max=f_max,
                mass_scale=mass_scale,
                target_re_arcsec=target_re_arcsec,
                target_flux_e_per_band=target_flux_e_per_band)
        except Exception:
            return None

    rot_k = int(rng.integers(0, 4))
    rot_angle = float(rng.uniform(0.0, 360.0))   # used when rebin ≥ threshold

    re_px: float | None = None
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
            rebin_factor=rebin, rot_k=rot_k, rot_angle=rot_angle,
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
