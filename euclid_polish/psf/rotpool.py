"""Pre-rotated PSF kernel pools — rotation amortised to a one-time build.

The FASRC benchmark put the order-3 rotation of a 511² kernel at ~92 ms — far
too slow per training example, and generation never paid it at all
(``psf_unrotated_prob`` defaults to 1.0, so the shipped TFRecords use
UNROTATED kernels). This module precomputes the rotations once:

  * :func:`build_rotation_pool` — for every cluster kernel of every band,
    draw K RANDOM roll angles (one shared angle table across bands, so pool
    index j is the same physical (cluster, roll) in every band — one
    pointing), rotate, and stream the results to one multi-extension FITS
    per band (``euclid_psf_rotpool_<BAND>.fits`` next to the source ePSFs).
    Each HDU carries ``SRCIDX`` (source cluster), ``ROLLDEG`` (0 for the
    included unrotated copy) and the source's ``NSTARS`` (so the star-count
    scene-draw weighting survives — a cluster's rotations inherit its
    weight).
  * :func:`load_band_rotpool` / :func:`load_all_band_rotpools` — load a pool
    as a plain :class:`PSFSet`, optionally BAGGING it: ``subset_clusters``
    picks a seeded random subset of source clusters (with all their
    rotations), so each ensemble member can train against its own PSF
    sub-population — same idea as bootstrap over fields. ``crop_to``
    centre-crops kernels at read time (the crop-local forward only needs the
    truncated support; a 257² crop is 4× less RAM than the full 511²).

Pool kernels are stored CLEANED (background floor + radial taper) and at the
target pixel scale — build from :func:`load_all_band_psf_sets` output — so
the loader must NOT re-clean them.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from astropy.io import fits

from euclid_polish.config import BandConfig, Config
from euclid_polish.psf.core import PSF
from euclid_polish.psf.psf_set import PSFSet

#: Default random roll draws per cluster kernel (the unrotated original is
#: always included as well, so a pool holds ``n_clusters × (rotations + 1)``).
DEFAULT_ROTATIONS = 12


def rotpool_path(band: BandConfig, psf_dir: str = Config.EUCLID_PSF_DIR) -> str:
    """Canonical pool path: ``<psf_dir>/euclid_psf_rotpool_<BAND>.fits``."""
    return os.path.join(psf_dir, f"euclid_psf_rotpool_{band.name}.fits")


def draw_angle_table(n_clusters: int, rotations: int, *, seed: int,
                     angle_min: float = 1.0, angle_max: float = 359.0,
                     ) -> np.ndarray:
    """``(n_clusters, rotations)`` random roll angles in degrees.

    One table for ALL bands: pool slot ``(cluster, k)`` must be the same
    physical telescope roll in every band, or a scene's four channels would
    see four different pointings.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(float(angle_min), float(angle_max),
                       size=(int(n_clusters), int(rotations)))


def _rotate_one(args) -> tuple[int, float, np.ndarray]:
    """Worker: rotate one kernel; returns ``(cluster_idx, angle, data)``."""
    idx, angle, data, pixel_scale = args
    psf = PSF(data=data, pixel_scale=pixel_scale)
    out = psf.rotated(float(angle), order=3).with_unit_sum().data
    return idx, float(angle), out.astype(np.float32)


def _kernel_header(*, pixel_scale: float, src_idx: int, angle: float,
                   n_stars: int | None, name: str) -> fits.Header:
    hdr = fits.Header()
    hdr["EXTNAME"] = name
    hdr["PXSCALE"] = (float(pixel_scale), "Pixel scale (arcsec/pixel)")
    hdr["SRCIDX"] = (int(src_idx), "Source cluster kernel index")
    hdr["ROLLDEG"] = (float(angle), "Roll rotation (deg; 0 = original)")
    if n_stars is not None:
        hdr["NSTARS"] = (int(n_stars),
                         "Stars the SOURCE ePSF was built from")
    return hdr


def build_band_rotation_pool(
    band: BandConfig,
    pset: PSFSet,
    angle_table: np.ndarray,
    *,
    psf_dir: str = Config.EUCLID_PSF_DIR,
    crop_to: int | None = None,
    workers: int = 1,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> str:
    """Rotate every kernel of ``pset`` by its ``angle_table`` row and write
    the pool FITS for ``band``. Returns the written path.

    The file is streamed HDU by HDU (a 356-cluster × 13-slot × 511² pool is
    ~4.8 GB — never all in memory). Ordering is deterministic: for each
    cluster ascending, the unrotated copy first, then its rotations in table
    order — identical across bands, which is what keeps pool indices
    physically aligned.
    """
    rotations = angle_table.shape[1]
    if crop_to is not None and crop_to % 2 == 0:
        raise ValueError(f"crop_to must be odd (a centred kernel), got {crop_to}")

    def _crop(a: np.ndarray) -> np.ndarray:
        if crop_to is None or a.shape[0] <= crop_to:
            return a
        c, h = a.shape[0] // 2, crop_to // 2
        out = a[c - h: c + h + 1, c - h: c + h + 1].astype(np.float64)
        s = out.sum()
        return (out / s if s > 0 else out).astype(np.float32)

    path = rotpool_path(band, psf_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"

    primary = fits.PrimaryHDU(data=_crop(pset.mean().data.astype(np.float32)))
    primary.header["PXSCALE"] = (float(pset.pixel_scale),
                                 "Pixel scale (arcsec/pixel)")
    primary.header["ROTPOOL"] = (True, "Pre-rotated kernel pool")
    primary.header["NSRCCLU"] = (int(pset.n), "Source cluster kernels")
    primary.header["NROT"] = (int(rotations), "Random rotations per cluster")
    primary.header["CLEANED"] = (True, "Members background-cleaned pre-build")
    primary.header["COMMENT"] = (
        "HDU0=mean of source set; HDU1..N = (cluster, roll) kernels, "
        "per cluster: unrotated first, then its random rolls.")
    fits.HDUList([primary]).writeto(tmp, overwrite=True)

    total = pset.n * (rotations + 1)
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for i, psf in enumerate(pset.psfs):
            n_stars = (pset.n_stars[i]
                       if pset.n_stars is not None and i < len(pset.n_stars)
                       else None)
            unit = psf.with_unit_sum()
            jobs = [(i, angle, unit.data, pset.pixel_scale)
                    for angle in angle_table[min(i, len(angle_table) - 1)]]
            # One cluster's slots: the unrotated original + its rolls, held
            # in memory only for this batch, appended in ONE append-mode
            # open (fits.append per HDU would re-walk every existing header
            # each call — O(N²) over a 4.6k-HDU pool).
            batch = [(0.0, unit.data.astype(np.float32))]
            batch += [(angle, data)
                      for _idx, angle, data in pool.map(_rotate_one, jobs)]
            with fits.open(tmp, mode="append") as hdul:
                for k, (angle, data) in enumerate(batch):
                    hdul.append(fits.ImageHDU(
                        data=_crop(data),
                        header=_kernel_header(
                            pixel_scale=pset.pixel_scale, src_idx=i,
                            angle=angle, n_stars=n_stars,
                            name=f"P{i:03d}R{k:03d}")))
            done += len(batch)
            if on_progress is not None:
                on_progress(done, total, f"{band.name} cluster {i}")
    os.replace(tmp, path)
    return path


def build_rotation_pool(
    psf_sets: dict[str, PSFSet],
    *,
    psf_dir: str = Config.EUCLID_PSF_DIR,
    rotations: int = DEFAULT_ROTATIONS,
    seed: int = 0,
    angle_min: float = 1.0,
    angle_max: float = 359.0,
    crop_to: int | None = None,
    workers: int = 1,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, str]:
    """Build the per-band pools from already-loaded (cleaned, resampled)
    ``psf_sets``. One shared angle table (see :func:`draw_angle_table`) keeps
    pool indices aligned across bands. Returns ``{band: path}``."""
    n_clusters = max(p.n for p in psf_sets.values())
    table = draw_angle_table(n_clusters, rotations, seed=seed,
                             angle_min=angle_min, angle_max=angle_max)
    out = {}
    for band in Config.BANDS:
        if band.name not in psf_sets:
            continue
        out[band.name] = build_band_rotation_pool(
            band, psf_sets[band.name], table, psf_dir=psf_dir,
            crop_to=crop_to, workers=workers, on_progress=on_progress)
    return out


# --------------------------------------------------------------------------- #
# Loading (+ per-training bagging)
# --------------------------------------------------------------------------- #
def load_band_rotpool(
    band: BandConfig,
    *,
    psf_dir: str = Config.EUCLID_PSF_DIR,
    subset_clusters: int | None = None,
    subset_seed: int | None = None,
    crop_to: int | None = None,
    target_pixel_scale: float = Config.DEFAULT_PIXEL_SCALE,
) -> PSFSet:
    """Load ``band``'s pre-rotated pool as a :class:`PSFSet`.

    ``subset_clusters`` bags the pool: a seeded random subset of SOURCE
    clusters is chosen (``subset_seed``) and only their kernels (unrotated +
    all rolls) are read — different seeds → different PSF sub-populations,
    the PSF analogue of bootstrap-over-fields. The choice is over cluster
    indices, so the same (seed, count) picks the same physical sky regions
    in every band. ``crop_to`` centre-crops kernels while reading (memmap →
    only the crop is materialised). NO background cleaning is applied — pool
    kernels were cleaned at build time.
    """
    path = rotpool_path(band, psf_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no rotation pool for {band.name} at {path} — build it with "
            "scripts/pregenerate_psf_rotations.py")
    psfs: list[PSF] = []
    counts: list[int] = []
    with fits.open(path, memmap=True) as hdul:
        n_src = int(hdul[0].header["NSRCCLU"])
        chosen: set[int] | None = None
        if subset_clusters is not None and 0 < int(subset_clusters) < n_src:
            rng = np.random.default_rng(subset_seed)
            chosen = set(rng.choice(n_src, size=int(subset_clusters),
                                    replace=False).tolist())
        for h in hdul[1:]:
            if h.data is None or "SRCIDX" not in h.header:
                continue
            if chosen is not None and int(h.header["SRCIDX"]) not in chosen:
                continue
            data = np.asarray(h.data, np.float32)
            if crop_to is not None and data.shape[0] > crop_to:
                c, half = data.shape[0] // 2, crop_to // 2
                data = data[c - half: c + half + 1, c - half: c + half + 1]
                s = float(data.sum())
                if s > 0:
                    data = (data / s).astype(np.float32)
            psfs.append(PSF(data=data.copy(),
                            pixel_scale=float(h.header["PXSCALE"])))
            counts.append(int(h.header.get("NSTARS", 0)))
    pset = PSFSet.from_psfs(psfs, n_stars=counts if any(counts) else None)
    if abs(pset.pixel_scale - target_pixel_scale) > 1e-6:
        pset = pset.resampled_to(target_pixel_scale)
    return pset


def load_all_band_rotpools(
    *,
    psf_dir: str = Config.EUCLID_PSF_DIR,
    subset_clusters: int | None = None,
    subset_seed: int | None = None,
    crop_to: int | None = None,
    target_pixel_scale: float = Config.DEFAULT_PIXEL_SCALE,
) -> dict[str, PSFSet] | None:
    """All four band pools with a SHARED bagging subset, or ``None`` when any
    band's pool file is missing (caller falls back to the live-rotation /
    unrotated sets)."""
    if not all(os.path.isfile(rotpool_path(b, psf_dir)) for b in Config.BANDS):
        return None
    return {b.name: load_band_rotpool(
                b, psf_dir=psf_dir, subset_clusters=subset_clusters,
                subset_seed=subset_seed, crop_to=crop_to,
                target_pixel_scale=target_pixel_scale)
            for b in Config.BANDS}
