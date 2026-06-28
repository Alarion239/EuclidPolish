"""``EuclidCatalog`` — the authenticated client for all Euclid archive operations.

Construction authenticates eagerly::

    cat = EuclidCatalog(login="user", password="pw")   # explicit
    cat = EuclidCatalog()                               # EUCLID_USER / EUCLID_PASSWORD

and raises :class:`EuclidAuthError` if no credentials are available or the login
fails. The instance owns the session (astroquery's ``Euclid`` is a process-global
singleton; a class-level lock serialises (re)login). Queries return
:class:`~euclid_polish.catalog.catalog_object.CatalogObject` lists; downloads
return cutout FITS / a multi-band :class:`~euclid_polish.image.Image`.
"""

from __future__ import annotations

import math
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits as _fits
from astroquery.esa.euclid import Euclid
from tqdm import tqdm

from euclid_polish.catalog.catalog_object import CatalogObject, _finite_float
from euclid_polish.catalog.downloader import (
    DownloadConfig, core_is_saturated, download_one_cutout, fetch_cutout_at,
    positions_match, resolve_mosaics, scan_cutouts,
)
from euclid_polish.catalog.photometry import adu_per_s_to_electrons_factor, uJy_to_ab_mag
from euclid_polish.catalog.validator import FitsValidator
from euclid_polish.config import Config
from euclid_polish.image import FitsWCS, Image, Role
from euclid_polish.provenance.defaults import mint_id
from euclid_polish.provenance.records import Stamp


class EuclidAuthError(RuntimeError):
    """Raised when Euclid authentication cannot be established."""



class EuclidCatalog:
    """Authenticated access to the Euclid science archive."""

    #: astroquery's ``Euclid`` is a process-global singleton — one login lock.
    _login_lock = threading.Lock()

    def __init__(self, login: Optional[str] = None, password: Optional[str] = None,
                 *, _skip_login: bool = False) -> None:
        self._user: Optional[str] = None
        self._password: Optional[str] = None
        if _skip_login:
            return
        user, pw = self._resolve_credentials(login, password)
        self._do_login(user, pw)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_credentials(login: Optional[str],
                             password: Optional[str]) -> "tuple[str, str]":
        if login and password:
            return login, password
        env_user = os.environ.get("EUCLID_USER")
        env_pw = os.environ.get("EUCLID_PASSWORD")
        if env_user and env_pw:
            return env_user, env_pw
        raise EuclidAuthError(
            "No Euclid credentials: set EUCLID_USER and EUCLID_PASSWORD, "
            "or pass login=/password=.")

    def _do_login(self, user: str, password: str) -> None:
        with self._login_lock:
            try:
                Euclid.login(user=user, password=password)
            except Exception as e:
                raise EuclidAuthError(f"Euclid login failed: {e}") from e
        self._user, self._password = user, password

    @property
    def user(self) -> Optional[str]:
        """The authenticated username, or ``None`` (unauthenticated client)."""
        return self._user

    @property
    def is_authenticated(self) -> bool:
        """True if this client was constructed with credentials."""
        return self._user is not None

    def relogin(self) -> bool:
        """Refresh the session after a mid-batch TAP expiry. ``False`` on failure."""
        if not self._user or not self._password:
            return False
        try:
            with self._login_lock:
                Euclid.login(user=self._user, password=self._password)
            return True
        except Exception:
            return False

    @classmethod
    def _unauthenticated(cls) -> "EuclidCatalog":
        """A client that skipped the network login — tests / offline only."""
        return cls(_skip_login=True)

    # ------------------------------------------------------------------
    # Queries → CatalogObject lists
    # ------------------------------------------------------------------

    def query_bright_stars(
        self, num_stars: int, *,
        ra: Optional[float] = None, dec: Optional[float] = None,
        radius: Optional[float] = None,
        magnitude_limit: Optional[float] = None,
        magnitude_min: Optional[float] = None,
        snr_min: Optional[float] = None,
        require_unmasked: bool = True,
    ) -> List[CatalogObject]:
        """Top-``num_stars`` mask-free point sources by VIS PSF flux (descending).

        Magnitudes are AB (``Config.AB_ZP_UJY``); the raw PSF flux + error are
        kept on each object. ``require_unmasked`` adds ``det_quality_flag = 0``
        (the MER clean-point-source cut) and ``flag_vis = 0`` (excludes sources
        with any SExtractor flag bits set, including saturation). ``snr_min``
        keeps ``flux ≥ snr·err``. A cone (``ra``/``dec``/``radius``) is optional
        but must be given together. Returns objects in flux order; persistence/dedup
        is the caller's job.
        """
        if num_stars <= 0:
            raise ValueError("num_stars must be positive")
        if (magnitude_min is not None and magnitude_limit is not None
                and magnitude_min >= magnitude_limit):
            raise ValueError("magnitude_min must be < magnitude_limit")
        cone = [v is not None for v in (ra, dec, radius)]
        if any(cone) and not all(cone):
            raise ValueError("ra, dec, and radius must be provided together")

        where = ["flux_vis_psf IS NOT NULL", "flux_vis_psf > 0",
                 "fluxerr_vis_psf IS NOT NULL", "fluxerr_vis_psf > 0"]
        if magnitude_limit is not None:
            where.append(f"flux_vis_psf > {10 ** ((Config.AB_ZP_UJY - magnitude_limit) / 2.5)}")
        if magnitude_min is not None:
            where.append(f"flux_vis_psf < {10 ** ((Config.AB_ZP_UJY - magnitude_min) / 2.5)}")
        if snr_min is not None and snr_min > 0:
            where.append(f"flux_vis_psf > {float(snr_min)} * fluxerr_vis_psf")
        if require_unmasked:
            where.append("det_quality_flag = 0")
            where.append("flag_vis = 0")
        if all(cone):
            where.append(
                f"CONTAINS(POINT('ICRS', right_ascension, declination), "
                f"CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1")

        query = (f"SELECT TOP {num_stars} right_ascension, declination, "
                 f"flux_vis_psf, fluxerr_vis_psf FROM catalogue.mer_catalogue "
                 f"WHERE {' AND '.join(where)} ORDER BY flux_vis_psf DESC")
        job = Euclid.launch_job_async(query)
        results = job.get_results() if job is not None else None

        objs: List[CatalogObject] = []
        for row in results or []:
            flux = _finite_float(row["flux_vis_psf"])
            if flux is None or flux <= 0:
                continue
            ferr = _finite_float(row["fluxerr_vis_psf"])
            objs.append(CatalogObject(
                ra=float(row["right_ascension"]), dec=float(row["declination"]),
                magnitude=uJy_to_ab_mag(flux), flux_psf_uJy=flux,
                fluxerr_psf_uJy=ferr if ferr is not None else float("nan"),
                kind="star"))
        return objs

    def query_galaxies(self, ra: float, dec: float, radius_deg: float, *,
                       limit: int = Config.GalaxySelection.MAX_RESULTS,
                       diam_lo_arcsec: float = Config.GalaxySelection.DIAM_LO_ARCSEC,
                       diam_hi_arcsec: float = Config.GalaxySelection.DIAM_HI_ARCSEC,
                       mag_floor: float = Config.GalaxySelection.MAG_FLOOR,
                       ) -> List[CatalogObject]:
        """Clean, resolved, bigger-end galaxies in a cone (``kind='galaxy'``).

        Server-side cuts: extended (``point_like_flag = 0``), not spurious, clean
        (``det_quality_flag = 0``), segmentation area within the
        ``diam_lo_arcsec``–``diam_hi_arcsec`` diameter window, capped at ``limit``
        rows. Here we drop sources fainter than ``mag_floor``. All four default to
        :class:`Config.GalaxySelection`.
        """
        if radius_deg <= 0:
            raise ValueError(f"radius_deg must be positive, got {radius_deg}")
        area_lo = math.pi * ((diam_lo_arcsec / 2.0) / Config.VIS_PIXEL_SCALE_ARCSEC) ** 2
        area_hi = math.pi * ((diam_hi_arcsec / 2.0) / Config.VIS_PIXEL_SCALE_ARCSEC) ** 2
        query = (
            f"SELECT TOP {limit} object_id, right_ascension, declination, "
            "segmentation_area, flux_vis_psf FROM catalogue.mer_catalogue "
            f"WHERE CONTAINS(POINT('ICRS', right_ascension, declination), "
            f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1 "
            "AND point_like_flag = 0 AND spurious_flag = 0 AND det_quality_flag = 0 "
            f"AND segmentation_area BETWEEN {area_lo:.1f} AND {area_hi:.1f} "
            "AND flux_vis_psf IS NOT NULL AND flux_vis_psf > 0")
        job = Euclid.launch_job(query)
        results = job.get_results() if job is not None else None

        objs: List[CatalogObject] = []
        for row in results or []:
            r = _finite_float(row["right_ascension"])
            d = _finite_float(row["declination"])
            flux = _finite_float(row["flux_vis_psf"])
            if r is None or d is None or flux is None or flux <= 0:
                continue
            mag = uJy_to_ab_mag(flux)
            if mag >= mag_floor:
                continue
            objs.append(CatalogObject(ra=r, dec=d, magnitude=mag,
                                      flux_psf_uJy=flux, kind="galaxy"))
        return objs

    # ------------------------------------------------------------------
    # Downloads
    # ------------------------------------------------------------------

    def fetch(
        self, ra: float, dec: float, size: int,
        *, bands: Sequence[str] = Config.LR_INPUT_BAND_NAMES,
        store=None,
    ) -> Image:
        """Download a 4-band electron-unit :class:`~euclid_polish.image.Image` at ``(ra, dec)``.

        Fetches each band, converts ADU s⁻¹ → electrons via the per-band
        ``MAGZERO`` header keyword, stacks to ``(H, W, len(bands))`` and tags
        the result ``role='real'`` with a freshly minted provenance stamp.
        The VIS band WCS is stored on the returned ``Image.fits_wcs`` so callers
        can build a scaled SR header via
        :func:`~euclid_polish.training.inference.scaled_wcs_header`.
        Pass the authenticated :class:`EuclidCatalog` instance from the calling
        context rather than constructing a new one here.

        Parameters
        ----------
        ra, dec : float
            ICRS coordinates in degrees.
        size : int
            Cutout side in VIS pixels (0.10″/pix reference grid).
        bands : sequence of str
            Band names (default ``Config.LR_INPUT_BAND_NAMES``).
        store : ProvStore, optional
            Provenance store; defaults to ``default_store()`` (guarded).
        """
        planes: List[np.ndarray] = []
        vis_wcs: Optional[FitsWCS] = None
        for band_name in bands:
            band = Config.get_band(band_name)
            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
                tmp_path = tf.name
            try:
                ok, err = fetch_cutout_at(
                    ra=ra, dec=dec, band_name=band_name,
                    output_file=tmp_path, cutout_size_vis_pixels=size,
                )
                if not ok:
                    raise RuntimeError(
                        f"EuclidCatalog.fetch: band {band_name} failed: {err}")
                with _fits.open(tmp_path) as hdul:
                    arr = hdul[0].data.astype(np.float32)
                    magzero = float(
                        hdul[0].header.get("MAGZERO", band.sim_zeropoint_e))
                    if band_name == "VIS":
                        vis_wcs = FitsWCS.from_header(hdul[0].header)
                factor = np.float32(adu_per_s_to_electrons_factor(magzero, band))
                planes.append((arr * factor).astype(np.float32))
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        data = np.stack(planes, axis=-1)
        return Image(
            data=data, pixel_scale_arcsec=Config.VIS_PIXEL_SCALE_ARCSEC,
            band_names=tuple(bands), is_clean=False, role=Role.REAL,
            stamp=Stamp(id=mint_id(store), schema_version=3),
            fits_wcs=vis_wcs)

    # ------------------------------------------------------------------
    # Batch cutout download (a whole CatalogObject list, one band)
    # ------------------------------------------------------------------

    def download_cutouts(
        self, objects: List[CatalogObject], output_dir: str,
        config: DownloadConfig, *,
        ids: Optional[List[int]] = None,
        show_progress: bool = True,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        retry_failed: bool = False,
        validator: Optional[FitsValidator] = None,
    ) -> Dict[str, Any]:
        """Download ``config.band`` cutouts for ``objects`` into ``output_dir``.

        Cutouts land in ``<output_dir>/cutouts/<band>/star_<id>_<size>.fits`` and
        each object's per-``(band, size)`` flags are updated in place; the list is
        persisted to ``<output_dir>/<CATALOG_FILE>`` via
        :meth:`CatalogObject.write`. Mutates ``objects`` in place.

        ``ids`` restricts the download to those object ids. ``retry_failed``
        clears this band's ``download_failed`` flags first so objects wrongly
        flagged by a transient TAP/session error get re-attempted.
        ``progress_cb(current, total, label)`` drives the WebUI progress bar.
        Returns a summary dict.
        """
        validator = validator or FitsValidator()
        band = config.band
        cutout_size = config.cutout_size
        catalog_path = os.path.join(output_dir, Config.CATALOG_FILE)
        cutout_dir = Config.cutout_dir_for_band(
            band, root=os.path.join(output_dir, Config.CUTOUTS_SUBDIR))
        os.makedirs(cutout_dir, exist_ok=True)

        # Optionally un-stick objects wrongly flagged ``download_failed`` by a
        # transient resolution error, so they re-enter the pending set below.
        if retry_failed:
            n_cleared = 0
            for o in objects:
                if o.is_download_failed(cutout_size, band=band):
                    o.clear_download_failed(cutout_size, band=band)
                    n_cleared += 1
            if n_cleared:
                print(f"  retry-failed: cleared {n_cleared} download_failed "
                      f"flags for {band} (size {cutout_size})")

        # Scan existing FITS files (size-aware) and reconcile corruption.
        existing_fits, corrupted_disk = scan_cutouts(cutout_dir, validator)
        if corrupted_disk:
            by_id = {o.id: o for o in objects}
            for star_id, size, filepath in corrupted_disk:
                o = by_id.get(star_id)
                if o is not None and size is not None:
                    o.set_corrupted(size, band=band)
                try:
                    os.remove(filepath)
                except Exception:
                    pass
            CatalogObject.write(objects, catalog_path)

        pixel_scale_arcmin = config.pixel_scale_arcsec / 60.0
        cutout_radius_arcmin = (cutout_size / 2.0) * pixel_scale_arcmin

        # Walk every existing file (all sizes) and reconcile validity.
        matched: set = set()  # objects with a valid cutout at cutout_size
        by_id = {o.id: o for o in objects}
        for star_id, files in existing_fits.items():
            o = by_id.get(star_id)
            if o is None:
                continue
            for fra, fdec, fsize, _ in files:
                if not positions_match(fra, fdec, o.ra, o.dec,
                                       config.position_tolerance):
                    continue
                if not o.is_valid(fsize, band=band):
                    o.set_valid(fsize, band=band)
                if abs(fsize - cutout_size) <= config.size_tolerance:
                    matched.add(star_id)

        # Pending = not corrupted/failed at the requested (band, size).
        pending = [o for o in objects
                   if not o.is_corrupted(cutout_size, band=band)
                   and not o.is_download_failed(cutout_size, band=band)]
        if ids is not None:
            pending = [o for o in pending if o.id in ids]
        needing = [o for o in pending
                   if o.id not in matched or not o.is_valid(cutout_size, band=band)]

        CatalogObject.write(objects, catalog_path)

        if not needing:
            valid = sum(1 for o in objects if o.is_valid(cutout_size, band=band))
            corrupted = sum(1 for o in objects
                            if o.is_corrupted(cutout_size, band=band))
            failed = sum(1 for o in objects
                         if o.is_download_failed(cutout_size, band=band))
            if failed:
                print(f"  {band}: nothing pending, but {failed} stars are "
                      f"flagged download_failed — rerun with --retry-failed.")
            return {'downloaded': 0, 'valid': valid, 'corrupted': corrupted,
                    'failed': failed, 'cutout_size': cutout_size, 'band': band}

        # Resolve every object → mosaic tile in ONE ADQL query.
        print(f"  Resolving mosaic tiles for {len(needing)} stars...")
        mosaic_lookup = resolve_mosaics(needing, config, relogin=self.relogin)
        unmatched_ids = {o.id for o in needing if o.id not in mosaic_lookup}
        if unmatched_ids:
            print(f"  ⚠️  {len(unmatched_ids)} stars not covered by any "
                  f"{band} tile — marking failed")
        with_mosaics = [o for o in needing if o.id in mosaic_lookup]

        rejected_ids: List[int] = []

        def _download_and_validate(o: CatalogObject) -> Tuple[int, bool]:
            mosaic = mosaic_lookup[o.id]
            output_file = os.path.join(
                cutout_dir, f"star_{o.id:04d}_{cutout_size}.fits")
            for _ in range(2):     # one retry on validation failure
                if not download_one_cutout(o.ra, o.dec, config,
                                           cutout_radius_arcmin, output_file,
                                           mosaic):
                    continue
                is_valid, _ = validator.validate_cutout(
                    output_file, o.ra, o.dec, config.position_tolerance)
                if is_valid:
                    # Reject oversaturated / clipped cores (MER zeros the peak
                    # of bright stars) — same check the PSF extractor applies.
                    data = validator.get_data(output_file)
                    if core_is_saturated(data, config.saturation_core_size):
                        try:
                            os.unlink(output_file)
                        except OSError:
                            pass
                        return o.id, False
                    return o.id, True
                try:
                    os.unlink(output_file)
                except OSError:
                    pass
            return o.id, False

        if with_mosaics:
            with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
                futures = [pool.submit(_download_and_validate, o)
                           for o in with_mosaics]
                progress = tqdm(
                    as_completed(futures), total=len(futures),
                    desc=f"Downloading (size {cutout_size}, {config.max_workers}x)",
                    disable=not show_progress)
                n_total = len(futures)
                for done_i, fut in enumerate(progress, start=1):
                    star_id, ok = fut.result()
                    if not ok:
                        rejected_ids.append(star_id)
                    if progress_cb is not None:
                        progress_cb(done_i, n_total, f"star_{star_id:04d}")

        # Update flags — per-(band, size).
        new_valid_ids = [o.id for o in with_mosaics if o.id not in rejected_ids]
        for o in objects:
            if o.id in new_valid_ids:
                o.set_valid(cutout_size, band=band)
            if o.id in rejected_ids:
                o.set_corrupted(cutout_size, band=band)
            if o.id in unmatched_ids:
                o.set_download_failed(cutout_size, band=band)

        CatalogObject.write(objects, catalog_path)

        final_valid = sum(1 for o in objects if o.is_valid(cutout_size, band=band))
        final_corrupted = sum(1 for o in objects
                              if o.is_corrupted(cutout_size, band=band))
        final_failed = sum(1 for o in objects
                           if o.is_download_failed(cutout_size, band=band))
        return {
            'downloaded': len(new_valid_ids),
            'valid': final_valid,
            'corrupted': final_corrupted,
            'failed': final_failed,
            'rejected_ids': rejected_ids,
            'unmatched_ids': sorted(unmatched_ids),
            'cutout_size': cutout_size,
            'band': band,
        }
