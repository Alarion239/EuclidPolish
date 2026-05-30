"""
Euclid cutout downloader module.

This module provides an object-oriented interface for downloading
Euclid VIS cutouts from the Euclid archive.
"""

import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
from astroquery.esa.euclid import Euclid
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm

from euclid_polish.euclid.catalog import StarCatalog


def core_is_saturated(data: np.ndarray, core_size: int) -> bool:
    """True if the central ``core_size × core_size`` region is clipped.

    Mirrors :meth:`PSFExtractor.extract_psf_star_from_cutout`: the Euclid
    MER pipeline zeros pixels above well capacity, so an oversaturated /
    clipped star has a hole (non-finite or ``<= 0`` values) at its centre.
    Cutouts are centred on the star, so we inspect the central core.
    ``core_size <= 0`` disables the check (returns False).
    """
    if core_size is None or core_size < 1 or data is None:
        return False
    ny, nx = data.shape[:2]
    cy, cx = ny // 2, nx // 2
    sh = int(core_size) // 2
    core = data[max(0, cy - sh): cy + sh + 1, max(0, cx - sh): cx + sh + 1]
    if core.size == 0:
        return True
    return bool((~np.isfinite(core)).any() or (core <= 0).any())
from euclid_polish.euclid.validator import FitsValidator, angular_separation_arcsec
from euclid_polish.config import Config


@dataclass
class DownloadConfig:
    """Configuration for cutout downloading (one band at a time).

    Driven by a :class:`euclid_polish.config.BandConfig`. The ``band``
    attribute selects:

      * which catalog flag slot to write (``valid``/``corrupted``/``failed``
        all become per-band)
      * which on-disk subdirectory to use (``cutouts/<band_name>/``)
      * which Euclid archive instrument + filter to query
      * which pixel scale to use for the cutout-radius calculation.

    Pass a :class:`BandConfig` directly via :meth:`for_band` for a fully
    populated config; the bare init exists for back-compat tests.
    """
    cutout_size: int = Config.DEFAULT_CUTOUT_SIZE
    cutout_radius: float = 0.2  # arcmin
    position_tolerance: float = Config.Matching.DOWNLOAD_POSITION_TOL_ARCSEC
    size_tolerance: int = Config.Matching.DOWNLOAD_SIZE_TOL_PIXELS
    environment: str = "PDR"
    max_workers: int = 8  # parallel cutout HTTPS fetches
    band: str = "VIS"
    instrument: str = "VIS"
    filter_name: Optional[str] = None
    pixel_scale_arcsec: float = Config.VIS_PIXEL_SCALE_ARCSEC
    # Saturation/clipping rejection: a downloaded cutout is rejected when
    # its central ``saturation_core_size × saturation_core_size`` region
    # (centred on the star) contains any non-finite or ``<= 0`` pixel —
    # the Euclid MER pipeline zeros out detector pixels above well
    # capacity, so a clipped/saturated core shows up as that hole. Same
    # check the PSF extractor applies (PSFExtractionConfig). 0 disables.
    saturation_core_size: int = 7

    @classmethod
    def for_band(
        cls,
        band_name: str,
        *,
        cutout_size_vis_pixels: Optional[int] = None,
        cutout_size: Optional[int] = None,
        **overrides,
    ) -> "DownloadConfig":
        """Build a download config from a band name.

        Cutout size can be specified two ways:

        * ``cutout_size_vis_pixels`` (preferred): the *reference* size in
          VIS pixels at 0.10″/pix. Each band's native cutout size is
          derived so every band covers the same angular field. Example:
          ``cutout_size_vis_pixels=512`` → 51.2″ on a side → VIS
          fetches 512 px, NISP fetches ~171 px.
        * ``cutout_size``: explicit native pixel count for this band
          (bypasses the conversion; useful when the user wants
          band-by-band control).

        Pulls instrument / filter / native pixel scale from the band's
        :class:`BandConfig`; remaining knobs (max_workers etc.) default
        unless overridden.
        """
        b = Config.get_band(band_name)

        if cutout_size is not None and cutout_size_vis_pixels is not None:
            raise ValueError(
                "Pass either cutout_size or cutout_size_vis_pixels, not both."
            )
        if cutout_size_vis_pixels is not None:
            arcsec_side = (
                int(cutout_size_vis_pixels)
                * Config.BAND_VIS.pixel_scale_lr_arcsec
            )
            cutout_size = b.cutout_size_for_arcsec(arcsec_side)

        defaults = dict(
            band=b.name,
            instrument=b.archive_instrument,
            filter_name=b.archive_filter or None,
            pixel_scale_arcsec=b.pixel_scale_lr_arcsec,
        )
        if cutout_size is not None:
            defaults["cutout_size"] = int(cutout_size)
        defaults.update(overrides)
        return cls(**defaults)

    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate configuration."""
        if self.cutout_size <= 0:
            return False, "Cutout size must be positive"
        if self.cutout_radius <= 0:
            return False, "Cutout radius must be positive"
        if self.position_tolerance <= 0:
            return False, "Position tolerance must be positive"
        if self.max_workers <= 0:
            return False, "max_workers must be positive"
        if self.instrument not in ("VIS", "NISP"):
            return False, f"instrument must be VIS or NISP, got {self.instrument!r}"
        if self.instrument == "NISP" and not self.filter_name:
            return False, "NISP downloads require a filter_name (e.g. 'NIR_Y')"
        if self.pixel_scale_arcsec <= 0:
            return False, "pixel_scale_arcsec must be positive"
        return True, None


class EuclidCutoutDownloader:
    """
    Download Euclid VIS cutouts for stars in a catalog.

    This class handles:
    - Querying the Euclid archive for mosaic tiles
    - Downloading cutouts around specified coordinates
    - Validating downloaded files
    - Updating the star catalog
    """

    def __init__(
        self,
        catalog: StarCatalog,
        config: Optional[DownloadConfig] = None,
        validator: Optional[FitsValidator] = None
    ):
        """
        Initialize the downloader.

        Parameters:
        -----------
        catalog : StarCatalog
            Star catalog manager.
        config : DownloadConfig, optional
            Download configuration. Uses defaults if not provided.
        validator : FitsValidator, optional
            FITS validator for checking downloaded files.
        """
        self.catalog = catalog
        self.config = config or DownloadConfig()
        self.validator = validator or FitsValidator()
        # Per-band subdirectory: ``data/euclid_stars/cutouts/<band>/``.
        self.cutout_dir = Config.cutout_dir_for_band(
            self.config.band,
            root=os.path.join(catalog.output_dir, Config.CUTOUTS_SUBDIR),
        )
        os.makedirs(self.cutout_dir, exist_ok=True)
        self.band = self.config.band

    def get_existing_cutouts(
        self,
    ) -> Tuple[
        Dict[int, List[Tuple[float, float, int, str]]],
        List[Tuple[int, Optional[int], str]],
    ]:
        """
        Scan FITS files and extract star positions from WCS headers.

        A single star can have multiple cutouts on disk at different sizes
        (e.g. ``star_0042_256.fits`` and ``star_0042_512.fits``); the scan
        returns *all* of them rather than collapsing per star.

        Returns:
        --------
        tuple
            (star_positions, corrupted_files)
            - star_positions: ``{star_id: [(ra, dec, size, filepath), ...]}``
            - corrupted_files: ``[(star_id, size_or_None, filepath), ...]``
        """
        star_positions: Dict[int, List[Tuple[float, float, int, str]]] = {}
        corrupted_files: List[Tuple[int, Optional[int], str]] = []

        fits_files = glob.glob(os.path.join(self.cutout_dir, "star_[0-9][0-9][0-9][0-9]_*.fits"))

        for filepath in fits_files:
            filename = os.path.basename(filepath)
            star_id: Optional[int] = None
            size: Optional[int] = None
            try:
                # Filename format: "star_XXXX_SIZE.fits"
                parts = filename.split('_')
                if len(parts) >= 3 and parts[0] == 'star':
                    star_id = int(parts[1])
                    size = int(parts[2].replace('.fits', ''))

                    header = self.validator.get_header(filepath)
                    if header and 'CRVAL1' in header and 'CRVAL2' in header:
                        ra = float(header['CRVAL1'])
                        dec = float(header['CRVAL2'])
                        star_positions.setdefault(star_id, []).append(
                            (ra, dec, size, filepath)
                        )
                    else:
                        corrupted_files.append((star_id, size, filepath))
            except (ValueError, IndexError, KeyError, OSError) as e:
                # File is corrupted — record what we managed to parse.
                if star_id is None:
                    parts = filename.split('_')
                    if len(parts) >= 2 and parts[0] == 'star':
                        try:
                            star_id = int(parts[1])
                        except (ValueError, IndexError):
                            star_id = None
                    if star_id is not None and len(parts) >= 3:
                        try:
                            size = int(parts[2].replace('.fits', ''))
                        except (ValueError, IndexError):
                            size = None
                if star_id is not None:
                    corrupted_files.append((star_id, size, filepath))
                print(f"Warning: Could not parse {filename}: {e}")
                continue

        return star_positions, corrupted_files

    def _resolve_mosaics(self, stars: List[dict]) -> Dict[int, Dict[str, Any]]:
        """
        Map every star to its VIS mosaic tile in a single ADQL query.

        Pulls the full VIS mosaic catalogue from ``sedm.mosaic_product`` once,
        then matches each star to its nearest tile center (tiles are ~0.5° wide,
        so the nearest center is the containing tile within tile-overlap noise).

        Returns ``{star_id: {'file_path': '<path>/<name>', 'tile_index': int}}``.
        Stars not covered by any tile are absent from the dict.
        """
        if not stars:
            return {}

        # Build the WHERE clause from instrument + optional filter.
        where_parts = [f"instrument_name = '{self.config.instrument}'"]
        if self.config.filter_name:
            # The mosaic_product table exposes the band/filter via a
            # ``filter_name`` column. We keep the ADQL parameterisable so
            # callers can supply either Euclid's archive id (``NIR_Y``) or
            # the project-level band id (``Y_E``).
            where_parts.append(f"filter_name = '{self.config.filter_name}'")
        query = (
            "SELECT file_path, file_name, tile_index, ra, dec "
            "FROM sedm.mosaic_product "
            f"WHERE {' AND '.join(where_parts)}"
        )
        try:
            job = Euclid.launch_job_async(query)
            mosaics = job.get_results()
        except Exception as e:
            print(f"  Mosaic lookup failed: {type(e).__name__}: {e}")
            return {}

        if mosaics is None or len(mosaics) == 0:
            print("  Mosaic lookup returned 0 VIS tiles")
            return {}

        mosaic_coords = SkyCoord(
            ra=np.asarray(mosaics['ra']) * u.degree,
            dec=np.asarray(mosaics['dec']) * u.degree,
            frame='icrs',
        )
        star_coords = SkyCoord(
            ra=np.asarray([s['ra'] for s in stars]) * u.degree,
            dec=np.asarray([s['dec'] for s in stars]) * u.degree,
            frame='icrs',
        )

        idx, sep, _ = star_coords.match_to_catalog_sky(mosaic_coords)

        max_sep = 0.5 * u.degree  # tile half-diagonal-ish; loose upper bound
        star_to_mosaic: Dict[int, Dict[str, Any]] = {}
        for i, star in enumerate(stars):
            if sep[i] > max_sep:
                continue
            m = mosaics[int(idx[i])]
            star_to_mosaic[star['id']] = {
                'file_path': f"{m['file_path']}/{m['file_name']}",
                'tile_index': int(m['tile_index']),
            }
        return star_to_mosaic

    def download_cutout(
        self,
        star: dict,
        cutout_radius_arcmin: float,
        output_file: str,
        mosaic: Dict[str, Any],
    ) -> bool:
        """
        Download a single cutout for a star using a pre-resolved mosaic tile.

        Parameters:
        -----------
        star : dict
            Star dictionary with 'ra', 'dec'.
        cutout_radius_arcmin : float
            Radius of cutout (arcmin).
        output_file : str
            Output file path.
        mosaic : dict
            ``{'file_path': '<path>/<name>', 'tile_index': int}`` from
            :meth:`_resolve_mosaics`.

        Returns:
        --------
        bool
            True if download succeeded and produced a non-empty file.
        """
        ra = star['ra']
        dec = star['dec']
        coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')

        try:
            Euclid.get_cutout(
                file_path=mosaic['file_path'],
                instrument=self.config.instrument,
                id=mosaic['tile_index'],
                coordinate=coord,
                radius=cutout_radius_arcmin * u.arcmin,
                output_file=output_file,
            )
        except Exception as e:
            tqdm.write(f"  get_cutout failed for ({ra:.4f}, {dec:.4f}): {type(e).__name__}: {e}")
            return False

        return os.path.exists(output_file) and os.path.getsize(output_file) > 0

    def positions_match(self, ra1: float, dec1: float, ra2: float, dec2: float) -> bool:
        """
        Check if two positions match within tolerance.

        Parameters:
        -----------
        ra1, dec1 : float
            First position (degrees).
        ra2, dec2 : float
            Second position (degrees).

        Returns:
        --------
        bool
            True if positions match within tolerance.
        """
        return angular_separation_arcsec(ra1, dec1, ra2, dec2) < self.config.position_tolerance

    def download(
        self,
        star_ids: Optional[List[int]] = None,
        show_progress: bool = True,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict:
        """
        Download cutouts for stars in the catalog.

        Parameters:
        -----------
        star_ids : list of int, optional
            Specific star IDs to download. If None, downloads all missing stars.
        show_progress : bool
            Whether to show progress bar.
        progress_cb : callable, optional
            Called as ``progress_cb(current, total, label)`` after each
            cutout finishes — used to drive the WebUI progress bar via the
            structured Reporter (the FASRC job has no terminal for tqdm).

        Returns:
        --------
        dict
            Download summary with counts.
        """
        catalog = self.catalog.load()
        stars = catalog['stars']
        cutout_size = self.config.cutout_size

        # Scan existing FITS files (size-aware)
        existing_fits, corrupted_disk_files = self.get_existing_cutouts()

        # Handle corrupted files on disk — flag corruption per-(band, size).
        if corrupted_disk_files:
            stars_by_id = {s['id']: s for s in stars}
            for star_id, size, filepath in corrupted_disk_files:
                star = stars_by_id.get(star_id)
                if star is not None and size is not None:
                    StarCatalog.set_corrupted(star, size, band=self.band)
                try:
                    os.remove(filepath)
                except Exception:
                    pass
            self.catalog.save(catalog)

        # Cutout radius for the requested size, in the instrument's native scale.
        pixel_scale_arcmin = self.config.pixel_scale_arcsec / 60.0
        cutout_radius_arcmin = (cutout_size / 2.0) * pixel_scale_arcmin

        # Walk every existing file (all sizes) and reconcile catalog validity.
        # A single star may have multiple cutouts on disk (e.g. 256 and 512);
        # we mark validity for whichever sizes match by position.
        matched_stars: set[int] = set()  # stars with a valid cutout at cutout_size
        for star_id, files in existing_fits.items():
            matching_star = next((s for s in stars if s['id'] == star_id), None)
            if matching_star is None:
                continue
            for fits_ra, fits_dec, fits_size, _ in files:
                if not self.positions_match(
                    fits_ra, fits_dec, matching_star['ra'], matching_star['dec']
                ):
                    continue
                if not StarCatalog.is_valid(matching_star, fits_size, band=self.band):
                    StarCatalog.set_valid(matching_star, fits_size, band=self.band)
                if abs(fits_size - cutout_size) <= self.config.size_tolerance:
                    matched_stars.add(star_id)

        # Pending = not corrupted/failed at the requested (band, size).
        pending_stars = [
            s for s in stars
            if not StarCatalog.is_corrupted(s, cutout_size, band=self.band)
            and not StarCatalog.is_download_failed(s, cutout_size, band=self.band)
        ]
        if star_ids is not None:
            pending_stars = [s for s in pending_stars if s['id'] in star_ids]

        # Stars needing download = pending and not already valid at (band, size).
        stars_needing_download = [
            s for s in pending_stars
            if s['id'] not in matched_stars
            or not StarCatalog.is_valid(s, cutout_size, band=self.band)
        ]

        self.catalog.save(catalog)

        if not stars_needing_download:
            valid_count = len([
                s for s in stars if StarCatalog.is_valid(s, cutout_size, band=self.band)
            ])
            corrupted_count = len([
                s for s in stars if StarCatalog.is_corrupted(s, cutout_size, band=self.band)
            ])
            return {
                'downloaded': 0,
                'valid': valid_count,
                'corrupted': corrupted_count,
                'cutout_size': cutout_size,
                'band': self.band,
            }

        # Resolve every star → VIS mosaic tile in ONE ADQL query (vs. one
        # cone_search per star in the old code).
        print(f"  Resolving mosaic tiles for {len(stars_needing_download)} stars...")
        mosaic_lookup = self._resolve_mosaics(stars_needing_download)
        unmatched_ids = {
            s['id'] for s in stars_needing_download if s['id'] not in mosaic_lookup
        }
        if unmatched_ids:
            print(f"  ⚠️  {len(unmatched_ids)} stars not covered by any VIS tile — marking failed")
        stars_with_mosaics = [
            s for s in stars_needing_download if s['id'] in mosaic_lookup
        ]

        corrupted_star_ids: List[int] = []

        def _download_and_validate(star: dict) -> Tuple[int, bool]:
            star_id = star['id']
            mosaic = mosaic_lookup[star_id]
            filename = f"star_{star_id:04d}_{cutout_size}.fits"
            output_file = os.path.join(self.cutout_dir, filename)
            # Try once, retry once on validation failure
            for _ in range(2):
                if not self.download_cutout(
                    star, cutout_radius_arcmin, output_file, mosaic
                ):
                    continue
                is_valid, _ = self.validator.validate_cutout(
                    output_file, star['ra'], star['dec'], self.config.position_tolerance
                )
                if is_valid:
                    # Reject oversaturated / clipped cores (MER zeros the
                    # peak of bright stars). Same check the PSF extractor
                    # applies; a saturated star fails on retry too, so it
                    # ends up flagged corrupted rather than feeding a bad
                    # cutout into PSF extraction / training.
                    data = self.validator.get_data(output_file)
                    if core_is_saturated(data, self.config.saturation_core_size):
                        if os.path.exists(output_file):
                            os.remove(output_file)
                        return star_id, False
                    return star_id, True
                if os.path.exists(output_file):
                    os.remove(output_file)
            return star_id, False

        if stars_with_mosaics:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
                futures = [
                    pool.submit(_download_and_validate, s) for s in stars_with_mosaics
                ]
                progress = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Downloading (size {cutout_size}, {self.config.max_workers}x)",
                    disable=not show_progress,
                )
                n_total = len(futures)
                for done_i, fut in enumerate(progress, start=1):
                    star_id, ok = fut.result()
                    if not ok:
                        corrupted_star_ids.append(star_id)
                    if progress_cb is not None:
                        progress_cb(done_i, n_total, f"star_{star_id:04d}")

        # Update catalog — per-(band, size) flags.
        new_valid_ids = [
            s['id'] for s in stars_with_mosaics if s['id'] not in corrupted_star_ids
        ]
        for star in catalog['stars']:
            if star['id'] in new_valid_ids:
                StarCatalog.set_valid(star, cutout_size, band=self.band)
            if star['id'] in corrupted_star_ids:
                StarCatalog.set_corrupted(star, cutout_size, band=self.band)
            if star['id'] in unmatched_ids:
                StarCatalog.set_download_failed(star, cutout_size, band=self.band)

        self.catalog.save(catalog)

        # Final status — (band, size)-specific.
        final_valid = len([
            s for s in catalog['stars']
            if StarCatalog.is_valid(s, cutout_size, band=self.band)
        ])
        final_corrupted = len([
            s for s in catalog['stars']
            if StarCatalog.is_corrupted(s, cutout_size, band=self.band)
        ])
        final_failed = len([
            s for s in catalog['stars']
            if StarCatalog.is_download_failed(s, cutout_size, band=self.band)
        ])

        return {
            'downloaded': len(new_valid_ids),
            'valid': final_valid,
            'corrupted': final_corrupted,
            'failed': final_failed,
            'corrupted_ids': corrupted_star_ids,
            'unmatched_ids': sorted(unmatched_ids),
            'cutout_size': cutout_size,
            'band': self.band,
        }


# ---------------------------------------------------------------------------
# Ad-hoc single-position cutout helper (used by the inference UI)
# ---------------------------------------------------------------------------

def fetch_cutout_at(
    ra: float,
    dec: float,
    band_name: str,
    output_file: str,
    cutout_size_vis_pixels: int = 512,
) -> Tuple[bool, Optional[str]]:
    """Fetch a single Euclid cutout at one (ra, dec) for one band.

    Unlike :class:`EuclidCutoutDownloader`, this bypasses the on-disk star
    catalog — it is meant for ad-hoc inference where the caller just wants
    the four-band cube at a given sky position.

    Parameters
    ----------
    ra, dec : float
        ICRS coordinates in degrees.
    band_name : str
        One of ``"VIS" / "Y_E" / "J_E" / "H_E"``.
    output_file : str
        Where to write the FITS. Parent dir must exist.
    cutout_size_vis_pixels : int
        Reference cutout size in VIS pixels at 0.10″/pix. Each band's
        native cutout count is derived so every band covers the same
        angular field (~51.2″ for the default 512 px).

    Returns
    -------
    (ok, error_message) :
        ``ok=True`` on success (file exists and is non-empty);
        ``ok=False`` with an explanation otherwise. Failures are reported
        without raising so the caller can surface them via its own
        progress / log channels.
    """
    cfg = DownloadConfig.for_band(
        band_name, cutout_size_vis_pixels=cutout_size_vis_pixels,
    )
    ok, err = cfg.validate()
    if not ok:
        return False, err

    arcsec_side = cfg.cutout_size * cfg.pixel_scale_arcsec
    cutout_radius_arcmin = (arcsec_side / 2.0) / 60.0

    where = [f"instrument_name = '{cfg.instrument}'"]
    if cfg.filter_name:
        where.append(f"filter_name = '{cfg.filter_name}'")
    query = (
        "SELECT file_path, file_name, tile_index, ra, dec "
        "FROM sedm.mosaic_product "
        f"WHERE {' AND '.join(where)}"
    )
    try:
        mosaics = Euclid.launch_job_async(query).get_results()
    except Exception as e:
        return False, f"mosaic lookup failed: {type(e).__name__}: {e}"
    if mosaics is None or len(mosaics) == 0:
        return False, "no mosaic tiles matched the band query"

    star_coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    tile_coords = SkyCoord(
        ra=np.asarray(mosaics["ra"]) * u.degree,
        dec=np.asarray(mosaics["dec"]) * u.degree,
        frame="icrs",
    )
    idx, sep, _ = star_coord.match_to_catalog_sky(tile_coords)
    if sep > 0.5 * u.degree:
        return False, (
            f"closest {band_name} tile is {sep[0].to_value(u.degree):.2f}° away "
            f"— position likely outside Euclid Q1 coverage"
        )

    m = mosaics[int(idx)]
    file_path = f"{m['file_path']}/{m['file_name']}"
    try:
        Euclid.get_cutout(
            file_path=file_path,
            instrument=cfg.instrument,
            id=int(m["tile_index"]),
            coordinate=star_coord,
            radius=cutout_radius_arcmin * u.arcmin,
            output_file=output_file,
        )
    except Exception as e:
        return False, f"get_cutout failed: {type(e).__name__}: {e}"

    if not (os.path.exists(output_file) and os.path.getsize(output_file) > 0):
        return False, "downloaded file is empty"
    return True, None
