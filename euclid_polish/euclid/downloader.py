"""
Euclid cutout downloader module.

This module provides an object-oriented interface for downloading
Euclid VIS cutouts from the Euclid archive.
"""

import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
from astroquery.esa.euclid import Euclid
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm

from euclid_polish.euclid.catalog import StarCatalog
from euclid_polish.euclid.validator import FitsValidator, angular_separation_arcsec
from euclid_polish.config import Config


# Constants
POSITION_TOLERANCE_ARCSEC = 0.5  # Tolerance for position matching (arcsec)
SIZE_TOLERANCE_PIXELS = 10  # Tolerance for cutout size matching (pixels)


@dataclass
class DownloadConfig:
    """Configuration for cutout downloading."""
    cutout_size: int = Config.DEFAULT_CUTOUT_SIZE
    cutout_radius: float = 0.2  # arcmin
    position_tolerance: float = POSITION_TOLERANCE_ARCSEC
    size_tolerance: int = SIZE_TOLERANCE_PIXELS
    environment: str = "PDR"
    max_workers: int = 8  # parallel cutout HTTPS fetches

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
        self.cutout_dir = os.path.join(catalog.output_dir, Config.CUTOUTS_SUBDIR)
        os.makedirs(self.cutout_dir, exist_ok=True)

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

        query = (
            "SELECT file_path, file_name, tile_index, ra, dec "
            "FROM sedm.mosaic_product "
            "WHERE instrument_name = 'VIS'"
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
                instrument='VIS',
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
        show_progress: bool = True
    ) -> dict:
        """
        Download cutouts for stars in the catalog.

        Parameters:
        -----------
        star_ids : list of int, optional
            Specific star IDs to download. If None, downloads all missing stars.
        show_progress : bool
            Whether to show progress bar.

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

        # Handle corrupted files on disk — flag corruption per-size, not whole-star
        if corrupted_disk_files:
            stars_by_id = {s['id']: s for s in stars}
            for star_id, size, filepath in corrupted_disk_files:
                star = stars_by_id.get(star_id)
                if star is not None and size is not None:
                    StarCatalog.set_corrupted(star, size)
                try:
                    os.remove(filepath)
                except Exception:
                    pass
            self.catalog.save(catalog)

        # Cutout radius for the requested size
        pixel_scale_arcmin = Config.VIS_PIXEL_SCALE_ARCSEC / 60.0
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
                if not StarCatalog.is_valid(matching_star, fits_size):
                    StarCatalog.set_valid(matching_star, fits_size)
                if abs(fits_size - cutout_size) <= self.config.size_tolerance:
                    matched_stars.add(star_id)

        # Pending = not corrupted/failed at the requested size
        pending_stars = [
            s for s in stars
            if not StarCatalog.is_corrupted(s, cutout_size)
            and not StarCatalog.is_download_failed(s, cutout_size)
        ]
        if star_ids is not None:
            pending_stars = [s for s in pending_stars if s['id'] in star_ids]

        # Stars needing download = pending and not already valid at this size
        stars_needing_download = [
            s for s in pending_stars
            if s['id'] not in matched_stars
            or not StarCatalog.is_valid(s, cutout_size)
        ]

        self.catalog.save(catalog)

        if not stars_needing_download:
            valid_count = len([s for s in stars if StarCatalog.is_valid(s, cutout_size)])
            corrupted_count = len([s for s in stars if StarCatalog.is_corrupted(s, cutout_size)])
            return {
                'downloaded': 0,
                'valid': valid_count,
                'corrupted': corrupted_count,
                'cutout_size': cutout_size,
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
                for fut in progress:
                    star_id, ok = fut.result()
                    if not ok:
                        corrupted_star_ids.append(star_id)

        # Update catalog — per-size flags
        new_valid_ids = [
            s['id'] for s in stars_with_mosaics if s['id'] not in corrupted_star_ids
        ]
        for star in catalog['stars']:
            if star['id'] in new_valid_ids:
                StarCatalog.set_valid(star, cutout_size)
            if star['id'] in corrupted_star_ids:
                StarCatalog.set_corrupted(star, cutout_size)
            if star['id'] in unmatched_ids:
                StarCatalog.set_download_failed(star, cutout_size)

        self.catalog.save(catalog)

        # Final status — size-specific
        final_valid = len([s for s in catalog['stars'] if StarCatalog.is_valid(s, cutout_size)])
        final_corrupted = len([s for s in catalog['stars'] if StarCatalog.is_corrupted(s, cutout_size)])
        final_failed = len([s for s in catalog['stars'] if StarCatalog.is_download_failed(s, cutout_size)])

        return {
            'downloaded': len(new_valid_ids),
            'valid': final_valid,
            'corrupted': final_corrupted,
            'failed': final_failed,
            'corrupted_ids': corrupted_star_ids,
            'unmatched_ids': sorted(unmatched_ids),
            'cutout_size': cutout_size,
        }
