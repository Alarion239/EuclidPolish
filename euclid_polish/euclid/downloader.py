"""
Euclid cutout downloader module.

This module provides an object-oriented interface for downloading
Euclid VIS cutouts from the Euclid archive.
"""

import os
import glob
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from astroquery.esa.euclid import Euclid
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm

from euclid_polish.euclid.catalog import StarCatalog
from euclid_polish.euclid.validator import FitsValidator
from euclid_polish.config import Config


# Constants
VIS_PIXEL_SCALE_ARCSEC = 0.10  # VIS pixel scale: 0.10 arcsec/pixel
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

    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate configuration."""
        if self.cutout_size <= 0:
            return False, "Cutout size must be positive"
        if self.cutout_radius <= 0:
            return False, "Cutout radius must be positive"
        if self.position_tolerance <= 0:
            return False, "Position tolerance must be positive"
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

    def get_existing_cutouts(self) -> Tuple[dict, List[Tuple[int, str]]]:
        """
        Scan FITS files and extract star positions from WCS headers.

        Returns:
        --------
        tuple
            (star_positions dict, corrupted_files list)
            - star_positions: dict mapping star_id to (ra, dec, size, filepath)
            - corrupted_files: list of (star_id, filepath) tuples that are corrupted
        """
        star_positions = {}
        corrupted_files = []

        fits_files = glob.glob(os.path.join(self.cutout_dir, "star_[0-9][0-9][0-9][0-9]_*.fits"))

        for filepath in fits_files:
            filename = os.path.basename(filepath)
            try:
                # Extract star_id and size from filename "star_XXXX_SIZE.fits"
                parts = filename.split('_')
                if len(parts) >= 3 and parts[0] == 'star':
                    star_id = int(parts[1])
                    size = int(parts[2].replace('.fits', ''))

                    # Use validator to get header
                    header = self.validator.get_header(filepath)
                    if header and 'CRVAL1' in header and 'CRVAL2' in header:
                        ra = float(header['CRVAL1'])
                        dec = float(header['CRVAL2'])
                        star_positions[star_id] = (ra, dec, size, filepath)
                    else:
                        # File is corrupted if we can't read header
                        corrupted_files.append((star_id, filepath))
            except (ValueError, IndexError, KeyError, OSError) as e:
                # File is corrupted - record it
                parts = filename.split('_')
                if len(parts) >= 2 and parts[0] == 'star':
                    try:
                        star_id = int(parts[1])
                        corrupted_files.append((star_id, filepath))
                    except (ValueError, IndexError):
                        pass
                print(f"Warning: Could not parse {filename}: {e}")
                continue

        return star_positions, corrupted_files

    def download_cutout(
        self,
        star: dict,
        cutout_radius_arcmin: float,
        output_file: str
    ) -> bool:
        """
        Download a single cutout for a star.

        Parameters:
        -----------
        star : dict
            Star dictionary with 'ra', 'dec', 'magnitude'.
        cutout_radius_arcmin : float
            Radius of cutout (arcmin).
        output_file : str
            Output file path.

        Returns:
        --------
        bool
            True if download succeeded, False otherwise.
        """
        ra = star['ra']
        dec = star['dec']
        coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')

        # Search for mosaic tiles
        search_radius = 0.5 * u.degree

        try:
            job = Euclid.cone_search(
                coordinate=coord,
                radius=search_radius,
                table_name="sedm.mosaic_product",
                ra_column_name="ra",
                dec_column_name="dec",
                columns="*",
            )
            mosaics = job.get_results()
        except Exception:
            return False

        # Filter for VIS instrument
        vis_mosaics = mosaics[mosaics['instrument_name'] == 'VIS']

        if len(vis_mosaics) == 0:
            return False

        # Use the first matching mosaic
        mosaic = vis_mosaics[0]
        file_path = mosaic['file_path'] + "/" + mosaic['file_name']
        obs_id = mosaic['tile_index']

        # Download cutout
        try:
            result = Euclid.get_cutout(
                file_path=file_path,
                instrument='VIS',
                id=obs_id,
                coordinate=coord,
                radius=cutout_radius_arcmin * u.arcmin,
                output_file=output_file,
            )

            # Basic validation - file exists and has content
            if not (os.path.exists(output_file) and os.path.getsize(output_file) > 0):
                return False

            return True

        except Exception:
            return False

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
        return self.validator._positions_match(
            ra1, dec1, ra2, dec2, self.config.position_tolerance
        )

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
        pending_stars = [s for s in stars if not s.get('corrupted', False)
                         and not s.get('download_failed', False)]

        # Scan existing FITS files
        existing_fits, corrupted_disk_files = self.get_existing_cutouts()

        # Handle corrupted files on disk
        if corrupted_disk_files:
            corrupted_ids = [star_id for star_id, _ in corrupted_disk_files]
            for star in catalog['stars']:
                if star['id'] in corrupted_ids:
                    star['corrupted'] = True
            for _, filepath in corrupted_disk_files:
                try:
                    os.remove(filepath)
                except Exception:
                    pass
            self.catalog.save(catalog)

        # Calculate cutout radius
        pixel_scale_arcmin = VIS_PIXEL_SCALE_ARCSEC / 60.0
        cutout_radius_arcmin = (self.config.cutout_size / 2.0) * pixel_scale_arcmin
        cutout_size = self.config.cutout_size

        # Identify stars needing download
        stars_needing_download = []
        matched_stars = set()

        # Check existing FITS files
        for star_id, (fits_ra, fits_dec, fits_size, filepath) in existing_fits.items():
            matching_star = None
            for star in stars:
                if star['id'] == star_id:
                    matching_star = star
                    break

            if matching_star is None:
                continue

            if not self.positions_match(fits_ra, fits_dec, matching_star['ra'], matching_star['dec']):
                continue

            size_diff = abs(fits_size - cutout_size)
            if size_diff > self.config.size_tolerance:
                continue

            matched_stars.add(star_id)
            if not matching_star.get('valid', False):
                matching_star['valid'] = True

        # Filter by star_ids if specified
        if star_ids is not None:
            pending_stars = [s for s in pending_stars if s['id'] in star_ids]

        # Find stars without files
        for star in pending_stars:
            if star['id'] not in matched_stars:
                stars_needing_download.append(star)

        # Save updated catalog
        self.catalog.save(catalog)

        if not stars_needing_download:
            valid_count = len([s for s in stars if s.get('valid', False)])
            return {
                'downloaded': 0,
                'valid': valid_count,
                'corrupted': len([s for s in stars if s.get('corrupted', False)]),
            }

        # Download with validation
        corrupted_star_ids = []

        iterator = tqdm(stars_needing_download, desc="Downloading", disable=not show_progress)

        for star in iterator:
            star_id = star['id']
            filename = f"star_{star_id:04d}_{cutout_size}.fits"
            output_file = os.path.join(self.cutout_dir, filename)

            # Download and validate
            download_ok = False
            validation_ok = False

            if self.download_cutout(star, cutout_radius_arcmin, output_file):
                download_ok = True
                is_valid, error_msg = self.validator.validate_cutout(
                    output_file, star['ra'], star['dec'], self.config.position_tolerance
                )
                if is_valid:
                    validation_ok = True
                else:
                    os.remove(output_file)

            # Retry once if validation failed
            if download_ok and not validation_ok:
                if self.download_cutout(star, cutout_radius_arcmin, output_file):
                    is_valid, _ = self.validator.validate_cutout(
                        output_file, star['ra'], star['dec'], self.config.position_tolerance
                    )
                    if is_valid:
                        validation_ok = True
                    else:
                        if os.path.exists(output_file):
                            os.remove(output_file)

            if not validation_ok:
                corrupted_star_ids.append(star_id)

        # Update catalog
        new_valid_ids = [s['id'] for s in stars_needing_download if s['id'] not in corrupted_star_ids]
        for star in catalog['stars']:
            if star['id'] in new_valid_ids:
                star['valid'] = True

        if corrupted_star_ids:
            for star in catalog['stars']:
                if star['id'] in corrupted_star_ids:
                    star['corrupted'] = True

        self.catalog.save(catalog)

        # Final status
        final_valid = len([s for s in catalog['stars'] if s.get('valid', False)])
        final_corrupted = len([s for s in catalog['stars'] if s.get('corrupted', False)])

        return {
            'downloaded': len(new_valid_ids),
            'valid': final_valid,
            'corrupted': final_corrupted,
            'corrupted_ids': corrupted_star_ids,
        }
