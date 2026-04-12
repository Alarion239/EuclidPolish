"""
Star catalog management module.

This module provides a centralized interface for working with the stars.json catalog,
eliminating code duplication across multiple scripts.
"""

import os
import json
from typing import Optional, List, Dict, Any

import numpy as np
from astroquery.esa.euclid import Euclid

from euclid_polish.config import Config

# Constants
POSITION_TOLERANCE_ARCSEC = 0.05  # Tolerance for duplicate detection (arcsec)


class StarCatalog:
    """
    Manage the stars.json catalog file.

    This class provides methods to load, save, and query the star catalog,
    ensuring consistent behavior across all CLI commands.
    """

    def __init__(self, output_dir: str = Config.DEFAULT_OUTPUT_DIR):
        """
        Initialize the catalog manager.

        Parameters:
        -----------
        output_dir : str
            Directory containing the stars.json file.
        """
        self.output_dir = output_dir
        self.catalog_path = os.path.join(output_dir, Config.CATALOG_FILE)

    def load(self) -> dict:
        """
        Load the catalog from JSON file.

        Returns:
        --------
        dict
            Catalog dictionary with 'stars' list and 'next_id' counter.
            Returns empty catalog if file doesn't exist.
        """
        if not os.path.exists(self.catalog_path):
            return {"stars": [], "next_id": 0}

        with open(self.catalog_path, 'r') as f:
            return json.load(f)

    def save(self, catalog: dict) -> None:
        """
        Save the catalog to JSON file.

        Parameters:
        -----------
        catalog : dict
            Catalog dictionary with 'stars' list and 'next_id' counter.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.catalog_path, 'w') as f:
            json.dump(catalog, f, indent=2)

    def exists(self) -> bool:
        """Check if the catalog file exists."""
        return os.path.exists(self.catalog_path)

    def get_stars_by_status(self) -> dict:
        """
        Categorize stars by their status.

        Returns:
        --------
        dict
            Dictionary with keys: 'valid', 'corrupted', 'failed', 'pending'
            Each containing a list of stars.
        """
        catalog = self.load()
        stars = catalog.get('stars', [])

        return {
            'valid': [s for s in stars if s.get('valid', False)],
            'corrupted': [s for s in stars if s.get('corrupted', False)],
            'failed': [s for s in stars if s.get('download_failed', False)],
            'pending': [s for s in stars if not s.get('valid', False)
                       and not s.get('corrupted', False)
                       and not s.get('download_failed', False)],
            'all': stars,
        }

    def get_star_by_id(self, star_id: int) -> Optional[dict]:
        """
        Get a specific star by ID.

        Parameters:
        -----------
        star_id : int
            The star ID to look up.

        Returns:
        --------
        dict or None
            Star dictionary if found, None otherwise.
        """
        catalog = self.load()
        for star in catalog.get('stars', []):
            if star.get('id') == star_id:
                return star
        return None

    def get_summary(self) -> dict:
        """
        Get a summary of catalog statistics.

        Returns:
        --------
        dict
            Summary with counts and metadata.
        """
        status = self.get_stars_by_status()
        catalog = self.load()

        summary = {
            'total': len(status['all']),
            'valid': len(status['valid']),
            'corrupted': len(status['corrupted']),
            'failed': len(status['failed']),
            'pending': len(status['pending']),
            'next_id': catalog.get('next_id', 0),
        }

        # Calculate magnitude range
        mags = [s['magnitude'] for s in status['all'] if s.get('magnitude') is not None]
        if mags:
            summary['mag_min'] = min(mags)
            summary['mag_max'] = max(mags)

        return summary

    def _is_duplicate_star(self, ra: float, dec: float, existing_stars: List[Dict],
                           tolerance_arcsec: float = POSITION_TOLERANCE_ARCSEC) -> bool:
        """
        Check if a star position is already in the existing catalog.

        Parameters:
        -----------
        ra : float
            Right ascension (degrees).
        dec : float
            Declination (degrees).
        existing_stars : list
            List of existing star dictionaries.
        tolerance_arcsec : float
            Position tolerance in arcseconds.

        Returns:
        --------
        bool
            True if star is a duplicate (within tolerance).
        """
        tolerance_deg = tolerance_arcsec / 3600.0  # Convert arcsec to degrees

        for star in existing_stars:
            # Calculate angular separation (simple approximation for small distances)
            ra_diff = (ra - star['ra']) * np.cos(np.deg2rad((dec + star['dec']) / 2))
            dec_diff = dec - star['dec']
            separation_deg = np.sqrt(ra_diff**2 + dec_diff**2)

            if separation_deg < tolerance_deg:
                return True

        return False

    def _query_bright_stars(self, ra: float, dec: float, radius: float,
                            magnitude_limit: float, num_stars: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query the Euclid catalog for bright stars in a region.

        Parameters:
        -----------
        ra : float
            Right ascension of region center (degrees).
        dec : float
            Declination of region center (degrees).
        radius : float
            Search radius (degrees).
        magnitude_limit : float
            Magnitude limit (brighter = smaller values).
        num_stars : int, optional
            Maximum number of stars to return.

        Returns:
        --------
        list of dict
            List of star dictionaries with 'ra', 'dec', 'magnitude'.
        """
        # Try ADQL query on mer_catalogue table
        query = f"""
        SELECT TOP 100000
            right_ascension,
            declination,
            flux_vis_1fwhm_aper
        FROM catalogue.mer_catalogue
        WHERE CONTAINS(
            POINT('ICRS', right_ascension, declination),
            CIRCLE('ICRS', {ra}, {dec}, {radius})
        ) = 1
            AND flux_vis_1fwhm_aper IS NOT NULL
        """

        try:
            job = Euclid.launch_job(query)
            results = job.get_results()

            if results is not None and len(results) > 0:
                # Sort by flux (brightest first)
                flux_values = np.array(results['flux_vis_1fwhm_aper'])
                sorted_indices = np.argsort(flux_values)[::-1]

                # Take top 10000 if we got more
                if len(sorted_indices) > 10000:
                    sorted_indices = sorted_indices[:10000]
                else:
                    results = results[sorted_indices]

                # VIS zeropoint
                vis_zeropoint = 26.2

                # Filter out invalid/missing flux values
                valid_flux_mask = []
                for flux in results['flux_vis_1fwhm_aper']:
                    is_valid = True
                    if flux is None:
                        is_valid = False
                    elif hasattr(flux, 'mask') and flux.mask:
                        is_valid = False
                    elif flux <= 0:
                        is_valid = False
                    valid_flux_mask.append(is_valid)

                valid_flux_mask = np.array(valid_flux_mask)
                results_valid = results[valid_flux_mask]

                if len(results_valid) == 0:
                    return []

                # Convert valid fluxes to magnitudes
                magnitudes = []
                for flux in results_valid['flux_vis_1fwhm_aper']:
                    mag = -2.5 * np.log10(flux) + vis_zeropoint
                    magnitudes.append(mag)

                # Add magnitude column
                results_valid.add_column(np.array(magnitudes), name='vis_magnitude')

                # Filter by magnitude limit
                bright_mask = results_valid['vis_magnitude'] < magnitude_limit
                bright_stars = results_valid[bright_mask]

                if len(bright_stars) == 0:
                    return []

                # Sort by magnitude (brightest first)
                bright_stars = bright_stars[np.argsort(bright_stars['vis_magnitude'])]

                # Convert to list of dictionaries
                stars = []
                for star in bright_stars:
                    stars.append({
                        'ra': float(star['right_ascension']),
                        'dec': float(star['declination']),
                        'magnitude': float(star['vis_magnitude'])
                    })

                    # Limit to num_stars if specified
                    if num_stars is not None and len(stars) >= num_stars:
                        break

                return stars
            else:
                return []
        except Exception as e:
            print(f"    Query failed: {e}")
            return []

    def query_euclid_catalog(self, ra: float, dec: float, radius: float,
                            magnitude_limit: float, num_stars: Optional[int] = None) -> Dict[str, Any]:
        """
        Query Euclid catalog for bright stars and add to this catalog.

        Parameters:
        -----------
        ra : float
            Right ascension of region center (degrees, 0-360).
        dec : float
            Declination of region center (degrees, -90 to 90).
        radius : float
            Search radius (degrees).
        magnitude_limit : float
            Magnitude limit for bright stars (fainter = more stars).
        num_stars : int, optional
            Maximum number of stars to add.

        Returns:
        --------
        dict
            Summary of changes with keys: 'added', 'skipped', 'total', 'next_id'.
        """
        catalog = self.load()
        existing_stars = catalog['stars']
        next_id = catalog['next_id']

        # Count available stars (not corrupted/failed)
        available_stars = [s for s in existing_stars
                          if not s.get('corrupted', False)
                          and not s.get('download_failed', False)]
        existing_available_count = len(available_stars)

        # Calculate how many more to add
        if num_stars is not None:
            num_to_add = max(0, num_stars - existing_available_count)
            if num_to_add == 0:
                return {
                    'added': 0,
                    'skipped': 0,
                    'total': len(existing_stars),
                    'next_id': next_id,
                    'message': f"Already have {existing_available_count}/{num_stars} available stars"
                }
        else:
            num_to_add = None

        # Query for new stars
        new_stars = self._query_bright_stars(
            ra=ra,
            dec=dec,
            radius=radius,
            magnitude_limit=magnitude_limit,
            num_stars=num_to_add,
        )

        if not new_stars:
            return {
                'added': 0,
                'skipped': 0,
                'total': len(existing_stars),
                'next_id': next_id,
                'message': "No new stars found in this region"
            }

        # Filter out duplicates and add stars
        added_count = 0
        skipped_count = 0

        for star in new_stars:
            if num_to_add is not None and added_count >= num_to_add:
                break
            if self._is_duplicate_star(star['ra'], star['dec'], existing_stars):
                skipped_count += 1
            else:
                star['id'] = next_id
                existing_stars.append(star)
                next_id += 1
                added_count += 1

        # Update and save catalog
        catalog['stars'] = existing_stars
        catalog['next_id'] = next_id
        self.save(catalog)

        return {
            'added': added_count,
            'skipped': skipped_count,
            'total': len(existing_stars),
            'next_id': next_id,
            'message': f"Added {added_count} stars → {len(existing_stars)} total in catalog"
        }
