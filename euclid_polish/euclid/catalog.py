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
from euclid_polish.euclid.validator import angular_separation_arcsec

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

    # ------------------------------------------------------------------
    # Per-(band, size) status flags
    # ------------------------------------------------------------------
    #
    # The on-disk format is a doubly-nested mapping
    #     ``star["valid"] = { band_name: { str(size): bool } }``
    # for each of ``valid`` / ``corrupted`` / ``download_failed``. Older
    # catalog files using the band-less shape
    #     ``star["valid"] = { str(size): bool }``
    # are auto-promoted: every existing entry is treated as VIS, since
    # that was the only band the band-less downloader produced.

    DEFAULT_BAND = "VIS"

    @staticmethod
    def _read_flag(star: dict, kind: str, band: str, size: int | None) -> bool:
        """Generic getter for any of valid/corrupted/download_failed.

        Handles legacy formats transparently. ``size is None`` returns
        True when *any* size for ``band`` is set.
        """
        raw = star.get(kind, False)
        if isinstance(raw, bool):
            # Bare bool — applies to default band (VIS) and unknown size.
            return bool(raw) if band == StarCatalog.DEFAULT_BAND else False
        if not isinstance(raw, dict) or not raw:
            return False
        # Detect nesting depth: band → size → bool vs size → bool.
        sample_value = next(iter(raw.values()))
        if isinstance(sample_value, dict):
            band_slot = raw.get(band, {})
        else:
            # Band-less: treat the whole mapping as the default band.
            band_slot = raw if band == StarCatalog.DEFAULT_BAND else {}
        if not isinstance(band_slot, dict):
            return False
        if size is None:
            return any(band_slot.values())
        return bool(band_slot.get(str(size), False))

    @staticmethod
    def _write_flag(star: dict, kind: str, band: str, size: int, value: bool) -> None:
        """Generic setter that always writes the nested band→size structure.

        Auto-promotes a band-less mapping to the new nested format under
        the default band (VIS) before writing the new entry.
        """
        raw = star.get(kind, {})
        if isinstance(raw, bool):
            # Promote bare bool: it was treated as default-band/unknown-size,
            # which is the same as "no info" once we move to per-(band, size)
            # tracking.
            raw = {}
        if isinstance(raw, dict) and raw:
            sample_value = next(iter(raw.values()))
            if not isinstance(sample_value, dict):
                # Band-less mapping → promote to {default_band: raw}.
                raw = {StarCatalog.DEFAULT_BAND: dict(raw)}
        band_slot = raw.get(band, {})
        if not isinstance(band_slot, dict):
            band_slot = {}
        if value:
            band_slot[str(size)] = True
        else:
            band_slot.pop(str(size), None)
        if band_slot:
            raw[band] = band_slot
        else:
            raw.pop(band, None)
        if raw:
            star[kind] = raw
        else:
            star.pop(kind, None)

    @staticmethod
    def is_valid(star: dict, size: int | None = None,
                 band: str = "VIS") -> bool:
        """Return True if ``star`` has a valid cutout for ``(band, size)``.

        ``size=None`` → True when *any* size in ``band`` is valid.
        """
        return StarCatalog._read_flag(star, "valid", band, size)

    @staticmethod
    def set_valid(star: dict, size: int, band: str = "VIS") -> None:
        """Mark a star valid for ``(band, size)`` and clear any matching
        corruption flag (a successful re-download supersedes prior failure)."""
        StarCatalog._write_flag(star, "valid", band, size, True)
        StarCatalog._write_flag(star, "corrupted", band, size, False)

    @staticmethod
    def is_corrupted(star: dict, size: int | None = None,
                     band: str = "VIS") -> bool:
        return StarCatalog._read_flag(star, "corrupted", band, size)

    @staticmethod
    def set_corrupted(star: dict, size: int, band: str = "VIS") -> None:
        """Mark a star's ``(band, size)`` cutout as corrupted and clear any
        matching validity flag."""
        StarCatalog._write_flag(star, "corrupted", band, size, True)
        StarCatalog._write_flag(star, "valid", band, size, False)

    @staticmethod
    def is_download_failed(star: dict, size: int | None = None,
                           band: str = "VIS") -> bool:
        return StarCatalog._read_flag(star, "download_failed", band, size)

    @staticmethod
    def set_download_failed(star: dict, size: int, band: str = "VIS") -> None:
        StarCatalog._write_flag(star, "download_failed", band, size, True)

    @staticmethod
    def valid_sizes(star: dict, band: str = "VIS") -> list[int]:
        """Return the list of cutout sizes for which ``star`` is valid in ``band``."""
        raw = star.get("valid", False)
        if isinstance(raw, bool):
            return []
        if not isinstance(raw, dict) or not raw:
            return []
        sample_value = next(iter(raw.values()))
        band_slot = (raw.get(band, {}) if isinstance(sample_value, dict)
                     else (raw if band == StarCatalog.DEFAULT_BAND else {}))
        if not isinstance(band_slot, dict):
            return []
        return [int(k) for k, ok in band_slot.items() if ok]

    @staticmethod
    def valid_bands(star: dict, size: int | None = None) -> list[str]:
        """Return the list of bands for which ``star`` has a valid cutout.

        ``size=None`` requires only that *some* size is valid; pass a
        specific size to require that size."""
        raw = star.get("valid", False)
        if isinstance(raw, bool):
            return [StarCatalog.DEFAULT_BAND] if raw else []
        if not isinstance(raw, dict) or not raw:
            return []
        sample_value = next(iter(raw.values()))
        if not isinstance(sample_value, dict):
            # Band-less → default band only.
            return [StarCatalog.DEFAULT_BAND] if any(raw.values()) else []
        out = []
        for band_name, sizes in raw.items():
            if not isinstance(sizes, dict):
                continue
            if size is None:
                if any(sizes.values()):
                    out.append(band_name)
            elif sizes.get(str(size), False):
                out.append(band_name)
        return out

    def _has_any_flag(self, star: dict, kind: str) -> bool:
        """True if ``star`` carries ``kind`` for any (band, size)."""
        raw = star.get(kind, False)
        if isinstance(raw, bool):
            return bool(raw)
        if not isinstance(raw, dict) or not raw:
            return False
        sample = next(iter(raw.values()))
        if isinstance(sample, dict):
            return any(any(sizes.values()) for sizes in raw.values()
                       if isinstance(sizes, dict))
        return any(raw.values())

    def get_stars_by_status(self) -> dict:
        """Categorize stars by ``any-(band, size)`` status."""
        catalog = self.load()
        stars = catalog.get('stars', [])
        return {
            'valid':     [s for s in stars if self._has_any_flag(s, 'valid')],
            'corrupted': [s for s in stars if self._has_any_flag(s, 'corrupted')],
            'failed':    [s for s in stars if self._has_any_flag(s, 'download_failed')],
            'pending':   [s for s in stars
                          if not self._has_any_flag(s, 'valid')
                          and not self._has_any_flag(s, 'corrupted')
                          and not self._has_any_flag(s, 'download_failed')],
            'all':       stars,
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
        catalog = self.load()
        stars = catalog.get('stars', [])

        valid     = [s for s in stars if self._has_any_flag(s, 'valid')]
        corrupted = [s for s in stars if self._has_any_flag(s, 'corrupted')]
        failed    = [s for s in stars if self._has_any_flag(s, 'download_failed')]
        pending   = [s for s in stars
                     if not self._has_any_flag(s, 'valid')
                     and not self._has_any_flag(s, 'corrupted')
                     and not self._has_any_flag(s, 'download_failed')]

        summary = {
            'total':    len(stars),
            'valid':    len(valid),
            'corrupted': len(corrupted),
            'failed':   len(failed),
            'pending':  len(pending),
            'next_id':  catalog.get('next_id', 0),
        }
        # Per-band breakdown — most useful when a star has cutouts in some
        # bands but not others.
        from euclid_polish.config import Config as _Cfg
        per_band = {}
        for b in _Cfg.BANDS:
            per_band[b.name] = sum(1 for s in stars if self.is_valid(s, band=b.name))
        summary['valid_by_band'] = per_band

        mags = [s['magnitude'] for s in stars if s.get('magnitude') is not None]
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
        for star in existing_stars:
            if angular_separation_arcsec(ra, dec, star['ra'], star['dec']) < tolerance_arcsec:
                return True
        return False

    def _query_bright_stars(self, ra: float, dec: float, radius: float,
                            magnitude_limit: float,
                            num_stars: Optional[int] = None,
                            magnitude_min: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Query the Euclid catalog for stars in a region within a magnitude window.

        Parameters:
        -----------
        ra : float
            Right ascension of region center (degrees).
        dec : float
            Declination of region center (degrees).
        radius : float
            Search radius (degrees).
        magnitude_limit : float
            Faint-end cutoff (mag < magnitude_limit).
        num_stars : int, optional
            Maximum number of stars to return.
        magnitude_min : float, optional
            Bright-end cutoff (mag > magnitude_min). Drops bright stars
            that would saturate the detector.

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
                    mag = -2.5 * np.log10(flux) + Config.DEFAULT_VIS_ZEROPOINT
                    magnitudes.append(mag)

                # Add magnitude column
                results_valid.add_column(np.array(magnitudes), name='vis_magnitude')

                # Filter by magnitude window: faint-end cap + optional bright-end floor
                window_mask = results_valid['vis_magnitude'] < magnitude_limit
                if magnitude_min is not None:
                    window_mask &= results_valid['vis_magnitude'] > magnitude_min
                bright_stars = results_valid[window_mask]

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
                            magnitude_limit: float, num_stars: Optional[int] = None,
                            magnitude_min: Optional[float] = None) -> Dict[str, Any]:
        """
        Query Euclid catalog for stars in a magnitude window and add to catalog.

        Parameters:
        -----------
        ra : float
            Right ascension of region center (degrees, 0-360).
        dec : float
            Declination of region center (degrees, -90 to 90).
        radius : float
            Search radius (degrees).
        magnitude_limit : float
            Faint-end cutoff (mag < magnitude_limit). Anything dimmer is dropped.
        num_stars : int, optional
            Maximum number of stars to add.
        magnitude_min : float, optional
            Bright-end cutoff (mag > magnitude_min). Anything brighter is
            dropped — use this to exclude stars that saturate on NISP.

        Returns:
        --------
        dict
            Summary of changes with keys: 'added', 'skipped', 'total', 'next_id'.
        """
        if (magnitude_min is not None
                and magnitude_min >= magnitude_limit):
            raise ValueError(
                f"magnitude_min ({magnitude_min}) must be < "
                f"magnitude_limit ({magnitude_limit}) — the window would be empty."
            )
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
            magnitude_min=magnitude_min,
        )

        if not new_stars:
            return {
                'added': 0,
                'skipped': 0,
                'total': len(existing_stars),
                'next_id': next_id,
                'message': "No new stars found in this region"
            }

        return self._ingest_stars(new_stars, catalog, num_to_add=num_to_add)

    def _ingest_stars(
        self,
        new_stars: List[Dict[str, Any]],
        catalog: dict,
        num_to_add: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Dedupe + assign IDs + persist. Shared by all query methods."""
        existing_stars = catalog['stars']
        next_id = catalog['next_id']

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

    def query_brightest_stars(
        self,
        num_stars: int,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius: Optional[float] = None,
        magnitude_limit: Optional[float] = None,
        magnitude_min: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Query the Euclid archive for the brightest ``num_stars`` stars, sorted
        server-side by VIS flux (descending).

        Uses ``launch_job_async``, which bypasses the 2000-row synchronous cap.
        Authenticated users get persistent server-side storage of the job; the
        job results themselves are fetched immediately either way.

        Parameters
        ----------
        num_stars : int
            Maximum number of bright stars to return (ADQL ``TOP N``).
        ra, dec, radius : float, optional
            Optional cone (degrees) to restrict the search. All three must be
            provided together; omit all to search the full mer_catalogue.
        magnitude_limit : float, optional
            Faint-end cutoff (mag < magnitude_limit). Drops dim stars;
            converted server-side to a flux LOWER bound via
            ``Config.DEFAULT_VIS_ZEROPOINT``.
        magnitude_min : float, optional
            Bright-end cutoff (mag > magnitude_min). Drops bright stars
            that saturate the detector (especially useful for NISP);
            converted server-side to a flux UPPER bound.

        Returns
        -------
        dict
            Same shape as :meth:`query_euclid_catalog`.
        """
        if num_stars <= 0:
            raise ValueError("num_stars must be positive")
        if (magnitude_min is not None and magnitude_limit is not None
                and magnitude_min >= magnitude_limit):
            raise ValueError(
                f"magnitude_min ({magnitude_min}) must be < "
                f"magnitude_limit ({magnitude_limit}) — the window would be empty."
            )

        cone_given = [v is not None for v in (ra, dec, radius)]
        if any(cone_given) and not all(cone_given):
            raise ValueError("ra, dec, and radius must be provided together")

        where_clauses = [
            "flux_vis_1fwhm_aper IS NOT NULL",
            "flux_vis_1fwhm_aper > 0",
        ]
        if magnitude_limit is not None:
            flux_min = 10 ** ((Config.DEFAULT_VIS_ZEROPOINT - magnitude_limit) / 2.5)
            where_clauses.append(f"flux_vis_1fwhm_aper > {flux_min}")
        if magnitude_min is not None:
            flux_max = 10 ** ((Config.DEFAULT_VIS_ZEROPOINT - magnitude_min) / 2.5)
            where_clauses.append(f"flux_vis_1fwhm_aper < {flux_max}")
        if all(cone_given):
            where_clauses.append(
                f"CONTAINS(POINT('ICRS', right_ascension, declination), "
                f"CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1"
            )

        query = (
            f"SELECT TOP {num_stars} right_ascension, declination, flux_vis_1fwhm_aper "
            f"FROM catalogue.mer_catalogue "
            f"WHERE {' AND '.join(where_clauses)} "
            f"ORDER BY flux_vis_1fwhm_aper DESC"
        )

        try:
            job = Euclid.launch_job_async(query)
            results = job.get_results()
        except Exception as e:
            print(f"    Async query failed: {e}")
            return {
                'added': 0, 'skipped': 0,
                'total': 0, 'next_id': 0,
                'message': f"Query failed: {e}",
            }

        if results is None or len(results) == 0:
            catalog = self.load()
            return {
                'added': 0, 'skipped': 0,
                'total': len(catalog.get('stars', [])),
                'next_id': catalog.get('next_id', 0),
                'message': "No stars returned",
            }

        new_stars: List[Dict[str, Any]] = []
        for row in results:
            flux_raw = row['flux_vis_1fwhm_aper']
            # Even with `flux_vis_1fwhm_aper IS NOT NULL` in the WHERE clause,
            # astropy still returns a masked Table — guard against masked /
            # None / NaN before converting to float.
            if flux_raw is None or (hasattr(flux_raw, 'mask') and bool(flux_raw.mask)):
                continue
            flux = float(flux_raw)
            if not np.isfinite(flux) or flux <= 0:
                continue
            mag = -2.5 * np.log10(flux) + Config.DEFAULT_VIS_ZEROPOINT
            new_stars.append({
                'ra': float(row['right_ascension']),
                'dec': float(row['declination']),
                'magnitude': float(mag),
            })

        catalog = self.load()
        return self._ingest_stars(new_stars, catalog, num_to_add=num_stars)
