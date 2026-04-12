"""
Unified FITS file validation module.

This module provides comprehensive FITS file validation functionality
used across multiple scripts, eliminating code duplication.
"""

import os
import numpy as np
from astropy.io import fits
from typing import Tuple, Optional


class FitsValidator:
    """
    Comprehensive FITS file validator.

    Provides methods to validate FITS files for various issues including
    corruption, data quality, and WCS header validation.
    """

    # Default validation thresholds
    MIN_SHAPE = 10  # Minimum dimension size
    ZERO_TOLERANCE = 1e-10  # Tolerance for considering values as zero
    CONSTANT_TOLERANCE = 1e-10  # Tolerance for detecting constant images

    def __init__(
        self,
        min_shape: int = 10,
        zero_tolerance: float = 1e-10,
        constant_tolerance: float = 1e-10
    ):
        """
        Initialize the validator.

        Parameters:
        -----------
        min_shape : int
            Minimum acceptable dimension size.
        zero_tolerance : float
            Tolerance for considering values as zero.
        constant_tolerance : float
            Tolerance for detecting constant images.
        """
        self.min_shape = min_shape
        self.zero_tolerance = zero_tolerance
        self.constant_tolerance = constant_tolerance

    def validate_basic_integrity(self, filepath: str) -> Tuple[bool, Optional[str]]:
        """
        Validate basic FITS file integrity.

        Checks:
        - File can be opened
        - Has valid data
        - 2D data array
        - Minimum shape requirements
        - No NaN values
        - No Inf values
        - Not all zeros
        - Not constant image

        Parameters:
        -----------
        filepath : str
            Path to FITS file.

        Returns:
        --------
        tuple
            (is_valid, error_message)
            - is_valid: True if file passes all checks
            - error_message: Description of issue if failed
        """
        try:
            with fits.open(filepath) as hdul:
                data = self._extract_data(hdul)
                if data is None:
                    return False, "No data found in any HDU"

                # Check dimensions
                if data.ndim != 2:
                    return False, f"Invalid dimensions: {data.ndim}D (expected 2D)"

                # Check shape
                if data.shape[0] < self.min_shape or data.shape[1] < self.min_shape:
                    return False, f"Invalid shape: {data.shape} (too small)"

                # Check for NaN
                if np.any(np.isnan(data)):
                    nan_count = np.sum(np.isnan(data))
                    return False, f"Contains {nan_count} NaN values"

                # Check for Inf
                if np.any(np.isinf(data)):
                    inf_count = np.sum(np.isinf(data))
                    return False, f"Contains {inf_count} Inf values"

                # Check if all zeros
                if np.all(np.abs(data) < self.zero_tolerance):
                    return False, "All values are zero"

                # Check if constant image
                if np.all(np.abs(data - data.flat[0]) < self.constant_tolerance):
                    return False, "All values are identical (constant image)"

                return True, None

        except fits.VerifyError as e:
            return False, f"FITS verification error: {e}"
        except OSError as e:
            return False, f"File read error: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"

    def validate_cutout(
        self,
        filepath: str,
        expected_ra: float,
        expected_dec: float,
        tolerance_arcsec: float = 0.5
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a FITS cutout against expected coordinates.

        Includes all basic integrity checks plus WCS position validation.

        Parameters:
        -----------
        filepath : str
            Path to FITS file.
        expected_ra : float
            Expected RA (degrees).
        expected_dec : float
            Expected Dec (degrees).
        tolerance_arcsec : float
            Position tolerance (arcseconds).

        Returns:
        --------
        tuple
            (is_valid, error_message)
        """
        # First check basic integrity
        is_valid, error_msg = self.validate_basic_integrity(filepath)
        if not is_valid:
            return False, error_msg

        # Check WCS position
        try:
            with fits.open(filepath) as hdul:
                header = hdul[0].header
                if 'CRVAL1' in header and 'CRVAL2' in header:
                    fits_ra = float(header['CRVAL1'])
                    fits_dec = float(header['CRVAL2'])

                    if not self._positions_match(
                        fits_ra, fits_dec,
                        expected_ra, expected_dec,
                        tolerance_arcsec
                    ):
                        separation = self._calculate_separation(
                            fits_ra, fits_dec,
                            expected_ra, expected_dec
                        )
                        return False, f"Center displaced: {separation:.2f} arcsec from expected"

        except Exception as e:
            return False, f"WCS validation error: {e}"

        return True, None

    def get_data(self, filepath: str) -> Optional[np.ndarray]:
        """
        Safely extract data array from FITS file.

        Parameters:
        -----------
        filepath : str
            Path to FITS file.

        Returns:
        --------
        numpy.ndarray or None
            Data array if successful, None otherwise.
        """
        try:
            with fits.open(filepath) as hdul:
                return self._extract_data(hdul)
        except Exception:
            return None

    def get_header(self, filepath: str) -> Optional[dict]:
        """
        Safely extract FITS header.

        Parameters:
        -----------
        filepath : str
            Path to FITS file.

        Returns:
        --------
        dict or None
            Header dictionary if successful, None otherwise.
        """
        try:
            with fits.open(filepath) as hdul:
                return dict(hdul[0].header)
        except Exception:
            return None

    def _extract_data(self, hdul) -> Optional[np.ndarray]:
        """
        Extract data from FITS HDU list.

        Parameters:
        -----------
        hdul : astropy.io.fits.HDUList
            Opened FITS file.

        Returns:
        --------
        numpy.ndarray or None
            First valid data array found.
        """
        for hdu in hdul:
            if hdu.data is not None:
                if hasattr(hdu.data, 'size') and hdu.data.size > 0:
                    return hdu.data
        return None

    def _positions_match(
        self,
        ra1: float, dec1: float,
        ra2: float, dec2: float,
        tolerance_arcsec: float
    ) -> bool:
        """
        Check if two positions match within tolerance.

        Parameters:
        -----------
        ra1, dec1 : float
            First position (degrees).
        ra2, dec2 : float
            Second position (degrees).
        tolerance_arcsec : float
            Tolerance in arcseconds.

        Returns:
        --------
        bool
            True if positions match within tolerance.
        """
        separation = self._calculate_separation(ra1, dec1, ra2, dec2)
        return separation < tolerance_arcsec

    def _calculate_separation(
        self,
        ra1: float, dec1: float,
        ra2: float, dec2: float
    ) -> float:
        """
        Calculate angular separation between two positions.

        Parameters:
        -----------
        ra1, dec1 : float
            First position (degrees).
        ra2, dec2 : float
            Second position (degrees).

        Returns:
        --------
        float
            Separation in arcseconds.
        """
        ra_diff = (ra1 - ra2) * np.cos(np.deg2rad((dec1 + dec2) / 2))
        dec_diff = dec1 - dec2
        separation_deg = np.sqrt(ra_diff**2 + dec_diff**2)
        return separation_deg * 3600.0


def validate_file_exists(filepath: str, name: str = "File") -> Tuple[bool, Optional[str]]:
    """
    Validate that a file exists.

    Parameters:
    -----------
    filepath : str
        Path to check.
    name : str
        Name for error messages.

    Returns:
    --------
    tuple
        (is_valid, error_message)
    """
    if not os.path.exists(filepath):
        return False, f"{name} not found: {filepath}"
    return True, None


def validate_directory_exists(dirpath: str, name: str = "Directory") -> Tuple[bool, Optional[str]]:
    """
    Validate that a directory exists.

    Parameters:
    -----------
    dirpath : str
        Path to check.
    name : str
        Name for error messages.

    Returns:
    --------
    tuple
        (is_valid, error_message)
    """
    if not os.path.isdir(dirpath):
        return False, f"{name} does not exist: {dirpath}"
    return True, None


def validate_range(
    value: float,
    min_val: float,
    max_val: float,
    name: str = "Value"
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a value is within range.

    Parameters:
    -----------
    value : float
        Value to check.
    min_val : float
        Minimum allowed value (inclusive).
    max_val : float
        Maximum allowed value (inclusive).
    name : str
        Name for error messages.

    Returns:
    --------
    tuple
        (is_valid, error_message)
    """
    if not (min_val <= value <= max_val):
        return False, f"{name} must be between {min_val} and {max_val}, got {value}"
    return True, None


def validate_positive(value: float, name: str = "Value") -> Tuple[bool, Optional[str]]:
    """
    Validate that a value is positive.

    Parameters:
    -----------
    value : float
        Value to check.
    name : str
        Name for error messages.

    Returns:
    --------
    tuple
        (is_valid, error_message)
    """
    if value <= 0:
        return False, f"{name} must be positive, got {value}"
    return True, None
