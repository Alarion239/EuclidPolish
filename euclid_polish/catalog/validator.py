"""FITS file validation utilities used by the cutout downloader."""


from typing import Any, cast

import numpy as np
from astropy.io import fits


def angular_separation_arcsec(
    ra1: float, dec1: float,
    ra2: float, dec2: float,
) -> float:
    """Small-angle angular separation between two sky positions, in arcseconds."""
    ra_diff  = (ra1 - ra2) * np.cos(np.deg2rad((dec1 + dec2) / 2))
    dec_diff = dec1 - dec2
    return np.sqrt(ra_diff**2 + dec_diff**2) * 3600.0


class FitsValidator:
    """Validates FITS cutout files for integrity, data quality, and WCS position."""

    def __init__(
        self,
        min_shape: int = 10,
        zero_tolerance: float = 1e-10,
        constant_tolerance: float = 1e-10
    ):
        self.min_shape = min_shape
        self.zero_tolerance = zero_tolerance
        self.constant_tolerance = constant_tolerance

    def validate_basic_integrity(self, filepath: str) -> tuple[bool, str | None]:
        """Return ``(True, None)`` iff the file opens, is 2D, non-trivial, and finite."""
        try:
            with fits.open(filepath) as hdul:
                data = self._extract_data(hdul)
                if data is None:
                    return False, "No data found in any HDU"
                if data.ndim != 2:
                    return False, f"Invalid dimensions: {data.ndim}D (expected 2D)"
                if data.shape[0] < self.min_shape or data.shape[1] < self.min_shape:
                    return False, f"Invalid shape: {data.shape} (too small)"
                if np.any(np.isnan(data)):
                    return False, f"Contains {np.sum(np.isnan(data))} NaN values"
                if np.any(np.isinf(data)):
                    return False, f"Contains {np.sum(np.isinf(data))} Inf values"
                if np.all(np.abs(data) < self.zero_tolerance):
                    return False, "All values are zero"
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
    ) -> tuple[bool, str | None]:
        """Basic integrity + WCS centre within ``tolerance_arcsec`` of expected."""
        is_valid, error_msg = self.validate_basic_integrity(filepath)
        if not is_valid:
            return False, error_msg
        try:
            with fits.open(filepath) as hdul:
                header = cast(fits.PrimaryHDU, hdul[0]).header
                if 'CRVAL1' in header and 'CRVAL2' in header:
                    sep = angular_separation_arcsec(
                        float(cast(Any, header['CRVAL1'])),
                        float(cast(Any, header['CRVAL2'])),
                        expected_ra, expected_dec)
                    if sep >= tolerance_arcsec:
                        return False, f"Center displaced: {sep:.2f} arcsec from expected"
        except Exception as e:
            return False, f"WCS validation error: {e}"
        return True, None

    def get_data(self, filepath: str) -> np.ndarray | None:
        """Safely return the first non-empty data array from a FITS file."""
        try:
            with fits.open(filepath) as hdul:
                return self._extract_data(hdul)
        except Exception:
            return None

    def get_header(self, filepath: str) -> dict | None:
        """Safely return the primary HDU header as a plain dict."""
        try:
            with fits.open(filepath) as hdul:
                return dict(cast(fits.PrimaryHDU, hdul[0]).header)
        except Exception:
            return None

    def _extract_data(self, hdul) -> np.ndarray | None:
        """Return the first HDU with non-empty data, or None."""
        for hdu in hdul:
            if hdu.data is not None and hasattr(hdu.data, 'size') and hdu.data.size > 0:
                return hdu.data
        return None
