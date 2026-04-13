"""
Typed data structures for Euclid instrument products.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from astropy.io import fits

from euclid_polish.config import Config


def estimate_fwhm(profile: np.ndarray) -> float:
    """
    Estimate FWHM from a 1D profile.

    Parameters:
    -----------
    profile : ndarray
        1D profile (e.g. a row or column through a PSF image).

    Returns:
    --------
    float
        Estimated FWHM in pixels.
    """
    peak = np.max(profile)
    half_max = peak / 2.0
    above_half = profile > half_max
    if not np.any(above_half):
        return 0.0
    indices = np.where(above_half)[0]
    return float(indices[-1] - indices[0] + 1)


@dataclass
class PSF:
    """A PSF kernel with its physical metadata."""

    data: np.ndarray          # float32, shape (H, W) — should be normalized to sum=1
    pixel_scale: float        # arcsec / pixel of the kernel grid
    fwhm_arcsec: Optional[float] = None   # estimated FWHM in arcseconds
    oversampling: Optional[int] = None    # EPSFBuilder oversampling factor

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    def central_profile(self) -> np.ndarray:
        """Return the central row of the PSF image."""
        return self.data[self.data.shape[0] // 2]

    def fwhm_pixels(self) -> float:
        """Estimate FWHM in pixels from the central row."""
        return estimate_fwhm(self.central_profile())

    def normalized(self) -> PSF:
        """Return a copy with data normalized to sum=1."""
        total = self.data.sum()
        if total == 0:
            raise ValueError("PSF data sums to zero; cannot normalize.")
        return PSF(
            data=self.data / total,
            pixel_scale=self.pixel_scale,
            fwhm_arcsec=self.fwhm_arcsec,
            oversampling=self.oversampling,
        )

    def save(
        self,
        output_dir: str,
        filename_fits: str = Config.DEFAULT_PSF_FITS_FILENAME,
        filename_npy: str = Config.DEFAULT_PSF_NPY_FILENAME,
    ) -> tuple[str, str]:
        """
        Save the PSF to FITS and numpy files.

        The FITS file includes PXSCALE and OVERSAMP header keywords so the
        kernel can be fully reconstructed from the file alone.

        Parameters:
        -----------
        output_dir : str
            Output directory (created if absent).
        filename_fits : str
            FITS filename.
        filename_npy : str
            Numpy filename.

        Returns:
        --------
        tuple[str, str]
            (fits_path, npy_path)
        """
        os.makedirs(output_dir, exist_ok=True)

        fits_path = os.path.join(output_dir, filename_fits)
        primary_hdu = fits.PrimaryHDU(data=self.data)
        primary_hdu.header['PXSCALE'] = (self.pixel_scale, 'Pixel scale (arcsec/pixel)')
        if self.oversampling is not None:
            primary_hdu.header['OVERSAMP'] = (self.oversampling, 'Oversampling factor')
        if self.fwhm_arcsec is not None:
            primary_hdu.header['FWHM'] = (self.fwhm_arcsec, 'FWHM (arcsec)')
        primary_hdu.header['COMMENT'] = 'Euclid VIS PSF extracted from bright star cutouts'
        fits.HDUList([primary_hdu]).writeto(fits_path, overwrite=True)

        npy_path = os.path.join(output_dir, filename_npy)
        np.save(npy_path, self.data)

        print(f"Saved PSF to: {fits_path}")
        print(f"Saved PSF numpy array to: {npy_path}")
        return fits_path, npy_path

    @classmethod
    def load(cls, npy_path: str, pixel_scale: float) -> PSF:
        """
        Load a PSF from a numpy file.

        Parameters:
        -----------
        npy_path : str
            Path to the .npy file.
        pixel_scale : float
            Pixel scale of the kernel grid in arcsec/pixel.

        Returns:
        --------
        PSF
        """
        data = np.load(npy_path).astype(np.float32)
        return cls(data=data, pixel_scale=pixel_scale)

    @classmethod
    def from_fits(cls, fits_path: str) -> PSF:
        """
        Load a PSF from a FITS file, reading pixel scale and oversampling
        from the header keywords written by :meth:`save`.

        Parameters:
        -----------
        fits_path : str
            Path to the FITS file saved by :meth:`save`.

        Returns:
        --------
        PSF
        """
        with fits.open(fits_path) as hdul:
            header = hdul[0].header
            data = hdul[0].data.astype(np.float32)
        pixel_scale = float(header.get('PXSCALE', 0.0))
        oversampling = int(header['OVERSAMP']) if 'OVERSAMP' in header else None
        fwhm_arcsec  = float(header['FWHM'])   if 'FWHM'    in header else None
        return cls(
            data=data,
            pixel_scale=pixel_scale,
            fwhm_arcsec=fwhm_arcsec,
            oversampling=oversampling,
        )
