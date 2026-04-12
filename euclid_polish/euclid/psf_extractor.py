"""
OOP PSF Extractor module for Euclid data.

This module provides an object-oriented interface for extracting effective PSF
from Euclid FITS cutouts using photutils.
"""

import os
import glob
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from photutils.psf import EPSFModel, EPSFBuilder, EPSFStars, EPSFStar
from tqdm import tqdm

from euclid_polish.config import Config


@dataclass
class PSFExtractionConfig:
    """Configuration for PSF extraction."""
    psf_size: int = Config.DEFAULT_PSF_SIZE
    fwhm: float = Config.DEFAULT_PSF_FWHM
    threshold: float = Config.DEFAULT_PSF_THRESHOLD
    max_iters: int = Config.DEFAULT_PSF_MAX_ITERS
    accuracy: float = Config.DEFAULT_PSF_ACCURACY
    progress_bar: bool = True
    oversampling: int = 4

    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate configuration."""
        if self.psf_size <= 0 or self.psf_size % 2 == 0:
            return False, "PSF size must be a positive odd integer"
        if self.fwhm <= 0:
            return False, "FWHM must be positive"
        if self.threshold <= 0:
            return False, "Threshold must be positive"
        if self.max_iters <= 0:
            return False, "Max iterations must be positive"
        if self.accuracy <= 0:
            return False, "Accuracy must be positive"
        return True, None


class PSFExtractor:
    """
    Extract effective PSF from Euclid FITS cutouts.

    This class encapsulates the PSF extraction workflow, including:
    - Loading and validating FITS cutouts
    - Extracting PSF stars from cutouts
    - Building effective PSF using photutils
    - Saving results
    """

    def __init__(self, config: Optional[PSFExtractionConfig] = None):
        """
        Initialize the PSF extractor.

        Parameters:
        -----------
        config : PSFExtractionConfig, optional
            Configuration for PSF extraction. Uses defaults if not provided.
        """
        self.config = config or PSFExtractionConfig()
        self.epsf: Optional[EPSFModel] = None
        self.fitted_stars = None

    def get_cutout_files(self, cutout_dir: str) -> List[Tuple[int, str]]:
        """
        Get all FITS cutout files from directory, sorted by index.

        Parameters:
        -----------
        cutout_dir : str
            Directory containing cutout files.

        Returns:
        --------
        list of tuple
            List of (index, filepath) tuples sorted by index.
        """
        fits_files = glob.glob(os.path.join(cutout_dir, "*.fits"))

        file_info = []
        for filepath in fits_files:
            filename = os.path.basename(filepath)
            try:
                # Extract index from "star_XXXX_..."
                parts = filename.split('_')
                if len(parts) >= 2 and parts[0] == 'star':
                    index = int(parts[1])
                    file_info.append((index, filepath))
            except (ValueError, IndexError):
                continue

        file_info.sort(key=lambda x: x[0])
        return file_info

    def select_files(
        self,
        all_files: List[Tuple[int, str]],
        indices: Optional[List[int]] = None,
        num_stars: Optional[int] = None
    ) -> List[Tuple[int, str]]:
        """
        Select files for PSF extraction.

        Parameters:
        -----------
        all_files : list of tuple
            All available (index, filepath) tuples.
        indices : list of int, optional
            Specific star indices to use.
        num_stars : int, optional
            Maximum number of stars to use.

        Returns:
        --------
        list of tuple
            Selected (index, filepath) tuples.
        """
        if indices is not None:
            selected = []
            for idx in indices:
                for file_idx, filepath in all_files:
                    if file_idx == idx:
                        selected.append((file_idx, filepath))
                        break
            return selected
        elif num_stars is not None:
            return all_files[:num_stars]
        else:
            return all_files

    def load_cutout(self, filepath: str) -> Optional[np.ndarray]:
        """
        Load a FITS cutout file.

        Parameters:
        -----------
        filepath : str
            Path to FITS file.

        Returns:
        --------
        numpy.ndarray or None
            Image data, or None if loading failed.
        """
        try:
            with fits.open(filepath) as hdul:
                for hdu in hdul:
                    if hdu.data is not None:
                        return hdu.data
        except Exception:
            return None
        return None

    def extract_psf_star_from_cutout(
        self,
        image_data: np.ndarray
    ) -> Optional[EPSFStar]:
        """
        Extract a PSF star from a cutout image.

        Assumes the star is centered in the image.

        Parameters:
        -----------
        image_data : numpy.ndarray
            Image data.

        Returns:
        --------
        EPSFStar or None
            Extracted PSF star, or None if extraction failed.
        """
        psf_size = self.config.psf_size
        ny, nx = image_data.shape
        center_y, center_x = ny // 2, nx // 2

        # Extract a centered cutout
        half_size = psf_size // 2
        y_min = center_y - half_size
        y_max = center_y + half_size + 1
        x_min = center_x - half_size
        x_max = center_x + half_size + 1

        # Handle edge cases
        if y_min < 0 or y_max > ny or x_min < 0 or x_max > nx:
            return None

        cutout = image_data[y_min:y_max, x_min:x_max]

        # Verify cutout shape
        if cutout.shape != (psf_size, psf_size):
            return None

        # Normalize cutout
        cutout_sum = np.sum(cutout)
        if cutout_sum > 0:
            cutout = cutout / cutout_sum
        else:
            return None

        # Create EPSFStar object
        epsf_star = EPSFStar(
            data=cutout,
            cutout_center=(half_size, half_size),
        )

        return epsf_star

    def extract_psf_stars_from_files(
        self,
        cutout_files: List[Tuple[int, str]]
    ) -> List[EPSFStar]:
        """
        Extract PSF stars from multiple cutout files.

        Parameters:
        -----------
        cutout_files : list of tuple
            List of (index, filepath) tuples.

        Returns:
        --------
        list of EPSFStar
            Extracted PSF stars.
        """
        all_epsf_stars = []

        iterator = tqdm(
            cutout_files,
            desc="Processing cutouts",
            disable=not self.config.progress_bar
        )

        for index, filepath in iterator:
            try:
                image_data = self.load_cutout(filepath)
                if image_data is None:
                    print(f"  Warning: Could not load {filepath}")
                    continue

                epsf_star = self.extract_psf_star_from_cutout(image_data)
                if epsf_star is not None:
                    all_epsf_stars.append(epsf_star)

            except Exception as e:
                print(f"  Warning: Error processing {filepath}: {e}")
                continue

        return all_epsf_stars

    def build_epsf(
        self,
        cutout_files: List[Tuple[int, str]]
    ) -> Tuple[EPSFModel, EPSFStars]:
        """
        Build effective PSF from cutout files.

        Parameters:
        -----------
        cutout_files : list of tuple
            List of (index, filepath) tuples.

        Returns:
        --------
        tuple
            (EPSFModel, EPSFStars) - The built ePSF and fitted stars.
        """
        # Extract PSF stars from cutouts
        all_epsf_stars = self.extract_psf_stars_from_files(cutout_files)

        if len(all_epsf_stars) == 0:
            raise ValueError("No valid PSF stars extracted from cutouts")

        print(f"Extracted {len(all_epsf_stars)} PSF stars from cutouts")

        # Build ePSF
        epsf_stars = EPSFStars(all_epsf_stars)

        print("Building effective PSF...")
        epsf_builder = EPSFBuilder(
            oversampling=self.config.oversampling,
            maxiters=self.config.max_iters,
            progress_bar=self.config.progress_bar,
            center_accuracy=self.config.accuracy,
            smoothing_kernel=None,
        )

        epsf, fitted_stars = epsf_builder(epsf_stars)

        self.epsf = epsf
        self.fitted_stars = fitted_stars

        return epsf, fitted_stars

    def save_psf(
        self,
        output_dir: str,
        filename_fits: str = "euclid_psf.fits",
        filename_npy: str = "euclid_psf.npy"
    ) -> Tuple[str, str]:
        """
        Save the extracted PSF to FITS and numpy files.

        Parameters:
        -----------
        output_dir : str
            Output directory.
        filename_fits : str
            FITS filename.
        filename_npy : str
            Numpy filename.

        Returns:
        --------
        tuple
            (fits_path, npy_path) - Paths to saved files.
        """
        if self.epsf is None:
            raise ValueError("No PSF has been built yet. Call build_epsf() first.")

        os.makedirs(output_dir, exist_ok=True)

        # Save as FITS
        fits_path = os.path.join(output_dir, filename_fits)
        psf_data = self.epsf.data

        primary_hdu = fits.PrimaryHDU(data=psf_data)
        primary_hdu.header['AUTHOR'] = 'PSFExtractor'
        oversamp_val = self.epsf.oversampling[0] if hasattr(self.epsf.oversampling, '__iter__') else self.epsf.oversampling
        primary_hdu.header['OVERSAMP'] = (oversamp_val, 'Oversampling factor')
        primary_hdu.header['COMMENT'] = 'Euclid VIS PSF extracted from bright star cutouts'

        hdul = fits.HDUList([primary_hdu])
        hdul.writeto(fits_path, overwrite=True)

        # Save as numpy
        npy_path = os.path.join(output_dir, filename_npy)
        np.save(npy_path, psf_data)

        print(f"Saved PSF to: {fits_path}")
        print(f"Saved PSF numpy array to: {npy_path}")

        return fits_path, npy_path

    def get_summary(self) -> dict:
        """
        Get summary of the PSF extraction.

        Returns:
        --------
        dict
            Summary information.
        """
        if self.epsf is None:
            return {
                'status': 'No PSF built yet',
            }

        return {
            'status': 'success',
            'shape': self.epsf.data.shape,
            'oversampling': self.epsf.oversampling,
            'data_type': str(self.epsf.data.dtype),
        }


def estimate_fwhm(profile: np.ndarray) -> float:
    """
    Estimate FWHM from a 1D PSF profile.

    Parameters:
    -----------
    profile : numpy.ndarray
        1D profile through the PSF.

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
