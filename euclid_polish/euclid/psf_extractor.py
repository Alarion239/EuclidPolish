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
from euclid_polish.euclid.types import PSF


@dataclass
class PSFExtractionConfig:
    """Configuration for PSF extraction.

    ``psf_size``      controls the *input* cutout extracted from each star
                      stamp (native pixels, must be odd).
    ``output_size``   controls the *output* PSF shape (in oversampled
                      pixels). ``None`` → photutils' default for the
                      ``oversampling`` setting (typically
                      ``psf_size × oversampling + 1``). When set, the
                      output PSF is exactly ``output_size × output_size``
                      (any even value is silently bumped down to the
                      nearest odd, e.g. ``1024 → 1023``).
    """
    psf_size: int = Config.DEFAULT_PSF_SIZE
    fwhm: float = Config.DEFAULT_PSF_FWHM
    threshold: float = Config.DEFAULT_PSF_THRESHOLD
    max_iters: int = Config.DEFAULT_PSF_MAX_ITERS
    accuracy: float = Config.DEFAULT_PSF_ACCURACY
    progress_bar: bool = True
    oversampling: int = Config.DEFAULT_REBIN_FACTOR
    output_size: Optional[int] = None
    # Saturation rejection: Euclid MER zeros out detector pixels that
    # exceed well capacity. A star is rejected if its central
    # ``saturation_core_size × saturation_core_size`` region (centred on
    # the star) contains any non-positive or NaN pixels. Set
    # ``saturation_core_size = 0`` to disable.
    saturation_core_size: int = 7

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
        if self.output_size is not None and self.output_size <= 0:
            return False, "output_size must be positive when set"
        return True, None

    @property
    def effective_output_size(self) -> Optional[int]:
        """Force-odd the output_size (1024 → 1023). ``None`` if unset."""
        if self.output_size is None:
            return None
        return self.output_size if self.output_size % 2 == 1 else self.output_size - 1


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
        # Per-run rejection counts populated by extract_psf_stars_from_files.
        self.n_rejected_saturated: int = 0
        self.n_rejected_edge:      int = 0
        self.n_rejected_load:      int = 0

    def get_cutout_files(
        self,
        cutout_dir: str,
        cutout_size: Optional[int] = None,
    ) -> List[Tuple[int, str]]:
        """
        Get FITS cutout files from directory, sorted by index.

        Parameters:
        -----------
        cutout_dir : str
            Directory containing cutout files.
        cutout_size : int, optional
            If given, only return files whose filename encodes this cutout
            size (``star_XXXX_<size>.fits``). PSF extraction must run on a
            single cutout size since the assumed image dimensions feed into
            the centered-crop math in :meth:`extract_psf_star_from_cutout`.

        Returns:
        --------
        list of tuple
            List of (index, filepath) tuples sorted by index.
        """
        if cutout_size is not None:
            pattern = f"star_[0-9][0-9][0-9][0-9]_{cutout_size}.fits"
        else:
            pattern = "*.fits"
        fits_files = glob.glob(os.path.join(cutout_dir, pattern))

        file_info = []
        for filepath in fits_files:
            filename = os.path.basename(filepath)
            try:
                # Filename: "star_XXXX_SIZE.fits"
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

        Assumes the star is centered in the image. Stars whose central
        ``saturation_core_size × saturation_core_size`` region contains
        any non-positive or NaN pixels are rejected — those are clipped
        saturated cores from the Euclid MER pipeline. The rejection
        leaves the bright wings intact and only avoids using stars where
        the very peak is unreliable.

        Returns ``None`` for any rejection (saturation, edge, etc.).
        Note: ``self.n_rejected_*`` counters are NOT updated here — they
        are only kept consistent inside
        :meth:`extract_psf_stars_from_files`, which catalogs the cause.
        """
        psf_size = self.config.psf_size
        ny, nx = image_data.shape
        center_y, center_x = ny // 2, nx // 2

        # ---- Saturation check: bright NISP stars have zero/NaN central cores
        #      from the MER pipeline clipping pixels above well capacity.
        sat_n = int(self.config.saturation_core_size)
        if sat_n >= 1:
            sh = sat_n // 2
            core = image_data[
                max(0, center_y - sh): center_y + sh + 1,
                max(0, center_x - sh): center_x + sh + 1,
            ]
            if core.size == 0 or (~np.isfinite(core)).any() or (core <= 0).any():
                return None

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

    def extract_accepted_stars(
        self,
        cutout_files: List[Tuple[int, str]]
    ) -> List[Tuple[int, EPSFStar]]:
        """Extract accepted PSF stars, **keeping each star's index**.

        One load pass over the files. Returns ``[(index, EPSFStar)]`` for
        every star that passes the saturation + edge checks — the index
        (parsed from the ``star_NNNN_*`` filename) lets callers join the
        star back to its catalog RA/Dec for spatial clustering.

        Updates ``self.n_rejected_*`` counters with the cause of every
        rejection (load failure / edge crop / saturation). Saturation
        rejection in particular catches MER-clipped bright NISP cores.
        """
        self.n_rejected_load = 0
        self.n_rejected_edge = 0
        self.n_rejected_saturated = 0
        accepted: List[Tuple[int, EPSFStar]] = []

        iterator = tqdm(
            cutout_files,
            desc="Processing cutouts",
            disable=not self.config.progress_bar
        )

        sat_n = int(self.config.saturation_core_size)

        def _is_saturated_core(image_data: np.ndarray) -> bool:
            if sat_n < 1:
                return False
            ny, nx = image_data.shape
            cy, cx = ny // 2, nx // 2
            sh = sat_n // 2
            core = image_data[max(0, cy - sh): cy + sh + 1,
                              max(0, cx - sh): cx + sh + 1]
            if core.size == 0:
                return True
            return bool((~np.isfinite(core)).any() or (core <= 0).any())

        for index, filepath in iterator:
            try:
                image_data = self.load_cutout(filepath)
                if image_data is None:
                    self.n_rejected_load += 1
                    print(f"  Warning: Could not load {filepath}")
                    continue

                # Pre-check saturation so we can attribute the rejection.
                if _is_saturated_core(image_data):
                    self.n_rejected_saturated += 1
                    continue

                epsf_star = self.extract_psf_star_from_cutout(image_data)
                if epsf_star is None:
                    # Wasn't saturated (checked above) → edge crop failure.
                    self.n_rejected_edge += 1
                    continue

                accepted.append((index, epsf_star))

            except Exception as e:
                self.n_rejected_load += 1
                print(f"  Warning: Error processing {filepath}: {e}")
                continue

        if self.n_rejected_saturated:
            print(f"  rejected {self.n_rejected_saturated} stars with clipped "
                  f"({sat_n}×{sat_n}) saturated cores")
        if self.n_rejected_edge:
            print(f"  rejected {self.n_rejected_edge} stars (edge crop failure)")
        if self.n_rejected_load:
            print(f"  rejected {self.n_rejected_load} stars (FITS load error)")

        return accepted

    def extract_psf_stars_from_files(
        self,
        cutout_files: List[Tuple[int, str]]
    ) -> List[EPSFStar]:
        """Accepted PSF stars without their indices (thin wrapper around
        :meth:`extract_accepted_stars` for callers that don't cluster)."""
        return [star for _, star in self.extract_accepted_stars(cutout_files)]

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
        return self.build_epsf_from_stars(all_epsf_stars)

    def build_epsf_from_stars(
        self,
        epsf_stars: List[EPSFStar],
    ) -> Tuple[EPSFModel, EPSFStars]:
        """Run EPSFBuilder on already-extracted ``EPSFStar`` objects.

        Split out of :meth:`build_epsf` so the per-cluster extraction
        path (cluster the accepted stars, then build one ePSF per group)
        can reuse the builder step without re-loading FITS. Sets
        ``self.epsf`` / ``self.fitted_stars`` to the most recent build.
        """
        if len(epsf_stars) == 0:
            raise ValueError("No valid PSF stars extracted from cutouts")

        print(f"Extracted {len(epsf_stars)} PSF stars from cutouts")

        # Build ePSF
        epsf_star_container = EPSFStars(list(epsf_stars))

        print("Building effective PSF...")
        builder_kwargs = dict(
            oversampling=self.config.oversampling,
            maxiters=self.config.max_iters,
            progress_bar=self.config.progress_bar,
            center_accuracy=self.config.accuracy,
            smoothing_kernel=None,
        )
        out_side = self.config.effective_output_size
        if out_side is not None:
            # Photutils accepts ``shape=(H, W)`` in *oversampled* pixels.
            builder_kwargs["shape"] = (out_side, out_side)
            print(f"  requesting output PSF shape {out_side}×{out_side}")
        epsf_builder = EPSFBuilder(**builder_kwargs)

        epsf, fitted_stars = epsf_builder(epsf_star_container)

        self.epsf = epsf
        self.fitted_stars = fitted_stars

        return epsf, fitted_stars

    @staticmethod
    def psf_from_epsf(epsf: EPSFModel, pixel_scale: float) -> PSF:
        """Wrap an ``EPSFModel`` in a typed :class:`PSF` (FWHM measured).

        Parametrised on ``epsf`` so the per-cluster loop can wrap each
        cluster's model; :meth:`to_psf` uses ``self.epsf``.

        ``pixel_scale`` is the oversampled-grid scale (native /
        oversampling) — e.g. 0.10/4 = 0.025″/pix for VIS at ovs=4.
        """
        oversamp_val = (
            epsf.oversampling[0]
            if hasattr(epsf.oversampling, '__iter__')
            else epsf.oversampling
        )
        psf = PSF(
            data=epsf.data.astype(np.float32),
            pixel_scale=pixel_scale,
            oversampling=int(oversamp_val),
        )
        psf.fwhm_arcsec = psf.fwhm_pixels() * pixel_scale
        return psf

    def to_psf(self, pixel_scale: float) -> PSF:
        """Wrap the most-recently-built ePSF (``self.epsf``) in a PSF."""
        if self.epsf is None:
            raise ValueError("No PSF has been built yet. Call build_epsf() first.")
        return self.psf_from_epsf(self.epsf, pixel_scale)

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
