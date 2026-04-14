"""
PSF Convolution module for sky generation.

This module provides classes for convolving high-resolution images
with a PSF and downsampling to create dirty/low-resolution images.
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple
from dataclasses import dataclass

from euclid_polish.config import Config
from euclid_polish.sky.types import SkyImage
from euclid_polish.euclid.types import PSF


@dataclass
class ConvolutionConfig:
    """Configuration for PSF convolution."""
    rebin_factor: int  = Config.DEFAULT_REBIN_FACTOR
    add_noise: bool    = Config.DEFAULT_ADD_NOISE
    noise_std: float   = Config.DEFAULT_NOISE_STD
    normalize: bool    = Config.DEFAULT_NORMALIZE
    nbit: int          = Config.DEFAULT_NBIT

    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate configuration."""
        if self.rebin_factor <= 0:
            return False, "Rebin factor must be positive"
        if self.noise_std < 0:
            return False, "Noise std must be non-negative"
        if self.nbit not in (8, 16):
            return False, "nbit must be 8 or 16"
        return True, None


class PSFConvolution:
    """
    Handle PSF convolution and downsampling operations.

    This class provides methods to:
    - Convolve HR images with a PSF kernel
    - Add realistic noise
    - Downsample to create LR images
    """

    def __init__(self, config: Optional[ConvolutionConfig] = None):
        """
        Initialize the convolution handler.

        Parameters:
        -----------
        config : ConvolutionConfig, optional
            Convolution configuration. Uses defaults if not provided.
        """
        self.config = config or ConvolutionConfig()

    @staticmethod
    def normalize_data(data: np.ndarray, nbit: int = 16) -> np.ndarray:
        """
        Normalize data to fit in bit range and convert to specified dtype.

        Parameters:
        -----------
        data : ndarray
            Input data.
        nbit : int
            Number of bits (8 or 16).

        Returns:
        --------
        ndarray
            Normalized data.
        """
        data = data - data.min()
        data = data / data.max()
        data *= (2**nbit - 1)
        if nbit == 16:
            data = data.astype(np.uint16)
        elif nbit == 8:
            data = data.astype(np.uint8)
        return data

    def convolve_with_psf(
        self,
        data: np.ndarray,
        kernel: np.ndarray,
        add_noise: Optional[bool] = None,
        noise_std: Optional[float] = None,
    ) -> np.ndarray:
        """
        Convolve data with PSF kernel.

        Parameters:
        -----------
        data : ndarray
            High-resolution input data.
        kernel : ndarray
            PSF kernel for convolution.
        add_noise : bool, optional
            Whether to add Gaussian noise. Uses config default if not specified.
        noise_std : float, optional
            Standard deviation of noise. Uses config default if not specified.

        Returns:
        --------
        ndarray
            Convolved data (same shape as input).
        """
        add_noise = add_noise if add_noise is not None else self.config.add_noise
        noise_std = noise_std if noise_std is not None else self.config.noise_std

        # Add noise if requested
        if add_noise:
            data_noise = data + np.random.normal(0, noise_std, data.shape)
        else:
            data_noise = data

        # Convolve with kernel using FFT
        data_convolved = signal.fftconvolve(data_noise, kernel, mode='same')

        return data_convolved

    def downsample(
        self,
        data: np.ndarray,
        rebin_factor: Optional[int] = None,
        offset: int = 0,
    ) -> np.ndarray:
        """
        Downsample data by taking every Nth pixel.

        Parameters:
        -----------
        data : ndarray
            Input data.
        rebin_factor : int, optional
            Downsampling factor. Uses config default if not specified.
        offset : int
            Starting offset for downsampling (default: 0).

        Returns:
        --------
        ndarray
            Downsampled data.
        """
        rebin_factor = rebin_factor if rebin_factor is not None else self.config.rebin_factor
        return data[rebin_factor // 2 + offset::rebin_factor, rebin_factor // 2 + offset::rebin_factor]

    def process_hr_to_lr(
        self,
        hr_image: SkyImage,
        psf: PSF,
        normalize: Optional[bool] = None,
        nbit: Optional[int] = None,
    ) -> Tuple[SkyImage, SkyImage]:
        """
        Complete pipeline: convolve HR SkyImage with PSF, downsample to LR SkyImage.

        Parameters:
        -----------
        hr_image : SkyImage
            High-resolution input image.
        psf : PSF
            PSF kernel. Must have the same pixel_scale as hr_image.
        normalize : bool, optional
            Whether to normalize output. Uses config default if not specified.
        nbit : int, optional
            Number of bits for output. Uses config default if not specified.

        Returns:
        --------
        lr_image : SkyImage
            Low-resolution (dirty) image with pixel_scale = hr_image.pixel_scale * rebin_factor.
        hr_noisy : SkyImage
            HR image with noise added (before convolution).

        Raises:
        -------
        ValueError
            If hr_image.pixel_scale != psf.pixel_scale.
        """
        if abs(hr_image.pixel_scale - psf.pixel_scale) > 1e-4:
            raise ValueError(
                f"Pixel scale mismatch: image has {hr_image.pixel_scale} arcsec/pix "
                f"but PSF has {psf.pixel_scale} arcsec/pix. "
                "Resample the PSF to match the image pixel scale before convolving."
            )

        normalize = normalize if normalize is not None else self.config.normalize
        nbit = nbit if nbit is not None else self.config.nbit

        # Add noise and convolve
        hr_noisy_data = self._add_noise(hr_image.data) if self.config.add_noise else hr_image.data
        lr_full = signal.fftconvolve(hr_noisy_data, psf.data, mode='same')

        # Downsample
        lr_data = self.downsample(lr_full)

        # Normalize if requested
        if normalize:
            lr_data = self.normalize_data(lr_data, nbit=nbit)

        rebin = self.config.rebin_factor
        lr_image = SkyImage(
            data=lr_data,
            pixel_scale=hr_image.pixel_scale * rebin,
            is_clean=False,
            index=hr_image.index,
            subset=hr_image.subset,
        )
        hr_noisy = SkyImage(
            data=hr_noisy_data,
            pixel_scale=hr_image.pixel_scale,
            is_clean=True,
            index=hr_image.index,
            subset=hr_image.subset,
            metadata=hr_image.metadata,
        )
        return lr_image, hr_noisy

    def _add_noise(self, data: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to data."""
        return data + np.random.normal(0, self.config.noise_std, data.shape)

    @staticmethod
    def gaussian2d_kernel(
        size: int = 8,
        std: float = 1.0
    ) -> np.ndarray:
        """
        Create a 2D Gaussian kernel.

        Parameters:
        -----------
        size : int
            Kernel size (will be size x size).
        std : float
            Standard deviation of Gaussian.

        Returns:
        --------
        ndarray
            2D Gaussian kernel.
        """
        kernel1D = signal.windows.gaussian(size, std=std).reshape(size, 1)
        return np.outer(kernel1D, kernel1D)
