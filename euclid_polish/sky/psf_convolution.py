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
    rebin_factor: int    = Config.DEFAULT_REBIN_FACTOR
    add_noise: bool      = Config.DEFAULT_ADD_NOISE
    noise_fraction: float = Config.DEFAULT_NOISE_FRACTION  # fraction of per-image mean flux
    nbit: int            = Config.DEFAULT_NBIT

    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate configuration."""
        if self.rebin_factor <= 0:
            return False, "Rebin factor must be positive"
        if self.noise_fraction < 0:
            return False, "Noise fraction must be non-negative"
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
        Min-max normalize and cast to uint dtype.

        Matches polish-pub ``normalize_data``: subtract min, divide by max,
        scale to [0, 2^nbit - 1], cast to uint16 or uint8.
        """
        data = data - data.min()
        dmax = data.max()
        if dmax > 0:
            data = data / dmax
        data *= (2**nbit - 1)
        if nbit == 16:
            data = data.astype(np.uint16)
        elif nbit == 8:
            data = data.astype(np.uint8)
        return data

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
    ) -> Tuple[SkyImage, SkyImage, SkyImage]:
        """
        Produce normalized training pair, matching polish-pub's pipeline.

        Steps (matching ``convolvehr`` + ``create_LR_image`` in polish-pub):
          1. Add noise to raw HR
          2. FFT convolve at full resolution
          3. ``normalize_data`` full-res convolved → uint16
          4. Stride downsample (on uint16 data)
          5. ``normalize_data`` downsampled → uint16  (re-stretches range)
          6. ``normalize_data`` noise-free HR → uint16

        Parameters:
        -----------
        hr_image : SkyImage
            High-resolution input image (raw flux).
        psf : PSF
            PSF kernel. Must have the same pixel_scale as hr_image.

        Returns:
        --------
        lr_norm : SkyImage
            Normalized dirty LR (uint16-valued float32, [0, 65535]).
        hr_norm : SkyImage
            Normalized noise-free clean HR (uint16-valued float32, [0, 65535]).
        lr_raw : SkyImage
            Raw dirty LR before normalization (for inspection).

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

        nbit = self.config.nbit
        rebin = self.config.rebin_factor

        # 1. Add noise
        if self.config.add_noise:
            noise_std = self.config.noise_fraction * abs(hr_image.data.mean())
            hr_noisy = hr_image.data + np.random.normal(0, noise_std, hr_image.data.shape)
        else:
            hr_noisy = hr_image.data

        # 2. FFT convolve at full resolution
        lr_fullres = signal.fftconvolve(hr_noisy, psf.data, mode='same')

        # 3. Normalize full-res convolved → uint16
        lr_fullres = self.normalize_data(lr_fullres, nbit=nbit)

        # 4. Stride downsample (on uint16 data)
        lr_downsampled = self.downsample(lr_fullres)

        # Keep a raw copy before second normalization (for inspection)
        lr_raw_data = lr_downsampled.astype(np.float32)

        # 5. Normalize downsampled → uint16 (re-stretches after downsample)
        lr_final = self.normalize_data(lr_downsampled, nbit=nbit).astype(np.float32)

        # 6. Normalize noise-free HR → uint16
        hr_final = self.normalize_data(hr_image.data.copy(), nbit=nbit).astype(np.float32)

        lr_ps = hr_image.pixel_scale * rebin
        lr_norm = SkyImage(
            data=lr_final, pixel_scale=lr_ps,
            is_clean=False, index=hr_image.index, subset=hr_image.subset,
        )
        hr_norm = SkyImage(
            data=hr_final, pixel_scale=hr_image.pixel_scale,
            is_clean=True, index=hr_image.index, subset=hr_image.subset,
            metadata=hr_image.metadata,
        )
        lr_raw = SkyImage(
            data=lr_raw_data, pixel_scale=lr_ps,
            is_clean=False, index=hr_image.index, subset=hr_image.subset,
        )
        return lr_norm, hr_norm, lr_raw
