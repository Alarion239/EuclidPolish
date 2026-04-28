"""
PSF convolution + realistic Euclid VIS noise model.

Pipeline (HR is 0.05"/pix electrons over the stacked integration; LR is
0.10"/pix electrons per detector pixel):

    1. Convolve HR with PSF              → hr_e (HR plane, electrons)
    2. Sum-rebin 2×2                      → lr_signal_e (per LR pixel)
    3. Add sky + dark, apply Poisson,     → lr_e (per LR pixel, post-noise)
       subtract sky/dark, add Gaussian
       read noise scaled by sqrt(N_exp)

The output of ``process_hr_to_lr`` is **raw electrons (float32)**; the tanh
stretch lives inside the model (see ``training/models/wdsr.py``) so that
network input/output stays in physical units. The numpy ``tanh_stretch`` /
``inverse_tanh_stretch`` helpers below are kept for diagnostics only.
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
    """Configuration for PSF convolution and Euclid VIS noise injection."""
    rebin_factor:         int   = Config.DEFAULT_REBIN_FACTOR
    add_noise:            bool  = Config.DEFAULT_ADD_NOISE
    exposure_time_s:      float = Config.EXPOSURE_TIME_S
    n_exposures:          int   = Config.N_EXPOSURES
    read_noise_e:         float = Config.READ_NOISE_E
    dark_e_per_s_per_pix: float = Config.DARK_E_PER_S_PER_PIX
    sky_e_per_s_per_arcsec2: float = Config.SKY_E_PER_S_PER_ARCSEC2
    lr_pixel_scale_arcsec: float = Config.VIS_PIXEL_SCALE_ARCSEC

    def validate(self) -> tuple[bool, Optional[str]]:
        if self.rebin_factor <= 0:
            return False, "Rebin factor must be positive"
        if self.exposure_time_s <= 0:
            return False, "exposure_time_s must be positive"
        if self.n_exposures <= 0:
            return False, "n_exposures must be positive"
        if self.read_noise_e < 0:
            return False, "read_noise_e must be non-negative"
        if self.dark_e_per_s_per_pix < 0:
            return False, "dark_e_per_s_per_pix must be non-negative"
        if self.sky_e_per_s_per_arcsec2 < 0:
            return False, "sky_e_per_s_per_arcsec2 must be non-negative"
        if self.lr_pixel_scale_arcsec <= 0:
            return False, "lr_pixel_scale_arcsec must be positive"
        return True, None


class PSFConvolution:
    """PSF convolution, sum-rebin to detector grid, and VIS noise injection."""

    def __init__(self, config: Optional[ConvolutionConfig] = None):
        self.config = config or ConvolutionConfig()

    @staticmethod
    def sum_rebin(data: np.ndarray, rebin_factor: int) -> np.ndarray:
        """Sum-rebin a 2D array by an integer factor (photometrically correct).

        Each output pixel is the sum of the (rebin_factor × rebin_factor) HR
        sub-pixels it covers, so total electron count is conserved. Trims any
        trailing rows/cols that don't fit a whole bin.
        """
        r = int(rebin_factor)
        h, w = data.shape
        h_t, w_t = (h // r) * r, (w // r) * r
        view = data[:h_t, :w_t].reshape(h_t // r, r, w_t // r, r)
        return view.sum(axis=(1, 3))

    @staticmethod
    def tanh_stretch(data_e: np.ndarray, scale_e: float = Config.TANH_SCALE_E) -> np.ndarray:
        """Map electrons → float32 in (-1, 1) via fixed tanh stretch (numpy).

        The model uses the equivalent TF op as its first layer. This numpy
        version is kept for diagnostics and visualization.
        """
        return np.tanh(data_e.astype(np.float64) / scale_e).astype(np.float32)

    @staticmethod
    def inverse_tanh_stretch(y: np.ndarray, scale_e: float = Config.TANH_SCALE_E) -> np.ndarray:
        """Inverse of :func:`tanh_stretch` (numpy). Clips to (-1, 1) − ε."""
        eps = 1e-12
        y_clipped = np.clip(y.astype(np.float64), -1.0 + eps, 1.0 - eps)
        return (np.arctanh(y_clipped) * scale_e).astype(np.float32)

    def _apply_vis_noise(
        self,
        lr_signal_e: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply realistic Euclid VIS noise to a sky-subtracted electron image.

        For each LR pixel:
            λ      = max(signal + sky + dark, 0)
            obs    = Poisson(λ) − (sky + dark)         # sky-subtracted
            read   = N(0, read_noise · sqrt(N_exp))    # uncorrelated reads
            output = obs + read
        """
        cfg = self.config
        t_total = cfg.exposure_time_s * cfg.n_exposures
        pixel_area = cfg.lr_pixel_scale_arcsec ** 2
        sky_e = cfg.sky_e_per_s_per_arcsec2 * pixel_area * t_total
        dark_e = cfg.dark_e_per_s_per_pix * t_total

        lam = np.clip(lr_signal_e.astype(np.float64) + sky_e + dark_e, 0.0, None)
        observed = rng.poisson(lam).astype(np.float64) - (sky_e + dark_e)

        read_sigma = cfg.read_noise_e * np.sqrt(cfg.n_exposures)
        read = rng.normal(0.0, read_sigma, size=lr_signal_e.shape)

        return (observed + read).astype(np.float32)

    def process_hr_to_lr(
        self,
        hr_image: SkyImage,
        psf: PSF,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[SkyImage, SkyImage]:
        """Produce a (dirty LR, clean HR) pair, both in raw electrons (float32).

        Parameters
        ----------
        hr_image : SkyImage
            HR image whose pixel values represent expected electrons over the
            stacked integration (see ``Config.SIM_VIS_ZEROPOINT_E``).
        psf : PSF
            Kernel matched to ``hr_image.pixel_scale``.
        rng : np.random.Generator, optional
            Reproducible noise source. Defaults to ``np.random.default_rng()``.

        Returns
        -------
        lr : SkyImage  — dirty LR in raw electrons (float32, can be negative).
        hr : SkyImage  — clean HR in raw electrons (float32, ≥ 0).
        """
        if abs(hr_image.pixel_scale - psf.pixel_scale) > 1e-4:
            raise ValueError(
                f"Pixel scale mismatch: image has {hr_image.pixel_scale} arcsec/pix "
                f"but PSF has {psf.pixel_scale} arcsec/pix."
            )

        if rng is None:
            rng = np.random.default_rng()

        cfg = self.config

        # 1. Convolve HR with PSF (still electrons, HR plane).
        hr_e = signal.fftconvolve(hr_image.data, psf.data, mode="same").astype(np.float32)

        # 2. Sum-rebin to LR detector grid (electrons per LR pixel).
        lr_signal_e = self.sum_rebin(hr_e, cfg.rebin_factor)

        # 3. Apply Euclid VIS noise (Poisson + sky/dark + read).
        if cfg.add_noise:
            lr_e = self._apply_vis_noise(lr_signal_e, rng)
        else:
            lr_e = lr_signal_e.astype(np.float32)

        lr_ps = hr_image.pixel_scale * cfg.rebin_factor
        lr = SkyImage(
            data=lr_e, pixel_scale=lr_ps,
            is_clean=False, index=hr_image.index, subset=hr_image.subset,
        )
        hr = SkyImage(
            data=hr_image.data.astype(np.float32, copy=True),
            pixel_scale=hr_image.pixel_scale,
            is_clean=True, index=hr_image.index, subset=hr_image.subset,
            metadata=hr_image.metadata,
        )
        return lr, hr
