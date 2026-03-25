
#!/usr/bin/env python3
"""
Generate Euclid-like HR/LR image pairs using GalSim + euclidlike.

Dataset semantics
-----------------
HR:
    Clean target image on the high-resolution grid (parametric COSMOS galaxies).
    No PSF, no detector noise.

LR:
    Euclid-like observed image on the low-resolution grid.
    Includes Euclid PSF, sky background, Poisson noise, and read noise.
    Use --no-noise to disable noise (LR will have only PSF convolution).

Output format
-------------
This script keeps the image-based pipeline layout:

    POLISH_train_HR/*.png
    POLISH_train_LR_bicubic/X{rebin}/*x{rebin}.png
    POLISH_valid_HR/*.png
    POLISH_valid_LR_bicubic/X{rebin}/*x{rebin}.png

It also saves:
    psf/*.npy
    objparams/*.json

Important
---------
- euclidlike is mandatory.
- --psf-dir is mandatory.
- GalSim COSMOSCatalog is mandatory (uses parametric galaxies).
- --catalog may be either:
    * the full path to real_galaxy_catalog_*.fits
    * the directory containing that file
- No per-image normalization is applied.
- Float images are converted to PNG using one fixed global scale factor.
- Use --no-noise to generate clean LR images (PSF only, no sky/detector noise).
- Use --flux-boost to increase galaxy/star brightness (e.g., 100 for ML-friendly data).
- Use --percentile-clip to handle outliers (e.g., 99.5 clips top/bottom 0.5%).
"""

import argparse
import glob
import json
import os
import sys
import warnings

import cv2
import galsim
import numpy as np
from astropy.io import fits
from scipy import signal

import euclidlike


VALID_BANDS = ("VIS", "NISP_H", "NISP_J", "NISP_Y")
DEFAULT_GAL_DENSITY_ARCMIN2 = 40.0
DEFAULT_STAR_DENSITY_ARCMIN2 = 2.0


def readfits(fnfits):
    """Read a FITS file and return image, header, pixel_scale, num_pix."""
    with fits.open(fnfits) as hdulist:
        data = hdulist[0].data
        header = hdulist[0].header

    dshape = data.shape
    if len(dshape) == 2:
        image = data
    elif len(dshape) == 3:
        image = data[0]
    elif len(dshape) == 4:
        image = data[0, 0]
    else:
        raise ValueError(f"Unsupported FITS dimensionality: {dshape}")

    pixel_scale = abs(header.get("CDELT1", 0.0))
    num_pix = abs(header.get("NAXIS1", image.shape[-1]))
    return image, header, pixel_scale, num_pix


def normalize_data(data, nbit=16, percentile_clip=None):
    """
    Normalize data to fit in bit range, convert to specified dtype.

    Parameters:
    -----------
    data : ndarray
        Input data to normalize
    nbit : int
        Number of bits for output (8 or 16)
    percentile_clip : float or None
        If specified, clip values at this percentile (e.g., 99.5) before scaling.
        This prevents outliers from dominating the normalization.
        Default: None (use min-max normalization)

    Returns:
    --------
    ndarray
        Normalized data as uint8 or uint16
    """
    data = np.asarray(data, dtype=np.float64)

    if percentile_clip is not None:
        # Percentile clipping: ignore extreme outliers
        vmin = np.percentile(data, 100 - percentile_clip)
        vmax = np.percentile(data, percentile_clip)
        data = np.clip(data, vmin, vmax)
        data = data - vmin
        dmax = vmax - vmin
    else:
        # Min-max normalization (original behavior)
        data = data - data.min()
        dmax = data.max()

    if dmax > 0:
        data = data / dmax
    data *= (2**nbit - 1)

    if nbit == 16:
        return data.astype(np.uint16)
    elif nbit == 8:
        return data.astype(np.uint8)
    raise ValueError(f"Unsupported nbit={nbit}. Use 8 or 16.")


def resolve_cosmos_catalog(catalog_arg):
    """
    Resolve a GalSim COSMOSCatalog from either:
    - a FITS catalog file path
    - a directory containing real_galaxy_catalog_*.fits

    Returns
    -------
    catalog : galsim.COSMOSCatalog
    catalog_file : str
    catalog_dir : str
    """
    if catalog_arg is None:
        raise ValueError("--catalog is required.")

    catalog_arg = os.path.abspath(catalog_arg)

    if os.path.isfile(catalog_arg):
        catalog_file = catalog_arg
        catalog_dir = os.path.dirname(catalog_file)
        return (
            galsim.COSMOSCatalog(
                file_name=os.path.basename(catalog_file),
                dir=catalog_dir,
            ),
            catalog_file,
            catalog_dir,
        )

    if os.path.isdir(catalog_arg):
        matches = sorted(
            m for m in glob.glob(os.path.join(catalog_arg, "real_galaxy_catalog_*.fits"))
            if "_fits.fits" not in os.path.basename(m)
            and "_selection.fits" not in os.path.basename(m)
        )

        if not matches:
            raise FileNotFoundError(
                f"No real_galaxy_catalog_*.fits found inside directory: {catalog_arg}"
            )

        if len(matches) > 1:
            print("Found multiple COSMOS catalogs. Using:", matches[0])

        catalog_file = matches[0]
        return (
            galsim.COSMOSCatalog(
                file_name=os.path.basename(catalog_file),
                dir=catalog_arg,
            ),
            catalog_file,
            catalog_arg,
        )

    raise FileNotFoundError(f"Catalog path does not exist: {catalog_arg}")


class EuclidSimulator:
    def __init__(
        self,
        image_size=2048,
        pixel_scale_hr=0.1,
        pixel_scale_lr=0.4,
        band="VIS",
        psf_dir=None,
        rng=None,
        gal_density_arcmin2=DEFAULT_GAL_DENSITY_ARCMIN2,
        star_density_arcmin2=DEFAULT_STAR_DENSITY_ARCMIN2,
        flux_boost=1.0,
    ):
        self.image_size = int(image_size)
        self.pixel_scale_hr = float(pixel_scale_hr)
        self.pixel_scale_lr = float(pixel_scale_lr)
        self.band = str(band).upper()
        self.psf_dir = psf_dir
        self.gal_density_arcmin2 = float(gal_density_arcmin2)
        self.star_density_arcmin2 = float(star_density_arcmin2)
        self.flux_boost = float(flux_boost)
        self.gsparams = galsim.GSParams(
            maximum_fft_size=16384,
            folding_threshold=1e-2,
            maxk_threshold=1e-2,
        )

        if self.band not in VALID_BANDS:
            raise ValueError(f"Invalid band '{self.band}'. Must be one of {VALID_BANDS}.")
        if self.pixel_scale_hr <= 0 or self.pixel_scale_lr <= 0:
            raise ValueError("Pixel scales must be positive.")
        if not self.psf_dir:
            raise ValueError("psf_dir is required.")
        if not os.path.isdir(self.psf_dir):
            raise FileNotFoundError(f"PSF directory does not exist: {self.psf_dir}")
        if self.gal_density_arcmin2 < 0 or self.star_density_arcmin2 < 0:
            raise ValueError("Source densities must be non-negative.")

        ratio = self.pixel_scale_lr / self.pixel_scale_hr
        rounded_ratio = round(ratio)
        if not np.isclose(ratio, rounded_ratio, rtol=0, atol=1e-8) or rounded_ratio < 1:
            raise ValueError(
                "pixel_scale_lr / pixel_scale_hr must be a positive integer. "
                f"Got {ratio}."
            )
        self.rebin_factor = int(rounded_ratio)

        if rng is None:
            rng = galsim.BaseDeviate()
        self.rng = rng
        self.ud = galsim.UniformDeviate(self.rng)

        self.euclid_params = self._setup_euclid_parameters()

    def _setup_euclid_parameters(self):
        return {
            "VIS": {
                "euclid_bandpass": "VIS",
                "pixel_scale": 0.101,
                "gain": 3.0,
                "read_noise": 4.5,
                "dark_current": 0.02,
                "exp_time": 565.0,
                "sky_brightness": 22.1,
                "lambda_eff_m": 677e-9,
                "zeropoint": 26.0,
            },
            "NISP_H": {
                "euclid_bandpass": "H",
                "pixel_scale": 0.3,
                "gain": 1.0,
                "read_noise": 10.0,
                "dark_current": 0.02,
                "exp_time": 247.0,
                "sky_brightness": 21.5,
                "lambda_eff_m": 1830e-9,
                "zeropoint": 24.5,
            },
            "NISP_J": {
                "euclid_bandpass": "J",
                "pixel_scale": 0.3,
                "gain": 1.0,
                "read_noise": 10.0,
                "dark_current": 0.02,
                "exp_time": 247.0,
                "sky_brightness": 22.0,
                "lambda_eff_m": 1250e-9,
                "zeropoint": 24.5,
            },
            "NISP_Y": {
                "euclid_bandpass": "Y",
                "pixel_scale": 0.3,
                "gain": 1.0,
                "read_noise": 10.0,
                "dark_current": 0.02,
                "exp_time": 247.0,
                "sky_brightness": 22.3,
                "lambda_eff_m": 990e-9,
                "zeropoint": 24.5,
            },
        }

    def get_euclid_psf(self, position=None, wavelength_nm=None, ccd=0, wcs=None):
        params = self.euclid_params[self.band]

        if position is None:
            position = galsim.PositionD(
                x=euclidlike.n_pix_col / 2,
                y=euclidlike.n_pix_row / 2,
            )

        if wavelength_nm is None:
            wavelength_nm = params["lambda_eff_m"] * 1e9

        psf = euclidlike.getPSF(
            ccd=ccd,
            bandpass=params["euclid_bandpass"],
            ccd_pos=position,
            wavelength=wavelength_nm,
            psf_dir=self.psf_dir,
            wcs=wcs,
        )
        psf = psf.withGSParams(self.gsparams)
        return psf

    def get_euclid_psf_for_lr(self, position=None, wavelength_nm=None, ccd=0, wcs=None):
        """
        Get the Euclid PSF pre-rendered and re-wrapped as an InterpolatedImage
        at the LR pixel scale. This band-limits maxk to pi/pixel_scale_lr,
        keeping the FFT size tractable when convolving with galaxies.

        Strategy:
        1. Render the native oversampled PSF to an HR-pixel-scale image
           using _render_psf_numpy (scipy zoom, no GalSim FFT).
        2. Pixel-bin the HR image to the LR grid.
        3. Wrap as an InterpolatedImage whose maxk is bounded by the
           LR Nyquist frequency.
        """
        psf = self.get_euclid_psf(position, wavelength_nm, ccd, wcs)

        # Render at HR pixel scale (no GalSim FFT)
        hr_stamp_size = 64 * self.rebin_factor  # e.g. 128 HR pixels for rebin=2
        psf_hr_img = self._render_psf_numpy(psf, hr_stamp_size, self.pixel_scale_hr)
        arr_hr = psf_hr_img.array

        # Pixel-bin from HR to LR
        r = self.rebin_factor
        ny_lr = hr_stamp_size // r
        nx_lr = hr_stamp_size // r
        arr_lr = arr_hr.reshape(ny_lr, r, nx_lr, r).sum(axis=(1, 3))

        # Normalize to unit flux
        total = arr_lr.sum()
        if total > 0:
            arr_lr = arr_lr / total

        psf_lr_img = galsim.ImageF(arr_lr.astype(np.float32), scale=self.pixel_scale_lr)

        # Wrap as InterpolatedImage — maxk is now limited by LR Nyquist.
        psf_ii = galsim.InterpolatedImage(
            psf_lr_img,
            x_interpolant='lanczos15',
            gsparams=self.gsparams,
        )
        return psf_ii

    def get_cosmos_galaxy(self, catalog=None, index=None):
        if catalog is None:
            raise ValueError("No COSMOSCatalog provided.")

        if not isinstance(catalog, galsim.COSMOSCatalog):
            raise TypeError(
                "catalog must be a galsim.COSMOSCatalog object. "
                "Load it once before simulation."
            )

        if catalog.nobjects < 1:
            raise ValueError("COSMOSCatalog is empty.")

        if index is None:
            index = int(self.ud() * catalog.nobjects)

        gal = catalog.makeGalaxy(
            index=index,
            gal_type="parametric",
            rng=self.rng,
            gsparams=self.gsparams,
        )

        # Apply flux boost if requested
        if self.flux_boost != 1.0:
            original_flux = gal.flux
            gal = gal.withFlux(original_flux * self.flux_boost)

        return gal, index

    def _sky_level_to_counts(self, sky_mag):
        params = self.euclid_params[self.band]
        zp = params["zeropoint"]
        exp_time = params["exp_time"]
        return 10 ** (-0.4 * (sky_mag - zp)) * exp_time

    def _add_noise(self, image, pixel_scale):
        """
        Add Euclid-like sky background and detector noise.
        Intended for LR observations only.

        Returns the per-pixel sky level that was added (before noise).
        """
        params = self.euclid_params[self.band]
        sky_counts_per_arcsec2 = self._sky_level_to_counts(params["sky_brightness"])
        sky_level = sky_counts_per_arcsec2 * (pixel_scale ** 2)

        image += sky_level
        image.addNoise(galsim.PoissonNoise(self.rng))
        image.addNoise(galsim.GaussianNoise(self.rng, sigma=params["read_noise"]))
        return float(sky_level)

    def _random_position_hr(self):
        x_pix = self.ud() * (self.image_size - 1) + 1
        y_pix = self.ud() * (self.image_size - 1) + 1
        return float(x_pix), float(y_pix)

    def _hr_to_lr_position(self, x_hr, y_hr):
        x_lr = (x_hr - 0.5) / self.rebin_factor + 0.5
        y_lr = (y_hr - 0.5) / self.rebin_factor + 0.5
        return float(x_lr), float(y_lr)

    def _make_stamp_bounds(self, x_center, y_center, stamp_size, image_bounds):
        half = stamp_size // 2
        ix = int(round(x_center))
        iy = int(round(y_center))
        bounds = galsim.BoundsI(ix - half, ix + half, iy - half, iy + half)
        return bounds & image_bounds

    def _draw_profile_to_image(
        self,
        profile,
        image,
        x_center,
        y_center,
        stamp_size,
        pixel_scale,
        method="auto",
        n_photons=None,
    ):
        bounds = self._make_stamp_bounds(x_center, y_center, stamp_size, image.bounds)
        if not bounds.isDefined():
            return

        stamp = galsim.ImageF(bounds, scale=pixel_scale)
        true_cx = 0.5 * (bounds.xmin + bounds.xmax)
        true_cy = 0.5 * (bounds.ymin + bounds.ymax)
        offset = galsim.PositionD(x_center - true_cx, y_center - true_cy)

        kwargs = {
            "image": stamp,
            "method": method,
            "offset": offset,
            "add_to_image": False,
        }

        if n_photons is not None and method == "phot":
            kwargs["n_photons"] = n_photons
            kwargs["rng"] = self.rng

        profile.drawImage(**kwargs)

        image[bounds] += stamp

    def simulate_pair(self, gal=None, catalog=None, gal_index=None, noise=True, ccd=0):
        """
        Simulate one HR/LR pair.

        HR:
            Clean parametric target, no PSF, no noise.
        LR:
            Euclid PSF-blurred, no noise.
        """
        if gal is None:
            gal, gal_index = self.get_cosmos_galaxy(catalog, gal_index)
        else:
            if gal_index is None:
                raise ValueError("gal_index is required when gal is provided.")

        image_hr = galsim.ImageF(self.image_size, self.image_size, scale=self.pixel_scale_hr)
        image_lr = galsim.ImageF(
            self.image_size // self.rebin_factor,
            self.image_size // self.rebin_factor,
            scale=self.pixel_scale_lr,
        )

        x_hr = 0.5 * (self.image_size + 1)
        y_hr = 0.5 * (self.image_size + 1)
        x_lr, y_lr = self._hr_to_lr_position(x_hr, y_hr)

        gal_flux = float(getattr(gal, "flux", 1e4))

        self._draw_profile_to_image(
            gal,
            image_hr,
            x_hr,
            y_hr,
            stamp_size=min(96, self.image_size),
            pixel_scale=self.pixel_scale_hr,
            method="no_pixel",
        )

        psf_raw = self.get_euclid_psf(position=galsim.PositionD(x=x_hr, y=y_hr), ccd=ccd)
        psf_lr = self.get_euclid_psf_for_lr(position=galsim.PositionD(x=x_hr, y=y_hr), ccd=ccd)
        profile_lr = galsim.Convolve(
            gal,
            psf_lr,
            gsparams=self.gsparams,
        )
        self._draw_profile_to_image(
            profile_lr,
            image_lr,
            x_lr,
            y_lr,
            stamp_size=min(48, image_lr.array.shape[0]),
            pixel_scale=self.pixel_scale_lr,
            method="no_pixel",
        )

        psf_hr_image = self._render_psf_numpy(psf_raw, 64, self.pixel_scale_hr)

        if noise:
            self._add_noise(image_lr, self.pixel_scale_lr)

        return image_hr.array, image_lr.array, psf_hr_image.array

    def simulate_field(self, n_galaxies=None, n_stars=None, catalog=None, noise=True, ccd=0):
        """
        Simulate one field.

        HR:
            Clean intrinsic scene (parametric galaxies, no PSF).
        LR:
            Euclid-observed scene with PSF + noise.
        """
        area_arcmin2 = (self.image_size * self.pixel_scale_hr / 60.0) ** 2

        if n_galaxies is None:
            n_galaxies = int(np.random.poisson(self.gal_density_arcmin2 * area_arcmin2))
        if n_stars is None:
            n_stars = int(np.random.poisson(self.star_density_arcmin2 * area_arcmin2))

        image_hr = galsim.ImageF(self.image_size, self.image_size, scale=self.pixel_scale_hr)
        image_lr = galsim.ImageF(
            self.image_size // self.rebin_factor,
            self.image_size // self.rebin_factor,
            scale=self.pixel_scale_lr,
        )

        galaxy_params = []
        star_params = []

        hr_gal_stamp = min(96, self.image_size)
        lr_gal_stamp = min(48, image_lr.array.shape[0])
        hr_star_stamp = min(32, self.image_size)
        lr_star_stamp = min(32, image_lr.array.shape[0])

        for _ in range(n_galaxies):
            success = False
            last_error = None

            for _attempt in range(10):
                try:
                    x_hr, y_hr = self._random_position_hr()
                    x_lr, y_lr = self._hr_to_lr_position(x_hr, y_hr)

                    gal, gal_index = self.get_cosmos_galaxy(catalog)
                    pos = galsim.PositionD(x=x_hr, y=y_hr)
                    euclid_psf_lr = self.get_euclid_psf_for_lr(position=pos, ccd=ccd)

                    self._draw_profile_to_image(
                        gal,
                        image_hr,
                        x_hr,
                        y_hr,
                        stamp_size=hr_gal_stamp,
                        pixel_scale=self.pixel_scale_hr,
                        method="no_pixel",
                    )

                    profile_lr = galsim.Convolve(
                        gal,
                        euclid_psf_lr,
                        gsparams=self.gsparams,
                    )
                    self._draw_profile_to_image(
                        profile_lr,
                        image_lr,
                        x_lr,
                        y_lr,
                        stamp_size=lr_gal_stamp,
                        pixel_scale=self.pixel_scale_lr,
                        method="no_pixel",
                    )

                    galaxy_params.append(
                        {
                            "type": "galaxy",
                            "x_pix_hr": float(x_hr),
                            "y_pix_hr": float(y_hr),
                            "x_pix_lr": float(x_lr),
                            "y_pix_lr": float(y_lr),
                            "flux": float(getattr(gal, "flux", 1e4)),
                        }
                    )

                    success = True
                    break

                except (galsim.GalSimFFTSizeWarning, galsim.GalSimError, RuntimeError) as e:
                    last_error = e

            if not success:
                print(f"Skipping one galaxy after repeated draw failures: {last_error}")

        for _ in range(n_stars):
            x_hr, y_hr = self._random_position_hr()
            x_lr, y_lr = self._hr_to_lr_position(x_hr, y_hr)

            pos = galsim.PositionD(x=x_hr, y=y_hr)
            psf_lr = self.get_euclid_psf_for_lr(position=pos, ccd=ccd)

            u = self.ud()
            if u < 0.7:
                mag = 22.0 + 3.0 * self.ud()
            elif u < 0.95:
                mag = 18.0 + 4.0 * self.ud()
            else:
                mag = 16.0 + 2.0 * self.ud()

            zp = self.euclid_params[self.band]["zeropoint"]
            flux = 10 ** (-0.4 * (mag - zp))

            # Apply flux boost to stars as well
            flux *= self.flux_boost

            # HR: place a point source (delta function) at the nearest pixel.
            # No GalSim draw needed — just deposit flux directly.
            ix_hr = int(round(x_hr))
            iy_hr = int(round(y_hr))
            if image_hr.bounds.includes(ix_hr, iy_hr):
                image_hr[ix_hr, iy_hr] += flux

            # LR: Convolve(delta, psf) = psf * flux.
            # Draw the pre-rendered LR PSF scaled by star flux.
            star_lr = psf_lr.withFlux(flux)
            self._draw_profile_to_image(
                star_lr,
                image_lr,
                x_lr,
                y_lr,
                stamp_size=lr_star_stamp,
                pixel_scale=self.pixel_scale_lr,
                method="no_pixel",
            )

            star_params.append(
                {
                    "type": "star",
                    "x_pix_hr": float(x_hr),
                    "y_pix_hr": float(y_hr),
                    "x_pix_lr": float(x_lr),
                    "y_pix_lr": float(y_lr),
                    "mag": float(mag),
                    "flux": float(flux),
                }
            )

        sky_level = 0.0
        if noise:
            sky_level = self._add_noise(image_lr, self.pixel_scale_lr)

        return image_hr.array, image_lr.array, {
            "field_area_arcmin2": float(area_arcmin2),
            "galaxy_density_arcmin2": float(self.gal_density_arcmin2),
            "star_density_arcmin2": float(self.star_density_arcmin2),
            "n_galaxies": int(n_galaxies),
            "n_stars": int(n_stars),
            "sky_level_per_pixel": float(sky_level),
            "galaxies": galaxy_params,
            "stars": star_params,
        }

    def _render_psf_numpy(self, psf, size, pixel_scale):
        """
        Render a PSF InterpolatedImage to a numpy array at the given
        pixel_scale, without using GalSim's FFT machinery.
        Returns a GalSim ImageD.
        """
        from scipy.ndimage import zoom as scipy_zoom

        psf_native_img = psf.image
        psf_arr = np.array(psf_native_img.array, dtype=np.float64)
        try:
            native_scale = psf_native_img.scale
        except (AttributeError, TypeError):
            native_scale = psf_native_img.wcs.minLinearScale()

        zoom_factor = native_scale / pixel_scale
        arr_resampled = scipy_zoom(psf_arr, zoom_factor, order=3)

        cur_h, cur_w = arr_resampled.shape
        if cur_h >= size and cur_w >= size:
            cy, cx = cur_h // 2, cur_w // 2
            half = size // 2
            arr_out = arr_resampled[cy - half:cy + half, cx - half:cx + half]
        else:
            arr_out = np.zeros((size, size), dtype=np.float64)
            ph = min(cur_h, size)
            pw = min(cur_w, size)
            sy, sx = (size - ph) // 2, (size - pw) // 2
            oy, ox = (cur_h - ph) // 2, (cur_w - pw) // 2
            arr_out[sy:sy + ph, sx:sx + pw] = arr_resampled[oy:oy + ph, ox:ox + pw]

        # Normalize so total flux = 1
        total = arr_out.sum()
        if total > 0:
            arr_out /= total

        return galsim.ImageD(arr_out, scale=pixel_scale)

    def render_psf_image(self, size=64, ccd=0):
        psf = self.get_euclid_psf(ccd=ccd)
        psf_image = self._render_psf_numpy(psf, size, self.pixel_scale_hr)
        return psf_image.array


def create_LR_image_sim(
    nimages,
    kernel,
    fdirout=".",
    catalog=None,
    subset="train",
    nstart=0,
    rebin=4,
    pixel_scale_hr=0.1,
    image_size=2048,
    nbit=16,
    plotit=False,
    save_img=True,
    band="VIS",
    psf_dir=None,
    star_density_arcmin2=DEFAULT_STAR_DENSITY_ARCMIN2,
    gal_density_arcmin2=DEFAULT_GAL_DENSITY_ARCMIN2,
    noise=True,
    flux_boost=1.0,
    percentile_clip=None,
):
    """
    Create Euclid-like HR/LR pairs and save them as PNGs.

    HR:
        clean parametric target (no PSF, no noise)
    LR:
        blurred/noisy observation (Euclid PSF + sky + Poisson + read noise)
        If noise=False, LR includes only PSF convolution (no sky or detector noise)

    Both HR and LR are independently normalized to the full uint16 [0, 65535] range.

    Parameters:
    -----------
    flux_boost : float
        Boost factor for galaxy and star fluxes. Values > 1.0 make sources brighter.
        Useful for creating ML-friendly data with better visibility.
        Default: 1.0 (use original COSMOS fluxes).
    percentile_clip : float or None
        If specified (e.g., 99.5), clip values at this percentile before normalization.
        This prevents extreme outliers from dominating the normalization range.
        Default: None (use min-max normalization).
    """
    if subset not in ("train", "valid"):
        raise ValueError("subset must be 'train' or 'valid'.")
    if psf_dir is None:
        raise ValueError("psf_dir is required.")
    if catalog is None:
        raise ValueError("A loaded COSMOSCatalog is required.")

    fdiroutHR = os.path.join(fdirout, f"POLISH_{subset}_HR")
    fdiroutLR = os.path.join(fdirout, f"POLISH_{subset}_LR_bicubic", f"X{rebin}")
    fdiroutPSF = os.path.join(fdirout, "psf")
    fdiroutObjParams = os.path.join(fdirout, "objparams")

    for d in (fdiroutHR, fdiroutLR, fdiroutPSF, fdiroutObjParams):
        os.makedirs(d, exist_ok=True)

    sim = EuclidSimulator(
        image_size=image_size,
        pixel_scale_hr=pixel_scale_hr,
        pixel_scale_lr=pixel_scale_hr * rebin,
        band=band,
        psf_dir=psf_dir,
        gal_density_arcmin2=gal_density_arcmin2,
        star_density_arcmin2=star_density_arcmin2,
        flux_boost=flux_boost,
    )

    images_lr = []
    images_hr = []

    for ii in range(nimages):
        base = f"{ii + nstart:04d}"

        fnoutLR = os.path.join(fdiroutLR, base + f"x{rebin}.png")
        fnoutHR = os.path.join(fdiroutHR, base + ".png")
        fnoutPSF = os.path.join(fdiroutPSF, base + "-psf.npy")
        fnoutObjParams = os.path.join(fdiroutObjParams, base + "ObjParams.json")

        if os.path.isfile(fnoutLR):
            print(f"File exists, skipping {fnoutLR}")
            continue

        if ii % 10 == 0:
            print(f"Finished {ii}/{nimages}")

        data_hr, data_lr, obj_params = sim.simulate_field(
            catalog=catalog,
            noise=noise,
            ccd=0,
        )

        psf_array = sim.render_psf_image(size=64, ccd=0).astype(np.float32)
        np.save(fnoutPSF, psf_array)

        obj_params["nbit"] = int(nbit)
        obj_params["band"] = band
        obj_params["rebin"] = int(rebin)
        obj_params["catalog_nobjects"] = int(catalog.nobjects)
        obj_params["flux_boost"] = float(flux_boost)
        obj_params["percentile_clip"] = percentile_clip
        obj_params["hr_float_min"] = float(data_hr.min())
        obj_params["hr_float_max"] = float(data_hr.max())
        obj_params["hr_float_mean"] = float(data_hr.mean())
        obj_params["lr_float_min"] = float(data_lr.min())
        obj_params["lr_float_max"] = float(data_lr.max())
        obj_params["lr_float_mean"] = float(data_lr.mean())

        with open(fnoutObjParams, "w") as f:
            json.dump(obj_params, f, indent=2)

        if save_img:
            # ----------------------------------------------------------
            # Sky subtraction for LR before normalization.
            #
            # The LR image is dominated by sky background (~820 counts/
            # pixel for VIS).  If we min-max normalize the raw LR, the
            # sky noise fills the entire uint16 range and the galaxies
            # are invisible.  The network then learns to output black.
            #
            # Fix: subtract the known sky level from LR and clip
            # negatives.  Now the LR image contains galaxy signal +
            # positive noise residual, similar in character to the HR
            # image (galaxy signal only).  min-max normalization then
            # stretches the galaxy features to fill the uint16 range,
            # with noise as realistic texture on top.
            # ----------------------------------------------------------
            sky_level = obj_params.get("sky_level_per_pixel", 0.0)

            data_lr_skysub = np.clip(data_lr - sky_level, 0, None)

            data_hr_out = normalize_data(data_hr, nbit=nbit, percentile_clip=percentile_clip)
            data_lr_out = normalize_data(data_lr_skysub, nbit=nbit, percentile_clip=percentile_clip)

            print(f"  [{base}] HR float  : min={data_hr.min():.4g}  max={data_hr.max():.4g}  "
                  f"mean={data_hr.mean():.4g}")
            print(f"  [{base}] LR float  : min={data_lr.min():.4g}  max={data_lr.max():.4g}  "
                  f"mean={data_lr.mean():.4g}  sky={sky_level:.1f}")
            print(f"  [{base}] LR skysub : min={data_lr_skysub.min():.4g}  "
                  f"max={data_lr_skysub.max():.4g}")

            if nbit == 8:
                cv2.imwrite(fnoutHR, data_hr_out.astype(np.uint8))
                cv2.imwrite(fnoutLR, data_lr_out.astype(np.uint8))
            elif nbit == 16:
                cv2.imwrite(fnoutHR, data_hr_out.astype(np.uint16))
                cv2.imwrite(fnoutLR, data_lr_out.astype(np.uint16))

        images_hr.append(data_hr)
        images_lr.append(data_lr)

        if plotit:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(np.log10(np.clip(data_hr, 1e-12, None)), cmap="viridis")
            axes[0].set_title("HR clean target")
            axes[1].imshow(np.log10(np.clip(data_lr, 1e-12, None)), cmap="viridis")
            axes[1].set_title("LR observed")
            axes[2].imshow(np.log10(psf_array + 1e-12), cmap="viridis")
            axes[2].set_title("log10 PSF")
            plt.tight_layout()
            plt.show()

    return images_lr, images_hr


def convolvehr_euclid(data, kernel, plotit=False, rebin=4, norm=True, nbit=16, noise=True):
    """Compatibility helper."""
    data = np.asarray(data)
    kernel = np.asarray(kernel)

    if data.ndim not in (2, 3):
        raise ValueError(f"Unsupported data shape: {data.shape}")
    if kernel.ndim not in (2, 3):
        raise ValueError(f"Unsupported kernel shape: {kernel.shape}")

    if data.ndim == 3 and kernel.ndim == 2:
        kernel = kernel[..., None]

    if noise:
        data_noise = data + np.random.normal(0.0, 5.0, data.shape)
    else:
        data_noise = data.copy()

    dataLR = signal.fftconvolve(data_noise, kernel, mode="same")
    dataLR = dataLR[rebin // 2::rebin, rebin // 2::rebin]

    if norm:
        dataLR = normalize_data(dataLR, nbit=nbit)
        data_noise = normalize_data(data_noise, nbit=nbit)

    if plotit:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(dataLR if dataLR.ndim == 2 else dataLR[..., 0], cmap="afmhot")
        axes[0].set_title("Convolved")
        axes[1].imshow(
            data if data.ndim == 2 else data[..., 0],
            cmap="afmhot",
            vmax=np.max(data) * 0.1,
        )
        axes[1].set_title("True")
        axes[2].imshow(
            kernel if kernel.ndim == 2 else kernel[..., 0],
            cmap="Greys",
            vmax=np.max(kernel) * 0.35,
        )
        axes[2].set_title("Kernel / PSF")
        plt.tight_layout()
        plt.show()

    return dataLR, data_noise


def parse_args():
    parser = argparse.ArgumentParser(
        prog="generate_euclid_data.py",
        description="Generate Euclid-like image pairs using GalSim + euclidlike.",
    )

    parser.add_argument("-o", "--fdout", default="./euclid_data", help="output directory")
    parser.add_argument("-p", "--plotit", action="store_true", help="plot images during generation")
    parser.add_argument("-r", "--rebin", type=int, default=4, help="downsampling factor")
    parser.add_argument("-b", "--nbit", type=int, default=16, help="output bits: 8 or 16")
    parser.add_argument("--ntrain", type=int, default=100, help="number of training images")
    parser.add_argument("--nvalid", type=int, default=20, help="number of validation images")
    parser.add_argument(
        "--catalog",
        type=str,
        required=True,
        help="path to RealGalaxy catalog FITS file OR directory containing real_galaxy_catalog_*.fits",
    )
    parser.add_argument("--pix", dest="pixel_size", type=float, default=0.1, help="HR pixel scale in arcsec")
    parser.add_argument("--nside", type=int, default=2048, help="HR image size")
    parser.add_argument("--band", type=str, default="VIS", help="VIS, NISP_H, NISP_J, NISP_Y")
    parser.add_argument("--no-save", dest="save_img", action="store_false", default=True, help="do not save images")
    parser.add_argument("--psf-dir", type=str, required=True, help="directory with euclidlike PSF FITS files")
    parser.add_argument(
        "--star-density",
        type=float,
        default=DEFAULT_STAR_DENSITY_ARCMIN2,
        help=f"default {DEFAULT_STAR_DENSITY_ARCMIN2} stars/arcmin^2",
    )
    parser.add_argument(
        "--gal-density",
        type=float,
        default=DEFAULT_GAL_DENSITY_ARCMIN2,
        help=f"default {DEFAULT_GAL_DENSITY_ARCMIN2} galaxies/arcmin^2",
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="disable noise in LR images (generate clean LR with only PSF convolution)",
    )
    parser.add_argument(
        "--flux-boost",
        type=float,
        default=1.0,
        help="boost galaxy and star fluxes by this factor (default: 1.0, use e.g. 100 for brighter galaxies)",
    )
    parser.add_argument(
        "--percentile-clip",
        type=float,
        default=None,
        help="clip outliers at this percentile before normalization (e.g., 99.5). "
             "Prevents extreme outliers from dominating the dynamic range. "
             "Default: None (use min-max normalization)",
    )

    return parser.parse_args()


def validate_options(options):
    options.band = options.band.upper()

    if options.band not in VALID_BANDS:
        raise ValueError(f"Invalid band '{options.band}'. Must be one of {VALID_BANDS}.")
    if not os.path.isdir(options.psf_dir):
        raise FileNotFoundError(f"PSF directory does not exist: {options.psf_dir}")
    if options.rebin < 1:
        raise ValueError("--rebin must be >= 1.")
    if options.nbit not in (8, 16):
        raise ValueError("--nbit must be 8 or 16.")
    if options.pixel_size <= 0:
        raise ValueError("--pix must be positive.")
    if options.nside <= 0:
        raise ValueError("--nside must be positive.")
    if options.ntrain < 0 or options.nvalid < 0:
        raise ValueError("--ntrain and --nvalid must be non-negative.")
    if options.star_density < 0:
        raise ValueError("--star-density must be non-negative.")
    if options.gal_density < 0:
        raise ValueError("--gal-density must be non-negative.")
    if not os.path.exists(options.catalog):
        raise FileNotFoundError(f"Catalog path does not exist: {options.catalog}")


def main():
    options = parse_args()

    try:
        validate_options(options)
        cosmos_catalog, catalog_file, catalog_dir = resolve_cosmos_catalog(options.catalog)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Generating Euclid-like data with {options.band} band")
    print(f"Pixel scale: {options.pixel_size} arcsec")
    print(f"Downsampling factor: {options.rebin}")
    print(f"Image size: {options.nside}")
    print(f"PSF directory: {options.psf_dir}")
    print(f"Catalog file: {catalog_file}")
    print(f"Catalog directory: {catalog_dir}")
    print(f"Catalog objects: {cosmos_catalog.nobjects}")
    print(f"Output directory: {options.fdout}")
    print(f"Galaxy density: {options.gal_density} galaxies/arcmin^2")
    print(f"Star density: {options.star_density} stars/arcmin^2")
    print(f"Flux boost: {options.flux_boost}x")
    print(f"Percentile clipping: {options.percentile_clip if options.percentile_clip else 'None (min-max)'}")
    print(f"Noise enabled: {not options.no_noise}")
    print("Using euclidlike for PSF generation")
    print("Using parametric COSMOS galaxies")

    os.makedirs(options.fdout, exist_ok=True)

    print("\nGenerating training set...")
    create_LR_image_sim(
        nimages=options.ntrain,
        kernel=None,
        fdirout=options.fdout,
        catalog=cosmos_catalog,
        subset="train",
        nstart=0,
        rebin=options.rebin,
        pixel_scale_hr=options.pixel_size,
        image_size=options.nside,
        nbit=options.nbit,
        plotit=options.plotit,
        save_img=options.save_img,
        band=options.band,
        psf_dir=options.psf_dir,
        star_density_arcmin2=options.star_density,
        gal_density_arcmin2=options.gal_density,
        noise=not options.no_noise,
        flux_boost=options.flux_boost,
        percentile_clip=options.percentile_clip,
    )

    print("\nGenerating validation set...")
    create_LR_image_sim(
        nimages=options.nvalid,
        kernel=None,
        fdirout=options.fdout,
        catalog=cosmos_catalog,
        subset="valid",
        nstart=options.ntrain,
        rebin=options.rebin,
        pixel_scale_hr=options.pixel_size,
        image_size=options.nside,
        nbit=options.nbit,
        plotit=options.plotit,
        save_img=options.save_img,
        band=options.band,
        psf_dir=options.psf_dir,
        star_density_arcmin2=options.star_density,
        gal_density_arcmin2=options.gal_density,
        noise=not options.no_noise,
        flux_boost=options.flux_boost,
        percentile_clip=options.percentile_clip,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
