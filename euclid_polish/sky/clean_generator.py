"""
Clean sky generator using GalSim COSMOS catalog.

This module provides classes for generating clean high-resolution galaxy
and star images using the GalSim COSMOS catalog.
"""

import os
import glob

import galsim
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from euclid_polish.config import Config
from euclid_polish.sky.types import SkyImage


@dataclass
class GeneratorConfig:
    """Configuration for clean sky generation."""
    image_size: int              = Config.DEFAULT_IMAGE_SIZE
    pixel_scale: float           = Config.DEFAULT_PIXEL_SCALE
    gal_density_arcmin2: float   = Config.DEFAULT_GAL_DENSITY_ARCMIN2
    star_density_arcmin2: float  = Config.DEFAULT_STAR_DENSITY_ARCMIN2
    donut_density_arcmin2: float = Config.DEFAULT_DONUT_DENSITY_ARCMIN2

    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate configuration."""
        if self.image_size <= 0:
            return False, "Image size must be positive"
        if self.pixel_scale <= 0:
            return False, "Pixel scale must be positive"
        if self.gal_density_arcmin2 < 0:
            return False, "Galaxy density must be non-negative"
        if self.star_density_arcmin2 < 0:
            return False, "Star density must be non-negative"
        if self.donut_density_arcmin2 < 0:
            return False, "Donut density must be non-negative"
        return True, None


class CleanGalaxySimulator:
    """Simulator for clean high-resolution galaxy and star images."""

    def __init__(
        self,
        image_size=Config.DEFAULT_IMAGE_SIZE,
        pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        rng=None,
        gal_density_arcmin2=Config.DEFAULT_GAL_DENSITY_ARCMIN2,
        star_density_arcmin2=Config.DEFAULT_STAR_DENSITY_ARCMIN2,
        donut_density_arcmin2=Config.DEFAULT_DONUT_DENSITY_ARCMIN2,
    ):
        self.image_size = int(image_size)
        self.pixel_scale = float(pixel_scale)
        self.gal_density_arcmin2 = float(gal_density_arcmin2)
        self.star_density_arcmin2 = float(star_density_arcmin2)
        self.donut_density_arcmin2 = float(donut_density_arcmin2)
        self.gsparams = galsim.GSParams(
            maximum_fft_size=Config.GALSIM_MAX_FFT_SIZE,
            folding_threshold=Config.GALSIM_FOLDING_THRESHOLD,
            maxk_threshold=Config.GALSIM_MAXK_THRESHOLD,
        )

        if self.pixel_scale <= 0:
            raise ValueError("Pixel scale must be positive.")
        for d in (self.gal_density_arcmin2, self.star_density_arcmin2, self.donut_density_arcmin2):
            if d < 0:
                raise ValueError("Source densities must be non-negative.")

        if rng is None:
            rng = galsim.BaseDeviate()
        self.rng = rng
        self.ud = galsim.UniformDeviate(self.rng)

    def get_cosmos_galaxy(self, catalog=None, index=None):
        """Get a COSMOS galaxy and rescale its flux to Euclid VIS electrons.

        ``galsim.COSMOSCatalog.makeGalaxy(gal_type='parametric')`` returns a
        profile whose flux is in COSMOS HST F814W native counts — typical
        magnitude-25 galaxies come out with ``gal.flux ≈ 2`` (vastly below
        the Euclid noise floor). We rescale to the expected total electrons
        over the stacked VIS integration using the catalog's ``mag_auto``
        and ``Config.SIM_VIS_ZEROPOINT_E``. Treating F814W as VIS magnitude
        is the standard simulator approximation (the bandpasses overlap
        substantially); a colour correction can be folded in later.
        """
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

        record = catalog.getParametricRecord(index)
        mag_auto = float(record["mag_auto"])
        electrons_total = 10.0 ** (-0.4 * (mag_auto - Config.SIM_VIS_ZEROPOINT_E))
        gal = gal.withFlux(electrons_total)

        return gal, index

    def _make_donut_profile(self, np_rng):
        """Build one toy gravitational-lens ring as a galsim.GSObject.

        A Gaussian-thickness annulus is rendered onto a small numpy stamp,
        wrapped in galsim.InterpolatedImage, then sheared / rotated to give
        the ring an arbitrary orientation. Some realisations look like full
        rings, others like elongated arcs (depending on the random shear).
        Total flux is rescaled to the expected electrons over the stack via
        ``Config.SIM_VIS_ZEROPOINT_E`` — same calibration used for stars
        and COSMOS galaxies.

        Returns
        -------
        prof : galsim.GSObject
            Profile ready for ``_draw_profile_to_image``.
        params : dict
            Diagnostic record (radius, thickness, shear, mag, flux).
        """
        radius_arcsec = float(np_rng.uniform(
            Config.DONUT_RADIUS_ARCSEC_MIN, Config.DONUT_RADIUS_ARCSEC_MAX))
        sigma_arcsec  = Config.DONUT_THICKNESS_FRAC * radius_arcsec
        theta_deg     = float(np_rng.uniform(0.0, 180.0))
        g_mag         = float(np_rng.uniform(0.0, Config.DONUT_ELLIPTICITY_MAX))
        g_phase       = float(np_rng.uniform(0.0, 2.0 * np.pi))
        g1            = g_mag * np.cos(g_phase)
        g2            = g_mag * np.sin(g_phase)
        mag           = float(np_rng.uniform(Config.DONUT_MAG_MIN, Config.DONUT_MAG_MAX))
        electrons_total = 10.0 ** (-0.4 * (mag - Config.SIM_VIS_ZEROPOINT_E))

        # Build numpy ring on a centred stamp.
        N = int(Config.DONUT_STAMP_PIX)
        yy, xx = np.indices((N, N), dtype=np.float64) - (N - 1) / 2.0
        r_arcsec = np.sqrt(xx * xx + yy * yy) * self.pixel_scale
        ring = np.exp(-((r_arcsec - radius_arcsec) ** 2) / (2.0 * sigma_arcsec ** 2))

        img = galsim.Image(ring.astype(np.float32), scale=self.pixel_scale)
        prof = galsim.InterpolatedImage(img, gsparams=self.gsparams)
        prof = prof.shear(g1=g1, g2=g2).rotate(theta_deg * galsim.degrees)
        prof = prof.withFlux(electrons_total)

        params = {
            "type": "donut",
            "radius_arcsec": radius_arcsec,
            "sigma_arcsec": sigma_arcsec,
            "theta_deg": theta_deg,
            "g1": g1, "g2": g2,
            "mag": mag,
            "flux": float(electrons_total),
        }
        return prof, params

    def _random_position(self):
        """Generate random position within the image."""
        x_pix = self.ud() * (self.image_size - 1) + 1
        y_pix = self.ud() * (self.image_size - 1) + 1
        return float(x_pix), float(y_pix)

    def _make_stamp_bounds(self, x_center, y_center, stamp_size, image_bounds):
        """Create bounds for drawing a galaxy stamp."""
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
    ):
        """Draw a galaxy profile to an image at the specified position."""
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

        profile.drawImage(**kwargs)
        image[bounds] += stamp

    def simulate_field(self, catalog=None, n_galaxies=None, n_stars=None, n_donuts=None, np_rng=None):
        """
        Simulate one clean field with galaxies, stars, and donut-galaxies.

        Parameters:
        -----------
        catalog : galsim.COSMOSCatalog
            COSMOS galaxy catalog.
        n_galaxies : int, optional
            Number of galaxies to generate. If None, sampled from Poisson distribution.
        n_stars : int, optional
            Number of stars to generate. If None, sampled from Poisson distribution.
        n_donuts : int, optional
            Number of donut-galaxies (toy lensing rings/arcs). If None, sampled
            from Poisson distribution.
        np_rng : np.random.Generator, optional
            NumPy random number generator for reproducible sampling.

        Returns:
        --------
        data_hr : ndarray
            Clean high-resolution image.
        obj_params : dict
            Parameters of generated objects.
        """
        area_arcmin2 = (self.image_size * self.pixel_scale / 60.0) ** 2

        rng_for_poisson = np_rng if np_rng is not None else np.random
        if n_galaxies is None:
            n_galaxies = int(rng_for_poisson.poisson(self.gal_density_arcmin2 * area_arcmin2))
        if n_stars is None:
            n_stars = int(rng_for_poisson.poisson(self.star_density_arcmin2 * area_arcmin2))
        if n_donuts is None:
            n_donuts = int(rng_for_poisson.poisson(self.donut_density_arcmin2 * area_arcmin2))

        # Donuts need a NumPy Generator (uniform sampling of geometry); fall back
        # to a fresh default RNG if the caller didn't pass one.
        donut_rng = np_rng if np_rng is not None else np.random.default_rng()

        image_hr = galsim.ImageF(self.image_size, self.image_size, scale=self.pixel_scale)
        galaxy_params = []
        star_params = []
        donut_params = []

        for _ in range(n_galaxies):
            success = False
            last_error = None

            for _attempt in range(10):
                try:
                    x_hr, y_hr = self._random_position()
                    gal, gal_index = self.get_cosmos_galaxy(catalog)

                    hr_gal_stamp = 2 * self.image_size

                    self._draw_profile_to_image(
                        gal,
                        image_hr,
                        x_hr,
                        y_hr,
                        stamp_size=hr_gal_stamp,
                        pixel_scale=self.pixel_scale,
                        method="no_pixel",
                    )

                    galaxy_params.append(
                        {
                            "type": "galaxy",
                            "cosmos_index": int(gal_index),
                            "x_pix": float(x_hr),
                            "y_pix": float(y_hr),
                            "flux": float(getattr(gal, "flux", 1e4)),
                        }
                    )

                    success = True
                    break

                except (galsim.GalSimFFTSizeWarning, galsim.GalSimError, RuntimeError) as e:
                    last_error = e

            if not success:
                print(f"Skipping one galaxy after repeated draw failures: {last_error}")

        # Generate stars as point sources
        for _ in range(n_stars):
            x_hr, y_hr = self._random_position()

            # Sample stellar magnitudes
            u = self.ud()
            if u < Config.STAR_MAG_PROB_FAINT:
                mag = Config.STAR_MAG_FAINT_BASE   + Config.STAR_MAG_FAINT_RANGE   * self.ud()
            elif u < Config.STAR_MAG_PROB_MID:
                mag = Config.STAR_MAG_MID_BASE     + Config.STAR_MAG_MID_RANGE     * self.ud()
            else:
                mag = Config.STAR_MAG_BRIGHT_BASE  + Config.STAR_MAG_BRIGHT_RANGE  * self.ud()

            # Total expected electrons over the stacked Euclid VIS integration.
            flux = 10 ** (-0.4 * (mag - Config.SIM_VIS_ZEROPOINT_E))

            # HR: place a point source at the nearest pixel
            ix_hr = int(round(x_hr))
            iy_hr = int(round(y_hr))
            if image_hr.bounds.includes(ix_hr, iy_hr):
                image_hr[ix_hr, iy_hr] += flux

            star_params.append(
                {
                    "type": "star",
                    "x_pix": float(x_hr),
                    "y_pix": float(y_hr),
                    "mag": float(mag),
                    "flux": float(flux),
                }
            )

        # Donut-galaxies (toy gravitational-lens rings / arcs)
        for _ in range(n_donuts):
            success = False
            last_error = None

            for _attempt in range(5):
                try:
                    x_hr, y_hr = self._random_position()
                    prof, params = self._make_donut_profile(donut_rng)

                    self._draw_profile_to_image(
                        prof,
                        image_hr,
                        x_hr,
                        y_hr,
                        stamp_size=2 * self.image_size,
                        pixel_scale=self.pixel_scale,
                        method="no_pixel",
                    )

                    params["x_pix"] = float(x_hr)
                    params["y_pix"] = float(y_hr)
                    donut_params.append(params)

                    success = True
                    break

                except (galsim.GalSimFFTSizeWarning, galsim.GalSimError, RuntimeError) as e:
                    last_error = e

            if not success:
                print(f"Skipping one donut after repeated draw failures: {last_error}")

        obj_params = {
            "field_area_arcmin2": float(area_arcmin2),
            "galaxy_density_arcmin2": float(self.gal_density_arcmin2),
            "star_density_arcmin2": float(self.star_density_arcmin2),
            "donut_density_arcmin2": float(self.donut_density_arcmin2),
            "n_galaxies": int(n_galaxies),
            "n_stars": int(n_stars),
            "n_donuts": int(n_donuts),
            "galaxies": galaxy_params,
            "stars": star_params,
            "donuts": donut_params,
        }
        return SkyImage(
            data=image_hr.array,
            pixel_scale=self.pixel_scale,
            is_clean=True,
            metadata=obj_params,
        ), obj_params


class CleanSkyGenerator:
    """
    Generator for clean sky images using GalSim COSMOS catalog.

    This class handles:
    - Loading and resolving COSMOS catalogs
    - Generating clean HR images
    - Saving to TFRecord format
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        Initialize the generator.

        Parameters:
        -----------
        config : GeneratorConfig, optional
            Generator configuration. Uses defaults if not provided.
        """
        self.config = config or GeneratorConfig()

    @staticmethod
    def resolve_cosmos_catalog(catalog_arg: str) -> Tuple[galsim.COSMOSCatalog, str, str]:
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
            raise ValueError("Catalog path is required.")

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

    def generate(
        self,
        catalog: galsim.COSMOSCatalog,
        subset: str = "train",
        nimages: int = Config.DEFAULT_NIMAGES,
        nstart: int = 0,
    ) -> Tuple[List[SkyImage], List[Dict]]:
        """
        Generate clean HR galaxy and star images.

        Parameters:
        -----------
        catalog : galsim.COSMOSCatalog
            COSMOS galaxy catalog.
        subset : str
            Either 'train' or 'validate'.
        nimages : int
            Number of images to generate.
        nstart : int
            Starting index (used to seed RNG, avoids train/validate overlap).

        Returns:
        --------
        images : list of SkyImage
            Generated clean HR images with pixel_scale and metadata attached.
        metadata : list of dict
            Metadata for each image.
        """
        if subset not in ("train", "validate"):
            raise ValueError("subset must be 'train' or 'validate'.")

        sim = CleanGalaxySimulator(
            image_size=self.config.image_size,
            pixel_scale=self.config.pixel_scale,
            gal_density_arcmin2=self.config.gal_density_arcmin2,
            star_density_arcmin2=self.config.star_density_arcmin2,
            donut_density_arcmin2=self.config.donut_density_arcmin2,
        )

        images = []
        metadata = []

        for ii in tqdm(range(nimages), desc=f"Generating {subset}", unit="img", ncols=100):
            np_rng = np.random.default_rng(ii + nstart)

            sky_image, obj_params = sim.simulate_field(
                catalog=catalog,
                np_rng=np_rng,
            )
            sky_image.index = int(ii)
            sky_image.subset = subset

            images.append(sky_image)
            meta = obj_params.copy()
            meta["image_index"] = int(ii)
            meta["image_size"] = int(self.config.image_size)
            meta["pixel_scale"] = float(self.config.pixel_scale)
            metadata.append(meta)

        return images, metadata
