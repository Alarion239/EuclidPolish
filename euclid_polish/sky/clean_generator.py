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


# Global worker state for multiprocessing
_worker_state = {}


@dataclass
class GeneratorConfig:
    """Configuration for clean sky generation."""
    image_size: int             = Config.DEFAULT_IMAGE_SIZE
    pixel_scale: float          = Config.DEFAULT_PIXEL_SCALE
    gal_density_arcmin2: float  = Config.DEFAULT_GAL_DENSITY_ARCMIN2
    star_density_arcmin2: float = Config.DEFAULT_STAR_DENSITY_ARCMIN2

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
    ):
        self.image_size = int(image_size)
        self.pixel_scale = float(pixel_scale)
        self.gal_density_arcmin2 = float(gal_density_arcmin2)
        self.star_density_arcmin2 = float(star_density_arcmin2)
        self.gsparams = galsim.GSParams(
            maximum_fft_size=Config.GALSIM_MAX_FFT_SIZE,
            folding_threshold=Config.GALSIM_FOLDING_THRESHOLD,
            maxk_threshold=Config.GALSIM_MAXK_THRESHOLD,
        )

        if self.pixel_scale <= 0:
            raise ValueError("Pixel scale must be positive.")
        if self.gal_density_arcmin2 < 0 or self.star_density_arcmin2 < 0:
            raise ValueError("Source densities must be non-negative.")

        if rng is None:
            rng = galsim.BaseDeviate()
        self.rng = rng
        self.ud = galsim.UniformDeviate(self.rng)

    def get_cosmos_galaxy(self, catalog=None, index=None):
        """Get a COSMOS galaxy from the catalog."""
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

        return gal, index

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

    def simulate_field(self, catalog=None, n_galaxies=None, n_stars=None, np_rng=None):
        """
        Simulate one clean field with galaxies and stars.

        Parameters:
        -----------
        catalog : galsim.COSMOSCatalog
            COSMOS galaxy catalog.
        n_galaxies : int, optional
            Number of galaxies to generate. If None, sampled from Poisson distribution.
        n_stars : int, optional
            Number of stars to generate. If None, sampled from Poisson distribution.
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

        if n_galaxies is None:
            if np_rng is not None:
                n_galaxies = int(np_rng.poisson(self.gal_density_arcmin2 * area_arcmin2))
            else:
                n_galaxies = int(np.random.poisson(self.gal_density_arcmin2 * area_arcmin2))
        if n_stars is None:
            if np_rng is not None:
                n_stars = int(np_rng.poisson(self.star_density_arcmin2 * area_arcmin2))
            else:
                n_stars = int(np.random.poisson(self.star_density_arcmin2 * area_arcmin2))

        image_hr = galsim.ImageF(self.image_size, self.image_size, scale=self.pixel_scale)
        galaxy_params = []
        star_params = []

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

            flux = 10 ** (-0.4 * (mag - Config.DEFAULT_VIS_ZEROPOINT))

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

        return image_hr.array, {
            "field_area_arcmin2": float(area_arcmin2),
            "galaxy_density_arcmin2": float(self.gal_density_arcmin2),
            "star_density_arcmin2": float(self.star_density_arcmin2),
            "n_galaxies": int(n_galaxies),
            "n_stars": int(n_stars),
            "galaxies": galaxy_params,
            "stars": star_params,
        }


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
        output_dir: str,
        subset: str = "train",
        nimages: int = Config.DEFAULT_NIMAGES,
        nstart: int = 0,
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Generate clean HR galaxy and star images.

        Parameters:
        -----------
        catalog : galsim.COSMOSCatalog
            COSMOS galaxy catalog.
        output_dir : str
            Output directory.
        subset : str
            Either 'train' or 'valid'.
        nimages : int
            Number of images to generate.
        nstart : int
            Starting index for naming.

        Returns:
        --------
        images : list of ndarray
            Generated images.
        metadata : list of dict
            Metadata for each image.
        """
        if subset not in ("train", "valid"):
            raise ValueError("subset must be 'train' or 'valid'.")

        output_dir_data = os.path.join(output_dir, subset)
        os.makedirs(output_dir_data, exist_ok=True)

        sim = CleanGalaxySimulator(
            image_size=self.config.image_size,
            pixel_scale=self.config.pixel_scale,
            gal_density_arcmin2=self.config.gal_density_arcmin2,
            star_density_arcmin2=self.config.star_density_arcmin2,
        )

        images = []
        metadata = []

        for ii in tqdm(range(nimages), desc=f"Generating {subset}", unit="img", ncols=100):
            np_rng = np.random.default_rng(ii + nstart)

            data_hr, obj_params = sim.simulate_field(
                catalog=catalog,
                np_rng=np_rng,
            )

            images.append(data_hr)
            meta = obj_params.copy()
            meta["image_index"] = int(ii)
            meta["image_size"] = int(self.config.image_size)
            meta["pixel_scale"] = float(self.config.pixel_scale)
            metadata.append(meta)

        return images, metadata
