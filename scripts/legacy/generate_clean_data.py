#!/usr/bin/env python3
"""
Generate clean HR galaxy and star images using GalSim COSMOS catalog.

Dataset semantics
-----------------
HR:
    Clean target image on the high-resolution grid (parametric COSMOS galaxies + stars).
    No PSF, no noise, no detector effects.

Output format
-------------
This script saves data as TensorFlow TFRecord files in float16 format:

    output_dir/train/*.tfrecord
    output_dir/valid/*.tfrecord

Each TFRecord contains float16 image tensors with shape (height, width).

Visualization
-------------
Use --visualize to enable image visualization during generation.
Use --num-viz to control how many images to visualize.
Use --viz-mode to choose between 'random' or 'first' N images.
"""

import os
# Set thread limits BEFORE importing numerical libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import glob
import sys

import galsim
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tensorflow as tf

# Global worker state for multiprocessing
_worker_state = {}


DEFAULT_GAL_DENSITY_ARCMIN2 = 40.0
DEFAULT_STAR_DENSITY_ARCMIN2 = 2.0

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


def to_tfrecord_feature(value):
    """Convert numpy array to tf.train.Feature as bytes."""
    # Convert to float16 and store as bytes
    arr_flat = value.flatten().astype(np.float16)
    bytes_list = tf.train.BytesList(value=[arr_flat.tobytes()])
    return tf.train.Feature(bytes_list=bytes_list)


def serialize_example(image, index, height, width):
    """Create a tf.train.Example from image data."""
    feature = {
        'image': to_tfrecord_feature(image),  # Stored as bytes (float16)
        'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfrecord(images, indices, output_path, image_shape):
    """Write a batch of images to a TFRecord file."""
    with tf.io.TFRecordWriter(output_path) as writer:
        for img, idx in zip(images, indices):
            example = serialize_example(img, idx, image_shape[0], image_shape[1])
            writer.write(example)


def save_metadata(metadata_list, output_path):
    """Save metadata for all images to a JSON file."""
    import json
    with open(output_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)


class CleanGalaxySimulator:
    """Simulator for clean high-resolution galaxy and star images."""

    def __init__(
        self,
        image_size=2048,
        pixel_scale=0.05,
        rng=None,
        gal_density_arcmin2=DEFAULT_GAL_DENSITY_ARCMIN2,
        star_density_arcmin2=DEFAULT_STAR_DENSITY_ARCMIN2,
    ):
        self.image_size = int(image_size)
        self.pixel_scale = float(pixel_scale)
        self.gal_density_arcmin2 = float(gal_density_arcmin2)
        self.star_density_arcmin2 = float(star_density_arcmin2)
        self.gsparams = galsim.GSParams(
            maximum_fft_size=16384,
            folding_threshold=1e-4,
            maxk_threshold=1e-2,
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
        zeropoint = 26.2  # VIS-like zeropoint for magnitude to flux conversion

        for _ in range(n_stars):
            x_hr, y_hr = self._random_position()

            # Sample stellar magnitudes
            u = self.ud()
            if u < 0.7:
                mag = 22.0 + 3.0 * self.ud()
            elif u < 0.95:
                mag = 18.0 + 4.0 * self.ud()
            else:
                mag = 16.0 + 2.0 * self.ud()

            flux = 10 ** (-0.4 * (mag - zeropoint))

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


def _init_worker(catalog_file, catalog_dir, image_size, pixel_scale, gal_density_arcmin2, star_density_arcmin2):
    """Initialize worker state for multiprocessing."""
    global _worker_state

    rng = galsim.BaseDeviate()

    catalog = galsim.COSMOSCatalog(
        file_name=os.path.basename(catalog_file),
        dir=catalog_dir,
    )

    sim = CleanGalaxySimulator(
        image_size=image_size,
        pixel_scale=pixel_scale,
        rng=rng,
        gal_density_arcmin2=gal_density_arcmin2,
        star_density_arcmin2=star_density_arcmin2,
    )

    _worker_state["catalog"] = catalog
    _worker_state["sim"] = sim


def _is_tfrecord_complete(fnout):
    """Check if a TFRecord file exists."""
    return os.path.isfile(fnout)


def _generate_single_image(task):
    """
    Worker function for parallel image generation.
    Processes one image using pre-initialized catalog and simulator.
    """
    global _worker_state

    ii, nstart, subset, base_seed = task

    sim = _worker_state["sim"]
    catalog = _worker_state["catalog"]

    seed = int((base_seed + ii) % (2**32))
    sim.rng = galsim.BaseDeviate(seed)
    sim.ud = galsim.UniformDeviate(sim.rng)
    np_rng = np.random.default_rng(seed)

    try:
        data_hr, obj_params = sim.simulate_field(
            catalog=catalog,
            np_rng=np_rng,
        )

        return {"status": "success", "index": ii, "data": data_hr, "params": obj_params}

    except Exception as e:
        return {"status": "error", "index": ii, "error": str(e)}


def generate_clean_data(
    nimages,
    fdirout=".",
    catalog=None,
    catalog_file=None,
    catalog_dir=None,
    subset="train",
    nstart=0,
    pixel_scale=0.1,
    image_size=2048,
    gal_density_arcmin2=DEFAULT_GAL_DENSITY_ARCMIN2,
    star_density_arcmin2=DEFAULT_STAR_DENSITY_ARCMIN2,
    nproc=1,
):
    """
    Generate clean HR galaxy and star images and save as TFRecord files.

    Parameters:
    -----------
    nimages : int
        Number of images to generate.
    fdirout : str
        Output directory.
    catalog : galsim.COSMOSCatalog, optional
        Pre-loaded COSMOS catalog (required for nproc=1).
    catalog_file : str, optional
        Path to COSMOS catalog FITS file (required for nproc>1).
    catalog_dir : str, optional
        Directory containing the catalog (required for nproc>1).
    subset : str
        Either 'train' or 'valid'.
    nstart : int
        Starting index for naming.
    pixel_scale : float
        Pixel scale in arcseconds.
    image_size : int
        Image size in pixels.
    gal_density_arcmin2 : float
        Galaxy density per square arcminute.
    star_density_arcmin2 : float
        Star density per square arcminute.
    nproc : int
        Number of parallel processes (1 for serial, 0/-1 for all CPUs).
    """
    if subset not in ("train", "valid"):
        raise ValueError("subset must be 'train' or 'valid'.")

    # Validate catalog requirements based on execution mode
    n_available = cpu_count()
    if nproc == 0 or nproc == -1:
        nproc = min(n_available, 8)
    elif nproc < 0:
        raise ValueError(f"Invalid nproc={nproc}. Use 0, -1, or positive integer.")
    else:
        nproc = min(nproc, 8)

    if nproc > 1:
        if catalog_file is None or catalog_dir is None:
            raise ValueError("catalog_file and catalog_dir are required for parallel mode (nproc > 1)")
    else:
        if catalog is None:
            raise ValueError("catalog (COSMOSCatalog object) is required for serial mode (nproc=1)")

    fdirout_data = os.path.join(fdirout, subset)
    os.makedirs(fdirout_data, exist_ok=True)

    if nproc > 1:
        import time
        base_seed = int(time.time() * 1000) % (2**32)
        tasks = [(ii, nstart, subset, base_seed) for ii in range(nimages)]

        tqdm.write(f"Using {nproc} workers (cap applied; host has {n_available} CPUs).")
        tqdm.write(f"Generating {nimages} {subset} images...")

        results = []
        with Pool(
            processes=nproc,
            initializer=_init_worker,
            initargs=(catalog_file, catalog_dir, image_size, pixel_scale, gal_density_arcmin2, star_density_arcmin2),
        ) as pool:
            with tqdm(total=nimages, desc=f"Generating {subset}", unit="img", ncols=100) as pbar:
                for result in pool.imap_unordered(_generate_single_image, tasks, chunksize=1):
                    results.append(result)
                    pbar.update(1)

        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")

        tqdm.write("\nGeneration complete:")
        tqdm.write(f"  Success: {success_count}")
        tqdm.write(f"  Errors: {error_count}")

        if error_count > 0:
            tqdm.write("\nErrors encountered:")
            for r in results:
                if r["status"] == "error":
                    tqdm.write(f"  Image {r['index']}: {r.get('error', 'Unknown error')}")

        # Collect results
        images = []
        indices = []
        metadata = []
        for r in results:
            if r["status"] == "success":
                images.append(r["data"])
                indices.append(r["index"])
                # Add index and image parameters to metadata
                meta = r["params"].copy()
                meta["image_index"] = int(r["index"])
                meta["image_size"] = int(image_size)
                meta["pixel_scale"] = float(pixel_scale)
                metadata.append(meta)

    else:
        # Serial execution
        sim = CleanGalaxySimulator(
            image_size=image_size,
            pixel_scale=pixel_scale,
            gal_density_arcmin2=gal_density_arcmin2,
            star_density_arcmin2=star_density_arcmin2,
        )

        images = []
        indices = []
        metadata = []

        for ii in tqdm(range(nimages), desc=f"Generating {subset}", unit="img", ncols=100):
            np_rng = np.random.default_rng(ii + nstart)

            data_hr, obj_params = sim.simulate_field(
                catalog=catalog,
                np_rng=np_rng,
            )

            images.append(data_hr)
            indices.append(ii)
            # Add index and image parameters to metadata
            meta = obj_params.copy()
            meta["image_index"] = int(ii)
            meta["image_size"] = int(image_size)
            meta["pixel_scale"] = float(pixel_scale)
            metadata.append(meta)

    # Write to TFRecord
    print(f"\nWriting {len(images)} images to TFRecord...")
    output_file = os.path.join(fdirout_data, f"{subset}_clean.tfrecord")
    write_tfrecord(images, indices, output_file, (image_size, image_size))
    print(f"Saved to {output_file}")

    # Write metadata to JSON
    print(f"\nWriting metadata to JSON...")
    metadata_file = os.path.join(fdirout_data, f"{subset}_clean_metadata.json")
    save_metadata(metadata, metadata_file)
    print(f"Saved metadata to {metadata_file}")

    return images


def parse_args():
    parser = argparse.ArgumentParser(
        prog="generate_clean_data.py",
        description="Generate clean HR galaxy images using GalSim COSMOS catalog.",
    )

    parser.add_argument("-o", "--fdout", default="./clean_data", help="output directory")
    parser.add_argument("--pix", dest="pixel_scale", type=float, default=0.05,
                       help="HR pixel scale in arcsec")
    parser.add_argument("--nside", type=int, default=256, help="HR image size")
    parser.add_argument("--ntrain", type=int, default=100, help="number of training images")
    parser.add_argument("--nvalid", type=int, default=20, help="number of validation images")
    parser.add_argument(
        "--catalog",
        type=str,
        required=True,
        help="path to RealGalaxy catalog FITS file OR directory containing real_galaxy_catalog_*.fits",
    )
    parser.add_argument(
        "--gal-density",
        type=float,
        default=DEFAULT_GAL_DENSITY_ARCMIN2,
        help=f"default {DEFAULT_GAL_DENSITY_ARCMIN2} galaxies/arcmin^2",
    )
    parser.add_argument(
        "--star-density",
        type=float,
        default=DEFAULT_STAR_DENSITY_ARCMIN2,
        help=f"default {DEFAULT_STAR_DENSITY_ARCMIN2} stars/arcmin^2",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="number of parallel processes (default: 1 for serial). "
             "Use 0 or -1 for all available CPUs.",
    )

    return parser.parse_args()


def validate_options(options):
    if options.pixel_scale <= 0:
        raise ValueError("--pix must be positive.")
    if options.nside <= 0:
        raise ValueError("--nside must be positive.")
    if options.ntrain < 0 or options.nvalid < 0:
        raise ValueError("--ntrain and --nvalid must be non-negative.")
    if options.gal_density < 0:
        raise ValueError("--gal-density must be non-negative.")
    if options.star_density < 0:
        raise ValueError("--star-density must be non-negative.")
    if not os.path.exists(options.catalog):
        raise FileNotFoundError(f"Catalog path does not exist: {options.catalog}")


def main():
    import warnings
    warnings.warn(
        "This script is deprecated and will be removed in a future version. "
        "Use 'python -m euclid_polish.cli.main' instead, which provides an "
        "interactive CLI with the same functionality.",
        DeprecationWarning,
        stacklevel=2
    )

    options = parse_args()

    try:
        validate_options(options)
        cosmos_catalog, catalog_file, catalog_dir = resolve_cosmos_catalog(options.catalog)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Generating clean galaxy and star data")
    print(f"Pixel scale: {options.pixel_scale} arcsec")
    print(f"Image size: {options.nside}")
    print(f"Catalog file: {catalog_file}")
    print(f"Catalog objects: {cosmos_catalog.nobjects}")
    print(f"Output directory: {options.fdout}")
    print(f"Galaxy density: {options.gal_density} galaxies/arcmin^2")
    print(f"Star density: {options.star_density} stars/arcmin^2")

    if options.nproc in (0, -1):
        effective_nproc = min(cpu_count(), 8)
    elif options.nproc > 0:
        effective_nproc = min(options.nproc, 8)
    else:
        effective_nproc = options.nproc

    print(f"Parallel processes: {effective_nproc}")
    print("Using parametric COSMOS galaxies and stars")

    os.makedirs(options.fdout, exist_ok=True)

    print("\nGenerating validation set...")
    generate_clean_data(
        nimages=options.nvalid,
        fdirout=options.fdout,
        catalog=cosmos_catalog,
        subset="valid",
        nstart=options.ntrain,
        pixel_scale=options.pixel_scale,
        image_size=options.nside,
        gal_density_arcmin2=options.gal_density,
        star_density_arcmin2=options.star_density,
        nproc=options.nproc,
        catalog_file=catalog_file,
        catalog_dir=catalog_dir,
    )

    print("\nGenerating training set...")
    generate_clean_data(
        nimages=options.ntrain,
        fdirout=options.fdout,
        catalog=cosmos_catalog,
        subset="train",
        nstart=0,
        pixel_scale=options.pixel_scale,
        image_size=options.nside,
        gal_density_arcmin2=options.gal_density,
        star_density_arcmin2=options.star_density,
        nproc=options.nproc,
        catalog_file=catalog_file,
        catalog_dir=catalog_dir,
    )

    # Generate 5 sample visualizations from training set
    print("\nGenerating sample visualizations...")
    import subprocess
    viz_script = os.path.join(os.path.dirname(__file__), "visualize_tfrecords.py")
    if os.path.exists(viz_script):
        result = subprocess.run(
            ["python", viz_script, "--data-dir", options.fdout, "--subset", "train",
             "--num-images", "5", "--mode", "first"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("Sample visualizations created successfully!")
        else:
            print(f"Warning: Visualization script failed: {result.stderr}")
    else:
        print(f"Warning: Visualization script not found at {viz_script}")

    print("\nDone!")


if __name__ == "__main__":
    main()
