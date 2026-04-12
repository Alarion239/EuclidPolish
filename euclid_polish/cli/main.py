#!/usr/bin/env python3
"""
Unified Interactive CLI for EuclidPolish.

This module provides an interactive command-line interface for all EuclidPolish operations.
"""

import glob
import json
import os
import subprocess
import sys
import traceback

import numpy as np
import tensorflow as tf
from astropy.io import fits
from tf_keras.losses import MeanAbsoluteError
from tf_keras.optimizers.schedules import PiecewiseConstantDecay
from tqdm import tqdm

try:
    from questionary import select, confirm, password
except ImportError:
    print("Installing questionary...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "questionary"])
    from questionary import select, confirm, password

from euclid_polish.config import Config
from euclid_polish.cli.utils import DisplayFormatter, ValidationResult
from euclid_polish.euclid import (
    StarCatalog,
    EuclidCutoutDownloader,
    DownloadConfig,
    PSFExtractor,
    PSFExtractionConfig,
    FitsValidator,
)
from euclid_polish.sky import CleanSkyGenerator
from euclid_polish.sky.clean_generator import GeneratorConfig
from euclid_polish.sky.psf_convolution import PSFConvolution, ConvolutionConfig
from euclid_polish.training import Trainer, RadioSky
from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.visualization import BaseVisualizer, estimate_fwhm
from euclid_polish.visualization.methods import draw_clean_image, draw_clean_dirty_pair
from euclid_polish.sky.tfrecord import read_tfrecord


class InteractiveCLI:
    """
    Interactive CLI with menu-driven interface.

    This class provides an interactive menu system for all EuclidPolish operations.
    """

    # Module definitions
    MODULES = {
        "euclid": {
            "name": "Euclid Operations",
            "description": "Query catalog, download cutouts, extract PSF",
            "icon": "🔭",
        },
        "sky": {
            "name": "Sky Generation",
            "description": "Generate clean and dirty sky images",
            "icon": "🌌",
        },
        "training": {
            "name": "Model Training",
            "description": "Train WDSR super-resolution models",
            "icon": "🧠",
        },
        "visualization": {
            "name": "Visualization",
            "description": "Visualize data and results",
            "icon": "📊",
        },
    }

    def __init__(self):
        """Initialize the CLI."""
        self.config = Config
        self.display = DisplayFormatter

    def run(self):
        """Run the interactive CLI."""
        while True:
            choice = select(
                "Select a module:",
                choices=[
                    {"name": f"{m['icon']} {m['name']} - {m['description']}", "value": key}
                    for key, m in self.MODULES.items()
                ] + [
                    {"name": "❌ Exit", "value": "exit"}
                ]
            ).ask()

            if choice == "exit" or choice is None:
                print("\nGoodbye!")
                break

            # Route to appropriate module
            if choice == "euclid":
                self._euclid_menu()
            elif choice == "sky":
                self._sky_menu()
            elif choice == "training":
                self._training_menu()
            elif choice == "visualization":
                self._visualization_menu()

    def _euclid_menu(self):
        """Euclid operations menu."""
        while True:
            choice = select(
                "🔭 Euclid Operations - Select an action:",
                choices=[
                    {"name": "📊 Show catalog info", "value": "info"},
                    {"name": "🔍 Query bright stars catalog", "value": "query"},
                    {"name": "⬇️  Download cutouts", "value": "download"},
                    {"name": "✨ Extract PSF", "value": "extract_psf"},
                    {"name": "✔️  Check cutouts integrity", "value": "check"},
                    {"name": "🔙 Back to main menu", "value": "back"},
                ]
            ).ask()

            if choice == "back" or choice is None:
                break

            if choice == "info":
                self._show_catalog_info()
            elif choice == "query":
                self._query_catalog()
            elif choice == "download":
                self._download_cutouts()
            elif choice == "extract_psf":
                self._extract_psf()
            elif choice == "check":
                self._check_integrity()

    def _show_catalog_info(self):
        """Show star catalog information."""
        output_dir = select(
            "Select output directory:",
            choices=[
                {"name": "./data/euclid_stars (default)", "value": "./data/euclid_stars"},
                {"name": "Custom path...", "value": "custom"},
            ]
        ).ask()

        if output_dir == "custom":
            output_dir = input("Enter path: ").strip()

        catalog = StarCatalog(output_dir)

        if not catalog.exists():
            print(f"\n⚠️  Catalog not found at {catalog.catalog_path}")
            if confirm("Query the catalog first?", default=True).ask():
                self._query_catalog()
            return

        summary = catalog.get_summary()
        status = catalog.get_stars_by_status()

        DisplayFormatter.print_header("📊 Star Catalog Summary")
        print(f"Total stars:        {summary['total']}")
        print(f"Valid:              {summary['valid']} ✓")
        print(f"Corrupted:          {summary['corrupted']} 🔴")
        print(f"Failed downloads:   {summary['failed']} ❌")
        print(f"Pending:            {summary['pending']} ⏳")

        if summary.get('mag_min'):
            print(f"\nMagnitude range:   {summary['mag_min']:.2f} - {summary['mag_max']:.2f}")

        print(f"\nNext ID:            {summary['next_id']}")
        print(f"Catalog file:       {catalog.catalog_path}")

    def _query_catalog(self):
        """Query Euclid archive for bright stars."""
        output_dir = select(
            "Select output directory:",
            choices=[
                {"name": "./data/euclid_stars (default)", "value": "./data/euclid_stars"},
                {"name": "Custom path...", "value": "custom"},
            ]
        ).ask()

        if output_dir == "custom":
            output_dir = input("Enter path: ").strip()

        ra = input("Enter RA (degrees, 0-360, default 270): ").strip() or "270"
        dec = input("Enter Dec (degrees, -90 to 90, default 66): ").strip() or "66"
        radius = input("Enter radius (degrees, default 1): ").strip() or "1"
        magnitude = input("Enter magnitude limit (default 20): ").strip() or "20"

        # Validate inputs
        ra_valid = ValidationResult.validate_ra(ra)
        if ra_valid is not True:
            print(f"\n✗ {ra_valid}")
            return

        dec_valid = ValidationResult.validate_dec(dec)
        if dec_valid is not True:
            print(f"\n✗ {dec_valid}")
            return

        radius_valid = ValidationResult.validate_positive_number(radius, "Radius")
        if radius_valid is not True:
            print(f"\n✗ {radius_valid}")
            return

        mag_valid = ValidationResult.validate_positive_number(magnitude, "Magnitude limit")
        if mag_valid is not True:
            print(f"\n✗ {mag_valid}")
            return

        # Parse values
        ra_val = float(ra)
        dec_val = float(dec)
        radius_val = float(radius)
        mag_val = float(magnitude)

        if confirm("\nQuery Euclid catalog with these parameters?", default=True).ask():
            print(f"\nQuerying RA={ra_val:.1f}°, Dec={dec_val:.1f}°, radius={radius_val}°, mag<{mag_val}...")

            try:
                catalog = StarCatalog(output_dir)
                result = catalog.query_euclid_catalog(
                    ra=ra_val,
                    dec=dec_val,
                    radius=radius_val,
                    magnitude_limit=mag_val,
                )

                print(f"\n{result['message']}")
                if result['skipped'] > 0:
                    print(f"  (Skipped {result['skipped']} duplicate stars)")

                if result['added'] > 0:
                    print("\n✓ Query completed successfully!")
                elif result['added'] == 0 and result['total'] > 0:
                    print("\n✓ Catalog already has the requested stars")

            except Exception as e:
                print(f"\n✗ Query failed: {e}")
                traceback.print_exc()

    def _download_cutouts(self):
        """Download Euclid cutouts."""
        output_dir = select(
            "Select output directory:",
            choices=[
                {"name": "./data/euclid_stars (default)", "value": "./data/euclid_stars"},
                {"name": "Custom path...", "value": "custom"},
            ]
        ).ask()

        if output_dir == "custom":
            output_dir = input("Enter path: ").strip()

        # Check if catalog exists
        catalog = StarCatalog(output_dir)
        if not catalog.exists():
            print(f"\n✗ Catalog not found at {catalog.catalog_path}")
            if confirm("Query the catalog first?", default=True).ask():
                self._query_catalog()
            return

        # Get catalog summary
        summary = catalog.get_summary()
        print(f"\n📊 Catalog contains {summary['total']} stars")
        print(f"  Valid: {summary['valid']}")
        print(f"  Corrupted: {summary['corrupted']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Pending: {summary['pending']}")

        if summary['pending'] == 0 and summary['total'] > 0:
            print(f"\n✓ All stars already have cutouts!")
            if confirm("Re-download existing cutouts?", default=False).ask():
                pass  # Continue with download
            else:
                return

        cutout_size_input = input("Enter cutout size in pixels (default 256): ").strip()
        cutout_size = int(cutout_size_input) if cutout_size_input else 256

        # Configure downloader
        config = DownloadConfig(
            cutout_size=cutout_size,
        )

        downloader = EuclidCutoutDownloader(catalog, config)

        # Confirm
        num_to_download = summary['pending'] if summary['pending'] > 0 else summary['total']
        if not confirm(f"\nDownload cutouts for {num_to_download} stars?", default=True).ask():
            return

        # Download
        try:
            print("\nDownloading...")
            result = downloader.download(show_progress=True)

            print(f"\n✓ Download completed!")
            print(f"  Downloaded: {result['downloaded']}")
            print(f"  Valid: {result['valid']}")
            print(f"  Corrupted: {result['corrupted']}")

            if result.get('corrupted_ids'):
                print(f"\n⚠️  Failed star IDs: {result['corrupted_ids']}")

        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            traceback.print_exc()

    def _extract_psf(self):
        """Extract PSF from cutouts."""
        output_dir = select(
            "Select output directory:",
            choices=[
                {"name": "./data/euclid_stars (default)", "value": "./data/euclid_stars"},
                {"name": "Custom path...", "value": "custom"},
            ]
        ).ask()

        if output_dir == "custom":
            output_dir = input("Enter path: ").strip()

        cutout_dir = f"{output_dir}/cutouts"
        psf_dir = select(
            "Select PSF output directory:",
            choices=[
                {"name": "./data/euclid_psf (default)", "value": "./data/euclid_psf"},
                {"name": "Custom path...", "value": "custom"},
            ]
        ).ask()

        if psf_dir == "custom":
            psf_dir = input("Enter path: ").strip()

        num_stars_input = input("Number of stars to use (default: all): ").strip()

        # PSF size configuration
        psf_size_input = input(f"PSF size in pixels (must be odd, default {Config.DEFAULT_PSF_SIZE}): ").strip()
        if psf_size_input:
            try:
                psf_size = int(psf_size_input)
                if psf_size <= 0 or psf_size % 2 == 0:
                    print(f"\n✗ PSF size must be a positive odd integer (got {psf_size})")
                    return
            except ValueError:
                print(f"\n✗ Invalid PSF size: must be an integer")
                return
        else:
            psf_size = Config.DEFAULT_PSF_SIZE

        # Check if cutout directory exists
        if not os.path.exists(cutout_dir):
            print(f"\n✗ Cutout directory not found: {cutout_dir}")
            return

        # Configure PSF extractor
        config = PSFExtractionConfig(
            psf_size=psf_size,
            progress_bar=True,
        )

        extractor = PSFExtractor(config)

        # Get cutout files
        all_files = extractor.get_cutout_files(cutout_dir)

        if len(all_files) == 0:
            print(f"\n✗ No FITS files found in {cutout_dir}")
            return

        print(f"\nFound {len(all_files)} cutout files")

        # Select files
        if num_stars_input:
            num_stars = int(num_stars_input)
            selected_files = extractor.select_files(all_files, num_stars=num_stars)
            print(f"Using first {num_stars} stars")
        else:
            selected_files = all_files
            print(f"Using all {len(all_files)} stars")

        # Confirm
        if not confirm(f"\nExtract PSF from {len(selected_files)} cutouts?", default=True).ask():
            return

        # Extract PSF
        try:
            epsf, fitted_stars = extractor.build_epsf(selected_files)
            fits_path, npy_path = extractor.save_psf(psf_dir)

            print(f"\n✓ PSF extraction completed!")
            print(f"  FITS file: {fits_path}")
            print(f"  NPY file: {npy_path}")

            # Show summary
            summary = extractor.get_summary()
            print(f"\nPSF Summary:")
            print(f"  Shape: {summary['shape']}")
            print(f"  Oversampling: {summary['oversampling']}")
            print(f"  Data type: {summary['data_type']}")

        except Exception as e:
            print(f"\n✗ PSF extraction failed: {e}")

    def _check_integrity(self):
        """Check cutouts integrity."""
        output_dir = select(
            "Select output directory:",
            choices=[
                {"name": "./data/euclid_stars (default)", "value": "./data/euclid_stars"},
                {"name": "Custom path...", "value": "custom"},
            ]
        ).ask()

        if output_dir == "custom":
            output_dir = input("Enter path: ").strip()

        cutout_dir = f"{output_dir}/cutouts"

        # Check if directory exists
        if not os.path.exists(cutout_dir):
            print(f"\n✗ Cutout directory not found: {cutout_dir}")
            return

        # Get all FITS files
        fits_files = glob.glob(os.path.join(cutout_dir, "*.fits"))

        if len(fits_files) == 0:
            print(f"\n✗ No FITS files found in {cutout_dir}")
            return

        print(f"\nChecking {len(fits_files)} FITS files...")

        validator = FitsValidator()

        results = {
            "valid": [],
            "corrupted": [],
        }

        for filepath in tqdm(fits_files, desc="Validating"):
            is_valid, error_msg = validator.validate_basic_integrity(filepath)
            if is_valid:
                results["valid"].append(filepath)
            else:
                results["corrupted"].append((filepath, error_msg))

        # Display results
        DisplayFormatter.print_header("Integrity Check Results")
        print(f"Total files:      {len(fits_files)}")
        print(f"Valid:            {len(results['valid'])} ✓")
        print(f"Corrupted:        {len(results['corrupted'])} 🔴")

        if results['corrupted']:
            print(f"\nCorrupted files:")
            for filepath, error_msg in results['corrupted'][:10]:  # Show first 10
                filename = os.path.basename(filepath)
                print(f"  🔴 {filename}: {error_msg}")
            if len(results['corrupted']) > 10:
                print(f"  ... and {len(results['corrupted']) - 10} more")

        # Update catalog if stars.json exists
        catalog = StarCatalog(output_dir)
        if catalog.exists():
            catalog_data = catalog.load()
            stars = catalog_data.get('stars', [])

            # Update star status based on validation
            for filepath, error_msg in results['corrupted']:
                filename = os.path.basename(filepath)
                # Try to extract star ID from filename
                parts = filename.split('_')
                if len(parts) >= 2 and parts[0] == 'star':
                    try:
                        star_id = int(parts[1])
                        for star in stars:
                            if star.get('id') == star_id:
                                star['corrupted'] = True
                                break
                    except ValueError:
                        pass

            catalog.save(catalog_data)
            print(f"\n✓ Updated catalog with validation results")

        print(f"\n✓ Integrity check completed!")

    def _sky_menu(self):
        """Sky generation menu."""
        while True:
            choice = select(
                "🌌 Sky Generation - Select an action:",
                choices=[
                    {"name": "✨ Generate clean sky data", "value": "generate_clean"},
                    {"name": "🌫️  Convolve HR to LR (dirty sky)", "value": "convolve"},
                    {"name": "🔙 Back to main menu", "value": "back"},
                ]
            ).ask()

            if choice == "back" or choice is None:
                break

            if choice == "generate_clean":
                self._generate_clean_data()
            elif choice == "convolve":
                self._convolve_hr_to_lr()

    def _convolve_hr_to_lr(self):
        """Convolve HR images to LR (dirty sky)."""
        data_dir = input("Clean data directory (default ./data/clean_data): ").strip() or "./data/clean_data"
        psf_path = input("PSF file path (default ./data/euclid_psf/euclid_psf.npy): ").strip() or "./data/euclid_psf/euclid_psf.npy"

        # Validate paths
        if not os.path.exists(data_dir):
            print(f"\n✗ Data directory not found: {data_dir}")
            return

        if not os.path.exists(psf_path):
            print(f"\n✗ PSF file not found: {psf_path}")
            return

        # Load PSF
        print(f"\nLoading PSF from {psf_path}...")
        psf_kernel = np.load(psf_path)
        print(f"  PSF shape: {psf_kernel.shape}, dtype: {psf_kernel.dtype}")

        # Configure convolution (no noise, normalize=False → store raw float values as fp16)
        config = ConvolutionConfig(
            rebin_factor=4,
            add_noise=False,
            normalize=False,
        )

        convolver = PSFConvolution(config)

        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'index': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
        }

        # Resolve which subsets exist
        subsets_to_run = []
        for subset in ("train", "valid"):
            tfrecord_path = os.path.join(data_dir, subset, f"{subset}_clean.tfrecord")
            if os.path.exists(tfrecord_path):
                metadata_path = os.path.join(data_dir, subset, f"{subset}_clean_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path) as f:
                        n_images = len(json.load(f))
                else:
                    print(f"\n⚠️  Metadata not found for {subset}, counting records (may be slow)...")
                    n_images = sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path))
                subsets_to_run.append((subset, tfrecord_path, n_images))
            else:
                print(f"  ⚠️  Skipping {subset}: {tfrecord_path} not found")

        if not subsets_to_run:
            print(f"\n✗ No clean TFRecords found in {data_dir}")
            return

        for subset, tfr_str, n_images in subsets_to_run:
            print(f"\n  {subset}: {n_images} images")
        total = sum(n for _, _, n in subsets_to_run)

        if not confirm(f"\nConvolve {total} images (train+valid) to dirty LR?", default=True).ask():
            return

        all_viz_pairs = []

        for subset, tfrecord_path, n_images in subsets_to_run:
            dirty_dir = os.path.join(data_dir, f"dirty_{subset}")
            os.makedirs(dirty_dir, exist_ok=True)
            dirty_tfrecord_path = os.path.join(dirty_dir, f"{subset}_dirty.tfrecord")

            n_ok = 0
            n_err = 0
            n_viz = 5
            viz_pairs = []
            raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

            with tf.io.TFRecordWriter(dirty_tfrecord_path) as writer:
                for raw_record in tqdm(raw_dataset, total=n_images, desc=f"Convolving {subset}"):
                    try:
                        example = tf.io.parse_single_example(raw_record, feature_description)
                        height = int(example['height'].numpy())
                        width = int(example['width'].numpy())
                        index = int(example['index'].numpy())

                        # Decode fp16 image and cast to float32 for processing
                        image_bytes = tf.io.decode_raw(example['image'], tf.float16)
                        hr_data = tf.cast(tf.reshape(image_bytes, [height, width]), tf.float32).numpy()

                        # Convolve and downsample (float output, no normalization)
                        lr_data, _ = convolver.process_hr_to_lr(hr_data, psf_kernel)

                        # Write dirty LR image as fp16 TFRecord
                        lr_h, lr_w = lr_data.shape
                        lr_fp16 = lr_data.flatten().astype(np.float16)
                        feature = {
                            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lr_fp16.tobytes()])),
                            'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[lr_h])),
                            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[lr_w])),
                        }
                        writer.write(
                            tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
                        )

                        if len(viz_pairs) < n_viz:
                            viz_pairs.append((hr_data, lr_data, index))

                        n_ok += 1

                    except Exception as e:
                        n_err += 1
                        tqdm.write(f"  ✗ Skipping record (error: {e})")

            print(f"  ✓ {subset}: {n_ok} ok, {n_err} skipped → {dirty_tfrecord_path}")
            all_viz_pairs.append((subset, viz_pairs))

        if any(pairs for _, pairs in all_viz_pairs):
            if confirm(f"\nVisualize sample HR/LR pairs?", default=True).ask():
                for subset, viz_pairs in all_viz_pairs:
                    if viz_pairs:
                        vis_dir = os.path.join(data_dir, "vis", f"dirty_{subset}")
                        os.makedirs(vis_dir, exist_ok=True)
                        for hr_data, lr_data, index in viz_pairs:
                            output_path = os.path.join(vis_dir, f'pair_{index:05d}.png')
                            draw_clean_dirty_pair(hr_data, lr_data, output_path, index=index)
                        print(f"  ✓ {subset}: {len(viz_pairs)} pair plots → {vis_dir}")

    def _generate_clean_data(self):
        """Generate clean sky data."""
        output_dir = input("Enter output directory (default ./data/clean_data): ").strip() or "./data/clean_data"
        catalog_path = input("Enter path to COSMOS catalog: ").strip()

        if not os.path.exists(catalog_path):
            print(f"\n✗ Catalog not found: {catalog_path}")
            return

        ntrain = input("Number of training images (default 100): ").strip() or "100"
        nvalid = input("Number of validation images (default 20): ").strip() or "20"
        pixel_scale = input("Pixel scale in arcsec (default 0.05): ").strip() or "0.05"
        image_size = input("Image size (default 256): ").strip() or "256"

        # Validate inputs
        try:
            ntrain_val = int(ntrain)
            nvalid_val = int(nvalid)
            pixel_scale_val = float(pixel_scale)
            image_size_val = int(image_size)
        except ValueError:
            print("\n✗ Invalid input: values must be numbers")
            return

        if confirm("\nGenerate clean sky data with these parameters?", default=True).ask():
            print(f"\nConfiguration:")
            print(f"  Output directory: {output_dir}")
            print(f"  Catalog: {catalog_path}")
            print(f"  Training images: {ntrain_val}")
            print(f"  Validation images: {nvalid_val}")
            print(f"  Pixel scale: {pixel_scale_val} arcsec")
            print(f"  Image size: {image_size_val}x{image_size_val}")

            try:
                # Resolve COSMOS catalog
                catalog, catalog_file, catalog_dir = CleanSkyGenerator.resolve_cosmos_catalog(catalog_path)
                print(f"\nCOSMOS catalog loaded: {catalog.nobjects} objects")

                # Create generator
                config = GeneratorConfig(
                    image_size=image_size_val,
                    pixel_scale=pixel_scale_val,
                )
                generator = CleanSkyGenerator(config)

                # Generate validation set
                print("\nGenerating validation set...")
                images_valid, metadata_valid = generator.generate(
                    catalog=catalog,
                    output_dir=output_dir,
                    subset='valid',
                    nimages=nvalid_val,
                    nstart=ntrain_val,
                )

                # Generate training set
                print("\nGenerating training set...")
                images_train, metadata_train = generator.generate(
                    catalog=catalog,
                    output_dir=output_dir,
                    subset='train',
                    nimages=ntrain_val,
                    nstart=0,
                )

                def _write_tfrecord(images, output_path, image_shape, desc):
                    h, w = image_shape
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with tf.io.TFRecordWriter(output_path) as writer:
                        for idx, img in enumerate(tqdm(images, desc=desc, unit="img")):
                            arr_fp16 = img.flatten().astype(np.float16)
                            feature = {
                                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr_fp16.tobytes()])),
                                'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[idx])),
                                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
                                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
                            }
                            writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())

                # Write training TFRecord
                train_output = os.path.join(output_dir, 'train', 'train_clean.tfrecord')
                _write_tfrecord(images_train, train_output, (image_size_val, image_size_val), "Saving train")
                print(f"  Saved: {train_output}")

                # Write validation TFRecord
                valid_output = os.path.join(output_dir, 'valid', 'valid_clean.tfrecord')
                _write_tfrecord(images_valid, valid_output, (image_size_val, image_size_val), "Saving valid")
                print(f"  Saved: {valid_output}")

                print("\n✓ Clean data generation completed!")

            except Exception as e:
                print(f"\n✗ Generation failed: {e}")
                traceback.print_exc()

    def _training_menu(self):
        """Model training menu."""
        while True:
            choice = select(
                "🧠 Model Training - Select an action:",
                choices=[
                    {"name": "🏋️  Train WDSR model", "value": "train"},
                    {"name": "📈 Evaluate model", "value": "evaluate"},
                    {"name": "🔄 Restore from checkpoint", "value": "restore"},
                    {"name": "🔙 Back to main menu", "value": "back"},
                ]
            ).ask()

            if choice == "back" or choice is None:
                break

            if choice == "train":
                self._train_model()
            elif choice == "evaluate":
                print("\n📈 Evaluate model")
                print("Use the training script with --evaluate flag")
            elif choice == "restore":
                print("\n🔄 Restore from checkpoint")
                print("See restore.py for examples")

    def _train_model(self):
        """Train WDSR model."""
        scale = input("Scale factor (default 2): ").strip() or "2"
        num_res_blocks = input("Number of residual blocks (default 32): ").strip() or "32"

        images_dir = input("Images directory (default ./data/clean_data): ").strip() or "./data/clean_data"
        checkpoint_dir = input("Checkpoint directory (default ./ckpt/wdsr): ").strip() or "./ckpt/wdsr"

        # Validate inputs
        try:
            scale_val = int(scale)
            num_res_blocks_val = int(num_res_blocks)
        except ValueError:
            print("\n✗ Invalid input: scale and num_res_blocks must be integers")
            return

        if not os.path.exists(images_dir):
            print(f"\n✗ Images directory not found: {images_dir}")
            return

        # Check for dirty images
        dirty_train_dir = os.path.join(images_dir, 'dirty_train')
        dirty_valid_dir = os.path.join(images_dir, 'dirty_valid')

        if not os.path.exists(dirty_train_dir):
            print(f"\n⚠️  Warning: Dirty training images not found at {dirty_train_dir}")
            print("    You may need to run HR to LR convolution first")
            if not confirm("Continue anyway?", default=False).ask():
                return

        print(f"\nConfiguration:")
        print(f"  Scale: {scale_val}x")
        print(f"  Residual blocks: {num_res_blocks_val}")
        print(f"  Images directory: {images_dir}")
        print(f"  Checkpoint directory: {checkpoint_dir}")

        if confirm("\nStart training?", default=True).ask():
            print("\n⚠️  Training will run until interrupted (Ctrl+C) or completion")

            try:
                # Create model
                model = wdsr(scale=scale_val, num_res_blocks=num_res_blocks_val, nchan=1)

                # Create trainer
                learning_rate = PiecewiseConstantDecay(boundaries=[200000], values=[1e-3, 5e-4])
                trainer = Trainer(
                    model=model,
                    loss=MeanAbsoluteError(),
                    learning_rate=learning_rate,
                    checkpoint_dir=checkpoint_dir,
                )

                # Create data loaders
                train_loader = RadioSky(
                    scale=scale_val,
                    subset='train',
                    images_dir=images_dir,
                )

                valid_loader = RadioSky(
                    scale=scale_val,
                    subset='valid',
                    images_dir=images_dir,
                )

                # Create datasets
                train_ds = train_loader.dataset(batch_size=16, random_transform=True)
                valid_ds = valid_loader.dataset(batch_size=1, random_transform=False, repeat_count=1)

                # Train
                print("\nStarting training...")
                trainer.train(train_ds, valid_ds.take(10), steps=10000)

                # Restore and evaluate
                trainer.restore()
                psnr = trainer.evaluate(valid_ds)
                print(f'\nFinal PSNR = {psnr.numpy():.3f}')

                print("\n✓ Training completed!")

            except KeyboardInterrupt:
                print("\n\n⚠️  Training interrupted by user")
            except Exception as e:
                print(f"\n✗ Training failed: {e}")
                traceback.print_exc()

    def _visualization_menu(self):
        """Visualization menu."""
        while True:
            choice = select(
                "📊 Visualization - Select an action:",
                choices=[
                    {"name": "🔭 Visualize Euclid cutouts", "value": "viz_cutouts"},
                    {"name": "✨ Visualize PSF", "value": "viz_psf"},
                    {"name": "🌌 Visualize training data", "value": "viz_training"},
                    {"name": "🔙 Back to main menu", "value": "back"},
                ]
            ).ask()

            if choice == "back" or choice is None:
                break

            if choice == "viz_cutouts":
                self._visualize_cutouts()
            elif choice == "viz_psf":
                self._visualize_psf()
            elif choice == "viz_training":
                self._visualize_training_data()

    def _visualize_cutouts(self):
        """Visualize Euclid cutouts."""
        output_dir = select(
            "Select output directory:",
            choices=[
                {"name": "./data/euclid_stars (default)", "value": "./data/euclid_stars"},
                {"name": "Custom path...", "value": "custom"},
            ]
        ).ask()

        if output_dir == "custom":
            output_dir = input("Enter path: ").strip()

        num_stars_input = input("Number of stars to visualize (default 5): ").strip() or "5"
        try:
            num_stars = int(num_stars_input)
        except ValueError:
            print("\n✗ Invalid input: number of stars must be an integer")
            return

        # Load catalog
        catalog = StarCatalog(output_dir)
        if not catalog.exists():
            print(f"\n✗ Catalog not found at {catalog.catalog_path}")
            return

        catalog_data = catalog.load()
        stars = catalog_data.get('stars', [])

        if len(stars) == 0:
            print(f"\n✗ No stars found in catalog")
            return

        # Select stars (first N)
        selected_stars = stars[:min(num_stars, len(stars))]

        # Visualize each star
        cutout_dir = os.path.join(output_dir, Config.CUTOUTS_SUBDIR)
        vis_dir = os.path.join(output_dir, Config.VIS_SUBDIR)
        os.makedirs(vis_dir, exist_ok=True)

        print(f"\nVisualizing {len(selected_stars)} stars...")

        for star in tqdm(selected_stars, desc="Creating visualizations"):
            # Load FITS data
            fits_files = glob.glob(os.path.join(cutout_dir, f"star_{star['id']:04d}_*.fits"))
            if not fits_files:
                print(f"  Warning: No cutout for star {star['id']}")
                continue

            try:
                with fits.open(fits_files[0]) as hdul:
                    data = hdul[0].data

                visualizer = BaseVisualizer()
                visualizer.add_scale_panel(data)
                visualizer.add_scale_panel(data, log_scale=True)
                visualizer.add_central_slices_panel(data)
                visualizer.add_contours_panel(data)
                visualizer.add_radial_profile_panel(data)

                mag_str = f"{star['magnitude']:.2f}" if star.get('magnitude') else "N/A"
                stats_dict = {
                    'title': 'Star Information:',
                    'stats': {
                        'ID': f"{star['id']:04d}",
                        'RA': f"{star['ra']:.6f}°",
                        'Dec': f"{star['dec']:.6f}°",
                        'Magnitude': mag_str,
                    },
                    'include_data_stats': True
                }
                visualizer.add_statistics_panel(data, stats_dict)

                # Save figure
                output_path = os.path.join(vis_dir, f'star_{star["id"]:04d}.png')
                visualizer.save_figure(output_path)

            except Exception as e:
                print(f"  Warning: Failed to visualize star {star['id']}: {e}")

        print(f"\n✓ Visualizations saved to {vis_dir}")

    def _visualize_psf(self):
        """Visualize PSF."""
        psf_dir = select(
            "Select PSF directory:",
            choices=[
                {"name": "./data/euclid_psf (default)", "value": "./data/euclid_psf"},
                {"name": "Custom path...", "value": "custom"},
            ]
        ).ask()

        if psf_dir == "custom":
            psf_dir = input("Enter path: ").strip()

        # Validate directory
        if not os.path.exists(psf_dir):
            print(f"\n✗ PSF directory not found: {psf_dir}")
            return

        # Look for PSF file
        psf_file = os.path.join(psf_dir, "euclid_psf.npy")
        if not os.path.exists(psf_file):
            print(f"\n✗ PSF file not found: {psf_file}")
            return

        # Load PSF
        print(f"\nLoading PSF from {psf_file}...")
        psf_data = np.load(psf_file)

        # Create visualization
        visualizer = BaseVisualizer()

        # 1. PSF image (linear scale with clipping)
        visualizer.add_scale_panel(psf_data, title_suffix='\nEuclid VIS PSF')

        # 2. PSF image (log scale)
        visualizer.add_scale_panel(psf_data, log_scale=True)

        # 3. Central slice through PSF
        visualizer.add_central_slices_panel(psf_data)

        # 4. PSF contour plot
        visualizer.add_contours_panel(psf_data)

        # 5. Radial profile
        visualizer.add_radial_profile_panel(psf_data)

        # 6. PSF statistics
        center_y, center_x = psf_data.shape[0] // 2, psf_data.shape[1] // 2
        x_slice = psf_data[center_y, :]
        y_slice = psf_data[:, center_x]

        # Calculate statistics
        total_flux = np.sum(psf_data)
        fwhm_y = estimate_fwhm(x_slice)
        fwhm_x = estimate_fwhm(y_slice)
        ellipticity = abs(fwhm_y - fwhm_x) / ((fwhm_y + fwhm_x) / 2) if (fwhm_y + fwhm_x) > 0 else 0

        stats_dict = {
            'title': 'PSF Statistics:',
            'stats': {
                'Shape': f"{psf_data.shape[0]} x {psf_data.shape[1]} pixels",
                'Total Flux': f"{total_flux:.6f}",
                'Center': f"({center_x}, {center_y})",
                '': '',  # spacer
                'FWHM X': f"{fwhm_x:.2f} pixels",
                'FWHM Y': f"{fwhm_y:.2f} pixels",
                'Ellipticity': f"{ellipticity:.4f}",
            },
            'include_data_stats': True
        }

        visualizer.add_statistics_panel(psf_data, stats_dict)

        plt.suptitle('Euclid VIS\nPSF', fontsize=16, y=0.98)

        # Save figure
        output_path = os.path.join(psf_dir, 'euclid_psf_visualization.png')
        visualizer.save_figure(output_path)

        print(f"\n✓ PSF visualization saved to: {output_path}")

    def _visualize_training_data(self):
        """Visualize training data."""
        data_dir = input("Enter data directory (default ./data/clean_data): ").strip() or "./data/clean_data"
        subset = select(
            "Select subset:",
            choices=[
                {"name": "train", "value": "train"},
                {"name": "valid", "value": "valid"},
            ]
        ).ask()

        num_images_input = input("Number of images to visualize (default 5): ").strip() or "5"
        try:
            num_images = int(num_images_input)
        except ValueError:
            print("\n✗ Invalid input: number of images must be an integer")
            return

        # Validate directory
        if not os.path.exists(data_dir):
            print(f"\n✗ Data directory not found: {data_dir}")
            return

        # Find TFRecord file
        tfrecord_path = os.path.join(data_dir, subset, f"{subset}_clean.tfrecord")
        if not os.path.exists(tfrecord_path):
            print(f"\n✗ TFRecord file not found: {tfrecord_path}")
            print(f"   (Expected: {subset}_clean.tfrecord)")
            return

        # Output directory for visualizations
        vis_dir = os.path.join(data_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)

        # Visualize
        print(f"\nVisualizing {num_images} images from {tfrecord_path}...")

        try:
            images = read_tfrecord(tfrecord_path, num_images=num_images, mode='first')
            for image, index, height, width in images:
                output_path = os.path.join(vis_dir, f'image_{index:04d}.png')
                draw_clean_image(image, output_path, index=index)

            print(f"\n✓ Visualizations saved to {vis_dir}")
            print(f"  Created {len(images)} images")

        except Exception as e:
            print(f"\n✗ Visualization failed: {e}")
            traceback.print_exc()


def main():
    """Main entry point for the interactive CLI."""
    print("\n" + "=" * 60)
    print("  🌌 EuclidPolish - Super-resolution for Astronomical Images")
    print("=" * 60 + "\n")

    cli = InteractiveCLI()
    try:
        cli.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
