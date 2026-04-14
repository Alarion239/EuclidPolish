#!/usr/bin/env python3
"""
Unified Interactive CLI for EuclidPolish.

This module provides an interactive command-line interface for all EuclidPolish operations.
"""

import glob
import os
import sys
import traceback

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from astropy.io import fits
from tf_keras.losses import MeanAbsoluteError
from tf_keras.optimizers.schedules import PiecewiseConstantDecay
from tqdm import tqdm
from questionary import select, confirm


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
from euclid_polish.sky.types import SkyImage
from euclid_polish.euclid.types import PSF
from euclid_polish.training import Trainer, EuclidDataset
from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.visualization import BaseVisualizer
from euclid_polish.euclid import estimate_fwhm
from euclid_polish.visualization.methods import draw_clean_image, draw_dirty_image, draw_clean_dirty_pair
from euclid_polish.sky.tfrecord import read_tfrecord, read_skyimages, shard_paths, open_shard_writers, write_skyimages


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
            epsf_pixel_scale = Config.VIS_PIXEL_SCALE_ARCSEC / config.oversampling
            psf = extractor.to_psf(epsf_pixel_scale)
            fits_path = psf.save(psf_dir)

            print(f"\n✓ PSF extraction completed!")
            print(f"  FITS file: {fits_path}")

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
        default_psf = os.path.join(Config.EUCLID_PSF_DIR, Config.DEFAULT_PSF_FITS_FILENAME)
        psf_path = input(f"PSF file path (default {default_psf}): ").strip() or default_psf

        if not os.path.exists(psf_path):
            print(f"\n✗ PSF file not found: {psf_path}")
            return

        # Load PSF with metadata
        print(f"\nLoading PSF from {psf_path}...")
        psf_kernel = PSF.from_fits(psf_path)

        print(f"  PSF shape: {psf_kernel.shape}, pixel_scale: {psf_kernel.pixel_scale} arcsec/pix")

        # Discover which subsets have clean TFRecords
        subsets_to_run = []
        for subset in ("train", "validate"):
            shards = sorted(glob.glob(os.path.join(Config.RECORDS_DIR, f"clean_{subset}-*.tfrecord")))
            if shards:
                n_images = sum(1 for _ in tf.data.TFRecordDataset(shards))
                subsets_to_run.append((subset, shards, n_images))
            else:
                print(f"  ⚠️  Skipping {subset}: no clean_{subset}-*.tfrecord in {Config.RECORDS_DIR}")

        if not subsets_to_run:
            print(f"\n✗ No clean TFRecords found in {Config.RECORDS_DIR}")
            return

        print(f"\nSource: {Config.RECORDS_DIR}")
        for subset, _, n_images in subsets_to_run:
            n_dirty = Config.shard_count(n_images)
            print(f"  clean_{subset}-*  →  dirty_{subset}-*  ({n_images} images → {n_dirty} shard{'s' if n_dirty > 1 else ''})")
        total = sum(n for _, _, n in subsets_to_run)

        if not confirm(f"\nConvolve {total} images (train+validate) to dirty LR?", default=True).ask():
            return

        convolver = PSFConvolution(ConvolutionConfig(rebin_factor=Config.DEFAULT_REBIN_FACTOR, add_noise=False, normalize=False))

        all_viz_pairs = []

        for subset, clean_shards, n_images in subsets_to_run:
            n_dirty_shards = Config.shard_count(n_images)
            dirty_writers = open_shard_writers(
                shard_paths(Config.RECORDS_DIR, f'dirty_{subset}', n_dirty_shards)
            )
            n_ok = n_err = 0
            viz_pairs = []

            try:
                for raw_record in tqdm(tf.data.TFRecordDataset(clean_shards),
                                       total=n_images, desc=f"Convolving {subset}"):
                    try:
                        hr_image = SkyImage.from_tfrecord(raw_record)
                        lr_image, _ = convolver.process_hr_to_lr(hr_image, psf_kernel)
                        dirty_writers[n_ok % n_dirty_shards].write(lr_image.to_tfrecord())

                        if len(viz_pairs) < 10:
                            viz_pairs.append((hr_image.data, lr_image.data, hr_image.index or n_ok))
                        n_ok += 1

                    except Exception as e:
                        n_err += 1
                        tqdm.write(f"  ✗ Skipping record (error: {e})")
            finally:
                for w in dirty_writers:
                    w.close()

            print(f"  ✓ {subset}: {n_ok} ok, {n_err} skipped → dirty_{subset}-* ({n_dirty_shards} shard{'s' if n_dirty_shards > 1 else ''})")
            all_viz_pairs.append((subset, viz_pairs))

        all_pairs = [pair for _, pairs in all_viz_pairs for pair in pairs]
        if all_pairs:
            if confirm("\nVisualize sample HR/LR pairs?", default=True).ask():
                chosen = np.random.default_rng(42).permutation(len(all_pairs))[:5]
                os.makedirs(Config.VIS_DIRTY_DIR, exist_ok=True)
                for i in chosen:
                    hr_data, lr_data, index = all_pairs[i]
                    draw_clean_dirty_pair(hr_data, lr_data,
                                          os.path.join(Config.VIS_DIRTY_DIR, f'pair_{index:05d}.png'),
                                          index=index)
                print(f"  ✓ {len(chosen)} pair plots → {Config.VIS_DIRTY_DIR}")

    def _generate_clean_data(self):
        """Generate clean sky data."""
        catalog_path = input(f"Enter path to COSMOS catalog (default {Config.DEFAULT_COSMOS_CATALOG_DIR}): ").strip() or Config.DEFAULT_COSMOS_CATALOG_DIR

        if not os.path.exists(catalog_path):
            print(f"\n✗ Catalog not found: {catalog_path}")
            return

        ntrain     = input(f"Number of training images (default {Config.DEFAULT_NIMAGES}): ").strip() or str(Config.DEFAULT_NIMAGES)
        nvalid     = input(f"Number of validation images (default {Config.DEFAULT_NIMAGES // 5}): ").strip() or str(Config.DEFAULT_NIMAGES // 5)
        pixel_scale = input(f"Pixel scale in arcsec (default {Config.DEFAULT_PIXEL_SCALE}): ").strip() or str(Config.DEFAULT_PIXEL_SCALE)
        image_size  = input(f"Image size in pixels (default {Config.DEFAULT_IMAGE_SIZE}): ").strip() or str(Config.DEFAULT_IMAGE_SIZE)

        try:
            ntrain_val      = int(ntrain)
            nvalid_val      = int(nvalid)
            pixel_scale_val = float(pixel_scale)
            image_size_val  = int(image_size)
        except ValueError:
            print("\n✗ Invalid input: values must be numbers")
            return

        print(f"\nConfiguration:")
        print(f"  COSMOS catalog:    {catalog_path}")
        print(f"  Training images:   {ntrain_val}")
        print(f"  Validation images: {nvalid_val}")
        print(f"  Pixel scale:       {pixel_scale_val} arcsec/pix")
        print(f"  Image size:        {image_size_val} x {image_size_val}")
        n_train_shards = Config.shard_count(ntrain_val)
        n_valid_shards = Config.shard_count(nvalid_val)
        print(f"  Output (TFRecord): {Config.RECORDS_DIR}/")
        print(f"    clean_train-*    ({n_train_shards} shard{'s' if n_train_shards > 1 else ''})")
        print(f"    clean_validate-* ({n_valid_shards} shard{'s' if n_valid_shards > 1 else ''})")

        if not confirm("\nGenerate clean sky data?", default=True).ask():
            return

        try:
            catalog, catalog_file, catalog_dir = CleanSkyGenerator.resolve_cosmos_catalog(catalog_path)
            print(f"\nCOSMOS catalog loaded: {catalog.nobjects} objects")

            config = GeneratorConfig(
                image_size=image_size_val,
                pixel_scale=pixel_scale_val,
            )
            generator = CleanSkyGenerator(config)

            print("\nGenerating training set...")
            images_train, _ = generator.generate(
                catalog=catalog,
                subset='train',
                nimages=ntrain_val,
                nstart=0,
            )

            print("\nGenerating validation set...")
            images_valid, _ = generator.generate(
                catalog=catalog,
                subset='validate',
                nimages=nvalid_val,
                nstart=ntrain_val,  # different seeds from training
            )

            write_skyimages(images_train, 'clean_train', n_train_shards)
            print(f"  ✓ clean_train-* → {Config.RECORDS_DIR} ({n_train_shards} shard{'s' if n_train_shards > 1 else ''})")

            write_skyimages(images_valid, 'clean_validate', n_valid_shards)
            print(f"  ✓ clean_validate-* → {Config.RECORDS_DIR} ({n_valid_shards} shard{'s' if n_valid_shards > 1 else ''})")

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
                    {"name": "🔬 Reconstruct image (inference)", "value": "reconstruct"},
                    {"name": "🔄 Inspect checkpoints", "value": "inspect"},
                    {"name": "🔙 Back to main menu", "value": "back"},
                ]
            ).ask()

            if choice == "back" or choice is None:
                break

            if choice == "train":
                self._train_model()
            elif choice == "evaluate":
                self._evaluate_model()
            elif choice == "reconstruct":
                self._reconstruct_image()
            elif choice == "inspect":
                self._inspect_checkpoints()

    def _train_model(self):
        """Train WDSR model."""
        scale = input(f"Scale factor (default {Config.DEFAULT_REBIN_FACTOR}): ").strip() or str(Config.DEFAULT_REBIN_FACTOR)
        num_res_blocks = input(f"Number of residual blocks (default {Config.DEFAULT_NUM_RES_BLOCKS}): ").strip() or str(Config.DEFAULT_NUM_RES_BLOCKS)
        checkpoint_dir = input(f"Checkpoint directory (default {Config.DEFAULT_CHECKPOINT_DIR}): ").strip() or Config.DEFAULT_CHECKPOINT_DIR
        steps = input(f"Training steps (default {Config.DEFAULT_TRAIN_STEPS}): ").strip() or str(Config.DEFAULT_TRAIN_STEPS)
        batch_size = input(f"Batch size (default {Config.DEFAULT_BATCH_SIZE}): ").strip() or str(Config.DEFAULT_BATCH_SIZE)
        evaluate_every = input(f"Evaluate every N steps (default {Config.DEFAULT_EVALUATE_EVERY}): ").strip() or str(Config.DEFAULT_EVALUATE_EVERY)

        try:
            scale_val = int(scale)
            num_res_blocks_val = int(num_res_blocks)
            steps_val = int(steps)
            batch_size_val = int(batch_size)
            evaluate_every_val = int(evaluate_every)
        except ValueError:
            print("\n✗ Invalid input: all values must be integers")
            return

        # Check that TFRecords exist
        clean_train = glob.glob(os.path.join(Config.RECORDS_DIR, "clean_train-*.tfrecord"))
        dirty_train = glob.glob(os.path.join(Config.RECORDS_DIR, "dirty_train-*.tfrecord"))

        if not clean_train or not dirty_train:
            print(f"\n✗ Training data not found in {Config.RECORDS_DIR}")
            print("  Run clean sky generation and HR→LR convolution first.")
            return

        dirty_valid = glob.glob(os.path.join(Config.RECORDS_DIR, "dirty_validate-*.tfrecord"))
        if not dirty_valid:
            print(f"\n⚠️  No validation shards in {Config.RECORDS_DIR} — will train without validation")

        print(f"\nConfiguration:")
        print(f"  Scale: {scale_val}x")
        print(f"  Residual blocks: {num_res_blocks_val}")
        print(f"  Training steps: {steps_val}")
        print(f"  Batch size: {batch_size_val}")
        print(f"  Evaluate every: {evaluate_every_val} steps")
        print(f"  Records: {Config.RECORDS_DIR}")
        print(f"  Checkpoint directory: {checkpoint_dir}")

        if confirm("\nStart training?", default=True).ask():
            print("\n⚠️  Training will run until interrupted (Ctrl+C) or completion")

            try:
                model = wdsr(scale=scale_val, num_res_blocks=num_res_blocks_val, nchan=1)

                learning_rate = PiecewiseConstantDecay(boundaries=[200000], values=[1e-3, 5e-4])
                trainer = Trainer(
                    model=model,
                    loss=MeanAbsoluteError(),
                    learning_rate=learning_rate,
                    checkpoint_dir=checkpoint_dir,
                )

                train_loader = EuclidDataset(
                    scale=scale_val,
                    subset='train',
                )

                valid_loader = EuclidDataset(
                    scale=scale_val,
                    subset='validate',
                )

                # Create datasets
                train_ds = train_loader.dataset(batch_size=batch_size_val, random_transform=True)
                valid_ds = valid_loader.dataset(batch_size=1, random_transform=False, repeat_count=1)

                # Train
                print("\nStarting training...")
                trainer.train(train_ds, valid_ds, steps=steps_val, evaluate_every=evaluate_every_val)

                # Restore and evaluate
                trainer.restore()
                psnr = trainer.evaluate(valid_ds)
                print(f'\nFinal PSNR = {psnr.numpy():.3f}')

                # Offer to export weights
                if confirm("\nExport trained weights to .h5 file?", default=True).ask():
                    weights_dir = os.path.dirname(checkpoint_dir) or "."
                    weights_path = os.path.join(weights_dir, f"wdsr_x{scale_val}.h5")
                    trainer.model.save_weights(weights_path)
                    print(f"  ✓ Weights saved to: {weights_path}")

                print("\n✓ Training completed!")

            except KeyboardInterrupt:
                print("\n\n⚠️  Training interrupted by user")
            except Exception as e:
                print(f"\n✗ Training failed: {e}")
                traceback.print_exc()

    def _evaluate_model(self):
        """Evaluate a trained model on the validation set."""
        checkpoint_dir = input(f"Checkpoint directory (default {Config.DEFAULT_CHECKPOINT_DIR}): ").strip() or Config.DEFAULT_CHECKPOINT_DIR
        scale = input(f"Scale factor (default {Config.DEFAULT_REBIN_FACTOR}): ").strip() or str(Config.DEFAULT_REBIN_FACTOR)
        num_res_blocks = input(f"Number of residual blocks (default {Config.DEFAULT_NUM_RES_BLOCKS}): ").strip() or str(Config.DEFAULT_NUM_RES_BLOCKS)

        try:
            scale_val = int(scale)
            num_res_blocks_val = int(num_res_blocks)
        except ValueError:
            print("\n✗ Invalid input: scale and num_res_blocks must be integers")
            return

        if not tf.train.latest_checkpoint(checkpoint_dir):
            print(f"\n✗ No checkpoints found in {checkpoint_dir}")
            return

        # Check that validation TFRecords exist
        dirty_valid = glob.glob(os.path.join(Config.RECORDS_DIR, "dirty_validate-*.tfrecord"))
        if not dirty_valid:
            print(f"\n✗ No validation shards found in {Config.RECORDS_DIR}")
            return

        try:
            from euclid_polish.training.inference import load_model_from_checkpoint

            print(f"\nLoading model from {checkpoint_dir}...")
            model = load_model_from_checkpoint(checkpoint_dir, scale_val, num_res_blocks_val)

            valid_loader = EuclidDataset(scale=scale_val, subset='validate')
            valid_ds = valid_loader.dataset(batch_size=1, random_transform=False, repeat_count=1)

            print("Evaluating on validation set...")
            psnr = Trainer(model=model, checkpoint_dir=checkpoint_dir).evaluate(valid_ds)
            print(f"\n✓ Validation PSNR = {psnr.numpy():.3f}")

        except Exception as e:
            print(f"\n✗ Evaluation failed: {e}")
            traceback.print_exc()

    def _inspect_checkpoints(self):
        """List available checkpoints in a directory."""
        checkpoint_dir = input(f"Checkpoint directory (default {Config.DEFAULT_CHECKPOINT_DIR}): ").strip() or Config.DEFAULT_CHECKPOINT_DIR

        if not os.path.isdir(checkpoint_dir):
            print(f"\n✗ Directory not found: {checkpoint_dir}")
            return

        ckpt_state = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt_state is None or not ckpt_state.all_model_checkpoint_paths:
            print(f"\n✗ No checkpoints found in {checkpoint_dir}")
            return

        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print(f"\nCheckpoints in {checkpoint_dir}:\n")

        for path in ckpt_state.all_model_checkpoint_paths:
            marker = " ← latest" if path == latest else ""
            print(f"  {os.path.basename(path)}{marker}")

        print(f"\n  Total: {len(ckpt_state.all_model_checkpoint_paths)} checkpoint(s)")

    def _reconstruct_image(self):
        """Apply super-resolution to a single LR image."""
        lr_path = input("Path to LR image (.npy or .png): ").strip()
        if not lr_path or not os.path.exists(lr_path):
            print(f"\n✗ File not found: {lr_path}")
            return

        source = select(
            "Load model from:",
            choices=[
                {"name": "Checkpoint directory", "value": "checkpoint"},
                {"name": ".h5 weights file", "value": "weights"},
            ]
        ).ask()

        scale = input(f"Scale factor (default {Config.DEFAULT_REBIN_FACTOR}): ").strip() or str(Config.DEFAULT_REBIN_FACTOR)
        num_res_blocks = input(f"Number of residual blocks (default {Config.DEFAULT_NUM_RES_BLOCKS}): ").strip() or str(Config.DEFAULT_NUM_RES_BLOCKS)

        try:
            scale_val = int(scale)
            num_res_blocks_val = int(num_res_blocks)
        except ValueError:
            print("\n✗ Invalid input: scale and num_res_blocks must be integers")
            return

        try:
            from euclid_polish.training.inference import (
                load_model_from_checkpoint,
                load_model_from_weights,
                reconstruct,
                plot_reconstruction,
            )

            if source == "checkpoint":
                ckpt_dir = input(f"Checkpoint directory (default {Config.DEFAULT_CHECKPOINT_DIR}): ").strip() or Config.DEFAULT_CHECKPOINT_DIR
                if not tf.train.latest_checkpoint(ckpt_dir):
                    print(f"\n✗ No checkpoints found in {ckpt_dir}")
                    return
                print(f"\nLoading model from checkpoint {ckpt_dir}...")
                model = load_model_from_checkpoint(ckpt_dir, scale_val, num_res_blocks_val)
            else:
                weights_path = input("Path to .h5 weights file: ").strip()
                if not weights_path or not os.path.exists(weights_path):
                    print(f"\n✗ Weights file not found: {weights_path}")
                    return
                print(f"\nLoading model from {weights_path}...")
                model = load_model_from_weights(weights_path, scale_val, num_res_blocks_val)

            hr_path = input("Path to HR ground truth (optional, press Enter to skip): ").strip() or None
            hr_data = None
            if hr_path:
                if not os.path.exists(hr_path):
                    print(f"  ⚠️  HR file not found, skipping: {hr_path}")
                elif hr_path.endswith(".npy"):
                    hr_data = np.load(hr_path)
                elif hr_path.endswith(".png"):
                    raw = tf.io.read_file(hr_path)
                    hr_data = tf.image.decode_png(raw, dtype=tf.uint16).numpy().astype(np.float32)
                    if hr_data.ndim == 3 and hr_data.shape[-1] == 1:
                        hr_data = hr_data[..., 0]

            print("Running super-resolution...")
            lr_data, sr_data = reconstruct(model, lr_path)

            os.makedirs(Config.VIS_RECONSTRUCTION_DIR, exist_ok=True)
            basename = os.path.splitext(os.path.basename(lr_path))[0]
            output_path = os.path.join(Config.VIS_RECONSTRUCTION_DIR, f"reconstruct_{basename}.png")

            plot_reconstruction(lr_data, sr_data, hr_data=hr_data, output_path=output_path)
            print(f"\n✓ Reconstruction saved to: {output_path}")

        except Exception as e:
            print(f"\n✗ Reconstruction failed: {e}")
            traceback.print_exc()

    def _ask_clip_percentile(self) -> float:
        """Prompt the user for a clip percentile, defaulting to Config.DEFAULT_CLIP_PERCENTILE."""
        default = Config.DEFAULT_CLIP_PERCENTILE
        raw = input(f"Clip percentile for visualization (default {default}): ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
            if not (0.0 < value <= 100.0):
                raise ValueError
            return value
        except ValueError:
            print(f"  ⚠️  Invalid value, using default ({default})")
            return default

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

        clip_percentile = self._ask_clip_percentile()

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
        vis_dir = Config.VIS_CUTOUTS_DIR
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

                visualizer = BaseVisualizer(clip_percentile=clip_percentile, rows=1, cols=3, figsize=(18, 6))
                visualizer.add_scale_panel(data)
                visualizer.add_scale_panel(data, log_scale=True)

                mag_str = f"{star['magnitude']:.2f}" if star.get('magnitude') else "N/A"
                visualizer.add_statistics_panel(data, {
                    'title': 'Star Information:',
                    'stats': {
                        'ID': f"{star['id']:04d}",
                        'RA': f"{star['ra']:.6f}°",
                        'Dec': f"{star['dec']:.6f}°",
                        'Magnitude': mag_str,
                    },
                    'include_data_stats': True,
                })

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

        # Look for PSF FITS file
        psf_file = os.path.join(psf_dir, Config.DEFAULT_PSF_FITS_FILENAME)
        if not os.path.exists(psf_file):
            print(f"\n✗ PSF file not found: {psf_file}")
            return

        clip_percentile = self._ask_clip_percentile()

        # Load PSF
        print(f"\nLoading PSF from {psf_file}...")
        psf = PSF.from_fits(psf_file)
        psf_data = psf.data

        # Create visualization (linear, log, stats)
        visualizer = BaseVisualizer(clip_percentile=clip_percentile, rows=1, cols=3, figsize=(18, 6))

        visualizer.add_scale_panel(psf_data, title_suffix='\nEuclid VIS PSF')
        visualizer.add_scale_panel(psf_data, log_scale=True)

        center_y, center_x = psf_data.shape[0] // 2, psf_data.shape[1] // 2
        x_slice = psf_data[center_y, :]
        y_slice = psf_data[:, center_x]
        fwhm_y = estimate_fwhm(x_slice)
        fwhm_x = estimate_fwhm(y_slice)
        ellipticity = abs(fwhm_y - fwhm_x) / ((fwhm_y + fwhm_x) / 2) if (fwhm_y + fwhm_x) > 0 else 0

        stats = {
            'Pixel Scale': f"{psf.pixel_scale} arcsec/pix",
            'Total Flux': f"{np.sum(psf_data):.6f}",
            'Center': f"({center_x}, {center_y})",
            'FWHM X': f"{fwhm_x:.2f} pixels",
            'FWHM Y': f"{fwhm_y:.2f} pixels",
            'Ellipticity': f"{ellipticity:.4f}",
        }
        if psf.oversampling is not None:
            stats['Oversampling'] = f"{psf.oversampling}x"
        if psf.fwhm_arcsec is not None:
            stats['FWHM'] = f"{psf.fwhm_arcsec:.3f} arcsec"

        visualizer.add_statistics_panel(psf_data, {
            'title': 'PSF Statistics:',
            'stats': stats,
            'include_data_stats': True,
        })

        plt.suptitle('Euclid VIS PSF', fontsize=16, y=1.02)

        # Save figure — name after the PSF directory
        os.makedirs(Config.VIS_PSF_DIR, exist_ok=True)
        dir_name = os.path.basename(os.path.normpath(psf_dir))
        output_path = os.path.join(Config.VIS_PSF_DIR, f'{dir_name}.png')
        visualizer.save_figure(output_path)

        print(f"\n✓ PSF visualization saved to: {output_path}")

    def _visualize_training_data(self):
        """Visualize training data (clean, dirty, or paired)."""
        mode = select(
            "What to visualize:",
            choices=[
                {"name": "Clean (HR) images", "value": "clean"},
                {"name": "Dirty (LR) images", "value": "dirty"},
                {"name": "Clean + Dirty pairs", "value": "pair"},
            ]
        ).ask()

        num_images_input = input("Number of images to visualize (default 5): ").strip() or "5"
        try:
            num_images = int(num_images_input)
        except ValueError:
            print("\n✗ Invalid input: number of images must be an integer")
            return

        clip_percentile = self._ask_clip_percentile()

        # Glob across both train and validate subsets
        clean_pattern = os.path.join(Config.RECORDS_DIR, "clean_*-*.tfrecord")
        dirty_pattern = os.path.join(Config.RECORDS_DIR, "dirty_*-*.tfrecord")

        need_clean = mode in ("clean", "pair")
        need_dirty = mode in ("dirty", "pair")

        if need_clean and not glob.glob(clean_pattern):
            print(f"\n✗ No clean TFRecord shards found in {Config.RECORDS_DIR}")
            return
        if need_dirty and not glob.glob(dirty_pattern):
            print(f"\n✗ No dirty TFRecord shards found in {Config.RECORDS_DIR}")
            return

        try:
            if mode == "clean":
                vis_dir = Config.VIS_CLEAN_DIR
                os.makedirs(vis_dir, exist_ok=True)
                images = read_skyimages(clean_pattern, num_images=num_images, mode='random')
                for img in images:
                    path = os.path.join(vis_dir, f'clean_{img.index:04d}.png')
                    draw_clean_image(img.data, path, index=img.index, clip_percentile=clip_percentile)
                print(f"\n✓ {len(images)} clean images → {vis_dir}")

            elif mode == "dirty":
                vis_dir = Config.VIS_DIRTY_DIR
                os.makedirs(vis_dir, exist_ok=True)
                images = read_skyimages(dirty_pattern, num_images=num_images, mode='random')
                for img in images:
                    path = os.path.join(vis_dir, f'dirty_{img.index:04d}.png')
                    draw_dirty_image(img.data, path, index=img.index, clip_percentile=clip_percentile)
                print(f"\n✓ {len(images)} dirty images → {vis_dir}")

            elif mode == "pair":
                vis_dir = Config.VIS_DIRTY_DIR
                os.makedirs(vis_dir, exist_ok=True)
                n_drawn = 0

                # Collect matched pairs per subset to avoid index-space collisions.
                # Both subsets independently use 0-based indices, so matching across
                # subsets with a combined dirty dict would pair wrong scenes.
                all_pairs: list[tuple[SkyImage, SkyImage]] = []
                for subset in ("train", "validate"):
                    clean_sub = os.path.join(Config.RECORDS_DIR, f"clean_{subset}-*.tfrecord")
                    dirty_sub = os.path.join(Config.RECORDS_DIR, f"dirty_{subset}-*.tfrecord")
                    if not glob.glob(clean_sub) or not glob.glob(dirty_sub):
                        continue
                    dirty_by_index = {
                        img.index: img
                        for img in read_skyimages(dirty_sub, num_images=9999)
                    }
                    for hr in read_skyimages(clean_sub, num_images=9999):
                        lr = dirty_by_index.get(hr.index)
                        if lr is not None:
                            all_pairs.append((hr, lr))

                if not all_pairs:
                    print("\n✗ No matched clean/dirty pairs found")
                    return

                rng = np.random.default_rng(42)
                chosen = rng.choice(len(all_pairs), min(num_images, len(all_pairs)), replace=False)
                for i in chosen:
                    hr, lr = all_pairs[i]
                    path = os.path.join(vis_dir, f'pair_{hr.index:04d}.png')
                    draw_clean_dirty_pair(hr.data, lr.data, path, index=hr.index, clip_percentile=clip_percentile)
                    n_drawn += 1
                print(f"\n✓ {n_drawn} pair plots → {vis_dir}")

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
