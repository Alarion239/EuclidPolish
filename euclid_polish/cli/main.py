#!/usr/bin/env python3
"""
Unified Interactive CLI for EuclidPolish.

This module provides an interactive command-line interface for all EuclidPolish operations.
"""

import contextlib
import getpass
import glob
import os
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from astropy.io import fits
from astroquery.esa.euclid import Euclid
from questionary import checkbox, confirm, select
from tqdm import tqdm

from euclid_polish.catalog import (
    CatalogObject,
    DownloadConfig,
    EuclidAuthError,
    EuclidCatalog,
    FitsValidator,
)
from euclid_polish.catalog.catalog_object import merge_new, summarize
from euclid_polish.cli.inference_ops import fetch_and_superresolve, reconstruct_and_render
from euclid_polish.cli.utils import (
    print_cancelled,
    print_error,
    print_header,
    print_success,
    validate_dec,
    validate_positive_number,
    validate_ra,
)
from euclid_polish.config import Config
from euclid_polish.eval.subsets import eval_subset
from euclid_polish.image import Image, ImageSet
from euclid_polish.image.tfio import (
    open_writer,
    read_images,
    tfrecord_path,
)
from euclid_polish.model import Model
from euclid_polish.psf import PSF
from euclid_polish.psf import estimate_fwhm_pixels_1d as estimate_fwhm
from euclid_polish.psf.psf_extractor import PSFExtractionConfig, PSFExtractor
from euclid_polish.psf.psf_library import (
    load_all_band_psf_sets,
    psf_inventory,
)
from euclid_polish.sky.generation.cosmos_tng_prior import CosmosTngPrior
from euclid_polish.sky.generation.sky_simulator import (
    SkySimulator,
    SkySimulatorConfig,
)
from euclid_polish.sky.observation.observation_simulator import (
    ObservationSimulator,
    ObservationSimulatorConfig,
)
from euclid_polish.training import Trainer
from euclid_polish.training.inference import (
    load_model_from_checkpoint,
    load_model_from_weights,
    plot_reconstruction,
    reconstruct,
)
from euclid_polish.training.log_plot import (
    default_log_path,
    plot_training_log,
)
from euclid_polish.visualization import BaseVisualizer
from euclid_polish.visualization.methods import (
    draw_clean_dirty_pair,
    draw_clean_image,
    draw_dirty_image,
    draw_star_positions,
)


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
        #: The logged-in Euclid client (set by the login menu), or None.
        self._euclid = None

    def _euclid_client(self):
        """The logged-in client, or an unauthenticated one that reuses whatever
        astroquery session is active (so queries work after an external login)."""
        return self._euclid or EuclidCatalog._unauthenticated()

    def _ingest_query(self, output_dir, candidates, *, limit=None):
        """Dedupe queried ``CatalogObject``s into the on-disk catalog + persist."""
        path = os.path.join(output_dir, Config.CATALOG_FILE)
        existing = CatalogObject.read(path)
        res = merge_new(existing, candidates, limit=limit)
        CatalogObject.write(existing, path)
        return {**res, "total": len(existing),
                "message": f"Added {res['added']} stars → "
                           f"{len(existing)} total in catalog"}

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
            choices = [
                {"name": "📊 Show catalog info", "value": "info"},
                {"name": "🔍 Query bright stars catalog (region)", "value": "query"},
                {"name": "🌟 Query brightest N stars", "value": "query_brightest"},
                {"name": "⬇️  Download cutouts", "value": "download"},
                {"name": "✨ Extract PSF (per band)", "value": "extract_psf"},
                {"name": "📂 Show per-band PSF inventory", "value": "psf_inventory"},
                {"name": "✔️  Check cutouts integrity", "value": "check"},
            ]
            if self._euclid is not None:
                user = self._euclid.user or "authenticated"
                choices.append({"name": f"🔓 Logout ({user})", "value": "logout"})
            else:
                choices.append({"name": "🔐 Login to Euclid archive", "value": "login"})
            choices.append({"name": "🔙 Back to main menu", "value": "back"})

            choice = select("🔭 Euclid Operations - Select an action:", choices=choices).ask()

            if choice == "back" or choice is None:
                break

            if choice == "info":
                self._show_catalog_info()
            elif choice == "query":
                self._query_catalog()
            elif choice == "query_brightest":
                self._query_brightest_stars()
            elif choice == "download":
                self._download_cutouts()
            elif choice == "extract_psf":
                self._extract_psf()
            elif choice == "psf_inventory":
                self._show_psf_inventory()
            elif choice == "check":
                self._check_integrity()
            elif choice == "login":
                self._login_euclid()
            elif choice == "logout":
                self._logout_euclid()

    def _show_psf_inventory(self):
        """Display which bands have empirical ePSF files on disk."""
        psf_dir = input(
            f"PSF directory (default {Config.EUCLID_PSF_DIR}): "
        ).strip() or Config.EUCLID_PSF_DIR
        inv = psf_inventory(psf_dir=psf_dir)
        print_header(f"PSF inventory @ {psf_dir}")
        for name, path in inv.items():
            tag = "✓ empirical" if path else "✗ Gaussian fallback"
            print(f"  {name:5s}  {tag}  {path or ''}")

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

        catalog_path = os.path.join(output_dir, Config.CATALOG_FILE)

        if not os.path.exists(catalog_path):
            print(f"\n⚠️  Catalog not found at {catalog_path}")
            if confirm("Query the catalog first?", default=True).ask():
                self._query_catalog()
            return

        summary = summarize(CatalogObject.read(catalog_path))

        print_header("📊 Star Catalog Summary")
        print(f"Total stars:        {summary['total']}")
        print(f"Valid:              {summary['valid']} ✓")
        print(f"Corrupted:          {summary['corrupted']} 🔴")
        print(f"Failed downloads:   {summary['failed']} ❌")
        print(f"Pending:            {summary['pending']} ⏳")

        if summary.get('mag_min'):
            print(f"\nMagnitude range:   {summary['mag_min']:.2f} - {summary['mag_max']:.2f}")

        print(f"\nNext ID:            {summary['next_id']}")
        print(f"Catalog file:       {catalog_path}")

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
        magnitude_min_raw = input(
            "Bright-end cutoff — min magnitude (Enter to skip; e.g. 15 to drop saturating stars): "
        ).strip()

        # Validate inputs
        ra_valid = validate_ra(ra)
        if ra_valid is not True:
            print(f"\n✗ {ra_valid}")
            return

        dec_valid = validate_dec(dec)
        if dec_valid is not True:
            print(f"\n✗ {dec_valid}")
            return

        radius_valid = validate_positive_number(radius, "Radius")
        if radius_valid is not True:
            print(f"\n✗ {radius_valid}")
            return

        mag_valid = validate_positive_number(magnitude, "Magnitude limit")
        if mag_valid is not True:
            print(f"\n✗ {mag_valid}")
            return

        # Parse values
        ra_val = float(ra)
        dec_val = float(dec)
        radius_val = float(radius)
        mag_val = float(magnitude)
        mag_min_val = None
        if magnitude_min_raw:
            mmv = validate_positive_number(magnitude_min_raw, "Min magnitude")
            if mmv is not True:
                print(f"\n✗ {mmv}")
                return
            mag_min_val = float(magnitude_min_raw)

        if confirm("\nQuery Euclid catalog with these parameters?", default=True).ask():
            window = f"mag<{mag_val}"
            if mag_min_val is not None:
                window = f"{mag_min_val} < mag < {mag_val}"
            print(f"\nQuerying RA={ra_val:.1f}°, Dec={dec_val:.1f}°, radius={radius_val}°, {window}...")

            try:
                candidates = self._euclid_client().query_bright_stars(
                    100000,
                    ra=ra_val,
                    dec=dec_val,
                    radius=radius_val,
                    magnitude_limit=mag_val,
                    magnitude_min=mag_min_val,
                )
                result = self._ingest_query(output_dir, candidates)

                print(f"\n{result['message']}")
                if result['skipped'] > 0:
                    print(f"  (Skipped {result['skipped']} duplicate stars)")

                if result['added'] > 0:
                    print("\n✓ Query completed successfully!")
                elif result['added'] == 0 and result['total'] > 0:
                    print("\n✓ Catalog already has the requested stars")

                # Auto-update star positions plot
                self._visualize_star_positions(output_dir=output_dir)

            except Exception as e:
                print(f"\n✗ Query failed: {e}")
                traceback.print_exc()

    def _download_cutouts(self):
        """Download Euclid cutouts for one or more bands."""
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
        catalog_path = os.path.join(output_dir, Config.CATALOG_FILE)
        if not os.path.exists(catalog_path):
            print(f"\n✗ Catalog not found at {catalog_path}")
            if confirm("Query the catalog first?", default=True).ask():
                self._query_catalog()
            return

        # --- Band selection (multi-select; default = all 4) ---
        band_options = [
            {"name": f"{b.name}  ({b.archive_instrument}"
                      + (f"/{b.archive_filter}" if b.archive_filter else "")
                      + f", {b.pixel_scale_lr_arcsec}\"/pix)",
             "value": b.name,
             "checked": True}
            for b in Config.BANDS
        ]
        selected_bands = checkbox(
            "Select bands to download (space to toggle, enter to confirm):",
            choices=band_options,
        ).ask()
        if not selected_bands:
            print("\n(no bands selected — aborting)")
            return

        # Size is requested as *VIS pixels* (0.10"/pix), so it maps to a
        # fixed angular field for every band. Each band's native pixel
        # count is derived internally — NISP fetches fewer pixels than
        # VIS to cover the same patch of sky.
        cutout_size_input = input(
            f"Enter cutout size in VIS pixels (0.10\"/pix; "
            f"default {Config.DEFAULT_CUTOUT_SIZE}): "
        ).strip()
        cutout_size_vis = int(cutout_size_input) if cutout_size_input else Config.DEFAULT_CUTOUT_SIZE
        arcsec_side = cutout_size_vis * Config.BAND_VIS.pixel_scale_lr_arcsec
        per_band_native = {
            name: Config.get_band(name).cutout_size_for_arcsec(arcsec_side)
            for name in selected_bands
        }
        print(f"  → angular field side = {arcsec_side:.2f}\"")
        print("  → native pixel counts: " + ", ".join(
            f"{name}={n}" for name, n in per_band_native.items()))

        default_workers = DownloadConfig.max_workers
        workers_input = input(
            f"Parallel workers per band (default {default_workers}, ESA fair-use ~8): "
        ).strip()
        try:
            max_workers = int(workers_input) if workers_input else default_workers
        except ValueError:
            print(f"  ⚠️  Invalid workers value, using default ({default_workers})")
            max_workers = default_workers

        # --- Pre-flight summary: pending per (band, native_size) ---
        objects = CatalogObject.read(catalog_path)
        total = len(objects)
        print(f"\n📊 Catalog: {total} stars  |  field = {arcsec_side:.2f}\" "
              f"(= {cutout_size_vis} VIS px)")
        any_pending = False
        for band_name in selected_bands:
            native_size = per_band_native[band_name]
            v = sum(1 for o in objects if o.is_valid(native_size, band=band_name))
            c = sum(1 for o in objects if o.is_corrupted(native_size, band=band_name))
            f = sum(1 for o in objects if o.is_download_failed(native_size, band=band_name))
            p = total - v - c - f
            mark = "—" if p <= 0 else f"{p} to fetch"
            print(f"  {band_name:5s}  native_size={native_size:4d}  "
                  f"valid={v:5d}  corrupted={c:3d}  failed={f:3d}  pending={mark}")
            if p > 0:
                any_pending = True

        if not any_pending:
            print("\n✓ Nothing to download — every selected band has all stars accounted for "
                  "at this field size")
            return

        if not confirm(
            f"\nDownload cutouts for selected bands ({', '.join(selected_bands)}, "
            f"{arcsec_side:.2f}\" field, {max_workers} parallel/band)?",
            default=True,
        ).ask():
            return

        # --- Loop over bands; the shared object list accumulates flags ---
        try:
            eclient = EuclidCatalog._unauthenticated()
            band_results = {}
            for band_name in selected_bands:
                native_size = per_band_native[band_name]
                print(f"\n=== {band_name}  (native_size = {native_size}) ===")
                cfg = DownloadConfig.for_band(
                    band_name,
                    cutout_size_vis_pixels=cutout_size_vis,
                    max_workers=max_workers,
                )
                result = eclient.download_cutouts(
                    objects, output_dir, cfg, show_progress=True)
                band_results[band_name] = result
                print(f"  → downloaded {result['downloaded']}, "
                      f"valid={result['valid']}, corrupted={result['corrupted']}, "
                      f"failed={result.get('failed', 0)}")

            # Summary
            print(f"\n✓ Multi-band download complete ({arcsec_side:.2f}\" field):")
            for band_name, r in band_results.items():
                print(f"  {band_name:5s}  size={per_band_native[band_name]:4d}  "
                      f"+{r['downloaded']:4d}  "
                      f"(valid={r['valid']}, corrupted={r['corrupted']}, "
                      f"failed={r.get('failed', 0)})")

            # Report bands with failures
            for band_name, r in band_results.items():
                if r.get('corrupted_ids'):
                    print(f"\n⚠️  {band_name}: corrupted star ids = {r['corrupted_ids']}")
                if r.get('unmatched_ids'):
                    print(f"⚠️  {band_name}: no tile coverage for ids = {r['unmatched_ids']}")

        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            traceback.print_exc()

    def _extract_psf(self):
        """Extract PSF from cutouts for a chosen band (VIS / Y_E / J_E / H_E)."""
        # Choose band first — drives default cutout dir + output filename.
        band_choices = [
            {"name": f"VIS  (0.10\"/pix, FWHM≈{Config.BAND_VIS.psf_fwhm_arcsec}\")",  "value": "VIS"},
            {"name": f"Y_E  (0.30\"/pix, FWHM≈{Config.BAND_Y_E.psf_fwhm_arcsec}\")",  "value": "Y_E"},
            {"name": f"J_E  (0.30\"/pix, FWHM≈{Config.BAND_J_E.psf_fwhm_arcsec}\")",  "value": "J_E"},
            {"name": f"H_E  (0.30\"/pix, FWHM≈{Config.BAND_H_E.psf_fwhm_arcsec}\")",  "value": "H_E"},
        ]
        band_name = select("Select band:", choices=band_choices).ask()
        if band_name is None:
            return
        band = Config.get_band(band_name)

        # New layout: data/euclid_stars/cutouts/<band>/. Falls back to the
        # legacy flat VIS layout if no per-band subdir exists yet.
        default_cutout_dir = Config.cutout_dir_for_band(band_name)
        if not os.path.isdir(default_cutout_dir) and band_name == "VIS":
            legacy = os.path.join(Config.DEFAULT_OUTPUT_DIR, "cutouts")
            if os.path.isdir(legacy):
                default_cutout_dir = legacy

        cutout_dir_choice = select(
            "Select cutout source directory:",
            choices=[
                {"name": f"{default_cutout_dir} (default for {band_name})",
                 "value": default_cutout_dir},
                {"name": "Custom path...", "value": "custom"},
            ]
        ).ask()
        cutout_dir = (input("Enter path: ").strip()
                      if cutout_dir_choice == "custom" else cutout_dir_choice)
        psf_dir = select(
            "Select PSF output directory:",
            choices=[
                {"name": f"{Config.EUCLID_PSF_DIR} (default)", "value": Config.EUCLID_PSF_DIR},
                {"name": "Custom path...", "value": "custom"},
            ]
        ).ask()

        if psf_dir == "custom":
            psf_dir = input("Enter path: ").strip()

        # Cutout size first — PSF extraction operates on a single cutout size,
        # since extract_psf_star_from_cutout does a centered crop based on the
        # input image dimensions.
        if not os.path.exists(cutout_dir):
            print(f"\n✗ Cutout directory not found: {cutout_dir}")
            return

        cutout_size_input = input(
            f"Cutout size to use (default {Config.DEFAULT_CUTOUT_SIZE}): "
        ).strip()
        try:
            cutout_size = int(cutout_size_input) if cutout_size_input else Config.DEFAULT_CUTOUT_SIZE
        except ValueError:
            print("\n✗ Invalid cutout size: must be an integer")
            return

        num_stars_input = input("Number of stars to use (default: all): ").strip()

        # PSF size — default to the largest odd value <= cutout_size - 1 so
        # the centered crop fits inside the cutout.
        default_psf_size = (cutout_size - 1) if (cutout_size % 2 == 0) else (cutout_size - 2)
        psf_size_input = input(
            f"PSF size in pixels (must be odd, default {default_psf_size}): "
        ).strip()
        if psf_size_input:
            try:
                psf_size = int(psf_size_input)
                if psf_size <= 0 or psf_size % 2 == 0:
                    print(f"\n✗ PSF size must be a positive odd integer (got {psf_size})")
                    return
                if psf_size > cutout_size:
                    print(f"\n✗ PSF size ({psf_size}) cannot exceed cutout size ({cutout_size})")
                    return
            except ValueError:
                print("\n✗ Invalid PSF size: must be an integer")
                return
        else:
            psf_size = default_psf_size

        # Optional explicit output size (in oversampled pixels). Even values
        # are bumped to odd — e.g. user types 1024 → output PSF is 1023×1023.
        output_size_input = input(
            "Output PSF size in oversampled pixels "
            "(blank = photutils default, e.g. 1024 → 1023 odd): "
        ).strip()
        output_size: int | None
        if output_size_input:
            try:
                output_size = int(output_size_input)
                if output_size <= 0:
                    print(f"\n✗ Output size must be positive, got {output_size}")
                    return
            except ValueError:
                print("\n✗ Invalid output size: must be an integer")
                return
        else:
            output_size = None

        config = PSFExtractionConfig(
            psf_size=psf_size,
            output_size=output_size,
            oversampling=band.epsf_oversampling,
            progress_bar=True,
        )
        extractor = PSFExtractor(config)
        print(f"  oversampling = {config.oversampling}  →  ePSF pixel "
              f"scale = {band.epsf_pixel_scale_arcsec:.4f}\"/pix")

        # Get cutout files filtered to the requested size
        all_files = extractor.get_cutout_files(cutout_dir, cutout_size=cutout_size)

        if len(all_files) == 0:
            print(f"\n✗ No FITS files of size {cutout_size} found in {cutout_dir}")
            return

        print(f"\nFound {len(all_files)} cutout files at size {cutout_size}")

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

        # Extract PSF — pixel scale and output filename are band-specific.
        try:
            epsf, fitted_stars = extractor.build_epsf(selected_files)
            # Saved ePSF lives at 0.05"/pix for every band via the
            # band-specific oversampling factor set above.
            epsf_pixel_scale = band.epsf_pixel_scale_arcsec
            psf = extractor.to_psf(epsf_pixel_scale)
            fits_path = psf.save(psf_dir, filename=band.psf_fits_filename)

            print(f"\n✓ PSF extraction completed for band {band_name}!")
            print(f"  FITS file: {fits_path}")

            summary = extractor.get_summary()
            print("\nPSF Summary:")
            print(f"  Shape: {summary['shape']}")
            print(f"  Pixel scale: {epsf_pixel_scale:.4f} arcsec/pix")
            print(f"  Oversampling: {summary['oversampling']}")
            print(f"  Data type: {summary['data_type']}")

        except Exception as e:
            print(f"\n✗ PSF extraction failed: {e}")
            traceback.print_exc()

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
        print_header("Integrity Check Results")
        print(f"Total files:      {len(fits_files)}")
        print(f"Valid:            {len(results['valid'])} ✓")
        print(f"Corrupted:        {len(results['corrupted'])} 🔴")

        if results['corrupted']:
            print("\nCorrupted files:")
            for filepath, error_msg in results['corrupted'][:10]:  # Show first 10
                filename = os.path.basename(filepath)
                print(f"  🔴 {filename}: {error_msg}")
            if len(results['corrupted']) > 10:
                print(f"  ... and {len(results['corrupted']) - 10} more")

        # Update catalog if stars.csv exists
        catalog_path = os.path.join(output_dir, Config.CATALOG_FILE)
        if os.path.exists(catalog_path):
            objects = CatalogObject.read(catalog_path)
            by_id = {o.id: o for o in objects}

            # Update star status based on validation — per-size, not whole-star
            for filepath, _error_msg in results['corrupted']:
                filename = os.path.basename(filepath)
                parts = filename.split('_')
                if len(parts) >= 3 and parts[0] == 'star':
                    try:
                        star_id = int(parts[1])
                        size = int(parts[2].replace('.fits', ''))
                    except ValueError:
                        continue
                    o = by_id.get(star_id)
                    if o is not None:
                        o.set_corrupted(size)

            CatalogObject.write(objects, catalog_path)
            print("\n✓ Updated catalog with validation results")

        print("\n✓ Integrity check completed!")

    def _login_euclid(self):
        """Log into the Euclid archive via env vars or an explicit username/password."""
        if self._euclid is not None:
            print_success(
                f"Already logged in as {self._euclid.user or '(unknown)'}. "
                f"Logout first to switch accounts."
            )
            return

        method = select(
            "Login method:",
            choices=[
                {"name": "Environment variables (EUCLID_USER / EUCLID_PASSWORD)", "value": "env"},
                {"name": "Enter username and password now", "value": "interactive"},
                {"name": "Cancel", "value": "cancel"},
            ]
        ).ask()

        if method in (None, "cancel"):
            print_cancelled()
            return

        try:
            if method == "env":
                self._euclid = EuclidCatalog()
            else:
                user = input("Euclid username: ").strip()
                password = getpass.getpass("Euclid password: ")
                self._euclid = EuclidCatalog(login=user, password=password)
        except EuclidAuthError as e:
            print_error(str(e))
            return

        print_success(
            f"Logged in as {self._euclid.user or '(unknown)'}. "
            f"All subsequent Euclid queries will use this session."
        )

    def _logout_euclid(self):
        """Log out from the Euclid archive."""
        if self._euclid is None:
            print_error("Not logged in.")
            return
        self._euclid = None
        with contextlib.suppress(Exception):
            Euclid.logout()
        print_success("Logged out.")

    def _query_brightest_stars(self):
        """Query the brightest N stars from the Euclid archive (async)."""
        output_dir = select(
            "Select output directory:",
            choices=[
                {"name": "./data/euclid_stars (default)", "value": "./data/euclid_stars"},
                {"name": "Custom path...", "value": "custom"},
            ]
        ).ask()
        if output_dir == "custom":
            output_dir = input("Enter path: ").strip()

        num_raw = input(f"Number of brightest stars (default {Config.DEFAULT_BRIGHTEST_N}): ").strip() \
            or str(Config.DEFAULT_BRIGHTEST_N)
        num_check = validate_positive_number(num_raw, "num_stars")
        if num_check is not True:
            print_error(num_check)
            return
        num_stars = int(float(num_raw))

        use_cone = confirm("Restrict to a sky region (cone)?", default=False).ask()
        ra_val = dec_val = radius_val = None
        if use_cone:
            ra = input("RA (degrees): ").strip()
            dec = input("Dec (degrees): ").strip()
            radius = input("Radius (degrees, default 1): ").strip() or "1"
            for _raw, check, _name in (
                (ra, validate_ra(ra), "RA"),
                (dec, validate_dec(dec), "Dec"),
                (radius, validate_positive_number(radius, "Radius"), "Radius"),
            ):
                if check is not True:
                    print_error(check)
                    return
            ra_val, dec_val, radius_val = float(ra), float(dec), float(radius)

        use_mag = confirm("Apply faint-end magnitude limit?", default=False).ask()
        mag_val = None
        if use_mag:
            mag = input(f"Magnitude limit (default {Config.DEFAULT_MAGNITUDE_LIMIT}): ").strip() \
                or str(Config.DEFAULT_MAGNITUDE_LIMIT)
            mag_check = validate_positive_number(mag, "Magnitude limit")
            if mag_check is not True:
                print_error(mag_check)
                return
            mag_val = float(mag)

        use_mag_min = confirm(
            "Apply bright-end magnitude cutoff (skip saturating stars)?",
            default=False,
        ).ask()
        mag_min_val = None
        if use_mag_min:
            mag_min = input("Min magnitude (brighter than this rejected, e.g. 15): ").strip()
            check = validate_positive_number(mag_min, "Min magnitude")
            if check is not True:
                print_error(check)
                return
            mag_min_val = float(mag_min)

        require_unmasked = confirm(
            "Require mask-free stars (det_quality_flag=0 — no saturation, "
            "blending, or bright-star masks; recommended for ePSF)?",
            default=True,
        ).ask()

        if self._euclid is None:
            print("\n⚠️  Not logged in: async job results are still fetched now, but")
            print("    the job record is garbage-collected after 72h on the server.")

        if not confirm(f"\nQuery brightest {num_stars} stars (async)?", default=True).ask():
            print_cancelled()
            return

        try:
            print("\nSubmitting async query...")
            candidates = self._euclid_client().query_bright_stars(
                num_stars,
                ra=ra_val,
                dec=dec_val,
                radius=radius_val,
                magnitude_limit=mag_val,
                magnitude_min=mag_min_val,
                require_unmasked=require_unmasked,
            )
            result = self._ingest_query(output_dir, candidates, limit=num_stars)
            print(f"\n{result['message']}")
            if result.get('skipped', 0) > 0:
                print(f"  (Skipped {result['skipped']} duplicate stars)")

            self._visualize_star_positions(output_dir=output_dir)
        except Exception as e:
            print_error(f"Query failed: {e}")
            traceback.print_exc()

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
        """Apply the multi-band forward model: HR 4-channel → LR 4-channel + HR 4-channel clean target."""
        psf_dir = input(
            f"PSF directory (default {Config.EUCLID_PSF_DIR}): "
        ).strip() or Config.EUCLID_PSF_DIR

        # Show PSF inventory so the user knows which bands use empirical vs Gaussian.
        inv = psf_inventory(psf_dir=psf_dir)
        print("\nPSF inventory:")
        for name, path in inv.items():
            status = f"empirical ({path})" if path else "Gaussian fallback"
            print(f"  {name}: {status}")

        require_emp_raw = input(
            "Require empirical PSF for every band? (y/n, default n): "
        ).strip().lower() or "n"
        require_empirical = require_emp_raw.startswith("y")

        try:
            psf_sets = load_all_band_psf_sets(
                psf_dir=psf_dir,
                require_empirical=require_empirical,
                target_pixel_scale=Config.DEFAULT_PIXEL_SCALE,
            )
        except FileNotFoundError as e:
            print(f"\n✗ {e}")
            return

        # Discover which subsets have v2 clean TFRecords.
        subsets_to_run = []
        for subset in ("train", "validate", "test"):
            clean_path = tfrecord_path(Config.RECORDS_DIR_V2, f"clean_{subset}")
            if os.path.exists(clean_path):
                n_images = sum(1 for _ in tf.data.TFRecordDataset(clean_path))
                subsets_to_run.append((subset, clean_path, n_images))
            else:
                print(f"  ⚠️  Skipping {subset}: {clean_path} not found")

        if not subsets_to_run:
            print(f"\n✗ No clean v2 TFRecords found in {Config.RECORDS_DIR_V2}")
            return

        print(f"\nSource: {Config.RECORDS_DIR_V2}")
        for subset, _, n_images in subsets_to_run:
            print(f"  clean_{subset}.tfrecord → dirty_{subset}.tfrecord ({n_images} images)")
        total = sum(n for _, _, n in subsets_to_run)
        print("  Noise: per-band Poisson + Gaussian read for VIS / Y_E / J_E / H_E")
        print(f"  NISP→VIS-LR resample: {Config.NISP_RESAMPLE_KERNEL}")
        print("  Output channels: LR=(VIS, Y_E, J_E, H_E) @ 0.10\"/pix; HR=(VIS, Y_E, J_E, H_E) @ 0.05\"/pix")

        if not confirm(f"\nRun forward model on {total} images?", default=True).ask():
            return

        forward = ObservationSimulator(
            psf_sets_by_band=psf_sets,
            config=ObservationSimulatorConfig(add_noise=True),
        )

        for subset, clean_file, n_images in subsets_to_run:
            # Entropy-seeded forward-model RNG so each run draws fresh
            # noise/CR/streak realisations. Master seed logged for replay.
            master_seed = int.from_bytes(os.urandom(8), "little")
            rng = np.random.default_rng(master_seed)
            print(f"  forward {subset}: master_seed={master_seed}")
            n_ok = n_err = 0

            # Stream LR + HR pairs directly to disk so memory scales with
            # one image (≈5 MB) instead of the whole set (~13 GB at 6400).
            with open_writer(
                    f"clean_{subset}", records_dir=Config.RECORDS_DIR_V2) as hr_w, \
                 open_writer(
                    f"dirty_{subset}", records_dir=Config.RECORDS_DIR_V2) as lr_w:
                for raw in tqdm(tf.data.TFRecordDataset(clean_file),
                                total=n_images, desc=f"Forward {subset}",
                                unit="img"):
                    try:
                        hr_4ch = Image.from_tfrecord(raw)
                        lr, hr = forward.process(hr_4ch, rng=rng)
                        hr_w.write(hr, index=n_ok)
                        lr_w.write(lr, index=n_ok)
                        n_ok += 1
                    except Exception as e:
                        n_err += 1
                        tqdm.write(f"  ✗ Skipping record (error: {e})")
            print(f"  ✓ {subset}: {n_ok} ok, {n_err} skipped → "
                  f"clean_{subset}.tfrecord (HR 4-ch) + dirty_{subset}.tfrecord (LR 4-ch)")

    def _generate_clean_data(self):
        """Generate 4-band clean HR sky data using the COSMOS2025 catalog."""
        catalog_path = input(
            f"COSMOS2025 catalog FITS (default {Config.COSMOS2025_CATALOG_PATH}): "
        ).strip() or Config.COSMOS2025_CATALOG_PATH
        if not os.path.isfile(catalog_path):
            print(f"\n✗ COSMOS2025 catalog not found at {catalog_path}. "
                  f"The pipeline requires the real Shuntov+ 2025 master catalog.")
            return

        ntrain     = (input(f"Number of training images (default {Config.DEFAULT_NIMAGES}): ").strip()
                      or str(Config.DEFAULT_NIMAGES))
        nvalid     = (
            input(f"Number of validation images (default {Config.DEFAULT_NIMAGES // 5}): ").strip()
            or str(Config.DEFAULT_NIMAGES // 5))
        ntest      = (
            input(f"Number of test images — held-out, for evals "
                  f"(default {Config.DEFAULT_NIMAGES // 5}): ").strip()
            or str(Config.DEFAULT_NIMAGES // 5))
        pixel_scale = (input(f"HR pixel scale in arcsec (default {Config.DEFAULT_PIXEL_SCALE}): ").strip()
                       or str(Config.DEFAULT_PIXEL_SCALE))
        image_size  = input(
            "HR image side in pixels (prefer a multiple of 6 to avoid edge trim; "
            "default 252): "
        ).strip() or "252"

        try:
            ntrain_val      = int(ntrain)
            nvalid_val      = int(nvalid)
            ntest_val       = int(ntest)
            pixel_scale_val = float(pixel_scale)
            image_size_val  = int(image_size)
        except ValueError:
            print("\n✗ Invalid input: values must be numbers")
            return

        if image_size_val % 6 != 0:
            trimmed = (image_size_val // 6) * 6
            lost = image_size_val - trimmed
            print(f"\nNote: image-size {image_size_val} is not divisible by 6. "
                  f"The forward model will use HR={trimmed}² (drops {lost} "
                  f"pixel{'s' if lost != 1 else ''} on each axis at the trailing "
                  f"edge); nearest no-trim sizes are {trimmed} or {trimmed + 6}.")

        print("\nConfiguration (multi-band, v2 schema):")
        print(f"  Catalog:           {catalog_path}")
        print(f"  Training images:   {ntrain_val}")
        print(f"  Validation images: {nvalid_val}")
        print(f"  Test images:       {ntest_val}  (held-out, for evals)")
        print(f"  Pixel scale:       {pixel_scale_val} arcsec/pix (HR)")
        print(f"  Image size:        {image_size_val} x {image_size_val} (4 channels)")
        print(f"  Output (TFRecord): {Config.RECORDS_DIR_V2}/")
        print("    clean_train.tfrecord")
        print("    clean_validate.tfrecord")
        print("    clean_test.tfrecord")

        if not confirm("\nGenerate clean multi-band sky data?", default=True).ask():
            return

        try:
            catalog = CosmosTngPrior(Config.COSMOS_TNG_PRIOR_PATH)
            print(
                f"\nCOSMOS joint prior: {len(catalog)} latent galaxies"
            )

            cfg = SkySimulatorConfig(
                image_size=image_size_val,
                pixel_scale=pixel_scale_val,
            )
            sim = SkySimulator(catalog, cfg)
            os.makedirs(Config.RECORDS_DIR_V2, exist_ok=True)

            for subset, n in (("train", ntrain_val), ("validate", nvalid_val),
                              ("test", ntest_val)):
                if n <= 0:
                    continue
                # Entropy-seeded master RNG: fresh fields every invocation.
                # Master seed is logged for replay.
                master_seed = int.from_bytes(os.urandom(8), "little")
                rng = np.random.default_rng(master_seed)
                print(f"\nGenerating {subset} set ({n} images)  "
                      f"[master_seed={master_seed}]...")
                # Stream so memory bounded to ~one image (otherwise 6400
                # 510² × 4-channel fields cost ~26 GB).
                with open_writer(
                        f"clean_{subset}", records_dir=Config.RECORDS_DIR_V2) as w:
                    for i in tqdm(range(n), desc=subset):
                        sky, _ = sim.simulate_field(rng)
                        sky.index = i
                        sky.subset = subset
                        w.write(sky, index=i)
                    path = w.path
                print(f"  ✓ {path}")

            print("\n✓ Multi-band clean data generation completed!")

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
                    {"name": "🌐 Fetch a sky position & super-resolve", "value": "fetch_sr"},
                    {"name": "🔄 Inspect checkpoints", "value": "inspect"},
                    {"name": "📉 Plot training log", "value": "plot_log"},
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
            elif choice == "fetch_sr":
                self._fetch_and_superresolve()
            elif choice == "inspect":
                self._inspect_checkpoints()
            elif choice == "plot_log":
                self._plot_training_log()

    def _train_model(self):
        """Train WDSR model on multi-band (4-channel LR → 4-channel VIS+NISP HR) data."""
        scale = (input(f"Scale factor (default {Config.DEFAULT_REBIN_FACTOR}): ").strip()
                 or str(Config.DEFAULT_REBIN_FACTOR))
        num_res_blocks = (
            input(f"Number of residual blocks (default {Config.DEFAULT_NUM_RES_BLOCKS}): ").strip()
            or str(Config.DEFAULT_NUM_RES_BLOCKS))
        checkpoint_dir = (input(f"Checkpoint directory (default {Config.DEFAULT_CHECKPOINT_DIR}): ").strip()
                          or Config.DEFAULT_CHECKPOINT_DIR)
        steps = (input(f"Training steps (default {Config.DEFAULT_TRAIN_STEPS}): ").strip()
                 or str(Config.DEFAULT_TRAIN_STEPS))
        batch_size = (input(f"Batch size (default {Config.DEFAULT_BATCH_SIZE}): ").strip()
                      or str(Config.DEFAULT_BATCH_SIZE))
        evaluate_every = (
            input(f"Evaluate every N steps (default {Config.DEFAULT_EVALUATE_EVERY}): ").strip()
            or str(Config.DEFAULT_EVALUATE_EVERY))

        try:
            scale_val = int(scale)
            num_res_blocks_val = int(num_res_blocks)
            steps_val = int(steps)
            batch_size_val = int(batch_size)
            evaluate_every_val = int(evaluate_every)
        except ValueError:
            print("\n✗ Invalid input: all values must be integers")
            return

        records_dir = Config.RECORDS_DIR_V2
        clean_train = tfrecord_path(records_dir, "clean_train")
        dirty_train = tfrecord_path(records_dir, "dirty_train")

        if not os.path.exists(clean_train) or not os.path.exists(dirty_train):
            print(f"\n✗ Training data not found in {records_dir}")
            print("  Run multi-band clean sky generation and HR→LR forward first.")
            return

        dirty_valid = tfrecord_path(records_dir, "dirty_validate")
        if not os.path.exists(dirty_valid):
            print(f"\n⚠️  No validation data in {records_dir} — will train without validation")

        print("\nConfiguration:")
        print(f"  Scale: {scale_val}x")
        print(f"  Residual blocks: {num_res_blocks_val}")
        print(f"  Training steps: {steps_val}")
        print(f"  Batch size: {batch_size_val}")
        print(f"  Evaluate every: {evaluate_every_val} steps")
        print(f"  Records: {records_dir}")
        print(f"  Checkpoint directory: {checkpoint_dir}")
        print(f"  Input channels: {Config.NUM_LR_CHANNELS} "
              f"({', '.join(Config.LR_INPUT_BAND_NAMES)})")
        print(f"  Output channels: {Config.NUM_HR_CHANNELS} ({Config.HR_TARGET_BAND_NAME})")

        if confirm("\nStart training?", default=True).ask():
            print("\n⚠️  Training will run until interrupted (Ctrl+C) or completion")

            try:
                m = Model(checkpoint_dir, scale=scale_val,
                          num_res_blocks=num_res_blocks_val)

                # Train
                print("\nStarting training...")
                m.train(
                    tfrecord_path(records_dir, "dirty_train"),
                    tfrecord_path(records_dir, "clean_train"),
                    steps=steps_val,
                    evaluate_every=evaluate_every_val,
                )

                # Post-training evaluation on the held-out test set (else
                # validate, for datasets predating the test split).
                eval_sub = eval_subset(records_dir)
                lr_val = ImageSet.read(tfrecord_path(records_dir, f"dirty_{eval_sub}"))
                hr_val = ImageSet.read(tfrecord_path(records_dir, f"clean_{eval_sub}"))
                valid_ds = m._build_training_pipeline(
                    lr_val.source_path, hr_val.source_path, 1, augment=False
                )
                metrics = Trainer(model=m._tf_model,
                                  checkpoint_dir=checkpoint_dir).evaluate(valid_ds)
                print(
                    f"\nFinal metrics ({eval_sub} set):\n"
                    f"  PSNR (stretched, loss-aligned): {float(metrics['psnr_stretched']):.3f} dB\n"
                    f"  PSNR (raw e⁻):                 {float(metrics['psnr_raw']):.3f} dB"
                )

                # Offer to export weights
                if confirm("\nExport trained weights to .h5 file?", default=True).ask():
                    weights_dir = os.path.dirname(checkpoint_dir) or "."
                    weights_path = os.path.join(weights_dir, f"wdsr_x{scale_val}.h5")
                    m._tf_model.save_weights(weights_path)
                    print(f"  ✓ Weights saved to: {weights_path}")

                print("\n✓ Training completed!")

            except KeyboardInterrupt:
                print("\n\n⚠️  Training interrupted by user")
            except Exception as e:
                print(f"\n✗ Training failed: {e}")
                traceback.print_exc()

    def _evaluate_model(self):
        """Evaluate a trained model on the held-out test set (else validate)."""
        checkpoint_dir = (input(f"Checkpoint directory (default {Config.DEFAULT_CHECKPOINT_DIR}): ").strip()
                          or Config.DEFAULT_CHECKPOINT_DIR)
        scale = (input(f"Scale factor (default {Config.DEFAULT_REBIN_FACTOR}): ").strip()
                 or str(Config.DEFAULT_REBIN_FACTOR))
        num_res_blocks = (
            input(f"Number of residual blocks (default {Config.DEFAULT_NUM_RES_BLOCKS}): ").strip()
            or str(Config.DEFAULT_NUM_RES_BLOCKS))

        try:
            scale_val = int(scale)
            num_res_blocks_val = int(num_res_blocks)
        except ValueError:
            print("\n✗ Invalid input: scale and num_res_blocks must be integers")
            return

        if not tf.train.latest_checkpoint(checkpoint_dir):
            print(f"\n✗ No checkpoints found in {checkpoint_dir}")
            return

        # Evaluate on the held-out test set (else validate) — v2 multi-band.
        records_dir = Config.RECORDS_DIR_V2
        eval_sub = eval_subset(records_dir)
        dirty_eval = tfrecord_path(records_dir, f"dirty_{eval_sub}")
        if not os.path.exists(dirty_eval):
            print(f"\n✗ No {eval_sub} data found in {records_dir}")
            return

        try:

            print(f"\nLoading model from {checkpoint_dir}...")
            m = Model(checkpoint_dir, scale=scale_val,
                      num_res_blocks=num_res_blocks_val)
            lr_val = ImageSet.read(tfrecord_path(records_dir, f"dirty_{eval_sub}"))
            hr_val = ImageSet.read(tfrecord_path(records_dir, f"clean_{eval_sub}"))
            valid_ds = m._build_training_pipeline(
                lr_val.source_path, hr_val.source_path, 1, augment=False
            )

            print(f"Evaluating on {eval_sub} set...")
            metrics = Trainer(model=m._tf_model, checkpoint_dir=checkpoint_dir).evaluate(valid_ds)
            print(
                f"\n✓ Validation metrics:\n"
                f"  PSNR (stretched, loss-aligned): {float(metrics['psnr_stretched']):.3f} dB\n"
                f"  PSNR (raw e⁻):                 {float(metrics['psnr_raw']):.3f} dB"
            )

        except Exception as e:
            print(f"\n✗ Evaluation failed: {e}")
            traceback.print_exc()

    def _inspect_checkpoints(self):
        """List available checkpoints in a directory."""
        checkpoint_dir = (input(f"Checkpoint directory (default {Config.DEFAULT_CHECKPOINT_DIR}): ").strip()
                          or Config.DEFAULT_CHECKPOINT_DIR)

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

    def _plot_training_log(self):
        """Plot the per-evaluation loss / PSNR curves from a training log."""

        checkpoint_dir = input(
            f"Checkpoint directory (default {Config.DEFAULT_CHECKPOINT_DIR}): "
        ).strip() or Config.DEFAULT_CHECKPOINT_DIR

        log_path = default_log_path(checkpoint_dir)
        if not os.path.exists(log_path):
            print(f"\n✗ No training log at {log_path}")
            print("  (Logs are written automatically by the trainer at each evaluation.)")
            return

        smooth_str = input("Smoothing window in evaluations (default 0 = none): ").strip() or "0"
        try:
            smooth_window = int(smooth_str)
        except ValueError:
            print("  ⚠️  Invalid window, using 0")
            smooth_window = 0

        os.makedirs(Config.VIS_DIR, exist_ok=True)
        output_path = os.path.join(Config.VIS_DIR, "training_log.png")
        try:
            n, last_step = plot_training_log(log_path, output_path, smooth_window=smooth_window)
            print(f"\n✓ Plotted {n} evaluations (last step: {last_step})")
            print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"\n✗ Plot failed: {e}")
            traceback.print_exc()

    def _reconstruct_image(self):
        """Apply super-resolution to a single LR image."""
        # Branch-local inputs, pre-bound so the later `input_source`-guarded
        # reads are always defined (one of the two branches below fills them).
        chosen_lr = chosen_hr = None
        num_reconstruct = 0
        lr_data_input = lr_path = None
        input_source = select(
            "Load LR image from:",
            choices=[
                {"name": "TFRecords (dirty images from training data)", "value": "tfrecord"},
                {"name": "File path (.npy or .png)", "value": "file"},
            ]
        ).ask()

        if input_source == "tfrecord":
            dirty_train = tfrecord_path(Config.RECORDS_DIR, "dirty_train")
            dirty_valid = tfrecord_path(Config.RECORDS_DIR, "dirty_validate")
            available = []
            if os.path.exists(dirty_train):
                available.append({"name": "dirty_train.tfrecord", "value": dirty_train})
            if os.path.exists(dirty_valid):
                available.append({"name": "dirty_validate.tfrecord", "value": dirty_valid})
            if not available:
                print(f"\n✗ No dirty TFRecords found in {Config.RECORDS_DIR}")
                return
            tfr_path = select("Which dataset:", choices=available).ask()
            num_str = input("Number of random images to reconstruct (default 5): ").strip() or "5"
            try:
                num_reconstruct = int(num_str)
            except ValueError:
                print("\n✗ Invalid number")
                return
            images = read_images(tfr_path, num_images=9999)
            if not images:
                print(f"\n✗ No images found in {tfr_path}")
                return
            num_reconstruct = min(num_reconstruct, len(images))
            rng = np.random.default_rng()
            chosen_indices = rng.choice(len(images), size=num_reconstruct, replace=False)
            chosen_lr = [images[i] for i in chosen_indices]

            # Also try to load matching HR for comparison
            clean_file = tfr_path.replace("dirty_", "clean_")
            chosen_hr = [None] * num_reconstruct
            if os.path.exists(clean_file):
                clean_images = read_images(clean_file, num_images=9999)
                clean_by_idx = {img.index: img for img in clean_images}
                for i, lr_img in enumerate(chosen_lr):
                    hr_match = clean_by_idx.get(lr_img.index)
                    if hr_match is not None:
                        chosen_hr[i] = hr_match
        else:
            lr_file = input("Path to LR image (.npy or .png): ").strip()
            if not lr_file or not os.path.exists(lr_file):
                print(f"\n✗ File not found: {lr_file}")
                return
            lr_data_input = lr_file
            lr_path = lr_file

        source = select(
            "Load model from:",
            choices=[
                {"name": "Checkpoint directory", "value": "checkpoint"},
                {"name": ".h5 weights file", "value": "weights"},
            ]
        ).ask()

        scale = (input(f"Scale factor (default {Config.DEFAULT_REBIN_FACTOR}): ").strip()
                 or str(Config.DEFAULT_REBIN_FACTOR))
        num_res_blocks = (
            input(f"Number of residual blocks (default {Config.DEFAULT_NUM_RES_BLOCKS}): ").strip()
            or str(Config.DEFAULT_NUM_RES_BLOCKS))

        try:
            scale_val = int(scale)
            num_res_blocks_val = int(num_res_blocks)
        except ValueError:
            print("\n✗ Invalid input: scale and num_res_blocks must be integers")
            return

        try:

            if source == "checkpoint":
                ckpt_dir = (input(f"Checkpoint directory (default {Config.DEFAULT_CHECKPOINT_DIR}): ").strip()
                            or Config.DEFAULT_CHECKPOINT_DIR)
                if not tf.train.latest_checkpoint(ckpt_dir):
                    print(f"\n✗ No checkpoints found in {ckpt_dir}")
                    return
                print(f"\nLoading model from checkpoint {ckpt_dir}...")
                if input_source == "tfrecord":
                    model = Model(ckpt_dir, scale=scale_val, num_res_blocks=num_res_blocks_val)
                    hr_for_render = [h for h in chosen_hr if h is not None]
                    saved = reconstruct_and_render(
                        chosen_lr, model, Config.VIS_RECONSTRUCTION_DIR,
                        hr_images=hr_for_render if len(hr_for_render) == len(chosen_lr) else None,
                    )
                    for p in saved:
                        print(f"  saved {p}")
                    print(f"\n✓ {len(saved)} reconstructions saved to {Config.VIS_RECONSTRUCTION_DIR}")
                    return
                else:
                    model = load_model_from_checkpoint(
                        ckpt_dir, scale_val, num_res_blocks_val,
                        nchan_out=Config.NUM_HR_CHANNELS,   # nchan_in inferred from ckpt
                    )
            else:
                weights_path = input("Path to .h5 weights file: ").strip()
                if not weights_path or not os.path.exists(weights_path):
                    print(f"\n✗ Weights file not found: {weights_path}")
                    return
                print(f"\nLoading model from {weights_path}...")
                model = load_model_from_weights(
                    weights_path, scale_val, num_res_blocks_val,
                    nchan=Config.NUM_LR_CHANNELS,
                )

            os.makedirs(Config.VIS_RECONSTRUCTION_DIR, exist_ok=True)
            vmax = self._ask_vmax()

            if input_source == "tfrecord":
                print(f"Running super-resolution on {num_reconstruct} images...")
                for i, lr_img in enumerate(chosen_lr):
                    lr_data, sr_data = reconstruct(model, lr_img.data)
                    hr_data = chosen_hr[i].data if chosen_hr[i] is not None else None
                    output_path = os.path.join(
                        Config.VIS_RECONSTRUCTION_DIR,
                        f"reconstruct_idx{lr_img.index}.png",
                    )
                    plot_reconstruction(lr_data, sr_data, hr_data=hr_data, output_path=output_path, vmax=vmax)
                    print(f"  ✓ [{i+1}/{num_reconstruct}] Index {lr_img.index} → {output_path}")
                print(f"\n✓ {num_reconstruct} reconstructions saved to {Config.VIS_RECONSTRUCTION_DIR}")
            else:
                hr_data = None
                hr_path = input("Path to HR ground truth (optional, press Enter to skip): ").strip() or None
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
                lr_data, sr_data = reconstruct(model, lr_data_input)
                basename = os.path.basename(lr_path).replace(".", "_")
                output_path = os.path.join(Config.VIS_RECONSTRUCTION_DIR, f"reconstruct_{basename}.png")
                plot_reconstruction(lr_data, sr_data, hr_data=hr_data, output_path=output_path)
                print(f"\n✓ Reconstruction saved to: {output_path}")

        except Exception as e:
            print(f"\n✗ Reconstruction failed: {e}")
            traceback.print_exc()


    def _fetch_and_superresolve(self):
        """Fetch a real Euclid sky position from the archive and super-resolve it."""
        ra_str = input("RA in degrees (ICRS): ").strip()
        dec_str = input("Dec in degrees (ICRS): ").strip()
        size_str = input("Cutout side in VIS pixels (default 256): ").strip() or "256"
        ckpt_dir = (input(f"Checkpoint directory (default {Config.DEFAULT_CHECKPOINT_DIR}): ").strip()
                    or Config.DEFAULT_CHECKPOINT_DIR)

        try:
            ra = float(ra_str)
            dec = float(dec_str)
            size = int(size_str)
        except ValueError:
            print("\n✗ Invalid input: RA/Dec must be floats, size must be an integer")
            return

        out_dir = os.path.join(Config.EUCLID_INFERENCE_DIR, "adhoc")
        try:
            model = Model(ckpt_dir, scale=Config.DEFAULT_REBIN_FACTOR,
                          num_res_blocks=Config.DEFAULT_NUM_RES_BLOCKS)
            fits_path, png_path = fetch_and_superresolve(
                ra=ra, dec=dec, size=size, model=model, out_dir=out_dir,
                catalog=self._euclid_client(),
            )
            print(f"\n✓ FITS: {fits_path}")
            print(f"  PNG:  {png_path}")
        except Exception as e:
            print(f"\n✗ Fetch/super-resolve failed: {e}")
            traceback.print_exc()

    @staticmethod
    def _ask_vmax(default: float | None = None) -> float | None:
        """Prompt for the upper colour-scale limit (vmax) for linear plots.

        Default ``None`` lets matplotlib auto-scale per image. Pixel values are
        raw electrons, so a fixed scale rarely fits both faint and bright fields.
        """
        prompt_default = "auto" if default is None else f"{default:.0f}"
        raw = input(f"Upper colour-scale limit for linear plots (default {prompt_default}): ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print(f"  ⚠️  Invalid value, using default ({prompt_default})")
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
                    {"name": "⭐ Visualize star positions", "value": "viz_star_positions"},
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
            elif choice == "viz_star_positions":
                self._visualize_star_positions()

    def _visualize_star_positions(self, output_dir: str = None):
        """Visualize star positions from the catalog."""
        if output_dir is None:
            output_dir = select(
                "Select catalog directory:",
                choices=[
                    {"name": "./data/euclid_stars (default)", "value": "./data/euclid_stars"},
                    {"name": "Custom path...", "value": "custom"},
                ]
            ).ask()
            if output_dir == "custom":
                output_dir = input("Enter path: ").strip()

        catalog_path = os.path.join(output_dir, Config.CATALOG_FILE)
        if not os.path.exists(catalog_path):
            print(f"\n✗ Catalog not found at {catalog_path}")
            return

        objects = CatalogObject.read(catalog_path)
        if not objects:
            print("\n✗ Catalog is empty — nothing to plot")
            return

        stars = [{"ra": o.ra, "dec": o.dec, "magnitude": o.magnitude,
                  "corrupted": o.has_any("corrupted")} for o in objects]
        output_path = Config.VIS_STAR_POSITIONS
        draw_star_positions(stars, output_path)
        print(f"\n✓ Star positions plot saved to {output_path} ({len(stars)} stars)")

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
        catalog_path = os.path.join(output_dir, Config.CATALOG_FILE)
        if not os.path.exists(catalog_path):
            print(f"\n✗ Catalog not found at {catalog_path}")
            return

        stars = CatalogObject.read(catalog_path)

        if len(stars) == 0:
            print("\n✗ No stars found in catalog")
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
            fits_files = glob.glob(os.path.join(cutout_dir, f"star_{star.id:04d}_*.fits"))
            if not fits_files:
                print(f"  Warning: No cutout for star {star.id}")
                continue

            try:
                with fits.open(fits_files[0]) as hdul:
                    data = hdul[0].data

                visualizer = BaseVisualizer(rows=1, cols=3, figsize=(18, 6),
                                            vmin=float(np.min(data)), vmax=float(np.max(data)))
                visualizer.add_scale_panel(data)
                visualizer.add_scale_panel(data, log_scale=True)

                mag_str = f"{star.magnitude:.2f}" if star.magnitude is not None else "N/A"
                visualizer.add_statistics_panel(data, {
                    'title': 'Star Information:',
                    'stats': {
                        'ID': f"{star.id:04d}",
                        'RA': f"{star.ra:.6f}°",
                        'Dec': f"{star.dec:.6f}°",
                        'Magnitude': mag_str,
                    },
                    'include_data_stats': True,
                })

                # Save figure
                output_path = os.path.join(vis_dir, f'star_{star.id:04d}.png')
                visualizer.save_figure(output_path)

            except Exception as e:
                print(f"  Warning: Failed to visualize star {star.id}: {e}")

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



        # Load PSF
        print(f"\nLoading PSF from {psf_file}...")
        psf = PSF.from_fits(psf_file)
        psf_data = psf.data

        # Create visualization (linear, log, stats) — PSF has its own flux range
        visualizer = BaseVisualizer(rows=1, cols=3, figsize=(18, 6),
                                    vmin=float(np.min(psf_data)), vmax=float(np.max(psf_data)))

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

        vmax = self._ask_vmax()

        need_clean = mode in ("clean", "pair")
        need_dirty = mode in ("dirty", "pair")

        def _find_subsets(prefix):
            found = []
            for subset in ("train", "validate"):
                path = tfrecord_path(Config.RECORDS_DIR, f"{prefix}_{subset}")
                if os.path.exists(path):
                    found.append((subset, path))
            return found

        clean_files = _find_subsets("clean") if need_clean else []
        dirty_files = _find_subsets("dirty") if need_dirty else []

        if need_clean and not clean_files:
            print(f"\n✗ No clean TFRecords found in {Config.RECORDS_DIR}")
            return
        if need_dirty and not dirty_files:
            print(f"\n✗ No dirty TFRecords found in {Config.RECORDS_DIR}")
            return

        try:
            if mode == "clean":
                vis_dir = Config.VIS_CLEAN_DIR
                os.makedirs(vis_dir, exist_ok=True)
                all_images = []
                for _, path in clean_files:
                    all_images.extend(read_images(path, num_images=9999))
                rng = np.random.default_rng(42)
                chosen = rng.choice(len(all_images), min(num_images, len(all_images)), replace=False)
                for i in chosen:
                    img = all_images[i]
                    out = os.path.join(vis_dir, f'clean_{img.index:04d}.png')
                    draw_clean_image(img.data, out, index=img.index, vmax=vmax)
                print(f"\n✓ {len(chosen)} clean images → {vis_dir}")

            elif mode == "dirty":
                vis_dir = Config.VIS_DIRTY_DIR
                os.makedirs(vis_dir, exist_ok=True)
                all_images = []
                for _, path in dirty_files:
                    all_images.extend(read_images(path, num_images=9999))
                rng = np.random.default_rng(42)
                chosen = rng.choice(len(all_images), min(num_images, len(all_images)), replace=False)
                for i in chosen:
                    img = all_images[i]
                    out = os.path.join(vis_dir, f'dirty_{img.index:04d}.png')
                    draw_dirty_image(img.data, out, index=img.index, vmax=vmax)
                print(f"\n✓ {len(chosen)} dirty images → {vis_dir}")

            elif mode == "pair":
                vis_dir = Config.VIS_DIRTY_DIR
                os.makedirs(vis_dir, exist_ok=True)
                n_drawn = 0

                # Collect matched pairs per subset to avoid index-space collisions.
                # Multi-band v2 records: HR is 1-channel (VIS), LR is 4-channel.
                all_pairs: list[tuple[Image, Image]] = []
                for subset in ("train", "validate"):
                    clean_sub = tfrecord_path(Config.RECORDS_DIR_V2, f"clean_{subset}")
                    dirty_sub = tfrecord_path(Config.RECORDS_DIR_V2, f"dirty_{subset}")
                    if not os.path.exists(clean_sub) or not os.path.exists(dirty_sub):
                        continue
                    dirty_by_index = {
                        img.index: img
                        for img in read_images(dirty_sub, num_images=9999)
                    }
                    for hr in read_images(clean_sub, num_images=9999):
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
                    # Multi-band: visualise VIS channel of both for a fair side-by-side
                    # comparison; the renderer expects 2-D arrays.
                    hr2d = hr.data[..., 0]
                    lr2d = lr.data[..., 0]
                    draw_clean_dirty_pair(hr2d, lr2d, path, index=hr.index, vmax=vmax)
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
