#!/usr/bin/env python
"""Extract empirical ePSFs for every Euclid band from local star cutouts.

For each of ``Config.BANDS`` this script:

  1. Looks for cutouts in the band-specific directory:
       VIS  → ``Config.DEFAULT_OUTPUT_DIR/cutouts``
       NISP → ``Config.NISP_DEFAULT_OUTPUT_DIR_BY_BAND[band.name]/cutouts``
  2. If cutouts are present, picks up to ``--num-stars`` of them at the
     ``--cutout-size`` size and runs :class:`PSFExtractor`.
  3. Saves the result to ``data/euclid_psf/<band.psf_fits_filename>``
     (e.g. ``euclid_psf_VIS.fits``).
  4. If no cutouts are present for a band, prints a clear note and
     continues — the loader will fall back to a Gaussian PSF for that band.

Usage:
    python scripts/extract_all_band_psfs.py
    python scripts/extract_all_band_psfs.py --num-stars 100 --cutout-size 256
    python scripts/extract_all_band_psfs.py --bands VIS,Y_E
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import BandConfig, Config
from euclid_polish.euclid.psf_extractor import (
    PSFExtractionConfig, PSFExtractor,
)


def _cutout_dir_for_band(band: BandConfig) -> str:
    """Per-band cutout directory.

    Resolution order:
      1. New layout: ``data/euclid_stars/cutouts/<band_name>/``.
      2. Legacy flat VIS layout: ``data/euclid_stars/cutouts/`` (for
         existing checkouts where the migration script has not yet run).
    """
    new_path = Config.cutout_dir_for_band(
        band.name,
        root=os.path.join(Config.DEFAULT_OUTPUT_DIR, "cutouts"),
    )
    if os.path.isdir(new_path):
        return new_path
    if band.name == "VIS":
        legacy = os.path.join(Config.DEFAULT_OUTPUT_DIR, "cutouts")
        if os.path.isdir(legacy):
            return legacy
    return new_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-stars", type=int, default=200,
                    help="Stars to use per band (default 200)")
    ap.add_argument("--cutout-size", type=int, default=None,
                    help="Cutout side in native pixels (must match filename "
                         "suffix). When set, the same value is used for every "
                         "band — fine for VIS-only runs, but the NISP cutouts "
                         "downloaded via cutout_size_vis_pixels=N have a "
                         "smaller native size, so prefer --vis-pixels.")
    ap.add_argument("--vis-pixels", type=int, default=None,
                    help="Pick a shared angular field via this VIS-pixel "
                         "count (0.10\"/pix); each band's native cutout "
                         "size is derived. Mutually exclusive with "
                         "--cutout-size.")
    ap.add_argument("--output-size", type=int, default=None,
                    help="Desired final PSF side in oversampled pixels. "
                         "Even values are bumped down to odd "
                         "(e.g. 1024 → 1023). None → photutils' default "
                         "(cutout_size × oversampling + 1).")
    ap.add_argument("--psf-dir", default=Config.EUCLID_PSF_DIR,
                    help="Output directory for the band-keyed PSF FITS files")
    ap.add_argument("--bands", default=",".join(b.name for b in Config.BANDS),
                    help="Comma-separated list of bands to process")
    return ap.parse_args()


def extract_band(band: BandConfig, args: argparse.Namespace) -> bool:
    cutout_dir = _cutout_dir_for_band(band)
    out_path   = os.path.join(args.psf_dir, band.psf_fits_filename)

    # Pick the native cutout size for this band: either user-supplied
    # ``--cutout-size`` (same for every band), or derived from the shared
    # angular field via ``--vis-pixels``.
    if args.vis_pixels is not None:
        arcsec = args.vis_pixels * Config.BAND_VIS.pixel_scale_lr_arcsec
        cutout_size = band.cutout_size_for_arcsec(arcsec)
    elif args.cutout_size is not None:
        cutout_size = args.cutout_size
    else:
        cutout_size = Config.DEFAULT_CUTOUT_SIZE

    header = f"=== {band.name} ==="
    print(header)
    print(f"  cutouts:     {cutout_dir}")
    print(f"  cutout-size: {cutout_size} px (native)")
    print(f"  output:      {out_path}")

    if not os.path.isdir(cutout_dir):
        print(f"  ⚠️  cutout directory not found — skipping. "
              f"Run the cutout downloader for band={band.name} first.")
        return False

    # ``psf_size`` is the *native* centred crop EPSFBuilder receives from
    # each star. For an output PSF of side ``output_size`` at oversampling
    # ``ovs``, the input crop must contain ``output_size / ovs`` native
    # pixels (the natural EPSF grid is ``psf_size × ovs + 1``). Without
    # this, EPSFBuilder fills the outer regions with zeros / noise because
    # no star contributes flux there.
    psf_size = cutout_size - 1 if cutout_size % 2 == 0 else cutout_size - 2
    cfg = PSFExtractionConfig(
        progress_bar=True,
        psf_size=psf_size,
        output_size=args.output_size,
        oversampling=band.epsf_oversampling,
    )
    print(f"  psf-size:    {psf_size} px (native centred crop from each star)")
    extractor = PSFExtractor(cfg)
    all_files = extractor.get_cutout_files(cutout_dir, cutout_size=cutout_size)
    if not all_files:
        print(f"  ⚠️  no cutouts of size {cutout_size} found — skipping.")
        return False

    selected = extractor.select_files(all_files, num_stars=args.num_stars)
    print(f"  using {len(selected)} of {len(all_files)} available stars")

    try:
        epsf, fitted = extractor.build_epsf(selected)
    except Exception as e:
        print(f"  ✗ extraction failed: {type(e).__name__}: {e}")
        return False

    # Pixel scale on the *oversampled* ePSF grid: native / oversampling.
    # By picking ``epsf_oversampling`` so this equals 0.05"/pix for every
    # band, all ePSFs land on the same HR grid the forward model uses.
    epsf_pixel_scale = band.epsf_pixel_scale_arcsec
    psf = extractor.to_psf(epsf_pixel_scale)
    os.makedirs(args.psf_dir, exist_ok=True)
    saved = psf.save(args.psf_dir, filename=band.psf_fits_filename)
    print(f"  ✓ saved {saved}")
    print(f"     shape={psf.shape}, "
          f"pixel_scale={psf.pixel_scale:.4f}\"/pix, "
          f"fwhm≈{psf.fwhm_arcsec or '?'}")
    return True


def main() -> int:
    args = parse_args()
    if args.cutout_size is not None and args.vis_pixels is not None:
        print("✗ Pass either --cutout-size or --vis-pixels, not both.")
        return 1
    requested = [name.strip() for name in args.bands.split(",") if name.strip()]
    bands = [Config.get_band(name) for name in requested]

    print(f"Extracting ePSF for bands: {[b.name for b in bands]}")
    print(f"  num-stars    = {args.num_stars}")
    if args.vis_pixels is not None:
        print(f"  vis-pixels   = {args.vis_pixels}  (per-band native size derived)")
    else:
        print(f"  cutout-size  = {args.cutout_size}")
    print(f"  output-size  = {args.output_size}")
    print(f"  psf-dir      = {args.psf_dir}\n")

    succeeded = []
    for band in bands:
        ok = extract_band(band, args)
        succeeded.append((band.name, ok))
        print()

    print("=" * 50)
    print("Summary:")
    for name, ok in succeeded:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {name}")
    n_ok = sum(1 for _, ok in succeeded if ok)
    print(f"\n{n_ok}/{len(succeeded)} bands extracted; missing bands will "
          "use Gaussian fallback in the forward model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
