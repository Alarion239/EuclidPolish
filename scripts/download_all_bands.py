#!/usr/bin/env python
"""Download cutouts for every Euclid band sharing one angular field size.

Multi-band wrapper around :class:`EuclidCutoutDownloader`. Picks one
``cutout_size_vis_pixels`` value, converts it to each band's native
pixel count via :meth:`BandConfig.cutout_size_for_arcsec`, and runs the
downloader once per band. Each band's catalog flags are tracked
independently under the shared ``stars.csv``.

Usage:
    python scripts/download_all_bands.py
    python scripts/download_all_bands.py --vis-pixels 512 --workers 8
    python scripts/download_all_bands.py --bands VIS,Y_E --vis-pixels 256
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.euclid.catalog import StarCatalog
from euclid_polish.euclid.downloader import (
    DownloadConfig, EuclidCutoutDownloader,
)
from euclid_polish.observability.reporter import Reporter


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=Config.DEFAULT_OUTPUT_DIR,
                    help="Star catalog root")
    ap.add_argument("--vis-pixels", type=int, default=Config.DEFAULT_CUTOUT_SIZE,
                    help="Cutout size in VIS pixels (= 0.10\"/pix). Each band "
                         "uses its own native pixel count covering the same "
                         "angular field.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel downloads per band")
    ap.add_argument("--bands",
                    default=",".join(b.name for b in Config.BANDS),
                    help="Comma-separated band list")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    reporter = Reporter.from_env()
    band_names = [n.strip() for n in args.bands.split(",") if n.strip()]
    arcsec = args.vis_pixels * Config.BAND_VIS.pixel_scale_lr_arcsec

    print(f"angular field   = {arcsec:.2f}\"  (= {args.vis_pixels} VIS px)")
    print(f"bands           = {band_names}")
    print(f"output dir      = {args.output_dir}")
    print(f"workers / band  = {args.workers}\n")

    cat = StarCatalog(args.output_dir)
    if not cat.exists():
        reporter.error(f"no catalog at {cat.catalog_path}")
        print(f"✗ no catalog at {cat.catalog_path}")
        return 1

    summary: dict[str, dict] = {}
    n_bands = len(band_names)
    t0 = time.perf_counter()
    for i, band_name in enumerate(band_names):
        band = Config.get_band(band_name)
        native = band.cutout_size_for_arcsec(arcsec)
        # One stage per band; the step bar tracks band progress (the
        # downloader's own tqdm covers per-file detail in the raw log).
        reporter.set_stage(f"downloading {band_name}")
        reporter.set_step(i, n_bands, band_name)
        print(f"=== {band_name}  (instrument={band.archive_instrument}"
              f"{('/' + band.archive_filter) if band.archive_filter else ''}, "
              f"native_size={native} px) ===")
        cfg = DownloadConfig.for_band(
            band_name,
            cutout_size_vis_pixels=args.vis_pixels,
            max_workers=args.workers,
        )
        downloader = EuclidCutoutDownloader(cat, cfg)
        t_band = time.perf_counter()
        result = downloader.download(show_progress=True)
        summary[band_name] = result
        if result.get("corrupted", 0) or result.get("failed", 0):
            reporter.warn(
                f"{band_name}: corrupted={result.get('corrupted', 0)}, "
                f"failed={result.get('failed', 0)}"
            )
        print(f"  → {band_name}: downloaded={result['downloaded']}, "
              f"valid={result['valid']}, corrupted={result['corrupted']}, "
              f"failed={result.get('failed', 0)}  "
              f"[{time.perf_counter() - t_band:.0f}s]\n")
    reporter.set_step(n_bands, n_bands, "done")

    print("=" * 50)
    print(f"Summary  ({(time.perf_counter() - t0) / 60:.1f} min total):")
    for name, r in summary.items():
        print(f"  {name:5s}  +{r['downloaded']:4d}  "
              f"valid={r['valid']}  corrupted={r['corrupted']}  "
              f"failed={r.get('failed', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
