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
from euclid_polish.euclid import auth
from euclid_polish.euclid.catalog import StarCatalog
from euclid_polish.euclid.cutout_integrity import validate_all_cutouts
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

    # Log in to the Euclid archive (proprietary cutouts need it). Tries
    # EUCLID_USER/EUCLID_PASSWORD env, then ~/.euclid_credentials (written
    # by the WebUI "Euclid archive login" form). Non-interactive on FASRC.
    reporter.set_stage("authenticating with Euclid archive")
    logged_in = auth.login(allow_interactive=False)
    if logged_in:
        print(f"✓ Euclid archive login OK (user={auth.current_user()})")
    else:
        reporter.warn(
            "not authenticated with the Euclid archive — proprietary "
            "cutouts will fail. Set credentials in the WebUI (Cutouts page)."
        )
        print("⚠️  proceeding unauthenticated (public data only)")

    summary: dict[str, dict] = {}
    n_bands = len(band_names)
    t0 = time.perf_counter()
    for i, band_name in enumerate(band_names):
        band = Config.get_band(band_name)
        native = band.cutout_size_for_arcsec(arcsec)
        # One stage per band; the per-cutout progress_cb below drives the
        # step bar (current/total within the band).
        reporter.set_stage(f"downloading {band_name} ({i + 1}/{n_bands})")
        print(f"=== {band_name}  (instrument={band.archive_instrument}"
              f"{('/' + band.archive_filter) if band.archive_filter else ''}, "
              f"native_size={native} px) ===")
        # Refresh the TAP session before each band — a long band (VIS can take
        # ~1h) lets the session lapse, which made the next band's mosaic query
        # return None and fail. Re-login is idempotent + cheap.
        if logged_in and i > 0:
            auth.login(allow_interactive=False)
        cfg = DownloadConfig.for_band(
            band_name,
            cutout_size_vis_pixels=args.vis_pixels,
            max_workers=args.workers,
        )
        downloader = EuclidCutoutDownloader(cat, cfg)
        t_band = time.perf_counter()
        # Per-cutout progress → WebUI bar (resets per band; stage names the
        # band). Emitting every cutout is cheap (JSONL append) and the
        # stderr echo is rate-limited inside Reporter.set_step.
        result = downloader.download(
            show_progress=True,
            progress_cb=lambda cur, tot, lbl, _b=band_name:
                reporter.set_step(cur, tot, f"{_b} {lbl}"),
        )
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

    # Integrity pass: open every cutout just downloaded and (re)derive the
    # catalog's per-(band, size) validity, so downstream tasks can trust
    # "valid in all 4 bands" without re-opening every file themselves.
    reporter.set_stage("validating cutouts (integrity)")
    integ = validate_all_cutouts(cat, cat.load(), band_names, reporter=reporter)
    print(f"\nIntegrity: checked {integ['checked']} cutouts, "
          f"{integ['unopenable']} unopenable → "
          f"{integ['valid_all_bands']} stars valid in all "
          f"{integ['n_bands']} bands")
    return 0


if __name__ == "__main__":
    sys.exit(main())
