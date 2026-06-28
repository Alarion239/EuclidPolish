#!/usr/bin/env python
"""Download cutouts for every Euclid band sharing one angular field size.

Multi-band wrapper around :meth:`EuclidCatalog.download_cutouts`. Picks one
``cutout_size_vis_pixels`` value, converts it to each band's native
pixel count via :meth:`BandConfig.cutout_size_for_arcsec`, and runs the
downloader once per band. Each band's catalog flags are tracked
independently under the shared ``stars.csv``.

By default all bands download concurrently (``--band-workers`` = up to one
thread per band); pass ``--band-workers 1`` for the old one-band-at-a-time
behaviour. Total concurrency is ``band-workers × workers``.

Usage:
    python scripts/download_all_bands.py
    python scripts/download_all_bands.py --vis-pixels 512 --workers 8
    python scripts/download_all_bands.py --band-workers 1   # sequential
    python scripts/download_all_bands.py --bands VIS,Y_E --vis-pixels 256
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.catalog.catalog_object import CatalogObject
from euclid_polish.catalog.client import EuclidAuthError, EuclidCatalog
from euclid_polish.catalog.cutout_integrity import validate_all_cutouts
from euclid_polish.catalog.downloader import DownloadConfig
from euclid_polish.config import Config
from euclid_polish.observability.reporter import Reporter


def _ensure_euclid_env() -> None:
    """Populate EUCLID_USER/EUCLID_PASSWORD from ~/.euclid_credentials if unset.

    The WebUI writes that two-line file on FASRC; :class:`EuclidCatalog` reads
    only the environment, so we bridge the file into env at the script boundary.
    """
    if os.environ.get("EUCLID_USER") and os.environ.get("EUCLID_PASSWORD"):
        return
    path = os.path.expanduser(Config.DEFAULT_CREDENTIALS_FILE)
    if not os.path.isfile(path):
        return
    try:
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError:
        return
    if len(lines) >= 2:
        os.environ.setdefault("EUCLID_USER", lines[0])
        os.environ.setdefault("EUCLID_PASSWORD", lines[1])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=Config.DEFAULT_OUTPUT_DIR,
                    help="Star catalog root")
    ap.add_argument("--vis-pixels", type=int, default=Config.DEFAULT_CUTOUT_SIZE,
                    help="Cutout size in VIS pixels (= 0.10\"/pix). Each band "
                         "uses its own native pixel count covering the same "
                         "angular field.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel cutout downloads within a single band")
    ap.add_argument("--band-workers", type=int, default=4,
                    help="How many bands to download concurrently (1 = the old "
                         "sequential behaviour). Total concurrency = "
                         "band-workers × workers.")
    ap.add_argument("--bands",
                    default=",".join(b.name for b in Config.BANDS),
                    help="Comma-separated band list")
    ap.add_argument("--retry-failed", action="store_true",
                    help="Clear each band's download_failed flags first, so "
                         "stars wrongly flagged by a transient TAP/session "
                         "error get re-attempted (truly uncovered ones re-flag).")
    return ap.parse_args()


def _download_one_band(band_name, *, eclient, output_dir, vis_pixels, workers,
                       arcsec, progress_cb, show_progress, retry_failed=False):
    """Download every cutout for one band. Returns the result dict.

    Self-contained (loads its own ``CatalogObject`` list + ``DownloadConfig``) so
    it is safe to run several of these concurrently — each band writes to its own
    ``cutouts/<band>/`` directory and tracks distinct per-(band, size) catalog
    flags, and ``CatalogObject.write`` is atomic.
    """
    band = Config.get_band(band_name)
    native = band.cutout_size_for_arcsec(arcsec)
    print(f"=== {band_name}  (instrument={band.archive_instrument}"
          f"{('/' + band.archive_filter) if band.archive_filter else ''}, "
          f"native_size={native} px) ===", flush=True)
    cfg = DownloadConfig.for_band(
        band_name, cutout_size_vis_pixels=vis_pixels, max_workers=workers,
    )
    objects = CatalogObject.read(os.path.join(output_dir, Config.CATALOG_FILE))
    t_band = time.perf_counter()
    result = eclient.download_cutouts(
        objects, output_dir, cfg, show_progress=show_progress,
        progress_cb=progress_cb, retry_failed=retry_failed)
    print(f"  → {band_name}: downloaded={result['downloaded']}, "
          f"valid={result['valid']}, corrupted={result['corrupted']}, "
          f"failed={result.get('failed', 0)}  "
          f"[{time.perf_counter() - t_band:.0f}s]\n", flush=True)
    return result


def run_bands(band_names, *, eclient, output_dir, vis_pixels, workers, arcsec,
              reporter, band_workers, logged_in, retry_failed=False):
    """Download all bands, sequentially or several at once.

    ``band_workers == 1`` keeps the old one-band-at-a-time path (with a TAP
    re-login between bands). ``band_workers > 1`` runs that many bands
    concurrently through a thread pool, aggregating their per-cutout progress
    into one combined bar. Returns ``{band_name: result_dict}``.
    """
    summary: dict[str, dict] = {}
    n_bands = len(band_names)
    band_workers = max(1, min(n_bands, band_workers))

    if band_workers > 1:
        reporter.set_stage(
            f"downloading {n_bands} bands ({band_workers} at a time)")
        print(f"Parallel download: {band_workers} bands at once × {workers} "
              f"cutout-workers = up to {band_workers * workers} concurrent "
              f"requests\n", flush=True)
        # Lock-protected per-band (cur, tot) so the combined bar is a clean sum
        # across whichever bands are in flight; tqdm is off to avoid garble.
        prog: dict[str, tuple[int, int]] = {}
        lock = threading.Lock()

        def make_cb(bn):
            def cb(cur, tot, lbl):
                with lock:
                    prog[bn] = (cur, tot)
                    g_cur = sum(c for c, _ in prog.values())
                    g_tot = sum(t for _, t in prog.values())
                reporter.set_step(g_cur, max(g_tot, 1), f"{bn} {lbl}")
            return cb

        with ThreadPoolExecutor(max_workers=band_workers) as pool:
            futs = {
                pool.submit(_download_one_band, bn, eclient=eclient,
                            output_dir=output_dir, vis_pixels=vis_pixels,
                            workers=workers, arcsec=arcsec,
                            progress_cb=make_cb(bn), show_progress=False,
                            retry_failed=retry_failed): bn
                for bn in band_names
            }
            for fut in as_completed(futs):
                bn = futs[fut]
                try:
                    summary[bn] = fut.result()
                except Exception as e:                       # noqa: BLE001
                    reporter.warn(f"{bn}: band download failed: {e}")
                    print(f"  ✗ {bn}: {type(e).__name__}: {e}", flush=True)
                    summary[bn] = {"downloaded": 0, "valid": 0,
                                   "corrupted": 0, "failed": 0}
    else:
        for i, bn in enumerate(band_names):
            reporter.set_stage(f"downloading {bn} ({i + 1}/{n_bands})")
            # Refresh the TAP session before each band — a long band (VIS can
            # take ~1h) lets the session lapse, which made the next band's
            # mosaic query return None and fail.
            if logged_in and i > 0:
                eclient.relogin()
            def cb(cur, tot, lbl, _b=bn):
                return (reporter.set_step(cur, tot, f"{_b} {lbl}"))
            summary[bn] = _download_one_band(
                bn, eclient=eclient, output_dir=output_dir, vis_pixels=vis_pixels,
                workers=workers, arcsec=arcsec, progress_cb=cb, show_progress=True,
                retry_failed=retry_failed)

    for bn, r in summary.items():
        if r.get("corrupted", 0) or r.get("failed", 0):
            reporter.warn(f"{bn}: corrupted={r.get('corrupted', 0)}, "
                          f"failed={r.get('failed', 0)}")
    reporter.set_step(n_bands, n_bands, "done")
    return summary


def main() -> int:
    args = parse_args()
    reporter = Reporter.from_env()
    band_names = [n.strip() for n in args.bands.split(",") if n.strip()]
    arcsec = args.vis_pixels * Config.BAND_VIS.pixel_scale_lr_arcsec

    print(f"angular field   = {arcsec:.2f}\"  (= {args.vis_pixels} VIS px)")
    print(f"bands           = {band_names}")
    print(f"output dir      = {args.output_dir}")
    print(f"workers / band  = {args.workers}")
    print(f"bands in parallel = {max(1, min(len(band_names), args.band_workers))}\n")

    catalog_path = os.path.join(args.output_dir, Config.CATALOG_FILE)
    if not os.path.exists(catalog_path):
        reporter.error(f"no catalog at {catalog_path}")
        print(f"✗ no catalog at {catalog_path}")
        return 1

    # Log in to the Euclid archive (proprietary cutouts need it). Reads
    # EUCLID_USER/EUCLID_PASSWORD env, bridging ~/.euclid_credentials (written
    # by the WebUI "Euclid archive login" form) into env first. Non-interactive.
    reporter.set_stage("authenticating with Euclid archive")
    _ensure_euclid_env()
    try:
        eclient = EuclidCatalog()
        logged_in = True
        print(f"✓ Euclid archive login OK (user={eclient.user})")
    except EuclidAuthError:
        eclient = EuclidCatalog._unauthenticated()
        logged_in = False
        reporter.warn(
            "not authenticated with the Euclid archive — proprietary "
            "cutouts will fail. Set credentials in the WebUI (Cutouts page)."
        )
        print("⚠️  proceeding unauthenticated (public data only)")

    t0 = time.perf_counter()
    summary = run_bands(
        band_names, eclient=eclient, output_dir=args.output_dir,
        vis_pixels=args.vis_pixels, workers=args.workers,
        arcsec=arcsec, reporter=reporter, band_workers=args.band_workers,
        logged_in=logged_in, retry_failed=args.retry_failed,
    )

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
    integ = validate_all_cutouts(args.output_dir, band_names, reporter=reporter)
    print(f"\nIntegrity: checked {integ['checked']} cutouts, "
          f"{integ['unopenable']} unopenable → "
          f"{integ['valid_all_bands']} stars valid in all "
          f"{integ['n_bands']} bands")
    return 0


if __name__ == "__main__":
    sys.exit(main())
