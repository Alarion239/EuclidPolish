#!/usr/bin/env python
"""Download COSMOS F814W HLSP mosaic tiles into ``$DATA_DIR/hst_hlsp/``.

This script is meant to run on a FASRC compute node via sbatch (see
:class:`euclid_polish.web.fasrc_pipeline.HSTDownloadStep`). It is also
runnable locally for testing — provide ``--dry-run`` to skip actual
downloads.

Source data: COSMOS HST/ACS-WFC F814W mosaic v1.3, 9×9 tile grid,
~1.9 GB per tile.

Output layout::

    $DATA_DIR/hst_hlsp/
        hlsp_cosmos_hst_acs-wfc_mosaic-X-Y_f814w_v1.3_img.fits   (~1.9 GB each)
        download_summary.json                                    (provenance)

Idempotent: re-running skips tiles already on disk with non-zero size.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config


HLSP_DIR_NAME = "hst_hlsp"
TARGET_NAME   = "COSMOS"
FILTER        = "F814W"
OBS_COLLECTION = "HLSP"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-tiles", type=int, default=25,
                   help="Number of tiles to download (max 81). Tiles are "
                        "ordered by central RA/Dec and the first N are taken.")
    p.add_argument("--output-dir", default=None,
                   help="Override the default $DATA_DIR/hst_hlsp/ location.")
    p.add_argument("--dry-run", action="store_true",
                   help="List the tiles that would be downloaded and exit.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir or os.path.join(Config.DATA_DIR, HLSP_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 64)
    print(f"  COSMOS HLSP F814W download")
    print("=" * 64)
    print(f"  target N tiles  = {args.n_tiles}")
    print(f"  output dir      = {out_dir}")
    print(f"  dry run         = {args.dry_run}")
    print()

    t0 = time.time()

    print("[1/3] querying MAST HLSP catalog ...")
    from astroquery.mast import Observations
    obs = Observations.query_criteria(
        obs_collection=OBS_COLLECTION,
        target_name=TARGET_NAME,
        filters=FILTER,
    )
    # Keep only mosaic tile entries (skip individual-source HLSP rows).
    mask = ["acs-wfc_mosaic" in str(r["obs_id"]) for r in obs]
    obs = obs[mask]
    print(f"      found {len(obs)} mosaic tiles in MAST")

    if len(obs) == 0:
        print("ERROR: no COSMOS HLSP tiles found — MAST query returned empty")
        return 1

    # Stable ordering: by obs_id (which encodes the X-Y grid position)
    obs.sort("obs_id")
    obs = obs[: int(args.n_tiles)]
    print(f"      selected {len(obs)} tiles for download")

    print("[2/3] resolving science-product file list ...")
    prods = Observations.get_product_list(obs)
    sci = prods[prods["productType"] == "SCIENCE"]
    sci_fits = sci[[str(f).endswith(".fits") for f in sci["productFilename"]]]
    print(f"      {len(sci_fits)} SCIENCE FITS products")

    # Skip downloads where the destination already exists and is non-empty.
    pending = []
    skipped = 0
    for row in sci_fits:
        fname = str(row["productFilename"])
        target_path = os.path.join(out_dir, fname)
        if os.path.isfile(target_path) and os.path.getsize(target_path) > 0:
            skipped += 1
            continue
        pending.append(row)
    print(f"      {skipped} already on disk (skipped)")
    print(f"      {len(pending)} pending download")

    if args.dry_run:
        print("\nDRY RUN — would download:")
        for row in pending:
            print(f"  {row['productFilename']}  ({row['size'] / 1e6:.0f} MB)")
        print(f"\nTotal pending: {sum(r['size'] for r in pending) / 1e9:.1f} GB")
        runtime = time.time() - t0
        print(f"\nRUNTIME_SECONDS={runtime:.1f}")
        return 0

    print("[3/3] downloading (this can take a while) ...")
    if pending:
        from astropy.table import Table
        manifest = Observations.download_products(
            Table(rows=pending), download_dir=out_dir,
            cache=True, mrp_only=False,
        )
        # MAST writes under mastDownload/HLSP/<id>/<file>. Flatten to out_dir.
        if manifest is not None:
            for r in manifest:
                src = str(r["Local Path"])
                if not os.path.isfile(src):
                    continue
                dst = os.path.join(out_dir, os.path.basename(src))
                if os.path.abspath(src) == os.path.abspath(dst):
                    continue
                try:
                    os.replace(src, dst)
                except OSError as e:
                    print(f"  failed to move {src} → {dst}: {e}")
        # Clean up the mastDownload scratch.
        scratch = os.path.join(out_dir, "mastDownload")
        if os.path.isdir(scratch):
            import shutil
            shutil.rmtree(scratch, ignore_errors=True)

    # Sanity audit: list final on-disk tiles.
    final = sorted(
        f for f in os.listdir(out_dir)
        if f.startswith("hlsp_cosmos_hst_acs-wfc_mosaic")
           and f.endswith(".fits")
    )
    print(f"\n  on-disk tiles    = {len(final)}")
    total_gb = sum(os.path.getsize(os.path.join(out_dir, f)) for f in final) / 1e9
    print(f"  on-disk size     = {total_gb:.1f} GB")

    summary = {
        "target":      TARGET_NAME,
        "filter":      FILTER,
        "n_requested": args.n_tiles,
        "n_skipped":   skipped,
        "n_downloaded": len(pending),
        "n_on_disk":   len(final),
        "size_gb":     round(total_gb, 2),
        "tiles":       final,
        "elapsed_s":   round(time.time() - t0, 1),
    }
    summary_path = os.path.join(out_dir, "download_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote summary    → {summary_path}")

    runtime = time.time() - t0
    print(f"\nRUNTIME_SECONDS={runtime:.1f}")
    print(f"N_TILES_ON_DISK={len(final)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
