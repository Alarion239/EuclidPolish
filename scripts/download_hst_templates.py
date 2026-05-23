#!/usr/bin/env python
"""Download HST/ACS F814W cutouts for a curated COSMOS sample.

Builds the HR-morphology template library used by
:class:`euclid_polish.sky.hst_templates.HSTTemplateRenderer`. The library
seeds the generator with real-galaxy morphology (spirals, irregulars,
AGN hosts) that the Sersic B+D decomposition cannot reproduce.

Selection strategy (see :func:`select_template_galaxies`):

* requires the COSMOS2025 master catalog already on disk
* prefers BRIGHT, EXTENDED galaxies (small fraction of the full catalog
  has both — but those are the ones where downsampling preserves the most
  structure when each template gets shrunk by a factor of 2-6)
* deduplicates against any templates already in the on-disk library

Usage:
    python scripts/download_hst_templates.py                     # 200 templates
    python scripts/download_hst_templates.py --n 1000 --workers 8
    python scripts/download_hst_templates.py --size 384 --max-mag 22.0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.hst.catalog import HSTTemplateLibrary
from euclid_polish.hst.downloader import (
    HSTCutoutDownloader, HSTDownloadConfig,
)
from euclid_polish.hst.types import HSTCutoutMetadata
from euclid_polish.sky.cosmos2025 import open_cosmos2025, Cosmos2025Catalog


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200,
                    help="Target number of templates to download")
    ap.add_argument("--size", type=int, default=Config.HST_DEFAULT_CUTOUT_PIX,
                    help="HST cutout side in native pixels (~0.04\"/pix)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent MAST requests (4 is polite)")
    ap.add_argument("--max-mag", type=float, default=22.0,
                    help="Faint cutoff on total VIS (F814W) magnitude — "
                         "templates fainter than this carry too little "
                         "signal to survive the downsample.")
    ap.add_argument("--min-disk-re-arcsec", type=float, default=0.30,
                    help="Lower bound on disk Re (arcsec) — extended sources "
                         "preserve more structure under rescaling.")
    ap.add_argument("--catalog", default=Config.COSMOS2025_CATALOG_PATH,
                    help="COSMOS2025 FITS path")
    ap.add_argument("--output-dir", default=Config.HST_TEMPLATES_DIR,
                    help="Where templates + index CSV are written")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for galaxy selection (reproducible)")
    return ap.parse_args()


def select_template_galaxies(
    cat: Cosmos2025Catalog,
    *,
    n_target: int,
    max_mag: float,
    min_disk_re_arcsec: float,
    rng: np.random.Generator,
) -> List[Tuple[int, float, float]]:
    """Pick a random sample of bright extended COSMOS galaxies.

    Filters down to galaxies with total F814W brighter than ``max_mag``
    and disk R_e ≥ ``min_disk_re_arcsec``, then returns
    ``[(cosmos_id, ra, dec), ...]`` for ``n_target`` randomly chosen
    survivors (without replacement). If the filter leaves fewer than
    ``n_target`` candidates, returns all of them.
    """
    # Total VIS flux in electrons → mag.
    vis_band = Config.BAND_VIS
    vis_idx  = Config.LR_INPUT_BAND_NAMES.index("VIS")
    total_flux_vis = cat.bulge_flux_e[:, vis_idx] + cat.disk_flux_e[:, vis_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        mag_total = (
            vis_band.zeropoint_ab_e_per_s
            + 2.5 * np.log10(vis_band.exposure_time_s * vis_band.n_exposures)
            - 2.5 * np.log10(np.maximum(total_flux_vis, 1e-30))
        )

    mask = (
        np.isfinite(mag_total)
        & (mag_total < max_mag)
        & (cat.disk_re_arcsec >= min_disk_re_arcsec)
    )
    cand = np.flatnonzero(mask)
    if cand.size == 0:
        raise RuntimeError(
            f"No COSMOS galaxies pass max_mag={max_mag}, "
            f"min_disk_re_arcsec={min_disk_re_arcsec}."
        )
    if cand.size > n_target:
        cand = rng.choice(cand, size=n_target, replace=False)

    return [
        (
            int(cat.catalog_id[i]),
            float(cat.ra_deg[i]),
            float(cat.dec_deg[i]),
        )
        for i in cand
    ]


def main() -> int:
    args = parse_args()
    print("=" * 64)
    print(f"  HST template library bootstrap")
    print("=" * 64)
    print(f"  target N      = {args.n}")
    print(f"  cutout size   = {args.size} native HST pixels "
          f"(~{args.size * Config.HST_NATIVE_PIXEL_SCALE_ARCSEC:.2f}\")")
    print(f"  max VIS mag   = {args.max_mag}")
    print(f"  min disk Re   = {args.min_disk_re_arcsec}\"")
    print(f"  output dir    = {args.output_dir}")
    print()

    rng = np.random.default_rng(args.seed)
    print(f"[1/3] loading COSMOS2025 catalog ({args.catalog}) ...")
    cat = open_cosmos2025(args.catalog)
    print(f"      {len(cat):,} galaxies after quality cuts")

    print(f"[2/3] selecting {args.n} candidate positions ...")
    positions = select_template_galaxies(
        cat,
        n_target=args.n,
        max_mag=args.max_mag,
        min_disk_re_arcsec=args.min_disk_re_arcsec,
        rng=rng,
    )
    print(f"      {len(positions)} positions selected")

    print(f"[3/3] downloading cutouts (this can take a few minutes) ...")
    dl = HSTCutoutDownloader(
        HSTDownloadConfig(
            output_dir=args.output_dir,
            size_pix=args.size,
            max_workers=args.workers,
        ),
    )
    t0 = time.perf_counter()
    summary = dl.download_many(positions, show_progress=True)
    dt = time.perf_counter() - t0
    print()
    print(f"  downloaded  = {summary['downloaded']}")
    print(f"  skipped     = {summary['skipped']}  (already in cache)")
    print(f"  failed      = {summary['failed']}  (no HAP coverage at position)")
    print(f"  elapsed     = {dt:.1f} s")

    # Reconcile the on-disk index with what we now have.
    print()
    print(f"      indexing library at {args.output_dir} ...")
    lib = HSTTemplateLibrary(root=args.output_dir)
    # Add any cutouts that completed in this run.
    pos_by_id = {cid: (ra, dec) for (cid, ra, dec) in positions}
    for cid in pos_by_id:
        if dl.already_have(cid):
            ra, dec = pos_by_id[cid]
            fname = os.path.basename(dl.cutout_path(cid))
            meta = HSTCutoutMetadata(
                cosmos_id          = cid,
                ra_deg             = ra,
                dec_deg            = dec,
                size_pix           = args.size,
                pixel_scale_arcsec = Config.HST_NATIVE_PIXEL_SCALE_ARCSEC,
                filename           = fname,
            )
            lib.add(meta, save=False)
    lib.save()
    print(f"      library now holds {len(lib)} templates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
