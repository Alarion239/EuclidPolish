#!/usr/bin/env python
"""Extract empirical ePSFs for every Euclid band from local star cutouts.

For each of ``Config.BANDS`` this script:

  1. Looks for cutouts in the band-specific directory:
       VIS  → ``Config.DEFAULT_OUTPUT_DIR/cutouts``
       NISP → ``Config.NISP_DEFAULT_OUTPUT_DIR_BY_BAND[band.name]/cutouts``
  2. Loads + accepts the good cutouts (saturation/edge rejection), then
     **spatially clusters** them by catalog sky position (K-Means++) into
     groups of ``--stars-per-psf`` and builds **one ePSF per cluster** — the
     Euclid PSF varies across the focal plane, so a band gets K≈n_good/N PSFs
     (e.g. 3000 good stars at N=100 → ~30 PSFs) instead of one average kernel.
  3. Saves them as a multi-extension FITS to
     ``data/euclid_psf/<band.psf_fits_filename>`` (e.g. ``euclid_psf_VIS.fits``):
     HDU[0] is the mean PSF (so single-PSF readers keep working), HDU[1..K] are
     the cluster PSFs. Generation draws a random convex blend of the K kernels.
  4. If no cutouts are present for a band, prints a clear note and
     continues — the loader will fall back to a Gaussian PSF for that band.

Usage:
    python scripts/extract_all_band_psfs.py --stars-per-psf 100 --vis-pixels 256
    python scripts/extract_all_band_psfs.py --num-stars 100 --cutout-size 256
    python scripts/extract_all_band_psfs.py --bands VIS,Y_E
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# Cap per-process BLAS threads BEFORE numpy/scipy import. We run the bands
# in a process pool (one worker per band); without this each worker would
# spawn one BLAS thread per core, oversubscribing the 4 CPUs the job locks.
# ``setdefault`` honours an explicit override from the environment.
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from sklearn.cluster import KMeans

from euclid_polish.config import BandConfig, Config
from euclid_polish.euclid.psf_extractor import (
    PSFExtractionConfig, PSFExtractor,
)
from euclid_polish.psf import PSF, PSFSet
from euclid_polish.observability.reporter import Reporter


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


def _load_star_positions(stars_csv: str) -> Dict[int, Tuple[float, float]]:
    """Map star ``id → (ra, dec)`` from the catalog CSV.

    Used to spatially cluster each band's good stars before extraction.
    Returns an empty dict if the file is missing — callers then fall
    back to a single ePSF from all accepted stars (the old behaviour).
    """
    if not os.path.isfile(stars_csv):
        return {}
    positions: Dict[int, Tuple[float, float]] = {}
    with open(stars_csv) as fh:
        for row in csv.DictReader(fh):
            try:
                positions[int(row["id"])] = (float(row["ra"]), float(row["dec"]))
            except (KeyError, ValueError, TypeError):
                continue
    return positions


def _merge_small_clusters(
    clusters: List[List[int]],
    coord_by_idx: Dict[int, np.ndarray],
    min_stars: int,
) -> List[List[int]]:
    """Merge any cluster smaller than ``min_stars`` into its nearest
    neighbour (by centroid) until every cluster meets the floor or only one
    remains. Guarantees ≥ ``min_stars`` per cluster whenever the total
    allows it (a single field with < min_stars stars stays as one cluster)."""
    clusters = [list(c) for c in clusters]

    def centroid(c: List[int]) -> np.ndarray:
        return np.mean([coord_by_idx[i] for i in c], axis=0)

    while len(clusters) > 1:
        sizes = [len(c) for c in clusters]
        s = int(np.argmin(sizes))
        if sizes[s] >= min_stars:
            break
        cs = centroid(clusters[s])
        nearest, best = None, None
        for j, c in enumerate(clusters):
            if j == s:
                continue
            d = float(np.sum((centroid(c) - cs) ** 2))
            if best is None or d < best:
                best, nearest = d, j
        clusters[nearest].extend(clusters[s])
        del clusters[s]
    return clusters


def cluster_star_indices(
    ids: List[int],
    positions: Dict[int, Tuple[float, float]],
    stars_per_psf: int,
    *,
    min_stars: int = 50,
) -> List[List[int]]:
    """Group the ``ids`` into spatially-coherent clusters of ~``stars_per_psf``.

    K = max(1, round(n / stars_per_psf)) clusters via K-Means++ on the sky
    positions (RA scaled by cos(dec₀) so Euclidean distance ≈ angular
    separation), then any cluster smaller than ``min_stars`` is merged into
    its nearest neighbour — so the average is ~``stars_per_psf`` but no ePSF
    is ever built from fewer than ``min_stars`` stars (noisy under-sampled
    PSFs). Returns a list of clusters, each a list of positions *into*
    ``ids`` (so callers can index their parallel star list).

    Stars without a catalog position are dropped. With < ``stars_per_psf``
    positioned stars, or no positions at all, returns a single cluster of
    every index (one ePSF, the old behaviour).
    """
    pts = []
    keep = []
    for i, sid in enumerate(ids):
        pos = positions.get(sid)
        if pos is not None:
            keep.append(i)
            pts.append(pos)
    if len(keep) < max(1, int(stars_per_psf)):
        # Too few positioned stars to split — one ePSF from everything.
        return [list(range(len(ids)))]

    arr = np.asarray(pts, dtype=np.float64)
    dec0 = float(np.mean(arr[:, 1]))
    scaled = np.column_stack([
        arr[:, 0] * math.cos(math.radians(dec0)),
        arr[:, 1],
    ])
    k = max(1, round(len(keep) / float(stars_per_psf)))
    if k == 1:
        return [list(keep)]
    labels = KMeans(
        n_clusters=k, init="k-means++", n_init=10, random_state=0,
    ).fit_predict(scaled)
    clusters: List[List[int]] = [[] for _ in range(k)]
    for local_i, lab in enumerate(labels):
        clusters[int(lab)].append(keep[local_i])
    clusters = [c for c in clusters if c]
    # Enforce the per-cluster floor (merge under-sized clusters).
    coord_by_idx = {keep[j]: scaled[j] for j in range(len(keep))}
    return _merge_small_clusters(clusters, coord_by_idx, int(min_stars))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-stars", type=int, default=None,
                    help="Optional cap on stars considered per band "
                         "(default: use ALL good cutouts). The stars are then "
                         "clustered into groups of --stars-per-psf.")
    ap.add_argument("--stars-per-psf", type=int, default=100,
                    help="Target (average) stars per extracted PSF (N). The "
                         "band's good stars are K-Means++ clustered by sky "
                         "position into K=round(n_good/N) groups, one ePSF "
                         "each. 3000 good stars at N=100 → ~30 PSFs. Default "
                         "100.")
    ap.add_argument("--min-stars-per-psf", type=int, default=50,
                    help="Hard floor on stars per cluster: clusters smaller "
                         "than this are merged into their nearest neighbour, "
                         "so no ePSF is built from too few (noisy) stars. "
                         "Default 50.")
    ap.add_argument("--stars-csv", default=os.path.join(
                        Config.DEFAULT_OUTPUT_DIR, "stars.csv"),
                    help="Catalog CSV (id, ra, dec) used to spatially cluster "
                         "stars. Missing → single ePSF from all good stars.")
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
    ap.add_argument("--max-procs", type=int, default=4,
                    help="Bands to extract in parallel (one process per "
                         "band; capped at the band count). The FASRC step "
                         "locks the allocation to 4 CPUs to match the 4 "
                         "Euclid bands.")
    return ap.parse_args()


def extract_band(band: BandConfig, args: argparse.Namespace,
                 reporter: Optional[Reporter] = None) -> bool:
    reporter = reporter or Reporter(events_path=None)  # no-op when unset
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
        reporter.warn(f"{band.name}: cutout directory not found — skipping")
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
    # tqdm (the "Processing cutouts" bar + EPSFBuilder's) only when running
    # interactively. Under SLURM (events path set) the Reporter drives the UI
    # progress bar, so the per-iteration tqdm would just flood the .err log.
    cfg = PSFExtractionConfig(
        progress_bar=(reporter.events_path is None),
        psf_size=psf_size,
        output_size=args.output_size,
        oversampling=band.epsf_oversampling,
    )
    print(f"  psf-size:    {psf_size} px (native centred crop from each star)")
    extractor = PSFExtractor(cfg)
    all_files = extractor.get_cutout_files(cutout_dir, cutout_size=cutout_size)
    if not all_files:
        reporter.warn(f"{band.name}: no cutouts of size {cutout_size} found — skipping")
        print(f"  ⚠️  no cutouts of size {cutout_size} found — skipping.")
        return False

    selected = extractor.select_files(all_files, num_stars=args.num_stars)
    print(f"  considering {len(selected)} of {len(all_files)} available stars")

    # Load + accept (saturation/edge reject) in one pass, keeping ids so we
    # can join each star to its catalog sky position for clustering.
    accepted = extractor.extract_accepted_stars(selected)
    if not accepted:
        reporter.warn(f"{band.name}: no usable (non-saturated) stars — skipping")
        print("  ⚠️  no usable stars after saturation/edge rejection — skipping.")
        return False
    ids = [sid for sid, _ in accepted]
    stars = [star for _, star in accepted]
    print(f"  accepted {len(stars)} good stars")

    # Spatially cluster the good stars into groups of ~--stars-per-psf and
    # extract one ePSF per cluster (the PSF varies across the field).
    positions = _load_star_positions(args.stars_csv)
    if not positions:
        print(f"  ⚠️  no positions in {args.stars_csv} — single ePSF from all "
              f"good stars (no spatial clustering).")
    clusters = cluster_star_indices(ids, positions, args.stars_per_psf,
                                    min_stars=args.min_stars_per_psf)
    print(f"  {len(clusters)} cluster(s) "
          f"(target {args.stars_per_psf} stars/PSF, "
          f"min {args.min_stars_per_psf}): "
          f"sizes {[len(c) for c in clusters]}")

    # Pixel scale on the *oversampled* ePSF grid: native / oversampling.
    # By picking ``epsf_oversampling`` so this equals 0.05"/pix for every
    # band, all ePSFs land on the same HR grid the forward model uses.
    epsf_pixel_scale = band.epsf_pixel_scale_arcsec

    psfs: List[PSF] = []
    centroids: List[Tuple[float, float]] = []
    star_counts: List[int] = []
    n_clusters = len(clusters)
    # Announce this band's share of the work up front so the cumulative
    # cross-band bar knows its total before the first ePSF lands.
    reporter.set_worker_step(band.name, 0, n_clusters, f"{band.name}: 0/{n_clusters}")
    for ci, cluster in enumerate(clusters):
        cluster_stars = [stars[i] for i in cluster]
        try:
            epsf, _ = extractor.build_epsf_from_stars(cluster_stars)
        except Exception as e:
            reporter.warn(f"{band.name}: cluster {ci} failed "
                          f"({type(e).__name__}: {e}) — skipping cluster")
            print(f"  ✗ cluster {ci} failed: {type(e).__name__}: {e}")
        else:
            psfs.append(extractor.psf_from_epsf(epsf, epsf_pixel_scale))
            star_counts.append(len(cluster_stars))   # sampling weight
            pts = [positions[ids[i]] for i in cluster if ids[i] in positions]
            if pts:
                centroids.append((float(np.mean([p[0] for p in pts])),
                                  float(np.mean([p[1] for p in pts]))))
            else:
                centroids.append((float("nan"), float("nan")))
        # Report after every cluster (built or skipped) so the bar always
        # advances to n_clusters; the consumer sums these across bands.
        reporter.set_worker_step(band.name, ci + 1, n_clusters,
                                 f"{band.name}: PSF {ci + 1}/{n_clusters} "
                                 f"({len(cluster_stars)} stars)")

    if not psfs:
        reporter.error(f"{band.name}: every cluster failed to build")
        print("  ✗ every cluster failed to build")
        return False

    have_centroids = bool(positions) and all(
        np.isfinite(c[0]) and np.isfinite(c[1]) for c in centroids)
    psf_set = PSFSet.from_psfs(
        psfs, centroids=centroids if have_centroids else None,
        n_stars=star_counts)
    os.makedirs(args.psf_dir, exist_ok=True)
    saved = psf_set.save(args.psf_dir, filename=band.psf_fits_filename)
    print(f"  ✓ saved {saved}")
    print(f"     {psf_set.n} PSF(s), shape={psf_set.shape}, "
          f"pixel_scale={psf_set.pixel_scale:.4f}\"/pix")
    return True


def _extract_band_worker(task: Tuple[str, argparse.Namespace]) -> Tuple[str, bool]:
    """Process-pool entry point: extract one band's ePSF.

    Top-level (picklable). Each worker builds its own :class:`Reporter`
    from the env — the per-job events file is append-only and POSIX-atomic
    for sub-PIPE_BUF writes, so the 4 band workers share it safely without
    passing the (unpicklable, lock-bearing) parent reporter across the
    fork.
    """
    band_name, args = task
    band = Config.get_band(band_name)
    reporter = Reporter.from_env()
    ok = extract_band(band, args, reporter=reporter)
    return band_name, ok


def main() -> int:
    args = parse_args()
    reporter = Reporter.from_env()
    if args.cutout_size is not None and args.vis_pixels is not None:
        print("✗ Pass either --cutout-size or --vis-pixels, not both.")
        return 1
    requested = [name.strip() for name in args.bands.split(",") if name.strip()]
    bands = [Config.get_band(name) for name in requested]
    n_bands = len(bands)
    n_procs = max(1, min(args.max_procs, n_bands))

    print(f"Extracting ePSF for bands: {[b.name for b in bands]}")
    print(f"  num-stars    = {args.num_stars if args.num_stars else 'all'} (cap)")
    print(f"  stars-per-psf= {args.stars_per_psf}  (→ K=round(n_good/N) PSFs/band)")
    print(f"  min-stars/psf= {args.min_stars_per_psf}  (smaller clusters merged)")
    print(f"  stars-csv    = {args.stars_csv}")
    if args.vis_pixels is not None:
        print(f"  vis-pixels   = {args.vis_pixels}  (per-band native size derived)")
    else:
        print(f"  cutout-size  = {args.cutout_size}")
    print(f"  output-size  = {args.output_size}")
    print(f"  psf-dir      = {args.psf_dir}")
    print(f"  parallel     = {n_procs} band(s) at once\n")

    reporter.set_stage(f"extracting {n_bands} ePSF(s) — {n_procs}-way parallel")
    # Per-PSF progress is reported by each band as a parallel "worker"
    # (worker_id = band name): the consumer sums the per-band (current/total)
    # into one cumulative cross-band bar. total=0 → the bar's denominator is
    # the sum of the per-band cluster counts the workers report (we don't know
    # ΣK until each band clusters). Works for both the pool and the serial
    # path (one worker per band either way).
    reporter.set_parallel(0, n_procs, label=f"extracting {n_bands} ePSF(s)")
    succeeded = []
    done = 0
    if n_procs == 1:
        # Single band (or forced serial): no pool overhead.
        for band in bands:
            ok = extract_band(band, args, reporter=reporter)
            succeeded.append((band.name, ok))
            done += 1
            print()
    else:
        with ProcessPoolExecutor(max_workers=n_procs) as pool:
            futures = {
                pool.submit(_extract_band_worker, (b.name, args)): b.name
                for b in bands
            }
            for fut in as_completed(futures):
                band_name = futures[fut]
                try:
                    name, ok = fut.result()
                except Exception as e:
                    reporter.warn(f"{band_name}: worker crashed: "
                                  f"{type(e).__name__}: {e}")
                    name, ok = band_name, False
                succeeded.append((name, ok))
                done += 1
                print(f"[{done}/{n_bands}] {name}: {'✓' if ok else '✗'}", flush=True)

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
