#!/usr/bin/env python
"""Extract empirical ePSFs for every Euclid band from local star cutouts.

For each of ``Config.BANDS`` this script:

  1. Looks for cutouts in the band-specific directory:
       VIS  → ``Config.DEFAULT_OUTPUT_DIR/cutouts``
       NISP → ``Config.NISP_DEFAULT_OUTPUT_DIR_BY_BAND[band.name]/cutouts``
  2. Loads + accepts the good cutouts (saturation/edge rejection) for every
     band, then **clusters ONCE** the stars that are valid in ALL four bands,
     by catalog sky position (K-Means++) into groups of ``--stars-per-psf``.
     Each band builds **one ePSF per cluster** from that band's cutouts of the
     cluster's stars — the Euclid PSF varies across the focal plane, so a band
     gets K≈n_common/N PSFs instead of one average kernel. The SHARED
     clustering means cluster ``ci`` is the same stars / field region in every
     band (consistent numbering across bands).
  3. Saves them as a multi-extension FITS to
     ``data/euclid_psf/<band.psf_fits_filename>`` (e.g. ``euclid_psf_VIS.fits``):
     HDU[0] is the mean PSF (so single-PSF readers keep working), HDU[1..K] are
     the cluster PSFs (same numbering in every band). Generation draws one
     cluster + roll per scene and applies it to all four bands.
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
from photutils.psf import EPSFStar
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
                    help="Parallel ePSF (cluster) builds — set to the number "
                         "of allocated CPUs. Bands are processed one at a time "
                         "(VIS, then Y_E, …) and that band's cluster PSFs are "
                         "built across all these workers, so a 30-cluster band "
                         "uses every CPU instead of one. The FASRC step passes "
                         "the CPU count you choose in the web form.")
    return ap.parse_args()


def load_accepted_band(
    band: BandConfig, args: argparse.Namespace, reporter: Reporter,
) -> Optional[Dict]:
    """Load one band's cutouts and reject saturated/edge stars.

    Returns ``{cfg, scale, filename, accepted}`` where ``accepted`` is
    ``{star_id: normalised-cutout ndarray}`` for every star that passes the
    checks, or ``None`` to skip the band (no cutouts / no usable stars). No
    clustering here — the clustering is done ONCE on the stars common to all
    bands, so cluster numbering is shared (see :func:`main`)."""
    cutout_dir = _cutout_dir_for_band(band)

    if args.vis_pixels is not None:
        arcsec = args.vis_pixels * Config.BAND_VIS.pixel_scale_lr_arcsec
        cutout_size = band.cutout_size_for_arcsec(arcsec)
    elif args.cutout_size is not None:
        cutout_size = args.cutout_size
    else:
        cutout_size = Config.DEFAULT_CUTOUT_SIZE

    print(f"=== {band.name} ===")
    print(f"  cutouts:     {cutout_dir}")
    print(f"  cutout-size: {cutout_size} px (native)")
    if not os.path.isdir(cutout_dir):
        reporter.warn(f"{band.name}: cutout directory not found — skipping")
        print(f"  ⚠️  cutout directory not found — skipping. "
              f"Run the cutout downloader for band={band.name} first.")
        return None

    # ``psf_size`` is the *native* centred crop EPSFBuilder receives from each
    # star (the natural EPSF grid is ``psf_size × ovs + 1``).
    psf_size = cutout_size - 1 if cutout_size % 2 == 0 else cutout_size - 2
    # tqdm only when interactive; under SLURM (events path set) the Reporter
    # drives the UI bar, so per-iteration tqdm would just flood the .err log.
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
        return None

    selected = extractor.select_files(all_files, num_stars=args.num_stars)
    print(f"  considering {len(selected)} of {len(all_files)} available stars")
    accepted = extractor.extract_accepted_stars(selected)
    if not accepted:
        reporter.warn(f"{band.name}: no usable (non-saturated) stars — skipping")
        print("  ⚠️  no usable stars after saturation/edge rejection — skipping.")
        return None
    print(f"  accepted {len(accepted)} good stars")
    return {
        "cfg":      cfg,
        "scale":    band.epsf_pixel_scale_arcsec,
        "filename": band.psf_fits_filename,
        # Normalised cutouts (ndarrays) — picklable; the worker rebuilds the
        # EPSFStar objects, so nothing photutils-specific crosses processes.
        "accepted": {sid: np.asarray(star.data, dtype=np.float32)
                     for sid, star in accepted},
    }


def _build_cluster_psf(payload):
    """Worker: build ONE cluster's ePSF from its star stamps.

    Top-level + picklable. ``payload`` is ``(cfg, epsf_pixel_scale, stamps)``
    where ``stamps`` is a list of normalised cutout ndarrays; the EPSFStar
    objects are reconstructed here so only plain arrays cross the process
    boundary. Returns a :class:`PSF` (picklable). EPSFBuilder runs
    single-threaded (BLAS capped at the module top), so N workers use N cores.
    """
    cfg, scale, stamps = payload
    extractor = PSFExtractor(cfg)
    stars = [EPSFStar(data=np.asarray(d, dtype=np.float32),
                      cutout_center=(d.shape[1] // 2, d.shape[0] // 2))
             for d in stamps]
    epsf, _ = extractor.build_epsf_from_stars(stars)
    return extractor.psf_from_epsf(epsf, scale)


def main() -> int:
    args = parse_args()
    reporter = Reporter.from_env()
    if args.cutout_size is not None and args.vis_pixels is not None:
        print("✗ Pass either --cutout-size or --vis-pixels, not both.")
        return 1
    requested = [name.strip() for name in args.bands.split(",") if name.strip()]
    bands = [Config.get_band(name) for name in requested]
    n_workers = max(1, int(args.max_procs))

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
    print(f"  workers      = {n_workers} parallel ePSF builds\n")

    # Phase 1 — load + reject each band (sequential, so "VIS first, then …").
    reporter.set_stage("loading + rejecting stars")
    band_data: Dict[str, Dict] = {}
    for bi, band in enumerate(bands, start=1):
        d = load_accepted_band(band, args, reporter)
        if d is not None:
            band_data[band.name] = d
        reporter.set_step(bi, len(bands), f"loaded {band.name}")
        print()
    if not band_data:
        print("No bands had usable cutouts; nothing to build.")
        return 0

    # Cluster ONCE on the stars accepted in ALL present bands, so every band's
    # ePSFs share the same clustering + numbering (cluster ci = the same stars,
    # same field region, in every band — required for the cross-band-consistent
    # generation sampler).
    present = [b.name for b in bands if b.name in band_data]      # band order
    common = set.intersection(*(set(band_data[b]["accepted"]) for b in present))
    print(f"{len(common)} star(s) valid in all {len(present)} band(s) "
          f"{present}")
    positions = _load_star_positions(args.stars_csv)
    common_pos = sorted(i for i in common if i in positions) if positions else []
    if common_pos:
        idx_clusters = cluster_star_indices(
            common_pos, positions, args.stars_per_psf,
            min_stars=args.min_stars_per_psf)
        clusters = [[common_pos[i] for i in c] for c in idx_clusters if c]
    elif common:
        print("  ⚠️  no catalog positions for the common stars — one shared "
              "ePSF per band from all common stars.")
        clusters = [sorted(common)]
    else:
        print("No stars are valid in every band; nothing to build.")
        return 0

    n_clusters = len(clusters)
    star_counts = [len(c) for c in clusters]
    centroids: List[Tuple[float, float]] = []
    for c in clusters:
        pts = [positions[i] for i in c if i in positions]
        centroids.append((float(np.mean([p[0] for p in pts])),
                          float(np.mean([p[1] for p in pts]))) if pts
                         else (float("nan"), float("nan")))
    have_centroids = bool(positions) and all(
        np.isfinite(a) and np.isfinite(b) for a, b in centroids)
    print(f"{n_clusters} shared cluster(s) (target {args.stars_per_psf}/PSF, "
          f"min {args.min_stars_per_psf}): sizes {star_counts}\n")

    # Drop the now-unused (non-common) stamps to save memory.
    used = set().union(*clusters)
    for b in present:
        band_data[b]["accepted"] = {i: band_data[b]["accepted"][i]
                                    for i in used if i in band_data[b]["accepted"]}

    # Phase 2 — build EVERY (band, cluster) ePSF in ONE pool of ``n_workers``
    # (all allocated CPUs in parallel), submitted in band order. One monotonic
    # bar over all PSFs.
    tasks = [(bn, ci, (band_data[bn]["cfg"], band_data[bn]["scale"],
                       [band_data[bn]["accepted"][i] for i in clusters[ci]]))
             for bn in present for ci in range(n_clusters)]
    total = len(tasks)
    reporter.set_stage(f"building {total} ePSF(s) on {n_workers} worker(s)")
    results = {bn: {} for bn in present}         # band -> {cluster_idx: PSF}
    done = 0
    reporter.set_step(0, total, "building ePSFs")

    def _record(band_name: str, ci: int, psf) -> None:
        nonlocal done
        if psf is not None:
            results[band_name][ci] = psf
        done += 1
        reporter.set_step(done, total, f"{band_name}: PSF {ci}")
        print(f"[{done}/{total}] {band_name} PSF {ci}", flush=True)

    if n_workers <= 1:
        for band_name, ci, payload in tasks:
            try:
                psf = _build_cluster_psf(payload)
            except Exception as e:
                reporter.warn(f"{band_name} cluster {ci} failed: "
                              f"{type(e).__name__}: {e}")
                psf = None
            _record(band_name, ci, psf)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futmap = {pool.submit(_build_cluster_psf, payload): (bn, ci)
                      for bn, ci, payload in tasks}
            for fut in as_completed(futmap):
                band_name, ci = futmap[fut]
                try:
                    psf = fut.result()
                except Exception as e:
                    reporter.warn(f"{band_name} cluster {ci} failed: "
                                  f"{type(e).__name__}: {e}")
                    psf = None
                _record(band_name, ci, psf)

    # Phase 3 — a cluster survives only if EVERY present band built it, so the
    # saved PSFSets keep identical numbering (cluster ci ↔ same stars in all
    # bands). Save one PSFSet per band over the surviving clusters.
    print("=" * 50)
    survive = [ci for ci in range(n_clusters)
               if all(ci in results[bn] for bn in present)]
    if len(survive) < n_clusters:
        print(f"  ⚠️  dropped {n_clusters - len(survive)} cluster(s) that failed "
              f"in ≥1 band (kept numbering consistent across bands)")
    succeeded = []
    for band in bands:
        if band.name not in present:
            succeeded.append((band.name, False))
            continue
        psfs = [results[band.name][ci] for ci in survive]
        if not psfs:
            reporter.error(f"{band.name}: no clusters survived in every band")
            print(f"  ✗ {band.name}: no clusters survived in every band")
            succeeded.append((band.name, False))
            continue
        psf_set = PSFSet.from_psfs(
            psfs,
            centroids=[centroids[ci] for ci in survive] if have_centroids else None,
            n_stars=[star_counts[ci] for ci in survive])
        os.makedirs(args.psf_dir, exist_ok=True)
        saved = psf_set.save(args.psf_dir, filename=band_data[band.name]["filename"])
        print(f"  ✓ {band.name}: {psf_set.n} PSF(s), shape={psf_set.shape} "
              f"→ {saved}")
        succeeded.append((band.name, True))

    print("Summary:")
    for name, ok in succeeded:
        print(f"  {'✓' if ok else '✗'} {name}")
    n_ok = sum(1 for _, ok in succeeded if ok)
    print(f"\n{n_ok}/{len(succeeded)} bands extracted; missing bands will "
          "use Gaussian fallback in the forward model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
