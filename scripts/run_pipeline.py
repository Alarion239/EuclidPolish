#!/usr/bin/env python
"""Non-interactive pipeline driver for the multi-band EuclidPolish workflow.

Mirrors the three CLI menu steps but drives them sequentially without prompts:

    1. Generate clean HR fields with COSMOS2025 galaxies + stars + lenses
       (saved as ``clean_{train,validate}.tfrecord`` in v2 schema, 4 channels).
    2. Run the per-band forward model HR → LR (PSF convolution + noise + NISP
       upsample to VIS LR grid). Saved as ``dirty_{train,validate}.tfrecord``,
       4-channel LR at 0.10″/pix.
    3. Train WDSR (4-channel input, 1-channel VIS HR target).

Any step can be skipped via ``--skip-{generate,convolve,train}``.

All file paths and constants come from :mod:`euclid_polish.config`.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Cap per-process BLAS threads BEFORE numpy import so the parallel
# generate+forward workers map 1:1 onto their CPUs instead of each
# spawning a thread per core. Affects only numpy/scipy (generation +
# forward model); TF training uses its own thread pools, so this does NOT
# slow the train step. ``setdefault`` honours an explicit override.
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

# Make ``euclid_polish`` importable when running this file directly.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tf_keras.optimizers.schedules import PiecewiseConstantDecay

from euclid_polish.config import Config
from euclid_polish.euclid.psf_library import load_all_band_psf_sets
from euclid_polish.observability.reporter import Reporter
from euclid_polish.observability.resource_sampler import ResourceSampler
from euclid_polish.sky.cosmos2025 import ensure_prefiltered_catalog, open_cosmos2025
from euclid_polish.sky.multiband_forward import (
    MultiBandForward, MultiBandForwardConfig,
)
from euclid_polish.sky.multiband_generator import (
    MultiBandGeneratorConfig, MultiBandSimulator,
)
from euclid_polish.sky.tfrecord import (
    open_multiband_writer, tfrecord_path, write_multiband_skyimages,
)
from euclid_polish.sky.types import MultiBandSkyImage
from euclid_polish.training.data_multiband import MultiBandEuclidDataset
from euclid_polish.training import Trainer
from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.training.stage_timer import StageTimer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _banner(msg: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n[{_ts()}] {msg}\n{bar}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog",        default=Config.COSMOS2025_CATALOG_PATH,
                    help="Path to the COSMOS2025 master FITS file (required)")
    ap.add_argument("--psf-dir",        default=Config.EUCLID_PSF_DIR,
                    help="Directory containing per-band ePSF FITS files; "
                         "missing bands fall back to Gaussian.")
    ap.add_argument("--records-dir",    default=Config.RECORDS_DIR_V2)
    ap.add_argument("--checkpoint-dir", default=Config.DEFAULT_CHECKPOINT_DIR)
    ap.add_argument("--ntrain",         type=int, default=6400)
    ap.add_argument("--nvalid",         type=int, default=100)
    ap.add_argument("--image-size",     type=int, default=252,
                    help="HR field side (HR pixels). Must be divisible by 6 "
                         "(LCM of VIS rebin=2 and NISP rebin=6).")
    ap.add_argument("--steps",          type=int, default=Config.DEFAULT_TRAIN_STEPS)
    ap.add_argument("--batch-size",     type=int, default=Config.DEFAULT_BATCH_SIZE)
    ap.add_argument("--evaluate-every", type=int, default=Config.DEFAULT_EVALUATE_EVERY)
    ap.add_argument("--num-res-blocks", type=int, default=Config.DEFAULT_NUM_RES_BLOCKS)
    ap.add_argument("--require-empirical-psf", action="store_true",
                    help="Fail if any band lacks a real ePSF (no Gaussian fallback).")
    ap.add_argument("--gen-workers", type=int, default=1,
                    help="Parallelise synthetic generation across this many "
                         "processes. >1 runs a COMBINED generate+forward pass "
                         "(each worker renders clean → hr+dirty for a "
                         "contiguous index range into its own TFRecord "
                         "shards, then the shards are concatenated in order). "
                         "Requires both generate and convolve (i.e. neither "
                         "--skip-generate nor --skip-convolve); falls back to "
                         "the serial two-step path otherwise.")
    ap.add_argument("--tng-fraction", type=float, default=0.0,
                    help="Fraction of synthetic galaxies drawn as real TNG50 "
                         "SKIRT stamps (random galaxy + orientation, downsampled "
                         "×1/×2/×3/×4) instead of analytic Sersic profiles. "
                         "0 = all Sersic (unchanged). Needs TNG galaxies "
                         "downloaded under $DATA_DIR/tng_skirt/.")
    ap.add_argument("--star-density-arcmin2", type=float,
                    default=Config.DEFAULT_STAR_DENSITY_ARCMIN2,
                    help="Stellar surface density (stars/arcmin²).")
    ap.add_argument("--star-mag-slope", type=float, default=Config.STAR_MAG_SLOPE,
                    help="Star-count slope α in dN/dm ∝ 10^(α·m) "
                         "(high-Galactic-latitude value ~0.14–0.35).")
    ap.add_argument("--star-mag-bright", type=float, default=Config.STAR_MAG_BRIGHT,
                    help="Brightest synthetic star (VIS mag).")
    ap.add_argument("--star-mag-faint", type=float, default=Config.STAR_MAG_FAINT,
                    help="Faintest synthetic star (VIS mag).")
    ap.add_argument("--lens-density-arcmin2", type=float,
                    default=Config.LENS_DENSITY_ARCMIN2,
                    help="Strong-lens surface density (lenses/arcmin²).")
    ap.add_argument("--lens-sigma-v-min-kms", type=float,
                    default=Config.LENS_SIGMA_V_MIN_KMS,
                    help="Min lens velocity dispersion (km/s); σ_v² sets θ_E.")
    ap.add_argument("--lens-sigma-v-max-kms", type=float,
                    default=Config.LENS_SIGMA_V_MAX_KMS,
                    help="Max lens velocity dispersion (km/s); σ_v² sets θ_E.")
    ap.add_argument("--skip-generate",  action="store_true")
    ap.add_argument("--skip-convolve",  action="store_true")
    ap.add_argument("--skip-train",     action="store_true")
    ap.add_argument("--stages-csv", default="",
                    help="Path to per-stage timings CSV. "
                         "Default: <records-dir>/stages_${SLURM_JOB_ID}.csv "
                         "(or stages_local.csv outside SLURM).")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Step 1: clean multi-band scene generation
# ---------------------------------------------------------------------------

def step_generate(args: argparse.Namespace) -> None:
    _banner(f"STEP 1: Generate clean 4-band HR fields  "
            f"({args.ntrain} train + {args.nvalid} valid, "
            f"{args.image_size}² @ {Config.DEFAULT_PIXEL_SCALE}\"/pix)")

    # Pre-filter the 10 GB master FITS to a small cached .npz once, then load
    # that — instant on repeat runs (and shared by the parallel path).
    cat = open_cosmos2025(path=ensure_prefiltered_catalog(args.catalog))
    _log(f"Catalog: {type(cat).__name__}  ({len(cat)} galaxies usable)")

    cfg = MultiBandGeneratorConfig(image_size=args.image_size,
                                   pixel_scale=Config.DEFAULT_PIXEL_SCALE,
                                   tng_fraction=args.tng_fraction,
                                   star_density_arcmin2=args.star_density_arcmin2,
                                   star_mag_slope=args.star_mag_slope,
                                   star_mag_bright=args.star_mag_bright,
                                   star_mag_faint=args.star_mag_faint,
                                   lens_density_arcmin2=args.lens_density_arcmin2,
                                   lens_sigma_v_min_kms=args.lens_sigma_v_min_kms,
                                   lens_sigma_v_max_kms=args.lens_sigma_v_max_kms)
    sim = MultiBandSimulator(cat, cfg)
    os.makedirs(args.records_dir, exist_ok=True)

    # Structured progress for the WebUI (no terminal for tqdm under SLURM).
    # One cumulative bar across train + validate.
    reporter = Reporter.from_env()
    # Sample CPU (and GPU, if the train step runs on one) through the whole
    # pipeline so the WebUI shows live utilisation — confirms the parallel
    # generate workers are actually busy. Daemon thread; dies at exit.
    ResourceSampler(reporter).start()
    reporter.set_stage("generating clean HR fields")
    grand_total = int(args.ntrain) + int(args.nvalid)
    done = 0

    for subset, n in (("train", args.ntrain), ("validate", args.nvalid)):
        # Entropy-seeded master RNG so repeat runs see fresh randomness.
        # The seed is logged so a curious-looking run can be replayed
        # later by hard-coding the printed value here.
        master_seed = int.from_bytes(os.urandom(8), "little")
        rng = np.random.default_rng(master_seed)
        _log(f"  {subset}: generating {n} images  (master_seed={master_seed})")
        t0 = time.perf_counter()
        # Stream each image to disk as it's generated — accumulating
        # 6400 510² × 4-channel float32 fields would cost ~26 GB of RSS
        # and OOM-kill on the FASRC default --mem=32G.
        with open_multiband_writer(f"clean_{subset}",
                                   records_dir=args.records_dir) as w:
            for i in tqdm(range(n), desc=f"  {subset}", unit="img"):
                sky, _ = sim.simulate_field(rng)
                sky.index = i
                sky.subset = subset
                w.write(sky, index=i)
                done += 1
                reporter.set_step(done, grand_total, f"generate {subset} {i + 1}/{n}")
            path, count = w.path, w.count
        _log(f"  {subset}: done — {count} → {path}  "
             f"({time.perf_counter() - t0:.1f} s)")


# ---------------------------------------------------------------------------
# Step 2: per-band PSF convolution + noise + NISP upsample
# ---------------------------------------------------------------------------

def step_convolve(args: argparse.Namespace) -> None:
    _banner("STEP 2: HR → LR  (per-band PSF + noise + NISP→VIS-LR resample)")

    psf_sets = load_all_band_psf_sets(
        psf_dir=args.psf_dir,
        require_empirical=args.require_empirical_psf,
        target_pixel_scale=Config.DEFAULT_PIXEL_SCALE,
    )
    for name, pset in psf_sets.items():
        _log(f"  PSF[{name}]: {pset.n} kernel(s), shape={pset.shape}, "
             f"{pset.pixel_scale}\"/pix")

    fwd = MultiBandForward(psf_sets_by_band=psf_sets,
                           config=MultiBandForwardConfig(add_noise=True))

    # Structured progress for the WebUI — one cumulative bar across both
    # subsets present. Pre-count the clean records (re-iterating is ~ms).
    reporter = Reporter.from_env()
    reporter.set_stage("forward-modelling HR → LR")
    counts = {}
    for subset in ("train", "validate"):
        p = tfrecord_path(args.records_dir, f"clean_{subset}")
        counts[subset] = (sum(1 for _ in tf.data.TFRecordDataset(p))
                          if os.path.exists(p) else 0)
    grand_total = sum(counts.values())
    done = 0

    for subset in ("train", "validate"):
        clean_path = tfrecord_path(args.records_dir, f"clean_{subset}")
        if not os.path.exists(clean_path):
            _log(f"⚠️  {clean_path} not found, skipping {subset}")
            continue

        # Stream records from the clean TFRecord (do NOT materialise the
        # whole list — same OOM hazard as step_generate at 6400 images).
        clean_ds = tf.data.TFRecordDataset(clean_path)
        n_total = counts[subset]
        # Entropy-seeded forward-model RNG — different noise / artifact
        # realisation every run. Master seed is logged for replay.
        master_seed = int.from_bytes(os.urandom(8), "little")
        rng = np.random.default_rng(master_seed)

        _log(f"  {subset}: forward-modelling {n_total} fields  "
             f"(master_seed={master_seed})")
        t0 = time.perf_counter()
        # Two streaming writers (one for hr_, one for dirty_); clean_ is
        # NOT rewritten — the 4-band record is kept for inspection.
        with open_multiband_writer(f"hr_{subset}",
                                   records_dir=args.records_dir) as hr_w, \
             open_multiband_writer(f"dirty_{subset}",
                                   records_dir=args.records_dir) as lr_w:
            for i, raw in enumerate(tqdm(clean_ds, desc=f"  {subset}",
                                         unit="img", total=n_total)):
                hr_4ch = MultiBandSkyImage.from_tfrecord(raw)
                lr, hr = fwd.process(hr_4ch, rng=rng)
                lr.index = i
                hr.index = i
                lr.subset = subset
                hr.subset = subset
                hr_w.write(hr, index=i)
                lr_w.write(lr, index=i)
                done += 1
                reporter.set_step(done, grand_total, f"forward {subset} {i + 1}/{n_total}")
        _log(f"  {subset}: done in {time.perf_counter() - t0:.1f} s "
             f"→ kept clean + wrote hr + dirty {subset}")


# ---------------------------------------------------------------------------
# Parallel combined generate + forward (one process per index range)
# ---------------------------------------------------------------------------

def _concat_tfrecords(part_paths: List[str], out_path: str) -> None:
    """Byte-concatenate TFRecord shard files into one TFRecord.

    A TFRecord file is a bare back-to-back sequence of self-framed records
    (no header/footer), so concatenating shards in order yields a valid
    TFRecord whose records appear in shard order. Missing/empty parts are
    skipped."""
    with open(out_path, "wb") as out:
        for p in part_paths:
            if os.path.exists(p) and os.path.getsize(p) > 0:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out, length=4 * 1024 * 1024)


def _shard_bounds(n: int, n_shards: int) -> List[Tuple[int, int]]:
    """Contiguous ``[start, end)`` ranges partitioning ``[0, n)``."""
    return [(round(k * n / n_shards), round((k + 1) * n / n_shards))
            for k in range(n_shards)]


def _generate_convolve_range(sim, fwd, records_dir: str, subset: str,
                             start: int, count: int, shard_id: int,
                             seed) -> Tuple[str, int, int]:
    """Generate clean → forward to hr+dirty for ``[start, start+count)`` and
    write the triple to per-shard TFRecords.

    Pure (no globals / pool) so it's unit-testable with an injected
    sim/fwd. Records are written ``clean[i] → hr[i] → dirty[i]`` in the same
    order, so concatenating shards in id order keeps clean/hr/dirty
    position-aligned (the dataset pairs by position, not by stored index)."""
    rng = np.random.default_rng(seed)
    tag = f"{subset}.part{shard_id:04d}"
    # Per-worker progress → the parent's events file (shared, append-atomic).
    # Keyed by shard_id so the consumer can sum a cumulative count and tell
    # how many processes are busy. Rate-limited so 16 workers don't flood
    # the file; register at 0 up front so a just-started shard shows.
    reporter = Reporter.from_env()
    reporter.set_worker_step(shard_id, 0, count, subset)
    last_emit = time.perf_counter()
    with open_multiband_writer(f"clean_{tag}", records_dir=records_dir) as cw, \
         open_multiband_writer(f"hr_{tag}",    records_dir=records_dir) as hw, \
         open_multiband_writer(f"dirty_{tag}", records_dir=records_dir) as dw:
        for local, i in enumerate(range(start, start + count), start=1):
            sky, _ = sim.simulate_field(rng)
            sky.index = i
            sky.subset = subset
            lr, hr = fwd.process(sky, rng=rng)
            hr.index = i
            hr.subset = subset
            lr.index = i
            lr.subset = subset
            cw.write(sky, index=i)
            hw.write(hr, index=i)
            dw.write(lr, index=i)
            now = time.perf_counter()
            if local == count or (now - last_emit) >= 2.0:
                reporter.set_worker_step(shard_id, local, count, subset)
                last_emit = now
    return subset, shard_id, count


# Worker-process globals, built once per worker by ``_gen_init_worker``.
_W_SIM = None
_W_FWD = None
_W_RECORDS_DIR = ""


def _gen_init_worker(catalog_path, image_size, psf_dir,
                     require_empirical_psf, records_dir,
                     tng_fraction=0.0,
                     star_density_arcmin2=Config.DEFAULT_STAR_DENSITY_ARCMIN2,
                     star_mag_slope=Config.STAR_MAG_SLOPE,
                     star_mag_bright=Config.STAR_MAG_BRIGHT,
                     star_mag_faint=Config.STAR_MAG_FAINT,
                     lens_density_arcmin2=Config.LENS_DENSITY_ARCMIN2,
                     lens_sigma_v_min_kms=Config.LENS_SIGMA_V_MIN_KMS,
                     lens_sigma_v_max_kms=Config.LENS_SIGMA_V_MAX_KMS) -> None:
    """ProcessPool initializer: build the (small, filtered) catalog +
    simulator + forward model once per worker. The COSMOS2025 FITS is
    memmapped and only the filtered columns are held, so each worker's copy
    is a few MB — no 10 GB-per-worker blow-up."""
    global _W_SIM, _W_FWD, _W_RECORDS_DIR
    cat = open_cosmos2025(path=catalog_path)
    _W_SIM = MultiBandSimulator(
        cat, MultiBandGeneratorConfig(image_size=image_size,
                                      pixel_scale=Config.DEFAULT_PIXEL_SCALE,
                                      tng_fraction=tng_fraction,
                                      star_density_arcmin2=star_density_arcmin2,
                                      star_mag_slope=star_mag_slope,
                                      star_mag_bright=star_mag_bright,
                                      star_mag_faint=star_mag_faint,
                                      lens_density_arcmin2=lens_density_arcmin2,
                                      lens_sigma_v_min_kms=lens_sigma_v_min_kms,
                                      lens_sigma_v_max_kms=lens_sigma_v_max_kms),
    )
    psf_sets = load_all_band_psf_sets(
        psf_dir=psf_dir, require_empirical=require_empirical_psf,
        target_pixel_scale=Config.DEFAULT_PIXEL_SCALE,
    )
    _W_FWD = MultiBandForward(psf_sets_by_band=psf_sets,
                              config=MultiBandForwardConfig(add_noise=True))
    _W_RECORDS_DIR = records_dir


def _gen_convolve_shard(task) -> Tuple[str, int, int]:
    """Top-level pool entry point → ``_generate_convolve_range`` with the
    worker-global sim/fwd."""
    subset, start, count, shard_id, seed = task
    return _generate_convolve_range(
        _W_SIM, _W_FWD, _W_RECORDS_DIR, subset, start, count, shard_id, seed,
    )


def step_generate_and_convolve_parallel(args: argparse.Namespace) -> None:
    _banner(f"STEP 1+2 (parallel ×{args.gen_workers}): generate clean HR + "
            f"forward-model to dirty Euclid LR")
    reporter = Reporter.from_env()
    os.makedirs(args.records_dir, exist_ok=True)
    workers = max(1, int(args.gen_workers))

    # Pre-filter the 10 GB master FITS to a small cached .npz ONCE (in the
    # parent), so each per-subset, per-worker pool initializer reloads a few-MB
    # file in milliseconds instead of re-parsing 784k rows every time.
    catalog_path = ensure_prefiltered_catalog(args.catalog)

    for subset, n in (("train", args.ntrain), ("validate", args.nvalid)):
        if n <= 0:
            continue
        # More shards than workers → finer progress + load balancing.
        # ~256 images/shard, but at least one shard per worker and never
        # more shards than images.
        n_shards = min(n, max(workers, math.ceil(n / 256)))
        bounds = _shard_bounds(n, n_shards)
        master_seed = int.from_bytes(os.urandom(8), "little")
        tasks = []
        for sid, (start, end) in enumerate(bounds):
            if end > start:
                # SeedSequence material → reproducible, independent per shard.
                seed = [master_seed, (1 if subset == "train" else 2), sid]
                tasks.append((subset, start, end - start, sid, seed))

        reporter.set_stage(f"generate+forward {subset} (×{workers})")
        # Announce the parallel phase: total items + worker count. The
        # workers report their own progress; the consumer sums a cumulative
        # bar and counts active processes.
        reporter.set_parallel(n, workers, label=subset)
        _log(f"  {subset}: {n} pairs across {len(tasks)} shards, "
             f"{workers} workers (master_seed={master_seed})")
        t0 = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_gen_init_worker,
            initargs=(catalog_path, args.image_size, args.psf_dir,
                      args.require_empirical_psf, args.records_dir,
                      args.tng_fraction, args.star_density_arcmin2,
                      args.star_mag_slope, args.star_mag_bright,
                      args.star_mag_faint, args.lens_density_arcmin2,
                      args.lens_sigma_v_min_kms, args.lens_sigma_v_max_kms),
        ) as pool:
            futs = [pool.submit(_gen_convolve_shard, t) for t in tasks]
            for fut in as_completed(futs):
                fut.result()   # surface worker exceptions; progress is
                               # driven by the workers' set_worker_step.

        # Merge shards IN ID ORDER so clean/hr/dirty stay position-aligned.
        reporter.set_stage(f"merging {subset} shards")
        for kind in ("clean", "hr", "dirty"):
            parts = [tfrecord_path(args.records_dir,
                                   f"{kind}_{subset}.part{sid:04d}")
                     for sid, (s, e) in enumerate(bounds) if e > s]
            _concat_tfrecords(parts, tfrecord_path(args.records_dir,
                                                   f"{kind}_{subset}"))
            for p in parts:
                try:
                    os.remove(p)
                except OSError:
                    pass
        _log(f"  {subset}: done in {time.perf_counter() - t0:.1f} s "
             f"→ clean + hr + dirty {subset}")


# ---------------------------------------------------------------------------
# Step 3: train WDSR (4-channel in, 1-channel out)
# ---------------------------------------------------------------------------

def step_train(args: argparse.Namespace) -> None:
    _banner(f"STEP 3: Train WDSR  ({args.steps} steps, batch {args.batch_size}, "
            f"eval every {args.evaluate_every})")

    scale = Config.DEFAULT_REBIN_FACTOR

    train_loader = MultiBandEuclidDataset(
        scale=scale, subset="train", records_dir=args.records_dir,
    )
    valid_loader = MultiBandEuclidDataset(
        scale=scale, subset="validate", records_dir=args.records_dir,
    )
    train_ds = train_loader.dataset(batch_size=args.batch_size,
                                    random_transform=True)
    valid_ds = valid_loader.dataset(batch_size=1,
                                    random_transform=False,
                                    repeat_count=1)

    print(f"  TFRecords:   {args.records_dir}")
    print(f"  Checkpoints: {args.checkpoint_dir}")

    model = wdsr(
        scale=scale,
        num_res_blocks=args.num_res_blocks,
        nchan_in=Config.NUM_LR_CHANNELS,
        nchan_out=Config.NUM_HR_CHANNELS,
    )

    learning_rate = PiecewiseConstantDecay(boundaries=[200_000],
                                           values=[1e-3, 5e-4])
    trainer = Trainer(model=model, learning_rate=learning_rate,
                      checkpoint_dir=args.checkpoint_dir)
    trainer.train(train_ds, valid_ds,
                  steps=args.steps,
                  evaluate_every=args.evaluate_every)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    # ``t_script_start`` brackets the "init" stage — everything that
    # happened between Python booting up and the first real stage
    # starting (module imports, dataset loaders ready, etc).
    t_script_start = time.time()
    t0_perf = time.perf_counter()
    args = parse_args()
    print(f"Pipeline started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  args = {vars(args)}")

    # Per-stage timings CSV — default sits next to the SLURM .out file
    # via ``$EUCLID_POLISH_DATA_DIR/images/records_v2/stages_<jobid>.csv``
    # so the FASRC dashboard can fetch it back without knowing the
    # exact records-dir layout.
    slurm_jobid = os.environ.get("SLURM_JOB_ID", "local")
    stages_path = args.stages_csv or os.path.join(
        args.records_dir, f"stages_{slurm_jobid}.csv",
    )
    timer = StageTimer(
        csv_path=stages_path,
        jobid=slurm_jobid,
        params=dict(
            n_train=args.ntrain, n_valid=args.nvalid,
            image_size=args.image_size, batch_size=args.batch_size,
            steps=args.steps,
        ),
    )
    print(f"  stage timings → {stages_path}")

    # ``init`` is everything from ``t_script_start`` up to the first
    # stage. Mark it now so it's persisted even if a later stage fails.
    timer.mark("init", params_dependent=False,
               started_at=t_script_start, ended_at=time.time())

    # Parallel combined path: one process per index range does generate +
    # forward in a single pass, then shards are concatenated. Only when both
    # stages are wanted (the standalone re-convolve case stays serial).
    parallel_gen = (int(getattr(args, "gen_workers", 1) or 1) > 1
                    and not args.skip_generate and not args.skip_convolve)
    if parallel_gen:
        with timer.stage("generate", params_dependent=True):
            step_generate_and_convolve_parallel(args)
    else:
        if not args.skip_generate:
            with timer.stage("generate", params_dependent=True):
                step_generate(args)
        else:
            print("STEP 1 skipped (--skip-generate)")

        if not args.skip_convolve:
            with timer.stage("convolve", params_dependent=True):
                step_convolve(args)
        else:
            print("STEP 2 skipped (--skip-convolve)")

    if not args.skip_train:
        with timer.stage("train", params_dependent=True):
            step_train(args)
    else:
        print("STEP 3 skipped (--skip-train)")

    dt = time.perf_counter() - t0_perf
    print(f"\nDone in {dt/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
