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
import os
import sys
import time

# Make ``euclid_polish`` importable when running this file directly.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tf_keras.optimizers.schedules import PiecewiseConstantDecay

from euclid_polish.config import Config
from euclid_polish.euclid.psf_library import load_all_band_psfs
from euclid_polish.observability.reporter import Reporter
from euclid_polish.sky.cosmos2025 import open_cosmos2025
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

    cat = open_cosmos2025(path=args.catalog)
    _log(f"Catalog: {type(cat).__name__}  ({len(cat)} galaxies usable)")

    cfg = MultiBandGeneratorConfig(image_size=args.image_size,
                                   pixel_scale=Config.DEFAULT_PIXEL_SCALE)
    sim = MultiBandSimulator(cat, cfg)
    os.makedirs(args.records_dir, exist_ok=True)

    # Structured progress for the WebUI (no terminal for tqdm under SLURM).
    # One cumulative bar across train + validate.
    reporter = Reporter.from_env()
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

    psfs = load_all_band_psfs(
        psf_dir=args.psf_dir,
        require_empirical=args.require_empirical_psf,
        target_pixel_scale=Config.DEFAULT_PIXEL_SCALE,
    )
    for name, psf in psfs.items():
        _log(f"  PSF[{name}]: shape={psf.shape}, "
             f"{psf.pixel_scale}\"/pix, fwhm≈{psf.fwhm_arcsec or '?'}")

    fwd = MultiBandForward(psfs_by_band=psfs,
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
