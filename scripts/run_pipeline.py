#!/usr/bin/env python
"""Non-interactive pipeline driver for batch / cluster jobs.

Mirrors the three steps of the interactive CLI but drives them sequentially
without prompts:

    1. Generate clean HR fields with COSMOS galaxies + stars + donut-galaxies
       (saved as ``clean_{train,validate}.tfrecord``).
    2. Convolve HR → LR using the Euclid VIS PSF, sum-rebin to detector grid,
       inject Poisson + read noise (saved as ``dirty_{train,validate}.tfrecord``).
    3. Train the WDSR model (checkpoints + JSONL log written to --checkpoint-dir).

Any step can be skipped via ``--skip-{generate,convolve,train}`` so the script
is safe to re-run after a partial failure.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Make `euclid_polish` importable when running this file directly
# (i.e. without `cd`-ing into the project root or installing the package).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tf_keras.optimizers.schedules import PiecewiseConstantDecay

from euclid_polish.config import Config
from euclid_polish.euclid.types import PSF
from euclid_polish.sky.clean_generator import CleanSkyGenerator, GeneratorConfig
from euclid_polish.sky.psf_convolution import ConvolutionConfig, PSFConvolution
from euclid_polish.sky.tfrecord import tfrecord_path, write_skyimages
from euclid_polish.sky.types import SkyImage
from euclid_polish.training import EuclidDataset, Trainer
from euclid_polish.training.models.wdsr import wdsr


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    """Timestamped log line, flushed (so it shows up in SLURM .out files
    while the job is still running)."""
    print(f"[{_ts()}] {msg}", flush=True)


def _banner(msg: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n[{_ts()}] {msg}\n{bar}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog",        default=Config.DEFAULT_COSMOS_CATALOG_DIR,
                    help="Path to a COSMOS catalog file or directory")
    ap.add_argument("--psf",            default=os.path.join(
                        Config.EUCLID_PSF_DIR, Config.DEFAULT_PSF_FITS_FILENAME),
                    help="PSF FITS path")
    ap.add_argument("--records-dir",    default=Config.RECORDS_DIR)
    ap.add_argument("--checkpoint-dir", default=Config.DEFAULT_CHECKPOINT_DIR)
    ap.add_argument("--ntrain",         type=int, default=6400)
    ap.add_argument("--nvalid",         type=int, default=1600)
    ap.add_argument("--image-size",     type=int, default=512,
                    help="HR field side in pixels (LR = image_size / rebin_factor)")
    ap.add_argument("--steps",          type=int, default=100_000)
    ap.add_argument("--batch-size",     type=int, default=Config.DEFAULT_BATCH_SIZE)
    ap.add_argument("--evaluate-every", type=int, default=Config.DEFAULT_EVALUATE_EVERY)
    ap.add_argument("--num-res-blocks", type=int, default=Config.DEFAULT_NUM_RES_BLOCKS)
    ap.add_argument("--skip-generate",  action="store_true")
    ap.add_argument("--skip-convolve",  action="store_true")
    ap.add_argument("--skip-train",     action="store_true")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Step 1: clean field generation
# ---------------------------------------------------------------------------
def step_generate(args: argparse.Namespace) -> None:
    _banner(f"STEP 1: Generate clean fields  ({args.ntrain} train + {args.nvalid} valid, "
            f"{args.image_size}² @ {Config.DEFAULT_PIXEL_SCALE}\"/pix)")

    cfg = GeneratorConfig(image_size=args.image_size,
                          pixel_scale=Config.DEFAULT_PIXEL_SCALE)
    gen = CleanSkyGenerator(cfg)
    catalog, cat_file, _ = CleanSkyGenerator.resolve_cosmos_catalog(args.catalog)
    _log(f"COSMOS catalog: {cat_file}  ({catalog.nobjects} objects)")

    os.makedirs(args.records_dir, exist_ok=True)

    _log(f"Train generation: starting ({args.ntrain} images)")
    t0 = time.perf_counter()
    train_imgs, _ = gen.generate(catalog, subset="train",
                                 nimages=args.ntrain, nstart=0)
    write_skyimages(train_imgs, "clean_train", args.records_dir)
    _log(f"Train generation: done — {len(train_imgs)} images, "
         f"{time.perf_counter() - t0:.1f} s")

    _log(f"Validate generation: starting ({args.nvalid} images)")
    t0 = time.perf_counter()
    valid_imgs, _ = gen.generate(catalog, subset="validate",
                                 nimages=args.nvalid, nstart=args.ntrain)
    write_skyimages(valid_imgs, "clean_validate", args.records_dir)
    _log(f"Validate generation: done — {len(valid_imgs)} images, "
         f"{time.perf_counter() - t0:.1f} s")


# ---------------------------------------------------------------------------
# Step 2: PSF convolution + noise
# ---------------------------------------------------------------------------
def step_convolve(args: argparse.Namespace) -> None:
    rebin = Config.DEFAULT_REBIN_FACTOR
    lr_size = args.image_size // rebin
    _banner(f"STEP 2: Convolve HR → LR  ({args.image_size}² → {lr_size}²)")

    if not os.path.exists(args.psf):
        raise FileNotFoundError(f"PSF not found: {args.psf}")
    psf = PSF.from_fits(args.psf)
    _log(f"PSF: {psf.shape}, {psf.pixel_scale}\"/pix")

    convolver = PSFConvolution(ConvolutionConfig())

    for subset in ("train", "validate"):
        clean_path = tfrecord_path(args.records_dir, f"clean_{subset}")
        dirty_path = tfrecord_path(args.records_dir, f"dirty_{subset}")
        if not os.path.exists(clean_path):
            _log(f"⚠️  {clean_path} not found, skipping")
            continue

        records = list(tf.data.TFRecordDataset(clean_path))
        rng = np.random.default_rng(0xEC11D + (1 if subset == "validate" else 0))
        writer = tf.io.TFRecordWriter(dirty_path)
        _log(f"Convolution ({subset}): starting on {len(records)} images")
        t0 = time.perf_counter()
        try:
            for raw in tqdm(records, desc=f"  Convolving {subset}", unit="img"):
                hr_img = SkyImage.from_tfrecord(raw)
                lr, _ = convolver.process_hr_to_lr(hr_img, psf, rng=rng)
                writer.write(lr.to_tfrecord())
        finally:
            writer.close()
        _log(f"Convolution ({subset}): done — {time.perf_counter() - t0:.1f} s "
             f"→ {dirty_path}")


# ---------------------------------------------------------------------------
# Step 3: train
# ---------------------------------------------------------------------------
def step_train(args: argparse.Namespace) -> None:
    _banner(f"STEP 3: Train WDSR for {args.steps} steps  (eval every "
            f"{args.evaluate_every}, batch {args.batch_size})")

    scale = Config.DEFAULT_REBIN_FACTOR
    train_loader = EuclidDataset(scale=scale, subset="train",
                                 records_dir=args.records_dir)
    valid_loader = EuclidDataset(scale=scale, subset="validate",
                                 records_dir=args.records_dir)

    train_ds = train_loader.dataset(batch_size=args.batch_size,
                                    random_transform=True)
    valid_ds = valid_loader.dataset(batch_size=1, random_transform=False,
                                    repeat_count=1)

    print(f"  TFRecords: {args.records_dir}")
    print(f"  Checkpoints: {args.checkpoint_dir}")

    model = wdsr(scale=scale,
                 num_res_blocks=args.num_res_blocks,
                 nchan=1)

    learning_rate = PiecewiseConstantDecay(boundaries=[200_000],
                                           values=[1e-3, 5e-4])
    trainer = Trainer(model=model, learning_rate=learning_rate,
                      checkpoint_dir=args.checkpoint_dir)
    trainer.train(train_ds, valid_ds,
                  steps=args.steps,
                  evaluate_every=args.evaluate_every)


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    print(f"Pipeline started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  args = {vars(args)}")

    if not args.skip_generate:
        step_generate(args)
    else:
        print("STEP 1 skipped (--skip-generate)")

    if not args.skip_convolve:
        step_convolve(args)
    else:
        print("STEP 2 skipped (--skip-convolve)")

    if not args.skip_train:
        step_train(args)
    else:
        print("STEP 3 skipped (--skip-train)")

    dt = time.perf_counter() - t0
    print(f"\nDone in {dt/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
