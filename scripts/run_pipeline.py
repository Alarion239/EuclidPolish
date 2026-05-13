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
from euclid_polish.sky.cosmos2025 import open_cosmos2025
from euclid_polish.sky.multiband_forward import (
    MultiBandForward, MultiBandForwardConfig,
)
from euclid_polish.sky.multiband_generator import (
    MultiBandGeneratorConfig, MultiBandSimulator,
)
from euclid_polish.sky.tfrecord import (
    tfrecord_path, write_multiband_skyimages,
)
from euclid_polish.sky.types import MultiBandSkyImage
from euclid_polish.training.data_multiband import MultiBandEuclidDataset
from euclid_polish.training import Trainer
from euclid_polish.training.models.wdsr import wdsr


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

    for subset, n, seed in (("train", args.ntrain, 0),
                            ("validate", args.nvalid, args.ntrain)):
        _log(f"  {subset}: generating {n} images")
        t0 = time.perf_counter()
        imgs = []
        for i in tqdm(range(n), desc=f"  {subset}", unit="img"):
            rng = np.random.default_rng(seed + i)
            sky, _ = sim.simulate_field(rng)
            sky.index = i
            sky.subset = subset
            imgs.append(sky)
        path = write_multiband_skyimages(imgs, f"clean_{subset}",
                                         records_dir=args.records_dir)
        _log(f"  {subset}: done — {len(imgs)} → {path}  "
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

    for subset in ("train", "validate"):
        clean_path = tfrecord_path(args.records_dir, f"clean_{subset}")
        if not os.path.exists(clean_path):
            _log(f"⚠️  {clean_path} not found, skipping {subset}")
            continue

        records = list(tf.data.TFRecordDataset(clean_path))
        rng = np.random.default_rng(0xEC11D + (1 if subset == "validate" else 0))
        lr_imgs, hr_imgs = [], []

        _log(f"  {subset}: forward-modelling {len(records)} fields")
        t0 = time.perf_counter()
        for i, raw in enumerate(tqdm(records, desc=f"  {subset}", unit="img")):
            # clean_{subset} stores the 4-channel HR clean field (kept
            # untouched for inspection). The 1-channel VIS HR target the
            # network consumes is written to a separate ``hr_{subset}``
            # file so all four bands of the clean record stay available.
            hr_4ch = MultiBandSkyImage.from_tfrecord(raw)
            lr, hr = fwd.process(hr_4ch, rng=rng)
            lr.index = i
            hr.index = i
            lr.subset = subset
            hr.subset = subset
            lr_imgs.append(lr)
            hr_imgs.append(hr)

        # clean_{subset} is NOT rewritten — leaves the 4-band record
        # intact for inspection. dirty_{subset} = 4-ch LR input;
        # hr_{subset} = 1-ch VIS HR target used by the loader.
        write_multiband_skyimages(hr_imgs, f"hr_{subset}",
                                  records_dir=args.records_dir)
        write_multiband_skyimages(lr_imgs, f"dirty_{subset}",
                                  records_dir=args.records_dir)
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
