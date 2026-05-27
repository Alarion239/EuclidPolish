#!/usr/bin/env python
"""Generate (HST-PSF-blurred, Euclid-PSF-blurred) training pairs.

Reads the synthetic clean HR records produced by the standard
``scripts/run_pipeline.py`` (or its FASRC sky-data step) and convolves
each scene with both PSFs:

    input  = clean_VIS ⊛ PSF_HST_on_HR    (no noise)
    target = clean_VIS ⊛ PSF_Euclid_on_HR (no noise)

Both PSFs are resampled onto the project HR grid
(``Config.DEFAULT_PIXEL_SCALE`` = 0.05″/pix) using the same flux-
preserving zoom + centroid-recenter machinery the analytic kernel
script uses, so the two PSFs differ only in physics, not in any
preprocessing convention.

Output: TFRecord pairs under
``$DATA_DIR/images/records_transition/{input,target}_{train,validate}.tfrecord``
in the same ``MultiBandSkyImage`` schema the rest of the pipeline
uses (single-channel "VIS" band, ``is_clean=True`` — these are
deterministic PSF-blurred scenes, not noisy LR observations).

This step is the prerequisite for training ``A_θ`` via
``scripts/fasrc_train_transition_model.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from astropy.io import fits
from scipy import signal as scipy_signal
from scipy.ndimage import zoom

from euclid_polish.psf import load_hst_f814w_psf
from euclid_polish.psf import load_euclid_band_psf
from euclid_polish.psf import PSF as _PSF

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.euclid.psf_library import psf_path_for_band
from euclid_polish.euclid.types import PSF
from euclid_polish.sky.tfrecord import (
    open_multiband_writer, read_multiband_skyimages, tfrecord_path,
)
from euclid_polish.sky.types import MultiBandSkyImage
from euclid_polish.training.transition_augmentations import (
    ensure_unit_sum, sum_rebin_2d,
)


SYNTHETIC_RECORDS_DIR = Config.RECORDS_DIR_V2
HST_PSF_PATH = os.path.join(Config.DATA_DIR, "hst_psf", "F814W.fits")
OUT_DIR      = os.path.join(Config.DATA_DIR, "images", "records_transition")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic-records-dir", default=SYNTHETIC_RECORDS_DIR,
                   help="Source clean-HR records (synthetic pipeline output).")
    p.add_argument("--hst-psf", default=HST_PSF_PATH,
                   help="HST F814W ePSF FITS (extracted by step 2 on FASRC).")
    p.add_argument("--output-dir", default=OUT_DIR,
                   help="Where to write the transition-pair TFRecords.")
    p.add_argument("--n-train", type=int, default=4000,
                   help="Training pairs to emit. Capped by the available "
                        "clean_train records on disk.")
    p.add_argument("--n-valid", type=int, default=400,
                   help="Validation pairs to emit. Capped by available "
                        "clean_validate records.")
    p.add_argument("--crop-size", type=int, default=256,
                   help="Centre-crop each clean scene to (crop_size,"
                        " crop_size) before convolving. The synthetic "
                        "records are ~512² but 256² is plenty for "
                        "training a per-pixel PSF transition and keeps "
                        "per-pair memory + the TFRecord size manageable. "
                        "Must be divisible by --rebin-factor.")
    p.add_argument("--rebin-factor", type=int, default=2,
                   help="Sum-rebin factor applied to the Euclid-blurred "
                        "target before writing it to disk. Default 2 "
                        "matches the Euclid HR→LR pixel-scale ratio "
                        "(0.05\"/pix → 0.10\"/pix). The trainer "
                        "compares ``sum_rebin(A_θ(input))`` (at LR) "
                        "against this LR target, so the model only has "
                        "to be correct at LR — high-frequency residuals "
                        "from the spike-geometry mismatch between HST "
                        "and Euclid PSFs get averaged away by the "
                        "rebin. Input stays at HR.")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan only — no FITS reads, no TFRecord writes.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# PSF loaders — thin wrappers that delegate the whole "FITS → HR-grid PSF"
# pipeline to ``euclid_polish.psf``. Kept here with numpy-array return
# signatures so older callers keep working unchanged; new code should
# call :func:`load_hst_f814w_psf` / :func:`load_euclid_band_psf` directly
# to get the typed :class:`PSF` instead.
# ---------------------------------------------------------------------------

def _resample_to_hr_grid(psf_data: np.ndarray, src_scale: float) -> np.ndarray:
    """Resample to ``Config.DEFAULT_PIXEL_SCALE`` (0.05″/pix).

    Thin wrapper around :meth:`euclid_polish.psf.PSF.resampled_to` so
    every PSF resampling site shares one implementation. Returns a
    sum=1 numpy array.
    """
    p = PSF(
        data=np.asarray(psf_data, dtype=np.float32),
        pixel_scale=float(src_scale),
    )
    return p.with_unit_sum().resampled_to(
        Config.DEFAULT_PIXEL_SCALE,
    ).data.astype(np.float64)


def _crop_to_odd_square(psf: np.ndarray, side: int) -> np.ndarray:
    """Centre-crop to ``(side, side)``, padding with zeros if needed,
    renormalised to sum=1. Delegates to
    :meth:`euclid_polish.psf.PSF.centre_cropped_to`."""
    p = PSF(
        data=np.asarray(psf, dtype=np.float32),
        pixel_scale=Config.DEFAULT_PIXEL_SCALE,    # only the data shape matters here
    )
    return p.centre_cropped_to(side).data


def _load_hst_psf_on_hr(path: str, target_side: int) -> np.ndarray:
    """Legacy entry point — returns the kernel as a numpy array."""
    return load_hst_f814w_psf(
        path, target_side=target_side,
    ).data.astype(np.float32)


def _load_euclid_vis_psf_on_hr(target_side: int) -> np.ndarray:
    """Legacy entry point — returns the kernel as a numpy array."""
    return load_euclid_band_psf(
        Config.BAND_VIS, target_side=target_side,
    ).data.astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic-scene reader
# ---------------------------------------------------------------------------

def _stream_clean_vis_scenes(
    records_dir: str, subset: str, n_max: int, *, crop: int,
):
    """Yield up to ``n_max`` clean-HR VIS scenes from the synthetic records.

    The synthetic ``clean_{subset}.tfrecord`` contains 4-channel HR
    fields (one per LR band) at 0.05″/pix. Only channel 0 (VIS) is used
    for transition-model training — both PSFs we resample are VIS-band
    PSFs, and the network operates per-channel anyway.

    Yields ``(idx, scene)`` with ``scene`` shape ``(crop, crop)`` float32
    in linear electron units. Scenes are centre-cropped to ``crop``;
    smaller-than-``crop`` source records are skipped (warnings printed).
    """
    path = tfrecord_path(records_dir, f"clean_{subset}")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"clean_{subset}.tfrecord not found at {path}. "
            "Run scripts/run_pipeline.py (or the FASRC sky-data step) first."
        )
    ds = tf.data.TFRecordDataset([path])
    yielded = 0
    skipped_small = 0
    for raw in ds:
        if yielded >= n_max:
            break
        try:
            img = MultiBandSkyImage.from_tfrecord(raw)
        except Exception:
            continue
        data = np.asarray(img.data, dtype=np.float32)
        if data.ndim != 3:
            continue
        H, W, C = data.shape
        if H < crop or W < crop:
            skipped_small += 1
            continue
        # VIS is always channel 0 in the project's band ordering.
        # Delegate the centre-crop to the central MultiBandSkyImage
        # helper so the math lives in one place.
        scene = MultiBandSkyImage.crop_array(data[:, :, 0], crop)
        yield img.index if img.index is not None else yielded, scene
        yielded += 1
    if skipped_small:
        print(f"      skipped {skipped_small} scenes smaller than "
              f"crop {crop}² in clean_{subset}.tfrecord")


# ---------------------------------------------------------------------------
# Pair synthesis
# ---------------------------------------------------------------------------

def _convolve_pair(
    scene: np.ndarray,
    psf_hst: np.ndarray,
    psf_euclid: np.ndarray,
    *,
    rebin_factor: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Produce ``(scene ⊛ PSF_HST at HR, sum_rebin(scene ⊛ PSF_Euclid))``.

    The input side is left at HR (same shape as ``scene``). The
    target side is convolved with the Euclid PSF then sum-rebinned by
    ``rebin_factor`` so it matches what downstream consumers see after
    the ``A_θ → sum_rebin`` chain. Both convolutions use
    ``scipy.signal.fftconvolve(mode='same')`` — the same convention
    as the analytic-kernel forward in
    ``fasrc_generate_hst_tfrecords.py``.

    Returns ``(input, target)`` where
    ``input.shape == scene.shape`` and
    ``target.shape == (H//rebin_factor, W//rebin_factor)``.
    """
    # Wrap the raw kernel arrays as PSF instances so the convolution
    # path goes through the central ``PSF.convolved_with`` method
    # (defensive sum=1 normalisation + fftconvolve mode="same") and
    # the rebin goes through ``MultiBandSkyImage.rebin_array``. No
    # inline fftconvolve / reshape().sum() duplication.
    psf_hst_obj    = _PSF(data=psf_hst,    pixel_scale=Config.DEFAULT_PIXEL_SCALE)
    psf_euclid_obj = _PSF(data=psf_euclid, pixel_scale=Config.DEFAULT_PIXEL_SCALE)
    inp_hr = psf_hst_obj.convolved_with(scene)
    tgt_hr = psf_euclid_obj.convolved_with(scene)
    tgt_lr = MultiBandSkyImage.rebin_array(tgt_hr, int(rebin_factor))
    return inp_hr, tgt_lr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 64)
    print(f"  Transition-model training-pair generation")
    print(f"  (input = clean ⊛ PSF_HST, target = sum_rebin(clean ⊛ PSF_Euclid))")
    print("=" * 64)
    print(f"  synthetic records = {args.synthetic_records_dir}")
    print(f"  HST PSF           = {args.hst_psf}")
    print(f"  output            = {args.output_dir}")
    print(f"  n_train / n_valid = {args.n_train} / {args.n_valid}")
    print(f"  crop_size (HR)    = {args.crop_size} px")
    print(f"  rebin_factor      = {args.rebin_factor} "
          f"(target side = {int(args.crop_size) // int(args.rebin_factor)} LR px)")
    print(f"  dry run           = {args.dry_run}")
    print()

    t0 = time.time()
    crop = int(args.crop_size)
    rebin = int(args.rebin_factor)
    if crop < 8:
        print(f"ERROR: --crop-size must be ≥ 8, got {crop}")
        return 1
    if rebin < 1:
        print(f"ERROR: --rebin-factor must be ≥ 1, got {rebin}")
        return 1
    if crop % rebin != 0:
        print(f"ERROR: --crop-size ({crop}) must be divisible by "
              f"--rebin-factor ({rebin})")
        return 1

    if not os.path.isfile(args.hst_psf):
        print(f"ERROR: HST PSF not found at {args.hst_psf}")
        print("       Run scripts/fasrc_extract_hst_psf.py first.")
        return 1

    print("[1/3] resampling both PSFs onto the HR grid ...")
    # PSF support: cap at ~21″ across (≈ 421 px at 0.05″/pix, +1 to be
    # odd). This covers Euclid VIS 6-spike + HST F814W F-shape well
    # while staying tractable for FFT convolution against 256² scenes.
    psf_side = 421
    psf_hst    = _load_hst_psf_on_hr(args.hst_psf, psf_side)
    psf_euclid = _load_euclid_vis_psf_on_hr(psf_side)
    print(f"      HST PSF on HR    : shape={psf_hst.shape}    sum={psf_hst.sum():.4f}")
    print(f"      Euclid PSF on HR : shape={psf_euclid.shape} sum={psf_euclid.sum():.4f}")

    if args.dry_run:
        print("\nDRY RUN — would emit "
              f"{args.n_train + args.n_valid} pairs at {crop}² HR.")
        runtime = time.time() - t0
        print(f"\nRUNTIME_SECONDS={runtime:.1f}")
        return 0

    print(f"[2/3] streaming clean scenes from "
          f"{args.synthetic_records_dir} ...")

    # ── Pre-flight: probe one synthetic scene and verify it can be
    # cropped to the requested size. If not, fail immediately BEFORE
    # opening any output writers (which would leave behind empty
    # 0-byte TFRecord files that then crash the trainer downstream).
    # Cheap — reads exactly one record from clean_train.tfrecord.
    print(f"      pre-flight: probing scene size in clean_train.tfrecord ...")
    src_path = tfrecord_path(args.synthetic_records_dir, "clean_train")
    if not os.path.isfile(src_path):
        print(f"ERROR: synthetic clean_train.tfrecord not found at "
              f"{src_path}. Generate synthetic data first "
              f"(scripts/run_pipeline.py or the FASRC sky-data step).")
        return 1
    src_records = read_multiband_skyimages(src_path, num_images=1)
    if not src_records:
        print(f"ERROR: synthetic clean_train.tfrecord at {src_path} "
              f"has no records.")
        return 1
    src_H, src_W, _ = src_records[0].data.shape
    print(f"      synthetic scenes are {src_H}×{src_W}")
    if src_H < crop or src_W < crop:
        # Round suggestion down to a multiple of rebin so it's
        # immediately usable in --crop-size.
        suggest = (min(src_H, src_W) // rebin) * rebin
        print()
        print("=" * 64)
        print("ERROR: --crop-size larger than synthetic scenes.")
        print("=" * 64)
        print(f"  Requested:    --crop-size {crop}")
        print(f"  Source scenes: {src_H}×{src_W} pixels")
        print()
        print(f"  Every scene would be skipped, the output shards would be")
        print(f"  0 bytes, and the trainer would crash on the empty files.")
        print()
        print(f"  Re-run with --crop-size {suggest} (or smaller).")
        print(f"  If you want a larger crop, regenerate synthetic data with")
        print(f"  a bigger --image-size first.")
        print()
        return 2

    summary = {"subsets": {}}
    pairs_written = 0
    pairs_per_subset = {"train": args.n_train, "validate": args.n_valid}

    print(f"[3/3] writing pairs to {args.output_dir} ...")
    for subset, target_n in pairs_per_subset.items():
        sub_done = 0
        with open_multiband_writer(
            f"input_{subset}", records_dir=args.output_dir,
        ) as iw, open_multiband_writer(
            f"target_{subset}", records_dir=args.output_dir,
        ) as tw:
            for idx, scene in _stream_clean_vis_scenes(
                args.synthetic_records_dir, subset, target_n, crop=crop,
            ):
                inp, tgt = _convolve_pair(
                    scene, psf_hst, psf_euclid, rebin_factor=rebin,
                )
                meta = {"source": "synthetic_psf_transition",
                        "rebin_factor": rebin}
                inp_img = MultiBandSkyImage(
                    data=inp[..., np.newaxis],
                    pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                    band_names=("VIS",), is_clean=True, metadata=meta,
                )
                # Target lives on the LR grid (HR scale × rebin_factor).
                tgt_img = MultiBandSkyImage(
                    data=tgt[..., np.newaxis],
                    pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE * rebin,
                    band_names=("VIS",), is_clean=True, metadata=meta,
                )
                iw.write(inp_img, index=sub_done)
                tw.write(tgt_img, index=sub_done)
                sub_done += 1
                pairs_written += 1
                if sub_done % 200 == 0:
                    print(f"      {subset}: {sub_done}/{target_n} written")
        summary["subsets"][subset] = {"written": sub_done, "target": target_n}
        print(f"      ✓ {subset}: {sub_done} pairs written")

    runtime = time.time() - t0
    summary["pairs_written"] = pairs_written
    summary["elapsed_s"]     = round(runtime, 1)
    summary["crop_size"]     = crop
    summary["rebin_factor"]  = rebin
    with open(os.path.join(args.output_dir, "generation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"  wrote {pairs_written} pairs total")
    print(f"\nRUNTIME_SECONDS={runtime:.1f}")
    print(f"PAIRS_WRITTEN={pairs_written}")

    # Hard fail when zero pairs were written. Returning 0 here would
    # leave behind a 0-record TFRecord that crashes the downstream
    # trainer with the much-less-useful "input_train.tfrecord is empty".
    # The most common cause is --crop-size larger than the synthetic
    # scene side (Config.DEFAULT_IMAGE_SIZE = 256), so spell that out.
    if pairs_written == 0:
        print()
        print("=" * 64)
        print("ERROR: 0 pairs written.")
        print("=" * 64)
        # ``Config`` is already imported at module scope (line ~42).
        # Re-importing it here as a local would have made it function-
        # local everywhere in main() via Python's "any binding makes the
        # name local for the whole function" rule, breaking the earlier
        # `Config.DEFAULT_PIXEL_SCALE` references in the writer loop
        # with ``UnboundLocalError``.
        print(f"  Most likely cause: --crop-size ({crop}) is LARGER than the")
        print(f"  synthetic clean-scene side in {args.synthetic_records_dir}/")
        print(f"  clean_*.tfrecord. Default synthetic image size is")
        print(f"  Config.DEFAULT_IMAGE_SIZE = {Config.DEFAULT_IMAGE_SIZE} px.")
        print()
        print(f"  Re-run with --crop-size {Config.DEFAULT_IMAGE_SIZE} (or smaller).")
        print(f"  If you want larger crops, regenerate your synthetic data with")
        print(f"  a bigger --image-size first.")
        print()
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
