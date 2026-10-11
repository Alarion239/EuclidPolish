#!/usr/bin/env python
"""Benchmark: how does the cost of a training step scale with batch size?

Times the real training step of a learned-knee multi-knee member (the
member-206 recipe: six asinh knees in, one image out, balanced L2 loss,
learned output knee) on one GPU, at several depths and batch sizes:

  * ``gpu``   — the compiled train step on one fixed batch, no input pipeline:
                the GPU's cost of a step;
  * ``input`` — the live on-the-fly input pipeline alone (clean field →
                forward model on the full field → crops → dihedral → stretch →
                batch): the crops per second the CPUs deliver;
  * ``live``  — the train step fed by the live pipeline: what training sees.

The step mirrors :meth:`euclid_polish.training.trainer.Trainer.train_step`
(gradient tape, global-norm clip, Adam) and the pipeline is
:meth:`euclid_polish.model.Model._build_onthefly_pipeline` itself, built with
the forward model training uses. ``clean_train`` is staged to node-local
scratch first (and removed at the end), as a training job does. Prints one row per configuration
(ms per step, crops per second, peak host RAM) and writes them as JSON.

Run it on a node of the partition training uses (the gpu partition):

    python scripts/benchmark_train_step.py --depths 16,32 --batches 4,8,16,32 \\
        --star-prior-file logs/pipeline/<job>.star-population.<sha>.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import tensorflow as tf
import tf_keras
from tf_keras.optimizers import Adam

from euclid_polish.config import Config
from euclid_polish.image.tfio import tfrecord_path
from euclid_polish.model import Model
from euclid_polish.training.forward_onthefly import OnTheFlyForward, member_psf_sets
from euclid_polish.training.staging import stage_records

#: The member-206 recipe's knees and starting output knee.
KNEES = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
OUTPUT_KNEE_E = 10.0


def _peak_rss_gb() -> float:
    """Peak resident memory of this process so far (KiB on Linux, bytes on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


def _gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20, check=False)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _make_step(model: Model, learning_rate: float):
    """The trainer's step: tape → loss → global-norm clip → Adam."""
    net = model._tf_model
    loss_fn = model._member_loss("l2", "balanced")
    optimizer = Adam(learning_rate)

    @tf.function
    def step(lr, hr):
        with tf.GradientTape() as tape:
            loss = loss_fn(net(lr, training=True), hr)
        grads = tape.gradient(loss, net.trainable_variables)
        grads, gnorm = tf.clip_by_global_norm(grads, clip_norm=Config.GRAD_CLIP_NORM)
        optimizer.apply_gradients(zip(grads, net.trainable_variables, strict=False))
        return loss, gnorm

    return step


def _time_gpu(step, lr, hr, warmup: int, steps: int) -> float:
    """ms per step on one fixed batch (one sync at the end, like training)."""
    for _ in range(warmup):
        loss, _g = step(lr, hr)
    float(loss)
    t0 = time.perf_counter()
    for _ in range(steps):
        loss, _g = step(lr, hr)
    float(loss)
    return (time.perf_counter() - t0) / steps * 1000.0


def _time_live(step, dataset, warmup: int, steps: int) -> float:
    """ms per step fed by the live pipeline."""
    it = iter(dataset)
    for _ in range(warmup):
        loss, _g = step(*next(it))
    float(loss)
    t0 = time.perf_counter()
    for _ in range(steps):
        loss, _g = step(*next(it))
    float(loss)
    return (time.perf_counter() - t0) / steps * 1000.0


def _time_input(dataset, batch: int, warmup: int, batches: int) -> float:
    """Crops per second the input pipeline alone delivers."""
    it = iter(dataset)
    for _ in range(warmup):
        next(it)
    t0 = time.perf_counter()
    for _ in range(batches):
        next(it)
    return batches * batch / (time.perf_counter() - t0)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depths", default="16,32", help="comma list of residual-block counts")
    p.add_argument("--batches", default="4,8,16,32", help="comma list of batch sizes")
    p.add_argument("--crops-per-field", default="8,16",
                   help="comma list; the first is used for gpu/live rows, all for input rows")
    p.add_argument("--hr-crop-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--records-dir", default=Config.RECORDS_DIR_V2)
    p.add_argument("--no-stage", action="store_true", help="read clean_train in place")
    p.add_argument("--star-prior-file", default=None,
                   help="a training job's star-population JSON (on-the-fly stars)")
    p.add_argument("--gpu-steps", type=int, default=100)
    p.add_argument("--live-steps", type=int, default=200)
    p.add_argument("--input-batches", type=int, default=60)
    p.add_argument("--seed", type=int, default=2898473870)
    p.add_argument("--out", default=None, help="JSON results path")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    depths = [int(v) for v in args.depths.split(",") if v.strip()]
    batches = [int(v) for v in args.batches.split(",") if v.strip()]
    ks = [int(v) for v in args.crops_per_field.split(",") if v.strip()]
    meta = {"host": platform.node(), "gpu": _gpu_name(), "cpus": os.cpu_count(),
            "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"), "tf": tf.__version__}
    print(json.dumps(meta), flush=True)

    t0 = time.perf_counter()
    records_dir = (args.records_dir if args.no_stage
                   else stage_records(args.records_dir, ["clean_train"]))
    clean = tfrecord_path(records_dir, "clean_train")
    print(f"records: {clean} ({time.perf_counter() - t0:.0f} s to stage)", flush=True)
    star_prior = None
    if args.star_prior_file:
        with open(args.star_prior_file) as fh:
            star_prior = json.load(fh)

    t0 = time.perf_counter()
    psf_sets, note = member_psf_sets(seed=args.seed, psf_subset=64)
    print(f"PSF: {note} ({time.perf_counter() - t0:.0f} s); peak RSS {_peak_rss_gb():.1f} GB", flush=True)
    forwards = {k: OnTheFlyForward(psf_sets, seed=args.seed, crops_per_field=k,
                                   hr_crop_size=args.hr_crop_size, scale=2, starless=False,
                                   psf_warp_prob=1.0, psf_warp_alpha_max=5.0, psf_warp_sigma=3.0,
                                   saturation_mask_prob=0.5,
                                   target_fwhm_arcsec=Config.TARGET_PSF_FWHM_ARCSEC,
                                   star_prior_payload=star_prior)
                for k in ks}

    rows = []
    tmp = tempfile.mkdtemp(prefix="bench_train_step_")
    try:
        # Input throughput depends on the CPUs and crops per field, not on the
        # depth: one row per crops-per-field value, at the largest batch.
        probe = Model(os.path.join(tmp, "probe"), scale=2, num_res_blocks=depths[0],
                      asinh_knees=KNEES, output_knee=OUTPUT_KNEE_E, learn_output_knee=True,
                      seed=args.seed)
        for k in ks:
            ds = probe._build_onthefly_pipeline(clean, max(batches), forwards[k])
            cps = _time_input(ds, max(batches), warmup=10, batches=args.input_batches)
            row = {"kind": "input", "crops_per_field": k, "batch": max(batches),
                   "crops_per_s": round(cps, 1), "peak_rss_gb": round(_peak_rss_gb(), 1)}
            rows.append(row)
            print(json.dumps(row), flush=True)
        del probe

        k0 = ks[0]
        for depth in depths:
            for batch in batches:
                model = Model(os.path.join(tmp, f"d{depth}_b{batch}"), scale=2,
                              num_res_blocks=depth, asinh_knees=KNEES, output_knee=OUTPUT_KNEE_E,
                              learn_output_knee=True, seed=args.seed)
                step = _make_step(model, args.learning_rate)
                ds = model._build_onthefly_pipeline(clean, batch, forwards[k0])
                lr, hr = next(iter(ds))
                gpu_ms = _time_gpu(step, lr, hr, warmup=20, steps=args.gpu_steps)
                live_ms = _time_live(step, ds, warmup=40, steps=args.live_steps)
                row = {"kind": "step", "depth": depth, "batch": batch, "crops_per_field": k0,
                       "gpu_ms": round(gpu_ms, 2), "live_ms": round(live_ms, 2),
                       "crops_per_s_live": round(batch / live_ms * 1000.0, 1),
                       "s_per_1k_steps": round(live_ms, 1),
                       "peak_rss_gb": round(_peak_rss_gb(), 1)}
                rows.append(row)
                print(json.dumps(row), flush=True)
                del model, step, ds
                tf_keras.backend.clear_session()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if records_dir != args.records_dir:          # the staged copy, as training does
            shutil.rmtree(records_dir, ignore_errors=True)

    print("\n depth  batch   gpu ms/step   live ms/step   crops/s (live)   vs batch 4 crops/s")
    for depth in depths:
        base = next((r for r in rows if r.get("depth") == depth and r["batch"] == batches[0]), None)
        for r in (r for r in rows if r.get("depth") == depth):
            gain = r["crops_per_s_live"] / base["crops_per_s_live"] if base else float("nan")
            print(f" {depth:5d}  {r['batch']:5d}   {r['gpu_ms']:11.1f}   {r['live_ms']:12.1f}   "
                  f"{r['crops_per_s_live']:14.1f}   {gain:8.2f}×")
    for r in (r for r in rows if r["kind"] == "input"):
        print(f" input pipeline alone, {r['crops_per_field']} crops/field: {r['crops_per_s']:.0f} crops/s")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"meta": meta, "rows": rows}, fh, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
