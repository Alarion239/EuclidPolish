#!/usr/bin/env python
"""Benchmark: what does a training step cost, and what makes it cheaper?

Times the real training step of a learned-knee multi-knee member (the
member-206 recipe: six asinh knees in, one image out, balanced L2 loss,
learned output knee) on one GPU.

Speed (``--mode speed``, the default), at several depths and batch sizes and
for every step variant in ``--variants``:

  * ``gpu``   — the compiled train step on one fixed batch, no input pipeline:
                the GPU's cost of a step;
  * ``input`` — the live on-the-fly input pipeline alone (clean field →
                forward model on the full field → crops → dihedral → stretch →
                batch): the crops per second the CPUs deliver;
  * ``live``  — the train step fed by the live pipeline: what training sees.

A variant is ``control`` (the step exactly as training runs it) or a
``+``-joined set of speed-ups:

  * ``xla``     — the step compiled with XLA (``jit_compile=True``);
  * ``bf16``    — the residual blocks' convolutions in bfloat16
                  (``wdsr(mixed_precision=True)``);
  * ``eagerwn`` — the weight norms initialised when built, without the
                  per-call check (``wdsr(eager_weight_norm_init=True)``).

The step mirrors :meth:`euclid_polish.training.trainer.Trainer.train_step`
(gradient tape, global-norm clip, Adam, the step counter and window
statistics) and the pipeline is
:meth:`euclid_polish.model.Model._build_onthefly_pipeline` itself, built with
the forward model training uses. ``clean_train`` is staged to node-local
scratch first (and removed at the end), as a training job does.

Precision (``--mode precision``): runs a trained member on the scoring fields
(knee-integrated PSNR, as the Leaderboard scores it) and takes its gradients
on fixed training-sized crops of those fields, under one arithmetic per
process: ``strict`` (float32, TF32 off), ``tf32`` (TensorFlow's default on
Ampere GPUs, as training runs) or ``bf16``; each without and with XLA.
``--mode compare`` reads those runs and reports, against ``strict`` without
XLA: the PSNR, how closely the outputs agree, and the gradient error next to
the batch-to-batch spread of the gradient itself.

Run it on a GPU node, e.g.:

    python scripts/benchmark_train_step.py --depths 16 --batches 4,8 \\
        --variants control,xla,bf16,xla+bf16 \\
        --star-prior-file logs/pipeline/<job>.star-population.<sha>.json
    python scripts/benchmark_train_step.py --mode precision --arith bf16 \\
        --member <member dir> --fields-json <fields.json> --out bf16.npz --sr-dir <dir>
    python scripts/benchmark_train_step.py --mode compare --inputs strict.npz tf32.npz bf16.npz
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

import numpy as np
import tensorflow as tf
import tf_keras
from tf_keras.optimizers import Adam

from euclid_polish.config import Config
from euclid_polish.eval.knee_psnr import KNEE_GRID_E, integrated_psnr, knee_psnr, stretched_truth
from euclid_polish.image.collection import ImageSet
from euclid_polish.image.tfio import tfrecord_path
from euclid_polish.model import Model
from euclid_polish.training.augmentation import asinh_stretch_multi_knee, stretch_pair
from euclid_polish.training.forward_onthefly import OnTheFlyForward, member_psf_sets
from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.training.staging import stage_records
from euclid_polish.training.target_blur import blur_target_array, validate_target_fwhm_arcsec
from euclid_polish.training.trainer import GRAD_CLIP_NORM, GRAD_SPIKE_SKIP_WARMUP_STEPS

#: The member-206 recipe's knees and starting output knee.
KNEES = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
OUTPUT_KNEE_E = 10.0
#: Speed-ups a step variant can combine.
VARIANT_FLAGS = ("xla", "bf16", "eagerwn")
ARITHMETICS = ("strict", "tf32", "bf16")


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


def parse_variant(name: str) -> frozenset[str]:
    """The speed-ups of a variant name: ``control`` → none, else its
    ``+``-joined flags."""
    if name == "control":
        return frozenset()
    flags = frozenset(name.split("+"))
    if not flags or not flags <= set(VARIANT_FLAGS):
        raise ValueError(f"unknown step variant {name!r}: use control or a '+'-joined "
                         f"subset of {', '.join(VARIANT_FLAGS)}")
    return flags


def _twin(depth: int, *, bf16: bool, eagerwn: bool, n_knees: int = len(KNEES)):
    """The training network as :class:`Model` builds it, with build options."""
    return wdsr(scale=2, num_res_blocks=depth, nchan_in=Config.NUM_LR_CHANNELS * n_knees,
                nchan_out=Config.NUM_HR_CHANNELS, input_knees=n_knees,
                learned_output_knee=OUTPUT_KNEE_E,
                mixed_precision=bf16, eager_weight_norm_init=eagerwn)


def _make_step(net, loss_fn, learning_rate: float, *, jit: bool = False, stretch=None):
    """The trainer's step: [the member's stretch of a raw batch, on the
    device] → tape → loss → global-norm clip → Adam → step counter and window
    statistics; ``jit`` compiles it with XLA."""
    optimizer = Adam(learning_rate)
    # int64, unlike the trainer's int32 counter: TensorFlow keeps an int32
    # variable in host memory, out of reach of a step XLA compiles for the GPU.
    counter = tf.Variable(tf.constant(0, tf.int64))
    window = {k: tf.Variable(0.0) for k in ("loss", "gnorm_sum", "gnorm_max", "steps", "spike")}

    def step(lr, hr):
        if stretch is not None:
            lr, hr = stretch(lr, hr)
        with tf.GradientTape() as tape:
            loss = loss_fn(net(lr, training=True), hr)
        grads = tape.gradient(loss, net.trainable_variables)
        grads, gnorm = tf.clip_by_global_norm(grads, clip_norm=GRAD_CLIP_NORM)
        optimizer.apply_gradients(zip(grads, net.trainable_variables, strict=False))
        n = counter.assign_add(1)
        window["loss"].assign_add(loss)
        window["gnorm_sum"].assign_add(gnorm)
        window["gnorm_max"].assign(tf.maximum(window["gnorm_max"], gnorm))
        window["steps"].assign_add(1.0)
        watched = tf.where(tf.math.is_finite(gnorm), gnorm, float("inf"))
        watched = tf.where(n > GRAD_SPIKE_SKIP_WARMUP_STEPS, watched, 0.0)
        window["spike"].assign(tf.maximum(window["spike"], watched))
        return loss, gnorm

    return tf.function(step, jit_compile=jit)


def _first_call_s(step, lr, hr) -> float:
    """Seconds of the first call: tracing, plus compiling under XLA."""
    t0 = time.perf_counter()
    loss, _g = step(lr, hr)
    float(loss)
    return time.perf_counter() - t0


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


def _speed(args: argparse.Namespace, meta: dict) -> int:
    depths = [int(v) for v in args.depths.split(",") if v.strip()]
    batches = [int(v) for v in args.batches.split(",") if v.strip()]
    ks = [int(v) for v in args.crops_per_field.split(",") if v.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    flags_of = {v: parse_variant(v) for v in variants}

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

        for depth in depths:
            for batch in batches:
                for variant in variants:
                    flags = flags_of[variant]
                    model = Model(os.path.join(tmp, f"d{depth}_b{batch}_{len(rows)}"), scale=2,
                                  num_res_blocks=depth, asinh_knees=KNEES, output_knee=OUTPUT_KNEE_E,
                                  learn_output_knee=True, seed=args.seed)
                    # control trains the model's own network; a build option
                    # needs a network built with it.
                    net = (model._tf_model if not flags & {"bf16", "eagerwn"}
                           else _twin(depth, bf16="bf16" in flags, eagerwn="eagerwn" in flags))
                    # The live pipeline yields raw electrons; the step
                    # stretches them on the device, as the trainer does.
                    step = _make_step(net, model._member_loss("l2", "balanced"),
                                      args.learning_rate, jit="xla" in flags,
                                      stretch=model.batch_stretch())
                    datasets = {k: model._build_onthefly_pipeline(clean, batch, forwards[k])
                                for k in ks}
                    lr, hr = next(iter(datasets[ks[0]]))
                    try:
                        compile_s = _first_call_s(step, lr, hr)
                    except (tf.errors.OpError, ValueError, TypeError) as exc:
                        # A variant that cannot run is a result too; measure the rest.
                        row = {"kind": "step", "depth": depth, "batch": batch, "variant": variant,
                               "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"}
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                        del model, net, step, datasets
                        tf_keras.backend.clear_session()
                        continue
                    gpu_ms = _time_gpu(step, lr, hr, warmup=20, steps=args.gpu_steps)
                    live_ms = {k: _time_live(step, datasets[k], warmup=40, steps=args.live_steps)
                               for k in ks}
                    row = {"kind": "step", "depth": depth, "batch": batch, "variant": variant,
                           "compile_s": round(compile_s, 1), "gpu_ms": round(gpu_ms, 2),
                           "live_ms": {str(k): round(ms, 2) for k, ms in live_ms.items()},
                           "crops_per_s_live": {str(k): round(batch / ms * 1000.0, 1)
                                                for k, ms in live_ms.items()},
                           "peak_rss_gb": round(_peak_rss_gb(), 1)}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    del model, net, step, datasets
                    tf_keras.backend.clear_session()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if records_dir != args.records_dir:          # the staged copy, as training does
            shutil.rmtree(records_dir, ignore_errors=True)

    live_heads = "".join(f"   live ms K={k:<3d} ×ctrl" for k in ks)
    print(f"\n depth  batch  variant               compile s   gpu ms   ×ctrl{live_heads}")
    for depth in depths:
        for batch in batches:
            group = [r for r in rows if r.get("depth") == depth and r.get("batch") == batch]
            base = next((r for r in group if r["variant"] == "control" and "error" not in r), None)
            for r in group:
                if "error" in r:
                    print(f" {depth:5d}  {batch:5d}  {r['variant']:<20s}  failed: {r['error'][:120]}")
                    continue
                gain = base["gpu_ms"] / r["gpu_ms"] if base else float("nan")
                live = "".join(
                    f"   {r['live_ms'][str(k)]:12.1f} "
                    f"{(base['live_ms'][str(k)] / r['live_ms'][str(k)] if base else float('nan')):5.2f}"
                    for k in ks)
                print(f" {depth:5d}  {batch:5d}  {r['variant']:<20s}  {r['compile_s']:9.1f}"
                      f"  {r['gpu_ms']:7.1f}  {gain:5.2f}{live}")
    for r in (r for r in rows if r["kind"] == "input"):
        print(f" input pipeline alone, {r['crops_per_field']} crops/field: {r['crops_per_s']:.0f} crops/s")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"meta": meta, "rows": rows}, fh, indent=1)
        print(f"wrote {args.out}")
    return 0


def _scoring_fields(records_dir: str, indices, fwhm_arcsec: float):
    """``(index, LR electrons, blurred HR target electrons)`` of the scoring
    fields, read the way the Leaderboard's knee-PSNR scoring reads them."""
    wanted = sorted(int(i) for i in indices)
    fwhm = validate_target_fwhm_arcsec(fwhm_arcsec)
    last = max(wanted) + 1
    targets = iter(ImageSet.read(tfrecord_path(records_dir, "hr_test"), num_images=last))
    lrs = iter(ImageSet.read(tfrecord_path(records_dir, "dirty_test"), num_images=last))
    target_rec, lr_rec = next(targets, None), next(lrs, None)
    fields = []
    for index in wanted:
        while target_rec is not None and target_rec.index < index:
            target_rec = next(targets, None)
        while lr_rec is not None and lr_rec.index < index:
            lr_rec = next(lrs, None)
        if target_rec is None or lr_rec is None or target_rec.index != index or lr_rec.index != index:
            continue
        target = np.asarray(blur_target_array(np.asarray(target_rec.data, np.float32), fwhm,
                                              pixel_scale_arcsec=target_rec.pixel_scale_arcsec),
                            np.float32)
        fields.append((index, np.asarray(lr_rec.data, np.float32), target))
    return fields


def _gradient_batches(fields, knees, *, n_batches: int, batch: int, hr_crop: int, seed: int):
    """Fixed training-sized batches cut from the scoring fields: block-aligned
    ``hr_crop`` HR crops with their LR crops, stretched the way the member
    trains. The same seed gives the same batches in every process."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_batches):
        lrs, hrs = [], []
        for _ in range(batch):
            _index, lr, hr = fields[int(rng.integers(len(fields)))]
            x = int(rng.integers(0, (hr.shape[0] - hr_crop) // 2 + 1)) * 2
            y = int(rng.integers(0, (hr.shape[1] - hr_crop) // 2 + 1)) * 2
            hrs.append(hr[x:x + hr_crop, y:y + hr_crop])
            lrs.append(lr[x // 2:(x + hr_crop) // 2, y // 2:(y + hr_crop) // 2])
        out.append(stretch_pair(tf.constant(np.stack(lrs)), tf.constant(np.stack(hrs)), knees=knees))
    return out


def _precision(args: argparse.Namespace, meta: dict) -> int:
    if args.arith == "strict":
        tf.config.experimental.enable_tensor_float_32_execution(False)
    with open(args.fields_json) as fh:
        identity = json.load(fh)
    member = Model(args.member)
    knees = member._asinh_knees
    if not (knees and member._learn_output_knee):
        raise SystemExit("--member must be a learned-knee multi-knee member")
    net = member._tf_model
    if args.arith == "bf16":
        twin = _twin(member._num_res_blocks, bf16=True, eagerwn=False, n_knees=len(knees))
        twin.set_weights(net.get_weights())
        net = twin
    loss_fn = member._member_loss(args.loss, args.knee_loss)
    fields = _scoring_fields(args.records_dir, identity["indices"], identity["target_psf_fwhm_arcsec"])
    print(f"{len(fields)} scoring fields; arithmetic {args.arith}; TF32 "
          f"{'on' if tf.config.experimental.tensor_float_32_execution_enabled() else 'off'}", flush=True)
    batches = _gradient_batches(fields, knees, n_batches=args.grad_batches, batch=args.grad_batch_size,
                                hr_crop=args.hr_crop_size, seed=args.seed)
    os.makedirs(args.sr_dir, exist_ok=True)

    curves, grads, losses, sr_paths, seconds = [], [], [], [], []
    for xla in (False, True):
        # Without XLA the model runs eagerly, as scoring runs it.
        forward = (tf.function(lambda t: net(t, training=False), jit_compile=True) if xla
                   else (lambda t: net(t, training=False)))

        def gradient(lr, hr):
            with tf.GradientTape() as tape:
                loss = loss_fn(net(lr, training=True), hr)
            g = tape.gradient(loss, net.trainable_variables)
            return loss, tf.concat([tf.reshape(v, [-1]) for v in g], axis=0)

        gradient = tf.function(gradient, jit_compile=xla)
        path = os.path.join(args.sr_dir, f"sr_{args.arith}_{'xla' if xla else 'graph'}.npy")
        sr_all = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                           shape=(len(fields),) + fields[0][2].shape)
        run_curves = []
        t0 = time.perf_counter()
        for i, (_index, lr, target) in enumerate(fields):
            stretched = asinh_stretch_multi_knee(tf.constant(lr[None]), knees)
            sr = np.asarray(forward(stretched)[0], np.float32)
            sr_all[i] = sr
            run_curves.append(knee_psnr(sr, target, truth_asinh=stretched_truth(target)))
        sr_all.flush()
        del sr_all
        seconds.append(time.perf_counter() - t0)
        run = [gradient(lr, hr) for lr, hr in batches]
        losses.append([float(loss) for loss, _g in run])
        grads.append(np.stack([np.asarray(g, np.float32) for _loss, g in run]))
        curves.append(np.asarray(run_curves))
        sr_paths.append(path)
        v = integrated_psnr(curves[-1].mean(0))
        print(f"{args.arith} {'XLA' if xla else 'no XLA'}: integrated PSNR "
              + " / ".join(f"{x:.3f}" for x in v) + f"  mean {v.mean():.3f}  "
              f"({seconds[-1]:.1f} s for {len(fields)} fields)", flush=True)

    np.savez(args.out, arith=args.arith, xla=np.array([False, True]),
             fields=np.array([f[0] for f in fields]), knees=np.asarray(KNEE_GRID_E),
             curves=np.stack(curves), grads=np.stack(grads), losses=np.asarray(losses),
             sr_paths=np.array(sr_paths), seconds=np.asarray(seconds),
             meta=json.dumps(meta))
    print(f"wrote {args.out}")
    return 0


def _agreement_db(sr, ref) -> np.ndarray:
    """Knee-integrated PSNR of ``sr`` against ``ref`` per field, ``(N, C)``:
    the arithmetic's error on the scale of the model's own error."""
    return np.stack([integrated_psnr(knee_psnr(sr[i], ref[i], truth_asinh=stretched_truth(ref[i])))
                     for i in range(len(ref))])


def _compare(args: argparse.Namespace) -> int:
    runs = {}
    for path in args.inputs:
        d = np.load(path, allow_pickle=False)
        for i, xla in enumerate(d["xla"]):
            runs[(str(d["arith"]), bool(xla))] = {
                "fields": d["fields"], "curves": d["curves"][i], "grads": d["grads"][i],
                "sr": str(d["sr_paths"][i]), "seconds": float(d["seconds"][i])}
    if ("strict", False) not in runs:
        raise SystemExit("compare needs the strict run (its no-XLA pass is the reference)")
    ref = runs[("strict", False)]
    ref_sr = np.load(ref["sr"], mmap_mode="r")
    g_ref = ref["grads"]
    g_norm = np.linalg.norm(g_ref, axis=1)
    # How much one batch's gradient differs from the mean over the batches:
    # the noise every SGD step carries anyway.
    spread = float(np.mean(np.linalg.norm(g_ref - g_ref.mean(0), axis=1) / g_norm))
    ref_psnr = float(integrated_psnr(ref["curves"].mean(0)).mean())
    order = {a: i for i, a in enumerate(ARITHMETICS)}
    rows = []
    for (arith, xla), r in sorted(runs.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1])):
        assert (r["fields"] == ref["fields"]).all()
        psnr = integrated_psnr(r["curves"].mean(0))
        agree = _agreement_db(np.load(r["sr"], mmap_mode="r"), ref_sr)
        g = r["grads"]
        err = np.linalg.norm(g - g_ref, axis=1) / g_norm
        cos = np.sum(g * g_ref, axis=1) / (np.linalg.norm(g, axis=1) * g_norm)
        rows.append({"arith": arith, "xla": xla, "psnr_bands": [round(float(x), 4) for x in psnr],
                     "psnr": round(float(psnr.mean()), 4),
                     "psnr_minus_ref": round(float(psnr.mean()) - ref_psnr, 4),
                     "agreement_db": round(float(np.mean(agree)), 2),
                     "agreement_db_worst_field": round(float(np.min(agree.mean(1))), 2),
                     "grad_rel_err": float(np.mean(err)), "grad_cos": float(np.mean(cos)),
                     "seconds": round(r["seconds"], 1)})
    print(f"\n reference: strict float32 without XLA; one batch's gradient differs from the "
          f"mean over {len(g_ref)} batches by {spread * 100:.1f}% (SGD noise)")
    print(" arithmetic  XLA   PSNR (VIS/Y/J/H)                 mean     Δ vs ref   agreement dB"
          "   worst field   gradient error   cos")
    for r in rows:
        agree = "exact" if r["agreement_db"] > 250 else f"{r['agreement_db']:.1f}"
        worst = "exact" if r["agreement_db_worst_field"] > 250 else f"{r['agreement_db_worst_field']:.1f}"
        print(f" {r['arith']:<10s}  {'yes' if r['xla'] else 'no ':<4s}  "
              + " / ".join(f"{x:.3f}" for x in r["psnr_bands"])
              + f"   {r['psnr']:.3f}   {r['psnr_minus_ref']:+.4f}   {agree:>12s}   {worst:>11s}"
              f"   {r['grad_rel_err'] * 100:13.3f}%   {r['grad_cos']:.6f}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"grad_batch_spread": spread, "rows": rows}, fh, indent=1)
        print(f"wrote {args.out}")
    return 0


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("speed", "precision", "compare"), default="speed")
    p.add_argument("--depths", default="16,32", help="comma list of residual-block counts")
    p.add_argument("--batches", default="4,8,16,32", help="comma list of batch sizes")
    p.add_argument("--variants", default="control",
                   help=f"comma list of step variants: control or '+'-joined {', '.join(VARIANT_FLAGS)}")
    p.add_argument("--crops-per-field", default="8,16",
                   help="comma list; the first feeds the gpu row, every value an input and a live row")
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
    p.add_argument("--arith", choices=ARITHMETICS, default="tf32", help="precision mode's arithmetic")
    p.add_argument("--member", default=None, help="precision mode: a trained member's checkpoint dir")
    p.add_argument("--fields-json", default=None,
                   help="precision mode: {'indices': [...], 'target_psf_fwhm_arcsec': x}")
    p.add_argument("--loss", default="l2", help="precision mode: the member's loss norm")
    p.add_argument("--knee-loss", default="balanced", help="precision mode: the member's knee loss")
    p.add_argument("--grad-batches", type=int, default=16)
    p.add_argument("--grad-batch-size", type=int, default=4)
    p.add_argument("--sr-dir", default=None, help="precision mode: where the SR outputs go")
    p.add_argument("--inputs", nargs="*", default=[], help="compare mode: precision-mode .npz files")
    p.add_argument("--out", default=None, help="results path (JSON; .npz in precision mode)")
    args = p.parse_args(argv)
    if args.mode == "precision" and not (args.member and args.fields_json and args.out and args.sr_dir):
        p.error("precision mode needs --member, --fields-json, --out and --sr-dir")
    if args.mode == "speed":
        for v in args.variants.split(","):
            if v.strip():
                try:
                    parse_variant(v.strip())
                except ValueError as exc:
                    p.error(str(exc))
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.mode == "compare":
        return _compare(args)
    meta = {"host": platform.node(), "gpu": _gpu_name(), "cpus": os.cpu_count(),
            "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"), "tf": tf.__version__,
            "mode": args.mode}
    print(json.dumps(meta), flush=True)
    if args.mode == "precision":
        return _precision(args, meta)
    return _speed(args, meta)


if __name__ == "__main__":
    raise SystemExit(main())
