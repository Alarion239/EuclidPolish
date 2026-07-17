#!/usr/bin/env python
"""Benchmark: can the forward model (PSF pick → rotate → convolve → rebin →
noise → artifacts → saturation) run ON THE FLY inside the training input
pipeline instead of being baked into the dirty TFRecords?

Times every stage of :class:`ObservationSimulator` on REAL data (clean_train
records + the extracted band ePSF sets), then projects what feeding a GPU
training step would need. Two candidate designs are measured:

  A. **full-field** — forward-model the whole 510² field per visit, then crop
     (exactly what generation does; one example = one full process() call);
  B. **crop-local** — cut an (crop + 2·pad)² HR patch, convolve with the PSF
     kernel centre-cropped to (2·pad+1)², rebin + noise the crop only. Cheaper
     by ~the area ratio; the pad trades wing accuracy for speed, so the
     truncation error is reported against the VIS read noise.

Written for a FASRC LOGIN node (CPU-only, netscratch I/O): every stage prints
wall-clock AND process CPU time — a big gap between them = contention from
other users, so trust the CPU column for compute stages and treat the read
throughput as a lower bound (worse than a compute node would see).

    python scripts/benchmark_forward_onthefly.py                # real records
    python scripts/benchmark_forward_onthefly.py --synthetic    # no records
    python scripts/benchmark_forward_onthefly.py --n-fields 8 --gpu-step-ms 75
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # login node: CPU only

from euclid_polish.config import Config  # noqa: E402
from euclid_polish.image import Image  # noqa: E402
from euclid_polish.image.tfio import tfrecord_path  # noqa: E402
from euclid_polish.psf import PSF  # noqa: E402
from euclid_polish.psf.psf_library import load_all_band_psf_sets  # noqa: E402
from euclid_polish.psf.psf_set import PSFSample  # noqa: E402
from euclid_polish.sky.observation.noise import apply_archive_noise  # noqa: E402
from euclid_polish.sky.observation.observation_simulator import (  # noqa: E402
    ObservationSimulator,
    ObservationSimulatorConfig,
)
from euclid_polish.sky.observation.saturation import (  # noqa: E402
    apply_saturation_masking,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records-dir", default=Config.RECORDS_DIR_V2,
                   help="Dir holding clean_train TFRecords (default: "
                        "Config.RECORDS_DIR_V2, i.e. the FASRC data dir when "
                        "EUCLID_POLISH_DATA_DIR is exported).")
    p.add_argument("--psf-dir", default=Config.EUCLID_PSF_DIR)
    p.add_argument("--n-fields", type=int, default=6,
                   help="Fields to read + benchmark on (median over them).")
    p.add_argument("--repeats", type=int, default=3,
                   help="Repeats per field per stage (median taken).")
    p.add_argument("--gpu-step-ms", type=float, default=75.0,
                   help="Assumed GPU train-step time for one batch — the "
                        "budget the input pipeline must stay under. Read the "
                        "real value off a recent ensemble-train job "
                        "(~1000-step wall time).")
    p.add_argument("--batch-size", type=int, default=Config.DEFAULT_BATCH_SIZE)
    p.add_argument("--pads", default="16,32,64,128",
                   help="Crop-local halo half-widths (HR px) to test; the PSF "
                        "kernel is centre-cropped to (2·pad+1)².")
    p.add_argument("--synthetic", action="store_true",
                   help="Skip the records and use random 510² fields "
                        "(smoke-test mode; read throughput not measured).")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# timing helpers
# --------------------------------------------------------------------------- #
def _bench(fn, repeats: int) -> tuple[float, float]:
    """Median (wall_ms, cpu_ms) of ``fn()`` over ``repeats`` runs."""
    walls, cpus = [], []
    for _ in range(repeats):
        w0, c0 = time.perf_counter(), time.process_time()
        fn()
        walls.append((time.perf_counter() - w0) * 1e3)
        cpus.append((time.process_time() - c0) * 1e3)
    return float(np.median(walls)), float(np.median(cpus))


def _row(label: str, wall: float, cpu: float, note: str = "") -> None:
    print(f"  {label:<38s} {wall:9.1f} ms wall {cpu:9.1f} ms cpu  {note}")


def _crop_kernel(psf: PSF, pad: int) -> PSF:
    """Centre-crop a kernel to (2·pad+1)² (or return it unchanged if smaller)."""
    k = np.asarray(psf.data, np.float32)
    side = 2 * int(pad) + 1
    if k.shape[0] <= side:
        return psf
    c = k.shape[0] // 2
    lo, hi = c - pad, c + pad + 1
    return PSF(data=k[lo:hi, lo:hi].copy(), pixel_scale=psf.pixel_scale)


# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    pads = [int(x) for x in args.pads.split(",") if x.strip()]
    hr_crop = int(Config.DEFAULT_HR_CROP_SIZE)          # 96 → LR 48
    bands = list(Config.LR_INPUT_BAND_NAMES)
    ncpu = os.cpu_count() or 1
    print("=== on-the-fly forward-model benchmark ===")
    print(f"host CPUs: {ncpu} · batch {args.batch_size} · HR crop {hr_crop}px "
          f"· GPU step budget {args.gpu_step_ms:.0f} ms/batch")
    print("(login node: wall ≫ cpu on a stage ⇒ contention; trust the cpu "
          "column for compute, read throughput is a lower bound)\n")

    # ---- data ------------------------------------------------------------ #
    if args.synthetic:
        rng0 = np.random.default_rng(0)
        fields = [Image(data=np.abs(rng0.normal(10, 5, (510, 510, 4))
                                    ).astype(np.float32),
                        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                        band_names=Config.LR_INPUT_BAND_NAMES,
                        is_clean=True, index=i)
                  for i in range(args.n_fields)]
        print(f"synthetic fields: {len(fields)} × {fields[0].data.shape}\n")
    else:
        clean_path = tfrecord_path(args.records_dir, "clean_train")
        if not os.path.exists(clean_path):
            print(f"✗ no clean_train records at {clean_path} — pass "
                  "--records-dir or use --synthetic")
            return 2
        from euclid_polish.image.tfio import read_images
        w0 = time.perf_counter()
        fields = read_images(clean_path, num_images=args.n_fields)
        read_s = time.perf_counter() - w0
        mb = sum(f.data.nbytes for f in fields) / 1e6
        print(f"data read: {len(fields)} fields ({mb:.0f} MB) in {read_s:.2f} s"
              f" → {len(fields) / read_s:.1f} fields/s, {mb / read_s:.0f} MB/s"
              f"  [{clean_path}]")
        if not fields:
            print("✗ zero records read")
            return 2
        print(f"field shape: {fields[0].data.shape} @ "
              f"{fields[0].pixel_scale_arcsec}\"/pix\n")

    # ---- PSFs ------------------------------------------------------------ #
    w0 = time.perf_counter()
    psf_sets = load_all_band_psf_sets(psf_dir=args.psf_dir,
                                      target_pixel_scale=Config.DEFAULT_PIXEL_SCALE)
    print(f"PSF sets loaded in {time.perf_counter() - w0:.2f} s (one-time):")
    for name in bands:
        ps = psf_sets[name]
        print(f"  {name}: {ps.n} kernel(s) {ps.shape} @ {ps.pixel_scale}\"/pix")
    print()

    fwd = ObservationSimulator(
        psf_sets_by_band=psf_sets,
        config=ObservationSimulatorConfig(add_noise=True, add_artifacts=True,
                                          add_saturation=True))
    rng = np.random.default_rng(1)
    field = fields[0]
    hr = np.asarray(field.data, np.float32)

    # ---- stage micro-benchmarks (full 510² field) ------------------------- #
    print("— per-stage, FULL field (one channel = VIS unless noted) —")
    sample = PSFSample(index=0, angle=137.0)            # force a rotation

    rotated: dict[str, PSF] = {}

    def _rotate_all():
        for name in bands:
            rotated[name] = psf_sets[name].apply_sample(sample)
    _row("PSF rotate (all 4 bands, order-3)", *_bench(_rotate_all, args.repeats))

    vis_psf = rotated["VIS"]
    conv_out: dict[str, np.ndarray] = {}

    def _conv_all():
        for name in bands:
            conv_out[name] = rotated[name].convolved_with(hr[..., bands.index(name)])
    _row("fftconvolve (all 4 bands, full field)", *_bench(_conv_all, args.repeats))

    reb: dict[str, np.ndarray] = {}

    def _rebin_all():
        for name in bands:
            reb[name] = ObservationSimulator.sum_rebin(conv_out[name], 2)
    _row("sum-rebin ×2 (all 4 bands)", *_bench(_rebin_all, args.repeats))

    def _noise_all():
        for name in bands:
            apply_archive_noise(
                reb[name], Config.get_band(name), rng, add_artifacts=False,
                resample_kernel=Config.NISP_RESAMPLE_KERNEL,
            )
    _row("MER noise (native NISP + resample, 4 bands)",
         *_bench(_noise_all, args.repeats))

    def _noise_art_all():
        for name in bands:
            apply_archive_noise(
                reb[name], Config.get_band(name), rng, add_artifacts=True,
                resample_kernel=Config.NISP_RESAMPLE_KERNEL,
            )
    _row("MER noise + artifacts (4 bands)",
         *_bench(_noise_art_all, args.repeats))

    lr_stack = np.stack([reb[n] for n in bands], axis=-1).astype(np.float32)

    def _sat():
        apply_saturation_masking(lr_stack.copy(), fwd._sat_model, rng,
                                 band_names=Config.LR_INPUT_BAND_NAMES)
    _row("saturation masking (stack)", *_bench(_sat, args.repeats))

    # end-to-end, median across fields
    walls, cpus = [], []
    for f in fields:
        w, c = _bench(lambda f=f: fwd.process(f, rng), 1)
        walls.append(w)
        cpus.append(c)
    full_wall, full_cpu = float(np.median(walls)), float(np.median(cpus))
    _row("FULL process() (4 bands, e2e)", full_wall, full_cpu,
         f"(median over {len(fields)} fields)")
    print()

    # ---- design A projection: full-field per example ---------------------- #
    batch = int(args.batch_size)
    print("— design A: full-field forward per training example —")
    per_batch = full_cpu * batch
    workers = per_batch / args.gpu_step_ms
    print(f"  {full_cpu:.0f} ms cpu/example → {per_batch / 1e3:.1f} s/batch({batch})"
          f" → needs ~{workers:.0f} parallel CPU workers to hide behind a "
          f"{args.gpu_step_ms:.0f} ms GPU step\n")

    # ---- design B: crop-local forward ------------------------------------- #
    print("— design B: crop-local forward (HR crop + pad halo, truncated "
          "kernel) —")
    # Reference: full-kernel full-field VIS convolutions to score each pad's
    # truncation error in the crop interior. The error is FIELD-DEPENDENT
    # (a bright star's wings dominate it), so it is scored on every loaded
    # field and reported as median + worst, in electrons vs the VIS read
    # noise.
    rn_vis = float(Config.get_band("VIS").read_noise_e)
    y0 = (hr.shape[0] - hr_crop) // 2 // 2 * 2          # block-aligned centre crop
    conv_refs = [vis_psf.convolved_with(np.asarray(f.data, np.float32)[..., 0])
                 [y0: y0 + hr_crop, y0: y0 + hr_crop] for f in fields]
    print(f"  {'pad':>4s} {'kernel':>8s} {'flux kept':>9s} "
          f"{'ms/example(cpu)':>16s} {'s/batch':>8s} {'workers':>8s}   "
          f"trunc RMS med/worst e⁻ (RN {rn_vis:.1f})")
    for pad in pads:
        patch = hr[y0 - pad: y0 + hr_crop + pad,
                   y0 - pad: y0 + hr_crop + pad, :]
        if patch.shape[0] != hr_crop + 2 * pad:
            print(f"  {pad:>4d}  (field too small for this pad — skipped)")
            continue
        kern = {n: _crop_kernel(rotated[n], pad) for n in bands}
        flux_kept = float(np.asarray(kern["VIS"].data, np.float64).sum()
                          / np.asarray(vis_psf.data, np.float64).sum())

        def _one_example(kern=kern, patch=patch, pad=pad):
            for j, name in enumerate(bands):
                c = kern[name].convolved_with(patch[..., j])
                core = c[pad: pad + hr_crop, pad: pad + hr_crop]
                lr = ObservationSimulator.sum_rebin(core, 2)
                apply_archive_noise(
                    lr, Config.get_band(name), rng, add_artifacts=True,
                    resample_kernel=Config.NISP_RESAMPLE_KERNEL,
                )

        wall, cpu = _bench(_one_example, args.repeats)
        # truncation accuracy per field: crop-local (truncated, renormalised)
        # vs the full-kernel full-field convolution over the same interior.
        rmss = []
        for f, c_ref in zip(fields, conv_refs, strict=True):
            fp = np.asarray(f.data, np.float32)[
                y0 - pad: y0 + hr_crop + pad, y0 - pad: y0 + hr_crop + pad, 0]
            c_loc = kern["VIS"].convolved_with(fp)[
                pad: pad + hr_crop, pad: pad + hr_crop]
            rmss.append(float(np.sqrt(np.mean((c_loc - c_ref) ** 2))))
        med, worst = float(np.median(rmss)), float(np.max(rmss))
        per_batch = cpu * batch
        workers = per_batch / args.gpu_step_ms
        print(f"  {pad:>4d} {2 * pad + 1:>7d}² {flux_kept:>8.4f} "
              f"{cpu:>13.1f} ms {per_batch / 1e3:>7.2f} s {workers:>8.1f}   "
              f"{med:.3f} / {worst:.3f} e⁻ "
              f"({med / rn_vis * 100:.0f}% / {worst / rn_vis * 100:.0f}% RN)")
    print()

    # ---- extras ------------------------------------------------------------ #
    # ---- design C: thread scaling of the SHIPPED training path ----------- #
    # tf.data parallelizes the numpy_function forward with THREADS, so the
    # scaling ceiling is set by how much of OnTheFlyForward releases the GIL
    # (FFT convolution mostly does; numpy random sampling mostly doesn't).
    # This measures aggregate throughput of the exact class training uses —
    # the answer to "will more allocated CPUs be utilized?".
    from concurrent.futures import ThreadPoolExecutor

    from euclid_polish.training.forward_onthefly import OnTheFlyForward

    print("— design C: OnTheFlyForward thread scaling (K=16 crops/field) —")
    fwd_hook = OnTheFlyForward(psf_sets, seed=0, crops_per_field=16)
    fld = np.asarray(fields[0].data, np.float32)
    fwd_hook.crops(fld)                                    # warm-up / traces
    base_rate = None
    print(f"  {'threads':>8s} {'fields/s':>9s} {'ex/s (K=16)':>12s} "
          f"{'speedup':>8s} {'ms/batch(16)':>13s}")
    for nthreads in (1, 2, 4, 8, 16):
        if nthreads > ncpu:
            break
        n_jobs = max(2 * nthreads, 6)
        w0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=nthreads) as ex:
            list(ex.map(lambda _i: fwd_hook.crops(fld), range(n_jobs)))
        dt = time.perf_counter() - w0
        rate = n_jobs / dt
        base_rate = base_rate or rate
        ex_s = rate * 16
        print(f"  {nthreads:>8d} {rate:>9.2f} {ex_s:>12.0f} "
              f"{rate / base_rate:>7.1f}x {16_000 / ex_s:>10.0f} ms")
    print("  (speedup plateaus where the GIL-bound stages — numpy random "
          "noise, python glue — saturate; allocate CPUs up to ~the plateau, "
          "then raise crops/field instead)\n")

    print("— notes —")
    print("  · flux kept < 1 and trunc RMS are the PHYSICS cost of the pad: "
        "wing flux the truncated kernel redistributes. Safe when RMS ≪ read noise.")
    print("  · rotation can be amortised: pre-rotate a pool of kernels per "
        "epoch (e.g. 64 rolls × clusters) and draw from it — the rotate row "
        "above then drops out of the per-example cost.")
    print("  · workers = CPU processes needed to hide the forward step behind "
        "the GPU step; the training nodes currently allocate 4 CPUs "
        "(2 effective under contention — see train-input-starvation).")
    print("  · a GPU-side variant (batched tf.nn.conv2d / FFT inside the "
        "input-consuming train step) pays ~0 CPU; this script bounds the "
        "CPU designs only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
