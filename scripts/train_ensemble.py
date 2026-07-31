#!/usr/bin/env python
"""Train ensemble members: add new ones, continue existing, or fork.

Members live in ``<base-dir>/member_NN/`` and train on the same train/validate
TFRecords, each with a distinct seed (recorded in its provenance).  A WebUI
submission uses a SLURM array and this process trains exactly one selected
member; direct CLI use without ``--array-task`` may still train sequentially.

Modes:
  --mode add       (default) create --count NEW members and train only them.
  --mode continue  train --members m,… either for --extra-steps MORE steps
                   each, or up to the absolute --target-steps checkpoint
                   (warm cosine restart: the schedule recomputes over the
                   new absolute total; warmup is already past).
  --mode fork      create --count new members initialized from
                   --fork-from's weights (--fork-track psnr|loss), step 0,
                   fresh optimizer + LR schedule.

New-member names normally arrive via --member-names (allocated from the
LOCAL registry at WebUI submit time so archived indices are never reused);
without it the script takes the next free on-disk indices.

    python -u scripts/train_ensemble.py --mode add --count 3 --steps 200000
    python -u scripts/train_ensemble.py --mode continue --members member_03 --extra-steps 50000
    python -u scripts/train_ensemble.py --mode continue --members member_03 --target-steps 200000
    python -u scripts/train_ensemble.py --mode fork --fork-from member_02 --count 2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config  # noqa: E402
from euclid_polish.ensemble import (  # noqa: E402
    EnsembleModel,
    MemberTrainSpec,
    member_is_starless,
)
from euclid_polish.image.tfio import tfrecord_path  # noqa: E402
from euclid_polish.observability import Reporter, ResourceSampler  # noqa: E402
from euclid_polish.training.forward_onthefly import (  # noqa: E402
    DEFAULT_CROPS_PER_FIELD,
    DEFAULT_ONTHEFLY_HR_CROP_SIZE,
)
from euclid_polish.training.inference import checkpoint_step  # noqa: E402
from euclid_polish.training.loss_names import LOSS_NAMES  # noqa: E402
from euclid_polish.training.staging import stage_records  # noqa: E402


def _default_base_dir() -> str:
    parent = os.path.dirname(Config.DEFAULT_CHECKPOINT_DIR.rstrip("/")) or "."
    return os.path.join(parent, "ensemble")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", default=None,
                   help="Ensemble root holding member_NN/ (default: "
                        "<ckpt parent>/ensemble).")
    p.add_argument("--records-dir", default=Config.RECORDS_DIR_V2)
    p.add_argument("--mode", choices=["add", "continue", "fork"],
                   default="add")
    p.add_argument("--count", type=int, default=None,
                   help="add/fork: how many new members "
                        "(default 5 for add, 1 for fork).")
    p.add_argument("--n-members", type=int, default=None,
                   help="DEPRECATED alias for --count (add mode).")
    p.add_argument("--member-names", default="",
                   help="Explicit new-member names (comma-separated), "
                        "allocated by the WebUI from the registry. Absent → "
                        "next free on-disk indices.")
    p.add_argument("--array-task", action="store_true",
                   help="Train only the member selected by "
                        "SLURM_ARRAY_TASK_ID. Requires explicit member names "
                        "for add/fork mode.")
    p.add_argument("--members", default="",
                   help="continue: comma-separated existing member names.")
    continue_steps = p.add_mutually_exclusive_group()
    continue_steps.add_argument(
        "--extra-steps", type=int, default=None,
        help="continue: train each member this many MORE steps "
             "(default: 50000 when neither continuation option is given).",
    )
    continue_steps.add_argument(
        "--target-steps", type=int, default=None,
        help="continue: train each member UP TO this absolute step; each "
             "member adds target-current, and members already there are "
             "successful no-ops.",
    )
    p.add_argument("--fork-from", default="",
                   help="fork: source member name (e.g. member_02).")
    p.add_argument("--fork-track", choices=["psnr", "loss"], default="psnr",
                   help="fork: which of the source's save-best checkpoints "
                        "seeds the new member's weights.")
    p.add_argument("--base-seed", type=int, default=-1,
                   help="Member i is seeded base_seed+i. -1 (default) draws a "
                        "fresh entropy base seed; the value used is recorded on "
                        "each member's provenance for replay.")
    p.add_argument("--steps", type=int, default=Config.DEFAULT_TRAIN_STEPS)
    # Diversity knobs — run-wide defaults; --member-spec overrides per member.
    p.add_argument("--loss", choices=list(LOSS_NAMES), default="l1",
                   help="Reconstruction loss on the asinh residuals: l1=MAE, "
                        "l2=RMSE, l3=root mean cube, mse=mean squared error. "
                        "Only MSE omits the outer p-th root. "
                        "(berhu was tried and deprecated.)")
    p.add_argument("--noise-aug", type=float, default=0.0,
                   help="Extra Gaussian LR noise in READ-NOISE units (VIS "
                        "3.6 e⁻, NISP 6.1 e⁻ per unit), drawn fresh per crop "
                        "on top of the baked-in realization. 0 = off. "
                        "Decorrelates members; ~0.5–1.0 is a sane range.")
    p.add_argument("--bootstrap", type=float, default=0.0,
                   help="Train each member on this fraction (0<f<1) of the "
                        "training fields — a deterministic pseudo-random "
                        "subset keyed by the member's seed, stable across "
                        "epochs/resumes. 0/1 = off (all fields).")
    p.add_argument("--forward-onthefly", type=int, default=0,
                   help="1/0 — train on the LIVE forward model instead of "
                        "the baked dirty records: each visit of a clean "
                        "field re-draws PSF (from the member's bagged "
                        "pre-rotated pool) + noise + artifacts on the FULL "
                        "field (out-of-crop stars contaminate crops exactly "
                        "as in a real exposure), then cuts crops-per-field "
                        "training crops. Needs clean_train records; dirty "
                        "records are ignored for training (validation "
                        "still uses them).")
    p.add_argument("--psf-subset", type=int, default=0,
                   help="PSF bagging: each member draws its kernels from "
                        "this many source clusters of the pre-rotated pool "
                        "(subset keyed by the member's seed; different "
                        "members → different instrument-response bags). "
                        "0 = default (64). Only with --forward-onthefly.")
    p.add_argument("--crops-per-field", type=int,
                   default=DEFAULT_CROPS_PER_FIELD,
                   help="On-the-fly mode: random crops cut per forward-"
                        "modelled field before the crop-level shuffle. "
                        "Default: 8.")
    p.add_argument("--hr-crop-size", type=int,
                   default=DEFAULT_ONTHEFLY_HR_CROP_SIZE,
                   help="On-the-fly mode: HR side of each aligned training "
                        "example. Default: 256 (128 LR), the nearest "
                        "scale-aligned half-side crop of a 510² field. Must "
                        "be divisible by the 2x scale.")
    p.add_argument("--psf-warp-prob", type=float,
                   default=Config.TRAIN_PSF_WARP_PROB,
                   help="On-the-fly mode: probability that the shared "
                        "observation PSF receives an elastic warp. Default 1; "
                        "0 disables.")
    p.add_argument("--psf-warp-alpha-max", type=float,
                   default=Config.TRAIN_PSF_WARP_ALPHA_MAX,
                   help="On-the-fly mode: upper bound for observation-PSF "
                        "elastic-warp alpha, sampled uniformly from [0,max] "
                        "per exposure. Default 20, matching polish-pub.")
    p.add_argument("--psf-warp-sigma", type=float,
                   default=Config.TRAIN_PSF_WARP_SIGMA,
                   help="On-the-fly mode: Gaussian smoothing scale of the "
                        "warp displacement field in HR pixels. Default 3, "
                        "matching polish-pub.")
    p.add_argument("--saturation-mask-prob", type=float,
                   default=Config.TRAIN_SATURATION_MASK_PROB,
                   help="On-the-fly mode: probability that an above-well "
                        "source receives a dark rectangular core. Default "
                        "0.2; maximum 0.5 so intact bright stars remain "
                        "common.")
    p.add_argument("--starless", type=int, default=1,
                   help="1/0 — STARLESS member: the forward injects a fresh "
                        "star realization each visit and the target is the "
                        "starless scene, so the model learns to ERASE stars "
                        "(scored on lr↔clean). 0 = starfull: keep the scene's "
                        "stars, reconstruct them (scored on lr↔hr). Recorded "
                        "in origin.json; drives the /ensemble eval mode. "
                        "Forks always inherit their source member's regime.")
    p.add_argument("--star-prior-json", default="",
                   help="JSON-encoded activated point-source prior for live "
                        "on-the-fly star magnitudes and colours.")
    p.add_argument("--asinh-knee", type=float, default=None,
                   help="Per-member asinh stretch knee in ELECTRONS — the "
                        "linear/log crossover of asinh(x/knee) the network "
                        "sees on both input and target. Default (unset) = the "
                        "per-band config 100 e⁻ (what every existing member "
                        "trained under). A different knee reweights the loss "
                        "across brightness (a lower knee compresses bright "
                        "cores harder, a higher one keeps them more linear) → "
                        "a genuinely different estimator → ensemble diversity, "
                        "with no TFRecord regen (it's an on-the-fly "
                        "normalization). ADD members only; continue/fork "
                        "inherit the existing/source member's knee.")
    p.add_argument("--icnr", action="store_true",
                   help="ICNR-initialise the sub-pixel (pixel-shuffle) convs "
                        "so the upsampler starts as a checkerboard-free "
                        "nearest-neighbour resize — kills the background "
                        "speckle / peristellar hot-dots that L2 otherwise "
                        "spends training budget learning to cancel. Init-only "
                        "(ADD members only; fork/continue load weights and "
                        "ignore it); old checkpoints restore identically.")
    p.add_argument("--member-spec", default="",
                   help='Per-member override JSON, applied positionally: '
                        '\'[{"loss":"l2","noise_aug":1.0,"bootstrap":0.7,'
                        '"num_res_blocks":16}, {}, …]\'. Members beyond the '
                        "list length use the run-wide flags. Keys: loss, "
                        "noise_aug, bootstrap, num_res_blocks (add mode), "
                        "seed, forward_onthefly, psf_subset, "
                        "crops_per_field, psf_warp_prob, "
                        "psf_warp_alpha_max, psf_warp_sigma, "
                        "saturation_mask_prob, icnr.")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Training examples per optimizer update. Default 4 "
                        "for 256² HR crops (about the pixels of one 510² "
                        "field while mixing parent exposures).")
    p.add_argument("--evaluate-every", type=int, default=Config.DEFAULT_EVALUATE_EVERY)
    p.add_argument("--num-res-blocks", type=int, default=Config.DEFAULT_NUM_RES_BLOCKS)
    # LR schedule (warmup → cosine) + reduce-LR-on-plateau guard. Defaults from
    # Config; the WebUI /config page injects these on FASRC submission.
    p.add_argument("--lr-peak", type=float, default=Config.LR_PEAK)
    p.add_argument("--lr-final", type=float, default=Config.LR_FINAL)
    p.add_argument("--lr-warmup-steps", type=int, default=Config.LR_WARMUP_STEPS)
    p.add_argument("--plateau-lr-enabled", type=int,
                   default=int(Config.PLATEAU_LR_ENABLED),
                   help="1/0 — reduce LR (+ rollback) when the val metric "
                        "stalls. Applies to L1 members ONLY (its job is "
                        "escaping L1's degenerate skip-only basin); forced off "
                        "for L2/L3/BerHu, which have no such basin.")
    p.add_argument("--plateau-lr-factor", type=float,
                   default=Config.PLATEAU_LR_FACTOR)
    p.add_argument("--plateau-lr-patience", type=int,
                   default=Config.PLATEAU_LR_PATIENCE)
    p.add_argument("--plateau-lr-min-delta", type=float,
                   default=Config.PLATEAU_LR_MIN_DELTA)
    p.add_argument("--plateau-lr-min-delta-rel", type=float,
                   default=Config.PLATEAU_LR_MIN_DELTA_REL,
                   help="Relative improvement threshold (fraction of |best|); "
                        "effective threshold = max(min_delta, |best|·rel).")
    p.add_argument("--plateau-lr-cooldown", type=int,
                   default=Config.PLATEAU_LR_COOLDOWN)
    p.add_argument("--plateau-lr-min-lr", type=float,
                   default=Config.PLATEAU_LR_MIN_LR)
    p.add_argument("--plateau-lr-metric", default=Config.PLATEAU_LR_METRIC,
                   choices=["combined_loss", "psnr_stretched"])
    p.add_argument("--plateau-rollback-min-gap", type=float,
                   default=Config.PLATEAU_ROLLBACK_MIN_GAP,
                   help="A plateau firing this far (dB) below the run's best "
                        "restores the best-PSNR checkpoint before cooling "
                        "(degenerate-basin escape) instead of reducing the "
                        "LR in place.")
    p.add_argument("--plateau-lr-recovery", type=int,
                   default=int(Config.PLATEAU_LR_RECOVERY),
                   help="1/0 — undo one plateau LR cut per NEW best PSNR "
                        "(cuts are provisional; the LR climbs back toward "
                        "the schedule value once training moves again).")
    return p.parse_args(argv)


def _has_ckpt(d: str) -> bool:
    return (os.path.isfile(os.path.join(d, "checkpoint"))
            or bool(glob.glob(os.path.join(d, "*.index"))))


def _next_free_names(base: str, k: int, *, start_past: int = -1) -> list[str]:
    used = [int(tail)
            for tail in (os.path.basename(d).removeprefix("member_")
                         for d in glob.glob(os.path.join(base, "member_*")))
            if tail.isdigit()]
    start = max([start_past, *used]) + 1
    return [f"member_{i:02d}" for i in range(start, start + k)]


def _member_index(name: str) -> int | None:
    tail = str(name).removeprefix("member_")
    return int(tail) if tail.isdigit() else None


def _claim_fresh_name(base: str, wanted: str | None, *,
                      reserved: list[str]) -> str:
    """Atomically reserve ``wanted``, shifting beyond sibling reservations.

    Several queued jobs may receive the same submit-time allocation.  The
    directory creation is the cross-process claim: if another task/job won
    it, keep advancing until this task claims a genuinely unused member.
    """
    if wanted:
        try:
            os.makedirs(os.path.join(base, wanted), exist_ok=False)
            return wanted
        except FileExistsError:
            pass

    reserved_indices = [i for i in map(_member_index, reserved)
                        if i is not None]
    start_past = max(reserved_indices, default=-1)
    while True:
        shifted = _next_free_names(base, 1, start_past=start_past)[0]
        try:
            os.makedirs(os.path.join(base, shifted), exist_ok=False)
        except FileExistsError:
            # Another array task claimed the same candidate between the scan
            # and mkdir. Re-scan and try the next index.
            continue
        if wanted:
            print(f"  ⚠ {wanted} already exists on disk — shifted to "
                  f"{shifted}")
        return shifted


def _fresh_names(args, base: str, k: int) -> list[str]:
    """Names for members this run CREATES: the submitted allocation, with a
    collision shift past any name that already exists on this filesystem
    (two queued jobs can allocate the same index at submit time). Each name is
    claimed atomically so concurrent tasks cannot share a directory."""
    wanted = [n.strip() for n in args.member_names.split(",") if n.strip()]
    out: list[str] = []
    reserved = wanted[:k]
    for n in reserved:
        out.append(_claim_fresh_name(base, n, reserved=reserved))
    while len(out) < k:
        out.append(_claim_fresh_name(base, None, reserved=reserved))
    return out


def _array_index(args, count: int) -> int | None:
    """Validate and return this process's array index, if array mode is on."""
    if not args.array_task:
        return None
    raw = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    try:
        index = int(raw)
    except ValueError as e:
        print("✗ --array-task requires integer SLURM_ARRAY_TASK_ID")
        raise SystemExit(2) from e
    if not 0 <= index < count:
        print(f"✗ SLURM_ARRAY_TASK_ID={index} is outside 0..{count - 1}")
        raise SystemExit(2)
    return index


def _member_overrides(args, k: int) -> list[dict]:
    """``--member-spec`` JSON → one (possibly empty) override dict per member.

    Applied positionally; a short list pads with ``{}`` (run-wide flags), a
    long one is truncated. Unknown keys are rejected loudly — a typo silently
    training 5 members with default knobs would be an expensive no-op.
    """
    if not args.member_spec.strip():
        return [{} for _ in range(k)]
    try:
        spec = json.loads(args.member_spec)
    except json.JSONDecodeError as e:
        print(f"✗ --member-spec is not valid JSON: {e}")
        raise SystemExit(2) from e
    if not isinstance(spec, list) or not all(isinstance(o, dict) for o in spec):
        print("✗ --member-spec must be a JSON LIST of objects (one per member)")
        raise SystemExit(2)
    allowed = {"loss", "noise_aug", "bootstrap", "num_res_blocks", "seed",
               "forward_onthefly", "psf_subset", "crops_per_field",
               "hr_crop_size", "icnr",
               "psf_warp_prob", "psf_warp_alpha_max", "psf_warp_sigma",
               "saturation_mask_prob",
               "starless", "asinh_knee"}
    for i, o in enumerate(spec):
        bad = set(o) - allowed
        if bad:
            print(f"✗ --member-spec[{i}]: unknown key(s) {sorted(bad)} "
                  f"(allowed: {sorted(allowed)})")
            raise SystemExit(2)
        if o.get("loss") is not None and o["loss"] not in LOSS_NAMES:
            print(f"✗ --member-spec[{i}]: loss must be one of "
                  f"{list(LOSS_NAMES)}, got {o['loss']!r}")
            raise SystemExit(2)
    return [dict(spec[i]) if i < len(spec) else {} for i in range(k)]


def _diversity_kwargs(args, over: dict) -> dict:
    """The spec kwargs shared by every mode: run-wide flag, member override."""
    boot = float(over.get("bootstrap", args.bootstrap) or 0.0)
    subset = int(over.get("psf_subset", args.psf_subset) or 0)
    knee = over.get("asinh_knee", args.asinh_knee)
    saturation_mask_prob = float(over.get(
        "saturation_mask_prob", args.saturation_mask_prob,
    ))
    if not 0.0 <= saturation_mask_prob <= \
            Config.TRAIN_SATURATION_MASK_PROB_MAX:
        print("✗ saturation_mask_prob must be in [0, "
              f"{Config.TRAIN_SATURATION_MASK_PROB_MAX:g}]")
        raise SystemExit(2)
    return {"loss_norm": str(over.get("loss", args.loss)),
            "asinh_knee": (float(knee) if knee not in (None, "", 0) else None),
            "noise_aug": float(over.get("noise_aug", args.noise_aug)),
            "bootstrap": boot if 0.0 < boot < 1.0 else None,
            "forward_onthefly": bool(over.get("forward_onthefly",
                                              args.forward_onthefly)),
            "psf_subset": subset if subset > 0 else None,
            "crops_per_field": int(over.get(
                "crops_per_field", args.crops_per_field,
            ) or DEFAULT_CROPS_PER_FIELD),
            "hr_crop_size": int(over.get("hr_crop_size",
                                          args.hr_crop_size)),
            "psf_warp_prob": float(over.get("psf_warp_prob",
                                              args.psf_warp_prob)),
            "psf_warp_alpha_max": float(over.get("psf_warp_alpha_max",
                                                   args.psf_warp_alpha_max)),
            "psf_warp_sigma": float(over.get("psf_warp_sigma",
                                               args.psf_warp_sigma)),
            "saturation_mask_prob": saturation_mask_prob,
            "icnr": bool(over.get("icnr", args.icnr)),
            "starless": bool(over.get("starless", args.starless))}


def build_specs(args, base: str) -> list[MemberTrainSpec]:
    """CLI args → the run's :class:`MemberTrainSpec` list (mode-dispatched)."""
    base_seed = (int.from_bytes(os.urandom(4), "little")
                 if args.base_seed < 0 else int(args.base_seed))

    if args.mode == "continue":
        names = [n.strip() for n in args.members.split(",") if n.strip()]
        if not names:
            print("✗ --mode continue needs --members")
            raise SystemExit(2)
        target = (int(args.target_steps)
                  if args.target_steps is not None else None)
        extra = (50_000 if args.extra_steps is None
                 else int(args.extra_steps))
        if target is not None and target <= 0:
            print("✗ --target-steps must be positive")
            raise SystemExit(2)
        if target is None and extra <= 0:
            print("✗ --extra-steps must be positive")
            raise SystemExit(2)
        array_index = _array_index(args, len(names))
        all_names = names
        overrides = _member_overrides(args, len(all_names))
        selected = range(len(all_names)) if array_index is None else [array_index]
        specs = []
        for i in selected:
            name = all_names[i]
            d = os.path.join(base, name)
            cur = checkpoint_step(d) if _has_ckpt(d) else None
            if cur is None:
                print(f"✗ {name}: no checkpoint to continue from in {d}")
                raise SystemExit(2)
            if target is not None and cur >= target:
                print(f"  ✓ {name}: already at step {cur:,} "
                      f"(target {target:,}); nothing to train")
                continue
            run_steps = target - cur if target is not None else extra
            absolute_target = target if target is not None else cur + extra
            specs.append(MemberTrainSpec(
                name=name, seed=int(overrides[i].get("seed", base_seed + i)),
                op="continue",
                target_steps=absolute_target,
                run_steps=run_steps,
                **_diversity_kwargs(args, overrides[i])))
        return specs

    k = int(args.count or args.n_members or (1 if args.mode == "fork" else 5))
    array_index = _array_index(args, k)
    init_from = forked_from = None
    fork_starless = None
    if args.mode == "fork":
        if not args.fork_from:
            print("✗ --mode fork needs --fork-from")
            raise SystemExit(2)
        source_member = os.path.join(base, args.fork_from)
        src = source_member
        if args.fork_track == "loss":
            src = os.path.join(src, "loss_best")
        if not _has_ckpt(src):
            print(f"✗ fork source has no checkpoint: {src}")
            raise SystemExit(2)
        init_from = src
        forked_from = f"{args.fork_from}·{args.fork_track}"
        # Regime is a property of the task the source model learned, just like
        # its architecture and input normalization. A fork may change its loss
        # or training distribution, but must not silently turn a starfull model
        # into a star eraser (or vice versa) because the form default differed.
        fork_starless = member_is_starless(source_member)
    if array_index is None:
        names_with_indices = list(enumerate(_fresh_names(args, base, k)))
    else:
        wanted = [n.strip() for n in args.member_names.split(",") if n.strip()]
        if len(wanted) != k or len(set(wanted)) != k:
            print("✗ array add/fork requires one unique --member-name per task")
            raise SystemExit(2)
        name = _claim_fresh_name(
            base, wanted[array_index], reserved=wanted,
        )
        names_with_indices = [(array_index, name)]
    overrides = _member_overrides(args, k)
    # Depth applies to ADD only: a fork inherits its source's depth (weights
    # are copied verbatim), and continue is dictated by the existing ckpt.
    specs = []
    for i, n in names_with_indices:
        over = overrides[i]
        blocks = (int(over.get("num_res_blocks", args.num_res_blocks))
                  if args.mode == "add" else None)
        diversity = _diversity_kwargs(args, over)
        if fork_starless is not None:
            diversity["starless"] = fork_starless
        specs.append(MemberTrainSpec(
            name=n, seed=int(over.get("seed", base_seed + i)), op=args.mode,
            target_steps=int(args.steps), run_steps=int(args.steps),
            init_from=init_from, forked_from=forked_from,
            num_res_blocks=blocks,
            **diversity))
    return specs


def required_record_names(specs) -> list[str]:
    """The TFRecord basenames this run actually reads.

    ``clean_train`` is the training target (and, on-the-fly, the scene the
    live forward model consumes); the validate pair feeds the metric stream
    for every member. ``dirty_train`` is only read by RECORD-mode members —
    an ``--onthefly-train`` generation deliberately doesn't produce it (the
    train split is clean-only), so an all-on-the-fly run must not demand it.
    """
    names = ["clean_train", "dirty_validate", "clean_validate"]
    if any(not s.forward_onthefly for s in specs):
        names.insert(0, "dirty_train")
    return names


def main() -> int:
    args = parse_args()
    base = args.base_dir or _default_base_dir()
    specs = build_specs(args, base)
    if not specs:
        print("✓ selected members are already at or beyond the requested "
              "target; nothing to train")
        return 0
    lr = tfrecord_path(args.records_dir, "dirty_train")
    hr = tfrecord_path(args.records_dir, "clean_train")
    needed = required_record_names(specs)
    missing = [n for n in needed
               if not os.path.exists(tfrecord_path(args.records_dir, n))]
    if missing:
        print(f"✗ training records missing in {args.records_dir}: "
              + ", ".join(missing) + ". Generate them first."
              + ("" if "dirty_train" in needed else
                 " (dirty_train is NOT required — every member trains "
                 "on-the-fly.)"))
        return 2
    label = {
        "add": f"add {len(specs)} member(s)",
        "continue": "continue " + ",".join(s.name for s in specs),
        "fork": f"fork {args.fork_from}·{args.fork_track} → "
                + ",".join(s.name for s in specs),
    }[args.mode]
    print(f"Ensemble training ({label}) → {base}")
    for s in specs:
        knobs = f" loss={s.loss_norm}"
        if s.asinh_knee is not None:
            knobs += f" asinh_knee={s.asinh_knee:g}e"
        if s.noise_aug:
            knobs += f" noise_aug={s.noise_aug:g}"
        if s.bootstrap:
            knobs += f" bootstrap={s.bootstrap:g}"
        if s.forward_onthefly:
            knobs += (f" forward=onthefly(psf_subset={s.psf_subset or 'dflt'},"
                      f" {s.crops_per_field} crops/field, "
                      f"warp=p{s.psf_warp_prob:g}/a{s.psf_warp_alpha_max:g}/"
                      f"s{s.psf_warp_sigma:g} · "
                      f"dark-core=p{s.saturation_mask_prob:g})")
        if s.icnr:
            knobs += " icnr" + ("" if s.op == "add"
                                else "(ignored: not an add)")
        print(f"  · {s.name}: seed={s.seed} target_steps={s.target_steps}"
              + knobs
              + (f" init_from={s.init_from}" if s.init_from else ""))

    # Structured progress for the WebUI (no terminal for tqdm under SLURM).
    # One cumulative bar across all specs' run_steps.
    reporter = Reporter.from_env()
    reporter.set_stage(f"ensemble: {label}")
    # Sample CPU + GPU utilisation every 10 s into the events stream so the job
    # log records gpu_util_mean/peak (the "GPU util" column). Daemon thread;
    # stopped when this member (or direct-CLI member sequence) is done.
    sampler = ResourceSampler(reporter).start()

    # Stage the hot train + validate records onto node-local scratch (fast NVMe,
    # no shared-netscratch contention — the input-pipeline stall behind the
    # 48s-vs-100s/1000-step variance). Best-effort: falls back to the original
    # dir if local scratch is missing/too small.
    def _stage_log(m: str) -> None:
        print(m)
        if m.lstrip().startswith("⚠"):      # surface staging fallbacks to the WebUI
            reporter.warn(m.strip())

    records_dir = stage_records(
        args.records_dir,
        needed,
        on_log=_stage_log,
    )
    staged = records_dir != args.records_dir
    lr = tfrecord_path(records_dir, "dirty_train")
    hr = tfrecord_path(records_dir, "clean_train")

    total = sum(max(1, s.run_steps) for s in specs)
    done_specs = [0]                            # specs finished so far

    def _done_before() -> int:
        return sum(max(1, s.run_steps) for s in specs[:done_specs[0]])

    def _spec_progress(s_local: int) -> int:
        # ``s_local`` is the member-local ABSOLUTE step from the trainer; a
        # continue spec only contributes its extra steps to the bar.
        spec = specs[min(done_specs[0], len(specs) - 1)]
        return _done_before() + max(0, s_local
                                    - (spec.target_steps - spec.run_steps))

    def _step_cb(s: int, _t: int) -> None:
        spec = specs[min(done_specs[0], len(specs) - 1)]
        reporter.set_step(_spec_progress(s), total,
                          f"{spec.name} ({done_specs[0] + 1}/{len(specs)}) "
                          f"· step {s}")

    def _eval_cb(row: dict) -> None:
        # Forward per-member eval metrics to the WebUI's structured stream as
        # ONE cumulative curve: offset by the specs already finished so the
        # validation history is continuous across members (the trainer only
        # knows its own member-local step). Tag the member for the UI.
        cum = dict(row)
        if cum.get("step") is not None:
            cum["step"] = _spec_progress(int(cum["step"]))
        cum["total"] = total
        cum["member"] = done_specs[0] + 1
        reporter.metric(cum)

    def _warn_cb(msg: str) -> None:
        # Surface restore/resume notices, gradient-spike rollbacks and LR
        # halvings (which otherwise only reached stdout) to the WebUI.
        spec = specs[min(done_specs[0], len(specs) - 1)]
        reporter.warn(f"{spec.name}: {msg}")

    def _on_member(i: int, n: int, _m) -> None:
        done_specs[0] = i                       # ``i`` is 1-indexed (specs done)
        print(f"\n=== {specs[i - 1].name} done ({i}/{n}) ===\n")
        reporter.set_step(_done_before(), total, f"{specs[i - 1].name} done")

    # Build the training handle WITHOUT discovering existing members:
    # train_members() resets self._models on entry and only constructs the
    # members it is told to train, so restoring every active checkpoint here
    # (the ctor's default) is pure waste — tens of GPU checkpoint loads that
    # are immediately discarded. This handle is used only to train.
    ens = EnsembleModel(base, scale=Config.DEFAULT_REBIN_FACTOR,
                        num_res_blocks=args.num_res_blocks, _models=[])
    star_prior_payload = None
    if args.star_prior_json.strip():
        try:
            star_prior_payload = json.loads(args.star_prior_json)
        except json.JSONDecodeError as exc:
            print(f"✗ --star-prior-json is not valid JSON: {exc}")
            return 2
        if not isinstance(star_prior_payload, dict):
            print("✗ --star-prior-json must contain a JSON object")
            return 2
    try:
        ens.train_members(
            lr, hr, specs,
            batch_size=args.batch_size,
            evaluate_every=args.evaluate_every,
            step_callback=_step_cb, eval_callback=_eval_cb, warn_callback=_warn_cb,
            on_member=_on_member,
            # LR schedule + plateau guard (forwarded to each member's Model.train).
            lr_peak=args.lr_peak, lr_final=args.lr_final,
            lr_warmup_steps=args.lr_warmup_steps,
            plateau_lr_enabled=bool(args.plateau_lr_enabled),
            plateau_lr_factor=args.plateau_lr_factor,
            plateau_lr_patience=args.plateau_lr_patience,
            plateau_lr_min_delta=args.plateau_lr_min_delta,
            plateau_lr_min_delta_rel=args.plateau_lr_min_delta_rel,
            plateau_lr_cooldown=args.plateau_lr_cooldown,
            plateau_lr_min_lr=args.plateau_lr_min_lr,
            plateau_lr_metric=args.plateau_lr_metric,
            plateau_rollback_min_gap=args.plateau_rollback_min_gap,
            plateau_lr_recovery=bool(args.plateau_lr_recovery),
            star_prior_payload=star_prior_payload,
        )

        reporter.set_step(total, total, "ensemble training complete")
    finally:
        sampler.stop()
        if staged:                          # remove the node-local staged copy
            shutil.rmtree(records_dir, ignore_errors=True)
    print("\n✓ Ensemble training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
