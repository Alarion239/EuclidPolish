"""Fit the spatial gating combiner and compare it with the current combiner.

Fits on the cached STARFULL validate member cubes (85 fields train, 15 held
out), optionally adding blackout-augmented copies of the training fields
(one extra member-inference pass, cached next to the cubes). ``--compare``
scores every method on the cached test cubes plus blackout-augmented test
fields and writes a JSON report and comparison figures.

    python scripts/fit_spatial_gate.py fit --out-name spatial_gate_trial
    python scripts/fit_spatial_gate.py compare --gates spatial_gate_combiner spatial_gate_trial

The scoring and fitting live in :mod:`euclid_polish.eval.spatial_gate_compare`
(the web console's Combiners tab runs the same code as local jobs). A fit
never writes the production gate: promote a variant from the console.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from euclid_polish.config import Config  # noqa: E402
from euclid_polish.ensemble_registry import default_ensemble_dir  # noqa: E402
from euclid_polish.eval.spatial_gate import MIX_LINEAR, MIX_SPACES, band_scales  # noqa: E402
from euclid_polish.eval.spatial_gate_compare import (  # noqa: E402
    PRODUCTION_DIR,
    best_member,
    fit_gate_variant,
    format_report,
    hole_masks,
    load_fit_fields,
    parse_loss_knees,
    run_compare,
)
from euclid_polish.eval.spatial_gate_fit import LazyMemberRunner  # noqa: E402
from euclid_polish.web.helpers.ensemble_viz import _eval_records_fingerprint  # noqa: E402
from euclid_polish.web.helpers.paths import _sky_records_local_dir  # noqa: E402

BANDS = tuple(Config.HR_TARGET_BAND_NAMES)


def _regime_dir() -> str:
    return os.path.abspath(os.path.join(Config.VIS_DIR, "ensemble", "starfull"))


def cmd_fit(args) -> None:
    # A fit can be stopped at any time: the best gate so far is already saved
    # (checkpointed by fit_gate_variant), and exiting on SIGTERM instead of
    # dying lets the temporary feature cache be removed at interpreter exit.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    regime = _regime_dir()
    out = os.path.join(regime, args.out_name)
    records = _sky_records_local_dir()
    fields, labels = load_fit_fields(os.path.join(regime, "cubes_validate"), records)
    runner = LazyMemberRunner(default_ensemble_dir(), starless=False, labels=labels)
    if args.members:
        print(f"pruned gate over members {args.members}")
    fit_gate_variant(
        fields, labels, out_dir=out, holdout=args.holdout, seed=args.seed,
        blackout_fields=args.blackout_fields, runner=runner,
        blackout_dir=os.path.join(regime, "cubes_validate_blackout"),
        # Same cache identity as the web fit, so neither invalidates the other's.
        source_fingerprint=str(_eval_records_fingerprint(records, "validate")),
        width=args.width, use_lr=args.lr_input, steps=args.steps, batch_size=args.batch,
        crop=args.crop, learning_rate=args.lr, eval_every=args.eval_every,
        members=[m for m in args.members.split(",") if m.strip()],
        loss_knees=parse_loss_knees("all" if args.knee_loss else "band"),
        mix_space=args.mix, extra_meta={"variant": args.out_name, "fitted_via": "script"},
        progress=lambda i, n, msg: print(f"  [{i}/{n}] {msg}", flush=True),
        log=lambda msg: print(msg, flush=True))
    print(f"saved {out}")


# --------------------------------------------------------------------------- #
# Comparison (the scoring lives in euclid_polish.eval.spatial_gate_compare)
# --------------------------------------------------------------------------- #

def _crop_figure(path, title, truth, panels, weights_dom, center, half=32):
    cy, cx = (int(np.clip(v, half, truth.shape[i] - half)) for i, v in enumerate(center))
    sl = (slice(cy - half, cy + half), slice(cx - half, cx + half))
    vmin, vmax = np.percentile(truth[sl], 1), np.percentile(truth[sl], 99.7)
    n = len(panels) + 1 + (weights_dom is not None)
    fig, ax = plt.subplots(2, n, figsize=(2.4 * n, 5.0))
    ax[0, 0].imshow(truth[sl], vmin=vmin, vmax=vmax, cmap="gray", origin="lower")
    ax[0, 0].set_title("truth (VIS asinh)", fontsize=8)
    ax[1, 0].axis("off")
    for j, (name, image) in enumerate(panels, 1):
        err = image[sl] - truth[sl]
        ax[0, j].imshow(image[sl], vmin=vmin, vmax=vmax, cmap="gray", origin="lower")
        ax[0, j].set_title(name, fontsize=8)
        ax[1, j].imshow(err, vmin=-0.15, vmax=0.15, cmap="RdBu_r", origin="lower")
        ax[1, j].set_title(f"error, rms {np.sqrt(np.mean(err ** 2)):.4f}", fontsize=8)
    if weights_dom is not None:
        dom, labels = weights_dom
        ax[0, -1].imshow(dom[sl], cmap="tab20", vmin=-0.5, vmax=19.5,
                         origin="lower", interpolation="nearest")
        ax[0, -1].set_title("gate: top member (VIS)", fontsize=8)
        present = np.unique(dom[sl])
        ax[1, -1].axis("off")
        ax[1, -1].text(0, 0.95, "\n".join(labels[i] for i in present[:12]),
                       fontsize=7, va="top", transform=ax[1, -1].transAxes)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title + "  (error panels ±0.15 asinh)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _figures(result, args) -> None:
    gates = [m for m in result.methods.values() if m.is_gate]
    if not gates:
        return
    os.makedirs(args.figures, exist_ok=True)
    gate = gates[-1]
    rbf = result.methods.get("rbf")
    labels = result.labels
    scales = band_scales(BANDS).astype(np.float32)
    best = best_member(result.report) or f"member:{labels[0]}"
    best_vis = labels.index(best.removeprefix("member:"))
    shown = 0
    for group, f, has_halo in result.figure_candidates:
        if shown >= args.max_figures:
            break
        if group == "natural" and not has_halo:
            continue
        members = f.members_e()
        truth = np.arcsinh(f.target_e / scales)[..., 0]
        lr = f.lr_e if gate.use_lr else None
        subset = members[gate.index]
        gate_out = np.arcsinh(gate.model.apply_field(subset, lr=lr) / scales)[..., 0]
        rbf_out = (np.arcsinh(rbf.model.apply_field(members[rbf.index]) / scales)[..., 0]
                   if rbf is not None else None)
        dom = gate.model.weights_field(subset, lr=lr)[..., 0].argmax(-1)
        if group == "blackout":
            holes = hole_masks(f, result.source_lr[f.index])[..., 0]
            if not holes.any():
                continue
            center = np.argwhere(holes).mean(0)
        else:
            center = np.unravel_index(np.argmax(truth), truth.shape)
        panels = [(f"best member {labels[best_vis]}",
                   np.arcsinh(members[best_vis, ..., 0] / scales[0])),
                  ("ensemble mean", np.arcsinh(members.mean(0)[..., 0] / scales[0]))]
        if rbf_out is not None:
            panels.append(("current combiner (RBF)", rbf_out))
        panels.append(("spatial gate", gate_out))
        path = os.path.join(args.figures, f"{group}_{f.index:05d}.png")
        _crop_figure(path, f"{group} test field {f.index}", truth, panels,
                     (dom, [labels[i] for i in gate.index]), center)
        shown += 1
        print(f"figure {path}")


def cmd_compare(args) -> None:
    regime = _regime_dir()
    records = _sky_records_local_dir()
    with open(os.path.join(regime, "cubes", "viz_index.json")) as handle:
        labels = json.load(handle)["member_labels"]
    runner = LazyMemberRunner(default_ensemble_dir(), starless=False, labels=labels)
    result = run_compare(
        regime_dir=regime, records_dir=records, gates=args.gates, runner=runner,
        blackout_fields=args.blackout_fields, seed=args.seed, knee=not args.no_knee,
        log=lambda msg: print(msg, flush=True))
    out = args.report or os.path.join(regime, "spatial_gate_comparison.json")
    with open(out, "w") as handle:
        json.dump(result.report, handle, indent=2)
    print(format_report(result.report))
    print(f"wrote {out}")
    if args.figures:
        _figures(result, args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--out-name", required=True,
                     help="variant directory, e.g. spatial_gate_trial (never the "
                          f"production {PRODUCTION_DIR}: fit a variant, then promote it)")
    fit.add_argument("--width", type=int, default=32)
    fit.add_argument("--lr-input", action="store_true",
                     help="also feed the LR image and its blackout mask to the gate "
                          "(off by default: no PSNR gain, more sky/halo error)")
    fit.add_argument("--steps", type=int, default=2000)
    fit.add_argument("--batch", type=int, default=8)
    fit.add_argument("--crop", type=int, default=192)
    fit.add_argument("--lr", type=float, default=2e-3)
    fit.add_argument("--eval-every", type=int, default=250)
    fit.add_argument("--holdout", type=int, default=15)
    fit.add_argument("--blackout-fields", type=int, default=40)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--knee-loss", action=argparse.BooleanOptionalAction, default=True,
                     help="score the loss at 11 knees from 0.1 to 1e4 e- (the "
                          "knee-integrated PSNR); --no-knee-loss scores the band knee only")
    fit.add_argument("--mix", choices=MIX_SPACES, default=MIX_LINEAR,
                     help="average the members in electrons (linear: knee-free, "
                          "flux-conserving) or in band-knee asinh space")
    fit.add_argument("--members", default="",
                     help="comma-separated member numbers for a pruned gate (e.g. 170,180)")
    fit.set_defaults(func=cmd_fit)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--gates", nargs="+", default=["spatial_gate_combiner"])
    cmp_.add_argument("--blackout-fields", type=int, default=40)
    cmp_.add_argument("--seed", type=int, default=0)
    cmp_.add_argument("--report", default=None)
    cmp_.add_argument("--figures", default=None)
    cmp_.add_argument("--max-figures", type=int, default=6)
    cmp_.add_argument("--no-knee", action="store_true",
                      help="skip the per-combiner PSNR-vs-knee curves")
    cmp_.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
