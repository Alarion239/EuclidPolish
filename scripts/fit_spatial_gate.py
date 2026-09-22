"""Fit the spatial gating combiner and compare it with the current combiner.

Fits on the cached STARFULL validate member cubes (85 fields train, 15 held
out), optionally adding blackout-augmented copies of the training fields
(one extra member-inference pass, cached next to the cubes). ``--compare``
scores every method on the cached test cubes plus blackout-augmented test
fields and writes a JSON report and comparison figures.

    python scripts/fit_spatial_gate.py fit --out-name spatial_gate_combiner
    python scripts/fit_spatial_gate.py compare --gates spatial_gate_combiner
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import binary_dilation, distance_transform_edt, maximum_filter  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from euclid_polish.config import Config  # noqa: E402
from euclid_polish.ensemble_registry import default_ensemble_dir  # noqa: E402
from euclid_polish.eval.combiner import (  # noqa: E402
    COMBINER_MODELS,
    RAW_INCREMENTAL_MINMEANMAX_RBF_KIND,
    load_combiner,
)
from euclid_polish.eval.spatial_gate import (  # noqa: E402
    band_scales,
    load_spatial_gate,
    save_spatial_gate,
)
from euclid_polish.eval.spatial_gate_fit import (  # noqa: E402
    LazyMemberRunner,
    build_blackout_fields,
    fit_spatial_gate,
    load_cube_fields,
    split_holdout,
)
from euclid_polish.web.helpers.paths import _sky_records_local_dir  # noqa: E402

BANDS = tuple(Config.HR_TARGET_BAND_NAMES)
BRIGHTNESS_EDGES = (0.02, 0.1, 0.5, 2.0)
BRIGHTNESS_NAMES = ("sky", "faint", "mid", "bright", "core")
PEAK_ASINH = 4.0
HALO_RADII = (3.0, 15.0)


def _regime_dir() -> str:
    return os.path.abspath(os.path.join(Config.VIS_DIR, "ensemble", "starfull"))


def _fwhm(cubes_dir: str) -> float:
    with open(os.path.join(cubes_dir, "viz_index.json")) as handle:
        return float(json.load(handle)["target_psf_fwhm_arcsec"])


def cmd_fit(args) -> None:
    regime = _regime_dir()
    cubes = os.path.join(regime, "cubes_validate")
    records = _sky_records_local_dir()
    fields, labels = load_cube_fields(cubes, records, "validate", target_name="hr",
                                      target_fwhm_arcsec=_fwhm(cubes))
    train, holdout = split_holdout(fields, args.holdout, args.seed)
    print(f"{len(train)} train fields, {len(holdout)} held out, {len(labels)} members")
    extra = []
    if args.blackout_fields > 0:
        runner = LazyMemberRunner(default_ensemble_dir(), starless=False, labels=labels)
        extra = build_blackout_fields(
            train, labels, runner, os.path.join(regime, "cubes_validate_blackout"),
            max_fields=args.blackout_fields, seed=args.seed,
            progress=lambda i, n, msg: print(f"  [{i}/{n}] {msg}", flush=True))
        print(f"{len(extra)} blackout-augmented training fields")
    active = None
    if args.members:
        wanted = {name.strip() for name in args.members.split(",") if name.strip()}
        active = [i for i, label in enumerate(labels) if label.split("·")[0] in wanted]
        missing = wanted - {labels[i].split("·")[0] for i in active}
        if missing:
            raise SystemExit(f"unknown members: {sorted(missing)}")
        print(f"pruned gate over {len(active)} members: {[labels[i] for i in active]}")
    comb = fit_spatial_gate(
        train + extra, holdout, labels, width=args.width, use_lr=args.lr_input,
        steps=args.steps, batch_size=args.batch, crop=args.crop,
        learning_rate=args.lr, eval_every=args.eval_every, seed=args.seed,
        active_members=active, log=lambda msg: print(msg, flush=True))
    comb.starfull = True
    comb.fit_meta["blackout_fields"] = len(extra)
    out = os.path.join(regime, args.out_name)
    save_spatial_gate(comb, out)
    print(f"saved {out}")


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

class Scores:
    """Per-method accumulators over test fields."""

    def __init__(self):
        self.band_psnr: list[np.ndarray] = []
        self.bins = np.zeros(len(BRIGHTNESS_NAMES))
        self.halo = np.zeros(len(BANDS))
        self.hole = np.zeros(len(BANDS))

    def summary(self, n_bins, n_halo, n_hole) -> dict:
        psnr = np.mean(self.band_psnr, axis=0) if self.band_psnr else np.full(len(BANDS), np.nan)
        return {"band_psnr": psnr.tolist(),
                "bin_mse": (self.bins / np.maximum(n_bins, 1)).tolist(),
                "halo_mse": (self.halo / np.maximum(n_halo, 1)).tolist(),
                "hole_mse": (self.hole / np.maximum(n_hole, 1)).tolist()}


def _psnr(mse: np.ndarray) -> np.ndarray:
    peak = float(Config.PSNR_PEAK_STRETCHED)
    return 10.0 * np.log10(peak * peak / np.maximum(mse, 1e-20))


def _halo_mask(truth_vis: np.ndarray) -> np.ndarray:
    peaks = (maximum_filter(truth_vis, size=7) == truth_vis) & (truth_vis > PEAK_ASINH)
    if not peaks.any():
        return np.zeros_like(peaks)
    dist = distance_transform_edt(~peaks)
    return (dist >= HALO_RADII[0]) & (dist <= HALO_RADII[1])


def _hole_masks(field, source_lr) -> np.ndarray:
    """(H, W, C) HR mask of pixels whose LR was zeroed by the stamping."""
    new = (field.lr_e == 0) & (source_lr != 0)
    grown = np.stack([binary_dilation(new[..., c], iterations=2)
                      for c in range(new.shape[-1])], -1)
    h, w = field.shape
    return np.kron(grown, np.ones((2, 2, 1), bool))[:h, :w]


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


def cmd_compare(args) -> None:
    regime = _regime_dir()
    cubes = os.path.join(regime, "cubes")
    records = _sky_records_local_dir()
    fields, labels = load_cube_fields(cubes, records, "test", target_name="hr",
                                      target_fwhm_arcsec=_fwhm(cubes))
    source_lr = {f.index: f.lr_e for f in fields}
    runner = LazyMemberRunner(default_ensemble_dir(), starless=False, labels=labels)
    blackouts = build_blackout_fields(
        fields, labels, runner, os.path.join(regime, "cubes_blackout"),
        max_fields=args.blackout_fields, seed=args.seed + 1,
        progress=lambda i, n, msg: print(f"  [{i}/{n}] {msg}", flush=True))
    rbf_kind = RAW_INCREMENTAL_MINMEANMAX_RBF_KIND
    rbf = load_combiner(regime, member_labels=labels,
                        artifact_dir=COMBINER_MODELS[rbf_kind].artifact_dir)
    gates = {}
    for name in args.gates:
        gate = load_spatial_gate(os.path.join(regime, name), member_labels=labels)
        if gate is None:
            raise SystemExit(f"no spatial gate at {name} for these members")
        gates[f"gate:{name}"] = gate
    scales = band_scales(BANDS).astype(np.float32)
    methods = ["mean", "rbf"] + list(gates)
    scores = {group: {m: Scores() for m in methods + [f"member:{l}" for l in labels]}
              for group in ("natural", "blackout")}
    counts = {group: {"bins": np.zeros(len(BRIGHTNESS_NAMES)), "halo": np.zeros(len(BANDS)),
                      "hole": np.zeros(len(BANDS))} for group in scores}
    usage = {name: np.zeros((len(labels), len(BANDS))) for name in gates}
    usage_src = {name: np.zeros((len(labels), len(BANDS))) for name in gates}
    n_usage = n_usage_src = 0
    timing = {name: [] for name in gates}
    timing["rbf"] = []
    figure_candidates = []
    for group, group_fields in (("natural", fields), ("blackout", blackouts)):
        for position, f in enumerate(group_fields, 1):
            members = f.members_e()
            truth = np.arcsinh(f.target_e / scales)
            outputs = {"mean": members.mean(0)}
            cached = os.path.join(cubes, f"{COMBINER_MODELS[rbf_kind].cube_prefix}_{f.index:05d}.npy")
            if group == "natural" and os.path.isfile(cached):
                outputs["rbf"] = np.load(cached)
            elif rbf is not None:
                started = time.time()
                outputs["rbf"] = rbf.apply_field(members)
                timing["rbf"].append(time.time() - started)
            weights = {}
            for name, gate in gates.items():
                lr = f.lr_e if gate.use_lr else None
                started = time.time()
                outputs[name] = gate.apply_field(members, lr=lr)
                timing[name].append(time.time() - started)
                weights[name] = gate.weights_field(members, lr=lr)
            for m, label in enumerate(labels):
                outputs[f"member:{label}"] = members[m]
            bins = np.digitize(truth[..., 0], BRIGHTNESS_EDGES)
            halo = _halo_mask(truth[..., 0])
            holes = _hole_masks(f, source_lr[f.index]) if group == "blackout" else None
            cnt = counts[group]
            cnt["bins"] += np.bincount(bins.ravel(), minlength=len(BRIGHTNESS_NAMES))
            cnt["halo"] += halo.sum()
            if holes is not None:
                cnt["hole"] += holes.reshape(-1, len(BANDS)).sum(0)
            for method, image in outputs.items():
                if method not in scores[group]:
                    continue
                err2 = (np.arcsinh(np.asarray(image, np.float32) / scales) - truth) ** 2
                s = scores[group][method]
                s.band_psnr.append(_psnr(err2.mean(axis=(0, 1))))
                s.bins += np.bincount(bins.ravel(), weights=err2[..., 0].ravel(),
                                      minlength=len(BRIGHTNESS_NAMES))
                s.halo += err2[halo].sum(0)
                if holes is not None:
                    s.hole += np.where(holes, err2, 0.0).reshape(-1, len(BANDS)).sum(0)
            if group == "natural":
                source = truth[..., 0] > 0.1
                for name, w in weights.items():
                    usage[name] += w.sum(axis=(0, 1))
                    usage_src[name] += w[source].sum(0)
                n_usage += truth.shape[0] * truth.shape[1]
                n_usage_src += int(source.sum())
            figure_candidates.append((group, f, halo.any()))
            print(f"  [{group} {position}/{len(group_fields)}] field {f.index}", flush=True)

    report = {"members": labels, "gates": {n: g.fit_meta.get("selected") for n, g in gates.items()},
              "groups": {}, "timing_s": {k: float(np.mean(v)) if v else None for k, v in timing.items()},
              "member_inference_s_per_field": float(np.mean(runner.seconds)) if runner.seconds else None,
              "members_needed": {name: len(g.needed_member_indices()) for name, g in gates.items()},
              "usage": {}}
    for group in scores:
        c = counts[group]
        report["groups"][group] = {m: s.summary(c["bins"], c["halo"], c["hole"])
                                   for m, s in scores[group].items()}
    n_gates = len(gates)
    for name in gates:
        report["usage"][name] = {"all_pixels": (usage[name] / max(n_usage, 1)).tolist(),
                                 "source_pixels": (usage_src[name] / max(n_usage_src, 1)).tolist()}
    out = args.report or os.path.join(regime, "spatial_gate_comparison.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)
    _print_report(report, labels, list(gates))
    print(f"wrote {out}")

    if args.figures and n_gates:
        os.makedirs(args.figures, exist_ok=True)
        gate_name = list(gates)[-1]
        gate = gates[gate_name]
        best_vis = int(np.argmax([report["groups"]["natural"][f"member:{l}"]["band_psnr"][0]
                                  for l in labels]))
        shown = 0
        for group, f, has_halo in figure_candidates:
            if shown >= args.max_figures:
                break
            if group == "natural" and not has_halo:
                continue
            members = f.members_e()
            truth = np.arcsinh(f.target_e / scales)[..., 0]
            lr = f.lr_e if gate.use_lr else None
            gate_out = np.arcsinh(gate.apply_field(members, lr=lr) / scales)[..., 0]
            rbf_out = (np.arcsinh(rbf.apply_field(members) / scales)[..., 0]
                       if rbf is not None else None)
            dom = gate.weights_field(members, lr=lr)[..., 0].argmax(-1)
            if group == "blackout":
                holes = _hole_masks(f, source_lr[f.index])[..., 0]
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
                         (dom, labels), center)
            shown += 1
            print(f"figure {path}")


def _print_report(report, labels, gates) -> None:
    for group, block in report["groups"].items():
        members = {k: v for k, v in block.items() if k.startswith("member:")}
        best = max(members, key=lambda k: members[k]["band_psnr"][0])
        print(f"\n== {group} test fields ==  (best single member by VIS: {best})")
        rows = ["mean", best, "rbf", *gates]
        print(f"{'method':32s} {'VIS':>7s} {'Y':>7s} {'J':>7s} {'H':>7s}   "
              + " ".join(f"{n:>7s}" for n in BRIGHTNESS_NAMES) + f"  {'halo':>7s}  {'holes':>7s}")
        ref = block[best]
        for row in rows:
            if row not in block:
                continue
            s = block[row]
            rel = [s["bin_mse"][i] / max(ref["bin_mse"][i], 1e-30) for i in range(len(BRIGHTNESS_NAMES))]
            halo = s["halo_mse"][0] / max(ref["halo_mse"][0], 1e-30)
            hole = (np.mean(s["hole_mse"]) / max(np.mean(ref["hole_mse"]), 1e-30)
                    if group == "blackout" else float("nan"))
            print(f"{row:32s} " + " ".join(f"{v:7.3f}" for v in s["band_psnr"])
                  + "   " + " ".join(f"{v:7.3f}" for v in rel) + f"  {halo:7.3f}  {hole:7.3f}")
    print("\n(bin/halo/hole columns: VIS squared error relative to the best single member)")
    for name in gates:
        use = np.asarray(report["usage"][name]["source_pixels"])
        order = np.argsort(-use[:, 0])[:6]
        print(f"{name} mean VIS weight on source pixels: "
              + ", ".join(f"{labels[i]} {use[i, 0]:.2f}" for i in order))
    print("timing (s/field):", {k: (round(v, 3) if v else v) for k, v in report["timing_s"].items()},
          "member inference:", report["member_inference_s_per_field"])
    print("members each gate needs:", report["members_needed"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--out-name", default="spatial_gate_combiner")
    fit.add_argument("--width", type=int, default=32)
    fit.add_argument("--lr-input", action="store_true",
                     help="also feed the LR image and its blackout mask to the gate "
                          "(off by default: no PSNR gain, more sky/halo error)")
    fit.add_argument("--steps", type=int, default=3000)
    fit.add_argument("--batch", type=int, default=8)
    fit.add_argument("--crop", type=int, default=192)
    fit.add_argument("--lr", type=float, default=2e-3)
    fit.add_argument("--eval-every", type=int, default=250)
    fit.add_argument("--holdout", type=int, default=15)
    fit.add_argument("--blackout-fields", type=int, default=40)
    fit.add_argument("--seed", type=int, default=0)
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
    cmp_.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
