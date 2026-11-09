"""Cross-validated hyperparameter search for the colour+SFR forest.

The forest is a conditional-DISTRIBUTION model: a draw resamples real Q1 rows
from the query's leaf neighbourhood. So a configuration is scored by how well
that neighbourhood predicts a held-out galaxy's measured outputs, with the
continuous ranked probability score (CRPS) of the weighted empirical
distribution, a proper scoring rule. Lower is better.

Protocol (K-fold, paired across configurations):
- rows are split into K folds with a fixed seed; each configuration is grown
  on K−1 folds, and its leaves are populated with the TRAINING rows only (as
  production populates them with every catalogue row), so a held-out row
  never predicts itself and never shaped a split;
- for each held-out row the QRF weights over training rows are the average
  over trees of w_j / W_leaf (exactly ``ConditionalColorSFRSampler._gather``);
- outputs scored: asinh(ratio/0.1) for Y, J, H (the split-target space) and
  log SFR censored at −5 (SFR-valid held-out rows only, predicted from the
  neighbourhood's SFR-valid rows, as the sampler's borrow policy does);
- skill = 1 − CRPS / CRPS(unconditional training distribution), per output;
  the headline score is the mean skill over the four outputs, weighted by
  galaxy weight, with fold-paired differences for error bars.

Scores are on the MEASURED (noisy) outputs, before the sampler's per-draw
deconvolution: measurement noise is the same for every configuration, so it
shifts every CRPS equally and does not change the ranking.

Usage:
    python scripts/tune_color_sfr_forest.py --out output/color_sfr_tuning/results.json
Re-running with the same --out skips configurations already scored.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from euclid_polish.population.conditional_color_sfr import (
    COLOR_SFR_ASINH_SOFTENING,
    COLOR_SFR_FIT_SEED,
    COLOR_SFR_SPLIT_CENSOR_LOGSFR,
)
from euclid_polish.population.conditional_color_sfr_fit import (
    color_sfr_forest,
    color_sfr_split_problem,
    read_color_sfr_rows,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/population_comparison/euclid_population.csv"
ACTIVE = ROOT / "data/population_comparison/calibrations/joint_galaxy_population_active.json"
OUTPUT_NAMES = ("asinh_y", "asinh_j", "asinh_h", "log_sfr")
NO_CENSOR = -1e9


class _ImputationLaw:
    """The radius law's mean, from the active artifact's re_imputation block."""

    def __init__(self, block: dict) -> None:
        self.pivot = float(block["pivot_mag"])
        self.intercept = float(block["intercept_log10_arcsec"])
        self.slope = float(block["slope_log10_arcsec_per_mag"])

    def mean(self, magnitude):
        return self.intercept + self.slope * (np.asarray(magnitude) - self.pivot)


def weighted_crps(x: np.ndarray, w: np.ndarray, y: float) -> float:
    """CRPS of the weighted empirical distribution (x, w) at observation y."""
    w = w / np.sum(w)
    order = np.argsort(x, kind="mergesort")
    xs, ws = x[order], w[order]
    before = np.cumsum(ws) - ws
    spread = 2.0 * np.sum(ws * xs * (2.0 * before + ws - 1.0))
    return float(np.sum(ws * np.abs(xs - y)) - 0.5 * spread)


def qrf_weights(train_leaves: np.ndarray, test_leaves: np.ndarray,
                weight: np.ndarray) -> sparse.csr_matrix:
    """(n_test, n_train) QRF weight matrix; every row sums to one."""
    n_train, n_trees = train_leaves.shape
    n_test = test_leaves.shape[0]
    total = None
    for t in range(n_trees):
        lt = train_leaves[:, t]
        order = np.argsort(lt, kind="mergesort")
        sorted_leaves = lt[order]
        leaf_ids, starts, counts = np.unique(
            sorted_leaves, return_index=True, return_counts=True,
        )
        leaf_weight = np.add.reduceat(weight[order], starts)
        at = test_leaves[:, t]
        pos = np.searchsorted(leaf_ids, at)
        pos = np.clip(pos, 0, leaf_ids.size - 1)
        hit = leaf_ids[pos] == at          # a test row's leaf may hold no training rows
        n_per = np.where(hit, counts[pos], 0)
        rows = np.repeat(np.arange(n_test), n_per)
        offsets = np.arange(rows.size) - np.repeat(np.cumsum(n_per) - n_per, n_per)
        cols = order[np.repeat(starts[pos], n_per) + offsets]
        data = weight[cols] / np.repeat(leaf_weight[pos], n_per)
        m = sparse.csr_matrix((data, (rows, cols)), shape=(n_test, n_train))
        total = m if total is None else total + m
    sums = np.asarray(total.sum(axis=1)).ravel()
    return sparse.diags(1.0 / np.maximum(sums, 1e-300)) @ total


def score_fold(config: dict, rows, outputs: np.ndarray, folds: np.ndarray,
               fold: int, test_pick: np.ndarray, climatology: dict) -> dict:
    features, targets = color_sfr_split_problem(
        rows, sfr_censor_logsfr=config["censor"],
    )
    train = folds != fold
    test = test_pick
    weight = rows.weight
    forest = color_sfr_forest(
        float(np.sum(weight[train])),
        tree_count=config["trees"],
        min_leaf_weight=config["min_leaf"],
        max_features=config["max_features"],
        fit_seed=COLOR_SFR_FIT_SEED,
    )
    started = time.time()
    forest.fit(features[train], targets[train], sample_weight=weight[train])
    fit_seconds = time.time() - started
    train_index = np.flatnonzero(train)
    W = qrf_weights(
        forest.apply(features[train]), forest.apply(features[test]), weight[train],
    )
    valid_train = rows.sfr_valid[train]
    crps = np.full((test.size, 4), np.nan)
    unique_rows = np.zeros(test.size)
    for i in range(test.size):
        start, stop = W.indptr[i], W.indptr[i + 1]
        cols, w = W.indices[start:stop], W.data[start:stop]
        unique_rows[i] = cols.size
        row = test[i]
        for k in range(3):
            crps[i, k] = weighted_crps(outputs[train_index[cols], k], w, outputs[row, k])
        if rows.sfr_valid[row]:
            ok = valid_train[cols]
            if np.any(ok):
                crps[i, 3] = weighted_crps(
                    outputs[train_index[cols[ok]], 3], w[ok], outputs[row, 3],
                )
            else:
                crps[i, 3] = weighted_crps(climatology["x"][3], climatology["w"][3], outputs[row, 3])
    return {
        "crps": crps, "unique_rows": unique_rows, "fit_seconds": fit_seconds,
        "leaves": float(np.mean([e.tree_.n_leaves for e in forest.estimators_])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "output/color_sfr_tuning/results.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-per-fold", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--stage", choices=("grid", "trees", "censor", "all"), default="all")
    parser.add_argument("--min-leaves", default="5,10,25,50,100,200",
                        help="comma-separated minimum leaf weights for the grid")
    parser.add_argument("--max-features", default="1.0,0.67,0.34",
                        help="comma-separated feature fractions for the grid")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    active = json.loads(ACTIVE.read_text())
    law = _ImputationLaw(active["color_sfr_model"]["re_imputation"])
    rows = read_color_sfr_rows(CATALOG, radius_law=law)
    n = rows.weight.size
    outputs = np.column_stack((
        np.arcsinh(rows.ratio / COLOR_SFR_ASINH_SOFTENING),
        np.maximum(rows.log_sfr, COLOR_SFR_SPLIT_CENSOR_LOGSFR),
    ))
    rng = np.random.default_rng(args.seed)
    folds = rng.permutation(n) % args.folds
    picks = {}
    for f in range(args.folds):
        members = np.flatnonzero(folds == f)
        picks[f] = np.sort(rng.choice(members, min(args.test_per_fold, members.size), replace=False))
    # unconditional reference: 6000 training rows per output, by weight
    climatology = {"x": {}, "w": {}}
    for k in range(4):
        pool = np.flatnonzero(rows.sfr_valid) if k == 3 else np.arange(n)
        sample = rng.choice(pool, 6000, replace=False)
        climatology["x"][k], climatology["w"][k] = outputs[sample, k], rows.weight[sample]

    results = json.loads(args.out.read_text()) if args.out.exists() else {
        "protocol": {"folds": args.folds, "test_per_fold": args.test_per_fold, "seed": args.seed,
                     "rows": int(n), "outputs": OUTPUT_NAMES},
        "configs": {},
    }

    def key(c):
        return (f"censor={c['censor']:g}|min_leaf={c['min_leaf']:g}"
                f"|max_features={c['max_features']:g}|trees={c['trees']}")

    def run(config):
        k = key(config)
        if k in results["configs"]:
            return
        per_fold = []
        for f in range(args.folds):
            r = score_fold(config, rows, outputs, folds, f, picks[f], climatology)
            test = picks[f]
            w = rows.weight[test]
            crps = r["crps"]
            ref = np.full_like(crps, np.nan)
            for i, row in enumerate(test):
                for kk in range(4):
                    if np.isfinite(crps[i, kk]):
                        ref[i, kk] = weighted_crps(
                            climatology["x"][kk], climatology["w"][kk], outputs[row, kk],
                        )
            fold_out = {"fit_seconds": r["fit_seconds"], "leaves_per_tree": r["leaves"],
                        "median_unique_rows": float(np.median(r["unique_rows"])),
                        "crps": [], "skill": [], "by_magnitude": {}}
            for kk in range(4):
                ok = np.isfinite(crps[:, kk])
                c = float(np.average(crps[ok, kk], weights=w[ok]))
                c0 = float(np.average(ref[ok, kk], weights=w[ok]))
                fold_out["crps"].append(c)
                fold_out["skill"].append(1.0 - c / c0)
            mag = rows.magnitude[test]
            for name, lo, hi in (("bright_lt23", -99, 23), ("mid_23_25", 23, 25), ("faint_ge25", 25, 99)):
                sel = (mag >= lo) & (mag < hi)
                skills = []
                for kk in range(4):
                    ok = sel & np.isfinite(crps[:, kk])
                    skills.append(1.0 - float(np.average(crps[ok, kk], weights=w[ok]))
                                  / float(np.average(ref[ok, kk], weights=w[ok])) if ok.any() else None)
                fold_out["by_magnitude"][name] = skills
            per_fold.append(fold_out)
            skill = np.round(fold_out["skill"], 4)
            print(f"{k} fold {f}: skill {skill} fit {r['fit_seconds']:.0f}s", flush=True)
        results["configs"][k] = {"config": config, "folds": per_fold}
        args.out.write_text(json.dumps(results, indent=1))

    base = {"censor": COLOR_SFR_SPLIT_CENSOR_LOGSFR, "min_leaf": 25.0, "max_features": 1.0, "trees": 50}
    if args.stage in ("grid", "all"):
        run({**base, "censor": NO_CENSOR})          # the forest as fitted on 2026-09-13
        for mf in (float(v) for v in args.max_features.split(",")):
            for leaf in (float(v) for v in args.min_leaves.split(",")):
                run({**base, "min_leaf": leaf, "max_features": mf})
    grid = [v for v in results["configs"].values()
            if v["config"]["censor"] == COLOR_SFR_SPLIT_CENSOR_LOGSFR and v["config"]["trees"] == 50]
    if args.stage in ("trees", "all"):
        best = max(grid, key=lambda v: np.mean([np.mean(f["skill"]) for f in v["folds"]]))["config"]
        for trees in (25, 100, 200):
            run({**best, "trees": trees})
    if args.stage in ("censor", "all"):
        # Does the SFR censor matter at the chosen leaf size? (At 25 it is a wash.)
        for leaf in (float(v) for v in args.min_leaves.split(",")):
            for mf in (float(v) for v in args.max_features.split(",")):
                run({**base, "min_leaf": leaf, "max_features": mf, "censor": NO_CENSOR})
    print("done", flush=True)


if __name__ == "__main__":
    main()
