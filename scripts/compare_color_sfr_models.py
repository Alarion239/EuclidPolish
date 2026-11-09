"""Cross-validated comparison: colour+SFR forest vs joint XD Gaussian mixture.

Both models are judged on what GENERATION would draw, pushed through each
held-out Q1 row's own measurement noise (a posterior-predictive check), and
compared with that row's measured values:

- forest: the production sampler's draw (pick a neighbourhood row by QRF
  weight, per-band analytic deconvolution, SFR own-or-borrowed), grown with
  the tuned settings (leaf 200, 2 of 3 inputs, SFR split censor);
- mixture: :mod:`euclid_polish.population.color_sfr_mixture`, conditioned on
  the row's magnitude and (when resolved) size.

Metrics per held-out row, averaged by galaxy weight, as skill against the
unconditional training distribution (higher is better):
- CRPS of asinh(ratio/0.1) for Y, J, H and of log SFR (censored −5);
- energy score of the four outputs jointly (SFR-valid rows), dimensions
  scaled by their robust spread — the joint metric the marginal CRPS misses;
plus Spearman correlation matrices and quenched fractions by magnitude/size.

Usage: python scripts/compare_color_sfr_models.py [--folds 5 --test-per-fold 2000 --components 2,3,4]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tune_color_sfr_forest import _ImputationLaw, qrf_weights

from euclid_polish.population.color_sfr_mixture import (
    ColorSFRMixtureSampler,
    fit_mixture_model,
    mixture_rows,
)
from euclid_polish.population.conditional_color_sfr import (
    COLOR_SFR_ASINH_SOFTENING,
    COLOR_SFR_FIT_SEED,
    COLOR_SFR_RATIO_FLOOR,
    weighted_quantile,
)
from euclid_polish.population.conditional_color_sfr_fit import (
    color_sfr_forest,
    color_sfr_split_problem,
    read_color_sfr_rows,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/population_comparison/euclid_population.csv"
ACTIVE = ROOT / "data/population_comparison/calibrations/joint_galaxy_population_active.json"
S = 120  # draws per held-out row


def load():
    law = _ImputationLaw(json.loads(ACTIVE.read_text())["color_sfr_model"]["re_imputation"])
    rows = read_color_sfr_rows(CATALOG, radius_law=law)
    c = pd.read_csv(CATALOG, low_memory=False)

    def col(k):
        return pd.to_numeric(c[k], errors="coerce") if k in c else pd.Series(np.nan, index=c.index)

    f, fe = col("flux_vis_2fwhm_aper_uJy"), col("fluxerr_vis_2fwhm_aper_uJy")
    ok = ((f > 0) & (fe > 0) & (f / fe >= 5) & (col("vis_det") == 1) & (col("det_quality_flag") < 4)
          & (col("spurious_prob") <= 0.5) & (col("point_like_prob") >= 0) & (col("point_like_prob") < 0.5)
          & (col("flag_vis").isna() | (col("flag_vis") == 0))
          & (col("spurious_flag").isna() | (col("spurious_flag") == 0))
          & (col("point_like_flag").isna() | (col("point_like_flag") != 1)))
    for b in "yjh":
        ok &= np.isfinite(col(f"flux_{b}_2fwhm_aper_uJy")) & (col(f"fluxerr_{b}_2fwhm_aper_uJy") > 0)
    d = c[ok]
    ratio_y = (col("flux_y_2fwhm_aper_uJy")[ok] / f[ok]).to_numpy()
    if d.shape[0] != rows.weight.size or not np.allclose(ratio_y, rows.ratio[:, 0]):
        raise RuntimeError("pandas selection does not reproduce read_color_sfr_rows")
    snr = (f[ok] / fe[ok]).to_numpy()
    mass = col("phz_pp_median_stellarmass")[ok].to_numpy()
    sfr_sig = (0.5 * (col("phz_pp_sfr_p84") - col("phz_pp_sfr_p16")))[ok].to_numpy()
    mass_sig = (0.5 * (col("phz_pp_stellarmass_p84") - col("phz_pp_stellarmass_p16")))[ok].to_numpy()
    sfr_sig = np.where(np.isfinite(sfr_sig), sfr_sig, 0.3)
    mass_sig = np.where(np.isfinite(mass_sig), mass_sig, 0.2)
    return rows, snr, mass, sfr_sig, mass_sig


def transform(ratios, lsfr):
    return np.column_stack((np.arcsinh(ratios / COLOR_SFR_ASINH_SOFTENING), np.maximum(lsfr, -5.0)))


def crps_samples(x, y):
    xs = np.sort(x)
    n = xs.size
    spread = 2.0 * np.sum(xs * (2 * np.arange(n) - n + 1)) / (n * n)
    return float(np.mean(np.abs(xs - y)) - 0.5 * spread)


def energy(X, y):
    t1 = np.mean(np.linalg.norm(X - y, axis=1))
    diff = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    t2 = diff.sum() / (len(X) * (len(X) - 1))
    return float(t1 - 0.5 * t2)


def colors_of(r):
    r = np.maximum(r, 1e-6)
    return np.column_stack((2.5 * np.log10(r[:, 0]), 2.5 * np.log10(r[:, 1] / r[:, 0]),
                            2.5 * np.log10(r[:, 2] / r[:, 1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test-per-fold", type=int, default=2000)
    ap.add_argument("--components", default="2,3,4")
    ap.add_argument("--out", type=Path, default=ROOT / "output/color_sfr_mixture/comparison.json")
    ap.add_argument("--summarise-only", action="store_true",
                    help="rebuild the summary from the saved .records.npz")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows, snr, mass, sfr_sig, mass_sig = load()
    if args.summarise_only:
        components = [int(k) for k in args.components.split(",")]
        summarise(args.out.with_suffix(".records.npz"), args.out, rows,
                  ["forest"] + [f"mixture_K{k}" for k in components], args.folds)
        return
    n = rows.weight.size
    values, noise, observed = mixture_rows(rows, snr, mass, sfr_sig, mass_sig)
    rng = np.random.default_rng(20261108)
    folds = rng.permutation(n) % args.folds
    out_t = transform(rows.ratio, rows.log_sfr)
    scale = (np.nanpercentile(out_t, 84, axis=0) - np.nanpercentile(out_t, 16, axis=0)) / 2
    components = [int(k) for k in args.components.split(",")]
    models = ["forest"] + [f"mixture_K{k}" for k in components]
    records = {m: {"crps": [], "es": [], "w": [], "fold": [], "mag": [], "lr": [], "res": [],
                   "draw_ratio": [], "draw_sfr": [], "draw_q": []} for m in models + ["reference"]}
    truth = {"idx": [], "fold": []}
    for fold in range(args.folds):
        t0 = time.time()
        train = np.flatnonzero(folds != fold)
        test = np.sort(rng.choice(np.flatnonzero(folds == fold), args.test_per_fold, replace=False))
        truth["idx"] += list(test); truth["fold"] += [fold] * test.size
        clim = rng.choice(train, 400, replace=False, p=rows.weight[train] / rows.weight[train].sum())
        clim_sfr = rng.choice(train[rows.sfr_valid[train]], 400, replace=False)
        # ---- forest on train ----
        features, targets = color_sfr_split_problem(rows)
        forest = color_sfr_forest(float(rows.weight[train].sum()), tree_count=50, min_leaf_weight=200,
                                  max_features=0.67, fit_seed=COLOR_SFR_FIT_SEED)
        forest.fit(features[train], targets[train], sample_weight=rows.weight[train])
        W = qrf_weights(forest.apply(features[train]), forest.apply(features[test]), rows.weight[train])
        # ---- mixtures on train ----
        samplers = {}
        sfr_ref = rows.log_sfr[train][rows.sfr_valid[train]]
        for k in components:
            bins = fit_mixture_model(values[train], noise[train], observed[train], rows.weight[train],
                                     rows.magnitude[train], rows.sfr_valid[train], n_components=k)
            samplers[k] = ColorSFRMixtureSampler(bins, sfr_ref, rows.weight[train][rows.sfr_valid[train]])
        print(f"fold {fold}: fits done in {time.time() - t0:.0f}s", flush=True)
        g = np.random.default_rng(fold)
        for i, row in enumerate(test):
            y = out_t[row]
            Si = noise[row, :3, :3]
            Li = np.linalg.cholesky(Si + 1e-12 * np.eye(3))
            draws = {}
            # forest: production-style draws
            a, b = W.indptr[i], W.indptr[i + 1]
            nb, wt = train[W.indices[a:b]], W.data[a:b]
            loc = np.empty(3); intr = np.empty(3)
            for band in range(3):
                lo_, mid, hi_ = weighted_quantile(rows.ratio[nb, band], wt, (0.16, 0.5, 0.84))
                vc = rows.ratio_var[nb, band]; mv = float(weighted_quantile(vc, wt, 0.5))
                sane = vc <= 25.0 * max(mv, 1e-300)
                loc[band] = mid
                intr[band] = max((0.5 * (hi_ - lo_)) ** 2 - np.average(vc[sane], weights=wt[sane]), 0.0)
            pick = nb[g.choice(nb.size, S, p=wt / wt.sum())]
            gain = intr / (intr + rows.ratio_var[pick])
            mu = loc + gain * (rows.ratio[pick] - loc)
            rat = np.maximum(mu + np.sqrt(np.maximum((1 - gain) * intr, 0)) * g.standard_normal((S, 3)),
                             COLOR_SFR_RATIO_FLOOR)
            valid_nb = rows.sfr_valid[nb]
            sfr = np.where(rows.sfr_valid[pick], rows.log_sfr[pick], np.nan)
            q = np.where(rows.sfr_valid[pick], rows.sfr_class[pick] == 0, False)
            miss = ~rows.sfr_valid[pick]
            if miss.any() and valid_nb.any():
                bw = wt[valid_nb] / wt[valid_nb].sum()
                donor = nb[valid_nb][g.choice(valid_nb.sum(), int(miss.sum()), p=bw)]
                sfr[miss] = rows.log_sfr[donor]; q[miss] = rows.sfr_class[donor] == 0
            draws["forest"] = (rat, sfr, q)
            lr = rows.log_radius[row] if rows.resolved[row] else None
            for k, smp in samplers.items():
                d = smp.sample(rows.magnitude[row], None if lr is None else 10 ** lr, g, size=S)
                # SFR and mass as PHZ would MEASURE them (the row's own PHZ errors), so the
                # class and SFR compare like-for-like with the catalogue values.
                sfr_m = d[:, 3] + sfr_sig[row] * g.standard_normal(S)
                mass_m = d[:, 4] + mass_sig[row] * g.standard_normal(S)
                draws[f"mixture_K{k}"] = (d[:, :3], sfr_m, (sfr_m - mass_m) < -11.0)
            draws["reference"] = (rows.ratio[clim], None, None)
            for m, (rat, sfr, q) in draws.items():
                if m == "reference":
                    noisy = rat  # already measured values
                    Xt = transform(noisy, np.full(len(noisy), np.nan))
                    Xt[:, 3] = np.maximum(rows.log_sfr[clim_sfr], -5.0)
                else:
                    noisy = rat + (Li @ g.standard_normal((3, len(rat)))).T
                    Xt = transform(noisy, sfr)
                c = [crps_samples(Xt[:, j], y[j]) for j in range(3)]
                c.append(crps_samples(Xt[:, 3], y[3]) if rows.sfr_valid[row] else np.nan)
                es = energy(Xt[:100] / scale, y / scale) if rows.sfr_valid[row] else np.nan
                rec = records[m]
                rec["crps"].append(c); rec["es"].append(es); rec["w"].append(rows.weight[row])
                rec["fold"].append(fold); rec["mag"].append(rows.magnitude[row])
                rec["lr"].append(rows.log_radius[row]); rec["res"].append(bool(rows.resolved[row]))
                if m != "reference":
                    rec["draw_ratio"].append(noisy[0].tolist()); rec["draw_sfr"].append(float(sfr[0]))
                    rec["draw_q"].append(float(np.mean(q)))
        print(f"fold {fold}: scored in {time.time() - t0:.0f}s", flush=True)

    records_path = args.out.with_suffix(".records.npz")
    np.savez_compressed(records_path, idx=np.array(truth["idx"]), fold=np.array(truth["fold"]),
                        **{f"{m}__{k}": np.array(v) for m, rec in records.items() for k, v in rec.items()})
    summarise(records_path, args.out, rows, models, args.folds)


def _skill(model_vals, ref_vals, weight):
    ok = np.isfinite(model_vals) & np.isfinite(ref_vals) & (weight > 0)
    return 1 - np.average(model_vals[ok], weights=weight[ok]) / np.average(ref_vals[ok], weights=weight[ok])


def _spearman(colors, sfr):
    return spearmanr(np.column_stack((colors, np.maximum(sfr, -5)))).correlation.tolist()


def summarise(records_path, out_path, rows, models, n_folds):
    z = np.load(records_path)
    idx, fl = z["idx"], z["reference__fold"]
    # Rows reporting a ratio error > 25x the typical one at their magnitude (any band) are left out
    # of the scores: pushing draws through that junk error makes their predictive absurdly wide
    # (the forest's own deconvolution trims the same rows). 14% of held-out rows on 2026-11-09.
    half_mag = np.floor(rows.magnitude * 2) / 2
    junk = np.zeros(rows.magnitude.size, bool)
    for value in np.unique(half_mag):
        s_ = half_mag == value
        junk[s_] = np.any(rows.ratio_var[s_] > 25 * np.median(rows.ratio_var[s_], axis=0), axis=1)
    keep = ~junk[idx]
    wts = np.where(keep, z["reference__w"], 0.0)
    ref_c, ref_e = z["reference__crps"], z["reference__es"]
    summary = {"models": models, "skill": {}, "paired_vs_forest": {}, "correlations": {},
               "quenched": {}, "quenched_by_size": {}, "junk_error_rows_excluded": float(np.mean(~keep))}
    for m in models:
        mc, me = z[f"{m}__crps"], z[f"{m}__es"]
        per_fold = []
        for f_ in range(n_folds):
            s = fl == f_
            per_fold.append([_skill(mc[s, j], ref_c[s, j], wts[s]) for j in range(4)]
                            + [_skill(me[s], ref_e[s], wts[s])])
        per_fold = np.array(per_fold)
        summary["skill"][m] = {"mean": per_fold.mean(0).tolist(), "folds": per_fold.tolist(),
                               "names": ["Y", "J", "H", "logSFR", "joint_energy"]}
    base = np.array(summary["skill"]["forest"]["folds"])
    for m in models:
        diff = np.array(summary["skill"][m]["folds"]) - base
        summary["paired_vs_forest"][m] = {"mean": diff.mean(0).tolist(),
                                          "se": (diff.std(0, ddof=1) / np.sqrt(n_folds)).tolist()}
    mags, lrs = rows.magnitude[idx], rows.log_radius[idx]
    positive = np.all(rows.ratio[idx] > 0, 1)
    meas_col = colors_of(np.where(rows.ratio[idx] > 0, rows.ratio[idx], 1.0))
    for lo, hi in ((19, 23), (23, 24), (24, 25), (25, 27)):
        sel = (mags >= lo) & (mags < hi) & rows.sfr_valid[idx] & positive
        entry = {"n": int(sel.sum()), "Q1": _spearman(meas_col[sel], rows.log_sfr[idx][sel])}
        for m in models:
            dr, ds = z[f"{m}__draw_ratio"], z[f"{m}__draw_sfr"]
            ok = sel & np.all(dr > 0, 1) & np.isfinite(ds)
            entry[m] = _spearman(colors_of(dr[ok]), ds[ok])
        summary["correlations"][f"{lo}-{hi}"] = entry
    for lo, hi in ((19, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 27)):
        sel = (mags >= lo) & (mags < hi)
        v = sel & rows.sfr_valid[idx]
        entry = {"Q1": float(np.average(rows.sfr_class[idx][v] == 0, weights=rows.weight[idx][v]))}
        for m in models:
            entry[m] = float(np.average(z[f"{m}__draw_q"][sel], weights=rows.weight[idx][sel]))
        summary["quenched"][f"{lo}-{hi}"] = entry
    for lo, hi in ((20, 22), (22, 23)):
        sel = (mags >= lo) & (mags < hi) & rows.resolved[idx]
        tb = np.digitize(lrs, np.quantile(lrs[sel], [1 / 3, 2 / 3]))
        entry = {}
        for name, k in (("small", 0), ("mid", 1), ("large", 2)):
            s2 = sel & (tb == k)
            v = s2 & rows.sfr_valid[idx]
            entry[name] = {"Q1": float(np.mean(rows.sfr_class[idx][v] == 0)),
                           **{m: float(np.mean(z[f"{m}__draw_q"][s2])) for m in models}}
        summary["quenched_by_size"][f"{lo}-{hi}"] = entry
    out_path.write_text(json.dumps(summary, indent=1))
    print(json.dumps({m: np.round(summary["skill"][m]["mean"], 4).tolist() for m in models}))
    print(json.dumps({m: np.round(summary["paired_vs_forest"][m]["mean"], 4).tolist() for m in models}))
    print("done", flush=True)


if __name__ == "__main__":
    main()
