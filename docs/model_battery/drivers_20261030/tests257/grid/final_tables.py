"""Compact report table + paired bootstrap differences (no TF).

    python final_tables.py > final_tables.txt
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize as S  # noqa: E402

MODELS = ["milestone_member_257_15000000", "universal_member_238", "universal_member_227",
          "milestone_member_257_1000000", "milestone_member_257_8000000"]
REFS = {"real": ["jwst", "lr", "white"], "syn": ["truth", "lr", "white"]}
RNG = np.random.default_rng(11)


def per_unit(z, valid, thr=3.0):
    """hits and counts per tile/field, pooled over bands."""
    U, C = z.shape[0], z.shape[-1]
    zz = z.reshape(U, -1, C)
    vv = np.broadcast_to(valid.reshape(U, -1, 1), zz.shape)
    return ((zz >= thr) & vv).sum((1, 2)).astype(float), vv.sum((1, 2)).astype(float)


def paired(a, b, n=4000):
    """95 % CI of share(a) - share(b), resampling tiles/fields (a, b = (hits, n))."""
    k = len(a[1])
    idx = RNG.integers(0, k, size=(n, k))
    da = a[0][idx].sum(1) / np.maximum(a[1][idx].sum(1), 1)
    db = b[0][idx].sum(1) / np.maximum(b[1][idx].sum(1), 1)
    d = da - db
    return float(np.median(d)), np.percentile(d, [2.5, 97.5])


def main():
    rows = {}
    masks = {"real": S.outskirt_mask("real", "clean"), "syn": S.outskirt_mask("syn", "truth-light")}
    for slug in MODELS + ["jwst", "truth", "lr", "white"]:
        r = {}
        for dom in ("real", "syn"):
            s = S.load(slug, dom)
            if s is None:
                continue
            if dom == "real":
                valid = s["pvalid32"] if slug == "jwst" else np.ones(s["pz32"].shape[:-1], bool)
                h, n = per_unit(s["pz32"], valid)
                r["real3"] = h.sum() / n.sum()
                r["real_units"] = (h, n)
            else:
                v = np.ones(s["pz0"].shape[:-1], bool)
                h, n = per_unit(s["pz0"], v)
                h6, _ = per_unit(s["pz0"], v, 6.0)
                r["syn3"], r["syn6"] = h.sum() / n.sum(), h6.sum() / n.sum()
                r["syn_units"] = (h, n)
            if slug == "white":
                continue
            m = masks[dom] & s["valid"]
            ho, no = per_unit(s["z_e"], m)
            r[f"{dom}_out3"] = ho.sum() / max(no.sum(), 1)
            r[f"{dom}_out_units"] = (ho, no)
            r[f"{dom}_out_med"] = float(np.median(s["z_e"][m]))          # pooled over bands
            r[f"{dom}_out_med_vis"] = float(np.median(s["z_e"][..., 0][m]))
            zz = s["z_e"]
            if zz.shape[-1] == 4:
                vis, h_ = (zz[..., 0] >= 3) & m, (zz[..., 3] >= 3) & m
                r[f"{dom}_vis_and_h"] = float((vis & h_).sum() / max(vis.sum(), 1))
                lr = S.load("lr", dom)
                light = lr["mean"]
                hit = (zz >= 3) & m[..., None]
                r[f"{dom}_amp_hit"] = float(np.median(s["amp_e"][hit] / light[hit])) if hit.any() else np.nan
                sig = np.asarray(S.META["sigma_lr_real" if dom == "real" else "sigma_lr_syn"]) / 4.0
                ampsig = s["amp_e"] / sig
                r[f"{dom}_amp_hit_sig"] = float(np.median(ampsig[hit])) if hit.any() else np.nan
        rows[slug] = r

    def f(x, p=1):
        return "  -  " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{p}f}"

    print("model      | real z>=3 | syn z>=3 / z>=6 | outskirt z>=3 real / syn | outskirt median z (4 bands) real / syn"
          " | VIS-hit also H-hit real/syn | amp/light in z>=3 outskirt windows real/syn | amp/sigma_SR real/syn")
    for slug, r in rows.items():
        print(f"{S.NAME[slug]:18s} | {f(r.get('real3'))}% | {f(r.get('syn3'))}% / {f(r.get('syn6'))}% | "
              f"{f(r.get('real_out3'))}% / {f(r.get('syn_out3'))}% | "
              f"{r.get('real_out_med', np.nan):.2f} / {r.get('syn_out_med', np.nan):.2f} | "
              f"{f(r.get('real_vis_and_h'), 0)}% / {f(r.get('syn_vis_and_h'), 0)}% | "
              f"{f(r.get('real_amp_hit'), 2)}% / {f(r.get('syn_amp_hit'), 2)}% | "
              f"{r.get('real_amp_hit_sig', np.nan):.4f} / {r.get('syn_amp_hit_sig', np.nan):.4f}")
    print("\nPaired bootstrap (tiles / fields resampled; same windows), difference of pooled z>=3 shares, pts [95% CI]:")
    a, b, c = MODELS[0], MODELS[1], MODELS[2]
    for key, lab in (("real_units", "real all"), ("syn_units", "syn all"), ("real_out_units", "real outskirt"),
                     ("syn_out_units", "syn outskirt")):
        for x, y in ((a, b), (c, a), (a, MODELS[3]), (a, MODELS[4])):
            if key in rows.get(x, {}) and key in rows.get(y, {}):
                d, ci = paired(rows[x][key], rows[y][key])
                print(f"  {lab:14s} {S.NAME[x]} - {S.NAME[y]}: {100 * d:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]")


if __name__ == "__main__":
    main()
