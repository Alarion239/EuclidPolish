"""Read-only: res_table.py (session realsky) pointed at the re-run dir — per-band
½-coherence (asinh_euclid, 1σ knee) raw AND noise-corrected, knee scan 0.3σ/3σ,
P8-16 = mean over 8<=period<16 px of (gain/coherence)^2, and the vs-production
comparison in sharper/blurrier wording; old (pre-fix) values alongside."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "jwst")
OLD = "/Users/alarion239/Desktop/EuclidPolish/data/jwst_euclid_overlap/milestones"
BANDS = ("VIS", "Y_E", "J_E", "H_E")
SPECS = ["production", "milestone:member_257@15000000", "universal:member_227"]


def entry(v):
    a = v["asinh_euclid"]
    period = np.array(a["period_px"])
    coh = np.array([np.nan if x is None else x for x in a["coherence"]])
    nc = np.array([np.nan if x is None else x for x in a["coherence_noise_corrected"]])
    gain = np.array([np.nan if x is None else x for x in a["gain"]])
    ratio = (gain / np.maximum(coh, 1e-6)) ** 2
    sel = (period >= 8) & (period < 16)
    scan = {round(x["multiple"], 2): (x["resolution_coherence"] or {}).get("arcsec") for x in v["knee_scan"]}
    fp = 1 - (coh[4] / nc[4]) ** 2
    return dict(code="post-fix" if fp < 0.05 else "PRE-FIX nc", fp=float(fp), windows=a["windows"],
                r1=(a["resolution_coherence"] or {}).get("arcsec"),
                nc1=(a["resolution_coherence_noise_corrected"] or {}).get("arcsec"),
                r03=scan.get(0.3), r3=scan.get(3.0), p816=float(np.nanmean(ratio[sel])))


def table(root):
    res = {}
    for band in BANDS:
        p = os.path.join(root, f"resolution_{band}.json")
        if not os.path.isfile(p):
            continue
        J = json.load(open(p))
        for s in SPECS:
            if s in J["models"]:
                res.setdefault(s, {})[band] = entry(J["models"][s])
    return res


def words(m):
    return f"{abs(1 - m) * 100:.1f}% {'sharper' if m < 1 else 'blurrier'}"


new, old = table(ROOT), table(OLD)
for s, b in new.items():
    print(s)
    for band, e in b.items():
        o = old.get(s, {}).get(band, {})
        print(f"   {band:4s} {e['code']:9s} fp={e['fp']:.4f} win={e['windows']}  1σ raw {e['r1']} (old {o.get('r1')})"
              f"  1σ nc {e['nc1']} (old pre-fix nc {o.get('nc1')})  0.3σ {e['r03']} (old {o.get('r03')})"
              f"  3σ {e['r3']} (old {o.get('r3')})  P8-16 {e['p816']:.4f} (old {o.get('p816', float('nan')):.4f})")
    ps = [b[x]["p816"] for x in BANDS if x in b]
    print(f"   P8-16 band mean {np.mean(ps):.4f}")
print("\nvs production (mean over 4 bands of the per-band ratio):")
for s in new:
    if s == "production" or "production" not in new:
        continue
    P = new["production"]
    for key, name in (("r1", "1σ raw"), ("nc1", "1σ noise-corr"), ("r03", "0.3σ raw"), ("r3", "3σ raw")):
        rs = [new[s][b][key] / P[b][key] for b in BANDS if b in new[s] and new[s][b][key] and P[b][key]]
        print(f"   {s:32s} {name:14s} {words(np.mean(rs)):16s} per band {[round(x, 4) for x in rs]}")
    ps = [new[s][b]["p816"] / P[b]["p816"] for b in BANDS if b in new[s]]
    print(f"   {s:32s} P8-16 ratio {np.mean(ps):.3f} {[round(x, 3) for x in ps]}")
json.dump({"new": new, "old_milestones_dir": old}, open(os.path.join(HERE, "res_table_new.json"), "w"), indent=1)
