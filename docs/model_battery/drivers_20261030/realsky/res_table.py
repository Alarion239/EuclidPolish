"""Read-only: per-band ½-coherence (asinh_euclid, 1σ knee, raw) + knee scan at
0.3σ/3σ + P8-16 texture (SR/JWST power, (gain/coh)^2 over 8<=p<16 px), from the
saved JWST resolution reports. Mirrors blur_compare.py of session 726fc9fe."""
import json, os
import numpy as np
ROOT = "/Users/alarion239/Desktop/EuclidPolish/data/jwst_euclid_overlap"
DIRS = ["galaxy_blur", "galaxy_blur_950k", "milestones", "galaxies"]
BANDS = ("VIS", "Y_E", "J_E", "H_E")
SPECS = ["production", "milestone:member_257@1000000", "milestone:member_257@8000000",
         "milestone:member_257@15000000", "milestone:member_258@950000", "milestone:member_258@500000",
         "universal:member_258", "universal:member_255", "universal:member_249", "universal:member_218",
         "universal:member_227", "lr"]
def load(band):
    out = {}
    for d in DIRS:
        p = os.path.join(ROOT, d, f"resolution_{band}.json")
        if not os.path.isfile(p):
            continue
        J = json.load(open(p))
        for k, v in J["models"].items():
            a = v["asinh_euclid"]
            coh = np.array([np.nan if x is None else x for x in a["coherence"]])
            nc = np.array([np.nan if x is None else x for x in a["coherence_noise_corrected"]])
            fp = 1 - (coh[4] / nc[4]) ** 2           # noise-correction fingerprint at ring 5
            out.setdefault(k, (d, v, "post-fix" if fp < 0.05 else "PRE-FIX nc"))
    return out
res = {}
for band in BANDS:
    M = load(band)
    for s in SPECS:
        if s not in M:
            continue
        d, v, code = M[s]
        a = v["asinh_euclid"]
        period = np.array(a["period_px"]); coh = np.array([np.nan if x is None else x for x in a["coherence"]])
        gain = np.array([np.nan if x is None else x for x in a["gain"]])
        ratio = (gain / np.maximum(coh, 1e-6)) ** 2
        sel = (period >= 8) & (period < 16)
        scan = {round(x["multiple"], 2): (x["resolution_coherence"] or {}).get("arcsec") for x in v["knee_scan"]}
        res.setdefault(s, {})[band] = dict(src=d, code=code, r1=(a["resolution_coherence"] or {}).get("arcsec"),
                                           r03=scan.get(0.3), r3=scan.get(3.0), p816=float(np.nanmean(ratio[sel])))
for s, b in res.items():
    print(s)
    for band, e in b.items():
        print(f"   {band:4s} src={e['src']:17s} {e['code']:11s} 1σ {e['r1']}  0.3σ {e['r03']}  3σ {e['r3']}  P8-16 {e['p816']:.4f}")
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "res_table.json"), "w"), indent=1)
# relative to production: sharper/blurrier wording
print("\nvs production (1σ, mean over 4 bands of the ratio):")
for s in res:
    if s == "production" or "production" not in res: continue
    rs = [res[s][b]["r1"] / res["production"][b]["r1"] for b in BANDS if b in res[s] and res[s][b]["r1"] and res["production"][b]["r1"]]
    ps = [res[s][b]["p816"] / res["production"][b]["p816"] for b in BANDS if b in res[s]]
    m = np.mean(rs) if rs else np.nan
    word = f"{abs(1-m)*100:.1f}% {'sharper' if m < 1 else 'blurrier'}"
    print(f"   {s:32s} 1σ res {word:16s} per band {[round(x,4) for x in rs]}  P8-16 ratio {np.mean(ps):.3f} {[round(x,3) for x in ps]}")
