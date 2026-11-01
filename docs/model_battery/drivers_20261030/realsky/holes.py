"""Read-only: hole % and flux ratio per model from Sky > Compare experiment records."""
import json, sys
import numpy as np
R = "/Users/alarion239/Desktop/EuclidPolish/data/euclid_inference/experiments/records/"
for rec in sys.argv[1:]:
    d = json.load(open(R + rec))
    print("=====", rec, d.get("label"), "tiles", len(d["tiles"]))
    print(f"{'model':32s} {'hole_mean':>9s} " + " ".join(f"{b+'_hole':>9s}" for b in ("VIS", "Y_E", "J_E", "H_E"))
          + " " + " ".join(f"{b+'_fr':>8s}" for b in ("VIS", "Y_E", "J_E", "H_E")) + "  R<0.5%  R<0.8%")
    for m in d["models"]:
        hm, hb, fr, r5, r8 = [], {b: [] for b in ("VIS", "Y_E", "J_E", "H_E")}, {b: [] for b in ("VIS", "Y_E", "J_E", "H_E")}, [], []
        for t in d["tiles"]:
            e = d["results"].get(t, {}).get(m)
            if not e or "metrics" not in e or not e["metrics"]:
                continue
            s = e["metrics"]["summary"]; hm.append(np.nan if s.get("hole_pct_mean") is None else s["hole_pct_mean"]); r5.append(np.nan if s.get("pct_R_lt_0p5") is None else s["pct_R_lt_0p5"]); r8.append(np.nan if s.get("pct_R_lt_0p8") is None else s["pct_R_lt_0p8"])
            for b, v in e["metrics"]["per_band"].items():
                hb[b].append(np.nan if v.get("hole_pct") is None else v["hole_pct"]); fr[b].append(np.nan if v.get("flux_ratio") is None else v["flux_ratio"])
        print(f"{m:32s} {np.nanmean(hm):9.2f} " + " ".join(f"{np.nanmean(hb[b]):9.2f}" for b in hb)
              + " " + " ".join(f"{np.nanmedian(fr[b]):8.3f}" for b in fr) + f"  {np.nanmean(r5):6.2f}  {np.nanmean(r8):6.2f}  (n={len(hm)})")
