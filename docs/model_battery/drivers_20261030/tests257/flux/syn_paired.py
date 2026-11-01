"""Read-only: paired synthetic comparison per source (same 900 sources):
257@15M - 238 (this run) and 257@15M - 257@8M (old syn_flux2.json, identical
sources) of SR/truth in the big Kron apertures, per truth-magnitude bin."""
import json, sys
import numpy as np
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
from euclid_polish.config import Config
from euclid_polish.photometry import electrons_to_ab_mag
OLD = "/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish--claude-worktrees-pixel-input-knee-dependency-efc29d/ab73eb83-3218-4250-ba41-4de98db6e1f6/scratchpad/syn_flux2.json"
old = json.load(open(OLD)); new = json.load(open(sys.argv[1]))
vis = Config.get_band("VIS")
mag = np.array([electrons_to_ab_mag(max(r["truth_big"], 1e-9), vis) for r in new])
tr = np.array([r["truth_big"] for r in new])
v = {"257@15M": np.array([r["milestone:member_257@15000000_big"] for r in new]) / tr,
     "238": np.array([r["universal:member_238_big"] for r in new]) / tr,
     "257@8M": np.array([r["milestone:member_257@8000000_big"] for r in old]) / tr}
rng = np.random.default_rng(1)
R = [(19, 23), (23, 24), (24, 25), (25, 26)]
for a, b in (("257@15M", "238"), ("257@15M", "257@8M")):
    out = []
    for lo, hi in R:
        s = (mag >= lo) & (mag < hi) & np.isfinite(v[a]) & np.isfinite(v[b])
        d = v[a][s] - v[b][s]
        meds = np.median(rng.choice(d, (4000, d.size)), axis=1)
        out.append(f"{np.median(d):+.3f}[{np.percentile(meds, 16):+.3f},{np.percentile(meds, 84):+.3f}] {np.mean(d > 0):.0%}>0")
    print(f"SR/truth {a} - {b}:", out)
