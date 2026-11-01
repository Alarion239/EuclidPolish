"""Read-only: synthetic Kron VIS SR/truth and SR/input (a>=18 px 'big' apertures)
binned by input VIS magnitude, from another session's syn_flux2.json."""
import json, sys
import numpy as np
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
from euclid_polish.config import Config
from euclid_polish.photometry import electrons_to_ab_mag
S = "/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish--claude-worktrees-pixel-input-knee-dependency-efc29d/ab73eb83-3218-4250-ba41-4de98db6e1f6/scratchpad/syn_flux2.json"
rows = json.load(open(S))
vis = Config.get_band("VIS")
specs = ["universal:member_249", "universal:member_218", "universal:member_227", "production", "milestone:member_257@8000000"]
mag = np.array([electrons_to_ab_mag(max(r["truth"], 1e-9), vis) for r in rows])
lr = np.array([r["lr"] for r in rows]); tr = np.array([r["truth"] for r in rows])
rng = np.random.default_rng(0)
def med(x):
    x = x[np.isfinite(x)]
    if x.size == 0: return "  -  "
    b = np.median(rng.choice(x, (2000, x.size)), axis=1)
    lo, hi = np.percentile(b, [16, 84])
    return f"{np.median(x):.3f}[{lo:.3f},{hi:.3f}]n{x.size}"
R = [(19, 23), (23, 24), (24, 25), (25, 26)]
print("truth-mag bins", R)
print("lr/truth  ", [med(lr[(mag >= a) & (mag < b)] / tr[(mag >= a) & (mag < b)]) for a, b in R])
for s in specs:
    sr = np.array([r[s + ""] for r in rows])
    print(f"{s:30s} SR/truth", [med(sr[(mag >= a) & (mag < b)] / tr[(mag >= a) & (mag < b)]) for a, b in R])
    print(f"{'':30s} SR/input", [med(sr[(mag >= a) & (mag < b)] / lr[(mag >= a) & (mag < b)]) for a, b in R])
