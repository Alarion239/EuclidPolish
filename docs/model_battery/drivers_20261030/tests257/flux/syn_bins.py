"""Read-only: synthetic Kron VIS SR/truth and SR/input (a>=18 px 'big' apertures)
binned by truth VIS magnitude.

Copy of realsky/syn_bins.py: reads the old syn_flux2.json (production = 238,
257@8M, ...) and this folder's re-run (257@15M + 238), checks the two runs
measured the same sources (same fields, apertures, input and truth fluxes) and
prints both side by side, plus small-aperture (MER-like Kron) ratios.

    python syn_bins.py <new syn_flux json>
"""
import json, sys
import numpy as np
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
from euclid_polish.config import Config
from euclid_polish.photometry import electrons_to_ab_mag
OLD = "/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish--claude-worktrees-pixel-input-knee-dependency-efc29d/ab73eb83-3218-4250-ba41-4de98db6e1f6/scratchpad/syn_flux2.json"
old = json.load(open(OLD))
new = json.load(open(sys.argv[1]))
vis = Config.get_band("VIS")

# --- same test set? -------------------------------------------------------
print(f"rows: old {len(old)}, new {len(new)}; fields old {len({r['field'] for r in old})}, new {len({r['field'] for r in new})}")
if len(old) == len(new):
    for key in ("field", "a", "lr_big", "truth_big", "lr", "truth", "err"):
        o = np.array([r[key] for r in old], float); n = np.array([r[key] for r in new], float)
        print(f"  {key:10s} identical: {np.array_equal(o, n)}  max |rel diff| {np.max(np.abs(n / o - 1)):.2e}")
    for ok, nk in (("production_big", "universal:member_238_big"), ("production", "universal:member_238")):
        o = np.array([r[ok] for r in old], float); n = np.array([r[nk] for r in new], float)
        rel = np.abs(n / o - 1)
        print(f"  238 old '{ok}' vs new '{nk}': median |rel diff| {np.median(rel):.2e}, max {np.max(rel):.2e}")

R = [(19, 23), (23, 24), (24, 25), (25, 26)]
rng = np.random.default_rng(0)
def med(x):
    x = x[np.isfinite(x)]
    if x.size == 0: return "  -  "
    b = np.median(rng.choice(x, (2000, x.size)), axis=1)
    lo, hi = np.percentile(b, [16, 84])
    return f"{np.median(x):.3f}[{lo:.3f},{hi:.3f}]n{x.size}"

def table(rows, specs, suffix, title):
    mag = np.array([electrons_to_ab_mag(max(r["truth_big"], 1e-9), vis) for r in rows])
    lr = np.array([r["lr" + suffix] for r in rows]); tr = np.array([r["truth" + suffix] for r in rows])
    print(f"\n{title}; truth-mag bins {R}")
    print(f"{'input/truth':38s}", [med(lr[(mag >= a) & (mag < b)] / tr[(mag >= a) & (mag < b)]) for a, b in R])
    for s, tag in specs:
        sr = np.array([r[s + suffix] for r in rows])
        print(f"{tag:26s} SR/truth  ", [med(sr[(mag >= a) & (mag < b)] / tr[(mag >= a) & (mag < b)]) for a, b in R])
        print(f"{'':26s} SR/input  ", [med(sr[(mag >= a) & (mag < b)] / lr[(mag >= a) & (mag < b)]) for a, b in R])

OLD_SPECS = [("production", "old: production (238)"), ("milestone:member_257@8000000", "old: 257@8M")]
NEW_SPECS = [("universal:member_238", "new: 238"), ("milestone:member_257@15000000", "new: 257@15M")]
for suffix, title in (("_big", "BIG Kron apertures (a >= 18 LR px)"), ("", "SMALL apertures (photutils Kron a)")):
    table(old, OLD_SPECS, suffix, "old syn_flux2.json, " + title)
    table(new, NEW_SPECS, suffix, "this run, " + title)
