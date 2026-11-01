"""Read-only: Kron VIS SR/input medians per magnitude range from the saved
flux_conservation npz reports, with bootstrap 68% CIs (isolation as in
scripts/plot_flux_conservation.py).

Copy of realsky/kron_bins.py with the report list taken from argv (the first
report sets the isolated sample and the input), the pull vs the MER catalogue
added per magnitude range, and a check that every report re-measured the same
input fluxes.

    python kron_bins.py <report_fluxes.npz> [<more_fluxes.npz> ...]
"""
import sys
import numpy as np
from astropy.table import Table
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
from euclid_polish.config import Config
from euclid_polish.photometry import uJy_to_electrons

D = "/Users/alarion239/Desktop/EuclidPolish/data/jwst_euclid_overlap/flux_conservation/"
paths = sys.argv[1:]
first = np.load(paths[0])
ids = np.asarray(first["object_id"]); mag = np.asarray(first["vis_mag"], float); lr = np.asarray(first["lr:kron_vis"], float)
models = {}
for p in paths:
    z = np.load(p)
    idx = {o: i for i, o in enumerate(np.asarray(z["object_id"]))}
    order = np.array([idx.get(o, -1) for o in ids])
    lr_here = np.asarray(z["lr:kron_vis"], float)[order]
    same = np.allclose(lr_here[order >= 0], lr[order >= 0], rtol=1e-9, atol=0)
    print(f"{p.rsplit('/', 1)[-1]}: {np.sum(order >= 0)}/{ids.size} ids matched, input Kron identical to first report: {same}"
          f" (max |rel diff| {np.max(np.abs(lr_here[order >= 0] / lr[order >= 0] - 1)):.2e})")
    for n in z.files:
        spec, _, key = n.rpartition(":")
        if key != "kron_vis" or spec == "lr" or spec in models:
            continue
        v = np.asarray(z[n], float)[order]; v[order < 0] = np.nan
        models[spec] = v
t = Table.read(D + "mer_nexus.fits")
row = {o: i for i, o in enumerate(np.asarray(t["object_id"]))}
take = np.array([row[o] for o in ids])
conv = uJy_to_electrons(1.0, Config.get_band(Config.LR_INPUT_BAND_NAMES[0]))
def col(n):
    c = t[n]; c = c.filled(np.nan) if hasattr(c, "filled") else c
    return np.asarray(c, float)[take] * conv
cat, err = col("flux_detection_total"), col("fluxerr_detection_total")
keep = (np.abs(lr - cat) <= 0.5 * err) & (err > 0) & (cat > 0)
print("isolated", keep.sum(), "mag range", np.nanmin(mag[keep]), np.nanmax(mag[keep]))
rng = np.random.default_rng(0)
RANGES = [(21.0, 23.0), (23.0, 24.0), (24.0, 24.5), (24.5, 25.0), (21.0, 25.01)]
def boot(x, B=4000):
    x = x[np.isfinite(x)]
    if x.size == 0: return np.nan, np.nan, np.nan, 0
    meds = np.median(rng.choice(x, (B, x.size), replace=True), axis=1)
    return np.median(x), *np.percentile(meds, [16, 84]), x.size
print("\nSR/input Kron VIS, median [68% bootstrap CI]")
print(f"{'model':34s} " + " ".join(f"{a:.1f}-{b:.1f}".center(24) for a, b in RANGES))
for spec, sr in models.items():
    r = sr / lr
    out = []
    for a, b in RANGES:
        s = keep & (mag >= a) & (mag < b) & np.isfinite(r)
        m, lo, hi, n = boot(r[s])
        out.append(f"{m:.3f} [{lo:.3f},{hi:.3f}] n{n:<4d}")
    print(f"{spec:34s} " + " ".join(out))
# pull vs catalogue in sigma: per range (median [68% CI]), overall (median, MAD) and faint
print("\npull (SR - MER) / MER error, median [68% bootstrap CI]")
print(f"{'model':34s} " + " ".join(f"{a:.1f}-{b:.1f}".center(24) for a, b in RANGES))
for spec, sr in [("input", lr), *models.items()]:
    pull = (sr - cat) / err
    out = []
    for a, b in RANGES:
        s = keep & (mag >= a) & (mag < b) & np.isfinite(pull)
        m, lo, hi, n = boot(pull[s])
        out.append(f"{m:+.2f} [{lo:+.2f},{hi:+.2f}] n{n:<4d}")
    print(f"{spec:34s} " + " ".join(out))
print()
for spec, sr in models.items():
    pull = (sr - cat) / err
    s = keep & np.isfinite(pull)
    f = s & (mag >= 24.5)
    print(f"{spec:34s} pull median {np.median(pull[s]):+.2f} (MAD {np.median(np.abs(pull[s]-np.median(pull[s]))):.2f}); faint>=24.5 median {np.median(pull[f]):+.2f}")
pl = (lr - cat) / err
print("input pull median", np.median(pl[keep]))
