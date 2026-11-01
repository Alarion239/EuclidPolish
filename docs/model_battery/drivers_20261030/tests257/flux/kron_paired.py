"""Read-only: paired real-sky comparison 257@15M vs 238 on the same isolated
galaxies (both measured in this folder's run): median per-galaxy difference of
SR/input (Kron VIS) per magnitude range with a paired bootstrap 68% CI, and the
fraction of galaxies where 257@15M keeps more flux. Also 257@15M vs 257@8M."""
import sys
import numpy as np
from astropy.table import Table
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
from euclid_polish.config import Config
from euclid_polish.photometry import uJy_to_electrons
D = "/Users/alarion239/Desktop/EuclidPolish/data/jwst_euclid_overlap/flux_conservation/"
F = "/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/6d444c86-12c6-4a7f-b4eb-7b73fa2e1c66/scratchpad/tests257/flux/"
new = np.load(F + "report_257_15M_238_fluxes.npz"); old = np.load(D + "report_fluxes.npz")
assert np.array_equal(new["object_id"], old["object_id"])
ids, mag, lr = new["object_id"], new["vis_mag"].astype(float), new["lr:kron_vis"].astype(float)
t = Table.read(D + "mer_nexus.fits"); row = {o: i for i, o in enumerate(np.asarray(t["object_id"]))}
take = np.array([row[o] for o in ids]); conv = uJy_to_electrons(1.0, Config.get_band(Config.LR_INPUT_BAND_NAMES[0]))
col = lambda n: np.asarray(t[n].filled(np.nan) if hasattr(t[n], "filled") else t[n], float)[take] * conv
cat, err = col("flux_detection_total"), col("fluxerr_detection_total")
keep = (np.abs(lr - cat) <= 0.5 * err) & (err > 0) & (cat > 0)
r = {"257@15M": new["milestone:member_257@15000000:kron_vis"] / lr, "238": new["universal:member_238:kron_vis"] / lr,
     "257@8M": old["milestone:member_257@8000000:kron_vis"] / lr, "238 old": old["production:kron_vis"] / lr}
print("238 new vs old run, max |rel diff| of SR flux on isolated:",
      np.nanmax(np.abs(r["238"][keep] / r["238 old"][keep] - 1)))
rng = np.random.default_rng(1)
RANGES = [(21.0, 23.0), (23.0, 24.0), (24.0, 24.5), (24.5, 25.0), (21.0, 25.01)]
def paired(a, b):
    out = []
    for lo, hi in RANGES:
        s = keep & (mag >= lo) & (mag < hi) & np.isfinite(a) & np.isfinite(b)
        d = a[s] - b[s]
        meds = np.median(rng.choice(d, (4000, d.size)), axis=1)
        out.append(f"{np.median(d):+.3f} [{np.percentile(meds, 16):+.3f},{np.percentile(meds, 84):+.3f}] {np.mean(d > 0):.0%}>0")
    return out
print(f"{'pair':18s} " + " ".join(f"{a:.1f}-{b:.1f}".center(30) for a, b in RANGES))
for a, b in (("257@15M", "238"), ("257@15M", "257@8M")):
    print(f"{a + ' - ' + b:18s} " + " ".join(x.ljust(30) for x in paired(r[a], r[b])))
