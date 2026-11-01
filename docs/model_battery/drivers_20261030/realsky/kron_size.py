"""Read-only: MER KRON_RADIUS (LR px) distribution of the 815 isolated NEXUS galaxies."""
import sys
import numpy as np
from astropy.table import Table
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
from euclid_polish.config import Config
from euclid_polish.photometry import uJy_to_electrons
D = "/Users/alarion239/Desktop/EuclidPolish/data/jwst_euclid_overlap/flux_conservation/"
z = np.load(D + "report_fluxes.npz")
ids = np.asarray(z["object_id"]); mag = np.asarray(z["vis_mag"], float); lr = np.asarray(z["lr:kron_vis"], float)
t = Table.read(D + "mer_nexus.fits"); row = {o: i for i, o in enumerate(np.asarray(t["object_id"]))}
take = np.array([row[o] for o in ids])
conv = uJy_to_electrons(1.0, Config.get_band(Config.LR_INPUT_BAND_NAMES[0]))
f = lambda n: np.asarray(t[n].filled(np.nan) if hasattr(t[n], "filled") else t[n], float)[take]
cat, err = f("flux_detection_total") * conv, f("fluxerr_detection_total") * conv
keep = (np.abs(lr - cat) <= 0.5 * err) & (err > 0) & (cat > 0)
kr = f("kron_radius")
for a, b in [(20, 23), (23, 24.5), (24.5, 25.3)]:
    s = keep & (mag >= a) & (mag < b)
    print(f"mag {a}-{b}: n={s.sum()} KRON_RADIUS px p10/50/90 = {np.percentile(kr[s], [10, 50, 90]).round(2)}")
print("semimajor_axis p50", np.nanmedian(f("semimajor_axis")[keep]))
