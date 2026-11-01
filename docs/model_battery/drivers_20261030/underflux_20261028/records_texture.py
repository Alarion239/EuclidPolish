"""Empty-sky texture of v5 vs v6 dirty_test records and real NEXUS tiles, same mask and stats."""
import numpy as np

from euclid_polish.image.collection import ImageSet
from mer_fair import sky_mask, stats

OUT = "/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/d1a24d4c-2869-4369-a6ae-08d92714c971/scratchpad"
V5 = "data/_fasrc_cache/n/netscratch/lconnor_lab/Lab/abelotserkovtsev/EuclidPolish/data/images/records_v2/dirty_test.tfrecord"
V6 = f"{OUT}/v6records/dirty_test.tfrecord"
BANDS = ["VIS", "Y", "J", "H"]


def table(fields):
    out = np.zeros((4, 3))
    for b in range(4):
        rows = []
        for f in fields:
            x = np.asarray(f[..., b], np.float64)
            base = (x != 0) & np.isfinite(x)
            rows.append(stats(x, sky_mask(x, base))[:3])
        out[b] = np.median(rows, 0)
    return out


v5 = [np.asarray(im.data) for im in ImageSet.read(V5)]
v6 = [np.asarray(im.data) for im in ImageSet.read(V6)]
real = np.load(f"{OUT}/underflux/out/lr.npy", mmap_mode="r")
real = [np.asarray(real[i]) for i in range(0, real.shape[0], 4)]
res = {"real NEXUS": table(real), "v5 records": table(v5), "v6 records": table(v6)}
print("lag1 diag lag2 (median over fields, 3σ source mask)")
for b in range(4):
    print(BANDS[b], "  ".join(f"{k}: " + " ".join(f"{v:.3f}" for v in t[b]) for k, t in res.items()))
