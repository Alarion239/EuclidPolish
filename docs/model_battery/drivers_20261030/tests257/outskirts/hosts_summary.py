"""Host injection (real NEXUS tiles vs synthetic test fields): pooled host m > 19.5 recovery per r/r_Kron ring,
old-vs-new check against the 10-28 inject_member_*.npz, paired field bootstrap.   python hosts_summary.py
"""
import json
import os

import numpy as np

W = "/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/6d444c86-12c6-4a7f-b4eb-7b73fa2e1c66/scratchpad/tests257/outskirts/runs"
U = "/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/d1a24d4c-2869-4369-a6ae-08d92714c971/scratchpad/underflux/out"
MODELS = [("257", "milestone_member_257_15000000", None), ("238", "member_238", "member_238"),
          ("249", "member_249", "member_249"), ("228", "member_228", None), ("229", "member_229", None)]
HOSTS = ["real NEXUS tiles", "synthetic test fields"]
RINGS = [(2, "1-1.5"), (3, "1.5-2"), (4, "2-2.5"), (5, "2.5-3")]
BAND = ["VIS", "Y", "J", "H"]


def pooled(d, host):
    """host m > 19.5 (mag bins 2+) → (ring, band) recovery."""
    return d[f"{host}_rec"][2:].sum(0) / d[f"{host}_tru"][2:].sum(0)


out = {"models": {}, "check": {}, "boot": {}}
new = {k: dict(np.load(f"{W}/hosts_{s}.npz")) for k, s, _ in MODELS if os.path.exists(f"{W}/hosts_{s}.npz")}
old = {k: dict(np.load(f"{U}/inject_{o}.npz")) for k, _, o in MODELS if o and os.path.exists(f"{U}/inject_{o}.npz")}
r = next(iter(new.values()))["r_edges"]
names = [f"{r[i]:g}-{r[i + 1]:g}" for i in range(len(r) - 1)]
for h, host in enumerate(("real", "syn")):
    print(f"\n=== {HOSTS[h]}: recovered fraction, host m > 19.5, 4-band mean   (r / r_Kron)")
    print(f"{'model':16s}" + "".join(f"{n:>9s}" for n in names))
    for k, _, _ in MODELS:
        for tag, src in (("new", new.get(k)), ("10-28", old.get(k))):
            if src is None:
                continue
            v = pooled(src, host)
            print(f"{k + ' [' + tag + ']':16s}" + "".join(f"{x:9.3f}" for x in v.mean(-1)))
            out["models"].setdefault(k, {}).setdefault(tag, {})[HOSTS[h]] = {
                "mean4": v.mean(-1).tolist(), "per_band": v.T.tolist()}
    print("   per band at 1-1.5 / 1.5-2 / 2-2.5 / 2.5-3:")
    for k in new:
        v = pooled(new[k], host)
        print(f"   {k:6s}" + "".join("   " + BAND[b] + " " + " ".join(f"{v[i, b]:4.2f}" for i, _ in RINGS)
                                 for b in range(4)))
print("\n=== comparability vs 10-28 inject_member_*.npz")
for k in new:
    if k in old:
        d = max(float(np.max(np.abs(new[k][f"{h}_{q}"] - old[k][f"{h}_{q}"]))) for h in ("real", "syn")
                for q in ("rec", "tru"))
        print(f"{k}: max |Δ| over rec/tru sums {d:.3e}")
        out["check"][k] = d
for k in new:
    a = new[k]["acc_f"].sum(0)
    for h, host in enumerate(("real", "syn")):
        assert np.allclose(a[h, 0], new[k][f"{host}_rec"][2:].sum(0)), (k, host)
rng = np.random.default_rng(12345)
n = next(iter(new.values()))["acc_f"].shape[0]
idx = rng.integers(0, n, size=(4000, n))
boot = {}
for k in new:
    s = new[k]["acc_f"][idx].sum(1)
    boot[k] = (s[:, :, 0] / s[:, :, 1]).mean(-1)              # (B, host, ring)
print("\n=== field bootstrap, 4-band mean: median [16–84%]; paired Δ (95%)")
for h in range(2):
    print(f"-- {HOSTS[h]}")
    for i, rn in RINGS:
        cells = []
        for k in boot:
            q = np.percentile(boot[k][:, h, i], [16, 50, 84])
            cells.append(f"{k} {q[1]:.2f} [{q[0]:.2f},{q[2]:.2f}]")
            out["boot"].setdefault(k, {}).setdefault(HOSTS[h], {})[rn] = q.tolist()
        print(f"  {rn:7s} " + "   ".join(cells))
    for a, b in (("257", "238"), ("257", "249"), ("228", "238"), ("228", "229")):
        if a in boot and b in boot:
            cells = []
            for i, rn in RINGS:
                q = np.percentile(boot[a][:, h, i] - boot[b][:, h, i], [2.5, 50, 97.5])
                cells.append(f"{rn} {q[1]:+.3f} [{q[0]:+.3f},{q[2]:+.3f}]")
                out["boot"].setdefault(f"{a}-{b}", {}).setdefault(HOSTS[h], {})[rn] = q.tolist()
            print(f"  Δ {a}−{b}: " + "  ".join(cells))
json.dump(out, open(f"{W}/hosts_summary.json", "w"), indent=1)
print("wrote", f"{W}/hosts_summary.json")
