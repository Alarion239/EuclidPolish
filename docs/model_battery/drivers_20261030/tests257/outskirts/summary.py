"""Outskirt injection test: per-ring recovery tables, old-vs-new check, paired field bootstrap.

python summary.py  → prints tables, writes runs/summary.json
recovery = Σ_ring [rebin(SR(noise + K⊛truth)) − rebin(SR(noise))] / Σ_ring truth   (galaxies, m > 19.5)
host 0 = Gaussian noise at REAL Euclid σ + lag-1 correlation, host 1 = TRAINING (v5) σ + correlation.
"""
import json
import os

import numpy as np

W = "/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/6d444c86-12c6-4a7f-b4eb-7b73fa2e1c66/scratchpad/tests257/outskirts/runs"
OLD = "/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/d7b359b4-16d6-4f7f-aa2b-959d682a91bd/scratchpad/lrcheck"
BAND = ["VIS", "Y", "J", "H"]
HOST = ["real-like noise", "training-like noise"]
# (key, label, new-run file stem or None, old 10-28 file or None)
MODELS = [("257", "257 @ 15M", "milestone_member_257_15000000", None),
          ("238", "238 production", "member_238", "member_238"),
          ("228", "228 noise jitter", "member_228", "member_228"),
          ("249", "249 (257 recipe, 200k)", "member_249", "member_249"),
          ("229", "229 (228 twin, no jitter)", None, "member_229"),
          ("258", "258 galaxy blur", None, "member_258")]
B_RINGS = [(2, "1-1.5"), (3, "1.5-2"), (4, "2-2.5"), (5, "2.5-3")]
NBOOT = 4000


def rec(acc):
    """(2 hosts, nr, 4 bands) recovered fraction."""
    return acc[:, 0] / acc[:, 1]


def load(path):
    return dict(np.load(path)) if os.path.exists(path) else None


def main():
    out = {"rings": None, "models": {}, "check": {}, "boot": {}}
    new, old, pf = {}, {}, {}
    for key, lab, stem, oldname in MODELS:
        if stem:
            new[key] = load(f"{W}/deck_{stem}.npz")
            pf[key] = load(f"{W}/pf_{stem}.npz")
        if oldname:
            old[key] = load(f"{OLD}/deck_{oldname}.npz")
    r = next(v for v in list(new.values()) + list(old.values()) if v is not None)["r_edges"]
    names = [f"{r[i]:g}-{r[i + 1]:g}" for i in range(len(r) - 1)]
    out["rings"] = names

    # ---- tables
    for h in range(2):
        print(f"\n=== recovered fraction, 4-band mean — {HOST[h]}   (r / r_Kron)")
        print(f"{'model':34s}" + "".join(f"{n:>9s}" for n in names))
        for key, lab, stem, oldname in MODELS:
            for tag, src in (("new", new.get(key)), ("10-28", old.get(key))):
                if src is None:
                    continue
                v = rec(src["acc"])[h].mean(-1)
                print(f"{lab + ' [' + tag + ']':34s}" + "".join(f"{x:9.3f}" for x in v))
                out["models"].setdefault(key, {}).setdefault(tag, {})[HOST[h]] = {
                    "mean4": v.tolist(), "per_band": rec(src["acc"])[h].T.tolist()}
    for h in range(2):
        print(f"\n=== per band at 1-1.5 / 1.5-2 / 2-2.5 / 2.5-3 r_Kron — {HOST[h]}")
        print(f"{'model':34s}" + "".join(f"{b:>26s}" for b in BAND))
        for key, lab, stem, oldname in MODELS:
            src = new.get(key) or old.get(key)
            tag = "new" if new.get(key) is not None else "10-28"
            v = rec(src["acc"])[h]
            print(f"{lab + ' [' + tag + ']':34s}" + "".join(
                "   " + " ".join(f"{v[i, b]:5.2f}" for i, _ in B_RINGS) for b in range(4)))

    # ---- old vs new (same code, same model, same noise seeds → should agree)
    print("\n=== comparability: new run vs 10-28 deck file")
    for key in new:
        if new.get(key) is None or old.get(key) is None:
            continue
        a, b = new[key]["acc"], old[key]["acc"]
        d_acc = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-9)))
        d_rec = float(np.max(np.abs(rec(a) - rec(b))))
        d_st = float(np.max(np.abs(new[key]["stamps"] - old[key]["stamps"])))
        print(f"{key}: max rel |Δacc| {d_acc:.2e}   max |Δrecovery| {d_rec:.2e}   max |Δstamp| {d_st:.2e} e⁻")
        out["check"][key] = {"max_rel_dacc": d_acc, "max_drec": d_rec, "max_dstamp": d_st}
    for key in pf:
        if pf.get(key) is not None and new.get(key) is not None:
            same = np.array_equal(pf[key]["acc"], new[key]["acc"])
            close = np.allclose(pf[key]["acc_f"].sum(0), pf[key]["acc"], rtol=1e-9, atol=1e-9)
            print(f"{key}: per-field rerun acc identical to deck rerun: {same}; Σ acc_f == acc: {close}")
            out["check"].setdefault(key, {})["pf_identical"] = bool(same)

    # ---- paired field bootstrap (same resampled fields for every model)
    have = [k for k in pf if pf.get(k) is not None]
    if have:
        n = pf[have[0]]["acc_f"].shape[0]
        rng = np.random.default_rng(12345)
        idx = rng.integers(0, n, size=(NBOOT, n))
        boot = {}
        for k in have:
            af = pf[k]["acc_f"]                                   # (N, 2, 2, nr, 4)
            s = af[idx].sum(1)                                    # (NBOOT, 2, 2, nr, 4)
            boot[k] = (s[:, :, 0] / s[:, :, 1]).mean(-1)          # (NBOOT, 2, nr) 4-band mean
        print("\n=== field bootstrap (40 fields, paired), 4-band mean recovery: median [16–84%]")
        for h in range(2):
            print(f"-- {HOST[h]}")
            for i, rn in B_RINGS:
                cells = []
                for k in have:
                    q = np.percentile(boot[k][:, h, i], [16, 50, 84])
                    cells.append(f"{k} {q[1]:.2f} [{q[0]:.2f},{q[2]:.2f}]")
                    out["boot"].setdefault(k, {}).setdefault(HOST[h], {})[rn] = q.tolist()
                print(f"  {rn:7s} " + "   ".join(cells))
            for a, b in (("257", "238"), ("257", "249"), ("257", "228"), ("238", "249")):
                if a in boot and b in boot:
                    cells = []
                    for i, rn in B_RINGS:
                        d = boot[a][:, h, i] - boot[b][:, h, i]
                        q = np.percentile(d, [2.5, 50, 97.5])
                        cells.append(f"{rn} {q[1]:+.3f} [{q[0]:+.3f},{q[2]:+.3f}]")
                        out["boot"].setdefault(f"{a}-{b}", {}).setdefault(HOST[h], {})[rn] = q.tolist()
                    print(f"  Δ {a}−{b} (95%): " + "  ".join(cells))
    json.dump(out, open(f"{W}/summary.json", "w"), indent=1)
    print("\nwrote", f"{W}/summary.json")


if __name__ == "__main__":
    main()
