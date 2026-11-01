"""Tables from stats/*.npz (no TF).  python summarize.py > summary.txt"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
STATS = os.path.join(HERE, "stats")
P = np.load(os.path.join(HERE, "prep.npz"))
META = json.load(open(os.path.join(HERE, "prep.json")))
ORDER = ["milestone_member_257_15000000", "universal_member_238", "universal_member_227",
         "milestone_member_257_1000000", "milestone_member_257_8000000", "truth", "hr", "lr", "jwst", "white"]
NAME = {"milestone_member_257_15000000": "257@15M", "universal_member_238": "238 (prod)",
        "universal_member_227": "227", "milestone_member_257_1000000": "257@1M",
        "milestone_member_257_8000000": "257@8M", "truth": "truth (target PSF)", "hr": "HR record",
        "lr": "LR input (cubic x2)", "jwst": "JWST F200W", "white": "white noise"}
BANDS = ["VIS", "Y", "J", "H"]
RNG = np.random.default_rng(0)


def load(slug, dom):
    path = os.path.join(STATS, f"{slug}__{dom}.npz")
    return dict(np.load(path)) if os.path.isfile(path) else None


def boot(per_unit_hits, per_unit_n, n=2000):
    """95 % bootstrap CI over tiles/fields of a pooled share."""
    k = len(per_unit_n)
    idx = RNG.integers(0, k, size=(n, k))
    shares = per_unit_hits[idx].sum(1) / np.maximum(per_unit_n[idx].sum(1), 1)
    return np.percentile(shares, [2.5, 97.5])


def share(z, valid, thr, bands=None):
    """z (U, ..., C), valid (U, ...) -> pooled share, per-band shares, CI of the pooled share."""
    U = z.shape[0]
    zz = z.reshape(U, -1, z.shape[-1])
    vv = np.broadcast_to(valid.reshape(U, -1, 1), zz.shape)
    b = list(range(zz.shape[-1])) if bands is None else bands
    hits = ((zz[..., b] >= thr) & vv[..., b]).sum((1, 2))
    n = vv[..., b].sum((1, 2))
    per_band = [float(((zz[..., c] >= thr) & vv[..., c]).sum() / max(vv[..., c].sum(), 1)) for c in range(zz.shape[-1])]
    return float(hits.sum() / max(n.sum(), 1)), per_band, boot(hits, n)


def pct(x):
    return f"{100 * x:5.1f}"


def standard(dom, key):
    print(f"\n=== {dom.upper()} protocol {key}: share of 64-px windows with z>=3 (z>=6); pooled over 4 bands "
          f"[95% CI over {'tiles' if dom == 'real' else 'fields'}]; per band VIS/Y/J/H")
    for slug in ORDER:
        s = load(slug, dom)
        if s is None:
            continue
        z = s[key]
        valid = s["pvalid32"] if (key == "pz32" and slug == "jwst") else np.ones(z.shape[:-1], bool)
        p3, b3, ci = share(z, valid, 3.0)
        p6, b6, _ = share(z, valid, 6.0)
        nwin = int(valid.sum()) * z.shape[-1]
        print(f"{NAME[slug]:22s} z>=3 {pct(p3)}% [{pct(ci[0])},{pct(ci[1])}]  z>=6 {pct(p6)}%   "
              f"per band z>=3 {'/'.join(pct(v).strip() for v in b3)}  z>=6 {'/'.join(pct(v).strip() for v in b6)}"
              f"  (n={nwin})")


def outskirt_mask(dom, variant):
    """Outskirt windows (U, N) of a domain and variant (see the module docstring)."""
    out = P["out_real" if dom == "real" else "out_syn"]
    if dom == "real":
        Q = np.load(os.path.join(HERE, "prep_cat.npz"))
        if variant in ("clean", "strict"):
            out = out & ~Q["star_near"] & Q["gal_in"]
        if variant == "strict":
            jw, lr = load("jwst", "real"), load("lr", "real")
            out = out & jw["valid"] & (jw["mean"][..., 0] >= 0.5 * lr["mean"][..., 3])
    else:
        if variant == "truth-light":
            tr, lr = load("truth", "syn"), load("lr", "syn")
            out = out & (tr["mean"][..., 0] >= 0.5 * lr["mean"][..., 0])
    return out


def outskirts(dom, variant="clean"):
    out = outskirt_mask(dom, variant)
    cls = P["cls_real" if dom == "real" else "cls_syn"]
    sky = (cls[..., 1] <= 4) & P["inner"][None]
    if dom == "real":
        sky = sky & ~np.load(os.path.join(HERE, "prep_cat.npz"))["star_near"]
    lrs = load("lr", dom)
    sig_sr = np.asarray(META["sigma_lr_real" if dom == "real" else "sigma_lr_syn"]) / 4.0
    print(f"\n=== {dom.upper()} OUTSKIRT windows [{variant}] (stride-32 64-px windows inside the 32-px border; >=50% of the "
          f"footprint with the smoothed LR VIS in [1,10] sigma_s and max <= 30 sigma_s): n = {int(out.sum())} "
          f"in {int((out.sum(1) > 0).sum())} {'tiles' if dom == 'real' else 'fields'}; sky windows (max <= 4 sigma_s) "
          f"n = {int(sky.sum())}")
    print("model                  outskirt z>=3 pooled [CI]  per band VIS/Y/J/H | median z VIS/Y/J/H | "
          "z>=3 asinh(0.2 sig) pooled | amp/light VIS/Y/J/H (median) | amp/sigma_SR VIS/H | sky z>=3 pooled")
    light = lrs["mean"] if lrs is not None else None
    for slug in ORDER:
        s = load(slug, dom)
        if s is None or slug == "white":
            continue
        valid = out & s["valid"]
        z = s["z_e"]
        p3, b3, ci = share(z, valid, 3.0)
        pa, _ba, _ = share(s["z_a"], valid, 3.0)
        med = [float(np.median(z[..., c][valid])) for c in range(z.shape[-1])]
        amp = s["amp_e"]
        if light is not None and z.shape[-1] == 4:
            rel = [float(np.median(amp[..., c][valid] / light[..., c][valid])) for c in range(4)]
            relsig = [float(np.median(amp[..., c][valid])) / sig_sr[c] for c in (0, 3)]
        elif light is not None:                                       # JWST: H light
            rel = [float(np.median(amp[..., 0][valid] / light[..., 3][valid]))]
            relsig = [float(np.median(amp[..., 0][valid])) / sig_sr[3]]
        else:
            rel, relsig = [np.nan], [np.nan]
        ps, _, _ = share(z, sky & s["valid"], 3.0)
        print(f"{NAME[slug]:22s} {pct(p3)}% [{pct(ci[0])},{pct(ci[1])}]  {'/'.join(pct(v).strip() for v in b3):22s} | "
              f"{'/'.join(f'{v:.2f}' for v in med):22s} | {pct(pa)}% | "
              f"{'/'.join(f'{100 * v:.2f}%' for v in rel):30s} | {'/'.join(f'{v:.3f}' for v in relsig):12s} | {pct(ps)}%"
              f"  (n={int(valid.sum())})")
        if light is not None and z.shape[-1] == 4:
            hit = [valid & (z[..., c] >= 3) for c in range(4)]
            relhit = [float(np.median(amp[..., c][hit[c]] / light[..., c][hit[c]])) if hit[c].any() else np.nan
                      for c in range(4)]
            p90 = [float(np.percentile(z[..., c][valid], 90)) for c in range(4)]
            print(f"{'':22s}   z p90 VIS/Y/J/H {'/'.join(f'{v:.2f}' for v in p90)};  amp/light in the z>=3 windows "
                  f"{'/'.join(f'{100 * v:.1f}%' for v in relhit)};  median amp (e-) VIS/H "
                  f"{np.median(amp[..., 0][valid]):.4f}/{np.median(amp[..., 3][valid]):.4f};  median light (e-/px) VIS/H "
                  f"{np.median(light[..., 0][valid]):.3f}/{np.median(light[..., 3][valid]):.3f}")


def main():
    for dom, keys in (("real", ["pz32"]), ("syn", ["pz0", "pz32"])):
        for key in keys:
            standard(dom, key)
    outskirts("real", "clean")
    outskirts("real", "strict")
    outskirts("real", "lr-only")
    outskirts("syn", "truth-light")
    outskirts("syn", "lr-only")


if __name__ == "__main__":
    main()
