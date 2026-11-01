"""Charts for the weekly-review deck (dark surface, 2x pixel density)."""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

mpl_dark()
plt.rcParams.update({"font.size": 18, "axes.titlesize": 20, "legend.fontsize": 17,
                     "xtick.labelsize": 16, "ytick.labelsize": 16, "axes.labelsize": 18})
D = f"{REPO}/data"
UNI = f"{D}/vis/ensemble/starfull/universal"
BANDS = ["VIS", "Y_E", "J_E", "H_E"]
BNAME = {"VIS": "VIS", "Y_E": "Y", "J_E": "J", "H_E": "H"}
BCOL = {"VIS": "#3987e5", "Y_E": "#199e70", "J_E": "#c98500", "H_E": "#e66767"}
STEPS = [200000, 500000, 1000000, 2000000, 4000000, 8000000]
TEAL = LinearSegmentedColormap.from_list("t", ["#1d4a52", "#2fd1c4", "#c8fff6"])
def mcol(i, n=len(STEPS)):
    return TEAL(0.12 + 0.88 * i / (n - 1))


def fig(w, h):
    return plt.figure(figsize=(w / 100, h / 100), dpi=200)


def glow(ax, x, y, col, lw=3):
    for w, a in ((lw * 5, 0.06), (lw * 3, 0.10), (lw * 1.8, 0.16)):
        ax.plot(x, y, color=col, lw=w, alpha=a, solid_capstyle="round", zorder=2)


def save(f, name):
    for ax in f.axes:
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    f.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(f)
    print("saved", name)


def fmt_steps(x, _=None):
    return f"{x/1e6:g}M"


# ---------------------------------------------------------------- resolution
def res_vs_sb():
    f = fig(1664, 640)
    axs = f.subplots(1, 4, sharey=True)
    series = [("lr", "Euclid input", C["lr"], "--"),
              ("universal:member_249", "249 (200k)", C["m249"], "-"),
              ("production", "238 production (1.9M)", C["prod"], "-"),
              ("milestone:member_257@8000000", "257 @ 8M", C["m257"], "-")]
    for ax, b in zip(axs, BANDS):
        r = json.load(open(f"{D}/jwst_euclid_overlap/galaxies/resolution_{b}.json"))
        for key, lab, col, ls in series:
            ks = r["models"][key]["knee_scan"]
            x = [e["sb"] for e in ks if e["resolution_coherence"]]
            y = [e["resolution_coherence"]["arcsec"] for e in ks if e["resolution_coherence"]]
            ax.plot(x, y, ls, color=col, marker="o", ms=7, label=lab,
                    mec=BG, mew=1.5)
        sb1 = r["meta"].get("euclid_noise_sb")
        if sb1:
            ax.axvline(sb1, color=INK2, lw=1, ls=":")
            ax.text(sb1 + 0.2, 0.105, "1σ/px", color=INK2, fontsize=14, ha="left", va="bottom")
        ax.set_yscale("log")
        ax.set_title(BNAME[b])
        ax.invert_xaxis()
        ax.set_yticks([0.1, 0.2, 0.5, 1, 2])
        ax.set_yticklabels(["0.1″", "0.2″", "0.5″", "1″", "2″"])
        ax.set_ylim(0.09, 3.5)
    axs[0].set_ylabel("½-coherence period vs JWST")
    h, l = axs[0].get_legend_handles_labels()
    f.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    f.supxlabel("surface brightness the stretch is centred on, μ (mag/arcsec²)  — fainter ←", color=INK2, fontsize=18)
    f.tight_layout(rect=(0, 0, 1, 0.9))
    save(f, "c_res_vs_sb.png")


def frc_curve():
    f = fig(1000, 640)
    ax = f.subplots()
    r = json.load(open(f"{D}/jwst_euclid_overlap/galaxies/resolution_H_E.json"))
    for key, lab, col, ls in [("lr", "Euclid input", C["lr"], "--"),
                              ("universal:member_249", "249", C["m249"], "-"),
                              ("production", "238 production", C["prod"], "-"),
                              ("milestone:member_257@8000000", "257 @ 8M", C["m257"], "-")]:
        e = r["models"][key]["asinh_euclid"]
        p = np.array(e["period_px"]) * 0.05
        ax.plot(p, e["coherence"], ls, color=col, label=lab)
    ax.axhline(0.5, color=INK2, lw=1.2, ls=":")
    ax.text(5.5, 0.52, "½", color=INK, fontsize=18)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks([4, 2, 1, 0.5, 0.2, 0.1])
    ax.set_xticklabels(["4″", "2″", "1″", "0.5″", "0.2″", "0.1″"])
    ax.set_xlabel("period (finer →)")
    ax.set_ylabel("coherence with JWST F200W")
    ax.set_title("H band, asinh stretch at the Euclid noise")
    ax.legend(loc="lower left")
    f.tight_layout()
    save(f, "c_frc_curve.png")


# ---------------------------------------------------------------- milestones
def ms_scores():
    ms = json.load(open(f"{UNI}/milestones/member_257.json"))["steps"]
    ix = json.load(open(f"{UNI}/index.json"))["models"]
    f = fig(1664, 660)
    a1, a2 = f.subplots(1, 2, sharey=True)
    for ax, xkey in ((a1, "scored_steps"), (a2, "scored_hours")):
        xs = [ms[str(s)][xkey] for s in STEPS]
        ys = [ms[str(s)]["mean"] for s in STEPS]
        glow(ax, xs, ys, C["m257"])
        ax.plot(xs, ys, "-", color=C["m257"], lw=3, zorder=3, label="257 milestones")
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax.plot(x, y, "o", ms=12, color=mcol(i), mec=BG, mew=2, zorder=4)
        refs = [("member_237", "237", C["m237"]), ("member_249", "249", C["m249"]),
                ("member_238", "238 prod", C["prod"]), ("member_218", "218", C["m218"]),
                ("member_227", "227", C["m227"])]
        for k, lab, col in refs:
            e = ix[k]
            x = e[xkey]
            ax.plot(x, e["mean"], "D", ms=11, color=col, mec=BG, mew=2, zorder=5)
            dx, dy, ha = {"237": (1, 0.2, "center"), "249": (1, -0.42, "center"),
                          "238 prod": (0.86, -0.42, "right"), "218": (1.1, 0.2, "left"),
                          "227": (1, 0.2, "center")}[lab]
            ax.text(x * dx, e["mean"] + dy, lab, color=INK, ha=ha, fontsize=16)
        ax.set_xscale("log")
        ax.set_ylim(58.6, 63.4)
    a1.set_xlabel("training steps")
    a1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt_steps))
    a1.set_xticks([2e5, 5e5, 1e6, 2e6, 4e6, 8e6, 1.55e7])
    a2.set_xlabel("A100 hours")
    a2.set_xticks([1, 2, 5, 10, 20, 40, 70])
    a2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g} h"))
    a1.set_ylabel("synthetic knee-integrated PSNR (dB)")
    a1.set_title("vs steps")
    a2.set_title("vs GPU time")
    f.tight_layout()
    save(f, "c_ms_psnr.png")

    # per band gains relative to 200k
    f = fig(1000, 640)
    ax = f.subplots()
    for bi, b in enumerate(BANDS):
        base = ms["200000"]["integrated"][bi]
        ax.plot(STEPS, [ms[str(s)]["integrated"][bi] - base for s in STEPS], "-o",
                color=BCOL[b], label=BNAME[b], ms=8, mec=BG, mew=1.5)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt_steps))
    ax.set_xticks(STEPS)
    ax.set_xlabel("training steps")
    ax.set_ylabel("Δ integrated PSNR vs 0.2M (dB)")
    ax.set_title("Per-band synthetic gain")
    ax.legend(loc="upper left")
    f.tight_layout()
    save(f, "c_ms_bands.png")

    # learned output knee per milestone
    f = fig(820, 640)
    ax = f.subplots()
    for bi, b in enumerate(BANDS):
        ax.plot(STEPS, [ms[str(s)]["learned_knees"][bi] for s in STEPS], "-o",
                color=BCOL[b], label=BNAME[b], ms=8, mec=BG, mew=1.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt_steps))
    ax.set_xticks(STEPS)
    ax.set_xlabel("training steps")
    ax.set_ylabel("learned output knee (e⁻)")
    ax.set_title("Output stretch knee")
    ax.legend(loc="upper right")
    f.tight_layout()
    save(f, "c_ms_knee.png")


def ms_resolution():
    f = fig(1664, 700)
    axs = f.subplots(1, 2)
    levels = [(1.0, "at the Euclid noise (1σ/px)"), (0.1, "below the noise (0.1σ/px)")]
    for ax, (mult, title) in zip(axs, levels):
        for b in BANDS:
            r = json.load(open(f"{D}/jwst_euclid_overlap/milestones/resolution_{b}.json"))
            def val(key):
                for e in r["models"][key]["knee_scan"]:
                    if e["multiple"] == mult and e["resolution_coherence"]:
                        return e["resolution_coherence"]["arcsec"]
            base = val("milestone:member_257@200000")
            ys = [100 * (base - val(f"milestone:member_257@{s}")) / base for s in STEPS]
            ax.plot(STEPS, ys, "-o", color=BCOL[b], label=BNAME[b], ms=9, mec=BG, mew=1.5)
            for key, mk in [("production", "D"), ("universal:member_227", "s")]:
                pass
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt_steps))
        ax.set_xticks(STEPS)
        ax.set_xlabel("training steps")
        ax.set_title(title)
        ax.set_ylim(-3.6, 5.2)
        ax.text(2.1e5, 4.75, "↑ sharper than 0.2M", color=INK2, fontsize=15)
        ax.text(2.1e5, -3.35, "↓ blurrier than 0.2M", color=INK2, fontsize=15)
    axs[0].set_ylabel("real-sky ½-period vs 257 @ 0.2M (%)")
    h, l = axs[0].get_legend_handles_labels()
    f.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    f.tight_layout(rect=(0, 0, 1, 0.92))
    save(f, "c_ms_res.png")

    # absolute table-like chart: 1σ resolution for every model incl. references
    f = fig(1664, 640)
    axs = f.subplots(1, 4)
    keys = [f"milestone:member_257@{s}" for s in STEPS] + ["universal:member_249", "production",
                                                           "universal:member_218", "universal:member_227"]
    labs = [fmt_steps(s) for s in STEPS] + ["249", "238", "218", "227"]
    cols = [mcol(i) for i in range(6)] + [C["m249"], C["prod"], C["m218"], C["m227"]]
    for ax, b in zip(axs, BANDS):
        r = json.load(open(f"{D}/jwst_euclid_overlap/milestones/resolution_{b}.json"))
        vals = []
        for k in keys:
            v = [e for e in r["models"][k]["knee_scan"] if e["multiple"] == 1.0][0]
            vals.append(v["resolution_coherence"]["arcsec"])
        xs = np.arange(len(keys)) + np.array([0] * 6 + [1] * 4) * 0.8
        ax.plot(xs[:6], vals[:6], "-", color=C["m257"], lw=2, alpha=0.6)
        for x, v, c in zip(xs, vals, cols):
            ax.plot(x, v, "o", ms=14, color=c, mec=BG, mew=2)
        span = max(vals) - min(vals)
        ax.set_ylim(min(vals) - 0.25 * span, max(vals) + 0.25 * span)
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, rotation=60, fontsize=14)
        ax.set_title(BNAME[b])
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.2f}″"))
    axs[0].set_ylabel("½-period at Euclid 1σ (lower = sharper)")
    f.tight_layout()
    save(f, "c_ms_res_bars.png")


# ---------------------------------------------------------------- training log
def train_log():
    txt = open(os.path.join(os.path.dirname(__file__), "..", "data", "train257.log")).read()
    blocks = re.findall(r"Step (\d+)/\d+: loss = ([\d.]+), PSNR\(str/raw\) = ([\d.]+)/[\d.]+ dB, "
                        r"\|g\| avg/max = ([\d.]+)/[\d.]+ \(([\d.]+)s\).*?isotropy penalty ([\d.]+) "
                        r"\(moving average (.*?)\)\n\s+per-knee PSNR \(dB\): (.*?)\n\s+learned output knee \(e⁻\): (.*?)\n",
                        txt)
    step = np.array([int(b[0]) for b in blocks])
    loss = np.array([float(b[1]) for b in blocks])
    psnr = np.array([float(b[2]) for b in blocks])
    sec = np.array([float(b[4]) for b in blocks])
    scales = ["2-3", "3-6", "6-12", "12-24", "24-48", "48-64"]
    iso = np.array([[float(re.search(rf"128px {s} ([\d.]+)", b[6]).group(1)) for s in scales] for b in blocks])
    knees = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    pk = np.array([[float(x.split()[-1]) for x in b[7].split("|")] for b in blocks])
    lk = np.array([[float(x.split()[-1]) for x in b[8].split("|")] for b in blocks])
    print("log rows", len(step), step[-1])

    def smooth(y, n=21):
        out = np.copy(y).astype(float)
        for i in range(len(y)):
            lo, hi = max(0, i - n // 2), min(len(y), i + n // 2 + 1)
            out[i] = np.median(y[lo:hi], axis=0)
        return out

    m = step >= 5000
    # 1. per-knee PSNR gain since 0.5M
    f = fig(1100, 660)
    ax = f.subplots()
    ref = np.median(pk[(step >= 4.5e5) & (step <= 5.5e5)], axis=0)
    cm = plt.get_cmap("plasma")
    for j, k in enumerate(knees):
        ax.plot(step[m], smooth(pk[m, j] - ref[j], 41), color=cm(0.28 + 0.7 * j / 7), lw=2.4,
                label=f"{k:g} e⁻")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xscale("log")
    ax.set_xlim(2.5e5, step[-1] * 1.05)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt_steps))
    ax.set_xticks([3e5, 5e5, 1e6, 2e6, 4e6, 8e6])
    ax.set_xlabel("training step")
    ax.set_ylabel("training-batch PSNR − value at 0.5M (dB)")
    ax.set_title("Per-knee training PSNR (running median)")
    ax.legend(title="loss knee", ncol=2, loc="upper left", fontsize=15, title_fontsize=15)
    f.tight_layout()
    save(f, "c_tr_knees.png")

    # 2. learned output knee
    f = fig(820, 660)
    ax = f.subplots()
    for bi, b in enumerate(BANDS):
        ax.plot(step[m], lk[m, bi], color=BCOL[b], lw=2.6, label=BNAME[b])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt_steps))
    ax.set_xticks([1e4, 1e5, 1e6, 1e7])
    ax.set_xlabel("training step")
    ax.set_ylabel("learned output knee (e⁻)")
    ax.set_title("Output knee, every 5k steps")
    ax.legend(loc="upper right")
    f.tight_layout()
    save(f, "c_tr_outknee.png")

    # 3. isotropy moving averages per scale
    f = fig(1100, 660)
    ax = f.subplots()
    sc = LinearSegmentedColormap.from_list("s", ["#e66767", "#c98500", "#2fd1c4", "#3987e5", "#9085e9"])
    for j, s in enumerate(scales):
        y = smooth(iso[m, j], 21)
        ax.plot(step[m], np.maximum(y, 1e-5), color=sc(j / 5), lw=2.4, label=f"{s} px")
    ax.axhline(0.05, color=INK2, lw=1, ls=":")
    ax.text(1.2e4, 0.058, "tolerance 0.05", color=INK2, fontsize=15)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt_steps))
    ax.set_xticks([1e4, 1e5, 1e6, 1e7])
    ax.set_xlabel("training step")
    ax.set_ylabel("residual anisotropy (moving average)")
    ax.set_title("Isotropy penalty by period band")
    ax.legend(ncol=2, loc="upper right", fontsize=15)
    f.tight_layout()
    save(f, "c_tr_iso.png")

    # 4. loss + speed
    f = fig(1100, 660)
    ax = f.subplots()
    glow(ax, step[m], smooth(psnr[m], 41), C["m257"])
    ax.plot(step[m], smooth(psnr[m], 41), color=C["m257"], lw=2.6)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt_steps))
    ax.set_xticks([1e4, 1e5, 1e6, 1e7])
    ax.set_xlabel("training step")
    ax.set_ylabel("training PSNR, stretched (dB)")
    ax.set_title("Training-batch PSNR (running median)")
    f.tight_layout()
    save(f, "c_tr_psnr.png")
    ms_per = sec / 5.0
    print("ms/step median last 1000 rows", np.median(ms_per[-1000:]), "first", np.median(ms_per[5:200]))
    json.dump({"last_step": int(step[-1]), "ms_per_step": float(np.median(ms_per[-500:])),
               "learned_knee_last": lk[-1].tolist()}, open(os.path.join(OUT, "train_meta.json"), "w"))


# ---------------------------------------------------------------- spectra
def spectra():
    def load(name):
        p = f"{UNI}/milestones/{name}.spectra.json" if "@" in name else f"{UNI}/{name}.spectra.json"
        return json.load(open(p))
    f = fig(1664, 640)
    a1, a2 = f.subplots(1, 2)
    for i, s in enumerate(STEPS):
        sp = load(f"member_257@{s}")
        p = np.array(sp["radial"]["period_px"])
        coh = np.mean(np.array(sp["radial"]["coherence"]), 0)
        tr = np.mean(np.array(sp["radial"]["transfer"]), 0)
        a1.plot(p, coh, color=mcol(i), lw=2.4, label=fmt_steps(s))
        a2.plot(p, tr, color=mcol(i), lw=2.4, label=fmt_steps(s))
    for ax in (a1, a2):
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlim(64, 2)
        ax.set_xticks([64, 32, 16, 8, 4, 2])
        ax.set_xticklabels(["64", "32", "16", "8", "4", "2"])
        ax.set_xlabel("period (SR px, finer →)")
    a1.set_ylabel("coherence with synthetic truth")
    a1.set_title("Coherence (4-band mean)")
    a2.axhline(1, color=INK2, lw=1, ls=":")
    a2.set_ylabel("amplitude transfer √(P_SR / P_truth)")
    a2.set_title("Transfer (4-band mean)")
    a1.legend(loc="lower left", ncol=2, title="257 milestone", title_fontsize=15)
    f.tight_layout()
    save(f, "c_spec_ms.png")

    # error-direction maps: angular structure only (map / ring mean)
    names = [("member_218", "218 (2M, no isotropy)"), ("member_227", "227 (15.5M)"),
             ("member_238", "238 production"), ("member_249", "249 (200k)"),
             ("member_257@8000000", "257 @ 8M")]
    f = fig(1664, 420)
    axs = f.subplots(1, len(names))
    cmap = LinearSegmentedColormap.from_list("d", ["#3987e5", "#1a2230", "#0f1319", "#3a2416", "#eb6834"])
    for ax, (n, lab) in zip(axs, names):
        sp = load(n)
        mp = np.mean(np.array(sp["map"]["values"], float), 0)
        k = np.array(sp["map"]["k"])
        kx, ky = np.meshgrid(k, k)
        kr = np.round(np.hypot(kx, ky)).astype(int)
        ring = np.zeros_like(mp)
        for rr in np.unique(kr):
            sel = kr == rr
            ring[sel] = np.nanmean(mp[sel])
        rel = mp / ring
        rel[kr < 3] = np.nan
        im = ax.imshow(rel, cmap=cmap, vmin=0.6, vmax=1.4, origin="lower",
                       extent=[k[0], k[-1], k[0], k[-1]])
        ax.set_title(lab, fontsize=17)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = f.colorbar(im, ax=axs, fraction=0.012, pad=0.01)
    cb.set_label("error power / ring mean", fontsize=15)
    cb.outline.set_visible(False)
    save(f, "c_spec_maps.png")


# ---------------------------------------------------------------- flux
def flux():
    z = np.load(f"{D}/jwst_euclid_overlap/flux_conservation/report_fluxes.npz")
    mag = z["vis_mag"]
    lr = z["lr:kron_vis"]
    f = fig(1100, 640)
    ax = f.subplots()
    bins = np.arange(20.5, 25.51, 0.5)
    mids = 0.5 * (bins[1:] + bins[:-1])
    for key, lab, col in [("universal:member_249", "249 (200k)", C["m249"]),
                          ("production", "238 production", C["prod"]),
                          ("universal:member_218", "218 (2M)", C["m218"]),
                          ("universal:member_227", "227 (15.5M)", C["m227"]),
                          ("milestone:member_257@8000000", "257 @ 8M", C["m257"])]:
        r = z[f"{key}:kron_vis"] / lr
        med = [np.nanmedian(r[(mag >= a) & (mag < b)]) if ((mag >= a) & (mag < b)).sum() > 10 else np.nan
               for a, b in zip(bins[:-1], bins[1:])]
        ax.plot(mids, med, "-o", color=col, label=lab, ms=8, mec=BG, mew=1.5)
    ax.axhline(1, color=INK2, lw=1.2, ls=":")
    ax.set_xlabel("VIS magnitude (MER catalogue)")
    ax.set_ylabel("Kron VIS flux, SR / Euclid input (median)")
    ax.set_title("Flux in MER’s own Kron ellipses, NEXUS tiles")
    ax.legend(loc="lower left", fontsize=16)
    f.tight_layout()
    save(f, "c_flux.png")


# ---------------------------------------------------------------- arms
def arms():
    ix = json.load(open(f"{UNI}/index.json"))["models"]
    rows = [  # (round, member, label, base, real faint % sharper (None = no number))
        ("1", 239, "replicate (seed 1729)", 237, -0.74),
        ("1", 240, "lr_final 1e-6", 237, 0.0),
        ("1", 245, "lr_peak 1e-3", 237, -3.4),
        ("1", 246, "knee tilt −0.25", 237, 0.9),
        ("1", 243, "knee tilt +0.25", 237, None),
        ("1", 244, "loss floor 1σ", 237, 0.69),
        ("1", 241, "isotropy 2–64 px", 237, 0.0),
        ("2", 247, "lr_peak 2e-3", 237, -3.2),
        ("2", 248, "knee tilt +0.5", 237, -8.3),
        ("2", 249, "isotropy 2–64 + floor 1σ", 237, 0.9),
        ("2", 250, "tilt −0.25 + isotropy", 237, 0.5),
        ("3", 251, "loss knees ≥ 1 e⁻ only", 249, -2.4),
        ("3", 252, "no PSF warps", 249, -0.8),
        ("4", 253, "+ brightness-consistency term", 249, -3.1),
        ("4", 254, "Lupton loss + isotropy", 249, 0.9),
        ("5", 255, "ICNR off", 249, 0.15),
        ("5", 256, "input knees ≥ 0.1 e⁻", 249, -0.55),
    ]
    n = len(rows)
    f = fig(1664, 820)
    a1, a2 = f.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [1, 1], "wspace": 0.05})
    ys = np.arange(n)[::-1].astype(float)
    # gap between rounds
    off = 0
    prev = rows[0][0]
    for i, r in enumerate(rows):
        if r[0] != prev:
            off += 0.6
            prev = r[0]
        ys[i] -= off
    for i, (rd, mem, lab, base, real) in enumerate(rows):
        d = ix[f"member_{mem}"]["mean"] - ix[f"member_{base}"]["mean"]
        col = C["m249"] if mem == 249 else ("#5d6470" if rd in "1" else "#7f8794")
        a1.plot([0, d], [ys[i]] * 2, color=GRID, lw=2.4, solid_capstyle="round")
        a1.plot(d, ys[i], "o", ms=12, color=C["m257"] if mem in (249, 255) else INK, mec=BG, mew=2)
        a1.text(max(d, 0) + 0.12 if d > -4.5 else 0.12, ys[i], f"{d:+.2f}", va="center", fontsize=14, color=INK2)
        if real is None:
            a2.text(0, ys[i], "not measured", va="center", ha="center", fontsize=14, color=INK2)
        else:
            a2.plot([0, real], [ys[i]] * 2, color=GRID, lw=2.4)
            a2.plot(real, ys[i], "o", ms=12, color=C["m257"] if mem in (249, 255) else INK, mec=BG, mew=2)
            a2.text(real + (0.35 if real >= 0 else -0.35), ys[i], f"{real:+.1f}%",
                    va="center", ha="left" if real >= 0 else "right", fontsize=14, color=INK2)
    a1.axvspan(-0.05, 0.05, color="#2a313b", zorder=0)
    a2.axvspan(-0.35, 0.35, color="#2a313b", zorder=0)
    for a in (a1, a2):
        a.axvline(0, color=INK2, lw=1)
        a.grid(axis="y", visible=False)
    a1.set_yticks(ys)
    a1.set_yticklabels([f"{r[1]}  {r[2]}  (vs {r[3]})" for r in rows], fontsize=15)
    a1.set_xlim(-5.4, 1.0)
    a2.set_xlim(-9.5, 3.0)
    a1.set_xlabel("Δ synthetic integrated PSNR vs base (dB)")
    a2.set_xlabel("faint real-sky resolution: % sharper (− blurrier)")
    a1.set_title("Synthetic test set")
    a2.set_title("Real sky (JWST NEXUS)")
    a1.text(0, ys[0] + 0.9, "seed spread", ha="center", fontsize=13, color=INK2)
    a2.text(0, ys[0] + 0.9, "seed spread", ha="center", fontsize=13, color=INK2)
    a1.set_ylim(ys[-1] - 0.8, ys[0] + 1.5)
    f.subplots_adjust(left=0.29, right=0.985, top=0.93, bottom=0.09, wspace=0.05)
    save(f, "c_arms.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["res_vs_sb", "frc_curve", "ms_scores", "ms_resolution",
                             "train_log", "spectra", "flux", "arms"]
    for w in which:
        globals()[w]()
