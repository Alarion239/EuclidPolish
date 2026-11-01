"""Review slides: noise correlation, injection recovery (228 jitter vs others), JWST baseline check."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from charts import fig, save, glow, plt, np, OUT, C, INK, INK2, GRID, BG
from matplotlib.lines import Line2D

L = "/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/d7b359b4-16d6-4f7f-aa2b-959d682a91bd/scratchpad/lrcheck"
MODELS = [("member_228", "228 · jitter + floor", "#2fd1c4"),
          ("member_229", "229 · same fork, no jitter", "#d55181"),
          ("member_238", "238 · production", "#d95926"),
          ("member_249", "249 · 200k", "#3987e5"),
          ("member_258", "258 · galaxy blur", "#e8c547")]
BN = ["VIS", "Y", "J", "H"]


def noise_corr():
    real = [0.196, 0.76, 0.762, 0.762]
    train_sky = [0.243, 0.829, 0.832, 0.826]
    fixed = [0.240, 0.860, 0.862, 0.863]
    jit_lo, jit_med, jit_hi = [0.183, 0.752, 0.751, 0.752], [0.230, 0.803, 0.806, 0.805], [0.289, 0.857, 0.863, 0.856]
    f = fig(1664, 560)
    gs = f.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.08, left=0.07, right=0.98, top=0.78, bottom=0.15)
    for k, (bands, xl) in enumerate((([0], (0.15, 0.31)), ([1, 2, 3], (0.73, 0.88)))):
        ax = f.add_subplot(gs[k])
        for i, b in enumerate(bands):
            y = len(bands) - 1 - i
            ax.plot([jit_lo[b], jit_hi[b]], [y, y], color="#2fd1c4", lw=14, alpha=0.22, solid_capstyle="round")
            ax.plot(jit_med[b], y, "|", color="#2fd1c4", ms=26, mew=3)
            ax.plot(fixed[b], y, "o", color="#d95926", ms=15, mec=BG, mew=2, zorder=4)
            ax.plot(train_sky[b], y + 0.18, "v", color="#d95926", ms=11, alpha=0.7, zorder=4)
            ax.plot(real[b], y, "D", color="#e9e6df", ms=14, mec=BG, mew=2, zorder=5)
        ax.set_yticks(range(len(bands)))
        ax.set_yticklabels([BN[b] for b in bands][::-1], fontsize=20)
        ax.set_ylim(-0.6, len(bands) - 0.4)
        ax.set_xlim(*xl)
        ax.set_xlabel("neighbour-pixel correlation (lag 1)")
        ax.grid(axis="x", color=GRID)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    h = [Line2D([], [], marker="D", ls="", color=INK, ms=13, label="real Euclid sky (NEXUS)"),
         Line2D([], [], marker="o", ls="", color="#d95926", ms=13, label="noise model, fixed texture (all current runs)"),
         Line2D([], [], marker="v", ls="", color="#d95926", ms=11, alpha=0.7, label="training fields, empty sky"),
         Line2D([], [], color="#2fd1c4", lw=12, alpha=0.3, label="noise_texture_jitter range, median |")]
    f.legend(handles=h, loc="upper center", ncol=2, frameon=False, fontsize=16, bbox_to_anchor=(0.52, 1.0))
    save(f, "rv_noise_corr.png")


def load_acc():
    out = {}
    for m, lab, col in MODELS:
        p = f"{L}/deck_{m}.npz"
        if os.path.exists(p):
            out[m] = np.load(p)
    return out


def recovery_curves(acc):
    f = fig(1664, 640)
    axs = f.subplots(1, 2, sharey=True)
    f.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.15, wspace=0.06)
    for ax, b in zip(axs, (0, 3)):
        for m, lab, col in MODELS:
            if m not in acc:
                continue
            a, r = acc[m]["acc"], acc[m]["r_edges"]
            x = 0.5 * (r[:-1] + r[1:])
            x[-1] = 5.0
            rec = a[0, 0, :, b] / a[0, 1, :, b]
            glow(ax, x, rec, col)
            ax.plot(x, rec, color=col, lw=3, marker="o", ms=8, mec=BG, mew=1.5, zorder=4)
        ax.axhline(1, color=INK2, lw=1.2, ls=":")
        ax.axvspan(2, 3, color="#ffffff", alpha=0.04)
        ax.set_title(f"{BN[b]}", loc="left", color=INK)
        ax.set_xlabel("distance from centre / Kron radius")
        ax.set_xlim(0, 5.3)
        ax.set_ylim(0, 1.25)
        ax.grid(color=GRID)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel("injected light recovered")
    h = [Line2D([], [], color=col, lw=4, marker="o", ms=9, label=lab) for m, lab, col in MODELS if m in acc]
    f.legend(handles=h, loc="upper center", ncol=5, frameon=False, fontsize=16, bbox_to_anchor=(0.52, 1.0))
    save(f, "rv_recovery.png")


def dumbbell(acc):
    f = fig(1664, 560)
    axs = f.subplots(1, 2, sharey=True)
    f.subplots_adjust(left=0.2, right=0.98, top=0.78, bottom=0.15, wspace=0.08)
    ms = [t for t in MODELS if t[0] in acc][::-1]
    for ax, (i0, title) in zip(axs, ((4, "2–2.5 Kron radii"), (5, "2.5–3 Kron radii"))):
        for y, (m, lab, col) in enumerate(ms):
            a = acc[m]["acc"]
            real = np.mean([a[0, 0, i0, b] / a[0, 1, i0, b] for b in range(4)])
            syn = np.mean([a[1, 0, i0, b] / a[1, 1, i0, b] for b in range(4)])
            ax.plot([real, syn], [y, y], color=col, lw=4, alpha=0.55, solid_capstyle="round")
            ax.plot(syn, y, "o", color=BG, mec=col, mew=3, ms=16, zorder=4)
            ax.plot(real, y, "o", color=col, mec=BG, mew=2, ms=17, zorder=5)
            ax.text(real - 0.025, y, f"{real:.2f}", ha="right", va="center", color=INK, fontsize=16)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlim(0.1, 1.05)
        ax.set_xlabel("injected light recovered (mean of 4 bands)")
        ax.grid(axis="x", color=GRID)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(range(len(ms)))
    axs[0].set_yticklabels([lab for m, lab, col in ms], fontsize=18)
    h = [Line2D([], [], marker="o", ls="", color=INK, ms=14, label="real Euclid noise level + correlation"),
         Line2D([], [], marker="o", ls="", color=BG, mec=INK, mew=3, ms=14, label="training noise level + correlation")]
    f.legend(handles=h, loc="upper center", ncol=2, frameon=False, fontsize=16, bbox_to_anchor=(0.58, 1.0))
    save(f, "rv_dumbbell.png")


K = "/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish--claude-worktrees-pixel-input-knee-dependency-efc29d/ab73eb83-3218-4250-ba41-4de98db6e1f6/scratchpad"


def rings():
    series = [("truth", "synthetic truth", INK, "--"),
              ("universal:member_249", "249 · 200k", C["m249"], "-"),
              ("production", "238 production · 1.9M", C["prod"], "-"),
              ("universal:member_227", "227 · 15.5M", C["m227"], "-"),
              ("milestone:member_257@8000000", "257 @ 8M", C["m257"], "-")]
    f = fig(1664, 660)
    axs = f.subplots(1, 2, sharey=True)
    f.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.16, wspace=0.06)
    labels = ["0–0.2", "0.2–0.4", "0.4–0.7", "0.7–1", "1–1.5", "1.5–2.5", "2.5–4"]
    x = np.arange(7)
    syn_truth = None
    for ax, mode, title in zip(axs, ("syn", "real"), ("Synthetic test fields · 141 galaxies", "Real NEXUS sky · 447 galaxies")):
        r = [g for g in json.load(open(f"{K}/ann_{mode}.json"))["rows"] if 23.5 <= g["mag"] < 25.5]
        inp = np.array([g["input"] for g in r])
        n15 = inp[:, :5].sum()
        if mode == "real" and syn_truth is not None:
            ax.plot(x[:6], syn_truth[:6], "--", color=INK, lw=2, alpha=0.45, zorder=3)
            ax.text(3, syn_truth[3] + 1.6, "synthetic truth", color=INK2, fontsize=14, ha="center")
        for key, lab, col, ls in series:
            if key not in r[0]:
                continue
            d = (np.array([g[key] for g in r]) - inp).sum(0) / n15 * 100
            if key == "truth":
                syn_truth = d
            if ls == "-":
                glow(ax, x[:6], d[:6], col)
            ax.plot(x[:6], d[:6], ls, color=col, lw=3, marker="o", ms=9, mec=BG, mew=1.5, zorder=4)
            ax.plot(x[5:], d[5:], ls, color=col, lw=2, alpha=0.35, zorder=3)
        ax.axhline(0, color=INK2, lw=1.2, ls=":")
        ax.axvspan(0.6, 4.4, color="#ffffff", alpha=0.035)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=15)
        ax.set_xlabel("elliptical ring, arcsec along the major axis")
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylim(-28, 25)
        ax.grid(color=GRID)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel("SR − input, % of input flux in 1.5″")
    h = [Line2D([], [], color=col, ls=ls, lw=3.5, marker="o", ms=9, label=lab) for k, lab, col, ls in series]
    f.legend(handles=h, loc="upper center", ncol=5, frameon=False, fontsize=16, bbox_to_anchor=(0.53, 1.0))
    save(f, "rv_rings.png")


def stamps(knee=3.0, white=600, field=5, cy=67, cx=114, half=60, only=None):
    from PIL import Image
    from common import rgb
    sl = (slice(cy - half, cy + half), slice(cx - half, cx + half))

    def write(cube, name, px=560):
        im = rgb(np.moveaxis(cube[sl], -1, 0), knee, white)
        Image.fromarray((im * 255).astype(np.uint8)).resize((px, px), Image.LANCZOS).save(os.path.join(OUT, name))
        return im

    ims = {}
    for m, lab, col in MODELS:
        if only and m not in only:
            continue
        p = f"{L}/deck_{m}.npz"
        if not os.path.exists(p):
            continue
        d = np.load(p)
        if "truth" not in ims:
            ims["truth"] = write(d["truth"][field], "st_truth.png", 800)
        for h, key in enumerate(("real", "syn")):
            ims[(m, key)] = write(d["stamps"][field, h], f"st_m{m[-3:]}_{key}.png")
        print("stamps", m)
    if ("member_228", "real") in ims and ("member_229", "real") in ims:
        a, b = ims[("member_229", "real")], ims[("member_228", "real")]
        n = a.shape[1] // 2
        comp = np.concatenate([a[:, :n], b[:, n:]], 1)
        im = Image.fromarray((comp * 255).astype(np.uint8)).resize((1080, 1080), Image.LANCZOS)
        im.crop((130, 0, 130 + 820, 1080)).save(os.path.join(OUT, "rv_sec_bg.png"))
        print("sec bg")


if __name__ == "__main__":
    what = sys.argv[1:] or ["noise"]
    if "noise" in what:
        noise_corr()
    if "rings" in what:
        rings()
    if "inj" in what:
        a = load_acc()
        recovery_curves(a)
        dumbbell(a)
    if "stamps" in what:
        stamps()
