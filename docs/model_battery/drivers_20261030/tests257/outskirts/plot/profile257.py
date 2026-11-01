"""Outskirt injection recovery profile: 257@15M vs references (deck style of review.py's rv-* slides).

python profile257.py  → out/outskirt_recovery_257.png (+ review.py's rv_recovery.png / rv_dumbbell.png with 257)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from charts import fig, save, glow, plt, np, OUT, C, INK, INK2, GRID, BG  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import review as rv  # noqa: E402

RUNS = "/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/6d444c86-12c6-4a7f-b4eb-7b73fa2e1c66/scratchpad/tests257/outskirts/runs"
OLD = rv.L
# key, label, colour, file, new run?
SERIES = [("257", "257 @ 15M", C["m257"], f"{RUNS}/deck_milestone_member_257_15000000.npz", True),
          ("238", "238 production · 1.9M", C["prod"], f"{RUNS}/deck_member_238.npz", True),
          ("249", "249 · 257 recipe, 200k", C["m249"], f"{RUNS}/deck_member_249.npz", True),
          ("228", "228 · noise jitter", "#e9e6df", f"{RUNS}/deck_member_228.npz", True),
          ("229", "229 · 228's twin, no jitter", C["m227"], f"{OLD}/deck_member_229.npz", False),
          ("258", "258 · galaxy blur", "#e8c547", f"{OLD}/deck_member_258.npz", False)]
HOST = ["real-like noise", "training-like noise"]
CAPTION = ("40 synthetic galaxies, host m > 19.5, 4-band mean · solid: re-run 10-29 (shading: 16–84% field bootstrap) · "
           "dashed: 10-28 deck files (Gaussian hosts only)")


def main():
    summ = json.load(open(f"{RUNS}/summary.json"))
    names = summ["rings"]
    f = fig(1664, 700)
    axs = f.subplots(1, 2, sharey=True)
    f.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.16, wspace=0.06)
    for h, ax in enumerate(axs):
        rows = []
        for key, lab, col, path, is_new in SERIES:
            if not os.path.exists(path):
                continue
            d = np.load(path)
            a, r = d["acc"], d["r_edges"]
            x = 0.5 * (r[:-1] + r[1:])
            x[-1] = 5.0
            y = (a[h, 0] / a[h, 1]).mean(-1)
            rows.append((key, col, y[4], y[5]))
            if is_new:
                b = summ["boot"].get(key, {}).get(HOST[h])
                if b:
                    lo = np.full(len(x), np.nan)
                    hi = np.full(len(x), np.nan)
                    for i, nm in enumerate(names):
                        if nm in b:
                            lo[i], hi[i] = b[nm][0], b[nm][2]
                    ok = np.isfinite(lo)
                    ax.fill_between(x[ok], lo[ok], hi[ok], color=col, alpha=0.18, lw=0, zorder=1)
                glow(ax, x, y, col)
                ax.plot(x, y, color=col, lw=3.2 if key == "257" else 2.6, marker="o", ms=8, mec=BG, mew=1.5,
                        zorder=5 if key == "257" else 4)
            else:
                ax.plot(x, y, color=col, lw=1.8, ls="--", alpha=0.6, zorder=3)
        _values_box(ax, rows)
        ax.axhline(1, color=INK2, lw=1.2, ls=":")
        ax.axvspan(2, 2.5, color="#ffffff", alpha=0.05)
        ax.set_title(f"{HOST[h]}  ·  4-band mean", loc="left", color=INK)
        ax.set_xlabel("distance from centre / Kron radius")
        ax.set_xlim(0, 5.3)
        ax.set_ylim(0, 1.25)
        ax.grid(color=GRID)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel("injected light recovered")
    hd = [Line2D([], [], color=col, lw=4 if new else 2, ls="-" if new else "--", marker="o" if new else None,
                 ms=9, label=lab) for key, lab, col, p, new in SERIES if os.path.exists(p)]
    f.legend(handles=hd, loc="upper center", ncol=3, frameon=False, fontsize=15, bbox_to_anchor=(0.52, 1.0))
    f.text(0.005, 0.005, CAPTION, ha="left", va="bottom", fontsize=12, color=INK2)
    save(f, "outskirt_recovery_257.png")

    # review.py's own two charts, with 257 added (VIS/H curves on real-like noise; dumbbell real vs training)
    rv.MODELS = [("m257", "257 @ 15M", C["m257"]), ("member_228", "228 · jitter + floor", "#2fd1c4"),
                 ("member_229", "229 · no jitter", "#d55181"), ("member_238", "238 · production", "#d95926"),
                 ("member_249", "249 · 200k", "#3987e5"), ("member_258", "258 · galaxy blur", "#e8c547")]
    files = {"m257": f"{RUNS}/deck_milestone_member_257_15000000.npz",
             "member_228": f"{RUNS}/deck_member_228.npz", "member_229": f"{OLD}/deck_member_229.npz",
             "member_238": f"{RUNS}/deck_member_238.npz", "member_249": f"{RUNS}/deck_member_249.npz",
             "member_258": f"{OLD}/deck_member_258.npz"}
    acc = {k: np.load(p) for k, p in files.items() if os.path.exists(p)}
    rv.recovery_curves(acc)
    rv.dumbbell(acc)
    if os.path.exists(f"{RUNS}/hosts_summary.json"):
        hosts_fig(summ)
    print("out dir", OUT)


def _values_box(ax, rows):
    """Recovery at 2–2.5 | 2.5–3 r_K per model, in the empty lower-left corner (instead of overlapping labels)."""
    ax.text(0.02, 0.60, "at 2–2.5 | 2.5–3 r_K", transform=ax.transAxes, ha="left", va="top", fontsize=13,
            color=INK2)
    for n, (key, col, v1, v2) in enumerate(rows):
        ax.text(0.02, 0.53 - 0.065 * n, f"{key}  {v1:.2f} | {v2:.2f}", transform=ax.transAxes, ha="left",
                va="top", fontsize=14, color=col, family="DejaVu Sans Mono")


def _curve(ax, x, y, col, lw, band=None, dashed=False, label_at=None):
    if band is not None:
        ok = np.isfinite(band[0])
        ax.fill_between(x[ok], band[0][ok], band[1][ok], color=col, alpha=0.18, lw=0, zorder=1)
    if dashed:
        ax.plot(x, y, color=col, lw=1.8, ls="--", alpha=0.6, zorder=3)
        return
    glow(ax, x, y, col)
    ax.plot(x, y, color=col, lw=lw, marker="o", ms=7, mec=BG, mew=1.5, zorder=4)


def _band(boot, names, n):
    lo, hi = np.full(n, np.nan), np.full(n, np.nan)
    for i, nm in enumerate(names):
        if boot and nm in boot:
            lo[i], hi[i] = boot[nm][0], boot[nm][2]
    return lo, hi


def hosts_fig(summ):
    """2×2: Gaussian real-like | real NEXUS tiles  /  Gaussian training-like | synthetic test fields."""
    hs = json.load(open(f"{RUNS}/hosts_summary.json"))
    names = summ["rings"]
    r = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 9.1])
    x = 0.5 * (r[:-1] + r[1:])
    x[-1] = 5.0
    f = fig(1664, 1180)
    axs = f.subplots(2, 2, sharex=True, sharey=True)
    f.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.10, wspace=0.05, hspace=0.16)
    titles = [["Gaussian noise · real Euclid σ + lag-1", "real NEXUS tiles as hosts (m > 19.5)"],
              ["Gaussian noise · training (v5) σ + lag-1", "synthetic test fields as hosts (v5 noise)"]]
    hosts_key = {"real NEXUS tiles": (0, 1), "synthetic test fields": (1, 1)}
    rows = {(rr, cc): [] for rr in range(2) for cc in range(2)}
    for key, lab, col, path, is_new in SERIES:
        if not os.path.exists(path):
            continue
        d = np.load(path)
        for h in range(2):
            y = (d["acc"][h, 0] / d["acc"][h, 1]).mean(-1)
            rows[(h, 0)].append((key, col, y[4], y[5]))
            band = _band(summ["boot"].get(key, {}).get(HOST[h]), names, len(x)) if is_new else None
            _curve(axs[h, 0], x, y, col, 3.2 if key == "257" else 2.4, band, dashed=not is_new,
                   label_at=4 if key in ("257", "238", "249", "228") else None)
    hcol = {k: c for k, _, c, _, _ in SERIES}
    for k, per in hs["models"].items():
        if "new" not in per:
            continue
        for host, (rr, cc) in hosts_key.items():
            y = np.array(per["new"][host]["mean4"])
            rows[(rr, cc)].append((k, hcol[k], y[4], y[5]))
            band = _band(hs["boot"].get(k, {}).get(host), names, len(x))
            _curve(axs[rr, cc], x, y, hcol[k if k != "257@15M" else "257"], 3.2 if k == "257" else 2.4, band,
                   label_at=4)
    for rr in range(2):
        for cc in range(2):
            ax = axs[rr, cc]
            _values_box(ax, rows[(rr, cc)])
            ax.axhline(1, color=INK2, lw=1.2, ls=":")
            ax.axvspan(2, 2.5, color="#ffffff", alpha=0.05)
            ax.set_title(titles[rr][cc], loc="left", color=INK, fontsize=17)
            ax.set_xlim(0, 5.3)
            ax.set_ylim(0, 1.25)
            ax.grid(color=GRID)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    f.supylabel("injected light recovered (4-band mean)", color=INK2, fontsize=18, x=0.012)
    for cc in range(2):
        axs[1, cc].set_xlabel("distance from centre / Kron radius")
    hd = [Line2D([], [], color=col, lw=4 if new else 2, ls="-" if new else "--", marker="o" if new else None,
                 ms=9, label=lab) for key, lab, col, p, new in SERIES if os.path.exists(p)]
    f.legend(handles=hd, loc="upper center", ncol=3, frameon=False, fontsize=15, bbox_to_anchor=(0.52, 1.0))
    f.text(0.005, 0.004, CAPTION, ha="left", va="bottom", fontsize=12, color=INK2)
    save(f, "outskirt_hosts_257.png")


if __name__ == "__main__":
    main()
