"""Talk outline board (three acts, callbacks C1-C6) in the deck's dark style."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path
from matplotlib.patches import PathPatch

BG = "#0f1218"; CARD = "#1a1f29"; EDGE = "#2c3340"; TXT = "#e6e8eb"; MUTED = "#8b93a1"
TEAL = "#2fd4bf"; CORAL = "#f08a6c"

cols = [
    ("01", "Introduction", "8–9 min · the story", [
        ("Cold open: one Euclid tile", "C1"), ("Why wide surveys", None),
        ("Always from the ground", "C2"), ("Space: sharp but tiny", "C3"),
        ("Super-resolving the ground", None), ("Radio did it first: POLISH", None),
        ("Euclid is the first · DR1", "C4"), ("Strong lenses", "C5"),
        ("We need a truth → TNG", None)]),
    ("02", "Method", "5 min · TNG, honestly", [
        ("TNG50-SKIRT atlas", None), ("What we don't trust", None),
        ("Building a Euclid sky", None), ("Quiz: which four are real?", "C6"),
        ("The network, zoomed", None), ("Cats: the learned prior", None)]),
    ("03", "Payoff", "4–5 min · the callbacks", [
        ("Quiz fields, POLISHed", "C6"), ("Ground → Euclid → JWST", "C2"),
        ("Lens rings", "C5"), ("NGC 6505's Einstein ring", "C4"),
        ("Ultra Deep Field vs Hubble", "C3"), ("Back to the first tile", "C1"),
        ("Limits + a question for you", None)]),
]

fig = plt.figure(figsize=(16, 9), dpi=200, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
ax.text(0.7, 8.35, "EUCLIDPOLISH · HERNQUIST GROUP · 4 DEC", color=TEAL, fontsize=11,
        family="monospace")
ax.text(0.7, 7.85, "Talk outline: every Euclid image comes back super-resolved",
        color=TXT, fontsize=22, weight="bold")

x0s = [0.7, 5.85, 11.0]; W = 4.3; H = 0.48; step = 0.6; ytop = 6.55
pill_pos = {}
for (num, title, sub, items), x0 in zip(cols, x0s):
    ax.text(x0, 7.05, num, color=TEAL, fontsize=26, weight="bold")
    ax.text(x0 + 0.85, 7.17, title, color=TXT, fontsize=16, weight="bold")
    ax.text(x0 + 0.85, 6.93, sub, color=MUTED, fontsize=10.5)
    for i, (label, cb) in enumerate(items):
        y = ytop - i * step
        payoff = num == "03"
        ax.add_patch(FancyBboxPatch((x0, y - H), W, H, boxstyle="round,pad=0,rounding_size=0.08",
                     fc="#14302d" if payoff else CARD, ec=TEAL if payoff else EDGE, lw=1))
        ax.text(x0 + 0.18, y - H / 2, label, color=TXT, fontsize=12, va="center")
        if cb:
            px = x0 + W - 0.62
            ax.add_patch(FancyBboxPatch((px, y - H + 0.09), 0.5, H - 0.18,
                         boxstyle="round,pad=0,rounding_size=0.15", fc=CORAL, ec="none"))
            ax.text(px + 0.25, y - H / 2, cb, color="#2a0f06", fontsize=10.5, weight="bold",
                    ha="center", va="center")
            pill_pos.setdefault(cb, []).append((px, y - H / 2, num))

ax.add_patch(FancyBboxPatch((0.7, 0.45), 0.5, 0.3, boxstyle="round,pad=0,rounding_size=0.15",
             fc=CORAL, ec="none"))
ax.text(0.95, 0.6, "Cn", color="#2a0f06", fontsize=10.5, weight="bold", ha="center", va="center")
ax.text(1.35, 0.6, "a Euclid image shown early with a small envelope mark, then super-resolved "
        "in the payoff (C1–C5 in the introduction, C6 in the method)", color=MUTED, fontsize=11,
        va="center")
fig.savefig("outline.png", facecolor=BG)
