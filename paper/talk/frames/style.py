"""House style for the Hernquist-talk images: dark canvas, Space Grotesk / Space Mono,
image panels in the style of the JWST triples (bold white title, band legend, bar, compass).

Every frame script imports this module. Images are written at 3840x2160 (frames) or as
square layers (callback panels that Prezi fades into each other at the same position).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
TALK = os.path.dirname(HERE)
OUT = os.path.join(TALK, "images")
LAYERS = os.path.join(OUT, "layers")
FONTS = os.path.join(TALK, "fonts")

for f in ("SpaceGrotesk-Regular.ttf", "SpaceGrotesk-Medium.ttf", "SpaceGrotesk-Bold.ttf",
          "SpaceMono-Regular.ttf", "SpaceMono-Bold.ttf"):
    font_manager.fontManager.addfont(os.path.join(FONTS, f))

SANS = "Space Grotesk"
MONO = "Space Mono"
BG = "#0f1218"
CARD = "#1a1f29"
EDGE = "#2c3340"
TXT = "#e6e8eb"
MUTED = "#8b93a1"
TEAL = "#2fd4bf"
CORAL = "#f08a6c"
BAND = {"VIS": "#7fb2ff", "Y": "#8fe08f", "J": "#8fe08f", "H": "#ff8a7a"}
LEGEND_COLORS = ("#7fb2ff", "#8fe08f", "#ff8a7a")

STROKE = [pe.withStroke(linewidth=3.0, foreground="black")]
SHOW_CALLBACK_MARK = False      # the coral "this comes back later" envelope; off since 11-22 (user)

W_PX, H_PX, DPI = 3840, 2160, 200


def frame():
    """A 3840x2160 dark canvas with a [0,16]x[0,9] data box (1 unit = 240 px)."""
    fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    return fig, ax


def eyebrow(ax, x, y, text, color=TEAL, size=13):
    ax.text(x, y, text.upper(), color=color, fontsize=size, family=MONO, va="baseline")


def title(ax, x, y, text, size=34, color=TXT, **kw):
    ax.text(x, y, text, color=color, fontsize=size, family=SANS, weight="bold", va="baseline", **kw)


def body(ax, x, y, text, size=17, color=MUTED, **kw):
    ax.text(x, y, text, color=color, fontsize=size, family=SANS, va="top", linespacing=1.45, **kw)


def image_panel(ax, img, *, x, y, size, label=None, legend=None, interp="lanczos",
                bar_arcsec=None, pix_arcsec=0.05, compass=False, envelope=False, coords=None,
                label_size=20, label_stroke=3.0):
    """Draw an RGB image as a square panel of side ``size`` (data units) at lower-left (x, y),
    with the triples' annotations: bold label top-left, band legend bottom-left, bar, compass."""
    ax.imshow(img, origin="lower", extent=(x, x + size, y, y + size), interpolation=interp,
              zorder=2)
    pad = size * 0.035
    if label:
        ax.text(x + pad, y + size - pad, label, color="white", fontsize=label_size, family=SANS,
                weight="bold", va="top", ha="left", zorder=3,
                path_effects=[pe.withStroke(linewidth=label_stroke, foreground="black")])
    if legend:
        xx = x + pad
        for name, col in zip(legend, LEGEND_COLORS):
            t = ax.text(xx, y + pad, name, color=col, fontsize=label_size * 0.62, family=SANS,
                        weight="medium", va="bottom", ha="left", path_effects=STROKE, zorder=3)
            ax.figure.canvas.draw()
            bb = t.get_window_extent().transformed(ax.transData.inverted())
            xx = bb.x1 + size * 0.025
    if coords:
        ax.text(x + size - pad, y + pad, coords, color="0.85", fontsize=label_size * 0.5,
                family=MONO, va="bottom", ha="right", path_effects=STROKE, zorder=3)
    n = img.shape[0]
    if bar_arcsec:
        L = bar_arcsec / (n * pix_arcsec) * size
        x1 = x + size - pad
        yb = y + pad * 1.4
        ax.plot([x1 - L, x1], [yb, yb], color="white", lw=2.6, solid_capstyle="butt", zorder=3,
                path_effects=[pe.withStroke(linewidth=5, foreground="black")])
        ax.text(x1 - L / 2, yb + size * 0.02, f"{bar_arcsec:g}″", color="white",
                fontsize=label_size * 0.55, family=SANS, ha="center", va="bottom",
                path_effects=STROKE, zorder=3)
    if compass:
        cx, cy, a = x + size - pad * 1.2, y + pad * 1.2, size * 0.08
        for dx, dy, lab in ((0, a, "N"), (-a, 0, "E")):
            ax.annotate("", xy=(cx + dx, cy + dy), xytext=(cx, cy), zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color="white", lw=1.6, mutation_scale=12,
                                        shrinkA=0, shrinkB=0))
            ax.text(cx + 1.35 * dx, cy + 1.35 * dy, lab, color="white", fontsize=label_size * 0.5,
                    family=SANS, ha="center", va="center", path_effects=STROKE, zorder=3)
    if envelope and SHOW_CALLBACK_MARK:
        envelope_mark(ax, x + size - pad - size * 0.075, y + size - pad - size * 0.05,
                      w=size * 0.075)


def envelope_mark(ax, x, y, w=0.4):
    """The callback mark: a small coral envelope (this image comes back later)."""
    from matplotlib.patches import FancyBboxPatch, Polygon
    h = w * 0.66
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={w*0.08}",
                                fc=CORAL, ec="none", zorder=4))
    ax.add_patch(Polygon([(x + w * 0.06, y + h * 0.92), (x + w / 2, y + h * 0.42),
                          (x + w * 0.94, y + h * 0.92)], closed=False, fill=False,
                         ec="#2a0f06", lw=2.2, zorder=5, joinstyle="round"))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_layer(img, name, px=2160, interp="lanczos"):
    """A bare square image layer (no text), for Prezi fades between callback states."""
    os.makedirs(LAYERS, exist_ok=True)
    fig = plt.figure(figsize=(px / DPI, px / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, origin="lower", interpolation=interp)
    ax.axis("off")
    path = os.path.join(LAYERS, name)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def save_panel(img, name, *, label=None, legend=None, interp="lanczos", bar_arcsec=None,
               pix_arcsec=0.05, compass=False, envelope=False, coords=None, px=2160, label_scale=1.0):
    """One square, annotated image (Euclid / POLISHed / JWST ...) as its own file.

    All panels of a callback share size and framing, so in Prezi the POLISHed or JWST panel
    can be dropped exactly on top of the Euclid one and faded in. label_scale > 1 for panels shown small
    (the cat card's 2 x 2 grid), so their labels stay readable."""
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(px / DPI, px / DPI), dpi=DPI, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    image_panel(ax, img, x=0, y=0, size=1, label=label, legend=legend, interp=interp,
                bar_arcsec=bar_arcsec, pix_arcsec=pix_arcsec, compass=compass,
                envelope=envelope, coords=coords, label_size=px / 2160 * 30 * label_scale,
                label_stroke=3.0 * label_scale)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, facecolor="black")
    plt.close(fig)
    return path
