#!/usr/bin/env python3
"""Render clean, borderless grayscale astro panels for the poster pipeline figure.

Each output PNG is a single square, axis-free, colorbar-free image with the
*house* asinh stretch (matching ``fasrc_tng_infographic._grayscale_norm``:
asinh at the 90th-percentile-of-positive-flux scale, then a [0.5, 99.5] clip).
Feed it whatever FITS you choose for each pipeline stage — the stretch is shared
so the panels read consistently side by side.

Examples
--------
# One file -> data/vis/poster/clean_validate_VIS_0000.png
python3 scripts/poster_render.py data/vis/sky_fits/clean_validate_VIS_0000.fits

# Several at once, custom output dir, tighter clip, center-crop to 400px
python3 scripts/poster_render.py data/vis/sky_fits/*.fits \
        --out data/vis/poster --clip 1 99 --crop 400

# Pin all panels to ONE file's stretch (so dirty/clean/SR share absolute scale)
python3 scripts/poster_render.py dirty.fits clean.fits sr.fits --ref clean.fits
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import make_lupton_rgb


# ---------------------------------------------------------------------------
# Stretch — house convention from fasrc_tng_infographic._grayscale_norm,
# generalized so the asinh scale and clip percentiles can be borrowed from a
# reference frame (keeps a clean/dirty/SR triple on one absolute scale).
# ---------------------------------------------------------------------------
def asinh_scale(arr: np.ndarray, scale_pct: float) -> float:
    d = np.clip(np.asarray(arr, dtype=np.float32), 0.0, None)
    pos = d[d > 0]
    return float(np.percentile(pos, scale_pct)) if pos.size else 1.0


def grayscale_norm(
    arr: np.ndarray,
    *,
    scale_pct: float = 90.0,
    clip: tuple[float, float] = (0.5, 99.5),
    scale: float | None = None,
    lohi: tuple[float, float] | None = None,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Return (normed[0,1], scale, (lo,hi)). Pass ``scale``/``lohi`` to reuse a
    reference frame's stretch instead of computing this frame's own."""
    d = np.clip(np.asarray(arr, dtype=np.float32), 0.0, None)
    if scale is None:
        scale = asinh_scale(d, scale_pct)
    s = np.arcsinh(d / max(scale, 1e-12))
    if lohi is None:
        lo, hi = np.percentile(s, list(clip))
        if hi <= lo:
            hi = lo + 1.0
    else:
        lo, hi = lohi
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0), scale, (lo, hi)


# ---------------------------------------------------------------------------
def load_fits(path: str, hdu: int | None) -> np.ndarray:
    with fits.open(path) as hl:
        if hdu is not None:
            data = hl[hdu].data
        else:  # first HDU that actually has 2-D image data
            data = next((h.data for h in hl if getattr(h, "data", None) is not None
                         and np.ndim(h.data) >= 2), None)
        if data is None:
            raise ValueError(f"no 2-D image data in {path}")
    data = np.asarray(data, dtype=np.float32)
    while data.ndim > 2:  # collapse leading singleton axes (e.g. (1,H,W))
        data = data[0]
    return data


def load_band(path: str, ident: str) -> np.ndarray:
    """Load one 2-D band by EXTNAME (e.g. ``H``) or integer HDU index."""
    with fits.open(path) as hl:
        try:
            data = hl[int(ident)].data          # numeric → HDU index
        except (ValueError, TypeError):
            data = hl[ident].data               # string → EXTNAME
    if data is None:
        raise ValueError(f"band '{ident}' has no data in {path}")
    data = np.asarray(data, dtype=np.float32)
    while data.ndim > 2:
        data = data[0]
    return data


def lupton_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray, *,
               q: float, stretch_pct: float) -> np.ndarray:
    """Asinh Lupton RGB (uint8 H,W,3), stretch auto-set from a flux percentile
    of the three channels — same recipe as fasrc_tng_infographic."""
    r, g, b = (np.clip(c, 0.0, None) for c in (r, g, b))
    ref = float(np.percentile(np.concatenate([r.ravel(), g.ravel(), b.ravel()]),
                              stretch_pct))
    return make_lupton_rgb(r, g, b, Q=q, stretch=max(ref, 1e-12))


def center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape
    size = min(size, h, w)
    y0, x0 = (h - size) // 2, (w - size) // 2
    return arr[y0:y0 + size, x0:x0 + size]


def block_mean(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr
    h, w = (arr.shape[0] // k) * k, (arr.shape[1] // k) * k
    return arr[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def save_panel(normed: np.ndarray, out_path: str, *, px: int, cmap: str,
               invert: bool) -> None:
    if invert:
        normed = 1.0 - normed
    dpi = 100
    fig = plt.figure(figsize=(px / dpi, px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # fills the whole canvas, no margins
    ax.imshow(normed, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0,
              interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(out_path, dpi=dpi, pad_inches=0)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fits", nargs="+", help="input FITS file(s)")
    p.add_argument("--out", default="data/vis/poster", help="output directory")
    p.add_argument("--hdu", type=int, default=None,
                   help="HDU index (default: first 2-D image HDU)")
    p.add_argument("--px", type=int, default=1200, help="output side length (px)")
    p.add_argument("--crop", type=int, default=0,
                   help="center-crop to NxN source px before stretch (0 = none)")
    p.add_argument("--downsample", type=int, default=1,
                   help="block-mean factor applied after crop")
    p.add_argument("--scale-pct", type=float, default=90.0,
                   help="percentile of positive flux for the asinh scale")
    p.add_argument("--clip", type=float, nargs=2, default=(0.5, 99.5),
                   metavar=("LO", "HI"), help="display clip percentiles")
    p.add_argument("--cmap", default="gray", help="matplotlib colormap")
    p.add_argument("--invert", action="store_true",
                   help="invert (dark sources on white — ink-saving for print)")
    p.add_argument("--ref", default=None,
                   help="FITS whose stretch (scale + lo/hi) all panels reuse")
    p.add_argument("--rgb", action="store_true",
                   help="composite a 3-colour image from a multi-band FITS")
    p.add_argument("--rgb-bands", default="H,J,VIS",
                   help="EXTNAMEs (or HDU indices) mapped to R,G,B "
                        "(default redder->R: H,J,VIS)")
    p.add_argument("--rgb-q", type=float, default=8.0, help="Lupton Q")
    p.add_argument("--rgb-stretch-pct", type=float, default=99.5,
                   help="flux percentile that sets the Lupton stretch")
    p.add_argument("--name", default=None,
                   help="output filename stem (default: input stem)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    clip = (args.clip[0], args.clip[1])

    # ---- RGB mode: one multi-band FITS -> one 3-colour PNG -------------------
    if args.rgb:
        path = args.fits[0]
        rid, gid, bid = [s.strip() for s in args.rgb_bands.split(",")]
        chans = []
        for ident in (rid, gid, bid):
            c = load_band(path, ident)
            if args.crop:
                c = center_crop(c, args.crop)
            chans.append(block_mean(c, args.downsample))
        rgb = lupton_rgb(*chans, q=args.rgb_q, stretch_pct=args.rgb_stretch_pct)
        stem = args.name or os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.out, stem + ".png")
        dpi = 100
        fig = plt.figure(figsize=(args.px / dpi, args.px / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(rgb, origin="lower", interpolation="nearest")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=dpi, pad_inches=0)
        plt.close(fig)
        print(f"  {path}  ->  {out_path}  RGB({rid},{gid},{bid})  "
              f"[{rgb.shape[0]}x{rgb.shape[1]}]")
        return

    ref_scale = ref_lohi = None
    if args.ref:
        ref = load_fits(args.ref, args.hdu)
        if args.crop:
            ref = center_crop(ref, args.crop)
        ref = block_mean(ref, args.downsample)
        _, ref_scale, ref_lohi = grayscale_norm(
            ref, scale_pct=args.scale_pct, clip=clip)
        print(f"reference stretch from {args.ref}: "
              f"scale={ref_scale:.4g} lo/hi={ref_lohi[0]:.3f}/{ref_lohi[1]:.3f}")

    for path in args.fits:
        arr = load_fits(path, args.hdu)
        if args.crop:
            arr = center_crop(arr, args.crop)
        arr = block_mean(arr, args.downsample)
        normed, _, _ = grayscale_norm(
            arr, scale_pct=args.scale_pct, clip=clip,
            scale=ref_scale, lohi=ref_lohi)
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.out, stem + ".png")
        save_panel(normed, out_path, px=args.px, cmap=args.cmap,
                   invert=args.invert)
        print(f"  {path}  ->  {out_path}  [{arr.shape[0]}x{arr.shape[1]}]")


if __name__ == "__main__":
    main()
