#!/usr/bin/env python
"""Inspect a HLSP tile remotely without transferring the whole file.

Designed to run on a FASRC login node, invoked over SSH by the web UI's
HST-tiles inspector page. Two modes:

  ``--mode header``
      Print the primary + first-image-extension FITS headers as JSON on
      stdout. Tiny (KB) output, suitable for a regular API endpoint.

  ``--mode cutout``
      Open the FITS with ``memmap=True`` so the multi-GB tile is NEVER
      fully loaded, read a single ``(size × size)`` slice at the chosen
      (or random) centre, apply an asinh stretch + grayscale PNG render,
      and write the PNG bytes to stdout.

The latter is the whole point of this script: a 256×256 cutout from a
10240×10240 float32 tile loads ~260 KB of pages off disk and produces
~30 KB of PNG bytes — well within the bounds of acceptable login-node
use (https://docs.rc.fas.harvard.edu/kb/transferring-data-on-the-cluster/).

Because the script writes binary PNG bytes to stdout in ``cutout`` mode,
all stderr-bound chatter (astropy warnings, our own status lines) is
explicitly muted so the calling SSH session gets clean bytes.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import warnings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", required=True,
                   help="Absolute path to the FITS tile on FASRC.")
    p.add_argument("--mode", choices=("header", "cutout"), required=True,
                   help="header: dump primary+ext FITS headers as JSON. "
                        "cutout: PNG bytes to stdout + cutout-stats JSON to stderr.")
    p.add_argument("--size", type=int, default=256,
                   help="Cutout side in pixels (cutout mode only).")
    p.add_argument("--cx", type=int, default=-1,
                   help="Cutout centre x in pixels; -1 → random.")
    p.add_argument("--cy", type=int, default=-1,
                   help="Cutout centre y in pixels; -1 → random.")
    p.add_argument("--seed", type=int, default=-1,
                   help="Optional RNG seed for the random cutout pick.")
    return p.parse_args()


def _dump_header_json(path: str) -> int:
    """Print {hdus: [{name, cards: [(key, val, comment), ...], shape, dtype}]}."""
    from astropy.io import fits
    out = {"path": path, "hdus": []}
    with fits.open(path, memmap=True) as hdul:
        for i, hdu in enumerate(hdul):
            cards = []
            for card in hdu.header.cards:
                key = str(card.keyword)
                val = str(card.value)
                if len(val) > 200:
                    val = val[:197] + "…"
                cards.append([key, val, str(card.comment)])
            shape = list(hdu.data.shape) if (
                hdu.is_image and hdu.data is not None
            ) else None
            dtype = str(hdu.data.dtype) if (
                hdu.is_image and hdu.data is not None
            ) else None
            out["hdus"].append({
                "index": i,
                "name":  hdu.name or f"HDU{i}",
                "kind":  type(hdu).__name__,
                "shape": shape,
                "dtype": dtype,
                "cards": cards,
            })
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()
    return 0


def _dump_cutout_png(
    path: str, size: int, cx: int, cy: int, seed: int,
) -> int:
    """Stream PNG bytes for a small random / specified-centre cutout.

    Stretch: ``asinh((x - median) / max(σ, ε))`` with sigma-clipped
    ``(median, σ)`` from the cutout itself. This is robust to bright
    sources (a single saturated star doesn't pull the noise estimate
    upward) and produces consistent contrast across cutouts of very
    different scenes (sky-only patches vs. bright-galaxy patches).
    Percentile clip at [0.5, 99.5] then linearly maps to 8-bit gray.
    """
    import numpy as np
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats

    with fits.open(path, memmap=True) as hdul:
        sci = next(
            (e for e in hdul if e.is_image and e.data is not None
             and getattr(e, "shape", None) is not None),
            None,
        )
        if sci is None:
            sys.stderr.write("no image HDU in file\n")
            return 2
        full_shape = sci.shape   # (H, W)
        H, W = int(full_shape[0]), int(full_shape[1])

        # Random centre if not specified. Stay inside the array with a
        # border so the cutout doesn't get clipped.
        rng = np.random.default_rng(seed if seed >= 0 else None)
        half = max(1, size // 2)
        if cx < 0:
            cx = int(rng.integers(half, max(half + 1, W - half)))
        if cy < 0:
            cy = int(rng.integers(half, max(half + 1, H - half)))

        # Clamp inside the array (defensive — random sampler covers this).
        x0 = max(0, min(W - size, cx - half))
        y0 = max(0, min(H - size, cy - half))
        x1, y1 = x0 + size, y0 + size

        # The slice load is what's cheap thanks to memmap.
        cutout = np.asarray(sci.data[y0:y1, x0:x1], dtype=np.float32)

    # NaN/inf scrub.
    finite = np.isfinite(cutout)
    if not finite.any():
        cutout = np.zeros_like(cutout)
    else:
        cutout = np.where(finite, cutout, np.nanmedian(cutout[finite]))

    # Sigma-clipped (median, std) → robust to bright outliers.
    mean, median, std = sigma_clipped_stats(cutout, sigma=3.0, maxiters=3)
    # If the cutout is essentially empty (zero-padded edge of the HLSP
    # mosaic where no exposures land), std ≈ 0 → render a flat gray
    # with a label-in-stderr so the UI can tell the user what happened.
    is_blank = float(std) < 1e-8

    if is_blank:
        img8 = np.full(cutout.shape, 128, dtype=np.uint8)
    else:
        # Asinh in noise-σ units, centred on the local sky → uniform
        # contrast regardless of absolute pixel values per tile.
        eps = max(float(std), 1e-8)
        stretched = np.arcsinh((cutout - float(median)) / eps)
        lo, hi = np.percentile(stretched, [0.5, 99.5])
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((stretched - lo) / (hi - lo), 0.0, 1.0)
        img8 = (255 * (1.0 - norm)).astype(np.uint8)   # gray_r style
    img8 = np.flipud(img8)                              # FITS → PIL origin

    from PIL import Image
    pil = Image.fromarray(img8, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    sys.stdout.buffer.write(buf.getvalue())
    sys.stdout.buffer.flush()
    # JSON sidecar to stderr — useful for diagnosing "why does this
    # cutout look like a blank gradient?" Includes the per-cutout stats
    # so the user can tell if they're looking at sky vs a bright source.
    sys.stderr.write(json.dumps({
        "cx": int(cx), "cy": int(cy),
        "x0": int(x0), "y0": int(y0),
        "size": int(size),
        "tile_shape": [int(H), int(W)],
        "stats": {
            "median": float(median),
            "std":    float(std),
            "min":    float(cutout.min()),
            "max":    float(cutout.max()),
            "blank":  bool(is_blank),
        },
    }) + "\n")
    return 0


def main() -> int:
    # Suppress every chatty astropy / numpy warning so cutout-mode stdout
    # stays clean. (Astropy likes to print VerifyWarning on legacy HLSP
    # headers — that'd corrupt the PNG byte stream.)
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    args = parse_args()
    if not os.path.isfile(args.path):
        sys.stderr.write(f"file not found: {args.path}\n")
        return 3
    if args.mode == "header":
        return _dump_header_json(args.path)
    if args.mode == "cutout":
        return _dump_cutout_png(
            args.path, args.size, args.cx, args.cy, args.seed,
        )
    sys.stderr.write(f"unknown mode: {args.mode}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
