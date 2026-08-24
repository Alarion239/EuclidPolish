#!/usr/bin/env python
"""Consistency check: is arbitrary-angle spline rotation of a TNG galaxy safe
*before* a block-mean downsample, as a function of the downsample factor K?

Motivation
----------
The TNG injector currently rotates a SKIRT stamp only by exact quarter-turns
(``np.rot90``, lossless). We want arbitrary orientations (any 0–360°), which
needs spline interpolation — and interpolation blurs. The intuition is that a
*large enough* downsample factor K averages that blur away, so rotate-then-
downsample becomes indistinguishable from a clean direct downsample. This
script measures where that holds.

The experiment (per galaxy, per K)
----------------------------------
A deliberately *worst-case* rotation stress test:

  1. Take the native VIS frame, cropped to a centred window that holds the
     whole galaxy comfortably inside its inscribed circle (so rotation never
     clips real flux).
  2. Apply ``n_rot`` successive ``angle_step``-degree spline rotations
     (default 360 × 1°). 360 cumulative one-degree rotations return to the
     original orientation but accumulate the most interpolation blur a single
     rotation ever could — an upper bound on a single arbitrary-angle rotation.
  3. ``D_rot   = block_mean(rotated_360, K)`` (the pipeline's SB-preserving
     downsample) and ``D_direct = block_mean(frame, K)``.
  4. Error = relative RMS of ``D_rot − D_direct`` over the galaxy footprint.
     Since step 2 returns to the original orientation, the two are directly
     comparable with no second rotation.

A decreasing error(K) curve confirms the hypothesis; the smallest K with
error < ``--tol`` is the per-galaxy "critical K" above which arbitrary-angle
rotation is safe. The 360-rotation is computed once per galaxy and reused for
every K (it is K-independent), so the script is cheap.

Output: a 2-panel plot (relative-RMS error and fractional flux error vs K, one
line per galaxy) plus a printed summary table with each galaxy's critical K.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import rotate as ndi_rotate  # noqa: E402

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config  # noqa: E402
from euclid_polish.skirt.image import block_mean, load_skirt_frame  # noqa: E402
from euclid_polish.sky.generation.tng_galaxy import (  # noqa: E402
    list_tng_galaxies,
    tng_fits_path,
)

DEFAULT_K_LIST = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32)


def _growth_radius_px(frame: np.ndarray, frac: float) -> float:
    """Radius (px) about the frame centre enclosing ``frac`` of the positive flux."""
    a = np.where(np.isfinite(frame) & (frame > 0.0), frame, 0.0).astype(np.float64)
    total = a.sum()
    if total <= 0.0:
        return min(frame.shape) / 2.0
    H, W = a.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy = np.arange(H)[:, None] - cy
    xx = np.arange(W)[None, :] - cx
    rint = np.sqrt(yy * yy + xx * xx).astype(np.int64)
    cum = np.cumsum(np.bincount(rint.ravel(), weights=a.ravel()))
    return float(np.searchsorted(cum, frac * total))


def _central_window(frame: np.ndarray, frac: float, margin: float) -> np.ndarray:
    """Crop a centred square holding the galaxy inside its inscribed circle.

    Half-side = ``r_frac · √2 · margin`` so the flux-enclosing disk of radius
    ``r_frac`` stays within the window's inscribed circle even after rotation —
    i.e. rotation never clips real flux, only (near-zero) sky in the corners.
    """
    r = _growth_radius_px(frame, frac)
    half = int(np.ceil(r * np.sqrt(2.0) * margin))
    H, W = frame.shape
    cy, cx = H // 2, W // 2
    half = min(half, cy, cx)
    win = frame[cy - half: cy + half, cx - half: cx + half]
    # Even side keeps the centre on a pixel boundary and eases block_mean.
    if win.shape[0] % 2:
        win = win[:-1, :-1]
    return np.ascontiguousarray(win, dtype=np.float32)


def _cumulative_rotate(img: np.ndarray, angle_step: float, n_rot: int,
                       order: int) -> np.ndarray:
    """Apply ``n_rot`` successive ``angle_step``-degree spline rotations.

    ``reshape=False`` keeps the array size; ``mode='constant'`` (cval 0) treats
    off-frame as sky. Negatives from spline overshoot are clipped to 0 each
    step (surface brightness is non-negative), matching how a rendered stamp
    would be used.
    """
    out = img.astype(np.float32, copy=True)
    for _ in range(n_rot):
        out = ndi_rotate(out, angle_step, reshape=False, order=order,
                         mode="constant", cval=0.0, prefilter=True)
        np.maximum(out, 0.0, out=out)
    return out


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    """Relative RMS error ``‖a − b‖ / ‖b‖`` (0 = identical)."""
    denom = float(np.sqrt(np.mean(b * b)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)) / denom)


def _flux_err(a: np.ndarray, b: np.ndarray) -> float:
    """Fractional flux difference ``|Σa − Σb| / Σb`` (block_mean conserves SB,
    so this also tracks total-flux drift after the pixel-count change)."""
    sb = float(b.sum())
    return float(abs(a.sum() - sb) / sb) if sb > 0 else float("nan")


def analyse_galaxy(frame: np.ndarray, *, k_list, angle_step, n_rot, order,
                   crop_frac, margin):
    """Return ``(rel_rms[K], flux_err[K])`` arrays for one galaxy frame."""
    win = _central_window(frame, crop_frac, margin)
    rotated = _cumulative_rotate(win, angle_step, n_rot, order)
    rms, flux = [], []
    for k in k_list:
        d_direct = block_mean(win, int(k))
        d_rot = block_mean(rotated, int(k))
        rms.append(_rel_rms(d_rot, d_direct))
        flux.append(_flux_err(d_rot, d_direct))
    return np.array(rms), np.array(flux), win.shape[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tng-dir", default=Config.TNG_SKIRT_DIR,
                   help="Directory of downloaded TNG galaxies "
                        f"(default {Config.TNG_SKIRT_DIR}).")
    p.add_argument("--band", default="VIS", choices=("VIS", "Y", "J", "H"),
                   help="Atlas band to test (default VIS — highest resolution).")
    p.add_argument("--orientation", type=int, default=1,
                   help="SKIRT viewpoint O1..O5 (default 1).")
    p.add_argument("--n-rot", type=int, default=360,
                   help="Number of successive rotations (default 360).")
    p.add_argument("--angle-step", type=float, default=1.0,
                   help="Degrees per rotation step (default 1.0 → 360×1° = 360°).")
    p.add_argument("--order", type=int, default=3,
                   help="Spline interpolation order (3 = cubic, default).")
    p.add_argument("--k-list", default=",".join(map(str, DEFAULT_K_LIST)),
                   help="Comma-separated downsample factors K.")
    p.add_argument("--crop-frac", type=float, default=0.999,
                   help="Flux fraction the central crop must enclose (default 0.999).")
    p.add_argument("--margin", type=float, default=1.05,
                   help="Extra factor on the crop half-side so rotation never "
                        "clips real flux (default 1.05).")
    p.add_argument("--tol", type=float, default=0.01,
                   help="Critical-K threshold on relative-RMS error (default 0.01 = 1%%).")
    p.add_argument("--max-galaxies", type=int, default=0,
                   help="Cap galaxies analysed (0 = all found).")
    p.add_argument("--out", default=None,
                   help="Output PNG (default <VIS_DIR>/tng_rotation_downsample.png).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]

    galaxies = list_tng_galaxies(args.tng_dir)
    if args.max_galaxies > 0:
        galaxies = galaxies[: args.max_galaxies]
    if not galaxies:
        print(f"✗ No TNG galaxies found under {args.tng_dir}")
        return 1

    print(f"Found {len(galaxies)} TNG galax{'y' if len(galaxies) == 1 else 'ies'}; "
          f"band={args.band} O{args.orientation}; "
          f"rotation = {args.n_rot}×{args.angle_step}° (spline order {args.order})")

    results = []  # (label, rms_array, flux_array, win_side)
    for gdir, gid in galaxies:
        path = tng_fits_path(gdir, gid, args.orientation, args.band)
        if not os.path.isfile(path):
            print(f"  ⚠ {gid}: missing {os.path.basename(path)} — skipping")
            continue
        frame = load_skirt_frame(path)
        rms, flux, side = analyse_galaxy(
            frame, k_list=k_list, angle_step=args.angle_step, n_rot=args.n_rot,
            order=args.order, crop_frac=args.crop_frac, margin=args.margin)
        k_crit = next((k for k, e in zip(k_list, rms, strict=True)
                       if np.isfinite(e) and e < args.tol), None)
        results.append((f"TNG{gid}", rms, flux, side, k_crit))
        kc = f"K≥{k_crit}" if k_crit is not None else f">{k_list[-1]} (none)"
        print(f"  TNG{gid}: win {side}²  "
              f"relRMS K=1 {rms[0]:.3f} → K={k_list[-1]} {rms[-1]:.4f}  "
              f"critical {kc}")

    if not results:
        print("✗ No galaxies analysed.")
        return 1

    # ---- plot ----
    fig, (ax_rms, ax_flux) = plt.subplots(1, 2, figsize=(13, 5))
    for label, rms, flux, _side, _kc in results:
        ax_rms.plot(k_list, rms, "o-", label=label)
        ax_flux.plot(k_list, flux, "o-", label=label)

    ax_rms.axhline(args.tol, color="k", ls="--", lw=1,
                   label=f"tol {args.tol:.0%}")
    ax_rms.set_yscale("log")
    ax_rms.set_xlabel("downsample factor  K  (block-mean)")
    ax_rms.set_ylabel("relative RMS error  ‖rot−direct‖ / ‖direct‖")
    ax_rms.set_title(f"Rotate ({args.n_rot}×{args.angle_step}° spline) then "
                     f"downsample\nvs direct downsample — {args.band}")
    ax_rms.grid(True, which="both", alpha=0.3)
    ax_rms.legend(fontsize=8)

    ax_flux.plot([], [])
    ax_flux.set_yscale("log")
    ax_flux.set_xlabel("downsample factor  K  (block-mean)")
    ax_flux.set_ylabel("fractional flux error  |Σrot − Σdirect| / Σdirect")
    ax_flux.set_title("Flux consistency vs K")
    ax_flux.grid(True, which="both", alpha=0.3)
    ax_flux.legend(fontsize=8)

    fig.tight_layout()
    out = args.out or os.path.join(Config.VIS_DIR, "tng_rotation_downsample.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Plot → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
