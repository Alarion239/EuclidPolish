#!/usr/bin/env python
"""Consistency check: is arbitrary-angle spline rotation of a TNG galaxy safe
*before* a block-mean downsample, as a function of the downsample factor K?

Motivation
----------
The physical-redshift renderer uses exact quarter-turns at small integer rebin
factors and arbitrary orientations once the rebin is large enough. Arbitrary
rotation needs spline interpolation, which blurs. The intuition is that a
*large enough* downsample factor K averages that blur away, so rotate-then-
downsample becomes indistinguishable from a clean direct downsample. This
script validates where that policy is safe.

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
from typing import Literal

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import rotate as ndi_rotate  # noqa: E402

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config  # noqa: E402
from euclid_polish.image import ImageCube  # noqa: E402
from euclid_polish.tng._image import _block_mean, _load_tng_plane  # noqa: E402
from euclid_polish.tng.atlas import TNGGalaxy  # noqa: E402

DEFAULT_K_LIST = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32)
type SplineOrder = Literal[0, 1, 2, 3, 4, 5]


def _growth_radius_px(image: ImageCube, frac: float) -> float:
    """Radius (px) about the frame centre enclosing ``frac`` of the positive flux."""
    frame = image.plane()
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


def _central_window(image: ImageCube, frac: float, margin: float) -> ImageCube:
    """Crop a centred square holding the galaxy inside its inscribed circle.

    Half-side = ``r_frac · √2 · margin`` so the flux-enclosing disk of radius
    ``r_frac`` stays within the window's inscribed circle even after rotation —
    i.e. rotation never clips real flux, only (near-zero) sky in the corners.
    """
    frame = image.plane()
    r = _growth_radius_px(image, frac)
    half = int(np.ceil(r * np.sqrt(2.0) * margin))
    H, W = frame.shape
    cy, cx = H // 2, W // 2
    half = min(half, cy, cx)
    win = frame[cy - half: cy + half, cx - half: cx + half]
    # Even side keeps the centre on a pixel boundary and eases block_mean.
    if win.shape[0] % 2:
        win = win[:-1, :-1]
    return image.with_data(np.ascontiguousarray(win[..., None], dtype=np.float32))


def _cumulative_rotate(image: ImageCube, angle_step: float, n_rot: int,
                       order: SplineOrder) -> ImageCube:
    """Apply ``n_rot`` successive ``angle_step``-degree spline rotations.

    ``reshape=False`` keeps the array size; ``mode='constant'`` (cval 0) treats
    off-frame as sky. Negatives from spline overshoot are clipped to 0 each
    step (surface brightness is non-negative), matching how a rendered stamp
    would be used.
    """
    out = image.as_array(copy=True)
    for _ in range(n_rot):
        out = ndi_rotate(
            out,
            angle_step,
            axes=(1, 0),
            reshape=False,
            order=order,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
        np.maximum(out, 0.0, out=out)
    return image.with_data(out)


def _rel_rms(a: ImageCube, b: ImageCube) -> float:
    """Relative RMS error ``‖a − b‖ / ‖b‖`` (0 = identical)."""
    a_values = a.as_array()
    b_values = b.as_array()
    denom = float(np.sqrt(np.mean(b_values * b_values)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sqrt(np.mean((a_values - b_values) ** 2)) / denom)


def _flux_err(a: ImageCube, b: ImageCube) -> float:
    """Fractional flux difference ``|Σa − Σb| / Σb`` (block_mean conserves SB,
    so this also tracks total-flux drift after the pixel-count change)."""
    sb = float(b.data.sum())
    return float(abs(a.data.sum() - sb) / sb) if sb > 0 else float("nan")


def analyse_galaxy(image: ImageCube, *, k_list, angle_step, n_rot,
                   order: SplineOrder,
                   crop_frac, margin):
    """Return ``(rel_rms[K], flux_err[K])`` arrays for one galaxy frame."""
    win = _central_window(image, crop_frac, margin)
    rotated = _cumulative_rotate(win, angle_step, n_rot, order)
    rms, flux = [], []
    for k in k_list:
        d_direct = _block_mean(win, int(k))
        d_rot = _block_mean(rotated, int(k))
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

    galaxies = TNGGalaxy.discover(args.tng_dir)
    if args.max_galaxies > 0:
        galaxies = galaxies[: args.max_galaxies]
    if not galaxies:
        print(f"✗ No TNG galaxies found under {args.tng_dir}")
        return 1

    print(f"Found {len(galaxies)} TNG galax{'y' if len(galaxies) == 1 else 'ies'}; "
          f"band={args.band} O{args.orientation}; "
          f"rotation = {args.n_rot}×{args.angle_step}° (spline order {args.order})")

    results = []  # (label, rms_array, flux_array, win_side)
    for galaxy in galaxies:
        gid = galaxy.subhalo_id
        path = galaxy.fits_path(args.orientation, args.band)
        if not path.is_file():
            print(f"  ⚠ {gid}: missing {path.name} — skipping")
            continue
        image = _load_tng_plane(path, args.band)
        rms, flux, side = analyse_galaxy(
            image, k_list=k_list, angle_step=args.angle_step, n_rot=args.n_rot,
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
