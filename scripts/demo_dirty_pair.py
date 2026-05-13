#!/usr/bin/env python
"""Render a 512² field with ~3 lenses, then show the HR clean + LR dirty.

The lens density is set by ``Config.LENS_DENSITY_ARCMIN2``; at the
project default (16.5 / arcmin²) a 512² HR field covers 0.182 arcmin² so
the Poisson expectation is ~3 lenses.

Outputs:
    data/vis/demo/multilens_hr_clean.png   — 4-band clean HR (0.05"/pix)
    data/vis/demo/multilens_lr_dirty.png   — 4-band dirty LR (0.10"/pix)
    data/vis/demo/multilens_compare.png    — VIS HR vs VIS LR overlay

Run:
    python scripts/demo_dirty_pair.py
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from euclid_polish.config import Config
from euclid_polish.sky.cosmos2025 import open_cosmos2025
from euclid_polish.sky.multiband_forward import (
    MultiBandForward, MultiBandForwardConfig,
)
from euclid_polish.sky.multiband_generator import (
    MultiBandGeneratorConfig, MultiBandSimulator,
)


OUT_DIR = os.path.join("data", "vis", "demo")


def _percentile_clip(arr, lo=1.0, hi=99.7):
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def render_4panel(data_4ch, title, out_path, lens_positions=None,
                  lens_theta_E_arcsec=None, pixel_scale=0.05):
    """Save a 2×2 grid of band panels with optional lens-circle overlays."""
    bands = Config.LR_INPUT_BAND_NAMES
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax, k in zip(axes.flat, range(4)):
        scale = Config.get_band(bands[k]).asinh_stretch_scale_e
        stretched = np.arcsinh(data_4ch[..., k] / scale)
        vmin, vmax = _percentile_clip(stretched)
        ax.imshow(stretched, cmap="gray_r", origin="lower",
                  vmin=vmin, vmax=vmax)
        ax.set_title(f"{bands[k]}  (total {data_4ch[..., k].sum():.2e} e⁻)",
                     fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if lens_positions:
            for (xp, yp), te in zip(lens_positions, lens_theta_E_arcsec):
                te_pix = te / pixel_scale
                ax.add_patch(mpatches.Circle(
                    (xp, yp), te_pix,
                    fill=False, ec="red", lw=1.0, ls="--",
                ))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ wrote {out_path}")


def render_hr_lr_compare(hr_4ch, lr_4ch, lens_positions_hr,
                          lens_theta_E_arcsec, out_path):
    """Single VIS pair: HR clean (0.05"/pix) next to LR dirty (0.10"/pix)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    hr_scale = Config.BAND_VIS.asinh_stretch_scale_e
    hr_stretched = np.arcsinh(hr_4ch[..., 0] / hr_scale)
    vmin, vmax = _percentile_clip(hr_stretched)
    axes[0].imshow(hr_stretched, cmap="gray_r", origin="lower",
                   vmin=vmin, vmax=vmax)
    axes[0].set_title(f"HR clean VIS  ({hr_4ch.shape[1]}×{hr_4ch.shape[0]} px @ 0.05\"/pix)",
                      fontsize=11)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for (xp, yp), te in zip(lens_positions_hr, lens_theta_E_arcsec):
        te_pix = te / 0.05
        axes[0].add_patch(mpatches.Circle((xp, yp), te_pix,
                                          fill=False, ec="red", lw=1.0, ls="--"))

    lr_stretched = np.arcsinh(lr_4ch[..., 0] / hr_scale)
    vmin, vmax = _percentile_clip(lr_stretched)
    axes[1].imshow(lr_stretched, cmap="gray_r", origin="lower",
                   vmin=vmin, vmax=vmax)
    axes[1].set_title(f"LR dirty VIS  ({lr_4ch.shape[1]}×{lr_4ch.shape[0]} px @ 0.10\"/pix)",
                      fontsize=11)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    for (xp, yp), te in zip(lens_positions_hr, lens_theta_E_arcsec):
        # HR pixel → LR pixel (factor of 2)
        te_pix = te / 0.10
        axes[1].add_patch(mpatches.Circle((xp / 2.0, yp / 2.0), te_pix,
                                          fill=False, ec="red", lw=1.0, ls="--"))

    fig.suptitle("HR clean vs LR dirty (VIS) — red dashed circles mark lens θ_E",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ wrote {out_path}")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    cat = open_cosmos2025()
    sim = MultiBandSimulator(cat, MultiBandGeneratorConfig(image_size=512))
    fwd = MultiBandForward(config=MultiBandForwardConfig(add_noise=True))

    print(f"Config.LENS_DENSITY_ARCMIN2 = {Config.LENS_DENSITY_ARCMIN2}")
    print(f"Field area (512² HR @ 0.05\"/pix)  = "
          f"{(512 * 0.05 / 60.0)**2:.3f} arcmin² "
          f"→ E[lenses] = {Config.LENS_DENSITY_ARCMIN2 * (512 * 0.05 / 60.0)**2:.2f}")

    sky_hr, meta = sim.simulate_field(np.random.default_rng(2))
    print(f"\nGenerated field: {meta['n_galaxies']} galaxies, "
          f"{meta['n_stars']} stars, {meta['n_lenses']} lenses")
    for i, L in enumerate(meta["lenses"]):
        print(f"  lens {i}: ({L['x_pix']:6.1f}, {L['y_pix']:6.1f}) px,  "
              f"θ_E = {L['theta_E_arcsec']:.2f}\",  "
              f"z_l = {L['z_lens']:.2f}, z_s = {L['z_source']:.2f}")

    lens_positions = [(L["x_pix"], L["y_pix"]) for L in meta["lenses"]]
    lens_theta_E   = [L["theta_E_arcsec"] for L in meta["lenses"]]

    # 1. HR clean (4 panels)
    render_4panel(
        sky_hr.data, "HR clean 512² (4 bands, 0.05\"/pix) — red dashed = lens θ_E",
        os.path.join(OUT_DIR, "multilens_hr_clean.png"),
        lens_positions=lens_positions, lens_theta_E_arcsec=lens_theta_E,
        pixel_scale=0.05,
    )

    # 2. Forward → LR dirty
    lr, hr_target = fwd.process(sky_hr, rng=np.random.default_rng(42))
    print(f"\nForward model output:")
    print(f"  LR shape:      {lr.shape},  pixel_scale = {lr.pixel_scale_arcsec}\"/pix")
    print(f"  HR target:     {hr_target.shape},  band = {hr_target.band_names}")
    # For LR display, lens centroids are at HR_pixel / 2 (HR was trimmed to a
    # multiple of 6 but only at the trailing edge, so lead-edge coords map directly).
    render_4panel(
        lr.data, "LR dirty 512² → 255² @ 0.10\"/pix (4 bands, Poisson+read noise)",
        os.path.join(OUT_DIR, "multilens_lr_dirty.png"),
        lens_positions=[(x / 2.0, y / 2.0) for (x, y) in lens_positions],
        lens_theta_E_arcsec=lens_theta_E,
        pixel_scale=0.10,
    )

    # 3. HR vs LR (VIS only) overlay
    render_hr_lr_compare(
        sky_hr.data, lr.data, lens_positions, lens_theta_E,
        os.path.join(OUT_DIR, "multilens_compare.png"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
