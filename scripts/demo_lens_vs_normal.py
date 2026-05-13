#!/usr/bin/env python
"""Render a 512² field with and without a strong lens (multi-band).

Usage:
    python scripts/demo_lens_vs_normal.py

Writes:
    data/vis/demo/normal_512.png       — 4-band 512² field, no lens
    data/vis/demo/lens_512.png         — same scene + 1 lens; red dashed circle = θ_E
    data/vis/demo/lens_zoom.png        — zoom on the lens

Uses the real COSMOS2025 catalog (mandatory). Galaxies + stars are
sampled from the default Poisson densities; the lens is forced via
``n_lenses=1`` (the default density is ~1/100 arcmin², which never hits
a 512² field on its own).
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")  # non-interactive; needed in headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from euclid_polish.config import Config
from euclid_polish.sky.cosmos2025 import open_cosmos2025
from euclid_polish.sky.multiband_generator import (
    MultiBandGeneratorConfig, MultiBandSimulator,
)


OUT_DIR = os.path.join("data", "vis", "demo")


def render_panel(data_4ch, title, out_path, lens_pos=None, theta_E_arcsec=None):
    bands = Config.LR_INPUT_BAND_NAMES
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax, k in zip(axes.flat, range(4)):
        scale = Config.get_band(bands[k]).asinh_stretch_scale_e
        stretched = np.arcsinh(data_4ch[..., k] / scale)
        vmin, vmax = np.percentile(stretched, [1.0, 99.8])
        ax.imshow(stretched, cmap="gray_r", origin="lower",
                  vmin=vmin, vmax=vmax)
        ax.set_title(f"{bands[k]}  (total {data_4ch[..., k].sum():.2e} e⁻)",
                     fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        if lens_pos is not None and theta_E_arcsec is not None:
            theta_E_pix = theta_E_arcsec / Config.DEFAULT_PIXEL_SCALE
            ax.add_patch(mpatches.Circle(
                lens_pos, theta_E_pix,
                fill=False, ec="red", lw=1.2, ls="--",
                label=f"θ_E = {theta_E_arcsec:.2f}\"",
            ))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ wrote {out_path}")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    cat = open_cosmos2025()
    sim = MultiBandSimulator(cat, MultiBandGeneratorConfig(image_size=512))

    print("Normal field (no lenses):")
    sky_normal, meta_normal = sim.simulate_field(
        np.random.default_rng(7), n_lenses=0,
    )
    print(f"  galaxies={meta_normal['n_galaxies']}, "
          f"stars={meta_normal['n_stars']}, lenses={meta_normal['n_lenses']}")
    render_panel(
        sky_normal.data,
        "Normal 512² field (no lenses)",
        os.path.join(OUT_DIR, "normal_512.png"),
    )

    print("\nLens field (forced 1 lens):")
    sky_lens, meta_lens = sim.simulate_field(
        np.random.default_rng(3), n_lenses=1,
    )
    L = meta_lens["lenses"][0]
    print(f"  galaxies={meta_lens['n_galaxies']}, "
          f"stars={meta_lens['n_stars']}, lenses={meta_lens['n_lenses']}")
    print(f"  lens centre = ({L['x_pix']:.1f}, {L['y_pix']:.1f}) px,  "
          f"θ_E = {L['theta_E_arcsec']:.3f}\",  "
          f"z_lens = {L['z_lens']:.2f},  z_src = {L['z_source']:.2f}")

    lens_pos = (L["x_pix"], L["y_pix"])
    render_panel(
        sky_lens.data,
        "Lens field 512² (red dashed = Einstein radius)",
        os.path.join(OUT_DIR, "lens_512.png"),
        lens_pos=lens_pos, theta_E_arcsec=L["theta_E_arcsec"],
    )

    # Tight zoom on the lens
    cx, cy = int(L["x_pix"]), int(L["y_pix"])
    theta_E_pix = int(np.ceil(L["theta_E_arcsec"] / Config.DEFAULT_PIXEL_SCALE))
    half = max(60, 4 * theta_E_pix)
    cy_lo, cy_hi = max(0, cy - half), min(512, cy + half)
    cx_lo, cx_hi = max(0, cx - half), min(512, cx + half)
    zoom = sky_lens.data[cy_lo:cy_hi, cx_lo:cx_hi]
    print(f"  zoom window: y∈[{cy_lo}, {cy_hi}) × x∈[{cx_lo}, {cx_hi})")
    render_panel(
        zoom,
        f"Lens zoom ({cx_hi - cx_lo}×{cy_hi - cy_lo} px,  "
        f"θ_E = {L['theta_E_arcsec']:.2f}\")",
        os.path.join(OUT_DIR, "lens_zoom.png"),
        lens_pos=(cx - cx_lo, cy - cy_lo),
        theta_E_arcsec=L["theta_E_arcsec"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
