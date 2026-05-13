#!/usr/bin/env python
"""Render an Einstein-ring lens configuration (source near caustic).

The default :class:`LensPopulation` samples the source impact parameter
uniformly in a disk of radius ``LENS_SOURCE_OFFSET_FRAC × θ_E``, which
typically yields discrete double / quadruple images. This demo overrides
that to place the source very close to the optical axis so the lensed
images blend into a recognisable Einstein ring or arc.

Run:
    python scripts/demo_lens_ring.py
"""

from __future__ import annotations

import dataclasses
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
from euclid_polish.sky.lens_population import (
    LensPopulation, render_lens_to_multiband_canvas,
)


OUT = "data/vis/demo/lens_ring.png"


def main() -> int:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cat = open_cosmos2025()
    pop = LensPopulation(cat)

    # Sample several lenses and pick one with a comfortable θ_E, then
    # override the source position to sit ~5% of θ_E from the optical axis
    # so the resulting images form a near-ring rather than discrete spots.
    rng = np.random.default_rng(1)
    for _ in range(20):
        lp = pop.sample(rng)
        if 0.6 < lp.theta_E_arcsec < 1.6:
            break

    # Pin source very close to the caustic.
    near_axis = dataclasses.replace(
        lp,
        src_dx_arcsec=0.05 * lp.theta_E_arcsec,
        src_dy_arcsec=0.00,
    )
    H = W = 256
    canvas = np.zeros((H, W, Config.NUM_LR_CHANNELS), dtype=np.float32)
    # Centre the lens in the canvas
    near_axis = dataclasses.replace(near_axis, centre_x_pix=W / 2, centre_y_pix=H / 2)
    render_lens_to_multiband_canvas(canvas, params=near_axis)

    print(f"θ_E = {near_axis.theta_E_arcsec:.3f}\"  "
          f"(z_lens={near_axis.z_lens:.2f}, z_src={near_axis.z_source:.2f})")
    print(f"source offset = ({near_axis.src_dx_arcsec:.3f}, "
          f"{near_axis.src_dy_arcsec:.3f}) arcsec  "
          f"(= {near_axis.src_dx_arcsec / near_axis.theta_E_arcsec:.2%} of θ_E)")

    bands = Config.LR_INPUT_BAND_NAMES
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, k in zip(axes.flat, range(4)):
        scale = Config.get_band(bands[k]).asinh_stretch_scale_e
        stretched = np.arcsinh(canvas[..., k] / scale)
        vmin, vmax = np.percentile(stretched, [1.0, 99.7])
        ax.imshow(stretched, cmap="gray_r", origin="lower",
                  vmin=vmin, vmax=vmax)
        ax.set_title(f"{bands[k]}  (total {canvas[..., k].sum():.2e} e⁻)",
                     fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        theta_E_pix = near_axis.theta_E_arcsec / Config.DEFAULT_PIXEL_SCALE
        ax.add_patch(mpatches.Circle(
            (W / 2, H / 2), theta_E_pix,
            fill=False, ec="red", lw=1.2, ls="--",
        ))
    fig.suptitle(
        f"Einstein-ring config 256² (source 5% of θ_E off-axis,  "
        f"θ_E={near_axis.theta_E_arcsec:.2f}\")",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
