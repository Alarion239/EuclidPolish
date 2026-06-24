"""Emit JSON reference values for the cutout-viewer JS parity check.

Runs the Python colour pipeline (euclid_polish.visualization.color) on a small
fixed set of inputs and prints the results as JSON on stdout. The Node script
``scripts/check_viewer_parity.mjs`` consumes this and asserts the in-browser
renderer (``static/cutout_viewer.js``) matches to a tight tolerance.
"""
from __future__ import annotations

import json

import numpy as np

from euclid_polish.config import Config
from euclid_polish.visualization import color as C

BANDS = list(Config.LR_INPUT_BAND_NAMES)


def main() -> None:
    out = {"bands": {}}

    # Per-band calibration constants.
    for name in BANDS:
        out["bands"][name] = {
            "ab_flux_norm": C._ab_flux_norm(name),
            "solar_balance": C._solar_balance(name),
        }

    # Planckian-locus hue chain at a spread of temperatures.
    temps = [2500.0, 3500.0, 5800.0, 8000.0, 12000.0, 20000.0]
    chain = []
    for T in temps:
        x, y = C._planckian_xy(np.array([T]))
        hue = C._xy_to_linear_srgb(x, y)[0]
        srgb = C._srgb_gamma_encode(hue)
        chain.append({"T": T, "srgb": [float(v) for v in srgb]})
    out["planck_chain"] = chain

    # Full eye_rgb on a small synthetic cube of varied SEDs (1 row × N cols).
    rng = np.random.default_rng(7)
    # Mix a few deliberate SEDs: flat, red-rising, blue-rising, plus noise.
    pixels = np.array([
        [100.0, 100.0, 100.0, 100.0],     # AB-ish flat
        [200.0, 150.0, 110.0, 90.0],      # blue-leaning
        [80.0, 120.0, 180.0, 240.0],      # red-leaning (rising to H)
        [10.0, 12.0, 9.0, 11.0],          # faint
        [3000.0, 2500.0, 2200.0, 2000.0], # bright
    ], dtype=np.float32)
    pixels = pixels + rng.normal(0, 1, pixels.shape).astype(np.float32)
    cube = pixels[None, :, :]  # (1, N, 4) = (H, W, C)
    eye = C.eye_rgb(cube, band_names=Config.LR_INPUT_BAND_NAMES,
                    stretch="asinh", asinh_scale_e=float(Config.STRETCH_SCALE_E))
    out["eye_rgb"] = {
        "asinh_scale_e": float(Config.STRETCH_SCALE_E),
        "cube": cube.reshape(-1, 4).tolist(),   # row-major pixels
        "rgb": eye.reshape(-1, 3).tolist(),
    }

    print(json.dumps(out))


if __name__ == "__main__":
    main()
