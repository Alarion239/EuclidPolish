"""Reference values for the viewer colour engine (parity + golden fixtures).

Two modes:

``python scripts/_viewer_parity_ref.py | node scripts/check_viewer_parity.mjs``
    Prints the Python colour pipeline's values (``visualization/color.py``)
    as JSON on stdout. The Node script checks the SPA colour module
    (``frontend/src/viewer/color.ts``) against them: per-band calibration
    constants, the Planckian-locus hue chain and ``eye_rgb`` (which the
    viewer's Temp mode at the default knee/brightness reproduces).

``python scripts/_viewer_parity_ref.py --write-golden [--engine <js>]``
    Writes ``frontend/src/viewer/__fixtures__/color_golden.json``: the same
    Python references plus the OLD engine's outputs for small synthetic
    cubes in every colour mode (gray per band, gray-log, Lupton, Temp,
    direct-RGB, display scale, camera mismatch, NaN/±∞ pixels), computed
    by running the pre-rework ``static/cutout_viewer.js`` in Node
    (``check_viewer_parity.mjs --emit-engine``). The vitest suite
    ``src/viewer/color.test.ts`` checks the TS port against the file, so the
    port is held to the old engine and, where that engine claimed parity,
    to ``color.py``. The old engine was deleted with the port; regenerate
    from git::

        git show 0ad56d9:euclid_polish/web/static/cutout_viewer.js > /tmp/cv.js
        python scripts/_viewer_parity_ref.py --write-golden --engine /tmp/cv.js

Run from the repository root with the project environment (``PYTHONPATH``
is set up automatically from this file's location).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from euclid_polish.config import Config  # noqa: E402
from euclid_polish.photometry import electrons_to_ab_mag  # noqa: E402
from euclid_polish.visualization import color as C  # noqa: E402
from euclid_polish.web.helpers.viewer_data import color_constants  # noqa: E402

BANDS = list(Config.LR_INPUT_BAND_NAMES)
GOLDEN = REPO / "euclid_polish" / "web" / "frontend" / "src" / "viewer" / "__fixtures__" / "color_golden.json"
# The pre-rework engine is no longer in the tree; it is read from git.
ENGINE_COMMIT = "0ad56d9"
ENGINE_GIT_PATH = "euclid_polish/web/static/cutout_viewer.js"
CHECK_SCRIPT = REPO / "scripts" / "check_viewer_parity.mjs"

# Display-only JWST filters, shaped exactly like viewer_data's
# ``extra_color_bands`` entries (a pivot, no AB zero point).
JWST_BANDS = {
    "F115W": {"pivot_um": 1.154, "zeropoint_ab_e_total": 1.0, "display_only": True},
    "F200W": {"pivot_um": 1.990, "zeropoint_ab_e_total": 1.0, "display_only": True},
    "F444W": {"pivot_um": 4.421, "zeropoint_ab_e_total": 1.0, "display_only": True},
}


def python_refs() -> dict:
    """The values ``visualization/color.py`` computes (the parity anchor)."""
    out: dict = {"bands": {}, "color_meta": color_constants()}
    for name in BANDS:
        out["bands"][name] = {
            "ab_flux_norm": C._ab_flux_norm(name),
            "solar_balance": C._solar_balance(name),
        }

    temps = [2500.0, 3500.0, 5800.0, 8000.0, 12000.0, 20000.0]
    chain = []
    for T in temps:
        x, y = C._planckian_xy(np.array([T]))
        hue = C._xy_to_linear_srgb(x, y)[0]
        srgb = C._srgb_gamma_encode(hue)
        chain.append({"T": T, "srgb": [float(v) for v in srgb]})
    out["planck_chain"] = chain

    rng = np.random.default_rng(7)
    pixels = np.array([
        [100.0, 100.0, 100.0, 100.0],      # AB-ish flat
        [200.0, 150.0, 110.0, 90.0],       # blue-leaning
        [80.0, 120.0, 180.0, 240.0],       # red-leaning (rising to H)
        [10.0, 12.0, 9.0, 11.0],           # faint
        [3000.0, 2500.0, 2200.0, 2000.0],  # bright
    ], dtype=np.float32)
    pixels = pixels + rng.normal(0, 1, pixels.shape).astype(np.float32)
    cube = pixels[None, :, :]  # (1, N, 4) = (H, W, C)
    eye = C.eye_rgb(cube, band_names=Config.LR_INPUT_BAND_NAMES,
                    stretch="asinh", asinh_scale_e=float(Config.STRETCH_SCALE_E))
    out["eye_rgb"] = {
        "asinh_scale_e": float(Config.STRETCH_SCALE_E),
        "cube": cube.reshape(-1, 4).tolist(),
        "rgb": eye.reshape(-1, 3).tolist(),
    }
    return out


def _encode(values) -> list:
    """JSON-safe float list: NaN / ±∞ become the strings the TS side decodes."""
    out = []
    for v in np.asarray(values, dtype=np.float64).ravel():
        if math.isnan(v):
            out.append("NaN")
        elif math.isinf(v):
            out.append("Infinity" if v > 0 else "-Infinity")
        else:
            out.append(float(v))
    return out


def _cube(rng: np.random.Generator, h: int, w: int, c: int, scale: float,
          offset: float = 0.0) -> np.ndarray:
    """A small float32 cube with faint, bright and negative pixels."""
    a = rng.lognormal(mean=math.log(scale), sigma=1.4, size=(h, w, c)) + offset
    a[0, 0, :] = -0.3 * scale              # negatives clip to black
    a[0, 1, :] = 0.0
    a[-1, -1, :] = 40.0 * scale            # clipped white
    return a.astype(np.float32)


def golden_cases() -> list[dict]:
    """Inputs for the old engine: one per colour-mode code path."""
    rng = np.random.default_rng(20260926)
    four = _cube(rng, 3, 4, 4, 60.0)
    four_b = _cube(rng, 4, 3, 4, 900.0)
    special = _cube(rng, 2, 3, 4, 50.0)
    special[1, 0, 0] = np.nan
    special[1, 1, :] = np.nan
    special[1, 2, 1] = np.inf
    jwst1 = (rng.lognormal(mean=math.log(0.02), sigma=1.0, size=(3, 3, 1))).astype(np.float32)
    jwst3 = (rng.lognormal(mean=math.log(40.0), sigma=1.2, size=(3, 3, 3))).astype(np.float32)
    rgb3 = (rng.uniform(-20.0, 3500.0, size=(3, 3, 3))).astype(np.float32)
    psf = (rng.lognormal(mean=math.log(1e-4), sigma=2.5, size=(3, 4, 1))).astype(np.float32)
    psf[0, 0, 0] = 0.0
    psf4 = (rng.lognormal(mean=math.log(1e-3), sigma=2.0, size=(2, 2, 4))).astype(np.float32)
    many = (rng.normal(100.0, 40.0, size=(2, 2, 6))).astype(np.float32)

    cases: list[dict] = []

    def add(name, arr, color, transfers, **rec):
        h, w, c = arr.shape
        cases.append({
            "name": name,
            "rec": {"h": h, "w": w, "c": c, **rec},
            "data": _encode(arr),
            "color": color,
            "render_mode": None,
            "transfers": transfers,
        })

    t_default = [[100.0, 1.0, 100.0]]
    t_var = [[100.0, 1.0, 100.0], [37.0, 2.5, 100.0], [5.0, 0.1, 100.0], [5000.0, 10.0, 100.0]]
    for band in BANDS:
        add(f"gray-{band}", four, band, t_var, bands=BANDS)
    add("gray-VIS-no-header-bands", four, "VIS", t_default)
    add("gray-lupton-without-bands", four[..., :1], "lupton", t_default, bands=["VIS"])
    add("gray-temp-one-band", four[..., :1], "temp", t_default, bands=["VIS"])
    add("gray-unknown-color", four, "F200W", t_default, bands=BANDS)
    add("gray-special-values", special, "VIS", t_var, bands=BANDS)
    add("gray-display-scale-jwst", jwst1, "VIS", t_var, bands=["F200W"], displayScale=17743.00471101362)
    add("gray-display-scale-invalid", jwst1, "VIS", t_default, bands=["F200W"], displayScale=-2.0)
    add("gray-many-channels", many, "VIS", t_default, bands=[f"ch{i}" for i in range(6)])
    add("lupton", four, "lupton", t_var, bands=BANDS)
    add("lupton-b", four_b, "lupton", t_var, bands=BANDS)
    add("lupton-special-values", special, "lupton", t_default, bands=BANDS)
    add("lupton-display-scale", four, "lupton", t_default, bands=BANDS, displayScale=3.5)
    add("temp", four, "temp", t_var, bands=BANDS)
    add("temp-b", four_b, "temp", t_var, bands=BANDS)
    add("temp-special-values", special, "temp", t_default, bands=BANDS)
    add("temp-jwst-display-only", jwst3, "temp", t_var, bands=list(JWST_BANDS))
    add("direct-rgb", rgb3, "VIS", t_var, bands=["F444W", "F200W", "F115W"], directRgb=True)
    add("direct-rgb-scaled", rgb3, "lupton", t_default, bands=["F444W", "F200W", "F115W"],
        directRgb=True, displayScale=0.25)
    cases.append({
        "name": "gray-log-psf",
        "rec": {"h": 3, "w": 4, "c": 1, "bands": ["VIS"]},
        "data": _encode(psf), "color": "VIS", "render_mode": "log",
        "transfers": [[100.0, 1.0, 100.0], [100.0, 2.0, 100.0], [100.0, 0.4, 100.0]],
    })
    cases.append({
        "name": "gray-log-psf-band",
        "rec": {"h": 2, "w": 2, "c": 4, "bands": BANDS},
        "data": _encode(psf4), "color": "J_E", "render_mode": "log",
        "transfers": t_default,
    })
    # Temp mode at the Python eye_rgb inputs (default knee = asinh_scale_e).
    eye = python_refs()["eye_rgb"]
    cube = np.asarray(eye["cube"], dtype=np.float32)[None, :, :]
    cases.append({
        "name": "temp-eye-rgb", "rec": {"h": 1, "w": cube.shape[1], "c": 4, "bands": BANDS},
        "data": _encode(cube), "color": "temp", "render_mode": None,
        "transfers": [[eye["asinh_scale_e"], 1.0, eye["asinh_scale_e"]]],
    })
    return cases


def magnitude_refs() -> list[dict]:
    """Whole-cube AB magnitudes per band (photometry.electrons_to_ab_mag) and
    the SR ± from a std cube: δm = (2.5/ln 10)·Σσ/ΣSR."""
    rng = np.random.default_rng(3)
    out = []
    for k, band in enumerate(BANDS):
        sr = rng.lognormal(mean=math.log(80.0), sigma=1.0, size=(4, 4, 4)).astype(np.float32)
        std = (0.1 * sr * rng.uniform(0.5, 1.5, size=sr.shape)).astype(np.float32)
        tot = float(np.sum(sr[..., k].astype(np.float64)))
        stot = float(np.sum(std[..., k].astype(np.float64)))
        out.append({
            "band": band, "index": k, "h": 4, "w": 4, "c": 4, "bands": BANDS,
            "data": _encode(sr), "std": _encode(std),
            "tot": tot, "mag": float(electrons_to_ab_mag(tot, Config.get_band(band))),
            "std_tot": stot, "dm": (2.5 / math.log(10.0)) * stot / tot,
        })
    return out


def write_golden(engine: Path, node: str, path: Path) -> None:
    refs = python_refs()
    color_meta = refs["color_meta"]
    color_meta["bands"].update(JWST_BANDS)
    cases = golden_cases()
    request = {"color_meta": color_meta, "cases": cases}
    proc = subprocess.run(
        [node, str(CHECK_SCRIPT), "--emit-engine", str(engine)],
        input=json.dumps(request), capture_output=True, text=True, check=False,
        cwd=str(REPO),
    )
    if proc.returncode != 0:
        raise SystemExit(f"engine run failed:\n{proc.stderr}")
    engine_out = json.loads(proc.stdout)
    golden = {
        "about": ("Golden values for src/viewer/color.ts. 'engine' = outputs of the "
                  "pre-rework static/cutout_viewer.js (_internals) run in Node; 'python' = "
                  "visualization/color.py. Regenerate: see scripts/_viewer_parity_ref.py."),
        "engine_source": os.path.relpath(engine, REPO) if engine.is_relative_to(REPO) else str(engine),
        "color_meta": color_meta,
        "python": {k: v for k, v in refs.items() if k != "color_meta"},
        "magnitudes": magnitude_refs(),
        "engine": engine_out,
        "cases": cases,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(golden, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size} bytes, {len(cases)} cases)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write-golden", nargs="?", const=str(GOLDEN), default=None,
                        metavar="PATH", help="write the golden fixture (default: the SPA fixture path)")
    parser.add_argument("--engine", default=None,
                        help=f"the old engine module to run (required with --write-golden; "
                             f"`git show {ENGINE_COMMIT}:{ENGINE_GIT_PATH} > /tmp/cv.js`)")
    parser.add_argument("--node", default=os.environ.get("NODE", "node"), help="node binary")
    args = parser.parse_args()
    if args.write_golden:
        engine = Path(args.engine).resolve() if args.engine else None
        if engine is None or not engine.is_file():
            raise SystemExit(
                f"--write-golden needs the pre-rework engine (--engine <file>); it was deleted "
                f"with the port. Extract it first:\n"
                f"    git show {ENGINE_COMMIT}:{ENGINE_GIT_PATH} > /tmp/cv.js\n"
                f"    python scripts/_viewer_parity_ref.py --write-golden --engine /tmp/cv.js")
        write_golden(engine, args.node, Path(args.write_golden).resolve())
        return
    print(json.dumps(python_refs()))


if __name__ == "__main__":
    main()
