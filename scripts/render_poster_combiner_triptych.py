#!/usr/bin/env python3
"""Run the current STARFULL combiner on the poster galaxy and render a triptych.

The poster source is the cached four-band Euclid LR cube for the target at
18:12:55.413 +68:21:49.16 — either the band-first ``original_stack.fits`` or
the ``LR_<band>`` extensions of an earlier results FITS written here (the same
electron-domain input).  This script runs the active STARFULL members, applies
the fitted combiner, and writes a compact FITS product plus a poster-style
Euclid/SR/Hubble plate.

The Hubble panel is the existing WFPC2 F814W poster reference.  It is kept as
the poster asset rather than redownloaded, so the comparison remains the same
target and field of view used in the original poster.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from astropy.io import fits
from PIL import Image as PILImage

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel
from euclid_polish.ensemble_registry import default_ensemble_dir, regime_labels
from euclid_polish.eval.combiner import (
    ACTIVE_COMBINER_KINDS,
    COMBINER_MODELS,
    load_combiner,
)
from euclid_polish.photometry import ab_mag_to_electrons
from euclid_polish.visualization.color import eye_rgb, planck_color_strip

BANDS = tuple(Config.LR_INPUT_BAND_NAMES)
MEMBER_RE = re.compile(r"^(\d+)·psnr$")


def _load_lr(path: str, side: int) -> tuple[np.ndarray, fits.Header, dict]:
    with fits.open(path, memmap=False) as hdul:
        names = {hdu.name for hdu in hdul}
        if all(f"LR_{band}" in names for band in BANDS):
            # A results FITS from this script: one LR extension per band.
            data = np.stack([np.asarray(hdul[f"LR_{band}"].data, np.float32)
                             for band in BANDS])
        else:
            data = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header.copy()
    if data.ndim != 3 or data.shape[0] != len(BANDS):
        raise ValueError(f"expected a band-first LR cube, got {data.shape}")
    cube = np.moveaxis(data, 0, -1)
    cube = _center_crop(cube, side)
    if cube.ndim != 3 or cube.shape[-1] != 4:
        raise ValueError(f"expected a four-band LR cube, got {cube.shape}")
    return cube, header, {
        "RA": 273.2308875,
        "DEC": 68.3636556,
        "PIXSCALE": float(header.get("PIXSCALE", 0.10)),
    }


def _member_id(label: str) -> str:
    match = MEMBER_RE.fullmatch(label)
    if match is None:
        raise ValueError(f"combiner contains a non-PSNR member label: {label!r}")
    return match.group(1)


def _run_members(lr: np.ndarray, *, ckpt_root: str,
                 labels: list[str]) -> np.ndarray:
    """Each member's SR in ``labels`` order, one STARFULL member at a time.

    Members load through :class:`EnsembleModel`, so each one self-corrects to
    its checkpoint's depth and un-stretches with its own asinh knee.
    """
    ensemble = EnsembleModel(ckpt_root, starless=False)
    by_label = dict(zip(ensemble.member_labels, ensemble.members, strict=True))
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise FileNotFoundError(f"no active STARFULL checkpoint for {missing}")
    predictions = []
    for label in labels:
        _member_id(label)
        print(f"  inferring {label} …", flush=True)
        pred = np.asarray(by_label[label].upsample_array(lr), dtype=np.float32)
        if pred.ndim != 3 or pred.shape[-1] != len(BANDS):
            raise ValueError(f"member {label} returned {pred.shape}, expected 4-band SR")
        predictions.append(pred)
        gc.collect()
    del ensemble, by_label
    tf.keras.backend.clear_session()
    return np.stack(predictions, axis=0)


def _active_combiner(combiner_root: str, ckpt_root: str):
    """The first fitted combiner whose members are the active STARFULL set."""
    labels = regime_labels(ckpt_root, starless=False)
    for kind in ACTIVE_COMBINER_KINDS:
        combiner = load_combiner(
            combiner_root, member_labels=labels,
            artifact_dir=COMBINER_MODELS[kind].artifact_dir,
        )
        if combiner is not None:
            return combiner
    return None


def _asinh_display(data: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(data, dtype=np.float32), 0.0, None)
    positive = values[values > 0]
    scale = float(np.percentile(positive, 90.0)) if positive.size else 1.0
    stretched = np.arcsinh(values / max(scale, 1e-12))
    lo, hi = np.percentile(stretched, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((stretched - lo) / (hi - lo), 0.0, 1.0)


def _asinh_display_shared(data: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Display ``data`` using one stretch fitted to all member images."""
    values = np.clip(np.asarray(data, dtype=np.float32), 0.0, None)
    ref = np.clip(np.asarray(reference, dtype=np.float32), 0.0, None)
    positive = ref[ref > 0]
    scale = float(np.percentile(positive, 90.0)) if positive.size else 1.0
    stretched_ref = np.arcsinh(ref / max(scale, 1e-12))
    lo, hi = np.percentile(stretched_ref, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    stretched = np.arcsinh(values / max(scale, 1e-12))
    return np.clip((stretched - lo) / (hi - lo), 0.0, 1.0)


def _center_crop(arr: np.ndarray, side: int) -> np.ndarray:
    side = min(int(side), arr.shape[0], arr.shape[1])
    y0 = (arr.shape[0] - side) // 2
    x0 = (arr.shape[1] - side) // 2
    return arr[y0:y0 + side, x0:x0 + side]


def _read_hubble(path: str, native_side: int) -> np.ndarray:
    with PILImage.open(path) as image:
        image = image.convert("L")
        arr = np.asarray(image, dtype=np.float32) / 255.0
    # PNG rows run top-down while every panel is drawn with origin="lower"
    # (FITS convention); flip so the reference keeps the Euclid orientation.
    arr = np.flipud(arr)
    # The reference is already a rendered poster panel; use a central crop to
    # match the Euclid LR field of view after converting pixel scales.
    return _center_crop(arr, native_side)


def _write_fits(path: str, lr: np.ndarray, sr: np.ndarray, header: fits.Header,
                *, metadata: dict) -> None:
    primary = fits.PrimaryHDU()
    for key, value in metadata.items():
        primary.header[key] = value
    primary.header["BUNIT"] = ("electron", "display products retain electron-domain arrays")
    primary.header["BANDS"] = (",".join(BANDS), "band order in extensions")
    primary.header["LRPIX"] = (0.10, "Euclid LR pixel scale, arcsec/pixel")
    primary.header["SRPIX"] = (0.05, "super-resolved pixel scale, arcsec/pixel")
    hdus = [primary]
    for index, name in enumerate(BANDS):
        hdu = fits.ImageHDU(lr[..., index], name=f"LR_{name}")
        hdu.header["PIXSCALE"] = 0.10
        hdus.append(hdu)
    for index, name in enumerate(BANDS):
        hdu = fits.ImageHDU(sr[..., index], name=f"SR_{name}")
        hdu.header["PIXSCALE"] = 0.05
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True, output_verify="silentfix")


def _render_triptych(path: str, euclid: np.ndarray, sr: np.ndarray,
                     hubble: np.ndarray, *, metadata: dict) -> None:
    panels = [
        (euclid, "Euclid VIS", "0.10\u2033/pix", "#1F6FB2"),
        (sr, "Super-resolved (ours)", "0.05\u2033/pix", "#2E8B57"),
        (hubble, "Hubble WFPC2 F814W", "0.046\u2033/pix", "#D9760A"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.45), dpi=240,
                             facecolor="black")
    for ax, (image, title, scale, color) in zip(axes, panels, strict=True):
        ax.imshow(_asinh_display(image), origin="lower", cmap="gray",
                  interpolation="nearest", vmin=0.0, vmax=1.0)
        ax.set_title(f"{title}\n{scale}", color="white", fontsize=13,
                     fontweight="bold", pad=12)
        ax.set_axis_off()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(3.0)
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.90,
                        wspace=0.035)
    fig.savefig(path, dpi=240, facecolor="black", edgecolor="none",
                pad_inches=0.03)
    plt.close(fig)


def _member_recipes(ckpt_root: str,
                    labels: list[str]) -> list[tuple[str, float]] | None:
    """``(loss, asinh knee in e⁻)`` per member from its ``origin.json``.

    A member without a recorded knee uses the per-band default. ``None`` when
    any member's loss is unknown, so the caller keeps the flat layout.
    """
    recipes = []
    for label in labels:
        path = os.path.join(ckpt_root, f"member_{int(_member_id(label)):02d}",
                            "origin.json")
        try:
            with open(path, encoding="utf-8") as handle:
                origin = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        loss = origin.get("loss_norm")
        if not loss:
            return None
        knee = origin.get("asinh_knee")
        recipes.append((str(loss).upper(),
                        float(Config.STRETCH_SCALE_E if knee is None else knee)))
    return recipes


def _recipe_grid(labels: list[str], recipes: list[tuple[str, float]],
                 ) -> tuple[list[str], list[float], dict[str, tuple[int, int]]]:
    """Rows grouped by loss, columns by knee; members sharing a recipe stack
    into repeated rows of that loss in member order.

    Returns the row names, the column knees and each label's ``(row, col)``.
    """
    knees = sorted({knee for _, knee in recipes})
    cells: dict[tuple[str, float], list[str]] = {}
    for label, recipe in sorted(zip(labels, recipes, strict=True),
                                key=lambda item: int(_member_id(item[0]))):
        cells.setdefault(recipe, []).append(label)
    row_names: list[str] = []
    positions: dict[str, tuple[int, int]] = {}
    for loss in sorted({loss for loss, _ in recipes}):
        depth = max(len(cells.get((loss, knee), [])) for knee in knees)
        for repeat in range(depth):
            for column, knee in enumerate(knees):
                cell = cells.get((loss, knee), [])
                if repeat < len(cell):
                    positions[cell[repeat]] = (len(row_names), column)
            row_names.append(loss)
    return row_names, knees, positions


def _temperature_panels(members: np.ndarray) -> list[np.ndarray]:
    """Each member in the viewer's "Temp" colour, on one shared stretch.

    :func:`eye_rgb` fits a per-pixel blackbody colour temperature to the
    AB-calibrated bands (hue) and applies an absolute asinh transfer to the
    VIS-equivalent intensity (brightness). Its knee and white point are the
    90th percentile of positive and the 99.5th percentile of all intensity
    across every member — the grey sheet's shared-stretch choice — so hue and
    brightness compare directly between members.
    """
    vis_ab0 = float(ab_mag_to_electrons(0.0, Config.get_band("VIS")))
    weights = np.array([vis_ab0 / float(ab_mag_to_electrons(0.0, Config.get_band(band)))
                        for band in BANDS], dtype=np.float32)
    intensity = (members[:, ::4, ::4, :] * weights).mean(axis=-1)
    positive = intensity[intensity > 0]
    knee = float(np.percentile(positive, 90.0)) if positive.size else 1.0
    white = max(float(np.percentile(intensity, 99.5)), 2.0 * knee)
    panels = []
    for member in members:
        rgb = eye_rgb(member, BANDS, asinh_scale_e=knee, white_e=white)
        panels.append(np.round(rgb * 255.0).astype(np.uint8))
    return panels


def _render_individual_members(
    output_dir: str, contact_path: str, members: np.ndarray, labels: list[str],
    *, recipes: list[tuple[str, float]] | None = None,
    temperature_path: str | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    vis = np.asarray(members[..., 0], dtype=np.float32)
    for image, label in zip(vis, labels, strict=True):
        member_id = _member_id(label)
        out = os.path.join(output_dir, f"member_{int(member_id):03d}_psnr.png")
        fig = plt.figure(figsize=(5.0, 5.0), dpi=220, facecolor="black")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(_asinh_display_shared(image, vis), origin="lower", cmap="gray",
                  interpolation="nearest", vmin=0.0, vmax=1.0)
        ax.set_axis_off()
        ax.set_title(label, color="white", fontsize=15, fontweight="bold", pad=10)
        fig.savefig(out, dpi=220, facecolor="black", edgecolor="none",
                    pad_inches=0.04)
        plt.close(fig)
    _render_member_sheet(
        contact_path, [_asinh_display_shared(image, vis) for image in vis],
        labels, recipes,
    )
    if temperature_path:
        _render_member_sheet(temperature_path, _temperature_panels(members),
                             labels, recipes, legend=True)


def _render_member_sheet(
    path: str, panels: list[np.ndarray], labels: list[str],
    recipes: list[tuple[str, float]] | None, *, legend: bool = False,
) -> None:
    """One panel per member: grey ``(H, W)`` in [0, 1] or ``(H, W, 3)`` RGB,
    on the loss × knee grid when ``recipes`` are known (else ID order), with
    a colour-temperature legend strip when ``legend``."""
    if recipes is None:
        ncols = min(5, max(1, len(labels)))
        nrows = (len(labels) + ncols - 1) // ncols
        row_names, knees = [], []
        positions = {label: divmod(index, ncols)
                     for index, label in enumerate(labels)}
    else:
        row_names, knees, positions = _recipe_grid(labels, recipes)
        nrows, ncols = len(row_names), len(knees)
    # Margins in inches, so the loss / knee headers fit at any grid size.
    left_in = 0.55 if row_names else 0.03
    top_in = 0.85 if knees else 0.25
    bottom_in = 0.95 if legend else 0.03
    width = ncols * 3.0 + left_in
    height = nrows * 3.4 + top_in + bottom_in
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width, height),
        dpi=220, facecolor="black", squeeze=False,
    )
    for ax in axes.flat:
        ax.set_visible(False)
    for image, label in zip(panels, labels, strict=True):
        ax = axes[positions[label]]
        ax.set_visible(True)
        if image.ndim == 2:
            ax.imshow(image, origin="lower", cmap="gray",
                      interpolation="nearest", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(image, origin="lower", interpolation="nearest")
        ax.set_title(label, color="white", fontsize=12, fontweight="bold", pad=7)
        ax.set_axis_off()
    fig.subplots_adjust(left=left_in / width, right=1 - 0.03 / width,
                        bottom=bottom_in / height, top=1 - top_in / height,
                        wspace=0.025, hspace=0.12)
    if legend:
        strip, temps = planck_color_strip()
        bar = fig.add_axes([0.30, 0.42 / height, 0.40, 0.20 / height])
        bar.imshow(strip, aspect="auto", extent=(0.0, 1.0, 0.0, 1.0))
        span = np.log(temps[-1] / temps[0])
        ticks = [3000, 5000, 10000, 20000]
        bar.set_xticks([np.log(t / temps[0]) / span for t in ticks],
                       [f"{t:,} K" for t in ticks])
        bar.set_yticks([])
        bar.tick_params(colors="white", labelsize=12)
        bar.set_title("per-pixel blackbody colour temperature (VIS+Y+J+H)",
                      color="white", fontsize=13, pad=6)
    for row, name in enumerate(row_names):
        box = axes[row, 0].get_position()
        fig.text(box.x0 - 0.12 / width, (box.y0 + box.y1) / 2, name,
                 rotation=90, ha="right", va="center", color="white",
                 fontsize=17, fontweight="bold")
    for column, knee in enumerate(knees):
        box = axes[0, column].get_position()
        fig.text((box.x0 + box.x1) / 2, box.y1 + 0.40 / height,
                 f"asinh knee {knee:g} e⁻", ha="center", va="bottom",
                 color="white", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=220, facecolor="black", edgecolor="none",
                pad_inches=0.04)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="data/euclid_inference/cutouts/ra273.23_dec68.36/original_stack.fits",
    )
    parser.add_argument("--hubble", default="poster/fig/poster/result_hubble.png")
    parser.add_argument("--ckpt-root", default=default_ensemble_dir())
    parser.add_argument("--combiner-root", default="data/vis/ensemble/starfull")
    parser.add_argument("--side", type=int, default=1024,
                        help="central Euclid LR side, matching the poster crop")
    parser.add_argument("--hubble-pixel-scale", type=float, default=0.046)
    parser.add_argument("--euclid-pixel-scale", type=float, default=0.10)
    parser.add_argument("--target-ra", type=float, default=273.2308875)
    parser.add_argument("--target-dec", type=float, default=68.3636556)
    parser.add_argument("--out-fits", default="poster/target_181255_combiner_results.fits")
    parser.add_argument("--out-png", default="poster/fig/poster/result_triptych_combiner.png")
    parser.add_argument("--individual-dir", default="poster/fig/poster/individual_sr")
    parser.add_argument("--individual-contact", default="poster/fig/poster/individual_sr_grid.png")
    parser.add_argument(
        "--individual-contact-temp", default=None,
        help="temperature-coloured member sheet; default <contact>_temp.png, "
             "empty string to skip",
    )
    parser.add_argument(
        "--members", default="",
        help="explicit comma-separated member IDs; bypasses the stored combiner",
    )
    args = parser.parse_args()

    lr, source_header, metadata = _load_lr(args.source, args.side)
    metadata["RA"] = float(args.target_ra)
    metadata["DEC"] = float(args.target_dec)
    print(f"source={args.source}  cropped LR={lr.shape}  "
          f"RA={metadata['RA']:.6f} Dec={metadata['DEC']:+.6f}")

    combiner = None
    if args.members.strip():
        ids = [item.strip() for item in args.members.split(",") if item.strip()]
        labels = [f"{int(item)}·psnr" for item in ids]
        combine_kind = "mean_explicit_members"
        print(f"explicit members={labels}")
    else:
        combiner = _active_combiner(args.combiner_root, args.ckpt_root)
        if combiner is None:
            raise RuntimeError(
                f"no combiner under {args.combiner_root} matches the active "
                "STARFULL members; fit one first")
        labels = combiner.member_labels
        combine_kind = combiner.kind
        print(f"combiner={combiner.kind}  members={labels}")
    members = _run_members(lr, ckpt_root=args.ckpt_root, labels=labels)
    temperature_contact = (
        f"{os.path.splitext(args.individual_contact)[0]}_temp.png"
        if args.individual_contact_temp is None else args.individual_contact_temp
    )
    _render_individual_members(args.individual_dir, args.individual_contact,
                               members, labels,
                               recipes=_member_recipes(args.ckpt_root, labels),
                               temperature_path=temperature_contact)
    if combiner is None:
        sr = np.asarray(np.mean(members, axis=0), dtype=np.float32)
    else:
        sr = np.asarray(combiner.apply_field(members), dtype=np.float32)
    print(f"combiner output: {sr.shape}")

    metadata.update({
        "COMB_KIND": combine_kind,
        "N_MEMBER": len(labels),
        "LRSIDE": int(args.side),
    })
    _write_fits(args.out_fits, lr, sr, source_header, metadata=metadata)
    hubble_native_side = round(sr.shape[0] * (args.euclid_pixel_scale / 2.0)
                               / args.hubble_pixel_scale)
    hubble = _read_hubble(args.hubble, hubble_native_side)
    _render_triptych(args.out_png, lr[..., 0], sr[..., 0], hubble,
                     metadata=metadata)
    with open(os.path.splitext(args.out_png)[0] + ".json", "w", encoding="utf-8") as handle:
        json.dump({
            "source": os.path.abspath(args.source),
            "hubble_reference": os.path.abspath(args.hubble),
            "combiner": combine_kind,
            "members": labels,
            "target_ra_deg": args.target_ra,
            "target_dec_deg": args.target_dec,
            "lr_shape": list(lr.shape),
            "sr_shape": list(sr.shape),
            "display": "per-panel asinh stretch, 0.5-99.5 percentile clip",
        }, handle, indent=2)
    print(f"wrote {args.out_fits}")
    print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
