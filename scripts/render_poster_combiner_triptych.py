#!/usr/bin/env python3
"""Run the current STARFULL combiner on the poster galaxy and render a triptych.

The poster source is the cached four-band Euclid LR cube for the target at
18:12:55.413 +68:21:49.16.  This script runs the active STARFULL members one
at a time, applies the fitted raw incremental combiner, and writes a compact
FITS product plus a poster-style Euclid/SR/Hubble plate.

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
from euclid_polish.eval.combiner import load_combiner
from euclid_polish.model import Model


BANDS = tuple(Config.LR_INPUT_BAND_NAMES)
MEMBER_RE = re.compile(r"^(\d+)·psnr$")


def _load_lr(path: str, side: int) -> tuple[np.ndarray, fits.Header, dict]:
    with fits.open(path, memmap=False) as hdul:
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


def _run_members(lr: np.ndarray, *, ckpt_root: str, combiner) -> np.ndarray:
    predictions = []
    for label in combiner.member_labels:
        member_id = _member_id(label)
        member_dir = os.path.join(ckpt_root, f"member_{int(member_id):02d}")
        if not os.path.isfile(os.path.join(member_dir, "checkpoint")):
            raise FileNotFoundError(f"no checkpoint for combiner member {label}")
        print(f"  loading {label} …", flush=True)
        model = Model(member_dir, scale=Config.DEFAULT_REBIN_FACTOR,
                      num_res_blocks=Config.DEFAULT_NUM_RES_BLOCKS)
        pred = np.asarray(model.upsample_array(lr), dtype=np.float32)
        if pred.ndim != 3 or pred.shape[-1] != len(BANDS):
            raise ValueError(f"member {label} returned {pred.shape}, expected 4-band SR")
        predictions.append(pred)
        del model
        tf.keras.backend.clear_session()
        gc.collect()
    return np.stack(predictions, axis=0)


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


def _render_individual_members(
    output_dir: str, contact_path: str, members: np.ndarray, labels: list[str],
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

    fig, axes = plt.subplots(2, 5, figsize=(15.0, 6.8), dpi=220,
                             facecolor="black")
    for ax, image, label in zip(axes.flat, vis, labels, strict=True):
        ax.imshow(_asinh_display_shared(image, vis), origin="lower", cmap="gray",
                  interpolation="nearest", vmin=0.0, vmax=1.0)
        ax.set_title(label, color="white", fontsize=12, fontweight="bold", pad=7)
        ax.set_axis_off()
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.94,
                        wspace=0.025, hspace=0.12)
    fig.savefig(contact_path, dpi=220, facecolor="black", edgecolor="none",
                pad_inches=0.04)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="data/euclid_inference/cutouts/ra273.23_dec68.36/original_stack.fits",
    )
    parser.add_argument("--hubble", default="poster/fig/poster/result_hubble.png")
    parser.add_argument("--ckpt-root", default="ckpt/ensemble")
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
    args = parser.parse_args()

    lr, source_header, metadata = _load_lr(args.source, args.side)
    metadata["RA"] = float(args.target_ra)
    metadata["DEC"] = float(args.target_dec)
    print(f"source={args.source}  cropped LR={lr.shape}  "
          f"RA={metadata['RA']:.6f} Dec={metadata['DEC']:+.6f}")

    combiner = load_combiner(args.combiner_root)
    if combiner is None:
        raise RuntimeError(f"no compatible combiner under {args.combiner_root}")
    print(f"combiner={combiner.kind}  members={combiner.member_labels}")
    members = _run_members(lr, ckpt_root=args.ckpt_root, combiner=combiner)
    _render_individual_members(args.individual_dir, args.individual_contact,
                               members, combiner.member_labels)
    sr = np.asarray(combiner.apply_field(members), dtype=np.float32)
    print(f"combiner output: {sr.shape}")

    metadata.update({
        "COMB_KIND": combiner.kind,
        "N_MEMBER": len(combiner.member_labels),
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
            "combiner": combiner.kind,
            "members": combiner.member_labels,
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
