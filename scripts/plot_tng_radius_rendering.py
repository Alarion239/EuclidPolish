#!/usr/bin/env python3
"""Plot offline pixelisation diagnostics for the one-pass TNG radius renderer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.skirt.image import measure_halflight_radius_px
from euclid_polish.sky.generation.tng_galaxy import (
    TNG_RADIUS_RENDERER_FINGERPRINT,
    TNG_RADIUS_RENDERING,
    list_tng_galaxies,
    tng_stamp_to_target_re,
)
from euclid_polish.sky.generation.tng_radius_manifest import (
    load_manifest,
    radius_lookup,
)

DEFAULT_TARGETS = (0.03, 0.05, 0.10, 0.30, 1.0, 10.0)


def _targets(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(not np.isfinite(item) or item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("targets must be positive finite arcseconds")
    return values


def build_diagnostic(
    *,
    tng_dir: str,
    manifest_path: str,
    targets: tuple[float, ...],
    image_size: int,
) -> dict:
    manifest = load_manifest(manifest_path)
    if not manifest or not manifest.get("valid"):
        raise ValueError("a valid TNG radius manifest is required")
    lookup = radius_lookup(manifest)
    galaxies = list_tng_galaxies(tng_dir)
    if not galaxies:
        raise ValueError(f"no complete TNG galaxies under {tng_dir!r}")
    max_output_side = 2 * int(image_size) + 1
    rows: list[dict] = []
    for galaxy_index, (galaxy_dir, subhalo_id) in enumerate(galaxies):
        for orientation in range(1, 6):
            native_re = lookup.get((str(subhalo_id), orientation))
            if native_re is None:
                continue
            for target_index, target in enumerate(targets):
                seed = (
                    10_000 * galaxy_index + 100 * orientation + target_index
                )
                stamp, metadata = tng_stamp_to_target_re(
                    galaxy_dir, subhalo_id, orientation, target,
                    rng=np.random.default_rng(seed),
                    pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                    native_re_px=native_re,
                    radius_manifest_fingerprint=str(
                        manifest.get("manifest_fingerprint", "")
                    ),
                    max_output_side=max_output_side,
                )
                measured = (
                    measure_halflight_radius_px(stamp[..., 0])
                    * Config.DEFAULT_PIXEL_SCALE
                )
                rows.append({
                    "subhalo_id": str(subhalo_id),
                    "orientation": orientation,
                    "nominal_re_arcsec": float(metadata["nominal_re_arcsec"]),
                    "offline_measured_re_arcsec": float(measured),
                    "radius_scale_factor": float(metadata["radius_scale_factor"]),
                    "render_support_clipped": bool(
                        metadata["render_support_clipped"]
                    ),
                    "stamp_shape": list(stamp.shape),
                })
    return {
        "version": 1,
        "radius_rendering": TNG_RADIUS_RENDERING,
        "radius_renderer_fingerprint": TNG_RADIUS_RENDERER_FINGERPRINT,
        "radius_manifest_fingerprint": manifest.get("manifest_fingerprint"),
        "pixel_scale_arcsec": Config.DEFAULT_PIXEL_SCALE,
        "image_size": int(image_size),
        "max_output_side": max_output_side,
        "targets_arcsec": list(targets),
        "rows": rows,
    }


def plot_diagnostic(payload: dict, output: str) -> None:
    rows = payload["rows"]
    nominal = np.asarray([row["nominal_re_arcsec"] for row in rows])
    measured = np.asarray([row["offline_measured_re_arcsec"] for row in rows])
    clipped = np.asarray([row["render_support_clipped"] for row in rows])
    bounds = (
        min(float(np.min(nominal)), float(np.min(measured))) * 0.8,
        max(float(np.max(nominal)), float(np.max(measured))) * 1.25,
    )
    figure, axis = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    axis.plot(bounds, bounds, color="#2b2b2b", linewidth=1.5, label="one-to-one")
    if np.any(~clipped):
        axis.scatter(
            nominal[~clipped], measured[~clipped], s=24,
            facecolors="none", edgecolors="#2478d4", linewidths=1.2,
            label="full rendered support",
        )
    if np.any(clipped):
        axis.scatter(
            nominal[clipped], measured[clipped], s=28,
            color="#e25543", marker="x", linewidths=1.4,
            label="field-bounded support",
        )
    pixel_scale = float(payload["pixel_scale_arcsec"])
    axis.axvline(
        pixel_scale, color="#888888", linestyle="--", linewidth=1.0,
        label=f"one HR pixel = {pixel_scale:g} arcsec",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(bounds)
    axis.set_ylim(bounds)
    axis.set_xlabel("Nominal Euclid Sersic half-light radius [arcsec]")
    axis.set_ylabel("Offline pixel-remeasured half-light radius [arcsec]")
    axis.grid(True, which="both", linewidth=0.5, alpha=0.22)
    axis.legend(frameon=False, fontsize=8)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tng-dir", default=Config.TNG_SKIRT_DIR)
    parser.add_argument(
        "--manifest",
        default=os.path.join(
            Config.DATA_DIR, "_tng_infographics", "tng_radius_manifest.json",
        ),
    )
    parser.add_argument(
        "--targets", type=_targets,
        default=DEFAULT_TARGETS,
        help="comma-separated nominal half-light radii in arcsec",
    )
    parser.add_argument("--image-size", type=int, default=Config.DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--output",
        default=os.path.join(
            Config.DATA_DIR, "_tng_infographics",
            "tng_radius_rendering_diagnostic.png",
        ),
    )
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)
    if args.image_size <= 0:
        parser.error("--image-size must be positive")
    payload = build_diagnostic(
        tng_dir=args.tng_dir, manifest_path=args.manifest,
        targets=args.targets, image_size=args.image_size,
    )
    plot_diagnostic(payload, args.output)
    json_path = args.json or str(Path(args.output).with_suffix(".json"))
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "plot": args.output,
        "json": json_path,
        "rows": len(payload["rows"]),
        "renderer": TNG_RADIUS_RENDERING,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
