"""Run-level visual summary for grouped SR evaluation results."""

from __future__ import annotations

import csv
import os

import numpy as np

GROUP_COLORS = {
    "A": "#2a5db0",
    "B": "#2e8b57",
    "C": "#b8860b",
    "gal": "#17a2b8",
    "syn-lens": "#b03a3a",
    "syn-gal": "#7a4fb0",
}


def _group_color(group: str) -> str:
    return GROUP_COLORS.get(str(group), "#666666")


def _load_vis_plane(fits_path: str) -> np.ndarray:
    """Return the VIS plane of a FITS image or cube as float32."""
    from astropy.io import fits

    with fits.open(fits_path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
    return data[0] if data.ndim == 3 else data


def _stretch_to_uint8(
    plane: np.ndarray,
    *,
    asinh_scale: float,
    pmin: float = 1.0,
    pmax: float = 99.5,
) -> np.ndarray:
    """Asinh-stretch and percentile-normalize an image to uint8."""
    array = np.arcsinh(
        np.nan_to_num(plane, nan=0.0, posinf=0.0, neginf=0.0)
        / float(asinh_scale)
    )
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(plane.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [pmin, pmax])
    if hi <= lo:
        hi = lo + 1e-6
    normalized = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
    return (normalized * 255.0 + 0.5).astype(np.uint8)


def _group_triptych(run_dir: str, obj_id: str, asinh: float, height: int = 110):
    """Build an RGB uint8 strip containing the available LR, SR, and HR views."""
    from PIL import Image

    parts = []
    for filename in ("original_stack.fits", "SR.fits", "HR.fits"):
        path = os.path.join(run_dir, obj_id, filename)
        if not os.path.isfile(path):
            continue
        image = Image.fromarray(
            _stretch_to_uint8(_load_vis_plane(path), asinh_scale=asinh),
            mode="L",
        ).convert("RGB")
        image = image.resize((height, height), Image.BILINEAR)
        parts.append(np.asarray(image, dtype=np.uint8))
    if not parts:
        return None
    separator = np.full((height, 3, 3), 255, np.uint8)
    output = parts[0]
    for part in parts[1:]:
        output = np.concatenate([output, separator, part], axis=1)
    return output


def render_transformation_summary(run_dir: str, out_png: str) -> str | None:
    """Render SR-to-HR recovery, flux conservation, and example transforms."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest_path = os.path.join(run_dir, "manifest.csv")
    if not os.path.isfile(manifest_path):
        return None
    with open(manifest_path, newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("id") and str(row.get("ok", "")).lower() == "true"
        ]
    if not rows:
        return None
    groups = list(dict.fromkeys(row.get("grade", "") for row in rows))

    def _float(row, key):
        try:
            return float(row.get(key, ""))
        except (TypeError, ValueError):
            return None

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.4))

    synthetic = [
        row
        for row in rows
        if _float(row, "psnr_lr_hr") is not None
        and _float(row, "psnr_sr_hr") is not None
    ]
    if synthetic:
        lr_psnr = [_float(row, "psnr_lr_hr") for row in synthetic]
        sr_psnr = [_float(row, "psnr_sr_hr") for row in synthetic]
        axes[0].scatter(
            lr_psnr,
            sr_psnr,
            c=[_group_color(row.get("grade", "")) for row in synthetic],
            s=40,
        )
        lo, hi = min(lr_psnr + sr_psnr), max(lr_psnr + sr_psnr)
        axes[0].plot([lo, hi], [lo, hi], color="#888", ls="--", label="no change")
        axes[0].set_xlabel("PSNR  LR vs HR (dB)")
        axes[0].set_ylabel("PSNR  SR vs HR (dB)")
        improved = sum(sr > lr for lr, sr in zip(lr_psnr, sr_psnr, strict=False))
        axes[0].set_title(
            "SR vs HR recovery (synthetic)\n"
            f"above line = closer to truth: {improved}/{len(synthetic)}"
        )
        axes[0].legend(fontsize=9)
    else:
        axes[0].axis("off")
        axes[0].text(
            0.5,
            0.5,
            "No synthetic group\n(SR-vs-HR needs HR truth)",
            ha="center",
            va="center",
            color="#888",
        )

    populated_groups = [group for group in groups if group]
    for index, group in enumerate(populated_groups):
        ratios = [
            _float(row, "flux_ratio_sr_over_lr")
            for row in rows
            if row.get("grade") == group
            and _float(row, "flux_ratio_sr_over_lr") is not None
        ]
        if not ratios:
            continue
        jitter = np.random.default_rng(0).normal(index, 0.06, len(ratios))
        axes[1].scatter(jitter, ratios, color=_group_color(group), s=26, alpha=0.8)
        axes[1].scatter(
            [index],
            [np.mean(ratios)],
            color=_group_color(group),
            s=240,
            marker="_",
            lw=2.5,
        )
    axes[1].axhline(1.0, color="#888", ls=":", lw=1)
    axes[1].set_xticks(range(len(populated_groups)))
    axes[1].set_xticklabels(populated_groups)
    axes[1].set_xlabel("group")
    axes[1].set_ylabel("flux Σ SR / Σ LR")
    axes[1].set_title("Flux conservation by group\n(1 = conserved; bar = mean)")

    from euclid_polish.config import Config

    asinh = float(Config.STRETCH_SCALE_E)
    strips = []
    labels = []
    width = 0
    for group in populated_groups or [""]:
        example = next((row for row in rows if row.get("grade") == group), None)
        if example is None:
            continue
        strip = _group_triptych(run_dir, example["out_subdir"], asinh)
        if strip is None:
            continue
        strips.append(strip)
        labels.append(group or "?")
        width = max(width, strip.shape[1])
    if strips:
        padded = [
            np.pad(
                strip,
                ((0, 0), (0, width - strip.shape[1]), (0, 0)),
                constant_values=255,
            )
            for strip in strips
        ]
        gap = np.full((6, width, 3), 255, np.uint8)
        stacked = padded[0]
        ticks = [padded[0].shape[0] / 2]
        current_y = padded[0].shape[0]
        for strip in padded[1:]:
            stacked = np.concatenate([stacked, gap, strip], axis=0)
            ticks.append(current_y + gap.shape[0] + strip.shape[0] / 2)
            current_y += gap.shape[0] + strip.shape[0]
        axes[2].imshow(stacked)
        axes[2].set_yticks(ticks)
        axes[2].set_yticklabels(labels)
        axes[2].set_xticks([])
        axes[2].set_title("Example transform per group\nLR | SR | HR", fontsize=11)
    else:
        axes[2].axis("off")

    figure.suptitle(f"How the data transformed — {os.path.basename(run_dir)}", fontsize=13)
    figure.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    figure.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return out_png
