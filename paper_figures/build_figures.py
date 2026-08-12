#!/usr/bin/env python3
"""Build the first-pass EuclidPolish manuscript figure set.

The image comparisons are composed from the exact publication exports used in
the recent presentation.  Pixel values are never regenerated or retouched;
Pillow only places complete exports on white canvases and adds row/panel labels.
The calibration and diagnostic figures are rendered from reviewed repository
caches.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
DOWNLOADS = Path(os.environ.get("EUCLIDPOLISH_FIGURE_EXPORTS", "/Users/alarion239/Downloads"))
POSTER = ROOT / "poster" / "fig" / "poster"
DATA = ROOT / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INK = "#172033"
MUTED = "#5f6b7c"
BLUE = "#1267d6"
GREEN = "#2e8b57"
ORANGE = "#d95f02"
RED = "#b73a3a"
GRID = "#d8dee8"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _open_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Required figure source is missing: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def _save(image: Image.Image, path: Path) -> None:
    image.save(path, format="PNG", dpi=(300, 300), compress_level=8)


def _fit_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _header(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], *, size: int = 48) -> None:
    draw.text(xy, text, fill=INK, font=_font(size, bold=True))


def compose_pipeline() -> None:
    panels = [
        (POSTER / "ingredients_grid.png", "(a) Simulated sources"),
        (POSTER / "clean_field.png", "(b) High-resolution sky"),
        (POSTER / "dirty_grid.png", "(c) Mock Euclid inputs"),
        (POSTER / "cluster_map.png", "(d) PSF-star regions"),
        (POSTER / "psf_grid.png", "(e) Empirical four-band ePSFs"),
        (POSTER / "sr_field.png", "(f) Super-resolved output"),
    ]
    side = 1024
    gap = 34
    label_height = 88
    margin = 34
    canvas = Image.new(
        "RGB",
        (3 * side + 2 * gap + 2 * margin, 2 * (side + label_height) + gap + 2 * margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (path, label) in enumerate(panels):
        row, col = divmod(index, 3)
        x = margin + col * (side + gap)
        y = margin + row * (side + label_height + gap)
        _header(draw, label, (x, y + 12), size=46)
        panel = _open_rgb(path).resize((side, side), Image.Resampling.LANCZOS)
        canvas.paste(panel, (x, y + label_height))
    _save(canvas, OUT / "fig01_pipeline.png")


def render_population_calibrations() -> None:
    from euclid_polish.web.helpers.publication_figures import (
        render_population_atlas,
        render_star_population_calibration,
    )

    galaxy_path = (
        DATA / "population_comparison" / "calibrations" /
        "joint_galaxy_population_active.json"
    )
    star_path = (
        DATA / "population_comparison" / "calibrations" /
        "star_population_active.json"
    )
    galaxy = json.loads(galaxy_path.read_text())
    star = json.loads(star_path.read_text())
    if not (galaxy.get("active") and galaxy.get("valid") and galaxy.get("validated")):
        raise ValueError("The active joint galaxy calibration is not reviewed and valid")
    if not (star.get("active") and star.get("valid")):
        raise ValueError("The active stellar calibration is not valid")
    (OUT / "fig02_galaxy_population_calibration.png").write_bytes(
        render_population_atlas(galaxy, output_format="png", dpi=300)
    )
    (OUT / "fig03_stellar_population_calibration.png").write_bytes(
        render_star_population_calibration(star, output_format="png", dpi=300)
    )


def compose_rows(
    rows: Sequence[tuple[str, Sequence[Path]]],
    output: str,
    *,
    gap: int = 26,
    header_height: int = 92,
) -> None:
    loaded = [(label, [_open_rgb(path) for path in paths]) for label, paths in rows]
    target_width = min(image.width for _, images in loaded for image in images)
    fitted = [
        (label, [_fit_width(image, target_width) for image in images])
        for label, images in loaded
    ]
    row_widths = [sum(image.width for image in images) + gap * (len(images) - 1)
                  for _, images in fitted]
    row_heights = [max(image.height for image in images) for _, images in fitted]
    margin = 28
    canvas = Image.new(
        "RGB",
        (
            max(row_widths) + 2 * margin,
            sum(height + header_height for height in row_heights)
            + gap * (len(rows) - 1) + 2 * margin,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    y = margin
    for (label, images), row_height in zip(fitted, row_heights, strict=True):
        _header(draw, label, (margin, y + 12), size=44)
        y += header_height
        x = margin
        for image in images:
            canvas.paste(image, (x, y))
            x += image.width + gap
        y += row_height + gap
    _save(canvas, OUT / output)


def compose_grid(
    cells: Sequence[tuple[str, Path]],
    output: str,
    *,
    columns: int = 2,
    gap: int = 28,
    header_height: int = 80,
) -> None:
    images = [(label, _open_rgb(path)) for label, path in cells]
    target_width = min(image.width for _, image in images)
    images = [(label, _fit_width(image, target_width)) for label, image in images]
    cell_height = max(image.height for _, image in images)
    rows = (len(images) + columns - 1) // columns
    margin = 28
    canvas = Image.new(
        "RGB",
        (
            columns * target_width + (columns - 1) * gap + 2 * margin,
            rows * (header_height + cell_height) + (rows - 1) * gap + 2 * margin,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(images):
        row, col = divmod(index, columns)
        x = margin + col * (target_width + gap)
        y = margin + row * (header_height + cell_height + gap)
        _header(draw, label, (x, y + 10), size=38)
        canvas.paste(image, (x, y + header_height))
    _save(canvas, OUT / output)


def compose_image_comparisons() -> None:
    compose_rows(
        [
            (
                "(a) Synthetic test field 73",
                [
                    DOWNLOADS / "ensemble_idx73_lr_VIS_figure.png",
                    DOWNLOADS / "ensemble_idx73_sr_VIS_figure.png",
                    DOWNLOADS / "ensemble_idx73_hr_VIS_figure.png",
                ],
            ),
            (
                "(b) Synthetic test field 52",
                [
                    DOWNLOADS / "ensemble_idx52_lr_VIS_figure (1).png",
                    DOWNLOADS / "ensemble_idx52_sr_VIS_figure (1).png",
                    DOWNLOADS / "ensemble_idx52_hr_VIS_figure (1).png",
                ],
            ),
        ],
        "fig04_synthetic_lr_sr_hr_fields.png",
    )

    compose_grid(
        [
            ("(a) Evaluation field 38", DOWNLOADS / "evaluation_idx38_LR-SR_temp_figure.png"),
            ("(b) Evaluation field 66", DOWNLOADS / "evaluation_idx66_LR-SR_temp_figure.png"),
            ("(c) Evaluation field 48", DOWNLOADS / "evaluation_idx48_LR-SR_temp_figure.png"),
            ("(d) Evaluation field 35", DOWNLOADS / "evaluation_idx35_LR-SR_temp_figure.png"),
        ],
        "fig05_evaluation_morphology_gallery.png",
    )

    compose_rows(
        [
            (
                "(a) NEXUS field 31 · temperature composite",
                [DOWNLOADS / "nexus-field_idx31_lr-sr-jwst_temp_figure.png"],
            ),
            (
                "(b) NEXUS field 16 · Euclid VIS",
                [DOWNLOADS / "nexus-field_idx16_lr-sr-jwst_VIS_figure (1).png"],
            ),
        ],
        "fig06_nexus_widefield_comparisons.png",
    )

    compose_rows(
        [
            (
                "(a) NEXUS field 31 · selected 1\"-scale structure",
                [DOWNLOADS / "nexus-field_idx31_lr-sr-jwst_temp_figure (1).png"],
            ),
            (
                "(b) NEXUS field 16 · compact internal structure",
                [DOWNLOADS / "nexus-field_idx16_lr-sr-jwst_temp_figure.png"],
            ),
        ],
        "fig07_nexus_closeup_comparisons.png",
    )

    compose_rows(
        [
            (
                "(a) Saturated bright-star stress case",
                [DOWNLOADS / "nexus-field_idx88_lr-sr-jwst_VIS_figure.png"],
            ),
            (
                "(b) Compact extended-source stress case",
                [DOWNLOADS / "nexus-field_idx18_lr-sr-jwst_H_E_figure.png"],
            ),
        ],
        "fig08_stress_and_limitations.png",
    )


def _finite(values: Iterable[float | None]) -> np.ndarray:
    return np.asarray([np.nan if value is None else value for value in values], dtype=float)


def render_ensemble_diagnostics() -> None:
    payload_path = DATA / "vis" / "ensemble" / "starfull" / "ensemble_evals.json"
    payload = json.loads(payload_path.read_text())
    if payload.get("regime") != "starfull" or int(payload.get("n_fields", 0)) <= 0:
        raise ValueError("The starfull evaluation cache is unavailable")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11.5,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.2,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    })
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.45))
    fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.17, wspace=0.31)

    # (a) Spatial-frequency fidelity.
    ax = axes[0]
    ps = payload["ps"]
    theta = _finite(ps["theta"])
    for curve in ps.get("r_members", []):
        ax.plot(theta, _finite(curve), color="#8f96a3", alpha=0.24, linewidth=0.8)
    ax.plot(theta, _finite(ps.get("r_lr", [])), color=RED, linestyle="--",
            linewidth=2.2, label="LR baseline")
    ax.plot(theta, _finite(ps.get("r_cross", [])), color="#6f7682", linestyle=":",
            linewidth=1.8, label="model–model agreement")
    ax.plot(theta, _finite(ps["r"]), color=BLUE, linewidth=2.4,
            marker="o", markersize=3.0, label="ensemble mean")
    combiner = (ps.get("model_combiners") or {}).get("raw_incremental_minmeanmax_rbf") or {}
    if combiner.get("r"):
        ax.plot(theta, _finite(combiner["r"]), color=GREEN, linewidth=2.2,
                marker="o", markersize=2.6, label="convex RBF combiner")
    guides = payload.get("guides") or {}
    ax.axvline(float(guides.get("lr_scale", 0.1)), color=INK, linewidth=1.2,
               linestyle="--", alpha=0.8)
    ax.axvline(float(guides.get("vis_fwhm", 0.16)), color=BLUE, linewidth=1.2,
               linestyle="--", alpha=0.65)
    ax.set_xscale("log")
    ax.set_xlim(float(guides.get("theta_min", np.nanmin(theta))), np.nanmax(theta))
    ax.set_ylim(0, 1.04)
    ax.set_xlabel(r"angular scale $\theta=1/(2k)$ [arcsec]")
    ax.set_ylabel(r"cross-correlation with HR, $r(k)$")
    ax.set_title("(a) Spatial-frequency fidelity", loc="left", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)

    # (b) Does ensemble disagreement predict actual error?
    ax = axes[1]
    std_err = payload["std_err"]
    model = (std_err.get("models") or {}).get("ensemble_mean") or std_err
    edges_log = np.asarray(model["edges"], dtype=float)
    edges = np.power(10.0, edges_log)
    hist = np.asarray(model["hist"], dtype=float).T
    positive = hist[hist > 0]
    norm = mcolors.LogNorm(vmin=max(1.0, float(np.min(positive))), vmax=float(np.max(positive)))
    mesh = ax.pcolormesh(edges, edges, np.ma.masked_less_equal(hist, 0), cmap="viridis",
                         norm=norm, shading="auto", rasterized=True)
    median_std = np.power(10.0, _finite(model["med_std"]))
    median_err = np.power(10.0, _finite(model["med_err"]))
    ax.plot(median_std, median_err, color=ORANGE, linewidth=2.2, marker="o",
            markersize=3.0, label="median |error|")
    line = np.geomspace(edges[0], edges[-1], 200)
    ax.plot(line, line, color=INK, linestyle="--", linewidth=1.2, label=r"|error|=$\sigma$")
    ax.plot(line, 0.6745 * line, color=INK, linestyle=":", linewidth=1.2,
            label=r"Gaussian median = 0.674$\sigma$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(edges[0], edges[-1])
    ax.set_xlabel(r"cross-member pixel std $\sigma$ [e$^{-}$]")
    ax.set_ylabel(r"actual |ensemble mean $-$ HR| [e$^{-}$]")
    ax.set_title("(b) Disagreement versus error", loc="left", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.047, pad=0.02)
    colorbar.set_label("pixels per bin", fontsize=10)

    # (c) Pixel-level uncertainty calibration.
    ax = axes[2]
    calibration = payload["calibration"]
    z_edges = np.asarray(calibration["z_edges"], dtype=float)
    centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    pdf = _finite(calibration["pdf"])
    z = np.linspace(z_edges[0], z_edges[-1], 800)
    normal = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    ax.plot(z, normal, color=INK, linestyle="--", linewidth=1.5,
            label=r"standard normal")
    ax.plot(centers, pdf, color=BLUE, linewidth=2.0, label="ensemble z scores")
    ax.set_yscale("log")
    ax.set_xlim(z_edges[0], z_edges[-1])
    positive_pdf = pdf[np.isfinite(pdf) & (pdf > 0)]
    ax.set_ylim(max(1e-7, float(np.min(positive_pdf)) * 0.65), 1.5)
    ax.set_xlabel(r"$z=(\mathrm{ensemble\ mean}-\mathrm{HR})/\sigma$")
    ax.set_ylabel("pixel probability density")
    ax.set_title("(c) Uncertainty calibration", loc="left", fontweight="bold")
    stats = calibration.get("stats") or {}
    note = (
        f"σ(z) = {float(stats.get('sigma_z', np.nan)):.2f}\n"
        f"coverage |z| < 1: {100 * float(stats.get('cover1', np.nan)):.1f}%\n"
        f"coverage |z| < 2: {100 * float(stats.get('cover2', np.nan)):.1f}%\n"
        f"coverage |z| < 3: {100 * float(stats.get('cover3', np.nan)):.1f}%"
    )
    ax.text(0.03, 0.04, note, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, bbox={"facecolor": "white", "edgecolor": GRID, "alpha": 0.92})
    ax.legend(loc="upper right", frameon=False)

    for ax in axes:
        ax.grid(True, which="both", color=GRID, linewidth=0.55, alpha=0.72)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Ensemble diagnostics · star-containing regime · "
        f"{payload['n_members']} models · {payload['n_fields']} test fields",
        x=0.055, ha="left", fontsize=16.5, fontweight="bold",
    )
    fig.savefig(OUT / "fig09_ensemble_diagnostics.png", dpi=300, facecolor="white")
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    compose_pipeline()
    render_population_calibrations()
    compose_image_comparisons()
    render_ensemble_diagnostics()
    for path in sorted(OUT.glob("fig*.png")):
        with Image.open(path) as image:
            print(f"{path.name}: {image.width}x{image.height}  sha256={sha256(path)[:16]}")


if __name__ == "__main__":
    main()
