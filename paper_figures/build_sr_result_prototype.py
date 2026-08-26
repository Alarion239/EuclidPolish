#!/usr/bin/env python3
"""Build tightly framed image grids for synthetic and real SR results.

The two regimes are intentionally separate.  Each layout uses A4-derived
physical dimensions, then trims the exported canvas to one gutter-width around
its content.  There are no arrows, captions, row labels, scale bars, or display
notes.

Synthetic panels are rendered from local four-band LR/SR/HR FITS triplets.
Real panels are clean crops of local Euclid/SR/NEXUS browser exports; no
astronomical pixels are generated, sharpened, or repainted.
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from PIL import Image

plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
DEFAULT_EXPORTS = Path(
    os.environ.get("EUCLIDPOLISH_FIGURE_EXPORTS", "/Users/alarion239/Downloads")
)

A4_WIDTH_IN = 210.0 / 25.4
A4_HEIGHT_IN = 297.0 / 25.4
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
DPI = 300

SIDE_MARGIN_MM = 7.0
GRID_GUTTER_MM = 4.0
UNTITLED_GRID_TOP_MM = 23.0
TITLED_GRID_TOP_MM = 29.5

BANDS = ("VIS", "Y_E", "J_E", "H_E")
VIS_INDEX = BANDS.index("VIS")
H_INDEX = BANDS.index("H_E")

SYNTHETIC_EXAMPLES = (
    ROOT / "data" / "eval_results" / "syn-lens_0036_0",
    ROOT / "data" / "eval_results" / "syn-gal_0086_0",
    ROOT / "data" / "eval_results" / "syn-lens_0035_0",
    ROOT / "data" / "eval_results" / "syn-lens_0037_0",
    ROOT / "data" / "eval_results" / "syn-lens_0004_0",
)

SYNTHETIC_OUTPUT_STEM = OUT / "fig10_synthetic_reconstruction_grid"
REAL_OUTPUT_STEM = OUT / "fig11_real_reconstruction_grid"

# Every local browser plate is 3864 x 1564 px.  The matched field-16 VIS and
# H_E exports contain the same selected square insets in all three panels.
SELECTED_INSET_BOXES = (
    (57, 843, 537, 1323),
    (1331, 843, 1811, 1323),
    (2605, 843, 3085, 1323),
)
INSET_CROP_SIDE = 400

# Two matched field-16 selections are available in both displayed bands.  They
# are repeated to fill the five-row A4 prototype, as requested when local
# examples are limited.
REAL_EXAMPLES = (
    (
        "nexus-field_idx16_lr-sr-jwst_VIS_figure.png",
        "nexus-field_idx16_lr-sr-jwst_H_E_figure.png",
    ),
    (
        "nexus-field_idx16_lr-sr-jwst_VIS_figure (1).png",
        "nexus-field_idx16_lr-sr-jwst_H_E_figure (1).png",
    ),
)


def _load_cube(path: Path) -> np.ndarray:
    """Load one four-band ``(band, y, x)`` FITS cube."""
    if not path.is_file():
        raise FileNotFoundError(f"Required synthetic source is missing: {path}")
    cube = np.asarray(fits.getdata(path), dtype=np.float32)
    if cube.ndim != 3 or cube.shape[0] != len(BANDS):
        raise ValueError(
            f"Expected a four-band (band, y, x) cube at {path}; got {cube.shape}"
        )
    return cube


def _asinh_normalize(image: np.ndarray, *, knee: float = 100.0) -> np.ndarray:
    """Create one robust grayscale view while retaining native sampling."""
    values = np.asarray(image, dtype=np.float64)
    stretched = np.arcsinh(np.clip(values, 0.0, None) / knee)
    finite = stretched[np.isfinite(stretched)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    high = float(np.percentile(finite, 99.7))
    if not np.isfinite(high) or high <= 0.0:
        high = 1.0
    return np.clip(stretched / high, 0.0, 1.0).astype(np.float32)


def _normalize_composite_channels(
    cubes: Sequence[np.ndarray],
    band_index: int,
    *,
    knee: float = 100.0,
) -> list[np.ndarray]:
    """Normalize one band with a single display window shared by SR and HR."""
    stretched = [
        np.arcsinh(
            np.clip(np.asarray(cube[band_index], dtype=np.float64), 0.0, None)
            / knee
        )
        for cube in cubes
    ]
    finite_parts = [part[np.isfinite(part)] for part in stretched]
    finite_parts = [part for part in finite_parts if part.size]
    if not finite_parts:
        return [np.zeros_like(part, dtype=np.float32) for part in stretched]
    high = float(np.percentile(np.concatenate(finite_parts), 99.7))
    if not np.isfinite(high) or high <= 0.0:
        high = 1.0
    return [
        np.clip(part / high, 0.0, 1.0).astype(np.float32)
        for part in stretched
    ]


def _vis_h_false_colour(vis: np.ndarray, h_band: np.ndarray) -> np.ndarray:
    """Map normalized VIS to cyan-blue and H_E to amber-red."""
    red = h_band + 0.08 * vis
    green = 0.54 * (vis + h_band)
    blue = vis + 0.08 * h_band
    rgb = np.stack((red, green, blue), axis=-1)
    return np.power(np.clip(rgb, 0.0, 1.0), 0.92).astype(np.float32)


def _synthetic_row(directory: Path) -> tuple[np.ndarray, ...]:
    lr = _load_cube(directory / "original_stack.fits")
    sr = _load_cube(directory / "SR.fits")
    hr = _load_cube(directory / "HR.fits")

    vis_norm = _normalize_composite_channels((sr, hr), VIS_INDEX)
    h_norm = _normalize_composite_channels((sr, hr), H_INDEX)
    return (
        _asinh_normalize(lr[VIS_INDEX]),
        _asinh_normalize(lr[H_INDEX]),
        _vis_h_false_colour(vis_norm[0], h_norm[0]),
        _vis_h_false_colour(vis_norm[1], h_norm[1]),
    )


def _center_crop(image: np.ndarray, side: int) -> np.ndarray:
    height, width = image.shape[:2]
    if side <= 0 or side > min(height, width):
        raise ValueError(f"Crop side {side} is invalid for image shape {image.shape}")
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    return image[y0:y0 + side, x0:x0 + side]


def _export_panels(path: Path) -> tuple[np.ndarray, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Required real-data export is missing: {path}")
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        panels = [np.asarray(rgb.crop(box)) for box in SELECTED_INSET_BOXES]
    return tuple(_center_crop(panel, INSET_CROP_SIDE) for panel in panels)


def _real_row(vis_path: Path, h_path: Path) -> tuple[np.ndarray, ...]:
    vis_panels = _export_panels(vis_path)
    h_panels = _export_panels(h_path)

    vis_sr = np.asarray(vis_panels[1], dtype=np.float32).mean(axis=-1) / 255.0
    h_sr = np.asarray(h_panels[1], dtype=np.float32).mean(axis=-1) / 255.0
    return (
        vis_panels[0],
        h_panels[0],
        _vis_h_false_colour(vis_sr, h_sr),
        vis_panels[2],
    )


def _available_real_rows(exports: Path, count: int) -> list[tuple[np.ndarray, ...]]:
    rows: list[tuple[np.ndarray, ...]] = []
    for vis_filename, h_filename in REAL_EXAMPLES:
        vis_path = exports / vis_filename
        h_path = exports / h_filename
        if vis_path.is_file() and h_path.is_file():
            rows.append(_real_row(vis_path, h_path))
    if not rows:
        required = ", ".join(
            filename
            for pair in REAL_EXAMPLES
            for filename in pair
        )
        raise FileNotFoundError(
            f"No real-data exports were found in {exports}; expected one of: {required}"
        )
    seed_rows = list(rows)
    while len(rows) < count:
        rows.append(seed_rows[len(rows) % len(seed_rows)])
    return rows[:count]


def _render_page(
    rows: Sequence[Sequence[np.ndarray]],
    *,
    page_title: str | None,
    column_titles: Sequence[str],
    output_stem: Path,
    origin: str,
    grayscale_columns: frozenset[int] = frozenset(),
) -> tuple[Path, Path]:
    if not rows:
        raise ValueError("At least one row is required")
    if any(len(row) != len(column_titles) for row in rows):
        raise ValueError("Every image row must match the number of column titles")

    column_count = len(column_titles)
    panel_side_mm = (
        A4_WIDTH_MM
        - 2.0 * SIDE_MARGIN_MM
        - (column_count - 1) * GRID_GUTTER_MM
    ) / column_count
    grid_height_mm = (
        len(rows) * panel_side_mm
        + (len(rows) - 1) * GRID_GUTTER_MM
    )
    grid_top_mm = TITLED_GRID_TOP_MM if page_title else UNTITLED_GRID_TOP_MM
    grid_bottom_mm = A4_HEIGHT_MM - grid_top_mm - grid_height_mm
    if grid_bottom_mm < 0.0:
        raise ValueError("The requested image grid does not fit on an A4 page")

    figure = plt.figure(
        figsize=(A4_WIDTH_IN, A4_HEIGHT_IN),
        dpi=DPI,
        facecolor="white",
    )
    grid = figure.add_gridspec(
        len(rows),
        len(column_titles),
        left=SIDE_MARGIN_MM / A4_WIDTH_MM,
        right=1.0 - SIDE_MARGIN_MM / A4_WIDTH_MM,
        bottom=grid_bottom_mm / A4_HEIGHT_MM,
        top=1.0 - grid_top_mm / A4_HEIGHT_MM,
        wspace=GRID_GUTTER_MM / panel_side_mm,
        hspace=GRID_GUTTER_MM / panel_side_mm,
    )
    top_axes: list[plt.Axes] = []
    for row_index, row in enumerate(rows):
        for column_index, image in enumerate(row):
            axis = figure.add_subplot(grid[row_index, column_index])
            if row_index == 0:
                top_axes.append(axis)
            if column_index in grayscale_columns:
                axis.imshow(
                    image,
                    origin=origin,
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                )
            else:
                axis.imshow(image, origin=origin, interpolation="nearest")
            axis.set_axis_off()

    figure.canvas.draw()
    if page_title:
        figure.text(
            0.5,
            0.973,
            page_title,
            ha="center",
            va="top",
            color="#172033",
            fontsize=17.0,
            fontweight="bold",
        )
    column_title_y = 1.0 - (grid_top_mm - 5.5) / A4_HEIGHT_MM
    for axis, title in zip(top_axes, column_titles, strict=True):
        position = axis.get_position()
        figure.text(
            (position.x0 + position.x1) / 2.0,
            column_title_y,
            title,
            ha="center",
            va="bottom",
            color="#172033",
            fontsize=10.5,
            fontweight="bold",
        )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    save_options = {
        "bbox_inches": "tight",
        "facecolor": "white",
        "pad_inches": GRID_GUTTER_MM / 25.4,
    }
    figure.savefig(png_path, dpi=DPI, **save_options)
    figure.savefig(pdf_path, **save_options)
    plt.close(figure)
    return png_path, pdf_path


def build_pages(
    exports_directory: Path,
    output_directory: Path,
) -> tuple[Path, Path, Path, Path]:
    synthetic_rows = [_synthetic_row(path) for path in SYNTHETIC_EXAMPLES]
    real_rows = _available_real_rows(exports_directory, count=5)

    synthetic_stem = output_directory / SYNTHETIC_OUTPUT_STEM.name
    real_stem = output_directory / REAL_OUTPUT_STEM.name
    synthetic_png, synthetic_pdf = _render_page(
        synthetic_rows,
        page_title=None,
        column_titles=(
            "Euclid-like VIS",
            r"Euclid-like H$_E$",
            "SR composite",
            "Known HR truth",
        ),
        output_stem=synthetic_stem,
        origin="lower",
        grayscale_columns=frozenset((0, 1)),
    )
    real_png, real_pdf = _render_page(
        real_rows,
        page_title="Real-data reconstructions",
        column_titles=(
            "Euclid VIS",
            r"Euclid H$_E$",
            "SR composite",
            "NEXUS F200W",
        ),
        output_stem=real_stem,
        origin="upper",
    )
    return synthetic_png, synthetic_pdf, real_png, real_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exports-directory",
        type=Path,
        default=DEFAULT_EXPORTS,
        help="Directory containing local NEXUS browser publication exports",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=OUT,
        help="Destination for the two tightly framed grids",
    )
    args = parser.parse_args()
    outputs = build_pages(args.exports_directory, args.output_directory)
    for output in outputs:
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
