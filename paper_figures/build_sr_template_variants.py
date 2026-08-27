#!/usr/bin/env python3
"""Build five title-only super-resolution figure templates.

The templates are deliberately data-first: each panel is either loaded from a
local FITS cube or cropped from a local browser publication export.  The
builder only arranges and display-maps those pixels; it does not sharpen,
denoise, repaint, or otherwise alter the astronomical content.

Each standalone figure contains five rows of square science panels, 4 mm
between panels, and 4 mm of white padding around the complete title-and-grid
content.  PDFs are written to ``output/pdf`` and same-stem PNG previews plus
JSON provenance sidecars are written to ``paper_figures``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from PIL import Image

plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGURES = Path(__file__).resolve().parent
DEFAULT_PDF_DIRECTORY = ROOT / "output" / "pdf"
DEFAULT_EXPORTS = Path(
    os.environ.get("EUCLIDPOLISH_FIGURE_EXPORTS", "/Users/alarion239/Downloads")
)

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
MM_PER_INCH = 25.4
DPI = 300
ROW_COUNT = 5
PADDING_MM = 4.0
GAP_MM = 4.0
COLUMN_TITLE_HEIGHT_MM = 10.0
ASINH_KNEE_E = 100.0
DISPLAY_PERCENTILE = 99.7

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

EXPECTED_EXPORT_SIZE = (3864, 1564)
SELECTED_INSET_BOXES = (
    (57, 843, 537, 1323),
    (1331, 843, 1811, 1323),
    (2605, 843, 3085, 1323),
)
INSET_CROP_SIDE = 400
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


@dataclass(frozen=True)
class ColumnSpec:
    """One titled column in a template."""

    title: str
    panel_key: str
    grayscale: bool = False


@dataclass(frozen=True)
class TemplateSpec:
    """Declarative definition of one five-row figure template."""

    stem: str
    regime: str
    columns: tuple[ColumnSpec, ...]


@dataclass(frozen=True)
class LoadedRow:
    """Rendered panels and the source files from which they were derived."""

    panels: Mapping[str, np.ndarray]
    source_paths: tuple[Path, ...]


@dataclass(frozen=True)
class LayoutGeometry:
    """Physical dimensions for one tight standalone figure."""

    panel_side_mm: float
    width_mm: float
    height_mm: float
    grid_width_mm: float
    grid_height_mm: float


TEMPLATES = (
    TemplateSpec(
        stem="real_vis_h_sr_template",
        regime="real",
        columns=(
            ColumnSpec("VIS Dirty", "vis_dirty"),
            ColumnSpec(r"H$_E$ Dirty", "h_dirty"),
            ColumnSpec(r"VIS + H$_E$ SR", "vis_h_sr"),
            ColumnSpec("NEXUS F200W", "nexus"),
        ),
    ),
    TemplateSpec(
        stem="real_bandwise_sr_template",
        regime="real",
        columns=(
            ColumnSpec("VIS Dirty", "vis_dirty"),
            ColumnSpec("VIS SR", "vis_sr"),
            ColumnSpec(r"H$_E$ Dirty", "h_dirty"),
            ColumnSpec(r"H$_E$ SR", "h_sr"),
            ColumnSpec("NEXUS F200W", "nexus"),
        ),
    ),
    TemplateSpec(
        stem="real_input_composite_template",
        regime="real",
        columns=(
            ColumnSpec("VIS Dirty", "vis_dirty"),
            ColumnSpec(r"VIS + H$_E$ Dirty", "vis_h_dirty"),
            ColumnSpec(r"VIS + H$_E$ SR", "vis_h_sr"),
            ColumnSpec("NEXUS F200W", "nexus"),
        ),
    ),
    TemplateSpec(
        stem="synthetic_bandwise_template",
        regime="synthetic",
        columns=(
            ColumnSpec("VIS Dirty", "vis_dirty", grayscale=True),
            ColumnSpec("VIS SR", "vis_sr", grayscale=True),
            ColumnSpec("VIS HR", "vis_hr", grayscale=True),
            ColumnSpec(r"H$_E$ Dirty", "h_dirty", grayscale=True),
            ColumnSpec(r"H$_E$ SR", "h_sr", grayscale=True),
            ColumnSpec(r"H$_E$ HR", "h_hr", grayscale=True),
        ),
    ),
    TemplateSpec(
        stem="synthetic_vis_h_template",
        regime="synthetic",
        columns=(
            ColumnSpec(r"VIS + H$_E$ Dirty", "vis_h_dirty"),
            ColumnSpec(r"VIS + H$_E$ SR", "vis_h_sr"),
            ColumnSpec(r"VIS + H$_E$ HR", "vis_h_hr"),
        ),
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
    if cube.shape[1] != cube.shape[2]:
        raise ValueError(f"Expected square science planes at {path}; got {cube.shape}")
    return cube


def _shared_band_transfer(
    cubes: Sequence[np.ndarray],
    band_index: int,
    *,
    knee: float = ASINH_KNEE_E,
) -> tuple[np.ndarray, ...]:
    """Apply one per-band display window shared by LR, SR, and HR.

    The native tiers can have different pixel dimensions, so their finite
    stretched pixels are flattened before calculating the common percentile.
    """
    stretched = tuple(
        np.arcsinh(
            np.clip(np.asarray(cube[band_index], dtype=np.float64), 0.0, None)
            / knee
        )
        for cube in cubes
    )
    finite_parts = tuple(part[np.isfinite(part)] for part in stretched)
    finite_parts = tuple(part for part in finite_parts if part.size)
    if not finite_parts:
        return tuple(np.zeros_like(part, dtype=np.float32) for part in stretched)
    high = float(np.percentile(np.concatenate(finite_parts), DISPLAY_PERCENTILE))
    if not np.isfinite(high) or high <= 0.0:
        high = 1.0
    return tuple(
        np.clip(part / high, 0.0, 1.0).astype(np.float32)
        for part in stretched
    )


def _vis_h_false_colour(vis: np.ndarray, h_band: np.ndarray) -> np.ndarray:
    """Map normalized VIS to cyan-blue and H_E to amber-red."""
    if vis.shape != h_band.shape:
        raise ValueError(
            f"VIS and H_E panels must have the same shape; got {vis.shape}, {h_band.shape}"
        )
    red = h_band + 0.08 * vis
    green = 0.54 * (vis + h_band)
    blue = vis + 0.08 * h_band
    rgb = np.stack((red, green, blue), axis=-1)
    return np.power(np.clip(rgb, 0.0, 1.0), 0.92).astype(np.float32)


def _synthetic_row(directory: Path) -> LoadedRow:
    lr_path = directory / "original_stack.fits"
    sr_path = directory / "SR.fits"
    hr_path = directory / "HR.fits"
    lr = _load_cube(lr_path)
    sr = _load_cube(sr_path)
    hr = _load_cube(hr_path)

    vis_lr, vis_sr, vis_hr = _shared_band_transfer((lr, sr, hr), VIS_INDEX)
    h_lr, h_sr, h_hr = _shared_band_transfer((lr, sr, hr), H_INDEX)
    return LoadedRow(
        panels={
            "vis_dirty": vis_lr,
            "vis_sr": vis_sr,
            "vis_hr": vis_hr,
            "h_dirty": h_lr,
            "h_sr": h_sr,
            "h_hr": h_hr,
            "vis_h_dirty": _vis_h_false_colour(vis_lr, h_lr),
            "vis_h_sr": _vis_h_false_colour(vis_sr, h_sr),
            "vis_h_hr": _vis_h_false_colour(vis_hr, h_hr),
        },
        source_paths=(lr_path, sr_path, hr_path),
    )


def _center_crop(image: np.ndarray, side: int) -> np.ndarray:
    height, width = image.shape[:2]
    if side <= 0 or side > min(height, width):
        raise ValueError(f"Crop side {side} is invalid for image shape {image.shape}")
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    return image[y0:y0 + side, x0:x0 + side]


def _export_panels(path: Path) -> tuple[np.ndarray, ...]:
    """Extract clean inner crops without export titles, borders, or scale bars."""
    if not path.is_file():
        raise FileNotFoundError(f"Required real-data export is missing: {path}")
    with Image.open(path) as source:
        if source.size != EXPECTED_EXPORT_SIZE:
            raise ValueError(
                f"Expected browser export size {EXPECTED_EXPORT_SIZE} at {path}; "
                f"got {source.size}"
            )
        rgb = source.convert("RGB")
        panels = [np.asarray(rgb.crop(box)) for box in SELECTED_INSET_BOXES]
    cropped = tuple(_center_crop(panel, INSET_CROP_SIDE) for panel in panels)
    if any(panel.shape[:2] != (INSET_CROP_SIDE, INSET_CROP_SIDE) for panel in cropped):
        raise ValueError(f"Could not extract clean square inset panels from {path}")
    return cropped


def _rgb_gray(image: np.ndarray) -> np.ndarray:
    """Recover the lossless grayscale display intensity from an RGB export."""
    return np.asarray(image, dtype=np.float32).mean(axis=-1) / 255.0


def _real_row(vis_path: Path, h_path: Path) -> LoadedRow:
    vis_dirty, vis_sr, nexus = _export_panels(vis_path)
    h_dirty, h_sr, _h_nexus = _export_panels(h_path)
    vis_dirty_gray = _rgb_gray(vis_dirty)
    vis_sr_gray = _rgb_gray(vis_sr)
    h_dirty_gray = _rgb_gray(h_dirty)
    h_sr_gray = _rgb_gray(h_sr)
    return LoadedRow(
        panels={
            "vis_dirty": vis_dirty,
            "vis_sr": vis_sr,
            "h_dirty": h_dirty,
            "h_sr": h_sr,
            "vis_h_dirty": _vis_h_false_colour(vis_dirty_gray, h_dirty_gray),
            "vis_h_sr": _vis_h_false_colour(vis_sr_gray, h_sr_gray),
            "nexus": nexus,
        },
        source_paths=(vis_path, h_path),
    )


def _available_real_rows(exports: Path, count: int = ROW_COUNT) -> list[LoadedRow]:
    rows = [
        _real_row(exports / vis_filename, exports / h_filename)
        for vis_filename, h_filename in REAL_EXAMPLES
        if (exports / vis_filename).is_file() and (exports / h_filename).is_file()
    ]
    if not rows:
        required = ", ".join(filename for pair in REAL_EXAMPLES for filename in pair)
        raise FileNotFoundError(
            f"No matched real-data exports were found in {exports}; "
            f"expected one of: {required}"
        )
    seed_rows = tuple(rows)
    while len(rows) < count:
        rows.append(seed_rows[len(rows) % len(seed_rows)])
    return rows[:count]


def _layout_geometry(column_count: int, row_count: int = ROW_COUNT) -> LayoutGeometry:
    if column_count <= 0 or row_count <= 0:
        raise ValueError("Template row and column counts must be positive")
    width_limited_side = (
        A4_WIDTH_MM - 2.0 * PADDING_MM - (column_count - 1) * GAP_MM
    ) / column_count
    height_limited_side = (
        A4_HEIGHT_MM
        - 2.0 * PADDING_MM
        - COLUMN_TITLE_HEIGHT_MM
        - (row_count - 1) * GAP_MM
    ) / row_count
    panel_side = min(width_limited_side, height_limited_side)
    grid_width = column_count * panel_side + (column_count - 1) * GAP_MM
    grid_height = row_count * panel_side + (row_count - 1) * GAP_MM
    width = 2.0 * PADDING_MM + grid_width
    height = 2.0 * PADDING_MM + COLUMN_TITLE_HEIGHT_MM + grid_height
    if width > A4_WIDTH_MM + 1e-9 or height > A4_HEIGHT_MM + 1e-9:
        raise ValueError(f"Template geometry exceeds A4: {width:g} x {height:g} mm")
    return LayoutGeometry(panel_side, width, height, grid_width, grid_height)


def _validate_rows(rows: Sequence[LoadedRow], spec: TemplateSpec) -> None:
    if len(rows) != ROW_COUNT:
        raise ValueError(f"{spec.stem} requires exactly {ROW_COUNT} rows")
    required = {column.panel_key for column in spec.columns}
    for row_index, row in enumerate(rows):
        missing = required.difference(row.panels)
        if missing:
            raise ValueError(
                f"{spec.stem} row {row_index} is missing panels: {sorted(missing)}"
            )
        for key in required:
            image = row.panels[key]
            if image.ndim not in (2, 3) or image.shape[0] != image.shape[1]:
                raise ValueError(
                    f"{spec.stem} row {row_index} panel {key} is not square: "
                    f"{image.shape}"
                )


def _render_template(
    spec: TemplateSpec,
    rows: Sequence[LoadedRow],
    *,
    preview_directory: Path,
    pdf_directory: Path,
) -> tuple[Path, Path, LayoutGeometry]:
    _validate_rows(rows, spec)
    geometry = _layout_geometry(len(spec.columns), len(rows))
    figure = plt.figure(
        figsize=(
            geometry.width_mm / MM_PER_INCH,
            geometry.height_mm / MM_PER_INCH,
        ),
        dpi=DPI,
        facecolor="white",
    )
    grid_bottom_mm = PADDING_MM
    grid_top_mm = PADDING_MM + geometry.grid_height_mm
    grid = figure.add_gridspec(
        len(rows),
        len(spec.columns),
        left=PADDING_MM / geometry.width_mm,
        right=1.0 - PADDING_MM / geometry.width_mm,
        bottom=grid_bottom_mm / geometry.height_mm,
        top=grid_top_mm / geometry.height_mm,
        wspace=GAP_MM / geometry.panel_side_mm,
        hspace=GAP_MM / geometry.panel_side_mm,
    )

    top_axes: list[plt.Axes] = []
    origin = "upper" if spec.regime == "real" else "lower"
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(spec.columns):
            axis = figure.add_subplot(grid[row_index, column_index])
            if row_index == 0:
                top_axes.append(axis)
            image = row.panels[column.panel_key]
            if column.grayscale:
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
    title_y = (
        grid_top_mm + 0.5 * COLUMN_TITLE_HEIGHT_MM
    ) / geometry.height_mm
    title_size = 9.0 if len(spec.columns) >= 6 else 10.0
    for axis, column in zip(top_axes, spec.columns, strict=True):
        position = axis.get_position()
        figure.text(
            (position.x0 + position.x1) / 2.0,
            title_y,
            column.title,
            ha="center",
            va="center",
            color="#172033",
            fontsize=title_size,
            fontweight="bold",
        )

    preview_directory.mkdir(parents=True, exist_ok=True)
    pdf_directory.mkdir(parents=True, exist_ok=True)
    png_path = preview_directory / f"{spec.stem}.png"
    pdf_path = pdf_directory / f"{spec.stem}.pdf"
    figure.savefig(
        png_path,
        dpi=DPI,
        facecolor="white",
        metadata={"Software": "EuclidPolish build_sr_template_variants.py"},
    )
    figure.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Title": spec.stem,
            "Creator": "EuclidPolish build_sr_template_variants.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return png_path, pdf_path, geometry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shown_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_provenance(
    spec: TemplateSpec,
    rows: Sequence[LoadedRow],
    *,
    png_path: Path,
    pdf_path: Path,
    geometry: LayoutGeometry,
) -> Path:
    unique_sources = tuple(dict.fromkeys(path for row in rows for path in row.source_paths))
    row_sources = [[_shown_path(path) for path in row.source_paths] for row in rows]
    payload = {
        "schema_version": 1,
        "template": spec.stem,
        "regime": spec.regime,
        "columns": [
            {
                "title": column.title,
                "panel_key": column.panel_key,
                "grayscale": column.grayscale,
            }
            for column in spec.columns
        ],
        "row_count": len(rows),
        "row_sources": row_sources,
        "sources": [
            {"path": _shown_path(path), "sha256": _sha256(path)}
            for path in unique_sources
        ],
        "display": {
            "false_colour": {
                "red": "H_E + 0.08 * VIS",
                "green": "0.54 * (VIS + H_E)",
                "blue": "VIS + 0.08 * H_E",
                "post_clip_power": 0.92,
                "role": "display-only, non-photometric",
            },
            "synthetic_band_transfer": {
                "formula": "asinh(max(pixel_e, 0) / knee_e) / shared_p99.7",
                "knee_e": ASINH_KNEE_E,
                "percentile": DISPLAY_PERCENTILE,
                "shared_across": "LR, SR, and HR within each row and band",
            },
            "real_band_transfer": (
                "preserved from matched 8-bit browser publication exports"
            ),
        },
        "layout_mm": {
            "standalone_width": geometry.width_mm,
            "standalone_height": geometry.height_mm,
            "panel_side": geometry.panel_side_mm,
            "horizontal_gap": GAP_MM,
            "vertical_gap": GAP_MM,
            "outer_padding": PADDING_MM,
            "column_title_height": COLUMN_TITLE_HEIGHT_MM,
        },
        "outputs": {
            "png": _shown_path(png_path),
            "pdf": _shown_path(pdf_path),
        },
    }
    path = png_path.with_suffix(".json")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def build_templates(
    *,
    exports_directory: Path,
    preview_directory: Path,
    pdf_directory: Path,
) -> tuple[Path, ...]:
    """Build all five declared standalone PNG/PDF templates and sidecars."""
    rows_by_regime: dict[str, Sequence[LoadedRow]] = {
        "real": _available_real_rows(exports_directory),
        "synthetic": tuple(_synthetic_row(path) for path in SYNTHETIC_EXAMPLES),
    }
    outputs: list[Path] = []
    for spec in TEMPLATES:
        rows = rows_by_regime[spec.regime]
        png_path, pdf_path, geometry = _render_template(
            spec,
            rows,
            preview_directory=preview_directory,
            pdf_directory=pdf_directory,
        )
        provenance_path = _write_provenance(
            spec,
            rows,
            png_path=png_path,
            pdf_path=pdf_path,
            geometry=geometry,
        )
        outputs.extend((png_path, pdf_path, provenance_path))
    return tuple(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exports-directory",
        type=Path,
        default=DEFAULT_EXPORTS,
        help="Directory containing matched field-16 browser publication exports",
    )
    parser.add_argument(
        "--preview-directory",
        type=Path,
        default=PAPER_FIGURES,
        help="Destination for PNG previews and JSON provenance sidecars",
    )
    parser.add_argument(
        "--pdf-directory",
        type=Path,
        default=DEFAULT_PDF_DIRECTORY,
        help="Destination for the five standalone PDF templates",
    )
    args = parser.parse_args()
    outputs = build_templates(
        exports_directory=args.exports_directory,
        preview_directory=args.preview_directory,
        pdf_directory=args.pdf_directory,
    )
    for output in outputs:
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
