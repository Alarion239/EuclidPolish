"""High-resolution, presentation-ready figures from cached WebUI artifacts.

These renderers deliberately consume the same compact payloads as the React
diagnostics.  They never re-fit a population or silently query an archive, so
an exported figure is a stable view of the data the user has already reviewed.
"""
from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any

import matplotlib
import numpy as np

from euclid_polish.visualization.presentation_style import (
    AXIS_LABEL_SIZE,
    FIGURE_TITLE_SIZE,
    LEGEND_SIZE,
    NOTE_SIZE,
    PANEL_TITLE_SIZE,
    TICK_LABEL_SIZE,
    presentation_rc,
)

PAPER = "#ffffff"
INK = "#172033"
MUTED = "#5f6b7c"
GRID = "#d8dee8"
COSMOS_MODEL = "#008c68"
EUCLID_OBS = "#1267d6"
TNG = "#7a3db8"
STAR_MODEL = "#1267d6"
STAR_PREDICTION = "#d95f02"
STAR_MEASURED = "#7a3db8"


def _xy(payload: Mapping[str, Any], field: str) -> tuple[np.ndarray, np.ndarray]:
    """Return finite x/y pairs without turning unavailable bins into zero."""
    x = np.asarray(payload.get("x", []), dtype=float)
    raw = payload.get(field, [])
    y = np.asarray([np.nan if value is None else value for value in raw], dtype=float)
    count = min(len(x), len(y))
    x, y = x[:count], y[:count]
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _observed(ax, payload: Mapping[str, Any], *, color: str, label: str) -> None:
    x, y = _xy(payload, "observed")
    ax.plot(
        x,
        y,
        linestyle="none",
        marker="o",
        markersize=5.4,
        markerfacecolor=PAPER,
        markeredgecolor=color,
        markeredgewidth=1.35,
        label=label,
        zorder=4,
    )


def _line(
    ax,
    payload: Mapping[str, Any],
    field: str,
    *,
    color: str,
    label: str,
    width: float = 2.2,
) -> None:
    x, y = _xy(payload, field)
    ax.plot(
        x,
        y,
        color=color,
        linewidth=width,
        linestyle="-",
        label=label,
        zorder=3,
    )


def _finish_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(
        title, loc="left", fontsize=PANEL_TITLE_SIZE,
        fontweight=700, pad=12,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=TICK_LABEL_SIZE, width=1.0, length=4.5)


def render_population_atlas(
    fit: Mapping[str, Any], *, output_format: str = "png", dpi: int = 300,
) -> bytes:
    """Render magnitude, redshift, and angular-size density in one figure.

    Unavailable bins remain missing and the TNG target is shown in full.
    """
    fmt = output_format.lower()
    if fmt not in {"png", "pdf", "svg"}:
        raise ValueError("output_format must be png, pdf, or svg")
    dpi = max(120, min(int(dpi), 600))
    diagnostics = fit.get("diagnostics") or {}
    magnitude = diagnostics.get("magnitude_counts") or {}
    redshift = diagnostics.get("redshift") or {}
    radius = diagnostics.get("angular_radius") or {}
    tng = diagnostics.get("tng_draw") or {}
    tng_full = tng.get("full") or {}
    if not magnitude or not redshift or not radius:
        raise ValueError("joint population fit has no publication diagnostics")

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.labelsize": AXIS_LABEL_SIZE,
    }
    with presentation_rc(style):
        fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.8))
        fig.subplots_adjust(
            left=0.062, right=0.992, bottom=0.23, top=0.93, wspace=0.30,
        )

        cosmos_mag = magnitude.get("cosmos") or {}
        euclid_mag = magnitude.get("euclid") or {}
        _observed(axes[0], cosmos_mag, color=COSMOS_MODEL, label="COSMOS data")
        _line(axes[0], cosmos_mag, "model", color=COSMOS_MODEL, label="COSMOS fit")
        _observed(axes[0], euclid_mag, color=EUCLID_OBS, label="Euclid data")
        _line(axes[0], euclid_mag, "model", color=EUCLID_OBS, label="Euclid fit")
        _line(axes[0], tng_full.get("magnitude") or {}, "density",
              color=TNG, label="TNG target", width=2.5)
        axes[0].set_xlim(18, 30)
        _finish_axis(
            axes[0], "Apparent-magnitude density", "survey AB magnitude",
            "objects arcmin$^{-2}$ mag$^{-1}$",
        )

        _observed(axes[1], redshift, color=COSMOS_MODEL, label="COSMOS data")
        _line(axes[1], redshift, "model", color=COSMOS_MODEL, label="COSMOS fit")
        _line(axes[1], tng_full.get("redshift") or {}, "density",
              color=TNG, label="TNG target", width=2.5)
        _finish_axis(
            axes[1], "Redshift density", "photometric redshift",
            r"objects arcmin$^{-2}$ $\Delta z^{-1}$",
        )

        cosmos_radius = radius.get("cosmos") or {}
        euclid_radius = radius.get("euclid") or {}
        _observed(axes[2], cosmos_radius, color=COSMOS_MODEL, label="COSMOS data")
        _line(axes[2], cosmos_radius, "model", color=COSMOS_MODEL,
              label="COSMOS fit")
        _observed(axes[2], euclid_radius, color=EUCLID_OBS, label="Euclid data")
        _line(axes[2], euclid_radius, "model", color=EUCLID_OBS,
              label="Euclid fit")
        _line(axes[2], tng_full.get("angular_radius") or {}, "density",
              color=TNG, label="TNG target", width=2.5)
        _finish_axis(
            axes[2], "Angular-radius density",
            "log$_{10}$ angular radius / arcsec",
            "objects arcmin$^{-2}$ dex$^{-1}$",
        )

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", ncol=5, frameon=False,
            fontsize=LEGEND_SIZE, handlelength=2.2,
            bbox_to_anchor=(0.5, 0.035), columnspacing=2.0,
        )

        buffer = io.BytesIO()
        fig.savefig(buffer, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return buffer.getvalue()


def render_star_population_calibration(
    calibration: Mapping[str, Any], *, output_format: str = "png", dpi: int = 300,
) -> bytes:
    """Render the active Gaia × Euclid stellar calibration as one plate.

    The density panel shows field-to-field variation without hiding it behind
    the fitted mean. The three colour panels separate the fitted true-colour
    population, inferred true colours, noise-simulated colours, and raw
    catalogue colours.
    """
    fmt = output_format.lower()
    if fmt not in {"png", "pdf", "svg"}:
        raise ValueError("output_format must be png, pdf, or svg")
    dpi = max(120, min(int(dpi), 600))
    diagnostics = calibration.get("diagnostics") or {}
    density = diagnostics.get("star_density_per_cone") or {}
    parameters = diagnostics.get("parameters") or {}
    colors = [parameters.get(key) or {} for key in ("vis_y", "y_j", "j_h")]
    if not density or any(not payload for payload in colors):
        raise ValueError("stellar calibration has no publication diagnostics")

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.labelsize": AXIS_LABEL_SIZE,
    }
    with presentation_rc(style):
        fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.8))
        fig.subplots_adjust(
            left=0.075, right=0.985, bottom=0.12, top=0.84,
            wspace=0.23, hspace=0.42,
        )

        ax_density = axes[0, 0]
        x_obs, y_obs = _xy(density, "observed")
        cone_count = len(x_obs)
        x_fit, y_fit = _xy(density, "fitted")
        ax_density.plot(
            x_obs, y_obs, linestyle="none", marker="o", markersize=7,
            markerfacecolor=PAPER, markeredgecolor=INK, markeredgewidth=1.5,
            label="probability-weighted cone", zorder=4,
        )
        ax_density.plot(
            x_fit, y_fit, color=STAR_MODEL, linewidth=2.5,
            linestyle=(0, (6, 3)), label="fitted survey mean", zorder=3,
        )
        ax_density.set_title("Point-source density by Euclid cone", loc="left")
        ax_density.set_xlabel("cone index")
        ax_density.set_ylabel(str(density.get("unit") or "point sources arcmin$^{-2}$"))
        if len(x_obs) <= 16:
            ax_density.set_xticks(x_obs)
        ax_density.set_ylim(bottom=0)
        ax_density.legend(loc="lower left", frameon=False, fontsize=LEGEND_SIZE)

        color_titles = {
            "vis_y": r"VIS $-$ Y$_E$ colour",
            "y_j": r"Y$_E$ $-$ J$_E$ colour",
            "j_h": r"J$_E$ $-$ H$_E$ colour",
        }
        legend_handles = None
        legend_labels = None
        for ax, key, payload in zip(
            (axes[0, 1], axes[1, 0], axes[1, 1]),
            ("vis_y", "y_j", "j_h"), colors, strict=True,
        ):
            x_fit, y_fit = _xy(payload, "fitted")
            x_obs, y_obs = _xy(payload, "observed")
            x_pred, y_pred = _xy(payload, "posterior_predictive")
            x_meas, y_meas = _xy(payload, "dirty_observed")
            ax.plot(
                x_fit, y_fit, color=STAR_MODEL, linewidth=2.5,
                label="Fitted true-colour population",
            )
            ax.plot(
                x_obs, y_obs, linestyle="none", marker="o", markersize=4.8,
                markerfacecolor=PAPER, markeredgecolor=INK,
                markeredgewidth=1.15,
                label="Estimated true colours of observed stars",
                zorder=5,
            )
            ax.plot(
                x_pred, y_pred, color=STAR_PREDICTION, linewidth=2.4,
                label="Estimated colours with simulated Euclid noise",
            )
            ax.plot(
                x_meas, y_meas, color=STAR_MEASURED, linewidth=2.0,
                linestyle=(0, (5, 3)), label="Raw Euclid catalogue colours",
            )
            ax.set_title(color_titles[key], loc="left")
            ax.set_xlabel("AB colour (mag)")
            ax.set_ylabel("probability density")
            ax.set_ylim(bottom=0)
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

        for ax in axes.ravel():
            ax.grid(True, color=GRID, linewidth=0.55, alpha=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(
                colors=INK, labelsize=TICK_LABEL_SIZE, width=1.0, length=4.5,
            )

        population = calibration.get("population") or {}
        coverage = calibration.get("coverage") or {}
        density_value = population.get("density_arcmin2")
        density_text = (
            f"{float(density_value):.2f} arcmin$^{{-2}}$"
            if density_value is not None else "—"
        )
        fig.suptitle(
            "Stellar population calibration · Gaia DR3 × Euclid MER",
            x=0.075, y=0.985, ha="left",
            fontsize=FIGURE_TITLE_SIZE, fontweight=700, color=INK,
        )
        fig.text(
            0.075, 0.915,
            f"$n_\\star$ = {density_text}  ·  "
            f"$N_\\mathrm{{match}}$ = {int(coverage.get('high_quality_matched_rows', 0)):,}  ·  "
            f"$N_\\mathrm{{cone}}$ = {cone_count}",
            ha="left", va="center", fontsize=NOTE_SIZE, color=MUTED,
        )
        if legend_handles and legend_labels:
            fig.legend(
                legend_handles, legend_labels, loc="lower center", ncol=4,
                frameon=False, fontsize=LEGEND_SIZE,
                bbox_to_anchor=(0.53, 0.025), columnspacing=1.8,
            )

        buffer = io.BytesIO()
        fig.savefig(buffer, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return buffer.getvalue()
