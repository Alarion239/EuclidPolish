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


def _finish_axis(
    ax, title: str, xlabel: str, ylabel: str, *, logarithmic_y: bool = False,
) -> None:
    ax.set_title(
        title, loc="left", fontsize=PANEL_TITLE_SIZE,
        fontweight=700, pad=12,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logarithmic_y:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=TICK_LABEL_SIZE, width=1.0, length=4.5)


def render_population_atlas(
    fit: Mapping[str, Any], *, magnitude_plot: Mapping[str, Any] | None = None,
    output_format: str = "png", dpi: int = 300,
) -> bytes:
    """Render independent Q1 brightness and staged geometry in one figure.

    Unavailable bins remain missing. The joint cube contributes only the
    brightness-marginalized geometry; final brightness is the Q1 2FWHM law.
    """
    fmt = output_format.lower()
    if fmt not in {"png", "pdf", "svg"}:
        raise ValueError("output_format must be png, pdf, or svg")
    dpi = max(120, min(int(dpi), 600))
    diagnostics = fit.get("diagnostics") or {}
    redshift = diagnostics.get("redshift") or {}
    radius = diagnostics.get("angular_radius") or {}
    tng = diagnostics.get("tng_draw") or {}
    tng_full = tng.get("full") or {}
    brightness = magnitude_plot or {}
    brightness_law = brightness.get("law") or {}
    if not brightness_law or not redshift or not radius:
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

        observed_brightness = brightness.get("observed") or {}
        _line(
            axes[0], brightness_law, "density", color=TNG,
            label="Q1 2FWHM straight law", width=2.7,
        )
        observed_x, observed_y = _xy(observed_brightness, "density")
        if observed_x.size:
            axes[0].plot(
                observed_x, observed_y, linestyle="none", marker="o",
                markersize=4.6, markerfacecolor=PAPER,
                markeredgecolor=EUCLID_OBS, markeredgewidth=1.2,
                label="Q1 MER + PHZ 2FWHM counts", zorder=4,
            )
        fit_interval = brightness.get("fit_interval") or []
        if len(fit_interval) == 2:
            axes[0].axvspan(
                float(fit_interval[0]), float(fit_interval[1]),
                color=EUCLID_OBS, alpha=0.07, linewidth=0,
            )
        extrapolated = brightness.get("extrapolated_interval") or []
        if len(extrapolated) == 2:
            axes[0].axvspan(
                float(extrapolated[0]), float(extrapolated[1]),
                color=TNG, alpha=0.08, linewidth=0,
            )
        axes[0].set_xlim(14, 29)
        _finish_axis(
            axes[0], "Independent goal brightness",
            "VIS 2FWHM aperture magnitude [AB]",
            "objects arcmin$^{-2}$ mag$^{-1}$",
            logarithmic_y=True,
        )

        _observed(axes[1], redshift, color=COSMOS_MODEL, label="COSMOS data")
        _line(axes[1], redshift, "model", color=COSMOS_MODEL, label="COSMOS fit")
        _line(axes[1], tng_full.get("redshift") or {}, "density",
              color=TNG, label="staged geometry target", width=2.5)
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
              color=TNG, label="staged geometry target", width=2.5)
        _finish_axis(
            axes[2], "Angular-radius density",
            "log$_{10}$ angular radius / arcsec",
            "objects arcmin$^{-2}$ dex$^{-1}$",
        )

        handles, labels = [], []
        for ax in axes:
            for handle, label in zip(
                *ax.get_legend_handles_labels(), strict=True,
            ):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        fig.legend(
            handles, labels, loc="lower center", ncol=4, frameon=False,
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
    """Render the active Q1 PHZ × Gaia × Euclid calibration as one plate.

    The density panel shows the Q1/Gaia shared-slope straight-line calibration.
    The three colour panels separate the fitted true-colour population,
    inferred true colours, noise-simulated colours, and raw catalogue colours.
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
        x_fit, y_fit = _xy(density, "fitted")
        ax_density.plot(
            x_obs, y_obs, linestyle="none", marker="o", markersize=7,
            markerfacecolor=PAPER, markeredgecolor=INK, markeredgewidth=1.5,
            label="Q1 PHZ VIS counts", zorder=4,
        )
        ax_density.plot(
            x_fit, y_fit, color=STAR_MODEL, linewidth=2.5,
            label="Q1-normalized straight law", zorder=3,
        )
        x_gaia, y_gaia = _xy(density, "gaia_observed")
        x_gaia_fit, y_gaia_fit = _xy(density, "gaia_fitted")
        if x_gaia.size:
            ax_density.plot(
                x_gaia, y_gaia, linestyle="none", marker="s", markersize=4.2,
                markerfacecolor=PAPER, markeredgecolor=TNG,
                markeredgewidth=1.1, label="native Gaia G$_{AB}$ counts",
                zorder=4,
            )
        if x_gaia_fit.size:
            ax_density.plot(
                x_gaia_fit, y_gaia_fit, color=TNG, linewidth=2.1,
                linestyle=(0, (6, 3)), label="Gaia-intercept shared-slope fit",
                zorder=3,
            )
        fit_ranges = density.get("fit_ranges") or {}
        for key, color in (("q1", STAR_MODEL), ("gaia", TNG)):
            interval = fit_ranges.get(key) or []
            if len(interval) == 2:
                ax_density.axvspan(
                    float(interval[0]), float(interval[1]),
                    color=color, alpha=0.055, linewidth=0,
                )
        ax_density.set_title("Shared-slope stellar brightness laws", loc="left")
        ax_density.set_xlabel(str(density.get("x_label") or "VIS PSF magnitude [AB]"))
        ax_density.set_ylabel(str(density.get("unit") or "stars arcmin$^{-2}$ mag$^{-1}$"))
        ax_density.set_xlim(12, 25)
        ax_density.set_yscale("log")
        ax_density.legend(loc="upper left", frameon=False, fontsize=LEGEND_SIZE)

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
        provenance = calibration.get("population_provenance") or {}
        fig.suptitle(
            "Stellar population calibration · Q1 PHZ × Gaia DR3 × Euclid MER",
            x=0.075, y=0.985, ha="left",
            fontsize=FIGURE_TITLE_SIZE, fontweight=700, color=INK,
        )
        fig.text(
            0.075, 0.915,
            f"$n_\\star$ = {density_text}  ·  "
            f"$N_\\mathrm{{match}}$ = {int(coverage.get('high_quality_matched_rows', 0)):,}  ·  "
            f"$A_\\mathrm{{Q1}}$ = {float(provenance.get('area_deg2', 63.1)):g} deg$^2$",
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
