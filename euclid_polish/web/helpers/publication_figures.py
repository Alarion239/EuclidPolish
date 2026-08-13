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
EUCLID_OBS = "#1267d6"
TNG = "#7a3db8"
STAR_MODEL = "#1267d6"
STAR_PREDICTION = "#d95f02"
STAR_MEASURED = "#7a3db8"


def _xy(
    payload: Mapping[str, Any], field: str, *, x_field: str = "x",
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite x/y pairs without turning unavailable bins into zero."""
    x = np.asarray(payload.get(x_field, payload.get("x", [])), dtype=float)
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
    zero_floor: bool = True,
) -> None:
    ax.set_title(
        title, loc="left", fontsize=PANEL_TITLE_SIZE,
        fontweight=700, pad=12,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logarithmic_y:
        ax.set_yscale("log")
    elif zero_floor:
        ax.set_ylim(bottom=0)
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=TICK_LABEL_SIZE, width=1.0, length=4.5)


def render_population_atlas(
    calibration: Mapping[str, Any], *, output_format: str = "png", dpi: int = 300,
) -> bytes:
    """Render the active Euclid-only brightness-radius calibration."""
    fmt = output_format.lower()
    if fmt not in {"png", "pdf", "svg"}:
        raise ValueError("output_format must be png, pdf, or svg")
    dpi = max(120, min(int(dpi), 600))
    brightness = calibration.get("magnitude_plot") or {}
    brightness_law = brightness.get("law") or {}
    plots = calibration.get("plots") or {}
    radius = plots.get("radius") or {}
    relation = plots.get("conditional_radius") or {}
    if not brightness_law or not radius or not relation:
        raise ValueError("Euclid joint fit has no publication diagnostics")

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
        generation_interval = brightness.get("generation_interval") or []
        if len(generation_interval) == 2:
            axes[0].axvline(
                float(generation_interval[1]), color="#168f65",
                linewidth=2.0, linestyle=(0, (2, 3)),
                label=r"generation faint cutoff ($n_\mathrm{gal}=100$ arcmin$^{-2}$)",
            )
        axes[0].set_xlim(14, 29)
        _finish_axis(
            axes[0], "Straight brightness law",
            "VIS 2FWHM aperture magnitude [AB]",
            "objects arcmin$^{-2}$ mag$^{-1}$",
            logarithmic_y=True,
        )

        radius_x, radius_observed = _xy(radius, "observed_density")
        if radius_x.size:
            axes[1].plot(
                radius_x, radius_observed, linestyle="none", marker="o",
                markersize=4.6, markerfacecolor=PAPER,
                markeredgecolor=EUCLID_OBS, markeredgewidth=1.2,
                label="Euclid PHZ/MER measured Sérsic $R_e$", zorder=4,
            )
        _line(
            axes[1], radius, "density", color=TNG,
            label="joint-fit $R_e$ marginal", width=2.7,
        )
        _finish_axis(
            axes[1], "Euclid half-light radius",
            r"log$_{10}(R_{e,\mathrm{VIS\ S\acute{e}rsic}}/\mathrm{arcsec})$",
            r"objects arcmin$^{-2}$ dex$^{-1}$",
            logarithmic_y=True,
        )

        relation_payload = {
            "x": relation.get("magnitude") or [],
            "observed": relation.get("observed_mean_log10_arcsec") or [],
            "model": relation.get("model_mean_log10_arcsec") or [],
            "low": relation.get("model_low_log10_arcsec") or [],
            "high": relation.get("model_high_log10_arcsec") or [],
        }
        relation_x, relation_low = _xy(relation_payload, "low")
        _, relation_high = _xy(relation_payload, "high")
        if relation_x.size and relation_x.size == relation_high.size:
            axes[2].fill_between(
                relation_x, relation_low, relation_high,
                color=TNG, alpha=0.14, linewidth=0,
                label=r"constant 1$\sigma$ scatter",
            )
        _observed(
            axes[2], relation_payload, color=EUCLID_OBS,
            label="Euclid binned mean",
        )
        _line(
            axes[2], relation_payload, "model", color=TNG,
            label="joint conditional mean", width=2.7,
        )
        axes[2].set_xlim(14, 29)
        if len(generation_interval) == 2:
            axes[2].axvline(
                float(generation_interval[1]), color="#168f65",
                linewidth=2.0, linestyle=(0, (2, 3)),
                label="generation faint cutoff",
            )
        _finish_axis(
            axes[2], "Joint brightness-size relation",
            "VIS 2FWHM aperture magnitude [AB]",
            r"mean log$_{10}(R_e/\mathrm{arcsec})$",
            zero_floor=False,
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
    density = diagnostics.get("stellar_density_by_magnitude") or {}
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
        x_gaia, y_gaia = _xy(
            density, "gaia_observed", x_field="gaia_x",
        )
        x_gaia_fit, y_gaia_fit = _xy(
            density, "gaia_fitted", x_field="gaia_x",
        )
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
