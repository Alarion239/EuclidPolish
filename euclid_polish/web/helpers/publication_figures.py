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

from euclid_polish.population.euclid_galaxy_prior import (
    ConditionalRadiusLaw,
    joint_density_grid,
)
from euclid_polish.population.magnitude_law import (
    ContinuousBrightBridgeFaintCappedMagnitudeLaw,
    EmpiricalBrightFaintCappedMagnitudeLaw,
    FaintCappedMagnitudeLaw,
    StraightMagnitudeLaw,
)
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
PREVIOUS_MODEL = "#7a3db8"
CANDIDATE_MODEL = "#168f65"


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


def _comparison_laws(
    calibration: Mapping[str, Any], *, label: str,
) -> tuple[
    StraightMagnitudeLaw
    | FaintCappedMagnitudeLaw
    | EmpiricalBrightFaintCappedMagnitudeLaw
    | ContinuousBrightBridgeFaintCappedMagnitudeLaw,
    ConditionalRadiusLaw,
]:
    """Reconstruct one explicit brightness-radius calibration payload."""
    try:
        magnitude_payload = calibration["magnitude_law"]
        if magnitude_payload.get("kind") == (
            "continuous_three_slope_bright_bridge_main_flat_faint_counts"
        ):
            magnitude_law = (
                ContinuousBrightBridgeFaintCappedMagnitudeLaw.from_payload(
                    magnitude_payload,
                )
            )
        elif magnitude_payload.get("kind") == (
            "empirical_bright_straight_middle_flat_faint_counts"
        ):
            magnitude_law = (
                EmpiricalBrightFaintCappedMagnitudeLaw.from_payload(
                    magnitude_payload,
                )
            )
        elif magnitude_payload.get("kind") == (
            "faint_capped_straight_log10_differential_counts"
        ):
            magnitude_law = FaintCappedMagnitudeLaw.from_payload(
                magnitude_payload,
            )
        else:
            magnitude_law = StraightMagnitudeLaw.from_payload(
                magnitude_payload,
            )
        radius_law = ConditionalRadiusLaw.from_payload(
            calibration["radius_law"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} galaxy calibration is malformed") from exc
    return magnitude_law, radius_law


def _aggregate_radius_grid(
    radius_aggregate: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return Q1 joint PHZ weight on its native magnitude-radius grid."""
    try:
        magnitude_edges = np.asarray(
            radius_aggregate["magnitude_edges"], dtype=np.float64,
        )
        radius_edges = np.asarray(
            radius_aggregate["radius_edges_arcsec"], dtype=np.float64,
        )
        area_arcmin2 = float(radius_aggregate["footprint_area_arcmin2"])
        joint_bins = radius_aggregate["joint_bins"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Q1 radius aggregate is malformed") from exc
    if (
        magnitude_edges.ndim != 1
        or radius_edges.ndim != 1
        or magnitude_edges.size < 3
        or radius_edges.size < 3
        or not np.all(np.isfinite(magnitude_edges))
        or not np.all(np.isfinite(radius_edges) & (radius_edges > 0.0))
        or not np.all(np.diff(magnitude_edges) > 0.0)
        or not np.all(np.diff(radius_edges) > 0.0)
        or not np.isfinite(area_arcmin2)
        or area_arcmin2 <= 0.0
        or not isinstance(joint_bins, list)
    ):
        raise ValueError("Q1 radius aggregate is malformed")
    weight = np.zeros(
        (magnitude_edges.size - 1, radius_edges.size - 1),
        dtype=np.float64,
    )
    try:
        for item in joint_bins:
            magnitude_index = int(item["magnitude_bin"])
            radius_index = int(item["radius_bin"])
            value = float(item["expected_radii"])
            if (
                not 0 <= magnitude_index < weight.shape[0]
                or not 0 <= radius_index < weight.shape[1]
                or not np.isfinite(value)
                or value < 0.0
            ):
                raise ValueError
            weight[magnitude_index, radius_index] += value
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("Q1 radius aggregate has malformed joint bins") from exc
    if not np.any(weight > 0.0):
        raise ValueError("Q1 radius aggregate contains no positive PHZ weight")
    return magnitude_edges, np.log10(radius_edges), weight, area_arcmin2


def _mass_contour_levels(
    density_per_unit: np.ndarray, cell_density: np.ndarray,
) -> np.ndarray:
    """Return thresholds enclosing 99.5, 99, 95, 80, 50, 25, and 10 percent."""
    density = np.asarray(density_per_unit, dtype=np.float64).ravel()
    mass = np.asarray(cell_density, dtype=np.float64).ravel()
    keep = np.isfinite(density) & np.isfinite(mass) & (density > 0.0) & (mass > 0.0)
    density, mass = density[keep], mass[keep]
    if density.size < 2 or float(np.sum(mass)) <= 0.0:
        raise ValueError("joint galaxy model has no positive density")
    order = np.argsort(density)[::-1]
    cumulative = np.cumsum(mass[order]) / np.sum(mass)
    thresholds = [
        density[order[min(np.searchsorted(cumulative, fraction), len(order) - 1)]]
        for fraction in (0.995, 0.99, 0.95, 0.80, 0.50, 0.25, 0.10)
    ]
    levels = np.unique(np.asarray(sorted(thresholds), dtype=np.float64))
    if levels.size < 2:
        low = float(np.min(density))
        high = float(np.max(density))
        if np.isclose(low, high):
            high = low * 1.001
        levels = np.linspace(low, high, 3, dtype=np.float64)
    return levels


def _bright_magnitude_slice(
    magnitude_edges: np.ndarray,
    observed_weight: np.ndarray,
    *,
    maximum_magnitude: float = 22.0,
) -> slice:
    """Return a contiguous, observed bright window with at least two bins."""
    magnitude_center = 0.5 * (magnitude_edges[:-1] + magnitude_edges[1:])
    observed_by_magnitude = np.sum(observed_weight, axis=1)
    positive = np.flatnonzero(observed_by_magnitude > 0.0)
    start = int(positive[0]) if positive.size else 0
    stop = int(np.searchsorted(magnitude_center, maximum_magnitude, side="left"))
    stop = min(len(magnitude_center), max(stop, start + 2))
    if stop - start < 2:
        start = max(0, stop - 2)
    return slice(start, stop)


def _physical_radius_ticks(
    ax, log_radius_edges: np.ndarray, *, axis: str = "y",
) -> None:
    """Label a log-radius coordinate with radii in arcseconds."""
    values = np.asarray((0.03, 0.1, 0.3, 1.0, 3.0, 10.0), dtype=np.float64)
    log_values = np.log10(values)
    keep = (
        (log_values >= float(log_radius_edges[0]) - 1e-9)
        & (log_values <= float(log_radius_edges[-1]) + 1e-9)
    )
    ticks = log_values[keep]
    labels = [f"{value:g}" for value in values[keep]]
    if axis == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    elif axis == "y":
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    else:
        raise ValueError("radius tick axis must be x or y")


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
    generation_law = brightness.get("generation_law") or {}
    plots = calibration.get("plots") or {}
    radius = plots.get("radius") or {}
    relation = plots.get("conditional_radius") or {}
    if not brightness_law or not generation_law or not radius or not relation:
        raise ValueError("Euclid joint fit has no publication diagnostics")
    magnitude_kind = str(
        (calibration.get("magnitude_law") or {}).get("kind") or ""
    )
    circularized_radius = "circularized" in str(calibration.get("kind") or "")

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
        fig, axes = plt.subplots(1, 3, figsize=(16.8, 6.7))
        fig.subplots_adjust(
            left=0.062, right=0.992, bottom=0.30, top=0.93, wspace=0.30,
        )

        observed_brightness = brightness.get("observed") or {}
        _line(
            axes[0], brightness_law, "density", color=TNG,
            label=(
                "Q1 2FWHM fitted main law"
                if magnitude_kind
                == (
                    "continuous_three_slope_bright_bridge_main_flat_"
                    "faint_counts"
                )
                else "Q1 2FWHM fitted middle law"
            ),
            width=2.7,
        )
        _line(
            axes[0], generation_law, "density", color="#168f65",
            label=(
                "generation law: continuous bright bridge + main + flat"
                if magnitude_kind
                == (
                    "continuous_three_slope_bright_bridge_main_flat_"
                    "faint_counts"
                )
                else (
                    "generation law: empirical + fitted + flat"
                    if magnitude_kind
                    == "empirical_bright_straight_middle_flat_faint_counts"
                    else "generation law: straight then flat"
                )
            ),
            width=3.0,
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
        break_magnitude = brightness.get("break_magnitude")
        if break_magnitude is not None:
            axes[0].axvline(
                float(break_magnitude), color="#168f65",
                linewidth=1.3, linestyle=(0, (2, 3)), alpha=0.75,
                label=r"break; faint tail = 100 arcmin$^{-2}$ mag$^{-1}$",
            )
        axes[0].set_xlim(14, 29)
        _finish_axis(
            axes[0], "VIS 2FWHM count distribution",
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
                label=(
                    "Euclid PHZ/MER cleaned circularized Sérsic $R_e$"
                    if circularized_radius
                    else "Euclid PHZ/MER measured Sérsic $R_e$"
                ),
                zorder=4,
            )
        q1_weighted_x, q1_weighted_density = _xy(
            radius, "q1_weighted_density",
        )
        if q1_weighted_x.size:
            axes[1].plot(
                q1_weighted_x, q1_weighted_density,
                color=TNG, linewidth=2.7,
                label="conditional fit × Q1 magnitude counts", zorder=3,
            )
            generation_radius_x, generation_radius_density = _xy(
                radius, "density",
            )
            axes[1].plot(
                generation_radius_x, generation_radius_density,
                color="#168f65", linewidth=2.2, linestyle=(0, (4, 3)),
                label="generation marginal (VIS 14–29)", zorder=2,
            )
        else:
            _line(
                axes[1], radius, "density", color=TNG,
                label="joint-fit $R_e$ marginal", width=2.7,
            )
        _finish_axis(
            axes[1], "Euclid half-light radius",
            (
                r"log$_{10}(R_{e,\mathrm{circ}}/\mathrm{arcsec})$"
                if circularized_radius else
                r"log$_{10}(R_{e,\mathrm{VIS\ S\acute{e}rsic}}/\mathrm{arcsec})$"
            ),
            r"objects arcmin$^{-2}$ dex$^{-1}$",
            logarithmic_y=True,
        )

        relation_payload = {
            "x": relation.get("magnitude") or [],
            "observed": relation.get("observed_mean_log10_arcsec") or [],
            "model": relation.get("model_mean_log10_arcsec") or [],
            "low": (
                relation.get("model_core_low_log10_arcsec")
                or relation.get("model_low_log10_arcsec") or []
            ),
            "high": (
                relation.get("model_core_high_log10_arcsec")
                or relation.get("model_high_log10_arcsec") or []
            ),
        }
        relation_x, relation_low = _xy(relation_payload, "low")
        _, relation_high = _xy(relation_payload, "high")
        if relation_x.size and relation_x.size == relation_high.size:
            axes[2].fill_between(
                relation_x, relation_low, relation_high,
                color=TNG, alpha=0.14, linewidth=0,
                label=r"Gaussian core 1$\sigma$ scatter",
            )
        conditional_fit_interval = relation.get("fit_interval") or []
        if len(conditional_fit_interval) == 2:
            axes[2].axvspan(
                float(conditional_fit_interval[0]),
                float(conditional_fit_interval[1]),
                color=EUCLID_OBS, alpha=0.05, linewidth=0,
            )
        _observed(
            axes[2], relation_payload, color=EUCLID_OBS,
            label="Euclid binned mean",
        )
        _line(
            axes[2], relation_payload, "model", color=TNG,
            label=(
                "straight conditional mean"
                if relation.get("model_kind")
                == "straight_truncated_gaussian_no_tail"
                else "broken conditional mean"
            ),
            width=2.7,
        )
        axes[2].set_xlim(14, 29)
        if break_magnitude is not None:
            axes[2].axvline(
                float(break_magnitude), color="#168f65",
                linewidth=2.0, linestyle=(0, (2, 3)),
                label="generation-law break",
            )
        radius_break_magnitude = relation.get("break_magnitude")
        if radius_break_magnitude is not None:
            axes[2].axvline(
                float(radius_break_magnitude), color=TNG,
                linewidth=1.4, linestyle=(0, (1, 2)),
                label="radius-law break",
            )
        _finish_axis(
            axes[2], "Joint brightness-size relation",
            "VIS 2FWHM aperture magnitude [AB]",
            (
                r"mean log$_{10}(R_{e,\mathrm{circ}}/\mathrm{arcsec})$"
                if circularized_radius else
                r"mean log$_{10}(R_e/\mathrm{arcsec})$"
            ),
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
            bbox_to_anchor=(0.5, 0.02), columnspacing=2.0,
        )

        buffer = io.BytesIO()
        fig.savefig(buffer, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return buffer.getvalue()


def render_population_fit_comparison(
    previous: Mapping[str, Any],
    candidate: Mapping[str, Any],
    radius_aggregate: Mapping[str, Any],
    *,
    output_format: str = "png",
    dpi: int = 300,
) -> bytes:
    """Render an old-versus-new galaxy-prior comparison without refitting.

    The caller supplies both calibration artifacts and the aggregate Q1
    magnitude-radius brackets explicitly.  The renderer performs no file I/O,
    archive query, activation, or field generation.
    """
    fmt = output_format.lower()
    if fmt not in {"png", "pdf", "svg"}:
        raise ValueError("output_format must be png, pdf, or svg")
    dpi = max(120, min(int(dpi), 600))
    previous_magnitude, previous_radius = _comparison_laws(
        previous, label="previous",
    )
    candidate_magnitude, candidate_radius = _comparison_laws(
        candidate, label="candidate",
    )
    magnitude_edges, log_radius_edges, observed_weight, area_arcmin2 = (
        _aggregate_radius_grid(radius_aggregate)
    )
    for label, law in (
        ("previous", previous_radius), ("candidate", candidate_radius),
    ):
        if not (
            np.isclose(log_radius_edges[0], law.log_radius_min)
            and np.isclose(log_radius_edges[-1], law.log_radius_max)
        ):
            raise ValueError(
                f"{label} radius support differs from the Q1 aggregate",
            )
    for label, law in (
        ("previous", previous_magnitude),
        ("candidate", candidate_magnitude),
    ):
        if (
            magnitude_edges[0] < law.mag_bright - 1e-9
            or magnitude_edges[-1] > law.mag_faint + 1e-9
        ):
            raise ValueError(
                f"{label} magnitude support does not cover the Q1 aggregate",
            )

    magnitude_width = np.diff(magnitude_edges)
    log_radius_width = np.diff(log_radius_edges)
    magnitude_center = 0.5 * (magnitude_edges[:-1] + magnitude_edges[1:])
    log_radius_center = 0.5 * (
        log_radius_edges[:-1] + log_radius_edges[1:]
    )
    observed_density = (
        observed_weight
        / area_arcmin2
        / magnitude_width[:, None]
        / log_radius_width[None, :]
    )
    observed_radius_density = (
        np.sum(observed_weight, axis=0)
        / area_arcmin2
        / log_radius_width
    )
    observed_by_magnitude = np.sum(observed_weight, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        observed_conditional_mean = (
            np.sum(observed_weight * log_radius_center[None, :], axis=1)
            / observed_by_magnitude
        )
    observed_conditional_mean[observed_by_magnitude <= 0.0] = np.nan

    previous_joint = joint_density_grid(
        previous_magnitude,
        previous_radius,
        magnitude_edges=magnitude_edges,
        log_radius_edges=log_radius_edges,
    )
    candidate_joint = joint_density_grid(
        candidate_magnitude,
        candidate_radius,
        magnitude_edges=magnitude_edges,
        log_radius_edges=log_radius_edges,
    )
    previous_joint_density = (
        previous_joint["density"]
        / magnitude_width[:, None]
        / log_radius_width[None, :]
    )
    candidate_joint_density = (
        candidate_joint["density"]
        / magnitude_width[:, None]
        / log_radius_width[None, :]
    )
    previous_marginal = joint_density_grid(
        previous_magnitude,
        previous_radius,
        log_radius_edges=log_radius_edges,
    )
    candidate_marginal = joint_density_grid(
        candidate_magnitude,
        candidate_radius,
        log_radius_edges=log_radius_edges,
    )
    previous_radius_density = (
        np.sum(previous_marginal["density"], axis=0) / log_radius_width
    )
    candidate_radius_density = (
        np.sum(candidate_marginal["density"], axis=0) / log_radius_width
    )
    candidate_joint_mass = np.asarray(
        candidate_joint["density"], dtype=np.float64,
    )
    candidate_mass_by_magnitude = np.sum(candidate_joint_mass, axis=1)
    candidate_conditional_radius_mass = np.divide(
        candidate_joint_mass,
        candidate_mass_by_magnitude[:, None],
        out=np.zeros_like(candidate_joint_mass),
        where=candidate_mass_by_magnitude[:, None] > 0.0,
    )
    candidate_q1_weighted_radius_density = (
        np.sum(
            candidate_conditional_radius_mass
            * observed_by_magnitude[:, None],
            axis=0,
        )
        / log_radius_width
    )
    observed_radius_shape = observed_radius_density / np.sum(
        observed_radius_density * log_radius_width,
    )
    previous_radius_shape = previous_radius_density / np.sum(
        previous_radius_density * log_radius_width,
    )
    candidate_radius_shape = candidate_radius_density / np.sum(
        candidate_radius_density * log_radius_width,
    )
    candidate_q1_weighted_radius_shape = (
        candidate_q1_weighted_radius_density
        / np.sum(candidate_q1_weighted_radius_density * log_radius_width)
    )
    previous_radius_definition = (
        "circularized"
        if "circularized" in str(previous.get("kind") or "").lower()
        else "major-axis"
    )
    candidate_radius_definition = (
        "circularized"
        if "circularized" in str(candidate.get("kind") or "").lower()
        else "major-axis"
    )

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

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
        fig, axes = plt.subplots(2, 2, figsize=(15.8, 11.6))
        fig.subplots_adjust(
            left=0.075, right=0.965, bottom=0.075, top=0.875,
            wspace=0.25, hspace=0.34,
        )

        ax_magnitude = axes[0, 0]
        observed_magnitude = (
            (candidate.get("magnitude_plot") or {}).get("observed") or {}
        )
        observed_x, observed_y = _xy(observed_magnitude, "density")
        observed_keep = observed_y > 0.0
        if np.any(observed_keep):
            ax_magnitude.plot(
                observed_x[observed_keep], observed_y[observed_keep],
                linestyle="none", marker="o", markersize=4.4,
                markerfacecolor=PAPER, markeredgecolor=EUCLID_OBS,
                markeredgewidth=1.15, label="Q1 VIS 2FWHM counts", zorder=4,
            )
        previous_magnitude_x = np.linspace(
            previous_magnitude.mag_bright,
            previous_magnitude.mag_faint,
            301,
            dtype=np.float64,
        )
        candidate_magnitude_x = np.linspace(
            candidate_magnitude.mag_bright,
            candidate_magnitude.mag_faint,
            301,
            dtype=np.float64,
        )
        ax_magnitude.plot(
            previous_magnitude_x,
            previous_magnitude.density(previous_magnitude_x),
            color=PREVIOUS_MODEL, linewidth=2.3, linestyle=(0, (6, 3)),
            label="Previous generation law",
        )
        ax_magnitude.plot(
            candidate_magnitude_x,
            candidate_magnitude.density(candidate_magnitude_x),
            color=CANDIDATE_MODEL, linewidth=2.8,
            label="Candidate generation law",
        )
        ax_magnitude.set_xlim(
            min(previous_magnitude.mag_bright, candidate_magnitude.mag_bright),
            max(previous_magnitude.mag_faint, candidate_magnitude.mag_faint),
        )
        _finish_axis(
            ax_magnitude, "VIS 2FWHM magnitude density",
            "VIS 2FWHM aperture magnitude [AB]",
            "objects arcmin$^{-2}$ mag$^{-1}$",
            logarithmic_y=True,
        )
        ax_magnitude.legend(frameon=False, fontsize=LEGEND_SIZE)

        ax_radius = axes[0, 1]
        radius_keep = observed_radius_density > 0.0
        ax_radius.plot(
            log_radius_center[radius_keep],
            observed_radius_shape[radius_keep],
            linestyle="none", marker="o", markersize=4.5,
            markerfacecolor=PAPER, markeredgecolor=EUCLID_OBS,
            markeredgewidth=1.15,
            label="Q1 clean circularized shape · normalized",
            zorder=4,
        )
        ax_radius.plot(
            log_radius_center, previous_radius_shape,
            color=PREVIOUS_MODEL, linewidth=2.3, linestyle=(0, (6, 3)),
            label=(
                f"Previous {previous_radius_definition} shape · "
                "full generation"
            ),
        )
        ax_radius.plot(
            log_radius_center, candidate_q1_weighted_radius_shape,
            color=CANDIDATE_MODEL, linewidth=2.8,
            label=(
                f"Candidate {candidate_radius_definition} shape · "
                "Q1-magnitude weighted"
            ),
        )
        ax_radius.plot(
            log_radius_center, candidate_radius_shape,
            color=CANDIDATE_MODEL, linewidth=2.0, linestyle=(0, (2, 2)),
            label=(
                f"Candidate {candidate_radius_definition} shape · "
                "full generation"
            ),
        )
        ax_radius.set_xlim(log_radius_edges[0], log_radius_edges[-1])
        _finish_axis(
            ax_radius, "Radius shape · Q1-weighted versus full generation",
            "$R_e$ [arcsec; logarithmic axis]",
            "probability density [dex$^{-1}$]",
            logarithmic_y=True,
        )
        _physical_radius_ticks(ax_radius, log_radius_edges, axis="x")
        ax_radius.legend(frameon=False, fontsize=LEGEND_SIZE)

        ax_conditional = axes[1, 0]
        conditional_keep = np.isfinite(observed_conditional_mean)
        ax_conditional.plot(
            magnitude_center[conditional_keep],
            observed_conditional_mean[conditional_keep],
            linestyle="none", marker="o", markersize=4.4,
            markerfacecolor=PAPER, markeredgecolor=EUCLID_OBS,
            markeredgewidth=1.15,
            label="Q1 circularized binned mean", zorder=4,
        )
        conditional_x = np.linspace(
            magnitude_edges[0], magnitude_edges[-1], 301, dtype=np.float64,
        )
        previous_core = previous_radius.core_mean(conditional_x)
        candidate_core = candidate_radius.core_mean(conditional_x)
        ax_conditional.fill_between(
            conditional_x,
            previous_core - previous_radius.scatter_dex,
            previous_core + previous_radius.scatter_dex,
            color=PREVIOUS_MODEL, alpha=0.08, linewidth=0,
        )
        ax_conditional.fill_between(
            conditional_x,
            candidate_core - candidate_radius.scatter_dex,
            candidate_core + candidate_radius.scatter_dex,
            color=CANDIDATE_MODEL, alpha=0.12, linewidth=0,
        )
        ax_conditional.plot(
            conditional_x, previous_radius.mean(conditional_x),
            color=PREVIOUS_MODEL, linewidth=2.3, linestyle=(0, (6, 3)),
            label=f"Previous {previous_radius_definition} mean",
        )
        ax_conditional.plot(
            conditional_x, candidate_radius.mean(conditional_x),
            color=CANDIDATE_MODEL, linewidth=2.8,
            label=f"Candidate {candidate_radius_definition} mean",
        )
        ax_conditional.set_xlim(magnitude_edges[0], magnitude_edges[-1])
        ax_conditional.set_ylim(log_radius_edges[0], log_radius_edges[-1])
        _finish_axis(
            ax_conditional, "Conditional Sérsic size",
            "VIS 2FWHM aperture magnitude [AB]",
            r"$R_e$ [arcsec; logarithmic coordinate]",
            zero_floor=False,
        )
        _physical_radius_ticks(ax_conditional, log_radius_edges)
        ax_conditional.legend(frameon=False, fontsize=LEGEND_SIZE)

        ax_joint = axes[1, 1]
        observed_positive = observed_density[
            np.isfinite(observed_density) & (observed_density > 0.0)
        ]
        lower = float(np.percentile(observed_positive, 5.0))
        upper = float(np.percentile(observed_positive, 99.5))
        if np.isclose(lower, upper):
            lower = max(upper / 10.0, np.finfo(np.float64).tiny)
        image = ax_joint.pcolormesh(
            magnitude_edges,
            log_radius_edges,
            np.ma.masked_less_equal(observed_density.T, 0.0),
            shading="auto", cmap="Greys", norm=LogNorm(vmin=lower, vmax=upper),
            rasterized=True,
        )
        ax_joint.contour(
            magnitude_center,
            log_radius_center,
            previous_joint_density.T,
            levels=_mass_contour_levels(
                previous_joint_density, previous_joint["density"],
            ),
            colors=PREVIOUS_MODEL, linewidths=1.8, linestyles="dashed",
        )
        ax_joint.contour(
            magnitude_center,
            log_radius_center,
            candidate_joint_density.T,
            levels=_mass_contour_levels(
                candidate_joint_density, candidate_joint["density"],
            ),
            colors=CANDIDATE_MODEL, linewidths=2.0, linestyles="solid",
        )
        bright_slice = _bright_magnitude_slice(
            magnitude_edges, observed_weight,
        )
        bright_edges = magnitude_edges[
            bright_slice.start:bright_slice.stop + 1
        ]
        bright_center = magnitude_center[bright_slice]
        bright_observed_density = observed_density[bright_slice]
        bright_positive = bright_observed_density[
            np.isfinite(bright_observed_density)
            & (bright_observed_density > 0.0)
        ]
        bright_lower = float(np.percentile(bright_positive, 5.0))
        bright_upper = float(np.percentile(bright_positive, 99.5))
        if np.isclose(bright_lower, bright_upper):
            bright_lower = max(
                bright_upper / 10.0, np.finfo(np.float64).tiny,
            )
        ax_bright = ax_joint.inset_axes([0.055, 0.075, 0.47, 0.40])
        ax_bright.pcolormesh(
            bright_edges,
            log_radius_edges,
            np.ma.masked_less_equal(bright_observed_density.T, 0.0),
            shading="auto", cmap="Greys",
            norm=LogNorm(vmin=bright_lower, vmax=bright_upper),
            rasterized=True,
        )
        ax_bright.contour(
            bright_center,
            log_radius_center,
            previous_joint_density[bright_slice].T,
            levels=_mass_contour_levels(
                previous_joint_density[bright_slice],
                previous_joint["density"][bright_slice],
            ),
            colors=PREVIOUS_MODEL, linewidths=1.25, linestyles="dashed",
        )
        ax_bright.contour(
            bright_center,
            log_radius_center,
            candidate_joint_density[bright_slice].T,
            levels=_mass_contour_levels(
                candidate_joint_density[bright_slice],
                candidate_joint["density"][bright_slice],
            ),
            colors=CANDIDATE_MODEL, linewidths=1.4, linestyles="solid",
        )
        ax_bright.set_xlim(bright_edges[0], bright_edges[-1])
        ax_bright.set_ylim(log_radius_edges[0], log_radius_edges[-1])
        ax_bright.set_title(
            f"Bright Q1 · {bright_edges[0]:g}≤VIS<{bright_edges[-1]:g}"
            " · locally scaled\n"
            "10/25/50/80/95/99/99.5% mass contours normalized within window",
            loc="left", fontsize=max(NOTE_SIZE * 0.48, 6.5), pad=2,
            fontweight=600,
            bbox={"facecolor": PAPER, "edgecolor": "none", "alpha": 0.88},
        )
        ax_bright.grid(True, color=GRID, linewidth=0.4, alpha=0.65)
        ax_bright.set_axisbelow(True)
        ax_bright.tick_params(
            colors=INK, labelsize=max(NOTE_SIZE * 0.42, 6),
            width=0.8, length=3,
        )
        _physical_radius_ticks(ax_bright, log_radius_edges)
        ax_joint.plot(
            [], [], color=PREVIOUS_MODEL, linewidth=1.8,
            linestyle=(0, (6, 3)),
            label=f"Previous {previous_radius_definition} model",
        )
        ax_joint.plot(
            [], [], color=CANDIDATE_MODEL, linewidth=2.0,
            label=f"Candidate {candidate_radius_definition} model",
        )
        ax_joint.set_xlim(magnitude_edges[0], magnitude_edges[-1])
        ax_joint.set_ylim(log_radius_edges[0], log_radius_edges[-1])
        _finish_axis(
            ax_joint, "Q1 circularized density · global model mass contours",
            "VIS 2FWHM aperture magnitude [AB]",
            r"$R_e$ [arcsec; logarithmic coordinate]",
            zero_floor=False,
        )
        _physical_radius_ticks(ax_joint, log_radius_edges)
        ax_joint.legend(
            frameon=True, framealpha=0.9, facecolor=PAPER, edgecolor="none",
            fontsize=NOTE_SIZE, loc="upper left",
        )
        colorbar = fig.colorbar(image, ax=ax_joint, pad=0.018, fraction=0.052)
        colorbar.set_label(
            "Q1 objects arcmin$^{-2}$ mag$^{-1}$ dex$^{-1}$",
            fontsize=AXIS_LABEL_SIZE,
        )
        colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)

        previous_fingerprint = str(previous.get("fingerprint") or "unknown")[:12]
        candidate_fingerprint = str(candidate.get("fingerprint") or "unknown")[:12]
        fig.suptitle(
            "Galaxy population fit · previous versus candidate",
            x=0.075, y=0.975, ha="left",
            fontsize=FIGURE_TITLE_SIZE, fontweight=700, color=INK,
        )
        fig.text(
            0.075, 0.943,
            f"previous {previous_fingerprint}  →  candidate "
            f"{candidate_fingerprint}  ·  no field regeneration",
            ha="left", va="center", fontsize=NOTE_SIZE, color=MUTED,
        )
        fig.text(
            0.075, 0.915,
            f"purple model = previous {previous_radius_definition} $R_e$  ·  "
            f"green candidate = {candidate_radius_definition} $R_e$  ·  "
            "blue Q1 = clean circularized morphology subset",
            ha="left", va="center", fontsize=NOTE_SIZE, color=MUTED,
        )

        buffer = io.BytesIO()
        fig.savefig(buffer, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return buffer.getvalue()


def render_galaxy_distribution_plate(
    payload: Mapping[str, Any], *, output_format: str = "png", dpi: int = 300,
) -> bytes:
    """Render the reviewed galaxy-distribution page as a fixed 2 × 2 plate.

    The figure consumes only the compact WebUI artifact: Q1 aggregates, actual
    generated-source measurements, and the active analytical law.  It never
    queries the archive, re-fits the population, or regenerates fields.
    """
    fmt = output_format.lower()
    if fmt not in {"png", "pdf", "svg"}:
        raise ValueError("output_format must be png, pdf, or svg")
    dpi = max(120, min(int(dpi), 600))
    try:
        parameters = payload["parameters"]
        brightness = parameters["magnitude"]["photometry_series"]
        radius = parameters["radius"]["radius_series"]
        joint = payload["joint_maps"]
        joint_maps = {
            str(item["key"]): item for item in joint["maps"]
        }
        magnitude_edges = np.asarray(
            joint["magnitude_edges"], dtype=np.float64,
        )
        log_radius_edges = np.asarray(
            joint["log_radius_edges"], dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("galaxy-distribution artifact is malformed") from exc
    if not joint.get("available") or "q1" not in joint_maps:
        raise ValueError("galaxy-distribution joint maps are unavailable")

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

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
        training_included = bool(payload.get("training_included"))
        generated_f2_label = (
            "test + validation VIS 2FWHM"
            if training_included else "current generated VIS 2FWHM"
        )
        requested_radius_label = (
            "all-catalogue requested Sérsic $R_e$"
            if training_included else "generated requested Sérsic $R_e$"
        )
        clean_radius_label = (
            "test + validation clean-image half light"
            if training_included else "generated clean-image half light"
        )
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 10.1))
        fig.subplots_adjust(
            left=0.085, right=0.955, bottom=0.075, top=0.965,
            wspace=0.29, hspace=0.34,
        )

        ax_brightness = axes[0, 0]
        brightness_specs = (
            ("q1_vis_f2", EUCLID_OBS, "Q1 VIS 2FWHM", "o", "none"),
            (
                "synthetic_vis_2fwhm", "#d39b32",
                generated_f2_label, "o", "full",
            ),
            (
                "generator_vis_f2", CANDIDATE_MODEL,
                "active generation law", None, None,
            ),
        )
        for key, color, label, marker, marker_fill in brightness_specs:
            curve = brightness.get(key) or {}
            x, y = _xy(curve, "density")
            keep = y > 0.0
            if not np.any(keep):
                continue
            if marker:
                ax_brightness.plot(
                    x[keep], y[keep], linestyle="-", linewidth=1.35,
                    marker=marker, markersize=3.8,
                    markerfacecolor=(color if marker_fill == "full" else PAPER),
                    markeredgecolor=color, markeredgewidth=1.0,
                    color=color, label=label,
                )
            else:
                ax_brightness.plot(
                    x[keep], y[keep], color=color, linewidth=2.6,
                    label=label,
                )
        _finish_axis(
            ax_brightness, "A · VIS 2FWHM magnitude density",
            "VIS 2FWHM aperture magnitude [AB]",
            "objects arcmin$^{-2}$ mag$^{-1}$", logarithmic_y=True,
        )
        ax_brightness.legend(frameon=False, fontsize=NOTE_SIZE)

        ax_radius = axes[0, 1]
        radius_specs = (
            (
                "euclid_sersic_re", EUCLID_OBS,
                "Q1 circularized Sérsic $R_e$", "o", "none", "none",
            ),
            (
                "synthetic_requested_re", "#d39b32",
                requested_radius_label, "o", "full", "none",
            ),
            (
                "synthetic_clean_half_light", "#f0b45b",
                clean_radius_label, "s", "full", "none",
            ),
            (
                "fit_re", CANDIDATE_MODEL,
                "active $R_e$ marginal", None, None, "solid",
            ),
        )
        for key, color, label, marker, marker_fill, linestyle in radius_specs:
            curve = radius.get(key) or {}
            x, y = _xy(curve, "density")
            keep = y > 0.0
            if not np.any(keep):
                continue
            ax_radius.plot(
                x[keep], y[keep], color=color,
                linewidth=2.3 if marker is None else 1.25,
                linestyle=linestyle,
                marker=marker, markersize=3.7,
                markerfacecolor=(
                    color if marker_fill == "full" else PAPER
                ),
                markeredgecolor=color, markeredgewidth=0.95,
                label=label,
            )
        ax_radius.set_xlim(log_radius_edges[0], log_radius_edges[-1])
        _finish_axis(
            ax_radius, "B · Half-light-radius surface density",
            "$R_e$ [arcsec; logarithmic axis]",
            "objects arcmin$^{-2}$ dex$^{-1}$", logarithmic_y=True,
        )
        _physical_radius_ticks(ax_radius, log_radius_edges, axis="x")
        ax_radius.legend(frameon=False, fontsize=NOTE_SIZE)

        ax_shape = axes[1, 0]
        shape_specs = (
            (
                "euclid_sersic_re_shape", EUCLID_OBS,
                "Q1 clean circularized shape", "o", "none", "none",
            ),
            (
                "fit_re_q1_weighted_shape", CANDIDATE_MODEL,
                "model · Q1-magnitude weighted", None, None, "solid",
            ),
            (
                "fit_re_full_generation_shape", CANDIDATE_MODEL,
                "model · full generation", None, None, "--",
            ),
        )
        for key, color, label, marker, marker_fill, linestyle in shape_specs:
            curve = radius.get(key) or {}
            x, y = _xy(curve, "density")
            keep = y > 0.0
            if not np.any(keep):
                continue
            ax_shape.plot(
                x[keep], y[keep], color=color,
                linewidth=2.35 if marker is None else 1.25,
                linestyle=linestyle, marker=marker, markersize=3.8,
                markerfacecolor=(
                    color if marker_fill == "full" else PAPER
                ),
                markeredgecolor=color, markeredgewidth=0.95,
                label=label,
            )
        ax_shape.set_xlim(log_radius_edges[0], log_radius_edges[-1])
        _finish_axis(
            ax_shape, "C · Normalized half-light shape",
            "$R_e$ [arcsec; logarithmic axis]",
            "probability density [dex$^{-1}$]", logarithmic_y=True,
        )
        _physical_radius_ticks(ax_shape, log_radius_edges, axis="x")
        ax_shape.legend(frameon=False, fontsize=NOTE_SIZE)

        ax_joint = axes[1, 1]
        q1_density = np.asarray(
            joint_maps["q1"]["density"], dtype=np.float64,
        )
        q1_positive = q1_density[
            np.isfinite(q1_density) & (q1_density > 0.0)
        ]
        if q1_positive.size < 2:
            raise ValueError("Q1 joint magnitude-radius map is empty")
        lower = max(float(np.percentile(q1_positive, 5.0)), 1e-6)
        upper = float(np.max(q1_positive))
        image = ax_joint.pcolormesh(
            magnitude_edges, log_radius_edges,
            np.ma.masked_less_equal(q1_density.T, 0.0),
            shading="auto", cmap="Greys", norm=LogNorm(lower, upper),
            rasterized=True,
        )
        magnitude_center = 0.5 * (
            magnitude_edges[:-1] + magnitude_edges[1:]
        )
        log_radius_center = 0.5 * (
            log_radius_edges[:-1] + log_radius_edges[1:]
        )
        magnitude_width = np.diff(magnitude_edges)
        log_radius_width = np.diff(log_radius_edges)
        for key, linestyle in (("synthetic", "dashed"), ("model", "solid")):
            item = joint_maps.get(key)
            if not item:
                continue
            serialized_contours = item.get("contours") or []
            if serialized_contours:
                for contour_index, contour in enumerate(
                    serialized_contours,
                ):
                    for path in contour.get("paths") or []:
                        ax_joint.plot(
                            path["x"], path["y"],
                            color=str(item["color"]),
                            linewidth=1.15 + 0.20 * contour_index,
                            linestyle=linestyle,
                        )
            else:
                density = np.asarray(item["density"], dtype=np.float64)
                cell_mass = (
                    density * magnitude_width[:, None]
                    * log_radius_width[None, :]
                )
                try:
                    levels = _mass_contour_levels(density, cell_mass)
                    ax_joint.contour(
                        magnitude_center, log_radius_center, density.T,
                        levels=levels, colors=str(item["color"]),
                        linewidths=1.8 if key != "model" else 2.1,
                        linestyles=linestyle,
                    )
                except ValueError:
                    continue
            ax_joint.plot(
                [], [], color=str(item["color"]), linewidth=2.0,
                linestyle=linestyle,
                label=(
                    {
                        "synthetic": str(
                            item.get("label") or "current generated"
                        ),
                        "model": "active model",
                    }[key]
                    + " · 10/25/50/80/95/99/99.5%"
                ),
            )
        ax_joint.set_xlim(magnitude_edges[0], magnitude_edges[-1])
        ax_joint.set_ylim(log_radius_edges[0], log_radius_edges[-1])
        _finish_axis(
            ax_joint, "D · Q1 density + generated/model contours",
            "VIS 2FWHM aperture magnitude [AB]",
            "$R_e$ [arcsec; logarithmic coordinate]", zero_floor=False,
        )
        _physical_radius_ticks(ax_joint, log_radius_edges)
        ax_joint.legend(
            frameon=True, framealpha=0.88, facecolor=PAPER,
            edgecolor="none", fontsize=8.5, loc="upper left",
        )
        colorbar = fig.colorbar(image, ax=ax_joint, pad=0.018, fraction=0.052)
        colorbar.set_label(
            "Q1 objects arcmin$^{-2}$ mag$^{-1}$ dex$^{-1}$",
            fontsize=AXIS_LABEL_SIZE,
        )
        colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)

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
