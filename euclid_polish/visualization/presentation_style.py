"""Shared typography for figures intended for talks and publication export."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib as mpl

FIGURE_TITLE_SIZE = 20
PANEL_TITLE_SIZE = 17
AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 12.5
LEGEND_SIZE = 11.5
NOTE_SIZE = 11.5

PRESENTATION_RC: dict[str, Any] = {
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": PANEL_TITLE_SIZE,
    "axes.titleweight": 600,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "figure.titlesize": FIGURE_TITLE_SIZE,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.2,
}


def presentation_rc(extra: Mapping[str, Any] | None = None):
    """Return a Matplotlib context with slide-readable typography."""
    style = dict(PRESENTATION_RC)
    if extra:
        style.update(extra)
    return mpl.rc_context(style)


def apply_presentation_figure(fig) -> None:
    """Enforce readable sizes after helpers with explicit small fonts run."""
    if fig._suptitle is not None:  # noqa: SLF001 - Matplotlib's public handle
        fig._suptitle.set_fontsize(FIGURE_TITLE_SIZE)  # noqa: SLF001
        fig._suptitle.set_fontweight(700)  # noqa: SLF001
    for ax in fig.axes:
        ax.title.set_fontsize(PANEL_TITLE_SIZE)
        ax.title.set_fontweight(600)
        ax.xaxis.label.set_fontsize(AXIS_LABEL_SIZE)
        ax.yaxis.label.set_fontsize(AXIS_LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, width=1.0, length=4.5)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(LEGEND_SIZE)
