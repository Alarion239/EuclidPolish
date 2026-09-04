"""Shared mechanics for compositing prepared image stamps onto generation canvases."""

from __future__ import annotations

import numpy as np


def composite_stamp(
    canvas: np.ndarray,
    stamp: np.ndarray,
    x0: float,
    y0: float,
) -> bool:
    """Add the exact stamp/canvas intersection and report whether it exists."""
    height, width = canvas.shape[:2]
    stamp_height, stamp_width = stamp.shape[:2]
    row0 = int(round(y0)) - stamp_height // 2
    col0 = int(round(x0)) - stamp_width // 2
    canvas_row_lo, canvas_row_hi = max(0, row0), min(height, row0 + stamp_height)
    canvas_col_lo, canvas_col_hi = max(0, col0), min(width, col0 + stamp_width)
    if canvas_row_lo >= canvas_row_hi or canvas_col_lo >= canvas_col_hi:
        return False
    stamp_row_lo = canvas_row_lo - row0
    stamp_col_lo = canvas_col_lo - col0
    canvas[canvas_row_lo:canvas_row_hi, canvas_col_lo:canvas_col_hi, :] += stamp[
        stamp_row_lo : stamp_row_lo + (canvas_row_hi - canvas_row_lo),
        stamp_col_lo : stamp_col_lo + (canvas_col_hi - canvas_col_lo),
        :,
    ]
    return True
