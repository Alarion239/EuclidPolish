"""Form-input parsers shared across web routes (extracted from app.py)."""
from __future__ import annotations

from typing import Optional


def _parse_asinh_scale(raw: str) -> Optional[float]:
    """Form parser for the asinh-scale knob. Empty string / 0 / bad
    input → None, which makes plot_reconstruction fall back to
    Config.STRETCH_SCALE_E (1000 e⁻, the training default)."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return val if val > 0 else None
