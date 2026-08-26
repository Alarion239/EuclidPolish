"""Regression tests for Matplotlib-compatible presentation styling."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType


def test_presentation_style_imports_without_runtime_rc_key_type(monkeypatch):
    """Older Matplotlib releases need not export the type-only RcKeyType."""
    import matplotlib as mpl

    fake_typing = ModuleType("matplotlib.typing")
    monkeypatch.setitem(sys.modules, "matplotlib.typing", fake_typing)

    module_path = (
        Path(__file__).parents[1]
        / "euclid_polish"
        / "visualization"
        / "presentation_style.py"
    )
    namespace = runpy.run_path(str(module_path))

    assert namespace["RcKeyType"] is str
    with namespace["presentation_rc"]({"font.size": 17}):
        assert mpl.rcParams["font.size"] == 17
