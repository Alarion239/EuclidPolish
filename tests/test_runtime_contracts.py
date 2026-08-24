"""Regression tests for cluster entry points and scientific dependencies."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_environment_pins_supported_photutils_release():
    """FastEPSFBuilder relies on the Photutils 2.3 builder implementation."""
    environment = (REPO_ROOT / "environment.yml").read_text(encoding="utf-8")

    assert "  - photutils=2.3.*\n" in environment


def test_quality_workflow_runs_full_suite_in_ci():
    """Resource-heavy full-suite verification belongs on the CI runner."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "quality.yml").read_text(
        encoding="utf-8"
    )

    assert "      - name: Run full test suite\n" in workflow
    assert '          EUCLID_POLISH_DISABLE_AUTO_SSH: "1"\n' in workflow
    assert '          NUMBA_DISABLE_JIT: "1"\n' in workflow
    assert '          PYTEST_DISABLE_PLUGIN_AUTOLOAD: "1"\n' in workflow
    assert "        run: python -m pytest -q\n" in workflow
