"""Regression tests for cluster entry points and scientific dependencies."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LENS_ISOLATION_SCRIPTS = (
    "lens_isolation_generate.py",
    "lens_isolation_train.py",
    "lens_isolation_evaluate.py",
    "lens_isolation_infer.py",
)


@pytest.mark.parametrize("script_name", LENS_ISOLATION_SCRIPTS)
def test_lens_isolation_scripts_run_outside_repo(script_name, tmp_path):
    """Cluster scripts must import the in-place package from any cwd."""
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script_name), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


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
