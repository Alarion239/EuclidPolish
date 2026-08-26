"""Import boundaries for the optional training stage in run_pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_generation_only_startup_does_not_import_training_stack(tmp_path):
    """A generation submission must reach its stage without model imports.

    Run this check in a clean interpreter so earlier test imports cannot hide
    an eager dependency. TensorFlow is blocked as well: the stubbed generation
    TensorFlow remains parent-preloaded because Linux generation workers fork
    after startup and share its initialized pages copy-on-write.
    """
    project_root = Path(__file__).resolve().parents[1]
    script = f"""
import importlib.abc
import sys

blocked = (
    "euclid_polish.model",
    "euclid_polish.training",
    "euclid_polish.visualization",
)

class BlockedImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".")
               for name in blocked):
            raise RuntimeError("eager optional import: " + fullname)
        return None

sys.meta_path.insert(0, BlockedImportFinder())
import scripts.run_pipeline as pipeline

pipeline.step_generate = lambda args: None
sys.argv = [
    "run_pipeline.py",
    "--records-dir", {str(tmp_path)!r},
    "--galaxy-density-arcmin2", "0",
    "--ntrain", "0",
    "--nvalid", "0",
    "--ntest", "0",
    "--skip-convolve",
    "--skip-train",
]
raise SystemExit(pipeline.main())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "STEP 2 skipped" in result.stdout
    assert "STEP 3 skipped" in result.stdout
