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


def test_population_payload_files_are_resolved_before_generation(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    galaxy_path = tmp_path / "galaxy.json"
    star_path = tmp_path / "stars.json"
    malformed_path = tmp_path / "malformed.json"
    list_path = tmp_path / "list.json"
    galaxy_path.write_text('{"fingerprint":"galaxy","version":13}')
    star_path.write_text('{"fingerprint":"stars","version":6}')
    malformed_path.write_text('{"version":')
    list_path.write_text('[1,2,3]')
    script = f"""
import json
import scripts.run_pipeline as pipeline

args = pipeline.parse_args([
    "--joint-galaxy-population-file", {str(galaxy_path)!r},
    "--star-prior-file", {str(star_path)!r},
])
pipeline._resolve_population_payload_files(args)
assert json.loads(args.joint_galaxy_population_json)["fingerprint"] == "galaxy"
assert json.loads(args.star_prior_json)["fingerprint"] == "stars"
assert pipeline._args_for_log(args)["joint_galaxy_population_json"].startswith(
    "<embedded JSON:"
)

def assert_rejected(inline, path):
    try:
        pipeline._resolve_json_payload_file(
            inline, path, label="test population",
        )
    except ValueError:
        return
    raise AssertionError("invalid population payload was accepted")

assert_rejected('{{"version":1}}', {str(galaxy_path)!r})
assert_rejected('', {str(malformed_path)!r})
assert_rejected('', {str(list_path)!r})
print("payload-files=ok")
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
    assert "payload-files=ok" in result.stdout
