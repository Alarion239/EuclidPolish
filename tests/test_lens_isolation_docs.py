from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readmes_describe_the_current_lens_isolation_commands_and_record_contract():
    top_level = (ROOT / "README.md").read_text(encoding="utf-8")
    experiment = (ROOT / "euclid_polish/experiments/lens_isolation/README.md").read_text(encoding="utf-8")
    assert "tng-fraction" not in top_level
    assert "--tng-density-arcmin2" in top_level
    assert "python scripts/lens_isolation_generate.py" in top_level
    assert "dirty_{train,validate,test}.tfrecord" in experiment
    assert "lens_{train,validate,test}.tfrecord" in experiment
    assert "forward_onthefly=False" in experiment
    assert "balanced" not in experiment.lower()
