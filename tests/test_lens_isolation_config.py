from __future__ import annotations

import os

import pytest

from euclid_polish.config import Config
from euclid_polish.experiments.lens_isolation.config import (
    DatasetConfig,
    ExperimentPaths,
    TrainConfig,
    assert_safe_output,
)


def test_default_paths_are_separate_from_production_artifacts():
    paths = ExperimentPaths()
    expected = os.path.join(Config.DATA_DIR, "experiments", "lens_isolation")
    assert paths.root == expected
    assert paths.records == os.path.join(expected, "records")
    assert paths.ensemble == os.path.join(expected, "ensemble")
    assert paths.evaluation == os.path.join(expected, "evaluation")
    assert os.path.commonpath([paths.records, Config.RECORDS_DIR_V2]) != paths.records


def test_safe_output_rejects_production_and_source_paths(tmp_path):
    production_records = tmp_path / "records_v2"
    production_ensemble = tmp_path / "ensemble"
    source = production_ensemble / "member_00"
    roots = (str(production_records), str(production_ensemble))

    for unsafe in (
        production_records,
        production_records / "child",
        production_ensemble,
        source,
        source / "fork",
    ):
        with pytest.raises(ValueError, match="protected|source"):
            assert_safe_output(str(unsafe), source=str(source), protected_roots=roots)


def test_safe_output_accepts_isolated_experiment_path(tmp_path):
    output = tmp_path / "experiments" / "lens_isolation" / "records"
    assert assert_safe_output(
        str(output),
        protected_roots=(str(tmp_path / "records_v2"), str(tmp_path / "ensemble")),
    ) == str(output)


def test_dataset_config_requires_even_balanced_splits():
    cfg = DatasetConfig(n_train=20, n_validate=6, n_test=4, image_size=96)
    assert cfg.positive_fraction == 0.5
    with pytest.raises(ValueError, match="even"):
        DatasetConfig(n_train=3)
    with pytest.raises(ValueError, match="0.5"):
        DatasetConfig(positive_fraction=0.25)


def test_train_config_validates_sources_and_steps():
    cfg = TrainConfig(sources=("member_00", "member_02"), steps=100)
    assert cfg.sources == ("member_00", "member_02")
    with pytest.raises(ValueError, match="source"):
        TrainConfig(sources=())
    with pytest.raises(ValueError, match="steps"):
        TrainConfig(sources=("member_00",), steps=0)
