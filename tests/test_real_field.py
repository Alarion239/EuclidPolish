"""Tests for the cached real-field diagnostics payload."""

from __future__ import annotations

import json

import numpy as np

from euclid_polish.web.helpers.real_field import (
    _accumulate_diagnostics,
    _diagnostic_accumulators,
    _diagnostic_payload,
    _preserve_matching_member_cubes,
    _restore_matching_member_cubes,
)


def test_real_field_diagnostics_use_model_power_cross_correlation():
    rng = np.random.default_rng(4)
    base = rng.normal(size=(64, 64, 4)).astype(np.float32)
    members = np.stack([base, base + rng.normal(0, 0.05, base.shape)], axis=0)
    acc = _diagnostic_accumulators({}, n_members=2)

    _accumulate_diagnostics(acc, members, {})
    payload = _diagnostic_payload(acc, ["m0", "m1"], {})

    json.dumps(payload)
    assert payload["version"] == 2
    assert "correlation" not in payload
    power = payload["model_power"]
    assert power["samples"] == 4
    assert power["pair_indices"] == [[0, 1]]
    assert len(power["r_pairs"]) == 1
    assert len(power["r_pairs"][0]) == len(power["k"])
    assert any(value is not None for value in power["r_cross"])


def test_matching_member_cubes_are_remapped_without_copying(tmp_path):
    cubes = tmp_path / "cubes"
    cubes.mkdir()
    for old_index, value in enumerate((10.0, 20.0, 30.0)):
        for tile in range(2):
            np.save(cubes / f"member{old_index}_{tile:03d}.npy", [value + tile])

    staging, preserved = _preserve_matching_member_cubes(
        cubes, ["a", "b", "c"], ["c", "a", "new"], count=2)
    for path in cubes.glob("*.npy"):
        path.unlink()
    _restore_matching_member_cubes(cubes, staging)

    assert preserved == 4
    np.testing.assert_allclose(np.load(cubes / "member0_000.npy"), [30.0])
    np.testing.assert_allclose(np.load(cubes / "member1_001.npy"), [11.0])
    assert not (cubes / "member2_000.npy").exists()
