from __future__ import annotations

import json
import os

import numpy as np

from euclid_polish.eval.ensemble_cube_cache import load_cached_member_stack


def _write_cache(cubes_dir, *, subset, indices, n_members, shape=(8, 8, 4)):
    os.makedirs(cubes_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    for idx in indices:
        for i in range(n_members):
            arr = rng.normal(10, 1, shape).astype(np.float32)
            np.save(os.path.join(cubes_dir, f"member{i}_{idx:05d}.npy"), arr)
    with open(os.path.join(cubes_dir, "viz_index.json"), "w") as f:
        json.dump({"subset": subset, "indices": list(indices),
                   "pca_n": 3, "pca_amps": {},
                   "member_labels": [f"{i:02d}" for i in range(n_members)]}, f)


def test_hit_returns_member_stack(tmp_path):
    d = str(tmp_path / "cubes")
    _write_cache(d, subset="test", indices=[3, 7], n_members=5)
    out = load_cached_member_stack(7, subset="test", cubes_dir=d)
    assert out is not None
    assert out.shape == (5, 8, 8, 4)
    expect = np.stack([np.load(os.path.join(d, f"member{i}_00007.npy"))
                       for i in range(5)], axis=0)
    assert np.allclose(out, expect)


def test_miss_uncached_field(tmp_path):
    d = str(tmp_path / "cubes")
    _write_cache(d, subset="test", indices=[3], n_members=5)
    assert load_cached_member_stack(9, subset="test", cubes_dir=d) is None


def test_miss_subset_mismatch(tmp_path):
    d = str(tmp_path / "cubes")
    _write_cache(d, subset="validate", indices=[3], n_members=5)
    assert load_cached_member_stack(3, subset="test", cubes_dir=d) is None


def test_miss_no_manifest(tmp_path):
    assert load_cached_member_stack(0, subset="test",
                                    cubes_dir=str(tmp_path / "nope")) is None


def test_miss_missing_member_file(tmp_path):
    d = str(tmp_path / "cubes")
    _write_cache(d, subset="test", indices=[2], n_members=3)
    os.remove(os.path.join(d, "member1_00002.npy"))
    assert load_cached_member_stack(2, subset="test", cubes_dir=d) is None
