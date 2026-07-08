from __future__ import annotations

import json
import os

import numpy as np

from euclid_polish.eval.ensemble_cube_cache import (
    cached_member_labels,
    load_cached_member_stack,
)


def _write_cache(cubes_dir, *, subset, indices, n_members, shape=(8, 8, 4)):
    os.makedirs(cubes_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    for idx in indices:
        for i in range(n_members):
            arr = rng.normal(10, 1, shape).astype(np.float32)
            np.save(os.path.join(cubes_dir, f"member{i}_{idx:05d}.npy"), arr)
    labels = [f"{i:02d}" for i in range(n_members)]
    with open(os.path.join(cubes_dir, "viz_index.json"), "w") as f:
        json.dump({"subset": subset, "indices": list(indices),
                   "pca_n": 3, "pca_amps": {},
                   "member_labels": labels}, f)
    return labels


def test_hit_returns_member_stack(tmp_path):
    d = str(tmp_path / "cubes")
    labels = _write_cache(d, subset="test", indices=[3, 7], n_members=5)
    out = load_cached_member_stack(7, subset="test", cubes_dir=d, active=labels)
    assert out is not None
    assert out.shape == (5, 8, 8, 4)
    expect = np.stack([np.load(os.path.join(d, f"member{i}_00007.npy"))
                       for i in range(5)], axis=0)
    assert np.allclose(out, expect)


def test_miss_uncached_field(tmp_path):
    d = str(tmp_path / "cubes")
    labels = _write_cache(d, subset="test", indices=[3], n_members=5)
    assert load_cached_member_stack(9, subset="test", cubes_dir=d,
                                    active=labels) is None


def test_miss_subset_mismatch(tmp_path):
    d = str(tmp_path / "cubes")
    labels = _write_cache(d, subset="validate", indices=[3], n_members=5)
    assert load_cached_member_stack(3, subset="test", cubes_dir=d,
                                    active=labels) is None


def test_miss_no_manifest(tmp_path):
    assert load_cached_member_stack(0, subset="test",
                                    cubes_dir=str(tmp_path / "nope")) is None


def test_miss_missing_member_file(tmp_path):
    d = str(tmp_path / "cubes")
    labels = _write_cache(d, subset="test", indices=[2], n_members=3)
    os.remove(os.path.join(d, "member1_00002.npy"))
    assert load_cached_member_stack(2, subset="test", cubes_dir=d,
                                    active=labels) is None


def test_stale_membership_deletes_cache(tmp_path):
    """Archived member since the cache was written → whole dir purged lazily."""
    d = str(tmp_path / "cubes")
    _write_cache(d, subset="test", indices=[3], n_members=2)
    out = load_cached_member_stack(3, subset="test", cubes_dir=d,
                                   active=["00"])          # member 01 retired
    assert out is None
    assert not os.path.isdir(d)                            # purged on read


def test_cached_member_labels(tmp_path):
    d = str(tmp_path / "cubes")
    labels = _write_cache(d, subset="test", indices=[3], n_members=2)
    assert cached_member_labels(d) == labels
    assert cached_member_labels(str(tmp_path / "nope")) is None


# ---------------------------------------------------------------------------
# subset-keyed cube directories + the factored per-field cube writer
# ---------------------------------------------------------------------------

def test_ensemble_cubes_dir_is_subset_keyed():
    from euclid_polish.web.helpers import ensemble_viz as ev
    test_dir = ev._ensemble_cubes_dir()               # no-arg = test bucket
    val_dir = ev._ensemble_cubes_dir("validate")
    assert test_dir.endswith(os.sep + "cubes")
    assert val_dir.endswith(os.sep + "cubes_validate")
    assert test_dir != val_dir


def test_cache_field_cubes_roundtrips_through_reader(tmp_path):
    """The factored writer lays down member/sr/std cubes that the cube-cache
    reader can load back as a stack (with a matching manifest)."""
    from euclid_polish.web.helpers import ensemble_viz as ev

    d = str(tmp_path / "cubes_validate")
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(1)
    preds = rng.normal(10, 1, (4, 8, 8, 4)).astype(np.float32)
    mean, std = preds.mean(0), preds.std(0)
    amps, var = ev._cache_field_cubes(d, 5, preds, mean, std)
    assert len(amps) == 3 and len(var) == 3
    assert os.path.isfile(os.path.join(d, "sr_00005.npy"))
    assert os.path.isfile(os.path.join(d, "member3_00005.npy"))

    labels = [f"{i:02d}" for i in range(4)]
    with open(os.path.join(d, "viz_index.json"), "w") as f:
        json.dump({"subset": "validate", "indices": [5],
                   "member_labels": labels}, f)
    out = load_cached_member_stack(5, subset="validate", cubes_dir=d,
                                   active=labels)
    assert out is not None and out.shape == (4, 8, 8, 4)
    np.testing.assert_allclose(out, preds, rtol=1e-6)
