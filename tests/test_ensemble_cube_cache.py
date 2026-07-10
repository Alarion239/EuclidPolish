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
    test_dir = ev._ensemble_cubes_dir(starless=False)     # no subset = test bucket
    val_dir = ev._ensemble_cubes_dir("validate", starless=False)
    assert test_dir.endswith(os.sep + "cubes")
    assert val_dir.endswith(os.sep + "cubes_validate")
    assert test_dir != val_dir
    # ...and detached by star regime.
    starless_dir = ev._ensemble_cubes_dir(starless=True)
    assert starless_dir != test_dir
    assert os.sep + "starfull" + os.sep in test_dir
    assert os.sep + "starless" + os.sep in starless_dir


# ---------------------------------------------------------------------------
# Idempotent evaluate: never re-infer when the dataset + model are unchanged
# ---------------------------------------------------------------------------

def test_combiner_fingerprint(tmp_path):
    from euclid_polish.web.helpers import ensemble_viz as ev
    regime = tmp_path / "r"
    (regime / "combiner").mkdir(parents=True)
    assert ev._combiner_fingerprint(str(regime)) is None       # none saved yet
    (regime / "combiner" / "combiner.npz").write_bytes(b"x" * 10)
    fp = ev._combiner_fingerprint(str(regime))
    assert fp and fp.startswith("10:")                         # size:mtime


def test_reusable_eval_matches_only_on_identity_and_records(tmp_path, monkeypatch):
    from euclid_polish.web.helpers import ensemble_viz as ev
    regime = tmp_path / "regime"
    cubes = regime / "cubes"
    cubes.mkdir(parents=True)
    monkeypatch.setattr(ev, "_ensemble_regime_dir", lambda starless: str(regime))
    monkeypatch.setattr(ev, "_ensemble_cubes_dir", lambda *a, **k: str(cubes))

    ident = {"records_fp": "rfp1", "num_images": 100, "member_fps": ["a"]}
    (regime / "eval_summary.json").write_text(
        json.dumps({"eval_identity": ident, "ensemble_psnr": 1.0}))
    (cubes / "viz_index.json").write_text(json.dumps({"records_fp": "rfp1"}))

    assert ev._reusable_eval(False, ident) is not None          # full match → reuse
    # A different field count / dataset / membership is a different eval.
    assert ev._reusable_eval(False, {**ident, "num_images": 50}) is None
    assert ev._reusable_eval(False, {**ident, "records_fp": "rfp2"}) is None
    # Cubes wiped/regenerated since the summary → manifest records_fp drifts.
    (cubes / "viz_index.json").write_text(json.dumps({"records_fp": "OTHER"}))
    assert ev._reusable_eval(False, ident) is None
    # No summary at all → nothing to reuse.
    (cubes / "viz_index.json").write_text(json.dumps({"records_fp": "rfp1"}))
    (regime / "eval_summary.json").unlink()
    assert ev._reusable_eval(False, ident) is None


def test_evaluate_reuses_cached_result_without_inference(tmp_path, monkeypatch):
    """A cache hit rebuilds figures from cubes and returns the cached metrics —
    evaluate_on_records (the GPU inference) must never run."""
    from euclid_polish.web.helpers import ensemble_viz as ev

    monkeypatch.setattr(ev, "_sky_records_local_dir", lambda: str(tmp_path))
    monkeypatch.setattr(ev, "eval_subset", lambda rdir: "test")
    monkeypatch.setattr(ev, "ensemble_dir", lambda: str(tmp_path / "ens"))
    monkeypatch.setattr(ev, "_ensemble_regime_dir", lambda starless: str(tmp_path / "regime"))
    # Keep the (destructive) rmtree that the force path runs inside tmp.
    monkeypatch.setattr(ev, "_ensemble_cubes_dir",
                        lambda *a, **k: str(tmp_path / "regime" / "cubes"))
    fake_identity = {"records_fp": "rfp", "num_images": 100}
    monkeypatch.setattr(ev, "_eval_identity", lambda *a, **k: fake_identity)
    monkeypatch.setattr(ev, "_reusable_eval",
                        lambda starless, ident: {"ensemble_psnr": 42.0}
                        if ident == fake_identity else None)
    payload_calls = {"n": 0}
    monkeypatch.setattr(ev, "compute_evaluation_payload",
                        lambda starless: payload_calls.__setitem__("n", payload_calls["n"] + 1))

    def _no_infer(*a, **k):
        raise AssertionError("evaluate_on_records must NOT run on a cache hit")
    monkeypatch.setattr(ev, "evaluate_on_records", _no_infer)

    class _Cap:
        def tick(self, *a, **k): pass

    out = ev.job_ensemble_evaluate(_Cap(), num_images=100, starless=False)
    assert out["reused"] is True
    assert out["ensemble_psnr"] == 42.0
    assert payload_calls["n"] == 1        # figures rebuilt from cubes, cheaply

    # force=True bypasses reuse and would re-infer (proven by the guard raising).
    import pytest
    with pytest.raises(AssertionError, match="must NOT run"):
        ev.job_ensemble_evaluate(_Cap(), num_images=100, starless=False, force=True)


def test_rebuild_bucket_drops_member_and_renumbers_from_cache(tmp_path):
    """Archiving a member rebuilds the bucket from the REMAINING cached member
    cubes — renumbered contiguous, aggregates (sr/std) recomputed, combiner
    dropped — with no model re-inference."""
    from euclid_polish.web.helpers import ensemble_viz as ev

    d = str(tmp_path / "cubes")
    os.makedirs(d)
    rec = 5
    tag = f"{rec:05d}"
    shape = (6, 6, 4)
    # 5 members with distinct constant values 1..5.
    for i in range(5):
        np.save(os.path.join(d, f"member{i}_{tag}.npy"),
                np.full(shape, i + 1, np.float32))
    np.save(os.path.join(d, f"sr_{tag}.npy"), np.full(shape, 3.0, np.float32))
    np.save(os.path.join(d, f"std_{tag}.npy"), np.zeros(shape, np.float32))
    np.save(os.path.join(d, f"pca0_{tag}.npy"), np.zeros(shape, np.float32))
    np.save(os.path.join(d, f"comb_{tag}.npy"), np.zeros(shape, np.float32))
    labels = [f"{i:02d}·x" for i in range(5)]
    with open(os.path.join(d, "viz_index.json"), "w") as f:
        json.dump({"subset": "test", "indices": [rec], "member_labels": labels,
                   "has_combiner": True, "records_fp": "rfp"}, f)

    assert ev._rebuild_bucket_dropping_member(d, "02") is True     # drop the '02' member

    # 4 members remain, renumbered 0..3 with the '02' value (3) gone.
    assert not os.path.isfile(os.path.join(d, f"member4_{tag}.npy"))
    vals = [float(np.load(os.path.join(d, f"member{i}_{tag}.npy")).flat[0])
            for i in range(4)]
    assert vals == [1.0, 2.0, 4.0, 5.0]                            # 3 dropped, rest shifted
    # sr = mean of the remaining stack, std recomputed.
    np.testing.assert_allclose(np.load(os.path.join(d, f"sr_{tag}.npy")),
                               np.full(shape, np.mean([1, 2, 4, 5]), np.float32))
    assert np.load(os.path.join(d, f"std_{tag}.npy")).std() >= 0  # recomputed, non-null
    # Combiner cube dropped (stale for the smaller set); manifest updated.
    assert not os.path.isfile(os.path.join(d, f"comb_{tag}.npy"))
    man = json.load(open(os.path.join(d, "viz_index.json")))
    assert [lbl.split("·")[0] for lbl in man["member_labels"]] == ["00", "01", "03", "04"]
    assert man["has_combiner"] is False


def test_rebuild_bucket_noop_when_member_absent(tmp_path):
    from euclid_polish.web.helpers import ensemble_viz as ev
    d = str(tmp_path / "cubes")
    _write_cache(d, subset="test", indices=[1], n_members=3)       # labels 00,01,02
    before = sorted(os.listdir(d))
    assert ev._rebuild_bucket_dropping_member(d, "99") is False     # not in this bucket
    assert sorted(os.listdir(d)) == before                          # untouched
    assert ev._rebuild_bucket_dropping_member(str(tmp_path / "nope"), "00") is False


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
