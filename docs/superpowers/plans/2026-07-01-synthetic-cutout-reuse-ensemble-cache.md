# Reuse cached test-field ensemble cubes for synthetic cutouts — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the synthetic evaluation (syn-lens + syn-gal) reuse the ensemble page's cached per-test-field member cubes — cropping them to each source stamp instead of re-running the CNN — with automatic fallback to inference for uncached fields.

**Architecture:** A new pure-ish reader `ensemble_cube_cache.load_cached_member_stack(field_index, subset)` returns the cached `(M,H,W,C)` member stack for a field (or `None`). `synthetic_runner`'s per-field branch tries the cache first; on a hit it sets `members_full`/`sr_arr` from the cache and skips `sr_from_model`; on a miss it runs inference as today. All downstream cropping / `write_disagreement_cubes` / stamp writes are unchanged, so the output is bit-identical.

**Tech Stack:** Python, numpy, pytest.

---

### Task 1: Cache reader `ensemble_cube_cache.load_cached_member_stack`

**Files:**
- Create: `euclid_polish/eval/ensemble_cube_cache.py`
- Test: `tests/test_ensemble_cube_cache.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ensemble_cube_cache.py`:

```python
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
    # equals the on-disk members for field 7, stacked
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_ensemble_cube_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: euclid_polish.eval.ensemble_cube_cache`.

- [ ] **Step 3: Implement the module**

Create `euclid_polish/eval/ensemble_cube_cache.py`:

```python
"""Read the ensemble page's cached per-field cubes so the synthetic evaluator can
reuse an already-computed member stack for a field instead of re-running the CNN.

The cubes are written by ``euclid_polish.web.helpers.ensemble_viz.job_ensemble_evaluate``
into ``euclid_polish.web.helpers.viewer_data._ensemble_cubes_dir()`` — one
``member{i}_{rec:05d}.npy`` per member per evaluated field, plus a ``viz_index.json``
manifest ``{subset, indices, member_labels, ...}``."""

from __future__ import annotations

import json
import os

import numpy as np

from euclid_polish.config import Config


def _default_cubes_dir() -> str:
    # Must match euclid_polish.web.helpers.viewer_data._ensemble_cubes_dir().
    return os.path.join(Config.VIS_DIR, "ensemble", "cubes")


def load_cached_member_stack(field_index: int, *, subset: str,
                             cubes_dir: str | None = None) -> np.ndarray | None:
    """Return the cached ``(M, H, W, C)`` member stack for ``field_index``, or ``None``.

    Returns ``None`` unless ``<cubes_dir>/viz_index.json`` loads, its ``subset`` equals
    ``subset``, ``field_index`` is among its ``indices``, and every
    ``member{i}_{field_index:05d}.npy`` (``i`` in ``range(len(member_labels))``) exists.
    Never raises — any error degrades to ``None`` so callers fall back to inference.
    """
    d = cubes_dir or _default_cubes_dir()
    try:
        with open(os.path.join(d, "viz_index.json")) as f:
            man = json.load(f)
        if str(man.get("subset", "")) != str(subset):
            return None
        indices = {int(i) for i in man.get("indices", [])}
        if int(field_index) not in indices:
            return None
        n_members = len(man.get("member_labels", []) or [])
        if n_members <= 0:
            return None
        stack = []
        for i in range(n_members):
            p = os.path.join(d, f"member{i}_{int(field_index):05d}.npy")
            if not os.path.isfile(p):
                return None
            stack.append(np.load(p).astype(np.float32))
        return np.stack(stack, axis=0)
    except (OSError, ValueError, KeyError, TypeError):
        return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_ensemble_cube_cache.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/eval/ensemble_cube_cache.py tests/test_ensemble_cube_cache.py
git commit -m "eval: cache reader for ensemble per-field member cubes"
```

---

### Task 2: Wire the cache into `synthetic_runner` (both subgroups, with fallback)

**Files:**
- Modify: `euclid_polish/eval/synthetic_runner.py` (subset capture near line 159-161; deferred-import group near line ~140; per-field branch at lines 247-251)
- Test: `tests/test_synthetic_runner_hr.py`

- [ ] **Step 1: Read the current code and the existing test fixture**

Read `euclid_polish/eval/synthetic_runner.py` lines 135-260 and `tests/test_synthetic_runner_hr.py` in full. Confirm: (a) `sub = eval_subset(rdir)` at line ~159 is later **reassigned** to the object id `sub = f"{grade}_{idx:04d}_{rank}"` at line ~237 (so the subset name must be preserved in a separate variable before the loop); (b) the deferred-import group near line ~140 already imports `sr_from_model` (added in a prior task); (c) `lr_cube` is cropped for the LR stamp later, so it must be loaded on every new field regardless of cache. Note the field index(es) and `member_labels`/member-count the existing test's fixture produces — you'll need them for the reuse test.

- [ ] **Step 2: Write the failing test**

Add to `tests/test_synthetic_runner_hr.py` (adapt fixture names to the file's existing helpers you read in Step 1 — reuse its records/source-catalog setup; only the cache setup + assertions below are new):

```python
def test_reuses_cached_ensemble_cubes(tmp_path, monkeypatch):
    # Arrange the same synthetic inputs the existing tests use (records +
    # sources_<subset>.csv in a temp records dir), then ALSO stage an ensemble
    # cube cache for the field(s) the run will touch, and prove inference is skipped.
    import json
    import numpy as np
    from euclid_polish.config import Config
    from euclid_polish.eval import synthetic_runner

    # --- reuse the existing fixture to build records + source catalog ---
    # (Use the same setup the other tests in this file use; it yields a records
    # dir with dirty_/hr_/sources_<subset>, and the field indices present.)
    env = _make_synth_fixture(tmp_path)   # <- replace with this file's actual helper
    records_dir = env["records_dir"]
    subset = env["subset"]                # e.g. "test" or "validate"
    field_indices = env["field_indices"]  # indices present in the records + catalog
    hr_shape = env["hr_field_shape"]      # (H, W, C) of an HR field

    # --- stage the ensemble cube cache under a temp Config.VIS_DIR ---
    vis_dir = tmp_path / "vis"
    cubes_dir = vis_dir / "ensemble" / "cubes"
    cubes_dir.mkdir(parents=True)
    n_members = 4
    rng = np.random.default_rng(0)
    for idx in field_indices:
        for i in range(n_members):
            np.save(cubes_dir / f"member{i}_{idx:05d}.npy",
                    rng.normal(10, 1, hr_shape).astype(np.float32))
    with open(cubes_dir / "viz_index.json", "w") as f:
        json.dump({"subset": subset, "indices": list(field_indices),
                   "pca_n": 3, "pca_amps": {},
                   "member_labels": [f"{i:02d}" for i in range(n_members)]}, f)
    monkeypatch.setattr(Config, "VIS_DIR", str(vis_dir))

    # --- make ANY inference fail: a cache hit must not call sr_from_model ---
    def _boom(*a, **k):
        raise AssertionError("sr_from_model called — cache reuse failed")
    monkeypatch.setattr("euclid_polish.eval.ensemble_infer.sr_from_model", _boom)

    # Act
    out = synthetic_runner.run_synthetic_eval(
        str(tmp_path / "out"), n=1, model=object(),   # model unused on cache hit
        records_dir=records_dir, seed=0)

    # Assert: produced at least one ok cutout, and it has the disagreement cubes,
    # all WITHOUT calling sr_from_model.
    ok_rows = [r for r in out["rows"] if r["ok"]]
    assert ok_rows, "no synthetic cutouts produced from cache"
    sub0 = ok_rows[0]["out_subdir"]
    for name in ("SR.fits", "std.fits", "pca0.fits"):
        assert (tmp_path / "out" / sub0 / name).is_file()
```

If `tests/test_synthetic_runner_hr.py` has no reusable fixture helper, build the records + `sources_<subset>.csv` inline the same way the file's existing tests do (copy their setup), keeping `field_indices`/`hr_shape` consistent with what you write into the cache.

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/test_synthetic_runner_hr.py::test_reuses_cached_ensemble_cubes -v`
Expected: FAIL — currently `sr_from_model` is always called, so `_boom` raises `AssertionError`.

- [ ] **Step 4: Preserve the subset name (before the loop clobbers `sub`)**

In `euclid_polish/eval/synthetic_runner.py`, right after the source catalog is loaded (after line ~161 `by_field = read_sources(src_csv)`), add:

```python
    field_subset = sub                          # preserve subset; `sub` is reused as the
                                                # per-object id inside the loop below
```

- [ ] **Step 5: Add the deferred import**

In the function's deferred-import group near line ~140 (where `sr_from_model` is imported), add:

```python
    from euclid_polish.eval.ensemble_cube_cache import load_cached_member_stack
```

- [ ] **Step 6: Branch on cache hit in the per-field block**

Replace the per-field branch (lines 247-251):

```python
            if idx != cur_idx:                  # same field → reuse the SR cube
                lr_cube = np.asarray(lr_by[idx].data, dtype=np.float32)   # (H,W,4)
                _, sr_data, members_full = sr_from_model(model, lr_cube)
                sr_arr = np.asarray(sr_data, dtype=np.float32)            # (2H,2W,4)
                cur_idx = idx
```

with:

```python
            if idx != cur_idx:                  # same field → reuse per-field arrays
                lr_cube = np.asarray(lr_by[idx].data, dtype=np.float32)   # (H,W,4)
                cached = load_cached_member_stack(idx, subset=field_subset)
                if cached is not None:          # reuse the ensemble page's cubes
                    members_full = cached                              # (M,2H,2W,C)
                    sr_arr = members_full.mean(axis=0).astype(np.float32)
                    _emit(f"  field {idx}: reused ensemble cache "
                          f"({members_full.shape[0]} members)")
                else:                           # no cache → run inference on the field
                    _, sr_data, members_full = sr_from_model(model, lr_cube)
                    sr_arr = np.asarray(sr_data, dtype=np.float32)     # (2H,2W,C)
                    _emit(f"  field {idx}: inference")
                cur_idx = idx
```

(`lr_cube` is loaded in both paths because the LR stamp is cropped from it downstream; only the CNN inference is skipped on a hit.)

- [ ] **Step 7: Run the test to verify it passes**

Run: `pytest tests/test_synthetic_runner_hr.py::test_reuses_cached_ensemble_cubes -v`
Expected: PASS (cutouts produced with no `sr_from_model` call).

- [ ] **Step 8: Run the synthetic-runner regression + the new module tests**

Run: `pytest tests/ -k "synthetic or ensemble_cube_cache or grouped or eval_catalog" -v`
Expected: PASS (the existing synthetic tests still pass — they don't stage a cache, so they exercise the inference fallback).

- [ ] **Step 9: Commit**

```bash
git add euclid_polish/eval/synthetic_runner.py tests/test_synthetic_runner_hr.py
git commit -m "eval(synthetic): reuse cached ensemble member cubes, fall back to inference"
```

---

### Task 3: Full suite

- [ ] **Step 1: Run the full suite**

Run: `pytest tests/ -q`
Expected: PASS (no regressions). Report counts.

- [ ] **Step 2: Confirm the workflow note is discoverable**

Confirm the per-field `_emit` logs (`reused ensemble cache` / `inference`) are present so a run visibly reports reuse vs inference. No commit needed unless a log/comment tweak is warranted.

---

## Self-review notes

- Spec coverage: cache reader with subset+index+member-file guards and never-raises contract (Task 1) ✅; both subgroups reuse via the shared per-field branch (Task 2 — the branch is above the `_SUBGROUPS` loop's per-source work, so it serves lens and galaxy alike) ✅; fallback to inference on miss (Task 2 Step 6 `else`) ✅; bit-identical (crop cached members → existing `write_disagreement_cubes`) ✅; `cubes_dir` from `Config.VIS_DIR`, no web import ✅; subset-name preserved past the `sub` reassignment ✅; `lr_cube` still loaded for the LR stamp ✅.
- Placeholder scan: the only non-literal is the reuse-test's fixture helper `_make_synth_fixture`/inline setup, which Step 1/Step 2 explicitly direct the implementer to source from the existing `tests/test_synthetic_runner_hr.py` — the cache-staging + assertions (the actual new logic under test) are fully specified.
- Type consistency: `load_cached_member_stack(field_index, *, subset, cubes_dir=None) -> np.ndarray|None` defined in Task 1 and called identically in Task 2; `field_subset` used as the `subset` arg.
