# Stale-Cube Purge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete cached cubes automatically as soon as no reader can reuse them (spec: `docs/superpowers/specs/2026-10-06-stale-cube-purge-design.md`).

**Architecture:** A pure, file-system-only `purge_stale_bucket` in the cube-cache module applies the writers' own staleness rules; sibling purges cover real-field cubes and the experiments member-SR cache. Jobs that make cubes stale write a durable "purge pending" flag; a job-registry finish hook starts one visible purge job when the registry is idle.

**Tech Stack:** Python 3.11, NumPy, Flask job registry (`euclid_polish/web/jobs.py`), pytest.

Run tests with the project env: `conda run -n EuclidPolishEnv python -m pytest …` (below abbreviated `pytest`).

---

## File map

| File | Change |
|---|---|
| `euclid_polish/eval/ensemble_cube_cache.py` | quiet removals; `PurgeResult`, `tree_usage`, `recorded_records_fp`, `purge_stale_bucket` |
| `euclid_polish/eval/spatial_gate_fit.py` | `build_blackout_fields` prunes fields not in `indices` |
| `euclid_polish/web/helpers/real_field.py` | `member_fps` in the manifest; fingerprint-aware reuse + refresh; `purge_stale_real_field(s)` |
| `euclid_polish/web/helpers/experiments.py` | `purge_stale_member_sr_cache` |
| `euclid_polish/web/helpers/purge_requests.py` (new) | pending flag: `request_stale_purge`, `read_pending`, `clear_pending` |
| `euclid_polish/web/jobs.py` | `JobRegistry.add_finish_hook` |
| `euclid_polish/web/helpers/stale_purge.py` (new) | `purge_stale_caches`, `job_stale_purge`, `maybe_start_stale_purge`, `on_job_finished` |
| `euclid_polish/web/helpers/ensemble_viz.py`, `routes/fasrc.py`, `routes/views.py`, `web/app.py` | triggers + hook install |
| tests | `tests/test_ensemble_cube_cache_purge.py`, `tests/test_stale_purge.py` (new); additions to `test_real_field.py`, `test_experiments.py`, `test_jobs.py`, `test_spatial_gate_fit_blackout_cache.py` |

---

### Task 1: `purge_stale_bucket`

**Files:** Modify `euclid_polish/eval/ensemble_cube_cache.py`; Test `tests/test_ensemble_cube_cache_purge.py`

- [ ] **Step 1: Write the failing tests**

```python
"""purge_stale_bucket deletes exactly what a bucket's next writer would
discard: departed and continued members (with the aggregates their stack
made), unproven positional cubes, buckets made from other records, files of
unlisted members and — blackout buckets only — fields outside the manifest."""
from __future__ import annotations

import json
import os

import numpy as np

from euclid_polish.eval.ensemble_cube_cache import (
    BLACKOUT_INDEX,
    VIZ_INDEX,
    _remove_quietly,
    member_cube_path,
    purge_stale_bucket,
    write_bucket_manifest,
)

A, B = "01·psnr", "02·psnr"
FPS = {A: "ckpt-1:a", B: "ckpt-1:b"}
AGGREGATES = ("sr", "std", "pca0", "comb_spatial_gate")


def _bucket(tmp_path, *, labels=(A, B), fps=None, indices=(0, 1), name=VIZ_INDEX,
            records_fp="rec-1", positional=False):
    d = tmp_path / "cubes"
    d.mkdir()
    for position, label in enumerate(labels):
        for rec in indices:
            path = (d / f"member{position}_{rec:05d}.npy" if positional
                    else member_cube_path(str(d), label, rec))
            np.save(path, np.full(4, position, np.float32))
    for rec in indices:
        np.save(d / f"lr_{rec:05d}.npy", np.zeros(4, np.float32))
        for prefix in AGGREGATES:
            np.save(d / f"{prefix}_{rec:05d}.npy", np.zeros(4, np.float32))
    manifest = {"member_labels": list(labels), "indices": list(indices),
                "pca_amps": {"0": [1.0]}, "pca_var": {"0": [0.5]},
                "has_combiner_spatial_gate": True}
    if not positional:
        manifest["member_fps"] = dict(FPS if fps is None else fps)
    if name == VIZ_INDEX:
        manifest["records_fp"] = records_fp
    else:
        manifest["identity"] = {"seed": 0, "source": records_fp}
    write_bucket_manifest(str(d), manifest, name)
    return d


def _manifest(d, name=VIZ_INDEX):
    return json.loads((d / name).read_text())


def test_a_current_bucket_is_left_alone(tmp_path):
    d = _bucket(tmp_path)
    before = sorted(os.listdir(d))

    result = purge_stale_bucket(str(d), current=FPS, records_fp="rec-1")

    assert result.dropped == [] and result.wiped is None
    assert result.bytes_freed == 0 and result.files_deleted == 0
    assert sorted(os.listdir(d)) == before


def test_a_departed_member_loses_its_cubes_and_the_aggregates(tmp_path):
    d = _bucket(tmp_path)

    result = purge_stale_bucket(str(d), current={A: FPS[A]}, records_fp="rec-1")

    assert result.dropped == [B]
    assert not any(n.startswith("member_02_") for n in os.listdir(d))
    assert all((d / f"member_01_{rec:05d}.npy").is_file() for rec in (0, 1))
    assert all((d / f"lr_{rec:05d}.npy").is_file() for rec in (0, 1))
    assert not any(n.startswith(AGGREGATES) for n in os.listdir(d))
    manifest = _manifest(d)
    assert manifest["member_labels"] == [A]
    assert manifest["member_fps"] == {A: FPS[A]}
    assert manifest["has_combiner_spatial_gate"] is False
    assert manifest["pca_amps"] == {} and manifest["pca_var"] == {}
    assert result.files_deleted == 2 + 2 * len(AGGREGATES)
    assert result.bytes_freed > 0


def test_a_continued_member_is_dropped(tmp_path):
    d = _bucket(tmp_path)

    result = purge_stale_bucket(str(d), current={**FPS, B: "ckpt-2:b"}, records_fp="rec-1")

    assert result.dropped == [B]
    assert _manifest(d)["member_labels"] == [A]


def test_an_unproven_positional_bucket_drops_every_member(tmp_path):
    d = _bucket(tmp_path, positional=True)

    result = purge_stale_bucket(str(d), current=FPS, records_fp="rec-1")

    assert result.dropped == [A, B]
    assert sorted(os.listdir(d)) == ["lr_00000.npy", "lr_00001.npy", VIZ_INDEX]
    assert _manifest(d)["member_labels"] == []


def test_a_positional_bucket_keeps_members_whose_fingerprints_are_proven(tmp_path):
    d = _bucket(tmp_path, positional=True)

    result = purge_stale_bucket(str(d), current=FPS, records_fp="rec-1", adopt=FPS)

    assert result.dropped == []
    assert (d / "member_02_00001.npy").is_file()
    assert _manifest(d)["member_fps"] == FPS


def test_a_bucket_made_from_other_records_is_wiped(tmp_path):
    d = _bucket(tmp_path)

    result = purge_stale_bucket(str(d), current=FPS, records_fp="rec-2")

    assert result.wiped == "made from other records"
    assert not d.exists()
    assert result.files_deleted == 2 * 2 + 2 * (1 + len(AGGREGATES)) + 1


def test_unknown_current_records_never_wipe(tmp_path):
    d = _bucket(tmp_path)

    result = purge_stale_bucket(str(d), current=FPS, records_fp=None)

    assert result.wiped is None and d.is_dir()


def test_a_blackout_bucket_is_checked_against_its_stamping_source(tmp_path):
    d = _bucket(tmp_path, name=BLACKOUT_INDEX, records_fp=None)

    result = purge_stale_bucket(str(d), name=BLACKOUT_INDEX, current=FPS, records_fp="rec-1")

    assert result.wiped == "made from other records"


def test_cubes_without_a_manifest_are_wiped_but_an_empty_dir_is_not(tmp_path):
    d = tmp_path / "cubes"
    d.mkdir()
    assert purge_stale_bucket(str(d), current=FPS).wiped is None
    np.save(d / "member_01_00000.npy", np.zeros(4, np.float32))

    result = purge_stale_bucket(str(d), current=FPS)

    assert result.wiped == "no manifest" and not d.exists()


def test_files_of_unlisted_members_are_deleted(tmp_path):
    d = _bucket(tmp_path)
    np.save(d / "member_09_00000.npy", np.zeros(4, np.float32))
    np.save(d / "member3_00000.npy", np.zeros(4, np.float32))

    result = purge_stale_bucket(str(d), current=FPS, records_fp="rec-1")

    assert result.dropped == [] and result.files_deleted == 2
    assert (d / "sr_00000.npy").is_file()


def test_fields_outside_the_manifest_go_only_from_blackout_buckets(tmp_path):
    for name in (VIZ_INDEX, BLACKOUT_INDEX):
        root = tmp_path / name
        root.mkdir()
        d = _bucket(root, name=name, indices=(0,))
        np.save(d / "lr_00001.npy", np.zeros(4, np.float32))
        np.save(member_cube_path(str(d), A, 1), np.zeros(4, np.float32))

        purge_stale_bucket(str(d), name=name, current=FPS, records_fp="rec-1")

        kept = (d / "lr_00001.npy").is_file() and (d / "member_01_00001.npy").is_file()
        assert kept == (name == VIZ_INDEX)


def test_removing_a_missing_file_is_quiet(tmp_path):
    _remove_quietly(str(tmp_path / "gone.npy"))
```

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/test_ensemble_cube_cache_purge.py -q`
Expected: ImportError (`_remove_quietly`, `purge_stale_bucket`).

- [ ] **Step 3: Implement** — in `ensemble_cube_cache.py`:

Add `import contextlib` and `import shutil` to the imports. Next to `_POSITIONAL_FILE` / `_FIELD_FILE`:

```python
_LABEL_KEYED_FILE = re.compile(r"member_([A-Za-z0-9-]+)_\d{5}\.npy")
#: Per-field files a membership change leaves valid: member cubes and the LR input.
_MEMBERSHIP_FREE_FILE = re.compile(r"(member_[A-Za-z0-9-]+|lr)_\d{5}\.npy")
```

After `_member_files`:

```python
def _remove_quietly(path: str) -> None:
    """Delete ``path``; a file another writer or a purge already removed is fine."""
    with contextlib.suppress(FileNotFoundError):
        os.remove(path)
```

Replace the `os.remove(...)` calls in `prune_bucket_fields`, `migrate_positional_bucket` and `sync_bucket_members` with `_remove_quietly(...)`, and wrap the `os.replace` in `migrate_positional_bucket` in `with contextlib.suppress(FileNotFoundError):`.

New section before `# Readers`:

```python
# --------------------------------------------------------------------------- #
# Purge (no inference)
# --------------------------------------------------------------------------- #

@dataclass
class PurgeResult:
    """What :func:`purge_stale_bucket` deleted from one bucket."""

    cubes_dir: str
    bytes_freed: int = 0
    files_deleted: int = 0
    wiped: str | None = None                            # why the whole bucket went
    dropped: list[str] = field(default_factory=list)    # members whose cubes went


def tree_usage(path: str) -> tuple[int, int]:
    """``(bytes, files)`` under ``path``; ``(0, 0)`` when it is missing."""
    total = files = 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(root, name)).st_size
                files += 1
    return total, files


def recorded_records_fp(manifest: Mapping) -> str | None:
    """The records fingerprint a bucket was made from: ``records_fp`` (test and
    validate buckets) or the blackout stamping's ``identity.source``."""
    if "records_fp" in manifest:
        return manifest.get("records_fp")
    return (manifest.get("identity") or {}).get("source")


def purge_stale_bucket(cubes_dir: str, *, name: str = VIZ_INDEX,
                       current: Mapping[str, str | None],
                       records_fp: str | None = None,
                       adopt: Mapping[str, str | None] | None = None) -> PurgeResult:
    """Delete what the bucket's next writer would discard, without inference.

    ``current`` maps the regime's ACTIVE member labels to their checkpoint
    fingerprints now; ``records_fp`` is the fingerprint of the records the
    bucket's writer would use now (``None``: unknown, never wipes).

    - ``.npy`` files without a manifest, or a bucket made from other records:
      the directory goes (every writer would empty it);
    - a positional bucket is migrated first (:func:`migrate_positional_bucket`,
      adopting ``adopt``);
    - a member that is not active or whose recorded fingerprint differs from
      ``current`` loses its cubes and its manifest entry, and so do member
      files of labels the manifest does not keep;
    - when a member went, the per-field aggregates (all but member cubes and
      ``lr_``) go too, with the ``has_combiner*`` flags and PCA amplitudes;
    - a blackout bucket also loses every per-field file of a field outside
      its ``indices`` (nothing proves it belongs to the current stamping). A
      test or validate bucket keeps them: an interrupted fill's cubes, which
      the next fill reuses.

    Afterwards every label the manifest lists has current cubes."""
    result = PurgeResult(cubes_dir)
    if not os.path.isdir(cubes_dir):
        return result
    before = tree_usage(cubes_dir)
    manifest = read_bucket_manifest(cubes_dir, name)
    if manifest is None and not any(n.endswith(".npy") for n in os.listdir(cubes_dir)):
        return result
    if manifest is None or (records_fp is not None
                            and recorded_records_fp(manifest) != records_fp):
        result.wiped = "no manifest" if manifest is None else "made from other records"
        shutil.rmtree(cubes_dir, ignore_errors=True)
        result.bytes_freed, result.files_deleted = before
        return result
    if not is_label_keyed(manifest):
        manifest = migrate_positional_bucket(cubes_dir, name, adopt=adopt) or {}
    recorded = recorded_fingerprints(manifest)
    labels = manifest_member_labels(manifest)
    keep = [label for label in labels
            if label in current and recorded[label] == current[label]]
    result.dropped = [label for label in labels if label not in keep]
    keys = {member_cube_key(label) for label in keep}
    listed = {int(i) for i in manifest.get("indices", []) or []}
    for file_name in os.listdir(cubes_dir):
        field_match = _FIELD_FILE.fullmatch(file_name)
        if field_match is None:
            continue
        member = _LABEL_KEYED_FILE.fullmatch(file_name)
        if ((member is not None and member.group(1) not in keys)
                or _POSITIONAL_FILE.fullmatch(file_name)
                or (name == BLACKOUT_INDEX and int(field_match.group(1)) not in listed)
                or (result.dropped and not _MEMBERSHIP_FREE_FILE.fullmatch(file_name))):
            _remove_quietly(os.path.join(cubes_dir, file_name))
    if result.dropped:
        manifest = {**manifest, "member_labels": keep,
                    MEMBER_FPS_KEY: {label: recorded[label] for label in keep}}
        for key in [k for k in manifest if k.startswith("has_combiner")]:
            manifest[key] = False
        for key in ("pca_amps", "pca_var"):
            if key in manifest:
                manifest[key] = {}
        write_bucket_manifest(cubes_dir, manifest, name)
    after = tree_usage(cubes_dir)
    result.bytes_freed = max(0, before[0] - after[0])
    result.files_deleted = max(0, before[1] - after[1])
    return result
```

- [ ] **Step 4: Run the tests** — `pytest tests/test_ensemble_cube_cache_purge.py tests/test_ensemble_cube_cache*.py -q` → all pass.

- [ ] **Step 5: Commit** — `git add euclid_polish/eval/ensemble_cube_cache.py tests/test_ensemble_cube_cache_purge.py && git commit -m "Purge a cube bucket of what its next writer would discard"`

---

### Task 2: Blackout builder drops unlisted fields

**Files:** Modify `euclid_polish/eval/spatial_gate_fit.py` (`build_blackout_fields`); Test `tests/test_spatial_gate_fit_blackout_cache.py`

- [ ] **Step 1: Failing test** (append; uses the file's `_Runner`, `_build`, `stamped` helpers):

```python
def test_a_leftover_field_outside_the_manifest_is_not_reused(tmp_path, stamped):
    runner = _Runner({"a": "fp-a"})
    out = tmp_path / "blackout"
    _build(out, ["a"], runner, max_fields=2)
    np.save(out / "lr_00002.npy", np.full((4, 4, 4), -99.0, np.float32))
    np.save(member_cube_path(str(out), "a", 2), np.full((8, 8, 4), -99.0, np.float32))
    runner.calls.clear()

    fields = _build(out, ["a"], runner, max_fields=3)

    assert runner.ran() == ["a"]                      # field 2 inferred afresh
    assert float(np.load(out / "lr_00002.npy").min()) >= 0.0
    assert float(fields[-1].lr_e.min()) >= 0.0
```

- [ ] **Step 2:** `pytest tests/test_spatial_gate_fit_blackout_cache.py -q` → the new test fails (runner did not run; −99 reused).

- [ ] **Step 3: Implement** — import `prune_bucket_fields` from `ensemble_cube_cache` in `spatial_gate_fit.py`; in `build_blackout_fields`, right after `done = {int(i) for i in manifest.get("indices", [])}` add:

```python
    # A field outside the manifest is a leftover nothing ties to this
    # stamping (an earlier records set or run): never reuse its files.
    prune_bucket_fields(out_dir, done)
```

- [ ] **Step 4:** the blackout test file passes.
- [ ] **Step 5: Commit** — "Never reuse a blackout field the manifest does not list".

---

### Task 3: Real-field fingerprints and purge

**Files:** Modify `euclid_polish/web/helpers/real_field.py`; Test `tests/test_real_field.py`

- [ ] **Step 1: Failing tests** (append; use the file's `field_env`, `_cubes`, `_tick`, `_Ensemble`, `LABELS`):

```python
def test_the_cache_records_member_fingerprints_and_skips_a_continued_member(field_env, monkeypatch):
    fps = dict.fromkeys(LABELS, "ckpt-1")
    monkeypatch.setattr(real_field, "member_fingerprints",
                        lambda _base, labels: {lb: fps[lb] for lb in labels})
    manifest = real_field.cache_real_field(1.0, 2.0, progress=_tick)
    assert manifest["member_fps"] == fps
    _Ensemble.built.clear()
    fps["172·psnr"] = "ckpt-2"

    real_field.cache_real_field(1.0, 2.0, progress=_tick)

    assert _Ensemble.built == [{"starless": False, "labels": ["172·psnr"]}]


def test_refresh_calls_a_continued_member_stale(field_env, monkeypatch):
    fps = dict.fromkeys(LABELS, "ckpt-1")
    monkeypatch.setattr(real_field, "member_fingerprints",
                        lambda _base, labels: {lb: fps[lb] for lb in labels})
    real_field.cache_real_field(1.0, 2.0, progress=_tick)
    fps["170·psnr"] = "ckpt-2"

    with pytest.raises(RuntimeError, match="member cache is stale"):
        real_field.refresh_real_field_combiners(real_field.field_id(1.0, 2.0), progress=_tick)


def test_purge_drops_stale_members_and_renumbers_the_rest(field_env):
    real_field.cache_real_field(1.0, 2.0, progress=_tick, all_members=True)
    identifier = real_field.field_id(1.0, 2.0)
    kept = np.load(_cubes() / "member2_001.npy")

    out = real_field.purge_stale_real_fields({"170·psnr": None, "172·psnr": None})

    assert [r["dropped"] for r in out] == [["171·psnr"]] and out[0]["bytes_freed"] > 0
    manifest = real_field._read_manifest(identifier)
    assert manifest["member_labels"] == ["170·psnr", "172·psnr"]
    assert manifest["run_members"] == [0, 1] and manifest["combiner_kinds"] == []
    np.testing.assert_array_equal(np.load(_cubes() / "member1_001.npy"), kept)
    names = {p.name for p in _cubes().glob("*.npy")}
    assert not any(n.startswith(("sr_", "std_", "pca", "member2_")) for n in names)
    assert {f"lr_{t:03d}.npy" for t in range(4)} <= names
    assert real_field.purge_stale_real_fields({"170·psnr": None, "172·psnr": None}) == []


def test_purge_drops_every_member_of_a_cache_without_fingerprints(field_env):
    real_field.cache_real_field(1.0, 2.0, progress=_tick)
    path = real_field.manifest_path(real_field.field_id(1.0, 2.0))
    manifest = json.loads(path.read_text())
    del manifest["member_fps"]
    path.write_text(json.dumps(manifest))

    out = real_field.purge_stale_real_fields(dict.fromkeys(LABELS))

    assert out[0]["dropped"] == LABELS
    assert sorted(p.name[:3] for p in _cubes().glob("*.npy")) == ["lr_"] * 4
```

- [ ] **Step 2:** `pytest tests/test_real_field.py -q` → new tests fail.

- [ ] **Step 3: Implement** in `real_field.py`:

Imports: `from euclid_polish.ensemble import EnsembleModel, default_ensemble_dir, member_fingerprints, pca_field`; `from euclid_polish.eval.ensemble_cube_cache import tree_usage`; add `Mapping` to the `collections.abc` import.

`_preserve_matching_member_cubes` gains `reusable: Callable[[str], bool] | None = None` (keyword-only, after `count`); inside the loop, `if old_index is None or (reusable is not None and not reusable(str(label))): continue`. Docstring: "``reusable`` (default: every matching label) narrows it."

In `cache_real_field`, replace the `if old_labels != labels:` block's condition and call:

```python
    fingerprints = member_fingerprints(default_ensemble_dir(), labels)
    old_fps = old.get("member_fps")

    def reusable(label: str) -> bool:
        # A cache without fingerprints cannot prove which checkpoint made it.
        return (isinstance(old_fps, dict) and label in old_fps
                and old_fps[label] == fingerprints.get(label))

    if old_labels != labels or not all(reusable(str(label)) for label in old_labels):
        # Preserve member cubes whose label AND checkpoint still match,
        # remapping their positional indices through hard links. All
        # aggregate/PCA/combiner products are membership-dependent and are
        # rebuilt below.
        staging, preserved = _preserve_matching_member_cubes(
            cubes,
            [str(label) for label in old_labels],
            labels,
            count=int(old.get("count", GRID_SIDE * GRID_SIDE)),
            reusable=reusable,
        )
```

and add `"member_fps": {label: fingerprints.get(label) for label in labels},` to the manifest dict.

In `refresh_real_field_combiners`, after the `labels != active_labels` check:

```python
    recorded_fps = manifest.get("member_fps")
    current_fps = member_fingerprints(default_ensemble_dir(), labels)
    if not isinstance(recorded_fps, dict) or any(
            recorded_fps.get(label) != current_fps.get(label) for label in labels):
        raise RuntimeError(
            "real-field member cache is stale (a member's checkpoint changed); "
            "run the full field cache once")
```

New functions after `refresh_real_field_combiners`:

```python
def purge_stale_real_field(identifier: str,
                           current: Mapping[str, str | None]) -> dict[str, Any] | None:
    """Drop the cubes of the members a field's cache can no longer reuse.

    ``current`` maps the ACTIVE STARFULL labels to their checkpoint
    fingerprints now. A member that is not active, has no recorded
    fingerprint or was continued loses its cubes; the kept members are
    renumbered (the hard-link remap :func:`cache_real_field` uses) and every
    membership-dependent product (``sr_``, ``std_``, ``pcaN_``, combiner
    outputs) goes. ``lr_`` tiles, ``raw/`` and the stack stay, so the viewer
    shows the LR until the next refresh re-infers the dropped members.
    ``None`` when nothing was stale."""
    manifest = _read_manifest(identifier)
    if manifest is None:
        return None
    labels = [str(label) for label in manifest.get("member_labels", []) or []]
    recorded = manifest.get("member_fps")
    recorded = recorded if isinstance(recorded, dict) else {}
    keep = [label for label in labels
            if label in current and label in recorded and recorded[label] == current[label]]
    dropped = [label for label in labels if label not in keep]
    if not dropped:
        return None
    cubes = field_dir(identifier) / "cubes"
    before = tree_usage(str(cubes))
    staging, _preserved = _preserve_matching_member_cubes(
        cubes, labels, keep, count=int(manifest.get("count", GRID_SIDE * GRID_SIDE)))
    for path in cubes.glob("*.npy"):
        if not path.name.startswith("lr_"):
            path.unlink(missing_ok=True)
    _restore_matching_member_cubes(cubes, staging)
    old_run = [int(i) for i in manifest.get("run_members", range(len(labels)))]
    run = [keep.index(labels[i]) for i in old_run if i < len(labels) and labels[i] in keep]
    manifest.update({
        "member_labels": keep, "member_fps": {label: recorded[label] for label in keep},
        "run_members": run, "run_member_labels": [keep[i] for i in run],
        "combiner_kinds": [], "combiner_state": {}, "pca_amps": {}, "pca_var": {},
        "pca_n": 0,
    })
    _write_json(manifest_path(identifier), manifest)
    after = tree_usage(str(cubes))
    return {"field_id": identifier, "dropped": dropped,
            "bytes_freed": max(0, before[0] - after[0]),
            "files_deleted": max(0, before[1] - after[1])}


def purge_stale_real_fields(current: Mapping[str, str | None]) -> list[dict[str, Any]]:
    """:func:`purge_stale_real_field` over every cached field (what changed)."""
    root = fields_root()
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.iterdir()):
        if path.is_dir():
            purged = purge_stale_real_field(path.name, current)
            if purged is not None:
                out.append(purged)
    return out
```

- [ ] **Step 4:** `pytest tests/test_real_field.py tests/test_viewer_backend.py tests/test_real_tiles.py -q` → pass.
- [ ] **Step 5: Commit** — "Key real-field member cubes by checkpoint and purge stale ones".

---

### Task 4: Experiments member-SR cache purge

**Files:** Modify `euclid_polish/web/helpers/experiments.py`; Test `tests/test_experiments.py`

- [ ] **Step 1: Failing test** (append):

```python
def test_purge_removes_member_sr_entries_made_by_another_checkpoint(tmp_path, monkeypatch):
    from euclid_polish.web.helpers import experiments, model_catalog
    root = tmp_path / "cache"
    monkeypatch.setattr(model_catalog, "member_cache_root", lambda: root)
    current = {"member_170": "fp-170", "member_171": None}       # 171 archived
    monkeypatch.setattr(experiments, "member_fingerprint",
                        lambda path: current.get(os.path.basename(path)))
    tile = root / "tile" / "t1"
    tile.mkdir(parents=True)
    for slug, label, fp in (("member_170", "170·psnr", "fp-170"),
                            ("member_171", "171·psnr", "fp-171"),
                            ("member_172", "172·psnr", "fp-old"),
                            ("single", "wdsr", "x")):
        np.save(tile / f"{slug}.npy", np.zeros(4, np.float32))
        (tile / f"{slug}.json").write_text(json.dumps(
            {"label": label, "member_fingerprint": fp}))

    out = experiments.purge_stale_member_sr_cache(str(tmp_path / "ensemble"))

    assert out["members"] == ["171·psnr", "172·psnr"] and out["files_deleted"] == 4
    assert sorted(p.name for p in tile.iterdir()) == [
        "member_170.json", "member_170.npy", "single.json", "single.npy"]
```

(Check the file's imports: add `import json`, `import os`, `import numpy as np` at the top if missing — module scope.)

- [ ] **Step 2:** fails (no `purge_stale_member_sr_cache`).

- [ ] **Step 3: Implement** — `from euclid_polish.ensemble import member_fingerprint` at the top of `experiments.py`; next to `_label_slug`:

```python
#: Labels of ensemble members (``170·psnr``) — the only entries a purge judges.
_MEMBER_LABEL = re.compile(r"\d+·psnr")


def purge_stale_member_sr_cache(base_dir: str | None = None) -> dict[str, Any]:
    """Delete member-SR cache entries their member can no longer reuse: made
    by another checkpoint than the member's current one (an archived member
    has none). Entries of other models are kept. ``base_dir``: the ensemble
    (default the canonical one)."""
    base = base_dir or model_catalog.ensemble_dir()
    root = model_catalog.member_cache_root()
    freed = deleted = 0
    members: set[str] = set()
    current: dict[str, str | None] = {}
    for meta_path in sorted(root.rglob("*.json")) if root.is_dir() else []:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        label = str(meta.get("label", "")) if isinstance(meta, dict) else ""
        if not _MEMBER_LABEL.fullmatch(label):
            continue
        if label not in current:
            current[label] = member_fingerprint(
                os.path.join(base, model_catalog.member_name(label)))
        recorded = meta.get("member_fingerprint")
        if recorded is not None and recorded == current[label]:
            continue
        for item in (meta_path.with_suffix(".npy"), meta_path):
            with contextlib.suppress(OSError):
                size = item.stat().st_size
                item.unlink()
                freed += size
                deleted += 1
        members.add(label)
    return {"bytes_freed": freed, "files_deleted": deleted, "members": sorted(members)}
```

- [ ] **Step 4:** `pytest tests/test_experiments.py -q` → pass.
- [ ] **Step 5: Commit** — "Purge member-SR cache entries of stale checkpoints".

---

### Task 5: Pending-purge flag

**Files:** Create `euclid_polish/web/helpers/purge_requests.py`; Test `tests/test_stale_purge.py`

- [ ] **Step 1: Failing tests** (new file, first part):

```python
"""The automatic stale-cube purge: the durable request flag, the idle-time
job the registry hook starts, and the end-to-end effect on the member-cube
buckets (an Evaluate after a purge infers only what the purge removed)."""
from __future__ import annotations

import threading
import time

import pytest

from euclid_polish.web.helpers import purge_requests


def test_requests_accumulate_and_clear_only_with_their_token():
    assert purge_requests.read_pending() is None
    purge_requests.request_stale_purge("archived member_03")
    first = purge_requests.read_pending()
    purge_requests.request_stale_purge("pulled member_04")
    second = purge_requests.read_pending()

    assert second["reasons"] == ["archived member_03", "pulled member_04"]
    assert purge_requests.clear_pending(first["requested_at"]) is False   # newer request
    assert purge_requests.clear_pending(second["requested_at"]) is True
    assert purge_requests.read_pending() is None
```

(`Config.VIS_DIR` is redirected to a per-test tmp dir by `tests/conftest.py`, so the flag file is isolated.)

- [ ] **Step 2:** fails (module missing).

- [ ] **Step 3: Implement** `purge_requests.py`:

```python
"""Durable "a stale-cube purge is due" flag.

Jobs that make cached cubes stale (archiving or pulling members, the FASRC
ensemble mirror, a records sync) call :func:`request_stale_purge`; the job
registry's finish hook (:mod:`euclid_polish.web.helpers.stale_purge`) starts
the purge once no job runs. A file under ``<vis>/ensemble`` so a request
survives a console restart. Config-only imports: the modules that request a
purge must not import the purge itself (an import cycle via ensemble_viz)."""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from typing import Any

from euclid_polish.config import Config

PENDING_NAME = "stale_purge_pending.json"
_MAX_REASONS = 20
_LOCK = threading.Lock()


def pending_path() -> str:
    return os.path.join(os.path.abspath(Config.VIS_DIR), "ensemble", PENDING_NAME)


def read_pending() -> dict[str, Any] | None:
    """The pending request (``reasons``, ``requested_at``) or ``None``."""
    try:
        with open(pending_path()) as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def request_stale_purge(reason: str) -> None:
    """Record that cached cubes may have gone stale (``reason``: what
    happened). Each request gets a fresh ``requested_at`` token."""
    with _LOCK:
        reasons = [str(r) for r in (read_pending() or {}).get("reasons", [])]
        payload = {"reasons": [*reasons, str(reason)][-_MAX_REASONS:],
                   "requested_at": time.time_ns()}
        path = pending_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as handle:
            json.dump(payload, handle)
        os.replace(tmp, path)


def clear_pending(token: int | None) -> bool:
    """Remove the request if it is still the one stamped ``token``: one made
    while the purge ran survives it. Whether it was removed."""
    with _LOCK:
        pending = read_pending()
        if pending is None or pending.get("requested_at") != token:
            return False
        with contextlib.suppress(FileNotFoundError):
            os.remove(pending_path())
        return True
```

- [ ] **Step 4:** test passes.
- [ ] **Step 5: Commit** — "Add the durable stale-purge request flag".

---

### Task 6: Job-registry finish hooks

**Files:** Modify `euclid_polish/web/jobs.py`; Test `tests/test_jobs.py`

- [ ] **Step 1: Failing test** (append):

```python
def test_finish_hooks_run_once_the_status_is_final_and_never_fail_the_job():
    registry = JobRegistry()
    seen: list[tuple[str, str]] = []

    def hook(job):
        seen.append((job.label, job.status))

    def broken(_job):
        raise RuntimeError("hook bug")

    registry.add_finish_hook(hook)
    registry.add_finish_hook(hook)                 # added once
    registry.add_finish_hook(broken)
    ok = registry.spawn("ok", lambda _cap: None)
    bad = registry.spawn("bad", lambda _cap: 1 / 0)
    _wait_for_done(registry, ok, bad)
    deadline = time.monotonic() + 2
    while len(seen) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)

    assert sorted(seen) == [("bad", "failed"), ("ok", "done")]
    assert registry.get(ok).status == "done"
```

- [ ] **Step 2:** fails (`add_finish_hook` missing).

- [ ] **Step 3: Implement** — in `JobRegistry.__init__` add `self._finish_hooks: builtins.list[Callable[[Job], None]] = []`; add methods:

```python
    def add_finish_hook(self, hook: Callable[[Job], None]) -> None:
        """Call ``hook(job)`` after every job of this registry ends (done,
        failed or cancelled), in the job's thread once its status is final.
        A hook's exception is printed, never raised."""
        with self._lock:
            if hook not in self._finish_hooks:
                self._finish_hooks.append(hook)

    def _run_finish_hooks(self, job: Job) -> None:
        with self._lock:
            hooks = builtins.list(self._finish_hooks)
        for hook in hooks:
            try:
                hook(job)
            except Exception:  # noqa: BLE001 - a hook must never fail the job
                traceback.print_exc()
```

and in `_start._runner`'s `finally:` call `self._run_finish_hooks(job)` after `self._evict_finished()`.

- [ ] **Step 4:** `pytest tests/test_jobs.py -q` → pass.
- [ ] **Step 5: Commit** — "Let the job registry call hooks when a job ends".

---

### Task 7: The purge orchestrator, job and hook

**Files:** Create `euclid_polish/web/helpers/stale_purge.py`; Test `tests/test_stale_purge.py`

- [ ] **Step 1: Failing tests** (append to `tests/test_stale_purge.py`; add these imports at the top of the file):

```python
import numpy as np

from euclid_polish.config import Config
from euclid_polish import ensemble_registry as er
from euclid_polish.web.helpers import ensemble_viz as ev
from euclid_polish.web.helpers import stale_purge
from euclid_polish.web.jobs import JobRegistry
from tests._ensemble_cube_cache_fixtures import (
    N_FIELDS, Cap, FakeEnsemble, make_env, member_dir, regenerate_records, run_counts,
    set_checkpoint,
)
```

```python
@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "EUCLID_INFERENCE_DIR", str(tmp_path / "inference"))
    return make_env(tmp_path, monkeypatch)


def _evaluate():
    return ev.job_ensemble_evaluate(Cap(), num_images=N_FIELDS, starless=False)


def _member_files(directory, key):
    return sorted(p.name for p in directory.glob(f"member_{key}_*.npy"))


def test_after_a_continued_member_the_purge_leaves_only_that_member_to_infer(env):
    _evaluate()
    set_checkpoint(env["base"], 2, step=2)

    report = stale_purge.purge_stale_caches()

    assert report["bytes_freed"] > 0
    assert [b["dropped"] for b in report["buckets"]] == [["02·psnr"]]
    assert _member_files(env["cubes"], "02") == []
    FakeEnsemble.reset()
    _evaluate()
    assert run_counts() == {"02·psnr": N_FIELDS}
    for rec in range(N_FIELDS):
        stack = np.stack([np.load(env["cubes"] / f"member_{k}_{rec:05d}.npy")
                          for k in ("01", "02", "03")])
        np.testing.assert_allclose(np.load(env["cubes"] / f"sr_{rec:05d}.npy"),
                                   stack.mean(0), rtol=1e-6)


def test_after_an_archive_the_next_evaluate_runs_no_member(env):
    _evaluate()
    er.archive_member_entry(str(env["base"]), "member_03", zip_path="models/x.zip")
    ev._mark_archive_stale(False, "member_03")
    shutil.rmtree(member_dir(env["base"], 3))

    stale_purge.purge_stale_caches()

    assert _member_files(env["cubes"], "03") == []
    FakeEnsemble.reset()
    summary = _evaluate()
    assert run_counts() == {}
    assert summary["member_labels"] == ["01·psnr", "02·psnr"]


def test_regenerated_records_wipe_the_bucket(env):
    _evaluate()
    regenerate_records(env["records"], "test")

    report = stale_purge.purge_stale_caches()

    assert report["buckets"][0]["wiped"] == "made from other records"
    assert not env["cubes"].exists()


def test_an_unreadable_ensemble_purges_nothing(env):
    _evaluate()
    shutil.rmtree(env["base"])

    report = stale_purge.purge_stale_caches()

    assert report["bytes_freed"] == 0 and env["cubes"].is_dir()


def test_the_hook_starts_the_purge_only_when_the_registry_is_idle(monkeypatch):
    registry = JobRegistry()
    ran = threading.Event()
    monkeypatch.setattr(stale_purge, "purge_stale_caches",
                        lambda progress=None: (ran.set(), _empty_report())[1])
    registry.add_finish_hook(lambda job: stale_purge.on_job_finished(job, registry))
    release = threading.Event()
    blocker = registry.spawn("blocker", lambda _cap: release.wait(2))
    purge_requests.request_stale_purge("archived member_03")
    registry.spawn("archive", lambda _cap: None)
    time.sleep(0.1)
    assert not ran.is_set()                          # the blocker still runs

    release.set()
    assert ran.wait(2)
    _wait_idle(registry)
    assert purge_requests.read_pending() is None
    assert [j["label"] for j in registry.list() if j["kind"] == stale_purge.JOB_KIND] \
        == [stale_purge.JOB_LABEL]


def test_a_failing_purge_does_not_restart_itself(monkeypatch):
    registry = JobRegistry()
    calls = []

    def failing(progress=None):
        calls.append(1)
        raise RuntimeError("disk error")

    monkeypatch.setattr(stale_purge, "purge_stale_caches", failing)
    registry.add_finish_hook(lambda job: stale_purge.on_job_finished(job, registry))
    purge_requests.request_stale_purge("pulled member_04")
    registry.spawn("pull", lambda _cap: None)
    _wait_idle(registry)
    time.sleep(0.1)

    assert calls == [1]
    assert purge_requests.read_pending() is not None   # retried after the next job


def _empty_report():
    return {"buckets": [], "real_fields": [], "bytes_freed": 0, "files_deleted": 0,
            "member_sr_cache": {"bytes_freed": 0, "files_deleted": 0, "members": []}}


def _wait_idle(registry):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        time.sleep(0.01)
        if not any(j["status"] == "running" for j in registry.list(summary=True)):
            return
    raise AssertionError("registry did not go idle")
```

(Add `import shutil` at the top.)

- [ ] **Step 2:** fails (module missing).

- [ ] **Step 3: Implement** `stale_purge.py`:

```python
"""Automatic purge of cached cubes no reader can reuse.

Jobs that make cubes stale request a purge
(:func:`euclid_polish.web.helpers.purge_requests.request_stale_purge`); the
job registry's finish hook (:func:`on_job_finished`) starts one job of kind
:data:`JOB_KIND` once no job runs, so a running Evaluate, fit or compare never
loses cubes it reads. The purge deletes exactly what each cache's next writer
would discard (:func:`~euclid_polish.eval.ensemble_cube_cache.purge_stale_bucket`,
:func:`~euclid_polish.web.helpers.real_field.purge_stale_real_fields`,
:func:`~euclid_polish.web.helpers.experiments.purge_stale_member_sr_cache`).
Spec: docs/superpowers/specs/2026-10-06-stale-cube-purge-design.md."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from euclid_polish.ensemble import member_fingerprints
from euclid_polish.ensemble_registry import default_ensemble_dir
from euclid_polish.eval.ensemble_cube_cache import (
    BLACKOUT_INDEX,
    VIZ_INDEX,
    is_label_keyed,
    purge_stale_bucket,
    read_bucket_manifest,
)
from euclid_polish.eval.subsets import eval_subset
from euclid_polish.web.helpers import ensemble_viz, experiments, real_field
from euclid_polish.web.helpers.purge_requests import clear_pending, read_pending
from euclid_polish.web.jobs import REGISTRY, Job, JobRegistry

JOB_KIND = "storage-purge-stale"
JOB_LABEL = "Storage: delete stale cubes"
#: ``(bucket dir, manifest name, records subset)``; ``"eval"`` is the
#: regime's evaluation subset (the blackout copies of the test cubes too).
_BUCKETS = (("cubes", VIZ_INDEX, "eval"),
            ("cubes_validate", VIZ_INDEX, "validate"),
            ("cubes_blackout", BLACKOUT_INDEX, "eval"),
            ("cubes_validate_blackout", BLACKOUT_INDEX, "validate"))

Progress = Callable[[int, int, str], None]


def _records_fps(records_dir: str | None, starless: bool) -> dict[str, str | None]:
    """The records fingerprints the bucket writers use now (``None``: unknown)."""
    if not records_dir:
        return {"eval": None, "validate": None}
    return {"eval": ensemble_viz._eval_records_fingerprint(
                records_dir, eval_subset(records_dir), starless=starless),
            "validate": ensemble_viz._eval_records_fingerprint(
                records_dir, "validate", starless=starless)}


def _test_adoption(starless: bool, cubes_dir: str) -> dict[str, str | None] | None:
    """The fingerprints a positional test bucket's migration would adopt."""
    manifest = read_bucket_manifest(cubes_dir)
    if manifest is None or is_label_keyed(manifest):
        return None
    return ensemble_viz._proven_test_fingerprints(starless, manifest)


def purge_stale_caches(progress: Progress | None = None) -> dict[str, Any]:
    """Delete every cached cube no reader can reuse; what went, per cache.

    A regime without active members is skipped (an unreadable ensemble dir
    must not read as "every member departed")."""
    base = default_ensemble_dir()
    records_dir = ensemble_viz._sky_records_local_dir()
    steps = 2 * len(_BUCKETS) + 2
    buckets: list[dict[str, Any]] = []
    for regime, starless in enumerate((False, True)):
        labels = ensemble_viz._regime_labels(base, starless)
        current = member_fingerprints(base, labels)
        fps = _records_fps(records_dir, starless)
        regime_dir = ensemble_viz._regime_dir_ro(starless)
        slug = ensemble_viz._regime_slug(starless)
        for position, (dir_name, manifest_name, subset) in enumerate(_BUCKETS):
            if progress is not None:
                progress(regime * len(_BUCKETS) + position, steps, f"{slug}/{dir_name}")
            if not labels:
                continue
            cubes_dir = os.path.join(regime_dir, dir_name)
            result = purge_stale_bucket(
                cubes_dir, name=manifest_name, current=current, records_fp=fps[subset],
                adopt=_test_adoption(starless, cubes_dir) if dir_name == "cubes" else None)
            if result.files_deleted or result.dropped or result.wiped:
                buckets.append({"bucket": f"{slug}/{dir_name}", "wiped": result.wiped,
                                "dropped": result.dropped,
                                "bytes_freed": result.bytes_freed,
                                "files_deleted": result.files_deleted})
    if progress is not None:
        progress(steps - 2, steps, "real fields")
    starfull = ensemble_viz._regime_labels(base, False)
    fields = (real_field.purge_stale_real_fields(member_fingerprints(base, starfull))
              if starfull else [])
    if progress is not None:
        progress(steps - 1, steps, "experiments member-SR cache")
    member_sr = (experiments.purge_stale_member_sr_cache(base) if starfull
                 else {"bytes_freed": 0, "files_deleted": 0, "members": []})
    if progress is not None:
        progress(steps, steps, "done")
    parts = [*buckets, *fields, member_sr]
    return {"buckets": buckets, "real_fields": fields, "member_sr_cache": member_sr,
            "bytes_freed": sum(int(p["bytes_freed"]) for p in parts),
            "files_deleted": sum(int(p["files_deleted"]) for p in parts)}


def _size(nbytes: int) -> str:
    return f"{nbytes / 1e9:.2f} GB" if nbytes >= 1e8 else f"{nbytes / 1e6:.1f} MB"


def job_stale_purge(cap) -> dict[str, Any]:
    """The purge as a job: log every deletion, then clear the request it served."""
    token = (read_pending() or {}).get("requested_at")
    report = purge_stale_caches(progress=cap.tick)
    for bucket in report["buckets"]:
        what = bucket["wiped"] or (
            f"dropped {', '.join(bucket['dropped'])}" if bucket["dropped"]
            else "leftover files")
        print(f"  {bucket['bucket']}: {what} — {bucket['files_deleted']} files, "
              f"{_size(bucket['bytes_freed'])}")
    for field in report["real_fields"]:
        print(f"  real field {field['field_id']}: dropped {', '.join(field['dropped'])} — "
              f"{field['files_deleted']} files, {_size(field['bytes_freed'])}")
    member_sr = report["member_sr_cache"]
    if member_sr["files_deleted"]:
        print(f"  experiments member-SR cache: {', '.join(member_sr['members'])} — "
              f"{member_sr['files_deleted']} files, {_size(member_sr['bytes_freed'])}")
    print(f"freed {_size(report['bytes_freed'])} in {report['files_deleted']} files"
          if report["files_deleted"] else "nothing stale")
    clear_pending(token)
    return report


def maybe_start_stale_purge(registry: JobRegistry | None = None) -> str | None:
    """Start the purge job when one is requested and no job runs; its id."""
    registry = registry or REGISTRY
    if read_pending() is None:
        return None
    if any(job.get("status") == "running" for job in registry.list(summary=True)):
        return None
    job, started = registry.spawn_exclusive(JOB_LABEL, job_stale_purge, kind=JOB_KIND)
    return job.job_id if started else None


def on_job_finished(job: Job, registry: JobRegistry | None = None) -> None:
    """Registry finish hook. Never reacts to the purge's own job, so a failing
    purge retries after the next other job instead of in a loop."""
    if job.kind != JOB_KIND:
        maybe_start_stale_purge(registry)
```

- [ ] **Step 4:** `pytest tests/test_stale_purge.py -q` → pass.
- [ ] **Step 5: Commit** — "Run the stale-cube purge as a job when the console is idle".

---

### Task 8: Wire the triggers and the hook

**Files:** Modify `ensemble_viz.py` (`job_archive_member`, `job_ensemble_pull`), `routes/fasrc.py` (mirror `run`), `routes/views.py` (`_job_sky_sync`), `web/app.py` (`start_background_services`); Test: append to `tests/test_stale_purge.py`

- [ ] **Step 1: Failing test** — the sky sync and pull paths are exercised through their existing test helpers; the minimal direct check:

```python
def test_records_sync_requests_a_purge(monkeypatch):
    from euclid_polish.web.routes import views

    class _Result:
        ok, size_bytes, error = True, 10, None

    monkeypatch.setattr(views._fasrc_fetcher, "fetch_one_file", lambda *a, **k: _Result())
    monkeypatch.setattr(views, "_pull_generation_sidecars", lambda *a, **k: None)

    views._job_sky_sync(Cap(), {"dirty_test": "/remote/dirty_test.tfrecord"}, ["test"])

    assert purge_requests.read_pending()["reasons"] == ["synced records dirty_test"]
```

and add to the pull/archive tests (`tests/test_ensemble_pull_skip.py`, `tests/test_ensemble_archive.py`) one assertion each after their successful run: `assert purge_requests.read_pending() is not None` (and for a pull that changed nothing / a dry run: `is None`).

- [ ] **Step 2:** fails.

- [ ] **Step 3: Implement**

`ensemble_viz.py` — import `from euclid_polish.web.helpers.purge_requests import request_stale_purge`; at the end of `job_archive_member` (before `return`): `request_stale_purge(f"archived {name}")`; in `job_ensemble_pull` after the orphan-checkpoint sweep: `if changed: request_stale_purge(f"pulled {', '.join(sorted(changed))}")`. Update `job_archive_member`'s docstring: the cubes now go in the follow-up purge job (no inference is needed afterwards).

`routes/fasrc.py` mirror `run(cap)`: after the `last_rc` check, `request_stale_purge("mirrored the FASRC ensemble")` (import at top).

`routes/views.py` `_job_sky_sync`: before `cap.tick(total, total, "done")`:

```python
    pulled = [key for key, entry in results.items() if entry.get("ok")]
    if pulled:
        request_stale_purge(f"synced records {', '.join(pulled)}")
```

`web/app.py`: import `from euclid_polish.web.helpers import provenance_index, sky_atlas, stale_purge` and `from euclid_polish.web.jobs import REGISTRY as JOB_REGISTRY`; in `start_background_services` add `JOB_REGISTRY.add_finish_hook(stale_purge.on_job_finished)` and mention it in the docstring.

- [ ] **Step 4:** `pytest tests/test_stale_purge.py tests/test_ensemble_archive.py tests/test_ensemble_pull_skip.py tests/test_web.py -q` → pass.
- [ ] **Step 5: Commit** — "Request a stale-cube purge after archive, pull, mirror and records sync".

---

### Task 9: Docs, full suite, push

- [ ] `euclid_polish/web/API.md`: under the jobs section, document kind `storage-purge-stale` (started by the console when idle after archive/pull/mirror/records sync; result `{buckets, real_fields, member_sr_cache, bytes_freed, files_deleted}`).
- [ ] README storage/caches paragraph (if one exists): one sentence on the automatic purge.
- [ ] `conda run -n EuclidPolishEnv ruff check euclid_polish tests` → clean.
- [ ] `conda run -n EuclidPolishEnv python -m pytest -q -x -n auto` (or without `-n` if xdist is absent) → all pass.
- [ ] Commit, push to `origin main`.
- [ ] Update memory (`project_stale_cube_purge.md` + MEMORY.md line).
