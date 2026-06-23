# Resumable Synthetic Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `scripts/run_pipeline.py` skip already-complete data subsets on a resubmitted job, so a job killed after `train` finishes only regenerates `validate`.

**Architecture:** Subset-level resume driven by a record-count completeness check (no marker files). Final TFRecords are "complete" when their example count equals the requested `n`; the source sidecar is made atomic so its mere existence is a sound signal. A resubmit auto-resumes; `--force` regenerates from scratch.

**Tech Stack:** Python, TensorFlow (`tf.data.TFRecordDataset`), pytest. All changes in `scripts/run_pipeline.py` and `euclid_polish/sky/source_catalog.py`.

**Spec:** `docs/superpowers/specs/2026-06-23-resumable-synthetic-generation-design.md`

---

## File structure

- **Modify** `euclid_polish/sky/source_catalog.py` — make `concat_source_csvs` atomic (temp + `os.replace`).
- **Modify** `scripts/run_pipeline.py` — add `glob`/`csv` imports, four completeness/cleanup helpers, the `--force` flag, and the per-subset skip guards in all three step functions.
- **Create** `tests/test_run_pipeline_resume.py` — unit tests for the helpers + a serial-convolve resume/force integration test.
- **Modify** `tests/test_source_catalog_atomic.py` (new) — atomic + sparse merge test for `concat_source_csvs`.

---

## Task 1: Atomic `concat_source_csvs`

**Files:**
- Modify: `euclid_polish/sky/source_catalog.py:102-113`
- Test: `tests/test_source_catalog_atomic.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_source_catalog_atomic.py`:

```python
"""concat_source_csvs merges shard sidecars atomically and tolerates
sparse shards (a field that rendered no galaxies/lenses writes no row)."""

import os

from euclid_polish.sky.source_catalog import SOURCE_COLS, concat_source_csvs


def _write_part(path, field_indices):
    with open(path, "w", newline="") as f:
        f.write(",".join(SOURCE_COLS) + "\r\n")
        for fi in field_indices:
            row = {c: "" for c in SOURCE_COLS}
            row["field_index"] = str(fi)
            row["type"] = "galaxy"
            f.write(",".join(row[c] for c in SOURCE_COLS) + "\r\n")


def test_concat_merges_in_order_with_single_header(tmp_path):
    p0 = str(tmp_path / "sources_train.part0000.csv")
    p1 = str(tmp_path / "sources_train.part0001.csv")
    _write_part(p0, [0, 1])
    _write_part(p1, [])          # sparse shard: header only, no rows
    out = str(tmp_path / "sources_train.csv")

    concat_source_csvs([p0, p1], out)

    lines = [ln for ln in open(out).read().splitlines() if ln]
    assert lines[0] == ",".join(SOURCE_COLS)     # exactly one header
    assert sum(1 for ln in lines[1:]) == 2       # two data rows, sparse part ok
    assert ",".join(SOURCE_COLS) not in lines[1:]


def test_concat_leaves_no_temp_file(tmp_path):
    p0 = str(tmp_path / "sources_train.part0000.csv")
    _write_part(p0, [0])
    out = str(tmp_path / "sources_train.csv")

    concat_source_csvs([p0], out)

    leftovers = [n for n in os.listdir(tmp_path)
                 if n.startswith("sources_train.csv") and n != "sources_train.csv"]
    assert leftovers == []                       # temp file replaced, not left
    assert os.path.exists(out)
```

- [ ] **Step 2: Run the test to verify the temp-file test fails**

Run: `pytest tests/test_source_catalog_atomic.py -v`
Expected: `test_concat_merges_in_order_with_single_header` PASSES (current behavior already merges), `test_concat_leaves_no_temp_file` PASSES too — current code writes straight to `out` with no temp. So this task is REFACTOR-toward-atomicity, not behavior change. Confirm both pass first to lock current behavior.

- [ ] **Step 3: Make the merge atomic**

Replace `euclid_polish/sky/source_catalog.py:102-113` with:

```python
def concat_source_csvs(part_paths: List[str], out_path: str) -> None:
    """Concatenate shard CSVs (in the given order) into one, single header.

    Atomic: build a sibling temp file then ``os.replace`` it into place, so a
    crash mid-merge never leaves a truncated ``sources_<subset>.csv`` that a
    resumed run would mistake for complete (see the resume design doc)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", newline="") as out:
        out.write(",".join(SOURCE_COLS) + "\r\n")
        for p in part_paths:
            if not os.path.isfile(p):
                continue
            with open(p, newline="") as f:
                next(f, None)                     # skip shard header
                for line in f:
                    out.write(line)
    os.replace(tmp_path, out_path)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_source_catalog_atomic.py -v`
Expected: both PASS (temp file `.tmp` is replaced, so no leftover).

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/sky/source_catalog.py tests/test_source_catalog_atomic.py
git commit -m "source_catalog: make concat_source_csvs atomic (temp + os.replace)"
```

---

## Task 2: `_count_tfrecords` helper

**Files:**
- Modify: `scripts/run_pipeline.py` (add `glob`, `csv` imports near line 20-22; add helper after `_shard_bounds` at `:341`)
- Test: `tests/test_run_pipeline_resume.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_pipeline_resume.py`:

```python
"""Resume logic in scripts/run_pipeline.py: a subset is 'complete' when its
final TFRecords have the requested record count and the sidecar exists."""

from __future__ import annotations

import importlib.util
import os

import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.euclid.psf_library import load_all_band_psfs
from euclid_polish.sky.multiband_forward import (
    MultiBandForward, MultiBandForwardConfig,
)
from euclid_polish.sky.multiband_generator import (
    MultiBandGeneratorConfig, MultiBandSimulator,
)
from euclid_polish.sky.tfrecord import tfrecord_path
from tests._tiny_catalog import TinyCosmosCatalog


def _load_run_pipeline():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "run_pipeline.py",
    )
    spec = importlib.util.spec_from_file_location("run_pipeline_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rp = _load_run_pipeline()


def _write_dummy_tfrecord(path: str, n: int) -> None:
    with tf.io.TFRecordWriter(path) as w:
        for i in range(n):
            w.write(f"rec{i}".encode())


def test_count_tfrecords_counts_records(tmp_path):
    p = str(tmp_path / "x.tfrecord")
    _write_dummy_tfrecord(p, 5)
    assert rp._count_tfrecords(p) == 5


def test_count_tfrecords_missing_returns_none(tmp_path):
    assert rp._count_tfrecords(str(tmp_path / "nope.tfrecord")) is None


def test_count_tfrecords_truncated_returns_none(tmp_path):
    p = str(tmp_path / "trunc.tfrecord")
    _write_dummy_tfrecord(p, 3)
    with open(p, "r+b") as f:          # chop the last 4 bytes → DataLossError
        f.truncate(os.path.getsize(p) - 4)
    assert rp._count_tfrecords(p) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_run_pipeline_resume.py -k count_tfrecords -v`
Expected: FAIL with `AttributeError: module 'run_pipeline_mod' has no attribute '_count_tfrecords'`.

- [ ] **Step 3: Add imports and the helper**

In `scripts/run_pipeline.py`, add to the stdlib import block (after `import argparse` at `:20`):

```python
import csv
import glob
```

Then add after `_shard_bounds` (after `:341`):

```python
def _count_tfrecords(path: str) -> int | None:
    """Number of examples in a TFRecord file, or None if it can't be read in
    full. A missing file or a record truncated by a job killed mid-merge
    (``tf.errors.DataLossError``) both return None — i.e. 'not complete'."""
    if not os.path.exists(path):
        return None
    try:
        return sum(1 for _ in tf.data.TFRecordDataset(path))
    except tf.errors.DataLossError:
        return None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_run_pipeline_resume.py -k count_tfrecords -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_pipeline.py tests/test_run_pipeline_resume.py
git commit -m "run_pipeline: add _count_tfrecords (None on missing/truncated)"
```

---

## Task 3: `_sources_complete` helper

**Files:**
- Modify: `scripts/run_pipeline.py` (add helper after `_count_tfrecords`)
- Test: `tests/test_run_pipeline_resume.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_pipeline_resume.py`:

```python
def test_sources_complete_existing_file(tmp_path):
    p = str(tmp_path / "sources_train.csv")
    open(p, "w").write("field_index,type\n0,galaxy\n")
    assert rp._sources_complete(p, expected_n=4) is True


def test_sources_complete_missing_file(tmp_path):
    assert rp._sources_complete(str(tmp_path / "nope.csv"), expected_n=4) is False


def test_sources_complete_zero_expected_is_trivially_true(tmp_path):
    assert rp._sources_complete(str(tmp_path / "nope.csv"), expected_n=0) is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_run_pipeline_resume.py -k sources_complete -v`
Expected: FAIL — `_sources_complete` not defined.

- [ ] **Step 3: Add the helper**

In `scripts/run_pipeline.py`, add immediately after `_count_tfrecords`:

```python
def _sources_complete(csv_path: str, expected_n: int) -> bool:
    """True iff the source sidecar exists (expected_n <= 0 is trivially OK).

    Sidecar rows are sparse — a field that renders no galaxies/lenses writes no
    row — so a field_index-coverage check would false-flag a complete run whose
    last field is empty. ``concat_source_csvs`` is atomic, so the final CSV only
    ever exists in complete form; existence is therefore a sound signal. The
    per-subset TFRecord count check is the authoritative guard."""
    if expected_n <= 0:
        return True
    return os.path.exists(csv_path)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_run_pipeline_resume.py -k sources_complete -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_pipeline.py tests/test_run_pipeline_resume.py
git commit -m "run_pipeline: add _sources_complete (existence-based, atomic-merge backed)"
```

---

## Task 4: `_subset_complete` helper

**Files:**
- Modify: `scripts/run_pipeline.py` (add helper after `_sources_complete`)
- Test: `tests/test_run_pipeline_resume.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_pipeline_resume.py`:

```python
def _make_subset(tmp_path, subset, n, kinds=("clean", "hr", "dirty")):
    for kind in kinds:
        _write_dummy_tfrecord(tfrecord_path(str(tmp_path), f"{kind}_{subset}"), n)
    sidecar = tfrecord_path(str(tmp_path), f"sources_{subset}").replace(
        ".tfrecord", ".csv")
    open(sidecar, "w").write("field_index,type\n0,galaxy\n")


def test_subset_complete_all_kinds_present(tmp_path):
    _make_subset(tmp_path, "train", 4)
    assert rp._subset_complete(
        str(tmp_path), "train", ("clean", "hr", "dirty", "sources"), 4) is True


def test_subset_incomplete_when_kind_short(tmp_path):
    _make_subset(tmp_path, "train", 4)
    _write_dummy_tfrecord(tfrecord_path(str(tmp_path), "hr_train"), 3)  # short
    assert rp._subset_complete(
        str(tmp_path), "train", ("clean", "hr", "dirty", "sources"), 4) is False


def test_subset_incomplete_when_count_mismatch(tmp_path):
    _make_subset(tmp_path, "train", 4)
    assert rp._subset_complete(            # asked for 8, only 4 on disk
        str(tmp_path), "train", ("clean", "hr", "dirty", "sources"), 8) is False


def test_subset_incomplete_when_sidecar_missing(tmp_path):
    _make_subset(tmp_path, "train", 4)
    os.remove(tfrecord_path(str(tmp_path), "sources_train").replace(
        ".tfrecord", ".csv"))
    assert rp._subset_complete(
        str(tmp_path), "train", ("clean", "hr", "dirty", "sources"), 4) is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_run_pipeline_resume.py -k subset_complete -v`
Expected: FAIL — `_subset_complete` not defined.

- [ ] **Step 3: Add the helper**

In `scripts/run_pipeline.py`, add immediately after `_sources_complete`:

```python
def _subset_complete(records_dir: str, subset: str,
                     kinds, expected_n: int) -> bool:
    """True iff every TFRecord ``kind`` for ``subset`` has exactly ``expected_n``
    records and, when 'sources' is requested, the sidecar exists. A count that
    differs from ``expected_n`` (e.g. a resubmit with a different n) is treated
    as incomplete, so the subset is regenerated to match the request."""
    for kind in kinds:
        if kind == "sources":
            csv_path = tfrecord_path(records_dir, f"sources_{subset}").replace(
                ".tfrecord", ".csv")
            if not _sources_complete(csv_path, expected_n):
                return False
        else:
            if _count_tfrecords(tfrecord_path(records_dir, f"{kind}_{subset}")) \
                    != expected_n:
                return False
    return True
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_run_pipeline_resume.py -k subset_complete -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_pipeline.py tests/test_run_pipeline_resume.py
git commit -m "run_pipeline: add _subset_complete (per-kind count + sidecar check)"
```

---

## Task 5: `_cleanup_parts` helper

**Files:**
- Modify: `scripts/run_pipeline.py` (add helper after `_subset_complete`)
- Test: `tests/test_run_pipeline_resume.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_pipeline_resume.py`:

```python
def test_cleanup_parts_removes_only_subset_parts(tmp_path):
    rdir = str(tmp_path)
    # Orphan parts from a dead train run, plus a final file and a validate part.
    for name in ("clean_train.part0000.tfrecord", "hr_train.part0003.tfrecord",
                 "dirty_train.part0001.tfrecord", "sources_train.part0000.csv"):
        open(os.path.join(rdir, name), "w").close()
    open(os.path.join(rdir, "clean_train.tfrecord"), "w").close()      # final
    open(os.path.join(rdir, "clean_validate.part0000.tfrecord"), "w").close()

    rp._cleanup_parts(rdir, "train")

    left = sorted(os.listdir(rdir))
    assert "clean_train.tfrecord" in left                  # final kept
    assert "clean_validate.part0000.tfrecord" in left      # other subset kept
    assert not any(".part" in n and "train" in n for n in left)  # train parts gone
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_run_pipeline_resume.py -k cleanup_parts -v`
Expected: FAIL — `_cleanup_parts` not defined.

- [ ] **Step 3: Add the helper**

In `scripts/run_pipeline.py`, add immediately after `_subset_complete`:

```python
def _cleanup_parts(records_dir: str, subset: str) -> None:
    """Remove leftover per-shard part files for ``subset`` from a prior run.

    Orphan parts survive when a resumed run uses a different shard count (the
    merge only reads the freshly-computed parts list), wasting disk; deleting
    them before regenerating keeps the records dir clean."""
    for kind in ("clean", "hr", "dirty", "sources"):
        for p in glob.glob(os.path.join(records_dir, f"{kind}_{subset}.part*")):
            try:
                os.remove(p)
            except OSError:
                pass
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_run_pipeline_resume.py -k cleanup_parts -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_pipeline.py tests/test_run_pipeline_resume.py
git commit -m "run_pipeline: add _cleanup_parts to clear stale shard files"
```

---

## Task 6: `--force` CLI flag

**Files:**
- Modify: `scripts/run_pipeline.py:166` (next to the `--skip-*` flags)

- [ ] **Step 1: Add the flag**

In `parse_args`, after the line `ap.add_argument("--skip-train", action="store_true")` (`:166`), add:

```python
    ap.add_argument("--force", action="store_true",
                    help="Regenerate every subset from scratch, ignoring "
                         "already-complete data on disk (default: resume — "
                         "skip subsets whose records + sidecar are complete).")
```

- [ ] **Step 2: Verify it parses**

Run: `python scripts/run_pipeline.py --help 2>&1 | grep -A2 -- --force`
Expected: the `--force` help text prints.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_pipeline.py
git commit -m "run_pipeline: add --force flag (opt out of resume)"
```

---

## Task 7: Skip guard in the parallel combined path

**Files:**
- Modify: `scripts/run_pipeline.py:462-469` (`step_generate_and_convolve_parallel`)
- Test: `tests/test_run_pipeline_resume.py`

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_run_pipeline_resume.py`:

```python
def _sim_fwd():
    cat = TinyCosmosCatalog(n_galaxies=200, seed=0)
    sim = MultiBandSimulator(
        cat, MultiBandGeneratorConfig(image_size=96,
                                      pixel_scale=Config.DEFAULT_PIXEL_SCALE))
    psfs = load_all_band_psfs(psf_dir="/nonexistent_dir_for_test")  # Gaussian
    fwd = MultiBandForward(psfs_by_band=psfs,
                           config=MultiBandForwardConfig(add_noise=True))
    return sim, fwd


def _build_complete_subset(rdir, subset, n):
    """Generate one shard covering [0, n) and merge it to final files —
    exactly the on-disk state of a completed subset."""
    sim, fwd = _sim_fwd()
    rp._generate_convolve_range(sim, fwd, rdir, subset, 0, n, 0, seed=[1, 1, 0])
    for kind in ("clean", "hr", "dirty"):
        rp._concat_tfrecords(
            [tfrecord_path(rdir, f"{kind}_{subset}.part0000")],
            tfrecord_path(rdir, f"{kind}_{subset}"))
    rp.concat_source_csvs(
        [tfrecord_path(rdir, f"sources_{subset}.part0000").replace(
            ".tfrecord", ".csv")],
        tfrecord_path(rdir, f"sources_{subset}").replace(".tfrecord", ".csv"))


def test_completed_subset_detected_and_truncation_busts_it(tmp_path):
    rdir = str(tmp_path)
    _build_complete_subset(rdir, "train", 4)
    kinds = ("clean", "hr", "dirty", "sources")
    assert rp._subset_complete(rdir, "train", kinds, 4) is True
    assert rp._subset_complete(rdir, "validate", kinds, 4) is False  # not built

    # A clean TFRecord truncated by a mid-merge kill must NOT read as complete.
    clean = tfrecord_path(rdir, "clean_train")
    with open(clean, "r+b") as f:
        f.truncate(os.path.getsize(clean) - 8)
    assert rp._subset_complete(rdir, "train", kinds, 4) is False
```

Note: `rp.concat_source_csvs` is the module's re-exported name — it is imported into `run_pipeline.py` from `euclid_polish.sky.source_catalog`, so `rp.concat_source_csvs` resolves.

- [ ] **Step 2: Run the test to verify it passes for detection (guard not yet wired)**

Run: `pytest tests/test_run_pipeline_resume.py -k completed_subset_detected -v`
Expected: PASS (this validates the helpers compose; the guard wiring below has no separate unit because exercising the full pool is covered by `test_run_pipeline_parallel.py`).

- [ ] **Step 3: Wire the guard into the parallel path**

In `step_generate_and_convolve_parallel`, find the loop body start (`:462-464`):

```python
    for subset, n in (("train", args.ntrain), ("validate", args.nvalid)):
        if n <= 0:
            continue
```

Insert immediately after the `if n <= 0: continue` block:

```python
        if not args.force and _subset_complete(
                args.records_dir, subset,
                ("clean", "hr", "dirty", "sources"), n):
            _log(f"  {subset}: already complete ({n} records) — skipping")
            continue
        _cleanup_parts(args.records_dir, subset)
```

- [ ] **Step 4: Run the full resume + parallel suites**

Run: `pytest tests/test_run_pipeline_resume.py tests/test_run_pipeline_parallel.py -v`
Expected: all PASS (existing parallel tests unaffected; resume tests pass).

- [ ] **Step 5: Commit**

```bash
git add scripts/run_pipeline.py tests/test_run_pipeline_resume.py
git commit -m "run_pipeline: skip complete subsets in the parallel gen+convolve path"
```

---

## Task 8: Skip guard in serial `step_generate`

**Files:**
- Modify: `scripts/run_pipeline.py:220-226` (`step_generate`)

- [ ] **Step 1: Wire the guard**

In `step_generate`, the loop is (`:220-226`):

```python
    for subset, n in (("train", args.ntrain), ("validate", args.nvalid)):
        # Entropy-seeded master RNG so repeat runs see fresh randomness.
        # The seed is logged so a curious-looking run can be replayed
        # later by hard-coding the printed value here.
        master_seed = int.from_bytes(os.urandom(8), "little")
```

Insert at the very top of the loop body, before `master_seed = ...`:

```python
        if not args.force and _subset_complete(
                args.records_dir, subset, ("clean", "sources"), n):
            done += n
            _log(f"  {subset}: clean already complete ({n} records) — skipping")
            reporter.set_step(done, grand_total, f"{subset} already complete")
            continue
```

- [ ] **Step 2: Verify import-time sanity**

Run: `python -c "import importlib.util,os; s=importlib.util.spec_from_file_location('m','scripts/run_pipeline.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('ok', hasattr(m,'step_generate'))"`
Expected: `ok True`

- [ ] **Step 3: Run the resume suite**

Run: `pytest tests/test_run_pipeline_resume.py -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_pipeline.py
git commit -m "run_pipeline: skip complete subsets in serial step_generate"
```

---

## Task 9: Skip guard in serial `step_convolve` + force test

**Files:**
- Modify: `scripts/run_pipeline.py:280-284` (`step_convolve`)
- Test: `tests/test_run_pipeline_resume.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_pipeline_resume.py`:

```python
class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _convolve_args(rdir, ntrain, nvalid, force):
    return _Args(records_dir=rdir, psf_dir="/nonexistent_dir_for_test",
                 require_empirical_psf=False, ntrain=ntrain, nvalid=nvalid,
                 force=force)


def test_step_convolve_resumes_then_force_regenerates(tmp_path, monkeypatch):
    rdir = str(tmp_path)
    # Build a complete clean_train (4 records) — the input convolve reads.
    sim, fwd = _sim_fwd()
    rp._generate_convolve_range(sim, fwd, rdir, "train", 0, 4, 0, seed=[1, 1, 0])
    rp._concat_tfrecords([tfrecord_path(rdir, "clean_train.part0000")],
                         tfrecord_path(rdir, "clean_train"))
    # Remove hr/dirty so the first convolve must produce them.
    for kind in ("hr", "dirty"):
        os.remove(tfrecord_path(rdir, f"{kind}_train.part0000"))

    opened = []
    real = rp.open_multiband_writer

    def spy(name, **kw):
        opened.append(name)
        return real(name, **kw)

    monkeypatch.setattr(rp, "open_multiband_writer", spy)

    # First run: hr_train + dirty_train get written.
    rp.step_convolve(_convolve_args(rdir, ntrain=4, nvalid=0, force=False))
    assert "hr_train" in opened and "dirty_train" in opened

    # Second run: train is complete now → no writers opened (skipped).
    opened.clear()
    rp.step_convolve(_convolve_args(rdir, ntrain=4, nvalid=0, force=False))
    assert opened == []

    # --force: writers reopen even though train is complete.
    opened.clear()
    rp.step_convolve(_convolve_args(rdir, ntrain=4, nvalid=0, force=True))
    assert "hr_train" in opened and "dirty_train" in opened
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_run_pipeline_resume.py -k step_convolve_resumes -v`
Expected: FAIL on the second-run assertion (`opened == []`) — without the guard, convolve rewrites hr/dirty every run. (It may also fail earlier if `args.force` is read before the flag guard is added — that is expected; the guard line introduces the read.)

- [ ] **Step 3: Wire the guard**

In `step_convolve`, the loop body begins (`:280-284`):

```python
    for subset in ("train", "validate"):
        clean_path = tfrecord_path(args.records_dir, f"clean_{subset}")
        if not os.path.exists(clean_path):
            _log(f"⚠️  {clean_path} not found, skipping {subset}")
            continue
```

Insert immediately after that `if not os.path.exists(clean_path): continue` block:

```python
        n_expected = args.ntrain if subset == "train" else args.nvalid
        if not args.force and _subset_complete(
                args.records_dir, subset, ("hr", "dirty"), n_expected):
            done += counts[subset]
            _log(f"  {subset}: hr+dirty already complete — skipping")
            reporter.set_step(done, grand_total, f"{subset} already complete")
            continue
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_run_pipeline_resume.py -k step_convolve_resumes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_pipeline.py tests/test_run_pipeline_resume.py
git commit -m "run_pipeline: skip complete subsets in serial step_convolve (+force)"
```

---

## Task 10: Full suite + manual smoke check

**Files:** none (verification only)

- [ ] **Step 1: Run the affected test files**

Run: `pytest tests/test_run_pipeline_resume.py tests/test_run_pipeline_parallel.py tests/test_source_catalog_atomic.py -v`
Expected: all PASS.

- [ ] **Step 2: Manual smoke — resume skips a complete subset (tiny, local)**

Run a tiny two-subset generation, then re-run and confirm the WebUI/log shows a skip:

```bash
python scripts/run_pipeline.py --records-dir /tmp/euclid_resume_smoke \
  --ntrain 2 --nvalid 2 --image-size 96 --gen-workers 2 \
  --tng-fraction 1.0 --skip-train
python scripts/run_pipeline.py --records-dir /tmp/euclid_resume_smoke \
  --ntrain 2 --nvalid 2 --image-size 96 --gen-workers 2 \
  --tng-fraction 1.0 --skip-train
```

Expected: the second run logs `train: already complete (2 records) — skipping` and `validate: already complete (2 records) — skipping`, and does NOT re-run the pool.

- [ ] **Step 3: Manual smoke — `--force` regenerates**

Run: append `--force` to the second command above and confirm it regenerates both subsets (no "already complete" lines).

- [ ] **Step 4: Commit any cleanup**

```bash
rm -rf /tmp/euclid_resume_smoke
git status   # should be clean; nothing to commit if no changes
```

---

## Self-review notes

- **Spec coverage:** atomic sources merge (Task 1), `_count_tfrecords` (Task 2), `_sources_complete` (Task 3), `_subset_complete` (Task 4), `_cleanup_parts` (Task 5), `--force` (Task 6), guards in all three paths (Tasks 7–9), progress accounting on skip (Tasks 7–9), truncation/sparse edge cases tested (Tasks 1, 2, 7). All spec sections mapped.
- **Type consistency:** `_subset_complete(records_dir, subset, kinds, expected_n)`, `_sources_complete(csv_path, expected_n)`, `_count_tfrecords(path)`, `_cleanup_parts(records_dir, subset)` — signatures match across all call sites and tests.
- **Note for executor:** `int | None` return annotation is safe — `run_pipeline.py` has `from __future__ import annotations` (`:18`).
