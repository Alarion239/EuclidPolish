# Resumable synthetic data generation

**Date:** 2026-06-23
**Status:** Approved (design)
**Component:** `scripts/run_pipeline.py`

## Problem

A FASRC synthetic-data job generates `train` then `validate`. When the job hits
its SLURM wall-clock limit after `train` is already on disk, resubmitting today
regenerates everything from scratch — wasting the completed `train` work. We want
a resubmitted job to detect already-complete data and skip it, so it only
generates what's left (typically just `validate`).

## Goals

- A resubmitted `run_pipeline.py` invocation skips any subset whose final outputs
  are already complete, and generates only the remaining subset(s).
- Detection is robust against a job killed *mid-merge* (a final file that exists
  but is truncated must NOT be treated as complete).
- Zero changes required to the sbatch / FASRC templates — resume is automatic.

## Non-goals

- Shard-level resume within a partially-generated subset. An interrupted subset
  is regenerated from scratch. (Subset granularity is sufficient: the parallel
  path finishes and merges `train` fully before `validate` begins, so the common
  failure leaves a cleanly-complete `train` and an absent/partial `validate`.)
- A persisted sentinel/marker file. Completeness is derived from the data itself.

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Resume granularity | **Subset-level** — skip a fully-complete subset. |
| Done signal | **Final files exist + record-count check** (no marker files). |
| Trigger | **Automatic / idempotent**; `--force` to regenerate from scratch. |
| Count mismatch (n changed) | **Regenerate** — count `!=` expected ⇒ not complete. |
| Path scope | **All three paths** (parallel combined, serial generate, serial convolve). |

## Background: current pipeline

Three code paths in `scripts/run_pipeline.py`, all looping
`for subset in ("train", "validate")`:

- `step_generate_and_convolve_parallel` (`:447`) — the FASRC default
  (`--gen-workers > 1`). Per subset: fan out shards via `ProcessPoolExecutor`,
  each worker writes `clean/hr/dirty_{subset}.part####.tfrecord` +
  `sources_{subset}.part####.csv`, then the parent merges in shard-id order into
  final `clean/hr/dirty_{subset}.tfrecord` + `sources_{subset}.csv` and deletes
  the parts. `train` completes fully before `validate` starts.
- `step_generate` (`:178`) — serial. Writes `clean_{subset}.tfrecord` +
  `sources_{subset}.csv`.
- `step_convolve` (`:253`) — serial. Reads `clean_{subset}`, writes
  `hr/dirty_{subset}.tfrecord`. Already skips a subset whose `clean_` is missing
  and already counts clean records at `:275`.

Final outputs per subset are addressed via `tfrecord_path(records_dir, name)`;
the sidecar CSV is the same path with `.tfrecord` → `.csv`.

## Design

### Completeness helpers (pure, unit-testable)

```python
def _count_tfrecords(path: str) -> int | None:
    """Number of examples in a TFRecord, or None if missing/corrupt.

    Returns None on a missing file or a tf.errors.DataLossError (a record
    truncated by a job killed mid-merge) — both mean 'not complete'.
    """

def _sources_complete(csv_path: str, expected_n: int) -> bool:
    """True iff the source sidecar exists (expected_n <= 0 is trivially OK).

    The sidecar's rows are SPARSE — a field that renders no galaxies/lenses
    writes no row — so a max(field_index) check would false-flag a complete
    run whose last field is empty. Instead the sources merge is made ATOMIC
    (write temp + os.replace, see concat_source_csvs change below), so the
    final CSV only ever exists in complete form and existence is a sound
    completeness signal. The per-subset TFRecord count check is the
    authoritative guard; this just catches the kill-between-TFRecord-merge-
    and-sources-merge window (final CSV absent).
    """

def _subset_complete(records_dir: str, subset: str,
                     kinds: Sequence[str], expected_n: int) -> bool:
    """True iff every TFRecord kind has exactly expected_n records and,
    when 'sources' is in kinds, the sidecar is complete."""
```

`kinds` per path (each step only checks what it produces):

| Step | kinds |
|---|---|
| `step_generate_and_convolve_parallel` | `clean, hr, dirty, sources` |
| `step_generate` | `clean, sources` |
| `step_convolve` | `hr, dirty` |

`expected_n` is `args.ntrain` for `train`, `args.nvalid` for `validate`. A
complete-but-wrong-count subset (resubmit with a different `n`) fails the count
check and is regenerated — on-disk data always matches the requested `n`.

### Atomic sources merge

`concat_source_csvs` (`euclid_polish/sky/source_catalog.py`) currently streams
directly to its final path. Change it to write a sibling temp file and
`os.replace` it into place on success (the atomic-write pattern already used by
`StarCatalog.save`). This guarantees the final `sources_{subset}.csv` is only ever
observed complete, which is what makes the existence-based `_sources_complete`
check sound. The TFRecord merge (`_concat_tfrecords`) needs no such change — a
truncated TFRecord is already caught by the record-count check.

### Stale shard cleanup

```python
def _cleanup_parts(records_dir: str, subset: str) -> None:
    """Delete leftover *_{subset}.part####.{tfrecord,csv} from a dead run."""
```

Called in the parallel path immediately before (re)generating a subset. Without
it, orphan parts from a previous run with a different shard count linger on disk
(the merge only reads the freshly-computed `parts` list). Cleanup keeps disk tidy
and removes any chance of a stale part being misread.

### Guard placement

At the top of each per-subset loop iteration:

```python
if not args.force and _subset_complete(records_dir, subset, KINDS, n):
    _log(f"  {subset}: already complete ({n} records) — skipping")
    done += n                       # keep the cumulative progress bar honest
    reporter.set_step(done, grand_total, f"{subset} already complete")
    continue
# parallel path only, before fan-out:
_cleanup_parts(records_dir, subset)
```

- `step_generate_and_convolve_parallel` — guard + `_cleanup_parts` before the
  shard fan-out at `:469`.
- `step_generate` — guard before the `open_multiband_writer`/`SourceCatalogWriter`
  block at `:231`.
- `step_convolve` — guard before the hr/dirty writers at `:300` (in addition to
  the existing missing-`clean_` skip).

### CLI

- Add `--force` (`action="store_true"`) near the existing `--skip-*` flags
  (`:164`): regenerate everything, ignoring completeness. Current behavior becomes
  opt-in via this flag.
- No change to sbatch / FASRC templates — a plain resubmit auto-resumes.

### Progress

On a skipped subset, advance the cumulative `done` counter by `n` and emit one
`reporter` line so the WebUI bar reflects the skip rather than stalling. For the
parallel path, simply omit the per-subset `reporter.set_parallel(...)` call when
skipping.

## Testing

Extend `tests/test_run_pipeline_parallel.py` (uses `TinyCosmosCatalog`,
`image_size=96`, counts via `tf.data.TFRecordDataset`):

- **Unit** — `_count_tfrecords`: normal count; missing file → `None`; truncated
  file (write a valid TFRecord, chop trailing bytes) → `None`.
- **Unit** — `_sources_complete`: existing sidecar → `True`; missing file →
  `False`; `expected_n <= 0` → `True` regardless.
- **Unit** — `concat_source_csvs` is atomic: a sparse sidecar (a field with no
  rows) still merges correctly, and the merged file appears only on success.
- **Unit** — `_subset_complete`: all kinds at expected_n → `True`; one kind short
  → `False`; count `!=` expected → `False`.
- **Unit** — `_cleanup_parts`: removes only `*_{subset}.part*`, leaves final files
  and the other subset's parts.
- **Integration** — generate `train` shards + merge, then assert
  `_subset_complete(..., "train", ...)` is `True` and re-running the per-subset
  loop does not rewrite `train` (final-file mtime unchanged) while `validate`
  still generates.
- **Integration** — `--force` regenerates a subset that `_subset_complete` reports
  complete.

## Risks / edge cases

- **Truncated final file (mid-merge kill):** handled — `_count_tfrecords` returns
  `None` on `DataLossError`, so the subset is regenerated.
- **Kill in the merge window between TFRecord concat and CSV concat:** handled —
  the sources merge is atomic, so the final sidecar is absent (not partial) until
  it completes; `sources` is in the parallel path's `kinds`, so its absence marks
  the subset incomplete.
- **Sparse sidecar rows (a field with no rendered sources):** handled — the
  sources check is existence-based, not a field_index-coverage count, so a
  complete run whose last field is empty is not false-flagged.
- **Resubmit with a different `--ntrain/--nvalid`:** count mismatch ⇒ regenerate.
- **`_count_tfrecords` cost:** one streaming pass per final file on resume only;
  `step_convolve` already does this at `:275`, so the cost is acceptable.
