"""Stage TFRecord files onto node-local scratch for fast, contention-free reads.

FASRC's shared ``/n/netscratch`` is fast in aggregate but contended: under load
the training input pipeline stalls on shared-filesystem reads, leaving the CPUs
idle and the GPU starved (we measured GPU utilisation tracking step speed, not
the allocated GPU). FASRC's own guidance is to use **node-local** ``/scratch``
for I/O-intensive jobs, with the documented per-job pattern
``TMPDIR=/scratch mktemp -d``. Copying the handful of training records there once
at job start removes the per-epoch shared-FS reads entirely.

The copy is best-effort: if local scratch is missing or too small (records don't
fit), :func:`stage_records` logs and returns the original directory so training
still runs — just from netscratch.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Sequence


def local_scratch_base() -> str:
    """Best node-local scratch base.

    Prefer ``$TMPDIR`` (SLURM points it at a per-job local dir on FASRC), then
    ``/scratch`` (the documented node-local store), then the system temp dir so
    this is harmless off-cluster.
    """
    for base in (os.environ.get("TMPDIR"), "/scratch"):
        if base and os.path.isdir(base) and os.access(base, os.W_OK):
            return base
    return tempfile.gettempdir()


def stage_records(
    records_dir: str,
    names: Sequence[str],
    *,
    on_log: Callable[[str], None] = print,
) -> str:
    """Copy ``<records_dir>/<name>.tfrecord`` for each name to node-local scratch.

    Returns the directory training should read from: the new local directory on
    success, or ``records_dir`` unchanged if nothing could be staged (no local
    space, copy error, or none of the files exist). Staging is **all-or-nothing**
    — a partial copy is removed and the original directory returned, so callers
    never see a half-staged split.

    The returned local directory is the caller's to delete when training ends
    (it differs from ``records_dir`` exactly when staging happened).
    """
    present = [(n, os.path.join(records_dir, f"{n}.tfrecord")) for n in names]
    present = [(n, p) for n, p in present if os.path.isfile(p)]
    if not present:
        return records_dir

    base = local_scratch_base()
    try:
        local_dir = tempfile.mkdtemp(prefix="euclid_records_", dir=base)
    except OSError as e:
        on_log(f"⚠ node-local scratch unavailable ({e}); "
               f"reading records from {records_dir}")
        return records_dir

    total_bytes = 0
    try:
        for name, src in present:
            dst = os.path.join(local_dir, f"{name}.tfrecord")
            shutil.copy2(src, dst)
            total_bytes += os.path.getsize(dst)
    except OSError as e:                            # e.g. no space left on device
        shutil.rmtree(local_dir, ignore_errors=True)
        on_log(f"⚠ could not stage records to {base} ({e}); "
               f"reading from {records_dir}")
        return records_dir

    on_log(f"  ✓ staged {len(present)} record file(s) "
           f"({total_bytes / 1e9:.2f} GB) → {local_dir}")
    return local_dir
