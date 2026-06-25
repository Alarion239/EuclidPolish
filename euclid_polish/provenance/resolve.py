"""Resolve records by identity — a non-breaking superset of the old lookups.

Today the pipeline finds a record by a hardcoded ``<role>_<subset>.tfrecord``
name (and picks the "current" checkpoint by file mtime). This module lets
callers resolve by *identity* instead: prefer an id-tokenized file
(``clean_train.4b1e7a90.tfrecord``) and fall back to the legacy plain name with
a sentinel stamp, so existing layouts keep working and new code can ask which
run produced a record.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Optional

from euclid_polish.provenance.ids import ProvId


@dataclass(frozen=True)
class ResolvedRecord:
    """The path to a resolved record plus what is known of its provenance."""

    path: str
    prov_id: ProvId
    legacy: bool


def resolve_record(records_dir: str, *, role: str, subset: str,
                   ext: str = "tfrecord") -> ResolvedRecord:
    """Resolve ``<role>_<subset>`` in ``records_dir``.

    Prefers the newest id-tokenized file ``<role>_<subset>.<id8>.<ext>``; if
    none exists, falls back to the legacy ``<role>_<subset>.<ext>`` with the
    sentinel ("unknown provenance") id.
    """
    stem = f"{role}_{subset}"
    tokenized = sorted(glob.glob(os.path.join(records_dir, f"{stem}.*.{ext}")))
    if tokenized:
        path = tokenized[-1]
        token = os.path.basename(path).split(".")[-2]
        try:
            pid = ProvId(token)
        except ValueError:
            pid = ProvId.sentinel()
        return ResolvedRecord(path=path, prov_id=pid, legacy=False)

    legacy_path = os.path.join(records_dir, f"{stem}.{ext}")
    return ResolvedRecord(path=legacy_path, prov_id=ProvId.sentinel(), legacy=True)
