"""The :class:`ProvStore` — the one persistence class for provenance records.

Truth lives on disk as one ``<id>.<kind>.json`` sidecar per object, written
next to the data it describes (so it rsyncs with the data and survives the
time-travel worktrees). The in-memory index is a *derived* lookup, rebuildable
at any time by scanning for sidecars — it is never the source of truth.

Two producers keep their id inside a *stamp file* instead of a sidecar name: a
checkpoint dir's ``provenance.json`` and a catalog's ``<path>.prov.json``. The
scan reads those too, so a fresh mint never reuses a checkpoint or catalog id.
"""

from __future__ import annotations

import glob as _glob
import json
import os
import re

from euclid_polish.provenance._util import _atomic_write_json
from euclid_polish.provenance.checkpoint import PROVENANCE_FILENAME
from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.records import ProvRecord, record_from_dict

# Sidecar filename: <8-hex id>.<kind>.json
_SIDECAR_RE = re.compile(r"([0-9a-f]{8})\.[a-z0-9_]+\.json")

# Stamp-file suffix of a catalog's identity (``<catalog path>.prov.json``).
_CATALOG_STAMP_SUFFIX = ".prov.json"


def _stamp_file_id(path: str) -> str | None:
    """The id held in a stamp file, or ``None`` if it is not a readable stamp.

    ``provenance.json`` is also the name of some non-provenance JSON (e.g. a
    legacy plate run's manifest), so anything without a valid ``"id"`` is
    skipped rather than treated as an error.
    """
    try:
        with open(path) as fp:
            payload = json.load(fp)
        return str(ProvId(payload["id"]))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


class ProvStore:
    """Reads and writes provenance sidecars; mints collision-free ids.

    ``index_dir`` holds records written without an explicit ``sidecar_dir`` (and
    is where a future on-disk index cache would live). ``data_roots`` are the
    extra directories scanned for sidecars (and stamp files) co-located with
    their data.
    """

    def __init__(self, index_dir: str, data_roots: list[str] | None = None):
        self.index_dir = index_dir
        self.data_roots = list(data_roots) if data_roots else [index_dir]
        os.makedirs(index_dir, exist_ok=True)
        self._index: dict[str, str] = {}     # id -> sidecar path
        self._stamped: dict[str, str] = {}   # id -> stamp file (checkpoint / catalog)
        # Ids this instance has minted: reserved even before their artifact is
        # written, so two mints in one process never return the same id.
        self._minted: set[str] = set()
        self.rebuild_index()

    # -- roots / scanning -- #

    def _roots(self) -> set:
        return set(self.data_roots) | {self.index_dir}

    def rebuild_index(self) -> None:
        """Re-scan all roots for sidecars and stamp files; rebuild the index."""
        self._index.clear()
        self._stamped.clear()
        for root in self._roots():
            pattern = os.path.join(root, "**", "*.json")
            for path in _glob.glob(pattern, recursive=True):
                name = os.path.basename(path)
                m = _SIDECAR_RE.fullmatch(name)
                if m:
                    self._index[m.group(1)] = path
                elif name == PROVENANCE_FILENAME or name.endswith(_CATALOG_STAMP_SUFFIX):
                    stamp_id = _stamp_file_id(path)
                    if stamp_id is not None:
                        self._stamped[stamp_id] = path

    # -- minting / existence -- #

    def mint(self) -> ProvId:
        """Mint a fresh id guaranteed absent from the store, and reserve it."""
        pid = ProvId.mint(self.exists)
        self._minted.add(str(pid))
        return pid

    def exists(self, pid: ProvId) -> bool:
        """``True`` if any sidecar, stamp file or id-tokenized file uses this
        id, or this store already minted it."""
        s = str(pid)
        if s in self._index or s in self._stamped or s in self._minted:
            return True
        for root in self._roots():
            if _glob.glob(os.path.join(root, "**", f"{s}.*.json"), recursive=True):
                return True
            # Defensive: an id-tokenized artifact (e.g. clean_train.<id>.tfrecord)
            # may exist before its sidecar is indexed.
            if _glob.glob(os.path.join(root, "**", f"*.{s}.*"), recursive=True):
                return True
        return False

    # -- read / write -- #

    def put(self, record: ProvRecord, sidecar_dir: str | None = None) -> str:
        """Write ``record`` to a sidecar (next to its data if ``sidecar_dir``)."""
        directory = sidecar_dir or self.index_dir
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{record.id}.{record.kind}.json")
        _atomic_write_json(path, record.to_dict())
        self._index[str(record.id)] = path
        return path

    def get(self, pid: ProvId) -> ProvRecord:
        """Load the record for ``pid``; raises ``KeyError`` if absent."""
        s = str(pid)
        path = self._index.get(s)
        if path is None or not os.path.exists(path):
            self.rebuild_index()
            path = self._index.get(s)
        if path is None:
            raise KeyError(s)
        with open(path) as fp:
            return record_from_dict(json.load(fp))

    def get_or_none(self, pid: ProvId) -> ProvRecord | None:
        try:
            return self.get(pid)
        except KeyError:
            return None

    def all_records(self) -> list[ProvRecord]:
        out = []
        for path in self._index.values():
            try:
                with open(path) as fp:
                    out.append(record_from_dict(json.load(fp)))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
        return out

    def find(self, *, kind: str | None = None,
             produced_by: ProvId | None = None) -> list[ProvRecord]:
        """Return records matching the given filters (index scan)."""
        out = []
        for rec in self.all_records():
            if kind is not None and rec.kind != kind:
                continue
            if produced_by is not None and getattr(rec, "produced_by", None) != produced_by:
                continue
            out.append(rec)
        return out
