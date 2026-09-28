"""The local cache of fetched study-field products (least recently used first).

An attached field lives on holylabs as one file per product; fetching copies
products — never the whole field implicitly — into
``<Config.VIS_DIR>/study_fields/<study id>/<field id>/`` (not inside the
tracking dir, which the holylabs mirror pushes). "Fetch field" brings the
core products (:data:`CORE_PRODUCTS`); member SRs are fetched explicitly.

The cache is bounded like the experiments' member-SR cache
(:class:`experiments.MemberCacheBudget`), per product file: at most
:data:`FIELD_CACHE_BUDGET_BYTES` over every study, least recently used
product files evicted first, and no fetch that would leave less than
``experiments.MIN_FREE_BYTES`` (5 GiB) free. A product counts as cached only
once the field's ``.fetched.json`` marker records it — written after its
sha256 matched the study manifest.

This module only manages the cache directory; the transfer itself is
:func:`euclid_polish.studies.fields.fetch_field`.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import threading
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from euclid_polish.config import Config
from euclid_polish.studies.store import check_field_id, check_study_id
from euclid_polish.web.helpers import atomic_files, experiments, model_catalog

#: Upper bound of the fetched-product cache over every study.
FIELD_CACHE_BUDGET_BYTES = 2 * 1024 ** 3
#: What "Fetch field" brings (those a field has); members are fetched explicitly.
CORE_PRODUCTS = ("field.json", "truth.json", "lr", "hr", "mean", "gate", "mask")
MARKER = ".fetched.json"
_LOCK = threading.RLock()
#: Bytes the running freeze / fetch jobs are about to write locally: every
#: disk-margin check subtracts them, so two jobs cannot jointly break it.
_RESERVED = 0
_RESERVE_LOCK = threading.Lock()


def reserved_bytes() -> int:
    with _RESERVE_LOCK:
        return _RESERVED


@contextlib.contextmanager
def claim(path: Path | str, nbytes: int, *, min_free: int | None = None):
    """Reserve ``nbytes`` on the disk holding ``path`` for the ``with``
    block: :class:`experiments.DiskSpaceError` when the free space minus
    every other reservation would drop below ``min_free`` (5 GiB)."""
    global _RESERVED
    margin = int(experiments.MIN_FREE_BYTES if min_free is None else min_free)
    need = int(nbytes)
    with _RESERVE_LOCK:
        free = experiments.free_bytes(Path(path))
        if free - _RESERVED - need < margin:
            raise experiments.DiskSpaceError(
                f"not enough disk space for {need / 1024 ** 2:.0f} MB: {free / 1024 ** 3:.2f} GiB "
                f"free, {_RESERVED / 1024 ** 2:.0f} MB reserved by running study jobs and "
                f"{margin / 1024 ** 3:.0f} GiB must stay free", needed=need, free=free)
        _RESERVED += need
    try:
        yield need
    finally:
        with _RESERVE_LOCK:
            _RESERVED -= need


def member_product(label: str) -> str:
    """A member's product name: ``"170·psnr"`` → ``member_170``."""
    return model_catalog.member_name(label)


def cache_root() -> Path:
    return Path(Config.VIS_DIR) / "study_fields"


def staging_root() -> Path:
    """Where a freeze packs one product at a time (hidden, never cached)."""
    return cache_root() / ".staging"


def product_file(name: str) -> str:
    """A product's file name (``hr`` → ``hr.npz``; JSON products keep theirs)."""
    if "/" in name or ".." in name or not name:
        raise ValueError(f"bad product name {name!r}")
    return name if name.endswith(".json") else f"{name}.npz"


class FieldCache:
    """Fetched products under ``root`` with a byte ``budget`` and a free-disk
    margin ``min_free``."""

    def __init__(self, root: Path | None = None, *, budget: int | None = None,
                 min_free: int | None = None) -> None:
        self.root = Path(root) if root is not None else cache_root()
        self.budget = int(FIELD_CACHE_BUDGET_BYTES if budget is None else budget)
        self.min_free = int(experiments.MIN_FREE_BYTES if min_free is None else min_free)

    # ----------------------------- layout --------------------------------

    def field_dir(self, study_id: str, fid: str) -> Path:
        return self.root / check_study_id(study_id) / check_field_id(fid)

    def product_path(self, study_id: str, fid: str, name: str) -> Path:
        return self.field_dir(study_id, fid) / product_file(name)

    def marker(self, study_id: str, fid: str) -> dict[str, Any]:
        try:
            payload = json.loads((self.field_dir(study_id, fid) / MARKER).read_text("utf-8"))
        except (OSError, ValueError):
            return {"products": {}}
        return payload if isinstance(payload, dict) else {"products": {}}

    def _write_marker(self, directory: Path, marker: Mapping[str, Any]) -> None:
        atomic_files.write_json(directory / MARKER, dict(marker), indent=2)

    # ----------------------------- state ---------------------------------

    def cached_products(self, study_id: str, fid: str,
                        products: Mapping[str, Mapping[str, Any]] | None = None) -> set[str]:
        """Products fetched and verified (against the manifest's ``products``
        hashes when given) whose file is still there."""
        recorded = self.marker(study_id, fid).get("products") or {}
        out = set()
        for name, info in recorded.items():
            if products is not None and (products.get(name) or {}).get("sha256") != (
                    info or {}).get("sha256"):
                continue
            with contextlib.suppress(ValueError):
                if self.product_path(study_id, fid, name).is_file():
                    out.add(name)
        return out

    def is_cached(self, study_id: str, fid: str, names: Iterable[str],
                  products: Mapping[str, Mapping[str, Any]] | None = None) -> bool:
        wanted = set(names)
        return bool(wanted) and wanted <= self.cached_products(study_id, fid, products)

    def entries(self) -> list[tuple[int, Path, int]]:
        """``(last use ns, product file, bytes)`` of every cached product,
        least recently used first."""
        out = []
        if not self.root.is_dir():
            return out
        for study in self.root.iterdir():
            if study.name.startswith(".") or not study.is_dir():
                continue
            for field in study.iterdir():
                if field.name.startswith(".") or not field.is_dir():
                    continue
                for item in field.iterdir():
                    if item.name.startswith(".") or not item.is_file():
                        continue
                    with contextlib.suppress(OSError):
                        stat = item.stat()
                        out.append((stat.st_mtime_ns, item, int(stat.st_size)))
        return sorted(out, key=lambda entry: (entry[0], str(entry[1])))

    def total(self) -> int:
        return sum(size for _t, _p, size in self.entries())


    # ----------------------------- changes -------------------------------

    def make_room(self, nbytes: int, *, keep: Iterable[Path] = ()) -> bool:
        """Evict least recently used product files (never those in ``keep``)
        until ``nbytes`` more fit the budget; ``False`` when they cannot."""
        protected = {Path(p) for p in keep}
        with _LOCK:
            entries = self.entries()
            total = sum(size for _t, _p, size in entries)
            for _t, path, size in entries:
                if total + int(nbytes) <= self.budget:
                    break
                if path in protected:
                    continue
                self._evict(path)
                total -= size
            return total + int(nbytes) <= self.budget

    def _evict(self, path: Path) -> None:
        directory = path.parent
        marker = {"products": {}}
        with contextlib.suppress(OSError, ValueError):
            marker = json.loads((directory / MARKER).read_text("utf-8"))
        name = path.name.removesuffix(".npz")
        (marker.get("products") or {}).pop(name, None)
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        self._write_marker(directory, marker)

    def record(self, study_id: str, fid: str, name: str, info: Mapping[str, Any]) -> None:
        """Mark one verified product as cached."""
        with _LOCK:
            directory = self.field_dir(study_id, fid)
            marker = self.marker(study_id, fid)
            marker.setdefault("products", {})[name] = {
                "sha256": info.get("sha256"), "bytes": info.get("bytes"),
                "fetched": datetime.now(UTC).isoformat(timespec="seconds")}
            self._write_marker(directory, marker)

    def touch(self, study_id: str, fid: str, name: str) -> None:
        with contextlib.suppress(OSError, ValueError):
            os.utime(self.product_path(study_id, fid, name))

    def load(self, study_id: str, fid: str, product: str) -> np.ndarray:
        """One cached product array (:class:`FileNotFoundError` when absent)."""
        path = self.product_path(study_id, fid, product)
        if not path.is_file() or product.endswith(".json"):
            raise FileNotFoundError(f"{fid} has no cached product {product!r}")
        with np.load(path) as handle:
            return np.asarray(handle["data"])

    def read_json(self, study_id: str, fid: str, name: str) -> dict[str, Any]:
        return json.loads(self.product_path(study_id, fid, name).read_text(encoding="utf-8"))

    def remove_study(self, study_id: str) -> None:
        with _LOCK:
            shutil.rmtree(self.root / check_study_id(study_id), ignore_errors=True)


__all__ = ["CORE_PRODUCTS", "FIELD_CACHE_BUDGET_BYTES", "FieldCache", "cache_root", "claim",
           "member_product", "product_file", "reserved_bytes", "staging_root"]
