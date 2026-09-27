"""Provenance lineage index for the Ops › Provenance browser.

Truth lives on disk as one ``<id8>.<kind>.json`` sidecar per object (the
:mod:`euclid_polish.provenance` records), in ``data/_prov`` and next to the
data they describe, plus one ``provenance.json`` stamp per checkpoint
directory (the model's identity). :class:`ProvIndex` reads them all once into
an in-memory graph:

* **upstream** edges run from a record to what it came from — its
  ``parents``, its ``produced_by`` process and a process's ``inputs``
  (the same edges as :class:`euclid_polish.provenance.lineage.Lineage`);
* **downstream** edges are their reverse.

A checkpoint stamp becomes a ``checkpointartifact`` entry (source
``checkpoint``) so an SR product parented on a model resolves to the member
that made it. The **verdict** of an artifact or inference run compares the
model(s) behind it with the *current* models (the active ensemble members'
ids): ``current`` when every model is active, ``stale`` when one is not,
``unknown`` when no model was recorded (legacy / un-stamped products).

The scan (``os.walk`` of the data roots) is cheap but not free (~2 s for
11 k sidecars), so :func:`get_index` caches the index for ``TTL_S`` and
rebuilds early when ``data/_prov`` changes; ``rebuild=True`` forces it.
Read-only: nothing here writes to disk.
"""

from __future__ import annotations

import glob
import json
import os
import re
import threading
import time
from collections import deque
from collections.abc import Iterable
from typing import Any

from euclid_polish import ensemble_registry
from euclid_polish.config import Config

SIDECAR_RE = re.compile(r"^([0-9a-f]{8})\.([a-z0-9_]+)\.json$")
_ID_RE = re.compile(r"^[0-9a-f]{8}$")
SENTINEL = "00000000"
#: Record kinds that identify "the model" behind an artifact.
MODEL_KINDS = frozenset({"checkpointartifact", "trainingrun"})
PROCESS_KINDS = frozenset({"process", "generationrun", "trainingrun", "inferencerun"})
#: Kinds that get a current/stale verdict (products of a model).
VERDICT_KINDS = frozenset({"srcutoutartifact", "inferencerun"})
VERDICTS = ("current", "stale", "unknown")

TTL_S = 300.0
#: Cap on files visited while walking the data roots (a runaway tree must not
#: stall the request); the summary reports ``truncated`` when it is hit.
MAX_FILES = 500_000
#: Cap on the ids listed per lineage direction in a record's detail.
LINEAGE_CAP = 300
#: Directory names never descended into (caches, VCS, sandboxes).
_SKIP_DIRS = frozenset({".git", ".timetravel", "__pycache__", "node_modules"})


def _rel(path: str) -> str:
    try:
        rel = os.path.relpath(path, os.getcwd())
    except ValueError:
        return path
    return path if rel.startswith("..") else rel


def _ids(values: Any) -> list[str]:
    """The valid, non-sentinel 8-hex ids of a list field."""
    if not isinstance(values, list | tuple):
        return []
    out = []
    for v in values:
        s = str(v).strip().lower()
        if _ID_RE.match(s) and s != SENTINEL and s not in out:
            out.append(s)
    return out


def _one_id(value: Any) -> str | None:
    s = str(value or "").strip().lower()
    return s if _ID_RE.match(s) and s != SENTINEL else None


def _load(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            value = json.load(fh)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _label(rec: dict[str, Any], kind: str) -> str:
    if kind in PROCESS_KINDS:
        config = rec.get("config") if isinstance(rec.get("config"), dict) else {}
        ctype = config.get("config_type") if config else None
        return f"{kind}{f' · {ctype}' if ctype else ''}"
    path = rec.get("path")
    if isinstance(path, str) and path:
        parts = [p for p in path.replace("\\", "/").split("/") if p and p != "."]
        return "/".join(parts[-2:]) if parts else kind
    return kind


def _entry(rec: dict[str, Any], *, pid: str, kind: str, file: str,
           source: str) -> dict[str, Any]:
    git = rec.get("git") if isinstance(rec.get("git"), dict) else {}
    desc = rec.get("descriptors") if isinstance(rec.get("descriptors"), dict) else {}
    config = rec.get("config") if isinstance(rec.get("config"), dict) else {}
    return {
        "id": pid, "kind": kind, "source": source, "file": file,
        "category": ("process" if kind in PROCESS_KINDS else "artifact"),
        "created_at": rec.get("created_at"),
        "status": rec.get("status"),
        "path": rec.get("path") if isinstance(rec.get("path"), str) else None,
        "format": rec.get("format"),
        "label": _label(rec, kind),
        "git": git.get("short") or (str(git.get("hash"))[:7] if git.get("hash") else None),
        "dirty": bool(git.get("dirty")) if git else None,
        "config_type": config.get("config_type") if config else None,
        "seed": rec.get("seed"),
        "produced_by": _one_id(rec.get("produced_by")),
        "parents": _ids(rec.get("parents")),
        "inputs": _ids(rec.get("inputs")),
        "outputs": _ids(rec.get("outputs")),
        "ra": desc.get("ra") if isinstance(desc.get("ra"), int | float) else None,
        "dec": desc.get("dec") if isinstance(desc.get("dec"), int | float) else None,
        "member": None,
    }


def _upstream(entry: dict[str, Any]) -> list[tuple[str, str]]:
    """``(role, id)`` of everything immediately upstream of ``entry``."""
    out: list[tuple[str, str]] = [("parent", p) for p in entry["parents"]]
    if entry["produced_by"]:
        out.append(("produced_by", entry["produced_by"]))
    out.extend(("input", i) for i in entry["inputs"])
    seen: set[str] = set()
    uniq = []
    for role, pid in out:
        if pid not in seen:
            seen.add(pid)
            uniq.append((role, pid))
    return uniq


class ProvIndex:
    """The in-memory lineage graph of every provenance record found."""

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.children: dict[str, set[str]] = {}
        self.roots: list[dict[str, Any]] = []
        self.current_models: list[dict[str, Any]] = []
        self.truncated = False
        self.duplicates = 0
        self.built_at = time.time()
        self.build_seconds = 0.0

    # -- building ----------------------------------------------------------

    def add_sidecar(self, path: str, source: str) -> bool:
        m = SIDECAR_RE.match(os.path.basename(path))
        if not m:
            return False
        rec = _load(path)
        if rec is None:
            return False
        pid = _one_id(rec.get("id")) or m.group(1)
        if pid in self.entries:
            self.duplicates += 1
            return False
        self.entries[pid] = _entry(rec, pid=pid, kind=str(rec.get("kind") or m.group(2)),
                                   file=_rel(path), source=source)
        return True

    def add_checkpoint(self, stamp_path: str, member: str | None) -> bool:
        stamp = _load(stamp_path)
        pid = _one_id((stamp or {}).get("id"))
        if not stamp or not pid:
            return False
        ckpt_dir = os.path.dirname(stamp_path)
        entry = _entry({"parents": stamp.get("parents"), "produced_by": stamp.get("produced_by"),
                        "path": _rel(ckpt_dir)},
                       pid=pid, kind="checkpointartifact", file=_rel(stamp_path),
                       source="checkpoint")
        entry["member"] = member
        entry["label"] = member or _rel(ckpt_dir)
        entry["format"] = "ckpt"
        with_mtime = os.path.getmtime(stamp_path) if os.path.exists(stamp_path) else None
        entry["created_at"] = (time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(with_mtime))
                               if with_mtime else None)
        existing = self.entries.get(pid)
        if existing is not None and existing["source"] != "checkpoint":
            # A real sidecar for this model id wins; just name its member.
            existing["member"] = existing.get("member") or member
            return False
        if existing is None:
            self.entries[pid] = entry
        return True

    def link(self) -> None:
        self.children = {}
        for pid, entry in self.entries.items():
            for _role, up in _upstream(entry):
                self.children.setdefault(up, set()).add(pid)

    # -- queries -----------------------------------------------------------

    def get(self, pid: str) -> dict[str, Any] | None:
        return self.entries.get(str(pid).lower())

    def upstream(self, pid: str) -> list[dict[str, Any]]:
        entry = self.get(pid)
        if entry is None:
            return []
        return [self._ref(up, role) for role, up in _upstream(entry)]

    def downstream(self, pid: str) -> list[dict[str, Any]]:
        kids = sorted(self.children.get(str(pid).lower(), ()),
                      key=lambda k: str((self.entries.get(k) or {}).get("created_at") or ""),
                      reverse=True)
        return [self._ref(k, "child") for k in kids]

    def _ref(self, pid: str, role: str) -> dict[str, Any]:
        entry = self.entries.get(pid)
        return {"id": pid, "role": role, "exists": entry is not None,
                "kind": entry["kind"] if entry else None,
                "label": entry["label"] if entry else None,
                "member": entry.get("member") if entry else None}

    def walk(self, pid: str, direction: str, cap: int = LINEAGE_CAP
             ) -> tuple[list[dict[str, Any]], int]:
        """Transitive ``ancestors`` / ``descendants`` (BFS order, with the hop
        ``depth``): ``(first cap items, total count)``."""
        start = str(pid).lower()
        seen = {start}
        order: list[tuple[str, int]] = []
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        while queue:
            current, depth = queue.popleft()
            if direction == "ancestors":
                entry = self.entries.get(current)
                nxt = [up for _r, up in _upstream(entry)] if entry else []
            else:
                nxt = list(self.children.get(current, ()))
            for n in nxt:
                if n not in seen:
                    seen.add(n)
                    order.append((n, depth + 1))
                    queue.append((n, depth + 1))
        items = []
        for n, depth in order[:cap]:
            ref = self._ref(n, direction[:-1])
            ref["depth"] = depth
            items.append(ref)
        return items, len(order)

    def models_of(self, pid: str) -> list[str]:
        """The model ids behind a record: model-kind parents/inputs, one hop
        through its producing process (``Lineage.model_of``, all of them)."""
        entry = self.get(pid)
        if entry is None:
            return []
        candidates = list(entry["parents"]) + list(entry["inputs"])
        proc = self.entries.get(entry["produced_by"]) if entry["produced_by"] else None
        if proc is not None:
            candidates += proc["parents"] + proc["inputs"]
        out = []
        for c in candidates:
            e = self.entries.get(c)
            if e is not None and e["kind"] in MODEL_KINDS and c not in out:
                out.append(c)
        return out

    def verdict(self, pid: str) -> str | None:
        """``current`` / ``stale`` / ``unknown`` for a model product, else
        ``None`` (generation runs, checkpoints, …)."""
        entry = self.get(pid)
        if entry is None or entry["kind"] not in VERDICT_KINDS:
            return None
        models = self.models_of(pid)
        if not models:
            return "unknown"
        current = {m["id"] for m in self.current_models}
        return "current" if all(m in current for m in models) else "stale"

    def row(self, entry: dict[str, Any]) -> dict[str, Any]:
        """The listing row of an entry (verdict + lineage counts added)."""
        pid = entry["id"]
        models = self.models_of(pid) if entry["kind"] in VERDICT_KINDS else []
        return {**entry, "verdict": self.verdict(pid),
                "models": [{"id": m, "member": (self.entries.get(m) or {}).get("member")}
                           for m in models],
                "n_upstream": len(_upstream(entry)),
                "n_downstream": len(self.children.get(pid, ()))}

    def counts(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        verdicts = dict.fromkeys(VERDICTS, 0)
        for pid, entry in self.entries.items():
            kinds[entry["kind"]] = kinds.get(entry["kind"], 0) + 1
            v = self.verdict(pid)
            if v:
                verdicts[v] += 1
        return {"kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
                "verdicts": verdicts}

    def search(self, *, q: str = "", kind: str = "", verdict: str = "",
               source: str = "") -> list[dict[str, Any]]:
        """Matching entries, newest first. ``q`` is ANDed whitespace tokens,
        each a substring of the id, kind, label, path, git commit, member or
        config type; ``kind``/``source``/``verdict`` are exact."""
        tokens = [t for t in q.lower().split() if t]
        kinds = {k for k in kind.split(",") if k}
        out = []
        for pid, entry in self.entries.items():
            if kinds and entry["kind"] not in kinds:
                continue
            if source and entry["source"] != source:
                continue
            if verdict and self.verdict(pid) != verdict:
                continue
            if tokens:
                hay = " ".join(str(entry.get(k) or "") for k in
                               ("id", "kind", "label", "path", "git", "member",
                                "config_type", "status", "produced_by")).lower()
                if not all(t in hay for t in tokens):
                    continue
            out.append(entry)
        out.sort(key=lambda e: str(e.get("created_at") or ""), reverse=True)
        return out


# ---------------------------------------------------------------------------
# Building from disk
# ---------------------------------------------------------------------------

def _walk_sidecars(root: str, skip: Iterable[str], budget: list[int]) -> Iterable[str]:
    skip_real = {os.path.realpath(s) for s in skip}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")
                       and os.path.realpath(os.path.join(dirpath, d)) not in skip_real]
        for name in filenames:
            budget[0] -= 1
            if budget[0] <= 0:
                return
            if SIDECAR_RE.match(name):
                yield os.path.join(dirpath, name)


def current_model_ids(ensemble_dir: str | None = None) -> list[dict[str, Any]]:
    """``[{id, member, regime, dir}]`` of the active ensemble members that
    carry a ``provenance.json`` (legacy members have no model id)."""
    base = ensemble_dir or ensemble_registry.default_ensemble_dir()
    out = []
    try:
        dirs = ensemble_registry.active_member_dirs(base)
    except OSError:
        dirs = []
    for d in dirs:
        stamp = _load(os.path.join(d, "provenance.json"))
        pid = _one_id((stamp or {}).get("id"))
        if not pid:
            continue
        out.append({"id": pid, "member": os.path.basename(d),
                    "regime": "starless" if ensemble_registry.member_is_starless(d) else "starfull",
                    "dir": _rel(d)})
    return out


def build_index(*, prov_dir: str, data_dirs: list[str], ckpt_root: str | None,
                current_models: list[dict[str, Any]],
                max_files: int = MAX_FILES) -> ProvIndex:
    """Scan ``prov_dir`` (flat), walk ``data_dirs`` for co-located sidecars
    and read the checkpoint stamps under ``ckpt_root`` (``*/provenance.json``,
    ``*/*/provenance.json``)."""
    started = time.monotonic()
    index = ProvIndex()
    index.current_models = list(current_models)
    budget = [max_files]
    count = 0
    if os.path.isdir(prov_dir):
        for name in sorted(os.listdir(prov_dir)):
            count += index.add_sidecar(os.path.join(prov_dir, name), "prov")
    index.roots.append({"path": _rel(prov_dir), "role": "index", "records": count})
    for root in data_dirs:
        if not os.path.isdir(root):
            continue
        count = 0
        for path in _walk_sidecars(root, [prov_dir], budget):
            count += index.add_sidecar(path, "sidecar")
        index.roots.append({"path": _rel(root), "role": "data", "records": count})
    index.truncated = budget[0] <= 0
    if ckpt_root and os.path.isdir(ckpt_root):
        count = 0
        members = {m["id"]: m["member"] for m in current_models}
        for pattern in ("*/provenance.json", "*/*/provenance.json"):
            for path in sorted(glob.glob(os.path.join(glob.escape(ckpt_root), pattern))):
                d = os.path.dirname(path)
                member = os.path.basename(d) if os.path.basename(d).startswith("member_") else None
                stamp = _load(path)
                pid = _one_id((stamp or {}).get("id"))
                count += index.add_checkpoint(path, member or members.get(pid or ""))
        index.roots.append({"path": _rel(ckpt_root), "role": "checkpoints", "records": count})
    index.link()
    index.build_seconds = round(time.monotonic() - started, 3)
    return index


def _default_roots() -> dict[str, Any]:
    ckpt_root = os.path.dirname(os.path.realpath(Config.DEFAULT_CHECKPOINT_DIR))
    return {"prov_dir": Config.PROV_DIR, "data_dirs": [Config.DATA_DIR], "ckpt_root": ckpt_root}


_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"index": None, "key": None, "at": 0.0}


def _prov_signature(prov_dir: str) -> float | None:
    try:
        return os.stat(prov_dir).st_mtime
    except OSError:
        return None


def get_index(*, rebuild: bool = False) -> ProvIndex:
    """The cached project index (rebuilt after ``TTL_S``, when ``data/_prov``
    changes, when the roots change, or on ``rebuild``)."""
    roots = _default_roots()
    key = (roots["prov_dir"], tuple(roots["data_dirs"]), roots["ckpt_root"],
           _prov_signature(roots["prov_dir"]))
    with _LOCK:
        cached = _CACHE["index"]
        fresh = (cached is not None and _CACHE["key"] == key
                 and time.monotonic() - _CACHE["at"] < TTL_S)
        if fresh and not rebuild:
            return cached
        index = build_index(current_models=current_model_ids(), **roots)
        _CACHE.update(index=index, key=key, at=time.monotonic())
        return index


def reset_cache() -> None:
    with _LOCK:
        _CACHE.update(index=None, key=None, at=0.0)


def read_record(entry: dict[str, Any]) -> dict[str, Any] | None:
    """The record as stored on disk (the sidecar, or the checkpoint stamp)."""
    path = entry.get("file")
    if not path:
        return None
    full = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
    return _load(full)
