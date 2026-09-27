"""What code is this server running? (contract C3, ``GET /api/version``).

The server has no auto-reloader, so after an edit, ``git pull`` or checkout
it keeps running the backend code it loaded at boot. ``behind`` answers the
one question that needs a restart: **did a backend Python file this process
loaded change on disk since it was loaded?** :class:`BackendSources` records
every ``euclid_polish/**/*.py`` module in ``sys.modules`` at boot (stat +
content hash; ``web/frontend`` and ``web/static`` excluded — a new SPA build
never needs a restart) and compares them with the files on disk, at most once
per :data:`CHANGED_TTL_S`. A new mtime with the same bytes (a checkout, a
stash pop, ``touch``) is not a change; a deleted loaded file is.

A module imported lazily after boot is first seen at the next scan, not at
its import. So that an edit made in between is not missed, a first-seen file
is checked against the source stamp (mtime + size) in the header of its
``__pycache__`` bytecode, which Python rewrites whenever it compiles the
source it imports: a stamp that no longer matches the file means the file
changed after it was loaded (reported until a restart, since the loaded bytes
are unknown). Without a usable stamp — ``sys.dont_write_bytecode``, a
hash-based or foreign ``.pyc``, no ``.pyc`` — the file is baselined as it is
when first seen, and an edit between its import and that first scan goes
unreported (a window of at most one poll while a browser is open).
``changed_digest`` names the whole changed set, so a client can remember a
dismissal until the set itself changes (re-saving an already-changed file
reorders the capped ``changed_files`` list but is not a new change).

Commit ids are informational only: committing the code the server already
runs moves ``HEAD`` but is not "older code" (the old false positive). The
payload still reports the boot commit and the live ``HEAD``, uncommitted edits
to tracked files (``dirty``) and which SPA build is served
(``static/dist/index.html`` mtime + content hash, and its entry script, which
a page compares with the script it was itself loaded from).

All probes are cheap and read-only: a stat of the ~200 loaded files (hashing
only files whose stat moved) and local ``git`` calls (~10–25 ms, no optional
index locks, so polling never races a concurrent ``git commit``). Nothing
raises: outside a git checkout the commit fields are ``None`` and
``dirty``/``behind`` false.
"""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

_WEB_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = _WEB_DIR.parent
REPO_ROOT = _WEB_DIR.parents[1]
DIST_INDEX = _WEB_DIR / "static" / "dist" / "index.html"

#: Trees under the package whose files never need a server restart.
EXCLUDED_DIRS = ("web/frontend", "web/static")
#: The file check is reused for this long (the SPA polls every 60 s).
CHANGED_TTL_S = 10.0
#: How many changed files the payload lists (``changed_count`` has them all).
MAX_CHANGED_LISTED = 8

_GIT_TIMEOUT_S = 5
_SHORT = 7
_HASH_CHARS = 16
#: The build's entry chunk: ``<script type="module" … src="…">`` in index.html.
_ENTRY_RE = re.compile(r'<script\b[^>]*\btype="module"[^>]*\bsrc="([^"]+)"', re.IGNORECASE)


def _git(repo: Path, *args: str) -> str | None:
    """stdout of ``git -C repo args`` or None when git fails / is missing.

    Every probe is read-only: ``--no-optional-locks`` (and
    ``GIT_OPTIONAL_LOCKS=0`` for child gits) stops ``git status`` from
    refreshing the index in place under ``.git/index.lock``, which would make
    a concurrent ``git commit``/``git add`` fail while the SPA polls
    ``/api/version`` (git's advice for background scripts).
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "--no-optional-locks", *args],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def git_head(repo: Path) -> str | None:
    """Full SHA of ``HEAD`` in ``repo`` (None outside a git checkout)."""
    out = _git(repo, "rev-parse", "--verify", "--quiet", "HEAD")
    sha = (out or "").strip()
    return sha or None


def git_dirty(repo: Path) -> bool:
    """True when tracked files differ from ``HEAD`` (untracked files ignored)."""
    out = _git(repo, "status", "--porcelain", "--untracked-files=no")
    return bool(out and out.strip())


def _iso(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp, tz=dt.UTC).isoformat()


def dist_info(index: Path) -> dict[str, str | None]:
    """``{built_at, index_hash, entry}`` of the served SPA build (nulls when
    absent). ``entry`` is the src of its module entry script (its name carries
    the content hash of the whole build)."""
    try:
        data = index.read_bytes()
        built = index.stat().st_mtime
    except OSError:
        return {"built_at": None, "index_hash": None, "entry": None}
    match = _ENTRY_RE.search(data.decode("utf-8", "replace"))
    return {
        "built_at": _iso(built),
        "index_hash": hashlib.sha256(data).hexdigest()[:_HASH_CHARS],
        "entry": match.group(1) if match else None,
    }


def _short(sha: str | None) -> str | None:
    return sha[:_SHORT] if sha else None


class _BackendFilter:
    """``.py`` under ``root`` and outside :data:`EXCLUDED_DIRS`, by string
    prefix on resolved paths (pathlib's ``is_relative_to`` over the ~7000
    entries of ``sys.modules`` costs ~0.3 s; this ~2 ms)."""

    def __init__(self, root: Path) -> None:
        base = os.path.realpath(root)
        self.prefix = base + os.sep
        self.excluded = tuple(os.path.join(base, *d.split("/")) + os.sep for d in EXCLUDED_DIRS)
        self._real: dict[str, str] = {}

    def resolve(self, raw: str) -> str:
        real = self._real.get(raw)
        if real is None:
            real = self._real[raw] = os.path.realpath(raw)
        return real

    def accepts(self, real: str) -> bool:
        return (real.endswith(".py") and real.startswith(self.prefix)
                and not real.startswith(self.excluded))


def loaded_backend_files(root: Path = PACKAGE_ROOT, _filter: _BackendFilter | None = None) -> list[Path]:
    """The ``.py`` files under ``root`` of the modules in ``sys.modules``
    (``web/frontend`` and ``web/static`` excluded), resolved."""
    flt = _filter or _BackendFilter(root)
    out: list[Path] = []
    for module in list(sys.modules.values()):
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, str) or not raw.endswith(".py"):
            continue
        real = flt.resolve(raw)
        if flt.accepts(real):
            out.append(Path(real))
    return out


def _stat_key(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _loaded_stamp_differs(path: Path, key: tuple[int, int]) -> bool:
    """True when ``path``'s bytecode was compiled from a different source
    stamp than the file has now (``key`` = its stat): the file changed after
    it was imported. False whenever the stamp is unusable (see the module
    docstring)."""
    if sys.dont_write_bytecode:
        return False
    try:
        with open(importlib.util.cache_from_source(str(path)), "rb") as fh:
            header = fh.read(16)
    except (OSError, ValueError, NotImplementedError):
        return False
    if len(header) < 16 or header[:4] != importlib.util.MAGIC_NUMBER:
        return False
    if int.from_bytes(header[4:8], "little") != 0:   # hash-based .pyc: no stamp
        return False
    mtime = int.from_bytes(header[8:12], "little")
    size = int.from_bytes(header[12:16], "little")
    now_mtime = (key[0] // 1_000_000_000) & 0xFFFFFFFF
    return (mtime, size) != (now_mtime, key[1] & 0xFFFFFFFF)


class BackendSources:
    """The backend files this process loaded, as they were when loaded, vs
    the files on disk now (see the module docstring)."""

    def __init__(
        self,
        root: Path = PACKAGE_ROOT,
        *,
        rel_to: Path | None = None,
        loaded: Callable[[], Iterable[Path]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        ttl: float = CHANGED_TTL_S,
    ) -> None:
        self.root = Path(root).resolve()
        self.rel_to = Path(rel_to).resolve() if rel_to is not None else self.root.parent
        self._filter = _BackendFilter(self.root)
        self._loaded = loaded or (lambda: loaded_backend_files(self.root, self._filter))
        self._clock = clock
        self._ttl = ttl
        self._lock = threading.Lock()
        # path → (stat when loaded, content hash when loaded)
        self._base: dict[Path, tuple[tuple[int, int] | None, str | None]] = {}
        # path → (stat last checked, changed?) so unchanged stats never re-hash
        self._seen: dict[Path, tuple[tuple[int, int] | None, bool]] = {}
        self._cache: tuple[float, list[str]] | None = None
        with self._lock:
            self._scan()  # the boot snapshot

    def _files(self) -> set[Path]:
        out = set()
        for p in self._loaded():
            real = self._filter.resolve(str(p))
            if self._filter.accepts(real):
                out.add(Path(real))
        return out

    def _scan(self) -> list[tuple[int, str]]:
        """(mtime_ns, relative path) of every loaded file that changed."""
        changed: list[tuple[int, str]] = []
        for path in self._files():
            key = _stat_key(path)
            if path not in self._base:
                if key is not None and _loaded_stamp_differs(path, key):
                    # Edited between its import and now: the loaded bytes are
                    # unknown, so it stays changed until a restart.
                    self._base[path] = (None, None)
                else:                           # first seen: loaded from this content
                    self._base[path] = (key, _digest(path) if key else None)
                    self._seen[path] = (key, False)
                    continue
            seen = self._seen.get(path)
            if seen is not None and seen[0] == key:
                is_changed = seen[1]
            else:
                base_key, base_hash = self._base[path]
                if key is None:
                    is_changed = True           # deleted (or unreadable)
                elif key == base_key:
                    is_changed = False
                else:
                    is_changed = _digest(path) != base_hash
                self._seen[path] = (key, is_changed)
            if is_changed:
                rel = path.relative_to(self.rel_to) if path.is_relative_to(self.rel_to) else path
                changed.append((key[0] if key else 0, rel.as_posix()))
        return changed

    def changed(self) -> list[str]:
        """Loaded backend files that differ from what was loaded, newest first
        (relative to ``rel_to``); cached for ``ttl`` seconds."""
        with self._lock:
            now = self._clock()
            if self._cache is not None and now - self._cache[0] < self._ttl:
                return list(self._cache[1])
            found = self._scan()
            # Newest first; a deleted file (no mtime) sorts last.
            result = [rel for _, rel in sorted(found, key=lambda t: (-t[0], t[1]))]
            self._cache = (now, result)
            return list(result)


def _set_digest(paths: Iterable[str]) -> str | None:
    """Short hash of a set of paths (order-free); None for the empty set."""
    items = sorted(set(paths))
    if not items:
        return None
    return hashlib.sha1("\n".join(items).encode()).hexdigest()[:_HASH_CHARS]


class VersionTracker:
    """Boot commit (fixed at construction) vs live ``HEAD`` of one checkout,
    and the loaded backend sources vs the files on disk."""

    def __init__(
        self,
        repo: Path = REPO_ROOT,
        dist_index: Path = DIST_INDEX,
        sources: BackendSources | None = None,
    ) -> None:
        self.repo = Path(repo)
        self.dist_index = Path(dist_index)
        self.boot_commit = git_head(self.repo)
        self.started_at = time.time()
        self.pid = os.getpid()
        self.sources = sources if sources is not None else BackendSources(
            self.repo / PACKAGE_ROOT.name, rel_to=self.repo)

    def payload(self) -> dict[str, Any]:
        head = git_head(self.repo)
        changed = self.sources.changed()
        return {
            "boot_commit": self.boot_commit,
            "boot_short": _short(self.boot_commit),
            "head_commit": head,
            "head_short": _short(head),
            # A loaded backend file changed on disk: restart to load it.
            "behind": bool(changed),
            "changed_files": changed[:MAX_CHANGED_LISTED],
            "changed_count": len(changed),
            # Names the whole changed set (dismissal key): None when unchanged.
            "changed_digest": _set_digest(changed),
            "dirty": git_dirty(self.repo),
            "started_at": _iso(self.started_at),
            "pid": self.pid,
            "dist": dist_info(self.dist_index),
        }


@functools.lru_cache(maxsize=1)
def process_tracker() -> VersionTracker:
    """The server process's tracker; its first call fixes the boot state."""
    return VersionTracker()


__all__ = [
    "CHANGED_TTL_S",
    "DIST_INDEX",
    "EXCLUDED_DIRS",
    "MAX_CHANGED_LISTED",
    "PACKAGE_ROOT",
    "REPO_ROOT",
    "BackendSources",
    "VersionTracker",
    "dist_info",
    "git_dirty",
    "git_head",
    "loaded_backend_files",
    "process_tracker",
]
