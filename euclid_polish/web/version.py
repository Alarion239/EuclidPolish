"""What code is this server running? (contract C3, ``GET /api/version``).

The server has no auto-reloader, so after a ``git pull`` / commit it keeps
running the code it booted with. :class:`VersionTracker` records the commit at
boot (captured once, when :func:`euclid_polish.web.app.create_app` first asks
for :func:`process_tracker`) and compares it with the checkout's live ``HEAD``
on each request, so the SPA can show a "server behind HEAD — restart" banner.
It also reports uncommitted edits to tracked files (``dirty``) and which SPA
build is being served (``static/dist/index.html`` mtime + content hash).

All probes are cheap, read-only local ``git`` calls (~10–25 ms, no optional
index locks, so polling never races a concurrent ``git commit``) and never
raise: outside a git checkout the commit fields are ``None`` and
``dirty``/``behind`` false.
"""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any

_WEB_DIR = Path(__file__).resolve().parent
REPO_ROOT = _WEB_DIR.parents[1]
DIST_INDEX = _WEB_DIR / "static" / "dist" / "index.html"

_GIT_TIMEOUT_S = 5
_SHORT = 7
_HASH_CHARS = 16


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
    """``{built_at, index_hash}`` of the served SPA build (nulls when absent)."""
    try:
        data = index.read_bytes()
        built = index.stat().st_mtime
    except OSError:
        return {"built_at": None, "index_hash": None}
    return {
        "built_at": _iso(built),
        "index_hash": hashlib.sha256(data).hexdigest()[:_HASH_CHARS],
    }


def _short(sha: str | None) -> str | None:
    return sha[:_SHORT] if sha else None


class VersionTracker:
    """Boot commit (fixed at construction) vs live ``HEAD`` of one checkout."""

    def __init__(self, repo: Path = REPO_ROOT, dist_index: Path = DIST_INDEX) -> None:
        self.repo = Path(repo)
        self.dist_index = Path(dist_index)
        self.boot_commit = git_head(self.repo)
        self.started_at = time.time()
        self.pid = os.getpid()

    def payload(self) -> dict[str, Any]:
        head = git_head(self.repo)
        return {
            "boot_commit": self.boot_commit,
            "boot_short": _short(self.boot_commit),
            "head_commit": head,
            "head_short": _short(head),
            "behind": bool(self.boot_commit and head and self.boot_commit != head),
            "dirty": git_dirty(self.repo),
            "started_at": _iso(self.started_at),
            "pid": self.pid,
            "dist": dist_info(self.dist_index),
        }


@functools.lru_cache(maxsize=1)
def process_tracker() -> VersionTracker:
    """The server process's tracker; its first call fixes the boot commit."""
    return VersionTracker()


__all__ = [
    "DIST_INDEX",
    "REPO_ROOT",
    "VersionTracker",
    "dist_info",
    "git_dirty",
    "git_head",
    "process_tracker",
]
