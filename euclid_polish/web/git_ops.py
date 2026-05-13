"""Local git helpers for the web UI's Git tab.

All commands run in the project root (the cwd of the Flask process).
Each helper returns a structured dict that the page renders verbatim —
no hidden state.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, List, Optional


def _run(args: List[str], cwd: Optional[str] = None,
         timeout: int = 30) -> subprocess.CompletedProcess:
    # stdin=DEVNULL: when git inherits a TTY stdin from the Flask process,
    # ``git log`` with a custom format can hang waiting on input. Forcing
    # /dev/null makes every invocation strictly non-interactive.
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        stdin=subprocess.DEVNULL,
    )


def repo_root() -> str:
    """Top-level directory of the current git repo, or '' if not in one."""
    r = _run(["git", "rev-parse", "--show-toplevel"])
    return r.stdout.strip() if r.returncode == 0 else ""


def status() -> Dict[str, Any]:
    """Branch, ahead/behind counters, dirty files, last commit summary."""
    root = repo_root()
    if not root:
        return {"in_repo": False}

    # Branch + tracking.
    head_r = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    branch = head_r.stdout.strip()
    upstream_r = _run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        cwd=root,
    )
    upstream = upstream_r.stdout.strip() if upstream_r.returncode == 0 else ""

    # Ahead / behind counts (only if upstream exists).
    ahead = behind = 0
    if upstream:
        r = _run(["git", "rev-list", "--left-right", "--count",
                  f"HEAD...{upstream}"], cwd=root)
        if r.returncode == 0:
            parts = r.stdout.split()
            if len(parts) == 2:
                ahead, behind = int(parts[0]), int(parts[1])

    # Working tree state.
    files: List[Dict[str, str]] = []
    r = _run(["git", "status", "--porcelain"], cwd=root)
    if r.returncode == 0:
        for line in r.stdout.splitlines():
            if len(line) < 3:
                continue
            xy, path = line[:2], line[3:]
            files.append({"xy": xy, "path": path})

    # Last commit on this branch.
    last = _run(
        ["git", "log", "-1", "--pretty=format:%h%x09%s%x09%cr"], cwd=root,
    )
    last_commit: Dict[str, str] = {}
    if last.returncode == 0 and last.stdout.strip():
        parts = last.stdout.strip().split("\t", 2)
        if len(parts) == 3:
            last_commit = {"hash": parts[0], "subject": parts[1],
                           "relative": parts[2]}

    return {
        "in_repo":    True,
        "root":       root,
        "branch":     branch,
        "upstream":   upstream,
        "ahead":      ahead,
        "behind":     behind,
        "files":      files,
        "last":       last_commit,
        "clean":      not files,
    }


def log(n: int = 12) -> List[Dict[str, str]]:
    """Last ``n`` commits as a list of dicts."""
    root = repo_root()
    if not root:
        return []
    r = _run(["git", "log", f"-{int(n)}",
              "--pretty=format:%h%x09%an%x09%s%x09%cr"], cwd=root)
    if r.returncode != 0:
        return []
    out: List[Dict[str, str]] = []
    for line in r.stdout.splitlines():
        parts = line.split("\t", 3)
        if len(parts) == 4:
            out.append({"hash": parts[0], "author": parts[1],
                        "subject": parts[2], "relative": parts[3]})
    return out


def diff(staged: bool = False, max_chars: int = 60_000) -> str:
    """Return ``git diff`` (or ``git diff --cached``) truncated for UI display."""
    root = repo_root()
    if not root:
        return ""
    args = ["git", "diff"]
    if staged:
        args.append("--cached")
    r = _run(args, cwd=root, timeout=20)
    out = r.stdout
    if len(out) > max_chars:
        out = (out[:max_chars]
               + f"\n\n[…truncated, {len(r.stdout) - max_chars} more chars]")
    return out


def commit(message: str, paths: Optional[List[str]] = None
           ) -> Dict[str, Any]:
    """Stage ``paths`` (or everything if None) and create one commit."""
    root = repo_root()
    if not root:
        return {"ok": False, "error": "not in a git repo"}
    if not message.strip():
        return {"ok": False, "error": "empty commit message"}

    if paths:
        add = _run(["git", "add", "--", *paths], cwd=root)
    else:
        add = _run(["git", "add", "-A"], cwd=root)
    if add.returncode != 0:
        return {"ok": False, "error": f"git add failed: {add.stderr.strip()}"}

    c = _run(["git", "commit", "-m", message], cwd=root, timeout=60)
    if c.returncode != 0:
        return {"ok": False,
                "error": (c.stderr.strip() or c.stdout.strip()
                          or "git commit failed")}
    return {"ok": True, "stdout": c.stdout.strip()}


def push() -> Dict[str, Any]:
    root = repo_root()
    if not root:
        return {"ok": False, "error": "not in a git repo"}
    r = _run(["git", "push"], cwd=root, timeout=120)
    if r.returncode != 0:
        return {"ok": False,
                "error": (r.stderr.strip() or r.stdout.strip()
                          or "git push failed")}
    return {"ok": True,
            "stdout": (r.stdout + r.stderr).strip()}


def pull() -> Dict[str, Any]:
    root = repo_root()
    if not root:
        return {"ok": False, "error": "not in a git repo"}
    r = _run(["git", "pull", "--ff-only"], cwd=root, timeout=120)
    if r.returncode != 0:
        return {"ok": False,
                "error": (r.stderr.strip() or r.stdout.strip()
                          or "git pull failed")}
    return {"ok": True,
            "stdout": (r.stdout + r.stderr).strip()}


def fetch() -> Dict[str, Any]:
    """Update remote-tracking branches without merging."""
    root = repo_root()
    if not root:
        return {"ok": False, "error": "not in a git repo"}
    r = _run(["git", "fetch"], cwd=root, timeout=60)
    if r.returncode != 0:
        return {"ok": False,
                "error": (r.stderr.strip() or "git fetch failed")}
    return {"ok": True,
            "stdout": (r.stdout + r.stderr).strip() or "(up to date)"}
