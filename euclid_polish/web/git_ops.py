"""Local git helpers for the web UI's Git tab.

All commands run in the project root (the cwd of the Flask process).
Each helper returns a structured dict that the page renders verbatim —
no hidden state.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any

#: A commit reference the Git tab accepts: an abbreviated or full hex hash.
_HASH_RE = re.compile(r"^[0-9a-fA-F]{4,40}$")


def _run(args: list[str], cwd: str | None = None,
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


def status() -> dict[str, Any]:
    """Branch, ahead/behind counters, dirty files, last commit summary.

    ``files`` is ``[{xy, path, orig}]``: ``path`` is exactly what
    :func:`commit` accepts back (see the working-tree comment below).
    """
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

    # Working tree state — the same NUL-separated porcelain ``commit`` matches
    # against, so every listed ``path`` (raw, unquoted; the NEW path of a
    # rename, whose source is ``orig``) can be posted back to /git/commit
    # unchanged. Untracked directories stay collapsed (``dir/``); a directory
    # path selects every file under it.
    files: list[dict[str, Any]] = [
        _file_entry(root, xy, path, orig)
        for xy, path, orig in _changed_files(root, untracked="normal")
    ]

    # Last commit on this branch.
    last = _run(
        ["git", "log", "-1", "--pretty=format:%h%x09%s%x09%cr"], cwd=root,
    )
    last_commit: dict[str, str] = {}
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


def log(n: int = 12, skip: int = 0) -> list[dict[str, str]]:
    """``n`` commits (newest first, after skipping ``skip``) as dicts:
    ``hash`` (short), ``full``, ``author``, ``subject``, ``relative``,
    ``date`` (ISO 8601)."""
    root = repo_root()
    if not root:
        return []
    r = _run(["git", "log", f"-{max(0, int(n))}", f"--skip={max(0, int(skip))}",
              "--pretty=format:%h%x09%H%x09%an%x09%cI%x09%cr%x09%s"], cwd=root)
    if r.returncode != 0:
        return []
    out: list[dict[str, str]] = []
    for line in r.stdout.splitlines():
        parts = line.split("\t", 5)
        if len(parts) == 6:
            out.append({"hash": parts[0], "full": parts[1], "author": parts[2],
                        "date": parts[3], "relative": parts[4],
                        "subject": parts[5]})
    return out


def log_page(skip: int = 0, limit: int = 50) -> dict[str, Any]:
    """One page of the history: ``{commits, total, skip, limit, has_more}``
    (``total`` = commits reachable from HEAD)."""
    skip = max(0, int(skip))
    limit = max(1, min(int(limit), 500))
    root = repo_root()
    total = 0
    if root:
        r = _run(["git", "rev-list", "--count", "HEAD"], cwd=root)
        if r.returncode == 0 and r.stdout.strip().isdigit():
            total = int(r.stdout.strip())
    commits = log(limit, skip) if root else []
    return {"commits": commits, "total": total, "skip": skip, "limit": limit,
            "has_more": skip + len(commits) < total}


def head() -> str:
    """Full hash of HEAD, or '' outside a repo / before the first commit."""
    root = repo_root()
    if not root:
        return ""
    r = _run(["git", "rev-parse", "HEAD"], cwd=root)
    return r.stdout.strip() if r.returncode == 0 else ""


def relation(local: str, remote: str) -> dict[str, Any]:
    """How a remote checkout's HEAD relates to the local ``local`` commit.

    ``relation`` is ``same``, ``remote_behind`` (``remote`` is an ancestor of
    ``local``: the remote needs a pull), ``remote_ahead`` (the local checkout
    needs a pull), ``diverged`` or ``unknown`` (a commit this repo does not
    have, e.g. before a fetch). ``ahead``/``behind`` count local commits the
    remote lacks and remote commits the local checkout lacks.
    """
    if not local or not remote or not (_HASH_RE.match(local) and _HASH_RE.match(remote)):
        return {"relation": "unknown", "ahead": None, "behind": None}
    root = repo_root()
    if not root:
        return {"relation": "unknown", "ahead": None, "behind": None}
    full = []
    for ref in (local, remote):
        r = _run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], cwd=root)
        if r.returncode != 0:
            return {"relation": "unknown", "ahead": None, "behind": None}
        full.append(r.stdout.strip())
    if full[0] == full[1]:
        return {"relation": "same", "ahead": 0, "behind": 0}
    r = _run(["git", "rev-list", "--left-right", "--count", f"{full[0]}...{full[1]}"], cwd=root)
    parts = r.stdout.split() if r.returncode == 0 else []
    ahead, behind = (int(parts[0]), int(parts[1])) if len(parts) == 2 else (None, None)
    if ahead and not behind:
        kind = "remote_behind"
    elif behind and not ahead:
        kind = "remote_ahead"
    elif ahead is None:
        kind = "unknown"
    else:
        kind = "diverged"
    return {"relation": kind, "ahead": ahead, "behind": behind}


def show(rev: str, max_chars: int = 60_000) -> dict[str, Any]:
    """One commit: metadata, ``--stat`` and the (truncated) patch."""
    if not rev or not _HASH_RE.match(rev):
        return {"ok": False, "error": "not a commit hash"}
    root = repo_root()
    if not root:
        return {"ok": False, "error": "not in a git repo"}
    meta = _run(["git", "show", "-s", "--pretty=format:%H%x09%h%x09%an%x09%ae%x09%cI%x09%s%x00%b",
                 rev, "--"], cwd=root)
    if meta.returncode != 0:
        return {"ok": False, "error": meta.stderr.strip() or "unknown commit"}
    head_part, _, body = meta.stdout.partition("\0")
    fields = head_part.split("\t", 5)
    if len(fields) != 6:
        return {"ok": False, "error": "could not parse the commit"}
    stat = _run(["git", "show", "--stat", "--format=", rev, "--"], cwd=root)
    patch = _run(["git", "show", "--format=", "--patch", rev, "--"], cwd=root, timeout=30)
    text = patch.stdout
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + f"\n\n[…truncated, {len(patch.stdout) - max_chars} more chars]"
    return {"ok": True, "full": fields[0], "hash": fields[1], "author": fields[2],
            "email": fields[3], "date": fields[4], "subject": fields[5],
            "body": body.strip(), "stat": stat.stdout.strip(), "patch": text,
            "truncated": truncated}


def _inside(root: str, path: str) -> bool:
    real = os.path.realpath(os.path.join(root, path))
    base = os.path.realpath(root)
    return real == base or real.startswith(base + os.sep)


def diff(staged: bool = False, max_chars: int = 60_000,
         path: str | None = None) -> str:
    """``git diff`` (``--cached`` when ``staged``) truncated for UI display.

    ``path`` limits it to one file or directory (taken literally); an
    untracked file's unstaged "diff" is its whole content (``--no-index``
    against ``/dev/null``). A path outside the repo gives ''.
    """
    root = repo_root()
    if not root:
        return ""
    if path is not None:
        path = path.strip()
        if not path or not _inside(root, path):
            return ""
    args = ["git", "--literal-pathspecs", "diff"]
    if staged:
        args.append("--cached")
    if path:
        args += ["--", path]
    r = _run(args, cwd=root, timeout=20)
    out = r.stdout
    if path and not staged and not out.strip() and os.path.isfile(os.path.join(root, path)):
        tracked = _run(["git", "--literal-pathspecs", "ls-files", "--error-unmatch", "--", path],
                       cwd=root)
        if tracked.returncode != 0:
            r = _run(["git", "diff", "--no-index", "--", os.devnull, path], cwd=root, timeout=20)
            out = r.stdout
    if len(out) > max_chars:
        out = (out[:max_chars]
               + f"\n\n[…truncated, {len(r.stdout) - max_chars} more chars]")
    return out


#: Any file above this is refused (unless forced): the repo must not grow
#: accidental multi-MB blobs.
MAX_COMMIT_FILE_BYTES = 10 * 1024 * 1024
#: Untracked data products above this are refused (unless forced).
MAX_UNTRACKED_BINARY_BYTES = 1024 * 1024
UNTRACKED_BINARY_SUFFIXES = (".fits", ".npy", ".zip", ".jpg", ".png")


def _changed_files(root: str, untracked: str = "all"
                   ) -> list[tuple[str, str, str | None]]:
    """``(xy, path, orig)`` of every changed file — staged or not — parsed
    from NUL-separated porcelain v1 (raw paths: no C quoting, no
    ``old -> new``). ``orig`` is the source path of a staged rename/copy (the
    entry itself reports the new path), else ``None``. ``untracked="all"``
    lists untracked files one by one; ``"normal"`` collapses a wholly
    untracked directory to ``dir/``."""
    r = _run(["git", "status", "--porcelain=v1", "-z", f"-u{untracked}"],
             cwd=root)
    if r.returncode != 0:
        return []
    entries = r.stdout.split("\0")
    out: list[tuple[str, str, str | None]] = []
    i = 0
    while i < len(entries):
        entry = entries[i]
        i += 1
        if len(entry) < 4:
            continue
        xy, path = entry[:2], entry[3:]
        orig: str | None = None
        if "R" in xy or "C" in xy:
            orig = entries[i] if i < len(entries) else None
            i += 1                      # the original path of a rename/copy
        out.append((xy, path, orig))
    return out


def _selected(changed: list[tuple[str, str, str | None]], paths: list[str] | None
              ) -> list[tuple[str, str, str | None]]:
    """The changed files covered by ``paths`` (files or directories);
    every changed file when ``paths`` is None."""
    if paths is None:
        return changed
    wanted = [p.strip().rstrip("/") for p in paths if p.strip()]
    return [entry for entry in changed
            if any(entry[1] == w or entry[1].startswith(w + "/") for w in wanted)]


def _is_new(xy: str) -> bool:
    """Not in HEAD yet: untracked (``??``) or staged as an addition (``A?``)."""
    return xy == "??" or xy[:1] == "A"


def _refusals(root: str, selected: list[tuple[str, str, str | None]]
              ) -> list[dict[str, Any]]:
    refused: list[dict[str, Any]] = []
    for xy, path, _orig in selected:
        size, reason = _guard_reason(root, xy, path)   # deletions/dirs: no size
        if reason:
            refused.append({"path": path, "size": size, "reason": reason})
    return refused


def _guard_reason(root: str, xy: str, path: str) -> tuple[int | None, str | None]:
    """``(size, reason)``: the file's size and why :func:`commit` would refuse
    it without ``force`` (``None`` when it would not)."""
    full = os.path.join(root, path)
    if not os.path.isfile(full):
        return None, None
    size = os.path.getsize(full)
    if size > MAX_COMMIT_FILE_BYTES:
        return size, "file > 10 MB"
    if (_is_new(xy) and path.lower().endswith(UNTRACKED_BINARY_SUFFIXES)
            and size > MAX_UNTRACKED_BINARY_BYTES):
        return size, "untracked binary > 1 MB"
    return size, None


def _file_entry(root: str, xy: str, path: str, orig: str | None) -> dict[str, Any]:
    """One ``status()`` file: porcelain ``xy``, the index/worktree split
    (``staged``, ``unstaged``, ``untracked``), the size and the commit
    guard's verdict (``guard``) so the UI can warn before committing."""
    untracked = xy == "??"
    size, guard = _guard_reason(root, xy, path)
    return {"xy": xy, "path": path, "orig": orig,
            "staged": not untracked and xy[:1] not in (" ", "?"),
            "unstaged": untracked or xy[1:2] not in (" ", ""),
            "untracked": untracked, "size": size, "guard": guard}


def stage(paths: list[str]) -> dict[str, Any]:
    """``git add -A`` exactly the changed files covered by ``paths``."""
    root = repo_root()
    if not root:
        return {"ok": False, "error": "not in a git repo"}
    wanted = [p for p in paths if p.strip()]
    if not wanted:
        return {"ok": False, "code": "no_selection", "error": "choose the files to stage"}
    selected = _selected(_changed_files(root), wanted)
    if not selected:
        return {"ok": False, "code": "nothing_selected",
                "error": "no changed files match the selection"}
    staged = [path for _xy, path, _orig in selected]
    r = _run(["git", "--literal-pathspecs", "add", "-A", "--", *staged], cwd=root, timeout=60)
    if r.returncode != 0:
        return {"ok": False, "error": f"git add failed: {r.stderr.strip()}"}
    return {"ok": True, "staged": staged}


def unstage(paths: list[str]) -> dict[str, Any]:
    """Take ``paths`` out of the index (``git restore --staged``); the
    working-tree edits are kept."""
    root = repo_root()
    if not root:
        return {"ok": False, "error": "not in a git repo"}
    wanted = [p.strip().rstrip("/") for p in paths if p.strip()]
    if not wanted:
        return {"ok": False, "code": "no_selection", "error": "choose the files to unstage"}
    if not all(_inside(root, p) for p in wanted):
        return {"ok": False, "error": "path outside the repository"}
    r = _run(["git", "--literal-pathspecs", "restore", "--staged", "--", *wanted],
             cwd=root, timeout=60)
    if r.returncode != 0:
        return {"ok": False, "error": r.stderr.strip() or "git restore failed"}
    return {"ok": True, "unstaged": wanted}


def commit(message: str, paths: list[str] | None = None, *,
           all_files: bool = False, force: bool = False) -> dict[str, Any]:
    """Commit exactly the chosen files — nothing else in the index.

    ``paths`` (files or directories) or ``all_files=True`` must say what to
    commit — there is no implicit ``git add -A``. The selection is staged
    and committed with ``git commit --only -- <selection>``, so a file
    staged earlier but outside ``paths`` stays staged and is NOT committed
    (with ``all_files`` every changed file, staged or not, is the selection
    and the fully staged index is committed as is — which also concludes a
    merge). A staged rename brings its source path along. Files above
    10 MB and new (untracked or staged-new) ``.fits/.npy/.zip/.jpg/.png``
    above 1 MB are refused (``code="refused_files"`` with the ``refused``
    list) unless ``force``. Pathspecs are literal (no glob magic).
    """
    root = repo_root()
    if not root:
        return {"ok": False, "error": "not in a git repo"}
    if not message.strip():
        return {"ok": False, "error": "empty commit message"}
    if not paths and not all_files:
        return {"ok": False, "code": "no_selection",
                "error": "choose the files to commit (paths) or pass all=1"}

    selected = _selected(_changed_files(root), None if all_files else paths)
    if not selected:
        return {"ok": False, "code": "nothing_selected",
                "error": "no changed files match the selection"}
    refused = [] if force else _refusals(root, selected)
    if refused:
        return {"ok": False, "code": "refused_files", "refused": refused,
                "error": (f"{len(refused)} file(s) are too large or untracked "
                          "binaries; commit them deliberately with force=1")}

    staged = [path for _xy, path, _orig in selected]
    add = _run(["git", "--literal-pathspecs", "add", "-A", "--", *staged],
               cwd=root, timeout=60)
    if add.returncode != 0:
        return {"ok": False, "error": f"git add failed: {add.stderr.strip()}"}

    if all_files:
        # Every change is now staged, so the whole index IS the selection;
        # a plain commit (no pathspec) can also conclude a merge, which git
        # refuses to do as a partial ``--only`` commit.
        args = ["git", "commit", "-m", message]
    else:
        pathspec = list(dict.fromkeys(
            [*staged, *[orig for _xy, _path, orig in selected if orig]]))
        args = ["git", "--literal-pathspecs", "commit", "-m", message,
                "--only", "--", *pathspec]
    c = _run(args, cwd=root, timeout=60)
    if c.returncode != 0:
        return {"ok": False,
                "error": (c.stderr.strip() or c.stdout.strip()
                          or "git commit failed")}
    return {"ok": True, "stdout": c.stdout.strip(), "committed": staged}


def push() -> dict[str, Any]:
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


def pull() -> dict[str, Any]:
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


def fetch() -> dict[str, Any]:
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
