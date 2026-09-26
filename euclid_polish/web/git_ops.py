"""Local git helpers for the web UI's Git tab.

All commands run in the project root (the cwd of the Flask process).
Each helper returns a structured dict that the page renders verbatim —
no hidden state.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any


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
    files: list[dict[str, str | None]] = [
        {"xy": xy, "path": path, "orig": orig}
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


def log(n: int = 12) -> list[dict[str, str]]:
    """Last ``n`` commits as a list of dicts."""
    root = repo_root()
    if not root:
        return []
    r = _run(["git", "log", f"-{int(n)}",
              "--pretty=format:%h%x09%an%x09%s%x09%cr"], cwd=root)
    if r.returncode != 0:
        return []
    out: list[dict[str, str]] = []
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
        full = os.path.join(root, path)
        if not os.path.isfile(full):
            continue                    # deletions and directories stage fine
        size = os.path.getsize(full)
        if size > MAX_COMMIT_FILE_BYTES:
            refused.append({"path": path, "size": size, "reason": "file > 10 MB"})
        elif (_is_new(xy) and path.lower().endswith(UNTRACKED_BINARY_SUFFIXES)
              and size > MAX_UNTRACKED_BINARY_BYTES):
            refused.append({"path": path, "size": size,
                            "reason": "untracked binary > 1 MB"})
    return refused


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
