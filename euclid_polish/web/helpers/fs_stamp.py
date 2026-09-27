"""Cheap on-disk stamps for memoising listings and summaries.

A stamp is a tuple of ``(name, mtime_ns, size)`` over the files and
directories a computation reads. A directory's mtime moves whenever an entry
is created, removed or atomically replaced (``os.replace``) inside it, so a
stamp that reaches the directories a reader lists sees every new or
rewritten file without reading any of them. Memoising on a stamp (instead of
a short timer) keeps warm requests warm however long the server idles, and
still sees every change on the next request.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path


def stat_key(path: Path | str, name: str | None = None) -> tuple:
    """``(name, mtime_ns, size)`` of one path (``None``s when missing)."""
    label = os.fspath(path) if name is None else name
    try:
        stat = os.stat(path)
    except OSError:
        return (label, None, None)
    return (label, stat.st_mtime_ns, stat.st_size)


def tree_stamp(root: Path | str, depth: int) -> tuple:
    """:func:`stat_key` of ``root`` (full path) and of every entry up to
    ``depth`` levels below it (relative names; directories are descended;
    sorted, so the stamp is stable)."""
    out = [stat_key(root)]
    frontier: list[tuple[str, str]] = [(os.fspath(root), "")]
    for _level in range(depth):
        next_frontier: list[tuple[str, str]] = []
        for directory, prefix in frontier:
            try:
                children = sorted(os.scandir(directory), key=lambda entry: entry.name)
            except OSError:
                continue
            for child in children:
                name = f"{prefix}{child.name}"
                out.append(stat_key(child.path, name))
                with contextlib.suppress(OSError):
                    if child.is_dir():
                        next_frontier.append((child.path, name + "/"))
        frontier = next_frontier
    return tuple(out)


__all__ = ["stat_key", "tree_stamp"]
