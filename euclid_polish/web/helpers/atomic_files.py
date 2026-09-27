"""Atomic file writes with collision-free temporary names.

Every web job runs as a thread of the one server process, so a temp name
built from ``os.getpid()`` is shared by two concurrent jobs of the same kind
(two clicks on the same tile, a mosaic fill racing a pair download) and one
job's half-written file can replace the other's. :func:`temporary_sibling`
creates the temp file with :func:`tempfile.mkstemp` in the destination's own
directory (so the final :func:`os.replace` is atomic on one filesystem) and
removes it unless it was moved into place. mkstemp creates the file 0600
and ``os.replace`` keeps that inode, so the temp is chmod-ed to the normal
``0666 & ~umask`` first: replaced files stay group/world-readable like any
other write (the tracking mirror copies permissions to the group share).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


def _read_umask() -> int:
    umask = os.umask(0)
    os.umask(umask)
    return umask


#: The process umask, read once at import (single-threaded then): reading it
#: needs a brief ``umask(0)``, which is process-wide and would race files
#: other server threads create.
_FILE_MODE = 0o666 & ~_read_umask()


@contextlib.contextmanager
def temporary_sibling(path: Path | str, suffix: str = ".tmp") -> Iterator[Path]:
    """A new, unique, empty temp file next to ``path`` (hidden, ``suffix``).

    The caller writes it and ``os.replace``-s it onto ``path``; whatever is
    left at exit (an error, or an unused temp) is deleted. ``suffix`` keeps
    the extension writers infer the format from (``.fits``, ``.npy``).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=suffix,
                                    dir=target.parent)
    try:
        os.fchmod(handle, _FILE_MODE)
    finally:
        os.close(handle)
    temporary = Path(name)
    try:
        yield temporary
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def write_text(path: Path | str, text: str) -> None:
    """Write ``text`` (UTF-8) to ``path`` atomically."""
    target = Path(path)
    with temporary_sibling(target) as temporary:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, target)


def write_json(path: Path | str, payload: Mapping[str, Any], **dumps: Any) -> None:
    """Write ``payload`` as JSON to ``path`` atomically (``dumps`` → json.dumps)."""
    write_text(path, json.dumps(payload, **dumps) + "\n")


__all__ = ["temporary_sibling", "write_json", "write_text"]
