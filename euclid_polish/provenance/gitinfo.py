"""Best-effort git-commit capture for provenance records.

This lives in the provenance core (the lowest layer) on purpose: the tracking
module already has an identical ``git_commit_info`` and should, in time, import
this one — provenance must not depend on tracking.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Optional

# Project root = two levels up from this file (…/euclid_polish/provenance/).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def capture_git(repo_root: str = _PROJECT_ROOT) -> Optional[Dict[str, Any]]:
    """Current HEAD as ``{hash, short, branch, dirty}`` — ``None`` if no repo.

    Best-effort: any git failure (not a repo, git missing) returns ``None`` so
    provenance never hard-depends on version control being present.
    """
    def _git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", repo_root, *args],
            capture_output=True, text=True, timeout=10,
        )

    try:
        head = _git("rev-parse", "HEAD")
        if head.returncode != 0:
            return None
        status = _git("status", "--porcelain")
        return {
            "hash":   head.stdout.strip(),
            "short":  _git("rev-parse", "--short", "HEAD").stdout.strip(),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip(),
            "dirty":  bool(status.stdout.strip()),
        }
    except (OSError, subprocess.SubprocessError):
        return None
