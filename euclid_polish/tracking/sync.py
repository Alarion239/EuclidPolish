"""Mirror the local tracking store up to persistent FASRC holylabs storage.

The tracking folder is intentionally gitignored (it holds heavy
checkpoints / FITS / images). Durability comes from rsync-ing it to
holylabs — *not* netscratch, which is purged. This module is deliberately
free of any ``euclid_polish.web`` import so the tracking package stays
standalone; the caller passes in a connected ssh session (anything with
an ``rsync_push(local, remote_dir, timeout=…) -> (rc, out, err)`` method,
e.g. :class:`euclid_polish.web.remote.SSHSession`) and the resolved
remote directory.
"""

from __future__ import annotations

import os
from typing import Any

from euclid_polish.config import Config


def remote_tracking_dir(repo_path: str, override: str = "") -> str:
    """Holylabs destination: ``override`` if set, else ``<repo_path>/tracking``.

    ``repo_path`` is the FASRC checkout on holylabs (``FasrcConfig.repo_path``),
    so the default keeps the lab notebook beside the code it documents.
    """
    return (override or "").strip() or f"{repo_path.rstrip('/')}/tracking"


def push(ssh, remote_dir: str, *,
         local_root: str | None = None,
         timeout: int = 900) -> dict[str, Any]:
    """Best-effort push of the local tracking root → ``remote_dir`` on holylabs.

    Returns a jsonify-able dict; never raises for the ordinary "not
    connected" / "nothing to sync yet" cases so a backup is never lost just
    because the mirror was unreachable.
    """
    local_root = os.path.abspath(local_root or Config.TRACKING_DIR)
    if not os.path.isdir(local_root):
        return {"ok": False, "error": "nothing to sync yet (no tracking dir)"}
    if ssh is None or not getattr(ssh, "is_connected", lambda: False)():
        return {"ok": False, "error": "not connected to FASRC"}
    try:
        rc, out, err = ssh.rsync_push(local_root, remote_dir, timeout=timeout)
    except Exception as e:  # SSHError, timeout, …
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if rc != 0:
        return {"ok": False, "error": f"rsync exit {rc}: {err.strip()[:400]}"}
    return {"ok": True, "remote_dir": remote_dir, "local_root": local_root}
