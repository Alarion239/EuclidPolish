"""One-shot rsync of the ENSEMBLE checkpoints from FASRC to this laptop.

Manual only (decided in W-Ops, 2026-09-26): the periodic poller used to be
started by the training-status poll, which was deleted with the classic
console, and an automatic ``rsync --delete-after`` that silently removes
local-only files is not something to run unasked. ``POST
/api/fasrc/mirror/trigger`` (with ``confirm=1``) runs :meth:`Mirror.trigger`
as a local job; ``GET /api/fasrc/mirror/status`` reports the last run.

The mirror syncs the remote *ensemble* dir (sibling of ``cfg.ckpt_dir``) into
the local ensemble dir. The member registry lives OUTSIDE the ensemble dir
precisely so this mirror's ``--delete-after`` can't wipe it, and its archived
tombstones keep a mirrored-back member from re-activating.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from euclid_polish.ensemble_registry import default_ensemble_dir
from euclid_polish.web import fasrc_config
from euclid_polish.web.remote import STATE


def remote_ensemble_dir(cfg) -> str:
    """The ensemble dir on FASRC: sibling of the remote checkpoint dir
    (members in ``member_NN/``, as :func:`default_ensemble_dir` locally)."""
    parent = os.path.dirname(cfg.ckpt_dir.rstrip("/")) or "."
    return os.path.join(parent, "ensemble")


@dataclass
class MirrorStatus:
    last_run_at: float | None = None
    last_rc:     int | None = None
    last_error:  str = ""
    last_stdout: str = ""
    remote_dir:  str = ""
    local_dir:   str = ""


class Mirror:
    """rsync ``<remote ensemble>/`` → the local ensemble dir, on request."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status = MirrorStatus()

    def trigger(self) -> MirrorStatus:
        """One synchronous sync (serialised: a second caller waits)."""
        with self._lock:
            self._sync_once()
        return self.status

    def _sync_once(self) -> None:
        if STATE.ssh is None or not STATE.ssh.is_connected():
            self.status.last_error = "ssh not connected"
            self.status.last_run_at = time.time()
            return
        cfg = fasrc_config.load()
        remote = remote_ensemble_dir(cfg).rstrip("/") + "/"
        local  = (cfg.local_ckpt_mirror or default_ensemble_dir()).rstrip("/")
        os.makedirs(local, exist_ok=True)
        try:
            rc, out, err = STATE.ssh.rsync_pull(
                remote, local,
                extra_args=["--delete-after"],
                timeout=600,
            )
        except Exception as e:
            self.status.last_run_at = time.time()
            self.status.last_rc = -1
            self.status.last_error = f"{type(e).__name__}: {e}"
            return
        self.status.last_run_at = time.time()
        self.status.last_rc = rc
        self.status.last_error  = err.strip() if rc != 0 else ""
        self.status.last_stdout = out.strip()[-2000:]
        self.status.remote_dir  = remote
        self.status.local_dir   = local


MIRROR = Mirror()
