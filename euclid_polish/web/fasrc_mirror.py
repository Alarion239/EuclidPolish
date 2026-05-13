"""Background poller that rsync's new checkpoint files from FASRC.

One mirror thread per Flask process; ``start()`` / ``stop()`` toggle it.
We don't try to detect "new" files ourselves — rsync's incremental
transfer handles that natively. We just call it on a timer.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

from euclid_polish.config import Config
from euclid_polish.web import fasrc_config
from euclid_polish.web.remote import STATE


@dataclass
class MirrorStatus:
    enabled:     bool = False
    last_run_at: Optional[float] = None
    last_rc:     Optional[int] = None
    last_error:  str = ""
    last_stdout: str = ""
    remote_dir:  str = ""
    local_dir:   str = ""


class Mirror:
    """Periodic rsync from ``cfg.ckpt_dir`` on FASRC → local mirror dir."""

    def __init__(self, period_seconds: int = 60) -> None:
        self.period = max(15, int(period_seconds))
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self.status = MirrorStatus()

    # ------------------------------ control -------------------------------

    def start(self) -> MirrorStatus:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return self.status
            self._stop_evt.clear()
            cfg = fasrc_config.load()
            self.status.enabled = True
            self.status.remote_dir = cfg.ckpt_dir
            self.status.local_dir  = (cfg.local_ckpt_mirror or
                                      Config.DEFAULT_CHECKPOINT_DIR)
            self._thread = threading.Thread(
                target=self._loop, name="fasrc-mirror", daemon=True,
            )
            self._thread.start()
        return self.status

    def stop(self) -> MirrorStatus:
        self._stop_evt.set()
        with self._lock:
            self.status.enabled = False
        return self.status

    def trigger(self) -> MirrorStatus:
        """One-shot sync (does not change the periodic schedule)."""
        self._sync_once()
        return self.status

    # ------------------------------ loop ----------------------------------

    def _loop(self) -> None:
        # First sync immediately so the UI feels responsive.
        self._sync_once()
        while not self._stop_evt.wait(self.period):
            self._sync_once()

    def _sync_once(self) -> None:
        if STATE.ssh is None or not STATE.ssh.is_connected():
            self.status.last_error = "ssh not connected"
            self.status.last_run_at = time.time()
            return
        cfg = fasrc_config.load()
        remote = cfg.ckpt_dir.rstrip("/") + "/"
        local  = (cfg.local_ckpt_mirror or Config.DEFAULT_CHECKPOINT_DIR)
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


MIRROR = Mirror(period_seconds=60)
