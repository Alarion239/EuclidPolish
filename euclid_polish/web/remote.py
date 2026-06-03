"""SSH ControlMaster plumbing for the FASRC tab.

Single responsibility: open an SSH ControlMaster socket to the FASRC
login node using public-key authentication, then reuse the multiplexed
socket for every subsequent command (``run`` / ``stream`` / ``rsync``).

There is no interactive auth, no password, no OTP, no Bitwarden. If
your key isn't installed on FASRC the connect will fail with whatever
``ssh`` says (usually ``Permission denied (publickey).``); install your
key once with:

    ssh-copy-id <user>@login.rc.fas.harvard.edu

…and the next ``Connect`` will succeed without ceremony. From outside
Harvard's network FASRC may still ask for a 2FA verification code —
in that case ``ssh`` will prompt on its own terminal; running the
Flask app from a terminal lets you type the code into stdin.

The ``RemoteState`` singleton holds one ``SSHSession`` shared across
Flask requests.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# SSH ControlMaster session
# ---------------------------------------------------------------------------

@dataclass
class SSHConfig:
    user:            str
    host:            str
    socket:          str = "/tmp/euclid-polish-fasrc.sock"
    control_persist: str = "8h"

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


class SSHError(RuntimeError):
    pass


class SSHSession:
    """Owns the ControlMaster socket. Uses public-key auth — first
    ``connect()`` opens the socket, every later call goes through the
    multiplexed channel with no auth at all.
    """

    # ``ssh -O check`` is a 50–500 ms subprocess (worst case 5 s on a
    # hung ControlMaster). Called on every public_status() request and
    # at the top of every other route. Cache its result for a short TTL
    # so a burst of requests (page render + sidebar poll + status route)
    # only fires one check.
    _CONNECTED_TTL_S: float = 2.0

    def __init__(self, cfg: SSHConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        # Per-instance is_connected() cache so each SSHSession has its
        # own (test isolation; otherwise a class-level tuple leaks).
        self._connected_cache: tuple[float, bool] = (0.0, False)

    # ----------------------------- status ---------------------------------

    def is_connected(self) -> bool:
        """``ssh -O check`` against the socket; True if the master is alive.

        Result is cached for ``_CONNECTED_TTL_S`` seconds so rapid-fire
        page loads + sidebar polling don't fire one subprocess each.
        """
        now = time.monotonic()
        cached_at, cached_ok = self._connected_cache
        if now - cached_at < self._CONNECTED_TTL_S:
            return cached_ok
        if not os.path.exists(self.cfg.socket):
            self._connected_cache = (now, False)
            return False
        try:
            r = subprocess.run(
                ["ssh", "-S", self.cfg.socket, "-O", "check", self.cfg.target],
                capture_output=True, timeout=5,
            )
            ok = r.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
        self._connected_cache = (now, ok)
        return ok

    def _invalidate_connection_cache(self) -> None:
        self._connected_cache = (0.0, False)

    def disconnect(self) -> None:
        """Close the master socket cleanly."""
        try:
            subprocess.run(
                ["ssh", "-S", self.cfg.socket, "-O", "exit", self.cfg.target],
                capture_output=True, timeout=5,
            )
        except subprocess.TimeoutExpired:
            pass
        # Best-effort cleanup if -O exit didn't remove it.
        if os.path.exists(self.cfg.socket):
            try:
                os.unlink(self.cfg.socket)
            except OSError:
                pass
        # Forget the cached "connected" answer so the next is_connected()
        # call re-probes the (now-missing) socket.
        self._invalidate_connection_cache()

    # ----------------------------- connect --------------------------------

    def connect(self, timeout: int = 30) -> None:
        """Open the ControlMaster session via public-key auth.

        Uses ``ssh -M -f -N``: ``-f`` forks to the background only
        *after* authentication completes, and ``-N`` says "no remote
        command, I just want the master open". The combination means
        the foreground ssh exits 0 exactly when the socket is ready —
        no race with ControlPersist trying to fork after a quick
        remote command, which is why ``ssh -M ... target true`` used
        to return "success" before the socket existed.

        ``BatchMode=yes`` makes ssh fail fast (no prompts) instead of
        hanging if the key isn't installed — the caller gets a clean
        error and we never wait on a stdin that isn't there. From
        outside Harvard's network FASRC may still demand a 2FA prompt
        even with a valid key, and BatchMode=yes will refuse it; in
        that case do the connect once from a Harvard-network session
        (or VPN) so ControlPersist holds the master open.
        """
        if self.is_connected():
            return

        # Make sure no stale socket is lying around from a prior crash.
        if os.path.exists(self.cfg.socket):
            try:
                os.unlink(self.cfg.socket)
            except OSError:
                pass

        args = [
            "ssh",
            "-M", "-S", self.cfg.socket,
            "-f", "-N",
            "-o", f"ControlPersist={self.cfg.control_persist}",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            self.cfg.target,
        ]
        try:
            r = subprocess.run(args, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise SSHError(
                f"ssh to {self.cfg.target} timed out after {timeout} s"
            ) from e
        if r.returncode != 0:
            err = (r.stderr.decode("utf-8", errors="replace").strip()
                   or r.stdout.decode("utf-8", errors="replace").strip()
                   or f"ssh exit {r.returncode}")
            # By far the most common failure here is "key not installed
            # on FASRC". Surface the actual ssh message but also hint
            # at the one-shot fix the user can run from any terminal.
            raise SSHError(
                f"{err}\n\nIf this says 'Permission denied (publickey)', "
                f"install your key on FASRC with:\n"
                f"    ssh-copy-id {self.cfg.target}"
            )

        # Sanity-check: did ControlMaster actually persist? With -f -N
        # the socket should exist by the time ssh returns 0; if it
        # doesn't, ControlPersist or a stale socket cleanup raced us.
        if not self.is_connected():
            raise SSHError(
                "ssh appeared to succeed but the ControlMaster socket is gone"
            )

    # ----------------------------- exec -----------------------------------

    def run(self, cmd: str, timeout: int = 60,
            binary: bool = False) -> Tuple[int, Any, str]:
        """Run a one-shot command via the multiplexed connection.

        Returns (returncode, stdout, stderr). Caller decides how to handle
        non-zero rc — we never raise on shell failure.

        ``binary=True`` returns stdout as raw bytes (no UTF-8 decode) for
        callers that need to ferry binary blobs (tarballs, etc.) over
        the multiplexed link. stderr is always decoded as UTF-8 since
        SLURM / SSH error messages are text.
        """
        if not self.is_connected():
            raise SSHError("not connected")
        with self._lock:
            r = subprocess.run(
                ["ssh", "-S", self.cfg.socket, self.cfg.target, cmd],
                capture_output=True, timeout=timeout,
            )
        stdout = r.stdout if binary else r.stdout.decode("utf-8", errors="replace")
        return (r.returncode, stdout,
                r.stderr.decode("utf-8", errors="replace"))

    def run_check(self, cmd: str, timeout: int = 60) -> str:
        """Like :meth:`run` but raises :class:`SSHError` on non-zero rc."""
        rc, out, err = self.run(cmd, timeout=timeout)
        if rc != 0:
            raise SSHError(f"`{cmd}` exit {rc}: {err.strip() or out.strip()}")
        return out

    def stream(self, cmd: str) -> Iterator[str]:
        """Yield stdout lines from ``cmd`` as they arrive (for ``tail -f``).

        The caller is responsible for terminating the generator (close it
        to send SIGTERM to the remote process).
        """
        if not self.is_connected():
            raise SSHError("not connected")
        proc = subprocess.Popen(
            ["ssh", "-S", self.cfg.socket, self.cfg.target, cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                yield line.rstrip("\n")
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                proc.kill()

    # ----------------------------- file xfer ------------------------------

    def rsync_pull(self, remote_path: str, local_dir: str,
                   extra_args: Optional[List[str]] = None,
                   timeout: int = 600) -> Tuple[int, str, str]:
        """Mirror ``remote_path`` → ``local_dir`` via the ControlMaster.

        Uses ``rsync -e "ssh -S <socket>"`` so no fresh auth is needed.
        """
        if not self.is_connected():
            raise SSHError("not connected")
        os.makedirs(local_dir, exist_ok=True)
        ssh_cmd = f"ssh -S {shlex.quote(self.cfg.socket)}"
        args = ["rsync", "-az", "--partial",
                "-e", ssh_cmd,
                *(extra_args or []),
                f"{self.cfg.target}:{remote_path}",
                local_dir.rstrip("/") + "/"]
        with self._lock:
            r = subprocess.run(args, capture_output=True, timeout=timeout)
        return (r.returncode,
                r.stdout.decode("utf-8", errors="replace"),
                r.stderr.decode("utf-8", errors="replace"))

    def rsync_push(self, local_path: str, remote_dir: str,
                   extra_args: Optional[List[str]] = None,
                   timeout: int = 900) -> Tuple[int, str, str]:
        """Mirror the *contents* of ``local_path`` → ``remote_dir`` (holylabs).

        The inverse of :meth:`rsync_pull`: pushes a local directory up
        through the ControlMaster (no fresh auth). The remote parent is
        created first (``mkdir -p``) so the very first sync into a clean
        holylabs path works. A trailing slash is forced on the source so
        the directory's contents land *inside* ``remote_dir`` rather than
        nested one level deeper.
        """
        if not self.is_connected():
            raise SSHError("not connected")
        # Create the destination up front — rsync won't mkdir more than the
        # final component, and holylabs may not have the tree yet.
        self.run(f"mkdir -p {shlex.quote(remote_dir)}", timeout=30)
        ssh_cmd = f"ssh -S {shlex.quote(self.cfg.socket)}"
        args = ["rsync", "-az", "--partial",
                "-e", ssh_cmd,
                *(extra_args or []),
                local_path.rstrip("/") + "/",
                f"{self.cfg.target}:{remote_dir.rstrip('/')}/"]
        with self._lock:
            r = subprocess.run(args, capture_output=True, timeout=timeout)
        return (r.returncode,
                r.stdout.decode("utf-8", errors="replace"),
                r.stderr.decode("utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# Process-global state
# ---------------------------------------------------------------------------

class RemoteState:
    """Singleton: one SSH session, shared across Flask requests."""

    def __init__(self) -> None:
        self.ssh: Optional[SSHSession] = None
        self.connected_at: Optional[float] = None

    def public_status(self) -> dict:
        ssh_ok = bool(self.ssh and self.ssh.is_connected())
        return {
            "ssh_connected":  ssh_ok,
            "connected_at":   self.connected_at if ssh_ok else None,
            "socket":         self.ssh.cfg.socket if self.ssh else None,
        }


STATE = RemoteState()
