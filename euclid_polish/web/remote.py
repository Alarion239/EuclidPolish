"""Bitwarden + SSH plumbing for the FASRC tab.

Two cooperating pieces:

  * :class:`BitwardenClient` wraps the ``bw`` CLI. ``unlock(master)`` returns
    a ``BW_SESSION`` token that is then passed back to subsequent ``bw get``
    calls. The token lives only in this process's memory.

  * :class:`SSHSession` owns the SSH ControlMaster socket. The first
    ``connect()`` does a one-shot ``pexpect`` login that sends
    ``password,totp`` at the password prompt and starts the multiplexed
    socket; every later ``run`` / ``stream`` / ``rsync`` reuses the
    socket and needs no credentials.

The flow is therefore:

    bw  = BitwardenClient(); bw.unlock(master_password)
    ssh = SSHSession(...)
    pwd  = bw.get_password(item="FASRC")
    totp = bw.get_totp   (item="FASRC")
    ssh.connect(password=pwd, totp=totp)
    # All credentials are forgotten here — only the socket remains.
    ssh.run("squeue -u $USER")

The ``RemoteState`` singleton ties both together so Flask routes can
share one connection across requests.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Bitwarden
# ---------------------------------------------------------------------------

class BitwardenError(RuntimeError):
    pass


class BitwardenClient:
    """Stateless wrapper around the ``bw`` CLI.

    Holds a single in-memory session token. Caller is responsible for
    calling :meth:`lock` to discard it. Does not touch the disk.
    """

    def __init__(self, bw_path: str = "bw") -> None:
        self.bw_path = bw_path
        # Per-instance cache for is_installed() so multiple clients (and
        # tests) don't see each other's cached state.
        self._installed_cache: Optional[bool] = None
        self._session: Optional[str] = None

    # ----------------------------- public ---------------------------------

    @property
    def unlocked(self) -> bool:
        return self._session is not None

    # ``bw --version`` is a 50–200 ms subprocess. It's called on every
    # page render via STATE.public_status — that's a noticeable fraction
    # of the page-load latency. The result is monotone for a given
    # ``bw_path`` (once installed, always installed during this process)
    # so we cache it per instance for the lifetime of the client.
    def is_installed(self) -> bool:
        if self._installed_cache is not None:
            return self._installed_cache
        try:
            subprocess.run([self.bw_path, "--version"],
                           check=True, capture_output=True, timeout=5)
            ok = True
        except (FileNotFoundError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired):
            ok = False
        self._installed_cache = ok
        return ok

    def status(self) -> dict:
        """Returns the parsed JSON from ``bw status`` (no session needed)."""
        try:
            out = subprocess.run([self.bw_path, "status"],
                                 check=True, capture_output=True, timeout=10)
        except subprocess.CalledProcessError as e:
            raise BitwardenError(e.stderr.decode(errors="replace")) from e
        try:
            return json.loads(out.stdout.decode())
        except json.JSONDecodeError:
            return {"raw": out.stdout.decode(errors="replace")}

    def unlock(self, master_password: str) -> None:
        """Authenticate with the master password; cache the session token."""
        if not master_password:
            raise BitwardenError("master password required")
        if not self.is_installed():
            raise BitwardenError(
                "Bitwarden CLI (`bw`) is not installed. "
                "Install with `brew install bitwarden-cli` on macOS, "
                "`npm install -g @bitwarden/cli` elsewhere, or see "
                "https://bitwarden.com/help/cli/"
            )
        try:
            proc = subprocess.run(
                [self.bw_path, "unlock", master_password, "--raw"],
                capture_output=True, timeout=20,
            )
        except FileNotFoundError as e:
            raise BitwardenError(f"`bw` not found in PATH: {e}") from e
        if proc.returncode != 0:
            raise BitwardenError(
                proc.stderr.decode(errors="replace").strip()
                or "unlock failed"
            )
        token = proc.stdout.decode().strip()
        if not token:
            raise BitwardenError("bw unlock returned empty session")
        self._session = token

    def lock(self) -> None:
        if self._session is None:
            return
        try:
            subprocess.run([self.bw_path, "lock"], check=False,
                           capture_output=True, timeout=5,
                           env=self._env())
        finally:
            self._session = None

    def get_password(self, item: str) -> str:
        return self._get("password", item)

    def get_username(self, item: str) -> str:
        return self._get("username", item)

    def get_totp(self, item: str) -> str:
        return self._get("totp", item)

    def sync(self) -> None:
        """Refresh local vault state. Skip silently if the network is unreachable."""
        try:
            subprocess.run([self.bw_path, "sync"], check=False,
                           capture_output=True, timeout=15,
                           env=self._env())
        except subprocess.TimeoutExpired:
            pass

    # ---------------------------- internal --------------------------------

    def _env(self) -> dict:
        env = os.environ.copy()
        if self._session:
            env["BW_SESSION"] = self._session
        return env

    def _get(self, field: str, item: str) -> str:
        if not self.unlocked:
            raise BitwardenError("vault is locked — call unlock() first")
        proc = subprocess.run(
            [self.bw_path, "get", field, item],
            capture_output=True, timeout=15, env=self._env(),
        )
        if proc.returncode != 0:
            raise BitwardenError(
                f"bw get {field} {item}: "
                + (proc.stderr.decode(errors="replace").strip()
                   or "unknown error")
            )
        return proc.stdout.decode().strip()


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
    """Owns the ControlMaster socket — first connect authenticates with
    ``password,totp`` via pexpect; subsequent calls go through the
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

    def connect(self, password: str, totp: str, timeout: int = 30) -> None:
        """Open the ControlMaster session, handling any prompt order.

        FASRC offers two flavours of interactive auth:

          * Password + 2FA: prompts ``Password:`` then ``VerificationCode:``.
          * SSH key + 2FA:  no password prompt, only ``VerificationCode:``.

        The two prompts also have inconsistent spacing in real life — the
        live FASRC banner is ``VerificationCode:`` (no space). The loop
        below recognises either order and sends the *right* secret to
        the *right* prompt instead of a combined ``password,totp`` string.
        """
        if self.is_connected():
            return
        # Defer pexpect import so the rest of the web app still works on
        # machines where it isn't installed yet.
        try:
            import pexpect
        except ImportError as e:
            raise SSHError(
                "pexpect is required for interactive SSH auth — "
                "install it with `pip install pexpect` or `mamba install pexpect`"
            ) from e

        # Make sure no stale socket is lying around from a prior crash.
        if os.path.exists(self.cfg.socket):
            try:
                os.unlink(self.cfg.socket)
            except OSError:
                pass

        args = [
            "ssh",
            "-M", "-S", self.cfg.socket,
            "-o", f"ControlPersist={self.cfg.control_persist}",
            "-o", "BatchMode=no",
            "-o", "NumberOfPasswordPrompts=1",
            "-o", "StrictHostKeyChecking=accept-new",
            self.cfg.target,
            "echo __EUCLID_POLISH_CONNECTED__",
        ]
        child = pexpect.spawn(
            args[0], args=args[1:], timeout=timeout, encoding="utf-8",
        )

        pat_password = r"(?i)password\s*:"
        # Match ``VerificationCode:`` / ``Verification Code:`` / ``OTP:`` /
        # ``Two-factor code:`` / etc. — spacing varies between sshd versions.
        pat_verify   = (r"(?i)(verification\s*code|verification|two[- ]?factor"
                        r"|one[- ]?time|otp|duo passcode)\s*:")
        pat_ok       = r"__EUCLID_POLISH_CONNECTED__"
        pat_denied   = r"(?i)(permission denied|authentication failed|"
        pat_denied  += r"access denied|access denied,)"

        sent_password = False
        sent_totp     = False
        try:
            for _ in range(6):                    # ≤ 2 prompts in any order; cap loops anyway
                idx = child.expect([
                    pat_password,
                    pat_verify,
                    pat_ok,
                    pat_denied,
                    pexpect.EOF,
                ], timeout=timeout)
                if idx == 0:
                    if sent_password:
                        raise SSHError("FASRC re-prompted for password — likely wrong")
                    child.sendline(password)
                    sent_password = True
                elif idx == 1:
                    if sent_totp:
                        raise SSHError("FASRC re-prompted for verification code — "
                                       "likely wrong or expired (TOTP cycles every 30s)")
                    child.sendline(totp)
                    sent_totp = True
                elif idx == 2:                    # __EUCLID_POLISH_CONNECTED__ seen
                    break
                elif idx == 3:
                    raise SSHError("FASRC refused credentials — "
                                   + (child.before or "").strip())
                elif idx == 4:                    # EOF
                    raise SSHError("ssh closed unexpectedly: "
                                   + (child.before or "").strip())
            else:
                raise SSHError("auth handshake didn't converge")

            child.expect(pexpect.EOF, timeout=timeout)
        finally:
            try:
                child.close(force=True)
            except Exception:
                pass

        # Sanity-check: did ControlMaster actually persist?
        if not self.is_connected():
            raise SSHError(
                "ssh appeared to succeed but the ControlMaster socket is gone"
            )

    # ----------------------------- exec -----------------------------------

    def run(self, cmd: str, timeout: int = 60) -> Tuple[int, str, str]:
        """Run a one-shot command via the multiplexed connection.

        Returns (returncode, stdout, stderr). Caller decides how to handle
        non-zero rc — we never raise on shell failure.
        """
        if not self.is_connected():
            raise SSHError("not connected")
        with self._lock:
            r = subprocess.run(
                ["ssh", "-S", self.cfg.socket, self.cfg.target, cmd],
                capture_output=True, timeout=timeout,
            )
        return (r.returncode,
                r.stdout.decode("utf-8", errors="replace"),
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


# ---------------------------------------------------------------------------
# Process-global state
# ---------------------------------------------------------------------------

class RemoteState:
    """Singleton: one BW client + one SSH session, shared across Flask requests."""

    def __init__(self) -> None:
        self.bw: BitwardenClient = BitwardenClient()
        self.ssh: Optional[SSHSession] = None
        self.connected_at: Optional[float] = None

    def public_status(self) -> dict:
        ssh_ok = bool(self.ssh and self.ssh.is_connected())
        return {
            "bw_installed":   self.bw.is_installed(),
            "bw_unlocked":    self.bw.unlocked,
            "ssh_connected":  ssh_ok,
            "connected_at":   self.connected_at if ssh_ok else None,
            "socket":         self.ssh.cfg.socket if self.ssh else None,
        }


STATE = RemoteState()
