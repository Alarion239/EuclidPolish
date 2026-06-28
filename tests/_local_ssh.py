"""Test double for :class:`euclid_polish.web.remote.SSHSession`.

We can't reliably ssh to localhost in CI (port 22 is closed on a default
macOS install). The interesting code paths in the FASRC tab are:

  * generating the right sbatch script,
  * shelling out, streaming stdout, tracking progress in sqlite,
  * parsing squeue output,
  * rsync'ing files back.

None of those need a real network hop. ``LocalSSHSession`` keeps the
same public surface as ``SSHSession`` but executes every command with
``bash -c`` in a per-test temp directory, so we get full coverage of
the dispatch / parsing logic without leaving the box.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class FakeSSHConfig:
    user:            str = "tester"
    host:            str = "localhost"
    socket:          str = "/tmp/fake.sock"
    control_persist: str = "8h"

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


class LocalSSHSession:
    """Runs every command locally in ``cwd`` with the test's PATH overrides."""

    def __init__(self, cwd: str, env: dict | None = None) -> None:
        self.cfg = FakeSSHConfig()
        self.cwd = cwd
        self.env = env
        self._lock = threading.Lock()
        os.makedirs(cwd, exist_ok=True)

    # ----------------------------- lifecycle ------------------------------

    def is_connected(self) -> bool:
        return True

    def connect(self, password: str, totp: str, timeout: int = 30) -> None:  # noqa: D401
        return None

    def disconnect(self) -> None:
        return None

    # ----------------------------- exec -----------------------------------

    def run(self, cmd: str, timeout: int = 60) -> tuple[int, str, str]:
        with self._lock:
            r = subprocess.run(
                ["bash", "-c", cmd],
                cwd=self.cwd, env=self._merged_env(),
                capture_output=True, timeout=timeout,
            )
        return (r.returncode,
                r.stdout.decode("utf-8", errors="replace"),
                r.stderr.decode("utf-8", errors="replace"))

    def run_check(self, cmd: str, timeout: int = 60) -> str:
        rc, out, err = self.run(cmd, timeout=timeout)
        if rc != 0:
            raise RuntimeError(f"`{cmd}` exit {rc}: {err.strip() or out.strip()}")
        return out

    def stream(self, cmd: str) -> Iterator[str]:
        # start_new_session puts bash + its children (e.g. `tail -F`) in a
        # fresh process group so we can SIGTERM the whole tree on cleanup
        # — otherwise ``tail -F`` is orphaned when bash exits and survives
        # the test, leaking file descriptors and lingering processes.
        import os
        import signal
        proc = subprocess.Popen(
            ["bash", "-c", cmd],
            cwd=self.cwd, env=self._merged_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                yield line.rstrip("\n")
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=3)
            except Exception:
                with contextlib.suppress(Exception): proc.kill()

    def rsync_pull(self, remote_path: str, local_dir: str,
                   extra_args: list[str] | None = None,
                   timeout: int = 60) -> tuple[int, str, str]:
        os.makedirs(local_dir, exist_ok=True)
        # Treat ``remote_path`` as a path inside ``self.cwd`` — same as a real
        # remote rsync, just with the transport stripped.
        src = remote_path
        if not os.path.isabs(src):
            src = os.path.join(self.cwd, src)
        args = ["rsync", "-az", "--partial", *(extra_args or []),
                src, local_dir.rstrip("/") + "/"]
        r = subprocess.run(args, capture_output=True, timeout=timeout)
        return (r.returncode,
                r.stdout.decode("utf-8", errors="replace"),
                r.stderr.decode("utf-8", errors="replace"))

    # ----------------------------- helpers --------------------------------

    def _merged_env(self) -> dict:
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        return env
