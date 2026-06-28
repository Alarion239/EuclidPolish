"""Time-travel sandboxes: re-run the exact code a backup was made with.

A backup records the git commit it was created at. Time-travel turns that
stamp into an isolated *parallel universe*:

  * **Code** — a ``git worktree`` checked out at the commit, so the old
    code runs from its own directory without disturbing the live working
    tree (no checkout, no stash, HEAD never moves).
  * **Data** — a sandbox ``data/`` dir where read-mostly **inputs**
    (TFRecords, PSFs, cutouts, …) are *symlinked* from the live tree and
    **outputs** (vis, inference, ckpt) are fresh dirs, so a replay reads
    real inputs but can never overwrite live results.
  * **Config + job DB** — the second server is launched with
    ``HOME=<sandbox>/home``, so the old code reads/writes a sandbox
    ``~/.euclid_polish/fasrc.json`` + jobs DB instead of the real ones.

Layout under ``Config.TIMETRAVEL_DIR`` (``./.timetravel``, gitignored)::

    <short>/
      worktree/                 # git worktree @ commit (the old code)
      data/                     # EUCLID_POLISH_DATA_DIR — symlinked inputs
      ckpt/wdsr/                # EUCLID_POLISH_CKPT_DIR — seeded w/ backup
      home/.euclid_polish/...   # HOME override
      sandbox.json              # this sandbox's metadata + runtime (pid/port)
      server.log                # the spawned WebUI's stdout/stderr

This module is intentionally free of any ``euclid_polish.web`` import: the
remote (FASRC) helper takes a duck-typed ssh session, and the caller
writes the sandbox ``fasrc.json`` from the live config it already holds.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any

from euclid_polish.config import Config

from ._utils import _now_iso, _read_json, _write_json

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
#: Public alias for the local repo root (callers push from / worktree off it).
PROJECT_ROOT = _PROJECT_ROOT

#: Read-mostly input subtrees (relative to the data dir) that are symlinked
#: into a sandbox so a replay reads live inputs without copying GBs. Anything
#: NOT listed here (vis/, euclid_inference/, …) becomes a fresh writable dir
#: in the sandbox, so outputs stay isolated.
INPUT_RELPATHS = [
    "images/records_v2",
    "images/records_v2_hst",
    "images/records_v2_euclid_roundtrip",
    "euclid_psf",
    "hst_psf",
    "hst_hlsp",
    "euclid_stars",
    "euclid_sky",
    "_fasrc_cache",
]

_FIRST_PORT = 8766


class TimeTravelError(RuntimeError):
    """Raised for invalid time-travel operations (bad commit, etc.)."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _git(repo_root: str, *args: str,
         timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", repo_root, *args],
        capture_output=True, text=True, timeout=timeout,
    )


def commit_exists(commit: str, repo_root: str = _PROJECT_ROOT) -> bool:
    """True if ``commit`` resolves to a commit object in ``repo_root``."""
    if not commit:
        return False
    r = _git(repo_root, "cat-file", "-e", f"{commit}^{{commit}}")
    return r.returncode == 0


def resolve_short(commit: str, repo_root: str = _PROJECT_ROOT) -> str:
    r = _git(repo_root, "rev-parse", "--short", commit)
    return r.stdout.strip() or commit[:7]


def timetravel_root() -> str:
    return os.path.abspath(Config.TIMETRAVEL_DIR)


def sandbox_dir(short: str) -> str:
    return os.path.join(timetravel_root(), short)


def read_sandbox(short: str) -> dict[str, Any] | None:
    return _read_json(os.path.join(sandbox_dir(short), "sandbox.json"))


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _free_port(start: int = _FIRST_PORT) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:   # nothing listening
                return port
    raise TimeTravelError("no free port found in 8766–8815")


# ---------------------------------------------------------------------------
# remote (FASRC) path conventions — pure string helpers
# ---------------------------------------------------------------------------

def remote_worktree_path(repo_path: str, short: str) -> str:
    """Sibling worktree of the holylabs checkout: ``<repo_path>-tt/<short>``."""
    return f"{repo_path.rstrip('/')}-tt/{short}"


def remote_sandbox_base(data_dir: str, short: str) -> str:
    """``<parent-of-data_dir>/sandbox/<short>`` (e.g. on netscratch)."""
    parent = os.path.dirname(data_dir.rstrip("/"))
    return f"{parent}/sandbox/{short}"


# ---------------------------------------------------------------------------
# local sandbox lifecycle
# ---------------------------------------------------------------------------

def _seed_checkpoint(seed_ckpt_dir: str, dest_ckpt: str) -> list[str]:
    """Copy a backup's checkpoint files into the sandbox ckpt dir."""
    copied: list[str] = []
    os.makedirs(dest_ckpt, exist_ok=True)
    for name in sorted(os.listdir(seed_ckpt_dir)):
        if name == "meta.json":          # tracking-internal sidecar; skip
            continue
        src = os.path.join(seed_ckpt_dir, name)
        dst = os.path.join(dest_ckpt, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        copied.append(name)
    return copied


def _symlink_inputs(live_data_dir: str, sandbox_data: str) -> list[str]:
    """Symlink each existing read-only input subtree into the sandbox."""
    linked: list[str] = []
    for rel in INPUT_RELPATHS:
        src = os.path.join(live_data_dir, rel)
        if not os.path.exists(src):
            continue
        dst = os.path.join(sandbox_data, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(src), dst)
            linked.append(rel)
    return linked


def prepare_local_sandbox(
    commit: str, *,
    repo_root: str = _PROJECT_ROOT,
    live_data_dir: str | None = None,
    seed_ckpt_dir: str | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create (or reuse) a local sandbox for ``commit`` and return its record.

    Idempotent: if a sandbox for the commit's short hash already exists it is
    returned unchanged (with ``reused=True``).
    """
    if not commit_exists(commit, repo_root):
        raise TimeTravelError(
            f"commit {commit!r} is not reachable in this repo — was the "
            "backup made on an unpushed/garbage-collected commit?"
        )
    short = resolve_short(commit, repo_root)
    root = sandbox_dir(short)
    meta_path = os.path.join(root, "sandbox.json")
    existing = _read_json(meta_path)
    if existing and os.path.isdir(os.path.join(root, "worktree")):
        existing["reused"] = True
        return existing

    live_data_dir = os.path.realpath(live_data_dir or Config.DATA_DIR)
    worktree = os.path.join(root, "worktree")
    data_dir = os.path.join(root, "data")
    ckpt_dir = os.path.join(root, "ckpt", "wdsr")
    home = os.path.join(root, "home")
    os.makedirs(root, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(home, exist_ok=True)

    # 1. git worktree at the commit (detached HEAD; old code lives here).
    if not os.path.isdir(worktree):
        r = _git(repo_root, "worktree", "add", "--detach", worktree, commit)
        if r.returncode != 0:
            raise TimeTravelError(
                f"git worktree add failed: {r.stderr.strip() or r.stdout.strip()}"
            )

    # 2. symlink read-only inputs; outputs stay fresh sandbox dirs.
    linked = _symlink_inputs(live_data_dir, data_dir)

    # 3. seed the backup's checkpoint so inference/resume uses the exact model.
    seeded: list[str] = []
    if seed_ckpt_dir and os.path.isdir(seed_ckpt_dir):
        seeded = _seed_checkpoint(seed_ckpt_dir, ckpt_dir)
    else:
        os.makedirs(ckpt_dir, exist_ok=True)

    meta = {
        "short":          short,
        "commit":         commit,
        "root":           root,
        "worktree":       worktree,
        "data_dir":       data_dir,
        "ckpt_dir":       ckpt_dir,
        "home":           home,
        "server_log":     os.path.join(root, "server.log"),
        "created_at":     _now_iso(),
        "source":         source or {},
        "linked_inputs":  linked,
        "seeded_files":   seeded,
        "remote":         None,    # filled by prepare_remote_sandbox
        "port":           None,
        "pid":            None,
        "url":            None,
        "reused":         False,
    }
    _write_json(meta_path, meta)
    return meta


def write_home_fasrc_config(short: str, cfg_dict: dict[str, Any]) -> str:
    """Write a sandbox ``~/.euclid_polish/fasrc.json`` (under the sandbox HOME).

    The caller supplies the full config dict (live config with the remote
    sandbox paths already overridden), keeping this module web-agnostic.
    """
    home = os.path.join(sandbox_dir(short), "home")
    path = os.path.join(home, ".euclid_polish", "fasrc.json")
    _write_json(path, cfg_dict)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)
    return path


def set_sandbox_remote(short: str, remote: dict[str, Any] | None) -> None:
    """Record the remote-sandbox result on a sandbox's metadata."""
    meta = read_sandbox(short)
    if meta is None:
        return
    meta["remote"] = remote
    _write_json(os.path.join(meta["root"], "sandbox.json"), meta)


def list_sandboxes() -> list[dict[str, Any]]:
    """All sandboxes (newest first), each annotated with live ``running``."""
    root = timetravel_root()
    out: list[dict[str, Any]] = []
    if os.path.isdir(root):
        for name in os.listdir(root):
            meta = _read_json(os.path.join(root, name, "sandbox.json"))
            if meta:
                meta["running"] = _pid_alive(meta.get("pid"))
                out.append(meta)
    out.sort(key=lambda m: m.get("created_at") or "", reverse=True)
    return out


# ---------------------------------------------------------------------------
# spawn / stop the second WebUI (old code)
# ---------------------------------------------------------------------------

# Version-robust launcher: import the worktree's own create_app and serve it.
# Avoids depending on the old ``main()``'s --port handling.
_SHIM = (
    "from euclid_polish.web.app import create_app; "
    "create_app().run(host='127.0.0.1', port={port}, "
    "use_reloader=False, threaded=True)"
)


def spawn_server(short: str, *, port: int | None = None,
                 wait_s: float = 25.0) -> dict[str, Any]:
    """Launch the sandbox's old code as a second WebUI; record pid + url.

    Blocks up to ``wait_s`` for the port to answer — the old code imports
    TensorFlow on boot (~10–15 s), so the default is generous enough that
    the returned URL is live by the time the caller hands it to the user.
    """
    meta = read_sandbox(short)
    if not meta:
        raise TimeTravelError(f"no sandbox {short!r}")
    if _pid_alive(meta.get("pid")):
        return {"ok": True, "already_running": True, "url": meta.get("url"),
                "port": meta.get("port"), "pid": meta.get("pid")}

    port = port or _free_port()
    env = dict(os.environ)
    env["HOME"] = meta["home"]
    env["EUCLID_POLISH_DATA_DIR"] = meta["data_dir"]
    env["EUCLID_POLISH_CKPT_DIR"] = meta["ckpt_dir"]
    # Keep the sandbox's own tracking + time-travel stores under its HOME so
    # a replay can't write into the live store or nest sandboxes.
    env["EUCLID_POLISH_TRACKING_DIR"] = os.path.join(meta["root"], "tracking")
    env["EUCLID_POLISH_TIMETRAVEL_DIR"] = os.path.join(meta["root"], ".tt")
    # Make the worktree win the import race even under an editable install.
    env["PYTHONPATH"] = meta["worktree"] + os.pathsep + env.get("PYTHONPATH", "")

    log = open(meta["server_log"], "ab")
    proc = subprocess.Popen(
        [sys.executable, "-c", _SHIM.format(port=port)],
        cwd=meta["worktree"], env=env,
        stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{port}/"
    # Wait until the port answers or the process dies.
    deadline = time.monotonic() + wait_s
    up = False
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                up = True
                break
        time.sleep(0.2)

    meta.update({"port": port, "pid": proc.pid, "url": url})
    _write_json(os.path.join(meta["root"], "sandbox.json"), meta)
    if proc.poll() is not None:
        return {"ok": False, "error": "server exited on startup — see "
                f"{meta['server_log']}", "pid": proc.pid}
    return {"ok": True, "url": url, "port": port, "pid": proc.pid,
            "responding": up}


def stop_server(short: str) -> dict[str, Any]:
    """Terminate a sandbox's running server (process group)."""
    meta = read_sandbox(short)
    if not meta:
        raise TimeTravelError(f"no sandbox {short!r}")
    pid = meta.get("pid")
    if _pid_alive(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGTERM)
    meta["pid"] = None
    meta["url"] = None
    meta["port"] = None
    _write_json(os.path.join(meta["root"], "sandbox.json"), meta)
    return {"ok": True}


def remove_sandbox(short: str, *,
                   repo_root: str = _PROJECT_ROOT) -> dict[str, Any]:
    """Stop the server, drop the git worktree, and delete the sandbox dir."""
    root = sandbox_dir(short)
    if not os.path.isdir(root):
        raise TimeTravelError(f"no sandbox {short!r}")
    with contextlib.suppress(TimeTravelError):
        stop_server(short)
    worktree = os.path.join(root, "worktree")
    if os.path.isdir(worktree):
        _git(repo_root, "worktree", "remove", "--force", worktree)
    _git(repo_root, "worktree", "prune")
    # Safety: only ever rmtree inside the time-travel root.
    if os.path.realpath(root).startswith(os.path.realpath(timetravel_root())):
        shutil.rmtree(root, ignore_errors=True)
    return {"ok": True, "removed": short}


# ---------------------------------------------------------------------------
# remote (FASRC) sandbox — over a duck-typed ssh session
# ---------------------------------------------------------------------------

def prepare_remote_sandbox(
    ssh, *,
    repo_path: str,
    data_dir: str,
    commit: str,
    short: str,
    input_relpaths: list[str] | None = None,
    push_origin_cmd: list[str] | None = None,
) -> dict[str, Any]:
    """Create a FASRC worktree + netscratch sandbox so jobs stay isolated.

    ``ssh`` must expose ``is_connected()`` and ``run(cmd, timeout=…)``.
    ``push_origin_cmd`` (if given) is a local git command run when the
    commit isn't on the remote yet (push it to origin so the cluster can
    fetch). Returns the resolved remote paths (jsonify-able) or an error.
    """
    if ssh is None or not getattr(ssh, "is_connected", lambda: False)():
        return {"ok": False, "error": "not connected to FASRC"}
    input_relpaths = input_relpaths or INPUT_RELPATHS
    wt = remote_worktree_path(repo_path, short)
    base = remote_sandbox_base(data_dir, short)
    r_data = f"{base}/data"
    r_ckpt = f"{base}/ckpt/wdsr"

    # 1. Make sure the commit is on the remote (fetch; push from local first
    #    if it still can't be found and a push command was provided).
    rc, _o, _e = ssh.run(
        f"cd {repo_path} && git cat-file -e {commit}^{{commit}}", timeout=30)
    if rc != 0:
        if push_origin_cmd is not None:
            try:
                subprocess.run(push_origin_cmd, check=True,
                               capture_output=True, timeout=120)
            except subprocess.SubprocessError as e:
                return {"ok": False,
                        "error": f"commit not on FASRC and push failed: {e}"}
        ssh.run(f"cd {repo_path} && git fetch --all --quiet", timeout=120)
        rc, _o, e2 = ssh.run(
            f"cd {repo_path} && git cat-file -e {commit}^{{commit}}",
            timeout=30)
        if rc != 0:
            return {"ok": False,
                    "error": f"commit {commit[:10]} not reachable on FASRC; "
                             "push it to origin first"}

    # 2. Remote worktree at the commit (idempotent).
    rc, _o, err = ssh.run(
        f"cd {repo_path} && (test -d {wt} || "
        f"git worktree add --detach {wt} {commit})", timeout=120)
    if rc != 0:
        return {"ok": False, "error": f"remote worktree add failed: {err.strip()}"}

    # 3. netscratch sandbox: symlink inputs, leave outputs writable.
    mk = [f"mkdir -p {r_data} {r_ckpt}"]
    for rel in input_relpaths:
        src = f"{data_dir.rstrip('/')}/{rel}"
        dst = f"{r_data}/{rel}"
        mk.append(
            f"if [ -e {src} ]; then mkdir -p $(dirname {dst}); "
            f"[ -e {dst} ] || ln -s {src} {dst}; fi")
    rc, _o, err = ssh.run(" && ".join(mk), timeout=120)
    if rc != 0:
        return {"ok": False, "error": f"remote sandbox setup failed: {err.strip()}"}

    return {"ok": True, "worktree": wt, "data_dir": r_data,
            "ckpt_dir": r_ckpt, "base": base}
