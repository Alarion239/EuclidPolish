"""
Pull-on-demand file fetcher for the FASRC remote tree.

The UI's HST PSF / HST cutouts tabs need to read FITS files that live
on FASRC netscratch, not locally. We rsync them through the existing
SSH ControlMaster into ``data/_fasrc_cache/`` and route subsequent
inspector views through the cached path.

Safeguards (informed by FASRC's documented best practices — see
https://docs.rc.fas.harvard.edu/kb/transferring-data-on-the-cluster/,
https://docs.rc.fas.harvard.edu/kb/rsync/, https://docs.rc.fas.harvard.edu/kb/faq/):

  * **Size cap** (default 50 MB) — anything bigger is refused with an
    explanation. Big-file transfer should go through Globus or a
    compute-node-mediated batch script, not a login-node rsync.
  * **Cache-first**: the same file pulled within ``cache_ttl`` (default
    5 min) is served from disk without re-rsync-ing.
  * **Single-flight**: a per-path lock prevents concurrent pulls of the
    same file from competing for SSH.
  * **LRU eviction**: when the cache exceeds ``max_cache_bytes``
    (default 2 GB) we delete the oldest files.
  * **Allowed roots**: pulls are restricted to known data dirs on the
    remote (``data_dir``, ``ckpt_dir``, ``repo_path/logs``). Anything
    else gets a 403.

The web layer NEVER polls this in the background — every call is user-
initiated by clicking a thumbnail or a "fetch from FASRC" button.
"""

from __future__ import annotations

import os
import shlex
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from euclid_polish.config import Config
from euclid_polish.web import fasrc_config

from euclid_polish.web.remote import STATE


# Local on-disk cache. Lives under the project ``data/`` so the existing
# ``/inspect`` page (which whitelists subtrees of ``data/``) can serve it
# without further config.
CACHE_SUBDIR  = "_fasrc_cache"
CACHE_DIR     = os.path.join(Config.DATA_DIR, CACHE_SUBDIR)


# ---------------------------------------------------------------------------
# Configuration knobs (module-level constants — change in code if needed)
# ---------------------------------------------------------------------------

# Hard cap on any single pull. 50 MB keeps small FITS (PSFs, kernels, star
# cutouts, log tails) inside the budget and refuses HLSP tiles, full
# TFRecords, etc.
MAX_PULL_BYTES   = 50 * 1024 * 1024

# How long a cached file is considered fresh. Clicks within this window
# serve from disk and don't touch SSH. 5 minutes balances "see updates
# from a just-finished sbatch" against "don't re-pull the same PSF every
# refresh."
CACHE_TTL_SECONDS = 300

# Cap on the cache directory. Cheap LRU eviction kicks in when this is
# exceeded — oldest mtime first. Bigger files are evicted first within
# the oldest bucket so a single small file doesn't push everything out.
MAX_CACHE_BYTES   = 2 * 1024 * 1024 * 1024     # 2 GB


# ---------------------------------------------------------------------------
# Safety: which remote paths are reachable
# ---------------------------------------------------------------------------

def allowed_remote_roots() -> List[str]:
    """Real prefixes a fetched path must live under on the remote.

    Computed from the persisted FASRC config so a user changing
    ``data_dir`` / ``ckpt_dir`` / ``repo_path`` doesn't strand the
    fetcher.
    """
    cfg = fasrc_config.load()
    roots = []
    for p in (cfg.data_dir, cfg.ckpt_dir,
              os.path.join(cfg.repo_path, "logs")):
        if p:
            # The remote path is just a string here — we have no way to
            # resolve symlinks remotely cheaply, so we treat the
            # user-configured path as canonical. Anything not under one
            # of these prefixes is rejected.
            roots.append(p.rstrip("/"))
    return roots


def is_allowed_remote_path(remote_path: str) -> bool:
    """True iff ``remote_path`` is under one of :func:`allowed_remote_roots`."""
    if not remote_path or ".." in remote_path.split("/"):
        return False
    rp = remote_path.rstrip("/")
    for root in allowed_remote_roots():
        if rp == root or rp.startswith(root + "/"):
            return True
    return False


# ---------------------------------------------------------------------------
# Local cache layout
# ---------------------------------------------------------------------------

def _local_path_for(remote_path: str) -> str:
    """Deterministic local cache path for a given remote absolute path.

    We strip the leading ``/`` and place under ``CACHE_DIR`` so two
    different remote paths can't collide and ``os.path.realpath`` gives a
    predictable result. The path is also what the inspector page sees in
    its safety check (already whitelists subtrees of ``data/``).
    """
    rel = remote_path.lstrip("/")
    return os.path.realpath(os.path.join(CACHE_DIR, rel))


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ---------------------------------------------------------------------------
# Single-flight locks per path
# ---------------------------------------------------------------------------

_LOCKS_REGISTRY: Dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    """Return a process-global lock for ``path`` (one lock per path)."""
    with _LOCKS_GUARD:
        lock = _LOCKS_REGISTRY.get(path)
        if lock is None:
            lock = threading.Lock()
            _LOCKS_REGISTRY[path] = lock
        return lock


# ---------------------------------------------------------------------------
# Result type — keep the public API small
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FetchResult:
    """Outcome of one ``fetch_one_file`` call.

    Attributes
    ----------
    ok
        True iff the cached local path exists and was either fresh or
        successfully refreshed.
    local_path
        Where on disk the cached copy lives (always inside ``CACHE_DIR``).
        Valid only when ``ok=True``.
    from_cache
        True if the request was satisfied without touching SSH.
    error
        Human-readable failure cause when ``ok=False``.
    size_bytes
        Size of the cached file (when known).
    """

    ok:           bool
    local_path:   Optional[str] = None
    from_cache:   bool          = False
    error:        Optional[str] = None
    size_bytes:   Optional[int] = None


# ---------------------------------------------------------------------------
# Cache maintenance
# ---------------------------------------------------------------------------

def _cache_files() -> List[Tuple[str, int, float]]:
    """List ``(path, size, mtime)`` for every file under ``CACHE_DIR``."""
    out = []
    if not os.path.isdir(CACHE_DIR):
        return out
    for dirpath, _dirs, files in os.walk(CACHE_DIR):
        for fname in files:
            full = os.path.join(dirpath, fname)
            try:
                st = os.stat(full)
            except OSError:
                continue
            out.append((full, int(st.st_size), float(st.st_mtime)))
    return out


def cache_size_bytes() -> int:
    return sum(s for _p, s, _t in _cache_files())


def _evict_lru_until_under(limit: int) -> int:
    """Delete oldest cached files until total size ≤ ``limit``. Returns bytes freed."""
    items = _cache_files()
    items.sort(key=lambda x: x[2])    # oldest first
    total = sum(s for _p, s, _t in items)
    freed = 0
    for path, size, _t in items:
        if total <= limit:
            break
        try:
            os.remove(path)
        except OSError:
            continue
        total -= size
        freed += size
    return freed


# ---------------------------------------------------------------------------
# The fetcher
# ---------------------------------------------------------------------------

def _remote_size_bytes(remote_path: str) -> Tuple[bool, Optional[int], Optional[str]]:
    """``stat -c %s <path>`` over SSH → ``(ok, size_or_None, err)``."""
    if STATE.ssh is None or not STATE.ssh.is_connected():
        return False, None, "ssh not connected"
    cmd = f"stat -c %s {shlex.quote(remote_path)}"
    try:
        rc, out, err = STATE.ssh.run(cmd, timeout=10)
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"
    if rc != 0:
        return False, None, (err.strip() or "file not found on remote")
    try:
        return True, int(out.strip()), None
    except ValueError:
        return False, None, f"unparseable stat output: {out!r}"


def fetch_one_file(
    remote_path: str,
    *,
    max_bytes: int = MAX_PULL_BYTES,
    cache_ttl: int = CACHE_TTL_SECONDS,
    force: bool = False,
) -> FetchResult:
    """Pull one file from FASRC into the local cache.

    Returns a :class:`FetchResult` — never raises. Idempotent for the
    same ``remote_path`` within ``cache_ttl``.

    ``force=True`` bypasses the TTL cache entirely and always re-rsyncs.
    Use this for explicit "Sync now" buttons — rsync itself is still
    incremental at the byte level, so a no-op sync of an unchanged file
    is cheap.

    The function intentionally does no batching or background sync —
    every call corresponds to a user-initiated action (click on a
    file). FASRC's docs flag bulk parallel transfers on login nodes,
    so we keep it strictly one-at-a-time.
    """
    if not is_allowed_remote_path(remote_path):
        return FetchResult(ok=False,
                           error=f"path not under allowed FASRC roots: {remote_path}")

    local = _local_path_for(remote_path)
    # 1. Serve from cache if fresh and present — skipped when force=True.
    if not force and os.path.isfile(local):
        try:
            age = time.time() - os.path.getmtime(local)
            size = os.path.getsize(local)
        except OSError:
            age, size = float("inf"), 0
        if age < cache_ttl:
            return FetchResult(
                ok=True, local_path=local, from_cache=True,
                size_bytes=size,
            )

    # 2. Stat the remote to enforce the size cap before transfer.
    ok, remote_size, err = _remote_size_bytes(remote_path)
    if not ok:
        return FetchResult(ok=False, error=err)
    if remote_size is not None and remote_size > max_bytes:
        return FetchResult(
            ok=False,
            size_bytes=remote_size,
            error=(f"file is {remote_size / 1e6:.0f} MB — over the "
                   f"{max_bytes // (1024 * 1024)} MB pull cap. "
                   "Use Globus or a compute-node-mediated transfer for "
                   "large files."),
        )

    # 3. Single-flight: serialise concurrent pulls of the same file.
    lock = _lock_for(remote_path)
    with lock:
        # Re-check after acquiring the lock in case another thread
        # finished the pull while we were waiting — skipped on force.
        if not force and os.path.isfile(local):
            try:
                age = time.time() - os.path.getmtime(local)
            except OSError:
                age = float("inf")
            if age < cache_ttl:
                return FetchResult(
                    ok=True, local_path=local, from_cache=True,
                    size_bytes=os.path.getsize(local),
                )

        _ensure_parent_dir(local)
        try:
            rc, _out, err = STATE.ssh.rsync_pull(
                remote_path,
                os.path.dirname(local),
                # ``-t`` preserves mtime so the cache-TTL math is meaningful;
                # ``--inplace`` avoids the temp-file rename for big-but-allowed files.
                extra_args=["-t", "--inplace"],
                timeout=120,
            )
        except Exception as e:
            return FetchResult(ok=False,
                               error=f"rsync failed: {type(e).__name__}: {e}")
        if rc != 0:
            return FetchResult(ok=False,
                               error=f"rsync exit {rc}: {err.strip()}")

        if not os.path.isfile(local):
            return FetchResult(
                ok=False,
                error=f"rsync reported success but local file missing: {local}",
            )

        # Background cleanup: keep cache below the cap. Cheap; only walks
        # the cache subtree.
        if cache_size_bytes() > MAX_CACHE_BYTES:
            _evict_lru_until_under(MAX_CACHE_BYTES)

        return FetchResult(
            ok=True, local_path=local, from_cache=False,
            size_bytes=os.path.getsize(local),
        )


# ---------------------------------------------------------------------------
# Remote Python invocation — for the tile inspector
# ---------------------------------------------------------------------------

def run_remote_python(
    script_rel_path: str, args: List[str], *,
    binary: bool = False, timeout: int = 30,
) -> Tuple[int, object, str]:
    """Run a project script on FASRC via the existing SSH ControlMaster.

    Returns ``(rc, stdout, stderr)`` — stdout is bytes when ``binary=True``.
    The command activates the configured conda env so astropy/numpy/PIL
    are available, then invokes the script with its argv tail.

    Used by the HST tile inspector to call ``scripts/fasrc_inspect_tile.py``
    on a login node without rsync'ing the multi-GB tile back to local.
    """
    if STATE.ssh is None or not STATE.ssh.is_connected():
        return 1, (b"" if binary else ""), "ssh not connected"
    cfg = fasrc_config.load()
    py = f"{cfg.conda_env_path.rstrip('/')}/bin/python"
    script = f"{cfg.repo_path.rstrip('/')}/{script_rel_path.lstrip('/')}"
    cmd = (
        f"{shlex.quote(py)} {shlex.quote(script)} "
        + " ".join(shlex.quote(a) for a in args)
    )
    try:
        return STATE.ssh.run(cmd, timeout=timeout, binary=binary)
    except Exception as e:
        return 1, (b"" if binary else ""), f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Remote directory listing (cheap — single SSH round-trip)
# ---------------------------------------------------------------------------

def list_remote_dir(
    remote_dir: str,
    *,
    glob_pattern: str = "*",
    max_entries: int = 500,
) -> Tuple[bool, List[Dict[str, object]], Optional[str]]:
    """List one remote directory via ``find`` over the existing SSH session.

    Returns ``(ok, entries, error)`` where each entry is
    ``{"name": ..., "size": int, "mtime": float}``.

    Never recurses by default (``-maxdepth 1``) — large recursive walks
    on a shared login node would be discourteous and slow. Cap of
    ``max_entries`` to keep the response payload bounded.
    """
    if not is_allowed_remote_path(remote_dir):
        return False, [], f"path not under allowed FASRC roots: {remote_dir}"
    if STATE.ssh is None or not STATE.ssh.is_connected():
        return False, [], "ssh not connected"

    # find ... -printf '%f|%s|%T@\n'  → name | size | mtime
    cmd = (
        f"find {shlex.quote(remote_dir)} -maxdepth 1 -type f "
        f"-name {shlex.quote(glob_pattern)} "
        f"-printf '%f|%s|%T@\\n' | head -n {int(max_entries)}"
    )
    try:
        rc, out, err = STATE.ssh.run(cmd, timeout=15)
    except Exception as e:
        return False, [], f"{type(e).__name__}: {e}"
    if rc != 0:
        # Treat missing directory as empty rather than error.
        if "No such file or directory" in err:
            return True, [], None
        return False, [], err.strip() or "find failed"

    entries: List[Dict[str, object]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        try:
            entries.append({
                "name":  parts[0],
                "size":  int(parts[1]),
                "mtime": float(parts[2]),
            })
        except ValueError:
            continue
    return True, entries, None
