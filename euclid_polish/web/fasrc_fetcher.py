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
    (default 4 GB) we delete the oldest files.
  * **Allowed roots**: pulls are restricted to known data dirs on the
    remote (``data_dir``, ``ckpt_dir``, ``repo_path/logs``). Anything
    else gets a 403.

The web layer NEVER polls this in the background — every call is user-
initiated by clicking a thumbnail or a "fetch from FASRC" button.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import threading
import time
from dataclasses import dataclass

from euclid_polish.config import Config
from euclid_polish.web import fasrc_config, fasrc_jobs
from euclid_polish.web.remote import STATE

# ---------------------------------------------------------------------------
# Safety: which remote paths are reachable
# ---------------------------------------------------------------------------

def allowed_remote_roots() -> list[str]:
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
    return any(rp == root or rp.startswith(root + "/") for root in allowed_remote_roots())


# ---------------------------------------------------------------------------
# Local cache layout
# ---------------------------------------------------------------------------

def _local_path_for(remote_path: str) -> str:
    """Deterministic local cache path for a given remote absolute path.

    We strip the leading ``/`` and place under ``Config.FASRC_CACHE_DIR``
    so two different remote paths can't collide and ``os.path.realpath``
    gives a predictable result. The path is also what the inspector page
    sees in its safety check (already whitelists subtrees of ``data/``).
    """
    rel = remote_path.lstrip("/")
    return os.path.realpath(os.path.join(Config.FASRC_CACHE_DIR, rel))


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ---------------------------------------------------------------------------
# Single-flight locks per path
# ---------------------------------------------------------------------------

_LOCKS_REGISTRY: dict[str, threading.Lock] = {}
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
        Where on disk the cached copy lives (always inside
        ``Config.FASRC_CACHE_DIR``). Valid only when ``ok=True``.
    from_cache
        True if the request was satisfied without touching SSH.
    error
        Human-readable failure cause when ``ok=False``.
    size_bytes
        Size of the cached file (when known).
    """

    ok:           bool
    local_path:   str | None = None
    from_cache:   bool          = False
    error:        str | None = None
    size_bytes:   int | None = None


# ---------------------------------------------------------------------------
# Cache maintenance
# ---------------------------------------------------------------------------

def _cache_files() -> list[tuple[str, int, float]]:
    """List ``(path, size, mtime)`` for every file under ``Config.FASRC_CACHE_DIR``."""
    out = []
    if not os.path.isdir(Config.FASRC_CACHE_DIR):
        return out
    for dirpath, _dirs, files in os.walk(Config.FASRC_CACHE_DIR):
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


def _evict_lru_until_under(limit: int, protect: set | None = None) -> int:
    """Delete oldest cached files until total size ≤ ``limit``. Returns bytes freed.

    ``protect`` is a set of absolute paths that must never be evicted —
    pass the file a fetch just pulled so the eviction can't delete the
    very file the caller is about to ``stat`` (which previously raised
    a ``FileNotFoundError`` and broke the "never raises" contract). The
    protected file still counts toward ``total``, so if it alone exceeds
    ``limit`` the cache simply stays over budget for this call rather
    than corrupting the fetch.
    """
    protect = {os.path.abspath(p) for p in (protect or ())}
    items = _cache_files()
    items.sort(key=lambda x: x[2])    # oldest first
    total = sum(s for _p, s, _t in items)
    freed = 0
    for path, size, _t in items:
        if total <= limit:
            break
        if os.path.abspath(path) in protect:
            continue
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

def _remote_size_bytes(remote_path: str) -> tuple[bool, int | None, str | None]:
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
    max_bytes: int = Config.WebFetch.MAX_PULL_BYTES,
    cache_ttl: int = Config.WebFetch.CACHE_TTL_SECONDS,
    force: bool = False,
    protect_paths: set[str] | None = None,
) -> FetchResult:
    """Pull one file from FASRC into the local cache.

    Returns a :class:`FetchResult` — never raises. Idempotent for the
    same ``remote_path`` within ``cache_ttl``.

    ``force=True`` bypasses the TTL cache entirely and always re-rsyncs.
    Use this for explicit "Sync now" buttons — rsync itself is still
    incremental at the byte level, so a no-op sync of an unchanged file
    is cheap.

    ``protect_paths`` lets an explicit multi-file sync keep its whole requested
    set intact while the LRU makes room.  Paths are local cache paths; they are
    never read or written except as exclusion entries for eviction.

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
        ssh = STATE.ssh
        if ssh is None:
            return FetchResult(ok=False, error="ssh not connected")
        try:
            rc, _out, err = ssh.rsync_pull(
                remote_path,
                os.path.dirname(local),
                # NO ``-t``: we deliberately let the local copy take the
                # *fetch* time as its mtime, not the remote's. The cache
                # is keyed on "how long ago did WE pull this" for both the
                # TTL freshness check AND the LRU eviction order — using the
                # remote mtime made a file pulled from an old remote look
                # instantly stale (defeating the TTL) and look like the
                # least-recently-used file (so the LRU evicted the file we
                # had just fetched, then crashed stat-ing it). ``--inplace``
                # avoids the temp-file rename for big-but-allowed files.
                extra_args=["--inplace"],
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

        # ``rsync --inplace`` may leave an unchanged destination carrying its
        # old remote mtime.  The cache's TTL/LRU semantics are about when *we*
        # fetched it, so stamp the successful local fetch explicitly.
        with contextlib.suppress(OSError):
            os.utime(local, None)

        # Background cleanup: keep cache below the cap. Cheap; only walks
        # the cache subtree. ``protect`` the file we just pulled so the
        # eviction can never delete it out from under the ``getsize`` below
        # (it now also has a fresh mtime, so it's the LAST eviction
        # candidate anyway — belt and suspenders).
        if cache_size_bytes() > Config.WebFetch.MAX_CACHE_BYTES:
            protected = {local}
            protected.update(protect_paths or ())
            _evict_lru_until_under(Config.WebFetch.MAX_CACHE_BYTES,
                                   protect=protected)

        # Defensive ``getsize``: honour the "never raises" contract even if
        # a concurrent pull/eviction removed the file in the gap above.
        try:
            size_bytes = os.path.getsize(local)
        except OSError:
            size_bytes = remote_size if remote_size is not None else None
        return FetchResult(
            ok=True, local_path=local, from_cache=False,
            size_bytes=size_bytes,
        )


# ---------------------------------------------------------------------------
# Remote Python invocation — for the tile inspector
# ---------------------------------------------------------------------------

def run_remote_python(
    script_rel_path: str, args: list[str], *,
    binary: bool = False, timeout: int = 30,
) -> tuple[int, object, str]:
    """Run a project script on FASRC via the existing SSH ControlMaster.

    Returns ``(rc, stdout, stderr)`` — stdout is bytes when ``binary=True``.
    Delegates command construction to :func:`fasrc_jobs.build_remote_python_command`
    so there is a single canonical implementation of the conda-activation and
    environment-setup logic. Called by the HST tile inspector (``routes/hst.py``)
    to run ``scripts/fasrc_inspect_tile.py`` on a login node without rsync'ing
    the multi-GB tile back to local.
    """
    if STATE.ssh is None or not STATE.ssh.is_connected():
        return 1, (b"" if binary else ""), "ssh not connected"
    cfg = fasrc_config.load()
    cmd = fasrc_jobs.build_remote_python_command(cfg, [script_rel_path, *args])
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
    max_depth: int = 1,
) -> tuple[bool, list[dict[str, object]], str | None]:
    """List one remote directory via ``find`` over the existing SSH session.

    Returns ``(ok, entries, error)`` where each entry is
    ``{"name": ..., "size": int, "mtime": float}``.

    Defaults to ``max_depth=1`` so the listing is fast on a shared
    login node. Callers that need to see in-progress downloads stranded
    under a scratch subdirectory (e.g. ``mastDownload/HLSP/<id>/<file>``
    before the flatten step runs) can pass a larger depth. The caller
    is responsible for de-duplicating by basename when raising the
    depth, because the same logical file can briefly exist at both
    nested and flat layouts during a partial flatten.
    """
    if not is_allowed_remote_path(remote_dir):
        return False, [], f"path not under allowed FASRC roots: {remote_dir}"
    if STATE.ssh is None or not STATE.ssh.is_connected():
        return False, [], "ssh not connected"

    # find ... -printf '%f|%s|%T@\n'  → name | size | mtime
    cmd = (
        f"find {shlex.quote(remote_dir)} -maxdepth {int(max_depth)} -type f "
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

    entries: list[dict[str, object]] = []
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
