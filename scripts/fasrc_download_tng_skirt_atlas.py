#!/usr/bin/env python
"""Bulk-download the *entire* IllustrisTNG TNG50-1 SKIRT atlas (~1153 galaxies).

This is the FASRC, all-galaxies counterpart of the single-galaxy
``scripts/download_tng_skirt.py``. It lists every entry under the API's
``files/skirt_atlas/`` endpoint and downloads them concurrently with a thread
pool (the work is network-I/O bound — one worker per allocated CPU keeps the
pipe full while a few threads decompress). For each galaxy it keeps **only** the
dusty Euclid frames — ``TNG<id>_O<k>_Euclid_<band>.fits`` for band ∈ {VIS, Y, J,
H} and orientation k ∈ {1..5} → 20 FITS/galaxy — discarding the ``_nodust``
twins and the 2MASS/SDSS/GALEX renders, exactly like the single-galaxy script.

Layout (under ``--out-dir``, default ``Config.TNG_SKIRT_DIR`` =
``$EUCLID_POLISH_DATA_DIR/tng_skirt`` → netscratch on FASRC)::

    tng_skirt/
      <subhalo_id>/
        TNG<id>_O1_Euclid_VIS.fits
        TNG<id>_O1_Euclid_Y.fits
        … (4 bands × 5 orientations)
        .done                    # completion sentinel — re-runs skip this galaxy

Resumability: a galaxy whose folder holds a ``.done`` marker is skipped, so the
job can be re-submitted after a time-limit or a transient failure and only the
unfinished galaxies are fetched.

API key — the IllustrisTNG public API needs your *personal* token
(https://www.tng-project.org/users/profile/ → "API Token"). It is NEVER
committed and never passes through the WebUI form. Provide it on FASRC via,
in order of precedence:

    1. ``--api-key`` (discouraged — visible in the process table)
    2. ``$TNG_API_KEY``
    3. ``--api-key-file`` (default ``~/.tng_api_key``, a single-line file in
       your shared FASRC home — mirrors the ``~/.euclid_credentials`` pattern)

Create it once on the FASRC login node::

    echo 'YOUR_TOKEN_HERE' > ~/.tng_api_key && chmod 600 ~/.tng_api_key

Usage::

    python scripts/fasrc_download_tng_skirt_atlas.py --workers 16
    python scripts/fasrc_download_tng_skirt_atlas.py --workers 16 --limit 5   # smoke test
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import os
import sys
import tarfile
import tempfile
import time
from typing import List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.observability.reporter import Reporter

# Reuse the single-galaxy script's battle-tested API helpers + filter so the
# atlas-listing parsing and the Euclid-only member regex live in one place.
from scripts.download_tng_skirt import (
    SKIRT_ATLAS,
    _KEEP_RE,
    _TAR_SUFFIXES,
    _list_atlas,
    _name_from_url,
    _request,
)


# ---------------------------------------------------------------------------
# API key resolution (arg → env → file), never through the WebUI
# ---------------------------------------------------------------------------

def _load_key(args: argparse.Namespace) -> str:
    """Resolve the TNG API token: ``--api-key`` → ``$TNG_API_KEY`` → key file."""
    if args.api_key:
        return args.api_key.strip()
    env = os.environ.get("TNG_API_KEY", "").strip()
    if env:
        return env
    path = os.path.expanduser(args.api_key_file)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            key = f.readline().strip()
        if key:
            return key
    sys.exit(
        "No TNG API key. Set $TNG_API_KEY, pass --api-key, or write your token "
        f"to {args.api_key_file} (one line). Get it from "
        "https://www.tng-project.org/users/profile/."
    )


# ---------------------------------------------------------------------------
# Per-galaxy download + Euclid-only extract
# ---------------------------------------------------------------------------

def _galaxy_id_from_name(name: str) -> str:
    """``665976.tar.gz`` → ``665976`` (the subhalo id; the per-galaxy folder)."""
    base = os.path.basename(name)
    for suf in _TAR_SUFFIXES:
        if base.endswith(suf):
            return base[: -len(suf)]
    return base.split(".", 1)[0]


def _stream_to_file(url: str, key: str, dest_path: str) -> int:
    """Stream a (multi-GB) URL to ``dest_path`` in 1 MiB chunks. Returns bytes."""
    r = _request(url, key, stream=True)
    done = 0
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            f.write(chunk)
            done += len(chunk)
    return done


def _extract_euclid(archive: str, dest_dir: str) -> int:
    """Stream-extract only ``TNG*_O?_Euclid_<band>.fits`` members in ONE pass.

    Opened in streaming mode (``r|gz``) so the gzip stream is inflated exactly
    once. The previous random-access path (``getmembers()`` then
    ``extractall``) decompressed each multi-GB archive ~twice — first to
    enumerate members, then again to seek back to them — which is pure wasted
    CPU on data we mostly discard. Quiet (no prints — many run concurrently);
    returns the count of Euclid FITS written. Members are flattened to their
    basename so each lands directly in ``<dest_dir>/<name>.fits``.
    """
    os.makedirs(dest_dir, exist_ok=True)
    n = 0
    with tarfile.open(archive, "r|gz") as tf:     # streaming → single inflate
        for m in tf:
            if not (m.isfile() and _KEEP_RE.search(m.name)):
                continue
            m.name = os.path.basename(m.name)      # flatten
            try:
                tf.extract(m, dest_dir, filter="data")   # py3.12+
            except TypeError:
                tf.extract(m, dest_dir)                   # older pythons
            n += 1
    return n


def _download_one(
    *,
    name: str,
    url: Optional[str],
    key: str,
    out_dir: str,
    keep_archive: bool,
) -> dict:
    """Fetch + extract one atlas galaxy. Returns a status dict for the tally.

    ``{"status": "written"|"cached"|"failed", "id": str, "n_fits": int,
       "bytes": int, "dl_secs": float, "ex_secs": float, "errors": [str]}``.
    The per-phase timings let the parent report aggregate MB/s and an
    "effective concurrency" factor (worker-seconds / wall) so we can SEE
    whether the job is network- or CPU-bound. The tarball is streamed to a
    private temp dir and removed after extraction (unless ``--keep-archive``),
    so transient disk is bounded by ``workers × tarball_size`` — never the
    whole 2-3 TB atlas.
    """
    gid = _galaxy_id_from_name(name)
    galaxy_dir = os.path.join(out_dir, gid)
    done_marker = os.path.join(galaxy_dir, Config.Tng.DONE_MARKER)
    base = {"id": gid, "n_fits": 0, "bytes": 0,
            "dl_secs": 0.0, "ex_secs": 0.0, "errors": []}
    if os.path.isfile(done_marker):
        return {**base, "status": "cached"}

    if not url:                        # listing gave only a name → build the URL
        url = SKIRT_ATLAS + name

    tmp_dir = tempfile.mkdtemp(prefix=f"tng_{gid}_", dir=out_dir)
    archive = os.path.join(tmp_dir, name if name.endswith(_TAR_SUFFIXES)
                           else f"{gid}.tar.gz")
    try:
        t0 = time.perf_counter()
        nbytes = _stream_to_file(url, key, archive)
        t1 = time.perf_counter()
        n_fits = _extract_euclid(archive, galaxy_dir)
        t2 = time.perf_counter()
        timings = {"bytes": nbytes, "dl_secs": t1 - t0, "ex_secs": t2 - t1}
        if n_fits == 0:
            return {**base, **timings, "status": "failed",
                    "errors": ["no FITS extracted from archive"]}
        if keep_archive:
            os.replace(archive, os.path.join(galaxy_dir, os.path.basename(archive)))
        # Sentinel last: its presence means the galaxy is fully materialised.
        with open(done_marker, "w", encoding="utf-8") as f:
            f.write(f"{n_fits} fits\n")
        return {**base, **timings, "status": "written", "n_fits": n_fits}
    except SystemExit as e:            # _request aborts on 401/403 (bad key)
        return {**base, "status": "failed", "errors": [str(e)]}
    except Exception as e:
        return {**base, "status": "failed",
                "errors": [f"{type(e).__name__}: {e}"]}
    finally:
        try:
            if os.path.isfile(archive):
                os.remove(archive)
        except OSError:
            pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=Config.TNG_SKIRT_DIR,
                   help=f"Root for the per-galaxy folders. Default: "
                        f"{Config.TNG_SKIRT_DIR}")
    p.add_argument("--workers", type=int, default=16,
                   help="Concurrent galaxy downloads in flight. Default 16. "
                        "May exceed the CPU count (downloads are I/O-bound).")
    p.add_argument("--executor", choices=("process", "thread"), default="process",
                   help="Parallelism backend. 'process' (default) gives true "
                        "multi-core download+extract with no GIL contention — "
                        "threads cap extraction at ~2 cores and starve each "
                        "other's download loops. 'thread' is lighter and fine "
                        "when purely network-bound.")
    p.add_argument("--limit", type=int, default=0,
                   help="Only download the first N atlas entries (0 = all "
                        "~1153). Use a small value for a smoke test.")
    p.add_argument("--keep-archive", action="store_true",
                   help="Keep each galaxy's .tar.gz (in its folder) after "
                        "extracting. Default: delete it to save disk.")
    p.add_argument("--api-key", default="",
                   help="TNG API token (discouraged — prefer $TNG_API_KEY or "
                        "--api-key-file so it stays off the process table).")
    p.add_argument("--api-key-file", default=Config.Tng.API_KEY_FILE,
                   help=f"Single-line file holding the TNG API token. "
                        f"Default: {Config.Tng.API_KEY_FILE}")
    p.add_argument("--dry-run", action="store_true",
                   help="List the atlas and report what would be downloaded, "
                        "then exit without fetching anything.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    key = _load_key(args)
    reporter = Reporter.from_env()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 64)
    print("  IllustrisTNG TNG50-1 SKIRT atlas — bulk Euclid-frame download")
    print("=" * 64)
    print(f"  out dir   = {out_dir}")
    print(f"  workers   = {args.workers}  ({args.executor})")
    print(f"  bands     = VIS + NISP Y/J/H (dusty), 5 orientations / galaxy")

    reporter.set_stage("listing atlas")
    entries: List[Tuple[str, Optional[str], Optional[int]]] = _list_atlas(key)
    if not entries:
        print("  ✗ no atlas entries parsed — check the API key / endpoint.")
        return 1
    if args.limit and args.limit > 0:
        entries = entries[: args.limit]
    n_total = len(entries)
    print(f"  galaxies  = {n_total}")
    print()

    if args.dry_run:
        n_done = sum(
            1 for (name, _u, _s) in entries
            if os.path.isfile(os.path.join(out_dir, _galaxy_id_from_name(name),
                                           Config.Tng.DONE_MARKER))
        )
        print(f"  DRY RUN — {n_done} already complete, "
              f"{n_total - n_done} would be downloaded "
              f"(20 Euclid FITS each).")
        return 0

    t0 = time.perf_counter()
    reporter.set_stage(f"downloading ({args.executor}, {args.workers} workers)")

    # Process pool = true multi-core (no GIL); fork (Linux default) lets each
    # worker inherit the already-imported modules for free. Fall back to the
    # platform default start method if fork is unavailable (non-Linux).
    if args.executor == "process":
        Executor = concurrent.futures.ProcessPoolExecutor
        pool_kwargs = {"max_workers": max(1, args.workers)}
        try:
            pool_kwargs["mp_context"] = multiprocessing.get_context("fork")
        except ValueError:
            pass
    else:
        Executor = concurrent.futures.ThreadPoolExecutor
        pool_kwargs = {"max_workers": max(1, args.workers)}

    n_written = n_cached = n_failed = n_fits_total = 0
    bytes_total = 0
    dl_secs_total = ex_secs_total = 0.0
    completed = 0
    with Executor(**pool_kwargs) as pool:
        fut_to_id = {
            pool.submit(
                _download_one,
                name=name, url=url, key=key,
                out_dir=out_dir, keep_archive=args.keep_archive,
            ): _galaxy_id_from_name(name)
            for (name, url, _size) in entries
        }
        for fut in concurrent.futures.as_completed(fut_to_id):
            gid = fut_to_id[fut]
            completed += 1
            try:
                res = fut.result()
            except Exception as e:
                n_failed += 1
                reporter.set_step(completed, n_total, f"TNG{gid}")
                reporter.warn(f"galaxy {gid} crashed: {type(e).__name__}: {e}")
                continue
            status = res["status"]
            if status == "written":
                n_written += 1
                n_fits_total += res["n_fits"]
                bytes_total   += res.get("bytes", 0)
                dl_secs_total += res.get("dl_secs", 0.0)
                ex_secs_total += res.get("ex_secs", 0.0)
            elif status == "cached":
                n_cached += 1
            else:
                n_failed += 1
                for err in res["errors"]:
                    reporter.warn(f"galaxy {gid} failed: {err}")
            # Live aggregate throughput + ETA, surfaced in the WebUI step label.
            elapsed = time.perf_counter() - t0
            mbps = (bytes_total / 1e6 / elapsed) if elapsed > 0 else 0.0
            rate = (n_written / elapsed) if (n_written and elapsed > 0) else 0.0
            eta_min = ((n_total - completed) / rate / 60.0) if rate > 0 else 0.0
            reporter.set_step(
                completed, n_total,
                f"TNG{gid} · {mbps:.0f} MB/s · ETA {eta_min:.0f}m")
            if completed % 25 == 0 or completed == n_total:
                print(f"  [{completed}/{n_total}] {bytes_total/1e9:.1f} GB · "
                      f"{mbps:.0f} MB/s agg · ETA {eta_min:.0f} min · "
                      f"written={n_written} cached={n_cached} failed={n_failed}",
                      flush=True)

    runtime = time.perf_counter() - t0
    print()
    print("=" * 64)
    print(f"Summary  ({runtime / 60:.1f} min wall):")
    print(f"  galaxies written = {n_written}  ({n_fits_total} Euclid FITS)")
    print(f"  galaxies cached  = {n_cached}  (already had .done; skipped)")
    print(f"  galaxies failed  = {n_failed}")
    agg_mbps = (bytes_total / 1e6 / runtime) if runtime > 0 else 0.0
    print(f"  downloaded       = {bytes_total / 1e9:.2f} GB "
          f"@ {agg_mbps:.0f} MB/s aggregate")
    if n_written:
        print(f"  per galaxy       = {bytes_total / 1e6 / n_written:.0f} MB · "
              f"download {dl_secs_total / n_written:.1f}s · "
              f"extract {ex_secs_total / n_written:.1f}s (mean)")
    # Effective concurrency = sum of per-galaxy worker-seconds ÷ wall-seconds.
    # ≈ the number of galaxies actually being processed in parallel. If this is
    # ~= workers, you're fully parallel; if it's ~2-4 with 32 workers, a
    # bottleneck (GIL under threads, or a network/server cap) is serialising.
    if runtime > 0:
        eff = (dl_secs_total + ex_secs_total) / runtime
        print(f"  effective concurrency ≈ {eff:.1f}x  "
              f"(of {args.workers} {args.executor} workers)")
    print(f"  out dir          = {out_dir}")
    print(f"\nRUNTIME_SECONDS={runtime:.1f}")
    # Non-zero exit only if *everything* failed (so a few flaky galaxies on a
    # 1153-way run don't fail the whole SLURM job — re-submit to fill the gaps).
    return 1 if (n_written == 0 and n_cached == 0 and n_failed > 0) else 0


if __name__ == "__main__":
    sys.exit(main())
