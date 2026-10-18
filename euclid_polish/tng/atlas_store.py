"""The TNG atlas store and its scratch working copy.

The SKIRT atlas is kept permanently in lab storage
(``Config.TNG_SKIRT_STORE_DIR``; holylabs on FASRC) and generation reads a
working copy on scratch (``Config.TNG_SKIRT_DIR``; netscratch, purged after
90 idle days — on 2026-10-18 that forced a full re-download from an
overloaded server). :func:`sync_working_copy` copies every galaxy that is
complete in the store but not in the working copy: each FITS through a
temporary name and an atomic rename, then the ``.done`` marker, keeping
modification times so the radius manifest's inventory fingerprint does not
change. Nothing in the working copy is deleted.

    python -m euclid_polish.tng.atlas_store [--store DIR] [--work DIR] [--workers N]
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from euclid_polish.config import Config
from euclid_polish.tng.atlas import TNGGalaxy

FITS_PATTERN = "TNG*_O*_Euclid_*.fits"


def _galaxy_files(directory: Path) -> list[Path]:
    return sorted(directory.glob(FITS_PATTERN))


def _in_sync(source: Path, target: Path) -> bool:
    """The working-copy galaxy has the marker and every store FITS at full size."""
    if not (target / Config.Tng.DONE_MARKER).is_file():
        return False
    for path in _galaxy_files(source):
        copy = target / path.name
        if not copy.is_file() or copy.stat().st_size != path.stat().st_size:
            return False
    return True


def _copy_galaxy(source: Path, target: Path) -> int:
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in _galaxy_files(source):
        temporary = target / f".{path.name}.part"
        shutil.copy2(path, temporary)
        os.replace(temporary, target / path.name)
        copied += path.stat().st_size
    shutil.copy2(source / Config.Tng.DONE_MARKER, target / Config.Tng.DONE_MARKER)
    return copied


def sync_working_copy(
    store_dir: str | Path, work_dir: str | Path, *, workers: int = 8,
) -> dict[str, int | str]:
    """Copy the store's complete galaxies that the working copy lacks (or
    holds only partly) into it. Returns counts and the bytes copied."""
    store, work = Path(store_dir), Path(work_dir)
    if not store.is_dir():
        raise FileNotFoundError(f"TNG atlas store {store} does not exist")
    work.mkdir(parents=True, exist_ok=True)
    if store.resolve() == work.resolve():
        return {"store_galaxies": len(TNGGalaxy.discover(store)), "copied": 0, "in_sync": 0, "bytes": 0,
                "note": "store and working copy are the same directory"}
    galaxies = TNGGalaxy.discover(store)
    todo = [g for g in galaxies if not _in_sync(g.directory, work / g.subhalo_id)]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        copied_bytes = sum(pool.map(lambda g: _copy_galaxy(g.directory, work / g.subhalo_id), todo))
    return {"store_galaxies": len(galaxies), "copied": len(todo), "in_sync": len(galaxies) - len(todo),
            "bytes": int(copied_bytes)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy the TNG atlas store's galaxies to the scratch working copy.")
    parser.add_argument("--store", default=Config.TNG_SKIRT_STORE_DIR)
    parser.add_argument("--work", default=Config.TNG_SKIRT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if not args.store:
        parser.error("no store: set EUCLID_POLISH_TNG_STORE_DIR or pass --store")
    result = sync_working_copy(args.store, args.work, workers=args.workers)
    print(f"TNG atlas: {result['copied']} galaxies copied ({int(result['bytes']) / 1e9:.1f} GB), "
          f"{result['in_sync']} already in sync, {result['store_galaxies']} in the store")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["sync_working_copy"]
