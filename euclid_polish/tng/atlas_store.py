"""The TNG atlas: a compressed store in lab storage, a working copy on scratch.

Generation reads the SKIRT atlas from a working copy on scratch
(``Config.TNG_SKIRT_DIR``; netscratch on FASRC). Netscratch purges files
whose modification time is over 90 days old, and tar extraction used to keep
the archives' 2023 timestamps, so freshly downloaded frames were swept
(2026-07-06, and again before 2026-10-18: each time a full re-download from
an overloaded server). The permanent copy is a store in lab storage
(``Config.TNG_SKIRT_STORE_DIR``; holylabs on FASRC) that holds every frame
gzip-compressed: lossless and ~4x smaller (the full atlas with its dust-free
twins is ~0.1 TiB instead of 0.45 TiB), so it fits in the lab quota.

* :func:`archive_to_store` compresses into the store every working-copy
  galaxy whose download finished with its dust-free twins and that the store
  lacks (or holds only partly).
* :func:`sync_working_copy` restores into the working copy every complete
  store galaxy that it lacks (or holds only partly), decompressing
  ``.fits.gz`` frames; a store of plain ``.fits`` frames works too.

Only missing or wrong-sized frames are written, each through a temporary name
and an atomic rename, with the ``.done`` marker last; nothing is deleted.
Restored frames are dated to the restore: they are new on scratch (FASRC
allows dating new data to its placement), and the radius manifest
re-measures the frames that changed.

    python -m euclid_polish.tng.atlas_store [sync|archive] [--store DIR] [--work DIR] [--workers N]
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import struct
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import BinaryIO

from euclid_polish.config import Config

FITS_PATTERN = "TNG*_O*_Euclid_*.fits"
GZIP_SUFFIX = ".gz"
COMPRESS_LEVEL = 6
_CHUNK = 1 << 20


def _marker(directory: Path) -> str | None:
    try:
        return (directory / Config.Tng.DONE_MARKER).read_text(encoding="utf-8")
    except OSError:
        return None


def _promised(marker: str) -> int:
    """The frame count a ``.done`` marker promises (``"40 fits with-nodust"``)."""
    try:
        return max(int(marker.split()[0]), 1)
    except (ValueError, IndexError):
        return 1


def _work_frames(directory: Path) -> list[Path]:
    return sorted(directory.glob(FITS_PATTERN))


def _store_frames(directory: Path) -> list[Path]:
    return sorted([*directory.glob(FITS_PATTERN + GZIP_SUFFIX), *directory.glob(FITS_PATTERN)])


def _raw_name(frame: Path) -> str:
    return frame.name[: -len(GZIP_SUFFIX)] if frame.name.endswith(GZIP_SUFFIX) else frame.name


def _raw_size(frame: Path) -> int:
    """Uncompressed size of a store frame: the gzip trailer's ISIZE (exact
    below 4 GiB; a frame is 10 MB) or, for a plain ``.fits``, its size."""
    if not frame.name.endswith(GZIP_SUFFIX):
        return frame.stat().st_size
    with open(frame, "rb") as f:
        f.seek(-4, os.SEEK_END)
        return struct.unpack("<I", f.read(4))[0]


def _galaxy_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir())


def _complete_store_galaxies(store: Path) -> list[Path]:
    """Store galaxies whose marker is written and whose frames are all there."""
    return [d for d in _galaxy_dirs(store)
            if (marker := _marker(d)) is not None and len(_store_frames(d)) >= _promised(marker)]


def _finished_downloads(work: Path) -> list[Path]:
    """Working-copy galaxies whose download finished with the dust-free twins
    (both downloaders write the tagged marker after the last frame)."""
    return [d for d in _galaxy_dirs(work)
            if (marker := _marker(d)) is not None and Config.Tng.DUST_FREE_TAG in marker.split()
            and len(_work_frames(d)) >= _promised(marker)]


def _write_atomically(target: Path, fill: Callable[[BinaryIO], None]) -> None:
    temporary = target.with_name(f".{target.name}.part")
    try:
        with open(temporary, "wb") as out:
            fill(out)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _compress(frame: Path, out: BinaryIO, level: int) -> None:
    with open(frame, "rb") as raw, gzip.GzipFile(
            filename=frame.name, mode="wb", compresslevel=level, fileobj=out, mtime=0) as packed:
        shutil.copyfileobj(raw, packed, _CHUNK)


def _decompress(frame: Path, out: BinaryIO) -> None:
    opener = gzip.open if frame.name.endswith(GZIP_SUFFIX) else open
    with opener(frame, "rb") as source:
        shutil.copyfileobj(source, out, _CHUNK)


def _write_marker(text: str, target: Path) -> None:
    _write_atomically(target / Config.Tng.DONE_MARKER, lambda out: out.write(text.encode("utf-8")))


def _unarchived(source: Path, target: Path) -> list[Path]:
    """Working-copy frames that the store lacks or holds at another size."""
    return [f for f in _work_frames(source)
            if not (packed := target / (f.name + GZIP_SUFFIX)).is_file()
            or _raw_size(packed) != f.stat().st_size]


def _unrestored(source: Path, target: Path) -> list[Path]:
    """Store frames that the working copy lacks or holds at another size."""
    return [f for f in _store_frames(source)
            if not (copy := target / _raw_name(f)).is_file()
            or copy.stat().st_size != _raw_size(f)]


def _archive_galaxy(source: Path, target: Path, frames: list[Path], level: int) -> int:
    target.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        _write_atomically(target / (frame.name + GZIP_SUFFIX),
                          lambda out, frame=frame: _compress(frame, out, level))
    _write_marker(_marker(source) or "", target)
    return sum(frame.stat().st_size for frame in frames)


def _restore_galaxy(source: Path, target: Path, frames: list[Path]) -> int:
    target.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        _write_atomically(target / _raw_name(frame), lambda out, frame=frame: _decompress(frame, out))
    _write_marker(_marker(source) or "", target)
    return sum(_raw_size(frame) for frame in frames)


def archive_to_store(
    work_dir: str | Path, store_dir: str | Path, *, workers: int = 8, level: int = COMPRESS_LEVEL,
) -> dict[str, int | str]:
    """Compress the working copy's finished downloads that the store lacks (or
    holds only partly) into it. Returns counts, the raw bytes compressed and
    the store's compressed size."""
    work, store = Path(work_dir), Path(store_dir)
    if not work.is_dir():
        raise FileNotFoundError(f"TNG atlas working copy {work} does not exist")
    store.mkdir(parents=True, exist_ok=True)
    if store.resolve() == work.resolve():
        return {"finished": 0, "archived": 0, "in_store": 0, "bytes_read": 0, "store_galaxies": 0,
                "store_bytes": 0, "note": "store and working copy are the same directory"}
    finished = _finished_downloads(work)
    todo = [(galaxy, store / galaxy.name, frames) for galaxy in finished
            if (frames := _unarchived(galaxy, store / galaxy.name))
            or _marker(store / galaxy.name) != _marker(galaxy)]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        read = sum(pool.map(lambda job: _archive_galaxy(*job, level), todo))
    complete = _complete_store_galaxies(store)
    return {"finished": len(finished), "archived": len(todo), "in_store": len(finished) - len(todo),
            "bytes_read": int(read), "store_galaxies": len(complete),
            "store_bytes": sum(f.stat().st_size for galaxy in complete for f in _store_frames(galaxy))}


def sync_working_copy(
    store_dir: str | Path, work_dir: str | Path, *, workers: int = 8,
) -> dict[str, int | str]:
    """Restore the store's complete galaxies that the working copy lacks (or
    holds only partly) into it. Returns counts and the bytes written."""
    store, work = Path(store_dir), Path(work_dir)
    if not store.is_dir():
        return {"store_galaxies": 0, "copied": 0, "in_sync": 0, "bytes": 0,
                "note": f"no atlas store at {store} yet"}
    work.mkdir(parents=True, exist_ok=True)
    galaxies = _complete_store_galaxies(store)
    if store.resolve() == work.resolve():
        return {"store_galaxies": len(galaxies), "copied": 0, "in_sync": len(galaxies), "bytes": 0,
                "note": "store and working copy are the same directory"}
    todo = [(galaxy, work / galaxy.name, frames) for galaxy in galaxies
            if (frames := _unrestored(galaxy, work / galaxy.name))
            or _marker(work / galaxy.name) is None]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        written = sum(pool.map(lambda job: _restore_galaxy(*job), todo))
    return {"store_galaxies": len(galaxies), "copied": len(todo), "in_sync": len(galaxies) - len(todo),
            "bytes": int(written)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Restore the scratch working copy of the TNG atlas from its compressed store "
                    "(sync), or compress finished downloads into the store (archive).")
    parser.add_argument("action", nargs="?", choices=("sync", "archive"), default="sync")
    parser.add_argument("--store", default=Config.TNG_SKIRT_STORE_DIR)
    parser.add_argument("--work", default=Config.TNG_SKIRT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if not args.store:
        parser.error("no store: set EUCLID_POLISH_TNG_STORE_DIR or pass --store")
    if args.action == "archive":
        result = archive_to_store(args.work, args.store, workers=args.workers)
        print(f"TNG atlas store: {result['archived']} galaxies archived "
              f"({int(result['bytes_read']) / 1e9:.1f} GB compressed), {result['in_store']} already there; "
              f"the store holds {result['store_galaxies']} galaxies in "
              f"{int(result['store_bytes']) / 2**30:.1f} GiB")
    else:
        result = sync_working_copy(args.store, args.work, workers=args.workers)
        print(f"TNG atlas: {result['copied']} galaxies restored ({int(result['bytes']) / 1e9:.1f} GB), "
              f"{result['in_sync']} already in sync, {result['store_galaxies']} in the store")
    if "note" in result:
        print(f"  note: {result['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["archive_to_store", "sync_working_copy"]
