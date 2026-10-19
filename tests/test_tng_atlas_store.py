"""The TNG atlas is kept permanently in a gzip-compressed store (lab storage):
finished downloads are archived into it, and generation restores what the
scratch working copy lacks from it."""
from __future__ import annotations

import argparse
import gzip
import os
import time

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.tng.atlas import TNGGalaxy
from euclid_polish.tng.atlas_store import archive_to_store, main, sync_working_copy
from euclid_polish.tng.types import TNG_FITS_BANDS
from scripts import fasrc_download_tng_skirt_atlas as downloader
from scripts import run_pipeline

OLD = 1_700_000_000.0                       # 2023, the archives' timestamps
FINISHED = f"40 fits {Config.Tng.DUST_FREE_TAG}\n"


def _names(gid: str, twins: bool) -> list[str]:
    dusty = [f"TNG{gid}_O{o}_Euclid_{b}.fits" for o in range(1, 6) for b in TNG_FITS_BANDS]
    return dusty + ([name.replace(".fits", "_nodust.fits") for name in dusty] if twins else [])


def _galaxy(root, gid: str, *, files: int | None = None, size: int = 64, twins: bool = False,
            marker: str | None = None):
    """A galaxy folder as extraction left it: frames dated 2023 (mostly zero
    floats, like the frames beyond the stars) and the ``.done`` marker."""
    directory = root / gid
    directory.mkdir(parents=True, exist_ok=True)
    names = _names(gid, twins)[:files]
    rng = np.random.default_rng(int(gid))
    for name in names:
        data = np.zeros(size // 4, ">f4")
        data[: len(data) // 8] = rng.random(len(data) // 8)
        path = directory / name
        path.write_bytes(data.tobytes())
        os.utime(path, (OLD, OLD))
    (directory / Config.Tng.DONE_MARKER).write_text(marker or f"{len(names)} fits\n")
    return directory


def test_sync_restores_missing_and_partial_galaxies_dated_now(tmp_path):
    store, work = tmp_path / "store", tmp_path / "work"
    for gid in ("11", "22", "33"):
        _galaxy(store, gid)
    _galaxy(store, "44", files=7, marker="20 fits\n")        # frames missing in the store: never restored
    _galaxy(work, "11")                                       # already in sync
    partial = _galaxy(work, "22")
    (partial / "TNG22_O5_Euclid_H.fits").write_bytes(b"x")    # truncated
    t0 = time.time()
    result = sync_working_copy(store, work, workers=2)
    assert result == {"store_galaxies": 3, "copied": 2, "in_sync": 1, "bytes": 21 * 64}
    assert {g.subhalo_id for g in TNGGalaxy.discover(work)} == {"11", "22", "33"}
    frame = "TNG22_O5_Euclid_H.fits"
    assert (work / "22" / frame).read_bytes() == (store / "22" / frame).read_bytes()
    # Restored frames are new on scratch and dated so; frames in sync are untouched.
    assert (work / "33" / "TNG33_O1_Euclid_VIS.fits").stat().st_mtime >= t0 - 1
    assert (work / "11" / "TNG11_O1_Euclid_VIS.fits").stat().st_mtime == pytest.approx(OLD)
    assert not list(work.rglob(".*.part"))
    assert not (work / "44").exists()


def test_archive_then_restore_round_trips_exactly(tmp_path):
    work, store, fresh = tmp_path / "work", tmp_path / "store", tmp_path / "fresh"
    _galaxy(work, "11", twins=True, size=40_000, marker=FINISHED)
    result = archive_to_store(work, store, workers=2)
    assert result["archived"] == 1 and result["bytes_read"] == 40 * 40_000
    assert len(list((store / "11").glob("*.fits.gz"))) == 40 and not list((store / "11").glob("*.fits"))
    assert 0 < result["store_bytes"] < 40 * 40_000 / 4               # lossless and much smaller
    t0 = time.time()
    restored = sync_working_copy(store, fresh)
    assert restored["copied"] == 1 and restored["bytes"] == 40 * 40_000
    for name in _names("11", twins=True):
        assert (fresh / "11" / name).read_bytes() == (work / "11" / name).read_bytes()
        assert (fresh / "11" / name).stat().st_mtime >= t0 - 1
    assert (fresh / "11" / Config.Tng.DONE_MARKER).read_text() == FINISHED
    with gzip.open(store / "11" / "TNG11_O1_Euclid_VIS.fits.gz") as f:      # plain gzip FITS
        assert f.read() == (work / "11" / "TNG11_O1_Euclid_VIS.fits").read_bytes()


def test_only_finished_downloads_are_archived(tmp_path):
    work, store = tmp_path / "work", tmp_path / "store"
    _galaxy(work, "11", twins=True, marker=FINISHED)
    _galaxy(work, "22")                                       # dusty frames only (an old download)
    _galaxy(work, "33", twins=True, files=25, marker=FINISHED)  # download still extracting
    result = archive_to_store(work, store)
    assert (result["finished"], result["archived"], result["store_galaxies"]) == (1, 1, 1)
    assert sorted(p.name for p in store.iterdir()) == ["11"]


def test_archive_adds_only_what_the_store_lacks_and_deletes_nothing(tmp_path):
    work, store = tmp_path / "work", tmp_path / "store"
    _galaxy(work, "11", twins=True, marker=FINISHED)
    _galaxy(store, "99")                                      # only in the store: left alone
    archive_to_store(work, store)
    again = archive_to_store(work, store)
    assert (again["archived"], again["in_store"], again["bytes_read"]) == (0, 1, 0)
    frame = work / "11" / "TNG11_O3_Euclid_J_nodust.fits"
    frame.write_bytes(b"\0" * 128)                            # re-rendered at another size
    third = archive_to_store(work, store)
    assert (third["archived"], third["bytes_read"]) == (1, 128)
    with gzip.open(store / "11" / (frame.name + ".gz")) as f:
        assert f.read() == b"\0" * 128
    assert (store / "99").exists() and third["store_galaxies"] == 2
    assert not list(store.rglob(".*.part"))


def test_sync_is_idempotent_and_deletes_nothing(tmp_path):
    store, work = tmp_path / "store", tmp_path / "work"
    _galaxy(store, "11")
    extra = _galaxy(work, "99")                               # only on scratch: left alone
    sync_working_copy(store, work)
    again = sync_working_copy(store, work)
    assert again["copied"] == 0 and again["in_sync"] == 1
    assert extra.exists()


def test_no_store_yet_and_the_same_directory_are_no_ops(tmp_path):
    missing = sync_working_copy(tmp_path / "not-yet", tmp_path / "work")
    assert missing["copied"] == 0 and "no atlas store" in str(missing["note"])
    assert not (tmp_path / "work").exists()
    _galaxy(tmp_path / "atlas", "11", twins=True, marker=FINISHED)
    same = sync_working_copy(tmp_path / "atlas", tmp_path / "atlas")
    assert same["copied"] == 0 and same["store_galaxies"] == 1
    assert archive_to_store(tmp_path / "atlas", tmp_path / "atlas")["archived"] == 0


def test_the_cli_archives_and_restores(tmp_path, capsys):
    work, store, fresh = tmp_path / "work", tmp_path / "store", tmp_path / "fresh"
    _galaxy(work, "11", twins=True, marker=FINISHED)
    assert main(["archive", "--store", str(store), "--work", str(work), "--workers", "1"]) == 0
    assert "1 galaxies archived" in capsys.readouterr().out
    assert main(["--store", str(store), "--work", str(fresh)]) == 0
    assert "1 galaxies restored" in capsys.readouterr().out
    assert len(list((fresh / "11").glob("*.fits"))) == 40


def test_the_downloader_writes_to_the_working_copy(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "TNG_SKIRT_STORE_DIR", str(tmp_path / "store"))
    assert downloader.parse_args([]).out_dir == Config.TNG_SKIRT_DIR


def test_generation_restores_from_the_store_first(monkeypatch, tmp_path):
    store, work = tmp_path / "store", tmp_path / "work"
    archive_source = tmp_path / "downloads"
    _galaxy(archive_source, "11", twins=True, marker=FINISHED)
    archive_to_store(archive_source, store)
    args = argparse.Namespace(tng_dir=str(work), galaxy_density_arcmin2=0.0, lens_density_arcmin2=0.0)
    monkeypatch.setattr(Config, "TNG_SKIRT_STORE_DIR", str(store))
    run_pipeline._sync_tng_atlas_from_store(args, workers=1)        # no TNG in this run
    assert not work.exists()
    args.galaxy_density_arcmin2 = 150.0
    run_pipeline._sync_tng_atlas_from_store(args, workers=1)
    assert [g.subhalo_id for g in TNGGalaxy.discover(work)] == ["11"]
    monkeypatch.setattr(Config, "TNG_SKIRT_STORE_DIR", str(tmp_path / "not-yet"))   # store not made yet
    run_pipeline._sync_tng_atlas_from_store(args, workers=1)
    monkeypatch.setattr(Config, "TNG_SKIRT_STORE_DIR", "")             # no store: nothing to do
    run_pipeline._sync_tng_atlas_from_store(argparse.Namespace(tng_dir=str(tmp_path / "w2"),
                                                               galaxy_density_arcmin2=150.0), workers=1)
    assert not (tmp_path / "w2").exists()
