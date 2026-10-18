"""The TNG atlas lives in a persistent store (lab storage); generation copies
the galaxies the scratch copy lacks from it, and the downloader writes to it."""
from __future__ import annotations

import argparse
import os

import pytest

from euclid_polish.config import Config
from euclid_polish.tng.atlas import TNGGalaxy
from euclid_polish.tng.atlas_store import sync_working_copy
from euclid_polish.tng.types import TNG_FITS_BANDS
from scripts import fasrc_download_tng_skirt_atlas as downloader
from scripts import run_pipeline


def _galaxy(root, gid: str, *, files: int = 20, size: int = 64, mtime: float = 1_700_000_000.0):
    directory = root / gid
    directory.mkdir(parents=True, exist_ok=True)
    names = [f"TNG{gid}_O{o}_Euclid_{b}.fits" for o in range(1, 6) for b in TNG_FITS_BANDS][:files]
    for name in names:
        path = directory / name
        path.write_bytes(bytes([len(name) % 251]) * size)
        os.utime(path, (mtime, mtime))
    (directory / Config.Tng.DONE_MARKER).write_text(f"{files} fits\n")
    return directory


def test_sync_copies_missing_and_partial_galaxies_and_keeps_times(tmp_path):
    store, work = tmp_path / "store", tmp_path / "work"
    for gid in ("11", "22", "33"):
        _galaxy(store, gid)
    _galaxy(store, "44", files=7)                      # incomplete in the store: never copied
    _galaxy(work, "11")                                # already in sync
    partial = _galaxy(work, "22")
    (partial / "TNG22_O5_Euclid_H.fits").write_bytes(b"x")      # truncated
    result = sync_working_copy(store, work, workers=2)
    assert result == {"store_galaxies": 3, "copied": 2, "in_sync": 1, "bytes": 2 * 20 * 64}
    assert {g.subhalo_id for g in TNGGalaxy.discover(work)} == {"11", "22", "33"}
    assert (work / "22" / "TNG22_O5_Euclid_H.fits").stat().st_size == 64
    assert (work / "33" / "TNG33_O1_Euclid_VIS.fits").stat().st_mtime == pytest.approx(1_700_000_000.0)
    assert not list(work.rglob(".*.part"))
    assert not (work / "44").exists()


def test_sync_is_idempotent_and_deletes_nothing(tmp_path):
    store, work = tmp_path / "store", tmp_path / "work"
    _galaxy(store, "11")
    extra = _galaxy(work, "99")                       # only on scratch: left alone
    sync_working_copy(store, work)
    again = sync_working_copy(store, work)
    assert again["copied"] == 0 and again["in_sync"] == 1
    assert extra.exists()


def test_the_same_directory_is_a_no_op(tmp_path):
    _galaxy(tmp_path / "atlas", "11")
    result = sync_working_copy(tmp_path / "atlas", tmp_path / "atlas")
    assert result["copied"] == 0 and result["store_galaxies"] == 1


def test_the_downloader_writes_to_the_store(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "TNG_SKIRT_STORE_DIR", str(tmp_path / "store"))
    assert downloader.parse_args([]).out_dir == str(tmp_path / "store")
    monkeypatch.setattr(Config, "TNG_SKIRT_STORE_DIR", "")
    assert downloader.parse_args([]).out_dir == Config.TNG_SKIRT_DIR


def test_generation_copies_from_the_store_first(monkeypatch, tmp_path):
    store, work = tmp_path / "store", tmp_path / "work"
    _galaxy(store, "11")
    args = argparse.Namespace(tng_dir=str(work), galaxy_density_arcmin2=0.0, lens_density_arcmin2=0.0)
    monkeypatch.setattr(Config, "TNG_SKIRT_STORE_DIR", str(store))
    run_pipeline._sync_tng_atlas_from_store(args, workers=1)        # no TNG in this run
    assert not work.exists()
    args.galaxy_density_arcmin2 = 150.0
    run_pipeline._sync_tng_atlas_from_store(args, workers=1)
    assert [g.subhalo_id for g in TNGGalaxy.discover(work)] == ["11"]
    monkeypatch.setattr(Config, "TNG_SKIRT_STORE_DIR", "")             # no store: nothing to do
    run_pipeline._sync_tng_atlas_from_store(argparse.Namespace(tng_dir=str(tmp_path / "w2"),
                                                               galaxy_density_arcmin2=150.0), workers=1)
    assert not (tmp_path / "w2").exists()
