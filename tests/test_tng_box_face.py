"""Galaxies near a face of TNG50-1's periodic box lost every star beyond it in
the SKIRT atlas (a straight edge through their light): they are no donors."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.tng import box_face
from euclid_polish.tng.atlas import TNGGalaxy, _scan_complete_galaxies
from scripts import run_pipeline


def _galaxy(root: Path, gid: str) -> None:
    folder = root / gid
    folder.mkdir(parents=True)
    frame = np.ones((8, 8), dtype=">f4")
    for o in range(1, 6):
        for band in ("VIS", "Y", "J", "H"):
            fits.PrimaryHDU(frame).writeto(folder / f"TNG{gid}_O{o}_Euclid_{band}.fits")
            fits.PrimaryHDU(frame).writeto(folder / f"TNG{gid}_O{o}_Euclid_{band}_nodust.fits")
    (folder / Config.Tng.DONE_MARKER).write_text("40 fits with-nodust\n")


def test_the_table_lists_the_cut_galaxies():
    distances = box_face.face_distances_kpc()
    assert distances["229934"] == pytest.approx(16.24)       # straight edge through its halo
    assert distances["580250"] == pytest.approx(0.66)        # cut through its disc
    assert all(0.0 <= d < box_face.table_reach_kpc() for d in distances.values())
    assert box_face.table_reach_kpc() >= Config.Tng.MIN_BOX_FACE_DISTANCE_KPC
    assert len(box_face.cut_galaxies()) == 21


def test_cut_galaxies_are_no_donors(tmp_path):
    for gid in ("229934", "622298", "1"):                    # 16 kpc, 119 kpc, not near a face
        _galaxy(tmp_path, gid)
    assert [g.subhalo_id for g in _scan_complete_galaxies(tmp_path)] == ["1", "622298"]
    assert [g.subhalo_id for g in TNGGalaxy.discover(tmp_path)] == ["1", "622298"]
    assert [g.subhalo_id for g in _scan_complete_galaxies(tmp_path, include_box_galaxies=True)] == [
        "1", "229934", "622298"]


def test_the_distance_limit_sets_who_is_left_out(tmp_path, monkeypatch):
    for gid in ("229934", "546114"):                         # 16 and 23 kpc from a face
        _galaxy(tmp_path, gid)
    monkeypatch.setattr(Config.Tng, "MIN_BOX_FACE_DISTANCE_KPC", 20.0)
    assert [g.subhalo_id for g in _scan_complete_galaxies(tmp_path)] == ["546114"]
    monkeypatch.setattr(Config.Tng, "MIN_BOX_FACE_DISTANCE_KPC", 200.0)
    with pytest.raises(ValueError, match="reach"):
        box_face.cut_galaxies()


def test_generation_logs_the_galaxies_it_leaves_out(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Config.Tng, "REQUIRE_DUST_FREE", True)
    for gid in ("229934", "1"):
        _galaxy(tmp_path, gid)
    args = argparse.Namespace(tng_dir=str(tmp_path), galaxy_density_arcmin2=150.0, lens_density_arcmin2=0.0)
    run_pipeline._refresh_box_light_table(args, workers=1)
    assert "TNG periodic box: 1 galaxies left out as donors" in capsys.readouterr().out
    assert [g.subhalo_id for g in _scan_complete_galaxies(tmp_path)] == ["1"]
