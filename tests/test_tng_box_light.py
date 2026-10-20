"""Galaxies whose SKIRT frames carry the octree dust box (light where no star
is) are left out as donors; a table at the atlas root records the share."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.tng import box_light
from euclid_polish.tng.atlas import TNGGalaxy, _scan_complete_galaxies
from scripts import run_pipeline

SIZE = 64
BANDS = ("VIS", "Y", "J", "H")


def _stars() -> np.ndarray:
    yy, xx = np.mgrid[:SIZE, :SIZE]
    r = np.hypot(yy - SIZE / 2, xx - SIZE / 2)
    return np.where(r < 12, np.exp(-r / 3.0), 0.0)


def _galaxy(root: Path, gid: str, box: float) -> Path:
    """Dusty frames = stars + a flat box of scattered light; twins = stars."""
    folder = root / gid
    folder.mkdir(parents=True)
    for o in range(1, 6):
        for band in BANDS:
            fits.PrimaryHDU((_stars() + box).astype(">f4")).writeto(folder / f"TNG{gid}_O{o}_Euclid_{band}.fits")
            fits.PrimaryHDU(_stars().astype(">f4")).writeto(folder / f"TNG{gid}_O{o}_Euclid_{band}_nodust.fits")
    (folder / Config.Tng.DONE_MARKER).write_text("40 fits with-nodust\n")
    return folder


def _expected_share(box: float) -> float:
    dusty = _stars() + box
    return float(dusty[_stars() == 0].sum() / dusty.sum())


def test_the_share_of_light_where_no_star_is(tmp_path):
    _galaxy(tmp_path, "1", box=1e-3)
    _galaxy(tmp_path, "2", box=0.0)
    boxed, clean = sorted(_scan_complete_galaxies(tmp_path), key=lambda g: g.subhalo_id)
    assert box_light.outside_star_fractions(boxed) == pytest.approx([_expected_share(1e-3)] * 5, rel=1e-5)
    assert box_light.outside_star_fractions(clean) == [0.0] * 5


def test_box_galaxies_are_no_donors(tmp_path, monkeypatch):
    monkeypatch.setattr(Config.Tng, "REQUIRE_DUST_FREE", True)
    _galaxy(tmp_path, "1", box=1e-3)                  # ~7 % of its light in the box
    _galaxy(tmp_path, "2", box=0.0)
    result = box_light.refresh(tmp_path, workers=2)
    assert (result["galaxies"], result["measured"], result["left_out"]) == (2, 2, ["1"])
    assert [g.subhalo_id for g in _scan_complete_galaxies(tmp_path)] == ["2"]
    assert [g.subhalo_id for g in TNGGalaxy.discover(tmp_path)] == ["2"]
    assert [g.subhalo_id for g in _scan_complete_galaxies(tmp_path, include_box_galaxies=True)] == ["1", "2"]
    table = json.loads((tmp_path / Config.Tng.BOX_LIGHT_TABLE).read_text())
    assert table["galaxies"]["1"]["fractions"] == pytest.approx([_expected_share(1e-3)] * 5, rel=1e-4)


def test_refresh_measures_only_new_or_changed_galaxies(tmp_path):
    folder = _galaxy(tmp_path, "1", box=1e-3)
    box_light.refresh(tmp_path)
    again = box_light.refresh(tmp_path)
    assert (again["measured"], again["reused"]) == (0, 1)
    frame = folder / "TNG1_O3_Euclid_VIS.fits"
    os.utime(frame, ns=(frame.stat().st_atime_ns, frame.stat().st_mtime_ns + 10**9))
    _galaxy(tmp_path, "2", box=0.0)
    third = box_light.refresh(tmp_path)
    assert (third["measured"], third["reused"]) == (2, 0)


def test_generation_on_fasrc_needs_the_table_and_keeps_one_atlas_per_run(tmp_path, monkeypatch):
    _galaxy(tmp_path, "1", box=0.0)
    assert [g.subhalo_id for g in _scan_complete_galaxies(tmp_path)] == ["1"]   # no table: atlas as it is
    monkeypatch.setattr(Config.Tng, "REQUIRE_DUST_FREE", True)
    with pytest.raises(FileNotFoundError, match="box_light"):
        _scan_complete_galaxies(tmp_path)
    box_light.refresh(tmp_path)
    _galaxy(tmp_path, "2", box=0.0)                   # finished downloading after the table was written
    assert [g.subhalo_id for g in _scan_complete_galaxies(tmp_path)] == ["1"]   # joins at the next refresh
    box_light.refresh(tmp_path)
    assert [g.subhalo_id for g in _scan_complete_galaxies(tmp_path)] == ["1", "2"]


def test_generation_refreshes_the_table_first(tmp_path, monkeypatch):
    monkeypatch.setattr(Config.Tng, "REQUIRE_DUST_FREE", True)
    _galaxy(tmp_path, "1", box=1e-3)
    args = argparse.Namespace(tng_dir=str(tmp_path), galaxy_density_arcmin2=0.0, lens_density_arcmin2=0.0)
    run_pipeline._refresh_box_light_table(args, workers=1)                  # no TNG in this run
    assert not (tmp_path / Config.Tng.BOX_LIGHT_TABLE).exists()
    args.galaxy_density_arcmin2 = 150.0
    run_pipeline._refresh_box_light_table(args, workers=1)
    assert _scan_complete_galaxies(tmp_path) == ()


def test_on_fasrc_a_galaxy_counts_once_its_download_finished(tmp_path, monkeypatch):
    """Every folder keeps an untagged marker from the dusty-only download; the
    tagged one is written after the last frame of the new download."""
    monkeypatch.setattr(Config.Tng, "REQUIRE_DUST_FREE", True)
    folder = _galaxy(tmp_path, "1", box=0.0)
    (folder / Config.Tng.DONE_MARKER).write_text("20 fits\n")             # frames still arriving
    box_light.refresh(tmp_path)
    assert _scan_complete_galaxies(tmp_path, include_box_galaxies=True) == ()
    (folder / Config.Tng.DONE_MARKER).write_text("40 fits with-nodust\n")  # download finished
    box_light.refresh(tmp_path)
    assert [g.subhalo_id for g in _scan_complete_galaxies(tmp_path)] == ["1"]
