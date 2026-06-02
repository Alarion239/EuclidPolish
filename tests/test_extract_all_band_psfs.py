"""Tests for the spatial-clustering helpers + the per-PSF progress reporting
in ``scripts/extract_all_band_psfs.py`` — the K-Means++ grouping of good stars
into ~N-sized clusters (one ePSF each), the catalog-position loader, and the
parallel ``set_worker_step`` reporting (cumulative across bands)."""

from __future__ import annotations

import argparse
import importlib
import json

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.euclid.psf_extractor import PSFExtractor
from euclid_polish.observability.reporter import Reporter
from euclid_polish.psf import PSF

gen = importlib.import_module("scripts.extract_all_band_psfs")


def test_cluster_splits_into_round_n_over_k_groups():
    # 30 stars in 3 tight spatial blobs; N=10 → K=round(30/10)=3.
    # min_stars=5 so the (intended) 10-star clusters aren't merged.
    ids = list(range(30))
    positions = {i: (10.0 + (i // 10) * 0.5, 2.0 + (i % 10) * 1e-3)
                 for i in ids}
    clusters = gen.cluster_star_indices(ids, positions, stars_per_psf=10,
                                        min_stars=5)
    assert len(clusters) == 3
    assert sorted(len(c) for c in clusters) == [10, 10, 10]
    # Every id assigned exactly once.
    flat = sorted(i for c in clusters for i in c)
    assert flat == ids


def test_cluster_enforces_min_stars_floor():
    """No cluster ends up below the min floor: one sparse far-away clump is
    merged into a neighbour rather than forming a tiny under-sampled PSF."""
    # 200 stars in a tight blob + 10 stars in a far-away clump.
    ids = list(range(210))
    positions = {}
    for i in range(200):
        positions[i] = (10.0 + (i % 20) * 1e-3, 2.0 + (i // 20) * 1e-3)
    for i in range(200, 210):
        positions[i] = (50.0 + (i - 200) * 1e-3, 40.0)     # far clump of 10
    clusters = gen.cluster_star_indices(ids, positions, stars_per_psf=100,
                                        min_stars=50)
    assert all(len(c) >= 50 for c in clusters), [len(c) for c in clusters]
    # Every id still assigned exactly once.
    assert sorted(i for c in clusters for i in c) == ids


def test_cluster_returns_single_group_without_positions():
    ids = list(range(30))
    clusters = gen.cluster_star_indices(ids, {}, stars_per_psf=10)
    assert len(clusters) == 1
    assert sorted(clusters[0]) == ids


def test_cluster_single_group_when_too_few_positioned():
    ids = list(range(30))
    positions = {0: (1.0, 2.0)}            # only one star has a position
    clusters = gen.cluster_star_indices(ids, positions, stars_per_psf=10)
    assert len(clusters) == 1
    assert sorted(clusters[0]) == ids


def test_cluster_k_one_when_n_exceeds_count():
    ids = list(range(8))
    positions = {i: (10.0 + i * 0.01, 2.0) for i in ids}
    # 8 stars, N=100 → fewer than N positioned → one cluster.
    clusters = gen.cluster_star_indices(ids, positions, stars_per_psf=100)
    assert len(clusters) == 1


def test_load_star_positions_reads_id_ra_dec(tmp_path):
    csv_path = tmp_path / "stars.csv"
    csv_path.write_text(
        "id,ra,dec,magnitude\n"
        "0,150.1,2.2,19.0\n"
        "1,150.2,2.3,20.0\n"
        "2,,,\n"                # malformed row is skipped
    )
    positions = gen._load_star_positions(str(csv_path))
    assert positions[0] == pytest.approx((150.1, 2.2))
    assert positions[1] == pytest.approx((150.2, 2.3))
    assert 2 not in positions


def test_load_star_positions_missing_file_is_empty(tmp_path):
    assert gen._load_star_positions(str(tmp_path / "nope.csv")) == {}


# ---------------------------------------------------------------------------
# Phase 1 (cluster_band) + per-PSF progress reporting
# ---------------------------------------------------------------------------

class _FakeStar:
    def __init__(self, side=5):
        self.data = np.zeros((side, side), dtype=np.float32)


def _fake_args(tmp_path, **over):
    base = dict(
        num_stars=None, cutout_size=64, vis_pixels=None, output_size=None,
        psf_dir=str(tmp_path / "psf"), stars_per_psf=100, min_stars_per_psf=50,
        stars_csv=str(tmp_path / "stars.csv"), max_procs=1,
        bands="VIS,Y_E")
    base.update(over)
    return argparse.Namespace(**base)


def test_cluster_band_returns_jobs_and_gates_tqdm(tmp_path, monkeypatch):
    """``cluster_band`` loads + clusters into per-cluster build jobs (stamps +
    star count + centroid), and disables tqdm under a job (events path set)."""
    cutdir = tmp_path / "cutouts"
    cutdir.mkdir()
    monkeypatch.setattr(gen, "_cutout_dir_for_band", lambda b: str(cutdir))
    monkeypatch.setattr(gen, "_load_star_positions", lambda csv: {})
    monkeypatch.setattr(gen, "cluster_star_indices",
                        lambda ids, pos, spp, min_stars=50:
                        [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    monkeypatch.setattr(PSFExtractor, "get_cutout_files",
                        lambda self, d, cutout_size=None:
                        [(i, f"f{i}") for i in range(9)])
    monkeypatch.setattr(PSFExtractor, "extract_accepted_stars",
                        lambda self, files: [(i, _FakeStar()) for i in range(9)])

    seen = {}
    orig_init = PSFExtractor.__init__

    def spy_init(self, config=None):
        seen["progress_bar"] = config.progress_bar if config else None
        orig_init(self, config)
    monkeypatch.setattr(PSFExtractor, "__init__", spy_init)

    # Job context (events path) → tqdm OFF.
    plan = gen.cluster_band(Config.BAND_VIS, _fake_args(tmp_path),
                            Reporter(events_path=str(tmp_path / "j.events")))
    assert seen["progress_bar"] is False
    assert len(plan["jobs"]) == 3
    for ci, stamps, n, centroid in plan["jobs"]:
        assert len(stamps) == 3 and n == 3
        assert all(isinstance(s, np.ndarray) for s in stamps)

    # Interactive (no events path) → tqdm ON.
    gen.cluster_band(Config.BAND_VIS, _fake_args(tmp_path),
                     Reporter(events_path=None))
    assert seen["progress_bar"] is True


def test_build_cluster_psf_builds_from_stamps(monkeypatch):
    """``_build_cluster_psf`` rebuilds EPSFStars from the stamp arrays and
    returns a PSF (EPSFBuilder stubbed)."""
    monkeypatch.setattr(PSFExtractor, "build_epsf_from_stars",
                        lambda self, stars: (object(), None))
    monkeypatch.setattr(
        PSFExtractor, "psf_from_epsf",
        staticmethod(lambda epsf, scale:
                     PSF(data=np.full((5, 5), 1.0 / 25, np.float32),
                         pixel_scale=scale)))
    from euclid_polish.euclid.psf_extractor import PSFExtractionConfig
    cfg = PSFExtractionConfig(psf_size=5, progress_bar=False)
    psf = gen._build_cluster_psf((cfg, 0.025,
                                  [np.ones((5, 5), np.float32) for _ in range(3)]))
    assert isinstance(psf, PSF)
    assert psf.pixel_scale == 0.025


def test_main_reports_one_monotonic_bar_over_all_psfs(tmp_path, monkeypatch):
    """``main`` builds every cluster in one pool (serial here) and reports a
    single monotonic set_step bar over ALL PSFs across the bands (3 VIS + 2
    Y_E = 5), in band order."""
    def fake_plan(band, n):
        return {"cfg": None, "scale": band.epsf_pixel_scale_arcsec,
                "filename": band.psf_fits_filename,
                "jobs": [(ci, [np.ones((5, 5), np.float32)], 3, (1.0, 2.0))
                         for ci in range(n)],
                "have_positions": True}

    plans = {"VIS": fake_plan(Config.BAND_VIS, 3),
             "Y_E": fake_plan(Config.get_band("Y_E"), 2)}
    monkeypatch.setattr(gen, "cluster_band",
                        lambda band, args, reporter: plans.get(band.name))
    monkeypatch.setattr(gen, "_build_cluster_psf",
                        lambda payload: PSF(
                            data=np.full((5, 5), 1.0 / 25, np.float32),
                            pixel_scale=payload[1]))
    monkeypatch.setattr(gen, "parse_args",
                        lambda: _fake_args(tmp_path, bands="VIS,Y_E", max_procs=1))

    ev = tmp_path / "job.events"
    monkeypatch.setenv("EUCLID_POLISH_EVENTS_PATH", str(ev))
    assert gen.main() == 0

    steps = [json.loads(l)["value"] for l in ev.read_text().splitlines()
             if json.loads(l)["kind"] == "step"]
    # The building stage uses total = 5 (3 VIS + 2 Y_E).
    build = [s["current"] for s in steps if s["total"] == 5]
    assert build == [0, 1, 2, 3, 4, 5]              # monotonic, one per PSF
    # Both bands' PSFSets were written.
    assert (tmp_path / "psf" / Config.BAND_VIS.psf_fits_filename).exists()
