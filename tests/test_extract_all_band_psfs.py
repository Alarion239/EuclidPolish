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
from euclid_polish.web.job_status import fold_events

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
# Per-PSF progress reporting (parallel worker steps)
# ---------------------------------------------------------------------------

def test_extract_band_reports_a_worker_step_per_psf(tmp_path, monkeypatch):
    """``extract_band`` emits one ``set_worker_step`` per cluster PSF (keyed by
    band name), so the bar advances as each PSF lands — exercised with the
    heavy EPSFBuilder / IO stubbed out."""
    band = Config.BAND_VIS
    cutdir = tmp_path / "cutouts"
    cutdir.mkdir()

    # Stub the IO + heavy build so only the reporting wiring runs.
    monkeypatch.setattr(gen, "_cutout_dir_for_band", lambda b: str(cutdir))
    monkeypatch.setattr(gen, "_load_star_positions", lambda csv: {})
    monkeypatch.setattr(gen, "cluster_star_indices",
                        lambda ids, pos, spp, min_stars=50:
                        [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    monkeypatch.setattr(PSFExtractor, "get_cutout_files",
                        lambda self, d, cutout_size=None:
                        [(i, f"f{i}") for i in range(9)])
    monkeypatch.setattr(PSFExtractor, "extract_accepted_stars",
                        lambda self, files: [(i, object()) for i in range(9)])
    monkeypatch.setattr(PSFExtractor, "build_epsf_from_stars",
                        lambda self, stars: (object(), None))
    monkeypatch.setattr(
        PSFExtractor, "psf_from_epsf",
        staticmethod(lambda epsf, scale:
                     PSF(data=np.full((5, 5), 1.0 / 25, np.float32),
                         pixel_scale=scale)))

    ev = tmp_path / "job.events"
    reporter = Reporter(events_path=str(ev))
    args = argparse.Namespace(
        num_stars=None, cutout_size=64, vis_pixels=None, output_size=None,
        psf_dir=str(tmp_path / "psf"), stars_per_psf=100, min_stars_per_psf=50,
        stars_csv=str(tmp_path / "stars.csv"))

    assert gen.extract_band(band, args, reporter=reporter) is True

    worker = [json.loads(l)["value"] for l in ev.read_text().splitlines()
              if json.loads(l)["kind"] == "worker"]
    vis = [(w["current"], w["total"]) for w in worker
           if w["worker_id"] == band.name]
    assert (0, 3) in vis                     # announced up front
    assert [c for c, _ in vis] == [0, 1, 2, 3]   # one step per PSF, to 3/3
    assert all(t == 3 for _, t in vis)


def test_parallel_worker_steps_sum_across_bands(tmp_path):
    """Two bands reporting interleaved worker steps fold into ONE cumulative
    bar (the consumer sums ``current`` across worker ids) — the property the
    parallel band extraction relies on."""
    ev = tmp_path / "job.events"
    r = Reporter(events_path=str(ev))
    r.set_parallel(0, 2, label="extract")     # total=0 → summed from workers
    r.set_worker_step("VIS", 0, 3)
    r.set_worker_step("Y_E", 0, 2)
    r.set_worker_step("VIS", 1, 3)
    r.set_worker_step("Y_E", 1, 2)
    r.set_worker_step("VIS", 2, 3)
    r.set_worker_step("Y_E", 2, 2)
    r.set_worker_step("VIS", 3, 3)

    st = fold_events(ev.read_text())
    assert st.parallel is not None
    assert st.parallel.current == 5          # 3 (VIS) + 2 (Y_E)
    assert st.parallel.total == 5            # summed per-band totals
    assert st.step.current == 5 and st.step.total == 5   # bar mirrors it
