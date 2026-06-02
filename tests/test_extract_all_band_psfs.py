"""Tests for the spatial-clustering helpers in
``scripts/extract_all_band_psfs.py`` — the K-Means++ grouping of good stars
into ~N-sized clusters (one ePSF each) and the catalog-position loader."""

from __future__ import annotations

import importlib

import pytest

gen = importlib.import_module("scripts.extract_all_band_psfs")


def test_cluster_splits_into_round_n_over_k_groups():
    # 30 stars in 3 tight spatial blobs; N=10 → K=round(30/10)=3.
    ids = list(range(30))
    positions = {i: (10.0 + (i // 10) * 0.5, 2.0 + (i % 10) * 1e-3)
                 for i in ids}
    clusters = gen.cluster_star_indices(ids, positions, stars_per_psf=10)
    assert len(clusters) == 3
    assert sorted(len(c) for c in clusters) == [10, 10, 10]
    # Every id assigned exactly once.
    flat = sorted(i for c in clusters for i in c)
    assert flat == ids


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
