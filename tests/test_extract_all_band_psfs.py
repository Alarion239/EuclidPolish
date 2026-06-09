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
# Phase 1 (load_accepted_band) + common clustering + per-PSF reporting
# ---------------------------------------------------------------------------

class _FakeStar:
    def __init__(self, side=5):
        self.data = np.zeros((side, side), dtype=np.float32)


def _fake_args(tmp_path, **over):
    base = dict(
        num_stars=None, cutout_size=64, vis_pixels=None, output_size=None,
        psf_dir=str(tmp_path / "psf"), stars_per_psf=100, min_stars_per_psf=50,
        stars_csv=str(tmp_path / "stars.csv"), max_procs=1,
        bands="VIS,Y_E", cache_dir=None, keep_cache=False)
    base.update(over)
    return argparse.Namespace(**base)


def test_load_accepted_band_returns_stamps_and_gates_tqdm(tmp_path, monkeypatch):
    """``load_accepted_band`` returns ``{star_id: stamp}`` for the accepted
    stars and disables tqdm under a job (events path set)."""
    cutdir = tmp_path / "cutouts"
    cutdir.mkdir()
    monkeypatch.setattr(gen, "_cutout_dir_for_band", lambda b: str(cutdir))
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
    d = gen.load_accepted_band(Config.BAND_VIS, _fake_args(tmp_path),
                               Reporter(events_path=str(tmp_path / "j.events")))
    assert seen["progress_bar"] is False
    assert set(d["accepted"]) == set(range(9))
    assert all(isinstance(s, np.ndarray) for s in d["accepted"].values())

    # Interactive (no events path) → tqdm ON.
    gen.load_accepted_band(Config.BAND_VIS, _fake_args(tmp_path),
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


def _patch_main(tmp_path, monkeypatch, accepts, idx_clusters, positions=None):
    """Stub load_accepted_band / clustering / build / args so ``main`` runs
    over fake data. ``accepts`` maps band name → set of accepted star ids."""
    def fake_band(band, args, reporter):
        return {"cfg": None, "scale": band.epsf_pixel_scale_arcsec,
                "filename": band.psf_fits_filename,
                "accepted": {i: np.ones((5, 5), np.float32)
                             for i in accepts[band.name]}}
    monkeypatch.setattr(gen, "load_accepted_band", fake_band)
    monkeypatch.setattr(gen, "_load_star_positions",
                        lambda csv: positions if positions is not None
                        else {i: (10.0 + i * 0.01, 2.0) for i in range(6)})
    monkeypatch.setattr(gen, "cluster_star_indices",
                        lambda ids, pos, spp, min_stars=50: idx_clusters)
    monkeypatch.setattr(gen, "parse_args",
                        lambda: _fake_args(tmp_path, bands="VIS,Y_E", max_procs=1))


def test_main_builds_shared_clusters_with_one_bar(tmp_path, monkeypatch):
    """All four bands share the clustering: 2 bands × 2 clusters build under one
    monotonic bar, and both bands are saved with identical (consistent)
    numbering + star counts."""
    _patch_main(tmp_path, monkeypatch,
                accepts={"VIS": set(range(6)), "Y_E": set(range(6))},
                idx_clusters=[[0, 1, 2], [3, 4, 5]])
    monkeypatch.setattr(gen, "_build_cluster_psf",
                        lambda payload: PSF(
                            data=np.full((5, 5), 1.0 / 25, np.float32),
                            pixel_scale=payload[1]))
    ev = tmp_path / "job.events"
    monkeypatch.setenv("EUCLID_POLISH_EVENTS_PATH", str(ev))
    assert gen.main() == 0

    steps = [json.loads(l)["value"] for l in ev.read_text().splitlines()
             if json.loads(l)["kind"] == "step"]
    build = [s["current"] for s in steps if s["total"] == 4]   # 2 bands × 2
    assert build == [0, 1, 2, 3, 4]

    from euclid_polish.psf import PSFSet
    vis = PSFSet.from_fits(str(tmp_path / "psf" / Config.BAND_VIS.psf_fits_filename))
    y_e = PSFSet.from_fits(str(tmp_path / "psf" / Config.get_band("Y_E").psf_fits_filename))
    assert vis.n == y_e.n == 2
    assert vis.n_stars == y_e.n_stars == [3, 3]    # same clustering → same counts


def test_main_clusters_only_all_band_common_stars(tmp_path, monkeypatch):
    """Only stars accepted in EVERY band are clustered/built — so a band's
    cluster ci is the same stars in every band."""
    # VIS accepts 0..3, Y_E accepts 2..5 → common = {2, 3}.
    _patch_main(tmp_path, monkeypatch,
                accepts={"VIS": {0, 1, 2, 3}, "Y_E": {2, 3, 4, 5}},
                idx_clusters=[[0, 1]])             # one cluster of the 2 common
    built_sizes = []
    monkeypatch.setattr(gen, "_build_cluster_psf",
                        lambda payload: built_sizes.append(len(payload[2])) or
                        PSF(data=np.full((5, 5), 1.0 / 25, np.float32),
                            pixel_scale=payload[1]))
    monkeypatch.setenv("EUCLID_POLISH_EVENTS_PATH", str(tmp_path / "e.events"))
    assert gen.main() == 0

    assert built_sizes == [2, 2]                   # each band built the 2 common stars
    from euclid_polish.psf import PSFSet
    vis = PSFSet.from_fits(str(tmp_path / "psf" / Config.BAND_VIS.psf_fits_filename))
    assert vis.n == 1 and vis.n_stars == [2]


def test_resume_reuses_cached_epsfs(tmp_path, monkeypatch):
    """A time-limited job that is re-submitted resumes: cached (band, cluster)
    ePSFs are reused and only the missing one is rebuilt."""
    _patch_main(tmp_path, monkeypatch,
                accepts={"VIS": set(range(6)), "Y_E": set(range(6))},
                idx_clusters=[[0, 1, 2], [3, 4, 5]])
    calls = {"n": 0}

    def fake_build(payload):
        calls["n"] += 1
        return PSF(data=np.full((5, 5), 1.0 / 25, np.float32),
                   pixel_scale=payload[1])
    monkeypatch.setattr(gen, "_build_cluster_psf", fake_build)
    # keep the cache after the (successful) first run so we can resume
    monkeypatch.setattr(gen, "parse_args",
                        lambda: _fake_args(tmp_path, bands="VIS,Y_E",
                                           max_procs=1, keep_cache=True))

    # Run 1 — builds all 4 (2 bands × 2 clusters); cache retained.
    assert gen.main() == 0
    assert calls["n"] == 4
    cache = tmp_path / "psf" / ".epsf_cache"
    cached = sorted(p.name for p in cache.glob("*_PSF*.fits"))
    assert len(cached) == 4

    # Simulate a stop that left one ePSF unbuilt: drop one cache slot.
    (cache / cached[0]).unlink()

    # Run 2 — only the missing one is rebuilt; both bands' FITS still written
    # with consistent numbering.
    calls["n"] = 0
    assert gen.main() == 0
    assert calls["n"] == 1                          # resumed, not restarted

    from euclid_polish.psf import PSFSet
    vis = PSFSet.from_fits(str(tmp_path / "psf" / Config.BAND_VIS.psf_fits_filename))
    y_e = PSFSet.from_fits(str(tmp_path / "psf" / Config.get_band("Y_E").psf_fits_filename))
    assert vis.n == y_e.n == 2


def test_changed_inputs_invalidate_cache(tmp_path, monkeypatch):
    """If the clustering/params change, the stale cache is dropped (no reuse of
    a kernel that no longer matches its (band, cluster) slot)."""
    _patch_main(tmp_path, monkeypatch,
                accepts={"VIS": set(range(6)), "Y_E": set(range(6))},
                idx_clusters=[[0, 1, 2], [3, 4, 5]])
    calls = {"n": 0}

    def fake_build(payload):
        calls["n"] += 1
        return PSF(data=np.full((5, 5), 1.0 / 25, np.float32),
                   pixel_scale=payload[1])
    monkeypatch.setattr(gen, "_build_cluster_psf", fake_build)
    monkeypatch.setattr(gen, "parse_args",
                        lambda: _fake_args(tmp_path, bands="VIS,Y_E",
                                           max_procs=1, keep_cache=True))
    assert gen.main() == 0
    assert calls["n"] == 4

    # Re-cluster differently → signature changes → cache invalidated → rebuild.
    monkeypatch.setattr(gen, "cluster_star_indices",
                        lambda ids, pos, spp, min_stars=50: [[0, 1, 2, 3, 4, 5]])
    calls["n"] = 0
    assert gen.main() == 0
    assert calls["n"] == 2                          # 1 cluster × 2 bands, fresh
