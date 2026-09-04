"""Focused, offline tests for the independent-parent VIS sampler."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from astropy.io import fits

from scripts import fasrc_download_euclid_sky_cutouts as sampling


def _archive_row(*, oid: str = "oid-1", release: str = "Q1_R1") -> dict:
    return {
        "mosaic_product_oid": oid,
        "release_name": release,
        "product_type": "DpdMerBksMosaic",
        "fov": "POLYGON ICRS 9 19 11 19 11 21 9 21",
        "file_path": "/archive/q1",
        "file_name": f"{oid}.fits",
        "tile_index": "1007",
        "instrument_name": "VIS",
        "filter_name": "VIS",
        "technique": "IMAGE",
        "ra": 10.0,
        "dec": 20.0,
    }


def test_archive_query_freezes_release_product_and_returns_provenance():
    seen = {}

    def query_runner(query):
        seen["query"] = query
        return [_archive_row()], ""

    parents, error = sampling.exact_vis_parents(
        10.0, 20.0, 0.2, source_release="Q1_R1", query_runner=query_runner,
    )

    assert error == ""
    assert len(parents) == 1
    assert "mosaic_product_oid, release_name, product_type, fov" in seen["query"]
    assert "product_type = 'DpdMerBksMosaic'" in seen["query"]
    assert "release_name = 'Q1_R1'" in seen["query"]
    assert "INTERSECTS(mosaic_product.fov, CIRCLE('ICRS'" in seen["query"]
    assert parents[0]["mosaic_product_oid"] == "oid-1"
    assert parents[0]["release_name"] == "Q1_R1"
    assert parents[0]["product_type"] == "DpdMerBksMosaic"
    assert parents[0]["coverage_clearance_deg"] > 0.9


def test_archive_prefilter_hit_is_rejected_when_circle_crosses_fov_edge():
    parents, error = sampling.exact_vis_parents(
        10.9,
        20.0,
        0.2,
        source_release="Q1_R1",
        query_runner=lambda _query: ([_archive_row()], ""),
    )

    assert parents == []
    assert "did not fully contain" in error


def test_parent_identity_includes_product_oid_and_release():
    first, _ = sampling.exact_vis_parents(
        10.0,
        20.0,
        0.2,
        source_release="Q1_R1",
        query_runner=lambda _query: ([_archive_row(oid="oid-1")], ""),
    )
    second, _ = sampling.exact_vis_parents(
        10.0,
        20.0,
        0.2,
        source_release="Q1_R1",
        query_runner=lambda _query: ([_archive_row(oid="oid-2")], ""),
    )

    assert first[0]["parent_id"] != second[0]["parent_id"]


def test_unique_parent_assignment_uses_alternate_candidate():
    candidates = pd.DataFrame.from_records([
        {"anchor_id": "a", "field": "EDF-N", "slot": 0, "candidate_rank": 0,
         "ra": 10.0, "dec": 20.0},
        {"anchor_id": "b", "field": "EDF-S", "slot": 0, "candidate_rank": 0,
         "ra": 20.0, "dec": -20.0},
        {"anchor_id": "b", "field": "EDF-S", "slot": 0, "candidate_rank": 1,
         "ra": 21.0, "dec": -20.0},
    ])

    def resolve(ra, _dec, _radius):
        parent_id = "parent-2" if ra == 21.0 else "parent-1"
        return [{"parent_id": parent_id, "tile_index": parent_id}], ""

    selected, rejections = sampling.assign_unique_parents(
        candidates,
        samples_per_anchor=1,
        cutout_radius_deg=0.05,
        minimum_sample_separation_arcmin=0.0,
        parent_resolver=resolve,
    )

    assert [row["parent_id"] for row in selected] == ["parent-1", "parent-2"]
    assert selected[1]["candidate_rank"] == 1
    assert rejections["parent mosaic already selected"] == 1


def test_equal_area_support_discards_star_density_as_a_weight():
    dense = pd.DataFrame({
        "ra": np.r_[10.0 + np.arange(50) * 1e-5, 20.0],
        "dec": np.r_[np.full(50, 20.0), 20.0],
        "field": ["EDF-N"] * 51,
    })

    support = sampling._equal_area_support(dense, cell_area_deg2=0.04)

    assert len(support) == 2
    assert sorted(support["star_count"].tolist()) == [1, 50]
    # K-means consumes these two rows, so the 50-star cell and one-star cell
    # each provide exactly one coverage-support vote.


def test_q1_deep_field_coordinate_labels_are_not_swapped():
    stars = pd.DataFrame({
        "ra": [61.241, 52.932, 269.733],
        "dec": [-48.423, -28.088, 66.018],
    })

    labelled = sampling._assign_q1_regions(stars)

    assert labelled["field"].tolist() == ["EDF-F", "EDF-S", "EDF-N"]


def test_star_support_default_has_dedicated_non_roundtrip_root(monkeypatch, tmp_path):
    monkeypatch.setattr(sampling.Config, "EUCLID_SKY_DIR", str(tmp_path / "euclid_sky"))

    assert sampling.default_vis_noise_output_dir() == str(
        tmp_path / "euclid_sky" / "vis_noise_samples"
    )


def test_plan_fingerprint_freezes_stars_and_every_selection_input(tmp_path):
    stars = tmp_path / "stars.csv"
    stars.write_text("ra,dec\n10,20\n")
    args = SimpleNamespace(
        star_support_csv=str(stars),
        source_release="Q1_R1",
        seed=42,
        n_clusters=44,
        samples_per_cluster=1,
        support_cell_area_deg2=0.04,
        jitter_radius_deg=0.15,
        avoid_star_arcsec=30.0,
        minimum_separation_arcmin=6.5,
        candidates_per_sample=24,
        vis_pixels=2560,
    )
    original = sampling._star_support_plan(args)

    args.seed = 43
    changed_input = sampling._star_support_plan(args)
    args.seed = 42
    stars.write_text("ra,dec\n10,20\n11,21\n")
    changed_stars = sampling._star_support_plan(args)

    assert sampling._plan_fingerprint(original) != sampling._plan_fingerprint(changed_input)
    assert sampling._plan_fingerprint(original) != sampling._plan_fingerprint(changed_stars)


def test_cached_sample_requires_matching_parent_release_and_position(tmp_path):
    path = tmp_path / "sky_0000.fits"
    header = fits.Header()
    header["POS_ID"] = 0
    header["VIS_PIX"] = 8
    header["PARENT"] = "parent-1"
    header["RELEASE"] = "Q1_R1"
    header["HIERARCH MOSAIC_PRODUCT_OID"] = "oid-1"
    header["PRODTYPE"] = "DpdMerBksMosaic"
    header["RA"] = 10.0
    header["DEC"] = 20.0
    fits.HDUList([
        fits.PrimaryHDU(header=header),
        fits.ImageHDU(np.zeros((9, 7), dtype=np.float32), name="VIS"),
    ]).writeto(path)
    sample = {
        "sample_id": 0,
        "parent_id": "parent-1",
        "ra": 10.0,
        "dec": 20.0,
        "parent": {
            "release_name": "Q1_R1",
            "mosaic_product_oid": "oid-1",
            "product_type": "DpdMerBksMosaic",
        },
    }

    assert sampling._cached_planned_bundle_matches(
        str(path), sample, vis_pixels=8, source_release="Q1_R1",
    )
    sample["parent_id"] = "different-parent"
    assert not sampling._cached_planned_bundle_matches(
        str(path), sample, vis_pixels=8, source_release="Q1_R1",
    )
