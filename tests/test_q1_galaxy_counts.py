"""Focused tests for Q1-wide bright-galaxy aperture counts."""

from __future__ import annotations

import json

import pytest

from euclid_polish.config import Config
from euclid_polish.web.helpers import q1_galaxy_counts


class _Job:
    def __init__(self, row):
        self.row = row

    def get_results(self):
        return [self.row]


def test_q1_galaxy_counts_use_four_apertures_and_phz_galaxy_selection(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    queries: list[str] = []
    rows = iter(
        {
            "selected_galaxies": count,
            "expected_galaxies": count / 2,
            "classification_variance": count / 4,
        }
        for count in range(1, 9)
    )

    def launch(query):
        queries.append(query)
        return _Job(next(rows))

    monkeypatch.setattr(q1_galaxy_counts.Euclid, "launch_job_async", launch)
    progress = []
    result = q1_galaxy_counts.query_q1_galaxy_aperture_counts(
        bright=14.0,
        faint=14.2,
        bin_width=0.1,
        progress=lambda done, total, label: progress.append(
            (done, total, label),
        ),
    )

    assert result["edges"] == pytest.approx([14.0, 14.1, 14.2])
    assert result["query_count"] == 8
    assert len(queries) == 8
    for query, (column, _label, _estimator) in zip(
        queries[:4], q1_galaxy_counts.Q1_VIS_APERTURES.values(), strict=True,
    ):
        assert query == q1_galaxy_counts._aperture_bin_query(
            column, 14.0, 14.1,
        )
    for query, (column, _label, _estimator) in zip(
        queries[4:], q1_galaxy_counts.Q1_VIS_APERTURES.values(), strict=True,
    ):
        assert query == q1_galaxy_counts._aperture_bin_query(
            column, 14.1, 14.2,
        )
    for multiple in range(1, 5):
        assert sum(
            f"mer.flux_vis_{multiple}fwhm_aper" in query
            for query in queries
        ) == 2
    assert all("mer.extended_flag" not in query for query in queries)
    assert all("mer.point_like_flag IS NULL" in query for query in queries)
    assert all("JOIN catalogue.phz_classification AS cls" in query for query in queries)
    assert all("SUM(cls.phz_gal_prob)" in query for query in queries)
    assert all("cls.phz_gal_prob >= 0.5" in query for query in queries)
    assert all("det_quality_flag" not in query for query in queries)
    assert all("flag_vis" not in query for query in queries)
    first = result["apertures"]["f1"]["bins"][0]
    assert first["selected_galaxies"] == 1
    assert first["expected_galaxies"] == pytest.approx(0.5)
    assert first["density_arcmin2_mag"] == pytest.approx(
        0.5 / (63.1 * 3600.0) / 0.1,
    )
    assert result["apertures"]["f4"]["selected_galaxies"] == 12
    assert result["completed_queries"] == result["total_queries"] == 8
    assert result["phases_completed"] == result["phase_count"] == 2
    assert progress[-1][:2] == (8, 8)
    cached = json.loads(
        q1_galaxy_counts.q1_galaxy_counts_path().read_text()
    )
    assert cached["selection"] == result["selection"]


def test_q1_galaxy_query_checkpoints_and_resumes_without_requery(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    first_queries = []

    def fail_after_first(query):
        first_queries.append(query)
        if len(first_queries) == 1:
            return _Job({
                "selected_galaxies": 2,
                "expected_galaxies": 1.5,
                "classification_variance": 0.25,
            })
        return None

    monkeypatch.setattr(q1_galaxy_counts.Euclid, "launch_job_async", fail_after_first)

    with pytest.raises(RuntimeError, match="rejected"):
        q1_galaxy_counts.query_q1_galaxy_aperture_counts(
            bright=14.0, faint=14.2,
        )

    partial = q1_galaxy_counts.read_q1_galaxy_aperture_counts()
    assert partial["completed_queries"] == 1
    assert partial["complete"] is False

    resumed_queries = []
    monkeypatch.setattr(
        q1_galaxy_counts.Euclid,
        "launch_job_async",
        lambda query: resumed_queries.append(query) or _Job({
            "selected_galaxies": 1,
            "expected_galaxies": 0.75,
            "classification_variance": 0.1,
        }),
    )
    complete = q1_galaxy_counts.query_q1_galaxy_aperture_counts(
        bright=14.0, faint=14.2,
    )

    assert len(resumed_queries) == 7
    assert complete["completed_queries"] == 8
    assert complete["complete"] is True
    assert complete["apertures"]["f1"]["bins"][0][
        "selected_galaxies"
    ] == 2

    monkeypatch.setattr(
        q1_galaxy_counts.Euclid,
        "launch_job_async",
        lambda _query: pytest.fail("completed bins must not be queried again"),
    )
    cached = q1_galaxy_counts.query_q1_galaxy_aperture_counts(
        bright=14.0, faint=14.2,
    )
    assert cached["completed_queries"] == 8

    extension_queries = []
    monkeypatch.setattr(
        q1_galaxy_counts.Euclid,
        "launch_job_async",
        lambda query: extension_queries.append(query) or _Job({
            "selected_galaxies": 3,
            "expected_galaxies": 2.5,
            "classification_variance": 0.2,
        }),
    )
    extended = q1_galaxy_counts.query_q1_galaxy_aperture_counts(
        bright=14.0, faint=14.3,
    )
    assert len(extension_queries) == 4
    assert extended["completed_queries"] == 12
    assert extended["apertures"]["f1"]["bins"][0][
        "selected_galaxies"
    ] == 2


def test_q1_aperture_fit_uses_cached_counts_and_tracks_fingerprint(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    area = 63.1 * 3600.0
    edges = [14.0 + 0.1 * index for index in range(52)]
    apertures = {}
    for aperture_index, (key, (_column, label, estimator)) in enumerate(
        q1_galaxy_counts.Q1_VIS_APERTURES.items(),
    ):
        bins = []
        for index, (lower, upper) in enumerate(
            zip(edges[:-1], edges[1:], strict=True),
        ):
            density = float(
                (aperture_index + 1) * 10 ** (0.4 * (0.5 * (lower + upper) - 14.0))
            )
            expected = density * area * 0.1
            bins.append({
                "bin_index": index,
                "phase": index,
                "mag_lo": lower,
                "mag_hi": upper,
                "selected_galaxies": int(round(expected)),
                "expected_galaxies": expected,
                "classification_variance": expected * 0.1,
                "density_arcmin2_mag": density,
            })
        apertures[key] = {
            "label": label,
            "estimator": estimator,
            "bins": bins,
            "selected_galaxies": sum(
                item["selected_galaxies"] for item in bins
            ),
            "expected_galaxies": sum(
                item["expected_galaxies"] for item in bins
            ),
        }
    payload = {
        "version": q1_galaxy_counts.Q1_GALAXY_COUNT_VERSION,
        "selection_version": q1_galaxy_counts.Q1_GALAXY_SELECTION_VERSION,
        "bright": 14.0,
        "faint": edges[-1],
        "bin_width": 0.1,
        "edges": edges,
        "footprint_area_arcmin2": area,
        "footprint_area_deg2": 63.1,
        "completed_queries": 4 * (len(edges) - 1),
        "apertures": apertures,
    }
    counts_path = q1_galaxy_counts.q1_galaxy_counts_path()
    counts_path.parent.mkdir(parents=True)
    counts_path.write_text(json.dumps(payload))

    fitted = q1_galaxy_counts.fit_q1_galaxy_aperture_counts()

    assert fitted["version"] == q1_galaxy_counts.Q1_GALAXY_FIT_VERSION
    assert set(fitted["apertures"]) == {"f2"}
    assert fitted["scope"].startswith("apparent-brightness")
    curve = fitted["apertures"]["f2"]
    assert curve["law"]["mag_bright"] == 14.0
    assert curve["law"]["mag_faint"] == 29.0
    assert curve["law"]["fit_faint"] - curve["law"]["fit_bright"] >= 4.0
    assert curve["law"]["r_squared"] >= 0.998
    assert q1_galaxy_counts.read_q1_galaxy_aperture_fit() == fitted

    payload["updated"] = True
    counts_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="stale"):
        q1_galaxy_counts.read_q1_galaxy_aperture_fit()
