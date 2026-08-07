from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.photometry import ab_mag_to_uJy, electrons_to_ab_mag
from euclid_polish.sky.generation import tng_galaxy
from euclid_polish.sky.generation.phz_galaxy_prior import (
    PHZ_EMPIRICAL_KIND,
    PhzGalaxyPopulationPrior,
    build_phz_galaxy_population_payload,
    population_prior_from_payload,
)


def _tiny_cache(tmp_path):
    catalog = tmp_path / "euclid_population.csv"
    with catalog.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "object_id", "point_like_prob", "spurious_prob",
            "semimajor_axis", "ellipticity", "flux_detection_total_uJy",
        ))
        writer.writeheader()
        writer.writerows([
            {
                "object_id": "large-bright", "point_like_prob": 0.25,
                "spurious_prob": 0.01, "semimajor_axis": 20.0,
                "ellipticity": 0.0,
                "flux_detection_total_uJy": float(ab_mag_to_uJy(18.0)),
            },
            {
                "object_id": "small-faint", "point_like_prob": 0.75,
                "spurious_prob": 0.01, "semimajor_axis": 1.0,
                "ellipticity": 0.0,
                "flux_detection_total_uJy": float(ab_mag_to_uJy(26.0)),
            },
            {
                "object_id": "bad-spurious", "point_like_prob": 0.0,
                "spurious_prob": 0.9, "semimajor_axis": 3.0,
                "ellipticity": 0.0,
                "flux_detection_total_uJy": float(ab_mag_to_uJy(22.0)),
            },
        ])
    pdf = tmp_path / "euclid_population_phz_pdf.npz"
    np.savez_compressed(
        pdf,
        object_id=np.asarray([
            "large-bright", "small-faint", "bad-spurious",
        ]),
        probability=np.asarray([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ], dtype=np.float32),
        z_edges=np.asarray([0.1, 0.5, 1.0], dtype=np.float64),
    )
    meta = tmp_path / "euclid_population_meta.json"
    meta.write_text(json.dumps({
        "catalog_version": 6,
        "area_arcmin2": 2.0,
    }))
    return catalog, pdf, meta


def test_payload_preserves_weighted_joint_phz_kron_size_support(tmp_path):
    payload = build_phz_galaxy_population_payload(*_tiny_cache(tmp_path))

    assert payload["kind"] == PHZ_EMPIRICAL_KIND
    assert payload["validated"] is False
    assert payload["source"]["selected_rows"] == 2
    assert payload["source"]["selected_galaxy_weight"] == pytest.approx(1.0)
    assert payload["generation"]["surface_density_arcmin2"] == pytest.approx(0.5)
    assert min(payload["grid"]["kron_magnitude_edges"]) <= 18.0
    assert max(payload["grid"]["kron_magnitude_edges"]) > 26.0
    assert 10.0 ** max(payload["grid"]["log_radius_edges"]) > 2.0
    assert payload["selection"]["magnitude_clipping"] is None
    assert "no post-PSF Kron" in payload["measurement_model"]["rendering_anchor"]

    prior = population_prior_from_payload(payload)
    assert isinstance(prior, PhzGalaxyPopulationPrior)
    rng = np.random.default_rng(21)
    draws = [prior.sample(rng) for _ in range(2000)]

    # The bright/large/low-z and faint/small/high-z rows occupy separate joint
    # cells.  Binning may broaden each row, but must never create cross-pairs.
    for draw in draws:
        if draw.target_vis_mag < 22.0:
            assert draw.re_arcsec > 1.0
            assert draw.z < 0.5
        else:
            assert draw.re_arcsec < 0.2
            assert draw.z >= 0.5
        assert electrons_to_ab_mag(
            draw.target_vis_flux_e, Config.get_band("VIS"),
        ) == pytest.approx(draw.target_vis_mag)
        assert draw.target_vis_estimator == "MER detection-band Kron flux"


def test_payload_is_deterministic_and_compact(tmp_path):
    paths = _tiny_cache(tmp_path)
    first = build_phz_galaxy_population_payload(*paths)
    second = build_phz_galaxy_population_payload(*paths)

    assert first["fingerprint"] == second["fingerprint"]
    assert first["grid"]["density_zlib_base64"] == second["grid"][
        "density_zlib_base64"
    ]
    assert len(json.dumps(first)) < 20_000

    first["grid"]["density_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="density fingerprint"):
        PhzGalaxyPopulationPrior(first)


def test_size_is_resolved_before_single_vis_normalisation(monkeypatch):
    calls: list[float | None] = []

    monkeypatch.setattr(
        tng_galaxy,
        "_prepare_tng_continuous_source",
        lambda *_args, **_kwargs: (np.ones((2, 2, 4)), False),
    )

    def fake_render(*_args, **kwargs):
        calls.append(kwargs["target_vis_flux_e"])
        return np.ones((2, 2, 4), dtype=np.float32), {}, 0.2

    monkeypatch.setattr(tng_galaxy, "_render_target_re", fake_render)
    stamp, meta, achieved, _scale = tng_galaxy._match_target_re(
        "unused", "1", 1,
        initial_scale=1.0,
        target_re_arcsec=0.2,
        pixel_scale_arcsec=0.05,
        rot_k=0,
        rot_angle=None,
        target_vis_flux_e=20.0,
    )

    assert calls == [None]
    assert achieved == pytest.approx(0.2)
    assert stamp[..., 0].sum() == pytest.approx(20.0)
    assert meta["brightness_scale"] == pytest.approx(5.0)
    assert meta["photometric_scaling"].endswith("after_size_match")
