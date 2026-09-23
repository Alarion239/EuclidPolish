"""Knee-independent PSNR: the metric, its integral, the payload and the route."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.eval.knee_psnr import (
    KNEE_GRID_E,
    integrated_psnr,
    knee_psnr,
    stretched_truth,
)
from euclid_polish.web.app import create_app
from euclid_polish.web.helpers import ensemble_viz


def test_knee_grid_spans_0p1_to_1e4():
    assert KNEE_GRID_E[0] == 0.1 and KNEE_GRID_E[-1] == 10000.0
    assert list(KNEE_GRID_E) == sorted(KNEE_GRID_E)


def test_knee_psnr_matches_the_closed_form_per_band():
    truth = np.zeros((4, 4, 2), np.float32)
    pred = np.full((4, 4, 2), 50.0, np.float32)
    pred[..., 1] = 5.0
    knees = (1.0, 100.0)
    curve = knee_psnr(pred, truth, knees=knees, peak_e=1e5)
    assert curve.shape == (2, 2)
    for k, q in enumerate(knees):
        for b, value in enumerate((50.0, 5.0)):
            expected = 10 * np.log10(np.arcsinh(1e5 / q) ** 2 / np.arcsinh(value / q) ** 2)
            assert curve[k, b] == pytest.approx(expected, rel=1e-5)
    reused = knee_psnr(pred, truth, knees=knees, peak_e=1e5,
                       truth_asinh=stretched_truth(truth, knees))
    np.testing.assert_allclose(reused, curve)


def test_integrated_psnr_is_the_mean_over_log_knee():
    knees = (1.0, 10.0, 100.0)
    assert integrated_psnr(np.array([30.0, 30.0, 30.0]), knees) == pytest.approx(30.0)
    # Linear in log10(knee) -> the mean equals the midpoint value.
    assert integrated_psnr(np.array([10.0, 20.0, 30.0]), knees) == pytest.approx(20.0)
    per_band = integrated_psnr(np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]), knees)
    np.testing.assert_allclose(per_band, [1.0, 2.0])
    with pytest.raises(ValueError):
        integrated_psnr(np.zeros(4), knees)


def _fake_cubes(tmp_path, monkeypatch):
    """Two test fields, two members, the mean and a baked spatial gate."""
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path))
    labels = ["1·psnr", "2·psnr"]
    rng = np.random.default_rng(0)
    cubes = ensemble_viz._ensemble_cubes_dir(starless=False)
    os.makedirs(cubes, exist_ok=True)
    truths = {}
    for rec in (0, 1):
        truth = rng.exponential(200.0, (16, 16, 4)).astype(np.float32)
        truths[rec] = truth
        members = [truth + rng.normal(0, s, truth.shape).astype(np.float32) for s in (5.0, 50.0)]
        for i, member in enumerate(members):
            np.save(os.path.join(cubes, f"member{i}_{rec:05d}.npy"), member)
        np.save(os.path.join(cubes, f"sr_{rec:05d}.npy"), np.mean(members, axis=0))
        np.save(os.path.join(cubes, f"comb_spatial_gate_{rec:05d}.npy"), members[0])
    with open(os.path.join(cubes, "viz_index.json"), "w") as handle:
        json.dump({"subset": "test", "indices": [0, 1], "member_labels": labels,
                   "records_fp": "fp", "target_psf_fwhm_arcsec": 0.066,
                   "has_combiner_spatial_gate": True}, handle)
    records = tmp_path / "records"
    records.mkdir()
    (records / "hr_test.tfrecord").write_bytes(b"")
    images = [SimpleNamespace(index=rec, data=truths[rec], pixel_scale_arcsec=0.05)
              for rec in (0, 1)]
    monkeypatch.setattr(ensemble_viz, "_sky_records_local_dir", lambda: str(records))
    monkeypatch.setattr(ensemble_viz, "tfrecord_path",
                        lambda d, name: os.path.join(d, f"{name}.tfrecord"))
    monkeypatch.setattr(ensemble_viz, "_eval_records_fingerprint", lambda *a, **k: "fp")
    monkeypatch.setattr(ensemble_viz, "_regime_labels", lambda *a: labels)
    monkeypatch.setattr(ensemble_viz, "_member_meta_from_labels",
                        lambda ls: [{"loss": "l2", "asinh_knee": 10, "blocks": 32} for _ in ls])
    monkeypatch.setattr(ensemble_viz, "ImageSet",
                        SimpleNamespace(read=lambda *a, **k: iter(images)))
    return labels


def test_payload_scores_members_mean_and_combiners(tmp_path, monkeypatch):
    labels = _fake_cubes(tmp_path, monkeypatch)
    assert ensemble_viz.knee_psnr_status(False)["available"] is False
    payload = ensemble_viz.compute_knee_psnr_payload(False)
    assert payload["n_fields"] == 2 and payload["knees"] == list(KNEE_GRID_E)
    ids = [m["id"] for m in payload["models"]]
    assert ids == ["member_0", "member_1", "ensemble_mean", "spatial_gate"]
    by_id = {m["id"]: m for m in payload["models"]}
    assert by_id["member_0"]["label"] == labels[0] and by_id["member_0"]["asinh_knee"] == 10
    assert np.array(by_id["member_0"]["psnr"]).shape == (len(KNEE_GRID_E), 4)
    # The low-noise member beats the noisy one at every knee and band.
    assert np.all(np.array(by_id["member_0"]["integrated"])
                  > np.array(by_id["member_1"]["integrated"]))
    status = ensemble_viz.knee_psnr_status(False)
    assert status["available"] and status["stale"] is False
    # Reused when nothing changed; stale once a combiner is refitted.
    assert ensemble_viz.compute_knee_psnr_payload(False) == payload
    monkeypatch.setattr(ensemble_viz, "_combiner_fingerprint", lambda *a, **k: "refit")
    assert ensemble_viz.knee_psnr_status(False)["stale"] is True


def test_knee_psnr_route_reports_missing_curves(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path))
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        payload = client.get("/ensemble/knee-psnr.json?mode=starfull").get_json()
    assert payload["available"] is False
