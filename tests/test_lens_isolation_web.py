from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ["EUCLID_POLISH_DISABLE_AUTO_SSH"] = "1"

from euclid_polish.web.app import create_app


def test_react_page_and_status_are_registered(tmp_path, monkeypatch):
    from euclid_polish.web.helpers import lens_isolation_viz

    monkeypatch.setattr(lens_isolation_viz, "output_dir", lambda: str(tmp_path))
    app = create_app()
    client = app.test_client()
    page = client.get("/lens-isolation")
    assert page.status_code == 200
    assert b'id="root"' in page.data
    status = client.get("/api/lens-isolation/status")
    assert status.status_code == 200
    assert "experiments/lens_isolation" in status.get_json()["root"]


def test_sync_defaults_to_evaluation_only_when_offline(monkeypatch):
    from euclid_polish.web.remote import STATE

    monkeypatch.setattr(STATE, "ssh", None)
    app = create_app()
    response = app.test_client().post("/api/lens-isolation/sync")
    assert response.status_code == 400
    assert response.get_json()["error"] == "not connected"


def test_sync_can_pull_only_corresponding_record_splits(tmp_path, monkeypatch):
    from euclid_polish.web.remote import STATE
    from euclid_polish.web.routes import lens_isolation as routes

    calls = []

    class SSH:
        def is_connected(self):
            return True

        def rsync_pull(self, remote, local, timeout):
            calls.append((remote, local, timeout))
            return 0, "", ""

    monkeypatch.setattr(STATE, "ssh", SSH())
    monkeypatch.setattr(routes.fasrc_config, "load", lambda: SimpleNamespace(data_dir="/remote/data"))
    monkeypatch.setattr(routes, "_status", lambda: {"ok": True})
    monkeypatch.setattr(
        routes,
        "ExperimentPaths",
        lambda: SimpleNamespace(
            root=str(tmp_path),
            records=str(tmp_path / "records"),
            ensemble=str(tmp_path / "ensemble"),
            evaluation=str(tmp_path / "evaluation"),
        ),
    )
    app = create_app()
    response = app.test_client().post(
        "/api/lens-isolation/sync",
        data={"subsets": "validate,test", "evaluation": "0", "ensemble": "0"},
    )

    assert response.status_code == 200
    remotes = [remote for remote, _local, _timeout in calls]
    assert remotes == [
        "/remote/data/experiments/lens_isolation/records/dataset.json",
        "/remote/data/experiments/lens_isolation/records/dirty_validate.tfrecord",
        "/remote/data/experiments/lens_isolation/records/lens_validate.tfrecord",
        "/remote/data/experiments/lens_isolation/records/sources_validate.csv",
        "/remote/data/experiments/lens_isolation/records/split_validate.json",
        "/remote/data/experiments/lens_isolation/records/dirty_test.tfrecord",
        "/remote/data/experiments/lens_isolation/records/lens_test.tfrecord",
        "/remote/data/experiments/lens_isolation/records/sources_test.csv",
        "/remote/data/experiments/lens_isolation/records/split_test.json",
    ]
    assert all(local == str(tmp_path / "records") for _remote, local, _timeout in calls)


def test_sync_rejects_unknown_record_split(monkeypatch):
    from euclid_polish.web.remote import STATE

    monkeypatch.setattr(
        STATE,
        "ssh",
        SimpleNamespace(is_connected=lambda: True),
    )
    app = create_app()
    response = app.test_client().post(
        "/api/lens-isolation/sync", data={"subsets": "validate,production"}
    )
    assert response.status_code == 400
    assert "production" in response.get_json()["error"]


def test_lens_ensemble_routes_bound_and_validate_parameters(monkeypatch):
    from euclid_polish.web.routes import lens_isolation as routes

    jobs = []
    monkeypatch.setattr(
        routes.REGISTRY,
        "spawn",
        lambda label, target: jobs.append((label, target)) or "job-1",
    )
    monkeypatch.setattr(
        routes,
        "job_evaluate",
        lambda _cap, **kwargs: jobs.append(("evaluate args", kwargs)),
    )
    monkeypatch.setattr(
        routes,
        "job_combiner_fit",
        lambda _cap, **kwargs: jobs.append(("combiner args", kwargs)),
    )
    app = create_app()
    client = app.test_client()

    evaluate = client.post(
        "/api/lens-isolation/ensemble/evaluate", data={"num_images": "99999", "force": "1"}
    )
    assert evaluate.get_json() == {"job_id": "job-1"}
    jobs[-1][1](object())
    assert jobs[-1] == ("evaluate args", {"num_images": 2000, "force": True})

    fit = client.post(
        "/api/lens-isolation/ensemble/combiner/fit",
        data={"num_images": "0", "n_kernels": "256", "min_usage": "0.9"},
    )
    assert fit.get_json() == {"job_id": "job-1"}
    jobs[-1][1](object())
    assert jobs[-1] == (
        "combiner args",
        {"num_images": 1, "n_kernels": 256, "min_usage": 0.5},
    )


def test_lens_isolation_viewer_has_independent_target_and_cubes(tmp_path, monkeypatch):
    from euclid_polish.web.helpers import lens_isolation_viz as lv
    from euclid_polish.web.helpers import viewer_data as vd

    cubes = tmp_path / "cubes"
    records = tmp_path / "records"
    cubes.mkdir()
    records.mkdir()
    (records / "lens_test.tfrecord").write_bytes(b"")
    np.save(cubes / "sr_00003.npy", np.ones((4, 4, 4), np.float32))
    manifest = {
        "subset": "test",
        "indices": [3],
        "member_labels": ["00·psnr"],
        "pca_n": 0,
        "has_combiner": False,
    }
    monkeypatch.setattr(vd, "_lens_isolation_manifest", lambda: manifest)
    monkeypatch.setattr(vd, "_load_lens_isolation_combiner", lambda _labels: None)
    monkeypatch.setattr(lv, "cubes_dir", lambda _subset=None: str(cubes))
    monkeypatch.setattr(lv, "records_dir", lambda: str(records))

    meta = vd.get_meta("lens-isolation", {})
    assert meta["collection"] == "lens-isolation"
    assert meta["target_label"] == "lens target"
    assert "hr" in [tier["key"] for tier in meta["tiers"]]
    assert "bhr" in [tier["key"] for tier in meta["tiers"]]
    cube, info = vd.get_cube("lens-isolation", 0, "sr", {})
    assert cube.shape == (4, 4, 4)
    assert "ensemble mean" in info["label"]


def test_lens_isolation_bhr_blurs_raw_lens_target(monkeypatch):
    from euclid_polish.web.helpers import viewer_data as vd

    impulse = np.zeros((9, 9, 4), np.float32)
    impulse[4, 4, 0] = 1.0
    monkeypatch.setattr(vd, "_lens_isolation_manifest", lambda: {
        "subset": "test", "indices": [3], "member_labels": [],
        "target_psf_fwhm_arcsec": 0.05,
    })
    monkeypatch.setattr(
        vd, "_lens_isolation_record_cube",
        lambda *_args: (impulse.copy(), 0.05),
    )

    raw, _ = vd._lens_isolation_cube(0, "hr", {})
    blurred, info = vd._lens_isolation_cube(0, "bhr", {})

    assert raw[4, 4, 0] == 1.0
    assert blurred[4, 4, 0] < raw[4, 4, 0]
    assert "BHR (blurred lens target)" in info["label"]


def test_react_generation_card_uses_only_fasrc_cpu_resources():
    page = Path("euclid_polish/web/frontend/src/pages/LensIsolation.tsx").read_text(encoding="utf-8")

    assert 'label="workers"' not in page
    assert "setWorkers" not in page
    assert "Replace all experiment records and recompute from scratch" in page
    assert 'force: forceGeneration ? "1" : "0"' in page
    assert "Retrain and replace the existing lens-isolation members" in page
    assert 'force: forceTraining ? "1" : "0"' in page
    assert 'extraParams={generation} embedded showHistory' in page
    assert 'extraParams={training} embedded showHistory' in page
    assert 'stepId="lens_isolation_evaluate" embedded showHistory' in page


def test_classic_generation_card_uses_only_fasrc_cpu_resources():
    page = Path("euclid_polish/web/static/fasrc_step_card.js").read_text(encoding="utf-8")
    card = page.split("case 'lens_isolation_generate':", 1)[1].split(
        "case 'lens_isolation_train':", 1
    )[0]

    assert 'name="workers"' not in card


def test_classic_training_card_exposes_only_real_training_controls():
    page = Path("euclid_polish/web/static/fasrc_step_card.js").read_text(encoding="utf-8")
    card = page.split("case 'lens_isolation_train':", 1)[1].split(
        "case 'lens_isolation_evaluate':", 1
    )[0]

    assert 'name="force"' in card
    assert 'name="lens_weight"' not in card
    assert 'name="flux_weight"' not in card
