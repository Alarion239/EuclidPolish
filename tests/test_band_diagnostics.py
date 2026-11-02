"""Models › Diagnostics beyond VIS: the Y/J/H evaluation payloads computed
from the cached cubes (helpers/ensemble_viz.compute_band_evaluation_payloads;
their routes went with the panel clean-up, the helpers stay for the scripts)
and the angular power spectrum served as curves
(``/api/evaluation/angular-power-spectrum.json``)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.eval import power_spectrum
from euclid_polish.web.app import create_app
from euclid_polish.web.helpers import ensemble_viz as ev
from euclid_polish.web.routes import evaluation as evaluation_routes

LABELS = ["196·psnr", "178·psnr", "170·psnr"]


def _field(rng, band: int, n: int = 64):
    """One field's planes of ``band``: a blob scene, three noisy members."""
    y, x = np.mgrid[:n, :n]
    hr = 50.0 * (band + 1) * np.exp(-((x - n / 2) ** 2 + (y - n / 2) ** 2) / 40.0) + rng.normal(0, 1, (n, n))
    members = np.stack([hr + rng.normal(0, 2, (n, n)) for _ in LABELS], 0)
    return hr, members.mean(0), members, {"spatial_gate": members.mean(0)}, hr + rng.normal(0, 3, (n, n))


@pytest.fixture
def cubes(tmp_path, monkeypatch):
    """Three cached test fields (4 bands) and a membership identity."""
    identity = {"records_fp": "r1", "subset": "test", "indices": [0, 1, 2], "member_labels": LABELS}
    rng = np.random.default_rng(0)
    fields = [(rec, {b: _field(rng, b) for b in (1, 2, 3)}) for rec in range(3)]

    def iterate(starless, bands):
        for rec, planes in fields:
            yield rec, {b: planes[b] for b in bands if b in planes}

    monkeypatch.setattr(ev, "_ensemble_regime_dir", lambda starless: str(tmp_path))
    monkeypatch.setattr(ev, "_read_test_manifest", lambda starless: {"subset": "test", "indices": [0, 1, 2],
                                                                    "member_labels": LABELS})
    monkeypatch.setattr(ev, "_knee_psnr_identity", lambda starless, manifest: {
        "schema": 2, "knees": [1], **identity})
    monkeypatch.setattr(ev, "_iter_cached_field_bands", iterate)
    monkeypatch.setattr(ev, "_member_meta_from_labels", lambda labels: [{"loss": "l2"} for _ in labels])
    return tmp_path, identity


def test_the_nisp_bands_are_measured_in_one_sweep_and_written_per_band(cubes):
    tmp_path, identity = cubes
    ticks: list[tuple[int, int, str]] = []
    out = ev.compute_band_evaluation_payloads(False, progress=lambda *a: ticks.append(a))
    assert sorted(out) == ["H_E", "J_E", "Y_E"]
    for band in ("Y_E", "J_E", "H_E"):
        payload = json.loads((tmp_path / f"ensemble_evals_{band}.json").read_text())
        assert payload["band"] == band and payload["n_fields"] == 3 and payload["n_members"] == 3
        assert payload["guides"]["band"] == band
        assert payload["guides"]["psf_fwhm"] == pytest.approx(Config.get_band(band).psf_fwhm_arcsec)
        assert payload["ps"]["r"] and payload["ps"]["r_pairs"] == []      # no member-pair spectra
        assert payload["calibration"]["stats"]["cover1"] is not None
        assert payload["identity"]["member_labels"] == LABELS
        assert (tmp_path / f"ensemble_diag_samples_{band}.json").is_file()
    assert [t[0] for t in ticks] == [1, 2, 3] and ticks[-1][1] == 3
    # the bands differ: each is its own measurement
    assert out["Y_E"]["bright_std"]["hist"] != out["H_E"]["bright_std"]["hist"]
    assert ev.band_evals_state(False) == {"Y_E": "current", "J_E": "current", "H_E": "current"}
    assert ev.read_band_evals(False, "J_E")["stale"] is False


def test_a_changed_evaluation_marks_the_bands_stale_and_evaluate_refreshes_them(cubes, monkeypatch):
    _tmp, identity = cubes
    assert ev.band_evals_state(False) == {"Y_E": "missing", "J_E": "missing", "H_E": "missing"}
    assert ev.read_band_evals(False, "Y_E") is None
    ev._refresh_band_evals(False, None)                    # Evaluate's hook: missing → computed
    assert set(ev.band_evals_state(False).values()) == {"current"}
    calls: list[str] = []
    monkeypatch.setattr(ev, "compute_band_evaluation_payloads", lambda starless, progress=None: calls.append("run"))
    ev._refresh_band_evals(False, None)                    # all current → skipped
    assert calls == []
    identity["member_labels"] = LABELS[:2]                 # a member archived
    assert ev.band_evals_state(False)["H_E"] == "stale"
    assert ev.read_band_evals(False, "H_E")["stale"] is True
    ev._refresh_band_evals(False, None)
    assert calls == ["run"]
    with pytest.raises(ValueError, match="unknown band"):
        ev.read_band_evals(False, "VIS")


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_the_bands_sidecar_path():
    # the /ensemble/evals.json?band= and /ensemble/pixel-trace.json routes went
    # with the panel clean-up; the helpers (scripts) keep the per-band sidecars
    assert ev.diag_samples_path(False, "J_E").endswith("ensemble_diag_samples_J_E.json")


def test_the_angular_power_spectrum_is_served_as_curves_cache_only(client, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
    os.makedirs(Config.EVAL_RESULTS_DIR)
    # no synced records: the cache's target record cannot have changed since
    monkeypatch.setattr(evaluation_routes, "_sky_records_local_dir", lambda: str(tmp_path / "records"))
    missing = client.get("/api/evaluation/angular-power-spectrum.json")
    assert missing.status_code == 404 and "not measured yet" in missing.get_json()["error"]
    data = {"subset": "validate", "n_fields": 2, "field_n": 64, "pixel_scale": 0.05, "lr_scale": 0.1,
            "theta_max": 4.0, "band_names": ["VIS"], "bands": {"VIS": {"psf_fwhm": 0.16, "linear": {}, "asinh": {}}},
            "target": {"kind": "hr", "records": "hr_validate.tfrecord", "fingerprint": "f"}}
    monkeypatch.setattr(power_spectrum, "power_spectrum_summary_data", lambda subset=None: data)
    monkeypatch.setattr(power_spectrum, "render_power_spectrum_figure", lambda out_png, d: out_png)
    assert client.post("/api/evaluation/angular-power-spectrum").get_json() == {"ok": True, "rendered": True}
    served = client.get("/api/evaluation/angular-power-spectrum.json")
    assert served.status_code == 200 and served.get_json() == data


def test_the_spectrum_curves_are_json_ready():
    assert power_spectrum._finite_list([1.0, np.nan, np.inf, 2]) == [1.0, None, None, 2.0]
