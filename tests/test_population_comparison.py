"""Focused checks for the real-versus-synthetic field-statistics workspace."""
from __future__ import annotations

import json

import numpy as np

from euclid_polish.web.helpers.population_comparison import (
    _field_payload,
    _FieldAccumulator,
    _normalise_field,
    _parameter_payload,
    _read_synthetic_sources,
)


def test_population_field_payload_is_json_safe_and_keeps_four_bands():
    rng = np.random.default_rng(17)
    synthetic = _FieldAccumulator()
    real = _FieldAccumulator()
    for _ in range(3):
        base = rng.normal(100.0, 8.0, (256, 256, 4)).astype(np.float32)
        base[0, 0, :] = 0.0
        synthetic.add(base)
        real.add(base * 1.2 + rng.normal(0, 0.5, base.shape))

    payload = _field_payload(synthetic, real)

    json.dumps(payload)
    assert payload["bands"] == ["VIS", "Y_E", "J_E", "H_E"]
    assert set(payload["histograms"]) == set(payload["bands"])
    assert len(payload["power"]["VIS"]["k"]) == 24
    assert len(payload["mean_cross_correlation"]["VIS"]["r"]) == 24
    assert any(value is not None
               for value in payload["mean_cross_correlation"]["VIS"]["r"])
    vis_range = payload["histograms"]["VIS"]["range"]
    assert vis_range[0] == min(
        float(np.min(samples)) for samples in synthetic.samples[0] + real.samples[0]
    )
    assert vis_range[1] == max(
        float(np.max(samples)) for samples in synthetic.samples[0] + real.samples[0]
    )
    assert np.isclose(sum(payload["histograms"]["VIS"]["synthetic"]), 1.0)
    assert np.isclose(sum(payload["histograms"]["VIS"]["real"]), 1.0)
    assert payload["histograms"]["VIS"]["y_label"] == (
        "fraction of sampled pixels / bin"
    )
    zero_bin = payload["histograms"]["VIS"]["zero_bin"]
    assert zero_bin is not None
    centers = payload["histograms"]["VIS"]["x"]
    width = centers[1] - centers[0]
    assert centers[zero_bin] - width / 2 <= 0 <= centers[zero_bin] + width / 2


def test_population_field_normalisation_accepts_fits_plane_order():
    cube = np.arange(4 * 256 * 256, dtype=np.float32).reshape(4, 256, 256)
    normalized = _normalise_field(cube)
    assert normalized.shape == (255, 255, 4)
    np.testing.assert_array_equal(normalized[..., 2], cube[2, :255, :255])


def test_population_parameter_payload_plots_every_available_parameter():
    rows = [
        {"field_index": 0, "type": "galaxy", "mag_vis": 21.0, "z": 0.4},
        {"field_index": 0, "type": "star", "mag_vis": 18.0,
         "temperature_k": 5400.0},
        {"field_index": 1, "type": "galaxy", "mag_vis": 22.0, "z": 0.8},
    ]
    payload = _parameter_payload(rows, area_arcmin2=2.0,
                                 include_per_field=True)

    assert payload["counts"] == {"galaxy": 2, "star": 1}
    assert payload["density_arcmin2"]["galaxy"] == 1.0
    assert {"objects_per_field", "mag_vis", "z", "temperature_k"} <= set(
        payload["parameters"]
    )


def test_synthetic_lenses_are_merged_into_galaxies(tmp_path):
    sources = tmp_path / "sources.csv"
    sources.write_text(
        "field_index,type,mag_vis,theta_E_arcsec\n"
        "0,galaxy,21.0,\n"
        "0,lens,20.0,1.2\n"
        "0,star,18.0,\n"
    )
    rows = _read_synthetic_sources([sources])
    payload = _parameter_payload(rows, area_arcmin2=1.0,
                                 include_per_field=True)

    assert payload["counts"] == {"galaxy": 2, "star": 1}
    assert "theta_E_arcsec" not in payload["parameters"]


def test_ensure_ssh_connected_builds_shared_session(monkeypatch):
    from types import SimpleNamespace

    from euclid_polish.web import remote

    created = []

    class FakeSession:
        def __init__(self, cfg):
            self.cfg = cfg
            self.connected = False
            created.append(self)

        def is_connected(self):
            return self.connected

        def connect(self):
            self.connected = True

    cfg = SimpleNamespace(
        ssh_user="astro",
        ssh_host="cluster.example",
        control_socket="/tmp/test-population-ssh.sock",
        control_persist="8h",
    )
    monkeypatch.setattr(remote.STATE, "ssh", None)
    monkeypatch.setattr(remote.STATE, "connected_at", None)
    monkeypatch.setattr(remote, "SSHSession", FakeSession)
    monkeypatch.setattr("euclid_polish.web.fasrc_config.load", lambda: cfg)

    session = remote.ensure_ssh_connected()

    assert session is created[0]
    assert session.connected
    assert remote.STATE.ssh is session
    assert remote.STATE.connected_at is not None


def test_population_comparison_page_and_status_route(monkeypatch):
    from euclid_polish.web.app import create_app
    from euclid_polish.web.routes import population_comparison as routes

    expected_availability = {
        "synthetic": {"fields": 200},
        "real": {"fields": 302},
        "field_area_arcmin2": 0.18,
        "default_cone": {"ra": 1.0, "dec": 2.0, "radius_arcmin": 3.0,
                         "area_arcmin2": 28.27},
        "euclid_catalog": {"cached": False, "meta": None},
    }
    monkeypatch.setattr(routes, "availability",
                        lambda: expected_availability)
    monkeypatch.setattr(routes, "read_comparison", lambda: None)
    monkeypatch.setattr(routes.euclid_session, "is_authenticated",
                        lambda: False)
    client = create_app().test_client()

    page = client.get("/population-comparison")
    assert page.status_code == 200
    assert b'<div id="root">' in page.data

    status = client.get("/api/population-comparison")
    assert status.status_code == 200
    assert status.get_json() == {
        "comparison": None,
        "availability": expected_availability,
        "authenticated": False,
    }
