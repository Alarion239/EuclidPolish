"""Synthetic-workspace realism backend (the former Realism workspace): the
readiness overview (Synthetic › Status), the synthetic_generate
gate parity, read-only GET handlers, the noise-position endpoint and the one
training-catalogue sync job."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from euclid_polish.config import Config
from euclid_polish.web import fasrc_pipeline
from euclid_polish.web.app import create_app
from euclid_polish.web.helpers import galaxy_distributions, star_population
from euclid_polish.web.helpers import realism_overview as overview
from euclid_polish.web.helpers.noise_levels import noise_position
from euclid_polish.web.routes import population_comparison as pc_routes
from euclid_polish.web.routes import realism as realism_routes

ACTIVE_GALAXY = {"fingerprint": "g" * 64, "valid": True, "version": 15,
                 "generation": {"surface_density_arcmin2": 372.8}}
ACTIVE_STAR = {"fingerprint": "s" * 64, "valid": True, "version": 6,
               "population": {"density_arcmin2": 0.41}}


def _availability(**patch):
    base = {
        "synthetic": {"fields": 200, "train_source_catalog": False, "population_fields": 200,
                      "population_fields_with_training": 200},
        "real": {"fields": 176, "independent_parents": 44, "ready": True, "current": True,
                 "unavailable_reason": None, "collection_fingerprint": "c" * 64},
        "comparison_cache": {"present": True, "schema_current": False, "fresh": False,
                             "reason": "comparison cache uses an older schema"},
    }
    base.update(patch)
    return base


@pytest.fixture
def states(monkeypatch):
    """Monkeypatchable galaxy / star states and cheap stand-ins for the
    remaining probes (no TFRecords are read)."""
    box = {"galaxy": {"candidate": None, "active": None, "is_active": False},
           "star": {"candidate": None, "active": None, "is_active": False},
           "availability": _availability()}
    monkeypatch.setattr(overview, "joint_galaxy_state", lambda: box["galaxy"])
    monkeypatch.setattr(overview, "star_state", lambda: box["star"])
    monkeypatch.setattr(overview.population_comparison, "availability", lambda: box["availability"])
    monkeypatch.setattr(realism_routes, "check_records_noise", lambda: {
        "state": "unknown", "title": "Noise model of the local records unverified",
        "detail": "not in the local store", "to": "/data/records", "facts": {"local_only": True}})
    monkeypatch.setattr(overview.galaxy_distributions, "artifact_state", lambda: {
        "present": True, "stale": False, "reason": None, "version": 25, "built_at": None})
    monkeypatch.setattr(overview, "read_q1_galaxy_aperture_counts", lambda: {
        "completed_queries": 560, "total_queries": 560})
    monkeypatch.setattr(overview, "read_q1_galaxy_radius_statistics", lambda: {
        "completed_queries": 170, "total_queries": 170})
    monkeypatch.setattr(overview.euclid_session, "is_authenticated", lambda: False)
    monkeypatch.setattr(overview, "psf_inventory_payload", lambda: {"bands": [
        {"name": "VIS", "state": "empirical", "fwhm": 0.16, "measured_fwhm": 0.17},
        {"name": "Y_E", "state": "not_cached", "fwhm": 0.4},
    ]})
    monkeypatch.setattr(overview, "local_records_time", lambda records_dir=None: None)
    return box


def _items(payload):
    return {item["id"]: item for item in payload["items"]}


def test_overview_route_lists_every_prior_and_the_gate(states):
    response = create_app().test_client().get("/api/realism/overview")
    assert response.status_code == 200
    payload = response.get_json()
    items = _items(payload)
    assert list(items) == [
        "galaxy-model", "star-prior", "noise-model", "psf", "tng-radii", "saturation",
        "training-catalog", "galaxy-plots", "comparison-cache",
    ]
    assert [i["id"] for i in payload["items"] if i["group"] == overview.DIAGNOSTIC] == [
        "galaxy-plots", "comparison-cache"]
    for item in payload["items"]:
        assert item["state"] in overview.STATES
        assert set(item) >= {"id", "label", "state", "title", "detail", "to", "action", "facts",
                             "group", "records"}
    # The records' noise model is the Noise row's tick, not a row of its own.
    assert items["noise-model"]["records"]["state"] == "unknown"
    assert payload["records"] == {"generated_at": None, "splits": ["test", "validate"]}
    assert payload["gate"]["ready"] is False
    assert [b["id"] for b in payload["gate"]["blockers"]] == ["galaxy-model", "star-prior"]
    assert payload["counts"]["bad"] == 2          # both priors unfitted
    assert items["comparison-cache"]["state"] == "warn"
    assert items["comparison-cache"]["action"]["url"] == "/api/population-comparison/build"
    assert items["training-catalog"]["state"] == "unknown"
    assert payload["training"]["available"] is False
    assert payload["training"]["sync"]["params"] == {"rebuild": "1"}
    # The training sync is a self-connecting local job (never gated).
    assert payload["training"]["sync"]["requires_fasrc"] is False
    assert payload["training"]["sync"]["self_connects"] is True


def test_fasrc_actions_say_whether_they_are_gated_or_self_connect():
    """The UI disables only gated fixes offline; self-connecting jobs stay
    enabled (they open the connection themselves and report failure)."""
    stale = _availability(real={"ready": True, "current": False, "fields": 176,
                                "independent_parents": 44, "unavailable_reason": "changed"})
    archive = overview.comparison_item(stale)["action"]
    assert archive["url"] == "/api/archive-fields/sync"
    assert (archive["requires_fasrc"], archive["self_connects"]) == (False, True)
    training = overview.training_sync_action()
    assert (training["requires_fasrc"], training["self_connects"]) == (False, True)


def test_galaxy_item_walks_unfitted_candidate_active(states):
    item = overview.galaxy_item({"candidate": None, "active": None, "is_active": False})
    assert (item["state"], item["action"]) == ("bad", None)
    candidate = {**ACTIVE_GALAXY, "fingerprint": "n" * 64}
    item = overview.galaxy_item({"candidate": candidate, "active": None, "is_active": False})
    assert item["state"] == "warn"
    assert item["action"]["url"] == "/api/galaxy-distributions/activate"
    assert item["action"]["confirm"]
    item = overview.galaxy_item({"candidate": candidate, "active": ACTIVE_GALAXY, "is_active": False})
    assert item["title"] == "A newer galaxy candidate is not active"
    item = overview.galaxy_item({"candidate": {**candidate, "valid": False}, "active": None,
                                 "is_active": False})
    assert item["state"] == "bad"
    item = overview.galaxy_item({"candidate": ACTIVE_GALAXY, "active": ACTIVE_GALAXY, "is_active": True})
    assert item["state"] == "ok"
    assert item["facts"]["surface_density_arcmin2"] == pytest.approx(372.8)
    assert item["facts"]["aperture_checkpoints"] == [560, 560]


def test_star_item_surfaces_the_refit_warning(states):
    stale = {"fingerprint": "x" * 64, "valid": False, "warnings": ["refit required: stellar counts"]}
    item = overview.star_item({"candidate": stale, "active": None, "is_active": False})
    assert item["state"] == "bad"
    assert item["detail"] == "refit required: stellar counts"
    item = overview.star_item({"candidate": ACTIVE_STAR, "active": None, "is_active": False})
    assert item["action"]["url"] == "/api/star-distribution/activate"
    item = overview.star_item({"candidate": ACTIVE_STAR, "active": ACTIVE_STAR, "is_active": True})
    assert item["state"] == "ok"
    # The density is the row's verdict and the fingerprints the inspector's: no detail.
    assert item["detail"] is None
    assert item["facts"]["density_arcmin2"] == pytest.approx(0.41)
    assert item["facts"]["active_fingerprint"] == "s" * 64


def test_records_tick_compares_the_records_with_the_prior():
    assert overview.records_tick(100.0, 200.0, "stellar prior")["state"] == "current"
    late = overview.records_tick(300.0, 200.0, "stellar prior")
    assert late["state"] == "predates"
    assert "predate the stellar prior" in late["detail"]
    assert late["records_at"].startswith("1970-01-01T00:03:20")
    assert overview.records_tick(100.0, None, "stellar prior")["state"] == "unknown"
    assert overview.records_tick(None, 200.0, "stellar prior")["state"] == "unknown"
    assert overview.noise_records_tick({"state": "bad", "title": "old"})["state"] == "predates"
    assert overview.noise_records_tick({"state": "ok"})["state"] == "current"
    assert overview.noise_records_tick({"state": "unknown"})["state"] == "unknown"


def test_local_records_time_is_the_oldest_local_dirty_shard(tmp_path):
    assert overview.local_records_time(str(tmp_path)) is None
    for split, stamp in (("test", 2_000.0), ("validate", 1_000.0), ("train", 10.0)):
        path = tmp_path / f"dirty_{split}.tfrecord"
        path.write_bytes(b"x")
        os.utime(path, (stamp, stamp))
    # train is FASRC-only (never local as a rule) and is not a generation tick
    assert overview.local_records_time(str(tmp_path)) == pytest.approx(1_000.0)


def test_psf_item_names_the_fallback_and_the_uncached_bands():
    band = lambda name, state: {"name": name, "state": state, "fwhm": 0.4}  # noqa: E731
    ok = overview.psf_item(lambda: {"bands": [band("VIS", "empirical"), band("Y_E", "empirical")]})
    assert (ok["state"], ok["title"], ok["action"]) == ("ok", "Empirical ePSF in every band", None)
    fallback = overview.psf_item(lambda: {"bands": [band("VIS", "empirical"), band("J_E", "no_empirical"),
                                                    band("H_E", "no_empirical")]})
    assert (fallback["state"], fallback["title"]) == ("warn", "Gaussian fallback in J, H")
    assert fallback["action"]["requires_fasrc"] is True
    uncached = overview.psf_item(lambda: {"bands": [band("VIS", "not_cached")]})
    assert uncached["state"] == "unknown"
    assert uncached["facts"]["not_cached"] == ["VIS"]
    broken = overview.psf_item(lambda: (_ for _ in ()).throw(OSError("disk")))
    assert (broken["state"], broken["detail"]) == ("unknown", "disk")


def test_saturation_item_states_the_blackout_ramp():
    class Cfg:
        saturation_mask_prob = 0.25
    item = overview.saturation_item(lambda: Cfg())
    assert item["state"] == "ok"
    assert item["facts"]["base_probability"] == pytest.approx(0.25)
    assert item["facts"]["bright_probability"] == pytest.approx(Config.SATURATION_MASK_PROB_BRIGHT)
    assert item["facts"]["ramp_well_ratios"] == list(Config.SATURATION_MASK_RAMP_WELL_RATIOS)
    assert set(item["facts"]["wells_e"]) == {"VIS", "Y", "J", "H"}


@pytest.mark.parametrize(("galaxy", "star"), [
    ({"candidate": None, "active": None, "is_active": False},
     {"candidate": None, "active": None, "is_active": False}),
    ({"candidate": ACTIVE_GALAXY, "active": ACTIVE_GALAXY, "is_active": True},
     {"candidate": None, "active": None, "is_active": False}),
    ({"candidate": ACTIVE_GALAXY, "active": {"fingerprint": "g" * 64}, "is_active": True},
     {"candidate": ACTIVE_STAR, "active": ACTIVE_STAR, "is_active": True}),
    ({"candidate": ACTIVE_GALAXY, "active": ACTIVE_GALAXY, "is_active": True},
     {"candidate": ACTIVE_STAR, "active": ACTIVE_STAR, "is_active": True}),
])
def test_gate_matches_synthetic_generate_prepare_params(monkeypatch, galaxy, star):
    """The overview gate says ready exactly when the real step would accept a
    submission, and its message is the step's own refusal."""
    monkeypatch.setattr(fasrc_pipeline.population_calibration, "joint_galaxy_state", lambda: galaxy)
    monkeypatch.setattr(fasrc_pipeline.population_calibration, "star_state", lambda: star)
    gate = overview.generation_gate(galaxy, star)
    step = fasrc_pipeline.SyntheticGenerateStep()
    try:
        step.prepare_params({})
    except ValueError as exc:
        assert gate["ready"] is False
        assert gate["message"] == str(exc)
    else:
        assert gate["ready"] is True
        assert gate["blockers"] == []


def test_tng_radii_item_reads_the_validation_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    assert overview.tng_radii_item()["state"] == "unknown"
    path = overview.tng_radii_cache_path()
    path.parent.mkdir(parents=True)
    now = time.time()
    path.write_text(json.dumps({"valid": True, "valid_count": 5770, "expected_count": 5770,
                                "checked_at": now - 10}))
    item = overview.tng_radii_item(now)
    assert item["state"] == "ok"
    assert item["detail"] == "5770/5770 radii valid"
    assert item["label"] == "TNG radii"
    assert item["action"]["confirm"]
    assert item["facts"]["stale"] is False
    assert overview.tng_radii_item(now + 7200)["facts"]["stale"] is True
    path.write_text(json.dumps({"valid": False, "failed": True, "reasons": ["ssh timeout"],
                                "checked_at": now}))
    item = overview.tng_radii_item(now)
    assert (item["state"], item["detail"]) == ("warn", "ssh timeout")
    assert item["action"]["url"] == "/api/tng/radii/refresh"


def test_galaxy_artifact_state_reports_missing_and_stale(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    assert galaxy_distributions.artifact_state()["present"] is False
    monkeypatch.setattr(galaxy_distributions, "_inputs", lambda: {"x": 1})
    path = galaxy_distributions.artifact_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"version": galaxy_distributions.ARTIFACT_VERSION, "inputs": {"x": 1}}))
    state = galaxy_distributions.artifact_state()
    assert (state["present"], state["stale"], state["reason"]) == (True, False, None)
    path.write_text(json.dumps({"version": galaxy_distributions.ARTIFACT_VERSION, "inputs": {"x": 2}}))
    assert "inputs changed" in galaxy_distributions.artifact_state()["reason"]


def _tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else []


def test_realism_get_handlers_never_write_into_data(monkeypatch, tmp_path):
    """Every Realism GET is read-only (spec §9.5: mutations are POSTs/jobs),
    even with nothing cached yet."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(Config, "DATA_DIR", str(data))
    monkeypatch.setattr(realism_routes, "check_records_noise", lambda: {"state": "unknown", "title": "x"})
    client = create_app().test_client()
    for url in ("/api/realism/overview", "/api/noise", "/api/noise/positions/102021990",
                "/api/galaxy-distributions", "/api/galaxy-distributions/joint-pair?x=vis&y=log_re",
                "/api/star-distribution", "/api/star-distribution?include_training=1",
                "/api/population-comparison", "/api/archive-fields"):
        response = client.get(url)
        assert response.status_code == 200, (url, response.get_data(as_text=True)[:300])
    assert _tree(data) == []


def test_star_distribution_get_computes_in_memory_and_fit_persists(monkeypatch, tmp_path):
    """A stale stellar plot cache is recomputed in memory by the GET (no
    write); the POST fit job is what persists it."""
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    euclid = tmp_path / "euclid.csv"
    gaia = tmp_path / "gaia.csv"
    euclid.write_text("x\n")
    gaia.write_text("x\n")
    monkeypatch.setattr(star_population, "euclid_catalog_path", lambda: euclid)
    monkeypatch.setattr(star_population, "gaia_catalog_path", lambda: gaia)
    calls = []

    def fake_rows(*_a, **_kw):
        calls.append(1)
        return {"matched_stars": 3, "version": -1}

    monkeypatch.setattr(star_population, "_star_distribution_from_rows", fake_rows)
    monkeypatch.setattr(star_population, "_synthetic_paths",
                        lambda include_training=False: ([], []))
    star_population._DISTRIBUTION_MEMO.clear()
    target = star_population.star_distribution_path()
    first = star_population.star_distribution_payload()
    again = star_population.star_distribution_payload()
    assert first["matched_stars"] == 3 and again is first
    assert len(calls) == 1                      # memoised for the process
    assert not target.exists()                  # the GET path wrote nothing
    star_population.star_distribution_payload(persist=True)
    assert json.loads(target.read_text())["matched_stars"] == 3
    star_population._DISTRIBUTION_MEMO.clear()


def test_noise_position_endpoint_serves_the_sub_grid():
    client = create_app().test_client()
    position = client.get("/api/noise/positions/102021990").get_json()
    assert position["field"] == "EDF-S"
    assert position["bands"] == ["VIS", "Y_E", "J_E", "H_E"]
    assert position["grid_side"] == 4
    assert len(position["sub_levels_e"]["Y_E"]) == 16
    # This tile's NISP grids step ×~1.2 between the top row and the rest.
    assert position["steps"]["Y_E"]["step"] > 1.1
    assert position["levels_e"]["VIS"] == pytest.approx(27.0469)
    missing = client.get("/api/noise/positions/nope")
    assert missing.status_code == 404
    assert "no measured noise position" in missing.get_json()["error"]
    assert noise_position("nope") is None


def test_training_sync_job_can_rebuild_the_galaxy_plots(monkeypatch):
    class Result:
        ok = True
        error = None
        local_path = "/tmp/sources_train.csv"
        size_bytes = 12

    class Cap:
        def __init__(self):
            self.ticks = []

        def tick(self, *args):
            self.ticks.append(args)

        def write(self, _text):
            pass

    order = []
    monkeypatch.setattr(pc_routes, "ensure_ssh_connected", lambda: order.append("ssh"))
    monkeypatch.setattr(pc_routes.fasrc_fetcher, "fetch_one_file",
                        lambda *_a, **_kw: order.append("fetch") or Result())
    monkeypatch.setattr(pc_routes, "refresh_population_comparison", lambda: order.append("census"))
    monkeypatch.setattr(pc_routes, "build_galaxy_distributions",
                        lambda: order.append("plots") or {"version": 25})
    captured = {}

    def spawn(*, label, target):
        captured["target"] = target
        return "job-1"

    monkeypatch.setattr(pc_routes.REGISTRY, "spawn", spawn)
    client = create_app().test_client()
    assert client.post("/api/population-comparison/sync-training-catalog",
                       data={"rebuild": "1"}).get_json() == {"ok": True, "job_id": "job-1"}
    cap = Cap()
    result = captured["target"](cap)
    assert order == ["ssh", "fetch", "census", "plots"]
    assert result["galaxy_plots_version"] == 25
    assert cap.ticks[-1][:2] == (4, 4)

    order.clear()
    client.post("/api/population-comparison/sync-training-catalog")
    captured["target"](Cap())
    assert order == ["ssh", "fetch", "census"]   # plain sync: no rebuild


def test_field_statistics_row_owns_the_real_reference_problem():
    """No separate archive row: a missing or changed real reference is the
    field statistics row's warning, with the sync as its fix."""
    missing = overview.comparison_item(_availability(real={"ready": False, "unavailable_reason": "not synced"}))
    assert (missing["state"], missing["title"]) == ("warn", "Real reference fields not synced")
    assert missing["to"] == "/synthetic/fields?ref=1"
    assert missing["action"]["url"] == "/api/archive-fields/sync"
    changed = overview.comparison_item(_availability(real={"ready": True, "current": False, "fields": 220}))
    assert changed["title"] == "Real reference fields changed upstream"
    ready = overview.comparison_item(_availability(real={"ready": True, "current": True, "fields": 220,
                                                         "compared_fields": 176}))
    assert ready["action"]["url"] == "/api/population-comparison/build"
    assert ready["facts"]["real_fields"] == 176
