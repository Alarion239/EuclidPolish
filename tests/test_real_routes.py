"""HTTP surface of contract C9 (plan WP-B2 T2/T3/T5/T6/T8): ``/api/real/*``,
``/api/models``, ``/api/experiments`` and the ``real`` viewer collection —
JSON errors, local jobs, 2-D ``image.fits`` slices and per-tier WCS."""

from __future__ import annotations

import io
import json
import time

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from euclid_polish.web import app as web_app
from euclid_polish.web import remote
from euclid_polish.web.helpers import experiments, jwst_euclid, model_catalog, real_tiles
from euclid_polish.web.jobs import REGISTRY
from tests import _real_fixtures as fx

TOL_DEG = 1e-9


@pytest.fixture
def world(tmp_path, monkeypatch):
    store = fx.point_store(tmp_path, monkeypatch)
    fx.stub_regime(tmp_path, monkeypatch)
    fx.make_nexus_field(n_tiles=2)
    runner = fx.StubRunner()
    monkeypatch.setattr(model_catalog, "EnsembleMemberRunner", lambda *a, **k: runner)
    return {**store, "runner": runner}


@pytest.fixture
def client(world):
    app = web_app.create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _wait(job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = REGISTRY.get(job_id)
        if job is not None and job.status != "running":
            return job.to_dict()
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


def _run_experiment(client, tiles="nexus/f200w-0000", models="mean,member:member_1"):
    response = client.post("/api/experiments", data={"tiles": tiles, "models": models})
    assert response.status_code == 200, response.get_json()
    payload = response.get_json()
    job = _wait(payload["job_id"])
    assert job["status"] == "done", job["error"]
    assert job["kind"] == "real-experiment"
    return payload, job


def test_sources_list_and_card(client):
    sources = client.get("/api/real/sources").get_json()["sources"]
    nexus = next(item for item in sources if item["id"] == "nexus")
    assert nexus["count"] == 2 and nexus["has_jwst"] and nexus["ready"]
    listing = client.get("/api/real/nexus").get_json()
    assert listing["count"] == 2
    row = listing["tiles"][0]
    assert row["ref"] == "nexus/f200w-0000" and row["production_state"] == "missing"
    assert row["models"] == {} and row["tiers"] == ["lr", "jwst"]
    card = client.get("/api/real/nexus/f200w-0000").get_json()
    assert card["q1_tile"]["tile"] == "102158584"
    assert card["viewer"] == {"collection": "real", "params": {"source": "nexus"},
                              "id": "f200w-0000"}
    assert "production" in card["runnable_models"]
    assert set(card["image_urls"]) == {"lr", "jwst"}


@pytest.mark.parametrize("path,status", [
    ("/api/real/mars", 404),
    ("/api/real/nexus/f200w-9999", 404),
    ("/api/real/nexus/f200w-0000/nothing-here", 404),
    ("/api/experiments/20260926-000000-abcdef", 404),
])
def test_errors_are_json(client, path, status):
    response = client.get(path)
    assert response.status_code == status
    assert response.is_json and response.get_json()["error"]


def test_models_payload(client):
    payload = client.get("/api/models").get_json()
    assert payload["regime"] == "starfull" and payload["members"] == fx.LABELS
    specs = {item["spec"]: item for item in payload["models"]}
    assert specs["production"]["available"] and specs["gate:two"]["available"]
    assert specs["production"]["members"] == fx.LABELS
    assert not specs["rbf"]["available"] and specs["rbf"]["reason"]


def test_experiment_job_record_and_card(client, world):
    payload, job = _run_experiment(client)
    assert payload["models"] == ["mean", "member:member_1"]
    assert job["result"]["experiment_id"] == payload["experiment_id"]
    record = client.get(f"/api/experiments/{payload['experiment_id']}").get_json()
    assert record["status"] == "done" and record["job_id"] == payload["job_id"]
    vis = record["results"]["nexus/f200w-0000"]["mean"]["metrics"]["per_band"]["VIS"]
    assert vis["flux_ratio"] == pytest.approx(2.0, rel=1e-5)
    listing = client.get("/api/experiments").get_json()["experiments"]
    assert listing[0]["id"] == payload["experiment_id"]
    card = client.get("/api/real/nexus/f200w-0000").get_json()
    assert card["models"]["mean"]["state"] == "current"
    assert card["models"]["mean"]["metrics"]["summary"]["n_peaks"] >= 1
    assert card["experiments"] == [payload["experiment_id"]]
    row = client.get("/api/real/nexus").get_json()["tiles"][0]
    assert row["models"]["mean"]["state"] == "current"


@pytest.mark.parametrize("data,status", [
    ({"tiles": "", "models": "mean"}, 400),
    ({"tiles": "nexus/f200w-0000", "models": "bogus"}, 400),
    ({"tiles": "nexus/f200w-0000", "models": "rbf"}, 400),
    ({"tiles": "nexus/f200w-0099", "models": "mean"}, 404),
    ({"tiles": "mars/1", "models": "mean"}, 404),
])
def test_experiment_request_validation(client, data, status):
    response = client.post("/api/experiments", data=data)
    assert response.status_code == status
    assert response.get_json()["ok"] is False


def test_image_fits_slices_carry_their_wcs(client, world):
    _run_experiment(client, models="mean")
    lr = client.get("/api/real/nexus/f200w-0000/image.fits?tier=lr&band=J_E")
    assert lr.status_code == 200 and lr.mimetype == "application/fits"
    with fits.open(io.BytesIO(lr.data)) as hdul:
        lr_data, lr_header = hdul[0].data, hdul[0].header
    assert lr_data.shape == (40, 40) and lr_header["BUNIT"] == "electron"
    tile = real_tiles.get_tile("nexus", "f200w-0000")
    np.testing.assert_allclose(lr_data, tile.lr_e[..., 2])
    source = WCS(tile.wcs_header).pixel_to_world_values(0, 0)
    served = WCS(lr_header).pixel_to_world_values(0, 0)
    assert abs(float(served[0]) - float(source[0])) < TOL_DEG
    sr = client.get("/api/real/nexus/f200w-0000/image.fits?tier=m:mean&band=VIS")
    with fits.open(io.BytesIO(sr.data)) as hdul:
        sr_data, sr_header = hdul[0].data, hdul[0].header
    assert sr_data.shape == (80, 80)
    # SR pixel (0, 0) centre = LR pixel (-0.25, -0.25)
    expected = WCS(tile.wcs_header).pixel_to_world_values(-0.25, -0.25)
    got = WCS(sr_header).pixel_to_world_values(0, 0)
    assert abs(float(got[0]) - float(expected[0])) < TOL_DEG
    assert abs(float(got[1]) - float(expected[1])) < TOL_DEG
    jwst = client.get("/api/real/nexus/f200w-0000/image.fits?tier=jwst")
    with fits.open(io.BytesIO(jwst.data)) as hdul:
        assert hdul[0].header["BUNIT"] == "MJy/sr" and hdul[0].data.shape == (133, 133)
    assert client.get("/api/real/nexus/f200w-0000/image.fits?tier=lr&band=K").status_code == 400
    assert client.get("/api/real/nexus/f200w-0000/image.fits?tier=zz").status_code == 400
    missing = client.get("/api/real/nexus/f200w-0000/image.fits?tier=m:production")
    assert missing.status_code == 404 and "has not been run" in missing.get_json()["error"]


def test_real_viewer_collection(client, world):
    _run_experiment(client, models="mean")
    meta = client.get("/viewer/meta/real?source=nexus").get_json()
    assert meta["collection"] == "real" and meta["count"] == 2
    assert [tier["key"] for tier in meta["tiers"]] == ["lr", "jwst", "m:mean"]
    first, second = meta["objects"]
    assert first["id"] == "f200w-0000" and "m:mean" in first["tiers"]
    assert first["model_states"] == {"mean": "current"}
    assert "m:mean" not in second["tiers"]
    lr = client.get("/viewer/cube/real/0?source=nexus&tier=lr")
    assert lr.status_code == 200 and lr.headers["X-Cube-Unit"] == "e-"
    lr_wcs = fx.served_wcs(lr.headers["X-Cube-WCS"])
    sr = client.get("/viewer/cube/real?source=nexus&tier=m:mean&id=f200w-0000")
    assert sr.status_code == 200 and sr.headers["X-Cube-Index"] == "0"
    sr_wcs = fx.served_wcs(sr.headers["X-Cube-WCS"])
    expected = lr_wcs.pixel_to_world_values(-0.25, -0.25)
    got = sr_wcs.pixel_to_world_values(0, 0)
    assert abs(float(got[0]) - float(expected[0])) < TOL_DEG
    jwst = client.get("/viewer/cube/real/0?source=nexus&tier=jwst")
    assert jwst.headers["X-Cube-Unit"] == "MJy/sr"
    explicit = client.get("/viewer/meta/real?source=nexus&models=production,member:1").get_json()
    assert [t["key"] for t in explicit["tiers"]][2:] == ["m:production", "m:member:member_1"]
    assert client.get("/viewer/cube/real/1?source=nexus&tier=m:mean").status_code == 404
    assert client.get("/viewer/meta/real?source=mars").status_code == 404
    assert client.get("/viewer/meta/real?source=nexus&models=bogus").status_code == 400
    assert client.get("/viewer/cube/real/9?source=nexus&tier=lr").status_code == 404


def test_cache_tile_endpoint(client, world, monkeypatch):
    assert client.post("/api/real/tiles", data={"ra": "x", "dec": "1"}).status_code == 400
    assert client.post("/api/real/tiles", data={"ra": "400", "dec": "1"}).status_code == 400
    outside = client.post("/api/real/tiles", data={"ra": "150.1", "dec": "2.2"})
    assert outside.status_code == 400 and outside.get_json()["code"] == "outside_q1"
    assert client.post("/api/real/tiles", data={"ra": "268.4", "dec": "65.2",
                                                "run": "bogus"}).status_code == 400
    # inside Q1 but only in a tile measured as unobserved (102018211, EDF-S)
    empty = client.post("/api/real/tiles", data={"ra": "57.9990518", "dec": "-51.4999861"})
    assert empty.status_code == 400 and empty.get_json()["code"] == "unobserved_q1"
    assert empty.get_json()["tile"] == "102018211"
    monkeypatch.setattr(jwst_euclid, "fetch_q1_cutout", fx.fake_cutout_writer())
    response = client.post("/api/real/tiles", data={"ra": str(fx.NEXUS_RA),
                                                    "dec": str(fx.NEXUS_DEC), "run": "mean"})
    payload = response.get_json()
    assert response.status_code == 200 and payload["ref"].startswith("tile/")
    job = _wait(payload["job_id"])
    assert job["status"] == "done", job["error"]
    assert job["kind"] == "real-tile"
    assert job["result"]["tile"] == payload["id"]
    assert job["result"]["experiment"]["experiment_id"] == payload["experiment_id"]
    tiles = client.get("/api/real/tile").get_json()["tiles"]
    assert [row["id"] for row in tiles] == [payload["id"]]
    assert tiles[0]["models"]["mean"]["state"] == "current"
    deleted = client.post(f"/api/real/tile/{payload['id']}/delete-outputs").get_json()
    assert deleted["ok"] and deleted["removed_count"] > 0
    assert client.get("/api/real/tile").get_json()["tiles"][0]["models"] == {}


def test_every_new_endpoint_works_offline(client, monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", None)
    for path in ("/api/real/sources", "/api/real/nexus", "/api/real/nexus/f200w-0000",
                 "/api/models", "/api/experiments", "/api/sky/layers",
                 "/api/sky/at?ra=268.4625&dec=65.19917"):
        response = client.get(path)
        assert response.status_code == 200, (path, response.get_json())
        json.dumps(response.get_json())


def _sr_matches_lr_grid(lr_header: fits.Header, sr_header: fits.Header) -> None:
    """LR pixel centre (x, y) == SR pixel (2x + 0.5, 2y + 0.5) on the sky."""
    wl, ws = WCS(lr_header).celestial, WCS(sr_header).celestial
    for x, y in [(0.0, 0.0), (10.0, 3.0), (39.0, 39.0)]:
        a = wl.pixel_to_world_values(x, y)
        b = ws.pixel_to_world_values(2 * x + 0.5, 2 * y + 0.5)
        assert abs(float(a[0]) - float(b[0])) < TOL_DEG
        assert abs(float(a[1]) - float(b[1])) < TOL_DEG


def test_nexus_field_production_inference_is_served_by_the_real_api(client, world):
    """The whole-field NEXUS path writes the legacy per-tile SR only; the C9
    API must still list it, serve it as ``m:production`` and call it current."""
    field = jwst_euclid.nexus_field_id("F200W")
    jwst_euclid.run_starfull_nexus_field_inference(field, tiles=["f200w-0000"],
                                                   runner=world["runner"])
    real_tiles.invalidate()
    card = client.get("/api/real/nexus/f200w-0000").get_json()
    assert card["production_state"] == "current"
    production = card["models"]["production"]
    assert production["state"] == "current" and production["legacy"] is True
    assert production["origin"] == "nexus-field"
    assert "m:production" in card["image_urls"]
    lr = client.get("/api/real/nexus/f200w-0000/image.fits?tier=lr&band=VIS")
    sr = client.get("/api/real/nexus/f200w-0000/image.fits?tier=m:production&band=H_E")
    assert sr.status_code == 200, sr.get_json()
    with fits.open(io.BytesIO(sr.data)) as hdul:
        assert hdul[0].data.shape == (80, 80)
        sr_header = hdul[0].header
    _sr_matches_lr_grid(fits.getheader(io.BytesIO(lr.data)), sr_header)
    listing = client.get("/api/real/nexus").get_json()["tiles"]
    assert [row["production_state"] for row in listing] == ["current", "missing"]
    meta = client.get("/viewer/meta/real?source=nexus").get_json()
    assert "m:production" in [tier["key"] for tier in meta["tiers"]]
    first = meta["objects"][0]
    assert "m:production" in first["tiers"] and first["model_states"]["production"] == "current"
    assert first["legacy_models"] == ["production"]
    cube = client.get("/viewer/cube/real/0?source=nexus&tier=m:production")
    assert cube.status_code == 200 and cube.headers["X-Cube-Unit"] == "e-"
    _sr_matches_lr_grid(fits.getheader(io.BytesIO(lr.data)),
                        fits.Header(json.loads(cube.headers["X-Cube-WCS"])))
    layer = client.get("/api/sky/layer/nexus-tiles").get_json()
    assert {f["id"]: f["props"]["state"] for f in layer["features"]} == {
        "f200w-0000": "current", "f200w-0001": "missing"}
    # an experiment reuses the current legacy SR (no member run) and scores it
    world["runner"].calls.clear()
    payload, _job = _run_experiment(client, models="production")
    assert world["runner"].calls == []
    record = client.get(f"/api/experiments/{payload['experiment_id']}").get_json()
    assert record["results"]["nexus/f200w-0000"]["production"]["state"] == "reused"


def test_legacy_sr_wcs_comes_from_the_lr_grid_not_the_old_file(client, world):
    """Pre-fix NEXUS SR files carry CRPIX = 2c − 1; the served WCS is always
    the LR WCS ×2 (CRPIX = 2c − 0.5)."""
    field = jwst_euclid.nexus_field_id("F200W")
    directory = jwst_euclid.nexus_field_root() / field
    manifest = json.loads((directory / "manifest.json").read_text())
    tile = manifest["tiles"][1]
    lr_header = fits.getheader(directory / tile["lr_file"])
    bad = fits.Header()
    for key in ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2"):
        bad[key] = lr_header[key]
    bad["CRPIX1"], bad["CRPIX2"] = 2 * lr_header["CRPIX1"] - 1, 2 * lr_header["CRPIX2"] - 1
    bad["CD1_1"], bad["CD2_2"] = lr_header["CD1_1"] / 2, lr_header["CD2_2"] / 2
    bad["CD1_2"] = bad["CD2_1"] = 0.0
    fits.PrimaryHDU(np.ones((4, 80, 80), np.float32), header=bad).writeto(
        directory / "tiles" / "starfull_combiner_0001.fits")
    tile["inference"] = {"combiner_kind": "raw_incremental_minmeanmax_rbf",
                         "combiner_fingerprint": "old", "member_labels": fx.LABELS,
                         "member_fingerprints": ["a", "b", "c"],
                         "files": {"lr": tile["lr_file"],
                                   "starfull": "tiles/starfull_combiner_0001.fits"}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    real_tiles.invalidate()
    card = client.get("/api/real/nexus/f200w-0001").get_json()
    assert card["production_state"] == "stale"            # an older production SR
    assert card["models"]["rbf"]["legacy"] and card["models"]["rbf"]["state"] == "unavailable"
    sr = client.get("/api/real/nexus/f200w-0001/image.fits?tier=m:rbf&band=VIS")
    assert sr.status_code == 200, sr.get_json()
    _sr_matches_lr_grid(lr_header, fits.getheader(io.BytesIO(sr.data)))


def test_pair_inference_outputs_are_served_and_tracked(client, world):
    pair = fx.make_pair()
    real_tiles.invalidate()
    before = client.get(f"/api/real/pair/{pair}").get_json()
    assert before["production_state"] == "missing" and before["model_ready"]
    jwst_euclid.run_starfull_pair_inference(pair, runner=world["runner"])
    jwst_euclid.run_starfull_pair_inference(pair, spec="mean", runner=world["runner"])
    real_tiles.invalidate()
    card = client.get(f"/api/real/pair/{pair}").get_json()
    assert card["production_state"] == "current"
    assert card["legacy"]["spec"] == "production" and card["legacy"]["identity"]
    assert {spec: m["state"] for spec, m in card["models"].items()} == {
        "production": "current", "mean": "current"}
    image = client.get(f"/api/real/pair/{pair}/image.fits?tier=m:mean&band=VIS")
    assert image.status_code == 200
    with fits.open(io.BytesIO(image.data)) as hdul:
        assert hdul[0].data.shape == (60, 60)


def test_poster_sr_is_served_as_a_legacy_tier(client, world):
    fx.make_poster(world["poster"])
    real_tiles.invalidate()
    card = client.get("/api/real/poster/target_181255_test").get_json()
    rbf = card["models"]["rbf"]
    assert rbf["legacy"] and rbf["origin"] == "poster" and rbf["state"] == "unavailable"
    image = client.get("/api/real/poster/target_181255_test/image.fits?tier=m:rbf&band=J_E")
    assert image.status_code == 200
    with fits.open(io.BytesIO(image.data)) as hdul:
        assert hdul[0].data.shape == (64, 64)


def test_experiment_refused_when_disk_is_full(client, monkeypatch):
    monkeypatch.setattr(experiments, "free_bytes", lambda _path: 0)
    response = client.post("/api/experiments", data={"tiles": "nexus/f200w-0000",
                                                     "models": "mean"})
    payload = response.get_json()
    assert response.status_code == 507 and payload["code"] == "insufficient_storage"
    assert payload["needed_bytes"] > 0 and payload["free_bytes"] == 0


def test_card_disk_counts_the_member_cache(client, world):
    _run_experiment(client, models="mean")
    disk = client.get("/api/real/nexus/f200w-0000").get_json()["disk"]
    assert disk["cache_bytes"] >= 3 * 80 * 80 * 4 * 4 and disk["output_bytes"] > 0
    assert disk["total_bytes"] == sum(disk[key] for key in (
        "tile_bytes", "output_bytes", "cache_bytes", "legacy_bytes"))
