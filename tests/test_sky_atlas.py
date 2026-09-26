"""Sky atlas backend (contract C9, plan WP-B2 T7): layer catalogue and compact
feature shapes over local data, point lookup, JWST discovery against the
committed Q1 polygons (MAST mocked), footprint cone queries, pairing."""

from __future__ import annotations

import csv
import json
import time

import pytest

from euclid_polish.sky.observation import q1_mer_tiles
from euclid_polish.web import app as web_app
from euclid_polish.web.helpers import jwst_euclid, sky_atlas
from euclid_polish.web.jobs import REGISTRY
from tests import _real_fixtures as fx

JADES = (53.16, -27.78)
# A NIRCam footprint around JADES (inside Q1 tile 102044185, EDF-F).
JADES_REGION = "POLYGON ICRS 53.10 -27.83 53.22 -27.83 53.22 -27.73 53.10 -27.73"


@pytest.fixture
def world(tmp_path, monkeypatch):
    store = fx.point_store(tmp_path, monkeypatch)
    fx.stub_regime(tmp_path, monkeypatch)
    fx.make_nexus_field(n_tiles=2)
    fx.make_eval_store()
    fx.make_poster(store["poster"])
    return store


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


def _mast_rows():
    return [
        {"obs_id": "jw01180-o001_t001_nircam_clear-f200w", "instrument_name": "NIRCAM/IMAGE",
         "filters": "F200W", "target_name": "GOODS-S", "dataRights": "PUBLIC",
         "t_exptime": 1000.0, "s_ra": 53.16, "s_dec": -27.78, "s_region": JADES_REGION,
         "proposal_id": "1180"},
        {"obs_id": "jw9-miri", "instrument_name": "MIRI/IMAGE", "filters": "F770W",
         "dataRights": "PUBLIC", "s_ra": 53.16, "s_dec": -27.78, "s_region": ""},
        {"obs_id": "jw9-spec", "instrument_name": "NIRSPEC/MSA", "dataRights": "PUBLIC",
         "s_ra": 53.16, "s_dec": -27.78, "s_region": JADES_REGION},
        {"obs_id": "jw9-private", "instrument_name": "NIRCAM/IMAGE", "filters": "F444W",
         "dataRights": "EXCLUSIVE_ACCESS", "s_ra": 53.16, "s_dec": -27.78,
         "s_region": JADES_REGION},
    ]


@pytest.fixture
def mast(monkeypatch):
    calls = []

    def fake(scope, *, cache_dir, refresh):
        calls.append(scope)
        return _mast_rows()

    monkeypatch.setattr(jwst_euclid, "_mast_rows_for_scope", fake)
    return calls


def test_layer_catalogue_lists_every_spec_layer(world):
    payload = sky_atlas.layers_payload()
    assert payload["groups"] == ["coverage", "results", "catalogues"]
    layers = {layer["id"]: layer for layer in payload["layers"]}
    for key in ("q1-tiles", "q1-fields", "nexus-footprint", "nexus-tiles", "real-tiles",
                "real-fields", "poster", "pairs", "archive-fields", "eval-objects",
                "experiments", "lens-candidates", "galaxies", "stars", "psf-clusters",
                "noise-positions", "population-cones", "gaia-fields", "jwst-mast"):
        assert key in layers, key
        layer = layers[key]
        assert {"id", "label", "group", "kind", "count", "bbox", "style", "ready",
                "reason", "fill_action", "url"} <= set(layer)
    assert layers["q1-tiles"]["count"] == 352 and layers["q1-tiles"]["ready"]
    assert layers["noise-positions"]["count"] == 294
    assert layers["nexus-tiles"]["count"] == 2 and layers["poster"]["count"] == 1
    assert layers["eval-objects"]["count"] == 1
    assert not layers["jwst-mast"]["ready"]
    assert layers["jwst-mast"]["fill_action"]["url"] == "/api/sky/jwst/discover"
    assert layers["stars"]["fill_action"]["requires_fasrc"] is True
    json.dumps(payload, allow_nan=False)


def test_feature_shapes(world):
    tiles = sky_atlas.layer_features("q1-tiles")
    assert tiles["kind"] == "polygons" and tiles["count"] == 352
    feature = tiles["features"][0]
    assert len(feature["polygon"]) == 4 and feature["inspect"]["kind"] == "source"
    assert feature["props"]["state"] in {"measured", "rejected", "unmeasured"}
    nexus = sky_atlas.layer_features("nexus-tiles")
    first = nexus["features"][0]
    assert first["inspect"] == {"kind": "realtile", "id": "nexus/f200w-0000"}
    assert first["props"]["state"] == "missing" and first["props"]["has_jwst"]
    evals = sky_atlas.layer_features("eval-objects")
    assert evals["kind"] == "points" and evals["columns"][:2] == ["ra", "dec"]
    assert evals["inspect"] == {"kind": "realtile", "prefix": "eval/", "id_column": "id"}
    fields = sky_atlas.layer_features("q1-fields")
    assert fields["kind"] == "circles" and fields["features"][0]["radius_deg"] == 6.0
    hull = sky_atlas.layer_features("nexus-footprint")["features"][0]["polygon"]
    assert q1_mer_tiles.point_in_polygon(fx.NEXUS_RA + 0.001, fx.NEXUS_DEC, hull)


def test_catalogue_layers_from_local_files(world, tmp_path, monkeypatch):
    stars = tmp_path / "stars.csv"
    with stars.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "ra", "dec", "magnitude", "valid:VIS:255", "valid:H_E:511",
                         "valid:J_E:255"])
        writer.writerow([0, 268.1, 65.1, 17.5, "True", "True", ""])
        writer.writerow([1, 53.0, -27.9, 18.2, "", "", "True"])
    monkeypatch.setattr(sky_atlas, "stars_path", lambda: stars)
    payload = sky_atlas.layer_features("stars")
    assert payload["columns"] == ["ra", "dec", "mag", "flags"]
    assert payload["rows"] == [[268.1, 65.1, 17.5, 1 | 8], [53.0, -27.9, 18.2, 4]]
    assert payload["flag_bits"] == {"VIS": 1, "Y_E": 2, "J_E": 4, "H_E": 8}
    lenses = sky_atlas.lens_catalog_path()
    lenses.parent.mkdir(parents=True)
    lenses.write_text("id,ra,dec,grade,subset\nL1,53.1,-27.7,A,discovery_engine\n")
    rows = sky_atlas.layer_features("lens-candidates")["rows"]
    assert rows == [[53.1, -27.7, "A", "discovery_engine", "L1", "EDF-F"]]
    meta = sky_atlas.population_meta_path()
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"radius_arcmin": 5.0,
                                "cones": [{"ra": 61.0, "dec": -48.0, "rows": 10}]}))
    cone = sky_atlas.layer_features("population-cones")["features"][0]
    assert cone["radius_deg"] == pytest.approx(5.0 / 60.0) and cone["props"]["field"] == "EDF-S"


def test_at_reports_q1_tile_real_tiles_and_jwst(world, mast):
    payload = sky_atlas.at(fx.NEXUS_RA, fx.NEXUS_DEC)
    assert payload["in_q1"] and payload["best_tile"] == "102158584"
    assert payload["field"] == "EDF-N"
    assert [item["ref"] for item in payload["nexus"]] == ["nexus/f200w-0000"]
    assert payload["q1_tiles"][0]["margin_arcsec"] > 0
    assert not payload["jwst_discovered"]
    jwst_euclid.discover_jwst_overlap(fields=["EDF-F"])
    jades = sky_atlas.at(*JADES)
    assert [item["obs_id"] for item in jades["jwst"]] == [
        "jw01180-o001_t001_nircam_clear-f200w"]
    outside = sky_atlas.at(150.1, 2.2)
    assert not outside["in_q1"] and outside["q1_tiles"] == [] and outside["field"] is None
    assert outside["q1_verdict"] == "outside" and not outside["q1_observed"]
    assert payload["q1_verdict"] == "observed" and payload["q1_observed"]


def test_at_flags_points_inside_only_unobserved_q1_tiles(world):
    payload = sky_atlas.at(57.9990518, -51.4999861)             # rejected tile 102018211
    assert payload["in_q1"] and not payload["q1_observed"]
    assert payload["q1_verdict"] == "unobserved"
    assert payload["q1_tiles"][0]["rejected"].startswith("no coverage")


def test_parse_s_region_handles_multiple_polygons():
    single = jwst_euclid.parse_s_region(JADES_REGION)
    assert single == [[[53.10, -27.83], [53.22, -27.83], [53.22, -27.73], [53.10, -27.73]]]
    double = jwst_euclid.parse_s_region(
        "POLYGON 10 1 11 1 11 2 10 2 POLYGON ICRS 20 1 21 1 21 2 20 2")
    assert len(double) == 2 and double[1][0] == [20.0, 1.0]
    assert jwst_euclid.parse_s_region("CIRCLE ICRS 10 20 1") == []
    assert jwst_euclid.parse_s_region("") == []


def test_discovery_writes_exact_rows_footprints_and_merges(world, mast, monkeypatch):
    result = jwst_euclid.discover_jwst_overlap(fields=["EDF-F"])
    assert result["fields"] == ["EDF-F"] and mast
    rows, _status = jwst_euclid.overlap_rows()
    by_obs = {}
    for row in rows:
        by_obs.setdefault(row["jwst_observation_id"], []).append(row)
    assert set(by_obs) == {"jw01180-o001_t001_nircam_clear-f200w", "jw9-miri"}
    exact = by_obs["jw01180-o001_t001_nircam_clear-f200w"]
    assert "102044185" in {row["euclid_tile_index"] for row in exact}
    assert {row["footprint_status"] for row in exact} == {"exact_intersection"}
    assert {row["footprint_status"] for row in by_obs["jw9-miri"]} == {"candidate_only"}
    footprints = jwst_euclid.load_footprints()["footprints"]
    assert set(footprints) == {"jw01180-o001_t001_nircam_clear-f200w", "jw9-miri"}
    assert footprints["jw01180-o001_t001_nircam_clear-f200w"]["fields"] == ["EDF-F"]
    cone = jwst_euclid.footprints_in_cone(*JADES, 0.2)
    assert cone["ready"] and cone["count"] == 2
    assert jwst_euclid.footprints_in_cone(268.0, 65.0, 0.2)["count"] == 0
    groups, _ = jwst_euclid.location_groups()
    assert groups                      # discovery feeds the pairing input
    # a later discovery merges instead of overwriting
    mast_rows = _mast_rows()[:1]
    mast_rows[0] = {**mast_rows[0], "obs_id": "jw-second"}
    monkeypatch.setattr(jwst_euclid, "_mast_rows_for_scope", lambda *a, **k: mast_rows)
    jwst_euclid.discover_jwst_overlap(fields=["EDF-F"])
    ids = {row["jwst_observation_id"] for row in jwst_euclid.overlap_rows()[0]}
    assert {"jw-second", "jw01180-o001_t001_nircam_clear-f200w"} <= ids


def test_discovery_scope_validation(world):
    with pytest.raises(ValueError):
        jwst_euclid.discover_jwst_overlap(region=(150.1, 2.2, 0.1))
    assert len(jwst_euclid.discovery_tiles(fields=["LDN1641"])) == 8
    assert len(jwst_euclid.discovery_tiles(fields=["edf-f"])) == 72


def test_convex_hull():
    points = [(10.0, 0.0), (11.0, 0.0), (11.0, 1.0), (10.0, 1.0), (10.5, 0.5)]
    hull = sky_atlas.convex_hull(points)
    assert len(hull) == 4 and [10.5, 0.5] not in hull
    assert sky_atlas.convex_hull([(1.0, 1.0)]) == []


def test_sky_routes(client, mast):
    assert client.get("/api/sky/layers").get_json()["layers"]
    layer = client.get("/api/sky/layer/nexus-tiles").get_json()
    assert layer["kind"] == "polygons" and layer["count"] == 2
    missing = client.get("/api/sky/layer/nope")
    assert missing.status_code == 404 and missing.is_json
    assert client.get("/api/sky/at?ra=268.4625").status_code == 400
    assert client.get("/api/sky/at?ra=999&dec=0").status_code == 400
    assert client.get("/api/sky/at?ra=268.4625&dec=65.19917").get_json()["in_q1"]
    assert client.get("/api/sky/nothing").is_json          # prefix-wide JSON errors
    assert client.get("/api/sky/jwst/footprints?ra=53&dec=-27&r=9").status_code == 400
    assert client.get("/api/sky/jwst/footprints?ra=53&dec=-27").get_json()["ready"] is False
    bad_field = client.post("/api/sky/jwst/discover", data={"fields": "COSMOS"})
    assert bad_field.status_code == 400 and "unknown Q1 field" in bad_field.get_json()["error"]
    assert client.post("/api/sky/jwst/discover", data={"region": "1,2"}).status_code == 400
    assert client.post("/api/sky/jwst/discover",
                       data={"region": "150.1,2.2,0.1"}).status_code == 400
    started = client.post("/api/sky/jwst/discover", data={"fields": "EDF-F"}).get_json()
    assert started["ok"] and started["tile_count"] == 72
    job = _wait(started["job_id"])
    assert job["status"] == "done", job["error"]
    assert job["kind"] == "jwst-discover" and job["result"]["footprint_count"] == 2
    cone = client.get("/api/sky/jwst/footprints?ra=53.16&dec=-27.78&r=0.3").get_json()
    assert cone["count"] == 2 and cone["footprints"][0]["polygons"]


def test_pair_route(client, mast, monkeypatch):
    not_found = client.post("/api/sky/jwst/pair", data={"obs_id": "jw-unknown"})
    assert not_found.status_code == 404 and not_found.get_json()["code"] == "not_discovered"
    nowhere = client.post("/api/sky/jwst/pair", data={"ra": "61.0", "dec": "-48.0"})
    assert nowhere.status_code == 404
    assert client.post("/api/sky/jwst/pair", data={"ra": "61", "dec": "-48",
                                                   "size_arcsec": "500"}).status_code == 400
    jwst_euclid.discover_jwst_overlap(fields=["EDF-F"])
    downloads, inputs = [], []
    monkeypatch.setattr(jwst_euclid, "download_and_align_pair",
                        lambda group, **kw: downloads.append(group) or
                        {"field_id": group["field_id"]})
    monkeypatch.setattr(jwst_euclid, "pair_lr_input",
                        lambda identifier, **kw: inputs.append(identifier))
    response = client.post("/api/sky/jwst/pair",
                           data={"obs_id": "jw01180-o001_t001_nircam_clear-f200w"})
    payload = response.get_json()
    assert response.status_code == 200 and payload["mode"] == "archive"
    job = _wait(payload["job_id"])
    assert job["status"] == "done", job["error"]
    assert job["kind"] == "jwst-pair" and inputs == [payload["pair_id"]]
    nexus_calls = []
    monkeypatch.setattr(jwst_euclid, "download_nexus_pair",
                        lambda **kw: nexus_calls.append(kw) or {"field_id": "nexus-pair"})
    near = client.post("/api/sky/jwst/pair",
                       data={"ra": str(fx.NEXUS_RA + 0.001), "dec": str(fx.NEXUS_DEC)}).get_json()
    assert near["mode"] == "nexus"
    assert _wait(near["job_id"])["status"] == "done"
    assert nexus_calls and nexus_calls[0]["filter_name"] == "F200W"


def test_pair_route_tests_the_nexus_tiles_not_their_hull(tmp_path, monkeypatch):
    """An L of three NEXUS tiles: the hull holds the empty corner, the tiles do not."""
    fx.point_store(tmp_path, monkeypatch)
    fx.stub_regime(tmp_path, monkeypatch)
    fx.make_nexus_field(offsets=[(0, 0), (1, 0), (0, 1)])
    app = web_app.create_app()
    app.config["TESTING"] = True
    corner = fx.tile_position(1.0, 1.0)                        # centre of the missing cell
    hull = sky_atlas.layer_features("nexus-footprint")["features"][0]["polygon"]
    assert q1_mer_tiles.point_in_polygon(*fx.tile_position(0.8, 0.8), hull)
    calls = []
    monkeypatch.setattr(jwst_euclid, "download_nexus_pair",
                        lambda **kw: calls.append(kw) or {"field_id": "nexus-pair"})
    monkeypatch.setattr(jwst_euclid, "pair_lr_input", lambda *a, **k: None)
    with app.test_client() as client:
        gap = client.post("/api/sky/jwst/pair", data={"ra": str(corner[0] - 0.0001),
                                                      "dec": str(corner[1] - 0.0001)})
        assert gap.status_code == 404 and gap.get_json()["code"] == "not_discovered"
        inside = client.post("/api/sky/jwst/pair", data={"ra": str(fx.tile_position(0, 1)[0]),
                                                         "dec": str(fx.tile_position(0, 1)[1])})
        assert inside.status_code == 200 and inside.get_json()["mode"] == "nexus"
        assert _wait(inside.get_json()["job_id"])["status"] == "done"
    assert calls and calls[0]["filter_name"] == "F200W"
