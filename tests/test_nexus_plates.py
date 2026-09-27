"""NEXUS × Euclid comparison plates (helpers/nexus_plates.py) and the
Figures endpoints that run and list them (routes/figures.py)."""
from __future__ import annotations

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from flask import Flask
from PIL import Image

from euclid_polish.web import jobs
from euclid_polish.web.helpers import nexus_plates
from euclid_polish.web.routes import figures


def _cube(side: int, channels: int, base: float) -> np.ndarray:
    y, x = np.mgrid[:side, :side]
    blob = np.exp(-((x - side / 2) ** 2 + (y - side / 2) ** 2) / (2 * (side / 8) ** 2))
    return np.stack([base * (i + 1) * blob + 1.0 for i in range(channels)], axis=-1).astype(np.float32)


OBJECTS = [
    {"id": "f200w-0040", "label": "NEXUS F200W tile 0040", "ra": 268.47, "dec": 65.14,
     "model_states": {"rbf": "current", "production": "current"}},
    {"id": "f200w-0042", "label": "NEXUS F200W tile 0042", "ra": 268.48, "dec": 65.15,
     "model_states": {"rbf": "current"}},
    {"id": "f200w-0070", "label": "NEXUS F200W tile 0070", "ra": 268.49, "dec": 65.16,
     "model_states": {"rbf": "stale"}},
]


@pytest.fixture
def plates(tmp_path, monkeypatch):
    monkeypatch.setenv("EUCLID_POLISH_NEXUS_PLATES_DIR", str(tmp_path / "plates"))
    loads: list[tuple[int, str, dict]] = []

    def fake_meta(collection, params):
        assert collection == "real" and params["source"] == "nexus"
        return {"objects": [dict(item) for item in OBJECTS]}

    def fake_cube(collection, index, tier, params):
        loads.append((index, tier, dict(params)))
        if tier == "lr":
            return _cube(16, 4, 50.0), {"bands": ["VIS", "Y_E", "J_E", "H_E"], "asinh": 100.0}
        if tier.startswith("m:"):
            return _cube(32, 4, 20.0), {"bands": ["VIS", "Y_E", "J_E", "H_E"], "asinh": 100.0,
                                       "label": f"{tier} · legacy", "legacy": True, "model_state": "current"}
        assert tier == "jwst"
        return _cube(48, 1, 3.0), {"bands": ["F200W"]}

    monkeypatch.setattr(nexus_plates.viewer_data, "get_meta", fake_meta)
    monkeypatch.setattr(nexus_plates.viewer_data, "get_cube", fake_cube)
    monkeypatch.setattr(nexus_plates.model_catalog, "list_specs", lambda: [
        SimpleNamespace(spec="rbf", label="RBF · minibatched", fingerprint="abc123", available=True),
        SimpleNamespace(spec="production", label="Production · gate", fingerprint="def456", available=True),
    ])
    monkeypatch.setattr(nexus_plates.real_tiles, "get_entry",
                        lambda source, identifier: SimpleNamespace(extras={"field_id": "nexus-field-1"}))
    monkeypatch.setattr(nexus_plates.real_tiles, "list_entries", lambda source: [
        SimpleNamespace(id="f200w-0040", extras={"field_id": "other-field", "source_index": 40}),
        SimpleNamespace(id="f200w-0040b", extras={"field_id": "nexus-qdr", "source_index": 40}),
        SimpleNamespace(id="f200w-0042", extras={"field_id": "nexus-qdr", "source_index": 42}),
    ])
    return tmp_path / "plates", loads


def test_render_writes_tile_plates_a_sheet_and_provenance(plates):
    root, loads = plates
    ticks = []
    record = nexus_plates.render_plates("40,42", band="VIS", model="rbf", tag="m169-188",
                                        progress=lambda *a: ticks.append(a))
    run = root / "m169-188"
    assert sorted(path.name for path in run.iterdir()) == [
        "nexus_tile040_VIS__rbf.png", "nexus_tile042_VIS__rbf.png",
        "nexus_tiles_VIS__rbf.png", "plates.json",
    ]
    with Image.open(run / "nexus_tile040_VIS__rbf.png") as image:
        assert image.size == (2700, 980)
    # every panel came from the real collection with this model spec
    assert {tier for _i, tier, _p in loads} == {"lr", "m:rbf", "jwst"}
    assert all(p == {"source": "nexus", "models": "rbf"} for _i, _t, p in loads)
    assert record["model"] == "rbf" and record["model_label"] == "RBF · minibatched"
    assert record["field_id"] == "nexus-field-1" and record["filter"] == "F200W"
    assert [tile["index"] for tile in record["tiles"]] == [40, 42]
    assert record["tiles"][0]["ref"] == "nexus/f200w-0040" and record["tiles"][0]["legacy"] is True
    assert ticks[-1][0] == ticks[-1][1]
    manifest = json.loads((run / "plates.json").read_text())
    assert len(manifest["renders"]) == 1
    # a second render of the same (band, model) replaces its record; another band adds one
    nexus_plates.render_plates([40], band="VIS", model="rbf", tag="m169-188")
    nexus_plates.render_plates(["nexus/f200w-0042"], band="temp", model="rbf", tag="m169-188")
    manifest = json.loads((run / "plates.json").read_text())
    assert sorted((item["band"], len(item["tiles"])) for item in manifest["renders"]) == [("VIS", 1), ("temp", 1)]
    with Image.open(run / "nexus_tile042_temp__rbf.png") as image:
        assert image.mode in ("RGB", "RGBA")


def test_request_validation(plates):
    with pytest.raises(nexus_plates.PlateError, match="has not been run on f200w-0042") as error:
        nexus_plates.resolve_request("40,42", "production")
    assert error.value.code == 400 and error.value.extra["missing"] == ["f200w-0042"]
    with pytest.raises(nexus_plates.PlateError, match="unknown NEXUS tile") as error:
        nexus_plates.resolve_request("999", "rbf")
    assert error.value.code == 404
    with pytest.raises(nexus_plates.PlateError, match="at least one"):
        nexus_plates.resolve_request(" , ", "rbf")
    with pytest.raises(nexus_plates.PlateError, match="unknown model spec"):
        nexus_plates.resolve_request("40", "bogus")
    with pytest.raises(nexus_plates.PlateError, match="band must be"):
        nexus_plates.check_band("F200W")
    with pytest.raises(nexus_plates.PlateError, match="tag must be"):
        nexus_plates.check_tag("../x")
    assert nexus_plates.check_band("vis") == "VIS"
    spec, tiles, _meta = nexus_plates.resolve_request(["40", "f200w-0040", "nexus/f200w-0042"], "member:rbf".replace("member:", ""))
    assert spec == "rbf" and [tile.id for tile in tiles] == ["f200w-0040", "f200w-0042"]
    assert nexus_plates.model_short("member:member_196") == "member 196"
    assert nexus_plates.model_short("gate:v1") == "gate v1"
    assert nexus_plates.default_tag("gate:26m").startswith("gate-26m-")


def test_list_runs_reads_new_and_legacy_runs(plates):
    root, _loads = plates
    nexus_plates.render_plates("40", band="VIS", model="rbf", tag="new-run")
    legacy = root / "m169-188"
    legacy.mkdir(parents=True)
    for name in ("nexus_tile040_VIS.png", "nexus_tile040_temp.png", "nexus_tiles_temp.png"):
        Image.new("RGB", (8, 4)).save(legacy / name)
    (legacy / "notes.txt").write_text("ignored")
    (legacy / "provenance.json").write_text(json.dumps({
        "field_id": "nexus-qdr", "band": "temp",
        "tiles": [{"index": 40, "ra_deg": 268.47, "dec_deg": 65.14,
                   "inference": {"combiner_label": "minibatched convex all-asinh RBF"}}],
    }))
    listing = nexus_plates.list_runs()
    assert listing["bands"] == list(nexus_plates.BANDS)
    tags = {run["tag"]: run for run in listing["runs"]}
    assert set(tags) == {"new-run", "m169-188"}
    old = tags["m169-188"]
    assert {item["name"] for item in old["files"]} == {
        "nexus_tile040_VIS.png", "nexus_tile040_temp.png", "nexus_tiles_temp.png"}
    renders = {render["band"]: render for render in old["renders"]}
    assert set(renders) == {"VIS", "temp"}
    assert renders["temp"]["legacy"] is True and renders["temp"]["sheet"] == "nexus_tiles_temp.png"
    assert renders["VIS"]["sheet"] is None
    assert renders["temp"]["tiles"][0]["ra_deg"] == 268.47
    assert renders["temp"]["model_label"] == "minibatched convex all-asinh RBF"
    # legacy tiles resolve to their real tile (the provenance field wins a tie)
    assert renders["temp"]["tiles"][0]["id"] == "f200w-0040b"
    assert renders["temp"]["tiles"][0]["ref"] == "nexus/f200w-0040b"
    new = tags["new-run"]
    assert new["renders"][0]["model"] == "rbf" and new["renders"][0]["sheet"] == "nexus_tiles_VIS__rbf.png"


def test_legacy_tile_ids_survive_a_missing_catalogue(plates, monkeypatch):
    root, _loads = plates
    legacy = root / "old"
    legacy.mkdir(parents=True)
    Image.new("RGB", (8, 4)).save(legacy / "nexus_tile042_VIS.png")
    Image.new("RGB", (8, 4)).save(legacy / "nexus_tile099_VIS.png")
    tiles = nexus_plates.list_runs()["runs"][0]["renders"][0]["tiles"]
    # no provenance field: the only tile 42 is used; an unknown tile stays unresolved
    assert [(tile["index"], tile["id"], tile["ref"]) for tile in tiles] == [
        (42, "f200w-0042", "nexus/f200w-0042"), (99, None, None)]

    def boom(source):
        raise nexus_plates.real_tiles.RealTileError(503, "no NEXUS cache")

    monkeypatch.setattr(nexus_plates.real_tiles, "list_entries", boom)
    tiles = nexus_plates.list_runs()["runs"][0]["renders"][0]["tiles"]
    assert [tile["id"] for tile in tiles] == [None, None]


def test_contact_sheet_stays_bounded():
    assert nexus_plates.sheet_dpi(1) == nexus_plates.SHEET_DPI
    assert nexus_plates.sheet_dpi(4) == nexus_plates.SHEET_DPI
    for rows in (13, 24):
        dpi = nexus_plates.sheet_dpi(rows)
        assert dpi < nexus_plates.SHEET_DPI
        # the whole canvas stays within the pixel budget (a 24-tile sheet was ~36 Mpx)
        assert nexus_plates.SHEET_WIDTH_IN * dpi * nexus_plates.SHEET_ROW_IN * rows * dpi <= nexus_plates.SHEET_MAX_PIXELS
    # a sheet panel is display-ready uint8, never larger than its slot on the sheet
    grey = np.linspace(0, 1, 850 * 850, dtype=np.float32).reshape(850, 850)
    image, style = nexus_plates.sheet_panel(grey, {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}, 300)
    assert image.dtype == np.uint8 and image.shape == (300, 300)
    assert style == {"cmap": "gray", "vmin": 0, "vmax": 255}
    rgb = np.ones((64, 64, 3), dtype=np.float64) * 0.5
    image, style = nexus_plates.sheet_panel(rgb, {}, 300)
    assert image.dtype == np.uint8 and image.shape == (64, 64, 3) and style == {}
    assert int(image[0, 0, 0]) == 128


def test_files_are_jailed_and_thumbnails_shrink(plates):
    root, _loads = plates
    nexus_plates.render_plates("40", band="VIS", model="rbf", tag="run")
    path = nexus_plates.run_file("run", "nexus_tile040_VIS__rbf.png")
    assert path.is_file()
    for tag, name in (("run", "plates.json"), ("run", "../x.png"), ("..", "nexus_tile040_VIS__rbf.png"),
                      ("missing", "nexus_tile040_VIS__rbf.png")):
        with pytest.raises(nexus_plates.PlateError):
            nexus_plates.run_file(tag, name)
    body = nexus_plates.thumbnail(path, 320)
    with Image.open(io.BytesIO(body)) as image:
        assert image.format == "JPEG" and max(image.size) == 320
    assert nexus_plates.thumbnail(path, 320) is body          # memoised


@pytest.fixture
def client(plates, monkeypatch):
    def run_now(label, target, kind=None):
        cap = SimpleNamespace(tick=lambda *a: None)
        run_now.result = target(cap)
        run_now.calls.append((label, kind))
        return "job12345"

    run_now.calls = []
    monkeypatch.setattr(jobs.REGISTRY, "spawn", run_now)
    app = Flask(__name__)
    app.config.update(TESTING=True)
    figures.register(app)
    return app.test_client(), run_now


def test_endpoints_render_list_serve_and_delete(client):
    http, spawn = client
    response = http.post("/api/figures/nexus-plates", data={
        "tiles": "40,42", "band": "VIS", "model": "rbf", "tag": "web-run"})
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body == {"ok": True, "job_id": "job12345", "tag": "web-run", "band": "VIS",
                    "model": "rbf", "tiles": ["f200w-0040", "f200w-0042"]}
    assert spawn.calls[0][1] == "figure-nexus-plates"
    assert spawn.result["sheet"] == "nexus_tiles_VIS__rbf.png"

    listing = http.get("/api/figures/nexus-plates").get_json()
    assert [run["tag"] for run in listing["runs"]] == ["web-run"]

    png = http.get("/api/figures/nexus-plates/web-run/nexus_tiles_VIS__rbf.png")
    assert png.status_code == 200 and png.mimetype == "image/png"
    thumb = http.get("/api/figures/nexus-plates/web-run/nexus_tiles_VIS__rbf.png?thumb=200")
    assert thumb.status_code == 200 and thumb.mimetype == "image/jpeg"
    assert http.get("/api/figures/nexus-plates/web-run/nexus_tiles_VIS__rbf.png?thumb=x").status_code == 400
    download = http.get("/api/figures/nexus-plates/web-run/nexus_tiles_VIS__rbf.png?download=1")
    assert download.headers["Content-Disposition"].startswith("attachment")
    assert http.get("/api/figures/nexus-plates/web-run/plates.json").status_code == 404

    assert http.post("/api/figures/nexus-plates/web-run/delete").get_json() == {"ok": True, "tag": "web-run"}
    assert http.get("/api/figures/nexus-plates").get_json()["runs"] == []
    assert http.post("/api/figures/nexus-plates/web-run/delete").status_code == 404


def test_endpoint_validation_is_synchronous(client):
    http, spawn = client
    response = http.post("/api/figures/nexus-plates", data={"tiles": "40,42", "model": "production"})
    assert response.status_code == 400
    assert response.get_json()["missing"] == ["f200w-0042"]
    assert http.post("/api/figures/nexus-plates", data={"tiles": "40", "band": "F200W"}).status_code == 400
    assert http.post("/api/figures/nexus-plates", data={"tiles": "40", "model": "rbf", "tag": ".x"}).status_code == 400
    assert http.get("/api/figures/nope").get_json()["error"]
    assert spawn.calls == []
    # a default tag is the model slug + date
    ok = http.post("/api/figures/nexus-plates", json={"tiles": "40", "model": "rbf"}).get_json()
    assert ok["tag"].startswith("rbf-")
