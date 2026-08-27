"""Focused tests for durable viewer-result bundles and grid rendering."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from flask import Flask
from PIL import Image
from pypdf import PdfReader

from euclid_polish.web.helpers import viewer_data, viewer_results
from euclid_polish.web.routes import viewer


def _cube(side: int, channel_values: tuple[float, ...]) -> np.ndarray:
    y, x = np.mgrid[:side, :side]
    return np.stack(
        [value + 0.1 * y + 0.01 * x for value in channel_values],
        axis=-1,
    ).astype(np.float32)


@pytest.fixture
def saved_result_client(tmp_path, monkeypatch):
    root = tmp_path / "viewer-results"
    monkeypatch.setenv("EUCLID_POLISH_RESULTS_DIR", str(root))

    cubes = {
        "LR": (_cube(12, (100.0, 200.0, 300.0, 400.0)), 0.2),
        "SR": (_cube(24, (110.0, 210.0, 310.0, 410.0)), 0.1),
        "HR": (_cube(48, (120.0, 220.0, 320.0, 420.0)), 0.05),
        "morph": (_cube(24, (1.0, 2.0, 3.0, 4.0)), 0.1),
        "lr": (_cube(12, (100.0, 200.0, 300.0, 400.0)), 0.2),
        "sr": (_cube(24, (110.0, 210.0, 310.0, 410.0)), 0.1),
        "jwst": (_cube(30, (10.0,)), 0.04),
        "jwst_colour": (_cube(30, (10.0, 20.0, 30.0)), 0.04),
    }

    def fake_meta(collection: str, params: dict[str, str]):
        if collection == "evaluation":
            tiers = [
                {"key": "LR", "label": "LR"},
                {"key": "SR", "label": "SR"},
                {"key": "HR", "label": "HR"},
                {"key": "morph", "label": "movie"},
            ]
            return {
                "collection": collection,
                "count": 1,
                "tiers": tiers,
                "default_tier": "SR",
                "band_names": ["VIS", "Y_E", "J_E", "H_E"],
                "objects": [{
                    "label": "synthetic lens 7",
                    "subdir": "syn-lens-7",
                    "tiers": [item["key"] for item in tiers],
                }],
            }
        if collection == "nexus-field":
            assert params.get("field") in {"nexus-f200w", "nexus-f444w"}
            tiers = [
                {"key": "lr", "label": "Euclid"},
                {"key": "sr", "label": "SR"},
                {"key": "jwst", "label": "NEXUS F200W"},
            ]
            return {
                "collection": collection,
                "count": 1,
                "tiers": tiers,
                "default_tier": "lr",
                "band_names": ["VIS", "Y_E", "J_E", "H_E"],
                "objects": [{"label": "NEXUS tile 12", "tiers": ["lr", "sr", "jwst"]}],
            }
        if collection == "jwst-euclid":
            tiers = [
                {"key": "lr", "label": "Euclid"},
                {"key": "sr", "label": "SR"},
                {"key": "jwst", "label": "JWST"},
            ]
            return {
                "collection": collection,
                "count": 1,
                "tiers": tiers,
                "default_tier": "lr",
                "band_names": ["VIS", "Y_E", "J_E", "H_E"],
                "objects": [{
                    "label": "paired field 2",
                    "tiers": ["lr", "sr", "jwst"],
                    "jwst_bands": ["colour", "F200W"],
                }],
            }
        raise viewer_data.ViewerError(404, "unknown collection")

    def fake_cube(collection: str, index: int, tier: str, params: dict[str, str]):
        assert index == 0
        if tier == "jwst" and collection == "jwst-euclid" and (
            params.get("jwst_band") or "colour"
        ).upper() != "F200W":
            cube, pixscale = cubes["jwst_colour"]
            return cube.copy(), {
                "label": "derived JWST colour",
                "pixscale": pixscale,
                "bands": ["JWST-R", "JWST-G", "JWST-B"],
                "display_scale": 5.0,
                "direct_rgb": True,
                "transfer_group": "jwst",
            }
        cube, pixscale = cubes[tier]
        if tier == "jwst":
            band = "F444W" if params.get("field") == "nexus-f444w" else "F200W"
            return cube.copy(), {
                "label": f"NEXUS {band}",
                "pixscale": pixscale,
                "bands": [band],
                "display_scale": 10.0,
                "transfer_group": "jwst",
            }
        info = {
            "label": tier,
            "pixscale": pixscale,
            "bands": ["VIS", "Y_E", "J_E", "H_E"],
            "transfer_group": "euclid",
        }
        if collection in {"nexus-field", "jwst-euclid"}:
            info["display_scale"] = 2.0 if tier == "lr" else 4.0
        return cube.copy(), info

    monkeypatch.setattr(viewer_results.viewer_data, "get_meta", fake_meta)
    monkeypatch.setattr(viewer_results.viewer_data, "get_cube", fake_cube)

    app = Flask(__name__)
    app.config.update(TESTING=True)
    viewer.register(app)
    return app.test_client(), root, cubes


def _synthetic_payload() -> dict:
    return {
        "collection": "evaluation",
        "index": 0,
        "tiers": ["LR", "SR", "HR"],
        "params": {"bhr_fwhm_arcsec": "0.066"},
        "selection": {
            "u": 0.5,
            "v": 0.5,
            "angular_side_arcsec": 1.0,
            "relative_side": 0.25,
            "revision": 3,
        },
        "display": {
            "color": "VIS_H",
            "knee": 125.0,
            "gain": 1.2,
            "transfers": {"euclid": {"knee": 125.0, "gain": 1.2}},
        },
    }


def _post_result(client, payload: dict | None = None) -> dict:
    response = client.post("/viewer/results", json=payload or _synthetic_payload())
    assert response.status_code == 201, response.get_json()
    body = response.get_json()
    assert body["id"] == body["result_id"] == body["result"]["id"]
    assert response.headers["Location"] == f"/viewer/results/{body['id']}"
    return body


def test_mixed_resolution_crop_is_raw_hashed_and_atomic(saved_result_client):
    client, root, cubes = saved_result_client
    body = _post_result(client)
    result_id = body["id"]
    manifest_path = root / result_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["complete"] is True
    assert manifest["logical_tiers"] == ["dirty", "sr", "hr"]
    assert manifest["wcs_preserved"] is False
    assert manifest["display"]["applied_to_fits"] is False
    assert manifest["display"]["knee"] == 125.0
    assert manifest["source"]["regime"] == "synthetic"
    assert manifest["selection"]["angular_side_arcsec"] == 1.0
    assert {tier: bounds["side_pixels"] for tier, bounds in manifest["bounds"].items()} == {
        "dirty": 5,
        "sr": 10,
        "hr": 20,
    }
    assert all(
        bounds["actual_angular_side_arcsec"] == pytest.approx(1.0)
        for bounds in manifest["bounds"].values()
    )

    source_key = {"dirty": "LR", "sr": "SR", "hr": "HR"}
    for logical, key in source_key.items():
        entry = manifest["files"][logical]
        path = root / result_id / entry["filename"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        saved = np.moveaxis(np.asarray(fits.getdata(path)), 0, -1)
        bounds = manifest["bounds"][logical]
        expected = cubes[key][0][
            bounds["y0"]:bounds["y1"], bounds["x0"]:bounds["x1"], :
        ]
        np.testing.assert_array_equal(saved, expected)
        header = fits.getheader(path)
        assert header["DSPAPPL"] is False
        assert header["WCSKEEP"] is False

    # The content-addressed ID is stable and an identical form submission is
    # idempotent rather than creating a second bundle.
    payload = _synthetic_payload()
    form = {
        "collection": payload["collection"],
        "index": str(payload["index"]),
        "tiers": json.dumps(payload["tiers"]),
        "params": json.dumps(payload["params"]),
        "selection": json.dumps(payload["selection"]),
        "display": json.dumps(payload["display"]),
    }
    repeated = client.post("/viewer/results", data=form)
    assert repeated.status_code == 201
    assert repeated.get_json()["id"] == result_id
    assert [path.name for path in root.iterdir()] == [result_id]


def test_submission_validation_and_path_safety(saved_result_client, monkeypatch):
    client, root, _cubes = saved_result_client

    duplicate = _synthetic_payload()
    duplicate["tiers"] = ["LR", "lr"]
    assert client.post("/viewer/results", json=duplicate).status_code == 400

    morph = _synthetic_payload()
    morph["tiers"] = ["morph"]
    response = client.post("/viewer/results", json=morph)
    assert response.status_code == 400
    assert "morph" in response.get_json()["error"]

    path_like = {
        **_synthetic_payload(),
        "collection": "nexus-field",
        "tiers": ["lr"],
        "params": {"field": "../../secret"},
    }
    assert client.post("/viewer/results", json=path_like).status_code == 400

    outside = _synthetic_payload()
    outside["selection"] = {"u": 1.01, "v": 0.5, "angularSide": 1.0}
    assert client.post("/viewer/results", json=outside).status_code == 400

    edge = _synthetic_payload()
    edge["selection"] = {"u": 0.05, "v": 0.5, "angularSide": 1.0}
    response = client.post("/viewer/results", json=edge)
    assert response.status_code == 422
    assert "matched crop" in response.get_json()["error"]

    huge = _synthetic_payload()
    huge["selection"] = {"u": 0.5, "v": 0.5, "angularSide": 121.0}
    assert client.post("/viewer/results", json=huge).status_code == 400

    relative = _synthetic_payload()
    relative["selection"] = {"u": 0.5, "v": 0.5, "relativeSide": 0.25}
    assert client.post("/viewer/results", json=relative).status_code == 400
    relative["selection"]["relativeFallbackSafe"] = True
    assert client.post("/viewer/results", json=relative).status_code == 201

    nonfinite = _synthetic_payload()
    nonfinite["selection"] = {"u": float("nan"), "v": 0.5, "angularSide": 1.0}
    with pytest.raises(viewer_results.ViewerResultError, match="finite") as error:
        viewer_results.save_result(nonfinite)
    assert error.value.code == 400

    original_get_cube = viewer_results.viewer_data.get_cube

    def zero_pixscale(*args, **kwargs):
        cube, info = original_get_cube(*args, **kwargs)
        return cube, {**info, "pixscale": 0.0}

    monkeypatch.setattr(viewer_results.viewer_data, "get_cube", zero_pixscale)
    response = client.post("/viewer/results", json=_synthetic_payload())
    assert response.status_code == 415
    assert "pixel scale" in response.get_json()["error"]

    assert client.get("/viewer/results/not-a-result/panel.png?tier=dirty&mode=VIS").status_code == 404
    assert client.get("/viewer/results/grid.png?result=../escape&row=dirty:VIS").status_code == 404
    assert not (root.parent / "escape").exists()


def test_partial_write_never_publishes_bundle(saved_result_client, monkeypatch):
    client, root, _cubes = saved_result_client
    original = viewer_results._write_fits
    writes = 0

    def fail_second(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("simulated disk failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(viewer_results, "_write_fits", fail_second)
    response = client.post("/viewer/results", json=_synthetic_payload())
    assert response.status_code == 500
    assert response.get_json() == {"error": "could not save the viewer result"}
    assert root.is_dir()
    assert list(root.iterdir()) == []
    assert client.get("/viewer/results").get_json()["results"] == []


def test_list_panel_and_a4_five_by_three_grid(saved_result_client, tmp_path):
    client, root, _cubes = saved_result_client
    result_id = _post_result(client)["id"]

    listing = client.get("/viewer/results")
    assert listing.status_code == 200
    payload = listing.get_json()
    assert payload["axis_defaults"] == {"columns": "results", "rows": "recipes"}
    assert payload["limits"] == {"max_results": 12, "max_rows": 16}
    assert payload["supported"]["transfer"] == {
        "kind": "asinh_absolute",
        "knee_e": 100.0,
        "white_e": 3000.0,
    }
    summary = payload["results"][0]
    assert summary["id"] == result_id
    assert summary["regime"] == "synthetic"
    assert all(isinstance(recipe, str) and ":" in recipe for recipe in summary["recipes"])
    assert "dirty:VIS_H" in summary["recipes"]
    labels = {option["key"]: option["label"] for option in summary["recipe_options"]}
    assert labels["dirty:VIS"] == "VIS Dirty"
    assert labels["sr:H_E"] == "H_E SR"
    assert labels["hr:VIS_H"] == "VIS + H_E HR"

    detail = client.get(f"/viewer/results/{result_id}")
    assert detail.status_code == 200
    assert detail.get_json()["result"] == summary

    panel = client.get(f"/viewer/results/{result_id}/panel.png?tier=dirty&mode=VIS_H")
    assert panel.status_code == 200
    assert panel.mimetype == "image/png"
    panel_path = tmp_path / "panel.png"
    panel_path.write_bytes(panel.data)
    with Image.open(panel_path) as image:
        assert image.size == (5, 5)
        assert image.mode == "RGB"

    query = [
        *(('result', result_id) for _ in range(3)),
        ("row", "dirty:VIS"),
        ("row", "dirty:H_E"),
        ("row", "dirty:VIS_H"),
        ("row", "sr:VIS"),
        ("row", "sr:VIS_H"),
        ("dpi", "120"),
    ]
    png = client.get("/viewer/results/grid.png", query_string=query)
    assert png.status_code == 200
    assert png.mimetype == "image/png"
    grid_png = tmp_path / "grid.png"
    grid_png.write_bytes(png.data)
    with Image.open(grid_png) as image:
        assert image.height > image.width
        assert image.size == (992, 1403)

    assert client.get(
        "/viewer/results/grid.png",
        query_string=[("result", result_id), ("row", "dirty:VIS"), ("dpi", "71")],
    ).status_code == 400

    pdf = client.get("/viewer/results/grid.pdf", query_string=query)
    assert pdf.status_code == 200
    assert pdf.mimetype == "application/pdf"
    assert pdf.data.startswith(b"%PDF")
    grid_pdf = tmp_path / "grid.pdf"
    grid_pdf.write_bytes(pdf.data)
    reader = PdfReader(grid_pdf)
    assert len(reader.pages) == 1
    media_box = reader.pages[0].mediabox
    assert float(media_box.width) == pytest.approx(210.0 / 25.4 * 72.0, abs=0.1)
    assert float(media_box.height) == pytest.approx(297.0 / 25.4 * 72.0, abs=0.1)

    # A manifest without every referenced FITS file is never listed.
    incomplete_id = "vr-" + "0" * 24
    incomplete = root / incomplete_id
    incomplete.mkdir()
    bad_manifest = json.loads((root / result_id / "manifest.json").read_text())
    bad_manifest["id"] = incomplete_id
    (incomplete / "manifest.json").write_text(json.dumps(bad_manifest))
    ids = [item["id"] for item in client.get("/viewer/results").get_json()["results"]]
    assert ids == [result_id]

    # A syntactically valid ID cannot escape the result root through a local
    # symlink, even if the target otherwise contains a complete-looking copy.
    linked_id = "vr-" + "1" * 24
    outside = tmp_path / "outside-bundle"
    shutil.copytree(root / result_id, outside)
    linked_manifest = json.loads((outside / "manifest.json").read_text())
    linked_manifest["id"] = linked_id
    (outside / "manifest.json").write_text(json.dumps(linked_manifest))
    (root / linked_id).symlink_to(outside, target_is_directory=True)
    ids = [item["id"] for item in client.get("/viewer/results").get_json()["results"]]
    assert ids == [result_id]
    assert client.get(
        f"/viewer/results/{linked_id}/panel.png?tier=dirty&mode=VIS"
    ).status_code == 404


def test_grid_geometry_budget_downsampling_and_cache(saved_result_client, monkeypatch):
    client, _root, _cubes = saved_result_client
    result_id = _post_result(client)["id"]
    manifest = viewer_results.get_result(result_id)
    recipes = [
        ("dirty", "VIS"),
        ("dirty", "H_E"),
        ("dirty", "VIS_H"),
        ("sr", "VIS"),
        ("sr", "VIS_H"),
    ]
    geometry = viewer_results._grid_geometry(5, 3, 120)

    assert geometry.gap_mm == geometry.outer_padding_mm == 4.0
    assert geometry.cell_side_pixels == round(geometry.cell_side_mm / 25.4 * 120)
    first = geometry.panel_bounds_mm(0, 0)
    right = geometry.panel_bounds_mm(0, 1)
    below = geometry.panel_bounds_mm(1, 0)
    last = geometry.panel_bounds_mm(4, 2)
    assert right[0] - (first[0] + first[2]) == pytest.approx(geometry.gap_mm)
    assert first[1] - (below[1] + below[3]) == pytest.approx(geometry.gap_mm)
    assert geometry.grid_left_mm - geometry.row_title_track_mm == pytest.approx(
        geometry.outer_padding_mm
    )
    assert last[0] + last[2] == pytest.approx(
        viewer_results.A4_WIDTH_MM - geometry.outer_padding_mm
    )
    assert last[1] == pytest.approx(geometry.outer_padding_mm)
    grid_top = first[1] + first[3]
    assert (
        viewer_results.A4_HEIGHT_MM - grid_top - geometry.column_title_track_mm
    ) == pytest.approx(geometry.outer_padding_mm)

    budget = viewer_results._grid_request_budget(
        [manifest, manifest, manifest], recipes, geometry,
    )
    assert budget["page_pixels"] == geometry.page_width_pixels * geometry.page_height_pixels
    assert budget["estimated_peak_bytes"] <= viewer_results.MAX_GRID_RENDER_BYTES

    original_render = viewer_results._render_panel_modes
    calls: list[tuple[str, tuple[str, ...], int | None]] = []

    def recording_render(manifest_arg, logical, modes, *, target_side_pixels=None):
        rendered = original_render(
            manifest_arg, logical, modes, target_side_pixels=target_side_pixels,
        )
        calls.append((logical, tuple(modes), target_side_pixels))
        assert all(
            panel.shape == (
                geometry.cell_side_pixels,
                geometry.cell_side_pixels,
                3,
            ) and panel.dtype == np.uint8
            for panel in rendered.values()
        )
        return rendered

    monkeypatch.setattr(viewer_results, "_render_panel_modes", recording_render)
    body = viewer_results.render_grid(
        [result_id, result_id, result_id],
        [f"{tier}:{mode}" for tier, mode in recipes],
        "png",
        120,
    )
    assert body.startswith(b"\x89PNG")
    assert calls == [
        ("dirty", ("VIS", "H_E", "VIS_H"), geometry.cell_side_pixels),
        ("sr", ("VIS", "VIS_H"), geometry.cell_side_pixels),
    ]

    original_output_budget = viewer_results.MAX_GRID_OUTPUT_PIXELS
    original_source_budget = viewer_results.MAX_GRID_SOURCE_BYTES
    original_memory_budget = viewer_results.MAX_GRID_RENDER_BYTES
    monkeypatch.setattr(viewer_results, "MAX_GRID_OUTPUT_PIXELS", 1)
    with pytest.raises(viewer_results.ViewerResultError, match="output-pixel budget") as error:
        viewer_results.render_grid([result_id], ["dirty:VIS"], "png", 120)
    assert error.value.code == 413
    monkeypatch.setattr(viewer_results, "MAX_GRID_OUTPUT_PIXELS", original_output_budget)
    monkeypatch.setattr(viewer_results, "MAX_GRID_SOURCE_BYTES", 1)
    with pytest.raises(viewer_results.ViewerResultError, match="request I/O budget") as error:
        viewer_results.render_grid([result_id], ["dirty:VIS"], "png", 120)
    assert error.value.code == 413
    monkeypatch.setattr(viewer_results, "MAX_GRID_SOURCE_BYTES", original_source_budget)
    monkeypatch.setattr(viewer_results, "MAX_GRID_RENDER_BYTES", 1)
    with pytest.raises(viewer_results.ViewerResultError, match="render-memory budget") as error:
        viewer_results.render_grid([result_id], ["dirty:VIS"], "png", 120)
    assert error.value.code == 413
    monkeypatch.setattr(viewer_results, "MAX_GRID_RENDER_BYTES", original_memory_budget)


def test_euclid_display_scale_applies_to_lr_sr_and_composite(saved_result_client):
    client, _root, cubes = saved_result_client
    payload = {
        "collection": "nexus-field",
        "index": 0,
        "tiers": ["lr", "sr"],
        "params": {"field": "nexus-f200w"},
        "selection": {"u": 0.5, "v": 0.5, "angular_side_arcsec": 0.4},
        "display": {"color": "VIS_H", "knee": 100.0, "gain": 1.0},
    }
    manifest = viewer_results.get_result(_post_result(client, payload)["id"])
    assert manifest["files"]["dirty"]["display_scale"] == 2.0
    assert manifest["files"]["sr"]["display_scale"] == 4.0

    dirty = viewer_results._render_panel_modes(
        manifest, "dirty", ["VIS", "VIS_H"],
    )
    sr = viewer_results._render_panel_modes(manifest, "sr", ["H_E"])

    dirty_bounds = manifest["bounds"]["dirty"]
    dy, dx = dirty_bounds["y0"], dirty_bounds["x0"]
    dirty_vis_raw = cubes["lr"][0][dy, dx, 0]
    dirty_h_raw = cubes["lr"][0][dy, dx, 3]
    vis = viewer_results._absolute_asinh(np.asarray(dirty_vis_raw), display_scale=2.0)
    h_band = viewer_results._absolute_asinh(np.asarray(dirty_h_raw), display_scale=2.0)
    expected_vis = viewer_results._rgb_uint8(np.repeat(vis[..., None], 3, axis=-1))
    expected_composite = viewer_results._rgb_uint8(
        viewer_results._vis_h_false_colour(vis, h_band)
    )
    np.testing.assert_array_equal(dirty["VIS"][0, 0], expected_vis)
    np.testing.assert_array_equal(dirty["VIS_H"][0, 0], expected_composite)

    sr_bounds = manifest["bounds"]["sr"]
    sy, sx = sr_bounds["y0"], sr_bounds["x0"]
    sr_h_raw = cubes["sr"][0][sy, sx, 3]
    sr_h = viewer_results._absolute_asinh(np.asarray(sr_h_raw), display_scale=4.0)
    expected_sr_h = viewer_results._rgb_uint8(np.repeat(sr_h[..., None], 3, axis=-1))
    np.testing.assert_array_equal(sr["H_E"][0, 0], expected_sr_h)


def test_jwst_save_requires_exact_explicit_native_f200w(saved_result_client):
    client, _root, _cubes = saved_result_client
    base = {
        "collection": "jwst-euclid",
        "index": 0,
        "tiers": ["jwst"],
        "params": {},
        "selection": {"u": 0.5, "v": 0.5, "angular_side_arcsec": 0.4},
        "display": {"color": "F200W", "knee": 100.0, "gain": 1.0},
    }
    response = client.post("/viewer/results", json=base)
    assert response.status_code == 422
    assert "explicit native F200W" in response.get_json()["error"]

    derived = {**base, "params": {"jwst_band": "colour"}}
    response = client.post("/viewer/results", json=derived)
    assert response.status_code == 422
    assert "derived/default colour" in response.get_json()["error"]

    native = {**base, "params": {"jwst_band": "F200W"}}
    result = _post_result(client, native)["result"]
    assert result["bands"]["jwst"] == ["F200W"]
    assert result["recipes"] == ["jwst:native"]

    wrong_filter = {
        **base,
        "collection": "nexus-field",
        "params": {"field": "nexus-f444w"},
    }
    response = client.post("/viewer/results", json=wrong_filter)
    assert response.status_code == 422
    assert "other filters" in response.get_json()["error"]


def test_saved_fits_checksum_is_cached_and_tampering_is_rejected(
    saved_result_client, monkeypatch,
):
    client, root, _cubes = saved_result_client
    result_id = _post_result(client)["id"]
    manifest = viewer_results.get_result(result_id)
    path = root / result_id / manifest["files"]["dirty"]["filename"]
    viewer_results._SHA_CACHE.clear()
    original_sha = viewer_results._sha256
    calls = 0

    def counting_sha(path_arg):
        nonlocal calls
        calls += 1
        return original_sha(path_arg)

    monkeypatch.setattr(viewer_results, "_sha256", counting_sha)
    url = f"/viewer/results/{result_id}/panel.png?tier=dirty&mode=VIS"
    assert client.get(url).status_code == 200
    assert client.get(url).status_code == 200
    assert calls == 1

    with path.open("ab") as stream:
        stream.write(b"tamper")
    response = client.get(url)
    assert response.status_code == 409
    assert "checksum" in response.get_json()["error"]
    assert calls == 2
    assert client.get("/viewer/results").get_json()["results"] == []


def test_grid_recipes_are_intersection_of_every_result(saved_result_client):
    client, _root, _cubes = saved_result_client
    synthetic_id = _post_result(client)["id"]
    real_payload = {
        "collection": "nexus-field",
        "index": 0,
        "tiers": ["lr", "sr", "jwst"],
        "params": {"field": "nexus-f200w"},
        "selection": {"u": 0.5, "v": 0.5, "angular_side_arcsec": 0.4},
        "display": {"color": "VIS_H", "knee": 100.0, "gain": 1.0},
    }
    real = _post_result(client, real_payload)["result"]
    assert real["regime"] == "real"
    assert "jwst:native" in real["recipes"]
    assert {
        option["key"]: option["label"] for option in real["recipe_options"]
    }["jwst:native"] == "NEXUS F200W"

    common = [
        ("result", synthetic_id),
        ("result", real["id"]),
        ("row", "dirty:VIS"),
        ("row", "sr:VIS_H"),
        ("dpi", "120"),
    ]
    assert client.get("/viewer/results/grid.png", query_string=common).status_code == 200

    missing_from_real = [
        ("result", synthetic_id),
        ("result", real["id"]),
        ("row", "hr:VIS"),
    ]
    response = client.get("/viewer/results/grid.png", query_string=missing_from_real)
    assert response.status_code == 400
    assert "every result" in response.get_json()["error"]

    missing_from_synthetic = [
        ("result", synthetic_id),
        ("result", real["id"]),
        ("row", "jwst:native"),
    ]
    assert client.get(
        "/viewer/results/grid.png", query_string=missing_from_synthetic
    ).status_code == 400

    # With no explicit rows, defaults are also drawn only from the common
    # recipe intersection; the unmatched HR and JWST panels are not blanks.
    defaults = [("result", synthetic_id), ("result", real["id"]), ("dpi", "120")]
    assert client.get("/viewer/results/grid.png", query_string=defaults).status_code == 200


def test_only_exact_f200w_is_advertised_as_nexus_native():
    base = {
        "files": {
            "jwst": {
                "bands": ["F200W"],
                "direct_rgb": False,
            },
        },
    }
    assert [item["key"] for item in viewer_results._supported_recipes(base)] == [
        "jwst:native"
    ]
    base["files"]["jwst"]["bands"] = ["F444W"]
    assert viewer_results._supported_recipes(base) == []
    base["files"]["jwst"]["bands"] = ["F200W", "F444W"]
    assert viewer_results._supported_recipes(base) == []


def test_jwst_native_honors_saved_display_scale(saved_result_client, tmp_path):
    client, _root, cubes = saved_result_client
    payload = {
        "collection": "nexus-field",
        "index": 0,
        "tiers": ["jwst"],
        "params": {"field": "nexus-f200w"},
        "selection": {"u": 0.5, "v": 0.5, "angularSide": 0.4},
        "display": {"color": "F200W", "knee": 100.0, "gain": 1.0},
    }
    result_id = _post_result(client, payload)["id"]
    response = client.get(
        f"/viewer/results/{result_id}/panel.png?tier=jwst&mode=native"
    )
    assert response.status_code == 200
    path = tmp_path / "jwst.png"
    path.write_bytes(response.data)
    with Image.open(path) as image:
        pixel = np.asarray(image)[0, 0, 0]
    raw_first_crop_pixel = float(cubes["jwst"][0][10, 10, 0])
    expected = np.arcsinh(
        raw_first_crop_pixel * 10.0 / viewer_results.ASINH_KNEE_E
    ) / np.arcsinh(30.0)
    assert int(pixel) == pytest.approx(round(255.0 * expected), abs=1)
