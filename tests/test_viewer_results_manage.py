"""Saved viewer results as figure-workspace entities: rename, delete, grid
layouts, thumbnails, FITS download, WCS-matched crops that keep their WCS,
and saves from the C9 ``real`` collection (W-Figures)."""
from __future__ import annotations

import io
import json
import math

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from flask import Flask
from PIL import Image

from euclid_polish.web.helpers import viewer_data, viewer_results
from euclid_polish.web.routes import viewer

LR_SIDE = 40
LR_SCALE = 0.1          # arcsec / px
JWST_SIDE = 120
JWST_SCALE = 0.03


def _cube(side: int, channel_values: tuple[float, ...]) -> np.ndarray:
    y, x = np.mgrid[:side, :side]
    return np.stack(
        [value + 0.1 * y + 0.01 * x for value in channel_values], axis=-1,
    ).astype(np.float32)


def _wcs(side: int, scale_arcsec: float, *, crval=(268.4, 65.1), shift_px=(0.0, 0.0)) -> dict:
    """North-up TAN WCS keywords (CD form) with the reference at the centre
    (plus a sub-footprint shift, in pixels, to mimic NEXUS's JWST offset)."""
    cd = scale_arcsec / 3600.0
    return {
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRVAL1": crval[0], "CRVAL2": crval[1],
        "CRPIX1": side / 2 + 0.5 + shift_px[0], "CRPIX2": side / 2 + 0.5 + shift_px[1],
        "CD1_1": -cd, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": cd,
    }


def _sr_wcs(lr: dict) -> dict:
    out = dict(lr)
    for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
        out[key] = lr[key] / 2
    out["CRPIX1"] = 2 * lr["CRPIX1"] - 0.5
    out["CRPIX2"] = 2 * lr["CRPIX2"] - 0.5
    return out


LR_WCS = _wcs(LR_SIDE, LR_SCALE)
# The JWST footprint is offset by 7 JWST px in x and -5 in y: a normalised
# centre would put it on a different piece of sky.
JWST_WCS = _wcs(JWST_SIDE, JWST_SCALE, shift_px=(7.0, -5.0))


@pytest.fixture
def client(tmp_path, monkeypatch):
    root = tmp_path / "viewer-results"
    monkeypatch.setenv("EUCLID_POLISH_RESULTS_DIR", str(root))
    cubes = {
        "lr": (_cube(LR_SIDE, (100.0, 200.0, 300.0, 400.0)), LR_SCALE, LR_WCS, ["VIS", "Y_E", "J_E", "H_E"]),
        "m:production": (_cube(2 * LR_SIDE, (110.0, 210.0, 310.0, 410.0)), LR_SCALE / 2,
                         _sr_wcs(LR_WCS), ["VIS", "Y_E", "J_E", "H_E"]),
        "m:mean": (_cube(2 * LR_SIDE, (111.0, 211.0, 311.0, 411.0)), LR_SCALE / 2,
                   _sr_wcs(LR_WCS), ["VIS", "Y_E", "J_E", "H_E"]),
        "jwst": (_cube(JWST_SIDE, (5.0,)), JWST_SCALE, JWST_WCS, ["F200W"]),
    }
    synthetic = {
        "LR": (_cube(12, (100.0, 200.0, 300.0, 400.0)), 0.2),
        "SR": (_cube(24, (110.0, 210.0, 310.0, 410.0)), 0.1),
    }

    def fake_meta(collection, params):
        if collection == "real":
            assert params.get("source") == "nexus"
            tiers = [{"key": "lr"}, {"key": "jwst"}, {"key": "m:production"}, {"key": "m:mean"}]
            return {
                "count": 1, "tiers": tiers, "default_tier": "lr",
                "band_names": ["VIS", "Y_E", "J_E", "H_E"],
                "objects": [{
                    "id": "f200w-0040", "label": "NEXUS F200W tile 0040", "ra": 268.4, "dec": 65.1,
                    "ref": "nexus/f200w-0040", "field": "EDF-N",
                    "tiers": ["lr", "jwst", "m:production", "m:mean"],
                }],
            }
        if collection == "evaluation":
            return {
                "count": 1, "tiers": [{"key": "LR"}, {"key": "SR"}], "default_tier": "SR",
                "band_names": ["VIS", "Y_E", "J_E", "H_E"],
                "objects": [{"id": "syn-lens-7", "label": "synthetic lens 7", "tiers": ["LR", "SR"]}],
            }
        raise viewer_data.ViewerError(404, "unknown collection")

    def fake_cube(collection, index, tier, params):
        assert index == 0
        if collection == "evaluation":
            cube, scale = synthetic[tier]
            return cube.copy(), {"label": tier, "pixscale": scale,
                                 "bands": ["VIS", "Y_E", "J_E", "H_E"]}
        cube, scale, wcs, bands = cubes[tier]
        info = {"label": f"{tier} label", "pixscale": scale, "bands": bands, "wcs": dict(wcs),
                "unit": "MJy/sr" if tier == "jwst" else "e-"}
        if tier == "jwst":
            info["transfer_group"] = "jwst"
        return cube.copy(), info

    monkeypatch.setattr(viewer_results.viewer_data, "get_meta", fake_meta)
    monkeypatch.setattr(viewer_results.viewer_data, "get_cube", fake_cube)
    app = Flask(__name__)
    app.config.update(TESTING=True)
    viewer.register(app)
    return app.test_client(), root, cubes


def _real_payload(tiers=("lr", "m:production", "jwst"), u=0.5, v=0.5, side=1.2, **selection) -> dict:
    return {
        "collection": "real", "index": 0, "tiers": list(tiers),
        "params": {"source": "nexus", "models": "production,mean"},
        "selection": {"u": u, "v": v, "angular_side_arcsec": side, **selection},
        "display": {"color": "VIS", "knee": 100.0, "gain": 1.0},
    }


def _synthetic_payload() -> dict:
    return {
        "collection": "evaluation", "index": 0, "tiers": ["LR", "SR"], "params": {},
        "selection": {"u": 0.5, "v": 0.5, "angular_side_arcsec": 1.0},
        "display": {"color": "VIS", "knee": 100.0, "gain": 1.0},
    }


def _save(client, payload) -> dict:
    response = client.post("/viewer/results", json=payload)
    assert response.status_code == 201, response.get_json()
    return response.get_json()["result"]


def _world(keywords: dict, x0: int, y0: int, side: int) -> tuple[float, float]:
    """Sky position of a crop centre (continuous crop centre → 0-based px)."""
    w = WCS(fits.Header(keywords))
    ra, dec = w.pixel_to_world_values(x0 + side / 2 - 0.5, y0 + side / 2 - 0.5)
    return float(ra), float(dec)


# ---------------------------------------------------------------------------
# real collection + WCS
# ---------------------------------------------------------------------------

def test_real_collection_save_maps_model_tiers_and_keeps_object_identity(client):
    http, _root, _cubes = client
    result = _save(http, _real_payload())
    assert result["regime"] == "real"
    assert result["logical_tiers"] == ["dirty", "sr", "jwst"]
    assert result["source"]["collection"] == "real"
    assert result["source"]["params"] == {"source": "nexus", "models": "production,mean"}
    obj = result["source"]["object"]
    assert obj["id"] == "f200w-0040" and obj["ref"] == "nexus/f200w-0040"
    assert obj["ra"] == pytest.approx(268.4)
    assert result["files"]["sr"]["source_tier"] == "m:production"
    assert result["files"]["sr"]["source_label"] == "m:production label"
    assert "jwst:native" in result["recipes"]


def test_two_model_tiers_cannot_both_be_the_sr_panel(client):
    http, _root, _cubes = client
    response = http.post("/viewer/results", json=_real_payload(tiers=("lr", "m:production", "m:mean")))
    assert response.status_code == 400
    assert "logical tier sr" in response.get_json()["error"]


def test_models_param_is_validated(client):
    http, _root, _cubes = client
    bad = _real_payload()
    bad["params"]["models"] = "production,../../x"
    assert http.post("/viewer/results", json=bad).status_code == 400


def test_crops_are_matched_through_each_tiers_wcs_and_keep_it(client):
    http, root, cubes = client
    result = _save(http, _real_payload(u=0.4, v=0.6, side=1.2))
    assert result["wcs_preserved"] is True
    assert result["wcs_tiers"] == ["dirty", "sr", "jwst"]
    manifest = viewer_results.get_result(result["id"])
    centres = {}
    for logical, source in (("dirty", "lr"), ("sr", "m:production"), ("jwst", "jwst")):
        bounds = manifest["bounds"][logical]
        centres[logical] = _world(cubes[source][2], bounds["x0"], bounds["y0"], bounds["side_pixels"])
        # the saved FITS carries the crop's own WCS: its centre is the same sky
        path = root / result["id"] / manifest["files"][logical]["filename"]
        header = fits.getheader(path)
        assert header["WCSKEEP"] is True
        side = bounds["side_pixels"]
        w = WCS(header).celestial
        ra, dec = w.pixel_to_world_values(side / 2 - 0.5, side / 2 - 0.5)
        assert (float(ra), float(dec)) == pytest.approx(centres[logical], abs=1e-9)
    # every tier's crop is centred on the same sky within half a pixel of its grid
    ref_ra, ref_dec = centres["dirty"]
    for logical, scale in (("sr", LR_SCALE / 2), ("jwst", JWST_SCALE)):
        ra, dec = centres[logical]
        sep = math.hypot((ra - ref_ra) * math.cos(math.radians(ref_dec)), dec - ref_dec) * 3600
        assert sep <= 0.75 * scale + 1e-9, (logical, sep)
    assert result["center"]["ra"] == pytest.approx(ref_ra, abs=LR_SCALE / 3600)
    assert result["center"]["dec"] == pytest.approx(ref_dec, abs=LR_SCALE / 3600)


def test_the_normalised_centre_would_miss_the_offset_jwst_footprint(client):
    """Guard for the test above: without WCS matching, the JWST crop at the
    same normalised (u, v) is ~7 JWST px away from the LR crop's sky."""
    _http, _root, cubes = client
    side = 40
    u, v = 0.4, 0.6
    x0 = round(u * JWST_SIDE - side / 2)
    y0 = round(v * JWST_SIDE - side / 2)
    ra, dec = _world(cubes["jwst"][2], x0, y0, side)
    lr_x0 = round(u * LR_SIDE - 6)
    lr_y0 = round(v * LR_SIDE - 6)
    lr_ra, lr_dec = _world(cubes["lr"][2], lr_x0, lr_y0, 12)
    sep = math.hypot((ra - lr_ra) * math.cos(math.radians(lr_dec)), dec - lr_dec) * 3600
    assert sep > 5 * JWST_SCALE


def test_selection_source_tier_anchors_the_centre(client):
    """A selection made on the JWST frame (``source_tier``) is centred on
    that frame's point; the LR crop follows it through the WCS."""
    http, _root, cubes = client
    result = _save(http, _real_payload(tiers=("lr", "jwst"), u=0.5, v=0.5, side=1.2, source_tier="jwst"))
    manifest = viewer_results.get_result(result["id"])
    jb = manifest["bounds"]["jwst"]
    assert (jb["x0"], jb["y0"]) == (40, 40)                   # 0.5·120 − 20, the frame's own centre
    lb = manifest["bounds"]["dirty"]
    jra, jdec = _world(cubes["jwst"][2], jb["x0"], jb["y0"], jb["side_pixels"])
    lra, ldec = _world(cubes["lr"][2], lb["x0"], lb["y0"], lb["side_pixels"])
    sep = math.hypot((jra - lra) * math.cos(math.radians(ldec)), jdec - ldec) * 3600
    assert sep <= 0.75 * LR_SCALE + 1e-9                     # ≤ half a pixel per axis (rounding)
    # a normalised centre would have put the LR crop at (14, 14)
    assert (lb["x0"], lb["y0"]) != (14, 14)


def test_synthetic_results_have_no_wcs(client):
    http, _root, _cubes = client
    result = _save(http, _synthetic_payload())
    assert result["wcs_preserved"] is False
    assert result["wcs_tiers"] == []
    assert result["center"] is None
    assert result["source"]["object"]["id"] == "syn-lens-7"


# ---------------------------------------------------------------------------
# rename / delete
# ---------------------------------------------------------------------------

def test_rename_sets_and_resets_the_label(client):
    http, root, _cubes = client
    result = _save(http, _synthetic_payload())
    assert result["label"] == "synthetic lens 7"
    assert result["default_label"] == "synthetic lens 7"
    response = http.post(f"/viewer/results/{result['id']}/rename", json={"label": "  Lens A — core  "})
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["ok"] is True and body["result"]["label"] == "Lens A — core"
    listing = http.get("/viewer/results").get_json()["results"]
    assert listing[0]["label"] == "Lens A — core"
    # the id is content-addressed and does not change with the label
    assert listing[0]["id"] == result["id"]
    assert (root / result["id"] / "manifest.json").is_file()
    # form bodies work too; an empty label restores the default
    response = http.post(f"/viewer/results/{result['id']}/rename", data={"label": ""})
    assert response.get_json()["result"]["label"] == "synthetic lens 7"
    # validation
    assert http.post(f"/viewer/results/{result['id']}/rename", json={"label": "x" * 121}).status_code == 400
    assert http.post(f"/viewer/results/{result['id']}/rename", json={"label": "a\x00b"}).status_code == 400
    assert http.post("/viewer/results/vr-000000000000000000000000/rename", json={"label": "a"}).status_code == 404


def test_delete_removes_the_bundle_and_prunes_layouts(client):
    http, root, _cubes = client
    a = _save(http, _synthetic_payload())
    b = _save(http, _real_payload())
    layout = http.post("/viewer/grid-layouts", json={
        "name": "mixed", "results": [a["id"], b["id"]], "rows": ["dirty:VIS"], "regime": "real",
    }).get_json()["layout"]
    response = http.post(f"/viewer/results/{a['id']}/delete")
    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "id": a["id"]}
    assert not (root / a["id"]).exists()
    assert [path.name for path in root.iterdir() if path.is_dir()] == [b["id"]]
    assert [item["id"] for item in http.get("/viewer/results").get_json()["results"]] == [b["id"]]
    layouts = http.get("/viewer/grid-layouts").get_json()["layouts"]
    assert layouts[0]["id"] == layout["id"] and layouts[0]["results"] == [b["id"]]
    assert http.post(f"/viewer/results/{a['id']}/delete").status_code == 404
    assert http.post("/viewer/results/../etc/delete").status_code == 404
    # GET is never a mutation
    assert http.get(f"/viewer/results/{b['id']}/delete").status_code in (404, 405)


# ---------------------------------------------------------------------------
# grid layouts
# ---------------------------------------------------------------------------

def test_grid_layouts_save_update_list_and_delete(client):
    http, _root, _cubes = client
    a = _save(http, _synthetic_payload())
    assert http.get("/viewer/grid-layouts").get_json() == {"layouts": []}
    first = http.post("/viewer/grid-layouts", json={
        "name": "Lens sheet", "results": [a["id"]], "rows": ["dirty:VIS", "sr:VIS_H"], "regime": "synthetic",
    })
    assert first.status_code == 201, first.get_json()
    layout = first.get_json()["layout"]
    assert layout["id"].startswith("gl-")
    assert layout["rows"] == ["dirty:VIS", "sr:VIS_H"] and layout["results"] == [a["id"]]
    # the same name (any case) overwrites that layout, keeping its id
    again = http.post("/viewer/grid-layouts", data={
        "name": "lens SHEET", "results": a["id"], "rows": "sr:VIS", "regime": "synthetic",
    })
    assert again.status_code == 200
    updated = again.get_json()["layout"]
    assert updated["id"] == layout["id"] and updated["rows"] == ["sr:VIS"] and updated["name"] == "lens SHEET"
    assert len(http.get("/viewer/grid-layouts").get_json()["layouts"]) == 1
    # validation
    assert http.post("/viewer/grid-layouts", json={"name": "", "rows": ["sr:VIS"]}).status_code == 400
    assert http.post("/viewer/grid-layouts", json={"name": "x", "rows": ["bad:VIS"]}).status_code == 400
    assert http.post("/viewer/grid-layouts", json={"name": "x", "rows": []}).status_code == 400
    assert http.post("/viewer/grid-layouts", json={
        "name": "x", "rows": ["sr:VIS"], "results": ["vr-" + "0" * 24]}).status_code == 400
    assert http.post("/viewer/grid-layouts", json={
        "name": "x", "rows": ["sr:VIS"] * 17}).status_code == 400
    # delete
    assert http.post(f"/viewer/grid-layouts/{layout['id']}/delete").get_json() == {"ok": True, "id": layout["id"]}
    assert http.get("/viewer/grid-layouts").get_json() == {"layouts": []}
    assert http.post(f"/viewer/grid-layouts/{layout['id']}/delete").status_code == 404


def test_layout_store_is_ignored_by_the_result_listing(client):
    http, root, _cubes = client
    a = _save(http, _synthetic_payload())
    http.post("/viewer/grid-layouts", json={"name": "one", "results": [a["id"]], "rows": ["sr:VIS"]})
    assert any(path.is_file() for path in root.iterdir())
    assert [item["id"] for item in http.get("/viewer/results").get_json()["results"]] == [a["id"]]


# ---------------------------------------------------------------------------
# thumbnails + FITS download + summary
# ---------------------------------------------------------------------------

def test_panel_thumbnail_size_default_recipe_and_etag(client):
    http, _root, _cubes = client
    result = _save(http, _real_payload())
    url = f"/viewer/results/{result['id']}/panel.png"
    # without tier/mode the thumbnail recipe is used (SR composite first)
    assert result["thumbnail"] == "sr:VIS_H"
    full = http.get(url)
    assert full.status_code == 200 and full.mimetype == "image/png"
    with Image.open(io.BytesIO(full.data)) as image:
        assert image.size == (24, 24)                      # 1.2″ / 0.05″
    small = http.get(url + "?size=16")
    with Image.open(io.BytesIO(small.data)) as image:
        assert image.size == (16, 16)
    # never upsampled
    big = http.get(url + "?tier=dirty&mode=VIS&size=512")
    with Image.open(io.BytesIO(big.data)) as image:
        assert image.size == (12, 12)
    etag = small.headers["ETag"]
    assert http.get(url + "?size=16", headers={"If-None-Match": etag}).status_code == 304
    assert http.get(url + "?size=abc").status_code == 400
    assert http.get(url + "?size=4").status_code == 400


def test_fits_download_and_public_file_summary(client):
    http, _root, _cubes = client
    result = _save(http, _real_payload())
    files = result["files"]
    assert set(files) == {"dirty", "sr", "jwst"}
    assert files["dirty"]["shape_hwc"] == [12, 12, 4]
    assert files["jwst"]["bands"] == ["F200W"] and files["jwst"]["wcs"] is True
    assert "sha256" not in files["dirty"] and "path" not in files["dirty"]
    assert result["bytes"] > 0
    assert result["inspect_paths"]["sr"].endswith(f"{result['id']}/sr.fits")
    response = http.get(f"/viewer/results/{result['id']}/sr.fits")
    assert response.status_code == 200
    assert response.headers["Content-Disposition"].startswith("attachment")
    with fits.open(io.BytesIO(response.data)) as hdul:
        assert hdul[0].data.shape == (4, 24, 24)
    assert http.get(f"/viewer/results/{result['id']}/hr.fits").status_code == 404
    assert http.get(f"/viewer/results/{result['id']}/evil.fits").status_code == 400


def test_old_bundles_without_new_fields_still_list(client):
    """A bundle saved before W-Figures (no object id, no WCS fields) lists."""
    http, root, _cubes = client
    result = _save(http, _synthetic_payload())
    path = root / result["id"] / "manifest.json"
    manifest = json.loads(path.read_text())
    for key in ("center", "wcs_tiers", "label"):
        manifest.pop(key, None)
    for entry in manifest["files"].values():
        entry.pop("wcs", None)
    manifest["source"]["object"].pop("id", None)
    path.write_text(json.dumps(manifest))
    listed = http.get("/viewer/results").get_json()["results"][0]
    assert listed["id"] == result["id"]
    assert listed["wcs_preserved"] is False and listed["center"] is None
    assert listed["files"]["dirty"]["wcs"] is False
