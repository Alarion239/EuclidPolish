"""Viewer backend contract C6: JSON errors, object ids/positions, tier units,
per-tier celestial WCS (``X-Cube-WCS``) and multi-channel cubes.

Every real collection publishes the FITS WCS of *the served tier's* pixel grid
(FITS 1-based convention, axis 1 = column); SR grids are the LR WCS magnified
×2 (``CD/2``, ``CRPIX → 2·CRPIX − 0.5``). The round-trip checks evaluate the
served keywords at pixel (0, 0) and compare with the source FITS WCS at the
matching source pixel (1e-9 deg).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from euclid_polish.catalog.catalog_object import CatalogObject
from euclid_polish.config import Config
from euclid_polish.web import app as web_app
from euclid_polish.web.helpers import archive_fields
from euclid_polish.web.helpers import status as web_status
from euclid_polish.web.helpers import viewer_data as vd

TOL_DEG = 1e-9


def _header(*, crval=(150.1, 2.2), crpix=(4.5, 4.5), scale=0.1 / 3600,
            pc=False, bunit="electron") -> fits.Header:
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    header["CRVAL1"], header["CRVAL2"] = crval
    header["CRPIX1"], header["CRPIX2"] = crpix
    if pc:
        header["PC1_1"], header["PC2_2"] = -1.0, 1.0
        header["CDELT1"], header["CDELT2"] = scale, scale
    else:
        header["CD1_1"], header["CD1_2"] = -scale, 0.0
        header["CD2_1"], header["CD2_2"] = 0.0, scale
    if bunit:
        header["BUNIT"] = bunit
    return header


def _write(path: Path, data: np.ndarray, header: fits.Header | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.asarray(data, np.float32), header=header).writeto(path)


def _served_wcs(response) -> WCS:
    return WCS(fits.Header(json.loads(response.headers["X-Cube-WCS"])))


def _world(wcs: WCS, x: float, y: float) -> tuple[float, float]:
    ra, dec = wcs.celestial.pixel_to_world_values(x, y)
    return float(ra), float(dec)


def _assert_same_sky(a: tuple[float, float], b: tuple[float, float]) -> None:
    assert abs(a[0] - b[0]) < TOL_DEG and abs(a[1] - b[1]) < TOL_DEG, (a, b)


@pytest.fixture
def client():
    app = web_app.create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# WCS keyword helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pc", [False, True])
def test_wcs_keywords_round_trip_cd_and_pc(pc):
    header = _header(pc=pc)
    keywords = vd.celestial_wcs_keywords(header)
    assert set(keywords) == {"CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1",
                             "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"}
    for x, y in ((0, 0), (7, 3)):
        _assert_same_sky(_world(WCS(fits.Header(keywords)), x, y),
                         _world(WCS(header), x, y))


def test_wcs_keywords_of_a_cube_header_and_missing_wcs():
    header = _header()
    header["NAXIS"], header["NAXIS1"], header["NAXIS2"], header["NAXIS3"] = 3, 8, 8, 4
    assert vd.celestial_wcs_keywords(header)["CTYPE1"] == "RA---TAN"
    assert vd.celestial_wcs_keywords(fits.Header()) is None
    assert vd.celestial_wcs_keywords(None) is None


def test_shifted_and_scaled_wcs_keywords():
    source = WCS(_header())
    keywords = vd.celestial_wcs_keywords(_header())
    shifted = WCS(fits.Header(vd.shifted_wcs_keywords(keywords, dx=3, dy=5)))
    _assert_same_sky(_world(shifted, 0, 0), _world(source, 3, 5))
    scaled = WCS(fits.Header(vd.scaled_wcs_keywords(keywords, 2)))
    # SR pixels (0,0),(1,0),(0,1),(1,1) subdivide LR pixel (0,0).
    corners = [_world(scaled, x, y) for x, y in ((0, 0), (1, 0), (0, 1), (1, 1))]
    mean = (float(np.mean([c[0] for c in corners])), float(np.mean([c[1] for c in corners])))
    _assert_same_sky(mean, _world(source, 0, 0))
    assert vd.scaled_wcs_keywords(None, 2) is None


def test_unit_from_bunit():
    assert vd.unit_from_header(fits.Header({"BUNIT": "electron"})) == "e-"
    assert vd.unit_from_header(fits.Header({"BUNIT": "MJy/sr"})) == "MJy/sr"
    assert vd.unit_from_header(fits.Header({"BUNIT": "ADU/s"})) == "ADU/s"
    assert vd.unit_from_header(fits.Header(), default="e-") == "e-"


# ---------------------------------------------------------------------------
# JSON errors
# ---------------------------------------------------------------------------

def test_viewer_errors_are_json(client, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "empty"))
    meta = client.get("/viewer/meta/nope")
    assert meta.status_code == 404
    assert meta.get_json() == {"error": "unknown collection"}
    cube = client.get("/viewer/cube/evaluation/3?tier=LR")
    assert cube.status_code == 404
    assert cube.get_json() == {"error": "index out of range"}
    bad = client.get("/viewer/meta/sky?subset=bogus")
    assert bad.status_code == 400
    assert "subset" in bad.get_json()["error"]
    unrouted = client.get("/viewer/cube/sky/not-a-number")
    assert unrouted.status_code == 404
    assert "error" in unrouted.get_json()


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

@pytest.fixture
def eval_store(tmp_path, monkeypatch):
    root = tmp_path / "eval"
    monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(root))
    obj = root / "lensA"
    lr_header = _header(crval=(58.68, -51.28), crpix=(4.5, 4.5))
    _write(obj / "original_stack.fits", np.ones((4, 8, 8)), lr_header)
    _write(obj / "SR.fits", np.ones((4, 16, 16)), fits.Header({"BUNIT": "electron"}))
    _write(root / "synth" / "original_stack.fits", np.ones((4, 8, 8)))
    _write(root / "synth" / "SR.fits", np.ones((4, 16, 16)))
    _write(root / "synth" / "HR.fits", np.ones((4, 16, 16)))
    with (root / "manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "ra", "dec", "grade", "ok", "error", "out_subdir"])
        writer.writerow(["lensA", "58.68", "-51.28", "A", "True", "", "lensA"])
        writer.writerow(["s1", "", "", "syn-lens", "True", "", "synth"])
    return root, lr_header


def test_eval_meta_objects_carry_ids_positions_and_tier_units(client, eval_store):
    meta = client.get("/viewer/meta/evaluation").get_json()
    lens, synth = meta["objects"]
    assert lens["id"] == "lensA" and synth["id"] == "synth"
    assert (lens["ra"], lens["dec"]) == (58.68, -51.28)
    assert "ra" not in synth and "dec" not in synth
    units = {tier["key"]: tier.get("unit") for tier in meta["tiers"]}
    assert units["LR"] == units["SR"] == units["HR"] == "e-"


def test_objects_resolve_by_id(client, eval_store):
    """``?id=`` addresses an object by its stable meta id (spec §9.4)."""
    by_id = client.get("/viewer/cube/evaluation?id=synth&tier=HR")
    assert by_id.status_code == 200, by_id.get_json()
    by_index = client.get("/viewer/cube/evaluation/1?tier=HR")
    assert by_id.data == by_index.data
    assert by_id.headers["X-Cube-Index"] == by_index.headers["X-Cube-Index"] == "1"
    assert "X-Cube-Index" in by_id.headers["Access-Control-Expose-Headers"].split(",")
    lens = client.get("/viewer/cube/evaluation?id=lensA&tier=LR")
    assert lens.headers["X-Cube-Index"] == "0" and "X-Cube-WCS" in lens.headers

    assert client.get("/viewer/meta/evaluation?id=synth").get_json()["index"] == 1
    assert "index" not in client.get("/viewer/meta/evaluation").get_json()
    for url in ("/viewer/cube/evaluation?id=nope&tier=LR",
                "/viewer/meta/evaluation?id=nope"):
        missing = client.get(url)
        assert missing.status_code == 404
        assert missing.get_json() == {"error": "unknown object id: nope"}
    no_id = client.get("/viewer/cube/evaluation?tier=LR")
    assert no_id.status_code == 400 and "id" in no_id.get_json()["error"]


def test_eval_cube_wcs_matches_the_fits_and_sr_is_magnified(client, eval_store):
    _root, lr_header = eval_store
    lr = client.get("/viewer/cube/evaluation/0?tier=LR")
    assert lr.status_code == 200
    assert lr.headers["X-Cube-Unit"] == "e-"
    exposed = lr.headers["Access-Control-Expose-Headers"].split(",")
    assert {"X-Cube-WCS", "X-Cube-Unit"} <= set(exposed)
    _assert_same_sky(_world(_served_wcs(lr), 0, 0), _world(WCS(lr_header), 0, 0))

    sr = client.get("/viewer/cube/evaluation/0?tier=SR")
    served = _served_wcs(sr)
    corners = [_world(served, x, y) for x, y in ((0, 0), (1, 0), (0, 1), (1, 1))]
    _assert_same_sky(
        (float(np.mean([c[0] for c in corners])), float(np.mean([c[1] for c in corners]))),
        _world(WCS(lr_header), 0, 0))
    # A synthetic object has no sky position.
    assert "X-Cube-WCS" not in client.get("/viewer/cube/evaluation/1?tier=HR").headers


# ---------------------------------------------------------------------------
# archive-fields
# ---------------------------------------------------------------------------

def test_archive_field_cube_serves_the_vis_hdu_wcs(client, tmp_path, monkeypatch):
    path = tmp_path / "field_0001.fits"
    vis_header = _header(crval=(61.2, -48.4), crpix=(128.5, 128.5))
    hdus = [fits.PrimaryHDU()]
    for band in vd.BAND_NAMES:
        hdus.append(fits.ImageHDU(np.zeros((4, 4), np.float32), header=vis_header, name=band))
    fits.HDUList(hdus).writeto(path)
    field = archive_fields.ArchiveField(
        sample_id=7, source_sample_id=1, parent_id="p1", field="EDF-F",
        ra=61.2, dec=-48.4, source_release="Q1_R1", source_plan_fingerprint="x",
        position_index=0, position_name="north", path=path, bundle_sha256="",
        bands={}, record={})
    monkeypatch.setattr(vd.archive_fields, "availability", lambda: {"ready": True})
    monkeypatch.setattr(vd.archive_fields, "iter_comparison_fields", lambda: iter([field]))
    monkeypatch.setattr(vd.archive_fields, "load_field",
                        lambda _field: np.zeros((256, 256, 4), np.float32))

    meta = client.get("/viewer/meta/archive-fields").get_json()
    assert meta["objects"][0]["id"] == "7"
    assert meta["tiers"][0]["unit"] == "e-"
    cube = client.get("/viewer/cube/archive-fields/0?tier=lr")
    assert cube.headers["X-Cube-Unit"] == "e-"
    _assert_same_sky(_world(_served_wcs(cube), 0, 0), _world(WCS(vis_header), 0, 0))


def test_archive_field_labels_come_from_the_position(client, tmp_path, monkeypatch):
    """EDF-F/EDF-S were swapped in the stored label; the payload exposes the
    position-derived ``field`` and keeps ``stored_field``."""
    field = archive_fields.ArchiveField(
        sample_id=3, source_sample_id=0, parent_id="p", field="EDF-S",
        ra=52.93, dec=-28.09, source_release="Q1_R1", source_plan_fingerprint="x",
        position_index=0, position_name="c", path=tmp_path / "x.fits",
        bundle_sha256="", bands={}, record={})
    monkeypatch.setattr(vd.archive_fields, "availability", lambda: {"ready": True})
    monkeypatch.setattr(vd.archive_fields, "iter_comparison_fields", lambda: iter([field]))
    obj = client.get("/viewer/meta/archive-fields").get_json()["objects"][0]
    assert obj["field"] == "EDF-F"
    assert obj["stored_field"] == "EDF-S"


# ---------------------------------------------------------------------------
# real-field
# ---------------------------------------------------------------------------

def test_real_field_tiles_carry_offset_wcs_and_positions(client, tmp_path, monkeypatch):
    field_header = _header(crval=(267.42, 64.89), crpix=(10.5, 10.5))
    _write(tmp_path / "original_stack.fits", np.zeros((4, 20, 20)), field_header)
    cubes = tmp_path / "cubes"
    for tile in range(4):
        cubes.mkdir(exist_ok=True)
        np.save(cubes / f"lr_{tile:03d}.npy", np.zeros((10, 10, 4), np.float32))
        np.save(cubes / f"sr_{tile:03d}.npy", np.zeros((20, 20, 4), np.float32))
    manifest = {"field_id": "f1", "count": 4, "grid_side": 2, "tile_size": 10,
                "member_labels": [], "combiner_kinds": [], "pca_n": 0}
    monkeypatch.setattr(vd, "_real_field_manifest", lambda _params: manifest)
    monkeypatch.setattr(vd.real_field, "field_dir", lambda _identifier: tmp_path)

    meta = client.get("/viewer/meta/real-field").get_json()
    tile3 = meta["objects"][3]
    assert tile3["id"] == "f1/003"
    center = _world(WCS(field_header), 14.5, 14.5)      # tile (1,1) centre
    _assert_same_sky((tile3["ra"], tile3["dec"]), center)

    lr = client.get("/viewer/cube/real-field/3?tier=lr")
    _assert_same_sky(_world(_served_wcs(lr), 0, 0), _world(WCS(field_header), 10, 10))
    sr = client.get("/viewer/cube/real-field/3?tier=sr")
    served = _served_wcs(sr)
    corners = [_world(served, x, y) for x, y in ((0, 0), (1, 0), (0, 1), (1, 1))]
    _assert_same_sky(
        (float(np.mean([c[0] for c in corners])), float(np.mean([c[1] for c in corners]))),
        _world(WCS(field_header), 10, 10))
    assert sr.headers["X-Cube-Unit"] == "e-"


# ---------------------------------------------------------------------------
# nexus-field and jwst-euclid
# ---------------------------------------------------------------------------

def test_nexus_tiles_serve_lr_sr_and_jwst_wcs(client, tmp_path, monkeypatch):
    lr_header = _header(crval=(268.28, 65.1), crpix=(1768.0, -3629.0), pc=True,
                        bunit="")
    jwst_header = _header(crval=(268.46, 65.1), crpix=(-3035.5, 11806.5),
                          scale=0.03 / 3600, pc=True, bunit="MJy/sr")
    _write(tmp_path / "tiles" / "lr.fits", np.zeros((4, 5, 5)), lr_header)
    _write(tmp_path / "tiles" / "sr.fits", np.zeros((4, 10, 10)), lr_header)
    _write(tmp_path / "tiles" / "jwst.fits", np.zeros((17, 17)), jwst_header)
    manifest = {"field_id": "nx", "filter": "F200W", "tiles": [{
        "index": 0, "ra_deg": 268.3, "dec_deg": 65.1,
        "lr_file": "tiles/lr.fits", "jwst_file": "tiles/jwst.fits",
        "inference": {"files": {"starfull": "tiles/sr.fits"},
                      "pixel_scale_arcsec": 0.05},
        "jwst_metadata": {"pixel_scale_arcsec": [0.03, 0.03]},
    }]}
    monkeypatch.setattr(vd, "_nexus_field", lambda _params: (manifest, str(tmp_path)))

    meta = client.get("/viewer/meta/nexus-field?field=nx").get_json()
    obj = meta["objects"][0]
    assert (obj["id"], obj["ra"], obj["dec"]) == ("nx/0000", 268.3, 65.1)
    units = {tier["key"]: tier.get("unit") for tier in meta["tiers"]}
    assert units == {"lr": "e-", "sr": "e-", "jwst": "MJy/sr", "jwst_blur": "MJy/sr"}

    lr = client.get("/viewer/cube/nexus-field/0?tier=lr&field=nx")
    _assert_same_sky(_world(_served_wcs(lr), 0, 0), _world(WCS(lr_header), 0, 0))
    jwst = client.get("/viewer/cube/nexus-field/0?tier=jwst&field=nx")
    assert jwst.headers["X-Cube-Unit"] == "MJy/sr"
    _assert_same_sky(_world(_served_wcs(jwst), 0, 0), _world(WCS(jwst_header), 0, 0))
    blur = client.get("/viewer/cube/nexus-field/0?tier=jwst_blur&field=nx")
    _assert_same_sky(_world(_served_wcs(blur), 0, 0), _world(WCS(jwst_header), 0, 0))
    sr = client.get("/viewer/cube/nexus-field/0?tier=sr&field=nx")
    served = _served_wcs(sr)
    corners = [_world(served, x, y) for x, y in ((0, 0), (1, 0), (0, 1), (1, 1))]
    _assert_same_sky(
        (float(np.mean([c[0] for c in corners])), float(np.mean([c[1] for c in corners]))),
        _world(WCS(lr_header), 0, 0))


def test_jwst_euclid_pair_serves_lr_and_native_jwst_wcs(client, tmp_path, monkeypatch):
    lr_header = _header(crval=(53.16, -27.78), crpix=(3.0, 3.0))
    jwst_header = _header(crval=(53.16, -27.78), crpix=(8.0, 8.0),
                          scale=0.03 / 3600, bunit="MJy/sr")
    _write(tmp_path / "lr.fits", np.zeros((4, 6, 6)), lr_header)
    _write(tmp_path / "jwst.fits", np.zeros((16, 16)), jwst_header)
    manifest = {
        "field_id": "pair1", "target_name": "JADES", "ra_deg": 53.16,
        "dec_deg": -27.78, "size_arcsec": 0.6,
        "files": {"euclid": "lr.fits"},
        "jwst_bands": [{"key": "jwst0", "filter": "F200W", "file": "jwst.fits",
                        "native_is_field_cutout": True,
                        "metadata": {"pixel_scale_arcsec": [0.03, 0.03]}}],
    }
    monkeypatch.setattr(vd, "_saved_jwst_euclid_pairs",
                        lambda: [(manifest, str(tmp_path))])

    meta = client.get("/viewer/meta/jwst-euclid").get_json()
    obj = meta["objects"][0]
    assert (obj["id"], obj["ra"], obj["dec"]) == ("pair1", 53.16, -27.78)
    lr = client.get("/viewer/cube/jwst-euclid/0?tier=lr")
    _assert_same_sky(_world(_served_wcs(lr), 0, 0), _world(WCS(lr_header), 0, 0))
    jwst = client.get("/viewer/cube/jwst-euclid/0?tier=jwst&jwst_band=F200W")
    assert jwst.headers["X-Cube-Unit"] == "MJy/sr"
    _assert_same_sky(_world(_served_wcs(jwst), 0, 0), _world(WCS(jwst_header), 0, 0))


# ---------------------------------------------------------------------------
# ensemble collection (contract C6)
# ---------------------------------------------------------------------------

@pytest.fixture
def ensemble_cubes(tmp_path, monkeypatch):
    manifest = {"subset": "test", "indices": [3], "member_labels": ["00·a", "01·b"]}
    monkeypatch.setattr(vd, "_ensemble_manifest", lambda _starless: manifest)
    monkeypatch.setattr(vd, "_ensemble_cubes_dir", lambda _starless: str(tmp_path))
    monkeypatch.setattr(vd, "_sky_records_local_dir", lambda: "")
    np.save(tmp_path / "sr_00003.npy", np.full((6, 6, 4), 2.0, np.float32))
    return tmp_path


def test_ensemble_defaults_to_the_starfull_regime():
    assert vd._ensemble_starless({}) is False
    assert vd._ensemble_starless({"mode": "starless"}) is True


def test_ensemble_sr_is_the_production_gate_and_mean_is_its_own_tier(
        ensemble_cubes, monkeypatch):
    loadable = {vd.SPATIAL_GATE_KIND, vd.RAW_INCREMENTAL_MINMEANMAX_RBF_KIND}
    monkeypatch.setattr(vd, "_load_field_combiner",
                        lambda _s, _labels, kind: object() if kind in loadable else None)
    meta = vd._ensemble_meta({})
    tiers = {tier["key"]: tier for tier in meta["tiers"]}
    assert tiers["sr"]["label"] == "SR · production gate"
    assert tiers["mean"]["label"] == "Mean of members"
    rbf = vd.COMBINER_MODELS[vd.RAW_INCREMENTAL_MINMEANMAX_RBF_KIND].cube_prefix
    gate = vd.COMBINER_MODELS[vd.SPATIAL_GATE_KIND].cube_prefix
    assert rbf in tiers and gate not in tiers          # gate is `sr`, not duplicated
    assert meta["default_tier"] == "sr"
    assert meta["morph_base_tier"] == "mean"
    assert all(tier.get("unit") == "e-" for key, tier in tiers.items()
               if key in {"lr", "sr", "mean", "std"})
    assert [obj["id"] for obj in meta["objects"]] == ["test:3"]

    seen = []

    def combined(_starless, rec_index, labels, model_kind):
        seen.append((rec_index, tuple(labels), model_kind))
        return np.full((6, 6, 4), 9.0, np.float32)

    monkeypatch.setattr(vd, "_combiner_field_cube", combined)
    sr, sr_info = vd._ensemble_cube(0, "sr", {})
    mean, mean_info = vd._ensemble_cube(0, "mean", {})
    assert float(sr.mean()) == 9.0 and seen == [(3, ("00·a", "01·b"), vd.SPATIAL_GATE_KIND)]
    assert "production gate" in sr_info["label"]
    assert float(mean.mean()) == 2.0 and "Mean of members" in mean_info["label"]


def test_ensemble_without_a_gate_offers_the_mean(ensemble_cubes, monkeypatch):
    monkeypatch.setattr(vd, "_load_field_combiner", lambda *_args: None)
    meta = vd._ensemble_meta({"mode": "starfull"})
    keys = [tier["key"] for tier in meta["tiers"]]
    assert "sr" not in keys and "mean" in keys
    assert meta["default_tier"] == "mean"


def test_ensemble_member_subset_serves_the_subset_mean(ensemble_cubes, monkeypatch):
    for index, value in enumerate((1.0, 3.0)):
        np.save(ensemble_cubes / f"member{index}_00003.npy",
                np.full((6, 6, 4), value, np.float32))
    for tier in ("sr", "mean"):
        cube, info = vd._ensemble_cube(0, tier, {"members": "0,1"})
        assert np.allclose(cube, 2.0)
        assert "subset mean" in info["label"]


# ---------------------------------------------------------------------------
# multi-channel cubes and ids
# ---------------------------------------------------------------------------

def test_channel_first_cubes_with_many_channels_are_not_misread():
    assert vd._as_hwc(np.zeros((24, 64, 64))).shape == (64, 64, 24)
    assert vd._as_hwc(np.zeros((4, 53, 53))).shape == (53, 53, 4)
    assert vd._as_hwc(np.zeros((64, 64, 24))).shape == (64, 64, 24)
    assert vd._as_hwc(np.zeros((53, 53, 4))).shape == (53, 53, 4)
    # FITS cubes are channel-first by convention, even when tiny.
    assert vd._as_hwc(np.zeros((24, 16, 16)), layout="chw").shape == (16, 16, 24)
    assert vd._as_hwc(np.zeros((16, 16, 24)), layout="hwc").shape == (16, 16, 24)


def test_a_24_channel_fits_cube_is_served_channel_last(tmp_path, monkeypatch):
    root = tmp_path / "eval"
    monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(root))
    _write(root / "o" / "original_stack.fits", np.zeros((4, 8, 8)))
    _write(root / "o" / "SR.fits", np.zeros((24, 16, 16)))
    (root / "manifest.csv").write_text("id,ok,out_subdir\no,True,o\n")
    cube, _info = vd._eval_cube(0, "SR", {})
    assert cube.shape == (16, 16, 24)


def test_route_serves_a_24_channel_cube(client, monkeypatch):
    monkeypatch.setattr(
        vd, "get_cube",
        lambda *_args: (np.zeros((16, 16, 24), np.float32),
                        {"label": "multi", "pixscale": 0.05, "unit": "arb"}))
    response = client.get("/viewer/cube/ensemble/0?tier=member0")
    assert response.status_code == 200, response.get_json()
    assert response.headers["X-Cube-Shape"] == "16,16,24"
    assert len(response.headers["X-Cube-Bands"].split(",")) == 24
    assert len(response.data) == 16 * 16 * 24 * 4


def test_cutouts_objects_carry_their_catalogue_positions(tmp_path, monkeypatch):
    """C6: real Euclid star cutouts publish ``ra``/``dec`` from ``stars.csv``
    (skipped when not finite); ids stay the sorted star ids."""
    stars = []
    for sid, ra, dec in ((12, 52.5, -28.25), (3, 61.0, -48.5), (7, float("nan"), 1.0),
                         (9, 10.0, 20.0)):
        star = CatalogObject(ra=ra, dec=dec, id=sid)
        for band in vd.BAND_NAMES:
            star.set_valid(64, band=band)
        stars.append(star)
    stars[-1].flags["valid"].pop("H_E", None)          # not valid in all 4 bands
    CatalogObject.write(stars, str(tmp_path / Config.CATALOG_FILE))
    monkeypatch.setattr(web_status, "_fasrc_catalog_dir",
                        lambda force=False: str(tmp_path))

    meta = vd.get_meta("cutouts", {})

    assert [obj["id"] for obj in meta["objects"]] == ["3", "7", "12"]
    by_id = {obj["id"]: obj for obj in meta["objects"]}
    assert (by_id["3"]["ra"], by_id["3"]["dec"]) == (61.0, -48.5)
    assert (by_id["12"]["ra"], by_id["12"]["dec"]) == (52.5, -28.25)
    assert "ra" not in by_id["7"] and "dec" not in by_id["7"]
    assert meta["count"] == 3
    assert web_status._valid_4band_stars() == (64, [3, 7, 12])


def test_sky_and_psf_objects_have_stable_ids(tmp_path, monkeypatch):
    for kind in ("dirty", "hr"):
        (tmp_path / f"{kind}_test.tfrecord").touch()
    monkeypatch.setattr(vd, "_sky_records_local_dir", lambda: str(tmp_path))
    monkeypatch.setattr(vd, "_record_count", lambda *_args: 2)
    monkeypatch.setattr(vd.sky_records, "sr_count", lambda _subset: 0)
    sky = vd.get_meta("sky", {})
    assert [obj["id"] for obj in sky["objects"]] == ["test:0", "test:1"]
    assert all(tier.get("unit") == "e-" for tier in sky["tiers"])

    path = tmp_path / "psf.fits"
    hdus = [fits.PrimaryHDU(np.zeros((3, 3), np.float32))]
    hdus += [fits.ImageHDU(np.zeros((3, 3), np.float32)) for _ in range(2)]
    fits.HDUList(hdus).writeto(path)
    clusters = tmp_path / "clusters.json"
    clusters.write_text(json.dumps({"clusters": [
        {"ra": 1.0, "dec": 2.0, "n_stars": 5}, {"ra": 3.0, "dec": 4.0, "n_stars": 6}]}))
    monkeypatch.setattr(vd, "_psf_paths", lambda: {"VIS": str(path)})
    monkeypatch.setattr(vd, "_cached_psf_clusters_json", lambda: str(clusters))
    psfs = vd.get_meta("psfs", {})
    assert [obj["id"] for obj in psfs["objects"]] == ["cluster-001", "cluster-002"]
    assert (psfs["objects"][1]["ra"], psfs["objects"][1]["dec"]) == (3.0, 4.0)
    assert psfs["tiers"][0]["unit"] == "arb"
