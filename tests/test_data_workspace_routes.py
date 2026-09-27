"""Backend of the Data workspace (spec §8.4): records (inventory, SR tier
state, truth sources, sync + generate-SR jobs, the viewer's clean tier), the
star catalogue explorer and cutouts (offline, FASRC-mirror only), the PSF
inventory + sync jobs, and the TNG explorer / result pulls.

No network, no TensorFlow model: tiny TFRecords/FITS/CSVs in ``tmp_path``
and stubbed fetches / ensembles.
"""

from __future__ import annotations

import csv
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.catalog.catalog_object import CatalogObject
from euclid_polish.config import Config
from euclid_polish.image import Image, Role
from euclid_polish.image.tfio import write_images
from euclid_polish.web import fasrc_fetcher, remote
from euclid_polish.web.app import create_app
from euclid_polish.web.helpers import sky_records, star_catalog, status, tng_explorer
from euclid_polish.web.helpers import viewer_data as vd
from euclid_polish.web.jobs import REGISTRY
from euclid_polish.web.routes import psfs as psfs_routes
from euclid_polish.web.routes import tng as tng_routes
from euclid_polish.web.routes import views as views_routes

OFFLINE = {"ok": False, "error": "FASRC not connected", "code": "fasrc_offline"}


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", None)


def _wait(job_id: str, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = REGISTRY.get(job_id).to_dict()
        if job["status"] != "running":
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished")


def _images(n: int, size: int, role: Role = Role.HR) -> list[Image]:
    return [Image(data=np.full((size, size, 4), float(i), np.float32), pixel_scale_arcsec=0.05,
                  band_names=("VIS", "Y_E", "J_E", "H_E"), is_clean=True, role=role, index=i)
            for i in range(n)]


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

@pytest.fixture
def records(tmp_path, monkeypatch):
    rdir = tmp_path / "records"
    rdir.mkdir()
    write_images(_images(3, 4, Role.LR), "dirty_test", records_dir=str(rdir))
    write_images(_images(3, 8), "hr_test", records_dir=str(rdir))
    write_images(_images(3, 8), "clean_test", records_dir=str(rdir))
    with open(rdir / "sources_test.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["field_index", "type", "x_pix", "y_pix",
                                                    "flux_vis_e", "mag_vis", "off_field"])
        writer.writeheader()
        writer.writerow({"field_index": 1, "type": "star", "x_pix": 2.5, "y_pix": 3.0,
                         "flux_vis_e": 1e5, "mag_vis": 18.5, "off_field": 0})
        writer.writerow({"field_index": 1, "type": "galaxy", "x_pix": 6, "y_pix": 1,
                         "flux_vis_e": 900, "off_field": 1})
    for module in (views_routes, vd):
        monkeypatch.setattr(module, "_sky_records_local_dir", lambda: str(rdir))
    monkeypatch.setattr(sky_records, "sky_sr_dir", lambda: str(tmp_path / "sky_sr"))
    monkeypatch.setattr(views_routes, "_current_identity", lambda: {
        "member_labels": ["01·psnr"], "combiner_kind": None, "combiner_fingerprint": None})
    return rdir


def test_sr_status_lists_every_split_and_answers_offline(client, records, offline):
    r = client.get("/api/sky/sr-status")
    assert r.status_code == 200
    body = r.get_json()
    test = body["splits"]["test"]
    assert test["count"] == 3
    assert test["files"]["clean"]["count"] == 3
    assert test["files"]["sources"]["size_bytes"] > 0
    assert test["sr"]["state"] == "missing"
    assert body["splits"]["validate"]["present"] is False
    assert body["subsets"] == ["test"]
    assert body["model"]["member_labels"] == ["01·psnr"]
    assert body["sync_job"] is None and body["generate_job"] is None


def test_record_sources_one_record_and_census(client, records):
    body = client.get("/api/sky/records/sources?subset=test&index=1").get_json()
    assert [s["type"] for s in body["sources"]] == ["star", "galaxy"]
    assert body["sources"][0]["x_pix"] == 2.5 and body["sources"][1]["off_field"] is True
    assert body["counts"]["star"] == 1 and body["counts"]["off_field"] == 1
    assert body["geometry"]["hr"]["width"] == 8 and body["geometry"]["lr"]["width"] == 4
    census = client.get("/api/sky/records/sources?subset=test").get_json()
    assert census["fields"][0]["field_index"] == 1
    assert census["fields"][0]["brightest_star_mag"] == 18.5
    detail = client.get("/api/sky/records/source?subset=test&index=1&row=1").get_json()
    assert detail["values"]["type"] == "galaxy"
    missing = client.get("/api/sky/records/source?subset=test&index=1&row=5")
    assert missing.status_code == 404 and "no source" in missing.get_json()["error"]


@pytest.mark.parametrize("query", ["subset=bogus&index=0", "subset=test&index=x",
                                   "subset=test&index=-1"])
def test_record_sources_rejects_bad_arguments(client, records, query):
    r = client.get(f"/api/sky/records/sources?{query}")
    assert r.status_code == 400
    assert r.get_json()["error"]


def test_sky_viewer_offers_the_clean_tier_and_reads_by_position(records):
    meta = vd.get_meta("sky", {"subset": "test"})
    keys = [t["key"] for t in meta["tiers"]]
    assert keys == ["dirty", "hr", "bhr", "clean", "sr"]
    assert next(t for t in meta["tiers"] if t["key"] == "clean")["label"].startswith("Clean")
    assert meta["count"] == 3
    cube, info = vd.get_cube("sky", 2, "clean", {"subset": "test"})
    assert cube.shape == (8, 8, 4) and float(cube[0, 0, 0]) == 2.0
    assert "clean" in info["label"]
    lr, _ = vd.get_cube("sky", 1, "dirty", {"subset": "test"})
    assert lr.shape == (4, 4, 4) and float(lr[0, 0, 0]) == 1.0


def test_sky_sync_runs_as_a_job_and_pulls_the_selection(client, records, monkeypatch):
    pulled = []

    def fake_fetch(remote_path, **kwargs):
        pulled.append((remote_path, kwargs.get("force"), kwargs.get("max_bytes")))
        return fasrc_fetcher.FetchResult(ok=not remote_path.endswith("clean_validate.tfrecord"),
                                         size_bytes=12, error="No such file")

    monkeypatch.setattr(views_routes._fasrc_fetcher, "fetch_one_file", fake_fetch)
    r = client.post("/api/sky/sync")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["subsets"] == ["test", "validate"]
    job = _wait(body["job_id"])
    assert job["status"] == "done" and job["kind"] == "sky-sync"
    result = job["result"]
    assert result["files"]["sources_test"]["ok"] is True
    assert result["files"]["clean_validate"] == {"ok": False, "size_bytes": 12, "error": "No such file"}
    assert not any(p.endswith("_train.tfrecord") for p, _f, _m in pulled)
    assert all(force and max_bytes == 5 * 1024 ** 3 for _p, force, max_bytes in pulled)

    pulled.clear()
    job = _wait(client.post("/api/sky/sync", data={"subsets": "train", "kinds": "clean,sources"})
                .get_json()["job_id"])
    assert sorted(p.rsplit("/", 1)[1] for p, _f, _m in pulled) == ["clean_train.tfrecord",
                                                                    "sources_train.csv"]
    assert job["result"]["include_train"] is True


def test_sky_sync_validation_and_offline_gate(client, records, monkeypatch):
    assert client.post("/api/sky/sync", data={"subsets": "bogus"}).status_code == 400
    assert client.post("/api/sky/sync", data={"kinds": "hr,nope"}).status_code == 400
    monkeypatch.setattr(remote.STATE, "ssh", None)
    r = client.post("/api/sky/sync")
    assert r.status_code == 503 and r.get_json() == OFFLINE


class _FakeModel:
    label = "fake gate over 1 STARFULL models"
    member_labels = ["01·psnr"]
    combiner_kind = None

    def __init__(self, calls):
        self.calls = calls

    def upsample_batch(self, images, *, on_progress=None, log=None):
        images = list(images)
        self.calls.append(len(images))
        out = []
        for i, _image in enumerate(images):
            if on_progress:
                on_progress(i + 1, len(images), f"field {i}")
            out.append(SimpleNamespace(data=np.full((8, 8, 4), 7.0, np.float32)))
        return out


def test_generate_sr_job_writes_cubes_and_the_model_identity(client, records, monkeypatch):
    calls = []
    monkeypatch.setattr(views_routes, "load_eval_ensemble", lambda log=None: _FakeModel(calls))
    monkeypatch.setattr(views_routes.sky_records, "checkpoint_present", lambda *_a: True)
    monkeypatch.setattr(views_routes, "eval_model_identity", lambda model: {
        "member_labels": ["01·psnr"], "combiner_kind": None, "combiner_fingerprint": None})
    body = client.post("/api/sky/generate-sr").get_json()
    assert body["subsets"] == ["test"] and body["overwrite"] is False
    job = _wait(body["job_id"])
    assert job["status"] == "done", job["error"]
    assert job["result"]["generated"] == {"test": 3}
    assert sky_records.sr_count("test") == 3
    manifest = sky_records.read_sr_manifest("test")
    assert manifest["count"] == 3 and manifest["model_label"].startswith("fake")
    assert client.get("/api/sky/sr-status").get_json()["splits"]["test"]["sr"]["state"] == "current"

    # Existing SR is kept without overwrite …
    job = _wait(client.post("/api/sky/generate-sr").get_json()["job_id"])
    assert job["result"]["skipped"] == ["test"] and calls == [3]
    # … and regenerated from scratch with it.
    np.save(sky_records.sr_path("test", 9), np.zeros((1, 1, 1), np.float32))   # a stray old cube
    job = _wait(client.post("/api/sky/generate-sr", data={"overwrite": "1", "subsets": "test"})
                .get_json()["job_id"])
    assert job["result"]["generated"] == {"test": 3} and calls == [3, 3]
    assert sky_records.sr_count("test") == 3


def test_generate_sr_rejects_unknown_or_absent_splits(client, records, monkeypatch):
    monkeypatch.setattr(views_routes.sky_records, "checkpoint_present", lambda *_a: True)
    r = client.post("/api/sky/generate-sr", data={"subsets": "test,nope"})
    assert r.status_code == 400 and "unknown" in r.get_json()["error"]
    r = client.post("/api/sky/generate-sr", data={"subsets": "validate"})
    assert r.status_code == 400 and "no dirty records" in r.get_json()["error"]


# ---------------------------------------------------------------------------
# star catalogue + cutouts
# ---------------------------------------------------------------------------

def _star(sid, ra, dec, mag, valid=(), corrupted=(), failed=()):
    star = CatalogObject(ra=ra, dec=dec, id=sid, magnitude=mag, flux_psf_uJy=10.0 * sid)
    for band, size in valid:
        star.set_valid(size, band=band)
    for band, size in corrupted:
        star.set_corrupted(size, band=band)
    for band, size in failed:
        star.set_download_failed(size, band=band)
    return star


ALL4 = [(b, 511) for b in ("VIS", "Y_E", "J_E", "H_E")]


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    """A FASRC-mirror stars.csv in tmp (and a stale 'local copy' that must be ignored)."""
    cache = tmp_path / "cache" / "euclid_stars" / Config.CATALOG_FILE
    cache.parent.mkdir(parents=True)
    monkeypatch.setattr(status, "_fasrc_catalog_remote_path", lambda: "/remote/euclid_stars/stars.csv")
    monkeypatch.setattr(status, "_local_path_for", lambda _remote: str(cache))
    local = tmp_path / "local"
    local.mkdir()
    CatalogObject.write([_star(99, 1.0, 1.0, 12.0, valid=ALL4)], str(local / Config.CATALOG_FILE))
    monkeypatch.setattr(Config, "DEFAULT_OUTPUT_DIR", str(local))

    def write(stars):
        CatalogObject.write(stars, str(cache))
        return cache

    return write


def test_catalog_stars_absent_mirror_never_falls_back_to_the_local_copy(client, mirror):
    body = client.get("/api/catalog/stars").get_json()
    assert body["present"] is False and body["rows"] == []


def test_catalog_stars_payload_rows_codes_and_summary(client, mirror, offline):
    mirror([
        _star(1, 269.7, 66.0, 17.5, valid=ALL4),                                   # navigator
        _star(2, 61.2, -48.4, 18.2, valid=[("VIS", 255), ("VIS", 511)], corrupted=[("H_E", 511)]),
        _star(3, 10.0, 10.0, 18.9, failed=[("J_E", 511)]),
        _star(4, 52.9, -28.1, 16.4),                                                 # pending
    ])
    body = client.get("/api/catalog/stars").get_json()
    assert body["present"] is True and body["source"] == "fasrc-mirror"
    assert body["sizes"] == [255, 511]
    cols = body["columns"]
    rows = {row[0]: dict(zip(cols, row, strict=True)) for row in body["rows"]}
    assert rows[1]["field"] == "EDF-N" and rows[2]["field"] == "EDF-S" and rows[4]["field"] == "EDF-F"
    assert rows[3]["field"] == ""
    bits = body["bits"]
    assert rows[1]["nav"] == 1 and rows[2]["nav"] == 0
    assert rows[2]["b_VIS"] == bits["valid"] | (1 << bits["size_shift"]) | (1 << (bits["size_shift"] + 1))
    assert rows[2]["b_H_E"] == bits["corrupted"]
    assert rows[3]["b_J_E"] == bits["failed"]
    assert rows[4]["b_VIS"] == 0
    summary = body["summary"]
    assert summary["total"] == 4 and summary["valid_all4"] == 1 and summary["pending"] == 1
    assert summary["navigator"] == {"size": 511, "count": 1}
    assert (summary["mag_min"], summary["mag_max"]) == (16.4, 18.9)
    vis = next(b for b in body["band_stats"] if b["band"] == "VIS")
    assert vis["valid"] == 2 and vis["by_size"] == {"255": 1, "511": 2}
    assert body["age_s"] >= 0


def test_catalog_band_states_are_exclusive_and_sum_to_the_total(client, mirror, offline):
    # one band carrying every bit at once: valid at 511 after corrupted at 255 and a failed retry
    mirror([
        _star(1, 269.7, 66.0, 17.5, valid=[("Y_E", 511)], corrupted=[("Y_E", 255)], failed=[("Y_E", 1023)]),
        _star(2, 61.2, -48.4, 18.2, corrupted=[("Y_E", 511)], failed=[("Y_E", 255), ("VIS", 511)]),
        _star(3, 10.0, 10.0, 18.9, failed=[("Y_E", 511)]),
        _star(4, 52.9, -28.1, 16.4),
    ])
    body = client.get("/api/catalog/stars").get_json()
    summary = body["summary"]
    assert summary["total"] == 4
    for band in body["band_stats"]:
        assert sum(band[k] for k in ("valid", "corrupted", "failed", "pending")) == 4, band
    y = next(b for b in body["band_stats"] if b["band"] == "Y_E")
    assert (y["valid"], y["corrupted"], y["failed"], y["pending"]) == (1, 1, 1, 1)
    vis = next(b for b in body["band_stats"] if b["band"] == "VIS")
    assert (vis["valid"], vis["corrupted"], vis["failed"], vis["pending"]) == (0, 0, 1, 3)
    # the star-level state is the star's best band (the "Overall" filter)
    assert sum(summary[k] for k in ("valid", "corrupted", "failed", "pending")) == 4
    assert (summary["valid"], summary["corrupted"], summary["failed"], summary["pending"]) == (1, 1, 1, 1)


def test_star_cutout_totals_answer_offline_from_the_mirror(client, mirror, offline):
    mirror([_star(1, 269.7, 66.0, 17.5, valid=ALL4), _star(2, 269.8, 66.1, 17.6, valid=ALL4)])
    r = client.get("/api/star-cutouts/totals")
    assert r.status_code == 200
    body = r.get_json()
    assert body["count"] == 2 and body["size"] == 511 and body["cached"] is True
    assert body["catalog"]["present"] is True


def test_cutouts_viewer_meta_reads_the_mirror_only(mirror, monkeypatch):
    mirror([_star(5, 269.7, 66.0, 17.25, valid=ALL4)])
    monkeypatch.setattr(status._fasrc_fetcher, "fetch_one_file", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("the cutouts meta must not fetch")))
    meta = vd.get_meta("cutouts", {})
    assert meta["objects"][0]["id"] == "5" and meta["objects"][0]["mag"] == 17.25
    assert "VIS 17.25" in meta["objects"][0]["label"]


def test_non_forced_catalog_read_falls_back_to_the_stale_mirror(mirror, monkeypatch):
    mirror([_star(1, 269.7, 66.0, 17.5, valid=ALL4)])
    monkeypatch.setattr(status._fasrc_fetcher, "fetch_one_file",
                        lambda *_a, **_k: fasrc_fetcher.FetchResult(ok=False, error="ssh not connected"))
    assert status._fasrc_catalog_dir(force=False) is not None
    assert status._fasrc_catalog_dir(force=True) is None


def test_catalog_reads_are_memoised_per_file_state(mirror, monkeypatch):
    path = mirror([_star(1, 269.7, 66.0, 17.5, valid=ALL4)])
    calls = []
    real = CatalogObject.read
    monkeypatch.setattr(status.CatalogObject, "read", classmethod(
        lambda cls, p: calls.append(p) or real(p)))
    status.read_catalog_objects(str(path))
    status.read_catalog_objects(str(path))
    assert len(calls) <= 1


def test_cutout_gallery_items_carry_the_star(client, mirror, tmp_path, monkeypatch):
    mirror([_star(12, 269.7, 66.0, 17.5, valid=ALL4)])
    out = tmp_path / "gallery"
    band_dir = out / "cutouts" / "VIS"
    band_dir.mkdir(parents=True)
    (band_dir / "star_0012_511.fits").write_bytes(b"")
    (band_dir / "star_0077_255.fits").write_bytes(b"")
    body = client.get(f"/api/cutouts/VIS/list.json?output_dir={out}").get_json()
    items = {item["id"]: item for item in body["items"]}
    assert items[12]["size"] == 511 and items[12]["mag"] == 17.5 and items[12]["ra"] == 269.7
    assert items[77]["ra"] is None
    assert body["files"] == ["star_0012_511.fits", "star_0077_255.fits"]


# ---------------------------------------------------------------------------
# PSFs
# ---------------------------------------------------------------------------

def _psf_file(path, n_clusters=2, fwhm=0.16):
    hdus = [fits.PrimaryHDU(np.zeros((9, 9), np.float32))]
    hdus[0].header["NPSF"] = n_clusters
    hdus[0].header["PXSCALE"] = 0.05
    for i in range(n_clusters):
        hdu = fits.ImageHDU(np.zeros((9, 9), np.float32))
        hdu.header.update({"RA": 269.0 + i, "DEC": 66.0, "NSTARS": 10 + i, "FWHM": fwhm + 0.01 * i})
        hdus.append(hdu)
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(path, overwrite=True)


@pytest.fixture
def psf_cache(tmp_path, monkeypatch):
    cache = tmp_path / "fasrc_cache"
    monkeypatch.setattr(Config, "FASRC_CACHE_DIR", str(cache))
    monkeypatch.setattr(status, "_local_path_for",
                        lambda remote_path: str(cache / remote_path.lstrip("/")))
    monkeypatch.setattr(status.fasrc_config, "load", lambda: SimpleNamespace(
        data_dir="/remote/data", conda_env_path="/env"))
    return cache


def test_psf_inventory_distinguishes_not_cached_from_no_empirical(client, psf_cache, offline):
    body = client.get("/api/euclid-psf/inventory").get_json()
    assert {b["state"] for b in body["bands"]} == {"not_cached"}
    assert body["clusters"] == []

    _psf_file(psf_cache / "remote/data/euclid_psf" / Config.BAND_VIS.psf_fits_filename)
    status.write_psf_sync_status({
        "Y_E": {"ok": False, "error": "stat: cannot stat 'x': No such file or directory"},
        "J_E": {"ok": False, "error": "rsync exit 12: connection reset"},
    })
    body = client.get("/api/euclid-psf/inventory").get_json()
    states = {b["name"]: b for b in body["bands"]}
    assert states["VIS"]["state"] == "empirical" and states["VIS"]["n_psf"] == 2
    assert states["Y_E"]["state"] == "no_empirical"
    assert states["J_E"]["state"] == "not_cached" and "connection reset" in states["J_E"]["error"]
    assert states["H_E"]["state"] == "not_cached" and states["H_E"]["error"] is None
    assert body["clusters_source"] == "vis_headers"
    first = body["clusters"][0]
    assert first["id"] == "cluster-001" and first["n_stars"] == 10
    assert first["fwhm_by_band"]["VIS"] == pytest.approx(0.16)


def test_psf_cluster_table_prefers_the_metadata_json(client, psf_cache):
    meta = psf_cache / "remote/data/euclid_psf" / status.PSF_CLUSTERS_META
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"clusters": [
        {"index": 1, "ra": 1.5, "dec": 2.5, "n_stars": 7, "fwhm_arcsec": 0.17,
         "fwhm_by_band": {"VIS": 0.17, "H_E": 0.5}}]}))
    body = client.get("/api/euclid-psf/inventory").get_json()
    assert body["clusters_source"] == "metadata"
    assert body["clusters"][0] == {"index": 1, "id": "cluster-001", "ra": 1.5, "dec": 2.5,
                                   "n_stars": 7, "fwhm_by_band": {"VIS": 0.17, "H_E": 0.5}}
    assert body["clusters_meta"]["present"] is True


def test_psf_sync_job_records_each_band_outcome(client, psf_cache, monkeypatch):
    monkeypatch.setattr(psfs_routes.fasrc_config, "load", lambda: SimpleNamespace(
        data_dir="/remote/data", conda_env_path="/env"))

    def fake_fetch(remote_path, **kwargs):
        if remote_path.endswith(Config.BAND_VIS.psf_fits_filename):
            local = psf_cache / remote_path.lstrip("/")
            _psf_file(local)
            return fasrc_fetcher.FetchResult(ok=True, local_path=str(local), size_bytes=10)
        return fasrc_fetcher.FetchResult(ok=False, error="file not found on remote")

    monkeypatch.setattr(psfs_routes._fasrc_fetcher, "fetch_one_file", fake_fetch)
    monkeypatch.setattr(psfs_routes, "_sync_clusters_meta", lambda cfg: {"ok": True, "n_clusters": 2})
    body = client.post("/api/euclid-psf/sync").get_json()
    job = _wait(body["job_id"])
    assert job["status"] == "done" and job["kind"] == "psf-sync"
    assert job["result"]["files"]["Y_E"]["missing_remote"] is True
    states = {b["name"]: b["state"] for b in client.get("/api/euclid-psf/inventory").get_json()["bands"]}
    assert states == {"VIS": "empirical", "Y_E": "no_empirical", "J_E": "no_empirical",
                      "H_E": "no_empirical"}

    meta = client.post("/api/euclid-psf/sync-meta").get_json()
    assert _wait(meta["job_id"])["result"] == {"ok": True, "n_clusters": 2}


def test_psf_syncs_are_gated_offline(client, offline):
    for path in ("/api/euclid-psf/sync", "/api/euclid-psf/sync-meta"):
        r = client.post(path)
        assert r.status_code == 503 and r.get_json() == OFFLINE


def test_clusters_meta_dump_reads_every_band_header():
    cmd = psfs_routes._clusters_meta_remote_cmd(SimpleNamespace(data_dir="/d", conda_env_path="/e"))
    assert "fwhm_by_band" in cmd and "lazy_load_hdus=True" in cmd
    for band in Config.BANDS:
        assert band.psf_fits_filename in cmd


# ---------------------------------------------------------------------------
# TNG
# ---------------------------------------------------------------------------

@pytest.fixture
def tng_dir(tmp_path, monkeypatch):
    root = tmp_path / "_tng_infographics"
    root.mkdir()
    (root / "tng_properties.csv").write_text(
        "id,sfr,mass_stars,m_halo,reff\n1,0.0,3e11,2e12,8.4\n2,0.5,1e10,,3.0\n")
    (root / "tng_atlas_parameters.csv").write_text(
        "subhalo_id,orientation,native_re_px,native_re_kpc,sfr_msun_yr,mass_stars_msun,"
        "m_halo_msun,groupcat_reff_kpc\n"
        "1,2,50,5.0,0,3e11,2e12,8.4\n1,1,70,7.0,0,3e11,2e12,8.4\n9,1,10,1.0,2,1e9,1e11,1.5\n")
    (root / "tng_atlas_parameters.csv.meta.json").write_text(json.dumps({"valid": True, "row_count": 3}))
    skirt = tmp_path / "tng_skirt"
    (skirt / "9").mkdir(parents=True)
    (skirt / "9" / "TNG9_O1_Euclid_VIS.fits").write_bytes(b"")
    monkeypatch.setattr(tng_explorer, "calibration_dir", lambda: str(root))
    monkeypatch.setattr(Config, "TNG_SKIRT_DIR", str(skirt))
    return root


def test_tng_properties_explorer_from_local_csvs(client, tng_dir, offline):
    body = client.get("/api/tng/properties").get_json()
    assert body["present"] is True
    rows = {row[0]: dict(zip(body["columns"], row, strict=True)) for row in body["rows"]}
    assert sorted(rows) == [1, 2, 9]
    assert rows[1]["re_kpc"] == pytest.approx(6.0) and rows[1]["n_orient"] == 2
    assert (rows[1]["re_kpc_min"], rows[1]["re_kpc_max"]) == (5.0, 7.0)
    assert rows[2]["m_halo"] is None and rows[2]["n_orient"] == 0
    assert rows[9]["sfr"] == 2.0 and rows[9]["local"] == 1          # from the atlas copy
    assert body["orientations"]["1"] == [[1, 70.0, 7.0], [2, 50.0, 5.0]]
    assert body["summary"] == {"n": 3, "n_quenched": 1, "n_missing_sfr": 0, "n_in_atlas": 2,
                               "n_local": 1}
    assert body["files"]["atlas"]["rows"] == 3 and body["atlas_meta"]["valid"] is True


def test_tng_result_get_is_cache_only_and_the_pull_is_a_job(client, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setattr(tng_routes, "_local_path_for", lambda remote_path: str(cache / remote_path.lstrip("/")))
    monkeypatch.setattr(tng_routes, "fetch_one_file", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("a GET must not pull")))
    r = client.get("/tng/result/grid.png")
    assert r.status_code == 404 and "pull" in r.get_json()["error"]
    assert client.get("/api/tng/results").get_json()["grid"]["present"] is False

    png = b"\x89PNG\r\n\x1a\n" + b"grid"

    def fake_fetch(remote_path, **kwargs):
        local = cache / remote_path.lstrip("/")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(png)
        return fasrc_fetcher.FetchResult(ok=True, local_path=str(local), size_bytes=len(png))

    monkeypatch.setattr(tng_routes, "fetch_one_file", fake_fetch)
    job = _wait(client.post("/api/tng/result/pull", data={"kind": "grid"}).get_json()["job_id"])
    assert job["status"] == "done" and job["result"]["grid"]["ok"] is True
    r = client.get("/tng/result/grid.png")
    assert r.status_code == 200 and r.data == png
    assert client.get("/api/tng/results").get_json()["grid"]["present"] is True
    archived = os.listdir(os.path.join(Config.VIS_DIR, "tng"))
    assert len(archived) == 1 and archived[0].startswith("tng_grid_")
    assert client.post("/api/tng/result/pull", data={"kind": "nope"}).status_code == 400


def test_tng_pulls_and_refresh_are_gated_offline(client, offline):
    for path in ("/api/tng/result/pull", "/api/tng/properties/refresh"):
        r = client.post(path)
        assert r.status_code == 503 and r.get_json() == OFFLINE
    # the cached result GETs answer offline (404 until pulled)
    assert client.get("/tng/result/grid.png").status_code in (200, 404)


def test_tng_properties_refresh_job_queries_missing_galaxies(client, tng_dir, monkeypatch):
    class _Ssh:
        def is_connected(self):
            return True

        def run(self, cmd, timeout=None):
            return (0, "/remote/tng_skirt/1\n/remote/tng_skirt/9\n", "")

    monkeypatch.setattr(remote.STATE, "ssh", _Ssh())
    monkeypatch.setenv("TNG_API_KEY", "k")
    seen = {}

    def fake_gather(work, ids, key, reporter=None):
        seen.update(work=work, ids=ids, key=key)
        reporter.set_step(1, 2, "TNG1")
        return {"1": {}, "9": {}}

    monkeypatch.setattr(tng_routes, "gather_properties", fake_gather)
    job = _wait(client.post("/api/tng/properties/refresh").get_json()["job_id"])
    assert job["status"] == "done", job["error"]
    assert job["result"] == {"n_ids": 2, "n_resolved": 2}
    assert seen["ids"] == ["1", "9"] and seen["key"] == "k" and seen["work"] == str(tng_dir)


def test_star_catalog_module_exports():
    assert star_catalog.COLUMNS[0] == "id" and star_catalog.COLUMNS[-1] == "nav"


def test_training_log_view_renders_in_memory_and_never_writes_data(client, tmp_path, monkeypatch):
    """GET /view/training-log is cache-only: nothing lands under VIS_DIR."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    vis = tmp_path / "vis"
    monkeypatch.setattr(Config, "VIS_DIR", str(vis))
    header = "step,wall_time,loss,psnr_stretched,psnr_raw,save_best_score,combined_loss,is_baseline\n"
    (ckpt / "training_log.csv").write_text(header + "1000,1.0,0.04,46.6,39.9,46.6,0.003,\n")
    r = client.get(f"/view/training-log?checkpoint_dir={ckpt}")
    assert r.status_code == 200 and r.data[:8] == b"\x89PNG\r\n\x1a\n"
    first = r.data
    assert client.get(f"/view/training-log?checkpoint_dir={ckpt}").data == first   # memoised
    (ckpt / "training_log.csv").write_text(header)                                 # mid-write
    assert client.get(f"/view/training-log?checkpoint_dir={ckpt}&force=1").data == first
    assert not vis.exists()
    assert not list(ckpt.glob("*.png"))
