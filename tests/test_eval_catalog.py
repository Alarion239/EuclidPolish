"""Tests for the catalog-based evaluation pipeline.

Covers the generic catalog reader, the local catalog runner (auto-fetch /
early-out), the shared per-object ``reconstruct_cutout_at`` helper (driven with
stubbed download + model so no network / TF weights are needed), and the
``/evaluation`` routes including the local run-eval / run-zoobot jobs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.euclid.eval_catalog import CatalogError, read_eval_catalog
from euclid_polish.web.app import create_app
from euclid_polish.web.fasrc_pipeline import REGISTRY


# --------------------------------------------------------------------------- #
# Catalog reader
# --------------------------------------------------------------------------- #

def _write_csv(tmp_path, text: str) -> str:
    p = tmp_path / "cat.csv"
    p.write_text(text)
    return str(p)


class TestReadEvalCatalog:
    def test_basic_and_aliases(self, tmp_path):
        # Uses the lens-catalog-style headers (id_str / right_ascension /
        # declination) to exercise the alias resolution.
        path = _write_csv(tmp_path,
            "id_str,right_ascension,declination,grade,subset\n"
            "lensA,12.5,-30.1,A,discovery_engine\n"
            "lensB,200.0,5.0,B,gz_euclid\n")
        rows = read_eval_catalog(path)
        assert [r["id"] for r in rows] == ["lensA", "lensB"]
        assert rows[0]["ra"] == 12.5 and rows[0]["dec"] == -30.1
        assert rows[0]["grade"] == "A"
        # Non-canonical columns are preserved under ``extra``.
        assert rows[0]["extra"]["subset"] == "discovery_engine"

    def test_grade_filter_and_max_n(self, tmp_path):
        path = _write_csv(tmp_path,
            "id,ra,dec,grade\n"
            "a,1,1,A\nb,2,2,B\nc,3,3,A\nd,4,4,A\n")
        a_rows = read_eval_catalog(path, grade="A")
        assert [r["id"] for r in a_rows] == ["a", "c", "d"]
        # max_n caps after filtering, in file order.
        assert [r["id"] for r in read_eval_catalog(path, grade="A", max_n=2)] \
            == ["a", "c"]

    def test_missing_required_column_raises(self, tmp_path):
        path = _write_csv(tmp_path, "id,ra\nx,1\n")  # no dec
        with pytest.raises(CatalogError):
            read_eval_catalog(path)

    def test_grade_filter_without_grade_column_raises(self, tmp_path):
        path = _write_csv(tmp_path, "id,ra,dec\nx,1,2\n")
        with pytest.raises(CatalogError):
            read_eval_catalog(path, grade="A")

    def test_bad_coordinate_raises(self, tmp_path):
        path = _write_csv(tmp_path, "id,ra,dec\nx,not_a_number,2\n")
        with pytest.raises(CatalogError):
            read_eval_catalog(path)


# --------------------------------------------------------------------------- #
# Catalog runner (local; auto-fetch + early-out paths, no model needed)
# --------------------------------------------------------------------------- #

class TestRunCatalogEval:
    def test_not_a_fasrc_step(self):
        # Catalog eval runs locally now, not as a FASRC pipeline step.
        with pytest.raises(KeyError):
            REGISTRY.get("eval_catalog")

    def test_autofetch_then_empty_returns_cleanly(self, tmp_path, monkeypatch):
        from euclid_polish.eval import catalog_runner
        from euclid_polish.euclid import lens_catalog
        monkeypatch.setattr(Config, "EVAL_CATALOG_DIR", str(tmp_path / "cat"))
        called = {}

        def fake_fetch(out_csv=None, **k):
            called["out"] = out_csv
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            with open(out_csv, "w") as f:
                f.write("id,ra,dec\n")          # header only → 0 objects
            return out_csv, 0
        monkeypatch.setattr(lens_catalog, "fetch", fake_fetch)
        logs = []
        res = catalog_runner.run_catalog_eval(
            out_dir=str(tmp_path / "out"), catalog_path=None,
            log=logs.append)
        assert res["n"] == 0 and called["out"].endswith("lenses.csv")

    def test_explicit_missing_catalog_raises(self, tmp_path):
        from euclid_polish.eval import catalog_runner
        with pytest.raises(FileNotFoundError):
            catalog_runner.run_catalog_eval(
                out_dir=str(tmp_path / "out"),
                catalog_path=str(tmp_path / "nope.csv"))


# --------------------------------------------------------------------------- #
# Shared per-object helper (no network, no TF weights)
# --------------------------------------------------------------------------- #

class TestReconstructCutoutAt:
    def test_writes_outputs_and_metrics(self, tmp_path, monkeypatch):
        from euclid_polish.web.helpers import jobs_impl

        h = w = 16

        def fake_fetch(*, ra, dec, band_name, output_file, cutout_size_vis_pixels):
            data = np.ones((h, w), dtype=np.float32)
            hdr = fits.Header()
            hdr["MAGZERO"] = 30.0
            fits.PrimaryHDU(data, header=hdr).writeto(output_file, overwrite=True)
            return True, None

        def fake_reconstruct(model, lr_cube):
            # SR cube is 2× the LR grid, 4 bands — matches the real model shape.
            sr = np.ones((2 * h, 2 * w, lr_cube.shape[-1]), dtype=np.float32)
            return lr_cube[..., 0], sr

        monkeypatch.setattr(jobs_impl, "fetch_cutout_at", fake_fetch)
        monkeypatch.setattr(jobs_impl, "reconstruct", fake_reconstruct)

        out_dir = str(tmp_path / "obj")
        res = jobs_impl.reconstruct_cutout_at(
            model=None, ra=12.3, dec=-4.5, cutout_size_vis_pixels=h,
            out_dir=out_dir, render=False, checkpoint_dir="ckpt-x",
        )

        assert os.path.isfile(os.path.join(out_dir, "SR.fits"))
        assert os.path.isfile(os.path.join(out_dir, "original_stack.fits"))
        for band in Config.LR_INPUT_BAND_NAMES:
            assert os.path.isfile(os.path.join(out_dir, f"{band}.fits"))

        # SR FITS carries the provenance + band header.
        with fits.open(res["sr_fits_path"]) as hdul:
            hdr = hdul[0].header
            assert hdr["RA"] == pytest.approx(12.3)
            assert hdr["CSIZE"] == h
            assert "VIS" in hdr["BANDS"]

        # Metrics: flux conservation only (no forward-model residual for real
        # cutouts — the true PSF is unknown). SR is the fake ones(32²) cube, so
        # Σ SR_VIS = 1024; the ratio is self-consistent with the totals.
        m = res["metrics"]
        assert set(m) == {"lr_total_e", "sr_total_e", "flux_ratio_sr_over_lr"}
        assert m["sr_total_e"] == pytest.approx(2 * h * 2 * w)
        assert m["flux_ratio_sr_over_lr"] == pytest.approx(
            m["sr_total_e"] / m["lr_total_e"])
        assert res["png_paths"] == []   # render=False


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


class TestEvaluationRoutes:
    def test_page_renders(self, client):
        r = client.get("/evaluation")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        # Local run forms (eval + zoobot) + the catalog-fetch + results
        # controls are present (no FASRC step cards).
        assert "runEvalBtn" in body and "runZoobotBtn" in body
        assert "jobPanel" in body
        assert "fetchCatBtn" in body and "runSelect" in body

    def test_runs_api_empty(self, client):
        r = client.get("/api/evaluation/runs")
        assert r.status_code == 200
        assert "runs" in r.get_json()

    def test_runs_api_missing_run_404(self, client):
        assert client.get("/api/evaluation/runs?run=nope").status_code == 404

    def test_eval_files_traversal_blocked(self, client):
        assert client.get("/eval-files/../../etc/passwd").status_code == 403

    def test_render_on_demand_from_fits(self, client, tmp_path, monkeypatch):
        # FASRC writes only FITS; the server renders the PNG locally on first
        # request. Lay down SR.fits + original_stack.fits and confirm a missing
        # eye.png is rendered and served.
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        obj = os.path.join(Config.EVAL_RESULTS_DIR, "run1", "lensA")
        os.makedirs(obj, exist_ok=True)
        h = w = 16
        sr = np.ones((4, 2 * h, 2 * w), dtype=np.float32)
        sr[0, 12:20, 12:20] = 50.0
        fits.PrimaryHDU(sr, header=fits.Header({"ASINH": 100.0})).writeto(
            os.path.join(obj, "SR.fits"))
        stack = np.ones((4, h, w), dtype=np.float32)
        stack[0, 6:10, 6:10] = 50.0
        fits.PrimaryHDU(stack).writeto(os.path.join(obj, "original_stack.fits"))

        assert not os.path.isfile(os.path.join(obj, "eye.png"))
        r = client.get("/eval-files/run1/lensA/eye.png")
        assert r.status_code == 200 and r.mimetype == "image/png"
        assert os.path.isfile(os.path.join(obj, "eye.png"))

    def test_render_clip_caches_per_clip(self, client, tmp_path, monkeypatch):
        # The "Dirty clip %ile" control: ?clip=99.9 renders to its own cache
        # file so different clips coexist; the default clip uses the plain name.
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        obj = os.path.join(Config.EVAL_RESULTS_DIR, "run1", "lensA")
        os.makedirs(obj, exist_ok=True)
        h = w = 16
        sr = np.ones((4, 2 * h, 2 * w), dtype=np.float32)
        sr[0, 12:20, 12:20] = 50.0
        fits.PrimaryHDU(sr, header=fits.Header({"ASINH": 100.0})).writeto(
            os.path.join(obj, "SR.fits"))
        stack = np.ones((4, h, w), dtype=np.float32)
        stack[0, 6:10, 6:10] = 50.0
        fits.PrimaryHDU(stack).writeto(os.path.join(obj, "original_stack.fits"))

        r = client.get("/eval-files/run1/lensA/eye.png?clip=99.9")
        assert r.status_code == 200 and r.mimetype == "image/png"
        assert os.path.isfile(os.path.join(obj, "eye__c99.9.png"))
        # Default clip keeps the plain filename.
        assert client.get("/eval-files/run1/lensA/eye.png").status_code == 200
        assert os.path.isfile(os.path.join(obj, "eye.png"))
        # The interactive viewer also drives the asinh knee; clip+asinh cache
        # to a combined per-setting filename.
        r = client.get("/eval-files/run1/lensA/eye.png?clip=99.9&asinh=300")
        assert r.status_code == 200
        assert os.path.isfile(os.path.join(obj, "eye__c99.9__a300.png"))

    def test_rerender_drops_cached_pngs(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        obj = os.path.join(Config.EVAL_RESULTS_DIR, "run1", "lensA")
        os.makedirs(obj, exist_ok=True)
        with open(os.path.join(obj, "eye.png"), "wb") as f:
            f.write(b"stale")
        r = client.post("/api/evaluation/rerender", data={"run": "run1"})
        assert r.status_code == 200 and r.get_json()["removed"] == 1
        assert not os.path.isfile(os.path.join(obj, "eye.png"))

    def test_rerender_rejects_traversal(self, client):
        assert client.post("/api/evaluation/rerender",
                           data={"run": "../etc"}).status_code == 400


    def test_fetch_catalog_endpoint(self, client, monkeypatch):
        # Stub the (network) fetch so the route is exercised offline.
        from euclid_polish.euclid import lens_catalog

        monkeypatch.setattr(
            lens_catalog, "fetch",
            lambda *a, **k: (os.path.join(Config.EVAL_CATALOG_DIR,
                                          "lens_catalog", "lenses.csv"), 309))
        r = client.post("/api/evaluation/fetch-catalog")
        assert r.status_code == 200
        j = r.get_json()
        assert j["ok"] and j["rows"] == 309
        assert j["rel"].endswith("lenses.csv")

    def test_fetch_catalog_endpoint_reports_failure(self, client, monkeypatch):
        from euclid_polish.euclid import lens_catalog

        def boom(*a, **k):
            raise RuntimeError("zenodo down")
        monkeypatch.setattr(lens_catalog, "fetch", boom)
        r = client.post("/api/evaluation/fetch-catalog")
        assert r.status_code == 502
        assert "zenodo down" in r.get_json()["error"]

    def test_runs_api_reads_manifest(self, client, tmp_path, monkeypatch):
        # Redirect the results dir to a tmp tree so we don't touch real data/.
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        # Lay down a fake run dir with a manifest and confirm it surfaces.
        run_dir = os.path.join(Config.EVAL_RESULTS_DIR, "run1")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "manifest.csv"), "w") as f:
            f.write("id,ra,dec,grade,ok,error,out_subdir,flux_ratio_sr_over_lr\n")
            f.write("lensA,1,2,A,True,,lensA,1.02\n")
        r = client.get("/api/evaluation/runs?run=run1")
        assert r.status_code == 200
        rows = r.get_json()["rows"]
        assert rows and rows[0]["id"] == "lensA"


# --------------------------------------------------------------------------- #
# Zoobot morphology helpers (pure — no torch / zoobot)
# --------------------------------------------------------------------------- #

def _write_cube(path, value=1.0, bands=4, h=8, w=8):
    cube = np.full((bands, h, w), value, dtype=np.float32)
    fits.PrimaryHDU(cube).writeto(path, overwrite=True)


class TestZoobotMorphHelpers:
    def test_discover_objects(self, tmp_path):
        from euclid_polish.eval import zoobot_morph as zm

        run = tmp_path / "run"
        # obj1: full set (before/after/hr); obj2: after only; bad: no SR.
        for name, files in {
            "obj1": ("SR.fits", "original_stack.fits", "HR.fits"),
            "obj2": ("SR.fits",),
            "bad":  ("original_stack.fits",),
        }.items():
            d = run / name
            d.mkdir(parents=True)
            for fn in files:
                _write_cube(str(d / fn))

        objs = {o["id"]: o for o in zm.discover_objects(str(run))}
        assert set(objs) == {"obj1", "obj2"}        # 'bad' skipped (no SR)
        assert objs["obj1"]["before"] and objs["obj1"]["hr"]
        assert objs["obj2"]["before"] is None and objs["obj2"]["hr"] is None

    def test_stretch_and_render_png(self, tmp_path):
        from PIL import Image

        from euclid_polish.eval import zoobot_morph as zm

        fits_path = str(tmp_path / "SR.fits")
        cube = np.zeros((4, 16, 16), dtype=np.float32)
        cube[0, 4:12, 4:12] = 500.0          # a bright VIS blob
        fits.PrimaryHDU(cube).writeto(fits_path, overwrite=True)

        u8 = zm.stretch_to_uint8(zm.load_vis_plane(fits_path), asinh_scale=100.0)
        assert u8.dtype == np.uint8 and u8.max() > u8.min()

        out_png = str(tmp_path / "after.png")
        zm.render_vis_png(fits_path, out_png, asinh_scale=100.0, size=64)
        with Image.open(out_png) as im:
            assert im.size == (64, 64) and im.mode == "RGB"

    def test_vector_deltas_without_ref(self):
        from euclid_polish.eval import zoobot_morph as zm

        d = zm.vector_deltas([0.0, 0.0], [3.0, 4.0])
        assert d["l2_before_after"] == pytest.approx(5.0)
        assert "closer_to_ref" not in d

    def test_vector_deltas_with_ref(self):
        from euclid_polish.eval import zoobot_morph as zm

        # after (1.0) is closer to ref (1.0) than before (0.0) is.
        d = zm.vector_deltas([0.0, 0.0], [1.0, 1.0], ref=[1.0, 1.0])
        assert d["closer_to_ref"] is True
        assert d["ref_improvement"] > 0
        assert d["l2_after_ref"] == pytest.approx(0.0)

    def test_write_morph_manifest(self, tmp_path):
        from euclid_polish.eval import zoobot_morph as zm

        out = str(tmp_path / "m.csv")
        zm.write_morph_manifest(out, [
            {"id": "a", "l2_before_after": 1.0, "closer_to_ref": True},
            {"id": "b", "l2_before_after": 2.0},
        ])
        with open(out) as f:
            text = f.read()
        assert "id," in text.splitlines()[0]
        assert "closer_to_ref" in text


class TestLensCatalogModule:
    def test_normalize_grade_filter(self, tmp_path):
        from euclid_polish.euclid import lens_catalog

        raw = tmp_path / "raw.csv"
        raw.write_text(
            "subset,id_str,right_ascension,declination,grade\n"
            "discovery_engine,a,1.0,2.0,A\n"
            "gz_euclid,b,3.0,4.0,B\n"
            "discovery_engine,c,5.0,6.0,A\n")
        out = str(tmp_path / "lenses.csv")
        n = lens_catalog.normalize(str(raw), out, grade="A")
        assert n == 2
        rows = read_eval_catalog(out)
        assert [r["id"] for r in rows] == ["a", "c"]
        assert rows[0]["extra"]["subset"] == "discovery_engine"

    def test_fetch_uses_source_without_network(self, tmp_path):
        # source= short-circuits the download, so this never touches the net.
        from euclid_polish.euclid import lens_catalog

        raw = tmp_path / "raw.csv"
        raw.write_text(
            "subset,id_str,right_ascension,declination,grade\n"
            "discovery_engine,a,1.0,2.0,A\n")
        out = str(tmp_path / "out.csv")
        got, n = lens_catalog.fetch(out, source=str(raw))
        assert got == out and n == 1


class TestLocalRunRoutes:
    """The eval + zoobot jobs run locally via /api/evaluation/run-*."""

    @pytest.fixture
    def client(self):
        from euclid_polish.web.app import create_app
        app = create_app()
        app.config.update(TESTING=True)
        return app.test_client()

    def test_run_eval_spawns_local_job(self, client, tmp_path, monkeypatch):
        from euclid_polish.eval import catalog_runner
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        # Don't actually run the model — stub the runner.
        monkeypatch.setattr(catalog_runner, "run_catalog_eval",
                            lambda **k: {"n": 0})
        r = client.post("/api/evaluation/run-eval",
                        data={"run_name": "lensesA", "grade": "A", "max_n": "3"})
        assert r.status_code == 200 and r.get_json()["job_id"]

    def test_run_eval_rejects_bad_run(self, client):
        r = client.post("/api/evaluation/run-eval", data={"run_name": "../x"})
        assert r.status_code == 400

    def test_run_zoobot_env_missing_hint(self, client, tmp_path, monkeypatch):
        from euclid_polish.web.routes import evaluation as evmod
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        os.makedirs(os.path.join(Config.EVAL_RESULTS_DIR, "run1"))
        monkeypatch.setattr(evmod, "_zoobot_python", lambda: None)
        r = client.post("/api/evaluation/run-zoobot", data={"run": "run1"})
        assert r.status_code == 400
        assert "Zoobot env not found" in r.get_json()["error"]

    def test_run_zoobot_spawns_job(self, client, tmp_path, monkeypatch):
        from euclid_polish.web.routes import evaluation as evmod
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        os.makedirs(os.path.join(Config.EVAL_RESULTS_DIR, "run1"))
        monkeypatch.setattr(evmod, "_zoobot_python", lambda: "/fake/python")
        monkeypatch.setattr(evmod, "_spawn_subprocess_job",
                            lambda label, cmd, result: "job1")
        r = client.post("/api/evaluation/run-zoobot", data={"run": "run1"})
        assert r.status_code == 200 and r.get_json()["job_id"] == "job1"

    def test_run_zoobot_missing_run_404(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        os.makedirs(Config.EVAL_RESULTS_DIR, exist_ok=True)
        assert client.post("/api/evaluation/run-zoobot",
                           data={"run": "nope"}).status_code == 404
