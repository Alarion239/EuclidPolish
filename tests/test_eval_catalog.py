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
from euclid_polish.eval.eval_catalog import CatalogError, read_eval_catalog
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
        from euclid_polish.eval import catalog_runner, lens_catalog
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

    def test_manifest_upsert_preserves_existing_rows(self, tmp_path):
        from euclid_polish.eval import catalog_runner

        manifest = tmp_path / "manifest.csv"
        catalog_runner.write_manifest_upsert(str(manifest), [
            {"id": "old", "ra": 1.0, "dec": 2.0, "grade": "A", "ok": True,
             "error": "", "out_subdir": "old", "lr_total_e": 1,
             "sr_total_e": 2, "flux_ratio_sr_over_lr": 2},
        ], catalog_runner.MANIFEST_COLS)
        catalog_runner.write_manifest_upsert(str(manifest), [
            {"id": "new", "ra": 3.0, "dec": 4.0, "grade": "B", "ok": True,
             "error": "", "out_subdir": "new", "lr_total_e": 2,
             "sr_total_e": 2, "flux_ratio_sr_over_lr": 1},
        ], catalog_runner.MANIFEST_COLS)

        rows = read_eval_catalog(str(manifest))
        assert [r["id"] for r in rows] == ["old", "new"]


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

    def test_reuses_existing_band_fits_without_fetch(self, tmp_path, monkeypatch):
        from euclid_polish.web.helpers import jobs_impl

        h = w = 12
        out_dir = str(tmp_path / "obj")
        os.makedirs(out_dir, exist_ok=True)
        for band_name in Config.LR_INPUT_BAND_NAMES:
            band = Config.get_band(band_name)
            data = np.ones((h, w), dtype=np.float32)
            hdr = fits.Header()
            hdr["MAGZERO"] = band.sim_zeropoint_e
            fits.PrimaryHDU(data, header=hdr).writeto(
                os.path.join(out_dir, f"{band_name}.fits"))

        monkeypatch.setattr(
            jobs_impl, "fetch_cutout_at",
            lambda *a, **k: pytest.fail("cached band FITS should be reused"))

        def fake_reconstruct(model, lr_cube):
            sr = np.ones((2 * h, 2 * w, lr_cube.shape[-1]), dtype=np.float32)
            return lr_cube[..., 0], sr

        monkeypatch.setattr(jobs_impl, "reconstruct", fake_reconstruct)

        res = jobs_impl.reconstruct_cutout_at(
            model=None, ra=12.3, dec=-4.5, cutout_size_vis_pixels=h,
            out_dir=out_dir, render=False, checkpoint_dir="ckpt-x",
        )

        assert os.path.isfile(res["stack_fits_path"])
        assert os.path.isfile(res["sr_fits_path"])


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
        # Local run forms (grouped + zoobot + lens-finder) + the catalog-fetch
        # + results controls are present (no FASRC step cards).
        assert "runGroupedBtn" in body and "runZoobotBtn" in body
        assert "runLensfinderBtn" in body
        # The single-grade form and all stamp-size / model knobs were removed —
        # sizes are fixed (53² LR / 106² SR·HR) and the modes auto-resolve.
        assert "runEvalBtn" not in body and "evGrade" not in body
        for gone in ("gpCutout", "gpStampM", "zbTree", "lfHeadsDir"):
            assert gone not in body
        assert "jobPanel" in body
        assert "fetchCatBtn" in body and "runSelect" not in body
        assert "Shared evaluation store" in body
        # PCA shown in 3D and 2D side by side (MDS dropped — identical metric);
        # the 2D plot is JS-drawn in the same panel, with group/HR/arrows toggles.
        assert "space3dPanel" in body and "space3dPca" in body
        assert "space3dPca2d" in body and "space3dMds" not in body
        assert "space3dFilters" in body and "space3dHrToggle" in body
        assert "space3dArrowsToggle" in body
        assert "morphImg" not in body          # server PNG replaced by JS 2D PCA
        for group in ("A", "B", "C", "syn-lens", "syn-gal"):
            assert f'data-space3d-group="{group}"' in body

    def test_runs_api_empty(self, client, tmp_path, monkeypatch):
        # Isolate the shared store to a fresh dir; otherwise this reads whatever
        # data/eval_results happens to hold on the dev machine.
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        r = client.get("/api/evaluation/runs")
        assert r.status_code == 200
        j = r.get_json()
        assert j["run"] == "eval_results"
        assert j["rows"] == []

    def test_runs_api_uses_shared_root(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        os.makedirs(Config.EVAL_RESULTS_DIR, exist_ok=True)
        with open(os.path.join(Config.EVAL_RESULTS_DIR, "manifest.csv"), "w") as f:
            f.write("id,ra,dec,grade,ok,error,out_subdir,flux_ratio_sr_over_lr\n")
            f.write("lensA,1,2,A,True,,lensA,1.02\n")
        r = client.get("/api/evaluation/runs")
        assert r.status_code == 200
        j = r.get_json()
        assert j["run"] == "eval_results"
        assert j["rows"][0]["id"] == "lensA"
        assert j["n_ok"] == 1

    def test_eval_files_traversal_blocked(self, client):
        assert client.get("/eval-files/../../etc/passwd").status_code == 403

    def test_run_grouped_spawns_job(self, client, tmp_path, monkeypatch):
        from euclid_polish.eval import grouped_runner
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        captured = {}
        monkeypatch.setattr(grouped_runner, "run_grouped_analysis",
                            lambda **k: captured.update(k) or {"n": 0})
        r = client.post("/api/evaluation/run-grouped",
                        data={"run_name": "a1", "n": "3"})
        assert r.status_code == 200 and r.get_json()["job_id"]
        assert captured["out_dir"] == Config.EVAL_RESULTS_DIR
        assert captured["n"] == 3
        assert client.post("/api/evaluation/run-grouped",
                           data={"run_name": "../x"}).status_code == 200

    def test_run_grouped_always_includes_galaxies(self, client, tmp_path, monkeypatch):
        """Real galaxies are a fixed negative control — no UI toggle disables them."""
        from euclid_polish.eval import grouped_runner
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        captured = {}
        monkeypatch.setattr(grouped_runner, "run_grouped_analysis",
                            lambda **k: captured.update(k) or {"n": 0})
        # Even a stray galaxies=0 in the form must not turn them off.
        r = client.post("/api/evaluation/run-grouped",
                        data={"n": "3", "galaxies": "0"})
        assert r.status_code == 200
        assert captured["include_galaxies"] is True
        captured.clear()
        r = client.post("/api/evaluation/run-grouped", data={"n": "3"})
        assert r.status_code == 200
        assert captured["include_galaxies"] is True

    def test_query_galaxies_requires_login(self, client, monkeypatch):
        """The standalone galaxy step needs an authenticated session; without
        one it returns 400 rather than silently doing nothing."""
        from euclid_polish.web import euclid_session
        monkeypatch.setattr(euclid_session, "catalog", lambda: None)
        r = client.post("/api/evaluation/query-galaxies", data={"n_galaxies": "5"})
        assert r.status_code == 400
        assert "log in" in r.get_json()["error"].lower()

    def test_query_galaxies_spawns_job_with_session(self, client, tmp_path, monkeypatch):
        """When logged in, the step spawns a job that runs galaxy_catalog.build
        with the WebUI's session client and this step's own n_galaxies."""
        import time as _time

        from euclid_polish.eval import catalog_runner, galaxy_catalog
        from euclid_polish.web import euclid_session
        sentinel = object()
        monkeypatch.setattr(euclid_session, "catalog", lambda: sentinel)
        lenses = tmp_path / "lenses.csv"
        lenses.write_text("id,ra,dec,grade\nL1,10.0,-5.0,A\n")
        monkeypatch.setattr(catalog_runner, "default_catalog_path",
                            lambda: str(lenses))
        monkeypatch.setattr(galaxy_catalog, "default_out_csv",
                            lambda: str(tmp_path / "galaxies.csv"))
        captured = {}

        def fake_build(out_csv=None, *, n_galaxies, lens_catalog_path,
                       client=None, log=None, **kw):
            captured.update(n_galaxies=n_galaxies, client=client,
                            lens=lens_catalog_path)
            return out_csv, n_galaxies

        monkeypatch.setattr(galaxy_catalog, "build", fake_build)
        r = client.post("/api/evaluation/query-galaxies", data={"n_galaxies": "9"})
        assert r.status_code == 200 and r.get_json()["job_id"]
        # The build runs in a background thread; give it a moment to land.
        for _ in range(100):
            if captured:
                break
            _time.sleep(0.02)
        assert captured["n_galaxies"] == 9
        assert captured["client"] is sentinel
        assert captured["lens"] == str(lenses)

    def test_transformation_404_then_renders(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        run = Config.EVAL_RESULTS_DIR
        os.makedirs(run, exist_ok=True)
        assert client.get("/api/evaluation/transformation").status_code == 404
        assert client.get("/api/evaluation/transformation?run=../e").status_code == 400
        with open(os.path.join(run, "manifest.csv"), "w") as f:
            f.write("id,ra,dec,grade,ok,error,out_subdir,lr_total_e,sr_total_e,"
                    "flux_ratio_sr_over_lr,psnr_lr_hr,psnr_sr_hr\n")
            for g in ("A", "B", "C"):
                f.write(f"{g}1,1,2,{g},True,,{g}1,1e6,1e6,1.0,,\n")
            f.write("s1,,,syn-gal,True,,s1,1e6,1e6,0.9,80,95\n")
            f.write("s2,,,syn-lens,True,,s2,1e6,1e6,0.95,82,99\n")
        r = client.get("/api/evaluation/transformation")
        assert r.status_code == 200 and r.mimetype == "image/png"
        assert os.path.isfile(os.path.join(run, "transformation_summary.png"))

    def test_morphology_404_without_manifest(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        os.makedirs(Config.EVAL_RESULTS_DIR)
        assert client.get("/api/evaluation/morphology").status_code == 404
        assert client.get("/api/evaluation/morphology?run=../e").status_code == 400

    def test_morphology_renders_from_manifest(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        run = Config.EVAL_RESULTS_DIR
        os.makedirs(run, exist_ok=True)
        with open(os.path.join(run, "morphology_manifest.csv"), "w") as f:
            f.write("id,l2_before_after,cosine_before_after,"
                    "pearson_before_after,mode,has_hr\n")
            for i in range(5):
                f.write(f"obj{i},{10+i},{0.1+i*0.01},{0.8+i*0.01},"
                        f"representation,False\n")
        r = client.get("/api/evaluation/morphology")
        assert r.status_code == 200 and r.mimetype == "image/png"
        assert os.path.isfile(os.path.join(run, "morphology_summary.png"))

    def test_morphology_embedding_endpoint(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        run = Config.EVAL_RESULTS_DIR
        os.makedirs(run, exist_ok=True)
        with open(os.path.join(run, "manifest.csv"), "w") as f:
            f.write("id,grade,ok\nobj0,A,True\nobj1,syn-gal,True\n")
        with open(os.path.join(run, "zoobot_predictions.csv"), "w") as f:
            f.write("id_str,feat_0,feat_1,feat_2,feat_3\n")
            f.write("obj0__before,0,0,0,0\n")
            f.write("obj0__after,1,0,0,0\n")
            f.write("obj1__before,0,1,0,0\n")
            f.write("obj1__after,0,2,0,1\n")
            f.write("obj1__hr,0,3,0,1\n")

        r = client.get("/api/evaluation/morphology-embedding")
        assert r.status_code == 200
        j = r.get_json()
        assert j["run"] == "eval_results"
        assert {p["view"] for p in j["points"]} == {"before", "after", "hr"}
        assert len(j["embeddings"]["pca"]["points"]) == len(j["points"])
        assert len(j["embeddings"]["mds"]["points"]) == len(j["points"])
        assert all(len(p["xyz"]) == 3 for p in j["embeddings"]["pca"]["points"])
        assert j["embeddings"]["pca"]["variance_pct"][0] > 0
        assert {"from", "to", "kind"} <= set(j["edges"][0])

        assert client.get("/api/evaluation/morphology-embedding?run=../x").status_code == 400

    def test_morphology_embedding_endpoint_404_without_predictions(
            self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        os.makedirs(Config.EVAL_RESULTS_DIR)
        assert client.get("/api/evaluation/morphology-embedding").status_code == 404

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
        from euclid_polish.eval import lens_catalog

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
        from euclid_polish.eval import lens_catalog

        def boom(*a, **k):
            raise RuntimeError("zenodo down")
        monkeypatch.setattr(lens_catalog, "fetch", boom)
        r = client.post("/api/evaluation/fetch-catalog")
        assert r.status_code == 502
        assert "zenodo down" in r.get_json()["error"]

    def test_runs_api_reads_manifest(self, client, tmp_path, monkeypatch):
        # Redirect the results dir to a tmp tree so we don't touch real data/.
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        # Lay down a fake shared manifest and confirm it surfaces.
        run_dir = Config.EVAL_RESULTS_DIR
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "manifest.csv"), "w") as f:
            f.write("id,ra,dec,grade,ok,error,out_subdir,flux_ratio_sr_over_lr\n")
            f.write("lensA,1,2,A,True,,lensA,1.02\n")
        r = client.get("/api/evaluation/runs")
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

    def test_morphology_summary_pca_and_mds_with_hr_truth(self, tmp_path):
        # The PCA and MDS shift maps fold the HR-truth vector into the
        # projection for synthetic objects and render a ★ anchor — exercise
        # that branch end to end (predictions with a 'hr' view +
        # has_hr/closer_to_ref manifest).
        from PIL import Image

        from euclid_polish.eval import zoobot_morph as zm

        run = str(tmp_path / "run")
        os.makedirs(run, exist_ok=True)
        # group map: one synthetic galaxy + one synthetic lens (both HR).
        with open(os.path.join(run, "manifest.csv"), "w") as f:
            f.write("id,grade,ok\nsyn0,syn-gal,True\nlensA,syn-lens,True\n")
        # 3-d feature vectors; syn0 after sits between before and hr.
        with open(os.path.join(run, "zoobot_predictions.csv"), "w") as f:
            f.write("id_str,feat_0,feat_1,feat_2\n")
            f.write("syn0__before,0,0,0\n")
            f.write("syn0__after,1,1,0\n")
            f.write("syn0__hr,2,2,0\n")
            f.write("lensA__before,0,1,1\n")
            f.write("lensA__after,1,0,2\n")
        with open(os.path.join(run, "morphology_manifest.csv"), "w") as f:
            f.write("id,l2_before_after,pearson_before_after,mode,has_hr,"
                    "closer_to_ref\n")
            f.write("syn0,1.4,0.9,representation,True,True\n")
            f.write("lensA,1.7,0.5,representation,False,\n")

        out_png = str(tmp_path / "morph.png")
        assert zm.render_morphology_summary(run, out_png) == out_png
        assert os.path.isfile(out_png)
        with Image.open(out_png) as im:
            width, height = im.size
        assert width / height > 2.8

    def test_morphology_embedding_payload_has_pca_and_mds(self, tmp_path):
        from euclid_polish.eval import zoobot_morph as zm

        run = str(tmp_path / "run")
        os.makedirs(run, exist_ok=True)
        with open(os.path.join(run, "manifest.csv"), "w") as f:
            f.write("id,grade,ok\nobj0,A,True\nobj1,syn-lens,True\n")
        with open(os.path.join(run, "zoobot_predictions.csv"), "w") as f:
            f.write("id_str,feat_0,feat_1,feat_2,feat_3\n")
            f.write("obj0__before,0,0,0,0\n")
            f.write("obj0__after,1,0,0,0\n")
            f.write("obj1__before,0,1,0,0\n")
            f.write("obj1__after,0,2,0,1\n")
            f.write("obj1__hr,0,3,0,1\n")

        payload = zm.morphology_embedding_payload(run)
        assert payload is not None
        assert payload["count"] == 5
        assert set(payload["embeddings"]) == {"pca", "mds", "pca_lr", "pca_srhr"}
        # The joint fits span every view and stay row-aligned with payload points.
        for method in ("pca", "mds"):
            points = payload["embeddings"][method]["points"]
            assert [p["key"] for p in points] == [p["key"] for p in payload["points"]]
            assert all(len(p["xyz"]) == 3 for p in points)
            assert np.asarray([p["xyz"] for p in points]).shape == (5, 3)
        # The LR fit holds only `before` views; the SR+HR fit only `after`/`hr`.
        lr = payload["embeddings"]["pca_lr"]
        assert {p["view"] for p in lr["points"]} == {"before"}
        assert len(lr["points"]) == 2
        assert all(len(p["xyz"]) == 3 for p in lr["points"])
        assert lr["variance_pct"][0] > 0
        srhr = payload["embeddings"]["pca_srhr"]
        assert {p["view"] for p in srhr["points"]} == {"after", "hr"}
        assert len(srhr["points"]) == 3
        assert all(len(p["xyz"]) == 3 for p in srhr["points"])
        assert srhr["variance_pct"][0] > 0
        assert payload["edges"] == [
            {"from": "obj0__before", "to": "obj0__after", "kind": "before-after"},
            {"from": "obj1__before", "to": "obj1__after", "kind": "before-after"},
            {"from": "obj1__after", "to": "obj1__hr", "kind": "after-hr"},
        ]

    def test_morphology_embedding_payload_refits_on_selected_groups(self, tmp_path):
        from euclid_polish.eval import zoobot_morph as zm

        run = str(tmp_path / "run")
        os.makedirs(run, exist_ok=True)
        with open(os.path.join(run, "manifest.csv"), "w") as f:
            f.write("id,grade,ok\nobjA,A,True\nobjL,syn-lens,True\n"
                    "objG,syn-gal,True\n")
        with open(os.path.join(run, "zoobot_predictions.csv"), "w") as f:
            f.write("id_str,feat_0,feat_1,feat_2,feat_3\n")
            f.write("objA__before,0,0,0,0\n")
            f.write("objA__after,1,0,0,0\n")
            f.write("objL__before,0,1,0,0\n")
            f.write("objL__after,0,2,0,1\n")
            f.write("objL__hr,0,3,0,1\n")
            f.write("objG__before,1,0,1,0\n")
            f.write("objG__after,1,0,2,0\n")
            f.write("objG__hr,1,0,3,0\n")

        full = zm.morphology_embedding_payload(run)
        assert full["count"] == 8

        # Selecting only the two synthetic classes refits on just those rows.
        sub = zm.morphology_embedding_payload(run, groups={"syn-lens", "syn-gal"})
        assert sub["count"] == 6
        assert {p["group"] for p in sub["points"]} == {"syn-lens", "syn-gal"}
        for method in ("pca", "mds", "pca_lr", "pca_srhr"):
            pts = sub["embeddings"][method]["points"]
            assert {p["group"] for p in pts} <= {"syn-lens", "syn-gal"}
        # The fit changes: PC1 variance differs from the all-class fit.
        assert sub["embeddings"]["pca"]["variance_pct"] != \
            full["embeddings"]["pca"]["variance_pct"]
        # Edges only connect surviving (selected) points — no objA edge.
        assert all(e["from"].startswith(("objL", "objG")) for e in sub["edges"])

        # A single class refits on that class alone.
        one = zm.morphology_embedding_payload(run, groups={"A"})
        assert one["count"] == 2
        assert {p["group"] for p in one["points"]} == {"A"}
        assert {p["view"] for p in one["embeddings"]["pca_lr"]["points"]} == {"before"}

        # Empty selection yields a valid empty payload (not a 404/None).
        empty = zm.morphology_embedding_payload(run, groups=set())
        assert empty is not None
        assert empty["count"] == 0
        assert empty["points"] == []
        assert empty["embeddings"]["pca"]["points"] == []


class TestLensCatalogModule:
    def test_normalize_grade_filter(self, tmp_path):
        from euclid_polish.eval import lens_catalog

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
        from euclid_polish.eval import lens_catalog

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

    def test_run_zoobot_env_missing_hint(self, client, tmp_path, monkeypatch):
        from euclid_polish.web.routes import evaluation as evmod
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        os.makedirs(Config.EVAL_RESULTS_DIR)
        monkeypatch.setattr(evmod, "_zoobot_python", lambda: None)
        r = client.post("/api/evaluation/run-zoobot", data={"run": "run1"})
        assert r.status_code == 400
        assert "Zoobot env not found" in r.get_json()["error"]

    def test_run_zoobot_spawns_job(self, client, tmp_path, monkeypatch):
        from euclid_polish.web.routes import evaluation as evmod
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        os.makedirs(Config.EVAL_RESULTS_DIR)
        monkeypatch.setattr(evmod, "_zoobot_python", lambda: "/fake/python")
        captured = {}
        monkeypatch.setattr(evmod, "_spawn_subprocess_job",
                            lambda label, cmd, result: captured.update(
                                {"label": label, "cmd": cmd, "result": result})
                            or "job1")
        r = client.post("/api/evaluation/run-zoobot", data={"run": "run1"})
        assert r.status_code == 200 and r.get_json()["job_id"] == "job1"
        assert "--run-dir" in captured["cmd"]
        assert Config.EVAL_RESULTS_DIR in captured["cmd"]

    def test_run_zoobot_missing_run_404(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        # Shared store itself missing is the only not-found state now.
        assert client.post("/api/evaluation/run-zoobot",
                           data={"run": "nope"}).status_code == 404


class TestSyntheticCutouts:
    def test_select_central_source_picks_closest_fitting(self):
        from euclid_polish.eval import synthetic_runner as sr
        srcs = [
            {"type": "galaxy", "x_pix": 128.0, "y_pix": 128.0},  # center, fits
            {"type": "galaxy", "x_pix": 130.0, "y_pix": 131.0},  # near center
            {"type": "galaxy", "x_pix": 5.0,   "y_pix": 5.0},    # edge, rejected
            {"type": "lens",   "x_pix": 128.0, "y_pix": 128.0},  # wrong type
        ]
        pick = sr.select_central_source(srcs, "galaxy", field=256, m=64)
        assert pick is not None and pick["x_pix"] == 128.0

    def test_select_central_source_can_prefer_bright_galaxy(self):
        from euclid_polish.eval import synthetic_runner as sr
        srcs = [
            {"type": "galaxy", "x_pix": 128.0, "y_pix": 128.0,
             "flux_vis_e": 50.0},
            {"type": "galaxy", "x_pix": 160.0, "y_pix": 128.0,
             "flux_vis_e": 5000.0},
            {"type": "galaxy", "x_pix": 250.0, "y_pix": 128.0,
             "flux_vis_e": 1e9},  # too close to edge for a 64px stamp
        ]
        pick = sr.select_central_source(
            srcs, "galaxy", field=256, m=64, prefer_bright=True)
        assert pick is not None
        assert pick["flux_vis_e"] == 5000.0

    def test_select_central_source_rejects_all_when_edge(self):
        from euclid_polish.eval import synthetic_runner as sr
        srcs = [{"type": "lens", "x_pix": 10.0, "y_pix": 10.0}]
        assert sr.select_central_source(srcs, "lens", field=256, m=64) is None

    def test_grouped_reuses_existing_lens_outputs_without_model_load(
            self, tmp_path, monkeypatch):
        from astropy.io import fits

        from euclid_polish.eval import catalog_runner, grouped_runner

        catalog = tmp_path / "lenses.csv"
        catalog.write_text(
            "id,ra,dec,grade\n"
            "a0,1.0,2.0,A\n"
            "b0,3.0,4.0,B\n"
            "c0,5.0,6.0,C\n")
        out = tmp_path / "run"
        for oid in ("a0", "b0", "c0"):
            d = out / oid
            d.mkdir(parents=True)
            # Above the canonical 53²/106² so enforce_object_sizes crops, not drops.
            stack = np.ones((4, 64, 64), dtype=np.float32)
            sr = np.full((4, 128, 128), 2.0, dtype=np.float32)
            fits.PrimaryHDU(stack).writeto(d / "original_stack.fits")
            fits.PrimaryHDU(sr).writeto(d / "SR.fits")

        monkeypatch.setattr(catalog_runner, "load_eval_model",
                            lambda *a, **k: pytest.fail("model should not load"))
        monkeypatch.setattr(catalog_runner, "eval_catalog_object",
                            lambda *a, **k: pytest.fail("lens should be reused"))

        res = grouped_runner.run_grouped_analysis(
            str(out), n=1, catalog_path=str(catalog), include_synthetic=False,
            log=lambda m: None)

        assert res["n_ok"] == 3
        rows = read_eval_catalog(str(out / "manifest.csv"))
        assert [r["id"] for r in rows] == ["a0", "b0", "c0"]
        # Reused FITS are held at the canonical eval geometry.
        with fits.open(out / "a0" / "original_stack.fits") as h:
            assert h[0].data.shape == (4, 53, 53)
        with fits.open(out / "a0" / "SR.fits") as h:
            assert h[0].data.shape == (4, 106, 106)

    def test_crop_stamp_hr_and_lr(self):
        import numpy as np

        from euclid_polish.eval import synthetic_runner as sr
        hr = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
        stamp = sr.crop_stamp(hr, cx=128.0, cy=100.0, m=64)
        assert stamp.shape == (64, 64)
        # center (cx,cy) maps to rows [cy-32:cy+32], cols [cx-32:cx+32]
        assert stamp[0, 0] == hr[68, 96]
        lr = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
        lstamp = sr.crop_stamp(lr, cx=64.0, cy=50.0, m=32)
        assert lstamp.shape == (32, 32)

    def test_grouped_skips_synthetic_without_records(self, tmp_path, monkeypatch):
        # No validation records → synthetic raises internally; the grouped run
        # must still finish and write a (possibly empty) manifest. grades=() so
        # there's no lens network work.
        from euclid_polish.eval import grouped_runner, synthetic_runner
        monkeypatch.setattr(synthetic_runner, "default_records_dir",
                            lambda: None)
        out = str(tmp_path / "run")
        res = grouped_runner.run_grouped_analysis(
            out, n=1, grades=(), include_synthetic=True, stamp_m=64,
            log=lambda m: None)
        assert res["groups"] == {}
        assert res["manifest"] is None or os.path.isfile(res["manifest"])

class TestCenterCropAndReuse:
    def test_center_crop_plane_and_cube(self):
        from euclid_polish.eval.catalog_runner import center_crop
        plane = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
        c = center_crop(plane, 63)
        assert c.shape == (63, 63)
        # centered: top-left pixel at offset (256-63)//2 = 96
        assert c[0, 0] == plane[96, 96]
        cube = np.zeros((4, 256, 256), dtype=np.float32)
        assert center_crop(cube, 63).shape == (4, 63, 63)
        # no-op when already small enough
        small = np.zeros((10, 10), dtype=np.float32)
        assert center_crop(small, 63).shape == (10, 10)

    def test_enforce_object_sizes_crops_then_drops(self, tmp_path):
        from euclid_polish.eval import catalog_runner as cr
        # Oversized LR/SR/HR are center-cropped to the canonical geometry.
        d = tmp_path / "ok"
        d.mkdir()
        fits.PrimaryHDU(np.ones((4, 64, 64), np.float32)).writeto(
            d / "original_stack.fits")
        fits.PrimaryHDU(np.ones((4, 128, 128), np.float32)).writeto(d / "SR.fits")
        fits.PrimaryHDU(np.ones((4, 200, 200), np.float32)).writeto(d / "HR.fits")
        assert cr.enforce_object_sizes(str(d)) is True
        with fits.open(d / "original_stack.fits") as h:
            assert h[0].data.shape == (4, cr.EVAL_LR_SIZE, cr.EVAL_LR_SIZE)
        for name in ("SR.fits", "HR.fits"):
            with fits.open(d / name) as h:
                assert h[0].data.shape == (4, cr.EVAL_HR_SIZE, cr.EVAL_HR_SIZE)
        # Already exact → second pass is a no-op that still validates True.
        assert cr.enforce_object_sizes(str(d)) is True

        # An SR smaller than the target → drop (False); no file is modified.
        small = tmp_path / "small"
        small.mkdir()
        fits.PrimaryHDU(
            np.ones((4, cr.EVAL_LR_SIZE, cr.EVAL_LR_SIZE), np.float32)).writeto(
            small / "original_stack.fits")
        fits.PrimaryHDU(np.ones((4, 64, 64), np.float32)).writeto(small / "SR.fits")
        assert cr.enforce_object_sizes(str(small)) is False
        with fits.open(small / "SR.fits") as h:
            assert h[0].data.shape == (4, 64, 64)

    def test_seed_object_from_cache(self, tmp_path):
        from euclid_polish.eval import catalog_runner
        src = tmp_path / "cache"
        out = tmp_path / "run"
        sd = catalog_runner.object_output_dir(str(src), "x0")
        os.makedirs(sd)
        fits.PrimaryHDU(np.ones((4, 8, 8), np.float32)).writeto(
            os.path.join(sd, "original_stack.fits"))
        fits.PrimaryHDU(np.ones((4, 16, 16), np.float32)).writeto(
            os.path.join(sd, "SR.fits"))
        assert catalog_runner.seed_object_from_cache(str(src), str(out), "x0")
        assert catalog_runner.can_reuse_eval_object(
            catalog_runner.object_output_dir(str(out), "x0"))
        # already present → no re-copy
        assert not catalog_runner.seed_object_from_cache(str(src), str(out), "x0")

    def test_grouped_seeds_and_crops_from_source_without_download(
            self, tmp_path, monkeypatch):
        """A fresh out_dir reuses cached cutouts from lens_source_dir, crops them
        to the canonical eval geometry, and never loads the model or downloads."""
        from euclid_polish.eval import catalog_runner, grouped_runner

        catalog = tmp_path / "lenses.csv"
        catalog.write_text("id,ra,dec,grade\na0,1.0,2.0,A\n")
        source = tmp_path / "cache"
        sd = source / "a0"
        sd.mkdir(parents=True)
        fits.PrimaryHDU(np.ones((4, 256, 256), np.float32)).writeto(
            sd / "original_stack.fits")
        fits.PrimaryHDU(np.ones((4, 512, 512), np.float32)).writeto(
            sd / "SR.fits")

        monkeypatch.setattr(catalog_runner, "load_eval_model",
                            lambda *a, **k: pytest.fail("model should not load"))
        monkeypatch.setattr(catalog_runner, "eval_catalog_object",
                            lambda *a, **k: pytest.fail("should not download"))

        out = tmp_path / "run"
        res = grouped_runner.run_grouped_analysis(
            str(out), n=1, catalog_path=str(catalog), include_synthetic=False,
            lens_source_dir=str(source), log=lambda m: None)

        assert res["n_ok"] == 1
        # Seeded cutouts are center-cropped to the canonical 53² LR / 106² SR.
        with fits.open(out / "a0" / "original_stack.fits") as h:
            assert h[0].data.shape == (4, 53, 53)
        with fits.open(out / "a0" / "SR.fits") as h:
            assert h[0].data.shape == (4, 106, 106)
        # source cache stays untouched at full size
        with fits.open(source / "a0" / "original_stack.fits") as h:
            assert h[0].data.shape == (4, 256, 256)

    def test_tqdm_progress_callback_runs(self):
        from euclid_polish.eval.progress import tqdm_progress
        cb = tqdm_progress("t")
        cb(0, 3, "a")
        cb(1, 3, "b")
        cb(3, 3, "done")         # closes the bar without raising

    def test_classical_mds_matches_naive_distances(self):
        """The memory-safe MDS must equal the naive (N,N,D) pairwise form.

        Guards the ‖xᵢ‖²+‖xⱼ‖²−2xᵢxⱼ rewrite that replaced an X[:,None,:] -
        X[None,:,:] broadcast (which spiked the embedding endpoint to GBs).
        Classical MDS is exact for Euclidean data, so the embedded pairwise
        distances must match the originals up to rotation.
        """
        from euclid_polish.eval.zoobot_morph import _classical_mds3
        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 8))
        coords, var = _classical_mds3(X)
        assert coords.shape == (40, 3)

        def _pdist(A):
            d = A[:, None, :] - A[None, :, :]
            return np.sqrt(np.sum(d * d, axis=2))

        # Top-3 components can't perfectly reconstruct 8-D distances, but the
        # leading structure must be recovered (strong correlation, no blow-up).
        orig, emb = _pdist(X), _pdist(coords)
        iu = np.triu_indices(40, 1)
        r = np.corrcoef(orig[iu], emb[iu])[0, 1]
        assert r > 0.6
        assert var[0] >= var[1] >= var[2] >= 0.0


def test_run_catalog_eval_accepts_preloaded_model(tmp_path, monkeypatch):
    """When a model is passed in, load_eval_model is NOT called and the
    supplied model object is threaded to eval_catalog_object."""
    from euclid_polish.eval import catalog_runner

    # A catalog with one object that is NOT cached (forces needs_model path).
    cat = tmp_path / "cat.csv"
    cat.write_text("id,ra,dec,grade\nobj1,10.0,-5.0,A\n")

    def _boom(*a, **k):
        raise AssertionError("load_eval_model must NOT be called when model= is supplied")
    monkeypatch.setattr(catalog_runner, "load_eval_model", _boom)

    seen = {}

    def _fake_eval_object(model, obj, out_dir, **kwargs):
        seen["model"] = model
        rec = dict.fromkeys(catalog_runner.MANIFEST_COLS, "")
        rec.update({"id": obj["id"], "ra": obj["ra"], "dec": obj["dec"],
                    "grade": obj.get("grade", ""), "ok": True,
                    "out_subdir": obj["id"]})
        return rec
    monkeypatch.setattr(catalog_runner, "eval_catalog_object", _fake_eval_object)

    sentinel = object()
    result = catalog_runner.run_catalog_eval(
        out_dir=str(tmp_path / "out"), catalog_path=str(cat), model=sentinel)
    assert seen["model"] is sentinel
    assert result["n_ok"] >= 1


def test_run_grouped_accepts_preloaded_model(tmp_path, monkeypatch):
    """A supplied model= skips load_eval_model and threads through to eval_catalog_object."""
    from euclid_polish.eval import catalog_runner, grouped_runner

    def _boom(*a, **k):
        raise AssertionError("load_eval_model must NOT be called when model= supplied")
    monkeypatch.setattr(catalog_runner, "load_eval_model", _boom)

    seen = {}

    def _fake_eval_object(model, obj, out_dir, **kwargs):
        seen["model"] = model
        rec = dict.fromkeys(catalog_runner.MANIFEST_COLS, "")
        rec.update({"id": obj["id"], "ra": obj["ra"], "dec": obj["dec"],
                    "grade": obj.get("grade", ""), "ok": True,
                    "out_subdir": obj["id"]})
        return rec
    monkeypatch.setattr(catalog_runner, "eval_catalog_object", _fake_eval_object)

    # enforce_object_sizes must pass (True) so reuse_catalog_object is called.
    monkeypatch.setattr(catalog_runner, "enforce_object_sizes",
                        lambda obj_dir, **kw: True)

    def _fake_reuse(obj, out_dir, *, grade=None, log=None):
        rec = dict.fromkeys(catalog_runner.MANIFEST_COLS, "")
        rec.update({"id": obj["id"], "ra": obj["ra"], "dec": obj["dec"],
                    "grade": grade or obj.get("grade", ""), "ok": True,
                    "out_subdir": obj["id"]})
        return rec
    monkeypatch.setattr(catalog_runner, "reuse_catalog_object", _fake_reuse)

    # A tiny lens catalog with one un-cached grade-A object; no synthetic/galaxies.
    cat = tmp_path / "lenses.csv"
    cat.write_text("id,ra,dec,grade\nlensA,10.0,-5.0,A\n")

    sentinel = object()
    result = grouped_runner.run_grouped_analysis(
        str(tmp_path / "out"), 0, catalog_path=str(cat),
        include_synthetic=False, include_galaxies=False, model=sentinel,
        log=lambda m: None)
    assert seen["model"] is sentinel
    assert result["n_ok"] >= 1


def test_galaxy_plan_reads_all_cached(monkeypatch, tmp_path):
    """Cache-only: _galaxy_plan reads every row of the galaxies.csv the
    standalone Query-galaxies step wrote — it never queries the archive."""
    import csv as _csv

    from euclid_polish.eval import galaxy_catalog, grouped_runner
    gal_csv = tmp_path / "galaxies.csv"
    with open(gal_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["id", "ra", "dec", "grade"])
        for i in range(7):
            w.writerow([f"gal_{i}", 10.0 + i * 1e-3, -5.0, "gal"])
    monkeypatch.setattr(galaxy_catalog, "default_out_csv", lambda: str(gal_csv))
    # build() must never be reached on the grouped (cache-only) path.
    monkeypatch.setattr(galaxy_catalog, "build",
                        lambda *a, **k: pytest.fail("grouped run must not query"))
    rows = grouped_runner._galaxy_plan(lambda m: None)
    assert len(rows) == 7 and all(r["grade"] == "gal" for r in rows)


def test_galaxy_plan_empty_when_no_cache(monkeypatch, tmp_path):
    from euclid_polish.eval import galaxy_catalog, grouped_runner
    monkeypatch.setattr(galaxy_catalog, "default_out_csv",
                        lambda: str(tmp_path / "absent.csv"))
    msgs = []
    rows = grouped_runner._galaxy_plan(msgs.append)
    assert rows == []                              # galaxies must not kill A/B/C
    assert any("Query-galaxies" in m for m in msgs)   # actionable hint logged


def test_grouped_downloads_padded_size(monkeypatch, tmp_path):
    """Real cutouts are requested a few px larger than the canonical LR side, so
    the Euclid cutout service's round-down never drops them below EVAL_LR_SIZE."""
    from euclid_polish.eval import catalog_runner, grouped_runner
    cat = tmp_path / "lenses.csv"
    cat.write_text("id,ra,dec,grade\na0,1.0,2.0,A\n")
    seen = {}

    def fake_eval(model, obj, out_dir, *, cutout_size, **kw):
        seen["cutout_size"] = cutout_size
        return {"ok": False, "error": "stub"}      # not produced → harmless drop

    monkeypatch.setattr(catalog_runner, "eval_catalog_object", fake_eval)
    monkeypatch.setattr(catalog_runner, "load_eval_model", lambda *a, **k: object())
    grouped_runner.run_grouped_analysis(
        str(tmp_path / "out"), n=1, catalog_path=str(cat),
        include_synthetic=False, include_galaxies=False, log=lambda m: None)
    assert seen["cutout_size"] == grouped_runner.EVAL_LR_SIZE + grouped_runner.DOWNLOAD_PAD
