"""Tests for the catalog-based evaluation pipeline.

Covers the generic catalog reader, the ``eval_catalog`` FASRC step's
``build_command`` argv, the shared per-object ``reconstruct_cutout_at`` helper
(driven with stubbed download + model so no network / TF weights are needed),
and the read-only ``/evaluation`` routes.
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
# FASRC step
# --------------------------------------------------------------------------- #

class TestCatalogEvalStep:
    def test_registered(self):
        step = REGISTRY.get("eval_catalog")
        assert step.job_name == "eval-catalog"
        assert step.conda_env is None        # uses cluster default env
        assert not step.experimental

    def test_build_command_minimal(self):
        argv = REGISTRY.get("eval_catalog").build_command({})
        assert argv[0] == "scripts/fasrc_eval_catalog.py"
        assert "--run-name" in argv and "--cutout-size" in argv
        # No optional flags when params are absent.
        assert "--grade" not in argv and "--max-n" not in argv

    def test_build_command_full(self):
        argv = REGISTRY.get("eval_catalog").build_command({
            "run_name": "lensesA", "grade": "A", "cutout_size": 128,
            "max_n": 30, "asinh_scale": 100, "num_res_blocks": 32,
            "catalog": "data/eval_catalogs/lens_catalog/lenses.csv",
            "no_render": "true",
        })
        assert argv[argv.index("--run-name") + 1] == "lensesA"
        assert argv[argv.index("--cutout-size") + 1] == "128"
        assert argv[argv.index("--grade") + 1] == "A"
        assert argv[argv.index("--max-n") + 1] == "30"
        assert argv[argv.index("--num-res-blocks") + 1] == "32"
        assert argv[argv.index("--catalog") + 1].endswith("lenses.csv")
        assert "--no-render" in argv


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

        # Metrics dict always has the full key set (values may be None when the
        # forward residual can't be computed without a PSF).
        for k in ("residual_std_e", "residual_mae_e", "residual_rmse_e",
                  "residual_max_abs_e", "residual_chi",
                  "flux_ratio_fwd_over_lr"):
            assert k in res["metrics"]
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
        assert "evalForm" in body
        assert "/api/fasrc/hst/eval_catalog/submit" in body

    def test_runs_api_empty(self, client):
        r = client.get("/api/evaluation/runs")
        assert r.status_code == 200
        assert "runs" in r.get_json()

    def test_runs_api_missing_run_404(self, client):
        assert client.get("/api/evaluation/runs?run=nope").status_code == 404

    def test_eval_files_traversal_blocked(self, client):
        assert client.get("/eval-files/../../etc/passwd").status_code == 403

    def test_runs_api_reads_manifest(self, client, tmp_path, monkeypatch):
        # Redirect the results dir to a tmp tree so we don't touch real data/.
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        # Lay down a fake run dir with a manifest and confirm it surfaces.
        run_dir = os.path.join(Config.EVAL_RESULTS_DIR, "run1")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "manifest.csv"), "w") as f:
            f.write("id,ra,dec,grade,ok,error,out_subdir,residual_chi\n")
            f.write("lensA,1,2,A,True,,lensA,1.5\n")
        r = client.get("/api/evaluation/runs?run=run1")
        assert r.status_code == 200
        rows = r.get_json()["rows"]
        assert rows and rows[0]["id"] == "lensA"
