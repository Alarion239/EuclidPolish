"""Smoke tests for the localhost web UI.

We don't actually start an HTTP server — Flask's ``test_client``
dispatches requests directly into the app. The tests check that every
route renders, that the job tracker accepts and runs a synthetic task,
and that the static PNG server refuses path-traversal attempts.
"""

from __future__ import annotations

import os
import time

import pytest

from euclid_polish.web.app import create_app
from euclid_polish.web.jobs import REGISTRY


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Pages render
# ---------------------------------------------------------------------------

def test_dashboard_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.data.decode()
    assert "EuclidPolish" in body
    assert "Dashboard" in body
    assert "Catalog" in body and "PSFs" in body


def test_catalog_page_renders(client):
    r = client.get("/catalog")
    assert r.status_code == 200
    assert b"Star catalog" in r.data


def test_psfs_page_renders(client):
    r = client.get("/psfs")
    assert r.status_code == 200
    # All four band names appear in the inventory table.
    body = r.data.decode()
    for name in ("VIS", "Y_E", "J_E", "H_E"):
        assert name in body


def test_sky_page_renders(client):
    r = client.get("/sky")
    assert r.status_code == 200
    # Form for generation
    assert b"Generate clean" in r.data
    assert b"Forward model" in r.data


def test_visualization_page_renders(client):
    r = client.get("/visualization")
    assert r.status_code == 200
    assert b"Quick lens demo" in r.data
    # The gallery is the central viz pane on /visualization.
    assert b"data/vis/" in r.data


def test_cutouts_page_renders(client):
    r = client.get("/cutouts")
    assert r.status_code == 200
    body = r.data.decode()
    assert "Cutouts" in body
    # All four bands appear as checkboxes
    for name in ("VIS", "Y_E", "J_E", "H_E"):
        assert f'value="{name}"' in body


def test_training_page_renders(client):
    r = client.get("/training")
    assert r.status_code == 200
    body = r.data.decode()
    assert "Training" in body
    assert "Evaluate" in body
    assert "Plot training log" in body


def test_inference_page_renders(client):
    r = client.get("/inference")
    assert r.status_code == 200
    assert b"Reconstruct" in r.data


# ---------------------------------------------------------------------------
# Endpoint smoke tests: every POST endpoint accepts a valid payload and
# returns a job_id. We don't wait for completion — that's covered by the
# job-tracker tests above.
# ---------------------------------------------------------------------------

def test_post_catalog_integrity_returns_job_id(client):
    r = client.post("/catalog/integrity", data={"output_dir": "/tmp/no_such"})
    assert r.status_code == 200
    assert "job_id" in r.get_json()


def test_post_cutouts_download_requires_bands(client):
    r = client.post("/cutouts/download", data={"cutout_size_vis_pixels": 64})
    assert r.status_code == 400
    body = r.get_json()
    assert body.get("ok") is False


def test_post_cutouts_download_accepts_multi_band(client):
    r = client.post("/cutouts/download", data={
        "bands": ["VIS", "Y_E"],
        "cutout_size_vis_pixels": 64,
        "max_workers": 2,
    })
    assert r.status_code == 200
    assert "job_id" in r.get_json()


def test_post_psfs_extract_accepts_band(client):
    r = client.post("/psfs/extract", data={
        "band": "VIS", "num_stars": 8, "cutout_size": 65,
    })
    assert r.status_code == 200
    assert "job_id" in r.get_json()


def test_post_psfs_visualize_returns_job_id(client):
    r = client.post("/psfs/visualize", data={"band": "VIS"})
    assert r.status_code == 200
    assert "job_id" in r.get_json()


def test_post_training_plot_log_returns_job_id(client):
    r = client.post("/training/plot-log", data={"checkpoint_dir": "/tmp/nope"})
    assert r.status_code == 200
    assert "job_id" in r.get_json()


def test_post_inference_reconstruct_returns_job_id(client):
    r = client.post("/inference/reconstruct", data={
        "checkpoint_dir": "/tmp/nope", "subset": "validate", "n_images": 2,
    })
    assert r.status_code == 200
    assert "job_id" in r.get_json()


def test_post_viz_star_positions_returns_job_id(client):
    r = client.post("/visualization/star-positions",
                    data={"output_dir": "/tmp"})
    assert r.status_code == 200
    assert "job_id" in r.get_json()


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def test_job_tick_updates_progress_fields():
    """Calling cap.tick() during a job updates ``progress_*`` fields."""
    def _target(cap):
        for i in range(5):
            cap.tick(i + 1, 5, f"step {i+1}")
        return {"ok": True}
    job_id = REGISTRY.spawn("tick test", _target)
    deadline = time.time() + 2.0
    while time.time() < deadline:
        job = REGISTRY.get(job_id)
        assert job is not None
        if job.status != "running":
            break
        time.sleep(0.05)
    assert job.status == "done"
    assert job.progress_current == 5
    assert job.progress_total   == 5
    assert "step 5" in job.progress_label


def test_job_to_dict_exposes_progress():
    def _target(cap):
        cap.tick(3, 10, "mid")
        # leave running so we can read progress
        import time as _t
        _t.sleep(0.05)
        return None

    job_id = REGISTRY.spawn("progress test", _target)
    # Wait briefly for the tick to land
    time.sleep(0.1)
    job = REGISTRY.get(job_id)
    d = job.to_dict()
    assert "progress" in d
    assert d["progress"]["current"] >= 0
    assert d["progress"]["total"]   >= 0


# ---------------------------------------------------------------------------
# JSON status endpoints
# ---------------------------------------------------------------------------

def test_api_status_returns_all_sections(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    payload = r.get_json()
    assert set(payload.keys()) == {"catalog", "psfs", "tfrecords", "checkpoints"}
    # PSF section contains all four bands.
    band_names = {b["name"] for b in payload["psfs"]["bands"]}
    assert band_names == {"VIS", "Y_E", "J_E", "H_E"}


def test_api_jobs_initially_returns_a_list(client):
    r = client.get("/api/jobs")
    assert r.status_code == 200
    assert isinstance(r.get_json(), list)


def test_api_job_unknown_id_404(client):
    r = client.get("/api/jobs/deadbeef")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Job tracker runs a synthetic task end-to-end
# ---------------------------------------------------------------------------

def test_job_runs_and_captures_stdout():
    """Spawn a task that prints + returns; check status flips to ``done``."""
    def _target(cap):
        print("hello from job")
        return {"ok": True}

    job_id = REGISTRY.spawn("test", _target)
    # Wait up to 2 s for the daemon thread to finish.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        job = REGISTRY.get(job_id)
        assert job is not None
        if job.status != "running":
            break
        time.sleep(0.05)
    assert job.status == "done", f"got {job.status}: {job.error}"
    assert "hello from job" in job.log
    assert job.result == {"ok": True}


def test_failed_job_records_error():
    def _bad(_cap):
        raise RuntimeError("boom")

    job_id = REGISTRY.spawn("bad", _bad)
    deadline = time.time() + 2.0
    while time.time() < deadline:
        job = REGISTRY.get(job_id)
        if job and job.status != "running":
            break
        time.sleep(0.05)
    assert job.status == "failed"
    assert "boom" in (job.error or "")


# ---------------------------------------------------------------------------
# Cutout visualization
# ---------------------------------------------------------------------------

def test_cutouts_gallery_page_renders(client):
    """The per-band gallery page renders even with no cutouts on disk."""
    r = client.get("/cutouts/VIS")
    assert r.status_code == 200
    body = r.data.decode()
    assert "VIS cutouts" in body
    # Either there are thumbnails or the "no cutouts on disk" notice fires.
    assert "gallery" in body or "No cutouts" in body


def test_cutouts_gallery_unknown_band_404(client):
    r = client.get("/cutouts/NOPE")
    assert r.status_code == 404


def test_cutout_image_unknown_band_404(client):
    r = client.get("/cutout-image/NOPE/star_0000_512.fits")
    assert r.status_code == 404


def test_cutout_image_rejects_bad_filename(client):
    # The route's <path:...> converter forwards the literal filename;
    # our regex must reject anything that isn't a plain *.fits leaf.
    r = client.get("/cutout-image/VIS/not_a_fits.png")
    assert r.status_code == 400


def test_cutout_image_rejects_bad_size(client):
    r = client.get("/cutout-image/VIS/anything.fits?size=4")
    assert r.status_code == 400


def test_cutout_image_renders_real_fits(client, tmp_path):
    """Drop a tiny FITS into the VIS cutout dir and round-trip a render."""
    import numpy as np
    from astropy.io import fits
    from euclid_polish.config import Config
    band_dir = Config.cutout_dir_for_band(
        "VIS", root=os.path.join(Config.DEFAULT_OUTPUT_DIR, "cutouts"),
    )
    os.makedirs(band_dir, exist_ok=True)
    fname = "test_cutout_999.fits"
    full = os.path.join(band_dir, fname)
    # Synthetic 16×16 frame; one bright pixel so asinh stretch has content.
    arr = np.zeros((16, 16), dtype=np.float32)
    arr[8, 8] = 5000.0
    fits.PrimaryHDU(arr).writeto(full, overwrite=True)
    try:
        r = client.get(f"/cutout-image/VIS/{fname}?size=64")
        assert r.status_code == 200
        assert r.headers["Content-Type"] == "image/png"
        assert len(r.data) > 0
    finally:
        os.remove(full)


# ---------------------------------------------------------------------------
# Live view renderers (PNG)
# ---------------------------------------------------------------------------

def test_view_psfs_all_returns_png(client):
    r = client.get("/view/psfs?band=all")
    assert r.status_code == 200
    assert r.headers["Content-Type"] == "image/png"
    assert len(r.data) > 100


def test_view_psfs_per_band_returns_png(client):
    r = client.get("/view/psfs?band=VIS")
    assert r.status_code == 200
    assert r.headers["Content-Type"] == "image/png"


def test_view_psfs_unknown_band_404(client):
    r = client.get("/view/psfs?band=NOPE")
    assert r.status_code == 404


def test_view_catalog_positions_returns_png(client):
    r = client.get("/view/catalog?view=positions")
    # 200 if a catalog exists, else 404 — both are valid for the route.
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert r.headers["Content-Type"] == "image/png"


def test_view_catalog_unknown_view_400(client):
    r = client.get("/view/catalog?view=bogus")
    # 400 (bad view) when a catalog is present; 404 (no catalog) is also acceptable.
    assert r.status_code in (400, 404)


def test_view_sky_invalid_subset_400(client):
    r = client.get("/view/sky?subset=foo&kind=clean&band=VIS&i=0")
    assert r.status_code == 400


def test_view_sky_invalid_kind_400(client):
    r = client.get("/view/sky?subset=train&kind=foo&band=VIS&i=0")
    assert r.status_code == 400


def test_view_sky_invalid_band_400(client):
    r = client.get("/view/sky?subset=train&kind=clean&band=BOGUS&i=0")
    assert r.status_code == 400


def test_api_sky_totals_returns_json(client):
    r = client.get("/api/sky/totals")
    assert r.status_code == 200
    body = r.get_json()
    assert set(body.keys()) >= {"clean_train", "clean_validate", "dirty_train", "dirty_validate"}


# ---------------------------------------------------------------------------
# /hst-pairs (HST Catalog) — same viewer as /sky over FASRC-cached records
# ---------------------------------------------------------------------------

def test_hst_pairs_page_renders(client):
    r = client.get("/hst-pairs")
    assert r.status_code == 200
    body = r.data.decode()
    assert "HST Catalog" in body
    assert "Sync from FASRC" in body
    # Toolbar bands match /sky's set so the same chip layout works.
    for n in ("VIS", "Y_E", "J_E", "H_E", "color"):
        assert n in body


def test_view_hst_pair_invalid_subset_400(client):
    r = client.get("/view/hst-pair?subset=foo&kind=clean&band=VIS&i=0")
    assert r.status_code == 400


def test_view_hst_pair_invalid_kind_400(client):
    r = client.get("/view/hst-pair?subset=validate&kind=foo&band=VIS&i=0")
    assert r.status_code == 400


def test_view_hst_pair_invalid_band_400(client):
    r = client.get("/view/hst-pair?subset=validate&kind=clean&band=BOGUS&i=0")
    assert r.status_code == 400


def test_view_hst_pair_404_when_not_synced(client):
    """No local cache file → ``_render_sky_record_png`` aborts 404."""
    r = client.get("/view/hst-pair?subset=validate&kind=clean&band=VIS&i=0")
    assert r.status_code == 404


def test_api_hst_pairs_totals_returns_json_with_all_six_files(client):
    r = client.get("/api/hst-pairs/totals")
    assert r.status_code == 200
    body = r.get_json()
    # Even when the cache is empty, every key must be present so the JS
    # can build its index labels deterministically. Counts can be 0.
    assert set(body.keys()) == {
        "clean_train", "clean_validate",
        "dirty_train", "dirty_validate",
        "hr_train",    "hr_validate",
    }
    for v in body.values():
        assert isinstance(v, int)
        assert v >= 0


def test_api_hst_pairs_status_lists_cache_dir(client):
    r = client.get("/api/hst-pairs/status")
    assert r.status_code == 200
    body = r.get_json()
    assert "dir" in body and "files" in body
    # The dir must live under the local FASRC cache, never some arbitrary
    # path — that's the contract the sync route depends on too.
    assert "_fasrc_cache" in body["dir"]


def test_api_hst_pairs_sync_defaults_to_validate_only(client, monkeypatch):
    """No ``include_train`` form arg → only the three validate files
    are requested. This guards the "don't accidentally pull 25 GB"
    invariant — if a refactor flips the default, this test catches it."""
    requested: list = []

    class _R:
        ok = True
        local_path = "/tmp/nope"      # never actually opened in this test
        size_bytes = 0
        from_cache = False
        error = None

    def _fake_fetch(remote_path, *, force=False, max_bytes=None, **_):
        requested.append((remote_path, force, max_bytes))
        return _R()

    monkeypatch.setattr(
        "euclid_polish.web.fasrc_fetcher.fetch_one_file", _fake_fetch,
    )
    r = client.post("/api/hst-pairs/sync")
    assert r.status_code == 200
    data = r.get_json()
    assert data["include_train"] is False
    requested_names = {p.rsplit("/", 1)[-1] for (p, _, _) in requested}
    assert requested_names == {
        "clean_validate.tfrecord",
        "dirty_validate.tfrecord",
        "hr_validate.tfrecord",
    }
    # Every fetch must be force=True (the user explicitly clicked Sync)
    # and over the default 50 MB cap (these files are big).
    for (_path, force, max_bytes) in requested:
        assert force is True
        assert max_bytes is not None and max_bytes > 50 * 1024 * 1024


def test_api_hst_pairs_sync_include_train_pulls_six_files(client, monkeypatch):
    """``include_train=true`` adds the three train files on top."""
    requested: list = []

    class _R:
        ok = True
        local_path = "/tmp/nope"
        size_bytes = 0
        from_cache = False
        error = None

    def _fake_fetch(remote_path, *, force=False, max_bytes=None, **_):
        requested.append(remote_path)
        return _R()

    monkeypatch.setattr(
        "euclid_polish.web.fasrc_fetcher.fetch_one_file", _fake_fetch,
    )
    r = client.post("/api/hst-pairs/sync",
                    data={"include_train": "true"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["include_train"] is True
    names = {p.rsplit("/", 1)[-1] for p in requested}
    assert names == {
        "clean_validate.tfrecord", "dirty_validate.tfrecord",
        "hr_validate.tfrecord",    "clean_train.tfrecord",
        "dirty_train.tfrecord",    "hr_train.tfrecord",
    }


def test_api_hst_pairs_sync_surfaces_fetch_errors_per_file(client, monkeypatch):
    """If one file fails, the response still lists it (ok=False, error
    set) so the UI can show partial-success status."""
    class _OK:
        ok = True
        local_path = "/tmp/ok"
        size_bytes = 100
        from_cache = False
        error = None

    class _BAD:
        ok = False
        local_path = None
        size_bytes = None
        from_cache = False
        error = "rsync exit 23"

    def _fake_fetch(remote_path, *, force=False, max_bytes=None, **_):
        # First request "fails", rest succeed.
        if remote_path.endswith("clean_validate.tfrecord"):
            return _BAD()
        return _OK()

    monkeypatch.setattr(
        "euclid_polish.web.fasrc_fetcher.fetch_one_file", _fake_fetch,
    )
    r = client.post("/api/hst-pairs/sync")
    assert r.status_code == 200
    data = r.get_json()
    # Overall ok=True as long as ANY file succeeded — partial success
    # is the common case (e.g. train file present, validate not yet).
    assert data["ok"] is True
    assert data["files"]["clean_validate"]["ok"] is False
    assert data["files"]["clean_validate"]["error"] == "rsync exit 23"
    assert data["files"]["dirty_validate"]["ok"] is True


def test_view_training_log_404_when_missing(client):
    r = client.get("/view/training-log?checkpoint_dir=/tmp/nope_dir")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Static PNG server
# ---------------------------------------------------------------------------

def test_serve_vis_rejects_path_traversal(client):
    """A URL like /vis/../etc/passwd must be 403, not 200."""
    r = client.get("/vis/../etc/passwd")
    # Flask normalises the path so the route may not match → 404.
    # Either way it must not return the file.
    assert r.status_code in (403, 404)


def test_serve_vis_returns_existing_png(client, tmp_path):
    """If a PNG exists under data/vis, we can fetch it through the server."""
    from euclid_polish.config import Config
    # Use an existing demo PNG if present; otherwise drop a tiny test one.
    test_png = os.path.join(Config.VIS_DIR, "test_serve.png")
    os.makedirs(os.path.dirname(test_png), exist_ok=True)
    # Minimal valid 1x1 PNG.
    minimal_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\rIDATx\x9cc\xfc\xcf\xc0\x00\x00\x00\x03\x00\x01\x9b\xc8"
        b"\x9d\xed\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    with open(test_png, "wb") as fh:
        fh.write(minimal_png)
    try:
        r = client.get("/vis/test_serve.png")
        assert r.status_code == 200
        assert r.headers["Content-Type"] == "image/png"
    finally:
        os.remove(test_png)


def test_serve_vis_unknown_file_404(client):
    r = client.get("/vis/does_not_exist.png")
    assert r.status_code == 404
