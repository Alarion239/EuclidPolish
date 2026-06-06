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

from euclid_polish.config import Config
from euclid_polish.web.app import create_app
from euclid_polish.web.jobs import REGISTRY


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_view_training_log_empty_is_404_not_500(client, tmp_path, monkeypatch):
    """An empty/header-only training log must 404 (placeholder), never 500."""
    ckpt = tmp_path / "ckpt" / "wdsr"
    ckpt.mkdir(parents=True)
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR", str(ckpt))
    # RELATIVE VIS_DIR (like the real "./data/vis"): exercises the bug where
    # Flask's send_file resolves a relative path against app.root_path
    # (euclid_polish/web/) instead of the CWD → a 500 on a file that exists.
    monkeypatch.setattr(Config, "VIS_DIR", os.path.relpath(str(tmp_path / "vis")))
    header = ("step,wall_time,loss,psnr_stretched,psnr_raw,"
              "save_best_score,combined_loss,is_baseline\n")

    # header-only (no data rows yet) → 404, not a 500 traceback
    (ckpt / "training_log.csv").write_text(header)
    r = client.get(f"/view/training-log?checkpoint_dir={ckpt}&force=1")
    assert r.status_code == 404

    # once a data row exists → 200 PNG
    (ckpt / "training_log.csv").write_text(
        header + "1000,1.0,0.04,46.6,39.9,46.6,0.003,\n")
    r = client.get(f"/view/training-log?checkpoint_dir={ckpt}&force=1")
    assert r.status_code == 200
    assert r.content_type.startswith("image/png")

    # a later transient truncation still serves the last good render (no 500)
    (ckpt / "training_log.csv").write_text(header)
    r = client.get(f"/view/training-log?checkpoint_dir={ckpt}&force=1")
    assert r.status_code == 200


def test_delete_model_wipes_local_keeps_tracking(client, tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt" / "wdsr"
    ckpt.mkdir(parents=True)
    (ckpt / "ckpt-5.index").write_bytes(b"x")
    (ckpt / "training_log.csv").write_text("step\n1\n")
    (ckpt / "loss_best").mkdir()
    (ckpt / "loss_best" / "ckpt-1.index").write_bytes(b"y")
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR", str(ckpt))
    # A tracking store with content that must NOT be touched.
    trk = tmp_path / "tracking" / "current"
    trk.mkdir(parents=True)
    (trk / "keep.txt").write_text("important")
    monkeypatch.setattr(Config, "TRACKING_DIR", str(tmp_path / "tracking"))

    r = client.post("/api/fasrc/delete-model", data={"confirm": "yes"})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json()["results"]["local"]["ok"] is True
    # ckpt dir recreated empty; tracking untouched
    assert ckpt.is_dir() and list(ckpt.iterdir()) == []
    assert (trk / "keep.txt").read_text() == "important"


def test_delete_model_requires_confirm(client, tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt" / "wdsr"
    ckpt.mkdir(parents=True)
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR", str(ckpt))
    r = client.post("/api/fasrc/delete-model", data={})
    assert r.status_code == 400


def test_delete_model_refused_when_job_active(client, tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt" / "wdsr"
    ckpt.mkdir(parents=True)
    (ckpt / "ckpt-5.index").write_bytes(b"x")
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR", str(ckpt))
    from euclid_polish.web import fasrc_jobs
    fasrc_jobs.DB.insert("12345", label="t", params={}, script_path="s",
                         log_path="l", err_path="e")  # inserts as PENDING
    r = client.post("/api/fasrc/delete-model", data={"confirm": "yes"})
    assert r.status_code == 400
    assert "cancel" in r.get_json()["error"].lower()
    assert (ckpt / "ckpt-5.index").exists()      # nothing deleted


def test_delete_model_refuses_unsafe_local_path(client, tmp_path, monkeypatch):
    # A path without "ckpt" must be refused so we can't wipe an arbitrary dir.
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "f.bin").write_bytes(b"x")
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR", str(weights))
    r = client.post("/api/fasrc/delete-model", data={"confirm": "yes"})
    assert r.get_json()["results"]["local"]["ok"] is False
    assert (weights / "f.bin").exists()          # untouched


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
    assert b"sky positions" in r.data


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
    # Synthetic generation is now a FASRC step card (mounted by step_id),
    # not the old local generate/forward forms.
    assert b"Synthetic training pairs" in r.data
    assert b"synthetic_generate" in r.data


def test_visualization_page_renders(client):
    r = client.get("/visualization")
    assert r.status_code == 200
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
    # The single FASRC training card mounts the "train" step.
    assert 'data-step-id="train"' in body or 'step_id="train"' in body \
        or "Training on FASRC" in body
    # Legacy preset + local-training sections were removed.
    assert "run_pipeline.py</code> presets" not in body
    assert "Local training (deprecated)" not in body


def test_inference_page_renders(client):
    r = client.get("/inference")
    assert r.status_code == 200
    assert b"Reconstruct" in r.data


# ---------------------------------------------------------------------------
# Endpoint smoke tests: every POST endpoint accepts a valid payload and
# returns a job_id. We don't wait for completion — that's covered by the
# job-tracker tests above.
# ---------------------------------------------------------------------------

def test_removed_routes_are_gone(client):
    """Region-cone, standalone integrity, the old local download/extract
    routes, and the bespoke catalog-query / verify-photometry routes were all
    removed — cutout download + PSF extraction + the catalog query+verify are
    now FASRC pipeline steps submitted via /api/fasrc/hst/<step_id>/submit."""
    # 404 = no rule matches; 405 = the URL now only matches a different
    # rule (e.g. POST /cutouts/download hits the GET-only gallery route).
    # Either way the old POST endpoint is gone.
    for path in ("/catalog/query-region", "/catalog/integrity",
                 "/cutouts/download", "/psfs/extract",
                 "/catalog/query-brightest", "/cutouts/verify-photometry"):
        r = client.post(path, data={})
        assert r.status_code in (404, 405), f"{path} should be removed"


# The catalog query + photometry verify are now two separate pipeline steps
# (``euclid_query``, ``euclid_verify_photometry``); their argv construction is
# tested in test_fasrc_pipeline.py and the submit route's connection/confirm
# guards in test_step_history.py.


def test_psfs_page_reads_cache_without_rsync(client, monkeypatch):
    """Loading /psfs reads the local ePSF cache only — it must NOT rsync from
    FASRC on page load (the slow behaviour we replaced with a button)."""
    from euclid_polish.web import fasrc_fetcher
    calls = []

    def spy(*a, **k):
        calls.append((a, k))
        return fasrc_fetcher.FetchResult(ok=False)

    monkeypatch.setattr(fasrc_fetcher, "fetch_one_file", spy)
    assert client.get("/psfs").status_code == 200
    assert calls == []                         # cache-only; no fetch on load


def test_euclid_psf_sync_forces_each_band_with_larger_cap(client, monkeypatch):
    """The Synchronise button force-fetches all four bands using the larger
    ePSF pull cap, so the multi-extension VIS file (tens-to-hundreds of MB)
    isn't rejected by the generic 50 MB cap."""
    from euclid_polish.web import fasrc_fetcher
    seen = []

    def fake(remote, *, force=False, max_bytes=None, **k):
        seen.append((remote, force, max_bytes))
        return fasrc_fetcher.FetchResult(ok=True, local_path="/tmp/x",
                                         size_bytes=76_000_000)

    monkeypatch.setattr(fasrc_fetcher, "fetch_one_file", fake)
    r = client.post("/api/euclid-psf/sync")
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True
    assert set(d["files"]) == {b.name for b in Config.BANDS}
    assert all(force for _, force, _ in seen)               # force=True
    assert all(mb == Config.WebFetch.MAX_PSF_PULL_BYTES for _, _, mb in seen)


def test_euclid_auth_save_writes_remote_credentials(client, monkeypatch):
    """Saving Euclid credentials writes ~/.euclid_credentials on FASRC via a
    quoted heredoc (password as stdin, not argv), mode 600. Nothing is
    stored on the laptop."""
    from euclid_polish.web import remote as web_remote
    captured = {}

    class _CapSSH:
        def is_connected(self): return True
        def run(self, cmd, timeout=60):
            captured["cmd"] = cmd
            return (0, "", "")

    monkeypatch.setattr(web_remote.STATE, "ssh", _CapSSH())
    r = client.post("/euclid-auth/save",
                    data={"euclid_user": "alice", "euclid_password": "s3cr3t!$x"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    cmd = captured["cmd"]
    assert "alice" in cmd and "s3cr3t!$x" in cmd
    assert ".euclid_credentials" in cmd
    assert "umask 077" in cmd and "chmod 600" in cmd
    # Heredoc terminator present so the password lands as file content.
    assert "__EUCLID_CREDS_EOF__" in cmd


def test_euclid_auth_save_rejects_blank(client, monkeypatch):
    from euclid_polish.web import remote as web_remote

    class _OkSSH:
        def is_connected(self): return True
        def run(self, cmd, timeout=60): return (0, "", "")

    monkeypatch.setattr(web_remote.STATE, "ssh", _OkSSH())
    r = client.post("/euclid-auth/save",
                    data={"euclid_user": "", "euclid_password": "x"})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_euclid_auth_status_reports_presence(client, monkeypatch):
    from euclid_polish.web import remote as web_remote

    class _PresentSSH:
        def is_connected(self): return True
        def run(self, cmd, timeout=60): return (0, "alice\n", "")

    monkeypatch.setattr(web_remote.STATE, "ssh", _PresentSSH())
    r = client.get("/euclid-auth/status")
    assert r.status_code == 200
    body = r.get_json()
    assert body["present"] is True and body["user"] == "alice"


def test_tng_auth_save_writes_remote_token(client, monkeypatch):
    """Saving the TNG token writes ~/.tng_api_key on FASRC via a quoted
    heredoc (token as stdin, not argv), mode 600. Nothing is stored locally."""
    from euclid_polish.web import remote as web_remote
    captured = {}

    class _CapSSH:
        def is_connected(self): return True
        def run(self, cmd, timeout=60):
            captured["cmd"] = cmd
            return (0, "", "")

    monkeypatch.setattr(web_remote.STATE, "ssh", _CapSSH())
    r = client.post("/tng-auth/save", data={"tng_token": "abc123DEADBEEF"})
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True and d["chars"] == len("abc123DEADBEEF")
    cmd = captured["cmd"]
    assert "abc123DEADBEEF" in cmd
    assert ".tng_api_key" in cmd
    assert "umask 077" in cmd and "chmod 600" in cmd
    # Heredoc terminator present so the token lands as file content.
    assert "__TNG_KEY_EOF__" in cmd
    # The token must never reach the response payload.
    assert "abc123DEADBEEF" not in r.get_data(as_text=True)


def test_tng_auth_save_rejects_blank(client, monkeypatch):
    from euclid_polish.web import remote as web_remote

    class _OkSSH:
        def is_connected(self): return True
        def run(self, cmd, timeout=60): return (0, "", "")

    monkeypatch.setattr(web_remote.STATE, "ssh", _OkSSH())
    r = client.post("/tng-auth/save", data={"tng_token": "   "})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_tng_auth_save_requires_connection(client, monkeypatch):
    """No FASRC connection → the request is refused and NOTHING is written.

    The global SSH gate (``_enforce_ssh_gate`` before_request) redirects a
    disconnected request to /connection-error (302); the endpoint's own
    ``is_connected`` guard is belt-and-suspenders behind it. Either way the
    token must never be written — the stub's ``run`` raises if touched."""
    from euclid_polish.web import remote as web_remote

    class _Down:
        def is_connected(self): return False
        def run(self, cmd, timeout=60):
            raise AssertionError("must not run without a connection")

    monkeypatch.setattr(web_remote.STATE, "ssh", _Down())
    r = client.post("/tng-auth/save", data={"tng_token": "x"})
    # Refused — never a successful save (no write happened: the stub would
    # have raised). 302 = gate redirect, 400 = endpoint guard.
    assert r.status_code in (302, 400)


def test_tng_auth_status_reports_presence_without_leaking_token(client, monkeypatch):
    from euclid_polish.web import remote as web_remote

    class _PresentSSH:
        def is_connected(self): return True
        def run(self, cmd, timeout=60): return (0, "39\n", "")   # wc -c output

    monkeypatch.setattr(web_remote.STATE, "ssh", _PresentSSH())
    r = client.get("/tng-auth/status")
    assert r.status_code == 200
    body = r.get_json()
    assert body["present"] is True and body["chars"] == 39
    assert "token" not in body and "tng_token" not in body


# ---------------------------------------------------------------------------
# TNG infographics (rendered on FASRC, streamed back)
# ---------------------------------------------------------------------------

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _stub_fetch(monkeypatch, *, local_path=None, ok=True, error=None):
    """Make routes/tng.fetch_one_file return a canned FetchResult + capture
    the (remote_path, kwargs) the route asked for."""
    from euclid_polish.web.routes import tng as tng_routes
    from euclid_polish.web.fasrc_fetcher import FetchResult
    captured = {}

    def fake(remote, **kw):
        captured["remote"] = remote
        captured["kw"] = kw
        return FetchResult(ok=ok, local_path=local_path, error=error)
    monkeypatch.setattr(tng_routes, "fetch_one_file", fake)
    return captured


def test_tng_histograms_png_renders_locally(client, monkeypatch):
    """Histograms render in-process (not a job): the route calls the local
    render with the FASRC id list + key and streams the PNG."""
    from euclid_polish.web.routes import tng as tng_routes
    seen = {}

    def fake_render(work, ids, key, **kw):
        seen["work"] = work
        return _PNG_MAGIC + b"hist"
    monkeypatch.setattr(tng_routes, "render_histograms_for_ids", fake_render)
    r = client.get("/tng/histograms.png")
    assert r.status_code == 200 and r.mimetype == "image/png"
    assert r.data.startswith(_PNG_MAGIC)
    assert "_tng_infographics" in seen["work"]      # local cache dir


def test_tng_result_grid_serves_png(client, tmp_path, monkeypatch):
    p = tmp_path / "grid.png"
    p.write_bytes(_PNG_MAGIC + b"x")
    cap = _stub_fetch(monkeypatch, local_path=str(p))
    r = client.get("/tng/result/grid.png")
    assert r.status_code == 200 and r.mimetype == "image/png"
    assert cap["remote"].endswith("/_infographics/grid.png")


def test_tng_result_stack_is_attachment_with_large_cap(client, tmp_path, monkeypatch):
    p = tmp_path / "stack.fits"
    p.write_bytes(b"SIMPLE  =" + b"\x00" * 1024)
    cap = _stub_fetch(monkeypatch, local_path=str(p))
    r = client.get("/tng/result/stack.fits")
    assert r.status_code == 200 and r.mimetype == "application/fits"
    cd = r.headers.get("Content-Disposition", "")
    assert "attachment" in cd and "TNG_stack.fits" in cd
    assert cap["remote"].endswith("/_infographics/stack.fits")
    # Must request the larger-than-50 MB cap so the ~51 MB stack isn't refused.
    assert cap["kw"].get("max_bytes") == Config.WebFetch.MAX_PSF_PULL_BYTES


def test_tng_result_missing_returns_404(client, monkeypatch):
    _stub_fetch(monkeypatch, ok=False, local_path=None, error="not found")
    r = client.get("/tng/result/grid.png")
    assert r.status_code == 404
    assert r.get_json()["ok"] is False
    assert "submit the job" in r.get_json()["error"]


def test_post_inference_generate_reconstruct_returns_job_id(client):
    # Spawns a background job and returns its id even without a live FASRC
    # connection — the job itself fails fast ("not connected") in that case.
    r = client.post("/inference/generate-reconstruct", data={
        "checkpoint_dir": "/tmp/nope", "hr_image_size": 510, "n_pairs": 1,
        "tng_fraction": "0.2",          # accepted (TNG injection in inference)
    })
    assert r.status_code == 200
    assert "job_id" in r.get_json()


def test_login_node_generate_cmd_injects_tng_fraction():
    """The inference login-node generation forwards --tng-fraction to
    run_pipeline.py only when > 0, so the synthetic scenes include TNG galaxies."""
    from euclid_polish.web.helpers.jobs_impl import _login_node_generate_cmd
    from euclid_polish.web.fasrc_config import FasrcConfig
    cfg = FasrcConfig(data_dir="/n/d", conda_env_path="/n/env", repo_path="/n/repo")
    base = _login_node_generate_cmd(cfg, "/n/tmp", 510, 2)
    assert "scripts/run_pipeline.py" in base
    assert "--tng-fraction" not in base
    withtng = _login_node_generate_cmd(cfg, "/n/tmp", 510, 2, tng_fraction=0.3)
    assert "--tng-fraction 0.3" in withtng
    assert "--tng-fraction" not in _login_node_generate_cmd(
        cfg, "/n/tmp", 510, 2, tng_fraction=0.0)


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
    # The panel renders the FASRC-pulled ePSFs: 200 PNG when present, else
    # 404 (no test SSH session → nothing to pull). Both are valid.
    r = client.get("/view/psfs?band=all")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert r.headers["Content-Type"] == "image/png"
        assert len(r.data) > 100


def test_view_psfs_per_band_returns_png(client):
    r = client.get("/view/psfs?band=VIS")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
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
    # Pair (triptych) view chip — the default landing view.
    assert "pair (triptych)" in body, (
        "expected 'pair (triptych)' chip in the Type toolbar — the "
        "side-by-side clean/dirty/HR view should be available + the "
        "page's initial selection."
    )


def test_view_hst_pair_pair_kind_404_when_not_synced(
        client, tmp_path, monkeypatch):
    """The triptych path reads three shards (clean/dirty/hr); when
    none are cached, it must 404 — not 500 — same as single-image
    kinds. Regression on the multi-shard composite path."""
    from euclid_polish.web import fasrc_fetcher as ff
    monkeypatch.setattr(Config, "FASRC_CACHE_DIR", str(tmp_path))
    r = client.get("/view/hst-pair?subset=validate&kind=pair&band=VIS&i=0")
    assert r.status_code == 404


def test_view_hst_pair_pair_kind_rejects_bad_band(client):
    """Invalid band string must still 400 even on the triptych path."""
    r = client.get("/view/hst-pair?subset=validate&kind=pair&band=BOGUS&i=0")
    assert r.status_code == 400


def test_view_hst_pair_pair_kind_rejects_bad_subset(client):
    r = client.get("/view/hst-pair?subset=foo&kind=pair&band=VIS&i=0")
    assert r.status_code == 400


def test_view_hst_pair_pair_kind_renders_real_png(
        client, tmp_path, monkeypatch):
    """End-to-end: write synthetic clean/dirty/hr shards into a tmp
    cache dir, request the triptych for idx=0, and assert the response
    is a real PNG. Catches any composite-layout / matplotlib bug that
    the 400/404 tests can't see (those bail before the renderer runs).
    """
    import numpy as np
    from euclid_polish.config import Config
    from euclid_polish.sky.tfrecord import (
        open_multiband_writer, tfrecord_path,
    )
    from euclid_polish.sky.types import MultiBandSkyImage
    from euclid_polish.web import fasrc_fetcher as ff
    from euclid_polish.web import remote as web_remote
    from euclid_polish.web import fasrc_config

    # Point the local cache at tmp_path and the remote at a fixed
    # absolute path so _hst_pairs_local_dir resolves under tmp_path.
    monkeypatch.setattr(Config, "FASRC_CACHE_DIR", str(tmp_path))
    # Override the FASRC config's data_dir → we want
    # _hst_pairs_remote_dir() → "{data_dir}/images/records_v2_hst" to
    # land somewhere stable, but the renderer only reads the LOCAL
    # cache (set above). The remote dir just feeds into the cache-path
    # hash used by ``_local_path_for``.
    cfg = fasrc_config.load()
    monkeypatch.setattr(
        fasrc_config, "load",
        lambda *_a, **_kw: cfg.__class__(
            **{**cfg.__dict__, "data_dir": "/tmp/fasrc-data"}
        ),
    )

    # Resolve the local cache dir the same way the app does, then
    # write three matching synthetic shards into it.
    from euclid_polish.web.fasrc_fetcher import _local_path_for
    remote_dir = "/tmp/fasrc-data/images/records_v2_hst"
    local_dir = os.path.dirname(
        _local_path_for(f"{remote_dir}/clean_validate.tfrecord")
    )
    os.makedirs(local_dir, exist_ok=True)

    H, W = 32, 32
    rng = np.random.default_rng(0)
    for kind, n_bands, scale in [
        ("clean", len(Config.LR_INPUT_BAND_NAMES), 0.05),  # HR grid
        ("dirty", len(Config.LR_INPUT_BAND_NAMES), 0.10),  # LR grid
        ("hr",    1,                               0.05),  # VIS HR
    ]:
        data = rng.uniform(0, 100, size=(H, W, n_bands)).astype(np.float32)
        band_names = (Config.LR_INPUT_BAND_NAMES if n_bands == 4
                      else ("VIS",))
        img = MultiBandSkyImage(
            data=data, pixel_scale_arcsec=scale,
            band_names=band_names, is_clean=(kind != "dirty"),
            metadata={"source": "test"},
        )
        with open_multiband_writer(
            f"{kind}_validate", records_dir=local_dir,
        ) as w:
            w.write(img, index=0)
        assert os.path.exists(
            tfrecord_path(local_dir, f"{kind}_validate")
        ), f"failed to write {kind}_validate.tfrecord"

    r = client.get(
        "/view/hst-pair?subset=validate&kind=pair&band=VIS&i=0"
    )
    assert r.status_code == 200, (
        f"triptych endpoint returned {r.status_code}; "
        f"body={r.data[:200]!r}"
    )
    # Real PNG starts with the 8-byte magic header.
    assert r.data[:8] == b"\x89PNG\r\n\x1a\n", (
        "response is not a PNG — composite renderer probably errored "
        "and matplotlib returned empty bytes"
    )


def test_view_hst_pair_invalid_subset_400(client):
    r = client.get("/view/hst-pair?subset=foo&kind=clean&band=VIS&i=0")
    assert r.status_code == 400


def test_view_hst_pair_invalid_kind_400(client):
    r = client.get("/view/hst-pair?subset=validate&kind=foo&band=VIS&i=0")
    assert r.status_code == 400


def test_view_hst_pair_invalid_band_400(client):
    r = client.get("/view/hst-pair?subset=validate&kind=clean&band=BOGUS&i=0")
    assert r.status_code == 400


def test_view_hst_pair_404_when_not_synced(client, tmp_path, monkeypatch):
    """No local cache file → ``_render_sky_record_png`` aborts 404.

    Re-point the FASRC cache dir at an empty tmp dir so this test
    isn't tripped by whatever happens to be cached on the developer's
    machine (which is exactly what regressed the first time I wrote
    this — my own ``data/_fasrc_cache/`` had real validate files)."""
    from euclid_polish.web import fasrc_fetcher as ff
    monkeypatch.setattr(Config, "FASRC_CACHE_DIR", str(tmp_path))
    r = client.get("/view/hst-pair?subset=validate&kind=clean&band=VIS&i=0")
    assert r.status_code == 404


def test_api_hst_pairs_totals_returns_json_with_all_six_files(client):
    r = client.get("/api/hst-pairs/totals")
    assert r.status_code == 200
    body = r.get_json()
    # Every key must be present even when the cache is empty so the JS
    # can build its index labels deterministically. Per key:
    #   0    — file absent / empty (renders as "0")
    #   int  — full record count
    #   None — file present but partially corrupt (truncated rsync,
    #          DataLossError on read). Renders as "—" in the UI.
    # Refusing to accept None here would mean a single bad shard
    # 500-s the whole endpoint and the UI shows 0/0 across the board.
    assert set(body.keys()) == {
        "clean_train", "clean_validate",
        "dirty_train", "dirty_validate",
        "hr_train",    "hr_validate",
    }
    for v in body.values():
        assert v is None or (isinstance(v, int) and v >= 0)


def test_record_count_handles_truncated_tfrecord(tmp_path):
    """A truncated tfrecord (interrupted rsync, bad header etc.) must
    not 500 the totals endpoint — return None so callers render "—".

    This is the regression for the bug where one bad ``clean_train``
    shard on disk poisoned the entire /hst-pairs viewer: the API
    raised ``DataLossError``, the response 500'd, and every count
    (including the valid validate files) silently became 0 in the UI.
    """
    from euclid_polish.web.helpers.status import _record_count

    # ``_record_count(name)`` reads ``<dir>/<name>.tfrecord``; write a
    # garbage-bytes shard at that exact path so TF rejects the header.
    bad = tmp_path / "garbage.tfrecord"
    bad.write_bytes(b"\x00" * 1024 + b"not a real record" + b"\xff" * 1024)
    assert _record_count("garbage", records_dir=str(tmp_path)) is None

    # An absent file is distinct from a bad one — returns 0, not None.
    assert _record_count("does_not_exist", records_dir=str(tmp_path)) == 0


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


# ---------------------------------------------------------------------------
# FASRC HST-pipeline status API — round-trip wiring
# ---------------------------------------------------------------------------
#
# /api/fasrc/hst/status returns the registered pipeline steps + an
# artifact-existence dict. After the round-trip feature landed there
# should be two new steps and two new artifact keys; these tests pin
# the wiring on the *server* side so the UI's JS dispatch (form fields
# + status badges) can rely on them being there.

def test_hst_status_exposes_roundtrip_steps_and_artifacts(client):
    r = client.get("/api/fasrc/hst/status")
    assert r.status_code == 200
    body = r.get_json()

    # Steps registry: round-trip steps must appear alongside the
    # original HST pipeline so the UI auto-renders cards for them.
    step_ids = {s["step_id"] for s in body["steps"]}
    assert "euclid_sky_download"          in step_ids
    assert "euclid_roundtrip_tfrecords"   in step_ids

    # Artifact keys must appear in the dict regardless of SSH state
    # (when disconnected they're None — meaning "unknown"; when SSH
    # is up the probe returns True/False per actual existence). The
    # JS side keys off these names; missing names would silently
    # break the badge rendering for the new steps.
    artifacts = body["artifacts"]
    assert "euclid_sky"         in artifacts
    assert "roundtrip_records"  in artifacts
    # Values must be None or bool — never raw strings / ints — so the
    # JS ``=== true`` / ``=== false`` checks behave predictably.
    for k in ("euclid_sky", "roundtrip_records"):
        v = artifacts[k]
        assert v is None or isinstance(v, bool), (
            f"artifacts[{k!r}] = {v!r} (type {type(v).__name__}) — "
            "must be None or bool"
        )


def test_hst_status_keeps_pre_existing_artifact_keys(client):
    """Backward-compat: the original 5 keys must still be present so
    nothing on the JS side that depends on ``artifacts.tiles`` / etc.
    silently breaks."""
    r = client.get("/api/fasrc/hst/status")
    artifacts = r.get_json()["artifacts"]
    for key in ("tiles", "psf", "kernel", "records", "ckpt"):
        assert key in artifacts, f"original artifact key '{key}' missing"


def test_hst_status_omits_deleted_two_stage_chain_keys(client, monkeypatch):
    """The deleted two-stage chain (``train_denoiser`` /
    ``train_transition`` / ``transition_pairs``) and its on-disk
    artifacts must no longer surface via /api/fasrc/hst/status — they
    were ripped out wholesale and any lingering reference would render
    a broken UI card."""
    # Pin ssh=None so the endpoint returns its static step/artifact maps
    # and skips the live SSH probe entirely. This test only asserts the
    # *shape* (step ids + artifact keys), and pinning makes it immune to
    # whatever ssh stub a prior test happened to leave on the global STATE.
    from euclid_polish.web import remote as web_remote
    monkeypatch.setattr(web_remote.STATE, "ssh", None)
    r = client.get("/api/fasrc/hst/status")
    body = r.get_json()
    step_ids = {s["step_id"] for s in body["steps"]}
    for gone in ("train_denoiser", "train_transition", "transition_pairs"):
        assert gone not in step_ids, (
            f"deleted step '{gone}' is still registered"
        )
    artifacts = body["artifacts"]
    for gone in ("denoiser", "transition_model", "transition_pairs"):
        assert gone not in artifacts, (
            f"deleted artifact key '{gone}' is still in the probe map"
        )


def test_inference_lists_synthetic_runs_and_drops_deprecated(client, tmp_path, monkeypatch):
    inf = tmp_path / "euclid_inference"
    cut = inf / "cutouts" / "latest"; cut.mkdir(parents=True)
    for n in ("original_stack.fits", "SR.fits", "SR_forward.fits", "residual.fits"):
        (cut / n).write_bytes(b"x")          # deprecated present ON DISK
    syn = inf / "synthetic" / "gensynth_510px_idx0000"; syn.mkdir(parents=True)
    for n in ("original_stack.fits", "SR.fits", "HR.fits"):
        (syn / n).write_bytes(b"x")
    monkeypatch.setattr(Config, "EUCLID_INFERENCE_DIR", str(inf))

    body = client.get("/inference").get_data(as_text=True)
    # synthetic run is listed with its inspectable FITS set
    assert "Synthetic reconstructions" in body
    assert "gensynth_510px_idx0000" in body
    assert "synthetic/gensynth_510px_idx0000/SR.fits" in body
    assert "HR.fits" in body
    # the real-Euclid run still shows the kept files
    assert "cutouts/latest/SR.fits" in body
    # deprecated files are NOT listed anywhere, though they exist on disk
    assert "SR_forward.fits" not in body
    assert "residual.fits" not in body
