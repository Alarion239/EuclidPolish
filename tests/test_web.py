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


def test_transition_pairs_page_renders(client):
    r = client.get("/transition-pairs")
    assert r.status_code == 200
    body = r.data.decode()
    # Toolbar pills for the original three view modes.
    assert "input (HST)" in body
    assert "target (Euclid)" in body
    assert "residual" in body
    # And the two new denoiser-pair views.
    assert "noisy HST" in body
    assert "denoiser pair" in body
    # And the denoiser-output views.
    assert "denoised HST" in body
    assert "denoiser strip" in body
    # Noise controls present (hidden by default until a noise kind
    # chip is clicked, but the elements have to be in the DOM).
    assert 'id="noise-alpha"' in body
    assert 'id="noise-sigma-floor"' in body
    assert 'id="noise-reshuffle"' in body
    # Should mention what the page is for.
    assert "PSF_HST" in body
    assert "PSF_Euclid" in body


def test_transition_pair_view_renders_noisy_hst(client, tmp_path,
                                                   monkeypatch):
    """End-to-end: write a synthetic input_validate.tfrecord into the
    transition-pair cache, request the ``noisy_hst`` view, assert a
    real PNG comes back. The renderer has to add HLSP noise to the
    clean record and produce a usable image."""
    import numpy as np
    from euclid_polish.config import Config
    from euclid_polish.sky.tfrecord import open_multiband_writer
    from euclid_polish.sky.types import MultiBandSkyImage
    from euclid_polish.web import fasrc_fetcher as ff
    from euclid_polish.web import fasrc_config

    monkeypatch.setattr(ff, "CACHE_DIR", str(tmp_path))
    cfg = fasrc_config.load()
    monkeypatch.setattr(
        fasrc_config, "load",
        lambda *_a, **_kw: cfg.__class__(
            **{**cfg.__dict__, "data_dir": "/tmp/fasrc-data-noisy"}
        ),
    )

    from euclid_polish.web.fasrc_fetcher import _local_path_for
    remote_dir = "/tmp/fasrc-data-noisy/images/records_transition"
    local_dir = os.path.dirname(
        _local_path_for(f"{remote_dir}/input_validate.tfrecord")
    )
    os.makedirs(local_dir, exist_ok=True)

    rng = np.random.default_rng(0)
    data = rng.uniform(0, 500, size=(32, 32, 1)).astype(np.float32)
    img = MultiBandSkyImage(
        data=data, pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=("VIS",), is_clean=True,
    )
    with open_multiband_writer("input_validate", records_dir=local_dir) as w:
        w.write(img, index=0)

    r = client.get(
        "/view/transition-pair?subset=validate&kind=noisy_hst"
        "&i=0&alpha=0.8&sigma_floor=12"
    )
    assert r.status_code == 200, (
        f"noisy_hst render failed: {r.status_code} body={r.data[:200]!r}"
    )
    assert r.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_transition_pair_view_renders_hst_pair(client, tmp_path,
                                               monkeypatch):
    """Side-by-side clean+noisy. Same fixture pattern as the
    noisy_hst test, different `kind` query."""
    import numpy as np
    from euclid_polish.config import Config
    from euclid_polish.sky.tfrecord import open_multiband_writer
    from euclid_polish.sky.types import MultiBandSkyImage
    from euclid_polish.web import fasrc_fetcher as ff
    from euclid_polish.web import fasrc_config

    monkeypatch.setattr(ff, "CACHE_DIR", str(tmp_path))
    cfg = fasrc_config.load()
    monkeypatch.setattr(
        fasrc_config, "load",
        lambda *_a, **_kw: cfg.__class__(
            **{**cfg.__dict__, "data_dir": "/tmp/fasrc-data-pair"}
        ),
    )

    from euclid_polish.web.fasrc_fetcher import _local_path_for
    remote_dir = "/tmp/fasrc-data-pair/images/records_transition"
    local_dir = os.path.dirname(
        _local_path_for(f"{remote_dir}/input_validate.tfrecord")
    )
    os.makedirs(local_dir, exist_ok=True)

    rng = np.random.default_rng(1)
    data = rng.uniform(0, 500, size=(32, 32, 1)).astype(np.float32)
    img = MultiBandSkyImage(
        data=data, pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=("VIS",), is_clean=True,
    )
    with open_multiband_writer("input_validate", records_dir=local_dir) as w:
        w.write(img, index=0)

    r = client.get(
        "/view/transition-pair?subset=validate&kind=hst_pair&i=0"
    )
    assert r.status_code == 200, (
        f"hst_pair render failed: {r.status_code} body={r.data[:200]!r}"
    )
    assert r.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_transition_pair_view_404s_on_missing_denoiser_weights(
        client, tmp_path, monkeypatch):
    """Without trained denoiser weights in the local cache, the
    ``denoised_hst`` and ``denoiser_strip`` kinds must 404 with a
    clear message — not silently render a randomly initialised
    denoiser's garbage output."""
    import numpy as np
    from euclid_polish.config import Config
    from euclid_polish.sky.tfrecord import open_multiband_writer
    from euclid_polish.sky.types import MultiBandSkyImage
    from euclid_polish.web import fasrc_fetcher as ff
    from euclid_polish.web import fasrc_config

    monkeypatch.setattr(ff, "CACHE_DIR", str(tmp_path))
    cfg = fasrc_config.load()
    monkeypatch.setattr(
        fasrc_config, "load",
        lambda *_a, **_kw: cfg.__class__(
            **{**cfg.__dict__, "data_dir": "/tmp/fasrc-no-denoiser"}
        ),
    )

    from euclid_polish.web.fasrc_fetcher import _local_path_for
    remote_dir = "/tmp/fasrc-no-denoiser/images/records_transition"
    local_dir = os.path.dirname(
        _local_path_for(f"{remote_dir}/input_validate.tfrecord")
    )
    os.makedirs(local_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    img = MultiBandSkyImage(
        data=rng.uniform(0, 500, (32, 32, 1)).astype(np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=("VIS",), is_clean=True,
    )
    with open_multiband_writer("input_validate", records_dir=local_dir) as w:
        w.write(img, index=0)

    for kind in ("denoised_hst", "denoiser_strip"):
        r = client.get(
            f"/view/transition-pair?subset=validate&kind={kind}&i=0"
        )
        assert r.status_code == 404, (
            f"{kind} should 404 when denoiser weights are missing, "
            f"got {r.status_code}"
        )


def test_transition_pair_view_renders_denoiser_strip(
        client, tmp_path, monkeypatch):
    """End-to-end: tiny synthetic denoiser checkpoint + input shard +
    the matching summary JSON → the strip view returns a real PNG.
    Demonstrates that the pipeline goes
    input_shard → noise → load denoiser → forward pass → render. """
    import json
    import numpy as np
    from euclid_polish.config import Config
    from euclid_polish.sky.tfrecord import open_multiband_writer
    from euclid_polish.sky.types import MultiBandSkyImage
    from euclid_polish.training.transition_model import (
        HSTDenoiser, save_denoiser_weights,
    )
    from euclid_polish.web import fasrc_fetcher as ff
    from euclid_polish.web import fasrc_config

    monkeypatch.setattr(ff, "CACHE_DIR", str(tmp_path))
    cfg = fasrc_config.load()
    monkeypatch.setattr(
        fasrc_config, "load",
        lambda *_a, **_kw: cfg.__class__(
            **{**cfg.__dict__, "data_dir": "/tmp/fasrc-with-denoiser"}
        ),
    )

    from euclid_polish.web.fasrc_fetcher import _local_path_for
    # Input shard
    remote_records = "/tmp/fasrc-with-denoiser/images/records_transition"
    local_records = os.path.dirname(
        _local_path_for(f"{remote_records}/input_validate.tfrecord")
    )
    os.makedirs(local_records, exist_ok=True)
    rng = np.random.default_rng(1)
    img = MultiBandSkyImage(
        data=rng.uniform(0, 500, (32, 32, 1)).astype(np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=("VIS",), is_clean=True,
    )
    with open_multiband_writer("input_validate", records_dir=local_records) as w:
        w.write(img, index=0)

    # Denoiser weights + summary in the local cache.
    den_weights_remote = "/tmp/fasrc-with-denoiser/hst_psf/hst_denoiser.weights.h5"
    den_weights_local = _local_path_for(den_weights_remote)
    os.makedirs(os.path.dirname(den_weights_local), exist_ok=True)
    m = HSTDenoiser(channels=4, n_inner_layers=1, kernel_size=3)
    save_denoiser_weights(m, den_weights_local)

    summary_local = _local_path_for(
        "/tmp/fasrc-with-denoiser/hst_psf/hst_denoiser_summary.json"
    )
    with open(summary_local, "w") as f:
        json.dump({
            "channels": 4, "n_inner_layers": 1, "kernel_size": 3,
        }, f)

    r = client.get(
        "/view/transition-pair?subset=validate&kind=denoiser_strip&i=0"
        "&alpha=0.8&sigma_floor=12"
    )
    assert r.status_code == 200, (
        f"denoiser_strip render failed: {r.status_code} "
        f"body={r.data[:200]!r}"
    )
    assert r.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_transition_pair_fits_download_returns_real_fits(
        client, tmp_path, monkeypatch):
    """End-to-end: write a synthetic input_validate.tfrecord, hit
    ``/view/transition-pair.fits``, parse the returned bytes with
    astropy and verify the array is the same shape + dtype as the
    record content. No asinh, no clip — raw linear values."""
    import io as _io
    import numpy as np
    from astropy.io import fits
    from euclid_polish.config import Config
    from euclid_polish.sky.tfrecord import open_multiband_writer
    from euclid_polish.sky.types import MultiBandSkyImage
    from euclid_polish.web import fasrc_fetcher as ff
    from euclid_polish.web import fasrc_config

    monkeypatch.setattr(ff, "CACHE_DIR", str(tmp_path))
    cfg = fasrc_config.load()
    monkeypatch.setattr(
        fasrc_config, "load",
        lambda *_a, **_kw: cfg.__class__(
            **{**cfg.__dict__, "data_dir": "/tmp/fasrc-fits-test"}
        ),
    )
    from euclid_polish.web.fasrc_fetcher import _local_path_for
    remote_dir = "/tmp/fasrc-fits-test/images/records_transition"
    local_dir = os.path.dirname(
        _local_path_for(f"{remote_dir}/input_validate.tfrecord")
    )
    os.makedirs(local_dir, exist_ok=True)
    rng = np.random.default_rng(123)
    payload = rng.uniform(0, 500, size=(32, 32, 1)).astype(np.float32)
    img = MultiBandSkyImage(
        data=payload,
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=("VIS",), is_clean=True,
    )
    with open_multiband_writer("input_validate", records_dir=local_dir) as w:
        w.write(img, index=0)

    r = client.get(
        "/view/transition-pair.fits?subset=validate&kind=input&i=0"
    )
    assert r.status_code == 200, (
        f"FITS endpoint returned {r.status_code}; body={r.data[:200]!r}"
    )
    assert r.mimetype == "application/fits"
    cd = r.headers.get("Content-Disposition", "")
    assert "transition_input_validate_0.fits" in cd, (
        f"Content-Disposition should suggest a filename; got {cd!r}"
    )
    with fits.open(_io.BytesIO(r.data)) as h:
        assert len(h) >= 1
        # PrimaryHDU carries the array.
        arr = np.asarray(h[0].data, dtype=np.float32)
        assert arr.shape == (32, 32)
        # Linear values, not asinh-stretched / clipped.
        assert arr.min() >= 0 and arr.max() <= 500 + 1e-3
        # Metadata propagated.
        hdr = h[0].header
        assert hdr.get("KIND") == "input"
        assert hdr.get("SUBSET") == "validate"
        assert int(hdr.get("INDEX")) == 0


def test_transition_pair_fits_hst_pair_kind_has_two_hdus(
        client, tmp_path, monkeypatch):
    """``kind=hst_pair`` should return a multi-HDU FITS with both
    CLEAN and NOISY arrays under their own EXTNAMEs."""
    import io as _io
    import numpy as np
    from astropy.io import fits
    from euclid_polish.config import Config
    from euclid_polish.sky.tfrecord import open_multiband_writer
    from euclid_polish.sky.types import MultiBandSkyImage
    from euclid_polish.web import fasrc_fetcher as ff
    from euclid_polish.web import fasrc_config

    monkeypatch.setattr(ff, "CACHE_DIR", str(tmp_path))
    cfg = fasrc_config.load()
    monkeypatch.setattr(
        fasrc_config, "load",
        lambda *_a, **_kw: cfg.__class__(
            **{**cfg.__dict__, "data_dir": "/tmp/fasrc-fits-pair"}
        ),
    )
    from euclid_polish.web.fasrc_fetcher import _local_path_for
    remote_dir = "/tmp/fasrc-fits-pair/images/records_transition"
    local_dir = os.path.dirname(
        _local_path_for(f"{remote_dir}/input_validate.tfrecord")
    )
    os.makedirs(local_dir, exist_ok=True)
    rng = np.random.default_rng(99)
    img = MultiBandSkyImage(
        data=rng.uniform(0, 200, (32, 32, 1)).astype(np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=("VIS",), is_clean=True,
    )
    with open_multiband_writer("input_validate", records_dir=local_dir) as w:
        w.write(img, index=0)

    r = client.get(
        "/view/transition-pair.fits?subset=validate&kind=hst_pair&i=0"
        "&alpha=0.8&sigma_floor=10"
    )
    assert r.status_code == 200
    with fits.open(_io.BytesIO(r.data)) as h:
        extnames = {hdu.header.get("EXTNAME") for hdu in h}
        assert "CLEAN" in extnames
        assert "NOISY" in extnames
        # Noise params recorded.
        prim = h[0].header
        assert float(prim.get("ALPHA")) == 0.8
        assert float(prim.get("SIGFLOOR")) == 10.0


def test_hst_pair_fits_download_returns_full_band_cube(
        client, tmp_path, monkeypatch):
    """``/view/hst-pair.fits`` must preserve all 4 bands of a
    multi-band record (not just the displayed one). The band chip
    on the page is a display-time choice; the FITS download always
    carries every channel for offline analysis."""
    import io as _io
    import numpy as np
    from astropy.io import fits
    from euclid_polish.config import Config
    from euclid_polish.sky.tfrecord import open_multiband_writer
    from euclid_polish.sky.types import MultiBandSkyImage
    from euclid_polish.web import fasrc_fetcher as ff
    from euclid_polish.web import fasrc_config

    monkeypatch.setattr(ff, "CACHE_DIR", str(tmp_path))
    cfg = fasrc_config.load()
    monkeypatch.setattr(
        fasrc_config, "load",
        lambda *_a, **_kw: cfg.__class__(
            **{**cfg.__dict__, "data_dir": "/tmp/fasrc-fits-hst"}
        ),
    )
    from euclid_polish.web.fasrc_fetcher import _local_path_for
    remote_dir = "/tmp/fasrc-fits-hst/images/records_v2_hst"
    local_dir = os.path.dirname(
        _local_path_for(f"{remote_dir}/clean_validate.tfrecord")
    )
    os.makedirs(local_dir, exist_ok=True)
    rng = np.random.default_rng(7)
    data = rng.uniform(0, 800, size=(32, 32, 4)).astype(np.float32)
    img = MultiBandSkyImage(
        data=data,
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES,
        is_clean=True,
    )
    with open_multiband_writer("clean_validate", records_dir=local_dir) as w:
        w.write(img, index=0)

    r = client.get(
        "/view/hst-pair.fits?subset=validate&kind=clean&i=0&band=VIS"
    )
    assert r.status_code == 200
    with fits.open(_io.BytesIO(r.data)) as h:
        arr = np.asarray(h[0].data, dtype=np.float32)
        # Full 4-band cube, NOT just the VIS slice the band chip would
        # have displayed.
        assert arr.shape == (32, 32, 4), (
            f"FITS download should keep all bands; got shape {arr.shape}"
        )


def test_transition_pair_view_rejects_negative_alpha_gracefully(client):
    """Malformed noise param shouldn't 500; the route's safe-float
    fallback should drop into a default. Even with no records cached
    the response is 404 (no file), never 5xx."""
    r = client.get(
        "/view/transition-pair?subset=validate&kind=noisy_hst&i=0"
        "&alpha=not-a-number"
    )
    # Either 404 (no cached shard) or 400/200 — but never 500.
    assert r.status_code < 500


def test_transition_pairs_totals_api(client):
    """Totals API returns one entry per shard name, all None when the
    cache directory is empty (which it is in a fresh test env)."""
    r = client.get("/api/transition-pairs/totals")
    assert r.status_code == 200
    j = r.get_json()
    for name in ("input_train", "input_validate",
                 "target_train", "target_validate"):
        assert name in j


def test_transition_pairs_status_api(client):
    r = client.get("/api/transition-pairs/status")
    assert r.status_code == 200
    j = r.get_json()
    assert "files" in j
    assert "dir" in j
    # Path mirrors the documented FASRC cache location.
    assert "records_transition" in j["dir"]


def test_transition_pair_view_404s_on_invalid_input(client):
    """The render route should 404 (not 500) for an out-of-range index,
    regardless of whether the cache happens to contain real shards from
    a prior FASRC sync. Use an absurd index that no plausible shard
    contains."""
    r = client.get("/view/transition-pair?subset=validate&kind=input&i=99999999")
    assert r.status_code in (404, 400)


def test_transition_pair_view_rejects_bad_kind(client):
    """Unknown ``kind`` values must be rejected with 4xx, not crash
    the renderer."""
    r = client.get("/view/transition-pair?subset=validate&kind=garbage&i=0")
    assert r.status_code in (400, 404)


def test_transition_pair_view_rejects_bad_subset(client):
    r = client.get("/view/transition-pair?subset=garbage&kind=input&i=0")
    assert r.status_code in (400, 404)


def test_transition_model_validate_404s_when_weights_missing(client, tmp_path,
                                                              monkeypatch):
    """The transition-model inspector endpoint loads the trained
    A_θ weights from local FASRC cache. When the weights file hasn't
    been synced yet, the route should 404 with a clear hint, not 500."""
    # Point fasrc_config at a tmp dir so the cache resolves to an
    # empty location and the route's existence check fires.
    from euclid_polish.web import fasrc_config
    cfg = fasrc_config.FasrcConfig(
        ssh_user="x", repo_path=str(tmp_path),
        data_dir=str(tmp_path / "data"),
        ckpt_dir=str(tmp_path / "ckpt"),
    )
    monkeypatch.setattr(fasrc_config, "load", lambda: cfg)
    r = client.get("/hst-psf/transition-validate.png")
    assert r.status_code == 404
    # Description should mention what's missing (HST PSF, weights,
    # or Euclid PSF). Body is HTML for abort(404, description=...).
    assert b"missing" in r.data.lower()


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
    monkeypatch.setattr(ff, "CACHE_DIR", str(tmp_path))
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
    monkeypatch.setattr(ff, "CACHE_DIR", str(tmp_path))
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
    monkeypatch.setattr(ff, "CACHE_DIR", str(tmp_path))
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
    from euclid_polish.web.app import _record_count

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


def test_hst_status_exposes_denoiser_artifact_key(client):
    """The two-stage Phase-1 step (``train_denoiser``) and its on-disk
    artifact (``denoiser``) must be wired into /api/fasrc/hst/status
    so the UI can render the badge + the per-step form."""
    r = client.get("/api/fasrc/hst/status")
    body = r.get_json()
    step_ids = {s["step_id"] for s in body["steps"]}
    assert "train_denoiser" in step_ids, (
        "Phase-1 denoiser-training step is missing from the registry "
        "→ no UI card will appear for it."
    )
    artifacts = body["artifacts"]
    assert "denoiser" in artifacts
    assert artifacts["denoiser"] is None or isinstance(
        artifacts["denoiser"], bool,
    )
