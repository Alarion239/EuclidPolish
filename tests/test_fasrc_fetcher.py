"""Tests for the FASRC pull-on-demand file fetcher + connection gate."""

from __future__ import annotations

import os

import pytest

from euclid_polish.web import fasrc_config
from euclid_polish.web.fasrc_fetcher import (
    CACHE_DIR,
    MAX_PULL_BYTES,
    FetchResult,
    _local_path_for,
    allowed_remote_roots,
    cache_size_bytes,
    fetch_one_file,
    is_allowed_remote_path,
)


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

class TestPathSafety:

    def test_allowed_roots_match_config(self):
        cfg = fasrc_config.load()
        roots = allowed_remote_roots()
        assert cfg.data_dir in roots
        assert cfg.ckpt_dir in roots
        # logs dir is repo_path + /logs
        assert f"{cfg.repo_path}/logs" in roots

    def test_path_under_data_dir_allowed(self):
        cfg = fasrc_config.load()
        assert is_allowed_remote_path(f"{cfg.data_dir}/hst_psf/F814W.fits")

    def test_path_traversal_rejected(self):
        cfg = fasrc_config.load()
        assert not is_allowed_remote_path(f"{cfg.data_dir}/../etc/passwd")

    def test_etc_passwd_rejected(self):
        assert not is_allowed_remote_path("/etc/passwd")
        assert not is_allowed_remote_path("")
        assert not is_allowed_remote_path("/")


# ---------------------------------------------------------------------------
# Local cache path layout
# ---------------------------------------------------------------------------

class TestCachePathLayout:

    def test_local_path_under_cache_dir(self):
        local = _local_path_for("/n/netscratch/foo/bar/baz.fits")
        assert local.startswith(os.path.realpath(CACHE_DIR))
        assert local.endswith("/baz.fits")

    def test_unique_local_paths_for_different_remotes(self):
        a = _local_path_for("/n/data/a.fits")
        b = _local_path_for("/n/data/b.fits")
        assert a != b


# ---------------------------------------------------------------------------
# Fetcher refuses bad inputs without touching SSH
# ---------------------------------------------------------------------------

class TestFetcherRejectsBadInputs:

    def test_rejects_path_outside_roots(self):
        r = fetch_one_file("/etc/passwd")
        assert not r.ok
        assert "allowed FASRC roots" in r.error

    def test_rejects_traversal(self):
        cfg = fasrc_config.load()
        r = fetch_one_file(f"{cfg.data_dir}/../etc/passwd")
        assert not r.ok
        assert "allowed FASRC roots" in r.error

    def test_rejects_empty_path(self):
        r = fetch_one_file("")
        assert not r.ok


# ---------------------------------------------------------------------------
# Cache eviction
# ---------------------------------------------------------------------------

class TestCacheEviction:

    def test_cache_size_reads_existing_files(self, tmp_path, monkeypatch):
        from euclid_polish.web import fasrc_fetcher as ff
        monkeypatch.setattr(ff, "CACHE_DIR", str(tmp_path))
        # No files → 0
        assert ff.cache_size_bytes() == 0
        # Write a couple of files
        (tmp_path / "a").write_bytes(b"x" * 100)
        (tmp_path / "b").write_bytes(b"y" * 200)
        assert ff.cache_size_bytes() == 300

    def test_evict_lru_frees_oldest_first(self, tmp_path, monkeypatch):
        from euclid_polish.web import fasrc_fetcher as ff
        monkeypatch.setattr(ff, "CACHE_DIR", str(tmp_path))
        f1 = tmp_path / "old"
        f2 = tmp_path / "new"
        f1.write_bytes(b"x" * 1000)
        f2.write_bytes(b"y" * 1000)
        # Make f1 older.
        os.utime(str(f1), (1_000_000, 1_000_000))
        os.utime(str(f2), (2_000_000, 2_000_000))
        ff._evict_lru_until_under(1500)   # only one file fits
        assert not f1.exists()
        assert f2.exists()


# ---------------------------------------------------------------------------
# Connection gate
# ---------------------------------------------------------------------------

class TestConnectionGate:

    @pytest.fixture
    def app_with_no_ssh(self, monkeypatch):
        from euclid_polish.web.app import create_app
        from euclid_polish.web.remote import STATE
        app = create_app()
        monkeypatch.setattr(STATE, "ssh", None)
        return app

    def test_root_redirects_when_disconnected(self, app_with_no_ssh):
        client = app_with_no_ssh.test_client()
        r = client.get("/", follow_redirects=False)
        assert r.status_code == 302
        assert "/connection-error" in r.headers["Location"]

    def test_catalog_redirects_when_disconnected(self, app_with_no_ssh):
        client = app_with_no_ssh.test_client()
        r = client.get("/catalog", follow_redirects=False)
        assert r.status_code == 302

    def test_connection_error_page_reachable(self, app_with_no_ssh):
        client = app_with_no_ssh.test_client()
        r = client.get("/connection-error")
        assert r.status_code == 200
        assert b"FASRC connection required" in r.data

    def test_static_assets_reachable(self, app_with_no_ssh):
        client = app_with_no_ssh.test_client()
        r = client.get("/static/style.css")
        # 200 if the file exists; 404 is also acceptable here — what matters
        # is the gate doesn't redirect.
        assert r.status_code in (200, 404)
        assert r.status_code != 302

    def test_api_fasrc_always_reachable(self, app_with_no_ssh):
        client = app_with_no_ssh.test_client()
        r = client.get("/api/fasrc/status")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Routes registered
# ---------------------------------------------------------------------------

class TestRoutesRegistered:

    def test_all_new_routes_present(self):
        from euclid_polish.web.app import create_app
        app = create_app()
        urls = {str(r) for r in app.url_map.iter_rules()}
        assert "/fasrc/file/inspect" in urls
        assert "/fasrc/file/download" in urls
        assert "/connection-error" in urls
        assert "/api/connection/retry" in urls
        assert "/hst-psf" in urls
        assert "/hst-cutouts" in urls
        assert "/hst-psf/preview.png" in urls
        assert "/hst-cutouts/preview.png" in urls
        assert "/hst-tiles" in urls
        assert "/fasrc/tile/header" in urls
        assert "/fasrc/tile/cutout.png" in urls


# ---------------------------------------------------------------------------
# Tile inspector — path safety + input validation
# ---------------------------------------------------------------------------

class TestTileInspectorRoutes:

    @pytest.fixture
    def client(self):
        from euclid_polish.web.app import create_app
        return create_app().test_client()

    def test_header_rejects_path_outside_roots(self, client):
        r = client.get("/fasrc/tile/header?path=/etc/passwd")
        assert r.status_code == 400

    def test_header_rejects_empty_path(self, client):
        r = client.get("/fasrc/tile/header")
        assert r.status_code == 400

    def test_cutout_rejects_oversized_request(self, client):
        cfg = fasrc_config.load()
        legit_path = f"{cfg.data_dir}/hst_hlsp/foo.fits"
        r = client.get(f"/fasrc/tile/cutout.png?path={legit_path}&size=99999")
        assert r.status_code == 400

    def test_cutout_rejects_undersized_request(self, client):
        cfg = fasrc_config.load()
        legit_path = f"{cfg.data_dir}/hst_hlsp/foo.fits"
        r = client.get(f"/fasrc/tile/cutout.png?path={legit_path}&size=1")
        assert r.status_code == 400

    def test_cutout_rejects_path_outside_roots(self, client):
        r = client.get("/fasrc/tile/cutout.png?path=/etc/passwd&size=256")
        assert r.status_code == 400

    def test_page_renders_with_no_tiles_on_disk(self, client):
        # Even when zero tiles exist on FASRC the page should render.
        r = client.get("/hst-tiles")
        assert r.status_code == 200
