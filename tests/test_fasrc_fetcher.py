"""Tests for the FASRC pull-on-demand file fetcher + connection gate."""

from __future__ import annotations

import os

import pytest

from euclid_polish.config import Config
from euclid_polish.web import fasrc_config
from euclid_polish.web.fasrc_fetcher import (
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
        assert local.startswith(os.path.realpath(Config.FASRC_CACHE_DIR))
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

class TestForceBypassesCache:
    """``fetch_one_file(..., force=True)`` skips the TTL cache and
    always re-rsyncs. Powers the "Sync from FASRC" button so a kernel
    you rebuilt 30 s ago shows up without waiting for the 5-min TTL."""

    def test_force_bypasses_fresh_cache(self, tmp_path, monkeypatch):
        from euclid_polish.web import fasrc_fetcher as ff
        # Point the cache at tmp_path and the safety check at "always ok".
        monkeypatch.setattr(Config, "FASRC_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(ff, "is_allowed_remote_path", lambda p: True)

        # Plant a fresh "cached" file so the no-force path would short-circuit.
        remote = "/n/netscratch/foo/kernel.fits"
        local = ff._local_path_for(remote)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, "wb") as fh:
            fh.write(b"old cached content")
        # mtime = now() so age << TTL — without force, this would short-circuit.

        calls = {"rsync": 0, "size": 0}

        # Fake SSHSession that records calls and produces a "new" file when
        # rsync runs. Subclassing isn't needed — duck typing via STATE.ssh.
        class _FakeSSH:
            def rsync_pull(self_inner, remote_path, local_dir, **kw):
                calls["rsync"] += 1
                # Simulate rsync writing the new contents into local_dir.
                dest = os.path.join(local_dir, os.path.basename(remote_path))
                with open(dest, "wb") as fh:
                    fh.write(b"new content")
                return 0, "", ""

        # Patch the size probe + the SSH session.
        monkeypatch.setattr(
            ff, "_remote_size_bytes",
            lambda p: (True, 11, None),
        )
        from euclid_polish.web import remote as remote_module
        monkeypatch.setattr(remote_module.STATE, "ssh", _FakeSSH())

        # Sanity: without force, we get the cached blob and never rsync.
        r1 = ff.fetch_one_file(remote)
        assert r1.ok and r1.from_cache is True
        assert calls["rsync"] == 0

        # With force, we rsync even though the cached file is brand new.
        r2 = ff.fetch_one_file(remote, force=True)
        assert r2.ok and r2.from_cache is False
        assert calls["rsync"] == 1
        with open(local, "rb") as fh:
            assert fh.read() == b"new content"


class TestCacheEviction:

    def test_cache_size_reads_existing_files(self, tmp_path, monkeypatch):
        from euclid_polish.web import fasrc_fetcher as ff
        monkeypatch.setattr(Config, "FASRC_CACHE_DIR", str(tmp_path))
        # No files → 0
        assert ff.cache_size_bytes() == 0
        # Write a couple of files
        (tmp_path / "a").write_bytes(b"x" * 100)
        (tmp_path / "b").write_bytes(b"y" * 200)
        assert ff.cache_size_bytes() == 300

    def test_evict_lru_frees_oldest_first(self, tmp_path, monkeypatch):
        from euclid_polish.web import fasrc_fetcher as ff
        monkeypatch.setattr(Config, "FASRC_CACHE_DIR", str(tmp_path))
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

    def test_evict_lru_never_deletes_protected_path(self, tmp_path, monkeypatch):
        """A ``protect``-ed file is never evicted, even when it's the oldest
        and the cache is over budget — the cache just stays over budget."""
        from euclid_polish.web import fasrc_fetcher as ff
        monkeypatch.setattr(Config, "FASRC_CACHE_DIR", str(tmp_path))
        oldest = tmp_path / "oldest"   # would normally be evicted first
        other  = tmp_path / "other"
        oldest.write_bytes(b"x" * 1000)
        other.write_bytes(b"y" * 1000)
        os.utime(str(oldest), (1_000_000, 1_000_000))
        os.utime(str(other), (2_000_000, 2_000_000))
        ff._evict_lru_until_under(500, protect={str(oldest)})
        # Protected oldest survives; the unprotected newer file is evicted.
        assert oldest.exists()
        assert not other.exists()


class TestFetchSelfEvictionRegression:
    """Regression: a large pull that pushed the cache over budget used to
    evict the file it had JUST fetched (the rsync ``-t`` gave it the remote's
    old mtime, so the LRU saw it as the stalest file) and then crash
    ``stat``-ing the now-missing file — breaking the "never raises" contract
    and failing every gen+reconstruct that needed the 374 MB FASRC ePSF."""

    def test_fetch_over_cap_keeps_just_pulled_file(self, tmp_path, monkeypatch):
        from euclid_polish.web import fasrc_fetcher as ff
        from euclid_polish.web import remote as remote_module
        monkeypatch.setattr(Config, "FASRC_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(ff, "is_allowed_remote_path", lambda p: True)
        # Tiny cache budget so any pull trips the eviction path.
        monkeypatch.setattr(Config.WebFetch, "MAX_CACHE_BYTES", 500)

        remote = "/n/netscratch/foo/euclid_psf/euclid_psf_VIS.fits"

        class _FakeSSH:
            def rsync_pull(self_inner, remote_path, local_dir, **kw):
                dest = os.path.join(local_dir, os.path.basename(remote_path))
                os.makedirs(local_dir, exist_ok=True)
                with open(dest, "wb") as fh:
                    fh.write(b"z" * 2000)        # 2 KB ≫ the 500 B cap
                # Simulate the OLD ``-t`` behaviour explicitly: stamp the
                # freshly-pulled file with an ancient mtime. Even so, the
                # fetcher must not evict it.
                os.utime(dest, (1_000_000, 1_000_000))
                return 0, "", ""

        monkeypatch.setattr(ff, "_remote_size_bytes", lambda p: (True, 2000, None))
        monkeypatch.setattr(remote_module.STATE, "ssh", _FakeSSH())

        # Must NOT raise (old code raised FileNotFoundError on getsize) and
        # must report success with the file still present.
        res = ff.fetch_one_file(
            remote, force=True,
            max_bytes=Config.WebFetch.MAX_PSF_PULL_BYTES)
        assert res.ok is True, res.error
        assert res.local_path and os.path.isfile(res.local_path)
        assert res.size_bytes == 2000

    def test_fetch_preserves_the_rest_of_an_explicit_sync_set(self, tmp_path, monkeypatch):
        """A later shard in a batch must not evict an earlier requested shard."""
        from euclid_polish.web import fasrc_fetcher as ff
        from euclid_polish.web import remote as remote_module
        monkeypatch.setattr(Config, "FASRC_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(Config.WebFetch, "MAX_CACHE_BYTES", 500)
        monkeypatch.setattr(ff, "is_allowed_remote_path", lambda p: True)

        earlier_remote = "/n/netscratch/foo/images/records_v2/hr_test.tfrecord"
        later_remote = "/n/netscratch/foo/images/records_v2/hr_validate.tfrecord"
        earlier_local = ff._local_path_for(earlier_remote)
        os.makedirs(os.path.dirname(earlier_local), exist_ok=True)
        with open(earlier_local, "wb") as fh:
            fh.write(b"a" * 400)
        os.utime(earlier_local, (1_000_000, 1_000_000))

        class _FakeSSH:
            def rsync_pull(self_inner, remote_path, local_dir, **kw):
                os.makedirs(local_dir, exist_ok=True)
                with open(os.path.join(local_dir, os.path.basename(remote_path)), "wb") as fh:
                    fh.write(b"b" * 400)
                return 0, "", ""

        monkeypatch.setattr(ff, "_remote_size_bytes", lambda p: (True, 400, None))
        monkeypatch.setattr(remote_module.STATE, "ssh", _FakeSSH())
        res = ff.fetch_one_file(later_remote, force=True, max_bytes=1000,
                                protect_paths={earlier_local})
        assert res.ok is True, res.error
        assert os.path.isfile(earlier_local)
        assert res.local_path and os.path.isfile(res.local_path)


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

    def test_root_serves_react_console_when_disconnected(self, app_with_no_ssh):
        client = app_with_no_ssh.test_client()
        r = client.get("/", follow_redirects=False)
        assert r.status_code == 200
        assert b'id="root"' in r.data

    def test_catalog_serves_react_console_when_disconnected(self, app_with_no_ssh):
        client = app_with_no_ssh.test_client()
        r = client.get("/catalog", follow_redirects=False)
        assert r.status_code == 200
        assert b'id="root"' in r.data

    def test_connection_error_page_reachable(self, app_with_no_ssh):
        client = app_with_no_ssh.test_client()
        r = client.get("/connection-error")
        assert r.status_code == 200
        assert b'id="root"' in r.data

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
        # EXPERIMENTAL-lane pages (HST supervision) are disabled by
        # default — none of their routes may exist in a stock app.
        for gated in ("/hst-psf", "/hst-cutouts", "/hst-tiles",
                      "/hst-pairs", "/roundtrip",
                      "/fasrc/tile/header", "/fasrc/tile/cutout.png"):
            assert gated not in urls, f"{gated} should be hidden by default"

    def test_experimental_lane_routes_register_when_enabled(
            self, experimental_lanes_on):
        from euclid_polish.web.app import create_app
        app = create_app()
        urls = {str(r) for r in app.url_map.iter_rules()}
        assert "/hst-psf" in urls
        assert "/hst-cutouts" in urls
        assert "/hst-psf/preview.png" in urls
        assert "/hst-cutouts/preview.png" in urls
        assert "/hst-tiles" in urls
        assert "/hst-pairs" in urls
        assert "/roundtrip" in urls
        assert "/fasrc/tile/header" in urls
        assert "/fasrc/tile/cutout.png" in urls


# ---------------------------------------------------------------------------
# Tile inspector — path safety + input validation
# ---------------------------------------------------------------------------

class TestTileInspectorRoutes:

    @pytest.fixture
    def client(self, experimental_lanes_on):
        # The tile inspector belongs to the EXPERIMENTAL HST lane —
        # enable the flag before building the app so its routes exist.
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
