"""Unit tests for the bulk TNG50 SKIRT-atlas downloader.

Network-free: the API helpers (listing, streaming) are reused from the
single-galaxy script and are exercised elsewhere / monkeypatched here. We
cover the pure logic that's new in the bulk script — galaxy-id parsing,
Euclid-only extraction, the ``.done`` resumability skip, the success path
(with a stubbed stream), and API-key precedence.
"""

from __future__ import annotations

import io
import os
import tarfile

import pytest

from euclid_polish.config import Config
from scripts import fasrc_download_tng_skirt_atlas as mod

# ---------------------------------------------------------------------------
# galaxy-id parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("665976.tar.gz", "665976"),
    ("123.tgz", "123"),
    ("/a/b/665976.tar.gz", "665976"),
    ("plain", "plain"),
])
def test_galaxy_id_from_name(name, expected):
    assert mod._galaxy_id_from_name(name) == expected


# ---------------------------------------------------------------------------
# Euclid-only extraction
# ---------------------------------------------------------------------------

def _make_atlas_tar(path: str) -> None:
    """Write a tiny tar.gz mimicking a real atlas member layout."""
    keep = [
        "TNG123_O1_Euclid_VIS.fits",
        "TNG123_O1_Euclid_Y.fits",
        "TNG123_O5_Euclid_H.fits",
    ]
    drop = [
        "TNG123_O1_Euclid_VIS_nodust.fits",   # band token has '_' → excluded
        "TNG123_O1_2MASS_J.fits",             # other survey
        "readme.txt",                          # junk
    ]
    with tarfile.open(path, "w:gz") as tf:
        for member in keep + drop:
            data = b"\x00" * 16
            info = tarfile.TarInfo(name=f"TNG123/{member}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_extract_euclid_keeps_only_dusty_euclid(tmp_path):
    archive = tmp_path / "123.tar.gz"
    _make_atlas_tar(str(archive))
    dest = tmp_path / "123"
    n = mod._extract_euclid(str(archive), str(dest))
    assert n == 3
    written = sorted(os.listdir(dest))
    # Flattened to basenames, dusty Euclid frames only.
    assert written == [
        "TNG123_O1_Euclid_VIS.fits",
        "TNG123_O1_Euclid_Y.fits",
        "TNG123_O5_Euclid_H.fits",
    ]
    # The _nodust twin, the 2MASS frame and the junk file are all dropped.
    assert "TNG123_O1_Euclid_VIS_nodust.fits" not in written
    assert "TNG123_O1_2MASS_J.fits" not in written
    assert "readme.txt" not in written


# ---------------------------------------------------------------------------
# resumability + success path
# ---------------------------------------------------------------------------

def _complete_galaxy_dir(out_dir: str, gid: str, n_fits: int = 3) -> str:
    """A galaxy dir whose .done marker is BACKED by its promised FITS."""
    gdir = os.path.join(out_dir, gid)
    os.makedirs(gdir, exist_ok=True)
    for i in range(n_fits):
        with open(os.path.join(gdir, f"frame_{i}.fits"), "w") as f:
            f.write("x")
    with open(os.path.join(gdir, Config.Tng.DONE_MARKER), "w") as f:
        f.write(f"{n_fits} fits\n")
    return gdir


def test_download_one_skips_completed_galaxy(tmp_path, monkeypatch):
    """A galaxy whose .done marker is backed by its FITS is cached/skipped
    before any network call (the stream stub must never fire)."""
    out_dir = str(tmp_path)
    _complete_galaxy_dir(out_dir, "665976", n_fits=20)

    def _boom(*a, **k):
        raise AssertionError("stream attempted for an already-complete galaxy")

    monkeypatch.setattr(mod, "_stream_to_file", _boom)
    res = mod._download_one(
        name="665976.tar.gz", url=None, key="k",
        out_dir=out_dir, keep_archive=False)
    assert res["status"] == "cached"
    assert res["id"] == "665976"


def test_download_one_redownloads_gutted_galaxy(tmp_path, monkeypatch):
    """A .done marker whose FITS were swept away (2026-07-06 netscratch
    cleanup: hidden markers survived, every FITS deleted) must NOT count as
    cached — the galaxy re-downloads."""
    out_dir = str(tmp_path)
    gdir = os.path.join(out_dir, "777")
    os.makedirs(gdir)
    with open(os.path.join(gdir, Config.Tng.DONE_MARKER), "w") as f:
        f.write("20 fits\n")                    # marker present, zero FITS

    def _fake_stream(url, key, dest_path):
        _make_atlas_tar(dest_path)
        return os.path.getsize(dest_path)

    monkeypatch.setattr(mod, "_stream_to_file", _fake_stream)
    res = mod._download_one(
        name="777.tar.gz", url="http://example/777.tar.gz", key="k",
        out_dir=out_dir, keep_archive=False)
    assert res["status"] == "written"
    assert res["n_fits"] == 3


def test_download_one_force_redownloads_complete_galaxy(tmp_path, monkeypatch):
    """--force discards a genuinely complete galaxy (marker + FITS) and
    re-fetches it from scratch."""
    out_dir = str(tmp_path)
    gdir = _complete_galaxy_dir(out_dir, "777", n_fits=5)
    stale = os.path.join(gdir, "frame_0.fits")

    def _fake_stream(url, key, dest_path):
        _make_atlas_tar(dest_path)
        return os.path.getsize(dest_path)

    monkeypatch.setattr(mod, "_stream_to_file", _fake_stream)
    res = mod._download_one(
        name="777.tar.gz", url="http://example/777.tar.gz", key="k",
        out_dir=out_dir, keep_archive=False, force=True)
    assert res["status"] == "written"
    assert not os.path.exists(stale)            # old contents discarded


def test_galaxy_complete_legacy_marker(tmp_path):
    """A legacy marker without a parseable count needs at least one FITS."""
    gdir = os.path.join(str(tmp_path), "9")
    os.makedirs(gdir)
    with open(os.path.join(gdir, Config.Tng.DONE_MARKER), "w") as f:
        f.write("done")
    assert not mod._galaxy_complete(gdir)
    with open(os.path.join(gdir, "a.fits"), "w") as f:
        f.write("x")
    assert mod._galaxy_complete(gdir)


def test_tng_step_forwards_force_flag():
    from euclid_polish.web.fasrc_pipeline import REGISTRY
    step = REGISTRY.get("download_tng_skirt")
    assert "--force" in step.build_command({"n_cpus": 4, "force": "1"})
    assert "--force" not in step.build_command({"n_cpus": 4})


def test_download_one_success_writes_marker(tmp_path, monkeypatch):
    """Happy path: stream (stubbed) → extract Euclid frames → write .done,
    and the transient archive is removed (keep_archive=False)."""
    out_dir = str(tmp_path)

    def _fake_stream(url, key, dest_path):
        _make_atlas_tar(dest_path)
        return os.path.getsize(dest_path)

    monkeypatch.setattr(mod, "_stream_to_file", _fake_stream)

    res = mod._download_one(
        name="777.tar.gz", url="http://example/777.tar.gz", key="k",
        out_dir=out_dir, keep_archive=False)
    assert res["status"] == "written"
    assert res["id"] == "777"
    assert res["n_fits"] == 3
    gdir = os.path.join(out_dir, "777")
    assert os.path.isfile(os.path.join(gdir, Config.Tng.DONE_MARKER))
    # 3 FITS, no leftover archive, no stray temp dirs under out_dir.
    fits = [f for f in os.listdir(gdir) if f.endswith(".fits")]
    assert len(fits) == 3
    assert not any(f.endswith(".tar.gz") for f in os.listdir(gdir))
    assert list(os.listdir(out_dir)) == ["777"]


def test_download_one_keep_archive(tmp_path, monkeypatch):
    """--keep-archive moves the source tarball next to the extracted FITS."""
    out_dir = str(tmp_path)
    monkeypatch.setattr(
        mod, "_stream_to_file",
        lambda url, key, dest: (_make_atlas_tar(dest), os.path.getsize(dest))[1])
    res = mod._download_one(
        name="888.tar.gz", url=None, key="k",
        out_dir=out_dir, keep_archive=True)
    assert res["status"] == "written"
    gdir = os.path.join(out_dir, "888")
    assert any(f.endswith(".tar.gz") for f in os.listdir(gdir))


def test_download_one_empty_archive_is_failure(tmp_path, monkeypatch):
    """A tarball with no Euclid members → failed (no .done written)."""
    out_dir = str(tmp_path)

    def _fake_stream(url, key, dest_path):
        with tarfile.open(dest_path, "w:gz") as tf:
            data = b"x"
            info = tarfile.TarInfo(name="junk/readme.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        return os.path.getsize(dest_path)

    monkeypatch.setattr(mod, "_stream_to_file", _fake_stream)
    res = mod._download_one(
        name="999.tar.gz", url=None, key="k",
        out_dir=out_dir, keep_archive=False)
    # The fallback extracts the junk file, so n_fits == 0 → failure, no marker.
    assert res["status"] == "failed"
    assert not os.path.isfile(
        os.path.join(out_dir, "999", Config.Tng.DONE_MARKER))


# ---------------------------------------------------------------------------
# API-key precedence: arg → env → file → error
# ---------------------------------------------------------------------------

def _args(**kw):
    import argparse
    ns = argparse.Namespace(
        api_key="", api_key_file="~/.tng_api_key")
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_load_key_prefers_explicit_arg(monkeypatch):
    monkeypatch.setenv("TNG_API_KEY", "from-env")
    assert mod._load_key(_args(api_key="from-arg")) == "from-arg"


def test_load_key_uses_env_when_no_arg(monkeypatch):
    monkeypatch.setenv("TNG_API_KEY", "from-env")
    assert mod._load_key(_args()) == "from-env"


def test_load_key_reads_file_when_no_arg_or_env(tmp_path, monkeypatch):
    monkeypatch.delenv("TNG_API_KEY", raising=False)
    keyfile = tmp_path / "tngkey"
    keyfile.write_text("from-file\n")
    assert mod._load_key(_args(api_key_file=str(keyfile))) == "from-file"


def test_load_key_exits_when_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("TNG_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        mod._load_key(_args(api_key_file=str(tmp_path / "does-not-exist")))
