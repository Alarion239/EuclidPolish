"""Tests for the universal FITS inspector + sky FITS exporter."""

from __future__ import annotations

import io
import os
import tempfile

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.web.app import create_app
from euclid_polish.web.helpers.fits_render import (
    _fits_file_info,
    _read_fits_header_rows,
)
from euclid_polish.web.helpers.paths import (
    _inspectable_roots,
    _resolve_inspectable_fits,
    _safe_relpath,
)
from euclid_polish.web.helpers.sky_render import _export_sky_record_fits

# ---------------------------------------------------------------------------
# Helpers used across tests
# ---------------------------------------------------------------------------

def _write_test_fits(path: str, *, shape=(8, 8), extra_keys=None) -> None:
    arr = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    hdu = fits.PrimaryHDU(arr)
    if extra_keys:
        for k, v in extra_keys.items():
            hdu.header[k] = v
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdu.writeto(path, overwrite=True)


@pytest.fixture
def cutout_fits(tmp_path, monkeypatch):
    """Place a fake FITS inside the configured cutouts root for a test."""
    # Redirect the cutouts root onto tmp_path so this test never touches
    # real on-disk data.
    monkeypatch.setattr(Config, "DEFAULT_OUTPUT_DIR", str(tmp_path / "cutouts_root"))
    root = Config.DEFAULT_OUTPUT_DIR
    p = os.path.join(root, "cutouts", "VIS", "star_0001_512.fits")
    _write_test_fits(p, extra_keys={
        "TESTKEY": ("hello", "test value"),
        "OBJECT":  "Inspector unit test",
    })
    return p


# ---------------------------------------------------------------------------
# Path-resolution primitives
# ---------------------------------------------------------------------------

class TestPathResolution:

    def test_inspectable_roots_present(self):
        roots = _inspectable_roots()
        assert len(roots) >= 1
        for r in roots:
            assert os.path.isabs(r)

    def test_resolve_rejects_empty(self):
        from flask import Flask
        app = Flask(__name__)
        with app.test_request_context():
            with pytest.raises(Exception) as exc:
                _resolve_inspectable_fits("")
            # werkzeug.exceptions.BadRequest raises HTTPException with code 400
            assert getattr(exc.value, "code", None) == 400

    def test_resolve_rejects_non_fits_extension(self, tmp_path):
        from flask import Flask
        app = Flask(__name__)
        bad = tmp_path / "thing.txt"
        bad.write_text("nope")
        with app.test_request_context():
            with pytest.raises(Exception) as exc:
                _resolve_inspectable_fits(str(bad))
            assert getattr(exc.value, "code", None) == 400

    def test_resolve_rejects_outside_roots(self, tmp_path):
        from flask import Flask
        app = Flask(__name__)
        # Real FITS but not under any inspectable root → 403.
        bad = tmp_path / "outside.fits"
        _write_test_fits(str(bad))
        with app.test_request_context():
            with pytest.raises(Exception) as exc:
                _resolve_inspectable_fits(str(bad))
            assert getattr(exc.value, "code", None) == 403

    def test_resolve_404_for_missing(self):
        from flask import Flask
        app = Flask(__name__)
        with app.test_request_context():
            with pytest.raises(Exception) as exc:
                _resolve_inspectable_fits("data/euclid_stars/cutouts/VIS/nope.fits")
            assert getattr(exc.value, "code", None) == 404

    def test_resolve_accepts_path_under_root(self, cutout_fits):
        from flask import Flask
        app = Flask(__name__)
        with app.test_request_context():
            resolved = _resolve_inspectable_fits(cutout_fits)
            assert resolved == os.path.realpath(cutout_fits)


# ---------------------------------------------------------------------------
# Header reader
# ---------------------------------------------------------------------------

class TestHeaderReader:

    def test_returns_one_row_per_hdu(self, cutout_fits):
        rows = _read_fits_header_rows(cutout_fits)
        assert len(rows) >= 1
        primary = rows[0]
        assert primary["hdu_index"] == 0
        assert primary["shape"] == [8, 8]
        # FITS stores big-endian by default → dtype like ">f4".
        # Just check the size: 4 bytes (float32).
        assert np.dtype(primary["dtype"]).itemsize == 4

    def test_includes_custom_card(self, cutout_fits):
        rows = _read_fits_header_rows(cutout_fits)
        keys = {c[0] for c in rows[0]["cards"]}
        assert "TESTKEY" in keys
        assert "OBJECT" in keys

    def test_file_info_basename_and_size(self, cutout_fits):
        info = _fits_file_info(cutout_fits)
        assert info["basename"] == "star_0001_512.fits"
        assert info["size_kb"] > 0


# ---------------------------------------------------------------------------
# Routes via test client
# ---------------------------------------------------------------------------

class TestInspectRoutes:

    @pytest.fixture
    def client(self):
        app = create_app()
        return app.test_client()

    def test_routes_are_registered(self):
        app = create_app()
        urls = {str(r) for r in app.url_map.iter_rules()}
        assert "/inspect" in urls
        assert "/api/inspect" in urls
        assert "/inspect/download" in urls
        assert "/inspect/preview.png" in urls
        assert "/sky/fits" in urls
        assert "/sky/inspect" in urls

    def test_inspect_with_unknown_path_404(self, client):
        r = client.get("/api/inspect?fits=data/euclid_stars/cutouts/VIS/nope.fits")
        assert r.status_code == 404

    def test_inspect_empty_400(self, client):
        r = client.get("/api/inspect?fits=")
        assert r.status_code == 400

    def test_inspect_renders_with_real_fits(self, client, cutout_fits):
        rel = _safe_relpath(os.path.realpath(cutout_fits))
        r = client.get(f"/api/inspect?fits={rel}")
        assert r.status_code == 200
        cards = [card for hdu in r.get_json()["hdus"] for card in hdu["cards"]]
        assert any(card[0] == "TESTKEY" for card in cards)
        assert any(card[0] == "OBJECT" for card in cards)

    def test_download_serves_fits_bytes(self, client, cutout_fits):
        rel = _safe_relpath(os.path.realpath(cutout_fits))
        r = client.get(f"/inspect/download?fits={rel}")
        assert r.status_code == 200
        assert r.data[:6] == b"SIMPLE"

    def test_preview_serves_png(self, client, cutout_fits):
        rel = _safe_relpath(os.path.realpath(cutout_fits))
        r = client.get(f"/inspect/preview.png?fits={rel}&size=64")
        assert r.status_code == 200
        assert r.data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_preview_size_clamp(self, client, cutout_fits):
        rel = _safe_relpath(os.path.realpath(cutout_fits))
        r = client.get(f"/inspect/preview.png?fits={rel}&size=99999")
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Sky FITS export
# ---------------------------------------------------------------------------

class TestSkyFitsExport:

    def test_rejects_bad_subset(self):
        from flask import Flask
        app = Flask(__name__)
        with app.test_request_context():
            with pytest.raises(Exception) as exc:
                _export_sky_record_fits("bogus", "clean", "VIS", 0)
            assert getattr(exc.value, "code", None) == 400

    def test_rejects_bad_kind(self):
        from flask import Flask
        app = Flask(__name__)
        with app.test_request_context():
            with pytest.raises(Exception) as exc:
                _export_sky_record_fits("train", "bogus", "VIS", 0)
            assert getattr(exc.value, "code", None) == 400

    def test_rejects_bad_band(self):
        from flask import Flask
        app = Flask(__name__)
        with app.test_request_context():
            with pytest.raises(Exception) as exc:
                _export_sky_record_fits("train", "clean", "NOPE", 0)
            assert getattr(exc.value, "code", None) == 400

    def test_rejects_negative_index(self):
        from flask import Flask
        app = Flask(__name__)
        with app.test_request_context():
            with pytest.raises(Exception) as exc:
                _export_sky_record_fits("train", "clean", "VIS", -1)
            assert getattr(exc.value, "code", None) == 400

    def test_404_when_no_records(self, tmp_path, monkeypatch):
        """When the requested TFRecord doesn't exist, abort with 404."""
        from flask import Flask
        app = Flask(__name__)
        # Point the records dir at an empty tmp path so the test doesn't
        # depend on what happens to be on disk and doesn't have to read
        # any real TFRecord.
        monkeypatch.setattr(Config, "RECORDS_DIR_V2", str(tmp_path / "empty"))
        with app.test_request_context():
            with pytest.raises(Exception) as exc:
                _export_sky_record_fits("train", "clean", "VIS", 0)
            assert getattr(exc.value, "code", None) == 404
