"""Inspect workspace backend (spec §8.6): roots + browser, header-first HDU
summaries, planes of any dimensionality (binning, scaling, WCS), statistics,
table pages / column stats, provenance lookup, the ``fits`` viewer collection
and the JSON error surface of the ``/api/inspect*`` endpoints."""

from __future__ import annotations

import gzip
import json
import os
from io import BytesIO

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image
from werkzeug.exceptions import HTTPException

from euclid_polish.config import Config
from euclid_polish.photometry import adu_per_s_to_electrons_factor
from euclid_polish.web.app import create_app
from euclid_polish.web.helpers import fits_inspect, paths, viewer_data

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _tan_header(nx: int, ny: int, *, ra: float = 150.0, dec: float = 2.0,
                scale_arcsec: float = 0.1) -> fits.Header:
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    header["CRVAL1"], header["CRVAL2"] = ra, dec
    header["CRPIX1"], header["CRPIX2"] = (nx + 1) / 2.0, (ny + 1) / 2.0
    header["CD1_1"], header["CD1_2"] = -scale_arcsec / 3600.0, 0.0
    header["CD2_1"], header["CD2_2"] = 0.0, scale_arcsec / 3600.0
    return header


_REAL_ROOT_SPECS = paths._root_specs


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """Point the data roots at tmp_path and keep only those, so no test (a
    search across every root included) touches real data."""
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(eval_dir))
    monkeypatch.setattr(Config, "TRACKING_DIR", str(tmp_path / "tracking"))
    monkeypatch.setattr(Config, "PROV_DIR", str(tmp_path / "prov"))
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR", str(tmp_path / "ckpt" / "wdsr"))
    base = os.path.realpath(tmp_path)
    monkeypatch.setattr(paths, "_root_specs", lambda: [
        spec for spec in _REAL_ROOT_SPECS()
        if os.path.realpath(spec[2]).startswith(base + os.sep)])
    return {"eval": eval_dir, "tmp": tmp_path}


def _write(path, hdus) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return str(path)


@pytest.fixture
def band_cube(roots):
    """(4, 16, 20) electron stack with BANDS + TAN WCS (an eval SR.fits)."""
    data = np.arange(4 * 16 * 20, dtype=np.float32).reshape(4, 16, 20)
    header = _tan_header(20, 16)
    header["BANDS"] = "VIS,Y_E,J_E,H_E"
    header["BUNIT"] = "electron"
    return _write(roots["eval"] / "obj" / "SR.fits", [fits.PrimaryHDU(data, header=header)])


@pytest.fixture
def poster_like(roots):
    """Primary RA/DEC/PIXSCALE (no WCS) + LR_<band>/SR_<band> image HDUs."""
    primary = fits.PrimaryHDU()
    primary.header["RA"], primary.header["DEC"], primary.header["PIXSCALE"] = 273.23, 68.36, 0.1
    primary.header["BUNIT"] = "electron"
    hdus = [primary]
    for prefix, side, scale in (("LR_", 8, 0.1), ("SR_", 16, 0.05)):
        for k, band in enumerate(("VIS", "Y_E", "J_E", "H_E")):
            hdu = fits.ImageHDU(np.full((side, side), k + 1, np.float32), name=f"{prefix}{band}")
            hdu.header["PIXSCALE"] = scale
            hdus.append(hdu)
    return _write(roots["eval"] / "poster_results.fits", hdus)


@pytest.fixture
def table_file(roots):
    cols = [
        fits.Column(name="id", format="J", array=np.array([3, 1, 2, 5, 4], np.int32)),
        fits.Column(name="flux", format="E", unit="e-",
                    array=np.array([1.5, np.nan, 3.0, -2.0, 10.0], np.float32)),
        fits.Column(name="name", format="6A", array=np.array(["gal", "star", "gal", "qso", "Gal"])),
        fits.Column(name="ok", format="L", array=np.array([True, False, True, True, False])),
        fits.Column(name="vec", format="3E", array=np.arange(15, dtype=np.float32).reshape(5, 3)),
        fits.Column(name="big", format="20E", array=np.ones((5, 20), np.float32)),
    ]
    table = fits.BinTableHDU.from_columns(cols, name="CAT")
    return _write(roots["eval"] / "cat.fits", [fits.PrimaryHDU(), table])


@pytest.fixture
def client(roots):
    return create_app().test_client()


def _rel(path: str) -> str:
    return paths._safe_relpath(os.path.realpath(path))


# ---------------------------------------------------------------------------
# roots + browser
# ---------------------------------------------------------------------------

class TestRoots:

    def test_roots_cover_the_spec_list(self, roots):
        ids = {spec[0] for spec in _REAL_ROOT_SPECS()}
        for wanted in ("eval", "viewer-results", "jwst", "sky", "inference", "poster",
                       "output", "tracking", "stars", "psf", "fasrc-cache"):
            assert wanted in ids
        assert {r["id"] for r in paths.inspect_roots()} == {"eval", "tracking"}
        eval_root = next(r for r in paths.inspect_roots() if r["id"] == "eval")
        assert eval_root["path"] == os.path.realpath(roots["eval"])
        assert eval_root["exists"] is True
        tracking = next(r for r in paths.inspect_roots() if r["id"] == "tracking")
        assert tracking["exists"] is False           # tmp/tracking was never created
        assert os.path.realpath(roots["eval"]) in paths._inspectable_roots()

    def test_poster_and_output_are_repo_directories(self):
        by_id = {spec[0]: spec[2] for spec in _REAL_ROOT_SPECS()}
        assert os.path.realpath(by_id["poster"]) == os.path.realpath(paths.REPO_ROOT / "poster")
        assert os.path.realpath(by_id["output"]) == os.path.realpath(paths.REPO_ROOT / "output")
        assert os.path.realpath(by_id["tracking"]) == os.path.realpath(Config.TRACKING_DIR)

    def test_browse_without_dir_lists_existing_roots(self, client, roots):
        body = client.get("/api/inspect/browse").get_json()
        assert body["dir"] is None
        kinds = {e["kind"] for e in body["entries"]}
        assert kinds == {"root"}
        rels = {e["rel"] for e in body["entries"]}
        assert _rel(str(roots["eval"])) in rels
        assert all(e["exists"] for e in body["entries"])
        assert any(r["id"] == "tracking" and not r["exists"] for r in body["roots"])

    def test_browse_lists_dirs_then_fits_and_counts_others(self, client, band_cube, roots):
        (roots["eval"] / "notes.txt").write_text("x")
        (roots["eval"] / ".hidden").mkdir()
        _write(roots["eval"] / "b.fits.gz", [fits.PrimaryHDU(np.zeros((2, 2), np.float32))])
        body = client.get(f"/api/inspect/browse?dir={_rel(str(roots['eval']))}").get_json()
        names = [(e["kind"], e["name"]) for e in body["entries"]]
        assert names == [("dir", "obj"), ("fits", "b.fits.gz")]
        assert body["other"] == 1
        assert body["crumbs"][0]["name"] == "Evaluation results"
        assert body["root"]["id"] == "eval"
        sub = client.get(f"/api/inspect/browse?dir={_rel(str(roots['eval'] / 'obj'))}").get_json()
        assert [c["name"] for c in sub["crumbs"]] == ["Evaluation results", "obj"]
        assert [e["name"] for e in sub["entries"]] == ["SR.fits"]
        assert sub["entries"][0]["size"] > 0

    def test_browse_drops_symlinks_out_of_the_roots(self, client, roots):
        outside = roots["tmp"] / "outside"
        outside.mkdir()
        os.symlink(outside, roots["eval"] / "escape")
        body = client.get(f"/api/inspect/browse?dir={_rel(str(roots['eval']))}").get_json()
        assert "escape" not in [e["name"] for e in body["entries"]]
        denied = client.get(f"/api/inspect/browse?dir={_rel(str(roots['eval'] / 'escape'))}")
        assert denied.status_code == 403
        assert "outside the inspectable" in denied.get_json()["error"]

    def test_browse_outside_and_missing_are_json_errors(self, client, roots):
        outside = client.get(f"/api/inspect/browse?dir={roots['tmp']}")
        assert outside.status_code == 403 and "error" in outside.get_json()
        missing = client.get(f"/api/inspect/browse?dir={roots['eval'] / 'nope'}")
        assert missing.status_code == 404 and "no such directory" in missing.get_json()["error"]

    def test_search_matches_every_token_under_a_dir(self, client, band_cube, poster_like, roots):
        base = _rel(str(roots["eval"]))
        hits = client.get(f"/api/inspect/browse?dir={base}&q=obj%20sr").get_json()
        assert [e["name"] for e in hits["entries"]] == ["SR.fits"]
        assert hits["truncated"] is False
        everywhere = client.get("/api/inspect/browse?q=poster").get_json()
        assert [e["name"] for e in everywhere["entries"]] == ["poster_results.fits"]

    def test_search_is_bounded(self, roots, monkeypatch):
        for k in range(5):
            _write(roots["eval"] / f"f{k}.fits", [fits.PrimaryHDU(np.zeros((2, 2), np.float32))])
        monkeypatch.setattr(paths, "SEARCH_MAX_RESULTS", 2)
        found = paths.search_inspect_tree([str(roots["eval"])], "f")
        assert len(found["entries"]) == 2 and found["truncated"] is True

    def test_trackable_files_follow_the_roots_with_messages(self, band_cube, roots):
        """The Track button backs up any inspectable file (any extension)."""
        png = roots["eval"] / "panel.png"
        png.write_bytes(b"\x89PNG")
        with create_app().test_request_context():
            assert paths._resolve_trackable_file(_rel(band_cube)) == os.path.realpath(band_cube)
            assert paths._resolve_trackable_file(_rel(str(png))) == os.path.realpath(png)
            outside = roots["tmp"] / "secret.txt"
            outside.write_text("x")
            for raw, code, text in ((str(outside), 403, "outside the inspectable"),
                                    (_rel(str(roots["eval"] / "nope.fits")), 404, "no such file"),
                                    ("", 400, "pass path=")):
                with pytest.raises(HTTPException) as err:
                    paths._resolve_trackable_file(raw)
                assert err.value.code == code and text in (err.value.description or "")

    def test_track_path_errors_are_json_on_any_route(self, roots, monkeypatch):
        """/api/tracking is not a JSON-error prefix: the resolver's own
        response still carries {ok:false, error} so the Track toast shows it."""
        def never(*_args):
            raise AssertionError("a refused path must not be backed up")
        store = type("Store", (), {"backup_fits": staticmethod(never)})()
        monkeypatch.setattr("euclid_polish.web.routes.tracking.tracking_default_store", lambda: store)
        outside = roots["tmp"] / "secret.txt"
        outside.write_text("x")
        client = create_app().test_client()
        for raw, code, text in ((str(outside), 403, "outside the inspectable"),
                                ("data/eval_results/nope.fits", 404, "no such file"),
                                ("", 400, "pass path=")):
            res = client.post("/api/tracking/backup", data={"kind": "fits", "path": raw})
            assert res.status_code == code and res.is_json, (raw, res.data[:80])
            body = res.get_json()
            assert body["ok"] is False and text in body["error"]

    def test_fz_and_gz_are_fits_names(self):
        assert paths.is_fits_name("a.fits.fz") and paths.is_fits_name("A.FITS.GZ")
        assert not paths.is_fits_name("a.json")


# ---------------------------------------------------------------------------
# header summaries
# ---------------------------------------------------------------------------

class TestSummary:

    def test_band_cube_summary(self, band_cube):
        summary = fits_inspect.file_summary(band_cube)
        (hdu,) = summary["hdus"]
        assert hdu["type"] == "image" and hdu["shape"] == [4, 16, 20]
        assert hdu["planes"] == 4 and hdu["plane_axes"] == [4]
        assert hdu["bands"] == ["VIS", "Y_E", "J_E", "H_E"] and hdu["bands_assumed"] is False
        assert hdu["bunit"] == "electron"
        wcs = hdu["wcs"]
        assert wcs["ra"] == pytest.approx(150.0, abs=1e-6) and wcs["dec"] == pytest.approx(2.0, abs=1e-6)
        assert wcs["pixscale_arcsec"] == pytest.approx(0.1)
        assert wcs["width_arcsec"] == pytest.approx(2.0) and wcs["constructed"] is False
        assert len(wcs["corners"]) == 4

    def test_poster_band_groups_and_constructed_wcs(self, poster_like):
        summary = fits_inspect.file_summary(poster_like)
        groups = {g["id"]: g for g in summary["band_groups"]}
        assert set(groups) == {"b:LR_", "b:SR_"}
        assert groups["b:LR_"]["hdus"] == [1, 2, 3, 4] and groups["b:SR_"]["shape"] == [16, 16]
        lr_vis = summary["hdus"][1]
        assert lr_vis["band"] == "VIS" and lr_vis["bunit"] == "electron"   # inherited
        assert lr_vis["wcs"]["constructed"] is True
        assert lr_vis["wcs"]["ra"] == pytest.approx(273.23)
        assert summary["hdus"][5]["wcs"]["pixscale_arcsec"] == pytest.approx(0.05)
        assert summary["hdus"][0]["type"] == "empty"

    def test_bare_nisp_band_names_group(self, roots):
        """poster/Gal_O1_bands.fits names its HDUs VIS, Y, J, H."""
        hdus = [fits.PrimaryHDU()]
        for k, name in enumerate(("VIS", "Y", "J", "H")):
            hdus.append(fits.ImageHDU(np.full((6, 6), k, np.float32), name=name))
        # NISP_<band> spellings and a FILTER card group the same way
        for k, name in enumerate(("LR_VIS", "LR_NISP_Y", "LR_NISP_J", "LR_NISP_H")):
            hdus.append(fits.ImageHDU(np.full((4, 4), k, np.float32), name=name))
        for k, filt in enumerate(("VIS", "NISP_Y", "J", "H")):
            hdu = fits.ImageHDU(np.full((5, 5), k, np.float32), name=f"IMG{k}")
            hdu.header["FILTER"] = filt
            hdus.append(hdu)
        # a word merely ending in H/J/Y is no band
        hdus.append(fits.ImageHDU(np.zeros((3, 3), np.float32), name="DEPTH"))
        path = _write(roots["eval"] / "bands.fits", hdus)
        summary = fits_inspect.file_summary(path)
        bands = [h.get("band") for h in summary["hdus"]]
        assert bands[1:5] == ["VIS", "Y_E", "J_E", "H_E"]
        assert bands[5:9] == ["VIS", "Y_E", "J_E", "H_E"]
        assert bands[9:13] == ["VIS", "Y_E", "J_E", "H_E"]
        assert bands[13] is None
        groups = {g["id"]: g for g in summary["band_groups"]}
        # (FILTER-only HDUs share the empty prefix; the first four of it win)
        assert set(groups) == {"b:", "b:LR_"}
        assert groups["b:"]["hdus"] == [1, 2, 3, 4] and groups["b:"]["shape"] == [6, 6]
        assert groups["b:LR_"]["hdus"] == [5, 6, 7, 8]

    def test_scaled_ints_vectors_and_tables_from_headers(self, roots, table_file):
        ints = fits.PrimaryHDU(np.arange(12, dtype=np.int16).reshape(3, 4))
        ints.header["BSCALE"], ints.header["BZERO"] = 2.0, 10.0
        vec = fits.ImageHDU(np.arange(7, dtype=np.float64), name="SPEC")
        path = _write(roots["eval"] / "mixed.fits", [ints, vec])
        hdus = fits_inspect.file_summary(path)["hdus"]
        assert hdus[0]["dtype"] == ">i2" and hdus[0]["scaling"] == {"bscale": 2.0, "bzero": 10.0}
        assert hdus[1]["type"] == "vector" and hdus[1]["viewable"] is False
        table = fits_inspect.file_summary(table_file)["hdus"][1]
        assert table["type"] == "table" and table["nrows"] == 5 and table["ncols"] == 6
        assert table["columns"][1] == {"name": "flux", "format": "E", "unit": "e-", "dim": None, "null": None}

    def test_stamp_is_read_from_the_primary(self, roots):
        hdu = fits.PrimaryHDU(np.zeros((2, 2), np.float32))
        hdu.header["PROVID"], hdu.header["PRODBY"], hdu.header["PROVPAR"] = "aaaaaaaa", "bbbbbbbb", "cccccccc"
        path = _write(roots["eval"] / "stamped.fits", [hdu])
        assert fits_inspect.file_summary(path)["stamp"]["id"] == "aaaaaaaa"

    def test_large_gzip_lists_only_the_primary(self, roots, monkeypatch):
        path = str(roots["eval"] / "big.fits.gz")
        raw = roots["eval"] / "big.fits"
        _write(raw, [fits.PrimaryHDU(np.zeros((4, 4), np.float32)), fits.ImageHDU(np.zeros((4, 4)))])
        with open(raw, "rb") as src, gzip.open(path, "wb") as dst:
            dst.write(src.read())
        monkeypatch.setattr(fits_inspect, "GZIP_SCAN_BYTES", 1)
        summary = fits_inspect.file_summary(path, cards=True)
        assert len(summary["hdus"]) == 1 and summary["scan_truncated"] is True
        assert summary["hdus"][0]["shape"] == [4, 4] and summary["hdus"][0]["cards"]
        assert fits_inspect.read_plane(path, 0).data.shape == (4, 4)
        with pytest.raises(fits_inspect.InspectError) as exc:
            fits_inspect.read_plane(path, 1)
        assert exc.value.code == 413


# ---------------------------------------------------------------------------
# planes
# ---------------------------------------------------------------------------

class TestPlanes:

    def test_scaling_and_blank(self, roots):
        hdu = fits.PrimaryHDU(np.arange(6, dtype=np.int16).reshape(2, 3))
        hdu.header["BSCALE"], hdu.header["BZERO"], hdu.header["BLANK"] = 2.0, 10.0, 4
        path = _write(roots["eval"] / "ints.fits", [hdu])
        served = fits_inspect.read_plane(path, 0)
        assert served.data.dtype == np.float32
        np.testing.assert_array_equal(served.data[0], [10.0, 12.0, 14.0])
        assert np.isnan(served.data[1, 1]) and served.data[1, 2] == 20.0

    def test_nd_plane_index(self, roots):
        data = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
        path = _write(roots["eval"] / "nd.fits", [fits.PrimaryHDU(data)])
        served = fits_inspect.read_plane(path, 0, 4)
        assert served.index == (1, 1)
        np.testing.assert_array_equal(served.data, data[1, 1])
        assert fits_inspect.plane_label(data.shape, 4) == "[1, 1]"
        assert fits_inspect.plane_label((3, 4, 5), 2) == "plane 2"
        assert fits_inspect.plane_label((4, 4, 5), 1, ["VIS", "Y_E", "J_E", "H_E"]) == "Y_E"
        with pytest.raises(fits_inspect.InspectError) as exc:
            fits_inspect.read_plane(path, 0, 6)
        assert exc.value.code == 404

    def test_block_mean_bin_and_its_wcs(self, roots):
        data = np.arange(64, dtype=np.float32).reshape(8, 8)
        header = _tan_header(8, 8)
        path = _write(roots["eval"] / "grid.fits", [fits.PrimaryHDU(data, header=header)])
        served = fits_inspect.read_plane(path, 0, bin=2)
        assert served.method == "mean" and served.bin == 2 and served.offset == 0.5
        assert served.data.shape == (4, 4)
        assert served.data[0, 0] == pytest.approx(np.mean([0, 1, 8, 9]))
        source = WCS(header)
        binned = fits_inspect.binned_wcs(source, served.bin, served.offset)
        # served pixel (1, 2) is centred on source pixel (2·1 + 0.5, 2·2 + 0.5)
        np.testing.assert_allclose(binned.pixel_to_world_values(1, 2),
                                   source.pixel_to_world_values(2.5, 4.5), atol=1e-10)

    def test_stride_sample_above_the_block_limit(self, roots, monkeypatch):
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        path = _write(roots["eval"] / "big.fits", [fits.PrimaryHDU(data)])
        monkeypatch.setattr(fits_inspect, "BLOCK_MEAN_MAX_PIXELS", 10)
        served = fits_inspect.read_plane(path, 0, bin=3)
        assert served.method == "stride" and served.offset == 1.0
        np.testing.assert_array_equal(served.data, data[1::3, 1::3])

    def test_auto_bin_and_output_cap(self, roots, monkeypatch):
        path = _write(roots["eval"] / "wide.fits", [fits.PrimaryHDU(np.zeros((6, 40), np.float32))])
        monkeypatch.setattr(fits_inspect, "MAX_VIEW_SIDE", 10)
        assert fits_inspect.read_plane(path, 0).bin == 4
        monkeypatch.setattr(fits_inspect, "MAX_OUTPUT_SIDE", 20)
        assert fits_inspect.read_plane(path, 0, bin=1, max_side=1000).bin == 2   # the cap wins
        assert fits_inspect.choose_bin(10, 10, None, 2048) == 1

    def test_compressed_plane_too_large_is_413(self, roots, monkeypatch):
        path = str(roots["eval"] / "c.fits.gz")
        raw = roots["eval"] / "c.fits"
        _write(raw, [fits.PrimaryHDU(np.zeros((8, 8), np.float32))])
        with open(raw, "rb") as src, gzip.open(path, "wb") as dst:
            dst.write(src.read())
        assert fits_inspect.read_plane(path, 0).data.shape == (8, 8)
        monkeypatch.setattr(fits_inspect, "MAX_DECOMPRESS_BYTES", 16)
        with pytest.raises(fits_inspect.InspectError) as exc:
            fits_inspect.read_plane(path, 0)
        assert exc.value.code == 413 and "download" in str(exc.value)

    def test_tile_compressed_hdu(self, roots):
        data = np.arange(30 * 40, dtype=np.float32).reshape(30, 40)
        path = _write(roots["eval"] / "t.fits.fz", [fits.PrimaryHDU(), fits.CompImageHDU(data, quantize_level=0)])
        summary = fits_inspect.file_summary(path)["hdus"][1]
        assert summary["compressed"] is True and summary["shape"] == [30, 40]
        np.testing.assert_allclose(fits_inspect.read_plane(path, 1).data, data)

    def test_non_images_are_415(self, table_file):
        with pytest.raises(fits_inspect.InspectError) as exc:
            fits_inspect.read_plane(table_file, 1)
        assert exc.value.code == 415


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

class TestStats:

    def test_array_stats(self):
        values = np.array([[1.0, 2.0, np.nan], [3.0, np.inf, -4.0]])
        stats = fits_inspect.array_stats(values, bins=4)
        assert stats["n"] == 6 and stats["n_finite"] == 4 and stats["n_nan"] == 1
        assert stats["n_posinf"] == 1 and stats["n_negative"] == 1
        assert stats["min"] == -4.0 and stats["max"] == 3.0 and stats["mean"] == pytest.approx(0.5)
        assert stats["median"] == pytest.approx(1.5)
        hist = stats["histogram"]
        assert len(hist["edges"]) == 5
        assert sum(hist["counts"]) + hist["below"] + hist["above"] == 4

    def test_all_nan_plane(self):
        stats = fits_inspect.array_stats(np.full((3, 3), np.nan))
        assert stats["n_finite"] == 0 and stats["min"] is None and stats["histogram"] is None

    def test_plane_stats_samples_large_planes(self, band_cube, monkeypatch):
        stats = fits_inspect.plane_stats(band_cube, 0, 2)
        assert stats["plane"] == 2 and stats["sampled"] is None
        assert stats["min"] == 2 * 16 * 20 and stats["sum"] is not None
        monkeypatch.setattr(fits_inspect, "MAX_STATS_PIXELS", 16)
        sampled = fits_inspect.plane_stats(band_cube, 0, 0)
        assert sampled["sampled"] > 1 and sampled["sum"] is None

    def test_vector_series(self, roots, monkeypatch):
        path = _write(roots["eval"] / "v.fits", [fits.PrimaryHDU(np.arange(10, dtype=np.float32))])
        monkeypatch.setattr(fits_inspect, "VECTOR_MAX_POINTS", 4)
        series = fits_inspect.vector_series(path, 0)
        assert series["step"] == 3 and series["series"]["x"] == [0, 3, 6, 9]
        assert series["series"]["y"] == [0.0, 3.0, 6.0, 9.0] and series["n_points"] == 10


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

class TestTables:

    def test_page_and_cells(self, table_file):
        page = fits_inspect.table_page(table_file, 1, offset=1, limit=2)
        assert page["total"] == 5 and page["row_index"] == [1, 2]
        kinds = {c["name"]: c["kind"] for c in page["columns"]}
        assert kinds == {"id": "numeric", "flux": "numeric", "name": "text", "ok": "bool",
                         "vec": "array", "big": "array"}
        first = page["rows"][0]
        assert first[:4] == [1, None, "star", False]            # NaN → null
        assert first[4] == [3.0, 4.0, 5.0]
        assert first[5].startswith("[1, 1, 1, … ×20")

    def test_sort_desc_puts_nan_last(self, table_file):
        page = fits_inspect.table_page(table_file, 1, sort="flux", desc=True)
        flux = [row[1] for row in page["rows"]]
        assert flux == [10.0, 3.0, 1.5, -2.0, None]
        by_name = fits_inspect.table_page(table_file, 1, sort="name")
        assert [row[2] for row in by_name["rows"]][:3] == ["gal", "gal", "Gal"]   # case-insensitive, stable
        for bad in ("nope", "vec"):
            with pytest.raises(fits_inspect.InspectError) as exc:
                fits_inspect.table_page(table_file, 1, sort=bad)
            assert exc.value.code == 400

    def test_offset_past_the_end_and_limit_cap(self, table_file):
        assert fits_inspect.table_page(table_file, 1, offset=50)["rows"] == []
        assert fits_inspect.table_page(table_file, 1, limit=10**9)["limit"] == fits_inspect.TABLE_PAGE_MAX

    def test_column_stats(self, table_file):
        stats = {c["name"]: c for c in fits_inspect.table_stats(table_file, 1)["columns"]}
        assert stats["flux"]["n_null"] == 1 and stats["flux"]["max"] == 10.0
        assert stats["id"]["n_unique"] == 5
        assert stats["name"]["n_unique"] == 4 and stats["name"]["top"][0] == ["gal", 2]
        assert stats["ok"]["n_true"] == 3 and stats["ok"]["n_false"] == 2
        assert stats["vec"]["kind"] == "array" and stats["vec"]["shape"] == [3]


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

class TestProvenance:

    def test_stamp_sidecars_and_related_records(self, roots):
        hdu = fits.PrimaryHDU(np.zeros((2, 2), np.float32))
        hdu.header["PROVID"], hdu.header["PRODBY"], hdu.header["PROVPAR"] = "aaaaaaaa", "bbbbbbbb", "cccccccc"
        obj = roots["eval"] / "gal"
        path = _write(obj / "SR.fits", [hdu])
        rel_record_path = os.path.relpath(path, os.getcwd())
        (obj / "aaaaaaaa.srcutoutartifact.json").write_text(json.dumps({
            "id": "aaaaaaaa", "kind": "srcutoutartifact", "path": rel_record_path,
            "produced_by": "bbbbbbbb", "parents": ["cccccccc"], "created_at": "2026-07-02"}))
        (obj / "dddddddd.srcutoutartifact.json").write_text(json.dumps({
            "id": "dddddddd", "kind": "srcutoutartifact", "path": rel_record_path,
            "created_at": "2026-07-01"}))
        (obj / "eeeeeeee.srcutoutartifact.json").write_text(json.dumps({
            "id": "eeeeeeee", "kind": "srcutoutartifact", "path": "./elsewhere/SR.fits"}))
        prov = roots["tmp"] / "prov"
        prov.mkdir()
        (prov / "bbbbbbbb.inferencerun.json").write_text(json.dumps({"id": "bbbbbbbb", "kind": "inferencerun"}))
        member = roots["tmp"] / "ckpt" / "ensemble" / "member_7"
        member.mkdir(parents=True)
        (member / "provenance.json").write_text(json.dumps({"id": "cccccccc", "parents": []}))

        out = fits_inspect.provenance(path)
        assert out["stamp"]["id"] == "aaaaaaaa"
        assert [s["id"] for s in out["sidecars"]] == ["aaaaaaaa", "dddddddd"]
        assert out["sidecars"][0]["current"] is True and out["stale_sidecars"] == 1
        related = {r["id"]: r for r in out["related"]}
        assert related["bbbbbbbb"]["role"] == "produced_by" and related["bbbbbbbb"]["kind"] == "inferencerun"
        assert related["cccccccc"]["kind"] == "checkpoint"
        assert related["cccccccc"]["checkpoint"].endswith("member_7")

    def test_unstamped_file(self, band_cube):
        out = fits_inspect.provenance(band_cube)
        assert out == {"stamp": None, "sidecars": [], "related": [], "stale_sidecars": 0}


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

class TestRoutes:

    def test_rules_are_registered(self):
        urls = {str(r) for r in create_app().url_map.iter_rules()}
        for rule in ("/api/inspect", "/api/inspect/browse", "/api/inspect/image/stats",
                     "/api/inspect/table", "/api/inspect/table/stats", "/api/inspect/provenance",
                     "/inspect/download", "/inspect/preview.png"):
            assert rule in urls

    def test_inspect_payload(self, client, poster_like):
        body = client.get(f"/api/inspect?fits={_rel(poster_like)}").get_json()
        assert body["file"]["basename"] == "poster_results.fits" and body["file"]["size"] > 0
        assert len(body["hdus"]) == 9 and body["hdus"][1]["cards"]
        assert {g["id"] for g in body["band_groups"]} == {"b:LR_", "b:SR_"}
        assert body["root"]["id"] == "eval" and body["roots"]

    def test_errors_are_json(self, client, roots, table_file):
        assert client.get("/api/inspect?fits=").get_json()["error"]
        r = client.get(f"/api/inspect?fits={roots['tmp'] / 'x.fits'}")
        assert r.status_code == 404 and "no such FITS" in r.get_json()["error"]
        outside = roots["tmp"] / "outside.fits"
        _write(outside, [fits.PrimaryHDU(np.zeros((2, 2), np.float32))])
        r = client.get(f"/api/inspect?fits={outside}")
        assert r.status_code == 403 and "outside" in r.get_json()["error"]
        r = client.get(f"/api/inspect/image/stats?fits={_rel(table_file)}&hdu=1")
        assert r.status_code == 415 and "not an image" in r.get_json()["error"]
        r = client.get(f"/api/inspect/table?fits={_rel(table_file)}&hdu=1&limit=0")
        assert r.status_code == 400 and "limit" in r.get_json()["error"]
        r = client.get(f"/api/inspect/table?fits={_rel(table_file)}&hdu=x")
        assert r.status_code == 400
        r = client.get(f"/inspect/preview.png?fits={_rel(table_file)}&size=99999")
        assert r.status_code == 400 and r.get_json()["error"]

    def test_image_stats_table_and_provenance_routes(self, client, band_cube, table_file, roots):
        stats = client.get(f"/api/inspect/image/stats?fits={_rel(band_cube)}&hdu=0&plane=1").get_json()
        assert stats["plane"] == 1 and stats["min"] == 16 * 20
        page = client.get(f"/api/inspect/table?fits={_rel(table_file)}&hdu=1&limit=2&sort=id").get_json()
        assert [row[0] for row in page["rows"]] == [1, 2]
        cols = client.get(f"/api/inspect/table/stats?fits={_rel(table_file)}&hdu=1").get_json()
        assert len(cols["columns"]) == 6
        prov = client.get(f"/api/inspect/provenance?fits={_rel(band_cube)}").get_json()
        assert prov["stamp"] is None
        vec_path = _write(roots["eval"] / "vec.fits", [fits.PrimaryHDU(np.arange(5, dtype=np.float32))])
        vec = client.get(f"/api/inspect/image/stats?fits={_rel(vec_path)}&hdu=0").get_json()
        assert vec["series"]["y"] == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_preview_of_any_plane_keeps_aspect(self, client, band_cube, poster_like):
        r = client.get(f"/inspect/preview.png?fits={_rel(band_cube)}&size=40&plane=3")
        assert r.status_code == 200 and r.data[:8] == b"\x89PNG\r\n\x1a\n"
        assert Image.open(BytesIO(r.data)).size == (40, 32)        # 20×16 → 40×32
        r = client.get(f"/inspect/preview.png?fits={_rel(poster_like)}&size=32&hdu=6")
        assert r.status_code == 200
        r = client.get(f"/inspect/preview.png?fits={_rel(poster_like)}&size=32&hdu=0")
        assert r.status_code == 415


# ---------------------------------------------------------------------------
# the ``fits`` viewer collection
# ---------------------------------------------------------------------------

class TestViewerCollection:

    def test_band_cube_is_one_colour_object(self, client, band_cube):
        meta = client.get(f"/viewer/meta/fits?path={_rel(band_cube)}").get_json()
        assert meta["count"] == 1 and meta["default_tier"] == "h0"
        assert meta["objects"][0]["ra"] == pytest.approx(150.0)
        assert meta["fits"]["stacked"] is True
        r = client.get(f"/viewer/cube/fits/0?path={_rel(band_cube)}&tier=h0")
        assert r.status_code == 200
        assert r.headers["X-Cube-Shape"] == "16,20,4"
        assert r.headers["X-Cube-Bands"] == "VIS,Y_E,J_E,H_E"
        assert r.headers["X-Cube-Unit"] == "e-"
        assert "X-Cube-Display-Scale" not in r.headers          # electrons keep the locked transfer
        wcs = json.loads(r.headers["X-Cube-WCS"])
        assert wcs["CRVAL1"] == pytest.approx(150.0)
        cube = np.frombuffer(r.data, dtype="<f4").reshape(16, 20, 4)
        assert cube[0, 0, 1] == 16 * 20                         # channel 1 = plane 1

    def test_planes_mode_steps_through_planes(self, client, band_cube):
        rel = _rel(band_cube)
        meta = client.get(f"/viewer/meta/fits?path={rel}&stack=planes").get_json()
        assert meta["count"] == 4 and [o["label"] for o in meta["objects"]] == ["VIS", "Y_E", "J_E", "H_E"]
        r = client.get(f"/viewer/cube/fits?path={rel}&stack=planes&tier=h0&id=p2")
        assert r.headers["X-Cube-Shape"] == "16,20,1" and r.headers["X-Cube-Bands"] == "J_E"
        assert r.headers["X-Cube-Index"] == "2"

    def test_band_groups_are_colour_tiers_and_bin_scales_the_wcs(self, client, poster_like):
        rel = _rel(poster_like)
        meta = client.get(f"/viewer/meta/fits?path={rel}&hdu=b:SR_").get_json()
        keys = [t["key"] for t in meta["tiers"]]
        assert keys[:8] == [f"h{k}" for k in range(1, 9)] and keys[8:] == ["b:LR_", "b:SR_"]
        assert meta["default_tier"] == "b:SR_" and meta["count"] == 1
        full = client.get(f"/viewer/cube/fits/0?path={rel}&hdu=b:SR_&tier=b:SR_")
        assert full.headers["X-Cube-Shape"] == "16,16,4"
        binned = client.get(f"/viewer/cube/fits/0?path={rel}&hdu=b:SR_&tier=b:SR_&bin=2")
        assert binned.headers["X-Cube-Shape"] == "8,8,4"
        assert float(binned.headers["X-Cube-Pixscale"]) == pytest.approx(0.1)
        w_full = json.loads(full.headers["X-Cube-WCS"])
        w_bin = json.loads(binned.headers["X-Cube-WCS"])
        assert w_bin["CD2_2"] == pytest.approx(2 * w_full["CD2_2"])
        assert "binned x2" in binned.headers["X-Cube-Label"]

    def test_a_file_of_band_hdus_opens_as_its_colour_group(self, client, poster_like):
        meta = client.get(f"/viewer/meta/fits?path={_rel(poster_like)}").get_json()
        assert meta["default_tier"] == "b:LR_" and meta["fits"]["hdu"] == "b:LR_"
        assert meta["fits"]["stacked"] is True and meta["count"] == 1

    def test_unknown_units_get_a_display_scale(self, client, roots):
        path = _write(roots["eval"] / "psf.fits", [fits.PrimaryHDU(np.full((6, 6), 1e-3, np.float32))])
        r = client.get(f"/viewer/cube/fits/0?path={_rel(path)}&tier=h0")
        assert r.headers["X-Cube-Unit"] == "arb"
        assert float(r.headers["X-Cube-Display-Scale"]) == pytest.approx(3000.0 / 1e-3, rel=1e-4)
        meta = client.get(f"/viewer/meta/fits?path={_rel(path)}&render=log").get_json()
        assert meta["render_mode"] == "log"

    def test_archive_rate_bands_display_as_electrons(self, roots):
        primary = fits.PrimaryHDU()
        hdus = [primary]
        for band in ("VIS", "Y_E", "J_E", "H_E"):
            hdu = fits.ImageHDU(np.ones((4, 4), np.float32), name=band)
            hdu.header["BUNIT"], hdu.header["MAGZERO"] = "adu/s", 24.6
            hdus.append(hdu)
        path = _write(roots["eval"] / "archive.fits", hdus)
        factor = adu_per_s_to_electrons_factor(24.6, Config.get_band("VIS"))
        data, info = viewer_data.get_cube("fits", 0, "b:", {"path": path, "hdu": "b:"})
        assert info["unit"] == "e-" and "display_scale" not in info
        assert float(data[0, 0, 0]) == pytest.approx(factor, rel=1e-5)
        assert "MAGZERO" in info["label"]
        data, info = viewer_data.get_cube("fits", 0, "h1", {"path": path})
        assert info["unit"] == "ADU/s" and float(data[0, 0, 0]) == 1.0          # native readout
        assert info["display_scale"] == pytest.approx(factor)

    def test_other_hdus_serve_their_matching_plane(self, roots):
        cube = fits.PrimaryHDU(np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4))
        flat = fits.ImageHDU(np.full((4, 4), 7.0, np.float32), name="MASK")
        path = _write(roots["eval"] / "two.fits", [cube, flat])
        meta = viewer_data.get_meta("fits", {"path": path})
        assert meta["count"] == 3 and meta["objects"][2]["tiers"] == ["h0", "h1"]
        data, info = viewer_data.get_cube("fits", 2, "h1", {"path": path})
        assert float(data[0, 0, 0]) == 7.0 and info["bands"] == ("MASK",)
        data, info = viewer_data.get_cube("fits", 2, "h0", {"path": path})
        assert float(data[0, 0, 0]) == 32.0 and info["label"] == "PRIMARY · plane 2"

    def test_errors(self, client, roots, table_file):
        assert client.get("/viewer/meta/fits").status_code == 400
        r = client.get(f"/viewer/meta/fits?path={roots['tmp'] / 'nope.fits'}")
        assert r.status_code == 404 and "error" in r.get_json()
        r = client.get(f"/viewer/meta/fits?path={_rel(table_file)}&hdu=1")
        assert r.status_code == 415 and "not a viewable image" in r.get_json()["error"]
        r = client.get(f"/viewer/meta/fits?path={_rel(table_file)}")
        assert r.status_code == 415
        r = client.get(f"/viewer/meta/fits?path={_rel(table_file)}&bin=0")
        assert r.status_code == 400
