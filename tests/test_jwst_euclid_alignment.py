from __future__ import annotations

import json

import numpy as np

from euclid_polish.config import Config
from euclid_polish.web.helpers import jwst_euclid, viewer_data


def test_field_id_is_stable_and_path_safe():
    identifier = jwst_euclid.field_id("ESA", "123/456", "jw-obs:1", 30.0)
    assert identifier == "ESA-123-456-jw-obs-1-s30"
    assert "/" not in identifier
    assert " " not in identifier


def test_nexus_pair_id_and_public_product_options_are_stable():
    identifier = jwst_euclid.nexus_pair_id(268.4625, 65.19917, "f444w", 30.0)
    assert identifier == "nexus-qdr-ep05-F444W-location-ra268.4625000-dec65.1991700-s30"
    assert "/" not in identifier
    options = {item["filter"]: item for item in jwst_euclid.nexus_product_options()}
    assert options["F200W"]["pixel_scale_mas"] == 30
    assert options["F444W"]["pixel_scale_mas"] == 60


def test_nexus_source_tiles_use_exact_255_pixel_euclid_footprints(monkeypatch):
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [268.4625, 65.19917]
    wcs.wcs.cdelt = [-0.03 / 3600.0, 0.03 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    data = np.ones((1700, 1700), dtype=np.float32)
    monkeypatch.setattr(
        jwst_euclid,
        "_find_image",
        lambda _path: (data, wcs.to_header(), wcs, "PRIMARY"),
    )
    _data, _header, _wcs, tiles = jwst_euclid._nexus_source_tiles(
        __import__("pathlib").Path("nexus.fits"), filter_name="F200W",
    )
    assert tiles == [
        (0, 0, 850, 850), (850, 0, 1700, 850),
        (0, 850, 850, 1700), (850, 850, 1700, 1700),
    ]


def test_nexus_field_viewer_reads_saved_255_pixel_tiles(tmp_path, monkeypatch):
    from astropy.io import fits

    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path / "data"))
    identifier = jwst_euclid.nexus_field_id("F200W")
    root = jwst_euclid.nexus_field_root() / identifier / "tiles"
    root.mkdir(parents=True)
    fits.PrimaryHDU(np.ones((255, 255), dtype=np.float32)).writeto(root / "euclid.fits")
    fits.PrimaryHDU(np.ones((850, 850), dtype=np.float32)).writeto(root / "jwst.fits")
    manifest = {
        "field_id": identifier, "filter": "F200W", "count": 1,
        "tiles": [{"ra_deg": 268.4625, "dec_deg": 65.19917,
                   "euclid_file": "tiles/euclid.fits", "jwst_file": "tiles/jwst.fits",
                   "jwst_metadata": {"pixel_scale_arcsec": [0.03, 0.03]}}],
    }
    (root.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    meta = viewer_data._nexus_field_meta({"field": identifier})
    euclid, _euclid_meta = viewer_data._nexus_field_cube(0, "lr", {"field": identifier})
    jwst, jwst_meta = viewer_data._nexus_field_cube(0, "jwst", {"field": identifier})

    assert meta["count"] == 1
    assert meta["tiers"][0]["label"] == "LR · Euclid · 255 px"
    assert meta["objects"][0]["tiers"] == ["lr", "jwst"]
    assert euclid.shape == (255, 255, 1)
    assert jwst.shape == (850, 850, 1)
    assert jwst_meta["pixscale"] == 0.03


def test_nexus_download_reuses_cached_tiles_while_filling_missing_bands(tmp_path, monkeypatch):
    from astropy.io import fits
    from astropy.wcs import WCS

    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path / "data"))
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [268.4625, 65.19917]
    wcs.wcs.cdelt = [-0.1 / 3600.0, 0.1 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    source = np.ones((850, 850), dtype=np.float32)
    calls: list[str] = []

    monkeypatch.setattr(jwst_euclid, "_download_nexus_mosaic", lambda *_args, **_kwargs: tmp_path / "nexus.fits")
    monkeypatch.setattr(
        jwst_euclid, "_nexus_source_tiles",
        lambda *_args, **_kwargs: (source, header, wcs, [(0, 0, 850, 850)]),
    )

    def write_jwst(_data, _header, _wcs, _bounds, destination, **_kwargs):
        fits.PrimaryHDU(source, header=header).writeto(destination, overwrite=True)
        return source, header, wcs, 268.4625, 65.19917

    def fetch_cutout_at(*, band_name, output_file, cutout_size_vis_pixels, **_kwargs):
        calls.append(band_name)
        data = np.ones((cutout_size_vis_pixels, cutout_size_vis_pixels), dtype=np.float32)
        cutout_header = header.copy()
        if band_name != "VIS":
            cutout_header["CRPIX1"] = 5.0
            cutout_header["CRPIX2"] = 5.0
        fits.PrimaryHDU(data, header=cutout_header).writeto(output_file, overwrite=True)
        return True, None

    monkeypatch.setattr(jwst_euclid, "_write_nexus_source_tile", write_jwst)
    monkeypatch.setattr("euclid_polish.catalog.downloader.fetch_cutout_at", fetch_cutout_at)

    first = jwst_euclid.download_nexus_field(filter_name="F200W")
    second = jwst_euclid.download_nexus_field(filter_name="F200W")

    assert calls == ["VIS", "Y_E", "J_E", "H_E"]
    assert "four_band_error" not in first["tiles"][0], first["tiles"][0].get("four_band_error")
    assert first["tiles"][0]["lr_file"] == second["tiles"][0]["lr_file"]
    assert set(first["tiles"][0]["euclid_files"]) == set(Config.LR_INPUT_BAND_NAMES)


def test_nexus_field_viewer_exposes_registered_lr_and_sr(tmp_path, monkeypatch):
    from astropy.io import fits

    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path / "data"))
    identifier = jwst_euclid.nexus_field_id("F444W")
    root = jwst_euclid.nexus_field_root() / identifier / "tiles"
    root.mkdir(parents=True)
    fits.PrimaryHDU(np.ones((425, 425), dtype=np.float32)).writeto(root / "jwst.fits")
    fits.PrimaryHDU(np.ones((4, 255, 255), dtype=np.float32)).writeto(root / "lr.fits")
    fits.PrimaryHDU(np.ones((4, 1020, 1020), dtype=np.float32)).writeto(root / "sr.fits")
    manifest = {
        "field_id": identifier, "filter": "F444W", "count": 2,
        "tiles": [{"ra_deg": 268.4625, "dec_deg": 65.19917,
                   "euclid_file": "tiles/lr.fits", "lr_file": "tiles/lr.fits",
                   "jwst_file": "tiles/jwst.fits",
                   "inference": {"combiner_label": "STARFULL", "pixel_scale_arcsec": 0.025,
                                 "files": {"starfull": "tiles/sr.fits"}}},
                  {"ra_deg": 268.4626, "dec_deg": 65.19918,
                   "euclid_file": "tiles/lr.fits", "lr_file": "tiles/lr.fits",
                   "jwst_file": "tiles/jwst.fits"}],
    }
    (root.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    meta = viewer_data._nexus_field_meta({"field": identifier})
    lr, lr_meta = viewer_data._nexus_field_cube(0, "lr", {"field": identifier})
    sr, sr_meta = viewer_data._nexus_field_cube(0, "sr", {"field": identifier})

    assert [tier["key"] for tier in meta["tiers"]] == ["lr", "sr", "jwst"]
    assert meta["objects"][0]["tiers"] == ["lr", "sr", "jwst"]
    assert meta["objects"][1]["tiers"] == ["lr", "sr", "jwst"]
    assert meta["transfer_groups"] == ["euclid", "jwst"]
    assert lr.shape == (255, 255, 4)
    assert lr_meta["bands"] == list(Config.LR_INPUT_BAND_NAMES)
    assert "display_scale" not in lr_meta
    assert lr_meta["transfer_group"] == "euclid"
    assert sr.shape == (1020, 1020, 4)
    assert sr_meta["pixscale"] == 0.025
    assert "display_scale" not in sr_meta
    assert sr_meta["transfer_group"] == "euclid"


def test_nexus_starfull_inference_reuses_current_and_replaces_stale_sr(
        tmp_path, monkeypatch):
    from types import SimpleNamespace

    from astropy.io import fits

    import euclid_polish.ensemble as ensemble_module
    import euclid_polish.eval.combiner as combiner_module

    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path / "data"))
    identifier = jwst_euclid.nexus_field_id("F200W")
    root = jwst_euclid.nexus_field_root() / identifier / "tiles"
    root.mkdir(parents=True)
    files = {}
    for band_name in Config.LR_INPUT_BAND_NAMES:
        filename = f"{band_name}.fits"
        fits.PrimaryHDU(np.ones((255, 255), dtype=np.float32)).writeto(root / filename)
        files[band_name] = f"tiles/{filename}"
    fits.PrimaryHDU(np.ones((4, 255, 255), dtype=np.float32)).writeto(root / "lr.fits")
    manifest = {
        "field_id": identifier, "filter": "F200W", "tiles": [{
            "index": 0, "source_index": 0, "euclid_file": files["VIS"],
            "euclid_files": files, "lr_file": "tiles/lr.fits",
        }],
    }
    (root.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    class FakeEnsemble:
        member_labels = ["member"]

        def __init__(self, *_args, **_kwargs):
            pass

        def member_arrays(self, cube):
            return cube

    class FakeCombiner:
        def apply_field(self, members):
            applied.append((fingerprint["value"], member_fingerprint["value"]))
            multiplier = 2 if fingerprint["value"] == "fp-one" else 3
            return members * multiplier

    fingerprint = {"value": "fp-one"}
    member_fingerprint = {"value": "member-one"}
    applied: list[tuple[str, str]] = []
    monkeypatch.setattr(ensemble_module, "EnsembleModel", FakeEnsemble)
    monkeypatch.setattr(ensemble_module, "default_ensemble_dir", lambda: "unused")
    monkeypatch.setattr(
        jwst_euclid, "_starfull_member_fingerprints",
        lambda *_args: [member_fingerprint["value"]],
    )
    monkeypatch.setattr(combiner_module, "ACTIVE_COMBINER_KINDS", ("fake",))
    monkeypatch.setattr(
        combiner_module, "COMBINER_MODELS",
        {"fake": SimpleNamespace(artifact_dir="fake", label="Fake STARFULL")},
    )
    monkeypatch.setattr(combiner_module, "load_combiner", lambda *_args, **_kwargs: FakeCombiner())
    monkeypatch.setattr(
        combiner_module, "combiner_artifact_fingerprint",
        lambda *_args, **_kwargs: fingerprint["value"],
    )

    result = jwst_euclid.run_starfull_nexus_field_inference(identifier)
    source = result["tiles"][0]["inference"]["files"]["starfull"]
    assert (root.parent / source).is_file()
    assert result["tiles"][0]["inference"]["combiner_fingerprint"] == "fp-one"
    assert applied == [("fp-one", "member-one")]

    # The same exact artifact reuses the completed SR.
    assert jwst_euclid.run_starfull_nexus_field_inference(identifier)["field_id"] == identifier
    assert applied == [("fp-one", "member-one")]

    # Retraining a member under the same label invalidates the cached SR even
    # while the fitted combiner artifact itself is unchanged.
    member_fingerprint["value"] = "member-two"
    member_refreshed = jwst_euclid.run_starfull_nexus_field_inference(identifier)
    assert applied == [("fp-one", "member-one"), ("fp-one", "member-two")]
    assert member_refreshed["tiles"][0]["inference"][
        "member_fingerprints"
    ] == ["member-two"]

    # A refit under the same combiner kind changes the artifact hash. The old
    # FITS stays in place until its replacement is complete, then the manifest
    # and image move to the new identity together.
    fingerprint["value"] = "fp-two"
    refreshed = jwst_euclid.run_starfull_nexus_field_inference(identifier)
    assert applied == [
        ("fp-one", "member-one"), ("fp-one", "member-two"),
        ("fp-two", "member-two"),
    ]
    assert refreshed["tiles"][0]["inference"]["combiner_fingerprint"] == "fp-two"
    with fits.open(root.parent / source) as hdul:
        assert np.all(np.asarray(hdul[0].data) == 3)


def test_nexus_field_status_marks_changed_combiner_sr_stale(
        tmp_path, monkeypatch):
    from astropy.io import fits

    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path / "data"))
    identifier = jwst_euclid.nexus_field_id("F200W")
    directory = jwst_euclid.nexus_field_root() / identifier
    root = directory / "tiles"
    root.mkdir(parents=True)
    files = {}
    for band_name in Config.LR_INPUT_BAND_NAMES:
        filename = f"{band_name}.fits"
        fits.PrimaryHDU(np.ones((255, 255), dtype=np.float32)).writeto(
            root / filename,
        )
        files[band_name] = f"tiles/{filename}"
    fits.PrimaryHDU(np.ones((850, 850), dtype=np.float32)).writeto(
        root / "jwst.fits",
    )
    fits.PrimaryHDU(np.ones((4, 255, 255), dtype=np.float32)).writeto(
        root / "lr.fits",
    )
    fits.PrimaryHDU(np.ones((4, 1020, 1020), dtype=np.float32)).writeto(
        root / "sr.fits",
    )
    manifest = {
        "field_id": identifier, "filter": "F200W", "count": 1,
        "tiles": [{
            "euclid_file": files["VIS"], "euclid_files": files,
            "lr_file": "tiles/lr.fits", "jwst_file": "tiles/jwst.fits",
            "inference": {
                "combiner_kind": "fake",
                "combiner_fingerprint": "old-fingerprint",
                "files": {"starfull": "tiles/sr.fits"},
            },
        }],
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    monkeypatch.setattr(
        jwst_euclid, "_active_starfull_combiner_artifact",
        lambda: {
            "combiner_kind": "fake",
            "combiner_fingerprint": "new-fingerprint",
        },
    )

    (field,) = jwst_euclid.nexus_fields()
    assert (field["sr_count"], field["current_sr_count"],
            field["stale_sr_count"]) == (1, 0, 1)

    manifest["tiles"][0]["inference"][
        "combiner_fingerprint"
    ] = "new-fingerprint"
    (directory / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    (field,) = jwst_euclid.nexus_fields()
    assert (field["sr_count"], field["current_sr_count"],
            field["stale_sr_count"]) == (1, 1, 0)


def test_euclid_product_path_joins_directory_and_filename():
    assert jwst_euclid.euclid_product_path({
        "file_path": "/archive/VIS",
        "file_name": "tile.fits",
    }) == "/archive/VIS/tile.fits"


def test_field_coordinates_prefer_jwst_footprint_center():
    assert jwst_euclid.field_coordinates({
        "jwst_ra_deg": "58.91",
        "jwst_dec_deg": "-46.29",
        "euclid_ra_deg": "59.02",
        "euclid_dec_deg": "-46.50",
    }) == (58.91, -46.29)


def test_aligned_header_removes_invalid_extension_cards(tmp_path):
    from astropy.io import fits

    header = fits.Header()
    header["EXTNAME"] = 1
    header["EXTVER"] = 1
    safe = jwst_euclid._aligned_primary_header(header, "jwst_i2d.fits")
    assert "EXTNAME" not in safe
    assert safe["ALIGN"] == "JWST-EUCLID"
    fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32), header=safe).writeto(
        tmp_path / "aligned.fits",
    )
    assert jwst_euclid.euclid_product_path({
        "file_path": "/archive/VIS/tile.fits",
        "file_name": "tile.fits",
    }) == "/archive/VIS/tile.fits"


def test_jwst_product_selection_prefers_resampled_image():
    rows = [
        {"filename": "jw0001_rate.fits"},
        {"filename": "jw0001_i2d.fits"},
        {"filename": "jw0001_uncal.fits"},
    ]
    assert jwst_euclid._choose_jwst_product(rows) == "jw0001_i2d.fits"


def test_jwst_colour_groups_use_every_filter_in_wavelength_order():
    entries = [
        {"filter": "F356W"},
        {"filter": "F115W"},
        {"filter": "F200W"},
        {"filter": "F444W"},
        {"filter": "F150W"},
    ]
    blue, green, red = viewer_data._jwst_colour_channel_groups(entries)
    assert blue == [1, 4]
    assert green == [2, 0]
    assert red == [3]
    assert sorted(blue + green + red) == list(range(len(entries)))


def test_jwst_temperature_uses_filter_name_as_display_only_pivot():
    name, band = viewer_data._jwst_approx_color_band({"filter": "F150W2"}) or (None, None)
    assert name == "F150W2"
    assert band["pivot_um"] == 1.5
    assert band["display_only"] is True
    assert viewer_data._jwst_approx_color_band({"filter": "CLEAR"}) is None


def test_jwst_colour_and_temperature_preserve_native_band_brightness(monkeypatch):
    entries = [{"filter": "F100W"}, {"filter": "F200W"}, {"filter": "F300M"}]
    planes = [
        np.full((2, 2), 2.0, dtype=np.float32),
        np.full((2, 2), 10.0, dtype=np.float32),
        np.full((2, 2), 50.0, dtype=np.float32),
    ]
    monkeypatch.setattr(
        viewer_data,
        "_jwst_aligned_planes",
        lambda _manifest, _directory: (entries, planes, "F100W", 0.03),
    )

    colour, colour_meta = viewer_data._jwst_colour_cube({}, "")
    temperature, temperature_meta = viewer_data._jwst_temperature_cube({}, "")

    np.testing.assert_array_equal(colour[0, 0], [50.0, 10.0, 2.0])
    np.testing.assert_array_equal(temperature[0, 0], [2.0, 10.0, 50.0])
    assert colour_meta["display_scale"] == 60.0
    assert temperature_meta["display_scale"] == 60.0


def test_readable_fits_rejects_empty_archive_placeholder(tmp_path):
    from astropy.io import fits

    empty = tmp_path / "empty.fits"
    empty.touch()
    valid = tmp_path / "valid.fits"
    fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32)).writeto(valid)
    assert not jwst_euclid._is_readable_fits(empty)
    assert jwst_euclid._is_readable_fits(valid)


def test_euclid_download_recovers_valid_file_from_placeholder(tmp_path):
    from astropy.io import fits

    extracted = tmp_path / "extracted.fits"
    fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32)).writeto(extracted)
    destination = tmp_path / "requested.fits"

    class FakeEuclid:
        @staticmethod
        def get_cutout(**kwargs):
            destination.touch()
            return [str(extracted)]

    jwst_euclid._download_euclid_cutout(
        FakeEuclid,
        file_path="archive/file",
        tile_index="T123",
        coordinate=None,
        radius=None,
        destination=destination,
    )
    assert jwst_euclid._is_readable_fits(destination)


def test_align_to_target_preserves_an_identity_grid():
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [2.0, 2.0]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.cdelt = [-0.001, 0.001]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    aligned = jwst_euclid.align_to_target(data, wcs, wcs, data.shape)
    np.testing.assert_allclose(aligned[1:-1, 1:-1], data[1:-1, 1:-1], atol=1e-4)


def test_align_to_target_corrects_a_one_pixel_wcs_offset():
    from astropy.wcs import WCS

    target_wcs = WCS(naxis=2)
    target_wcs.wcs.crpix = [3.0, 3.0]
    target_wcs.wcs.crval = [10.0, 20.0]
    target_wcs.wcs.cdelt = [-0.001, 0.001]
    target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    source_wcs = target_wcs.deepcopy()
    source_wcs.wcs.crpix[0] += 1.0
    source_data = np.tile(np.arange(7, dtype=np.float32), (7, 1))

    registered = jwst_euclid.align_to_target(
        source_data, source_wcs, target_wcs, (5, 5),
    )

    # The target centre maps to source x=3 (rather than its own x=2).
    assert registered[2, 2] == 3.0


def test_native_sky_cutout_keeps_source_pixel_scale():
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [51.0, 51.0]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.cdelt = [-0.00001, 0.00001]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    source_scale = proj_plane_pixel_scales(wcs)[:2]
    cutout, cutout_wcs = jwst_euclid._native_sky_cutout(
        np.ones((101, 101), dtype=np.float32),
        wcs,
        SkyCoord(ra=10.0, dec=20.0, unit="deg"),
        1.0,
    )

    assert cutout.shape == (28, 28)
    np.testing.assert_allclose(proj_plane_pixel_scales(cutout_wcs)[:2], source_scale)


def test_pixel_metadata_reports_shape_scale_and_detector():
    from astropy.io import fits
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [2.0, 2.0]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.cdelt = [-0.001, 0.001]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = fits.Header(wcs.to_header())
    header["DETECTOR"] = "NRCA3"
    header["EXPTIME"] = 12.5
    metadata = jwst_euclid._pixel_metadata(np.ones((4, 6), dtype=np.float32), wcs, header)

    assert metadata["shape"] == [4, 6]
    np.testing.assert_allclose(metadata["pixel_scale_arcsec"], [3.6, 3.6])
    assert metadata["detector"] == "NRCA3"
    assert metadata["exposure_s"] == 12.5


def test_overlap_rows_reads_cached_csv_and_marks_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path / "data"))
    root = jwst_euclid.overlap_root()
    root.mkdir(parents=True)
    (root / "esa_partial.csv").write_text(
        "euclid_tile_index,euclid_ra_deg,euclid_dec_deg,jwst_archive,jwst_observation_id,jwst_target_name\n"
        "T123,10,20,esa,jwobs,Target\n",
        encoding="utf-8",
    )
    rows, status = jwst_euclid.overlap_rows()
    assert status["partial"] is False
    assert rows[0]["field_id"] == jwst_euclid.field_id("esa", "T123", "jwobs", 30.0)
    assert rows[0]["available"] is False

    pair = jwst_euclid.pair_root() / rows[0]["field_id"]
    pair.mkdir(parents=True)
    (pair / "manifest.json").write_text(json.dumps({"field_id": rows[0]["field_id"]}), encoding="utf-8")
    assert jwst_euclid.overlap_rows()[0][0]["available"] is False


def test_location_groups_collect_filters_at_one_sky_position(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path / "data"))
    root = jwst_euclid.overlap_root()
    root.mkdir(parents=True)
    (root / "esa_partial.csv").write_text(
        "euclid_tile_index,euclid_ra_deg,euclid_dec_deg,jwst_archive,jwst_observation_id,jwst_target_name,jwst_ra_deg,jwst_dec_deg,jwst_instrument\n"
        "T123,10,20,esa,jw0001-o001_nircam_clear-f150w,Target,10.0,20.0,NIRCAM/IMAGE\n"
        "T123,10,20,esa,jw0001-o002_nircam_clear-f150w,Target,10.0,20.0,NIRCAM/IMAGE\n"
        "T123,10,20,esa,jw0001-o003_nircam_clear-f322w2,Target,10.00001,20.0,NIRCAM/IMAGE\n"
        "T124,11,21,esa,jw0002-o001_nircam_clear-f200w,Elsewhere,11.0,21.0,NIRCAM/IMAGE\n",
        encoding="utf-8",
    )

    groups, status = jwst_euclid.location_groups()

    target = next(group for group in groups if group["jwst_target_name"] == "Target")
    assert status["count"] == 2
    assert status["product_count"] == 4
    assert target["jwst_row_count"] == 3
    assert target["jwst_product_count"] == 2
    assert target["jwst_filters"] == "F150W, F322W2"


def test_download_remaining_locations_skips_saved_and_keeps_going(monkeypatch):
    rows = [
        {"field_id": "saved", "available": True},
        {"field_id": "blank", "available": False, "euclid_coverage_status": "not_covered"},
        {"field_id": "good", "available": False, "jwst_target_name": "Good field"},
        {"field_id": "bad", "available": False, "jwst_target_name": "Bad field"},
    ]
    calls = []
    ticks = []

    monkeypatch.setattr(jwst_euclid, "location_groups", lambda: (rows, {}))

    def fake_download(row, *, size_arcsec, progress=None):
        calls.append((row["field_id"], size_arcsec))
        if row["field_id"] == "bad":
            raise RuntimeError("archive unavailable")
        return {"field_id": row["field_id"]}

    monkeypatch.setattr(jwst_euclid, "download_and_align_pair", fake_download)
    result = jwst_euclid.download_remaining_locations(
        size_arcsec=30.0,
        progress=lambda current, total, label: ticks.append((current, total, label)),
    )

    assert calls == [("good", 30.0), ("bad", 30.0)]
    assert result["already_saved_count"] == 1
    assert result["known_no_coverage_count"] == 1
    assert result["downloaded_count"] == 1
    assert result["failed_count"] == 1
    assert ticks[-1][:2] == (2, 2)


def test_scan_euclid_coverage_caches_unique_field_centers(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path / "data"))
    root = jwst_euclid.overlap_root()
    root.mkdir(parents=True)
    (root / "esa_partial.csv").write_text(
        "euclid_tile_index,euclid_ra_deg,euclid_dec_deg,jwst_archive,jwst_observation_id,jwst_target_name,jwst_ra_deg,jwst_dec_deg\n"
        "T123,10,20,esa,jwobs-1,Covered,10.0,20.0\n"
        "T123,10,20,esa,jwobs-2,Same center,10.0,20.0\n"
        "T124,11,21,esa,jwobs-3,Uncovered,11.0,21.0\n",
        encoding="utf-8",
    )

    calls = []
    probes = []

    def fake_coverage(ra, dec, *, strict=False):
        calls.append((ra, dec, strict))
        return [{"tile_index": "VIS-T123", "file_name": "vis.fits", "file_path": "/archive"}] if ra == 10.0 else []

    monkeypatch.setattr(jwst_euclid, "euclid_tiles_covering", fake_coverage)
    def fake_probe(client, tiles, **kwargs):
        probes.append(list(tiles))
        return (tiles[0], 0, []) if tiles else (None, 1, [])

    monkeypatch.setattr(jwst_euclid, "_probe_euclid_tiles", fake_probe)
    summary = jwst_euclid.scan_euclid_coverage()

    assert summary["unique_count"] == 2
    assert summary["covered_count"] == 1
    assert summary["not_covered_count"] == 1
    assert len(calls) == 2
    assert len(probes) == 1
    rows, _ = jwst_euclid.overlap_rows()
    assert [row["euclid_coverage_status"] for row in rows] == ["covered", "covered", "not_covered"]

    jwst_euclid.scan_euclid_coverage()
    assert len(calls) == 2


def test_euclid_probe_skips_blank_product_for_alternate(tmp_path, monkeypatch):
    tiles = [
        {"tile_index": "VIS-BLANK", "file_path": "/archive", "file_name": "blank.fits"},
        {"tile_index": "VIS-GOOD", "file_path": "/archive", "file_name": "good.fits"},
    ]
    downloaded = []

    def fake_download(client, *, file_path, tile_index, coordinate, radius, destination):
        downloaded.append(tile_index)
        destination.touch()

    def fake_find_image(path):
        data = np.zeros((2, 2), dtype=np.float32) if downloaded[-1] == "VIS-BLANK" else np.ones((2, 2), dtype=np.float32)
        return data, None, None, "PRIMARY"

    monkeypatch.setattr(jwst_euclid, "_download_euclid_cutout", fake_download)
    monkeypatch.setattr(jwst_euclid, "_find_image", fake_find_image)
    selected, blank_count, errors = jwst_euclid._probe_euclid_tiles(
        object(), tiles, coordinate=None, radius=None, destination_dir=tmp_path,
    )

    assert selected["tile_index"] == "VIS-GOOD"
    assert blank_count == 1
    assert errors == []
    assert downloaded == ["VIS-BLANK", "VIS-GOOD"]
