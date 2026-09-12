import io
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.noise_assessment import BANDS
from euclid_polish.noise_assessment.archive import (
    Archive,
    acquire_mer,
    digest,
    freeze_example_references,
    read_json,
    save_json,
)
from euclid_polish.noise_assessment.exposures import native_conversion
from euclid_polish.noise_assessment.measurement import crop_to_patch, measure_mer, select_validation


def test_mer_resume_preserves_completed_exposure_results_and_provenance(tmp_path):
    save_json(tmp_path / "manifest.json", {"patches": []})
    previous = {
        "exposure_measurements": [{"status": "measured", "arrays": "kept.npz"}],
        "mer_predictions": [{"status": "conditional prediction", "draws": 512}],
        "exposure_measured_at": "2026-09-12T12:00:00+00:00",
        "exposure_processing_parameters": {"output_size": 80, "code_sha256": {"exposures.py": "saved"}},
    }
    save_json(tmp_path / "summary.json", previous)
    result = measure_mer(tmp_path)
    assert all(result[key] == value for key, value in previous.items())


def test_cache_reuses_bytes_and_detects_corruption(tmp_path, monkeypatch):
    archive = Archive(tmp_path)
    requests = []

    def get(url, **kwargs):
        requests.append(url)
        return b"verified scientific bytes", {"ETag": '"multipart-3"'}, 200

    monkeypatch.setattr(archive, "get", get)
    path, record = archive.cached("https://irsa.ipac.caltech.edu/test")
    assert record["sha256"] == digest(path)
    assert "unavailable" in record["upstream_checksum"]
    assert archive.cached("https://irsa.ipac.caltech.edu/test")[0] == path
    assert len(requests) == 1
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        archive.cached("https://irsa.ipac.caltech.edu/test")


def test_frozen_examples_survive_removal_of_original_workspace_paths(tmp_path):
    original = tmp_path / "source.fits"
    pixels = np.ones((256, 256), dtype=np.float32)
    hdu = fits.ImageHDU(pixels, name="VIS")
    hdu.header.update(
        {
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": 10.0,
            "CRVAL2": 20.0,
            "CRPIX1": 128.0,
            "CRPIX2": 128.0,
            "CDELT1": -0.1 / 3600,
            "CDELT2": 0.1 / 3600,
        }
    )
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(original)
    old_array = tmp_path / "example.npz"
    np.savez_compressed(old_array, original_e=pixels)
    root = tmp_path / "assessment"
    root.mkdir()
    manifest = {
        "patches": [
            {
                "patch_id": "example",
                "size": 256,
                "example_array": str(old_array),
                "legacy_example": {
                    "source_file": str(original),
                    "source_sha256": digest(original),
                    "patch_grid_column_zero_based": 0,
                    "patch_grid_row_zero_based": 0,
                },
            }
        ]
    }
    freeze_example_references(root, manifest)
    patch = manifest["patches"][0]
    original.unlink()
    old_array.unlink()
    assert digest(root / patch["example_array"]) == patch["example_array_sha256"]
    cropped, _ = crop_to_patch(patch, [pixels], hdu.header)
    np.testing.assert_array_equal(cropped[0], pixels)


def test_native_exposure_time_units_and_nir_integrated_zeropoint():
    primary = {"EXPTIME": 100}
    a = native_conversion(primary, {"BUNIT": "ADU", "MAGZEROP": 24.6}, "VIS")
    b = native_conversion({"EXPTIME": 200}, {"BUNIT": "ADU", "MAGZEROP": 24.6}, "VIS")
    assert a["science_factor"] == 2 * b["science_factor"]
    nir = native_conversion(primary, {"BUNIT": "ELECTRON", "ZPAB": 29.8}, "Y")
    nir2 = native_conversion({"EXPTIME": 200}, {"BUNIT": "ELECTRON", "ZPAB": 29.8}, "Y")
    assert nir["science_factor"] == nir2["science_factor"]
    for band, header in (
        ("VIS", {"BUNIT": "ELECTRON", "MAGZEROP": 24.6}),
        ("Y", {"BUNIT": "ADU", "ZPAB": 29.8}),
    ):
        with pytest.raises(ValueError):
            native_conversion(primary, header, band)


def test_nine_distinct_strata_and_separate_stress_case(tmp_path):
    rows, patches = [], []
    for field, offset in zip(("EDF-N", "EDF-S", "EDF-F"), (0, 10, 20), strict=True):
        for i in range(5):
            rows.append(
                {
                    "band": "VIS",
                    "kind": "central",
                    "status": "measured",
                    "sample_id": offset + i,
                    "field": field,
                    "official_rms_median": 10 + i,
                    "patch_id": f"p{offset + i}",
                }
            )
    patches.append({"kind": "extreme", "patch_id": "star31", "sample_id": 31, "field": "EDF-F"})
    save_json(tmp_path / "manifest.json", {"patches": patches})
    save_json(tmp_path / "summary.json", {"mer_measurements": rows})
    selected = select_validation(tmp_path)
    assert len({r["sample_id"] for r in selected}) == 10
    for field in ("EDF-N", "EDF-S", "EDF-F"):
        assert [r.get("selection_vis_rms") for r in selected if r["field"] == field][:3] == [10, 12, 14]
    assert selected[-1]["stratum"] == "bright-star stress case"


@pytest.mark.parametrize("problem", [None, "misaligned", "wrong_release", "missing_rms", "wrong_parent"])
def test_pilot_requires_matched_products_without_image_noise_fallback(tmp_path, monkeypatch, problem):
    parents, metadata = {}, []
    for band in BANDS:
        instrument = "VIS" if band == "VIS" else "NISP"
        parents[band] = {
            "tile_index": "123",
            "instrument_name": instrument,
            "file_name": band + "_science.fits",
        }
        for role in ("science", "rms", "flags"):
            if problem == "missing_rms" and band == "VIS" and role == "rms":
                continue
            filename = (
                band + "_science.fits" if role == "science" else band + "-" + role.upper() + "_test.fits"
            )
            if role == "flags":
                filename = band + "-FLAG_test.fits"
            if problem == "wrong_parent" and band == "VIS" and role == "science":
                filename = "wrong_science.fits"
            metadata.append(
                {
                    "energy_bandpassname": band,
                    "obs_id": "123_" + instrument,
                    "obs_publisher_did": "parent_" + band,
                    "access_url": "https://irsa.ipac.caltech.edu/" + filename,
                    "dataproduct_subtype": {"science": "science", "rms": "noise", "flags": "auxiliary"}[role],
                    "s_xel1": 32,
                    "s_xel2": 32,
                }
            )
    save_json(
        tmp_path / "manifest.json",
        {
            "pilot_verified": False,
            "patches": [{"patch_id": "pilot", "ra": 10.0, "dec": 20.0, "parents": parents, "size": 32}],
        },
    )
    monkeypatch.setattr(Archive, "sia", lambda *_: (metadata, {"query": "mock metadata"}))
    monkeypatch.setattr(Archive, "ancestry", lambda *_: ([{"observationid": "native1"}], {}))

    def get(self, url, **kwargs):
        hdu = fits.PrimaryHDU(np.ones((32, 32), np.float32))
        header = hdu.header
        header.update(
            {
                "DATASETR": "Q1_R1",
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CRVAL1": 10.0,
                "CRVAL2": 20.0,
                "CRPIX1": 16.0,
                "CRPIX2": 16.0,
                "CDELT1": -0.1 / 3600,
                "CDELT2": 0.1 / 3600,
                "PPOID": "same-parent",
            }
        )
        if problem == "misaligned" and "VIS-RMS_" in url:
            header["CRPIX1"] = 17.0
        if problem == "wrong_release" and "VIS-RMS_" in url:
            header["DATASETR"] = "Q2_R1"
        stream = io.BytesIO()
        hdu.writeto(stream)
        return stream.getvalue(), {}, 200

    monkeypatch.setattr(Archive, "get", get)
    manifest = acquire_mer(tmp_path, pilot=True)
    assert manifest["pilot_verified"] == (problem is None)
    vis = manifest["patches"][0]["mer"]["VIS"]
    if problem:
        assert vis["status"].startswith("unavailable")
        assert "image_estimated_rms" not in vis
    else:
        assert all(r["status"] == "verified" for r in manifest["patches"][0]["mer"].values())
        assert read_json(tmp_path / "manifest.json")["pilot_verified"]
