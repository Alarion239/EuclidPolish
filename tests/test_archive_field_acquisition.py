"""Focused offline tests for matched multipoint archive acquisition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from euclid_polish.web.helpers import archive_fields
from scripts import fasrc_download_euclid_sky_cutouts as sampling


def _parent(band: str, source_id: int) -> dict:
    return {
        "parent_id": f"{band.lower()}-{source_id:02d}",
        "mosaic_product_oid": f"{band}-oid-{source_id:02d}",
        "release_name": "Q1_R1",
        "product_type": "DpdMerBksMosaic",
        "tile_index": str(1000 + source_id),
        "file_path": f"/archive/{band}/{source_id}.fits",
        "file_name": f"{source_id}.fits",
        "instrument_name": "VIS" if band == "VIS" else "NISP",
        "filter_name": band,
        "technique": "IMAGE",
        "fov": "POLYGON ICRS 9 19 11 19 11 21 9 21",
        "coverage_clearance_deg": 0.2,
    }


def _source_manifest() -> dict:
    plan = {
        "source_release": "Q1_R1",
        "cutout_size_vis_pixels": 2560,
        "seed": 42,
    }
    samples = []
    fields = ("EDF-N", "EDF-F", "EDF-S")
    for source_id in range(44):
        samples.append({
            "sample_id": source_id,
            "anchor_id": f"{fields[source_id % 3]}-{source_id:03d}",
            "slot": 0,
            "field": fields[source_id % 3],
            "parent_id": f"vis-{source_id:02d}",
            "parent": _parent("VIS", source_id),
            "ra": 10.0 + source_id * 0.1,
            "dec": 20.0,
            "status": "written",
            "output_path": f"/remote/cutouts/sky_{source_id:04d}.fits",
        })
    return {
        "version": 1,
        "kind": "euclid_vis_noise_sampling",
        "source_release": "Q1_R1",
        "plan": plan,
        "plan_fingerprint": sampling._plan_fingerprint(plan),
        "samples": samples,
    }


def _header(*, shift_x: float = 0.0, shift_y: float = 0.0) -> fits.Header:
    header = fits.Header()
    header["RADESYS"] = "ICRS"
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = 20.0
    header["CRPIX1"] = 301.0 + shift_x
    header["CRPIX2"] = 301.0 + shift_y
    header["CD1_1"] = -1.0 / 36000.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 1.0 / 36000.0
    header["MAGZERO"] = 25.0
    header["BUNIT"] = "adu/s"
    return header


def _raw_bands() -> dict[str, tuple[np.ndarray, fits.Header]]:
    return {
        "VIS": (np.full((600, 600), 1.0, dtype=np.float32), _header()),
        "Y_E": (
            np.full((630, 630), 2.0, dtype=np.float32),
            _header(shift_x=10.0, shift_y=12.0),
        ),
        "J_E": (
            np.full((640, 640), 3.0, dtype=np.float32),
            _header(shift_x=17.0, shift_y=9.0),
        ),
        "H_E": (
            np.full((620, 620), 4.0, dtype=np.float32),
            _header(shift_x=8.0, shift_y=8.0),
        ),
    }


def test_plan_is_44_by_5_deterministic_and_does_not_change_source(tmp_path):
    source_path = tmp_path / "vis_noise_sampling_manifest.json"
    source_path.write_text(json.dumps(_source_manifest()), encoding="utf-8")
    before = source_path.read_bytes()

    source, source_sha = sampling._archive_source_manifest(source_path)
    first = sampling._new_archive_fields_manifest(source, source_sha)
    second = sampling._new_archive_fields_manifest(source, source_sha)

    assert source_path.read_bytes() == before
    assert len(first["samples"]) == 220
    assert first["samples"] == second["samples"]
    assert [sample["position_name"] for sample in first["samples"][:5]] == [
        "center", "southwest", "southeast", "northwest", "northeast",
    ]
    assert first["samples"][219]["source_sample_id"] == 43
    assert first["samples"][219]["sample_id"] == 219
    assert first["plan"]["minimum_parent_download_size_vis_pixels"] == 1856
    assert first["plan"]["parent_download_size_vis_pixels"] == 1920
    manifest_path = tmp_path / "archive_fields_manifest.json"
    manifest_path.write_text(json.dumps(first), encoding="utf-8")
    loaded = archive_fields.load_manifest(manifest_path)
    assert loaded["collection_fingerprint"] == first["collection_fingerprint"]


def test_forced_redownload_is_persisted_and_ordinary_resume_does_not_reuse(
    tmp_path, monkeypatch,
):
    source = _source_manifest()
    source_sha = __import__("hashlib").sha256(
        json.dumps(source, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = sampling._new_archive_fields_manifest(source, source_sha)
    sample = manifest["samples"][0]
    sample.update({
        "status": "cached",
        "output_path": str(tmp_path / "cutouts" / "field_0000.fits"),
        "bundle_sha256": "a" * 64,
        "bands": {"old": True},
    })
    validations = []

    def validate(target, candidate):
        validations.append((target, candidate["sample_id"]))
        return {"bundle_sha256": "b" * 64, "bands": {"new": True}}

    monkeypatch.setattr(sampling, "_validate_archive_field_bundle", validate)

    sampling._prepare_archive_samples_for_download(
        [sample],
        output_dir=str(tmp_path),
        plan_fingerprint=manifest["plan_fingerprint"],
        force_redownload=True,
    )
    assert validations == []
    assert sample["status"] == "planned"
    assert sample["redownload_required"] is True
    assert "bundle_sha256" not in sample
    assert sample["bands"] == {}

    # This is the next invocation after an interrupted force run.  The normal
    # resume path must honor the persisted marker and not reclaim the old file.
    sampling._prepare_archive_samples_for_download(
        [sample],
        output_dir=str(tmp_path),
        plan_fingerprint=manifest["plan_fingerprint"],
        force_redownload=False,
    )
    assert validations == []
    assert sample["status"] == "planned"
    assert sample["redownload_required"] is True

    # Successful atomic replacement clears the marker in the acquisition
    # loop; future ordinary runs can then validate and reuse that new bundle.
    sample.pop("redownload_required")
    sampling._prepare_archive_samples_for_download(
        [sample],
        output_dir=str(tmp_path),
        plan_fingerprint=manifest["plan_fingerprint"],
        force_redownload=False,
    )
    assert len(validations) == 1
    assert sample["status"] == "cached"
    assert sample["bundle_sha256"] == "b" * 64
    assert sample["bands"] == {"new": True}


def test_crop_requires_exact_common_pixel_grid():
    raw = _raw_bands()
    centre = WCS(raw["VIS"][1]).celestial.pixel_to_world(300.0, 300.0)

    crops = sampling._aligned_archive_crops(
        raw, ra=float(centre.ra.deg), dec=float(centre.dec.deg),
    )

    assert [float(crops[name][0][0, 0]) for name in crops] == [1.0, 2.0, 3.0, 4.0]
    reference_world = WCS(crops["VIS"][1]).celestial.pixel_to_world(0.0, 255.0)
    for tile, header in crops.values():
        assert tile.shape == (256, 256)
        x, y = WCS(header).celestial.world_to_pixel(reference_world)
        assert x == pytest.approx(0.0, abs=1e-6)
        assert y == pytest.approx(255.0, abs=1e-6)

    bad = _raw_bands()
    bad["H_E"][1]["CD1_1"] *= 1.01
    with pytest.raises(ValueError, match="not on the VIS pixel grid"):
        sampling._aligned_archive_crops(
            bad, ra=float(centre.ra.deg), dec=float(centre.dec.deg),
        )


def test_1920_parent_cutout_covers_all_five_offset_tiles():
    header = _header(shift_x=660.0, shift_y=660.0)
    pixels = np.broadcast_to(
        np.float32(1.0),
        (
            sampling.ARCHIVE_FIELDS_PARENT_DOWNLOAD_VIS_PIXELS,
            sampling.ARCHIVE_FIELDS_PARENT_DOWNLOAD_VIS_PIXELS,
        ),
    )
    raw = {
        band: (pixels, header.copy()) for band in archive_fields.BAND_NAMES
    }
    for _name, east, north in sampling.ARCHIVE_FIELDS_PATTERN:
        ra, dec = sampling._offset_coordinate(10.0, 20.0, east, north)
        crops = sampling._aligned_archive_crops(raw, ra=ra, dec=dec)
        for tile, crop_header in crops.values():
            assert tile.shape == (256, 256)
            assert 0 <= int(crop_header["CROPX0"]) <= 1664
            assert 0 <= int(crop_header["CROPY0"]) <= 1664


def test_atomic_bundle_round_trips_through_shared_provider(tmp_path):
    source = _source_manifest()
    encoded = json.dumps(source, sort_keys=True).encode("utf-8")
    source_sha = __import__("hashlib").sha256(encoded).hexdigest()
    manifest = sampling._new_archive_fields_manifest(source, source_sha)
    sample = manifest["samples"][0]
    sample["archive_parents"] = {
        band: _parent(band, 0) for band in archive_fields.BAND_NAMES
    }
    centre = WCS(_raw_bands()["VIS"][1]).celestial.pixel_to_world(300.0, 300.0)
    sample["ra"] = float(centre.ra.deg)
    sample["dec"] = float(centre.dec.deg)
    crops = sampling._aligned_archive_crops(
        _raw_bands(), ra=sample["ra"], dec=sample["dec"],
    )
    root = tmp_path / "archive_fields"
    target = root / "cutouts" / "field_0000.fits"

    metadata = sampling._write_archive_field_bundle(
        str(target), sample=sample, crops=crops,
    )

    sample.update(metadata)
    sample["status"] = "written"
    sample["output_path"] = str(target)
    manifest["collection_fingerprint"] = archive_fields.compute_collection_fingerprint(
        manifest
    )
    manifest_path = root / "archive_fields_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cube = archive_fields.load_field(0, manifest_file=manifest_path)
    assert cube.shape == (256, 256, 4)
    assert np.all(np.isfinite(cube))
    assert metadata["bundle_sha256"] == sampling._sha256_file(target)
    assert not target.with_suffix(".fits.tmp").exists()


def test_parent_download_retries_bounded_and_validates(monkeypatch, tmp_path):
    calls = []

    def download(_ra, _dec, _config, _radius, output_file, _parent):
        calls.append(output_file)
        if len(calls) < 3:
            return False
        fits.PrimaryHDU(
            data=np.ones((300, 300), dtype=np.float32), header=_header(),
        ).writeto(output_file)
        return True

    monkeypatch.setattr(sampling, "download_one_cutout", download)
    monkeypatch.setattr(
        sampling, "ARCHIVE_FIELDS_DOWNLOAD_RETRY_DELAYS_SECONDS", (0.0, 0.0),
    )
    target = tmp_path / "Y_E.fits"

    sampling._download_parent_band(
        band_name="Y_E",
        ra=10.0,
        dec=20.0,
        parent_download_vis_pixels=1920,
        parent=_parent("Y_E", 0),
        output_path=str(target),
    )

    assert len(calls) == 3
    assert target.is_file()


def test_raw_archive_reader_accepts_numeric_extname_before_normalization(tmp_path):
    target = tmp_path / "Y_E.fits"
    header = _header()
    header["EXTNAME"] = 7
    fits.HDUList([
        fits.PrimaryHDU(
            data=np.ones((300, 300), dtype=np.float32),
            header=header,
        ),
    ]).writeto(target, output_verify="ignore")

    data, normalized_header = sampling._read_archive_image(target, "Y_E")

    assert data.shape == (300, 300)
    assert normalized_header["EXTNAME"] == 7
