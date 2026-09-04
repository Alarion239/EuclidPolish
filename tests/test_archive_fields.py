"""Focused contracts for the multipoint archive-field provider and viewer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.photometry import adu_per_s_to_electrons_factor
from euclid_polish.web.helpers import archive_fields, viewer_data


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(path: Path, *, bad_band_shape: bool = False) -> None:
    primary = fits.PrimaryHDU()
    primary.header["SAMPLEID"] = 0
    primary.header["FIELD_ID"] = 0
    primary.header["SRC_ID"] = 0
    primary.header["PARENT"] = "parent-000"
    primary.header["Q1FIELD"] = "EDF-N"
    primary.header["RA"] = 17.5
    primary.header["DEC"] = 66.25
    primary.header["RELEASE"] = "Q1_R1"
    primary.header["SRCPLAN"] = "a" * 64
    hdus: list[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU] = [primary]
    for index, band_name in enumerate(Config.LR_INPUT_BAND_NAMES):
        side = 255 if bad_band_shape and band_name == "H_E" else 256
        hdu = fits.ImageHDU(
            np.full((side, side), index + 1, dtype=np.float32),
            name=band_name,
        )
        hdu.header["MAGZERO"] = 24.6
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path)


def _write_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    bad_band_shape: bool = False,
) -> tuple[Path, Path]:
    # Keep the fixture tiny while exercising the same strict cardinality rule
    # as the production 44 x 5 collection.
    monkeypatch.setattr(archive_fields, "SOURCE_SAMPLE_COUNT", 1)
    monkeypatch.setattr(archive_fields, "POSITIONS_PER_PARENT", 1)
    monkeypatch.setattr(archive_fields, "SAMPLE_COUNT", 1)
    sky = tmp_path / "euclid_sky"
    source_path = sky / "vis_noise_samples" / "vis_noise_sampling_manifest.json"
    source_path.parent.mkdir(parents=True)
    remote_source = {
        "kind": "euclid_vis_noise_sampling",
        "version": 1,
        "source_release": "Q1_R1",
        "plan_fingerprint": "a" * 64,
        "samples": [],
    }
    remote_source_bytes = _json_bytes(remote_source)
    remote_source_sha = hashlib.sha256(remote_source_bytes).hexdigest()
    # A local sync rewrites paths/status and adds sync provenance, so its byte
    # digest is intentionally not the immutable remote digest.
    local_source = {
        **remote_source,
        "sync": {"remote_manifest_sha256": remote_source_sha},
    }
    source_path.write_bytes(_json_bytes(local_source))

    root = sky / "archive_fields"
    cutouts = root / "cutouts"
    cutouts.mkdir(parents=True)
    bundle_path = cutouts / "field_0000.fits"
    _write_bundle(bundle_path, bad_band_shape=bad_band_shape)

    plan = {
        "source_manifest_sha256": remote_source_sha,
        "source_plan_fingerprint": "a" * 64,
        "source_release": "Q1_R1",
        "source_sample_count": 1,
        "positions_per_parent": 1,
        "cutout_size_vis_pixels": 256,
        "bands": list(Config.LR_INPUT_BAND_NAMES),
        "offset_pattern_arcsec": [
            {"name": "center", "east": 0.0, "north": 0.0},
        ],
        "registration_method": "celestial WCS common-grid crop",
    }
    plan_fingerprint = archive_fields.compute_plan_fingerprint(plan)
    with fits.open(bundle_path, mode="update", memmap=False) as hdul:
        hdul[0].header["PLANHASH"] = plan_fingerprint
        hdul.flush()
    sample = {
        "sample_id": 0,
        "field_id": 0,
        "source_sample_id": 0,
        "anchor_id": 0,
        "parent_id": "parent-000",
        "field": "EDF-N",
        "position_index": 0,
        "position_name": "center",
        "east_offset_arcsec": 0.0,
        "north_offset_arcsec": 0.0,
        "ra": 17.5,
        "dec": 66.25,
        "source_ra": 17.5,
        "source_dec": 66.25,
        "source_plan_fingerprint": "a" * 64,
        "plan_fingerprint": plan_fingerprint,
        "source_release": "Q1_R1",
        "status": "written",
        "error": None,
        "output_path": "/remote/data/euclid_sky/archive_fields/cutouts/field_0000.fits",
        "bundle_sha256": _sha256(bundle_path),
        "bands": {
            name: {"shape": [256, 256], "source_shape": [300, 300], "header": {"MAGZERO": 24.6}}
            for name in Config.LR_INPUT_BAND_NAMES
        },
    }
    manifest = {
        "version": 1,
        "kind": "euclid_archive_fields",
        "created_at": "2026-09-04T12:00:00Z",
        "source_release": "Q1_R1",
        "source": {
            "manifest_kind": "euclid_vis_noise_sampling",
            "manifest_version": 1,
            "manifest_sha256": remote_source_sha,
            "plan_fingerprint": "a" * 64,
        },
        "plan": plan,
        "plan_fingerprint": plan_fingerprint,
        "samples": [sample],
    }
    manifest["collection_fingerprint"] = archive_fields.compute_collection_fingerprint(
        manifest,
    )
    manifest_path = root / "archive_fields_manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    monkeypatch.setattr(Config, "EUCLID_SKY_DIR", str(sky))
    return manifest_path, source_path


def test_archive_provider_accepts_synced_source_and_converts_four_bands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, source_path = _write_collection(tmp_path, monkeypatch)
    status = archive_fields.availability()
    assert status["valid"] is True
    assert status["complete"] is True
    assert status["current"] is True
    assert status["ready"] is True
    assert status["sample_count"] == 1
    assert status["parent_count"] == 1
    assert status["fields"] == {"EDF-N": 1}
    assert status["manifest_fingerprint"] == _sha256(manifest_path)
    assert status["source_manifest_sha256"] != _sha256(source_path)

    field = next(archive_fields.iter_fields())
    assert field.sample_id == 0
    assert field.source_sample_id == 0
    assert field.parent_id == "parent-000"
    assert field.path == manifest_path.parent / "cutouts" / "field_0000.fits"
    cube = archive_fields.load_field(field)
    assert cube.shape == (256, 256, 4)
    assert cube.dtype == np.float32
    for index, band_name in enumerate(Config.LR_INPUT_BAND_NAMES):
        expected = (index + 1) * adu_per_s_to_electrons_factor(
            24.6, Config.get_band(band_name),
        )
        np.testing.assert_allclose(cube[..., index], expected, rtol=1e-6)


def test_archive_provider_rejects_stale_source_and_bad_bundle_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source_path = _write_collection(
        tmp_path, monkeypatch, bad_band_shape=True,
    )
    with pytest.raises(archive_fields.ArchiveFieldError, match="H_E shape"):
        archive_fields.load_field(0)

    source = json.loads(source_path.read_text())
    source["plan_fingerprint"] = "b" * 64
    source_path.write_bytes(_json_bytes(source))
    status = archive_fields.availability()
    assert status["valid"] is True
    assert status["complete"] is True
    assert status["current"] is False
    assert status["ready"] is False
    assert "source VIS-pointing plan has changed" in status["reasons"]


def test_archive_manifest_fingerprint_detects_metadata_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = _write_collection(tmp_path, monkeypatch)
    manifest = json.loads(path.read_text())
    manifest["samples"][0]["ra"] = 17.6
    path.write_bytes(_json_bytes(manifest))
    with pytest.raises(archive_fields.ArchiveFieldError, match="collection_fingerprint"):
        archive_fields.load_manifest()


def test_archive_fields_viewer_exposes_provenance_and_raw_cube(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_collection(tmp_path, monkeypatch)
    meta = viewer_data.get_meta("archive-fields", {})
    assert meta["count"] == 1
    assert meta["default_tier"] == "lr"
    assert meta["band_names"] == ["VIS", "Y_E", "J_E", "H_E"]
    assert meta["archive"]["parent_count"] == 1
    assert meta["objects"][0] == {
        "label": "EDF-N · pointing 1 · center · sample 1",
        "tiers": ["lr"],
        "sample_id": 0,
        "source_sample_id": 0,
        "parent_id": "parent-000",
        "field": "EDF-N",
        "ra": 17.5,
        "dec": 66.25,
        "position_name": "center",
    }
    cube, info = viewer_data.get_cube("archive-fields", 0, "lr", {})
    assert cube.shape == (256, 256, 4)
    assert info["bands"] == ["VIS", "Y_E", "J_E", "H_E"]
    with pytest.raises(viewer_data.ViewerError) as error:
        viewer_data.get_cube("archive-fields", 0, "sr", {})
    assert error.value.code == 400


def test_archive_fields_http_surface_serves_the_manifest_backed_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_collection(tmp_path, monkeypatch)
    from euclid_polish.web.app import create_app

    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        status = client.get("/api/archive-fields")
        assert status.status_code == 200
        assert status.get_json()["ready"] is True

        meta = client.get("/viewer/meta/archive-fields")
        assert meta.status_code == 200
        assert meta.get_json()["count"] == 1

        cube = client.get("/viewer/cube/archive-fields/0?tier=lr")
        assert cube.status_code == 200
        assert cube.headers["X-Cube-Shape"] == "256,256,4"
        assert cube.headers["X-Cube-Bands"] == "VIS,Y_E,J_E,H_E"
        assert len(cube.data) == 256 * 256 * 4 * 4
