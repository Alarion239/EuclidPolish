"""Native-radius manifest contracts and typed-renderer integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.sky.generation.tng_galaxy import (
    TNGRenderer,
    list_tng_galaxies,
    tng_fits_path,
)
from euclid_polish.sky.generation.tng_radius_manifest import (
    build_manifest,
    ensure_manifest,
    load_parameter_summary,
    radius_lookup,
    validate_manifest,
    write_parameter_summary,
)
from euclid_polish.sky.generation.tng_types import TNGView


def _write_frame(path: Path, data: np.ndarray) -> None:
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=">f4"))
    hdu.header["BUNIT"] = "MJy/sr"
    hdu.header["CDELT1"] = 100.0
    hdu.header["CUNIT1"] = "pc"
    hdu.header["CDELT2"] = 100.0
    hdu.header["CUNIT2"] = "pc"
    hdu.writeto(path, overwrite=True)


def _atlas(root: Path, subhalo_id: str = "42", size: int = 64) -> Path:
    folder = root / subhalo_id
    folder.mkdir(parents=True)
    yy, xx = np.mgrid[:size, :size]
    frame = np.exp(
        -((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / 60.0
    ).astype(np.float32)
    for orientation in range(1, 6):
        for band, amplitude in zip(
            ("VIS", "Y", "J", "H"),
            (1.0, 2.0, 3.0, 4.0),
            strict=True,
        ):
            _write_frame(
                folder
                / f"TNG{subhalo_id}_O{orientation}_Euclid_{band}.fits",
                frame * amplitude,
            )
    (folder / Config.Tng.DONE_MARKER).touch()
    return folder


def _properties(path: Path, *subhalo_ids: str) -> None:
    rows = ["id,sfr,mass_stars,m_halo,reff"]
    rows.extend(
        f"{subhalo_id},1,1e10,1e12,2" for subhalo_id in subhalo_ids
    )
    path.write_text("\n".join(rows) + "\n")


def test_manifest_is_atomic_and_validates_inventory(tmp_path):
    atlas = tmp_path / "tng_skirt"
    atlas.mkdir()
    _atlas(atlas)
    properties = tmp_path / "props.csv"
    _properties(properties, "42")
    output = tmp_path / "manifest.json"

    report = build_manifest(
        str(atlas),
        properties_path=str(properties),
        output_path=str(output),
    )

    assert report["valid"] and report["valid_count"] == 5
    assert not (tmp_path / "manifest.json.tmp").exists()
    assert validate_manifest(
        str(atlas),
        properties_path=str(properties),
        manifest_path_value=str(output),
    )["valid"]

    path = atlas / "42" / "TNG42_O3_Euclid_VIS.fits"
    _write_frame(path, np.ones((64, 64), dtype=np.float32))
    status = validate_manifest(
        str(atlas),
        properties_path=str(properties),
        manifest_path_value=str(output),
    )
    assert not status["valid"]
    assert any("changed" in reason for reason in status["reasons"])


def test_ensure_manifest_measures_only_new_orientations(tmp_path, monkeypatch):
    import euclid_polish.sky.generation.tng_radius_manifest as module

    atlas = tmp_path / "tng_skirt"
    atlas.mkdir()
    _atlas(atlas, "42")
    properties = tmp_path / "props.csv"
    _properties(properties, "42")
    output = tmp_path / "manifest.json"
    build_manifest(
        str(atlas),
        properties_path=str(properties),
        output_path=str(output),
    )

    _atlas(atlas, "43")
    _properties(properties, "42", "43")
    loaded: list[str] = []
    original_load = module.load_skirt_image

    def counted_load(path, band_name):
        loaded.append(str(path))
        return original_load(path, band_name)

    monkeypatch.setattr(module, "load_skirt_image", counted_load)
    result = ensure_manifest(
        str(atlas),
        properties_path=str(properties),
        manifest_path_value=str(output),
        workers=2,
    )

    assert result["valid"] and result["repaired"]
    assert result["reused_count"] == 5
    assert result["measured_count"] == 5
    assert len(loaded) == 5


def test_zero_padded_atlas_filenames_are_not_excluded(tmp_path):
    atlas = tmp_path / "tng_skirt"
    atlas.mkdir()
    folder = _atlas(atlas, "1")
    for path in list(folder.glob("TNG1_*.fits")):
        path.rename(folder / path.name.replace("TNG1_", "TNG000001_"))

    galaxies = list_tng_galaxies(atlas)

    assert len(galaxies) == 1
    assert tng_fits_path(folder, "1", 1, "VIS").endswith(
        "TNG000001_O1_Euclid_VIS.fits"
    )


def test_manifest_payload_is_strict_json(tmp_path):
    atlas = tmp_path / "tng_skirt"
    atlas.mkdir()
    _atlas(atlas)
    properties = tmp_path / "props.csv"
    _properties(properties, "42")

    report = build_manifest(
        str(atlas), properties_path=str(properties), workers=2
    )

    json.dumps(report, allow_nan=False)


def test_parameter_summary_joins_properties_without_reopening_pixels(tmp_path):
    atlas = tmp_path / "tng_skirt"
    atlas.mkdir()
    _atlas(atlas)
    properties = tmp_path / "props.csv"
    _properties(properties, "42")
    report = build_manifest(str(atlas), properties_path=str(properties))
    summary = tmp_path / "tng_atlas_parameters.csv"

    metadata = write_parameter_summary(
        summary, report, properties_path=str(properties)
    )
    loaded = load_parameter_summary(summary)

    assert metadata["galaxy_count"] == 1
    assert metadata["row_count"] == 5
    assert len(loaded["rows"]) == 5
    assert loaded["rows"][0]["mass_stars_msun"] == pytest.approx(1e10)
    assert loaded["rows"][0]["native_re_kpc"] == pytest.approx(
        loaded["rows"][0]["native_re_px"] * 0.1
    )
    summary.write_text(summary.read_text() + "\n")
    with pytest.raises(ValueError, match="fingerprint"):
        load_parameter_summary(summary)


def test_manifest_radius_constructs_view_and_drives_typed_render(tmp_path):
    atlas = tmp_path / "tng_skirt"
    atlas.mkdir()
    folder = _atlas(atlas)
    properties = tmp_path / "props.csv"
    _properties(properties, "42")
    report = build_manifest(str(atlas), properties_path=str(properties))
    radii = radius_lookup(report)
    native_re_px = radii[("42", 1)]
    view = TNGView(
        galaxy_dir=folder,
        subhalo_id="42",
        orientation=1,
        native_re_px=native_re_px,
        radius_manifest_fingerprint=report["manifest_fingerprint"],
    )
    target = 0.5 * native_re_px * Config.DEFAULT_PIXEL_SCALE

    rendered = TNGRenderer().render_for_radius(
        view, target, target_vis_flux_e=1e5
    )

    assert rendered.trace.view.radius_manifest_fingerprint == (
        report["manifest_fingerprint"]
    )
    assert rendered.trace.geometry.target_re_arcsec == pytest.approx(target)
    assert rendered.flux_e("VIS") == pytest.approx(1e5, rel=2e-6)


def test_manifest_rejects_fits_without_physical_unit_metadata(tmp_path):
    atlas = tmp_path / "tng_skirt"
    atlas.mkdir()
    folder = _atlas(atlas)
    fits.PrimaryHDU(np.ones((64, 64), dtype=np.float32)).writeto(
        folder / "TNG42_O2_Euclid_VIS.fits", overwrite=True
    )
    properties = tmp_path / "props.csv"
    _properties(properties, "42")

    report = build_manifest(str(atlas), properties_path=str(properties))

    assert not report["valid"]
    invalid = [entry for entry in report["entries"] if not entry["valid"]]
    assert len(invalid) == 1
    assert "BUNIT" in invalid[0]["error"]
