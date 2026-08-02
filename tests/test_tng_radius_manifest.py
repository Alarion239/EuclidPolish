import json

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.sky.generation.tng_galaxy import (
    list_tng_galaxies,
    prepare_tng_galaxy_continuous,
    tng_fits_path,
    tng_stamp_to_target_re,
)
from euclid_polish.sky.generation.tng_radius_manifest import (
    build_manifest,
    load_parameter_summary,
    validate_manifest,
    write_parameter_summary,
)


def _atlas(root, gid="42", size=64):
    folder = root / gid
    folder.mkdir(parents=True)
    y, x = np.mgrid[:size, :size]
    frame = np.exp(-((x - size / 2) ** 2 + (y - size / 2) ** 2) / 60.0).astype("f4")
    for orientation in range(1, 6):
        for band, amplitude in zip(("VIS", "Y", "J", "H"), (1.0, 2.0, 3.0, 4.0)):
            fits.PrimaryHDU(frame * amplitude).writeto(
                folder / f"TNG{gid}_O{orientation}_Euclid_{band}.fits"
            )
    (folder / Config.Tng.DONE_MARKER).touch()


def test_manifest_is_atomic_and_validates_inventory(tmp_path):
    atlas = tmp_path / "tng_skirt"
    atlas.mkdir()
    _atlas(atlas)
    properties = tmp_path / "props.csv"
    properties.write_text("id,sfr,mass_stars,m_halo,reff\n42,1,1e10,1e12,2\n")
    output = tmp_path / "manifest.json"

    report = build_manifest(str(atlas), properties_path=str(properties), output_path=str(output))
    assert report["valid"] and report["valid_count"] == 5
    assert not (tmp_path / "manifest.json.tmp").exists()
    assert validate_manifest(str(atlas), properties_path=str(properties),
                             manifest_path_value=str(output))["valid"]

    # Any frame replacement invalidates the inventory fingerprint.
    path = atlas / "42" / "TNG42_O3_Euclid_VIS.fits"
    fits.PrimaryHDU(np.ones((64, 64), dtype="f4")).writeto(path, overwrite=True)
    status = validate_manifest(str(atlas), properties_path=str(properties),
                               manifest_path_value=str(output))
    assert not status["valid"]
    assert any("changed" in reason for reason in status["reasons"])


def test_zero_padded_atlas_filenames_are_not_excluded(tmp_path):
    atlas = tmp_path / "tng_skirt"
    atlas.mkdir()
    _atlas(atlas, gid="1")
    folder = atlas / "1"
    for path in list(folder.glob("TNG1_*.fits")):
        path.rename(folder / path.name.replace("TNG1_", "TNG000001_"))

    galaxies = list_tng_galaxies(str(atlas))

    assert len(galaxies) == 1
    assert tng_fits_path(str(folder), "1", 1, "VIS").endswith(
        "TNG000001_O1_Euclid_VIS.fits"
    )


def test_radius_match_ignores_atlas_frame_side(tmp_path):
    atlas_a = tmp_path / "a"
    atlas_b = tmp_path / "b"
    atlas_a.mkdir(); atlas_b.mkdir()
    _atlas(atlas_a, size=64)
    _atlas(atlas_b, size=128)
    stamp_a, meta_a = tng_stamp_to_target_re(
        str(atlas_a / "42"), "42", 1, 0.20, target_vis_flux_e=1e5
    )
    stamp_b, meta_b = tng_stamp_to_target_re(
        str(atlas_b / "42"), "42", 1, 0.20, target_vis_flux_e=1e5
    )
    assert meta_a["achieved_re_arcsec"] == pytest.approx(
        meta_b["achieved_re_arcsec"], abs=0.5 * Config.DEFAULT_PIXEL_SCALE
    )
    assert meta_a["target_re_arcsec"] == meta_b["target_re_arcsec"] == 0.20
    assert np.isfinite(stamp_a).all() and np.isfinite(stamp_b).all()


def test_manifest_payload_is_json_serializable(tmp_path):
    atlas = tmp_path / "tng_skirt"
    atlas.mkdir(); _atlas(atlas)
    properties = tmp_path / "props.csv"
    properties.write_text("id,sfr,mass_stars,m_halo,reff\n42,1,1e10,1e12,2\n")
    report = build_manifest(
        str(atlas), properties_path=str(properties), workers=2,
    )
    json.dumps(report, allow_nan=False)


def test_parameter_summary_joins_properties_without_atlas_pixels(tmp_path):
    atlas = tmp_path / "tng_skirt"
    atlas.mkdir(); _atlas(atlas)
    properties = tmp_path / "props.csv"
    properties.write_text("id,sfr,mass_stars,m_halo,reff\n42,1,1e10,1e12,2\n")
    report = build_manifest(str(atlas), properties_path=str(properties))
    summary = tmp_path / "tng_atlas_parameters.csv"

    meta = write_parameter_summary(
        summary, report, properties_path=str(properties),
    )
    loaded = load_parameter_summary(summary)

    assert meta["galaxy_count"] == 1
    assert meta["row_count"] == 5
    assert len(loaded["rows"]) == 5
    assert loaded["rows"][0]["mass_stars_msun"] == pytest.approx(1e10)
    assert loaded["rows"][0]["native_re_kpc"] == pytest.approx(
        loaded["rows"][0]["native_re_px"] * 0.1
    )
    summary.write_text(summary.read_text() + "\n")
    with pytest.raises(ValueError, match="fingerprint"):
        load_parameter_summary(summary)


def test_target_re_uses_one_shared_cube_scale(tmp_path):
    atlas = tmp_path / "tng_skirt"
    atlas.mkdir(); _atlas(atlas)
    folder = str(atlas / "42")
    native, _ = prepare_tng_galaxy_continuous(
        folder, "42", 1, scale=1.0,
    )
    scaled, meta = tng_stamp_to_target_re(
        folder, "42", 1, 0.20, target_vis_flux_e=1e5,
    )
    native_flux = native.sum(axis=(0, 1), dtype=np.float64)
    scaled_flux = scaled.sum(axis=(0, 1), dtype=np.float64)
    assert scaled_flux / scaled_flux[0] == pytest.approx(
        native_flux / native_flux[0], rel=2e-5,
    )
    assert meta["photometric_scaling"] == "single_shared_vis_anchor"
    assert scaled_flux[0] == pytest.approx(1e5, rel=2e-5)
