import json

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.sky.generation import tng_galaxy
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
        for band, amplitude in zip(
            ("VIS", "Y", "J", "H"), (1.0, 2.0, 3.0, 4.0), strict=True,
        ):
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


def test_nominal_radius_scale_ignores_atlas_frame_side(tmp_path):
    atlas_a = tmp_path / "a"
    atlas_b = tmp_path / "b"
    atlas_a.mkdir(); atlas_b.mkdir()
    _atlas(atlas_a, size=64)
    _atlas(atlas_b, size=128)
    native_a = tng_galaxy.native_halflight_px(
        str(atlas_a / "42"), "42", 1,
    )
    native_b = tng_galaxy.native_halflight_px(
        str(atlas_b / "42"), "42", 1,
    )
    stamp_a, meta_a = tng_stamp_to_target_re(
        str(atlas_a / "42"), "42", 1, 0.20, target_vis_flux_e=1e5,
        native_re_px=native_a,
    )
    stamp_b, meta_b = tng_stamp_to_target_re(
        str(atlas_b / "42"), "42", 1, 0.20, target_vis_flux_e=1e5,
        native_re_px=native_b,
    )
    assert meta_a["nominal_re_arcsec"] == pytest.approx(0.20)
    assert meta_b["nominal_re_arcsec"] == pytest.approx(0.20)
    assert meta_a["target_re_arcsec"] == meta_b["target_re_arcsec"] == 0.20
    assert meta_a["radius_remeasured"] is meta_b["radius_remeasured"] is False
    assert "achieved_re_arcsec" not in meta_a
    assert "achieved_re_arcsec" not in meta_b
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
    native_re_px = tng_galaxy.native_halflight_px(folder, "42", 1)
    scaled, meta = tng_stamp_to_target_re(
        folder, "42", 1, 0.20, target_vis_flux_e=1e5,
        native_re_px=native_re_px,
    )
    native_flux = native.sum(axis=(0, 1), dtype=np.float64)
    scaled_flux = scaled.sum(axis=(0, 1), dtype=np.float64)
    assert scaled_flux / scaled_flux[0] == pytest.approx(
        native_flux / native_flux[0], rel=2e-5,
    )
    assert meta["photometric_scaling"] == (
        "single_shared_vis_anchor_after_nominal_scale"
    )
    assert scaled_flux[0] == pytest.approx(1e5, rel=2e-5)


def test_target_re_requires_validated_native_radius(tmp_path):
    atlas = tmp_path / "tng_skirt"
    atlas.mkdir(); _atlas(atlas)
    with pytest.raises(ValueError, match="validated radius manifest"):
        tng_stamp_to_target_re(str(atlas / "42"), "42", 1, 0.20)


def test_subpixel_radius_uses_one_resize_and_cached_native_source(
    tmp_path, monkeypatch,
):
    atlas = tmp_path / "tng_skirt"
    atlas.mkdir(); _atlas(atlas)
    original_load = tng_galaxy.load_tng_frame
    native_re_px = tng_galaxy.measure_halflight_radius_px(original_load(
        str(atlas / "42" / "TNG42_O1_Euclid_VIS.fits")
    ))
    loaded_paths = []
    resize_count = 0
    original_resize = tng_galaxy.resample_surface_brightness

    def counted_load(path):
        loaded_paths.append(path)
        return original_load(path)

    def counted_resize(values, scale, **kwargs):
        nonlocal resize_count
        resize_count += 1
        return original_resize(values, scale, **kwargs)

    def forbidden_remeasurement(*_args, **_kwargs):
        raise AssertionError("one-pass rendering must not remeasure output R_e")

    tng_galaxy._clear_tng_source_cache()
    monkeypatch.setattr(tng_galaxy, "load_tng_frame", counted_load)
    monkeypatch.setattr(
        tng_galaxy, "resample_surface_brightness", counted_resize,
    )
    monkeypatch.setattr(
        tng_galaxy, "measure_halflight_radius_px", forbidden_remeasurement,
    )
    _stamp, meta = tng_stamp_to_target_re(
        str(atlas / "42"), "42", 1, 0.055,
        target_vis_flux_e=1e5, native_re_px=native_re_px,
    )

    assert meta["nominal_re_arcsec"] == pytest.approx(0.055)
    assert meta["radius_remeasured"] is False
    assert resize_count == 1
    assert len(loaded_paths) == 4

    # A repeated orientation is served from the byte-bounded source cache.
    tng_stamp_to_target_re(
        str(atlas / "42"), "42", 1, 0.20,
        target_vis_flux_e=1e5, native_re_px=native_re_px,
    )
    assert resize_count == 2
    assert len(loaded_paths) == 4


def test_registered_source_cache_is_byte_bounded(tmp_path, monkeypatch):
    atlas = tmp_path / "tng_skirt"
    atlas.mkdir(); _atlas(atlas)
    folder = str(atlas / "42")
    tng_galaxy._clear_tng_source_cache()
    monkeypatch.setattr(tng_galaxy, "_TNG_SOURCE_CACHE_MAX_BYTES", 70_000)

    for orientation in (1, 2, 3):
        tng_stamp_to_target_re(
            folder, "42", orientation, 0.20, native_re_px=5.0,
        )

    assert tng_galaxy._TNG_SOURCE_CACHE_BYTES <= 70_000
    assert sum(
        source.nbytes for source in tng_galaxy._TNG_SOURCE_CACHE.values()
    ) == tng_galaxy._TNG_SOURCE_CACHE_BYTES
    assert len(tng_galaxy._TNG_SOURCE_CACHE) == 1


def test_registered_source_cache_invalidates_replaced_fits(
    tmp_path, monkeypatch,
):
    atlas = tmp_path / "tng_skirt"
    atlas.mkdir(); _atlas(atlas)
    folder = str(atlas / "42")
    original_load = tng_galaxy.load_tng_frame
    loaded_paths = []

    def counted_load(path):
        loaded_paths.append(path)
        return original_load(path)

    tng_galaxy._clear_tng_source_cache()
    monkeypatch.setattr(tng_galaxy, "load_tng_frame", counted_load)
    tng_stamp_to_target_re(folder, "42", 1, 0.20, native_re_px=5.0)
    path = atlas / "42" / "TNG42_O1_Euclid_VIS.fits"
    fits.PrimaryHDU(np.ones((64, 64), dtype="f4")).writeto(
        path, overwrite=True,
    )
    tng_stamp_to_target_re(folder, "42", 1, 0.20, native_re_px=5.0)

    assert len(loaded_paths) == 8


@pytest.mark.parametrize("target_re", (0.03, 0.05, 0.10, 0.30, 1.0, 10.0))
def test_nominal_radius_boundaries_render_once(tmp_path, target_re):
    atlas = tmp_path / "tng_skirt"
    atlas.mkdir(); _atlas(atlas)
    stamp, meta = tng_stamp_to_target_re(
        str(atlas / "42"), "42", 1, target_re,
        native_re_px=5.0, max_output_side=65,
    )

    assert meta["nominal_re_arcsec"] == pytest.approx(target_re)
    assert meta["radius_scale_factor"] == pytest.approx(
        target_re / (5.0 * Config.DEFAULT_PIXEL_SCALE)
    )
    assert max(stamp.shape[:2]) <= 65
    assert meta["radius_remeasured"] is False


def test_bounded_support_matches_unbounded_central_render(tmp_path):
    atlas = tmp_path / "tng_skirt"
    atlas.mkdir(); _atlas(atlas)
    folder = str(atlas / "42")
    full, _ = tng_stamp_to_target_re(
        folder, "42", 1, 2.0, native_re_px=5.0,
    )
    bounded, meta = tng_stamp_to_target_re(
        folder, "42", 1, 2.0, native_re_px=5.0, max_output_side=65,
    )
    height, width = bounded.shape[:2]
    y0 = (full.shape[0] - height) // 2
    x0 = (full.shape[1] - width) // 2

    np.testing.assert_allclose(
        bounded, full[y0:y0 + height, x0:x0 + width],
        rtol=2e-3, atol=1e-3,
    )
    assert meta["render_support_clipped"] is True


@pytest.mark.parametrize("position", ((0.0, 0.0), (95.0, 95.0), (48.0, 48.0)))
def test_bounded_arbitrary_rotation_preserves_field_pixels(tmp_path, position):
    atlas = tmp_path / "tng_skirt"
    atlas.mkdir(); _atlas(atlas)
    folder = str(atlas / "42")
    full, _ = tng_stamp_to_target_re(
        folder, "42", 1, 2.0, native_re_px=5.0,
        rng=np.random.default_rng(71),
    )
    bounded, meta = tng_stamp_to_target_re(
        folder, "42", 1, 2.0, native_re_px=5.0,
        rng=np.random.default_rng(71), max_output_side=193,
    )
    full_canvas = np.zeros((96, 96, 4), dtype=np.float32)
    bounded_canvas = np.zeros_like(full_canvas)
    tng_galaxy.composite_stamp(full_canvas, full, *position)
    tng_galaxy.composite_stamp(bounded_canvas, bounded, *position)
    residual = bounded_canvas - full_canvas

    assert np.linalg.norm(residual) / np.linalg.norm(full_canvas) < 5e-4
    assert meta["render_support_clipped"] is True
