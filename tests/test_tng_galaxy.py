"""Focused contracts for typed TNG atlas rendering."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from scipy.signal import fftconvolve

from euclid_polish.config import Config
from euclid_polish.image import Image, Role
from euclid_polish.photometry import (
    mjy_per_sr_to_electrons_factor,
    pixel_solid_angle_sr,
    uJy_to_electrons,
)
from euclid_polish.tng import (
    TNGAtlas,
    TNGGalaxy,
    TNGPropertyCatalog,
    TNGRadiusManifest,
)
from euclid_polish.tng._image import _load_tng_plane
from euclid_polish.tng.renderer import (
    TNG_RADIUS_RENDERING,
    TNGRenderer,
    _circularize_psf_kernel,
)
from euclid_polish.tng.types import (
    PhysicalRedshiftGeometry,
    RenderedTNG,
    TNGView,
    VIS2FWHMNormalization,
)


def _write_frame(path: Path, data: np.ndarray) -> None:
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=">f4"))
    hdu.header["BUNIT"] = "MJy/sr"
    hdu.header["CDELT1"] = 100.0
    hdu.header["CUNIT1"] = "pc"
    hdu.header["CDELT2"] = 100.0
    hdu.header["CUNIT2"] = "pc"
    hdu.writeto(path, overwrite=True)


def _write_fake_galaxy(
    root: Path,
    subhalo_id: str,
    *,
    size: int = 64,
    done: bool = True,
) -> Path:
    directory = root / subhalo_id
    directory.mkdir(parents=True, exist_ok=True)
    centre = size // 2
    base = np.zeros((size, size), dtype=np.float32)
    base[centre - 8 : centre + 8, centre - 2 : centre + 2] = 500.0
    base[centre - 2 : centre + 10, centre - 6 : centre + 6] += 200.0
    for orientation in range(1, 6):
        for index, band in enumerate(("VIS", "Y", "J", "H"), start=1):
            _write_frame(
                directory
                / f"TNG{subhalo_id}_O{orientation}_Euclid_{band}.fits",
                base * index,
            )
    if done:
        (directory / Config.Tng.DONE_MARKER).touch()
    return directory


def _view(directory: Path, *, native_re_px: float = 20.0) -> TNGView:
    return TNGView(
        galaxy_dir=directory,
        subhalo_id=directory.name,
        orientation=1,
        native_re_px=native_re_px,
        radius_manifest_fingerprint="fixture-manifest",
    )


def test_surface_brightness_conversion_matches_ujy_route():
    solid_angle = pixel_solid_angle_sr(0.05)
    expected = uJy_to_electrons(1.0e12 * solid_angle, Config.BAND_VIS)
    assert mjy_per_sr_to_electrons_factor(
        Config.BAND_VIS, 0.05
    ) == pytest.approx(expected, rel=1e-9)
    assert mjy_per_sr_to_electrons_factor(
        Config.BAND_VIS, 0.10
    ) == pytest.approx(4.0 * expected, rel=1e-9)


def test_tng_plane_loader_rejects_invalid_units_grids_and_shape(tmp_path):
    wrong_unit = tmp_path / "wrong_unit.fits"
    _write_frame(wrong_unit, np.ones((4, 4), dtype=np.float32))
    with fits.open(wrong_unit, mode="update") as hdul:
        hdul[0].header["BUNIT"] = "Jy/sr"
    with pytest.raises(ValueError, match="BUNIT must be"):
        _load_tng_plane(wrong_unit, "VIS")

    angular_grid = tmp_path / "angular_grid.fits"
    _write_frame(angular_grid, np.ones((4, 4), dtype=np.float32))
    with fits.open(angular_grid, mode="update") as hdul:
        hdul[0].header["CUNIT1"] = "arcsec"
        hdul[0].header["CUNIT2"] = "arcsec"
    with pytest.raises(ValueError, match="physical pixel scale"):
        _load_tng_plane(angular_grid, "VIS")

    rectangular_grid = tmp_path / "rectangular_grid.fits"
    _write_frame(rectangular_grid, np.ones((4, 4), dtype=np.float32))
    with fits.open(rectangular_grid, mode="update") as hdul:
        hdul[0].header["CDELT2"] = 200.0
    with pytest.raises(ValueError, match="must be square"):
        _load_tng_plane(rectangular_grid, "VIS")

    cube = tmp_path / "cube.fits"
    _write_frame(cube, np.ones((2, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="two-dimensional"):
        _load_tng_plane(cube, "VIS")


def test_tng_galaxy_discovers_complete_inventory_and_resolves_paths(tmp_path):
    _write_fake_galaxy(tmp_path, "111")
    _write_fake_galaxy(tmp_path, "222")
    _write_fake_galaxy(tmp_path, "333", done=False)

    galaxies = TNGGalaxy.discover(tmp_path)

    assert [galaxy.subhalo_id for galaxy in galaxies] == ["111", "222"]
    assert galaxies[0].fits_path(1, "VIS").is_file()
    assert TNGGalaxy.discover(tmp_path / "missing") == ()


def test_atlas_filters_orientations_that_would_enlarge(tmp_path):
    directory = _write_fake_galaxy(tmp_path, "111")
    galaxy = TNGGalaxy(directory=directory, subhalo_id="111")
    radii = {
        ("111", 1): 2.0,
        ("111", 2): 12.0,
        ("111", 3): 12.0,
        ("111", 4): 12.0,
        ("111", 5): 12.0,
    }
    atlas = TNGAtlas(
        root=tmp_path,
        galaxies=(galaxy,),
        properties=TNGPropertyCatalog({}, (None, 0, 0)),
        radii=TNGRadiusManifest(radii, "fixture-manifest"),
    )

    views = atlas.eligible_views(
        galaxy,
        target_re_arcsec=0.5,
        pixel_scale_arcsec=0.05,
    )

    assert tuple(view.orientation for view in views) == (2, 3, 4, 5)
    assert atlas.eligible_galaxies(0.5, 0.05) == (galaxy,)
    assert atlas.eligible_views(galaxy, 0.7, 0.05) == ()


def test_render_observed_radius_returns_clean_image_and_trace(tmp_path):
    directory = _write_fake_galaxy(tmp_path, "111")
    renderer = TNGRenderer(pixel_scale_arcsec=0.05)
    view = _view(directory)

    rendered = renderer.render_observed_radius(
        view, 0.5, rng=np.random.default_rng(8)
    )
    fields = rendered.record_fields()

    assert isinstance(rendered, RenderedTNG)
    assert isinstance(rendered.image, Image)
    assert rendered.image.is_clean is True
    assert rendered.image.role is Role.CLEAN
    assert rendered.pixel_scale_arcsec == pytest.approx(0.05)
    assert rendered.bands == tuple(Config.LR_INPUT_BAND_NAMES)
    assert rendered.data.dtype == np.float32
    assert np.isfinite(rendered.data).all() and (rendered.data >= 0.0).all()
    assert fields["target_re_arcsec"] == pytest.approx(0.5)
    assert fields["radius_scale_factor"] == pytest.approx(0.5)
    assert fields["radius_rendering"] == TNG_RADIUS_RENDERING


def test_render_observed_radius_rejects_enlargement(tmp_path):
    directory = _write_fake_galaxy(tmp_path, "111")
    renderer = TNGRenderer(pixel_scale_arcsec=0.05)

    with pytest.raises(ValueError, match="cannot be enlarged"):
        renderer.render_observed_radius(_view(directory, native_re_px=2.0), 0.5)


def test_render_observed_radius_normalizes_only_after_geometry(tmp_path):
    directory = _write_fake_galaxy(tmp_path, "111")
    renderer = TNGRenderer(pixel_scale_arcsec=0.05)
    view = _view(directory)
    unscaled = renderer.render_observed_radius(
        view, 0.5, rng=np.random.default_rng(8)
    )
    target_vis = 2.5 * unscaled.flux_e("VIS")

    normalized = renderer.render_observed_radius(
        view,
        0.5,
        rng=np.random.default_rng(8),
        target_vis_flux_e=target_vis,
    )

    assert normalized.shape == unscaled.shape
    assert normalized.flux_e("VIS") == pytest.approx(target_vis, rel=2e-6)
    assert normalized.trace.geometry == unscaled.trace.geometry
    assert normalized.record_fields()["brightness_scale"] == pytest.approx(2.5)
    assert normalized.record_fields()["photometric_scaling"] == (
        "single_shared_total_vis_anchor"
    )


def test_render_observed_radius_bounds_support_and_preserves_centre_parity(tmp_path):
    directory = _write_fake_galaxy(tmp_path, "111", size=96)
    renderer = TNGRenderer(pixel_scale_arcsec=0.05, max_output_side=15)

    rendered = renderer.render_observed_radius(_view(directory), 0.9)

    assert rendered.shape[0] <= 15 and rendered.shape[1] <= 15
    assert rendered.trace.render_support_clipped is True


def test_native_source_and_photometry_are_cached_by_renderer(tmp_path):
    directory = _write_fake_galaxy(tmp_path, "111")
    renderer = TNGRenderer(pixel_scale_arcsec=0.05)
    view = _view(directory)

    first = renderer.native_photometry(view)
    second = renderer.native_photometry(view)
    info = renderer.cache_info()

    assert second is first
    assert info["source_entries"] == 1
    assert info["native_photometry_entries"] == 1
    assert info["source_bytes"] > 0
    renderer.clear_caches()
    assert renderer.cache_info()["source_entries"] == 0


def test_physical_redshift_render_supports_typed_total_vis_normalization(tmp_path):
    directory = _write_fake_galaxy(tmp_path, "111", size=96)
    renderer = TNGRenderer(pixel_scale_arcsec=0.05)
    view = _view(directory)
    native = renderer.render_physical_at_redshift(
        view,
        0.8,
        rng=np.random.default_rng(9),
        surface_brightness_cut_mag_arcsec2=0.0,
    )
    target_vis = 2.5 * native.flux_e("VIS")
    normalized = native.normalised_to_total_vis(target_vis)

    assert isinstance(native.trace.geometry, PhysicalRedshiftGeometry)
    assert native.trace.redshift is not None
    assert normalized.flux_e("VIS") == pytest.approx(target_vis, rel=2e-6)
    np.testing.assert_allclose(
        np.asarray(normalized.fluxes_e) / normalized.flux_e("VIS"),
        np.asarray(native.fluxes_e) / native.flux_e("VIS"),
        rtol=2e-6,
    )


def test_nominal_redshift_render_records_typed_photometry(tmp_path):
    directory = _write_fake_galaxy(tmp_path, "111")
    renderer = TNGRenderer(pixel_scale_arcsec=0.05)

    rendered = renderer.render_observed_radius_at_redshift(
        _view(directory), 0.5, 1.0, rng=np.random.default_rng(10)
    )

    assert rendered.trace.redshift is not None
    assert rendered.trace.redshift.redshift == pytest.approx(1.0)
    assert rendered.record_fields()["dimming"] == pytest.approx(0.125)


def test_2fwhm_normalization_uses_one_scale_and_preserves_colours(tmp_path):
    directory = _write_fake_galaxy(tmp_path, "111", size=128)
    renderer = TNGRenderer(pixel_scale_arcsec=0.05)
    rendered = renderer.render_observed_radius(_view(directory), 0.8)
    original_ratios = np.asarray(rendered.fluxes_e) / rendered.flux_e("VIS")
    yy, xx = np.indices((17, 17), dtype=np.float64)
    psf = np.exp(-0.5 * ((yy - 8.0) ** 2 + (xx - 8.0) ** 2) / 2.0**2)

    normalized = renderer.normalize_vis_2fwhm(
        rendered,
        target_flux_e=1234.5,
        psf_kernel=psf.astype(np.float32),
        psf_fwhm_arcsec=0.2,
        psf_identity="fixture-psf",
    )
    fields = normalized.record_fields()

    assert isinstance(normalized.trace.normalization, VIS2FWHMNormalization)
    assert fields["target_vis_2fwhm_flux_e"] == pytest.approx(1234.5)
    assert fields["achieved_vis_2fwhm_flux_e"] == pytest.approx(1234.5)
    np.testing.assert_allclose(
        np.asarray(normalized.fluxes_e) / normalized.flux_e("VIS"),
        original_ratios,
        rtol=2e-6,
    )


@pytest.mark.parametrize(
    ("image_shape", "psf_shape"),
    [((81, 81), (9, 9)), ((80, 80), (8, 8)), ((80, 81), (7, 10))],
)
def test_compact_aperture_response_matches_full_fft(image_shape, psf_shape):
    rng = np.random.default_rng(81)
    vis = rng.random(image_shape, dtype=np.float32)
    psf = _circularize_psf_kernel(rng.random(psf_shape, dtype=np.float32))
    blurred = fftconvolve(vis.astype(np.float64), psf.astype(np.float64), mode="same")
    yy, xx = np.indices(image_shape, dtype=np.float64)
    centre_y = 0.5 * (image_shape[0] - 1)
    centre_x = 0.5 * (image_shape[1] - 1)
    expected = float(
        np.sum(
            blurred[np.hypot(yy - centre_y, xx - centre_x) <= 4.0],
            dtype=np.float64,
        )
    )

    actual = TNGRenderer(
        pixel_scale_arcsec=0.05
    )._measure_vis_2fwhm_aperture_flux(
        vis,
        circular_psf=psf,
        psf_fwhm_arcsec=0.2,
        psf_identity="fixture",
    )

    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_psf_caches_are_instance_owned_and_parity_sensitive(tmp_path):
    directory = _write_fake_galaxy(tmp_path, "111", size=128)
    renderer = TNGRenderer(pixel_scale_arcsec=0.05)
    stamp = renderer.render_observed_radius(_view(directory), 0.8)
    psf = np.ones((7, 7), dtype=np.float32)

    renderer.normalize_vis_2fwhm(
        stamp,
        target_flux_e=10.0,
        psf_kernel=psf,
        psf_fwhm_arcsec=0.2,
        psf_identity="empirical-vis:3",
    )
    renderer.normalize_vis_2fwhm(
        RenderedTNG(
            image=replace(
                stamp.image,
                data=stamp.data[:-1, :-1, :],
            ),
            trace=stamp.trace,
        ),
        target_flux_e=10.0,
        psf_kernel=psf,
        psf_fwhm_arcsec=0.2,
        psf_identity="empirical-vis:3",
    )
    info = renderer.cache_info()

    assert info["circular_psf_entries"] == 1
    assert info["aperture_entries"] == 2
    assert info["circular_psf_bytes"] > 0
    assert info["aperture_bytes"] > 0


def test_invalid_pixel_scale_is_rejected():
    with pytest.raises(ValueError, match="pixel_scale_arcsec"):
        TNGRenderer(pixel_scale_arcsec=float("nan"))
    with pytest.raises(ValueError, match="pixel_scale_arcsec"):
        TNGRenderer(pixel_scale_arcsec=0.0)


def test_pixel_solid_angle_closed_form():
    scale_rad = 0.05 * math.pi / 180.0 / 3600.0
    assert pixel_solid_angle_sr(0.05) == pytest.approx(scale_rad**2, rel=1e-12)
