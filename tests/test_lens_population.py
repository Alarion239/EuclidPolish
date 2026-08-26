"""Tests for the active TNG-backed strong-lens geometry and renderer."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.image import Image, Role
from euclid_polish.sky.generation.compositing import composite_stamp
from euclid_polish.sky.generation.lens_population import (
    LensParams,
    einstein_radius_sis,
    render_lens_to_multiband_canvas,
    sample_lens_geometry,
)
from euclid_polish.tng.renderer import TNGRenderer
from euclid_polish.tng.types import (
    NominalRadiusGeometry,
    RenderedTNG,
    TNGRenderTrace,
    TNGRotation,
    TNGView,
)


def test_einstein_radius_monotonic_in_sigma_v():
    theta_low = einstein_radius_sis(150.0, 0.5, 2.0)
    theta_high = einstein_radius_sis(300.0, 0.5, 2.0)
    assert theta_high > theta_low > 0.0
    assert theta_high / theta_low == pytest.approx(4.0, rel=1e-3)


def test_einstein_radius_zero_when_source_below_lens():
    assert einstein_radius_sis(250.0, 1.0, 1.0) == 0.0
    assert einstein_radius_sis(250.0, 1.0, 0.5) == 0.0


def test_einstein_radius_realistic_magnitude():
    theta = einstein_radius_sis(250.0, 0.5, 2.0)
    assert 0.8 < theta < 1.5


def test_sample_lens_geometry_priors_and_reproducibility():
    first = sample_lens_geometry(np.random.default_rng(11), 250.0)
    second = sample_lens_geometry(np.random.default_rng(11), 250.0)
    assert isinstance(first, LensParams)
    assert first == second
    assert Config.LENS_Z_LENS_MIN <= first.z_lens <= Config.LENS_Z_LENS_MAX
    assert (
        first.z_lens + Config.LENS_Z_SOURCE_OFFSET
        <= first.z_source
        <= Config.LENS_Z_SOURCE_MAX
    )
    assert 0.10 < first.theta_E_arcsec < 3.5
    assert Config.LENS_AXIS_RATIO_MIN <= first.lens_q <= Config.LENS_AXIS_RATIO_MAX
    offset = math.hypot(first.src_dx_arcsec, first.src_dy_arcsec)
    assert offset <= Config.LENS_SOURCE_OFFSET_FRAC * first.theta_E_arcsec


def _rendered_stamp(
    data: np.ndarray,
    *,
    pixel_scale: float = Config.DEFAULT_PIXEL_SCALE,
) -> RenderedTNG:
    native_re_px = 20.0
    scale_factor = 0.5
    image = Image(
        data=np.asarray(data, dtype=np.float32),
        pixel_scale_arcsec=pixel_scale,
        band_names=tuple(Config.LR_INPUT_BAND_NAMES),
        is_clean=True,
        role=Role.CLEAN,
    )
    trace = TNGRenderTrace(
        view=TNGView(Path("unused"), "fixture", 1, native_re_px),
        rotation=TNGRotation(),
        geometry=NominalRadiusGeometry(
            target_re_arcsec=native_re_px * pixel_scale * scale_factor,
            scale_factor=scale_factor,
            radius_rendering="test_shrink_only",
            radius_renderer_fingerprint="test-fixture",
        ),
    )
    return RenderedTNG(image=image, trace=trace)


def _bright_stamp(
    n: int = 80,
    core: int = 8,
    value: float = 50.0,
    *,
    pixel_scale: float = Config.DEFAULT_PIXEL_SCALE,
) -> RenderedTNG:
    stamp = np.zeros((n, n, Config.NUM_LR_CHANNELS), np.float32)
    centre = n // 2
    stamp[
        centre - core : centre + core,
        centre - core : centre + core,
        :,
    ] = value
    stamp[
        centre + 4 : centre + 4 + core,
        centre + 6 : centre + 6 + core,
        :,
    ] += 2.0 * value
    return _rendered_stamp(stamp, pixel_scale=pixel_scale)


def _write_physical_tng_view(root: Path) -> TNGView:
    directory = root / "111"
    directory.mkdir()
    side = 96
    centre = side // 2
    surface_brightness = np.zeros((side, side), dtype=np.float32)
    surface_brightness[
        centre - 12 : centre + 12,
        centre - 5 : centre + 5,
    ] = 500.0
    surface_brightness[
        centre - 4 : centre + 16,
        centre - 14 : centre + 14,
    ] += 200.0
    for band_index, band in enumerate(("VIS", "Y", "J", "H"), start=1):
        hdu = fits.PrimaryHDU(
            np.asarray(surface_brightness * band_index, dtype=">f4")
        )
        hdu.header["BUNIT"] = "MJy/sr"
        hdu.header["CDELT1"] = 100.0
        hdu.header["CUNIT1"] = "pc"
        hdu.header["CDELT2"] = 100.0
        hdu.header["CUNIT2"] = "pc"
        hdu.writeto(directory / f"TNG111_O1_Euclid_{band}.fits")
    return TNGView(
        galaxy_dir=directory,
        subhalo_id="111",
        orientation=1,
        native_re_px=20.0,
        radius_manifest_fingerprint="fixture-manifest",
    )


def test_rendered_tng_source_stamp_is_lensed_and_magnified():
    params = sample_lens_geometry(np.random.default_rng(7), 250.0)
    assert params is not None
    params = replace(
        params,
        centre_x_pix=64.0,
        centre_y_pix=64.0,
        src_dx_arcsec=0.0,
        src_dy_arcsec=0.0,
    )
    lens = _bright_stamp(core=6, value=30.0)
    source = _bright_stamp()
    canvas = np.zeros((128, 128, Config.NUM_LR_CHANNELS), np.float32)
    render_lens_to_multiband_canvas(
        canvas,
        params=params,
        lens_light_stamp=lens,
        source_stamp=source,
    )

    unlensed = np.zeros_like(canvas)
    composite_stamp(unlensed, lens.data, 64.0, 64.0)
    composite_stamp(unlensed, source.data, 64.0, 64.0)
    assert canvas.sum() > 1.05 * unlensed.sum()


def test_physical_tng_renders_flow_into_lens_compositor(tmp_path: Path) -> None:
    pixel_scale = 0.04
    renderer = TNGRenderer(pixel_scale_arcsec=pixel_scale)
    view = _write_physical_tng_view(tmp_path)
    params = sample_lens_geometry(np.random.default_rng(19), 250.0)
    assert params is not None
    params = replace(
        params,
        centre_x_pix=48.0,
        centre_y_pix=48.0,
        src_dx_arcsec=0.0,
        src_dy_arcsec=0.0,
    )
    lens = renderer.render_physical_at_redshift(
        view,
        params.z_lens,
        surface_brightness_cut_mag_arcsec2=99.0,
    )
    source = renderer.render_physical_at_redshift(
        view,
        params.z_source,
        surface_brightness_cut_mag_arcsec2=99.0,
    )
    lens_before = lens.as_array(copy=True)
    source_before = source.as_array(copy=True)
    lens_record = lens.record_fields()
    source_record = source.record_fields()
    canvas = np.zeros((96, 96, Config.NUM_LR_CHANNELS), dtype=np.float32)

    returned = render_lens_to_multiband_canvas(
        canvas,
        params=params,
        pixel_scale=pixel_scale,
        lens_light_stamp=lens,
        source_stamp=source,
    )

    assert returned is canvas
    assert np.all(np.isfinite(canvas))
    assert float(np.sum(canvas, dtype=np.float64)) > 0.0
    np.testing.assert_array_equal(lens.data, lens_before)
    np.testing.assert_array_equal(source.data, source_before)
    assert lens.record_fields() == lens_record
    assert source.record_fields() == source_record
    assert lens.trace.redshift is not None
    assert source.trace.redshift is not None
    assert lens.trace.redshift.redshift == pytest.approx(params.z_lens)
    assert source.trace.redshift.redshift == pytest.approx(params.z_source)


def test_lens_renderer_rejects_mismatched_pixel_scale() -> None:
    params = sample_lens_geometry(np.random.default_rng(7), 250.0)
    assert params is not None
    stamp = _bright_stamp(n=8, core=2, pixel_scale=0.1)
    compatible = _bright_stamp(n=8, core=2)
    canvas = np.zeros((16, 16, Config.NUM_LR_CHANNELS), np.float32)
    with pytest.raises(ValueError, match="does not match canvas pixel scale"):
        render_lens_to_multiband_canvas(
            canvas,
            params=params,
            lens_light_stamp=stamp,
            source_stamp=compatible,
        )


def test_lens_renderer_rejects_raw_array_stamps() -> None:
    params = sample_lens_geometry(np.random.default_rng(7), 250.0)
    assert params is not None
    raw = np.ones((8, 8, Config.NUM_LR_CHANNELS), np.float32)
    canvas = np.zeros((16, 16, Config.NUM_LR_CHANNELS), np.float32)
    with pytest.raises(TypeError, match="RenderedTNG"):
        render_lens_to_multiband_canvas(
            canvas,
            params=params,
            lens_light_stamp=cast(RenderedTNG, raw),
            source_stamp=_bright_stamp(n=8, core=2),
        )
