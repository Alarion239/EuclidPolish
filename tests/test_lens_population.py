"""Tests for the active TNG-backed strong-lens geometry and renderer."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.image.cube import AngularGrid, ImageCube, PhysicalGrid, PixelUnit
from euclid_polish.sky.generation.compositing import composite_stamp
from euclid_polish.sky.generation.lens_population import (
    LensParams,
    LensStamp,
    einstein_radius_sis,
    render_lens_to_multiband_canvas,
    sample_lens_geometry,
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


def _electron_cube(
    data: np.ndarray,
    *,
    pixel_scale: float = Config.DEFAULT_PIXEL_SCALE,
    bands: tuple[str, ...] = Config.LR_INPUT_BAND_NAMES,
) -> ImageCube:
    return ImageCube(
        data=data,
        bands=bands,
        unit=PixelUnit.ELECTRONS_PER_PIXEL,
        grid=AngularGrid(pixel_scale),
    )


def _bright_stamp(n: int = 80, core: int = 8, value: float = 50.0) -> ImageCube:
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
    return _electron_cube(stamp)


def test_cube_source_stamp_is_lensed_and_magnified():
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


@pytest.mark.parametrize(
    ("stamp", "message"),
    [
        (
            ImageCube(
                data=np.ones((8, 8, Config.NUM_LR_CHANNELS), np.float32),
                bands=Config.LR_INPUT_BAND_NAMES,
                unit=PixelUnit.MJY_PER_SR,
                grid=AngularGrid(Config.DEFAULT_PIXEL_SCALE),
            ),
            "electrons/pixel",
        ),
        (
            ImageCube(
                data=np.ones((8, 8, Config.NUM_LR_CHANNELS), np.float32),
                bands=Config.LR_INPUT_BAND_NAMES,
                unit=PixelUnit.ELECTRONS_PER_PIXEL,
                grid=PhysicalGrid(100.0),
            ),
            "angular grid",
        ),
        (
            _electron_cube(
                np.ones((8, 8, Config.NUM_LR_CHANNELS), np.float32),
                bands=("Y_E", "VIS", "J_E", "H_E"),
            ),
            "bands must be",
        ),
        (
            _electron_cube(
                np.ones((8, 8, Config.NUM_LR_CHANNELS), np.float32),
                pixel_scale=0.1,
            ),
            "does not match canvas pixel scale",
        ),
    ],
)
def test_lens_renderer_rejects_incompatible_stamp_cubes(
    stamp: ImageCube,
    message: str,
) -> None:
    params = sample_lens_geometry(np.random.default_rng(7), 250.0)
    assert params is not None
    compatible = _bright_stamp(n=8, core=2)
    canvas = np.zeros((16, 16, Config.NUM_LR_CHANNELS), np.float32)
    with pytest.raises(ValueError, match=message):
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
    with pytest.raises(TypeError, match="CubeLike image"):
        render_lens_to_multiband_canvas(
            canvas,
            params=params,
            lens_light_stamp=cast(LensStamp, raw),
            source_stamp=_bright_stamp(n=8, core=2),
        )
