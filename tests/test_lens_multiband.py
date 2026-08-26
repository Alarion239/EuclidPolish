"""Tests for the four-band TNG lens renderer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.image import Image, Role
from euclid_polish.sky.generation.lens_population import (
    render_lens_to_multiband_canvas,
    sample_lens_geometry,
)
from euclid_polish.tng.types import (
    NominalRadiusGeometry,
    RenderedTNG,
    TNGRenderTrace,
    TNGRotation,
    TNGView,
)


def _rendered_stamp(data: np.ndarray) -> RenderedTNG:
    pixel_scale = Config.DEFAULT_PIXEL_SCALE
    native_re_px = 20.0
    scale_factor = 0.5
    image = Image(
        data=data,
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


@pytest.fixture
def lens_case():
    params = sample_lens_geometry(np.random.default_rng(7), 250.0)
    assert params is not None
    params = replace(params, centre_x_pix=64.0, centre_y_pix=64.0)
    lens_data = np.zeros((48, 48, Config.NUM_LR_CHANNELS), np.float32)
    source_data = np.zeros_like(lens_data)
    lens_data[18:30, 18:30, :] = np.arange(1, 5, dtype=np.float32) * 20.0
    source_data[20:28, 20:28, :] = np.arange(1, 5, dtype=np.float32) * 40.0
    return (
        params,
        _rendered_stamp(lens_data),
        _rendered_stamp(source_data),
    )


def _render(lens_case) -> np.ndarray:
    params, lens, source = lens_case
    canvas = np.zeros((128, 128, Config.NUM_LR_CHANNELS), np.float32)
    return render_lens_to_multiband_canvas(
        canvas,
        params=params,
        lens_light_stamp=lens,
        source_stamp=source,
    )


def test_multiband_render_adds_flux_to_every_channel(lens_case):
    canvas = _render(lens_case)
    assert np.all(canvas.sum(axis=(0, 1)) > 0.0)


def test_multiband_render_is_additive(lens_case):
    once = _render(lens_case)
    params, lens, source = lens_case
    twice = np.zeros_like(once)
    for _ in range(2):
        render_lens_to_multiband_canvas(
            twice,
            params=params,
            lens_light_stamp=lens,
            source_stamp=source,
        )
    np.testing.assert_allclose(twice, 2.0 * once, rtol=1e-5, atol=1e-4)


def test_multiband_render_has_extended_ring_structure(lens_case):
    params, _, _ = lens_case
    canvas = _render(lens_case)
    yy, xx = np.indices(canvas.shape[:2])
    radius_arcsec = np.hypot(yy - 64.0, xx - 64.0) * Config.DEFAULT_PIXEL_SCALE
    ring = (
        (radius_arcsec >= 0.8 * params.theta_E_arcsec)
        & (radius_arcsec <= 1.5 * params.theta_E_arcsec)
    )
    assert np.all(canvas[ring].sum(axis=0) > 0.0)
