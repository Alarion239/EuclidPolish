"""Contracts for native TNG MJy/sr images and their private resampling policy."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from euclid_polish.tng._image import (
    _block_mean,
    _downsample_surface_brightness,
    _measure_halflight_radius_px,
    _rotate_surface_brightness,
)
from euclid_polish.tng.image import TNGSurfaceBrightnessImage


def _image(
    data: np.ndarray | None = None,
    *,
    bands: tuple[str, ...] = ("VIS", "Y_E"),
    pixel_scale_pc: float = 100.0,
) -> TNGSurfaceBrightnessImage:
    if data is None:
        data = np.arange(48, dtype=np.float32).reshape(4, 6, 2)
    return TNGSurfaceBrightnessImage(
        data=data,
        bands=bands,
        pixel_scale_pc=pixel_scale_pc,
    )


def test_tng_image_package_import_does_not_load_tensorflow():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from euclid_polish.tng import TNGSurfaceBrightnessImage; "
            "assert 'tensorflow' not in sys.modules; "
            "assert TNGSurfaceBrightnessImage.__name__ == "
            "'TNGSurfaceBrightnessImage'",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_tng_image_owns_canonical_read_only_hwc_float32_data():
    source = np.arange(48, dtype=np.float64).reshape(6, 4, 2).transpose(1, 0, 2)
    image = _image(source)

    assert image.shape == (4, 6, 2)
    assert image.spatial_shape == (4, 6)
    assert image.num_channels == 2
    assert image.data.dtype == np.float32
    assert image.data.flags.c_contiguous
    assert not image.data.flags.writeable
    assert image.bands == ("VIS", "Y_E")
    assert image.pixel_scale_pc == pytest.approx(100.0)

    source[...] = -1.0
    assert np.all(image.data >= 0.0)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"data": np.zeros((2, 2))}, "shape"),
        ({"data": np.zeros((0, 2, 2))}, "non-empty"),
        ({"data": np.full((2, 2, 2), np.nan)}, "finite"),
        ({"bands": ("VIS",)}, "channels"),
        ({"bands": ("VIS", "VIS")}, "unique"),
        ({"bands": ("VIS", " ")}, "non-empty"),
        ({"pixel_scale_pc": 0.0}, "finite and positive"),
        ({"pixel_scale_pc": np.nan}, "finite and positive"),
    ],
)
def test_tng_image_rejects_invalid_state(kwargs, error):
    values = {
        "data": np.zeros((2, 2, 2), dtype=np.float32),
        "bands": ("VIS", "Y_E"),
        "pixel_scale_pc": 100.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=error):
        TNGSurfaceBrightnessImage(**values)


def test_tng_image_repr_and_equality_are_array_safe():
    first = _image()
    equal = _image()
    changed = first.with_data(first.data + 1.0)

    assert first == equal
    assert first != changed
    assert first != object()
    text = repr(first)
    assert "shape=(4, 6, 2)" in text
    assert "pixel_scale_pc=100.0" in text
    assert "[[[" not in text


def test_tng_image_band_lookup_plane_and_array_access():
    image = _image()

    assert image.band_index("Y_E") == 1
    np.testing.assert_array_equal(image.plane("VIS"), image.data[..., 0])
    assert not image.plane("VIS").flags.writeable
    assert image.as_array() is image.data
    copied = image.as_array(copy=True)
    assert copied.flags.writeable
    assert not np.shares_memory(copied, image.data)
    with pytest.raises(ValueError, match="not in"):
        image.band_index("H_E")
    with pytest.raises(ValueError, match="band is required"):
        image.plane()

    one_band = _image(image.data[..., :1], bands=("VIS",))
    np.testing.assert_array_equal(one_band.plane(), image.data[..., 0])


def test_tng_image_crop_rotation_and_replacement_preserve_domain_metadata():
    image = _image()

    cropped = image.cropped(slice(1, 4), slice(2, 5))
    quarter = image.rotated_quarter(1)
    replaced = image.with_data(
        np.ones_like(image.data),
        pixel_scale_pc=200.0,
    )

    np.testing.assert_array_equal(cropped.data, image.data[1:4, 2:5, :])
    np.testing.assert_array_equal(
        quarter.data,
        np.rot90(image.data, axes=(0, 1)),
    )
    assert cropped.bands == quarter.bands == image.bands
    assert cropped.pixel_scale_pc == quarter.pixel_scale_pc == 100.0
    assert replaced.pixel_scale_pc == 200.0
    assert replaced.bands == image.bands
    assert not np.shares_memory(replaced.data, image.data)
    with pytest.raises(TypeError, match="must be slices"):
        image.cropped(0, slice(None))  # type: ignore[arg-type]


def test_tng_image_stack_combines_only_compatible_native_images():
    vis = _image(
        np.ones((2, 3, 1), dtype=np.float32),
        bands=("VIS",),
    )
    nisp = _image(
        np.full((2, 3, 2), 2.0, dtype=np.float32),
        bands=("Y_E", "J_E"),
    )
    stacked = TNGSurfaceBrightnessImage.stack((vis, nisp))

    assert stacked.shape == (2, 3, 3)
    assert stacked.bands == ("VIS", "Y_E", "J_E")
    np.testing.assert_array_equal(stacked.plane("VIS"), vis.plane())
    np.testing.assert_array_equal(stacked.data[..., 1:], nisp.data)
    with pytest.raises(ValueError, match="at least one"):
        TNGSurfaceBrightnessImage.stack(())
    with pytest.raises(ValueError, match="spatial shape"):
        TNGSurfaceBrightnessImage.stack(
            (vis, nisp.cropped(slice(0, 1), slice(0, 1)))
        )
    with pytest.raises(ValueError, match="pixel scale"):
        TNGSurfaceBrightnessImage.stack(
            (vis, nisp.with_data(nisp.data, pixel_scale_pc=200.0))
        )


def test_tng_resampling_updates_physical_sampling_not_surface_brightness():
    source = _image(
        np.full((4, 4, 1), 7.0, dtype=np.float32),
        bands=("VIS",),
    )

    rebinned = _block_mean(source, 2)
    downsampled = _downsample_surface_brightness(source, 0.5)

    assert rebinned.shape == downsampled.shape == (2, 2, 1)
    assert rebinned.pixel_scale_pc == downsampled.pixel_scale_pc == 200.0
    np.testing.assert_allclose(rebinned.data, 7.0)
    np.testing.assert_allclose(downsampled.data, 7.0)
    with pytest.raises(ValueError, match="cannot be enlarged"):
        _downsample_surface_brightness(source, 1.01)


def test_tng_arbitrary_rotation_and_half_light_measurement_keep_native_type():
    data = np.zeros((9, 9, 1), dtype=np.float32)
    data[2:7, 3:6, 0] = 1.0
    source = _image(data, bands=("VIS",))

    rotated = _rotate_surface_brightness(source, 17.0)

    assert isinstance(rotated, TNGSurfaceBrightnessImage)
    assert rotated.shape == source.shape
    assert rotated.bands == source.bands
    assert rotated.pixel_scale_pc == source.pixel_scale_pc
    assert np.all(rotated.data >= 0.0)
    assert np.isfinite(_measure_halflight_radius_px(source))
