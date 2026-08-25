from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from euclid_polish.image import (
    AngularGrid,
    CubeLike,
    ImageCube,
    PhysicalGrid,
    PixelUnit,
)


def _cube(data=None, *, bands=("VIS", "Y_E"), grid=None):
    if data is None:
        data = np.arange(48, dtype=np.float32).reshape(4, 6, 2)
    return ImageCube(
        data=data,
        bands=bands,
        unit=PixelUnit.MJY_PER_SR,
        grid=grid or PhysicalGrid(100.0),
    )


def test_cube_module_import_does_not_load_tensorflow():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from euclid_polish.image import ImageCube; "
            "assert 'tensorflow' not in sys.modules; assert ImageCube.__name__ == 'ImageCube'",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("grid", [PhysicalGrid(100.0), AngularGrid(0.05)])
def test_cube_canonicalises_owned_hwc_float32_data(grid):
    source = np.arange(48, dtype=np.float64).reshape(6, 4, 2).transpose(1, 0, 2)
    cube = _cube(source, grid=grid)

    assert cube.shape == (4, 6, 2)
    assert cube.spatial_shape == (4, 6)
    assert cube.num_channels == 2
    assert cube.data.dtype == np.float32
    assert cube.data.flags.c_contiguous
    assert cube.bands == ("VIS", "Y_E")

    source[...] = -1.0
    assert np.all(cube.data >= 0.0)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"data": np.zeros((2, 2))}, "shape"),
        ({"data": np.zeros((0, 2, 2))}, "non-empty"),
        ({"data": np.full((2, 2, 2), np.nan)}, "finite"),
        ({"bands": ("VIS",)}, "channels"),
        ({"bands": ("VIS", "VIS")}, "unique"),
        ({"bands": ("VIS", " ")}, "non-empty"),
    ],
)
def test_cube_rejects_invalid_data_and_bands(kwargs, error):
    values = {
        "data": np.zeros((2, 2, 2), dtype=np.float32),
        "bands": ("VIS", "Y_E"),
        "unit": PixelUnit.MJY_PER_SR,
        "grid": PhysicalGrid(100.0),
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=error):
        ImageCube(**values)


@pytest.mark.parametrize("grid_type", [PhysicalGrid, AngularGrid])
@pytest.mark.parametrize("scale", [0.0, -1.0, np.inf, np.nan])
def test_grid_rejects_nonpositive_or_nonfinite_scale(grid_type, scale):
    with pytest.raises(ValueError, match="finite and positive"):
        grid_type(scale)


def test_cube_rejects_unknown_unit_and_grid():
    with pytest.raises(ValueError, match="unsupported pixel unit"):
        ImageCube(np.zeros((1, 1, 1)), ("VIS",), "counts", AngularGrid(0.1))
    with pytest.raises(TypeError, match="PhysicalGrid or AngularGrid"):
        ImageCube(np.zeros((1, 1, 1)), ("VIS",), PixelUnit.MJY_PER_SR, object())


def test_repr_and_equality_are_array_safe():
    first = _cube()
    equal = _cube()
    changed = first.with_data(first.data + 1.0)

    assert first == equal
    assert first != changed
    assert first != object()
    text = repr(first)
    assert "shape=(4, 6, 2)" in text
    assert "MJy/sr" in text
    assert "[[[" not in text


def test_band_lookup_and_plane_access():
    cube = _cube()

    assert cube.band_index("Y_E") == 1
    assert np.array_equal(cube.plane("VIS"), cube.data[..., 0])
    with pytest.raises(ValueError, match="not in"):
        cube.band_index("H_E")
    with pytest.raises(ValueError, match="band is required"):
        cube.plane()

    one_band = ImageCube(
        cube.data[..., :1], ("VIS",), cube.unit, cube.grid,
    )
    assert np.array_equal(one_band.plane(), cube.data[..., 0])
    assert not cube.data.flags.writeable
    assert not cube.plane("VIS").flags.writeable
    assert cube.as_array(copy=True).flags.writeable


def test_crop_and_center_crop_preserve_semantics():
    cube = _cube()
    cropped = cube.cropped(slice(1, 4), slice(2, 5))
    centered = cube.center_cropped(2, 4)

    assert np.array_equal(cropped.data, cube.data[1:4, 2:5, :])
    assert np.array_equal(centered.data, cube.data[1:3, 1:5, :])
    assert cropped.bands == cube.bands
    assert cropped.unit == cube.unit
    assert cropped.grid == cube.grid
    assert cube.center_cropped(100, 100) == cube
    with pytest.raises(ValueError, match="positive"):
        cube.center_cropped(0)


def test_quarter_rotation_is_exact_and_arbitrary_rotation_has_same_shape():
    cube = _cube()

    quarter = cube.rotated_quarter(1)
    assert np.array_equal(quarter.data, np.rot90(cube.data, axes=(0, 1)))
    assert cube.rotated_quarter(4) == cube

    arbitrary = cube.rotated(17.0)
    assert arbitrary.shape == cube.shape
    assert arbitrary.bands == cube.bands
    assert arbitrary.unit == cube.unit
    assert arbitrary.grid == cube.grid


def test_arbitrary_rotation_direction_agrees_with_nearby_np_rot90():
    data = np.zeros((9, 9, 1), dtype=np.float32)
    data[2, 6, 0] = 1.0  # asymmetric marker in the upper-right quadrant
    cube = ImageCube(
        data, ("VIS",), PixelUnit.ELECTRONS_PER_PIXEL, AngularGrid(0.05),
    )

    counter_clockwise = cube.rotated(89.0).plane()
    clockwise = cube.rotated(-89.0).plane()
    ccw_peak = np.unravel_index(np.argmax(counter_clockwise), counter_clockwise.shape)
    cw_peak = np.unravel_index(np.argmax(clockwise), clockwise.shape)
    exact_peak = np.unravel_index(
        np.argmax(np.rot90(data[..., 0])), data.shape[:2],
    )

    assert ccw_peak == exact_peak
    assert ccw_peak[0] < 4 and ccw_peak[1] < 4
    assert cw_peak[0] > 4 and cw_peak[1] > 4


def test_as_array_diagnostics_and_with_data_grid_override():
    cube = _cube()

    assert cube.as_array() is cube.data
    copied = cube.as_array(copy=True)
    assert np.array_equal(copied, cube.data)
    assert copied is not cube.data

    diagnostics = cube.diagnostics()
    assert diagnostics["shape"] == cube.shape
    assert diagnostics["bands"] == cube.bands
    assert diagnostics["unit"] == "MJy/sr"
    assert diagnostics["grid"] == {
        "kind": "physical",
        "pixel_scale_pc": 100.0,
    }

    replacement_grid = PhysicalGrid(200.0)
    replaced = cube.with_data(np.ones_like(cube.data), grid=replacement_grid)
    assert replaced.grid == replacement_grid
    assert replaced.unit == cube.unit
    assert replaced.bands == cube.bands


def test_stack_combines_channels_and_checks_semantics():
    vis = ImageCube(
        np.ones((2, 3, 1)), ("VIS",), PixelUnit.MJY_PER_SR, PhysicalGrid(100.0),
    )
    nisp = ImageCube(
        np.full((2, 3, 2), 2.0), ("Y_E", "J_E"),
        PixelUnit.MJY_PER_SR, PhysicalGrid(100.0),
    )
    stacked = ImageCube.stack((vis, nisp))

    assert stacked.shape == (2, 3, 3)
    assert stacked.bands == ("VIS", "Y_E", "J_E")
    assert np.array_equal(stacked.plane("VIS"), vis.plane())
    assert np.array_equal(stacked.data[..., 1:], nisp.data)

    with pytest.raises(ValueError, match="at least one"):
        ImageCube.stack(())
    with pytest.raises(ValueError, match="spatial shape"):
        ImageCube.stack((vis, nisp.center_cropped(1, 1)))
    with pytest.raises(ValueError, match="unit"):
        ImageCube.stack((
            vis,
            ImageCube(
                nisp.data, nisp.bands, PixelUnit.ELECTRONS_PER_PIXEL, nisp.grid,
            ),
        ))


def test_existing_image_satisfies_cube_protocol():
    from euclid_polish.image import Image

    image = Image(
        data=np.ones((2, 3, 1), dtype=np.float32),
        pixel_scale_arcsec=0.05,
        band_names=("VIS",),
        is_clean=True,
    )

    assert isinstance(image, CubeLike)
    assert image.bands == image.band_names
    assert image.spatial_shape == (2, 3)
    assert image.unit is PixelUnit.ELECTRONS_PER_PIXEL
    assert image.grid == AngularGrid(0.05)

    with pytest.raises(ValueError, match="finite and positive"):
        Image(
            data=np.ones((2, 3, 1), dtype=np.float32),
            pixel_scale_arcsec=-0.05,
            band_names=("VIS",),
            is_clean=True,
        )
    with pytest.raises(ValueError, match="unique"):
        Image(
            data=np.ones((2, 3, 2), dtype=np.float32),
            pixel_scale_arcsec=0.05,
            band_names=("VIS", "VIS"),
            is_clean=True,
        )
