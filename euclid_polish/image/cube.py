"""Dependency-light image-cube value types.

This module deliberately depends only on NumPy and the Python standard
library.  It describes pixel units and spatial sampling explicitly without
pulling in TensorFlow, FITS I/O, simulators, or instrument-specific policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np


class PixelUnit(str, Enum):
    """Physical meaning of one stored pixel value."""

    MJY_PER_SR = "MJy/sr"
    ELECTRONS_PER_PIXEL = "electrons/pixel"


def _positive_finite(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return number


@dataclass(frozen=True, slots=True)
class PhysicalGrid:
    """A Cartesian image grid sampled in physical parsecs per pixel."""

    pixel_scale_pc: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pixel_scale_pc",
            _positive_finite(self.pixel_scale_pc, "pixel_scale_pc"),
        )


@dataclass(frozen=True, slots=True)
class AngularGrid:
    """An image grid sampled in angular arcseconds per pixel."""

    pixel_scale_arcsec: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pixel_scale_arcsec",
            _positive_finite(self.pixel_scale_arcsec, "pixel_scale_arcsec"),
        )


Grid = PhysicalGrid | AngularGrid


@runtime_checkable
class CubeLike(Protocol):
    """Small structural interface shared by lightweight cubes and ``Image``."""

    @property
    def data(self) -> np.ndarray: ...

    @property
    def bands(self) -> tuple[str, ...]: ...

    @property
    def unit(self) -> PixelUnit: ...

    @property
    def grid(self) -> Grid: ...

    @property
    def shape(self) -> tuple[int, int, int]: ...

    @property
    def num_channels(self) -> int: ...

    def band_index(self, name: str) -> int: ...

    def plane(self, band: str | None = None) -> np.ndarray: ...


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class ImageCube:
    """Validated ``(height, width, channels)`` float32 image data.

    ``ImageCube`` is a numerical value object rather than a persistence type.
    Construction takes an owned, read-only, C-contiguous float32 copy so
    neither the caller's input nor a returned plane can silently mutate it.
    """

    data: np.ndarray = field(repr=False)
    bands: tuple[str, ...]
    unit: PixelUnit
    grid: Grid

    def __post_init__(self) -> None:
        values = np.array(self.data, dtype=np.float32, order="C", copy=True)
        if values.ndim != 3:
            raise ValueError(
                f"ImageCube.data must have shape (H, W, C), got {values.shape}"
            )
        if any(side <= 0 for side in values.shape):
            raise ValueError(f"ImageCube.data must be non-empty, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("ImageCube.data must contain only finite values")

        bands = tuple(str(name) for name in self.bands)
        if len(bands) != values.shape[-1]:
            raise ValueError(
                f"bands has {len(bands)} entries but data has "
                f"{values.shape[-1]} channels"
            )
        if any(not name.strip() for name in bands):
            raise ValueError("band names must be non-empty strings")
        if len(set(bands)) != len(bands):
            raise ValueError(f"band names must be unique, got {bands!r}")

        try:
            unit = self.unit if isinstance(self.unit, PixelUnit) else PixelUnit(self.unit)
        except ValueError as exc:
            raise ValueError(f"unsupported pixel unit {self.unit!r}") from exc
        if not isinstance(self.grid, (PhysicalGrid, AngularGrid)):
            raise TypeError(
                "grid must be a PhysicalGrid or AngularGrid, "
                f"got {type(self.grid).__name__}"
            )

        values.setflags(write=False)
        object.__setattr__(self, "data", values)
        object.__setattr__(self, "bands", bands)
        object.__setattr__(self, "unit", unit)

    def __repr__(self) -> str:
        return (
            f"ImageCube(shape={self.shape!r}, bands={self.bands!r}, "
            f"unit={self.unit.value!r}, grid={self.grid!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageCube):
            return NotImplemented
        return (
            self.bands == other.bands
            and self.unit == other.unit
            and self.grid == other.grid
            and np.array_equal(self.data, other.data)
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.data.shape[:2]  # type: ignore[return-value]

    @property
    def num_channels(self) -> int:
        return self.data.shape[-1]

    def band_index(self, name: str) -> int:
        try:
            return self.bands.index(name)
        except ValueError as exc:
            raise ValueError(f"band {name!r} not in {self.bands!r}") from exc

    def plane(self, band: str | None = None) -> np.ndarray:
        """Return one 2-D channel plane without copying it."""
        if band is None:
            if self.num_channels != 1:
                raise ValueError("band is required for a multi-channel cube")
            return self.data[..., 0]
        return self.data[..., self.band_index(band)]

    def cropped(self, rows: slice, columns: slice) -> ImageCube:
        """Return a spatial slice while retaining every channel."""
        if not isinstance(rows, slice) or not isinstance(columns, slice):
            raise TypeError("rows and columns must be slices")
        return self.with_data(self.data[rows, columns, :])

    def center_cropped(
        self,
        max_height: int,
        max_width: int | None = None,
    ) -> ImageCube:
        """Return the largest centred crop within the requested dimensions."""
        height_limit = int(max_height)
        width_limit = height_limit if max_width is None else int(max_width)
        if height_limit <= 0 or width_limit <= 0:
            raise ValueError("center-crop dimensions must be positive")
        height = min(self.shape[0], height_limit)
        width = min(self.shape[1], width_limit)
        row0 = (self.shape[0] - height) // 2
        col0 = (self.shape[1] - width) // 2
        return self.cropped(
            slice(row0, row0 + height),
            slice(col0, col0 + width),
        )

    def rotated_quarter(self, k: int) -> ImageCube:
        """Rotate counter-clockwise by ``k`` exact quarter turns."""
        return self.with_data(np.rot90(self.data, k=int(k) % 4, axes=(0, 1)))

    def rotated(self, angle_deg: float) -> ImageCube:
        """Rotate counter-clockwise about the image centre with bilinear sampling.

        Output has the same spatial shape and zero outside the input support.
        Exact multiples of 90 degrees use :meth:`rotated_quarter` so they stay
        lossless.  The implementation is NumPy-only to keep this module usable
        without importing an imaging framework.
        """
        angle = float(angle_deg)
        if not np.isfinite(angle):
            raise ValueError(f"angle_deg must be finite, got {angle_deg!r}")
        wrapped = angle % 360.0
        quarter = round(wrapped / 90.0)
        if np.isclose(wrapped, quarter * 90.0, rtol=0.0, atol=1e-12):
            return self.rotated_quarter(quarter)

        height, width = self.spatial_shape
        centre_y = 0.5 * (height - 1)
        centre_x = 0.5 * (width - 1)
        yy, xx = np.indices((height, width), dtype=np.float64)
        x_out = xx - centre_x
        y_out = yy - centre_y
        radians = np.deg2rad(wrapped)
        cosine = float(np.cos(radians))
        sine = float(np.sin(radians))
        source_x = cosine * x_out - sine * y_out + centre_x
        source_y = sine * x_out + cosine * y_out + centre_y

        x0 = np.floor(source_x).astype(np.int64)
        y0 = np.floor(source_y).astype(np.int64)
        x1 = x0 + 1
        y1 = y0 + 1
        wx = source_x - x0
        wy = source_y - y0
        output = np.zeros(self.shape, dtype=np.float32)

        neighbours = (
            (y0, x0, (1.0 - wy) * (1.0 - wx)),
            (y0, x1, (1.0 - wy) * wx),
            (y1, x0, wy * (1.0 - wx)),
            (y1, x1, wy * wx),
        )
        for source_rows, source_columns, weight in neighbours:
            valid = (
                (source_rows >= 0)
                & (source_rows < height)
                & (source_columns >= 0)
                & (source_columns < width)
            )
            if np.any(valid):
                output[valid] += (
                    self.data[source_rows[valid], source_columns[valid], :]
                    * weight[valid, None].astype(np.float32)
                )
        return self.with_data(output)

    def as_array(self, *, copy: bool = False) -> np.ndarray:
        """Return the HWC ndarray, optionally as an independent copy."""
        return self.data.copy() if copy else self.data

    def diagnostics(self) -> dict[str, object]:
        """Return a compact, serialization-friendly numerical summary."""
        if isinstance(self.grid, PhysicalGrid):
            grid = {
                "kind": "physical",
                "pixel_scale_pc": self.grid.pixel_scale_pc,
            }
        else:
            grid = {
                "kind": "angular",
                "pixel_scale_arcsec": self.grid.pixel_scale_arcsec,
            }
        return {
            "shape": self.shape,
            "spatial_shape": self.spatial_shape,
            "num_channels": self.num_channels,
            "bands": self.bands,
            "unit": self.unit.value,
            "grid": grid,
            "dtype": str(self.data.dtype),
            "min": float(np.min(self.data)),
            "max": float(np.max(self.data)),
            "sum": float(np.sum(self.data, dtype=np.float64)),
        }

    def with_data(
        self,
        data: np.ndarray,
        *,
        grid: Grid | None = None,
    ) -> ImageCube:
        """Return replacement pixels, optionally on a new sampling grid."""
        return type(self)(
            data=data,
            bands=self.bands,
            unit=self.unit,
            grid=self.grid if grid is None else grid,
        )

    @classmethod
    def stack(cls, cubes: Sequence[CubeLike]) -> ImageCube:
        """Concatenate compatible cubes along their channel axis."""
        values = tuple(cubes)
        if not values:
            raise ValueError("at least one cube is required")
        first = values[0]
        spatial_shape = tuple(first.data.shape[:2])
        unit = first.unit
        grid = first.grid
        arrays: list[np.ndarray] = []
        bands: list[str] = []
        for index, cube in enumerate(values):
            array = np.asarray(cube.data)
            if array.ndim != 3:
                raise ValueError(f"cube {index} is not HWC: shape {array.shape}")
            if tuple(array.shape[:2]) != spatial_shape:
                raise ValueError(
                    f"cube {index} has spatial shape {array.shape[:2]}, "
                    f"expected {spatial_shape}"
                )
            if cube.unit != unit:
                raise ValueError(f"cube {index} has unit {cube.unit!r}, expected {unit!r}")
            if cube.grid != grid:
                raise ValueError(f"cube {index} has grid {cube.grid!r}, expected {grid!r}")
            arrays.append(array)
            bands.extend(cube.bands)
        return cls(
            data=np.concatenate(arrays, axis=-1),
            bands=tuple(bands),
            unit=unit,
            grid=grid,
        )
