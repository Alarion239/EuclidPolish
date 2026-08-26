"""Native TNG surface-brightness image value object.

The TNG SKIRT atlas has one deliberately narrow numerical image domain:
finite ``float32`` surface brightness in MJy/sr sampled on a physical
parsec-per-pixel grid.  Encoding that domain in a dedicated type keeps native
atlas pixels separate from detector images without adding generic unit/grid
metadata to the rest of the project.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


def _positive_finite(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return number


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class TNGSurfaceBrightnessImage:
    """Owned HWC MJy/sr pixels on a physical TNG grid.

    Construction always takes a C-contiguous ``float32`` copy and marks it
    read-only.  The class therefore owns its numerical state even when the
    caller supplied a view, a non-native-endian FITS array, or a mutable input.
    """

    data: np.ndarray = field(repr=False)
    bands: tuple[str, ...]
    pixel_scale_pc: float

    def __post_init__(self) -> None:
        values = np.array(self.data, dtype=np.float32, order="C", copy=True)
        if values.ndim != 3:
            raise ValueError(
                "TNGSurfaceBrightnessImage.data must have shape (H, W, C), "
                f"got {values.shape}"
            )
        if any(side <= 0 for side in values.shape):
            raise ValueError(
                "TNGSurfaceBrightnessImage.data must be non-empty, "
                f"got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "TNGSurfaceBrightnessImage.data must contain only finite values"
            )

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

        values.setflags(write=False)
        object.__setattr__(self, "data", values)
        object.__setattr__(self, "bands", bands)
        object.__setattr__(
            self,
            "pixel_scale_pc",
            _positive_finite(self.pixel_scale_pc, "pixel_scale_pc"),
        )

    def __repr__(self) -> str:
        return (
            "TNGSurfaceBrightnessImage("
            f"shape={self.shape!r}, bands={self.bands!r}, "
            f"pixel_scale_pc={self.pixel_scale_pc!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TNGSurfaceBrightnessImage):
            return NotImplemented
        return (
            self.bands == other.bands
            and self.pixel_scale_pc == other.pixel_scale_pc
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
        """Return one read-only 2-D channel plane without copying it."""
        if band is None:
            if self.num_channels != 1:
                raise ValueError("band is required for a multi-channel image")
            return self.data[..., 0]
        return self.data[..., self.band_index(band)]

    def cropped(
        self,
        rows: slice,
        columns: slice,
    ) -> TNGSurfaceBrightnessImage:
        """Return a spatial slice while retaining every band."""
        if not isinstance(rows, slice) or not isinstance(columns, slice):
            raise TypeError("rows and columns must be slices")
        return self.with_data(self.data[rows, columns, :])

    def rotated_quarter(self, k: int) -> TNGSurfaceBrightnessImage:
        """Rotate counter-clockwise by ``k`` exact quarter turns."""
        return self.with_data(np.rot90(self.data, k=int(k) % 4, axes=(0, 1)))

    def as_array(self, *, copy: bool = False) -> np.ndarray:
        """Return the HWC array, optionally as an independent writable copy."""
        return self.data.copy() if copy else self.data

    def with_data(
        self,
        data: np.ndarray,
        *,
        pixel_scale_pc: float | None = None,
    ) -> TNGSurfaceBrightnessImage:
        """Return replacement pixels, optionally at a new physical sampling."""
        return type(self)(
            data=data,
            bands=self.bands,
            pixel_scale_pc=(
                self.pixel_scale_pc
                if pixel_scale_pc is None
                else pixel_scale_pc
            ),
        )

    @classmethod
    def stack(
        cls,
        images: Sequence[TNGSurfaceBrightnessImage],
    ) -> TNGSurfaceBrightnessImage:
        """Concatenate compatible native images along the band axis."""
        values = tuple(images)
        if not values:
            raise ValueError("at least one image is required")
        first = values[0]
        if not isinstance(first, cls):
            raise TypeError(
                "native TNG stacks require TNGSurfaceBrightnessImage values"
            )
        spatial_shape = first.spatial_shape
        pixel_scale_pc = first.pixel_scale_pc
        arrays: list[np.ndarray] = []
        bands: list[str] = []
        for index, image in enumerate(values):
            if not isinstance(image, cls):
                raise TypeError(
                    f"image {index} is not a TNGSurfaceBrightnessImage"
                )
            if image.spatial_shape != spatial_shape:
                raise ValueError(
                    f"image {index} has spatial shape {image.spatial_shape}, "
                    f"expected {spatial_shape}"
                )
            if image.pixel_scale_pc != pixel_scale_pc:
                raise ValueError(
                    f"image {index} has pixel scale {image.pixel_scale_pc!r} pc, "
                    f"expected {pixel_scale_pc!r} pc"
                )
            arrays.append(image.data)
            bands.extend(image.bands)
        return cls(
            data=np.concatenate(arrays, axis=-1),
            bands=tuple(bands),
            pixel_scale_pc=pixel_scale_pc,
        )
