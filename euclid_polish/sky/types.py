"""
Typed data structures for sky images.

Carrying pixel_scale alongside the data enables validation at convolution
time: the PSF kernel must be sampled at the same scale as the image it
will be convolved with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SkyImage:
    """A sky image with its physical metadata."""

    data: np.ndarray          # float32, shape (H, W)
    pixel_scale: float        # arcsec / pixel
    is_clean: bool            # True → HR clean image; False → LR dirty image
    index: Optional[int] = None      # position in the dataset
    subset: Optional[str] = None     # 'train' or 'validate'
    metadata: dict = field(default_factory=dict)  # galaxy/star params from simulate_field

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]
