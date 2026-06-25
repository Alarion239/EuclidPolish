"""Typed, behaviour-bearing cutouts.

A class hierarchy where each leaf owns its verb and carries provenance:

    Cutout
    ├── HRCutout (0.05″/pix)
    │   ├── SyntheticHRCutout  — .convolve(forward) -> SyntheticLRCutout
    │   └── SRCutout           — output of Model.upsample(lr, ...)
    └── LRCutout (0.10″/pix)
        ├── SyntheticLRCutout
        └── EuclidLRCutout     — .query(ra, dec, size) from the archive
"""

from __future__ import annotations

from euclid_polish.cutout.base import (
    Cutout,
    EuclidLRCutout,
    HRCutout,
    LRCutout,
    SRCutout,
    SyntheticHRCutout,
    SyntheticLRCutout,
)

__all__ = [
    "Cutout",
    "LRCutout",
    "HRCutout",
    "EuclidLRCutout",
    "SyntheticLRCutout",
    "SyntheticHRCutout",
    "SRCutout",
]
