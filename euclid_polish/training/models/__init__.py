"""
Model architectures for EuclidPolish.

This module contains neural network architectures for super-resolution.
"""

from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.training.models.common import (
    resolve_single,
    evaluate,
    normalize_minmax,
)

__all__ = [
    "wdsr",
    "resolve_single",
    "evaluate",
    "normalize_minmax",
]
