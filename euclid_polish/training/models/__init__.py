"""
Model architectures for EuclidPolish.

This module contains neural network architectures for super-resolution.
"""

from euclid_polish.training.models.common import (
    evaluate,
    resolve_single,
)
from euclid_polish.training.models.wdsr import wdsr

__all__ = [
    "wdsr",
    "resolve_single",
    "evaluate",
]
