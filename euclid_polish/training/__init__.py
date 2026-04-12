"""
Model Training module for EuclidPolish.

This module provides classes for training super-resolution models,
including WDSR architecture, data loaders, and training utilities.
"""

from euclid_polish.training.trainer import Trainer
from euclid_polish.training.data import RadioSky
from euclid_polish.training.models.wdsr import wdsr

__all__ = [
    "Trainer",
    "RadioSky",
    "wdsr",
]
