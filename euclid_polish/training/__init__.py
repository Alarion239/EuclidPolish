"""
Model Training module for EuclidPolish.

Multi-band training pipeline (4-channel LR input, 1-channel VIS HR target).
"""

from euclid_polish.training.inference import (
    load_model_from_checkpoint,
    load_model_from_weights,
    plot_reconstruction,
    reconstruct,
)
from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.training.target_blur import blur_target_array
from euclid_polish.training.trainer import Trainer

__all__ = [
    "Trainer",
    "wdsr",
    "reconstruct",
    "plot_reconstruction",
    "load_model_from_checkpoint",
    "load_model_from_weights",
    "blur_target_array",
]
