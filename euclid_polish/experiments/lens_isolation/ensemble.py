"""Explicit, registry-independent lens-isolation ensemble inference."""

from __future__ import annotations

import glob
import os

import numpy as np

from euclid_polish.model import Model


def _has_checkpoint(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "checkpoint")) or bool(glob.glob(os.path.join(path, "*.index")))


def detection_score(reconstruction: np.ndarray, aperture: int | None = None) -> float:
    """Positive reconstructed flux, optionally in a central square aperture."""
    image = np.maximum(np.asarray(reconstruction, np.float32), 0)
    if aperture is not None:
        radius = int(aperture)
        if radius < 0:
            raise ValueError("aperture must be non-negative")
        cy, cx = image.shape[0] // 2, image.shape[1] // 2
        image = image[
            max(0, cy - radius) : cy + radius + 1,
            max(0, cx - radius) : cx + radius + 1,
        ]
    return float(image.sum())


class LensIsolationEnsemble:
    """Load only ``member_*`` checkpoints beneath the experiment base."""

    def __init__(self, base_dir: str, *, model_factory=Model) -> None:
        self.base_dir = os.path.abspath(base_dir)
        dirs = sorted(
            path
            for path in glob.glob(os.path.join(self.base_dir, "member_*"))
            if os.path.isdir(path) and _has_checkpoint(path)
        )
        if not dirs:
            raise RuntimeError(f"no lens-isolation members found under {self.base_dir!r}")
        self._member_dirs = dirs
        self._members = [model_factory(path) for path in dirs]

    @property
    def members(self):
        return list(self._members)

    @property
    def member_names(self) -> list[str]:
        return [os.path.basename(path) for path in self._member_dirs]

    def member_arrays(self, lr: np.ndarray) -> np.ndarray:
        return np.stack([model.upsample_array(lr) for model in self._members])

    def predict(self, lr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predictions = self.member_arrays(lr)
        return predictions.mean(axis=0), predictions.std(axis=0)
