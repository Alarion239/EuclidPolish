"""Dihedral equivariance on real Euclid sky: a truth-free training penalty.

The super-resolution of a flipped or rotated input should be the flipped or
rotated super-resolution: ``f(g·x) = g·f(x)`` for every ``g`` in the dihedral
group D4 (four 90° rotations, each with or without a flip). Test-time
augmentation (TTA8) enforces that at inference by averaging the eight
orientations, and gains ~0.13–0.17 dB on synthetic scenes; on real tiles the
orientation spread is 1.05–1.2× the synthetic one (1.3–3× in NISP above
100 e⁻), so the missing equivariance lives where there is no truth.

This penalty asks for it during training, on unlabeled real LR crops::

    a = f(x)            b = g⁻¹·f(g·x)
    penalty = loss(a, stretch(b))

with the member's own reconstruction loss, so the weight is in units of the
supervised loss. Both branches carry gradients (neither is a fixed target),
and one forward pass evaluates both (``[x, g·x]`` stacked on the batch).
``g`` cycles through the seven non-identity elements, one per step.

Why it cannot move the optimum: training draws PSF orientations at random
and applies a uniform dihedral augmentation, so the best estimator for the
training distribution is itself D4-equivariant. The penalty only removes
orientation-dependent error, which is what TTA8 averages away.

D4 is written as ``F_W^c ∘ F_H^b ∘ T^a``: an optional transpose followed by
optional flips of the height and width axes. The three bits of
``code ∈ [0, 8)`` select them (code 0 is the identity) and all eight elements
come out once. Each part is its own inverse, so ``g⁻¹ = T^a ∘ F_H^b ∘ F_W^c``.
Square, even-sided crops keep every element exact on both the 0.10" input and
the 0.05" output grid (a 2× block maps onto a 2× block).
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
import tensorflow as tf

#: Number of D4 elements; codes 1..7 are the non-identity ones.
D4_ORDER = 8


def _bits(code: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    code = tf.cast(code, tf.int32)
    return (tf.bitwise.bitwise_and(code, 1) > 0,
            tf.bitwise.bitwise_and(code, 2) > 0,
            tf.bitwise.bitwise_and(code, 4) > 0)


def _transpose(x: tf.Tensor) -> tf.Tensor:
    return tf.transpose(x, [0, 2, 1, 3])


def dihedral(x: tf.Tensor, code: tf.Tensor | int) -> tf.Tensor:
    """Apply the D4 element ``code`` (0..7) to a square ``(N, S, S, C)`` batch."""
    transpose, flip_h, flip_w = _bits(tf.convert_to_tensor(code))
    x = tf.where(transpose, _transpose(x), x)
    x = tf.where(flip_h, tf.reverse(x, [1]), x)
    return tf.where(flip_w, tf.reverse(x, [2]), x)


def dihedral_inverse(x: tf.Tensor, code: tf.Tensor | int) -> tf.Tensor:
    """Undo :func:`dihedral` with the same ``code``."""
    transpose, flip_h, flip_w = _bits(tf.convert_to_tensor(code))
    x = tf.where(flip_w, tf.reverse(x, [2]), x)
    x = tf.where(flip_h, tf.reverse(x, [1]), x)
    return tf.where(transpose, _transpose(x), x)


def step_code(step: tf.Tensor) -> tf.Tensor:
    """The non-identity D4 element for training ``step``: 1..7 in turn."""
    return tf.cast(1 + tf.math.floormod(tf.cast(step, tf.int64), D4_ORDER - 1), tf.int32)


class RealEquivariance:
    """``loss(f(x), stretch(g⁻¹·f(g·x)))`` on stretched real LR crops.

    ``input_stretch`` turns a raw-electron real batch into the network's input
    (the member's LR stretch); ``target_stretch`` turns an electron
    prediction into the loss's target format (the member's HR stretch).
    """

    def __init__(self, input_stretch: Callable[[tf.Tensor], tf.Tensor],
                 target_stretch: Callable[[tf.Tensor], tf.Tensor], *,
                 source: str = ""):
        self._input_stretch = input_stretch
        self._target_stretch = target_stretch
        self.source = str(source)

    def penalty(self, model: tf.keras.Model, real_e: tf.Tensor,
                loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                code: tf.Tensor) -> tf.Tensor:
        """The penalty for one batch of raw-electron real crops ``(N, S, S, C)``."""
        x = self._input_stretch(real_e)
        n = tf.shape(x)[0]
        both = model(tf.concat([x, dihedral(x, code)], axis=0), training=True)
        direct = both[:n]
        turned_back = dihedral_inverse(both[n:], code)
        return loss(direct, self._target_stretch(turned_back))

    def describe(self) -> str:
        return ("f(g·x) = g·f(x) on real crops, member loss, g = D4 element "
                "1..7 by step" + (f", from {self.source}" if self.source else ""))


# --------------------------------------------------------------------------- #
# the real crops
# --------------------------------------------------------------------------- #

def load_real_tiles(path: str) -> np.ndarray:
    """The real LR tiles ``(N, H, W, 4)`` float32 electrons from a ``.npy``.

    Memory-mapped (the full set can be several GB). Non-finite pixels must
    have been cleaned when the set was packed; this refuses a set with any.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"real-equivariance tiles not found: {path}")
    tiles = np.load(path, mmap_mode="r")
    if tiles.ndim != 4 or tiles.shape[-1] != 4 or tiles.shape[1] != tiles.shape[2]:
        raise ValueError(f"real tiles must be (N, S, S, 4), got {tiles.shape}")
    if tiles.dtype != np.float32:
        raise ValueError(f"real tiles must be float32, got {tiles.dtype}")
    return tiles


def real_crop_dataset(tiles: np.ndarray, *, crop: int, batch_size: int,
                      seed: int) -> tf.data.Dataset:
    """An endless dataset of ``(batch_size, crop, crop, 4)`` random real crops.

    Each element picks a random tile and a random even-aligned crop position
    (no dihedral augmentation: the penalty's ``g`` does that). Seeded, so a
    member's real crops replay with its seed.
    """
    n, side = int(tiles.shape[0]), int(tiles.shape[1])
    crop = int(crop)
    if crop % 2 or crop > side:
        raise ValueError(f"crop must be even and at most the tile side {side}, got {crop}")
    rng = np.random.default_rng(seed)

    def gen():
        span = (side - crop) // 2 + 1
        while True:
            i = int(rng.integers(n))
            y, x = (int(v) * 2 for v in rng.integers(span, size=2))
            yield np.asarray(tiles[i, y:y + crop, x:x + crop, :], np.float32)

    spec = tf.TensorSpec((crop, crop, int(tiles.shape[-1])), tf.float32)
    return (tf.data.Dataset.from_generator(gen, output_signature=spec)
            .batch(int(batch_size), drop_remainder=True))
