"""Dihedral equivariance on real Euclid sky: a truth-free training penalty.

The super-resolution of a flipped or rotated input should be the flipped or
rotated super-resolution: ``f(g·x) = g·f(x)`` for every ``g`` in the dihedral
group D4 (four 90° rotations, each with or without a flip). Test-time
augmentation (TTA8) enforces that at inference by averaging the eight
orientations, and gains ~0.13–0.17 dB on synthetic scenes; on real tiles the
orientation spread is 1.05–1.2× the synthetic one (1.3–3× in NISP above
100 e⁻), so the missing equivariance lives where there is no truth.

This penalty asks for it during training, on unlabeled real LR crops::

    a = y(x)            b = g⁻¹·y(g·x)
    penalty = mean((a − b)²)

``y`` is the network's stretched output before the learned-knee head (the
electrons are ``k·sinh(clip(y, ±20))``). Both branches carry gradients
(neither is a fixed target), and one forward pass evaluates both
(``[x, g·x]`` stacked on the batch). ``g`` cycles through the seven
non-identity elements, one per step. The weight ramps in linearly over a
stretch of the run (:func:`ramp_weight`), after the member has learned the
supervised task; before the ramp the extra forward pass is skipped.

Why this form (v2, 11-22). v1 scored ``f(x)`` against ``g⁻¹·f(g·x)`` in
electrons with the member's own loss, from step 0, and broke both 100k runs
(266/267: 37 dB against the control's 49.5). The balanced loss is a geometric
mean over band × knee channels, so on the gap between two outputs of one
network its gradient stays O(λ) however small the gap gets: |g| sat at
20–130 all run and every step was clipped, starving the supervised loss.
And in electrons, pixels pinned at the head's sinh clip are identical in both
branches, so saturating was a free way to satisfy it. A plain MSE of ``y``
gives a small gap a small gradient and sees past the clip.

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

The mask (11-25). Two places where the penalty did harm, measured on the
λ 1 / 0.3 fine-tunes (271 / 278) against their no-penalty control (277):

* **Blank input pixels.** Under a large blank (exactly zero) NISP patch on a
  very bright core the penalty members leave the patch empty in every band
  (poster nucleus B; a synthetic core with a 7.5-px blank disc keeps 0.83 →
  0.00 of its light from 36k to 720k e⁻ peak, while 277 keeps 1.00–1.07).
  An empty patch is trivially equivariant, and at that brightness the
  supervised scenes give no counter-signal. Pixels within ``blank_margin``
  LR px of an input pixel that is exactly zero in any band are not charged.
* **The crop border.** Without the penalty the orientation spread at a crop
  edge is ~8× the interior (a ``same``-padded conv treats the four edges
  differently), so the penalty's pull concentrates there and leaves a
  frame-locked rim (+0.03 e⁻ on the outer 3 SR px of every side). The outer
  ``border`` LR px of the crop are not charged.

The mask is computed on the input grid from the raw crop, is D4-invariant
(both branches are compared in the original frame) and is repeated onto the
output grid. The penalty is the mean square over the kept output pixels;
``border = blank_margin = 0`` is exactly the unmasked v2 penalty.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
import tensorflow as tf

#: Number of D4 elements; codes 1..7 are the non-identity ones.
D4_ORDER = 8

#: Default ramp of the penalty's weight, as fractions of the run's steps:
#: zero until 20%, linear to the full λ at 40%. The member learns the
#: supervised task first; the penalty then only removes orientation-
#: dependent error.
DEFAULT_RAMP = (0.2, 0.4)

#: The penalty's form, recorded in origin.json (v1 had none: the member's
#: loss on electrons).
FORM = "pre_head_mse"

#: Default mask of new runs, (crop border, margin around blank input pixels)
#: in LR px. A member recorded without one (before 11-25) ran unmasked.
DEFAULT_MASK = (4, 4)


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


def ramp_weight(step: tf.Tensor, weight: float, start: int, full: int) -> tf.Tensor:
    """The penalty's weight at ``step``: 0 until ``start``, linear to
    ``weight`` at ``full``, then ``weight`` (``full <= start``: a step)."""
    step = tf.cast(step, tf.float32)
    if int(full) > int(start):
        frac = tf.clip_by_value((step - float(start)) / float(int(full) - int(start)), 0.0, 1.0)
    else:
        frac = tf.cast(step >= float(start), tf.float32)
    return tf.constant(float(weight), tf.float32) * frac


def pre_head_view(model: tf.keras.Model) -> tf.keras.Model:
    """``model`` up to its learned-knee head: the stretched output ``y``
    (shares the variables). A model without a head is its own view."""
    head = model.layers[-1]
    if type(head).__name__ != "LearnedOutputKnee":
        return model
    return type(model)(model.inputs, head.input, name=f"{model.name}_pre_head")


def penalty_mask(real_e: tf.Tensor, *, border: int, blank_margin: int) -> tf.Tensor:
    """``(N, S, S, 1)`` float32 on the input grid: 1 where the penalty is
    charged, 0 within ``border`` px of the crop edge or within
    ``blank_margin`` px (a square neighbourhood) of an input pixel that is
    exactly zero in any band."""
    x = tf.convert_to_tensor(real_e)
    keep = tf.ones_like(x[..., :1], tf.float32)
    if blank_margin > 0:
        blank = tf.cast(tf.reduce_any(tf.equal(x, 0.0), axis=-1, keepdims=True), tf.float32)
        size = 2 * int(blank_margin) + 1
        near = tf.nn.max_pool2d(blank, ksize=size, strides=1, padding="SAME")
        keep = keep * (1.0 - near)
    if border > 0:
        side_h, side_w = tf.shape(x)[1], tf.shape(x)[2]
        rows = tf.range(side_h)
        cols = tf.range(side_w)
        inner_r = tf.cast((rows >= border) & (rows < side_h - border), tf.float32)
        inner_c = tf.cast((cols >= border) & (cols < side_w - border), tf.float32)
        keep = keep * (inner_r[None, :, None, None] * inner_c[None, None, :, None])
    return keep


class RealEquivariance:
    """``mean((y(x) − g⁻¹·y(g·x))²)`` on stretched real LR crops, ``y`` the
    network's output before its learned-knee head (:func:`pre_head_view`).

    ``input_stretch`` turns a raw-electron real batch into the network's input
    (the member's LR stretch). ``border`` / ``blank_margin`` (LR px) mask the
    crop edge and the neighbourhood of blank input pixels
    (:func:`penalty_mask`); 0 / 0 is the unmasked penalty.
    """

    def __init__(self, input_stretch: Callable[[tf.Tensor], tf.Tensor], *, source: str = "",
                 border: int = 0, blank_margin: int = 0):
        if int(border) < 0 or int(blank_margin) < 0:
            raise ValueError(f"the mask's border and blank margin must be non-negative, got "
                             f"{border!r}, {blank_margin!r}")
        self._input_stretch = input_stretch
        self.source = str(source)
        self.border = int(border)
        self.blank_margin = int(blank_margin)

    def penalty(self, net: tf.keras.Model, real_e: tf.Tensor, code: tf.Tensor) -> tf.Tensor:
        """The penalty for one batch of raw-electron real crops ``(N, S, S, C)``;
        ``net`` gives the stretched output (a :func:`pre_head_view`)."""
        x = self._input_stretch(real_e)
        n = tf.shape(x)[0]
        both = tf.cast(net(tf.concat([x, dihedral(x, code)], axis=0), training=True), tf.float32)
        gap = both[:n] - dihedral_inverse(both[n:], code)
        if not (self.border or self.blank_margin):
            return tf.reduce_mean(tf.square(gap))
        keep = penalty_mask(real_e, border=self.border, blank_margin=self.blank_margin)
        factor = tf.shape(gap)[1] // tf.shape(keep)[1]
        keep = tf.repeat(tf.repeat(keep, factor, axis=1), factor, axis=2)
        total = tf.reduce_sum(keep * tf.square(gap))
        count = tf.reduce_sum(keep) * tf.cast(tf.shape(gap)[-1], tf.float32)
        return tf.math.divide_no_nan(total, count)

    def describe(self) -> str:
        mask = (f", masked: {self.border} LR px border, {self.blank_margin} LR px around blank "
                f"input pixels" if self.border or self.blank_margin else "")
        return ("f(g·x) = g·f(x) on real crops: MSE of the stretched output before the "
                "knee head, g = D4 element 1..7 by step" + mask
                + (f", from {self.source}" if self.source else ""))


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
