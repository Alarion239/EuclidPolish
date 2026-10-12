"""WDSR-A super-resolution model.

The network operates **entirely in asinh-stretched space**: the data loader
applies ``asinh(x / Config.STRETCH_SCALE_E)`` to both LR and HR before
batches reach the model, and the model's output is also in stretched space.
Inference callers (see ``training/inference.py``) apply ``sinh`` to recover
electron counts.

Why no in-graph stretch / inverse:
- ``arctanh`` saturates and explodes near ±1, killing gradients on bright
  pixels (Euclid bright stars in raw electron units).
- Doing the loader-side stretch keeps targets, predictions, and gradients all
  in the same well-behaved space (~ ±10 for our typical data range).
- WDSR is BN-free by design; loader-side normalization is the appropriate
  preconditioning.

Exception: a member with a learned output knee (``learned_output_knee``)
ends in :class:`~euclid_polish.training.models.output_knee.LearnedOutputKnee`,
which turns the stretched output into electrons with one trainable knee per
band, so that model outputs electrons.
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tf_keras.layers import Add, Concatenate, Conv2D, Input, Lambda
from tf_keras.models import Model

from euclid_polish.training.models.common import ICNR, pixel_shuffle
from euclid_polish.training.models.output_knee import LearnedOutputKnee

#: Keras dtype policy of the residual blocks' convolutions under mixed
#: precision: bfloat16 arithmetic, float32 variables.
MIXED_POLICY = "mixed_bfloat16"


class PrecisionWeightNorm(tfp.layers.weight_norm.WeightNorm):
    """tfp's :class:`WeightNorm` with two changes. Its variables, and so the
    checkpoint layout, are unchanged.

    * The kernel is normalised in float32 (this wrapper's dtype) and then cast
      to the wrapped conv's compute dtype, so a bfloat16 conv gets a bfloat16
      kernel without the norm itself being taken in bfloat16.
    * ``eager_init``: ``g`` takes its starting value (the norm of ``v``) when
      the layer is built rather than on its first call. That removes tfp's
      per-call ``initialized`` check, a ``tf.cond`` in every layer of every
      training step; the computed values are the same.
    """

    def __init__(self, layer, *, eager_init=False, **kwargs):
        super().__init__(layer, data_init=False, **kwargs)
        self.eager_init = bool(eager_init)

    def build(self, input_shape=None):
        super().build(input_shape)
        if self.eager_init:
            self._init_norm()
            self.initialized.assign(True)

    def _compute_weights(self):
        super()._compute_weights()
        self.layer.kernel = tf.cast(self.layer.kernel, self.layer.compute_dtype)

    def call(self, inputs):
        if not self.eager_init:
            return super().call(inputs)
        self._compute_weights()
        return self.layer(inputs)


def conv2d_weightnorm(filters, kernel_size, padding="same", activation=None, *,
                      mixed_precision=False, eager_init=False, **kwargs):
    """A weight-normalised conv. ``mixed_precision``: the conv runs in
    bfloat16 (``MIXED_POLICY``); ``eager_init``: see :class:`PrecisionWeightNorm`.
    Without either it is tfp's stock ``WeightNorm``, as every checkpoint was
    trained."""
    if not (mixed_precision or eager_init):
        return tfp.layers.weight_norm.WeightNorm(
            Conv2D(
                filters,
                kernel_size,
                padding=padding,
                activation=activation,
                **kwargs,
            ),
            data_init=False,
        )
    if mixed_precision:
        kwargs["dtype"] = MIXED_POLICY
    conv = Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs)
    # A float32 wrapper that passes its input through uncast: the conv casts
    # it to its own compute dtype.
    return PrecisionWeightNorm(conv, eager_init=eager_init, dtype="float32", autocast=False)


def res_block(x_in, num_filters, expansion, kernel_size, scaling, *,
              mixed_precision=False, eager_init=False):
    linear = 0.8
    opts = {"mixed_precision": mixed_precision, "eager_init": eager_init}
    x = conv2d_weightnorm(num_filters * expansion, 1, padding='same', activation='relu',
                          **opts)(x_in)
    x = conv2d_weightnorm(int(num_filters * linear), 1, padding='same', **opts)(x)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same', **opts)(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def wdsr(scale, num_filters=32, num_res_blocks=8, res_block_expansion=6,
         res_block_scaling=None, nchan_in=1, nchan_out=None,
         entry_kernel_size=3, skip_kernel_size=5,
         per_band_skip=None, icnr=False, input_knees=1,
         learned_output_knee=None, mixed_precision=False,
         eager_weight_norm_init=False):
    """WDSR-A model with potentially asymmetric input/output channels.

    Parameters
    ----------
    scale        : integer SR factor (HR size = ``scale * LR size``).
    num_filters  : feature-map width inside the trunk.
    num_res_blocks : depth of the residual trunk.
    res_block_expansion : WDSR-A wide-activation expansion ratio.
    res_block_scaling   : optional output scaling per block.
    nchan_in     : input channel count (4 for the multi-band pipeline:
                   VIS, Y_E, J_E, H_E).
    nchan_out    : output channel count (4 for the VIS+NISP HR target;
                   1 for the VIS-only model).
                   Defaults to ``nchan_in`` for backward compatibility.
    entry_kernel_size : 2-D kernel size for the first trunk conv.
    skip_kernel_size : kernel of the skip branch's conv(s).
    icnr         : initialise the pre-``pixel_shuffle`` convs (the main
                   branch's and every skip conv) with
                   :class:`~euclid_polish.training.models.common.ICNR` so the
                   upsampler starts as a checkerboard-free nearest-neighbour
                   resize (see that class). Init-only — the layer graph is
                   unchanged, so it is a per-member knob for NEW members and
                   old checkpoints restore identically with it off.
    per_band_skip : make the residual (skip) branch PER-BAND — output
                   band ``k``'s skip conv sees ONLY input channel ``k``,
                   so every cross-band path runs through the shared
                   trunk (which mixes all bands from the entry conv on)
                   and the skip stays an identity-like per-band
                   upsampler. ``None`` (default) auto-enables it when
                   ``nchan_in == nchan_out > 1`` (the 4-band model);
                   the dense all-to-all skip is kept for asymmetric /
                   single-channel configs, preserving the legacy layer
                   graph so old checkpoints still restore.
    input_knees  : knee blocks in the input of a single-image multi-knee
                   member (``nchan_in = input_knees · nchan_out``, knee-major:
                   channel ``j·nchan_out + k`` is band ``k`` at knee ``j``).
                   Output band ``k``'s per-band skip then reads band ``k`` at
                   every knee, so bands still only mix through the trunk.
    learned_output_knee : starting knee (e⁻) of a learned output head: the
                   model then ends in ``LearnedOutputKnee`` and outputs
                   electrons, one trainable knee per output band. ``None``
                   (default) keeps the stretched output.
    mixed_precision : run the residual blocks' convolutions in bfloat16
                   (``MIXED_POLICY``), about 95% of the arithmetic. The
                   residual stream between blocks, the entry and tail convs,
                   the skip branch and the learned knee stay float32:
                   bfloat16 keeps 8 significant bits, too coarse for the
                   stretched image itself (a stretched value of 10 rounds to
                   steps of 0.06). Variables stay float32 either way, so the
                   weights load into either build.
    eager_weight_norm_init : give every weight norm its starting value when
                   the layer is built, dropping the per-call initialisation
                   check (see :class:`PrecisionWeightNorm`). Same values.

    Input  : asinh-stretched LR tensor, shape ``(B, H, W, nchan_in)``.
    Output : asinh-stretched SR tensor, shape ``(B, scale*H, scale*W, nchan_out)``.
    """
    if nchan_out is None:
        nchan_out = nchan_in
    knees = int(input_knees)
    if per_band_skip is None:
        per_band_skip = (nchan_in == nchan_out * knees and nchan_out > 1)
    if per_band_skip and nchan_in != nchan_out * knees:
        raise ValueError(
            f"per_band_skip needs nchan_in == nchan_out (band k → band k); "
            f"got nchan_in={nchan_in}, nchan_out={nchan_out}, input_knees={knees}"
        )

    x_in = Input(shape=(None, None, nchan_in))

    # ICNR init for the sub-pixel convs (checkerboard-free upsample at step 0);
    # ``None`` → Conv2D's own default (glorot_uniform), i.e. bit-identical to
    # the pre-knob build. Each pre-shuffle conv gets its OWN ICNR instance so
    # their random base draws stay independent (they output different planes).
    def _shuffle_init():
        return ICNR(scale) if icnr else None

    # Main branch — the SHARED trunk: the entry conv mixes all input
    # bands into one feature stack, so every res block learns the
    # cross-band correlations jointly.
    eager = {"eager_init": eager_weight_norm_init}
    m = conv2d_weightnorm(num_filters, entry_kernel_size, padding='same', **eager)(x_in)
    for _ in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion,
                      kernel_size=3, scaling=res_block_scaling,
                      mixed_precision=mixed_precision, **eager)
    m = conv2d_weightnorm(nchan_out * scale ** 2, 3, padding='same',
                          name=f'conv2d_main_scale_{scale}',
                          kernel_initializer=_shuffle_init(), **eager)(m)
    m = Lambda(pixel_shuffle(scale))(m)

    # Skip branch
    if per_band_skip:
        # PER-BAND residual: band k's skip conv reads input channel k
        # only (1 → scale² feature maps → pixel-shuffle → 1 HR plane),
        # so the skip cannot mix bands — all cross-band information has
        # to flow through the trunk above.
        s_bands = []
        for k in range(nchan_out):
            if knees == 1:
                band_in = Lambda(
                    lambda t, k=k: t[..., k:k + 1],
                    name=f'skip_band_{k}_slice',
                )(x_in)
            else:
                # Band k at every knee (knee-major input: stride nchan_out).
                band_in = Lambda(
                    lambda t, k=k: t[..., k::nchan_out],
                    name=f'skip_band_{k}_slice',
                )(x_in)
            s_k = conv2d_weightnorm(
                scale ** 2, skip_kernel_size, padding='same',
                name=f'conv2d_skip_band_{k}_scale_{scale}',
                kernel_initializer=_shuffle_init(), **eager,
            )(band_in)
            s_bands.append(Lambda(pixel_shuffle(scale))(s_k))
        s = Concatenate(axis=-1)(s_bands)
    else:
        # Dense all-to-all skip (legacy layer graph — old checkpoints
        # restore against this structure).
        s = conv2d_weightnorm(nchan_out * scale ** 2, skip_kernel_size,
                              padding='same',
                              name=f'conv2d_skip_scale_{scale}',
                              kernel_initializer=_shuffle_init(), **eager)(x_in)
        s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    if learned_output_knee is not None:
        # A learned output knee: one trainable asinh knee per band turns the
        # stretched output into electrons, so the model outputs electrons.
        x = LearnedOutputKnee(init_knee_e=learned_output_knee)(x)
    return Model(x_in, x, name="wdsr")
