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
"""

import tensorflow_probability as tfp
from tf_keras.layers import Add, Concatenate, Conv2D, Input, Lambda
from tf_keras.models import Model

from euclid_polish.training.models.common import ICNR, pixel_shuffle


def conv2d_weightnorm(filters, kernel_size, padding="same", activation=None, **kwargs):
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


def res_block(x_in, num_filters, expansion, kernel_size, scaling):
    linear = 0.8
    x = conv2d_weightnorm(num_filters * expansion, 1, padding='same', activation='relu')(x_in)
    x = conv2d_weightnorm(int(num_filters * linear), 1, padding='same')(x)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def wdsr(scale, num_filters=32, num_res_blocks=8, res_block_expansion=6,
         res_block_scaling=None, nchan_in=1, nchan_out=None,
         entry_kernel_size=3, skip_kernel_size=5,
         nchan=None, per_band_skip=None, icnr=False):
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
    nchan        : back-compat alias for ``nchan_in`` (also sets
                   ``nchan_out = nchan`` if ``nchan_out`` is None).
    icnr         : initialise the three pre-``pixel_shuffle`` convs with
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

    Input  : asinh-stretched LR tensor, shape ``(B, H, W, nchan_in)``.
    Output : asinh-stretched SR tensor, shape ``(B, scale*H, scale*W, nchan_out)``.
    """
    if nchan is not None:
        nchan_in = nchan
        if nchan_out is None:
            nchan_out = nchan
    if nchan_out is None:
        nchan_out = nchan_in
    if per_band_skip is None:
        per_band_skip = (nchan_in == nchan_out and nchan_out > 1)
    if per_band_skip and nchan_in != nchan_out:
        raise ValueError(
            f"per_band_skip needs nchan_in == nchan_out (band k → band k); "
            f"got nchan_in={nchan_in}, nchan_out={nchan_out}"
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
    m = conv2d_weightnorm(num_filters, entry_kernel_size, padding='same')(x_in)
    for _ in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion,
                      kernel_size=3, scaling=res_block_scaling)
    m = conv2d_weightnorm(nchan_out * scale ** 2, 3, padding='same',
                          name=f'conv2d_main_scale_{scale}',
                          kernel_initializer=_shuffle_init())(m)
    m = Lambda(pixel_shuffle(scale))(m)

    # Skip branch
    if per_band_skip:
        # PER-BAND residual: band k's skip conv reads input channel k
        # only (1 → scale² feature maps → pixel-shuffle → 1 HR plane),
        # so the skip cannot mix bands — all cross-band information has
        # to flow through the trunk above.
        s_bands = []
        for k in range(nchan_out):
            band_in = Lambda(
                lambda t, k=k: t[..., k:k + 1],
                name=f'skip_band_{k}_slice',
            )(x_in)
            s_k = conv2d_weightnorm(
                scale ** 2, skip_kernel_size, padding='same',
                name=f'conv2d_skip_band_{k}_scale_{scale}',
                kernel_initializer=_shuffle_init(),
            )(band_in)
            s_bands.append(Lambda(pixel_shuffle(scale))(s_k))
        s = Concatenate(axis=-1)(s_bands)
    else:
        # Dense all-to-all skip (legacy layer graph — old checkpoints
        # restore against this structure).
        s = conv2d_weightnorm(nchan_out * scale ** 2, skip_kernel_size,
                              padding='same',
                              name=f'conv2d_skip_scale_{scale}',
                              kernel_initializer=_shuffle_init())(x_in)
        s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    return Model(x_in, x, name="wdsr")
