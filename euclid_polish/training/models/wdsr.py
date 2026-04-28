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

from tf_keras.layers import Add, Conv2D, Input, Lambda
from tf_keras.models import Model

from euclid_polish.training.models.common import pixel_shuffle


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
         res_block_scaling=None, nchan=1):
    """WDSR-A model.

    Input  : asinh-stretched LR tensor, shape (B, H, W, nchan).
    Output : asinh-stretched SR tensor, shape (B, scale*H, scale*W, nchan).
    """
    x_in = Input(shape=(None, None, nchan))

    # Main branch
    m = conv2d_weightnorm(num_filters, nchan, padding='same')(x_in)
    for _ in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion,
                      kernel_size=3, scaling=res_block_scaling)
    m = conv2d_weightnorm(nchan * scale ** 2, 3, padding='same',
                          name=f'conv2d_main_scale_{scale}')(m)
    m = Lambda(pixel_shuffle(scale))(m)

    # Skip branch
    s = conv2d_weightnorm(nchan * scale ** 2, 5, padding='same',
                          name=f'conv2d_skip_scale_{scale}')(x_in)
    s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    return Model(x_in, x, name="wdsr")
