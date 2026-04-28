import tensorflow as tf
import tensorflow_probability as tfp

from tf_keras.layers import Add, Conv2D, Input, Lambda
from tf_keras.models import Model

from euclid_polish.config import Config
from euclid_polish.training.models.common import pixel_shuffle


# Network sees fluxes (electrons) at input, emits fluxes at output.
# Internally it operates on the tanh-stretched representation in (-1, 1).
_TANH_SCALE = float(Config.TANH_SCALE_E)
_TANH_EPS = 1e-6


def _normalize(x):
    """Electrons → (-1, 1) via fixed tanh stretch."""
    return tf.tanh(x / _TANH_SCALE)


def _denormalize(y):
    """(-1, 1) → electrons via arctanh, with a tiny clip to avoid ±∞."""
    y_clipped = tf.clip_by_value(y, -1.0 + _TANH_EPS, 1.0 - _TANH_EPS)
    return tf.atanh(y_clipped) * _TANH_SCALE


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


def wdsr(scale, num_filters=32, num_res_blocks=8, res_block_expansion=6, res_block_scaling=None, nchan=1):
    """WDSR model. Input/output are raw electrons (float32).

    The first layer applies a fixed tanh stretch (electrons → (-1, 1)); the
    final layer applies arctanh (back to electrons). The residual core
    operates on the normalized representation so values stay bounded.
    """
    x_in = Input(shape=(None, None, nchan))
    x = Lambda(_normalize)(x_in)   # electrons → (-1, 1)

    # main branch
    m = conv2d_weightnorm(num_filters, nchan, padding='same')(x)
    for i in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion, kernel_size=3, scaling=res_block_scaling)
    m = conv2d_weightnorm(nchan * scale ** 2, 3, padding='same', name=f'conv2d_main_scale_{scale}')(m)
    m = Lambda(pixel_shuffle(scale))(m)

    # skip branch
    s = conv2d_weightnorm(nchan * scale ** 2, 5, padding='same', name=f'conv2d_skip_scale_{scale}')(x)
    s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    x = Lambda(_denormalize)(x)    # (-1, 1) → electrons

    return Model(x_in, x, name="wdsr")
