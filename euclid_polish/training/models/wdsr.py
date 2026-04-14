import tensorflow_probability as tfp

from tf_keras.layers import Add, Conv2D, Input, Lambda
from tf_keras.models import Model

from euclid_polish.config import Config
from euclid_polish.training.models.common import pixel_shuffle

_HALF = 2.0 ** Config.DEFAULT_NBIT / 2.0   # 32768.0 — matches polish-pub


# Fixed linear transforms: [0, 65535] ↔ ~[-1, 1]  (matching polish-pub)
def _normalize(x):
    return (x - _HALF) / _HALF

def _denormalize(x):
    return x * _HALF + _HALF


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
    """WDSR model. Expects input in [0, 65535] (pre-normalized per image)."""
    x_in = Input(shape=(None, None, nchan))
    x = Lambda(_normalize)(x_in)   # [0, 65535] → [-1, 1]

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
    x = Lambda(_denormalize)(x)    # [-1, 1] → [0, 65535]

    return Model(x_in, x, name="wdsr")
