import tensorflow as tf

from euclid_polish.config import Config


# PSNR is reported in the **normalized** tanh space ([-1, 1]) so the metric
# is bounded and not dominated by saturated bright stars. The dynamic range
# is 2.0; ``max_val=2.0`` gives 0 dB at unit RMSE in that space.
_PSNR_MAX_VAL = tf.constant(2.0, dtype=tf.float32)
_TANH_SCALE = tf.constant(float(Config.TANH_SCALE_E), dtype=tf.float32)


def _to_tanh_space(x: tf.Tensor) -> tf.Tensor:
    return tf.tanh(x / _TANH_SCALE)


# ---------------------------------------------------------------------------
# Inference helpers
#
# Inputs and outputs of the model are raw electron fluxes (float32). The
# model itself wraps tanh/arctanh as its first/last layers (see wdsr.py).
# ---------------------------------------------------------------------------

def resolve_single(model, lr):
    """Run SR on a single image (raw float32 electrons) and return float32."""
    sr = model(tf.expand_dims(lr, axis=0))
    return sr[0]


def evaluate(model, dataset):
    """Evaluate model PSNR on a dataset of (lr, hr) pairs in flux space.

    The metric itself is computed in tanh-stretched space so saturated
    bright pixels don't dominate.
    """
    psnr_values = []
    for lr, hr in dataset:
        sr = model(lr)
        psnr_value = tf.image.psnr(
            _to_tanh_space(hr),
            _to_tanh_space(sr),
            max_val=_PSNR_MAX_VAL,
        )[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


# ---------------------------------------------------------------------------
# Sub-pixel upsampling
# ---------------------------------------------------------------------------

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
