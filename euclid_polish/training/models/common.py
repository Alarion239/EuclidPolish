import tensorflow as tf

from euclid_polish.config import Config

# Network operates in asinh-stretched space; helpers below invert the
# stretch when a metric needs raw electrons.
_STRETCH_SCALE = tf.constant(float(Config.STRETCH_SCALE_E), dtype=tf.float32)
_SINH_CLIP     = tf.constant(20.0, dtype=tf.float32)   # sinh(20)·k ≈ 2.4e8
_EPS           = tf.constant(1e-7, dtype=tf.float32)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def resolve_single(model, lr):
    """Run SR on one image. Input/output are asinh-stretched float32."""
    sr = model(tf.expand_dims(lr, axis=0))
    return sr[0]


def _to_electrons(stretched: tf.Tensor) -> tf.Tensor:
    """Invert the loader's asinh: stretched → raw electrons."""
    return tf.sinh(tf.clip_by_value(stretched, -_SINH_CLIP, _SINH_CLIP)) * _STRETCH_SCALE


def _snr_var_db(signal: tf.Tensor, residual: tf.Tensor) -> tf.Tensor:
    """Variance-ratio SNR in dB: ``10 · log10(var(signal) / var(residual))``.

    A scale-free analogue of PSNR — no peak-signal reference is required,
    so this works on un-normalised electron data.
    """
    var_s = tf.math.reduce_variance(signal)
    var_r = tf.math.reduce_variance(residual) + _EPS
    return 10.0 * tf.math.log(var_s / var_r) / tf.math.log(10.0)


def evaluate(model, dataset):
    """Validation metrics — set-mean SNR_var in both stretched and raw space.

    Returns
    -------
    dict
        ``snr_var_stretched``: variance-ratio SNR in asinh space (loss-aligned;
                               used for save-best decisions).
        ``snr_var_raw``:       same in raw electrons.

    No PSNR is reported because it requires a peak-signal reference
    (``max_val``) that has no natural value for un-normalised astronomical
    data.
    """
    snr_var_str_list = []
    snr_var_raw_list = []

    for lr, hr in dataset:
        sr = model(lr)
        residual_str = hr - sr
        snr_var_str_list.append(_snr_var_db(hr, residual_str))

        hr_e = _to_electrons(hr)
        sr_e = _to_electrons(sr)
        residual_e = hr_e - sr_e
        snr_var_raw_list.append(_snr_var_db(hr_e, residual_e))

    return {
        "snr_var_stretched": tf.reduce_mean(snr_var_str_list),
        "snr_var_raw":       tf.reduce_mean(snr_var_raw_list),
    }


# ---------------------------------------------------------------------------
# Sub-pixel upsampling
# ---------------------------------------------------------------------------

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
