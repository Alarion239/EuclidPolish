import tensorflow as tf

from euclid_polish.config import Config

MAX_VAL = tf.constant(Config.MAX_PIXEL_VALUE, dtype=tf.float32)


# ---------------------------------------------------------------------------
# Inference helpers
#
# The model expects input already in [0, 65535].  Training TFRecords are
# written with per-image min-max normalization at *generation* time (see
# CLI _convolve_hr_to_lr), so no additional normalization is needed here.
# ---------------------------------------------------------------------------

def resolve_single(model, lr):
    """Run SR on a single image already in [0, 65535].

    Matches polish-pub ``resolve16``: clip → round → cast to uint16.
    """
    sr = model(tf.expand_dims(lr, axis=0))
    sr = tf.clip_by_value(sr, 0.0, MAX_VAL)
    sr = tf.round(sr)
    sr = tf.cast(sr, tf.uint16)
    return sr[0]


def evaluate(model, dataset):
    """Evaluate model PSNR on a dataset of (lr, hr) pairs in [0, 65535] space."""
    psnr_values = []
    for lr, hr in dataset:
        sr = tf.clip_by_value(model(lr), 0.0, MAX_VAL)
        sr = tf.round(sr)
        sr = tf.cast(sr, tf.uint16)
        psnr_value = tf.image.psnr(hr, sr, max_val=MAX_VAL)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


# ---------------------------------------------------------------------------
# Sub-pixel upsampling
# ---------------------------------------------------------------------------

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
