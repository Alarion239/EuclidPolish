import tensorflow as tf

from euclid_polish.config import Config

MAX_VAL = tf.constant(Config.MAX_PIXEL_VALUE, dtype=tf.float32)


# ---------------------------------------------------------------------------
# Per-image min-max normalization to [0, 65535].
#
# TFRecords store raw flux values.  This function is applied per-image in
# the data loader (training) and at inference time so the model always
# receives input in [0, 65535], matching the polish-pub uint16 convention.
# ---------------------------------------------------------------------------

def normalize(x: tf.Tensor) -> tf.Tensor:
    """Per-image min-max normalization to [0, 65535]."""
    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    denom = tf.maximum(x_max - x_min, 1e-8)
    return (x - x_min) / denom * MAX_VAL


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def resolve_single(model, lr):
    """Run SR on a single image. Returns float32 tensor clipped to [0, 65535]."""
    sr = model(tf.expand_dims(lr, axis=0))
    return tf.clip_by_value(sr[0], 0.0, MAX_VAL)


def evaluate(model, dataset):
    """Evaluate model PSNR on a dataset of (lr, hr) pairs in [0, 65535] space."""
    psnr_values = []
    for lr, hr in dataset:
        sr = tf.clip_by_value(model(lr), 0.0, MAX_VAL)
        psnr_value = tf.image.psnr(hr, sr, max_val=MAX_VAL)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


# ---------------------------------------------------------------------------
# Sub-pixel upsampling
# ---------------------------------------------------------------------------

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
