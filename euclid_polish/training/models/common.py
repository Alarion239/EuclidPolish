import tensorflow as tf


# ---------------------------------------------------------------------------
# Per-image min-max normalization (used in the data pipeline, NOT in the model)
# ---------------------------------------------------------------------------

def normalize_minmax(x: tf.Tensor) -> tf.Tensor:
    """Per-image min-max normalization to [-1, 1]."""
    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    denom = tf.maximum(x_max - x_min, 1e-8)
    return (x - x_min) / denom * 2.0 - 1.0


def normalize_pair(lr: tf.Tensor, hr: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Normalize an (LR, HR) pair using shared min/max so the same
    physical value maps to the same normalized value in both images."""
    pair_min = tf.minimum(tf.reduce_min(lr), tf.reduce_min(hr))
    pair_max = tf.maximum(tf.reduce_max(lr), tf.reduce_max(hr))
    denom = tf.maximum(pair_max - pair_min, 1e-8)
    lr_norm = (lr - pair_min) / denom * 2.0 - 1.0
    hr_norm = (hr - pair_min) / denom * 2.0 - 1.0
    return lr_norm, hr_norm


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def resolve_single(model, lr):
    """Run SR on a single image. Returns float32 tensor."""
    sr = model(tf.expand_dims(lr, axis=0))
    return sr[0]


def evaluate(model, dataset):
    """Evaluate model PSNR on a dataset of (lr, hr) pairs in [-1, 1] space."""
    psnr_values = []
    for lr, hr in dataset:
        sr = model(lr)
        # Both sr and hr should be in [-1, 1]; compute PSNR with max_val=2.0
        psnr_value = tf.image.psnr(hr, sr, max_val=2.0)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


# ---------------------------------------------------------------------------
# Sub-pixel upsampling
# ---------------------------------------------------------------------------

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
