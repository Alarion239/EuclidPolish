import tensorflow as tf

# PSNR is reported in the **asinh-stretched** space (the same space the
# network and loss operate in). The typical bright-end value for our data is
# around asinh(5e6 / 1000) ≈ 9.2, so max_val = 10 yields meaningful dB
# numbers for the bulk of the dynamic range without being dominated by the
# rare extreme outliers.
_PSNR_MAX_VAL = tf.constant(10.0, dtype=tf.float32)


# ---------------------------------------------------------------------------
# Inference helpers
#
# The model takes asinh-stretched LR and emits asinh-stretched SR. Callers
# that want raw electrons should apply ``sinh(out) * STRETCH_SCALE_E`` (see
# ``training/inference.py``).
# ---------------------------------------------------------------------------

def resolve_single(model, lr):
    """Run SR on one image. Input/output are asinh-stretched float32."""
    sr = model(tf.expand_dims(lr, axis=0))
    return sr[0]


def evaluate(model, dataset):
    """Validation PSNR on stretched (lr, hr) pairs."""
    psnr_values = []
    for lr, hr in dataset:
        sr = model(lr)
        psnr_value = tf.image.psnr(hr, sr, max_val=_PSNR_MAX_VAL)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


# ---------------------------------------------------------------------------
# Sub-pixel upsampling
# ---------------------------------------------------------------------------

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
