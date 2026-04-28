import tensorflow as tf

from euclid_polish.config import Config

# PSNR is reported in two complementary spaces:
# - **stretched** (asinh): the space the loss actually optimises. max_val = 10
#   matches the typical bright-end stretched value ``asinh(5e6/1000) ≈ 9.2``.
#   This is the metric the trainer's "save best" decision uses.
# - **raw** (electrons after sinh inverse): for human-readable comparison
#   against typical photometric scales. ``max_val = 1e6`` ≈ a moderately
#   bright source pixel.
_PSNR_MAX_VAL_STRETCHED = tf.constant(10.0, dtype=tf.float32)
_PSNR_MAX_VAL_RAW       = tf.constant(float(Config.RAW_PSNR_MAX_VAL), dtype=tf.float32)
_STRETCH_SCALE          = tf.constant(float(Config.STRETCH_SCALE_E),  dtype=tf.float32)
_SINH_CLIP              = tf.constant(20.0, dtype=tf.float32)  # sinh(20)*1k ≈ 2.4e8


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


def _to_electrons(stretched: tf.Tensor) -> tf.Tensor:
    """Invert the data-loader stretch: stretched (asinh) → raw electrons."""
    return tf.sinh(tf.clip_by_value(stretched, -_SINH_CLIP, _SINH_CLIP)) * _STRETCH_SCALE


def evaluate(model, dataset):
    """Validation PSNR.

    Returns
    -------
    psnr_stretched : tf.Tensor
        Mean PSNR in asinh-stretched space (loss-aligned, used for "save
        best" decisions).
    psnr_raw : tf.Tensor
        Mean PSNR after sinh-inverse to electrons. Reported for information
        only.
    """
    ps_str, ps_raw = [], []
    for lr, hr in dataset:
        sr = model(lr)
        ps_str.append(tf.image.psnr(hr, sr, max_val=_PSNR_MAX_VAL_STRETCHED)[0])

        hr_e = _to_electrons(hr)
        sr_e = _to_electrons(sr)
        ps_raw.append(tf.image.psnr(hr_e, sr_e, max_val=_PSNR_MAX_VAL_RAW)[0])

    return tf.reduce_mean(ps_str), tf.reduce_mean(ps_raw)


# ---------------------------------------------------------------------------
# Sub-pixel upsampling
# ---------------------------------------------------------------------------

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
