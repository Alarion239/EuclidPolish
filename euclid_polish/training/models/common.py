import tensorflow as tf

from euclid_polish.config import Config

# PSNR peak references — derived from a mag-17 star's expected electron
# count over the stacked integration (a "very bright" but plausible source).
# Stretched peak is asinh(peak_e / STRETCH_SCALE_E) under the same scale the
# loader uses.
_PSNR_MAX_VAL_STRETCHED = tf.constant(float(Config.PSNR_PEAK_STRETCHED), dtype=tf.float32)
_PSNR_MAX_VAL_RAW       = tf.constant(float(Config.PSNR_PEAK_E),         dtype=tf.float32)

_STRETCH_SCALE = tf.constant(float(Config.STRETCH_SCALE_E), dtype=tf.float32)
_SINH_CLIP     = tf.constant(float(Config.SINH_STRETCH_CLIP), dtype=tf.float32)  # sinh(clip)·k ≈ 2.4e8


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


def evaluate(model, dataset):
    """Validation metrics — PSNR in both stretched and raw space.

    Peaks come from ``Config.PSNR_PEAK_*`` (mag-17 star electron count and
    its asinh-mapped value under STRETCH_SCALE_E). Set-mean of per-image
    PSNRs.

    Returns
    -------
    dict
        ``psnr_stretched``: mean PSNR in asinh space
                           (max_val ≈ asinh(mag17_e / k) ≈ 9.34).
                           Loss-aligned, used for save-best decisions.
        ``psnr_raw``:       mean PSNR in raw electrons
                           (max_val = mag-17 star ≈ 5.68×10⁶ e⁻).
    """
    psnr_str_list = []
    psnr_raw_list = []

    for lr, hr in dataset:
        sr = model(lr)
        psnr_str_list.append(tf.image.psnr(hr, sr, max_val=_PSNR_MAX_VAL_STRETCHED)[0])

        hr_e = _to_electrons(hr)
        sr_e = _to_electrons(sr)
        psnr_raw_list.append(tf.image.psnr(hr_e, sr_e, max_val=_PSNR_MAX_VAL_RAW)[0])

    return {
        "psnr_stretched": tf.reduce_mean(psnr_str_list),
        "psnr_raw":       tf.reduce_mean(psnr_raw_list),
    }


# ---------------------------------------------------------------------------
# Sub-pixel upsampling
# ---------------------------------------------------------------------------

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
