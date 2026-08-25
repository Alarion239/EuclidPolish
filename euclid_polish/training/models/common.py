import tensorflow as tf
from tf_keras.initializers import GlorotUniform, Initializer

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
        ``psnr_stretched``: mean JOINT PSNR in asinh space — the MSE is
                           pooled over all H×W×C pixels (all bands for
                           the 4-band model), max_val ≈
                           asinh(mag17_e / k) ≈ 9.34. Loss-aligned, used
                           for save-best decisions.
        ``psnr_raw``:       mean joint PSNR in raw electrons
                           (max_val = mag-17 star ≈ 5.68×10⁶ e⁻).
        ``psnr_band_stretched``: ``(C,)`` tensor of per-band stretched
                           PSNRs (channel k of HR vs channel k of SR) —
                           MONITORING ONLY, never feeds save-best. Lets
                           the training log show whether VIS and the
                           noisier NISP channels improve independently
                           of the NISP-dominated joint number.
    """
    psnr_str_list  = []
    psnr_raw_list  = []
    mae_str_list   = []
    psnr_band_list = []
    ln10  = tf.constant(2.302585092994046, dtype=tf.float32)
    peak2 = _PSNR_MAX_VAL_STRETCHED ** 2

    for lr, hr in dataset:
        sr = model(lr)
        psnr_str_list.append(tf.image.psnr(hr, sr, max_val=_PSNR_MAX_VAL_STRETCHED)[0])
        # Validation MAE in asinh space — the held-out analogue of the
        # MeanAbsoluteError training loss (same model output + stretch),
        # computed here for free since we already forward ``lr``.
        mae_str_list.append(tf.reduce_mean(tf.abs(hr - sr)))
        # Per-band PSNR: MSE over H×W only, one value per channel. Same
        # stretched peak as the joint metric (all bands share the asinh
        # knee, so the stretched values live on one scale).
        mse_band = tf.reduce_mean(tf.square(hr - sr), axis=[1, 2])      # (B, C)
        psnr_band = 10.0 * tf.math.log(
            peak2 / tf.maximum(mse_band, 1e-12)) / ln10
        psnr_band_list.append(psnr_band[0])

        hr_e = _to_electrons(hr)
        sr_e = _to_electrons(sr)
        psnr_raw_list.append(tf.image.psnr(hr_e, sr_e, max_val=_PSNR_MAX_VAL_RAW)[0])

    return {
        "psnr_stretched": tf.reduce_mean(psnr_str_list),
        "psnr_raw":       tf.reduce_mean(psnr_raw_list),
        "mae_stretched":  tf.reduce_mean(mae_str_list),
        "psnr_band_stretched": tf.reduce_mean(
            tf.stack(psnr_band_list), axis=0),                          # (C,)
    }


# ---------------------------------------------------------------------------
# Sub-pixel upsampling
# ---------------------------------------------------------------------------

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


class ICNR(Initializer):
    """ICNR init (Aitken+17) for the conv that feeds a ``pixel_shuffle``.

    A sub-pixel (pixel-shuffle) upsampler is ``depth_to_space`` over a conv
    whose ``C_out · scale²`` output channels are unpacked into a ``scale×scale``
    output block. With independent random filters those ``scale²`` sub-pixels
    start uncorrelated, so the upsampler emits a ``scale``-periodic checkerboard
    that the network then spends training budget learning to cancel — visible
    as background speckle / peristellar hot-dots, especially under L2.

    ICNR instead initialises the ``scale²`` sub-pixel filters of each output
    channel as **identical copies** of one base-initialised filter. Then every
    sub-pixel in a block gets the same value ⇒ the block is constant ⇒ the
    upsampler starts as an exact nearest-neighbour resize, checkerboard-free.
    Training departs from there and the copies diverge as needed; ICNR only
    fixes the starting point (it is init-only — the layer graph is unchanged,
    so existing checkpoints restore identically).

    ``base`` defaults to ``GlorotUniform`` — the Conv2D default — so with
    ``scale == 1`` (no upsampling) this is a plain passthrough of that init.

    The copy layout is a ``tile`` (not a ``repeat``): ``depth_to_space`` routes
    the ``scale²`` sub-pixels of output channel ``c`` to the *strided* input
    channels ``{c, C_out+c, 2·C_out+c, …}``, so the base block must repeat as a
    whole (``[s₀…s_{C-1}, s₀…s_{C-1}, …]``) for those strided picks to coincide.
    """

    def __init__(self, scale, base=None):
        self.scale = int(scale)
        self.base = base if base is not None else GlorotUniform()

    def __call__(self, shape, dtype=None, **kwargs):
        del kwargs
        scale2 = self.scale * self.scale
        out = int(shape[-1])
        if out % scale2 != 0:
            raise ValueError(
                f"ICNR: output channels {out} not divisible by scale² "
                f"({self.scale}² = {scale2})")
        sub_shape = list(shape[:-1]) + [out // scale2]
        sub = self.base(sub_shape, dtype=dtype)          # [kh, kw, in, C_out]
        return tf.tile(sub, [1, 1, 1, scale2])           # strided-copy layout

    def get_config(self):
        return {"scale": self.scale,
                "base": tf.keras.initializers.serialize(self.base)}
