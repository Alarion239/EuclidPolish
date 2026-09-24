import math
from collections.abc import Sequence

import tensorflow as tf
from tf_keras.initializers import GlorotUniform, Initializer

from euclid_polish.config import Config
from euclid_polish.training.augmentation import expand_to_knees

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


def evaluate(model, dataset, knees: Sequence[float] | None = None,
             output_knee: float | None = None):
    """Validation metrics — PSNR in both stretched and raw space.

    ``knees`` marks a multi-knee member (channels = knee-major blocks of
    bands, see ``asinh_stretch_multi_knee``); its metrics come from
    :func:`_evaluate_multi_knee`. ``output_knee`` marks one that outputs a
    single image at that knee, scored at every knee.

    Peaks come from ``Config.PSNR_PEAK_*`` (mag-17 star electron count and
    its asinh-mapped value under STRETCH_SCALE_E). Set-mean of per-image
    PSNRs.

    Returns
    -------
    dict
        ``mae_stretched``:  mean absolute error in asinh space.
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
    if knees:
        return _evaluate_multi_knee(model, dataset, knees, output_knee)
    psnr_str_list  = []
    psnr_raw_list  = []
    mae_str_list   = []
    psnr_band_list = []
    ln10  = tf.constant(2.302585092994046, dtype=tf.float32)
    peak2 = _PSNR_MAX_VAL_STRETCHED ** 2

    # Every list holds per-IMAGE values (batch axis kept), so each image
    # counts once whatever the batch size or the size of the last batch.
    for lr, hr in dataset:
        sr = model(lr)
        psnr_str_list.append(tf.image.psnr(hr, sr, max_val=_PSNR_MAX_VAL_STRETCHED))
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
        psnr_band_list.append(psnr_band)

        hr_e = _to_electrons(hr)
        sr_e = _to_electrons(sr)
        psnr_raw_list.append(tf.image.psnr(hr_e, sr_e, max_val=_PSNR_MAX_VAL_RAW))

    return {
        "psnr_stretched": tf.reduce_mean(tf.concat(psnr_str_list, axis=0)),
        "psnr_raw":       tf.reduce_mean(tf.concat(psnr_raw_list, axis=0)),
        "mae_stretched":  tf.reduce_mean(mae_str_list),
        "psnr_band_stretched": tf.reduce_mean(
            tf.concat(psnr_band_list, axis=0), axis=0),                 # (C,)
    }


def _evaluate_multi_knee(model, dataset, knees: Sequence[float],
                         output_knee: float | None = None) -> dict:
    """Validation metrics for a multi-knee member.

    Every channel (band x knee) is scored on its own, against its knee's
    stretched peak ``asinh(PSNR_PEAK_E / knee)`` — one shared peak or a pooled
    MSE would let the low knees and VIS, whose stretched errors are largest,
    decide every comparison. The save-best metric ``psnr_stretched`` is the
    mean PSNR over all channels: with log-spaced knees, the band-averaged
    knee-grid analogue of the knee-integrated PSNR, and the quantity the
    ``balanced`` loss maximises. ``psnr_knee`` ``(K,)`` (logged) and
    ``psnr_band_stretched`` ``(C,)`` average it over bands and over knees;
    ``psnr_raw`` scores, in electrons, the head whose knee is nearest the
    per-band default; ``mae_stretched`` is the plain mean absolute error over
    every channel (the loss track's metric). A single-image member
    (``output_knee``) has its one image re-stretched at every knee first.
    """
    n_k = len(knees)
    peaks = tf.constant([math.asinh(float(Config.PSNR_PEAK_E) / float(q)) for q in knees],
                        dtype=tf.float32)                                  # (K,)
    head = min(range(n_k), key=lambda i: abs(math.log(float(knees[i])
                                                      / float(Config.STRETCH_SCALE_E))))
    head_knee = tf.constant(float(knees[head]), dtype=tf.float32)
    ln10 = tf.constant(2.302585092994046, dtype=tf.float32)
    channel_list, mae_list, raw_list = [], [], []
    for lr, hr in dataset:
        sr = model(lr)
        if output_knee is not None:
            sr = expand_to_knees(sr, output_knee, knees)
        c = int(hr.shape[-1]) // n_k
        shape = tf.shape(hr)
        err2 = tf.reshape(tf.square(hr - sr), [shape[0], shape[1], shape[2], n_k, c])
        mse_kc = tf.reduce_mean(err2, axis=[1, 2])                          # (B, K, C)
        channel_list.append(10.0 * tf.math.log(
            (peaks ** 2)[None, :, None] / tf.maximum(mse_kc, 1e-30)) / ln10)
        mae_list.append(tf.reduce_mean(tf.abs(hr - sr)))
        block = slice(head * c, (head + 1) * c)
        hr_e = tf.sinh(tf.clip_by_value(hr[..., block], -_SINH_CLIP, _SINH_CLIP)) * head_knee
        sr_e = tf.sinh(tf.clip_by_value(sr[..., block], -_SINH_CLIP, _SINH_CLIP)) * head_knee
        raw_list.append(tf.image.psnr(hr_e, sr_e, max_val=_PSNR_MAX_VAL_RAW))
    psnr_kc = tf.reduce_mean(tf.concat(channel_list, axis=0), axis=0)     # (K, C)
    return {
        "psnr_stretched": tf.reduce_mean(psnr_kc),
        "psnr_knee": tf.reduce_mean(psnr_kc, axis=1),
        "psnr_raw": tf.reduce_mean(tf.concat(raw_list, axis=0)),
        "mae_stretched": tf.reduce_mean(mae_list),
        "psnr_band_stretched": tf.reduce_mean(psnr_kc, axis=0),
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
