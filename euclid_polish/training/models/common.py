import math

import tensorflow as tf

from euclid_polish.config import Config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# PSNR reference (legacy / SR-paper-style; max_val is somewhat arbitrary
# for un-normalised electron data).
_PSNR_MAX_VAL_STRETCHED = tf.constant(10.0, dtype=tf.float32)
_PSNR_MAX_VAL_RAW       = tf.constant(float(Config.RAW_PSNR_MAX_VAL), dtype=tf.float32)

_STRETCH_SCALE          = tf.constant(float(Config.STRETCH_SCALE_E),  dtype=tf.float32)
_SINH_CLIP              = tf.constant(20.0, dtype=tf.float32)  # sinh(20)*1k ≈ 2.4e8
_EPS                    = tf.constant(1e-7, dtype=tf.float32)


def _noise_floor_lr_e_var() -> float:
    """Expected per-LR-pixel noise variance (e⁻²) from the noise model.

    σ²_LR = sky + dark + N_exp · read²

    Used as the reference for the noise-floor SNR.
    """
    t_total = Config.EXPOSURE_TIME_S * Config.N_EXPOSURES
    pixel_area = Config.VIS_PIXEL_SCALE_ARCSEC ** 2
    sky_e   = Config.SKY_E_PER_S_PER_ARCSEC2 * pixel_area * t_total
    dark_e  = Config.DARK_E_PER_S_PER_PIX * t_total
    read_var = Config.READ_NOISE_E ** 2 * Config.N_EXPOSURES
    return float(sky_e + dark_e + read_var)


_NOISE_LR_STD = tf.constant(math.sqrt(_noise_floor_lr_e_var()), dtype=tf.float32)


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
    """Variance-ratio SNR in dB: 10·log10(var(signal) / var(residual))."""
    var_s = tf.math.reduce_variance(signal)
    var_r = tf.math.reduce_variance(residual) + _EPS
    return 10.0 * tf.math.log(var_s / var_r) / tf.math.log(10.0)


def _snr_floor_db(residual_e: tf.Tensor) -> tf.Tensor:
    """Noise-floor-referenced SNR in dB: 20·log10(σ_LR / RMSE_raw).

    > 0 dB means the residual is *below* the irreducible per-pixel noise floor
    (model fits within noise). 0 dB means the residual sits at the noise.
    """
    rmse = tf.sqrt(tf.reduce_mean(residual_e ** 2)) + _EPS
    return 20.0 * tf.math.log(_NOISE_LR_STD / rmse) / tf.math.log(10.0)


def evaluate(model, dataset):
    """Evaluate the model on a dataset; return a dict of metrics.

    Returns
    -------
    dict
        Keys:
          psnr_stretched   — set-mean PSNR in asinh space (loss-aligned, used
                             for save-best decisions)
          snr_var_stretched — set-mean variance-ratio SNR in asinh space
          snr_var_raw       — set-mean variance-ratio SNR in raw electrons
          snr_noise_raw     — set-mean noise-floor-referenced SNR in raw
                              electrons (σ_floor from the physics model)
    """
    psnr_str_list = []
    snr_var_str_list = []
    snr_var_raw_list = []
    snr_noise_raw_list = []

    for lr, hr in dataset:
        sr = model(lr)
        # PSNR in stretched space — kept as the loss-aligned save-best metric
        # and for SR-paper-style comparisons.
        psnr_str_list.append(tf.image.psnr(hr, sr, max_val=_PSNR_MAX_VAL_STRETCHED)[0])

        # Variance-ratio SNR in stretched space
        residual_str = hr - sr
        snr_var_str_list.append(_snr_var_db(hr, residual_str))

        # Convert to electrons for raw-space metrics
        hr_e = _to_electrons(hr)
        sr_e = _to_electrons(sr)
        residual_e = hr_e - sr_e

        snr_var_raw_list.append(_snr_var_db(hr_e, residual_e))
        snr_noise_raw_list.append(_snr_floor_db(residual_e))

    return {
        "psnr_stretched":    tf.reduce_mean(psnr_str_list),
        "snr_var_stretched": tf.reduce_mean(snr_var_str_list),
        "snr_var_raw":       tf.reduce_mean(snr_var_raw_list),
        "snr_noise_raw":     tf.reduce_mean(snr_noise_raw_list),
    }


# ---------------------------------------------------------------------------
# Sub-pixel upsampling
# ---------------------------------------------------------------------------

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
