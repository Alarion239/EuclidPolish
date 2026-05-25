"""Training-time augmentations for the transition model.

Two augmentations live here, both motivated by the analytic structure
of the problem (``A_θ`` is a linear convolution operator) rather than
by any photometric concern:

1. **Star injection.** The synthetic clean-scene generator produces
   galaxy-dominated fields; bright stars are under-represented. But the
   model's "PSF identity" probe is literally
   ``A_θ(PSF_HST) ≈ PSF_Euclid`` — i.e. a single point source IS the
   single most important sample. We inject synthetic star fields
   (sum of scaled PSFs at random positions) into the training mix so
   the loss sees that case explicitly every batch.

2. **Linear-combination augmentation.** Convolution is linear:
   ``A_θ(α·sceneA + β·sceneB)`` should equal
   ``α·A_θ(sceneA) + β·A_θ(sceneB)``. We pair each training example
   with a shuffled second example and (with some probability per
   element) emit the linear combination of the two. This pushes the
   network toward learning the linearity it ought to satisfy.

Neither augmentation produces "physically plausible" scenes; that's
not a concern because ``A_θ`` is a PSF transition operator, not a
scene generator. Its only job is to be a faithful linear-image-to-
linear-image map.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Star field synthesis (numpy core — used by both eager and graph-mode paths)
# ---------------------------------------------------------------------------

def make_star_field(
    psf_hst: np.ndarray,
    psf_euclid: np.ndarray,
    *,
    image_size: int,
    n_stars: int,
    amp_min: float = 10.0,
    amp_max: float = 1000.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Make one ``(input, target)`` star-field pair on the HR grid.

    Strategy: build a delta-image at random pixel positions with random
    amplitudes, then convolve once with each PSF. Because convolution
    is linear, this is mathematically equivalent to placing PSF stamps
    at each star's position and summing — but cheaper (one FFT per
    PSF, not one per star) and edge-clean (``fftconvolve(mode='same')``
    handles stars near the boundary correctly).

    Parameters
    ----------
    psf_hst, psf_euclid
        2-D float arrays on the same HR grid, each normalised to
        sum=1. The two convolutions must use *identical* delta-image
        positions and amplitudes so the resulting (input, target) pair
        is consistent with the linear-image-formation contract.
    image_size
        Side length of the square output (HR pixels).
    n_stars
        Number of point sources to inject. Each gets a random position
        and a random amplitude in ``[amp_min, amp_max]``.
    amp_min, amp_max
        Bounds of the per-star uniform amplitude draw (linear
        electrons; matches the unit space of saved pair shards).
    rng
        Optional ``np.random.Generator``. A fresh one is created when
        omitted — pass an explicit ``rng`` for reproducible tests.

    Returns
    -------
    (input_image, target_image)
        Each ``(image_size, image_size, 1)`` float32 — the trailing
        channel axis matches the schema ``A_θ`` expects.
    """
    from scipy import signal as scipy_signal

    if rng is None:
        rng = np.random.default_rng()
    if psf_hst.shape != psf_euclid.shape:
        raise ValueError(
            f"PSFs must share shape; got {psf_hst.shape} vs {psf_euclid.shape}"
        )
    if image_size < 8:
        raise ValueError(f"image_size must be ≥ 8, got {image_size}")
    if n_stars < 0:
        raise ValueError(f"n_stars must be ≥ 0, got {n_stars}")

    deltas = np.zeros((image_size, image_size), dtype=np.float64)
    for _ in range(int(n_stars)):
        cy = int(rng.integers(0, image_size))
        cx = int(rng.integers(0, image_size))
        amp = float(rng.uniform(amp_min, amp_max))
        deltas[cy, cx] += amp

    inp = scipy_signal.fftconvolve(
        deltas, psf_hst.astype(np.float64), mode="same",
    ).astype(np.float32)
    tgt = scipy_signal.fftconvolve(
        deltas, psf_euclid.astype(np.float64), mode="same",
    ).astype(np.float32)
    return inp[..., np.newaxis], tgt[..., np.newaxis]


# ---------------------------------------------------------------------------
# tf.data factories
# ---------------------------------------------------------------------------

def build_star_pair_dataset(
    psf_hst: np.ndarray,
    psf_euclid: np.ndarray,
    *,
    image_size: int,
    n_stars_min: int = 1,
    n_stars_max: int = 8,
    seed: Optional[int] = None,
):
    """An infinite tf.data.Dataset of ``(input, target)`` star pairs.

    Yields a fresh random field per draw; no two batches are identical.
    The underlying numpy generator is wrapped in
    ``tf.data.Dataset.from_generator`` so the TF graph sees standard
    tensors. ``prefetch(AUTOTUNE)`` keeps the producer ahead of the
    trainer.
    """
    import tensorflow as tf

    if n_stars_max < n_stars_min:
        raise ValueError(
            f"n_stars_max ({n_stars_max}) must be ≥ n_stars_min "
            f"({n_stars_min})"
        )

    rng = np.random.default_rng(seed)

    def _gen():
        # Local rng for thread safety — tf.data may spawn workers.
        local_rng = np.random.default_rng(rng.integers(0, 2**31))
        while True:
            n_stars = int(local_rng.integers(n_stars_min, n_stars_max + 1))
            yield make_star_field(
                psf_hst, psf_euclid,
                image_size=image_size, n_stars=n_stars,
                rng=local_rng,
            )

    sig = (
        tf.TensorSpec(shape=(image_size, image_size, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(image_size, image_size, 1), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(_gen, output_signature=sig)


# ---------------------------------------------------------------------------
# PSF identity pair (validation set seed)
# ---------------------------------------------------------------------------

def embed_in_canvas(arr: np.ndarray, target_side: int) -> np.ndarray:
    """Centre a square 2-D array in a ``target_side × target_side`` canvas.

    If ``arr`` is larger than ``target_side`` along either axis, returns
    a centre-crop. If smaller, returns a zero-padded canvas with ``arr``
    centred. Equal-size returns a copy. Square inputs only — the
    augmentation pipeline assumes square fields throughout.

    Used to place the HST and Euclid PSF kernels (which live on a
    421-pixel HR grid in the pair generator) onto whatever pixel grid
    the saved validation shards use (typically 128 or 256).
    """
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {arr.shape}")
    H, W = arr.shape
    if H != W:
        raise ValueError(f"expected square array, got shape {arr.shape}")
    if target_side < 1:
        raise ValueError(f"target_side must be ≥ 1, got {target_side}")
    if H == target_side:
        return arr.copy()
    if H > target_side:
        # Centre-crop. ``(H - target_side) // 2`` matches the convention
        # used by the PSF resampler so the centroid stays on the
        # canvas's centre pixel.
        i0 = (H - target_side) // 2
        return arr[i0:i0 + target_side, i0:i0 + target_side].copy()
    # H < target_side → pad. Symmetric padding keeps the PSF centroid
    # on the centre pixel of the canvas.
    pad = (target_side - H) // 2
    out = np.zeros((target_side, target_side), dtype=arr.dtype)
    out[pad:pad + H, pad:pad + H] = arr
    return out


def make_psf_identity_pair(
    psf_hst: np.ndarray,
    psf_euclid: np.ndarray,
    *,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """The literal "PSF identity" pair, sized for the validation canvas.

    Returns ``(input, target)`` where ``input`` is the HST PSF and
    ``target`` is the Euclid PSF, both centred in an
    ``image_size × image_size`` canvas. This is the **single most
    important validation sample**: the model's correctness as a PSF
    transition operator literally reduces to
    ``A_θ(PSF_HST) ≈ PSF_Euclid``. Including it in the validation
    set means the reported ``val_L1`` metric is bounded below by the
    error on this case — there's no way for the model to "trick" the
    metric by overfitting to the synthetic galaxy distribution while
    failing on point sources.

    Output shape is ``(image_size, image_size, 1)`` to match the
    on-disk TFRecord schema and ``A_θ``'s input layout.
    """
    if psf_hst.shape != psf_euclid.shape:
        raise ValueError(
            f"PSFs must share shape; got {psf_hst.shape} vs {psf_euclid.shape}"
        )
    inp = embed_in_canvas(psf_hst,    image_size).astype(np.float32)
    tgt = embed_in_canvas(psf_euclid, image_size).astype(np.float32)
    return inp[..., np.newaxis], tgt[..., np.newaxis]


# ---------------------------------------------------------------------------
# Linear-combination augmentation
# ---------------------------------------------------------------------------

def apply_linear_combo_augmentation(
    ds,
    *,
    fraction: float,
    seed: int,
    alpha_min: float = 0.3,
    alpha_max: float = 1.5,
    beta_min:  float = -0.5,
    beta_max:  float = 0.5,
):
    """Wrap a ``(input, target)`` dataset with a linear-combination mix.

    With probability ``fraction`` per example, the emitted pair is
    ``(α·inp_a + β·inp_b, α·tgt_a + β·tgt_b)`` where ``(inp_b, tgt_b)``
    is drawn from a shuffled copy of the same dataset. With
    probability ``1 - fraction`` the original ``(inp_a, tgt_a)`` is
    passed through unchanged.

    Why this works: convolution is linear, so a model that maps
    ``inp → tgt`` correctly for both A and B individually must also
    map their linear combinations correctly. Training on the combos
    directly is a free constraint — it costs nothing semantically
    (the pair is still a valid input/target pair of the same operator)
    and pushes the network toward the linearity it should already
    satisfy.

    ``α`` and ``β`` are drawn per-example from uniform ranges. The
    defaults keep ``α`` positive (preserving the dominant-image
    structure) while letting ``β`` swing to negative values
    (introducing subtraction-like cases that span more of the
    operator's input distribution).

    Parameters
    ----------
    ds
        Source ``tf.data.Dataset`` yielding ``(input, target)`` pairs.
        Should already be ``.repeat()``ed if you want infinite mixing —
        the shuffled second stream is also infinite by default here.
    fraction
        Probability per example of applying the mix. ``0`` returns the
        source dataset unchanged.
    seed
        RNG seed for the shuffled second stream + the per-example coin
        flips. Picking a different seed than the source's shuffle seed
        keeps the two streams uncorrelated.
    """
    import tensorflow as tf

    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    if fraction <= 0.0:
        return ds

    ds_b = ds.shuffle(buffer_size=256, seed=seed,
                      reshuffle_each_iteration=True)
    zipped = tf.data.Dataset.zip((ds, ds_b))

    frac = tf.constant(float(fraction), dtype=tf.float32)

    def _combine(pair_a, pair_b):
        inp_a, tgt_a = pair_a
        inp_b, tgt_b = pair_b
        # Per-example coin flip + random scalars.
        do_mix = tf.random.uniform([], dtype=tf.float32) < frac
        alpha  = tf.random.uniform(
            [], minval=alpha_min, maxval=alpha_max, dtype=tf.float32,
        )
        beta   = tf.random.uniform(
            [], minval=beta_min, maxval=beta_max, dtype=tf.float32,
        )
        # Use tf.where on the scalar boolean so both branches are
        # built into the graph; that's cheap because tf.cond would
        # serialise the dataset producer.
        inp_out = tf.where(do_mix, alpha * inp_a + beta * inp_b, inp_a)
        tgt_out = tf.where(do_mix, alpha * tgt_a + beta * tgt_b, tgt_a)
        return inp_out, tgt_out

    return zipped.map(_combine, num_parallel_calls=tf.data.AUTOTUNE)
