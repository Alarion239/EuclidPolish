"""
Multi-band data loader for training.

Reads v2 TFRecords produced by the new pipeline:

  * ``dirty_{subset}.tfrecord`` — (H_lr, W_lr, 4) LR float32 electrons,
    band order :attr:`Config.LR_INPUT_BAND_NAMES`.
  * ``hr_{subset}.tfrecord`` — (H_hr, W_hr, 1) HR float32 electrons,
    VIS-only target written by the forward step.
  * ``clean_{subset}.tfrecord`` — (H_hr, W_hr, 4) full 4-band HR clean
    record kept for inspection only (not used by the loader).

Each channel is asinh-stretched with its own per-band knee from
:attr:`BandConfig.asinh_stretch_scale_e` before the network sees it. The
per-band scale is a constant, so the stretch is applied in-graph via a
broadcast multiply — no Python loop per batch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

from euclid_polish.config import Config
from euclid_polish.sky.tfrecord import parse_record_graph_v2, tfrecord_path

import os as _os


# ---------------------------------------------------------------------------
# Per-band asinh stretch (graph constants)
# ---------------------------------------------------------------------------
#
# The stretch scales are immutable constants per-band. We materialise them
# as float32 numpy arrays at import time and rely on TF's auto-conversion
# from numpy → tensor inside ``tf.asinh`` / ``tf.sinh``. This is faster
# than building a fresh ``tf.constant`` each call AND avoids the
# lru_cache-across-test-fixtures pitfalls that would come from holding TF
# tensors in module-level caches.

_LR_STRETCH_SCALE_NP = np.array(
    [Config.get_band(name).asinh_stretch_scale_e
     for name in Config.LR_INPUT_BAND_NAMES],
    dtype=np.float32,
)  # shape (4,)

_HR_STRETCH_SCALE_NP = np.array(
    [Config.get_band(Config.HR_TARGET_BAND_NAME).asinh_stretch_scale_e],
    dtype=np.float32,
)  # shape (1,)


def _lr_stretch_scale() -> np.ndarray:
    """Length-4 vector of asinh stretch scales, one per LR channel."""
    return _LR_STRETCH_SCALE_NP


def _hr_stretch_scale() -> np.ndarray:
    """Length-1 scalar broadcasted against ``(B, H, W, 1)``."""
    return _HR_STRETCH_SCALE_NP


def asinh_stretch_lr(x: tf.Tensor) -> tf.Tensor:
    """asinh(x / k) per channel; ``x`` has shape ``(..., 4)``."""
    return tf.asinh(x / _lr_stretch_scale())


def asinh_stretch_hr(x: tf.Tensor) -> tf.Tensor:
    """asinh(x / k) for the VIS-only HR target; ``x`` has shape ``(..., 1)``."""
    return tf.asinh(x / _hr_stretch_scale())


def inverse_asinh_stretch_lr(y: tf.Tensor) -> tf.Tensor:
    """Inverse of :func:`asinh_stretch_lr` (per-band)."""
    return tf.sinh(y) * _lr_stretch_scale()


def inverse_asinh_stretch_hr(y: tf.Tensor) -> tf.Tensor:
    """Inverse of :func:`asinh_stretch_hr` (VIS only)."""
    return tf.sinh(y) * _hr_stretch_scale()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MultiBandEuclidDataset:
    """Reads paired v2 (clean HR-VIS, dirty LR-4channel) records.

    Up to three data sources feed training. :meth:`dataset` streams the
    primary synthetic source alone as ``(lr, hr)`` 2-tuples (the
    pure-supervised path + every validation stream).
    :meth:`dataset_fixed_layout` combines the sources into **fixed
    contiguous-block batches** ``[n_syn | n_hst | n_rt]`` for
    :meth:`Trainer.train_step_sky`, which slices each lane by its static
    count (no per-example source tags, no random interleaving):

      * **synthetic** (always present): the primary
        ``records_dir`` with full ``(lr, hr)`` pairs from the simulated
        forward model.
      * **HST** (optional): ``hst_records_dir`` + ``hst_fraction`` —
        forward-modelled HST→Euclid pairs from
        ``scripts/fasrc_generate_hst_tfrecords.py``. ``hr`` is the
        observed HST image; ``train_step_sky`` supervises ``H ⊛ SR``
        against it.
      * **round-trip** (optional): ``roundtrip_records_dir`` +
        ``roundtrip_fraction`` — *real* Euclid LR cutouts from
        ``scripts/fasrc_generate_euclid_roundtrip_tfrecords.py``. These
        records have **no HR side** (a dummy zero HR rides along, never
        read); ``train_step_sky`` applies the self-supervised round-trip
        loss ``|asinh(rebin(E ⊛ SR)) - lr_vis|``.

    Parameters
    ----------
    subset
        ``'train'`` or ``'validate'``.
    records_dir
        Directory containing ``clean_{subset}.tfrecord`` etc.
    scale
        Super-resolution factor (HR pixel scale / LR pixel scale).
        For our pipeline this is 2 (0.05″ HR / 0.10″ LR).
    hr_patch_size
        HR crop size during training (96 by default).
    hst_records_dir
        Optional secondary records dir containing HST-derived pairs.
        Has no effect when ``hst_fraction == 0`` or the directory is
        missing the right TFRecords.
    hst_fraction
        Per-example draw weight for the HST source.
    roundtrip_records_dir
        Optional tertiary records dir containing LR-only Euclid
        cutouts. Has no effect when ``roundtrip_fraction == 0`` or
        the ``dirty_{subset}.tfrecord`` is missing.
    roundtrip_fraction
        Per-example draw weight for the round-trip source.
        ``hst_fraction + roundtrip_fraction`` must be ≤ 1; the
        remainder is the synthetic weight.
    """

    def __init__(
        self,
        subset: str = "train",
        records_dir: str = Config.RECORDS_DIR_V2,
        scale: int = Config.DEFAULT_REBIN_FACTOR,
        hr_patch_size: int = Config.DEFAULT_HR_CROP_SIZE,
        hst_records_dir: Optional[str] = None,
        hst_fraction: float = 0.0,
        roundtrip_records_dir: Optional[str] = None,
        roundtrip_fraction: float = 0.0,
    ):

        if subset not in ("train", "validate"):
            raise ValueError("subset must be 'train' or 'validate'")
        if not (0.0 <= hst_fraction <= 1.0):
            raise ValueError(f"hst_fraction must be in [0, 1], got {hst_fraction}")
        if not (0.0 <= roundtrip_fraction <= 1.0):
            raise ValueError(
                f"roundtrip_fraction must be in [0, 1], got {roundtrip_fraction}"
            )
        if hst_fraction + roundtrip_fraction > 1.0 + 1e-6:
            raise ValueError(
                f"hst_fraction ({hst_fraction}) + roundtrip_fraction "
                f"({roundtrip_fraction}) must be ≤ 1 — the remainder "
                "is the synthetic weight"
            )
        self.scale              = int(scale)
        self.hr_patch_size      = int(hr_patch_size)
        self.subset             = subset
        self.hst_fraction       = float(hst_fraction)
        self.roundtrip_fraction = float(roundtrip_fraction)

        self.clean_file, self.dirty_file = self._resolve_pair(
            records_dir, subset,
        )

        # Secondary HST source: optional. When the directory exists and
        # ``hst_fraction > 0``, ``dataset()`` interleaves the two streams.
        self.hst_clean_file: Optional[str] = None
        self.hst_dirty_file: Optional[str] = None
        if hst_records_dir is not None and self.hst_fraction > 0:
            try:
                self.hst_clean_file, self.hst_dirty_file = self._resolve_pair(
                    hst_records_dir, subset,
                )
            except FileNotFoundError:
                # Fall back to single-source cleanly — better to keep
                # training than to abort with a missing-records error.
                self.hst_clean_file = None
                self.hst_dirty_file = None

        # Tertiary round-trip source: LR-only, no clean/HR side. Resolve
        # only the dirty file. Same lenient fallback as HST so a missing
        # tertiary doesn't kill the training run.
        self.roundtrip_dirty_file: Optional[str] = None
        if roundtrip_records_dir is not None and self.roundtrip_fraction > 0:
            rt_dirty = tfrecord_path(roundtrip_records_dir, f"dirty_{subset}")
            if _os.path.exists(rt_dirty):
                self.roundtrip_dirty_file = rt_dirty

    @staticmethod
    def _resolve_pair(records_dir: str, subset: str) -> tuple[str, str]:
        """Return ``(hr_file, dirty_file)`` for one records dir / subset."""
        hr_candidate = tfrecord_path(records_dir, f"hr_{subset}")
        legacy_clean = tfrecord_path(records_dir, f"clean_{subset}")
        dirty        = tfrecord_path(records_dir, f"dirty_{subset}")
        if _os.path.exists(hr_candidate):
            hr_file = hr_candidate
        elif _os.path.exists(legacy_clean):
            hr_file = legacy_clean
        else:
            raise FileNotFoundError(
                f"No HR/clean TFRecord found in {records_dir} for {subset}"
            )
        if not _os.path.exists(dirty):
            raise FileNotFoundError(f"No dirty TFRecord found at {dirty}")
        return hr_file, dirty

    # ------------------------------------------------------------------ #

    def _build_single_source(
        self, dirty_file: str, clean_file: str,
        *, random_transform: bool, repeat_count: Optional[int],
        cache: bool,
    ) -> tf.data.Dataset:
        """One (dirty, clean) → (lr, hr) tf.data pipeline, pre-batch."""
        n_lr = Config.NUM_LR_CHANNELS
        n_hr = Config.NUM_HR_CHANNELS

        def _parse_lr(raw):
            return asinh_stretch_lr(parse_record_graph_v2(raw, n_lr))

        def _parse_hr(raw):
            return asinh_stretch_hr(parse_record_graph_v2(raw, n_hr))

        dirty_ds = tf.data.TFRecordDataset(dirty_file).map(
            _parse_lr, num_parallel_calls=AUTOTUNE,
        )
        clean_ds = tf.data.TFRecordDataset(clean_file).map(
            _parse_hr, num_parallel_calls=AUTOTUNE,
        )
        ds = tf.data.Dataset.zip((dirty_ds, clean_ds))
        if cache:
            ds = ds.cache()

        if random_transform:
            hr_patch = self.hr_patch_size
            scale    = self.scale
            ds = ds.shuffle(buffer_size=200)
            ds = ds.map(
                lambda lr, hr: _augment_multiband(lr, hr, hr_patch, scale),
                num_parallel_calls=AUTOTUNE,
            )
        return ds.repeat(repeat_count)

    def _build_roundtrip_source(
        self, dirty_file: str,
        *, random_transform: bool, repeat_count: Optional[int],
        cache: bool,
    ) -> tf.data.Dataset:
        """LR-only round-trip stream → ``(lr, dummy_hr)``.

        Round-trip records have no HR side; ``train_step_sky`` computes
        the round-trip loss entirely on LR + the model's own forward
        reconstruction. ``dummy_hr`` is a zero-tensor at the supervised
        HR patch shape (training) or scaled-up from the LR record shape
        (validation); it is never read by the loss (the round-trip lane
        is identified by its fixed position in the batch, not a tag).
        """
        n_lr = Config.NUM_LR_CHANNELS

        def _parse_lr(raw):
            return asinh_stretch_lr(parse_record_graph_v2(raw, n_lr))

        ds = tf.data.TFRecordDataset(dirty_file).map(
            _parse_lr, num_parallel_calls=AUTOTUNE,
        )
        if cache:
            ds = ds.cache()

        scale = self.scale
        if random_transform:
            lr_patch = self.hr_patch_size // scale
            hr_patch = self.hr_patch_size

            def _crop_lr_with_dummy_hr(lr):
                lr_h = tf.shape(lr)[0]
                lr_w = tf.shape(lr)[1]
                # Same aligned-crop policy as ``_augment_multiband`` —
                # snap to a multiple of ``scale`` so the dummy HR's
                # virtual top-left aligns with the LR crop.
                max_x = (lr_h - lr_patch) // 1 * 1
                max_y = (lr_w - lr_patch) // 1 * 1
                lr_x = tf.random.uniform([], 0, max_x + 1, dtype=tf.int32)
                lr_y = tf.random.uniform([], 0, max_y + 1, dtype=tf.int32)
                lr_c = lr[lr_x : lr_x + lr_patch, lr_y : lr_y + lr_patch, :]
                hr_c = tf.zeros(
                    [hr_patch, hr_patch, Config.NUM_HR_CHANNELS],
                    dtype=lr.dtype,
                )
                return lr_c, hr_c

            ds = ds.shuffle(buffer_size=200)
            ds = ds.map(_crop_lr_with_dummy_hr, num_parallel_calls=AUTOTUNE)
        else:
            # Validation / inference: keep records at native size, build
            # a shape-matched dummy HR by upscaling the LR shape.
            def _attach_dummy_hr(lr):
                lr_h = tf.shape(lr)[0]
                lr_w = tf.shape(lr)[1]
                hr_c = tf.zeros(
                    [lr_h * scale, lr_w * scale, Config.NUM_HR_CHANNELS],
                    dtype=lr.dtype,
                )
                return lr, hr_c
            ds = ds.map(_attach_dummy_hr, num_parallel_calls=AUTOTUNE)

        return ds.repeat(repeat_count)

    def dataset_fixed_layout(
        self,
        n_syn: int,
        n_hst: int,
        n_rt: int,
        *,
        random_transform: bool = True,
        repeat_count: Optional[int] = None,
    ) -> tf.data.Dataset:
        """Fixed contiguous-block batches for :meth:`Trainer.train_step_sky`.

        Every batch is laid out as ``[n_syn synthetic | n_hst HST |
        n_rt round-trip]`` — exactly these per-lane counts, in this
        order, no source tags. The trainer slices each lane by its
        static count and applies that lane's forward op, so there is no
        per-example branching. Each lane is shuffled and batched
        independently (``drop_remainder=True`` for static shapes), then
        the per-lane batches are concatenated along the batch axis.

        A lane with count 0 is omitted. Requesting a lane whose records
        weren't configured (``n_hst>0`` without HST records, ``n_rt>0``
        without round-trip records) is an error — unlike the lenient
        single-source fallback, a fixed layout must get what it asks for.
        Yields ``(lr, hr)`` 2-tuples with ``lr`` ``[B, h, w, 4]`` and
        ``hr`` ``[B, H, W, 1]`` where ``B = n_syn + n_hst + n_rt``.
        """
        lanes: list = []
        if n_syn > 0:
            syn = self._build_single_source(
                self.dirty_file, self.clean_file,
                random_transform=random_transform, repeat_count=repeat_count,
                cache=True,
            )
            lanes.append(syn.batch(n_syn, drop_remainder=True))
        if n_hst > 0:
            if self.hst_clean_file is None or self.hst_dirty_file is None:
                raise ValueError(
                    "dataset_fixed_layout: n_hst > 0 but no HST records are "
                    "configured (pass hst_records_dir and hst_fraction > 0)"
                )
            hst = self._build_single_source(
                self.hst_dirty_file, self.hst_clean_file,
                random_transform=random_transform, repeat_count=repeat_count,
                cache=True,
            )
            lanes.append(hst.batch(n_hst, drop_remainder=True))
        if n_rt > 0:
            if self.roundtrip_dirty_file is None:
                raise ValueError(
                    "dataset_fixed_layout: n_rt > 0 but no round-trip records "
                    "are configured (pass roundtrip_records_dir and "
                    "roundtrip_fraction > 0)"
                )
            rt = self._build_roundtrip_source(
                self.roundtrip_dirty_file,
                random_transform=random_transform, repeat_count=repeat_count,
                cache=True,
            )
            lanes.append(rt.batch(n_rt, drop_remainder=True))

        if not lanes:
            raise ValueError("dataset_fixed_layout: all lane counts are 0")
        if len(lanes) == 1:
            return lanes[0].prefetch(AUTOTUNE)

        def _concat_blocks(*blocks):
            lrs = [lr for lr, _ in blocks]
            hrs = [hr for _, hr in blocks]
            return tf.concat(lrs, axis=0), tf.concat(hrs, axis=0)

        zipped = tf.data.Dataset.zip(tuple(lanes))
        return zipped.map(
            _concat_blocks, num_parallel_calls=AUTOTUNE,
        ).prefetch(AUTOTUNE)

    def dataset(
        self,
        batch_size: int = Config.DEFAULT_BATCH_SIZE,
        random_transform: bool = True,
        repeat_count: Optional[int] = None,
    ) -> tf.data.Dataset:
        """Build the single-source streaming ``tf.data.Dataset``.

        Yields ``(lr, hr)`` 2-tuples from the primary (synthetic)
        records — the pure-supervised path used by ``run_pipeline.py``,
        the CLI, the web inference helpers, and every validation stream.
        Multi-source training (synthetic + HST + round-trip) goes through
        :meth:`dataset_fixed_layout`, which lays the sources out in fixed
        contiguous blocks for :meth:`Trainer.train_step_sky` instead of
        randomly interleaving them.
        """
        primary = self._build_single_source(
            self.dirty_file, self.clean_file,
            random_transform=random_transform, repeat_count=repeat_count,
            cache=True,
        )
        return primary.batch(batch_size).prefetch(AUTOTUNE)


def lr_only_dataset(dirty_file: str, *, batch_size: int) -> tf.data.Dataset:
    """LR-only validation stream from a ``dirty_{subset}.tfrecord``.

    Round-trip records carry no HR side, so the supervised
    :func:`MultiBandEuclidDataset.dataset` builder doesn't fit them. This
    standalone helper reads the dirty file, applies the same per-band
    asinh stretch the training path uses, batches, and prefetches — no
    shuffle, no repeat. Yields LR tensors only, shape ``[B, H, W, 4]``,
    suitable for :meth:`Trainer.evaluate_roundtrip`.
    """
    n_lr = Config.NUM_LR_CHANNELS

    def _parse_lr(raw):
        return asinh_stretch_lr(parse_record_graph_v2(raw, n_lr))

    ds = tf.data.TFRecordDataset(dirty_file).map(
        _parse_lr, num_parallel_calls=AUTOTUNE,
    )
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def _augment_multiband(
    lr: tf.Tensor, hr: tf.Tensor, hr_patch_size: int, scale: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Random aligned LR/HR crop.

    Flips and rotations are intentionally disabled: the empirical VIS
    ePSF is non-symmetric, so a flipped HR target is not what you would
    obtain by convolving the flipped clean field with the same PSF.
    """
    lr_patch_size = hr_patch_size // scale
    hr_h = tf.shape(hr)[0]
    hr_w = tf.shape(hr)[1]

    max_x = (hr_h - hr_patch_size) // scale * scale
    max_y = (hr_w - hr_patch_size) // scale * scale
    hr_x = tf.random.uniform([], 0, max_x + 1, dtype=tf.int32)
    hr_y = tf.random.uniform([], 0, max_y + 1, dtype=tf.int32)
    hr_x = hr_x // scale * scale
    hr_y = hr_y // scale * scale

    hr = hr[hr_x : hr_x + hr_patch_size, hr_y : hr_y + hr_patch_size, :]
    lr_x = hr_x // scale
    lr_y = hr_y // scale
    lr = lr[lr_x : lr_x + lr_patch_size, lr_y : lr_y + lr_patch_size, :]
    return lr, hr
