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

# Source-tag constants for the (lr, hr, source) tuples emitted when
# ``with_source_tag=True``. The trainer routes per-example loss off
# these — keep them in sync with ``Trainer._SOURCE_*`` and with the
# round-trip handling in ``Trainer.train_step``.
SOURCE_SYNTHETIC = 0
SOURCE_HST       = 1
SOURCE_ROUNDTRIP = 2


class MultiBandEuclidDataset:
    """Reads paired v2 (clean HR-VIS, dirty LR-4channel) records.

    Up to three data sources can be mixed at the per-example level via
    :func:`tf.data.Dataset.sample_from_datasets`:

      * **synthetic** (always present): the primary
        ``records_dir`` with full ``(lr, hr)`` pairs from the simulated
        forward model.
      * **HST** (optional): ``hst_records_dir`` + ``hst_fraction`` —
        forward-modelled HST→Euclid pairs from
        ``scripts/fasrc_generate_hst_tfrecords.py``.
      * **round-trip** (optional): ``roundtrip_records_dir`` +
        ``roundtrip_fraction`` — *real* Euclid LR cutouts from
        ``scripts/fasrc_generate_euclid_roundtrip_tfrecords.py``. These
        records have **no HR side**; the trainer detects them via the
        per-example source tag and switches to the self-supervised
        round-trip loss ``|asinh(Conv(M(lr))/k) - lr|`` instead of the
        supervised L1.

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
        import os as _os

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
        import os as _os
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
        """LR-only round-trip stream → ``(lr, dummy_hr, source=ROUNDTRIP)``.

        Round-trip records have no HR side; the trainer's round-trip
        loss is computed entirely on LR + the model's own forward
        reconstruction. We still emit a 3-tuple so the sampled mix is
        shape-homogeneous; ``dummy_hr`` is a zero-tensor at the
        supervised HR patch shape (training) or scaled-up from the LR
        record shape (validation). The trainer masks it out via the
        source tag so the zeros never contribute to the loss.
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

        src = tf.constant(SOURCE_ROUNDTRIP, dtype=tf.int32)
        ds = ds.map(lambda lr, hr: (lr, hr, src),
                    num_parallel_calls=AUTOTUNE)
        return ds.repeat(repeat_count)

    @staticmethod
    def _attach_source_tag(
        ds: tf.data.Dataset, source_id: int,
    ) -> tf.data.Dataset:
        """Map ``(lr, hr) → (lr, hr, source_id)`` so streams are 3-tuple."""
        src = tf.constant(int(source_id), dtype=tf.int32)
        return ds.map(lambda lr, hr: (lr, hr, src),
                      num_parallel_calls=AUTOTUNE)

    def dataset(
        self,
        batch_size: int = Config.DEFAULT_BATCH_SIZE,
        random_transform: bool = True,
        repeat_count: Optional[int] = None,
        with_source_tag: bool = False,
    ) -> tf.data.Dataset:
        """Build the streaming ``tf.data.Dataset``.

        When ``with_source_tag=False`` (default — preserves the old API
        across the existing callers), yields ``(lr, hr)`` 2-tuples. When
        ``True``, yields ``(lr, hr, source)`` 3-tuples with the source
        tag in ``{SOURCE_SYNTHETIC, SOURCE_HST, SOURCE_ROUNDTRIP}``. The
        round-trip trainer requires source tags to route per-example
        loss; setting ``roundtrip_fraction > 0`` without
        ``with_source_tag=True`` is an error because the dummy HR would
        otherwise look like real ground truth.

        Up to three streams are mixed via
        :func:`tf.data.Dataset.sample_from_datasets`. Weights:

          * synthetic = ``1 - hst_fraction - roundtrip_fraction``
          * HST       = ``hst_fraction``       (if configured & valid)
          * round-trip= ``roundtrip_fraction`` (if configured & valid)

        A configured-but-missing secondary/tertiary source silently
        falls back to the remaining streams (its weight is dropped) —
        same lenient behaviour as the pre-round-trip code.
        """
        if self.roundtrip_fraction > 0 and not with_source_tag:
            raise ValueError(
                "roundtrip_fraction > 0 requires with_source_tag=True so "
                "the trainer can detect round-trip examples; the dummy "
                "HR placeholder would otherwise be silently treated as "
                "ground truth"
            )

        primary = self._build_single_source(
            self.dirty_file, self.clean_file,
            random_transform=random_transform, repeat_count=repeat_count,
            cache=True,
        )

        # Collect (stream, weight, source_id) for each active source.
        # Weights track what the *intended* fractions are; we adjust for
        # any source that falls back to "missing" (re-normalising the
        # remaining weights so they still sum to 1).
        streams: list = [(primary, 1.0, SOURCE_SYNTHETIC)]

        if (self.hst_clean_file is not None
            and self.hst_dirty_file is not None
            and self.hst_fraction > 0):
            secondary = self._build_single_source(
                self.hst_dirty_file, self.hst_clean_file,
                random_transform=random_transform, repeat_count=repeat_count,
                cache=True,
            )
            streams.append((secondary, self.hst_fraction, SOURCE_HST))

        if (self.roundtrip_dirty_file is not None
            and self.roundtrip_fraction > 0):
            tertiary = self._build_roundtrip_source(
                self.roundtrip_dirty_file,
                random_transform=random_transform, repeat_count=repeat_count,
                cache=True,
            )
            # Round-trip stream already carries its own source tag —
            # build_single_source doesn't yet, so we have to wrap the
            # primary/secondary below to match.
            streams.append((tertiary, self.roundtrip_fraction, SOURCE_ROUNDTRIP))

        # Attach source tags to the synthetic/HST streams (the round-
        # trip helper already attaches its own). Always 3-tuple
        # internally; we strip back to 2-tuple at the end if the caller
        # asked for the old API.
        tagged_streams: list = []
        for stream, weight, src_id in streams:
            if src_id == SOURCE_ROUNDTRIP:
                tagged_streams.append((stream, weight))
            else:
                tagged_streams.append(
                    (self._attach_source_tag(stream, src_id), weight),
                )

        if len(tagged_streams) == 1:
            mixed = tagged_streams[0][0]
        else:
            # Primary weight = 1 − (sum of secondary weights).
            primary_weight = max(
                0.0, 1.0 - sum(w for _, w in tagged_streams[1:])
            )
            weights = [primary_weight] + [w for _, w in tagged_streams[1:]]
            mixed = tf.data.Dataset.sample_from_datasets(
                [s for s, _ in tagged_streams],
                weights=weights,
                stop_on_empty_dataset=False,
            )

        if not with_source_tag:
            # Backward-compat: drop the source tag for existing callers
            # that destructure ``lr, hr = batch``.
            mixed = mixed.map(
                lambda lr, hr, src: (lr, hr),
                num_parallel_calls=AUTOTUNE,
            )

        return mixed.batch(batch_size).prefetch(AUTOTUNE)


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
