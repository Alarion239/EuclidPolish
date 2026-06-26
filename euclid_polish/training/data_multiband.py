"""
Multi-band data loader for training.

Reads v2 TFRecords produced by the new pipeline:

  * ``dirty_{subset}.tfrecord`` — (H_lr, W_lr, 4) LR float32 electrons,
    band order :attr:`Config.LR_INPUT_BAND_NAMES`.
  * ``hr_{subset}.tfrecord`` / ``clean_{subset}.tfrecord`` —
    (H_hr, W_hr, 4) HR float32 electrons, the clean 4-band target
    (band order :attr:`Config.HR_TARGET_BAND_NAMES` = the LR order).
    Records written before the 4-band-output change carry 1-channel
    VIS-only HR and must be regenerated — the in-graph channel assert
    fails loudly on them.

The EXPERIMENTAL HST / star-anchor lane records keep a 1-channel HR side
(the observed F814W image / the VIS delta-target); those lanes parse it
as 1 channel and zero-pad to ``NUM_HR_CHANNELS`` so fixed-layout batches
concatenate — the trainer compares only channel 0 (VIS) for them.

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
from euclid_polish.image.tfio import parse_record_graph_v2, tfrecord_path

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
    [Config.get_band(name).asinh_stretch_scale_e
     for name in Config.HR_TARGET_BAND_NAMES],
    dtype=np.float32,
)  # shape (4,) — one knee per HR target band


def _lr_stretch_scale() -> np.ndarray:
    """Length-4 vector of asinh stretch scales, one per LR channel."""
    return _LR_STRETCH_SCALE_NP


def _hr_stretch_scale() -> np.ndarray:
    """Length-4 vector of asinh stretch scales, one per HR target band.

    Broadcasting note: a stretched/unstretched tensor with FEWER trailing
    channels (the 1-channel HST / star-anchor HR sides, or a VIS-only
    slice) still works — numpy/TF broadcasting aligns trailing dims, so
    ``(..., 1) / (4,)`` would NOT be valid; those callers slice the scale
    via :func:`asinh_stretch_hr`'s ``num_channels`` argument instead.
    """
    return _HR_STRETCH_SCALE_NP


def asinh_stretch_lr(x: tf.Tensor) -> tf.Tensor:
    """asinh(x / k) per channel; ``x`` has shape ``(..., 4)``."""
    return tf.asinh(x / _lr_stretch_scale())


def _hr_scale_for(x: tf.Tensor, num_channels: "int | None") -> np.ndarray:
    """The per-band knee vector sliced to ``x``'s channel count.

    The leading HR bands are ``(VIS, Y_E, J_E, H_E)``, so a C-channel
    tensor (C ≤ 4) always means "the first C bands": full 4-band targets,
    the 1-channel HST / star-anchor HR sides, and VIS-only slices all
    resolve correctly. Slicing (instead of relying on broadcasting) is
    load-bearing — broadcasting ``(..., 1)`` against the length-4 vector
    would silently widen the tensor to 4 channels.
    """
    n = num_channels if num_channels is not None else x.shape[-1]
    k = _hr_stretch_scale()
    return k[:int(n)] if n is not None else k


def asinh_stretch_hr(x: tf.Tensor, num_channels: "int | None" = None) -> tf.Tensor:
    """asinh(x / k) per HR band; ``x`` has shape ``(..., C)``, C ≤ 4.

    The knee count follows ``x``'s static channel dim (or an explicit
    ``num_channels`` when that dim is dynamic).
    """
    return tf.asinh(x / _hr_scale_for(x, num_channels))


def inverse_asinh_stretch_lr(y: tf.Tensor) -> tf.Tensor:
    """Inverse of :func:`asinh_stretch_lr` (per-band)."""
    return tf.sinh(y) * _lr_stretch_scale()


def inverse_asinh_stretch_hr(y: tf.Tensor, num_channels: "int | None" = None) -> tf.Tensor:
    """Inverse of :func:`asinh_stretch_hr` (per-band)."""
    return tf.sinh(y) * _hr_scale_for(y, num_channels)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MultiBandEuclidDataset:
    """Reads paired v2 (clean 4-band HR, dirty 4-band LR) records.

    Up to three data sources feed training. :meth:`dataset` streams the
    primary synthetic source alone as ``(lr, hr)`` 2-tuples (the
    pure-supervised path + every validation stream).
    :meth:`dataset_fixed_layout` combines the sources into **fixed
    contiguous-block batches** ``[n_syn | n_hst | n_anchor]`` for
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
      * **star-anchor** (optional): ``anchor_records_dir`` — real Euclid
        star cutouts from
        ``scripts/fasrc_generate_star_anchor_tfrecords.py``, paired with a
        sparse HR delta-target (one pixel = the catalog VIS flux at the
        star). Structurally a ``(dirty, hr)`` pair like the synthetic lane;
        ``train_step_sky`` supervises ``SR`` against the delta but **masks
        the loss to the star pixel** — operator-free (no PSF).

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
    anchor_records_dir
        Optional tertiary records dir containing star-anchor pairs
        (``dirty_anchor_{subset}`` + ``hr_anchor_{subset}``). Has no
        effect when ``n_anchor == 0`` or the records are missing.
    anchor_fraction
        Retained for the constructor's ≤1 sum-validation / back-compat;
        ``hst_fraction + anchor_fraction`` must be ≤ 1. The actual lane
        size is driven by ``n_anchor`` in ``dataset_fixed_layout``.
    """

    def __init__(
        self,
        subset: str = "train",
        records_dir: str = Config.RECORDS_DIR_V2,
        scale: int = Config.DEFAULT_REBIN_FACTOR,
        hr_patch_size: int = Config.DEFAULT_HR_CROP_SIZE,
        hst_records_dir: Optional[str] = None,
        hst_fraction: float = 0.0,
        anchor_records_dir: Optional[str] = None,
        anchor_fraction: float = 0.0,
        vis_only: bool = False,
    ):

        if subset not in ("train", "validate"):
            raise ValueError("subset must be 'train' or 'validate'")
        if not (0.0 <= hst_fraction <= 1.0):
            raise ValueError(f"hst_fraction must be in [0, 1], got {hst_fraction}")
        if not (0.0 <= anchor_fraction <= 1.0):
            raise ValueError(
                f"anchor_fraction must be in [0, 1], got {anchor_fraction}"
            )
        if hst_fraction + anchor_fraction > 1.0 + 1e-6:
            raise ValueError(
                f"hst_fraction ({hst_fraction}) + anchor_fraction "
                f"({anchor_fraction}) must be ≤ 1 — the remainder "
                "is the synthetic weight"
            )
        self.scale              = int(scale)
        self.hr_patch_size      = int(hr_patch_size)
        self.subset             = subset
        self.hst_fraction       = float(hst_fraction)
        self.anchor_fraction    = float(anchor_fraction)
        # VIS-only: feed the model just the VIS channel (index 0) instead of
        # the full VIS+NISP stack. Records still store all 4 LR channels; we
        # slice to VIS in-graph after the per-band asinh stretch. Applies to
        # every lane (syn/HST/anchor) since all route through
        # ``_build_single_source``.
        self.vis_only           = bool(vis_only)

        self.clean_file, self.dirty_file = self._resolve_pair(
            records_dir, subset,
        )

        # Secondary HST source: optional. Resolved whenever a records dir
        # is given; ``dataset_fixed_layout`` decides per-run whether to use
        # it (via its ``n_hst`` count). ``hst_fraction`` is retained only
        # for the constructor's ≤1 sum-validation and back-compat callers.
        self.hst_clean_file: Optional[str] = None
        self.hst_dirty_file: Optional[str] = None
        if hst_records_dir is not None:
            try:
                self.hst_clean_file, self.hst_dirty_file = self._resolve_pair(
                    hst_records_dir, subset,
                )
            except FileNotFoundError:
                # Fall back cleanly — better to surface "no HST records"
                # at dataset_fixed_layout than to abort construction.
                self.hst_clean_file = None
                self.hst_dirty_file = None

        # Tertiary star-anchor source: real Euclid star cutouts + a sparse
        # HR delta-target (one pixel = catalog flux). Structurally a
        # (dirty, hr) pair like the synthetic lane, so it reuses
        # ``_build_single_source``; the trainer masks the loss to the star
        # pixel. Resolved whenever the records dir is given; same lenient
        # fallback as HST.
        self.anchor_dirty_file: Optional[str] = None
        self.anchor_clean_file: Optional[str] = None
        if anchor_records_dir is not None:
            a_dirty = tfrecord_path(anchor_records_dir, f"dirty_anchor_{subset}")
            a_hr    = tfrecord_path(anchor_records_dir, f"hr_anchor_{subset}")
            if _os.path.exists(a_dirty) and _os.path.exists(a_hr):
                self.anchor_dirty_file = a_dirty
                self.anchor_clean_file = a_hr

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
        cache: bool, hr_channels: Optional[int] = None,
    ) -> tf.data.Dataset:
        """One (dirty, clean) → (lr, hr) tf.data pipeline, pre-batch.

        ``hr_channels`` is how many channels the HR record stores —
        default the full :attr:`Config.NUM_HR_CHANNELS` (synthetic 4-band
        target). The EXPERIMENTAL HST / star-anchor lanes pass ``1``
        (observed F814W image / VIS delta-target); their HR side is
        zero-padded back to ``NUM_HR_CHANNELS`` so fixed-layout batches
        concatenate across lanes — the trainer only compares channel 0
        (VIS) for those lanes, so the padding never carries gradient.
        """
        n_lr = Config.NUM_LR_CHANNELS
        n_hr_total = Config.NUM_HR_CHANNELS
        n_hr = int(hr_channels) if hr_channels is not None else n_hr_total
        vis_only = self.vis_only

        def _parse_lr(raw):
            # Always parse the 4 stored channels and stretch with the
            # per-band (length-4) scale, then slice VIS (channel 0) when
            # vis_only so VIS keeps its own asinh knee.
            lr = asinh_stretch_lr(parse_record_graph_v2(raw, n_lr))
            return lr[..., :1] if vis_only else lr

        def _parse_hr(raw):
            hr = asinh_stretch_hr(parse_record_graph_v2(raw, n_hr),
                                  num_channels=n_hr)
            if vis_only:
                # VIS-only model: 1-channel target (channel 0 = VIS).
                return hr[..., :1]
            if n_hr < n_hr_total:
                # EXPERIMENTAL lane record with a 1-channel HR side —
                # zero-pad (asinh(0) = 0) up to the full band count.
                pad = tf.zeros(tf.concat(
                    [tf.shape(hr)[:-1], [n_hr_total - n_hr]], axis=0),
                    dtype=hr.dtype)
                return tf.concat([hr, pad], axis=-1)
            return hr

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

    def dataset_fixed_layout(
        self,
        n_syn: int,
        n_hst: int,
        n_anchor: int,
        *,
        random_transform: bool = True,
        repeat_count: Optional[int] = None,
    ) -> tf.data.Dataset:
        """Fixed contiguous-block batches for :meth:`Trainer.train_step_sky`.

        Every batch is laid out as ``[n_syn synthetic | n_hst HST |
        n_anchor star-anchor]`` — exactly these per-lane counts, in this
        order, no source tags. The trainer slices each lane by its
        static count, so there is no per-example branching. Each lane is
        shuffled and batched independently (``drop_remainder=True`` for
        static shapes), then the per-lane batches are concatenated along
        the batch axis.

        The star-anchor lane is a ``(dirty_anchor, hr_anchor)`` pair like
        the synthetic lane — real Euclid star cutouts + a sparse HR
        delta-target — so it reuses ``_build_single_source``; the trainer
        masks its loss to the (single) star pixel.

        A lane with count 0 is omitted. Requesting a lane whose records
        weren't configured (``n_hst>0`` without HST records, ``n_anchor>0``
        without star-anchor records) is an error — unlike the lenient
        single-source fallback, a fixed layout must get what it asks for.
        Yields ``(lr, hr)`` 2-tuples with ``lr`` ``[B, h, w, 4]`` and
        ``hr`` ``[B, H, W, 4]`` where ``B = n_syn + n_hst + n_anchor``
        (the experimental lanes' 1-channel HR sides are zero-padded).
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
            # EXPERIMENTAL lane: the HST HR side is the 1-channel observed
            # F814W image (zero-padded to the full band count in-graph).
            hst = self._build_single_source(
                self.hst_dirty_file, self.hst_clean_file,
                random_transform=random_transform, repeat_count=repeat_count,
                cache=True, hr_channels=1,
            )
            lanes.append(hst.batch(n_hst, drop_remainder=True))
        if n_anchor > 0:
            if self.anchor_dirty_file is None or self.anchor_clean_file is None:
                raise ValueError(
                    "dataset_fixed_layout: n_anchor > 0 but no star-anchor "
                    "records are configured (pass anchor_records_dir)"
                )
            # EXPERIMENTAL lane: the anchor HR side is the 1-channel VIS
            # delta-target (zero-padded to the full band count in-graph;
            # the pad stays outside the trainer's ``target > 0`` mask).
            anc = self._build_single_source(
                self.anchor_dirty_file, self.anchor_clean_file,
                random_transform=random_transform, repeat_count=repeat_count,
                cache=True, hr_channels=1,
            )
            lanes.append(anc.batch(n_anchor, drop_remainder=True))

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
        hr_channels: Optional[int] = None,
    ) -> tf.data.Dataset:
        """Build the single-source streaming ``tf.data.Dataset``.

        Yields ``(lr, hr)`` 2-tuples from the primary (synthetic)
        records — the pure-supervised path used by ``run_pipeline.py``,
        the CLI, the web inference helpers, and every validation stream.
        Multi-source training (synthetic + HST + star-anchor) goes through
        :meth:`dataset_fixed_layout`, which lays the sources out in fixed
        contiguous blocks for :meth:`Trainer.train_step_sky` instead of
        randomly interleaving them.

        ``hr_channels`` — channel count stored in the HR record; pass 1
        when pointing this at the EXPERIMENTAL HST records (their HR is
        the 1-channel observed F814W image, zero-padded in-graph).
        """
        primary = self._build_single_source(
            self.dirty_file, self.clean_file,
            random_transform=random_transform, repeat_count=repeat_count,
            cache=True, hr_channels=hr_channels,
        )
        return primary.batch(batch_size).prefetch(AUTOTUNE)

    def anchor_dataset(
        self,
        batch_size: int = 1,
        *,
        random_transform: bool = False,
        repeat_count: Optional[int] = 1,
    ) -> Optional[tf.data.Dataset]:
        """Star-anchor ``(lr, hr)`` stream for validation/monitoring.

        Yields the ``(dirty_anchor, hr_anchor)`` pair — ``hr`` is the sparse
        HR delta-target (zero except the star pixel). Used by
        :meth:`Trainer.evaluate_anchor`, which masks to the star pixel.
        Returns ``None`` when no star-anchor records are configured.
        """
        if self.anchor_dirty_file is None or self.anchor_clean_file is None:
            return None
        ds = self._build_single_source(
            self.anchor_dirty_file, self.anchor_clean_file,
            random_transform=random_transform, repeat_count=repeat_count,
            cache=False, hr_channels=1,
        )
        return ds.batch(batch_size).prefetch(AUTOTUNE)


def lr_only_dataset(dirty_file: str, *, batch_size: int) -> tf.data.Dataset:
    """LR-only stream from a ``dirty_{subset}.tfrecord``.

    For records that carry no HR side, the supervised
    :func:`MultiBandEuclidDataset.dataset` builder doesn't fit them. This
    standalone helper reads the dirty file, applies the same per-band
    asinh stretch the training path uses, batches, and prefetches — no
    shuffle, no repeat. Yields LR tensors only, shape ``[B, H, W, 4]``
    (e.g. for feeding real cutouts through the model at evaluation time).
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
