"""The :class:`Model` operator — the public face of a trained checkpoint.

Wraps :func:`~euclid_polish.training.inference.load_model_from_checkpoint`
and :func:`~euclid_polish.training.inference.reconstruct`. ``upsample`` consumes
and produces plain :class:`~euclid_polish.image.Image` objects (the leaf data
atom), so the image layer never needs to import this module.
"""

from __future__ import annotations

import glob as _glob
import os as _os
from collections.abc import Callable

import numpy as np
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

from euclid_polish.config import Config
from euclid_polish.eval import catalog_runner, grouped_runner
from euclid_polish.image import Image, ImageSet, Role
from euclid_polish.image.tfio import parse_example
from euclid_polish.provenance.checkpoint import model_id_of_checkpoint
from euclid_polish.provenance.defaults import mint_id
from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.records import Stamp
from euclid_polish.training.augmentation import (
    _augment_multiband,
    asinh_stretch_hr,
    asinh_stretch_lr,
)
from euclid_polish.training.inference import (
    load_model_from_checkpoint as _default_load,
)
from euclid_polish.training.inference import (
    reconstruct as _default_reconstruct,
)
from euclid_polish.training.models.wdsr import wdsr as _wdsr_build
from euclid_polish.training.trainer import Trainer, seed_everything

_HR_SCALE = Config.DEFAULT_PIXEL_SCALE   # 0.05 arcsec/pix


def _checkpoint_exists(checkpoint_dir: str) -> bool:
    """Return True when a TF checkpoint is present in ``checkpoint_dir``."""
    return (
        _os.path.isfile(_os.path.join(checkpoint_dir, "checkpoint"))
        or bool(_glob.glob(_os.path.join(checkpoint_dir, "*.index")))
    )


class Model:
    """The public face of a trained WDSR checkpoint.

    Parameters
    ----------
    checkpoint_dir : str
        Path to the TF checkpoint directory.
    scale : int
        Super-resolution upscale factor (default 2).
    num_res_blocks : int
        Number of residual blocks (default ``Config.DEFAULT_NUM_RES_BLOCKS``).
    seed : int, optional
        Master RNG seed. When set, the global RNGs are seeded *before* the model
        is built (so weight initialisation is reproducible) and the seed flows
        into :meth:`train`, which records it on a ``Process.training`` provenance
        record. ``None`` (default) keeps the entropy-driven behaviour. See
        :func:`~euclid_polish.training.trainer.seed_everything` for the GPU
        determinism caveat.
    deterministic : bool
        Enable TF op-determinism for bit-exact GPU replay (slower).
    _load_fn, _reconstruct_fn : callable, optional
        Test injection points replacing ``load_model_from_checkpoint`` /
        ``reconstruct``.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        scale: int = 2,
        num_res_blocks: int = Config.DEFAULT_NUM_RES_BLOCKS,
        seed: int | None = None,
        deterministic: bool = False,
        _load_fn: Callable | None = None,
        _reconstruct_fn: Callable | None = None,
    ) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._scale = scale
        self._num_res_blocks = num_res_blocks
        self._seed = seed
        self._deterministic = bool(deterministic)
        # Seed BEFORE building so a from-scratch model's weight init is
        # reproducible; train() re-asserts the seed for the data pipeline.
        if seed is not None:
            seed_everything(seed, deterministic=self._deterministic)
        self._load_fn: Callable = _load_fn if _load_fn is not None else _default_load

        if _load_fn is not None or _checkpoint_exists(checkpoint_dir):
            self._tf_model = self._load_fn(checkpoint_dir, scale, num_res_blocks)
            self.id: ProvId | None = model_id_of_checkpoint(checkpoint_dir)
        else:
            # Fresh build (no checkpoint to introspect): match the current
            # multi-band pipeline (VIS+NISP → VIS+NISP). The wdsr default is
            # 1-channel (legacy VIS-only); without this a from-scratch model —
            # e.g. each fresh ensemble member_NN/ dir — builds a 1-channel net
            # and dies on 4-band data ("expected shape (…, 1), found (…, 4)").
            # Resumed checkpoints introspect their own nchan in load_model_from_
            # checkpoint, so they are unaffected.
            self._tf_model = _wdsr_build(
                scale=scale, num_res_blocks=num_res_blocks,
                nchan_in=Config.NUM_LR_CHANNELS,
                nchan_out=Config.NUM_HR_CHANNELS)
            self.id = None

        self._reconstruct_fn: Callable = (
            _reconstruct_fn if _reconstruct_fn is not None else _default_reconstruct
        )

    # -- training interface --

    def _build_training_pipeline(
        self,
        dirty_path: str,
        clean_path: str,
        batch_size: int,
        *,
        augment: bool = True,
    ):
        """TF data pipeline: TFRecord → parse → [crop] → asinh stretch → batch.

        The asinh stretch is applied AFTER the random crop, so the (per-band,
        elementwise) transcendental runs on the small training patch rather than
        the full field — ~28x less work for a 96px crop out of a 512px field.
        ``asinh`` is elementwise, so this is bit-identical to stretching first:
        ``asinh(crop(x)) == crop(asinh(x))``. With ``augment=False`` (validation)
        there is no crop, so the full field is stretched as before.
        """
        n_lr = Config.NUM_LR_CHANNELS
        n_hr = Config.NUM_HR_CHANNELS

        def _parse_lr(raw):
            return parse_example(raw, n_lr)

        def _parse_hr(raw):
            return parse_example(raw, n_hr)

        lr_ds = (tf.data.TFRecordDataset(dirty_path)
                 .map(_parse_lr, num_parallel_calls=AUTOTUNE))
        hr_ds = (tf.data.TFRecordDataset(clean_path)
                 .map(_parse_hr, num_parallel_calls=AUTOTUNE))
        ds = tf.data.Dataset.zip((lr_ds, hr_ds))
        if augment:
            scale = self._scale
            hr_crop = Config.DEFAULT_HR_CROP_SIZE

            def _crop_then_stretch(lr, hr):
                lr, hr = _augment_multiband(lr, hr, hr_crop, scale)
                return asinh_stretch_lr(lr), asinh_stretch_hr(hr)

            ds = (ds.shuffle(200)
                    .map(_crop_then_stretch, num_parallel_calls=AUTOTUNE)
                    .repeat())
        else:
            ds = ds.map(lambda lr, hr: (asinh_stretch_lr(lr), asinh_stretch_hr(hr)),
                        num_parallel_calls=AUTOTUNE)
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    def train(
        self,
        lr_path: str,
        hr_path: str,
        steps: int = 300_000,
        batch_size: int = 16,
        **kwargs,
    ) -> None:
        """Train the model on TFRecord files at ``lr_path`` and ``hr_path``.

        Validation paths are derived by replacing ``_train.`` with
        ``_validate.`` in the filenames. All extra ``kwargs`` are forwarded
        verbatim to :class:`~euclid_polish.training.trainer.Trainer.train`
        (e.g. ``evaluate_every``, ``step_callback``, ``eval_callback``).
        """
        # Re-assert the seed before the data pipeline is defined so its shuffle
        # / augmentation draws are reproducible (the model was already seeded +
        # built in __init__).
        if self._seed is not None:
            seed_everything(self._seed, deterministic=self._deterministic)
        train_ds = self._build_training_pipeline(lr_path, hr_path, batch_size)
        valid_lr_path = lr_path.replace("_train.", "_validate.")
        valid_hr_path = hr_path.replace("_train.", "_validate.")
        if valid_lr_path == lr_path:
            raise ValueError(
                "Cannot derive validation path: expected _train. in filename "
                f"(got {lr_path!r})"
            )
        valid_ds = self._build_training_pipeline(
            valid_lr_path, valid_hr_path, batch_size, augment=False
        )

        trainer = Trainer(self._tf_model, checkpoint_dir=self._checkpoint_dir,
                          seed=self._seed, deterministic=self._deterministic)
        trainer.train(train_ds, valid_ds, steps=steps, **kwargs)

        if _checkpoint_exists(self._checkpoint_dir):
            self._tf_model = self._load_fn(
                self._checkpoint_dir, self._scale, self._num_res_blocks
            )
            self.id = model_id_of_checkpoint(self._checkpoint_dir)

    # -- inference interface --

    def upsample_array(self, arr: np.ndarray) -> np.ndarray:
        """Super-resolve a bare numpy array (raw electrons); return the SR array.

        Input is 2D ``(H, W)`` or 3D ``(H, W, C)``. Output is
        ``(H·scale, W·scale)`` (single-output) or ``(H·scale, W·scale, C)``
        (4-band model).
        """
        _lr_display, sr_data = self._reconstruct_fn(self._tf_model, arr)
        return np.asarray(sr_data, dtype=np.float32)

    def upsample(self, lr: Image, *, store=None) -> Image:
        """Super-resolve an LR :class:`Image` into an SR Image (role ``'sr'``).

            sr = model.upsample(lr)

        The SR image's stamp parents are ``(self.id, lr's id)`` (``self.id`` is
        omitted when ``None`` — a legacy checkpoint with no provenance sidecar;
        ``lr``'s id comes from its embedded stamp when present). The new id is
        minted via ``store`` (or ``default_store()``), guarded so a store
        failure degrades to an unstamped-but-correct artifact.
        """
        _lr_display, sr_data = self._reconstruct_fn(self._tf_model, lr.data)
        sr_data = np.asarray(sr_data, dtype=np.float32)
        bands = lr.band_names if sr_data.ndim == 3 and sr_data.shape[-1] == len(lr.band_names) else ("VIS",)
        lr_id = lr.stamp.id if lr.stamp is not None else None
        parents = tuple(p for p in (self.id, lr_id) if p is not None)
        return Image(
            data=sr_data, pixel_scale_arcsec=_HR_SCALE, band_names=bands,
            is_clean=True, role=Role.SR, subset=lr.subset,
            stamp=Stamp(id=mint_id(store), parents=parents, schema_version=3,
                        subset=lr.subset))

    def upsample_batch(
        self,
        lr_images: ImageSet,
        *,
        on_progress: Callable | None = None,
        log: Callable | None = None,
    ) -> ImageSet:
        """Super-resolve each image in ``lr_images``; return an eager :class:`ImageSet`.

        Streams ``lr_images`` lazily — if it is backed by a TFRecord, only one
        image is held in memory at a time. The returned set is eager (all SR
        results materialised in a list).
        """
        emit = log or (lambda m: None)
        sr_list = []
        for i, lr in enumerate(lr_images):
            sr = self.upsample(lr)
            sr_list.append(sr)
            emit(f"SR idx {i} → {sr.data.shape}")
            if on_progress:
                on_progress(i + 1, None, f"idx {i}")
        return ImageSet.from_images(sr_list)

    # -- evaluation interface --

    def eval_catalog(self, *, out_dir, **kwargs):
        """Evaluate this model on a lens catalog.

        Thin wrapper over
        :func:`~euclid_polish.eval.catalog_runner.run_catalog_eval`, passing
        this model's loaded TF graph so it is not re-loaded from disk. All
        other keyword args (``catalog_path``, ``checkpoint``, ``cutout_size``,
        ``grade``, ``max_n``, ``render``, ``on_progress``, ``log`` …) pass
        straight through. Returns the run summary dict.
        """
        return catalog_runner.run_catalog_eval(
            out_dir=out_dir, model=self, **kwargs)

    def eval_grouped(self, out_dir, n, **kwargs):
        """Run grouped evaluation (lens grades A/B/C + real galaxies +
        synthetic) with this model.

        Thin wrapper over
        :func:`~euclid_polish.eval.grouped_runner.run_grouped_analysis`,
        passing this model's loaded TF graph so it is not re-loaded. All
        other kwargs (``catalog_path``, ``checkpoint``, ``include_synthetic``,
        ``include_galaxies``, ``seed``, ``on_progress``, ``log`` …) pass
        through. Returns the run summary dict.
        """
        return grouped_runner.run_grouped_analysis(
            out_dir, n, model=self, **kwargs)
