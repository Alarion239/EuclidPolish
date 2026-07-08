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
from euclid_polish.image import Image, ImageSet, Role
from euclid_polish.image.tfio import parse_example
from euclid_polish.provenance.checkpoint import model_id_of_checkpoint
from euclid_polish.provenance.defaults import mint_id
from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.records import Stamp
from euclid_polish.training.augmentation import (
    _augment_multiband,
    add_lr_noise,
    asinh_stretch_hr,
    asinh_stretch_lr,
    random_dihedral,
)
from euclid_polish.training.inference import (
    infer_checkpoint_num_res_blocks as _infer_num_res_blocks,
)
from euclid_polish.training.inference import (
    load_model_from_checkpoint as _default_load,
)
from euclid_polish.training.inference import (
    reconstruct as _default_reconstruct,
)
from euclid_polish.training.forward_onthefly import (
    DEFAULT_CROPS_PER_FIELD,
    OnTheFlyForward,
    member_psf_sets,
)
from euclid_polish.training.losses import build_loss
from euclid_polish.training.lr_schedule import WarmupCosineDecay
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
    init_weights_from : str, optional
        Checkpoint dir of an EXISTING model this one is forked from. The new
        member is built AS the source — its whole architecture (trunk depth,
        channel counts, skip structure) is introspected from the source
        checkpoint and the weights are copied verbatim. Requires a virgin
        ``checkpoint_dir`` — the trainer then starts at step 0 with a fresh
        optimizer and LR schedule.
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
        init_weights_from: str | None = None,
        icnr: bool = False,
        _load_fn: Callable | None = None,
        _reconstruct_fn: Callable | None = None,
    ) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._scale = scale
        self._num_res_blocks = num_res_blocks
        # ICNR init only shapes a FROM-SCRATCH build (below); fork/resume load
        # weights from a checkpoint, so it has no effect there.
        self._icnr = bool(icnr)
        self._seed = seed
        self._deterministic = bool(deterministic)
        # Seed BEFORE building so a from-scratch model's weight init is
        # reproducible; train() re-asserts the seed for the data pipeline.
        if seed is not None:
            seed_everything(seed, deterministic=self._deterministic)
        self._load_fn: Callable = _load_fn if _load_fn is not None else _default_load

        # Fork guard: starting from another member's weights only makes sense
        # for a NEW member — step 0, fresh optimizer, a fresh LR schedule.
        if init_weights_from is not None:
            if _checkpoint_exists(checkpoint_dir):
                raise ValueError(
                    "init_weights_from requires a virgin target dir; "
                    f"{checkpoint_dir!r} already has a checkpoint.")
            if not _checkpoint_exists(init_weights_from):
                raise ValueError(
                    f"init_weights_from: no checkpoint in {init_weights_from!r}")

        if _load_fn is not None or _checkpoint_exists(checkpoint_dir):
            self._tf_model = self._load_fn(checkpoint_dir, scale, num_res_blocks)
            # The load introspects the checkpoint's own depth (members of
            # different depths coexist in one ensemble) — reflect the actual
            # architecture, not the caller's default.
            actual = _infer_num_res_blocks(checkpoint_dir)
            if actual is not None:
                self._num_res_blocks = actual
            self.id: ProvId | None = model_id_of_checkpoint(checkpoint_dir)
        elif init_weights_from is not None:
            # Fork: build the new member AS the source model — loading the
            # source checkpoint directly preserves its WHOLE architecture
            # (trunk depth, channel counts, skip structure; the loader
            # introspects all of them) and copies the weights in one step.
            # Nothing else carries over — the trainer's ckpt.step stays 0
            # and the warmup→cosine LR schedule restarts.
            self._tf_model = self._load_fn(init_weights_from, scale,
                                           num_res_blocks)
            src_blocks = _infer_num_res_blocks(init_weights_from)
            if src_blocks is not None:
                if src_blocks != num_res_blocks:
                    print(f"  note: fork source has num_res_blocks="
                          f"{src_blocks} (requested {num_res_blocks}); "
                          "the source's architecture is preserved.")
                self._num_res_blocks = src_blocks
            self.id = None
            print(f"  ✓ fork: architecture + weights from {init_weights_from}")
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
                nchan_out=Config.NUM_HR_CHANNELS,
                icnr=self._icnr)
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
        noise_aug_rn: float = 0.0,
        bootstrap_keep: float | None = None,
        bootstrap_seed: int = 0,
    ):
        """TF data pipeline: TFRecord → parse → [bootstrap] → [crop + dihedral
        + noise] → asinh stretch → batch.

        The asinh stretch is applied AFTER the random crop, so the (per-band,
        elementwise) transcendental runs on the small training patch rather than
        the full field — ~28x less work for a 96px crop out of a 512px field.
        ``asinh`` is elementwise, so this is bit-identical to stretching first:
        ``asinh(crop(x)) == crop(asinh(x))``. With ``augment=False`` (validation)
        there is no crop/flip/noise, so the full field is stretched as before.

        Ensemble-diversity knobs (training only, all default-off/neutral):
        ``noise_aug_rn`` adds extra Gaussian noise to the LR patch in
        read-noise units (electrons, pre-stretch); ``bootstrap_keep`` ∈ (0, 1)
        keeps a deterministic pseudo-random subset of the training FIELDS —
        keyed by (``bootstrap_seed``, field index), so the same member always
        trains on the same subset across epochs and resumes.
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
            if bootstrap_keep is not None and 0.0 < float(bootstrap_keep) < 1.0:
                keep = float(bootstrap_keep)
                seed0 = int(bootstrap_seed) & 0x7FFFFFFF

                def _keep(i, _pair):
                    u = tf.random.stateless_uniform(
                        [], seed=tf.stack([tf.constant(seed0, tf.int32),
                                           tf.cast(i, tf.int32)]))
                    return u < keep

                ds = (ds.enumerate()
                        .filter(_keep)
                        .map(lambda _i, pair: pair))
            scale = self._scale
            hr_crop = Config.DEFAULT_HR_CROP_SIZE
            noise_rn = float(noise_aug_rn)

            def _crop_then_stretch(lr, hr):
                lr, hr = _augment_multiband(lr, hr, hr_crop, scale)
                lr, hr = random_dihedral(lr, hr)
                lr = add_lr_noise(lr, noise_rn)
                return asinh_stretch_lr(lr), asinh_stretch_hr(hr)

            ds = (ds.shuffle(200)
                    .map(_crop_then_stretch, num_parallel_calls=AUTOTUNE)
                    .repeat())
        else:
            ds = ds.map(lambda lr, hr: (asinh_stretch_lr(lr), asinh_stretch_hr(hr)),
                        num_parallel_calls=AUTOTUNE)
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    def _build_onthefly_pipeline(
        self,
        clean_path: str,
        batch_size: int,
        fwd: OnTheFlyForward,
        *,
        noise_aug_rn: float = 0.0,
        bootstrap_keep: float | None = None,
        bootstrap_seed: int = 0,
    ):
        """Training pipeline with the LIVE forward model: clean TFRecord →
        parse → [bootstrap] → full-field forward (PSF + rebin + noise +
        artifacts) → K crops → dihedral → asinh stretch → batch.

        The dirty records are never read — every visit of a field re-draws
        the PSF (from the member's bagged pool) and the noise realization on
        the WHOLE field, then cuts ``fwd.crops_per_field`` aligned crops, so
        out-of-crop bright stars contaminate the crops exactly as they would
        a real exposure (no truncated-kernel approximation). A batch of B
        examples costs B/K field forwards. Crops of one field share one
        exposure; epochs re-realize everything.
        """
        n_lr = Config.NUM_LR_CHANNELS
        n_hr = Config.NUM_HR_CHANNELS
        c, s, k = fwd.hr_crop_size, fwd.scale, fwd.crops_per_field

        ds = (tf.data.TFRecordDataset(clean_path)
              .map(lambda raw: parse_example(raw, n_hr),
                   num_parallel_calls=AUTOTUNE))
        if bootstrap_keep is not None and 0.0 < float(bootstrap_keep) < 1.0:
            keep = float(bootstrap_keep)
            seed0 = int(bootstrap_seed) & 0x7FFFFFFF

            def _keep(i, _field):
                u = tf.random.stateless_uniform(
                    [], seed=tf.stack([tf.constant(seed0, tf.int32),
                                       tf.cast(i, tf.int32)]))
                return u < keep

            ds = ds.enumerate().filter(_keep).map(lambda _i, f: f)

        def _fwd(field):
            lr, hr = tf.numpy_function(
                fwd.crops, [field], [tf.float32, tf.float32])
            lr.set_shape((k, c // s, c // s, n_lr))
            hr.set_shape((k, c, c, n_hr))
            return lr, hr

        noise_rn = float(noise_aug_rn)

        def _augment_then_stretch(lr, hr):
            lr, hr = random_dihedral(lr, hr)
            lr = add_lr_noise(lr, noise_rn)
            return asinh_stretch_lr(lr), asinh_stretch_hr(hr)

        # Field-level shuffle is small (each element is a full 510² field);
        # the crop-level shuffle after unbatch de-correlates the K siblings
        # of one field so a batch isn't K consecutive same-exposure crops —
        # sized to hold ≥16 fields' worth of crops whatever K is.
        return (ds.shuffle(32)
                  .map(_fwd, num_parallel_calls=AUTOTUNE)
                  .unbatch()
                  .shuffle(max(256, 4 * batch_size, 16 * k))
                  .map(_augment_then_stretch, num_parallel_calls=AUTOTUNE)
                  .repeat()
                  .batch(batch_size)
                  .prefetch(AUTOTUNE))

    def train(
        self,
        lr_path: str,
        hr_path: str,
        steps: int = 300_000,
        batch_size: int = 16,
        lr_peak: float = Config.LR_PEAK,
        lr_final: float = Config.LR_FINAL,
        lr_warmup_steps: int = Config.LR_WARMUP_STEPS,
        plateau_lr_enabled: bool = Config.PLATEAU_LR_ENABLED,
        plateau_lr_factor: float = Config.PLATEAU_LR_FACTOR,
        plateau_lr_patience: int = Config.PLATEAU_LR_PATIENCE,
        plateau_lr_min_delta: float = Config.PLATEAU_LR_MIN_DELTA,
        plateau_lr_min_delta_rel: float = Config.PLATEAU_LR_MIN_DELTA_REL,
        plateau_lr_cooldown: int = Config.PLATEAU_LR_COOLDOWN,
        plateau_lr_min_lr: float = Config.PLATEAU_LR_MIN_LR,
        plateau_lr_metric: str = Config.PLATEAU_LR_METRIC,
        plateau_rollback_min_gap: float = Config.PLATEAU_ROLLBACK_MIN_GAP,
        plateau_lr_recovery: bool = Config.PLATEAU_LR_RECOVERY,
        resume_track: str = "latest",
        loss_norm: str = "l1",
        noise_aug: float = 0.0,
        bootstrap: float | None = None,
        forward_onthefly: bool = False,
        psf_subset: int | None = None,
        crops_per_field: int = DEFAULT_CROPS_PER_FIELD,
        **kwargs,
    ) -> None:
        """Train the model on TFRecord files at ``lr_path`` and ``hr_path``.

        Validation paths are derived by replacing ``_train.`` with
        ``_validate.`` in the filenames. All extra ``kwargs`` are forwarded
        verbatim to :class:`~euclid_polish.training.trainer.Trainer.train`
        (e.g. ``evaluate_every``, ``step_callback``, ``eval_callback``).

        Ensemble-diversity knobs: ``loss_norm`` picks the reconstruction
        loss (``l1``/``l2``/``l3`` p-norms or ``berhu`` reverse-Huber, see
        :func:`~euclid_polish.training.losses.build_loss`); ``noise_aug`` adds
        extra LR noise in read-noise units; ``bootstrap`` ∈ (0, 1) trains on
        that fraction of the fields (deterministic subset keyed by the seed).

        ``forward_onthefly`` switches the TRAINING inputs to the live forward
        model (see :class:`~euclid_polish.training.forward_onthefly
        .OnTheFlyForward`): the dirty records are ignored; each visit of a
        clean field re-draws PSF + noise on the FULL field and cuts
        ``crops_per_field`` crops. ``psf_subset`` bags the member's PSF pool
        to that many source clusters (keyed by the seed).

        Validation always uses the record-based full set, no noise/forward
        randomness, and the L1 metric-space pipeline — members with
        different knobs stay comparable.
        """
        # Re-assert the seed before the data pipeline is defined so its shuffle
        # / augmentation draws are reproducible (the model was already seeded +
        # built in __init__).
        if self._seed is not None:
            seed_everything(self._seed, deterministic=self._deterministic)
        if forward_onthefly:
            psf_sets, note = member_psf_sets(seed=self._seed,
                                             psf_subset=psf_subset)
            print(f"  on-the-fly forward: {note} · "
                  f"{int(crops_per_field)} crops/field")
            fwd = OnTheFlyForward(psf_sets, seed=self._seed,
                                  crops_per_field=int(crops_per_field),
                                  scale=self._scale)
            train_ds = self._build_onthefly_pipeline(
                hr_path, batch_size, fwd,
                noise_aug_rn=float(noise_aug),
                bootstrap_keep=bootstrap,
                bootstrap_seed=(self._seed if self._seed is not None else 0))
        else:
            train_ds = self._build_training_pipeline(
                lr_path, hr_path, batch_size,
                noise_aug_rn=float(noise_aug),
                bootstrap_keep=bootstrap,
                bootstrap_seed=(self._seed if self._seed is not None else 0))
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

        # Warmup → cosine LR, scaled to THIS run's `steps`. The old flat 5e-4
        # head (piecewise, step-downs at 50%/80%) was too hot to settle, so
        # PSNR climbed then drifted back to the degenerate skip-only floor and
        # sat there until the fixed 50%-of-steps step-down — the long ~43.5 dB
        # plateau. Decaying smoothly from the start removes that flat region.
        # The warmup also tames the early gradient spikes on the regenerated
        # (saturation-masked) data. Both LR guards (spike + plateau) apply on
        # top of this schedule.
        lr_schedule = WarmupCosineDecay(
            peak_lr=lr_peak, final_lr=lr_final,
            warmup_steps=lr_warmup_steps, total_steps=steps,
        )
        trainer = Trainer(self._tf_model, learning_rate=lr_schedule,
                          checkpoint_dir=self._checkpoint_dir,
                          loss=build_loss(loss_norm),
                          seed=self._seed, deterministic=self._deterministic,
                          plateau_lr_enabled=plateau_lr_enabled,
                          plateau_lr_factor=plateau_lr_factor,
                          plateau_lr_patience=plateau_lr_patience,
                          plateau_lr_min_delta=plateau_lr_min_delta,
                          plateau_lr_min_delta_rel=plateau_lr_min_delta_rel,
                          plateau_lr_cooldown=plateau_lr_cooldown,
                          plateau_lr_min_lr=plateau_lr_min_lr,
                          plateau_lr_metric=plateau_lr_metric,
                          plateau_rollback_min_gap=plateau_rollback_min_gap,
                          plateau_lr_recovery=plateau_lr_recovery,
                          resume_track=resume_track)
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

    # NOTE: the eval interface lives on the runners (run_catalog_eval /
    # run_grouped_analysis) and loads the ENSEMBLE — Model is the internal
    # per-member implementation and is never evaluated directly anymore.
