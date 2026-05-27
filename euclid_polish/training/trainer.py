"""
Trainer module for WDSR super-resolution models.

This module provides the Trainer class for training WDSR models. It
supports two batch formats:

  * 2-tuple ``(lr, hr)`` — the pre-existing supervised path. Used by
    ``scripts/run_pipeline.py``, ``cli/main.py``, the web inference
    helpers, etc. Loss = ``MeanAbsoluteError(sr, hr)``.
  * 3-tuple ``(lr, hr, source)`` — emitted when the dataset is built
    with ``with_source_tag=True``. ``source`` is a per-example int32
    tensor; the trainer routes the loss per element:

      - ``SOURCE_SYNTHETIC`` / ``SOURCE_HST``: supervised L1
        (``|sr - hr|`` in asinh space, as before).
      - ``SOURCE_ROUNDTRIP``: self-supervised reconstruction
        ``|asinh(Conv(inverse_asinh(sr)) / k) - lr_vis|`` — the
        synthetic forward op (PSF + 2× sum-rebin, deterministic, no
        noise) takes the model's HR prediction back down to LR and
        compares against the input LR's VIS channel.

The split is per *example*, not per batch: a single batch can carry a
mix of supervised + round-trip examples, and gradients are computed in
one tape pass.
"""
import csv
import os
import time

import tensorflow as tf

from tf_keras.losses import MeanAbsoluteError
from tf_keras.metrics import Mean
from tf_keras.optimizers import Adam
from tf_keras.optimizers.schedules import PiecewiseConstantDecay
from tqdm import tqdm

from euclid_polish.config import Config
from euclid_polish.training.data_multiband import (
    SOURCE_ROUNDTRIP, asinh_stretch_hr, inverse_asinh_stretch_hr,
)
from euclid_polish.training.models.common import evaluate

# Append-only CSV — one row per evaluate_every batch — readable by Excel,
# pandas, the FASRC dashboard, and the in-tree plot_training_log helper
# without any custom parser. Validation history persists in real time so
# a job killed mid-training still leaves a usable log behind.
TRAINING_LOG_FILENAME = "training_log.csv"
TRAINING_LOG_COLUMNS  = (
    "step", "wall_time", "loss",
    "psnr_stretched", "psnr_raw",
    "gnorm_avg", "gnorm_max", "clip_norm", "duration_s",
)

# Gradient clipping by global L2 norm — see ``Config.GRAD_CLIP_NORM``.
# Direction-preserving; has no effect when natural gradient norm < clip
# value. Set ``Config.GRAD_CLIP_NORM = math.inf`` to disable.
GRAD_CLIP_NORM = float(Config.GRAD_CLIP_NORM)


class Trainer:
    """Trainer for WDSR super-resolution models."""

    def __init__(
        self,
        model,
        loss=MeanAbsoluteError(),
        learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-3, 5e-4]),
        checkpoint_dir='./ckpt/wdsr',
        forward_op=None,
        roundtrip_loss_weight: float = 1.0,
    ):
        """
        Initialize the trainer.

        Parameters
        ----------
        model : tf.keras.Model
            WDSR model to train.
        loss : tf.keras.losses.Loss
            Loss function for supervised batches (synthetic + HST).
        learning_rate : tf.keras.optimizers.schedules.LearningRateSchedule
            Learning rate schedule.
        checkpoint_dir : str
            Directory for saving checkpoints.
        forward_op : tf.keras.layers.Layer, optional
            Differentiable Euclid VIS forward op (PSF + 2× sum-rebin)
            used for round-trip examples. ``None`` (default) disables
            the round-trip path even if source tags arrive — in that
            case round-trip examples fall through to the supervised
            L1, which compares ``sr`` against the dummy zeros ``hr``;
            this is rarely what you want, so configure the forward op
            whenever ``roundtrip_fraction > 0`` on the dataset.
        roundtrip_loss_weight : float
            Multiplier on the per-example round-trip loss before
            averaging with the supervised loss. Default 1.0 makes the
            two losses contribute equally per example (so a 60/20/20
            synthetic/HST/round-trip mix produces an 80/20 supervised/
            round-trip gradient contribution by batch). Bump above 1 to
            up-weight the round-trip path; set to 0 to disable while
            keeping the round-trip dataset wired (useful for ablations).
        """
        self.now = None
        self.loss = loss
        self.forward_op = forward_op
        self.roundtrip_loss_weight = float(roundtrip_loss_weight)
        # Cache the SOURCE_ROUNDTRIP constant on the trainer so the
        # @tf.function below doesn't capture a Python int that'd force
        # a retrace if its value ever changed.
        self._source_roundtrip = tf.constant(SOURCE_ROUNDTRIP, dtype=tf.int32)
        # ``psnr`` tracks the best PSNR_stretched seen so far (used by
        # save-best). max_val for PSNR is set in models/common.py from
        # Config.PSNR_PEAK_STRETCHED ≈ asinh(mag-17 star / k).
        self.checkpoint = tf.train.Checkpoint(
            step=tf.Variable(0),
            psnr=tf.Variable(-1.0),
            optimizer=Adam(learning_rate),
            model=model,
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=checkpoint_dir,
            max_to_keep=3,
        )

        self.restore()

    @property
    def model(self):
        """Get the model."""
        return self.checkpoint.model

    def train(
        self,
        train_dataset,
        valid_dataset,
        steps=300000,
        evaluate_every=1000,
        save_best_only=True,
        validate_images=Config.DEFAULT_VALIDATE_IMAGES,
        step_callback=None,
        step_callback_every=50,
    ):
        """
        Train the model.

        Parameters:
        -----------
        train_dataset : tf.data.Dataset
            Training dataset.
        valid_dataset : tf.data.Dataset
            Validation dataset.
        steps : int
            Number of training steps.
        evaluate_every : int
            Evaluate every N steps.
        save_best_only : bool
            Only save checkpoints when PSNR (stretched) improves.
        validate_images : int
            Max number of validation images to evaluate on during training.
        step_callback : Optional[Callable[[int, int], None]]
            Called as ``step_callback(current_step, total_steps)`` every
            ``step_callback_every`` steps. Used to feed an external
            progress reporter (e.g. the JSONL events file the FASRC
            scripts write via :class:`Reporter`) without coupling the
            trainer to it. ``None`` (default) → no callback.
        step_callback_every : int
            Cadence of ``step_callback`` invocations. 50 keeps the
            JSONL events file small on a 200k-step run (~4k lines) while
            still updating the UI's progress bar every ~5 s of wall
            time at typical step rates.
        """
        loss_mean = Mean()
        gnorm_mean = Mean()
        gnorm_max  = tf.Variable(0.0, trainable=False)

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        start_step = int(ckpt.step.numpy())
        remaining = steps - start_step

        log_path = os.path.join(ckpt_mgr.directory, TRAINING_LOG_FILENAME)
        os.makedirs(ckpt_mgr.directory, exist_ok=True)

        pbar = tqdm(
            train_dataset.take(remaining),
            total=remaining,
            initial=0,
            desc="Training",
            unit="step",
            ncols=120,
        )

        self.now = time.perf_counter()

        for batch in pbar:
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()
            # Backward-compatible dispatch: 2-tuple → supervised-only
            # (pre-round-trip behaviour, used by run_pipeline.py /
            # cli/main.py / the web inference helpers); 3-tuple →
            # mixed source-aware path used by the round-trip trainer.
            # tf.function specialises on signature, so each branch
            # compiles independently and the Python-level switch is
            # free per batch.
            if isinstance(batch, tuple) and len(batch) == 3:
                lr, hr, source = batch
                loss, gnorm = self.train_step_mixed(lr, hr, source)
            else:
                lr, hr = batch
                loss, gnorm = self.train_step(lr, hr)
            loss_mean(loss)
            gnorm_mean(gnorm)
            gnorm_max.assign(tf.maximum(gnorm_max, gnorm))

            if step % 50 == 0:
                pbar.set_postfix(loss=f"{loss.numpy():.4f}", refresh=False)

            # External progress callback (e.g. the JSONL events file).
            # Cadence-gated so a 200k-step run doesn't write 200k JSONL
            # lines; the first step always fires so "did training start?"
            # is answerable immediately. ``int(step)`` because tf returns
            # a numpy int64 here and the callback's typed contract is
            # plain Python int.
            if step_callback is not None and (
                step == start_step + 1 or step % step_callback_every == 0
            ):
                step_callback(int(step), int(steps))

            if step % evaluate_every == 0:
                loss_value  = loss_mean.result()
                gnorm_avg   = gnorm_mean.result()
                gnorm_peak  = float(gnorm_max.numpy())
                loss_mean.reset_state()
                gnorm_mean.reset_state()
                gnorm_max.assign(0.0)

                # Compute validation PSNR (stretched: loss-aligned, used for
                # save-best; raw: photometric).
                metrics  = self.evaluate(valid_dataset.take(validate_images))
                psnr_str = float(metrics["psnr_stretched"].numpy())
                psnr_raw = float(metrics["psnr_raw"].numpy())

                duration = time.perf_counter() - self.now
                pbar.set_postfix(
                    loss=f"{loss_value.numpy():.3f}",
                    PSNRs=f"{psnr_str:.2f}",
                    PSNRr=f"{psnr_raw:.2f}",
                )
                tqdm.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Step {step}/{steps}: loss = {loss_value.numpy():.4f}, "
                    f"PSNR(str/raw) = {psnr_str:.3f}/{psnr_raw:.3f} dB, "
                    f"|g| avg/max = {gnorm_avg.numpy():.3g}/{gnorm_peak:.3g} "
                    f"({duration:.2f}s)"
                )

                # Persist for later plotting. Append-only CSV so each row
                # is durable the moment ``evaluate_every`` fires — a job
                # OOM-killed mid-training still leaves a complete log.
                row = {
                    "step":           int(step),
                    "wall_time":      time.time(),
                    "loss":           float(loss_value.numpy()),
                    "psnr_stretched": psnr_str,
                    "psnr_raw":       psnr_raw,
                    "gnorm_avg":      float(gnorm_avg.numpy()),
                    "gnorm_max":      float(gnorm_peak),
                    "clip_norm":      float(GRAD_CLIP_NORM),
                    "duration_s":     float(duration),
                }
                write_header = (not os.path.exists(log_path)
                                or os.path.getsize(log_path) == 0)
                with open(log_path, "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=TRAINING_LOG_COLUMNS)
                    if write_header:
                        w.writeheader()
                    w.writerow(row)

                # save-best on PSNR_stretched (loss-aligned).
                if save_best_only and metrics["psnr_stretched"] <= ckpt.psnr:
                    self.now = time.perf_counter()
                    continue

                ckpt.psnr = metrics["psnr_stretched"]
                ckpt_mgr.save()
                tqdm.write(
                    f"  ✓ Checkpoint saved (PSNR str={psnr_str:.3f}, "
                    f"raw={psnr_raw:.3f} dB)"
                )

                self.now = time.perf_counter()

        pbar.close()

    @tf.function
    def train_step(self, lr, hr):
        """
        Perform one supervised training step (pre-round-trip API).

        Returns
        -------
        loss_value : tf.Tensor
            Loss for this batch.
        gnorm : tf.Tensor
            Global L2 norm of the gradient *before* clipping (useful for
            monitoring; ``GRAD_CLIP_NORM`` is the rescaled magnitude actually
            applied).
        """
        with tf.GradientTape() as tape:
            sr = self.checkpoint.model(lr, training=True)
            loss_value = self.loss(sr, hr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        gradients, gnorm = tf.clip_by_global_norm(gradients, clip_norm=GRAD_CLIP_NORM)
        self.checkpoint.optimizer.apply_gradients(
            zip(gradients, self.checkpoint.model.trainable_variables)
        )

        return loss_value, gnorm

    @tf.function
    def train_step_mixed(self, lr, hr, source):
        """
        Source-aware training step for heterogeneous batches.

        Per element of the batch, the loss is chosen from the source tag:

          * ``source ≠ SOURCE_ROUNDTRIP`` → supervised
            ``|sr - hr|`` (asinh space, ``hr`` is real ground truth).
          * ``source == SOURCE_ROUNDTRIP`` → round-trip
            ``|asinh(Conv(inverse_asinh(sr)) / k) - lr_vis|`` (asinh
            space, ``hr`` is dummy zeros and never enters the loss).

        The two per-example loss vectors are masked + summed, then
        normalised by the batch size to keep the scalar loss
        independent of the source mix. ``forward_op`` MUST be set when
        any round-trip examples can arrive — otherwise the round-trip
        path silently degrades to comparing ``sr`` against the dummy
        zeros, which would push the model toward outputting zeros for
        round-trip-tagged batches.

        Returns
        -------
        loss_value, gnorm : tf.Tensor
            Same semantics as :meth:`train_step`.
        """
        with tf.GradientTape() as tape:
            sr = self.checkpoint.model(lr, training=True)

            # Supervised L1 per example over (H, W, C) → shape [B].
            # ``reduce_mean`` over the spatial+channel axes keeps the
            # per-pixel scale comparable to the round-trip term below.
            sup_per_example = tf.reduce_mean(tf.abs(sr - hr), axis=[1, 2, 3])

            if self.forward_op is None:
                # No forward op installed — fall back to supervised for
                # *all* examples (dummy HR poisons the round-trip ones,
                # but the trainer can't compute the round-trip loss
                # without the op). Caller should configure the op
                # whenever the dataset is built with roundtrip_fraction>0.
                per_example = sup_per_example
            else:
                # Round-trip loss in asinh space (matches the supervised
                # loss space so the per-example magnitudes are
                # comparable). The VIS asinh knee on both sides
                # cancels the stretch's scale dependence.
                sr_linear        = inverse_asinh_stretch_hr(sr)
                lr_recon_linear  = self.forward_op(sr_linear)
                lr_recon_stretch = asinh_stretch_hr(lr_recon_linear)
                lr_vis           = lr[..., 0:1]
                rt_per_example   = tf.reduce_mean(
                    tf.abs(lr_recon_stretch - lr_vis), axis=[1, 2, 3],
                )
                is_rt = tf.equal(source, self._source_roundtrip)
                per_example = tf.where(
                    is_rt,
                    self.roundtrip_loss_weight * rt_per_example,
                    sup_per_example,
                )

            loss_value = tf.reduce_mean(per_example)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        gradients, gnorm = tf.clip_by_global_norm(gradients, clip_norm=GRAD_CLIP_NORM)
        self.checkpoint.optimizer.apply_gradients(
            zip(gradients, self.checkpoint.model.trainable_variables)
        )

        return loss_value, gnorm

    def evaluate(self, dataset):
        """
        Evaluate the model on a dataset.

        Parameters:
        -----------
        dataset : tf.data.Dataset
            Dataset to evaluate on.

        Returns:
        --------
        metrics : dict
            See ``models.common.evaluate`` — keys are ``psnr_stretched``
            and ``psnr_raw``.
        """
        return evaluate(self.checkpoint.model, dataset)

    def restore(self):
        """Restore model from checkpoint if available."""
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(
                f"Model restored from checkpoint at step {self.checkpoint.step.numpy()}."
            )
