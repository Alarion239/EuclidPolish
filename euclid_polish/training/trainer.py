"""
Trainer module for WDSR super-resolution models.

This module provides the Trainer class for training WDSR models.
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
    ):
        """
        Initialize the trainer.

        Parameters:
        -----------
        model : tf.keras.Model
            WDSR model to train.
        loss : tf.keras.losses.Loss
            Loss function to use.
        learning_rate : tf.keras.optimizers.schedules.LearningRateSchedule
            Learning rate schedule.
        checkpoint_dir : str
            Directory for saving checkpoints.
        """
        self.now = None
        self.loss = loss
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

        for lr, hr in pbar:
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()
            loss, gnorm = self.train_step(lr, hr)
            loss_mean(loss)
            gnorm_mean(gnorm)
            gnorm_max.assign(tf.maximum(gnorm_max, gnorm))

            if step % 50 == 0:
                pbar.set_postfix(loss=f"{loss.numpy():.4f}", refresh=False)

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
        Perform one training step.

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
