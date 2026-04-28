"""
Trainer module for WDSR super-resolution models.

This module provides the Trainer class for training WDSR models.
"""
import json
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

TRAINING_LOG_FILENAME = "training_log.jsonl"


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
            Only save checkpoints when PSNR improves.
        validate_images : int
            Max number of validation images to evaluate on during training.
        """
        loss_mean = Mean()

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
            loss = self.train_step(lr, hr)
            loss_mean(loss)

            if step % 50 == 0:
                pbar.set_postfix(loss=f"{loss.numpy():.4f}", refresh=False)

            if step % evaluate_every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_state()

                # Compute PSNR on validation dataset (capped at validate_images)
                psnr_value = self.evaluate(valid_dataset.take(validate_images))

                duration = time.perf_counter() - self.now
                pbar.set_postfix(
                    loss=f"{loss_value.numpy():.3f}",
                    PSNR=f"{psnr_value.numpy():.3f}",
                )
                tqdm.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Step {step}/{steps}: loss = {loss_value.numpy():.4f}, "
                    f"PSNR = {psnr_value.numpy():.3f} ({duration:.2f}s)"
                )

                # Persist (step, loss, PSNR) for later plotting. Append-only
                # so multiple training sessions accumulate into one log.
                with open(log_path, "a") as fh:
                    fh.write(json.dumps({
                        "step":      int(step),
                        "loss":      float(loss_value.numpy()),
                        "psnr":      float(psnr_value.numpy()),
                        "duration_s": float(duration),
                        "wall_time": time.time(),
                    }) + "\n")

                if save_best_only and psnr_value <= ckpt.psnr:
                    self.now = time.perf_counter()
                    continue

                ckpt.psnr = psnr_value
                ckpt_mgr.save()
                tqdm.write(f"  ✓ Checkpoint saved (PSNR: {psnr_value.numpy():.3f})")

                self.now = time.perf_counter()

        pbar.close()

    @tf.function
    def train_step(self, lr, hr):
        """
        Perform one training step.

        Parameters:
        -----------
        lr : tf.Tensor
            Low-resolution input.
        hr : tf.Tensor
            High-resolution target.

        Returns:
        --------
        loss_value : tf.Tensor
            Loss value.
        """
        with tf.GradientTape() as tape:
            sr = self.checkpoint.model(lr, training=True)
            loss_value = self.loss(sr, hr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(
            zip(gradients, self.checkpoint.model.trainable_variables)
        )

        return loss_value

    def evaluate(self, dataset):
        """
        Evaluate the model on a dataset.

        Parameters:
        -----------
        dataset : tf.data.Dataset
            Dataset to evaluate on.

        Returns:
        --------
        psnr_value : tf.Tensor
            Mean PSNR value.
        """
        return evaluate(self.checkpoint.model, dataset)

    def restore(self):
        """Restore model from checkpoint if available."""
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(
                f"Model restored from checkpoint at step {self.checkpoint.step.numpy()}."
            )
