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
    SOURCE_HST, SOURCE_ROUNDTRIP, asinh_stretch_hr, inverse_asinh_stretch_hr,
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
    # Multi-source validation (additive). Empty string when the
    # corresponding source dataset isn't wired for this run, so old
    # plots that parse-float these columns skip them instead of
    # choking on the text "nan".
    "psnr_stretched_hst", "psnr_raw_hst", "roundtrip_val_psnr",
    # The composite scalar save-best actually keys on (weighted blend of
    # the per-source metrics; see ``save_best_weights``). Logged so the
    # decision is auditable from the CSV alone.
    "save_best_score",
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
        hst_forward_op=None,
        synthetic_loss_weight: float = 1.0,
        hst_loss_weight: float = 1.0,
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
        synthetic_loss_weight, hst_loss_weight, roundtrip_loss_weight : float
            Per-source multipliers on the per-example loss before the
            batch mean. Each example is weighted by the knob matching its
            source tag (synthetic / HST / round-trip). All default to 1.0,
            which reproduces the previous behaviour (every example
            contributes equally; the round-trip term used to be the only
            one weighted). Raise a knob to up-weight that source's
            gradient contribution; set to 0 to keep the dataset wired but
            zero out its loss (ablation). The round-trip term is also
            gated on ``forward_op`` being set (see above) — without it,
            round-trip examples fall through to the (dummy-HR) supervised
            loss, so configure the op whenever ``roundtrip_fraction > 0``.
        """
        self.now = None
        self.loss = loss
        self.forward_op = forward_op
        # HST forward op (H ⊛ SR, no rebin) for the SR=sky objective —
        # the HST supervised loss compares H⊛SR to the observed HST image.
        self.hst_forward_op = hst_forward_op
        self.synthetic_loss_weight = float(synthetic_loss_weight)
        self.hst_loss_weight       = float(hst_loss_weight)
        self.roundtrip_loss_weight = float(roundtrip_loss_weight)
        # Cache the source-tag constants on the trainer so the
        # @tf.function below doesn't capture Python ints that'd force a
        # retrace if their values ever changed.
        self._source_roundtrip = tf.constant(SOURCE_ROUNDTRIP, dtype=tf.int32)
        self._source_hst       = tf.constant(SOURCE_HST, dtype=tf.int32)
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
        hst_valid_dataset=None,
        roundtrip_valid_dataset=None,
        save_best_weights=(1.0, 1.0, 0.0),
    ):
        """
        Train the model.

        Parameters:
        -----------
        train_dataset : tf.data.Dataset
            Training dataset.
        valid_dataset : tf.data.Dataset
            Validation dataset (synthetic). Drives save-best.
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
        hst_valid_dataset : Optional[tf.data.Dataset]
            HST ``(lr, hr)`` validation dataset. When provided, each
            evaluation also records HST PSNR (stretched/raw) so a regime
            change's effect on the HST source is visible even when the
            synthetic metric dips. Purely additive — does NOT influence
            save-best. ``None`` (default) leaves the HST columns empty.
        roundtrip_valid_dataset : Optional[tf.data.Dataset]
            LR-only round-trip validation dataset (see
            :func:`euclid_polish.training.data_multiband.lr_only_dataset`).
            When provided, each evaluation records the mean round-trip
            reconstruction loss. Requires ``self.forward_op`` to be set;
            otherwise the logged value is ``nan``. Additive only — does
            NOT influence save-best. ``None`` leaves the column empty.
        save_best_weights : tuple[float, float, float]
            ``(w_syn, w_hst, w_rt)`` weights for the composite save-best
            score, higher = better:

                score = w_syn·PSNR_syn + w_hst·PSNR_hst + w_rt·PSNR_rt

            All three terms are now PSNR in dB, so they share one scale
            and the round-trip term is ADDED like the others (no scale-gap
            fudge factor needed). Note the round-trip PSNR is taken at LR
            resolution, so it tends to sit higher in absolute dB than the
            HR-resolution synthetic/HST PSNRs — keep that in mind when
            weighting. A term drops out automatically when its validation
            dataset is absent (e.g. ``w_hst`` has no effect when
            ``hst_valid_dataset is None``). Default ``(1, 1, 0)`` →
            synthetic + HST PSNR, equally weighted, round-trip
            monitored-only. With HST/RT absent this reduces to plain
            synthetic-PSNR save-best (backwards compatible).
        """
        loss_mean = Mean()
        gnorm_mean = Mean()
        gnorm_max  = tf.Variable(0.0, trainable=False)

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        start_step = int(ckpt.step.numpy())
        remaining = steps - start_step

        # The first validation of *this* run always force-saves and
        # re-baselines ckpt.psnr (see the save-best block below).
        # ``ckpt.psnr`` is checkpointed, so a run resumed under a
        # changed training regime (e.g. a new HST / round-trip mix)
        # inherits the *previous* regime's best — which the new mix
        # typically can't beat on the synthetic metric — and would
        # otherwise never write a checkpoint, discarding the new
        # training. Local to this call, so it resets every resubmit.
        first_eval_this_run = True

        log_path = os.path.join(ckpt_mgr.directory, TRAINING_LOG_FILENAME)
        os.makedirs(ckpt_mgr.directory, exist_ok=True)

        # Resume-with-old-header guard. The CSV only writes a header when
        # the file is new/empty; a run resumed against a log written with
        # the OLD column set would append the new fieldnames' rows under
        # the old header → misaligned columns. If the existing file's
        # first line doesn't match the current header, rotate it out of
        # the way and start fresh so each file is internally consistent.
        expected_header = ",".join(TRAINING_LOG_COLUMNS)
        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            with open(log_path, "r", newline="") as fh:
                first_line = fh.readline().rstrip("\r\n")
            if first_line != expected_header:
                backup = os.path.join(
                    ckpt_mgr.directory,
                    f"training_log.{time.strftime('%Y%m%d-%H%M%S')}.bak",
                )
                os.rename(log_path, backup)
                tqdm.write(
                    f"  ↻ Rotated training log with stale header → "
                    f"{os.path.basename(backup)} (new columns added)"
                )

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

                # Multi-source validation (additive — never touches
                # save-best). Empty string when a source isn't wired so
                # downstream float-parsers skip the cell rather than
                # choke on "nan" text.
                psnr_str_hst: object = ""
                psnr_raw_hst: object = ""
                rt_val_psnr:  object = ""
                if hst_valid_dataset is not None:
                    hst_metrics  = self.evaluate(
                        hst_valid_dataset.take(validate_images))
                    psnr_str_hst = float(hst_metrics["psnr_stretched"].numpy())
                    psnr_raw_hst = float(hst_metrics["psnr_raw"].numpy())
                if roundtrip_valid_dataset is not None:
                    rt_val_psnr = float(self.evaluate_roundtrip(
                        roundtrip_valid_dataset.take(validate_images)))

                # Composite save-best score (higher = better). Synthetic
                # is always in; HST/RT terms join only when their
                # validation set is wired. All three terms are now PSNR in
                # dB, so RT is added like the others. All-zero weights
                # would make every score 0 and freeze save-best, so fall
                # back to synthetic PSNR.
                w_syn, w_hst, w_rt = save_best_weights
                if (w_syn == 0 and w_hst == 0 and w_rt == 0):
                    save_best_score = psnr_str
                else:
                    save_best_score = w_syn * psnr_str
                    if psnr_str_hst != "":
                        save_best_score += w_hst * float(psnr_str_hst)
                    if rt_val_psnr != "":
                        save_best_score += w_rt * float(rt_val_psnr)

                duration = time.perf_counter() - self.now
                pbar.set_postfix(
                    loss=f"{loss_value.numpy():.3f}",
                    PSNRs=f"{psnr_str:.2f}",
                    PSNRr=f"{psnr_raw:.2f}",
                    score=f"{save_best_score:.2f}",
                )
                status = (
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Step {step}/{steps}: loss = {loss_value.numpy():.4f}, "
                    f"PSNR(str/raw) = {psnr_str:.3f}/{psnr_raw:.3f} dB"
                )
                if hst_valid_dataset is not None:
                    status += (
                        f" | HST PSNR(str/raw) = "
                        f"{psnr_str_hst:.3f}/{psnr_raw_hst:.3f} dB"
                    )
                if roundtrip_valid_dataset is not None:
                    status += f" | RT PSNR = {rt_val_psnr:.3f} dB"
                status += (
                    f", |g| avg/max = {gnorm_avg.numpy():.3g}/{gnorm_peak:.3g} "
                    f"({duration:.2f}s)"
                )
                tqdm.write(status)

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
                    "psnr_stretched_hst": psnr_str_hst,
                    "psnr_raw_hst":       psnr_raw_hst,
                    "roundtrip_val_psnr": rt_val_psnr,
                    "save_best_score":    save_best_score,
                }
                write_header = (not os.path.exists(log_path)
                                or os.path.getsize(log_path) == 0)
                with open(log_path, "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=TRAINING_LOG_COLUMNS)
                    if write_header:
                        w.writeheader()
                    w.writerow(row)

                # save-best on the composite score (see save_best_weights).
                # ``ckpt.psnr`` is the checkpointed best-score threshold —
                # the name predates the composite; it now holds whatever
                # save_best_weights blends, not the bare synthetic PSNR.
                # The first validation of this run force-saves +
                # re-baselines it so a regime change (new HST/round-trip
                # mix) that dips below the inherited best still updates the
                # weights; subsequent validations resume monotonic
                # save-best on the composite.
                is_first = first_eval_this_run
                first_eval_this_run = False
                should_save = (
                    not save_best_only         # save-every mode
                    or is_first                # re-baseline this run's 1st eval
                    or save_best_score > ckpt.psnr  # genuine improvement
                )
                if not should_save:
                    self.now = time.perf_counter()
                    continue

                # ``.assign`` keeps ckpt.psnr a tf.Variable (so it stays
                # checkpoint-tracked across resumes and exposes .numpy());
                # the prior ``ckpt.psnr = <tensor>`` replaced the Variable
                # with a bare tensor and silently dropped that tracking.
                ckpt.psnr.assign(save_best_score)
                ckpt_mgr.save()
                why = ("re-baseline (first eval this run)"
                       if is_first and save_best_only else "best so far")
                tqdm.write(
                    f"  ✓ Checkpoint saved [{why}] (score={save_best_score:.3f}; "
                    f"PSNR str={psnr_str:.3f}, raw={psnr_raw:.3f} dB)"
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

            is_rt  = tf.equal(source, self._source_roundtrip)
            is_hst = tf.equal(source, self._source_hst)
            is_syn = tf.logical_not(tf.logical_or(is_rt, is_hst))

            if self.forward_op is None:
                # No forward op installed — fall back to supervised for
                # *all* examples (dummy HR poisons the round-trip ones,
                # but the trainer can't compute the round-trip loss
                # without the op). Caller should configure the op
                # whenever the dataset is built with roundtrip_fraction>0.
                base_per_example = sup_per_example
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
                base_per_example = tf.where(is_rt, rt_per_example,
                                            sup_per_example)

            # Per-source weight: each example scaled by the knob matching
            # its source tag (exactly one mask is 1 per example).
            weight = (
                tf.cast(is_syn, tf.float32) * self.synthetic_loss_weight
                + tf.cast(is_hst, tf.float32) * self.hst_loss_weight
                + tf.cast(is_rt, tf.float32) * self.roundtrip_loss_weight
            )
            loss_value = tf.reduce_mean(weight * base_per_example)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        gradients, gnorm = tf.clip_by_global_norm(gradients, clip_norm=GRAD_CLIP_NORM)
        self.checkpoint.optimizer.apply_gradients(
            zip(gradients, self.checkpoint.model.trainable_variables)
        )

        return loss_value, gnorm

    def train_step_sky(self, lr, hr, n_syn: int, n_hst: int, n_rt: int):
        """``SR = deconvolved sky`` step on a fixed-layout batch.

        The batch lanes are laid out in fixed, contiguous blocks so the
        slices below are static Python ints (no per-step ``tf.gather``, no
        retracing): ``[0:n_syn]`` synthetic, then ``n_hst`` HST, then
        ``n_rt`` round-trip. Each source supervises the single sky
        estimate ``SR`` through its own instrument's *forward* PSF:

          * synthetic — ``|SR − scene|`` (direct; LR was E⊛scene, so the
            clean target *is* the sky).
          * HST       — ``|asinh(H ⊛ SR_lin) − HST_image|`` (no rebin;
            HST shares SR's 0.05″ grid).
          * round-trip— ``|asinh(rebin(E ⊛ SR_lin)) − LR_vis|``.

        All comparisons are in asinh space (the stretch the records and
        the model output already use); the PSF convs run in linear space
        (``inverse_asinh`` → conv → ``asinh``). Per-source loss weights
        scale each block. ``hr`` carries the scene / HST image for the
        supervised lanes (dummy for round-trip lanes, never read).
        """
        with tf.GradientTape() as tape:
            sr     = self.checkpoint.model(lr, training=True)   # asinh space
            sr_lin = inverse_asinh_stretch_hr(sr)               # linear electrons
            per_example = []   # list of [n_*] loss vectors

            i = 0
            if n_syn > 0:
                s = slice(i, i + n_syn)
                d = tf.reduce_mean(tf.abs(sr[s] - hr[s]), axis=[1, 2, 3])
                per_example.append(self.synthetic_loss_weight * d)
                i += n_syn
            if n_hst > 0:
                s = slice(i, i + n_hst)
                hconv = asinh_stretch_hr(self.hst_forward_op(sr_lin[s]))
                d = tf.reduce_mean(tf.abs(hconv - hr[s]), axis=[1, 2, 3])
                per_example.append(self.hst_loss_weight * d)
                i += n_hst
            if n_rt > 0:
                s = slice(i, i + n_rt)
                econv = asinh_stretch_hr(self.forward_op(sr_lin[s]))
                lr_vis = lr[s][..., 0:1]
                d = tf.reduce_mean(tf.abs(econv - lr_vis), axis=[1, 2, 3])
                per_example.append(self.roundtrip_loss_weight * d)

            loss_value = tf.reduce_mean(tf.concat(per_example, axis=0))

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

    def evaluate_roundtrip(self, lr_dataset) -> float:
        """Mean round-trip reconstruction PSNR (dB) over an LR-only dataset.

        Forward-models the SR back to Euclid LR and compares it to the
        observed LR VIS, in eval mode and without gradients::

            sr        = model(lr, training=False)
            sr_lin    = inverse_asinh_stretch_hr(sr)
            recon     = forward_op(sr_lin)
            recon_str = asinh_stretch_hr(recon)
            psnr      = PSNR(lr[..., 0:1], recon_str)   # asinh space

        PSNR is taken in the same asinh-stretched space and against the
        same peak (``Config.PSNR_PEAK_STRETCHED``) as the synthetic and
        HST PSNRs, so all three validation metrics share one dB scale.
        Higher = the SR is more self-consistent with the observed Euclid
        LR (cycle consistency).

        Parameters
        ----------
        lr_dataset : tf.data.Dataset
            Yields LR tensors ``[B, H, W, 4]`` (e.g. from
            :func:`euclid_polish.training.data_multiband.lr_only_dataset`).

        Returns
        -------
        float
            Set-mean round-trip PSNR in dB, or ``nan`` when
            ``self.forward_op`` is ``None`` (the operator is required to
            map SR back to LR).
        """
        if self.forward_op is None:
            return float("nan")
        running = Mean()
        max_val = tf.constant(float(Config.PSNR_PEAK_STRETCHED), dtype=tf.float32)
        for lr in lr_dataset:
            sr               = self.checkpoint.model(lr, training=False)
            sr_linear        = inverse_asinh_stretch_hr(sr)
            lr_recon_linear  = self.forward_op(sr_linear)
            lr_recon_stretch = asinh_stretch_hr(lr_recon_linear)
            lr_vis           = lr[..., 0:1]
            batch_psnr       = tf.reduce_mean(
                tf.image.psnr(lr_vis, lr_recon_stretch, max_val=max_val))
            running(batch_psnr)
        return float(running.result().numpy())

    def restore(self):
        """Restore model from checkpoint if available."""
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(
                f"Model restored from checkpoint at step {self.checkpoint.step.numpy()}."
            )
