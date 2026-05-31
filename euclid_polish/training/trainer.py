"""
Trainer module for WDSR super-resolution models.

The model always estimates one quantity: the **deconvolved sky** ``SR``.
Every source supervises that single estimate through its own
instrument's forward operator, so the objective is consistent across
sources (no source pulls ``SR`` toward a PSF-blurred image). The trainer
drives this via two batch formats, dispatched on ``lane_counts`` in
:meth:`Trainer.train`:

  * 2-tuple ``(lr, hr)`` with ``lane_counts=None`` — the pure-supervised
    path (``|SR - hr|``). Used by ``scripts/run_pipeline.py``,
    ``cli/main.py``, the web inference helpers, and every validation
    stream. → :meth:`train_step`.
  * 2-tuple ``(lr, hr)`` with ``lane_counts=(n_syn, n_hst, n_rt)`` — a
    **fixed contiguous layout** batch (first ``n_syn`` synthetic, then
    ``n_hst`` HST, then ``n_rt`` round-trip), produced by
    :meth:`MultiBandEuclidDataset.dataset_fixed_layout`. →
    :meth:`train_step_sky`, which slices each lane by its static count
    and applies that lane's forward op:

      - synthetic — ``|SR - scene|`` (direct; the clean target is the sky).
      - HST       — ``|asinh(H ⊛ SR) - HST_image|`` (HST F814W PSF, no rebin).
      - round-trip— ``|asinh(rebin(E ⊛ SR)) - LR_vis|`` (VIS PSF + 2× rebin).

The split is by fixed *position* (static Python-int slices), not a
per-example ``tf.where`` mask — so there is no branching and each lane's
conv is a single batched op over its block.
"""
import csv
import os
import time

import numpy as np
import tensorflow as tf

from tf_keras.losses import MeanAbsoluteError
from tf_keras.metrics import Mean
from tf_keras.optimizers import Adam
from tf_keras.optimizers.schedules import PiecewiseConstantDecay
from tqdm import tqdm

from euclid_polish.config import Config
from euclid_polish.training.data_multiband import (
    asinh_stretch_hr, inverse_asinh_stretch_hr,
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
    # "1" on the single pre-training row written when a run resumes: the
    # restored checkpoint's score measured under THIS run's validation
    # setup. It seeds the save-best threshold (no force-save) and the log
    # plot draws it as a dashed "bar to beat" line. Empty on normal rows.
    "is_baseline",
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
        nonneg_sr_weight: float = Config.NONNEG_SR_WEIGHT,
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
        nonneg_sr_weight : float
            Weight of the non-negativity penalty ``λ · mean(relu(-SR))``
            added to every step's loss. SR is the model's single output
            (the deconvolved sky), shared by all three lanes, so one term
            on it constrains them all toward physically valid (≥ 0) flux.
            Penalised in asinh space (scale-matched to the MAE loss). 0
            disables it. Default ``Config.NONNEG_SR_WEIGHT``. A soft
            penalty makes negatives rare/small, not impossible — clamp the
            delivered product for a hard guarantee.
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
        self.nonneg_sr_weight      = float(nonneg_sr_weight)
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

    def _validate(
        self, valid_dataset, hst_valid_dataset, roundtrip_valid_dataset,
        validate_images, save_best_weights,
    ) -> dict:
        """Run every wired validation source and the composite save-best score.

        Single source of truth for the metric block: used both for the
        pre-training baseline eval (the restored checkpoint's score) and
        for each in-loop evaluation, so the two are computed identically
        and the baseline is directly comparable to later points.

        Returns a dict with ``psnr_str`` / ``psnr_raw`` (synthetic, always)
        plus ``psnr_str_hst`` / ``psnr_raw_hst`` / ``rt_val_psnr`` (empty
        string when that source isn't wired) and the composite
        ``save_best_score``.
        """
        metrics  = self.evaluate(valid_dataset.take(validate_images))
        psnr_str = float(metrics["psnr_stretched"].numpy())
        psnr_raw = float(metrics["psnr_raw"].numpy())

        psnr_str_hst: object = ""
        psnr_raw_hst: object = ""
        rt_val_psnr:  object = ""
        if hst_valid_dataset is not None:
            # SR=sky-consistent: score H⊛SR vs the observed HST image (see
            # evaluate_hst), not SR vs HST directly. NaN (no HST forward op)
            # is treated as "not wired" so it can't poison the composite.
            hm = self.evaluate_hst(hst_valid_dataset.take(validate_images))
            psnr_str_hst = float(hm["psnr_stretched"].numpy())
            psnr_raw_hst = float(hm["psnr_raw"].numpy())
            if not np.isfinite(psnr_str_hst):
                psnr_str_hst = ""
                psnr_raw_hst = ""
        if roundtrip_valid_dataset is not None:
            rt_val_psnr = float(self.evaluate_roundtrip(
                roundtrip_valid_dataset.take(validate_images)))

        # Composite (higher = better). Synthetic always in; HST/RT join only
        # when wired. All-zero weights would freeze save-best, so fall back
        # to bare synthetic PSNR.
        w_syn, w_hst, w_rt = save_best_weights
        if w_syn == 0 and w_hst == 0 and w_rt == 0:
            save_best_score = psnr_str
        else:
            save_best_score = w_syn * psnr_str
            if psnr_str_hst != "":
                save_best_score += w_hst * float(psnr_str_hst)
            if rt_val_psnr != "":
                save_best_score += w_rt * float(rt_val_psnr)

        return {
            "psnr_str":        psnr_str,
            "psnr_raw":        psnr_raw,
            "psnr_str_hst":    psnr_str_hst,
            "psnr_raw_hst":    psnr_raw_hst,
            "rt_val_psnr":     rt_val_psnr,
            "save_best_score": save_best_score,
        }

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
        lane_counts=None,
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

        # Resume baseline: measure the RESTORED checkpoint's score under
        # *this* run's validation setup and seed the save-best threshold
        # with it — instead of force-saving on the first eval. The previous
        # checkpoint stays the best until genuinely beaten, and the log gets
        # a single ``is_baseline`` row the plot draws as a dashed "bar to
        # beat" line. Only when resuming (start_step > 0) and in save-best
        # mode; a fresh run has nothing to validate and starts from the
        # checkpoint's initial ``psnr`` sentinel.
        if save_best_only and start_step > 0:
            b = self._validate(
                valid_dataset, hst_valid_dataset, roundtrip_valid_dataset,
                validate_images, save_best_weights,
            )
            ckpt.psnr.assign(b["save_best_score"])
            base_row = {
                "step":               int(start_step),
                "wall_time":          time.time(),
                "loss":               "",
                "psnr_stretched":     b["psnr_str"],
                "psnr_raw":           b["psnr_raw"],
                "gnorm_avg":          "",
                "gnorm_max":          "",
                "clip_norm":          float(GRAD_CLIP_NORM),
                "duration_s":         "",
                "psnr_stretched_hst": b["psnr_str_hst"],
                "psnr_raw_hst":       b["psnr_raw_hst"],
                "roundtrip_val_psnr": b["rt_val_psnr"],
                "save_best_score":    b["save_best_score"],
                "is_baseline":        "1",
            }
            write_header = (not os.path.exists(log_path)
                            or os.path.getsize(log_path) == 0)
            with open(log_path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=TRAINING_LOG_COLUMNS)
                if write_header:
                    w.writeheader()
                w.writerow(base_row)
            tqdm.write(
                f"  ▏baseline (restored ckpt @ step {start_step}): "
                f"score={b['save_best_score']:.3f} "
                f"(PSNR str={b['psnr_str']:.3f} dB) — bar to beat, no save"
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
            # Dispatch on ``lane_counts``. When None (the pure-supervised
            # path used by run_pipeline.py / cli/main.py / the web
            # inference helpers and every validation stream), batches are
            # plain ``(lr, hr)`` 2-tuples → ``train_step``. When set, the
            # dataset is a fixed contiguous ``[n_syn | n_hst | n_rt]``
            # layout → ``train_step_sky``, which slices each lane by its
            # static count and applies that lane's forward op (no
            # per-example branching). The counts are Python ints, so the
            # @tf.function traces once for the run's fixed layout.
            lr, hr = batch
            if lane_counts is not None:
                loss, gnorm = self.train_step_sky(lr, hr, *lane_counts)
            else:
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

                # Validation + composite score (same code path as the
                # resume baseline, so the two are directly comparable).
                v = self._validate(
                    valid_dataset, hst_valid_dataset, roundtrip_valid_dataset,
                    validate_images, save_best_weights,
                )
                psnr_str        = v["psnr_str"]
                psnr_raw        = v["psnr_raw"]
                psnr_str_hst    = v["psnr_str_hst"]
                psnr_raw_hst    = v["psnr_raw_hst"]
                rt_val_psnr     = v["rt_val_psnr"]
                save_best_score = v["save_best_score"]

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
                    "is_baseline":        "",
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
                # On a resume it was seeded by the baseline eval above, so
                # we only save on a genuine improvement over the restored
                # checkpoint's measured score — no force-save churn.
                should_save = (
                    not save_best_only              # save-every mode
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
                tqdm.write(
                    f"  ✓ Checkpoint saved [best so far] "
                    f"(score={save_best_score:.3f}; "
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
            loss_value = self._add_nonneg_penalty(loss_value, sr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        gradients, gnorm = tf.clip_by_global_norm(gradients, clip_norm=GRAD_CLIP_NORM)
        self.checkpoint.optimizer.apply_gradients(
            zip(gradients, self.checkpoint.model.trainable_variables)
        )

        return loss_value, gnorm

    def _add_nonneg_penalty(self, loss_value, sr):
        """Add ``λ · mean(relu(-SR))`` to ``loss_value`` (no-op when λ=0).

        SR is the model's single output, so this one term — applied to the
        whole batch's ``sr`` — penalises negativity for every lane at once
        (synthetic, HST, round-trip all branch from this SR). Penalised in
        asinh space; ``relu(-sr)`` is 0 where sr ≥ 0 and grows linearly
        with how negative it is, a constant upward push on negative pixels.
        ``λ`` is a Python float, so this resolves at trace time.
        """
        if self.nonneg_sr_weight > 0:
            penalty = tf.reduce_mean(tf.nn.relu(-sr))
            return loss_value + self.nonneg_sr_weight * penalty
        return loss_value

    @tf.function
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

        ``n_syn``/``n_hst``/``n_rt`` are Python ints, so these guards and
        the slice logic resolve at trace time — a lane is compiled into
        the graph only when its count is non-zero, and the required
        forward op must be installed or tracing raises.
        """
        if n_hst > 0 and self.hst_forward_op is None:
            raise ValueError(
                "train_step_sky: n_hst > 0 requires hst_forward_op to be set "
                "(HSTForwardOp); the HST lane convolves SR with the HST PSF"
            )
        if n_rt > 0 and self.forward_op is None:
            raise ValueError(
                "train_step_sky: n_rt > 0 requires forward_op to be set "
                "(EuclidVISForwardOp); the round-trip lane convolves SR with "
                "the VIS PSF"
            )
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
            # One non-negativity penalty on the shared SR covers all lanes.
            loss_value = self._add_nonneg_penalty(loss_value, sr)

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

    def evaluate_hst(self, hst_dataset) -> dict:
        """HST-source validation through the forward op (SR=sky-consistent).

        Mirrors the HST training lane: compares ``H ⊛ SR`` to the observed
        HST image rather than ``SR`` directly. Scoring ``SR`` against the
        HST image would reward HST-PSF blur and contradict the synthetic /
        round-trip lanes (which target the deconvolved sky), so the metric
        must apply the same forward op the loss does. For each ``(lr, hr)``
        — ``hr`` the observed HST image, asinh-stretched on the 0.05″ grid::

            sr     = model(lr, training=False)
            sr_lin = inverse_asinh(sr)
            hconv  = hst_forward_op(sr_lin)        # H ⊛ SR, linear e⁻, HR grid
            psnr_stretched = PSNR(hr, asinh(hconv), PSNR_PEAK_STRETCHED)
            psnr_raw       = PSNR(inverse_asinh(hr), hconv, PSNR_PEAK_E)

        Returns ``{"psnr_stretched", "psnr_raw"}`` as set-mean scalar
        tensors (same keys as :func:`models.common.evaluate`), or NaN
        tensors when ``hst_forward_op`` is ``None``.
        """
        if self.hst_forward_op is None:
            nan = tf.constant(float("nan"), dtype=tf.float32)
            return {"psnr_stretched": nan, "psnr_raw": nan}
        str_mean = Mean()
        raw_mean = Mean()
        max_str = tf.constant(float(Config.PSNR_PEAK_STRETCHED), dtype=tf.float32)
        max_raw = tf.constant(float(Config.PSNR_PEAK_E), dtype=tf.float32)
        for lr, hr in hst_dataset:
            sr        = self.checkpoint.model(lr, training=False)
            sr_lin    = inverse_asinh_stretch_hr(sr)
            hconv_lin = self.hst_forward_op(sr_lin)
            hconv_str = asinh_stretch_hr(hconv_lin)
            str_mean(tf.reduce_mean(
                tf.image.psnr(hr, hconv_str, max_val=max_str)))
            hr_lin = inverse_asinh_stretch_hr(hr)
            raw_mean(tf.reduce_mean(
                tf.image.psnr(hr_lin, hconv_lin, max_val=max_raw)))
        return {"psnr_stretched": str_mean.result(),
                "psnr_raw": raw_mean.result()}

    def restore(self):
        """Restore model from checkpoint if available."""
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(
                f"Model restored from checkpoint at step {self.checkpoint.step.numpy()}."
            )
