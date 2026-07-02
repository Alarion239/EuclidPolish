"""Trainer module for WDSR super-resolution models."""
import contextlib
import math
import os
import random
import time

import numpy as np
import tensorflow as tf
from tf_keras.losses import MeanAbsoluteError
from tf_keras.metrics import Mean
from tf_keras.optimizers import Adam
from tf_keras.optimizers.schedules import PiecewiseConstantDecay
from tqdm import tqdm

from euclid_polish.config import Config
from euclid_polish.observability.training_log import TrainingLog
from euclid_polish.provenance.checkpoint import (
    read_checkpoint_provenance,
    write_checkpoint_provenance,
)
from euclid_polish.provenance.defaults import default_store
from euclid_polish.provenance.gitinfo import capture_git
from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.records import ConfigSnapshot, Process, Stamp
from euclid_polish.training.augmentation import (
    asinh_stretch_hr,
    inverse_asinh_stretch_hr,
)
from euclid_polish.training.models.common import evaluate
from euclid_polish.training.plateau import PlateauLRReducer


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and TensorFlow global RNGs for a reproducible run.

    Covers the training stochasticity — data-pipeline shuffle order and
    augmentation crops/flips — and weight initialisation *when called before the
    model is built* (which :class:`~euclid_polish.model.Model` does).

    CAVEAT — GPU nondeterminism: even with a fixed seed, cuDNN / cuBLAS kernels
    can introduce small run-to-run differences on GPU. Pass ``deterministic=True``
    to additionally enable TensorFlow op-determinism for bit-exact replay
    (noticeably slower, and a few ops are unsupported). On CPU the seed alone is
    fully reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if deterministic:
        # Older / unsupported TF: the seed still applies, just not op-determinism.
        with contextlib.suppress(Exception):
            tf.config.experimental.enable_op_determinism()

# Append-only CSV — one row per evaluate_every batch — readable by Excel,
# pandas, the FASRC dashboard, and the in-tree plot_training_log helper
# without any custom parser. Validation history persists in real time so
# a job killed mid-training still leaves a usable log behind. Owned by
# ``euclid_polish.observability.training_log.TrainingLog`` (schema +
# append + resume rotation); the trainer just builds rows.
TRAINING_LOG_FILENAME = TrainingLog.FILENAME
# Per-band validation PSNR columns (4-band model; a VIS-only run fills just
# the first). MONITORING ONLY — save-best keys on the joint ``psnr_stretched``;
# these exist so the log shows VIS and the noisier NISP channels separately.
PER_BAND_PSNR_COLUMNS = tuple(
    f"psnr_{name.lower()}" for name in Config.HR_TARGET_BAND_NAMES
)
TRAINING_LOG_COLUMNS  = (
    "step", "wall_time",
    # ``loss`` is the optimised composite (mean over the eval window of the
    # concatenated per-lane weighted losses). ``loss_{syn,hst,anchor}`` are
    # the per-lane *unweighted* mean losses — the whole training history of
    # each mode, no log-parsing. Empty when a lane is off this run.
    "loss", "loss_syn", "loss_hst", "loss_anchor",
    "psnr_stretched", "psnr_raw",
    *PER_BAND_PSNR_COLUMNS,
    "gnorm_avg", "gnorm_max", "clip_norm", "duration_s",
    # Multi-source validation (additive). Empty string when the
    # corresponding source dataset isn't wired for this run, so old
    # plots that parse-float these columns skip them instead of
    # choking on the text "nan".
    "psnr_stretched_hst", "psnr_raw_hst", "anchor_val_psnr",
    # The composite scalar save-best actually keys on (weighted blend of
    # the per-source metrics; see ``save_best_weights``). Logged so the
    # decision is auditable from the CSV alone.
    "save_best_score",
    # The combined VALIDATION loss the SECOND save-best track keys on (lower
    # = better; same lanes/weights as ``save_best_score``, but on the held-out
    # MAEs gathered in _validate alongside PSNR — NOT the training window).
    # Its checkpoints live in ``loss_best/``; /inference loads PSNR- or
    # loss-best.
    "combined_loss",
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
# Spike guard: a post-warmup gradient spike triggers a checkpoint ROLLBACK
# (handled in the train loop, eager) rather than an in-graph skip — skipping a
# diverged model only freezes it; restoring continues from the last good state.
# See ``Config.GRAD_SPIKE_*``.
GRAD_SPIKE_SKIP_NORM = float(Config.GRAD_SPIKE_SKIP_NORM)
GRAD_SPIKE_SKIP_WARMUP_STEPS = int(Config.GRAD_SPIKE_SKIP_WARMUP_STEPS)
GRAD_SPIKE_MAX_ROLLBACKS = int(Config.GRAD_SPIKE_MAX_ROLLBACKS)
GRAD_SPIKE_MAX_LR_HALVINGS = int(Config.GRAD_SPIKE_MAX_LR_HALVINGS)


def _is_grad_spike(gnorm, step) -> bool:
    """True iff this step is a post-warmup gradient spike worth rolling back.

    A spike is a PRE-clip global grad norm that is non-finite or exceeds
    ``GRAD_SPIKE_SKIP_NORM`` (steady-state is ~0.5). Inert for the first
    ``GRAD_SPIKE_SKIP_WARMUP_STEPS`` steps, where early-training gradients are
    legitimately large, and when the threshold is ``0`` (disabled). Evaluated
    eagerly in the train loop (``gnorm``/``step`` are already materialised
    there) so the rollback — restoring the last checkpoint — can run outside
    the @tf.function graph.
    """
    if GRAD_SPIKE_SKIP_NORM <= 0 or int(step) <= GRAD_SPIKE_SKIP_WARMUP_STEPS:
        return False
    g = float(gnorm)
    return (not math.isfinite(g)) or g > GRAD_SPIKE_SKIP_NORM

# Cap on the per-star anchor PSNR. The anchor target is a single delta
# pixel, so a model that nails that one pixel drives its MSE → 0 and the
# raw PSNR → +∞ (well above the realistic synthetic/HST ~50–70 dB range),
# spiking the metric and squashing the plot's y-axis. Clip each star's PSNR
# to this ceiling — "≥ this = essentially exact" — so it stays finite and on
# the same scale as the other PSNRs.
ANCHOR_PSNR_MAX_DB = 80.0


def _plateau_recovery_step(lr_scale: float, factor: float,
                           cuts: int) -> tuple[float, int]:
    """Undo ONE plateau LR cut after a new-best PSNR (proof the stall broke).

    Returns ``(new_scale, remaining_cuts)``. Only cuts the plateau guard made
    are undone (the counter), so spike-guard halvings — which exist because
    the LR was outright divergent — are never re-raised, and the scale never
    exceeds the schedule value (cap at 1.0).
    """
    if cuts <= 0:
        return lr_scale, 0
    return min(lr_scale / float(factor), 1.0), cuts - 1


def _plateau_wants_rollback(current_score, best_score, *, min_gap: float,
                            has_best_ckpt: bool, save_best_only: bool) -> bool:
    """True iff a firing plateau is DEGENERATE — the score sits ≥ ``min_gap``
    below the run's best — and a rollback target exists.

    Two plateau regimes need different responses. A *converged* plateau
    (score ≈ best) wants the LR reduced in place. A *degenerate* plateau
    (e.g. the ~43.5 dB skip-only basin, entered from a hot restart) must NOT
    be cooled in place: that polishes the collapsed solution deeper. The fix
    is restore-best-then-cool — continue from the pre-collapse weights at the
    reduced LR. Pure function so the decision is unit-testable.
    """
    if not save_best_only or not has_best_ckpt:
        return False
    try:
        cur, best = float(current_score), float(best_score)
    except (TypeError, ValueError):
        return False
    if not (np.isfinite(cur) and np.isfinite(best)):
        return False
    return (best - cur) >= float(min_gap)


def _combined_loss(loss_syn, loss_hst, loss_anchor, weights) -> float:
    """Weighted blend of the per-lane *losses* (lower = better), keyed on the
    SAME ``(w_syn, w_hst, w_anchor)`` the PSNR save-best composite uses.

    The second save-best track minimises this. Inactive lanes (value ``""``)
    are skipped; all-zero weights fall back to the equal-weight mean of the
    active lanes (mirrors the PSNR composite's "don't freeze on all-zero").
    Returns ``+inf`` if no lane is active (nothing to score this eval).
    """
    parts = []
    for w, v in zip(weights, (loss_syn, loss_hst, loss_anchor), strict=False):
        if v != "" and v is not None:
            parts.append((float(w), float(v)))
    if not parts:
        return float("inf")
    if all(w == 0 for w, _ in parts):
        return sum(v for _, v in parts) / len(parts)
    return sum(w * v for w, v in parts)


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
        star_anchor_loss_weight: float = 1.0,
        nonneg_sr_weight: float = Config.NONNEG_SR_WEIGHT,
        seed: int | None = None,
        deterministic: bool = False,
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
            Differentiable Euclid VIS forward op (PSF + 2× sum-rebin).
            Retained for back-compat; the star-anchor objective is
            operator-free, so this is no longer used by training (the
            ``EuclidVISForwardOp`` class still backs the ``/inference``
            forward(SR) panel). ``None`` (default).
        synthetic_loss_weight, hst_loss_weight, star_anchor_loss_weight : float
            Per-source multipliers on the per-example loss before the
            batch mean. Each lane (synthetic / HST / star-anchor) is
            scaled by its knob. All default to 1.0. Raise a knob to
            up-weight that lane's gradient contribution; set to 0 to keep
            the dataset wired but zero out its loss (ablation). The
            star-anchor lane supervises ``SR`` against a sparse HR
            delta-target (catalog flux at the star), masked to the star
            pixel — no PSF.
        nonneg_sr_weight : float
            Weight of the non-negativity penalty ``λ · mean(relu(-SR))``
            added to every step's loss. SR is the model's single output
            (the deconvolved sky), shared by all three lanes, so one term
            on it constrains them all toward physically valid (≥ 0) flux.
            Penalised in asinh space (scale-matched to the MAE loss). 0
            disables it. Default ``Config.NONNEG_SR_WEIGHT``. A soft
            penalty makes negatives rare/small, not impossible — clamp the
            delivered product for a hard guarantee.
        seed : int, optional
            Master RNG seed for the run. When set, ``train()`` seeds Python /
            NumPy / TF and records the seed on a ``Process.training`` provenance
            record (the checkpoint stamp then points back to it), so the run can
            be replayed. ``None`` (default) keeps the previous entropy-driven
            behaviour. NOTE: for weight-init reproducibility the seed must also
            be applied *before the model is built* —
            :class:`~euclid_polish.model.Model` does this when constructed with
            ``seed=``.
        deterministic : bool
            Also enable TF op-determinism for bit-exact GPU replay (slower).
            Without it, cuDNN may introduce small nondeterminism on GPU even
            with a fixed seed; on CPU the seed alone is fully reproducible.
        """
        self.now = None
        self.loss = loss
        self.forward_op = forward_op
        # HST forward op (H ⊛ SR, no rebin) for the SR=sky objective —
        # the HST supervised loss compares H⊛SR to the observed HST image.
        self.hst_forward_op = hst_forward_op
        self.synthetic_loss_weight   = float(synthetic_loss_weight)
        self.hst_loss_weight         = float(hst_loss_weight)
        self.star_anchor_loss_weight = float(star_anchor_loss_weight)
        self.nonneg_sr_weight        = float(nonneg_sr_weight)
        # Build Adam with a CONSTANT, settable learning rate so the divergence
        # guard can halve it on repeated rollbacks (an Adam built with a
        # LearningRateSchedule is not settable). A passed schedule is kept and
        # applied MANUALLY each step via ``_apply_lr`` (sampled at the loop's
        # step), scaled by ``self._lr_scale`` which the guard halves.
        if isinstance(learning_rate, (int, float)):
            self._lr_schedule = None
            self._initial_lr  = float(learning_rate)
        else:                                  # a LearningRateSchedule (callable)
            self._lr_schedule = learning_rate
            self._initial_lr  = float(learning_rate(tf.constant(0, tf.int64)))
        # ``_lr_scale`` is the shared multiplicative LR knob: the schedule value
        # at each step is multiplied by it. BOTH guards drive it down — the
        # gradient-spike guard (on divergence) and the plateau guard below (on
        # stagnation) — and ``_apply_lr`` clamps the product at ``_min_lr``.
        self._lr_scale = 1.0
        self._min_lr = float(plateau_lr_min_lr)
        self._plateau_lr_factor = float(plateau_lr_factor)
        self._plateau_lr_metric = str(plateau_lr_metric)
        self._plateau_rollback_min_gap = float(plateau_rollback_min_gap)
        self._plateau_lr_recovery = bool(plateau_lr_recovery)
        self._plateau_cuts = 0          # cuts the plateau guard made (undoable)
        if resume_track not in ("latest", "psnr"):
            raise ValueError(f"resume_track must be 'latest' or 'psnr', "
                             f"got {resume_track!r}")
        self._resume_track = resume_track
        self._plateau = (
            PlateauLRReducer(
                mode="max" if self._plateau_lr_metric == "psnr_stretched" else "min",
                patience=int(plateau_lr_patience),
                min_delta=float(plateau_lr_min_delta),
                min_delta_rel=float(plateau_lr_min_delta_rel),
                cooldown=int(plateau_lr_cooldown),
            )
            if plateau_lr_enabled else None
        )
        # Degenerate-basin detector state (PSNR-based; see the eval loop):
        # last step a new best PSNR was saved, and how many CONSECUTIVE evals
        # scored >= rollback_min_gap below the best.
        self._psnr_best_step = 0
        self._gap_streak = 0
        # ``psnr`` tracks the best PSNR_stretched seen so far (used by
        # save-best). max_val for PSNR is set in models/common.py from
        # Config.PSNR_PEAK_STRETCHED ≈ asinh(mag-17 star / k).
        self.checkpoint = tf.train.Checkpoint(
            step=tf.Variable(0),
            psnr=tf.Variable(-1.0),
            best_loss=tf.Variable(float("inf")),
            optimizer=Adam(self._initial_lr),
            model=model,
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=checkpoint_dir,
            max_to_keep=3,
        )
        # Second save-best track, keyed on the combined LOSS (lower = better)
        # with the SAME lane weights as the PSNR composite. Its own checkpoint
        # set lives in a ``loss_best/`` subdir so the two never collide and
        # the mirror pulls both; /inference can load from either. Wraps the
        # SAME checkpoint object — only the save *trigger* and directory
        # differ. ``best_loss`` is the persisted lowest-combined-loss bar.
        self.loss_checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=os.path.join(checkpoint_dir, "loss_best"),
            max_to_keep=3,
        )

        # Provenance: this checkpoint dir's identity. Resolved lazily on the
        # first save (or reused from an existing sidecar on resume).
        self.checkpoint_dir = checkpoint_dir
        self._model_prov_id = None
        # Reproducibility: the run's master seed (or None). Recorded on a
        # Process.training at train() start; the checkpoint stamp's produced_by
        # then points to that run. Seeding the RNGs is the caller's job (Model
        # does it before building the model); train() re-asserts it.
        self._seed = seed
        self._deterministic = bool(deterministic)
        self._training_run_id: ProvId | None = None

        self.restore(self._resume_track)

    @property
    def model(self):
        """Get the model."""
        return self.checkpoint.model

    def _emit_checkpoint_provenance(self) -> None:
        """Write this checkpoint dir's identity sidecar (best-effort).

        Mints the model :class:`ProvId` once, or reuses the id already on disk
        so a resumed run keeps its identity. Any failure is swallowed — a
        provenance hiccup must never break a training run.
        """
        try:
            if self._model_prov_id is None:
                existing = read_checkpoint_provenance(self.checkpoint_dir)
                self._model_prov_id = (
                    existing.id if existing is not None
                    else ProvId.mint(lambda _id: False)
                )
            stamp = Stamp(id=self._model_prov_id,
                          produced_by=self._training_run_id, schema_version=3)
            write_checkpoint_provenance(self.checkpoint_dir, stamp)
            loss_dir = os.path.join(self.checkpoint_dir, "loss_best")
            if os.path.isdir(loss_dir):
                write_checkpoint_provenance(loss_dir, stamp)
        except Exception as exc:  # never break training over provenance
            tqdm.write(f"  [provenance] checkpoint id not written: {exc}")

    def _begin_reproducible_run(self, *, steps: int, evaluate_every: int) -> None:
        """Seed the RNGs and record a ``Process.training`` carrying the seed.

        No-op when no seed was given. Best-effort on the provenance side — a
        store hiccup must never break training; the seed is still applied so the
        run is reproducible even if the record isn't written.
        """
        if self._seed is None:
            return
        seed_everything(self._seed, deterministic=self._deterministic)
        tqdm.write(f"  [reproducibility] seed={self._seed}"
                   + ("  (op-determinism on)" if self._deterministic else ""))
        try:
            store = default_store()
            run = Process.training(
                id=store.mint(),
                git=capture_git(),
                status="running",
                seed=self._seed,
                config=ConfigSnapshot("Training", {
                    "steps": int(steps),
                    "evaluate_every": int(evaluate_every),
                    "deterministic": self._deterministic,
                    "synthetic_loss_weight": self.synthetic_loss_weight,
                    "hst_loss_weight": self.hst_loss_weight,
                    "star_anchor_loss_weight": self.star_anchor_loss_weight,
                }),
            )
            store.put(run)
            self._training_run_id = run.id
        except Exception as exc:  # noqa: BLE001 — provenance is best-effort
            tqdm.write(f"  [provenance] training run not recorded: {exc}")

    def _apply_lr(self, step) -> float:
        """Set the optimiser LR for ``step`` = base schedule value × the
        guards' halving scale, clamped at ``_min_lr``, and return it.

        The optimiser was built with a constant LR, whose backing variable
        assigns in place, so the traced train step picks up the new value with
        no retracing. Called every step when a schedule is active (to follow
        the decay) and after every rollback/halving (a restore can reset the
        optimiser LR, so re-assert the intended value). ``_lr_scale`` is driven
        by BOTH the gradient-spike guard and the plateau guard."""
        if self._lr_schedule is not None:
            base = float(self._lr_schedule(tf.constant(int(step), tf.int64)))
        else:
            base = self._initial_lr
        lr = max(self._min_lr, base * self._lr_scale)
        self.checkpoint.optimizer.learning_rate = lr
        return lr

    def _validate(
        self, valid_dataset, hst_valid_dataset, anchor_valid_dataset,
        validate_images, save_best_weights,
    ) -> dict:
        """Run every wired validation source and the composite save-best score.

        Single source of truth for the metric block: used both for the
        pre-training baseline eval (the restored checkpoint's score) and
        for each in-loop evaluation, so the two are computed identically
        and the baseline is directly comparable to later points.

        Returns a dict with ``psnr_str`` / ``psnr_raw`` (synthetic, always)
        plus ``psnr_str_hst`` / ``psnr_raw_hst`` / ``anchor_val_psnr``
        (empty string when that source isn't wired) and the composite
        ``save_best_score``.
        """
        metrics  = self.evaluate(valid_dataset.take(validate_images))
        psnr_str = float(metrics["psnr_stretched"].numpy())
        psnr_raw = float(metrics["psnr_raw"].numpy())
        # Per-band PSNRs (monitoring only — never feeds save-best). Mapped
        # positionally onto PER_BAND_PSNR_COLUMNS; a VIS-only model has one
        # channel, so the NISP columns stay "" (blank in the CSV).
        band_vals = metrics.get("psnr_band_stretched")
        psnr_bands: dict = dict.fromkeys(PER_BAND_PSNR_COLUMNS, "")
        if band_vals is not None:
            for col, v in zip(PER_BAND_PSNR_COLUMNS, band_vals.numpy(), strict=False):
                psnr_bands[col] = float(v)
        # Per-lane VALIDATION MAE (asinh space), gathered alongside PSNR from
        # the same forward pass. ``""`` for an unwired lane so the loss
        # composite skips it — exactly like the PSNR composite.
        mae_syn: float          = float(metrics["mae_stretched"].numpy())
        mae_hst:    object      = ""
        mae_anchor: object      = ""

        psnr_str_hst:    object = ""
        psnr_raw_hst:    object = ""
        anchor_val_psnr: object = ""
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
            else:
                mae_hst = float(hm["mae_stretched"].numpy())
        if anchor_valid_dataset is not None:
            anchor_val_psnr, _anchor_mae = self.evaluate_anchor(
                anchor_valid_dataset.take(validate_images))
            if not np.isfinite(anchor_val_psnr):
                anchor_val_psnr = ""
            else:
                mae_anchor = float(_anchor_mae)

        # Composite (higher = better). Synthetic always in; HST/anchor join
        # only when wired. All-zero weights would freeze save-best, so fall
        # back to bare synthetic PSNR.
        w_syn, w_hst, w_anchor = save_best_weights
        if w_syn == 0 and w_hst == 0 and w_anchor == 0:
            save_best_score = psnr_str
        else:
            save_best_score = w_syn * psnr_str
            if psnr_str_hst != "":
                save_best_score += w_hst * float(psnr_str_hst)
            if anchor_val_psnr != "":
                save_best_score += w_anchor * float(anchor_val_psnr)

        # Combined VALIDATION loss (lower = better) — the second save-best
        # key, same weights/lanes as ``save_best_score`` but on the held-out
        # MAEs computed above.
        combined_loss = _combined_loss(
            mae_syn, mae_hst, mae_anchor, save_best_weights)

        return {
            "psnr_str":        psnr_str,
            "psnr_raw":        psnr_raw,
            "psnr_bands":      psnr_bands,
            "psnr_str_hst":    psnr_str_hst,
            "psnr_raw_hst":    psnr_raw_hst,
            "anchor_val_psnr": anchor_val_psnr,
            "save_best_score": save_best_score,
            "combined_loss":   combined_loss,
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
        eval_callback=None,
        warn_callback=None,
        hst_valid_dataset=None,
        anchor_valid_dataset=None,
        save_best_weights=(1.0, 1.0, 0.0),
        lane_counts=None,
        compute_resume_baseline=True,
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
        eval_callback : Optional[Callable[[dict], None]]
            Called once per evaluate with that evaluate's full metrics row
            (step, total, loss + per-lane losses, PSNR str/raw + HST/anchor
            variants, gradient norms, duration, ``save_best_score`` and a
            ``saved`` flag set when the eval wrote a checkpoint). Used to
            feed the structured event stream (``Reporter.metric``) so the
            WebUI reads training progress from events, not by parsing logs.
            ``None`` (default) → no callback. Fires every ``evaluate_every``
            steps, so it's already sparse — no extra cadence gating.
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
        anchor_valid_dataset : Optional[tf.data.Dataset]
            Star-anchor ``(lr, hr)`` validation dataset (real Euclid star
            cutouts + sparse HR delta-target). When provided, each
            evaluation records the masked star-anchor PSNR (dB) — how well
            SR matches the catalog flux at the star pixel. Additive only —
            does NOT influence save-best unless ``w_anchor`` > 0. ``None``
            leaves the column empty.
        save_best_weights : tuple[float, float, float]
            ``(w_syn, w_hst, w_anchor)`` weights for the composite
            save-best score, higher = better:

                score = w_syn·PSNR_syn + w_hst·PSNR_hst + w_anchor·PSNR_anchor

            All three terms are PSNR in dB, so they share one scale and add
            directly. A term drops out automatically when its validation
            dataset is absent (e.g. ``w_hst`` has no effect when
            ``hst_valid_dataset is None``). Default ``(1, 1, 0)`` →
            synthetic + HST PSNR, equally weighted, star-anchor
            monitored-only. With HST/anchor absent this reduces to plain
            synthetic-PSNR save-best (backwards compatible).
        compute_resume_baseline : bool
            On a *resumed* run (save-best mode), whether to measure the
            restored checkpoint's score as the bar to beat (``True``,
            default) or ignore it and overwrite the previous best
            (``False``). Set ``False`` when the architecture changed so the
            old checkpoint's score is meaningless — the save-best threshold
            is reset and this run's first eval saves. No effect on a fresh
            run.
        """
        # Reproducibility: apply the run's seed (if any) and record it on a
        # Process.training so the run can be replayed. Done first, before any
        # dataset iteration consumes randomness.
        self._begin_reproducible_run(steps=steps, evaluate_every=evaluate_every)

        loss_mean = Mean()
        # Per-lane unweighted mean losses over the eval window — the
        # synthetic lane is always present; HST / star-anchor only when their
        # lane is active this run (tracked by ``active_hst`` / ``active_anchor``).
        loss_syn_mean    = Mean()
        loss_hst_mean    = Mean()
        loss_anchor_mean = Mean()
        active_hst    = lane_counts is not None and lane_counts[1] > 0
        active_anchor = lane_counts is not None and lane_counts[2] > 0
        gnorm_mean = Mean()
        gnorm_max  = tf.Variable(0.0, trainable=False)

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        start_step = int(ckpt.step.numpy())
        # The degenerate-basin detector measures "steps since the last new
        # best PSNR" — anchor it at the resume point so a resumed run isn't
        # instantly "stalled" (best_step would otherwise read 0).
        self._psnr_best_step = start_step
        self._gap_streak = 0
        # Surface resume to the structured stream too — restore() only prints to
        # stdout, so without this the WebUI gives no sign a run picked up from a
        # checkpoint (it looks like a fresh start).
        if start_step > 0 and warn_callback is not None:
            warn_callback(f"resumed from checkpoint at step {start_step}")

        os.makedirs(ckpt_mgr.directory, exist_ok=True)
        # The structured, resume-continuous metrics CSV lives with the
        # checkpoint and owns its own schema + stale-header rotation.
        train_log = TrainingLog(
            os.path.join(ckpt_mgr.directory, TRAINING_LOG_FILENAME),
            TRAINING_LOG_COLUMNS,
        )
        if train_log.rotated_backup:
            tqdm.write(
                f"  ↻ Rotated training log with stale header → "
                f"{os.path.basename(train_log.rotated_backup)} (new columns)"
            )

        # Resume handling (save-best mode, resumed run only — a fresh run has
        # nothing to validate and starts from the checkpoint's initial
        # ``psnr`` sentinel).
        #
        #   compute_resume_baseline=True  (default): measure the RESTORED
        #     checkpoint's score under *this* run's validation setup and seed
        #     the save-best threshold with it — instead of force-saving on the
        #     first eval. The previous checkpoint stays the best until
        #     genuinely beaten; the log gets one ``is_baseline`` row the plot
        #     draws as a dashed "bar to beat" line.
        #
        #   compute_resume_baseline=False: ignore the restored best entirely
        #     (e.g. the architecture changed, so its score is meaningless or
        #     incomparable). Reset the threshold to -inf so this run's first
        #     eval saves and overwrites the previous best. No baseline row.
        if save_best_only and start_step > 0:
            if compute_resume_baseline:
                b = self._validate(
                    valid_dataset, hst_valid_dataset, anchor_valid_dataset,
                    validate_images, save_best_weights,
                )
                ckpt.psnr.assign(b["save_best_score"])
                # Seed BOTH bars from the restored checkpoint's measured
                # held-out metrics, so neither track force-saves on resume.
                if np.isfinite(b["combined_loss"]):
                    ckpt.best_loss.assign(float(b["combined_loss"]))
                base_row = {
                    "step":               int(start_step),
                    "wall_time":          time.time(),
                    # No training step has run yet → loss columns blank.
                    "loss":               "",
                    "loss_syn":           "",
                    "loss_hst":           "",
                    "loss_anchor":        "",
                    "psnr_stretched":     b["psnr_str"],
                    "psnr_raw":           b["psnr_raw"],
                    **b["psnr_bands"],
                    "gnorm_avg":          "",
                    "gnorm_max":          "",
                    "clip_norm":          float(GRAD_CLIP_NORM),
                    "duration_s":         "",
                    "psnr_stretched_hst": b["psnr_str_hst"],
                    "psnr_raw_hst":       b["psnr_raw_hst"],
                    "anchor_val_psnr":    b["anchor_val_psnr"],
                    "save_best_score":    b["save_best_score"],
                    "combined_loss":      b["combined_loss"],
                    "is_baseline":        "1",
                }
                train_log.append(base_row)
                tqdm.write(
                    f"  ▏baseline (restored ckpt @ step {start_step}): "
                    f"score={b['save_best_score']:.3f} "
                    f"(PSNR str={b['psnr_str']:.3f} dB) — bar to beat, no save"
                )
            else:
                ckpt.psnr.assign(float("-inf"))
                ckpt.best_loss.assign(float("inf"))
                tqdm.write(
                    "  ▏resume baseline disabled — save-best threshold reset; "
                    "this run overwrites the previous best on its first eval"
                )

        # Honest, step-based loop: a rollback rewinds the model AND ckpt.step to
        # the restored checkpoint, and the run keeps going until the model has
        # done ``steps`` *actual* forward steps — rolled-back steps don't count
        # toward progress. The training dataset repeats infinitely, so iterate
        # it directly and stop on the step count; the progress bar is driven
        # manually (its position moves backwards on a rollback, and the rate/ETA
        # in job_status skips the negative interval rather than going negative).
        pbar = tqdm(
            total=steps,
            initial=start_step,
            desc="Training",
            unit="step",
            ncols=120,
        )

        self.now = time.perf_counter()
        n_rollbacks = 0   # rollbacks since the last LR halving
        n_halvings  = 0   # LR halvings this run (divergence guard)

        for batch in train_dataset:
            if int(ckpt.step.numpy()) >= steps:
                break
            ckpt.step.assign_add(1)
            step = int(ckpt.step.numpy())
            # Follow the LR schedule (× the guard's halving scale). No-op for a
            # constant LR with no halvings yet; assigns in place otherwise.
            if self._lr_schedule is not None or self._lr_scale != 1.0:
                self._apply_lr(step)
            # Dispatch on ``lane_counts``. When None (the pure-supervised
            # path used by run_pipeline.py / cli/main.py / the web
            # inference helpers and every validation stream), batches are
            # plain ``(lr, hr)`` 2-tuples → ``train_step``. When set, the
            # dataset is a fixed contiguous ``[n_syn | n_hst | n_anchor]``
            # layout → ``train_step_sky``, which slices each lane by its
            # static count and applies that lane's forward op (no
            # per-example branching). The counts are Python ints, so the
            # @tf.function traces once for the run's fixed layout.
            lr, hr = batch
            if lane_counts is not None:
                loss, gnorm, lane_loss = self.train_step_sky(lr, hr, *lane_counts)
            else:
                loss, gnorm = self.train_step(lr, hr)
                lane_loss = None

            # Divergence rollback (checked BEFORE accumulation so the spike's
            # huge values never poison the window means or trigger an eval). A
            # post-warmup spike means a bad batch / Adam-v→0 step is dragging
            # the model toward the collapse basin; skipping it only FREEZES a
            # diverged model, so instead restore the last good checkpoint
            # (model + optimiser state) and continue from before the spike.
            if _is_grad_spike(gnorm, step):
                n_rollbacks += 1
                msg = (f"⚠ gradient spike |g|={float(gnorm):.3g} at step {step}"
                       f" — restored last checkpoint "
                       f"(rollback {n_rollbacks}/{GRAD_SPIKE_MAX_ROLLBACKS})")
                tqdm.write("  " + msg)
                if warn_callback is not None:
                    warn_callback(msg)
                if ckpt_mgr.latest_checkpoint:
                    # ``restore`` rewinds the model, ckpt.step AND the optimiser
                    # LR to the checkpoint. We KEEP the rewound step (honest: the
                    # run re-trains the rolled-back steps so it still does
                    # ``steps`` real forward steps), and re-assert the intended
                    # (possibly halved) LR.
                    self.checkpoint.restore(
                        ckpt_mgr.latest_checkpoint).expect_partial()
                step = int(ckpt.step.numpy())   # rewound model step
                self._apply_lr(step)
                # Move the progress bar back to the model's real step. The
                # rate/ETA in job_status skips the negative interval (it never
                # folds Δsteps ≤ 0), so this doesn't poison speed/ETA.
                pbar.n = max(0, step)
                pbar.refresh()
                if step_callback is not None:
                    step_callback(step, int(steps))
                # Discard the current eval window — its samples came from the
                # now rolled-back model state, so they'd skew the next mean.
                loss_mean.reset_state()
                loss_syn_mean.reset_state()
                loss_hst_mean.reset_state()
                loss_anchor_mean.reset_state()
                gnorm_mean.reset_state()
                gnorm_max.assign(0.0)
                # A rollback rewinds ckpt.step, so forget the plateau guard's
                # stall history — otherwise ``step - best_step`` goes negative
                # and the guard silently disarms until the step catches back up.
                if self._plateau is not None:
                    self._plateau.reset(step)
                self._psnr_best_step = min(self._psnr_best_step, step)
                self._gap_streak = 0
                # Repeated divergence ⇒ the LR is too hot ⇒ HALVE it and keep
                # going, rather than aborting. Give up only after too many
                # halvings (LR cut to a useless fraction of the original).
                if n_rollbacks >= GRAD_SPIKE_MAX_ROLLBACKS:
                    n_halvings += 1
                    self._lr_scale *= 0.5
                    n_rollbacks = 0
                    new_lr = self._apply_lr(step)
                    hmsg = (f"↓ {GRAD_SPIKE_MAX_ROLLBACKS} rollbacks — halved "
                            f"learning rate to {new_lr:.3g} "
                            f"(halving {n_halvings}/{GRAD_SPIKE_MAX_LR_HALVINGS})"
                            f" and continuing")
                    tqdm.write("  " + hmsg)
                    if warn_callback is not None:
                        warn_callback(hmsg)
                    if n_halvings > GRAD_SPIKE_MAX_LR_HALVINGS:
                        abort = (f"✗ {n_halvings} LR halvings and still "
                                 f"diverging — aborting. The setup is unstable; "
                                 f"check the data/loss scale.")
                        tqdm.write("  " + abort)
                        if warn_callback is not None:
                            warn_callback(abort)
                        break
                continue   # skip metric accumulation / eval for the spiked step

            pbar.update(1)   # normal forward step (model advanced by 1)

            if lane_loss is not None:
                # ``lane_loss`` = (syn, hst, anchor) unweighted mean losses
                # (0.0 for off lanes). Accumulate only the active lanes so a
                # disabled lane's column stays blank, not a misleading 0.
                loss_syn_mean(lane_loss[0])
                if active_hst:
                    loss_hst_mean(lane_loss[1])
                if active_anchor:
                    loss_anchor_mean(lane_loss[2])
            else:
                loss_syn_mean(loss)          # pure-supervised = synthetic only
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
                # Per-lane mean losses over the window; "" for off lanes so
                # the column is blank rather than a misleading 0.0.
                loss_syn    = float(loss_syn_mean.result().numpy())
                loss_hst    = float(loss_hst_mean.result().numpy()) if active_hst else ""
                loss_anchor = float(loss_anchor_mean.result().numpy()) if active_anchor else ""
                gnorm_avg   = gnorm_mean.result()
                gnorm_peak  = float(gnorm_max.numpy())
                loss_mean.reset_state()
                loss_syn_mean.reset_state()
                loss_hst_mean.reset_state()
                loss_anchor_mean.reset_state()
                gnorm_mean.reset_state()
                gnorm_max.assign(0.0)

                # Validation + composite score (same code path as the
                # resume baseline, so the two are directly comparable).
                v = self._validate(
                    valid_dataset, hst_valid_dataset, anchor_valid_dataset,
                    validate_images, save_best_weights,
                )
                psnr_str        = v["psnr_str"]
                psnr_raw        = v["psnr_raw"]
                psnr_str_hst    = v["psnr_str_hst"]
                psnr_raw_hst    = v["psnr_raw_hst"]
                anchor_val_psnr = v["anchor_val_psnr"]
                save_best_score = v["save_best_score"]
                # Second save-best key: combined VALIDATION loss (lower =
                # better), same lanes/weights as the PSNR composite, computed
                # on the held-out images in _validate (NOT the training
                # window) — the held-out analogue of the optimised loss.
                combined_loss = v["combined_loss"]

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
                if anchor_valid_dataset is not None and anchor_val_psnr != "":
                    status += f" | anchor PSNR = {anchor_val_psnr:.3f} dB"
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
                    "loss_syn":       loss_syn,
                    "loss_hst":       loss_hst,
                    "loss_anchor":    loss_anchor,
                    "psnr_stretched": psnr_str,
                    "psnr_raw":       psnr_raw,
                    **v["psnr_bands"],
                    "gnorm_avg":      float(gnorm_avg.numpy()),
                    "gnorm_max":      float(gnorm_peak),
                    "clip_norm":      float(GRAD_CLIP_NORM),
                    "duration_s":     float(duration),
                    "psnr_stretched_hst": psnr_str_hst,
                    "psnr_raw_hst":       psnr_raw_hst,
                    "anchor_val_psnr":    anchor_val_psnr,
                    "save_best_score":    save_best_score,
                    "combined_loss":      combined_loss,
                    "is_baseline":        "",
                }
                train_log.append(row)

                # TWO independent save-best tracks, each with its own
                # checkpoint set. ``ckpt.psnr`` / ``ckpt.best_loss`` are the
                # checkpointed bars (seeded by the baseline eval on resume).
                #   - PSNR track  (higher = better) → ``ckpt_mgr`` (root dir)
                #   - LOSS track  (lower  = better) → ``loss_mgr`` (loss_best/)
                # ``save_best_only=False`` (save-every) makes both fire each
                # eval. The combined loss is +inf when no lane is active, so
                # the loss track simply never triggers in that degenerate case.
                psnr_improved = (
                    not save_best_only or save_best_score > ckpt.psnr)
                loss_improved = (
                    not save_best_only or combined_loss < float(ckpt.best_loss))

                # Emit this evaluate's metrics to the structured event stream
                # BEFORE any ``continue`` so every eval reaches the WebUI.
                # ``saved`` is true if EITHER track saved a checkpoint.
                if eval_callback is not None:
                    eval_callback({**row, "total": int(steps),
                                   "saved": bool(psnr_improved or loss_improved)})

                # ``.assign`` keeps the bars tf.Variables (checkpoint-tracked
                # across resumes, expose .numpy()) rather than replacing them
                # with bare tensors.
                if psnr_improved:
                    ckpt.psnr.assign(save_best_score)
                    ckpt_mgr.save()
                    self._psnr_best_step = step
                    self._gap_streak = 0
                    tqdm.write(
                        f"  ✓ Checkpoint saved [best PSNR] "
                        f"(score={save_best_score:.3f}; "
                        f"PSNR str={psnr_str:.3f}, raw={psnr_raw:.3f} dB)"
                    )
                    # Plateau cuts are provisional: a NEW best at the reduced
                    # LR proves the stall broke, so hand back one cut (raise
                    # the LR ×1/factor toward the schedule value). Only in
                    # save-best mode — save-every "improves" each eval.
                    if (save_best_only and self._plateau_lr_recovery
                            and self._plateau_cuts > 0):
                        before = self._apply_lr(step)
                        self._lr_scale, self._plateau_cuts = \
                            _plateau_recovery_step(
                                self._lr_scale, self._plateau_lr_factor,
                                self._plateau_cuts)
                        after = self._apply_lr(step)
                        # A recovery is proof of progress — re-arm the stall
                        # counter, so the loss watcher can't cut the LR back
                        # in the very same eval (the ↑…↓ churn at the end of
                        # job 27315806).
                        if self._plateau is not None:
                            self._plateau.reset(step)
                        rmsg = (f"↑ new best at reduced LR — raised learning "
                                f"rate back {before:.3g} → {after:.3g} "
                                f"({self._plateau_cuts} plateau cut(s) "
                                f"still applied)")
                        tqdm.write("  " + rmsg)
                        if warn_callback is not None:
                            warn_callback(rmsg)
                if loss_improved and np.isfinite(combined_loss):
                    ckpt.best_loss.assign(combined_loss)
                    self.loss_checkpoint_manager.save()
                    tqdm.write(
                        f"  ✓ Checkpoint saved [best LOSS] → loss_best/ "
                        f"(combined_loss={combined_loss:.5f})"
                    )

                # Stamp the checkpoint dir with its model identity whenever a
                # track saved, so SR outputs can later tell this model apart
                # from a stale one. Best-effort; never raises.
                if psnr_improved or loss_improved:
                    self._emit_checkpoint_provenance()

                # ── Degenerate-basin detector (PSNR-based). The basin's
                # signature lives in PSNR, not the loss: the score sits FLAT
                # and well below the run's best (the frozen ~43.5 dB skip-only
                # floor) while combined_loss micro-creeps. Fire only when the
                # PSNR has made no new best for ``patience`` steps AND the
                # gap persisted for ``PLATEAU_ROLLBACK_MIN_EVALS`` consecutive
                # evals (a single noisy validation dip must not trigger).
                # Response: restore the best-PSNR checkpoint and cool — an
                # in-place cut would only polish the collapsed solution.
                if save_best_only and self._plateau is not None:
                    below = _plateau_wants_rollback(
                        save_best_score, float(ckpt.psnr.numpy()),
                        min_gap=self._plateau_rollback_min_gap,
                        has_best_ckpt=bool(ckpt_mgr.latest_checkpoint),
                        save_best_only=save_best_only)
                    self._gap_streak = self._gap_streak + 1 if below else 0
                    stalled = (step - self._psnr_best_step
                               >= self._plateau.patience)
                    before = self._apply_lr(step)
                    floored = before <= self._min_lr * (1.0 + 1e-9)
                    if (stalled and not floored and self._gap_streak
                            >= int(Config.PLATEAU_ROLLBACK_MIN_EVALS)):
                        best_score = float(ckpt.psnr.numpy())
                        self._lr_scale *= self._plateau_lr_factor
                        self._plateau_cuts += 1
                        # Same mechanics as the gradient-spike rollback:
                        # weights + optimizer + step rewind to the best-PSNR
                        # checkpoint; the eval-window stats came from the
                        # collapsed model, so discard them; re-arm both
                        # watchers at the rewound step.
                        self.checkpoint.restore(
                            ckpt_mgr.latest_checkpoint).expect_partial()
                        step = int(ckpt.step.numpy())
                        pbar.n = max(0, step)
                        pbar.refresh()
                        if step_callback is not None:
                            step_callback(step, int(steps))
                        loss_mean.reset_state()
                        loss_syn_mean.reset_state()
                        loss_hst_mean.reset_state()
                        loss_anchor_mean.reset_state()
                        gnorm_mean.reset_state()
                        gnorm_max.assign(0.0)
                        self._plateau.reset(step)
                        self._psnr_best_step = step
                        self._gap_streak = 0
                        after = self._apply_lr(step)
                        pmsg = (f"↺ degenerate plateau (PSNR stalled "
                                f"≥{self._plateau.patience} steps at "
                                f"{float(save_best_score):.3f} vs best "
                                f"{best_score:.3f}) — restored best-PSNR "
                                f"checkpoint @ step {step} and reduced "
                                f"learning rate {before:.3g} → {after:.3g}")
                        tqdm.write("  " + pmsg)
                        if warn_callback is not None:
                            warn_callback(pmsg)

                # ── Converged plateau (the watched metric, combined_loss by
                # default, flat for ``patience`` steps with a RELATIVE
                # min-delta): cut the LR in place via the SAME ``_lr_scale``
                # the spike guard uses. Skip once at the ``min_lr`` floor.
                if self._plateau is not None:
                    metric = (combined_loss
                              if self._plateau_lr_metric == "combined_loss"
                              else psnr_str)
                    if self._plateau.should_reduce(step, metric):
                        before = self._apply_lr(step)
                        floored = before <= self._min_lr * (1.0 + 1e-9)
                        if not floored:
                            self._lr_scale *= self._plateau_lr_factor
                            self._plateau_cuts += 1
                            after = self._apply_lr(step)
                            pmsg = (f"↓ plateau ({self._plateau_lr_metric} "
                                    f"flat for ≥{self._plateau.patience} "
                                    f"steps) — reduced learning rate "
                                    f"{before:.3g} → {after:.3g}")
                            tqdm.write("  " + pmsg)
                            if warn_callback is not None:
                                warn_callback(pmsg)

                self.now = time.perf_counter()

        pbar.close()

    @tf.function
    def train_step(self, lr, hr):
        """
        Perform one supervised training step (pure-supervised API).

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
        gradients, gnorm = tf.clip_by_global_norm(
            gradients, clip_norm=GRAD_CLIP_NORM)
        self.checkpoint.optimizer.apply_gradients(
            zip(gradients, self.checkpoint.model.trainable_variables, strict=False)
        )

        return loss_value, gnorm

    def _add_nonneg_penalty(self, loss_value, sr):
        """Add ``λ · mean(relu(-SR))`` to ``loss_value`` (no-op when λ=0).

        SR is the model's single output, so this one term — applied to the
        whole batch's ``sr`` — penalises negativity for every lane at once
        (synthetic, HST, star-anchor all branch from this SR). Penalised in
        asinh space; ``relu(-sr)`` is 0 where sr ≥ 0 and grows linearly
        with how negative it is, a constant upward push on negative pixels.
        ``λ`` is a Python float, so this resolves at trace time.
        """
        if self.nonneg_sr_weight > 0:
            penalty = tf.reduce_mean(tf.nn.relu(-sr))
            return loss_value + self.nonneg_sr_weight * penalty
        return loss_value

    @tf.function
    def train_step_sky(self, lr, hr, n_syn: int, n_hst: int, n_anchor: int):
        """``SR = deconvolved sky`` step on a fixed-layout batch.

        The batch lanes are laid out in fixed, contiguous blocks so the
        slices below are static Python ints (no per-step ``tf.gather``, no
        retracing): ``[0:n_syn]`` synthetic, then ``n_hst`` HST, then
        ``n_anchor`` star-anchor. Each lane supervises the single sky
        estimate ``SR`` (4-band since the VIS+NISP-output change):

          * synthetic — ``|SR − scene|`` (direct; LR was E⊛scene, so the
            clean 4-band target *is* the sky). Supervises every band.
          * HST       — ``|asinh(H ⊛ SR_VIS) − HST_image|`` (no rebin;
            HST shares SR's 0.05″ grid). F814W is a VIS analogue only,
            so this lane supervises SR's channel 0 (VIS) alone — the
            loader zero-pads the 1-channel HST target and the slice
            below compares just that channel.
          * star-anchor — masked ``|SR − delta_target|``, evaluated ONLY at
            the star pixels (``delta_target > 0``). The target is a sparse
            HR image: catalog VIS flux at the star, zero elsewhere (the
            zero-padded NISP planes stay outside the mask, so this lane
            too touches only the VIS channel). Operator-free — no PSF —
            so it cannot be poisoned by PSF mismatch; it pins real stars
            to points of known flux.

        All comparisons are in asinh space (the stretch the records and
        the model output already use); the HST conv runs in linear space
        (``inverse_asinh`` → conv → ``asinh``). Per-lane loss weights scale
        each block. ``hr`` carries the scene / HST image / delta-target.

        ``n_syn``/``n_hst``/``n_anchor`` are Python ints, so these guards and
        the slice logic resolve at trace time — a lane is compiled into the
        graph only when its count is non-zero.

        Returns ``(loss_value, gnorm, (syn_loss, hst_loss, anchor_loss))`` —
        the optimised composite loss, the gradient global-norm, and the three
        per-lane *unweighted* mean losses (0.0 for an off lane) for the
        metrics CSV.
        """
        if n_hst > 0 and self.hst_forward_op is None:
            raise ValueError(
                "train_step_sky: n_hst > 0 requires hst_forward_op to be set "
                "(HSTForwardOp); the HST lane convolves SR with the HST PSF"
            )
        with tf.GradientTape() as tape:
            sr     = self.checkpoint.model(lr, training=True)   # asinh space
            sr_lin = inverse_asinh_stretch_hr(sr)               # linear electrons
            per_example = []   # list of [n_*] loss vectors

            i = 0
            # Per-lane *unweighted* mean losses, returned for the metrics
            # CSV (0.0 for an off lane). The optimised ``loss_value`` still
            # uses the per-lane weights via ``per_example``.
            syn_loss    = tf.zeros([], tf.float32)
            hst_loss    = tf.zeros([], tf.float32)
            anchor_loss = tf.zeros([], tf.float32)
            if n_syn > 0:
                s = slice(i, i + n_syn)
                d = tf.reduce_mean(tf.abs(sr[s] - hr[s]), axis=[1, 2, 3])
                per_example.append(self.synthetic_loss_weight * d)
                syn_loss = tf.reduce_mean(d)
                i += n_syn
            if n_hst > 0:
                s = slice(i, i + n_hst)
                # F814W ≈ VIS only: forward-model SR's VIS channel and
                # compare against the (zero-padded) target's channel 0.
                hconv = asinh_stretch_hr(
                    self.hst_forward_op(sr_lin[s][..., :1]))
                d = tf.reduce_mean(tf.abs(hconv - hr[s][..., :1]),
                                   axis=[1, 2, 3])
                per_example.append(self.hst_loss_weight * d)
                hst_loss = tf.reduce_mean(d)
                i += n_hst
            if n_anchor > 0:
                s = slice(i, i + n_anchor)
                target = hr[s]                                  # asinh delta-target
                mask   = tf.cast(target > 0, target.dtype)      # star pixels only
                num    = tf.reduce_sum(mask * tf.abs(sr[s] - target), axis=[1, 2, 3])
                den    = tf.reduce_sum(mask, axis=[1, 2, 3]) + 1e-6
                anchor_per_example = num / den
                per_example.append(self.star_anchor_loss_weight * anchor_per_example)
                anchor_loss = tf.reduce_mean(anchor_per_example)

            loss_value = tf.reduce_mean(tf.concat(per_example, axis=0))
            # One non-negativity penalty on the shared SR covers all lanes.
            loss_value = self._add_nonneg_penalty(loss_value, sr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        gradients, gnorm = tf.clip_by_global_norm(
            gradients, clip_norm=GRAD_CLIP_NORM)
        self.checkpoint.optimizer.apply_gradients(
            zip(gradients, self.checkpoint.model.trainable_variables, strict=False)
        )
        return loss_value, gnorm, (syn_loss, hst_loss, anchor_loss)

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

    def evaluate_anchor(self, anchor_dataset) -> "tuple[float, float]":
        """Mean star-anchor ``(PSNR_dB, MAE)`` over the masked star pixels.

        Scores SR only where the sparse HR delta-target is non-zero (the
        star pixel = catalog VIS flux), in eval mode without gradients::

            sr   = model(lr, training=False)              # asinh space
            mask = (hr > 0)                               # star pixel(s)
            mse  = mean_over_mask((sr − hr)²)             # per image
            psnr = 10·log10(PSNR_PEAK_STRETCHED² / mse)

        Same asinh space and peak (``Config.PSNR_PEAK_STRETCHED``) as the
        synthetic / HST PSNRs, so all metrics share one dB scale. Higher =
        the model places the right flux at the star. Operator-free (no PSF).
        Because the target is a single pixel, a near-exact match would send
        the raw PSNR → +∞; each star's PSNR is therefore clipped to
        ``ANCHOR_PSNR_MAX_DB`` so the metric stays finite and bounded. The
        second value is the masked validation MAE at the star pixel(s) — the
        held-out analogue of the anchor training loss, for the combined-loss
        save-best track. Returns ``(nan, nan)`` when no image carries a star.

        Parameters
        ----------
        anchor_dataset : tf.data.Dataset
            Yields ``(lr, hr)`` with ``hr`` the asinh-stretched delta-target.
        """
        running = Mean()
        saw = False
        peak2   = tf.constant(float(Config.PSNR_PEAK_STRETCHED) ** 2, dtype=tf.float32)
        ln10    = tf.constant(2.302585092994046, dtype=tf.float32)
        mae_running = Mean()
        max_psnr = tf.constant(ANCHOR_PSNR_MAX_DB, dtype=tf.float32)
        for lr, hr in anchor_dataset:
            sr   = self.checkpoint.model(lr, training=False)
            mask = tf.cast(hr > 0, sr.dtype)
            cnt  = tf.reduce_sum(mask, axis=[1, 2, 3])
            sse  = tf.reduce_sum(mask * (sr - hr) ** 2, axis=[1, 2, 3])
            # Masked MAE at the star pixel(s) — the validation analogue of
            # the star-anchor training loss (same mask, asinh space).
            sae  = tf.reduce_sum(mask * tf.abs(sr - hr), axis=[1, 2, 3])
            valid = cnt > 0
            mse  = sse / tf.maximum(cnt, 1.0)
            mae  = sae / tf.maximum(cnt, 1.0)
            psnr = 10.0 * tf.math.log(peak2 / tf.maximum(mse, 1e-12)) / ln10
            # Clip per-star so one nailed pixel can't spike the metric to ~140 dB.
            psnr = tf.minimum(psnr, max_psnr)
            psnr = tf.boolean_mask(psnr, valid)
            if int(tf.size(psnr)) > 0:
                running(tf.reduce_mean(psnr))
                mae_running(tf.reduce_mean(tf.boolean_mask(mae, valid)))
                saw = True
        if not saw:
            return float("nan"), float("nan")
        return float(running.result().numpy()), float(mae_running.result().numpy())

    def evaluate_hst(self, hst_dataset) -> dict:
        """HST-source validation through the forward op (SR=sky-consistent).

        Mirrors the HST training lane: compares ``H ⊛ SR`` to the observed
        HST image rather than ``SR`` directly. Scoring ``SR`` against the
        HST image would reward HST-PSF blur and contradict the synthetic /
        anchor lane (which targets the deconvolved sky), so the metric
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
            return {"psnr_stretched": nan, "psnr_raw": nan, "mae_stretched": nan}
        str_mean = Mean()
        raw_mean = Mean()
        mae_mean = Mean()
        max_str = tf.constant(float(Config.PSNR_PEAK_STRETCHED), dtype=tf.float32)
        max_raw = tf.constant(float(Config.PSNR_PEAK_E), dtype=tf.float32)
        for lr, hr in hst_dataset:
            sr        = self.checkpoint.model(lr, training=False)
            # F814W ≈ VIS only — score SR's channel 0 against the
            # (possibly zero-padded) HST target's channel 0.
            sr_vis    = sr[..., :1]
            hr_vis    = hr[..., :1]
            sr_lin    = inverse_asinh_stretch_hr(sr_vis)
            hconv_lin = self.hst_forward_op(sr_lin)
            hconv_str = asinh_stretch_hr(hconv_lin)
            str_mean(tf.reduce_mean(
                tf.image.psnr(hr_vis, hconv_str, max_val=max_str)))
            # Validation MAE in asinh space, on the SAME forward-consistent
            # quantity the HST lane optimises (H⊛SR vs observed HST).
            mae_mean(tf.reduce_mean(tf.abs(hr_vis - hconv_str)))
            hr_lin = inverse_asinh_stretch_hr(hr_vis)
            raw_mean(tf.reduce_mean(
                tf.image.psnr(hr_lin, hconv_lin, max_val=max_raw)))
        return {"psnr_stretched": str_mean.result(),
                "psnr_raw": raw_mean.result(),
                "mae_stretched": mae_mean.result()}

    def restore(self, track: str = "latest"):
        """Resume from a checkpoint.

        ``track="latest"`` (default): the MOST RECENT checkpoint across both
        save-best tracks. The PSNR track (root dir) and the LOSS track
        (``loss_best/``) wrap the same checkpoint object but save on different
        triggers, so the higher ``step`` is whichever metric improved most
        recently.

        ``track="psnr"``: the PSNR-best track ONLY — the model evaluation
        actually uses. Continue-mode training resumes here: after a degenerate
        stretch (skip-only collapse) the loss track holds the collapsed
        weights at a HIGHER step, so max-step resume would continue from the
        wrong model.
        """
        managers = ((self.checkpoint_manager,) if track == "psnr"
                    else (self.checkpoint_manager, self.loss_checkpoint_manager))
        candidates = [
            m.latest_checkpoint for m in managers if m.latest_checkpoint
        ]
        if not candidates:
            return
        best, best_step = None, -1
        for path in candidates:
            # Restore is cheap (small model); read the authoritative step the
            # same way the rollback path does, then keep the latest one.
            self.checkpoint.restore(path).expect_partial()
            s = int(self.checkpoint.step.numpy())
            if s >= best_step:
                best, best_step = path, s
        if best != candidates[-1]:
            self.checkpoint.restore(best).expect_partial()
        print(f"Model restored from checkpoint at step {self.checkpoint.step.numpy()}.")
