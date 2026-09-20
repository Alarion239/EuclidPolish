"""Tests for the supervised SR trainer.

The model estimates one quantity — the deconvolved sky ``SR`` — and
``train_step(lr, hr)`` supervises it on synthetic pairs. These tests pin
the gradient path, the LR guards, save-best bookkeeping, the resume
baseline, the non-negativity penalty and the training-log schema.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.image.tfio import open_writer, tfrecord_path
from euclid_polish.training.augmentation import (
    asinh_stretch_hr,
    asinh_stretch_lr,
    lr_only_dataset,
)
from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.training.trainer import (
    GRAD_SPIKE_SKIP_NORM,
    GRAD_SPIKE_SKIP_WARMUP_STEPS,
    TRAINING_LOG_COLUMNS,
    Trainer,
    _is_grad_spike,
)


def test_is_grad_spike_detects_post_warmup_spike_and_nan():
    """After warmup, a non-finite or over-threshold pre-clip norm is a spike
    (→ the loop rolls back); an ordinary norm is not."""
    assert GRAD_SPIKE_SKIP_NORM > 0
    post = GRAD_SPIKE_SKIP_WARMUP_STEPS + 1

    assert not _is_grad_spike(0.5, post)                       # ordinary
    assert _is_grad_spike(GRAD_SPIKE_SKIP_NORM * 100.0, post)  # huge spike
    assert _is_grad_spike(float("inf"), post)                  # non-finite
    assert _is_grad_spike(float("nan"), post)                  # non-finite


def test_is_grad_spike_inert_during_warmup():
    """During warmup the guard is inert — even a huge early gradient is not a
    spike, so the model learns through the warmup instead of rolling back."""
    spike = GRAD_SPIKE_SKIP_NORM * 100.0
    assert not _is_grad_spike(spike, max(0, GRAD_SPIKE_SKIP_WARMUP_STEPS - 1))
    assert not _is_grad_spike(spike, 1)


def test_apply_lr_follows_schedule_and_halves(tmp_path):
    """The optimiser is built with a settable constant LR; ``_apply_lr``
    follows a passed schedule (× the guard's halving scale)."""
    from tf_keras.optimizers.schedules import PiecewiseConstantDecay

    from euclid_polish.training.models.wdsr import wdsr

    m = wdsr(scale=2, num_res_blocks=1, nchan_in=4, nchan_out=1)
    sch = PiecewiseConstantDecay(boundaries=[100], values=[1e-3, 5e-4])
    tr = Trainer(model=m, learning_rate=sch, checkpoint_dir=str(tmp_path))

    assert tr._lr_schedule is not None
    assert abs(tr._apply_lr(10) - 1e-3) < 1e-9       # pre-boundary
    assert abs(tr._apply_lr(150) - 5e-4) < 1e-9      # post-boundary (follows schedule)

    tr._lr_scale *= 0.5                              # guard halves
    assert abs(tr._apply_lr(10) - 5e-4) < 1e-9       # 1e-3 × 0.5
    # the optimiser actually picked up the halved value
    assert abs(float(tr.checkpoint.optimizer.learning_rate) - 5e-4) < 1e-9


# ---------------------------------------------------------------------------
# Helpers — tiny model so tests stay fast on CPU
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model():
    """Smallest WDSR that still has the (4-ch in, 1-ch out, scale=2) contract."""
    return wdsr(scale=2, nchan_in=4, nchan_out=1, num_res_blocks=1, num_filters=4)


@pytest.fixture
def tiny_trainer(tiny_model, tmp_path):
    return Trainer(tiny_model, checkpoint_dir=str(tmp_path / "ckpt"))


# ---------------------------------------------------------------------------
# Reproducibility: seed application + provenance recording
# ---------------------------------------------------------------------------

def test_seed_everything_is_reproducible():
    from euclid_polish.training.trainer import seed_everything
    seed_everything(7)
    a = tf.random.uniform([5]).numpy()
    seed_everything(7)
    b = tf.random.uniform([5]).numpy()
    assert (a == b).all()


def test_trainer_records_seed_and_links_checkpoint(tiny_model, tmp_path,
                                                   monkeypatch):
    """A seeded Trainer records a Process.training carrying the seed, and the
    checkpoint identity stamp's produced_by points back to that run."""
    import euclid_polish.training.trainer as trainer_mod
    from euclid_polish.provenance.checkpoint import read_checkpoint_provenance
    from euclid_polish.provenance.store import ProvStore

    store = ProvStore(str(tmp_path / "prov"))
    monkeypatch.setattr(trainer_mod, "default_store", lambda: store)

    ckpt = str(tmp_path / "ckpt")
    tr = Trainer(
        tiny_model,
        checkpoint_dir=ckpt,
        seed=12345,
        provenance_fields={"noise_model": "test-noise-model"},
    )
    tr._begin_reproducible_run(steps=10, evaluate_every=5)

    assert tr._training_run_id is not None
    run = store.get(tr._training_run_id)
    assert run.kind == "trainingrun" and run.seed == 12345
    assert run.config.fields["noise_model"] == "test-noise-model"

    tr._emit_checkpoint_provenance()
    stamp = read_checkpoint_provenance(ckpt)
    assert stamp is not None and stamp.produced_by == tr._training_run_id


def test_unseeded_trainer_records_no_run(tiny_model, tmp_path):
    """Default (no seed) keeps prior behaviour: no Process.training recorded."""
    tr = Trainer(tiny_model, checkpoint_dir=str(tmp_path / "ckpt"))
    tr._begin_reproducible_run(steps=10, evaluate_every=5)
    assert tr._training_run_id is None


def _rand_batch(batch_size: int = 2, lr_side: int = 8, hr_side: int = 16,
                seed: int = 0):
    rng = np.random.default_rng(seed)
    lr = tf.constant(
        rng.normal(size=(batch_size, lr_side, lr_side, 4)).astype(np.float32),
    )
    hr = tf.constant(
        rng.normal(size=(batch_size, hr_side, hr_side, 1)).astype(np.float32),
    )
    return lr, hr


# ---------------------------------------------------------------------------
# Supervised training path
# ---------------------------------------------------------------------------

class TestSupervisedTraining:

    def test_supervised_train_step_returns_finite_loss(self, tiny_trainer):
        """The supervised ``train_step(lr, hr)`` path stays finite."""
        lr, hr = _rand_batch()
        loss, gnorm = tiny_trainer.train_step(lr, hr)
        assert np.isfinite(float(loss.numpy()))
        assert np.isfinite(float(gnorm.numpy()))

    def test_supervised_loss_decreases_on_repeated_steps(self, tiny_trainer):
        """A few supervised steps on a fixed batch should reduce the loss —
        catches catastrophic regressions in the existing gradient path."""
        lr, hr = _rand_batch()
        losses = []
        for _ in range(5):
            loss, _ = tiny_trainer.train_step(lr, hr)
            losses.append(float(loss.numpy()))
        # The 5th-step loss must be lower than the 1st — not strict
        # monotone because optimiser state shifts on the first call.
        assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def _valid_pairs_dataset(n: int = 2, lr_side: int = 8, hr_side: int = 16,
                         seed: int = 1, batch_size: int = 1):
    """Tiny ``(lr, hr)`` validation dataset the trainer's ``evaluate``
    consumes — batched 4-D tensors, hr at 2× the lr side."""
    rng = np.random.default_rng(seed)
    lr = rng.normal(size=(n, lr_side, lr_side, 4)).astype(np.float32)
    hr = rng.normal(size=(n, hr_side, hr_side, 1)).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((lr, hr))
    return ds.batch(batch_size)


def _train_pairs_dataset(n: int = 4, lr_side: int = 8, hr_side: int = 16,
                         seed: int = 2, batch_size: int = 2):
    """A small repeating supervised training dataset (2-tuples)."""
    rng = np.random.default_rng(seed)
    lr = rng.normal(size=(n, lr_side, lr_side, 4)).astype(np.float32)
    hr = rng.normal(size=(n, hr_side, hr_side, 1)).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((lr, hr))
    return ds.batch(batch_size).repeat()


def _write_lr_only_tfrecord(path_dir: str, subset: str = "validate",
                            n: int = 3, side: int = 8, seed: int = 3) -> str:
    """Write a tiny LR-only ``dirty_{subset}.tfrecord`` and return its path."""
    rng = np.random.default_rng(seed)
    with open_writer(f"dirty_{subset}", records_dir=path_dir) as w:
        for i in range(n):
            data = rng.normal(size=(side, side, 4)).astype(np.float32)
            img = Image(
                data=data,
                pixel_scale_arcsec=0.10,
                band_names=Config.LR_INPUT_BAND_NAMES,
                is_clean=False,
                index=i,
                subset=subset,
            )
            w.write(img, index=i)
    return tfrecord_path(path_dir, f"dirty_{subset}")


class TestLrOnlyDataset:

    def test_lr_only_dataset_shape(self, tmp_path):
        """``lr_only_dataset`` yields batched LR tensors ``[B, H, W, 4]``."""
        path = _write_lr_only_tfrecord(str(tmp_path / "rt"), n=3, side=8)
        ds = lr_only_dataset(path, batch_size=2)
        batches = list(ds)
        assert len(batches) == 2          # 3 records → batches of 2 + 1
        b0 = batches[0]
        assert b0.shape.as_list() == [2, 8, 8, 4]
        assert batches[1].shape.as_list() == [1, 8, 8, 4]
        assert b0.dtype == tf.float32


class TestValidationLogging:

    def _read_log(self, ckpt_dir: str):
        import csv
        log_path = os.path.join(ckpt_dir, "training_log.csv")
        with open(log_path, newline="") as fh:
            return list(csv.DictReader(fh))

    def test_second_track_saves_loss_best_checkpoints(
        self, tiny_model, tmp_path,
    ):
        """The combined-loss save-best track writes its OWN checkpoint set
        under ``loss_best/`` (distinct from the PSNR checkpoints at the root)
        and logs a finite ``combined_loss`` per eval."""
        import tensorflow as tf
        ckpt_dir = str(tmp_path / "ckpt_two_track")
        trainer = Trainer(tiny_model, checkpoint_dir=ckpt_dir)
        # save-every so both tracks fire each eval regardless of the metric.
        trainer.train(
            _train_pairs_dataset(), _valid_pairs_dataset(seed=10),
            steps=2, evaluate_every=1, save_best_only=False, validate_images=4,
        )
        # PSNR-best checkpoints at the root; loss-best in the subfolder.
        assert tf.train.latest_checkpoint(ckpt_dir) is not None
        assert tf.train.latest_checkpoint(
            os.path.join(ckpt_dir, "loss_best")) is not None
        rows = self._read_log(ckpt_dir)
        assert rows, "no training-log rows"
        for r in rows:
            assert r["combined_loss"] != ""
            assert np.isfinite(float(r["combined_loss"]))

    def test_step_based_loop_ends_exactly_at_steps(self, tiny_model, tmp_path):
        """The honest step-based loop runs until ckpt.step == steps (it counts
        *actual* forward steps, so a rollback would re-train rather than
        overshoot the counter)."""
        ckpt_dir = str(tmp_path / "ckpt_steps")
        trainer = Trainer(tiny_model, checkpoint_dir=ckpt_dir)
        trainer.train(
            _train_pairs_dataset(), _valid_pairs_dataset(seed=10),
            steps=5, evaluate_every=5, save_best_only=False, validate_images=2,
        )
        assert int(trainer.checkpoint.step.numpy()) == 5

class TestLogHeaderRotation:

    def test_stale_header_rotated_to_bak(self, tiny_model, tmp_path):
        """A pre-existing log written with the OLD column set is rotated
        to ``training_log.<ts>.bak`` and a fresh file with the new header
        is started."""
        import csv
        import glob
        ckpt_dir = str(tmp_path / "ckpt_rot")
        os.makedirs(ckpt_dir, exist_ok=True)
        log_path = os.path.join(ckpt_dir, "training_log.csv")
        # Write a log with the OLD (pre-multi-source) columns.
        old_cols = [
            "step", "wall_time", "loss", "psnr_stretched", "psnr_raw",
            "gnorm_avg", "gnorm_max", "clip_norm", "duration_s",
        ]
        with open(log_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=old_cols)
            w.writeheader()
            w.writerow(dict.fromkeys(old_cols, 0))

        trainer = Trainer(tiny_model, checkpoint_dir=ckpt_dir)
        trainer.train(
            _train_pairs_dataset(), _valid_pairs_dataset(seed=10),
            steps=1, evaluate_every=1, save_best_only=True,
            validate_images=4,
        )

        # Old file rotated out.
        baks = glob.glob(os.path.join(ckpt_dir, "training_log.*.bak"))
        assert len(baks) == 1, f"expected exactly one .bak, got {baks}"
        # New file has the NEW header.
        with open(log_path, newline="") as fh:
            header = fh.readline().rstrip("\r\n")
        assert header == ",".join(TRAINING_LOG_COLUMNS)
        # The rotated backup retains the old header.
        with open(baks[0], newline="") as fh:
            assert fh.readline().rstrip("\r\n") == ",".join(old_cols)


# ---------------------------------------------------------------------------
# Resume baseline — validate the restored checkpoint instead of force-saving
# ---------------------------------------------------------------------------

def _tiny_wdsr():
    return wdsr(scale=2, nchan_in=4, nchan_out=1, num_res_blocks=1, num_filters=4)


class TestResumeBaseline:

    def _read_log(self, ckpt_dir: str):
        import csv
        with open(os.path.join(ckpt_dir, "training_log.csv"), newline="") as fh:
            return list(csv.DictReader(fh))

    def test_fresh_run_writes_no_baseline_row(self, tmp_path):
        """A from-scratch run has nothing to validate → no is_baseline row."""
        ckpt_dir = str(tmp_path / "ckpt_fresh")
        t = Trainer(_tiny_wdsr(), checkpoint_dir=ckpt_dir)
        t.train(_train_pairs_dataset(), _valid_pairs_dataset(seed=10),
                steps=2, evaluate_every=1, save_best_only=True,
                validate_images=4)
        rows = self._read_log(ckpt_dir)
        assert rows and all(r.get("is_baseline", "") != "1" for r in rows)

    def test_resume_writes_baseline_row_and_seeds_threshold(self, tmp_path):
        """On resume the restored checkpoint is validated under this run's
        setup, one is_baseline row is written at the resumed step, and
        ckpt.psnr is seeded with that score (the bar to beat) — no
        force-save."""
        ckpt_dir = str(tmp_path / "ckpt_resume")
        t1 = Trainer(_tiny_wdsr(), checkpoint_dir=ckpt_dir)
        t1.train(_train_pairs_dataset(), _valid_pairs_dataset(seed=10),
                 steps=2, evaluate_every=1, save_best_only=True,
                 validate_images=4)
        resumed_step = int(t1.checkpoint.step.numpy())
        assert resumed_step == 2

        # New Trainer on the same dir restores the checkpoint (step > 0).
        t2 = Trainer(_tiny_wdsr(), checkpoint_dir=ckpt_dir)
        assert int(t2.checkpoint.step.numpy()) == resumed_step
        t2.train(_train_pairs_dataset(), _valid_pairs_dataset(seed=10),
                 steps=resumed_step + 2, evaluate_every=1,
                 save_best_only=True, validate_images=4)

        rows = self._read_log(ckpt_dir)
        base = [r for r in rows if r.get("is_baseline") == "1"]
        assert len(base) == 1, "exactly one baseline row per resume"
        assert int(base[0]["step"]) == resumed_step
        baseline_score = float(base[0]["psnr_stretched"])
        # The threshold was seeded by the baseline; later evals only raise it.
        assert float(t2.checkpoint.psnr.numpy()) >= baseline_score - 1e-3

    def test_resume_baseline_does_not_overwrite_unbeaten_best(self, tmp_path):
        """If the resumed run never beats the baseline, the seeded threshold
        is preserved (no save-best regression to a worse score)."""
        ckpt_dir = str(tmp_path / "ckpt_keep")
        t1 = Trainer(_tiny_wdsr(), checkpoint_dir=ckpt_dir)
        t1.train(_train_pairs_dataset(), _valid_pairs_dataset(seed=10),
                 steps=2, evaluate_every=1, save_best_only=True,
                 validate_images=4)

        t2 = Trainer(_tiny_wdsr(), checkpoint_dir=ckpt_dir)
        t2.train(_train_pairs_dataset(), _valid_pairs_dataset(seed=10),
                 steps=int(t2.checkpoint.step.numpy()) + 1, evaluate_every=1,
                 save_best_only=True, validate_images=4)
        rows = self._read_log(ckpt_dir)
        base = [r for r in rows if r.get("is_baseline") == "1"][0]
        # ckpt.psnr is never below the measured baseline.
        assert float(t2.checkpoint.psnr.numpy()) >= float(base["psnr_stretched"]) - 1e-3


# ---------------------------------------------------------------------------
# SR non-negativity penalty (λ · mean(relu(-SR)))
# ---------------------------------------------------------------------------

class TestNonNegPenalty:

    def test_penalty_math(self, tiny_model, tmp_path):
        t = Trainer(tiny_model, checkpoint_dir=str(tmp_path / "nn"),
                    nonneg_sr_weight=2.0)
        sr   = tf.constant([[-1.0, 0.0, 3.0, -2.0]], dtype=tf.float32)
        base = tf.constant(0.5, dtype=tf.float32)
        out  = float(t._add_nonneg_penalty(base, sr).numpy())
        # mean(relu(-sr)) = mean([1, 0, 0, 2]) = 0.75 → 0.5 + 2·0.75 = 2.0
        assert out == pytest.approx(2.0, abs=1e-6)

    def test_zero_weight_is_noop(self, tiny_model, tmp_path):
        t = Trainer(tiny_model, checkpoint_dir=str(tmp_path / "nn0"),
                    nonneg_sr_weight=0.0)
        sr   = tf.constant([[-5.0, -5.0]], dtype=tf.float32)
        base = tf.constant(1.0, dtype=tf.float32)
        assert float(t._add_nonneg_penalty(base, sr).numpy()) == pytest.approx(1.0)

    def test_default_weight_from_config(self, tiny_model, tmp_path):
        from euclid_polish.config import Config as _Cfg
        t = Trainer(tiny_model, checkpoint_dir=str(tmp_path / "nnd"))
        assert t.nonneg_sr_weight == pytest.approx(float(_Cfg.NONNEG_SR_WEIGHT))

    def test_penalty_drives_sr_less_negative(self, tmp_path):
        """A few steps with a strong penalty and a ≥0 target reduce the
        negative part of SR — the model learns to output non-negative."""
        model = _tiny_wdsr()
        t = Trainer(model, checkpoint_dir=str(tmp_path / "nn_drive"),
                    nonneg_sr_weight=10.0)
        rng = np.random.default_rng(0)
        lr = tf.constant(rng.normal(size=(4, 8, 8, 4)).astype(np.float32))
        hr = tf.constant(np.abs(rng.normal(size=(4, 16, 16, 1))).astype(np.float32))

        neg_before = float(tf.reduce_mean(tf.nn.relu(-model(lr))).numpy())
        for _ in range(40):
            t.train_step(lr, hr)
        neg_after = float(tf.reduce_mean(tf.nn.relu(-model(lr))).numpy())
        if neg_before < 1e-4:
            pytest.skip("SR already non-negative at init — nothing to drive")
        assert neg_after < neg_before, (
            f"penalty did not reduce negativity: {neg_before:.4f} -> {neg_after:.4f}"
        )

# ---------------------------------------------------------------------------
# Per-band PSNR logging (monitoring only — save-best stays on the joint PSNR)
# ---------------------------------------------------------------------------

class TestPerBandPSNRLogging:

    def _read_log(self, ckpt_dir: str):
        import csv
        with open(os.path.join(ckpt_dir, "training_log.csv"), newline="") as fh:
            return list(csv.DictReader(fh))

    def _pairs_4band(self, n=4, lr_side=8, hr_side=16, seed=2, batch_size=2,
                     repeat=True):
        rng = np.random.default_rng(seed)
        lr = rng.normal(size=(n, lr_side, lr_side, 4)).astype(np.float32)
        hr = rng.normal(size=(n, hr_side, hr_side, 4)).astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((lr, hr)).batch(batch_size)
        return ds.repeat() if repeat else ds

    def test_4band_run_logs_one_psnr_per_band(self, tmp_path):
        """A 4-band model fills every psnr_<band> column with a finite dB
        value; the joint psnr_stretched still drives save-best."""
        from euclid_polish.training.trainer import PER_BAND_PSNR_COLUMNS
        ckpt_dir = str(tmp_path / "ckpt_band")
        model = wdsr(scale=2, nchan_in=4, nchan_out=4,
                     num_res_blocks=1, num_filters=4)
        t = Trainer(model, checkpoint_dir=ckpt_dir)
        t.train(self._pairs_4band(), self._pairs_4band(repeat=False, seed=10),
                steps=1, evaluate_every=1, save_best_only=True,
                validate_images=2)
        rows = self._read_log(ckpt_dir)
        assert rows
        for col in PER_BAND_PSNR_COLUMNS:
            v = rows[-1][col]
            assert v not in ("", None), f"{col} not logged"
            assert np.isfinite(float(v)), f"{col} = {v!r}"
        # Joint metric still present — it is what save-best keys on.
        assert np.isfinite(float(rows[-1]["psnr_stretched"]))

    def test_vis_only_run_leaves_nisp_columns_blank(self, tmp_path):
        """A 1-channel (VIS-only) model logs psnr_vis and leaves the NISP
        band columns blank — not 0, not NaN."""
        from euclid_polish.training.trainer import PER_BAND_PSNR_COLUMNS
        ckpt_dir = str(tmp_path / "ckpt_vis")
        t = Trainer(_tiny_wdsr(), checkpoint_dir=ckpt_dir)
        t.train(_train_pairs_dataset(), _valid_pairs_dataset(seed=10),
                steps=1, evaluate_every=1, save_best_only=True,
                validate_images=2)
        rows = self._read_log(ckpt_dir)
        assert np.isfinite(float(rows[-1][PER_BAND_PSNR_COLUMNS[0]]))
        for col in PER_BAND_PSNR_COLUMNS[1:]:
            assert rows[-1][col] == "", f"{col} should be blank for VIS-only"
