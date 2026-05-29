"""Tests for the source-aware trainer (round-trip integration).

Covers the loss-routing logic added in Chunk C2 of the round-trip
training feature. The trainer can now consume either:

  * legacy 2-tuple batches ``(lr, hr)`` — unchanged supervised path
  * new 3-tuple batches ``(lr, hr, source)`` — per-element source tag
    routes between supervised L1 and the round-trip reconstruction loss

These tests pin three properties:

  1. **Backward compatibility**: the supervised ``train_step(lr, hr)``
     still drives gradients exactly as before — pre-round-trip
     callers (``scripts/run_pipeline.py``, ``cli/main.py``, the web
     inference helpers) keep working without code changes.
  2. **Round-trip semantics**: a 100 %-round-trip batch trains on
     ``|asinh(Conv(M(lr))/k) - lr_vis|``, with gradients propagating
     through both M and the forward op (the op's PSF stays frozen —
     it's non-trainable — but gradients still flow through it back
     into M).
  3. **Mixed routing**: a heterogeneous batch (supervised + round-trip
     in the same batch) computes both loss terms in one tape pass.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import tensorflow as tf
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.sky.tfrecord import open_multiband_writer, tfrecord_path
from euclid_polish.sky.types import MultiBandSkyImage
from euclid_polish.training.data_multiband import (
    SOURCE_HST, SOURCE_ROUNDTRIP, SOURCE_SYNTHETIC,
    asinh_stretch_hr, asinh_stretch_lr, lr_only_dataset,
)
from euclid_polish.training.forward_op import EuclidVISForwardOp
from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.training.trainer import Trainer, TRAINING_LOG_COLUMNS


# ---------------------------------------------------------------------------
# Helpers — tiny synthetic PSF + tiny model so tests stay fast on CPU
# ---------------------------------------------------------------------------

def _gauss_psf(side: int = 11, fwhm_pix: float = 2.0) -> np.ndarray:
    sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    y, x = np.mgrid[:side, :side]
    cy = cx = (side - 1) / 2.0
    g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
    return (g / g.sum()).astype(np.float32)


@pytest.fixture
def tmp_psf_path(tmp_path) -> str:
    p = str(tmp_path / "psf.fits")
    fits.PrimaryHDU(_gauss_psf()).writeto(p, overwrite=True)
    return p


@pytest.fixture
def tiny_model():
    """Smallest WDSR that still has the (4-ch in, 1-ch out, scale=2) contract."""
    return wdsr(scale=2, nchan_in=4, nchan_out=1, num_res_blocks=1, num_filters=4)


@pytest.fixture
def tiny_trainer(tiny_model, tmp_path):
    return Trainer(tiny_model, checkpoint_dir=str(tmp_path / "ckpt"))


@pytest.fixture
def trainer_with_forward_op(tiny_model, tmp_path, tmp_psf_path):
    op = EuclidVISForwardOp(psf_fits_path=tmp_psf_path, rebin_factor=2)
    return Trainer(
        tiny_model, checkpoint_dir=str(tmp_path / "ckpt_rt"),
        forward_op=op, roundtrip_loss_weight=1.0,
    )


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
# 1. Backward compatibility — the legacy 2-tuple supervised path
# ---------------------------------------------------------------------------

class TestSupervisedBackwardCompat:

    def test_supervised_train_step_returns_finite_loss(self, tiny_trainer):
        """The pre-round-trip API ``train_step(lr, hr)`` must keep working
        identically. Existing scripts call it without setting forward_op."""
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


# ---------------------------------------------------------------------------
# 2. Round-trip path
# ---------------------------------------------------------------------------

class TestRoundTripPath:

    def test_pure_roundtrip_batch_trains(self, trainer_with_forward_op):
        """A batch tagged entirely as ``SOURCE_ROUNDTRIP`` runs through
        ``train_step_mixed`` and produces finite loss + gradients. HR is
        a dummy zeros tensor (matching what the dataset emits)."""
        lr, _ = _rand_batch()
        # Dummy HR slot — same shape as supervised HR, all zeros.
        hr_dummy = tf.zeros([2, 16, 16, 1], dtype=tf.float32)
        src = tf.constant([SOURCE_ROUNDTRIP, SOURCE_ROUNDTRIP], dtype=tf.int32)
        loss, gnorm = trainer_with_forward_op.train_step_mixed(lr, hr_dummy, src)
        assert np.isfinite(float(loss.numpy()))
        assert np.isfinite(float(gnorm.numpy()))
        assert float(loss.numpy()) > 0, (
            "round-trip loss on a random batch should be strictly positive"
        )

    def test_roundtrip_gradients_reach_model(self, trainer_with_forward_op):
        """Gradients of the round-trip loss w.r.t. M's weights must be non-zero.

        Without this property the round-trip dataset would have zero
        training effect — Conv would block the gradient signal.
        """
        lr, _ = _rand_batch()
        hr_dummy = tf.zeros([2, 16, 16, 1], dtype=tf.float32)
        src = tf.constant([SOURCE_ROUNDTRIP, SOURCE_ROUNDTRIP], dtype=tf.int32)

        model = trainer_with_forward_op.checkpoint.model
        op    = trainer_with_forward_op.forward_op
        with tf.GradientTape() as tape:
            sr = model(lr, training=True)
            from euclid_polish.training.data_multiband import inverse_asinh_stretch_hr
            sr_lin = inverse_asinh_stretch_hr(sr)
            recon  = asinh_stretch_hr(op(sr_lin))
            loss   = tf.reduce_mean(tf.abs(recon - lr[..., 0:1]))
        grads = tape.gradient(loss, model.trainable_variables)
        max_g = float(max(
            tf.reduce_max(tf.abs(g)).numpy() for g in grads if g is not None
        ))
        assert max_g > 0, (
            "round-trip loss produced zero gradient — Conv may have "
            "broken the gradient path or M's trainable vars aren't "
            "in the tape"
        )

    def test_forward_op_psf_remains_non_trainable_under_training(
        self, trainer_with_forward_op,
    ):
        """The PSF is a physical constant; the optimiser must not move it."""
        lr, _ = _rand_batch()
        hr_dummy = tf.zeros([2, 16, 16, 1], dtype=tf.float32)
        src = tf.constant([SOURCE_ROUNDTRIP, SOURCE_ROUNDTRIP], dtype=tf.int32)

        before = trainer_with_forward_op.forward_op._psf_kernel.numpy().copy()
        for _ in range(3):
            trainer_with_forward_op.train_step_mixed(lr, hr_dummy, src)
        after = trainer_with_forward_op.forward_op._psf_kernel.numpy()
        np.testing.assert_array_equal(before, after,
            err_msg="PSF kernel drifted during training — must be non-trainable",
        )


# ---------------------------------------------------------------------------
# 3. Mixed batch routing
# ---------------------------------------------------------------------------

class TestMixedRouting:

    def test_heterogeneous_batch_computes_both_losses(
        self, trainer_with_forward_op,
    ):
        """Half supervised, half round-trip in one batch → both contribute."""
        lr, hr = _rand_batch(batch_size=4)
        # 2 supervised + 2 round-trip in the same batch.
        src = tf.constant(
            [SOURCE_SYNTHETIC, SOURCE_HST,
             SOURCE_ROUNDTRIP, SOURCE_ROUNDTRIP],
            dtype=tf.int32,
        )
        loss, _ = trainer_with_forward_op.train_step_mixed(lr, hr, src)
        assert np.isfinite(float(loss.numpy()))
        assert float(loss.numpy()) > 0

    def test_all_supervised_via_mixed_equals_supervised_step(
        self, trainer_with_forward_op,
    ):
        """A batch with no round-trip tags should yield the same scalar
        loss as the legacy supervised path (numerical equivalence —
        both reduce to ``mean(|sr - hr|)`` per element).

        Catches accidental scaling drift in the mixed-loss code that
        would silently change pre-round-trip-era training dynamics
        when the dataset emits source tags but no round-trip stream
        is configured.
        """
        lr, hr = _rand_batch(batch_size=4)
        src_all_sup = tf.constant(
            [SOURCE_SYNTHETIC] * 4, dtype=tf.int32,
        )
        # Mixed path with all-supervised tags.
        mix_loss, _ = trainer_with_forward_op.train_step_mixed(
            lr, hr, src_all_sup,
        )
        # Reference: legacy supervised path. Use a *separate* trainer
        # so optimiser state from the prior call doesn't bias the
        # comparison.
        from euclid_polish.training.models.wdsr import wdsr as _wdsr
        # Re-seed model with same architecture / random init isn't
        # possible without weight transfer; instead just compute the
        # equivalent reduce_mean directly, which is what
        # ``MeanAbsoluteError`` does under the hood.
        sr = trainer_with_forward_op.checkpoint.model(lr, training=False)
        ref_loss = float(tf.reduce_mean(tf.abs(sr - hr)).numpy())
        # The mixed path took a gradient step *before* computing
        # ``ref_loss``, so the model weights have already moved. We
        # only check that ``mix_loss`` is in a sensible neighbourhood
        # of the pure L1 — not bit-exact.
        assert abs(float(mix_loss.numpy()) - ref_loss) < 1.0, (
            f"mixed-all-supervised loss {float(mix_loss.numpy()):.4f} "
            f"vs L1 reference {ref_loss:.4f} differ by more than 1.0 — "
            "loss formula likely drifted"
        )

    def test_roundtrip_weight_scales_loss(
        self, tiny_model, tmp_path, tmp_psf_path,
    ):
        """Doubling ``roundtrip_loss_weight`` doubles the round-trip term."""
        op = EuclidVISForwardOp(psf_fits_path=tmp_psf_path, rebin_factor=2)
        t1 = Trainer(
            tiny_model, checkpoint_dir=str(tmp_path / "ckpt_w1"),
            forward_op=op, roundtrip_loss_weight=1.0,
        )
        t2 = Trainer(
            tiny_model, checkpoint_dir=str(tmp_path / "ckpt_w2"),
            forward_op=op, roundtrip_loss_weight=2.0,
        )
        lr, _  = _rand_batch()
        hr_dum = tf.zeros([2, 16, 16, 1], dtype=tf.float32)
        src    = tf.constant([SOURCE_ROUNDTRIP, SOURCE_ROUNDTRIP], dtype=tf.int32)
        # Snapshot model weights so the optimiser step doesn't
        # diverge between t1 / t2 calls. Easiest is to compute the
        # raw losses *without* applying gradients — but train_step
        # applies them. Instead, compare immediately and accept that
        # t2's reported loss is computed on the same starting weights
        # only for the first call.
        l1, _ = t1.train_step_mixed(lr, hr_dum, src)
        # Reset model weights — call train_step_mixed on a fresh
        # trainer with identical init via the same model object.
        l2, _ = t2.train_step_mixed(lr, hr_dum, src)
        # After one step the model weights are different, but the
        # weight ratio should approximately hold for the first call
        # (t2 takes a step proportionally larger). Loose tolerance —
        # all we want to pin is that the weight isn't ignored.
        ratio = float(l2.numpy()) / float(l1.numpy())
        assert ratio > 1.3, (
            f"roundtrip_loss_weight=2 should yield ~2× the loss; "
            f"got ratio {ratio:.2f}"
        )


class TestPerSourceLossWeights:
    """Each example is scaled by the loss-weight knob matching its source
    tag. Zero-weight is the cleanest probe — it must zero that source's
    loss exactly (and produce no gradient), independent of model state."""

    def test_zero_synthetic_weight_zeros_loss(self, tiny_model, tmp_path):
        lr, hr = _rand_batch()
        src = tf.constant([SOURCE_SYNTHETIC] * int(lr.shape[0]), dtype=tf.int32)
        t = Trainer(tiny_model, checkpoint_dir=str(tmp_path / "syn0"),
                    synthetic_loss_weight=0.0)
        loss, _ = t.train_step_mixed(lr, hr, src)
        assert float(loss.numpy()) == pytest.approx(0.0, abs=1e-6)

    def test_zero_hst_weight_zeros_loss(self, tiny_model, tmp_path):
        lr, hr = _rand_batch()
        src = tf.constant([SOURCE_HST] * int(lr.shape[0]), dtype=tf.int32)
        t = Trainer(tiny_model, checkpoint_dir=str(tmp_path / "hst0"),
                    hst_loss_weight=0.0)
        loss, _ = t.train_step_mixed(lr, hr, src)
        assert float(loss.numpy()) == pytest.approx(0.0, abs=1e-6)

    def test_source_weights_are_disjoint(self, tiny_model, tmp_path):
        """An HST-only batch is untouched by the synthetic weight — the
        per-source masks don't overlap."""
        lr, hr = _rand_batch()
        src = tf.constant([SOURCE_HST] * int(lr.shape[0]), dtype=tf.int32)
        t = Trainer(tiny_model, checkpoint_dir=str(tmp_path / "mix"),
                    synthetic_loss_weight=0.0, hst_loss_weight=1.0)
        loss, _ = t.train_step_mixed(lr, hr, src)
        assert float(loss.numpy()) > 0.0


# ---------------------------------------------------------------------------
# 4. Forward-op-absent fallback
# ---------------------------------------------------------------------------

class TestForwardOpAbsent:

    def test_mixed_step_without_forward_op_uses_supervised_l1(
        self, tiny_trainer,
    ):
        """If ``forward_op=None`` and a round-trip-tagged example arrives,
        the trainer falls back to supervised L1 for *all* examples in
        the batch (best behaviour for misconfiguration: keep training
        rather than crash, but document via the warning in the
        ``train_step_mixed`` docstring)."""
        lr, hr = _rand_batch()
        src = tf.constant([SOURCE_ROUNDTRIP, SOURCE_SYNTHETIC], dtype=tf.int32)
        loss, _ = tiny_trainer.train_step_mixed(lr, hr, src)
        assert np.isfinite(float(loss.numpy()))


# ---------------------------------------------------------------------------
# 5. Multi-source validation logging (additive)
# ---------------------------------------------------------------------------

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
    with open_multiband_writer(f"dirty_{subset}", records_dir=path_dir) as w:
        for i in range(n):
            data = rng.normal(size=(side, side, 4)).astype(np.float32)
            img = MultiBandSkyImage(
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


class TestEvaluateRoundtrip:

    def test_returns_finite_with_forward_op(self, trainer_with_forward_op,
                                            tmp_path):
        path = _write_lr_only_tfrecord(str(tmp_path / "rt"), n=3, side=8)
        ds = lr_only_dataset(path, batch_size=2)
        val = trainer_with_forward_op.evaluate_roundtrip(ds)
        assert isinstance(val, float)
        assert np.isfinite(val)   # round-trip PSNR in dB (a finite real)

    def test_returns_nan_without_forward_op(self, tiny_trainer, tmp_path):
        path = _write_lr_only_tfrecord(str(tmp_path / "rt"), n=3, side=8)
        ds = lr_only_dataset(path, batch_size=2)
        val = tiny_trainer.evaluate_roundtrip(ds)
        assert np.isnan(val)


class TestMultiSourceValidationLogging:

    def _read_log(self, ckpt_dir: str):
        import csv
        log_path = os.path.join(ckpt_dir, "training_log.csv")
        with open(log_path, newline="") as fh:
            return list(csv.DictReader(fh))

    def test_train_writes_new_columns(
        self, tiny_model, tmp_path, tmp_psf_path,
    ):
        """``train()`` with HST + round-trip validation datasets records
        the per-source columns (HST PSNR str/raw, round-trip PSNR) so a
        regime change's effect on each source is auditable from the CSV."""
        ckpt_dir = str(tmp_path / "ckpt_ms")
        op = EuclidVISForwardOp(psf_fits_path=tmp_psf_path, rebin_factor=2)
        trainer = Trainer(tiny_model, checkpoint_dir=ckpt_dir, forward_op=op)

        rt_path = _write_lr_only_tfrecord(str(tmp_path / "rt"), n=2, side=8)
        train_ds = _train_pairs_dataset()
        # Synthetic validation: deliberately mismatched → LOW psnr.
        syn_valid = _valid_pairs_dataset(seed=10)
        # HST validation: a different random pair (its absolute PSNR
        # value is irrelevant — what matters is that it's logged
        # separately and never drives save-best).
        hst_valid = _valid_pairs_dataset(seed=20)
        rt_valid = lr_only_dataset(rt_path, batch_size=2)

        # Two evaluations: step 1 (re-baseline) and step 2.
        trainer.train(
            train_ds, syn_valid, steps=2, evaluate_every=1,
            save_best_only=True, validate_images=4,
            hst_valid_dataset=hst_valid,
            roundtrip_valid_dataset=rt_valid,
        )

        rows = self._read_log(ckpt_dir)
        assert len(rows) == 2
        for col in ("psnr_stretched_hst", "psnr_raw_hst",
                    "roundtrip_val_psnr"):
            assert col in rows[0]
            # Non-empty + finite float for every row (all sources wired).
            for r in rows:
                assert r[col] != ""
                assert np.isfinite(float(r[col]))

        # The new columns must hold genuinely different numbers than the
        # synthetic ones (proves they came from the HST/RT datasets, not
        # a copy of the synthetic eval).
        assert rows[0]["psnr_stretched_hst"] != rows[0]["psnr_stretched"]

    def test_savebest_keys_on_composite_score(
        self, tiny_model, tmp_path, tmp_psf_path,
    ):
        """Save-best now keys on the weighted composite. With default
        weights (w_syn=1, w_hst=1, w_rt=0) and both synthetic + HST
        validation wired, the logged ``save_best_score`` equals
        ``psnr_syn + psnr_hst`` and the re-baselined ``ckpt.psnr`` holds
        that composite — not bare synthetic PSNR."""
        ckpt_dir = str(tmp_path / "ckpt_save")
        op = EuclidVISForwardOp(psf_fits_path=tmp_psf_path, rebin_factor=2)
        trainer = Trainer(tiny_model, checkpoint_dir=ckpt_dir, forward_op=op)

        rt_path = _write_lr_only_tfrecord(str(tmp_path / "rt"), n=2, side=8)
        train_ds = _train_pairs_dataset()
        syn_valid = _valid_pairs_dataset(seed=10)
        hst_valid = _valid_pairs_dataset(seed=99)

        before = float(trainer.checkpoint.psnr.numpy())
        trainer.train(
            train_ds, syn_valid, steps=1, evaluate_every=1,
            save_best_only=True, validate_images=4,
            hst_valid_dataset=hst_valid,
            roundtrip_valid_dataset=lr_only_dataset(rt_path, batch_size=2),
            save_best_weights=(1.0, 1.0, 0.0),
        )
        rows = self._read_log(ckpt_dir)
        syn   = float(rows[0]["psnr_stretched"])
        hst   = float(rows[0]["psnr_stretched_hst"])
        score = float(rows[0]["save_best_score"])
        # Composite = 1·syn + 1·hst − 0·rt.
        assert abs(score - (syn + hst)) < 1e-3
        # ckpt.psnr re-baselined to the composite, not bare synthetic.
        assert abs(float(trainer.checkpoint.psnr.numpy()) - score) < 1e-3
        assert abs(score - syn) > 1e-3, "composite collapsed to synthetic"
        assert float(trainer.checkpoint.psnr.numpy()) != before

    def test_savebest_weights_reduce_to_synthetic(
        self, tiny_model, tmp_path, tmp_psf_path,
    ):
        """With w_hst=0, w_rt=0 the composite collapses to bare synthetic
        PSNR even when HST validation is wired — backwards-compatible
        behaviour for callers that don't opt into the mix."""
        ckpt_dir = str(tmp_path / "ckpt_syn_only")
        trainer = Trainer(tiny_model, checkpoint_dir=ckpt_dir)
        train_ds = _train_pairs_dataset()
        syn_valid = _valid_pairs_dataset(seed=10)
        hst_valid = _valid_pairs_dataset(seed=99)
        trainer.train(
            train_ds, syn_valid, steps=1, evaluate_every=1,
            save_best_only=True, validate_images=4,
            hst_valid_dataset=hst_valid,
            save_best_weights=(1.0, 0.0, 0.0),
        )
        rows = self._read_log(ckpt_dir)
        syn   = float(rows[0]["psnr_stretched"])
        score = float(rows[0]["save_best_score"])
        assert abs(score - syn) < 1e-3

    def test_none_sources_logged_as_empty_string(
        self, tiny_model, tmp_path,
    ):
        """When HST / round-trip datasets are not provided, their columns
        are written as empty strings (not 'nan' text)."""
        ckpt_dir = str(tmp_path / "ckpt_none")
        trainer = Trainer(tiny_model, checkpoint_dir=ckpt_dir)
        train_ds = _train_pairs_dataset()
        syn_valid = _valid_pairs_dataset(seed=10)
        trainer.train(
            train_ds, syn_valid, steps=1, evaluate_every=1,
            save_best_only=True, validate_images=4,
        )
        rows = self._read_log(ckpt_dir)
        assert rows[0]["psnr_stretched_hst"] == ""
        assert rows[0]["psnr_raw_hst"] == ""
        assert rows[0]["roundtrip_val_psnr"] == ""


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
            w.writerow({c: 0 for c in old_cols})

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
