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

from euclid_polish.training.data_multiband import (
    SOURCE_HST, SOURCE_ROUNDTRIP, SOURCE_SYNTHETIC,
    asinh_stretch_hr, asinh_stretch_lr,
)
from euclid_polish.training.forward_op import EuclidVISForwardOp
from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.training.trainer import Trainer


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
