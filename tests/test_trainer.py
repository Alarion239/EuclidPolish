"""Tests for the SR=sky trainer (``train_step_sky``).

The model estimates one quantity — the deconvolved sky ``SR`` — and each
lane supervises it on a **fixed contiguous-block batch**
``[n_syn | n_hst | n_anchor]`` (no per-example source tags, no
``tf.where`` branching). These tests pin:

  1. **Backward compatibility**: the supervised ``train_step(lr, hr)``
     (``lane_counts=None`` path) still drives gradients as before — the
     pure-synthetic callers (``run_pipeline.py``, ``cli/main.py``, the
     web inference helpers) are unaffected.
  2. **Star-anchor lane**: an all-anchor layout trains on the masked
     ``|SR - delta_target|`` at the star pixel — operator-free (no PSF).
  3. **HST lane**: an all-HST layout trains on
     ``|asinh(H ⊛ SR) - HST_image|`` through the (frozen) HST forward op.
  4. **Mixed layout**: a single ``[syn | hst | anchor]`` batch sums all
     three lane losses in one tape pass via static slices.
  5. **Per-lane weights** and the **missing-op guard** (HST only).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import tensorflow as tf
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.image.tfio import open_writer, tfrecord_path
from euclid_polish.image import Image
from euclid_polish.training.augmentation import (
    asinh_stretch_hr, asinh_stretch_lr, lr_only_dataset,
)
from euclid_polish.training.forward_op import HSTForwardOp
from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.training.trainer import (
    Trainer, TRAINING_LOG_COLUMNS, _is_grad_spike,
    GRAD_SPIKE_SKIP_NORM, GRAD_SPIKE_SKIP_WARMUP_STEPS,
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
def trainer_with_anchor(tiny_model, tmp_path):
    """Trainer for the operator-free star-anchor lane (no forward op)."""
    return Trainer(
        tiny_model, checkpoint_dir=str(tmp_path / "ckpt_anchor"),
        star_anchor_loss_weight=1.0,
    )


@pytest.fixture
def trainer_with_both_ops(tiny_model, tmp_path, tmp_psf_path):
    """Trainer wired so a full ``[syn | hst | anchor]`` layout can run: the
    HST lane needs the HST forward op; the anchor lane is operator-free.
    The HST op reuses the tiny Gaussian PSF (rebin_factor forced to 1)."""
    hst_op = HSTForwardOp(psf_fits_path=tmp_psf_path)
    return Trainer(
        tiny_model, checkpoint_dir=str(tmp_path / "ckpt_both"),
        hst_forward_op=hst_op, hst_loss_weight=1.0, star_anchor_loss_weight=1.0,
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


def _anchor_batch(batch_size: int = 2, lr_side: int = 8, hr_side: int = 16,
                  seed: int = 0):
    """``(lr, hr)`` where ``hr`` is a sparse delta-target: zero except one
    positive 'star' pixel per image (what the masked anchor loss reads)."""
    rng = np.random.default_rng(seed)
    lr = tf.constant(
        rng.normal(size=(batch_size, lr_side, lr_side, 4)).astype(np.float32),
    )
    hr = np.zeros((batch_size, hr_side, hr_side, 1), dtype=np.float32)
    hr[:, hr_side // 2, hr_side // 2, 0] = 5.0
    return lr, tf.constant(hr)


def _anchor_valid_dataset(n: int = 2, lr_side: int = 8, seed: int = 1,
                          batch_size: int = 1):
    """Batched anchor ``(lr, hr)`` validation dataset for evaluate_anchor."""
    rng = np.random.default_rng(seed)
    hr_side = lr_side * 2
    lr = rng.normal(size=(n, lr_side, lr_side, 4)).astype(np.float32)
    hr = np.zeros((n, hr_side, hr_side, 1), dtype=np.float32)
    hr[:, hr_side // 2, hr_side // 2, 0] = 5.0
    return tf.data.Dataset.from_tensor_slices((lr, hr)).batch(batch_size)


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
# 2. Star-anchor lane (operator-free)
# ---------------------------------------------------------------------------

class TestAnchorLane:

    def test_pure_anchor_batch_trains(self, trainer_with_anchor):
        """An all-anchor layout ``(0, 0, B)`` runs through ``train_step_sky``
        (no forward op) and produces finite, positive loss + gradients. ``hr``
        is the sparse delta-target; the loss is masked to the star pixel."""
        lr, hr = _anchor_batch()
        loss, gnorm, _ = trainer_with_anchor.train_step_sky(lr, hr, 0, 0, 2)
        assert np.isfinite(float(loss.numpy()))
        assert np.isfinite(float(gnorm.numpy()))
        assert float(loss.numpy()) > 0, (
            "anchor loss on a random SR vs a 5.0 delta should be positive"
        )

    def test_anchor_gradients_reach_model(self, trainer_with_anchor):
        """Gradients of the masked anchor loss w.r.t. M's weights must be
        non-zero — otherwise the anchor lane has no training effect."""
        lr, hr = _anchor_batch()
        model = trainer_with_anchor.checkpoint.model
        with tf.GradientTape() as tape:
            sr = model(lr, training=True)
            mask = tf.cast(hr > 0, sr.dtype)
            num = tf.reduce_sum(mask * tf.abs(sr - hr), axis=[1, 2, 3])
            den = tf.reduce_sum(mask, axis=[1, 2, 3]) + 1e-6
            loss = tf.reduce_mean(num / den)
        grads = tape.gradient(loss, model.trainable_variables)
        max_g = float(max(
            tf.reduce_max(tf.abs(g)).numpy() for g in grads if g is not None
        ))
        assert max_g > 0, "masked anchor loss produced zero gradient"

    def test_anchor_loss_only_sees_star_pixels(self, trainer_with_anchor):
        """The masked loss must ignore non-star pixels: changing ``hr`` ONLY
        off the star pixel leaves the anchor loss unchanged."""
        lr, hr = _anchor_batch(seed=7)
        hr2 = hr.numpy().copy()
        hr2[:, 0, 0, 0] = -1234.0                # non-star pixel — mask>0 excludes it
        model = trainer_with_anchor.checkpoint.model

        def masked(target):
            sr = model(lr, training=False)
            m = tf.cast(target > 0, sr.dtype)
            num = tf.reduce_sum(m * tf.abs(sr - target), axis=[1, 2, 3])
            den = tf.reduce_sum(m, axis=[1, 2, 3]) + 1e-6
            return float(tf.reduce_mean(num / den).numpy())

        assert masked(hr) == pytest.approx(masked(tf.constant(hr2)), rel=1e-6)

    def test_returns_per_lane_losses(self, trainer_with_both_ops):
        """train_step_sky returns (loss, gnorm, (syn, hst, anchor)); an off
        lane is exactly 0.0 and the active lane is finite > 0 (these feed the
        per-lane loss columns of the metrics CSV)."""
        lr, hr = _anchor_batch(batch_size=2)
        loss, gnorm, lane = trainer_with_both_ops.train_step_sky(lr, hr, 0, 0, 2)
        syn, hst, anchor = (float(x.numpy()) for x in lane)
        assert syn == 0.0 and hst == 0.0          # not active in an all-anchor layout
        assert np.isfinite(anchor) and anchor > 0


# ---------------------------------------------------------------------------
# 3. HST lane
# ---------------------------------------------------------------------------

class TestHstLane:

    def test_pure_hst_batch_trains_through_forward_op(
        self, trainer_with_both_ops,
    ):
        """An all-HST layout ``(0, B, 0)`` trains on ``|H⊛SR - HST|``,
        finite + positive, with the HST op in the gradient path."""
        lr, hr = _rand_batch(batch_size=2)
        loss, gnorm, _ = trainer_with_both_ops.train_step_sky(lr, hr, 0, 2, 0)
        assert np.isfinite(float(loss.numpy()))
        assert np.isfinite(float(gnorm.numpy()))
        assert float(loss.numpy()) > 0

    def test_hst_psf_remains_non_trainable(self, trainer_with_both_ops):
        """The HST PSF is a physical constant — the optimiser must not move it."""
        lr, hr = _rand_batch(batch_size=2)
        before = trainer_with_both_ops.hst_forward_op._psf_kernel.numpy().copy()
        for _ in range(3):
            trainer_with_both_ops.train_step_sky(lr, hr, 0, 2, 0)
        after = trainer_with_both_ops.hst_forward_op._psf_kernel.numpy()
        np.testing.assert_array_equal(before, after,
            err_msg="HST PSF kernel drifted — must be non-trainable")


# ---------------------------------------------------------------------------
# 4. Mixed fixed-layout batch
# ---------------------------------------------------------------------------

class TestMixedLayout:

    def test_full_layout_sums_all_three_lanes(self, trainer_with_both_ops):
        """A single ``[syn | hst | anchor]`` batch (1, 1, 2) runs all three
        lanes in one tape pass and yields finite, positive loss. ``hr`` has a
        positive star pixel per image, so the anchor lane's mask is non-empty."""
        lr, hr = _anchor_batch(batch_size=4)
        loss, _, _ = trainer_with_both_ops.train_step_sky(lr, hr, 1, 1, 2)
        assert np.isfinite(float(loss.numpy()))
        assert float(loss.numpy()) > 0

    def test_all_synthetic_layout_equals_supervised(
        self, trainer_with_both_ops,
    ):
        """An all-synthetic layout ``(B, 0, 0)`` reduces to the supervised
        ``mean(|sr - hr|)`` — the synthetic lane applies no forward op."""
        lr, hr = _rand_batch(batch_size=4)
        sr = trainer_with_both_ops.checkpoint.model(lr, training=False)
        ref_loss = float(tf.reduce_mean(tf.abs(sr - hr)).numpy())
        loss, _, _ = trainer_with_both_ops.train_step_sky(lr, hr, 4, 0, 0)
        # The step takes a gradient update before we read ref_loss off the
        # moved weights, so only a neighbourhood check is meaningful.
        assert abs(float(loss.numpy()) - ref_loss) < 1.0, (
            f"all-synthetic layout loss {float(loss.numpy()):.4f} vs L1 "
            f"reference {ref_loss:.4f} differ by >1.0 — formula drifted"
        )

    def test_anchor_weight_scales_loss(self, tiny_model, tmp_path):
        """Doubling ``star_anchor_loss_weight`` ~doubles an all-anchor loss."""
        # nonneg_sr_weight=0 so the loss is purely the anchor term and the
        # ×2 weight ratio isn't diluted by the additive penalty.
        t1 = Trainer(
            tiny_model, checkpoint_dir=str(tmp_path / "ckpt_w1"),
            star_anchor_loss_weight=1.0, nonneg_sr_weight=0.0,
        )
        t2 = Trainer(
            tiny_model, checkpoint_dir=str(tmp_path / "ckpt_w2"),
            star_anchor_loss_weight=2.0, nonneg_sr_weight=0.0,
        )
        lr, hr = _anchor_batch()
        l1, _, _ = t1.train_step_sky(lr, hr, 0, 0, 2)
        l2, _, _ = t2.train_step_sky(lr, hr, 0, 0, 2)
        ratio = float(l2.numpy()) / float(l1.numpy())
        assert ratio > 1.3, (
            f"star_anchor_loss_weight=2 should yield ~2× the loss; "
            f"got ratio {ratio:.2f}"
        )


class TestPerLaneLossWeights:
    """Each lane is scaled by its loss-weight knob. Zero-weight is the
    cleanest probe — it must zero that lane's loss exactly."""

    def test_zero_synthetic_weight_zeros_loss(self, tiny_model, tmp_path):
        lr, hr = _rand_batch()
        # nonneg_sr_weight=0 so the loss is purely the (zeroed) lane term.
        t = Trainer(tiny_model, checkpoint_dir=str(tmp_path / "syn0"),
                    synthetic_loss_weight=0.0, nonneg_sr_weight=0.0)
        loss, _, _ = t.train_step_sky(lr, hr, int(lr.shape[0]), 0, 0)
        assert float(loss.numpy()) == pytest.approx(0.0, abs=1e-6)

    def test_zero_hst_weight_zeros_loss(
        self, tiny_model, tmp_path, tmp_psf_path,
    ):
        lr, hr = _rand_batch()
        hst_op = HSTForwardOp(psf_fits_path=tmp_psf_path)
        t = Trainer(tiny_model, checkpoint_dir=str(tmp_path / "hst0"),
                    hst_forward_op=hst_op, hst_loss_weight=0.0,
                    nonneg_sr_weight=0.0)
        loss, _, _ = t.train_step_sky(lr, hr, 0, int(lr.shape[0]), 0)
        assert float(loss.numpy()) == pytest.approx(0.0, abs=1e-6)

    def test_hst_lane_nonzero_when_weighted(
        self, tiny_model, tmp_path, tmp_psf_path,
    ):
        """An all-HST layout with a non-zero HST weight produces loss > 0."""
        lr, hr = _rand_batch()
        hst_op = HSTForwardOp(psf_fits_path=tmp_psf_path)
        t = Trainer(tiny_model, checkpoint_dir=str(tmp_path / "hst1"),
                    hst_forward_op=hst_op,
                    synthetic_loss_weight=0.0, hst_loss_weight=1.0)
        loss, _, _ = t.train_step_sky(lr, hr, 0, int(lr.shape[0]), 0)
        assert float(loss.numpy()) > 0.0


# ---------------------------------------------------------------------------
# 5. Missing-op guards
# ---------------------------------------------------------------------------

class TestForwardOpGuards:

    def test_hst_lane_without_hst_op_raises(self, tiny_trainer):
        """``n_hst > 0`` without an HST forward op is a config error."""
        lr, hr = _rand_batch()
        with pytest.raises(ValueError, match="hst_forward_op"):
            tiny_trainer.train_step_sky(lr, hr, 0, int(lr.shape[0]), 0)

    def test_anchor_lane_needs_no_op(self, tiny_trainer):
        """The anchor lane is operator-free — ``n_anchor > 0`` runs on a bare
        trainer (no forward op) without raising."""
        lr, hr = _anchor_batch()
        loss, _, _ = tiny_trainer.train_step_sky(lr, hr, 0, 0, int(lr.shape[0]))
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


class TestEvaluateAnchor:

    def test_returns_finite_masked_psnr(self, tiny_trainer):
        """evaluate_anchor returns a finite dB PSNR masked to star pixels,
        within the cap."""
        from euclid_polish.training.trainer import ANCHOR_PSNR_MAX_DB
        ds = _anchor_valid_dataset(n=3, batch_size=2)
        val, mae = tiny_trainer.evaluate_anchor(ds)
        assert isinstance(val, float) and isinstance(mae, float)
        assert np.isfinite(val) and np.isfinite(mae) and mae >= 0.0
        assert val <= ANCHOR_PSNR_MAX_DB + 1e-4

    def test_exact_match_is_capped_not_inf(self, tiny_trainer):
        """A star pixel the model matches exactly drives MSE→0; the PSNR must
        be the finite cap, not +inf (single-pixel target ⇒ unbounded raw)."""
        from euclid_polish.training.trainer import ANCHOR_PSNR_MAX_DB
        rng = np.random.default_rng(3)
        lr = rng.normal(size=(3, 8, 8, 4)).astype(np.float32)
        sr = tiny_trainer.checkpoint.model(tf.constant(lr), training=False).numpy()
        hr = np.zeros((3, 16, 16, 1), dtype=np.float32)
        for b in range(sr.shape[0]):
            flat = sr[b, ..., 0]
            r, c = np.unravel_index(int(np.argmax(flat)), flat.shape)
            # Set the masked star pixel to the model's exact value there (>0
            # so the mask keeps it) → that pixel's MSE is 0.
            hr[b, r, c, 0] = max(float(flat[r, c]), 1e-3)
        ds = tf.data.Dataset.from_tensor_slices((lr, hr)).batch(1)
        val, _mae = tiny_trainer.evaluate_anchor(ds)
        assert np.isfinite(val)
        assert val <= ANCHOR_PSNR_MAX_DB + 1e-4

    def test_returns_nan_when_no_stars(self, tiny_trainer):
        """A dataset whose ``hr`` is all-zero (no star pixel) → nan."""
        lr = np.zeros((2, 8, 8, 4), dtype=np.float32)
        hr = np.zeros((2, 16, 16, 1), dtype=np.float32)
        ds = tf.data.Dataset.from_tensor_slices((lr, hr)).batch(1)
        val, mae = tiny_trainer.evaluate_anchor(ds)
        assert np.isnan(val) and np.isnan(mae)


class TestMultiSourceValidationLogging:

    def _read_log(self, ckpt_dir: str):
        import csv
        log_path = os.path.join(ckpt_dir, "training_log.csv")
        with open(log_path, newline="") as fh:
            return list(csv.DictReader(fh))

    def test_train_writes_new_columns(
        self, tiny_model, tmp_path, tmp_psf_path,
    ):
        """``train()`` with HST + star-anchor validation datasets records
        the per-source columns (HST PSNR str/raw, anchor PSNR) so a regime
        change's effect on each source is auditable from the CSV."""
        ckpt_dir = str(tmp_path / "ckpt_ms")
        hst_op = HSTForwardOp(psf_fits_path=tmp_psf_path)
        trainer = Trainer(tiny_model, checkpoint_dir=ckpt_dir,
                          hst_forward_op=hst_op)

        train_ds = _train_pairs_dataset()
        # Synthetic validation: deliberately mismatched → LOW psnr.
        syn_valid = _valid_pairs_dataset(seed=10)
        hst_valid = _valid_pairs_dataset(seed=20)
        anchor_valid = _anchor_valid_dataset(n=2, batch_size=2)

        # Two evaluations: step 1 (re-baseline) and step 2.
        trainer.train(
            train_ds, syn_valid, steps=2, evaluate_every=1,
            save_best_only=True, validate_images=4,
            hst_valid_dataset=hst_valid,
            anchor_valid_dataset=anchor_valid,
        )

        rows = self._read_log(ckpt_dir)
        assert len(rows) == 2
        for col in ("psnr_stretched_hst", "psnr_raw_hst",
                    "anchor_val_psnr"):
            assert col in rows[0]
            # Non-empty + finite float for every row (all sources wired).
            for r in rows:
                assert r[col] != ""
                assert np.isfinite(float(r[col]))

        # Per-lane training losses: this run trains pure-supervised (no
        # lane_counts), so the synthetic loss is logged and HST/anchor are
        # blank — the whole-history loss columns exist and behave.
        for r in rows:
            assert r["loss_syn"] != "" and np.isfinite(float(r["loss_syn"]))
            assert r["loss_hst"] == ""
            assert r["loss_anchor"] == ""

        # The new columns must hold genuinely different numbers than the
        # synthetic ones (proves they came from the HST/anchor datasets, not
        # a copy of the synthetic eval).
        assert rows[0]["psnr_stretched_hst"] != rows[0]["psnr_stretched"]

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

    def test_savebest_keys_on_composite_score(
        self, tiny_model, tmp_path, tmp_psf_path,
    ):
        """Save-best now keys on the weighted composite. With default
        weights (w_syn=1, w_hst=1, w_anchor=0) and both synthetic + HST
        validation wired, the logged ``save_best_score`` equals
        ``psnr_syn + psnr_hst`` and the re-baselined ``ckpt.psnr`` holds
        that composite — not bare synthetic PSNR."""
        ckpt_dir = str(tmp_path / "ckpt_save")
        hst_op = HSTForwardOp(psf_fits_path=tmp_psf_path)
        trainer = Trainer(tiny_model, checkpoint_dir=ckpt_dir,
                          hst_forward_op=hst_op)

        train_ds = _train_pairs_dataset()
        syn_valid = _valid_pairs_dataset(seed=10)
        hst_valid = _valid_pairs_dataset(seed=99)

        before = float(trainer.checkpoint.psnr.numpy())
        trainer.train(
            train_ds, syn_valid, steps=1, evaluate_every=1,
            save_best_only=True, validate_images=4,
            hst_valid_dataset=hst_valid,
            anchor_valid_dataset=_anchor_valid_dataset(n=2, batch_size=2),
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
        hst_op = HSTForwardOp(psf_fits_path=tmp_psf_path)
        trainer = Trainer(tiny_model, checkpoint_dir=ckpt_dir,
                          hst_forward_op=hst_op)
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
        """When HST / star-anchor datasets are not provided, their columns
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
        assert rows[0]["anchor_val_psnr"] == ""


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
        baseline_score = float(base[0]["save_best_score"])
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
        assert float(t2.checkpoint.psnr.numpy()) >= float(base["save_best_score"]) - 1e-3


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

    def test_train_step_sky_runs_with_penalty(self, trainer_with_both_ops):
        """The penalty term integrates into the composite SR=sky step."""
        lr, hr = _rand_batch(batch_size=4)
        loss, gnorm, _ = trainer_with_both_ops.train_step_sky(lr, hr, 1, 1, 2)
        assert np.isfinite(float(loss.numpy()))
        assert np.isfinite(float(gnorm.numpy()))


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
        # Joint metric still present and used for the save-best score.
        assert np.isfinite(float(rows[-1]["psnr_stretched"]))
        assert np.isfinite(float(rows[-1]["save_best_score"]))

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
