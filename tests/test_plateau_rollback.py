"""Plateau guard v2: rollback-to-best on degenerate plateaus + resume tracks."""
from __future__ import annotations

import pytest

from euclid_polish.training.trainer import Trainer, _plateau_wants_rollback


# ── pure regime decision ────────────────────────────────────────────────────

def test_rollback_when_score_sits_below_best():
    # member_09 case: best 43.972, stuck at 43.53 → gap 0.44 ≥ 0.2 → rollback
    assert _plateau_wants_rollback(43.53, 43.972, min_gap=0.2,
                                   has_best_ckpt=True, save_best_only=True)


def test_no_rollback_on_converged_plateau():
    # score ≈ best → this is convergence; reduce LR in place
    assert not _plateau_wants_rollback(43.95, 43.972, min_gap=0.2,
                                       has_best_ckpt=True, save_best_only=True)


def test_no_rollback_without_checkpoint_or_save_best():
    assert not _plateau_wants_rollback(43.0, 44.0, min_gap=0.2,
                                       has_best_ckpt=False, save_best_only=True)
    assert not _plateau_wants_rollback(43.0, 44.0, min_gap=0.2,
                                       has_best_ckpt=True, save_best_only=False)


def test_no_rollback_on_nonfinite_scores():
    assert not _plateau_wants_rollback(float("nan"), 44.0, min_gap=0.2,
                                       has_best_ckpt=True, save_best_only=True)
    assert not _plateau_wants_rollback(43.0, float("-inf"), min_gap=0.2,
                                       has_best_ckpt=True, save_best_only=True)


# ── Trainer.restore resume_track ────────────────────────────────────────────

@pytest.fixture(scope="module")
def _tiny_model():
    from euclid_polish.training.models.wdsr import wdsr
    return wdsr(scale=2, num_res_blocks=1, nchan_in=4, nchan_out=4)


def _two_track_dir(model, d: str) -> None:
    """A member dir where loss_best/ has a HIGHER step than the psnr root —
    the post-degenerate-run layout that used to hijack resume."""
    t = Trainer(model, checkpoint_dir=d, plateau_lr_enabled=False)
    t.checkpoint.step.assign(5)
    t.checkpoint_manager.save()                # psnr root @ step 5
    t.checkpoint.step.assign(9)
    t.loss_checkpoint_manager.save()           # loss_best @ step 9


def test_restore_latest_picks_max_step_track(_tiny_model, tmp_path):
    d = str(tmp_path / "m")
    _two_track_dir(_tiny_model, d)
    t = Trainer(_tiny_model, checkpoint_dir=d, plateau_lr_enabled=False)
    assert int(t.checkpoint.step.numpy()) == 9      # historical behavior


def test_restore_psnr_track_ignores_higher_step_loss_track(_tiny_model, tmp_path):
    d = str(tmp_path / "m")
    _two_track_dir(_tiny_model, d)
    t = Trainer(_tiny_model, checkpoint_dir=d, plateau_lr_enabled=False,
                resume_track="psnr")
    assert int(t.checkpoint.step.numpy()) == 5      # the track eval trusts
