"""Mixed-depth ensembles: num_res_blocks introspection + self-correcting loads."""
from __future__ import annotations

import os

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.model import Model
from euclid_polish.training.inference import (
    infer_checkpoint_num_res_blocks,
    load_model_from_checkpoint,
)
from euclid_polish.training.models.wdsr import wdsr


def _save_model_ckpt(model, d: str) -> None:
    ck = tf.train.Checkpoint(step=tf.Variable(1), model=model)
    tf.train.CheckpointManager(ck, d, max_to_keep=1).save()


@pytest.mark.parametrize("blocks", [1, 2])
def test_infer_num_res_blocks(tmp_path, blocks):
    m = wdsr(scale=2, num_res_blocks=blocks, nchan_in=4, nchan_out=4)
    d = str(tmp_path / f"b{blocks}")
    _save_model_ckpt(m, d)
    assert infer_checkpoint_num_res_blocks(d) == blocks


def test_infer_num_res_blocks_none_when_missing(tmp_path):
    assert infer_checkpoint_num_res_blocks(str(tmp_path / "nope")) is None


def test_load_model_corrects_requested_depth(tmp_path):
    deep = wdsr(scale=2, num_res_blocks=2, nchan_in=4, nchan_out=4)
    d = str(tmp_path / "deep")
    _save_model_ckpt(deep, d)
    # Request the WRONG depth — the checkpoint's depth must win, restoring
    # every layer instead of leaving a mismatched net half-initialized.
    loaded = load_model_from_checkpoint(d, scale=2, num_res_blocks=1)
    assert len(loaded.get_weights()) == len(deep.get_weights())
    for a, b in zip(loaded.get_weights(), deep.get_weights(), strict=True):
        assert np.array_equal(a, b)


def test_fork_preserves_channel_architecture(tmp_path):
    """A fork is built AS the source: channel counts come from the source
    checkpoint, not from Config — the whole architecture is preserved."""
    src_net = wdsr(scale=2, num_res_blocks=1, nchan_in=1, nchan_out=1)
    src_dir = str(tmp_path / "src")
    _save_model_ckpt(src_net, src_dir)

    dst = Model(str(tmp_path / "dst"), num_res_blocks=1, seed=2,
                init_weights_from=src_dir)
    assert dst._tf_model.input_shape[-1] == 1          # not Config's 4
    assert dst._tf_model.output_shape[-1] == 1
    for a, b in zip(src_net.get_weights(),
                    dst._tf_model.get_weights(), strict=True):
        assert np.array_equal(a, b)


def test_fork_inherits_source_depth(tmp_path):
    src_dir = str(tmp_path / "src")
    src = Model(src_dir, num_res_blocks=2, seed=1)
    _save_model_ckpt(src._tf_model, src_dir)
    # Target requests a DIFFERENT depth — the fork must build at the source's
    # depth (weights are copied verbatim) and say so via _num_res_blocks.
    dst = Model(str(tmp_path / "dst"), num_res_blocks=1, seed=2,
                init_weights_from=src_dir)
    assert dst._num_res_blocks == 2
    for a, b in zip(src._tf_model.get_weights(),
                    dst._tf_model.get_weights(), strict=True):
        assert np.array_equal(a, b)


def test_model_reports_actual_depth_on_load(tmp_path):
    d = str(tmp_path / "m")
    m2 = Model(d, num_res_blocks=2, seed=1)
    _save_model_ckpt(m2._tf_model, d)
    reloaded = Model(d, num_res_blocks=32)      # stale global default
    assert reloaded._num_res_blocks == 2


def test_prune_orphaned_checkpoints(tmp_path):
    """Files from a pre-resume run that no manifest references are deleted;
    manifest-tracked files survive. (Timeout-killed + resubmitted jobs left
    such orphans, doubling member dirs: 44.6 → 89.2 MB.)"""
    from euclid_polish.training.trainer import prune_orphaned_checkpoints

    d = str(tmp_path / "m")
    ck = tf.train.Checkpoint(step=tf.Variable(1))
    mgr = tf.train.CheckpointManager(ck, d, max_to_keep=2)
    mgr.save()
    mgr.save()                                   # manifest: ckpt-1, ckpt-2
    # fabricate orphans from an "earlier run"
    import shutil as _sh
    for stem in ("ckpt-35", "ckpt-37"):
        _sh.copyfile(f"{d}/ckpt-1.index", f"{d}/{stem}.index")
        _sh.copyfile(f"{d}/ckpt-1.data-00000-of-00001",
                     f"{d}/{stem}.data-00000-of-00001")
    n = prune_orphaned_checkpoints(d)
    assert n == 4                                # 2 stems × (index + data)
    names = set(os.listdir(d))
    assert "ckpt-35.index" not in names and "ckpt-37.index" not in names
    assert "ckpt-1.index" in names and "ckpt-2.index" in names
    # idempotent + safe on a manifest-less dir
    assert prune_orphaned_checkpoints(d) == 0
    assert prune_orphaned_checkpoints(str(tmp_path / "empty")) == 0


def test_mixed_depth_ensemble_loads_and_predicts(tmp_path, monkeypatch):
    """A 1-block and a 2-block member coexist: each loads at ITS depth and the
    ensemble mean/std work — the whole point of per-checkpoint introspection."""
    from euclid_polish.ensemble import EnsembleModel

    base = str(tmp_path / "ensemble")
    for name, blocks in (("member_00", 1), ("member_01", 2)):
        d = f"{base}/{name}"
        m = Model(d, num_res_blocks=blocks, seed=blocks)
        _save_model_ckpt(m._tf_model, d)

    ens = EnsembleModel(base, num_res_blocks=32)   # stale global default
    assert ens.n_members == 2
    assert sorted(m._num_res_blocks for m in ens.members) == [1, 2]
    lr = np.random.default_rng(0).normal(0, 1, (8, 8, 4)).astype(np.float32)
    mean, std = ens.predict(lr)
    assert mean.shape == (16, 16, 4) and std.shape == (16, 16, 4)
    assert float(std.mean()) > 0                    # genuinely different nets
