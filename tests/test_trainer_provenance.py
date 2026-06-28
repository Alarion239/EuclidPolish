"""The trainer writes a checkpoint identity sidecar, and a resumed run keeps it.

Targeted at the emission method so it needs no full training loop.
"""

from __future__ import annotations

import tensorflow as tf

from euclid_polish.provenance.checkpoint import read_checkpoint_provenance
from euclid_polish.training.trainer import Trainer


def _tiny_model():
    inp = tf.keras.Input(shape=(None, None, 1))
    out = tf.keras.layers.Conv2D(1, 3, padding="same")(inp)
    return tf.keras.Model(inp, out)


def test_trainer_emits_checkpoint_id(tmp_path):
    ckpt = str(tmp_path / "ckpt")
    tr = Trainer(_tiny_model(), checkpoint_dir=ckpt)
    tr._emit_checkpoint_provenance()
    stamp = read_checkpoint_provenance(ckpt)
    assert stamp is not None
    assert not stamp.id.is_sentinel


def test_resumed_trainer_keeps_the_same_id(tmp_path):
    ckpt = str(tmp_path / "ckpt")
    tr = Trainer(_tiny_model(), checkpoint_dir=ckpt)
    tr._emit_checkpoint_provenance()
    first = read_checkpoint_provenance(ckpt).id

    # A fresh Trainer over the same dir simulates resuming a run.
    tr2 = Trainer(_tiny_model(), checkpoint_dir=ckpt)
    tr2._emit_checkpoint_provenance()
    assert read_checkpoint_provenance(ckpt).id == first


def test_emit_is_best_effort_and_never_raises(tmp_path, monkeypatch):
    ckpt = str(tmp_path / "ckpt")
    tr = Trainer(_tiny_model(), checkpoint_dir=ckpt)

    # Force the write to fail; the trainer must swallow it, not crash training.
    import euclid_polish.training.trainer as trainer_mod
    monkeypatch.setattr(trainer_mod, "write_checkpoint_provenance",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    tr._emit_checkpoint_provenance()   # must not raise
