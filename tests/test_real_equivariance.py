"""The real-sky dihedral equivariance penalty (training/equivariance.py)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish import ensemble as ens_mod
from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel, MemberTrainSpec
from euclid_polish.image import Image
from euclid_polish.image.tfio import tfrecord_path, write_images
from euclid_polish.model import Model
from euclid_polish.sky.observation import q1_mer_tiles
from euclid_polish.sky.observation.q1_fields import q1_field_for
from euclid_polish.training.equivariance import (
    D4_ORDER,
    RealEquivariance,
    dihedral,
    dihedral_inverse,
    load_real_tiles,
    real_crop_dataset,
    step_code,
)
from euclid_polish.training.losses import channel_balanced_loss, lp_loss
from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.training.trainer import Trainer
from scripts.build_real_equivariance_tiles import (
    EDGE_MARGIN_DEG,
    candidate_tiles,
    draw_positions,
)
from scripts.build_real_equivariance_tiles import main as build_tiles_main
from scripts.train_ensemble import _continue_identity, build_specs, parse_args
from tests.test_forward_onthefly import _stellar_prior_payload

KNEES = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def _square(seed=0, side=6, channels=2):
    return tf.constant(np.random.default_rng(seed).normal(size=(2, side, side, channels)),
                       tf.float32)


def _reference_d4(x):
    """The eight orientations the usual way: rot90 × k, each with and without a flip."""
    out = []
    for k in range(4):
        r = tf.image.rot90(x, k)
        out += [r, tf.image.flip_left_right(r)]
    return out


def test_the_eight_codes_are_exactly_the_dihedral_group():
    x = _square()
    mine = [dihedral(x, c).numpy() for c in range(D4_ORDER)]
    ref = [r.numpy() for r in _reference_d4(x)]
    assert np.array_equal(mine[0], x.numpy())
    for a in mine:                           # every code is one of the 8 ...
        assert sum(np.array_equal(a, r) for r in ref) == 1
    for r in ref:                            # ... and every element appears once
        assert sum(np.array_equal(a, r) for a in mine) == 1


def test_inverse_undoes_every_code_and_step_codes_skip_identity():
    x = _square(1)
    for c in range(D4_ORDER):
        assert np.array_equal(dihedral_inverse(dihedral(x, c), c).numpy(), x.numpy())
    codes = [int(step_code(tf.constant(s))) for s in range(14)]
    assert sorted(set(codes)) == list(range(1, D4_ORDER))
    assert codes[:7] == codes[7:]


def _mse(a, b):
    return tf.reduce_mean(tf.square(a - b))


def test_penalty_is_zero_for_an_equivariant_upsampler_and_positive_otherwise():
    x = tf.abs(_square(2, side=8, channels=4)) * 100.0
    nearest = tf.keras.Sequential([tf.keras.layers.UpSampling2D(2)])
    eq = RealEquivariance(lambda v: tf.math.asinh(v), lambda v: v)
    for code in range(1, D4_ORDER):
        assert float(eq.penalty(nearest, x, _mse, tf.constant(code))) == pytest.approx(0.0, abs=1e-12)

    kernel = np.zeros((3, 3, 4, 4), np.float32)
    for c in range(4):
        kernel[0, 2, c, c] = 1.0               # a one-sided shift: not equivariant
    lopsided = tf.keras.Sequential([
        tf.keras.layers.Conv2D(4, 3, padding="same", use_bias=False,
                               kernel_initializer=tf.constant_initializer(kernel)),
        tf.keras.layers.UpSampling2D(2)])
    assert float(eq.penalty(lopsided, x, _mse, tf.constant(1))) > 1e-3


def test_crops_are_seeded_even_aligned_and_from_the_tiles(tmp_path):
    side = 16
    tiles = np.arange(3 * side * side * 4, dtype=np.float32).reshape(3, side, side, 4)
    path = tmp_path / "tiles.npy"
    np.save(path, tiles)
    loaded = load_real_tiles(str(path))
    a = next(iter(real_crop_dataset(loaded, crop=8, batch_size=5, seed=7))).numpy()
    b = next(iter(real_crop_dataset(loaded, crop=8, batch_size=5, seed=7))).numpy()
    assert a.shape == (5, 8, 8, 4) and np.array_equal(a, b)
    flat = tiles.reshape(-1, 4)[:, 0]
    for crop in a:
        index = int(np.flatnonzero(flat == crop[0, 0, 0])[0])
        t, rest = divmod(index, side * side)
        y, x = divmod(rest, side)
        assert y % 2 == 0 and x % 2 == 0
        assert np.array_equal(crop, tiles[t, y:y + 8, x:x + 8])
    with pytest.raises(ValueError, match="even"):
        real_crop_dataset(loaded, crop=7, batch_size=1, seed=0)


def test_bad_tile_sets_are_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_real_tiles(str(tmp_path / "missing.npy"))
    np.save(tmp_path / "f64.npy", np.zeros((1, 4, 4, 4)))
    with pytest.raises(ValueError, match="float32"):
        load_real_tiles(str(tmp_path / "f64.npy"))
    np.save(tmp_path / "rect.npy", np.zeros((1, 4, 6, 4), np.float32))
    with pytest.raises(ValueError, match="S, S, 4"):
        load_real_tiles(str(tmp_path / "rect.npy"))


# -- the trainer ---------------------------------------------------------------

def _electron_model(seed=7):
    tf.keras.utils.set_random_seed(seed)
    return wdsr(scale=2, nchan_in=4, nchan_out=4, num_res_blocks=1, num_filters=4)


def _batches(seed=0):
    rng = np.random.default_rng(seed)
    lr = tf.constant(rng.normal(size=(2, 16, 16, 4)).astype(np.float32))
    hr = tf.constant(np.abs(rng.normal(size=(2, 32, 32, 4))).astype(np.float32))
    real = tf.constant(np.abs(rng.normal(size=(2, 16, 16, 4))).astype(np.float32) * 10.0)
    return lr, hr, real


def _trainer(path, *, weight=None, steps_per_call=1):
    eq = None if weight is None else RealEquivariance(lambda v: v, lambda v: v)
    return Trainer(_electron_model(), loss=_mse, checkpoint_dir=str(path), learning_rate=1e-3,
                   output_electrons=True, batch_stretch=lambda lr, hr: (lr, hr),
                   equivariance=eq, equivariance_weight=weight or 0.0,
                   steps_per_call=steps_per_call)


def test_the_trainer_adds_the_penalty_to_the_gradient_not_to_the_logged_loss(tmp_path):
    lr, hr, real = _batches()
    plain, zero, one = (_trainer(tmp_path / "p"), _trainer(tmp_path / "z", weight=0.0),
                        _trainer(tmp_path / "o", weight=1.0))
    losses = {"p": [float(plain.train_step(lr, hr)[0]) for _ in range(3)],
              "z": [float(zero.train_step(lr, hr, real)[0]) for _ in range(3)],
              "o": [float(one.train_step(lr, hr, real)[0]) for _ in range(3)]}
    assert losses["p"] == losses["z"]
    for a, b in zip(plain.model.get_weights(), zero.model.get_weights(), strict=True):
        np.testing.assert_array_equal(a, b)
    assert losses["o"][0] == losses["p"][0]
    assert any(not np.array_equal(a, b) for a, b in
               zip(plain.model.get_weights(), one.model.get_weights(), strict=True))
    assert one._read_window()["equivariance"] > 0.0
    assert plain._read_window()["equivariance"] == 0.0


def test_the_compiled_step_reads_no_host_memory_variable(tmp_path):
    """TF keeps int32 variables (the step counter) in host memory; an XLA step
    on a GPU cannot read them (11-19 crash, jobs 62219486/62233031). The D4
    code must come in as an argument, so the compiled gradient step captures
    no int32 variable."""
    lr, hr, real = _batches(2)
    eq = RealEquivariance(lambda v: v, lambda v: v)
    trainer = Trainer(_electron_model(), loss=_mse, checkpoint_dir=str(tmp_path / "x"),
                      learning_rate=1e-3, output_electrons=True,
                      batch_stretch=lambda lr, hr: (lr, hr), equivariance=eq,
                      equivariance_weight=1.0, jit_compile=True)
    trainer.train_step(lr, hr, real)
    traced = trainer._gradient_step.get_concrete_function(lr, hr, real, tf.constant(1))
    captured = [t for t in traced.captured_inputs if t.dtype == tf.resource]
    int32 = [v.name for v in trainer.checkpoint.model.variables + [trainer.checkpoint.step]
             if v.dtype == tf.int32 and any(v.handle is t for t in captured)]
    assert int32 == []


def test_multi_step_calls_take_three_element_batches(tmp_path):
    lr, hr, real = _batches(1)
    trainer = _trainer(tmp_path / "m", weight=1.0, steps_per_call=2)
    ds = tf.data.Dataset.from_tensors((lr, hr, real)).repeat()
    trainer.train_steps(iter(ds), tf.constant([1e-3, 1e-3], tf.float32))
    assert int(trainer.checkpoint.step.numpy()) == 2
    assert trainer._read_window()["equivariance"] > 0.0


def test_the_trainer_validates_the_penalty(tmp_path):
    eq = RealEquivariance(lambda v: v, lambda v: v)
    with pytest.raises(ValueError, match="electrons"):
        Trainer(_electron_model(), checkpoint_dir=str(tmp_path / "a"), equivariance=eq,
                equivariance_weight=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        Trainer(_electron_model(), checkpoint_dir=str(tmp_path / "b"), output_electrons=True,
                equivariance=eq, equivariance_weight=-1.0)



# -- Model.train, member spec, origin.json and continue -------------------------

def test_model_train_validates_the_knob(tmp_path):
    learned = Model(str(tmp_path / "l"), scale=2, num_res_blocks=1, asinh_knees=KNEES,
                    output_knee=10.0, learn_output_knee=True)
    with pytest.raises(ValueError, match="live"):
        learned.train("missing_lr", "missing_hr", steps=1, real_equivariance=1.0,
                      real_equivariance_tiles="t.npy")
    with pytest.raises(ValueError, match="tiles"):
        learned.train("missing_lr", "missing_hr", steps=1, real_equivariance=1.0,
                      forward_onthefly=True)
    with pytest.raises(ValueError, match="non-negative"):
        learned.train("missing_lr", "missing_hr", steps=1, real_equivariance=-1.0)
    fixed = Model(str(tmp_path / "f"), scale=2, num_res_blocks=1, asinh_knees=KNEES, output_knee=10.0)
    with pytest.raises(ValueError, match="learned-output-knee"):
        fixed.train("missing_lr", "missing_hr", steps=1, real_equivariance=1.0,
                    real_equivariance_tiles="t.npy", forward_onthefly=True)


class _Recorder:
    def __init__(self, checkpoint_dir, *, scale=2, num_res_blocks=32, seed=None, init_weights_from=None,
                 icnr=False, asinh_knee=None, asinh_knees=None, output_knee=None, learn_output_knee=False):
        self._num_res_blocks = num_res_blocks
        self._asinh_knee = asinh_knee
        self._asinh_knees = tuple(asinh_knees) if asinh_knees else None
        self._learn_output_knee = bool(learn_output_knee)
        self._output_knee = None if learn_output_knee else output_knee
        self.trained: dict = {}

    def train(self, lr, hr, **kwargs):
        self.trained = kwargs


def _specs(tmp_path, member, name="b"):
    return build_specs(parse_args(["--count", "1", "--steps", "10", "--member-spec",
                                   json.dumps([member])]), str(tmp_path / name))


def test_the_knob_from_member_spec_origin_and_continue(tmp_path, monkeypatch):
    member = {"asinh_knees": list(KNEES), "learn_output_knee": True, "forward_onthefly": True,
              "real_equivariance": 1.0, "real_equivariance_tiles": "real/edf_fs.npy",
              "real_equivariance_batch": 2}
    spec = _specs(tmp_path, member)[0]
    assert (spec.real_equivariance, spec.real_equivariance_tiles, spec.real_equivariance_batch) == (
        1.0, "real/edf_fs.npy", 2)
    off = _specs(tmp_path, {"asinh_knees": list(KNEES), "learn_output_knee": True}, "b0")[0]
    assert (off.real_equivariance, off.real_equivariance_tiles, off.real_equivariance_batch) == (
        0.0, None, None)
    for bad in ({**member, "real_equivariance": -1.0},
                {**member, "real_equivariance": True},
                {**member, "real_equivariance_tiles": None},
                {**member, "real_equivariance_batch": 0.5},
                {**member, "forward_onthefly": False},
                {**member, "learn_output_knee": False},
                {"asinh_knees": list(KNEES), "learn_output_knee": True, "forward_onthefly": True,
                 "real_equivariance_tiles": "real/edf_fs.npy"}):
        with pytest.raises(SystemExit):
            _specs(tmp_path, bad, "bad")
    monkeypatch.setattr(ens_mod, "Model", _Recorder)
    base = tmp_path / "ensemble"
    ens = EnsembleModel(str(base), _models=[])
    common = {"seed": 1, "target_steps": 10, "run_steps": 10, "loss_norm": "l2", "asinh_knees": KNEES,
              "knee_loss": "balanced", "output_knee": 10.0, "learn_output_knee": True,
              "forward_onthefly": True}
    ens.train_members("lr", "hr", [
        MemberTrainSpec(name="member_401", real_equivariance=1.0,
                        real_equivariance_tiles="real/edf_fs.npy", real_equivariance_batch=2, **common),
        MemberTrainSpec(name="member_402", **common)])
    origin = json.loads((base / "member_401" / "origin.json").read_text())
    assert (origin["real_equivariance"], origin["real_equivariance_tiles"],
            origin["real_equivariance_batch"]) == (1.0, "real/edf_fs.npy", 2)
    assert "real_equivariance" not in json.loads((base / "member_402" / "origin.json").read_text())
    assert ens._models[0].trained["real_equivariance"] == 1.0
    assert ens._models[0].trained["real_equivariance_tiles"] == "real/edf_fs.npy"
    assert ens._models[0].trained["real_equivariance_batch"] == 2
    assert "real_equivariance" not in ens._models[1].trained
    kept = _continue_identity("member_401", str(base / "member_401"), {}, {})
    assert (kept["real_equivariance"], kept["real_equivariance_tiles"],
            kept["real_equivariance_batch"]) == (1.0, "real/edf_fs.npy", 2)
    assert "real_equivariance" not in _continue_identity("member_402", str(base / "member_402"), {}, {})


# -- the real tile set builder --------------------------------------------------

def test_builder_draws_seeded_positions_inside_the_allowed_fields():
    tiles = candidate_tiles(("EDF-F", "EDF-S"))
    assert tiles and {t.field for t in tiles} == {"EDF-F", "EDF-S"}
    assert not any(t.rejected for t in tiles)
    a = draw_positions(tiles, 50, 3, fields=("EDF-F", "EDF-S"))
    assert a == draw_positions(tiles, 50, 3, fields=("EDF-F", "EDF-S"))
    by_id = {t.tile: t for t in tiles}
    for ra, dec, tile in a:
        assert q1_field_for(ra, dec) in ("EDF-F", "EDF-S")
        assert q1_mer_tiles.polygon_margin_deg(ra, dec, by_id[tile].polygon) >= EDGE_MARGIN_DEG


def test_builder_refuses_the_battery_field(tmp_path):
    assert build_tiles_main(["--fields", "EDF-F,EDF-N", "--store-dir", str(tmp_path),
                 "--out", str(tmp_path / "x.npy")]) == 2


def _write_records(directory, n_fields=2, field=128):
    rng = np.random.default_rng(0)

    def clean(i):
        f = np.abs(rng.normal(20, 5, (field, field, 4))).astype(np.float32)
        f[field // 3, field // 3, :] += 5e4
        return Image(data=f, pixel_scale_arcsec=0.05, band_names=Config.HR_TARGET_BAND_NAMES,
                     is_clean=True, index=i)

    write_images([clean(i) for i in range(n_fields)], "clean_train", records_dir=str(directory))
    write_images([clean(i) for i in range(n_fields)], "hr_validate", records_dir=str(directory))
    dirty = [Image(data=np.abs(rng.normal(20, 5, (field // 2, field // 2, 4))).astype(np.float32),
                   pixel_scale_arcsec=0.1, band_names=Config.LR_INPUT_BAND_NAMES,
                   is_clean=False, index=i) for i in range(n_fields)]
    write_images(dirty, "dirty_validate", records_dir=str(directory))


@pytest.mark.parametrize("fast", [False, True], ids=["plain", "xla-bf16"])
def test_a_live_member_trains_with_the_penalty_end_to_end(tmp_path, capsys, fast):
    """Model.train zips the real crops into the live batches, the trainer
    scores them, the window logs the penalty and the checkpoint records it."""
    _write_records(tmp_path)
    tiles = np.abs(np.random.default_rng(1).normal(5, 2, (3, 64, 64, 4))).astype(np.float32)
    np.save(tmp_path / "real.npy", tiles)
    m = Model(str(tmp_path / "ckpt"), scale=2, num_res_blocks=1, asinh_knees=KNEES,
              output_knee=10.0, learn_output_knee=True, seed=3)
    m.train(tfrecord_path(str(tmp_path), "dirty_train"), tfrecord_path(str(tmp_path), "clean_train"),
            steps=4, batch_size=2, forward_onthefly=True, hr_crop_size=64, crops_per_field=2,
            real_equivariance=1.0, real_equivariance_tiles=str(tmp_path / "real.npy"),
            star_prior_payload=_stellar_prior_payload(), evaluate_every=2,
            jit_compile=fast, mixed_precision=fast)
    out = capsys.readouterr().out
    assert "real-sky equivariance: λ=1, 2 × 32² LR crops/step from 3 tiles" in out
    assert "equivariance penalty" in out


# -- the p-norm root at an exact match -------------------------------------------

@pytest.mark.parametrize("norm", ["l2", "l3"])
def test_the_lp_root_has_a_finite_gradient_at_an_exact_match(norm):
    """The penalty compares two outputs of the same network, which can agree
    exactly (any D4-symmetric crop, e.g. a blank one). ``(mean|d|^p)^(1/p)``
    then has an infinite slope at 0, and the balanced loss's clamp turned it
    into 0·inf = NaN (11-20: every window of the λ1/λ10 runs spiked to
    |g|=inf). An exact match must give a zero gradient instead."""
    a = tf.Variable(tf.ones((1, 4, 4, 2)))
    b = tf.ones((1, 4, 4, 2))
    for loss in (lp_loss(norm), channel_balanced_loss(norm)):
        with tf.GradientTape() as tape:
            value = loss(a, b)
        grad = tape.gradient(value, a).numpy()
        assert np.all(np.isfinite(grad)), grad
        assert np.all(grad == 0.0)


@pytest.mark.parametrize("norm", ["l2", "l3"])
def test_the_lp_root_is_unchanged_away_from_zero(norm):
    p = {"l2": 2, "l3": 3}[norm]
    rng = np.random.default_rng(4)
    a = tf.Variable(rng.normal(size=(2, 8, 8, 3)).astype(np.float32))
    b = tf.constant(rng.normal(size=(2, 8, 8, 3)).astype(np.float32))
    with tf.GradientTape() as tape:
        value = lp_loss(norm)(a, b)
    grad = tape.gradient(value, a)
    with tf.GradientTape() as tape:
        reference = tf.reduce_mean(tf.abs(a - b) ** p) ** (1.0 / p)
    expected = tape.gradient(reference, a)
    assert float(value) == float(reference)
    assert np.array_equal(grad.numpy(), expected.numpy())
