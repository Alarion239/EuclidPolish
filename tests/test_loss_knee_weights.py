"""Per-knee weights of the balanced multi-knee loss (``loss_knee_weights``):
one weight per scored knee, repeated per band and normalised to a mean of 1;
under a loss floor, weight 3 on 0.1 e⁻ of a 0.1…1e4 grid scores like the
eight-knee grid whose 1e-3/1e-2/0.1 e⁻ channels are copies; the Model /
member-spec / CLI / continue knob."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.model import Model
from euclid_polish.training.augmentation import asinh_stretch_multi_knee
from euclid_polish.training.losses import channel_balanced_loss, knee_channel_weights, knee_floored_loss
from scripts.train_ensemble import _continue_identity, build_specs, parse_args

KNEES = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
SIX = KNEES[2:]


def test_knee_weights_repeat_per_band_with_a_mean_of_one():
    w = knee_channel_weights([3, 1, 1, 1, 1, 1], 4)
    assert len(w) == 24 and math.isclose(sum(w) / 24, 1.0)
    assert w[:4] == [w[0]] * 4 and math.isclose(w[0] / w[4], 3.0)
    for bad in ([], [1, 0], [1, -2], [1, float("nan")]):
        with pytest.raises(ValueError, match="positive"):
            knee_channel_weights(bad, 4)


def test_under_a_floor_weight_three_on_the_lowest_knee_stands_for_the_sub_noise_copies():
    rng = np.random.default_rng(0)
    floor = [5.0, 0.5, 0.5, 0.5]
    hr_e = rng.gamma(0.5, 30.0, (2, 16, 16, 4)).astype(np.float32)          # a truth is ≥ 0

    def scored(sr_e, knees, weights=None):
        per_channel = None if weights is None else knee_channel_weights(weights, 4)
        loss = knee_floored_loss(channel_balanced_loss("l2", per_channel), knees, floor, 4)
        return float(loss(tf.constant(sr_e), asinh_stretch_multi_knee(tf.constant(hr_e), knees)))

    # a prediction that stays above −f (249's own: never below on 15 test fields)
    sr_e = (hr_e * rng.lognormal(0.0, 0.2, hr_e.shape) + rng.normal(0, 0.05, hr_e.shape)).astype(np.float32)
    assert (sr_e + np.asarray(floor, np.float32) > 0).all()
    eight = scored(sr_e, KNEES)
    assert scored(sr_e, SIX, [3, 1, 1, 1, 1, 1]) == pytest.approx(eight, rel=0.01)
    assert scored(sr_e, SIX) < 0.8 * eight                           # the weight is what matters
    # below −f the copies part: asinh flips sign and the tiny knees punish ~ln(1/q) harder
    dipped = sr_e.copy()
    dipped[..., 1][hr_e[..., 1] < 0.05] = -2.0
    # (a few Y pixels move the 32-channel geometric mean by ~5 %)
    assert scored(dipped, KNEES) > 1.03 * scored(dipped, SIX, [3, 1, 1, 1, 1, 1])


def test_the_model_checks_the_weights_against_the_scored_knees(tmp_path):
    single = Model(str(tmp_path / "s"), scale=2, num_res_blocks=1)
    with pytest.raises(ValueError, match="loss_knee_weights"):
        single._member_loss("l2", "plain", loss_knee_weights=[1, 2])
    multi = Model(str(tmp_path / "m"), scale=2, num_res_blocks=1, asinh_knees=SIX,
                  output_knee=10.0, learn_output_knee=True)
    with pytest.raises(ValueError, match="loss_knee_weights"):
        multi._member_loss("l2", "plain", loss_knee_weights=[3, 1, 1, 1, 1, 1])
    with pytest.raises(ValueError, match="one weight per scored knee"):
        multi._member_loss("l2", "balanced", loss_knee_weights=[3, 1])
    assert callable(multi._member_loss("l2", "balanced", loss_floor=1.0, loss_knee_weights=[3, 1, 1, 1, 1, 1]))
    # with a tilt the two multiply
    assert callable(multi._member_loss("l2", "balanced", knee_tilt=-0.25, loss_knee_weights=[3, 1, 1, 1, 1, 1]))
    with pytest.raises(ValueError, match="lupton"):
        multi._member_loss("l2", "balanced", lupton=True, loss_knee_weights=[3, 1, 1, 1, 1, 1])


def test_member_spec_cli_and_continue_carry_the_weights(tmp_path):
    member = {"asinh_knees": list(SIX), "learn_output_knee": True, "knee_loss": "balanced",
              "loss_floor": 1.0, "loss_knee_weights": [3, 1, 1, 1, 1, 1]}
    spec = build_specs(parse_args(["--count", "1", "--steps", "10", "--member-spec",
                                   json.dumps([member])]), str(tmp_path / "e1"))[0]
    assert spec.loss_knee_weights == (3.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    cli = build_specs(parse_args(["--count", "1", "--steps", "10", "--asinh-knees",
                                  ",".join(map(str, SIX)), "--learn-output-knee",
                                  "--knee-loss", "balanced", "--loss-knee-weights", "3,1,1,1,1,1"]),
                      str(tmp_path / "e2"))[0]
    assert cli.loss_knee_weights == (3.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert build_specs(parse_args(["--count", "1", "--steps", "10"]), str(tmp_path / "e3"))[0].loss_knee_weights is None
    for bad in ({**member, "loss_knee_weights": [3, 1]}, {**member, "loss_knee_weights": [3, 0, 1, 1, 1, 1]},
                {**member, "loss_knee_weights": "x"}):
        with pytest.raises(SystemExit):
            build_specs(parse_args(["--count", "1", "--steps", "10", "--member-spec",
                                    json.dumps([bad])]), str(tmp_path / "e4"))
    d = tmp_path / "member_312"
    d.mkdir()
    (d / "origin.json").write_text(json.dumps({"asinh_knees": list(SIX), "knee_loss": "balanced",
                                               "loss_knee_weights": [3, 1, 1, 1, 1, 1]}))
    assert _continue_identity("member_312", str(d), {}, {})["loss_knee_weights"] == (3.0, 1.0, 1.0, 1.0, 1.0, 1.0)
