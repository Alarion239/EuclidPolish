"""The knee tilt of the balanced multi-knee loss (``knee_tilt``): per-knee
weights ``(k / k_ref)**tilt`` normalised to a mean of 1, a weighted geometric
mean over the band × knee channels, the Model / member-spec / CLI / continue
knob."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.model import Model
from euclid_polish.training.losses import channel_balanced_loss, knee_tilt_weights
from scripts.train_ensemble import _continue_identity, build_specs, parse_args

KNEES = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def test_tilt_weights_lean_toward_small_or_large_knees_with_a_mean_of_one():
    flat = knee_tilt_weights(KNEES, 0.0, 4)
    assert flat == [1.0] * 32
    small = knee_tilt_weights(KNEES, -0.25, 4)
    assert len(small) == 32 and math.isclose(sum(small) / 32, 1.0)
    per_knee = small[::4]
    assert per_knee == sorted(per_knee, reverse=True)          # small knees weigh most
    assert math.isclose(per_knee[0] / per_knee[-1], 10 ** (0.25 * 7))   # 7 decades
    assert small[:4] == [per_knee[0]] * 4                        # every band of a knee alike
    large = knee_tilt_weights(KNEES, 0.25, 4)[::4]
    assert large == sorted(large)


def test_the_weighted_balanced_loss_is_a_weighted_geometric_mean():
    a = tf.zeros((1, 2, 2, 3))
    b = tf.constant(np.stack([np.full((2, 2), v, np.float32) for v in (1.0, 2.0, 4.0)], -1)[None])
    per_channel = np.array([1.0, 2.0, 4.0])                      # l2 = RMS of each channel
    plain = float(channel_balanced_loss("l2")(a, b))
    assert plain == pytest.approx(np.exp(np.log(per_channel).mean()), rel=1e-5)
    w = [3.0, 1.0, 0.5]
    tilted = float(channel_balanced_loss("l2", w)(a, b))
    assert tilted == pytest.approx(np.exp((np.array(w) * np.log(per_channel)).sum() / sum(w)), rel=1e-5)
    with pytest.raises(ValueError, match="weights"):
        channel_balanced_loss("l2", [1.0, 1.0])(a, b)


def test_the_model_refuses_a_tilt_without_a_balanced_multi_knee_loss(tmp_path):
    single = Model(str(tmp_path / "s"), scale=2, num_res_blocks=1)
    with pytest.raises(ValueError, match="knee_tilt"):
        single._member_loss("l2", "plain", knee_tilt=-0.25)
    multi = Model(str(tmp_path / "m"), scale=2, num_res_blocks=1, asinh_knees=KNEES[2:],
                  output_knee=10.0, learn_output_knee=True)
    with pytest.raises(ValueError, match="knee_tilt"):
        multi._member_loss("l2", "plain", knee_tilt=-0.25)
    assert callable(multi._member_loss("l2", "balanced", knee_tilt=-0.25))


def test_member_spec_cli_and_continue_carry_the_tilt(tmp_path):
    member = {"asinh_knees": list(KNEES), "learn_output_knee": True, "knee_loss": "balanced",
              "knee_tilt": -0.25}
    spec = build_specs(parse_args(["--count", "1", "--steps", "10", "--member-spec",
                                   json.dumps([member])]), str(tmp_path / "e1"))[0]
    assert spec.knee_tilt == -0.25
    cli = build_specs(parse_args(["--count", "1", "--steps", "10", "--asinh-knees",
                                  ",".join(map(str, KNEES)), "--learn-output-knee",
                                  "--knee-loss", "balanced", "--knee-tilt", "0.25"]),
                      str(tmp_path / "e2"))[0]
    assert cli.knee_tilt == 0.25
    assert build_specs(parse_args(["--count", "1", "--steps", "10"]), str(tmp_path / "e3"))[0].knee_tilt == 0.0
    for bad in ({**member, "knee_tilt": "x"}, {**member, "knee_tilt": 3}):
        with pytest.raises(SystemExit):
            build_specs(parse_args(["--count", "1", "--steps", "10", "--member-spec",
                                    json.dumps([bad])]), str(tmp_path / "e4"))
    d = tmp_path / "member_312"
    d.mkdir()
    (d / "origin.json").write_text(json.dumps({"asinh_knees": list(KNEES), "knee_loss": "balanced",
                                               "knee_tilt": -0.25}))
    assert _continue_identity("member_312", str(d), {}, {})["knee_tilt"] == -0.25
    assert "knee_tilt" not in _continue_identity("member_312", str(d), {}, {"knee_tilt": 0.0})
