"""The ``bright_stars`` recipe: parsing, the live forward, the ensemble CLI /
member spec, origin.json and continue identity."""

import argparse
import json

import pytest

from euclid_polish import ensemble as ens_mod
from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel, MemberTrainSpec
from euclid_polish.sky.observation.bright_stars import (
    canonical_bright_star_recipe,
    parse_bright_star_recipe,
)
from euclid_polish.training.forward_onthefly import OnTheFlyForward, bright_star_kwargs
from scripts.run_pipeline import _infield_kwargs, _observation_config_from_args
from scripts.train_ensemble import _continue_identity, build_specs, parse_args
from tests.test_bright_stars import _sets


def test_recipe_parsing_and_canonical_form():
    off = parse_bright_star_recipe("off")
    assert not any(v for k, v in off.items() if k != "infield_bright_mag_min")
    assert parse_bright_star_recipe(None) == off == parse_bright_star_recipe("")
    kw = parse_bright_star_recipe(" trails , far,off_field,infield=8.5,nisp_taper")
    assert kw == {"bright_star_far_field": True, "nisp_psf_taper": True, "off_field_stars": True,
                  "bright_star_trails": True, "infield_bright_stars": True, "infield_bright_mag_min": 8.5}
    assert canonical_bright_star_recipe(" trails , far,off_field,infield=8.5,nisp_taper",
                                        default_cap=9.0) == "far,nisp_taper,off_field,trails,infield=8.5"
    assert canonical_bright_star_recipe("infield", default_cap=9.0) == "infield=9"
    assert canonical_bright_star_recipe("OFF", default_cap=9.0) == "off"
    for bad in ("far,glow", "off_field", "infield=13", "infield=x", "far=1"):
        with pytest.raises(ValueError):
            parse_bright_star_recipe(bad)
    assert bright_star_kwargs("infield")["infield_bright_mag_min"] == Config.TRAIN_INFIELD_BRIGHT_MAG_MIN


def test_live_forward_takes_the_recipe():
    sets = _sets()                     # spiky 511²/1023² kernels a far field can join
    fwd = OnTheFlyForward(sets, seed=1, add_artifacts=False, add_saturation=False, inject_stars=False,
                          psf_warp_prob=0.0, crops_per_field=1, blank_pixels="dr1",
                          **bright_star_kwargs("far,nisp_taper,trails"))
    cfg = fwd._sim.config
    assert cfg.bright_star_far_field and cfg.nisp_psf_taper and cfg.bright_star_trails
    assert not cfg.off_field_stars
    plain = OnTheFlyForward(sets, seed=1, inject_stars=False, crops_per_field=1, blank_pixels="off",
                            **bright_star_kwargs("trails"))
    assert not plain._sim.config.bright_star_trails          # trails need blank pixels


def test_member_spec_cli_and_continue_carry_bright_stars(tmp_path):
    base = ["--count", "1", "--steps", "10", "--forward-onthefly", "1"]
    spec = build_specs(parse_args(base + ["--member-spec", json.dumps([{"bright_stars": "off_field, far"}])]),
                       str(tmp_path / "e1"))[0]
    assert spec.bright_stars == "far,off_field"
    cli = build_specs(parse_args(base + ["--bright-stars", "far,trails"]), str(tmp_path / "e2"))[0]
    assert cli.bright_stars == "far,trails"
    assert build_specs(parse_args(base), str(tmp_path / "e3"))[0].bright_stars == "off"
    for bad in (["--bright-stars", "off_field"], ["--bright-stars", "trails", "--blank-pixels", "off"]):
        with pytest.raises(SystemExit):
            build_specs(parse_args(base + bad), str(tmp_path / "e4"))
    with pytest.raises(SystemExit):                            # records cannot carry it
        build_specs(parse_args(["--count", "1", "--steps", "10", "--bright-stars", "far"]),
                    str(tmp_path / "e5"))
    d = tmp_path / "member_330"
    d.mkdir()
    (d / "origin.json").write_text(json.dumps({"bright_stars": "far,nisp_taper"}))
    assert _continue_identity("member_330", str(d), {}, {})["bright_stars"] == "far,nisp_taper"
    old = tmp_path / "member_331"
    old.mkdir()
    (old / "origin.json").write_text(json.dumps({}))
    assert _continue_identity("member_331", str(old), {}, {})["bright_stars"] == "off"


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


def test_train_members_records_and_passes_bright_stars(tmp_path, monkeypatch):
    monkeypatch.setattr(ens_mod, "Model", _Recorder)
    base = tmp_path / "ensemble"
    ens = EnsembleModel(str(base), _models=[])
    live = {"target_steps": 10, "run_steps": 10, "loss_norm": "l2", "forward_onthefly": True}
    ens.train_members("lr", "hr", [
        MemberTrainSpec(name="member_332", seed=1, bright_stars="far,off_field", **live),
        MemberTrainSpec(name="member_333", seed=2, **live),
    ])
    for k, (name, value) in enumerate((("member_332", "far,off_field"), ("member_333", "off"))):
        assert json.loads((base / name / "origin.json").read_text())["bright_stars"] == value
        assert ens._models[k].trained["bright_stars"] == value


def test_generation_takes_the_recipe():
    cfg = _observation_config_from_args(argparse.Namespace(bright_stars="far,off_field,trails",
                                                           blank_pixels="dr1"))
    assert cfg.bright_star_far_field and cfg.off_field_stars and cfg.bright_star_trails
    assert not cfg.nisp_psf_taper
    plain = _observation_config_from_args(argparse.Namespace(blank_pixels="dr1"))
    assert not (plain.bright_star_far_field or plain.off_field_stars or plain.bright_star_trails)
    with pytest.raises(ValueError):
        _observation_config_from_args(argparse.Namespace(bright_stars="trails", blank_pixels="off"))
    assert _infield_kwargs("infield=8") == {"infield_bright_stars": True, "infield_bright_mag_min": 8.0}
    assert _infield_kwargs("off")["infield_bright_stars"] is False
