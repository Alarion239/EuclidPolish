"""Galaxy pre-blur (2026-10-28): live training smooths the clean galaxy field
(never the stars) on a share of visits, so the truth holds smoother faint
light than the TNG stamps alone."""
from __future__ import annotations

import json

import numpy as np
import pytest

from euclid_polish import ensemble as ens_mod
from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel, MemberTrainSpec
from euclid_polish.psf.psf_library import load_all_band_psf_sets
from euclid_polish.training import forward_onthefly as fwd_mod
from euclid_polish.training.forward_onthefly import OnTheFlyForward, validate_galaxy_blur
from scripts.train_ensemble import _continue_identity, build_specs, parse_args

KW = {"seed": 1, "crops_per_field": 1, "hr_crop_size": 64, "add_noise": False,
      "add_artifacts": False, "add_saturation": False, "inject_stars": False,
      "psf_warp_prob": 0.0}


def _field(n=96):
    rng = np.random.default_rng(0)
    field = np.zeros((n, n, 4), np.float32)
    field[rng.integers(24, n - 24, 40), rng.integers(24, n - 24, 40), :] = 100.0   # interior speckles
    return field


def test_validate_galaxy_blur_ranges():
    assert validate_galaxy_blur(0.5, 0.4) == (0.5, 0.4)
    for bad in ((-0.1, 0.4), (1.1, 0.4), (0.5, 0.0), (0.5, Config.TARGET_PSF_FWHM_MAX_ARCSEC + 1)):
        with pytest.raises(ValueError):
            validate_galaxy_blur(*bad)


def test_blur_smooths_galaxies_conserves_flux_and_is_off_by_default():
    rng = np.random.default_rng(3)
    field = _field()
    sets = load_all_band_psf_sets(psf_dir="/nonexistent_dir_for_test")
    off = OnTheFlyForward(sets, **KW)
    assert off.galaxy_blur_prob == 0.0
    assert off._blur_galaxies(field, rng) is field
    on = OnTheFlyForward(sets, galaxy_blur_prob=1.0, galaxy_blur_fwhm_max_arcsec=0.4, **KW)
    blurred = on._blur_galaxies(field, np.random.default_rng(4))
    assert blurred.max() < field.max()
    np.testing.assert_allclose(blurred.sum(axis=(0, 1)), field.sum(axis=(0, 1)), rtol=1e-4)


def test_blur_applies_before_the_stars(monkeypatch):
    """Stars are deposited onto their own plane after the blur: they stay sharp."""
    order = []
    real_blur = fwd_mod.blur_target_array

    def spy_blur(arr, *args, **kwargs):
        order.append(("blur", float(np.asarray(arr).max())))
        return real_blur(arr, *args, **kwargs)

    def fake_stars(self, canvas, rng):
        order.append(("stars", None))
        canvas[10, 10, :] = 5000.0

    monkeypatch.setattr(fwd_mod, "blur_target_array", spy_blur)
    monkeypatch.setattr(OnTheFlyForward, "_inject_stars", fake_stars)
    sets = load_all_band_psf_sets(psf_dir="/nonexistent_dir_for_test")
    fwd = OnTheFlyForward(sets, galaxy_blur_prob=1.0, **{**KW, "target_fwhm_arcsec": 0.0})
    field = _field()
    fwd.crops(field)
    # The galaxy blur sees the star-free field (max = galaxy pixels), then stars come.
    assert order[0] == ("blur", 100.0) and order[1] == ("stars", None)


def test_member_spec_cli_and_continue_carry_the_blur(tmp_path):
    member = {"forward_onthefly": True, "galaxy_blur_prob": 0.5, "galaxy_blur_fwhm_max": 0.3}
    spec = build_specs(parse_args(["--count", "1", "--steps", "10", "--member-spec", json.dumps([member])]),
                       str(tmp_path / "e1"))[0]
    assert spec.galaxy_blur_prob == 0.5 and spec.galaxy_blur_fwhm_max == 0.3
    cli = build_specs(parse_args(["--count", "1", "--steps", "10", "--forward-onthefly", "1",
                                  "--galaxy-blur-prob", "0.25"]), str(tmp_path / "e2"))[0]
    assert cli.galaxy_blur_prob == 0.25
    assert cli.galaxy_blur_fwhm_max == Config.TRAIN_GALAXY_BLUR_FWHM_MAX_ARCSEC
    plain = build_specs(parse_args(["--count", "1", "--steps", "10"]), str(tmp_path / "e3"))[0]
    assert plain.galaxy_blur_prob == 0.0
    for bad in ({**member, "galaxy_blur_prob": 2}, {"galaxy_blur_prob": 0.5}):   # range; needs live forward
        with pytest.raises(SystemExit):
            build_specs(parse_args(["--count", "1", "--steps", "10", "--member-spec", json.dumps([bad])]),
                        str(tmp_path / "e4"))
    d = tmp_path / "member_320"
    d.mkdir()
    (d / "origin.json").write_text(json.dumps({"galaxy_blur_prob": 0.5, "galaxy_blur_fwhm_max": 0.3}))
    identity = _continue_identity("member_320", str(d), {}, {})
    assert identity["galaxy_blur_prob"] == 0.5 and identity["galaxy_blur_fwhm_max"] == 0.3


class _Recorder:
    def __init__(self, checkpoint_dir, **kwargs):
        self._num_res_blocks = kwargs.get("num_res_blocks", 32)
        self._asinh_knee = None
        self._asinh_knees = None
        self.trained: dict = {}

    def train(self, lr, hr, **kwargs):
        self.trained = kwargs


def test_train_members_records_and_passes_the_blur(tmp_path, monkeypatch):
    monkeypatch.setattr(ens_mod, "Model", _Recorder)
    base = tmp_path / "ensemble"
    spec = MemberTrainSpec(name="member_321", seed=1, target_steps=10, run_steps=10,
                           forward_onthefly=True, galaxy_blur_prob=0.5, galaxy_blur_fwhm_max=0.3)
    ens = EnsembleModel(str(base), _models=[])
    ens.train_members("lr", "hr", [spec])
    origin = json.loads((base / "member_321" / "origin.json").read_text())
    assert origin["galaxy_blur_prob"] == 0.5 and origin["galaxy_blur_fwhm_max"] == 0.3
    kw = ens._models[0].trained
    assert kw["galaxy_blur_prob"] == 0.5 and kw["galaxy_blur_fwhm_max"] == 0.3
