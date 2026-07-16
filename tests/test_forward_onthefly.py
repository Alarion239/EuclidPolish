"""On-the-fly forward training: full-field forward → K aligned crops.

Key invariants: HR crops are exact block-aligned sub-tiles of the input
clean field (the clean target IS the scene); LR crops are the matching
tiles of the full-field forward output (so out-of-crop PSF-wing flux is
present — no truncated-kernel approximation); the whole draw is seeded;
and the tf.data pipeline yields stretched batches of the right shape.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.image.tfio import tfrecord_path, write_images
from euclid_polish.model import Model
from euclid_polish.psf.psf_library import load_all_band_psf_sets
from euclid_polish.training.forward_onthefly import (
    OnTheFlyForward,
    member_psf_sets,
)
from scripts.train_ensemble import build_specs, parse_args

FIELD = 128                                  # ≥ DEFAULT_HR_CROP_SIZE (96)
CROP = int(Config.DEFAULT_HR_CROP_SIZE)


@pytest.fixture(scope="module")
def gaussian_sets():
    # No FITS on disk → 1-kernel Gaussian fallback per band (deterministic
    # PSF pick, so noise-off forwards are fully reproducible).
    return load_all_band_psf_sets(psf_dir="/nonexistent_dir_for_test")


def _field(seed=0, n=FIELD):
    rng = np.random.default_rng(seed)
    f = np.abs(rng.normal(20, 5, (n, n, 4))).astype(np.float32)
    f[n // 3, n // 3, :] += 5e4              # a bright star with wings
    return f


def test_crop_shapes_and_count(gaussian_sets):
    fwd = OnTheFlyForward(gaussian_sets, seed=1, crops_per_field=4)
    lr, hr = fwd.crops(_field())
    assert lr.shape == (4, CROP // 2, CROP // 2, 4)
    assert hr.shape == (4, CROP, CROP, 4)
    assert np.isfinite(lr).all() and np.isfinite(hr).all()


def test_hr_crops_are_subtiles_and_lr_matches_full_forward(gaussian_sets):
    """With noise off the forward is deterministic: every HR crop must be a
    block-aligned tile of the input field, and the LR crop must equal the
    SAME tile of the full-field forward output (wings included)."""
    field = _field()
    fwd = OnTheFlyForward(gaussian_sets, seed=2, crops_per_field=3,
                          add_noise=False, add_artifacts=False,
                          add_saturation=False, inject_stars=False,
                          psf_warp_prob=0.0)
    lr_crops, hr_crops = fwd.crops(field)
    # Reference full-field forward (deterministic: 1-kernel sets, no noise).
    img = Image(data=field, pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True)
    lr_full, hr_full = fwd._sim.process(img, np.random.default_rng(0))
    lr_full, hr_full = np.asarray(lr_full.data), np.asarray(hr_full.data)

    for k in range(3):
        # locate the HR crop in the field (block-aligned offsets only)
        found = None
        for x in range(0, FIELD - CROP + 1, 2):
            for y in range(0, FIELD - CROP + 1, 2):
                if np.array_equal(hr_crops[k],
                                  hr_full[x: x + CROP, y: y + CROP, :]):
                    found = (x, y)
                    break
            if found:
                break
        assert found is not None, f"crop {k} is not a block-aligned subtile"
        x, y = found
        np.testing.assert_allclose(
            lr_crops[k],
            lr_full[x // 2: (x + CROP) // 2, y // 2: (y + CROP) // 2, :],
            rtol=1e-5,
            err_msg="LR crop != matching tile of the full-field forward")


def test_inject_stars_adds_flux_before_forward(gaussian_sets):
    """The injection primitive deposits fresh star flux onto the scene copy
    (HR deltas, pre-PSF) — directly, without the crop-offset RNG in play."""
    fwd = OnTheFlyForward(gaussian_sets, seed=5, inject_stars=True,
                          star_density_arcmin2=500.0)
    field = _field()
    scene = field.copy()
    fwd._inject_stars(scene, np.random.default_rng(0))
    assert scene.sum() > field.sum()             # stars deposited → more flux
    # ...and only added flux (deltas are non-negative on top of the field).
    assert np.all(scene >= field - 1e-3)


def test_crops_target_is_starless_even_with_injection(gaussian_sets):
    """Stars-as-artifacts: with injection on, every HR crop is still an exact
    block-aligned sub-tile of the ORIGINAL starless field — the injected stars
    reach the LR but never the target, so the model is supervised to erase
    them."""
    field = _field()
    fwd = OnTheFlyForward(gaussian_sets, seed=5, crops_per_field=4,
                          add_noise=False, add_artifacts=False,
                          add_saturation=False, inject_stars=True,
                          star_density_arcmin2=500.0)
    _lr, hr_crops = fwd.crops(field)
    for k in range(hr_crops.shape[0]):
        found = any(
            np.array_equal(hr_crops[k], field[x: x + CROP, y: y + CROP, :])
            for x in range(0, FIELD - CROP + 1, 2)
            for y in range(0, FIELD - CROP + 1, 2))
        assert found, f"HR crop {k} is not a starless sub-tile of the field"


def test_both_regimes_inject_stars_only_target_differs(gaussian_sets):
    """Both starless and starfull inject the SAME fresh stars (identical LR);
    the target is what differs — starless erases them, starfull keeps them."""
    field = _field()
    kw = {"seed": 5, "crops_per_field": 4, "add_noise": False,
          "add_artifacts": False, "add_saturation": False, "inject_stars": True,
          "star_density_arcmin2": 500.0}
    lr_less, hr_less = OnTheFlyForward(gaussian_sets, starless=True, **kw).crops(field)
    lr_full, hr_full = OnTheFlyForward(gaussian_sets, starless=False, **kw).crops(field)
    # Same seed → same injected stars → IDENTICAL LR in both regimes.
    np.testing.assert_array_equal(lr_less, lr_full)
    # The starfull target keeps the stars (more flux); starless erases them.
    assert hr_full.sum() > hr_less.sum()
    # ...and each starless HR crop is a pure sub-tile of the ORIGINAL field.
    for k in range(hr_less.shape[0]):
        assert any(
            np.array_equal(hr_less[k], field[x: x + CROP, y: y + CROP, :])
            for x in range(0, FIELD - CROP + 1, 2)
            for y in range(0, FIELD - CROP + 1, 2))


def test_inject_stars_off_reproduces_plain_forward(gaussian_sets):
    """inject_stars=False (validate/test-style) is deterministic per seed."""
    field = _field()
    a = OnTheFlyForward(gaussian_sets, seed=9, crops_per_field=2,
                        add_noise=False, inject_stars=False).crops(field)
    b = OnTheFlyForward(gaussian_sets, seed=9, crops_per_field=2,
                        add_noise=False, inject_stars=False).crops(field)
    np.testing.assert_array_equal(a[0], b[0])


def test_seeded_draws_reproduce_and_differ(gaussian_sets):
    field = _field()
    a1 = OnTheFlyForward(gaussian_sets, seed=7, crops_per_field=2).crops(field)
    a2 = OnTheFlyForward(gaussian_sets, seed=7, crops_per_field=2).crops(field)
    b = OnTheFlyForward(gaussian_sets, seed=8, crops_per_field=2).crops(field)
    np.testing.assert_array_equal(a1[0], a2[0])
    np.testing.assert_array_equal(a1[1], a2[1])
    assert not np.array_equal(a1[0], b[0])   # different seed → different draw


def test_revisits_redraw_noise(gaussian_sets):
    fwd = OnTheFlyForward(gaussian_sets, seed=3, crops_per_field=1)
    field = _field()
    first = fwd.crops(field)[0]
    second = fwd.crops(field)[0]             # next epoch's visit
    assert not np.array_equal(first, second)


def test_member_psf_sets_fallback_note(tmp_path):
    sets, note = member_psf_sets(seed=1, psf_dir=str(tmp_path))
    assert "no rotation pool" in note
    assert set(sets) == {b.name for b in Config.BANDS}


def test_member_psf_sets_uses_pool_bag(tmp_path):
    from euclid_polish.psf.core import PSF
    from euclid_polish.psf.psf_set import PSFSet
    from euclid_polish.psf.rotpool import build_rotation_pool

    rng = np.random.default_rng(0)
    sets = {b.name: PSFSet.from_psfs(
                [PSF(data=(lambda k: k / k.sum())(
                     np.abs(rng.normal(1, 0.3, (11, 11))).astype(np.float32)),
                     pixel_scale=Config.DEFAULT_PIXEL_SCALE)
                 for _ in range(6)])
            for b in Config.BANDS}
    build_rotation_pool(sets, psf_dir=str(tmp_path), rotations=2, seed=0,
                        workers=1)
    bag, note = member_psf_sets(seed=5, psf_subset=3, psf_dir=str(tmp_path))
    assert "bagged to 3 clusters" in note
    assert bag["VIS"].n == 3 * 3              # 3 clusters × (2 rolls + orig)


def test_onthefly_pipeline_yields_batches(gaussian_sets, tmp_path):
    imgs = [Image(data=_field(seed=i), pixel_scale_arcsec=0.05,
                  band_names=Config.HR_TARGET_BAND_NAMES, is_clean=True,
                  index=i) for i in range(3)]
    write_images(imgs, "clean_train", records_dir=str(tmp_path))
    m = Model(str(tmp_path / "ckpt"), scale=2, num_res_blocks=1)
    fwd = OnTheFlyForward(gaussian_sets, seed=4, crops_per_field=4)
    ds = m._build_onthefly_pipeline(
        tfrecord_path(str(tmp_path), "clean_train"), 8, fwd)
    lr, hr = next(iter(ds))
    assert lr.shape == (8, CROP // 2, CROP // 2, 4)
    assert hr.shape == (8, CROP, CROP, 4)
    # asinh-stretched values, not raw electrons
    assert float(tf.reduce_max(tf.abs(hr))) < 25.0
    assert np.isfinite(lr.numpy()).all()


def test_build_specs_forward_flags(tmp_path):
    args = parse_args(["--mode", "add", "--count", "2", "--steps", "10",
                       "--forward-onthefly", "1", "--psf-subset", "48",
                       "--crops-per-field", "8",
                       "--psf-warp-prob", "0.75",
                       "--psf-warp-alpha-max", "12",
                       "--psf-warp-sigma", "4",
                       "--member-spec",
                       '[{"forward_onthefly": false}]'])
    s0, s1 = build_specs(args, str(tmp_path / "ens"))
    assert (s0.forward_onthefly, s1.forward_onthefly) == (False, True)
    assert s1.psf_subset == 48 and s1.crops_per_field == 8
    assert s1.psf_warp_prob == pytest.approx(0.75)
    assert s1.psf_warp_alpha_max == pytest.approx(12.0)
    assert s1.psf_warp_sigma == pytest.approx(4.0)


def test_ensemble_train_step_forwards_onthefly_flags(monkeypatch):
    from euclid_polish.web.fasrc_pipeline import EnsembleTrainStep
    monkeypatch.setattr(
        "euclid_polish.web.fasrc_pipeline.next_member_names",
        lambda base, k: ["member_00"])
    cmd = " ".join(EnsembleTrainStep().build_command({
        "mode": "add", "count": "1", "steps": "1000",
        "forward_onthefly": "1", "psf_subset": "48",
        "crops_per_field": "8", "psf_warp_prob": "0.75",
        "psf_warp_alpha_max": "12", "psf_warp_sigma": "4"}))
    assert "--forward-onthefly 1" in cmd
    assert "--psf-subset 48" in cmd
    assert "--crops-per-field 8" in cmd
    assert "--psf-warp-prob 0.75" in cmd
    assert "--psf-warp-alpha-max 12" in cmd
    assert "--psf-warp-sigma 4" in cmd
    # unchecked box → none of the on-the-fly flags
    cmd = " ".join(EnsembleTrainStep().build_command({
        "mode": "add", "count": "1", "steps": "1000",
        "crops_per_field": "8"}))
    assert "--forward-onthefly" not in cmd and "--crops-per-field" not in cmd
