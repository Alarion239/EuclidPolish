"""Ensemble-diversity knobs: Lp loss, dihedral augmentation, LR noise
augmentation, and the bootstrap field subset — plus their CLI plumbing.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.image.tfio import tfrecord_path, write_images
from euclid_polish.model import Model
from euclid_polish.training.augmentation import (
    _LR_READ_NOISE_NP,
    add_lr_noise,
    random_dihedral,
)
from euclid_polish.training.losses import lp_loss
from scripts.train_ensemble import build_specs, parse_args


# --------------------------------------------------------------------------- #
# Lp loss                                                                      #
# --------------------------------------------------------------------------- #
def test_lp_loss_l1_is_mae():
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.5, 1.0], [3.0, 6.0]])
    assert float(lp_loss("l1")(a, b)) == pytest.approx(
        float(tf.reduce_mean(tf.abs(a - b))))


def test_lp_loss_l2_is_rmse():
    rng = np.random.default_rng(0)
    a = tf.constant(rng.normal(size=(8, 8)).astype(np.float32))
    b = tf.zeros_like(a)
    assert float(lp_loss("l2")(a, b)) == pytest.approx(
        float(np.sqrt(np.mean(a.numpy() ** 2))), rel=1e-5)


def test_lp_losses_share_scale_and_order():
    """The p-th root keeps every norm in the residual's units: for the same
    residuals, L1 ≤ L2 ≤ L3 ≤ max|residual| (power-mean inequality)."""
    rng = np.random.default_rng(1)
    a = tf.constant(rng.normal(size=(32, 32)).astype(np.float32))
    b = tf.zeros_like(a)
    l1, l2, l3 = (float(lp_loss(n)(a, b)) for n in ("l1", "l2", "l3"))
    mx = float(np.abs(a.numpy()).max())
    assert l1 <= l2 <= l3 <= mx
    assert l3 > 0.5 * l1                     # same scale, not p-th-power tiny


def test_lp_loss_rejects_unknown_norm():
    with pytest.raises(ValueError, match="unknown loss norm"):
        lp_loss("l4")


# --------------------------------------------------------------------------- #
# Dihedral augmentation                                                        #
# --------------------------------------------------------------------------- #
def test_random_dihedral_applies_same_transform_to_both():
    """LR and HR receive the SAME orientation: an LR that equals the HR's
    2×2-average rebin still equals it after the transform (rebin commutes
    with the dihedral group on block-aligned squares)."""
    rng = np.random.default_rng(2)
    hr = rng.normal(size=(8, 8, 1)).astype(np.float32)
    lr = hr.reshape(4, 2, 4, 2, 1).mean(axis=(1, 3))
    tf.random.set_seed(0)
    for _ in range(16):                     # cover all 8 group elements
        lr_t, hr_t = random_dihedral(tf.constant(lr), tf.constant(hr))
        rebinned = hr_t.numpy().reshape(4, 2, 4, 2, 1).mean(axis=(1, 3))
        np.testing.assert_allclose(lr_t.numpy(), rebinned, rtol=1e-5)


def test_random_dihedral_covers_multiple_orientations():
    hr = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    lr = hr[::2, ::2]                       # any aligned array works here
    tf.random.set_seed(3)
    seen = {tuple(random_dihedral(tf.constant(lr), tf.constant(hr))[1]
                  .numpy().ravel()) for _ in range(64)}
    assert len(seen) >= 6                   # 8 orientations, allow collisions


# --------------------------------------------------------------------------- #
# Noise augmentation                                                           #
# --------------------------------------------------------------------------- #
def test_add_lr_noise_zero_is_identity():
    x = tf.constant(np.ones((8, 8, 4), np.float32))
    assert add_lr_noise(x, 0.0) is x


def test_add_lr_noise_std_scales_with_band_read_noise():
    x = tf.zeros((256, 256, 4))
    tf.random.set_seed(4)
    noised = add_lr_noise(x, 2.0).numpy()
    for c in range(4):
        expected = 2.0 * float(_LR_READ_NOISE_NP[c])
        assert noised[..., c].std() == pytest.approx(expected, rel=0.05)


# --------------------------------------------------------------------------- #
# Bootstrap field subset (through the real pipeline)                           #
# --------------------------------------------------------------------------- #
def _write_marked_records(tmp_path, n=40):
    """n LR/HR pairs whose pixel values encode their index (LR = index)."""
    size = Config.DEFAULT_HR_CROP_SIZE
    lr_imgs, hr_imgs = [], []
    for i in range(n):
        lr = np.full((size // 2, size // 2, 4), float(i), np.float32)
        hr = np.full((size, size, 4), float(i), np.float32)
        lr_imgs.append(Image(data=lr, pixel_scale_arcsec=0.10,
                             band_names=Config.LR_INPUT_BAND_NAMES,
                             is_clean=False, index=i))
        hr_imgs.append(Image(data=hr, pixel_scale_arcsec=0.05,
                             band_names=Config.HR_TARGET_BAND_NAMES,
                             is_clean=True, index=i))
    d = str(tmp_path)
    write_images(lr_imgs, "dirty_train", records_dir=d)
    write_images(hr_imgs, "clean_train", records_dir=d)
    return (tfrecord_path(d, "dirty_train"), tfrecord_path(d, "clean_train"))


def _seen_indices(model, lr_path, hr_path, *, seed, n, keep=0.5):
    ds = model._build_training_pipeline(
        lr_path, hr_path, batch_size=1,
        bootstrap_keep=keep, bootstrap_seed=seed)
    # index survives the constant field: asinh(i/k) → sinh⁻¹ back via set
    # membership on the stretched VIS value of each of n candidates.
    from euclid_polish.training.augmentation import asinh_stretch_lr
    stretched = {round(float(asinh_stretch_lr(
        tf.constant(np.full((1, 1, 4), float(i), np.float32)))[0, 0, 0]), 5): i
        for i in range(n)}
    seen = set()
    for lr, _hr in ds.take(3 * n):
        seen.add(stretched[round(float(lr[0, 0, 0, 0]), 5)])
    return seen


@pytest.fixture(scope="module")
def _boot_model(tmp_path_factory):
    return Model(str(tmp_path_factory.mktemp("ckpt")), scale=2,
                 num_res_blocks=1)


def test_bootstrap_keeps_stable_fraction(tmp_path, _boot_model):
    n = 40
    lr_path, hr_path = _write_marked_records(tmp_path, n)
    seen = _seen_indices(_boot_model, lr_path, hr_path, seed=7, n=n)
    assert 0.25 * n <= len(seen) <= 0.75 * n          # ≈ keep=0.5
    # deterministic: the same seed re-selects exactly the same subset
    again = _seen_indices(_boot_model, lr_path, hr_path, seed=7, n=n)
    assert seen == again


def test_bootstrap_seed_changes_subset(tmp_path, _boot_model):
    n = 40
    lr_path, hr_path = _write_marked_records(tmp_path, n)
    a = _seen_indices(_boot_model, lr_path, hr_path, seed=7, n=n)
    b = _seen_indices(_boot_model, lr_path, hr_path, seed=8, n=n)
    assert a != b


def test_bootstrap_off_sees_every_field(tmp_path, _boot_model):
    n = 12
    lr_path, hr_path = _write_marked_records(tmp_path, n)
    seen = _seen_indices(_boot_model, lr_path, hr_path, seed=7, n=n, keep=None)
    assert seen == set(range(n))


# --------------------------------------------------------------------------- #
# CLI plumbing: --loss/--noise-aug/--bootstrap + positional --member-spec      #
# --------------------------------------------------------------------------- #
def test_build_specs_run_wide_diversity_flags(tmp_path):
    args = parse_args(["--mode", "add", "--count", "2", "--steps", "10",
                       "--loss", "l2", "--noise-aug", "0.5",
                       "--bootstrap", "0.7"])
    specs = build_specs(args, str(tmp_path / "ens"))
    assert all(s.loss_norm == "l2" and s.noise_aug == 0.5
               and s.bootstrap == 0.7 for s in specs)


def test_build_specs_member_spec_overrides_positionally(tmp_path):
    args = parse_args([
        "--mode", "add", "--count", "3", "--steps", "10", "--base-seed", "5",
        "--loss", "l1", "--num-res-blocks", "32",
        "--member-spec",
        '[{"loss":"l2","noise_aug":1.0},'
        ' {"bootstrap":0.6,"num_res_blocks":16,"seed":99}]'])
    s0, s1, s2 = build_specs(args, str(tmp_path / "ens"))
    assert (s0.loss_norm, s0.noise_aug, s0.bootstrap) == ("l2", 1.0, None)
    assert (s1.loss_norm, s1.bootstrap, s1.num_res_blocks, s1.seed) == \
        ("l1", 0.6, 16, 99)
    assert (s2.loss_norm, s2.noise_aug, s2.bootstrap,
            s2.num_res_blocks, s2.seed) == ("l1", 0.0, None, 32, 7)


def test_member_spec_rejects_unknown_keys(tmp_path):
    args = parse_args(["--count", "1", "--steps", "10",
                       "--member-spec", '[{"lr_peak": 0.1}]'])
    with pytest.raises(SystemExit):
        build_specs(args, str(tmp_path / "ens"))


def test_member_spec_rejects_bad_json(tmp_path):
    args = parse_args(["--count", "1", "--steps", "10",
                       "--member-spec", "not json"])
    with pytest.raises(SystemExit):
        build_specs(args, str(tmp_path / "ens"))


def test_member_spec_rejects_bad_loss(tmp_path):
    args = parse_args(["--count", "1", "--steps", "10",
                       "--member-spec", '[{"loss":"l9"}]'])
    with pytest.raises(SystemExit):
        build_specs(args, str(tmp_path / "ens"))


def test_ensemble_train_step_forwards_diversity_flags(monkeypatch):
    from euclid_polish.web.fasrc_pipeline import EnsembleTrainStep
    monkeypatch.setattr(
        "euclid_polish.web.fasrc_pipeline.next_member_names",
        lambda base, k: [f"member_{i:02d}" for i in range(k)])
    cmd = EnsembleTrainStep().build_command({
        "mode": "add", "count": "2", "steps": "1000",
        "loss": "l3", "noise_aug": "1.0", "bootstrap": "0.7",
        "member_spec": '[{"loss": "l1"}]'})
    joined = " ".join(cmd)
    assert "--loss l3" in joined
    assert "--noise-aug 1" in joined
    assert "--bootstrap 0.7" in joined
    assert "--member-spec" in cmd
    assert '"l1"' in cmd[cmd.index("--member-spec") + 1]


def test_ensemble_train_step_rejects_bad_member_spec(monkeypatch):
    from euclid_polish.web.fasrc_pipeline import EnsembleTrainStep
    monkeypatch.setattr(
        "euclid_polish.web.fasrc_pipeline.next_member_names",
        lambda base, k: ["member_00"])
    with pytest.raises(ValueError, match="not valid JSON"):
        EnsembleTrainStep().build_command({
            "mode": "add", "count": "1", "steps": "1000",
            "member_spec": "{oops"})
