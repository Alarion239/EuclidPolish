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
from euclid_polish.training.losses import berhu_loss, build_loss, lp_loss
from euclid_polish.training.models.common import ICNR
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
# BerHu (reverse-Huber) loss                                                   #
# --------------------------------------------------------------------------- #
def test_berhu_l1_below_threshold_l2_above():
    """L1 on residuals ≤ δ, quadratic on the outlier above δ. With c=0.2 the
    batch max sets δ = 0.2·max, so the outlier is amplified while the small
    residuals stay linear. Hand-computed: residuals [0.1,0.1,0.1,1.0] →
    δ=0.2; the three 0.1s contribute 0.1 each, the 1.0 → (1+0.04)/0.4 = 2.6;
    mean = (0.3 + 2.6)/4 = 0.725."""
    a = tf.constant([0.1, 0.1, 0.1, 1.0])
    b = tf.zeros_like(a)
    assert float(berhu_loss(c=0.2)(a, b)) == pytest.approx(0.725, rel=1e-6)


def test_berhu_amplifies_outliers_beyond_mae():
    """The whole point vs L1: a bright-peak residual costs more than its
    absolute value (quadratic branch), so BerHu > MAE on a peaked batch."""
    a = tf.constant([0.1, 0.1, 0.1, 1.0])
    b = tf.zeros_like(a)
    mae = float(lp_loss("l1")(a, b))
    assert float(berhu_loss(c=0.2)(a, b)) > mae


def test_berhu_collapses_to_mae_when_threshold_covers_batch():
    """δ ≥ max|residual| ⇒ every residual is in the L1 branch ⇒ exactly MAE.
    c=1.0 puts δ at the batch max, so the loss equals the mean abs error."""
    a = tf.constant([0.1, 0.1, 0.1, 1.0])
    b = tf.zeros_like(a)
    assert float(berhu_loss(c=1.0)(a, b)) == pytest.approx(
        float(lp_loss("l1")(a, b)), rel=1e-6)


def test_berhu_is_continuous_and_finite_on_perfect_batch():
    """All-zero residuals (δ→0 guarded) must not divide by zero."""
    a = tf.zeros([4, 4])
    assert float(berhu_loss()(a, a)) == pytest.approx(0.0, abs=1e-6)


def test_berhu_gradient_flows():
    a = tf.Variable(tf.constant([0.1, 0.5, 2.0]))
    b = tf.zeros(3)
    with tf.GradientTape() as tape:
        loss = berhu_loss(c=0.2)(a, b)
    g = tape.gradient(loss, a)
    assert g is not None and bool(tf.reduce_all(tf.math.is_finite(g)))


def test_build_loss_dispatches_berhu_and_pnorms():
    a = tf.constant([0.1, 0.1, 0.1, 1.0])
    b = tf.zeros_like(a)
    assert float(build_loss("berhu")(a, b)) == pytest.approx(
        float(berhu_loss()(a, b)))
    assert float(build_loss("l2")(a, b)) == pytest.approx(
        float(lp_loss("l2")(a, b)))


def test_build_loss_rejects_unknown():
    with pytest.raises(ValueError, match="unknown loss"):
        build_loss("l7")


# --------------------------------------------------------------------------- #
# ICNR sub-pixel init (checkerboard-free upsampler)                            #
# --------------------------------------------------------------------------- #
def test_icnr_upsampler_is_nearest_neighbor_at_init():
    """The whole point: a conv ICNR-initialised then pixel-shuffled must emit
    CONSTANT scale×scale output blocks at init (an exact nearest-neighbour
    resize) — i.e. zero checkerboard. This also pins the tile-vs-repeat channel
    layout: get it wrong and the sub-pixels of a block differ."""
    from tf_keras.initializers import GlorotUniform
    from tf_keras.layers import Conv2D

    scale, c_out = 2, 3
    conv = Conv2D(c_out * scale ** 2, 3, padding="same",
                  kernel_initializer=ICNR(scale, GlorotUniform(seed=0)))
    x = tf.random.stateless_normal([1, 6, 6, 4], seed=[1, 2])
    y = tf.nn.depth_to_space(conv(x), scale)               # [1, 12, 12, c_out]
    # variance WITHIN each 2×2 output block (across the sub-pixel offsets):
    blocks = tf.reshape(y, [1, 6, scale, 6, scale, c_out])
    within = tf.math.reduce_variance(blocks, axis=[2, 4])
    assert float(tf.reduce_max(within)) < 1e-10


def test_icnr_rejects_indivisible_channels():
    with pytest.raises(ValueError, match="not divisible"):
        ICNR(2)([3, 3, 4, 5])                              # 5 not divisible by 4


def test_icnr_survives_weightnorm():
    """Production wraps the sub-pixel conv in tfp WeightNorm (data_init=False).
    WeightNorm reparametrises W = g·v/‖v‖ with v = kernel, g = ‖kernel‖, so the
    EFFECTIVE weights at init still equal the ICNR kernel — the nearest-neighbour
    (checkerboard-free) property must hold through the wrapper too."""
    from euclid_polish.training.models.common import ICNR
    from euclid_polish.training.models.wdsr import conv2d_weightnorm

    scale, c_out = 2, 2
    conv = conv2d_weightnorm(c_out * scale ** 2, 3, padding="same",
                             kernel_initializer=ICNR(scale))
    x = tf.random.stateless_normal([1, 5, 5, 3], seed=[3, 4])
    y = tf.nn.depth_to_space(conv(x), scale)
    blocks = tf.reshape(y, [1, 5, scale, 5, scale, c_out])
    within = tf.math.reduce_variance(blocks, axis=[2, 4])
    assert float(tf.reduce_max(within)) < 1e-8


def test_wdsr_icnr_builds_and_upsamples():
    """The flag threads through the real network: builds, and the output is
    the scale-upsampled shape (4-band config)."""
    from euclid_polish.training.models.wdsr import wdsr
    model = wdsr(scale=2, num_res_blocks=1, nchan_in=4, nchan_out=4, icnr=True)
    y = model(tf.zeros([1, 8, 8, 4]))
    assert y.shape.as_list() == [1, 16, 16, 4]


def test_build_specs_icnr_flag_and_default(tmp_path):
    on = parse_args(["--count", "2", "--steps", "10", "--icnr"])
    assert all(s.icnr for s in build_specs(on, str(tmp_path / "a")))
    off = parse_args(["--count", "1", "--steps", "10"])
    assert build_specs(off, str(tmp_path / "b"))[0].icnr is False


def test_member_spec_icnr_override(tmp_path):
    args = parse_args(["--count", "2", "--steps", "10",
                       "--member-spec", '[{"icnr": true}, {}]'])
    specs = build_specs(args, str(tmp_path / "ens"))
    assert specs[0].icnr is True and specs[1].icnr is False


# --------------------------------------------------------------------------- #
# Star regime tag (starless / starfull)                                        #
# --------------------------------------------------------------------------- #
def test_member_is_starless_reads_origin(tmp_path):
    from euclid_polish.ensemble import member_is_starless
    d = tmp_path / "member_00"
    d.mkdir()
    assert member_is_starless(str(d)) is False          # no origin → starfull
    (d / "origin.json").write_text('{"starless": true}')
    assert member_is_starless(str(d)) is True
    (d / "origin.json").write_text('{"starless": false}')
    assert member_is_starless(str(d)) is False


def test_build_specs_starless_default_and_flag(tmp_path):
    on = parse_args(["--count", "2", "--steps", "10"])   # default --starless 1
    assert all(s.starless for s in build_specs(on, str(tmp_path / "a")))
    off = parse_args(["--count", "1", "--steps", "10", "--starless", "0"])
    assert build_specs(off, str(tmp_path / "b"))[0].starless is False


def test_member_spec_starless_override(tmp_path):
    args = parse_args(["--count", "2", "--steps", "10",
                       "--member-spec", '[{"starless": false}, {}]'])
    specs = build_specs(args, str(tmp_path / "ens"))
    assert specs[0].starless is False and specs[1].starless is True


# --------------------------------------------------------------------------- #
# Plateau LR guard is L1-only (degenerate-basin escape)                        #
# --------------------------------------------------------------------------- #
def test_plateau_guard_applies_l1_only():
    from euclid_polish.training.loss_names import plateau_guard_applies
    assert plateau_guard_applies("l1") is True
    assert plateau_guard_applies("L1") is True                # case-insensitive
    for n in ("l2", "l3", "berhu"):
        assert plateau_guard_applies(n) is False


def _write_train_val_records(tmp_path, n=4):
    """Minimal 4-band train+validate record pairs so ``Model.train`` can build
    both pipelines (it derives the validate path from the train path)."""
    size = Config.DEFAULT_HR_CROP_SIZE
    d = str(tmp_path)
    for split in ("train", "validate"):
        lrs, hrs = [], []
        for i in range(n):
            lrs.append(Image(
                data=np.full((size // 2, size // 2, 4), float(i), np.float32),
                pixel_scale_arcsec=0.10, band_names=Config.LR_INPUT_BAND_NAMES,
                is_clean=False, index=i))
            hrs.append(Image(
                data=np.full((size, size, 4), float(i), np.float32),
                pixel_scale_arcsec=0.05, band_names=Config.HR_TARGET_BAND_NAMES,
                is_clean=True, index=i))
        write_images(lrs, f"dirty_{split}", records_dir=d)
        write_images(hrs, f"clean_{split}", records_dir=d)
    return tfrecord_path(d, "dirty_train"), tfrecord_path(d, "clean_train")


class _CaptureTrainer:
    """Stand-in Trainer that records the kwargs it was built with; its train()
    is a no-op (writes no checkpoint, so Model.train's reload tail is skipped)."""
    last: dict = {}

    def __init__(self, *args, **kwargs):
        _CaptureTrainer.last = kwargs

    def train(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("loss,guard_on",
                         [("l1", True), ("l2", False),
                          ("l3", False), ("berhu", False)])
def test_train_gates_plateau_guard_by_loss(tmp_path, monkeypatch, loss, guard_on):
    """model.train forces the plateau guard off for the large-residual losses
    even when plateau_lr_enabled=True — it is meaningful only for L1."""
    import euclid_polish.model as model_mod
    monkeypatch.setattr(model_mod, "Trainer", _CaptureTrainer)
    lr, hr = _write_train_val_records(tmp_path)
    m = Model(str(tmp_path / "ckpt"), scale=2, num_res_blocks=1)
    m.train(lr, hr, steps=1, batch_size=1, loss_norm=loss,
            plateau_lr_enabled=True)
    assert _CaptureTrainer.last["plateau_lr_enabled"] is guard_on


def test_train_respects_explicit_plateau_off_for_l1(tmp_path, monkeypatch):
    """The loss gate only ever DISABLES; an explicit off on L1 stays off."""
    import euclid_polish.model as model_mod
    monkeypatch.setattr(model_mod, "Trainer", _CaptureTrainer)
    lr, hr = _write_train_val_records(tmp_path)
    m = Model(str(tmp_path / "ckpt"), scale=2, num_res_blocks=1)
    m.train(lr, hr, steps=1, batch_size=1, loss_norm="l1",
            plateau_lr_enabled=False)
    assert _CaptureTrainer.last["plateau_lr_enabled"] is False


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


def test_member_spec_rejects_deprecated_berhu(tmp_path):
    """BerHu is deprecated — no longer SELECTABLE for a new member (though
    build_loss still dispatches it so existing members load/continue)."""
    args = parse_args(["--count", "1", "--steps", "10",
                       "--member-spec", '[{"loss":"berhu"}]'])
    with pytest.raises(SystemExit):
        build_specs(args, str(tmp_path / "ens"))


def test_build_loss_still_dispatches_deprecated_berhu():
    """Back-compat: the loss stays constructible for the existing members."""
    a = tf.constant([0.1, 0.1, 0.1, 1.0])
    b = tf.zeros_like(a)
    assert float(build_loss("berhu")(a, b)) == pytest.approx(
        float(berhu_loss()(a, b)))


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


def test_ensemble_train_step_forwards_icnr(monkeypatch):
    from euclid_polish.web.fasrc_pipeline import EnsembleTrainStep
    monkeypatch.setattr(
        "euclid_polish.web.fasrc_pipeline.next_member_names",
        lambda base, k: [f"member_{i:02d}" for i in range(k)])
    cmd = EnsembleTrainStep().build_command({
        "mode": "add", "count": "1", "steps": "1000", "icnr": "1"})
    assert "--icnr" in cmd
    off = EnsembleTrainStep().build_command({
        "mode": "add", "count": "1", "steps": "1000"})
    assert "--icnr" not in off


def test_ensemble_train_step_rejects_bad_member_spec(monkeypatch):
    from euclid_polish.web.fasrc_pipeline import EnsembleTrainStep
    monkeypatch.setattr(
        "euclid_polish.web.fasrc_pipeline.next_member_names",
        lambda base, k: ["member_00"])
    with pytest.raises(ValueError, match="not valid JSON"):
        EnsembleTrainStep().build_command({
            "mode": "add", "count": "1", "steps": "1000",
            "member_spec": "{oops"})
