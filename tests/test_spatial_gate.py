"""Spatial gating combiner: NumPy/TF parity, convexity, persistence, fitting."""

import json
import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel
from euclid_polish.eval import spatial_gate_compare as sgc
from euclid_polish.eval.combiner import (
    ACTIVE_COMBINER_KINDS,
    COMBINER_MODELS,
    RawIncrementalMinMeanMaxRBFCombiner,
    combiner_artifact_fingerprint,
    load_combiner,
    normalize_model_kind,
    save_combiner,
)
from euclid_polish.eval.ensemble_cube_cache import (
    load_cached_field_lr,
    save_cached_field_lr,
)
from euclid_polish.eval.spatial_gate import (
    BAND_NAMES,
    MIX_ASINH,
    MIX_LINEAR,
    SPATIAL_GATE_KIND,
    SpatialGateCombiner,
    band_scales,
    gate_logits,
    joined_after_fit,
    load_spatial_gate,
    lr_features,
    member_features,
    reads_available,
    restrict_to_available,
    save_spatial_gate,
    space_to_depth,
    upsample2x,
)
from euclid_polish.eval.spatial_gate_fit import (
    ALL_KNEE_LOSS,
    GateField,
    fit_spatial_gate,
    init_params,
    split_holdout,
    stamp_blackouts,
    tf_gate_logits,
    tf_mix,
)
from euclid_polish.web.helpers import ensemble_viz

N_BANDS = len(BAND_NAMES)


def _random_params(n_members, width, use_lr, seed=3):
    params = init_params(n_members, N_BANDS, width, use_lr, [0] * N_BANDS, seed)
    rng = np.random.default_rng(seed)
    params["w_out"] = rng.normal(0, 0.3, params["w_out"].shape).astype(np.float32)
    for name in params:
        if name.startswith("b_"):
            params[name] = rng.normal(0, 0.2, params[name].shape).astype(np.float32)
    return params


def _members_and_lr(n_members=3, h=32, w=48, seed=5):
    rng = np.random.default_rng(seed)
    lr = rng.exponential(200.0, (h // 2, w // 2, N_BANDS)).astype(np.float32)
    lr[3:6, 4:9] = 0.0
    members = np.repeat(np.kron(lr, np.ones((2, 2, 1)))[None] / 4.0, n_members, 0)
    members = members + rng.normal(0, 30.0, members.shape)
    return members.astype(np.float32), lr


def test_space_to_depth_and_upsample_match_tensorflow():
    x = np.random.default_rng(0).normal(size=(6, 8, 3)).astype(np.float32)
    np.testing.assert_allclose(space_to_depth(x),
                               tf.nn.space_to_depth(x[None], 2)[0].numpy())
    up = upsample2x(x)
    assert up.shape == (12, 16, 3)
    np.testing.assert_allclose(up[::2, ::2].mean(), x.mean(), atol=0.2)
    constant = np.full((3, 4, 1), 2.5, np.float32)
    np.testing.assert_allclose(upsample2x(constant), 2.5)


@pytest.mark.parametrize("use_lr, mix_space",
                         [(True, MIX_ASINH), (False, MIX_ASINH), (True, MIX_LINEAR)])
def test_numpy_forward_matches_tensorflow_graph(use_lr, mix_space):
    members, lr = _members_and_lr()
    params = _random_params(len(members), 8, use_lr)
    scales = band_scales()
    x = member_features(members, scales)
    lf = lr_features(lr, scales)
    ours = gate_logits(params, x, lf if use_lr else None)
    tf_params = {k: tf.constant(v) for k, v in params.items()}
    theirs = tf_gate_logits(tf_params, tf.constant(x[None]),
                            tf.constant(lf[None]), use_lr)[0].numpy()
    np.testing.assert_allclose(ours, theirs, rtol=1e-4, atol=1e-4)

    comb = SpatialGateCombiner([f"m{i}" for i in range(len(members))], params,
                               width=8, use_lr=use_lr, mix_space=mix_space)
    mixed = tf_mix(tf.constant(theirs[None]), tf.constant(x[None]),
                   len(members), N_BANDS, mix_space)[0].numpy()
    out = comb.apply_field(members, lr=lr)
    np.testing.assert_allclose(np.arcsinh(out / scales), mixed, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("mix_space", [MIX_ASINH, MIX_LINEAR])
def test_output_is_convex_in_members_and_handles_odd_shapes(mix_space):
    members, lr = _members_and_lr(n_members=4)
    members = members[:, :31, :47]
    comb = SpatialGateCombiner([f"m{i}" for i in range(4)],
                               _random_params(4, 8, True), width=8, use_lr=True,
                               mix_space=mix_space)
    out = comb.apply_field(members, lr=lr)
    assert out.shape == (31, 47, N_BANDS)
    lo, hi = members.min(0), members.max(0)
    assert np.all(out >= lo - 1e-3 * np.abs(lo) - 1e-3)
    assert np.all(out <= hi + 1e-3 * np.abs(hi) + 1e-3)
    weights = comb.weights_field(members, lr=lr)
    np.testing.assert_allclose(weights.sum(axis=2), 1.0, atol=1e-5)
    if mix_space == MIX_LINEAR:
        # Averaging in electrons: the output is exactly the weighted flux.
        np.testing.assert_allclose(out, np.einsum("hwmc,mhwc->hwc", weights, members),
                                   rtol=1e-5, atol=1e-3)


def test_linear_and_asinh_mixing_differ_only_where_members_disagree():
    members, lr = _members_and_lr(n_members=3)
    params = _random_params(3, 8, False)
    linear = SpatialGateCombiner(["a", "b", "c"], params, width=8, use_lr=False,
                                 mix_space=MIX_LINEAR).apply_field(members)
    asinh = SpatialGateCombiner(["a", "b", "c"], params, width=8,
                                use_lr=False).apply_field(members)
    assert not np.allclose(linear, asinh)
    agree = np.repeat(members[:1], 3, axis=0)
    np.testing.assert_allclose(
        SpatialGateCombiner(["a", "b", "c"], params, width=8, use_lr=False,
                            mix_space=MIX_LINEAR).apply_field(agree),
        agree[0], rtol=1e-5, atol=1e-3)
    with pytest.raises(ValueError):
        SpatialGateCombiner(["a"], params, width=8, use_lr=False, mix_space="log")


def test_initial_gate_reproduces_each_bands_best_member():
    members, lr = _members_and_lr(n_members=3)
    best = [2, 0, 1, 2]
    params = init_params(3, N_BANDS, 8, True, best, seed=0)
    comb = SpatialGateCombiner(["a", "b", "c"], params, width=8, use_lr=True)
    weights = comb.weights_field(members, lr=lr)
    for band, member in enumerate(best):
        np.testing.assert_allclose(weights[..., member, band], 0.9, atol=1e-5)


def test_lr_gate_requires_lr_and_marks_blackouts():
    members, lr = _members_and_lr()
    comb = SpatialGateCombiner(["a", "b", "c"], _random_params(3, 8, True),
                               width=8, use_lr=True)
    with pytest.raises(ValueError, match="lr="):
        comb.apply_field(members)
    mask = lr_features(lr, band_scales())[..., -1]
    assert mask[3:6, 4:9].all() and not mask[12:, 16:].any()


def test_save_load_roundtrip_and_staleness(tmp_path):
    members, lr = _members_and_lr()
    comb = SpatialGateCombiner(["a", "b", "c"], _random_params(3, 8, False),
                               width=8, use_lr=False, records_fp="fp",
                               fit_meta={"note": 1}, mix_space=MIX_LINEAR)
    save_spatial_gate(comb, str(tmp_path))
    loaded = load_spatial_gate(str(tmp_path), member_labels=["a", "b", "c"])
    assert loaded is not None and loaded.records_fp == "fp"
    assert loaded.mix_space == MIX_LINEAR
    np.testing.assert_array_equal(loaded.apply_field(members), comb.apply_field(members))
    # Gates saved before the mixing space was recorded mixed in asinh.
    manifest_path = tmp_path / "combiner.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["mix_space"]
    manifest_path.write_text(json.dumps(manifest))
    assert load_spatial_gate(str(tmp_path)).mix_space == MIX_ASINH
    assert load_spatial_gate(str(tmp_path), member_labels=["a", "b"]) is None
    assert load_spatial_gate(str(tmp_path / "missing")) is None


def test_stamp_blackouts_zeroes_bright_sources_only():
    lr = np.full((40, 40, N_BANDS), 50.0, np.float32)
    wells = np.asarray([Config.STAR_SATURATION_WELL_E[b] for b in BAND_NAMES])
    lr[20, 20] = wells * 0.5
    stamped = stamp_blackouts(lr, np.random.default_rng(0))
    assert np.all(stamped[20, 20] == 0.0)
    assert np.count_nonzero(stamped == 0.0) < 40 * N_BANDS
    assert np.all(stamped[:5, :5] == 50.0)
    np.testing.assert_array_equal(lr[20, 20], wells * 0.5)


def test_fit_learns_a_spatially_varying_member_choice(tmp_path, monkeypatch):
    """Member 0 is right on the left half, member 1 on the right half: no
    single member or global blend matches the target, a spatial gate does."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    rng = np.random.default_rng(1)
    fields = []
    for index in range(6):
        target = rng.exponential(150.0, (96, 96, N_BANDS)).astype(np.float32)
        noise = rng.normal(0, 120.0, (2, 96, 96, N_BANDS)).astype(np.float32)
        good = np.zeros((2, 96, 96, 1), np.float32)
        good[0, :, :48] = good[1, :, 48:] = 1.0
        members = target[None] + noise * (1.0 - good)
        paths = []
        for m in range(2):
            path = tmp_path / f"member{m}_{index:05d}.npy"
            np.save(path, members[m])
            paths.append(str(path))
        lr = target.reshape(48, 2, 48, 2, N_BANDS).sum(axis=(1, 3))
        fields.append(GateField(index, paths, target, lr))
    checkpoints = []
    comb = fit_spatial_gate(fields[:5], fields[5:], ["a", "b"], width=8,
                            steps=240, batch_size=4, crop=64, eval_every=60,
                            learning_rate=1e-2, warmup_steps=5, seed=0,
                            # Band-knee loss: sign-flipping additive noise makes
                            # the low-knee terms a pathological toy objective.
                            loss_knees=None, checkpoint=checkpoints.append)
    meta = comb.fit_meta
    assert meta["selected"]["loss"] < 0.6 * meta["baseline_holdout"]["loss"]
    # Every improvement hands over the best gate so far; the last one is the
    # returned gate, so a fit stopped early keeps its best checkpoint.
    assert checkpoints and all(not c.fit_meta["complete"] for c in checkpoints)
    assert [c.fit_meta["selected"]["step"] for c in checkpoints] == sorted(
        c.fit_meta["selected"]["step"] for c in checkpoints)
    for name, value in comb.params.items():
        np.testing.assert_array_equal(checkpoints[-1].params[name], value)
    assert meta["complete"] and meta["steps_run"] == 240
    weights = comb.weights_field(np.stack([np.load(p) for p in fields[5].member_paths]),
                                 lr=fields[5].lr_e)
    assert weights[20:76, 8:40, 0].mean() > 0.8
    assert weights[20:76, 56:88, 1].mean() > 0.8
    assert not list(scratch.glob("spatial_gate_features_*")), \
        "the training feature cache must be removed"


def test_combiner_registry_prefers_and_round_trips_the_spatial_gate(tmp_path):
    assert ACTIVE_COMBINER_KINDS[0] == SPATIAL_GATE_KIND
    assert normalize_model_kind("spatial_gate") == SPATIAL_GATE_KIND
    members, lr = _members_and_lr()
    comb = SpatialGateCombiner(["a", "b", "c"], _random_params(3, 8, True),
                               width=8, use_lr=True)
    spec = COMBINER_MODELS[SPATIAL_GATE_KIND]
    save_combiner(comb, str(tmp_path))
    loaded = load_combiner(str(tmp_path), member_labels=["a", "b", "c"],
                           artifact_dir=spec.artifact_dir)
    assert isinstance(loaded, SpatialGateCombiner)
    np.testing.assert_array_equal(loaded.apply_field(members, lr=lr),
                                  comb.apply_field(members, lr=lr))
    assert load_combiner(str(tmp_path), member_labels=["a", "b"],
                         artifact_dir=spec.artifact_dir) is None


def test_rbf_combiner_accepts_and_ignores_lr():
    members, lr = _members_and_lr()
    m, c = len(members), N_BANDS
    rbf = RawIncrementalMinMeanMaxRBFCombiner(
        member_labels=["a", "b", "c"], n_kernels=0,
        coefficients=np.zeros((0, m), np.float32),
        centers=np.zeros((0, m * c), np.float32),
        scales=np.ones(m * c, np.float32), sigmas=np.zeros(0, np.float32),
        increment_ids=np.zeros(0, np.int32),
        reference_features=np.zeros(m * c, np.float32),
        output_floors=np.zeros(c, np.float32),
        global_logits=np.array([0.0, 1.0, -1.0], np.float32))
    np.testing.assert_array_equal(rbf.apply_field(members, lr=lr),
                                  rbf.apply_field(members))


def test_cached_field_lr_reads_the_bucket_copy(tmp_path):
    lr = np.arange(2 * 3 * N_BANDS, dtype=np.float32).reshape(2, 3, N_BANDS)
    assert load_cached_field_lr(str(tmp_path), 7, records_dir=None, subset="test") is None
    save_cached_field_lr(str(tmp_path), 7, lr)
    np.testing.assert_array_equal(
        load_cached_field_lr(str(tmp_path), 7, records_dir=None, subset="test"), lr)


def test_split_holdout_is_deterministic_and_disjoint():
    fields = [GateField(i, [], np.zeros((2, 2, N_BANDS)), np.zeros((1, 1, N_BANDS)))
              for i in range(20)]
    train, held = split_holdout(fields, 3, seed=0)
    again, held_again = split_holdout(fields, 3, seed=0)
    assert [f.index for f in held] == [f.index for f in held_again]
    assert len(held) == 3 and len(train) == 17
    assert not {f.index for f in train} & {f.index for f in held}


def test_spatial_gate_payload_reports_member_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path))
    monkeypatch.setattr(ensemble_viz, "_member_meta_from_labels",
                        lambda labels: [{} for _ in labels])
    monkeypatch.setattr(ensemble_viz, "_sky_records_local_dir", lambda: None)
    labels = ["a", "b", "c"]
    members, lr = _members_and_lr()
    val_dir = ensemble_viz._ensemble_cubes_dir("validate", starless=False)
    os.makedirs(val_dir, exist_ok=True)
    for rec in (0, 1):
        for i, member in enumerate(members):
            np.save(os.path.join(val_dir, f"member{i}_{rec:05d}.npy"), member)
        save_cached_field_lr(val_dir, rec, lr)
    with open(os.path.join(val_dir, "viz_index.json"), "w") as handle:
        json.dump({"subset": "validate", "indices": [0, 1], "member_labels": labels}, handle)
    comb = SpatialGateCombiner(labels, _random_params(3, 8, True), width=8,
                               use_lr=True, fit_meta={"holdout_fields": [1]})
    save_combiner(comb, ensemble_viz._ensemble_regime_dir(False))

    payload = ensemble_viz.compute_combiner_payload(False, model_kind=SPATIAL_GATE_KIND)
    diag = payload["gate_diagnostics"]
    assert payload["kind"] == SPATIAL_GATE_KIND and diag["available"]
    assert diag["n_fields"] == 1
    for band in BAND_NAMES:
        assert sum(diag["usage"][band]) == pytest.approx(1.0, abs=1e-5)
    cached = ensemble_viz.compute_combiner_payload(False, model_kind=SPATIAL_GATE_KIND)
    assert cached["gate_diagnostics"]["artifact_fp"] == diag["artifact_fp"]


def test_pruned_gate_reads_only_its_active_members(tmp_path):
    members, lr = _members_and_lr(n_members=4)
    comb = SpatialGateCombiner(["a", "b", "c", "d"], _random_params(2, 8, True),
                               width=8, use_lr=True, active_members=(1, 3))
    full = comb.apply_field(members, lr=lr)
    np.testing.assert_array_equal(full, comb.apply_field(members[[1, 3]], lr=lr))
    weights = comb.weights_field(members, lr=lr)
    assert weights.shape[2] == 4 and not weights[:, :, [0, 2]].any()
    np.testing.assert_allclose(weights.sum(axis=2), 1.0, atol=1e-5)
    assert comb.needed_member_indices() == [1, 3]
    assert comb.surviving_members()["source"] == [False, True, False, True]

    shrunk = comb.without_member(0)
    assert shrunk.member_labels == ["b", "c", "d"] and shrunk.active == [0, 2]
    np.testing.assert_array_equal(shrunk.apply_field(members[1:], lr=lr), full)
    with pytest.raises(ValueError, match="refitted"):
        comb.without_member(1)

    save_spatial_gate(comb, str(tmp_path))
    loaded = load_spatial_gate(str(tmp_path), member_labels=["a", "b", "c", "d"])
    assert loaded is not None and loaded.active_members == (1, 3)
    np.testing.assert_array_equal(loaded.apply_field(members, lr=lr), full)


def test_gate_validity_keys_on_the_members_it_reads(tmp_path):
    """add / archive-unread / archive-read: a (pruned) gate loads for an
    available membership while every member it READS is in it."""
    comb = SpatialGateCombiner(["a", "b", "c", "d"], _random_params(2, 8, False),
                               width=8, use_lr=False, active_members=(1, 3))
    assert comb.read_labels == ["b", "d"]
    save_spatial_gate(comb, str(tmp_path))
    fitted = ["a", "b", "c", "d"]
    for available in (fitted, [*fitted, "e", "f"], ["b", "d"], ["d", "x", "b"]):
        loaded = load_spatial_gate(str(tmp_path), available_labels=available)
        assert loaded is not None and loaded.member_labels == fitted, available
    assert load_spatial_gate(str(tmp_path), available_labels=["a", "b", "c"]) is None
    assert load_spatial_gate(str(tmp_path), available_labels=[]) is None
    # the exact-list check the fit and its caches use is unchanged
    assert load_spatial_gate(str(tmp_path), member_labels=[*fitted, "e"]) is None
    assert reads_available(["b", "d"], ["d", "b", "z"]) and not reads_available([], ["a"])
    assert joined_after_fit(fitted, ["a", "e", "b", "f"]) == ["e", "f"]

    unpruned = SpatialGateCombiner(["a", "b"], _random_params(2, 8, False),
                                   width=8, use_lr=False)
    prod = os.path.join(str(tmp_path), "regime")
    save_combiner(unpruned, prod)
    artifact = COMBINER_MODELS[SPATIAL_GATE_KIND].artifact_dir
    assert load_combiner(prod, artifact_dir=artifact,
                         available_labels=["a", "b", "c"]) is not None   # c joined
    assert load_combiner(prod, artifact_dir=artifact, available_labels=["a"]) is None


def test_payload_reports_peak_weights_and_the_used_rule(tmp_path, monkeypatch):
    """The combiner payload carries each member's peak weight (max over
    bands and brightness bins) and whether the gate uses it (>= 0.5 %); a
    validate cache that gained a member still serves the diagnostic."""
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path))
    monkeypatch.setattr(ensemble_viz, "_member_meta_from_labels",
                        lambda labels: [{} for _ in labels])
    monkeypatch.setattr(ensemble_viz, "_sky_records_local_dir", lambda: None)
    monkeypatch.setattr(ensemble_viz, "_regime_labels",
                        lambda base, starless: ["a", "b", "c", "d", "e"])
    members, lr = _members_and_lr(n_members=5)
    val_dir = ensemble_viz._ensemble_cubes_dir("validate", starless=False)
    os.makedirs(val_dir, exist_ok=True)
    for i, member in enumerate(members):
        np.save(os.path.join(val_dir, f"member{i}_00000.npy"), member)
    with open(os.path.join(val_dir, "viz_index.json"), "w") as handle:
        json.dump({"subset": "validate", "indices": [0],
                   "member_labels": ["a", "b", "c", "d", "e"]}, handle)
    comb = SpatialGateCombiner(["a", "b", "c", "d"], _random_params(2, 8, False),
                               width=8, use_lr=False, active_members=(0, 2))
    save_combiner(comb, ensemble_viz._ensemble_regime_dir(False))
    payload = ensemble_viz.compute_combiner_payload(False, model_kind=SPATIAL_GATE_KIND)
    assert payload["gate_diagnostics"]["available"]
    assert payload["stale"] is False and payload["joined_after_fit"] == ["e"]
    assert payload["read_labels"] == ["a", "c"]
    peaks = payload["member_peak_weights"]
    assert len(peaks) == 4 and peaks[1] == 0.0 and peaks[3] == 0.0   # never read
    assert payload["used_by_gate"] == [p >= 0.005 for p in peaks]
    assert payload["used_by_gate"][1] is False and payload["used_by_gate"][3] is False
    assert payload["used_threshold"] == 0.005
    monkeypatch.setattr(ensemble_viz, "_regime_labels", lambda base, starless: ["a", "b"])
    assert ensemble_viz.compute_combiner_payload(
        False, model_kind=SPATIAL_GATE_KIND)["stale"] is True        # c (read) left


def test_archiving_an_unread_member_keeps_a_gate_fitted_before_members_joined(
        tmp_path, monkeypatch):
    regime = str(tmp_path / "starfull")
    comb = SpatialGateCombiner(["a·psnr", "b·psnr", "c·psnr"], _random_params(2, 8, False),
                               width=8, use_lr=False, active_members=(0, 2))
    comb.fit_meta["complete"] = True
    save_combiner(comb, regime)
    artifact = COMBINER_MODELS[SPATIAL_GATE_KIND].artifact_dir
    fp = combiner_artifact_fingerprint(regime, artifact)
    monkeypatch.setattr(ensemble_viz, "_combiner_payload_path",
                        lambda starless, kind=None: str(tmp_path / "payload.json"))
    # b left (unread); d joined after the fit → the gate is kept BYTE-IDENTICAL
    # (its fingerprint, and so every output it made, stays current).
    monkeypatch.setattr(ensemble_viz, "_regime_labels",
                        lambda base, starless: ["a·psnr", "c·psnr", "d·psnr"])
    assert ensemble_viz._reconcile_combiner_on_archives(
        regime, False, ["b"], SPATIAL_GATE_KIND) is True
    kept = load_spatial_gate(os.path.join(regime, artifact))
    assert kept.member_labels == ["a·psnr", "b·psnr", "c·psnr"] and kept.active == [0, 2]
    assert combiner_artifact_fingerprint(regime, artifact) == fp
    # It still applies to the stack without b, by label, with the same output.
    restricted = restrict_to_available(kept, ["a·psnr", "c·psnr", "d·psnr"])
    assert restricted.member_labels == ["a·psnr", "c·psnr"] and restricted.active == [0, 1]
    stack = np.random.default_rng(0).exponential(50.0, (3, 16, 16, N_BANDS)).astype(np.float32)
    np.testing.assert_allclose(restricted.apply_field(stack[[0, 2]]), kept.apply_field(stack),
                               rtol=1e-5)
    assert restrict_to_available(kept, ["c·psnr", "d·psnr"]) is None      # a is read
    # Archiving a member the gate reads retires it to a promotable backup.
    monkeypatch.setattr(ensemble_viz, "_regime_labels",
                        lambda base, starless: ["c·psnr", "d·psnr"])
    assert ensemble_viz._reconcile_combiner_on_archives(
        regime, False, ["a"], SPATIAL_GATE_KIND) is False
    assert load_spatial_gate(os.path.join(regime, artifact)) is None
    backups = [d for d in os.listdir(regime) if d.startswith(ensemble_viz.GATE_BACKUP_PREFIX)]
    assert len(backups) == 1
    backed_up = load_spatial_gate(os.path.join(regime, backups[0]))
    assert backed_up.member_labels == kept.member_labels
    assert combiner_artifact_fingerprint(regime, backups[0]) == fp


def _cube_bucket(directory, labels, members, subset):
    os.makedirs(directory, exist_ok=True)
    for i, member in enumerate(members):
        np.save(os.path.join(directory, f"member{i}_00000.npy"), member)
    with open(os.path.join(directory, "viz_index.json"), "w") as handle:
        json.dump({"subset": subset, "indices": [0], "member_labels": labels}, handle)


def test_gate_applies_to_cubes_that_gained_and_lost_members(tmp_path, monkeypatch):
    """A pruned gate fitted for a, b, c (reads a, c) still scores and bakes
    the TEST cubes, and its weight diagnostic still runs on the validate
    cubes, once b (unread) was archived and d joined: by label, same output."""
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path))
    monkeypatch.setattr(ensemble_viz, "_member_meta_from_labels",
                        lambda labels: [{} for _ in labels])
    monkeypatch.setattr(ensemble_viz, "_sky_records_local_dir", lambda: None)
    monkeypatch.setattr(ensemble_viz, "_regime_labels",
                        lambda base, starless: ["a", "c", "d"])
    full, _lr = _members_and_lr(n_members=3)                 # a, b, c
    comb = SpatialGateCombiner(["a", "b", "c"], _random_params(2, 8, False),
                               width=8, use_lr=False, active_members=(0, 2))
    regime = ensemble_viz._ensemble_regime_dir(False)
    save_combiner(comb, regime)
    now = np.stack([full[0], full[2], full[1] * 0.5])        # a, c, d (joined)
    test_dir = ensemble_viz._ensemble_cubes_dir(starless=False)
    _cube_bucket(test_dir, ["a", "c", "d"], now, "test")
    _cube_bucket(ensemble_viz._ensemble_cubes_dir("validate", starless=False),
                 ["a", "c", "d"], now, "validate")

    assert ensemble_viz._apply_combiner_to_test_cubes(False, SPATIAL_GATE_KIND)
    baked = np.load(os.path.join(test_dir, f"{COMBINER_MODELS[SPATIAL_GATE_KIND].cube_prefix}"
                                           "_00000.npy"))
    np.testing.assert_allclose(baked, comb.apply_field(full), rtol=1e-5)
    with open(os.path.join(test_dir, "viz_index.json")) as handle:
        assert json.load(handle)[f"has_combiner_{SPATIAL_GATE_KIND}"] is True

    payload = ensemble_viz.compute_combiner_payload(False, model_kind=SPATIAL_GATE_KIND)
    diag = payload["gate_diagnostics"]
    assert diag["available"] and payload["stale"] is False
    for band in BAND_NAMES:
        assert len(diag["usage"][band]) == 3 and diag["usage"][band][1] == 0.0   # b unread
        assert sum(diag["usage"][band]) == pytest.approx(1.0, abs=1e-5)
    assert payload["member_peak_weights"][1] == 0.0

    # A cube bucket without a member the gate READS cannot be scored.
    _cube_bucket(test_dir, ["c", "d"], now[1:], "test")
    assert not ensemble_viz._apply_combiner_to_test_cubes(False, SPATIAL_GATE_KIND)


def test_compare_loads_a_gate_whose_unread_members_left_the_cubes(tmp_path):
    comb = SpatialGateCombiner(["a", "b", "c"], _random_params(2, 8, False),
                               width=8, use_lr=False, active_members=(0, 2))
    save_spatial_gate(comb, str(tmp_path / "spatial_gate_p2"))
    methods = sgc.load_gate_methods(str(tmp_path), ["spatial_gate_p2"], ["c", "x", "a"])
    method = methods["gate:spatial_gate_p2"]
    assert method.model.member_labels == ["a", "c"] and method.index == [2, 0]
    full, _lr = _members_and_lr(n_members=3)
    cubes = np.stack([full[2], full[1], full[0]])            # c, x, a
    np.testing.assert_allclose(method.model.apply_field(cubes[method.index]),
                               comb.apply_field(full), rtol=1e-5)
    with pytest.raises(ValueError, match="lack: a"):
        sgc.load_gate_methods(str(tmp_path), ["spatial_gate_p2"], ["b", "c"])


def test_fit_can_prune_to_a_member_subset(tmp_path):
    rng = np.random.default_rng(2)
    fields = []
    for index in range(4):
        target = rng.exponential(150.0, (64, 64, N_BANDS)).astype(np.float32)
        members = np.stack([target + rng.normal(0, s, target.shape).astype(np.float32)
                            for s in (20.0, 400.0, 30.0)])
        paths = []
        for m in range(3):
            path = tmp_path / f"member{m}_{index:05d}.npy"
            np.save(path, members[m])
            paths.append(str(path))
        lr = target.reshape(32, 2, 32, 2, N_BANDS).sum(axis=(1, 3))
        fields.append(GateField(index, paths, target, lr))
    comb = fit_spatial_gate(fields[:3], fields[3:], ["a", "b", "c"], width=8,
                            steps=20, batch_size=2, crop=48, eval_every=10,
                            warmup_steps=2, seed=0, active_members=[0, 2])
    assert comb.member_labels == ["a", "b", "c"] and comb.active_members == (0, 2)
    assert comb.fit_meta["active_member_labels"] == ["a", "c"]
    stack = np.stack([np.load(p) for p in fields[3].member_paths])
    assert comb.apply_field(stack, lr=fields[3].lr_e).shape == (64, 64, N_BANDS)


def test_member_arrays_can_run_a_member_subset():
    class Member:
        def __init__(self, value):
            self.value = value

        def upsample_array(self, lr):
            return np.full((2, 2, 1), self.value, np.float32)

    ens = EnsembleModel.__new__(EnsembleModel)
    ens._models = [Member(v) for v in (1.0, 2.0, 3.0)]
    assert ens.member_arrays(np.zeros((1, 1, 1)), indices=[2, 0])[:, 0, 0, 0].tolist() == [3.0, 1.0]
    assert ens.member_arrays(np.zeros((1, 1, 1))).shape[0] == 3


@pytest.mark.parametrize("mix_space", [MIX_ASINH, MIX_LINEAR])
def test_all_knee_loss_fits_and_records_its_knees(tmp_path, mix_space):
    rng = np.random.default_rng(3)
    fields = []
    for index in range(4):
        # Multiplicative (sign-preserving) noise: additive noise that flips
        # signs makes the low-knee terms a pathological toy objective.
        target = rng.exponential(150.0, (64, 64, N_BANDS)).astype(np.float32)
        good = np.zeros((2, 64, 64, 1), np.float32)
        good[0, :, :32] = good[1, :, 32:] = 1.0
        factor = np.exp(rng.normal(0, 0.7, (2, 64, 64, N_BANDS))).astype(np.float32)
        members = target[None] * (good + (1 - good) * factor)
        paths = []
        for m in range(2):
            path = tmp_path / f"member{m}_{index:05d}.npy"
            np.save(path, members[m])
            paths.append(str(path))
        lr = target.reshape(32, 2, 32, 2, N_BANDS).sum(axis=(1, 3))
        fields.append(GateField(index, paths, target, lr))
    comb = fit_spatial_gate(fields[:3], fields[3:], ["a", "b"], width=8, steps=120,
                            batch_size=4, crop=48, eval_every=40, learning_rate=1e-2,
                            warmup_steps=5, seed=0, loss_knees=ALL_KNEE_LOSS,
                            mix_space=mix_space)
    meta = comb.fit_meta
    assert comb.mix_space == meta["mix_space"] == mix_space
    assert meta["loss_knees_e"] == pytest.approx(list(ALL_KNEE_LOSS))
    assert meta["selected"]["loss"] < meta["baseline_holdout"]["loss"]
    assert len(meta["selected"]["integrated_psnr"]) == N_BANDS
