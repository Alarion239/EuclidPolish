"""Model catalogue (contract C9, plan WP-B2 T3): runnable STARFULL model specs
with labels, fingerprints, member requirements and availability, prediction
through stub members, and the per-(tile, spec) output store."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from euclid_polish.config import Config
from euclid_polish.eval.spatial_gate import MIX_LINEAR, SpatialGateCombiner, save_spatial_gate
from euclid_polish.eval.spatial_gate_fit import init_params
from euclid_polish.web.helpers import model_catalog as mc

LABELS = ["1·psnr", "2·psnr", "3·psnr"]


def _uniform_gate(labels, *, use_lr=False) -> SpatialGateCombiner:
    params = init_params(len(labels), 4, 8, use_lr, [0, 0, 0, 0], seed=1)
    params["b_out"] = np.zeros_like(params["b_out"])     # uniform convex weights
    return SpatialGateCombiner(list(labels), params, width=8, use_lr=use_lr,
                               mix_space=MIX_LINEAR,
                               fit_meta={"loss": "test", "steps": 5})


@pytest.fixture
def regime(tmp_path, monkeypatch):
    """A fake STARFULL regime: production gate for all three members, two
    variants (one runnable, one needing an inactive member), no RBF."""
    root = tmp_path / "starfull"
    save_spatial_gate(_uniform_gate(LABELS), str(root / "spatial_gate_combiner"))
    save_spatial_gate(_uniform_gate(LABELS[:2]), str(root / "spatial_gate_two"))
    save_spatial_gate(_uniform_gate(["1·psnr", "9·psnr"]), str(root / "spatial_gate_old"))
    fingerprints = {label: f"ckpt-{i}:10:100" for i, label in enumerate(LABELS)}
    universal: list[str] = []
    monkeypatch.setattr(mc, "regime_dir", lambda: root)
    monkeypatch.setattr(mc, "ensemble_dir", lambda: str(tmp_path / "ensemble"))
    monkeypatch.setattr(mc, "active_member_labels", lambda: list(LABELS))
    monkeypatch.setattr(mc, "universal_labels", lambda: list(universal))
    monkeypatch.setattr(mc, "member_fingerprints",
                        lambda labels: {label: fingerprints.get(label) for label in labels})
    return {"root": root, "fingerprints": fingerprints, "universal": universal,
            "ensemble": tmp_path / "ensemble"}


def test_canonical_specs_accept_aliases():
    assert mc.canonical_spec("production") == "production"
    assert mc.canonical_spec(" Mean ") == "mean"
    for alias in ("member:170", "member:member_170", "member:170·psnr"):
        assert mc.canonical_spec(alias) == "member:member_170"
    assert mc.canonical_spec("gate:spatial_gate_26m") == "gate:26m"
    assert mc.canonical_spec("gate:26m") == "gate:26m"
    assert mc.canonical_spec("rbf") == "rbf"
    for bad in ("", "foo", "member:", "gate:", "gate:../x", "member:abc"):
        with pytest.raises(ValueError):
            mc.canonical_spec(bad)


def test_spec_slugs_are_path_safe_and_unique():
    specs = ["production", "mean", "rbf", "member:member_170", "gate:26m", "gate:v2_lr"]
    slugs = [mc.spec_slug(spec) for spec in specs]
    assert len(set(slugs)) == len(slugs)
    assert all("/" not in slug and ":" not in slug for slug in slugs)


def test_list_specs_reports_availability_members_and_reasons(regime):
    specs = {spec.spec: spec for spec in mc.list_specs()}
    assert list(specs)[:3] == ["production", "mean", "rbf"]
    production = specs["production"]
    assert production.available and production.kind == "production"
    assert list(production.member_labels) == LABELS
    assert production.combiner_kind == "spatial_gate"
    assert production.fingerprint and production.combiner_fingerprint
    assert specs["mean"].available and list(specs["mean"].member_labels) == LABELS
    for i, label in enumerate(LABELS):
        member = specs[f"member:member_{i + 1}"]
        assert member.available and list(member.member_labels) == [label]
    assert specs["gate:two"].available
    assert list(specs["gate:two"].member_labels) == LABELS[:2]
    assert specs["gate:two"].details["mix_space"] == "linear"
    old = specs["gate:old"]
    assert not old.available and "9·psnr" in old.reason
    assert not specs["rbf"].available and "not fitted" in specs["rbf"].reason
    assert "gate:combiner" not in specs            # production is not a variant
    # A chip shows the label before " · ": each variant's must name it.
    assert specs["gate:two"].label.split(" · ")[0] == "Gate two"
    assert specs["gate:old"].label.split(" · ")[0] == "Gate old"
    payload = production.to_dict()
    json.dumps(payload)
    assert {"spec", "kind", "label", "members", "available", "reason",
            "fingerprint"} <= set(payload)


def test_fingerprints_follow_members_and_combiner_artifacts(regime):
    before = {spec.spec: spec.fingerprint for spec in mc.list_specs()}
    regime["fingerprints"]["2·psnr"] = "ckpt-9:11:200"      # member 2 retrained
    after = {spec.spec: spec.fingerprint for spec in mc.list_specs()}
    assert after["member:member_1"] == before["member:member_1"]
    assert after["member:member_2"] != before["member:member_2"]
    assert after["mean"] != before["mean"]
    assert after["production"] != before["production"]
    save_spatial_gate(_uniform_gate(LABELS, use_lr=True),
                      str(regime["root"] / "spatial_gate_combiner"))   # refit
    refit = {spec.spec: spec.fingerprint for spec in mc.list_specs()}
    assert refit["production"] != after["production"]
    assert refit["mean"] == after["mean"]


def test_production_stays_available_when_members_join_after_the_fit(regime, monkeypatch):
    """Members the gate never saw are a soft note, not unavailability."""
    before = mc.resolve_spec("production")
    fps = regime["fingerprints"]
    fps["4·psnr"] = "ckpt-4:10:100"
    monkeypatch.setattr(mc, "active_member_labels", lambda: LABELS + ["4·psnr"])
    production = mc.resolve_spec("production")
    assert production.available and production.reason is None
    assert production.note == "1 member(s) joined after this fit; refit to consider them"
    assert production.details["joined_after_fit"] == ["4·psnr"]
    assert production.to_dict()["note"] == production.note
    assert production.fingerprint == before.fingerprint      # outputs stay current
    assert before.note is None


def test_production_unavailable_when_a_member_it_reads_left(regime, monkeypatch):
    monkeypatch.setattr(mc, "active_member_labels", lambda: LABELS[:2])
    production = mc.resolve_spec("production")
    assert not production.available
    assert "3·psnr" in production.reason


def _pruned_gate(labels, active) -> SpatialGateCombiner:
    """A uniform gate fitted for ``labels`` that reads only ``active``."""
    reads = _uniform_gate([labels[i] for i in active])
    return dataclasses.replace(reads, member_labels=list(labels),
                               active_members=tuple(active))


def _prune_production(regime, active=(0, 2)):
    save_spatial_gate(_pruned_gate(LABELS, active),
                      str(regime["root"] / "spatial_gate_combiner"))


def test_pruned_production_keys_on_the_members_it_reads(regime, monkeypatch):
    """add / archive-unread / retrain-unread keep a pruned production gate
    available and its fingerprint unchanged; archive-read makes it
    unavailable, retrain-read makes its outputs stale."""
    _prune_production(regime)                     # reads 1·psnr and 3·psnr
    base = mc.resolve_spec("production")
    assert base.available and list(base.reads) == ["1·psnr", "3·psnr"]
    assert list(base.member_labels) == LABELS and base.to_dict()["n_members"] == 2
    fps = regime["fingerprints"]
    # add: a member registered after the fit
    fps["4·psnr"] = "ckpt-4:10:100"
    monkeypatch.setattr(mc, "active_member_labels", lambda: LABELS + ["4·psnr"])
    added = mc.resolve_spec("production")
    assert added.available and added.fingerprint == base.fingerprint and added.note
    # archive-unread: 2·psnr (not read) leaves the ensemble
    monkeypatch.setattr(mc, "active_member_labels", lambda: ["1·psnr", "3·psnr"])
    unread_left = mc.resolve_spec("production")
    assert unread_left.available and unread_left.fingerprint == base.fingerprint
    # retrain-unread: 2·psnr gets a new checkpoint under the same label
    monkeypatch.setattr(mc, "active_member_labels", lambda: list(LABELS))
    fps["2·psnr"] = "ckpt-99:12:300"
    assert mc.resolve_spec("production").fingerprint == base.fingerprint
    # retrain-read: 3·psnr gets a new checkpoint → cached outputs go stale
    fps["3·psnr"] = "ckpt-98:12:300"
    retrained = mc.resolve_spec("production")
    assert retrained.available and retrained.fingerprint != base.fingerprint
    # archive-read: 3·psnr (read) leaves → unavailable, no fallback
    monkeypatch.setattr(mc, "active_member_labels", lambda: ["1·psnr", "2·psnr"])
    read_left = mc.resolve_spec("production")
    assert not read_left.available and "3·psnr" in read_left.reason
    assert read_left.fingerprint is None


def test_pruned_gate_variant_needs_only_its_reads(regime, monkeypatch):
    save_spatial_gate(_pruned_gate(["1·psnr", "9·psnr", "3·psnr"], (0, 2)),
                      str(regime["root"] / "spatial_gate_skip9"))
    variant = mc.resolve_spec("gate:skip9")
    assert variant.available, variant.reason      # 9·psnr is fitted but never read
    members = _Members()
    out = mc.predict(variant, np.ones((4, 4, 4), np.float32), members)
    assert members.calls == ["1·psnr", "3·psnr"]
    np.testing.assert_allclose(out, members.base * 2.0, rtol=1e-5)


def test_resolve_unknown_spec_raises(regime):
    with pytest.raises(KeyError):
        mc.resolve_spec("member:member_77")
    with pytest.raises(KeyError):
        mc.resolve_spec("gate:nope")


class _Members:
    """Stub member store: member k predicts (k+1)·base."""

    def __init__(self, shape=(8, 8, 4)):
        self.base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1.0
        self.calls: list[str] = []

    def get(self, label: str) -> np.ndarray:
        self.calls.append(label)
        return self.base * float(label.split("·")[0])


def test_predict_mean_member_and_gates(regime):
    members = _Members()
    lr = np.ones((4, 4, 4), np.float32)
    mean = mc.predict(mc.resolve_spec("mean"), lr, members)
    np.testing.assert_allclose(mean, members.base * 2.0)
    one = mc.predict(mc.resolve_spec("member:member_3"), lr, members)
    np.testing.assert_allclose(one, members.base * 3.0)
    production = mc.predict(mc.resolve_spec("production"), lr, members)
    np.testing.assert_allclose(production, members.base * 2.0, rtol=1e-5)
    members.calls.clear()
    two = mc.predict(mc.resolve_spec("gate:two"), lr, members)
    np.testing.assert_allclose(two, members.base * 1.5, rtol=1e-5)
    assert members.calls == LABELS[:2]           # only the gate's own members run
    with pytest.raises(ValueError, match="unavailable"):
        mc.predict(mc.resolve_spec("gate:old"), lr, members)


def test_pruned_gate_counts_the_members_it_reads(regime):
    """``n_members`` is what the spec runs (``reads``); a pruned gate keeps
    the total it was fitted with in ``n_fitted`` (API.md /api/models)."""
    pruned = dataclasses.replace(_uniform_gate(LABELS), active_members=(0, 2))
    save_spatial_gate(pruned, str(regime["root"] / "spatial_gate_pruned"))
    payload = mc.resolve_spec("gate:pruned").to_dict()
    assert payload["reads"] == [LABELS[0], LABELS[2]]
    assert payload["n_members"] == 2
    assert payload["n_fitted"] == 3
    assert payload["members"] == LABELS
    full = mc.resolve_spec("gate:two").to_dict()
    assert full["n_members"] == full["n_fitted"] == 2


def test_needed_members_union(regime):
    specs = [mc.resolve_spec(s) for s in ("gate:two", "member:member_3")]
    assert mc.needed_members(specs) == LABELS


def test_output_store_round_trip(tmp_path, monkeypatch, regime):
    monkeypatch.setattr(Config, "EUCLID_INFERENCE_DIR", str(tmp_path / "inference"))
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    header["CRVAL1"], header["CRVAL2"] = 268.4, 65.2
    header["CRPIX1"], header["CRPIX2"] = 2.5, 2.5
    header["CD1_1"], header["CD1_2"] = -0.1 / 3600, 0.0
    header["CD2_1"], header["CD2_2"] = 0.0, 0.1 / 3600
    spec = mc.resolve_spec("mean")
    sr = np.full((8, 8, 4), 3.0, np.float32)
    meta = mc.save_output("nexus", "f200w-0001", spec, sr, lr_header=header,
                          lr_sha="abc", extra={"experiment_id": "e1"})
    assert meta["fingerprint"] == spec.fingerprint and meta["lr_sha"] == "abc"
    outputs = mc.list_outputs("nexus", "f200w-0001")
    assert list(outputs) == ["mean"]
    assert outputs["mean"]["experiment_id"] == "e1"
    cube, out_header, loaded = mc.load_output("nexus", "f200w-0001", "mean")
    np.testing.assert_allclose(cube, sr)
    assert loaded["spec"] == "mean"
    # SR grid = LR WCS magnified x2: CRPIX -> 2*CRPIX - 0.5, CD / 2
    sr_wcs = WCS(out_header).celestial
    assert sr_wcs.wcs.crpix[0] == pytest.approx(4.5)
    assert sr_wcs.pixel_scale_matrix[1, 1] == pytest.approx(0.05 / 3600)
    assert out_header["BUNIT"] == "electron"
    assert mc.output_state(outputs["mean"], {"mean": spec.fingerprint}) == "current"
    assert mc.output_state(outputs["mean"], {"mean": "other"}) == "stale"
    removed = mc.delete_outputs("nexus", "f200w-0001")
    assert removed and mc.list_outputs("nexus", "f200w-0001") == {}
    with pytest.raises(FileNotFoundError):
        mc.load_output("nexus", "f200w-0001", "mean")


def test_delete_outputs_matches_spec_slugs_exactly(tmp_path, monkeypatch):
    """``gate:v1`` must not take ``gate:v1.2`` with it (variants may hold dots)."""
    monkeypatch.setattr(Config, "EUCLID_INFERENCE_DIR", str(tmp_path / "inference"))
    sr = np.zeros((4, 4, 4), np.float32)
    for spec in ("gate:v1", "gate:v1.2", "mean"):
        mc.save_output("nexus", "f200w-0001",
                       mc.ModelSpec(spec, "gate", spec, (), (), True, fingerprint="fp"), sr,
                       lr_header=None, lr_sha=None)
    removed = mc.delete_outputs("nexus", "f200w-0001", specs=["gate:v1"])
    assert sorted(p.rsplit("/", 1)[1] for p in removed) == ["gate-v1.fits", "gate-v1.json"]
    assert set(mc.list_outputs("nexus", "f200w-0001")) == {"gate:v1.2", "mean"}
    removed = mc.delete_outputs("nexus", "f200w-0001", specs=["gate:v1.2"])
    assert sorted(p.rsplit("/", 1)[1] for p in removed) == ["gate-v1.2.fits", "gate-v1.2.json"]
    assert set(mc.list_outputs("nexus", "f200w-0001")) == {"mean"}


def test_spec_fingerprint_is_the_catalogue_formula(regime):
    """Legacy SR records (NEXUS / pair inference) rebuild a spec fingerprint
    from their recorded identity with the same formula as the catalogue."""
    specs = {spec.spec: spec for spec in mc.list_specs()}
    for name in ("production", "gate:two", "mean", "member:member_2"):
        item = specs[name]
        assert mc.spec_fingerprint(
            item.kind, combiner_kind=item.combiner_kind,
            combiner_fingerprint=item.combiner_fingerprint,
            member_labels=item.member_labels,
            member_fingerprints=item.member_fingerprints) == item.fingerprint
    assert mc.spec_fingerprint("production", combiner_kind="spatial_gate",
                               combiner_fingerprint=None, member_labels=LABELS,
                               member_fingerprints=["a", "b", "c"]) is None
    assert mc.spec_fingerprint("mean", member_labels=LABELS,
                               member_fingerprints=["a", None, "c"]) is None


class _LabelEnsemble:
    """Fake ``EnsembleModel``: restores exactly ``labels=`` (every active
    member without it); member ``k·psnr`` predicts ``k·lr``."""

    active: list[str] = []
    built: list[dict] = []

    def __init__(self, base_dir, **kwargs):
        type(self).built.append(kwargs)
        self.member_labels = list(kwargs.get("labels") or type(self).active)

    def member_arrays(self, lr, indices):
        return np.stack([lr * float(self.member_labels[i].split("·")[0]) for i in indices])


def test_member_runner_restores_only_the_members_it_needs(monkeypatch):
    """The runner restores exactly the requested members (``labels=``), not a
    registry prefix: a gate reading members 2 and 4 loads 2 checkpoints."""
    active = ["1·psnr", "2·psnr", "3·psnr", "4·psnr"]
    monkeypatch.setattr(mc, "active_member_labels", lambda: list(active))
    monkeypatch.setattr(mc, "universal_labels", lambda: [])
    monkeypatch.setattr(_LabelEnsemble, "active", active)
    monkeypatch.setattr(_LabelEnsemble, "built", [])
    runner = mc.EnsembleMemberRunner("/nowhere", factory=_LabelEnsemble,
                                     labels=["4·psnr", "2·psnr", "9·psnr"])
    out = runner.predict(np.ones((2, 2, 4), np.float32), "4·psnr")
    np.testing.assert_allclose(out, 4.0)
    # the unknown 9·psnr is left out of the restore, and reported at predict
    assert _LabelEnsemble.built == [{"starless": False, "labels": ["4·psnr", "2·psnr"]}]
    with pytest.raises(KeyError):
        runner.predict(np.ones((2, 2, 4), np.float32), "9·psnr")      # not active at all
    assert len(_LabelEnsemble.built) == 1                             # no rebuild for it
    everything = mc.EnsembleMemberRunner("/nowhere", factory=_LabelEnsemble)
    everything.predict(np.ones((2, 2, 4), np.float32), "4·psnr")
    assert _LabelEnsemble.built[-1] == {"starless": False}


# ---------------------------------------------------------------------------
# universal models (design 2026-10-13 §4)
# ---------------------------------------------------------------------------

UNIVERSAL = "211·psnr"
LOG_HEADER = "step,wall_time,loss,psnr_stretched,duration_s,combined_loss,is_baseline\n"


def _write_member(directory, *, origin, log_rows=()):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "origin.json").write_text(json.dumps(origin), encoding="utf-8")
    if log_rows:
        (directory / "training_log.csv").write_text(
            LOG_HEADER + "".join(f"{row}\n" for row in log_rows), encoding="utf-8")


LEARNED = {"num_res_blocks": 16, "target_steps": 1_000_000, "batch_size": 4,
           "crops_per_field": 8, "asinh_knees": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
           "learned_output_knee": {"init_e": 10.0, "min_e": 0.1, "max_e": 1000.0},
           "jit_compile": True, "mixed_precision": True, "starless": False}
FIXED = {"num_res_blocks": 32, "target_steps": 70000, "batch_size": 4, "crops_per_field": 8,
         "asinh_knees": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0], "output_knee": 10.0}
# 1000/2000 trained (3600 s + 1800 s), a resume baseline at 2000 (no duration),
# a rolled-back window re-trained 2000→3000 (1800 s): every non-baseline row counts.
LOG = ["1000,1.0,0.5,40.0,3600.0,0.3,", "2000,2.0,0.4,41.0,1800.0,0.2,",
       "2000,3.0,,41.0,,0.2,1", "3000,4.0,0.3,42.0,1800.0,0.1,"]


@pytest.fixture
def universe(regime):
    """``regime`` + a pulled universal model 211 (learned knee), a registered
    universal 212 still being pulled (no checkpoint), and active member 3 that
    is itself a universal model (fixed knee)."""
    ensemble = regime["ensemble"]
    _write_member(ensemble / "member_211", origin=LEARNED, log_rows=LOG)
    _write_member(ensemble / "member_3", origin=FIXED, log_rows=LOG[:2])
    _write_member(ensemble / "member_1", origin={"num_res_blocks": 32, "asinh_knee": 10.0})
    regime["fingerprints"][UNIVERSAL] = "ckpt-211:10:100"
    regime["universal"][:] = [UNIVERSAL, "212·psnr"]
    return regime


def test_canonical_universal_specs():
    for alias in ("universal:211", "universal:member_211", "universal:211·psnr",
                  " Universal:member_211 "):
        assert mc.canonical_spec(alias) == "universal:member_211"
    assert mc.spec_slug("universal:211") == "universal-member_211"
    assert mc.spec_slug("universal:211") != mc.spec_slug("member:211")
    for bad in ("universal:", "universal:abc", "universal:../x", "universal:2111111"):
        with pytest.raises(ValueError, match="universal"):
            mc.canonical_spec(bad)
    with pytest.raises(ValueError, match=r"unknown model spec 'foo'.*universal:<name>"):
        mc.canonical_spec("foo")
    assert mc.spec_kind("universal:member_211") == "universal"
    assert mc.spec_kind("member:member_211") == "member"
    assert mc.spec_kind("gate:two") == "gate" and mc.spec_kind("mean") == "mean"
    assert mc.spec_member_label("universal:211") == UNIVERSAL
    assert mc.spec_member_label("member:member_3") == "3·psnr"
    assert mc.spec_member_label("production") is None


def test_universal_models_list_last_and_stay_out_of_the_ensemble(universe):
    specs = mc.list_specs()
    kinds = [item.kind for item in specs]
    assert [item.spec for item in specs[:3]] == ["production", "mean", "rbf"]
    assert kinds[3:] == ["member"] * 3 + ["gate"] * 2 + ["universal"] * 2
    by_spec = {item.spec: item for item in specs}
    model = by_spec["universal:member_211"]
    assert model.kind == "universal" and model.label == "Universal 211"
    assert model.member_labels == model.reads == (UNIVERSAL,)
    assert model.available and model.reason is None
    assert model.member_fingerprints == ("ckpt-211:10:100",)
    assert model.fingerprint == mc._hash("universal", UNIVERSAL, "ckpt-211:10:100")
    assert model.fingerprint == mc.spec_fingerprint(
        "universal", member_labels=(UNIVERSAL,), member_fingerprints=("ckpt-211:10:100",))
    assert model.fingerprint != mc.spec_fingerprint(
        "member", member_labels=(UNIVERSAL,), member_fingerprints=("ckpt-211:10:100",))
    # a universal model is never part of the mean, the gates or the members list
    assert list(by_spec["mean"].member_labels) == LABELS
    assert "member:member_211" not in by_spec
    payload = mc.catalog_payload()
    assert payload["members"] == LABELS
    json.dumps(payload)
    row = next(item for item in payload["models"] if item["spec"] == "universal:member_211")
    assert row["slug"] == "universal-member_211" and row["n_members"] == 1


def test_a_universal_model_without_a_checkpoint_is_listed_unavailable(universe):
    pulling = mc.resolve_spec("universal:212")
    assert pulling.kind == "universal" and not pulling.available
    assert "no checkpoint" in pulling.reason and pulling.fingerprint is None
    assert mc.current_fingerprints()["universal:member_212"] is None


def test_a_universal_model_without_its_recipe_is_unavailable(universe):
    """A checkpoint without ``origin.json`` (a pull mid-way) would load as a
    single-knee model: unavailable until the recipe arrives."""
    (universe["ensemble"] / "member_211" / "origin.json").unlink()
    model = mc.resolve_spec("universal:211")
    assert not model.available and "origin.json" in model.reason
    assert model.fingerprint is None


def test_universal_details_are_cheap_recipe_and_log_reads(universe):
    details = dict(mc.resolve_spec("universal:211").details)
    assert details == {
        "universal": True, "class": "universal", "num_res_blocks": 16, "steps": 3000,
        "target_steps": 1_000_000, "batch_size": 4, "crops_per_field": 8,
        "knee": "learned", "output_knee": None, "jit_compile": True,
        "mixed_precision": True, "train_hours": pytest.approx(7200.0 / 3600.0)}
    # an active member that is a universal model: the same details, class member
    member = mc.resolve_spec("member:member_3")
    assert member.kind == "member"
    assert dict(member.details) == {
        "universal": True, "class": "member", "num_res_blocks": 32, "steps": 2000,
        "target_steps": 70000, "batch_size": 4, "crops_per_field": 8, "knee": "fixed",
        "output_knee": 10.0, "jit_compile": False, "mixed_precision": False,
        "train_hours": pytest.approx(1.5)}
    assert dict(mc.resolve_spec("member:member_1").details) == {}     # single-knee member
    assert dict(mc.resolve_spec("member:member_2").details) == {}     # no recipe on disk
    # still being pulled: the full key set, nothing known yet
    pulling = dict(mc.resolve_spec("universal:212").details)
    assert pulling["universal"] is True and pulling["class"] == "universal"
    assert pulling["steps"] is None and pulling["train_hours"] is None


def test_universal_details_are_cached_until_their_files_change(universe):
    mc.list_specs()
    before = mc._training_log_cached.cache_info().hits
    mc.list_specs()
    assert mc._training_log_cached.cache_info().hits > before        # no re-parse
    log = universe["ensemble"] / "member_211" / "training_log.csv"
    log.write_text(LOG_HEADER + "".join(f"{row}\n" for row in LOG)
                   + "4000,5.0,0.2,43.0,3600.0,0.1,\n", encoding="utf-8")
    details = mc.resolve_spec("universal:211").details
    assert details["steps"] == 4000 and details["train_hours"] == pytest.approx(3.0)


def test_universal_fingerprint_follows_its_checkpoint(universe):
    before = mc.resolve_spec("universal:211").fingerprint
    universe["fingerprints"][UNIVERSAL] = "ckpt-212:11:200"
    after = mc.resolve_spec("universal:211")
    assert after.available and after.fingerprint != before


def test_predict_a_universal_model(universe):
    members = _Members()
    out = mc.predict(mc.resolve_spec("universal:211"), np.ones((4, 4, 4), np.float32), members)
    np.testing.assert_allclose(out, members.base * 211.0)
    assert members.calls == [UNIVERSAL]
    with pytest.raises(KeyError, match="unknown model spec"):
        mc.resolve_spec("universal:999")
    with pytest.raises(ValueError, match="unavailable"):
        mc.predict(mc.resolve_spec("universal:212"), np.ones((4, 4, 4), np.float32), members)


def test_member_runner_restores_universal_models_with_explicit_labels(universe, monkeypatch):
    """A universal model runs as an ensemble of one through
    ``EnsembleModel(include_universal=True, labels=…)``; an active-only job
    keeps the plain ensemble call, and one being pulled is never restored."""
    monkeypatch.setattr(_LabelEnsemble, "active", list(LABELS))
    monkeypatch.setattr(_LabelEnsemble, "built", [])
    lr = np.ones((2, 2, 4), np.float32)
    runner = mc.EnsembleMemberRunner("/nowhere", factory=_LabelEnsemble,
                                     labels=["2·psnr", UNIVERSAL, "212·psnr"])
    np.testing.assert_allclose(runner.predict(lr, UNIVERSAL), 211.0)
    assert _LabelEnsemble.built == [{"starless": False, "labels": ["2·psnr", UNIVERSAL],
                                     "include_universal": True}]
    with pytest.raises(KeyError, match="universal model"):
        runner.predict(lr, "212·psnr")                                 # no checkpoint
    assert len(_LabelEnsemble.built) == 1
    # an active-only runner widens to a universal model on demand
    widened = mc.EnsembleMemberRunner("/nowhere", factory=_LabelEnsemble, labels=["1·psnr"])
    widened.predict(lr, "1·psnr")
    assert _LabelEnsemble.built[-1] == {"starless": False, "labels": ["1·psnr"]}
    np.testing.assert_allclose(widened.predict(lr, UNIVERSAL), 211.0)
    assert _LabelEnsemble.built[-1] == {"starless": False, "labels": ["1·psnr", UNIVERSAL],
                                        "include_universal": True}


def test_member_runner_never_plans_a_label_twice(universe, monkeypatch):
    """A planned universal model that is not runnable at the first build (its
    checkpoint mid-pull) is left out; once it can run, the rebuild restores it
    ONCE — never ``member_212 is listed twice`` for every later predict."""
    monkeypatch.setattr(_LabelEnsemble, "active", list(LABELS))
    monkeypatch.setattr(_LabelEnsemble, "built", [])
    lr = np.ones((2, 2, 4), np.float32)
    runner = mc.EnsembleMemberRunner("/nowhere", factory=_LabelEnsemble,
                                     labels=["2·psnr", "212·psnr"])
    np.testing.assert_allclose(runner.predict(lr, "2·psnr"), 2.0)
    assert _LabelEnsemble.built == [{"starless": False, "labels": ["2·psnr"]}]
    with pytest.raises(KeyError):
        runner.predict(lr, "212·psnr")                                 # still being pulled
    # the pull lands: a checkpoint and a STARFULL recipe
    _write_member(universe["ensemble"] / "member_212", origin=LEARNED)
    universe["fingerprints"]["212·psnr"] = "ckpt-212:10:100"
    np.testing.assert_allclose(runner.predict(lr, "212·psnr"), 212.0)
    np.testing.assert_allclose(runner.predict(lr, "2·psnr"), 2.0)
    assert _LabelEnsemble.built[1:] == [{"starless": False, "labels": ["2·psnr", "212·psnr"],
                                         "include_universal": True}]
    assert runner._labels == ["2·psnr", "212·psnr"]


def test_member_runner_restores_each_label_once(monkeypatch):
    """Backstop: duplicates never reach ``EnsembleModel`` whatever the plan holds."""
    monkeypatch.setattr(mc, "active_member_labels", lambda: list(LABELS))
    monkeypatch.setattr(mc, "universal_labels", lambda: [])
    assert mc.EnsembleMemberRunner._runnable(["1·psnr", "2·psnr", "1·psnr"]) == (
        ["1·psnr", "2·psnr"], set())
