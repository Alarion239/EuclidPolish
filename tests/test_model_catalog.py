"""Model catalogue (contract C9, plan WP-B2 T3): runnable STARFULL model specs
with labels, fingerprints, member requirements and availability, prediction
through stub members, and the per-(tile, spec) output store."""

from __future__ import annotations

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
    monkeypatch.setattr(mc, "regime_dir", lambda: root)
    monkeypatch.setattr(mc, "active_member_labels", lambda: list(LABELS))
    monkeypatch.setattr(mc, "member_fingerprints",
                        lambda labels: {label: fingerprints.get(label) for label in labels})
    return {"root": root, "fingerprints": fingerprints}


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


def test_production_unavailable_when_fitted_for_other_members(regime, monkeypatch):
    monkeypatch.setattr(mc, "active_member_labels", lambda: LABELS + ["4·psnr"])
    production = mc.resolve_spec("production")
    assert not production.available
    assert "3 members" in production.reason and "4" in production.reason


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


def test_member_runner_restores_only_the_members_it_needs(monkeypatch):
    """The runner caps ``EnsembleModel`` at the registry prefix holding the
    requested members instead of restoring every active checkpoint."""
    active = ["1·psnr", "2·psnr", "3·psnr", "4·psnr"]
    monkeypatch.setattr(mc, "active_member_labels", lambda: list(active))
    built: list[dict] = []

    class FakeEnsemble:
        def __init__(self, base_dir, **kwargs):
            built.append(kwargs)
            self.member_labels = active[:kwargs.get("n_members") or len(active)]

        def member_arrays(self, lr, indices):
            return np.stack([lr * (i + 1) for i in indices])

    runner = mc.EnsembleMemberRunner("/nowhere", factory=FakeEnsemble,
                                     labels=["2·psnr", "1·psnr"])
    out = runner.predict(np.ones((2, 2, 4), np.float32), "2·psnr")
    np.testing.assert_allclose(out, 2.0)
    assert built == [{"starless": False, "n_members": 2}]
    with pytest.raises(KeyError):
        runner.predict(np.ones((2, 2, 4), np.float32), "9·psnr")      # not active at all
    everything = mc.EnsembleMemberRunner("/nowhere", factory=FakeEnsemble)
    everything.predict(np.ones((2, 2, 4), np.float32), "4·psnr")
    assert built[-1] == {"starless": False}
