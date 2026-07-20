"""Combiner web layer: fit route (per star regime), combiner.json, the local
fit job's reuse-of-cached-validate-inference path, and the payload builder."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.image.core import Image
from euclid_polish.image.tfio import tfrecord_path, write_images
from euclid_polish.web.app import create_app
from euclid_polish.web.helpers import ensemble_viz as ev

BANDS = tuple(Config.HR_TARGET_BAND_NAMES)


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _fit_tiny_combiner(starless=False):
    """A saved combiner in a regime dir (default starfull) w/o a real ensemble."""
    from euclid_polish.eval.combiner import fit_combiner, save_combiner
    rng = np.random.default_rng(0)
    n = 2000
    y = np.arcsinh(np.abs(rng.normal(30, 20, n)) ** 2 / 100.0).astype(np.float32)
    X = np.stack([y + rng.normal(0, .02, n), rng.normal(0, 1, n)], 1).astype(np.float32)
    buffers = dict.fromkeys(BANDS, (X, y))
    comb = fit_combiner(buffers, ["00·psnr", "01·psnr"], n_kernels=6, steps=150)
    save_combiner(comb, ev._ensemble_regime_dir(starless))
    return comb


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

def test_combiner_fit_allows_starless(client):
    # The combiner is now available in BOTH regimes: a starless fit is accepted
    # (it spawns a job) rather than rejected.
    r = client.post("/ensemble/combiner/fit", data={"mode": "starless"})
    assert r.status_code == 200
    assert "job_id" in r.get_json()


def test_combiner_fit_api_preserves_explicit_zero_min_usage(client, monkeypatch):
    from euclid_polish.web.routes import ensemble as ensemble_routes

    seen = {}

    def fake_job(_cap, **kwargs):
        seen.update(kwargs)

    def fake_spawn(_description, target):
        target(_Cap())
        return "zero-min-usage"

    monkeypatch.setattr(ensemble_routes, "job_combiner_fit", fake_job)
    monkeypatch.setattr(ensemble_routes.REGISTRY, "spawn", fake_spawn)
    response = client.post("/ensemble/combiner/fit", data={
        "mode": "starfull", "model_kind": "stats_rbf_gate",
        "n_kernels": "32", "min_usage": "0",
    })
    assert response.status_code == 200
    assert response.get_json()["job_id"] == "zero-min-usage"
    assert seen["min_usage"] == 0.0


def test_combiner_json_reports_unavailable_before_fit(client):
    response = client.get("/ensemble/combiner.json")
    assert response.status_code == 200
    assert response.get_json()["available"] is False


def test_combined_combiner_unavailable_payload_has_stable_collections(client):
    payload = client.get("/ensemble/combined-combiner.json").get_json()
    assert payload["available"] is False
    assert payload["member_labels"] == []
    assert payload["members"] == []
    assert payload["band_names"] == []
    assert payload["surviving"] == {}
    assert payload["eff_weights"] == {}


def test_combiner_json_served_after_fit(client):
    _fit_tiny_combiner()
    r = client.get("/ensemble/combiner.json?fresh=1")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["available"] is True
    assert payload["member_labels"] == ["00·psnr", "01·psnr"]
    assert set(payload["eff_weights"]) == set(BANDS)
    # each band exposes a brightness sweep + per-member Jacobian
    ew = payload["eff_weights"]["VIS"]
    assert len(ew["brightness_e"]) == len(ew["jacobian"])


# ---------------------------------------------------------------------------
# compute_combiner_payload
# ---------------------------------------------------------------------------

def test_compute_combiner_payload_from_saved():
    _fit_tiny_combiner()
    payload = ev.compute_combiner_payload(False)
    assert payload is not None
    assert payload["available"] and set(payload["surviving"]) == set(BANDS)
    assert os.path.isfile(ev._combiner_payload_path(False))


def test_hr_weight_diagnostic_groups_stack_weights_by_target(tmp_path, monkeypatch):
    """HR is only the grouping variable; the combiner still sees predictions."""
    from euclid_polish.eval.combiner import BandCombiner, Combiner

    rdir = str(tmp_path / "records")
    os.makedirs(rdir, exist_ok=True)
    _write_records(rdir, "hr_validate", 1, shape=(8, 8, 4))
    monkeypatch.setattr(ev, "_sky_records_local_dir", lambda: rdir)

    cubes = tmp_path / "cubes_validate"
    cubes.mkdir()
    rng = np.random.default_rng(4)
    for i in range(2):
        np.save(cubes / f"member{i}_00000.npy",
                np.abs(rng.normal(50, 10, (8, 8, 4))).astype(np.float32))
    with open(cubes / "viz_index.json", "w") as f:
        json.dump({"subset": "validate", "indices": [0],
                   "member_labels": ["00", "01"], "records_fp": "fp"}, f)

    bands = {}
    for name in BANDS:
        bands[name] = BandCombiner(
            V=np.zeros((3, 2), np.float32), a=np.zeros(2, np.float32),
            centers=np.linspace(-1, 13, 3, dtype=np.float32), sigma=1.0,
            surviving=np.ones(2, bool))
    comb = Combiner(member_labels=["00", "01"], n_kernels=3,
                    sigma_scale=1.0, min_usage=0.0, bands=bands,
                    band_names=BANDS, records_fp="fp")

    payload = ev._hr_weight_diagnostic_from_bucket(
        comb, starless=False, cubes_dir=str(cubes), target="hr")
    assert payload["available"] is True
    assert payload["n_fields"] == 1
    assert payload["n_pixels"] == 64
    for band in BANDS:
        data = payload["bands"][band]
        assert any(count > 0 for count in data["counts"])
        for row in data["mean"]:
            if row[0] is not None:
                assert sum(row) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# the fit job: reuse cached validate member cubes (no re-inference)
# ---------------------------------------------------------------------------

class _Cap:
    def tick(self, *a, **k):
        pass


def _write_records(rdir, name, n, shape=(16, 16, 4)):
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    imgs = [Image(data=np.abs(rng.normal(50, 30, shape)).astype(np.float32),
                  pixel_scale_arcsec=0.05, band_names=BANDS, is_clean=True,
                  index=i) for i in range(n)]
    write_images(imgs, name, records_dir=rdir)


def test_job_combiner_fit_reuses_validate_cubes(tmp_path, monkeypatch):
    rdir = str(tmp_path / "records")
    os.makedirs(rdir, exist_ok=True)
    _write_records(rdir, "dirty_validate", 2, shape=(8, 8, 4))
    _write_records(rdir, "hr_validate", 2, shape=(16, 16, 4))
    monkeypatch.setattr(ev, "_sky_records_local_dir", lambda: rdir)

    fp = ev._eval_records_fingerprint(rdir, "validate")
    val_dir = ev._ensemble_cubes_dir("validate", starless=False)
    os.makedirs(val_dir, exist_ok=True)
    labels = ["00·psnr", "01·psnr"]
    rng = np.random.default_rng(3)
    for idx in (0, 1):
        for mi in range(2):
            np.save(os.path.join(val_dir, f"member{mi}_{idx:05d}.npy"),
                    np.abs(rng.normal(50, 30, (16, 16, 4))).astype(np.float32))
    with open(os.path.join(val_dir, "viz_index.json"), "w") as f:
        json.dump({"subset": "validate", "indices": [0, 1],
                   "member_labels": labels, "records_fp": fp}, f)
    monkeypatch.setattr(ev, "_regime_labels", lambda *_args: labels)

    # A guard so we KNOW inference never ran (reuse path only).
    def _boom(*a, **k):
        raise AssertionError("evaluate_on_records ran despite a valid cache")
    monkeypatch.setattr(ev, "evaluate_on_records", _boom)

    summary = ev.job_combiner_fit(_Cap(), num_images=2, n_kernels=6)
    assert summary["n_members"] == 2
    assert set(summary["surviving"]) == set(BANDS)

    from euclid_polish.eval.combiner import load_combiner
    comb = load_combiner(ev._ensemble_regime_dir(False))
    assert comb is not None and comb.member_labels == labels
    assert comb.records_fp == fp
    assert os.path.isfile(ev._combiner_payload_path(False))


def test_job_combiner_fit_requires_validate_records(tmp_path, monkeypatch):
    rdir = str(tmp_path / "records")
    os.makedirs(rdir, exist_ok=True)          # no validate tfrecords written
    monkeypatch.setattr(ev, "_sky_records_local_dir", lambda: rdir)
    with pytest.raises(RuntimeError, match="validate records"):
        ev.job_combiner_fit(_Cap(), num_images=2)


# ---------------------------------------------------------------------------
# combiner as a first-class series in the evaluations payload
# ---------------------------------------------------------------------------

def test_compute_evaluation_payload_includes_combiner(tmp_path, monkeypatch):
    """When the cached fields carry a combiner plane, the evals payload gains
    ps.T_comb/r_comb and a combiner comparison block (combiner vs mean vs best
    member)."""
    n, M = 48, 3
    def _field(seed, rec):
        r = np.random.default_rng(seed)
        hr = np.cumsum(r.normal(0, 1, (n, n)), axis=0).astype(np.float32) * 20
        members = np.stack([hr + r.normal(0, 3, (n, n)) for _ in range(M)])
        # target, mean, members, combiner, combined combiner, LR, record index
        return hr, members.mean(0), members, {"rbf_gate": hr}, None, None, rec

    fields = [_field(1, 0), _field(2, 1)]
    monkeypatch.setattr(ev, "_iter_cached_fields", lambda starless: iter(fields))

    cubes = ev._ensemble_cubes_dir(starless=False)
    os.makedirs(cubes, exist_ok=True)
    with open(os.path.join(cubes, "viz_index.json"), "w") as f:
        json.dump({"subset": "test", "indices": [0, 1],
                   "member_labels": ["00·psnr", "01·psnr", "02·psnr"],
                   "has_combiner": True}, f)

    payload = ev.compute_evaluation_payload(False)
    assert payload is not None
    assert payload["ps"] is not None and "T_comb" in payload["ps"]
    cb = payload["combiner"]
    assert cb is not None and cb["available"] is True
    assert cb["psnr"] is not None and cb["ensemble_mean_psnr"] is not None
    # perfect combiner (== HR) beats the noisy ensemble mean
    assert cb["psnr"] > cb["ensemble_mean_psnr"]


def test_viewer_advertises_combiner_tier_only_when_present(monkeypatch):
    from euclid_polish.web.helpers import viewer_data as vd
    base = {"subset": "test", "indices": [0, 1], "member_labels": ["00·psnr"]}
    monkeypatch.setattr(vd, "_sky_records_local_dir", lambda: "")   # no hr tier

    monkeypatch.setattr(
        vd, "_ensemble_manifest", lambda _starless: {**base, "has_combiner": True}
    )
    keys = [t["key"] for t in vd._ensemble_meta({})["tiers"]]
    assert "comb_rbf" in keys

    monkeypatch.setattr(
        vd, "_ensemble_manifest", lambda _starless: {**base, "has_combiner": False}
    )
    assert "comb_rbf" not in [t["key"] for t in vd._ensemble_meta({})["tiers"]]


# ---------------------------------------------------------------------------
# auto-score the combiner on test after a fit (no re-inference)
# ---------------------------------------------------------------------------

def test_apply_combiner_to_test_cubes_from_cached_members(tmp_path, monkeypatch):
    """Applying a fitted combiner to the cached TEST member cubes writes comb_
    cubes equal to combiner.apply_field(stack) and flips the manifest flag —
    the mechanism behind auto-scoring a fresh combiner without re-inference."""
    from euclid_polish.eval.combiner import fit_combiner, load_combiner, save_combiner

    regime = tmp_path / "starfull"
    cubes = regime / "cubes"
    cubes.mkdir(parents=True)
    monkeypatch.setattr(ev, "_ensemble_regime_dir",
                        lambda sl: str(tmp_path / ("starless" if sl else "starfull")))
    monkeypatch.setattr(ev, "_ensemble_cubes_dir", lambda *a, **k: str(cubes))

    rng = np.random.default_rng(0)
    n = 2000
    y = np.arcsinh(np.abs(rng.normal(30, 20, n)) ** 2 / 100.0).astype(np.float32)
    X = np.stack([y + rng.normal(0, .02, n), rng.normal(0, 1, n)], 1).astype(np.float32)
    comb = fit_combiner(dict.fromkeys(BANDS, (X, y)), ["00·p", "01·p"],
                        n_kernels=6, steps=120)
    save_combiner(comb, str(regime))

    labels = ["00·p", "01·p"]
    for rec in (3, 7):                       # cached test member cubes, 2 members
        for i in range(2):
            np.save(cubes / f"member{i}_{rec:05d}.npy",
                    np.abs(rng.normal(200, 150, (5, 5, 4))).astype(np.float32))
    (cubes / "viz_index.json").write_text(json.dumps(
        {"subset": "test", "indices": [3, 7], "member_labels": labels,
         "has_combiner": False}))

    assert ev._apply_combiner_to_test_cubes(False) is True
    man = json.load(open(cubes / "viz_index.json"))
    assert man["has_combiner"] is True

    loaded = load_combiner(str(regime), member_labels=labels)
    for rec in (3, 7):
        stack = np.stack([np.load(cubes / f"member{i}_{rec:05d}.npy")
                          for i in range(2)], 0)
        got = np.load(cubes / f"comb_{rec:05d}.npy")
        np.testing.assert_allclose(got, loaded.apply_field(stack),
                                   rtol=1e-5, atol=1e-2)


def test_apply_all_combiners_writes_independent_rbf_and_stats_cubes(tmp_path, monkeypatch):
    """Fitting one model must not skip or overwrite the other model's cache."""
    from euclid_polish.eval.combiner import fit_combiner, save_combiner

    regime = tmp_path / "starfull"
    cubes = regime / "cubes"
    cubes.mkdir(parents=True)
    monkeypatch.setattr(ev, "_ensemble_regime_dir", lambda sl: str(regime))
    monkeypatch.setattr(ev, "_ensemble_cubes_dir", lambda *a, **k: str(cubes))

    rng = np.random.default_rng(21)
    n = 500
    y = rng.normal(1.0, .3, n).astype(np.float32)
    X = np.stack([y + rng.normal(0, .02, n), y + rng.normal(0, .04, n)], 1)
    buffers = dict.fromkeys(BANDS, (X, y))
    labels = ["00·p", "01·p"]
    rbf = fit_combiner(buffers, labels, n_kernels=4, steps=20)
    stats = fit_combiner(buffers, labels, model_kind="stats_rbf_gate",
                         n_kernels=4, steps=20, batch=128)
    save_combiner(rbf, str(regime), artifact_dir="combiner")
    save_combiner(stats, str(regime), artifact_dir="stats_rbf_combiner")

    rec, tag = 3, "00003"
    for i in range(2):
        np.save(cubes / f"member{i}_{tag}.npy",
                np.abs(rng.normal(200, 50, (5, 5, 4))).astype(np.float32))
    (cubes / "viz_index.json").write_text(json.dumps(
        {"subset": "test", "indices": [rec], "member_labels": labels}))

    assert ev._apply_all_combiners_to_test_cubes(False) is True
    assert (cubes / f"comb_{tag}.npy").is_file()
    assert (cubes / f"comb_stats_rbf_{tag}.npy").is_file()
    man = json.loads((cubes / "viz_index.json").read_text())
    assert man["has_combiner_rbf_gate"] is True
    assert man["has_combiner_stats_rbf_gate"] is True


def test_apply_combiner_to_test_cubes_noops_without_combiner(tmp_path, monkeypatch):
    from euclid_polish.web.helpers import ensemble_viz as ev2
    cubes = tmp_path / "starfull" / "cubes"
    cubes.mkdir(parents=True)
    monkeypatch.setattr(ev2, "_ensemble_regime_dir", lambda sl: str(tmp_path / "starfull"))
    monkeypatch.setattr(ev2, "_ensemble_cubes_dir", lambda *a, **k: str(cubes))
    (cubes / "viz_index.json").write_text(json.dumps(
        {"subset": "test", "indices": [0], "member_labels": ["00·p"]}))
    assert ev2._apply_combiner_to_test_cubes(False) is False    # no combiner saved


def test_reuse_validate_cubes_rejects_member_set_change(tmp_path):
    """A fit must NOT reuse cached validate cubes whose member set no longer
    matches the active ensemble — else the combiner is fit over a stale set and
    shows 'stale' the instant it's fitted."""
    vd = tmp_path / "cubes_validate"
    vd.mkdir()
    (vd / "viz_index.json").write_text(json.dumps(
        {"subset": "validate", "indices": [0, 1],
         "member_labels": ["00·psnr", "01·psnr"], "records_fp": "fp1"}))

    # same set + fingerprint → reuse
    assert ev._reuse_validate_cubes(str(vd), "fp1", ["00·psnr", "01·psnr"]) is not None
    # a member was added → reject (re-infer over the current members)
    assert ev._reuse_validate_cubes(str(vd), "fp1",
                                    ["00·psnr", "01·psnr", "02·psnr"]) is None
    # records changed → reject (existing behaviour)
    assert ev._reuse_validate_cubes(str(vd), "fp2", ["00·psnr", "01·psnr"]) is None
    # no active_labels passed → member check skipped (back-compat)
    assert ev._reuse_validate_cubes(str(vd), "fp1") is not None
