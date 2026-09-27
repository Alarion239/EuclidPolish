"""W-SkyResults backend (spec §7.2–7.4 + the phase-3 carry-overs):

* ``POST /api/jwst-euclid/nexus/infer`` takes a tile subset and one model spec;
* catalogue-evaluation objects keep a correct WCS through the canonical crop
  (``enforce_object_sizes`` shifts ``CRPIX`` and nests the 2× grids);
* the eval reuse key includes the production combiner identity;
* ``write_disagreement_cubes`` persists the member mean (``mean.fits``) and the
  ``evaluation`` viewer collection exposes it as the movie centre;
* the ensemble cube cache and the synthetic evaluator default to STARFULL and
  reconstruct cached stacks through the production combiner;
* ``/api/evaluation/*`` reports per-object staleness and answers JSON errors.
"""

from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from euclid_polish.config import Config
from euclid_polish.eval import (
    catalog_runner,
    disagreement,
    ensemble_cube_cache,
    grouped_runner,
    synthetic_runner,
)
from euclid_polish.eval.combiner import combiner_artifact_fingerprint
from euclid_polish.eval.ensemble_infer import EvalEnsemble, ProductionPlan, production_plan
from euclid_polish.eval.spatial_gate import save_spatial_gate
from euclid_polish.web import app as web_app
from euclid_polish.web.helpers import ensemble_viz, viewer_data
from euclid_polish.web.jobs import REGISTRY
from euclid_polish.web.routes import jwst_euclid as jwst_routes
from tests import _real_fixtures as fx


def _wait(job_id: str, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = REGISTRY.get(job_id)
        if job is not None and job.status != "running":
            return job.to_dict()
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish")


# ---------------------------------------------------------------------------
# POST /api/jwst-euclid/nexus/infer — tile subset + model spec
# ---------------------------------------------------------------------------

@pytest.fixture
def nexus(tmp_path, monkeypatch):
    fx.point_store(tmp_path, monkeypatch)
    fx.stub_regime(tmp_path, monkeypatch)
    field = fx.make_nexus_field(n_tiles=2)
    calls: list[tuple[str, dict]] = []

    def fake_run(identifier, **kwargs):
        calls.append((identifier, kwargs))
        return {"ok": True}

    monkeypatch.setattr(jwst_routes, "run_starfull_nexus_field_inference", fake_run)
    app = web_app.create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield {"client": client, "field": field, "calls": calls}


def test_nexus_infer_passes_tile_subset_and_spec(nexus):
    response = nexus["client"].post("/api/jwst-euclid/nexus/infer", data={
        "field_id": nexus["field"], "tiles": "f200w-0001, 0", "spec": "member:2"})
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["ok"] is True and body["spec"] == "member:member_2"
    assert body["tiles"] == ["f200w-0001", "0"]
    job = _wait(body["job_id"])
    assert job["kind"] == "nexus-inference"
    identifier, kwargs = nexus["calls"][0]
    assert identifier == nexus["field"]
    assert kwargs["tiles"] == ["f200w-0001", "0"]
    assert kwargs["spec"] == "member:member_2"


def test_nexus_infer_defaults_to_production_on_every_tile(nexus):
    response = nexus["client"].post("/api/jwst-euclid/nexus/infer",
                                    data={"field_id": nexus["field"]})
    assert response.status_code == 200, response.get_json()
    _wait(response.get_json()["job_id"])
    _identifier, kwargs = nexus["calls"][0]
    assert kwargs["tiles"] is None
    assert kwargs["spec"] == "production"


@pytest.mark.parametrize(("form", "needle"), [
    ({"spec": "gate:../evil"}, "gate spec"),
    ({"spec": "wizard"}, "unknown model spec"),
    ({"spec": "mean,production"}, "one model spec"),
    ({"spec": "gate:nosuch"}, "gate:nosuch"),
    ({"tiles": "f200w-9999"}, "unknown NEXUS tile"),
])
def test_nexus_infer_rejects_bad_tiles_or_spec(nexus, form, needle):
    response = nexus["client"].post("/api/jwst-euclid/nexus/infer",
                                    data={"field_id": nexus["field"], **form})
    assert response.status_code == 400
    body = response.get_json()
    assert body["ok"] is False and needle in body["error"]
    assert nexus["calls"] == []


def test_nexus_infer_unknown_field_is_404(nexus):
    response = nexus["client"].post("/api/jwst-euclid/nexus/infer",
                                    data={"field_id": "nope"})
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# enforce_object_sizes — WCS-correct, nested crops
# ---------------------------------------------------------------------------

def _write(path, cube, header=None):
    fits.PrimaryHDU(np.asarray(cube, np.float32), header=header).writeto(path, overwrite=True)


def _sky(path, x, y):
    """RA/Dec of 0-based pixel (x, y) of a FITS file's celestial WCS."""
    with fits.open(path) as hdul:
        wcs = WCS(hdul[0].header).celestial
    ra, dec = wcs.all_pix2world([[x, y]], 0)[0]
    return float(ra), float(dec)


@pytest.mark.parametrize("lr_side", [55, 54, 57])
def test_enforce_object_sizes_keeps_the_sky_under_every_pixel(tmp_path, lr_side):
    lr_header = fx.wcs_header(150.0, 2.0, 0.1, (lr_side, lr_side), bunit="electron")
    sr_header = fx.wcs_header(150.0, 2.0, 0.05, (2 * lr_side, 2 * lr_side), bunit="electron")
    _write(tmp_path / "original_stack.fits", np.zeros((4, lr_side, lr_side)), lr_header)
    _write(tmp_path / "SR.fits", np.zeros((4, 2 * lr_side, 2 * lr_side)), sr_header)
    _write(tmp_path / "mean.fits", np.zeros((4, 2 * lr_side, 2 * lr_side)), sr_header)
    lr_before = _sky(tmp_path / "original_stack.fits", 5.0, 7.0)
    sr_before = _sky(tmp_path / "SR.fits", 12.0, 16.0)

    assert catalog_runner.enforce_object_sizes(str(tmp_path)) is True

    off = (lr_side - catalog_runner.EVAL_LR_SIZE) // 2
    lr_after = _sky(tmp_path / "original_stack.fits", 5.0 - off, 7.0 - off)
    sr_after = _sky(tmp_path / "SR.fits", 12.0 - 2 * off, 16.0 - 2 * off)
    assert lr_after == pytest.approx(lr_before, abs=1e-10)
    assert sr_after == pytest.approx(sr_before, abs=1e-10)
    for name, side in (("original_stack.fits", catalog_runner.EVAL_LR_SIZE),
                       ("SR.fits", catalog_runner.EVAL_HR_SIZE),
                       ("mean.fits", catalog_runner.EVAL_HR_SIZE)):
        with fits.open(tmp_path / name) as hdul:
            assert hdul[0].data.shape[-2:] == (side, side), name


def test_enforce_object_sizes_nests_the_sr_crop_in_the_lr_crop(tmp_path):
    """An even LR side (54 → 53: offset 0) must not shift the 2× grid by one SR
    pixel (108 → 106 centre-crops at 1): the SR stamp stays exactly 2× the LR
    stamp, so SR pixel (2i, 2j)'s corner is LR pixel (i, j)'s corner."""
    lr = np.arange(54 * 54, dtype=np.float32).reshape(1, 54, 54)
    sr = np.kron(lr, np.ones((1, 2, 2), np.float32))
    _write(tmp_path / "original_stack.fits", lr)
    _write(tmp_path / "SR.fits", sr)
    assert catalog_runner.enforce_object_sizes(str(tmp_path)) is True
    with fits.open(tmp_path / "original_stack.fits") as hdul:
        lr_c = np.asarray(hdul[0].data)
    with fits.open(tmp_path / "SR.fits") as hdul:
        sr_c = np.asarray(hdul[0].data)
    np.testing.assert_array_equal(sr_c[0, ::2, ::2], lr_c[0, :, :])


# ---------------------------------------------------------------------------
# eval model identity: STARFULL members + production combiner
# ---------------------------------------------------------------------------

@pytest.fixture
def regime(tmp_path, monkeypatch):
    """A production gate over fx.LABELS under ``<VIS_DIR>/ensemble/starfull``."""
    vis = tmp_path / "vis"
    monkeypatch.setattr(Config, "VIS_DIR", str(vis))
    gate_dir = vis / "ensemble" / "starfull" / "spatial_gate_combiner"
    save_spatial_gate(fx.uniform_gate(fx.LABELS), str(gate_dir))
    fp = combiner_artifact_fingerprint(str(vis / "ensemble" / "starfull"), "spatial_gate_combiner")
    return {"labels": list(fx.LABELS), "fingerprint": fp, "gate_dir": gate_dir}


class _Model:
    def __init__(self, labels, kind="spatial_gate"):
        self.member_labels = list(labels)
        self.n_members = len(labels)
        self.combiner_kind = kind


def test_model_identity_names_members_and_production_gate(regime):
    ident = catalog_runner.eval_model_identity(_Model(regime["labels"]))
    assert ident == {"member_labels": regime["labels"], "combiner_kind": "spatial_gate",
                     "combiner_fingerprint": regime["fingerprint"],
                     "run_labels": regime["labels"]}
    mean = catalog_runner.eval_model_identity(_Model(regime["labels"], kind=None))
    assert mean["combiner_kind"] is None and mean["combiner_fingerprint"] is None


def test_current_identity_without_loading_the_model(regime):
    ident = catalog_runner.current_eval_identity(labels=regime["labels"])
    assert ident["combiner_kind"] == "spatial_gate"
    assert ident["combiner_fingerprint"] == regime["fingerprint"]
    # A membership that lacks a member the gate reads reconstructs through the mean.
    other = catalog_runner.current_eval_identity(labels=regime["labels"][:2])
    assert other["combiner_kind"] is None and other["combiner_fingerprint"] is None
    assert other["member_labels"] == other["run_labels"] == regime["labels"][:2]


def _pruned_regime(regime, active_members=(0, 2)):
    gate = fx.uniform_gate(fx.LABELS)
    gate.active_members = tuple(active_members)
    save_spatial_gate(gate, str(regime["gate_dir"]))
    return combiner_artifact_fingerprint(str(regime["gate_dir"].parent),
                                         "spatial_gate_combiner")


def test_current_identity_of_a_pruned_gate_survives_new_and_unread_members(regime):
    """The probe keys on the gate's full fitted list and names the members it
    reads; members that join later, or unread members that leave, change
    nothing. Losing a read member falls back to the mean."""
    fp = _pruned_regime(regime)
    labels = regime["labels"]
    reads = [labels[0], labels[2]]
    ident = catalog_runner.current_eval_identity(labels=labels)
    assert ident == {"member_labels": labels, "combiner_kind": "spatial_gate",
                     "combiner_fingerprint": fp, "run_labels": reads}
    joined = catalog_runner.current_eval_identity(labels=[*labels, "4·psnr", "5·psnr"])
    assert joined == ident                                  # members added: unchanged
    unread_left = catalog_runner.current_eval_identity(labels=reads)
    assert unread_left == ident                             # an unread member archived
    read_left = catalog_runner.current_eval_identity(labels=labels[:2])
    assert read_left["combiner_kind"] is None               # a read member archived
    assert read_left["run_labels"] == labels[:2]


def test_archiving_an_unread_member_leaves_every_production_output_current(
        regime, monkeypatch):
    """The archive reconcile at the next Evaluate must not rewrite a gate
    whose reads all survive: its fingerprint — and so the identity every
    catalog, grouped, records and Sky output was made with — is unchanged."""
    fp = _pruned_regime(regime)
    labels = regime["labels"]
    before = catalog_runner.current_eval_identity(labels=labels)
    after_archive = [labels[0], labels[2]]                 # 2·psnr (unread) archived
    monkeypatch.setattr(ensemble_viz, "_regime_labels", lambda base, starless: after_archive)
    monkeypatch.setattr(ensemble_viz, "_combiner_payload_path",
                        lambda starless, kind=None: str(regime["gate_dir"].parent / "p.json"))
    assert ensemble_viz._reconcile_combiner_on_archives(
        str(regime["gate_dir"].parent), False, ["2"], "spatial_gate") is True
    assert combiner_artifact_fingerprint(str(regime["gate_dir"].parent),
                                         "spatial_gate_combiner") == fp
    assert catalog_runner.current_eval_identity(labels=after_archive) == before


def test_loaded_and_probed_identities_of_a_pruned_gate_agree(regime):
    """What a run writes (from the loaded model) equals what the next run
    probes from the registry, so nothing is re-run spuriously."""
    _pruned_regime(regime)
    active = [*regime["labels"], "4·psnr"]
    plan = production_plan(active)

    class _Ens:
        member_labels = list(plan.run_labels)
        n_members = len(plan.run_labels)

    model = EvalEnsemble(_Ens(), plan.combiner, "spatial_gate",
                         member_labels=plan.member_labels, joined=plan.joined)
    assert (catalog_runner.eval_model_identity(model)
            == catalog_runner.current_eval_identity(labels=active))


def test_reuse_requirements_follow_the_members_that_ran():
    ident = {"member_labels": ["1·psnr", "2·psnr", "3·psnr"], "combiner_kind": "spatial_gate",
             "combiner_fingerprint": "f" * 64, "run_labels": ["2·psnr"]}
    reuse = catalog_runner.reuse_requirements(ident)
    # One member ran: no disagreement cubes can exist, but the model is checked.
    assert reuse == {"require_disagreement": False, "member_labels": ident["member_labels"],
                     "identity": ident}
    both = catalog_runner.reuse_requirements({**ident, "run_labels": ["1·psnr", "3·psnr"]})
    assert both["require_disagreement"] is True
    legacy = catalog_runner.reuse_requirements({k: v for k, v in ident.items()
                                                if k != "run_labels"})
    assert legacy["require_disagreement"] is True          # all fitted members ran
    single = catalog_runner.reuse_requirements({"member_labels": ["1·psnr"],
                                                "combiner_kind": None})
    assert single == {"require_disagreement": False, "member_labels": None, "identity": None}


def test_one_member_run_reuses_an_object_without_disagreement_cubes(tmp_path):
    ident = {"member_labels": ["1·psnr", "2·psnr"], "combiner_kind": "spatial_gate",
             "combiner_fingerprint": "f" * 64, "run_labels": ["2·psnr"]}
    d = tmp_path / "obj"
    d.mkdir()
    for name in ("original_stack.fits", "SR.fits"):
        (d / name).write_bytes(b"x")
    reuse = catalog_runner.reuse_requirements(ident)
    assert not catalog_runner.can_reuse_eval_object(str(d), **reuse)   # no members.json
    catalog_runner.record_model_identity(str(d), ident)
    assert catalog_runner.can_reuse_eval_object(str(d), **reuse)
    with open(d / "members.json") as f:
        assert json.load(f)["run_labels"] == ["2·psnr"]
    refit = {**ident, "combiner_fingerprint": "0" * 64}
    assert not catalog_runner.can_reuse_eval_object(
        str(d), **catalog_runner.reuse_requirements(refit))


def test_objects_made_before_run_labels_stay_current_for_an_unpruned_gate(tmp_path, regime):
    """Landing the pruning change must not mass-invalidate cached objects:
    an old members.json (no run_labels) of the same unpruned gate reuses."""
    ident = catalog_runner.current_eval_identity(labels=regime["labels"])
    d = _object(tmp_path)
    old = {k: ident[k] for k in ("member_labels", "combiner_kind", "combiner_fingerprint")}
    (d / "members.json").write_text(json.dumps(old))
    assert catalog_runner.can_reuse_eval_object(
        str(d), **catalog_runner.reuse_requirements(ident))


def _object(tmp_path, name="obj"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    for f in ("original_stack.fits", "SR.fits", "std.fits", "pca0.fits"):
        (d / f).write_bytes(b"x")
    return d


def test_reuse_requires_the_production_combiner_identity(tmp_path, regime):
    d = _object(tmp_path)
    ident = catalog_runner.current_eval_identity(labels=regime["labels"])
    (d / "members.json").write_text(json.dumps({"member_labels": regime["labels"]}))
    # Same members but no recorded combiner → made by another combiner: stale.
    assert not catalog_runner.can_reuse_eval_object(
        str(d), require_disagreement=True, member_labels=regime["labels"], identity=ident)
    catalog_runner.record_model_identity(str(d), ident)
    assert catalog_runner.can_reuse_eval_object(
        str(d), require_disagreement=True, member_labels=regime["labels"], identity=ident)
    changed = {**ident, "combiner_fingerprint": "0" * 64}
    assert not catalog_runner.can_reuse_eval_object(
        str(d), require_disagreement=True, member_labels=regime["labels"], identity=changed)
    # The membership check of the older API still holds.
    assert catalog_runner.can_reuse_eval_object(
        str(d), require_disagreement=True, member_labels=regime["labels"])


def test_object_state_explains_staleness(tmp_path, regime):
    ident = catalog_runner.current_eval_identity(labels=regime["labels"])
    d = _object(tmp_path)
    assert catalog_runner.object_model_state(str(d), ident)["state"] == "unknown"
    (d / "members.json").write_text(json.dumps({"member_labels": ["105·psnr"]}))
    stale = catalog_runner.object_model_state(str(d), ident)
    assert stale["state"] == "stale" and "membership" in stale["reason"]
    assert stale["n_members"] == 1
    catalog_runner.record_model_identity(str(d), {**ident, "member_labels": regime["labels"]})
    current = catalog_runner.object_model_state(str(d), ident)
    assert current["state"] == "current" and current["combiner_kind"] == "spatial_gate"
    catalog_runner.record_model_identity(str(d), {**ident, "combiner_fingerprint": "f" * 64})
    assert "combiner" in catalog_runner.object_model_state(str(d), ident)["reason"]


def test_eval_catalog_object_records_the_model_identity(tmp_path, regime, monkeypatch):
    def fake_reconstruct(model, ra, dec, size, out_dir, **_kw):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "members.json"), "w") as f:
            json.dump({"member_labels": list(model.member_labels)}, f)
        return {"metrics": {"lr_total_e": 1.0, "sr_total_e": 1.0, "flux_ratio_sr_over_lr": 1.0}}

    monkeypatch.setattr(catalog_runner, "reconstruct_cutout_at", fake_reconstruct)
    obj = {"id": "a0", "ra": 1.0, "dec": 2.0, "grade": "A"}
    rec = catalog_runner.eval_catalog_object(
        _Model(regime["labels"]), obj, str(tmp_path), cutout_size=55,
        asinh_scale=None, checkpoint="")
    assert rec["ok"] is True
    with open(tmp_path / "a0" / "members.json") as f:
        recorded = json.load(f)
    assert recorded["member_labels"] == regime["labels"]
    assert recorded["combiner_kind"] == "spatial_gate"
    assert recorded["combiner_fingerprint"] == regime["fingerprint"]


def test_grouped_reuse_keys_on_starfull_members_and_combiner(tmp_path, regime, monkeypatch):
    """Objects made by an older combiner (same members) are re-run, current
    ones reused; the grouped runner never uses the starless members."""
    catalog = tmp_path / "lenses.csv"
    catalog.write_text("id,ra,dec,grade\na0,1.0,2.0,A\nb0,3.0,4.0,B\n")
    out = tmp_path / "run"
    ident = catalog_runner.current_eval_identity(labels=regime["labels"])
    for oid in ("a0", "b0"):
        d = out / oid
        d.mkdir(parents=True)
        for name, shape in (("original_stack.fits", (4, 53, 53)), ("SR.fits", (4, 106, 106)),
                            ("std.fits", (4, 106, 106)), ("pca0.fits", (4, 106, 106))):
            _write(d / name, np.ones(shape))
    catalog_runner.record_model_identity(str(out / "a0"), ident)
    (out / "b0" / "members.json").write_text(json.dumps({"member_labels": regime["labels"]}))

    seen_regime = []
    monkeypatch.setattr(grouped_runner, "regime_labels",
                        lambda _dir, starless: seen_regime.append(starless) or list(regime["labels"]))
    ran = []

    def fake_eval(model, obj, out_dir, **_kw):
        ran.append(obj["id"])
        return {"ok": False, "error": "stub"}

    monkeypatch.setattr(catalog_runner, "eval_catalog_object", fake_eval)
    monkeypatch.setattr(grouped_runner, "load_eval_ensemble", lambda *a, **k: _Model(regime["labels"]))
    grouped_runner.run_grouped_analysis(
        str(out), n=1, catalog_path=str(catalog), include_synthetic=False,
        include_galaxies=False, log=lambda m: None)
    assert ran == ["b0"]
    assert seen_regime and not any(seen_regime)


# ---------------------------------------------------------------------------
# mean.fits + the evaluation viewer collection
# ---------------------------------------------------------------------------

def test_disagreement_cubes_persist_the_member_mean_and_identity(tmp_path):
    rng = np.random.default_rng(1)
    members = rng.normal(5.0, 1.0, (4, 6, 6, 4)).astype(np.float32)
    ident = {"member_labels": ["1·psnr"] * 4, "combiner_kind": "spatial_gate",
             "combiner_fingerprint": "ab" * 32}
    disagreement.write_disagreement_cubes(str(tmp_path), members,
                                          member_labels=ident["member_labels"], identity=ident)
    with fits.open(tmp_path / "mean.fits") as hdul:
        mean = np.moveaxis(np.asarray(hdul[0].data), 0, -1)
    np.testing.assert_allclose(mean, members.mean(axis=0), rtol=1e-6)
    with open(tmp_path / "members.json") as f:
        assert json.load(f) == ident


def _eval_store(root, *, with_mean: bool):
    for sub in ("o_mean", "o_old"):
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        header = fx.wcs_header(150.0, 2.0, 0.1, (8, 8), bunit="electron")
        _write(d / "original_stack.fits", np.ones((4, 8, 8)), header)
        _write(d / "SR.fits", np.full((4, 16, 16), 2.0))
        _write(d / "pca0.fits", np.zeros((4, 16, 16)))
        (d / "disagreement.json").write_text(json.dumps({"pca_n": 1, "pca_amps": [1.0]}))
    if with_mean:
        _write(root / "o_mean" / "mean.fits", np.full((4, 16, 16), 3.0))
    with (root / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "ra", "dec", "grade", "ok", "out_subdir"])
        writer.writeheader()
        writer.writerow({"id": "o_mean", "ra": 150.0, "dec": 2.0, "grade": "A", "ok": "True",
                         "out_subdir": "o_mean"})
        writer.writerow({"id": "o_old", "ra": 150.0, "dec": 2.0, "grade": "B", "ok": "True",
                         "out_subdir": "o_old"})


def test_evaluation_collection_centres_the_movie_on_the_member_mean(tmp_path, monkeypatch):
    root = tmp_path / "eval_results"
    monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(root))
    _eval_store(root, with_mean=True)
    meta = viewer_data.get_meta("evaluation", {})
    assert meta["morph_base_tier"] == "mean"
    assert "mean" in [t["key"] for t in meta["tiers"]]
    by_id = {o["id"]: o for o in meta["objects"]}
    assert "mean" in by_id["o_mean"]["tiers"] and "mean" not in by_id["o_old"]["tiers"]
    cube, info = viewer_data.get_cube("evaluation", 0, "mean", {})
    assert float(cube.mean()) == pytest.approx(3.0)
    assert info["wcs"]["CD1_1"] == pytest.approx(-0.05 / 3600)
    # An object written before the mean was persisted centres on its SR.
    cube, info = viewer_data.get_cube("evaluation", 1, "mean", {})
    assert float(cube.mean()) == pytest.approx(2.0)
    assert "SR" in info["label"]


def test_evaluation_collection_without_any_mean_keeps_sr_centre(tmp_path, monkeypatch):
    root = tmp_path / "eval_results"
    monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(root))
    _eval_store(root, with_mean=False)
    meta = viewer_data.get_meta("evaluation", {})
    assert "morph_base_tier" not in meta
    assert "mean" not in [t["key"] for t in meta["tiers"]]


# ---------------------------------------------------------------------------
# STARFULL defaults: the cube cache and the synthetic evaluator
# ---------------------------------------------------------------------------

def test_cube_cache_defaults_to_starfull(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    assert ensemble_cube_cache._default_cubes_dir().endswith(
        os.path.join("ensemble", "starfull", "cubes"))
    seen = []
    monkeypatch.setattr(ensemble_cube_cache, "regime_labels",
                        lambda _d, starless: seen.append(starless) or [])
    d = tmp_path / "vis" / "ensemble" / "starfull" / "cubes"
    d.mkdir(parents=True)
    (d / "viz_index.json").write_text(json.dumps({"subset": "test", "indices": [0],
                                                   "member_labels": []}))
    assert ensemble_cube_cache.load_cached_member_stack(0, subset="test") is None
    assert seen == [False]


class _Gate:
    use_lr = False

    def apply_field(self, stack, lr=None):
        return np.asarray(stack, np.float32)[0] * 10.0


def test_synthetic_field_reconstruction_uses_the_production_combiner(monkeypatch):
    stack = np.stack([np.full((4, 4, 4), v, np.float32) for v in (1.0, 3.0)])
    labels = ["1·psnr", "2·psnr"]
    asked = []
    monkeypatch.setattr(synthetic_runner, "load_cached_member_stack",
                        lambda *a, **k: asked.append(k.get("active")) or stack)
    # Without a loaded model the cached stack goes through the production gate …
    gate_plan = ProductionPlan(tuple(labels), tuple(labels), _Gate())
    sr, members, got, kind, run = synthetic_runner.field_reconstruction(
        None, 0, np.zeros((2, 2, 4), np.float32), subset="test", log=lambda m: None,
        plan=gate_plan)
    assert kind == "spatial_gate" and got == labels and members is stack
    assert run == labels and asked == [labels]
    np.testing.assert_allclose(sr, 10.0)
    # … and only falls back to the plain mean when no current gate loads.
    sr, _members, _labels, kind, _run = synthetic_runner.field_reconstruction(
        None, 0, np.zeros((2, 2, 4), np.float32), subset="test", log=lambda m: None,
        plan=ProductionPlan(tuple(labels), tuple(labels), None))
    assert kind is None
    np.testing.assert_allclose(sr, 2.0)


def test_synthetic_field_reconstruction_reads_only_the_gate_members(tmp_path, monkeypatch):
    """A pruned gate reads only its members' cube files from the cache; a
    cache that lacks one of them is reported and never deleted."""
    cubes = tmp_path / "vis" / "ensemble" / "starfull" / "cubes"
    cubes.mkdir(parents=True)
    cache_labels = ["1·psnr", "2·psnr", "3·psnr"]
    for i in range(3):
        np.save(cubes / f"member{i}_00000.npy", np.full((4, 4, 4), float(i + 1), np.float32))
    (cubes / "viz_index.json").write_text(json.dumps(
        {"subset": "test", "indices": [0], "member_labels": cache_labels}))
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))

    class _Pick:
        use_lr = False

        def apply_field(self, stack, lr=None):
            return np.asarray(stack, np.float32).sum(axis=0)

    pruned = ProductionPlan(tuple(cache_labels), ("1·psnr", "3·psnr"), _Pick())
    sr, members, got, kind, run = synthetic_runner.field_reconstruction(
        None, 0, np.zeros((2, 2, 4), np.float32), subset="test", log=lambda m: None,
        plan=pruned)
    np.testing.assert_allclose(sr, 1.0 + 3.0)             # members 1 and 3 only
    assert members.shape[0] == 2 and run == ["1·psnr", "3·psnr"] and got == cache_labels
    logged = []
    joined = ProductionPlan(("1·psnr", "4·psnr"), ("1·psnr", "4·psnr"), _Pick())
    with pytest.raises(RuntimeError, match="no cached member stack"):
        synthetic_runner.field_reconstruction(
            None, 0, np.zeros((2, 2, 4), np.float32), subset="test", log=logged.append,
            plan=joined)
    assert any("lacks 1 member" in m and "4·psnr" in m for m in logged)
    assert (cubes / "viz_index.json").is_file()            # never purged


def test_synthetic_field_reconstruction_resolves_the_plan_from_the_registry(monkeypatch):
    """Neither a model nor a plan: the production plan comes from the
    registry's active STARFULL labels (no network) and picks the members."""
    labels = ["1·psnr", "2·psnr", "3·psnr"]
    seen: dict = {}

    def fake_plan(active):
        seen["active"] = list(active)
        return ProductionPlan(tuple(active), ("1·psnr", "3·psnr"), _Gate())

    monkeypatch.setattr(synthetic_runner, "regime_labels", lambda base, starless: labels)
    monkeypatch.setattr(synthetic_runner, "production_plan", fake_plan)
    monkeypatch.setattr(synthetic_runner, "load_cached_member_stack",
                        lambda *a, **k: seen.update(asked=k.get("active"))
                        or np.ones((2, 4, 4, 4), np.float32))
    sr, _members, got, kind, run = synthetic_runner.field_reconstruction(
        None, 0, np.zeros((2, 2, 4), np.float32), subset="test", log=lambda m: None)
    assert seen["active"] == labels and seen["asked"] == ["1·psnr", "3·psnr"]
    assert got == labels and run == ["1·psnr", "3·psnr"] and kind == "spatial_gate"
    np.testing.assert_allclose(sr, 10.0)


def test_synthetic_field_reconstruction_prefers_the_loaded_model(monkeypatch):
    stack = np.ones((2, 4, 4, 4), np.float32)

    class Model:
        member_labels = ["1·psnr", "2·psnr"]
        n_members = 2
        combiner_kind = "spatial_gate"

        def combine(self, members, lr):
            return np.full(members.shape[1:], 7.0, np.float32)

    monkeypatch.setattr(synthetic_runner, "load_cached_member_stack", lambda *a, **k: stack)
    monkeypatch.setattr(synthetic_runner, "cached_member_labels", lambda *a, **k: Model.member_labels)
    sr, _m, _l, kind, _run = synthetic_runner.field_reconstruction(
        Model(), 0, np.zeros((2, 2, 4), np.float32), subset="test", log=lambda m: None)
    np.testing.assert_allclose(sr, 7.0)
    assert kind == "spatial_gate"


# ---------------------------------------------------------------------------
# /api/evaluation — staleness per object, the object card, JSON errors
# ---------------------------------------------------------------------------

@pytest.fixture
def evalapi(tmp_path, monkeypatch, regime):
    root = tmp_path / "eval_results"
    monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(root))
    _eval_store(root, with_mean=True)
    ident = catalog_runner.current_eval_identity(labels=regime["labels"])
    catalog_runner.record_model_identity(str(root / "o_mean"), ident)
    (root / "o_old" / "members.json").write_text(json.dumps({"member_labels": ["105·psnr"]}))
    monkeypatch.setattr(catalog_runner, "current_eval_identity",
                        lambda *a, **k: dict(ident))
    app = web_app.create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield {"client": client, "root": root, "identity": ident}


def test_runs_rows_carry_model_state_field_and_viewer_id(evalapi):
    body = evalapi["client"].get("/api/evaluation/runs").get_json()
    rows = {r["id"]: r for r in body["rows"]}
    assert rows["o_mean"]["state"] == "current"
    assert rows["o_old"]["state"] == "stale"
    assert "membership" in rows["o_old"]["state_reason"]
    assert rows["o_mean"]["viewer_id"] == "o_mean"
    assert rows["o_mean"]["field"] in ("EDF-N", "EDF-S", "EDF-F", "LDN1641", None)
    assert body["counts"] == {"current": 1, "stale": 1, "unknown": 0}
    assert body["current"]["combiner_kind"] == "spatial_gate"
    assert body["current"]["n_members"] == len(fx.LABELS)
    assert body["groups"] == {"A": 1, "B": 1}


def test_viewer_id_is_the_evaluation_viewer_object_id(evalapi):
    """A manifest id sanitised into its out_subdir: the row's viewer_id and the
    evaluation collection's object id are both the subdir (the table row and
    the viewer key on the same field)."""
    root = evalapi["root"]
    with (root / "manifest.csv").open("a", newline="") as handle:
        csv.writer(handle).writerow(["lens A/1", 150.0, 2.0, "A", "True", "o_old"])
    rows = evalapi["client"].get("/api/evaluation/runs").get_json()["rows"]
    row = next(r for r in rows if r["id"] == "lens A/1")
    meta = viewer_data.get_meta("evaluation", {})
    ids = [o["id"] for o in meta["objects"]]
    assert row["viewer_id"] == "o_old"
    assert row["viewer_id"] in ids and "lens A/1" not in ids


def test_object_card_shows_provenance(evalapi):
    body = evalapi["client"].get("/api/evaluation/objects/o_mean").get_json()
    assert body["id"] == "o_mean" and body["state"] == "current"
    names = {f["name"] for f in body["files"]}
    assert {"original_stack.fits", "SR.fits", "mean.fits", "members.json"} <= names
    assert body["members"]["combiner_kind"] == "spatial_gate"
    assert body["disagreement"]["pca_n"] == 1
    assert body["realtile"] == "eval/o_mean"
    assert body["downloads"]["SR"] == "/eval-files/o_mean/SR.fits"


def test_evaluation_errors_are_json(evalapi):
    missing = evalapi["client"].get("/api/evaluation/objects/nope")
    assert missing.status_code == 404 and "nope" in missing.get_json()["error"]
    bad = evalapi["client"].get("/api/evaluation/runs", query_string={"run": "../x"})
    assert bad.status_code == 400 and bad.get_json()["error"]
    escape = evalapi["client"].get("/api/evaluation/objects/..%2F..%2Fetc")
    assert escape.status_code in (400, 404) and escape.get_json()["error"]
