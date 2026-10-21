"""Universal models (helpers/universal_models.py + the ensemble_viz read side):
training time from the training log (rollbacks counted, never deduped), the
job ledger's recipes and hardware, the FASRC-only rows, the payload of
``GET /ensemble/universal.json`` (scores, combiners' share-weighted hours),
the score job (Leaderboard metric, own store, reuse by fingerprint, gate
weights per variant) and the pull job (registered BEFORE its rsync, never
an ensemble member). Fakes only: no TensorFlow model, no SSH."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from euclid_polish import ensemble_registry as er
from euclid_polish.config import Config
from euclid_polish.ensemble import member_fingerprint
from euclid_polish.eval.combiner import combiner_artifact_fingerprint
from euclid_polish.eval.knee_psnr import KNEE_GRID_E, integrated_psnr, knee_psnr, stretched_truth
from euclid_polish.eval.spatial_gate import MIX_LINEAR, SpatialGateCombiner, save_spatial_gate
from euclid_polish.eval.spatial_gate_fit import init_params
from euclid_polish.observability import JobLog, JobRecord
from euclid_polish.tracking import default_store
from euclid_polish.training.target_blur import blur_target_array
from euclid_polish.web import fasrc_jobs
from euclid_polish.web.helpers import ensemble_viz as ev
from euclid_polish.web.helpers import purge_requests
from euclid_polish.web.helpers import universal_models as um
from euclid_polish.web.helpers import universal_spectra as us
from euclid_polish.web.remote import STATE

BANDS = ("VIS", "Y_E", "J_E", "H_E")
KNEES6 = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
FIXED = {"asinh_knees": KNEES6, "knee_loss": "balanced", "output_knee": 10.0,
         "num_res_blocks": 32, "batch_size": 4, "crops_per_field": 8, "bootstrap": 0.7,
         "seed": 7, "commit": "abc1234", "created_at": "2026-09-24T00:00:00+00:00",
         "target_steps": 4000}
LEARNED = {"asinh_knees": KNEES6, "knee_loss": "balanced",
           "learned_output_knee": {"init_e": 10.0, "min_e": 0.1, "max_e": 1e4},
           "num_res_blocks": 16, "batch_size": 4, "crops_per_field": 16, "bootstrap": None,
           "seed": 2898473870, "jit_compile": True, "mixed_precision": True,
           "target_steps": 4000}
SINGLE_KNEE = {"asinh_knee": 10.0, "num_res_blocks": 32, "target_steps": 2000}
LOG_HEADER = ("step,wall_time,loss,psnr_stretched,psnr_raw,psnr_vis,psnr_y_e,psnr_j_e,"
              "psnr_h_e,gnorm_avg,gnorm_max,clip_norm,duration_s,combined_loss,is_baseline\n")
T0 = 1_790_000_000.0


class _Cap:
    def __init__(self):
        self.ticks: list[tuple] = []

    def tick(self, *args, **_kwargs):
        self.ticks.append(args)


def _log_text(rows) -> str:
    """``rows`` = (step, wall, duration or None, baseline)."""
    out = [LOG_HEADER]
    for step, wall, duration, baseline in rows:
        out.append(f"{step},{wall},0.5,40,90,39,41,40,40,3.0,9.0,5,"
                   f"{'' if duration is None else duration},0.1,{'1' if baseline else ''}\n")
    return "".join(out)


def _member(base, n, *, origin=None, rows=None, ckpt="ckpt-5"):
    d = os.path.join(base, f"member_{n:02d}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "checkpoint"), "w") as f:
        f.write(f'model_checkpoint_path: "{ckpt}"\n')
    with open(os.path.join(d, f"{ckpt}.index"), "wb") as f:
        f.write(b"idx")
    rows = rows if rows is not None else [(1000, T0, 1800.0, False), (2000, T0 + 1800, 1800.0, False)]
    with open(os.path.join(d, "training_log.csv"), "w") as f:
        f.write(_log_text(rows))
    with open(os.path.join(d, "origin.json"), "w") as f:
        json.dump(origin if origin is not None else SINGLE_KNEE, f)
    return d


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR", str(tmp_path / "ckpt/wdsr"))
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    monkeypatch.setattr(Config, "TRACKING_DIR", str(tmp_path / "tracking"))
    monkeypatch.setattr(ev, "_sky_records_local_dir", lambda: None)
    monkeypatch.setattr(ev, "infer_checkpoint_num_res_blocks", lambda d: 32)
    monkeypatch.setattr(fasrc_jobs, "JOBLOG", JobLog(str(tmp_path / "joblog.csv")))
    return tmp_path


def _regime(env):
    d = os.path.join(str(env), "vis", "ensemble", "starfull")
    os.makedirs(d, exist_ok=True)
    return d


def _job(jobid, names, *, mode="add", state="COMPLETED", partition="gpu", spec=None,
         submitted="2026-10-10T00:00:00Z", ended="2026-10-10T05:00:00", **params):
    key = "member_names" if mode in ("add", "fork") else "members"
    payload = {"mode": mode, key: ",".join(names), **params}
    if spec is not None:
        payload["member_spec"] = json.dumps(spec)
    fasrc_jobs.JOBLOG.record_submission(JobRecord(
        jobid=jobid, step_id="ensemble_train", label="Train", submitted_at=submitted,
        partition=partition, state=state, ended_at=ended, params_json=json.dumps(payload)))


SPEC_LEARNED = {"loss": "l2", "asinh_knees": KNEES6, "knee_loss": "balanced", "output_knee": 10,
                "learn_output_knee": True, "num_res_blocks": 16}
SPEC_FIXED = {"loss": "l2", "asinh_knees": KNEES6, "knee_loss": "balanced", "output_knee": 10,
              "num_res_blocks": 32}
SPEC_HEADS = {"loss": "l2", "asinh_knees": KNEES6, "knee_loss": "balanced", "num_res_blocks": 32}


# ---- training time ----------------------------------------------------------- #

def test_train_time_counts_every_window_including_rollbacks(env, tmp_path):
    rows = [(1000, T0, 100.0, False), (2000, T0 + 100, 110.0, False),
            (3000, T0 + 210, 120.0, False),
            (2000, T0 + 300, None, True),                 # resume marker: no training
            (2000, T0 + 400, 130.0, False),               # rolled back → re-trained
            (3000, T0 + 530, 140.0, False), (4000, T0 + 670, 150.0, False)]
    d = _member(str(tmp_path / "m"), 5, rows=rows)
    t = um.train_time(d)
    assert t["hours"] == pytest.approx(round(750.0 / 3600, 4))
    assert t["rows"] == 6 and t["rollback_rows"] == 2 and t["steps"] == 4000
    assert t["first_wall"].startswith("2026-") and t["first_wall"] < t["last_wall"]
    # the same log inside an archived member's tracking zip
    z = tmp_path / "ensemble-member-05.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("training_log.csv", _log_text(rows))
        zf.writestr("checkpoint", "x")
    assert um.train_time(str(z)) == t
    missing = um.train_time(str(tmp_path / "nowhere"))
    assert missing["hours"] is None and missing["rows"] == 0


# ---- ledger: recipes, universal names, hardware, FASRC rows ------------------ #

def test_ledger_recipes_merge_run_wide_params_with_the_member_spec(env):
    _job("51", ["member_206"], spec=[SPEC_LEARNED], steps="100000")
    _job("52", ["member_207", "member_208"], steps="200000",
         spec=[{**SPEC_LEARNED, "num_res_blocks": 64}, {**SPEC_LEARNED, "num_res_blocks": 32}])
    _job("53", ["member_197", "member_203"], spec=[SPEC_HEADS, {"asinh_knee": 0.3}])
    _job("55", ["member_214"], partition="gpu_test", spec=[SPEC_LEARNED], xla="1", bf16="1",
         batch_size="4", crops_per_field="8")
    _job("56", ["member_220"], asinh_knees="0.1,1,10", output_knee="10", steps="5000")
    recipes = ev.ledger_member_recipes()
    assert recipes["member_207"]["num_res_blocks"] == 64
    assert recipes["member_208"]["num_res_blocks"] == 32 and recipes["member_208"]["learn_output_knee"]
    assert recipes["member_214"]["xla"] and recipes["member_214"]["bf16"]
    assert recipes["member_214"]["crops_per_field"] == 8
    assert recipes["member_220"]["asinh_knees"] == [0.1, 1.0, 10.0]
    assert recipes["member_220"]["output_knee"] == 10.0 and recipes["member_220"]["steps"] == 5000
    assert ev.ledger_universal_names() == {"member_206", "member_207", "member_208",
                                           "member_214", "member_220"}


def test_hardware_and_ledger_jobs_follow_every_job_naming_the_member(env):
    _job("61", ["member_209"], spec=[SPEC_LEARNED], submitted="2026-10-10T00:00:00Z")
    _job("62", ["member_209"], mode="continue", partition="gpu_test",
         submitted="2026-10-11T00:00:00Z")
    _job("63", ["member_214"], partition="gpu_test", spec=[SPEC_LEARNED],
         submitted="2026-10-12T00:00:00Z")
    jobs = um.ledger_jobs("209")
    assert [j["jobid"] for j in jobs] == ["62", "61"] and jobs[0]["task_index"] == 0
    assert um.hardware("member_209") == {"partition": "mixed", "label": "mixed", "jobs": ["62", "61"]}
    assert um.hardware("214") == {"partition": "gpu_test", "label": "A100 MIG slice", "jobs": ["63"]}
    assert um.hardware("member_99") == {"partition": None, "label": None, "jobs": []}


def test_a_fork_inherits_its_sources_knees_so_a_fork_of_a_universal_model_is_one(env):
    """A fork takes its source's knees, output knee and depth (model.py's
    fork path), whatever its own params say: resolved from the source's
    local origin.json, else its ledger recipe (through chains of forks)."""
    base = ev.ensemble_dir()
    _member(base, 196, origin=FIXED)                       # an active universal member
    _member(base, 170)                                     # single knee
    _job("90", ["member_211"], spec=[SPEC_LEARNED], submitted="2026-10-09T00:00:00Z")
    _job("91", ["member_300"], mode="fork", fork_from="member_211", spec=[{"loss": "l1"}],
         submitted="2026-10-10T00:00:00Z")
    _job("92", ["member_301"], mode="fork", fork_from="196", submitted="2026-10-10T01:00:00Z")
    _job("93", ["member_302"], mode="fork", fork_from="member_300",
         submitted="2026-10-10T02:00:00Z")                 # a fork of a fork
    _job("94", ["member_303"], mode="fork", fork_from="member_170",
         spec=[{"asinh_knees": KNEES6, "output_knee": 10}], submitted="2026-10-10T03:00:00Z")
    _job("95", ["member_304"], mode="fork", fork_from="member_305",
         submitted="2026-10-10T04:00:00Z")                 # a cycle: never resolves
    _job("96", ["member_305"], mode="fork", fork_from="member_304",
         submitted="2026-10-10T05:00:00Z")
    _job("97", ["member_306"], mode="fork", fork_from="member_999",
         submitted="2026-10-10T06:00:00Z")                 # unknown source
    recipes = ev.ledger_member_recipes()
    r300 = recipes["member_300"]
    assert r300["asinh_knees"] == KNEES6 and r300["learn_output_knee"] is True
    assert r300["num_res_blocks"] == 16 and r300["fork_from"] == "member_211"
    r301 = recipes["member_301"]                           # from 196's origin.json
    assert r301["output_knee"] == 10.0 and r301["learn_output_knee"] is False
    assert r301["num_res_blocks"] == 32
    assert recipes["member_302"]["learn_output_knee"] is True
    assert recipes["member_303"]["asinh_knees"] is None    # 170 is single-knee
    assert recipes["member_304"]["asinh_knees"] is None
    assert ev.ledger_universal_names() == {"member_211", "member_300", "member_301",
                                           "member_302"}
    # the per-job recipe resolves the same way
    fork = next(j for j in ev.training_jobs() if j["jobid"] == "91")
    assert er.is_universal_origin(ev.job_member_recipe(fork, 0))
    # and the pulls / mirror keep the forks out of the ensemble
    assert ev.unregistered_universal_names(base) == ["member_211", "member_300",
                                                     "member_301", "member_302"]
    assert [r["name"] for r in um.fasrc_universal_rows()] == [
        "member_211", "member_300", "member_301", "member_302"]


def test_names_a_gate_or_cube_manifest_reads_are_never_fasrc_only_universal(env):
    """195/196 are universal by recipe but ensemble members: when their dirs
    vanish (the registry drops them from active), a gate or cube manifest
    that reads them keeps them out of the FASRC-only universal set, so the
    ensemble pull restores them as members."""
    base = ev.ensemble_dir()
    _member(base, 170)
    er.load_registry(base)
    _job("70", ["member_195", "member_196"], spec=[SPEC_FIXED, SPEC_FIXED])
    _job("71", ["member_211"], spec=[SPEC_LEARNED])
    assert ev.unregistered_universal_names(base) == ["member_195", "member_196", "member_211"]
    regime = _regime(env)
    _gate(regime, "spatial_gate_backup_20260101-000000", ["170·psnr", "196·psnr"])
    os.makedirs(os.path.join(regime, "cubes_validate"))
    with open(os.path.join(regime, "cubes_validate", "viz_index.json"), "w") as f:
        json.dump({"member_labels": ["170·psnr", "195·psnr"]}, f)
    assert ev.gate_and_cube_member_names() == {"member_170", "member_195", "member_196"}
    assert ev.unregistered_universal_names(base) == ["member_211"]
    assert [r["name"] for r in um.fasrc_universal_rows()] == ["member_211"]


def test_fasrc_rows_list_only_universal_models_that_live_only_on_fasrc(env):
    base = ev.ensemble_dir()
    er.add_universal_entry(base, "211")
    _member(base, 196, origin=FIXED)
    _member(base, 150)
    er.load_registry(base)
    er.archive_member_entry(base, "member_150", zip_path="models/z.zip", commit=None)
    _job("70", ["member_196"], spec=[SPEC_FIXED], submitted="2026-09-24T00:00:00Z")
    _job("71", ["member_150"], spec=[SPEC_LEARNED], submitted="2026-09-25T00:00:00Z")
    _job("72", ["member_206"], spec=[SPEC_LEARNED], steps="100000",
         submitted="2026-10-08T00:00:00Z", ended="2026-10-08T03:00:00")
    _job("73", ["member_209"], spec=[SPEC_LEARNED], steps="500000", state="TIMEOUT",
         submitted="2026-10-10T00:00:00Z")
    _job("74", ["member_209"], mode="continue", continue_basis="target",
         target_steps="500000", submitted="2026-10-10T12:00:00Z", ended="2026-10-10T15:00:00")
    _job("75", ["member_211"], spec=[SPEC_LEARNED], submitted="2026-10-11T00:00:00Z")
    _job("76", ["member_218"], spec=[SPEC_LEARNED], steps="2000000", state="",
         ended="", submitted="2026-10-13T00:00:00Z")
    _job("77", ["member_203"], spec=[{"asinh_knee": 0.3}], submitted="2026-10-13T01:00:00Z")
    _job("78", ["member_220"], spec=[SPEC_LEARNED], state="OUT_OF_MEMORY",
         submitted="2026-10-13T02:00:00Z")
    rows = um.fasrc_universal_rows()
    # 211 is registered but not on disk (a failed pull, or the mirror removed
    # it): listed again so it can be pulled
    assert [r["name"] for r in rows] == ["member_206", "member_209", "member_211",
                                         "member_218", "member_220"]
    r206, r209, r211, r218, r220 = rows
    assert r206 == {"name": "member_206", "jobids": ["72"], "state": "COMPLETED",
                    "num_res_blocks": 16, "steps": 100000, "knee": "learned",
                    "submitted_at": "2026-10-08T00:00:00Z", "ended_at": "2026-10-08T03:00:00",
                    "partition": "gpu", "finished": True, "pullable": True,
                    "registered": False}
    assert r209["jobids"] == ["74", "73"] and r209["steps"] == 500000
    assert r209["submitted_at"] == "2026-10-10T00:00:00Z" and r209["finished"] is True
    assert r209["pullable"] is True                       # the continue COMPLETED
    assert r211["registered"] is True and r211["pullable"] is True
    assert r218["state"] is None and r218["finished"] is False and r218["pullable"] is False
    # an OOM job is final but leaves no checkpoint: not offered for a pull
    assert r220["finished"] is True and r220["pullable"] is False
    # once 211 is on disk with a checkpoint it is a local model, not a FASRC row
    _member(base, 211, origin=LEARNED)
    assert "member_211" not in [r["name"] for r in um.fasrc_universal_rows()]


# ---- payload ------------------------------------------------------------------ #

def test_payload_answers_on_an_empty_ensemble(env):
    p = um.universal_payload()
    assert p["available"] is False and p["models"] == [] and p["fasrc"] == []
    assert p["combiners"] == [] and p["identity"] is None and p["production"] is None
    assert p["comparable"] is False
    assert p["knees"] == list(KNEE_GRID_E) and p["bands"] == list(BANDS)
    assert p["integration"] == {"from_e": 0.1, "to_e": 10000.0}
    assert not os.path.exists(os.path.join(str(env), "vis", "ensemble", "starfull", "universal"))
    assert um.universal_payload(starless=True)["notes"] == ["universal models are STARFULL only"]


def _gate(regime, name, labels, *, active=None, fit_seconds=None):
    params = init_params(len(labels), 4, 8, False, [0, 0, 0, 0], seed=1)
    meta = {"loss": "test", "complete": True}
    if fit_seconds is not None:
        meta["fit_seconds"] = fit_seconds
    comb = SpatialGateCombiner(list(labels), params, width=8, use_lr=False,
                               mix_space=MIX_LINEAR, fit_meta=meta,
                               active_members=None if active is None else tuple(active))
    d = os.path.join(regime, name)
    save_spatial_gate(comb, d)
    return d


def _diag(rows_usage, rows_source, rows_bins):
    return {"available": True, "schema": 1, "brightness_names": ["sky", "core"],
            "usage": dict.fromkeys(BANDS, rows_usage),
            "usage_source": dict.fromkeys(BANDS, rows_source),
            "usage_by_brightness": dict.fromkeys(BANDS, rows_bins)}


def _world(env):
    """196 (active, fixed knee), 211 (registered universal, learned knee),
    170 (active, single knee), 150 (archived: hours from its zip), a scored
    store, the knee payload, the production gate + payload, two variants."""
    base = ev.ensemble_dir()
    er.add_universal_entry(base, "211")                 # registered BEFORE its dir lands
    _member(base, 170, rows=[(1000, T0, 3600.0, False), (2000, T0 + 3600, 3600.0, False)])
    _member(base, 196, origin=FIXED, rows=[(1000, T0, 3600.0, False)])
    d211 = _member(base, 211, origin=LEARNED,
                   rows=[(2000, T0, 3600.0, False), (4000, T0 + 3600, 3600.0, False),
                         (4000, T0 + 7200, 1800.0, False)])
    _member(base, 150)
    er.load_registry(base)
    er.archive_member_entry(base, "member_150", zip_path="models/ensemble-member-150.zip",
                            commit=None)
    shutil.rmtree(os.path.join(base, "member_150"))
    models = os.path.join(str(env), "tracking", "current", "models")
    os.makedirs(models)
    with zipfile.ZipFile(os.path.join(models, "ensemble-member-150.zip"), "w") as zf:
        zf.writestr("training_log.csv", _log_text([(1000, T0, 1800.0, False)]))
    reg = er.load_registry(base)
    assert reg["active"] == ["member_170", "member_196"] and reg["universal"] == ["member_211"]

    _job("80", ["member_196"], spec=[SPEC_FIXED], submitted="2026-09-24T00:00:00Z")
    _job("81", ["member_211"], spec=[SPEC_LEARNED], submitted="2026-10-10T00:00:00Z")
    _job("82", ["member_211"], mode="continue", partition="gpu_test",
         submitted="2026-10-11T00:00:00Z")
    _job("83", ["member_218"], spec=[SPEC_LEARNED], state="RUNNING", ended="",
         submitted="2026-10-13T00:00:00Z")

    regime = _regime(env)
    identity = {"records_fp": "fp", "subset": "test", "indices": [0, 1],
                "target_psf_fwhm_arcsec": 0.066, "knees": list(KNEE_GRID_E)}
    with open(os.path.join(regime, "ensemble_knee_psnr.json"), "w") as f:
        json.dump({"available": True, "knees": list(KNEE_GRID_E), "bands": list(BANDS),
                   "identity": {**identity, "member_labels": ["170·psnr", "196·psnr"]},
                   "models": [
                       {"id": "member_0", "kind": "member", "label": "170·psnr",
                        "integrated": [58, 59, 60, 61], "psnr": [[50] * 4] * 21},
                       {"id": "ensemble_mean", "kind": "mean", "label": "ensemble mean",
                        "integrated": [59, 60, 61, 62], "psnr": [[51] * 4] * 21},
                       {"id": "spatial_gate", "kind": "combiner", "label": "spatial gate",
                        "integrated": [60, 61, 62, 63], "psnr": [[52] * 4] * 21}]}, f)
    store = os.path.join(regime, "universal")
    os.makedirs(store)
    with open(os.path.join(store, "index.json"), "w") as f:
        json.dump({"schema": 1, "identity": identity, "models": {
            "member_211": {"fingerprint": member_fingerprint(d211),
                           "integrated": [61, 62, 62, 63], "mean": 62.0, "n_fields": 2,
                           "scored_at": "2026-10-13T00:00:00+00:00", "seconds": 9.0,
                           "learned_knees": [5.0, 6.0, 7.0, 8.0], "served_step": 3000},
            "member_196": {"fingerprint": "ckpt-1:1:1", "integrated": [59, 60, 60, 61],
                           "mean": 60.0, "n_fields": 2, "scored_at": "2026-10-01T00:00:00+00:00",
                           "seconds": 9.0, "learned_knees": None, "served_step": 1000}}}, f)
    curves = np.full((2, len(KNEE_GRID_E), 4), 62.0)
    np.savez(os.path.join(store, "member_211.npz"), fields=np.array([0, 1]),
             knees=np.array(KNEE_GRID_E), curves=curves)

    prod = _gate(regime, "spatial_gate_combiner", ["170·psnr", "196·psnr"], fit_seconds=3600.0)
    diag = _diag([0.75, 0.25], [0.5, 0.5], [[0.9, 0.1], [0.2, 0.8]])
    diag["artifact_fp"] = combiner_artifact_fingerprint(regime, "spatial_gate_combiner")
    with open(os.path.join(regime, "spatial_gate_combiner_evals.json"), "w") as f:
        json.dump({"available": True, "member_labels": ["170·psnr", "196·psnr"],
                   "band_names": list(BANDS), "gate_diagnostics": diag}, f)
    _gate(regime, "spatial_gate_small", ["170·psnr", "150·psnr"], active=[0])
    small = _diag([0.6, 0.4], [0.7, 0.3], [[0.5, 0.5], [0.4, 0.6]])
    with open(os.path.join(regime, "spatial_gate_small_evals.json"), "w") as f:
        json.dump({"artifact_fp": combiner_artifact_fingerprint(regime, "spatial_gate_small"),
                   "gate_diagnostics": small}, f)
    _gate(regime, "spatial_gate_old", ["150·psnr", "170·psnr"])
    _gate(regime, "spatial_gate_backup_20260101-000000", ["170·psnr"])
    with open(os.path.join(regime, "spatial_gate_comparison.json"), "w") as f:
        json.dump({"id": "20990101-000000", "created": "2099-01-01T00:00:00+00:00",
                   "knee": {"methods": {"gate:spatial_gate_small": {
                       "integrated": [58, 59, 59, 60], "psnr": [[1] * 4] * 21}}}}, f)
    return base, regime, d211, prod


def test_payload_joins_models_scores_hardware_and_combiner_hours(env):
    _world(env)
    p = um.universal_payload()
    assert p["available"] is True
    assert p["production"] == {"integrated": [60.0, 61.0, 62.0, 63.0], "mean": 61.5, "stale": True}
    assert p["mean_model"]["mean"] == 60.5 and p["comparable"] is True
    assert p["identity"]["indices"] == [0, 1]
    m196, m211 = p["models"]
    assert m196["name"] == "member_196" and m196["class"] == "member"
    assert m196["spec"] == "member:member_196" and m196["knee"] == "fixed"
    assert m196["output_knee"] == 10.0 and m196["num_res_blocks"] == 32
    assert m196["score"]["stale"] is True             # its checkpoint changed since
    # a stale score keeps the step it was made at; legacy entries have no
    # scoring-time training hours/steps
    assert m196["learned_knees"] is None and m196["served_step"] == 1000
    assert m196["scored_hours"] is None and m196["scored_steps"] is None
    assert m196["vs_production_db"] == pytest.approx(-1.5)
    assert m196["hardware"] == {"partition": "gpu", "label": "A100", "jobs": ["80"]}
    assert m211["spec"] == "universal:member_211" and m211["class"] == "universal"
    assert m211["knee"] == "learned" and m211["output_knee"] is None
    assert m211["learned_knees"] == [5.0, 6.0, 7.0, 8.0] and m211["served_step"] == 3000
    assert m211["score"] == {"integrated": [61.0, 62.0, 62.0, 63.0], "mean": 62.0, "n_fields": 2,
                             "stale": False, "scored_at": "2026-10-13T00:00:00+00:00"}
    assert m211["vs_production_db"] == pytest.approx(0.5)
    assert m211["train_hours"] == pytest.approx(2.5) and m211["train_rows"] == 3
    assert m211["steps"] == 4000 and m211["jit_compile"] and m211["mixed_precision"]
    assert m211["hardware"]["label"] == "mixed" and m211["hardware"]["jobs"] == ["82", "81"]
    assert m211["fingerprint"].startswith("ckpt-5:3:")
    assert [r["name"] for r in p["fasrc"]] == ["member_218"]

    ids = [c["id"] for c in p["combiners"]]
    assert ids == ["production", "gate:old", "gate:small"]     # backups left out
    prod, old, small = p["combiners"]
    assert prod["spec"] == "production" and prod["score_source"] == "knee-psnr"
    assert prod["integrated"] == [60, 61, 62, 63] and prod["stale"] is True
    assert prod["comparable"] is True                 # the knee payload's identity
    assert prod["weights_source"] == "production" and prod["fit_hours"] == pytest.approx(1.0)
    by = {m["label"]: m for m in prod["members"]}
    assert by["170·psnr"]["hours"] == pytest.approx(2.0) and by["196·psnr"]["hours"] == pytest.approx(1.0)
    assert by["170·psnr"]["max_weight"] == pytest.approx(0.9)
    assert by["196·psnr"]["max_weight"] == pytest.approx(0.8)
    assert by["170·psnr"]["mean_weight"] == pytest.approx(0.75)
    assert prod["hours"] == {"max": pytest.approx(2.6), "mean": pytest.approx(1.75),
                             "sum": pytest.approx(3.0)}
    # a pruned variant: only the member it reads counts; its knee row from the
    # newest compare report that scored it
    assert small["reads"] == ["170·psnr"] and small["score_source"] == "compare"
    assert small["integrated"] == [58, 59, 59, 60] and small["mean"] == 59.0
    assert small["stale"] is False and small["weights_source"] == "variant"
    assert small["comparable"] is False               # a legacy report: no identity
    assert small["hours"] == {"max": pytest.approx(2.0 * 0.7), "mean": pytest.approx(1.2),
                              "sum": pytest.approx(2.0)}
    # an unscored variant reading an archived member: hours from its zip,
    # weights unavailable with the reason
    assert old["integrated"] is None and old["score_source"] is None
    assert old["weights_source"] is None and "validate" in old["weights_reason"]
    assert old["hours"] == {"max": None, "mean": None, "sum": pytest.approx(2.5)}
    assert old["missing_hours"] == [] and old["comparable"] is False
    assert p["weights_pending"] == []                 # no validate cubes to measure on
    assert any("stale" in n for n in p["notes"])


def test_payload_flags_stale_scores_when_the_test_fields_change(env):
    _base, regime, _d211, _prod = _world(env)
    path = os.path.join(regime, "ensemble_knee_psnr.json")
    knee = json.load(open(path))
    knee["identity"]["indices"] = [0, 1, 2]
    json.dump(knee, open(path, "w"))
    p = um.universal_payload()
    m211 = next(m for m in p["models"] if m["name"] == "member_211")
    assert m211["score"]["stale"] is True
    assert m211["vs_production_db"] is None           # other fields than the knee payload
    assert p["comparable"] is False
    assert any("other fields" in n for n in p["notes"])


def test_variant_weights_of_an_earlier_fit_are_not_used(env):
    _base, regime, _d211, _prod = _world(env)
    with open(os.path.join(regime, "spatial_gate_small_evals.json"), "w") as f:
        json.dump({"artifact_fp": "old", "gate_diagnostics": {"available": True}}, f)
    small = next(c for c in um.universal_payload()["combiners"] if c["id"] == "gate:small")
    assert small["weights_source"] is None and "earlier fit" in small["weights_reason"]
    assert small["hours"]["max"] is None and small["hours"]["sum"] == pytest.approx(2.0)


def _compare_report(regime, *, created, identity=None, member_fps=None):
    report = {"id": "20990101-000000", "created": created,
              "knee": {"methods": {"gate:spatial_gate_small": {
                  "integrated": [58, 59, 59, 60], "psnr": [[1] * 4] * 21}}}}
    if identity is not None:
        report["identity"] = identity
    if member_fps is not None:
        report["member_fps"] = member_fps
    with open(os.path.join(regime, "spatial_gate_comparison.json"), "w") as f:
        json.dump(report, f)


def _small(env):
    return next(c for c in um.universal_payload()["combiners"] if c["id"] == "gate:small")


def test_a_compare_score_is_comparable_only_on_the_scores_identity(env):
    base, regime, _d211, _prod = _world(env)
    fps = {"170·psnr": member_fingerprint(os.path.join(base, "member_170"))}
    same = {"records_fp": "fp", "subset": "test", "indices": [1, 0],
            "target_psf_fwhm_arcsec": 0.066}
    _compare_report(regime, created="2099-01-01T00:00:00+00:00", identity=same, member_fps=fps)
    small = _small(env)
    assert small["comparable"] is True and small["stale"] is False
    for key, value in (("indices", [0, 1, 2]), ("records_fp", "other"),
                       ("target_psf_fwhm_arcsec", 0.1)):
        _compare_report(regime, created="2099-01-01T00:00:00+00:00",
                        identity={**same, key: value}, member_fps=fps)
        assert _small(env)["comparable"] is False, key


def test_a_compare_score_goes_stale_when_a_member_it_read_is_retrained(env):
    base, regime, _d211, _prod = _world(env)
    d170 = os.path.join(base, "member_170")
    _compare_report(regime, created="2099-01-01T00:00:00+00:00",
                    member_fps={"170·psnr": member_fingerprint(d170)})
    assert _small(env)["stale"] is False
    # continued on FASRC and pulled: a new checkpoint, the report scored the old one
    _member(base, 170, ckpt="ckpt-9")
    assert _small(env)["stale"] is True


def test_a_legacy_compare_score_goes_stale_when_a_read_checkpoint_is_newer(env):
    base, regime, _d211, _prod = _world(env)
    index = os.path.join(base, "member_170", "ckpt-5.index")
    _compare_report(regime, created="2026-10-01T00:00:00+00:00")
    old = time.mktime((2026, 9, 1, 0, 0, 0, 0, 0, 0))
    os.utime(index, (old, old))
    variant = next(v for v in ev.combiner_variants(False)["variants"]
                   if v["name"] == "spatial_gate_small")
    variant["fitted_at"] = None                       # isolate the checkpoint rule
    hit = um._compare_knee_rows(regime)["gate:spatial_gate_small"]
    assert hit["member_fps"] is None and hit["identity"] is None
    fingerprint_of = um._fingerprint_resolver(base)
    assert um._compare_stale(hit, variant, fingerprint_of) is False
    new = time.mktime((2026, 10, 5, 0, 0, 0, 0, 0, 0))
    os.utime(index, (new, new))
    assert um._compare_stale(hit, variant, um._fingerprint_resolver(base)) is True


def test_weights_pending_lists_variants_measurable_from_the_validate_cubes(env):
    _base, regime, _d211, _prod = _world(env)
    os.makedirs(os.path.join(regime, "cubes_validate"))
    with open(os.path.join(regime, "cubes_validate", "viz_index.json"), "w") as f:
        json.dump({"member_labels": ["150·psnr", "170·psnr", "196·psnr"]}, f)
    p = um.universal_payload()
    old = next(c for c in p["combiners"] if c["id"] == "gate:old")
    assert old["weights_source"] is None and "Measure gate weights" in old["weights_reason"]
    # production keeps its own payload; small has current shares
    assert p["weights_pending"] == ["gate:old"]
    assert any("without measured weights" in n for n in p["notes"])
    with open(os.path.join(regime, "cubes_validate", "viz_index.json"), "w") as f:
        json.dump({"member_labels": ["170·psnr", "196·psnr"]}, f)
    assert um.universal_payload()["weights_pending"] == []    # lacks 150: not measurable


def test_members_payload_and_member_detail_serve_registered_universal_models(env):
    base, _regime_dir, _d211, _prod = _world(env)
    payload = ev.members_payload(False)
    assert payload["universal"] == ["member_211"]
    assert [r["name"] for r in payload["members"]] == ["member_170", "member_196"]
    d = ev.member_detail("211")
    assert d["class"] == "universal" and d["universal"] is True and d["active"] is False
    row = d["row"]
    assert row["class"] == "universal" and row["name"] == "member_211"
    assert row["gate_usage"] is None and row["used_by_gate"] is None and row["gate_usage_peak"] is None
    assert row["knee_integrated"]["mean"] == pytest.approx(62.0) and row["score_stale"] is False
    assert d["gate"] is None and d["curves"]["psnr"]
    knee = d["knee"]
    own = next(m for m in knee["models"] if m["kind"] == "member")
    assert own["label"] == "211·psnr" and own["integrated"] == [62.0] * 4
    assert {m["kind"] for m in knee["models"]} == {"member", "mean", "combiner"}
    assert d["comparable"] is True
    assert ev.member_detail("196")["class"] == "member"
    assert ev.member_detail("196")["comparable"] is None
    assert ev.member_detail("196")["universal"] is False      # an ensemble member
    # registered but not pulled yet: the inspector opens, without a row
    er.add_universal_entry(base, "230")
    pending = ev.member_detail("230")
    assert pending["class"] == "universal" and pending["row"] is None
    assert any("member_230" in n for n in um.universal_payload()["notes"])


def test_member_detail_leaves_the_reference_curves_out_when_not_comparable(env):
    _base, regime, d211, _prod = _world(env)
    path = os.path.join(regime, "ensemble_knee_psnr.json")
    knee = json.load(open(path))
    knee["identity"]["records_fp"] = "regenerated"
    json.dump(knee, open(path, "w"))
    d = ev.member_detail("211")
    assert d["comparable"] is False
    assert [m["kind"] for m in d["knee"]["models"]] == ["member"]
    assert d["row"]["score_stale"] is False           # its own score is still current
    _member(ev.ensemble_dir(), 211, origin=LEARNED, ckpt="ckpt-9")
    assert ev.member_detail("211")["row"]["score_stale"] is True


# ---- score job ---------------------------------------------------------------- #

class _FakeEnsemble:
    """An ensemble of one: SR = nearest-neighbour ×2 of the LR, scaled per
    model (``SCALE``) — 211 perfect, 196 10 % bright."""

    SCALE = {"211·psnr": 1.0, "196·psnr": 1.1}
    made: list[dict] = []

    def __init__(self, base_dir, *, labels, include_universal, starless):
        self.made.append({"labels": list(labels), "include_universal": include_universal,
                          "starless": starless})
        self.scale = self.SCALE[labels[0]]

    def member_arrays(self, lr):
        up = np.repeat(np.repeat(np.asarray(lr, np.float32), 2, axis=0), 2, axis=1)
        return (up * self.scale)[None]


def _records(env, monkeypatch, *, fp="fp", size=4):
    rng = np.random.default_rng(3)
    lrs = {i: rng.exponential(100.0, (size, size, 4)).astype(np.float32) for i in (0, 1, 2)}
    hrs = {i: np.repeat(np.repeat(lr, 2, axis=0), 2, axis=1) for i, lr in lrs.items()}
    records = os.path.join(str(env), "records")
    os.makedirs(records, exist_ok=True)

    def read(path, num_images=None):
        source = hrs if "hr_test" in path else lrs
        return [SimpleNamespace(index=i, data=source[i], pixel_scale_arcsec=0.1)
                for i in sorted(source)][:num_images]

    monkeypatch.setattr(ev, "_sky_records_local_dir", lambda: records)
    monkeypatch.setattr(ev, "_eval_records_fingerprint", lambda *a, **k: fp)
    monkeypatch.setattr(um, "ImageSet", SimpleNamespace(read=read))
    monkeypatch.setattr(um, "tfrecord_path", lambda d, name: os.path.join(d, name))
    _FakeEnsemble.made = []
    monkeypatch.setattr(um, "EnsembleModel", _FakeEnsemble)
    monkeypatch.setattr(um, "infer_checkpoint_learned_output_knee",
                        lambda d: (5.0, 6.0, 7.0, 8.0) if d.endswith("211") else None)
    monkeypatch.setattr(um, "checkpoint_step", lambda d: 3000)
    return lrs, hrs


def _expected(lrs, hrs, scale, fields=(0, 1)):
    curves = []
    for i in fields:
        sr = np.repeat(np.repeat(lrs[i], 2, axis=0), 2, axis=1) * scale
        truth = blur_target_array(hrs[i], Config.TARGET_PSF_FWHM_ARCSEC, pixel_scale_arcsec=0.1)
        curves.append(knee_psnr(sr, truth, truth_asinh=stretched_truth(truth)))
    return integrated_psnr(np.mean(curves, axis=0))


def _score_world(env):
    base = ev.ensemble_dir()
    er.add_universal_entry(base, "211")
    _member(base, 170)
    _member(base, 196, origin=FIXED)
    _member(base, 211, origin=LEARNED)
    regime = _regime(env)
    with open(os.path.join(regime, "ensemble_knee_psnr.json"), "w") as f:
        json.dump({"available": True, "identity": {"subset": "test", "indices": [0, 1],
                                                   "target_psf_fwhm_arcsec": 0.066}}, f)
    return base, regime


def test_score_job_scores_with_the_leaderboard_metric_into_its_own_store(env, monkeypatch):
    base, regime = _score_world(env)
    lrs, hrs = _records(env, monkeypatch)
    cap = _Cap()
    out = um.job_universal_score(cap)
    assert out["scored"] == ["member_196", "member_211"] and out["reused"] == []
    assert out["failed"] == {} and out["identity"]["indices"] == [0, 1]
    assert _FakeEnsemble.made == [
        {"labels": ["196·psnr"], "include_universal": True, "starless": False},
        {"labels": ["211·psnr"], "include_universal": True, "starless": False}]
    index = ev.read_universal_scores()
    assert index["identity"] == {"records_fp": "fp", "subset": "test", "indices": [0, 1],
                                 "target_psf_fwhm_arcsec": 0.066, "knees": list(KNEE_GRID_E)}
    e211 = index["models"]["member_211"]
    np.testing.assert_allclose(e211["integrated"], _expected(lrs, hrs, 1.0), atol=1e-4)
    np.testing.assert_allclose(index["models"]["member_196"]["integrated"],
                               _expected(lrs, hrs, 1.1), atol=1e-4)
    assert e211["learned_knees"] == [5.0, 6.0, 7.0, 8.0] and e211["served_step"] == 3000
    # the training the scored checkpoint had, recorded at scoring time
    assert e211["scored_hours"] == pytest.approx(1.0) and e211["scored_steps"] == 2000
    assert index["models"]["member_196"]["learned_knees"] is None
    assert e211["fingerprint"] == member_fingerprint(os.path.join(base, "member_211"))
    assert e211["n_fields"] == 2
    with np.load(os.path.join(regime, "universal", "member_211.npz")) as data:
        assert data["fields"].tolist() == [0, 1] and data["curves"].shape == (2, 21, 4)
    # never a cube bucket
    assert not os.path.exists(os.path.join(regime, "cubes"))
    assert ("member_211: field 2/2" in [t[2] for t in cap.ticks if len(t) > 2])

    again = um.job_universal_score(_Cap())
    assert again["scored"] == [] and again["reused"] == ["member_196", "member_211"]
    assert len(_FakeEnsemble.made) == 2                       # nothing reloaded
    forced = um.job_universal_score(_Cap(), members=["211"], force=True)
    assert forced["scored"] == ["member_211"] and forced["reused"] == []
    with pytest.raises(RuntimeError, match="member_170"):
        um.job_universal_score(_Cap(), members=["170"])


def test_score_job_voids_scores_made_on_other_records(env, monkeypatch):
    _base, regime = _score_world(env)
    _records(env, monkeypatch)
    um.job_universal_score(_Cap(), members=["196"])
    assert set(ev.read_universal_scores()["models"]) == {"member_196"}
    _records(env, monkeypatch, fp="regenerated")
    out = um.job_universal_score(_Cap(), members=["211"])
    assert out["scored"] == ["member_211"]
    index = ev.read_universal_scores()
    assert set(index["models"]) == {"member_211"} and index["identity"]["records_fp"] == "regenerated"
    assert not os.path.exists(os.path.join(regime, "universal", "member_196.npz"))


def test_score_job_needs_the_test_records(env):
    _score_world(env)
    with pytest.raises(RuntimeError, match="records"):
        um.job_universal_score(_Cap())


def test_a_stale_score_keeps_where_it_was_made(env, monkeypatch):
    """Scored at 2000 steps / 1 h; continued to 4000 steps / 2 h since: the
    row's train_hours/steps move on, scored_hours/scored_steps and
    served_step stay where the score was made."""
    base, _regime = _score_world(env)
    _records(env, monkeypatch)
    um.job_universal_score(_Cap(), members=["211"])
    _member(base, 211, origin=LEARNED, ckpt="ckpt-9",
            rows=[(1000, T0, 1800.0, False), (2000, T0 + 1800, 1800.0, False),
                  (3000, T0 + 3600, 1800.0, False), (4000, T0 + 5400, 1800.0, False)])
    row = next(m for m in um.universal_payload()["models"] if m["name"] == "member_211")
    assert row["score"]["stale"] is True
    assert row["train_hours"] == pytest.approx(2.0) and row["steps"] == 4000
    assert row["scored_hours"] == pytest.approx(1.0) and row["scored_steps"] == 2000
    assert row["served_step"] == 3000


# ---- spectra (Models › Spectra) ------------------------------------------------ #

def test_the_score_job_writes_each_models_spectra_next_to_its_curves(env, monkeypatch):
    """96-px LR → 192-px SR: one 128-px radial window after the 32-px border."""
    base, regime = _score_world(env)
    _records(env, monkeypatch, size=96)
    out = um.job_universal_score(_Cap())
    assert out["spectra"] == ["member_196", "member_211"]
    record = ev.read_universal_spectra("member_211")
    assert os.path.isfile(os.path.join(regime, "universal", "member_211.spectra.json"))
    assert record["available"] is True and record["n_fields"] == 2
    assert record["fingerprint"] == member_fingerprint(os.path.join(base, "member_211"))
    assert record["identity"] == ev.read_universal_scores()["identity"]
    assert record["served_step"] == 3000 and record["bands"] == list(BANDS)
    coherence = np.array(record["radial"]["coherence"], float)
    assert coherence.shape == (4, 63) and np.all(coherence[:, :8] > 0.9)
    assert len(record["map"]["values"]) == 4 and len(record["map"]["values"][0]) == 64

    payload = us.spectra_payload()
    assert payload["records_local"] is True and payload["missing"] == []
    assert payload["period_bands"] == [[2, 3], [3, 6], [6, 12], [12, 21], [21, 42]]
    assert len(payload["angle_deg"]) == 36 and len(payload["period_px"]) == 63
    assert np.array(payload["truth_power"], float).shape == (4, 63)
    rows = {m["name"]: m for m in payload["models"]}
    assert list(rows) == ["member_196", "member_211"]
    row = rows["member_211"]
    assert row["status"] == "current" and row["number"] == 211 and row["depth"] == 16
    assert row["knees"] == KNEES6 and row["isotropy_penalty"] == 0.0 and row["steps"] == 3000
    assert row["mean_psnr"] == ev.read_universal_scores()["models"]["member_211"]["mean"]
    assert set(row["radial"]) == {"power_sr", "coherence", "transfer"}
    assert "map" not in row and set(row["angular"]) == {"2-3", "3-6", "6-12", "12-21"}

    detail = us.spectra_map_payload("member_211")
    assert detail["name"] == "member_211" and detail["status"] == "current"
    assert detail["map"]["k"][0] == -32 and detail["map"]["period_lines_px"] == [21, 3]
    assert us.spectra_map_payload("member_170") is None          # not universal

    # a new checkpoint makes them stale
    _member(base, 211, origin=LEARNED, ckpt="ckpt-9")
    assert {m["name"]: m["status"] for m in us.spectra_payload()["models"]} == {
        "member_196": "current", "member_211": "stale"}


def test_spectra_rerun_a_model_whose_score_is_current_only_when_asked(env, monkeypatch):
    _base, regime = _score_world(env)
    _records(env, monkeypatch, size=96)
    um.job_universal_score(_Cap())
    os.remove(os.path.join(regime, "universal", "member_196.spectra.json"))
    assert {m["name"]: m["status"] for m in us.spectra_payload()["models"]} == {
        "member_196": "missing", "member_211": "current"}
    assert us.spectra_payload()["missing"] == ["member_196"]
    assert us.spectra_map_payload("member_196") is None
    plain = um.job_universal_score(_Cap())
    assert plain["scored"] == [] and plain["spectra"] == []
    asked = um.job_universal_score(_Cap(), spectra=True)
    assert asked["scored"] == ["member_196"] and asked["spectra"] == ["member_196"]
    assert asked["reused"] == ["member_211"]
    assert ev.read_universal_spectra("member_196")["available"] is True


def test_spectra_of_fields_smaller_than_the_windows_say_why(env, monkeypatch):
    _score_world(env)
    _records(env, monkeypatch)                                    # 8-px SR
    um.job_universal_score(_Cap(), members=["211"])
    row = next(m for m in us.spectra_payload()["models"] if m["name"] == "member_211")
    assert row["status"] == "current" and row["radial"] is None and row["angular"] is None
    assert "smaller than" in row["reason"]
    assert us.spectra_map_payload("member_211") is None


def test_the_spectra_payload_answers_on_an_empty_ensemble(env):
    payload = us.spectra_payload()
    assert payload["models"] == [] and payload["truth_power"] is None
    assert payload["records_local"] is False and payload["schema"] == 1
    assert any("not local" in note for note in payload["notes"])


def _fake_gate_diagnostic(monkeypatch, calls):
    def fake_diag(comb, *, starless):
        calls.append(list(comb.member_labels))
        return _diag([0.5, 0.5], [0.4, 0.6], [[0.3, 0.7], [0.9, 0.1]])

    monkeypatch.setattr(ev, "_spatial_gate_weight_diagnostic", fake_diag)


def test_score_job_measures_gate_weights_first_even_without_test_records(env, monkeypatch):
    """Gate weights need only the validate cubes: measured before anything
    else, so a missing test-record sync never blocks them; with nothing to
    score the job succeeds, else it fails after measuring them."""
    _base, regime = _score_world(env)
    _gate(regime, "spatial_gate_a", ["170·psnr", "196·psnr"])
    calls: list[list[str]] = []
    _fake_gate_diagnostic(monkeypatch, calls)
    with pytest.raises(RuntimeError, match="records"):
        um.job_universal_score(_Cap())                 # 196, 211 unscored
    assert calls == [["170·psnr", "196·psnr"]]
    assert os.path.isfile(os.path.join(regime, "spatial_gate_a_evals.json"))
    # every model scored (records present then), the records gone since
    _records(env, monkeypatch)
    um.job_universal_score(_Cap())
    monkeypatch.setattr(ev, "_sky_records_local_dir", lambda: None)
    os.remove(os.path.join(regime, "spatial_gate_a_evals.json"))
    out = um.job_universal_score(_Cap())
    assert out["scored"] == [] and out["reused"] == ["member_196", "member_211"]
    assert out["gates"]["computed"] == ["spatial_gate_a"] and out["failed"] == {}
    assert set(ev.read_universal_scores()["models"]) == {"member_196", "member_211"}


def test_score_job_measures_gate_weights_with_nothing_to_score(env, monkeypatch):
    regime = _regime(env)
    _gate(regime, "spatial_gate_a", ["170·psnr", "196·psnr"])
    calls: list[list[str]] = []
    _fake_gate_diagnostic(monkeypatch, calls)
    out = um.job_universal_score(_Cap())               # no universal model, no records
    assert out["scored"] == [] and out["reused"] == []
    assert out["gates"]["computed"] == ["spatial_gate_a"]


def test_one_broken_gate_never_stops_the_weight_measurement(env, monkeypatch):
    regime = _regime(env)
    _gate(regime, "spatial_gate_a", ["170·psnr", "196·psnr"])
    _gate(regime, "spatial_gate_b", ["170·psnr", "196·psnr"])

    def flaky(comb, *, starless):
        if flaky.n == 0:
            flaky.n += 1
            raise ValueError("corrupt cube")
        return _diag([0.5, 0.5], [0.4, 0.6], [[0.3, 0.7], [0.9, 0.1]])

    flaky.n = 0
    monkeypatch.setattr(ev, "_spatial_gate_weight_diagnostic", flaky)
    out = um.job_universal_score(_Cap())
    assert out["gates"]["unavailable"] == {"spatial_gate_a": "ValueError: corrupt cube"}
    assert out["gates"]["computed"] == ["spatial_gate_b"]


def test_score_job_measures_missing_variant_gate_weights(env, monkeypatch):
    _base, regime = _score_world(env)
    _records(env, monkeypatch)
    _gate(regime, "spatial_gate_combiner", ["170·psnr", "196·psnr"])
    _gate(regime, "spatial_gate_a", ["170·psnr", "196·psnr"])
    _gate(regime, "spatial_gate_b", ["170·psnr", "150·psnr"])
    _gate(regime, "spatial_gate_backup_20260101-000000", ["170·psnr"])
    calls: list[list[str]] = []

    def fake_diag(comb, *, starless):
        calls.append(list(comb.member_labels))
        if "150·psnr" in comb.member_labels:
            return {"available": False, "reason": "validation cube cache lacks members this gate reads"}
        return _diag([0.5, 0.5], [0.4, 0.6], [[0.3, 0.7], [0.9, 0.1]])

    monkeypatch.setattr(ev, "_spatial_gate_weight_diagnostic", fake_diag)
    out = um.job_universal_score(_Cap(), members=["211"])
    assert out["gates"]["computed"] == ["spatial_gate_a"]
    assert out["gates"]["unavailable"] == {
        "spatial_gate_b": "validation cube cache lacks members this gate reads"}
    saved = json.load(open(os.path.join(regime, "spatial_gate_a_evals.json")))
    assert saved["artifact_fp"] == combiner_artifact_fingerprint(regime, "spatial_gate_a")
    assert saved["gate_diagnostics"]["artifact_fp"] == saved["artifact_fp"]
    assert not os.path.exists(os.path.join(regime, "spatial_gate_b_evals.json"))
    assert calls == [["170·psnr", "196·psnr"], ["170·psnr", "150·psnr"]]
    again = um.job_universal_score(_Cap(), members=["211"])
    assert again["gates"]["reused"] == ["spatial_gate_a"] and len(calls) == 3


# ---- pull job ----------------------------------------------------------------- #

class _PullSSH:
    """FASRC stand-in: ``origins`` = the remote member dirs with a checkpoint
    (name → origin.json), ``no_checkpoint`` = dirs with an origin.json but no
    checkpoint; ``rsync`` = what a pull of a name does (default: lands the
    model with a checkpoint)."""

    def __init__(self, base, origins, *, no_checkpoint=None, rsync=None):
        self.base = base
        self.origins = origins
        self.no_checkpoint = no_checkpoint or {}
        self.rsync = rsync or {}
        self.rsyncs: list[tuple[str, str]] = []
        self.registered_at_rsync: list[tuple[list, list]] = []

    def is_connected(self):
        return True

    def run(self, cmd, timeout=None):
        for name, origin in {**self.origins, **self.no_checkpoint}.items():
            if f"/{name}/checkpoint" in cmd and f"/{name}/origin.json" in cmd:
                head = um._CKPT_YES if name in self.origins else um._CKPT_NO
                return 0, f"{head}\n{json.dumps(origin)}", ""
        return 1, f"{um._CKPT_NO}\n", "cat: No such file or directory"

    def rsync_pull(self, remote, local, extra_args=None, timeout=600):
        reg = er.load_registry(self.base)                 # what a page load would do
        self.registered_at_rsync.append((list(reg["universal"]), list(reg["active"])))
        self.rsyncs.append((remote, local))
        name = os.path.basename(local.rstrip("/"))
        action = self.rsync.get(name)
        if action is not None:
            return action(local)
        _member(self.base, int(name.removeprefix("member_")), origin=LEARNED, ckpt="ckpt-9")
        with open(os.path.join(local, "ckpt-1.data-00000-of-00001"), "wb") as f:
            f.write(b"orphan")                            # an old generation: pruned
        return 0, "", ""


def _pull_env(env, monkeypatch, ssh):
    monkeypatch.setattr(ev, "remote_ensemble_dir", lambda: "/remote/ensemble")
    monkeypatch.setattr(STATE, "ssh", ssh)


def test_pull_registers_before_the_rsync_and_never_activates(env, monkeypatch):
    base = ev.ensemble_dir()
    _member(base, 196, origin=FIXED)
    _member(base, 150)
    er.load_registry(base)
    er.archive_member_entry(base, "member_150", zip_path="models/z.zip", commit=None)
    ssh = _PullSSH(base, {"member_211": LEARNED, "member_197": {"asinh_knees": KNEES6}})
    _pull_env(env, monkeypatch, ssh)
    out = um.job_universal_pull(_Cap(), members=["211", "196", "150", "bad!", "197", "212", "211"])
    assert out["pulled"] == ["member_211"]
    assert set(out["refused"]) == {"member_196", "member_150", "bad!", "member_197", "member_212"}
    assert "active ensemble member" in out["refused"]["member_196"]
    assert "tombstoned" in out["refused"]["member_150"]
    assert "not a universal model" in out["refused"]["member_197"]
    assert "no checkpoint on FASRC" in out["refused"]["member_212"]    # no remote dir
    assert ssh.rsyncs == [("/remote/ensemble/member_211/", os.path.join(base, "member_211"))]
    # registered before its rsync, so the bootstrap never made it active
    assert ssh.registered_at_rsync == [(["member_211"], ["member_196"])]
    reg = er.load_registry(base)
    assert reg["universal"] == ["member_211"] and "member_211" not in reg["active"]
    assert not os.path.exists(os.path.join(base, "member_211", "ckpt-1.data-00000-of-00001"))
    assert "member_211" in um.universal_targets(base)
    assert purge_requests.read_pending()["reasons"] == ["pulled universal member_211"]


def test_pull_refuses_a_remote_dir_without_a_checkpoint_before_registering(env, monkeypatch):
    """A job that ended (OOM, cancelled) before its first save leaves an
    origin.json but no checkpoint: refused, nothing registered or rsynced."""
    base = ev.ensemble_dir()
    ssh = _PullSSH(base, {}, no_checkpoint={"member_240": LEARNED})
    _pull_env(env, monkeypatch, ssh)
    out = um.job_universal_pull(_Cap(), members=["240"])
    assert out == {"pulled": [], "refused": {"member_240": out["refused"]["member_240"]}}
    assert "no checkpoint on FASRC" in out["refused"]["member_240"]
    assert ssh.rsyncs == [] and er.load_registry(base)["universal"] == []


def test_a_failed_rsync_rolls_back_the_registration_and_the_next_name_still_pulls(
        env, monkeypatch):
    base = ev.ensemble_dir()

    def partial(local):                     # the manifest landed, the weights did not
        os.makedirs(local, exist_ok=True)
        with open(os.path.join(local, "checkpoint"), "w") as f:
            f.write('model_checkpoint_path: "ckpt-9"\n')
        return 255, "", "connection reset"

    def dies(local):
        os.makedirs(local, exist_ok=True)
        with open(os.path.join(local, "checkpoint"), "w") as f:
            f.write('model_checkpoint_path: "ckpt-9"\n')
        raise subprocess.TimeoutExpired("rsync", 3600)

    ssh = _PullSSH(base, {"member_211": LEARNED, "member_212": LEARNED, "member_213": LEARNED},
                   rsync={"member_211": partial, "member_212": dies})
    _pull_env(env, monkeypatch, ssh)
    out = um.job_universal_pull(_Cap(), members=["211", "212", "213"])
    assert out["pulled"] == ["member_213"]
    assert "no loadable checkpoint" in out["refused"]["member_211"]
    assert "not registered" in out["refused"]["member_211"]
    assert "TimeoutExpired" in out["refused"]["member_212"]
    reg = er.load_registry(base)
    # the partial dirs this pull created are gone, so nothing is bootstrapped
    assert reg["universal"] == ["member_213"] and reg["active"] == []
    assert not os.path.exists(os.path.join(base, "member_211"))
    assert not os.path.exists(os.path.join(base, "member_212"))


def test_a_failed_refresh_of_a_registered_model_keeps_its_registration(env, monkeypatch):
    """A registered model whose dir is gone (a failed pull, the mirror) is
    pulled again from its FASRC row; when that fails too it stays
    registered, its partial dir is removed, and it is still offered."""
    base = ev.ensemble_dir()
    er.add_universal_entry(base, "211")             # registered, its dir removed
    _job("81", ["member_211"], spec=[SPEC_LEARNED])
    assert [(r["name"], r["registered"]) for r in um.fasrc_universal_rows()] == [
        ("member_211", True)]

    def partial(local):
        os.makedirs(local, exist_ok=True)
        with open(os.path.join(local, "checkpoint"), "w") as f:
            f.write('model_checkpoint_path: "ckpt-9"\n')
        return 255, "", "boom"

    ssh = _PullSSH(base, {"member_211": LEARNED}, rsync={"member_211": partial})
    _pull_env(env, monkeypatch, ssh)
    out = um.job_universal_pull(_Cap(), members=["211"])
    assert "stays registered" in out["refused"]["member_211"]
    assert er.load_registry(base)["universal"] == ["member_211"]
    assert not os.path.exists(os.path.join(base, "member_211"))
    assert [r["name"] for r in um.fasrc_universal_rows()] == ["member_211"]


def test_pull_refuses_a_name_a_local_gate_or_cube_manifest_reads(env, monkeypatch):
    """195/196 are universal by recipe but ensemble members: if their dirs
    vanish (dropped from active), the gate that reads them still names them,
    so the universal pull must not turn them into standalone models."""
    base = ev.ensemble_dir()
    _member(base, 170)
    er.load_registry(base)
    regime = _regime(env)
    _gate(regime, "spatial_gate_combiner", ["170·psnr", "196·psnr"])
    os.makedirs(os.path.join(regime, "cubes"))
    with open(os.path.join(regime, "cubes", "viz_index.json"), "w") as f:
        json.dump({"member_labels": ["170·psnr", "195·psnr"]}, f)
    ssh = _PullSSH(base, {"member_195": FIXED, "member_196": FIXED})
    _pull_env(env, monkeypatch, ssh)
    out = um.job_universal_pull(_Cap(), members=["195", "196"])
    assert out["pulled"] == [] and set(out["refused"]) == {"member_195", "member_196"}
    assert all("ensemble member" in r for r in out["refused"].values())
    assert ssh.rsyncs == [] and er.load_registry(base)["universal"] == []


def test_the_remote_probe_command_reads_the_checkpoint_and_origin_in_one_call(tmp_path):
    """The shell line really run on FASRC, run here by sh."""
    class _LocalSSH:
        def run(self, cmd, timeout=None):
            done = subprocess.run(["sh", "-c", cmd], capture_output=True, text=True, check=False)
            return done.returncode, done.stdout, done.stderr

    root = tmp_path / "remote ensemble"            # a space: the paths are quoted
    for name, files in (("member_01", {"checkpoint": "x", "origin.json": json.dumps(LEARNED)}),
                        ("member_02", {"ckpt-3.index": "x", "origin.json": json.dumps(FIXED)}),
                        ("member_03", {"origin.json": json.dumps(LEARNED)}),
                        ("member_04", {"checkpoint": "x"}),
                        ("member_05", {"checkpoint": "x", "origin.json": "{not json"})):
        (root / name).mkdir(parents=True)
        for fn, text in files.items():
            (root / name / fn).write_text(text)
    ssh = _LocalSSH()
    assert um._remote_probe(ssh, str(root), "member_01") == (LEARNED, "")
    assert um._remote_probe(ssh, str(root), "member_02") == (FIXED, "")
    assert "no checkpoint on FASRC" in um._remote_probe(ssh, str(root), "member_03")[1]
    assert "no origin.json" in um._remote_probe(ssh, str(root), "member_04")[1]
    assert "unreadable" in um._remote_probe(ssh, str(root), "member_05")[1]
    assert "no checkpoint on FASRC" in um._remote_probe(ssh, str(root), "member_06")[1]


def test_pull_needs_fasrc_and_names(env, monkeypatch):
    monkeypatch.setattr(STATE, "ssh", None)
    with pytest.raises(RuntimeError, match="not connected"):
        um.job_universal_pull(_Cap(), members=["211"])
    with pytest.raises(RuntimeError, match="name the universal models"):
        um.job_universal_pull(_Cap(), members=[])


# ---- archive / restore -------------------------------------------------------- #

def test_a_universal_model_archives_and_restores_as_a_universal_model(env, monkeypatch):
    default_store().create_campaign("universal archive")
    monkeypatch.setattr(STATE, "ssh", None)             # the FASRC delete is only flagged
    base = ev.ensemble_dir()
    er.add_universal_entry(base, "211")
    _member(base, 211, origin=LEARNED)
    _member(base, 170)
    er.load_registry(base)
    out = ev.job_archive_member(_Cap(), name="member_211")
    assert out["class"] == "universal" and out["stale_regime"] is None
    assert out["zip"] == "universal-member-211.zip"
    assert ev._pending_archived_members(False) == []       # no evaluation goes stale
    reg = er.load_registry(base)
    assert reg["universal"] == [] and reg["active"] == ["member_170"]
    assert reg["archived"][0]["class"] == "universal"
    assert not os.path.isdir(os.path.join(base, "member_211"))
    archived = ev.member_detail("211")                    # a tombstone now
    assert archived["class"] == "universal" and archived["universal"] is False
    assert archived["active"] is False and archived["archived"]["class"] == "universal"
    restored = ev.job_restore_member(_Cap(), name="211")
    assert restored["member"] == "member_211"
    reg = er.load_registry(base)
    assert reg["universal"] == ["member_211"] and reg["active"] == ["member_170"]
    assert um.universal_targets(base)["member_211"][0] == "universal"
