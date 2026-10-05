"""The Models workspace backend (spec §8.2, named Ensemble there) in helpers/ensemble_viz.py:
curves payload, the joined members table, member detail, the combiner
variant registry, promote / restore / pull jobs, the overview checks and the
train-command preview. Local files only; no TensorFlow."""
from __future__ import annotations

import json
import os
import zipfile

import numpy as np
import pytest

from euclid_polish import ensemble_registry as er
from euclid_polish.config import Config
from euclid_polish.observability import JobLog, JobRecord
from euclid_polish.web import fasrc_jobs
from euclid_polish.web.helpers import ensemble_viz as ev
from euclid_polish.web.remote import STATE


class _Cap:
    def tick(self, *_a, **_k):
        pass


LOG_HEADER = ("step,wall_time,loss,psnr_stretched,psnr_raw,psnr_vis,psnr_y_e,psnr_j_e,"
              "psnr_h_e,gnorm_avg,gnorm_max,clip_norm,duration_s,combined_loss,is_baseline\n")


def _member(base, i, *, steps=(1000, 2000), origin=None, ckpt="ckpt-5"):
    d = os.path.join(base, f"member_{i:02d}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "checkpoint"), "w") as f:
        f.write(f'model_checkpoint_path: "{ckpt}"\n')
    with open(os.path.join(d, f"{ckpt}.index"), "wb") as f:
        f.write(b"idx")
    rows = "".join(f"{s},1,0.5,{40 + s / 1000},90,39,41,40,40,3.0,9.0,5,{100 + s / 100},0.1,\n"
                   for s in steps)
    with open(os.path.join(d, "training_log.csv"), "w") as f:
        f.write(LOG_HEADER + rows)
    with open(os.path.join(d, "origin.json"), "w") as f:
        json.dump({"loss_norm": "l2", "seed": 1000 + i, "target_steps": 2000, **(origin or {})}, f)
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


def _gate_dir(regime, name, labels, **extra):
    d = os.path.join(regime, name)
    os.makedirs(d, exist_ok=True)
    manifest = {"schema": 1, "kind": "spatial_gate", "member_labels": labels,
                "band_names": ["VIS", "Y_E", "J_E", "H_E"], "width": 32, "use_lr": False,
                "mix_space": "linear", "active_members": None, "dilations": [1, 2, 4, 8],
                "fit_meta": {"steps": 2000, "steps_run": 2000, "complete": True,
                             "history": [{"step": 0, "loss": 1.0},
                                         {"step": 250, "loss": 0.7}],
                             "selected": {"loss": 0.7}, "train_fields": [1, 2, 3]},
                **extra}
    with open(os.path.join(d, "combiner.json"), "w") as f:
        json.dump(manifest, f)
    np.savez(os.path.join(d, "combiner.npz"), w=np.zeros(3))
    return d


# ---- curves ----------------------------------------------------------------- #

def test_training_curves_keep_the_loss_series_and_add_bands_gnorm_step_time(env):
    base = ev.ensemble_dir()
    _member(base, 1, origin={"asinh_knees": [0.1, 10.0], "output_knee": 10.0})
    (m,) = ev.training_curves_payload()
    assert m["loss_norm"] == "l2"
    assert m["loss_series"] == [[1000, 0.1], [2000, 0.1]]
    assert m["loss"] == m["loss_series"]                 # never overwritten by the norm
    assert m["psnr"] == [[1000, 41.0], [2000, 42.0]]
    assert set(m["band_psnr"]) == {"VIS", "Y_E", "J_E", "H_E"}
    assert m["band_psnr"]["Y_E"][0] == [1000, 41.0]
    assert m["gnorm"][0] == [1000, 3.0] and m["gnorm_max"][0] == [1000, 9.0]
    # duration 110 s for the first 1000 steps, 120 s for the next 1000
    assert m["step_time"] == [[1000, 110.0], [2000, 120.0]]
    assert m["asinh_knees"] == [0.1, 10.0] and m["output_knee"] == 10.0
    assert m["target_steps"] == 2000 and m["label"] == "01·psnr"


def test_training_curves_skip_archived_members(env):
    base = ev.ensemble_dir()
    _member(base, 1)
    _member(base, 2)
    er.load_registry(base)
    er.archive_member_entry(base, "member_02", zip_path="models/z.zip", commit=None)
    assert [m["name"] for m in ev.training_curves_payload()] == ["member_01"]


# ---- progress / jobs --------------------------------------------------------- #

@pytest.mark.parametrize("step, job, status", [
    (2000, None, "complete"),
    (1500, None, "timeout"),
    (1500, {"state": "RUNNING", "mode": "add"}, "running"),
    (None, None, "unknown"),
    (2500, {"state": "COMPLETED", "mode": "continue", "continue_basis": "target",
            "target_steps": 3000}, "timeout"),
])
def test_member_progress(step, job, status):
    p = ev.member_progress(step, {"target_steps": 2000}, job)
    assert p["status"] == status
    assert p["timeout"] is (status == "timeout")


def _record_job(jobid, names, *, state="COMPLETED", mode="add", **params):
    fasrc_jobs.JOBLOG.record_submission(JobRecord(
        jobid=jobid, step_id="ensemble_train", label="Train", submitted_at=f"2026-09-2{jobid[-1]}T00:00:00Z",
        state=state, params_json=json.dumps({"mode": mode, "member_names": ",".join(names), **params})))


def test_training_jobs_map_members_to_their_newest_job(env):
    _record_job("41", ["member_01", "member_02"], steps="2000")
    _record_job("42", ["member_02"], state="RUNNING", mode="continue")
    jobs = ev.training_jobs()
    assert [j["jobid"] for j in jobs] == ["42", "41"]
    by = ev._jobs_by_member(jobs)
    assert by["member_01"]["jobid"] == "41" and by["member_02"]["jobid"] == "42"


# ---- the joined members table ---------------------------------------------- #

def test_members_payload_joins_knee_gate_coherence_and_jobs(env):
    base = ev.ensemble_dir()
    _member(base, 1)
    _member(base, 2, steps=(1000,))
    _member(base, 3, origin={"starless": True})
    regime = _regime(env)
    _record_job("41", ["member_01", "member_02"])
    labels = ["01·psnr", "02·psnr"]
    with open(os.path.join(regime, "ensemble_knee_psnr.json"), "w") as f:
        json.dump({"available": True, "identity": {}, "bands": ["VIS", "Y_E", "J_E", "H_E"],
                   "knees": [1, 10], "n_fields": 5,
                   "models": [{"id": "member_0", "kind": "member", "label": "01·psnr",
                               "integrated": [50, 60, 58, 56], "psnr": [[1, 2, 3, 4]] * 2},
                              {"id": "member_1", "kind": "member", "label": "02·psnr",
                               "integrated": [52, 62, 60, 58], "psnr": [[1, 2, 3, 4]] * 2}]}, f)
    with open(os.path.join(regime, "spatial_gate_combiner_evals.json"), "w") as f:
        json.dump({"available": True, "stale": False, "member_labels": labels,
                   "band_names": ["VIS", "Y_E", "J_E", "H_E"],
                   "gate_diagnostics": {"usage": {b: [0.25, 0.75] for b in ("VIS", "Y_E", "J_E", "H_E")},
                                        "usage_source": {"VIS": [0.1, 0.9]}}}, f)
    with open(os.path.join(regime, "ensemble_evals.json"), "w") as f:
        json.dump({"coherence": {"scores": [{"id": "member_0", "label": "01·psnr",
                                             "overall": 0.9, "sr": 0.5}]}}, f)

    p = ev.members_payload(False)
    assert [r["name"] for r in p["members"]] == ["member_01", "member_02"]
    assert p["other_regime_members"] == 1
    one, two = p["members"]
    assert one["status"] == "complete" and two["status"] == "timeout" and two["timeout"]
    assert one["job"]["jobid"] == "41"
    assert one["knee_integrated"]["VIS"] == 50 and one["knee_integrated"]["mean"] == pytest.approx(56)
    assert two["knee_rank"] == 1 and one["knee_rank"] == 2
    assert two["gate_usage"]["VIS"] == 0.75 and two["gate_usage_source"]["VIS"] == 0.9
    assert one["coherence"] == {"overall": 0.9, "sr": 0.5} and two["coherence"] is None
    assert one["seed"] == 1001 and one["loss"] == "l2"
    assert p["knee"]["available"] and p["gate"]["n_members"] == 2


def test_member_rows_carry_the_gate_peak_and_whether_production_runs_them(env):
    """#195's case: ~0 % over all pixels but half the weight in bright cores.
    The row gives the peak (max over bands, all/source pixels and every
    brightness bin) with where it is, and whether the production gate reads
    the member (so production SR runs it)."""
    base = ev.ensemble_dir()
    _member(base, 1)
    _member(base, 2)
    regime = _regime(env)
    bands = ("VIS", "Y_E", "J_E", "H_E")
    with open(os.path.join(regime, "spatial_gate_combiner_evals.json"), "w") as f:
        json.dump({"available": True, "stale": False, "member_labels": ["01·psnr", "02·psnr"],
                   "read_labels": ["01·psnr"], "band_names": list(bands),
                   "gate_diagnostics": {
                       "usage": {b: [0.999, 0.001] for b in bands},
                       "usage_source": {b: [0.99, 0.01] for b in bands},
                       "brightness_names": ["sky", "core"],
                       "usage_by_brightness": {
                           **{b: [[1.0, 0.0], [0.9, 0.1]] for b in bands},
                           "VIS": [[1.0, 0.0], [0.52, 0.48]]}}}, f)
    one, two = ev.members_payload(False)["members"]
    assert two["gate_usage_peak"] == {"value": 0.48, "band": "VIS", "bin": "core"}
    assert one["gate_usage_peak"]["value"] == 1.0
    assert one["used_by_gate"] is True and two["used_by_gate"] is False
    os.remove(os.path.join(regime, "spatial_gate_combiner_evals.json"))
    one, _two = ev.members_payload(False)["members"]
    assert one["gate_usage_peak"] is None and one["used_by_gate"] is None


def test_members_payload_joins_the_headline_vis_psnr_per_member(env):
    """The Overview's "Best member" tile is VIS asinh from eval_summary; each
    row carries that same number (``vis_psnr``) next to the 4-band cache."""
    base = ev.ensemble_dir()
    _member(base, 1)
    _member(base, 2)
    regime = _regime(env)
    with open(os.path.join(regime, "eval_summary.json"), "w") as f:
        json.dump({"member_labels": ["01·psnr", "02·psnr"], "psnr_metric": "vis_asinh",
                   "psnr_knee_e": 100.0, "n_scored": 7,
                   "per_member_vis_psnr": [57.5, 58.9]}, f)
    p = ev.members_payload(False)
    one, two = p["members"]
    assert one["vis_psnr"] == 57.5 and two["vis_psnr"] == 58.9
    assert p["vis_psnr"] == {"metric": "vis_asinh", "knee_e": 100.0, "n_scored": 7}
    # older cube-recomputed summaries carry the same VIS numbers under the
    # stretched key; a summary whose lists do not line up is ignored.
    with open(os.path.join(regime, "eval_summary.json"), "w") as f:
        json.dump({"member_labels": ["01·psnr", "02·psnr"], "recomputed_from_cubes": True,
                   "per_member_psnr_stretched": [56.0, 57.0]}, f)
    assert [r["vis_psnr"] for r in ev.members_payload(False)["members"]] == [56.0, 57.0]
    with open(os.path.join(regime, "eval_summary.json"), "w") as f:
        json.dump({"member_labels": ["01·psnr"], "per_member_vis_psnr": [56.0, 57.0]}, f)
    assert [r["vis_psnr"] for r in ev.members_payload(False)["members"]] == [None, None]


def test_member_detail_active_archived_unknown(env):
    base = ev.ensemble_dir()
    _member(base, 1)
    _member(base, 2)
    er.load_registry(base)
    er.archive_member_entry(base, "member_02", zip_path="models/z.zip", commit="abc")
    d = ev.member_detail("1")
    assert d["active"] and d["row"]["name"] == "member_01"
    assert "psnr_rank" in d["row"] and "knee_rank" in d["row"]
    assert d["curves"]["psnr"][-1] == [2000, 42.0]
    a = ev.member_detail("member_02")
    assert a["active"] is False and a["archived"]["commit"] == "abc" and a["row"] is None
    assert ev.member_detail("member_77") is None


# ---- variants / promote ------------------------------------------------------ #

def test_combiner_variants_registry(env):
    base = ev.ensemble_dir()
    _member(base, 1)
    _member(base, 2)
    regime = _regime(env)
    _gate_dir(regime, "spatial_gate_combiner", ["01·psnr", "02·psnr"])
    _gate_dir(regime, "spatial_gate_small", ["02·psnr"], active_members=[0])
    _gate_dir(regime, "spatial_gate_backup_20260101-000000", ["01·psnr"])
    with open(os.path.join(regime, "eval_summary.json"), "w") as f:
        json.dump({"spatial_gate_combiner_psnr": 59.2}, f)
    rows = {r["name"]: r for r in ev.combiner_variants(False)["variants"]}
    prod = rows["spatial_gate_combiner"]
    assert prod["production"] and prod["spec"] == "production" and prod["membership"]["current"]
    assert prod["eval"]["psnr"] == 59.2
    assert prod["history"][-1] == {"step": 250, "loss": 0.7}
    assert prod["fit"]["train_field_count"] == 3
    small = rows["spatial_gate_small"]
    assert small["spec"] == "gate:small" and small["pruned"] and small["reads"] == ["02·psnr"]
    assert small["membership"]["extra"] == ["01·psnr"]
    # current: every member it reads is active (01 joined after its fit)
    assert small["membership"]["current"] is True
    assert small["promotion"] == {"ok": True, "reason": None}
    assert prod["promotion"]["ok"] is False
    assert rows["spatial_gate_backup_20260101-000000"]["backup"] is True


def test_combiner_variants_never_list_the_legacy_rbf(env):
    base = ev.ensemble_dir()
    _member(base, 1)
    regime = _regime(env)
    _gate_dir(regime, "spatial_gate_combiner", ["01·psnr"])
    rbf = os.path.join(regime, ev.COMBINER_MODELS[ev._RBF_KIND].artifact_dir)
    os.makedirs(rbf, exist_ok=True)
    with open(os.path.join(rbf, "combiner.json"), "w") as f:
        json.dump({"kind": ev._RBF_KIND, "member_labels": ["01·psnr"], "n_kernels": 8}, f)
    names = [r["name"] for r in ev.combiner_variants(False)["variants"]]
    assert names == ["spatial_gate_combiner"]
    assert all(r["kind"] == "gate" for r in ev.combiner_variants(False)["variants"])


def test_combiner_variants_flag_reads_that_left_and_unfinished_fits(env):
    base = ev.ensemble_dir()
    _member(base, 1)
    regime = _regime(env)
    _gate_dir(regime, "spatial_gate_combiner", ["01·psnr"])
    _gate_dir(regime, "spatial_gate_gone", ["01·psnr", "05·psnr"])
    _gate_dir(regime, "spatial_gate_unread", ["01·psnr", "05·psnr"], active_members=[0])
    _gate_dir(regime, "spatial_gate_half", ["01·psnr"],
              fit_meta={"steps": 2000, "steps_run": 750, "complete": False})
    rows = {r["name"]: r for r in ev.combiner_variants(False)["variants"]}
    gone = rows["spatial_gate_gone"]
    assert gone["membership"]["current"] is False
    assert gone["membership"]["missing_reads"] == ["05·psnr"]
    assert gone["promotion"]["ok"] is False and "05·psnr" in gone["promotion"]["reason"]
    unread = rows["spatial_gate_unread"]            # 05 fitted, never read, now gone
    assert unread["membership"]["current"] is True and unread["promotion"]["ok"] is True
    half = rows["spatial_gate_half"]
    assert half["promotion"]["ok"] is False and "750" in half["promotion"]["reason"]


@pytest.mark.parametrize("name, message", [
    ("spatial_gate_combiner", "production"),
    ("spatial_gate_backup_x", "reserved"),
    ("gate:../x", "not a spatial gate"),
    ("other", "not a spatial gate"),
    ("spatial_gate_comparisons", "reserved"),
    ("gate:comparison.json", "reserved"),
    ("gate:comparison_2", "reserved"),
    ("gate:combiner_evals.json", "reserved"),
    ("gate:trial_evals", "reserved"),
])
def test_check_new_variant_name_refusals(env, name, message):
    with pytest.raises(ValueError, match=message):
        ev.check_new_variant_name(False, name)


def test_check_new_variant_name_existing_needs_overwrite(env):
    _gate_dir(_regime(env), "spatial_gate_trial", ["01·psnr"])
    with pytest.raises(ValueError, match="already exists"):
        ev.check_new_variant_name(False, "gate:trial")
    assert ev.check_new_variant_name(False, "spatial_gate_trial", overwrite=True).endswith(
        "spatial_gate_trial")


def test_check_new_variant_name_never_overwrites_a_non_variant_entry(env):
    """overwrite=1 replaces a variant, never a report folder or a sidecar
    that merely shares the spatial_gate_* prefix."""
    regime = _regime(env)
    os.makedirs(os.path.join(regime, "spatial_gate_notes"))
    with pytest.raises(ValueError, match="not a gate variant"):
        ev.check_new_variant_name(False, "gate:notes", overwrite=True)


def test_promote_backs_up_production_and_installs_the_variant(env, monkeypatch):
    base = ev.ensemble_dir()
    _member(base, 1)
    _member(base, 2)
    regime = _regime(env)
    _gate_dir(regime, "spatial_gate_combiner", ["01·psnr"])
    _gate_dir(regime, "spatial_gate_trial", ["01·psnr", "02·psnr"], width=16)
    monkeypatch.setattr(ev, "compute_combiner_payload", lambda *a, **k: None)
    monkeypatch.setattr(ev, "_apply_combiner_to_test_cubes", lambda *a, **k: False)
    out = ev.job_combiner_promote(_Cap(), starless=False, variant="gate:trial")
    assert out["promoted"] == "spatial_gate_trial" and out["backup"].startswith("spatial_gate_backup_")
    with open(os.path.join(regime, "spatial_gate_combiner", "combiner.json")) as f:
        prod = json.load(f)
    assert prod["width"] == 16 and prod["fit_meta"]["promoted_from"] == "spatial_gate_trial"
    with open(os.path.join(regime, out["backup"], "combiner.json")) as f:
        assert json.load(f)["member_labels"] == ["01·psnr"]
    assert os.path.isdir(os.path.join(regime, "spatial_gate_trial"))     # variant kept
    assert not [n for n in os.listdir(regime) if n.startswith(".")]      # no temp dirs left


def test_promote_refuses_other_members_without_force(env, monkeypatch):
    base = ev.ensemble_dir()
    _member(base, 1)
    _member(base, 2)
    regime = _regime(env)
    _gate_dir(regime, "spatial_gate_combiner", ["01·psnr", "02·psnr"])
    _gate_dir(regime, "spatial_gate_old", ["01·psnr", "03·psnr"])     # 03 is not active
    monkeypatch.setattr(ev, "compute_combiner_payload", lambda *a, **k: None)
    monkeypatch.setattr(ev, "_apply_combiner_to_test_cubes", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="force"):
        ev.job_combiner_promote(_Cap(), starless=False, variant="spatial_gate_old")
    with pytest.raises(RuntimeError, match="already the production"):
        ev.job_combiner_promote(_Cap(), starless=False, variant="production")
    out = ev.job_combiner_promote(_Cap(), starless=False, variant="spatial_gate_old", force=True)
    assert out["promoted"] == "spatial_gate_old"


def test_promote_accepts_a_gate_fitted_before_members_joined(env, monkeypatch):
    """A (pruned) gate stays promotable when members registered after its fit
    or unread members left: only the members it reads must be active."""
    base = ev.ensemble_dir()
    for i in (1, 2, 3):
        _member(base, i)
    regime = _regime(env)
    _gate_dir(regime, "spatial_gate_combiner", ["01·psnr"])
    _gate_dir(regime, "spatial_gate_p1", ["01·psnr", "02·psnr", "07·psnr"],
              active_members=[0, 1])              # 07 archived, never read; 03 joined
    monkeypatch.setattr(ev, "compute_combiner_payload", lambda *a, **k: None)
    monkeypatch.setattr(ev, "_apply_combiner_to_test_cubes", lambda *a, **k: False)
    out = ev.job_combiner_promote(_Cap(), starless=False, variant="gate:p1")
    assert out["promoted"] == "spatial_gate_p1"


@pytest.mark.parametrize("fit_meta, message", [
    ({"steps": 2000, "steps_run": 500, "complete": False}, "step 500 of 2000"),
    ({"steps": 2000}, "no completion flag"),
])
def test_promote_refuses_an_incomplete_fit_even_with_force(env, monkeypatch, fit_meta, message):
    base = ev.ensemble_dir()
    _member(base, 1)
    regime = _regime(env)
    _gate_dir(regime, "spatial_gate_combiner", ["01·psnr"])
    _gate_dir(regime, "spatial_gate_wip", ["01·psnr"], fit_meta=fit_meta)
    monkeypatch.setattr(ev, "compute_combiner_payload", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match=message):
        ev.job_combiner_promote(_Cap(), starless=False, variant="gate:wip", force=True)
    with open(os.path.join(regime, "spatial_gate_combiner", "combiner.json")) as f:
        assert json.load(f)["member_labels"] == ["01·psnr"]      # production untouched
    assert not [n for n in os.listdir(regime) if n.startswith("spatial_gate_backup_")]


def test_promote_refuses_a_variant_a_fit_is_still_writing(env, monkeypatch):
    base = ev.ensemble_dir()
    _member(base, 1)
    regime = _regime(env)
    _gate_dir(regime, "spatial_gate_combiner", ["01·psnr"])
    wip = _gate_dir(regime, "spatial_gate_wip", ["01·psnr"])      # complete, but …
    with open(os.path.join(wip, ev.sgc.FIT_MARKER), "w") as f:  # … a live fit rewrites it
        json.dump({"pid": os.getpid(), "host": ev.sgc.socket.gethostname(),
                   "started": "2026-09-27T12:00:00+00:00"}, f)
    with pytest.raises(RuntimeError, match="still being fitted"):
        ev.job_combiner_promote(_Cap(), starless=False, variant="gate:wip", force=True)
    # A marker left by a dead process on this host does not block it.
    with open(os.path.join(wip, ev.sgc.FIT_MARKER), "w") as f:
        json.dump({"pid": 2 ** 22 + 12345, "host": ev.sgc.socket.gethostname()}, f)
    monkeypatch.setattr(ev.sgc, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(ev, "compute_combiner_payload", lambda *a, **k: None)
    monkeypatch.setattr(ev, "_apply_combiner_to_test_cubes", lambda *a, **k: False)
    assert ev.job_combiner_promote(_Cap(), starless=False, variant="gate:wip")["promoted"]


def test_compare_reports_listing_and_reading(env):
    regime = _regime(env)
    folder = os.path.join(regime, "spatial_gate_comparisons")
    os.makedirs(folder)
    for rid in ("20260925-100000", "20260926-100000"):
        with open(os.path.join(folder, f"{rid}.json"), "w") as f:
            json.dump({"id": rid, "created": rid, "methods": ["mean"], "n_fields": {"natural": 3}}, f)
    with open(os.path.join(regime, "spatial_gate_comparison.json"), "w") as f:
        json.dump({"id": "20260926-100000"}, f)
    assert [r["id"] for r in ev.compare_reports(False)] == ["20260926-100000", "20260925-100000"]
    assert ev.read_compare_report(False)["id"] == "20260926-100000"
    assert ev.read_compare_report(False, "20260925-100000")["id"] == "20260925-100000"
    with pytest.raises(ValueError):
        ev.read_compare_report(False, "../x")


# ---- restore ---------------------------------------------------------------- #

def test_restore_member_from_its_tracking_zip(env):
    base = ev.ensemble_dir()
    _member(base, 1)
    src = _member(base, 2)
    er.load_registry(base)
    models = env / "tracking" / "current" / "models"
    models.mkdir(parents=True)
    (env / "tracking" / "current" / "metadata.json").write_text("{}")
    with zipfile.ZipFile(models / "ensemble-member-02.zip", "w") as zf:
        for fn in os.listdir(src):
            zf.write(os.path.join(src, fn), fn)
    er.archive_member_entry(base, "member_02", zip_path="models/ensemble-member-02.zip", commit=None)
    for fn in os.listdir(src):
        os.remove(os.path.join(src, fn))
    os.rmdir(src)

    out = ev.job_restore_member(_Cap(), name="02")
    assert out["member"] == "member_02" and out["regime"] == "starfull"
    assert er.load_registry(base)["active"] == ["member_01", "member_02"]
    assert os.path.isfile(os.path.join(base, "member_02", "checkpoint"))
    with pytest.raises(RuntimeError, match="not archived"):
        ev.job_restore_member(_Cap(), name="member_02")


def test_restore_refuses_a_zip_slip(env):
    base = ev.ensemble_dir()
    _member(base, 1)
    _member(base, 2)
    er.load_registry(base)
    models = env / "tracking" / "current" / "models"
    models.mkdir(parents=True)
    with zipfile.ZipFile(models / "evil.zip", "w") as zf:
        zf.writestr("../../escape.txt", "x")
        zf.writestr("checkpoint", "x")
    er.archive_member_entry(base, "member_02", zip_path="models/evil.zip", commit=None)
    for fn in os.listdir(os.path.join(base, "member_02")):
        os.remove(os.path.join(base, "member_02", fn))
    os.rmdir(os.path.join(base, "member_02"))
    with pytest.raises(RuntimeError, match="unsafe"):
        ev.job_restore_member(_Cap(), name="member_02")
    assert not (env / "escape.txt").exists()
    assert "member_02" not in er.load_registry(base)["active"]


# ---- pull ------------------------------------------------------------------- #

class _FakeSSH:
    def __init__(self, probe):
        self.probe = probe
        self.pulled: list[str] = []

    def is_connected(self):
        return True

    def rsync_pull(self, remote, local, extra_args=None, timeout=600):
        if extra_args and "--dry-run" in extra_args:
            return self.probe
        self.pulled.append(remote)
        return (0, "", "")


def test_pull_dry_run_and_member_picker(env, monkeypatch):
    base = ev.ensemble_dir()
    _member(base, 1)
    _member(base, 2)
    monkeypatch.setattr(ev, "remote_ensemble_dir", lambda: "/remote/ensemble")
    monkeypatch.setattr(ev, "job_member_psnr", lambda cap: {})
    probe = (0, ">f.st...... member_01/ckpt-9.index\n>f+++++++++ member_03/checkpoint\n", "")
    ssh = _FakeSSH(probe)
    monkeypatch.setattr(STATE, "ssh", ssh)
    dry = ev.job_ensemble_pull(_Cap(), dry_run=True)
    assert dry == {"dry_run": True, "changed": ["member_01", "member_03"], "tombstoned_skipped": []}
    assert ssh.pulled == []
    out = ev.job_ensemble_pull(_Cap(), members=["03", "member_02"])
    assert ssh.pulled == ["/remote/ensemble/member_03/"]
    assert out["changed"] == ["member_03"] and out["up_to_date"] == ["member_02"]
    assert out["requested"] == ["member_02", "member_03"]


# ---- overview / headline ----------------------------------------------------- #

def test_overview_checks_flag_membership_and_gate_changes(env):
    base = ev.ensemble_dir()
    _member(base, 1)
    _member(base, 2)
    regime = _regime(env)
    _gate_dir(regime, "spatial_gate_combiner", ["01·psnr"])
    with open(os.path.join(regime, "eval_summary.json"), "w") as f:
        json.dump({"member_labels": ["01·psnr"], "ensemble_psnr": 58.0,
                   "spatial_gate_combiner_psnr": 59.0,
                   "eval_identity": {"combiner_fps": {"spatial_gate": "old"}}}, f)
    o = ev.ensemble_overview(False)
    checks = {c["id"]: c for c in o["checks"]}
    assert checks["eval-members"]["ok"] is False
    # The gate reads 01 (active); 02 joined after its fit: a note, not a failure.
    assert checks["gate-members"]["ok"] is False and checks["gate-members"]["tone"] == "info"
    assert "1 joined after this fit" in checks["gate-members"]["detail"]
    assert checks["eval-gate"]["ok"] is False
    assert checks["knee"]["ok"] is False
    assert o["headline"]["production"]["psnr"] == 59.0
    assert o["headline"]["mean"]["psnr"] == 58.0


def test_overview_gate_check_fails_when_a_read_member_left(env):
    base = ev.ensemble_dir()
    _member(base, 1)
    regime = _regime(env)
    _gate_dir(regime, "spatial_gate_combiner", ["01·psnr", "04·psnr"])
    checks = {c["id"]: c for c in ev.ensemble_overview(False)["checks"]}
    assert checks["gate-members"]["ok"] is False and checks["gate-members"]["tone"] == "warn"
    assert "04·psnr" in checks["gate-members"]["detail"]
    _gate_dir(regime, "spatial_gate_combiner", ["01·psnr", "04·psnr"], active_members=[0])
    checks = {c["id"]: c for c in ev.ensemble_overview(False)["checks"]}
    assert checks["gate-members"]["ok"] is True                 # 04 is never read


def test_summary_headline_single_metric():
    acc = ev._CombinerMetricAcc()
    hr = np.full((2, 2), 100.0, np.float32)
    members = np.stack([np.full_like(hr, 90.0), np.full_like(hr, 120.0)])
    acc.add(hr, members.mean(0), members, np.full_like(hr, 101.0))
    out = ev._summary_headline({"spatial_gate": acc}, ["01·psnr", "02·psnr"])
    assert out["psnr_metric"] == "vis_asinh" and out["best_member_label"] == "01·psnr"
    assert out["ensemble_gain_db"] == pytest.approx(out["ensemble_vs_mean_member_db"])
    assert out["ensemble_vs_best_member_db"] == pytest.approx(
        out["ensemble_psnr"] - out["best_member_psnr"])
    assert out["spatial_gate_combiner_psnr"] > out["ensemble_psnr"]
    assert "combiner_psnr" not in out            # the bare keys are the RBF's


# ---- train preview ----------------------------------------------------------- #

def test_train_preview_names_and_command(env):
    base = ev.ensemble_dir()
    _member(base, 5)
    er.load_registry(base)
    er.archive_member_entry(base, "member_05", zip_path="models/z.zip", commit=None)
    out = ev.train_command_preview({
        "mode": "add", "count": "2", "steps": "70000",
        "member_spec": json.dumps([{"loss": "l2", "asinh_knees": [0.1, 10], "output_knee": 10},
                                   {"loss": "l1"}])})
    assert out["member_names"] == ["member_06", "member_07"]     # tombstone 05 never reused
    assert out["array"] == {"tasks": 2, "max_parallel": 2}
    assert out["command"][:3] == ["python", "scripts/train_ensemble.py", "--mode"]
    assert "--member-names" in out["command"] and "member_06,member_07" in out["command"]
    assert "--base-seed" not in out["command"] and out["base_seed"] is None


def test_train_preview_refuses_what_the_submit_refuses(env):
    with pytest.raises(ValueError):
        ev.train_command_preview({"mode": "continue", "members": ""})
    with pytest.raises(ValueError):
        ev.train_command_preview({"mode": "add", "count": "1", "member_spec": "{not json"})
