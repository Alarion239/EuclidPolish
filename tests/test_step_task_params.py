"""Schema-driven FASRC step task parameters (contract C5).

Every registered step declares ``task_params`` — the knobs its
``build_command`` reads — so the SPA renders them generically. The submit
route fills absent task params from the schema defaults and rejects invalid
ones (400); ``/api/fasrc/steps/status`` publishes the schema plus the task
params of the newest successful run (``last_params``).
"""

from __future__ import annotations

import json
import time

import pytest

from euclid_polish.observability import JobLog, JobRecord
from euclid_polish.web import fasrc_jobs, fasrc_pipeline, fasrc_queue, job_config
from euclid_polish.web import remote as web_remote
from euclid_polish.web.app import create_app
from euclid_polish.web.fasrc_pipeline import REGISTRY, TaskParam, TaskParamError

RESOURCES = {"n_cpus": "1", "n_gpus": "0", "memory": "4G", "time_limit": "0:30:00"}


# ---------------------------------------------------------------------------
# TaskParam
# ---------------------------------------------------------------------------

def test_task_param_dict_follows_the_contract():
    param = TaskParam("num_stars", "int", 10000, help="how many", min=1)
    assert param.to_dict() == {
        "name": "num_stars", "type": "int", "default": 10000, "help": "how many",
        "min": 1,
    }
    choice = TaskParam("band", "choice", "VIS", help="b", choices=("VIS", "H"),
                       required=True)
    assert choice.to_dict() == {
        "name": "band", "type": "choice", "default": "VIS", "help": "b",
        "choices": ["VIS", "H"], "required": True,
    }


@pytest.mark.parametrize("param,raw,value", [
    (TaskParam("n", "int", 1), "12", 12),
    (TaskParam("n", "int", 1), "12.0", 12),
    (TaskParam("x", "float", 1.0), "0.25", 0.25),
    (TaskParam("b", "bool", False), "on", True),
    (TaskParam("b", "bool", False), "0", False),
    (TaskParam("b", "bool", False), "", False),
    (TaskParam("s", "str", ""), " a b ", "a b"),
    (TaskParam("c", "choice", "x", choices=("x", "y")), "y", "y"),
    (TaskParam("j", "json", None), '[{"loss": "l2"}]', [{"loss": "l2"}]),
])
def test_task_param_parse(param, raw, value):
    assert param.parse(raw) == value


@pytest.mark.parametrize("param,raw", [
    (TaskParam("n", "int", 1), "abc"),
    (TaskParam("n", "int", 1), "1.5"),
    (TaskParam("n", "int", 1, min=1), "0"),
    (TaskParam("x", "float", 1.0, max=1.0), "1.5"),
    (TaskParam("x", "float", 1.0), "nan"),
    (TaskParam("b", "bool", False), "maybe"),
    (TaskParam("c", "choice", "x", choices=("x", "y")), "z"),
    (TaskParam("j", "json", None), "[oops"),
])
def test_task_param_rejects_invalid_values(param, raw):
    with pytest.raises(TaskParamError) as exc:
        param.parse(raw)
    assert param.name in str(exc.value)


def test_task_param_form_value_round_trips():
    assert TaskParam("b", "bool", True).form_value(True) == "1"
    assert TaskParam("b", "bool", False).form_value(False) == "0"
    assert TaskParam("n", "int", 3).form_value(3) == "3"
    assert TaskParam("j", "json", [1]).form_value([1]) == "[1]"


# ---------------------------------------------------------------------------
# Every registered step
# ---------------------------------------------------------------------------

@pytest.fixture
def hermetic_build(monkeypatch):
    """``build_command`` without local caches (TNG picks, member registry)."""
    monkeypatch.setattr(fasrc_pipeline, "_tng_select",
                        lambda mode, n, temperature: [str(i) for i in range(n)])
    monkeypatch.setattr(fasrc_pipeline, "next_member_names",
                        lambda _base, count: [f"member_{900 + i}" for i in range(count)])


@pytest.mark.parametrize("step", REGISTRY.all(), ids=lambda s: s.step_id)
def test_every_step_declares_a_valid_schema(step):
    names = [param.name for param in step.task_params]
    assert names, f"{step.step_id} declares no task_params"
    assert len(names) == len(set(names))
    resource_keys = {"partition", "n_cpus", "n_gpus", "memory", "time_limit"}
    assert not resource_keys & set(names)
    config_keys = set(job_config.FASRC_STEP_PARAMS.get(step.step_id, {}))
    assert not config_keys & set(names), "job-config knobs belong to /config"
    for param in step.task_params:
        assert param.type in {"int", "float", "str", "bool", "choice", "json"}
        assert param.help
        if param.type == "choice":
            assert param.choices and param.default in param.choices
        if param.default is not None:
            assert param.parse(param.form_value(param.default)) == param.default


@pytest.mark.parametrize("step", REGISTRY.all(), ids=lambda s: s.step_id)
def test_every_step_builds_its_command_from_the_schema_defaults(
        step, hermetic_build):
    params = {**step.defaults.to_dict(),
              **{k: str(v) for k, v in step.defaults.to_dict().items()},
              "galaxy_density_arcmin2": "100"}
    params.update(step.fill_task_params({}))
    argv = step.build_command(params)
    assert argv and argv[0].startswith("scripts/")


def test_fill_task_params_keeps_explicit_values_and_blanks():
    step = REGISTRY.get("euclid_query")
    filled = step.fill_task_params({"num_stars": "500", "magnitude_limit": ""})
    assert filled["num_stars"] == "500"
    assert filled["magnitude_limit"] == ""          # explicit blank = unset
    assert filled["magnitude_min"] == "18"
    assert filled["snr_min"] == "50"


@pytest.mark.parametrize("blank", ["", " ", "\t"])
def test_blank_num_stars_takes_the_schema_default_never_200(blank):
    """Clearing the field in a generic form posts ``num_stars=""``; that must
    ask for the schema's 10,000 stars, never the old 200 overwrite."""
    step = REGISTRY.get("euclid_query")
    filled = step.fill_task_params({"num_stars": blank})
    assert filled["num_stars"] == "10000"
    joined = " ".join(step.build_command(filled))
    assert "--num-stars 10000" in joined and "200" not in joined
    # Even a direct build (no fill) never falls back to 200.
    direct = " ".join(step.build_command({"num_stars": blank.strip()}))
    assert "--num-stars 10000" in direct


def _blank_param_cases():
    for step in REGISTRY.all():
        for param in step.task_params:
            if param.default is not None:
                yield pytest.param(step, param, id=f"{step.step_id}.{param.name}")


@pytest.mark.parametrize("step,param", list(_blank_param_cases()))
def test_a_blank_task_param_takes_its_default_unless_blank_means_unset(step, param):
    """A blank value is the schema default, like an absent one — except the
    params whose help gives blank its own meaning (``blank="unset"``)."""
    for blank in ("", "  "):
        filled = step.fill_task_params({param.name: blank})
        if param.blank == "unset":
            assert filled[param.name] == ""
        else:
            assert filled[param.name] == step.fill_task_params({})[param.name]


def test_only_the_euclid_query_cuts_treat_blank_as_no_cut():
    unset = {(step.step_id, param.name) for step in REGISTRY.all()
             for param in step.task_params
             if param.default is not None and param.blank == "unset"}
    assert unset == {("euclid_query", "magnitude_min"),
                     ("euclid_query", "magnitude_limit"),
                     ("euclid_query", "snr_min")}
    argv = " ".join(REGISTRY.get("euclid_query").build_command(
        REGISTRY.get("euclid_query").fill_task_params(
            {"magnitude_min": " ", "snr_min": ""})))
    assert "--magnitude-min" not in argv and "--snr-min" not in argv
    assert "--magnitude-limit 19" in argv


@pytest.mark.parametrize("step", REGISTRY.all(), ids=lambda s: s.step_id)
def test_every_step_builds_with_every_task_param_blank(step, hermetic_build):
    """Whitespace-only values are blanks too: they never reach ``int(" ")``."""
    filled = step.fill_task_params({param.name: " " for param in step.task_params})
    params = {**step.defaults.to_dict(), "galaxy_density_arcmin2": "100", **filled}
    assert step.build_command(params)[0].startswith("scripts/")


def test_fill_task_params_rejects_invalid_values():
    with pytest.raises(TaskParamError, match="num_stars"):
        REGISTRY.get("euclid_query").fill_task_params({"num_stars": "lots"})
    with pytest.raises(TaskParamError, match="band"):
        REGISTRY.get("tng_grid").fill_task_params({"band": "UV"})


def test_euclid_query_defaults_are_the_last_real_run():
    defaults = {p.name: p.default for p in REGISTRY.get("euclid_query").task_params}
    assert defaults == {"num_stars": 10000, "magnitude_min": 18,
                        "magnitude_limit": 19, "snr_min": 50}
    argv = REGISTRY.get("euclid_query").build_command(
        REGISTRY.get("euclid_query").fill_task_params({}))
    joined = " ".join(argv)
    assert "--num-stars 10000" in joined
    assert "--magnitude-min 18" in joined
    assert "--magnitude-limit 19" in joined
    assert "--snr-min 50" in joined


def test_ensemble_train_emits_the_multi_knee_knobs(hermetic_build):
    step = REGISTRY.get("ensemble_train")
    params = step.fill_task_params({
        "asinh_knees": "0.1,1,10,100,1000,10000", "output_knee": "10",
        "knee_loss": "balanced", "evaluate_every": "2500",
    })
    joined = " ".join(step.build_command(params))
    assert "--asinh-knees 0.1,1,10,100,1000,10000" in joined
    assert "--output-knee 10" in joined
    assert "--knee-loss balanced" in joined
    assert "--evaluate-every 2500" in joined
    plain = " ".join(step.build_command(step.fill_task_params({})))
    assert "--asinh-knees" not in plain and "--knee-loss" not in plain


def test_last_task_params_is_the_newest_completed_run(tmp_path):
    log = JobLog(str(tmp_path / "log.csv"))
    step = REGISTRY.get("euclid_query")
    for jobid, state, stars in (("1", "COMPLETED", "5000"), ("2", "FAILED", "7"),
                                ("3", "COMPLETED", "10000"), ("4", "RUNNING", "9")):
        log.record_submission(JobRecord(
            jobid=jobid, submitted_at=f"2026-09-2{jobid}T00:00:00Z",
            step_id="euclid_query", label="q",
            params_json=json.dumps({"num_stars": stars, "magnitude_min": "18",
                                    "magnitude_limit": "", "step_id": "euclid_query",
                                    "_private": "x"})))
        log.record_post_mortem(jobid, {"state": state})
    assert step.last_task_params(log.history_for_step("euclid_query")) == {
        "num_stars": 10000, "magnitude_min": 18.0, "magnitude_limit": None,
    }
    assert step.last_task_params([]) is None


@pytest.mark.parametrize("step_id,name", [
    ("ensemble_train", "base_seed"),
    ("psf_rotation_pool", "seed"),
    ("poster_cutout", "seed"),
])
def test_last_task_params_never_prefills_a_fresh_entropy_seed(step_id, name):
    """A "blank = fresh entropy" seed is resolved at submit and stored as a
    concrete number; prefilling it would replay the previous batch's seeds
    (ensemble members are seeded ``base_seed + i`` → duplicate members)."""
    step = REGISTRY.get(step_id)
    row = {"state": "COMPLETED",
           "params_json": json.dumps({name: 4039766209, "step_id": step_id})}
    assert step.last_task_params([row]) == {name: None}


def test_last_task_params_keeps_a_deterministic_sampling_seed():
    step = REGISTRY.get("vis_noise_sample")
    row = {"state": "COMPLETED", "params_json": json.dumps({"seed": "7"})}
    assert step.last_task_params([row]) == {"seed": 7}


def test_blank_seed_ensemble_run_yields_a_blank_seed_prefill(
        client, captured, hermetic_build, monkeypatch):
    monkeypatch.setattr(fasrc_pipeline.population_calibration, "active_star",
                        lambda: None)
    response = client.post("/api/fasrc/steps/ensemble_train/submit", data={
        "confirm": "yes", **RESOURCES, "partition": "gpu", "n_gpus": "1",
        "mode": "add", "count": "2", "base_seed": ""})
    assert response.status_code == 200, response.get_json()
    stored = captured["params"]
    assert isinstance(stored["base_seed"], int)           # resolved at submit
    fasrc_jobs.JOBLOG.record_submission(JobRecord(
        jobid="4242", step_id="ensemble_train", label="t",
        params_json=json.dumps({k: v for k, v in stored.items()
                                if k not in RESOURCES and k != "partition"})))
    fasrc_jobs.JOBLOG.record_post_mortem("4242", {"state": "COMPLETED"})
    body = client.get("/api/fasrc/steps/status").get_json()
    last = {s["step_id"]: s for s in body["steps"]}["ensemble_train"]["last_params"]
    assert last["base_seed"] is None
    assert last["count"] == 2


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

class _Up:
    def is_connected(self) -> bool:
        return True

    def run(self, cmd, timeout=None, binary=False):
        return 0, "", ""


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(fasrc_jobs, "JOBLOG", JobLog(str(tmp_path / "log.csv")))
    monkeypatch.setattr(web_remote.STATE, "ssh", _Up())
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def captured(monkeypatch):
    seen: dict = {}

    def fake_submit(_ssh, *, cfg, built, label, params, step_id):
        seen.update(built=built, params=params, step_id=step_id)
        return "4242", {"ok": True, "jobid": "4242", "step_id": step_id}

    monkeypatch.setattr(fasrc_jobs, "submit_sbatch_script", fake_submit)
    return seen


def test_steps_status_publishes_the_schema_and_last_params(client, tmp_path):
    fasrc_jobs.JOBLOG.record_submission(JobRecord(
        jobid="77", step_id="euclid_query", label="q",
        params_json=json.dumps({"num_stars": "321", "snr_min": "40"})))
    fasrc_jobs.JOBLOG.record_post_mortem("77", {"state": "COMPLETED"})

    body = client.get("/api/fasrc/steps/status").get_json()
    by_id = {step["step_id"]: step for step in body["steps"]}
    assert set(by_id) == {step.step_id for step in REGISTRY.all()}
    query = by_id["euclid_query"]
    assert {p["name"] for p in query["task_params"]} == {
        "num_stars", "magnitude_min", "magnitude_limit", "snr_min"}
    assert query["last_params"] == {"num_stars": 321, "snr_min": 40.0}
    assert by_id["tng_grid"]["last_params"] is None
    for step in body["steps"]:
        for param in step["task_params"]:
            assert {"name", "type", "default", "help"} <= set(param)


def test_euclid_query_submit_without_task_params_uses_the_schema_defaults(
        client, captured):
    response = client.post("/api/fasrc/steps/euclid_query/submit",
                           data={"confirm": "yes", **RESOURCES})
    assert response.status_code == 200, response.get_json()
    body = captured["built"]["body"]
    assert "--num-stars" in body and "10000" in body
    assert "--magnitude-min" in body and "--magnitude-limit" in body
    assert "200" not in captured["built"]["params"]["num_stars"]
    assert captured["params"]["num_stars"] == "10000"


def test_submit_rejects_an_invalid_task_param_before_any_ssh(client, captured):
    response = client.post("/api/fasrc/steps/euclid_query/submit",
                           data={"confirm": "yes", **RESOURCES, "num_stars": "0"})
    assert response.status_code == 400
    assert "num_stars" in response.get_json()["error"]
    assert captured == {}
    response = client.post("/api/fasrc/steps/tng_grid/submit",
                           data={"confirm": "yes", **RESOURCES, "band": "UV"})
    assert response.status_code == 400
    assert "band" in response.get_json()["error"]


def test_blank_workers_submit_uses_the_schema_default(client, captured):
    response = client.post("/api/fasrc/steps/download_euclid_cutouts/submit",
                           data={"confirm": "yes", **RESOURCES, "workers": ""})
    assert response.status_code == 200, response.get_json()
    assert captured["params"]["workers"] == "8"
    assert "--workers" in captured["built"]["body"]


def test_blank_num_stars_submit_never_asks_for_200_stars(client, captured):
    response = client.post("/api/fasrc/steps/euclid_query/submit",
                           data={"confirm": "yes", **RESOURCES, "num_stars": " "})
    assert response.status_code == 200, response.get_json()
    assert captured["params"]["num_stars"] == "10000"


@pytest.fixture
def busy_lane(monkeypatch):
    """A cluster job is running: submits are queued, not sbatch'd."""
    monkeypatch.setattr(fasrc_queue.QUEUE, "active_is_running", lambda _db: True)
    return fasrc_queue.QUEUE


def test_queued_blank_task_params_are_stored_resolved(client, captured, busy_lane):
    response = client.post("/api/fasrc/steps/download_euclid_cutouts/submit",
                           data={"confirm": "yes", **RESOURCES, "workers": ""})
    body = response.get_json()
    assert response.status_code == 200, body
    assert body["queued"] is True and captured == {}
    (item,) = busy_lane.items
    assert item["spec"]["form"]["workers"] == "8"


@pytest.mark.parametrize("form,error", [
    ({"mode": "continue"}, "at least one member"),                 # prepare_params
    ({"mode": "continue", "members": "member_01",
      "continue_basis": "target"}, "target_steps"),                # build_command
])
def test_an_unbuildable_submit_is_refused_even_when_it_would_queue(
        client, captured, busy_lane, hermetic_build, monkeypatch, form, error):
    """Queued specs are built only at promotion, where a failure halts the
    whole queue — so the submit dry-runs the build first and 400s."""
    monkeypatch.setattr(fasrc_pipeline.population_calibration, "active_star",
                        lambda: None)
    response = client.post("/api/fasrc/steps/ensemble_train/submit", data={
        "confirm": "yes", **RESOURCES, "partition": "gpu", "n_gpus": "1", **form})
    assert response.status_code == 400, response.get_json()
    assert error in response.get_json()["error"]
    assert busy_lane.items == [] and captured == {}


def test_an_unbuildable_submit_is_refused_on_a_free_lane_too(
        client, captured, hermetic_build, monkeypatch):
    monkeypatch.setattr(fasrc_pipeline.population_calibration, "active_star",
                        lambda: None)
    response = client.post("/api/fasrc/steps/ensemble_train/submit", data={
        "confirm": "yes", **RESOURCES, "partition": "gpu", "n_gpus": "1",
        "mode": "continue"})
    assert response.status_code == 400
    assert captured == {}
    assert fasrc_queue.QUEUE.items == []


class _Squeue(_Up):
    def __init__(self, text: str) -> None:
        self.text = text

    def run(self, cmd, timeout=None, binary=False):
        if cmd.startswith("squeue"):
            return 0, self.text, ""
        return 0, "", ""


def test_current_submission_lists_every_live_job(client, monkeypatch):
    for jobid, state in (("501", "RUNNING"), ("502", "PENDING"), ("400", "DONE")):
        fasrc_jobs.DB.insert(jobid, label=f"job {jobid}", params={},
                             script_path=".", log_path=".", err_path=".")
        fasrc_jobs.DB.update_state(jobid, state=state)
        time.sleep(0.01)
    squeue = ("501|a|RUNNING|0:10|1:00:00|1|holy1|2026-09-26T01:00:00\n"
              "502|b|PENDING|0:00|1:00:00|1|Priority|N/A\n")
    monkeypatch.setattr(web_remote.STATE, "ssh", _Squeue(squeue))
    monkeypatch.setattr(fasrc_jobs, "reconcile_with_squeue", lambda *_a, **_k: {})

    body = client.get("/api/fasrc/current-submission").get_json()

    assert body["ok"] is True
    live = {row["jobid"]: row for row in body["live"]}
    assert set(live) == {"501", "502"}
    assert live["502"]["reason"] == "Priority"
    assert live["501"]["nodes"] == "1"
    assert body["current"]["job"]["jobid"] == "502"     # newest live


def test_current_submission_live_includes_jobs_older_than_the_recent_window(
        client, monkeypatch):
    """C5: *every* PENDING/RUNNING job — not just those among the newest rows."""
    fasrc_jobs.DB.insert("600", label="old runner", params={},
                         script_path=".", log_path=".", err_path=".")
    fasrc_jobs.DB.update_state("600", state="RUNNING")
    for offset in range(60):
        jobid = str(700 + offset)
        fasrc_jobs.DB.insert(jobid, label=f"job {jobid}", params={},
                             script_path=".", log_path=".", err_path=".")
        fasrc_jobs.DB.update_state(jobid, state="COMPLETED")
    fasrc_jobs.DB.insert("900", label="new", params={},
                         script_path=".", log_path=".", err_path=".")
    monkeypatch.setattr(web_remote.STATE, "ssh", _Squeue(""))
    monkeypatch.setattr(fasrc_jobs, "reconcile_with_squeue", lambda *_a, **_k: {})

    body = client.get("/api/fasrc/current-submission").get_json()

    assert [row["jobid"] for row in body["live"]] == ["900", "600"]
    assert [row["jobid"] for row in fasrc_jobs.DB.list_live()] == ["900", "600"]


def test_current_submission_live_is_empty_without_live_jobs(client, monkeypatch):
    monkeypatch.setattr(web_remote.STATE, "ssh", _Squeue(""))
    body = client.get("/api/fasrc/current-submission").get_json()
    assert body["current"] is None
    assert body["live"] == []


@pytest.mark.parametrize("step_id,form", [
    # Train members (add / continue / fork) as the current page posts them.
    ("ensemble_train", {
        "mode": "add", "count": "1", "steps": "60000",
        "member_spec": '[{"loss":"l2","noise_aug":0,"bootstrap":0}]',
        "array_max_parallel": "2", "batch_size": "4", "forward_onthefly": "1",
        "hr_crop_size": "256", "crops_per_field": "8", "psf_subset": "64",
        "psf_warp_prob": "1", "psf_warp_alpha_max": "20", "psf_warp_sigma": "3",
        "saturation_mask_prob": "0.2", "target_psf_fwhm_arcsec": "0.066"}),
    ("ensemble_train", {
        "mode": "continue", "members": "member_195,member_196",
        "continue_basis": "target", "target_steps": "100000",
        "array_max_parallel": "2", "psf_subset": "0"}),
    ("ensemble_train", {"mode": "fork", "count": "2", "fork_from": "member_02",
                        "member_spec": "[{},{}]"}),
    # Records: targeted split regeneration.
    ("synthetic_generate", {"regenerate_splits": "", "extra_flags": ""}),
    ("synthetic_generate", {"regenerate_splits": "validate,test",
                            "extra_flags": "--regenerate-splits=validate,test"}),
    # Step cards' own controls.
    ("download_tng_skirt", {"workers": "32"}),
    ("vis_noise_sample", {"regenerate_catalog": "1"}),
    ("archive_field_sample", {"force_redownload": "0"}),
])
def test_current_spa_payloads_pass_validation(step_id, form, hermetic_build):
    step = REGISTRY.get(step_id)
    filled = step.fill_task_params(form)
    for key, value in form.items():
        assert filled[key] == value                 # posted strings are kept
    params = {**step.defaults.to_dict(), "galaxy_density_arcmin2": "100", **filled}
    assert step.build_command(params)[0].startswith("scripts/")
