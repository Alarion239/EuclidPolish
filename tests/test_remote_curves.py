"""Live curves of members still on FASRC (web/helpers/remote_curves.py):
which job members get an entry, the event-stream read and parse, the entry
shape Models › Curves draws, the cache, and the curves endpoint merge."""
from __future__ import annotations

import json
import subprocess

import pytest

from euclid_polish.config import Config
from euclid_polish.web.app import create_app
from euclid_polish.web.helpers import remote_curves as rc
from euclid_polish.web.routes import ensemble as routes

NOW = 2_000_000_000.0
KNEES = [0.1, 1, 10, 100, 1000, 10000]


def _row(jobid, names, *, age_s=3600.0, array=True, specs=None, **params):
    params = {"mode": "add", "member_names": ",".join(names), "steps": "200000",
              "loss": "l1", "array_count": len(names) if array else 1,
              "member_spec": json.dumps(specs or []), **params}
    path = ("/logs/ensemble-train-x-%A_%a.events" if array
            else f"/logs/ensemble-train-{jobid}.events")
    return {"jobid": jobid, "submitted_at": NOW - age_s, "events_path": path,
            "params_json": json.dumps(params)}


SPEC = {"loss": "l2", "asinh_knees": KNEES, "knee_loss": "balanced", "output_knee": 10,
        "learn_output_knee": True, "bootstrap": 0}


def _metric(step, psnr, *, total=200000, member=0):
    return {"kind": "metric", "ts": 1.0, "value": {
        "step": step, "psnr_stretched": psnr, "psnr_vis": psnr - 5, "psnr_y_e": psnr + 3,
        "psnr_j_e": psnr, "psnr_h_e": psnr - 1, "loss": 0.1, "combined_loss": 0.09,
        "gnorm_avg": 0.3, "gnorm_max": 2.0, "duration_s": 137.0, "is_baseline": "",
        "total": total, "member": member}}


def test_array_job_members_resolve_their_task_streams_and_spec_facets():
    rows = [_row("51626411", ["member_207", "member_208"],
                 specs=[{**SPEC, "num_res_blocks": 64}, {**SPEC, "num_res_blocks": 32}])]
    tasks = rc.remote_tasks(rows, set(), now=NOW)
    assert [t["name"] for t in tasks] == ["member_207", "member_208"]
    assert [t["events_path"] for t in tasks] == [
        "/logs/ensemble-train-x-51626411_0.events", "/logs/ensemble-train-x-51626411_1.events"]
    f = tasks[0]["facets"]
    assert f["blocks"] == 64 and tasks[1]["facets"]["blocks"] == 32
    assert f["loss_norm"] == "l2" and f["asinh_knees"] == KNEES and f["knee_loss"] == "balanced"
    assert f["learned_output_knee"] is True and f["output_knee"] is None
    assert f["target_steps"] == 200000 and f["starless"] is False


def test_single_job_uses_its_own_stream_and_top_level_params():
    rows = [_row("51435884", ["member_206"], array=False, loss="l2", num_res_blocks="16",
                 asinh_knee="10")]
    (task,) = rc.remote_tasks(rows, set(), now=NOW)
    assert task["events_path"] == "/logs/ensemble-train-51435884.events"
    assert task["facets"]["blocks"] == 16 and task["facets"]["asinh_knee"] == 10.0
    assert task["facets"]["asinh_knees"] is None and task["facets"]["knee_loss"] is None


def test_pulled_archived_old_and_superseded_members_are_left_out():
    rows = [_row("3", ["member_210"], array=False),            # newest: keeps 210
            _row("2", ["member_209", "member_210"]),
            _row("1", ["member_150"], age_s=rc.MAX_AGE_S + 1, array=False)]
    tasks = rc.remote_tasks(rows, {"member_209"}, now=NOW)
    assert [(t["name"], t["jobid"]) for t in tasks] == [("member_210", "3")]


def test_unresolved_or_missing_streams_are_skipped():
    row = _row("9", ["member_211"], array=False)
    row["events_path"] = None
    assert rc.remote_tasks([row], set(), now=NOW) == []
    row["events_path"] = "/logs/x-%A.events"                     # a template, not a file
    assert rc.remote_tasks([row], set(), now=NOW) == []


def test_fetch_command_reads_only_metric_lines_of_every_stream(tmp_path):
    a, b = tmp_path / "a.events", tmp_path / "b.events"
    a.write_text("\n".join([json.dumps({"kind": "resource", "value": {"gpu": 70}}),
                            json.dumps(_metric(1000, 42.0)),
                            json.dumps(_metric(2000, 46.0))]) + "\n")
    b.write_text(json.dumps(_metric(1000, 41.0, member=1)) + "\n{broken\n")
    missing = tmp_path / "missing.events"
    paths = [str(a), str(missing), str(b)]
    out = subprocess.run(["bash", "-c", rc._fetch_command(paths)],
                         capture_output=True, text=True, check=True).stdout
    streams = rc.parse_metric_streams(out, 3)
    assert [[v["step"] for v in s] for s in streams] == [[1000, 2000], [], [1000]]


def test_entries_carry_series_progress_and_the_finished_flag():
    tasks = rc.remote_tasks([_row("7", ["member_207", "member_208"],
                                  specs=[{**SPEC, "num_res_blocks": 64}, SPEC], steps="3000")],
                            set(), now=NOW)
    streams = [[_metric(1000, 42.0)["value"], _metric(2000, 46.0)["value"],
                _metric(2000, 46.5)["value"]],                     # a rollback re-logs 2000
               [_metric(s, 40.0 + s / 1000)["value"] for s in (1000, 2000, 3000)]]
    first, second = rc.remote_entries(tasks, streams)
    assert first["name"] == "member_207" and first["label"] == "207·psnr"
    assert first["psnr"] == [[1000, 42.0], [2000, 46.5]]
    assert first["band_psnr"]["VIS"] == [[1000, 37.0], [2000, 41.5]]
    assert first["loss"] == first["loss_series"] == [[1000, 0.09], [2000, 0.09]]
    assert first["step_time"][0] == [1000, 137.0]
    assert first["remote"] is True and first["jobid"] == "7" and first["blocks"] == 64
    assert first["last_step"] == 2000 and first["finished"] is False
    assert second["last_step"] == 3000 and second["finished"] is True


def test_a_stream_without_validations_gives_no_entry():
    tasks = rc.remote_tasks([_row("7", ["member_207"], array=False)], set(), now=NOW)
    assert rc.remote_entries(tasks, [[]]) == []


class _SSH:
    def __init__(self, text, connected=True):
        self.text, self.connected, self.calls = text, connected, 0

    def is_connected(self):
        return self.connected

    def run(self, cmd, timeout=60):
        self.calls += 1
        return 0, self.text, ""


@pytest.fixture
def fake_fasrc(monkeypatch):
    rows = [_row("51626411", ["member_207", "member_208"],
                 specs=[{**SPEC, "num_res_blocks": 64}, SPEC])]
    text = (f"{rc._RECORD_SEP}0\n{json.dumps(_metric(1000, 42.0))}\n"
            f"{rc._RECORD_SEP}1\n{json.dumps(_metric(1000, 41.0, member=1))}\n")
    ssh = _SSH(text)
    monkeypatch.setattr(rc.STATE, "ssh", ssh)
    monkeypatch.setattr(rc.fasrc_jobs.DB, "list_by_step", lambda step: rows)
    monkeypatch.setattr(rc.time, "time", lambda: NOW)
    monkeypatch.setattr(rc, "_cache", {"at": 0.0, "key": None, "members": []})
    return ssh


def test_remote_curves_read_once_per_ttl(fake_fasrc, monkeypatch):
    out = rc.remote_training_curves(["member_100"], [])
    assert [m["name"] for m in out] == ["member_207", "member_208"]
    assert rc.remote_training_curves(["member_100"], []) == out and fake_fasrc.calls == 1
    monkeypatch.setattr(rc.time, "time", lambda: NOW + rc.TTL_S + 1)
    rc.remote_training_curves(["member_100"], [])
    assert fake_fasrc.calls == 2


def test_pulled_or_archived_members_are_not_fetched(fake_fasrc):
    assert [m["name"] for m in rc.remote_training_curves(["member_207"], ["member_208"])] == []
    assert fake_fasrc.calls == 0


def test_no_connection_means_no_remote_curves(fake_fasrc):
    fake_fasrc.connected = False
    assert rc.remote_training_curves([], []) == [] and fake_fasrc.calls == 0


def test_curves_endpoint_appends_remote_members(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR", str(tmp_path / "ckpt/wdsr"))
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    seen = {}

    def fake_remote(active, archived):
        seen.update(active=list(active), archived=list(archived))
        return [{"name": "member_207", "remote": True}]

    monkeypatch.setattr(routes, "remote_training_curves", fake_remote)
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        body = client.get("/ensemble/training-curves.json").get_json()
    assert body["members"] == [{"name": "member_207", "remote": True}]
    assert seen == {"active": [], "archived": []}
