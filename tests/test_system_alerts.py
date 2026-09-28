"""The Loop staleness service (helpers/system_alerts.py): Home's Loop strip and
System › Lineage read the same seven verdicts from it."""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from euclid_polish.web.helpers import system_alerts as sa

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC).timestamp()
DAY = 86400.0


def iso(days_ago: float) -> str:
    return datetime.fromtimestamp(NOW - days_ago * DAY, UTC).isoformat()


def check(check_id, state, title, **patch):
    return {"id": check_id, "label": check_id, "state": state, "title": title, "detail": None, "to": None, **patch}


def alerts(**patch):
    checks = [
        check("disk", "ok", "400 GiB free on the data disk"),
        check("real-sr", "ok", "All 12 real production SRs are current", facts={"current": 12, "stale": 0}),
        check("combiner", "ok", "Production gate fitted for the current 30 members", facts={"members": 30}),
        check("evaluation", "ok", "Evaluation current (30 members, test records)", facts={"evaluated_at": iso(0.1)}),
        check("knee", "ok", "PSNR-vs-knee curves current"),
        check("records-noise", "unknown", "Noise model of the local records unverified"),
        check("tracking", "ok", "Tracking log up to date"),
    ]
    return {"checks": [{**c, **patch.get(c["id"].replace("-", "_"), {})} for c in checks]}


def item(item_id, label, state, title, **patch):
    return {"id": item_id, "label": label, "state": state, "title": title, "group": "generation",
            "to": f"/synthetic/{label.lower()}", "facts": {}, "records": None, **patch}


def overview(items=None, gate=None):
    return {
        "gate": gate or {"ready": True, "blockers": []},
        "records": {"generated_at": iso(2)},
        "items": items if items is not None else [
            item("galaxy-model", "Galaxies", "ok", "Galaxy model active", facts={"version": 15},
                 records={"state": "current"}),
            item("star-prior", "Stars", "ok", "Stellar prior active", records={"state": "current"}),
            item("noise-model", "Noise", "ok", "Noise model v5",
                 facts={"noise_model": "euclid-q1-mer-noise-levels-dithered-bilinear-v5"},
                 records={"state": "unknown"}),
            item("psf", "PSF", "unknown", "ePSFs not synced to this machine"),
            item("galaxy-plots", "Galaxy plots", "ok", "Galaxy plots current", group="diagnostic"),
        ],
    }


MEMBERS = {
    "members": [{"name": f"member_{169 + i}", "status": "complete", "timeout": False, "step": 70000}
                for i in range(30)],
    "archived": [{"name": "member_168"}],
}
PLATES = {"runs": [{"tag": "prod-0926", "updated": iso(1), "renders": [
    {"band": "VIS", "model": "production", "model_label": "Production · spatial gate",
     "model_fingerprint": "p1", "created": iso(1), "sheet": "s.png", "tiles": []}]}]}
CATALOG = {"models": [{"spec": "production", "fingerprint": "p1"}]}


def stages(**patch):
    kw = {"alerts": alerts(), "overview": overview(), "members": MEMBERS, "jobs": [], "plates": PLATES,
          "catalog": CATALOG, "now": NOW, **patch}
    return {s["id"]: s for s in sa.loop_stages(**kw)}


def test_seven_stages_in_loop_order_each_with_one_short_reason():
    out = sa.loop_stages(alerts=alerts(), overview=overview(), members=MEMBERS, jobs=[], plates=PLATES,
                         catalog=CATALOG, now=NOW)
    assert [s["label"] for s in out] == ["Priors", "Records", "Members", "Evaluation", "Gate", "Real SR", "Figures"]
    assert {s["state"] for s in out} == {"current"}
    s = {x["id"]: x for x in out}
    assert s["priors"]["reason"] == "galaxies v15 · stars · noise v5"
    assert s["priors"]["detail"] == "Not checked here: PSF: ePSFs not synced to this machine."
    assert s["records"]["reason"] == "built 2 d ago, after the priors"
    assert "noise model" in s["records"]["detail"]
    assert s["members"]["reason"] == "30 active"
    assert s["evaluation"]["reason"] == "evaluated 2 h ago"
    assert s["gate"]["reason"] == "fitted for the 30 members"
    assert s["real-sr"]["reason"] == "12 current"
    assert s["figures"]["reason"] == "NEXUS plates from production"
    assert all(len(x["reason"]) <= sa.REASON_MAX for x in out)


def test_a_source_that_has_not_answered_reads_checking_never_current():
    s = stages(alerts=None, overview=None, members=None, plates=None)
    assert {x["state"] for x in s.values()} == {"loading"}
    assert {x["reason"] for x in s.values()} == {"checking"}


def test_priors_blocked_and_stale():
    blocked = stages(overview=overview(gate={"ready": False, "blockers": [
        {"id": "star-prior", "message": "activate a valid stellar calibration"}]}))
    assert blocked["priors"] == {**blocked["priors"], "state": "blocked",
                                 "reason": "blocked by 1: the stellar prior", "to": "/synthetic/status"}
    items = [({**i, "state": "warn", "title": "A newer galaxy candidate is not active",
               "detail": "v16 was fitted on 09-20."} if i["id"] == "galaxy-model" else i)
             for i in overview()["items"]]
    stale = stages(overview=overview(items))["priors"]
    assert stale["state"] == "stale" and stale["to"] == "/synthetic/galaxies"
    assert stale["reason"] == "Galaxies needs a look"
    assert stale["detail"].startswith("Galaxies: a newer galaxy candidate is not active.")
    short = [({**i, "state": "warn", "title": "Not active"} if i["id"] == "galaxy-model" else i)
             for i in overview()["items"]]
    assert stages(overview=overview(short))["priors"]["reason"] == "Galaxies: not active"


def test_records_stale_when_they_predate_a_prior_or_the_noise_model():
    items = [({**i, "records": {"state": "predates"}} if i["id"] == "star-prior" else i)
             for i in overview()["items"]]
    rec = stages(overview=overview(items))["records"]
    assert (rec["state"], rec["reason"], rec["to"]) == ("stale", "predate the stellar prior", "/synthetic/status")
    noise = stages(alerts=alerts(records_noise={"state": "bad", "title": "Training records use an older noise model"}))
    assert noise["records"]["reason"] == "use an older noise model"
    none = stages(overview={**overview(), "records": {"generated_at": None}})["records"]
    assert (none["state"], none["reason"]) == ("unknown", "no local records")


def test_members_finished_on_fasrc_and_not_pulled():
    jobs = [
        {"jobid": "1", "state": "COMPLETED", "mode": "add", "ended_at": iso(1),
         "member_names": ["member_199", "member_200", "member_168", "member_170"], "params": {}},
        {"jobid": "2", "state": "TIMEOUT", "mode": "add", "ended_at": iso(2), "member_names": ["member_201"],
         "params": {}},
        {"jobid": "3", "state": "RUNNING", "mode": "add", "ended_at": None, "member_names": ["member_202"],
         "params": {}},
        {"jobid": "4", "state": "COMPLETED", "mode": "add", "ended_at": iso(40), "member_names": ["member_150"],
         "params": {}},
        {"jobid": "5", "state": "COMPLETED", "mode": "add", "ended_at": iso(1), "member_names": ["member_210"],
         "params": {"starless": "1"}},
        {"jobid": "6", "state": "COMPLETED", "mode": "add", "ended_at": iso(1), "member_names": ["member_211"],
         "params": {"member_spec": json.dumps([{"starless": True}])}},
        {"jobid": "7", "state": "COMPLETED", "mode": "continue", "ended_at": iso(1), "target_steps": 90000,
         "member_names": ["member_171"], "params": {}},
    ]
    m = stages(jobs=jobs)["members"]
    assert (m["state"], m["reason"], m["to"]) == ("stale", "4 new on FASRC", "/models/starfull/members")
    assert m["detail"].startswith("members 171, 199–201 finished on FASRC")


def test_evaluation_gate_and_real_sr_from_their_checks():
    s = stages(alerts=alerts(
        evaluation={"state": "warn", "title": "The evaluation predates the current members"},
        combiner={"state": "warn", "title": "The production gate does not match the members"},
        real_sr={"state": "warn", "title": "450 real SR products are stale", "facts": {"current": 0, "stale": 450}},
    ))
    assert (s["evaluation"]["state"], s["evaluation"]["reason"]) == ("stale", "predates the current members")
    assert s["gate"]["reason"] == "does not match the members"
    assert (s["real-sr"]["reason"], s["real-sr"]["to"]) == ("450 stale", "/sky/targets?state=stale")
    # With the per-source counts, the chip lands on exactly the sets it counted
    # (Targets' "All" also holds the catalogue sets, which this check does not count).
    by_source = stages(alerts=alerts(real_sr={"state": "warn", "title": "450 real SR products are stale", "facts": {
        "current": 0, "stale": 450, "sources": [
            {"source": "nexus", "stale": 445}, {"source": "tile", "stale": 1}, {"source": "field", "stale": 0},
            {"source": "poster", "stale": 4}]}}))
    assert by_source["real-sr"]["to"] == "/sky/targets?state=stale&set=nexus%2Cposter%2Ccached"
    knee = stages(alerts=alerts(knee={"state": "warn", "title": "PSNR-vs-knee curves are stale"}))
    assert knee["evaluation"]["reason"] == "knee curves are stale"


def test_figures_stale_for_legacy_or_earlier_nexus_plates_and_stale_galaxy_plots():
    legacy = {"runs": [{"tag": "old", "renders": [{"band": "temp", "model": None, "legacy": True,
                                                  "model_label": "minibatched convex all-knee RBF",
                                                  "sheet": None, "tiles": []}]}]}
    f = stages(plates=legacy)["figures"]
    assert (f["state"], f["reason"], f["to"]) == ("stale", "NEXUS plates use a legacy SR",
                                                  "/figures/plates?plate=nexus")
    assert "minibatched convex all-knee RBF" in f["detail"]
    earlier = {"runs": [{**PLATES["runs"][0], "renders": [{**PLATES["runs"][0]["renders"][0],
                                                           "model_fingerprint": "p0"}]}]}
    assert stages(plates=earlier)["figures"]["reason"] == "NEXUS plates predate this fit"
    items = [({**i, "state": "warn"} if i["id"] == "galaxy-plots" else i) for i in overview()["items"]]
    plots = stages(overview=overview(items))["figures"]
    assert (plots["reason"], plots["to"]) == ("galaxy plots need a rebuild", "/synthetic/status")
    assert stages(plates={"runs": []})["figures"]["reason"] == "no NEXUS plates yet"


def test_relative_and_member_range_match_the_console_formats():
    assert sa.relative(iso(2), NOW) == "2 d ago"
    assert sa.relative(iso(0.0001), NOW) == "just now"
    assert sa.relative("not a date", NOW) is None
    assert sa.member_range(["member_199", "member_200", "member_201", "member_205"]) == "members 199–201, 205"
    assert sa.member_range(["member_195", "member_196"]) == "members 195, 196"
    assert sa.member_range(["x"]) is None


@pytest.fixture
def sources(monkeypatch):
    calls = {"n": 0}

    def overview_payload(*, check_records_noise):
        calls["n"] += 1
        return overview()

    monkeypatch.setattr(sa, "overview_payload", overview_payload)
    monkeypatch.setattr(sa, "members_payload", lambda starless: MEMBERS)
    monkeypatch.setattr(sa, "training_jobs", lambda: [])
    monkeypatch.setattr(sa.nexus_plates, "list_runs", lambda: PLATES)
    monkeypatch.setattr(sa.model_catalog, "catalog_payload", lambda: CATALOG)
    sa.clear_cache()
    yield calls
    sa.clear_cache()


def test_loop_payload_gathers_memoises_and_recomputes_when_fresh(sources):
    out = sa.loop_payload(alerts=alerts, check_records_noise=dict)
    assert [s["id"] for s in out["stages"]] == list(sa.STAGE_ORDER)
    assert out["counts"]["current"] == 7 and out["errors"] == {}
    sa.loop_payload(alerts=alerts, check_records_noise=dict)
    assert sources["n"] == 1
    sa.loop_payload(alerts=alerts, check_records_noise=dict, fresh=True)
    assert sources["n"] == 2


def test_a_broken_source_is_named_and_never_reads_current(sources, monkeypatch):
    def boom(starless):
        raise OSError("members.json unreadable")

    monkeypatch.setattr(sa, "members_payload", boom)
    out = sa.loop_payload(alerts=alerts, check_records_noise=dict, fresh=True)
    members = next(s for s in out["stages"] if s["id"] == "members")
    assert members["state"] == "unknown"
    assert out["errors"] == {"members": "OSError: members.json unreadable"}
