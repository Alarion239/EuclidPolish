"""Publication figures of a study (``euclid_polish/studies/render.py``): each
of the six charts renders PNG / PDF / SVG from a tiny frozen study, and its
CSV is exactly the plotted numbers."""
from __future__ import annotations

import copy
import csv
import io

import numpy as np
import pytest

from euclid_polish.eval.knee_psnr import integrated_psnr
from euclid_polish.studies import freeze, numbers, render, stats
from euclid_polish.studies.store import StudyError, StudyStore
from euclid_polish.web.helpers import experiments
from tests._studies_fixtures import LABELS, FakeRemote, make_env

MAGIC = {"png": b"\x89PNG", "pdf": b"%PDF", "svg": b"<?xml"}


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    tmp_path = tmp_path_factory.mktemp("study")
    try:
        make_env(tmp_path, mp)
        mp.setattr(experiments, "free_bytes", lambda _path: 10 ** 15)
        store = StudyStore(tmp_path / "studies")
        sid = freeze.start(store, name="Render", note="", starless=False, fids=[],
                           connected=False)
        freeze.run(store, sid, ssh=FakeRemote(tmp_path / "remote", connected=False),
                   remote_tracking_dir="/n/a/b/c/tracking")
        data = render.StudyData.load(store, sid)
    finally:
        mp.undo()
    return data


def _rows(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


@pytest.mark.parametrize("chart", ["knee", "integrated", "paired", "gate", "training"])
@pytest.mark.parametrize("fmt", ["png", "pdf", "svg"])
def test_each_chart_renders_every_format(study, chart, fmt):
    body = render.render(chart, study, render.Selection(), output_format=fmt, dpi=150)
    assert body[:len(MAGIC[fmt])] == MAGIC[fmt] or (fmt == "svg" and b"<svg" in body[:400])


def test_knee_csv_is_the_plotted_curves(study):
    columns, rows = render.table("knee", study, render.Selection())
    assert columns == ["series", "kind", "n_members", "band", "knee_e", "psnr", "lo", "hi"]
    member = [r for r in rows if r["series"] == LABELS[0] and r["band"] == "VIS"]
    psnr = np.asarray(study.knee["psnr"])[0].mean(axis=0)[:, 0]
    assert [round(r["psnr"], 4) for r in member] == pytest.approx(np.round(psnr, 4).tolist())
    assert {r["series"] for r in rows} >= {"mean", "gate"}
    grouped = render.table("knee", study, render.Selection(group="loss"))[1]
    assert {r["series"] for r in grouped if r["kind"] == "group"} == {"l1", "l2"}
    l1 = next(r for r in grouped if r["series"] == "l1")
    assert l1["n_members"] == 2 and l1["lo"] <= l1["psnr"] <= l1["hi"]


def test_integrated_rows_follow_the_recipe(study):
    columns, rows = render.table("integrated", study, render.Selection())
    assert columns[:4] == ["series", "kind", "loss", "training_knee"]
    one = next(r for r in rows if r["series"] == LABELS[2] and r["band"] == "VIS")
    assert one["loss"] == "l1" and one["training_knee"] == "100"
    psnr = np.asarray(study.knee["psnr"])[2]
    expected = np.mean([integrated_psnr(psnr[f])[0] for f in range(psnr.shape[0])])
    assert one["integrated_psnr"] == pytest.approx(expected, abs=1e-6)


def test_paired_uses_the_bootstrap_over_fields(study):
    sel = render.Selection(group="loss", reference="mean", seed=3)
    columns, rows = render.table("paired", study, sel)
    assert "mean_delta" in columns and "lo" in columns and "n_resamples" in columns
    row = next(r for r in rows if r["target"] == "l2" and r["band"] == "VIS")
    integ = study.per_field_integrated()                  # (models, fields, bands)
    ids = [m["id"] for m in study.knee["models"]]
    expected = stats.paired_bootstrap(integ[ids.index(LABELS[1]), :, 0],
                                      integ[ids.index("mean"), :, 0], seed=3)
    assert row["mean_delta"] == pytest.approx(expected["mean"])
    assert row["lo"] == pytest.approx(expected["lo"]) and row["seed"] == 3
    assert row["n_fields"] == 3 and row["reference"] == "mean"


def test_paired_against_the_best_member(study):
    rows = render.table("paired", study, render.Selection(reference="best"))[1]
    best = rows[0]["reference"]
    assert best in LABELS
    assert all(r["mean_delta"] == 0 for r in rows if r["target"] == best)


def test_gate_rows_per_member_and_family(study):
    columns, rows = render.table("gate", study, render.Selection(group="loss"))
    assert columns == ["level", "name", "family", "band", "weight", "uniform", "source"]
    families = {r["name"]: r["weight"] for r in rows if r["level"] == "family" and r["band"] == "VIS"}
    assert families["l1"] == pytest.approx(2 / 3, abs=1e-3)
    sources = render.table("gate", study, render.Selection(source="sources"))[1]
    assert next(r for r in sources if r["name"] == LABELS[0] and r["band"] == "VIS")["weight"] == 0.5


def test_training_rows_are_the_logged_series(study):
    rows = render.table("training", study, render.Selection(group="loss"))[1]
    assert {r["group"] for r in rows} == {"l1", "l2"}
    assert [r["step"] for r in rows if r["member"] == LABELS[0]] == [1000, 2000]


def test_selection_restricts_members_and_rejects_unknown_ones(study):
    rows = render.table("integrated", study, render.Selection(members=[LABELS[0]]))[1]
    assert {r["series"] for r in rows if r["kind"] == "member"} == {LABELS[0]}
    with pytest.raises(StudyError):
        render.table("integrated", study, render.Selection(members=["99·psnr"]))
    with pytest.raises(StudyError):
        render.table("knee", study, render.Selection(group="colour"))


def test_a_chart_without_its_data_says_what_is_missing(study):
    with pytest.raises(StudyError) as err:
        render.table("real", study, render.Selection())
    assert err.value.code == 404 and "Sky › Compare" in str(err.value)
    with pytest.raises(StudyError):
        render.render("nope", study, render.Selection(), output_format="png")
    with pytest.raises(StudyError):
        render.render("knee", study, render.Selection(), output_format="gif")


def test_real_chart_from_a_frozen_experiment(study):
    real = {"experiments": [{"id": "20260927-180342-e3237d", "created": "2026-09-27",
                             "tiles": ["poster/x"], "specs": {
                                 "production": {"label": "Production", "member_label": None,
                                                "summary": _summary(1.0)},
                                 "member:member_01": {"label": "Member 01·psnr",
                                                      "member_label": LABELS[0],
                                                      "summary": _summary(3.0)}}}]}
    data = render.StudyData(manifest=study.manifest, knee=study.knee, curves=study.curves,
                            gate=study.gate, real=real, selections=[])
    columns, rows = render.table("real", data, render.Selection())
    assert "hole_pct" in columns and {r["model"] for r in rows} == {"production", LABELS[0]}
    body = render.render("real", data, render.Selection(), output_format="png", dpi=150)
    assert body[:4] == MAGIC["png"]


def _summary(holes):
    return {"n_tiles": 1, "per_band": {b: {"hole_pct": holes, "pct_R_lt_0p8": 2 * holes,
                                           "median_R": 1.1, "flux_ratio": 0.99}
                                       for b in ("VIS", "Y_E", "J_E", "H_E")}}


def test_csv_text_roundtrips(study):
    columns, rows = render.table("knee", study, render.Selection())
    text = render.csv_text(columns, rows)
    assert _rows(text)[0].keys() == set(columns) or list(_rows(text)[0]) == columns


def test_one_nan_aware_rule_across_charts_and_numbers(study):
    knee = copy.deepcopy(study.knee)
    knee["psnr"][0][1][3][0] = None
    data = render.StudyData(manifest=study.manifest, knee=knee, curves=study.curves,
                            gate=study.gate, real=study.real, selections=[])
    row = next(r for r in render.table("integrated", data, render.Selection())[1]
               if r["series"] == LABELS[0] and r["band"] == "VIS")
    assert row["integrated_psnr"] == pytest.approx(numbers.mean_integrated(knee)[LABELS[0]]["VIS"])
    paired = render.table("paired", data, render.Selection(reference="mean"))[1]
    assert next(r for r in paired if r["target"] == LABELS[0] and r["band"] == "VIS")[
        "n_fields"] == 2
    same = render.table("paired", data, render.Selection(members=[LABELS[0]],
                                                         reference=LABELS[0]))[1]
    assert same[0]["n_fields"] == 2 and same[0]["mean_delta"] == 0


def test_paired_without_two_finite_fields_is_a_clear_409(study):
    knee = copy.deepcopy(study.knee)
    for f in (0, 1):
        knee["psnr"][0][f][3][0] = None                     # only one VIS field left
    data = render.StudyData(manifest=study.manifest, knee=knee, curves=study.curves,
                            gate=study.gate, real=study.real, selections=[])
    with pytest.raises(StudyError) as err:
        render.table("paired", data, render.Selection(members=[LABELS[0]], reference="mean"))
    assert err.value.code == 409 and "VIS" in str(err.value) and "2 fields" in str(err.value)


def test_study_data_is_memoized_per_manifest(tmp_path, monkeypatch):
    make_env(tmp_path, monkeypatch)
    monkeypatch.setattr(experiments, "free_bytes", lambda _path: 10 ** 15)
    store = StudyStore(tmp_path / "studies")
    sid = freeze.start(store, name="Memo", note="", starless=False, fids=[], connected=False)
    freeze.run(store, sid, ssh=FakeRemote(tmp_path / "remote", connected=False),
               remote_tracking_dir="/n/a/b/c/tracking")
    first = render.StudyData.load(store, sid)
    reads = []
    real_read = store.read_json
    monkeypatch.setattr(store, "read_json", lambda *a: reads.append(a) or real_read(*a))
    second = render.StudyData.load(store, sid)
    assert reads == [] and second.knee is first.knee
    store.set_selections(sid, [{"name": "one", "members": [LABELS[0]]}])
    assert render.StudyData.load(store, sid).selections[0]["name"] == "one"
