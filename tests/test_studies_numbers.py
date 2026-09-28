"""The numbers files of a study (``euclid_polish/studies/numbers.py``):
shapes, per-field knee curves, integrated = ``integrated_psnr`` of the
per-field curves, and a missing compare report handled."""
from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest

from euclid_polish.eval.knee_psnr import KNEE_GRID_E, integrated_psnr
from euclid_polish.studies import numbers, stats
from euclid_polish.studies.store import StudyError
from euclid_polish.web.helpers import ensemble_viz as ev
from tests._studies_fixtures import LABELS, N_FIELDS, make_env


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    make_env(tmp_path, monkeypatch)
    return numbers.build(False)


def _csv(text: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text.decode("utf-8"))))


def test_bundle_has_every_numbers_file(bundle):
    assert set(bundle.files) == {"members.csv", "knee_psnr.json", "integrated.csv",
                                 "training_curves.json", "gate.json", "real.json"}
    assert all(isinstance(v, bytes) for v in bundle.files.values())


def test_knee_curves_are_per_model_per_field(bundle):
    knee = json.loads(bundle.files["knee_psnr.json"])
    assert knee["knees"] == list(KNEE_GRID_E) and knee["bands"] == ["VIS", "Y_E", "J_E", "H_E"]
    assert knee["fields"] == list(range(N_FIELDS))
    ids = [m["id"] for m in knee["models"]]
    assert ids == [*LABELS, "mean", "gate"]
    assert [m["kind"] for m in knee["models"]] == ["member"] * 3 + ["mean", "gate"]
    psnr = np.asarray(knee["psnr"])
    assert psnr.shape == (5, N_FIELDS, len(KNEE_GRID_E), 4)
    # The low-noise member beats the noisy one on every field and band at the
    # faint knees (the bright knees are dominated by the target blur).
    assert np.all(psnr[0][:, :5] > psnr[1][:, :5])


def test_the_mean_over_fields_is_the_leaderboard_curve(tmp_path, monkeypatch):
    make_env(tmp_path, monkeypatch)
    bundle = numbers.build(False)
    knee = json.loads(bundle.files["knee_psnr.json"])
    payload = ev.compute_knee_psnr_payload(False, force=True)
    board = np.asarray(payload["models"][0]["psnr"])
    np.testing.assert_allclose(np.asarray(knee["psnr"])[0].mean(axis=0), board, atol=2e-3)


def test_integrated_is_integrated_psnr_of_the_per_field_curves(bundle):
    knee = json.loads(bundle.files["knee_psnr.json"])
    psnr = np.asarray(knee["psnr"])
    rows = _csv(bundle.files["integrated.csv"])
    assert len(rows) == 5 * N_FIELDS * 4
    assert set(rows[0]) == {"model", "kind", "field", "band", "integrated_psnr"}
    for row in rows[:12]:
        m = [m["id"] for m in knee["models"]].index(row["model"])
        f = knee["fields"].index(int(row["field"]))
        b = knee["bands"].index(row["band"])
        expected = integrated_psnr(psnr[m, f])[b]
        assert float(row["integrated_psnr"]) == pytest.approx(expected, abs=1e-3)


def test_members_csv_carries_recipe_and_scores(bundle):
    rows = {r["label"]: r for r in _csv(bundle.files["members.csv"])}
    assert list(rows) == LABELS
    one = rows["01·psnr"]
    assert one["loss"] == "l1" and float(one["asinh_knee"]) == 10.0 and one["seed"] == "1001"
    assert one["step"] == "2000" and one["status"] == "complete" and one["fingerprint"]
    assert float(one["knee_int_VIS"]) > float(rows["02·psnr"]["knee_int_VIS"])
    assert one["gate_usage_VIS"] and one["used_by_gate"] == "True"


def test_snapshot_records_ensemble_gate_and_records(bundle):
    snap = bundle.snapshot
    members = snap["ensemble"]["members"]
    assert [m["label"] for m in members] == LABELS
    assert members[0]["origin"]["loss_norm"] == "l1" and members[0]["fingerprint"]
    assert snap["gate"]["name"] == "spatial_gate_p2" and snap["gate"]["fingerprint"]
    assert snap["gate"]["reads"] == LABELS
    assert snap["records"]["records_fp"] == "fp" and snap["records"]["indices"] == [0, 1, 2]
    assert snap["identity"]["labels"] == LABELS
    assert set(snap["identity"]["fingerprints"]) == set(LABELS)


def test_training_curves_and_gate_without_a_compare_report(bundle):
    curves = json.loads(bundle.files["training_curves.json"])
    assert [c["label"] for c in curves] == LABELS and curves[0]["psnr"]
    gate = json.loads(bundle.files["gate.json"])
    assert gate["compare"] is None and "no combiner comparison" in gate["compare_note"].lower()
    assert gate["diagnostic"]["available"] and gate["diagnostic"]["labels"] == LABELS
    assert [v["name"] for v in gate["variants"]] == ["spatial_gate_combiner"]
    real = json.loads(bundle.files["real.json"])
    assert real["experiments"] == []


def test_a_compare_report_is_kept(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    (env["regime"] / "spatial_gate_comparison.json").write_text(json.dumps({
        "id": "20260928-101010", "members": LABELS, "methods": ["mean"],
        "groups": {"natural": {}, "blackout": {}}}))
    gate = json.loads(numbers.build(False).files["gate.json"])
    assert gate["compare"]["id"] == "20260928-101010" and gate["compare_note"] is None


def test_stale_cubes_refuse_to_freeze(tmp_path, monkeypatch):
    make_env(tmp_path, monkeypatch, labels=LABELS[:2])
    with pytest.raises(StudyError) as err:
        numbers.build(False)
    assert err.value.code == 409 and "re-evaluate" in str(err.value).lower()


class _Stop(Exception):
    pass


def test_build_honours_cancellation_inside_the_knee_loop(tmp_path, monkeypatch):
    make_env(tmp_path, monkeypatch)
    calls = []

    def check():
        calls.append(1)
        if len(calls) > 2:                    # past the start, inside the per-field loop
            raise _Stop()

    with pytest.raises(_Stop):
        numbers.build(False, check=check)
    assert len(calls) == 3


def test_mean_integrated_ignores_nan_fields(bundle):
    knee = json.loads(bundle.files["knee_psnr.json"])
    knee["psnr"][0][1][3][0] = None                         # one knee of one field is NaN
    per_field = stats.field_integrated(knee["psnr"], knee["knees"])
    assert np.isnan(per_field[0, 1, 0])
    expected = np.mean(per_field[0, [0, 2], 0])
    assert numbers.mean_integrated(knee)[LABELS[0]]["VIS"] == pytest.approx(expected)


def test_fields_dropped_by_a_missing_cube_are_recorded(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    (env["cubes"] / "member2_00001.npy").unlink()
    bundle = numbers.build(False)
    knee = json.loads(bundle.files["knee_psnr.json"])
    assert knee["fields"] == [0, 2] and knee["dropped_fields"] == [1]
    assert any("1 of 3 test fields" in w for w in bundle.snapshot["warnings"])
