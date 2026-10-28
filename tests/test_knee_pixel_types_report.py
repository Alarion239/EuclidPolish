"""Pixel types in the console: a tiny run analysed and compared end to end
(the figures, summaries and listing), the file guard, request validation and
the routes (nothing starts on a GET)."""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from euclid_polish.web.helpers import knee_attribution_runs as runs
from euclid_polish.web.helpers import knee_pixel_types_report as report
from tests.test_knee_attribution_runs import _Cap, tiny  # noqa: F401 — the fixture

RUN = "20261025-000000-abcdef"


def _field(n=40, seed=0):
    """A synthetic field with sky, a galaxy and a star (LR n×n, HR 2n×2n)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:2 * n, :2 * n]
    galaxy = (300.0 * np.exp(-((yy - 30) ** 2 + (xx - 44) ** 2) / (2 * 6.0 ** 2)))[..., None] * np.ones(4)
    star = np.zeros_like(galaxy)
    star[60:62, 20:22] = 2000.0
    truth = galaxy + star
    lr = truth.reshape(n, 2, n, 2, 4).sum(axis=(1, 3)) + rng.normal(0, 3.0, (n, n, 4))
    return lr.astype(np.float32), galaxy.astype(np.float32), star.astype(np.float32)


@pytest.fixture()
def analysed_run(tiny, monkeypatch):
    lr, galaxy, star = _field()
    monkeypatch.setattr(runs, "synthetic_indices", lambda n: [3][:n])
    monkeypatch.setattr(runs, "_synthetic_fields", lambda indices: iter([("test/3", lr, galaxy, star)]))
    monkeypatch.setattr(runs, "_real_lr", lambda ref: lr)
    runs.job_knee_attribution(_Cap(), run_id=RUN, models=["universal:member_1", "universal:member_2"],
                              synthetic=1, tiles=["poster/t1"], per_class=6, seed=0, ablation=False,
                              sampling="mixed", per_band=150)
    return RUN


def test_an_analysis_writes_figures_a_summary_and_is_listed(analysed_run):
    cap = _Cap()
    out = report.job_analyse(cap, run_id=analysed_run, models=["universal:member_1"])
    assert out["folders"] == [f"{RUN}/universal__member_1"]
    assert cap.ticks[-1][0] == cap.ticks[-1][1]
    folder = report.report_root() / RUN / "universal__member_1"
    names = {p.name for p in folder.iterdir()}
    for kind in ("synthetic", "real"):
        assert {f"knee_map_{kind}.png", f"mechanics_{kind}.png", f"look_around_{kind}.png",
                f"types_by_brightness_{kind}.png", f"group_features_learned_{kind}.png"} <= names
        assert f"types_{kind}_VIS.png" in names
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["model"] == "universal:member_1" and summary["run_id"] == RUN
    assert set(summary["synthetic"]) == {"VIS", "Y_E", "J_E", "H_E"}
    vis = summary["synthetic"]["VIS"]
    assert vis["n"] > 0 and vis["noise_sigma_e"] > 0 and "bins_learned" in vis and vis["types"]
    listing = report.list_reports()
    row = listing["analyses"][0]
    assert row["folder"] == f"{RUN}/universal__member_1" and row["model"] == "universal:member_1"
    assert row["kinds"] == ["synthetic", "real"] and "knee_map_synthetic.png" in row["files"]
    assert listing["comparisons"] == [] and listing["overviews"] == []
    # the console's native data: one view per field set, one stamp file per band
    view = json.loads((folder / "view_synthetic.json").read_text())
    assert view["model"] == "universal:member_1" and len(view["centres_log"]) == len(view["bin_labels"])
    vis = view["bands"]["VIS"]
    assert len(vis["curve"]) == len(view["bin_labels"]) and vis["sigma"] > 0
    assert len(vis["heat"]["z"]) == len(vis["heat"]["x_edges"]) - 1
    assert set(vis["features"]) == {"raw", "learned"} and vis["types"]
    stamps = json.loads((folder / "stamps_synthetic_VIS.json").read_text())
    side = stamps["side"]
    row0 = next(r for rows in stamps["groups"].values() for r in rows)
    cell = next(iter(row0["cells"].values()))
    assert len(base64.b64decode(cell["sr"])) == side * side and len(cell["radial"]) == side // 2 + 1
    assert len(stamps["types"]) == len(vis["types"])


def test_a_comparison_pairs_two_models_of_a_run(analysed_run):
    params = report.parse_compare_request({"a": f"{RUN}|universal:member_1", "b": f"{RUN}|universal:member_2",
                                           "tag": "t1"})
    out = report.job_compare(_Cap(), **params)
    assert out["folder"] == "compare_member_1_vs_member_2_t1"
    folder = report.report_root() / out["folder"]
    names = {p.name for p in folder.iterdir()}
    assert {"compare_knee_synthetic.png", "compare_share_diff_synthetic.png",
            "compare_look_around_synthetic.png", "summary.json"} <= names
    summary = json.loads((folder / "summary.json").read_text())
    # the tiny fixture serves one network for every spec: the same pixels, Δ = 0
    assert summary["synthetic"]["VIS"]["n"] > 0
    deltas = [b["delta_median"] for b in summary["synthetic"]["VIS"]["bins"] if b["delta_median"] is not None]
    assert deltas and max(abs(d) for d in deltas) < 1e-6
    row = report.list_reports()["comparisons"][0]
    assert row["folder"] == out["folder"] and row["models"] == {"member_1": "universal:member_1",
                                                                "member_2": "universal:member_2"}
    assert row["tag"] == "t1"
    view = json.loads((folder / "view_synthetic.json").read_text())
    assert view["models"] == ["member_1", "member_2"] and set(view["curves"]["VIS"]) == {"member_1", "member_2"}
    assert np.allclose(np.nan_to_num(np.asarray(view["share_diff"]["VIS"]["matrix"], dtype=float)), 0.0)
    pairs = json.loads((folder / "pairs_synthetic_VIS.json").read_text())
    assert pairs["models"] == ["member_1", "member_2"] and pairs["n"] > 0


def test_an_overview_is_listed_with_its_curves(analysed_run):
    out = report.overview([(RUN, "universal:member_1"), (RUN, "universal:member_2")], "series")
    assert out.name == "overview_series"
    row = report.list_reports()["overviews"][0]
    assert row["folder"] == "overview_series" and row["tag"] == "series" and "synthetic" in row["kinds"]
    view = json.loads(report.report_file("overview_series", "view_synthetic.json").read_text())
    assert view["models"] == ["member_1", "member_2"] and "share_diff" not in view


def test_report_files_are_guarded(analysed_run):
    report.job_analyse(_Cap(), run_id=analysed_run, models=["universal:member_1"])
    good = report.report_file(f"{RUN}/universal__member_1", "knee_map_synthetic.png")
    assert good.is_file()
    for folder, name in [(f"{RUN}/universal__member_1", "../../x.png"), ("..", "summary.json"),
                         (f"{RUN}/../{RUN}", "summary.json"), ("compare_x", "a.txt"),
                         (f"{RUN}/universal__member_1", "missing.png"), ("not-a-run/m", "summary.json")]:
        with pytest.raises(KeyError):
            report.report_file(folder, name)


def test_a_milestone_comparison_folder_is_path_safe():
    folder = report.compare_folder("member_257@200000", "member_257@8000000", "training_length")
    assert folder == "compare_member_257__200000_vs_member_257__8000000_training_length"
    assert report._SAFE.match(folder)


def test_requests_are_validated(analysed_run, monkeypatch):
    assert report.parse_analyse_request({"run_id": RUN}) == {
        "run_id": RUN, "models": ["universal:member_1", "universal:member_2"]}
    assert report.parse_analyse_request({"run_id": RUN, "models": "universal:member_2"})["models"] == [
        "universal:member_2"]
    for form, message in [({"run_id": "nope"}, "unknown"), ({"run_id": RUN, "models": "universal:member_9"}, "not in run")]:
        with pytest.raises(ValueError, match=message):
            report.parse_analyse_request(form)
    for form, message in [({"a": RUN, "b": f"{RUN}|universal:member_2"}, "must be"),
                          ({"a": f"{RUN}|universal:member_1", "b": f"{RUN}|universal:member_1"}, "two different"),
                          ({"a": f"{RUN}|universal:member_1", "b": f"{RUN}|universal:member_2", "tag": "a/b"}, "tag")]:
        with pytest.raises(ValueError, match=message):
            report.parse_compare_request(form)
    # two runs with different seeds sampled different pixels
    other = dict(runs.get_run(RUN), id="20261025-000001-abcdef")
    other["params"] = dict(other["params"], seed=5)
    runs._write(other)
    with pytest.raises(ValueError, match="different seeds"):
        report.parse_compare_request({"a": f"{RUN}|universal:member_1", "b": f"{other['id']}|universal:member_2"})


def test_routes_list_serve_and_start_without_starting_on_get(analysed_run, monkeypatch):
    from euclid_polish.web.app import create_app
    from euclid_polish.web.jobs import REGISTRY

    report.job_analyse(_Cap(), run_id=analysed_run, models=["universal:member_1"])
    spawned = []
    monkeypatch.setattr(REGISTRY, "spawn", lambda label, target, kind=None: spawned.append((label, kind)) or "job-7")
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    body = client.get("/api/knee-attribution/pixel-types").get_json()
    assert [a["folder"] for a in body["analyses"]] == [f"{RUN}/universal__member_1"]
    png = client.get(f"/api/knee-attribution/pixel-types/file/{RUN}/universal__member_1/knee_map_synthetic.png")
    assert png.status_code == 200 and png.mimetype == "image/png" and png.data[:8] == b"\x89PNG\r\n\x1a\n"
    assert client.get(f"/api/knee-attribution/pixel-types/file/{RUN}/universal__member_1/nope.png").status_code == 404
    assert client.get("/api/knee-attribution/pixel-types/file/..%2F..%2Fsecret.json").status_code == 404
    assert spawned == []                                    # nothing starts on a GET
    r = client.post("/api/knee-attribution/pixel-types/analyse", data={"run_id": RUN})
    assert r.status_code == 200 and r.get_json()["job_id"] == "job-7"
    r = client.post("/api/knee-attribution/pixel-types/compare",
                    data={"a": f"{RUN}|universal:member_1", "b": f"{RUN}|universal:member_2"})
    assert r.status_code == 200 and r.get_json()["ok"]
    assert spawned == [(f"knee pixel types · 2 model(s) of run {RUN}", "knee-pixel-types"),
                       ("knee pixel types · compare member_1 vs member_2", "knee-pixel-types")]
    r = client.post("/api/knee-attribution/pixel-types/analyse", data={"run_id": "nope"})
    assert r.status_code == 400 and "unknown" in r.get_json()["error"]


def test_the_share_difference_aligns_two_knee_sets(tmp_path):
    rng = np.random.default_rng(1)

    def model(knees):
        n = 400
        share = rng.dirichlet(np.ones(len(knees)), n)
        samples = {"band": np.repeat(np.arange(4), n // 4), "blank": np.zeros(n, bool),
                   "bright": 10 ** rng.uniform(-1, 3, n), "share": share, "weight": np.ones(n)}
        return {"samples": samples, "knees": np.asarray(knees, np.float64)}

    six = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    eight = [0.001, 0.01, *six]
    path = tmp_path / "diff.png"
    report.compare_share_diff(path, {"member_218": model(six), "member_227": model(eight)}, "synthetic")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
