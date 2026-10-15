"""Knee-attribution runs and routes: request validation, a whole run on a tiny
random multi-knee network over a hand-built synthetic field and a real tile,
the stored record and field images, and the API (nothing starts on a GET)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.eval import knee_attribution as ka
from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.web.helpers import knee_attribution_runs as runs
from euclid_polish.web.helpers import real_tiles
from euclid_polish.web.jobs import JobCancelled

KNEES = (0.01, 1.0, 100.0)


class _Cap:
    def __init__(self):
        self.ticks = []

    def tick(self, done, total, label=""):
        self.ticks.append((done, total, label))

    def check_cancelled(self):
        return None


class _Member:
    """What a run needs of a member: the full-field SR in electrons."""

    def __init__(self, net):
        self.net = net

    def upsample_array(self, lr_e):
        stretched = tf.constant(ka.stretch(lr_e[None], self.net.knees))
        return self.net.predict_electrons(stretched).numpy()[0]


@pytest.fixture()
def tiny(monkeypatch, tmp_path):
    """A tiny learned-knee network served for any spec; runs write under tmp."""
    tf.keras.utils.set_random_seed(3)
    model = wdsr(scale=2, num_res_blocks=1, nchan_in=4 * len(KNEES), nchan_out=4,
                 input_knees=len(KNEES), learned_output_knee=10.0, eager_weight_norm_init=True)
    net = ka.KneeNet(model=model, knees=KNEES, depth=1, output="electrons")
    monkeypatch.setattr(runs, "root", lambda: tmp_path / "knee_attribution")
    monkeypatch.setattr(runs, "_load", lambda spec: (net, _Member(net)))
    return net


def _synthetic_field(n=24, seed=0):
    rng = np.random.default_rng(seed)
    lr = np.abs(rng.normal(20, 5, (n, n, 4))).astype(np.float32)
    lr[5, 5, :] = 0.0                                   # a blanked (saturated) core
    galaxy = np.full((2 * n, 2 * n, 4), 3.0, np.float32)
    galaxy[20:24, 20:24] = 400.0                         # a galaxy core
    star = np.zeros_like(galaxy)
    star[10:12, 10:12] = 900.0                           # a saturated star (LR 5,5 blanked)
    star[30:32, 30:32] = 20.0                            # a star wing
    return lr, galaxy, star


def test_parse_request_validates(monkeypatch):
    monkeypatch.setattr(runs, "candidate_models", lambda: [
        {"spec": "universal:member_9", "available": True, "reason": None},
        {"spec": "member:member_3", "available": False, "reason": "a single-knee model"}])
    monkeypatch.setattr(real_tiles, "get_entry", lambda s, i: object())
    ok = runs.parse_request({"models": "universal:member_9", "synthetic": "5", "tiles": "poster/t1",
                             "per_class": "8", "seed": "2"})
    assert ok == {"models": ["universal:member_9"], "synthetic": 5, "tiles": ["poster/t1"],
                  "per_class": 8, "seed": 2, "ablation": True}
    for form, message in [({}, "at least one"),
                          ({"models": "member:member_3", "synthetic": "1"}, "single-knee"),
                          ({"models": "universal:member_9", "synthetic": "101"}, "synthetic"),
                          ({"models": "universal:member_9", "synthetic": "0"}, "pick synthetic"),
                          ({"models": "universal:member_9", "per_class": "0", "synthetic": "1"}, "per_class")]:
        with pytest.raises(ValueError, match=message):
            runs.parse_request(form)


def test_a_run_writes_samples_aggregates_and_field_images(tiny, monkeypatch):
    lr, galaxy, star = _synthetic_field()
    monkeypatch.setattr(runs, "synthetic_indices", lambda n: [7][:n])
    monkeypatch.setattr(runs, "_synthetic_fields",
                        lambda indices: iter([("test/7", lr, galaxy, star)]))
    monkeypatch.setattr(runs, "_real_lr", lambda ref: lr)
    cap = _Cap()
    result = runs.job_knee_attribution(cap, run_id="20261015-000000-abcdef", models=["universal:member_9"],
                      synthetic=1, tiles=["poster/t1"], per_class=3, seed=1)
    assert result["status"] == "done" and result["errors"] == {}
    record = runs.get_run("20261015-000000-abcdef")
    entry = record["models"]["universal:member_9"]
    assert entry["knees"] == list(KNEES) and entry["depth"] == 1
    kinds = {f["ref"]: f["kind"] for f in entry["fields"]}
    assert kinds == {"test/7": "synthetic", "poster/t1": "real"}
    synth = next(f for f in entry["fields"] if f["ref"] == "test/7")
    assert record["version"] == runs.RECORD_VERSION == 2
    np.testing.assert_allclose(synth["noise_sigma_e"], ka.noise_sigma(lr), rtol=1e-3)
    assert "knee_share_by_input" not in synth["samples"]       # aggregates only
    classes = set(synth["samples"]["class"])
    assert {"saturated", "galaxy core", "star wing"} <= classes
    for share in synth["samples"]["knee_share"]:
        assert len(share) == len(KNEES) and abs(sum(share) - 1.0) < 1e-3
    agg = entry["aggregate"]
    assert set(agg) == {"synthetic", "real"}
    assert "galaxy core" in agg["synthetic"]["by_class"]["0"]
    split = agg["synthetic"]["by_class"]["0"]["galaxy core"]["knee_share_by_input"]
    assert np.asarray(split).shape == (4, len(KNEES))
    np.testing.assert_allclose(np.asarray(split).sum(axis=1), 1.0, rtol=1e-3)
    assert set(agg["real"]["by_class"]["0"]) <= set(ka.REAL_CLASSES)
    assert len(agg["synthetic"]["by_brightness"]["edges_e"]) == len(ka.LOG_EDGES_E)
    # the field explorer's image
    png = runs.field_png(record["id"], runs.slug("universal:member_9"), runs.slug("test/7"),
                         band="VIS", layer="sr")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert cap.ticks[-1][0] == cap.ticks[-1][1]
    assert [r["id"] for r in runs.list_runs()] == ["20261015-000000-abcdef"]


def test_a_failing_model_is_recorded_not_fatal(tiny, monkeypatch):
    def load(spec):
        raise ValueError("boom")
    monkeypatch.setattr(runs, "_load", load)
    monkeypatch.setattr(runs, "synthetic_indices", lambda n: [])
    result = runs.job_knee_attribution(_Cap(), run_id="20261015-000001-abcdef", models=["universal:member_9"],
                      synthetic=0, tiles=[], per_class=2, seed=0)
    assert result["status"] == "failed"
    assert "boom" in result["errors"]["universal:member_9"]


def test_a_cancelled_run_says_so_and_keeps_its_finished_models(tiny, monkeypatch):
    lr, galaxy, star = _synthetic_field()
    monkeypatch.setattr(runs, "synthetic_indices", lambda n: [7][:n])
    monkeypatch.setattr(runs, "_synthetic_fields",
                        lambda indices: iter([("test/7", lr, galaxy, star)]))

    class _CancelOnSecondModel(_Cap):
        def check_cancelled(self):
            if self.ticks:                              # the first model's field is done
                raise JobCancelled("job")

    cap = _CancelOnSecondModel()
    with pytest.raises(JobCancelled):
        runs.job_knee_attribution(cap, run_id="20261015-000003-abcdef",
                                  models=["universal:member_1", "universal:member_2"],
                                  synthetic=1, tiles=[], per_class=2, seed=0)
    record = runs.get_run("20261015-000003-abcdef")
    assert record["status"] == "cancelled" and record["finished"]
    assert list(record["models"]) == ["universal:member_1"]


def test_routes_list_read_and_start_without_starting_on_get(tiny, monkeypatch):
    from euclid_polish.web.app import create_app
    from euclid_polish.web.jobs import REGISTRY

    spawned = []
    monkeypatch.setattr(REGISTRY, "spawn", lambda label, target, kind=None: spawned.append(
        (label, kind)) or "job-1")
    monkeypatch.setattr(runs, "candidate_models", lambda: [
        {"spec": "universal:member_9", "label": "u9", "kind": "universal", "knees": list(KNEES),
         "depth": 1, "available": True, "reason": None}])
    monkeypatch.setattr(runs, "synthetic_indices", lambda n: list(range(n)))
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    assert client.get("/api/knee-attribution").get_json() == {"runs": []}
    assert client.get("/api/knee-attribution/models").get_json()["models"][0]["spec"] == "universal:member_9"
    assert client.get("/api/knee-attribution/20261015-000000-abcdef").status_code == 404
    assert client.get("/api/knee-attribution/nonsense").status_code == 404
    assert spawned == []                                    # nothing starts on a GET
    r = client.post("/api/knee-attribution", data={"models": "universal:member_9", "synthetic": "3"})
    body = r.get_json()
    assert r.status_code == 200 and body["ok"] and body["job_id"] == "job-1"
    assert runs._RUN_ID.fullmatch(body["run_id"])
    assert spawned == [("knee attribution · 1 model(s) × 3 field(s)", "knee-attribution")]
    r = client.post("/api/knee-attribution", data={"models": "", "synthetic": "3"})
    assert r.status_code == 400 and "at least one" in r.get_json()["error"]


def test_record_is_valid_json_without_nan(tiny, monkeypatch, tmp_path):
    lr, galaxy, star = _synthetic_field(seed=4)
    monkeypatch.setattr(runs, "synthetic_indices", lambda n: [0])
    monkeypatch.setattr(runs, "_synthetic_fields", lambda indices: iter([("test/0", lr, galaxy, star)]))
    runs.job_knee_attribution(_Cap(), run_id="20261015-000002-abcdef", models=["universal:member_9"],
             synthetic=1, tiles=[], per_class=2, seed=0)
    text = (tmp_path / "knee_attribution" / "20261015-000002-abcdef.json").read_text()
    json.loads(text)                                      # strict JSON (allow_nan=False wrote it)
    assert "NaN" not in text
