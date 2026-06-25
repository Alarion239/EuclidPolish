"""Tests for the lens-finder → /evaluation wiring (consumer side, no torch)."""

from __future__ import annotations

import math
import os

from euclid_polish.config import Config
from euclid_polish.eval import lensfinder_eval as le
from euclid_polish.web.app import create_app


def _write_scores(run_dir, rows):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "lens_scores.csv"), "w") as f:
        f.write("id,grade,p_lens_lr,p_lens_sr,p_lens_hr\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


class TestReadScores:
    def test_read_and_per_object(self, tmp_path):
        _write_scores(str(tmp_path), [
            ("a", "A", 0.2, 0.8, ""),         # real lens: no HR truth
            ("s", "syn-lens", 0.3, 0.9, 0.95),
        ])
        rows = le.read_lens_scores(str(tmp_path))
        assert len(rows) == 2
        a = next(r for r in rows if r["id"] == "a")
        assert a["sr"] == 0.8 and math.isnan(a["hr"])      # blank → NaN
        by = le.per_object_plens(str(tmp_path))
        assert by["s"]["hr"] == 0.95 and by["a"]["lr"] == 0.2

    def test_absent_file(self, tmp_path):
        assert le.read_lens_scores(str(tmp_path)) == []
        assert le.per_object_plens(str(tmp_path)) == {}


class TestRenderSummary:
    def test_none_without_scores(self, tmp_path):
        assert le.render_lensfinder_summary(str(tmp_path),
                                            str(tmp_path / "o.png")) is None

    def test_renders_all_panels(self, tmp_path):
        # A/B/C + a separable synthetic pair so every panel has data.
        rows = [("a1", "A", 0.6, 0.8, ""), ("b1", "B", 0.5, 0.6, ""),
                ("c1", "C", 0.4, 0.5, "")]
        for i in range(6):
            rows.append((f"sl{i}", "syn-lens", 0.5 + 0.04 * i, 0.7 + 0.03 * i, 0.9))
            rows.append((f"sg{i}", "syn-gal", 0.2 + 0.02 * i, 0.1 + 0.02 * i, 0.1))
        _write_scores(str(tmp_path), rows)
        out = le.render_lensfinder_summary(str(tmp_path), str(tmp_path / "o.png"))
        assert out and os.path.isfile(out) and os.path.getsize(out) > 0


class TestRoutes:
    def _client(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
        app = create_app()
        app.config.update(TESTING=True)
        return app.test_client()

    def test_summary_404_without_scores(self, tmp_path, monkeypatch):
        c = self._client(tmp_path, monkeypatch)
        os.makedirs(Config.EVAL_RESULTS_DIR, exist_ok=True)
        assert c.get("/api/evaluation/lensfinder-summary").status_code == 404

    def test_summary_served_when_scored(self, tmp_path, monkeypatch):
        c = self._client(tmp_path, monkeypatch)
        _write_scores(Config.EVAL_RESULTS_DIR,
                      [("a", "A", 0.2, 0.8, ""), ("s", "syn-lens", 0.3, 0.9, 0.9),
                       ("g", "syn-gal", 0.6, 0.2, 0.1)])
        r = c.get("/api/evaluation/lensfinder-summary")
        assert r.status_code == 200 and r.mimetype == "image/png"

    def test_run_lensfinder_env_missing_hint(self, tmp_path, monkeypatch):
        from euclid_polish.web.routes import evaluation as evmod
        c = self._client(tmp_path, monkeypatch)
        os.makedirs(Config.EVAL_RESULTS_DIR, exist_ok=True)
        monkeypatch.setattr(evmod, "_zoobot_python", lambda: None)
        r = c.post("/api/evaluation/run-lensfinder", data={})
        assert r.status_code == 400 and "Zoobot env" in r.get_json()["error"]


def test_gal_group_registered():
    from euclid_polish.eval import lensfinder_eval
    from euclid_polish.eval.zoobot_morph import GROUP_COLORS, _group_color
    assert "gal" in lensfinder_eval.GROUPS
    assert "gal" in GROUP_COLORS
    # distinct from every other group's colour
    assert _group_color("gal") not in {GROUP_COLORS[g] for g in GROUP_COLORS if g != "gal"}
