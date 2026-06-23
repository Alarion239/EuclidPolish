"""Tests for euclid_polish.lensfinder.predictions (pure, no torch/TF)."""

from __future__ import annotations

import numpy as np

from euclid_polish.lensfinder import predictions as pr


def _write_pred(tmp_path, rows, header="id_str,p_notlens,p_lens"):
    p = tmp_path / "predictions.csv"
    lines = [header] + [",".join(str(x) for x in r) for r in rows]
    p.write_text("\n".join(lines) + "\n")
    return str(p)


class TestReadScores:
    def test_reads_p_lens_column(self, tmp_path):
        p = _write_pred(tmp_path, [("a__sr", 0.2, 0.8), ("b__sr", 0.7, 0.3)])
        s = pr.read_scores(p)
        assert s == {"a__sr": 0.8, "b__sr": 0.3}

    def test_falls_back_when_no_p_lens(self, tmp_path):
        # only a generic 'lens' column present
        p = _write_pred(tmp_path, [("a", 0.9)], header="id_str,lens")
        s = pr.read_scores(p)
        assert s["a"] == 0.9


class TestJoinAndArrays:
    def _catalog(self):
        return [
            {"id_str": "a__sr", "is_lens": 1, "theta_E_arcsec": "1.5",
             "recon": "sr", "split": "test"},
            {"id_str": "b__sr", "is_lens": 0, "theta_E_arcsec": "",
             "recon": "sr", "split": "test"},
            {"id_str": "c__sr", "is_lens": 1, "theta_E_arcsec": "2.0",
             "recon": "sr", "split": "train"},   # filtered out by split
        ]

    def test_join_restricts_to_split_and_drops_unmatched(self):
        scores = {"a__sr": 0.8, "b__sr": 0.1}     # c not predicted
        j = pr.join_scores(self._catalog(), scores, split="test")
        ids = {r["id_str"] for r in j}
        assert ids == {"a__sr", "b__sr"}
        a = next(r for r in j if r["id_str"] == "a__sr")
        assert a["is_lens"] == 1 and a["score"] == 0.8 and a["theta_E_arcsec"] == 1.5

    def test_scored_arrays(self):
        scores = {"a__sr": 0.8, "b__sr": 0.1}
        j = pr.join_scores(self._catalog(), scores, split="test")
        s, y, te = pr.scored_arrays(j)
        assert s.shape == y.shape == te.shape == (2,)
        assert set(y.tolist()) == {0, 1}
        assert np.isnan(te[y == 0]).all()         # galaxy has no theta_E
