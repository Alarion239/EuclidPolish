from __future__ import annotations

import os

from euclid_polish.experiments.lens_isolation.evaluation import write_report


def test_write_report_emits_machine_readable_files_and_roc_plot(tmp_path):
    metrics = {"ensemble": {"auc": 1.0}, "zero": {"auc": 0.5}}
    rows = [
        {"index": 0, "approach": "ensemble", "label": 1, "theta_E_arcsec": 1.0, "score": 9.0},
        {"index": 1, "approach": "ensemble", "label": 0, "theta_E_arcsec": "", "score": 1.0},
        {"index": 0, "approach": "zero", "label": 1, "theta_E_arcsec": 1.0, "score": 0.0},
        {"index": 1, "approach": "zero", "label": 0, "theta_E_arcsec": "", "score": 0.0},
    ]
    paths = write_report(str(tmp_path), metrics, rows)
    assert set(paths) == {"metrics", "predictions", "roc"}
    assert all(os.path.isfile(path) for path in paths.values())
