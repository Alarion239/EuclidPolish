"""Focused tests for the active synthetic-field detection diagnostics."""

from __future__ import annotations

import numpy as np

import euclid_polish.web.helpers.tng_prior as tng_prior
from euclid_polish.web.helpers.tng_prior import (
    DetectionAccumulator,
    detection_payload,
)


def test_detection_accumulator_matches_current_truth_records(monkeypatch):
    monkeypatch.setattr(
        tng_prior,
        "_segment_centroids",
        lambda plane: ([(5.0, 6.0), (10.0, 12.0), (25.0, 25.0)], 1),
    )
    accumulator = DetectionAccumulator()
    accumulator.add(
        np.zeros((32, 32), dtype=np.float32),
        [
            {"type": "galaxy", "x_pix": 10.0, "y_pix": 12.0},
            {"type": "star", "x_pix": 20.0, "y_pix": 24.0},
            {"type": "galaxy", "x_pix": 100.0, "y_pix": 100.0},
        ],
    )

    assert accumulator.payload() == {
        "positive": [3],
        "negative": [1],
        "matched_galaxies": [1],
        "matched_stars": [1],
        "truth_galaxies": [1],
    }


def test_detection_payload_records_current_settings():
    synthetic = DetectionAccumulator(
        positive=[3], negative=[1], matched_galaxies=[1],
        matched_stars=[1], truth_galaxies=[2],
    )
    real = DetectionAccumulator(
        positive=[4], negative=[1], matched_galaxies=[0],
        matched_stars=[0], truth_galaxies=[0],
    )

    payload = detection_payload(synthetic, real)

    assert payload["settings"]["band"] == "VIS"
    assert payload["settings"]["threshold_sigma"] == 4.0
    assert payload["synthetic"] == synthetic.payload()
    assert payload["real"] == real.payload()
