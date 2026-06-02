"""Smoke test for the unified catalog orchestrator
(``scripts/euclid_catalog_step.py``) — it imports and parses the mode +
query + verify args it forwards to the two phase scripts."""

from __future__ import annotations

import importlib
import sys

import pytest

orch = importlib.import_module("scripts.euclid_catalog_step")


def _parse(argv):
    old = sys.argv
    sys.argv = ["euclid_catalog_step.py", *argv]
    try:
        return orch.parse_args()
    finally:
        sys.argv = old


def test_defaults_to_both_mode():
    a = _parse([])
    assert a.mode == "both"
    assert a.verify_n == 40 and a.verify_size == 256


def test_parses_query_and_verify_knobs():
    a = _parse(["--mode", "verify", "--num-stars", "3000",
                "--magnitude-min", "16", "--magnitude-limit", "19",
                "--snr-min", "50", "--verify-n", "60", "--verify-size", "512"])
    assert a.mode == "verify"
    assert a.num_stars == 3000
    assert a.magnitude_min == 16 and a.magnitude_limit == 19
    assert a.snr_min == 50
    assert a.verify_n == 60 and a.verify_size == 512


def test_rejects_unknown_mode():
    with pytest.raises(SystemExit):
        _parse(["--mode", "bogus"])
