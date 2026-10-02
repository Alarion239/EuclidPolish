"""Every JSON response of the console is standard JSON: a NaN or ±Infinity
anywhere in a payload goes out as ``null`` (``JSON.parse`` rejects the bare
``NaN`` token Python's ``json`` writes, so one non-finite float made a whole
page fail with "response … is not valid JSON")."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
from flask import jsonify

from euclid_polish.web.app import create_app


def _strict(text: str):
    def refuse(token):
        raise ValueError(f"non-standard JSON token {token}")
    return json.loads(text, parse_constant=refuse)


@dataclasses.dataclass
class _Row:
    psnr: float


def _client(payload):
    app = create_app()
    app.config["TESTING"] = True
    app.add_url_rule("/api/test-json", "test_json", lambda: jsonify(payload))
    return app.test_client()


def test_non_finite_floats_go_out_as_null():
    body = _client({
        "band_psnr": [59.2, math.nan, np.float64("nan")],
        "nested": {"hi": math.inf, "lo": -math.inf, "ok": 1.5},
        "row": _Row(math.nan),
        "tuple": (1, math.nan),
    }).get("/api/test-json").get_data(as_text=True)
    assert _strict(body) == {
        "band_psnr": [59.2, None, None],
        "nested": {"hi": None, "lo": None, "ok": 1.5},
        "row": {"psnr": None},
        "tuple": [1, None],
    }


def test_finite_payloads_are_unchanged():
    payload = {"b": [1, 2.5, "x", None, True], "a": {"z": 0.1}}
    body = _client(payload).get("/api/test-json").get_data(as_text=True)
    assert _strict(body) == payload
