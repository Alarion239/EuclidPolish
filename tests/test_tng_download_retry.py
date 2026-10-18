"""The atlas downloader waits out overloaded servers (503) instead of
failing the galaxy, and still stops at once on a rejected key."""
from __future__ import annotations

import pytest
import requests

from scripts import download_tng_skirt as dl


class _Reply:
    def __init__(self, status: int):
        self.status_code = status
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def close(self):
        self.closed = True


def _serve(monkeypatch, statuses):
    calls = []

    def fake_get(url, **kwargs):
        status = statuses[len(calls)]
        calls.append(url)
        if status == "drop":
            raise requests.ConnectionError("reset")
        return _Reply(status)

    monkeypatch.setattr(dl.requests, "get", fake_get)
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    return calls


def test_503s_and_dropped_connections_are_retried(monkeypatch):
    calls = _serve(monkeypatch, [503, "drop", 502, 200])
    assert dl._request("u", "k").status_code == 200
    assert len(calls) == 4


def test_it_gives_up_after_the_last_delay(monkeypatch):
    calls = _serve(monkeypatch, [503] * 3)
    with pytest.raises(requests.HTTPError):
        dl._request("u", "k", delays=(1.0, 1.0))
    assert len(calls) == 3


def test_a_rejected_key_stops_at_once(monkeypatch):
    calls = _serve(monkeypatch, [403, 200])
    with pytest.raises(SystemExit):
        dl._request("u", "k")
    assert len(calls) == 1


def test_a_missing_file_is_not_retried(monkeypatch):
    calls = _serve(monkeypatch, [404, 200])
    with pytest.raises(requests.HTTPError):
        dl._request("u", "k")
    assert len(calls) == 1
