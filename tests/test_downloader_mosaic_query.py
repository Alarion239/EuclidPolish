"""The mosaic-tile query helper: None-guard + re-login retry.

Reproduces the bug where ``Euclid.launch_job_async`` returned ``None`` (an
expired TAP session after a long band) and ``.get_results()`` then raised
``AttributeError: 'NoneType'`` — failing every NISP band of a cutout download.
The session refresh is now an injected ``relogin`` callback (the owning
``EuclidCatalog`` passes ``self.relogin``).
"""

from __future__ import annotations

from euclid_polish.catalog import downloader as D


class _FakeJob:
    def __init__(self, results):
        self._r = results

    def get_results(self):
        return self._r


def _relogin_counter(counter):
    """A relogin callback that bumps ``counter['relogin']`` each call."""
    return lambda: counter.__setitem__("relogin", counter["relogin"] + 1) or True


def test_query_returns_results(monkeypatch):
    c = {"relogin": 0}
    monkeypatch.setattr(D.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(D.Euclid, "launch_job_async", lambda q: _FakeJob(["t1", "t2"]))
    res, err = D.query_mosaic_tiles("SELECT ...", relogin=_relogin_counter(c))
    assert res == ["t1", "t2"] and err == ""
    assert c["relogin"] == 0                       # no retry needed


def test_none_then_success_retries_with_relogin(monkeypatch):
    c = {"relogin": 0, "q": 0}
    monkeypatch.setattr(D.time, "sleep", lambda *_a, **_k: None)

    def fake_launch(q):
        c["q"] += 1
        return None if c["q"] == 1 else _FakeJob(["ok"])   # expired, then fresh
    monkeypatch.setattr(D.Euclid, "launch_job_async", fake_launch)
    res, err = D.query_mosaic_tiles("q", retries=2, relogin=_relogin_counter(c))
    assert res == ["ok"] and c["q"] == 2 and c["relogin"] == 1


def test_all_none_returns_none_not_attributeerror(monkeypatch):
    c = {"relogin": 0}
    monkeypatch.setattr(D.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(D.Euclid, "launch_job_async", lambda q: None)
    res, err = D.query_mosaic_tiles("q", retries=2, relogin=_relogin_counter(c))
    assert res is None and "None" in err          # graceful, not a crash
    assert c["relogin"] == 1                        # re-login attempted before giving up


def test_exception_is_caught_and_reported(monkeypatch):
    c = {"relogin": 0}
    monkeypatch.setattr(D.time, "sleep", lambda *_a, **_k: None)

    def boom(q):
        raise RuntimeError("server 500")
    monkeypatch.setattr(D.Euclid, "launch_job_async", boom)
    res, err = D.query_mosaic_tiles("q", retries=2, relogin=_relogin_counter(c))
    assert res is None and "RuntimeError" in err and "server 500" in err


def test_relogin_optional_when_omitted(monkeypatch):
    # No relogin callback → still retries, just without a session refresh.
    monkeypatch.setattr(D.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(D.Euclid, "launch_job_async", lambda q: None)
    res, err = D.query_mosaic_tiles("q", retries=2)
    assert res is None and "None" in err
