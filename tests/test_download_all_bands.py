"""Cross-band parallelism in ``scripts/download_all_bands.py``.

Exercises ``run_bands`` — the dispatcher that either downloads bands one at a
time (with a TAP re-login between them) or several at once through a thread
pool — with the actual per-band download mocked out, so no network/SSH is
touched.
"""

from __future__ import annotations

import importlib
import threading

dab = importlib.import_module("scripts.download_all_bands")


class _FakeReporter:
    """Records reporter calls; thread-safe set_step (parallel path hits it)."""

    def __init__(self):
        self.stages = []
        self.steps = []
        self.warnings = []
        self._lock = threading.Lock()

    def set_stage(self, msg):
        with self._lock:
            self.stages.append(msg)

    def set_step(self, cur, tot, lbl=""):
        with self._lock:
            self.steps.append((cur, tot, lbl))

    def warn(self, msg):
        with self._lock:
            self.warnings.append(msg)

    def error(self, msg):
        with self._lock:
            self.warnings.append(msg)


def _ok(band):
    return {"downloaded": 3, "valid": 3, "corrupted": 0, "failed": 0, "band": band}


def _fake_one_band_recorder(seen, lock):
    def fake(band_name, *, cat, vis_pixels, workers, arcsec, progress_cb,
             show_progress, retry_failed=False):
        with lock:
            seen.append((band_name, show_progress))
        # Drive the progress callback so the combined-bar aggregation runs.
        progress_cb(workers, workers, "done")
        return _ok(band_name)
    return fake


BANDS = ["VIS", "Y_E", "J_E", "H_E"]


def test_parallel_runs_all_bands(monkeypatch):
    seen, lock = [], threading.Lock()
    monkeypatch.setattr(dab, "_download_one_band",
                        _fake_one_band_recorder(seen, lock))
    rep = _FakeReporter()
    summary = dab.run_bands(
        BANDS, cat=object(), vis_pixels=255, workers=8, arcsec=25.5,
        reporter=rep, band_workers=4, logged_in=True)

    assert set(summary) == set(BANDS)
    assert {b for b, _ in seen} == set(BANDS)
    # Parallel path turns tqdm off (combined bar instead).
    assert all(show is False for _, show in seen)
    # Final step bar pegged to n_bands/n_bands.
    assert rep.steps[-1] == (len(BANDS), len(BANDS), "done")


def test_parallel_relogin_not_called_per_band(monkeypatch):
    # In parallel mode the per-band re-login loop must NOT run (the initial
    # login + the downloader's own session-refresh retry cover it).
    calls = {"n": 0}
    monkeypatch.setattr(dab.auth, "login",
                        lambda **k: calls.__setitem__("n", calls["n"] + 1) or True)
    monkeypatch.setattr(dab, "_download_one_band",
                        _fake_one_band_recorder([], threading.Lock()))
    dab.run_bands(BANDS, cat=object(), vis_pixels=255, workers=8, arcsec=25.5,
                  reporter=_FakeReporter(), band_workers=4, logged_in=True)
    assert calls["n"] == 0


def test_sequential_relogins_between_bands(monkeypatch):
    # band_workers=1 keeps the old path: re-login before every band after the
    # first (i > 0), so a long band can't leave a stale TAP session.
    calls = {"n": 0}
    monkeypatch.setattr(dab.auth, "login",
                        lambda **k: calls.__setitem__("n", calls["n"] + 1) or True)
    seen, lock = [], threading.Lock()
    monkeypatch.setattr(dab, "_download_one_band",
                        _fake_one_band_recorder(seen, lock))
    rep = _FakeReporter()
    dab.run_bands(["VIS", "Y_E", "J_E"], cat=object(), vis_pixels=255,
                  workers=8, arcsec=25.5, reporter=rep, band_workers=1,
                  logged_in=True)
    assert calls["n"] == 2                       # before band 2 and band 3
    # Sequential path keeps tqdm on.
    assert all(show is True for _, show in seen)


def test_sequential_no_relogin_when_not_logged_in(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(dab.auth, "login",
                        lambda **k: calls.__setitem__("n", calls["n"] + 1) or True)
    monkeypatch.setattr(dab, "_download_one_band",
                        _fake_one_band_recorder([], threading.Lock()))
    dab.run_bands(["VIS", "Y_E"], cat=object(), vis_pixels=255, workers=8,
                  arcsec=25.5, reporter=_FakeReporter(), band_workers=1,
                  logged_in=False)
    assert calls["n"] == 0


def test_band_workers_clamped_to_band_count(monkeypatch):
    # Asking for more band-workers than bands must not error; behaves parallel.
    seen, lock = [], threading.Lock()
    monkeypatch.setattr(dab, "_download_one_band",
                        _fake_one_band_recorder(seen, lock))
    summary = dab.run_bands(["VIS", "Y_E"], cat=object(), vis_pixels=255,
                            workers=4, arcsec=25.5, reporter=_FakeReporter(),
                            band_workers=99, logged_in=True)
    assert set(summary) == {"VIS", "Y_E"}
    assert all(show is False for _, show in seen)


def test_one_band_failing_does_not_sink_the_rest(monkeypatch):
    def fake(band_name, **k):
        if band_name == "J_E":
            raise RuntimeError("server 500")
        return _ok(band_name)
    monkeypatch.setattr(dab, "_download_one_band", fake)
    rep = _FakeReporter()
    summary = dab.run_bands(BANDS, cat=object(), vis_pixels=255, workers=8,
                            arcsec=25.5, reporter=rep, band_workers=4,
                            logged_in=True)
    assert set(summary) == set(BANDS)
    assert summary["J_E"] == {"downloaded": 0, "valid": 0,
                              "corrupted": 0, "failed": 0}
    assert summary["VIS"]["downloaded"] == 3
    assert any("J_E" in w for w in rep.warnings)


def test_retry_failed_forwarded_to_each_band(monkeypatch):
    seen = []

    def fake(band_name, *, retry_failed, **k):
        seen.append((band_name, retry_failed))
        return _ok(band_name)
    monkeypatch.setattr(dab, "_download_one_band", fake)

    # Parallel mode.
    dab.run_bands(BANDS, cat=object(), vis_pixels=255, workers=8, arcsec=25.5,
                  reporter=_FakeReporter(), band_workers=4, logged_in=True,
                  retry_failed=True)
    assert all(rf is True for _, rf in seen)

    # Sequential mode (default retry_failed=False when omitted).
    seen.clear()
    dab.run_bands(["VIS"], cat=object(), vis_pixels=255, workers=8, arcsec=25.5,
                  reporter=_FakeReporter(), band_workers=1, logged_in=True)
    assert seen == [("VIS", False)]


def test_warns_on_corrupted_or_failed(monkeypatch):
    def fake(band_name, **k):
        if band_name == "H_E":
            return {"downloaded": 1, "valid": 0, "corrupted": 2, "failed": 1}
        return _ok(band_name)
    monkeypatch.setattr(dab, "_download_one_band", fake)
    rep = _FakeReporter()
    dab.run_bands(BANDS, cat=object(), vis_pixels=255, workers=8, arcsec=25.5,
                  reporter=rep, band_workers=1, logged_in=True)
    assert any("H_E" in w and "corrupted=2" in w for w in rep.warnings)
