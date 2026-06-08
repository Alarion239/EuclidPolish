"""Unit tests for euclid_polish.tng.properties.

Network-free: the TNG-API getter is monkeypatched. Covers the cleaned
``?format=json`` field parsing + unit conversion, the concurrent per-galaxy
fetch (only-missing + cache), and the histogram PNG rendering.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from euclid_polish.tng import properties as P

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _raise(*a, **k):
    raise RuntimeError("no network in tests")


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

def test_load_api_key_env_then_file(tmp_path, monkeypatch):
    monkeypatch.setenv("TNG_API_KEY", "from-env")
    assert P.load_api_key() == "from-env"
    monkeypatch.delenv("TNG_API_KEY", raising=False)
    kf = tmp_path / "key"
    kf.write_text("from-file\n")
    assert P.load_api_key(str(kf)) == "from-file"
    assert P.load_api_key(str(tmp_path / "nope")) == ""


# ---------------------------------------------------------------------------
# parsing + unit conversion (?format=json cleaned fields)
# ---------------------------------------------------------------------------

def test_parse_subhalo_format_json():
    # Documented fields only (mass = SubhaloMass, code units → ×1e10/h M⊙).
    sub = {"sfr": 1.5, "mass_stars": 2.0, "halfmassrad_stars": 3.0, "mass": 5.0}
    p = P.parse_subhalo(sub)
    assert p["sfr"] == pytest.approx(1.5)
    assert p["mass_stars"] == pytest.approx(2.0 * 1e10 / P.H_LITTLE)
    assert p["reff"] == pytest.approx(3.0 / P.H_LITTLE)
    assert p["m_halo"] == pytest.approx(5.0 * 1e10 / P.H_LITTLE)


def test_parse_subhalo_prefers_documented_mass_over_log():
    # Both present (as the live API returns): the documented raw ``mass`` wins.
    sub = {"mass": 5.0, "mass_log_msun": 99.0}     # 99 is absurd → must be ignored
    assert P.parse_subhalo(sub)["m_halo"] == pytest.approx(5.0 * 1e10 / P.H_LITTLE)


def test_parse_subhalo_mass_log_fallback():
    # Only the undocumented convenience field present → still works.
    sub = {"sfr": 0.0, "mass_stars": 1.0, "halfmassrad_stars": 1.0,
           "mass_log_msun": 12.0}
    assert P.parse_subhalo(sub)["m_halo"] == pytest.approx(10.0 ** 12.0)


def test_parse_subhalo_missing_fields_are_nan():
    p = P.parse_subhalo({})
    assert all(np.isnan(v) for v in p.values())


def test_fetch_properties_single_fast_call(monkeypatch):
    calls = []

    def fake_get(url, key, timeout=30):
        calls.append(url)
        return {"sfr": 2.0, "mass_stars": 1.0, "halfmassrad_stars": 1.0,
                "mass_log_msun": 13.0}
    monkeypatch.setattr(P, "_get_json", fake_get)
    p = P.fetch_properties("123", "key")
    # ONE request, to the cleaned (fast) endpoint — not info.json, no halo call.
    assert len(calls) == 1
    assert "?format=json" in calls[0] and "info.json" not in calls[0]
    assert p["sfr"] == pytest.approx(2.0)
    assert p["m_halo"] == pytest.approx(10.0 ** 13.0)


def test_fetch_properties_network_failure_is_nan(monkeypatch):
    monkeypatch.setattr(P, "_get_json", _raise)
    assert all(np.isnan(v) for v in P.fetch_properties("123", "key").values())


# ---------------------------------------------------------------------------
# cache + concurrent gather
# ---------------------------------------------------------------------------

def test_cache_round_trip(tmp_path):
    path = str(tmp_path / "props.csv")
    rows = {"111": {"sfr": 1.0, "mass_stars": 2.0, "m_halo": 3.0, "reff": 4.0},
            "22":  {"sfr": float("nan"), "mass_stars": 5.0,
                    "m_halo": 6.0, "reff": 7.0}}
    P._write_cache(path, rows)
    back = P._read_cache(path)
    assert back["111"]["sfr"] == pytest.approx(1.0)
    assert np.isnan(back["22"]["sfr"])
    assert back["22"]["mass_stars"] == pytest.approx(5.0)


def test_gather_queries_only_missing_concurrently(tmp_path, monkeypatch):
    work = str(tmp_path)
    P._write_cache(os.path.join(work, P.PROPERTIES_CSV),
                   {"111": {"sfr": 1.0, "mass_stars": 2.0,
                            "m_halo": 3.0, "reff": 4.0}})
    calls = []

    def fake_fetch(gid, key, timeout=30):
        calls.append(gid)
        return {"sfr": 9.0, "mass_stars": 9.0, "m_halo": 9.0, "reff": 9.0}
    monkeypatch.setattr(P, "fetch_properties", fake_fetch)
    out = P.gather_properties(work, ["111", "222", "333"], "key")
    # 111 served from cache; only the missing two queried (concurrent → set).
    assert set(calls) == {"222", "333"}
    assert set(out) == {"111", "222", "333"}
    assert "222" in P._read_cache(os.path.join(work, P.PROPERTIES_CSV))
    # Second call: all cached → no fetch.
    monkeypatch.setattr(P, "fetch_properties", _raise)
    out2 = P.gather_properties(work, ["111", "222", "333"], "key")
    assert set(out2) == {"111", "222", "333"}


def test_gather_no_key_skips_network(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "fetch_properties", _raise)
    assert P.gather_properties(str(tmp_path), ["1"], "") == {}


def test_gather_retries_all_nan_rows(tmp_path, monkeypatch):
    """An all-NaN cache row (a failed fetch) is re-queried; resolved/quenched
    rows are served from cache untouched."""
    work = str(tmp_path)
    nan = float("nan")
    P._write_cache(os.path.join(work, P.PROPERTIES_CSV), {
        "ok":       {"sfr": 1.0, "mass_stars": 2.0, "m_halo": 3.0, "reff": 4.0},
        "quenched": {"sfr": 0.0, "mass_stars": 2.0, "m_halo": 3.0, "reff": 4.0},
        "failed":   {"sfr": nan, "mass_stars": nan, "m_halo": nan, "reff": nan},
    })
    calls = []

    def fake_fetch(gid, key, timeout=30):
        calls.append(gid)
        return {"sfr": 5.0, "mass_stars": 5.0, "m_halo": 5.0, "reff": 5.0}
    monkeypatch.setattr(P, "fetch_properties", fake_fetch)
    out = P.gather_properties(work, ["ok", "quenched", "failed"], "key")
    # only the all-NaN row is retried — quenched (SFR=0) is real data, kept.
    assert calls == ["failed"]
    assert out["failed"]["sfr"] == pytest.approx(5.0)
    assert out["quenched"]["sfr"] == pytest.approx(0.0)
    # now persisted as resolved → a second gather does not re-fetch it.
    monkeypatch.setattr(P, "fetch_properties", _raise)
    out2 = P.gather_properties(work, ["ok", "quenched", "failed"], "key")
    assert out2["failed"]["sfr"] == pytest.approx(5.0)


def test_is_failed_row():
    nan = float("nan")
    assert P._is_failed_row({"sfr": nan, "mass_stars": nan,
                             "m_halo": nan, "reff": nan})
    # quenched (SFR=0 but masses present) is NOT a failure
    assert not P._is_failed_row({"sfr": 0.0, "mass_stars": 2.0,
                                 "m_halo": 3.0, "reff": 4.0})
    assert not P._is_failed_row({"sfr": 1.0, "mass_stars": 2.0,
                                 "m_halo": 3.0, "reff": 4.0})


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def test_plot_histograms_returns_png():
    props = {g: {"sfr": 1.0 + i, "mass_stars": 10 ** (10 + i),
                 "m_halo": 10 ** (12 + i), "reff": 2.0 + i}
             for i, g in enumerate(("1", "2", "3"))}
    assert P.plot_histograms(props)[:8] == _PNG_MAGIC


def test_sfr_population_splits_quenched_and_missing():
    nan = float("nan")
    props = {
        "a": {"sfr": 1.0},     # log10 = 0
        "b": {"sfr": 10.0},    # log10 = 1
        "c": {"sfr": 0.0},     # quenched
        "d": {"sfr": 0.0},     # quenched
        "e": {"sfr": nan},     # missing
    }
    logv, n_quenched, n_missing = P.sfr_population(props)
    assert logv.size == 2
    assert np.allclose(np.sort(logv), [0.0, 1.0])
    assert n_quenched == 2
    assert n_missing == 1


def test_plot_histograms_with_quenched_and_missing():
    nan = float("nan")
    props = {
        "1": {"sfr": 1.0, "mass_stars": 1e10, "m_halo": 1e12, "reff": 2.0},
        "2": {"sfr": 0.0, "mass_stars": 2e10, "m_halo": 2e12, "reff": 3.0},
        "3": {"sfr": nan, "mass_stars": nan, "m_halo": nan, "reff": nan},
    }
    assert P.plot_histograms(props)[:8] == _PNG_MAGIC


def test_render_histograms_for_ids_from_cache(tmp_path):
    work = str(tmp_path)
    P._write_cache(
        os.path.join(work, P.PROPERTIES_CSV),
        {g: {"sfr": 1.0 + i, "mass_stars": 10 ** (10 + i),
             "m_halo": 10 ** (12 + i), "reff": 2.0 + i}
         for i, g in enumerate(("1", "2", "3"))})
    png = P.render_histograms_for_ids(work, ["1", "2", "3"], "")  # key="" → cache
    assert png[:8] == _PNG_MAGIC


def test_render_histograms_for_ids_no_ids_is_placeholder(tmp_path):
    assert P.render_histograms_for_ids(str(tmp_path), [], "")[:8] == _PNG_MAGIC
