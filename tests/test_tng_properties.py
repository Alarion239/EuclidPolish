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
    sub = {"sfr": 1.5, "mass_stars": 2.0, "halfmassrad_stars": 3.0,
           "mass_log_msun": 12.0}
    p = P.parse_subhalo(sub)
    assert p["sfr"] == pytest.approx(1.5)
    assert p["mass_stars"] == pytest.approx(2.0 * 1e10 / P.H_LITTLE)
    assert p["reff"] == pytest.approx(3.0 / P.H_LITTLE)
    # total mass comes straight from mass_log_msun (already log10 Msun).
    assert p["m_halo"] == pytest.approx(10.0 ** 12.0)


def test_parse_subhalo_mass_fallback_when_no_log():
    sub = {"sfr": 0.0, "mass_stars": 1.0, "halfmassrad_stars": 1.0, "mass": 5.0}
    p = P.parse_subhalo(sub)
    assert p["m_halo"] == pytest.approx(5.0 * 1e10 / P.H_LITTLE)


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


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def test_plot_histograms_returns_png():
    props = {g: {"sfr": 1.0 + i, "mass_stars": 10 ** (10 + i),
                 "m_halo": 10 ** (12 + i), "reff": 2.0 + i}
             for i, g in enumerate(("1", "2", "3"))}
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
