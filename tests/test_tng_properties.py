"""Unit tests for euclid_polish.tng.properties.

Network-free: the TNG-API getters are monkeypatched. Covers API field parsing
+ unit conversion, the bulk group-catalog path (5 field arrays → per-galaxy
props), the property cache round-trip, the bulk→cache→fallback flow, and the
histogram PNG rendering.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from euclid_polish.tng import properties as P


def _raise(*a, **k):
    raise RuntimeError("no network in tests")


def _bulk_arrays():
    return {
        "SubhaloSFR": np.array([0.0, 1.5, 2.5, 9.9], dtype="f4"),
        "SubhaloMassType": np.tile(
            np.array([0, 0, 0, 0, 2.0, 0], dtype="f4"), (4, 1)),
        "SubhaloHalfmassRadType": np.tile(
            np.array([0, 0, 0, 0, 3.0, 0], dtype="f4"), (4, 1)),
        "SubhaloGrNr": np.array([0, 0, 1, 1], dtype="i8"),
        "Group_M_Crit200": np.array([10.0, 20.0], dtype="f4"),
    }


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
# per-galaxy info.json parsing (fallback path)
# ---------------------------------------------------------------------------

def test_parse_subhalo_raw_fields():
    sub = {"SubhaloSFR": 1.5, "SubhaloMassType_4": 2.0,
           "SubhaloHalfmassRadType_4": 3.0, "SubhaloGrNr": 7}
    p = P.parse_subhalo(sub)
    assert p["sfr"] == pytest.approx(1.5)
    assert p["mass_stars"] == pytest.approx(2.0 * 1e10 / P.H_LITTLE)
    assert p["reff"] == pytest.approx(3.0 / P.H_LITTLE)


def test_parse_subhalo_list_fallback():
    sub = {"SubhaloSFR": 1.0,
           "SubhaloMassType": [0, 0, 0, 0, 5.0, 0],
           "SubhaloHalfmassRadType": [0, 0, 0, 0, 4.0, 0]}
    p = P.parse_subhalo(sub)
    assert p["mass_stars"] == pytest.approx(5.0 * 1e10 / P.H_LITTLE)
    assert p["reff"] == pytest.approx(4.0 / P.H_LITTLE)


def test_fetch_properties_fills_halo_mass(monkeypatch):
    def fake_get(url, key, timeout=10):
        if "/subhalos/" in url:
            return {"SubhaloSFR": 2.0, "SubhaloMassType_4": 1.0,
                    "SubhaloHalfmassRadType_4": 1.0, "SubhaloGrNr": 99}
        if "/halos/99/" in url:
            return {"Group_M_Crit200": 50.0}
        raise AssertionError(f"unexpected url {url}")
    monkeypatch.setattr(P, "_get_json", fake_get)
    p = P.fetch_properties("123", "key")
    assert p["sfr"] == pytest.approx(2.0)
    assert p["m_halo"] == pytest.approx(50.0 * 1e10 / P.H_LITTLE)


def test_fetch_properties_network_failure_is_nan(monkeypatch):
    monkeypatch.setattr(P, "_get_json", _raise)
    p = P.fetch_properties("123", "key")
    assert all(np.isnan(v) for v in p.values())


# ---------------------------------------------------------------------------
# bulk group-catalog fetch
# ---------------------------------------------------------------------------

def test_row_stars():
    assert P._row_stars([0, 0, 0, 0, 5.0, 0]) == 5.0
    assert P._row_stars(7.0) == 7.0


def test_properties_from_arrays_indexing_and_units():
    props = P.properties_from_arrays(_bulk_arrays(), ["1", "2"])
    assert props["1"]["sfr"] == pytest.approx(1.5)
    assert props["1"]["mass_stars"] == pytest.approx(2.0 * 1e10 / P.H_LITTLE)
    assert props["1"]["reff"] == pytest.approx(3.0 / P.H_LITTLE)
    # subhalo 1 → group 0 → M_Crit200 = 10 ; subhalo 2 → group 1 → 20
    assert props["1"]["m_halo"] == pytest.approx(10.0 * 1e10 / P.H_LITTLE)
    assert props["2"]["m_halo"] == pytest.approx(20.0 * 1e10 / P.H_LITTLE)


def test_properties_from_arrays_out_of_range_is_nan():
    props = P.properties_from_arrays(_bulk_arrays(), ["999"])
    assert all(np.isnan(v) for v in props["999"].values())


@pytest.mark.parametrize("nested", [False, True])
def test_read_field_array_roundtrip(tmp_path, nested):
    import h5py
    p = str(tmp_path / "f.hdf5")
    with h5py.File(p, "w") as f:
        target = f.create_group("Subhalo") if nested else f
        target.create_dataset("SubhaloSFR", data=np.arange(5, dtype="f4"))
    arr = P._read_field_array(p, "SubhaloSFR")
    assert list(arr) == [0.0, 1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# cache + gather
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


def test_gather_uses_bulk_then_cache(tmp_path, monkeypatch):
    work = str(tmp_path)
    calls = {"bulk": 0}

    def fake_bulk(key, cache_dir, *, reporter=None):
        calls["bulk"] += 1
        return _bulk_arrays()
    monkeypatch.setattr(P, "bulk_fetch_fields", fake_bulk)
    monkeypatch.setattr(P, "fetch_properties",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("bulk path must not hit per-galaxy")))
    out = P.gather_properties(work, ["1", "2", "3"], "key")
    assert calls["bulk"] == 1 and set(out) == {"1", "2", "3"}
    assert out["2"]["sfr"] == pytest.approx(2.5)
    # Second call: cached → bulk not called again.
    monkeypatch.setattr(P, "bulk_fetch_fields", _raise)
    out2 = P.gather_properties(work, ["1", "2", "3"], "key")
    assert set(out2) == {"1", "2", "3"}


def test_gather_only_queries_missing_on_fallback(tmp_path, monkeypatch):
    work = str(tmp_path)
    P._write_cache(os.path.join(work, P.PROPERTIES_CSV),
                   {"111": {"sfr": 1.0, "mass_stars": 2.0,
                            "m_halo": 3.0, "reff": 4.0}})
    monkeypatch.setattr(P, "bulk_fetch_fields", _raise)   # force fallback
    calls = []
    monkeypatch.setattr(P, "fetch_properties",
                        lambda gid, key, timeout=10: calls.append(gid) or
                        {"sfr": 9.0, "mass_stars": 9.0, "m_halo": 9.0, "reff": 9.0})
    out = P.gather_properties(work, ["111", "222"], "key", max_new=10)
    assert calls == ["222"] and set(out) == {"111", "222"}


def test_gather_max_new_zero_skips_network(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "bulk_fetch_fields", _raise)
    monkeypatch.setattr(P, "fetch_properties",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("should not query")))
    out = P.gather_properties(str(tmp_path), ["111"], "key", max_new=0)
    assert out == {}


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


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
    # key="" → no network; reads the seeded cache.
    png = P.render_histograms_for_ids(work, ["1", "2", "3"], "", max_new=0)
    assert png[:8] == _PNG_MAGIC


def test_render_histograms_for_ids_no_ids_is_placeholder(tmp_path):
    assert P.render_histograms_for_ids(str(tmp_path), [], "")[:8] == _PNG_MAGIC
