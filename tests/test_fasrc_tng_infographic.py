"""Unit tests for the TNG infographic render script.

Network-free: the TNG-API getter is monkeypatched; FITS are tiny synthetic
frames. Covers galaxy enumeration, seeded selection, API field parsing +
unit conversion, the property cache round-trip, the three render outputs
(PNG / multi-extension FITS), and band validation.
"""

from __future__ import annotations

import io
import os

import numpy as np
import pytest
from astropy.io import fits

from scripts import fasrc_tng_infographic as mod


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_galaxy(tng_dir, gid, *, bands=("VIS", "Y", "J", "H"), size=16,
                 done=True):
    d = os.path.join(tng_dir, gid)
    os.makedirs(d, exist_ok=True)
    for o in (1, 2, 3, 4, 5):
        for b in bands:
            arr = np.abs(np.random.randn(size, size)).astype(">f4")
            fits.PrimaryHDU(arr).writeto(
                os.path.join(d, f"TNG{gid}_O{o}_Euclid_{b}.fits"),
                overwrite=True)
    if done:
        open(os.path.join(d, mod.Config.Tng.DONE_MARKER), "w").close()
    return d


# ---------------------------------------------------------------------------
# enumeration + selection
# ---------------------------------------------------------------------------

def test_list_downloaded_ids_requires_done_marker(tmp_path):
    tng = str(tmp_path)
    _make_galaxy(tng, "111")
    _make_galaxy(tng, "222")
    _make_galaxy(tng, "333", done=False)        # no .done → excluded
    assert mod.list_downloaded_ids(tng) == ["111", "222"]   # numeric sort


def test_list_downloaded_ids_missing_dir(tmp_path):
    assert mod.list_downloaded_ids(str(tmp_path / "nope")) == []


def test_pick_ids_deterministic_and_bounded():
    ids = [str(i) for i in range(50)]
    a = mod.pick_ids(ids, 5, seed=42)
    b = mod.pick_ids(ids, 5, seed=42)
    c = mod.pick_ids(ids, 5, seed=43)
    assert a == b and len(a) == 5 and len(set(a)) == 5
    assert a != c                                # different seed → different pick
    # Fewer than k → returns all (shuffled), no error.
    assert sorted(mod.pick_ids(["1", "2"], 5, 1)) == ["1", "2"]
    assert mod.pick_ids([], 5, 1) == []


# ---------------------------------------------------------------------------
# API parsing + unit conversion
# ---------------------------------------------------------------------------

def test_parse_subhalo_raw_fields():
    sub = {"SubhaloSFR": 1.5, "SubhaloMassType_4": 2.0,
           "SubhaloHalfmassRadType_4": 3.0, "SubhaloGrNr": 7}
    p = mod.parse_subhalo(sub)
    assert p["sfr"] == pytest.approx(1.5)
    assert p["mass_stars"] == pytest.approx(2.0 * 1e10 / mod.H_LITTLE)
    assert p["reff"] == pytest.approx(3.0 / mod.H_LITTLE)


def test_parse_subhalo_list_fallback():
    sub = {"SubhaloSFR": 1.0,
           "SubhaloMassType": [0, 0, 0, 0, 5.0, 0],
           "SubhaloHalfmassRadType": [0, 0, 0, 0, 4.0, 0]}
    p = mod.parse_subhalo(sub)
    assert p["mass_stars"] == pytest.approx(5.0 * 1e10 / mod.H_LITTLE)
    assert p["reff"] == pytest.approx(4.0 / mod.H_LITTLE)


def test_parse_subhalo_cleaned_fields():
    sub = {"sfr": 0.3, "mass_stars": 1.0, "halfmassrad_stars": 2.0, "grnr": 1}
    p = mod.parse_subhalo(sub)
    assert p["sfr"] == pytest.approx(0.3)
    assert np.isfinite(p["mass_stars"]) and np.isfinite(p["reff"])


def test_fetch_properties_fills_halo_mass(monkeypatch):
    def fake_get(url, key, timeout=10):
        if "/subhalos/" in url:
            return {"SubhaloSFR": 2.0, "SubhaloMassType_4": 1.0,
                    "SubhaloHalfmassRadType_4": 1.0, "SubhaloGrNr": 99}
        if "/halos/99/" in url:
            return {"Group_M_Crit200": 50.0}
        raise AssertionError(f"unexpected url {url}")
    monkeypatch.setattr(mod, "_get_json", fake_get)
    p = mod.fetch_properties("123", "key")
    assert p["sfr"] == pytest.approx(2.0)
    assert p["m_halo"] == pytest.approx(50.0 * 1e10 / mod.H_LITTLE)


def test_fetch_properties_network_failure_is_nan(monkeypatch):
    def boom(url, key, timeout=10):
        raise RuntimeError("network down")
    monkeypatch.setattr(mod, "_get_json", boom)
    p = mod.fetch_properties("123", "key")
    assert all(np.isnan(v) for v in p.values())


# ---------------------------------------------------------------------------
# property cache round-trip + gather
# ---------------------------------------------------------------------------

def test_cache_round_trip(tmp_path):
    path = str(tmp_path / "props.csv")
    rows = {"111": {"sfr": 1.0, "mass_stars": 2.0, "m_halo": 3.0, "reff": 4.0},
            "22":  {"sfr": float("nan"), "mass_stars": 5.0,
                    "m_halo": 6.0, "reff": 7.0}}
    mod._write_cache(path, rows)
    back = mod._read_cache(path)
    assert back["111"]["sfr"] == pytest.approx(1.0)
    assert np.isnan(back["22"]["sfr"])
    assert back["22"]["mass_stars"] == pytest.approx(5.0)


def test_gather_properties_queries_only_missing(tmp_path, monkeypatch):
    tng = str(tmp_path)
    # Pre-seed cache with 111 only; 222 is missing → one query.
    mod._write_cache(os.path.join(tng, mod.PROPERTIES_CSV),
                     {"111": {"sfr": 1.0, "mass_stars": 2.0,
                              "m_halo": 3.0, "reff": 4.0}})
    calls = []

    def fake_fetch(gid, key, timeout=10):
        calls.append(gid)
        return {"sfr": 9.0, "mass_stars": 9.0, "m_halo": 9.0, "reff": 9.0}
    monkeypatch.setattr(mod, "fetch_properties", fake_fetch)
    out = mod.gather_properties(tng, ["111", "222"], "key", max_new=10)
    assert calls == ["222"]                       # 111 served from cache
    assert set(out) == {"111", "222"}
    # 222 now persisted to the cache CSV.
    assert "222" in mod._read_cache(os.path.join(tng, mod.PROPERTIES_CSV))


def test_gather_properties_max_new_zero_skips_network(tmp_path, monkeypatch):
    tng = str(tmp_path)
    monkeypatch.setattr(mod, "fetch_properties",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("should not query")))
    out = mod.gather_properties(tng, ["111"], "key", max_new=0)
    assert out == {}                              # nothing cached, none queried


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def test_render_histograms_from_cache(tmp_path):
    tng = str(tmp_path)
    for g in ("111", "222", "333"):
        _make_galaxy(tng, g)
    mod._write_cache(
        os.path.join(tng, mod.PROPERTIES_CSV),
        {g: {"sfr": 1.0 + i, "mass_stars": 10 ** (10 + i),
             "m_halo": 10 ** (12 + i), "reff": 2.0 + i}
         for i, g in enumerate(("111", "222", "333"))})
    png = mod.render_histograms(tng, api_key="", max_new=0)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_histograms_no_galaxies_is_placeholder(tmp_path):
    png = mod.render_histograms(str(tmp_path), api_key="", max_new=0)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"        # placeholder, not a crash


@pytest.mark.parametrize("band", ["VIS", "Y", "J", "H", "RGB"])
@pytest.mark.parametrize("ds", [1, 2, 4])
def test_render_grid_bands_and_downsample(tmp_path, band, ds):
    tng = str(tmp_path)
    for g in ("111", "222"):
        _make_galaxy(tng, g)
    png = mod.render_grid(tng, band, ds, seed=3)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_cell_missing_file_returns_none(tmp_path):
    tng = str(tmp_path)
    _make_galaxy(tng, "111", bands=("VIS",))      # only VIS present
    gdir = os.path.join(tng, "111")
    assert mod.render_cell(gdir, "111", 1, "VIS", 1) is not None
    assert mod.render_cell(gdir, "111", 1, "J", 1) is None     # no J frame
    # RGB needs all of H,J,VIS → missing H/J → None.
    assert mod.render_cell(gdir, "111", 1, "RGB", 1) is None


def test_grayscale_norm_in_unit_range():
    arr = np.abs(np.random.randn(16, 16)).astype(np.float32)
    n = mod._grayscale_norm(arr)
    assert n.shape == (16, 16)
    assert n.min() >= 0.0 and n.max() <= 1.0


# ---------------------------------------------------------------------------
# stacked FITS
# ---------------------------------------------------------------------------

def test_build_stack_hdul_preserves_frames(tmp_path):
    tng = str(tmp_path)
    _make_galaxy(tng, "111", size=16)
    hdul = mod.build_stack_hdul(tng, "111", "VIS")
    assert len(hdul) == 6                          # primary + O1..O5
    assert [h.name for h in hdul[1:]] == ["O1", "O2", "O3", "O4", "O5"]
    assert all(hdul[i].data.shape == (16, 16) for i in range(1, 6))
    assert hdul[0].header["TNGID"] == "111"
    # Frame O1 equals the on-disk VIS frame.
    on_disk = mod.load_tng_frame(
        mod.tng_fits_path(os.path.join(tng, "111"), "111", 1, "VIS"))
    np.testing.assert_allclose(np.asarray(hdul[1].data), on_disk, rtol=0, atol=0)


def test_build_stack_bytes_is_valid_fits(tmp_path):
    tng = str(tmp_path)
    _make_galaxy(tng, "111")
    data = mod.build_stack_bytes(tng, "111", "H")
    assert data[:6] == b"SIMPLE"
    with fits.open(io.BytesIO(data)) as hdul:
        assert len(hdul) == 6


def test_build_stack_missing_galaxy_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mod.build_stack_hdul(str(tmp_path), "999", "VIS")


# ---------------------------------------------------------------------------
# CLI band validation
# ---------------------------------------------------------------------------

def test_main_rejects_bad_grid_band(tmp_path, capsys):
    rc = mod.main(["--mode", "grid", "--tng-dir", str(tmp_path), "--band", "ZZ"])
    assert rc == 2


def test_main_rejects_rgb_for_stack(tmp_path):
    rc = mod.main(["--mode", "stack", "--tng-dir", str(tmp_path), "--band", "RGB"])
    assert rc == 2


def test_main_stack_no_galaxies(tmp_path):
    rc = mod.main(["--mode", "stack", "--tng-dir", str(tmp_path), "--band", "VIS"])
    assert rc == 3


# ---------------------------------------------------------------------------
# job artifact output (--save / --out)
# ---------------------------------------------------------------------------

def test_default_output_path():
    p = mod.default_output_path("/data/tng_skirt", "grid")
    assert p == os.path.join("/data/tng_skirt", "_infographics", "grid.png")
    assert mod.default_output_path("/d", "stack").endswith("_infographics/stack.fits")


def test_main_save_writes_standard_artifact(tmp_path):
    """--save (what the SLURM jobs pass) writes the standard artifact path
    instead of streaming to stdout."""
    tng = str(tmp_path)
    for g in ("111", "222"):
        _make_galaxy(tng, g)
    rc = mod.main(["--mode", "grid", "--tng-dir", tng, "--band", "VIS",
                   "--seed", "1", "--save"])
    assert rc == 0
    out = mod.default_output_path(tng, "grid")
    assert os.path.isfile(out)
    with open(out, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


def test_main_out_writes_explicit_path_and_stack(tmp_path):
    tng = str(tmp_path / "tng")
    os.makedirs(tng)
    _make_galaxy(tng, "111")
    dest = str(tmp_path / "sub" / "mystack.fits")     # parent created on demand
    rc = mod.main(["--mode", "stack", "--tng-dir", tng, "--band", "VIS",
                   "--id", "111", "--out", dest])
    assert rc == 0 and os.path.isfile(dest)
    with fits.open(dest) as hdul:
        assert len(hdul) == 6


def test_pick_ids_negative_seed_is_random(monkeypatch):
    """seed < 0 → fresh random pick (so a re-submitted grid/stack re-rolls)."""
    ids = [str(i) for i in range(40)]
    picks = {tuple(mod.pick_ids(ids, 5, -1)) for _ in range(8)}
    assert len(picks) > 1                              # not all identical
