"""Unit tests for the TNG image-infographic render script (grid / stack).

The property + histogram logic now lives in euclid_polish.tng.properties (see
test_tng_properties.py); this file covers the script's own bits: galaxy
enumeration, seeded selection, the 5×5 grid + RGB/downsample, the stacked
multi-extension FITS, and the CLI (band validation + --save/--out output).
"""

from __future__ import annotations

import io
import os

import numpy as np
import pytest
from astropy.io import fits

from scripts import fasrc_tng_infographic as mod


def _make_galaxy(tng_dir, gid, *, bands=("VIS", "Y", "J", "H"), size=16,
                 done=True):
    d = os.path.join(tng_dir, gid)
    os.makedirs(d, exist_ok=True)
    for o in (1, 2, 3, 4, 5):
        for b in bands:
            arr = np.abs(np.random.randn(size, size)).astype(">f4")
            hdu = fits.PrimaryHDU(arr)
            hdu.header["BUNIT"] = "MJy/sr"
            hdu.header["CDELT1"] = 100.0
            hdu.header["CDELT2"] = 100.0
            hdu.header["CUNIT1"] = "pc"
            hdu.header["CUNIT2"] = "pc"
            hdu.writeto(
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
    assert a != c
    assert sorted(mod.pick_ids(["1", "2"], 5, 1)) == ["1", "2"]
    assert mod.pick_ids([], 5, 1) == []


def test_pick_ids_negative_seed_is_random():
    ids = [str(i) for i in range(40)]
    picks = {tuple(mod.pick_ids(ids, 5, -1)) for _ in range(8)}
    assert len(picks) > 1


def test_default_output_path():
    p = mod.default_output_path("/data/tng_skirt", "grid")
    assert p == os.path.join("/data/tng_skirt", "_infographics", "grid.png")
    assert mod.default_output_path("/d", "stack").endswith("_infographics/stack.fits")


# ---------------------------------------------------------------------------
# 5×5 grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("band", ["VIS", "Y", "J", "H", "RGB"])
@pytest.mark.parametrize("ds", [1, 2, 4])
def test_render_grid_bands_and_downsample(tmp_path, band, ds):
    tng = str(tmp_path)
    for g in ("111", "222"):
        _make_galaxy(tng, g)
    png = mod.render_grid(tng, band, ds, seed=3)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_grid_no_galaxies_is_placeholder(tmp_path):
    assert mod.render_grid(str(tmp_path), "VIS", 1, 0)[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_grid_explicit_ids(tmp_path):
    """Explicit ids (chosen by the selection mode) are rendered; a missing id
    just gives blank cells, no crash."""
    tng = str(tmp_path)
    for g in ("111", "222", "333"):
        _make_galaxy(tng, g)
    png = mod.render_grid(tng, "VIS", 1, 0, ids=["111", "999"],
                          note="most massive · T=0.30")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_main_grid_with_ids(tmp_path):
    tng = str(tmp_path)
    _make_galaxy(tng, "111")
    rc = mod.main(["--mode", "grid", "--tng-dir", tng, "--band", "VIS",
                   "--ids", "111", "--note", "most massive", "--save"])
    assert rc == 0 and os.path.isfile(mod.default_output_path(tng, "grid"))


def test_render_cell_missing_file_returns_none(tmp_path):
    tng = str(tmp_path)
    _make_galaxy(tng, "111", bands=("VIS",))      # only VIS present
    gdir = os.path.join(tng, "111")
    assert mod.render_cell(gdir, "111", 1, "VIS", 1) is not None
    assert mod.render_cell(gdir, "111", 1, "J", 1) is None     # no J frame
    assert mod.render_cell(gdir, "111", 1, "RGB", 1) is None    # needs H,J,VIS


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
    on_disk = mod.load_skirt_image(
        mod.tng_fits_path(os.path.join(tng, "111"), "111", 1, "VIS"),
        "VIS",
    ).plane("VIS")
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
# CLI
# ---------------------------------------------------------------------------

def test_main_rejects_bad_grid_band(tmp_path):
    assert mod.main(["--mode", "grid", "--tng-dir", str(tmp_path),
                     "--band", "ZZ"]) == 2


def test_main_rejects_rgb_for_stack(tmp_path):
    assert mod.main(["--mode", "stack", "--tng-dir", str(tmp_path),
                     "--band", "RGB"]) == 2


def test_main_stack_no_galaxies(tmp_path):
    assert mod.main(["--mode", "stack", "--tng-dir", str(tmp_path),
                     "--band", "VIS"]) == 3


def test_main_save_writes_standard_artifact(tmp_path):
    """--save (what the SLURM jobs pass) writes the standard artifact path."""
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
