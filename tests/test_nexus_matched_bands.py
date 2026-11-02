"""The NIRCam filters matched to the Euclid bands on the NEXUS field
(VIS–F090W, Y–F115W, J–F150W, H–F200W): the cache job that resamples the
early-release mosaics onto the field's F200W tiles
(``jwst_euclid.cache_nexus_matched_bands``), the real tiles and viewer cube
that carry them, and the plate panel that shows the matched filter."""

from __future__ import annotations

import json

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from euclid_polish.config import Config
from euclid_polish.web.helpers import fits_plane, jwst_euclid, nexus_plates, real_tiles, viewer_data
from tests import _real_fixtures as fx

MATCHED = ("F090W", "F115W", "F150W")
MOSAIC_SIDE = 600


@pytest.fixture
def store(tmp_path, monkeypatch):
    return fx.point_store(tmp_path, monkeypatch)


def _mosaic_value(x, y, offset: float):
    """The synthetic mosaic: linear in its pixels, so bilinear resampling is exact."""
    return offset + 0.01 * x + 0.02 * y


def _write_mosaics(filters=MATCHED) -> WCS:
    """Gzip mosaics 18″ on a side, centred between tiles (0, 0) and (0, 1)."""
    ra, dec = fx.tile_position(0, 0.5)
    header = fx.wcs_header(ra, dec, 0.03, (MOSAIC_SIDE, MOSAIC_SIDE), bunit="MJy/sr")
    yy, xx = np.mgrid[0:MOSAIC_SIDE, 0:MOSAIC_SIDE]
    for offset, name in enumerate(filters, start=1):
        path = jwst_euclid.nexus_edr_mosaic_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(_mosaic_value(xx, yy, 10.0 * offset).astype(np.float32),
                        header=header).writeto(path)
    return WCS(header)


@pytest.fixture
def field(store):
    # tile 0 north of tile 1 (rows out of manifest order); tile 2 far outside the mosaics
    identifier = fx.make_nexus_field(offsets=[(0, 1), (0, 0), (0, 8)])
    return identifier, jwst_euclid.nexus_field_root() / identifier


def _manifest(directory):
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def test_matched_filters_skip_the_fields_own():
    assert Config.JWST_PARTNER_FILTER == {"VIS": "F090W", "Y_E": "F115W", "J_E": "F150W", "H_E": "F200W"}
    assert jwst_euclid.nexus_matched_filters({"filter": "F200W"}) == list(MATCHED)
    assert jwst_euclid.nexus_matched_filters({"filter": "F444W"}) == [*MATCHED, "F200W"]


def test_cache_resamples_each_mosaic_onto_the_tile_grids(field, monkeypatch):
    identifier, directory = field
    mosaic_wcs = _write_mosaics()
    opens = []
    original = fits_plane.FitsPlane._open_stream

    def counted(self):
        opens.append(self.path.name)
        return original(self)

    monkeypatch.setattr(fits_plane.FitsPlane, "_open_stream", counted)
    summary = jwst_euclid.cache_nexus_matched_bands(identifier)
    assert summary == {name: {"written": 2, "existing": 0, "uncovered": 1} for name in MATCHED}
    # each gzip mosaic streamed once, forward: the tiles were taken in row order
    assert sorted(opens) == sorted(jwst_euclid.nexus_edr_mosaic_path(n).name for n in MATCHED)

    manifest = _manifest(directory)
    assert manifest["jwst_band_uncovered"] == {name: [2] for name in MATCHED}
    assert manifest["jwst_band_sources"]["F090W"] == "wide_ep1_01_f090w_i2d_data.fits.gz"
    first = manifest["tiles"][0]
    assert first["jwst_band_files"] == {name: f"tiles/jwst_{name.lower()}_0000.fits" for name in MATCHED}
    assert "jwst_band_files" not in manifest["tiles"][2]

    # on the F200W tile's own grid, the mosaic's values at those sky positions
    with fits.open(directory / first["jwst_file"]) as hdul:
        own = WCS(hdul[0].header)
        shape = hdul[0].data.shape
    with fits.open(directory / first["jwst_band_files"]["F115W"]) as hdul:
        data, header = hdul[0].data, hdul[0].header
    assert data.shape == shape and data.dtype.kind == "f" and data.dtype.itemsize == 4
    assert header["BUNIT"] == "MJy/sr" and header["FILTER"] == "F115W"
    assert header["SRCFILE"] == "wide_ep1_01_f115w_i2d_data.fits.gz"
    assert WCS(header).wcs.crval == pytest.approx(own.wcs.crval)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    mx, my = mosaic_wcs.world_to_pixel_values(*own.pixel_to_world_values(xx, yy))
    np.testing.assert_allclose(data, _mosaic_value(mx, my, 20.0), rtol=1e-5)


def test_a_rerun_reuses_every_file_without_opening_a_mosaic(field, monkeypatch):
    identifier, directory = field
    _write_mosaics()
    jwst_euclid.cache_nexus_matched_bands(identifier)

    def refuse(_path):
        raise AssertionError("nothing left to resample")

    with monkeypatch.context() as patch:
        patch.setattr(jwst_euclid.fits_plane, "open_plane", refuse)
        summary = jwst_euclid.cache_nexus_matched_bands(identifier)
    assert summary == {name: {"written": 0, "existing": 2, "uncovered": 1} for name in MATCHED}
    # one deleted file is filled again alone
    (directory / "tiles/jwst_f150w_0001.fits").unlink()
    summary = jwst_euclid.cache_nexus_matched_bands(identifier)
    assert summary["F150W"] == {"written": 1, "existing": 1, "uncovered": 1}


def test_a_missing_mosaic_is_named(field):
    identifier, _directory = field
    _write_mosaics(filters=("F090W", "F115W"))
    with pytest.raises(RuntimeError, match="wide_ep1_01_f150w_i2d_data.fits.gz"):
        jwst_euclid.cache_nexus_matched_bands(identifier)


def test_real_tiles_and_the_viewer_serve_the_filters_in_band_order(field):
    identifier, _directory = field
    _write_mosaics()
    jwst_euclid.cache_nexus_matched_bands(identifier)
    real_tiles.invalidate()
    entries = real_tiles.list_entries("nexus")
    assert [b["filter"] for b in entries[0].extras["jwst_bands"]] == [*MATCHED, "F200W"]
    assert [b["filter"] for b in entries[2].extras["jwst_bands"]] == ["F200W"]    # uncovered tile
    assert [p["band"] for p in real_tiles.jwst_planes(entries[0])] == [*MATCHED, "F200W"]

    cube, info = viewer_data.get_cube("real", 0, "jwst", {"source": "nexus"})
    assert cube.shape == (133, 133, 4) and info["bands"] == [*MATCHED, "F200W"]
    assert info["transfer_group"] == "jwst" and info["label"].startswith("NEXUS F090W·F115W·F150W·F200W")
    np.testing.assert_allclose(cube[..., 3], 0.5)                                 # the field's own F200W
    meta = viewer_data.get_meta("real", {"source": "nexus"})
    bands = meta["color"]["bands"]
    assert {name: bands[name]["euclid_band"] for name in (*MATCHED, "F200W")} == {
        "F090W": "VIS", "F115W": "Y_E", "F150W": "J_E", "F200W": "H_E"}
    assert all(bands[name]["display_only"] for name in MATCHED)
    assert bands["F090W"]["pivot_um"] == pytest.approx(0.9)
    # Euclid's own bands keep their calibrated constants
    assert not bands["VIS"].get("display_only")

    cube, info = viewer_data.get_cube("real", 2, "jwst", {"source": "nexus"})
    assert info["bands"] == ["F200W"]


def test_the_plate_panel_shows_the_matched_filter():
    info = {"bands": [*MATCHED, "F200W"]}
    cube = np.stack([np.full((2, 2), float(k)) for k in range(4)], axis=-1)
    for k, band in enumerate(Config.LR_INPUT_BAND_NAMES):
        assert nexus_plates._plane(cube, info, band)[0, 0] == k
        assert nexus_plates.shown_filter(info, band) == [*MATCHED, "F200W"][k]
    assert nexus_plates.shown_filter(info, "temp") == "F090W"
    assert nexus_plates.shown_filter({"bands": ["F200W"]}, "VIS") == "F200W"     # not cached yet
    # a Euclid cube still shows its own band
    assert nexus_plates._plane(cube, {"bands": list(Config.LR_INPUT_BAND_NAMES)}, "J_E")[0, 0] == 2
