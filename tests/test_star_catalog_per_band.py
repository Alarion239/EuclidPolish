"""Tests for the per-band CatalogObject flag schema and CSV round-trip."""

from __future__ import annotations

import os

import pytest

from euclid_polish.config import Config
from euclid_polish.catalog.catalog_object import (
    CatalogObject,
    next_id,
    summarize,
)
from euclid_polish.catalog.downloader import DownloadConfig


# ---------------------------------------------------------------------------
# Per-band flag round-trips
# ---------------------------------------------------------------------------

def test_set_and_is_valid_per_band():
    o = CatalogObject(ra=0.0, dec=0.0)
    o.set_valid(512, band="VIS")
    assert o.is_valid(512, band="VIS") is True
    # Same object, different band: not valid until we say so.
    assert o.is_valid(512, band="Y_E") is False
    o.set_valid(512, band="Y_E")
    assert o.is_valid(512, band="Y_E") is True
    assert o.is_valid(256, band="VIS") is False


def test_set_valid_clears_matching_corruption():
    o = CatalogObject(ra=0.0, dec=0.0)
    o.set_corrupted(512, band="VIS")
    assert o.is_corrupted(512, band="VIS") is True
    o.set_valid(512, band="VIS")
    assert o.is_corrupted(512, band="VIS") is False
    assert o.is_valid(512, band="VIS") is True


def test_set_corrupted_clears_matching_validity():
    o = CatalogObject(ra=0.0, dec=0.0)
    o.set_valid(256, band="J_E")
    assert o.is_valid(256, band="J_E") is True
    o.set_corrupted(256, band="J_E")
    assert o.is_corrupted(256, band="J_E") is True
    assert o.is_valid(256, band="J_E") is False


def test_is_valid_any_size_when_size_none():
    o = CatalogObject(ra=0.0, dec=0.0)
    o.set_valid(256, band="H_E")
    assert o.is_valid(band="H_E") is True
    assert o.is_valid(band="VIS") is False


def test_valid_bands_lists_only_bands_with_flag():
    o = CatalogObject(ra=0.0, dec=0.0)
    o.set_valid(512, band="VIS")
    o.set_valid(512, band="J_E")
    assert sorted(o.valid_bands(size=512)) == ["J_E", "VIS"]


def test_valid_sizes_per_band():
    o = CatalogObject(ra=0.0, dec=0.0)
    o.set_valid(256, band="VIS")
    o.set_valid(512, band="VIS")
    o.set_valid(256, band="Y_E")
    assert sorted(o.valid_sizes(band="VIS")) == [256, 512]
    assert o.valid_sizes(band="Y_E") == [256]
    assert o.valid_sizes(band="H_E") == []


# ---------------------------------------------------------------------------
# CSV round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_preserves_per_band_flags(tmp_path):
    path = os.path.join(str(tmp_path), Config.CATALOG_FILE)
    star = CatalogObject(ra=12.3, dec=45.6, id=0, magnitude=21.4)
    star.set_valid(512, band="VIS")
    star.set_valid(512, band="Y_E")
    star.set_corrupted(256, band="H_E")
    CatalogObject.write([star], path)

    # CSV on disk, not JSON.
    assert path.endswith(".csv")
    assert os.path.isfile(path)

    loaded = CatalogObject.read(path)
    s2 = loaded[0]
    assert s2.is_valid(512, band="VIS")
    assert s2.is_valid(512, band="Y_E")
    assert not s2.is_valid(256, band="H_E")
    assert s2.is_corrupted(256, band="H_E")


def test_psf_flux_columns_roundtrip_when_present(tmp_path):
    """A star carrying the raw PSF flux + error round-trips them; a star
    without them reads back with ``flux_psf_uJy is None``."""
    path = os.path.join(str(tmp_path), Config.CATALOG_FILE)
    with_flux = CatalogObject(ra=1.0, dec=2.0, id=0, magnitude=19.0,
                              flux_psf_uJy=12.5, fluxerr_psf_uJy=0.2)
    without = CatalogObject(ra=3.0, dec=4.0, id=1, magnitude=20.0)
    CatalogObject.write([with_flux, without], path)

    loaded = {o.id: o for o in CatalogObject.read(path)}
    assert loaded[0].flux_psf_uJy == 12.5
    assert loaded[0].fluxerr_psf_uJy == 0.2
    assert loaded[1].flux_psf_uJy is None       # absent reads back as None


def test_csv_columns_use_kind_band_size_naming(tmp_path):
    import pandas as pd
    path = os.path.join(str(tmp_path), Config.CATALOG_FILE)
    star = CatalogObject(ra=1.0, dec=2.0, id=0, magnitude=20.0)
    star.set_valid(256, band="VIS")
    star.set_corrupted(512, band="J_E")
    star.set_download_failed(128, band="H_E")
    CatalogObject.write([star], path)

    df = pd.read_csv(path)
    assert set(df.columns) >= {"id", "ra", "dec", "magnitude",
                               "valid:VIS:256",
                               "corrupted:J_E:512",
                               "download_failed:H_E:128"}


def test_load_recomputes_next_id_from_data(tmp_path):
    path = os.path.join(str(tmp_path), Config.CATALOG_FILE)
    CatalogObject.write([
        CatalogObject(ra=1.0, dec=2.0, id=5, magnitude=22.0),
        CatalogObject(ra=3.0, dec=4.0, id=9, magnitude=23.0),
    ], path)
    # next_id is derived from the data — one past the largest id present.
    assert next_id(CatalogObject.read(path)) == 10


def test_empty_catalog_round_trips(tmp_path):
    path = os.path.join(str(tmp_path), Config.CATALOG_FILE)
    CatalogObject.write([], path)
    loaded = CatalogObject.read(path)
    assert loaded == []
    assert next_id(loaded) == 0


# ---------------------------------------------------------------------------
# DownloadConfig.for_band
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("band_name", ["VIS", "Y_E", "J_E", "H_E"])
def test_download_config_for_band_pulls_archive_ids(band_name):
    b = Config.get_band(band_name)
    dl = DownloadConfig.for_band(band_name)
    assert dl.band == band_name
    assert dl.instrument == b.archive_instrument
    if b.archive_filter:
        assert dl.filter_name == b.archive_filter
    else:
        assert dl.filter_name is None
    assert dl.pixel_scale_arcsec == b.pixel_scale_lr_arcsec


def test_download_config_for_band_passes_overrides():
    dl = DownloadConfig.for_band("Y_E", cutout_size=128, max_workers=4)
    assert dl.cutout_size == 128
    assert dl.max_workers == 4
    assert dl.band == "Y_E"
    assert dl.instrument == "NISP"


# ---------------------------------------------------------------------------
# VIS-pixel reference: same angular field, per-band native size
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vis_pixels", [256, 512, 1024])
def test_cutout_size_for_arcsec_is_uniform_across_bands(vis_pixels):
    """Every band's native pixel scale is 0.10″, so the helper returns the
    same size for every band — the archive delivers a uniform 0.10″/pix grid."""
    arcsec = vis_pixels * Config.BAND_VIS.pixel_scale_lr_arcsec
    sizes = {b.name: b.cutout_size_for_arcsec(arcsec) for b in Config.BANDS}
    assert set(sizes.values()) == {vis_pixels}, sizes


def test_for_band_with_vis_pixels_routes_through_helper():
    dl_vis = DownloadConfig.for_band("VIS", cutout_size_vis_pixels=512)
    dl_y   = DownloadConfig.for_band("Y_E", cutout_size_vis_pixels=512)
    assert dl_vis.cutout_size == 512
    # All bands share 0.10″/pix → same native size as VIS.
    assert dl_y.cutout_size == 512


def test_for_band_rejects_both_size_args():
    with pytest.raises(ValueError, match="not both"):
        DownloadConfig.for_band("VIS", cutout_size=256, cutout_size_vis_pixels=512)


def test_for_band_explicit_native_size_bypasses_conversion():
    """When the user supplies cutout_size, it's used verbatim."""
    dl = DownloadConfig.for_band("Y_E", cutout_size=300)
    assert dl.cutout_size == 300


@pytest.mark.parametrize("band_name", ["VIS", "Y_E", "J_E", "H_E"])
def test_for_band_default_cutout_size_when_no_size_args(band_name):
    """No size args → fallback to the dataclass default."""
    dl = DownloadConfig.for_band(band_name)
    assert dl.cutout_size == Config.DEFAULT_CUTOUT_SIZE


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("band_name", ["VIS", "Y_E", "J_E", "H_E"])
def test_cutout_dir_for_band_includes_band_name(band_name):
    p = Config.cutout_dir_for_band(band_name)
    assert p.endswith(os.sep + band_name)


def test_summary_includes_per_band_breakdown(tmp_path):
    path = os.path.join(str(tmp_path), Config.CATALOG_FILE)
    stars = [CatalogObject(ra=1.0, dec=1.0, id=i, magnitude=21.0 + i)
             for i in range(3)]
    stars[0].set_valid(512, band="VIS")
    stars[1].set_valid(512, band="VIS")
    stars[1].set_valid(512, band="Y_E")
    CatalogObject.write(stars, path)

    summary = summarize(CatalogObject.read(path))
    assert summary["total"] == 3
    assert summary["valid_by_band"]["VIS"] == 2
    assert summary["valid_by_band"]["Y_E"] == 1
    assert summary["valid_by_band"]["J_E"] == 0
    assert summary["valid_by_band"]["H_E"] == 0
