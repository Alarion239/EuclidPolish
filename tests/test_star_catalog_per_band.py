"""Tests for the per-band StarCatalog flag schema and CSV round-trip."""

from __future__ import annotations

import os

import pytest

from euclid_polish.config import Config
from euclid_polish.euclid.catalog import StarCatalog
from euclid_polish.euclid.downloader import DownloadConfig


# ---------------------------------------------------------------------------
# Per-band flag round-trips
# ---------------------------------------------------------------------------

def test_set_and_is_valid_per_band():
    s: dict = {}
    StarCatalog.set_valid(s, 512, band="VIS")
    assert StarCatalog.is_valid(s, 512, band="VIS") is True
    # Same star, different band: not valid until we say so.
    assert StarCatalog.is_valid(s, 512, band="Y_E") is False
    StarCatalog.set_valid(s, 512, band="Y_E")
    assert StarCatalog.is_valid(s, 512, band="Y_E") is True
    assert StarCatalog.is_valid(s, 256, band="VIS") is False


def test_set_valid_clears_matching_corruption():
    s: dict = {}
    StarCatalog.set_corrupted(s, 512, band="VIS")
    assert StarCatalog.is_corrupted(s, 512, band="VIS") is True
    StarCatalog.set_valid(s, 512, band="VIS")
    assert StarCatalog.is_corrupted(s, 512, band="VIS") is False
    assert StarCatalog.is_valid(s, 512, band="VIS") is True


def test_set_corrupted_clears_matching_validity():
    s: dict = {}
    StarCatalog.set_valid(s, 256, band="J_E")
    assert StarCatalog.is_valid(s, 256, band="J_E") is True
    StarCatalog.set_corrupted(s, 256, band="J_E")
    assert StarCatalog.is_corrupted(s, 256, band="J_E") is True
    assert StarCatalog.is_valid(s, 256, band="J_E") is False


def test_is_valid_any_size_when_size_none():
    s: dict = {}
    StarCatalog.set_valid(s, 256, band="H_E")
    assert StarCatalog.is_valid(s, band="H_E") is True
    assert StarCatalog.is_valid(s, band="VIS") is False


def test_valid_bands_lists_only_bands_with_flag():
    s: dict = {}
    StarCatalog.set_valid(s, 512, band="VIS")
    StarCatalog.set_valid(s, 512, band="J_E")
    assert sorted(StarCatalog.valid_bands(s, size=512)) == ["J_E", "VIS"]


def test_valid_sizes_per_band():
    s: dict = {}
    StarCatalog.set_valid(s, 256, band="VIS")
    StarCatalog.set_valid(s, 512, band="VIS")
    StarCatalog.set_valid(s, 256, band="Y_E")
    assert sorted(StarCatalog.valid_sizes(s, band="VIS")) == [256, 512]
    assert StarCatalog.valid_sizes(s, band="Y_E") == [256]
    assert StarCatalog.valid_sizes(s, band="H_E") == []


# ---------------------------------------------------------------------------
# CSV round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_preserves_per_band_flags(tmp_path):
    cat = StarCatalog(str(tmp_path))
    star = {"id": 0, "ra": 12.3, "dec": 45.6, "magnitude": 21.4}
    StarCatalog.set_valid(star, 512, band="VIS")
    StarCatalog.set_valid(star, 512, band="Y_E")
    StarCatalog.set_corrupted(star, 256, band="H_E")
    cat.save({"stars": [star], "next_id": 1})

    # CSV on disk, not JSON.
    assert cat.catalog_path.endswith(".csv")
    assert os.path.isfile(cat.catalog_path)

    loaded = cat.load()
    s2 = loaded["stars"][0]
    assert StarCatalog.is_valid(s2, 512, band="VIS")
    assert StarCatalog.is_valid(s2, 512, band="Y_E")
    assert not StarCatalog.is_valid(s2, 256, band="H_E")
    assert StarCatalog.is_corrupted(s2, 256, band="H_E")


def test_psf_flux_columns_roundtrip_when_present(tmp_path):
    """A star carrying the raw PSF flux + error round-trips them; a star
    without them stays clean (no flux keys, no forced NaN column)."""
    cat = StarCatalog(str(tmp_path))
    with_flux = {"id": 0, "ra": 1.0, "dec": 2.0, "magnitude": 19.0,
                 "flux_psf_uJy": 12.5, "fluxerr_psf_uJy": 0.2}
    without   = {"id": 1, "ra": 3.0, "dec": 4.0, "magnitude": 20.0}
    cat.save({"stars": [with_flux, without], "next_id": 2})

    loaded = {s["id"]: s for s in cat.load()["stars"]}
    assert loaded[0]["flux_psf_uJy"] == 12.5
    assert loaded[0]["fluxerr_psf_uJy"] == 0.2
    assert "flux_psf_uJy" not in loaded[1]      # absent stays absent


def test_csv_columns_use_kind_band_size_naming(tmp_path):
    import pandas as pd
    cat = StarCatalog(str(tmp_path))
    star = {"id": 0, "ra": 1.0, "dec": 2.0, "magnitude": 20.0}
    StarCatalog.set_valid(star, 256, band="VIS")
    StarCatalog.set_corrupted(star, 512, band="J_E")
    StarCatalog.set_download_failed(star, 128, band="H_E")
    cat.save({"stars": [star]})

    df = pd.read_csv(cat.catalog_path)
    assert set(df.columns) >= {"id", "ra", "dec", "magnitude",
                               "valid:VIS:256",
                               "corrupted:J_E:512",
                               "download_failed:H_E:128"}


def test_load_recomputes_next_id_from_data(tmp_path):
    cat = StarCatalog(str(tmp_path))
    cat.save({"stars": [
        {"id": 5, "ra": 1.0, "dec": 2.0, "magnitude": 22.0},
        {"id": 9, "ra": 3.0, "dec": 4.0, "magnitude": 23.0},
    ]})
    # next_id is derived from the data — one past the largest id present.
    assert cat.load()["next_id"] == 10


def test_empty_catalog_round_trips(tmp_path):
    cat = StarCatalog(str(tmp_path))
    cat.save({"stars": []})
    loaded = cat.load()
    assert loaded["stars"] == []
    assert loaded["next_id"] == 0


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
    cat = StarCatalog(str(tmp_path))
    stars = [{"id": i, "ra": 1.0, "dec": 1.0, "magnitude": 21.0 + i}
             for i in range(3)]
    StarCatalog.set_valid(stars[0], 512, band="VIS")
    StarCatalog.set_valid(stars[1], 512, band="VIS")
    StarCatalog.set_valid(stars[1], 512, band="Y_E")
    cat.save({"stars": stars})

    summary = cat.get_summary()
    assert summary["total"] == 3
    assert summary["valid_by_band"]["VIS"] == 2
    assert summary["valid_by_band"]["Y_E"] == 1
    assert summary["valid_by_band"]["J_E"] == 0
    assert summary["valid_by_band"]["H_E"] == 0
