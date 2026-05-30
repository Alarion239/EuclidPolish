"""Oversaturation / clipping rejection for downloaded Euclid cutouts.

The Euclid MER pipeline zeros pixels above well capacity, so a saturated
star has a non-finite / ``<= 0`` hole at its centre. ``core_is_saturated``
flags that on the central core — the same check the PSF extractor uses —
so the downloader can reject clipped cutouts instead of keeping them.
"""

from __future__ import annotations

import numpy as np

from euclid_polish.config import Config
from euclid_polish.euclid.downloader import DownloadConfig, core_is_saturated


def _clean(n: int = 21) -> np.ndarray:
    data = np.ones((n, n), dtype=float) * 5.0
    data[n // 2, n // 2] = 500.0  # a clean, unsaturated bright peak
    return data


def test_clean_core_is_not_saturated():
    assert core_is_saturated(_clean(), 7) is False


def test_zeroed_core_is_saturated():
    data = _clean()
    c = data.shape[0] // 2
    data[c - 1:c + 2, c - 1:c + 2] = 0.0   # MER-zeroed saturated core
    assert core_is_saturated(data, 7) is True


def test_negative_core_is_saturated():
    data = _clean()
    c = data.shape[0] // 2
    data[c, c] = -3.0
    assert core_is_saturated(data, 7) is True


def test_nan_core_is_saturated():
    data = _clean()
    c = data.shape[0] // 2
    data[c, c] = np.nan
    assert core_is_saturated(data, 7) is True


def test_zero_outside_core_is_fine():
    # A zero in the wings (outside the 7x7 core) must NOT trip the check.
    data = _clean(41)
    data[0, 0] = 0.0
    assert core_is_saturated(data, 7) is False


def test_core_size_zero_disables_check():
    data = np.zeros((21, 21))
    assert core_is_saturated(data, 0) is False


def test_download_config_default_core_size():
    # The download path defaults to the same core size as PSF extraction.
    cfg = DownloadConfig.for_band("VIS", cutout_size_vis_pixels=64)
    assert cfg.saturation_core_size == 7
