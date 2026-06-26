"""Tests for the reconstruction plotting (euclid_polish.visualization.reconstruction).

This lives in the visualization layer (which imports the image layer, never the
reverse); ``plot_imageset`` is the OO entry that renders an ImageSet by role.
"""
import os

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.image import Image, ImageSet, Role
from euclid_polish.visualization.reconstruction import plot_imageset


def _4band(side, scale, role):
    rng = np.random.default_rng(int(side * 10 + (scale * 100)))
    data = (np.abs(rng.normal(size=(side, side, 4))) * 100.0).astype(np.float32)
    return Image(data=data, pixel_scale_arcsec=scale,
                 band_names=Config.LR_INPUT_BAND_NAMES,
                 is_clean=(role is not Role.LR), role=role)


def test_plot_imageset_writes_png(tmp_path):
    s = ImageSet.from_images([
        _4band(8, 0.10, Role.LR),
        _4band(16, 0.05, Role.SR),
        _4band(16, 0.05, Role.HR),
    ])
    out = str(tmp_path / "recon.png")
    assert plot_imageset(s, out) == out
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_plot_imageset_requires_sr():
    with pytest.raises(ValueError):
        plot_imageset(ImageSet.from_images([_4band(8, 0.10, Role.LR)]), "x.png")
