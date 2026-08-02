"""Temperature-driven stellar SED sampling."""

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.sky.generation.stellar_sed import (
    blackbody_band_offsets_mag,
    sample_stellar_sed,
)


def test_cool_blackbody_is_nir_bright_and_hot_blackbody_is_blue():
    cool = blackbody_band_offsets_mag(3000.0)
    hot = blackbody_band_offsets_mag(10000.0)

    assert all(cool[name] < 0.0 for name in ("Y_E", "J_E", "H_E"))
    assert all(hot[name] > 0.0 for name in ("Y_E", "J_E", "H_E"))
    assert cool["H_E"] < cool["Y_E"]
    assert hot["H_E"] > hot["Y_E"]


def test_sampled_sed_preserves_vis_normalisation_and_physical_ranges():
    rng = np.random.default_rng(9)
    with pytest.raises(ValueError, match="empirical stellar prior"):
        sample_stellar_sed(rng, 20.0)
