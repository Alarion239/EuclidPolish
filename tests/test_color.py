"""Tests for the 4-band → RGB color renderer.

The naive (R, G, B) = (H_E, J_E, VIS) stack of raw electron counts is
dominated by the per-band exposure time and zeropoint — so AB-flat
objects come out blue, redder-than-AB-flat objects also come out
blue, etc. ``lupton_rgb`` fixes that by flux-calibrating each band
before mixing. These tests pin the calibration behaviour:

* An AB-flat input → grey output under the ``"ab_flat"`` reference.
* A solar-SED input → grey output under the ``"solar"`` reference.
* A redder-than-reference input → red dominates.
* A bluer-than-reference input → blue dominates.
* Output is always in ``[0, 1]``.
"""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.visualization import color


_BANDS = Config.LR_INPUT_BAND_NAMES   # ("VIS", "Y_E", "J_E", "H_E")


def _flat_in_ab(scale: float = 1e6) -> np.ndarray:
    """A constant cube that, after AB-flux normalisation, comes out
    equal-valued in every channel.

    Pixel values are picked per-band so that:
        e⁻ / (t_total · 10^(0.4·zp)) = scale
    is the same for every band.
    """
    img = np.zeros((4, 4, 4), dtype=np.float32)
    for k, name in enumerate(_BANDS):
        b = Config.get_band(name)
        img[..., k] = float(scale * b.t_total_s * 10 ** (0.4 * b.zeropoint_ab_e_per_s))
    return img


def _flat_in_solar(scale: float = 1e6) -> np.ndarray:
    """A cube whose calibrated values are equal across all bands AFTER
    solar normalisation — i.e., a source with the sun's SED."""
    img = np.zeros((4, 4, 4), dtype=np.float32)
    for k, name in enumerate(_BANDS):
        b = Config.get_band(name)
        # ab_flat factor:
        ab = b.t_total_s * 10 ** (0.4 * b.zeropoint_ab_e_per_s)
        # multiply by inverse solar balance so the calibrate() step's
        # solar balance brings it back to ``scale``:
        sb = 10 ** (-0.4 * color.SOLAR_AB_MAG[name])
        img[..., k] = float(scale * ab * sb)
    return img


# ---------------------------------------------------------------------------
# Calibration invariants
# ---------------------------------------------------------------------------

def test_ab_flat_input_calibrates_to_grey_under_ab_reference():
    """An AB-flat input must produce identical values across bands
    after :func:`calibrate` with the ``ab_flat`` reference."""
    img = _flat_in_ab(scale=1e3)
    out = color.calibrate(img, reference="ab_flat")
    # Same value (within float32 rounding) in every band.
    for k in range(1, 4):
        assert np.allclose(out[..., 0], out[..., k], rtol=1e-5), (
            f"channel {k} differs from channel 0: "
            f"{out[..., k].mean()} vs {out[..., 0].mean()}"
        )


def test_solar_input_calibrates_to_grey_under_solar_reference():
    """A solar-SED input must produce identical values across bands
    after :func:`calibrate` with the ``solar`` reference."""
    img = _flat_in_solar(scale=1e3)
    out = color.calibrate(img, reference="solar")
    for k in range(1, 4):
        assert np.allclose(out[..., 0], out[..., k], rtol=1e-5)


def test_unknown_reference_raises():
    img = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="reference must be"):
        color.calibrate(img, reference="garbage")


# ---------------------------------------------------------------------------
# Lupton stretch
# ---------------------------------------------------------------------------

def test_lupton_rgb_output_shape_and_bounds():
    cube = np.random.rand(16, 16, 4).astype(np.float32) * 1e5
    rgb = color.lupton_rgb(cube)
    assert rgb.shape == (16, 16, 3)
    assert rgb.dtype == np.float32
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0


def test_lupton_grey_input_renders_neutral():
    """AB-flat input under ab_flat reference must render as
    R = G = B (zero color saturation)."""
    img = _flat_in_ab(scale=10.0)
    # Set stretch low enough that the result isn't fully clipped.
    rgb = color.lupton_rgb(img, reference="ab_flat",
                           scheme="vis_nisp", Q=8.0, stretch=1e-4)
    # All three channels equal at every pixel.
    assert np.allclose(rgb[..., 0], rgb[..., 1], atol=1e-5)
    assert np.allclose(rgb[..., 1], rgb[..., 2], atol=1e-5)


def test_redder_source_renders_red_dominant():
    """A source with 10× more H_E flux than VIS (in ab_flat units)
    must render with R > B in the final RGB."""
    img = _flat_in_ab(scale=1.0)   # baseline calibrated to grey
    # 10× boost in H_E channel (index 3 for ("VIS","Y_E","J_E","H_E")).
    img[..., 3] *= 10.0
    rgb = color.lupton_rgb(img, reference="ab_flat",
                           scheme="vis_nisp", Q=8.0, stretch=1e-4)
    r_mean = float(rgb[..., 0].mean())
    b_mean = float(rgb[..., 2].mean())
    assert r_mean > b_mean, f"R={r_mean:.4f} should exceed B={b_mean:.4f}"


def test_bluer_source_renders_blue_dominant():
    """A source with 10× more VIS flux than H_E (in ab_flat units)
    must render with B > R."""
    img = _flat_in_ab(scale=1.0)
    img[..., 0] *= 10.0   # VIS boost
    rgb = color.lupton_rgb(img, reference="ab_flat",
                           scheme="vis_nisp", Q=8.0, stretch=1e-4)
    r_mean = float(rgb[..., 0].mean())
    b_mean = float(rgb[..., 2].mean())
    assert b_mean > r_mean, f"B={b_mean:.4f} should exceed R={r_mean:.4f}"


def test_zero_input_renders_black():
    """All-zero cube must produce an all-zero RGB."""
    img = np.zeros((8, 8, 4), dtype=np.float32)
    rgb = color.lupton_rgb(img, reference="ab_flat")
    assert float(rgb.max()) == 0.0


def test_negative_input_does_not_break_renderer():
    """Sky-subtracted bands can be slightly negative; the renderer must
    not produce NaN or Inf."""
    rng = np.random.default_rng(0)
    img = rng.normal(0.0, 100.0, size=(16, 16, 4)).astype(np.float32)
    rgb = color.lupton_rgb(img, reference="ab_flat", stretch=1e-4)
    assert np.isfinite(rgb).all()
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0


# ---------------------------------------------------------------------------
# Scheme dispatch + input validation
# ---------------------------------------------------------------------------

def test_unknown_scheme_raises():
    img = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="scheme must"):
        color.lupton_rgb(img, scheme="rainbow")


def test_mismatched_band_names_raises():
    img = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="band_names length"):
        color.lupton_rgb(img, band_names=("VIS", "Y_E"))


def test_scheme_can_skip_unused_bands():
    """If band_names omits one of the four standard bands but the
    scheme doesn't need it, rendering should still succeed."""
    img = np.zeros((4, 4, 3), dtype=np.float32) + 1.0
    out = color.lupton_rgb(
        img, band_names=("VIS", "J_E", "H_E"),
        scheme="vis_nisp",            # needs (H_E, J_E, VIS) — all present
        reference="ab_flat", stretch=1e-4,
    )
    assert out.shape == (4, 4, 3)


def test_scheme_missing_required_band_raises():
    img = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="needs band"):
        color.lupton_rgb(
            img, band_names=("VIS", "Y_E", "J_E"),
            scheme="vis_nisp",        # needs H_E, not in band_names
        )
