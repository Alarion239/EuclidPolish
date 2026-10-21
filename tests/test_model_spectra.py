"""Models › Spectra math (eval/model_spectra.py): radial power, coherence
and transfer against the truth; the error's direction structure (1 = no
preferred direction, a pixel-axis grid on the axes of its period band);
fields smaller than the windows reported, not measured; partial sums that
add like one pass."""
from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.eval import model_spectra as ms

BANDS = ["VIS", "Y_E", "J_E", "H_E"]
H = 320                     # 256 px after the border: 2×2 radial windows


def _smooth(rng, h=H, alpha=-3.0):
    """An isotropic Gaussian field with P ∝ k^alpha per band, unit RMS."""
    k = np.fft.fftfreq(h)
    ky, kx = np.meshgrid(k, k, indexing="ij")
    kr = np.hypot(kx, ky)
    kr[0, 0] = 1.0
    out = [np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(h, h))) * kr ** (alpha / 2)))
           for _ in BANDS]
    x = np.stack(out, axis=-1)
    return x / x.std()


def _values(rows):
    return np.array([[np.nan if v is None else v for v in row] for row in rows], float)


def test_an_output_equal_to_the_truth_has_coherence_and_transfer_one():
    truth = 5.0 * _smooth(np.random.default_rng(0)) + 10.0
    acc = ms.SpectraAccumulator()
    acc.add(truth, truth)
    radial = acc.result(BANDS)["radial"]
    assert len(radial["period_px"]) == 63 and radial["period_px"][0] == 128.0
    np.testing.assert_allclose(_values(radial["coherence"]), 1.0, atol=1e-6)
    np.testing.assert_allclose(_values(radial["transfer"]), 1.0, atol=1e-6)
    assert _values(radial["power_sr"]).shape == (4, 63)
    assert radial["power_sr"] == radial["power_truth"]


def test_white_noise_power_is_its_variance_and_its_error_has_no_direction():
    """Near zero flux asinh(x / 1 e⁻) ≈ x: white noise of 0.01 e⁻ RMS has a
    flat power of 1e-4; its error in asinh(x / 0.01 e⁻) is round."""
    rng = np.random.default_rng(1)
    truth = np.zeros((H, H, 4))
    acc = ms.SpectraAccumulator()
    for _ in range(3):
        noise = 0.01 * rng.normal(size=truth.shape)
        acc.add(noise, truth)
    record = acc.result(BANDS)
    power = _values(record["radial"]["power_sr"])
    assert np.median(power) == pytest.approx(1e-4, rel=0.05)
    for band, rows in record["direction"]["axis_diag"].items():
        for axis, diag in rows:
            assert axis == pytest.approx(1.0, abs=0.12), band
            assert diag == pytest.approx(1.0, abs=0.12), band
    mean_map = np.nanmean(_values(record["map"]["values"][0]))
    assert mean_map == pytest.approx(1.0, abs=0.02)


def test_a_pixel_axis_grid_shows_on_the_axes_of_its_period_band():
    rng = np.random.default_rng(2)
    truth = np.zeros((H, H, 4))
    y = np.arange(H)
    grid = 0.02 * (np.sin(2 * np.pi * y / 8.0)[:, None, None] + np.sin(2 * np.pi * y / 8.0)[None, :, None])
    acc = ms.SpectraAccumulator()
    acc.add(0.01 * rng.normal(size=truth.shape) + grid, truth)
    direction = acc.result(BANDS)["direction"]
    axis, diag = direction["axis_diag"]["6-12"][0]
    assert axis > 2.0 * diag                        # the 8-px grid sits on the axes
    for band in ("2-3", "3-6", "12-21"):
        a, d = direction["axis_diag"][band][0]
        assert a < 1.5 and d < 1.5, band           # other periods stay round
    angular = direction["angular"]["6-12"][0]
    peak = max(range(ms.ANGLE_BINS), key=lambda i: -1.0 if angular[i] is None else angular[i])
    assert peak in (0, 17, 18, 35)                  # 0° (along x) or 90° (along y)


def test_fields_smaller_than_the_windows_are_reported_not_measured():
    acc = ms.SpectraAccumulator()
    acc.add(np.ones((8, 8, 4)), np.ones((8, 8, 4)))
    record = acc.result(BANDS)
    assert record["available"] is False and record["n_fields"] == 1
    assert "smaller than" in record["reason"] and "radial" not in record


def test_partial_sums_add_like_one_pass():
    rng = np.random.default_rng(4)
    fields = [(_smooth(rng) + 0.1 * rng.normal(size=(H, H, 4)), _smooth(rng)) for _ in range(2)]
    one = ms.SpectraAccumulator()
    for sr, truth in fields:
        one.add(sr, truth)
    parts = [ms.field_sums(sr, truth) for sr, truth in fields]
    merged = ms.SpectraAccumulator()
    for part in reversed(parts):
        merged.merge(part)
    assert merged.result(BANDS) == one.result(BANDS)
    assert one.result(BANDS)["n_fields"] == 2


def test_map_blanks_bins_outside_the_measured_disc_and_axes_are_shared():
    rng = np.random.default_rng(5)
    acc = ms.SpectraAccumulator()
    acc.add(0.01 * rng.normal(size=(H, H, 4)), np.zeros((H, H, 4)))
    record = acc.result(BANDS)
    m = record["map"]
    assert m["k"][0] == -32 and m["k"][-1] == 31 and len(m["k"]) == 64
    values = m["values"][0]
    assert values[32][32] is None                   # DC
    assert values[0][0] is None                     # beyond the Nyquist circle
    assert values[32][0] is not None                # the axis Nyquist bin is measured
    assert record["direction"]["angle_deg"] == ms.angle_centres()
    assert ms.angle_centres()[0] == 2.5 and ms.angle_centres()[-1] == 177.5
    assert set(record["direction"]["axis_diag"]) == {"2-3", "3-6", "6-12", "12-21", "21-42"}
    assert set(record["direction"]["angular"]) == {"2-3", "3-6", "6-12", "12-21"}
