"""TNG stamps end softly: a soft circular window before rotation (no light
on a straight frame or crop edge) and a smooth, VIS-driven surface-
brightness roll-off for lensed sources (no hard per-band isophote)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.photometry import ab_mag_to_electrons
from euclid_polish.tng._image import (
    SOFT_WINDOW_INNER_FRACTION,
    SOFT_WINDOW_RIM_MARGIN_PX,
    _rotate_arbitrary_array,
    _soft_circular_window_array,
)
from euclid_polish.tng.renderer import SB_CUT_ROLLOFF_DEX, TNGRenderer
from euclid_polish.tng.types import TNGRotation, TNGView


def _write_galaxy(root: Path, subhalo_id: str, frame: np.ndarray,
                  band_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)) -> TNGView:
    directory = root / subhalo_id
    directory.mkdir(parents=True)
    for band, k in zip(("VIS", "Y", "J", "H"), band_scale, strict=True):
        hdu = fits.PrimaryHDU(np.asarray(frame * k, dtype=">f4"))
        hdu.header["BUNIT"] = "MJy/sr"
        for key, value in (("CDELT1", 100.0), ("CUNIT1", "pc"),
                           ("CDELT2", 100.0), ("CUNIT2", "pc")):
            hdu.header[key] = value
        hdu.writeto(directory / f"TNG{subhalo_id}_O1_Euclid_{band}.fits")
    (directory / Config.Tng.DONE_MARKER).touch()
    return TNGView(directory, subhalo_id, 1, native_re_px=20.0)


def _edge_lit_frame(size: int = 200) -> np.ndarray:
    """A bright core on light that runs into every edge of the frame — the
    large-donor case whose stamps used to end on a hard square step."""
    frame = np.ones((size, size), np.float32)
    c = size // 2
    frame[c - 8:c + 8, c - 8:c + 8] = 300.0
    return frame


def _ring(a: np.ndarray, width: int = 2) -> np.ndarray:
    mask = np.ones(a.shape[:2], bool)
    mask[width:-width, width:-width] = False
    return a[mask]


class _FixedAngle(TNGRenderer):
    angle = 0.0

    def _draw_nominal_rotation(self, rng):  # type: ignore[override]
        return TNGRotation(angle_deg=float(self.angle))


def test_window_is_one_inside_zero_beyond_the_rim_and_the_same_in_every_band():
    cube = np.ones((101, 101, 4), np.float32)
    w = _soft_circular_window_array(cube)
    rim = 101 / 2.0 - SOFT_WINDOW_RIM_MARGIN_PX
    rows, cols = np.ogrid[:101, :101]
    r = np.hypot(rows - 50.0, cols - 50.0)
    assert np.all(w[r <= SOFT_WINDOW_INNER_FRACTION * rim] == 1.0)
    assert np.all(w[r >= rim] == 0.0)
    for b in range(1, 4):
        np.testing.assert_array_equal(w[..., b], w[..., 0])
    inside = (r > SOFT_WINDOW_INNER_FRACTION * rim) & (r < rim)
    assert np.all((w[..., 0][inside] > 0.0) & (w[..., 0][inside] < 1.0))


@pytest.mark.parametrize("angle", [0.0, 17.0, 30.0, 45.0])
def test_a_windowed_frame_has_no_light_on_its_edge_at_any_rotation(angle):
    cube = np.ones((120, 120, 4), np.float32)          # lit to every edge
    rotated = _rotate_arbitrary_array(_soft_circular_window_array(cube), angle)
    assert _ring(rotated).max() < 1e-4
    plain = _rotate_arbitrary_array(cube, angle)       # what it used to be
    assert _ring(plain).max() > 0.5


@pytest.mark.parametrize("angle", [0.0, 30.0])
@pytest.mark.parametrize("shrink", [0.5, 0.08])     # 0.08: ~12 native px per output px
def test_a_rendered_stamp_of_an_edge_lit_donor_ends_softly(tmp_path, angle, shrink):
    view = _write_galaxy(tmp_path, "901", _edge_lit_frame())
    renderer = _FixedAngle(pixel_scale_arcsec=0.05)
    renderer.angle = angle
    stamp = renderer.render_observed_radius(
        view, view.native_re_arcsec(0.05) * shrink, rng=np.random.default_rng(0)
    ).as_array()
    for b in range(4):
        band = stamp[..., b]
        assert _ring(band, 1).max() < 1e-4 * band.max()


def test_a_stamp_clipped_to_the_maximum_side_ends_softly(tmp_path):
    view = _write_galaxy(tmp_path, "903", _edge_lit_frame(400))
    renderer = _FixedAngle(pixel_scale_arcsec=0.05, max_output_side=41)
    rendered = renderer.render_observed_radius(
        view, view.native_re_arcsec(0.05) * 0.9, rng=np.random.default_rng(0))
    assert rendered.trace.render_support_clipped
    stamp = rendered.as_array()
    assert max(stamp.shape[:2]) <= 41
    assert _ring(stamp[..., 0], 1).max() < 1e-4 * stamp[..., 0].max()


def test_rendering_still_draws_exactly_one_random_number(tmp_path):
    view = _write_galaxy(tmp_path, "904", _edge_lit_frame())
    rng = np.random.default_rng(123)
    TNGRenderer(pixel_scale_arcsec=0.05).render_observed_radius(
        view, view.native_re_arcsec(0.05) * 0.5, rng=rng)
    reference = np.random.default_rng(123)
    reference.uniform(0.0, 360.0)                   # the rotation angle
    assert rng.bit_generator.state == reference.bit_generator.state


@pytest.mark.parametrize("cut", [0.0, 28.0])
def test_a_lens_stamp_in_the_arbitrary_rotation_branch_ends_softly(tmp_path, cut):
    view = _write_galaxy(tmp_path, "905", _edge_lit_frame(400))
    renderer = TNGRenderer(pixel_scale_arcsec=0.05)
    rendered = renderer.render_physical_at_redshift(
        view, 2.0, rng=np.random.default_rng(5), surface_brightness_cut_mag_arcsec2=cut)
    assert rendered.trace.rotation.is_arbitrary       # rebin >= 4 at z = 2
    stamp = rendered.as_array()
    for b in range(4):
        assert _ring(stamp[..., b], 1).max() < 1e-4 * stamp[..., b].max()


def _profile_frame(size: int = 160) -> np.ndarray:
    """A smooth exponential disc, so surface brightness crosses any cut
    gradually rather than in one step."""
    rows, cols = np.ogrid[:size, :size]
    r = np.hypot(rows - size / 2.0, cols - size / 2.0)
    return (300.0 * np.exp(-r / 6.0)).astype(np.float32)


def test_the_lens_surface_brightness_cut_rolls_off_smoothly_with_one_support(tmp_path):
    # Different brightness per band, so a per-band cut would give each band
    # its own support.
    view = _write_galaxy(tmp_path, "902", _profile_frame(), band_scale=(1.0, 0.1, 3.0, 10.0))
    renderer = TNGRenderer(pixel_scale_arcsec=0.05)
    cut = 24.0
    truncated = renderer.render_physical_at_redshift(
        view, 0.5, surface_brightness_cut_mag_arcsec2=cut).as_array()
    full = renderer.render_physical_at_redshift(
        view, 0.5, surface_brightness_cut_mag_arcsec2=0.0).as_array()
    threshold = ab_mag_to_electrons(cut, Config.get_band("VIS")) * 0.05 ** 2
    vis = truncated[..., 0]
    assert vis.max() > 10 * threshold                  # the cut lies inside the disc
    support = vis > 0
    for b in range(1, 4):                              # one support for every band
        np.testing.assert_array_equal(truncated[..., b] > 0, support)
    # A roll-off, not a hard isophote: light survives just below the cut.
    assert np.any(support & (vis < threshold))
    # The same pixels in the untruncated render (the cut's centred crop).
    ref = full
    for axis, n in ((0, vis.shape[0]), (1, vis.shape[1])):
        if n < full.shape[axis]:
            start = int(round((full.shape[axis] - 1) / 2.0)) - (n - 1) // 2
            ref = np.take(ref, np.arange(start, start + n), axis=axis)
    # Below the roll-off nothing survives; well above it nothing changes.
    faint = ref[..., 0] < threshold * 10 ** (-SB_CUT_ROLLOFF_DEX)
    assert np.all(truncated[faint] == 0.0)
    # (inside the flat core of the stamp's closing soft window).
    rows, cols = np.ogrid[:vis.shape[0], :vis.shape[1]]
    r = np.hypot(rows - (vis.shape[0] - 1) / 2.0, cols - (vis.shape[1] - 1) / 2.0)
    core = r <= SOFT_WINDOW_INNER_FRACTION * (min(vis.shape) / 2.0 - SOFT_WINDOW_RIM_MARGIN_PX)
    bright = core & (ref[..., 0] > threshold * 10 ** (SB_CUT_ROLLOFF_DEX + 0.01))
    assert bright.sum() > 20
    np.testing.assert_allclose(truncated[bright], ref[bright], rtol=1e-6)
