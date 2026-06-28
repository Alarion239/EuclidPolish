"""Tests for the Euclid round-trip cutout pipeline (Chunk A).

Covers the pure helpers in the two new scripts:

  * ``scripts/fasrc_download_euclid_sky_cutouts.py`` —
    :func:`_uniform_disk_positions`
  * ``scripts/fasrc_generate_euclid_roundtrip_tfrecords.py`` —
    :func:`_load_4band_cube`, :func:`_chop_cube`, :func:`_stamp_is_usable`

We don't exercise the FASRC download itself (needs the Euclid archive)
or the full main() runs (need network + tile coverage). Both are
covered indirectly by the FASRC pipeline integration test once cutouts
are staged.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest
from astropy.io import fits

_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts",
)


def _load_script(filename: str, modname: str):
    spec = importlib.util.spec_from_file_location(
        modname, os.path.join(_SCRIPTS_DIR, filename),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dl_mod():
    return _load_script(
        "fasrc_download_euclid_sky_cutouts.py", "_skydl",
    )


@pytest.fixture(scope="module")
def tf_mod():
    return _load_script(
        "fasrc_generate_euclid_roundtrip_tfrecords.py", "_rttf",
    )


# ---------------------------------------------------------------------------
# Random position generator
# ---------------------------------------------------------------------------

class TestUniformDiskPositions:
    """Generated positions must (a) sit inside the requested disk, (b) be
    distributed uniformly on the sphere, (c) handle high-latitude
    ``cos(dec)`` stretch correctly."""

    def test_yields_requested_count(self, dl_mod):
        rng = np.random.default_rng(0)
        df = dl_mod._uniform_disk_positions(
            ra_centre_deg=270.0, dec_centre_deg=66.0, radius_deg=2.0,
            n_positions=50, rng=rng,
        )
        assert len(df) == 50
        assert set(df.columns) >= {"id", "ra", "dec", "magnitude"}

    def test_all_positions_inside_disk(self, dl_mod):
        """Every generated position must fall within radius_deg of centre.

        Uses the small-angle approximation the helper itself uses
        (flat-sky after cos(dec) correction) — for a 2° radius at
        dec=66° the great-circle vs. flat approximation differs by
        ≪ 0.001°, well below our tolerance.
        """
        rng = np.random.default_rng(1)
        radius = 2.0
        df = dl_mod._uniform_disk_positions(
            ra_centre_deg=270.0, dec_centre_deg=66.0, radius_deg=radius,
            n_positions=200, rng=rng,
        )
        cos_dec = np.cos(np.deg2rad(66.0))
        dx = (df["ra"]  - 270.0) * cos_dec
        dy = (df["dec"] - 66.0)
        r  = np.hypot(dx, dy)
        # Tolerance covers float round-trip + the rejection-sample acceptance test.
        assert (r <= radius + 1e-9).all(), (
            f"some positions outside disk: max r = {r.max():.6f} (radius {radius})"
        )

    def test_uses_cos_dec_for_ra_stretch(self, dl_mod):
        """At dec=66°, RA must stretch by 1/cos(66°) ≈ 2.46×.

        If the helper forgot the cos(dec) factor, the RA spread would
        equal the Dec spread (both ~2°); with the factor, RA spread is
        ~5°. Test pins the order-of-magnitude so a future regression
        flagging back to flat-RA is caught immediately.
        """
        rng = np.random.default_rng(2)
        df = dl_mod._uniform_disk_positions(
            ra_centre_deg=270.0, dec_centre_deg=66.0, radius_deg=2.0,
            n_positions=500, rng=rng,
        )
        ra_range  = df["ra"].max()  - df["ra"].min()
        dec_range = df["dec"].max() - df["dec"].min()
        ratio = ra_range / dec_range
        # cos(66°) = 0.4067 → ratio ≈ 1 / 0.4067 ≈ 2.46. Loose check
        # because finite samples don't fill the disk edges exactly.
        assert 2.0 < ratio < 3.0, (
            f"expected RA/Dec range ratio ≈ 2.46 at dec=66°; got {ratio:.2f}"
        )

    def test_pole_rejection(self, dl_mod):
        """Centring at dec=90° must raise — flat-sky breaks at the pole."""
        rng = np.random.default_rng(3)
        with pytest.raises(ValueError, match="too close to a pole"):
            dl_mod._uniform_disk_positions(
                ra_centre_deg=0.0, dec_centre_deg=90.0, radius_deg=1.0,
                n_positions=10, rng=rng,
            )

    def test_ra_wraps_into_zero_360_range(self, dl_mod):
        """Centring near RA=0° must produce positions in [0, 360°), not negatives."""
        rng = np.random.default_rng(4)
        df = dl_mod._uniform_disk_positions(
            ra_centre_deg=0.5, dec_centre_deg=0.0, radius_deg=2.0,
            n_positions=200, rng=rng,
        )
        assert (df["ra"] >= 0.0).all()
        assert (df["ra"] <  360.0).all()


# ---------------------------------------------------------------------------
# 4-band cube assembly
# ---------------------------------------------------------------------------

def _layout_one_position(
    tmp_path, pid: int, side: int, *, missing_band: str | None = None,
    bad_shape_band: str | None = None,
):
    """Write a bundled multi-HDU FITS for one position.

    Layout matches the new download script:
        ``<tmp_path>/sky_NNNN.fits``
    with one ``ImageHDU`` per band (``EXTNAME`` ∈ ``LR_INPUT_BAND_NAMES``).
    """
    from euclid_polish.config import Config

    os.makedirs(str(tmp_path), exist_ok=True)
    hdul = fits.HDUList([fits.PrimaryHDU()])
    for i, band_name in enumerate(Config.LR_INPUT_BAND_NAMES):
        if band_name == missing_band:
            continue
        this_side = side - 1 if band_name == bad_shape_band else side
        data = np.full((this_side, this_side), float(i + 1), dtype=np.float32)
        hdu = fits.ImageHDU(data=data, name=band_name)
        # The real download bundle copies the source MAGZERO into every band
        # header (fasrc_download_euclid_sky_cutouts.py); mirror it so the
        # zeropoint conversion exercises the real path.
        hdu.header["MAGZERO"] = 24.6
        hdul.append(hdu)
    hdul.writeto(
        os.path.join(str(tmp_path), f"sky_{pid:04d}.fits"), overwrite=True,
    )


class TestLoadFourBandCube:

    def test_returns_cube_when_all_bands_present(self, tf_mod, tmp_path):
        from euclid_polish.config import Config

        _layout_one_position(tmp_path, pid=7, side=64)
        # ``scale_to_electrons=False`` so the constant-value channels
        # round-trip exactly; the units-scaling path is exercised
        # below in ``test_scales_each_band_by_its_integration_time``.
        cube = tf_mod._load_4band_cube(
            str(tmp_path), 7, vis_pixels=64, scale_to_electrons=False,
        )
        assert cube is not None
        assert cube.shape == (64, 64, 4)
        # Channels stacked in canonical order — value embedded by helper.
        for i in range(len(Config.LR_INPUT_BAND_NAMES)):
            assert (cube[..., i] == float(i + 1)).all(), (
                f"channel {i} ({Config.LR_INPUT_BAND_NAMES[i]}) holds "
                f"wrong value — stack order may be wrong"
            )

    def test_returns_none_when_any_band_missing(self, tf_mod, tmp_path):
        _layout_one_position(tmp_path, pid=7, side=64, missing_band="J_E")
        assert tf_mod._load_4band_cube(
            str(tmp_path), 7, vis_pixels=64, scale_to_electrons=False,
        ) is None

    def test_returns_none_when_any_band_wrong_shape(self, tf_mod, tmp_path):
        _layout_one_position(tmp_path, pid=7, side=64, bad_shape_band="Y_E")
        assert tf_mod._load_4band_cube(
            str(tmp_path), 7, vis_pixels=64, scale_to_electrons=False,
        ) is None

    def test_returns_none_when_position_absent(self, tf_mod, tmp_path):
        assert tf_mod._load_4band_cube(
            str(tmp_path), 99, vis_pixels=64, scale_to_electrons=False,
        ) is None

    def test_scales_each_band_by_its_magzero_zeropoint(self, tf_mod, tmp_path):
        """Default ``scale_to_electrons=True`` puts archive e⁻/s cutouts on the
        synthetic/HST total-electron scale via the band's MAGZERO zeropoint
        factor — the verify_star_photometry-validated conversion shared with
        the direct-cutout and star-anchor lanes (NOT the old ~2.3×-too-faint
        ``× t_total_s``).

        Without a consistent electron scale the round-trip LR would sit at a
        different brightness than synthetic LR after asinh stretch, skewing
        the round-trip vs supervised loss balance.
        """
        from euclid_polish.catalog.photometry import adu_per_s_to_electrons_factor
        from euclid_polish.config import Config

        _layout_one_position(tmp_path, pid=7, side=64)   # MAGZERO=24.6 per band
        cube = tf_mod._load_4band_cube(
            str(tmp_path), 7, vis_pixels=64, scale_to_electrons=True,
        )
        assert cube is not None
        for i, band_name in enumerate(Config.LR_INPUT_BAND_NAMES):
            band = Config.get_band(band_name)
            factor = adu_per_s_to_electrons_factor(24.6, band)
            expected = float(i + 1) * factor           # input value × ZP factor
            assert np.allclose(cube[..., i], expected), (
                f"channel {i} ({band_name}) scaled wrong: "
                f"got {cube[0, 0, i]}, expected {expected} "
                f"(input={i + 1}, factor={factor})"
            )


# ---------------------------------------------------------------------------
# Chopper
# ---------------------------------------------------------------------------

class TestChopCube:

    def test_count_matches_grid(self, tf_mod):
        cube = np.zeros((512, 512, 4), dtype=np.float32)
        stamps = list(tf_mod._chop_cube(cube, stamp_size=128))
        assert len(stamps) == 16              # 4×4 grid
        for s in stamps:
            assert s.shape == (128, 128, 4)

    def test_stamps_dont_overlap(self, tf_mod):
        cube = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)[..., None]
        stamps = list(tf_mod._chop_cube(cube, stamp_size=8))
        # 4 stamps, all distinct content.
        assert len(stamps) == 4
        seen_first_pixels = {float(s[0, 0, 0]) for s in stamps}
        assert len(seen_first_pixels) == 4

    def test_trailing_strip_discarded_when_not_multiple(self, tf_mod):
        """vis_pixels=100, stamp_size=64 → only 1 stamp, last 36 px dropped."""
        cube = np.zeros((100, 100, 4), dtype=np.float32)
        stamps = list(tf_mod._chop_cube(cube, stamp_size=64))
        assert len(stamps) == 1


# ---------------------------------------------------------------------------
# Stamp usability filter
# ---------------------------------------------------------------------------

class TestStampUsable:

    def test_accepts_clean_stamp(self, tf_mod):
        s = np.ones((32, 32, 4), dtype=np.float32)
        assert tf_mod._stamp_is_usable(s, max_zero_fraction=0.05)

    def test_rejects_nan(self, tf_mod):
        s = np.ones((32, 32, 4), dtype=np.float32)
        s[0, 0, 0] = np.nan
        assert not tf_mod._stamp_is_usable(s, max_zero_fraction=0.05)

    def test_rejects_high_zero_fraction(self, tf_mod):
        """One band > 5% zeros → reject the whole stamp.

        The round-trip Conv applies VIS only but the model takes all 4
        bands as input — junk in any input band poisons the prediction.
        """
        s = np.ones((10, 10, 4), dtype=np.float32)
        # Zero out 10/100 = 10% of channel 2 (J_E).
        s[:1, :, 2] = 0.0
        assert not tf_mod._stamp_is_usable(s, max_zero_fraction=0.05)

    def test_below_threshold_zeros_pass(self, tf_mod):
        s = np.ones((10, 10, 4), dtype=np.float32)
        # 4/100 = 4% zeros in one band, below the 5% threshold.
        s[0, :4, 0] = 0.0
        assert tf_mod._stamp_is_usable(s, max_zero_fraction=0.05)
