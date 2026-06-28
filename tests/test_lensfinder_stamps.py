"""Tests for euclid_polish.lensfinder.stamps (geometry + render; no torch)."""

from __future__ import annotations

import numpy as np

from euclid_polish.lensfinder import stamps as st

FIELD = 256          # HR-grid field size
M = 128              # HR-grid stamp size → LR stamp 64


class TestSourceSelection:
    def _sources(self):
        return [
            {"type": "lens", "x_pix": 128.0, "y_pix": 128.0},     # center: fits
            {"type": "lens", "x_pix": 250.0, "y_pix": 250.0},     # edge: rejected
            {"type": "galaxy", "x_pix": 10.0, "y_pix": 10.0},     # edge: rejected
            {"type": "galaxy", "x_pix": 130.0, "y_pix": 120.0, "flux_vis_e": 50},
            {"type": "galaxy", "x_pix": 120.0, "y_pix": 140.0, "flux_vis_e": 999},
        ]

    def test_iter_filters_type_and_edges(self):
        lenses = st.iter_field_sources(self._sources(), want_type="lens",
                                       field=FIELD, m=M)
        assert len(lenses) == 1 and lenses[0]["x_pix"] == 128.0
        gals = st.iter_field_sources(self._sources(), want_type="galaxy",
                                     field=FIELD, m=M)
        assert len(gals) == 2          # the edge galaxy is rejected

    def test_sample_negatives_prefers_bright(self):
        gals = st.iter_field_sources(self._sources(), want_type="galaxy",
                                     field=FIELD, m=M)
        rng = np.random.default_rng(0)
        keep = st.sample_galaxy_negatives(gals, 1, rng=rng, prefer_bright=True)
        assert len(keep) == 1 and keep[0]["flux_vis_e"] == 999

    def test_sample_caps_at_available(self):
        rng = np.random.default_rng(0)
        keep = st.sample_galaxy_negatives([{"flux_vis_e": 1}], 5, rng=rng)
        assert len(keep) == 1


class TestTripletGeometry:
    def test_cut_triplet_keeps_four_bands_at_native_sizes(self):
        # Field big enough that a centered 106px stamp fits; LR is half-grid.
        lr_cube = np.random.default_rng(1).random((64, 64, 4)).astype(np.float32)
        sr_cube = np.random.default_rng(2).random((128, 128, 4)).astype(np.float32)
        hr_cube = np.random.default_rng(3).random((128, 128, 4)).astype(np.float32)
        t = st.cut_triplet(lr_cube, sr_cube, hr_cube, cx=64.0, cy=64.0, m=106)
        assert set(t) == {"lr", "sr", "hr"}
        assert t["lr"].shape == (53, 53, 4)      # LR half-grid, all 4 bands
        assert t["sr"].shape == (106, 106, 4)
        assert t["hr"].shape == (106, 106, 4)

    def test_cut_triplet_shares_crop_center(self):
        lr_cube = np.zeros((64, 64, 4), np.float32)
        sr_cube = np.zeros((128, 128, 4), np.float32)
        hr_cube = np.zeros((128, 128, 4), np.float32)
        sr_cube[64, 64, 0] = 9.0                 # mark HR-grid center
        hr_cube[64, 64, 0] = 9.0
        lr_cube[32, 32, 0] = 9.0                 # same point on LR half-grid
        t = st.cut_triplet(lr_cube, sr_cube, hr_cube, cx=64.0, cy=64.0, m=106)
        assert t["sr"][53, 53, 0] == 9.0         # center lands at stamp center
        assert t["hr"][53, 53, 0] == 9.0
        assert t["lr"][26, 26, 0] == 9.0


class TestRender:
    def test_render_stamp_rgb_writes_424_rgb(self, tmp_path):
        from PIL import Image
        rng = np.random.default_rng(0)
        stamp4 = (rng.random((106, 106, 4)).astype(np.float32) * 500.0)
        out = str(tmp_path / "sr.png")
        st.render_stamp_rgb(stamp4, out, size=424)
        with Image.open(out) as im:
            assert im.size == (424, 424) and im.mode == "RGB"

    def test_render_stamp_rgb_handles_flat_stamp(self, tmp_path):
        # A uniform (zero-contrast) stamp must not crash or produce NaNs.
        stamp4 = np.full((53, 53, 4), 3.0, np.float32)
        out = str(tmp_path / "lr.png")
        st.render_stamp_rgb(stamp4, out, size=424)
        import os
        assert os.path.exists(out)


class TestEvalRender:
    def _write_fits(self, path, arr):
        from astropy.io import fits
        fits.PrimaryHDU(np.ascontiguousarray(arr)).writeto(path, overwrite=True)

    def test_load_fits_cube_band_first_to_band_last(self, tmp_path):
        p = str(tmp_path / "sr.fits")
        self._write_fits(p, np.zeros((4, 12, 10), np.float32))   # (C, H, W)
        cube = st.load_fits_cube(p)
        assert cube.shape == (12, 10, 4)                          # (H, W, C)

    def test_load_fits_cube_2d_becomes_single_band(self, tmp_path):
        p = str(tmp_path / "hr_vis.fits")
        self._write_fits(p, np.zeros((12, 10), np.float32))       # legacy VIS-only
        cube = st.load_fits_cube(p)
        assert cube.shape == (12, 10, 1)

    def test_render_eval_stamp_crops_and_writes_424_rgb(self, tmp_path):
        from PIL import Image
        rng = np.random.default_rng(0)
        # SR-like 4-band 128px frame; crop to the 106 training size.
        arr = (rng.random((4, 128, 128)).astype(np.float32) * 500.0)
        src = str(tmp_path / "SR.fits")
        self._write_fits(src, arr)
        out = str(tmp_path / "after.png")
        st.render_eval_stamp(src, out, crop_m=106, size=424)
        with Image.open(out) as im:
            assert im.size == (424, 424) and im.mode == "RGB"

    def test_render_eval_stamp_centers_crop(self, tmp_path):
        # A single hot pixel at the frame center must survive the center-crop
        # (proves the crop is centered, not corner-anchored).
        arr = np.zeros((4, 128, 128), np.float32)
        arr[:, 64, 64] = 9.0
        src = str(tmp_path / "SR.fits")
        self._write_fits(src, arr)
        cube = st.load_fits_cube(src)
        from euclid_polish.eval.synthetic_runner import crop_stamp
        cropped = crop_stamp(cube[..., 0], cx=64.0, cy=64.0, m=106)
        assert cropped[53, 53] == 9.0          # center lands at stamp center

    def test_render_eval_stamp_fewer_than_4_bands_does_not_crash(self, tmp_path):
        import os
        # Legacy VIS-only HR.fits (2-D) must still render via band replication.
        arr = (np.random.default_rng(1).random((64, 64)).astype(np.float32) * 50)
        src = str(tmp_path / "HR.fits")
        self._write_fits(src, arr)
        out = str(tmp_path / "hr.png")
        st.render_eval_stamp(src, out, crop_m=53, size=424)
        assert os.path.exists(out)


def test_crop_path_does_not_pull_heavy_synthetic_runner():
    """The stamp crop path must not import euclid.eval.synthetic_runner, whose
    module-level imports drag in the astroquery-backed euclid stack — that broke
    lensfinder_score_eval in the PyTorch-only Zoobot env (no astroquery)."""
    import sys
    sys.modules.pop("euclid_polish.eval.synthetic_runner", None)
    lr = np.zeros((64, 64, 4), np.float32)
    sr = np.zeros((128, 128, 4), np.float32)
    hr = np.zeros((128, 128, 4), np.float32)
    st.cut_triplet(lr, sr, hr, cx=64.0, cy=64.0, m=106)
    assert "euclid_polish.eval.synthetic_runner" not in sys.modules
