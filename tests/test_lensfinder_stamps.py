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
