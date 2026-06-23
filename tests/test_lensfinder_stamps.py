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
    def _field(self):
        lr_cube = np.random.default_rng(1).random((128, 128, 4)).astype(np.float32)
        sr_arr = np.random.default_rng(2).random((256, 256, 4)).astype(np.float32)
        hr_raw = np.random.default_rng(3).random((256, 256)).astype(np.float32)
        return lr_cube, sr_arr, hr_raw

    def test_cut_triplet_shapes(self):
        lr_cube, sr_arr, hr_raw = self._field()
        t = st.cut_triplet(lr_cube, sr_arr, hr_raw, cx=128.0, cy=100.0, m=M)
        assert t["lr_vis"].shape == (M // 2, M // 2)     # 64×64 LR
        assert t["sr_vis"].shape == (M, M)               # 128×128 SR
        assert t["hr_vis"].shape == (M, M)               # 128×128 HR

    def test_recon_planes_common_grid(self):
        lr_cube, sr_arr, hr_raw = self._field()
        t = st.cut_triplet(lr_cube, sr_arr, hr_raw, cx=128.0, cy=128.0, m=M)
        planes = st.recon_planes(t)
        assert set(planes) == {"lr", "sr", "hr"}
        for p in planes.values():
            assert p.shape == (M, M)                     # LR upsampled to M×M

    def test_lr_upsample_doubles(self):
        a = np.arange(16, dtype=np.float32).reshape(4, 4)
        up = st.lr_upsample_to_grid(a)
        assert up.shape == (8, 8)
        assert up[0, 0] == a[0, 0] and up[1, 1] == a[0, 0]   # nearest 2×


class TestRender:
    def test_render_stamp_png(self, tmp_path):
        from PIL import Image
        plane = np.random.default_rng(0).random((128, 128)).astype(np.float32)
        out = str(tmp_path / "sr.png")
        st.render_stamp_png(plane, out, asinh_scale=100.0, size=424)
        with Image.open(out) as im:
            assert im.size == (424, 424) and im.mode == "RGB"
