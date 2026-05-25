"""Tests for the transition-model pair-generation script.

Covers the standalone helpers (PSF resampling, odd-square cropping,
clean-scene streaming) but not the full ``main()`` — running main() end
to end requires real synthetic clean records and Euclid VIS PSF FITS,
both of which live outside the test working tree.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))


def _load_script():
    import fasrc_generate_transition_pairs as mod
    return mod


# ---------------------------------------------------------------------------
# _resample_to_hr_grid
# ---------------------------------------------------------------------------

class TestResampleToHrGrid:

    def test_noop_when_scales_match(self):
        mod = _load_script()
        # Build a centred Gaussian at the HR scale already.
        side = 31
        sigma = 2.0
        y, x = np.mgrid[:side, :side]
        cy = cx = (side - 1) / 2.0
        g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
        g = (g / g.sum()).astype(np.float32)
        from euclid_polish.config import Config
        out = mod._resample_to_hr_grid(g, src_scale=Config.DEFAULT_PIXEL_SCALE)
        np.testing.assert_allclose(out, g, atol=1e-6)

    def test_renormalises_to_unit_flux(self):
        mod = _load_script()
        side = 41
        sigma = 3.0
        y, x = np.mgrid[:side, :side]
        cy = cx = (side - 1) / 2.0
        g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
        g = (g / g.sum()).astype(np.float32)
        # 0.03 → 0.05 = HLSP → HR. The zoom + renormalise should keep
        # sum=1 to within tiny float roundoff.
        out = mod._resample_to_hr_grid(g, src_scale=0.03)
        assert abs(out.sum() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# _crop_to_odd_square
# ---------------------------------------------------------------------------

class TestCropToOddSquare:

    def test_returns_odd_square(self):
        mod = _load_script()
        arr = np.zeros((100, 100), dtype=np.float32)
        arr[50, 50] = 1.0
        out = mod._crop_to_odd_square(arr, side=21)
        assert out.shape == (21, 21)

    def test_pads_when_smaller(self):
        mod = _load_script()
        arr = np.zeros((15, 15), dtype=np.float32)
        arr[7, 7] = 1.0
        out = mod._crop_to_odd_square(arr, side=21)
        assert out.shape == (21, 21)
        # The single hot pixel survived; check it's roughly centred.
        hot = np.argwhere(out > 0.5)
        assert hot.shape[0] == 1
        cy, cx = hot[0]
        assert abs(cy - 10) <= 1
        assert abs(cx - 10) <= 1

    def test_even_side_rejected(self):
        mod = _load_script()
        with pytest.raises(ValueError, match="odd"):
            mod._crop_to_odd_square(np.zeros((10, 10)), side=8)


# ---------------------------------------------------------------------------
# _convolve_pair — produces (HST-blurred, Euclid-blurred) at matching size
# ---------------------------------------------------------------------------

class TestConvolvePair:

    def test_shapes_match_input(self):
        mod = _load_script()
        scene = np.zeros((64, 64), dtype=np.float32)
        scene[32, 32] = 100.0   # single point source
        psf = np.ones((9, 9), dtype=np.float32) / 81.0
        inp, tgt = mod._convolve_pair(scene, psf, psf)
        assert inp.shape == scene.shape
        assert tgt.shape == scene.shape

    def test_flux_conserved_when_psfs_sum_to_unity(self):
        mod = _load_script()
        scene = np.zeros((64, 64), dtype=np.float32)
        # Multi-source so total flux is non-trivial.
        scene[16, 16] = 50.0
        scene[48, 48] = 70.0
        # Two distinct PSFs, both unit-flux.
        psf_hst = np.ones((5, 5), dtype=np.float32) / 25.0
        psf_eu  = np.zeros((9, 9), dtype=np.float32)
        psf_eu[4, 4] = 1.0
        psf_eu /= psf_eu.sum()
        inp, tgt = mod._convolve_pair(scene, psf_hst, psf_eu)
        # Both convolutions must preserve total flux exactly (within
        # FFT round-off + boundary handling for fftconvolve mode='same').
        np.testing.assert_allclose(inp.sum(), scene.sum(), rtol=1e-4)
        np.testing.assert_allclose(tgt.sum(), scene.sum(), rtol=1e-4)


# ---------------------------------------------------------------------------
# _stream_clean_vis_scenes — handshake with the synthetic TFRecord schema
# ---------------------------------------------------------------------------

class TestStreamCleanVisScenes:
    """Verifies the stream helper reads the right schema and centre-
    crops. We synthesise a minimal multiband TFRecord on the fly so the
    test doesn't depend on the synthetic pipeline being run first."""

    def _write_minimal_clean_record(self, tmp_path, name, n_scenes, side):
        """Write a tiny 4-channel clean TFRecord with N scenes."""
        from euclid_polish.config import Config
        from euclid_polish.sky.tfrecord import open_multiband_writer
        from euclid_polish.sky.types import MultiBandSkyImage
        rng = np.random.default_rng(0)
        with open_multiband_writer(name, records_dir=str(tmp_path)) as w:
            for i in range(n_scenes):
                data = rng.uniform(0, 100, size=(side, side, 4)).astype(np.float32)
                img = MultiBandSkyImage(
                    data=data,
                    pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                    band_names=Config.LR_INPUT_BAND_NAMES,
                    is_clean=True,
                )
                w.write(img, index=i)

    def test_yields_correct_shape_and_count(self, tmp_path):
        mod = _load_script()
        self._write_minimal_clean_record(
            tmp_path, "clean_validate", n_scenes=5, side=64,
        )
        out = list(mod._stream_clean_vis_scenes(
            str(tmp_path), "validate", n_max=10, crop=32,
        ))
        assert len(out) == 5     # source had only 5 scenes
        for _, scene in out:
            assert scene.shape == (32, 32)
            assert scene.dtype == np.float32

    def test_caps_at_n_max(self, tmp_path):
        mod = _load_script()
        self._write_minimal_clean_record(
            tmp_path, "clean_train", n_scenes=20, side=64,
        )
        out = list(mod._stream_clean_vis_scenes(
            str(tmp_path), "train", n_max=7, crop=32,
        ))
        assert len(out) == 7

    def test_skips_scenes_smaller_than_crop(self, tmp_path):
        mod = _load_script()
        self._write_minimal_clean_record(
            tmp_path, "clean_train", n_scenes=3, side=32,
        )
        # crop > source side → all scenes get skipped.
        out = list(mod._stream_clean_vis_scenes(
            str(tmp_path), "train", n_max=10, crop=64,
        ))
        assert out == []

    def test_missing_file_raises(self, tmp_path):
        mod = _load_script()
        with pytest.raises(FileNotFoundError):
            list(mod._stream_clean_vis_scenes(
                str(tmp_path), "validate", n_max=1, crop=32,
            ))
