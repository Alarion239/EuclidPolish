"""Tests for HST template library, renderer, and generator integration.

No network access — every test fabricates synthetic HST-like FITS
cutouts in a tmp_path and indexes them in an :class:`HSTTemplateLibrary`.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.hst.catalog import HSTTemplateLibrary
from euclid_polish.hst.types import HSTCutoutMetadata
from euclid_polish.sky.hst_templates import (
    HSTTemplateRenderConfig,
    HSTTemplateRenderer,
    _annulus_median,
    _normalise_to_unit_flux,
    _rescale_2d,
)
from euclid_polish.sky.multiband_generator import (
    MultiBandGeneratorConfig,
    MultiBandSimulator,
)
from tests._tiny_catalog import TinyCosmosCatalog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_fake_cutout(
    path: str, side: int = 64, peak_e_per_s: float = 1.0, seed: int = 0,
) -> None:
    """Write a Gaussian-blob FITS cutout that looks vaguely HST-like."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:side, :side]
    cy = cx = (side - 1) / 2.0
    sigma = side / 8.0
    blob = peak_e_per_s * np.exp(
        -((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2)
    )
    # Realistic-ish small additive sky noise (e/s scale).
    noise = rng.normal(0.0, 0.005, size=(side, side))
    data = (blob + noise).astype(np.float32)

    primary = fits.PrimaryHDU()
    sci = fits.ImageHDU(data, name="SCI")
    sci.header["CRVAL1"]   = 150.0
    sci.header["CRVAL2"]   = 2.0
    sci.header["CD1_1"]    = -1.11111e-5     # 0.04"/pix
    sci.header["CD2_2"]    =  1.11111e-5
    sci.header["FILTER"]   = "F814W"
    sci.header["BUNIT"]    = "ELECTRONS/S"
    fits.HDUList([primary, sci]).writeto(path, overwrite=True)


@pytest.fixture
def library_with_three_templates(tmp_path):
    """Library populated with 3 synthetic FITS cutouts."""
    root = tmp_path / "hst_templates"
    root.mkdir()
    metas: List[HSTCutoutMetadata] = []
    lib = HSTTemplateLibrary(root=str(root))
    for k, cid in enumerate([1001, 1002, 1003]):
        fname = Config.HST_CUTOUT_FILENAME_FMT.format(
            cosmos_id=cid, size=64,
        )
        _write_fake_cutout(str(root / fname), side=64, seed=k)
        meta = HSTCutoutMetadata(
            cosmos_id=cid, ra_deg=150.0, dec_deg=2.0, size_pix=64,
            pixel_scale_arcsec=0.04, filename=fname,
        )
        lib.add(meta, save=False)
        metas.append(meta)
    lib.save()
    return lib, metas


# ---------------------------------------------------------------------------
# Preprocessing primitives
# ---------------------------------------------------------------------------

class TestPreprocessingHelpers:

    def test_annulus_median_of_constant_image(self):
        img = np.full((20, 20), 1.5, dtype=np.float32)
        assert _annulus_median(img, annulus_pix=3) == pytest.approx(1.5)

    def test_annulus_median_ignores_bright_center(self):
        img = np.full((20, 20), 1.0, dtype=np.float32)
        img[8:12, 8:12] = 1e6   # bright source in the centre
        # Annulus should still be the background level.
        assert _annulus_median(img, annulus_pix=3) == pytest.approx(1.0)

    def test_annulus_median_returns_zero_for_too_small_input(self):
        img = np.ones((4, 4), dtype=np.float32)
        assert _annulus_median(img, annulus_pix=3) == 0.0

    def test_rescale_2d_changes_shape_correctly(self):
        img = np.ones((64, 64), dtype=np.float32)
        out = _rescale_2d(img, factor=4.0)
        assert out.shape == (16, 16)
        assert out.dtype == np.float32

    def test_rescale_2d_preserves_mean(self):
        rng = np.random.default_rng(0)
        img = rng.uniform(0, 1, size=(128, 128)).astype(np.float32)
        out = _rescale_2d(img, factor=2.0)
        # Cubic spline preserves mean to within a few % on smooth-ish input.
        assert out.mean() == pytest.approx(img.mean(), rel=0.05)

    def test_normalise_to_unit_flux(self):
        img = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        out = _normalise_to_unit_flux(img)
        assert out is not None
        assert out.sum() == pytest.approx(1.0, rel=1e-5)

    def test_normalise_blank_image_returns_none(self):
        assert _normalise_to_unit_flux(np.zeros((4, 4))) is None
        assert _normalise_to_unit_flux(np.array([[np.nan]])) is None

    def test_normalise_clips_negatives(self):
        img = np.array([[1.0, -2.0], [3.0, -1.0]], dtype=np.float32)
        out = _normalise_to_unit_flux(img)
        assert out is not None
        assert (out >= 0).all()
        # Total mass on the positive part = 1 + 3 = 4 → each survives as
        # its share of unity.
        assert out[0, 0] == pytest.approx(0.25, rel=1e-5)
        assert out[1, 0] == pytest.approx(0.75, rel=1e-5)


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

class TestLibrary:

    def test_library_round_trip_persists_csv(self, library_with_three_templates):
        lib, _ = library_with_three_templates
        # Reopen — should load from CSV.
        lib2 = HSTTemplateLibrary(root=lib.root)
        assert len(lib2) == 3
        assert sorted(lib2.cutout_ids()) == [1001, 1002, 1003]

    def test_library_get_by_id(self, library_with_three_templates):
        lib, _ = library_with_three_templates
        meta = lib.get(1002)
        assert meta is not None
        assert meta.cosmos_id == 1002

    def test_library_get_missing_returns_none(self, library_with_three_templates):
        lib, _ = library_with_three_templates
        assert lib.get(99999) is None

    def test_library_rebuild_from_disk(self, library_with_three_templates):
        lib, _ = library_with_three_templates
        os.remove(lib.catalog_path)
        # New instance: catalog gone, but FITS still on disk.
        lib2 = HSTTemplateLibrary(root=lib.root)
        assert len(lib2) == 0
        n = lib2.rebuild_from_disk()
        assert n == 3
        assert os.path.exists(lib2.catalog_path)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class TestRenderer:

    def test_construction_requires_non_empty_library(self, tmp_path):
        empty = HSTTemplateLibrary(root=str(tmp_path / "empty"))
        with pytest.raises(ValueError, match="empty"):
            HSTTemplateRenderer(empty)

    def test_deposit_conserves_flux(self, library_with_three_templates):
        lib, _ = library_with_three_templates
        # Tight rescale range so we don't sample sizes that fall off-canvas.
        cfg = HSTTemplateRenderConfig(rescale_min=2.0, rescale_max=2.0)
        r = HSTTemplateRenderer(lib, cfg)
        canvas = np.zeros((128, 128, 4), dtype=np.float32)
        flux = np.array([1000.0, 500.0, 200.0, 100.0], dtype=np.float32)
        info = r.deposit_into_canvas(
            canvas, flux_per_band=flux, x_pix=64, y_pix=64,
            rng=np.random.default_rng(0),
        )
        assert info is not None
        # Per-band sums must equal the assigned per-band flux (up to
        # canvas-edge clipping, which is zero here because the template
        # comfortably fits at 64x64 → 32x32 after downsample-by-2 at
        # center 64,64).
        for k, f_target in enumerate(flux):
            assert canvas[..., k].sum() == pytest.approx(f_target, rel=1e-3)

    def test_deposit_clips_to_canvas_bounds(self, library_with_three_templates):
        lib, _ = library_with_three_templates
        r = HSTTemplateRenderer(lib, HSTTemplateRenderConfig(
            rescale_min=2.0, rescale_max=2.0,
        ))
        canvas = np.zeros((64, 64, 4), dtype=np.float32)
        flux = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32)
        # Deposit centered at the bottom-right corner of the canvas — only
        # ~1/4 of the patch overlaps. Should still write a valid amount,
        # not raise.
        info = r.deposit_into_canvas(
            canvas, flux_per_band=flux, x_pix=63, y_pix=63,
            rng=np.random.default_rng(0),
        )
        # Either deposit happens (partial overlap) or returns None — both
        # are valid; no crash, no NaNs.
        assert np.all(np.isfinite(canvas))

    def test_deposit_validates_canvas_shape(self, library_with_three_templates):
        lib, _ = library_with_three_templates
        r = HSTTemplateRenderer(lib, HSTTemplateRenderConfig())
        with pytest.raises(ValueError, match=r"\(H, W, C\)"):
            r.deposit_into_canvas(
                np.zeros((128, 128), dtype=np.float32),
                flux_per_band=np.zeros(4), x_pix=0, y_pix=0,
                rng=np.random.default_rng(0),
            )

    def test_deposit_validates_flux_length(self, library_with_three_templates):
        lib, _ = library_with_three_templates
        r = HSTTemplateRenderer(lib, HSTTemplateRenderConfig())
        with pytest.raises(ValueError, match="length"):
            r.deposit_into_canvas(
                np.zeros((64, 64, 4), dtype=np.float32),
                flux_per_band=np.zeros(3),     # wrong length
                x_pix=32, y_pix=32,
                rng=np.random.default_rng(0),
            )

    def test_deposit_reports_rescale_factor(self, library_with_three_templates):
        lib, _ = library_with_three_templates
        cfg = HSTTemplateRenderConfig(rescale_min=3.0, rescale_max=3.0)
        r = HSTTemplateRenderer(lib, cfg)
        canvas = np.zeros((128, 128, 4), dtype=np.float32)
        info = r.deposit_into_canvas(
            canvas, flux_per_band=np.ones(4),
            x_pix=64, y_pix=64,
            rng=np.random.default_rng(0),
        )
        assert info is not None
        assert info["rescale_factor"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Generator integration
# ---------------------------------------------------------------------------

class TestGeneratorIntegration:

    def test_simulator_rejects_fraction_without_renderer(self):
        cat = TinyCosmosCatalog(n_galaxies=50, seed=0)
        cfg = MultiBandGeneratorConfig(
            image_size=64, hst_template_fraction=0.5,
        )
        with pytest.raises(ValueError, match="hst_template_renderer"):
            MultiBandSimulator(cat, cfg)

    def test_simulator_uses_templates_when_enabled(
        self, library_with_three_templates,
    ):
        lib, _ = library_with_three_templates
        cat = TinyCosmosCatalog(n_galaxies=50, seed=0)
        cfg = MultiBandGeneratorConfig(
            image_size=128,
            gal_density_arcmin2=0,   # explicit n_galaxies below
            star_density_arcmin2=0,
            lens_density_arcmin2=0,
            hst_template_fraction=1.0,   # always use templates
        )
        renderer = HSTTemplateRenderer(
            lib, HSTTemplateRenderConfig(rescale_min=2.0, rescale_max=2.0),
        )
        sim = MultiBandSimulator(cat, cfg, hst_template_renderer=renderer)
        _, meta = sim.simulate_field(
            np.random.default_rng(0), n_galaxies=5, n_stars=0, n_lenses=0,
        )
        # All five galaxies should be rendered through the template path
        # (fraction = 1.0 and the synthetic library is non-empty).
        for g in meta["galaxies"]:
            assert g["render"] == "hst_template"

    def test_simulator_default_still_uses_sersic(self):
        cat = TinyCosmosCatalog(n_galaxies=50, seed=0)
        cfg = MultiBandGeneratorConfig(image_size=64)
        sim = MultiBandSimulator(cat, cfg)
        _, meta = sim.simulate_field(
            np.random.default_rng(0), n_galaxies=3, n_stars=0, n_lenses=0,
        )
        for g in meta["galaxies"]:
            assert g["render"] == "sersic"


# ---------------------------------------------------------------------------
# Downloader (offline-safe — does not hit the network)
# ---------------------------------------------------------------------------

class TestDownloaderConfig:

    def test_default_config_validates(self):
        from euclid_polish.hst.downloader import HSTDownloadConfig
        cfg = HSTDownloadConfig()
        ok, err = cfg.validate()
        assert ok, err

    def test_invalid_size_rejected(self):
        from euclid_polish.hst.downloader import HSTDownloadConfig
        ok, err = HSTDownloadConfig(size_pix=0).validate()
        assert not ok
        assert "size" in err.lower()

    def test_cutout_path_uses_filename_template(self, tmp_path):
        from euclid_polish.hst.downloader import (
            HSTCutoutDownloader, HSTDownloadConfig,
        )
        dl = HSTCutoutDownloader(HSTDownloadConfig(
            output_dir=str(tmp_path), size_pix=256,
        ))
        path = dl.cutout_path(42)
        assert path.endswith(
            Config.HST_CUTOUT_FILENAME_FMT.format(cosmos_id=42, size=256)
        )

    def test_already_have_reads_disk(self, tmp_path):
        from euclid_polish.hst.downloader import (
            HSTCutoutDownloader, HSTDownloadConfig,
        )
        dl = HSTCutoutDownloader(HSTDownloadConfig(
            output_dir=str(tmp_path), size_pix=256,
        ))
        assert not dl.already_have(7)
        # Write a non-empty stub file at the expected path → should now
        # report present.
        with open(dl.cutout_path(7), "wb") as f:
            f.write(b"dummy")
        assert dl.already_have(7)
