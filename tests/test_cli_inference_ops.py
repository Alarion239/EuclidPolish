# tests/test_cli_inference_ops.py
from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from euclid_polish.cli.inference_ops import (
    evaluate_production_sr,
    fetch_and_superresolve,
    production_sr,
    reconstruct_and_render,
)
from euclid_polish.config import Config
from euclid_polish.eval.ensemble_infer import PRODUCTION_COMBINER_KIND, EvalEnsemble
from euclid_polish.image import Image, Role
from euclid_polish.provenance.records import Stamp
from euclid_polish.provenance.store import ProvStore

BANDS = ("VIS", "Y_E", "J_E", "H_E")


class _Members:
    """Two members: all-zero and all-100 e⁻ SR (mean 50 e⁻)."""

    member_labels = ["00·psnr", "01·psnr"]
    n_members = 2

    def member_arrays(self, lr, *_a, **_k):
        h, w = np.asarray(lr).shape[:2]
        return np.stack([np.zeros((2 * h, 2 * w, 4), np.float32),
                         np.full((2 * h, 2 * w, 4), 100.0, np.float32)])

    def upsample(self, lr, **_k):                    # the plain member mean
        raise AssertionError("production SR must go through the combiner")


class _Gate:
    """A fitted production combiner that outputs 7 e⁻ everywhere."""

    member_labels = ["00·psnr", "01·psnr"]
    use_lr = False

    def apply_field(self, stack, lr=None):
        return np.full(stack.shape[1:], 7.0, np.float32)


def _production_model():
    return EvalEnsemble(_Members(), _Gate(), PRODUCTION_COMBINER_KIND)


def _lr_img(h=4, w=4, index=0):
    return Image(
        data=np.zeros((h, w, 4), np.float32), pixel_scale_arcsec=0.10,
        band_names=BANDS, is_clean=False, index=index)


def _hr_img(h=8, w=8, value=0.0, index=0):
    return Image(
        data=np.full((h, w, 4), value, np.float32), pixel_scale_arcsec=0.05,
        band_names=BANDS, is_clean=True, index=index)


def test_production_sr_applies_the_combiner(tmp_path):
    store = ProvStore(str(tmp_path / "prov"))
    lr = _lr_img(index=3)
    lr.stamp = Stamp(id=store.mint())
    sr = production_sr(_production_model(), lr, store=store)
    assert sr.role is Role.SR and sr.index == 3
    assert sr.shape == (8, 8, 4) and sr.band_names == BANDS
    assert float(sr.data.mean()) == pytest.approx(7.0)      # the gate, not the 50 e⁻ mean
    assert sr.pixel_scale_arcsec == Config.DEFAULT_PIXEL_SCALE
    assert sr.stamp is not None and sr.stamp.parents == (lr.stamp.id,)


def test_reconstruct_and_render_writes_pngs(tmp_path):
    store = ProvStore(str(tmp_path / "prov"))
    lrs = [_lr_img(), _lr_img()]
    paths = reconstruct_and_render(lrs, _production_model(), str(tmp_path / "out"), store=store)
    assert len(paths) == 2
    for p in paths:
        assert os.path.exists(p) and os.path.getsize(p) > 0


def test_reconstruct_and_render_with_hr(tmp_path):
    store = ProvStore(str(tmp_path / "prov"))
    paths = reconstruct_and_render(
        [_lr_img()], _production_model(), str(tmp_path / "out"),
        hr_images=[_hr_img()], store=store)
    assert len(paths) == 1
    assert os.path.exists(paths[0]) and os.path.getsize(paths[0]) > 0


def test_fetch_and_superresolve_writes_fits_and_png(tmp_path):
    store = ProvStore(str(tmp_path / "prov"))
    mock_catalog = MagicMock()
    mock_catalog.fetch.return_value = _lr_img(8, 8)

    fits_path, png_path = fetch_and_superresolve(
        ra=10.0, dec=-5.0, size=8, model=_production_model(),
        out_dir=str(tmp_path / "out"), store=store, catalog=mock_catalog)
    assert os.path.exists(fits_path) and os.path.getsize(fits_path) > 0
    assert os.path.exists(png_path) and os.path.getsize(png_path) > 0
    assert float(Image.from_fits(fits_path).data.mean()) == pytest.approx(7.0)


def test_evaluate_production_sr_pairs_by_index_and_averages_fields():
    knee = float(Config.STRETCH_SCALE_E)
    peak, peak_str = float(Config.PSNR_PEAK_E), float(Config.PSNR_PEAK_STRETCHED)
    lrs = [_lr_img(index=0), _lr_img(index=1), _lr_img(index=2)]
    hrs = [_hr_img(value=9.0, index=1), _hr_img(value=8.0, index=0)]   # field 2: no target
    ticks = []
    out = evaluate_production_sr(_production_model(), lrs, hrs,
                                 on_progress=lambda i, n: ticks.append((i, n)))
    assert out["n_scored"] == 2
    assert out["psnr_raw"] == pytest.approx(np.mean(
        [10 * np.log10(peak ** 2 / (v - 7.0) ** 2) for v in (8.0, 9.0)]))
    assert out["psnr_stretched"] == pytest.approx(np.mean(
        [10 * np.log10(peak_str ** 2
                       / (np.arcsinh(v / knee) - np.arcsinh(7.0 / knee)) ** 2)
         for v in (8.0, 9.0)]))
    assert ticks == [(1, 3), (2, 3), (3, 3)]


def test_evaluate_production_sr_without_targets_is_nan():
    out = evaluate_production_sr(_production_model(), [_lr_img()], [])
    assert out["n_scored"] == 0 and np.isnan(out["psnr_raw"])
