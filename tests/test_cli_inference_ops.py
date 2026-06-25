# tests/test_cli_inference_ops.py
from __future__ import annotations

import numpy as np

from euclid_polish.model import Model
from euclid_polish.provenance.store import ProvStore
from euclid_polish.sky.types import MultiBandSkyImage

BANDS = ("VIS", "Y_E", "J_E", "H_E")


def _fake_model():
    m = Model.__new__(Model)
    m._tf_model = object()
    m.id = None
    m._reconstruct_fn = lambda mdl, a: (
        np.asarray(a, np.float32)[..., 0],
        np.zeros((np.asarray(a).shape[0] * 2, np.asarray(a).shape[1] * 2, 4), np.float32),
    )
    return m


def _lr_img(h=4, w=4):
    return MultiBandSkyImage(
        data=np.zeros((h, w, 4), np.float32), pixel_scale_arcsec=0.10,
        band_names=BANDS, is_clean=False)


def _hr_img(h=8, w=8):
    return MultiBandSkyImage(
        data=np.zeros((h, w, 4), np.float32), pixel_scale_arcsec=0.05,
        band_names=BANDS, is_clean=True)


def test_reconstruct_and_render_writes_pngs(tmp_path):
    from euclid_polish.cli.inference_ops import reconstruct_and_render
    store = ProvStore(str(tmp_path / "prov"))
    lrs = [_lr_img(), _lr_img()]
    paths = reconstruct_and_render(lrs, _fake_model(), str(tmp_path / "out"), store=store)
    assert len(paths) == 2
    for p in paths:
        import os
        assert os.path.exists(p) and os.path.getsize(p) > 0


def test_reconstruct_and_render_with_hr(tmp_path):
    from euclid_polish.cli.inference_ops import reconstruct_and_render
    store = ProvStore(str(tmp_path / "prov"))
    paths = reconstruct_and_render(
        [_lr_img()], _fake_model(), str(tmp_path / "out"),
        hr_images=[_hr_img()], store=store)
    assert len(paths) == 1
    import os
    assert os.path.exists(paths[0]) and os.path.getsize(paths[0]) > 0
