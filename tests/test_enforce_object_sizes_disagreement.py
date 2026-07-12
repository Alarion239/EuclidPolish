from __future__ import annotations

import numpy as np
from astropy.io import fits

from euclid_polish.eval.catalog_runner import (
    EVAL_HR_SIZE,
    EVAL_LR_SIZE,
    enforce_object_sizes,
)


def _wr(path, arr):
    fits.PrimaryHDU(np.ascontiguousarray(arr.astype(np.float32))).writeto(
        path, overwrite=True, output_verify="silentfix")


def test_enforce_crops_std_and_pca(tmp_path):
    big = EVAL_HR_SIZE + 6
    _wr(tmp_path / "original_stack.fits", np.zeros((4, EVAL_LR_SIZE + 4, EVAL_LR_SIZE + 4)))
    _wr(tmp_path / "SR.fits", np.zeros((4, big, big)))
    _wr(tmp_path / "std.fits", np.zeros((4, big, big)))
    _wr(tmp_path / "pca0.fits", np.zeros((4, big, big)))
    assert enforce_object_sizes(str(tmp_path)) is True
    for name in ("SR.fits", "std.fits", "pca0.fits"):
        with fits.open(tmp_path / name) as h:
            assert h[0].data.shape[-2:] == (EVAL_HR_SIZE, EVAL_HR_SIZE)
