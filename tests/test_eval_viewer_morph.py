from __future__ import annotations

import json

import numpy as np
from astropy.io import fits

import euclid_polish.web.helpers.viewer_data as vd


def _obj(dirpath, with_pca=True):
    def _wr(name, arr):
        fits.PrimaryHDU(np.ascontiguousarray(arr.astype(np.float32))).writeto(
            str(dirpath / name), overwrite=True, output_verify="silentfix")
    _wr("original_stack.fits", np.zeros((4, 8, 8)))
    _wr("SR.fits", np.zeros((4, 16, 16)))
    if with_pca:
        _wr("std.fits", np.zeros((4, 16, 16)))
        for i in range(3):
            _wr(f"pca{i}.fits", np.ones((4, 16, 16)) * (i + 1))
        with open(dirpath / "disagreement.json", "w") as f:
            json.dump({"pca_n": 3, "pca_amps": [0.3, 0.2, 0.1]}, f)


def test_eval_meta_advertises_morph(tmp_path, monkeypatch):
    d = tmp_path / "obj_a"
    d.mkdir()
    _obj(d)
    monkeypatch.setattr(vd, "_eval_objects", lambda: [{
        "subdir": "obj_a", "label": "a", "grade": "A",
        "tiers": ["LR", "SR", "std"], "plens": {},
        "pca_n": 3, "pca_amps": [0.3, 0.2, 0.1]}])
    meta = vd._eval_meta({})
    assert any(t["key"] == "morph" for t in meta["tiers"])
    assert meta["pca_n"] == 3
    assert meta["pca_amps"] == [[0.3, 0.2, 0.1]]
