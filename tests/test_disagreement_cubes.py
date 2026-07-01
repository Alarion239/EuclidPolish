from __future__ import annotations

import json
import os

import numpy as np
from astropy.io import fits

from euclid_polish.eval.disagreement import write_disagreement_cubes


def test_write_disagreement_cubes(tmp_path):
    rng = np.random.default_rng(0)
    members = rng.normal(10.0, 1.0, (5, 8, 8, 4)).astype(np.float32)  # (M,H,W,C)
    amps = write_disagreement_cubes(str(tmp_path), members, n_components=3)

    assert len(amps) == 3
    assert all(a >= 0 for a in amps)
    with fits.open(tmp_path / "std.fits") as h:
        std = np.asarray(h[0].data)
    assert std.shape == (4, 8, 8)                       # channel-first (C,H,W)
    assert np.allclose(np.moveaxis(std, 0, -1), members.std(axis=0), atol=1e-4)
    for i in range(3):
        assert (tmp_path / f"pca{i}.fits").is_file()
    meta = json.load(open(tmp_path / "disagreement.json"))
    assert meta["pca_n"] == 3 and len(meta["pca_amps"]) == 3


def test_write_disagreement_cubes_few_members(tmp_path):
    members = np.ones((1, 6, 6, 4), np.float32)          # M=1 -> 0 pca comps
    amps = write_disagreement_cubes(str(tmp_path), members, n_components=3)
    assert amps == []
    assert (tmp_path / "std.fits").is_file()
    assert not (tmp_path / "pca0.fits").exists()
