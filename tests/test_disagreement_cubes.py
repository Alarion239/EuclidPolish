from __future__ import annotations

import json
import os

import numpy as np
import pytest
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
    with open(tmp_path / "disagreement.json") as f:
        meta = json.load(f)
    assert meta["pca_n"] == 3 and len(meta["pca_amps"]) == 3


def test_write_disagreement_cubes_few_members(tmp_path):
    members = np.ones((1, 6, 6, 4), np.float32)          # M=1 -> 0 pca comps
    amps = write_disagreement_cubes(str(tmp_path), members, n_components=3)
    assert amps == []
    assert (tmp_path / "std.fits").is_file()
    assert not (tmp_path / "pca0.fits").exists()


def test_disagreement_cubes_record_the_members_they_span(tmp_path):
    """A pruned production gate runs only the members it reads: members.json
    keeps the gate's full fitted list as the identity and records which
    members the std/PCA cubes were computed over."""
    members = np.random.default_rng(1).normal(5.0, 1.0, (2, 6, 6, 4)).astype(np.float32)
    ident = {"member_labels": ["1·psnr", "2·psnr", "3·psnr"], "combiner_kind": "spatial_gate",
             "combiner_fingerprint": "ab" * 32, "run_labels": ["1·psnr", "3·psnr"]}
    write_disagreement_cubes(str(tmp_path), members, member_labels=ident["member_labels"],
                             identity=ident, disagreement_members=ident["run_labels"])
    with open(tmp_path / "members.json") as f:
        recorded = json.load(f)
    assert recorded == {**ident, "disagreement_members": ["1·psnr", "3·psnr"]}
    with pytest.raises(ValueError, match="3 disagreement member labels"):
        write_disagreement_cubes(str(tmp_path), members, member_labels=["1·psnr"],
                                 disagreement_members=["1·psnr", "2·psnr", "3·psnr"])
