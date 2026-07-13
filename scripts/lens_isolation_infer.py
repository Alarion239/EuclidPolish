#!/usr/bin/env python3
"""Run the isolated ensemble on a four-band LR FITS cube."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np
from astropy.io import fits

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.experiments.lens_isolation.config import ExperimentPaths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_fits")
    parser.add_argument("--ensemble-dir", default=ExperimentPaths().ensemble)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--aperture", type=int, default=None)
    return parser.parse_args(argv)


def _write_fits_atomic(path: str, data: np.ndarray) -> None:
    fd, temporary = tempfile.mkstemp(prefix=os.path.basename(path) + ".tmp-", dir=os.path.dirname(path))
    os.close(fd)
    try:
        fits.PrimaryHDU(np.moveaxis(np.asarray(data, np.float32), -1, 0)).writeto(temporary, overwrite=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None) -> int:
    args = parse_args(argv)
    from euclid_polish.experiments.lens_isolation.ensemble import (
        LensIsolationEnsemble,
        detection_score,
    )

    with fits.open(args.input_fits) as hdul:
        cube = np.asarray(hdul[0].data, np.float32)
    if cube.ndim != 3:
        raise ValueError("input FITS must be a three-dimensional four-band cube")
    if cube.shape[0] == 4:
        cube = np.moveaxis(cube, 0, -1)
    if cube.shape[-1] != 4:
        raise ValueError("input FITS must contain exactly four bands")
    ensemble = LensIsolationEnsemble(args.ensemble_dir)
    mean, disagreement = ensemble.predict(cube)
    os.makedirs(args.out_dir, exist_ok=True)
    mean_path = os.path.join(args.out_dir, "lens_isolation_mean.fits")
    disagreement_path = os.path.join(args.out_dir, "lens_isolation_disagreement.fits")
    _write_fits_atomic(mean_path, mean)
    _write_fits_atomic(disagreement_path, disagreement)
    metadata = {
        "experiment": "lens_isolation",
        "input": os.path.abspath(args.input_fits),
        "ensemble_dir": os.path.abspath(args.ensemble_dir),
        "members": ensemble.member_names,
        "detection_score": detection_score(mean, args.aperture),
        "aperture": args.aperture,
        "mean_fits": mean_path,
        "disagreement_fits": disagreement_path,
    }
    with open(os.path.join(args.out_dir, "lens_isolation.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
