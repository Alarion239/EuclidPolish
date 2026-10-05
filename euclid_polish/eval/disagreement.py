"""Write per-object ensemble-disagreement cubes for the evaluation movie viewer.

Given a ``(M, H, W, C)`` member stack, writes ``mean.fits`` (the member mean —
the centre the ``pcaN`` components are about, i.e. the disagreement movie's
base frame), ``std.fits`` (per-pixel member std), ``pca0..K.fits`` (the PCA
eigen-images of the member residuals) and a ``disagreement.json`` sidecar
``{pca_n, pca_amps, pca_var}``. FITS are channel-first ``(C, H, W)`` to match
``SR.fits`` so :func:`enforce_object_sizes` crops them consistently."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
from astropy.io import fits

from euclid_polish.ensemble import pca_field


def _write_cube_fits(path: str, hwc: np.ndarray) -> None:
    arr = np.asarray(hwc, dtype=np.float32)
    arr = np.moveaxis(arr, -1, 0) if arr.ndim == 3 else arr   # (C, H, W)
    hdr = fits.Header()
    hdr["BUNIT"] = "electron"
    fits.PrimaryHDU(np.ascontiguousarray(arr), header=hdr).writeto(
        path, overwrite=True, output_verify="silentfix")


def write_disagreement_cubes(obj_dir: str, members: np.ndarray,
                             *, n_components: int = 3,
                             member_labels: list[str] | None = None,
                             identity: Mapping[str, Any] | None = None,
                             disagreement_members: list[str] | None = None
                             ) -> list[float]:
    """Write ``mean.fits`` + ``std.fits`` + ``pca*.fits`` + ``disagreement.json``
    into ``obj_dir``. Returns the PCA amplitudes (population std along each
    component; empty when <2 members).

    ``member_labels`` (the ensemble's model labels, e.g. ``["00·psnr", …]``)
    is recorded in ``members.json`` as the membership fingerprint — reuse
    checks compare it against the registry's active labels so outputs from a
    since-changed ensemble are regenerated, not served stale. ``identity``
    (``catalog_runner.eval_model_identity``: members + production combiner
    kind/fingerprint) is recorded instead when given.
    ``disagreement_members`` names the members of ``members`` (the ones that
    ran — a pruned production gate runs only the members it reads) and is
    recorded as ``disagreement_members`` so the cubes say what they span.
    """
    mem = np.asarray(members, dtype=np.float32)
    os.makedirs(obj_dir, exist_ok=True)
    _write_cube_fits(os.path.join(obj_dir, "mean.fits"), mem.mean(axis=0))
    _write_cube_fits(os.path.join(obj_dir, "std.fits"), mem.std(axis=0))
    _mean, comps, amps, var_exp = pca_field(mem, n_components=n_components)
    for i, comp in enumerate(comps):
        _write_cube_fits(os.path.join(obj_dir, f"pca{i}.fits"), comp)
    amps_l = [float(a) for a in amps]
    with open(os.path.join(obj_dir, "disagreement.json"), "w") as f:
        json.dump({"pca_n": int(len(comps)), "pca_amps": amps_l,
                   "pca_var": [float(v) for v in var_exp]}, f)
    if identity is not None:
        record = {"member_labels": list(identity.get("member_labels") or member_labels or []),
                  "combiner_kind": identity.get("combiner_kind"),
                  "combiner_fingerprint": identity.get("combiner_fingerprint")}
        if identity.get("run_labels") is not None:
            record["run_labels"] = [str(v) for v in identity["run_labels"]]
    elif member_labels is not None:
        record = {"member_labels": list(member_labels)}
    else:
        record = None
    if record is not None and disagreement_members is not None:
        if len(disagreement_members) != mem.shape[0]:
            raise ValueError(f"{len(disagreement_members)} disagreement member labels "
                             f"for a {mem.shape[0]}-member stack")
        record["disagreement_members"] = [str(v) for v in disagreement_members]
    if record is not None:
        with open(os.path.join(obj_dir, "members.json"), "w") as f:
            json.dump(record, f)
    return amps_l
