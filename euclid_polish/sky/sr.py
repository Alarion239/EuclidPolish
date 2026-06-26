"""SR generation + status for sky records (the /sky "Generate SR" button).

Runs the trained super-resolution model over the locally-cached *dirty* (LR)
sky TFRecords and saves each record's SR cube as a ``.npy`` under
``data/vis/sky_sr/``. The unified cutout viewer then exposes those as the SR
tier (see ``web/helpers/viewer_data._sky_cube``).

Gating mirrors the UI: SR can only be generated when both the records and a
checkpoint are present locally (:func:`records_present`,
:func:`checkpoint_present`); the SR tier is only offered once at least one SR
cube exists (:func:`sr_count`).
"""
from __future__ import annotations

import glob
import os
from typing import Callable, Iterable, List, Optional

import numpy as np

from euclid_polish.config import Config
from euclid_polish.eval.catalog_runner import load_eval_model
from euclid_polish.provenance.checkpoint import model_id_of_checkpoint
from euclid_polish.provenance.defaults import default_store
from euclid_polish.provenance.gitinfo import capture_git
from euclid_polish.provenance.records import Format, InferenceRun, SRCutoutArtifact
from euclid_polish.image.tfio import read_multiband_skyimages, tfrecord_path
from euclid_polish.training.inference import reconstruct

#: Subsets we generate SR for, in priority order.
SUBSETS = ("validate", "train")


def sky_sr_dir() -> str:
    """Local directory holding generated sky SR cubes (one ``.npy`` each)."""
    return os.path.join(Config.VIS_DIR, "sky_sr")


def sr_path(subset: str, idx: int) -> str:
    return os.path.join(sky_sr_dir(), f"sr_{subset}_{int(idx):04d}.npy")


def sr_count(subset: str) -> int:
    """How many SR cubes have been generated for ``subset``."""
    return len(glob.glob(os.path.join(sky_sr_dir(), f"sr_{subset}_*.npy")))


def checkpoint_present(checkpoint: Optional[str] = None) -> bool:
    """True when a usable checkpoint is on disk (cheap; no TensorFlow import).

    Mirrors ``tf.train.latest_checkpoint`` without paying its import cost:
    a TF checkpoint dir carries a ``checkpoint`` pointer file plus per-step
    ``*.index`` shards.
    """
    ck = checkpoint or Config.DEFAULT_CHECKPOINT_DIR
    if not os.path.isdir(ck):
        return False
    return (os.path.isfile(os.path.join(ck, "checkpoint"))
            or bool(glob.glob(os.path.join(ck, "*.index"))))


def records_present(records_dir: str, subset: str = "validate") -> bool:
    """True when the dirty (LR) TFRecord for ``subset`` is in the local cache."""
    return os.path.exists(tfrecord_path(records_dir, f"dirty_{subset}"))


def present_subsets(records_dir: str) -> List[str]:
    return [s for s in SUBSETS if records_present(records_dir, s)]


def record_sr_cube(store, npy_path: str, subset: str, idx: int, *,
                   model_id=None, input_id=None, produced_by=None,
                   git=None, sidecar_dir: Optional[str] = None) -> SRCutoutArtifact:
    """Persist an :class:`SRCutoutArtifact` for one SR cube, next to the data.

    Parents are ``(model_id, input_id)`` (whichever are known) so the cube can
    later be told apart from a stale one. The sidecar is named by the artifact's
    id; the ``.npy`` keeps its viewer-visible name.
    """
    parents = tuple(p for p in (model_id, input_id) if p is not None)
    art = SRCutoutArtifact(
        id=store.mint(), git=git, produced_by=produced_by,
        format=Format.NPY, path=npy_path, parents=parents,
        descriptors={"subset": subset, "index": int(idx)},
    )
    store.put(art, sidecar_dir=sidecar_dir or sky_sr_dir())
    return art


def generate(records_dir: str, subsets: Iterable[str],
             checkpoint: Optional[str] = None,
             overwrite: bool = False,
             limit: Optional[int] = None,
             on_progress: Optional[Callable[[int, int, str], None]] = None,
             log: Optional[Callable[[str], None]] = None) -> int:
    """Run SR over the dirty records of each subset; save per-record ``.npy``.

    Skips records whose SR already exists unless ``overwrite``. ``limit`` caps
    the records per subset (used by tests). Returns the number of SR cubes
    written this call. The model is loaded once and reused.
    """
    emit = log or (lambda m: None)
    prog = on_progress or (lambda i, t, l: None)
    os.makedirs(sky_sr_dir(), exist_ok=True)

    # Enumerate the work first so progress has a denominator.
    work = []  # (subset, idx, record)
    for subset in subsets:
        path = tfrecord_path(records_dir, f"dirty_{subset}")
        if not os.path.exists(path):
            continue
        records = read_multiband_skyimages(path, num_images=10 ** 9)
        if limit is not None:
            records = records[:limit]
        for i, rec in enumerate(records):
            if overwrite or not os.path.isfile(sr_path(subset, i)):
                work.append((subset, i, rec))

    total = len(work)
    if total == 0:
        emit("Nothing to generate — SR already present for all records.")
        return 0

    emit(f"Loading model from {checkpoint or Config.DEFAULT_CHECKPOINT_DIR} …")
    model = load_eval_model(checkpoint=checkpoint)

    # Provenance context (best-effort): one InferenceRun per call, the model's
    # id from its checkpoint sidecar, and a single git stamp. Never fatal.
    prov = None
    try:
        store = default_store()
        model_id = model_id_of_checkpoint(checkpoint or Config.DEFAULT_CHECKPOINT_DIR)
        git = capture_git()
        run = InferenceRun(
            id=store.mint(), git=git, status="ok",
            inputs=tuple(x for x in (model_id,) if x is not None),
        )
        store.put(run)
        prov = (store, run.id, model_id, git)
    except Exception as exc:  # provenance must never block SR generation
        emit(f"[provenance] SR provenance disabled: {exc}")

    done = 0
    for subset, i, rec in work:
        cube = np.asarray(rec.data, dtype=np.float32)        # (H, W, C) LR
        _lr, sr = reconstruct(model, cube)                   # SR in electrons
        sr = np.asarray(sr, dtype=np.float32)
        if sr.ndim == 2:
            sr = sr[..., None]
        np.save(sr_path(subset, i), sr)
        if prov is not None:
            try:
                store, run_id, model_id, git = prov
                input_id = rec.prov_stamp().id if rec.prov_stamp() else None
                record_sr_cube(store, sr_path(subset, i), subset, i,
                               model_id=model_id, input_id=input_id,
                               produced_by=run_id, git=git)
            except Exception as exc:
                emit(f"[provenance] SR cube {subset} {i} not recorded: {exc}")
        done += 1
        prog(done, total, f"{subset} idx {i}")
        emit(f"SR {subset} idx {i} → {tuple(sr.shape)}")
    return done
