#!/usr/bin/env python
"""Run SR over lens-finder fields and persist sr_{subset} records (main TF env, GPU).

Decoupled from stamp cutting: this GPU step reconstructs every field once and
writes the 4-band SR field to ``sr_{subset}.tfrecord``; the CPU
``lensfinder_build_stamps`` step then crops stamps from it. Resumable — a
subset whose sr_ record already has one example per input field is skipped.
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.image.tfio import open_writer, tfrecord_path
from euclid_polish.observability.reporter import Reporter
from euclid_polish.provenance.checkpoint import model_id_of_checkpoint
from euclid_polish.provenance.defaults import default_store
from euclid_polish.provenance.gitinfo import capture_git
from euclid_polish.provenance.records import (
    Artifact,
    Format,
    Process,
    Stamp,
)


def _count_records(path: str) -> int | None:
    """Examples in a TFRecord, or None if missing/truncated (mid-write kill)."""
    if not os.path.exists(path):
        return None
    try:
        return sum(1 for _ in tf.data.TFRecordDataset(path))
    except tf.errors.DataLossError:
        return None


def _sr_complete(records_dir: str, subset: str, n_fields: int) -> bool:
    """True iff sr_{subset} already has one example per input field."""
    return _count_records(tfrecord_path(records_dir, f"sr_{subset}")) == n_fields


def run_sr_inference(records_dir: str, subset: str, sr_fn, *,
                     force: bool = False, reporter=None,
                     checkpoint: str | None = None) -> int:
    """Stream dirty_{subset} through ``sr_fn`` and write 4-band sr_{subset}.

    ``sr_fn(lr_cube_4band) -> sr_array_4band``. Returns the field count. Skips
    (without rewriting) a subset already complete unless ``force``. When
    ``checkpoint`` is given, the SR records are stamped with their (model, input)
    lineage — best-effort, never fatal."""
    in_path = tfrecord_path(records_dir, f"dirty_{subset}")
    n_fields = _count_records(in_path) or 0
    if n_fields == 0:
        return 0
    if not force and _sr_complete(records_dir, subset, n_fields):
        return n_fields
    ds = tf.data.TFRecordDataset(in_path)

    # Provenance context (best-effort): one inference run + one sr file id.
    prov = None
    try:
        store = default_store()
        model_id = model_id_of_checkpoint(checkpoint) if checkpoint else None
        git = capture_git()
        run = Process.inference(id=store.mint(), git=git, status="ok",
                                inputs=tuple(x for x in (model_id,) if x is not None))
        store.put(run)
        prov = (store, run.id, model_id, git, store.mint())
    except Exception:
        prov = None

    input_parent = None
    with open_writer(f"sr_{subset}", records_dir=records_dir) as w:
        for i, raw in enumerate(ds):
            img = Image.from_tfrecord(raw)
            sr = np.asarray(sr_fn(np.asarray(img.data, np.float32)), np.float32)
            idx = img.index if img.index is not None else i
            out = Image(
                data=sr, pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                band_names=Config.HR_TARGET_BAND_NAMES, is_clean=True,
                index=idx, subset=subset)
            if prov is not None:
                try:
                    _store, run_id, model_id, _git, sr_file_id = prov
                    in_stamp = img.prov_stamp()
                    input_id = in_stamp.id if in_stamp is not None else None
                    if input_parent is None:
                        input_parent = input_id
                    parents = tuple(x for x in (model_id, input_id) if x is not None)
                    out.stamp = Stamp(id=sr_file_id, produced_by=run_id,
                                      parents=parents, schema_version=3,
                                      subset=subset)
                except Exception:
                    pass
            w.write(out, index=idx)
            if reporter is not None:
                reporter.set_step(i + 1, n_fields, f"SR {subset} {i + 1}/{n_fields}")

    if prov is not None:
        try:
            store, run_id, model_id, git, sr_file_id = prov
            file_parents = tuple(
                x for x in (model_id, input_parent) if x is not None)
            store.put(Artifact.sr_cutout(
                id=sr_file_id, git=git, produced_by=run_id,
                format=Format.TFRECORD,
                path=tfrecord_path(records_dir, f"sr_{subset}"),
                parents=file_parents, descriptors={"subset": subset},
            ), sidecar_dir=records_dir)
        except Exception:
            pass
    return n_fields


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records-dir", required=True,
                   help="dir with dirty_{subset}.tfrecord; sr_{subset} written here")
    p.add_argument("--subset", default="", help="single subset; blank = train+validate")
    p.add_argument("--checkpoint", default=Config.DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--num-res-blocks", type=int, default=Config.DEFAULT_NUM_RES_BLOCKS)
    p.add_argument("--force", action="store_true",
                   help="regenerate sr_ records even if already complete")
    args = p.parse_args(argv)
    args.subset_all = not args.subset
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    reporter = Reporter.from_env()
    from euclid_polish.training.inference import load_model_from_checkpoint, reconstruct

    reporter.set_stage(f"loading SR model from {args.checkpoint}")
    model = load_model_from_checkpoint(
        args.checkpoint, Config.DEFAULT_REBIN_FACTOR, args.num_res_blocks,
        nchan_out=Config.NUM_HR_CHANNELS)

    def sr_fn(lr_cube):
        _, sr = reconstruct(model, lr_cube)
        return np.asarray(sr, np.float32)

    subsets = ("train", "validate") if args.subset_all else (args.subset,)
    for subset in subsets:
        reporter.set_stage(f"SR inference {subset}")
        n = run_sr_inference(args.records_dir, subset, sr_fn,
                             force=args.force, reporter=reporter,
                             checkpoint=args.checkpoint)
        print(f"  {subset}: {n} SR fields -> sr_{subset}.tfrecord")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
