#!/usr/bin/env python
"""Zoobot morphology before/after evaluation for super-resolution outputs.

Runs the Galaxy-Zoo deep-learning model `Zoobot
<https://github.com/mwalmsley/zoobot>`_ (Walmsley et al.; already validated on
Euclid VIS — Euclid prep. XLIII) on the VIS plane of an evaluation run's
outputs and scores how the model's morphology vector moves from the dirty LR
input (**before**) to the super-resolved image (**after**). When an HR ground
truth is present (synthetic targets), it also measures whether SR moves the
vector *toward* HR — the cleanest check that SR improves morphology recovery.

This runs **locally** (not on FASRC): you pull an eval run's FITS down with the
"Sync from FASRC" button, then score morphology on your own machine. Zoobot is
PyTorch and clashes with this repo's TensorFlow stack, so it runs in the
isolated env from ``environment-zoobot.yml`` — the WebUI "Run Zoobot" button
shells out to that env's Python for you, or run this directly (below). The pure
image-prep / metric helpers live in ``euclid_polish.eval.zoobot_morph`` and are
tested in the main env.

Two model modes:

* **representation** (default) — uses a published Zoobot encoder
  (``hf_hub:mwalmsley/zoobot-encoder-convnext_nano``) and compares the encoder's
  representation embeddings. Works off-the-shelf, no finetuning or GZ head.
* **votes** — pass ``--tree-checkpoint <ckpt>`` for a finetuned
  ``FinetuneableZoobotTree``; compares Galaxy-Zoo vote fractions named by a
  ``zoobot.shared.schemas`` schema (``--schema``).

Usage (locally, in the EuclidPolishZoobot env; ``auto`` picks MPS/CUDA/CPU)::

    conda run -n EuclidPolishZoobot python scripts/zoobot_morphology.py \
        --run-dir data/eval_results/lenses
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.eval import zoobot_morph as zm
from euclid_polish.observability.reporter import Reporter

# The three rendered views per object, in a stable order.
_VIEWS = ("before", "after", "hr")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-dir", default=None,
                   help="eval run directory (per-object SR.fits / "
                        "original_stack.fits). Default: "
                        "Config.EVAL_RESULTS_DIR/<run-name>")
    p.add_argument("--run-name", default="lenses",
                   help="run sub-directory under Config.EVAL_RESULTS_DIR when "
                        "--run-dir is not given")
    p.add_argument("--model-name",
                   default="hf_hub:mwalmsley/zoobot-encoder-convnext_nano",
                   help="HuggingFace Zoobot encoder (representation mode)")
    p.add_argument("--tree-checkpoint", default=None,
                   help="FinetuneableZoobotTree checkpoint → vote-fraction mode")
    p.add_argument("--schema", default="gz_evo_v1_public",
                   help="zoobot.shared.schemas schema name (vote mode); names "
                        "the output vote-fraction columns")
    p.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu", "mps"],
                   help="Lightning accelerator (auto picks MPS/CUDA/CPU)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--png-size", type=int, default=424,
                   help="rendered PNG size fed to Zoobot")
    p.add_argument("--asinh-scale", type=float, default=None,
                   help="asinh knee for the PNG render (default: "
                        "Config.STRETCH_SCALE_E)")
    return p.parse_args(argv)


def _build_catalog(objects, run_dir, asinh_scale, png_size, reporter):
    """Render before/after/hr PNGs for every object → a Zoobot catalog frame.

    Returns ``(catalog_df, index)`` where ``index`` maps ``id_str`` →
    ``(object_id, view)`` so predictions can be regrouped per object.
    """
    import pandas as pd

    rows = []
    index = {}
    n = len(objects)
    for i, obj in enumerate(objects):
        reporter.set_step(i, n, f"render {obj['id']}")
        png_dir = os.path.join(obj["dir"], "zoobot")
        for view in _VIEWS:
            src = obj.get(view)
            if not src:
                continue
            png = os.path.join(png_dir, f"{view}.png")
            zm.render_vis_png(src, png, asinh_scale=asinh_scale, size=png_size)
            id_str = f"{obj['id']}__{view}"
            rows.append({"id_str": id_str, "file_loc": png})
            index[id_str] = (obj["id"], view)
    return pd.DataFrame(rows), index


def _load_model_and_labels(args):
    """Load the Zoobot model and the prediction column names for the mode."""
    from zoobot.pytorch.training import representations
    from zoobot.pytorch.estimators import define_model

    if args.tree_checkpoint:
        # Vote-fraction mode: finetuned tree model + schema-named columns.
        from zoobot.pytorch.training import finetune
        from zoobot.shared import schemas

        model = finetune.FinetuneableZoobotTree.load_from_checkpoint(
            args.tree_checkpoint)
        schema = getattr(schemas, args.schema, None) \
            or getattr(schemas, f"{args.schema}_schema")
        label_cols = schema.label_cols
        mode = "votes"
    else:
        # Representation mode: published encoder + feat_N embedding columns.
        model = representations.ZoobotEncoder.load_from_name(args.model_name)
        encoder_dim = define_model.get_encoder_dim(model.encoder)
        label_cols = [f"feat_{k}" for k in range(encoder_dim)]
        mode = "representation"
    return model, label_cols, mode


def main(argv=None) -> int:
    args = _parse_args(argv)
    reporter = Reporter.from_env()

    run_dir = args.run_dir or os.path.join(
        Config.EVAL_RESULTS_DIR, args.run_name)
    asinh_scale = args.asinh_scale or Config.STRETCH_SCALE_E

    objects = zm.discover_objects(run_dir)
    print(f"run {run_dir}: {len(objects)} object(s)")
    if not objects:
        print("no objects with SR.fits found")
        return 0

    # Heavy imports (torch / zoobot / galaxy-datasets) are deferred to here so
    # --help and a missing env fail fast and legibly.
    from galaxy_datasets.transforms import (default_view_config,
                                            get_galaxy_transform)
    from zoobot.pytorch.predictions import predict_on_catalog

    reporter.set_stage("rendering PNGs")
    catalog, index = _build_catalog(
        objects, run_dir, asinh_scale, args.png_size, reporter)

    reporter.set_stage("loading Zoobot model")
    model, label_cols, mode = _load_model_and_labels(args)
    print(f"mode={mode}, {len(label_cols)} output columns, "
          f"{len(catalog)} images")

    cfg = default_view_config()
    cfg.output_size = (args.png_size, args.png_size)
    transform = get_galaxy_transform(cfg)

    reporter.set_stage("Zoobot inference")
    preds = predict_on_catalog.predict(
        catalog=catalog,
        model=model,
        label_cols=label_cols,
        inference_transform=transform,
        save_loc=os.path.join(run_dir, "zoobot_predictions.csv"),
        datamodule_kwargs={"batch_size": args.batch_size,
                           "num_workers": args.num_workers},
        trainer_kwargs={"accelerator": args.device, "devices": 1},
    )

    # Regroup per-image prediction vectors back to per-object before/after/hr.
    import numpy as np

    vectors = {}  # object_id -> {view: vector}
    for _, prow in preds.iterrows():
        obj_id, view = index[str(prow["id_str"])]
        vectors.setdefault(obj_id, {})[view] = \
            np.asarray([prow[c] for c in label_cols], dtype=np.float64)

    reporter.set_stage("scoring morphology deltas")
    rows = []
    for obj in objects:
        v = vectors.get(obj["id"], {})
        if "before" not in v or "after" not in v:
            continue
        d = zm.vector_deltas(v["before"], v["after"], ref=v.get("hr"))
        d["id"] = obj["id"]
        d["mode"] = mode
        d["has_hr"] = "hr" in v
        rows.append(d)

    out_csv = os.path.join(run_dir, "morphology_manifest.csv")
    zm.write_morph_manifest(out_csv, rows)
    n_closer = sum(1 for r in rows if r.get("closer_to_ref"))
    print(f"\n✓ scored {len(rows)} object(s) → {out_csv}")
    if any(r.get("has_hr") for r in rows):
        print(f"  SR moved morphology toward HR for {n_closer}/{len(rows)}")
    reporter.metric({"objects": len(rows), "closer_to_hr": n_closer})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
