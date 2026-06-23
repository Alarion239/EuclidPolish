#!/usr/bin/env python
"""Finetune Zoobot into a binary lens classifier, per reconstruction (torch env).

Trains THREE independent ``FinetuneableZoobotClassifier`` heads — one each for
the ``lr`` / ``sr`` / ``hr`` reconstruction — from the stamp catalog written by
``scripts/lensfinder_build_stamps.py``. Each head trains and is tested on its own
reconstruction (the POLISH++ fair-comparison rule). Per head it writes a
checkpoint and a test-split ``predictions.csv`` with the softmax columns
(``p_notlens``, ``p_lens``); ``scripts/lensfinder_evaluate.py`` turns those into
the TPR-vs-Einstein-radius comparison.

Runs in the EuclidPolishZoobot env (PyTorch/Lightning). Heavy imports are
deferred into ``main`` so ``--help`` and a missing env fail fast and legibly.

Usage (EuclidPolishZoobot env)::

    python scripts/lensfinder_train.py \
        --catalog data/lensfinder/stamps/catalog.csv \
        --out-dir data/lensfinder/heads --recon all --epochs 30
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.lensfinder import catalog as lf_catalog
from euclid_polish.observability.reporter import Reporter

_ENCODER = "hf_hub:mwalmsley/zoobot-encoder-convnext_nano"
_PRED_COLS = ["p_notlens", "p_lens"]


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", required=True, help="stamp catalog.csv")
    p.add_argument("--out-dir", required=True, help="output dir (per-recon subdirs)")
    p.add_argument("--recon", default="all", choices=["lr", "sr", "hr", "all"])
    p.add_argument("--encoder-name", default=_ENCODER,
                   help="HuggingFace Zoobot encoder to finetune")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--png-size", type=int, default=424)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu", "mps"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _frames(rows, recon):
    """Build per-split pandas frames (file_loc, label, id_str) for one recon."""
    import pandas as pd

    sub = lf_catalog.subset_for_recon(rows, recon)
    def _df(split):
        r = [{"id_str": x["id_str"], "file_loc": x["file_loc"],
              "label": int(float(x.get("is_lens", 0) or 0))}
             for x in sub if x.get("split") == split]
        return pd.DataFrame(r)
    return _df("train"), _df("val"), _df("test")


def _train_one(recon, rows, args, reporter):
    """Train + test one reconstruction head. Returns its metrics dict."""
    import pandas as pd
    from galaxy_datasets.transforms import (default_view_config,
                                            get_galaxy_transform,
                                            minimal_view_config)
    from galaxy_datasets.pytorch.galaxy_datamodule import CatalogDataModule
    from zoobot.pytorch.training import finetune
    from zoobot.pytorch.predictions import predict_on_catalog

    df_train, df_val, df_test = _frames(rows, recon)
    out = os.path.join(args.out_dir, recon)
    os.makedirs(out, exist_ok=True)
    reporter.set_stage(f"{recon}: {len(df_train)} train / {len(df_val)} val / "
                       f"{len(df_test)} test")

    train_cfg = default_view_config()
    train_cfg.output_size = (args.png_size, args.png_size)
    test_cfg = minimal_view_config()
    test_cfg.output_size = (args.png_size, args.png_size)
    train_tf = get_galaxy_transform(train_cfg)
    test_tf = get_galaxy_transform(test_cfg)

    dm = CatalogDataModule(
        label_cols=["label"],
        train_catalog=df_train, val_catalog=df_val, test_catalog=df_test,
        train_transform=train_tf, test_transform=test_tf,
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed)

    # Class balance is handled by the ~1:N sampling in build-stamps
    # (--neg-per-lens); zoobot's class_weights path casts the weight to the
    # label's Long dtype and crashes cross_entropy, so we don't use it.
    model = finetune.FinetuneableZoobotClassifier(
        num_classes=2, label_col="label",
        name=args.encoder_name, learning_rate=args.learning_rate)

    trainer = finetune.get_trainer(
        save_dir=out, max_epochs=args.epochs, patience=args.patience,
        devices=1, accelerator=args.device)
    trainer.fit(model, dm)

    # Predict P(lens) on the held-out test split.
    pred_csv = os.path.join(out, "predictions.csv")
    if len(df_test):
        predict_on_catalog.predict(
            catalog=df_test, model=model, label_cols=_PRED_COLS,
            inference_transform=test_tf, save_loc=pred_csv,
            datamodule_kwargs={"batch_size": args.batch_size,
                               "num_workers": args.num_workers},
            trainer_kwargs={"accelerator": args.device, "devices": 1})
    else:
        pd.DataFrame(columns=["id_str", *_PRED_COLS]).to_csv(pred_csv, index=False)
    return {"recon": recon, "n_train": len(df_train), "n_test": len(df_test),
            "predictions": pred_csv}


def main(argv=None) -> int:
    args = _parse_args(argv)
    reporter = Reporter.from_env()

    rows = lf_catalog.read_catalog(args.catalog)
    recons = (["lr", "sr", "hr"] if args.recon == "all" else [args.recon])
    print(f"catalog {args.catalog}: {len(rows)} rows; training heads {recons}")

    results = []
    for recon in recons:
        reporter.set_stage(f"training {recon} head")
        results.append(_train_one(recon, rows, args, reporter))
    for r in results:
        print(f"  ✓ {r['recon']}: {r['n_train']} train, {r['n_test']} test "
              f"→ {r['predictions']}")
    reporter.metric({"heads": len(results)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
