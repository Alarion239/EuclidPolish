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
        --out-dir data/lensfinder/heads --recon all --epochs 10
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
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--png-size", type=int, default=424)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--training-mode", default="head_only",
                   choices=["head_only", "full"],
                   help="head_only = freeze encoder, train only the linear head "
                        "(fast, clean probe); full = fine-tune all encoder params")
    p.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu", "mps"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="retrain every head from scratch, ignoring existing "
                        "predictions.csv / checkpoints (default: resume — skip "
                        "heads already trained, continue partial ones)")
    return p.parse_args(argv)


def _latest_checkpoint(out: str):
    """Newest .ckpt to resume a partially-trained head from, or None.

    Prefers ``last.ckpt`` (a rolling per-epoch snapshot), else the most recently
    written best-epoch checkpoint."""
    ckpt_dir = os.path.join(out, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    last = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last):
        return last
    ckpts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)
             if f.endswith(".ckpt")]
    return max(ckpts, key=os.path.getmtime) if ckpts else None


def _checkpoint_epoch(ckpt_path):
    """The training epoch a Lightning checkpoint was saved at, or None.

    Prefer the filename (``'{epoch}.ckpt'``, possibly ``'{epoch}-vN.ckpt'``);
    fall back to the ``epoch`` key in the checkpoint payload (e.g. last.ckpt)."""
    if not ckpt_path:
        return None
    stem = os.path.basename(ckpt_path)
    if stem.endswith(".ckpt"):
        stem = stem[:-len(".ckpt")]
    head = stem.split("-")[0]
    if head.isdigit():
        return int(head)
    try:
        import torch
        return int(torch.load(ckpt_path, map_location="cpu",
                              weights_only=False).get("epoch"))
    except Exception:
        return None


def _write_predictions(df_test, model, test_tf, pred_csv, args):
    """Write the test-split P(lens) predictions CSV (or an empty one)."""
    import pandas as pd
    from zoobot.pytorch.predictions import predict_on_catalog

    if len(df_test):
        predict_on_catalog.predict(
            catalog=df_test, model=model, label_cols=_PRED_COLS,
            inference_transform=test_tf, save_loc=pred_csv,
            datamodule_kwargs={"batch_size": args.batch_size,
                               "num_workers": args.num_workers},
            trainer_kwargs={"accelerator": args.device, "devices": 1})
    else:
        pd.DataFrame(columns=["id_str", *_PRED_COLS]).to_csv(pred_csv, index=False)


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


def _make_reporter_bridge(L, reporter, recon, max_epochs, step_offset):
    """A Lightning callback that forwards loss to the FASRC events stream.

    Zoobot/Lightning log loss to their own logger; the WebUI loss plot reads
    ``reporter.metric`` events instead — so on each validation epoch we pull the
    train/val loss out of ``trainer.callback_metrics`` and emit one metric
    sample (``step`` is offset per head so the three heads form one monotonic
    curve) plus a per-epoch progress tick. Hooked on ``on_train_epoch_end`` —
    which fires after validation, so both the aggregated train loss and the val
    loss are present, and the pre-training sanity-check pass is skipped.
    """
    class _ReporterBridge(L.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            cm = trainer.callback_metrics

            def _pick(is_val):
                for k, v in cm.items():
                    kl = k.lower()
                    if "loss" in kl and (("val" in kl) == is_val):
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            return None
                return None

            tl, vl = _pick(False), _pick(True)
            row = {"step": step_offset + int(trainer.global_step),
                   "recon": recon, "epoch": int(trainer.current_epoch)}
            if tl is not None:
                row["loss"] = tl
            if vl is not None:
                row["val_loss"] = vl
            if tl is not None or vl is not None:     # only emit real loss points
                reporter.metric(row)
            reporter.set_step(int(trainer.current_epoch) + 1, max_epochs,
                              f"{recon} epoch")

    return _ReporterBridge()


def _train_one(recon, rows, args, reporter, step_offset):
    """Train + test one reconstruction head. Returns its metrics dict.

    Resumable: a head whose ``predictions.csv`` already exists is skipped; a
    head with checkpoints but no predictions is resumed from its latest
    checkpoint (Lightning restores weights + optimizer + epoch + early-stop
    state). ``--force`` retrains from scratch.
    """
    out = os.path.join(args.out_dir, recon)
    os.makedirs(out, exist_ok=True)
    pred_csv = os.path.join(out, "predictions.csv")
    if not args.force and os.path.exists(pred_csv):
        reporter.set_stage(f"{recon}: already trained — skipping")
        print(f"  ⏭  {recon}: predictions.csv exists — skipping "
              f"(use --force to retrain)")
        return {"recon": recon, "n_train": 0, "n_test": 0,
                "predictions": pred_csv, "global_step": 0, "skipped": True}

    import lightning as L
    from galaxy_datasets.pytorch.galaxy_datamodule import CatalogDataModule
    from galaxy_datasets.transforms import (
        default_view_config,
        get_galaxy_transform,
        minimal_view_config,
    )
    from lightning.pytorch.callbacks import ModelCheckpoint
    from zoobot.pytorch.training import finetune

    df_train, df_val, df_test = _frames(rows, recon)
    reporter.set_stage(f"{recon}: {len(df_train)} train / {len(df_val)} val / "
                       f"{len(df_test)} test")

    test_cfg = minimal_view_config()
    test_cfg.output_size = (args.png_size, args.png_size)
    test_tf = get_galaxy_transform(test_cfg)

    # A checkpoint at/past the current ceiling can't be resumed (Lightning
    # rejects current_epoch >= max_epochs). Treat it as trained: load it and
    # just (re)write the predictions the evaluation needs.
    ckpt = None if args.force else _latest_checkpoint(out)
    ckpt_epoch = _checkpoint_epoch(ckpt)
    if ckpt and ckpt_epoch is not None and ckpt_epoch >= int(args.epochs):
        reporter.set_stage(f"{recon}: ckpt epoch {ckpt_epoch} >= max_epochs "
                           f"{args.epochs} — predicting only")
        print(f"  ⏭  {recon}: checkpoint epoch {ckpt_epoch} ≥ max_epochs "
              f"{args.epochs} — treating as trained, writing predictions only")
        model = finetune.FinetuneableZoobotClassifier.load_from_checkpoint(ckpt)
        _write_predictions(df_test, model, test_tf, pred_csv, args)
        return {"recon": recon, "n_train": len(df_train), "n_test": len(df_test),
                "predictions": pred_csv, "global_step": 0, "skipped": True}

    train_cfg = default_view_config()
    train_cfg.output_size = (args.png_size, args.png_size)
    train_tf = get_galaxy_transform(train_cfg)

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
        name=args.encoder_name, learning_rate=args.learning_rate,
        training_mode=args.training_mode)

    trainer = finetune.get_trainer(
        save_dir=out, max_epochs=args.epochs, patience=args.patience,
        devices=1, accelerator=args.device)
    # Save a rolling last.ckpt each epoch so a re-submit resumes with no redo.
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            cb.save_last = True
    trainer.callbacks.append(
        _make_reporter_bridge(L, reporter, recon, args.epochs, step_offset))
    if ckpt:
        reporter.set_stage(f"{recon}: resuming from {os.path.basename(ckpt)}")
        print(f"  ↻ {recon}: resuming from {os.path.basename(ckpt)}")
    trainer.fit(model, dm, ckpt_path=ckpt)

    # Predict P(lens) on the held-out test split.
    _write_predictions(df_test, model, test_tf, pred_csv, args)
    return {"recon": recon, "n_train": len(df_train), "n_test": len(df_test),
            "predictions": pred_csv, "global_step": int(trainer.global_step)}


def main(argv=None) -> int:
    args = _parse_args(argv)
    reporter = Reporter.from_env()

    import torch
    # A100/H100 Tensor Cores: TF32 matmuls — a large speedup with negligible
    # precision cost for this classifier (also silences Lightning's warning).
    torch.set_float32_matmul_precision("high")

    rows = lf_catalog.read_catalog(args.catalog)
    recons = (["lr", "sr", "hr"] if args.recon == "all" else [args.recon])
    print(f"catalog {args.catalog}: {len(rows)} rows; training heads {recons}")

    results = []
    step_offset = 0
    for recon in recons:
        reporter.set_stage(f"training {recon} head")
        res = _train_one(recon, rows, args, reporter, step_offset)
        step_offset += res["global_step"]        # heads share one monotonic axis
        results.append(res)
    for r in results:
        tag = " (skipped — already trained)" if r.get("skipped") else ""
        print(f"  ✓ {r['recon']}: {r['n_train']} train, {r['n_test']} test "
              f"→ {r['predictions']}{tag}")
    reporter.metric({"heads": len(results)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
