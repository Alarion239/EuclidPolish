#!/usr/bin/env python
"""One-shot: archive the legacy single-model checkpoints into tracking.

Zips ``Config.DEFAULT_CHECKPOINT_DIR`` (and its ``-vis`` sibling) into the
current tracking campaign via ``TrackingStore.archive_model_zip``, logs a
campaign note (incl. the FASRC-side cleanup reminder), then deletes the local
dirs. Idempotent: already-missing dirs are skipped. Run once after the
ensemble-only refactor lands — from then on THE model is the ensemble
(a single model is an ensemble of 1: ``scripts/train_ensemble.py
--n-members 1``).
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from euclid_polish.config import Config                      # noqa: E402
from euclid_polish.tracking import default_store             # noqa: E402


def migrate() -> dict:
    store = default_store()
    if not store.has_current():
        store.create_campaign("single-model retirement")
    archived: list[str] = []
    for d, tag in ((Config.DEFAULT_CHECKPOINT_DIR, "wdsr-single-model"),
                   (Config.DEFAULT_CHECKPOINT_DIR.rstrip("/") + "-vis",
                    "wdsr-vis-single-model")):
        if not os.path.isdir(d):
            continue
        if not any(files for _dp, _dn, files in os.walk(d)):
            shutil.rmtree(d)                 # empty husk — nothing to archive
            print(f"  ✓ {d} was empty; removed")
            continue
        meta = store.archive_model_zip(
            d, tag, comment="ensemble-only migration: single model retired")
        shutil.rmtree(d)
        archived.append(meta["name"])
        print(f"  ✓ {d} → models/{meta['name']} "
              f"({meta['size_bytes'] / 1e6:.1f} MB); local dir deleted")
    if archived:
        store.append_log(
            "Ensemble-only migration: archived " + ", ".join(
                f"`models/{n}`" for n in archived)
            + ". The single-model checkpoints are retired — THE model is the "
              "ensemble now. REMINDER: the FASRC-side single-model ckpt dir "
              "(cfg.ckpt_dir, e.g. .../ckpt/wdsr) still exists remotely — "
              "remove it manually when convenient.")
    return {"archived": archived}


if __name__ == "__main__":
    out = migrate()
    if out["archived"]:
        print(f"migrated: {out['archived']}")
    else:
        print("nothing to migrate — no single-model checkpoints on disk")
