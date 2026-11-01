"""Sky > Compare real-tile metrics (hole %, flux ratio, enclosed-flux R) for chosen
model specs on the tiles of an existing experiment record — the 'computed' branch
of euclid_polish.web.helpers.experiments.run_experiment, run in memory: no record,
no output-store write, no member-cache write (nothing in the repo's data dirs is
touched; milestones.materialize, the specs' own loader, refreshes its ensemble dir).

Writes a record-shaped JSON (tiles, models, fingerprints, results{tile{spec{metrics}}},
summary{spec: real_metrics.aggregate}) that the realsky/holes.py summariser reads.
Optionally saves each SR (.npy) and checks the fresh SR against a stored store
output of the same spec + fingerprint (reproducibility of the old record).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime

import numpy as np

from euclid_polish.web.helpers import model_catalog, real_metrics, real_tiles
from euclid_polish.web.helpers.experiments import _compact

RECORDS = "/Users/alarion239/Desktop/EuclidPolish/data/euclid_inference/experiments/records/"


class Members:
    """In-memory model_catalog.MemberSource for one tile (no disk cache)."""

    def __init__(self, lr_input: np.ndarray, runner) -> None:
        self.lr = lr_input
        self.runner = runner
        self.memory: dict[str, np.ndarray] = {}

    def get(self, label: str) -> np.ndarray:
        if label not in self.memory:
            self.memory[label] = np.asarray(self.runner.predict(self.lr, label), np.float32)
        return self.memory[label]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--record", required=True, help="experiment record whose tiles to use")
    p.add_argument("--specs", required=True, help="comma list of model specs")
    p.add_argument("--out", required=True, help="record-shaped JSON to write")
    p.add_argument("--save-sr", default=None, help="directory for SR/LR .npy per (tile, spec)")
    args = p.parse_args()
    source = json.load(open(os.path.join(RECORDS, args.record)))
    refs = [tuple(t.split("/", 1)) for t in source["tiles"]]
    catalog = model_catalog.list_specs()
    runnable = []
    for spec in model_catalog.parse_specs(args.specs):
        item = model_catalog.resolve_spec(spec, catalog)
        if not item.available:
            raise SystemExit(f"{spec}: unavailable ({item.reason})")
        runnable.append(item)
    print("specs:", [(i.spec, i.reads, i.fingerprint) for i in runnable], flush=True)
    runner = model_catalog.EnsembleMemberRunner(labels=model_catalog.needed_members(runnable))
    record = {"version": 1, "id": os.path.basename(args.out), "label": f"re-run of {args.record} tiles",
              "created": datetime.now(UTC).isoformat(), "source_record": args.record,
              "tiles": [f"{s}/{i}" for s, i in refs], "models": [i.spec for i in runnable],
              "fingerprints": {i.spec: i.fingerprint for i in runnable},
              "model_labels": {i.spec: i.label for i in runnable},
              "definitions": real_metrics.definitions(), "results": {}, "summary": {},
              "store_check": {}}
    per_spec: dict[str, list] = {i.spec: [] for i in runnable}
    started = time.monotonic()
    for n, (src, ident) in enumerate(refs, 1):
        entry = real_tiles.get_entry(src, ident)
        tile = real_tiles.get_tile(src, ident, entry=entry)
        lr = tile.lr_e
        finite = np.isfinite(lr)
        lr_input = np.where(finite, lr, 0.0).astype(np.float32)
        members = Members(lr_input, runner)
        stored = real_tiles.tile_outputs(entry, {i.spec: i.fingerprint for i in runnable})
        tile_results = {}
        for item in runnable:
            t0 = time.monotonic()
            sr = model_catalog.predict(item, lr_input, members)
            metrics = real_metrics.tile_metrics(lr, sr)
            per_spec[item.spec].append(metrics)
            tile_results[item.spec] = {"state": "computed", "fingerprint": item.fingerprint,
                                       "metrics": _compact(metrics)}
            meta = stored.get(item.spec)
            if meta is not None and meta.get("fingerprint") == item.fingerprint:
                old, _hdr, _m = real_tiles.load_output(entry, item.spec, current=item.fingerprint)
                diff = np.abs(old.astype(np.float64) - sr)
                scale = float(np.nanpercentile(np.abs(old), 99.9)) or 1.0
                old_metrics = real_metrics.tile_metrics(lr, old)
                record["store_check"].setdefault(entry.ref, {})[item.spec] = {
                    "stored_file": meta.get("file"), "max_abs_diff_e": float(np.nanmax(diff)),
                    "max_diff_over_p99.9": float(np.nanmax(diff) / scale),
                    "stored_hole_pct_mean": old_metrics["summary"]["hole_pct_mean"],
                    "fresh_hole_pct_mean": metrics["summary"]["hole_pct_mean"]}
            if args.save_sr:
                os.makedirs(args.save_sr, exist_ok=True)
                stem = f"{src}-{ident}"
                np.save(os.path.join(args.save_sr, f"{stem}__{model_catalog.spec_slug(item.spec)}.npy"),
                        np.asarray(sr, np.float32))
                lr_path = os.path.join(args.save_sr, f"{stem}__lr.npy")
                if not os.path.isfile(lr_path):
                    np.save(lr_path, np.asarray(lr, np.float32))
            s = metrics["summary"]
            print(f"[{n}/{len(refs)}] {entry.ref} {item.spec}: hole mean {s['hole_pct_mean']:.2f}% "
                  f"per band {[round(v['hole_pct'], 2) for v in metrics['per_band'].values()]} "
                  f"flux {[round(v['flux_ratio'], 3) for v in metrics['per_band'].values()]} "
                  f"({time.monotonic() - t0:.1f} s)", flush=True)
        record["results"][entry.ref] = tile_results
        record["summary"] = {spec: real_metrics.aggregate(items) for spec, items in per_spec.items() if items}
        with open(args.out + ".tmp", "w") as fh:
            json.dump(record, fh, indent=1, allow_nan=False)
        os.replace(args.out + ".tmp", args.out)
    record["finished"] = datetime.now(UTC).isoformat()
    record["duration_s"] = round(time.monotonic() - started, 1)
    with open(args.out + ".tmp", "w") as fh:
        json.dump(record, fh, indent=1, allow_nan=False)
    os.replace(args.out + ".tmp", args.out)
    print("store check:", json.dumps(record["store_check"], indent=1), flush=True)


if __name__ == "__main__":
    main()
