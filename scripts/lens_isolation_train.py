#!/usr/bin/env python3
"""Fork selected production members and train on fixed dirty/lens records."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.ensemble_registry import default_ensemble_dir
from euclid_polish.experiments.lens_isolation.config import (
    SCHEMA_VERSION,
    ExperimentPaths,
    TrainConfig,
    assert_safe_output,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, help="comma-separated source members")
    parser.add_argument("--source-base", default=default_ensemble_dir())
    parser.add_argument("--records-dir", default=ExperimentPaths().records)
    parser.add_argument("--out-dir", default=ExperimentPaths().ensemble)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--evaluate-every", type=int, default=500)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--lr-peak", type=float, default=1e-5)
    parser.add_argument("--lr-final", type=float, default=1e-6)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--loss-norm", choices=("l1", "l2", "l3", "berhu"), default="l1")
    parser.add_argument("--noise-aug", type=float, default=0.0)
    parser.add_argument("--bootstrap", type=float, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="retrain a complete replacement member set, publishing it only after every member succeeds",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _validate_records(records_dir: str) -> None:
    from euclid_polish.experiments.lens_isolation.records import validate_split

    metadata_path = os.path.join(records_dir, "dataset.json")
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing lens-isolation dataset metadata: {metadata_path}") from error
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "incompatible lens-isolation records; regenerate with lens_isolation_generate --force"
        )
    for subset in ("train", "validate", "test"):
        split_path = os.path.join(records_dir, f"split_{subset}.json")
        try:
            with open(split_path, encoding="utf-8") as handle:
                split = json.load(handle)
        except FileNotFoundError as error:
            raise ValueError(f"missing published {subset} split metadata; regenerate the dataset") from error
        if not validate_split(
            records_dir,
            subset,
            int(split.get("count", -1)),
            config_fingerprint=split.get("config_fingerprint"),
        ):
            raise ValueError(f"incompatible or incomplete {subset} records; regenerate the dataset")


def _existing_members(out_dir: str) -> list[str]:
    try:
        return sorted(
            entry.path for entry in os.scandir(out_dir)
            if entry.name.startswith("member_") and (not entry.is_dir() or os.listdir(entry.path))
        )
    except FileNotFoundError:
        return []


def main(argv=None) -> int:
    args = parse_args(argv)
    sources = tuple(item.strip() for item in args.sources.split(",") if item.strip())
    config = TrainConfig(
        sources=sources,
        steps=args.steps,
        batch_size=args.batch_size,
        evaluate_every=args.evaluate_every,
        lr_peak=args.lr_peak,
        lr_final=args.lr_final,
        lr_warmup_steps=args.lr_warmup_steps,
        loss_norm=args.loss_norm,
        noise_aug=args.noise_aug,
        bootstrap=args.bootstrap,
        base_seed=args.base_seed,
    )
    if len(set(sources)) != len(sources):
        raise ValueError("duplicate source members are not allowed")
    out_dir = assert_safe_output(args.out_dir)
    members = []
    for index, name in enumerate(sources):
        source = os.path.abspath(os.path.join(args.source_base, name))
        if not os.path.isdir(source):
            raise FileNotFoundError(source)
        members.append(
            {
                "source": source,
                "target": os.path.join(out_dir, f"member_{index:02d}"),
                "seed": config.base_seed + index,
            }
        )
    plan = {
        "experiment": "lens_isolation",
        "schema_version": SCHEMA_VERSION,
        "records_dir": os.path.abspath(args.records_dir),
        "out_dir": out_dir,
        "members": members,
        "steps": config.steps,
        "batch_size": config.batch_size,
        "evaluate_every": config.evaluate_every,
        "lr_peak": config.lr_peak,
        "lr_final": config.lr_final,
        "lr_warmup_steps": config.lr_warmup_steps,
        "loss_norm": config.loss_norm,
        "noise_aug": config.noise_aug,
        "bootstrap": config.bootstrap,
        "force": args.force,
        "forward_onthefly": False,
    }
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return 0

    existing = _existing_members(out_dir)
    if existing and not args.force:
        names = ", ".join(os.path.basename(path) for path in existing)
        raise ValueError(
            f"lens-isolation members already exist ({names}); rerun with --force "
            "to train replacements while keeping the current members until success"
        )

    from euclid_polish.experiments.lens_isolation.records import dataset_fingerprint
    from euclid_polish.experiments.lens_isolation.training import (
        fork_member,
        publish_replacement_members,
        train_member,
    )
    from euclid_polish.observability import Reporter, ResourceSampler

    _validate_records(args.records_dir)
    records_fingerprint = dataset_fingerprint(args.records_dir)
    reporter = Reporter.from_env()
    sampler = ResourceSampler(reporter).start()
    staging_dir = None
    try:
        if args.force:
            parent = os.path.dirname(out_dir)
            os.makedirs(parent, exist_ok=True)
            staging_dir = tempfile.mkdtemp(
                prefix=f".{os.path.basename(out_dir)}-retrain-",
                dir=parent,
            )
        for index, member in enumerate(members):
            reporter.set_stage(f"lens isolation: member {index + 1}/{len(members)}")
            target = (
                os.path.join(staging_dir, os.path.basename(member["target"]))
                if staging_dir is not None
                else member["target"]
            )
            model = fork_member(
                member["source"],
                target,
                seed=member["seed"],
                dataset_fingerprint=records_fingerprint,
            )
            train_member(
                model,
                args.records_dir,
                config,
                reporter=reporter,
                member_index=index,
                member_count=len(members),
            )
        if staging_dir is not None:
            reporter.set_stage("lens isolation: publishing replacement members")
            publish_replacement_members(
                staging_dir,
                out_dir,
                tuple(os.path.basename(member["target"]) for member in members),
            )
    finally:
        sampler.stop()
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
