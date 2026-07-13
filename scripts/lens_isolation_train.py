#!/usr/bin/env python3
"""Fork selected production SR members and train lens-isolation members."""

from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.ensemble_registry import default_ensemble_dir
from euclid_polish.experiments.lens_isolation.config import (
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
    parser.add_argument("--lens-weight", type=float, default=8.0)
    parser.add_argument("--flux-weight", type=float, default=0.1)
    parser.add_argument("--crops-per-field", type=int, default=16)
    parser.add_argument("--psf-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


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
        lens_weight=args.lens_weight,
        flux_weight=args.flux_weight,
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
        "records_dir": os.path.abspath(args.records_dir),
        "out_dir": out_dir,
        "members": members,
        "steps": config.steps,
        "batch_size": config.batch_size,
        "evaluate_every": config.evaluate_every,
        "lr_peak": config.lr_peak,
        "lr_final": config.lr_final,
        "lr_warmup_steps": config.lr_warmup_steps,
        "lens_weight": config.lens_weight,
        "flux_weight": config.flux_weight,
    }
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return 0

    from euclid_polish.config import Config
    from euclid_polish.experiments.lens_isolation.datasets import (
        build_fixed_dataset,
        build_live_dataset,
    )
    from euclid_polish.experiments.lens_isolation.forward import LensIsolationForward
    from euclid_polish.experiments.lens_isolation.loss import LensIsolationLoss
    from euclid_polish.experiments.lens_isolation.records import dataset_fingerprint
    from euclid_polish.experiments.lens_isolation.training import (
        LensIsolationTrainer,
        fork_member,
    )
    from euclid_polish.observability import Reporter, ResourceSampler
    from euclid_polish.sky.observation.observation_simulator import (
        ObservationSimulator,
        ObservationSimulatorConfig,
    )
    from euclid_polish.training.forward_onthefly import member_psf_sets

    records_fingerprint = dataset_fingerprint(args.records_dir)
    reporter = Reporter.from_env()
    sampler = ResourceSampler(reporter).start()
    try:
        for index, member in enumerate(members):
            reporter.set_stage(f"lens isolation: member {index + 1}/{len(members)}")
            model = fork_member(
                member["source"],
                member["target"],
                seed=member["seed"],
                dataset_fingerprint=records_fingerprint,
            )
            psf_sets, note = member_psf_sets(
                seed=member["seed"], psf_dir=args.psf_dir or Config.EUCLID_PSF_DIR
            )
            print(f"{member['target']}: {note}")
            observation = ObservationSimulator(
                psf_sets_by_band=psf_sets,
                config=ObservationSimulatorConfig(add_noise=True, add_artifacts=True, add_saturation=True),
            )
            forward = LensIsolationForward(
                observation,
                seed=member["seed"],
                crops_per_field=args.crops_per_field,
            )
            train_ds = build_live_dataset(args.records_dir, forward, batch_size=config.batch_size)
            validate_ds = build_fixed_dataset(args.records_dir, "validate", batch_size=config.batch_size)
            trainer = LensIsolationTrainer(
                model,
                member["target"],
                steps=config.steps,
                lr_peak=config.lr_peak,
                lr_final=config.lr_final,
                lr_warmup_steps=config.lr_warmup_steps,
                loss=LensIsolationLoss(config.lens_weight, config.flux_weight),
            )

            def report(metrics, member_index=index):
                reporter.metric({**metrics, "member": member_index + 1})
                reporter.set_step(
                    member_index * config.steps + metrics["step"],
                    len(members) * config.steps,
                    f"member {member_index + 1} step {metrics['step']}",
                )

            trainer.train(
                train_ds,
                validate_ds,
                steps=config.steps,
                evaluate_every=config.evaluate_every,
                callback=report,
            )
    finally:
        sampler.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
