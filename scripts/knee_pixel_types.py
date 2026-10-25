"""Pixel types by knee use from the command line (the console's Models › Knees
› Pixel types does the same as local jobs): run a knee attribution (no
ablation), analyse a run, or compare two models on the same pixels. The
analysis, figures and units are
:mod:`euclid_polish.web.helpers.knee_pixel_types_report`.

    python scripts/knee_pixel_types.py --models universal:member_249 --synthetic 30 \\
        --tiles poster/target_181255,nexus/f200w-0100 --per-band 1024 --per-class 48

    python scripts/knee_pixel_types.py --run-id 20261025-120000-abcdef

    python scripts/knee_pixel_types.py --compare-a "<run id>|universal:member_245" \\
        --compare-b "<run id>|universal:member_244" --tag floor

Run from the checkout whose ``data/`` and ``ckpt/`` it should use. Sampling:
``mixed`` (default) pools a uniform sample (``--per-band`` pixels per band)
with a class-stratified one (``--per-class``), weights halved: the weights
count field pixels without bias, while the stratified half keeps rare bright
pixels sampled. Two runs with the same ``--seed`` sample the same uniform
pixels of a field, so their models can be compared pixel by pixel.
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
from euclid_polish.web.helpers import knee_attribution_runs as runs  # noqa: E402
from euclid_polish.web.helpers import knee_pixel_types_report as report  # noqa: E402


class _PrintCap:
    """The job's progress hooks, printed."""

    def __init__(self):
        self.start = time.time()

    def tick(self, done, total, label=""):
        print(f"[{time.time() - self.start:7.1f}s] {done}/{total} {label}", flush=True)

    def check_cancelled(self):
        return None


def run_attribution(args) -> str:
    form = {"models": args.models, "synthetic": str(args.synthetic), "tiles": args.tiles,
            "per_class": str(args.per_class), "seed": str(args.seed), "ablation": "0",
            "sampling": args.sampling, "per_band": str(args.per_band)}
    params = runs.parse_request(form)
    run_id = runs.new_run_id()
    print(f"run {run_id}: {params}", flush=True)
    result = runs.job_knee_attribution(_PrintCap(), run_id=run_id, **params)
    print(f"run {run_id}: {result['status']} errors={result['errors']}", flush=True)
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", help="analyse this run (else run a new attribution first)")
    parser.add_argument("--models", default="universal:member_249")
    parser.add_argument("--synthetic", type=int, default=30)
    parser.add_argument("--tiles", default="poster/target_181255")
    parser.add_argument("--sampling", default="mixed", choices=runs.SAMPLINGS)
    parser.add_argument("--per-band", type=int, default=1024)
    parser.add_argument("--per-class", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--types", type=int, default=report.DEFAULT_TYPES, help="k of the knee types")
    parser.add_argument("--only", help="analyse only this model spec of the run")
    parser.add_argument("--compare-a", help="'<run id>|<model spec>': the reference model")
    parser.add_argument("--compare-b", help="'<run id>|<model spec>': the model compared with it")
    parser.add_argument("--tag", default="", help="suffix of the comparison's folder")
    parser.add_argument("--overview", help="'<run id>|<spec>,…': up to eight models' knee curves "
                                           "on one figure (pixel_types/overview_<tag>/)")
    args = parser.parse_args()

    def say(label: str) -> None:
        print(f"  {label}", flush=True)

    if args.overview:
        refs = [tuple(item.strip().split("|", 1)) for item in args.overview.split(",") if item.strip()]
        out = report.overview(refs, args.tag or "models", progress=say)
        print(f"overview → {out}", flush=True)
        return
    if args.compare_a or args.compare_b:
        params = report.parse_compare_request({"a": args.compare_a, "b": args.compare_b, "tag": args.tag})
        out = report.compare_models(params["a"], params["b"], tag=params["tag"], progress=say)
        print(f"compare → {out}", flush=True)
        return
    run_id = args.run_id or run_attribution(args)
    folders = report.analyse_run(run_id, [args.only] if args.only else None, k=args.types,
                                 seed=args.seed, progress=say)
    for folder in folders:
        print(f"→ {folder}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
