"""The model test battery (:mod:`euclid_polish.eval.battery`): every test of
the 2026-10-29/30 comparison of member_257 @ 15M with production 238 — the
synthetic score and error direction, the grid on real sky / synthetic fields /
galaxy outskirts, the JWST resolution and P8–16 texture, bright-core holes,
Kron VIS flux against MER, synthetic SR/truth flux, outskirt light injection
and the real-sky gain profile, real-galaxy curves of growth, and the figures —
on any single models, written as one comparison table (``battery.md`` +
``battery.json``) and ``figures/``.

    python scripts/model_battery.py universal:member_238 milestone:member_257@15000000 \\
        --out output/battery/257-vs-238

The first model is the baseline of every paired comparison. Run from the
repo root (it reads ``data/`` and ``ckpt/``); nothing under ``data/`` is
written. One model is loaded at a time; each model's SR outputs are cached in
``<out>/sr_cache`` (``--cache-dir`` to share one between runs) and the NEXUS
ones dropped once that model's tests are done (``--keep-sr`` keeps them;
≈2 GB per model). Results are stored per model under ``<out>/models/``: a
re-run with another model added measures only that model. About 40 min per
model on a laptop. ``--smoke`` caps every test to a few tiles / fields to try
the pipeline (numbers not comparable). ``--records-dir`` reads the synthetic
test set from another directory than the console's mirror (a newer noise
model's records, say); stored results follow the records fingerprint.

``--study NAME`` freezes the finished run as a battery study (Figures ›
Studies): the table, the paired comparisons, the report and the figures,
sealed and hashed (``euclid_polish.studies.battery``).

The outskirt test's small inputs (``euclid_polish/eval/battery/data/``: the
fitted LR kernel and the synthetic sources) are committed;
``--regenerate-outskirt-inputs DIR`` rebuilds them into DIR from the test
records, the NEXUS tiles and the MER catalogue (no model, ~20 min).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from euclid_polish.eval.battery import outskirts  # noqa: E402
from euclid_polish.eval.battery.common import Limits, use_records_dir  # noqa: E402
from euclid_polish.eval.battery.run import TESTS, run  # noqa: E402
from euclid_polish.studies import battery as battery_study  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("specs", nargs="*", help="model specs; the first is the baseline")
    p.add_argument("--out", help="output directory (results, table, figures)")
    p.add_argument("--tests", default=",".join(TESTS),
                   help=f"comma list from {', '.join(TESTS)} (default all)")
    p.add_argument("--cache-dir", default=None, help="SR cache directory (default <out>/sr_cache)")
    p.add_argument("--keep-sr", action="store_true", help="keep every NEXUS SR output in the cache")
    p.add_argument("--rescore", action="store_true",
                   help="measure the synthetic score even when the console's stored one is current")
    p.add_argument("--force", action="store_true", help="redo stored results")
    p.add_argument("--records-dir", default=None,
                   help="synthetic test records directory (default: the console's mirror)")
    p.add_argument("--study", metavar="NAME", default=None,
                   help="freeze the finished run as a battery study (Figures › Studies)")
    p.add_argument("--study-note", default="", help="the study's note")
    p.add_argument("--smoke", action="store_true",
                   help="a few tiles / fields per test (numbers not comparable)")
    p.add_argument("--regenerate-outskirt-inputs", metavar="DIR", default=None,
                   help="rebuild the outskirt test's committed inputs into DIR, then exit")
    args = p.parse_args()
    if args.regenerate_outskirt_inputs:
        for name, path in outskirts.regenerate_inputs(args.regenerate_outskirt_inputs).items():
            print(f"wrote {name}: {path}")
        return 0
    if not args.specs or not args.out:
        p.error("give one or more model specs and --out")
    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    if args.study and "figures" not in tests:
        p.error("a study needs the figures (keep 'figures' in --tests)")
    if args.records_dir:
        use_records_dir(args.records_dir)
    run(args.specs, args.out, tests=tests, cache_dir=args.cache_dir, keep_sr=args.keep_sr,
        limits=Limits.smoke() if args.smoke else Limits(), rescore=args.rescore, force=args.force,
        log=lambda msg: print(msg, flush=True))
    if args.study:
        study_id = battery_study.freeze(args.out, name=args.study, note=args.study_note)
        print(f"froze battery study {study_id} (Figures › Studies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
