"""Measure a long run's milestone checkpoints as they land
(``euclid_polish/web/helpers/milestones.py``): pull new milestones from
FASRC, score each on the universal scores' test fields (the synthetic
knee-PSNR curves of Models › Universal) and measure its four-band JWST
resolution (``scripts/jwst_resolution.py``, spec ``milestone:<member>@<N>``),
oldest first, until ``--last`` is done (or ``--max-hours`` pass). Each new
milestone is also run on the poster galaxy, in a fresh Sky › Compare
comparison of every milestone so far beside ``--poster-refs`` (outputs already
in the store are reused), so the Compare tab opens on the latest one.

    python scripts/process_milestones.py --member member_257 --watch --last 15000000 \\
        --res-dir data/jwst_euclid_overlap/milestones

Run from the checkout whose ``data/`` it should use. A milestone counts as
done once it has a synthetic score and a resolution in all four band
reports, so a restart picks up where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from euclid_polish.web import remote
from euclid_polish.web.helpers import experiments, milestones, real_tiles

BANDS = ("H_E", "J_E", "Y_E", "VIS")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _PrintCap:
    def tick(self, done, total, label=""):
        if done == total or done % 25 == 0:
            print(f"    {done}/{total} {label}", flush=True)

    def check_cancelled(self):
        return None


def say(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M"), message, flush=True)


def _report(res_dir: str, band: str) -> dict:
    try:
        with open(os.path.join(res_dir, f"resolution_{band}.json"), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {"models": {}}


def done_steps(member: str, res_dir: str) -> set[int]:
    scored = {int(s) for s in milestones.read_scores(member).get("steps", {})}
    measured = []
    for band in BANDS:
        keys = [k for k in _report(res_dir, band)["models"] if k.startswith("milestone:")]
        measured.append({step for name, step in map(milestones.parse_spec, keys) if name == member})
    return scored.intersection(*measured)


def process(member: str, step: int, res_dir: str, cache: str) -> None:
    say(f"{member} @ {step:,}: scoring the synthetic test fields")
    entry = milestones.score(_PrintCap(), member, step)
    say(f"  ∫PSNR {entry['mean']:.3f} dB ({' / '.join(f'{v:.2f}' for v in entry['integrated'])})")
    spec = milestones.spec_for(member, step)
    os.makedirs(res_dir, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": ROOT, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    for band in BANDS:
        with open(os.path.join(res_dir, f"resolution_{band}.log"), "a", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", os.path.join(ROOT, "scripts", "jwst_resolution.py"),
                            "--band", band, "--models", spec, "--cache", cache,
                            "--out", os.path.join(res_dir, f"resolution_{band}.json")],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        got = _report(res_dir, band)["models"].get(spec, {}).get("asinh_euclid", {})
        arcsec = (got.get("resolution_coherence") or {}).get("arcsec")
        say(f"  {band}: ½-coherence (Euclid-noise stretch) {arcsec}″")


def poster_comparison(member: str, refs: list[str], res_dir: str) -> None:
    """Every measured milestone of ``member`` (and ``refs``) on the poster
    galaxy, as one comparison; only outputs not in the store are computed."""
    steps = sorted(done_steps(member, res_dir))
    specs = [*refs, *(milestones.spec_for(member, step) for step in steps)]
    tiles = [(entry.source, entry.id) for entry in real_tiles.list_entries("poster")]
    say(f"  poster: {len(tiles)} tile(s) × {len(specs)} model(s)")
    last = steps[-1] if steps else 0
    record = experiments.run_experiment(
        tiles, specs, label=f"Poster galaxy · {member} milestones to {last:,} steps", log=lambda _m: None)
    counts = record.get("counts", {})
    errors = f" errors: {record['errors']}" if record.get("errors") else ""
    say(f"  poster: {record['id']} {record['status']} (computed {counts.get('outputs_computed')}, "
        f"reused {counts.get('outputs_reused')}){errors}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--member", required=True, help="member_NNN")
    p.add_argument("--res-dir", default="data/jwst_euclid_overlap/milestones",
                   help="where the per-band resolution reports go (merged into)")
    p.add_argument("--cache", default="data/jwst_euclid_overlap/resolution_cache")
    p.add_argument("--watch", action="store_true", help="keep polling FASRC for new milestones")
    p.add_argument("--last", type=int, default=None, help="stop once this milestone step is done")
    p.add_argument("--poll", type=int, default=900, help="seconds between polls (--watch)")
    p.add_argument("--max-hours", type=float, default=80.0)
    p.add_argument("--poster-refs", default="universal:member_249,universal:member_218,universal:member_227",
                   help="models shown beside the milestones in the poster comparison ('' for none)")
    p.add_argument("--no-poster", action="store_true", help="skip the poster comparison")
    args = p.parse_args()
    refs = [spec.strip() for spec in args.poster_refs.split(",") if spec.strip()]
    started = time.time()
    while True:
        try:
            remote.connect_from_config()
            new = milestones.pull(args.member)
            if new:
                say(f"pulled milestones {', '.join(f'{s:,}' for s in new)}")
        except Exception as exc:  # noqa: BLE001 — a dropped connection: try again next poll
            say(f"pull failed ({type(exc).__name__}: {exc}); using what is local")
        done = done_steps(args.member, args.res_dir)
        todo = [s for s in milestones.local_steps(args.member) if s not in done]
        for step in todo:
            try:
                process(args.member, step, args.res_dir, args.cache)
            except Exception as exc:  # noqa: BLE001 — one milestone never stops the rest
                say(f"{args.member} @ {step:,} failed: {type(exc).__name__}: {exc}")
        if todo and not args.no_poster:
            try:
                poster_comparison(args.member, refs, args.res_dir)
            except Exception as exc:  # noqa: BLE001 — the poster never stops the measurements
                say(f"poster comparison failed: {type(exc).__name__}: {exc}")
        finished = args.last is not None and args.last in done_steps(args.member, args.res_dir)
        if finished or not args.watch:
            say("done" if finished else "processed what was there")
            return 0
        if time.time() - started > args.max_hours * 3600:
            say(f"gave up after {args.max_hours} h")
            return 1
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
