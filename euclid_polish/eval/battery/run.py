"""Running the battery: every test on every model, one model at a time, then
the comparison table and the figures."""
from __future__ import annotations

import os
import time
import traceback
from collections.abc import Callable, Sequence

from euclid_polish.eval.battery import (
    figures,
    flux,
    grid,
    growth,
    holes,
    outskirts,
    report,
    resolution,
    synthetic,
)
from euclid_polish.eval.battery.common import (
    Battery,
    Limits,
    ModelInfo,
    ModelRunner,
    SRCache,
    Store,
    resolve,
    score_identity,
    write_json,
)

#: The tests in run order (``--tests`` picks a subset; figures need the others' results).
TESTS = ("synthetic", "grid", "resolution", "holes", "flux", "synflux", "outskirts", "growth", "figures")


def _per_model(test: str) -> list[Callable[[Battery, ModelInfo], dict]]:
    return {"synthetic": [synthetic.run],
            "grid": [grid.run],
            "resolution": [resolution.run],
            "holes": [holes.run],
            "flux": [flux.run_kron],
            "synflux": [flux.run_synthetic],
            "outskirts": [outskirts.run_deck, outskirts.run_hosts, outskirts.run_gain],
            "growth": [growth.run, growth.run_epsf]}.get(test, [])


def summaries(bt: Battery, infos: list[ModelInfo], tests: Sequence[str]) -> dict:
    """Every selected test's summary across the models (no model is run)."""
    out: dict = {}
    makers = {"synthetic": ("synthetic", synthetic.summary), "grid": ("grid", grid.summary),
              "resolution": ("resolution", resolution.summary), "holes": ("holes", holes.summary),
              "flux": ("kron", flux.kron_summary), "synflux": ("synflux", flux.synthetic_summary),
              "outskirts": ("outskirts", outskirts.summary), "growth": ("growth", growth.summary)}
    for test in tests:
        if test in makers:
            key, make = makers[test]
            out[key] = make(bt, infos)
    out["reference_check"] = report.reference_check(out, infos)
    return out


def run(specs: Sequence[str], out_dir: str, *, tests: Sequence[str] = TESTS, cache_dir: str | None = None,
        keep_sr: bool = False, limits: Limits | None = None, rescore: bool = False, force: bool = False,
        log: Callable[[str], None] = print) -> dict:
    """Run the battery; writes ``<out>/battery.md``, ``battery.json`` and
    ``figures/``; returns the JSON payload."""
    limits = limits or Limits()
    unknown = [t for t in tests if t not in TESTS]
    if unknown:
        raise ValueError(f"unknown test(s) {unknown}; choose from {', '.join(TESTS)}")
    infos = [resolve(s) for s in specs]
    if len({i.spec for i in infos}) != len(infos):
        raise ValueError("a model is listed twice")
    runner = ModelRunner()
    cache = SRCache(cache_dir or os.path.join(out_dir, "sr_cache"), runner.sr)
    records_fp = score_identity()["records_fp"]
    bt = Battery(store=Store(out_dir, limits.tag, records_fp), cache=cache,
                 runner=runner, limits=limits, log=log, rescore=rescore, force=force)
    started = time.monotonic()
    log(f"battery: {len(infos)} model(s) → {out_dir}" + (f" [limits {limits.tag}]" if limits.tag else ""))
    for info in infos:
        log(f"• {info.name} ({info.spec}, {info.member}, {info.fingerprint})")
        for test in tests:
            for step in _per_model(test):
                step(bt, info)
        runner.release()
        if not keep_sr:
            keep = figures.keep_keys(bt, tests) if "figures" in tests else set()
            freed = bt.cache.prune(info, lambda key, keep=keep: not key.startswith("nexus-") or key in keep)
            if freed:
                log(f"  pruned {freed / 1e9:.2f} GB of NEXUS SR outputs (kept the figures' tiles)")
    summary = summaries(bt, infos, tests)
    meta = {"created": time.strftime("%Y-%m-%d %H:%M"), "models": [i.__dict__ for i in infos],
            "tests": list(tests), "limits": limits.tag, "records_fp": records_fp, "sr_cache": bt.cache.root,
            "sr_computed": bt.cache.computed, "sr_read": bt.cache.read}
    paths = {"markdown": os.path.join(out_dir, "battery.md"), "json": os.path.join(out_dir, "battery.json")}

    def write(figure_list):
        meta["figures"] = figure_list
        meta["seconds"] = round(time.monotonic() - started, 1)
        payload = {"meta": meta, "table": [list(r) for r in report.table_rows(summary, infos)],
                   "paired": [list(r) for r in report.paired_rows(summary, infos)], **summary}
        write_json(paths["json"], payload)
        with open(paths["markdown"], "w", encoding="utf-8") as fh:
            fh.write(report.markdown(summary, infos, meta))
        return payload

    payload = write([])
    if "figures" in tests:
        written, failed = [], {}
        for name, job in figures.jobs(bt, infos, summary, out_dir, tests):
            try:
                written += [p for p in job() if p]
            except Exception as exc:  # noqa: BLE001 — one figure never voids the battery's numbers
                failed[name] = f"{type(exc).__name__}: {exc}"
                log(f"  ! figure {name} failed:\n{traceback.format_exc()}")
        meta["figures_failed"] = failed
        payload = write([os.path.relpath(p, out_dir) for p in written])
    log(f"battery done in {(time.monotonic() - started) / 60:.1f} min: {paths['markdown']}")
    return payload
