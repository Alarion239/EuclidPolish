"""Test 4 — holes in bright cores on real sky: Sky › Compare's real-tile
metrics (:mod:`euclid_polish.web.helpers.real_metrics`) on the tiles of two
saved experiment records — 13 NEXUS galaxy tiles and the poster galaxy — run
in memory from the SR cache (no record, output store or member cache written).

A hole is an SR pixel below half the LR flux per SR pixel inside the tile's
brightest 1 % of LR pixels; the headline is the mean over bands and tiles of
the hole % (``real_metrics.aggregate``'s ``hole_pct_mean``).
"""
from __future__ import annotations

from euclid_polish.eval.battery.common import Battery, ModelInfo, experiment_tiles, real_tile_input, tile_key
from euclid_polish.web.helpers import real_metrics
from euclid_polish.web.helpers.experiments import _compact

STAGE = "holes"
#: The Sky › Compare records whose tiles the test uses (2026-10-28).
RECORDS = {"nexus": "20261028-134110-710beb.json", "poster": "20261028-134110-f0d0bb.json"}
POSTER = "poster/target_181255"


def run(bt: Battery, info: ModelInfo) -> dict:
    def compute():
        out = {}
        for name, record in RECORDS.items():
            per_tile, items = {}, []
            for ref in experiment_tiles(record):
                lr, lr_input = real_tile_input(ref)
                sr = bt.cache.get(info, tile_key(ref), lr_input)
                metrics = real_metrics.tile_metrics(lr, sr)
                items.append(metrics)
                per_tile[ref] = _compact(metrics)
            out[name] = {"record": record, "tiles": per_tile, "summary": real_metrics.aggregate(items)}
        return out, None

    return bt.stage(info, STAGE, compute)


def figure_keys() -> set[str]:
    """The SR cache keys of the test's tiles (kept when the cache is pruned)."""
    return {tile_key(ref) for record in RECORDS.values() for ref in experiment_tiles(record)}


def summary(bt: Battery, infos: list[ModelInfo]) -> dict:
    out: dict = {"models": {}}
    for info in infos:
        res = bt.store.read(info, STAGE)
        if res is None:
            continue
        row = {}
        for name in RECORDS:
            s = res[name]["summary"]
            row[name] = {"hole_pct_mean": s["summary"].get("hole_pct_mean"), "n_tiles": s.get("n_tiles"),
                         "per_band": {b: v.get("hole_pct") for b, v in s["per_band"].items()},
                         "flux_ratio": {b: v.get("flux_ratio") for b, v in s["per_band"].items()}}
        out["models"][info.spec] = row
    return out
