"""Reduce-LR-on-plateau decision logic (pure Python; no TF)."""

from __future__ import annotations

from euclid_polish.training.plateau import PlateauLRReducer


def _feed(reducer, series):
    """Feed (step, metric) pairs; return the steps where a cut fired."""
    return [step for step, metric in series if reducer.should_reduce(step, metric)]


def test_fires_after_patience_of_no_progress_min_mode():
    r = PlateauLRReducer(mode="min", patience=5000, min_delta=1e-4, cooldown=2000)
    # Improve once, then sit flat. The cut fires patience steps after the last
    # improvement, not after the first flat eval.
    fires = _feed(r, [(1000, 0.010), (2000, 0.006)]        # improving
                     + [(s, 0.006) for s in range(3000, 11001, 1000)])
    assert fires == [7000]                                 # 2000 (best) + 5000


def test_micro_creep_below_min_delta_does_not_reset_patience():
    r = PlateauLRReducer(mode="min", patience=4000, min_delta=1e-4, cooldown=1000)
    # Loss drifts down by 1e-5 per eval — smaller than min_delta, so it counts
    # as no-progress and the guard still fires.
    series = [(1000, 0.00600)]
    series += [(s, 0.00600 - 1e-5 * ((s - 1000) // 1000))
               for s in range(2000, 9001, 1000)]
    fires = _feed(r, series)
    assert fires and fires[0] == 5000                      # 1000 + 4000


def test_real_improvement_resets_patience():
    r = PlateauLRReducer(mode="min", patience=3000, min_delta=1e-4, cooldown=0)
    # A genuine drop at step 3000 pushes the deadline out; no fire until 6000.
    series = [(1000, 0.010), (2000, 0.010), (3000, 0.005)]  # real improvement
    series += [(s, 0.005) for s in range(4000, 6001, 1000)]
    assert _feed(r, series) == [6000]


def test_cooldown_prevents_immediate_refire():
    r = PlateauLRReducer(mode="min", patience=2000, min_delta=1e-4, cooldown=3000)
    # Flat forever. First fire at 3000 (1000+2000). Cooldown holds off until
    # 3000+3000, then patience again → next fire at 6000+2000? cooldown re-anchors
    # best_step, so the next fire is patience after the cooldown clears.
    series = [(s, 0.006) for s in range(1000, 12001, 1000)]
    fires = _feed(r, series)
    assert fires[0] == 3000
    # No fire during the cooldown window (3000, 6000).
    assert all(not (3000 < f < 6000) for f in fires)


def test_max_mode_watches_for_increase():
    r = PlateauLRReducer(mode="max", patience=3000, min_delta=0.05, cooldown=0)
    # PSNR climbs then sits flat at 43.5 (sub-min_delta wobble) → fires.
    series = [(1000, 40.0), (2000, 43.5)]
    series += [(s, 43.5 + 0.01 * ((s // 1000) % 2)) for s in range(3000, 7001, 1000)]
    fires = _feed(r, series)
    assert fires and fires[0] == 5000                      # 2000 + 3000


def test_non_finite_metric_ignored():
    r = PlateauLRReducer(mode="min", patience=2000, min_delta=1e-4, cooldown=0)
    # inf metrics (no active lane) neither reset nor trip the guard.
    assert not r.should_reduce(1000, float("inf"))
    assert not r.should_reduce(2000, float("nan"))
    # A finite value then flat still fires on the finite baseline.
    series = [(3000, 0.006)] + [(s, 0.006) for s in range(4000, 6001, 1000)]
    assert _feed(r, series) == [5000]                      # 3000 + 2000


def test_reset_clears_history():
    r = PlateauLRReducer(mode="min", patience=2000, min_delta=1e-4, cooldown=0)
    r.should_reduce(1000, 0.006)
    r.should_reduce(2000, 0.006)
    r.reset(step=500)                                      # e.g. after a rollback
    # After reset the deadline is measured from the reset step's first eval.
    series = [(600, 0.006)] + [(s, 0.006) for s in range(700, 2601, 100)]
    fires = _feed(r, series)
    assert fires and fires[0] == 2600                      # 600 + 2000
