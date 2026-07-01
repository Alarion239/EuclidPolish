"""Warmup → cosine LR schedule (pure Python; no TF)."""

from __future__ import annotations

import math

from euclid_polish.training.lr_schedule import WarmupCosineDecay, warmup_cosine_lr


def test_warmup_ramps_linearly_to_peak():
    kw = dict(peak_lr=5e-4, final_lr=2e-5, warmup_steps=2000,
              total_steps=100_000, start_lr=1e-5)
    assert warmup_cosine_lr(0, **kw) == 1e-5                    # starts at start_lr
    mid = warmup_cosine_lr(1000, **kw)                          # halfway up the ramp
    assert math.isclose(mid, (1e-5 + 5e-4) / 2, rel_tol=1e-6)
    assert math.isclose(warmup_cosine_lr(2000, **kw), 5e-4, rel_tol=1e-9)  # peak


def test_cosine_decays_from_peak_to_final():
    kw = dict(peak_lr=5e-4, final_lr=2e-5, warmup_steps=2000, total_steps=100_000,
              start_lr=1e-5)
    # Just after warmup it's at the peak; at the end it's at the floor.
    assert math.isclose(warmup_cosine_lr(2000, **kw), 5e-4, rel_tol=1e-9)
    assert math.isclose(warmup_cosine_lr(100_000, **kw), 2e-5, rel_tol=1e-9)
    # Midpoint of the cosine phase is the average of peak and final.
    mid_step = 2000 + (100_000 - 2000) // 2
    assert math.isclose(warmup_cosine_lr(mid_step, **kw), (5e-4 + 2e-5) / 2,
                        rel_tol=1e-3)


def test_monotonic_decrease_through_decay_phase():
    kw = dict(peak_lr=5e-4, final_lr=2e-5, warmup_steps=2000, total_steps=100_000,
              start_lr=1e-5)
    prev = warmup_cosine_lr(2000, **kw)
    for s in range(3000, 100_001, 1000):
        cur = warmup_cosine_lr(s, **kw)
        assert cur <= prev + 1e-12                             # non-increasing
        prev = cur


def test_clamps_past_total_and_no_warmup():
    kw = dict(peak_lr=5e-4, final_lr=2e-5, warmup_steps=0, total_steps=1000,
              start_lr=1e-5)
    assert math.isclose(warmup_cosine_lr(0, **kw), 5e-4, rel_tol=1e-9)   # no ramp
    # Past the end it stays at the floor rather than overshooting.
    assert math.isclose(warmup_cosine_lr(5000, **kw), 2e-5, rel_tol=1e-9)


def test_callable_wrapper_matches_function_and_default_start():
    s = WarmupCosineDecay(peak_lr=5e-4, final_lr=2e-5, warmup_steps=2000,
                          total_steps=100_000)
    assert s.start_lr == 0.05 * 5e-4                            # small, not zero
    assert math.isclose(s(2000), 5e-4, rel_tol=1e-9)
    assert math.isclose(s(100_000), 2e-5, rel_tol=1e-9)
