"""The pure resource advisor: ledger-row parsing, step profiles, match
levels, the recommendation rules and the dashboard summaries.

Every ledger here is synthetic (plain dicts shaped like ``JobLog`` rows); no
test reads the real ``~/.euclid_polish`` ledger or the ``data/`` tree.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from euclid_polish.observability import resource_advisor as ra

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEQ = iter(range(1, 1_000_000))


def _row(step="synthetic_generate", *, state="COMPLETED", cpus=10, gpus=0, memory="20G",
         time_limit="1:00:00", elapsed=600.0, eff=0.6, rss_mb=15_000.0, jobstats_mb=None,
         gpu_util=None, params=None, day=1, jobid=None, **extra) -> dict[str, str]:
    """One ledger row (string cells, like ``JobLog.list_all``)."""
    n = next(_SEQ)
    row = {
        "jobid": jobid or str(n),
        "submitted_at": f"2026-09-{day:02d}T{n // 3600 % 24:02d}:{n // 60 % 60:02d}:{n % 60:02d}Z",
        "step_id": step, "label": f"{step} run", "partition": "gpu" if gpus else "shared",
        "req_cpus": str(cpus), "req_gpus": str(gpus), "req_memory": memory,
        "req_time_limit": time_limit, "params_json": json.dumps(params or {}),
        "state": state, "elapsed_seconds": "" if elapsed is None else str(elapsed),
        "cpu_efficiency": "" if eff is None else str(eff),
        "max_rss_mb": "" if rss_mb is None else str(rss_mb),
        "jobstats_cpu_memory_used_mb": "" if jobstats_mb is None else str(jobstats_mb),
        "alloc_cpus": str(cpus), "alloc_gpus": str(gpus),
        "jobstats_gpu_util": "" if gpu_util is None else str(gpu_util),
    }
    row.update({k: str(v) for k, v in extra.items()})
    return row


_VT = {"regenerate_splits": "validate,test", "n_train": 6400, "n_valid": 100, "n_test": 100,
       "image_size": 510}
_TRAIN = {**_VT, "regenerate_splits": "train"}
_ALL = {**_VT, "regenerate_splits": "", "force": "1"}                  # 6600 images
_SMOKE = {**_ALL, "n_train": 2, "n_valid": 50, "n_test": 50}            # 102 images
_ENS = {"mode": "add", "steps": "70000", "batch_size": "4", "hr_crop_size": "256",
        "member_spec": json.dumps([{"num_res_blocks": 32, "loss": "l2"}, {"num_res_blocks": 32}])}


def _gen(n, *, params=_VT, day=1, **kw):
    return [_row("synthetic_generate", params=params, day=day, **kw) for _ in range(n)]


def _ens(n, *, params=_ENS, day=1, **kw):
    kw.setdefault("cpus", 16)
    kw.setdefault("gpus", 1)
    kw.setdefault("memory", "32G")
    kw.setdefault("gpu_util", 70)
    kw.setdefault("eff", 0.4)
    return [_row("ensemble_train", params=params, day=day, **kw) for _ in range(n)]


def _change(rec, name):
    return next((c for c in rec["changes"] if c["field"] == name), None)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("text", "mb"), [
    ("32G", 32 * 1024), ("17GB", 17 * 1024), ("4000M", 4000), ("1T", 1024 ** 2),
    ("200MB", 200), ("512K", 0.5), ("16gib", 16 * 1024), ("32", 32), (" 8G ", 8 * 1024),
])
def test_memory_formats_follow_slurm(text, mb):
    assert ra.parse_memory_mb(text) == pytest.approx(mb)


@pytest.mark.parametrize("text", ["", None, "lots", "G", "-4G", "0G", "4 X"])
def test_bad_memory_is_none(text):
    assert ra.parse_memory_mb(text) is None


def test_a_bare_form_memory_can_mean_gigabytes():
    assert ra.parse_memory_mb("32", bare_unit="G") == 32 * 1024
    assert ra.parse_memory_mb("32M", bare_unit="G") == 32


@pytest.mark.parametrize(("text", "seconds"), [
    ("30", 1800), ("30:00", 1800), ("5:30", 330), ("2:00:00", 7200), ("0:10:0", 600),
    ("30:00:00", 30 * 3600), ("1-12", 36 * 3600), ("1-12:30", 36 * 3600 + 1800),
    ("2-00:00:00", 2 * 86400), ("3-04:05:06", 3 * 86400 + 4 * 3600 + 5 * 60 + 6),
])
def test_time_formats_follow_slurm(text, seconds):
    assert ra.parse_time_s(text) == seconds


@pytest.mark.parametrize("text", ["", None, "UNLIMITED", "abc", "1:2:3:4", "0:00:00"])
def test_bad_time_is_none(text):
    assert ra.parse_time_s(text) is None


def test_states_are_normalised():
    assert ra.normalize_state("CANCELLED by 12345") == "CANCELLED"
    assert ra.normalize_state("oom") == "OUT_OF_MEMORY"
    assert ra.normalize_state(None) == ""


def test_a_row_normalises_defensively():
    run = ra.normalize_row({
        "jobid": "7", "step_id": "euclid_query", "state": " completed ",
        "req_cpus": "x", "alloc_cpus": "", "req_memory": "lots", "alloc_memory_mb": "8192.0",
        "req_time_limit": "nope", "elapsed_seconds": "nan", "cpu_efficiency": "garbage",
        "max_rss_mb": "", "params_json": "{not json",
    })
    assert run.state == "COMPLETED"
    assert run.cpus is None and run.gpus is None
    assert run.req_memory_mb == 8192.0          # fell back to alloc_memory_mb
    assert run.req_time_s is None and run.elapsed_s is None
    assert run.cpu_efficiency is None and run.cores_used is None
    assert run.peak_mem_mb is None and run.mem_ratio is None and run.time_ratio is None
    assert run.params == {}
    assert run.units is None and run.key == () and not run.counted


def test_a_row_derives_usage():
    run = ra.normalize_row(_row(
        "ensemble_train", cpus=16, gpus=1, memory="32G", time_limit="2:00:00", elapsed=3600,
        eff=0.5, rss_mb=17_000, jobstats_mb=17_500, gpu_util=71, params=_ENS,
        gpu_util_mean=80, jobstats_gpu_memory_used_mb=80_384, gpu_mem_peak_mb=79_654))
    assert run.cpus == 16 and run.gpus == 1
    assert run.cores_used == pytest.approx(8.0)
    assert run.peak_mem_mb == 17_500                 # jobstats larger than MaxRSS
    assert run.mem_ratio == pytest.approx(17_500 / 32_768)
    assert run.time_ratio == pytest.approx(0.5)
    assert run.gpu_util == 71                        # jobstats preferred over the sampler
    assert run.gpu_mem_used_mb == 80_384
    assert run.units == 70_000 and run.units_label == "steps"
    assert run.counted


def test_allocation_falls_back_to_the_request():
    row = _row(cpus=8, gpus=0)
    row.update(alloc_cpus="0", jobstats_gpu_util="", gpu_util_mean="55")
    run = ra.normalize_row(row)
    assert run.cpus == 8
    assert run.gpu_util == 55


def test_run_json_shape():
    keys = set(ra.normalize_row(_row()).to_dict())
    assert keys == {
        "jobid", "submitted_at", "state", "partition", "cpus", "gpus", "req_memory",
        "req_memory_mb", "req_time_limit", "req_time_s", "elapsed_s", "cpu_efficiency",
        "cores_used", "peak_mem_mb", "mem_ratio", "time_ratio", "gpu_util", "gpu_mem_used_mb",
        "units", "units_label", "key_label", "label"}


# ---------------------------------------------------------------------------
# Statistics and formatting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [0, 25, 50, 75, 90, 100])
def test_percentile_matches_numpy(q):
    values = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.0]
    assert ra.percentile([*values, None], q) == pytest.approx(np.percentile(values, q))


def test_percentile_of_one_and_none():
    assert ra.percentile([7.0], 90) == 7.0
    assert ra.percentile([None], 90) is None
    assert ra.median([]) is None


def test_memory_rounds_up_to_four_gigabytes():
    assert ra.round_memory_gb(15.4 * 1024) == 16
    assert ra.round_memory_gb(16 * 1024) == 16
    assert ra.round_memory_gb(16.1 * 1024) == 20
    assert ra.round_memory_gb(100) == 4
    assert ra.format_memory(36) == "36G"


def test_time_rounds_up_to_fifteen_minutes_and_caps():
    assert ra.round_time_s(60) == 900
    assert ra.round_time_s(901) == 1800
    assert ra.round_time_s(10 * 86400) == 3 * 86400
    assert ra.format_time(900) == "0:15:00"
    assert ra.format_time(10_800) == "3:00:00"
    assert ra.format_time(90_000) == "1-01:00:00"


# ---------------------------------------------------------------------------
# Step profiles
# ---------------------------------------------------------------------------

def _units(step, params):
    return ra.profile_for(step).work(params)[0]


def _key(step, params):
    return ra.profile_for(step).similarity(params)


class TestSyntheticProfile:

    def test_regenerated_splits_count_their_images(self):
        assert _units("synthetic_generate", _VT) == 200
        assert _units("synthetic_generate", _TRAIN) == 6400
        assert _units("synthetic_generate", {**_VT, "regenerate_splits": ["train", "test"]}) == 6500

    def test_force_regenerates_everything(self):
        params = {**_VT, "regenerate_splits": "", "force": "1"}
        assert _units("synthetic_generate", params) == 6600
        assert _key("synthetic_generate", params)[0][0] == "all"
        assert (_key("synthetic_generate", {**_VT, "regenerate_splits": "train,validate,test"})[0]
                == _key("synthetic_generate", params)[0])

    def test_a_cache_first_resume_has_unknown_work(self):
        params = {**_VT, "regenerate_splits": ""}
        assert _units("synthetic_generate", params) is None
        key, label = _key("synthetic_generate", params)
        assert key[0] == "resume" and "resume" in label

    def test_split_order_does_not_change_the_key(self):
        a = _key("synthetic_generate", _VT)
        b = _key("synthetic_generate", {**_VT, "regenerate_splits": "test, validate"})
        assert a[0] == b[0] == ("test,validate", 510, False)
        assert a[1] == "validate+test · 510 px"

    def test_the_key_separates_image_size_and_onthefly(self):
        base = _key("synthetic_generate", _VT)[0]
        assert _key("synthetic_generate", {**_VT, "image_size": "96"})[0] != base
        onthefly = _key("synthetic_generate", {**_VT, "onthefly_train": "true"})
        assert onthefly[0] != base and "on-the-fly" in onthefly[1]

    def test_extra_flags_are_a_fallback(self):
        assert _units("synthetic_generate", {**_VT, "regenerate_splits": "",
                                             "extra_flags": "--regenerate-splits=train"}) == 6400
        assert _units("synthetic_generate", {**_VT, "regenerate_splits": "",
                                             "extra_flags": "--force --seed 3"}) == 6600

    def test_a_missing_count_makes_work_unknown(self):
        assert _units("synthetic_generate", {"regenerate_splits": "test", "n_valid": 5}) is None


class TestEnsembleProfile:

    def test_new_members_train_their_steps(self):
        assert _units("ensemble_train", _ENS) == 70_000
        assert _units("ensemble_train", {**_ENS, "mode": "fork", "steps": 5000}) == 5000
        assert _units("ensemble_train", {"steps": "100000"}) == 100_000   # pre-mode rows

    def test_continue_trains_extra_steps_or_unknown(self):
        extra = {"mode": "continue", "continue_basis": "extra", "extra_steps": "50000",
                 "steps": "70000"}
        assert _units("ensemble_train", extra) == 50_000
        assert _units("ensemble_train", {**extra, "continue_basis": ""}) == 50_000
        assert _units("ensemble_train", {**extra, "continue_basis": "target",
                                         "target_steps": "70000"}) is None

    def test_member_spec_as_string_or_list_gives_one_key(self):
        as_list = {**_ENS, "member_spec": json.loads(_ENS["member_spec"])}
        assert _key("ensemble_train", _ENS) == _key("ensemble_train", as_list)
        key, label = _key("ensemble_train", _ENS)
        assert key == ("new", 4, (32,), False, 256)
        assert label == "new members · batch 4 · 32 blocks · single knee · crop 256"

    def test_the_key_tracks_depths_knees_and_kind(self):
        base = _key("ensemble_train", _ENS)[0]
        mixed = {**_ENS, "member_spec": [{"num_res_blocks": 16}, {"num_res_blocks": 32}]}
        assert _key("ensemble_train", mixed)[0][2] == (16, 32)
        knees = {**_ENS, "member_spec": [{"num_res_blocks": 32, "asinh_knees": [0.1, 1, 10]}]}
        assert _key("ensemble_train", knees)[0][3] is True
        cont = _key("ensemble_train", {**_ENS, "mode": "continue"})[0]
        assert cont[0] == "continue" and cont != base
        assert _key("ensemble_train", {**_ENS, "batch_size": "8"})[0] != base

    def test_run_wide_depth_fills_spec_gaps(self):
        params = {**_ENS, "num_res_blocks": "16",
                  "member_spec": [{"num_res_blocks": 32}, {"loss": "l1"}]}
        assert _key("ensemble_train", params)[0][2] == (16, 32)
        assert _key("ensemble_train", {"mode": "add"})[0][2] == (None,)

    def test_a_bad_member_spec_is_tolerated(self):
        key, _ = _key("ensemble_train", {**_ENS, "member_spec": "{broken"})
        assert key[2] == (None,)


def test_other_steps_have_no_units_or_key():
    assert ra.profile_for("train").work({"steps": "5000"}) == (5000, "steps")
    assert ra.profile_for("euclid_query").work({"n": 4}) == (None, "")
    assert ra.profile_for("euclid_query").similarity({"n": 4}) == ((), "")


# ---------------------------------------------------------------------------
# Match levels
# ---------------------------------------------------------------------------

def _recommend(step, rows, params, **current):
    current = {"n_cpus": "10", "n_gpus": "0", "memory": "20G", "time_limit": "1:00:00",
               **current}
    return ra.recommend(step, rows, params=params, current=current,
                        needs_gpu=ra.profile_for(step).gpu)


def test_exact_level_needs_the_same_cpu_count():
    rows = _gen(3, cpus=10) + _gen(4, cpus=20)
    rec = _recommend("synthetic_generate", rows, _VT, n_cpus="10")
    assert rec["basis"]["level"] == "exact" and rec["basis"]["n_runs"] == 3
    assert "at 10 CPUs" in rec["basis"]["level_label"]


def test_similar_level_when_too_few_exact():
    rows = _gen(2, cpus=10) + _gen(4, cpus=20)
    rec = _recommend("synthetic_generate", rows, _VT, n_cpus="10")
    assert rec["basis"]["level"] == "similar" and rec["basis"]["n_runs"] == 6
    assert any("closest match" in n for n in rec["notes"])


def test_step_level_when_too_few_similar():
    rows = _gen(2, params=_TRAIN, cpus=20) + _gen(4, cpus=10)
    rec = _recommend("synthetic_generate", rows, _TRAIN, n_cpus="20")
    assert rec["basis"]["level"] == "step" and rec["basis"]["n_runs"] == 6


def test_most_specific_non_empty_level_when_all_are_thin():
    rows = _gen(1, cpus=10) + _gen(1, params=_TRAIN, cpus=20)
    rec = _recommend("synthetic_generate", rows, _VT, n_cpus="10")
    assert rec["basis"]["level"] == "exact" and rec["basis"]["n_runs"] == 1
    assert rec["confidence"] == "low"
    assert any("rough guide" in w for w in rec["warnings"])


def test_a_gpu_step_has_no_exact_level():
    rec = _recommend("ensemble_train", _ens(4, cpus=16) + _ens(3, cpus=8), _ENS,
                     n_cpus="16", n_gpus="1")
    assert rec["basis"]["level"] == "similar" and rec["basis"]["n_runs"] == 7


def test_only_the_twenty_newest_runs_count():
    old = _gen(10, day=1, elapsed=5000, time_limit="3:00:00")
    new = _gen(20, day=2, elapsed=600)
    rec = _recommend("synthetic_generate", old + new, _VT)
    assert rec["basis"]["n_runs"] == 20
    assert set(rec["basis"]["jobids"]) == {r["jobid"] for r in new}
    assert rec["resources"]["time_limit"] == "0:15:00"


def test_errors_and_live_runs_are_not_evidence():
    rows = (_gen(1, state="FAILED") + _gen(1, state="CANCELLED") + _gen(1, state="RUNNING")
            + _gen(1, state="") + _gen(1, elapsed=0))
    rec = _recommend("synthetic_generate", rows, _VT)
    assert rec["available"] is False and rec["basis"]["n_runs"] == 0


def test_no_history_echoes_the_current_values():
    rec = ra.recommend("synthetic_generate", [], params=_VT,
                       current={"n_cpus": 10, "memory": "17G"},
                       defaults={"n_cpus": 16, "n_gpus": 0, "memory": "64G",
                                 "time_limit": "6:00:00"})
    assert rec["available"] is False
    assert rec["changes"] == []
    assert rec["current"] == {"n_cpus": "10", "n_gpus": None, "memory": "17G", "time_limit": None}
    assert rec["resources"] == {"n_cpus": "10", "n_gpus": "0", "memory": "17G",
                                "time_limit": "6:00:00"}
    assert rec["basis"]["jobids"] == [] and rec["basis"]["units"] == 200


def test_other_steps_rows_are_ignored():
    rec = _recommend("synthetic_generate", _ens(5), _VT)
    assert rec["available"] is False


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def test_memory_is_the_p90_peak_with_headroom():
    rows = _gen(5, cpus=10, rss_mb=10_000, memory="32G")
    rec = _recommend("synthetic_generate", rows, _VT, memory="32G")
    # 10 000 MB × 1.2 = 11.7 GB → 12G.
    assert rec["resources"]["memory"] == "12G"
    change = _change(rec, "memory")
    assert change["current"] == "32G" and change["recommended"] == "12G"
    assert "p90 peak" in change["reason"] and "5 runs" in change["reason"]


def test_memory_scales_per_cpu_when_the_step_does():
    rows = _gen(4, cpus=10, rss_mb=15_360)          # 1.5 GB/CPU
    rec = _recommend("synthetic_generate", rows, _VT, n_cpus="20")
    assert rec["basis"]["level"] == "similar"
    # 1.5 GB × 20 CPUs × 1.2 = 36 GB.
    assert rec["resources"]["memory"] == "36G"
    assert "GB/CPU" in _change(rec, "memory")["reason"]


def test_gpu_memory_does_not_scale_with_cpus():
    rows = _ens(5, cpus=16, rss_mb=17_000)
    rec = _recommend("ensemble_train", rows, _ENS, n_cpus="32", n_gpus="1", memory="32G")
    assert rec["resources"]["memory"] == "20G"      # 17 000 × 1.2 = 19.9 GB
    assert ra.GPU_MEMORY_NOTE in rec["notes"]


def test_an_oom_run_sets_a_floor():
    rows = _gen(4, cpus=10, rss_mb=8_000, memory="16G") + _gen(
        1, cpus=10, state="OUT_OF_MEMORY", memory="20G", rss_mb=20_000, elapsed=100)
    rec = _recommend("synthetic_generate", rows, _VT, memory="16G")
    # peaks say 8 000 × 1.2 = 9.4 GB, the OOM at 20G says 25 GB → 28G.
    assert rec["resources"]["memory"] == "28G"
    assert "1 OOM at up to 20.0 GB" in _change(rec, "memory")["reason"]


def test_the_oom_floor_scales_per_cpu():
    rows = _gen(3, cpus=10, rss_mb=5_000) + _gen(
        1, cpus=10, state="OUT_OF_MEMORY", memory="10G", elapsed=100)
    rec = _recommend("synthetic_generate", rows, _VT, n_cpus="20")
    # OOM floor 10 GB / 10 CPUs × 20 × 1.25 = 25 GB → 28G.
    assert rec["resources"]["memory"] == "28G"


def test_equal_quantities_are_not_changes():
    rows = _gen(5, cpus=10, rss_mb=10_000)
    rec = _recommend("synthetic_generate", rows, _VT, memory="12288M")
    assert _change(rec, "memory") is None


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

def test_time_scales_with_planned_units():
    rows = _gen(5, cpus=10, elapsed=2000)             # 10 s per image
    small = _recommend("synthetic_generate", rows, _VT)
    assert small["basis"]["rate_s_per_unit"] == pytest.approx(10.0)
    assert small["resources"]["time_limit"] == "0:45:00"      # 2000 × 1.2 = 40 min → 45
    big = _recommend("synthetic_generate", rows, {**_VT, "n_valid": 300, "n_test": 300})
    assert big["resources"]["time_limit"] == "2:00:00"        # 6000 × 1.2 = 2 h
    assert big["basis"]["units"] == 600 and big["basis"]["units_label"] == "images"
    assert "s/image" in _change(big, "time_limit")["reason"]


def test_time_has_a_five_minute_margin_and_a_fifteen_minute_floor():
    rows = _gen(5, cpus=10, elapsed=1000)
    rec = _recommend("synthetic_generate", rows, _VT)
    assert rec["resources"]["time_limit"] == "0:30:00"        # 1000 + 300 > 1000 × 1.2
    tiny = _recommend("synthetic_generate", _gen(5, cpus=10, elapsed=30), _VT)
    assert tiny["resources"]["time_limit"] == "0:15:00"


def test_unknown_units_use_whole_run_elapsed():
    resume = {**_VT, "regenerate_splits": ""}
    rows = _gen(5, cpus=10, params=resume, elapsed=3000)
    rec = _recommend("synthetic_generate", rows, resume)
    assert rec["basis"]["rate_s_per_unit"] is None and rec["basis"]["units"] is None
    assert rec["resources"]["time_limit"] == "1:00:00"        # 3000 × 1.2 = 60 min
    assert any("Units unknown" in n for n in rec["notes"])
    assert rec["confidence"] == "medium"                     # units unknown: never high


def test_a_timeout_that_would_time_out_again_is_bumped():
    rows = _gen(4, cpus=10, elapsed=1000) + _gen(
        1, cpus=10, state="TIMEOUT", elapsed=1500, time_limit="0:25:00")
    rec = _recommend("synthetic_generate", rows, _VT)
    # p90 rate over [5, 5, 5, 5, 7.5] s/image = 6.5 → 1300 s + 5 min → 0:30:00, above
    # the timeout's 1500 s: no bump.
    assert rec["resources"]["time_limit"] == "0:30:00"
    assert not rec["warnings"]
    rows = _gen(9, cpus=10, elapsed=1000) + _gen(
        1, cpus=10, state="TIMEOUT", elapsed=7200, time_limit="2:00:00")
    rec = _recommend("synthetic_generate", rows, _VT)
    # p90 ≈ 1620 s → 1944 s → 0:45:00 < 7200 s: bumped to 1.5 × 7200 = 3 h.
    assert rec["resources"]["time_limit"] == "3:00:00"
    assert any("timed out" in w for w in rec["warnings"])
    assert "1 timeout" in _change(rec, "time_limit")["reason"]


def test_the_time_rate_comes_from_runs_of_comparable_size():
    rows = (_gen(3, params=_SMOKE, cpus=32, elapsed=400)             # ~3.9 s/image: mostly startup
            + _gen(3, params=_ALL, cpus=32, elapsed=2500)            # ~0.38 s/image
            + _gen(1, params=_SMOKE, cpus=32, state="TIMEOUT", elapsed=900, time_limit="0:15:00"))
    full = _recommend("synthetic_generate", rows, _ALL, n_cpus="32", time_limit="6:00:00")
    assert full["basis"]["level"] == "exact" and full["basis"]["n_runs"] == 7
    assert full["basis"]["rate_s_per_unit"] == pytest.approx(2500 / 6600)
    # 2500 s × 1.2 = 50 min → 1:00:00; the smoke runs (and the smoke timeout,
    # 58 000 s scaled to 6600 images) do not inflate it.
    assert full["resources"]["time_limit"] == "1:00:00"
    assert "over 3 runs of comparable size" in _change(full, "time_limit")["reason"]
    assert not full["warnings"]
    smoke = _recommend("synthetic_generate", rows, _SMOKE, n_cpus="32")
    assert smoke["basis"]["rate_s_per_unit"] > 3
    huge = _recommend("synthetic_generate", rows, {**_ALL, "n_train": 100_000}, n_cpus="32")
    assert any("within 4×" in n for n in huge["notes"])
    assert huge["basis"]["rate_s_per_unit"] is not None


def test_runs_at_more_cpus_are_stretched_to_the_planned_count():
    rows = _gen(4, cpus=20, elapsed=1000)                # 5 s/image at 20 CPUs
    rec = _recommend("synthetic_generate", rows, _VT, n_cpus="10")
    assert rec["basis"]["level"] == "similar"
    # One worker per CPU: measured at 20, planned at 10 → 10 s/image × 200 = 2000 s
    # (+20%) → 0:45:00.
    assert rec["basis"]["rate_s_per_unit"] == pytest.approx(10.0)
    assert rec["resources"]["time_limit"] == "0:45:00"
    assert "stretched to 10" in _change(rec, "time_limit")["reason"]
    # More CPUs than measured is not assumed to be faster.
    more = _recommend("synthetic_generate", rows, _VT, n_cpus="40")
    assert more["basis"]["rate_s_per_unit"] == pytest.approx(5.0)
    # A GPU step's time does not scale with its CPUs.
    gpu = _recommend("ensemble_train", _ens(5, cpus=16, elapsed=7000), _ENS, n_cpus="8", n_gpus="1")
    assert gpu["basis"]["rate_s_per_unit"] == pytest.approx(0.1)


def test_time_is_capped_at_three_days():
    rows = _gen(3, cpus=10, elapsed=3 * 86400, state="TIMEOUT", time_limit="3-00:00:00")
    rec = _recommend("synthetic_generate", rows, _VT)
    assert rec["resources"]["time_limit"] == "3-00:00:00"
    assert any("Capped" in w for w in rec["warnings"])


# ---------------------------------------------------------------------------
# CPUs
# ---------------------------------------------------------------------------

def test_an_idle_cpu_step_gets_fewer_cpus():
    rows = _gen(5, cpus=32, eff=0.1)                  # 3.2 busy cores
    rec = _recommend("synthetic_generate", rows, _VT, n_cpus="32")
    assert rec["resources"]["n_cpus"] == "5"          # ceil(3.2 × 1.3)
    assert "Fewer CPUs may lengthen the run." in rec["notes"]
    assert "CPU efficiency 10%" in _change(rec, "n_cpus")["reason"]


def test_a_busy_cpu_step_keeps_its_cpus():
    rec = _recommend("synthetic_generate", _gen(5, cpus=10, eff=0.6), _VT, n_cpus="10")
    assert rec["resources"]["n_cpus"] == "10" and _change(rec, "n_cpus") is None


def test_a_gpu_step_never_loses_cpus():
    rec = _recommend("ensemble_train", _ens(5, cpus=16, eff=0.05, gpu_util=90), _ENS,
                     n_cpus="16", n_gpus="1")
    assert rec["resources"]["n_cpus"] == "16"


def test_a_starved_gpu_gets_more_cpus():
    rec = _recommend("ensemble_train", _ens(5, cpus=4, eff=0.9, gpu_util=40), _ENS,
                     n_cpus="4", n_gpus="1")
    assert rec["resources"]["n_cpus"] == "6"
    assert any("starved" in n for n in rec["notes"])
    assert _change(rec, "n_cpus")["recommended"] == "6"


def test_fewer_cpus_learn_memory_and_time_from_runs_at_the_new_count():
    # Idle 32-CPU runs (1 GB/CPU, 250 s) beside busier 10-CPU ones (1.5 GB/CPU, 600 s).
    rows = (_gen(3, cpus=32, eff=0.3, rss_mb=32_000, elapsed=250, memory="64G")
            + _gen(8, cpus=10, eff=0.55, rss_mb=15_360, elapsed=600, memory="17G"))
    rec = _recommend("synthetic_generate", rows, _VT, n_cpus="32", memory="64G",
                     time_limit="0:15:00")
    assert rec["resources"]["n_cpus"] == "13"             # ceil(p75 9.6 busy cores × 1.3)
    assert "CPU efficiency 30%" in _change(rec, "n_cpus")["reason"]
    # Memory and time from the runs at 13 CPUs (no exact match: every run like
    # this, per CPU): 1.5 GB × 13 × 1.2 = 23.4 GB → 24G, not the 32-CPU runs'
    # 1 GB × 13 × 1.2 → 16G; 250 s at 32 CPUs is 615 s at 13 → 0:30:00, not 0:15:00.
    assert rec["basis"]["level"] == "similar" and "at 13 CPUs" not in rec["basis"]["level_label"]
    assert rec["resources"]["memory"] == "24G"
    assert rec["resources"]["time_limit"] == "0:30:00"


@pytest.mark.parametrize(("rows", "step", "cpus"), [
    ((_gen(3, cpus=32, eff=0.3, rss_mb=32_000, elapsed=250, memory="64G")
      + _gen(8, cpus=10, eff=0.55, rss_mb=15_360, elapsed=600, memory="17G")
      + _gen(1, cpus=10, state="OUT_OF_MEMORY", memory="15G", elapsed=300)),
     "synthetic_generate", "32"),
    (_gen(5, cpus=32, eff=0.1), "synthetic_generate", "32"),
    (_ens(6, cpus=16, eff=0.9, gpu_util=40), "ensemble_train", "16"),
    (_ens(5, cpus=16, eff=0.4, gpu_util=70), "ensemble_train", "16"),
])
def test_asking_again_with_the_recommendation_changes_nothing(rows, step, cpus):
    """Apply is stable: the recommended resources are their own recommendation."""
    gpus = "1" if step == "ensemble_train" else "0"
    params = _ENS if step == "ensemble_train" else _VT
    first = _recommend(step, rows, params, n_cpus=cpus, n_gpus=gpus, memory="64G",
                       time_limit="12:00:00")
    assert first["changes"]
    again = _recommend(step, rows, params, **first["resources"])
    assert again["changes"] == [] and again["resources"] == first["resources"]


def test_a_starved_gpu_settles_on_one_and_a_half_times_its_runs_cpus():
    rows = _ens(6, cpus=16, eff=0.9, gpu_util=40)
    first = _recommend("ensemble_train", rows, _ENS, n_cpus="16", n_gpus="1")
    assert first["resources"]["n_cpus"] == "24"
    assert "at 16 CPUs" in _change(first, "n_cpus")["reason"]
    # After Apply (24) the same runs say 24 again: no 24 → 36 → 54 spiral.
    assert _change(_recommend("ensemble_train", rows, _ENS, n_cpus="24", n_gpus="1"),
                   "n_cpus") is None
    # A form already above 1.5x keeps its CPUs (a GPU step never loses any).
    assert _recommend("ensemble_train", rows, _ENS, n_cpus="40",
                      n_gpus="1")["resources"]["n_cpus"] == "40"


def test_a_busy_gpu_keeps_its_cpus():
    rec = _recommend("ensemble_train", _ens(5, cpus=4, eff=0.9, gpu_util=75), _ENS,
                     n_cpus="4", n_gpus="1")
    assert rec["resources"]["n_cpus"] == "4"


def test_fixed_cpus_and_gpus_win():
    rec = ra.recommend("synthetic_generate", _gen(5, cpus=32, eff=0.05), params=_VT,
                       current={"n_cpus": "32", "n_gpus": "2", "memory": "20G",
                                "time_limit": "1:00:00"},
                       fixed_cpus=8, fixed_gpus=0)
    assert rec["resources"]["n_cpus"] == "8" and rec["resources"]["n_gpus"] == "0"
    assert _change(rec, "n_cpus")["reason"] == "fixed for this step"


def test_a_blank_field_is_a_change():
    rec = ra.recommend("synthetic_generate", _gen(5, cpus=10), params=_VT,
                       current={"n_cpus": "10", "n_gpus": "0", "memory": "",
                                "time_limit": "1:00:00"})
    change = _change(rec, "memory")
    assert change["current"] is None and change["recommended"] == rec["resources"]["memory"]


# ---------------------------------------------------------------------------
# Confidence and output shape
# ---------------------------------------------------------------------------

def test_confidence_levels():
    assert _recommend("synthetic_generate", _gen(5, cpus=10), _VT)["confidence"] == "high"
    assert _recommend("synthetic_generate", _gen(4, cpus=10), _VT)["confidence"] == "medium"
    assert _recommend("synthetic_generate", _gen(2, cpus=10), _VT)["confidence"] == "low"
    # The step level is never high, however many runs.
    other = _gen(8, cpus=10, params=_TRAIN)
    assert _recommend("synthetic_generate", other, _VT)["confidence"] == "medium"
    # A step without units or a key: every run is alike, so only the count matters.
    rows = [_row("euclid_query", cpus=1) for _ in range(5)]
    rec = _recommend("euclid_query", rows, {}, n_cpus="1")
    assert rec["basis"]["level"] == "step" and rec["confidence"] == "high"


def test_recommendation_shape():
    rec = _recommend("ensemble_train", _ens(5), _ENS, n_cpus="16", n_gpus="1")
    assert set(rec) == {"step_id", "available", "confidence", "resources", "current",
                        "changes", "basis", "notes", "warnings"}
    assert set(rec["resources"]) == {"n_cpus", "n_gpus", "memory", "time_limit"}
    assert all(isinstance(v, str) for v in rec["resources"].values())
    assert set(rec["basis"]) == {"level", "level_label", "n_runs", "jobids", "units",
                                 "units_label", "rate_s_per_unit"}
    for change in rec["changes"]:
        assert set(change) == {"field", "current", "recommended", "reason"}
        assert change["reason"]


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def test_step_summary_counts_and_hours():
    rows = (_gen(2, cpus=10, eff=0.5, elapsed=3600, memory="20G", rss_mb=10_240)
            + _gen(1, cpus=10, state="OUT_OF_MEMORY", elapsed=1800, memory="20G",
                   rss_mb=20_480, eff=0.5)
            + _gen(1, cpus=10, state="FAILED", elapsed=360, eff=0.0, rss_mb=100)
            + _gen(1, cpus=10, state="CANCELLED", elapsed=0)
            + _gen(1, state="RUNNING", elapsed=None))
    summary = ra.summarize_step("synthetic_generate", rows, label="Generate", needs_gpu=False)
    assert summary["states"] == {"completed": 2, "oom": 1, "timeout": 0, "failed": 1,
                                 "cancelled": 1, "running": 1}
    assert summary["runs"] == 6
    assert summary["success_rate"] == pytest.approx(2 / 5)
    assert summary["label"] == "Generate" and summary["needs_gpu"] is False
    # Evidence runs: the two completions and the OOM.
    assert summary["cpu_efficiency"] == pytest.approx(0.5)
    assert summary["mem_ratio"] == pytest.approx(0.5)
    # Finished with elapsed: 2 × 1 h + 0.5 h + 0.1 h at 10 CPUs.
    assert summary["cpu_hours_alloc"] == pytest.approx(26.0)
    assert summary["cpu_hours_used"] == pytest.approx(12.5)
    assert summary["gpu_hours_alloc"] == 0
    assert summary["mem_gb_hours_alloc"] == pytest.approx(20 * 2.6)
    assert summary["mem_gb_hours_used"] == pytest.approx(10 * 2 + 20 * 0.5 + 100 / 1024 * 0.1,
                                                         abs=1e-3)


def test_gpu_summary_hours_use_the_utilisation():
    summary = ra.summarize_step("ensemble_train", _ens(2, elapsed=7200, gpu_util=50))
    assert summary["needs_gpu"] is True and summary["label"] == "ensemble_train"
    assert summary["gpu_hours_alloc"] == pytest.approx(4.0)
    assert summary["gpu_hours_used"] == pytest.approx(2.0)
    assert summary["gpu_util"] == 50


def test_summary_of_an_empty_step():
    summary = ra.summarize_step("euclid_query", [])
    assert summary["runs"] == 0 and summary["success_rate"] is None
    assert summary["cpu_efficiency"] is None and summary["last_submitted_at"] is None


def test_summary_shape():
    assert set(ra.summarize_step("synthetic_generate", _gen(1))) == {
        "step_id", "label", "needs_gpu", "runs", "states", "success_rate",
        "last_submitted_at", "cpu_efficiency", "gpu_util", "mem_ratio", "time_ratio",
        "peak_mem_p90_mb", "cpu_hours_alloc", "cpu_hours_used", "gpu_hours_alloc",
        "gpu_hours_used", "mem_gb_hours_alloc", "mem_gb_hours_used"}


def test_summaries_pin_the_main_steps_first():
    rows = ([_row("euclid_query", day=5)] + [_row("tng_grid", day=9)]
            + _gen(1, day=2) + _ens(1, day=1))
    order = [s["step_id"] for s in ra.order_summaries(
        ra.summarize_step(s, rows) for s in ("euclid_query", "tng_grid",
                                             "synthetic_generate", "ensemble_train"))]
    assert order == ["ensemble_train", "synthetic_generate", "tng_grid", "euclid_query"]
