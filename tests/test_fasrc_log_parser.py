"""Tests for the FASRC log → structured-status parser.

Each parser is exercised against verbatim output snippets the real
pipeline emits — banner lines from ``run_pipeline.py``, tqdm.write
lines from the trainer, raw tqdm progress from .err, and JSONL
validation records.
"""

from __future__ import annotations

import json
import textwrap

from euclid_polish.web import fasrc_log_parser


# ---------------------------------------------------------------------------
# Stage detection from .out
# ---------------------------------------------------------------------------

def test_stage_init_when_no_banner_seen_yet():
    out = "Pipeline started at 2026-05-13 01:42:09\n  args = {...}\n"
    p = fasrc_log_parser.parse_stdout(out)
    assert p["stage"] == "init"
    assert p["stage_index"] == 0
    assert p["last_train_step"] is None


def test_stage_generate_after_step1_banner():
    out = textwrap.dedent("""\
        ======================================================================
        [2026-05-13 01:42:10] STEP 1: Generate clean 4-band HR fields (...)
        ======================================================================
        generating train 1/6400
    """)
    p = fasrc_log_parser.parse_stdout(out)
    assert p["stage"] == "generate"


def test_stage_train_after_step3_banner():
    out = textwrap.dedent("""\
        [...] STEP 1: Generate clean 4-band HR fields ...
        [...] STEP 2: HR → LR ...
        [...] STEP 3: Train WDSR  (400000 steps, batch 16, eval every 1000)
    """)
    p = fasrc_log_parser.parse_stdout(out)
    assert p["stage"] == "train"


def test_train_step_line_parsed():
    out = ("[2026-05-13 02:30:00] Step 12345/400000: loss = 4.2106, "
           "PSNR(str/raw) = 22.314/19.872 dB, |g| avg/max = 0.31/1.4 (3.51s)")
    p = fasrc_log_parser.parse_stdout(out)
    m = p["last_train_step"]
    assert m["step"]  == 12345
    assert m["total"] == 400000
    assert abs(m["loss"] - 4.2106) < 1e-6
    assert abs(m["psnr_stretched"] - 22.314) < 1e-3
    assert abs(m["psnr_raw"]       - 19.872) < 1e-3
    # Train-step line also pins the stage.
    assert p["stage"] == "train"


def test_checkpoint_saved_marker_picked_up():
    out = textwrap.dedent("""\
        [...] STEP 3: Train WDSR (400000 steps, ...)
        [...] Step 1000/400000: loss = 4.2, PSNR(str/raw) = 22/19 dB
          ✓ Checkpoint saved (PSNR str=22.314)
    """)
    p = fasrc_log_parser.parse_stdout(out)
    assert p["last_ckpt_line"]
    assert "Checkpoint saved" in p["last_ckpt_line"]


def test_pipeline_done_flag():
    p = fasrc_log_parser.parse_stdout("...\nDone in 65.2 min\n")
    assert p["pipeline_done"] is True


# ---------------------------------------------------------------------------
# .err tqdm parser
# ---------------------------------------------------------------------------

def test_tqdm_progress_picks_latest_pair():
    err = textwrap.dedent("""\
        Training:   0% 0/200000 [00:00<?, ?step/s]
        Training:   1% 2050/200000 [03:24<5:30:14, 10.0step/s, loss=4.2]
        Training:   3% 5500/200000 [09:00<5:14:00, 10.1step/s]
    """)
    out = fasrc_log_parser.parse_stderr_progress(err)
    assert out == {"current": 5500, "total": 200000}


def test_tqdm_rejects_small_totals():
    # Not a tqdm bar: "4/4" channel count etc.
    assert fasrc_log_parser.parse_stderr_progress("output shape 4/4") is None


def test_tqdm_returns_none_when_absent():
    assert fasrc_log_parser.parse_stderr_progress("just some warning text") is None


# ---------------------------------------------------------------------------
# training_log.jsonl parser
# ---------------------------------------------------------------------------

def test_training_log_jsonl_parsed():
    jsonl = "\n".join(json.dumps(d) for d in [
        {"step": 1000, "loss": 4.2,  "psnr_stretched": 22.0, "psnr_raw": 19.5,
         "duration_s": 3.4, "wall_time": 1700_000_000.0},
        {"step": 2000, "loss": 3.6,  "psnr_stretched": 23.1, "psnr_raw": 20.2,
         "duration_s": 3.5, "wall_time": 1700_000_100.0},
    ]) + "\n"
    rows = fasrc_log_parser.parse_training_log(jsonl)
    assert [r["step"] for r in rows] == [1000, 2000]
    assert abs(rows[1]["psnr_stretched"] - 23.1) < 1e-6


def test_training_log_skips_malformed_lines():
    jsonl = textwrap.dedent("""\
        not json
        {"step": 1000, "loss": 4.2, "psnr_stretched": 22.0, "psnr_raw": 19.5, "duration_s": 1, "wall_time": 0}
        [also not json]
        {"step": 2000, "loss": 3.6, "psnr_stretched": 23.1, "psnr_raw": 20.2, "duration_s": 1, "wall_time": 0}
    """)
    rows = fasrc_log_parser.parse_training_log(jsonl)
    assert len(rows) == 2


def test_training_log_csv_format_parsed():
    """Trainer now writes CSV; parser must auto-detect it from the header."""
    csv_text = textwrap.dedent("""\
        step,wall_time,loss,psnr_stretched,psnr_raw,gnorm_avg,gnorm_max,clip_norm,duration_s
        1000,1700000000.0,4.2,22.0,19.5,1.1,2.2,5.0,3.4
        2000,1700000100.0,3.6,23.1,20.2,1.0,2.1,5.0,3.5
        3000,1700000200.0,3.1,24.0,20.9,0.9,2.0,5.0,3.6
    """)
    rows = fasrc_log_parser.parse_training_log(csv_text)
    assert [r["step"] for r in rows] == [1000, 2000, 3000]
    assert abs(rows[1]["psnr_stretched"] - 23.1) < 1e-6
    assert abs(rows[-1]["loss"] - 3.1) < 1e-6


def test_training_log_csv_with_partial_columns():
    """Future-proofing: a CSV missing optional columns still parses."""
    csv_text = "step,wall_time,loss,psnr_stretched,psnr_raw\n1,0,1.0,10.0,9.0\n"
    rows = fasrc_log_parser.parse_training_log(csv_text)
    assert len(rows) == 1
    assert rows[0]["step"] == 1
    assert rows[0]["psnr_stretched"] == 10.0
    # Missing optional columns default to 0.0.
    assert rows[0]["duration_s"] == 0.0


def test_training_log_csv_header_anchor_excludes_garbage():
    """A file that doesn't start with ``step,`` is read as legacy JSONL,
    so a misnamed first row can't be misparsed as a CSV header."""
    garbage = "garbage\n{\"step\": 1, \"loss\": 1.0, \"psnr_stretched\": 10.0, \"psnr_raw\": 9.0, \"duration_s\": 0, \"wall_time\": 0}\n"
    rows = fasrc_log_parser.parse_training_log(garbage)
    assert len(rows) == 1
    assert rows[0]["step"] == 1


# ---------------------------------------------------------------------------
# ETA estimation
# ---------------------------------------------------------------------------

def test_eta_live_uses_elapsed_per_step():
    eta = fasrc_log_parser.estimate_eta(
        stage="train",
        train_step={"step": 1000, "total": 100_000},
        elapsed_seconds=60.0,
        history_secs_per_step=None,
    )
    # 60 s for 1000 steps → 99_000 steps remaining at 0.06 s/step ≈ 5940 s.
    assert 5800 < eta < 6000


def test_eta_falls_back_to_history_when_no_step_yet():
    # No train_step yet but we're in train stage and have history.
    eta = fasrc_log_parser.estimate_eta(
        stage="train",
        train_step={"step": 0, "total": 400_000},
        elapsed_seconds=10.0,
        history_secs_per_step=0.01,
    )
    # Live formula needs step > 0, so falls back to history.
    assert eta == 4000.0


def test_eta_none_when_no_data():
    eta = fasrc_log_parser.estimate_eta(
        stage="init", train_step=None,
        elapsed_seconds=0.0, history_secs_per_step=None,
    )
    assert eta is None


# ---------------------------------------------------------------------------
# summarise() — the dict the route returns
# ---------------------------------------------------------------------------

def test_summarise_produces_clean_payload():
    out_text = ("[t] STEP 3: Train WDSR (400000 steps, batch 16, eval every 1000)\n"
                "[t] Step 12345/400000: loss = 4.2, "
                "PSNR(str/raw) = 22.3/19.7 dB, |g| 0.3/1.4 (3.51s)\n"
                "  ✓ Checkpoint saved (PSNR str=22.3)\n")
    err_text  = "Training:   3% 12345/400000 [12:00<...]"
    jsonl     = json.dumps({"step": 12000, "loss": 4.21,
                            "psnr_stretched": 22.2, "psnr_raw": 19.6,
                            "duration_s": 3.5, "wall_time": 1.0}) + "\n"
    s = fasrc_log_parser.summarise(out_text, err_text, jsonl,
                                   elapsed_seconds=600.0,
                                   history_secs_per_step=0.05)
    assert s["stage"] == "train"
    assert s["stage_index"] == 3
    assert s["progress"]["current"] == 12345
    assert s["progress"]["total"]   == 400000
    assert s["latest_metrics"]["psnr_stretched"] == 22.3
    assert s["last_checkpoint"]
    assert s["latest_validation"]["step"] == 12000
    # Live ETA: 600s elapsed at step 12345 / 400000 → ~18800s remaining.
    assert 18_000 < s["eta_seconds"] < 19_500
