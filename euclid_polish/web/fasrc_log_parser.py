"""Parse SLURM ``.out`` / ``.err`` / ``training_log.jsonl`` into a
single structured summary for the FASRC dashboard.

The pipeline driver and trainer emit a few load-bearing markers; this
module is the canonical place that knows about them so the route layer
stays trivial:

  * ``.out`` (run_pipeline + trainer's tqdm.write):
      - ``"STEP 1: Generate clean ..."``       → stage = ``generate``
      - ``"STEP 2: HR → LR ..."``              → stage = ``convolve``
      - ``"STEP 3: Train WDSR ..."``           → stage = ``train``
      - ``"Step N/M: loss = X, PSNR(str/raw) = ..."``  → live training
      - ``"✓ Checkpoint saved (PSNR str=...)"`` → checkpoint event

  * ``.err``: tqdm's progress line. Format varies a bit by tqdm version
    but always contains ``current/total`` near the front.

  * ``training_log.jsonl``: one JSON object per validation event, with
    keys ``step``, ``loss``, ``psnr_stretched``, ``psnr_raw``, etc.

Outputs a compact dict the JSON layer can return verbatim.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


_STAGE_FROM_STEP_LINE = {
    "1": "generate",
    "2": "convolve",
    "3": "train",
}
_STAGE_INDEX = {"init": 0, "generate": 1, "convolve": 2, "train": 3}

# ``STEP 3: Train WDSR ...`` — captures the numeric step from the banner.
_RE_BANNER     = re.compile(r"STEP\s+(\d)\s*:", re.IGNORECASE)
# ``Step 12345/400000: loss = ...`` — the per-evaluate tqdm.write line.
_RE_TRAIN_STEP = re.compile(
    r"Step\s+(\d+)\s*/\s*(\d+)\s*:.*loss\s*=\s*([-\d.eE+]+).*"
    r"PSNR\(str/raw\)\s*=\s*([-\d.eE+]+)\s*/\s*([-\d.eE+]+)",
)
# Anywhere in a log line, look for "N/M" near the front — tqdm.
_RE_TQDM       = re.compile(r"(?:^|\s)(\d{1,9})\s*/\s*(\d{2,9})")
_RE_CKPT_SAVED = re.compile(r"Checkpoint\s+saved", re.IGNORECASE)


def parse_stdout(text: str, *, max_lines: int = 2_000) -> Dict[str, Any]:
    """Extract stage + last live-training line + checkpoint markers from .out."""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]

    stage: Optional[str] = None
    last_train_step: Optional[Dict[str, Any]] = None
    last_ckpt_line: Optional[str] = None
    pipeline_done = False
    last_banner_idx: int = -1

    for line in lines:
        m = _RE_BANNER.search(line)
        if m:
            stage = _STAGE_FROM_STEP_LINE.get(m.group(1))
            continue
        m = _RE_TRAIN_STEP.search(line)
        if m:
            last_train_step = {
                "step":           int(m.group(1)),
                "total":          int(m.group(2)),
                "loss":           float(m.group(3)),
                "psnr_stretched": float(m.group(4)),
                "psnr_raw":       float(m.group(5)),
            }
            stage = "train"
            continue
        if _RE_CKPT_SAVED.search(line):
            last_ckpt_line = line.strip()
            stage = "train"
            continue
        if "Done in" in line and "min" in line:
            pipeline_done = True

    if stage is None and lines:
        # Before any STEP banner — still in init.
        stage = "init"

    return {
        "stage":           stage,
        "stage_index":     _STAGE_INDEX.get(stage or "", 0),
        "last_train_step": last_train_step,
        "last_ckpt_line":  last_ckpt_line,
        "pipeline_done":   pipeline_done,
    }


def parse_stderr_progress(text: str) -> Optional[Dict[str, int]]:
    """Pull the most recent ``current/total`` pair out of a tqdm log.

    Returns ``None`` if no plausible progress line is present yet.
    """
    last: Optional[Dict[str, int]] = None
    for line in text.splitlines():
        m = _RE_TQDM.search(line)
        if not m:
            continue
        cur, tot = int(m.group(1)), int(m.group(2))
        # Filter false positives like "4/4" channel counts.
        if tot < 50 or cur > tot:
            continue
        last = {"current": cur, "total": tot}
    return last


def parse_training_log(text: str, *, max_records: int = 200) -> List[Dict[str, Any]]:
    """Parse the trainer's append-only JSONL into a list of validation rows."""
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        out.append({
            "step":           int(rec.get("step", 0)),
            "loss":           float(rec.get("loss", 0.0)),
            "psnr_stretched": float(rec.get("psnr_stretched", 0.0)),
            "psnr_raw":       float(rec.get("psnr_raw", 0.0)),
            "duration_s":     float(rec.get("duration_s", 0.0)),
            "wall_time":      float(rec.get("wall_time", 0.0)),
        })
    if len(out) > max_records:
        out = out[-max_records:]
    return out


def estimate_eta(*, stage: Optional[str],
                 train_step: Optional[Dict[str, Any]],
                 elapsed_seconds: float,
                 history_secs_per_step: Optional[float]) -> Optional[float]:
    """Wall-time ETA in seconds, refined live during the train stage.

    Live mode (train + ``train_step`` populated): linear extrapolation
    from observed pace. Pre-train mode: median ``sec/step`` from history
    × planned step count — caller provides the history estimate.
    """
    if (stage == "train" and train_step
            and train_step.get("total") and train_step.get("step")):
        total = train_step["total"]
        step  = train_step["step"]
        if step <= 0:
            return None
        remaining = max(0, total - step)
        # Use elapsed-vs-current-step to estimate pace.
        if elapsed_seconds > 0:
            return remaining * (elapsed_seconds / step)
    if history_secs_per_step is not None and train_step:
        total = train_step.get("total") or 0
        return history_secs_per_step * total
    return None


def summarise(out_text: str, err_text: str,
              jsonl_text: str = "",
              elapsed_seconds: float = 0.0,
              history_secs_per_step: Optional[float] = None,
              ) -> Dict[str, Any]:
    """Single dict the route hands to the UI verbatim."""
    out = parse_stdout(out_text)
    err = parse_stderr_progress(err_text)
    validations = parse_training_log(jsonl_text) if jsonl_text else []

    # Progress source preference: the trainer's structured "Step N/M:"
    # line on .out (rich; includes psnr) beats raw tqdm on .err.
    step_info = out.get("last_train_step") or err
    progress = None
    if step_info:
        cur = step_info.get("current") or step_info.get("step")
        tot = step_info.get("total")
        if cur and tot:
            progress = {
                "current": cur,
                "total":   tot,
                "pct":     round(100.0 * cur / tot, 2),
            }

    eta = estimate_eta(
        stage=out.get("stage"),
        train_step=out.get("last_train_step"),
        elapsed_seconds=elapsed_seconds,
        history_secs_per_step=history_secs_per_step,
    )

    return {
        "stage":             out.get("stage"),
        "stage_index":       out.get("stage_index", 0),
        "pipeline_done":     out.get("pipeline_done", False),
        "progress":          progress,
        "latest_metrics":    out.get("last_train_step"),
        "last_checkpoint":   out.get("last_ckpt_line"),
        "validations":       validations,
        "latest_validation": validations[-1] if validations else None,
        "eta_seconds":       eta,
    }
