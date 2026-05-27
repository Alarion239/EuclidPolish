"""Structured observability for FASRC jobs.

Public surface:

  * :class:`Reporter` — emit JSONL events from a running script.
  * :data:`ENV_EVENTS_PATH` — name of the env var the sbatch template
    sets to point at the per-job events file.
"""

from euclid_polish.observability.job_log import JobLog, JobRecord
from euclid_polish.observability.reporter import ENV_EVENTS_PATH, Reporter

__all__ = ["ENV_EVENTS_PATH", "JobLog", "JobRecord", "Reporter"]
