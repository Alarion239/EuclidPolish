#!/usr/bin/env python
"""One-shot backfill of GPU utilisation (+ all post-mortem actuals) for every
finished FASRC job in the local job log.

GPU util/mem now come from ``sacct``'s NVML job accounting
(``TRESUsageInTot`` → ``gres/gpuutil`` / ``gres/gpumem``), so any job ``sacct``
still retains can have its ``GPU util`` column filled retroactively — no in-job
sampling needed. This re-queries ``sacct`` for each terminal job and rewrites
the actuals via :func:`refresh_all_post_mortems`, then prints the GPU rows.

Run it locally (where the job log + FASRC SSH config live), e.g.::

    python scripts/backfill_gpu_stats.py

If the WebUI is already connected to FASRC its SSH ControlMaster is reused (no
re-auth, and it is left running); otherwise a temporary connection is opened
and torn down. Jobs ``sacct`` has already purged are skipped — their old rows
stay as-is.
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.web import fasrc_config  # noqa: E402
from euclid_polish.web.fasrc_jobs import JOBLOG, refresh_all_post_mortems  # noqa: E402
from euclid_polish.web.remote import SSHConfig, SSHError, SSHSession  # noqa: E402


def _is_gpu_row(row: dict) -> bool:
    """A job that used a GPU (so a blank GPU-util column is worth reporting)."""
    try:
        if int(float(row.get("alloc_gpus") or 0)) > 0:
            return True
    except (TypeError, ValueError):
        pass
    # Fall back to the recorded util in case alloc_gpus wasn't captured.
    return bool((row.get("gpu_util_mean") or "").strip())


def main() -> int:
    cfg = fasrc_config.load()
    if not cfg.ssh_user:
        print("✗ No ssh_user configured. Set it in the WebUI Settings (or "
              f"{fasrc_config.CONFIG_PATH}) first.")
        return 2

    ssh = SSHSession(SSHConfig(
        user=cfg.ssh_user, host=cfg.ssh_host,
        socket=cfg.control_socket, control_persist=cfg.control_persist,
    ))

    # Reuse a live ControlMaster (e.g. the running WebUI's) so we don't re-auth
    # or tear down a connection someone else is using. Only open + clean up our
    # own if none exists.
    created = False
    if not ssh.is_connected():
        print(f"Connecting to {ssh.cfg.target} …")
        try:
            ssh.connect()
        except SSHError as e:
            print(f"✗ SSH connect failed: {e}")
            return 1
        created = True
    else:
        print(f"Reusing existing SSH session to {ssh.cfg.target}.")

    total_jobs = len(JOBLOG.list_all())
    print(f"Re-querying sacct for the {total_jobs} job(s) in the log "
          "(terminal jobs only are re-recorded) …")
    try:
        res = refresh_all_post_mortems(ssh, job_log=JOBLOG)
    finally:
        if created:
            ssh.disconnect()

    if not res.get("ok"):
        print(f"✗ Backfill failed: {res.get('error', 'unknown error')}")
        return 1

    print(f"✓ Re-recorded {res['updated']} of {res['total']} terminal job(s).")

    # Report the GPU jobs and whether util is now populated.
    gpu_rows = [r for r in JOBLOG.list_all() if _is_gpu_row(r)]
    if not gpu_rows:
        print("No GPU jobs found in the log.")
        return 0

    filled = sum(1 for r in gpu_rows if (r.get("gpu_util_mean") or "").strip())
    print(f"\nGPU jobs: {filled}/{len(gpu_rows)} now have GPU util "
          "(blank = sacct purged the job, so its samples are gone):")
    print(f"  {'jobid':>12}  {'state':<11}  {'gpu_util':>8}  {'gpu_mem':>10}")
    for r in gpu_rows:
        util = (r.get("gpu_util_mean") or "").strip() or "—"
        # ``gpu_mem_peak`` is a legacy live-sampler percentage.  Post-mortem
        # sacct memory is now kept explicitly in MB so the two units cannot
        # be confused when a job has both event samples and accounting data.
        mem = (r.get("gpu_mem_peak_mb") or "").strip()
        mem_s = f"{float(mem):.0f} MB" if mem else "—"
        util_s = f"{util}%" if util != "—" else "—"
        print(f"  {str(r.get('jobid', '')):>12}  "
              f"{(r.get('state') or '').strip():<11}  {util_s:>8}  {mem_s:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
