"""Periodic GPU + CPU utilisation sampling for a running FASRC job.

A :class:`ResourceSampler` runs a daemon thread that, every
``interval_s`` seconds, reads node CPU%% (via ``psutil``) and per-GPU
utilisation + memory (via ``nvidia-smi``) and emits one ``resource``
event through a :class:`~euclid_polish.observability.reporter.Reporter`.
The web UI folds those samples into a smoothed live gauge
(:mod:`euclid_polish.web.job_status`); the post-mortem summarises them
(mean / peak) into the job log.

Both probes degrade gracefully: a node without ``nvidia-smi`` (a
CPU-only job) simply omits the GPU figures, and any probe error is
swallowed so the sampler never disturbs the work it's measuring.
"""

from __future__ import annotations

import shutil
import subprocess
import threading

try:                                            # psutil is a project dep, but
    import psutil  # never let its absence break a
except Exception:                               # training run.
    psutil = None                               # type: ignore[assignment]

from euclid_polish.observability.reporter import Reporter

#: nvidia-smi query: per-GPU compute util %, memory util %, used/total MB.
_NVIDIA_SMI_QUERY = (
    "--query-gpu=utilization.gpu,utilization.memory,"
    "memory.used,memory.total"
)


def read_cpu_percent() -> float | None:
    """Node-wide CPU utilisation in percent since the previous call.

    ``psutil.cpu_percent(interval=None)`` measures over the gap since the
    last call, so the sampler's loop interval *is* the averaging window.
    The first call after process start returns ``0.0`` (no baseline) — the
    sampler primes it once before the loop so real samples aren't biased.
    """
    if psutil is None:
        return None
    try:
        return float(psutil.cpu_percent(interval=None))
    except Exception:
        return None


def read_gpu() -> tuple[float, float, float, float] | None:
    """Return ``(util%, mem_util%, mem_used_mb, mem_total_mb)`` or ``None``.

    Compute/mem-util are averaged across visible GPUs; memory is summed.
    ``None`` when ``nvidia-smi`` is absent (CPU-only node) or every row is
    unparseable (``[N/A]`` / ``[Not Supported]`` on some drivers).
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, _NVIDIA_SMI_QUERY, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return None
    return parse_nvidia_smi(proc.stdout)


def parse_nvidia_smi(text: str) -> tuple[float, float, float, float] | None:
    """Fold ``nvidia-smi --query-gpu`` CSV text into one averaged tuple.

    Split out from :func:`read_gpu` so it's unit-testable without a GPU.
    """
    utils, mem_utils, used, total = [], [], [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            u, mu, us, to = (float(parts[0]), float(parts[1]),
                             float(parts[2]), float(parts[3]))
        except ValueError:
            continue                            # '[N/A]' / '[Not Supported]'
        utils.append(u)
        mem_utils.append(mu)
        used.append(us)
        total.append(to)
    if not utils:
        return None
    n = len(utils)
    return (sum(utils) / n, sum(mem_utils) / n, sum(used), sum(total))


def sample_once() -> dict[str, float]:
    """One CPU+GPU probe → a ``resource`` event payload (may be partial)."""
    payload: dict[str, float] = {}
    cpu = read_cpu_percent()
    if cpu is not None:
        payload["cpu"] = round(cpu, 1)
    gpu = read_gpu()
    if gpu is not None:
        util, mem_util, used_mb, total_mb = gpu
        payload["gpu"] = round(util, 1)
        payload["gpu_mem"] = round(mem_util, 1)
        payload["gpu_mem_used_mb"] = round(used_mb, 1)
        payload["gpu_mem_total_mb"] = round(total_mb, 1)
    return payload


class ResourceSampler:
    """Background CPU+GPU sampler that emits ``resource`` events.

    Use as a context manager around the work you want measured::

        with ResourceSampler(reporter):
            trainer.train(...)

    or explicitly via :meth:`start` / :meth:`stop`. The thread is a
    daemon, so a hard process exit won't hang on it; ``stop`` joins it
    cleanly when the work finishes normally.
    """

    def __init__(self, reporter: Reporter, *, interval_s: float = 10.0) -> None:
        self.reporter = reporter
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        # Prime psutil so the first real sample reflects a true interval
        # rather than the 0.0 a cold cpu_percent() returns.
        read_cpu_percent()
        while not self._stop.wait(self.interval_s):
            try:
                payload = sample_once()
            except Exception:
                payload = {}
            if payload:
                self.reporter.sample_resources(payload)

    def start(self) -> ResourceSampler:
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._loop, name="resource-sampler", daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 2.0)
            self._thread = None

    def __enter__(self) -> ResourceSampler:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
