"""
FASRC pipeline-step abstraction for every web-submitted cluster job.

All jobs (data downloads, PSF extraction, TFRecord generation,
training, …) share the same submission shape — write an sbatch script,
``sbatch`` it through the ControlMaster SSH, register in
:class:`JobDB`. The only things that differ per job are:

  * default SLURM resources (gpu vs cpu, runtime, memory)
  * the remote script to invoke + its CLI flags

This module factors all of that into one abstract base class
(:class:`FASRCPipelineStep`) with one concrete subclass per job.
Adding a new step (e.g. "ingest JWST F814W") is then a 30-line
delta — no Flask, web template, or sbatch-template duplication.

The companion in :mod:`euclid_polish.web.fasrc_jobs` (the existing
training-only sbatch builder) remains untouched for backwards
compatibility; new code uses this module instead.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import secrets
import shlex
import textwrap
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from euclid_polish.config import Config
from euclid_polish.ensemble_registry import default_ensemble_dir, next_member_names
from euclid_polish.tng import selection as tng_selection
from euclid_polish.training.loss_names import KNEE_LOSS_MODES, LOSS_NAMES
from euclid_polish.web import fasrc_config
from euclid_polish.web.fasrc_jobs import _conda_activate_snippet
from euclid_polish.web.helpers import population_calibration

# ---------------------------------------------------------------------------
# Resource preset (subset of SLURM knobs the user can override per submit)
# ---------------------------------------------------------------------------

def _normalize_memory(mem: str) -> str:
    """Give a bare-number SLURM memory string a gigabyte unit.

    SLURM reads a unitless ``--mem`` as MEGABYTES, so a memory field set to
    ``"16"`` silently becomes 16 MB — far too little for Python+TensorFlow to
    even import, and the job is OOM-killed before it prints a single line
    (a real, baffling footgun: the wrapper banner appears, then nothing).
    On these nodes a bare number always means gigabytes, so append ``G`` when
    no letter unit is present. Values that already carry a unit (``64G``,
    ``64GB``, ``512M``) pass through unchanged; blank stays blank.
    """
    s = str(mem).strip()
    if not s or s[-1].isalpha():        # already has a K/M/G/T(B) unit letter
        return s
    return s + "G"


@dataclass
class StepResources:
    """One SLURM allocation profile.

    Mirrors the fields the user can edit in the web form. Numeric fields
    are kept as ints/strings so the form's POST values can flow through
    with minimal munging.
    """

    partition:   str = "shared"
    n_cpus:      int = 4
    n_gpus:      int = 0
    memory:      str = "16G"
    time_limit:  str = "2:00:00"

    def __post_init__(self) -> None:
        self.memory = _normalize_memory(self.memory)

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition":  self.partition,
            "n_cpus":     int(self.n_cpus),
            "n_gpus":     int(self.n_gpus),
            "memory":     self.memory,
            "time_limit": self.time_limit,
        }

    @classmethod
    def from_form(cls, form: dict[str, Any], defaults: StepResources) -> StepResources:
        """Build from a Flask form (string-valued), filling gaps with ``defaults``."""
        def _get(k: str, fallback: Any) -> Any:
            v = form.get(k)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                return fallback
            return v
        try:
            return cls(
                partition=str(_get("partition", defaults.partition)),
                n_cpus=int(_get("n_cpus", defaults.n_cpus)),
                n_gpus=int(_get("n_gpus", defaults.n_gpus)),
                memory=str(_get("memory", defaults.memory)),
                time_limit=str(_get("time_limit", defaults.time_limit)),
            )
        except (TypeError, ValueError) as e:
            raise ValueError(f"invalid resource field: {e}") from e

    @classmethod
    def from_form_strict(cls, form: dict[str, Any]) -> StepResources:
        """Build from a Flask form, *rejecting* blank resource fields.

        Used by the pipeline-step submit endpoint after the
        history-driven-defaults UI change: there is no implicit
        fallback to a step's defaults anymore. If the form has no
        precedent in the CSV log and the user typed nothing, the
        submit must error out rather than silently allocate something
        the user didn't consent to.

        ``partition``, ``memory``, ``time_limit`` are required strings;
        ``n_cpus`` and ``n_gpus`` are required integers (``n_gpus=0``
        is fine, but blank is not).
        """
        missing: list[str] = []
        def _required(k: str) -> str:
            v = form.get(k)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                missing.append(k)
                return ""
            return v if isinstance(v, str) else str(v)
        partition  = _required("partition")
        n_cpus_s   = _required("n_cpus")
        n_gpus_s   = _required("n_gpus")
        memory     = _required("memory")
        time_limit = _required("time_limit")
        if missing:
            raise ValueError(
                "missing required resource field(s): " + ", ".join(missing)
                + " — enter a value or pick a previous run to prefill"
            )
        try:
            return cls(
                partition=partition,
                n_cpus=int(n_cpus_s),
                n_gpus=int(n_gpus_s),
                memory=memory,
                time_limit=time_limit,
            )
        except (TypeError, ValueError) as e:
            raise ValueError(f"invalid resource field: {e}") from e


# ---------------------------------------------------------------------------
# Task-parameter schema (contract C5)
# ---------------------------------------------------------------------------

TaskParamType = Literal["int", "float", "str", "bool", "choice", "json"]
#: What an explicitly blank (empty/whitespace) submitted value means.
BlankPolicy = Literal["default", "unset"]
_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
_FALSE_WORDS = frozenset({"", "0", "false", "no", "off"})


class TaskParamError(ValueError):
    """A submitted task parameter does not satisfy its step's schema."""


@dataclass(frozen=True)
class TaskParam:
    """One step-specific knob its ``build_command`` reads from the form.

    ``/api/fasrc/steps/status`` publishes these so the SPA renders every
    step form generically; the submit route fills absent ones from
    ``default`` (:meth:`FASRCPipelineStep.fill_task_params`) and rejects
    values :meth:`parse` refuses. A ``default`` of ``None`` means "unset":
    the step then uses its own fallback (or omits the CLI flag).
    Resource fields and the knobs ``/config`` injects
    (``job_config.FASRC_STEP_PARAMS``) are deliberately not task params.

    ``blank`` says what an explicitly blank posted value means (a generic
    form posts ``""`` when the user clears a field): ``"default"`` — the
    same as absent, i.e. ``default`` — or ``"unset"``, for the few params
    whose help gives blank its own meaning ("blank = no cut"). A param
    whose ``default`` is ``None`` is unset when blank either way.
    """

    name: str
    type: TaskParamType
    default: Any = None
    help: str = ""
    min: float | None = None
    max: float | None = None
    choices: tuple[str, ...] | None = None
    required: bool = False
    #: ``False`` for values resolved afresh at every submit when left blank
    #: (fresh-entropy seeds): the job DB stores the resolved number, and
    #: prefilling it would silently replay the previous run's seeds, so
    #: :meth:`FASRCPipelineStep.last_task_params` reports them as ``None``.
    prefill: bool = True
    blank: BlankPolicy = "default"

    def to_dict(self) -> dict[str, Any]:
        """The C5 wire shape (optional keys only when set)."""
        out: dict[str, Any] = {"name": self.name, "type": self.type,
                               "default": self.default, "help": self.help}
        if self.min is not None:
            out["min"] = self.min
        if self.max is not None:
            out["max"] = self.max
        if self.choices is not None:
            out["choices"] = list(self.choices)
        if self.required:
            out["required"] = True
        return out

    def _bad(self, raw: Any, why: str) -> TaskParamError:
        return TaskParamError(f"{self.name}: {why} (got {raw!r})")

    def _check_range(self, value: float, raw: Any) -> None:
        if not math.isfinite(value):
            raise self._bad(raw, "must be a finite number")
        if self.min is not None and value < self.min:
            raise self._bad(raw, f"must be ≥ {self.min:g}")
        if self.max is not None and value > self.max:
            raise self._bad(raw, f"must be ≤ {self.max:g}")

    def parse(self, raw: Any) -> Any:
        """Typed value of one (string) form value; raises :class:`TaskParamError`."""
        text = raw.strip() if isinstance(raw, str) else raw
        if self.type == "bool":
            if isinstance(text, bool):
                return text
            word = str(text).strip().lower()
            if word in _TRUE_WORDS:
                return True
            if word in _FALSE_WORDS:
                return False
            raise self._bad(raw, "must be a boolean (1/0, true/false)")
        if self.type == "int":
            try:
                number = float(text)
            except (TypeError, ValueError):
                raise self._bad(raw, "must be an integer") from None
            if not number.is_integer():
                raise self._bad(raw, "must be an integer")
            self._check_range(number, raw)
            return int(number)
        if self.type == "float":
            try:
                number = float(text)
            except (TypeError, ValueError):
                raise self._bad(raw, "must be a number") from None
            self._check_range(number, raw)
            return number
        if self.type == "choice":
            value = str(text)
            if self.choices is None or value not in self.choices:
                raise self._bad(raw, f"must be one of {list(self.choices or ())}")
            return value
        if self.type == "json":
            if not isinstance(text, str):
                return text
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise self._bad(raw, f"is not valid JSON ({exc.msg})") from None
        return str(text)

    def form_value(self, value: Any) -> str:
        """Serialise a typed value back to the string a form would post."""
        if self.type == "bool":
            return "1" if value else "0"
        if self.type == "json":
            return json.dumps(value, separators=(",", ":"))
        if self.type == "float" and float(value).is_integer():
            return str(int(value))
        return str(value)


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

@dataclass
class FASRCPipelineStep(ABC):
    """One submittable FASRC job.

    Subclasses implement :meth:`build_command` to produce the Python
    command line that runs on the remote node and declare the knobs it
    reads as :attr:`task_params`. Everything else (SLURM header, conda
    setup, log layout, runtime banner) is shared via
    :meth:`build_sbatch_body`.
    """

    #: The step-specific knobs :meth:`build_command` reads (contract C5).
    task_params: ClassVar[tuple[TaskParam, ...]] = ()

    #: Stable id used in URLs (``/api/fasrc/steps/<step_id>/submit``).
    step_id:   str
    #: Human label shown in the UI.
    label:     str
    #: Default SLURM allocation. Overridable per-submission via the form.
    defaults:  StepResources
    #: True iff this job needs GPUs (used by the form to enable/disable the gpu count field).
    needs_gpu: bool = False
    #: If set, the CPU count is locked to this value at submit time and
    #: the UI hides the corresponding form field. Use when the underlying
    #: work is single-threaded and asking SLURM for extra cores just
    #: wastes the allocation.
    fixed_cpus: int | None = None
    #: GPU equivalent of ``fixed_cpus``. Ensemble members are ordinary
    #: single-device models, so allocating extra GPUs to one array task would
    #: reserve hardware the trainer never uses.
    fixed_gpus: int | None = None
    #: Simple, stable SLURM job name (what the user sees in squeue/sacct
    #: and the log filenames). Describes WHAT the job does, independent of
    #: any pipeline grouping, so it survives feature reshuffles. Falls back
    #: to ``step_id`` when unset; the timestamp suffix keeps log paths
    #: unique per submission.
    job_name:  str | None = None
    #: Conda env to activate for this step. ``None`` uses the cluster default
    #: (``cfg.conda_env_path``). Set to an absolute env path or a named env
    #: when a step needs an isolated environment.
    conda_env: str | None = None

    @abstractmethod
    def build_command(self, params: dict[str, Any]) -> list[str]:
        """Return the Python command (argv) to run inside the sbatch script.

        ``params`` is a flat dict of form values; subclasses pick out
        whatever step-specific knobs they need. Return value is a list
        of shell-safe tokens — :meth:`build_sbatch_body` joins them with
        spaces and shell-quotes individually.
        """

    def task_param_schema(self) -> list[dict[str, Any]]:
        """``task_params`` in the C5 wire shape."""
        return [param.to_dict() for param in self.task_params]

    def fill_task_params(self, form: Mapping[str, Any]) -> dict[str, Any]:
        """``form`` with absent and blank task params resolved per the schema.

        An absent value takes the schema ``default``; so does an explicitly
        blank (empty/whitespace) one, unless the param's ``blank`` policy is
        ``"unset"`` or it has no default — then it is stored as ``""`` (the
        step's own "unset" meaning, never a whitespace string ``int()``
        chokes on). A ``required`` param refuses a blank value, and an
        absent one without a default. Present values are validated
        (:meth:`TaskParam.parse`) and kept as posted, so the job DB/history
        keep the exact form strings. Raises :class:`TaskParamError` on the
        first invalid value.
        """
        out = dict(form)
        for param in self.task_params:
            if param.name in out and not _blank(out[param.name]):
                param.parse(out[param.name])
                continue
            posted_blank = param.name in out
            if param.required and (posted_blank or param.default is None):
                raise TaskParamError(f"{param.name}: is required")
            if param.default is None or (posted_blank and param.blank == "unset"):
                if posted_blank:
                    out[param.name] = ""
                continue
            out[param.name] = param.form_value(param.default)
        return out

    def last_task_params(
        self, history: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Typed task params of the newest ``COMPLETED`` run in ``history``.

        ``history`` is :meth:`JobLog.history_for_step` (newest first). Only
        schema names are returned; a blank stored value becomes ``None``, a
        ``prefill=False`` param (a seed drawn at submit when blank) is always
        ``None``, and a value the current schema refuses is dropped. ``None``
        when the step never completed.
        """
        by_name = {param.name: param for param in self.task_params}
        for row in history:
            if str(row.get("state") or "").strip().upper() != "COMPLETED":
                continue
            try:
                stored = json.loads(row.get("params_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(stored, dict):
                continue
            out: dict[str, Any] = {}
            for name, raw in stored.items():
                param = by_name.get(name)
                if param is None:
                    continue
                if _blank(raw) or not param.prefill:
                    out[name] = None
                    continue
                with contextlib.suppress(TaskParamError):
                    out[name] = param.parse(raw)
            return out
        return None

    def prepare_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Resolve submission-time values before rendering and logging.

        Most steps are already fully specified by their form values.  Array
        steps override this hook to freeze values that every task must share
        (member names and the base seed, for example).
        """
        return dict(params)

    def prepare_payload_files(
        self,
        params: dict[str, Any],
        *,
        job_name: str,
        relative_log_dir: str,
    ) -> dict[str, str]:
        """Stage large immutable inputs beside the generated job script.

        Implementations may replace large in-memory parameters with short
        repo-relative file paths in ``params``.  The submission helper writes
        the returned path/content pairs over SSH before calling ``sbatch``.
        """
        del params, job_name, relative_log_dir
        return {}

    def array_shape(self, params: dict[str, Any]) -> tuple[int, int] | None:
        """Return ``(task_count, max_parallel)`` for an array submission."""
        return None

    # ------------------------------------------------------------------ #

    #: Subclasses override this class-level constant to control where
    #: their log files land. Pipeline steps live under
    #: ``logs/pipeline``; legacy training presets live under
    #: ``logs/jobs``. Declared as ``ClassVar`` so the dataclass
    #: machinery doesn't promote it to a per-instance field.
    log_dir_prefix:  ClassVar[str] = "logs/pipeline"

    def banner_line(self, label: str) -> str:
        """First echo line of the sbatch banner.

        Overrideable so each step family stays grep-searchable in the logs
        (``Pipeline step:`` vs ``Web-submitted job:`` etc.).
        """
        return f"Pipeline step: {self.step_id} — {label}"

    def build_sbatch_body(
        self,
        *,
        params: dict[str, Any],
        resources: StepResources,
        cfg: fasrc_config.FasrcConfig,
        label: str,
        relative_log_dir: str | None = None,
    ) -> dict[str, Any]:
        """Render the full sbatch script + the relative log paths.

        Returns ``{"body": str, "script": rel, "out": rel, "err": rel,
        "name": str}``. Thin wrapper that picks the log dir and job-name
        shape from the step and delegates to :func:`render_sbatch_body`.
        """
        log_dir   = relative_log_dir or self.log_dir_prefix
        ts        = time.strftime("%Y%m%d-%H%M%S")
        job_name  = f"{self.job_name or self.step_id}-{ts}"
        prepared = self.prepare_params(params)
        payload_files = self.prepare_payload_files(
            prepared,
            job_name=job_name,
            relative_log_dir=log_dir,
        )
        built = render_sbatch_body(
            job_name=job_name,
            relative_log_dir=log_dir,
            resources=resources,
            cfg=cfg,
            label=label,
            cmd_argv=self.build_command(prepared),
            banner_line=self.banner_line(label),
            step_id=self.step_id,
            conda_env_path=self.conda_env,
            array_shape=self.array_shape(prepared),
        )
        built["params"] = prepared
        built["payload_files"] = payload_files
        return built


# ---------------------------------------------------------------------------
# Script template — single source of truth shared by every step
# ---------------------------------------------------------------------------

def render_sbatch_body(
    *,
    job_name:         str,
    relative_log_dir: str,
    resources:        StepResources,
    cfg:              fasrc_config.FasrcConfig,
    label:            str,
    cmd_argv:         list[str],
    banner_line:      str,
    step_id:          str | None = None,
    conda_env_path:   str | None = None,
    array_shape:      tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Render an sbatch script body + the relative log paths.

    Every FASRC job submitted from the UI runs through here — there is no
    other place a SLURM template should live.

    Parameters
    ----------
    cmd_argv :
        Tokens for the ``python -u …`` invocation. Each token is
        shell-quoted individually.
    banner_line :
        First echoed line inside the script (after the ``====`` rule).
        Caller picks the wording so log greps stay stable per step family.
    step_id :
        If given, an ``STEP_ID=…`` echo is emitted at the tail of the
        script.

    Returns
    -------
    dict
        ``{"body": str, "script": rel, "out": rel, "err": rel, "name": str}``
    """
    script_rel = f"{relative_log_dir}/{job_name}.sh"
    array_count = int(array_shape[0]) if array_shape else 0
    array_parallel = int(array_shape[1]) if array_shape else 0
    array_suffix = "-%A_%a" if array_count > 1 else ""
    out_rel    = f"{relative_log_dir}/{job_name}{array_suffix}.out"
    err_rel    = f"{relative_log_dir}/{job_name}{array_suffix}.err"
    # JSONL stream of structured progress events. Producer:
    # :class:`euclid_polish.observability.Reporter`. Consumer: the
    # /api/fasrc/jobs/<jobid>/status endpoint.
    events_rel = f"{relative_log_dir}/{job_name}{array_suffix}.events"
    exit_rel = f"{relative_log_dir}/{job_name}{array_suffix}.exit"
    runtime_array_suffix = (
        "-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        if array_count > 1 else ""
    )
    events_runtime_rel = (
        f"{relative_log_dir}/{job_name}{runtime_array_suffix}.events"
    )
    exit_runtime_rel = f"{relative_log_dir}/{job_name}{runtime_array_suffix}.exit"

    # 14 spaces of leading indent on continuation lines — must equal
    # or exceed the heredoc body's 12-space dedent baseline, otherwise
    # textwrap.dedent strips a smaller common prefix and the whole
    # script gets a residual indent that breaks bash syntax.
    cmd_line = " \\\n              ".join(shlex.quote(a) for a in cmd_argv)

    n_gpus = int(resources.n_gpus)
    # IMPORTANT — the trailing 12 spaces on gres_line are required.
    #
    # The template uses ``textwrap.dedent`` with a 12-space baseline;
    # ``gres_line`` is inlined on the same template line as the next
    # ``#SBATCH`` directive. If we wrote just ``"#SBATCH --gres=...\n"``
    # the newline would break out of the f-string's indentation: the
    # line *after* the gres directive would land at column 0, which
    # makes ``textwrap.dedent`` find a common prefix of "" and strip
    # nothing — the resulting script's ``#!/bin/bash`` keeps 12 leading
    # spaces and sbatch rejects it with "first line must start with #!".
    # Padding the newline with 12 spaces re-aligns the next line back
    # onto the template's indent so dedent finds a uniform common
    # prefix again.
    gres_line = (
        f"#SBATCH --gres=gpu:{n_gpus}\n        " if n_gpus > 0 else ""
    )
    array_line = (
        f"#SBATCH --array=0-{array_count - 1}%{array_parallel}\n        "
        if array_count > 1 else ""
    )
    # Sanitize anything embedded inside a double-quoted ``echo`` line —
    # newlines would split the echo, and single quotes are stripped to
    # match what users see in the rendered log banner. The caller's
    # ``banner_line`` already includes the label so we sanitize it
    # in one place.
    safe_banner = (banner_line.replace("\n", " ")
                              .replace('"', "")
                              .replace("'", "")[:240])
    step_id_echo = (
        f'\n        echo "STEP_ID={step_id}"' if step_id else ""
    )

    _conda_block = _conda_activate_snippet(
        conda_env_path or cfg.conda_env_path,
        load_cuda=n_gpus > 0,
    )
    cuda_visibility = (
        'export CUDA_VISIBLE_DEVICES=""' if n_gpus == 0 else ""
    )
    body = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={shlex.quote(job_name)}
        #SBATCH --partition={shlex.quote(resources.partition)}
        {gres_line}{array_line}#SBATCH --cpus-per-task={int(resources.n_cpus)}
        #SBATCH --mem={resources.memory}
        #SBATCH --time={resources.time_limit}
        #SBATCH --output={out_rel}
        #SBATCH --error={err_rel}

        set -euo pipefail
        cd "$SLURM_SUBMIT_DIR"
        export PYTHONPATH="$(pwd):${{PYTHONPATH:-}}"
        mkdir -p {relative_log_dir}
        __FASRC_EXIT_PATH__={exit_runtime_rel}
        trap '__FASRC_RC__=$?; printf "%s\\n" "$__FASRC_RC__" > "$__FASRC_EXIT_PATH__"' EXIT

        echo "============================================================"
        echo "{safe_banner}"
        echo "Job id:   ${{SLURM_JOB_ID:-local}}"
        echo "Host:     $(hostname)"
        echo "Started:  $(date)"
        echo "Workdir:  $(pwd)"
        echo "Resources: cpus={int(resources.n_cpus)} gpus={n_gpus} "\\
             "mem={resources.memory} time={resources.time_limit} "\\
             "partition={resources.partition}"
        echo "============================================================"
        __FASRC_T0__=$(date +%s)

        export EUCLID_POLISH_DATA_DIR={shlex.quote(cfg.data_dir)}
        export EUCLID_POLISH_CKPT_DIR={shlex.quote(cfg.ckpt_dir)}
        # ``Reporter.from_env()`` reads this to open the per-job
        # structured events stream.
        export EUCLID_POLISH_EVENTS_PATH={events_runtime_rel}
        # The WebUI bar is driven by the Reporter, so silence every tqdm
        # progress bar in the job — otherwise they flood the .err log with
        # redundant ASCII frames. ``tqdm.write`` status lines (.out) stay.
        export TQDM_DISABLE=1
        mkdir -p "$EUCLID_POLISH_DATA_DIR" "$EUCLID_POLISH_CKPT_DIR"

        module purge
        {cuda_visibility}
        __CONDA_BLOCK__
        # ``module load`` puts the system (old) libstdc++ on LD_LIBRARY_PATH;
        # prepend the activated env's lib so its newer libstdc++ wins —
        # otherwise numpy/torch C-extensions fail with "GLIBCXX_3.4.xx not
        # found by /lib64/libstdc++.so.6".
        export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"

        echo "Python:  $(which python)"
        python -u {cmd_line}

        __FASRC_T1__=$(date +%s)
        __FASRC_RUNTIME__=$((__FASRC_T1__ - __FASRC_T0__))
        echo "============================================================"
        echo "Finished: $(date)"
        echo "RUNTIME_SECONDS=${{__FASRC_RUNTIME__}}"{step_id_echo}
        echo "============================================================"
    """).replace("__CONDA_BLOCK__", _conda_block)
    # Repo-relative entry script (e.g. "scripts/train_ensemble.py") — the
    # submitter checks it exists on FASRC BEFORE sbatching, so a repo that
    # hasn't been pulled fails with an actionable message instead of a
    # cryptic ENOENT after the job starts (and burns its queue slot).
    entry = (cmd_argv[0]
             if cmd_argv and not cmd_argv[0].startswith("/")
             and cmd_argv[0].endswith(".py") else None)
    return {
        "body":   body,
        "script": script_rel,
        "out":    out_rel,
        "err":    err_rel,
        "events": events_rel,
        "exit":   exit_rel,
        "name":   job_name,
        "entry":  entry,
        "array_count": array_count,
        "array_parallelism": array_parallel,
    }


# ---------------------------------------------------------------------------
# Concrete steps
# ---------------------------------------------------------------------------

class VISNoiseSampleStep(FASRCPipelineStep):
    """Download independent source-maskable VIS samples across Q1 support."""

    task_params = (
        TaskParam("n_clusters", "int", 44, "Star-footprint k-means clusters "
                  "(one sampling region each).", min=1),
        TaskParam("samples_per_cluster", "int", 1,
                  "VIS samples drawn per cluster.", min=1),
        TaskParam("vis_pixels", "int", 2560, "VIS sample side (0.1″ pixels).",
                  min=16),
        TaskParam("workers", "int", 1, "Parallel download workers.", min=1),
        TaskParam("seed", "int", 42, "Sampling seed."),
        TaskParam("source_release", "str", "Q1_R1", "Euclid archive release."),
        TaskParam("star_support_csv", "str", None,
                  "Star catalogue defining the support (blank = default)."),
        TaskParam("sampling_manifest", "str", None,
                  "Sampling-plan manifest path (blank = default)."),
        TaskParam("output_dir", "str", None, "Output directory (blank = default)."),
        TaskParam("regenerate_catalog", "bool", False,
                  "Rebuild the sampling plan instead of reusing it."),
    )

    def __init__(self):
        super().__init__(
            step_id="vis_noise_sample",
            label="Sample real VIS noise fields (star-footprint k-means)",
            job_name="vis-noise-samples",
            defaults=StepResources(
                partition="shared", n_cpus=1, n_gpus=0,
                memory="16G", time_limit="4:00:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: dict[str, Any]) -> list[str]:
        cmd = [
            "scripts/fasrc_download_euclid_sky_cutouts.py",
            "--sampling-mode", "star-support",
            "--n-clusters", str(int(params.get("n_clusters", 44) or 44)),
            "--samples-per-cluster", str(int(
                params.get("samples_per_cluster", 1) or 1
            )),
            "--vis-pixels", str(int(params.get("vis_pixels", 2560) or 2560)),
            "--workers", str(max(1, int(
                params.get("workers", 1) or 1
            ))),
            "--seed", str(int(params.get("seed", 42) or 42)),
            "--source-release", str(
                params.get("source_release", "Q1_R1") or "Q1_R1"
            ),
        ]
        for key, flag in (
            ("star_support_csv", "--star-support-csv"),
            ("sampling_manifest", "--sampling-manifest"),
            ("output_dir", "--output-dir"),
        ):
            value = str(params.get(key, "") or "").strip()
            if value:
                cmd += [flag, value]
        if str(params.get("regenerate_catalog", "")).strip().lower() in (
            "1", "true", "yes", "on",
        ):
            cmd.append("--regenerate-catalog")
        return cmd


class ArchiveFieldSampleStep(FASRCPipelineStep):
    """Derive compact matched four-band fields from the frozen VIS parents."""

    task_params = (
        TaskParam("workers", "int", 1, "Parallel download workers.", min=1),
        TaskParam("source_release", "str", "Q1_R1", "Euclid archive release."),
        TaskParam("source_sampling_manifest", "str", None,
                  "Frozen VIS sampling plan (blank = default)."),
        TaskParam("sampling_manifest", "str", None,
                  "Archive-field manifest path (blank = default)."),
        TaskParam("output_dir", "str", None, "Output directory (blank = default)."),
        TaskParam("regenerate_catalog", "bool", False,
                  "Rebuild the field plan instead of reusing it."),
        TaskParam("force_redownload", "bool", False,
                  "Re-download every bundle (needs its own confirmation)."),
    )

    def __init__(self):
        super().__init__(
            step_id="archive_field_sample",
            label="Download matched multipoint Euclid archive fields (44 × 5)",
            job_name="archive-fields",
            defaults=StepResources(
                partition="shared", n_cpus=1, n_gpus=0,
                memory="8G", time_limit="4:00:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: dict[str, Any]) -> list[str]:
        cmd = [
            "scripts/fasrc_download_euclid_sky_cutouts.py",
            "--sampling-mode", "archive-fields",
            "--vis-pixels", "256",
            "--workers", str(max(1, int(params.get("workers", 1) or 1))),
            "--source-release", str(
                params.get("source_release", "Q1_R1") or "Q1_R1"
            ),
        ]
        for key, flag in (
            ("source_sampling_manifest", "--source-sampling-manifest"),
            ("sampling_manifest", "--sampling-manifest"),
            ("output_dir", "--output-dir"),
        ):
            value = str(params.get(key, "") or "").strip()
            if value:
                cmd += [flag, value]
        if str(params.get("regenerate_catalog", "")).strip().lower() in (
            "1", "true", "yes", "on",
        ):
            cmd.append("--regenerate-catalog")
        if str(params.get("force_redownload", "")).strip().lower() in (
            "1", "true", "yes", "on",
        ):
            cmd.append("--force-redownload")
        return cmd


#: Brightest-N of the last real catalogue run — the schema default AND the
#: ``build_command`` fallback (the old 200 silently overwrote the catalogue).
EUCLID_QUERY_NUM_STARS = 10_000


class EuclidQueryStep(FASRCPipelineStep):
    # Defaults are the last real catalogue run (10,000 stars, 18 ≤ VIS ≤ 19,
    # S/N ≥ 50). A blank cut means "no cut"; a blank count is the default.
    task_params = (
        TaskParam("num_stars", "int", EUCLID_QUERY_NUM_STARS,
                  "Brightest N stars to keep.", min=1),
        TaskParam("magnitude_min", "float", 18.0,
                  "Brightest VIS magnitude kept (blank = no bright cut).",
                  blank="unset"),
        TaskParam("magnitude_limit", "float", 19.0,
                  "Faintest VIS magnitude kept (blank = no faint cut).",
                  blank="unset"),
        TaskParam("snr_min", "float", 50.0,
                  "Minimum VIS S/N (blank = no cut).", min=0, blank="unset"),
    )

    def __init__(self):
        super().__init__(
            step_id="euclid_query",
            label="Query Euclid catalog (brightest N)",
            job_name="star-catalog",
            defaults=StepResources(
                partition="shared", n_cpus=1, n_gpus=0,
                memory="4G", time_limit="30:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: dict[str, Any]) -> list[str]:
        num_stars = int(str(params.get("num_stars") or "").strip()
                        or EUCLID_QUERY_NUM_STARS)
        cmd = ["scripts/query_brightest_stars.py", "--num-stars", str(num_stars)]
        mag_min = str(params.get("magnitude_min", "")).strip()
        if mag_min:
            cmd += ["--magnitude-min", f"{float(mag_min):g}"]
        mag_lim = str(params.get("magnitude_limit", "")).strip()
        if mag_lim:
            cmd += ["--magnitude-limit", f"{float(mag_lim):g}"]
        snr_min = str(params.get("snr_min", "")).strip()
        if snr_min:
            cmd += ["--snr-min", f"{float(snr_min):g}"]
        return cmd


class EuclidVerifyPhotometryStep(FASRCPipelineStep):
    task_params = (
        TaskParam("n", "int", 40, "Stars checked.", min=1),
        TaskParam("size", "int", 256, "Cutout side (VIS pixels).", min=16),
    )

    def __init__(self):
        super().__init__(
            step_id="euclid_verify_photometry",
            label="Verify photometry scale",
            job_name="verify-photometry",
            defaults=StepResources(
                partition="shared", n_cpus=1, n_gpus=0,
                memory="8G", time_limit="30:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: dict[str, Any]) -> list[str]:
        n    = int(params.get("n", 40) or 40)
        size = int(params.get("size", 256) or 256)
        return ["scripts/verify_star_photometry.py",
                "--n", str(n), "--size", str(size)]


class EuclidCutoutDownloadStep(FASRCPipelineStep):
    # ``vis_pixels`` comes from /config (job_config.FASRC_STEP_PARAMS).
    task_params = (
        TaskParam("workers", "int", 8, "Parallel download workers.", min=1),
    )

    def __init__(self):
        super().__init__(
            step_id="download_euclid_cutouts",
            label="Download Euclid star cutouts (all bands)",
            job_name="star-cutouts",
            defaults=StepResources(
                partition="shared", n_cpus=8, n_gpus=0,
                memory="16G", time_limit="2:00:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: dict[str, Any]) -> list[str]:
        vis_pixels = int(params.get("vis_pixels", 512))
        workers    = int(str(params.get("workers") or "").strip() or 8)
        return [
            "scripts/download_all_bands.py",
            "--vis-pixels", str(vis_pixels),
            "--workers",    str(workers),
        ]


class EuclidPSFExtractStep(FASRCPipelineStep):
    # ``vis_pixels`` / ``output_size`` come from /config; workers = CPUs.
    task_params = (
        TaskParam("stars_per_psf", "int", Config.PSF_STARS_PER_CLUSTER,
                  "Target stars per spatial ePSF cluster.", min=1),
        TaskParam("min_stars_per_psf", "int", Config.PSF_MIN_STARS_PER_CLUSTER,
                  "Smaller clusters merge into a neighbour.", min=1),
        TaskParam("num_stars", "int", 0,
                  "Cap on stars considered per band (0 = every good cutout).",
                  min=0),
    )

    def __init__(self):
        super().__init__(
            step_id="extract_euclid_psf",
            label="Extract Euclid ePSFs (all 4 bands)",
            job_name="euclid-psf",
            defaults=StepResources(
                partition="shared", n_cpus=8, n_gpus=0,
                # Validation retains paths only. Peak star memory is one
                # ~400-star cluster per CPU worker while its ePSF is built.
                # Scale memory with the CPU count and cutout size.
                memory="48G", time_limit="6:00:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: dict[str, Any]) -> list[str]:
        vis_pixels    = int(params.get("vis_pixels", 512))
        stars_per_psf = int(str(params.get("stars_per_psf") or "").strip()
                            or Config.PSF_STARS_PER_CLUSTER)
        # Minimum cluster size: clusters smaller than this are merged into a
        # neighbour, so no ePSF is built from fewer than ``min_stars`` stars.
        min_stars = int(params.get(
            "min_stars_per_psf", Config.PSF_MIN_STARS_PER_CLUSTER,
        ) or Config.PSF_MIN_STARS_PER_CLUSTER)
        # Use every allocated CPU to build cluster PSFs in parallel.
        n_cpus = int(params.get("n_cpus", 8) or 8)
        cmd = [
            "scripts/extract_all_band_psfs.py",
            "--vis-pixels",        str(vis_pixels),
            "--stars-per-psf",     str(stars_per_psf),
            "--min-stars-per-psf", str(min_stars),
            "--max-procs",         str(n_cpus),
        ]
        # Optional cap on total stars considered per band (blank/0 → all good
        # cutouts, then clustered into groups of stars_per_psf).
        num_stars = int(params.get("num_stars", 0) or 0)
        if num_stars > 0:
            cmd += ["--num-stars", str(num_stars)]
        # Optional explicit final ePSF size (oversampled px); 0/blank →
        # photutils' default (cutout_size × oversampling + 1).
        output_size = int(params.get("output_size", 0) or 0)
        if output_size > 0:
            cmd += ["--output-size", str(output_size)]
        return cmd


class PSFRotationPoolStep(FASRCPipelineStep):
    """Precompute the pre-rotated PSF kernel pools (one job, all 4 bands).

    Shells out to ``scripts/pregenerate_psf_rotations.py``: for every cluster
    ePSF, K random telescope-roll rotations (one shared angle table across
    bands, so a pool index is one physical pointing in all four channels) are
    precomputed and streamed to ``euclid_psf_rotpool_<BAND>.fits`` next to
    the source ePSFs. Amortises the ~92 ms/kernel order-3 rotation to a
    one-time build; the pools feed roll augmentation in generation and the
    per-member PSF bagging of on-the-fly training. Re-run after every ePSF
    re-extraction (the pool is a derivative of the cluster kernels).
    """

    task_params = (
        TaskParam("rotations", "int", 12, "Random roll angles per cluster ePSF.",
                  min=1),
        TaskParam("seed", "int", None, "Angle-table seed (blank = random).",
                  prefill=False),
        TaskParam("crop", "int", 0, "Crop kernels to this side (0 = full).", min=0),
    )

    def __init__(self):
        super().__init__(
            step_id="psf_rotation_pool",
            label="Pre-rotate ePSF kernel pools (all 4 bands)",
            job_name="psf-rotpool",
            defaults=StepResources(
                partition="shared", n_cpus=16, n_gpus=0,
                # Rotation workers each hold one 511² kernel; the writer
                # batches one cluster (K+1 kernels) at a time. The source
                # sets (~1.5 GB for 4×356 kernels) dominate.
                memory="16G", time_limit="2:00:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: dict[str, Any]) -> list[str]:
        rotations = int(params.get("rotations", 12) or 12)
        cmd = [
            "scripts/pregenerate_psf_rotations.py",
            "--rotations", str(rotations),
            # Rotate on every allocated CPU.
            "--workers", str(int(params.get("n_cpus", 16) or 16)),
        ]
        seed = str(params.get("seed", "")).strip()
        if seed:
            with contextlib.suppress(ValueError):
                cmd += ["--seed", str(int(seed))]
        crop = int(params.get("crop", 0) or 0)
        if crop > 0:
            cmd += ["--crop", str(crop)]
        return cmd


class TngSkirtAtlasDownloadStep(FASRCPipelineStep):
    """Bulk-download the whole IllustrisTNG TNG50-1 SKIRT atlas (~1153 galaxies).

    Each galaxy is rendered as Euclid VIS + NISP (Y/J/H) FITS from 5
    orientations; we keep only the dusty frames (20 FITS/galaxy) into
    ``$DATA_DIR/tng_skirt/<subhalo_id>/``. Network-I/O bound, so it runs one
    download thread per allocated CPU. The TNG API token is read on the node
    from ``$TNG_API_KEY`` or ``~/.tng_api_key`` — never from the form.
    """

    task_params = (
        TaskParam("workers", "int", 0,
                  "Download workers (0 = one per allocated CPU).", min=0),
        TaskParam("executor", "choice", "process",
                  "Parallelism backend.", choices=("process", "thread")),
        TaskParam("limit", "int", 0, "Cap on galaxies (0 = all ~1153).", min=0),
        TaskParam("keep_archive", "bool", False,
                  "Keep each .tar.gz beside its FITS."),
        TaskParam("force", "bool", False,
                  "Re-download everything, ignoring .done markers."),
    )

    def __init__(self):
        super().__init__(
            step_id="download_tng_skirt",
            label="Download TNG50 SKIRT atlas (all galaxies, Euclid bands)",
            job_name="tng-atlas",
            defaults=StepResources(
                # ~1153 multi-GB tarballs streamed, filtered to ~20 FITS each.
                # I/O bound → many threads; generous wall-clock for the full set
                # (re-submittable — finished galaxies carry a .done marker).
                partition="shared", n_cpus=16, n_gpus=0,
                memory="32G", time_limit="1-00:00:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: dict[str, Any]) -> list[str]:
        # Workers default to one download thread per allocated CPU, but the
        # form can set them independently: the work is network-I/O bound, so a
        # single CPU drives many concurrent transfers — running more workers
        # than CPUs is the cheaper way to saturate the link without allocating
        # extra cores. Blank/0 in the form → fall back to the CPU count.
        n_cpus  = int(params.get("n_cpus", 16) or 16)
        workers = int(params.get("workers", 0) or 0)
        if workers <= 0:
            workers = n_cpus
        cmd = [
            "scripts/fasrc_download_tng_skirt_atlas.py",
            "--workers", str(max(1, workers)),
        ]
        # Parallelism backend: 'process' (default) is true multi-core; 'thread'
        # is lighter but GIL-capped on extraction. Only emit a recognised value.
        executor = str(params.get("executor", "process")).strip().lower()
        if executor in ("process", "thread"):
            cmd += ["--executor", executor]
        # Optional cap on galaxies (blank/0 → all ~1153; small N for a test).
        limit = int(params.get("limit", 0) or 0)
        if limit > 0:
            cmd += ["--limit", str(limit)]
        # Keep each .tar.gz alongside its FITS (default: delete to save disk).
        if str(params.get("keep_archive", "")).strip() in (
                "1", "true", "True", "on", "yes"):
            cmd += ["--keep-archive"]
        # Override: re-download everything, ignoring .done markers. Without
        # it a galaxy still re-downloads when its marker's FITS are missing
        # (integrity check), so a swept atlas self-heals either way.
        if str(params.get("force", "")).strip().lower() in (
                "1", "true", "yes", "on"):
            cmd += ["--force"]
        return cmd


class MeasureTngRadiiStep(FASRCPipelineStep):
    """Measure the centered VIS half-light radius of every atlas frame."""

    task_params = (
        TaskParam("tng_dir", "str", None, "TNG SKIRT atlas dir (blank = default)."),
        TaskParam("tng_properties", "str", None,
                  "TNG properties CSV (blank = default)."),
        TaskParam("tng_radius_manifest", "str", None,
                  "Output radius manifest (blank = default)."),
        TaskParam("tng_parameter_summary", "str", None,
                  "Output parameter summary (blank = default)."),
    )

    def __init__(self):
        super().__init__(
            step_id="measure_tng_radii",
            label="Measure TNG effective radii (all galaxies/orientations)",
            job_name="tng-radii",
            defaults=StepResources(
                partition="shared", n_cpus=4, n_gpus=0,
                memory="16G", time_limit="4:00:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: dict[str, Any]) -> list[str]:
        cmd = ["scripts/measure_tng_radii.py"]
        for key, flag in (("tng_dir", "--tng-dir"),
                          ("tng_properties", "--properties"),
                          ("tng_radius_manifest", "--output"),
                          ("tng_parameter_summary", "--summary")):
            value = str(params.get(key, "") or "").strip()
            if value:
                cmd += [flag, value]
        try:
            workers = int(params.get("n_cpus") or self.defaults.n_cpus)
        except (TypeError, ValueError):
            workers = self.defaults.n_cpus
        cmd += ["--workers", str(max(1, workers))]
        return cmd


# Galaxy-selection modes for the grid/stack. The pick happens locally (where
# the histogram property cache lives), and the chosen ids ride to the FASRC job
# via --ids/--id.
_TNG_MODES = ("random", "most_massive", "least_massive", "most_star_forming",
              "least_star_forming", "biggest_radius", "smallest_radius")


def _tng_mode(params: dict[str, Any]) -> str:
    m = str(params.get("mode", "random")).strip().lower()
    return m if m in _TNG_MODES else "random"


def _tng_temperature(params: dict[str, Any]) -> float:
    try:
        t = float(params.get("temperature", 0.3))
    except (TypeError, ValueError):
        t = 0.3
    return min(1.0, max(0.0, t))


def _tng_note(mode: str, temperature: float) -> str:
    pretty = mode.replace("_", " ")
    return pretty if mode == "random" else f"{pretty} · T={temperature:.2f}"


def _tng_select(mode: str, n: int, temperature: float) -> list[str]:
    """Select ``n`` galaxy ids locally by mode (empty if no property cache).

    Looked up on the module at call time so tests can patch
    ``euclid_polish.tng.selection.pick_by_mode``."""
    return tng_selection.pick_by_mode(mode, n, temperature=temperature)


class TngGridStep(FASRCPipelineStep):
    """Render the 5×5 (galaxies × viewpoints) image grid as a CPU job →
    ``_infographics/grid.png``. Band ∈ VIS/Y/J/H/RGB; downsample ×1/×2/×4. The
    5 galaxies are chosen by ``mode`` (random, or the most/least extreme in
    stellar mass / SFR / radius) with a temperature-weighted draw."""

    task_params = (
        TaskParam("band", "choice", "VIS", "Band (RGB = Lupton colour).",
                  choices=("VIS", "Y", "J", "H", "RGB")),
        TaskParam("downsample", "choice", "1", "Downsample factor.",
                  choices=("1", "2", "4")),
        TaskParam("mode", "choice", "random", "Galaxy selection.",
                  choices=_TNG_MODES),
        TaskParam("temperature", "float", 0.3,
                  "Selection temperature (0 = strict extremes).", min=0, max=1),
    )

    def __init__(self):
        super().__init__(
            step_id="tng_grid",
            label="TNG infographic — 5×5 image grid",
            job_name="tng-grid",
            defaults=StepResources(
                # RGB loads up to 5×5×3 frames of 1600² (~0.8 GB) + Lupton
                # intermediates; 16 G is comfortable headroom.
                partition="shared", n_cpus=2, n_gpus=0,
                memory="16G", time_limit="0:30:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: dict[str, Any]) -> list[str]:
        band = str(params.get("band", "VIS") or "VIS").upper()
        if band not in ("VIS", "Y", "J", "H", "RGB"):
            band = "VIS"
        downsample = int(params.get("downsample", 1) or 1)
        if downsample not in (1, 2, 4):
            downsample = 1
        mode = _tng_mode(params)
        temperature = _tng_temperature(params)
        cmd = [
            "scripts/fasrc_tng_infographic.py",
            "--mode", "grid", "--save",
            "--band", band,
            "--downsample", str(downsample),
        ]
        ids = _tng_select(mode, 5, temperature)
        if ids:
            cmd += ["--ids", ",".join(ids), "--note", _tng_note(mode, temperature)]
        else:
            # No local property cache yet → random pick on the node. (Render the
            # histograms first to enable the most/least modes.)
            cmd += ["--seed", "-1"]
        return cmd


class TngStackStep(FASRCPipelineStep):
    """Bundle one band's 5 viewpoint frames of a galaxy into a multi-extension
    FITS → ``_infographics/stack.fits``. The galaxy is an explicit id, else
    chosen by ``mode`` (temperature-weighted) like the grid."""

    task_params = (
        TaskParam("band", "choice", "VIS", "Band.", choices=("VIS", "Y", "J", "H")),
        TaskParam("galaxy_id", "str", None,
                  "Subhalo id (blank = pick by mode)."),
        TaskParam("mode", "choice", "random", "Galaxy selection.",
                  choices=_TNG_MODES),
        TaskParam("temperature", "float", 0.3,
                  "Selection temperature (0 = strict extremes).", min=0, max=1),
    )

    def __init__(self):
        super().__init__(
            step_id="tng_stack",
            label="TNG infographic — stacked FITS (5 viewpoints)",
            job_name="tng-stack",
            defaults=StepResources(
                partition="shared", n_cpus=1, n_gpus=0,
                memory="8G", time_limit="0:20:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: dict[str, Any]) -> list[str]:
        band = str(params.get("band", "VIS") or "VIS").upper()
        if band not in ("VIS", "Y", "J", "H"):
            band = "VIS"
        cmd = [
            "scripts/fasrc_tng_infographic.py",
            "--mode", "stack", "--save",
            "--band", band,
        ]
        gid = str(params.get("galaxy_id", "")).strip()
        if gid:
            cmd += ["--id", gid]
            return cmd
        # No explicit id → pick one by mode (temperature-weighted).
        mode = _tng_mode(params)
        temperature = _tng_temperature(params)
        ids = _tng_select(mode, 1, temperature)
        if ids:
            cmd += ["--id", ids[0]]
        else:
            cmd += ["--seed", "-1"]      # no cache → random pick on the node
        return cmd


class PosterCutoutStep(FASRCPipelineStep):
    """Generate one random star, lens, TNG galaxy, or full field as
    a clean 4-band HR FITS → ``_poster/poster_cutout.fits`` (+ a preview PNG).
    For the poster — the idealised, PSF-free, noise-free ground-truth object.
    TNG-backed modes need the downloaded SKIRT atlas, which is why the job runs
    on the node. Blank seed re-rolls each submit."""

    task_params = (
        TaskParam("mode", "choice", "tng", "Object kind.",
                  choices=("star", "lens", "tng", "field")),
        TaskParam("image_size", "int", 0, "HR side in pixels (0 = default).",
                  min=0),
        TaskParam("seed", "int", None, "Seed (blank = re-roll each submit).",
                  prefill=False),
    )

    def __init__(self):
        super().__init__(
            step_id="poster_cutout",
            label="Poster — random object cutout (clean 4-band FITS)",
            job_name="poster-cutout",
            defaults=StepResources(
                partition="shared", n_cpus=2, n_gpus=0,
                memory="48G", time_limit="0:45:00",
            ),
            needs_gpu=False,
        )

    def prepare_params(self, params: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(params)
        mode = str(prepared.get("mode", "tng") or "tng").lower()
        if mode not in ("star", "lens", "tng", "field"):
            mode = "tng"
        prepared["mode"] = mode
        if mode in ("star", "field"):
            stars = population_calibration.active_star()
            if not stars:
                raise ValueError(
                    "activate a valid Gaia+Euclid stellar calibration before "
                    f"rendering a poster {mode}"
                )
            prepared["_star_prior_json"] = json.dumps(
                stars, separators=(",", ":"), sort_keys=True,
            )
        return prepared

    def build_command(self, params: dict[str, Any]) -> list[str]:
        mode = str(params.get("mode", "tng") or "tng").lower()
        if mode not in ("star", "lens", "tng", "field"):
            mode = "tng"
        cmd = [
            "scripts/fasrc_poster_cutout.py",
            "--mode", mode, "--save",
        ]
        size = str(params.get("image_size", "")).strip()
        if size not in ("", "0"):
            cmd += ["--image-size", str(int(float(size)))]
        seed = str(params.get("seed", "")).strip()
        cmd += ["--seed", seed if seed != "" else "-1"]
        if params.get("_star_prior_json"):
            cmd += ["--star-prior-json", str(params["_star_prior_json"])]
        return cmd



class EnsembleTrainStep(FASRCPipelineStep):
    """Ensemble training with one independent model per SLURM array task.

    Shells out to ``scripts/train_ensemble.py``:

    * ``add`` — create N NEW members (fresh names allocated from the LOCAL
      registry at submit time, so archived/tombstoned indices are never
      reused) and train only them;
    * ``continue`` — train selected existing members either ``extra_steps``
      more each or up to one absolute ``target_steps`` checkpoint (warm cosine
      restart over each member's new absolute total);
    * ``fork`` — create N new members initialized from an existing member's
      weights (psnr or loss track), step 0, fresh optimizer + LR schedule.

    Held-out ensemble evaluation is intentionally local and is not part of
    these cluster jobs. Periodic validation within each training run remains.
    """

    # LR schedule, plateau guard and PSF-warp knobs come from /config
    # (job_config.FASRC_STEP_PARAMS; a posted value overrides them here).
    task_params = (
        TaskParam("mode", "choice", "add",
                  "add = new members, continue = train existing ones, "
                  "fork = new members from an existing member's weights.",
                  choices=("add", "continue", "fork")),
        TaskParam("count", "int", None,
                  "New members (blank = 5 for add, 1 for fork).", min=1),
        TaskParam("members", "str", None,
                  "Comma-separated members to continue (continue mode)."),
        TaskParam("continue_basis", "choice", "extra",
                  "Continue by extra steps or up to an absolute target.",
                  choices=("extra", "target")),
        TaskParam("extra_steps", "int", 50_000,
                  "Extra steps per continued member.", min=1),
        TaskParam("target_steps", "int", None,
                  "Absolute step target (continue_basis = target).", min=1),
        TaskParam("steps", "int", Config.DEFAULT_TRAIN_STEPS,
                  "Training steps for new members.", min=1),
        TaskParam("fork_from", "str", None, "Member to fork from (fork mode)."),
        TaskParam("fork_track", "choice", "psnr",
                  "Checkpoint track to fork from.", choices=("psnr", "loss")),
        TaskParam("num_res_blocks", "int", None,
                  f"Trunk depth of new members (blank = "
                  f"{Config.DEFAULT_NUM_RES_BLOCKS}).", min=1),
        TaskParam("batch_size", "int", None,
                  "Examples per update (blank = trainer default).", min=1),
        TaskParam("evaluate_every", "int", None,
                  f"Validation period in steps (blank = "
                  f"{Config.DEFAULT_EVALUATE_EVERY}).", min=1),
        TaskParam("loss", "choice", "l1", "Run-wide loss.", choices=LOSS_NAMES),
        TaskParam("noise_aug", "float", 0.0,
                  "Extra read-noise augmentation (RN units).", min=0),
        TaskParam("bootstrap", "float", 0.0,
                  "Bootstrap resampling fraction (0 = off).", min=0, max=1),
        TaskParam("asinh_knee", "float", None,
                  "Single asinh knee in e⁻ (blank = per-band 100 e⁻).", min=0),
        TaskParam("asinh_knees", "str", None,
                  "Comma-separated knees (e⁻) for a multi-knee member, "
                  "e.g. 0.1,1,10,100,1000,10000."),
        TaskParam("output_knee", "float", None,
                  "Multi-knee: output one image stretched at this knee (e⁻).",
                  min=0),
        TaskParam("knee_loss", "choice", "plain",
                  "Multi-knee channel weighting.", choices=KNEE_LOSS_MODES),
        TaskParam("target_psf_fwhm_arcsec", "float", None,
                  "Target Gaussian PSF FWHM (blank = config default).",
                  min=0, max=float(Config.TARGET_PSF_FWHM_MAX_ARCSEC)),
        TaskParam("icnr", "bool", False,
                  "ICNR-initialise the pixel-shuffle convs (add only)."),
        TaskParam("starless", "bool", False,
                  "Starless regime (erase stars); default starfull."),
        TaskParam("forward_onthefly", "bool", False,
                  "Live forward model (PSF + noise re-drawn each visit)."),
        TaskParam("psf_subset", "int", None,
                  "Per-member PSF bag size (on-the-fly; blank/0 = trainer "
                  "default).", min=0),
        TaskParam("crops_per_field", "int", None,
                  "Crops per field visit (on-the-fly; blank/0 = default).", min=0),
        TaskParam("hr_crop_size", "int", None,
                  "HR crop side (on-the-fly; blank/0 = default).", min=0),
        TaskParam("member_spec", "json", None,
                  'Per-member override list, e.g. [{"loss":"l2"},{}].'),
        TaskParam("base_seed", "int", None,
                  "Base seed (blank = fresh entropy, recorded per member).",
                  prefill=False),
        TaskParam("array_max_parallel", "int", 2,
                  "Array tasks running at once.", min=1),
    )

    def __init__(self) -> None:
        super().__init__(
            step_id="ensemble_train",
            label="Train ensemble (N members, distinct seeds)",
            job_name="ensemble-train",
            defaults=StepResources(
                partition="gpu", n_cpus=4, n_gpus=1,
                memory="32G", time_limit="48:00:00",
            ),
            needs_gpu=True,
            fixed_gpus=1,
        )

    @staticmethod
    def _members(params: dict[str, Any]) -> list[str]:
        return list(dict.fromkeys(
            name.strip()
            for name in str(params.get("members", "")).split(",")
            if name.strip()
        ))

    @staticmethod
    def _uses_forward_onthefly(params: dict[str, Any]) -> bool:
        """Whether any run-wide or per-member recipe uses the live forward."""
        run_wide = str(params.get("forward_onthefly", "")).strip().lower() in (
            "1", "true", "yes", "on",
        )
        raw_spec = str(params.get("member_spec", "") or "").strip()
        if not raw_spec:
            return run_wide
        try:
            spec = json.loads(raw_spec)
        except json.JSONDecodeError as exc:
            raise ValueError(f"per-member spec is not valid JSON: {exc}") from exc
        if not isinstance(spec, list) or not all(
            isinstance(item, dict) for item in spec
        ):
            raise ValueError(
                "per-member spec must be a JSON list of objects, one per member"
            )
        return run_wide or any(
            bool(item.get("forward_onthefly")) for item in spec
        )

    def prepare_params(self, params: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(params)
        mode = str(prepared.get("mode", "add") or "add").strip()
        if mode == "continue":
            names = self._members(prepared)
            if not names:
                raise ValueError("continue mode needs at least one member")
            prepared["members"] = ",".join(names)
        else:
            count = int(prepared.get("count", prepared.get("n_members", 0)) or
                        (1 if mode == "fork" else 5))
            if count <= 0:
                raise ValueError("member count must be positive")
            names = next_member_names(default_ensemble_dir(), count)
            prepared["count"] = count
            prepared["member_names"] = ",".join(names)
        if str(prepared.get("base_seed", "")).strip() in ("", "-1"):
            prepared["base_seed"] = secrets.randbits(32)
        prepared["array_count"] = len(names)
        requested = int(prepared.get("array_max_parallel", 2) or 2)
        prepared["array_max_parallel"] = max(1, min(requested, len(names)))
        uses_forward_onthefly = self._uses_forward_onthefly(prepared)
        stars = population_calibration.active_star()
        if stars:
            prepared["_star_prior_json"] = json.dumps(
                stars, separators=(",", ":"), sort_keys=True,
            )
        elif uses_forward_onthefly:
            raise ValueError(
                "activate a valid Gaia+Euclid stellar calibration before "
                "on-the-fly training"
            )
        return prepared

    def prepare_payload_files(
        self,
        params: dict[str, Any],
        *,
        job_name: str,
        relative_log_dir: str,
    ) -> dict[str, str]:
        """Stage immutable live-forward inputs outside the SLURM argv."""
        payload_files: dict[str, str] = {}
        for source_key, path_key, hash_key, fingerprint_key, suffix in (
            (
                "_star_prior_json",
                "_star_prior_file",
                "_star_prior_sha256",
                "_star_prior_fingerprint",
                "star-population",
            ),
        ):
            content = str(params.pop(source_key, "") or "").strip()
            if not content:
                continue
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            payload = json.loads(content)
            relative_path = (
                f"{relative_log_dir}/{job_name}.{suffix}.{digest[:12]}.json"
            )
            params[path_key] = relative_path
            params[hash_key] = digest
            if payload.get("fingerprint"):
                params[fingerprint_key] = str(payload["fingerprint"])
            payload_files[relative_path] = content + "\n"
        return payload_files

    def array_shape(self, params: dict[str, Any]) -> tuple[int, int] | None:
        count = int(params.get("array_count", 1) or 1)
        if count <= 1:
            return None
        return count, int(params["array_max_parallel"])

    def build_command(self, params: dict[str, Any]) -> list[str]:
        mode = str(params.get("mode", "add") or "add").strip()
        steps = int(params.get("steps", Config.DEFAULT_TRAIN_STEPS) or
                    Config.DEFAULT_TRAIN_STEPS)
        cmd = ["scripts/train_ensemble.py", "--mode", mode]
        batch_size = str(params.get("batch_size", "")).strip()
        if batch_size:
            with contextlib.suppress(ValueError):
                cmd += ["--batch-size", str(int(float(batch_size)))]
        if mode == "continue":
            members = self._members(params)
            if not members:
                raise ValueError("continue mode needs at least one member")
            basis = str(params.get("continue_basis", "extra") or
                        "extra").strip()
            cmd += ["--members", ",".join(members)]
            if basis == "target":
                target_steps = int(params.get("target_steps", 0) or 0)
                if target_steps <= 0:
                    raise ValueError("target_steps must be positive")
                cmd += ["--target-steps", str(target_steps)]
            elif basis == "extra":
                extra_steps = int(
                    params.get("extra_steps", 50_000) or 50_000,
                )
                if extra_steps <= 0:
                    raise ValueError("extra_steps must be positive")
                cmd += ["--extra-steps", str(extra_steps)]
            else:
                raise ValueError(
                    "continue_basis must be 'extra' or 'target'",
                )
        else:
            # add / fork create members → allocate names from the LOCAL
            # registry now (tombstones must never be reused, and the remote
            # dir still holds archived members' directories).
            count = int(params.get("count", params.get("n_members", 0)) or
                        (1 if mode == "fork" else 5))
            names = [name.strip() for name in
                     str(params.get("member_names", "")).split(",")
                     if name.strip()]
            if not names:
                # ``build_sbatch_body`` always prepares explicit names first;
                # keep direct command construction useful for CLI/tests.
                names = next_member_names(default_ensemble_dir(), count)
            if len(names) != count:
                raise ValueError("prepared member names do not match count")
            cmd += ["--count", str(count),
                    "--member-names", ",".join(names),
                    "--steps", str(steps)]
            if mode == "add":
                # Trunk depth for the NEW members (mixed-depth ensembles are
                # supported; fork inherits its source's depth instead).
                blocks = str(params.get("num_res_blocks", "")).strip()
                if blocks:
                    with contextlib.suppress(ValueError):
                        cmd += ["--num-res-blocks", str(int(blocks))]
            if mode == "fork":
                cmd += ["--fork-from",
                        str(params.get("fork_from", "")).strip(),
                        "--fork-track",
                        str(params.get("fork_track", "psnr") or "psnr")]
        # Diversity knobs: run-wide loss norm / extra-noise / bootstrap, plus
        # the per-member JSON override list (validated here so a typo 400s at
        # submit instead of burning a queued GPU job on an argparse exit).
        loss = str(params.get("loss", "")).strip()
        if loss in LOSS_NAMES:
            cmd += ["--loss", loss]
        for flag, key in (("--noise-aug", "noise_aug"),
                          ("--bootstrap", "bootstrap")):
            val = str(params.get(key, "")).strip()
            if val not in ("", "0", "0.0"):
                with contextlib.suppress(ValueError):
                    cmd += [flag, f"{float(val):g}"]
        # Per-member asinh stretch knee (electrons) — a fresh-member (add)
        # normalization knob; blank/0/default-100 → unset (per-band 100 e⁻).
        knee = str(params.get("asinh_knee", "")).strip()
        if knee not in ("", "0", "0.0", "100", "100.0"):
            with contextlib.suppress(ValueError):
                cmd += ["--asinh-knee", f"{float(knee):g}"]
        # Multi-knee member (add only): stretch at every knee; optionally one
        # output image scored at every knee, channels balanced or plain.
        knees = str(params.get("asinh_knees", "") or "").strip()
        if knees:
            try:
                values = [float(token) for token in knees.split(",")
                          if token.strip()]
            except ValueError as exc:
                raise ValueError(f"asinh_knees must be numbers: {knees!r}") from exc
            if not values or any(v <= 0 for v in values):
                raise ValueError("asinh_knees must be positive electrons")
            cmd += ["--asinh-knees", ",".join(f"{v:g}" for v in values)]
            output_knee = str(params.get("output_knee", "") or "").strip()
            if output_knee:
                cmd += ["--output-knee", f"{float(output_knee):g}"]
            knee_loss = str(params.get("knee_loss", "") or "").strip()
            if knee_loss and knee_loss != "plain":
                if knee_loss not in KNEE_LOSS_MODES:
                    raise ValueError(f"knee_loss must be one of {KNEE_LOSS_MODES}")
                cmd += ["--knee-loss", knee_loss]
        evaluate_every = str(params.get("evaluate_every", "") or "").strip()
        if evaluate_every:
            with contextlib.suppress(ValueError):
                cmd += ["--evaluate-every", str(int(float(evaluate_every)))]
        target_fwhm = str(params.get("target_psf_fwhm_arcsec", "")).strip()
        if target_fwhm != "":
            with contextlib.suppress(ValueError):
                cmd += ["--target-psf-fwhm-arcsec", f"{float(target_fwhm):g}"]
        # ICNR init: checkerboard-free sub-pixel upsampler (add members only).
        if str(params.get("icnr", "")).strip().lower() in (
                "1", "true", "yes", "on"):
            cmd += ["--icnr"]
        # Star regime (default starfull: reconstruct stars). Starless — erase
        # them — is the opt-in, so only that case emits the flag.
        if str(params.get("starless", "0")).strip().lower() in (
            "1", "true", "yes", "on"):
            cmd += ["--starless", "1"]
        star_prior_file = str(params.get("_star_prior_file", "") or "").strip()
        if star_prior_file:
            cmd += ["--star-prior-file", star_prior_file]
        elif params.get("_star_prior_json"):
            cmd += ["--star-prior-json", str(params["_star_prior_json"])]
        # Live forward model: full-field PSF+noise re-realization per visit.
        run_wide_forward = str(
            params.get("forward_onthefly", "")
        ).strip().lower() in ("1", "true", "yes", "on")
        if run_wide_forward:
            cmd += ["--forward-onthefly", "1"]
        if self._uses_forward_onthefly(params):
            for flag, key in (("--psf-subset", "psf_subset"),
                              ("--crops-per-field", "crops_per_field"),
                              ("--hr-crop-size", "hr_crop_size")):
                val = str(params.get(key, "")).strip()
                if val not in ("", "0"):
                    with contextlib.suppress(ValueError):
                        cmd += [flag, str(int(float(val)))]
            for flag, key in (("--psf-warp-prob", "psf_warp_prob"),
                              ("--psf-warp-alpha-max", "psf_warp_alpha_max"),
                              ("--psf-warp-sigma", "psf_warp_sigma"),
                              ("--saturation-mask-prob",
                               "saturation_mask_prob")):
                val = str(params.get(key, "")).strip()
                if val != "":
                    with contextlib.suppress(ValueError):
                        cmd += [flag, f"{float(val):g}"]
        member_spec = str(params.get("member_spec", "")).strip()
        if member_spec:
            try:
                spec = json.loads(member_spec)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"per-member spec is not valid JSON: {e}") from e
            if not isinstance(spec, list) or not all(
                    isinstance(o, dict) for o in spec):
                raise ValueError("per-member spec must be a JSON list of "
                                 "objects, one per member (e.g. "
                                 '[{"loss":"l2"},{"noise_aug":1}])')
            cmd += ["--member-spec", json.dumps(spec)]
        # Fixed base seed → fully reproducible ensemble; blank / -1 → entropy
        # (the value used is still recorded on each member's provenance).
        base_seed = str(params.get("base_seed", "")).strip()
        if base_seed not in ("", "-1"):
            with contextlib.suppress(ValueError):
                cmd += ["--base-seed", str(int(base_seed))]
        if int(params.get("array_count", 1) or 1) > 1:
            cmd.append("--array-task")
        # LR schedule + plateau-guard knobs, injected from the /config page via
        # FASRC_STEP_PARAMS. Passed through verbatim as --lr-* / --plateau-lr-*
        # flags (train_ensemble.py's argparse validates/typechecks them).
        for name in ("lr_peak", "lr_final", "lr_warmup_steps",
                     "plateau_lr_enabled", "plateau_lr_factor",
                     "plateau_lr_patience", "plateau_lr_min_delta",
                     "plateau_lr_min_delta_rel",
                     "plateau_lr_cooldown", "plateau_lr_min_lr",
                     "plateau_lr_metric", "plateau_rollback_min_gap",
                     "plateau_lr_recovery"):
            val = str(params.get(name, "")).strip()
            if val:
                cmd += [f"--{name.replace('_', '-')}", val]
        return cmd


# ---------------------------------------------------------------------------
# ``run_pipeline.py`` step base
# ---------------------------------------------------------------------------
#
# Shared base for jobs that shell out to ``scripts/run_pipeline.py``. The
# only surviving concrete step is :class:`SyntheticGenerateStep` (the
# synthetic training-pair generator). The old four-preset training form
# (gen_convolve / convolve_only / train_only / custom) was removed —
# training now goes exclusively through :class:`EnsembleTrainStep`.


@dataclass
class RunPipelineStep(FASRCPipelineStep):
    """A ``scripts/run_pipeline.py`` job.

    Subclasses set ``defaults`` (resources) and ``skip_flags``; the
    argv shape (--ntrain / --nvalid / --ntest / --image-size + extra) is
    shared. Generation is decoupled from training — no --batch-size / --steps
    (those were an artifact of when generation flowed straight into a train
    step; training now goes exclusively through ``EnsembleTrainStep``).
    """

    #: Extra ``--skip-…`` flags appended to the run_pipeline.py argv.
    #: Stored as a tuple so instances stay hashable (the dataclass
    #: ``defaults`` field is mutable, but this one is set per-class).
    skip_flags:        tuple[str, ...] = ()

    log_dir_prefix:  ClassVar[str] = "logs/jobs"

    def banner_line(self, label: str) -> str:
        return f"Web-submitted job: {label}"

    def build_command(self, params: dict[str, Any]) -> list[str]:
        # ``.get`` rather than ``[…]`` so missing keys don't blow up the
        # script renderer (the Flask handler validates numerics up-front
        # via ``StepResources.from_form`` + an explicit ``int()`` pass).
        cmd = [
            "scripts/run_pipeline.py",
            "--ntrain",     str(int(params.get("n_train",    0))),
            "--nvalid",     str(int(params.get("n_valid",    0))),
            "--ntest",      str(int(params.get("n_test",     0))),
            "--image-size", str(int(params.get("image_size", 0))),
        ]
        cmd.extend(self.skip_flags)
        # ``extra_flags`` predates the structured split selector. The Sky UI
        # intentionally mirrors the target into this channel so a resident
        # pre-upgrade Flask process can still forward the new run_pipeline CLI
        # option before it is restarted. New backends detect that mirror and
        # emit the option only once.
        extra = (params.get("extra_flags") or "").strip()
        extra_tokens = shlex.split(extra) if extra else []
        extra_has_regenerate_splits = any(
            token == "--regenerate-splits" or
            token.startswith("--regenerate-splits=")
            for token in extra_tokens
        )
        # Default: cache-first resume. ``force`` rebuilds everything, while
        # ``regenerate_splits`` deletes and rebuilds only the named splits;
        # run_pipeline leaves every unselected split entirely untouched.
        force = str(params.get("force", "")).strip().lower() in (
            "1", "true", "yes", "on")
        regenerate_raw = str(params.get("regenerate_splits", "")).strip()
        regenerate_splits = list(dict.fromkeys(
            part.strip().lower() for part in regenerate_raw.split(",")
            if part.strip()
        ))
        invalid_splits = [split_name for split_name in regenerate_splits
                          if split_name not in ("train", "validate", "test")]
        if invalid_splits:
            raise ValueError(
                "invalid regeneration split(s): " + ", ".join(invalid_splits))
        if force and regenerate_splits:
            raise ValueError(
                "force and targeted split regeneration are mutually exclusive")
        if force:
            cmd.append("--force")
        elif regenerate_splits and not extra_has_regenerate_splits:
            cmd += ["--regenerate-splits", ",".join(regenerate_splits)]
        # On-the-fly training reads clean_train directly and builds LR+target
        # live — generate the train split as clean-only (no hr, no dirty);
        # validate/test keep the full triple. Stale hr/dirty_train are deleted.
        if str(params.get("onthefly_train", "")).strip().lower() in (
                "1", "true", "yes", "on"):
            cmd.append("--onthefly-train")
        # Free-form user-supplied flags. ``shlex.split`` so individual
        # tokens get re-quoted by ``render_sbatch_body`` instead of being
        # smuggled in as one shell-expandable string.
        cmd.extend(extra_tokens)
        return cmd


class SyntheticGenerateStep(RunPipelineStep):
    """Synthetic training-pair generation for the /sky page.

    Renders synthetic clean HR scenes (PHZ-conditioned TNG galaxies + stars +
    strong lenses) and forward-models them to dirty Euclid LR with the
    empirical band PSFs, writing clean + HR + dirty TFRecords. Runs
    ``run_pipeline.py --skip-train`` as a dedicated, knob-bearing step
    card on /sky.
    """

    # Scene counts, image size, densities and PSF knobs come from /config.
    task_params = (
        TaskParam("force", "bool", False,
                  "Regenerate every split from scratch."),
        TaskParam("regenerate_splits", "str", None,
                  "Comma list of splits to rebuild (train,validate,test); "
                  "excludes force."),
        TaskParam("onthefly_train", "bool", False,
                  "Train split clean-only (on-the-fly training)."),
        TaskParam("extra_flags", "str", None,
                  "Extra run_pipeline.py flags."),
        TaskParam("tng_dir", "str", None, "TNG SKIRT atlas dir (blank = default)."),
        TaskParam("tng_properties", "str", None,
                  "TNG properties CSV (blank = default)."),
        TaskParam("tng_radius_manifest", "str", None,
                  "TNG radius manifest (blank = default)."),
    )

    def __init__(self) -> None:
        super().__init__(
            step_id="synthetic_generate",
            label="Generate synthetic training pairs (CPU)",
            job_name="synthetic-data",
            defaults=StepResources(
                partition="shared", n_cpus=16, n_gpus=0,
                memory="64G", time_limit="6:00:00",
            ),
            skip_flags=("--skip-train",),
        )

    def prepare_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Freeze the active Euclid brightness-radius population into the job."""
        prepared = super().prepare_params(params)
        joint_status = population_calibration.joint_galaxy_state()
        joint = joint_status.get("active") or {}
        if not joint_status.get("is_active") or not joint:
            raise ValueError(
                "activate the Euclid VIS 2FWHM × Sérsic-R_e galaxy fit before "
                "generating fields"
            )
        prepared["_joint_galaxy_population_json"] = json.dumps(
            joint, separators=(",", ":"), sort_keys=True,
        )
        try:
            prepared["galaxy_density_arcmin2"] = float(
                joint["generation"]["surface_density_arcmin2"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "empirical PHZ galaxy population has no finite density"
            ) from exc
        star_status = population_calibration.star_state()
        stars = star_status.get("active") or {}
        if not star_status.get("is_active") or not stars:
            raise ValueError(
                "activate a valid Gaia+Euclid stellar calibration before "
                "generating fields"
            )
        prepared["_star_prior_json"] = json.dumps(
            stars, separators=(",", ":")
        )
        population = stars.get("population") or {}
        if population.get("density_arcmin2") is not None:
            prepared["star_density_arcmin2"] = float(
                population["density_arcmin2"]
            )
        return prepared

    def prepare_payload_files(
        self,
        params: dict[str, Any],
        *,
        job_name: str,
        relative_log_dir: str,
    ) -> dict[str, str]:
        """Move frozen population artifacts out of the process argv.

        Linux limits each individual ``execve`` argument to roughly 128 KiB.
        The magnitude-conditioned Euclid population artifact is larger than
        that, so embedding it in ``--joint-galaxy-population-json`` prevents
        Python from starting.  Persist both population inputs next to the job
        script and pass only their short paths to ``run_pipeline.py``.
        """
        payload_files: dict[str, str] = {}
        for source_key, path_key, hash_key, fingerprint_key, suffix in (
            (
                "_joint_galaxy_population_json",
                "_joint_galaxy_population_file",
                "_joint_galaxy_population_sha256",
                "_joint_galaxy_population_fingerprint",
                "galaxy-population",
            ),
            (
                "_star_prior_json",
                "_star_prior_file",
                "_star_prior_sha256",
                "_star_prior_fingerprint",
                "star-population",
            ),
        ):
            content = str(params.pop(source_key, "") or "").strip()
            if not content:
                continue
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            payload = json.loads(content)
            relative_path = (
                f"{relative_log_dir}/{job_name}.{suffix}.{digest[:12]}.json"
            )
            params[path_key] = relative_path
            params[hash_key] = digest
            if payload.get("fingerprint"):
                params[fingerprint_key] = str(payload["fingerprint"])
            payload_files[relative_path] = content + "\n"
        return payload_files

    def build_command(self, params: dict[str, Any]) -> list[str]:
        # Parallelise generation across the allocated CPUs: one process per
        # CPU runs the combined generate+forward pass on its index range.
        cmd = super().build_command(params)
        try:
            workers = int(params.get("n_cpus") or self.defaults.n_cpus)
        except (TypeError, ValueError):
            workers = self.defaults.n_cpus
        cmd += ["--gen-workers", str(max(1, workers))]
        # One empirical PHZ draw supplies each TNG morphology's redshift while
        # preserving the calibrated observed radius and VIS 2FWHM anchor.
        raw_tng_density = params.get("galaxy_density_arcmin2")
        if raw_tng_density in (None, ""):
            raise ValueError(
                "an activated fitted galaxy density is required; refusing "
                "the Config fallback"
            )
        try:
            tng_density = float(raw_tng_density)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid TNG density: {raw_tng_density!r}"
            ) from exc
        if not (0.0 <= tng_density < float("inf")):
            raise ValueError(f"invalid TNG density: {raw_tng_density!r}")
        # This value is also embedded at full precision in the activated
        # population payload and checked as a hard upper bound by the remote
        # simulator.  The default ``:g`` precision can round the CLI value up
        # past that identical bound (for example 372.831718745... becomes
        # 372.832), so preserve enough digits for an exact float round trip.
        cmd += ["--galaxy-density-arcmin2", f"{tng_density:.17g}"]
        joint_population_file = str(
            params.get("_joint_galaxy_population_file", "") or ""
        ).strip()
        joint_population = str(
            params.get("_joint_galaxy_population_json", "") or ""
        ).strip()
        if joint_population_file:
            cmd += ["--joint-galaxy-population-file", joint_population_file]
        elif joint_population:
            cmd += ["--joint-galaxy-population-json", joint_population]
        for key, flag in (("tng_dir", "--tng-dir"),
                          ("tng_properties", "--tng-properties"),
                          ("tng_radius_manifest", "--tng-radius-manifest")):
            value = str(params.get(key, "") or "").strip()
            if value:
                cmd += [flag, value]
        star_prior_file = str(
            params.get("_star_prior_file", "") or ""
        ).strip()
        if star_prior_file:
            cmd += ["--star-prior-file", star_prior_file]
        elif params.get("_star_prior_json"):
            cmd += ["--star-prior-json", str(params["_star_prior_json"])]
        # Scene-population and forward-PSF knobs (from /config). Emit only when
        # supplied so direct programmatic callers can still rely on CLI
        # defaults. The warp is realised while each dirty exposure is rendered;
        # no warped kernels are precomputed or stored.
        for param, flag in (("star_density_arcmin2", "--star-density-arcmin2"),
                            ("lens_density_arcmin2", "--lens-density-arcmin2"),
                            ("lens_sigma_v_min_kms", "--lens-sigma-v-min-kms"),
                            ("lens_sigma_v_max_kms", "--lens-sigma-v-max-kms"),
                            ("psf_warp_prob",        "--psf-warp-prob"),
                            ("psf_warp_alpha_max",   "--psf-warp-alpha-max"),
                            ("psf_warp_sigma",       "--psf-warp-sigma"),
                            ("saturation_mask_prob",
                             "--saturation-mask-prob")):
            val = params.get(param)
            if val not in (None, ""):
                with contextlib.suppress(TypeError, ValueError):
                    cmd += [flag, f"{float(val):g}"]
        return cmd


# ---------------------------------------------------------------------------
# Registry — single source of truth for which steps exist
# ---------------------------------------------------------------------------

STEP_CLASSES: tuple[Callable[[], FASRCPipelineStep], ...] = (
    VISNoiseSampleStep,
    ArchiveFieldSampleStep,
    EuclidQueryStep,
    EuclidVerifyPhotometryStep,
    EuclidCutoutDownloadStep,
    EuclidPSFExtractStep,
    PSFRotationPoolStep,
    TngSkirtAtlasDownloadStep,
    MeasureTngRadiiStep,
    TngGridStep,
    TngStackStep,
    PosterCutoutStep,
    SyntheticGenerateStep,
    EnsembleTrainStep,
)


@dataclass(frozen=True)
class StepRegistry:
    """Lookup helper: ``REGISTRY.get("ensemble_train")`` → step instance."""

    by_id: dict[str, FASRCPipelineStep] = field(default_factory=dict)

    @classmethod
    def build(cls) -> StepRegistry:
        return cls(by_id={k.__name__ and step.step_id: step
                          for k in STEP_CLASSES
                          for step in [k()]})

    def get(self, step_id: str) -> FASRCPipelineStep:
        s = self.by_id.get(step_id)
        if s is None:
            raise KeyError(f"unknown pipeline step: {step_id!r}")
        return s

    def all(self) -> list[FASRCPipelineStep]:
        return list(self.by_id.values())


REGISTRY = StepRegistry.build()
