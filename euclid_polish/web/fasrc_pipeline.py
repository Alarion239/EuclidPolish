"""
FASRC pipeline-step abstraction for the HST → Euclid SR workflow.

Five jobs (download HLSP tiles → extract HST PSF → compute the
differential kernel → write HST-derived TFRecords → train) all share
the same submission shape — write an sbatch script, ``sbatch`` it
through the ControlMaster SSH, register in :class:`JobDB`. The only
things that differ per job are:

  * default SLURM resources (gpu vs cpu, runtime, memory)
  * the remote script to invoke + its CLI flags

This module factors all of that into one abstract base class
(:class:`FASRCPipelineStep`) with one concrete subclass per job.
Adding a sixth step (e.g. "ingest JWST F814W") is then a 30-line
delta — no Flask, web template, or sbatch-template duplication.

The companion in :mod:`euclid_polish.web.fasrc_jobs` (the existing
training-only sbatch builder) remains untouched for backwards
compatibility; new code uses this module instead.
"""

from __future__ import annotations

import shlex
import textwrap
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from euclid_polish.web import fasrc_config


# ---------------------------------------------------------------------------
# Resource preset (subset of SLURM knobs the user can override per submit)
# ---------------------------------------------------------------------------

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "partition":  self.partition,
            "n_cpus":     int(self.n_cpus),
            "n_gpus":     int(self.n_gpus),
            "memory":     self.memory,
            "time_limit": self.time_limit,
        }

    @classmethod
    def from_form(cls, form: Dict[str, Any], defaults: "StepResources") -> "StepResources":
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
    def from_form_strict(cls, form: Dict[str, Any]) -> "StepResources":
        """Build from a Flask form, *rejecting* blank resource fields.

        Used by the HST-pipeline submit endpoint after the
        history-driven-defaults UI change: there is no implicit
        fallback to a step's defaults anymore. If the form has no
        precedent in the CSV log and the user typed nothing, the
        submit must error out rather than silently allocate something
        the user didn't consent to.

        ``partition``, ``memory``, ``time_limit`` are required strings;
        ``n_cpus`` and ``n_gpus`` are required integers (``n_gpus=0``
        is fine, but blank is not).
        """
        missing: List[str] = []
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
# Abstract base
# ---------------------------------------------------------------------------

@dataclass
class FASRCPipelineStep(ABC):
    """One submittable FASRC job.

    Subclasses implement :meth:`build_command` to produce the Python
    command line that runs on the remote node. Everything else
    (SLURM header, conda setup, log layout, runtime banner) is shared
    via :meth:`build_sbatch_body`.
    """

    #: Stable id used in URLs (``/api/fasrc/hst/<step_id>/submit``).
    step_id:   str
    #: Human label shown in the UI.
    label:     str
    #: One-line description for the form's hint text.
    description: str
    #: Default SLURM allocation. Overridable per-submission via the form.
    defaults:  StepResources
    #: True iff this job needs GPUs (used by the form to enable/disable the gpu count field).
    needs_gpu: bool = False
    #: If set, the CPU count is locked to this value at submit time and
    #: the UI hides the corresponding form field. Use when the underlying
    #: work is single-threaded and asking SLURM for extra cores just
    #: wastes the allocation — see HSTPSFExtractStep.
    fixed_cpus: Optional[int] = None

    @abstractmethod
    def build_command(self, params: Dict[str, Any]) -> List[str]:
        """Return the Python command (argv) to run inside the sbatch script.

        ``params`` is a flat dict of form values; subclasses pick out
        whatever step-specific knobs they need. Return value is a list
        of shell-safe tokens — :meth:`build_sbatch_body` joins them with
        spaces and shell-quotes individually.
        """

    # ------------------------------------------------------------------ #

    #: Subclasses override these class-level constants to control where
    #: their log files land and how the job name is shaped. HST steps
    #: live under ``logs/hst_pipeline``; legacy training presets live
    #: under ``logs/jobs``. Declared as ``ClassVar`` so the dataclass
    #: machinery doesn't promote them to per-instance fields.
    log_dir_prefix:  ClassVar[str] = "logs/hst_pipeline"
    job_name_prefix: ClassVar[str] = "hst"

    def banner_line(self, label: str) -> str:
        """First echo line of the sbatch banner.

        Overrideable so each step family stays grep-searchable in the logs
        (``HST pipeline step:`` vs ``Web-submitted job:`` etc.).
        """
        return f"HST pipeline step: {self.step_id} — {label}"

    def build_sbatch_body(
        self,
        *,
        params: Dict[str, Any],
        resources: StepResources,
        cfg: fasrc_config.FasrcConfig,
        label: str,
        relative_log_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """Render the full sbatch script + the relative log paths.

        Returns ``{"body": str, "script": rel, "out": rel, "err": rel,
        "name": str}``. Thin wrapper that picks the log dir and job-name
        shape from the step and delegates to :func:`render_sbatch_body`.
        """
        log_dir   = relative_log_dir or self.log_dir_prefix
        ts        = time.strftime("%Y%m%d-%H%M%S")
        job_name  = f"{self.job_name_prefix}-{self.step_id}-{ts}"
        return render_sbatch_body(
            job_name=job_name,
            relative_log_dir=log_dir,
            resources=resources,
            cfg=cfg,
            label=label,
            cmd_argv=self.build_command(params),
            banner_line=self.banner_line(label),
            step_id=self.step_id,
        )


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
    cmd_argv:         List[str],
    banner_line:      str,
    step_id:          Optional[str] = None,
) -> Dict[str, str]:
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
        If given, an ``STEP_ID=…`` echo is emitted at the tail so
        :func:`fasrc_jobs.parse_step_id` can tag the runtime.

    Returns
    -------
    dict
        ``{"body": str, "script": rel, "out": rel, "err": rel, "name": str}``
    """
    script_rel = f"{relative_log_dir}/{job_name}.sh"
    out_rel    = f"{relative_log_dir}/{job_name}.out"
    err_rel    = f"{relative_log_dir}/{job_name}.err"
    # JSONL stream of structured progress events. Producer:
    # :class:`euclid_polish.observability.Reporter`. Consumer: the
    # /api/fasrc/jobs/<jobid>/status endpoint.
    events_rel = f"{relative_log_dir}/{job_name}.events"

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

    body = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={shlex.quote(job_name)}
        #SBATCH --partition={shlex.quote(resources.partition)}
        {gres_line}#SBATCH --cpus-per-task={int(resources.n_cpus)}
        #SBATCH --mem={resources.memory}
        #SBATCH --time={resources.time_limit}
        #SBATCH --output={out_rel}
        #SBATCH --error={err_rel}

        set -euo pipefail
        cd "$SLURM_SUBMIT_DIR"
        mkdir -p {relative_log_dir}

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
        export EUCLID_POLISH_EVENTS_PATH={shlex.quote(events_rel)}
        mkdir -p "$EUCLID_POLISH_DATA_DIR" "$EUCLID_POLISH_CKPT_DIR"

        module purge
        module load python
        module load cuda

        if [ -z "${{CONDA_SHLVL:-}}" ]; then
          CONDA_BASE="$(conda info --base 2>/dev/null || true)"
          if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$CONDA_BASE/etc/profile.d/conda.sh"
          fi
          if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/mamba.sh" ]; then
            # shellcheck disable=SC1091
            source "$CONDA_BASE/etc/profile.d/mamba.sh"
          fi
        fi
        mamba activate {shlex.quote(cfg.conda_env_path)}

        echo "Python:  $(which python)"
        python -u {cmd_line}

        __FASRC_T1__=$(date +%s)
        __FASRC_RUNTIME__=$((__FASRC_T1__ - __FASRC_T0__))
        echo "============================================================"
        echo "Finished: $(date)"
        echo "RUNTIME_SECONDS=${{__FASRC_RUNTIME__}}"{step_id_echo}
        echo "============================================================"
    """)
    return {
        "body":   body,
        "script": script_rel,
        "out":    out_rel,
        "err":    err_rel,
        "events": events_rel,
        "name":   job_name,
    }


# ---------------------------------------------------------------------------
# Concrete steps
# ---------------------------------------------------------------------------

class HSTDownloadStep(FASRCPipelineStep):
    def __init__(self):
        super().__init__(
            step_id="download",
            label="1. Download COSMOS HLSP tiles",
            description=(
                "Pull COSMOS F814W HLSP mosaic tiles (~1.9 GB each) "
                "from MAST into $DATA_DIR/hst_hlsp/. Idempotent."
            ),
            defaults=StepResources(
                partition="shared", n_cpus=2, n_gpus=0,
                memory="16G", time_limit="1:00:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
        n_tiles = int(params.get("n_tiles", 25))
        return [
            "scripts/fasrc_download_hst_hlsp.py",
            "--n-tiles", str(n_tiles),
        ]


class HSTPSFExtractStep(FASRCPipelineStep):
    def __init__(self):
        super().__init__(
            step_id="extract_psf",
            label="2. Extract HST F814W ePSF",
            description=(
                "Scan the downloaded HLSP tiles for bright unsaturated "
                "stars and run photutils EPSFBuilder. Writes "
                "$DATA_DIR/hst_psf/F814W.fits. Single-threaded — "
                "DAOStarFinder and EPSFBuilder don't parallelise, so "
                "the allocation is locked at 1 CPU to avoid wasting "
                "FASRC cores."
            ),
            defaults=StepResources(
                partition="shared", n_cpus=1, n_gpus=0,
                memory="8G", time_limit="0:10:00",
            ),
            needs_gpu=False,
            fixed_cpus=1,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
        n_stars = int(params.get("n_stars", 200))
        return [
            "scripts/fasrc_extract_hst_psf.py",
            "--n-stars", str(n_stars),
        ]


class DifferentialKernelStep(FASRCPipelineStep):
    def __init__(self):
        super().__init__(
            step_id="kernel",
            label="3. Build differential kernel A = E / H",
            description=(
                "Solve A ⊛ H = E in Fourier space with Wiener "
                "regularisation. Writes $DATA_DIR/hst_psf/diff_kernel_VIS.fits."
            ),
            defaults=StepResources(
                partition="shared", n_cpus=1, n_gpus=0,
                memory="4G", time_limit="0:05:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
        reg = float(params.get("regularisation", 1e-3))
        return [
            "scripts/fasrc_compute_differential_kernel.py",
            "--regularisation", f"{reg:g}",
        ]


class HSTTFRecordStep(FASRCPipelineStep):
    def __init__(self):
        super().__init__(
            step_id="tfrecords",
            label="4. Generate HST → Euclid TFRecord pairs",
            description=(
                "Cut HST targets from HLSP tiles, apply the analytic "
                "differential kernel A and the Euclid per-band noise "
                "model, rejecting stamps where the brightest pixel "
                "would produce A(ε) ringing above "
                "max_relative_noise × σ_LR. Writes paired clean (HST) "
                "+ dirty (Euclid-equivalent) TFRecords."
            ),
            defaults=StepResources(
                partition="shared", n_cpus=16, n_gpus=0,
                memory="64G", time_limit="0:10:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
        n_train = int(params.get("n_train", 6400))
        n_valid = int(params.get("n_valid", 200))
        image_size = int(params.get("image_size", 510))
        max_relative_noise = float(params.get("max_relative_noise", 5.0))
        return [
            "scripts/fasrc_generate_hst_tfrecords.py",
            "--n-train", str(n_train),
            "--n-valid", str(n_valid),
            "--image-size", str(image_size),
            "--max-relative-noise", f"{max_relative_noise:g}",
        ]


class EuclidSkyDownloadStep(FASRCPipelineStep):
    def __init__(self):
        super().__init__(
            step_id="euclid_sky_download",
            label="5a. Download real Euclid sky cutouts (round-trip)",
            description=(
                "Generate N random sky positions inside a 2° disk on "
                "the Euclid coverage map and pull large 4-band cutouts "
                "(VIS + NISP Y/J/H) from the Euclid archive. Cutouts "
                "land in $DATA_DIR/euclid_sky/cutouts/<band>/. Out-of-"
                "coverage positions silently drop out at the mosaic "
                "lookup step. This step is the prerequisite for the "
                "round-trip self-supervised training path."
            ),
            defaults=StepResources(
                partition="shared", n_cpus=8, n_gpus=0,
                memory="16G", time_limit="2:00:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
        n_positions = int(params.get("n_positions", 200))
        vis_pixels  = int(params.get("vis_pixels", 512))
        ra_centre   = float(params.get("ra_centre", 270.0))
        dec_centre  = float(params.get("dec_centre", 66.0))
        radius_deg  = float(params.get("radius_deg", 2.0))
        return [
            "scripts/fasrc_download_euclid_sky_cutouts.py",
            "--n-positions", str(n_positions),
            "--vis-pixels",  str(vis_pixels),
            "--ra-centre",   f"{ra_centre:g}",
            "--dec-centre",  f"{dec_centre:g}",
            "--radius-deg",  f"{radius_deg:g}",
        ]


class EuclidRoundtripTFRecordStep(FASRCPipelineStep):
    def __init__(self):
        super().__init__(
            step_id="euclid_roundtrip_tfrecords",
            label="5b. Stack + chop Euclid sky cutouts into round-trip TFRecords",
            description=(
                "Read per-band cutouts from 5a, stack into 4-channel "
                "(VIS + Y/J/H) cubes on the shared 0.10\"/pix archive "
                "grid, chop into smaller training stamps, and write "
                "LR-only ``dirty_{train,validate}.tfrecord`` under "
                "$DATA_DIR/images/records_v2_euclid_roundtrip/. "
                "Per-band pixel values are multiplied by their "
                "``t_total_s`` to convert archive e⁻/s units to the "
                "total electrons the synthetic/HST records use."
            ),
            defaults=StepResources(
                partition="shared", n_cpus=4, n_gpus=0,
                memory="16G", time_limit="1:00:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
        vis_pixels = int(params.get("vis_pixels", 512))
        stamp_size = int(params.get("stamp_size", 128))
        valid_fraction = float(params.get("valid_fraction", 0.1))
        return [
            "scripts/fasrc_generate_euclid_roundtrip_tfrecords.py",
            "--vis-pixels", str(vis_pixels),
            "--stamp-size", str(stamp_size),
            "--valid-fraction", f"{valid_fraction:g}",
        ]


class HSTTrainStep(FASRCPipelineStep):
    def __init__(self):
        super().__init__(
            step_id="train",
            label="6. Train WDSR with HST + round-trip mix",
            description=(
                "Train the WDSR model on a mix of synthetic + "
                "HST-derived + (optional) real-Euclid round-trip "
                "TFRecords. ``hst_fraction`` and ``roundtrip_fraction`` "
                "control per-batch sampling; their sum must be ≤ 1. "
                "When ``roundtrip_fraction > 0`` the trainer adds a "
                "self-supervised loss "
                "``|asinh(Conv(M(lr))/k) - lr_vis|`` for round-trip "
                "examples, using the VIS PSF FITS as a TF-graph "
                "forward op (PSF + 2× sum-rebin, deterministic)."
            ),
            defaults=StepResources(
                partition="gpu", n_cpus=4, n_gpus=1,
                memory="32G", time_limit="24:00:00",
            ),
            needs_gpu=True,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
        steps               = int(params.get("steps", 400_000))
        batch_size          = int(params.get("batch_size", 16))
        hst_fraction        = float(params.get("hst_fraction", 0.1))
        roundtrip_fraction  = float(params.get("roundtrip_fraction", 0.0))
        cmd = [
            "scripts/fasrc_train_with_hst.py",
            "--steps", str(steps),
            "--batch-size", str(batch_size),
            "--hst-fraction", f"{hst_fraction:g}",
        ]
        # Only emit the round-trip flag when non-default so existing
        # SLURM submissions stay byte-identical until the user opts in.
        if roundtrip_fraction > 0:
            cmd += ["--roundtrip-fraction", f"{roundtrip_fraction:g}"]
        return cmd


# ---------------------------------------------------------------------------
# ``run_pipeline.py`` presets
# ---------------------------------------------------------------------------
#
# The original ``/api/fasrc/submit`` endpoint exposed four canned
# "presets" (gen_convolve / convolve_only / train_only / custom) that
# differed in resources and which ``--skip-*`` flags they appended to
# ``scripts/run_pipeline.py``. They are now :class:`FASRCPipelineStep`
# subclasses so submission flows through the same render+submit helpers
# as every other job — one template, one DB write, one log directory.
#
# UI compatibility: :func:`fasrc_jobs.PRESETS` derives its dict from
# these subclasses, so the JS form's preset dropdown keeps working
# without changes.


@dataclass
class RunPipelineStep(FASRCPipelineStep):
    """A ``scripts/run_pipeline.py`` job.

    Subclasses set ``defaults`` (resources) and ``skip_flags``; the
    argv shape (--ntrain / --nvalid / --image-size / --batch-size /
    --steps + extra) is shared.
    """

    #: Extra ``--skip-…`` flags appended to the run_pipeline.py argv.
    #: Stored as a tuple so instances stay hashable (the dataclass
    #: ``defaults`` field is mutable, but this one is set per-class).
    skip_flags:        Tuple[str, ...] = ()
    #: Whether the UI should show the training-only knob fields
    #: (n_train / n_valid / image_size / batch_size / steps). The
    #: convolve presets don't need them; ``train_only`` and ``custom``
    #: do. Surfaced in :data:`fasrc_jobs.PRESETS` for the JS form.
    needs_train_knobs: bool = True

    log_dir_prefix:  ClassVar[str] = "logs/jobs"
    job_name_prefix: ClassVar[str] = "euclid"

    def banner_line(self, label: str) -> str:
        return f"Web-submitted job: {label}"

    def build_command(self, params: Dict[str, Any]) -> List[str]:
        # ``.get`` rather than ``[…]`` so missing keys don't blow up the
        # script renderer (the Flask handler validates numerics up-front
        # via ``StepResources.from_form`` + an explicit ``int()`` pass).
        cmd = [
            "scripts/run_pipeline.py",
            "--ntrain",     str(int(params.get("n_train",    0))),
            "--nvalid",     str(int(params.get("n_valid",    0))),
            "--image-size", str(int(params.get("image_size", 0))),
            "--batch-size", str(int(params.get("batch_size", 0))),
            "--steps",      str(int(params.get("steps",      0))),
        ]
        cmd.extend(self.skip_flags)
        # Free-form user-supplied flags. ``shlex.split`` so individual
        # tokens get re-quoted by ``render_sbatch_body`` instead of being
        # smuggled in as one shell-expandable string.
        extra = (params.get("extra_flags") or "").strip()
        if extra:
            cmd.extend(shlex.split(extra))
        return cmd


class GenConvolveStep(RunPipelineStep):
    def __init__(self) -> None:
        super().__init__(
            step_id="gen_convolve",
            label="Generate + convolve (CPU)",
            description=(
                "Render synthetic clean HR scenes and convolve them with "
                "the Euclid PSF into the dirty LR set. Skips training so "
                "the run finishes on a CPU partition."
            ),
            defaults=StepResources(
                partition="shared", n_cpus=16, n_gpus=0,
                memory="64G", time_limit="6:00:00",
            ),
            skip_flags=("--skip-train",),
            needs_train_knobs=False,
        )


class ConvolveOnlyStep(RunPipelineStep):
    def __init__(self) -> None:
        super().__init__(
            step_id="convolve_only",
            label="Convolve existing clean → dirty (CPU)",
            description=(
                "Re-convolve the already-generated clean HR scenes against "
                "the current PSF / kernel. Skips generation and training."
            ),
            defaults=StepResources(
                partition="shared", n_cpus=8, n_gpus=0,
                memory="32G", time_limit="2:00:00",
            ),
            skip_flags=("--skip-generate", "--skip-train"),
            needs_train_knobs=False,
        )


class TrainOnlyStep(RunPipelineStep):
    def __init__(self) -> None:
        super().__init__(
            step_id="train_only",
            label="Train (GPU)",
            description=(
                "Train the WDSR model on pre-generated TFRecords. Skips "
                "scene generation and convolution."
            ),
            defaults=StepResources(
                partition="gpu", n_cpus=4, n_gpus=1,
                memory="32G", time_limit="24:00:00",
            ),
            needs_gpu=True,
            skip_flags=("--skip-generate", "--skip-convolve"),
            needs_train_knobs=True,
        )


class CustomTrainStep(RunPipelineStep):
    def __init__(self) -> None:
        super().__init__(
            step_id="custom",
            label="Custom (use form values, no auto --skip-* flags)",
            description=(
                "Use the resource and training knobs straight from the "
                "form. No ``--skip-*`` flags are added automatically; "
                "pass them through ``extra_flags`` if you need them."
            ),
            # The form's resource fields ultimately override these, so
            # the defaults here are a sane "fits everything" baseline
            # rather than a tuned preset.
            defaults=StepResources(
                partition="gpu", n_cpus=8, n_gpus=1,
                memory="32G", time_limit="12:00:00",
            ),
            needs_gpu=True,
            skip_flags=(),
            needs_train_knobs=True,
        )


# ---------------------------------------------------------------------------
# Registry — single source of truth for which steps exist
# ---------------------------------------------------------------------------

STEP_CLASSES: tuple[type[FASRCPipelineStep], ...] = (
    HSTDownloadStep,
    HSTPSFExtractStep,
    DifferentialKernelStep,
    HSTTFRecordStep,
    EuclidSkyDownloadStep,
    EuclidRoundtripTFRecordStep,
    HSTTrainStep,
    # Legacy ``run_pipeline.py`` presets (kept for the existing form).
    GenConvolveStep,
    ConvolveOnlyStep,
    TrainOnlyStep,
    CustomTrainStep,
)


@dataclass(frozen=True)
class StepRegistry:
    """Lookup helper: ``REGISTRY.get("kernel")`` → step instance."""

    by_id: Dict[str, FASRCPipelineStep] = field(default_factory=dict)

    @classmethod
    def build(cls) -> "StepRegistry":
        return cls(by_id={k.__name__ and step.step_id: step
                          for k in STEP_CLASSES
                          for step in [k()]})

    def get(self, step_id: str) -> FASRCPipelineStep:
        s = self.by_id.get(step_id)
        if s is None:
            raise KeyError(f"unknown HST pipeline step: {step_id!r}")
        return s

    def all(self) -> List[FASRCPipelineStep]:
        return list(self.by_id.values())


REGISTRY = StepRegistry.build()
