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

from euclid_polish.config import Config
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
        # The WebUI bar is driven by the Reporter, so silence every tqdm
        # progress bar in the job — otherwise they flood the .err log with
        # redundant ASCII frames. ``tqdm.write`` status lines (.out) stay.
        export TQDM_DISABLE=1
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
            defaults=StepResources(
                partition="shared", n_cpus=1, n_gpus=0,
                # Peak ≈ a materialised ~500 MB HLSP tile + working copies
                # (~1–2 GB, half-side-independent) plus the held star stamps
                # (n_stars·(2·half+1)²·4 B → ~1 GB for 200 stars at 1023²)
                # and EPSFBuilder (~1 GB). 16 GB gives clear headroom for
                # large PSFs / higher n_stars; bump in the form if needed.
                memory="16G", time_limit="0:20:00",
            ),
            needs_gpu=False,
            fixed_cpus=1,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
        n_stars = int(params.get("n_stars", 200))
        half_side = int(params.get("half_side", 255))
        margin_frac = float(params.get("extract_margin_frac", 0.08))
        return [
            "scripts/fasrc_extract_hst_psf.py",
            "--n-stars", str(n_stars),
            "--half-side", str(half_side),
            "--extract-margin-frac", f"{margin_frac:g}",
        ]


class DifferentialKernelStep(FASRCPipelineStep):
    def __init__(self):
        super().__init__(
            step_id="kernel",
            label="3. Build differential kernel A = E / H",
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
            defaults=StepResources(
                partition="shared", n_cpus=16, n_gpus=0,
                memory="64G", time_limit="0:10:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
        n_train = int(params.get("n_train", 2000))
        n_valid = int(params.get("n_valid", 200))
        image_size = int(params.get("image_size", 256))
        max_relative_noise = float(params.get("max_relative_noise", 5.0))
        star_threshold_sigma = float(params.get("star_threshold_sigma", 20.0))
        min_source_sigma = float(params.get("min_source_sigma", 5.0))
        return [
            "scripts/fasrc_generate_hst_tfrecords.py",
            "--n-train", str(n_train),
            "--n-valid", str(n_valid),
            "--image-size", str(image_size),
            "--max-relative-noise", f"{max_relative_noise:g}",
            "--star-threshold-sigma", f"{star_threshold_sigma:g}",
            "--min-source-sigma", f"{min_source_sigma:g}",
        ]


class EuclidSkyDownloadStep(FASRCPipelineStep):
    def __init__(self):
        super().__init__(
            step_id="euclid_sky_download",
            label="5a. Download real Euclid sky cutouts (round-trip)",
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


class EuclidQueryStep(FASRCPipelineStep):
    def __init__(self):
        super().__init__(
            step_id="euclid_query",
            label="Query Euclid catalog (brightest N)",
            defaults=StepResources(
                partition="shared", n_cpus=1, n_gpus=0,
                memory="4G", time_limit="30:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
        num_stars = int(params.get("num_stars", 200) or 200)
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
    def __init__(self):
        super().__init__(
            step_id="euclid_verify_photometry",
            label="Verify photometry scale",
            defaults=StepResources(
                partition="shared", n_cpus=1, n_gpus=0,
                memory="8G", time_limit="30:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
        n    = int(params.get("n", 40) or 40)
        size = int(params.get("size", 256) or 256)
        return ["scripts/verify_star_photometry.py",
                "--n", str(n), "--size", str(size)]


class EuclidCutoutDownloadStep(FASRCPipelineStep):
    def __init__(self):
        super().__init__(
            step_id="download_euclid_cutouts",
            label="Download Euclid star cutouts (all bands)",
            defaults=StepResources(
                partition="shared", n_cpus=8, n_gpus=0,
                memory="16G", time_limit="2:00:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
        vis_pixels = int(params.get("vis_pixels", 512))
        workers    = int(params.get("workers", 8))
        return [
            "scripts/download_all_bands.py",
            "--vis-pixels", str(vis_pixels),
            "--workers",    str(workers),
        ]


class EuclidPSFExtractStep(FASRCPipelineStep):
    def __init__(self):
        super().__init__(
            step_id="extract_euclid_psf",
            label="Extract Euclid ePSFs (all 4 bands)",
            defaults=StepResources(
                partition="shared", n_cpus=8, n_gpus=0,
                # One band's stars held in memory + n_cpus EPSFBuilders (each
                # on a ~100-star cluster). Scale memory up with the CPU count
                # and the cutout size.
                memory="48G", time_limit="6:00:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
        vis_pixels    = int(params.get("vis_pixels", 512))
        stars_per_psf = int(params.get("stars_per_psf", 100))
        # Use every allocated CPU to build cluster PSFs in parallel.
        n_cpus = int(params.get("n_cpus", 8) or 8)
        cmd = [
            "scripts/extract_all_band_psfs.py",
            "--vis-pixels",    str(vis_pixels),
            "--stars-per-psf", str(stars_per_psf),
            "--max-procs",     str(n_cpus),
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


class TngSkirtAtlasDownloadStep(FASRCPipelineStep):
    """Bulk-download the whole IllustrisTNG TNG50-1 SKIRT atlas (~1153 galaxies).

    Each galaxy is rendered as Euclid VIS + NISP (Y/J/H) FITS from 5
    orientations; we keep only the dusty frames (20 FITS/galaxy) into
    ``$DATA_DIR/tng_skirt/<subhalo_id>/``. Network-I/O bound, so it runs one
    download thread per allocated CPU. The TNG API token is read on the node
    from ``$TNG_API_KEY`` or ``~/.tng_api_key`` — never from the form.
    """

    def __init__(self):
        super().__init__(
            step_id="download_tng_skirt",
            label="Download TNG50 SKIRT atlas (all galaxies, Euclid bands)",
            defaults=StepResources(
                # ~1153 multi-GB tarballs streamed, filtered to ~20 FITS each.
                # I/O bound → many threads; generous wall-clock for the full set
                # (re-submittable — finished galaxies carry a .done marker).
                partition="shared", n_cpus=16, n_gpus=0,
                memory="32G", time_limit="1-00:00:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
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
        return cmd


# Galaxy-selection modes for the grid/stack (mirrors euclid_polish.tng.selection
# .MODES). The pick happens *locally* (where the histogram property cache lives)
# and the chosen ids ride to the FASRC job via --ids/--id.
_TNG_MODES = ("random", "most_massive", "least_massive", "most_star_forming",
              "least_star_forming", "biggest_radius", "smallest_radius")


def _tng_mode(params: Dict[str, Any]) -> str:
    m = str(params.get("mode", "random")).strip().lower()
    return m if m in _TNG_MODES else "random"


def _tng_temperature(params: Dict[str, Any]) -> float:
    try:
        t = float(params.get("temperature", 0.3))
    except (TypeError, ValueError):
        t = 0.3
    return min(1.0, max(0.0, t))


def _tng_note(mode: str, temperature: float) -> str:
    pretty = mode.replace("_", " ")
    return pretty if mode == "random" else f"{pretty} · T={temperature:.2f}"


def _tng_select(mode: str, n: int, temperature: float) -> List[str]:
    """Select ``n`` galaxy ids locally by mode (empty if no property cache)."""
    # Lazy import: keeps matplotlib (pulled via tng.properties) out of the
    # pipeline module's import path. Resolved at call time so tests can patch.
    from euclid_polish.tng.selection import pick_by_mode
    return pick_by_mode(mode, n, temperature=temperature)


class TngGridStep(FASRCPipelineStep):
    """Render the 5×5 (galaxies × viewpoints) image grid as a CPU job →
    ``_infographics/grid.png``. Band ∈ VIS/Y/J/H/RGB; downsample ×1/×2/×4. The
    5 galaxies are chosen by ``mode`` (random, or the most/least extreme in
    stellar mass / SFR / radius) with a temperature-weighted draw."""

    def __init__(self):
        super().__init__(
            step_id="tng_grid",
            label="TNG infographic — 5×5 image grid",
            defaults=StepResources(
                # RGB loads up to 5×5×3 frames of 1600² (~0.8 GB) + Lupton
                # intermediates; 16 G is comfortable headroom.
                partition="shared", n_cpus=2, n_gpus=0,
                memory="16G", time_limit="0:30:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
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

    def __init__(self):
        super().__init__(
            step_id="tng_stack",
            label="TNG infographic — stacked FITS (5 viewpoints)",
            defaults=StepResources(
                partition="shared", n_cpus=1, n_gpus=0,
                memory="8G", time_limit="0:20:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
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


class EuclidStarAnchorTFRecordStep(FASRCPipelineStep):
    def __init__(self):
        super().__init__(
            step_id="euclid_star_anchor_tfrecords",
            label="Build star-anchor TFRecords (real Euclid stars → delta targets)",
            defaults=StepResources(
                partition="shared", n_cpus=4, n_gpus=0,
                memory="16G", time_limit="1:00:00",
            ),
            needs_gpu=False,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
        size        = int(params.get("size", 256))
        stamp       = int(params.get("stamp", 128))
        valid_every = int(params.get("valid_every", 10))
        cmd = [
            "scripts/fasrc_generate_star_anchor_tfrecords.py",
            "--size",        str(size),
            "--stamp",       str(stamp),
            "--valid-every", str(valid_every),
        ]
        # Optional SNR cut on the raw PSF photometry (blank/0 → keep all).
        snr_min = str(params.get("snr_min", "")).strip()
        if snr_min not in ("", "0"):
            cmd += ["--snr-min", f"{float(snr_min):g}"]
        # Optional row cap for a quick debugging pass (blank → all).
        limit = str(params.get("limit", "")).strip()
        if limit not in ("", "0"):
            cmd += ["--limit", str(int(float(limit)))]
        return cmd


class HSTTrainStep(FASRCPipelineStep):
    def __init__(self):
        super().__init__(
            step_id="train",
            label="6. Train WDSR with HST + star-anchor mix",
            defaults=StepResources(
                partition="gpu", n_cpus=4, n_gpus=1,
                memory="32G", time_limit="24:00:00",
            ),
            needs_gpu=True,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
        steps               = int(params.get("steps", 400_000))
        # Explicit per-lane batch composition (the batch is their sum). The
        # synthetic lane is always present; HST / star-anchor lanes are on
        # only when their count > 0.
        n_syn               = int(params.get("n_syn", Config.DEFAULT_BATCH_SIZE))
        n_hst               = int(params.get("n_hst", 0) or 0)
        n_anchor            = int(params.get("n_anchor", 0) or 0)
        w_syn               = float(params.get("save_best_w_syn", 1.0))
        w_hst               = float(params.get("save_best_w_hst", 1.0))
        w_anchor            = float(params.get("save_best_w_anchor", 0.0))
        lw_syn              = float(params.get("synthetic_loss_weight", 1.0))
        lw_hst              = float(params.get("hst_loss_weight", 1.0))
        lw_anchor           = float(params.get("star_anchor_loss_weight", 1.0))
        fwd_crop            = int(params.get("forward_op_crop_half", 0))
        learning_rate       = float(params.get("learning_rate", 0.0) or 0.0)
        nonneg_raw          = str(params.get("nonneg_sr_weight", "")).strip()
        # Checkbox: when checked the form sends a truthy value; unchecked →
        # absent → default (compute the resume baseline). Programmatic
        # callers that omit it keep the default too.
        overwrite_best      = str(params.get("overwrite_best", "")).strip() \
            in ("1", "true", "True", "on", "yes")
        cmd = [
            "scripts/fasrc_train_with_hst.py",
            "--steps", str(steps),
            "--n-syn", str(n_syn),
            "--save-best-w-syn", f"{w_syn:g}",
            "--save-best-w-hst", f"{w_hst:g}",
            "--save-best-w-anchor", f"{w_anchor:g}",
            "--synthetic-loss-weight", f"{lw_syn:g}",
        ]
        # Constant LR only when explicitly set; blank/0 keeps the default
        # decay schedule so an unspecified submit stays byte-identical.
        if learning_rate > 0:
            cmd += ["--learning-rate", f"{learning_rate:g}"]
        # SR non-negativity penalty weight, only when the form provides one
        # (blank → the script's Config.NONNEG_SR_WEIGHT default). 0 is a
        # valid value (disables the penalty), so emit on any non-blank.
        if nonneg_raw != "":
            cmd += ["--nonneg-sr-weight", f"{float(nonneg_raw):g}"]
        # Overwrite the previous best instead of computing a resume baseline
        # (e.g. after an architecture change). Default keeps the baseline.
        if overwrite_best:
            cmd += ["--no-resume-baseline"]
        # Emit a lane's flags only when its count > 0, so a supervised-only
        # submission stays minimal. ``--forward-op-crop-half`` now belongs to
        # the HST forward op (the anchor lane is operator-free).
        if n_hst > 0:
            cmd += [
                "--n-hst", str(n_hst),
                "--hst-loss-weight", f"{lw_hst:g}",
                "--forward-op-crop-half", str(fwd_crop),
            ]
        if n_anchor > 0:
            cmd += [
                "--n-anchor", str(n_anchor),
                "--star-anchor-loss-weight", f"{lw_anchor:g}",
            ]
        return cmd


# ---------------------------------------------------------------------------
# ``run_pipeline.py`` step base
# ---------------------------------------------------------------------------
#
# Shared base for jobs that shell out to ``scripts/run_pipeline.py``. The
# only surviving concrete step is :class:`SyntheticGenerateStep` (the
# synthetic training-pair generator). The old four-preset training form
# (gen_convolve / convolve_only / train_only / custom) was removed —
# training now goes exclusively through :class:`HSTTrainStep`.


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
    #: (n_train / n_valid / image_size / batch_size / steps).
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


class SyntheticGenerateStep(RunPipelineStep):
    """Synthetic training-pair generation for the /sky page.

    Renders synthetic clean HR scenes (COSMOS2025 galaxies + stars +
    strong lenses) and forward-models them to dirty Euclid LR with the
    empirical band PSFs, writing clean + HR + dirty TFRecords. Runs
    ``run_pipeline.py --skip-train`` as a dedicated, knob-bearing step
    card on /sky.
    """

    def __init__(self) -> None:
        super().__init__(
            step_id="synthetic_generate",
            label="Generate synthetic training pairs (CPU)",
            defaults=StepResources(
                partition="shared", n_cpus=16, n_gpus=0,
                memory="64G", time_limit="6:00:00",
            ),
            skip_flags=("--skip-train",),
            needs_train_knobs=True,
        )

    def build_command(self, params: Dict[str, Any]) -> List[str]:
        # Parallelise generation across the allocated CPUs: one process per
        # CPU runs the combined generate+forward pass on its index range.
        cmd = super().build_command(params)
        try:
            workers = int(params.get("n_cpus") or self.defaults.n_cpus)
        except (TypeError, ValueError):
            workers = self.defaults.n_cpus
        cmd += ["--gen-workers", str(max(1, workers))]
        # Proportion of galaxies drawn as real TNG50 stamps vs Sersic profiles.
        # Emit only when > 0 so a default submit stays byte-identical.
        try:
            tng_frac = float(params.get("tng_fraction") or 0.0)
        except (TypeError, ValueError):
            tng_frac = 0.0
        tng_frac = min(1.0, max(0.0, tng_frac))
        if tng_frac > 0.0:
            cmd += ["--tng-fraction", f"{tng_frac:g}"]
        # Star field knobs (from /config). Emit only when supplied so a default
        # submit stays unchanged.
        for param, flag in (("star_density_arcmin2", "--star-density-arcmin2"),
                            ("star_mag_slope",       "--star-mag-slope"),
                            ("star_mag_bright",      "--star-mag-bright"),
                            ("star_mag_faint",       "--star-mag-faint")):
            val = params.get(param)
            if val not in (None, ""):
                try:
                    cmd += [flag, f"{float(val):g}"]
                except (TypeError, ValueError):
                    pass
        return cmd


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
    EuclidQueryStep,
    EuclidVerifyPhotometryStep,
    EuclidCutoutDownloadStep,
    EuclidPSFExtractStep,
    TngSkirtAtlasDownloadStep,
    TngGridStep,
    TngStackStep,
    EuclidStarAnchorTFRecordStep,
    SyntheticGenerateStep,
    HSTTrainStep,
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
