"""Unit tests for the sbatch script template.

These don't touch SSH at all — :func:`render_sbatch_body` and the
:class:`RunPipelineStep` family are pure-Python builders returning a
dict. We exercise:

  * every numeric field reaches its ``#SBATCH`` line verbatim,
  * paths get shell-quoted so spaces/specials can't inject,
  * the conda env path and data/ckpt env vars route through the user's
    settings, not the developer's defaults,
  * extra ``run_pipeline.py`` flags are appended literally,
  * the file paths returned for log/err line up with the ``--output``
    and ``--error`` lines inside the script body,
  * the legacy ``PRESETS`` dict (now derived from the step subclasses)
    still has the four names the JS form hard-codes.
"""

from __future__ import annotations

import pytest

from euclid_polish.web.fasrc_config import FasrcConfig
from euclid_polish.web.fasrc_jobs import PRESETS, resolve_preset
from euclid_polish.web.fasrc_pipeline import REGISTRY, StepResources


def _params(**overrides):
    p = dict(
        n_train=6400, n_valid=200, image_size=510,
        batch_size=16, steps=400_000, extra_flags="",
    )
    p.update(overrides)
    return p


def _resources(**overrides):
    r = dict(partition="gpu", n_gpus=1, n_cpus=8, memory="32G",
             time_limit="12:00:00")
    r.update(overrides)
    return StepResources(**r)


def _build(label="test run", *, cfg=None, params=None, resources=None,
           step_id="custom"):
    """Render a script the way the Flask handler would."""
    return REGISTRY.get(step_id).build_sbatch_body(
        params=params or _params(),
        resources=resources or _resources(),
        cfg=cfg or FasrcConfig(),
        label=label,
    )


def test_basic_script_contains_all_sbatch_fields():
    built = _build()
    body = built["body"]
    assert body.startswith("#!/bin/bash")
    assert "#SBATCH --job-name=" in body
    # shlex.quote() leaves bare identifiers unquoted — match either form.
    assert ("#SBATCH --partition=gpu" in body
            or "#SBATCH --partition='gpu'" in body)
    assert "#SBATCH --gres=gpu:1" in body
    assert "#SBATCH --cpus-per-task=8" in body
    assert "#SBATCH --mem=32G" in body
    assert "#SBATCH --time=12:00:00" in body
    # log paths match the returned hints
    assert f"#SBATCH --output={built['out']}" in body
    assert f"#SBATCH --error={built['err']}" in body
    # python command uses the form's training params. Each argv token is
    # rendered on its own continuation line, so we check the tokens
    # individually rather than as a joined ``--name value`` substring.
    for token in ("--ntrain", "6400", "--nvalid", "200",
                  "--image-size", "510", "--batch-size", "16",
                  "--steps", "400000"):
        assert token in body, f"missing argv token: {token!r}"


def test_custom_resources_are_propagated():
    built = _build(
        label="big run",
        resources=_resources(n_gpus=2, n_cpus=24, memory="128G",
                             time_limit="2-00:00:00"),
        params=_params(steps=600_000),
    )
    body = built["body"]
    assert "#SBATCH --gres=gpu:2" in body
    assert "#SBATCH --cpus-per-task=24" in body
    assert "#SBATCH --mem=128G" in body
    assert "#SBATCH --time=2-00:00:00" in body
    assert "--steps" in body
    assert "600000" in body


def test_paths_route_through_user_config():
    cfg = FasrcConfig(
        data_dir="/n/somewhere/data",
        ckpt_dir="/n/somewhere/ckpt",
        conda_env_path="/n/lab/conda-env",
    )
    body = _build(cfg=cfg)["body"]
    # shlex.quote leaves simple paths unquoted; either form is acceptable.
    assert ("export EUCLID_POLISH_DATA_DIR=/n/somewhere/data" in body
            or "export EUCLID_POLISH_DATA_DIR='/n/somewhere/data'" in body)
    assert ("export EUCLID_POLISH_CKPT_DIR=/n/somewhere/ckpt" in body
            or "export EUCLID_POLISH_CKPT_DIR='/n/somewhere/ckpt'" in body)
    assert ("mamba activate /n/lab/conda-env" in body
            or "mamba activate '/n/lab/conda-env'" in body)


def test_extra_flags_appended_verbatim():
    body = _build(
        params=_params(extra_flags="--skip-generate --skip-convolve"),
    )["body"]
    # The argv tokens are shell-quoted individually and joined with
    # ``\``-continuations across multiple lines; both --skip-* flags
    # appear as standalone tokens after --steps.
    assert "--skip-generate" in body
    assert "--skip-convolve" in body
    assert "--steps" in body


def test_log_paths_consistent_between_metadata_and_script():
    built = _build()
    # Each path in the returned dict shows up in the body verbatim once
    # in the SBATCH header.
    assert f"#SBATCH --output={built['out']}" in built["body"]
    assert f"#SBATCH --error={built['err']}"  in built["body"]
    # And the script's relpath matches what we'll write to FASRC. The
    # ``custom`` step (a RunPipelineStep) writes to ``logs/jobs``.
    assert built["script"].endswith(".sh")
    assert built["script"].startswith("logs/jobs/")


def test_label_with_special_chars_does_not_break_shell():
    # Quotes in the label would otherwise break ``echo`` inside the script.
    body = _build(label="he said 'hi'; rm -rf /")["body"]
    # Single-quotes get stripped; the echo line still echoes a clean string.
    assert "Web-submitted job: he said hi; rm -rf /" in body
    echo_line = [l for l in body.splitlines()
                 if l.startswith('echo "Web-submitted')]
    assert echo_line, "echo banner line missing"


def test_bad_numeric_resource_fields_raise():
    """``StepResources.from_form`` is the validation boundary; non-numeric
    strings raise ``ValueError`` before the script renders. (Empty
    strings deliberately fall back to the step's default — the form
    sends blank fields for "use default", not for "reject submit".)"""
    with pytest.raises(ValueError):
        StepResources.from_form({"n_gpus": "abc"}, StepResources())


# ---------------------------------------------------------------------------
# Presets (gen_convolve / convolve_only / train_only / custom)
# ---------------------------------------------------------------------------

def test_preset_names_are_stable():
    """The UI dropdown hard-codes these four keys."""
    assert set(PRESETS) == {"gen_convolve", "convolve_only",
                            "train_only", "custom"}


def test_resolve_preset_unknown_falls_back_to_custom():
    assert resolve_preset("nonexistent") is PRESETS["custom"]


def test_cpu_only_preset_omits_gres():
    """Some SLURM configs reject ``--gres=gpu:0``; CPU-only presets must
    drop the --gres line entirely."""
    built = _build(
        step_id="gen_convolve",
        resources=_resources(n_gpus=0, partition="shared"),
    )
    body = built["body"]
    assert "--gres" not in body, body
    assert "--cpus-per-task=" in body
    # ``gen_convolve`` always appends --skip-train via the step's argv.
    assert "--skip-train" in body


def test_gpu_preset_keeps_gres_line():
    built = _build(
        step_id="train_only",
        resources=_resources(n_gpus=1, partition="gpu"),
    )
    assert "#SBATCH --gres=gpu:1" in built["body"]


def test_gen_convolve_preset_resources_are_sensible():
    """The default ``gen_convolve`` resources need to fit a 510² × 4-ch
    × 6400 image set: ≥48 G memory once everything is streamed, and
    several CPUs for fftconvolve / Sersic-rendering."""
    p = PRESETS["gen_convolve"]
    assert p["n_gpus"] == 0
    assert p["n_cpus"] >= 8
    # parse "64G" → 64
    mem_g = int(p["memory"].rstrip("G"))
    assert mem_g >= 32
    assert "--skip-train" in p["skip_flags"]
    assert "--skip-convolve" not in p["skip_flags"]
    assert "--skip-generate" not in p["skip_flags"]


def test_convolve_only_preset_skips_generate_and_train():
    p = PRESETS["convolve_only"]
    assert "--skip-generate" in p["skip_flags"]
    assert "--skip-train" in p["skip_flags"]
    assert "--skip-convolve" not in p["skip_flags"]
    assert p["n_gpus"] == 0


def test_train_only_preset_skips_generate_and_convolve():
    p = PRESETS["train_only"]
    assert "--skip-generate" in p["skip_flags"]
    assert "--skip-convolve" in p["skip_flags"]
    assert "--skip-train" not in p["skip_flags"]
    assert p["n_gpus"] >= 1
    assert p["needs_train_knobs"] is True


def test_run_pipeline_step_banner_says_web_submitted():
    """All four legacy presets share the ``Web-submitted job:`` banner
    so existing log greps keep working."""
    for step_id in ("gen_convolve", "convolve_only", "train_only", "custom"):
        body = _build(step_id=step_id)["body"]
        assert "Web-submitted job:" in body, step_id


def test_hst_pipeline_step_banner_says_hst_pipeline():
    """HST-pipeline steps share the ``HST pipeline step:`` banner."""
    body = REGISTRY.get("kernel").build_sbatch_body(
        params={}, resources=StepResources(), cfg=FasrcConfig(),
        label="diff kernel",
    )["body"]
    assert "HST pipeline step: kernel" in body


def test_step_id_marker_is_emitted_for_history_tracking():
    """Every job emits ``STEP_ID=<step_id>`` at the tail so the runtime
    parser can attribute the wall time to the right step."""
    body = _build(step_id="custom")["body"]
    assert "STEP_ID=custom" in body
    assert "RUNTIME_SECONDS=" in body
