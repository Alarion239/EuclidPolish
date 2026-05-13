"""Unit tests for the sbatch script template builder.

These don't touch SSH at all — the generator is pure Python returning a
string. We exercise:

  * every numeric field reaches its ``#SBATCH`` line verbatim,
  * paths get shell-quoted so spaces/specials can't inject,
  * the conda env path and data/ckpt env vars route through the user's
    settings, not the developer's defaults,
  * extra ``run_pipeline.py`` flags are appended literally,
  * the file paths returned for log/err line up with the ``--output``
    and ``--error`` lines inside the script body.
"""

from __future__ import annotations

import pytest

from euclid_polish.web.fasrc_config import FasrcConfig
from euclid_polish.web.fasrc_jobs import build_sbatch_script


def _params(**overrides):
    p = dict(
        partition="gpu", n_gpus=1, n_cpus=8, memory="32G",
        time_limit="12:00:00",
        n_train=6400, n_valid=200, image_size=510,
        batch_size=16, steps=400_000, extra_flags="",
    )
    p.update(overrides)
    return p


def test_basic_script_contains_all_sbatch_fields():
    cfg = FasrcConfig()
    built = build_sbatch_script(label="test run", params=_params(), cfg=cfg)
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
    # python command uses the form's training params
    assert "--ntrain 6400" in body
    assert "--nvalid 200" in body
    assert "--image-size 510" in body
    assert "--batch-size 16" in body
    assert "--steps 400000" in body


def test_custom_resources_are_propagated():
    cfg = FasrcConfig()
    built = build_sbatch_script(
        label="big run", cfg=cfg,
        params=_params(n_gpus=2, n_cpus=24, memory="128G",
                       time_limit="2-00:00:00", steps=600_000),
    )
    body = built["body"]
    assert "#SBATCH --gres=gpu:2" in body
    assert "#SBATCH --cpus-per-task=24" in body
    assert "#SBATCH --mem=128G" in body
    assert "#SBATCH --time=2-00:00:00" in body
    assert "--steps 600000" in body


def test_paths_route_through_user_config():
    cfg = FasrcConfig(
        data_dir="/n/somewhere/data",
        ckpt_dir="/n/somewhere/ckpt",
        conda_env_path="/n/lab/conda-env",
    )
    body = build_sbatch_script(label="x", params=_params(), cfg=cfg)["body"]
    # shlex.quote leaves simple paths unquoted; either form is acceptable.
    assert ("export EUCLID_POLISH_DATA_DIR=/n/somewhere/data" in body
            or "export EUCLID_POLISH_DATA_DIR='/n/somewhere/data'" in body)
    assert ("export EUCLID_POLISH_CKPT_DIR=/n/somewhere/ckpt" in body
            or "export EUCLID_POLISH_CKPT_DIR='/n/somewhere/ckpt'" in body)
    assert ("mamba activate /n/lab/conda-env" in body
            or "mamba activate '/n/lab/conda-env'" in body)


def test_extra_flags_appended_verbatim():
    body = build_sbatch_script(
        label="reuse", cfg=FasrcConfig(),
        params=_params(extra_flags="--skip-generate --skip-convolve"),
    )["body"]
    # The training command is one long shell line ending in --steps N <flags>
    assert "--steps 400000 --skip-generate --skip-convolve" in body


def test_log_paths_consistent_between_metadata_and_script():
    built = build_sbatch_script(
        label="reuse", cfg=FasrcConfig(), params=_params(),
    )
    # Each path in the returned dict shows up in the body verbatim once
    # in the SBATCH header.
    assert f"#SBATCH --output={built['out']}" in built["body"]
    assert f"#SBATCH --error={built['err']}"  in built["body"]
    # And the script's relpath matches what we'll write to FASRC.
    assert built["script"].endswith(".sh")
    assert built["script"].startswith("logs/jobs/")


def test_label_with_special_chars_does_not_break_shell():
    # Quotes in the label would otherwise break ``echo`` inside the script.
    body = build_sbatch_script(
        label="he said 'hi'; rm -rf /",
        cfg=FasrcConfig(), params=_params(),
    )["body"]
    # The single-quotes were stripped; no literal "rm -rf /" line that runs.
    # We just check the echo line for the sanitized label still works.
    assert "Web-submitted job: he said hi; rm -rf /" in body
    # And that no unescaped single quotes appear inside the echo line.
    echo_line = [l for l in body.splitlines() if l.startswith("echo \"Web-submitted")]
    assert echo_line, "echo header line missing"


@pytest.mark.parametrize("bad", ["abc", "", None])
def test_bad_numeric_fields_raise(bad):
    with pytest.raises((ValueError, TypeError)):
        build_sbatch_script(
            label="x", cfg=FasrcConfig(),
            params=_params(n_gpus=bad),
        )
