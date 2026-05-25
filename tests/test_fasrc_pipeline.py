"""Tests for FASRCPipelineStep base + concrete step registry."""

from __future__ import annotations

import shlex

import pytest

from euclid_polish.web import fasrc_config
from euclid_polish.web.fasrc_pipeline import (
    REGISTRY,
    DifferentialKernelStep,
    FASRCPipelineStep,
    HSTDownloadStep,
    HSTPSFExtractStep,
    HSTTFRecordStep,
    HSTTrainStep,
    StepRegistry,
    StepResources,
)


# ---------------------------------------------------------------------------
# StepResources
# ---------------------------------------------------------------------------

class TestStepResources:

    def test_defaults(self):
        r = StepResources()
        assert r.partition == "shared"
        assert r.n_cpus == 4
        assert r.n_gpus == 0

    def test_from_form_keeps_defaults_on_empty(self):
        defaults = StepResources(partition="shared", n_cpus=8, memory="16G")
        r = StepResources.from_form({}, defaults)
        assert r.n_cpus == 8
        assert r.memory == "16G"

    def test_from_form_coerces_ints(self):
        defaults = StepResources()
        r = StepResources.from_form({"n_cpus": "12", "n_gpus": "2"}, defaults)
        assert r.n_cpus == 12
        assert r.n_gpus == 2

    def test_from_form_rejects_garbage(self):
        with pytest.raises(ValueError, match="invalid resource"):
            StepResources.from_form({"n_cpus": "twelve"}, StepResources())


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_all_steps_present(self):
        """Registry must include the original 5 steps plus the two new
        round-trip steps (sky download + LR-only TFRecord build)."""
        ids = {s.step_id for s in REGISTRY.all()}
        assert ids == {
            "download", "extract_psf", "kernel", "tfrecords", "train",
            "euclid_sky_download", "euclid_roundtrip_tfrecords",
        }

    def test_lookup_by_id(self):
        assert isinstance(REGISTRY.get("kernel"), DifferentialKernelStep)
        assert isinstance(REGISTRY.get("download"), HSTDownloadStep)
        assert isinstance(REGISTRY.get("train"), HSTTrainStep)

    def test_unknown_step_raises(self):
        with pytest.raises(KeyError, match="unknown"):
            REGISTRY.get("nonexistent")

    def test_only_train_needs_gpu(self):
        gpu_steps = {s.step_id for s in REGISTRY.all() if s.needs_gpu}
        assert gpu_steps == {"train"}

    def test_extract_psf_is_single_threaded(self):
        step = REGISTRY.get("extract_psf")
        assert step.fixed_cpus == 1
        assert step.defaults.n_cpus == 1

    def test_other_steps_do_not_lock_cpus(self):
        """Only extract_psf should enforce fixed_cpus today."""
        locked = {s.step_id for s in REGISTRY.all() if s.fixed_cpus is not None}
        assert locked == {"extract_psf"}

    def test_every_step_has_label_and_description(self):
        for s in REGISTRY.all():
            assert s.label
            assert s.description
            assert s.step_id


# ---------------------------------------------------------------------------
# sbatch body rendering
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return fasrc_config.FasrcConfig(
        ssh_user="alice",
        repo_path="/n/foo/EuclidPolish",
        data_dir="/n/foo/data",
        ckpt_dir="/n/foo/ckpt",
        conda_env_path="/n/foo/conda",
    )


class TestSbatchRendering:

    def test_kernel_step_renders_one_argument(self, cfg):
        step = REGISTRY.get("kernel")
        out = step.build_sbatch_body(
            params={"regularisation": 1e-4},
            resources=step.defaults,
            cfg=cfg,
            label="kernel test",
        )
        body = out["body"]
        assert body.startswith("#!/bin/bash"), "body has residual indent"
        assert "--regularisation" in body
        assert "0.0001" in body
        assert "scripts/fasrc_compute_differential_kernel.py" in body
        assert "RUNTIME_SECONDS" in body
        assert "STEP_ID=kernel" in body

    def test_paths_use_step_id_and_timestamp(self, cfg):
        step = REGISTRY.get("download")
        out = step.build_sbatch_body(
            params={"n_tiles": 5},
            resources=step.defaults, cfg=cfg, label="x",
        )
        assert "hst-download-" in out["name"]
        assert out["script"].startswith("logs/hst_pipeline/")
        assert out["script"].endswith(".sh")
        assert out["out"].endswith(".out")
        assert out["err"].endswith(".err")

    def test_gpu_step_emits_gres_line(self, cfg):
        step = REGISTRY.get("train")
        out = step.build_sbatch_body(
            params={"steps": 1000, "batch_size": 8, "hst_fraction": 0.1},
            resources=step.defaults, cfg=cfg, label="x",
        )
        assert "--gres=gpu:1" in out["body"]
        assert "--partition=gpu" in out["body"]

    def test_cpu_step_omits_gres_line(self, cfg):
        # gpu:0 is silently rejected by some SLURM configs.
        step = REGISTRY.get("kernel")
        out = step.build_sbatch_body(
            params={"regularisation": 1e-3},
            resources=step.defaults, cfg=cfg, label="x",
        )
        assert "--gres=" not in out["body"]

    def test_fixed_cpus_renders_in_header(self, cfg):
        """extract_psf must always emit --cpus-per-task=1, no matter what."""
        step = REGISTRY.get("extract_psf")
        # Even if defaults say 1, render with the defaults to confirm.
        out = step.build_sbatch_body(
            params={"n_stars": 200},
            resources=step.defaults, cfg=cfg, label="x",
        )
        assert "--cpus-per-task=1" in out["body"]

    def test_resources_propagate_into_header(self, cfg):
        step = REGISTRY.get("tfrecords")
        custom = StepResources(
            partition="bigmem", n_cpus=32, n_gpus=0,
            memory="128G", time_limit="12:00:00",
        )
        out = step.build_sbatch_body(
            params={"n_train": 100, "n_valid": 10, "image_size": 256},
            resources=custom, cfg=cfg, label="x",
        )
        body = out["body"]
        assert "--partition=bigmem" in body
        assert "--cpus-per-task=32" in body
        assert "--mem=128G" in body
        assert "--time=12:00:00" in body

    def test_data_dir_exported(self, cfg):
        step = REGISTRY.get("download")
        out = step.build_sbatch_body(
            params={"n_tiles": 1},
            resources=step.defaults, cfg=cfg, label="x",
        )
        assert "EUCLID_POLISH_DATA_DIR" in out["body"]
        assert shlex.quote(cfg.data_dir) in out["body"]

    def test_train_step_omits_roundtrip_flag_when_zero(self, cfg):
        """Backward-compat invariant: ``roundtrip_fraction=0`` must
        produce the same argv it did before the round-trip feature
        existed — so re-submitting a pre-round-trip job stays
        byte-equivalent and doesn't accidentally enable the round-trip
        loss with no records on disk."""
        step = REGISTRY.get("train")
        out = step.build_sbatch_body(
            params={"steps": 100, "batch_size": 8, "hst_fraction": 0.1},
            resources=step.defaults, cfg=cfg, label="x",
        )
        assert "--roundtrip-fraction" not in out["body"], (
            "default rt=0 must not emit the flag — preserves "
            "pre-round-trip command line"
        )

    def test_train_step_emits_roundtrip_flag_when_positive(self, cfg):
        step = REGISTRY.get("train")
        out = step.build_sbatch_body(
            params={
                "steps": 100, "batch_size": 8,
                "hst_fraction": 0.2, "roundtrip_fraction": 0.2,
            },
            resources=step.defaults, cfg=cfg, label="x",
        )
        body = out["body"]
        assert "--hst-fraction" in body
        assert "0.2" in body
        assert "--roundtrip-fraction" in body

    def test_euclid_sky_download_step_args(self, cfg):
        step = REGISTRY.get("euclid_sky_download")
        out = step.build_sbatch_body(
            params={"n_positions": 50, "vis_pixels": 256},
            resources=step.defaults, cfg=cfg, label="x",
        )
        body = out["body"]
        assert "scripts/fasrc_download_euclid_sky_cutouts.py" in body
        assert "--n-positions" in body and "50" in body
        assert "--vis-pixels" in body and "256" in body
        assert "--ra-centre"  in body
        assert "--dec-centre" in body

    def test_euclid_roundtrip_tfrecords_step_args(self, cfg):
        step = REGISTRY.get("euclid_roundtrip_tfrecords")
        out = step.build_sbatch_body(
            params={"vis_pixels": 512, "stamp_size": 64, "valid_fraction": 0.15},
            resources=step.defaults, cfg=cfg, label="x",
        )
        body = out["body"]
        assert "scripts/fasrc_generate_euclid_roundtrip_tfrecords.py" in body
        assert "--vis-pixels" in body and "512" in body
        assert "--stamp-size" in body and "64" in body
        assert "--valid-fraction" in body and "0.15" in body

    def test_command_args_shell_quoted(self, cfg):
        """Even pathological params shouldn't break sbatch."""
        # Subclass to inject a problematic arg — the public API quotes argv.
        class _DangerStep(FASRCPipelineStep):
            def __init__(self):
                super().__init__(
                    step_id="danger", label="x", description="x",
                    defaults=StepResources(),
                )
            def build_command(self, params):
                return ["scripts/x.py", "--name", "has spaces", "--q", "'evil'"]
        out = _DangerStep().build_sbatch_body(
            params={}, resources=StepResources(),
            cfg=cfg, label="x",
        )
        # The dangerous arg must appear quoted.
        assert "'has spaces'" in out["body"]


# ---------------------------------------------------------------------------
# Sanity: every concrete subclass builds a non-empty command
# ---------------------------------------------------------------------------

class TestConcreteSteps:

    @pytest.mark.parametrize("step_cls", [
        HSTDownloadStep, HSTPSFExtractStep,
        DifferentialKernelStep, HSTTFRecordStep, HSTTrainStep,
    ])
    def test_builds_nonempty_command(self, step_cls):
        step = step_cls()
        argv = step.build_command({})    # defaults only
        assert argv
        assert argv[0].startswith("scripts/")
        assert all(isinstance(a, str) for a in argv)

    def test_train_passes_hst_fraction(self):
        argv = HSTTrainStep().build_command({"hst_fraction": 0.25})
        assert "--hst-fraction" in argv
        idx = argv.index("--hst-fraction")
        assert float(argv[idx + 1]) == pytest.approx(0.25)

    def test_tfrecords_passes_image_size(self):
        argv = HSTTFRecordStep().build_command({"image_size": 256})
        assert "--image-size" in argv
        idx = argv.index("--image-size")
        assert argv[idx + 1] == "256"


# ---------------------------------------------------------------------------
# Submit route enforces fixed_cpus
# ---------------------------------------------------------------------------

class TestFixedCpusEnforcement:
    """The submit route must override n_cpus to step.fixed_cpus."""

    def _stub_ssh(self, monkeypatch):
        """Make STATE.ssh report connected + capture sbatch commands."""
        from euclid_polish.web import remote
        class _StubSSH:
            calls: list = []
            def is_connected(self): return True
            def run(self, cmd, timeout=None):
                _StubSSH.calls.append(cmd)
                if cmd.startswith("mkdir"):
                    return (0, "", "")
                if "sbatch" in cmd:
                    return (0, "Submitted batch job 99999\n", "")
                return (0, "", "")
        stub = _StubSSH()
        monkeypatch.setattr(remote.STATE, "ssh", stub)
        return stub

    def test_form_n_cpus_overridden_by_fixed_cpus(self, monkeypatch):
        from euclid_polish.web.app import create_app
        self._stub_ssh(monkeypatch)
        app = create_app()
        client = app.test_client()
        r = client.post(
            "/api/fasrc/hst/extract_psf/submit",
            data={
                "n_cpus": "16",     # user tries to over-allocate
                "n_stars": "200",
                "memory": "8G",
                "time_limit": "2:00:00",
                "partition": "shared",
            },
        )
        j = r.get_json()
        assert r.status_code == 200
        assert j["ok"]
        # The recorded sbatch params reflect the override.
        assert j["params"]["n_cpus"] == 1

    def test_form_n_cpus_kept_when_no_fixed(self, monkeypatch):
        from euclid_polish.web.app import create_app
        self._stub_ssh(monkeypatch)
        app = create_app()
        client = app.test_client()
        r = client.post(
            "/api/fasrc/hst/tfrecords/submit",
            data={
                "n_cpus": "20",
                "n_train": "100",
                "n_valid": "10",
                "image_size": "256",
                "memory": "64G",
                "time_limit": "1:00:00",
                "partition": "shared",
            },
        )
        j = r.get_json()
        assert r.status_code == 200
        assert j["ok"]
        # tfrecords has no fixed_cpus → user's 20 should be honoured.
        assert j["params"]["n_cpus"] == 20
