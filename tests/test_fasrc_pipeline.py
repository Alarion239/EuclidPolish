"""Tests for FASRCPipelineStep base + concrete step registry."""

from __future__ import annotations

import os
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
        """Registry must include the original HST steps (download,
        extract-PSF, kernel, TFRecord generation, WDSR train), the two
        round-trip steps (sky download + LR-only TFRecord build), the two
        Euclid star-cutout steps (per-page cutout download + all-band ePSF
        extraction), and the synthetic generator. The legacy
        ``run_pipeline.py`` training presets were removed."""
        ids = {s.step_id for s in REGISTRY.all()}
        assert ids == {
            "download", "extract_psf", "kernel", "tfrecords", "train",
            "euclid_sky_download", "euclid_roundtrip_tfrecords",
            "download_euclid_cutouts", "extract_euclid_psf",
            "synthetic_generate",
        }

    def test_tfrecords_step_passes_max_relative_noise(self):
        """The bright-stamp rejection threshold must reach the
        analytic-A generator. Without ``--max-relative-noise`` the
        script would default to its own value and the form knob would
        silently do nothing."""
        step = REGISTRY.get("tfrecords")
        argv = step.build_command({
            "n_train": 100, "n_valid": 10, "image_size": 256,
            "max_relative_noise": 7.5,
        })
        assert "scripts/fasrc_generate_hst_tfrecords.py" in argv
        assert "--max-relative-noise" in argv
        idx = argv.index("--max-relative-noise")
        assert float(argv[idx + 1]) == pytest.approx(7.5)

    def test_tfrecords_step_default_max_relative_noise(self):
        """Form omitted → default 5.0 reaches the script."""
        step = REGISTRY.get("tfrecords")
        argv = step.build_command({})
        assert "--max-relative-noise" in argv
        idx = argv.index("--max-relative-noise")
        assert float(argv[idx + 1]) == pytest.approx(5.0)

    def test_tfrecords_step_does_not_emit_legacy_model_flags(self):
        """The deleted two-stage chain CLI must not leak back in.
        Past argv had ``--transition-model`` / ``--frozen-denoiser`` /
        ``--frozen-denoiser-summary``; none of those should appear."""
        step = REGISTRY.get("tfrecords")
        argv = step.build_command({})
        for flag in ("--transition-model", "--frozen-denoiser",
                     "--frozen-denoiser-summary"):
            assert flag not in argv

    def test_lookup_by_id(self):
        assert isinstance(REGISTRY.get("kernel"), DifferentialKernelStep)
        assert isinstance(REGISTRY.get("download"), HSTDownloadStep)
        assert isinstance(REGISTRY.get("train"), HSTTrainStep)

    def test_unknown_step_raises(self):
        with pytest.raises(KeyError, match="unknown"):
            REGISTRY.get("nonexistent")

    def test_gpu_steps_are_the_expected_set(self):
        """The only GPU step is the HST/WDSR trainer; everything else
        (download, kernel, PSF extract, synthetic generate, …) is CPU."""
        gpu_steps = {s.step_id for s in REGISTRY.all() if s.needs_gpu}
        assert gpu_steps == {"train"}

    def test_extract_psf_is_single_threaded(self):
        step = REGISTRY.get("extract_psf")
        assert step.fixed_cpus == 1
        assert step.defaults.n_cpus == 1

    def test_euclid_psf_extract_locks_four_cpus(self):
        """The all-band Euclid ePSF step runs one process per band and
        locks the allocation to 4 CPUs."""
        step = REGISTRY.get("extract_euclid_psf")
        assert step.fixed_cpus == 4
        assert step.defaults.n_cpus == 4
        assert "--max-procs" in step.build_command({})

    def test_other_steps_do_not_lock_cpus(self):
        """Only the two PSF-extraction steps enforce fixed_cpus today: the
        HST ePSF (single-threaded → 1 CPU) and the all-band Euclid ePSF
        (one process per band → 4 CPUs)."""
        locked = {s.step_id for s in REGISTRY.all() if s.fixed_cpus is not None}
        assert locked == {"extract_psf", "extract_euclid_psf"}

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

    def test_gpu_step_first_line_is_valid_shebang(self, cfg):
        """REGRESSION: a previous version of build_sbatch_body inlined
        the gres directive with an embedded ``\\n`` inside the
        textwrap.dedent template. For n_gpus > 0 this broke the indent
        accounting and left the script with 12 leading spaces before
        ``#!/bin/bash``, which sbatch correctly rejected with "first
        line must start with #!".

        Any GPU configuration must produce a body that LITERALLY
        starts with ``#!/bin/bash\\n``. No leading whitespace, no
        empty line, no BOM."""
        for step_id in ("train",):
            step = REGISTRY.get(step_id)
            # Force n_gpus to a non-zero value even if the default is 0,
            # so the test exercises the gres branch regardless of step
            # defaults shifting in future.
            resources = step.defaults
            from dataclasses import replace
            resources_gpu = replace(resources, n_gpus=1, partition="gpu")
            out = step.build_sbatch_body(
                params={}, resources=resources_gpu, cfg=cfg,
                label=f"gpu shebang test {step_id}",
            )
            body = out["body"]
            assert body.startswith("#!/bin/bash\n"), (
                f"step {step_id!r} GPU body does not start with a "
                f"valid shebang. First 40 bytes (repr): "
                f"{body[:40]!r}"
            )
            assert "--gres=gpu:1" in body
            # And no SBATCH line should accidentally end up on the
            # same line as the gres directive due to a missing newline.
            assert "#SBATCH --gres=gpu:1\n" in body or \
                   "#SBATCH --gres=gpu:1\n#SBATCH" in body

    def test_cpu_step_omits_gres_line(self, cfg):
        # gpu:0 is silently rejected by some SLURM configs.
        step = REGISTRY.get("kernel")
        out = step.build_sbatch_body(
            params={"regularisation": 1e-3},
            resources=step.defaults, cfg=cfg, label="x",
        )
        assert "--gres=" not in out["body"]

    def test_all_steps_first_line_is_valid_shebang(self, cfg):
        """Every registered step must produce a body that begins with
        ``#!/bin/bash``. Sanity guard against future edits to the
        template that would break the dedent invariant."""
        from dataclasses import replace
        for step in REGISTRY.all():
            for n_gpus in (0, 1, 2):
                resources = replace(step.defaults, n_gpus=n_gpus)
                out = step.build_sbatch_body(
                    params={}, resources=resources, cfg=cfg,
                    label=f"shebang test {step.step_id} n_gpus={n_gpus}",
                )
                body = out["body"]
                first_line = body.split("\n", 1)[0]
                assert first_line == "#!/bin/bash", (
                    f"step {step.step_id!r} (n_gpus={n_gpus}) first "
                    f"line is {first_line!r}, not '#!/bin/bash'"
                )

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

    def test_train_step_omits_lane_flags_when_zero(self, cfg):
        """A pure-synthetic submit (no HST / round-trip counts) emits only
        the synthetic lane — no HST or round-trip flags leak in and enable
        a lane whose records may not exist."""
        step = REGISTRY.get("train")
        out = step.build_sbatch_body(
            params={"steps": 100, "n_syn": 8},
            resources=step.defaults, cfg=cfg, label="x",
        )
        body = out["body"]
        assert "--n-rt" not in body
        assert "--n-hst" not in body

    def test_train_step_emits_lane_flags_when_positive(self, cfg):
        step = REGISTRY.get("train")
        out = step.build_sbatch_body(
            params={"steps": 100, "n_syn": 6, "n_hst": 2, "n_rt": 2},
            resources=step.defaults, cfg=cfg, label="x",
        )
        body = out["body"]
        assert "--n-hst" in body
        assert "--n-rt" in body

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

    def test_train_passes_lane_counts(self):
        argv = HSTTrainStep().build_command(
            {"n_syn": 24, "n_hst": 8, "n_rt": 0})
        assert argv[argv.index("--n-syn") + 1] == "24"
        assert argv[argv.index("--n-hst") + 1] == "8"
        # HST lane on → its loss-weight flag rides along.
        assert "--hst-loss-weight" in argv
        # Round-trip lane off → no round-trip flags.
        assert "--n-rt" not in argv
        assert "--roundtrip-loss-weight" not in argv

    def test_train_default_is_pure_synthetic(self):
        # No counts given → synthetic-only (n_syn default), no HST/RT lanes.
        argv = HSTTrainStep().build_command({})
        assert "--n-syn" in argv
        assert "--n-hst" not in argv
        assert "--n-rt" not in argv

    def test_train_emits_roundtrip_flags_when_n_rt_set(self):
        argv = HSTTrainStep().build_command({"n_syn": 6, "n_rt": 2})
        assert argv[argv.index("--n-rt") + 1] == "2"
        assert "--roundtrip-loss-weight" in argv
        assert "--forward-op-crop-half" in argv

    def test_train_emits_constant_learning_rate_when_set(self):
        argv = HSTTrainStep().build_command({"learning_rate": 0.001})
        assert "--learning-rate" in argv
        idx = argv.index("--learning-rate")
        assert float(argv[idx + 1]) == pytest.approx(0.001)

    def test_train_emits_nonneg_sr_weight_when_set(self):
        # Non-blank → emitted (0 is valid and disables the penalty).
        for val, expect in ((2.5, 2.5), (0, 0.0)):
            argv = HSTTrainStep().build_command({"nonneg_sr_weight": val})
            assert "--nonneg-sr-weight" in argv
            idx = argv.index("--nonneg-sr-weight")
            assert float(argv[idx + 1]) == pytest.approx(expect)

    def test_train_omits_nonneg_sr_weight_when_blank(self):
        # Blank → not emitted (script falls back to Config.NONNEG_SR_WEIGHT).
        assert "--nonneg-sr-weight" not in HSTTrainStep().build_command({})
        assert "--nonneg-sr-weight" not in HSTTrainStep().build_command(
            {"nonneg_sr_weight": ""})

    def test_train_omits_learning_rate_when_blank(self):
        # Blank / 0 keeps the default decay schedule — no flag emitted, so an
        # unspecified submit stays byte-identical to before.
        assert "--learning-rate" not in HSTTrainStep().build_command({})
        assert "--learning-rate" not in HSTTrainStep().build_command(
            {"learning_rate": 0})
        assert "--learning-rate" not in HSTTrainStep().build_command(
            {"learning_rate": ""})

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
                "confirm": "yes",
                "n_cpus": "16",     # user tries to over-allocate
                "n_gpus": "0",      # all resource fields are required
                                    # (StepResources.from_form_strict)
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
                "confirm": "yes",
                "n_cpus": "20",
                "n_gpus": "0",      # all resource fields are required
                                    # (StepResources.from_form_strict)
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

    def test_submit_rejected_without_confirm_token(self, monkeypatch):
        """The server-side confirmation guard must refuse any POST that
        lacks ``confirm=yes`` — no SLURM script gets written, no sbatch
        gets called, no job ID is returned. This is the load-bearing
        defence against accidental/autofilled submissions.

        Without it, a stray ``fetch()`` from cached JS, a browser
        extension, or a programmatic POST could submit jobs to FASRC
        without the user ever seeing the confirmation dialog.
        """
        from euclid_polish.web.app import create_app
        stub = self._stub_ssh(monkeypatch)
        app = create_app()
        client = app.test_client()
        r = client.post(
            "/api/fasrc/hst/tfrecords/submit",
            data={
                # Intentionally NO "confirm" field.
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
        assert r.status_code == 400
        assert not j["ok"]
        assert "confirm" in j["error"].lower()
        # The stub SSH must not have been touched — no sbatch invocation
        # of any kind, no mkdir, nothing. This is the strongest possible
        # assertion that the guard short-circuited before doing any work.
        assert stub.calls == []

    def test_submit_rejected_with_invalid_confirm_value(self, monkeypatch):
        """Token values other than 'yes' / 'true' / '1' (case-insensitive)
        must also be rejected, so a typo or stray default can't sneak
        a submission through."""
        from euclid_polish.web.app import create_app
        stub = self._stub_ssh(monkeypatch)
        app = create_app()
        client = app.test_client()
        for bad in ("no", "false", "", "0", "maybe", "ok", "y"):
            r = client.post(
                "/api/fasrc/hst/extract_psf/submit",
                data={"confirm": bad, "n_stars": "10"},
            )
            assert r.status_code == 400, f"value {bad!r} should be rejected"
        assert stub.calls == []

    def test_create_app_does_not_overwrite_stubbed_ssh(self, monkeypatch):
        """REGRESSION — the load-bearing isolation property.

        ``create_app()`` used to call ``_try_startup_ssh_connect()``
        synchronously during construction, which read the developer's
        real FASRC config and (via their pre-existing ControlMaster
        socket) silently opened a real SSH session — overwriting any
        ``STATE.ssh = stub`` the test had installed moments earlier.
        That meant every test posting to a submit endpoint actually
        ran sbatch on real FASRC through the developer's credentials.

        This test verifies the kill-switch env var (set in
        ``tests/conftest.py``) keeps the auto-connect off, so a stub
        installed *before* ``create_app()`` survives and intercepts
        every SSH call the route makes.
        """
        from euclid_polish.web import remote
        from euclid_polish.web.app import create_app
        # Install the stub BEFORE create_app.
        stub = self._stub_ssh(monkeypatch)
        # Sanity: env var must be set by conftest.py — otherwise
        # the auto-connect would clobber the stub right here.
        assert os.environ.get("EUCLID_POLISH_DISABLE_AUTO_SSH") == "1", (
            "tests/conftest.py must set EUCLID_POLISH_DISABLE_AUTO_SSH=1 "
            "before any test imports the app — otherwise create_app() "
            "silently dials out to real FASRC and submits real SLURM "
            "jobs through every submit test."
        )
        _ = create_app()
        # After create_app, STATE.ssh MUST still be the stub.
        assert remote.STATE.ssh is stub, (
            f"create_app() overwrote the stub! STATE.ssh is now "
            f"{type(remote.STATE.ssh).__name__}. This means submit "
            f"tests will hit real FASRC. Check that "
            f"_try_startup_ssh_connect honours "
            f"EUCLID_POLISH_DISABLE_AUTO_SSH."
        )

