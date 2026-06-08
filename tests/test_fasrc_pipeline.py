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
        round-trip steps (sky download + LR-only TFRecord build), the
        Euclid star-cutout steps (per-page cutout download + all-band ePSF
        extraction + star-anchor TFRecord build), and the synthetic
        generator. The legacy ``run_pipeline.py`` training presets were
        removed."""
        ids = {s.step_id for s in REGISTRY.all()}
        assert ids == {
            "download", "extract_psf", "kernel", "tfrecords", "train",
            "euclid_sky_download", "euclid_roundtrip_tfrecords",
            "euclid_query", "euclid_verify_photometry",
            "download_euclid_cutouts", "extract_euclid_psf",
            "euclid_star_anchor_tfrecords",
            "download_tng_skirt",
            "tng_grid", "tng_stack",
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

    def test_query_step_default_command(self):
        """The query step runs query_brightest_stars.py; window/SNR flags are
        emitted only when the form supplies them."""
        step = REGISTRY.get("euclid_query")
        argv = step.build_command({"num_stars": 3000})
        assert argv[0] == "scripts/query_brightest_stars.py"
        assert argv[argv.index("--num-stars") + 1] == "3000"
        assert "--magnitude-min" not in argv
        assert "--snr-min" not in argv

    def test_query_step_full_window(self):
        step = REGISTRY.get("euclid_query")
        argv = step.build_command({
            "num_stars": 3000, "magnitude_min": "16",
            "magnitude_limit": "19", "snr_min": "50"})
        assert argv[argv.index("--magnitude-min") + 1] == "16"
        assert argv[argv.index("--magnitude-limit") + 1] == "19"
        assert argv[argv.index("--snr-min") + 1] == "50"

    def test_verify_photometry_step_command(self):
        step = REGISTRY.get("euclid_verify_photometry")
        argv = step.build_command({"n": 60, "size": 512})
        assert argv[0] == "scripts/verify_star_photometry.py"
        assert argv[argv.index("--n") + 1] == "60"
        assert argv[argv.index("--size") + 1] == "512"

    def test_query_and_verify_steps_are_cpu_only(self):
        for sid in ("euclid_query", "euclid_verify_photometry"):
            step = REGISTRY.get(sid)
            assert step.needs_gpu is False
            assert step.defaults.n_gpus == 0

    def test_star_anchor_step_default_command(self):
        """Defaults reach the generator: 256-px cutouts, 128-px stamp,
        validate every 10th. The optional cuts are absent."""
        step = REGISTRY.get("euclid_star_anchor_tfrecords")
        argv = step.build_command({})
        assert argv[0] == "scripts/fasrc_generate_star_anchor_tfrecords.py"
        assert argv[argv.index("--size") + 1] == "256"
        assert argv[argv.index("--stamp") + 1] == "128"
        assert argv[argv.index("--valid-every") + 1] == "10"
        assert "--snr-min" not in argv
        assert "--limit" not in argv

    def test_star_anchor_step_optional_flags(self):
        """Form-supplied SNR cut + row limit reach the script."""
        step = REGISTRY.get("euclid_star_anchor_tfrecords")
        argv = step.build_command({"size": 512, "stamp": 160,
                                   "valid_every": 8, "snr_min": 50,
                                   "limit": 300})
        assert argv[argv.index("--size") + 1] == "512"
        assert argv[argv.index("--stamp") + 1] == "160"
        assert argv[argv.index("--snr-min") + 1] == "50"
        assert argv[argv.index("--limit") + 1] == "300"

    def test_star_anchor_step_blank_optionals_omitted(self):
        """Blank / 0 means 'no cut' / 'all rows' — the flag is dropped so
        the script keeps its own default behaviour."""
        step = REGISTRY.get("euclid_star_anchor_tfrecords")
        argv = step.build_command({"snr_min": "0", "limit": ""})
        assert "--snr-min" not in argv
        assert "--limit" not in argv

    def test_tng_skirt_step_default_command(self):
        """The atlas downloader runs the bulk script, ties --workers to the
        allocated CPU count, and (with no overrides) downloads the whole atlas
        — no --limit, no --keep-archive."""
        step = REGISTRY.get("download_tng_skirt")
        argv = step.build_command({"n_cpus": 16})
        assert argv[0] == "scripts/fasrc_download_tng_skirt_atlas.py"
        assert argv[argv.index("--workers") + 1] == "16"
        assert "--limit" not in argv
        assert "--keep-archive" not in argv

    def test_tng_skirt_step_optional_flags(self):
        """A galaxy cap and the keep-archive checkbox reach the script; a
        blank/0 limit is dropped so the default (all galaxies) holds."""
        step = REGISTRY.get("download_tng_skirt")
        argv = step.build_command(
            {"n_cpus": 8, "limit": 5, "keep_archive": "1"})
        assert argv[argv.index("--workers") + 1] == "8"
        assert argv[argv.index("--limit") + 1] == "5"
        assert "--keep-archive" in argv
        # Blank limit + unchecked box → neither flag.
        argv2 = step.build_command(
            {"n_cpus": 8, "limit": "", "keep_archive": ""})
        assert "--limit" not in argv2
        assert "--keep-archive" not in argv2

    def test_tng_skirt_step_explicit_workers_override_cpus(self):
        """A form-supplied worker count is used verbatim and is decoupled from
        the CPU allocation (downloads are I/O-bound → more threads than cores
        is fine)."""
        step = REGISTRY.get("download_tng_skirt")
        argv = step.build_command({"n_cpus": 8, "workers": 32})
        assert argv[argv.index("--workers") + 1] == "32"

    def test_tng_skirt_step_workers_fallback_to_cpus(self):
        """Blank / 0 / missing workers → fall back to the allocated CPU count."""
        step = REGISTRY.get("download_tng_skirt")
        for params in ({"n_cpus": 12, "workers": ""},
                       {"n_cpus": 12, "workers": "0"},
                       {"n_cpus": 12}):
            argv = step.build_command(params)
            assert argv[argv.index("--workers") + 1] == "12", params

    def test_tng_skirt_step_executor_default_is_process(self):
        """No executor in the form → the script runs the (true multi-core)
        process pool."""
        step = REGISTRY.get("download_tng_skirt")
        argv = step.build_command({"n_cpus": 16})
        assert argv[argv.index("--executor") + 1] == "process"

    def test_tng_skirt_step_executor_thread_passes_through(self):
        step = REGISTRY.get("download_tng_skirt")
        argv = step.build_command({"n_cpus": 16, "executor": "thread"})
        assert argv[argv.index("--executor") + 1] == "thread"

    def test_tng_skirt_step_executor_garbage_dropped(self):
        """An unrecognised executor value is not forwarded (the script keeps
        its own default)."""
        step = REGISTRY.get("download_tng_skirt")
        argv = step.build_command({"n_cpus": 16, "executor": "magic"})
        assert "--executor" not in argv

    def test_tng_skirt_step_is_cpu_only(self):
        step = REGISTRY.get("download_tng_skirt")
        assert step.needs_gpu is False
        assert step.defaults.n_gpus == 0
        assert step.fixed_cpus is None      # CPU count is user-editable

    _PICK = "euclid_polish.tng.selection.pick_by_mode"

    def test_tng_infographic_steps_run_the_script_and_save(self, monkeypatch):
        """The image-infographic jobs invoke the render script with --save (so
        they write the standard artifact path the /tng page fetches). The
        histogram is NOT a job — it renders locally."""
        monkeypatch.setattr(self._PICK, lambda *a, **k: [])
        assert "tng_histograms" not in REGISTRY.by_id
        for sid, mode in (("tng_grid", "grid"), ("tng_stack", "stack")):
            step = REGISTRY.get(sid)
            argv = step.build_command({})
            assert argv[0] == "scripts/fasrc_tng_infographic.py"
            assert argv[argv.index("--mode") + 1] == mode
            assert "--save" in argv
            assert step.needs_gpu is False and step.defaults.n_gpus == 0

    def test_tng_grid_step_band_and_downsample(self, monkeypatch):
        monkeypatch.setattr(self._PICK, lambda *a, **k: [])   # no cache
        step = REGISTRY.get("tng_grid")
        argv = step.build_command({"band": "rgb", "downsample": 4})
        assert argv[argv.index("--band") + 1] == "RGB"        # upper-cased
        assert argv[argv.index("--downsample") + 1] == "4"
        assert argv[argv.index("--seed") + 1] == "-1"         # random fallback
        # Bad values coerced to safe defaults.
        argv2 = step.build_command({"band": "ZZ", "downsample": 9})
        assert argv2[argv2.index("--band") + 1] == "VIS"
        assert argv2[argv2.index("--downsample") + 1] == "1"

    def test_tng_grid_step_mode_selects_ids_with_temperature(self, monkeypatch):
        """The grid picks galaxies locally by mode+temperature and passes them
        as --ids (plus a human note) — no --seed."""
        seen = {}

        def fake(mode, n, *, temperature=0.3, rng=None):
            seen.update(mode=mode, n=n, temperature=temperature)
            return ["167396", "504560", "275546", "1", "2"]
        monkeypatch.setattr(self._PICK, fake)
        argv = REGISTRY.get("tng_grid").build_command(
            {"band": "VIS", "mode": "most_massive", "temperature": "0.5"})
        assert seen == {"mode": "most_massive", "n": 5, "temperature": 0.5}
        assert argv[argv.index("--ids") + 1] == "167396,504560,275546,1,2"
        assert "most massive" in argv[argv.index("--note") + 1]
        assert "--seed" not in argv

    def test_tng_grid_step_unknown_mode_coerced_to_random(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            self._PICK, lambda mode, n, **k: seen.update(mode=mode) or [])
        REGISTRY.get("tng_grid").build_command({"mode": "bogus"})
        assert seen["mode"] == "random"

    def test_tng_stack_step_explicit_id_wins(self):
        argv = REGISTRY.get("tng_stack").build_command(
            {"band": "h", "galaxy_id": "167396"})
        assert argv[argv.index("--band") + 1] == "H"
        assert argv[argv.index("--id") + 1] == "167396"
        assert "--seed" not in argv

    def test_tng_stack_step_mode_picks_one(self, monkeypatch):
        monkeypatch.setattr(self._PICK, lambda *a, **k: ["999"])
        argv = REGISTRY.get("tng_stack").build_command(
            {"band": "VIS", "mode": "biggest_radius"})
        assert argv[argv.index("--id") + 1] == "999"

    def test_tng_stack_step_no_cache_falls_back_to_random(self, monkeypatch):
        monkeypatch.setattr(self._PICK, lambda *a, **k: [])
        argv = REGISTRY.get("tng_stack").build_command({"band": "VIS"})
        assert "--id" not in argv
        assert argv[argv.index("--seed") + 1] == "-1"

    def test_synthetic_generate_tng_fraction_flag(self):
        """The synthetic generator forwards a TNG fraction; 0/blank is omitted
        so a default submit stays byte-identical."""
        step = REGISTRY.get("synthetic_generate")
        base = {"n_train": 10, "n_valid": 2, "image_size": 252,
                "batch_size": 4, "steps": 100}
        argv = step.build_command({**base, "tng_fraction": "0.3"})
        assert argv[argv.index("--tng-fraction") + 1] == "0.3"
        assert "--tng-fraction" not in step.build_command(base)
        assert "--tng-fraction" not in step.build_command(
            {**base, "tng_fraction": "0"})

    def test_synthetic_generate_star_field_flags(self):
        """Star field knobs from /config reach run_pipeline; absent → omitted."""
        step = REGISTRY.get("synthetic_generate")
        base = {"n_train": 10, "n_valid": 2, "image_size": 252,
                "batch_size": 4, "steps": 100}
        argv = step.build_command({
            **base, "star_density_arcmin2": "3", "star_mag_slope": "0.25",
            "star_mag_bright": "10", "star_mag_faint": "24"})
        assert argv[argv.index("--star-density-arcmin2") + 1] == "3"
        assert argv[argv.index("--star-mag-slope") + 1] == "0.25"
        assert argv[argv.index("--star-mag-bright") + 1] == "10"
        assert argv[argv.index("--star-mag-faint") + 1] == "24"
        # absent → no flags
        plain = step.build_command(base)
        for flag in ("--star-density-arcmin2", "--star-mag-slope",
                     "--star-mag-bright", "--star-mag-faint"):
            assert flag not in plain

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

    def test_euclid_psf_extract_uses_chosen_cpu_count(self):
        """The all-band Euclid ePSF step no longer locks CPUs (the user picks
        the count in the form); it builds cluster PSFs in parallel across all
        allocated CPUs, passed through as ``--max-procs``."""
        step = REGISTRY.get("extract_euclid_psf")
        assert step.fixed_cpus is None              # CPU field is editable
        argv = step.build_command({"n_cpus": 16})
        assert argv[argv.index("--max-procs") + 1] == "16"

    def test_other_steps_do_not_lock_cpus(self):
        """Only the HST ePSF step (single-threaded → 1 CPU) still pins CPUs;
        the all-band Euclid ePSF now lets the user choose."""
        locked = {s.step_id for s in REGISTRY.all() if s.fixed_cpus is not None}
        assert locked == {"extract_psf"}

    def test_every_step_has_label(self):
        for s in REGISTRY.all():
            assert s.label
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
        """A pure-synthetic submit (no HST / star-anchor counts) emits only
        the synthetic lane — no HST or anchor flags leak in and enable a
        lane whose records may not exist."""
        step = REGISTRY.get("train")
        out = step.build_sbatch_body(
            params={"steps": 100, "n_syn": 8},
            resources=step.defaults, cfg=cfg, label="x",
        )
        body = out["body"]
        assert "--n-anchor" not in body
        assert "--n-hst" not in body

    def test_train_step_emits_lane_flags_when_positive(self, cfg):
        step = REGISTRY.get("train")
        out = step.build_sbatch_body(
            params={"steps": 100, "n_syn": 6, "n_hst": 2, "n_anchor": 2},
            resources=step.defaults, cfg=cfg, label="x",
        )
        body = out["body"]
        assert "--n-hst" in body
        assert "--n-anchor" in body

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
                    step_id="danger", label="x",
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
            {"n_syn": 24, "n_hst": 8, "n_anchor": 0})
        assert argv[argv.index("--n-syn") + 1] == "24"
        assert argv[argv.index("--n-hst") + 1] == "8"
        # HST lane on → its loss-weight flag rides along.
        assert "--hst-loss-weight" in argv
        # Star-anchor lane off → no anchor flags.
        assert "--n-anchor" not in argv
        assert "--star-anchor-loss-weight" not in argv

    def test_train_default_is_pure_synthetic(self):
        # No counts given → synthetic-only (n_syn default), no HST/anchor lanes.
        argv = HSTTrainStep().build_command({})
        assert "--n-syn" in argv
        assert "--n-hst" not in argv
        assert "--n-anchor" not in argv

    def test_train_emits_anchor_flags_when_n_anchor_set(self):
        argv = HSTTrainStep().build_command({"n_syn": 6, "n_anchor": 2})
        assert argv[argv.index("--n-anchor") + 1] == "2"
        assert "--star-anchor-loss-weight" in argv
        # forward-op-crop-half belongs to the HST lane now, not anchor.
        assert "--forward-op-crop-half" not in argv

    def test_train_emits_hst_crop_flag_when_n_hst_set(self):
        argv = HSTTrainStep().build_command({"n_syn": 6, "n_hst": 2})
        assert argv[argv.index("--n-hst") + 1] == "2"
        assert "--hst-loss-weight" in argv
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

    def test_tng_workers_field_reaches_sbatch_script(self, monkeypatch):
        """End-to-end: POST a workers value to the TNG submit endpoint and
        confirm the rendered sbatch body (written via the heredoc) carries
        ``--workers 32`` — decoupled from the 8 allocated CPUs. This proves the
        new form field flows form → build_command → script."""
        from euclid_polish.web.app import create_app
        stub = self._stub_ssh(monkeypatch)
        app = create_app()
        client = app.test_client()
        r = client.post(
            "/api/fasrc/hst/download_tng_skirt/submit",
            data={
                "confirm": "yes",
                "n_cpus": "8",      # allocation
                "n_gpus": "0",
                "workers": "32",    # independent worker count
                "memory": "32G",
                "time_limit": "1-00:00:00",
                "partition": "shared",
            },
        )
        j = r.get_json()
        assert r.status_code == 200 and j["ok"]
        # The CPU allocation stays at 8 …
        assert j["params"]["n_cpus"] == 8
        # … but the rendered script asks for 32 download workers. The sbatch
        # body is written via a captured ``cat > … <<EOF`` command.
        writes = [c for c in stub.calls if "--workers" in c]
        assert writes, "no sbatch write captured containing --workers"
        assert any("32" in c for c in writes)
        # And NOT the CPU count: 'workers 8' must not appear as the value.
        assert not any("--workers '8'" in c or "--workers 8\n" in c
                       for c in writes)

    def test_tng_workers_blank_falls_back_to_cpus_in_script(self, monkeypatch):
        """Blank workers field → the script is rendered with --workers = the
        allocated CPU count (8 here)."""
        from euclid_polish.web.app import create_app
        stub = self._stub_ssh(monkeypatch)
        app = create_app()
        client = app.test_client()
        r = client.post(
            "/api/fasrc/hst/download_tng_skirt/submit",
            data={
                "confirm": "yes",
                "n_cpus": "8", "n_gpus": "0",
                "workers": "",                 # blank → fall back
                "memory": "32G", "time_limit": "1-00:00:00",
                "partition": "shared",
            },
        )
        j = r.get_json()
        assert r.status_code == 200 and j["ok"]
        writes = [c for c in stub.calls if "--workers" in c]
        assert writes, "no sbatch write captured containing --workers"
        # 8 appears as the worker count somewhere in the rendered body.
        assert any("8" in c for c in writes)

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

