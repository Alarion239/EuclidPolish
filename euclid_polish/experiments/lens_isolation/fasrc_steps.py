"""Factory for additive lens-isolation FASRC pipeline steps."""

from __future__ import annotations

from typing import Any

EXPERIMENT_ROOT = "data/experiments/lens_isolation"


def build_step_classes(base_class, resources_class):
    """Return step classes without importing the registry back into itself."""

    class LensIsolationGenerateStep(base_class):
        def __init__(self):
            super().__init__(
                step_id="lens_isolation_generate",
                label="Generate normal lens-isolation pairs",
                job_name="lens-isolation-generate",
                defaults=resources_class(
                    partition="shared", n_cpus=16, n_gpus=0, memory="64G", time_limit="12:00:00"
                ),
            )

        def build_command(self, params: dict[str, Any]) -> list[str]:
            try:
                workers = int(params.get("n_cpus") or self.defaults.n_cpus)
            except (TypeError, ValueError):
                workers = self.defaults.n_cpus
            cmd = [
                "scripts/lens_isolation_generate.py",
                "--ntrain",
                str(int(params.get("ntrain", 6400) or 6400)),
                "--nvalid",
                str(int(params.get("nvalid", 100) or 100)),
                "--ntest",
                str(int(params.get("ntest", 100) or 100)),
                "--workers",
                str(max(1, workers)),
            ]
            seed = str(params.get("seed", "")).strip()
            if seed:
                cmd += ["--seed", seed]
            if str(params.get("force", "")).lower() in {"1", "true", "yes", "on"}:
                cmd.append("--force")
            return cmd

    class LensIsolationTrainStep(base_class):
        def __init__(self):
            super().__init__(
                step_id="lens_isolation_train",
                label="Train lens-isolation records",
                job_name="lens-isolation-train",
                defaults=resources_class(
                    partition="gpu", n_cpus=4, n_gpus=1, memory="32G", time_limit="48:00:00"
                ),
                needs_gpu=True,
            )

        def build_command(self, params: dict[str, Any]) -> list[str]:
            sources = str(params.get("sources", "")).strip()
            if not sources:
                raise ValueError("sources is required (comma-separated member names)")
            cmd = [
                "scripts/lens_isolation_train.py",
                "--sources",
                sources,
            ]
            for key in (
                "steps",
                "batch_size",
                "evaluate_every",
                "base_seed",
                "lr_peak",
                "lr_final",
                "lr_warmup_steps",
                "loss_norm",
                "noise_aug",
                "bootstrap",
            ):
                value = str(params.get(key, "")).strip()
                if value:
                    cmd += [f"--{key.replace('_', '-')}", value]
            return cmd

    class LensIsolationEvaluateStep(base_class):
        def __init__(self):
            super().__init__(
                step_id="lens_isolation_evaluate",
                label="Evaluate lens-isolation ensemble",
                job_name="lens-isolation-evaluate",
                defaults=resources_class(
                    partition="gpu", n_cpus=4, n_gpus=1, memory="32G", time_limit="8:00:00"
                ),
                needs_gpu=True,
            )

        def build_command(self, params: dict[str, Any]) -> list[str]:
            cmd = ["scripts/lens_isolation_evaluate.py"]
            for key in ("seed", "crop_size", "limit"):
                value = str(params.get(key, "")).strip()
                if value:
                    cmd += [f"--{key.replace('_', '-')}", value]
            return cmd

    return (
        LensIsolationGenerateStep,
        LensIsolationTrainStep,
        LensIsolationEvaluateStep,
    )
