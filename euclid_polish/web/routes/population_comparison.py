"""Routes for the field-statistics and population-comparison workspace."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from flask import jsonify, request

from euclid_polish.web import euclid_session, fasrc_config, fasrc_fetcher
from euclid_polish.web.helpers.paths import _sky_records_remote_dir
from euclid_polish.web.helpers.population_comparison import (
    availability,
    build_comparison,
    read_comparison,
    refresh_population_comparison,
)
from euclid_polish.web.helpers.vis_noise_calibration import (
    activate_vis_noise_candidate,
    default_sampling_manifest_path,
    fit_vis_noise_candidate,
    vis_noise_state,
)
from euclid_polish.web.jobs import REGISTRY
from euclid_polish.web.remote import ensure_ssh_connected


def register(app):
    @app.route("/api/population-comparison")
    def api_population_comparison():
        comparison = read_comparison()
        include_training = request.args.get(
            "include_training", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        if comparison is not None:
            comparison = dict(comparison)
            if include_training:
                comparison["population"] = comparison.get(
                    "population_with_training",
                    comparison.get("population"),
                )
            comparison.pop("population_with_training", None)
        return jsonify({
            "comparison": comparison,
            "availability": availability(),
            "authenticated": euclid_session.is_authenticated(),
            "vis_noise_calibration": vis_noise_state(),
        })

    @app.route("/api/population-comparison/build", methods=["POST"])
    def api_population_comparison_build():
        job_id = REGISTRY.spawn(
            label="population comparison: local fields",
            target=lambda cap: build_comparison(
                progress=lambda done, total, label: cap.tick(done, total, label)
            ),
        )
        return jsonify({"ok": True, "job_id": job_id})

    @app.post("/api/population-comparison/fit-vis-noise")
    def api_population_comparison_fit_vis_noise():
        def run(cap):
            cap.tick(0, 1, "fit source-masked VIS background noise")
            result = fit_vis_noise_candidate(
                progress=lambda done, total, label: cap.tick(done, total, label)
            )
            cap.tick(1, 1, "VIS noise candidate ready for review")
            cap.write(
                "fitted source-masked VIS background noise candidate "
                f"{str(result.get('fingerprint') or '')[:12]}…\n"
            )
            return result

        job_id = REGISTRY.spawn(
            label="population comparison: fit VIS background noise",
            target=run,
        )
        return jsonify({"ok": True, "job_id": job_id})

    @app.post("/api/population-comparison/activate-vis-noise")
    def api_population_comparison_activate_vis_noise():
        def run(cap):
            cap.tick(0, 1, "activate VIS background noise calibration")
            result = activate_vis_noise_candidate()
            cap.tick(1, 1, "VIS background noise calibration active")
            return result

        job_id = REGISTRY.spawn(
            label="population comparison: activate VIS background noise",
            target=run,
        )
        return jsonify({"ok": True, "job_id": job_id})

    @app.post("/api/population-comparison/sync-vis-noise-samples")
    def api_population_comparison_sync_vis_noise_samples():
        """Pull the completed FASRC manifest and VIS samples into local cache."""
        cfg = fasrc_config.load()
        remote_root = f"{cfg.data_dir}/euclid_sky/vis_noise_samples"
        remote_manifest = f"{remote_root}/vis_noise_sampling_manifest.json"

        def run(cap):
            cap.tick(0, 1, "connecting to FASRC")
            ensure_ssh_connected()
            manifest_result = fasrc_fetcher.fetch_one_file(
                remote_manifest,
                force=True,
                max_bytes=32 * 1024 * 1024,
            )
            if not manifest_result.ok or not manifest_result.local_path:
                raise RuntimeError(
                    manifest_result.error or "VIS-noise sampling manifest sync failed"
                )
            try:
                remote_payload = json.loads(
                    Path(manifest_result.local_path).read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("Synced VIS-noise manifest is unreadable") from exc
            if (
                not isinstance(remote_payload, dict)
                or remote_payload.get("kind") != "euclid_vis_noise_sampling"
                or remote_payload.get("version") != 1
            ):
                raise RuntimeError("Remote file is not a VIS-noise sampling manifest")

            samples = list(remote_payload.get("samples") or [])
            completed = [
                sample for sample in samples
                if isinstance(sample, dict)
                and sample.get("status") in {"written", "cached"}
            ]
            if not completed:
                raise RuntimeError("Remote VIS-noise manifest has no completed samples")

            protected = {manifest_result.local_path}
            local_manifest = default_sampling_manifest_path()
            local_cutouts = local_manifest.parent / "cutouts"
            local_cutouts.mkdir(parents=True, exist_ok=True)
            synced = 0
            failed = 0
            total = len(completed)
            for index, sample in enumerate(completed, start=1):
                sample_id = int(sample.get("sample_id", -1))
                remote_output = str(sample.get("output_path") or "").strip()
                if not remote_output:
                    remote_output = f"{remote_root}/cutouts/sky_{sample_id:04d}.fits"
                sample["remote_status"] = sample.get("status")
                sample["remote_output_path"] = remote_output
                if not remote_output.startswith(f"{remote_root}/cutouts/"):
                    sample["status"] = "failed"
                    sample["sync_error"] = "sample path is outside the VIS-noise area"
                    failed += 1
                    cap.tick(index, total, f"VIS field {index}/{total}")
                    continue
                result = fasrc_fetcher.fetch_one_file(
                    remote_output,
                    force=True,
                    max_bytes=512 * 1024 * 1024,
                    protect_paths=protected,
                )
                if result.ok and result.local_path:
                    protected.add(result.local_path)
                    installed = local_cutouts / f"sky_{sample_id:04d}.fits"
                    temporary_fits = installed.with_suffix(installed.suffix + ".tmp")
                    temporary_fits.unlink(missing_ok=True)
                    try:
                        os.link(result.local_path, temporary_fits)
                    except OSError:
                        shutil.copy2(result.local_path, temporary_fits)
                    os.replace(temporary_fits, installed)
                    sample["output_path"] = str(installed.resolve())
                    sample["status"] = "cached"
                    sample["sync_error"] = None
                    synced += 1
                else:
                    sample["status"] = "failed"
                    sample["sync_error"] = result.error or "sample sync failed"
                    failed += 1
                cap.tick(index, total, f"VIS field {index}/{total}")

            local_payload = dict(remote_payload)
            local_payload["samples"] = samples
            local_payload["sync"] = {
                "remote_manifest": remote_manifest,
                "remote_manifest_sha256": hashlib.sha256(
                    Path(manifest_result.local_path).read_bytes()
                ).hexdigest(),
                "completed_samples": synced,
                "failed_samples": failed,
            }
            local_manifest.parent.mkdir(parents=True, exist_ok=True)
            temporary = local_manifest.with_suffix(local_manifest.suffix + ".tmp")
            temporary.write_text(
                json.dumps(local_payload, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            os.replace(temporary, local_manifest)
            cap.write(
                f"synced {synced}/{total} independent VIS fields"
                + (f"; {failed} failed" if failed else "")
                + "\n"
            )
            return {
                "manifest_path": str(local_manifest),
                "completed_samples": synced,
                "failed_samples": failed,
            }

        job_id = REGISTRY.spawn(
            label="population comparison: sync VIS noise samples",
            target=run,
        )
        return jsonify({"ok": True, "job_id": job_id})

    @app.route("/api/population-comparison/sync-training-catalog", methods=["POST"])
    def api_population_comparison_sync_training_catalog():
        remote = f"{_sky_records_remote_dir()}/sources_train.csv"

        def run(cap):
            cap.tick(0, 3, "connecting to FASRC")
            ensure_ssh_connected()
            cap.tick(1, 3, "training source catalog")
            result = fasrc_fetcher.fetch_one_file(
                remote, force=True, max_bytes=1024 * 1024 * 1024
            )
            if not result.ok:
                raise RuntimeError(result.error or "training source catalog sync failed")
            cap.tick(2, 3, "population histograms")
            refresh_population_comparison()
            cap.tick(3, 3, "population histograms")
            cap.write(
                f"synced sources_train.csv ({result.size_bytes or 0:,} bytes)\n"
            )
            return {"path": result.local_path, "size_bytes": result.size_bytes}

        job_id = REGISTRY.spawn(
            label="population comparison: training source catalog",
            target=run,
        )
        return jsonify({"ok": True, "job_id": job_id})
