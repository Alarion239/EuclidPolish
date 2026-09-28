"""A tiny STARFULL ensemble for the model-study tests: three members (origin,
training log, checkpoint stub), test records (hr + dirty TFRecords), the
evaluation's test cubes, a production spatial gate with its diagnostic
payload, and blackout cubes — everything under ``tmp_path``, no TensorFlow
inference and no FASRC."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np

from euclid_polish import ensemble_registry as er
from euclid_polish.config import Config
from euclid_polish.ensemble import member_fingerprint
from euclid_polish.eval.spatial_gate import save_spatial_gate
from euclid_polish.image import Image, Role
from euclid_polish.image.tfio import write_images
from euclid_polish.web.helpers import ensemble_viz as ev
from tests._real_fixtures import uniform_gate

LABELS = ["01·psnr", "02·psnr", "03·psnr"]
RECIPES = [
    {"loss_norm": "l1", "asinh_knee": 10.0},
    {"loss_norm": "l2", "asinh_knee": 10.0},
    {"loss_norm": "l1", "asinh_knee": 100.0},
]
SIDE = 16
N_FIELDS = 3
NOISE = (2.0, 20.0, 6.0)
LOG_HEADER = ("step,wall_time,loss,psnr_stretched,psnr_raw,psnr_vis,psnr_y_e,psnr_j_e,"
              "psnr_h_e,gnorm_avg,gnorm_max,clip_norm,duration_s,combined_loss,is_baseline\n")


def _member(base: Path, i: int, recipe: dict) -> Path:
    d = base / f"member_{i:02d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "checkpoint").write_text('model_checkpoint_path: "ckpt-5"\n')
    (d / "ckpt-5.index").write_bytes(b"idx")
    rows = "".join(f"{s},1,0.5,{40 + s / 1000 + i},90,39,41,40,40,3.0,9.0,5,{100 + s / 100},0.1,\n"
                   for s in (1000, 2000))
    (d / "training_log.csv").write_text(LOG_HEADER + rows)
    (d / "origin.json").write_text(json.dumps({
        "seed": 1000 + i, "target_steps": 2000, "commit": "abc1234",
        "created_at": "2026-09-01T00:00:00Z", "noise_model": 5, **recipe}))
    return d


def _truth(rec: int) -> np.ndarray:
    rng = np.random.default_rng(100 + rec)
    return rng.exponential(200.0, (SIDE, SIDE, 4)).astype(np.float32)


def _lr(rec: int) -> np.ndarray:
    truth = _truth(rec)
    return truth.reshape(SIDE // 2, 2, SIDE // 2, 2, 4).sum(axis=(1, 3)).astype(np.float32)


def _members(rec: int, stamped: bool = False) -> list[np.ndarray]:
    rng = np.random.default_rng([rec, int(stamped)])
    truth = _truth(rec)
    return [(truth + rng.normal(0, s, truth.shape)).astype(np.float32) for s in NOISE]


def make_env(tmp_path: Path, monkeypatch, *, labels=LABELS, blackout_labels=None) -> dict:
    """Point Config + ensemble_viz at a fresh tiny ensemble (three active
    members; ``labels`` = the members the evaluation cubes and the gate were
    made for, ``blackout_labels`` those of the blackout cubes)."""
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR", str(tmp_path / "ckpt" / "wdsr"))
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    monkeypatch.setattr(Config, "TRACKING_DIR", str(tmp_path / "tracking"))
    monkeypatch.setattr(ev, "infer_checkpoint_num_res_blocks", lambda d: 32)
    base = tmp_path / "ckpt" / "ensemble"
    for i, recipe in enumerate(RECIPES, start=1):       # always three active members
        _member(base, i, recipe)
    er.load_registry(str(base))                          # bootstrap the registry now
    records = tmp_path / "records"
    hr = [Image(data=_truth(r), pixel_scale_arcsec=0.05, band_names=("VIS", "Y_E", "J_E", "H_E"),
                is_clean=True, role=Role.HR, index=r) for r in range(N_FIELDS)]
    lr = [Image(data=_lr(r), pixel_scale_arcsec=0.1, band_names=("VIS", "Y_E", "J_E", "H_E"),
                is_clean=False, role=Role.LR, index=r) for r in range(N_FIELDS)]
    write_images(hr, "hr_test", records_dir=str(records))
    write_images(lr, "dirty_test", records_dir=str(records))
    (records / "sources_test.csv").write_text(
        "field_index,type,x_pix,y_pix,flux_vis_e\n0,galaxy,4.0,5.0,100.0\n1,star,8.0,8.0,900.0\n")
    monkeypatch.setattr(ev, "_sky_records_local_dir", lambda: str(records))
    monkeypatch.setattr(ev, "_eval_records_fingerprint", lambda *a, **k: "fp")
    regime = tmp_path / "vis" / "ensemble" / "starfull"
    cubes = regime / "cubes"
    cubes.mkdir(parents=True)
    for rec in range(N_FIELDS):
        members = _members(rec)[:len(labels)]
        for i, member in enumerate(members):
            np.save(cubes / f"member{i}_{rec:05d}.npy", member)
        np.save(cubes / f"sr_{rec:05d}.npy", np.mean(members, axis=0))
        np.save(cubes / f"comb_spatial_gate_{rec:05d}.npy", members[0])
        np.save(cubes / f"lr_{rec:05d}.npy", _lr(rec))
    (cubes / "viz_index.json").write_text(json.dumps({
        "subset": "test", "indices": list(range(N_FIELDS)), "member_labels": list(labels),
        "records_fp": "fp", "target_psf_fwhm_arcsec": 0.066,
        "has_combiner_spatial_gate": True}))
    save_spatial_gate(uniform_gate(labels), str(regime / "spatial_gate_combiner"))
    manifest_path = regime / "spatial_gate_combiner" / "combiner.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fit_meta"] = {**(manifest.get("fit_meta") or {}), "promoted_from": "spatial_gate_p2"}
    manifest_path.write_text(json.dumps(manifest))
    write_eval_summary(regime, base, labels)
    n = len(labels)
    (regime / "spatial_gate_combiner_evals.json").write_text(json.dumps({
        "available": True, "member_labels": list(labels), "read_labels": list(labels),
        "band_names": ["VIS", "Y_E", "J_E", "H_E"],
        "gate_diagnostics": {
            "usage": {b: [round(1.0 / n, 4)] * n for b in ("VIS", "Y_E", "J_E", "H_E")},
            "usage_source": {b: [0.5] + [0.5 / (n - 1)] * (n - 1)
                             for b in ("VIS", "Y_E", "J_E", "H_E")},
            "usage_by_brightness": {b: [[1.0 / n] * n] * 5 for b in ("VIS", "Y_E", "J_E", "H_E")},
            "brightness_names": ["sky", "faint", "mid", "bright", "core"]}}))
    blackout = regime / "cubes_blackout"
    blackout.mkdir()
    for rec in (0, 2):
        stamped = _lr(rec)
        stamped[2:4, 2:4] = 0.0
        np.save(blackout / f"lr_{rec:05d}.npy", stamped)
        for i, member in enumerate(_members(rec, stamped=True)[:len(blackout_labels or labels)]):
            np.save(blackout / f"member{i}_{rec:05d}.npy", member)
    (blackout / "blackout_index.json").write_text(json.dumps({
        "identity": {"member_labels": list(blackout_labels or labels),
                     "well_fractions": [0.03, 0.1, 0.1, 0.1], "seed": 1, "source": None},
        "indices": [0, 2]}))
    return {"base": base, "records": records, "regime": regime, "cubes": cubes,
            "blackout": blackout, "labels": list(labels)}


def write_eval_summary(regime: Path, base: Path, labels) -> None:
    """The evaluation's identity: its members' checkpoint fingerprints and the
    production gate it baked (as ``job_ensemble_evaluate`` records them)."""
    (regime / "eval_summary.json").write_text(json.dumps({
        "member_labels": list(labels),
        "eval_identity": {
            "records_fp": "fp",
            "member_fps": [member_fingerprint(str(base / f"member_{str(lb).split('·')[0]}"))
                           for lb in labels],
            "combiner_fps": {"spatial_gate": ev._combiner_fingerprint(str(regime),
                                                                      "spatial_gate")}}}))


class FakeRemote:
    """An SSH double whose "remote" is a local directory: ``rsync_push`` /
    ``rsync_pull`` copy files, ``run`` understands ``mkdir -p``,
    ``sha256sum``, ``rm -rf``. ``corrupt`` names files whose remote copy gets
    flipped bytes (a failed transfer)."""

    def __init__(self, root: Path, *, connected: bool = True, corrupt=()) -> None:
        self.root = Path(root)
        self.connected = connected
        self.corrupt = set(corrupt)
        self.commands: list[str] = []
        self.pushed: list[str] = []
        self.staged_counts: list[int] = []
        self.pulled: list[str] = []
        #: When set, every pull waits for it (holds a fetch job running).
        self.pull_gate: threading.Event | None = None

    def is_connected(self) -> bool:
        return self.connected

    def local(self, remote_path: str) -> Path:
        return self.root / str(remote_path).lstrip("/")

    def run(self, cmd: str, timeout: int = 60):
        self.commands.append(cmd)
        argv = shlex.split(cmd)
        if argv[:2] == ["mkdir", "-p"]:
            self.local(argv[2]).mkdir(parents=True, exist_ok=True)
            return (0, "", "")
        if argv[0] == "sha256sum":
            path = self.local(argv[1])
            if not path.is_file():
                return (1, "", f"sha256sum: {argv[1]}: No such file")
            out = subprocess.run(["shasum", "-a", "256", str(path)], capture_output=True,
                                 text=True, check=True).stdout.split()[0]
            return (0, f"{out}  {argv[1]}\n", "")
        if argv[:2] == ["rm", "-rf"]:
            shutil.rmtree(self.local(argv[2]), ignore_errors=True)
            return (0, "", "")
        return (0, "", "")

    def rsync_push(self, local_path: str, remote_dir: str, extra_args=None, timeout=900):
        target = self.local(remote_dir)
        target.mkdir(parents=True, exist_ok=True)
        # How many files the local side held when this push ran (a field
        # upload must stage exactly one product at a time).
        self.staged_counts.append(sum(1 for p in Path(local_path).rglob("*") if p.is_file()))
        for path in Path(local_path).rglob("*"):
            if path.is_file():
                dest = target / path.relative_to(local_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
                self.pushed.append(str(dest.relative_to(self.root)))
                if path.name in self.corrupt:
                    with open(dest, "r+b") as handle:
                        handle.seek(0)
                        handle.write(b"\x00\x01\x02\x03")
        return (0, "", "")

    def rsync_pull(self, remote_path: str, local_dir: str, extra_args=None, timeout=600):
        if self.pull_gate is not None:
            self.pull_gate.wait(10)
        source = self.local(remote_path.rstrip("/"))
        os.makedirs(local_dir, exist_ok=True)
        self.pulled.append(remote_path)
        if not source.exists():
            return (23, "", "rsync: No such file or directory")
        if source.is_file():
            shutil.copy2(source, Path(local_dir) / source.name)
            return (0, "", "")
        for path in source.rglob("*"):
            if path.is_file():
                dest = Path(local_dir) / path.relative_to(source)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
        return (0, "", "")
