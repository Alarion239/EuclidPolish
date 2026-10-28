"""A long run's milestone checkpoints, measured along the way.

``Trainer.train(milestones=)`` keeps a member's state at chosen steps in
``<member>/milestones/step_<N>/`` (see ``--milestones``). This module measures
them without touching the member itself (its served checkpoint, its scores,
the registry):

* :func:`pull` copies the milestones and the member's ``origin.json``,
  ``provenance.json`` and ``training_log.csv`` from FASRC into
  ``<inference>/milestones/<member>/`` (a running member is fine);
* :func:`materialize` turns one milestone into an ensemble dir of its own —
  the member's metadata, that one checkpoint and the training log cut at its
  step — so it loads exactly like a member (:func:`load`);
* :func:`score` scores it on the universal scores' test fields (the same
  knee-PSNR curves and spectra as Models › Universal) into
  ``<universal scores>/milestones/<member>.json`` and ``<member>@<N>.npz``;
* ``milestone:<member>@<N>`` specs (:func:`parse_spec`) run it in
  ``scripts/jwst_resolution.py``.

``scripts/process_milestones.py`` drives all of it as milestones land.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shlex
import shutil
from pathlib import Path
from typing import Any

import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel
from euclid_polish.training.trainer import TRAINING_LOG_FILENAME
from euclid_polish.web import remote
from euclid_polish.web.helpers import atomic_files
from euclid_polish.web.helpers import ensemble_viz as ev
from euclid_polish.web.helpers import universal_models as um

_SPEC = re.compile(r"^milestone:(member_\d+)@(\d+)$")
_STEP_DIR = re.compile(r"^step_(\d{9})$")
#: The member files a milestone's own dir needs besides its checkpoint.
_METADATA = ("origin.json", "provenance.json")


def local_root() -> Path:
    """``<inference>/milestones``."""
    return Path(Config.EUCLID_INFERENCE_DIR) / "milestones"


def step_dirname(step: int) -> str:
    return f"step_{int(step):09d}"


def spec_for(member: str, step: int) -> str:
    return f"milestone:{member}@{int(step)}"


def parse_spec(spec: str) -> tuple[str, int]:
    """``milestone:member_257@200000`` → ``("member_257", 200000)``;
    :class:`ValueError` otherwise."""
    match = _SPEC.match(str(spec).strip())
    if not match:
        raise ValueError(f"not a milestone spec (milestone:member_NNN@STEP): {spec!r}")
    return match.group(1), int(match.group(2))


def _member_root(member: str) -> Path:
    if not re.fullmatch(r"member_\d+", member):
        raise ValueError(f"not a member name: {member!r}")
    return local_root() / member


def local_members() -> list[str]:
    """The members with milestones pulled, in number order."""
    root = local_root()
    names = [p.name for p in root.iterdir() if p.is_dir() and re.fullmatch(r"member_\d+", p.name)] \
        if root.is_dir() else []
    return sorted(names, key=lambda name: int(name.removeprefix("member_")))


def local_steps(member: str) -> list[int]:
    """The milestones pulled for ``member`` whose checkpoint is complete."""
    folder = _member_root(member) / "milestones"
    steps = []
    if folder.is_dir():
        for sub in folder.iterdir():
            match = _STEP_DIR.match(sub.name)
            if match and tf.train.latest_checkpoint(str(sub)):
                steps.append(int(match.group(1)))
    return sorted(steps)


def pull(member: str) -> list[int]:
    """Copy ``member``'s milestones (and its metadata and training log) from
    FASRC; returns the steps that arrived new. Needs the FASRC connection."""
    ssh = remote.STATE.ssh
    if ssh is None:
        raise RuntimeError("not connected to FASRC")
    before = set(local_steps(member))
    base = f"{ev.remote_ensemble_dir().rstrip('/')}/{member}"
    root = _member_root(member)
    rc, out, err = ssh.run(f"test -d {shlex.quote(base + '/milestones')} && echo yes || echo no", timeout=60)
    if (out or "").strip() == "yes":
        rc, _out, err = ssh.rsync_pull(f"{base}/milestones/", str(root / "milestones"), timeout=1800)
        if rc != 0:
            raise RuntimeError(f"rsync of {member}/milestones failed: {err.strip()[:300]}")
    for name in (*_METADATA, TRAINING_LOG_FILENAME):
        ssh.rsync_pull(f"{base}/{name}", str(root), timeout=600)
    return [s for s in local_steps(member) if s not in before]


def _log_until(src: Path, dst: Path, step: int) -> None:
    """``training_log.csv`` rows up to ``step`` (the milestone's own hours and
    steps for its score)."""
    if not src.is_file():
        return
    with open(src, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return
    kept = [r for r in rows if r.get("step") and int(float(r["step"])) <= step]
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(kept)


def materialize(member: str, step: int) -> str:
    """One milestone as an ensemble dir of its own:
    ``<root>/<member>/ensembles/<step dir>/<member>/`` with the member's
    metadata, the milestone's checkpoint (hard links, else copies), a
    ``checkpoint`` index naming it and the training log cut at ``step``;
    returns the ensemble dir (its parent). Idempotent."""
    root = _member_root(member)
    src = root / "milestones" / step_dirname(step)
    latest = tf.train.latest_checkpoint(str(src))
    if latest is None:
        raise FileNotFoundError(f"{member}: no milestone checkpoint at step {step} (pull it first)")
    ensemble = root / "ensembles" / step_dirname(step)
    dst = ensemble / member
    dst.mkdir(parents=True, exist_ok=True)
    prefix = os.path.basename(latest)
    for path in src.iterdir():
        if path.name.startswith(prefix + "."):
            target = dst / path.name
            if not target.exists():
                try:
                    os.link(path, target)
                except OSError:
                    shutil.copy2(path, target)
    for name in _METADATA:
        if (root / name).is_file():
            shutil.copy2(root / name, dst / name)
    _log_until(root / TRAINING_LOG_FILENAME, dst / TRAINING_LOG_FILENAME, int(step))
    (dst / "checkpoint").write_text(f'model_checkpoint_path: "{prefix}"\n'
                                    f'all_model_checkpoint_paths: "{prefix}"\n', encoding="utf-8")
    return str(ensemble)


def load(member: str, step: int) -> EnsembleModel:
    """The milestone as an ensemble of one (the ensemble-only rule)."""
    return EnsembleModel(materialize(member, step), labels=[member], include_universal=True, starless=False)


def scores_dir() -> Path:
    return Path(ev.universal_scores_dir()) / "milestones"


def read_scores(member: str) -> dict[str, Any]:
    """``{identity, steps: {"<N>": index entry}}`` (empty when none)."""
    try:
        return json.loads((scores_dir() / f"{member}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"identity": None, "steps": {}}


def score(cap, member: str, step: int) -> dict[str, Any]:
    """Score one milestone on the universal scores' test fields (the exact
    knee-PSNR curves and spectra of Models › Universal); the entry goes to
    ``<scores>/milestones/<member>.json`` and the curves to
    ``<member>@<N>.npz``. A store scored under another identity (fields,
    records, target) is started afresh."""
    records_dir = ev._sky_records_local_dir()
    identity = ev.universal_score_identity()
    if not records_dir or identity["records_fp"] is None:
        raise RuntimeError("the test records (dirty_test + hr_test) are not local")
    ensemble = materialize(member, step)
    member_dir = os.path.join(ensemble, member)
    n = len(identity["indices"])
    entry, curves, fields, record = um._score_model(cap, member, member_dir, identity, records_dir,
                                                    done=0, total=n, ensemble_dir=ensemble)
    store = scores_dir()
    store.mkdir(parents=True, exist_ok=True)
    index = read_scores(member)
    if index.get("identity") != identity:
        index = {"identity": identity, "steps": {}}
    entry = {**entry, "step": int(step)}
    index["steps"][str(int(step))] = entry
    um._save_curves(str(store), f"{member}@{int(step)}", fields, curves)
    atomic_files.write_json(store / f"{member}.json", index, indent=1)
    atomic_files.write_json(store / f"{member}@{int(step)}.spectra.json", record, default=float)
    return entry


# --------------------------------------------------------------------------- #
# the console: every measured milestone of every long run
# --------------------------------------------------------------------------- #

#: The bands of the JWST resolution reports, short → long wavelength.
RESOLUTION_BANDS = ("VIS", "Y_E", "J_E", "H_E")


def resolution_dir() -> Path:
    """Where ``scripts/process_milestones.py --res-dir`` writes the four
    JWST resolution reports (``resolution_<band>.json``)."""
    return Path(Config.DATA_DIR) / "jwst_euclid_overlap" / "milestones"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _knee_scan(model: dict[str, Any]) -> list[dict[str, float]]:
    """A model's ½-coherence resolution (arcsec) against JWST at each
    surface brightness of its knee scan."""
    out = []
    for e in model.get("knee_scan") or []:
        res = e.get("resolution_coherence") or {}
        if res.get("arcsec") is not None and e.get("sb") is not None:
            out.append({"multiple": float(e.get("multiple", 0.0)), "sb": float(e["sb"]),
                        "arcsec": float(res["arcsec"])})
    return out


def report_payload() -> dict[str, Any]:
    """Models › Universal › Milestones: every member with scored milestones,
    per step its synthetic score (∫PSNR per band and their mean, training
    hours, learned output knees), and per band the JWST resolution knee
    scans of its milestones beside the reference models measured in the same
    reports (production, other universal models, the input). Local files
    only."""
    reports = {band: _read_json(resolution_dir() / f"resolution_{band}.json") for band in RESOLUTION_BANDS}
    members = []
    for path in sorted(scores_dir().glob("member_*.json")):
        member = path.stem
        steps = read_scores(member).get("steps") or {}
        rows = []
        for key in sorted(steps, key=int):
            e = steps[key]
            rows.append({"step": int(key), "spec": spec_for(member, int(key)), "mean": e.get("mean"),
                         "integrated": e.get("integrated"), "hours": e.get("scored_hours"),
                         "learned_knees": e.get("learned_knees"), "scored_at": e.get("scored_at")})
        resolution = {}
        for band, rep in reports.items():
            models = {spec: _knee_scan(m) for spec, m in (rep.get("models") or {}).items()
                      if not spec.startswith("milestone:") or spec.startswith(f"milestone:{member}@")}
            if any(spec.startswith("milestone:") for spec in models):
                meta = rep.get("meta") or {}
                resolution[band] = {"jwst_filter": meta.get("jwst_filter"),
                                    "noise_sb": meta.get("euclid_noise_sb"),
                                    "models": {k: v for k, v in models.items() if v}}
        if rows:
            members.append({"member": member, "steps": rows, "resolution": resolution})
    return {"members": members}
