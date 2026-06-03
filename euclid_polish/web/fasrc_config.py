"""Persistent configuration for the FASRC web tab.

Lives in ``~/.euclid_polish/fasrc.json`` so it survives Flask restarts
and isn't tracked by git. The user can edit it directly or via the
"Settings" form in the FASRC tab.

All fields default to the values that match this developer's setup
(see git history / fasrc_train.sh); other users override them through
the UI on first use.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

CONFIG_DIR = os.path.expanduser("~/.euclid_polish")
CONFIG_PATH = os.path.join(CONFIG_DIR, "fasrc.json")


@dataclass
class FasrcConfig:
    # ── SSH ────────────────────────────────────────────────────────────────
    # Authentication is public-key only. Install your key once with
    # ``ssh-copy-id <user>@login.rc.fas.harvard.edu`` from any terminal.
    ssh_user:           str = ""
    ssh_host:           str = "login.rc.fas.harvard.edu"
    control_socket:     str = "/tmp/euclid-polish-fasrc.sock"
    control_persist:    str = "8h"

    # ── Remote paths ───────────────────────────────────────────────────────
    repo_path:          str = "/n/holylabs/lconnor_lab/Lab/abelotserkovtsev/EuclidPolish"
    conda_env_path:     str = "/n/holylabs/lconnor_lab/Lab/abelotserkovtsev/conda-env"
    data_dir:           str = "/n/netscratch/lconnor_lab/Lab/abelotserkovtsev/EuclidPolish/data"
    ckpt_dir:           str = "/n/netscratch/lconnor_lab/Lab/abelotserkovtsev/EuclidPolish/ckpt/wdsr"
    logs_subdir:        str = "logs"     # relative to repo_path
    # Persistent holylabs mirror for the experiment-tracking store. Blank
    # → "<repo_path>/tracking" (holylabs, NOT netscratch — netscratch is
    # purged). The local ./tracking folder is rsync'd up to here.
    tracking_remote_dir: str = ""

    # ── Local mirror for auto-pulled checkpoints ───────────────────────────
    local_ckpt_mirror:  str = ""         # blank → DEFAULT_CHECKPOINT_DIR

    # ── Default sbatch knobs (overridable per-submission) ──────────────────
    partition:          str = "gpu"
    n_gpus:             int = 1
    n_cpus:             int = 8
    memory:             str = "32G"
    time_limit:         str = "12:00:00"

    # ── Default training knobs ─────────────────────────────────────────────
    n_train:            int = 6400
    n_valid:            int = 200
    image_size:         int = 510
    batch_size:         int = 16
    steps:              int = 400_000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load() -> FasrcConfig:
    """Read the persisted config, falling back to defaults for missing keys."""
    if not os.path.isfile(CONFIG_PATH):
        return FasrcConfig()
    try:
        with open(CONFIG_PATH) as fp:
            data = json.load(fp) or {}
    except (OSError, json.JSONDecodeError):
        return FasrcConfig()
    cfg = FasrcConfig()
    for k, v in data.items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    return cfg


def save(cfg: FasrcConfig) -> None:
    # Refuse to persist obvious smoke-test breadcrumbs into the real
    # ``~/.euclid_polish/fasrc.json`` — temp dirs under ``/tmp`` /
    # ``/var/folders`` are not valid FASRC paths and once they leak in,
    # every subsequent sbatch fails with ``mkdir: cannot create
    # directory '/var/folders'`` because that path doesn't exist on the
    # remote. The guard only fires when CONFIG_PATH itself is the real
    # home-dir file, so integration tests that monkeypatch CONFIG_PATH
    # to point at a tmp config keep working.
    real_config = os.path.expanduser("~/.euclid_polish/fasrc.json")
    if os.path.abspath(CONFIG_PATH) == real_config:
        for field, value in (("repo_path", cfg.repo_path),
                             ("data_dir", cfg.data_dir),
                             ("ckpt_dir", cfg.ckpt_dir),
                             ("conda_env_path", cfg.conda_env_path)):
            if not value:
                continue
            if value.startswith(("/tmp/", "/var/folders/",
                                 "/private/var/folders/")):
                raise ValueError(
                    f"refusing to persist FasrcConfig.{field}={value!r} "
                    "into the real ~/.euclid_polish/fasrc.json: looks "
                    "like a smoke-test tmp dir. Use monkeypatch in tests."
                )
    os.makedirs(CONFIG_DIR, exist_ok=True)
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w") as fp:
        json.dump(cfg.to_dict(), fp, indent=2, sort_keys=True)
    os.replace(tmp, CONFIG_PATH)
    # Tighten perms — file may end up holding paths but never secrets.
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:
        pass


def update(patch: Dict[str, Any]) -> FasrcConfig:
    """Merge ``patch`` into the on-disk config and return the new state."""
    cfg = load()
    for k, v in patch.items():
        if hasattr(cfg, k) and v is not None and v != "":
            # Coerce numeric fields if the form sent strings.
            cur = getattr(cfg, k)
            if isinstance(cur, int) and not isinstance(v, bool):
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    continue
            setattr(cfg, k, v)
    save(cfg)
    return cfg
