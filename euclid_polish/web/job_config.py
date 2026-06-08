"""Universal job configuration shared across the web UI.

A small set of knobs that several pages used to each carry their own copy
of (VIS cutout size, stars-per-PSF, scene counts, HR image size, asinh
scale). They now live here, are edited once on the ``/config`` tab, persist
to ``~/.euclid_polish/job_config.json`` (survives reloads/relaunches), and
are injected into the relevant job submissions server-side.

Kept separate from :mod:`euclid_polish.web.fasrc_config` (SSH + sbatch +
remote paths) — this file is purely the *scientific* per-job parameters a
user tweaks between runs.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict

CONFIG_DIR = os.path.expanduser("~/.euclid_polish")
CONFIG_PATH = os.path.join(CONFIG_DIR, "job_config.json")

# Which JobConfig attribute feeds which param of which FASRC step. The submit
# route injects these into the form before handing off, so the step cards no
# longer render the fields. Keyed by step_id → {param_name: jobconfig_attr}.
FASRC_STEP_PARAMS: Dict[str, Dict[str, str]] = {
    "download_euclid_cutouts": {"vis_pixels": "vis_pixels"},
    "extract_euclid_psf":      {"vis_pixels": "vis_pixels",
                                "stars_per_psf": "stars_per_psf"},
    "synthetic_generate":      {"n_train": "n_train",
                                "n_valid": "n_valid",
                                "image_size": "hr_image_size"},
}


def _ensure_odd(n: int) -> int:
    """VIS cutout side must be odd (so the stamp has a true centre pixel)."""
    return n if n % 2 == 1 else n + 1


@dataclass
class JobConfig:
    # VIS cutout side in 0.10″/pix pixels. Shared by the Euclid cutout
    # download and the ePSF extraction (so they always match). Must be odd.
    vis_pixels:    int = 511
    # Good stars per ePSF cluster (PSF varies across the field).
    stars_per_psf: int = 100
    # Synthetic scene counts for generation (/sky).
    n_train:       int = 6400
    n_valid:       int = 100
    # HR scene side in 0.05″/pix pixels — feeds both synthetic generation
    # and inference. Kept a multiple of 6 (the NISP rebin factor).
    hr_image_size: int = 510
    # Brightness knee (e⁻) for the asinh display panels in inference.
    asinh_scale:   float = 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load() -> JobConfig:
    """Read the persisted config, defaulting any missing key."""
    if not os.path.isfile(CONFIG_PATH):
        return JobConfig()
    try:
        with open(CONFIG_PATH) as fp:
            data = json.load(fp) or {}
    except (OSError, json.JSONDecodeError):
        return JobConfig()
    cfg = JobConfig()
    for k, v in data.items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    cfg.vis_pixels = _ensure_odd(int(cfg.vis_pixels))
    return cfg


def save(cfg: JobConfig) -> None:
    cfg.vis_pixels = _ensure_odd(int(cfg.vis_pixels))
    os.makedirs(CONFIG_DIR, exist_ok=True)
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w") as fp:
        json.dump(cfg.to_dict(), fp, indent=2, sort_keys=True)
    os.replace(tmp, CONFIG_PATH)
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:
        pass


def update(patch: Dict[str, Any]) -> JobConfig:
    """Merge ``patch`` into the on-disk config and return the new state.

    Numeric fields are coerced from form strings; blanks are ignored so a
    partial form never wipes a value. VIS cutout is forced odd.
    """
    cfg = load()
    for k, v in patch.items():
        if not hasattr(cfg, k) or v is None or v == "":
            continue
        cur = getattr(cfg, k)
        try:
            v = float(v) if isinstance(cur, float) else int(v)
        except (TypeError, ValueError):
            continue
        setattr(cfg, k, v)
    save(cfg)
    return cfg
