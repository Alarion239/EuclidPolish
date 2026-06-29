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

import contextlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from euclid_polish.config import Config

CONFIG_DIR = os.path.expanduser("~/.euclid_polish")
CONFIG_PATH = os.path.join(CONFIG_DIR, "job_config.json")

# Which JobConfig attribute feeds which param of which FASRC step. The submit
# route injects these into the form before handing off, so the step cards no
# longer render the fields. Keyed by step_id → {param_name: jobconfig_attr}.
FASRC_STEP_PARAMS: dict[str, dict[str, str]] = {
    "download_euclid_cutouts": {"vis_pixels": "vis_pixels"},
    "extract_euclid_psf":      {"vis_pixels": "vis_pixels"},
    "synthetic_generate":      {"n_train": "n_train",
                                "n_valid": "n_valid",
                                "n_test": "n_test",
                                "image_size": "hr_image_size",
                                "star_density_arcmin2": "star_density_arcmin2",
                                "star_mag_slope": "star_mag_slope",
                                "star_mag_bright": "star_mag_bright",
                                "star_mag_faint": "star_mag_faint",
                                "lens_density_arcmin2": "lens_density_arcmin2",
                                "lens_sigma_v_min_kms": "lens_sigma_v_min_kms",
                                "lens_sigma_v_max_kms": "lens_sigma_v_max_kms"},
    "lensfinder_generate":     {"n_train": "lensfinder_n_fields",
                                "n_valid": "lensfinder_n_valid",
                                "image_size": "lensfinder_image_size",
                                "star_density_arcmin2": "star_density_arcmin2",
                                "star_mag_slope": "star_mag_slope",
                                "star_mag_bright": "star_mag_bright",
                                "star_mag_faint": "star_mag_faint",
                                "lens_density_arcmin2": "lens_density_arcmin2",
                                "lens_sigma_v_min_kms": "lens_sigma_v_min_kms",
                                "lens_sigma_v_max_kms": "lens_sigma_v_max_kms"},
    "lensfinder_train":        {"epochs": "lensfinder_epochs",
                                "patience": "lensfinder_patience",
                                "batch_size": "lensfinder_batch_size",
                                "learning_rate": "lensfinder_learning_rate",
                                "training_mode": "lensfinder_training_mode"},
}


def fasrc_params_for(step_id: str) -> dict[str, str]:
    """All job-config-derived params to inject into a FASRC step's form.

    Includes the direct ``FASRC_STEP_PARAMS`` mappings plus any *computed*
    params (currently the locked ePSF output size).
    """
    cfg = load()
    out = {param: str(getattr(cfg, attr))
           for param, attr in FASRC_STEP_PARAMS.get(step_id, {}).items()}
    if step_id == "extract_euclid_psf":
        # Output ePSF side (oversampled px) is locked to 2·(VIS cutout) + 1 —
        # the photutils convention ``cutout_size × oversampling + 1`` (with
        # oversampling = 2). The ``+1`` keeps it odd so the kernel has a true
        # centre sample. The field is removed from the form.
        out["output_size"] = str(2 * int(cfg.vis_pixels) + 1)
    return out


def _ensure_odd(n: int) -> int:
    """VIS cutout side must be odd (so the stamp has a true centre pixel)."""
    return n if n % 2 == 1 else n + 1


@dataclass
class JobConfig:
    # VIS cutout side in 0.10″/pix pixels. Shared by the Euclid cutout
    # download and the ePSF extraction (so they always match). Must be odd.
    vis_pixels:    int = 511
    # (stars_per_psf / min_stars_per_psf are PSF-extraction-specific and live
    #  on the /psfs extract step card, not here.)
    # Synthetic scene counts for generation (/sky).
    n_train:       int = 6400
    n_valid:       int = 100
    # Held-out test scenes — the eval set (train/save-best never touch it).
    n_test:        int = 100
    # HR scene side in 0.05″/pix pixels — feeds both synthetic generation
    # and inference. Kept a multiple of 6 (the NISP rebin factor).
    hr_image_size: int = 510
    # Brightness knee (e⁻) for the asinh display panels in inference.
    asinh_scale:   float = 1000.0
    # Star field: surface density (stars/arcmin²) + the smooth magnitude
    # distribution dN/dm ∝ 10^(slope·m) over [bright, faint] VIS mag.
    star_density_arcmin2: float = Config.DEFAULT_STAR_DENSITY_ARCMIN2
    star_mag_slope:       float = Config.STAR_MAG_SLOPE
    star_mag_bright:      float = Config.STAR_MAG_BRIGHT
    star_mag_faint:       float = Config.STAR_MAG_FAINT
    # Strong lenses: surface density + velocity-dispersion range (σ_v² sets the
    # Einstein radius θ_E via the SIS law).
    lens_density_arcmin2: float = Config.LENS_DENSITY_ARCMIN2
    lens_sigma_v_min_kms: float = Config.LENS_SIGMA_V_MIN_KMS
    lens_sigma_v_max_kms: float = Config.LENS_SIGMA_V_MAX_KMS
    # Lens-finder dataset: fewer, bigger fields generated into a dedicated
    # records dir (``data/images/records_lensfinder``) so the main 6400×510
    # training set is never clobbered. Stamps for the SR-vs-LR lens-finder are
    # cut from these. (Reuses the star/lens-density knobs above.)
    lensfinder_n_fields:   int = 800
    lensfinder_n_valid:    int = 80
    lensfinder_image_size: int = 2040
    # Lens-finder Zoobot fine-tuning knobs. ``epochs`` is the max-epoch ceiling;
    # ``patience`` drives early stopping on validation loss (training usually
    # stops well before the ceiling). Defaults mirror scripts/lensfinder_train.py.
    lensfinder_epochs:        int   = 10
    lensfinder_patience:      int   = 6
    lensfinder_batch_size:    int   = 64
    lensfinder_learning_rate: float = 1e-4
    # Zoobot fine-tune depth: "head_only" freezes the encoder and trains only
    # the linear head (fast, clean probe of the pretrained features);
    # "full" fine-tunes all encoder params (layer-decayed LR).
    lensfinder_training_mode: str   = "head_only"

    def to_dict(self) -> dict[str, Any]:
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
    with contextlib.suppress(OSError):
        os.chmod(CONFIG_PATH, 0o600)


def update(patch: dict[str, Any]) -> JobConfig:
    """Merge ``patch`` into the on-disk config and return the new state.

    Numeric fields are coerced from form strings; blanks are ignored so a
    partial form never wipes a value. VIS cutout is forced odd.
    """
    cfg = load()
    for k, v in patch.items():
        if not hasattr(cfg, k) or v is None or v == "":
            continue
        cur = getattr(cfg, k)
        if isinstance(cur, str):
            v = str(v)                       # string fields kept verbatim
        else:
            try:
                v = float(v) if isinstance(cur, float) else int(v)
            except (TypeError, ValueError):
                continue
        setattr(cfg, k, v)
    save(cfg)
    return cfg
