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
import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
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
                                "galaxy_density_arcmin2": "galaxy_density_arcmin2",
                                "star_density_arcmin2": "star_density_arcmin2",
                                "lens_density_arcmin2": "lens_density_arcmin2",
                                "lens_sigma_v_min_kms": "lens_sigma_v_min_kms",
                                "lens_sigma_v_max_kms": "lens_sigma_v_max_kms",
                                "psf_warp_prob": "psf_warp_prob",
                                "psf_warp_alpha_max": "psf_warp_alpha_max",
                                "psf_warp_sigma": "psf_warp_sigma",
                                "saturation_mask_prob": "saturation_mask_prob"},
    # WDSR SR training: LR schedule (warmup → cosine) + reduce-LR-on-plateau
    # guard. build_command translates these into --lr-* / --plateau-lr-* flags.
    "ensemble_train":          {"lr_peak": "lr_peak",
                                "lr_final": "lr_final",
                                "lr_warmup_steps": "lr_warmup_steps",
                                "psf_warp_prob": "psf_warp_prob",
                                "psf_warp_alpha_max": "psf_warp_alpha_max",
                                "psf_warp_sigma": "psf_warp_sigma",
                                "saturation_mask_prob": "saturation_mask_prob",
                                "plateau_lr_enabled": "plateau_lr_enabled",
                                "plateau_lr_factor": "plateau_lr_factor",
                                "plateau_lr_patience": "plateau_lr_patience",
                                "plateau_lr_min_delta": "plateau_lr_min_delta",
                                "plateau_lr_cooldown": "plateau_lr_cooldown",
                                "plateau_lr_min_lr": "plateau_lr_min_lr",
                                "plateau_lr_metric": "plateau_lr_metric"},
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
    # Generated TNG-galaxy density. The activated Q1 continuous
    # bright/main/flat count law owns the exact value.
    galaxy_density_arcmin2: float = Config.GALAXY_DENSITY_ARCMIN2
    # Brightness knee (e⁻) for the asinh display panels in inference.
    asinh_scale:   float = 1000.0
    # WDSR SR training — LR schedule (warmup → cosine) that decays smoothly from
    # the start (kills the old flat-5e-4 "skip-only" plateau) + reduce-LR-on-
    # plateau guard (cut the LR when the val metric stalls). Feed the
    # ensemble_train FASRC step. plateau_lr_enabled is 0/1 (form-coerced int).
    lr_peak:              float = Config.LR_PEAK
    lr_final:             float = Config.LR_FINAL
    lr_warmup_steps:      int   = Config.LR_WARMUP_STEPS
    # PSF-distribution augmentation shared by synthetic generation and live
    # training. Generated train/validate/test dirty exposures receive seeded,
    # reproducible draws; clean/HR targets are never deformed. In clean-only
    # on-the-fly training, train instead draws a fresh warp on every visit.
    psf_warp_prob:        float = Config.TRAIN_PSF_WARP_PROB
    psf_warp_alpha_max:   float = Config.TRAIN_PSF_WARP_ALPHA_MAX
    psf_warp_sigma:       float = Config.TRAIN_PSF_WARP_SIGMA
    saturation_mask_prob: float = Config.TRAIN_SATURATION_MASK_PROB
    plateau_lr_enabled:   int   = int(Config.PLATEAU_LR_ENABLED)
    plateau_lr_factor:    float = Config.PLATEAU_LR_FACTOR
    plateau_lr_patience:  int   = Config.PLATEAU_LR_PATIENCE
    plateau_lr_min_delta: float = Config.PLATEAU_LR_MIN_DELTA
    plateau_lr_cooldown:  int   = Config.PLATEAU_LR_COOLDOWN
    plateau_lr_min_lr:    float = Config.PLATEAU_LR_MIN_LR
    plateau_lr_metric:    str   = Config.PLATEAU_LR_METRIC
    # Star field surface density; the activated calibration owns magnitudes.
    star_density_arcmin2: float = Config.DEFAULT_STAR_DENSITY_ARCMIN2
    # Strong lenses: surface density + velocity-dispersion range (σ_v² sets the
    # Einstein radius θ_E via the SIS law).
    lens_density_arcmin2: float = Config.LENS_DENSITY_ARCMIN2
    lens_sigma_v_min_kms: float = Config.LENS_SIGMA_V_MIN_KMS
    lens_sigma_v_max_kms: float = Config.LENS_SIGMA_V_MAX_KMS
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FLAG_WORDS = {"1": True, "true": True, "yes": True, "on": True,
               "0": False, "false": False, "no": False, "off": False}


def parse_flag(raw: Any) -> bool | None:
    """A form/JSON flag (``1/0``, ``true/false``, ``yes/no``, ``on/off``, or a
    real bool) as a bool; ``None`` when it is not a flag word."""
    if isinstance(raw, bool):
        return raw
    return _FLAG_WORDS.get(str(raw).strip().lower())


def _field_type(name: str) -> type | None:
    """The declared type of a :class:`JobConfig` field (its default's type)."""
    for field in fields(JobConfig):
        if field.name == name:
            return type(field.default)
    return None


def coerce_field(name: str, raw: Any) -> Any:
    """``raw`` (a form string or a JSON value) as field ``name``'s declared
    type, or ``None`` when it is not a field or does not convert.

    ``str`` fields are kept verbatim; ``int`` fields accept whole numbers
    (``"5"``, ``"5.0"``) and flag words / bools as 0/1 (``plateau_lr_enabled``);
    ``float`` fields accept numbers.
    """
    kind = _field_type(name)
    if kind is None or raw is None:
        return None
    if kind is str:
        return str(raw)
    if kind is bool:
        return parse_flag(raw)
    if isinstance(raw, bool):
        return kind(raw)
    try:
        number = float(str(raw).strip())
    except ValueError:
        flag = parse_flag(raw) if kind is int else None
        return None if flag is None else int(flag)
    if kind is float:
        return number
    return int(number) if number.is_integer() else None


def load() -> JobConfig:
    """Read the persisted config, defaulting any missing key.

    Values are coerced to each field's declared type (a flag saved as a JSON
    ``true`` loads as ``1``); an unconvertible value keeps the default.
    """
    if not os.path.isfile(CONFIG_PATH):
        return JobConfig()
    try:
        with open(CONFIG_PATH) as fp:
            data = json.load(fp) or {}
    except (OSError, json.JSONDecodeError):
        return JobConfig()
    cfg = JobConfig()
    for k, v in data.items():
        value = coerce_field(k, v)
        if value is not None:
            setattr(cfg, k, value)
    cfg.vis_pixels = _ensure_odd(int(cfg.vis_pixels))
    cfg.saturation_mask_prob = min(
        max(float(cfg.saturation_mask_prob), 0.0),
        Config.TRAIN_SATURATION_MASK_PROB_MAX,
    )
    return cfg


def save(cfg: JobConfig) -> None:
    cfg.vis_pixels = _ensure_odd(int(cfg.vis_pixels))
    cfg.saturation_mask_prob = min(
        max(float(cfg.saturation_mask_prob), 0.0),
        Config.TRAIN_SATURATION_MASK_PROB_MAX,
    )
    os.makedirs(CONFIG_DIR, exist_ok=True)
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w") as fp:
        json.dump(cfg.to_dict(), fp, indent=2, sort_keys=True)
    os.replace(tmp, CONFIG_PATH)
    with contextlib.suppress(OSError):
        os.chmod(CONFIG_PATH, 0o600)


def version_of(values: dict[str, Any]) -> str:
    """Stable content hash of an effective config dict (``/api/config``'s
    ``version``): two loads of the same persisted values hash the same."""
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"),
                         default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def update(patch: dict[str, Any]) -> JobConfig:
    """Merge ``patch`` into the on-disk config and return the new state.

    Values are coerced to each field's declared type
    (:func:`coerce_field`); blanks and unconvertible values are ignored so a
    partial form never wipes a value. VIS cutout is forced odd.
    """
    cfg = load()
    for k, v in patch.items():
        if v is None or v == "":
            continue
        value = coerce_field(k, v)
        if value is not None:
            setattr(cfg, k, value)
    save(cfg)
    return cfg
