"""An ensemble of WDSR models trained on the same data with different seeds.

Each member is an ordinary :class:`~euclid_polish.model.Model` checkpoint living
in ``<base_dir>/member_NN/``, trained on the *same* train/validate data but with
a different RNG seed (different weight init + data-shuffle order → genuinely
different solutions). Every member's seed is recorded on its
``Process.training`` provenance, so the ensemble's diversity is traceable.

Why an ensemble — hallucination cross-check
-------------------------------------------
Super-resolution is ill-posed: many high-res images are consistent with one LR
image, so a single model must *invent* detail below the LR resolution. The
**ensemble mean** is the prediction; the **per-pixel spread across members** is
the epistemic disagreement:

* where members agree (low spread) the SR is constrained by the LR data and is
  trustworthy;
* where members disagree (high spread) each invented something different — that
  is a hallucination, not a measurement.

So the disagreement map localises which SR structure to believe, and aggregate
disagreement quantifies how much a run is hallucinating.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from euclid_polish import ensemble_registry
from euclid_polish.config import Config
from euclid_polish.ensemble_registry import default_ensemble_dir  # re-export
from euclid_polish.image import Image, ImageSet, Role
from euclid_polish.model import Model, _checkpoint_exists
from euclid_polish.provenance.gitinfo import capture_git

#: Per-member checkpoint sub-directory name, e.g. ``member_03``.
MEMBER_DIR_FMT = "member_{:02d}"
_MEMBER_GLOB = "member_*"


def member_dir(base_dir: str, i: int) -> str:
    """Checkpoint directory for ensemble member ``i`` under ``base_dir``."""
    return os.path.join(base_dir, MEMBER_DIR_FMT.format(int(i)))


def member_fingerprint(mdir: str) -> str | None:
    """Cheap identity of a member's PSNR-best checkpoint (no TF, no loading).

    ``<ckpt-name>:<bytes>:<index-mtime>`` of the manifest's current checkpoint.
    Any further training saves a new ``ckpt-N`` (new name); a re-pull of
    identical files keeps name/size/mtime (rsync preserves times), so the
    fingerprint changes iff the served weights change. ``None`` when the member
    has no checkpoint yet. Used to cache per-member eval results — a member is
    only re-scored when this changes.
    """
    manifest = os.path.join(mdir, "checkpoint")
    if not os.path.isfile(manifest):
        return None
    name = None
    with open(manifest) as f:
        for line in f:
            m = re.match(r'model_checkpoint_path:\s*"(.+)"', line.strip())
            if m:
                name = os.path.basename(m.group(1))
                break
    if not name:
        return None
    index = os.path.join(mdir, f"{name}.index")
    if not os.path.isfile(index):
        return None
    total = 0
    for fn in os.listdir(mdir):
        if fn == f"{name}.index" or re.fullmatch(
                re.escape(name) + r"\.data-\d+-of-\d+", fn):
            total += os.path.getsize(os.path.join(mdir, fn))
    return f"{name}:{total}:{int(os.path.getmtime(index))}"


@dataclass
class MemberTrainSpec:
    """One member's training job within a run.

    ``target_steps`` is the ABSOLUTE step target (the trainer's ``steps``);
    ``run_steps`` is how many steps this run actually executes (=
    ``target_steps`` for add/fork, the extra for continue) — used only for
    cumulative progress accounting. ``init_from`` is a checkpoint dir to copy
    weights from (fork). ``op`` ∈ {"add", "continue", "fork"}.
    ``num_res_blocks`` sets a NEW member's trunk depth (``None`` → the run
    default); members of different depths coexist — continue/fork ignore it
    because the existing checkpoint / fork source dictates the depth.

    Diversity knobs (all optional, per member — the point of an ensemble is
    that members differ): ``loss_norm`` ∈ {"l1","l2","l3"} picks the
    reconstruction p-norm; ``noise_aug`` adds extra LR noise (read-noise
    units); ``bootstrap`` ∈ (0, 1) trains on that deterministic fraction of
    the fields (keyed by the member's seed); ``forward_onthefly`` re-draws
    PSF + noise live on the full field each visit (``psf_subset`` bags the
    member's pre-rotated PSF pool, ``crops_per_field`` amortises one field
    forward over K crops). They apply to every op — continue/fork runs can
    adopt them too (they shape the *run*, not the architecture).
    """
    name: str
    seed: int
    target_steps: int
    op: str = "add"
    run_steps: int = 0
    init_from: str | None = None
    forked_from: str | None = None
    num_res_blocks: int | None = None
    loss_norm: str = "l1"
    noise_aug: float = 0.0
    bootstrap: float | None = None
    forward_onthefly: bool = False
    psf_subset: int | None = None
    crops_per_field: int = 4


def pca_field(members: np.ndarray, n_components: int = 3
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PCA of the per-member residuals about the ensemble mean.

    ``members`` is the ``(M, H, W, C)`` stack of member SR cubes. Returns
    ``(mean, components, amplitudes, var_explained)``:

    * ``mean``          — ``(H, W, C)`` ensemble mean,
    * ``components``    — ``(K, H, W, C)`` unit-norm eigen-images, K = min(
      ``n_components``, M-1),
    * ``amplitudes``    — ``(K,)`` population std of the member projections
      along each component (how far the members actually spread that way),
    * ``var_explained`` — ``(K,)`` fraction of the TOTAL residual variance
      each kept component carries (``sᵢ²/Σs²``); their sum is how much of the
      ensemble's real disagreement the movie shows.

    A smooth animation ``mean + Σ aᵢ·sin(2π fᵢ t)·componentᵢ`` then morphs
    through the ensemble's *real* (spatially-correlated) disagreement subspace —
    unlike per-pixel Gaussian draws, which would be spatial white noise.
    """
    mem = np.asarray(members, dtype=np.float32)
    m = int(mem.shape[0])
    mean = mem.mean(axis=0)
    k = int(min(n_components, max(0, m - 1)))
    if k == 0:
        return (mean, np.zeros((0, *mean.shape), np.float32),
                np.zeros((0,), np.float32), np.zeros((0,), np.float32))
    resid = (mem - mean).reshape(m, -1)                 # (M, D)
    # SVD: resid = U·diag(S)·Vt; Vt rows are unit eigen-images (principal dirs).
    _u, s, vt = np.linalg.svd(resid, full_matrices=False)
    comps = vt[:k].reshape((k, *mean.shape)).astype(np.float32)
    amps = (s[:k] / np.sqrt(m)).astype(np.float32)      # population std along comp
    s2 = s.astype(np.float64) ** 2
    total = float(s2.sum())
    var_explained = ((s2[:k] / total) if total > 0
                     else np.zeros(k)).astype(np.float32)
    return mean, comps, amps, var_explained


def _psnr(a: np.ndarray, b: np.ndarray, peak: float) -> float:
    """PSNR (dB) over the overlapping region; raw electrons, peak ``peak``."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a, b = np.asarray(a)[:h, :w], np.asarray(b)[:h, :w]
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(peak * peak / mse))


class EnsembleModel:
    """A seed-diverse ensemble of WDSR checkpoints under one base directory.

    Parameters
    ----------
    base_dir : str
        Directory holding ``member_NN/`` checkpoint sub-directories.
    scale, num_res_blocks : int
        Architecture of every member (shared).
    n_members : int, optional
        Cap on how many member *directories* to load (for eval); ``None`` loads
        all found.
    include_loss_best : bool, optional
        Also load each member's ``loss_best/`` checkpoint as a separate model
        (tagged ``NN·loss``), roughly doubling the ensemble. Default ``False``:
        the PSNR-best and LOSS-best checkpoints of one seed share weight
        history, so the extra model is a correlated vote that mostly makes
        evaluation ~2× slower now that training converges well. The
        ``loss_best/`` track is still SAVED during training and can seed
        forks — it just isn't part of the ensemble unless opted in.
    """

    def __init__(
        self,
        base_dir: str,
        *,
        scale: int = Config.DEFAULT_REBIN_FACTOR,
        num_res_blocks: int = Config.DEFAULT_NUM_RES_BLOCKS,
        n_members: int | None = None,
        include_loss_best: bool = False,
        _models: Sequence[Model] | None = None,
    ) -> None:
        self.base_dir = base_dir
        self._scale = int(scale)
        self._num_res_blocks = int(num_res_blocks)
        if _models is not None:                      # test / explicit injection
            self._models: list[Model] = list(_models)
            self._member_labels: list[str] = [
                f"m{i:02d}" for i in range(len(self._models))
            ]
            return
        # Registry-active members only — an archived member's directory may
        # still sit on disk (e.g. mirrored back from FASRC) but never loads.
        dirs = [d for d in ensemble_registry.active_member_dirs(base_dir)
                if os.path.isdir(d) and _checkpoint_exists(d)]
        if n_members is not None:
            dirs = dirs[: int(n_members)]
        models: list[Model] = []
        labels: list[str] = []
        for d in dirs:
            idx = os.path.basename(d).removeprefix("member_")
            models.append(
                Model(d, scale=self._scale, num_res_blocks=self._num_res_blocks))
            labels.append(f"{idx}·psnr")
            if include_loss_best:
                lb = os.path.join(d, "loss_best")
                if os.path.isdir(lb) and _checkpoint_exists(lb):
                    models.append(Model(
                        lb, scale=self._scale,
                        num_res_blocks=self._num_res_blocks))
                    labels.append(f"{idx}·loss")
        self._models = models
        self._member_labels = labels

    # -- introspection -- #

    @property
    def n_members(self) -> int:
        return len(self._models)

    @property
    def members(self) -> list[Model]:
        return list(self._models)

    @property
    def member_ids(self) -> list[str | None]:
        """Each member's model :class:`ProvId` (or ``None`` for a legacy ckpt)."""
        return [str(m.id) if m.id is not None else None for m in self._models]

    @property
    def member_labels(self) -> list[str]:
        """Per-model tag aligned with :attr:`members` — e.g. ``00·psnr`` (the
        best-PSNR checkpoint) and ``00·loss`` (the best-LOSS checkpoint of the
        same seed)."""
        return list(self._member_labels)

    def _require_members(self) -> None:
        if not self._models:
            raise RuntimeError(
                f"no ensemble members loaded from {self.base_dir!r} "
                f"({_MEMBER_GLOB}/); train the ensemble or check the path.")

    # -- training -- #

    def train(
        self,
        lr_path: str,
        hr_path: str,
        *,
        n_members: int,
        base_seed: int | None = None,
        steps: int = 300_000,
        batch_size: int = 16,
        on_member: Callable[[int, int, Model], None] | None = None,
        **train_kwargs,
    ) -> EnsembleModel:
        """Sequentially train ``n_members`` on the same data with distinct seeds.

        Member ``i`` is trained in ``<base_dir>/member_i/`` seeded ``base_seed +
        i`` (a fresh entropy ``base_seed`` is drawn when ``None``); the seed is
        recorded on that member's ``Process.training`` provenance. Extra
        ``train_kwargs`` (e.g. ``evaluate_every``) pass straight to
        :meth:`Model.train`. ``on_member(i, n, model)`` is called after each
        member finishes (progress hook). Returns ``self``.
        """
        if n_members < 1:
            raise ValueError(f"n_members must be >= 1, got {n_members}")
        if base_seed is None:
            base_seed = int.from_bytes(os.urandom(4), "little")
        specs = [MemberTrainSpec(
                     name=MEMBER_DIR_FMT.format(i), seed=int(base_seed) + i,
                     target_steps=int(steps), op="add", run_steps=int(steps))
                 for i in range(int(n_members))]
        return self.train_members(lr_path, hr_path, specs,
                                  batch_size=batch_size,
                                  on_member=on_member, **train_kwargs)

    def train_members(
        self,
        lr_path: str,
        hr_path: str,
        specs: Sequence[MemberTrainSpec],
        *,
        batch_size: int = 16,
        on_member: Callable[[int, int, Model], None] | None = None,
        **train_kwargs,
    ) -> EnsembleModel:
        """Run an explicit list of member training jobs sequentially.

        Unlike the legacy :meth:`train` (which blindly loops member_00..N-1),
        each spec names its member, seed, absolute step target and optional
        fork source. Members CREATED here (op add/fork) get an
        ``origin.json`` provenance sidecar that syncs down with the member.
        """
        if not specs:
            raise ValueError("no member specs to train")
        self._models = []
        for i, spec in enumerate(specs):
            d = os.path.join(self.base_dir, spec.name)
            created = not (os.path.isdir(d) and _checkpoint_exists(d))
            os.makedirs(d, exist_ok=True)
            m = Model(d, scale=self._scale,
                      num_res_blocks=(spec.num_res_blocks
                                      or self._num_res_blocks),
                      seed=int(spec.seed), init_weights_from=spec.init_from)
            if created and spec.op in ("add", "fork"):
                commit = (capture_git() or {}).get("short")
                with open(os.path.join(d, "origin.json"), "w") as f:
                    json.dump({
                        "op": spec.op,
                        "forked_from": spec.forked_from,
                        "seed": int(spec.seed),
                        "target_steps": int(spec.target_steps),
                        # The ACTUAL depth (a fork inherits its source's).
                        "num_res_blocks": int(m._num_res_blocks),
                        # Diversity knobs, recorded so the member's training
                        # distribution is reconstructible from its sidecar.
                        "loss_norm": spec.loss_norm,
                        "noise_aug": float(spec.noise_aug),
                        "bootstrap": spec.bootstrap,
                        "forward_onthefly": bool(spec.forward_onthefly),
                        "psf_subset": spec.psf_subset,
                        "crops_per_field": int(spec.crops_per_field),
                        "created_at": datetime.now(UTC).isoformat(
                            timespec="seconds"),
                        "commit": commit,
                    }, f, indent=2)
            # Continue resumes from the PSNR-best track — the model eval
            # actually uses. Max-step resume would pick loss_best/ when that
            # track ran ahead during a degenerate (skip-only) stretch.
            m.train(lr_path, hr_path, steps=int(spec.target_steps),
                    batch_size=batch_size,
                    resume_track=("psnr" if spec.op == "continue"
                                  else "latest"),
                    loss_norm=spec.loss_norm,
                    noise_aug=spec.noise_aug,
                    bootstrap=spec.bootstrap,
                    forward_onthefly=spec.forward_onthefly,
                    psf_subset=spec.psf_subset,
                    crops_per_field=spec.crops_per_field,
                    **train_kwargs)
            self._models.append(m)
            if on_member is not None:
                on_member(i + 1, len(specs), m)
        return self

    # -- prediction + disagreement -- #

    def member_arrays(self, lr_array: np.ndarray) -> np.ndarray:
        """``(M, H, W, C)`` stack of each member's SR array (raw electrons)."""
        self._require_members()
        return np.stack([m.upsample_array(lr_array) for m in self._models],
                        axis=0)

    def predict(self, lr_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(mean, std)``: the ensemble-mean SR and the per-pixel disagreement
        (population std across members). ``std`` is the hallucination signal —
        high where members invent different detail."""
        preds = self.member_arrays(lr_array)
        return preds.mean(axis=0), preds.std(axis=0)

    def disagreement(self, lr_array: np.ndarray) -> np.ndarray:
        """Per-pixel std across members (the raw hallucination map)."""
        return self.member_arrays(lr_array).std(axis=0)

    def upsample(self, lr: Image, *, store=None) -> Image:
        """Super-resolve ``lr`` to the ensemble-mean SR :class:`Image` (role ``sr``).

        The per-pixel disagreement is returned alongside by
        :meth:`predict_images`; this returns just the mean prediction so it drops
        into the same call sites as :meth:`Model.upsample`.
        """
        mean, _std = self.predict(lr.data)
        bands = (lr.band_names if mean.ndim == 3
                 and mean.shape[-1] == len(lr.band_names) else ("VIS",))
        return Image(data=mean.astype(np.float32),
                     pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                     band_names=bands, is_clean=True, role=Role.SR,
                     subset=lr.subset)

    def upsample_batch(self, lr_images, *, on_progress=None, log=None):
        """Ensemble-mean SR :class:`Image` for every LR image, in order.

        Drop-in for :meth:`Model.upsample_batch` at the call sites that used
        to run the single model (e.g. the sky-SR job)."""
        self._require_members()
        lr_list = list(lr_images)
        out = []
        for i, lr in enumerate(lr_list):
            out.append(self.upsample(lr))
            if on_progress is not None:
                on_progress(i + 1, len(lr_list), f"field {lr.index}")
            if log is not None:
                log(f"field {lr.index}: ensemble mean of {self.n_members}\n")
        return out

    def predict_images(self, lr: Image) -> tuple[Image, Image]:
        """``(mean_sr, disagreement)`` as :class:`Image` on the HR grid."""
        mean, std = self.predict(lr.data)
        bands = (lr.band_names if mean.ndim == 3
                 and mean.shape[-1] == len(lr.band_names) else ("VIS",))
        common = {"pixel_scale_arcsec": Config.DEFAULT_PIXEL_SCALE,
                  "band_names": bands, "subset": lr.subset}
        mean_img = Image(data=mean.astype(np.float32), is_clean=True,
                         role=Role.SR, **common)
        std_img = Image(data=std.astype(np.float32), is_clean=True,
                        role=Role.UNKNOWN, **common)
        return mean_img, std_img

    # -- evaluation -- #

    def evaluate(
        self,
        lr_images: Iterable[Image],
        hr_images: Iterable[Image] | None = None,
        *,
        rel_floor_e: float = Config.STRETCH_SCALE_E,
        hallucination_rel_tol: float = 0.5,
        on_field: Callable[
            [int, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
             np.ndarray | None], None
        ] | None = None,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        """Evaluate the ensemble on (test) fields.

        For each LR field: each member's SR, the ensemble mean, and the per-pixel
        disagreement. When matched HR images are given (by record index), PSNR is
        computed for every member and for the ensemble mean.

        Returns a summary dict::

            {
              "n_members", "n_fields",
              "per_member_psnr": [...],          # mean over fields, raw e⁻
              "per_member_psnr_stretched": [...],# same, asinh space (the
                                                 #   training psnr_stretched)
              "per_member_labels": [...],        # NN·psnr / NN·loss, aligned
              "mean_member_psnr", "ensemble_psnr",
              "ensemble_gain_db",                # ensemble − best member
              "disagreement": {
                 "mean_std_e",                   # mean per-pixel std (electrons)
                 "mean_rel_disagreement",        # std / (|mean| + floor)
                 "frac_flux_hallucinated",       # SR flux fraction where
                                                 #   rel-disagreement > tol
              },
            }

        ``hallucination_rel_tol`` is the relative-disagreement threshold above
        which a pixel counts as hallucinated; ``rel_floor_e`` keeps the ratio
        finite in the sky floor.
        """
        self._require_members()
        hr_by = {h.index: h for h in (hr_images or [])}
        peak = float(Config.PSNR_PEAK_E)
        knee = float(Config.STRETCH_SCALE_E)
        peak_str = float(Config.PSNR_PEAK_STRETCHED)

        per_member_sum = np.zeros(self.n_members, dtype=np.float64)
        per_member_str_sum = np.zeros(self.n_members, dtype=np.float64)
        ens_sum = 0.0
        n_scored = 0                       # fields with a matched HR (for PSNR)
        n_fields = 0
        std_e_sum = 0.0
        rel_sum = 0.0
        hall_flux = 0.0
        total_flux = 0.0

        lr_list = list(lr_images)
        total = len(lr_list)
        for i, lr in enumerate(lr_list):
            preds = self.member_arrays(lr.data)            # (M,H,W,C)
            mean = preds.mean(axis=0)
            std = preds.std(axis=0)
            n_fields += 1

            # Disagreement (no HR needed) — the hallucination cross-check.
            std_e_sum += float(std.mean())
            denom = np.abs(mean) + float(rel_floor_e)
            rel = std / denom
            rel_sum += float(rel.mean())
            amean = np.abs(mean)
            total_flux += float(amean.sum())
            hall_flux += float(amean[rel > hallucination_rel_tol].sum())

            # Accuracy vs HR (when present).
            hr = hr_by.get(lr.index)
            hr_data = np.asarray(hr.data, np.float32) if hr is not None else None
            if hr_data is not None:
                hr_str = np.arcsinh(hr_data / knee)
                for k in range(self.n_members):
                    per_member_sum[k] += _psnr(preds[k], hr_data, peak)
                    # asinh space — the training metric (psnr_stretched), so
                    # the per-member number is comparable to the curves.
                    per_member_str_sum[k] += _psnr(
                        np.arcsinh(preds[k] / knee), hr_str, peak_str)
                ens_sum += _psnr(mean, hr_data, peak)
                n_scored += 1

            # Per-field hook: hand back LR, the full member stack, ensemble mean
            # SR, per-pixel std and HR so a caller can persist cubes + the PCA
            # disagreement basis for the client-side viewer/animation.
            if on_field is not None:
                on_field(int(lr.index), np.asarray(lr.data, np.float32),
                         preds, mean, std, hr_data)

            if on_progress is not None:
                on_progress(i + 1, total, f"field {lr.index}")

        per_member = (per_member_sum / n_scored).tolist() if n_scored else []
        per_member_str = ((per_member_str_sum / n_scored).tolist()
                          if n_scored else [])
        ensemble_psnr = (ens_sum / n_scored) if n_scored else float("nan")
        mean_member = float(np.mean(per_member)) if per_member else float("nan")
        best_member = float(np.max(per_member)) if per_member else float("nan")
        return {
            "n_members": self.n_members,
            "n_fields": n_fields,
            "n_scored": n_scored,
            "per_member_psnr": per_member,
            "per_member_psnr_stretched": per_member_str,
            "per_member_labels": list(self._member_labels),
            "mean_member_psnr": mean_member,
            "ensemble_psnr": ensemble_psnr,
            "ensemble_gain_db": (ensemble_psnr - best_member
                                 if n_scored else float("nan")),
            "disagreement": {
                "mean_std_e": (std_e_sum / n_fields) if n_fields else float("nan"),
                "mean_rel_disagreement": (rel_sum / n_fields) if n_fields
                else float("nan"),
                "frac_flux_hallucinated": (hall_flux / total_flux)
                if total_flux > 0 else float("nan"),
            },
        }


def ensemble_available(base_dir: str | None = None) -> bool:
    """Cheap check (no model loading) for whether a usable ensemble exists —
    at least one registry-ACTIVE member with a checkpoint. Mirrors the member
    discovery in :class:`EnsembleModel` without constructing any keras models,
    so callers can decide *whether* to run the ensemble before paying to load
    it. Archived members (dir possibly still on disk) do not count."""
    base_dir = base_dir or default_ensemble_dir()
    return any(
        os.path.isdir(d) and _checkpoint_exists(d)
        for d in ensemble_registry.active_member_dirs(base_dir))


def load_ensemble(
    base_dir: str | None = None,
    *,
    scale: int = Config.DEFAULT_REBIN_FACTOR,
    num_res_blocks: int = Config.DEFAULT_NUM_RES_BLOCKS,
    n_members: int | None = None,
    include_loss_best: bool = False,
) -> EnsembleModel:
    """Load the ensemble under ``base_dir`` (default: ``<ckpt>/ensemble``).

    ``include_loss_best`` (default ``False``) would add each member's
    ``loss_best/`` checkpoint as a second, correlated model — see
    :class:`EnsembleModel`. Evaluation uses PSNR-best only.
    """
    if base_dir is None:
        base_dir = default_ensemble_dir()
    return EnsembleModel(base_dir, scale=scale, num_res_blocks=num_res_blocks,
                         n_members=n_members, include_loss_best=include_loss_best)


def evaluate_on_records(
    base_dir: str,
    records_dir: str,
    *,
    subset: str | None = None,
    num_images: int = 200,
    scale: int = Config.DEFAULT_REBIN_FACTOR,
    num_res_blocks: int = Config.DEFAULT_NUM_RES_BLOCKS,
    include_loss_best: bool = False,
    on_field: Callable[
        [int, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
         np.ndarray | None], None
    ] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Convenience: evaluate the ensemble on a generated subset's TFRecords.

    Reads ``dirty_<subset>`` (LR) + ``hr_<subset>`` (HR target) from
    ``records_dir`` — defaulting ``subset`` to the held-out eval split — and runs
    :meth:`EnsembleModel.evaluate`. Scores each member's PSNR-best checkpoint
    only; ``include_loss_best=True`` opts the correlated ``loss_best/`` models
    back in — see :class:`EnsembleModel`.
    """
    from euclid_polish.eval.subsets import eval_subset
    from euclid_polish.image.tfio import tfrecord_path

    sub = subset or eval_subset(records_dir)
    ens = EnsembleModel(base_dir, scale=scale, num_res_blocks=num_res_blocks,
                        include_loss_best=include_loss_best)
    lr = ImageSet.read(tfrecord_path(records_dir, f"dirty_{sub}"),
                       num_images=num_images)
    hr = ImageSet.read(tfrecord_path(records_dir, f"hr_{sub}"),
                       num_images=num_images)
    out = ens.evaluate(list(lr), list(hr), on_field=on_field,
                       on_progress=on_progress)
    out["subset"] = sub
    out["member_labels"] = ens.member_labels     # aligned with member_arrays order
    return out


def evaluate_member_on_records(
    mdir: str,
    records_dir: str,
    *,
    subset: str | None = None,
    num_images: int = 100,
    scale: int = Config.DEFAULT_REBIN_FACTOR,
    num_res_blocks: int = Config.DEFAULT_NUM_RES_BLOCKS,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """ONE member's test-set PSNR in asinh space (the training metric).

    Loads just this member (depth self-corrects from the checkpoint) and scores
    it over the subset's first ``num_images`` LR/HR pairs — so a single changed
    member can be re-scored without paying for the whole ensemble. Returns
    ``{"subset", "n_scored", "psnr_stretched"}``.
    """
    from euclid_polish.eval.subsets import eval_subset
    from euclid_polish.image.tfio import tfrecord_path

    sub = subset or eval_subset(records_dir)
    m = Model(mdir, scale=scale, num_res_blocks=num_res_blocks)
    lr = list(ImageSet.read(tfrecord_path(records_dir, f"dirty_{sub}"),
                            num_images=num_images))
    hr_by = {h.index: h for h in
             ImageSet.read(tfrecord_path(records_dir, f"hr_{sub}"),
                           num_images=num_images)}
    knee = float(Config.STRETCH_SCALE_E)
    peak_str = float(Config.PSNR_PEAK_STRETCHED)
    vals: list[float] = []
    for i, l in enumerate(lr):
        hr = hr_by.get(l.index)
        if hr is None:
            continue
        sr = m.upsample_array(np.asarray(l.data, np.float32))
        vals.append(_psnr(np.arcsinh(sr / knee),
                          np.arcsinh(np.asarray(hr.data, np.float32) / knee),
                          peak_str))
        if on_progress is not None:
            on_progress(i + 1, len(lr), f"field {l.index}")
    return {"subset": sub, "n_scored": len(vals),
            "psnr_stretched": (float(np.mean(vals)) if vals else float("nan"))}
