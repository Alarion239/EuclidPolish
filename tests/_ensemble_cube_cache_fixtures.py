"""A tiny STARFULL ensemble for the member-cube cache tests: fake members with
real-looking checkpoint files (so :func:`euclid_polish.ensemble.member_fingerprint`
returns a fingerprint that changes when a member is "continued"), tiny test and
validate records, and a fake :class:`~euclid_polish.ensemble.EnsembleModel`
that logs which members were loaded and run. No TensorFlow inference."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from euclid_polish import ensemble_registry as er
from euclid_polish.config import Config
from euclid_polish.eval import spatial_gate_fit
from euclid_polish.image import Image, Role
from euclid_polish.image.tfio import tfrecord_path, write_images
from euclid_polish.web.helpers import ensemble_viz as ev

BANDS = ("VIS", "Y_E", "J_E", "H_E")
LR_SIDE = 16
N_FIELDS = 3


class Cap:
    def tick(self, *a, **k):
        pass

    def write(self, *a, **k):
        pass


def member_dir(base: Path, number: int) -> Path:
    return base / f"member_{number:02d}"


def set_checkpoint(base: Path, number: int, step: int) -> None:
    """Point a member at checkpoint ``ckpt-<step>`` (a "continued" member
    gets a new step, hence a new fingerprint)."""
    d = member_dir(base, number)
    (d / "checkpoint").write_text(f'model_checkpoint_path: "ckpt-{step}"\n')
    (d / f"ckpt-{step}.index").write_bytes(b"i" * step)
    (d / f"ckpt-{step}.data-00000-of-00001").write_bytes(b"d" * (10 + step))


def add_member(base: Path, number: int, step: int = 1) -> str:
    d = member_dir(base, number)
    d.mkdir(parents=True, exist_ok=True)
    (d / "origin.json").write_text('{"starless": false}')
    set_checkpoint(base, number, step)
    er.load_registry(str(base))                       # bootstrap it as active
    return f"{number:02d}·psnr"


def member_output(lr: np.ndarray, number: int, step: int) -> np.ndarray:
    """What fake member ``number`` at checkpoint ``step`` predicts."""
    up = np.kron(np.asarray(lr, np.float32), np.ones((2, 2, 1), np.float32)) / 4.0
    return (up + number + 100.0 * step).astype(np.float32)


def checkpoint_step(base: Path, number: int) -> int:
    text = (member_dir(base, number) / "checkpoint").read_text()
    return int(text.split("ckpt-")[1].split('"')[0])


class FakeEnsemble:
    """Stands in for ``EnsembleModel(base, starless=…, labels=…)``: logs the
    members each construction loads and every member run."""

    loaded: list[list[str]] = []
    runs: list[str] = []

    def __init__(self, base_dir, *, starless=None, labels=None, **_kw):
        self.base = Path(base_dir)
        self.labels = [str(v) for v in labels]
        FakeEnsemble.loaded.append(list(self.labels))
        self.steps = [checkpoint_step(self.base, int(lb.split("·")[0])) for lb in self.labels]

    @property
    def member_labels(self):
        return list(self.labels)

    def member_arrays(self, lr, indices=None):
        out = []
        for i in (range(len(self.labels)) if indices is None else indices):
            label = self.labels[int(i)]
            FakeEnsemble.runs.append(label)
            out.append(member_output(lr, int(label.split("·")[0]), self.steps[int(i)]))
        return np.stack(out)

    @classmethod
    def reset(cls):
        cls.loaded = []
        cls.runs = []


def run_counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for label in FakeEnsemble.runs:
        out[label] = out.get(label, 0) + 1
    return out


def truth(rec: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng([seed, rec])
    return rng.exponential(200.0, (2 * LR_SIDE, 2 * LR_SIDE, 4)).astype(np.float32)


def write_records(records: Path, subset: str, seed: int = 0) -> None:
    hr = [Image(data=truth(r, seed), pixel_scale_arcsec=0.05, band_names=BANDS,
                is_clean=True, role=Role.HR, index=r) for r in range(N_FIELDS)]
    lr = [Image(data=truth(r, seed).reshape(LR_SIDE, 2, LR_SIDE, 2, 4).sum(axis=(1, 3)),
                pixel_scale_arcsec=0.1, band_names=BANDS, is_clean=False, role=Role.LR,
                index=r) for r in range(N_FIELDS)]
    records.mkdir(parents=True, exist_ok=True)
    write_images(hr, f"hr_{subset}", records_dir=str(records))
    write_images(lr, f"dirty_{subset}", records_dir=str(records))


def regenerate_records(records: Path, subset: str) -> None:
    """New records under the same names (a regenerated split): the records
    fingerprint changes."""
    write_records(records, subset, seed=7)
    for kind in ("dirty", "hr"):
        path = tfrecord_path(str(records), f"{kind}_{subset}")
        stat = os.stat(path)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000))


def make_env(tmp_path: Path, monkeypatch, members=(1, 2, 3)) -> dict:
    """Point Config + ensemble_viz at a fresh tiny ensemble with test and
    validate records; member inference goes through :class:`FakeEnsemble`."""
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR", str(tmp_path / "ckpt" / "wdsr"))
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    monkeypatch.setattr(Config, "TRACKING_DIR", str(tmp_path / "tracking"))
    monkeypatch.setattr(ev, "infer_checkpoint_num_res_blocks", lambda d: 32)
    monkeypatch.setattr(spatial_gate_fit, "EnsembleModel", FakeEnsemble)
    FakeEnsemble.reset()
    base = tmp_path / "ckpt" / "ensemble"
    labels = [add_member(base, n) for n in members]
    records = tmp_path / "records"
    write_records(records, "test")
    write_records(records, "validate", seed=3)
    monkeypatch.setattr(ev, "_sky_records_local_dir", lambda: str(records))
    regime = tmp_path / "vis" / "ensemble" / "starfull"
    return {"base": base, "records": records, "regime": regime,
            "cubes": regime / "cubes", "validate": regime / "cubes_validate",
            "labels": labels}


def lr_of(rec: int, seed: int = 0) -> np.ndarray:
    return truth(rec, seed).reshape(LR_SIDE, 2, LR_SIDE, 2, 4).sum(axis=(1, 3))
