"""Per-member test PSNR: checkpoint fingerprinting + cache-skip semantics.

The members table shows each member's asinh-space test PSNR with a rank; a
member is only re-evaluated when its checkpoint fingerprint changes — the
whole point is NOT paying inference for unchanged members.
"""
from __future__ import annotations

import os

import numpy as np


def _member(base, name, ckpt="ckpt-5", payload=b"weights"):
    d = os.path.join(base, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "checkpoint"), "w") as f:
        f.write(f'model_checkpoint_path: "{ckpt}"\n'
                f'all_model_checkpoint_paths: "{ckpt}"\n')
    with open(os.path.join(d, f"{ckpt}.index"), "wb") as f:
        f.write(b"idx")
    with open(os.path.join(d, f"{ckpt}.data-00000-of-00001"), "wb") as f:
        f.write(payload)
    return d


class _Cap:
    def tick(self, *a, **k):
        pass


def _setup(tmp_path, monkeypatch):
    from euclid_polish.config import Config
    from euclid_polish.web.helpers import ensemble_viz as ev
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR",
                        str(tmp_path / "ckpt/wdsr"))
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    monkeypatch.setattr(ev, "_sky_records_local_dir",
                        lambda: str(tmp_path / "records"))
    monkeypatch.setattr(ev, "eval_subset", lambda r: "test")
    return ev


# --------------------------------------------------------------------------- #
# member_fingerprint
# --------------------------------------------------------------------------- #

def test_fingerprint_identifies_the_served_checkpoint(tmp_path):
    from euclid_polish.ensemble import member_fingerprint
    d = _member(str(tmp_path), "member_00")
    fp = member_fingerprint(d)
    assert fp is not None and fp.startswith("ckpt-5:")

    # More training → the manifest points at a NEW ckpt name → new fingerprint.
    _member(str(tmp_path), "member_00", ckpt="ckpt-9")
    assert member_fingerprint(d) != fp

    # No checkpoint at all → None (member simply shows "—").
    assert member_fingerprint(str(tmp_path / "nope")) is None


def test_fingerprint_changes_when_weights_change_in_place(tmp_path):
    from euclid_polish.ensemble import member_fingerprint
    d = _member(str(tmp_path), "member_00", payload=b"v1")
    fp1 = member_fingerprint(d)
    _member(str(tmp_path), "member_00", payload=b"v2-longer")   # same ckpt name
    assert member_fingerprint(d) != fp1


# --------------------------------------------------------------------------- #
# job_member_psnr — the cache-skip contract
# --------------------------------------------------------------------------- #

def test_job_scores_once_then_skips_unchanged_members(tmp_path, monkeypatch):
    ev = _setup(tmp_path, monkeypatch)
    base = ev.ensemble_dir()
    _member(base, "member_00")
    _member(base, "member_01")

    calls: list[str] = []

    def fake_eval(mdir, rdir, *, subset, num_images, on_progress=None):
        calls.append(os.path.basename(mdir))
        return {"subset": subset, "n_scored": num_images,
                "psnr_stretched": 43.5 if mdir.endswith("00") else 42.0}

    monkeypatch.setattr(ev, "evaluate_member_on_records", fake_eval)

    out = ev.job_member_psnr(_Cap())
    assert sorted(calls) == ["member_00", "member_01"]
    assert out["evaluated"] == ["member_00", "member_01"]

    # Nothing changed → the second run must not evaluate ANYTHING.
    calls.clear()
    out = ev.job_member_psnr(_Cap())
    assert calls == []
    assert sorted(out["reused"]) == ["member_00", "member_01"]

    # One member trains further (new ckpt) → only IT is re-scored.
    _member(base, "member_01", ckpt="ckpt-7")
    calls.clear()
    out = ev.job_member_psnr(_Cap())
    assert calls == ["member_01"]
    assert out["reused"] == ["member_00"]


# --------------------------------------------------------------------------- #
# ensemble_status — PSNR + rank from the cache
# --------------------------------------------------------------------------- #

def test_status_shows_cached_psnr_with_rank(tmp_path, monkeypatch):
    from euclid_polish.ensemble import member_fingerprint
    ev = _setup(tmp_path, monkeypatch)
    base = ev.ensemble_dir()
    d0 = _member(base, "member_00")
    d1 = _member(base, "member_01")
    _member(base, "member_02")            # never scored → unranked

    ev.update_member_psnr_cache({
        "member_00": {"fingerprint": member_fingerprint(d0),
                      "psnr": 42.0, "n_scored": 100},
        "member_01": {"fingerprint": member_fingerprint(d1),
                      "psnr": 43.5, "n_scored": 100},
    }, "test")

    st = ev.ensemble_status()
    by = {m["name"]: m for m in st["members"]}
    assert by["member_01"]["psnr"] == 43.5 and by["member_01"]["psnr_rank"] == 1
    assert by["member_00"]["psnr"] == 42.0 and by["member_00"]["psnr_rank"] == 2
    assert by["member_02"]["psnr"] is None and by["member_02"]["psnr_rank"] is None


def test_status_hides_psnr_after_checkpoint_change(tmp_path, monkeypatch):
    """A member whose checkpoint changed since its score must read "—" (stale
    numbers describe a different model), not show the old value."""
    from euclid_polish.ensemble import member_fingerprint
    ev = _setup(tmp_path, monkeypatch)
    base = ev.ensemble_dir()
    d0 = _member(base, "member_00")
    ev.update_member_psnr_cache({
        "member_00": {"fingerprint": member_fingerprint(d0),
                      "psnr": 42.0, "n_scored": 100},
    }, "test")
    _member(base, "member_00", ckpt="ckpt-9")     # trained further

    st = ev.ensemble_status()
    (m,) = st["members"]
    assert m["psnr"] is None and m["psnr_rank"] is None


# --------------------------------------------------------------------------- #
# training_curves_payload — registry-filtered + enriched for coloring modes
# --------------------------------------------------------------------------- #

def test_curves_exclude_tombstoned_members(tmp_path, monkeypatch):
    """An archived member's dir (with its training_log.csv) can linger on disk
    or come back from a FASRC leftover — the curves must not show it.
    Regression: member_09 was archived but still plotted."""
    from euclid_polish import ensemble_registry
    ev = _setup(tmp_path, monkeypatch)
    base = ev.ensemble_dir()
    log = ("step,wall_time,loss,psnr_stretched,psnr_raw,"
           "save_best_score,combined_loss,is_baseline\n"
           "1000,1,0.1,40,33,40,0.01,\n")
    for name in ("member_00", "member_09"):
        d = _member(base, name)
        with open(os.path.join(d, "training_log.csv"), "w") as f:
            f.write(log)
    ensemble_registry.load_registry(base)          # bootstrap both as active
    ensemble_registry.archive_member_entry(base, "member_09",
                                           zip_path="models/m09.zip",
                                           commit="abc")

    payload = ev.training_curves_payload()
    assert [s["name"] for s in payload] == ["member_00"]
    # Enrichment for the coloring modes is present (None on fake ckpts is fine).
    assert "blocks" in payload[0] and "test_psnr" in payload[0]


# --------------------------------------------------------------------------- #
# EnsembleModel.evaluate — stretched per-member PSNR (feeds the cache for free)
# --------------------------------------------------------------------------- #

def test_evaluate_reports_per_member_stretched_psnr():
    from euclid_polish.ensemble import EnsembleModel
    from euclid_polish.image import Image

    class _Stub:
        def __init__(self, fn):
            self._fn = fn
            self.id = None

        def upsample_array(self, arr):
            return np.asarray(self._fn(arr), np.float32)

    rng = np.random.default_rng(0)
    hr_arr = (rng.random((8, 8, 1)) * 100.0).astype(np.float32)
    img = Image(data=hr_arr, pixel_scale_arcsec=0.05, band_names=("VIS",),
                is_clean=False, index=0)
    # Member 0 is closer to HR than member 1 → higher PSNR in BOTH spaces.
    ens = EnsembleModel("x", _models=[
        _Stub(lambda a: hr_arr + rng.normal(0, 1.0, hr_arr.shape).astype(np.float32)),
        _Stub(lambda a: hr_arr + rng.normal(0, 8.0, hr_arr.shape).astype(np.float32)),
    ])
    out = ens.evaluate([img], [img])
    ps = out["per_member_psnr_stretched"]
    assert len(ps) == 2 and all(np.isfinite(p) for p in ps)
    assert ps[0] > ps[1]
    assert out["per_member_psnr"][0] > out["per_member_psnr"][1]


def test_regenerated_eval_records_invalidate_cached_psnr(tmp_path, monkeypatch):
    """Same subset, same field count, same checkpoint — but the eval RECORDS
    were regenerated → the cached PSNR is a score against a different dataset
    and must not be served (the 2026-07 zeropoint regen made every old score
    silently stale)."""
    from euclid_polish.ensemble import member_fingerprint
    from euclid_polish.web.helpers import ensemble_viz as ev

    mdir = _member(str(tmp_path), "member_00")
    fp = member_fingerprint(mdir)
    cache = {"subset": "test", "num_images": ev.MEMBER_PSNR_FIELDS,
             "records_fp": "dirty:100:1|hr:100:1",
             "members": {"member_00": {"fingerprint": fp, "psnr": 50.0,
                                       "n_scored": 100}}}
    same = ev._member_psnr_entry(cache, "member_00", mdir, "test",
                                 records_fp="dirty:100:1|hr:100:1")
    assert same is not None and same["psnr"] == 50.0
    # regenerated records → different fingerprint → cache miss
    assert ev._member_psnr_entry(cache, "member_00", mdir, "test",
                                 records_fp="dirty:999:2|hr:999:2") is None
    # writing scores under the new fingerprint resets the cache wholesale
    monkeypatch.setattr(ev.Config, "VIS_DIR", str(tmp_path / "vis"))
    ev.update_member_psnr_cache(
        {"member_00": {"fingerprint": fp, "psnr": 44.0, "n_scored": 100}},
        "test", records_fp="dirty:999:2|hr:999:2")
    fresh = ev._load_member_psnr_cache()
    assert fresh["records_fp"] == "dirty:999:2|hr:999:2"
    assert fresh["members"]["member_00"]["psnr"] == 44.0


def test_eval_records_fingerprint_tracks_file_identity(tmp_path):
    from euclid_polish.image.tfio import tfrecord_path
    from euclid_polish.web.helpers import ensemble_viz as ev

    rdir = str(tmp_path)
    assert ev._eval_records_fingerprint(rdir, "test") is None   # files absent
    for kind in ("dirty", "hr"):
        with open(tfrecord_path(rdir, f"{kind}_test"), "wb") as f:
            f.write(b"x" * 10)
    fp1 = ev._eval_records_fingerprint(rdir, "test")
    assert fp1 and ev._eval_records_fingerprint(rdir, "test") == fp1
    with open(tfrecord_path(rdir, "hr_test"), "wb") as f:       # regen
        f.write(b"y" * 20)
    assert ev._eval_records_fingerprint(rdir, "test") != fp1
    assert ev._eval_records_fingerprint(None, "test") is None


def test_training_curves_payload_carries_loss_norm(tmp_path, monkeypatch):
    """The curves chart's "by loss" mode reads each member's reconstruction
    norm from origin.json; members predating the knob default to L1."""
    import json as _json

    from euclid_polish.web.helpers import ensemble_viz as ev

    base = str(tmp_path / "ens")
    d_new = _member(base, "member_00")
    with open(os.path.join(d_new, "origin.json"), "w") as f:
        _json.dump({"op": "add", "loss_norm": "l2"}, f)
    d_old = _member(base, "member_01")          # no origin.json → legacy L1

    monkeypatch.setattr(ev, "ensemble_dir", lambda: base)
    monkeypatch.setattr(
        "euclid_polish.ensemble_registry.active_member_dirs",
        lambda b: [d_new, d_old])
    monkeypatch.setattr(ev, "_sky_records_local_dir", lambda: None)
    monkeypatch.setattr(
        "euclid_polish.training.log_plot.ensemble_training_series",
        lambda b: [{"name": "member_00"}, {"name": "member_01"}])
    out = ev.training_curves_payload()
    assert {s["name"]: s["loss"] for s in out} == {
        "member_00": "l2", "member_01": "l1"}
