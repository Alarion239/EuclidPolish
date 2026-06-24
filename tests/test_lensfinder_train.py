"""Resume logic in scripts/lensfinder_train.py — testable without torch/zoobot.

The module's top-level imports are light; the skip path returns before any
heavy import, and _latest_checkpoint / _parse_args are pure helpers.
"""

from __future__ import annotations

import importlib.util
import os


def _load():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "lensfinder_train.py")
    spec = importlib.util.spec_from_file_location("lf_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lt = _load()


class _Rep:
    def set_stage(self, *a, **k): pass
    def set_step(self, *a, **k): pass
    def metric(self, *a, **k): pass


class _Args:
    def __init__(self, **k):
        self.__dict__.update(k)


def test_parse_args_force_default():
    assert lt._parse_args(["--catalog", "c", "--out-dir", "o"]).force is False
    assert lt._parse_args(["--catalog", "c", "--out-dir", "o", "--force"]).force


def test_latest_checkpoint_prefers_last(tmp_path):
    out = tmp_path / "sr"
    ck = out / "checkpoints"
    ck.mkdir(parents=True)
    (ck / "8.ckpt").write_text("x")
    (ck / "9.ckpt").write_text("x")
    (ck / "last.ckpt").write_text("x")
    assert os.path.basename(lt._latest_checkpoint(str(out))) == "last.ckpt"


def test_latest_checkpoint_falls_back_to_newest(tmp_path):
    out = tmp_path / "sr"
    ck = out / "checkpoints"
    ck.mkdir(parents=True)
    old = ck / "8.ckpt"
    old.write_text("x")
    os.utime(old, (10**9, 10**9))                  # age 8.ckpt into the past
    (ck / "9.ckpt").write_text("x")                # 9.ckpt keeps its (now) mtime
    assert os.path.basename(lt._latest_checkpoint(str(out))) == "9.ckpt"


def test_latest_checkpoint_none_when_empty(tmp_path):
    assert lt._latest_checkpoint(str(tmp_path / "nope")) is None


def test_checkpoint_epoch_from_filename(tmp_path):
    assert lt._checkpoint_epoch(str(tmp_path / "13.ckpt")) == 13
    assert lt._checkpoint_epoch(str(tmp_path / "1-v1.ckpt")) == 1   # dedup suffix
    assert lt._checkpoint_epoch(None) is None


def test_train_one_skips_when_predictions_exist(tmp_path):
    out = tmp_path / "sr"
    out.mkdir(parents=True)
    (out / "predictions.csv").write_text("id_str,p_notlens,p_lens\n")
    args = _Args(out_dir=str(tmp_path), force=False)
    res = lt._train_one("sr", rows=[], args=args, reporter=_Rep(), step_offset=0)
    assert res["skipped"] is True
    assert res["global_step"] == 0
    assert res["predictions"].endswith(os.path.join("sr", "predictions.csv"))
