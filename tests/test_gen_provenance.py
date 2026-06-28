"""Generation-run provenance helpers (used by run_pipeline to stamp records).

Unit-level: a fake config dataclass + a temp store, no simulator needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from euclid_polish.provenance.records import Format
from euclid_polish.provenance.store import ProvStore
from euclid_polish.sky.generation.gen_provenance import (
    GenerationContext, begin_generation_run, make_generation_context,
)


@dataclass
class FakeCfg:
    image_size: int = 96
    tng_fraction: float = 1.0


def test_begin_generation_run_persists_run_with_config(tmp_path):
    store = ProvStore(str(tmp_path))
    ctx = begin_generation_run(store, FakeCfg(), git=None)
    assert isinstance(ctx, GenerationContext)
    run = store.get(ctx.run_id)
    assert run.KIND == "generationrun"
    assert run.config.config_type == "FakeCfg"
    assert run.config.fields["image_size"] == 96


def test_stamp_shares_one_id_per_kind_subset(tmp_path):
    store = ProvStore(str(tmp_path))
    ctx = begin_generation_run(store, FakeCfg(), git=None)
    s1 = ctx.stamp("clean", "train")
    s2 = ctx.stamp("clean", "train")
    s3 = ctx.stamp("clean", "validate")
    assert s1.id == s2.id            # same file → records share one id
    assert s1.id != s3.id            # different subset → different file id
    assert s1.produced_by == ctx.run_id
    assert s1.subset == "train"


def test_finalize_persists_artifact_linked_to_run(tmp_path):
    store = ProvStore(str(tmp_path))
    ctx = begin_generation_run(store, FakeCfg(), git=None)
    s = ctx.stamp("clean", "train")
    art = ctx.finalize("clean", "train", str(tmp_path / "clean_train.tfrecord"))
    assert art.id == s.id            # the artifact id == the records' embedded id
    assert art.format is Format.TFRECORD
    assert art.produced_by == ctx.run_id
    assert store.get(art.id) == art


def test_finalize_dirty_can_parent_on_clean(tmp_path):
    store = ProvStore(str(tmp_path))
    ctx = begin_generation_run(store, FakeCfg(), git=None)
    clean = ctx.stamp("clean", "train")
    dirty = ctx.stamp("dirty", "train", parents=(clean.id,))
    assert dirty.parents == (clean.id,)
    assert dirty.id != clean.id


def test_stamped_record_round_trips_through_tfrecord(tmp_path):
    """The exact path step_generate uses: ctx.stamp → write → read back."""
    import numpy as np
    from euclid_polish.image.tfio import (
        read_images, write_images,
    )
    from euclid_polish.image import Image

    store = ProvStore(str(tmp_path / "store"))
    ctx = begin_generation_run(store, FakeCfg(), git=None)
    img = Image(
        data=np.zeros((4, 4, 4), np.float32), pixel_scale_arcsec=0.05,
        band_names=("VIS", "Y_E", "J_E", "H_E"), is_clean=True, subset="train",
    )
    img.stamp = ctx.stamp("clean", "train")
    path = write_images([img], "clean_train", records_dir=str(tmp_path))
    [back] = read_images(path, num_images=1)
    assert back.prov_stamp().id == ctx.file_id("clean", "train")
    assert back.prov_stamp().produced_by == ctx.run_id
    assert back.subset == "train"


def test_make_generation_context_is_guarded(monkeypatch):
    import euclid_polish.sky.generation.gen_provenance as gp
    monkeypatch.setattr(
        gp, "default_store",
        lambda: (_ for _ in ()).throw(OSError("no store")),
    )
    assert gp.make_generation_context(FakeCfg()) is None


def test_shard_stamp_plan_is_store_free_and_correct():
    from euclid_polish.provenance.ids import ProvId
    from euclid_polish.sky.generation.gen_provenance import ShardStampPlan
    plan = ShardStampPlan(
        run_id=ProvId("aaaaaaaa"), clean_id=ProvId("bbbbbbbb"),
        hr_id=ProvId("cccccccc"), dirty_id=ProvId("dddddddd"),
    )
    c = plan.clean_stamp("train")
    h = plan.hr_stamp("train")
    d = plan.dirty_stamp("validate")
    assert c.id == ProvId("bbbbbbbb") and c.produced_by == ProvId("aaaaaaaa")
    assert c.subset == "train"
    assert h.id == ProvId("cccccccc") and h.parents == (ProvId("bbbbbbbb"),)
    assert d.id == ProvId("dddddddd") and d.subset == "validate"


def test_shard_stamp_plan_is_picklable():
    """Workers run in separate processes — the plan must pickle."""
    import pickle
    from euclid_polish.provenance.ids import ProvId
    from euclid_polish.sky.generation.gen_provenance import ShardStampPlan
    plan = ShardStampPlan(ProvId("aaaaaaaa"), ProvId("bbbbbbbb"),
                          ProvId("cccccccc"), ProvId("dddddddd"))
    assert pickle.loads(pickle.dumps(plan)) == plan
