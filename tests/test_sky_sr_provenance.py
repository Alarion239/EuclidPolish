"""SR cubes get a provenance record linking model + input (synthetic .npy path).

The heavy ``generate()`` integration needs a real model; this tests the pure
record-building helper it delegates to, so the provenance logic is covered
without a checkpoint.
"""

from __future__ import annotations

from euclid_polish.provenance.records import Format
from euclid_polish.provenance.store import ProvStore
from euclid_polish.web.helpers.sky_records import record_sr_cube


def test_record_sr_cube_links_model_and_input(tmp_path):
    store = ProvStore(str(tmp_path / "store"))
    model_id = store.mint()
    input_id = store.mint()
    run_id = store.mint()
    sidecar_dir = str(tmp_path / "sky_sr")

    art = record_sr_cube(
        store, "sr_validate_0001.npy", "validate", 1,
        model_id=model_id, input_id=input_id, produced_by=run_id,
        git=None, sidecar_dir=sidecar_dir,
    )

    assert art.format is Format.NPY
    assert art.path == "sr_validate_0001.npy"
    assert set(art.parents) == {model_id, input_id}
    assert art.produced_by == run_id
    assert art.descriptors["subset"] == "validate"
    # The sidecar is discoverable next to the data.
    assert ProvStore(str(tmp_path / "store"), data_roots=[sidecar_dir]).get(art.id) == art


def test_record_sr_cube_tolerates_unknown_input(tmp_path):
    store = ProvStore(str(tmp_path / "store"))
    model_id = store.mint()
    art = record_sr_cube(
        store, "sr_train_0000.npy", "train", 0,
        model_id=model_id, input_id=None, produced_by=None,
        git=None, sidecar_dir=str(tmp_path / "sky_sr"),
    )
    assert art.parents == (model_id,)      # input unknown → only the model parent
