from __future__ import annotations

import json

import numpy as np
from astropy.io import fits

import euclid_polish.web.helpers.viewer_data as vd


def _obj(dirpath, with_pca=True):
    def _wr(name, arr):
        fits.PrimaryHDU(np.ascontiguousarray(arr.astype(np.float32))).writeto(
            str(dirpath / name), overwrite=True, output_verify="silentfix")
    _wr("original_stack.fits", np.zeros((4, 8, 8)))
    _wr("SR.fits", np.zeros((4, 16, 16)))
    if with_pca:
        _wr("std.fits", np.zeros((4, 16, 16)))
        for i in range(3):
            _wr(f"pca{i}.fits", np.ones((4, 16, 16)) * (i + 1))
        with open(dirpath / "disagreement.json", "w") as f:
            json.dump({"pca_n": 3, "pca_amps": [0.3, 0.2, 0.1]}, f)


def test_eval_meta_advertises_morph(tmp_path, monkeypatch):
    d = tmp_path / "obj_a"
    d.mkdir()
    _obj(d)
    monkeypatch.setattr(vd, "_eval_objects", lambda: [{
        "subdir": "obj_a", "label": "a", "grade": "A",
        "tiers": ["LR", "SR", "std"], "plens": {},
        "pca_n": 3, "pca_amps": [0.3, 0.2, 0.1]}])
    meta = vd._eval_meta({})
    assert any(t["key"] == "morph" for t in meta["tiers"])
    assert meta["pca_n"] == 3
    assert meta["pca_amps"] == [[0.3, 0.2, 0.1]]


def test_eval_cube_serves_pca(tmp_path, monkeypatch):
    d = tmp_path / "obj_a"
    d.mkdir()
    _obj(d)
    monkeypatch.setattr(vd.Config, "EVAL_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(vd, "_eval_objects", lambda: [{
        "subdir": "obj_a", "label": "a", "grade": "A",
        "tiers": ["LR", "SR", "std"], "plens": {},
        "pca_n": 3, "pca_amps": [0.3, 0.2, 0.1]}])
    cube, info = vd._eval_cube(0, "pca1", {})
    # _as_hwc returns (H, W, C); pca1.fits was written (4, 16, 16) -> (16, 16, 4).
    assert cube.shape[:2] == (16, 16)
    # pca1.fits was written as ones*2 -> non-zero content.
    assert float(np.asarray(cube).max()) > 0


def test_eval_cube_serves_sr_lowercase(tmp_path, monkeypatch):
    # The shared morph animation fetches the ensemble mean as lower-case "sr";
    # the eval SR tier key is upper-case "SR". _eval_cube must resolve it
    # case-insensitively, else startMorph shows "movie unavailable".
    d = tmp_path / "obj_a"
    d.mkdir()
    _obj(d)
    monkeypatch.setattr(vd.Config, "EVAL_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(vd, "_eval_objects", lambda: [{
        "subdir": "obj_a", "label": "a", "grade": "A",
        "tiers": ["LR", "SR", "std"], "plens": {},
        "pca_n": 3, "pca_amps": [0.3, 0.2, 0.1]}])
    cube, info = vd._eval_cube(0, "sr", {})          # lower-case, as the morph does
    assert cube.shape[:2] == (16, 16)


def test_eval_viewer_derives_bhr_from_hr(tmp_path, monkeypatch):
    d = tmp_path / "obj_a"
    d.mkdir()
    impulse = np.zeros((4, 16, 16), np.float32)
    impulse[0, 8, 8] = 1.0
    fits.PrimaryHDU(impulse).writeto(d / "HR.fits")
    monkeypatch.setattr(vd.Config, "EVAL_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(vd, "_eval_objects", lambda: [{
        "subdir": "obj_a", "label": "a", "grade": "A",
        "tiers": ["HR", "BHR"], "plens": {},
        "pca_n": 0, "pca_amps": []}])

    meta = vd._eval_meta({})
    hr, _ = vd._eval_cube(0, "HR", {})
    bhr, info = vd._eval_cube(0, "BHR", {})
    zero_blur, _ = vd._eval_cube(0, "BHR", {"bhr_fwhm_arcsec": "0"})

    assert [tier["key"] for tier in meta["tiers"]] == ["HR", "BHR"]
    assert hr[8, 8, 0] == 1.0
    assert bhr[8, 8, 0] < hr[8, 8, 0]
    np.testing.assert_array_equal(zero_blur, hr)
    assert "BHR (blurred HR)" in info["label"]


def test_eval_objects_add_bhr_whenever_hr_exists(tmp_path, monkeypatch):
    d = tmp_path / "obj_a"
    d.mkdir()
    fits.PrimaryHDU(np.zeros((4, 16, 16), np.float32)).writeto(d / "HR.fits")
    (tmp_path / "manifest.csv").write_text(
        "id,ok,out_subdir\nobj_a,true,obj_a\n", encoding="utf-8",
    )
    monkeypatch.setattr(vd.Config, "EVAL_RESULTS_DIR", str(tmp_path))

    obj = vd._eval_objects()[0]

    assert obj["tiers"] == ["HR", "BHR"]


def test_eval_objects_advertise_morph_per_object(tmp_path, monkeypatch):
    """The morph tier must appear in each object's OWN tier list (when its
    disagreement sidecar exists) — the viewer gates tiers per object, so a
    global-only "morph" entry leaves the movie chip disabled everywhere.
    Regression: the eval page offered the movie globally but every object
    said "no morph for this object"."""
    for sub, with_pca in (("obj_a", True), ("obj_b", False)):
        d = tmp_path / sub
        d.mkdir()
        _obj(d, with_pca=with_pca)
    with open(tmp_path / "manifest.csv", "w") as f:
        f.write("id,ok,out_subdir\n"
                "obj_a,true,obj_a\n"
                "obj_b,true,obj_b\n")
    monkeypatch.setattr(vd.Config, "EVAL_RESULTS_DIR", str(tmp_path))
    objs = vd._eval_objects()
    tiers = {o["subdir"]: o["tiers"] for o in objs}
    assert "morph" in tiers["obj_a"]          # has disagreement.json + pca fits
    assert "morph" not in tiers["obj_b"]      # no sidecar → legitimately gated
    # And the meta serialization forwards the per-object list to the client.
    meta = vd._eval_meta({})
    by_sub = {o["subdir"]: o for o in meta["objects"]}
    assert "morph" in by_sub["obj_a"]["tiers"]
    assert "morph" not in by_sub["obj_b"]["tiers"]


def test_eval_meta_no_morph_without_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(vd, "_eval_objects", lambda: [{
        "subdir": "obj_b", "label": "b", "grade": "B",
        "tiers": ["LR", "SR"], "plens": {}, "pca_n": 0, "pca_amps": []}])
    meta = vd._eval_meta({})
    assert meta["pca_n"] == 0
    assert not any(t["key"] == "morph" for t in meta["tiers"])
