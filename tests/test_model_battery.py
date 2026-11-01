"""The model battery's bookkeeping and summaries (euclid_polish/eval/battery):
the SR cache keys outputs by input, the store keeps stale and capped results
apart, and every table number — grid shares, flux medians per magnitude range,
outskirt recovery, resolution ratios, the synthetic score — is the reduction
the 2026-10-29/30 drivers made, on small hand-built inputs."""
from __future__ import annotations

import json

import numpy as np
import pytest

from euclid_polish.eval.battery import (
    figures,
    flux,
    grid,
    growth,
    holes,
    outskirts,
    report,
    resolution,
    synthetic,
)
from euclid_polish.eval.battery.common import (
    Battery,
    Limits,
    ModelInfo,
    SRCache,
    Store,
    input_digest,
    short,
    step_text,
)


def _info(spec="universal:member_238", member="member_238", step=None, fingerprint="ckpt-81:1:2") -> ModelInfo:
    return ModelInfo(spec, spec.replace(":", "-").replace("@", "-"), short(spec), "universal", member, member, step,
                     "/nowhere", fingerprint)


def _battery(tmp_path, limits=Limits()) -> Battery:
    calls = []

    def runner(info, lr):
        calls.append(info.spec)
        return np.full((2 * lr.shape[0], 2 * lr.shape[1], 4), float(lr.sum()), np.float32)

    bt = Battery(store=Store(str(tmp_path / "out"), limits.tag), cache=SRCache(str(tmp_path / "cache"), runner),
                 runner=None, limits=limits, log=lambda msg: None)
    bt.calls = calls
    return bt


# --------------------------------------------------------------------------- #
# common
# --------------------------------------------------------------------------- #

def test_sr_cache_runs_each_input_once_and_keeps_inputs_apart(tmp_path):
    bt = _battery(tmp_path)
    info = _info()
    a = np.ones((3, 3, 4), np.float32)
    b = a.copy()
    b[0, 0, 0] = 0.0
    first = bt.cache.get(info, "nexus-f200w-0001", a)
    again = bt.cache.get(info, "nexus-f200w-0001", a.copy())
    other = bt.cache.get(info, "nexus-f200w-0001", b)          # same key, another input: its own output
    assert bt.calls == [info.spec, info.spec]
    np.testing.assert_array_equal(first, again)
    assert other[0, 0, 0] == pytest.approx(b.sum())
    assert input_digest(a) != input_digest(b)
    assert input_digest(a) != input_digest(a.reshape(9, 4))      # the shape counts


def test_sr_cache_prune_keeps_what_the_figures_read(tmp_path):
    bt = _battery(tmp_path)
    info = _info()
    lr = np.ones((2, 2, 4), np.float32)
    for key in ("nexus-f200w-0001", "nexus-f200w-0002", "syn-007", "poster-target_181255"):
        bt.cache.get(info, key, lr)
    freed = bt.cache.prune(info, lambda key: not key.startswith("nexus-") or key == "nexus-f200w-0002")
    assert freed > 0
    assert bt.cache.keys(info) == ["nexus-f200w-0002", "poster-target_181255", "syn-007"]


def test_cache_dir_is_the_checkpoint_not_the_spec():
    a = _info("production", fingerprint="ckpt-116:9:9")
    b = _info("universal:member_257", fingerprint="ckpt-116:9:9")
    a = ModelInfo(a.spec, a.slug, a.name, a.kind, a.label, "member_257", None, a.member_dir, a.fingerprint)
    b = ModelInfo(b.spec, b.slug, b.name, b.kind, b.label, "member_257", None, b.member_dir, b.fingerprint)
    assert a.cache_id == b.cache_id
    moved = ModelInfo(a.spec, a.slug, a.name, a.kind, a.label, "member_257", None, a.member_dir, "ckpt-120:9:9")
    assert moved.cache_id != a.cache_id


def test_store_reads_back_only_the_same_version_limits_and_checkpoint(tmp_path):
    info = _info()
    store = Store(str(tmp_path), "")
    store.write(info, "grid", {"x": np.float32(1.5), "bad": float("nan")}, {"a": np.arange(3)})
    got = store.read(info, "grid")
    assert got["x"] == 1.5 and got["bad"] is None
    np.testing.assert_array_equal(store.arrays(info, "grid")["a"], [0, 1, 2])
    assert Store(str(tmp_path), "tiles=6").read(info, "grid") is None               # a smoke run's
    assert store.read(_info(fingerprint="ckpt-82:1:2"), "grid") is None              # another checkpoint
    (tmp_path / "models" / info.slug / "grid.npz").unlink()
    assert store.read(info, "grid") is None                                          # arrays gone


def test_resynced_test_records_void_only_the_synthetic_stages(tmp_path):
    info = _info()
    v5 = Store(str(tmp_path), "", "dirty:1|hr:1")
    for stage in ("synthetic", "outskirt_rc_40", "resolution", "holes"):
        v5.write(info, stage, {"stage": stage})
    v6 = Store(str(tmp_path), "", "dirty:2|hr:2")
    assert v6.read(info, "synthetic") is None and v6.read(info, "outskirt_rc_40") is None
    assert v6.read(info, "resolution")["stage"] == "resolution" and v6.read(info, "holes") is not None


def test_stage_computes_once_and_force_recomputes_once_per_run(tmp_path):
    runs = []

    def compute():
        runs.append(1)
        return {"n": len(runs)}, None

    bt = _battery(tmp_path)
    assert bt.stage(None, "s", compute)["n"] == 1
    assert bt.stage(None, "s", compute)["n"] == 1                   # stored: not recomputed
    again = _battery(tmp_path)                                       # a later run over the same output
    again.force = True
    assert again.stage(None, "s", compute)["n"] == 2
    assert again.stage(None, "s", compute)["n"] == 2                # once per run


def test_names_and_limits():
    assert step_text(15_000_000) == "15M" and step_text(200_000) == "200k"
    assert short("milestone:member_257@15000000") == "257@15M"
    assert short("universal:member_238") == "238"
    assert Limits().tag == "" and Limits.smoke().tag == "tiles=6,fields=3,hosts=2,grad_tiles=1"


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #

def test_grid_shares_pool_windows_and_bands_per_unit():
    z = np.zeros((2, 3, 4))
    z[0, 0, :] = 5.0           # unit 0: 4 hits of 12
    z[1, 2, 0] = 3.0           # unit 1: 1 hit of 12, exactly at the threshold
    hits, n = grid.per_unit(z, np.ones((2, 3), bool))
    assert list(hits) == [4.0, 1.0] and list(n) == [12.0, 12.0]
    assert grid.pooled((hits, n)) == pytest.approx(5 / 24)
    valid = np.array([[False, True, True], [True, True, True]])
    hits, n = grid.per_unit(z, valid)
    assert list(hits) == [0.0, 1.0] and list(n) == [8.0, 12.0]
    assert list(grid.per_unit(z, np.ones((2, 3), bool), 6.0)[0]) == [0.0, 0.0]


def test_grid_paired_difference_is_zero_for_identical_models_and_signed():
    a = (np.array([1.0, 2.0, 3.0, 0.0]), np.array([10.0, 10.0, 10.0, 10.0]))
    b = (np.array([0.0, 1.0, 1.0, 0.0]), np.array([10.0, 10.0, 10.0, 10.0]))
    same = grid.paired(a, a)
    assert same["median"] == 0.0 and same["ci95"] == [0.0, 0.0]
    d = grid.paired(a, b)
    assert d["median"] > 0 and d["ci95"][0] <= d["median"] <= d["ci95"][1]
    assert grid.paired(a, b) == d                                      # seeded: reproducible
    lo, hi = grid.share_ci(a)
    assert lo <= grid.pooled(a) <= hi


def test_grid_outskirt_masks():
    arrays = {"out_real": np.array([[True, True, False]]), "out_syn": np.array([[True, True, True]])}
    catalogue = {"star_near": np.array([[False, True, False]]), "gal_in": np.array([[True, True, True]])}
    refs = {"truth_syn": {"mean": np.array([[[1.0], [0.4], [0.6]]])},
            "lr_syn": {"mean": np.array([[[1.0], [1.0], [1.0]]])}}
    masks = grid.outskirt_masks(arrays, catalogue, refs)
    assert masks["real"].tolist() == [[True, False, False]]           # a star near the second window
    assert masks["syn"].tolist() == [[True, False, True]]             # truth < ½ input light in the second


def test_grid_stats_row_reads_the_protocol_windows():
    U, N = 2, 4
    s = {"pz32": np.zeros((U, 6, 6, 4)), "pz0": np.zeros((U, 7, 7, 4)), "z_e": np.zeros((U, N, 4)),
         "amp_e": np.ones((U, N, 4)), "valid": np.ones((U, N), bool), "pvalid32": np.ones((U, 6, 6), bool)}
    s["pz32"][0, 0, 0, :] = 4.0
    s["pz0"][1, 0, 0, 0] = 7.0
    s["z_e"][0, 1, :] = 3.5
    masks = {"real": np.ones((U, N), bool), "syn": np.ones((U, N), bool)}
    refs = {"lr_real": {"mean": np.full((U, N, 4), 2.0)}, "lr_syn": {"mean": np.full((U, N, 4), 2.0)}}
    meta = {"sigma_lr_real": [4.0] * 4, "sigma_lr_syn": [4.0] * 4}
    row = grid.stats_row(s, s, masks, refs, meta)
    assert row["real3"] == pytest.approx(4 / (2 * 36 * 4))
    assert row["syn3"] == pytest.approx(1 / (2 * 49 * 4)) and row["syn6"] == pytest.approx(1 / (2 * 49 * 4))
    assert row["real_out3"] == pytest.approx(4 / (U * N * 4))
    assert row["real_vis_and_h"] == 1.0
    assert row["real_amp_hit"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# flux
# --------------------------------------------------------------------------- #

def test_flux_ranges_and_bootstrapped_medians():
    assert [flux.range_name(a, b) for a, b in flux.KRON_RANGES] == ["21-23", "23-24", "24-24.5", "24.5-25",
                                                                    "21-25"]
    cell = flux.boot_median(np.array([1.0, 2.0, 3.0, np.nan, 4.0, 5.0]))
    assert cell["median"] == 3.0 and cell["n"] == 5
    assert cell["ci68"][0] <= 3.0 <= cell["ci68"][1]
    assert flux.boot_median(np.array([np.nan]))["median"] is None
    d = flux.paired_median(np.array([0.1, 0.2, -0.1, 0.3]))
    assert d["median"] == pytest.approx(0.15) and d["frac_positive"] == 0.75


def test_flux_isolated_is_the_input_matching_the_catalogue_within_half_sigma():
    keep = flux.isolated(np.array([100.0, 100.0, 100.0, 100.0]), np.array([104.0, 106.0, 100.0, 0.0]),
                         np.array([10.0, 10.0, 0.0, 10.0]))
    assert keep.tolist() == [True, False, False, False]


def test_flux_kron_summary_bins_sr_over_input_by_magnitude(tmp_path, monkeypatch):
    bt = _battery(tmp_path)
    base, new = _info(), _info("milestone:member_257@15000000", "member_257", 15_000_000)
    lr = np.array([100.0, 100.0, 100.0, 100.0])
    geo = {"lr_kron_vis": lr}
    cols = {"vis_mag": np.array([21.5, 22.5, 23.5, 24.7]), "cat:kron_vis": lr.copy(),
            "err:kron_vis": np.full(4, 10.0)}
    monkeypatch.setattr(flux, "geometry", lambda bt: geo)
    monkeypatch.setattr(flux, "catalogue_columns", lambda bt, g: cols)
    monkeypatch.setattr(flux, "report", lambda g, c, m: {})
    for info, sr in ((base, [98.0, 96.0, 93.0, 90.0]), (new, [97.0, 96.0, 95.0, 90.0])):
        bt.store.write(info, "flux", {}, {"kron_vis": np.array(sr)})
    s = flux.kron_summary(bt, [base, new])
    assert s["n_isolated"] == 4
    assert s["models"][base.spec]["21-23"]["median"] == pytest.approx(0.97)      # 0.98 and 0.96
    assert s["models"][base.spec]["23-24"]["median"] == pytest.approx(0.93)
    assert s["models"][new.spec]["24.5-25"]["median"] == pytest.approx(0.90)
    assert s["vs_baseline"][new.spec]["23-24"]["median"] == pytest.approx(0.02)
    assert s["vs_baseline"][new.spec]["24.5-25"]["frac_positive"] == 0.0


def test_flux_synthetic_summary_bins_by_truth_magnitude(tmp_path, monkeypatch):
    bt = _battery(tmp_path)
    info = _info()
    src = {"truth_big": np.array([1.0, 1.0, 1.0]), "lr_big": np.array([0.9, 0.9, 0.9])}
    monkeypatch.setattr(flux, "synthetic_sources", lambda bt: src)
    monkeypatch.setattr(flux, "truth_mag", lambda t: np.array([20.0, 23.5, 25.5]))
    bt.store.write(info, "synflux", {}, {"sr": np.zeros(3), "sr_big": np.array([0.99, 0.95, 0.9])})
    s = flux.synthetic_summary(bt, [info])
    m = s["models"][info.spec]
    assert m["sr_over_truth"]["19-23"]["median"] == pytest.approx(0.99)
    assert m["sr_over_truth"]["25-26"]["median"] == pytest.approx(0.9)
    assert m["sr_over_input"]["23-24"]["median"] == pytest.approx(0.95 / 0.9)
    assert m["sr_over_truth"]["24-25"]["median"] is None
    assert s["input_over_truth"]["19-23"]["median"] == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #

def _model_report(r1, gain_over_coh):
    period = [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    coh = [0.5] * len(period)
    return {"asinh_euclid": {"windows": 10, "period_px": period, "coherence": coh,
                             "gain": [c * gain_over_coh for c in coh],
                             "resolution_coherence": {"arcsec": r1},
                             "resolution_coherence_noise_corrected": {"arcsec": r1 - 0.01}},
            "knee_scan": [{"multiple": 0.3, "resolution_coherence": {"arcsec": r1 + 0.2}},
                          {"multiple": 3.0, "resolution_coherence": None}]}


def test_resolution_entry_texture_and_words():
    e = resolution.band_entry(_model_report(1.2, 0.3))
    assert e["r1"] == 1.2 and e["nc1"] == pytest.approx(1.19) and e["r03"] == pytest.approx(1.4)
    assert e["r3"] is None
    assert e["p816"] == pytest.approx(0.09)                  # (gain / coherence)² over 8 ≤ p < 16
    assert resolution.words(0.992) == "0.8% sharper" and resolution.words(1.007) == "0.7% blurrier"
    cur = {"VIS": {"r1": 0.99}, "Y_E": {"r1": 2.0}}
    base = {"VIS": {"r1": 1.0}, "Y_E": {"r1": 2.0}, "J_E": {"r1": 1.0}}
    m, per = resolution.mean_ratio(cur, base)
    assert m == pytest.approx(0.995) and per == [0.99, 1.0]


def test_resolution_summary_compares_with_the_baseline(tmp_path):
    bt = _battery(tmp_path)
    base, new = _info(), _info("milestone:member_257@15000000", "member_257", 15_000_000)
    for info, r1, t in ((base, 1.0, 0.3), (new, 0.99, 0.2)):
        bt.store.write(info, "resolution", {"bands": {b: _model_report(r1, t) for b in ("VIS", "Y_E", "J_E", "H_E")},
                                            "meta": {"VIS": {}}})
    s = resolution.summary(bt, [base, new])
    assert s["models"][base.spec]["p816"] == pytest.approx(0.09)
    assert s["vs_baseline"][new.spec]["r1"]["words"] == "1.0% sharper"
    assert s["vs_baseline"][new.spec]["p816_ratio"] == pytest.approx(0.04 / 0.09)


# --------------------------------------------------------------------------- #
# synthetic score
# --------------------------------------------------------------------------- #

def test_synthetic_summary_counts_fields_won_and_knee_deltas(tmp_path):
    bt = _battery(tmp_path)
    base, new = _info(), _info("milestone:member_257@15000000", "member_257", 15_000_000)
    knees = np.array(synthetic.KNEE_GRID_E)
    curves = np.full((3, len(knees), 4), 60.0)
    better = curves.copy()
    better[:2] += 1.0
    better[2] -= 0.5
    better[:, 0, 0] = 59.0
    pairs = {"2-3": [[1.5, 1.0]] * 4, "3-6": [[1.0, 1.0]] * 4, "21-42": [[1.04, 1.0]] * 4}
    for info, c in ((base, curves), (new, better)):
        bt.store.write(info, "synthetic", {"axis_diag_ratio": synthetic.axis_diag_ratios({"direction": {
            "axis_diag": pairs}}), "mean": 60.0, "integrated": [60.0] * 4, "n_fields": 3, "source": "test",
            "check": None, "learned_knees": None},
            {"fields": np.arange(3), "knees": knees, "curves": c})
    s = synthetic.summary(bt, [base, new])
    assert s["models"][base.spec]["axis_diag_2_3"] == [1.5, 1.5]
    assert s["models"][base.spec]["axis_diag_3_42"] == [1.0, 1.04]
    vs = s["vs_baseline"][new.spec]
    assert vs["fields_better"] == 2 and vs["n_fields"] == 3
    assert vs["vis_delta_at_lowest_knee_db"] == pytest.approx(-1.0)


def test_per_field_integrated_is_the_score_reduction_of_each_field():
    knees = synthetic.KNEE_GRID_E
    curves = np.stack([np.full((len(knees), 4), 50.0), np.tile(np.linspace(40, 60, len(knees))[:, None], 4)])
    out = synthetic.per_field_integrated(curves)
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out[0], 50.0)
    np.testing.assert_allclose(out[1], synthetic.integrated_psnr(curves[1]))


# --------------------------------------------------------------------------- #
# outskirts
# --------------------------------------------------------------------------- #

def test_outskirt_recoveries_and_paired_field_bootstrap(tmp_path):
    bt = _battery(tmp_path)
    base, new = _info(), _info("milestone:member_257@15000000", "member_257", 15_000_000)
    nr = len(outskirts.R_EDGES) - 1
    n = 5
    for info, frac in ((base, 0.6), (new, 0.5)):
        acc_f = np.zeros((n, 2, 2, nr, 4))
        acc_f[:, :, 1] = 1.0
        acc_f[:, :, 0] = frac
        bt.store.write(info, "outskirts_deck", {}, {"acc": acc_f.sum(0), "acc_f": acc_f})
        nm = len(outskirts.MAG_EDGES) - 1
        tru = np.ones((nm, nr, 4))
        rec = np.full((nm, nr, 4), frac)
        rec[:2] = 9.0                       # bright hosts (m < 19.5) are left out
        bt.store.write(info, "outskirts_hosts", {}, {"acc_f": acc_f, "real_rec": rec, "real_tru": tru,
                                                     "syn_rec": rec, "syn_tru": tru})
    s = outskirts.summary(bt, [base, new])
    i = outskirts.QUOTED_RING
    assert s["quoted_ring"] == "2-2.5"
    assert s["deck"]["models"][base.spec]["real-like noise"]["mean4"][i] == pytest.approx(0.6)
    assert s["hosts"]["models"][new.spec]["real NEXUS tiles"]["mean4"][i] == pytest.approx(0.5)
    assert s["hosts"]["vs_baseline"][new.spec]["real NEXUS tiles"]["2-2.5"] == pytest.approx([-0.1] * 3)


def test_outskirt_bins_and_the_shared_bootstrap():
    mi, ri = outskirts.bins(np.array([0.2, 2.2, 9.0, 50.0]), np.array([17.0, 20.0, 99.0, 30.0]))
    assert mi.tolist() == [0, 2, 5, 5] and ri.tolist() == [0, 4, 7, 7]
    np.testing.assert_array_equal(outskirts.boot_index(7), outskirts.boot_index(7))
    assert outskirts.boot_index(7).shape == (outskirts.N_BOOT, 7)


def test_outskirt_source_maps_assign_the_brighter_source():
    s = {"tile": np.array([0, 0]), "mag": np.array([20.0, 22.0]), "kron_px": np.array([3.0, 3.0]),
         "x": np.array([20.0 + outskirts.TRIM, 30.0 + outskirts.TRIM]), "y": np.array([20.0 + outskirts.TRIM] * 2),
         "theta": np.zeros(2), "ellipticity": np.zeros(2), "star": np.array([False, True])}
    rn, star, mag = outskirts.maps(s, 0, shape=(40, 40))
    assert rn[20, 20] == 0.0 and not star[20, 20] and mag[20, 20] == 20.0
    assert rn[20, 30] == 0.0 and star[20, 30]
    assert rn[0, 0] == pytest.approx(min(np.hypot(20, 20) / 3.0, 9.0))


# --------------------------------------------------------------------------- #
# holes and the report
# --------------------------------------------------------------------------- #

def test_holes_summary_reads_the_aggregate(tmp_path):
    bt = _battery(tmp_path)
    info = _info()
    agg = {"n_tiles": 13, "per_band": {"VIS": {"hole_pct": 3.5, "flux_ratio": 0.93}},
           "summary": {"hole_pct_mean": 11.55}}
    bt.store.write(info, "holes", {"nexus": {"summary": agg}, "poster": {"summary": {**agg, "n_tiles": 1}}})
    s = holes.summary(bt, [info])
    assert s["models"][info.spec]["nexus"]["hole_pct_mean"] == 11.55
    assert s["models"][info.spec]["poster"]["n_tiles"] == 1


def _summary_238():
    return {"synthetic": {"models": {"universal:member_238": {
                "mean": 62.1267, "integrated": [56.5, 66.3, 63.2, 62.4], "axis_diag_2_3": [1.48, 1.59],
                "axis_diag_3_42": [0.94, 1.04], "learned_knees": [0.21, 0.1, 0.11, 0.11], "source": "x"}}},
            "grid": {"models": {"universal:member_238": {"real3": 0.0244, "syn3": 0.074, "syn6": 0.032,
                                                         "real_out3": 0.137, "syn_out3": 0.198}}},
            "holes": {"models": {"universal:member_238": {"nexus": {"hole_pct_mean": 11.5495},
                                                          "poster": {"hole_pct_mean": 3.1137}}}},
            "kron": {"ranges": ["21-23"], "n_isolated": 815,
                     "models": {"universal:member_238": {"21-23": {"median": 0.9814}}}},
            "outskirts": {"quoted_ring": "2-2.5", "hosts": {"rings": outskirts.ring_names(), "models": {
                "universal:member_238": {"real NEXUS tiles": {"mean4": [0, 0, 0, 0, 0.5842, 0, 0, 0]}}}}}}


def test_reference_check_rounds_to_the_quoted_digits():
    assert report.matches(62.1267, 62.13, 2) and not report.matches(62.1349, 62.12, 2)
    assert report.matches(0.5842, 0.58, 2) and report.matches(2.44, 2.4, 1) and not report.matches(None, 1, 2)
    check = report.reference_check(_summary_238(), [_info()])["universal:member_238"]
    assert check["synthetic.mean"] == [62.13, 62.1267, True]
    assert check["grid.real3"][2] and check["grid.syn_out3"][2] and check["holes.nexus"][2]
    assert check["kron.21-23"][2] and check["outskirts.real"][2]
    assert check["resolution.p816"] == [0.0635, None, False]              # not measured here


def test_markdown_table_has_one_column_per_model():
    s = _summary_238()
    s["reference_check"] = report.reference_check(s, [_info()])
    text = report.markdown(s, [_info()], {"created": "now", "limits": ""})
    assert "| metric | 238 |" in text
    assert "| ∫PSNR, mean of 4 bands (dB) | 62.13 |" in text
    assert "| real sky (NEXUS clean tiles) | 2.4% |" in text
    assert "| NEXUS, 13 galaxy tiles | 11.55% |" in text
    assert "| real NEXUS tiles as hosts | 0.58 |" in text
    assert "Reference check: universal:member_238" in text
    assert "| resolution.p816 |" not in text                                  # not measured: left out
    json.dumps(report.table_rows(s, [_info()]))


# --------------------------------------------------------------------------- #
# curves of growth
# --------------------------------------------------------------------------- #

def test_growth_rings_mask_neighbours_inside_their_kron_ellipse_and_holes():
    shape = (140, 160)
    x, y = np.array([70.0, 100.0]), np.array([70.0, 70.0])          # target on a pixel centre, neighbour
    kron = np.array([2.0, 4.0])                                      # Kron ellipse = 2.5 r_Kron = 10 px
    owner, best = growth.ownership(x, y, kron, np.zeros(2), np.zeros(2), shape)
    hole = np.zeros(shape, bool)
    hole[20:24, 60:64] = True
    ring, masked = growth.rings(owner, best, hole, 0, x[0], y[0])
    off = growth.R_MAX                                               # local index of the centre
    outside = growth.R_MAX + 1
    assert ring[off, off] == 1                                       # the centre pixel is in the first ring
    assert ring[off, off + 30] == outside                            # the neighbour's own pixels
    assert ring[off, off + 15] != outside                            # its light beyond its Kron ellipse stays
    assert ring[22 - 70 + off, 62 - 70 + off] == outside             # a hole
    assert 0 < masked < growth.MAX_MASKED
    area = growth.cumulative(ring, np.ones(ring.shape + (1,)))[:, 0]
    assert area[-1] == (ring <= growth.R_MAX).sum() and np.all(np.diff(area) >= 0)
    flat = growth.cumulative(ring, np.full(ring.shape + (2,), 3.0))
    np.testing.assert_allclose(flat, 3.0 * area[:, None] * np.ones((1, 2)))


def _growth_geo(n=4):
    area = np.tile(3.0 * np.arange(1, growth.R_MAX + 1) ** 2, (n, 1))
    return {"tile": np.array(["t"] * n), "mag": np.array([21.0, 22.5, 23.5, 25.0])[:n],
            "kron_arcsec": np.full(n, 0.2), "area": area, "lr": area[:, :, None] * np.ones((1, 1, 4))}


def test_growth_summary_ratios_sky_matching_and_pairs(tmp_path, monkeypatch):
    bt = _battery(tmp_path)
    geo = _growth_geo()
    monkeypatch.setattr(growth, "geometry", lambda bt: geo)
    a, b = _info(), _info("universal:member_257", "member_257", fingerprint="ckpt-116:1:2")
    bt.store.write(a, "growth", {}, {"cog": 0.9 * geo["lr"], "sky": np.zeros((4, 4))})
    # b: the same 0.9 raw, but all of it a −0.1 e⁻/px blank-sky offset
    bt.store.write(b, "growth", {}, {"cog": geo["lr"] - 0.1 * geo["area"][:, :, None],
                                     "sky": np.full((4, 4), -0.1)})
    s = growth.summary(bt, [a, b])
    assert s["ranges"] == ["all", "<22", "22-23", "23-24", ">24"]
    vis_a, vis_b = s["models"][a.spec]["bands"]["VIS"], s["models"][b.spec]["bands"]["VIS"]
    assert vis_a["all"]["r3"]["median"] == pytest.approx(0.9) and vis_a["all"]["r3"]["n"] == 4
    assert vis_a["all"]["r3_sky"]["median"] == pytest.approx(0.9)
    assert vis_b["all"]["r3"]["median"] == pytest.approx(0.9)
    assert vis_b["all"]["r3_sky"]["median"] == pytest.approx(1.0)
    assert vis_b[">24"]["r3_sky"]["n"] == 1
    assert s["models"][a.spec]["bands"]["H"]["all"]["kron"]["median"] == pytest.approx(0.9)
    assert vis_b["all"]["r50"]["median"] == pytest.approx(1.0)
    assert s["models"][b.spec]["sky"] == pytest.approx([-0.1] * 4)
    assert s["vs_baseline"][b.spec]["VIS"]["all"]["median"] == pytest.approx(0.0)
    rows = {(sec, metric): cells for sec, metric, cells in report.table_rows({"growth": s}, [a, b])}
    sec = next(sec for sec, _m in rows if sec.startswith("Curve of growth"))
    assert rows[(sec, "VIS F(<3″), mag <22 / 22-23 / 23-24 / >24")] == ["0.900 / 0.900 / 0.900 / 0.900"] * 2
    assert rows[(sec, "VIS F(<3″) sky-matched, mag <22 / 22-23 / 23-24 / >24")][1] == "1.000 / 1.000 / 1.000 / 1.000"
    paired = report.paired_rows({"growth": s}, [a, b])
    assert any(metric == "VIS, all" and cells[0].startswith("+0.000") for _s, metric, cells in paired)


def test_growth_half_light_and_kron_aperture():
    r = growth.RADII
    curve = np.minimum(r / 1.0, 1.0)[None, :]                       # linear to 1″, then flat
    assert growth.half_light(curve)[0] == pytest.approx(0.5)
    ratio = np.tile(r, (2, 1))
    out = growth.at_kron(ratio, np.array([0.2, 2.0]))               # 0.5″ inside, 5″ beyond 3″
    assert out[0] == pytest.approx(0.5) and np.isnan(out[1])


def test_figure_jobs_are_only_those_of_the_tests_that_ran(tmp_path):
    bt = _battery(tmp_path)
    names = [name for name, _job in figures.jobs(bt, [_info()], {}, str(tmp_path), ["growth"])]
    assert names == ["curve of growth"]
    assert "real sky" in [name for name, _job in figures.jobs(bt, [_info()], {}, str(tmp_path), ["flux"])]
