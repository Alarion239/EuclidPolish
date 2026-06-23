"""Tests for euclid_polish.lensfinder.metrics (pure numpy, no torch/TF)."""

from __future__ import annotations

import numpy as np

from euclid_polish.lensfinder import metrics as m


class TestRocAuc:
    def test_perfect_separation_auc_one(self):
        scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
        labels = [1, 1, 1, 0, 0, 0]
        fpr, tpr, _ = m.roc_curve(scores, labels)
        assert abs(m.auc(fpr, tpr) - 1.0) < 1e-9

    def test_all_equal_scores_auc_half(self):
        scores = [0.5] * 6
        labels = [1, 0, 1, 0, 1, 0]
        fpr, tpr, _ = m.roc_curve(scores, labels)
        assert abs(m.auc(fpr, tpr) - 0.5) < 1e-9

    def test_curve_monotone_and_bounded(self):
        rng = np.random.default_rng(0)
        scores = rng.random(200)
        labels = (rng.random(200) < 0.3).astype(int)
        fpr, tpr, _ = m.roc_curve(scores, labels)
        assert np.all(np.diff(fpr) >= -1e-12)
        assert np.all(np.diff(tpr) >= -1e-12)
        assert fpr[0] == 0.0 and tpr[0] == 0.0
        assert abs(fpr[-1] - 1.0) < 1e-9 and abs(tpr[-1] - 1.0) < 1e-9


class TestThresholdAtFpr:
    def test_caps_false_positive_rate(self):
        # 1000 negatives uniform in [0,1); target 1e-3 admits ~1 false positive.
        rng = np.random.default_rng(1)
        neg = rng.random(1000)
        pos = rng.random(50)
        scores = np.r_[neg, pos]
        labels = np.r_[np.zeros(1000, int), np.ones(50, int)]
        thr = m.threshold_at_fpr(scores, labels, 1e-3)
        realized_fpr = float((neg >= thr).mean())
        assert realized_fpr <= 1e-3 + 1e-12

    def test_no_negatives_returns_inf(self):
        assert m.threshold_at_fpr([0.9, 0.8], [1, 1], 1e-3) == float("inf")

    def test_zero_target_admits_no_negative(self):
        scores = [0.9, 0.4, 0.3]
        labels = [1, 0, 0]
        thr = m.threshold_at_fpr(scores, labels, 0.0)
        assert (np.asarray([0.4, 0.3]) >= thr).sum() == 0


class TestTprVsThetaE:
    def test_single_global_threshold_and_binning(self):
        # Positives: small-theta lenses are HARD (low score), big-theta EASY.
        # One global threshold; TPR should rise with theta_E.
        neg = np.full(1000, 0.1)
        neg[0] = 0.95                              # one strong false positive
        # threshold@FPR=1e-3 lands just above the 2nd-highest negative (~0.1):
        pos_small = np.full(20, 0.05)              # below threshold → missed
        pos_big = np.full(20, 0.99)                # above threshold → recovered
        scores = np.r_[neg, pos_small, pos_big]
        labels = np.r_[np.zeros(1000, int), np.ones(40, int)]
        theta = np.r_[np.full(1000, np.nan), np.full(20, 0.5), np.full(20, 2.0)]
        bins = m.theta_e_bins(0.0, 3.0, 3)         # [0,1),[1,2),[2,3)
        res = m.tpr_vs_theta_e(scores, labels, theta, target_fpr=1e-3, bins=bins)
        by = {(b["theta_lo"], b["theta_hi"]): b for b in res["bins"]}
        small_bin = by[(0.0, 1.0)]
        big_bin = by[(2.0, 3.0)]
        assert small_bin["n_pos"] == 20 and big_bin["n_pos"] == 20
        assert small_bin["tpr"] == 0.0            # hard small-theta lenses missed
        assert big_bin["tpr"] == 1.0              # easy big-theta lenses found
        # a bin with no positives reports NaN, not a crash
        assert np.isnan(by[(1.0, 2.0)]["tpr"])

    def test_bootstrap_ci_brackets_point_estimate(self):
        rng = np.random.default_rng(2)
        neg = rng.random(800)
        pos = 0.5 + 0.5 * rng.random(80)
        scores = np.r_[neg, pos]
        labels = np.r_[np.zeros(800, int), np.ones(80, int)]
        thr = m.threshold_at_fpr(scores, labels, 1e-2)
        point = m.tpr_at_threshold(scores, labels, thr)
        ci = m.bootstrap_tpr_ci(scores, labels, target_fpr=1e-2, n_boot=200)
        assert ci["lo"] <= point <= ci["hi"] + 1e-9
