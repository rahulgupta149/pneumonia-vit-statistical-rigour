"""Tests for the statistical core, checked against independent references."""

import math

import numpy as np
import pytest
from scipy import stats

from pneumonia_stats.core import (
    auc_midrank,
    clopper_pearson,
    confusion_counts,
    delong_paired,
    delong_single,
    holm_adjust,
    mcnemar_exact,
    model_summary,
    paired_bootstrap,
    wilson_interval,
)


# ---------------------------------------------------------------------------
# Wilson and Clopper-Pearson
# ---------------------------------------------------------------------------
def test_wilson_matches_published_value():
    # statsmodels.stats.proportion.proportion_confint(81, 263, method="wilson")
    low, high = wilson_interval(81, 263)
    assert low == pytest.approx(0.255289, abs=1e-5)
    assert high == pytest.approx(0.366210, abs=1e-5)


def test_wilson_edge_cases():
    assert wilson_interval(0, 20)[0] == 0.0
    assert wilson_interval(20, 20)[1] == 1.0
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_wilson_contains_point_estimate():
    for successes, n in [(3, 10), (0, 7), (7, 7), (95, 380)]:
        low, high = wilson_interval(successes, n)
        assert low <= successes / n <= high
        assert 0.0 <= low <= high <= 1.0


def test_clopper_pearson_boundaries():
    assert clopper_pearson(0, 10)[0] == 0.0
    assert clopper_pearson(10, 10)[1] == 1.0
    low, high = clopper_pearson(5, 10)
    assert 0.0 < low < 0.5 < high < 1.0


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def test_confusion_counts():
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 1, 0]
    counts = confusion_counts(y_true, y_pred)
    assert counts == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}


def test_model_summary_structure():
    y_true = [0] * 10 + [1] * 20
    y_pred = [0] * 8 + [1] * 2 + [1] * 18 + [0] * 2
    summary = model_summary(y_true, y_pred)
    assert summary["n"] == 30
    assert summary["accuracy"]["successes"] == 26
    assert summary["sensitivity"]["successes"] == 18
    assert summary["specificity"]["successes"] == 8


# ---------------------------------------------------------------------------
# AUC and DeLong
# ---------------------------------------------------------------------------
def _brute_force_delong(y_true, scores_a, scores_b):
    """Slow O(mn) DeLong implementation used as the reference."""
    y_true = np.asarray(y_true).astype(int)
    positive = y_true == 1
    negative = ~positive
    x1, x2 = np.asarray(scores_a)[positive], np.asarray(scores_b)[positive]
    y1, y2 = np.asarray(scores_a)[negative], np.asarray(scores_b)[negative]

    def psi(first, second):
        first = first[:, None]
        second = second[None, :]
        return np.where(first > second, 1.0, np.where(first == second, 0.5, 0.0))

    p1, p2 = psi(x1, y1), psi(x2, y2)
    auc1, auc2 = p1.mean(), p2.mean()
    v10 = np.column_stack([p1.mean(axis=1), p2.mean(axis=1)])
    v01 = np.column_stack([p1.mean(axis=0), p2.mean(axis=0)])
    covariance = np.cov(v10, rowvar=False) / len(x1) + np.cov(v01, rowvar=False) / len(y1)
    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    z_stat = (auc2 - auc1) / math.sqrt(variance)
    p_value = 2 * stats.norm.sf(abs(z_stat))
    return auc1, auc2, z_stat, p_value


def test_auc_midrank_matches_sklearn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    for _ in range(5):
        y_true = rng.integers(0, 2, 200)
        scores = np.round(rng.random(200), 2)  # rounding forces ties
        assert auc_midrank(y_true, scores) == pytest.approx(
            metrics.roc_auc_score(y_true, scores), abs=1e-12
        )


def test_delong_matches_brute_force():
    rng = np.random.default_rng(11)
    for _ in range(5):
        n = int(rng.integers(40, 200))
        y_true = rng.integers(0, 2, n)
        if y_true.sum() in (0, n):
            continue
        scores_a = np.round(rng.normal(size=n), 2)
        scores_b = np.round(scores_a * 0.8 + rng.normal(0, 0.6, n), 2)

        got = delong_paired(y_true, scores_a, scores_b)
        auc_a, auc_b, z_stat, p_value = _brute_force_delong(
            y_true, scores_a, scores_b
        )

        assert got["auc_a"] == pytest.approx(auc_a, abs=1e-10)
        assert got["auc_b"] == pytest.approx(auc_b, abs=1e-10)
        assert got["z"] == pytest.approx(z_stat, abs=1e-9)
        assert got["p_value"] == pytest.approx(p_value, abs=1e-9)


def test_delong_identical_models():
    rng = np.random.default_rng(5)
    y_true = rng.integers(0, 2, 120)
    scores = rng.random(120)
    result = delong_paired(y_true, scores, scores)
    assert result["auc_diff"] == 0.0
    assert result["z"] == 0.0
    assert result["p_value"] == 1.0
    assert result["significant"] is False


def test_delong_single_ci_brackets_auc():
    rng = np.random.default_rng(8)
    y_true = rng.integers(0, 2, 150)
    scores = rng.random(150) + 0.8 * y_true
    result = delong_single(y_true, scores)
    assert 0.0 <= result["ci_low"] <= result["auc"] <= result["ci_high"] <= 1.0


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------
def test_mcnemar_effect_sizes_hand_checked():
    correct_a = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0], dtype=bool)
    correct_b = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 0], dtype=bool)
    result = mcnemar_exact(correct_a, correct_b)

    assert result["n01_baseline_wrong_challenger_right"] == 2
    assert result["n10_baseline_right_challenger_wrong"] == 1
    assert result["discordant"] == 3
    assert result["odds_ratio"] == pytest.approx(2.0)
    assert result["risk_difference"] == pytest.approx(0.1)

    low, high = result["risk_difference_ci"]
    assert low <= result["risk_difference"] <= high
    assert -1.0 <= low <= high <= 1.0


def test_mcnemar_exact_p_matches_binomial():
    # 70 agreed correct, 25 favour B, 5 favour A -> discordant 30
    correct_a = np.array([True] * 70 + [False] * 25 + [True] * 5)
    correct_b = np.array([True] * 70 + [True] * 25 + [False] * 5)
    result = mcnemar_exact(correct_a, correct_b)
    expected = stats.binomtest(5, 30, 0.5).pvalue
    assert result["n01_baseline_wrong_challenger_right"] == 25
    assert result["n10_baseline_right_challenger_wrong"] == 5
    assert result["p_exact"] == pytest.approx(expected)
    assert result["significant"] is True


def test_mcnemar_no_discordance():
    correct = np.array([True, False, True, True], dtype=bool)
    result = mcnemar_exact(correct, correct.copy())
    assert result["discordant"] == 0
    assert result["p_exact"] == 1.0
    assert result["significant"] is False


# ---------------------------------------------------------------------------
# Bootstrap and Holm
# ---------------------------------------------------------------------------
def test_bootstrap_identical_models():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, 100)
    result = paired_bootstrap(
        y_true, y_true.copy(), y_true.copy(), "accuracy", n_resamples=500
    )
    assert result["point_estimate"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0
    assert result["significant"] is False


def test_bootstrap_is_deterministic_for_a_seed():
    rng = np.random.default_rng(4)
    y_true = rng.integers(0, 2, 120)
    pred_a = rng.integers(0, 2, 120)
    pred_b = rng.integers(0, 2, 120)
    first = paired_bootstrap(y_true, pred_a, pred_b, "accuracy", 300, seed=7)
    second = paired_bootstrap(y_true, pred_a, pred_b, "accuracy", 300, seed=7)
    assert first == second


def test_bootstrap_accuracy_point_estimate():
    y_true = np.array([0, 0, 1, 1])
    pred_a = np.array([0, 0, 0, 0])   # 50% accuracy
    pred_b = np.array([0, 0, 1, 1])   # 100% accuracy
    result = paired_bootstrap(
        y_true, pred_a, pred_b, "accuracy", n_resamples=200, seed=1
    )
    assert result["point_estimate"] == pytest.approx(0.5)
    assert result["n_resamples"] == 200
    assert result["n_skipped"] == 0


def test_holm_matches_textbook_example():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_holm_caps_at_one_and_keeps_order():
    adjusted = holm_adjust([0.5, 0.9, 0.001])
    assert adjusted[0] == 1.0
    assert adjusted[1] == 1.0
    assert adjusted[2] == pytest.approx(0.003)


def test_holm_empty_input():
    assert holm_adjust([]).size == 0
