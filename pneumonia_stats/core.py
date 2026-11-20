"""Core paired statistical tests for comparing binary classifiers.

All public functions are pure and return plain Python values inside
dictionaries, so results serialise directly to JSON for a paper appendix.

Label convention: ``0 = NORMAL``, ``1 = PNEUMONIA``.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats

DEFAULT_ALPHA = 0.05
DEFAULT_N_RESAMPLES = 10_000
DEFAULT_SEED = 42
Z_95 = 1.959963984540054  # standard-normal 97.5th percentile

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_N_RESAMPLES",
    "DEFAULT_SEED",
    "Z_95",
    "auc_midrank",
    "clopper_pearson",
    "confusion_counts",
    "delong_paired",
    "delong_single",
    "holm_adjust",
    "mcnemar_exact",
    "model_summary",
    "paired_bootstrap",
    "proportion_summary",
    "wilson_interval",
]


# ---------------------------------------------------------------------------
# Proportions
# ---------------------------------------------------------------------------
def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Recommended over the Wald interval when proportions approach 0 or 1, or
    when the sample is small (Newcombe, 1998).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half_width = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    # Pin the endpoints exactly: for 0/n and n/n the formula can return a
    # value one floating-point epsilon outside the admissible range.
    low = 0.0 if successes == 0 else max(0.0, centre - half_width)
    high = 1.0 if successes == n else min(1.0, centre + half_width)
    return low, high


def proportion_summary(successes: int, n: int) -> dict:
    """Point estimate plus a 95% Wilson interval, packaged for tables and JSON."""
    low, high = wilson_interval(successes, n)
    return {
        "estimate": successes / n,
        "ci_low": float(low),
        "ci_high": float(high),
        "successes": int(successes),
        "n": int(n),
    }


def clopper_pearson(
    successes: int, n: int, alpha: float = DEFAULT_ALPHA
) -> tuple[float, float]:
    """Exact (Clopper-Pearson) confidence interval for a proportion."""
    low = (
        0.0
        if successes == 0
        else float(stats.beta.ppf(alpha / 2.0, successes, n - successes + 1))
    )
    high = (
        1.0
        if successes == n
        else float(stats.beta.ppf(1.0 - alpha / 2.0, successes + 1, n - successes))
    )
    return low, high


def _odds(share: float) -> float:
    """Convert a proportion to odds, guarding the boundaries."""
    if share <= 0.0:
        return 0.0
    if share >= 1.0:
        return float("inf")
    return share / (1.0 - share)


# ---------------------------------------------------------------------------
# Confusion matrix and per-model summaries
# ---------------------------------------------------------------------------
def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Return TN/FP/FN/TP for labels 0 = NORMAL and 1 = PNEUMONIA."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
    }


def model_summary(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Accuracy, sensitivity and specificity with 95% Wilson intervals."""
    counts = confusion_counts(y_true, y_pred)
    n = int(len(y_true))
    summary = {
        "n": n,
        "confusion": counts,
        "accuracy": proportion_summary(counts["tn"] + counts["tp"], n),
    }
    positives = counts["tp"] + counts["fn"]
    negatives = counts["tn"] + counts["fp"]
    if positives:
        summary["sensitivity"] = proportion_summary(counts["tp"], positives)
    if negatives:
        summary["specificity"] = proportion_summary(counts["tn"], negatives)
    return summary


# ---------------------------------------------------------------------------
# AUC and DeLong
# ---------------------------------------------------------------------------
def auc_midrank(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based (Mann-Whitney) AUC; identical to ``roc_auc_score``.

    Ties are handled with average ranks, matching scikit-learn.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    positive = y_true == 1
    m = int(positive.sum())
    n = int((~positive).sum())
    if m == 0 or n == 0:
        raise ValueError("AUC requires both classes in y_true")
    ranks = stats.rankdata(scores, method="average")
    return float((ranks[positive].sum() - m * (m + 1) / 2.0) / (m * n))


def _midrank(values: Sequence[float]) -> np.ndarray:
    """Mid-ranks (ties averaged) of a one-dimensional array."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end < len(values) and ordered[end] == ordered[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    out = np.empty(len(values), dtype=float)
    out[order] = ranks
    return out


def _delong_components(
    y_true: Sequence[int], score_rows: Sequence[Sequence[float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DeLong AUCs and structural components for one or more models.

    Implements the fast mid-rank formulation of Sun & Xu (2014), which is
    algebraically identical to DeLong, DeLong & Clarke-Pearson (1988).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.atleast_2d(np.asarray(score_rows, dtype=float))
    positive = y_true == 1
    negative = ~positive
    m = int(positive.sum())
    n = int(negative.sum())
    if m == 0 or n == 0:
        raise ValueError("AUC requires both classes in y_true")

    k = scores.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for row in range(k):
        tx[row] = _midrank(scores[row][positive])
        ty[row] = _midrank(scores[row][negative])
        tz[row] = _midrank(scores[row])

    aucs = (tz[:, positive].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, positive] - tx) / n          # placement values, positives
    v10 = 1.0 - (tz[:, negative] - ty) / m    # placement values, negatives
    return aucs, v01, v10


def delong_single(
    y_true: Sequence[int], scores: Sequence[float], alpha: float = DEFAULT_ALPHA
) -> dict:
    """AUC with an asymptotic DeLong 95% confidence interval (one model)."""
    aucs, v01, v10 = _delong_components(y_true, scores)
    m, n = v01.shape[1], v10.shape[1]
    variance = float(np.var(v01[0], ddof=1) / m + np.var(v10[0], ddof=1) / n)
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    half = z_crit * math.sqrt(variance)
    return {
        "auc": float(aucs[0]),
        "ci_low": float(max(0.0, aucs[0] - half)),
        "ci_high": float(min(1.0, aucs[0] + half)),
        "variance": variance,
    }


def delong_paired(
    y_true: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Two-sided DeLong z-test for ``AUC(B) - AUC(A)`` on paired predictions."""
    aucs, v01, v10 = _delong_components(
        y_true, np.vstack([scores_a, scores_b])
    )
    m, n = v01.shape[1], v10.shape[1]
    covariance = np.cov(v01) / m + np.cov(v10) / n
    delta = float(aucs[1] - aucs[0])
    variance = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    z_stat = delta / math.sqrt(variance) if variance > 0 else 0.0
    p_value = float(2.0 * stats.norm.sf(abs(z_stat)))
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_diff": delta,
        "auc_a_ci": [
            float(aucs[0] - z_crit * math.sqrt(covariance[0, 0])),
            float(aucs[0] + z_crit * math.sqrt(covariance[0, 0])),
        ],
        "auc_b_ci": [
            float(aucs[1] - z_crit * math.sqrt(covariance[1, 1])),
            float(aucs[1] + z_crit * math.sqrt(covariance[1, 1])),
        ],
        "z": float(z_stat),
        "p_value": p_value,
        "significant": bool(p_value < alpha),
    }


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------
def mcnemar_exact(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Exact McNemar test plus effect sizes.

    ``A`` is the baseline and ``B`` the challenger. ``n01`` counts images where
    A is wrong and B is right; ``n10`` counts the reverse. The odds ratio
    ``n01 / n10`` is greater than 1 when B is better.
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    n01 = int(np.sum(~correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    discordant = n01 + n10

    result = {
        "n01_baseline_wrong_challenger_right": n01,
        "n10_baseline_right_challenger_wrong": n10,
        "discordant": discordant,
        "both_correct": int(np.sum(correct_a & correct_b)),
        "both_wrong": int(np.sum(~correct_a & ~correct_b)),
        "risk_difference": float(correct_b.mean() - correct_a.mean()),
    }
    if discordant == 0:
        result.update(
            {
                "p_exact": 1.0,
                "chi2_edwards": 0.0,
                "p_chi2": 1.0,
                "odds_ratio": 1.0,
                "odds_ratio_ci": [1.0, 1.0],
                "risk_difference_ci": [0.0, 0.0],
                "significant": False,
            }
        )
        return result

    result["p_exact"] = float(
        stats.binomtest(min(n01, n10), discordant, 0.5).pvalue
    )

    chi2 = max((abs(n01 - n10) - 1.0) ** 2 / discordant, 0.0)
    result["chi2_edwards"] = float(chi2)
    result["p_chi2"] = float(stats.chi2.sf(chi2, df=1))

    share = n01 / discordant
    share_low, share_high = clopper_pearson(n01, discordant, alpha)
    result["odds_ratio"] = _odds(share)
    result["odds_ratio_ci"] = [_odds(share_low), _odds(share_high)]

    # Wald interval for the paired risk difference (Agresti, 2002).
    total = len(correct_a)
    standard_error = (
        math.sqrt(max(n01 + n10 - (n01 - n10) ** 2 / total, 0.0)) / total
    )
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    diff = (n01 - n10) / total
    result["risk_difference_ci"] = [
        float(max(-1.0, diff - z_crit * standard_error)),
        float(min(1.0, diff + z_crit * standard_error)),
    ]
    result["significant"] = bool(result["p_exact"] < alpha)
    return result


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------
def paired_bootstrap(
    y_true: Sequence[int],
    values_a: Sequence[float],
    values_b: Sequence[float],
    metric: str = "accuracy",
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Percentile bootstrap CI for ``metric(B) - metric(A)``.

    Both models are evaluated on the same resampled images, preserving the
    correlation between their per-image performance. ``metric`` is either
    ``"accuracy"`` (values are predicted labels) or ``"auc"`` (values are
    P(PNEUMONIA) scores).
    """
    y_true = np.asarray(y_true)
    a = np.asarray(values_a)
    b = np.asarray(values_b)
    n = len(y_true)
    if n < 2:
        raise ValueError("at least two observations are required")
    if not (len(a) == len(b) == n):
        raise ValueError("y_true, values_a and values_b must have equal length")

    if metric == "accuracy":

        def score(truth: np.ndarray, values: np.ndarray) -> float:
            return float(np.mean(truth == values))

    elif metric == "auc":

        def score(truth: np.ndarray, values: np.ndarray) -> float:
            return auc_midrank(truth, values)

    else:
        raise ValueError("metric must be 'accuracy' or 'auc'")

    point = score(y_true, b) - score(y_true, a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    valid = 0
    for _ in range(n_resamples):
        index = rng.integers(0, n, size=n)
        truth = y_true[index]
        try:
            diffs[valid] = score(truth, b[index]) - score(truth, a[index])
        except ValueError:  # a resample containing one class has no AUC
            continue
        valid += 1

    diffs = diffs[:valid]
    low, high = np.percentile(diffs, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])
    p_two = min(
        1.0, 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    )
    return {
        "metric": metric,
        "point_estimate": float(point),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(p_two),
        "significant": bool(low > 0 or high < 0),
        "n_resamples": int(valid),
        "n_skipped": int(n_resamples - valid),
    }


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------
def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values; controls the family-wise error rate."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return p_values
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(
            running_max, (len(p_values) - rank) * p_values[index]
        )
        adjusted[index] = min(running_max, 1.0)
    return adjusted
