"""Statistical tests for comparing classifiers and computing confidence intervals."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def wilson_ci(n_correct: int, n_total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for accuracy."""
    if n_total == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = n_correct / n_total
    denominator = 1 + z**2 / n_total
    centre = (p_hat + z**2 / (2 * n_total)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for AUC."""
    rng = np.random.default_rng(random_state)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except ValueError:
            continue
    alpha = (1 - confidence) / 2
    return float(np.percentile(aucs, 100 * alpha)), float(np.percentile(aucs, 100 * (1 - alpha)))


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> tuple[float, float]:
    """McNemar's test for comparing two classifiers.

    Returns (statistic, p_value).
    """
    a_right_b_wrong = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
    a_wrong_b_right = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))
    n = a_right_b_wrong + a_wrong_b_right
    if n == 0:
        return 0.0, 1.0
    # Mid-p McNemar (continuity correction)
    statistic = (abs(a_right_b_wrong - a_wrong_b_right) - 1) ** 2 / n
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    return float(statistic), float(p_value)


def _auc_variance(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Hanley-McNeil variance estimate for AUC."""
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    auc = roc_auc_score(y_true, y_prob)
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return var


def delong_auc_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
) -> tuple[float, float]:
    """Approximate DeLong's test for comparing two AUC values.

    Returns (z_statistic, p_value).
    """
    auc_a = roc_auc_score(y_true, y_prob_a)
    auc_b = roc_auc_score(y_true, y_prob_b)
    var_a = _auc_variance(y_true, y_prob_a)
    var_b = _auc_variance(y_true, y_prob_b)
    se = np.sqrt(var_a + var_b)
    if se == 0:
        return 0.0, 1.0
    z = (auc_a - auc_b) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p_value)
