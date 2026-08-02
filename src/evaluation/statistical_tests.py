"""
Statistical comparison tools (spec section 17): paired fold-level bootstrap
confidence intervals, the Diebold-Mariano test, a Wilcoxon signed-rank
sensitivity check, and multiple-comparison correction. All operate on
paired per-fold (or per-observation) ERROR arrays from two configurations/
models evaluated on the exact same outer folds - callers are responsible
for that pairing (see `evaluation.incremental_value.paired_fold_errors`).
"""
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

from ..core.logging_utils import get_logger


def paired_bootstrap_ci(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Paired fold-level bootstrap CI for the mean difference (errors_a - errors_b).

    A negative mean difference means `a` has lower error than `b` on
    average (e.g. a - b where a is the "alternative" and b the "baseline"
    gives the sign convention used by spec section 10's improvement formula).

    Args:
        errors_a, errors_b: Paired per-fold (or per-observation) absolute
            errors from two configurations/models evaluated on identical
            outer folds - must be the same length and index-aligned.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95 for a 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with mean_diff, ci_low, ci_high, n_pairs.
    """
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    if len(errors_a) != len(errors_b):
        raise ValueError(
            f"paired_bootstrap_ci: errors_a and errors_b must be the same "
            f"length (paired folds), got {len(errors_a)} and {len(errors_b)}"
        )
    n = len(errors_a)
    if n == 0:
        raise ValueError("paired_bootstrap_ci: no paired observations supplied")

    diffs = errors_a - errors_b
    rng = np.random.RandomState(seed)

    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = diffs[idx].mean()

    alpha = 1 - ci
    ci_low = float(np.quantile(boot_means, alpha / 2))
    ci_high = float(np.quantile(boot_means, 1 - alpha / 2))

    return {
        'mean_diff': float(diffs.mean()),
        'ci_low': ci_low,
        'ci_high': ci_high,
        'ci_level': ci,
        'n_pairs': n,
    }


def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
    loss: str = 'absolute',
) -> Dict[str, float]:
    """
    Diebold-Mariano test for equal predictive accuracy between two paired
    forecast error series (Diebold & Mariano, 1995), with the standard
    small-sample/autocorrelation-robust variance adjustment (Harvey,
    Leybourne & Newbold 1997 correction).

    H0: the two forecasts have equal expected loss. A significant (e.g.
    p < 0.05) negative DM statistic means `a` has significantly LOWER loss
    than `b`; positive means `a` has significantly HIGHER loss.

    Args:
        errors_a, errors_b: Paired per-fold forecast errors (actual -
            predicted), NOT already absolute-valued - the loss function
            below applies the transform.
        h: Forecast horizon (used for the Newey-West-style autocovariance
            truncation lag, h-1, and the small-sample correction).
        loss: 'absolute' (|e|) or 'squared' (e^2).

    Returns:
        Dict with dm_statistic, p_value, n_obs.
    """
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    if len(errors_a) != len(errors_b):
        raise ValueError("diebold_mariano_test: errors_a and errors_b must be the same length")
    n = len(errors_a)
    if n < 2:
        raise ValueError("diebold_mariano_test: need at least 2 paired observations")

    if loss == 'absolute':
        loss_a, loss_b = np.abs(errors_a), np.abs(errors_b)
    elif loss == 'squared':
        loss_a, loss_b = errors_a ** 2, errors_b ** 2
    else:
        raise ValueError(f"diebold_mariano_test: unknown loss '{loss}', expected 'absolute' or 'squared'")

    d = loss_a - loss_b
    d_mean = d.mean()

    # Autocovariance-based long-run variance estimate (truncated at lag h-1).
    max_lag = max(h - 1, 0)
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        cov = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
        var_d += 2 * cov

    if var_d <= 0 or not np.isfinite(var_d):
        return {'dm_statistic': float('nan'), 'p_value': float('nan'), 'n_obs': n}

    dm_stat = d_mean / np.sqrt(var_d / n)

    # Harvey-Leybourne-Newbold small-sample correction.
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = dm_stat * hln_factor

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_corrected), df=n - 1))

    return {
        'dm_statistic': float(dm_stat_corrected),
        'p_value': float(p_value),
        'n_obs': n,
    }


def wilcoxon_signed_rank_test(errors_a: np.ndarray, errors_b: np.ndarray) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test (spec section 17 sensitivity check) on paired
    absolute errors - a non-parametric alternative to the DM test, less
    sensitive to outliers/non-normality, appropriate given how few outer
    folds Panel 2 has.
    """
    logger = get_logger()
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    if len(errors_a) != len(errors_b):
        raise ValueError("wilcoxon_signed_rank_test: errors_a and errors_b must be the same length")

    diffs = np.abs(errors_a) - np.abs(errors_b)
    if np.allclose(diffs, 0):
        return {'statistic': 0.0, 'p_value': 1.0, 'n_pairs': len(diffs)}

    try:
        statistic, p_value = stats.wilcoxon(diffs)
    except ValueError as e:
        logger.warning(f"wilcoxon_signed_rank_test: {e}")
        return {'statistic': float('nan'), 'p_value': float('nan'), 'n_pairs': len(diffs)}

    return {'statistic': float(statistic), 'p_value': float(p_value), 'n_pairs': len(diffs)}


def bonferroni_correction(p_values: List[float]) -> List[float]:
    """Bonferroni multiple-comparison correction: adjusted_p = min(1, p * m)."""
    m = len(p_values)
    return [min(1.0, p * m) if np.isfinite(p) else p for p in p_values]


def benjamini_hochberg_correction(p_values: List[float]) -> List[float]:
    """
    Benjamini-Hochberg FDR correction, less conservative than Bonferroni -
    offered as the alternative multiple-comparison correction spec section
    17 asks for ("correct for multiple comparisons when necessary").
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    finite_mask = np.isfinite(p_values)
    adjusted = np.full(n, np.nan)

    finite_p = p_values[finite_mask]
    if len(finite_p) == 0:
        return adjusted.tolist()

    order = np.argsort(finite_p)
    ranked = finite_p[order]
    m = len(ranked)
    bh = ranked * m / (np.arange(1, m + 1))
    # Enforce monotonicity (BH step-up procedure).
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0, 1)

    result_finite = np.empty(m)
    result_finite[order] = bh
    adjusted[finite_mask] = result_finite
    return adjusted.tolist()
