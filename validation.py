"""
Hunter V36 — Statistical Validation Engine
=============================================
Adapted from HKUDS/Vibe-Trading backtest/validation.py (MIT License).

Provides three statistical tests to validate strategy robustness:
  1. Monte Carlo permutation test — is Sharpe better than random trade ordering?
  2. Bootstrap Sharpe CI — 95% confidence interval for the Sharpe ratio
  3. Walk-Forward consistency — are results consistent across time windows?

Zero external dependencies beyond numpy.

Usage:
    from validation import validate_strategy
    result = validate_strategy(trades, equity_curve, initial_capital)
    if result["is_significant"]:
        save_optimal_params(...)
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger("hunter.validation")

# ─────────────────────────────────────────────────────────────
# Monte Carlo Permutation Test
# ─────────────────────────────────────────────────────────────

def monte_carlo_test(
    trade_pnls: List[float],
    initial_capital: float,
    n_simulations: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Shuffle trade PnL order to test path significance.

    Null hypothesis: the observed Sharpe / max-drawdown is no better than
    a random ordering of the same trades.

    Args:
        trade_pnls: List of PnL values per trade (e.g. [1.2, -0.5, 0.8, ...])
        initial_capital: Starting capital.
        n_simulations: Number of random permutations.
        seed: Random seed for reproducibility.

    Returns:
        Dict with actual_sharpe, p_value_sharpe, actual_max_dd,
        p_value_max_dd, simulated_sharpes (percentiles).
    """
    if len(trade_pnls) < 3:
        return {"error": "need at least 3 trades", "p_value_sharpe": 1.0}

    pnls = np.array(trade_pnls, dtype=np.float64)
    actual = _path_metrics(pnls, initial_capital)

    rng = np.random.default_rng(seed)
    sharpe_count = 0
    dd_count = 0
    sim_sharpes = []

    for _ in range(n_simulations):
        shuffled = rng.permutation(pnls)
        sim = _path_metrics(shuffled, initial_capital)
        sim_sharpes.append(sim["sharpe"])
        if sim["sharpe"] >= actual["sharpe"]:
            sharpe_count += 1
        if sim["max_dd"] >= actual["max_dd"]:
            dd_count += 1

    sim_arr = np.array(sim_sharpes)
    return {
        "actual_sharpe": round(actual["sharpe"], 4),
        "actual_max_dd": round(actual["max_dd"], 4),
        "p_value_sharpe": round(sharpe_count / n_simulations, 4),
        "p_value_max_dd": round(dd_count / n_simulations, 4),
        "simulated_sharpe_mean": round(float(sim_arr.mean()), 4),
        "simulated_sharpe_std": round(float(sim_arr.std()), 4),
        "simulated_sharpe_p5": round(float(np.percentile(sim_arr, 5)), 4),
        "simulated_sharpe_p95": round(float(np.percentile(sim_arr, 95)), 4),
        "n_simulations": n_simulations,
        "n_trades": len(trade_pnls),
    }


def _path_metrics(pnls: np.ndarray, initial_capital: float) -> Dict[str, float]:
    """Compute Sharpe and max drawdown from a PnL sequence."""
    equity = initial_capital + np.cumsum(pnls)
    if len(equity) < 2:
        return {"sharpe": 0.0, "max_dd": 0.0}
    returns = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1.0)
    std = returns.std()
    sharpe = float(returns.mean() / (std + 1e-10) * np.sqrt(365 * 96))  # 15m bars
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(dd.min())
    return {"sharpe": sharpe, "max_dd": max_dd}


# ─────────────────────────────────────────────────────────────
# Bootstrap Sharpe Confidence Interval
# ─────────────────────────────────────────────────────────────

def bootstrap_sharpe_ci(
    trade_pnls: List[float],
    trade_size: float,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    bars_per_year: int = 365 * 96,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Resample trade returns to estimate Sharpe confidence interval.

    Args:
        trade_pnls: List of PnL values per trade.
        trade_size: Trade size for normalizing returns.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        bars_per_year: Annualisation factor.
        seed: Random seed.

    Returns:
        Dict with observed_sharpe, ci_lower, ci_upper, median_sharpe,
        prob_positive (fraction of samples with Sharpe > 0).
    """
    returns = np.array(trade_pnls, dtype=np.float64) / trade_size
    if len(returns) < 5:
        return {"error": "need at least 5 trades"}

    observed = _sharpe(returns, bars_per_year)

    rng = np.random.default_rng(seed)
    boot_sharpes = []
    for _ in range(n_bootstrap):
        sample = rng.choice(returns, size=len(returns), replace=True)
        boot_sharpes.append(_sharpe(sample, bars_per_year))

    arr = np.array(boot_sharpes)
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(arr, alpha * 100))
    upper = float(np.percentile(arr, (1 - alpha) * 100))
    prob_pos = float(np.mean(arr > 0))

    return {
        "observed_sharpe": round(observed, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "median_sharpe": round(float(np.median(arr)), 4),
        "prob_positive": round(prob_pos, 4),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


def _sharpe(returns: np.ndarray, bars_per_year: int = 365 * 96) -> float:
    """Annualized Sharpe ratio."""
    std = returns.std()
    return float(returns.mean() / (std + 1e-10) * np.sqrt(bars_per_year))


# ─────────────────────────────────────────────────────────────
# Walk-Forward Consistency
# ─────────────────────────────────────────────────────────────

def walk_forward_consistency(
    trade_pnls: List[float],
    n_windows: int = 5,
) -> Dict[str, Any]:
    """
    Split trades into sequential windows, check consistency.

    Args:
        trade_pnls: List of PnL values per trade.
        n_windows: Number of non-overlapping windows.

    Returns:
        Dict with per_window stats, consistency metrics.
    """
    if len(trade_pnls) < n_windows * 2:
        return {"error": f"need at least {n_windows * 2} trades for {n_windows} windows"}

    pnls = np.array(trade_pnls, dtype=np.float64)
    window_size = len(pnls) // n_windows
    windows = []

    for i in range(n_windows):
        start = i * window_size
        end = (i + 1) * window_size if i < n_windows - 1 else len(pnls)
        win_pnls = pnls[start:end]

        total_pnl = float(win_pnls.sum())
        wins = int((win_pnls > 0).sum())
        win_rate = wins / len(win_pnls) if len(win_pnls) > 0 else 0.0

        windows.append({
            "window": i + 1,
            "trades": len(win_pnls),
            "total_pnl": round(total_pnl, 4),
            "win_rate": round(win_rate, 4),
            "profitable": total_pnl > 0,
        })

    profitable_windows = sum(1 for w in windows if w["profitable"])
    pnl_list = [w["total_pnl"] for w in windows]

    return {
        "n_windows": n_windows,
        "windows": windows,
        "profitable_windows": profitable_windows,
        "consistency_rate": round(profitable_windows / n_windows, 4),
        "pnl_mean": round(float(np.mean(pnl_list)), 4),
        "pnl_std": round(float(np.std(pnl_list)), 4),
    }


# ─────────────────────────────────────────────────────────────
# Extended Backtest Metrics
# ─────────────────────────────────────────────────────────────

def compute_extended_metrics(trade_pnls: List[float]) -> Dict[str, Any]:
    """
    Compute extended metrics: profit factor, max consecutive loss,
    avg holding (here just avg trade PnL spread).

    Args:
        trade_pnls: List of PnL values per trade.

    Returns:
        Dict with profit_factor, max_consecutive_loss, avg_win, avg_loss.
    """
    if not trade_pnls:
        return {
            "profit_factor": 0.0,
            "max_consecutive_loss": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 1e-10
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else 0.0

    max_consec = 0
    cur_consec = 0
    for p in trade_pnls:
        if p < 0:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0

    return {
        "profit_factor": round(profit_factor, 4),
        "max_consecutive_loss": max_consec,
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "total_wins": len(wins),
        "total_losses": len(losses),
    }


# ─────────────────────────────────────────────────────────────
# Unified Validation Runner
# ─────────────────────────────────────────────────────────────

def validate_strategy(
    trade_pnls: List[float],
    initial_capital: float = 100.0,
    trade_size: float = 10.0,
    significance_level: float = 0.05,
    n_simulations: int = 1000,
) -> Dict[str, Any]:
    """
    Run all three validation tests and return combined result.

    Args:
        trade_pnls: List of PnL values per trade.
        initial_capital: Starting capital.
        trade_size: Trade size used for return normalization.
        significance_level: p-value threshold (default 0.05).
        n_simulations: Number of Monte Carlo / Bootstrap iterations.

    Returns:
        Dict with all validation results and is_significant flag.
    """
    result = {
        "n_trades": len(trade_pnls),
        "is_significant": False,
        "reason": "",
    }

    if len(trade_pnls) < 5:
        result["reason"] = f"Insufficient trades ({len(trade_pnls)} < 5)"
        logger.warning("⚠️ Validation SKIPPED: %s", result["reason"])
        return result

    # 1. Monte Carlo
    mc = monte_carlo_test(trade_pnls, initial_capital, n_simulations)
    result["monte_carlo"] = mc

    # 2. Bootstrap Sharpe CI
    bs = bootstrap_sharpe_ci(trade_pnls, trade_size, n_simulations)
    result["bootstrap_sharpe"] = bs

    # 3. Walk-Forward Consistency (only if enough trades)
    if len(trade_pnls) >= 10:
        n_win = min(5, len(trade_pnls) // 2)
        wf = walk_forward_consistency(trade_pnls, n_win)
        result["walk_forward"] = wf
    else:
        result["walk_forward"] = {"skipped": True, "reason": "< 10 trades"}

    # 4. Extended Metrics
    ext = compute_extended_metrics(trade_pnls)
    result["extended_metrics"] = ext

    # Significance decision
    p_val = mc.get("p_value_sharpe", 1.0)
    prob_pos = bs.get("prob_positive", 0.0)
    consistency = result.get("walk_forward", {}).get("consistency_rate", 0.0)

    if p_val <= significance_level and prob_pos >= 0.60:
        result["is_significant"] = True
        result["reason"] = (
            f"Monte Carlo p={p_val:.3f} (≤ {significance_level}), "
            f"P(Sharpe>0)={prob_pos:.1%}, "
            f"Consistency={consistency:.0%}"
        )
        logger.info("✅ VALIDATION PASSED: %s", result["reason"])
    else:
        reasons = []
        if p_val > significance_level:
            reasons.append(f"Monte Carlo p={p_val:.3f} (> {significance_level})")
        if prob_pos < 0.60:
            reasons.append(f"P(Sharpe>0)={prob_pos:.1%} (< 60%)")
        result["reason"] = " | ".join(reasons)
        logger.warning("❌ VALIDATION FAILED: %s", result["reason"])

    return result
