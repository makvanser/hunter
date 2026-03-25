"""
Hunter V34 — Walk-Forward Optimizer
=====================================
Offline parameter optimization engine that periodically retrains
strategy parameters on historical data using out-of-sample validation.

Usage:
    python optimizer.py                    # Run optimization with defaults (30d window)
    python optimizer.py --window 14        # 14-day training window
    python optimizer.py --dry-run          # Print optimal params without saving

Algorithm:
    1. Fetch last N days of OHLCV data for top traded pairs
    2. Split into 70% train / 30% test (walk-forward)
    3. Grid search over ATR_SL_MULT, ATR_TP_MULT, RSI thresholds
    4. Select parameter set with highest Sharpe on OOS (test) data
    5. Save optimal parameters to state.db for live bot to consume
"""

import argparse
import logging
import sqlite3
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timezone

from config import (
    DB_PATH, ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
    RSI_PERIOD, ATR_PERIOD, BB_PERIOD,
    TAKER_FEE, SLIPPAGE, TRADE_SIZE_USD,
)
from analysis import (
    compute_rsi, compute_atr, compute_bollinger,
    compute_macd, compute_adx, get_market_regime,
)

logger = logging.getLogger("hunter.optimizer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s")

# ─────────────────────────────────────────────────────────────
# Parameter Grid
# ─────────────────────────────────────────────────────────────
PARAM_GRID = {
    "atr_sl_mult":    [1.0, 1.25, 1.5, 2.0, 2.5],
    "atr_tp_mult":    [1.5, 2.0, 2.5, 3.0, 4.0],
    "rsi_oversold":   [25, 30, 35, 40],
    "rsi_overbought": [60, 65, 70, 75],
}


def _generate_combos(grid: Dict[str, List]) -> List[Dict[str, float]]:
    """Generate all combinations from parameter grid."""
    keys = list(grid.keys())
    combos = [{}]
    for key in keys:
        new_combos = []
        for combo in combos:
            for val in grid[key]:
                c = combo.copy()
                c[key] = val
                new_combos.append(c)
        combos = new_combos
    return combos


# ─────────────────────────────────────────────────────────────
# Simple Backtester
# ─────────────────────────────────────────────────────────────
def backtest(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
    params: Dict[str, float],
    trade_size: float = TRADE_SIZE_USD,
) -> Dict[str, Any]:
    """
    Run a simplified backtest over OHLCV data with given parameters.
    Returns performance metrics: total_pnl, sharpe, win_rate, n_trades.
    """
    atr_sl = params["atr_sl_mult"]
    atr_tp = params["atr_tp_mult"]
    rsi_os = params["rsi_oversold"]
    rsi_ob = params["rsi_overbought"]

    trades = []
    position = None  # {"side", "entry", "sl", "tp"}
    
    # Need at least 50 bars for indicators to warm up
    warmup = 50
    if len(closes) < warmup + 10:
        return {"total_pnl": 0, "sharpe": 0, "win_rate": 0, "n_trades": 0}

    for i in range(warmup, len(closes)):
        price = closes[i]
        high = highs[i]
        low = lows[i]

        # Compute indicators on lookback window
        window_closes = closes[:i+1]
        window_highs = highs[:i+1]
        window_lows = lows[:i+1]

        rsi = compute_rsi(window_closes, RSI_PERIOD)
        atr = compute_atr(window_highs, window_lows, window_closes, ATR_PERIOD)

        # Check existing position SL/TP
        if position:
            if position["side"] == "BUY":
                if low <= position["sl"]:
                    pnl = (position["sl"] - position["entry"]) / position["entry"] * trade_size
                    pnl -= (TAKER_FEE + SLIPPAGE) * 2 * trade_size
                    trades.append(pnl)
                    position = None
                    continue
                if high >= position["tp"]:
                    pnl = (position["tp"] - position["entry"]) / position["entry"] * trade_size
                    pnl -= (TAKER_FEE + SLIPPAGE) * 2 * trade_size
                    trades.append(pnl)
                    position = None
                    continue
            elif position["side"] == "SELL":
                if high >= position["sl"]:
                    pnl = (position["entry"] - position["sl"]) / position["entry"] * trade_size
                    pnl -= (TAKER_FEE + SLIPPAGE) * 2 * trade_size
                    trades.append(pnl)
                    position = None
                    continue
                if low <= position["tp"]:
                    pnl = (position["entry"] - position["tp"]) / position["entry"] * trade_size
                    pnl -= (TAKER_FEE + SLIPPAGE) * 2 * trade_size
                    trades.append(pnl)
                    position = None
                    continue

        # Entry signals (simplified)
        if position is None and atr > 0:
            if rsi < rsi_os:
                position = {
                    "side": "BUY",
                    "entry": price,
                    "sl": price - atr * atr_sl,
                    "tp": price + atr * atr_tp,
                }
            elif rsi > rsi_ob:
                position = {
                    "side": "SELL",
                    "entry": price,
                    "sl": price + atr * atr_sl,
                    "tp": price - atr * atr_tp,
                }

    # Close any open position at market
    if position:
        final = closes[-1]
        if position["side"] == "BUY":
            pnl = (final - position["entry"]) / position["entry"] * trade_size
        else:
            pnl = (position["entry"] - final) / position["entry"] * trade_size
        pnl -= (TAKER_FEE + SLIPPAGE) * 2 * trade_size
        trades.append(pnl)

    # Metrics
    if not trades:
        return {"total_pnl": 0, "sharpe": 0, "win_rate": 0, "n_trades": 0}

    total_pnl = sum(trades)
    wins = sum(1 for t in trades if t > 0)
    win_rate = wins / len(trades)
    
    # Sharpe Ratio (annualized, assuming 15min bars)
    returns = np.array(trades) / trade_size
    sharpe = 0.0
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365 * 96)  # 96 bars per day (15min)

    return {
        "total_pnl": round(total_pnl, 2),
        "sharpe": round(sharpe, 2),
        "win_rate": round(win_rate * 100, 1),
        "n_trades": len(trades),
    }


# ─────────────────────────────────────────────────────────────
# Walk-Forward Engine
# ─────────────────────────────────────────────────────────────
def walk_forward_optimize(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
    train_pct: float = 0.7,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Split data into train/test, grid search on train, validate on test.
    Returns (best_params, test_metrics).
    """
    split = int(len(closes) * train_pct)
    
    train_h, test_h = highs[:split], highs[split:]
    train_l, test_l = lows[:split], lows[split:]
    train_c, test_c = closes[:split], closes[split:]
    train_v, test_v = volumes[:split], volumes[split:]

    combos = _generate_combos(PARAM_GRID)
    logger.info("🔬 Running %d parameter combinations on %d training bars...", len(combos), split)

    best_sharpe = -999
    best_params = None
    best_train_metrics = None

    for combo in combos:
        metrics = backtest(train_h, train_l, train_c, train_v, combo)
        if metrics["sharpe"] > best_sharpe and metrics["n_trades"] >= 5:
            best_sharpe = metrics["sharpe"]
            best_params = combo
            best_train_metrics = metrics

    if best_params is None:
        logger.warning("⚠️ No valid parameter combination found!")
        return {}, {}

    # Validate on out-of-sample data
    test_metrics = backtest(test_h, test_l, test_c, test_v, best_params)

    logger.info("📊 TRAIN: Sharpe=%.2f PnL=$%.2f WR=%.1f%% Trades=%d",
                best_train_metrics["sharpe"], best_train_metrics["total_pnl"],
                best_train_metrics["win_rate"], best_train_metrics["n_trades"])
    logger.info("📊 TEST (OOS): Sharpe=%.2f PnL=$%.2f WR=%.1f%% Trades=%d",
                test_metrics["sharpe"], test_metrics["total_pnl"],
                test_metrics["win_rate"], test_metrics["n_trades"])
    logger.info("🏆 Best Params: %s", best_params)

    return best_params, test_metrics


def save_optimal_params(params: Dict[str, float], db_path: str = DB_PATH) -> None:
    """Save optimized parameters to state.db for live bot consumption."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute("CREATE TABLE IF NOT EXISTS optimized_params (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    
    now = datetime.now(timezone.utc).isoformat()
    for key, val in params.items():
        c.execute(
            "INSERT OR REPLACE INTO optimized_params (key, value, updated_at) VALUES (?, ?, ?)",
            (key, str(val), now)
        )
    
    # Also store as a single JSON blob for easy loading
    c.execute(
        "INSERT OR REPLACE INTO optimized_params (key, value, updated_at) VALUES (?, ?, ?)",
        ("_all_params_json", json.dumps(params), now)
    )
    
    conn.commit()
    conn.close()
    logger.info("💾 Saved %d optimized parameters to %s", len(params), db_path)


def load_optimal_params(db_path: str = DB_PATH) -> Dict[str, float]:
    """Load optimized parameters from state.db. Returns empty dict if none found."""
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT value FROM optimized_params WHERE key = '_all_params_json'")
        row = c.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
    except Exception:
        pass
    return {}


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────
async def run_optimization(window_days: int = 30, dry_run: bool = False):
    """Fetch data and run walk-forward optimization."""
    from provider import BinanceProvider
    
    logger.info("=" * 60)
    logger.info("  HUNTER V34 — Walk-Forward Optimizer")
    logger.info("  Window: %d days | Dry Run: %s", window_days, dry_run)
    logger.info("=" * 60)

    async with BinanceProvider() as provider:
        # Use BTCUSDT as the primary optimization target
        symbol = "BTCUSDT"
        interval = "15m"
        limit = min(window_days * 96, 1500)  # 96 bars per day at 15m

        highs, lows, closes, volumes = await provider.fetch_ohlcv(symbol, interval, limit)
        
        if len(closes) < 200:
            logger.error("❌ Insufficient data: only %d bars. Need at least 200.", len(closes))
            return

        logger.info("📥 Loaded %d bars for %s (%s)", len(closes), symbol, interval)

        best_params, test_metrics = walk_forward_optimize(highs, lows, closes, volumes)

        if not best_params:
            logger.error("❌ Optimization failed — no valid parameters found.")
            return

        if dry_run:
            logger.info("🔍 DRY RUN — Parameters NOT saved.")
            logger.info("   Would save: %s", best_params)
        else:
            save_optimal_params(best_params)
            logger.info("✅ Optimization complete. Parameters saved to database.")


if __name__ == "__main__":
    import asyncio
    
    parser = argparse.ArgumentParser(description="Hunter Walk-Forward Optimizer")
    parser.add_argument("--window", type=int, default=30, help="Training window in days (default: 30)")
    parser.add_argument("--dry-run", action="store_true", help="Print results without saving")
    args = parser.parse_args()

    asyncio.run(run_optimization(window_days=args.window, dry_run=args.dry_run))
