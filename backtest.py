"""
Hunter V22 — Backtesting Module
=================================
Fetches extended historical data (up to 4300+ candles via pagination)
from Binance and simulates the V22 strategy with realistic constraints:
- Fees & Slippage
- Walk-forward calculation (avoiding look-ahead bias)
- Kelly Criterion evaluation
- Persistent position simulation
- V22: Sharpe Ratio, Max Drawdown, extended data support

Usage:
    python backtest.py --symbol BTCUSDT --timeframe 1h --limit 4300
"""

import argparse
import logging
import math
import os
import sqlite3
import urllib.request
import urllib.error
import json
from datetime import datetime
from typing import List, Dict

from config import (
    ADX_PERIOD, ATR_PERIOD, BB_PERIOD, RSI_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL, SR_LOOKBACK, VWAP_BARS,
    INITIAL_BALANCE_USD, TAKER_FEE, SLIPPAGE,
    ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
)
from analysis import (
    compute_adx, compute_atr, compute_bollinger, compute_macd,
    compute_rsi, compute_rsi_series, compute_support_resistance,
    compute_vwap, detect_divergence, get_market_regime,
    compute_rsi_slope, compute_stoch_rsi, generate_signal,
    compute_composite_score, MarketState
)
from execution import PaperTrader
from ml import MLFilter
from report import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-15s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("hunter.backtest")

# ── Binance always uses mainnet for historical data ──────────────────────
MAINNET_URL = "https://fapi.binance.com"


def fetch_historical_ohlcv(symbol: str, interval: str, limit: int = 1000):
    """
    Fetch extended historical data via pagination.
    Binance API limit per request = 1500. For >1500 bars, we paginate backwards.
    """
    all_data = []
    end_time = None
    remaining = limit
    
    while remaining > 0:
        batch_size = min(remaining, 1500)
        url = f"{MAINNET_URL}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={batch_size}"
        if end_time:
            url += f"&endTime={end_time}"
        
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode())
        except Exception as e:
            logger.error("Failed to fetch history (remaining=%d): %s", remaining, e)
            break
        
        if not data:
            break
            
        all_data = data + all_data  # prepend older data
        remaining -= len(data)
        
        # Next batch ends 1ms before earliest fetched candle
        end_time = int(data[0][0]) - 1
        
        logger.info("📥 Fetched %d candles (total: %d, remaining: %d)", 
                     len(data), len(all_data), max(0, remaining))
    
    highs, lows, closes, volumes = [], [], [], []
    for kline in all_data:
        highs.append(float(kline[2]))
        lows.append(float(kline[3]))
        closes.append(float(kline[4]))
        volumes.append(float(kline[5]))

    # Never use the last (currently open) candle
    return highs[:-1], lows[:-1], closes[:-1], volumes[:-1]


def compute_sharpe_ratio(returns: List[float], periods_per_year: float = 8760) -> float:
    """
    Annualized Sharpe Ratio from per-trade returns.
    periods_per_year = 8760 for 1h candles (365 * 24).
    """
    if len(returns) < 2:
        return 0.0
    avg = sum(returns) / len(returns)
    variance = sum((r - avg) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance) if variance > 0 else 1e-9
    # Annualization factor
    trades_per_year = periods_per_year / (len(returns) / max(1, len(returns)))
    return (avg / std) * math.sqrt(min(trades_per_year, periods_per_year))


def run_backtest(symbol: str, timeframe: str, limit: int = 1000, train_ml: bool = False):
    logger.info("Starting V22 Backtest for %s on %s timeframe (limit=%d)", symbol, timeframe, limit)
    highs, lows, closes, volumes = fetch_historical_ohlcv(symbol, timeframe, limit)
    
    n_bars = len(closes)
    logger.info("Fetched %d closed candles", n_bars)
    
    if n_bars < 200:
        logger.error("Not enough data to backtest (need 200+ bars)")
        return

    # Initialize a fresh DB and trader for testing
    test_db = "backtest.db"
    if os.path.exists(test_db):
        os.unlink(test_db)
        
    trader = PaperTrader(db_path=test_db)
    
    warmup = 200
    
    wins = 0
    losses = 0
    total_pnl = 0.0
    trade_returns = []       # V22: per-trade returns for Sharpe
    equity_curve = []        # V22: balance after each bar for max drawdown

    # ML training data collection
    ml_filter = MLFilter()
    ml_features = []
    ml_outcomes = []
    LOOKAHEAD = 12  # bars to check outcome

    logger.info("Simulating %d bars...", n_bars - warmup)

    for i in range(warmup, n_bars):
        h = highs[:i]
        l = lows[:i]
        c = closes[:i]
        v = volumes[:i]
        
        current_price = c[-1]
        
        adx = compute_adx(h, l, c, ADX_PERIOD)
        regime = get_market_regime(adx)
        
        rsi = compute_rsi(c, RSI_PERIOD)
        lower, middle, upper = compute_bollinger(c, BB_PERIOD)
        macd_line, signal_line, histogram = compute_macd(c, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        atr = compute_atr(h, l, c, ATR_PERIOD)
        
        vwap_start = max(0, len(c) - VWAP_BARS)
        vwap = compute_vwap(h[vwap_start:], l[vwap_start:], c[vwap_start:], v[vwap_start:])
        vwap_diff_pct = ((current_price - vwap) / vwap * 100) if vwap > 0 else 0
        
        rsi_slope = compute_rsi_slope(c)
        stoch_rsi = compute_stoch_rsi(c)
        
        rsi_series = compute_rsi_series(c, RSI_PERIOD)
        divergence = detect_divergence(c, rsi_series, adx_value=adx)
        
        bb_range = upper - lower
        bb_position = (current_price - lower) / bb_range if bb_range > 0 else 0.5
        
        # S/R detection for near_resistance
        supports, resistances = compute_support_resistance(h, l, SR_LOOKBACK)
        near_resistance = False
        if resistances:
            nearest_r = min(resistances, key=lambda r: abs(r - current_price))
            if abs(nearest_r - current_price) / current_price * 100 < 0.5:
                near_resistance = True

        # Missing external data approximations for backtest
        ls_ratio = 1.0
        whale_vol = 0.0
        funding_rate = 0.0
        oi_delta = 0.0
        mtf_agreement = 0.0
        social_score = 0.0
        btc_correlation = 1.0
        btc_dominance = 50.0
        volume_confirm = True

        pos = trader.get_position(symbol)
        atr_pct = (atr / current_price * 100) if current_price else 0.0
        
        market_state = MarketState(
            current_price=current_price,
            rsi=rsi,
            ls_ratio=ls_ratio,
            whale_net_vol=whale_vol,
            regime=regime,
            social_score=social_score,
            macd_histogram=histogram,
            bb_position=bb_position,
            vwap_diff_pct=vwap_diff_pct,
            divergence=divergence,
            funding_rate=funding_rate,
            open_interest_delta=oi_delta,
            liq_imbalance=0.0,
            atr_pct=atr_pct,
            rsi_slope=rsi_slope,
            stoch_rsi=stoch_rsi,
            mtf_agreement=mtf_agreement,
            volume_confirm=volume_confirm,
            near_resistance=near_resistance,
            btc_correlation=btc_correlation,
            btc_dominance=btc_dominance
        )

        signal_dict = generate_signal(market_state, current_position=pos, use_composite=True, detailed=True)
        signal = signal_dict.get("action", "HOLD")
        
        if signal != "HOLD" or atr > 0:
            result = trader.execute_trade(signal, current_price, symbol, atr=atr)
            action = result.get("action", "")
            if action not in ["", "HOLD"]:
                ReportGenerator.print_cycle_report(symbol, market_state, signal_dict, result)
            if "CLOSED" in action:
                p = result.get("pnl", 0)
                total_pnl += p
                trade_returns.append(p)
                if p > 0:
                    wins += 1
                else:
                    losses += 1
                
                logger.info("Bar %d | Price: $%.2f | %s | PnL: $%.2f | Balance: $%.2f",
                            i, current_price, action, p, trader.balance)

        equity_curve.append(trader.balance)

    # Clean up open positions at end of backtest
    if trader.has_position(symbol):
        pos = trader.get_position(symbol)
        pnl = PaperTrader.simulate_pnl(pos["entry"], closes[-1], pos["size_usd"], pos["side"])
        total_pnl += pnl
        trade_returns.append(pnl)
        if pnl > 0: wins += 1
        else: losses += 1
        trader.balance += pos["size_usd"] + pnl
        equity_curve.append(trader.balance)
        logger.info("End of Data | Closing Open %s | PnL: $%.2f | Balance: $%.2f", 
                    pos["side"], pnl, trader.balance)
        
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    net_return = ((trader.balance - INITIAL_BALANCE_USD) / INITIAL_BALANCE_USD) * 100
    
    # V22: Max Drawdown
    peak = INITIAL_BALANCE_USD
    max_dd = 0.0
    for bal in equity_curve:
        if bal > peak:
            peak = bal
        dd = (peak - bal) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    # V22: Sharpe Ratio
    sharpe = compute_sharpe_ratio(trade_returns)
    
    # V22: Profit Factor
    gross_profit = sum(r for r in trade_returns if r > 0)
    gross_loss = abs(sum(r for r in trade_returns if r < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    # V22: Average trade
    avg_trade = total_pnl / total_trades if total_trades > 0 else 0
    avg_win = gross_profit / wins if wins > 0 else 0
    avg_loss = gross_loss / losses if losses > 0 else 0
    
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS (V22) - %s %s", symbol, timeframe)
    logger.info("=" * 60)
    logger.info("Bars Analyzed  : %d", n_bars - warmup)
    logger.info("Total Trades   : %d", total_trades)
    logger.info("Wins / Losses  : %d / %d", wins, losses)
    logger.info("Win Rate       : %.1f%%", win_rate)
    logger.info("Net PnL        : $%.2f (%.2f%%)", total_pnl, net_return)
    logger.info("Final Balance  : $%.2f", trader.balance)
    logger.info("─" * 60)
    logger.info("Avg Trade      : $%.2f", avg_trade)
    logger.info("Avg Win        : $%.2f", avg_win)
    logger.info("Avg Loss       : $%.2f", avg_loss)
    logger.info("Profit Factor  : %.2f", profit_factor)
    logger.info("Sharpe Ratio   : %.2f", sharpe)
    logger.info("Max Drawdown   : %.2f%%", max_dd)
    logger.info("=" * 60)

    # ── ML Training Phase ──────────────────────────────────────
    if train_ml:
        logger.info("\n🧠 ML TRAINING MODE: Collecting features from historical data...")
        ml_features = []
        ml_outcomes = []
        
        for i in range(warmup, n_bars - LOOKAHEAD):
            h = highs[:i]
            l = lows[:i]
            c = closes[:i]
            v = volumes[:i]
            current_price = c[-1]
            
            adx = compute_adx(h, l, c, ADX_PERIOD)
            regime = get_market_regime(adx)
            rsi = compute_rsi(c, RSI_PERIOD)
            lower, middle, upper = compute_bollinger(c, BB_PERIOD)
            _, _, histogram = compute_macd(c, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            atr = compute_atr(h, l, c, ATR_PERIOD)
            
            vwap_start = max(0, len(c) - VWAP_BARS)
            vwap = compute_vwap(h[vwap_start:], l[vwap_start:], c[vwap_start:], v[vwap_start:])
            vwap_diff_pct = ((current_price - vwap) / vwap * 100) if vwap > 0 else 0
            
            rsi_slope = compute_rsi_slope(c)
            stoch_rsi = compute_stoch_rsi(c)
            
            bb_range = upper - lower
            bb_position = (current_price - lower) / bb_range if bb_range > 0 else 0.5
            atr_pct = (atr / current_price * 100) if current_price else 0.0
            
            state = MarketState(
                current_price=current_price, rsi=rsi, ls_ratio=1.0,
                whale_net_vol=0.0, regime=regime, social_score=0.0,
                macd_histogram=histogram, bb_position=bb_position,
                vwap_diff_pct=vwap_diff_pct, divergence="NONE",
                funding_rate=0.0, open_interest_delta=0.0,
                liq_imbalance=0.0, atr_pct=atr_pct, rsi_slope=rsi_slope,
                stoch_rsi=stoch_rsi, mtf_agreement=0.0, volume_confirm=True,
                near_resistance=False, btc_correlation=1.0, btc_dominance=50.0,
            )
            
            score = compute_composite_score(state)
            
            # Only collect features when signal would fire
            if abs(score) > 0.15:  # Lower threshold for training data
                features = ml_filter.extract_features(state, composite_score=score,
                                                      closes=c, volumes=v, hour=i % 24)
                if features is not None:
                    # Evaluate outcome: did price move favorably in next LOOKAHEAD bars?
                    future_prices = closes[i:i + LOOKAHEAD]
                    if score > 0:  # BUY signal
                        max_future = max(future_prices)
                        move_pct = (max_future - current_price) / current_price * 100
                        outcome = 1 if move_pct > (atr_pct * ATR_SL_MULTIPLIER) else 0
                    else:  # SHORT signal
                        min_future = min(future_prices)
                        move_pct = (current_price - min_future) / current_price * 100
                        outcome = 1 if move_pct > (atr_pct * ATR_SL_MULTIPLIER) else 0
                    
                    ml_features.append(features)
                    ml_outcomes.append(outcome)
        
        if len(ml_features) >= 20:
            metrics = ml_filter.train(ml_features, ml_outcomes)
            ml_filter.save()
            logger.info("\n✅ ML Model Trained!")
            logger.info("   Samples: %d | Positive rate: %.1f%% | CV Accuracy: %.1f%%",
                        metrics['samples'], metrics['positive_rate'] * 100,
                        metrics['cv_accuracy'] * 100)
        else:
            logger.warning("⚠️ Only %d samples collected (need 20+). Model not trained.", len(ml_features))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hunter V22 Backtest")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--limit", type=int, default=4300, help="Number of candles to fetch (default: 4300 ≈ 6 months)")
    parser.add_argument("--train-ml", action="store_true", help="Train ML model on historical data")
    args = parser.parse_args()
    
    run_backtest(args.symbol, args.timeframe, args.limit, train_ml=args.train_ml)
