"""
Hunter V26 — Continuous Learning Module
========================================
Two training modes:
  1. Walk-Forward from Binance API (historical OHLCV)
  2. Journal-Based from signal_journal.db (real signals with outcomes)

The journal mode is preferred when enough data exists, as it uses
actual market conditions and outcomes rather than synthetic backtest.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timezone
from typing import List

import aiohttp

from ml import MLFilter
from config import (
    ADX_PERIOD, ATR_PERIOD, BB_PERIOD, RSI_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL, SR_LOOKBACK, VWAP_BARS,
    ATR_SL_MULTIPLIER, TIMEFRAME
)
from analysis import (
    compute_adx, compute_atr, compute_bollinger, compute_macd,
    compute_rsi, compute_rsi_series, compute_support_resistance,
    compute_vwap, detect_divergence, get_market_regime,
    compute_rsi_slope, compute_stoch_rsi, compute_composite_score,
    MarketState
)

logger = logging.getLogger("hunter.learner")


class ContinuousLearner:
    def __init__(self, ml_filter: MLFilter):
        self.ml_filter = ml_filter

    # ── Data Fetching ──────────────────────────────────────────

    async def fetch_historical_ohlcv_async(self, symbol: str, interval: str, limit: int = 1344):
        """Async paginated fetcher for large candle history."""
        all_data = []
        end_time = None
        remaining = limit

        async with aiohttp.ClientSession() as session:
            while remaining > 0:
                batch_size = min(remaining, 1500)
                url = (
                    f"https://fapi.binance.com/fapi/v1/klines"
                    f"?symbol={symbol}&interval={interval}&limit={batch_size}"
                )
                if end_time:
                    url += f"&endTime={end_time}"

                try:
                    async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as resp:
                        if resp.status != 200:
                            logger.error("Fetch failed: HTTP %s", resp.status)
                            break
                        data = await resp.json()
                except Exception as e:
                    logger.error("Async fetch error: %s", e)
                    break

                if not data:
                    break

                all_data = data + all_data  # prepend
                remaining -= len(data)
                end_time = int(data[0][0]) - 1

        highs = [float(k[2]) for k in all_data]
        lows = [float(k[3]) for k in all_data]
        closes = [float(k[4]) for k in all_data]
        volumes = [float(k[5]) for k in all_data]

        return highs[:-1], lows[:-1], closes[:-1], volumes[:-1]

    # ── Mode 1: Walk-Forward from Binance API ─────────────────

    async def retrain_model_walk_forward(self, symbols: List[str], limit_bars: int = 1344):
        """
        Walk-Forward training on historical OHLCV data.
        V26: Uses TIMEFRAME from config (15m by default), collects regime labels.
        """
        logger.info("🧠 Walk-Forward V26: Retraining on %d bars of %s for %s...",
                     limit_bars, TIMEFRAME, symbols)

        ml_features = []
        ml_outcomes = []
        ml_regimes = []
        LOOKAHEAD = 12
        warmup = 60

        for symbol in symbols:
            highs, lows, closes, volumes = await self.fetch_historical_ohlcv_async(
                symbol, TIMEFRAME, limit_bars
            )

            n_bars = len(closes)
            if n_bars < warmup + LOOKAHEAD:
                logger.warning("⚠️ %s: %d bars — insufficient. Skipping.", symbol, n_bars)
                continue

            for i in range(warmup, n_bars - LOOKAHEAD):
                h, l, c, v = highs[:i], lows[:i], closes[:i], volumes[:i]
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

                # V26: Lower threshold — we want more training samples
                if abs(score) > 0.05:
                    features = self.ml_filter.extract_features(
                        state, composite_score=score, closes=c, volumes=v, hour=i % 24
                    )
                    if features is not None:
                        future_prices = closes[i:i + LOOKAHEAD]
                        if score > 0:
                            max_future = max(future_prices)
                            move_pct = (max_future - current_price) / current_price * 100
                        else:
                            min_future = min(future_prices)
                            move_pct = (current_price - min_future) / current_price * 100

                        outcome = 1 if move_pct > (atr_pct * ATR_SL_MULTIPLIER * 0.5) else 0

                        ml_features.append(features)
                        ml_outcomes.append(outcome)
                        ml_regimes.append(regime)

        if len(ml_features) >= 50:
            metrics = self.ml_filter.train(ml_features, ml_outcomes, regimes=ml_regimes)
            self.ml_filter.save()
            logger.info("✅ Walk-Forward V26 Done: %d samples → %s", len(ml_features), metrics)
        else:
            logger.warning("⚠️ Only %d samples found. ML retraining skipped.", len(ml_features))

    # ── Mode 2: Train from Signal Journal ─────────────────────

    async def retrain_from_journal(self):
        """
        Train ML from signal_journal.db outcomes.
        Uses real signals with actual price outcomes — highest quality data.
        """
        try:
            from signal_journal import get_unanalyzed_signals
            import sqlite3

            conn = sqlite3.connect("signal_journal.db")
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM signals
                   WHERE would_have_profited IS NOT NULL
                     AND original_action != 'HOLD'
                   ORDER BY timestamp ASC"""
            ).fetchall()
            conn.close()

            if len(rows) < 50:
                logger.info("📓 Journal has %d labeled signals (need 50+). Using Walk-Forward instead.",
                            len(rows))
                return False

            logger.info("📓 Training ML V26 from %d journal signals...", len(rows))

            ml_features = []
            ml_outcomes = []
            ml_regimes = []

            for row in rows:
                row = dict(row)
                # Reconstruct a minimal MarketState from journal columns
                state = MarketState(
                    current_price=row["price_at_signal"],
                    rsi=row["rsi"],
                    ls_ratio=1.0,
                    whale_net_vol=0.0,
                    regime=row.get("regime", "CHOPPY"),
                    social_score=0.0,
                    macd_histogram=0.0,
                    bb_position=0.5,
                    vwap_diff_pct=0.0,
                    divergence="NONE",
                    funding_rate=0.0,
                    open_interest_delta=0.0,
                    liq_imbalance=row.get("obi", 0.0),
                    atr_pct=row.get("atr_pct", 1.0),
                    rsi_slope=0.0,
                    stoch_rsi=row["rsi"],  # Approximation
                    mtf_agreement=0.0,
                    volume_confirm=True,
                    near_resistance=False,
                    btc_correlation=1.0,
                    btc_dominance=50.0,
                )

                hour = 12  # Default; could parse from timestamp
                try:
                    ts = datetime.fromisoformat(row["timestamp"])
                    hour = ts.hour
                except Exception:
                    pass

                features = self.ml_filter.extract_features(state, composite_score=row["composite_score"],
                                                            closes=[], volumes=[], hour=hour)
                if features is not None:
                    ml_features.append(features)
                    ml_outcomes.append(row["would_have_profited"])
                    ml_regimes.append(row.get("regime", "CHOPPY"))

            if len(ml_features) >= 50:
                metrics = self.ml_filter.train(ml_features, ml_outcomes, regimes=ml_regimes)
                self.ml_filter.save()
                logger.info("✅ Journal Training V26 Done: %d samples → %s", len(ml_features), metrics)
                return True
            else:
                logger.warning("⚠️ Only %d usable journal samples. Need Walk-Forward.", len(ml_features))
                return False

        except Exception as e:
            logger.error("Journal training failed: %s", e)
            return False
