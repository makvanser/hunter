"""
Hunter V25 — Continuous Learning Module
=======================================
Implements Walk-Forward Optimization for the ML signal filter.
Periodically fetches the latest N days of OHLCV data, constructs features,
and retrains the Random Forest / Gradient Boosting model to adapt to fresh regimes.
"""

import asyncio
import logging
import json
import numpy as np
import urllib.request
from datetime import datetime, timezone
from typing import List

from ml import MLFilter
from config import (
    ADX_PERIOD, ATR_PERIOD, BB_PERIOD, RSI_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL, SR_LOOKBACK, VWAP_BARS,
    ATR_SL_MULTIPLIER
)
from analysis import (
    compute_adx, compute_atr, compute_bollinger, compute_macd,
    compute_rsi, compute_rsi_series, compute_support_resistance,
    compute_vwap, detect_divergence, get_market_regime,
    compute_rsi_slope, compute_stoch_rsi, compute_composite_score,
    MarketState
)

# Reuse the backtest pagination fetcher, but async
import aiohttp

logger = logging.getLogger("hunter.learner")


class ContinuousLearner:
    def __init__(self, ml_filter: MLFilter):
        self.ml_filter = ml_filter

    async def fetch_historical_ohlcv_async(self, symbol: str, interval: str, limit: int = 336):
        """Async paginated fetcher for large candle history without blocking."""
        all_data = []
        end_time = None
        remaining = limit
        
        async with aiohttp.ClientSession() as session:
            while remaining > 0:
                batch_size = min(remaining, 1500)
                url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={batch_size}"
                if end_time:
                    url += f"&endTime={end_time}"
                
                try:
                    async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as response:
                        if response.status != 200:
                            logger.error("Failed to fetch historical data: HTTP %s", response.status)
                            break
                        data = await response.json()
                except Exception as e:
                    logger.error("Async fetch error in continuous learner: %s", e)
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
        
        # Omit current unclosed candle
        return highs[:-1], lows[:-1], closes[:-1], volumes[:-1]

    async def retrain_model_walk_forward(self, symbols: List[str], limit_bars: int = 336):
        """
        Fetches the last `limit_bars` (e.g. 14 days of 1h candles) for multiple symbols.
        Builds feature/outcome arrays and retrains the ML filter.
        """
        logger.info("🧠 Walk-Forward Optimization: Retraining ML on last %d bars for %s...", limit_bars, symbols)
        
        ml_features = []
        ml_outcomes = []
        LOOKAHEAD = 12
        warmup = 50

        for symbol in symbols:
            highs, lows, closes, volumes = await self.fetch_historical_ohlcv_async(symbol, "1h", limit_bars)
            
            n_bars = len(closes)
            if n_bars < warmup + LOOKAHEAD:
                logger.warning("⚠️ Insufficient data for %s (%d bars). Skipping.", symbol, n_bars)
                continue
                
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
                
                # Record feature set for all distinct signals
                if abs(score) > 0.15: 
                    features = self.ml_filter.extract_features(state, composite_score=score,
                                                               closes=c, volumes=v, hour=i % 24)
                    if features is not None:
                        future_prices = closes[i:i + LOOKAHEAD]
                        if score > 0:  # LONG Prediction
                            max_future = max(future_prices)
                            move_pct = (max_future - current_price) / current_price * 100
                            outcome = 1 if move_pct > (atr_pct * ATR_SL_MULTIPLIER) else 0
                        else:  # SHORT Prediction
                            min_future = min(future_prices)
                            move_pct = (current_price - min_future) / current_price * 100
                            outcome = 1 if move_pct > (atr_pct * ATR_SL_MULTIPLIER) else 0
                        
                        ml_features.append(features)
                        ml_outcomes.append(outcome)

        if len(ml_features) >= 50:
            metrics = self.ml_filter.train(ml_features, ml_outcomes)
            self.ml_filter.save()
            logger.info("✅ Walk-Forward Training Completed: %d samples, %.1f%% CV Accuracy", 
                        metrics['samples'], metrics['cv_accuracy'] * 100)
        else:
            logger.warning("⚠️ Only %d signal samples found. ML Retraining skipped.", len(ml_features))
