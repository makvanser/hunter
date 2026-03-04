"""
Hunter V16 — Forecast Accuracy Evaluation
=========================================
Evaluates the predictive liquidity and accuracy of generate_signal()
by simulating signals on historical data and checking N bars into the future.
"""

import asyncio
import logging
from typing import List

from provider import BinanceProvider
from analysis import (
     MarketState, compute_rsi_series, compute_adx, compute_bollinger,
     compute_macd, compute_vwap, detect_divergence, compute_atr, generate_signal
)
from config import (
    RSI_PERIOD, ADX_PERIOD, BB_PERIOD, BB_STD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL, VWAP_BARS, ATR_PERIOD,
    TIMEFRAME
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("hunter.eval")

async def evaluate_forecasts(symbol: str = "BTCUSDT", forward_bars: int = 12):
    """
    Downloads historical data, generates signals, and checks whether the price
    moved in the predicted direction after `forward_bars`.
    """
    logger.info(f"Starting Forecast Evaluation for {symbol} on {TIMEFRAME}...")
    logger.info(f"Checking forward accuracy {forward_bars} bars into the future.")

    async with BinanceProvider() as provider:
        highs, lows, closes, volumes = await provider.fetch_ohlcv(symbol, TIMEFRAME, limit=1500)
        
    total_buy_signals = 0
    correct_buy_signals = 0
    
    total_sell_signals = 0
    correct_sell_signals = 0

    # Start loop after enough data is available for all indicators (e.g. 200 bars)
    start_idx = 200
    end_idx = len(closes) - forward_bars - 1
    
    for i in range(start_idx, end_idx):
        c_i = closes[i]
        future_c = closes[i + forward_bars]
        slice_volumes = volumes[:i+1]
        
        # Calculate indicators up to bar i
        slice_closes = closes[:i+1]
        slice_highs = highs[:i+1]
        slice_lows = lows[:i+1]
        
        rsi_series = compute_rsi_series(slice_closes, RSI_PERIOD)
        rsi = rsi_series[-1] if rsi_series else 50.0
        
        adx = compute_adx(slice_highs, slice_lows, slice_closes, ADX_PERIOD)
        regime = "TRENDING" if adx >= 25 else "CHOPPY"
        
        upper, middle, lower = compute_bollinger(slice_closes, BB_PERIOD, BB_STD)
        macd_val, signal_line, histogram = compute_macd(slice_closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        
        vwap = compute_vwap(slice_highs, slice_lows, slice_closes, slice_volumes)
        vwap_diff = ((c_i - vwap) / vwap * 100) if vwap else 0.0
        
        atr = compute_atr(slice_highs, slice_lows, slice_closes, ATR_PERIOD)
        atr_pct = (atr / c_i * 100) if c_i else 0.0
        
        divergence = detect_divergence(slice_closes, rsi_series, adx_value=adx)
        
        bb_range = upper - lower
        bb_position = (c_i - lower) / bb_range if bb_range > 0 else 0.5
        
        stoch_rsi = 50.0
        rsi_slope = 0.0

        market_state = MarketState(
            current_price=c_i,
            rsi=rsi,
            ls_ratio=1.0,         # Mock Macro/OnChain for pure Technical evaluation
            whale_net_vol=0.0,
            regime=regime,
            social_score=0.0,     # Neutral social
            macd_histogram=histogram,
            bb_position=bb_position,
            vwap_diff_pct=vwap_diff,
            divergence=divergence,
            funding_rate=0.0,
            open_interest_delta=0.0,
            liq_imbalance=0.0,
            atr_pct=atr_pct,
            rsi_slope=rsi_slope,
            stoch_rsi=stoch_rsi,
            mtf_agreement=0.0,
            volume_confirm=True,  # Neutralize
            near_resistance=False,
            btc_correlation=1.0,  # BTC is of course correlated with BTC
            btc_dominance=50.0
        )

        signal = generate_signal(market_state, current_position=None, use_composite=True)
        
        if signal == "BUY":
            total_buy_signals += 1
            if future_c > c_i:
                correct_buy_signals += 1
        elif signal == "SELL" or signal == "SHORT":
            total_sell_signals += 1
            if future_c < c_i:
                correct_sell_signals += 1

    logger.info("=========================================")
    logger.info(f"FORECAST ACCURACY REPORT (Prediction {forward_bars} bars ahead)")
    logger.info("=========================================")
    logger.info(f"Total BUY Signals evaluated: {total_buy_signals}")
    if total_buy_signals > 0:
        logger.info(f"✅ BUY Signal Accuracy: {correct_buy_signals / total_buy_signals * 100:.1f}%")
        
    logger.info(f"Total SELL Signals evaluated: {total_sell_signals}")
    if total_sell_signals > 0:
        logger.info(f"✅ SELL Signal Accuracy: {correct_sell_signals / total_sell_signals * 100:.1f}%")
        
    total_signals = total_buy_signals + total_sell_signals
    total_correct = correct_buy_signals + correct_sell_signals
    if total_signals > 0:
        logger.info(f"🎯 OVERALL FORECAST ACCURACY: {total_correct / total_signals * 100:.1f}%")
    logger.info("=========================================")


if __name__ == "__main__":
    asyncio.run(evaluate_forecasts(symbol="BTCUSDT", forward_bars=24))
    asyncio.run(evaluate_forecasts(symbol="ETHUSDT", forward_bars=24))
