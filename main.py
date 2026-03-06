"""
Hunter V17 — Main Loop
========================
Orchestrator: fetches asynchronous data via BinanceProvider → analyses → decides → executes.

V17: Async I/O refactoring, MarketState dataclass to fix parameter bloat.
V16: Persistence, Kelly Criterion, fees, StochRSI, Vol confirm.
V15: Short positions, funding rate sentiment, trailing SL.
V14: Multi-Timeframe analysis, MACD/ATR/VWAP indicators.
"""

import argparse
import asyncio
import inspect
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import (
    ADX_PERIOD,
    ATR_PERIOD,
    BB_PERIOD,
    CHECK_INTERVAL_SEC,
    LIVE_TRADING,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MTF_AGREEMENT_MIN,
    MULTI_TF_INTERVALS,
    TIMEFRAME,
    RSI_PERIOD,
    SR_LOOKBACK,
    SR_PROXIMITY_PCT,
    VOLUME_CONFIRM_BARS,
    VOLUME_CONFIRM_ENABLED,
    VWAP_BARS,
)
from analysis import (
    MarketState,
    compute_adx,
    compute_atr,
    compute_bollinger,
    compute_macd,
    compute_rsi,
    compute_rsi_series,
    compute_rsi_slope,
    compute_stoch_rsi,
    compute_support_resistance,
    compute_vwap,
    detect_divergence,
    generate_signal,
    get_market_regime,
)
from execution import PaperTrader
from live_execution import LiveTrader
from social import SocialManager
from macro import MacroManager
from ml import MLFilter
from provider import BinanceProvider
from report import ReportGenerator

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hunter.main")

# ─────────────────────────────────────────────────────────────
# Single Cycle (for one symbol)  — V17 Async
# ─────────────────────────────────────────────────────────────
async def run_cycle(
    trader, 
    symbol: str,  
    social_manager: SocialManager, 
    macro_manager: MacroManager,
    provider: BinanceProvider,
    prefetched_data: Dict = None,
    ml_filter: MLFilter = None,
) -> Dict:
    """Execute one analysis-and-trade cycle asynchronously for a given symbol."""
    logger.info("─" * 60)
    logger.info("▶ Analysing %s", symbol)

    # 1. Fetch ALL data concurrently or use WSS cache
    if prefetched_data:
        data = prefetched_data
    else:
        try:
            data = await provider.fetch_all_market_data(symbol, MULTI_TF_INTERVALS)
        except Exception as exc:
            logger.error("❌ Data fetch failed for %s: %s", symbol, exc)
            return {"action": "FETCH_ERROR", "symbol": symbol, "error": str(exc)}

    highs, lows, closes, volumes = data["highs"], data["lows"], data["closes"], data["volumes"]
    if not closes:
        return {"action": "FETCH_ERROR", "symbol": symbol, "error": "Empty OHLCV"}

    current_price = closes[-1]

    # 2. Market Regime Filter (ADX)
    adx = compute_adx(highs, lows, closes, ADX_PERIOD)
    regime = get_market_regime(adx)
    logger.info(
        "📊 %s @ $%.4f | ADX=%.2f → %s", symbol, current_price, adx, regime
    )

    if regime == "CHOPPY":
        # Even in CHOPPY, check SL/TP for open positions
        if trader.has_position(symbol):
            atr = compute_atr(highs, lows, closes, ATR_PERIOD)
            result = trader.execute_trade("HOLD", current_price, symbol, atr=atr)
            if result["action"] in ("CLOSED_SL", "CLOSED_TP"):
                logger.info("   ⚡ SL/TP triggered in CHOPPY market for %s", symbol)
                return result

        logger.info("⏸️  Market is CHOPPY (ADX < 25) — skipping %s", symbol)
        return {
            "action": "SKIP_CHOPPY",
            "symbol": symbol,
            "adx": adx,
            "regime": regime,
            "price": current_price,
        }

    # 3. Compute indicators
    rsi = compute_rsi(closes, RSI_PERIOD)
    lower, middle, upper = compute_bollinger(closes, BB_PERIOD)
    logger.info("   RSI=%.2f | BB=[%.2f / %.2f / %.2f]", rsi, lower, middle, upper)

    macd_line, signal_line, histogram = compute_macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    logger.info("   MACD: line=%.4f | signal=%.4f | hist=%.4f", macd_line, signal_line, histogram)

    atr = compute_atr(highs, lows, closes, ATR_PERIOD)
    logger.info("   ATR=%.4f", atr)

    vwap_start = max(0, len(closes) - VWAP_BARS)
    vwap = compute_vwap(highs[vwap_start:], lows[vwap_start:], closes[vwap_start:], volumes[vwap_start:])
    vwap_diff_pct = ((current_price - vwap) / vwap * 100) if vwap > 0 else 0
    logger.info("   VWAP=%.4f (price %+.2f%%) [bars=%d]", vwap, vwap_diff_pct, VWAP_BARS)

    rsi_slope = compute_rsi_slope(closes)
    stoch_rsi = compute_stoch_rsi(closes)

    rsi_series = compute_rsi_series(closes, RSI_PERIOD)
    divergence = detect_divergence(closes, rsi_series, adx_value=adx)

    supports, resistances = compute_support_resistance(highs, lows, SR_LOOKBACK)
    near_support = any(abs(current_price - s) / current_price * 100 < SR_PROXIMITY_PCT for s in supports)
    near_resistance = any(abs(current_price - r) / current_price * 100 < SR_PROXIMITY_PCT for r in resistances)
    
    if supports:
        logger.info("   🟢 Supports: %s %s", [f"${s:.2f}" for s in supports[-3:]], "← NEAR!" if near_support else "")
    if resistances:
        logger.info("   🔴 Resistances: %s %s", [f"${r:.2f}" for r in resistances[-3:]], "← NEAR!" if near_resistance else "")

    ls_ratio = data["ls_ratio"]
    whale_vol = data["whale_vol"]
    funding_rate = data["funding_rate"]
    oi_delta = data["oi_delta"]
    liq_imbalance = data["liq_imbalance"]

    volume_confirm = True
    if VOLUME_CONFIRM_ENABLED and len(volumes) >= VOLUME_CONFIRM_BARS * 2:
        recent_vol = sum(volumes[-VOLUME_CONFIRM_BARS:])
        prev_vol = sum(volumes[-(VOLUME_CONFIRM_BARS * 2):-VOLUME_CONFIRM_BARS])
        volume_confirm = recent_vol > prev_vol
            
    # 5. Social & Sentiment Score
    social_score = social_manager.get_social_score(symbol)

    btc_dom = await macro_manager.get_btc_dominance()
    btc_corr = await macro_manager.get_btc_correlation(symbol, provider)

    # 6. Compute MTF Agreement
    mtf_scores = []
    for tf, tf_closes in data["mtf_closes"].items():
        if tf_closes:
            tf_rsi = compute_rsi(tf_closes, RSI_PERIOD)
            score = 1.0 - 2.0 * (tf_rsi - 30.0) / (70.0 - 30.0)
            mtf_scores.append(max(-1.0, min(1.0, score)))
        else:
            mtf_scores.append(0.0)
            
    mtf_agreement = sum(mtf_scores) / len(mtf_scores) if mtf_scores else 0.0

    bb_range = upper - lower
    bb_position = (current_price - lower) / bb_range if bb_range > 0 else 0.5

    # 7. Create MarketState (V17 refactoring)
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
        liq_imbalance=liq_imbalance,
        atr_pct=atr_pct,
        rsi_slope=rsi_slope,
        stoch_rsi=stoch_rsi,
        mtf_agreement=mtf_agreement,
        volume_confirm=volume_confirm,
        near_resistance=near_resistance,
        btc_correlation=btc_corr,
        btc_dominance=btc_dom
    )

    # 8. Generate signal
    pos = trader.get_position(symbol)
    signal_dict = generate_signal(market_state, current_position=pos, use_composite=True, detailed=True)

    # 8.5 ML Filter Gate (V22 Phase 6)
    action = signal_dict.get("action", "HOLD")
    if action not in ("HOLD", "DCA_BUY", "DCA_SHORT") and ml_filter is not None:
        if not ml_filter.should_trade(market_state, 
                                       composite_score=signal_dict.get("composite_score", 0),
                                       closes=closes, volumes=volumes,
                                       hour=datetime.now(timezone.utc).hour):
            action = "HOLD"  # ML blocked this signal

    # 8.75 Microstructure Order Book Imbalance Gate (V23 Phase 2)
    bbo = provider.get_bbo(symbol)
    if bbo and action in ["BUY", "SHORT"]:
        bid_q = bbo['bid_qty']
        ask_q = bbo['ask_qty']
        obi = (bid_q - ask_q) / (bid_q + ask_q) if (bid_q + ask_q) > 0 else 0.0
        logger.info("   🔬 Microstructure: OBI=%+.2f (Bid:%.1f Ask:%.1f)", obi, bid_q, ask_q)
        
        if action == "BUY" and obi < -0.30:
            logger.warning("   ⛔ OBI BLOCKED BUY: Heavy Ask wall detected (OBI %+.2f)", obi)
            action = "HOLD"
        elif action == "SHORT" and obi > 0.30:
            logger.warning("   ⛔ OBI BLOCKED SHORT: Heavy Bid wall detected (OBI %+.2f)", obi)
            action = "HOLD"

    # 9. Execute
    result = trader.execute_trade(action, current_price, symbol, atr=atr)
    if inspect.iscoroutine(result):
        result = await result
    
    # 10. Generate beautiful Report
    ReportGenerator.print_cycle_report(symbol, market_state, signal_dict, result)
    
    return result


async def run_manual(symbol: str):
    """Manual Mode: analyse one symbol, print results, exit."""
    logger.info("==============================================")
    logger.info("  HUNTER V17 — Manual Mode (Async)")
    logger.info("  Symbol: %s", symbol)
    logger.info("==============================================")

    trader = LiveTrader() if LIVE_TRADING else PaperTrader()
    social_manager = SocialManager()
    macro_manager = MacroManager()
    
    async with BinanceProvider() as provider:
        fear_greed = social_manager.news_manager.get_fear_and_greed()
        logger.info(
            "  😱 Fear & Greed: %s (%s)",
            fear_greed[0],
            fear_greed[1],
        )

        result = await run_cycle(trader, symbol, social_manager, macro_manager, provider, ml_filter=None)

        logger.info("==============================================")
        logger.info("  Result: %s", json.dumps(result, indent=2) if isinstance(result, dict) else result)
        logger.info("==============================================")


async def run_pair_wss(symbol: str, provider: BinanceProvider, trader, social_manager: SocialManager, macro_manager: MacroManager, ml_filter: MLFilter = None):
    """Handles persistent WSS stream for a specific pair."""
    logger.info("📡 Starting Zero-Latency WSS listener for %s", symbol)
    while True:
        try:
            # Prime data via REST
            data = await provider.fetch_all_market_data(symbol, MULTI_TF_INTERVALS)
            
            async for kline in provider.stream_klines(symbol, TIMEFRAME):
                if kline.get('x'): # Candle closed
                    # 1. Update strictly needed OHLCV arrays instantly
                    data['highs'].pop(0); data['highs'].append(float(kline['h']))
                    data['lows'].pop(0);  data['lows'].append(float(kline['l']))
                    data['closes'].pop(0); data['closes'].append(float(kline['c']))
                    data['volumes'].pop(0); data['volumes'].append(float(kline['v']))
                    
                    # 2. Zero-latency execution using the cached auxiliary data!
                    await run_cycle(trader, symbol, social_manager, macro_manager, provider, prefetched_data=data, ml_filter=ml_filter)
                    
                    # 3. Refresh auxiliary data AFTER execution so it's ready for the next candle!
                    aux = await provider.fetch_all_market_data(symbol, MULTI_TF_INTERVALS)
                    data.update({k: v for k, v in aux.items() if k not in ['highs', 'lows', 'closes', 'volumes']})
                    
        except asyncio.CancelledError:
            logger.info("🛑 WSS Loop cancelled for %s", symbol)
            break
        except Exception as e:
            logger.error("❌ WSS Crash on %s: %s. Restarting...", symbol, e)
            await asyncio.sleep(5)

async def run_auto():
    """Auto Mode: scan top pairs + held positions, loop over them asynchronously."""
    logger.info("==============================================")
    logger.info("  HUNTER V19 — WSS Auto Mode Started")
    logger.info("==============================================")

    trader = LiveTrader() if LIVE_TRADING else PaperTrader()
    social_manager = SocialManager()
    macro_manager = MacroManager()
    ml_filter = MLFilter()
    ml_filter.load()  # Load pre-trained model if available

    # V22: Sync real exchange balance before any trades
    if LIVE_TRADING and hasattr(trader, 'sync_balance'):
        import asyncio as _aio
        bal = await trader.sync_balance()
        logger.info("💰 Exchange balance synced: $%.2f", bal)

    async with BinanceProvider() as provider:
        fg = social_manager.news_manager.get_fear_and_greed()
        logger.info("Global F&G Index: %s (%s)", fg[0], fg[1])

        try:
            top_pairs = await provider.scan_top_pairs()
            
            # Active positions must always be monitored
            for sym in trader.positions.keys():
                if sym not in top_pairs:
                    top_pairs.append(sym)
            
            logger.info("🎯 Starting WSS auto mode for %d pairs: %s", len(top_pairs), top_pairs)
            
            # Run persistent WSS tasks concurrently indefinitely
            tasks = []
            for sym in top_pairs:
                tasks.append(run_pair_wss(sym, provider, trader, social_manager, macro_manager, ml_filter))
                tasks.append(provider.stream_bbo(sym))
            
            await asyncio.gather(*tasks)

        except Exception as exc:
            logger.error("❌ Unhandled exception in main setup: %s", exc)
            await asyncio.sleep(5)


async def main():
    parser = argparse.ArgumentParser(description="Hunter V17 Trading Bot")
    parser.add_argument("--symbol", type=str, help="Run single analysis for a pair (e.g. BTCUSDT)")
    args = parser.parse_args()

    if args.symbol:
        await run_manual(args.symbol.upper())
    else:
        await run_auto()


if __name__ == "__main__":
    import json
    # Run asyncio event loop
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user. Graceful shutdown.")
    except Exception as e:
        logger.error("🛑 Fatal execution error: %s", e)
