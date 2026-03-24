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
from database import is_on_cooldown, reset_circuit_breaker
from provider import BinanceProvider
from report import ReportGenerator
from social import SocialManager
from macro import MacroManager
from ml import MLFilter
from learner import ContinuousLearner
from statarb import StatArbEngine
from telemetry import TelemetryManager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from signal_journal import init_journal, log_signal
from signal_analyzer import run_weekly_analysis
from strategy_router import StrategyRouter
from portfolio_risk import PortfolioManager

# ─────────────────────────────────────────────────────────────
# Global Instances
# ─────────────────────────────────────────────────────────────
ml_filter = MLFilter()
strategy_router = StrategyRouter()

# Global state for V29 Microstructure
PREV_OBI: Dict[str, float] = {}
CVD_LAST_TIME: Dict[str, float] = {}
CVD_LAST_VALUE: Dict[str, float] = {}
market_closes_cache: Dict[str, List[float]] = {}

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
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

    TelemetryManager.set_adx(symbol, adx)

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

    # V28 Phase 4: StatArb Spread Z-Score
    btc_spread_zscore = 0.0
    if symbol != "BTCUSDT" and hasattr(macro_manager, 'btc_closes_cache') and macro_manager.btc_closes_cache:
        import math
        min_len = min(len(closes), len(macro_manager.btc_closes_cache))
        if min_len >= 20:
            sym_slice = closes[-min_len:]
            btc_slice = macro_manager.btc_closes_cache[-min_len:]
            
            ratios = [s / b if b > 0 else 0 for s, b in zip(sym_slice, btc_slice)]
            mean_ratio = sum(ratios) / len(ratios)
            std_ratio = math.sqrt(sum((r - mean_ratio)**2 for r in ratios) / len(ratios))
            
            if std_ratio > 0:
                btc_spread_zscore = (ratios[-1] - mean_ratio) / std_ratio
                logger.info("   ⚖️ StatArb: %s vs BTC Z-Score = %+.2f", symbol, btc_spread_zscore)

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

    atr_pct = (atr / current_price * 100) if current_price else 0.0
    cvd_value = provider.get_cvd(symbol) if hasattr(provider, 'get_cvd') else 0.0
    
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
        btc_dominance=btc_dom,
        cvd=cvd_value,
        btc_spread_zscore=btc_spread_zscore
    )

    # 8. Strategy Router Evaluation (V27 Multi-Strategy)
    pos = trader.get_position(symbol)
    pos_side = pos["side"] if pos else None
    signal_dict = strategy_router.evaluate(market_state, current_position=pos_side)

    # 8.5 ML Filter Gate (V22 Phase 6)
    action = signal_dict.get("action", "HOLD")
    if action not in ("HOLD", "DCA_BUY", "DCA_SHORT") and ml_filter is not None:
        if not ml_filter.should_trade(market_state, 
                                       composite_score=signal_dict.get("composite_score", 0),
                                       closes=closes, volumes=volumes,
                                       hour=datetime.now(timezone.utc).hour):
            action = "HOLD"  # ML blocked this signal
            
        # Hook ML Confidence to Telemetry (if available)
        if hasattr(ml_filter, 'last_probability'):
            TelemetryManager.set_ml_confidence(symbol, action, ml_filter.last_probability)

    # 8.75 Microstructure Order Book Imbalance Gate (V23/V24 Phase 3)
    # V24 Phase 3: Using Deep OBI (Top 20 levels) instead of BBO for true wall detection
    obi = provider.get_deep_obi(symbol)
    TelemetryManager.set_deep_obi(symbol, obi)
    
    # V29 Phase 3: OBI Delta (Acceleration)
    prev_obi = PREV_OBI.get(symbol, obi)
    obi_delta = obi - prev_obi
    PREV_OBI[symbol] = obi

    # V29 Phase 3: Bid/Ask Spread
    bbo = provider.get_bbo(symbol)
    spread = 0.0
    if bbo:
        spread = (bbo['ask_price'] - bbo['bid_price']) / bbo['bid_price'] * 100 if bbo['bid_price'] > 0 else 0.0

    # V29 Phase 3: CVD Slope (Intensity of flow)
    now = time.time()
    last_t = CVD_LAST_TIME.get(symbol, now - 60)
    last_v = CVD_LAST_VALUE.get(symbol, cvd_value)
    dt = now - last_t
    cvd_slope = (cvd_value - last_v) / dt if dt > 0 else 0.0
    CVD_LAST_TIME[symbol] = now
    CVD_LAST_VALUE[symbol] = cvd_value

    if action in ["BUY", "SHORT"]:
        logger.info("   🔬 Micro: OBI=%+.2f (Δ%+.2f) | Spread=%.3f%% | CVD_S=%.1f", obi, obi_delta, spread, cvd_slope)
        
        if action == "BUY" and obi < -0.30:
            logger.warning("   ⛔ OBI BLOCKED BUY: Heavy Ask wall detected in Depth20 (OBI %+.2f)", obi)
            action = "HOLD"
        elif action == "SHORT" and obi > 0.30:
            logger.warning("   ⛔ OBI BLOCKED SHORT: Heavy Bid wall detected in Depth20 (OBI %+.2f)", obi)
            action = "HOLD"

    market_state.obi = obi
    market_state.obi_delta = obi_delta
    market_state.bid_ask_spread = spread
    market_state.cvd_slope = cvd_slope

    # V29 Phase 1: Portfolio Correlation VaR Gate
    if action in ["BUY", "SHORT"] and not trader.has_position(symbol):
        if hasattr(trader, 'portfolio_manager'):
            is_safe = await trader.portfolio_manager.check_trade_correlation(
                new_symbol=symbol,
                target_side=action,
                current_positions=trader.positions,
                provider=provider
            )
            if not is_safe:
                logger.warning("   🛡️ PORTFOLIO VaR BLOCKED: %s %s heavily correlated with open positions", action, symbol)
                action = "HOLD"

    # V26/V27: Diagnostic Signal Pipeline Log
    original_action = signal_dict.get("action", "HOLD")
    strategy_name = signal_dict.get("strategy", "None")
    ml_prob = getattr(ml_filter, 'last_probability', 0.0) if ml_filter else 0.0
    logger.info(
        "ROUTER [%s]: %s → %s(%.2f) → ML=%.0f%% → OBI=%+.2f → final=%s",
        strategy_name, symbol, original_action, signal_dict.get("confidence", 0.0),
        ml_prob * 100, obi, action
    )

    # V26: Signal Journal — record EVERY signal for weekly analysis
    blocked_by = None
    if original_action != action:
        if original_action != "HOLD":
            # Determine which filter blocked it
            if ml_filter and hasattr(ml_filter, 'last_probability') and ml_filter.last_probability < 0.51:
                blocked_by = "ML"
            elif abs(obi) > 0.30:
                blocked_by = "OBI"
            else:
                blocked_by = "OTHER"
    log_signal(
        symbol=symbol,
        composite_score=signal_dict.get("composite_score", 0.0),
        original_action=original_action,
        final_action=action,
        blocked_by=blocked_by,
        ml_confidence=ml_prob,
        obi=obi,
        obi_delta=obi_delta,
        cvd_slope=cvd_slope,
        bid_ask_spread=spread,
        rsi=market_state.rsi,
        adx={"CHOPPY": 15.0, "TRENDING": 30.0, "STRONG_UP": 45.0, "STRONG_DOWN": 45.0}.get(market_state.regime, 20.0),
        atr_pct=market_state.atr_pct,
        regime=market_state.regime,
        price_at_signal=current_price,
    )

    # 9. Execute with latency tracking
    with TelemetryManager.track_latency():
        result = trader.execute_trade(action, current_price, symbol, atr=atr, provider=provider)
        if inspect.iscoroutine(result):
            result = await result
            
        if result.get("action") not in ("NONE", "HOLD", "NO_ACTION"):
            TelemetryManager.inc_trade(result["action"], symbol)
    
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


async def run_pair_wss(symbol: str, provider: BinanceProvider, trader, ml_filter: MLFilter = None, macro_manager: MacroManager = None, strategy_router=None):
    """Handles persistent WSS stream for a specific pair."""
    logger.info("📡 Starting Zero-Latency WSS listener for %s", symbol)
    while True:
        # Track system metrics via Prometheus (V25)
        TelemetryManager.set_balance(trader.balance)
        TelemetryManager.set_open_positions(trader.open_positions_count())
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
                    
                    market_closes_cache[symbol] = data['closes'][-200:]
                    
                    # 2. Zero-latency execution using the cached auxiliary data!
                    await run_cycle(trader, symbol, social_manager, macro_manager, provider, prefetched_data=data, ml_filter=ml_filter)
                    
                    # 2.5 V28 Phase 2 CVD Reset
                    provider.reset_cvd(symbol)
                    
                    # 3. Refresh auxiliary data AFTER execution so it's ready for the next candle!
                    aux = await provider.fetch_all_market_data(symbol, MULTI_TF_INTERVALS)
                    data.update({k: v for k, v in aux.items() if k not in ['highs', 'lows', 'closes', 'volumes']})
                    
        except asyncio.CancelledError:
            logger.info("🛑 WSS Loop cancelled for %s", symbol)
            break
        except Exception as e:
            logger.error("❌ WSS Crash on %s: %s. Restarting...", symbol, e)
            await asyncio.sleep(5)

async def _statarb_monitor_loop():
    engine = StatArbEngine(z_score_threshold=2.5)
    while True:
        await asyncio.sleep(60 * 60) # Run every hour
        if bool(market_closes_cache):
            opps = engine.find_arbitrage_opportunities(market_closes_cache)
            if opps:
                logger.info("==============================================")
                logger.info("  📈 STATISTICAL ARBITRAGE OPPORTUNITIES")
                logger.info("==============================================")
                for opp in opps[:3]: # Top 3
                    logger.info("  %s -> Z-Score: %+.2f | Action: %s", opp['pair'], opp['z_score'], opp['action'])
                logger.info("==============================================")


async def run_auto():
    """Auto Mode: scan top pairs + held positions, loop over them asynchronously."""
    logger.info("==============================================")
    logger.info("  HUNTER V19 — WSS Auto Mode Started")
    logger.info("==============================================")

    trader = LiveTrader() if LIVE_TRADING else PaperTrader()
    social_manager = SocialManager()
    macro_manager = MacroManager()
    ml_filter.load()  # Load pre-trained model if available
    init_journal()     # V26: Initialize Signal Journal DB
    
    # Expose Prometheus HTTP Metrics Endpoint
    TelemetryManager.start_server(port=8000)

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
            
            # WSS Execution Warmup (V28 Phase 3)
            if LIVE_TRADING and hasattr(trader, 'connect_ws'):
                await trader.connect_ws()
            
            # Active positions must always be monitored
            for sym in trader.positions.keys():
                if sym not in top_pairs:
                    top_pairs.append(sym)
            
            logger.info("🎯 Starting WSS auto mode for %d pairs: %s", len(top_pairs), top_pairs)
            
            # Phase 2: ML Continuous Learning Pipeline (Walk-Forward Optimization)
            learner = ContinuousLearner(ml_filter)
            scheduler = AsyncIOScheduler()
            # Retrain once a week on Sundays at 00:00 (14 days = 336 hours limits)
            scheduler.add_job(
                learner.retrain_model_walk_forward, 
                'cron', day_of_week='sun', hour=0, minute=0,
                args=[top_pairs[:10], 1344]
            )
            # V26: Weekly Signal Analysis (Sundays at 02:00)
            scheduler.add_job(
                run_weekly_analysis,
                'cron', day_of_week='sun', hour=2, minute=0,
            )
            scheduler.start()
            logger.info("📅 Scheduled: ML Retraining (Sun 00:00) + Signal Analysis (Sun 02:00)")
            
            # Staggered startup to avoid hitting Binance REST API rate limit and WSS connection limit
            async def _staggered_start(symbol, index):
                await asyncio.sleep(index * 0.3)  # 300ms delay between each pair startup
                # Run the main run_cycle WSS loop, plus the two auxiliary streams
                await asyncio.gather(
                    run_pair_wss(symbol, provider, trader, social_manager, macro_manager, ml_filter),
                    provider.stream_depth(symbol),
                    provider.stream_agg_trades(symbol)
                )

            # Run persistent WSS tasks concurrently indefinitely
            tasks = []
            for i, sym in enumerate(top_pairs):
                tasks.append(_staggered_start(sym, i))
            
            tasks.append(_statarb_monitor_loop()) # V24 Phase 4: Background pairs monitor
            
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
