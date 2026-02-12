"""
Hunter V13 — Main Loop
========================
Orchestrator: fetches REAL data from Binance Futures API → analyses → decides → executes.

V13: Integrated NewsManager for sentiment-aware signal generation.

Modes:
  - Auto Mode  :  python main.py             → scans top pairs in a loop
  - Manual Mode:  python main.py --symbol ETHUSDT  → single analysis, then exit
"""

import argparse
import json
import logging
import ssl
import sys
import time
import urllib.request
import urllib.error
from typing import Dict, List, Tuple, Optional

from config import (
    BASE_URL,
    BLACKLIST,
    CHECK_INTERVAL_SEC,
    TIMEFRAME,
    KLINE_LIMIT,
    TOP_PAIRS_COUNT,
    ADX_PERIOD,
    RSI_PERIOD,
    BB_PERIOD,
)
from analysis import (
    compute_adx,
    compute_bollinger,
    compute_rsi,
    generate_signal,
    get_market_regime,
)
from execution import PaperTrader
from news import NewsManager

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hunter.main")

# Reusable SSL context (Binance requires TLS)
_ssl_ctx = ssl.create_default_context()


# ─────────────────────────────────────────────────────────────
# HTTP helper
# ─────────────────────────────────────────────────────────────
def _api_get(endpoint: str, params: Optional[Dict] = None) -> any:
    """
    GET request to Binance Futures API.
    Returns parsed JSON.
    """
    url = f"{BASE_URL}{endpoint}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"

    req = urllib.request.Request(url, headers={"User-Agent": "HunterV13/1.0"})
    with urllib.request.urlopen(req, context=_ssl_ctx, timeout=15) as resp:
        return json.loads(resp.read().decode())


# ─────────────────────────────────────────────────────────────
# Real Data Fetching
# ─────────────────────────────────────────────────────────────
def fetch_real_ohlcv(
    symbol: str,
    interval: str = TIMEFRAME,
    limit: int = KLINE_LIMIT,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Fetch OHLCV candles from Binance Futures /fapi/v1/klines.

    Returns (highs, lows, closes, volumes) as lists of floats.
    Raises on network / API error.
    """
    data = _api_get("/fapi/v1/klines", {
        "symbol": symbol,
        "interval": interval,
        "limit": str(limit),
    })

    # Kline format: [openTime, open, high, low, close, volume, ...]
    highs   = [float(k[2]) for k in data]
    lows    = [float(k[3]) for k in data]
    closes  = [float(k[4]) for k in data]
    volumes = [float(k[5]) for k in data]

    return highs, lows, closes, volumes


def fetch_long_short_ratio(symbol: str) -> float:
    """
    Fetch the global Long/Short account ratio from Binance Futures.
    Endpoint: /futures/data/globalLongShortAccountRatio

    Returns the latest ratio as a float (< 1 means more shorts).
    Falls back to 1.0 on error (neutral).
    """
    try:
        data = _api_get("/futures/data/globalLongShortAccountRatio", {
            "symbol": symbol,
            "period": "1h",
            "limit": "1",
        })
        if data:
            return float(data[0]["longShortRatio"])
    except Exception as exc:
        logger.warning("⚠️  L/S ratio fetch failed for %s: %s", symbol, exc)
    return 1.0  # Neutral fallback


def fetch_whale_net_volume(symbol: str) -> float:
    """
    Approximate whale activity using Binance Futures taker buy/sell volume.
    Endpoint: /futures/data/takerlongshortRatio

    whale_proxy = takerBuyVol - takerSellVol
    Positive = net buying pressure (bullish whale proxy).
    Falls back to 0.0 on error (neutral).
    """
    try:
        data = _api_get("/futures/data/takerlongshortRatio", {
            "symbol": symbol,
            "period": "1h",
            "limit": "1",
        })
        if data:
            buy_vol  = float(data[0].get("buyVol", 0))
            sell_vol = float(data[0].get("sellVol", 0))
            return buy_vol - sell_vol
    except Exception as exc:
        logger.warning("⚠️  Whale volume fetch failed for %s: %s", symbol, exc)
    return 0.0  # Neutral fallback


# ─────────────────────────────────────────────────────────────
# Dynamic Pair Scanner
# ─────────────────────────────────────────────────────────────
def scan_top_pairs(count: int = TOP_PAIRS_COUNT) -> List[str]:
    """
    Fetch the top USDT-margined futures pairs by 24h quote volume.

    Steps:
    1. GET /fapi/v1/ticker/24hr  → all tickers
    2. Filter to *USDT pairs only
    3. Remove blacklisted stablecoins
    4. Sort by quoteVolume descending
    5. Return top `count` symbols
    """
    logger.info("🔍 Scanning top %d futures pairs by 24h volume …", count)
    try:
        tickers = _api_get("/fapi/v1/ticker/24hr")
    except Exception as exc:
        logger.error("❌ Failed to fetch tickers: %s", exc)
        return ["BTCUSDT"]  # Safe fallback

    # Filter USDT pairs, exclude blacklist
    usdt_pairs = [
        t for t in tickers
        if t["symbol"].endswith("USDT")
        and t["symbol"] not in BLACKLIST
    ]

    # Sort by 24h quote volume (descending)
    usdt_pairs.sort(key=lambda t: float(t.get("quoteVolume", 0)), reverse=True)

    top = [t["symbol"] for t in usdt_pairs[:count]]
    logger.info("📋 Top pairs: %s", ", ".join(top))
    return top


# ─────────────────────────────────────────────────────────────
# Single Cycle (for one symbol)
# ─────────────────────────────────────────────────────────────
def run_cycle(trader: PaperTrader, symbol: str, news_manager: NewsManager) -> Dict:
    """Execute one analysis-and-trade cycle for a given symbol."""

    logger.info("─" * 50)
    logger.info("▶ Analysing %s", symbol)

    # 1. Fetch real candles
    try:
        highs, lows, closes, volumes = fetch_real_ohlcv(symbol)
    except Exception as exc:
        logger.error("❌ OHLCV fetch failed for %s: %s", symbol, exc)
        return {"action": "FETCH_ERROR", "symbol": symbol, "error": str(exc)}

    current_price = closes[-1]

    # 2. Market Regime Filter (ADX)
    adx = compute_adx(highs, lows, closes, ADX_PERIOD)
    regime = get_market_regime(adx)
    logger.info(
        "📊 %s @ $%.4f | ADX=%.2f → %s", symbol, current_price, adx, regime
    )

    if regime == "CHOPPY":
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
    upper, middle, lower = compute_bollinger(closes, BB_PERIOD)
    logger.info(
        "   RSI=%.2f | BB=[%.2f / %.2f / %.2f]", rsi, lower, middle, upper
    )

    # 4. External data (real Binance)
    ls_ratio = fetch_long_short_ratio(symbol)
    whale_vol = fetch_whale_net_volume(symbol)
    logger.info("   L/S Ratio=%.4f | Whale Net Vol=%.2f", ls_ratio, whale_vol)

    # 5. News Sentiment (V13)
    sentiment = news_manager.get_sentiment(symbol)
    logger.info("   📰 News Sentiment = %s", sentiment)

    # 6. Generate signal (V13: pass sentiment)
    signal = generate_signal(rsi, ls_ratio, whale_vol, regime, sentiment)
    logger.info("   🎯 Signal = %s", signal)

    # 7. Execute (multi-asset: pass symbol)
    result = trader.execute_trade(signal, current_price, symbol)
    logger.info("   ➜ Result: %s", result["action"])
    return result


# ─────────────────────────────────────────────────────────────
# CLI Argument Parser
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hunter V13 — Contrarian Crypto Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     Auto mode — scan top pairs in loop
  python main.py --symbol ETHUSDT    Manual mode — single analysis & exit
        """,
    )
    parser.add_argument(
        "--symbol", "-s",
        type=str,
        default=None,
        help="Run a SINGLE analysis for this symbol and exit (Manual Mode).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Entry Points
# ─────────────────────────────────────────────────────────────
def run_manual(symbol: str) -> None:
    """Manual Mode: analyse one symbol, print results, exit."""
    logger.info("=" * 60)
    logger.info("  HUNTER V13 — Manual Mode")
    logger.info("  Symbol: %s", symbol)
    logger.info("=" * 60)

    trader = PaperTrader()
    news_manager = NewsManager()

    # Log Fear & Greed at startup
    fng_value, fng_class = news_manager.get_fear_and_greed()
    logger.info("  😱 Fear & Greed: %d (%s)", fng_value, fng_class)

    result = run_cycle(trader, symbol, news_manager)

    logger.info("=" * 60)
    logger.info("  Result: %s", json.dumps(result, indent=2, default=str))
    logger.info("=" * 60)


def run_auto() -> None:
    """Auto Mode: scan top pairs + held positions, loop forever."""
    logger.info("=" * 60)
    logger.info("  HUNTER V13 — Auto Mode (Portfolio Aware + News Sentiment)")
    logger.info("=" * 60)

    trader = PaperTrader()
    news_manager = NewsManager()

    while True:
        try:
            # Log Fear & Greed at the start of each cycle
            fng_value, fng_class = news_manager.get_fear_and_greed()
            logger.info("😱 Fear & Greed: %d (%s)", fng_value, fng_class)

            # 1. Get top market pairs by volume
            market_top = scan_top_pairs()

            # 2. Merge with currently held positions (so we never abandon them)
            portfolio_symbols = list(trader.positions.keys())

            # 3. Deduplicate via set
            targets = list(set(market_top + portfolio_symbols))

            logger.info("📋 Cycle targets (%d): %s", len(targets), ", ".join(targets))

            for sym in targets:
                try:
                    run_cycle(trader, sym, news_manager)
                    # Small pause between symbols to respect API rate limits
                    time.sleep(1)
                except Exception as exc:
                    logger.exception("Error analysing %s: %s", sym, exc)

        except KeyboardInterrupt:
            logger.info("🛑 Shutting down gracefully …")
            sys.exit(0)
        except Exception as exc:
            logger.exception("Unhandled error in auto-cycle: %s", exc)

        logger.info(
            "💤 Sleeping %d seconds before next scan …\n", CHECK_INTERVAL_SEC
        )
        time.sleep(CHECK_INTERVAL_SEC)


def main() -> None:
    args = parse_args()

    if args.symbol:
        run_manual(args.symbol.upper())
    else:
        run_auto()


if __name__ == "__main__":
    main()
