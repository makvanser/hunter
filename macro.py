"""
Hunter V16 — Macro Intelligence Module
======================================
Adds Bitcoin dominance fetch (via CoinGecko) and 
asset correlation checks (vs BTC) to gauge macro market risks.
"""

import asyncio
import logging
import math
import time
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger("hunter.macro")


def get_pearson_correlation(x: List[float], y: List[float]) -> float:
    """Calculates Pearson correlation coefficient between two equal-length lists."""
    if not x or not y or len(x) != len(y) or len(x) < 2:
        return 0.0

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)

    denominator = math.sqrt(sum_sq_x * sum_sq_y)
    
    if denominator == 0:
        return 0.0
        
    return numerator / denominator


class MacroManager:
    """
    Handles global macro data like BTC Dominance (CoinGecko) 
    and BTC price correlation (Binance).
    """

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.btc_dom_cache: float = 50.0  # Fallback
        self.last_dom_fetch: float = 0.0
        self.DOM_CACHE_TTL = 3600 * 12  # 12 hours cache for CoinGecko 429 safety
        
        # Cache for BTC OHLCV to save Binance calls if comparing multiple symbols
        self.btc_closes_cache: List[float] = []
        self.last_btc_fetch: float = 0.0
        self.BTC_CACHE_TTL = 3600  # 1 hour

    async def init_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={"User-Agent": "HunterBot/16.0"}
            )

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def get_btc_dominance(self) -> float:
        """Fetch BTC dominance % from CoinGecko global API."""
        now = time.time()
        if now - self.last_dom_fetch < self.DOM_CACHE_TTL and self.btc_dom_cache != 50.0:
            return self.btc_dom_cache

        await self.init_session()
        url = "https://api.coingecko.com/api/v3/global"
        
        try:
            async with self.session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    dom = data.get("data", {}).get("market_cap_percentage", {}).get("btc")
                    if dom:
                        self.btc_dom_cache = float(dom)
                        self.last_dom_fetch = now
                        logger.info("🌍 BTC Dominance fetched: %.2f%%", self.btc_dom_cache)
                elif resp.status == 429:
                    logger.warning("⚠️ CoinGecko rate limit (429). Using cached dominance: %.2f%%", self.btc_dom_cache)
                    # Extend cache TTL so we don't spam 429
                    self.last_dom_fetch = now
                else:
                    logger.warning("⚠️ CoinGecko returned %s. Using cached %s%%", resp.status, self.btc_dom_cache)
        except Exception as e:
            logger.error("❌ Failed to fetch BTC Dominance: %s", e)

        return self.btc_dom_cache

    async def get_btc_correlation(self, symbol: str, provider, limit: int = 50) -> float:
        """
        Calculates Pearson correlation between the given symbol and BTCUSDT.
        Requires the `BinanceProvider` to fetch OHLCV.
        Returns a float between -1.0 and 1.0.
        """
        if symbol == "BTCUSDT":
            return 1.0

        now = time.time()
        
        # 1. Fetch BTC data (cached for this hour)
        if not self.btc_closes_cache or (now - self.last_btc_fetch) > self.BTC_CACHE_TTL:
            try:
                # Use provider's fetch_ohlcv
                _, _, btc_closes, _ = await provider.fetch_ohlcv("BTCUSDT", limit=limit)
                self.btc_closes_cache = btc_closes
                self.last_btc_fetch = now
            except Exception as e:
                logger.warning("⚠️ Failed to fetch BTC closes for correlation: %s", e)
                return 0.0
                
        # 2. Fetch Symbol data
        try:
            _, _, sym_closes, _ = await provider.fetch_ohlcv(symbol, limit=limit)
        except Exception as e:
            logger.warning("⚠️ Failed to fetch %s closes for correlation: %s", symbol, e)
            return 0.0

        # Align series lengths just in case of new pairs
        min_len = min(len(self.btc_closes_cache), len(sym_closes))
        if min_len < 10:
            return 0.0

        # Note: take the most recent 'min_len' bars
        btc_sliced = self.btc_closes_cache[-min_len:]
        sym_sliced = sym_closes[-min_len:]

        correlation = get_pearson_correlation(btc_sliced, sym_sliced)
        logger.info("🔗 %s correlation to BTC: %+.2f", symbol, correlation)
        return correlation
