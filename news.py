"""
Hunter V13 — News Sentiment Module
=====================================
Global-polling CryptoPanic news cache + keyword-based sentiment analysis.

Strategy:
  - Fetch the global CryptoPanic feed ONCE every 15 minutes (free-tier safe).
  - Cache all results locally.
  - For each symbol, search the cache and apply keyword NLP.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import requests

from config import CRYPTOPANIC_API_KEY, NEWS_POLL_INTERVAL_SEC

logger = logging.getLogger("hunter.news")

# ── Keyword dictionaries for basic NLP ──────────────────────
BULLISH_KEYWORDS = [
    "partnership", "launch", "mainnet", "blackrock",
    "etf", "integration", "approval", "upgrade",
]
BEARISH_KEYWORDS = [
    "hack", "exploit", "delist", "ban",
    "lawsuit", "breach", "scam", "fraud",
]

# ── Common symbol-to-name mapping for title search ──────────
SYMBOL_NAMES: Dict[str, List[str]] = {
    "BTC": ["bitcoin"],
    "ETH": ["ethereum"],
    "BNB": ["binance coin", "bnb"],
    "SOL": ["solana"],
    "XRP": ["ripple", "xrp"],
    "ADA": ["cardano"],
    "DOGE": ["dogecoin"],
    "AVAX": ["avalanche"],
    "DOT": ["polkadot"],
    "MATIC": ["polygon"],
    "LINK": ["chainlink"],
    "SHIB": ["shiba"],
    "LTC": ["litecoin"],
    "UNI": ["uniswap"],
    "ATOM": ["cosmos"],
}


class NewsManager:
    """
    CryptoPanic news manager with global polling cache.

    Free-tier constraint: max ~5 requests/hour.
    We fetch the main feed once every NEWS_POLL_INTERVAL_SEC (default 900s = 15 min)
    and search locally per symbol.
    """

    def __init__(self) -> None:
        self.api_key: str = CRYPTOPANIC_API_KEY
        self.cache: List[Dict] = []
        self.last_update: float = 0.0  # epoch timestamp

        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            logger.warning(
                "⚠️  CRYPTOPANIC_API_KEY not set — news sentiment disabled."
            )

    # ─────────────────────────────────────────────────────────
    # Cache Management
    # ─────────────────────────────────────────────────────────
    def update_cache(self) -> None:
        """Fetch global CryptoPanic feed if poll interval has elapsed."""
        now = time.time()
        if now - self.last_update < NEWS_POLL_INTERVAL_SEC:
            return  # Cache is still fresh

        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            return  # No key configured

        url = (
            f"https://cryptopanic.com/api/v1/posts/"
            f"?auth_token={self.api_key}&public=true"
        )
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            self.cache = data.get("results", [])
            self.last_update = now
            logger.info(
                "📰 News cache updated: %d articles fetched.", len(self.cache)
            )
        except requests.RequestException as exc:
            logger.warning("⚠️  CryptoPanic fetch failed: %s", exc)

    # ─────────────────────────────────────────────────────────
    # Per-Symbol Sentiment
    # ─────────────────────────────────────────────────────────
    def get_sentiment(self, symbol: str) -> str:
        """
        Return 'BULLISH', 'BEARISH', or 'NEUTRAL' for a given symbol.

        1. Refresh cache if stale.
        2. Filter news matching this symbol (currency tag or title keyword).
        3. Apply keyword NLP to determine sentiment.
        """
        self.update_cache()

        # Strip trailing "USDT" etc. → e.g. "BTCUSDT" → "BTC"
        base = symbol.replace("USDT", "").replace("BUSD", "").upper()

        # Build search terms: base ticker + common names
        search_terms = [base.lower()]
        if base in SYMBOL_NAMES:
            search_terms.extend(SYMBOL_NAMES[base])

        # Filter news relevant to this symbol
        relevant: List[Dict] = []
        for article in self.cache:
            # Check currency tags from CryptoPanic API
            currencies = article.get("currencies", []) or []
            currency_codes = [
                c.get("code", "").upper() for c in currencies
            ]
            if base in currency_codes:
                relevant.append(article)
                continue

            # Check title for keyword match
            title = (article.get("title") or "").lower()
            if any(term in title for term in search_terms):
                relevant.append(article)

        if not relevant:
            return "NEUTRAL"

        # Keyword NLP scoring
        bullish_count = 0
        bearish_count = 0

        for article in relevant:
            title = (article.get("title") or "").lower()
            if any(kw in title for kw in BULLISH_KEYWORDS):
                bullish_count += 1
            if any(kw in title for kw in BEARISH_KEYWORDS):
                bearish_count += 1

        if bullish_count > bearish_count:
            logger.info(
                "📈 %s sentiment: BULLISH (%d bullish / %d bearish in %d articles)",
                symbol, bullish_count, bearish_count, len(relevant),
            )
            return "BULLISH"
        elif bearish_count > bullish_count:
            logger.info(
                "📉 %s sentiment: BEARISH (%d bullish / %d bearish in %d articles)",
                symbol, bullish_count, bearish_count, len(relevant),
            )
            return "BEARISH"

        return "NEUTRAL"

    # ─────────────────────────────────────────────────────────
    # Fear & Greed Index
    # ─────────────────────────────────────────────────────────
    def get_fear_and_greed(self) -> Tuple[int, str]:
        """
        Fetch the Crypto Fear & Greed Index from Alternative.me.

        Returns (value: 0-100, classification: str).
        Falls back to (50, 'Neutral') on error.
        """
        try:
            resp = requests.get(
                "https://api.alternative.me/fng/", timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            entry = data["data"][0]
            value = int(entry["value"])
            classification = entry["value_classification"]
            logger.info(
                "😱 Fear & Greed Index: %d (%s)", value, classification
            )
            return value, classification
        except Exception as exc:
            logger.warning("⚠️  Fear & Greed fetch failed: %s", exc)
            return 50, "Neutral"
