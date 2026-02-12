"""
Hunter V13 — News Sentiment Module (Patched for API V2)
=======================================================
Fixes 404 Error by using the correct /api/developer/v2/ endpoint.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
import requests
from config import CRYPTOPANIC_API_KEY, NEWS_POLL_INTERVAL_SEC

logger = logging.getLogger("hunter.news")

# ── Keyword dictionaries ────────────────────────────────────
BULLISH_KEYWORDS = [
    "partnership", "launch", "mainnet", "blackrock",
    "etf", "integration", "approval", "upgrade",
]
BEARISH_KEYWORDS = [
    "hack", "exploit", "delist", "ban",
    "lawsuit", "breach", "scam", "fraud",
]

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
    def __init__(self) -> None:
        self.api_key: str = CRYPTOPANIC_API_KEY
        self.cache: List[Dict] = []
        self.last_update: float = 0.0

        if not self.api_key or "YOUR_KEY" in self.api_key:
            logger.warning("⚠️ CRYPTOPANIC_API_KEY not set.")

    def update_cache(self) -> None:
        """Fetch global CryptoPanic feed (V2 API)."""
        now = time.time()
        if now - self.last_update < NEWS_POLL_INTERVAL_SEC:
            return

        if not self.api_key or "YOUR_KEY" in self.api_key:
            return

        # 🛑 FIX: Changed from /api/v1/posts/ to /api/developer/v2/posts/
        url = "https://cryptopanic.com/api/developer/v2/posts/"
        params = {
            "auth_token": self.api_key,
            "public": "true",   # Recommended for bots
            "filter": "rising"  # Get only important/rising news
        }

        try:
            resp = requests.get(url, params=params, timeout=15)

            # Debug: если снова ошибка, покажем почему
            if resp.status_code != 200:
                logger.error(
                    "❌ CryptoPanic Error %d: %s", resp.status_code, resp.text
                )
                return

            data = resp.json()
            self.cache = data.get("results", [])
            self.last_update = now
            logger.info(
                "📰 News cache updated: %d articles fetched (V2).",
                len(self.cache),
            )

        except requests.RequestException as exc:
            logger.warning("⚠️ CryptoPanic fetch failed: %s", exc)

    def get_sentiment(self, symbol: str) -> str:
        """Return BULLISH / BEARISH / NEUTRAL for a symbol."""
        self.update_cache()
        base = symbol.replace("USDT", "").replace("BUSD", "").upper()

        search_terms = [base.lower()]
        if base in SYMBOL_NAMES:
            search_terms.extend(SYMBOL_NAMES[base])

        relevant: List[Dict] = []
        for article in self.cache:
            # Check currencies
            currencies = article.get("currencies", []) or []
            codes = [c.get("code", "").upper() for c in currencies]
            if base in codes:
                relevant.append(article)
                continue

            # Check title
            title = (article.get("title") or "").lower()
            if any(term in title for term in search_terms):
                relevant.append(article)

        if not relevant:
            return "NEUTRAL"

        bullish = 0
        bearish = 0
        for article in relevant:
            title = (article.get("title") or "").lower()
            if any(kw in title for kw in BULLISH_KEYWORDS):
                bullish += 1
            if any(kw in title for kw in BEARISH_KEYWORDS):
                bearish += 1

        if bullish > bearish:
            logger.info("📈 %s Sentiment: BULLISH via News", symbol)
            return "BULLISH"
        elif bearish > bullish:
            logger.info("📉 %s Sentiment: BEARISH via News", symbol)
            return "BEARISH"

        return "NEUTRAL"

    def get_fear_and_greed(self) -> Tuple[int, str]:
        """Fetch Crypto Fear & Greed Index. Falls back to (50, 'Neutral')."""
        try:
            resp = requests.get("https://api.alternative.me/fng/", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            item = data["data"][0]
            return int(item["value"]), item["value_classification"]
        except Exception as exc:
            logger.warning("⚠️ Fear & Greed fetch failed: %s", exc)
            return 50, "Neutral"
