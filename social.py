"""
Hunter V15 — Social & Sentiment Intelligence
=============================================
Unifies Fear & Greed index, CryptoPanic news sentiment, 
and Google Trends search momentum into a single `social_score`.
"""

import logging
import time
from typing import Dict, Tuple

from news import NewsManager

try:
    from pytrends.request import TrendReq
except ImportError:
    TrendReq = None

logger = logging.getLogger("hunter.social")


class SocialManager:
    """
    Orchestrates multiple social/sentiment APIs and returns a normalized
    score between -1.0 (Extreme Bearish) and +1.0 (Extreme Bullish).
    """

    def __init__(self) -> None:
        self.news_manager = NewsManager()
        self.pytrends = TrendReq(hl="en-US", tz=360) if TrendReq else None
        
        # Cache for Google Trends to avoid 429 Too Many Requests
        self.trends_cache: Dict[str, Tuple[float, float]] = {}
        self.TRENDS_CACHE_TTL = 3600 * 12  # Cache for 12 hours
        
        logger.info("SocialManager initialized (PyTrends: %s)", "OK" if self.pytrends else "MISSING")

    def get_google_trends_score(self, symbol: str) -> float:
        """
        Returns a score between -1.0 and 1.0 based on Google Trends 7-day momentum.
        Positive score means search popularity is increasing.
        """
        if not self.pytrends:
            return 0.0
            
        now = time.time()
        if symbol in self.trends_cache:
            score, ts = self.trends_cache[symbol]
            if now - ts < self.TRENDS_CACHE_TTL:
                return score
                
        base = symbol.replace("USDT", "").replace("BUSD", "").upper()
        # Fallbacks for common cryptos to their actual names for better search volume
        name_map = {
            "BTC": "Bitcoin",
            "ETH": "Ethereum",
            "SOL": "Solana",
            "BNB": "Binance Coin",
            "XRP": "Ripple",
            "ADA": "Cardano",
            "DOGE": "Dogecoin"
        }
        search_kw = name_map.get(base, base)
        
        try:
            self.pytrends.build_payload([search_kw], cat=0, timeframe="now 7-d", geo="", gprop="")
            data = self.pytrends.interest_over_time()
            if data.empty:
                score = 0.0
            else:
                # Compare the average of the last 48 hours vs the preceding 5 days
                series = data[search_kw]
                if len(series) > 48:
                    recent_avg = series[-48:].mean()
                    past_avg = series[:-48].mean()
                    if past_avg == 0:
                        score = 0.5 if recent_avg > 0 else 0.0
                    else:
                        pct_change = (recent_avg - past_avg) / past_avg
                        # Clamp pct_change to [-1.0, 1.0]
                        score = max(-1.0, min(1.0, pct_change))
                else:
                    score = 0.0
            logger.info("🔍 Google Trends '%s' momentum: %+.2f", search_kw, score)
        except Exception as e:
            logger.warning("⚠️ Google Trends fetch failed for %s: %s", search_kw, e)
            score = 0.0
            
        self.trends_cache[symbol] = (score, now)
        return score

    def get_social_score(self, symbol: str) -> float:
        """
        Aggregates Fear&Greed, CryptoPanic news, and Google Trends into a single score [-1.0, 1.0].
        - News Sentiment (50% weight) - specific to the coin.
        - Fear & Greed (30% weight) - broader market sentiment.
        - Google Trends (20% weight) - retail interest momentum.
        """
        # 1. Fear and Greed (0 to 100 -> -1.0 to 1.0)
        fg_value, _ = self.news_manager.get_fear_and_greed()
        fg_score = (fg_value - 50) / 50.0  
        
        # 2. CryptoPanic News (BULLISH=+0.8, BEARISH=-0.8, NEUTRAL=0.0)
        news_label = self.news_manager.get_sentiment(symbol)
        if news_label == "BULLISH":
            news_score = 0.8
        elif news_label == "BEARISH":
            news_score = -0.8
        else:
            news_score = 0.0
            
        # 3. Google Trends Momentum
        trends_score = self.get_google_trends_score(symbol)
        
        # Weighted aggregate
        score = (news_score * 0.5) + (fg_score * 0.3) + (trends_score * 0.2)
        
        return max(-1.0, min(1.0, score))
