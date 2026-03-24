"""
Hunter V29 — Portfolio Risk & Correlation Manager
=================================================
Evaluates proposed trades against the existing portfolio to 
prevent catastrophic correlated liquidations.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("hunter.portfolio")

# Import the math function from macro.py
from macro import get_pearson_correlation

class PortfolioManager:
    def __init__(self):
        self.closes_cache: Dict[str, Tuple[float, List[float]]] = {} # symbol -> (timestamp, closes)
        self.CACHE_TTL = 3600  # 1 hour
        self.CORRELATION_THRESHOLD = 0.85

    async def _get_closes(self, symbol: str, provider, limit: int = 50) -> List[float]:
        """Fetch closes with caching."""
        now = time.time()
        if symbol in self.closes_cache:
            last_fetch, closes = self.closes_cache[symbol]
            if now - last_fetch < self.CACHE_TTL:
                return closes
        
        try:
            _, _, closes, _ = await provider.fetch_ohlcv(symbol, limit=limit)
            self.closes_cache[symbol] = (now, closes)
            return closes
        except Exception as e:
            logger.warning("⚠️ Failed to fetch OHLCV for correlation check %s: %s", symbol, e)
            return []

    async def check_trade_correlation(
        self, new_symbol: str, target_side: str, current_positions: Dict[str, Dict], provider
    ) -> bool:
        """
        Returns True if the trade is safe (uncorrelated or hedges).
        Returns False if the trade increases highly correlated directional risk.
        """
        if not current_positions:
            return True  # Safe if portfolio is empty
            
        # Get target asset's history
        new_closes = await self._get_closes(new_symbol, provider)
        if not new_closes:
            return True # Fallback: allow trade if data missing

        for open_sym, pos in current_positions.items():
            # If we are adding the exact same position (DCA), skip correlation block
            if open_sym == new_symbol:
                continue
                
            open_closes = await self._get_closes(open_sym, provider)
            if not open_closes:
                continue
                
            min_len = min(len(new_closes), len(open_closes))
            if min_len < 20:
                continue
                
            new_slice = new_closes[-min_len:]
            open_slice = open_closes[-min_len:]
            
            corr = get_pearson_correlation(new_slice, open_slice)
            
            # If assets are highly correlated AND directions are the same = BAD
            if corr >= self.CORRELATION_THRESHOLD and target_side == pos["side"]:
                logger.warning(
                    "🛑 PORTFOLIO RISK: Blocked %s %s. Correlated to open %s %s (corr=%.2f ≥ %.2f)",
                    target_side, new_symbol, pos["side"], open_sym, corr, self.CORRELATION_THRESHOLD
                )
                return False
                
            # If assets are highly ANTI-correlated AND directions are DIFFERENT = BAD (Synthetic same direction)
            if corr <= -self.CORRELATION_THRESHOLD and target_side != pos["side"]:
                logger.warning(
                    "🛑 PORTFOLIO RISK: Blocked %s %s. Anti-correlated to open %s %s creates double-exposure (corr=%.2f ≤ -%.2f)",
                    target_side, new_symbol, pos["side"], open_sym, corr, self.CORRELATION_THRESHOLD
                )
                return False

        return True
