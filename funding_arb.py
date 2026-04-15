"""
Hunter V34 — Delta-Neutral Funding Arbitrage
==============================================
Captures funding rate yield without directional exposure.

Strategy:
    1. Monitor funding rates across all traded pairs
    2. When funding > THRESHOLD (e.g. +0.05% per 8h = 228% APR):
       - BUY spot equivalent
       - SHORT perpetual futures (1x leverage)
       - Delta = 0, all risk neutralized
    3. Collect funding payments every 8 hours
    4. Close both legs when funding normalizes

Note: This strategy requires SPOT trading capability.
      Currently operates as a signal generator for the StrategyRouter.
      Full SPOT+PERP execution requires V35 multi-venue support.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("hunter.funding_arb")

# Minimum annualized yield to trigger an arb position
MIN_FUNDING_RATE = 0.0005  # 0.05% per 8h = ~22.8% APR
MAX_FUNDING_RATE = 0.01    # 1% per 8h = absurdly high, likely delisting risk

# Funding settlement times (UTC hours)
FUNDING_SETTLEMENT_HOURS = [0, 8, 16]


class FundingArbEngine:
    """
    V34 Delta-Neutral Funding Rate Arbitrage Engine.
    
    Monitors funding rates and generates signals for the StrategyRouter
    when profitable delta-neutral positions are available.
    """
    
    def __init__(self):
        self.active_arbs: Dict[str, Dict[str, Any]] = {}  # symbol -> arb state
        self.cumulative_funding_pnl: float = 0.0
        self.total_settlements: int = 0
    
    def evaluate_funding_opportunity(
        self,
        symbol: str,
        funding_rate: float,
        current_price: float,
        open_interest_usd: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate whether a funding rate arbitrage opportunity exists.
        
        Args:
            symbol: Trading pair (e.g., "ETHUSDT")
            funding_rate: Current 8-hour funding rate (e.g., 0.001 = 0.1%)
            current_price: Current mark price
            open_interest_usd: Total OI in USD (for liquidity check)
            
        Returns:
            Dict with arb signal if opportunity exists, None otherwise.
        """
        abs_rate = abs(funding_rate)
        
        # Filter: Rate must be meaningful but not suspiciously high
        if abs_rate < MIN_FUNDING_RATE or abs_rate > MAX_FUNDING_RATE:
            return None
        
        # Annualized yield calculation
        # Funding is paid 3x per day (every 8h)
        annual_yield = abs_rate * 3 * 365 * 100  # As percentage
        
        # Direction: if funding > 0, longs pay shorts → we SHORT perp + BUY spot
        # If funding < 0, shorts pay longs → we LONG perp + SELL spot
        direction = "SHORT_PERP" if funding_rate > 0 else "LONG_PERP"
        
        signal = {
            "symbol": symbol,
            "action": "FUNDING_ARB",
            "direction": direction,
            "funding_rate": funding_rate,
            "annual_yield_pct": round(annual_yield, 1),
            "price": current_price,
            "confidence": min(0.95, 0.5 + abs_rate * 100),  # Higher rate = higher confidence
        }
        
        logger.info(
            "💰 FUNDING ARB [%s]: %s funding=%.4f%% → APR=%.1f%% direction=%s",
            symbol, "OPEN" if symbol not in self.active_arbs else "MONITOR",
            funding_rate * 100, annual_yield, direction
        )
        
        return signal
    
    def track_funding_settlement(
        self,
        symbol: str,
        funding_rate: float,
        position_size_usd: float,
    ) -> float:
        """
        Track a funding rate settlement event.
        
        Returns the funding PnL for this settlement.
        """
        # If we're short perp and funding > 0, we RECEIVE funding
        # If we're long perp and funding < 0, we RECEIVE funding
        # The actual direction is handled by the position manager
        
        funding_pnl = position_size_usd * funding_rate
        self.cumulative_funding_pnl += funding_pnl
        self.total_settlements += 1
        
        logger.info(
            "📊 FUNDING SETTLEMENT [%s]: rate=%.4f%% size=$%.2f → PnL=$%.4f | Cumulative=$%.2f",
            symbol, funding_rate * 100, position_size_usd, funding_pnl, self.cumulative_funding_pnl
        )
        
        return funding_pnl
    
    def get_top_opportunities(
        self,
        funding_rates: Dict[str, float],
        n: int = 5,
    ) -> List[Tuple[str, float, float]]:
        """
        Rank all symbols by absolute funding rate.
        Returns top N (symbol, rate, annualized_yield).
        """
        ranked = []
        for symbol, rate in funding_rates.items():
            abs_rate = abs(rate)
            if abs_rate >= MIN_FUNDING_RATE:
                apr = abs_rate * 3 * 365 * 100
                ranked.append((symbol, rate, apr))
        
        ranked.sort(key=lambda x: abs(x[1]), reverse=True)
        return ranked[:n]
    
    def should_close_arb(self, symbol: str, current_funding_rate: float) -> bool:
        """Check if funding has normalized enough to close the arb position."""
        return abs(current_funding_rate) < MIN_FUNDING_RATE * 0.5

    # ─────────────────────────────────────────────────────────
    # V36: Funding Regime Detection (from Vibe-Trading)
    # ─────────────────────────────────────────────────────────

    def detect_funding_regime(
        self,
        funding_history: List[float],
    ) -> Dict[str, Any]:
        """
        V36: Classify the current funding regime based on rolling statistics.

        Regimes:
          - CONTANGO: funding persistently positive → longs pay shorts → bearish bias
          - BACKWARDATION: funding persistently negative → shorts pay longs → bullish bias
          - NEUTRAL: funding oscillating around zero
          - EXTREME_CONTANGO: funding > 2σ → strong bearish, likely overleveraged longs
          - EXTREME_BACKW: funding < -2σ → strong bullish, shorts squeezable

        Args:
            funding_history: List of recent 8h funding rates (at least 21 values).

        Returns:
            Dict with regime, mean, std, zscore, bias.
        """
        import numpy as np

        if len(funding_history) < 7:
            return {"regime": "UNKNOWN", "bias": 0.0, "mean": 0.0, "std": 0.0, "zscore": 0.0}

        arr = np.array(funding_history, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std())
        current = arr[-1]
        zscore = (current - mean) / (std + 1e-10)

        if zscore > 2.0:
            regime = "EXTREME_CONTANGO"
            bias = -0.8  # Strong bearish bias
        elif zscore < -2.0:
            regime = "EXTREME_BACKW"
            bias = 0.8   # Strong bullish bias
        elif mean > MIN_FUNDING_RATE * 0.3:
            regime = "CONTANGO"
            bias = -0.3
        elif mean < -MIN_FUNDING_RATE * 0.3:
            regime = "BACKWARDATION"
            bias = 0.3
        else:
            regime = "NEUTRAL"
            bias = 0.0

        return {
            "regime": regime,
            "bias": round(bias, 4),
            "mean": round(mean, 6),
            "std": round(std, 6),
            "zscore": round(zscore, 4),
            "current": round(current, 6),
        }

    def oi_funding_signal(
        self,
        funding_rate: float,
        oi_change_pct: float,
        price_change_pct: float,
    ) -> Dict[str, Any]:
        """
        V36: OI × Funding matrix signal.

        Detects divergence between OI/funding direction and price action:
        - OI rising + funding rising + price flat/dropping → overleveraged longs, SHORT signal
        - OI rising + funding dropping + price flat/rising → overleveraged shorts, LONG signal
        - OI dropping + funding normalizing → wash-out complete, trend continuation

        Args:
            funding_rate: Current 8h funding rate.
            oi_change_pct: OI change % over last period.
            price_change_pct: Price change % over same period.

        Returns:
            Dict with signal(-1/0/1), description, and confidence.
        """
        signal = 0
        desc = "neutral"
        conf = 0.0

        # Case 1: OI surging + funding rising + price stalling → bearish divergence
        if oi_change_pct > 3.0 and funding_rate > MIN_FUNDING_RATE and price_change_pct < 1.0:
            signal = -1
            desc = "OI+Funding diverge: overleveraged longs"
            conf = min(0.8, 0.4 + abs(funding_rate) * 50)

        # Case 2: OI surging + funding dropping + price stalling → bullish divergence
        elif oi_change_pct > 3.0 and funding_rate < -MIN_FUNDING_RATE and price_change_pct > -1.0:
            signal = 1
            desc = "OI+Funding diverge: overleveraged shorts"
            conf = min(0.8, 0.4 + abs(funding_rate) * 50)

        # Case 3: OI flush (dropping) + funding normalizing → end of liquidation
        elif oi_change_pct < -5.0 and abs(funding_rate) < MIN_FUNDING_RATE * 0.5:
            signal = 0
            desc = "OI flush: liquidation complete, wait for setup"
            conf = 0.3

        return {
            "signal": signal,
            "description": desc,
            "confidence": round(conf, 4),
            "oi_change_pct": round(oi_change_pct, 2),
            "funding_rate": round(funding_rate, 6),
            "price_change_pct": round(price_change_pct, 2),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return cumulative funding arb statistics."""
        return {
            "cumulative_funding_pnl": round(self.cumulative_funding_pnl, 4),
            "total_settlements": self.total_settlements,
            "active_arbs": len(self.active_arbs),
        }
