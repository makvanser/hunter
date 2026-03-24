"""
Hunter V27 — Strategy Router
==============================
Multi-strategy ensemble. Evaluates multiple independent trading
strategies in parallel and selects the best one based on current
market regime and signal strength.
"""

import logging
from typing import Dict, Any, Tuple, List, Optional
from analysis import MarketState

logger = logging.getLogger("hunter.strategy")


class BaseStrategy:
    """Base class for all discrete strategies."""
    name = "Base"
    best_regimes = []

    def evaluate(self, state: MarketState, current_position: Optional[str] = None) -> Tuple[str, float]:
        """Returns (action, confidence_score_0_to_1)"""
        return "HOLD", 0.0


class GridStrategy(BaseStrategy):
    """
    Excels in CHOPPY regimes (ADX < 20).
    Looks for price extremes within the Bollinger Bands.
    """
    name = "GridBounce"
    best_regimes = ["CHOPPY"]

    def evaluate(self, state: MarketState, current_position: Optional[str] = None) -> Tuple[str, float]:
        # Grid wants to buy near bottom BB, sell near top BB
        action = "HOLD"
        conf = 0.0

        if state.regime not in self.best_regimes:
            return action, conf  # Suppress outside ideal regime

        # bb_position: 0.0 = lower band, 1.0 = upper band, 0.5 = middle
        if state.bb_position < 0.15 and state.rsi < 45:
            action = "BUY"
            conf = max(0.0, (0.25 - state.bb_position) * 4)  # higher conf closer to/below band
        elif state.bb_position > 0.85 and state.rsi > 55:
            action = "SHORT"
            conf = max(0.0, (state.bb_position - 0.75) * 4)

        # Dampen if heavy OBI against us
        if action == "BUY" and state.liq_imbalance < -0.2:
            conf *= 0.5
        elif action == "SHORT" and state.liq_imbalance > 0.2:
            conf *= 0.5

        return action, min(1.0, conf)


class MomentumStrategy(BaseStrategy):
    """
    Excels in TRENDING or STRONG regimes (ADX > 25).
    Follows MACD, RSI slopes, and volume confirmation.
    """
    name = "Momentum"
    best_regimes = ["TRENDING", "STRONG_UP", "STRONG_DOWN"]

    def evaluate(self, state: MarketState, current_position: Optional[str] = None) -> Tuple[str, float]:
        action = "HOLD"
        conf = 0.0

        if state.regime not in self.best_regimes:
            return action, conf

        # Simple momentum alignment: RSI > 50 + MACD > 0 + RSI rising
        if state.regime == "STRONG_UP" or (state.regime == "TRENDING" and state.macd_histogram > 0):
            if state.rsi > 55 and state.rsi_slope > 0 and not state.near_resistance:
                action = "BUY"
                conf = 0.6 + min(0.4, (state.rsi - 50) / 100)

        elif state.regime == "STRONG_DOWN" or (state.regime == "TRENDING" and state.macd_histogram < 0):
            if state.rsi < 45 and state.rsi_slope < 0:
                action = "SHORT"
                conf = 0.6 + min(0.4, (50 - state.rsi) / 100)

        return action, min(1.0, conf)


class MeanReversionStrategy(BaseStrategy):
    """
    Excels in extreme conditions (far from VWAP, over-extended RSI).
    Can trade any regime, but looks for snap-backs.
    """
    name = "MeanReversion"
    best_regimes = ["CHOPPY", "TRENDING", "STRONG_UP", "STRONG_DOWN"]

    def evaluate(self, state: MarketState, current_position: Optional[str] = None) -> Tuple[str, float]:
        action = "HOLD"
        conf = 0.0

        # E.g. price is 3% below VWAP and heavily oversold
        if state.vwap_diff_pct < -2.5 and state.rsi < 25:
            action = "BUY"
            conf = min(1.0, abs(state.vwap_diff_pct) / 5.0)

        elif state.vwap_diff_pct > 2.5 and state.rsi > 75:
            action = "SHORT"
            conf = min(1.0, state.vwap_diff_pct / 5.0)

        return action, conf


class FundingArbStrategy(BaseStrategy):
    """
    Capitalizes on extreme funding rates. Takes the contrarian side.
    """
    name = "FundingArb"
    best_regimes = ["CHOPPY", "TRENDING", "STRONG_UP", "STRONG_DOWN"]

    def evaluate(self, state: MarketState, current_position: Optional[str] = None) -> Tuple[str, float]:
        action = "HOLD"
        conf = 0.0

        # If funding rate is extremely negative (e.g., -0.1%), shorts are paying longs heavily.
        if state.funding_rate < -0.0005:
            action = "BUY"
            conf = min(1.0, abs(state.funding_rate) * 1000)
            
        elif state.funding_rate > 0.0005:
            action = "SHORT"
            conf = min(1.0, state.funding_rate * 1000)

        return action, conf


class StrategyRouter:
    """
    Evaluates all registered strategies and selects the best signal.
    """
    def __init__(self):
        self.strategies: List[BaseStrategy] = [
            GridStrategy(),
            MomentumStrategy(),
            MeanReversionStrategy(),
            FundingArbStrategy()
        ]
        self.min_confidence = 0.50

    def evaluate(self, state: MarketState, current_position: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs all strategies. Returns dict matching old generate_signal format for compatibility.
        """
        votes = []
        
        for strat in self.strategies:
            action, conf = strat.evaluate(state, current_position)
            if action != "HOLD" and conf >= self.min_confidence:
                votes.append({
                    "strategy": strat.name,
                    "action": action,
                    "confidence": conf
                })

        # Base case: no signals
        if not votes:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "composite_score": 0.0,
                "strategy": "None"
            }

        # Select highest confidence signal
        best = max(votes, key=lambda x: x["confidence"])

        # Check for consensus boost
        same_direction = [v for v in votes if v["action"] == best["action"]]
        if len(same_direction) > 1:
            # Boost confidence slightly if multiple strategies agree
            best["confidence"] = min(1.0, best["confidence"] * 1.2)
            logger.debug("🤝 Consensus boost! Strategies agreeing on %s: %s", 
                         best["action"], [v["strategy"] for v in same_direction])

        # Overrides for open positions (exit logic)
        # Inherited from old analysis.py directly into router level for safety
        if current_position == "BUY":
            if state.rsi > 70 + (state.atr_pct * 5):
                return {"action": "SELL", "confidence": 0.9, "strategy": "RiskExit", "composite_score": 0.0}
        elif current_position == "SELL":
            if state.rsi < 30 - (state.atr_pct * 5):
                return {"action": "COVER", "confidence": 0.9, "strategy": "RiskExit", "composite_score": 0.0}

        return {
            "action": best["action"],
            "confidence": best["confidence"],
            "composite_score": 0.0,  # Legacy 
            "strategy": best["strategy"],
            "all_votes": votes
        }
