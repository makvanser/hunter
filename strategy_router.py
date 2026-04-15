"""
Hunter V36 — Strategy Router
==============================
Multi-strategy ensemble. Evaluates multiple independent trading
strategies in parallel and selects the best one based on current
market regime and signal strength.

V36: Added PatternStrategy (chart patterns + Ichimoku) and
     SMCStrategy (Smart Money Concepts) from Vibe-Trading analytics.
"""

import logging
from typing import Dict, Any, Tuple, List, Optional
from analysis import MarketState
from patterns import get_pattern_signal, compute_ichimoku
from smc import SMCAnalyzer

logger = logging.getLogger("hunter.strategy")

# Lazy-initialized SMC analyzer (requires pandas + smartmoneyconcepts)
_smc_analyzer: Optional[SMCAnalyzer] = None

def _get_smc_analyzer() -> SMCAnalyzer:
    global _smc_analyzer
    if _smc_analyzer is None:
        _smc_analyzer = SMCAnalyzer(swing_length=10)
    return _smc_analyzer


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


class StatArbStrategy(BaseStrategy):
    """
    V28 Phase 4: Statistical Arbitrage against BTC.
    Trades the spread between the Altcoin and Bitcoin. 
    If Z-Score > 2.5, the Altcoin is overpriced relative to BTC -> SHORT.
    If Z-Score < -2.5, the Altcoin is underpriced relative to BTC -> BUY.
    Requires highly correlated pairs (BTC correlation > 0.75).
    """
    name = "StatArb"
    best_regimes = ["CHOPPY", "TRENDING", "STRONG_UP", "STRONG_DOWN"]

    def evaluate(self, state: MarketState, current_position: Optional[str] = None) -> Tuple[str, float]:
        action = "HOLD"
        conf = 0.0

        # Only trade highly correlated assets
        if state.btc_correlation < 0.75:
            return action, conf

        # Altcoin is heavily OVERVALUED relative to its historical beta to BTC
        if state.btc_spread_zscore > 2.5:
            action = "SHORT"
            conf = min(1.0, (state.btc_spread_zscore - 2.0) / 2.0) 

        # Altcoin is heavily UNDERVALUED relative to its historical beta to BTC
        elif state.btc_spread_zscore < -2.5:
            action = "BUY"
            conf = min(1.0, (abs(state.btc_spread_zscore) - 2.0) / 2.0)

        return action, conf


class PatternStrategy(BaseStrategy):
    """
    V36: Chart pattern detection strategy.
    Uses H&S, double top/bottom, triangles, candlestick, and Ichimoku.
    Best in trending regimes where structural patterns are more reliable.
    """
    name = "Pattern"
    best_regimes = ["TRENDING", "STRONG_UP", "STRONG_DOWN"]

    def evaluate(self, state: MarketState, current_position: Optional[str] = None) -> Tuple[str, float]:
        action = "HOLD"
        conf = 0.0

        # Use pattern_signal from MarketState if available (set by main.py)
        pattern_score = getattr(state, 'pattern_signal', 0)
        if pattern_score == 0:
            return action, conf

        if pattern_score > 0:
            action = "BUY"
            conf = min(1.0, abs(pattern_score) * 0.8 + 0.3)
        elif pattern_score < 0:
            action = "SHORT"
            conf = min(1.0, abs(pattern_score) * 0.8 + 0.3)

        return action, conf


class SMCStrategy(BaseStrategy):
    """
    V36: Smart Money Concepts strategy.
    Uses BOS/ChoCH/FVG/Order Blocks for institutional flow detection.
    Best in trending regimes where institutional footprints are clearest.
    """
    name = "SMC"
    best_regimes = ["TRENDING", "STRONG_UP", "STRONG_DOWN"]

    def evaluate(self, state: MarketState, current_position: Optional[str] = None) -> Tuple[str, float]:
        action = "HOLD"
        conf = 0.0

        # Use smc_signal from MarketState if available (set by main.py)
        smc_signal = getattr(state, 'smc_signal', 0)
        if smc_signal == 0:
            return action, conf

        if smc_signal > 0:
            action = "BUY"
            conf = 0.65  # SMC signals are high-conviction but rare
        elif smc_signal < 0:
            action = "SHORT"
            conf = 0.65

        return action, conf


class StrategyRouter:
    """
    V36: Evaluates all registered strategies and selects the best signal.
    Includes 7 strategies: Grid, Momentum, MeanReversion, FundingArb,
    StatArb, Pattern (V36), and SMC (V36).
    """
    def __init__(self):
        self.strategies: List[BaseStrategy] = [
            GridStrategy(),
            MomentumStrategy(),
            MeanReversionStrategy(),
            FundingArbStrategy(),
            StatArbStrategy(),
            PatternStrategy(),
            SMCStrategy(),
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
