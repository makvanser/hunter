"""
Hunter V22 — ML Signal Filter Module
======================================
Machine learning layer that filters composite signals to improve
win rate for small-capital ($100) trading.

Uses RandomForestClassifier trained on historical indicator features
to predict whether a signal will be profitable.

Usage:
    from ml import MLFilter
    ml = MLFilter()
    ml.load()  # Load pre-trained model
    if ml.should_trade(market_state, signal):
        # Execute trade
"""

import logging
import os
import math
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import asdict

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from config import (
    ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
)

logger = logging.getLogger("hunter.ml")

# ── Configuration ─────────────────────────────────────────────
ML_MODEL_PATH = "ml_model.pkl"
ML_CONFIDENCE_THRESHOLD = 0.60   # Minimum P(profit) to allow trade
LOOKAHEAD_BARS = 12              # How many bars ahead to evaluate outcome
MIN_PROFIT_PCT = 0.3             # Minimum % move to count as "profitable"


class MLFilter:
    """
    ML-based signal quality filter.
    
    Trained on historical features → binary outcome (profitable/not).
    At inference time, only allows trades where P(profit) > threshold.
    """

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.feature_names = [
            "rsi", "stoch_rsi", "rsi_slope",
            "macd_hist_norm", "bb_position", "adx",
            "atr_pct", "vwap_diff_pct", "composite_score",
            "volume_ratio", "price_change_5bar", "hour_sin",
            "funding_rate_norm", "oi_delta_norm"
        ]
        if not ML_AVAILABLE:
            logger.warning("⚠️ scikit-learn not installed. ML filter disabled.")

    def extract_features(self, market_state, composite_score: float = 0.0,
                          closes: List[float] = None, volumes: List[float] = None,
                          hour: int = 0) -> Optional[np.ndarray]:
        """
        Extract normalized feature vector from MarketState.
        Returns numpy array of shape (12,) or None if data insufficient.
        """
        if not ML_AVAILABLE:
            return None

        try:
            # Volume ratio: current volume vs 20-bar average
            vol_ratio = 1.0
            if volumes and len(volumes) >= 20:
                avg_vol = sum(volumes[-20:]) / 20
                vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0

            # Price change over last 5 bars (%)
            price_change_5 = 0.0
            if closes and len(closes) >= 6:
                price_change_5 = (closes[-1] - closes[-6]) / closes[-6] * 100

            # MACD histogram normalized by ATR
            macd_norm = market_state.macd_histogram / (market_state.current_price * market_state.atr_pct / 100) \
                if market_state.atr_pct > 0 else 0.0
            macd_norm = max(-5.0, min(5.0, macd_norm))  # Clip outliers

            # Hour of day as cyclical feature (sin component)
            hour_sin = math.sin(2 * math.pi * hour / 24)
            
            # Compute ADX from regime string
            adx_approx = {"CHOPPY": 15, "TRENDING": 30, "STRONG_UP": 45, "STRONG_DOWN": 45}.get(
                market_state.regime, 20
            )

            features = np.array([
                market_state.rsi / 100.0,           # Normalize 0-1
                market_state.stoch_rsi / 100.0,     # Normalize 0-1
                market_state.rsi_slope / 20.0,      # Normalize roughly -1 to 1
                macd_norm,
                market_state.bb_position,            # Already 0-1
                adx_approx / 100.0,                 # Normalize 0-1
                market_state.atr_pct / 5.0,          # Normalize: 5% ATR = 1.0
                market_state.vwap_diff_pct / 5.0,    # Normalize: 5% diff = 1.0
                composite_score,                     # Already -1 to 1
                min(vol_ratio, 5.0) / 5.0,           # Normalize: 5x = 1.0
                price_change_5 / 10.0,               # Normalize: 10% move = 1.0
                hour_sin,                            # Already -1 to 1
                getattr(market_state, 'funding_rate', 0.0) * 1000.0,  # V23: Normalize ~0.0001 -> 0.1
                getattr(market_state, 'open_interest_delta', 0.0) / 10.0, # V23: Normalize 10% = 1.0
            ], dtype=np.float64)

            # Replace NaN/Inf
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            return features

        except Exception as e:
            logger.error("Feature extraction failed: %s", e)
            return None

    def train(self, features_list: List[np.ndarray], outcomes: List[int]) -> Dict[str, float]:
        """
        Train the model on historical data.
        
        Args:
            features_list: List of feature vectors
            outcomes: List of binary outcomes (1=profitable, 0=not)
            
        Returns:
            Dict with training metrics (accuracy, cv_score, etc.)
        """
        if not ML_AVAILABLE:
            return {"error": "scikit-learn not installed"}

        X = np.array(features_list)
        y = np.array(outcomes)

        logger.info("🧠 Training ML model on %d samples (%.1f%% positive)", 
                     len(y), np.mean(y) * 100)

        # Use Gradient Boosting for better accuracy on small datasets
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        )
        
        # Cross-validation
        if len(y) >= 10:
            cv_scores = cross_val_score(self.model, X, y, cv=min(5, len(y) // 2), scoring="accuracy")
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            logger.info("📊 Cross-validation accuracy: %.1f%% ± %.1f%%", cv_mean * 100, cv_std * 100)
        else:
            cv_mean, cv_std = 0.0, 0.0

        # Train on full dataset
        self.model.fit(X, y)
        self.is_trained = True

        # Feature importances
        importances = dict(zip(self.feature_names, self.model.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("🏆 Top features: %s", 
                     ", ".join([f"{k}={v:.3f}" for k, v in top_features]))

        return {
            "samples": len(y),
            "positive_rate": float(np.mean(y)),
            "cv_accuracy": float(cv_mean),
            "cv_std": float(cv_std),
            "top_features": dict(top_features),
        }

    def predict(self, features: np.ndarray) -> float:
        """
        Predict probability of a profitable trade.
        Returns float in [0, 1].
        """
        if not self.is_trained or self.model is None:
            return 0.5  # Neutral if no model

        try:
            proba = self.model.predict_proba(features.reshape(1, -1))[0]
            # proba[1] = P(profitable)
            return float(proba[1]) if len(proba) > 1 else 0.5
        except Exception as e:
            logger.error("ML prediction failed: %s", e)
            return 0.5

    def should_trade(self, market_state, composite_score: float = 0.0,
                      closes: List[float] = None, volumes: List[float] = None,
                      hour: int = 0) -> bool:
        """
        Main entry point: should we take this trade?
        Returns True if ML confidence > threshold, or if model not available.
        """
        if not ML_AVAILABLE or not self.is_trained:
            return True  # Passthrough if ML not ready

        features = self.extract_features(market_state, composite_score, closes, volumes, hour)
        if features is None:
            return True

        confidence = self.predict(features)
        
        if confidence < ML_CONFIDENCE_THRESHOLD:
            logger.info("🤖 ML FILTER: BLOCKED (confidence=%.1f%%, threshold=%.1f%%)",
                        confidence * 100, ML_CONFIDENCE_THRESHOLD * 100)
            return False
        
        logger.info("🤖 ML FILTER: APPROVED (confidence=%.1f%%)", confidence * 100)
        return True

    def save(self, path: str = ML_MODEL_PATH):
        """Save trained model to disk."""
        if self.model is not None:
            joblib.dump(self.model, path)
            logger.info("💾 ML model saved to %s", path)

    def load(self, path: str = ML_MODEL_PATH) -> bool:
        """Load pre-trained model from disk."""
        if not ML_AVAILABLE:
            return False
        if not os.path.exists(path):
            logger.warning("⚠️ No ML model found at %s", path)
            return False
        try:
            self.model = joblib.load(path)
            self.is_trained = True
            logger.info("✅ ML model loaded from %s", path)
            return True
        except Exception as e:
            logger.error("Failed to load ML model: %s", e)
            return False
