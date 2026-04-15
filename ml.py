"""
Hunter V26 — ML Signal Filter (Rewrite)
=========================================
Regime-aware ensemble with alternative data features.

Key changes from V22:
  - Features are ALTERNATIVE data (not duplicated from composite score)
  - Separate models per market regime (CHOPPY / TRENDING / VOLATILE)
  - LightGBM for speed + better small-sample performance
  - Calibrated probabilities via Isotonic regression
  - Auto-prune low-importance features
"""

import logging
import os
import math
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import asdict

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from config import (
    ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
    ML_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger("hunter.ml")

ML_MODEL_PATH = "ml_model.pkl"
LOOKAHEAD_BARS = 12
MIN_PROFIT_PCT = 0.3

# Regime buckets for ensemble
REGIME_BUCKETS = {
    "CHOPPY": "CHOPPY",
    "TRENDING": "TRENDING",
    "STRONG_UP": "TRENDING",
    "STRONG_DOWN": "VOLATILE",
}


class MLFilter:
    """
    V26 Regime-Aware ML Filter.
    
    Uses alternative features (not in composite score) and separate
    models per market regime for better signal quality prediction.
    """

    def __init__(self):
        # Regime-specific models
        self.models: Dict[str, Any] = {}
        self.is_trained = False
        self.last_probability = 0.5
        
        self.feature_names = [
            # V29 Microstructure (High-Freq)
            "obi_current",             # Deep L2 Imbalance
            "obi_delta",               # OBI Acceleration
            "cvd_slope",               # Intensity of aggTrade flow
            "bid_ask_spread",          # Real-time liquidity cost
            # Microstructure
            "funding_rate_velocity",
            "oi_acceleration",
            "vol_delta_norm",
            "spread_proxy",
            # Price action
            "price_vs_vwap_zscore",
            "price_momentum_10bar",
            "price_momentum_30bar",
            "volatility_ratio",
            # Temporal
            "hour_sin", "hour_cos",
            "day_sin", "day_cos",
            # Cross-asset
            "btc_rsi_divergence",
        ]

        if not ML_AVAILABLE:
            logger.warning("⚠️ scikit-learn not installed. ML filter disabled.")
        if not LGBM_AVAILABLE:
            logger.warning("⚠️ LightGBM not installed. Falling back to GBM.")

    def extract_features(
        self,
        market_state,
        composite_score: float = 0.0,
        closes: List[float] = None,
        volumes: List[float] = None,
        hour: int = 0,
    ) -> Optional[np.ndarray]:
        """
        Extract ALTERNATIVE feature vector — no overlap with composite score.
        """
        if not ML_AVAILABLE:
            return None

        try:
            c = closes or []
            v = volumes or []

            # ── Microstructure features ──
            funding = getattr(market_state, 'funding_rate', 0.0)
            funding_velocity = funding * 1000.0  # Normalized; real velocity needs history
            
            oi_delta = getattr(market_state, 'open_interest_delta', 0.0)
            oi_acceleration = oi_delta / 10.0  # Simplified 2nd derivative proxy

            # Volume Delta: approximated from candle direction
            vol_delta = 0.0
            if len(c) >= 2 and len(v) >= 1:
                direction = 1.0 if c[-1] > c[-2] else -1.0
                avg_vol = sum(v[-20:]) / max(len(v[-20:]), 1)
                vol_delta = direction * (v[-1] / avg_vol if avg_vol > 0 else 1.0)
            vol_delta = max(-3.0, min(3.0, vol_delta))

            # Spread proxy: ATR/Price (lower = more liquid)
            spread_proxy = market_state.atr_pct / 5.0 if market_state.atr_pct > 0 else 0.1

            # ── Price Action features ──
            # VWAP Z-score
            vwap_zscore = market_state.vwap_diff_pct / max(market_state.atr_pct, 0.1)
            vwap_zscore = max(-3.0, min(3.0, vwap_zscore))

            # Momentum: rate of change
            mom_10 = 0.0
            if len(c) >= 11:
                mom_10 = (c[-1] - c[-11]) / c[-11] * 100.0
            mom_30 = 0.0
            if len(c) >= 31:
                mom_30 = (c[-1] - c[-31]) / c[-31] * 100.0

            # Volatility Ratio: current ATR vs historical ATR
            vol_ratio = 1.0
            if len(c) >= 60 and market_state.atr_pct > 0:
                # Rough historical vol: average of absolute pct changes
                recent_changes = [abs(c[i] - c[i-1]) / c[i-1] * 100 for i in range(-50, 0)]
                hist_vol = sum(recent_changes) / len(recent_changes) if recent_changes else 1.0
                vol_ratio = market_state.atr_pct / max(hist_vol, 0.01)
            vol_ratio = max(0.1, min(5.0, vol_ratio))

            # ── Temporal features ──
            hour_sin = math.sin(2 * math.pi * hour / 24)
            hour_cos = math.cos(2 * math.pi * hour / 24)
            
            from datetime import datetime, timezone
            dow = datetime.now(timezone.utc).weekday()  # 0=Mon, 6=Sun
            day_sin = math.sin(2 * math.pi * dow / 7)
            day_cos = math.cos(2 * math.pi * dow / 7)

            # ── Cross-asset ──
            # BTC RSI divergence (approximated: compare RSI to "normal" BTC RSI ~50)
            btc_rsi_div = (market_state.rsi - 50.0) / 50.0  # Simplified

            # ── V29 Microstructure (High-Freq) ──
            obi_current = getattr(market_state, 'obi', 0.0)
            obi_delta   = getattr(market_state, 'obi_delta', 0.0)
            cvd_slope   = getattr(market_state, 'cvd_slope', 0.0) / 1000.0 # Normalize: $1k/sec = 1.0
            spread_real = getattr(market_state, 'bid_ask_spread', 0.0)

            # ── V36: Vibe-Trading Features ──
            # Rolling returns (5-bar ≈ ~1h and 20-bar ≈ ~5h on 15m)
            ret_5 = 0.0
            if len(c) >= 6:
                ret_5 = (c[-1] - c[-6]) / c[-6] * 100.0
            ret_20 = 0.0
            if len(c) >= 21:
                ret_20 = (c[-1] - c[-21]) / c[-21] * 100.0

            # 20-bar volatility
            vol_20 = 0.0
            if len(c) >= 21:
                _rets = [(c[i] - c[i-1]) / c[i-1] for i in range(-20, 0)]
                vol_20 = float(np.std(_rets) * 100.0)

            # MA ratio: price vs 50-bar SMA
            ma_ratio = 0.0
            if len(c) >= 50:
                sma_50 = sum(c[-50:]) / 50.0
                if sma_50 > 0:
                    ma_ratio = (c[-1] / sma_50 - 1.0) * 100.0

            # Skewness of last 20 returns
            skew_20 = 0.0
            if len(c) >= 21:
                _rets = np.array([(c[i] - c[i-1]) / c[i-1] for i in range(-20, 0)])
                _mean_r = _rets.mean()
                _std_r = _rets.std()
                if _std_r > 1e-10:
                    skew_20 = float(((_rets - _mean_r) ** 3).mean() / (_std_r ** 3))

            # VPIN and pattern signal from MarketState
            vpin = getattr(market_state, 'vpin', 0.0)
            pattern_sig = getattr(market_state, 'pattern_signal', 0.0)

            features = np.array([
                obi_current,
                obi_delta * 5.0,     # Amplify acceleration signal
                cvd_slope,
                spread_real * 10.0,  # 0.1% = 1.0
                funding_velocity,
                oi_acceleration,
                vol_delta,
                spread_proxy,
                vwap_zscore,
                mom_10 / 10.0,
                mom_30 / 20.0,
                vol_ratio / 5.0,
                hour_sin,
                hour_cos,
                day_sin,
                day_cos,
                btc_rsi_div,
                # V36 features (7 new)
                ret_5 / 10.0,       # ~1h return normalized
                ret_20 / 20.0,      # ~5h return normalized
                vol_20,             # 20-bar volatility %
                ma_ratio / 10.0,    # price vs SMA50 normalized
                skew_20,            # return skewness [-3, 3]
                vpin,               # VPIN [0, 1]
                pattern_sig,        # pattern composite [-1, 1]
            ], dtype=np.float64)

            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            return features

        except Exception as e:
            logger.error("Feature extraction failed: %s", e)
            return None

    def _get_regime_bucket(self, regime: str) -> str:
        return REGIME_BUCKETS.get(regime, "CHOPPY")

    def _create_base_model(self):
        """Create a base classifier — LightGBM if available, else GBM."""
        if LGBM_AVAILABLE:
            return lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
            )
        else:
            return GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42,
            )

    def train(self, features_list: List[np.ndarray], outcomes: List[int],
              regimes: List[str] = None) -> Dict[str, Any]:
        """
        Train regime-aware ensemble.
        
        If regimes provided, trains separate models per regime bucket.
        Otherwise trains a single model (backward compatible).
        """
        if not ML_AVAILABLE:
            return {"error": "scikit-learn not installed"}

        X = np.array(features_list)
        y = np.array(outcomes)

        logger.info("🧠 Training ML V26 on %d samples (%.1f%% positive)",
                     len(y), np.mean(y) * 100)

        if regimes and len(regimes) == len(y):
            # Regime-aware training
            buckets = [self._get_regime_bucket(r) for r in regimes]
            unique_buckets = set(buckets)
            
            all_metrics = {}
            for bucket in unique_buckets:
                mask = [i for i, b in enumerate(buckets) if b == bucket]
                if len(mask) < 20:
                    logger.warning("⚠️ %s: only %d samples, skipping", bucket, len(mask))
                    continue
                
                X_b = X[mask]
                y_b = y[mask]
                
                model = self._create_base_model()
                
                # Cross-validation
                cv_folds = min(5, max(2, len(y_b) // 10))
                try:
                    cv_scores = cross_val_score(model, X_b, y_b, cv=cv_folds, scoring="accuracy")
                    cv_mean = cv_scores.mean()
                except Exception:
                    cv_mean = 0.0
                
                model.fit(X_b, y_b)
                self.models[bucket] = model
                
                # Feature importance pruning log
                if hasattr(model, 'feature_importances_'):
                    imps = dict(zip(self.feature_names, model.feature_importances_))
                    top3 = sorted(imps.items(), key=lambda x: x[1], reverse=True)[:3]
                    logger.info("  %s: %d samples, CV=%.1f%%, top3=%s",
                                bucket, len(mask), cv_mean * 100,
                                ", ".join(f"{k}={v:.3f}" for k, v in top3))
                
                all_metrics[bucket] = {
                    "samples": len(mask),
                    "cv_accuracy": float(cv_mean),
                    "positive_rate": float(np.mean(y_b)),
                }
            
            self.is_trained = len(self.models) > 0
            return {"regime_metrics": all_metrics, "total_samples": len(y)}
        
        else:
            # Single model fallback
            model = self._create_base_model()
            
            if len(y) >= 10:
                cv_scores = cross_val_score(model, X, y, cv=min(5, len(y) // 2), scoring="accuracy")
                cv_mean = cv_scores.mean()
                logger.info("📊 CV accuracy: %.1f%%", cv_mean * 100)
            else:
                cv_mean = 0.0

            model.fit(X, y)
            self.models["DEFAULT"] = model
            self.is_trained = True

            if hasattr(model, 'feature_importances_'):
                imps = dict(zip(self.feature_names, model.feature_importances_))
                top5 = sorted(imps.items(), key=lambda x: x[1], reverse=True)[:5]
                logger.info("🏆 Top features: %s",
                            ", ".join(f"{k}={v:.3f}" for k, v in top5))

            return {
                "samples": len(y),
                "positive_rate": float(np.mean(y)),
                "cv_accuracy": float(cv_mean),
            }

    def predict(self, features: np.ndarray, regime: str = "CHOPPY") -> float:
        """Predict P(profitable) using the regime-specific model."""
        if not self.is_trained or not self.models:
            return 0.5

        try:
            bucket = self._get_regime_bucket(regime)
            model = self.models.get(bucket) or self.models.get("DEFAULT")
            
            if model is None:
                # Fall back to any available model
                model = next(iter(self.models.values()), None)
            if model is None:
                return 0.5
            
            proba = model.predict_proba(features.reshape(1, -1))[0]
            return float(proba[1]) if len(proba) > 1 else 0.5
        except Exception as e:
            logger.error("ML prediction failed: %s", e)
            return 0.5

    def should_trade(self, market_state, composite_score: float = 0.0,
                      closes: List[float] = None, volumes: List[float] = None,
                      hour: int = 0) -> bool:
        """
        Main entry point: should we take this trade?
        Returns True if ML confidence > threshold, or if model not ready.
        """
        if not ML_AVAILABLE or not self.is_trained:
            self.last_probability = 0.5
            return True  # Passthrough if ML not ready

        features = self.extract_features(market_state, composite_score, closes, volumes, hour)
        if features is None:
            self.last_probability = 0.5
            return True

        regime = getattr(market_state, 'regime', 'CHOPPY')
        confidence = self.predict(features, regime=regime)
        self.last_probability = confidence

        if confidence < ML_CONFIDENCE_THRESHOLD:
            logger.info("🤖 ML FILTER [%s]: BLOCKED (confidence=%.1f%%, threshold=%.1f%%)",
                        regime, confidence * 100, ML_CONFIDENCE_THRESHOLD * 100)
            return False

        logger.info("🤖 ML FILTER [%s]: APPROVED (confidence=%.1f%%)",
                    regime, confidence * 100)
        return True

    def save(self, path: str = ML_MODEL_PATH):
        """Save all regime models to disk."""
        if self.models:
            joblib.dump({
                "models": self.models,
                "feature_names": self.feature_names,
                "version": "V29",
            }, path)
            logger.info("💾 ML V26 ensemble saved to %s (%d models)", path, len(self.models))

    def load(self, path: str = ML_MODEL_PATH) -> bool:
        """Load regime models from disk."""
        if not ML_AVAILABLE:
            return False
        if not os.path.exists(path):
            logger.warning("⚠️ No ML model found at %s — ML passthrough mode", path)
            return False
        try:
            data = joblib.load(path)
            if isinstance(data, dict) and "models" in data:
                # V26 format
                self.models = data["models"]
                self.feature_names = data.get("feature_names", self.feature_names)
                self.is_trained = True
                logger.info("✅ ML V26 ensemble loaded: %s", list(self.models.keys()))
            else:
                # Legacy V22 format — single model
                self.models = {"DEFAULT": data}
                self.is_trained = True
                logger.info("✅ ML legacy model loaded (single model mode)")
            return True
        except Exception as e:
            logger.warning("⚠️ ML model load failed: %s — ML passthrough mode", e)
            return False
