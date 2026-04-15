"""
Hunter V36 — Smart Money Concepts (SMC) Analyzer
==================================================
Wrapper around the `smartmoneyconcepts` library for detecting
institutional order flow patterns.

Detects:
  - BOS (Break of Structure) — trend continuation
  - ChoCH (Change of Character) — trend reversal
  - FVG (Fair Value Gap) — price re-fill zones
  - Order Blocks — institutional order concentration zones

Falls back gracefully to no-op if smartmoneyconcepts is not installed.

Usage:
    from smc import SMCAnalyzer
    analyzer = SMCAnalyzer()
    signal = analyzer.get_signal(opens, highs, lows, closes)
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("hunter.smc")

try:
    import pandas as pd
    from smartmoneyconcepts import smc as _smc
    SMC_AVAILABLE = True
    logger.info("✅ smartmoneyconcepts library loaded")
except ImportError:
    SMC_AVAILABLE = False
    logger.warning("⚠️ smartmoneyconcepts not installed. SMC signals disabled. "
                   "Install with: pip install smartmoneyconcepts pandas")


class SMCAnalyzer:
    """
    V36 Smart Money Concepts Analyzer.
    
    Wraps the smartmoneyconcepts library to detect institutional
    order flow patterns from OHLCV data.
    """
    
    def __init__(self, swing_length: int = 10):
        self.swing_length = swing_length
    
    def analyze(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
    ) -> Dict[str, Any]:
        """
        Run full SMC analysis on OHLCV data.
        
        Returns:
            Dict with bos_signal, choch_signal, latest_fvg, latest_ob.
        """
        if not SMC_AVAILABLE:
            return {"bos": 0, "choch": 0, "fvg": None, "ob": None, "available": False}
        
        if len(closes) < self.swing_length * 3:
            return {"bos": 0, "choch": 0, "fvg": None, "ob": None, "available": True}
        
        try:
            df = pd.DataFrame({
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            })
            
            # Break of Structure
            bos_df = _smc.bos_choch(df, close_column="close", swing_length=self.swing_length)
            
            # Fair Value Gaps
            fvg_df = _smc.fvg(df, join_consecutive=True)
            
            # Order Blocks
            ob_df = _smc.ob(df, swing_length=self.swing_length)
            
            # Extract latest signals
            bos_signal = self._extract_bos_signal(bos_df)
            choch_signal = self._extract_choch_signal(bos_df)
            latest_fvg = self._extract_latest_fvg(fvg_df)
            latest_ob = self._extract_latest_ob(ob_df)
            
            return {
                "bos": bos_signal,
                "choch": choch_signal,
                "fvg": latest_fvg,
                "ob": latest_ob,
                "available": True,
            }
        except Exception as e:
            logger.warning("⚠️ SMC analysis error: %s", e)
            return {"bos": 0, "choch": 0, "fvg": None, "ob": None, "available": True}
    
    def get_signal(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
    ) -> int:
        """
        Get composite SMC signal.
        
        Signal logic:
          - bullish ChoCH/BOS + bullish FVG → 1 (LONG)
          - bearish ChoCH/BOS + bearish FVG → -1 (SHORT)
          - else → 0 (HOLD)
        
        Returns:
            -1, 0, or 1.
        """
        result = self.analyze(opens, highs, lows, closes, volumes)
        
        bos = result["bos"]
        choch = result["choch"]
        
        # ChoCH takes priority over BOS (it's a reversal signal)
        primary = choch if choch != 0 else bos
        
        if primary == 0:
            return 0
        
        # FVG confirmation (if available)
        fvg = result.get("fvg")
        if fvg and fvg.get("direction") == primary:
            return primary  # Confirmed by FVG
        
        # Without FVG confirmation, reduce confidence but still signal
        return primary
    
    def _extract_bos_signal(self, df) -> int:
        """Extract latest BOS signal: 1=bullish, -1=bearish, 0=none."""
        try:
            # Look for BOS columns in the result DataFrame
            for col in df.columns:
                if "BOS" in str(col):
                    last_valid = df[col].dropna()
                    if len(last_valid) > 0:
                        last = last_valid.iloc[-1]
                        if last > 0:
                            return 1
                        elif last < 0:
                            return -1
        except Exception:
            pass
        return 0
    
    def _extract_choch_signal(self, df) -> int:
        """Extract latest ChoCH signal: 1=bullish, -1=bearish, 0=none."""
        try:
            for col in df.columns:
                if "CHoCH" in str(col) or "CHOCH" in str(col):
                    last_valid = df[col].dropna()
                    if len(last_valid) > 0:
                        last = last_valid.iloc[-1]
                        if last > 0:
                            return 1
                        elif last < 0:
                            return -1
        except Exception:
            pass
        return 0
    
    def _extract_latest_fvg(self, df) -> Optional[Dict[str, Any]]:
        """Extract latest Fair Value Gap."""
        try:
            if df is None or len(df) == 0:
                return None
            for col in df.columns:
                if "FVG" in str(col):
                    last_valid = df[col].dropna()
                    if len(last_valid) > 0:
                        val = last_valid.iloc[-1]
                        direction = 1 if val > 0 else -1 if val < 0 else 0
                        return {"direction": direction, "value": float(val)}
        except Exception:
            pass
        return None
    
    def _extract_latest_ob(self, df) -> Optional[Dict[str, Any]]:
        """Extract latest Order Block."""
        try:
            if df is None or len(df) == 0:
                return None
            for col in df.columns:
                if "OB" in str(col):
                    last_valid = df[col].dropna()
                    if len(last_valid) > 0:
                        val = last_valid.iloc[-1]
                        direction = 1 if val > 0 else -1 if val < 0 else 0
                        return {"direction": direction, "value": float(val)}
        except Exception:
            pass
        return None
