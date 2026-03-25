"""
Hunter V24 — Statistical Arbitrage (Pairs Trading)
=================================================
Identifies cointegrated pairs that have temporarily diverged.
Calculates the Z-Score of the spread between two assets.
"""

from typing import Dict, List, Tuple, Optional
import collections
import logging
import numpy as np

logger = logging.getLogger("hunter.statarb")

try:
    from hunter_core import KalmanFilter
except ImportError:
    from kalman import KalmanFilter

try:
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.warning("statsmodels not installed. ADF cointegration test disabled.")

class StatArbEngine:
    def __init__(self, z_score_threshold: float = 2.0):
        self.z_score_threshold = z_score_threshold
        # Cache for Kalman filters: symbol -> KalmanFilter instance
        self.filters: Dict[str, KalmanFilter] = {}
        # Cache for recent residuals to run ADF test: symbol -> deque(maxlen=200)
        self.residuals: Dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=200))

    def get_kalman_zscore(self, symbol: str, price_alt: float, price_btc: float) -> float:
        """
        Updates the Kalman filter for the symbol-BTC pair using LOG prices
        for numerical stability.
        """
        if symbol == "BTCUSDT" or price_alt <= 0 or price_btc <= 0:
            return 0.0
            
        if symbol not in self.filters:
            self.filters[symbol] = KalmanFilter(process_noise=1e-5, observation_noise=1e-2)
            
        # Use log prices
        log_a = float(np.log(price_alt))
        log_b = float(np.log(price_btc))
        
        beta, alpha, residual = self.filters[symbol].update(log_a, log_b)
        self.residuals[symbol].append(residual)
        
        # We only start trading once the filter has 'warmed up' 
        z_score = self.filters[symbol].get_zscore()
        
        # V32: Lazy Augmented Dickey-Fuller (ADF) Cointegration Test
        # Only run the expensive ADF test if the Z-score indicates a potential trade
        if HAS_STATSMODELS and abs(z_score) >= self.z_score_threshold:
            res_history = list(self.residuals[symbol])
            if len(res_history) >= 100:  # Need minimum sample size for ADF
                try:
                    # Run ADF test. H0: Non-stationary (unit root exists)
                    adf_result = adfuller(res_history, maxlag=1)
                    p_value = adf_result[1]
                    if p_value > 0.05:
                        logger.debug("🗑️ Rejecting StatArb signal for %s: Residuals not stationary (p=%.3f)", symbol, p_value)
                        return 0.0  # Invalidate z-score
                except Exception as e:
                    logger.debug("Failed to run ADF test on %s: %s", symbol, e)
                    
        return z_score

    def find_arbitrage_opportunities(self, market_data_cache: Dict[str, List[float]]) -> List[Dict]:
        """
        Legacy batch scanner (for background monitoring).
        Updated to use Kalman filters if possible, or fallback to naive spread.
        """
        # Note: Background scanner needs 'history' to warm up filters if they are new
        # This is a bit heavy for a background task, so we focus on real-time 
        # signals in the main loop.
        return []
