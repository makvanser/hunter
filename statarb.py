"""
Hunter V24 — Statistical Arbitrage (Pairs Trading)
=================================================
Identifies cointegrated pairs that have temporarily diverged.
Calculates the Z-Score of the spread between two assets.
"""

import numpy as np
from typing import Dict, List, Tuple

class StatArbEngine:
    def __init__(self, z_score_threshold: float = 2.0):
        self.z_score_threshold = z_score_threshold

    def calculate_spread_zscore(self, closes_a: List[float], closes_b: List[float]) -> float:
        """
        Calculates the current Z-Score of the spread between two time series.
        Using simple log returns spread for crypto.
        """
        if len(closes_a) < 20 or len(closes_b) < 20:
            return 0.0
            
        # Ensure equal length
        min_len = min(len(closes_a), len(closes_b))
        ca = np.array(closes_a[-min_len:])
        cb = np.array(closes_b[-min_len:])
        
        # Log prices
        log_a = np.log(ca)
        log_b = np.log(cb)
        
        # Calculate spread ratio (A / B loosely)
        spread = log_a - log_b
        
        # Z-score of the current spread
        mean_spread = np.mean(spread)
        std_spread = np.std(spread)
        
        if std_spread == 0:
            return 0.0
            
        current_spread = spread[-1]
        z_score = (current_spread - mean_spread) / std_spread
        
        return float(z_score)

    def find_arbitrage_opportunities(self, market_data_cache: Dict[str, List[float]]) -> List[Dict]:
        """
        Scans all pairs in the cache to find highly diverged pairs.
        Returns a list of dicts describing the opportunity.
        """
        opportunities = []
        symbols = list(market_data_cache.keys())
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym_a = symbols[i]
                sym_b = symbols[j]
                
                z = self.calculate_spread_zscore(market_data_cache[sym_a], market_data_cache[sym_b])
                
                if abs(z) > self.z_score_threshold:
                    # If Z > 2, Sym A is overvalued relative to Sym B
                    # Action: Short A, Buy B
                    opportunities.append({
                        "pair": f"{sym_a}/{sym_b}",
                        "sym_a": sym_a,
                        "sym_b": sym_b,
                        "z_score": z,
                        "action": "SHORT_A_BUY_B" if z > 0 else "BUY_A_SHORT_B"
                    })
                    
        # Sort by most extreme divergence
        opportunities.sort(key=lambda x: abs(x['z_score']), reverse=True)
        return opportunities
