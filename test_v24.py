import asyncio
import unittest
from statarb import StatArbEngine

class TestV24(unittest.TestCase):
    def test_statarb_zscore(self):
        engine = StatArbEngine()
        closes_a = [100.0, 101.0, 102.0, 101.5, 100.5] * 5  # 25 items
        closes_b = [50.0, 50.5, 51.0, 50.75, 50.25] * 5    # exactly half
        
        z = engine.calculate_spread_zscore(closes_a, closes_b)
        self.assertTrue(isinstance(z, float))
        # Z-score of perfectly correlated series with static ratio should be near 0
        self.assertAlmostEqual(z, 0.0, places=5)
        
    def test_statarb_divergence(self):
        engine = StatArbEngine()
        closes_a = [100.0] * 20 + [110.0] # Sudden spike in A
        closes_b = [50.0] * 20 + [50.0]   # B flat
        z = engine.calculate_spread_zscore(closes_a, closes_b)
        self.assertTrue(z > 1.0) # Should be positive as spread widened upwards

if __name__ == '__main__':
    unittest.main()
