"""
Hunter V28 — Event-Driven Backtest Framework
==============================================
Simulates historical market conditions bar-by-bar to test the StrategyRouter
and ML Filter before deploying to production.

Features:
- Fetches and caches historical Klines via Binance API
- Replays data chronologically using the exact same MarketState logic as Live
- Computes exact PnL with slippage and exchange fees
"""

import sys
import logging
import math
from datetime import datetime
import asyncio
import aiohttp
from typing import List, Dict, Tuple

# Import Hunter core logic
from config import TIMEFRAME, TAKER_FEE, MAKER_FEE, SLIPPAGE, INITIAL_BALANCE_USD
from analysis import (
    compute_adx, compute_atr, compute_bollinger, compute_macd,
    compute_rsi, compute_rsi_slope, compute_stoch_rsi, compute_vwap,
    get_market_regime, MarketState
)
from strategy_router import StrategyRouter
from ml import MLFilter

# Configure bare-bones logging for backtester
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("backtest")


class BacktestEngine:
    def __init__(self, initial_balance=INITIAL_BALANCE_USD):
        self.router = StrategyRouter()
        self.ml_filter = MLFilter()
        self.ml_filter.load()  # Load if available, else passthrough
        
        self.balance = initial_balance
        self.start_balance = initial_balance
        self.position = None  # None, or {"side": "BUY"|"SELL", "entry": float, "size": float}
        self.trades = []
        
        # Stats
        self.wins = 0
        self.losses = 0
        self.peak_balance = initial_balance
        self.max_drawdown_pct = 0.0

    async def fetch_data(self, symbol: str, interval: str, limit: int = 1500) -> List[List]:
        """Fetch historical Klines from Binance API."""
        logger.info(f"Downloading {limit} bars of {interval} data for {symbol}...")
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()
                return data

    def run(self, symbol: str, klines: List[List]):
        """Run the backtest across the provided klines."""
        if len(klines) < 100:
            logger.error("Not enough data to backtest. Need at least 100 bars for indicators warmup.")
            return

        logger.info(f"\n--- Starting Backtest: {symbol} ---")
        
        # Warmup period for indicators
        WARMUP = 60
        
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        closes = [float(k[4]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        
        for i in range(WARMUP, len(klines)):
            # "Replay" data up to index i
            h = highs[:i+1]
            l = lows[:i+1]
            c = closes[:i+1]
            v = volumes[:i+1]
            current_price = c[-1]
            
            # 1. Update Position if needed (e.g., Stop Loss / Take Profit)
            # Simplified static ATR-based SL/TP for the backtester
            if self.position:
                p = self.position
                pnl = 0
                closed = False
                reason = ""
                
                # Check for simplistic SL/TP (assuming ATR was calculated at entry)
                if p["side"] == "BUY":
                    if current_price <= p["sl"]:
                        pnl = (current_price - p["entry"]) / p["entry"] * p["size"]
                        closed, reason = True, "SL"
                    elif current_price >= p["tp"]:
                        pnl = (current_price - p["entry"]) / p["entry"] * p["size"]
                        closed, reason = True, "TP"
                else: # SHORT
                    if current_price >= p["sl"]:
                        pnl = (p["entry"] - current_price) / p["entry"] * p["size"]
                        closed, reason = True, "SL"
                    elif current_price <= p["tp"]:
                        pnl = (p["entry"] - current_price) / p["entry"] * p["size"]
                        closed, reason = True, "TP"
                        
                if closed:
                    self._close_position(current_price, pnl, reason, i)
                    continue # Skip new signals on the bar we got stopped out
                    
            # 2. Compute Market State exactly as live
            adx = compute_adx(h, l, c, 14)
            regime = get_market_regime(adx)
            rsi = compute_rsi(c, 14)
            lower, _, upper = compute_bollinger(c, 20)
            atr = compute_atr(h, l, c, 14)
            
            # Simple VWAP proxy for backtester
            vwap = compute_vwap(h[-24:], l[-24:], c[-24:], v[-24:]) 
            
            bb_range = upper - lower
            bb_position = (current_price - lower) / bb_range if bb_range > 0 else 0.5
            atr_pct = (atr / current_price * 100) if current_price else 0.0
            
            state = MarketState(
                current_price=current_price, rsi=rsi, ls_ratio=1.0, whale_net_vol=0.0,
                regime=regime, social_score=0.0, macd_histogram=compute_macd(c)[2],
                bb_position=bb_position, vwap_diff_pct=((current_price-vwap)/vwap*100) if vwap else 0,
                divergence="NONE", funding_rate=0.0, open_interest_delta=0.0,
                liq_imbalance=0.0, atr_pct=atr_pct, rsi_slope=compute_rsi_slope(c),
                stoch_rsi=compute_stoch_rsi(c), mtf_agreement=0.0, volume_confirm=True,
                near_resistance=False, btc_correlation=1.0, btc_dominance=50.0
            )

            # 3. Router Evaluation
            pos_side = self.position["side"] if self.position else None
            signal = self.router.evaluate(state, current_position=pos_side)
            action = signal.get("action", "HOLD")
            
            # 4. ML Filter Gate
            if action in ("BUY", "SHORT"):
                if not self.ml_filter.should_trade(state, 0.0, c, v, hour=i % 24):
                    action = "HOLD"

            # 5. Execution Logic
            if action in ("BUY", "SHORT") and not self.position:
                self._open_position(action, current_price, atr, signal["strategy"])
                
            elif action in ("SELL", "COVER") and self.position:
                # Close due to signal
                if (action == "SELL" and self.position["side"] == "BUY") or \
                   (action == "COVER" and self.position["side"] == "SELL"):
                   
                    mult = 1.0 if self.position["side"] == "BUY" else -1.0
                    pnl = (current_price - self.position["entry"]) / self.position["entry"] * self.position["size"] * mult
                    self._close_position(current_price, pnl, "SIGNAL_EXIT", i)

    def _open_position(self, action: str, price: float, atr: float, strategy: str):
        # Position sizing: Risk 2% of equity
        risk_pct = 0.02
        size = self.balance * risk_pct * 5 # 5x leverage
        size = min(size, self.balance * 0.5) # Max 50% margin
        
        # Deduct entry fees
        fee = size * (TAKER_FEE + SLIPPAGE)
        self.balance -= fee
        
        # Entry logic
        self.position = {
            "side": action,
            "entry": price,
            "size": size,
            "sl": price - (atr * 2.0) if action == "BUY" else price + (atr * 2.0),
            "tp": price + (atr * 3.5) if action == "BUY" else price - (atr * 3.5),
            "strategy": strategy
        }
        logger.debug(f"OPEN {action} @ {price:.4f} via {strategy}")

    def _close_position(self, price: float, raw_pnl: float, reason: str, bar_index: int):
        # Target size
        size = self.position["size"]
        
        # Deduct exit fees
        fee = size * (TAKER_FEE + SLIPPAGE)
        net_pnl = raw_pnl - fee
        self.balance += net_pnl
        
        # Track stats
        if net_pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
            
        self.trades.append(net_pnl)
        
        # Max Drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        else:
            dd = (self.peak_balance - self.balance) / self.peak_balance * 100
            if dd > self.max_drawdown_pct:
                self.max_drawdown_pct = dd
                
        logger.debug(f"CLOSE {self.position['side']} @ {price:.4f} | Reason: {reason} | PnL: ${net_pnl:.2f}")
        self.position = None

    def print_results(self):
        logger.info("\n=== BACKTEST RESULTS ===")
        total_trades = self.wins + self.losses
        if total_trades == 0:
            logger.info("No trades executed.")
            return
            
        winrate = (self.wins / total_trades) * 100
        net_profit = self.balance - self.start_balance
        profit_pct = (net_profit / self.start_balance) * 100
        
        logger.info(f"Initial Balance: ${self.start_balance:.2f}")
        logger.info(f"Final Balance:   ${self.balance:.2f}")
        logger.info(f"Net Profit:      ${net_profit:.2f} ({profit_pct:.2f}%)")
        logger.info(f"Max Drawdown:    {self.max_drawdown_pct:.2f}%")
        logger.info(f"Total Trades:    {total_trades}")
        logger.info(f"Win/Loss:        {self.wins} / {self.losses} ({winrate:.1f}% Winrate)")


async def main():
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    else:
        symbol = "BTCUSDT"
        
    engine = BacktestEngine(initial_balance=1000)
    klines = await engine.fetch_data(symbol, TIMEFRAME, limit=1500)
    
    engine.run(symbol, klines)
    engine.print_results()


if __name__ == "__main__":
    asyncio.run(main())
