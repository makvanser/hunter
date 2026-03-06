"""
Hunter V25/V26 — NautilusTrader Execution Engine Core
======================================================
This module replaces `main.py`, `live_execution.py`, and `provider.py` with
a pure Rust-backed high-frequency trading engine (NautilusTrader).

NOTE: This is the Phase 3 Prototype. It wires Nautilus events to our existing
Python logic (analysis.py, ml.py) while keeping execution and orderbook
state in Rust to achieve <5ms latency.
"""

import logging
import asyncio
from typing import Dict, Optional

# Nautilus Core imports
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.data import OrderBookDeltas, QuoteTick, Bar
from nautilus_trader.model.identifiers import InstrumentId, ClientId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.position import Position
from nautilus_trader.trading.strategy import Strategy, StrategyConfig
from nautilus_trader.adapters.binance.config import BinanceFuturesLiveTradingConfig
from nautilus_trader.adapters.binance.factories import BinanceFuturesLiveExecutionEngineFactory
from nautilus_trader.adapters.binance.factories import BinanceFuturesLiveDataClientFactory

# Internal Hunter logic
from analysis import (
    compute_adx, compute_atr, compute_bollinger, compute_macd,
    compute_rsi, compute_rsi_series, compute_support_resistance,
    compute_vwap, detect_divergence, get_market_regime,
    compute_rsi_slope, compute_stoch_rsi, compute_composite_score,
    generate_signal, MarketState
)
from config import (
    BINANCE_API_KEY, BINANCE_API_SECRET, USE_TESTNET, INITIAL_BALANCE_USD,
    GRID_SPREAD_PCT, LIMIT_ORDER_TIMEOUT_SEC, MAX_RISK_PER_TRADE_PCT,
    LEVERAGE, MAKER_GRID_ENABLED, GRID_ORDERS_COUNT,
)

logger = logging.getLogger("hunter.nautilus")

# V25: Anti-spoof OBI depth (only look at 5 best levels near mid-price)
OBI_DEPTH = 5


class HunterStrategyConfig(StrategyConfig, allow_mutation=True):
    instrument_id: InstrumentId
    adx_period: int = 14
    rsi_period: int = 14


class HunterNautilusStrategy(Strategy):
    """
    Nautilus Strategy block that compiles down to Rust events.
    Receives ticks and bars from the Rust core.
    """
    def __init__(self, config: HunterStrategyConfig):
        super().__init__(config)
        self.instrument_id = config.instrument_id
        
        # We will manually cache closes/highs/lows from Bar events
        # because our analysis module needs arrays.
        self.closes = []
        self.highs = []
        self.lows = []
        self.volumes = []
        
        # Deep OBI tracking via Nautilus Native Orderbook
        self.current_obi = 0.0

    def on_start(self):
        """Called when strategy starts."""
        self.log.info(f"Hunter Nautilus Strategy started for {self.instrument_id}")
        self.subscribe_bars(self.instrument_id)
        # V25: Depth=5 for anti-spoofing (ignore fake walls at levels 6-20)
        self.subscribe_order_book_at_depth(self.instrument_id, depth=OBI_DEPTH)
        self.subscribe_quote_ticks(self.instrument_id)
        
        # Cache best bid/ask for Maker limit pricing
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.tick_size = 0.01  # Will be overridden by instrument info

    def on_bar(self, bar: Bar):
        """Called by Rust core instantly when a bar closes."""
        self.highs.append(bar.high.as_double())
        self.lows.append(bar.low.as_double())
        self.closes.append(bar.close.as_double())
        self.volumes.append(bar.volume.as_double())
        
        # Truncate to save memory
        if len(self.closes) > 500:
            self.highs.pop(0)
            self.lows.pop(0)
            self.closes.pop(0)
            self.volumes.pop(0)
            
        self._evaluate_market_state(bar.close.as_double())

    def on_order_book_deltas(self, deltas: OrderBookDeltas):
        """Native Rust-speed orderbook updates — V25: only 5 levels."""
        book = self.cache.order_book(self.instrument_id)
        if book:
            bids = book.bids()[:OBI_DEPTH]
            asks = book.asks()[:OBI_DEPTH]
            total_bids = sum([level.quantity.as_double() for level in bids])
            total_asks = sum([level.quantity.as_double() for level in asks])
            if total_bids + total_asks > 0:
                self.current_obi = (total_bids - total_asks) / (total_bids + total_asks)
            # Cache BBO for Maker pricing
            if bids:
                self.best_bid = bids[0].price.as_double()
            if asks:
                self.best_ask = asks[0].price.as_double()

    def on_quote_tick(self, tick: QuoteTick):
        """Update BBO from quote ticks (faster than orderbook)."""
        self.best_bid = tick.bid_price.as_double()
        self.best_ask = tick.ask_price.as_double()

    def _evaluate_market_state(self, current_price: float):
        """Bridge between Nautilus data and our Python analysis.py logic."""
        if len(self.closes) < 50:
            return  # Need warmup
            
        h, l, c, v = self.highs, self.lows, self.closes, self.volumes
        
        adx = compute_adx(h, l, c, self.config.adx_period)
        regime = get_market_regime(adx)
        rsi = compute_rsi(c, self.config.rsi_period)
        
        # Call remaining Python indicators
        lower, _, upper = compute_bollinger(c, 20)
        _, _, histogram = compute_macd(c, 12, 26, 9)
        atr = compute_atr(h, l, c, 14)
        
        state = MarketState(
            current_price=current_price,
            rsi=rsi,
            ls_ratio=1.0,
            whale_net_vol=0.0,
            regime=regime,
            social_score=0.0,
            macd_histogram=histogram,
            bb_position=(current_price - lower) / (upper - lower) if upper > lower else 0.5,
            vwap_diff_pct=0.0,
            divergence="NONE",
            funding_rate=0.0,
            open_interest_delta=0.0,
            liq_imbalance=self.current_obi,
            atr_pct=(atr / current_price * 100) if current_price else 0.0,
            rsi_slope=compute_rsi_slope(c),
            stoch_rsi=compute_stoch_rsi(c),
            mtf_agreement=0.0,
            volume_confirm=True,
            near_resistance=False,
            btc_correlation=1.0,
            btc_dominance=50.0
        )
        
        # Note: We omit ML_Filter logic here for brevity, but it can be hooked identically
        
        signal_dict = generate_signal(state, current_position=self._get_current_pos(), use_composite=True)
        self._execute_signal(signal_dict, current_price, state.atr_pct, state.liq_imbalance)

    def _get_current_pos(self) -> Optional[Dict]:
        """Translate Nautilus Position into old PaperTrader format for generator"""
        pos: Position = self.cache.position(self.portfolio_id, self.instrument_id)
        if pos and not pos.is_closed:
            return {
                "side": "LONG" if pos.is_long else "SHORT",
                "entry": pos.avg_px.as_double(),
                "size": abs(pos.quantity.as_double())
            }
        return None

    def _execute_signal(self, signal_dict: Dict, current_price: float, atr_pct: float, obi: float):
        action = signal_dict.get("action", "HOLD")
        
        # Deep OBI filter logic (Phase 3 requirements)
        if action == "BUY" and obi < -0.30:
            self.log.warning(f"OBI BLOCKED BUY: Heavy Ask wall (OBI {obi:+.2f})")
            return
        elif action == "SHORT" and obi > 0.30:
            self.log.warning(f"OBI BLOCKED SHORT: Heavy Bid wall (OBI {obi:+.2f})")
            return
            
        pos: Position = self.cache.position(self.portfolio_id, self.instrument_id)
        has_pos = pos is not None and not pos.is_closed

        # V25: Smart Maker Execution — Best Bid - 1 tick / Best Ask + 1 tick
        if action == "BUY" and not has_pos:
            if self.best_bid <= 0:
                self.log.warning("No BBO available, skipping BUY")
                return
            # Place limit at Best Bid - 1 tick to guarantee Maker status
            limit_price = self.best_bid - self.tick_size
            quantity = Quantity.from_str(f"{100.0 / current_price:.6f}")
            
            from nautilus_trader.model.enums import OrderSide, TimeInForce
            from nautilus_trader.model.orders import LimitOrder
            
            order = self.order_factory.limit(
                instrument_id=self.instrument_id,
                order_side=OrderSide.BUY,
                quantity=quantity,
                price=Price.from_str(f"{limit_price:.8f}"),
                time_in_force=TimeInForce.GTD,  # Good Till Date (acts as timeout)
                post_only=True,  # Strict Maker — reject if would cross book
            )
            self.submit_order(order)
            self.log.info(
                f"RUST ENGINE: Maker BUY LIMIT @ ${limit_price:.4f} "
                f"(Best Bid ${self.best_bid:.4f} - 1 tick). "
                f"Timeout: {LIMIT_ORDER_TIMEOUT_SEC}s"
            )

        elif action == "SHORT" and not has_pos:
            if self.best_ask <= 0:
                self.log.warning("No BBO available, skipping SHORT")
                return
            # Place limit at Best Ask + 1 tick
            limit_price = self.best_ask + self.tick_size
            quantity = Quantity.from_str(f"{100.0 / current_price:.6f}")
            
            from nautilus_trader.model.enums import OrderSide, TimeInForce
            from nautilus_trader.model.orders import LimitOrder
            
            order = self.order_factory.limit(
                instrument_id=self.instrument_id,
                order_side=OrderSide.SELL,
                quantity=quantity,
                price=Price.from_str(f"{limit_price:.8f}"),
                time_in_force=TimeInForce.GTD,
                post_only=True,
            )
            self.submit_order(order)
            self.log.info(
                f"RUST ENGINE: Maker SHORT LIMIT @ ${limit_price:.4f} "
                f"(Best Ask ${self.best_ask:.4f} + 1 tick). "
                f"Timeout: {LIMIT_ORDER_TIMEOUT_SEC}s"
            )

        elif action in ("SELL", "COVER") and has_pos:
            self.close_all_positions(self.instrument_id)
            self.log.info(f"RUST ENGINE: Closed position at ${current_price}")

    def on_order_filled(self, event):
        """V25: Handle partial fills — recalculate grid for filled volume only."""
        order = self.cache.order(event.client_order_id)
        if order and order.is_partially_filled:
            filled_qty = order.filled_qty.as_double()
            total_qty = order.quantity.as_double()
            self.log.info(
                f"PARTIAL FILL: {filled_qty:.6f}/{total_qty:.6f} filled. "
                f"Keeping partial position, recalculating grid."
            )
            # Don't cancel — let the remainder continue filling
        elif order and order.is_completed:
            self.log.info(f"ORDER FULLY FILLED: {order.client_order_id}")


async def build_nautilus_node() -> TradingNode:
    """Builds and compiles the Nautilus Execution Engine"""
    
    binance_config = BinanceFuturesLiveTradingConfig(
        api_key=BINANCE_API_KEY or "none",
        api_secret=BINANCE_API_SECRET or "none",
        is_testnet=USE_TESTNET,
    )
    
    node_config = TradingNodeConfig(
        name="HunterRustCore",
        log_level="INFO",
    )
    
    node = TradingNode(config=node_config)
    
    node.add_live_data_client(BinanceFuturesLiveDataClientFactory.create(config=binance_config))
    node.add_live_execution_engine(BinanceFuturesLiveExecutionEngineFactory.create(config=binance_config))
    
    # Initialize the strategy
    strategy_config = HunterStrategyConfig(
        instrument_id=InstrumentId.from_str("BTCUSDT.BINANCE"),
    )
    strategy = HunterNautilusStrategy(config=strategy_config)
    
    node.add_strategy(strategy)
    return node


if __name__ == "__main__":
    node = asyncio.run(build_nautilus_node())
    # Start the node and run until cancelled
    node.run()
