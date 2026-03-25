"""
Hunter V17 — Data Provider
===========================
Handles all asynchronous HTTP requests to Binance Futures.
Strictly separated from Trading/Analysis logic (Single Responsibility Principle).
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Any, AsyncGenerator

try:
    from hunter_core import OrderBook
except ImportError:
    # Minimal fallback placeholder
    class OrderBook: 
        def update(self, b, a): pass
        def get_obi(self): return 0.0

import aiohttp

from config import (
    BASE_URL,
    BLACKLIST,
    KLINE_LIMIT,
    TIMEFRAME,
    TOP_PAIRS_COUNT,
    USE_TESTNET,
)

logger = logging.getLogger("hunter.provider")


class BinanceProvider:
    """Async provider for Binance Futures market data."""

    def __init__(self):
        # We will share a single session for all requests in a cycle
        self.session = None
        self.bbo_cache: Dict[str, Dict[str, float]] = {}  # V23 Phase 2
        self.depth_cache: Dict[str, Dict] = {}
        self.cvd_cache: Dict[str, float] = {}  # V28 Phase 2 (Cumulative Volume Delta)
        self.rust_orderbooks: Dict[str, OrderBook] = {}
        self.last_price_cache: Dict[str, float] = {}  # V32: Tick-level price tracking
        self.volume_profile: Dict[str, Dict[float, float]] = {}  # V33: VPVR histogram

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"User-Agent": "HunterBot/17.0"}
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    async def _api_get(self, endpoint: str, params: dict = None) -> Any:
        """Internal helper for async GET requests."""
        url = f"{BASE_URL}{endpoint}"
        try:
            async with self.session.get(url, params=params, timeout=10) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error("❌ Binance API Error on %s: %s", endpoint, e)
            return None

    async def fetch_ohlcv(
        self, symbol: str, interval: str = TIMEFRAME, limit: int = KLINE_LIMIT
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Fetch OHLCV candles. Returns empty lists on failure (graceful degradation)."""
        data = await self._api_get(
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": interval, "limit": str(limit)},
        )
        if not data:
            logger.warning("⚠️ No OHLCV data for %s — symbol may be delisted or unavailable.", symbol)
            return [], [], [], []

        highs = [float(k[2]) for k in data]
        lows = [float(k[3]) for k in data]
        closes = [float(k[4]) for k in data]
        volumes = [float(k[5]) for k in data]

        return highs, lows, closes, volumes

    async def fetch_ls_ratio(self, symbol: str) -> float:
        """Fetch global Long/Short account ratio."""
        if USE_TESTNET: return 1.0
        data = await self._api_get(
            "/futures/data/globalLongShortAccountRatio",
            {"symbol": symbol, "period": "1h", "limit": "1"},
        )
        if data and isinstance(data, list) and len(data) > 0:
            return float(data[0].get("longShortRatio", 1.0))
        return 1.0

    async def fetch_whale_net_volume(self, symbol: str) -> float:
        """Fetch taker buy/sell volume to estimate whale activity."""
        if USE_TESTNET: return 0.0
        data = await self._api_get(
            "/futures/data/takerlongshortRatio",
            {"symbol": symbol, "period": "1h", "limit": "1"},
        )
        if data and isinstance(data, list) and len(data) > 0:
            buy_vol = float(data[0].get("buyVol", 0))
            sell_vol = float(data[0].get("sellVol", 0))
            return buy_vol - sell_vol
        return 0.0

    async def fetch_funding_rate(self, symbol: str) -> float:
        """Fetch current funding rate."""
        data = await self._api_get(
            "/fapi/v1/fundingRate", {"symbol": symbol, "limit": "1"}
        )
        if data and isinstance(data, list) and len(data) > 0:
            return float(data[0].get("fundingRate", 0.0))
        return 0.0

    async def stream_klines(
        self, symbol: str, interval: str = TIMEFRAME
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Connects to Binance Futures WSS for continuous kline updates.
        Yields the 'k' (kline) dictionary from the stream.
        Handles automatic reconnection.
        """
        url = f"wss://fstream.binance.com/ws/{symbol.lower()}@kline_{interval}"
        logger.info("🔌 Initializing WSS connection to %s", url)
        
        while True:
            try:
                async with self.session.ws_connect(url, timeout=30) as ws:
                    logger.info("🟢 WSS Connected to %s", url)
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if 'e' in data and data['e'] == 'kline' and 'k' in data:
                                yield data['k']
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            logger.warning("🟠 WSS stream closed/error, breaking to reconnect...")
                            break
            except asyncio.CancelledError:
                logger.info("🛑 WSS stream cancelled.")
                raise
            except Exception as e:
                logger.error("❌ WSS Error on %s: %s. Reconnecting in 5s...", symbol, e)
                await asyncio.sleep(5)

    async def stream_bbo(self, symbol: str) -> None:
        """
        V23 Phase 2: Connects to Binance bookTicker WSS.
        Runs infinitely in background updating the BBO cache.
        """
        url = f"wss://fstream.binance.com/ws/{symbol.lower()}@bookTicker"
        logger.info("🔌 Initializing BBO WSS connection to %s", url)
        
        while True:
            try:
                async with self.session.ws_connect(url, timeout=30) as ws:
                    logger.info("🟢 BBO WSS Connected for %s", symbol)
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if 'b' in data and 'a' in data:
                                # Process bookTicker update
                                self.bbo_cache[symbol] = {
                                    'bid_price': float(data['b']),
                                    'bid_qty': float(data['B']),
                                    'ask_price': float(data['a']),
                                    'ask_qty': float(data['A'])
                                }
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
            except asyncio.CancelledError:
                logger.info("🛑 BBO WSS stream cancelled for %s.", symbol)
                raise
            except Exception as e:
                logger.error("❌ BBO WSS Error on %s: %s. Reconnecting in 5s...", symbol, e)
                await asyncio.sleep(5)

    def get_bbo(self, symbol: str) -> Optional[Dict[str, float]]:
        """Return cached BBO data for symbol."""
        return self.bbo_cache.get(symbol)

    async def stream_depth(self, symbol: str) -> None:
        """
        V24 Phase 3: Connects to Binance deep orderbook WSS (@depth20@100ms).
        Parses top 20 levels of bids and asks to calculate true liquidity walls.
        """
        url = f"wss://fstream.binance.com/ws/{symbol.lower()}@depth20@100ms"
        logger.info("🔌 Initializing Depth20 WSS connection to %s", url)
        
        while True:
            try:
                async with self.session.ws_connect(url, timeout=30) as ws:
                    logger.info("🟢 DEPTH WSS Connected for %s", symbol)
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if 'b' in data and 'a' in data:
                                # Data format: 'b': [[price, qty], [price, qty]...], 'a': [...]
                                bids_data = [[float(p), float(q)] for p, q in data['b']]
                                asks_data = [[float(p), float(q)] for p, q in data['a']]
                                
                                # Use Rust engine for OBI
                                if symbol not in self.rust_orderbooks:
                                    self.rust_orderbooks[symbol] = OrderBook()
                                
                                self.rust_orderbooks[symbol].update(bids_data, asks_data)

                                self.depth_cache[symbol] = {
                                    'deep_bid_vol': sum(l[1] for l in bids_data),
                                    'deep_ask_vol': sum(l[1] for l in asks_data),
                                    'bids': bids_data,
                                    'asks': asks_data
                                }
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
            except asyncio.CancelledError:
                logger.info("🛑 DEPTH WSS stream cancelled for %s.", symbol)
                raise
            except Exception as e:
                logger.error("❌ DEPTH WSS Error on %s: %s. Reconnecting in 5s...", symbol, e)
                await asyncio.sleep(5)

    def get_deep_obi(self, symbol: str) -> float:
        """
        V24 Phase 3: Returns the Order Book Imbalance (OBI) from the top 20 levels.
        OBI = (BidVol - AskVol) / (BidVol + AskVol)
        Returns a float between -1.0 (Heavy Sell Walls) and +1.0 (Heavy Buy Walls).
        """
        if symbol in self.rust_orderbooks:
            return self.rust_orderbooks[symbol].get_obi()
            
        depth = self.depth_cache.get(symbol)
        if not depth:
            return 0.0
            
        bids = depth['deep_bid_vol']
        asks = depth['deep_ask_vol']
        total = bids + asks
        
        if total == 0:
            return 0.0
            
        return (bids - asks) / total

    async def stream_agg_trades(self, symbol: str) -> None:
        """
        V28 Phase 2: Connects to Binance @aggTrade WSS.
        Calculates real-time Cumulative Volume Delta (CVD) for the current candle.
        """
        url = f"wss://fstream.binance.com/ws/{symbol.lower()}@aggTrade"
        logger.info("🔌 Initializing CVD WSS connection to %s", url)
        self.cvd_cache[symbol] = 0.0
        
        while True:
            try:
                async with self.session.ws_connect(url, timeout=30) as ws:
                    logger.info("🟢 CVD WSS Connected for %s", symbol)
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if 'm' in data and 'q' in data and 'p' in data:
                                # 'm' (is_buyer_maker) = True means it was a SELL order (maker was buyer)
                                # 'm' = False means it was a BUY order
                                price = float(data['p'])
                                qty = float(data['q'])
                                vol_usd = qty * price
                                if data['m']:
                                    self.cvd_cache[symbol] -= vol_usd
                                else:
                                    self.cvd_cache[symbol] += vol_usd
                                
                                # V33: Aggregate Volume Profile (0.1% price buckets)
                                bucket = round(price / (price * 0.001)) * (price * 0.001)
                                if symbol not in self.volume_profile:
                                    self.volume_profile[symbol] = {}
                                self.volume_profile[symbol][bucket] = self.volume_profile[symbol].get(bucket, 0.0) + vol_usd
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
            except asyncio.CancelledError:
                logger.info("🛑 CVD WSS stream cancelled for %s.", symbol)
                raise
            except Exception as e:
                logger.error("❌ CVD WSS Error on %s: %s. Reconnecting in 5s...", symbol, e)
                await asyncio.sleep(5)

    def get_cvd(self, symbol: str) -> float:
        """Return the current Cumulative Volume Delta (USD) for the symbol."""
        return self.cvd_cache.get(symbol, 0.0)

    def reset_cvd(self, symbol: str) -> None:
        """Reset the CVD for the symbol (call this on candle close)."""
        self.cvd_cache[symbol] = 0.0

    def get_last_price(self, symbol: str) -> float:
        """V32: Return the latest tick-level mark price for a symbol."""
        return self.last_price_cache.get(symbol, 0.0)

    def get_vpvr_poc(self, symbol: str) -> float:
        """V33: Return the Point of Control (highest volume price level)."""
        profile = self.volume_profile.get(symbol, {})
        if not profile:
            return 0.0
        return max(profile, key=profile.get)

    def get_vpvr_support_resistance(self, symbol: str, current_price: float, n: int = 3) -> tuple:
        """V33: Return top N volume nodes as (supports_below, resistances_above)."""
        profile = self.volume_profile.get(symbol, {})
        if not profile:
            return [], []
        
        # Sort by volume descending, take top N*2 levels
        top_levels = sorted(profile.items(), key=lambda x: x[1], reverse=True)[:n * 2]
        supports = sorted([p for p, _ in top_levels if p < current_price], reverse=True)[:n]
        resistances = sorted([p for p, _ in top_levels if p > current_price])[:n]
        return supports, resistances

    async def stream_mark_price(self, symbol: str) -> None:
        """
        V32: Connects to Binance @markPrice WSS (updates every 1s).
        Provides tick-level price data for real-time SL monitoring.
        """
        url = f"wss://fstream.binance.com/ws/{symbol.lower()}@markPrice@1s"
        
        while True:
            try:
                async with self.session.ws_connect(url, timeout=30) as ws:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if 'p' in data:
                                self.last_price_cache[symbol] = float(data['p'])
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(5)

    async def fetch_open_interest_delta(self, symbol: str) -> float:
        """Fetch % % change in Open Interest over the last hour."""
        if USE_TESTNET: return 0.0
        data = await self._api_get(
            "/futures/data/openInterestHist",
            {"symbol": symbol, "period": "15m", "limit": "4"},
        )
        if data and isinstance(data, list) and len(data) >= 2:
            try:
                oi_old = float(data[0]["sumOpenInterestValue"])
                oi_new = float(data[-1]["sumOpenInterestValue"])
                if oi_old > 0:
                    return ((oi_new - oi_old) / oi_old) * 100.0
            except (KeyError, ValueError, IndexError):
                pass
        return 0.0

    async def fetch_liquidation_data(self, symbol: str) -> float:
        """
        Fetch recent liquidation orders and calculate net liquidation imbalance in USD.
        Positive = Short Liquidations > Long Liquidations (Squeeze Up / Top signal)
        Negative = Long Liquidations > Short Liquidations (Squeeze Down / Bottom signal)
        """
        if USE_TESTNET: return 0.0
        data = await self._api_get(
            "/fapi/v1/allForceOrders",
            {"symbol": symbol, "limit": "50"}
        )
        if not data or not isinstance(data, list):
            return 0.0

        net_usd = 0.0
        for order in data:
            try:
                qty = float(order.get("executedQty", 0))
                price = float(order.get("averagePrice", 0))
                side = order.get("side", "")
                vol_usd = qty * price
                if side == "BUY": # Shorts liquidated
                    net_usd += vol_usd
                elif side == "SELL": # Longs liquidated
                    net_usd -= vol_usd
            except (ValueError, TypeError):
                pass
        return net_usd

    async def scan_top_pairs(self, count: int = TOP_PAIRS_COUNT) -> List[str]:
        """Fetch top USDT-margined futures pairs by 24h quote volume."""
        logger.info("🔍 Scanning top %d futures pairs by 24h volume …", count)
        tickers = await self._api_get("/fapi/v1/ticker/24hr")
        if not tickers:
            return ["BTCUSDT"]

        usdt_pairs = [
            t for t in tickers
            if t.get("symbol", "").endswith("USDT")
            and t.get("symbol") not in BLACKLIST
        ]

        usdt_pairs.sort(key=lambda t: float(t.get("quoteVolume", 0)), reverse=True)
        top = [t["symbol"] for t in usdt_pairs[:count]]
        logger.info("📋 Top pairs: %s", ", ".join(top))
        return top

    async def fetch_all_market_data(
        self, symbol: str, mtf_intervals: List[str]
    ) -> Dict[str, Any]:
        """
        Concurrency! Fetch ALL required data for a single symbol at once.
        Returns a dict containing all raw market data.
        """
        # Create tasks
        ohlcv_task = asyncio.create_task(self.fetch_ohlcv(symbol))
        ls_task = asyncio.create_task(self.fetch_ls_ratio(symbol))
        whale_task = asyncio.create_task(self.fetch_whale_net_volume(symbol))
        funding_task = asyncio.create_task(self.fetch_funding_rate(symbol))
        oi_task = asyncio.create_task(self.fetch_open_interest_delta(symbol))
        liq_task = asyncio.create_task(self.fetch_liquidation_data(symbol))

        # MTF tasks
        mtf_tasks = {
            tf: asyncio.create_task(self.fetch_ohlcv(symbol, interval=tf))
            for tf in mtf_intervals
        }

        # Await primary OHLCV first because if it fails, no point waiting for the rest
        try:
            highs, lows, closes, volumes = await ohlcv_task
        except Exception as e:
            logger.error("Primary OHLCV failed for %s", symbol)
            raise e

        # Await side-data
        ls_ratio, whale_vol, funding_rate, oi_delta, liq_imbalance = await asyncio.gather(
            ls_task, whale_task, funding_task, oi_task, liq_task, return_exceptions=True
        )

        # Handle potential side-data exceptions (fallback to defaults if failed)
        ls_ratio = ls_ratio if isinstance(ls_ratio, float) else 1.0
        whale_vol = whale_vol if isinstance(whale_vol, float) else 0.0
        funding_rate = funding_rate if isinstance(funding_rate, float) else 0.0
        oi_delta = oi_delta if isinstance(oi_delta, float) else 0.0
        liq_imbalance = liq_imbalance if isinstance(liq_imbalance, float) else 0.0

        # Await MTF
        mtf_data = {}
        for tf, task in mtf_tasks.items():
            try:
                _, _, mtf_closes, _ = await task
                mtf_data[tf] = mtf_closes
            except Exception:
                mtf_data[tf] = []

        return {
            "symbol": symbol,
            "highs": highs,
            "lows": lows,
            "closes": closes,
            "volumes": volumes,
            "ls_ratio": ls_ratio,
            "whale_vol": whale_vol,
            "funding_rate": funding_rate,
            "oi_delta": oi_delta,
            "liq_imbalance": liq_imbalance,
            "mtf_closes": mtf_data,  # Dict[tf, List[closes]]
        }
