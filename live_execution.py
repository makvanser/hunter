"""
Hunter V21 — Live Execution Module (Testnet & Mainnet)
======================================================
Handles cryptographic signing and actual order placement 
on the Binance Futures exchange. In-memory trailing SL/TP
logic is maintained within the engine to bypass complex 
API condition order endpoints (Algo Orders).
"""

import time
import hmac
import hashlib
import aiohttp
import asyncio
import logging
import json
import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from config import (
    API_KEY, API_SECRET, BASE_URL, 
    MAX_EXPOSURE_USD, TRADE_SIZE_USD, MAX_OPEN_POSITIONS, LEVERAGE,
    ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER, LIMIT_ORDER_TIMEOUT_SEC,
    MAKER_GRID_ENABLED, GRID_ORDERS_COUNT, GRID_SPREAD_PCT,
    DYNAMIC_TP_ENABLED, DYNAMIC_TP_MAX_MULT,
    TRAILING_SL_ENABLED, TRAILING_SL_ACTIVATION_PCT, TRAILING_SL_ATR_MULT,
    USE_TESTNET, USE_DYNAMIC_SIZING, MAX_RISK_PER_TRADE_PCT, INITIAL_BALANCE_USD,
    KELLY_ENABLED, KELLY_FRACTION, KELLY_MIN_TRADES
)
from database import log_trade, get_consecutive_losses, init_db, load_positions, save_position, delete_position
from execution import PaperTrader

logger = logging.getLogger("hunter.live_execution")

class LiveTrader(PaperTrader):
    """
    Executes live trades on Binance Futures via REST API.
    Inherits all core risk management and state logic from PaperTrader, 
    but overrides the execution hooks to hit real Binance endpoints via aiohttp.
    """

    def __init__(self, db_path="hunter_live.db"):
        # Initialise PaperTrader core (loads DB, configures limits)
        super().__init__(db_path=db_path)
        
        if not API_KEY or not API_SECRET:
            logger.error("❌ API_KEY or API_SECRET missing in config!")
            raise ValueError("Live Trading requires API keys.")
            
        # V22: Track real exchange balance for accurate drawdown calculation
        self._initial_balance_synced = False
        self._initial_balance = INITIAL_BALANCE_USD  # fallback until first sync
        self._symbol_infoCache: Dict[str, Dict[str, float]] = {}
        
        # V28 Phase 3: WebSocket Execution RPC
        self.ws_execution = None
        self.ws_listening_task = None
        self.ws_futures: Dict[str, asyncio.Future] = {}
        
        logger.info("✅ LiveTrader initialized for %s", "TESTNET" if USE_TESTNET else "MAINNET")
        
        # Note: balance is automatically fetched asynchronously later during the cycle.
        # self.balance is initialized from INITIAL_BALANCE_USD locally for safety, but we strictly 
        # use async fetched balance where possible.

    def _sign_payload(self, params: Dict[str, Any]) -> str:
        """Sign request parameters using HMAC-SHA256."""
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            API_SECRET.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    async def connect_ws(self):
        """V28 Phase 3: Establish a persistent RPC WebSocket connection for orders."""
        if self.ws_execution and not self.ws_execution.closed:
            return
            
        ws_url = "wss://testnet.binancefuture.com/ws-fapi/v1" if USE_TESTNET else "wss://ws-fapi.binance.com/ws-fapi/v1"
        logger.info("🔌 Connecting Execution WS to %s...", ws_url)
        
        try:
            self.ws_session = aiohttp.ClientSession()
            self.ws_execution = await self.ws_session.ws_connect(ws_url, timeout=30)
            logger.info("🟢 Execution WS Connected successfully!")
            self.ws_listening_task = asyncio.create_task(self._ws_listen_loop())
        except Exception as e:
            logger.error("❌ Failed to connect Execution WS: %s", e)

    async def _ws_listen_loop(self):
        """Continuously listen for responses from the WebSocket and resolve Futures."""
        try:
            async for msg in self.ws_execution:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    req_id = data.get('id')
                    if req_id and req_id in self.ws_futures:
                        if not self.ws_futures[req_id].done():
                            self.ws_futures[req_id].set_result(data)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("❌ Execution WS Listen Loop Error: %s", e)
        finally:
            logger.warning("🟠 Execution WS Disconnected! Reconnecting...")
            self.ws_execution = None
            asyncio.create_task(self.connect_ws())

    async def _ws_request(self, method: str, params: Dict[str, Any]) -> Dict:
        """Send a signed WS API request and wait for the response."""
        if not self.ws_execution or self.ws_execution.closed:
            await self.connect_ws()
            
        req_id = str(uuid.uuid4())
        fut = asyncio.Future()
        self.ws_futures[req_id] = fut
        
        params['apiKey'] = API_KEY
        params['timestamp'] = int(time.time() * 1000)
        params['signature'] = self._sign_payload(params)
        
        payload = {
            "id": req_id,
            "method": method,
            "params": params
        }
        
        try:
            await self.ws_execution.send_json(payload)
            # Wait for response with timeout
            response = await asyncio.wait_for(fut, timeout=10.0)
            
            # The WS API returns actual data in the 'result' field, and errors in 'error'
            if "error" in response:
                err = response["error"]
                logger.error("❌ WS Exec Error [%s]: %s", method, err)
                return {"error": err}
                
            return response.get("result", response)
            
        except asyncio.TimeoutError:
            logger.error("⏱️ WS Exec Timeout waiting for %s %s", method, req_id)
            return {"error": "Timeout"}
        except Exception as e:
            logger.error("❌ WS Exec Failed %s: %s", method, e)
            return {"error": str(e)}
        finally:
            self.ws_futures.pop(req_id, None)

    async def _api_post(self, endpoint: str, payload: Dict[str, Any]) -> Dict:
        """Fetch USDT balance from Binance Futures."""
        payload = {'timestamp': int(time.time() * 1000)}
        payload['signature'] = self._sign_payload(payload)
        url = f"{BASE_URL}/fapi/v2/balance"
        
        headers = {"X-MBX-APIKEY": API_KEY}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    for asset in data:
                        if asset['asset'] == 'USDT':
                            real_bal = float(asset['balance'])
                            self.balance = real_bal
                            # V22: Set initial balance from actual exchange 
                            # balance on first sync to prevent false drawdown
                            if not self._initial_balance_synced:
                                self._initial_balance_synced = True
                                self._initial_balance = real_bal
                                logger.info("💰 Initial balance synced from exchange: $%.2f", real_bal)
                            return real_bal
        return self.balance

    async def _get_symbol_info(self, symbol: str) -> Dict[str, float]:
        """Fetch and cache precision rules for a symbol."""
        if symbol in self._symbol_infoCache:
            return self._symbol_infoCache[symbol]
            
        url = f"{BASE_URL}/fapi/v1/exchangeInfo"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    for sym in data.get('symbols', []):
                        if sym['symbol'] == symbol:
                            tick_size = 0.01
                            step_size = 0.001
                            for f in sym.get('filters', []):
                                if f['filterType'] == 'PRICE_FILTER':
                                    tick_size = float(f['tickSize'])
                                elif f['filterType'] == 'LOT_SIZE':
                                    step_size = float(f['stepSize'])
                            
                            self._symbol_infoCache[symbol] = {
                                'tickSize': tick_size,
                                'stepSize': step_size
                            }
                            return self._symbol_infoCache[symbol]
        
        # Fallback defaults if fetch fails
        return {'tickSize': 0.01, 'stepSize': 0.001}

    async def set_leverage(self, symbol: str, leverage: int = 2) -> bool:
        """Set margin leverage prior to trading."""
        res = await self._api_post("/fapi/v1/leverage", {"symbol": symbol, "leverage": str(leverage)})
        return "error" not in res

    async def open_market_order(self, symbol: str, side: str, quantity: float) -> float:
        """Place a market order using WS API. Return executed price."""
        payload = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": str(quantity)}
        res = await self._ws_request("order.place", payload)
        if "error" in res:
            logger.error("💥 Failed to open %s market order: %s", side, res)
            return -1.0
            
        order_id = res.get("orderId")
        avg_price = float(res.get('avgPrice', 0) or 0)
        
        if avg_price == 0.0 and order_id:
            await asyncio.sleep(0.5)
            check = await self._ws_request("order.status", {"symbol": symbol, "orderId": order_id})
            avg_price = float(check.get('avgPrice', 0) or 0)
            
        logger.info("🚀 LIVE ORDER EXECUTED (WS): %s %s size=%s", side, symbol, quantity)
        return avg_price

    async def place_exchange_sl(self, symbol: str, side: str, quantity: float, stop_price: float) -> bool:
        """
        V32: Place a real STOP_MARKET order on Binance as a safety net.
        This ensures the exchange closes the position even if the bot crashes.
        
        Args:
            side: The CLOSING side ("SELL" for longs, "BUY" for shorts)
            stop_price: The price at which the stop triggers
        """
        # First cancel any existing SL orders for this symbol
        await self._cancel_all_sl_orders(symbol)
        
        info = await self._get_symbol_info(symbol)
        qty_str = self._round_to_step(quantity, info['stepSize'])
        price_str = self._round_to_step(stop_price, info['tickSize'])
        
        payload = {
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "quantity": qty_str,
            "stopPrice": price_str,
            "closePosition": "true",
            "workingType": "MARK_PRICE"
        }
        
        res = await self._ws_request("order.place", payload)
        if "error" in res:
            logger.error("❌ Failed to place exchange SL for %s: %s", symbol, res)
            return False
        
        logger.info("🛡️ EXCHANGE SL PLACED: %s %s @ %.4f (orderId=%s)", 
                    side, symbol, stop_price, res.get('orderId', '?'))
        return True

    async def _cancel_all_sl_orders(self, symbol: str) -> None:
        """Cancel all open STOP_MARKET orders for a symbol before placing a new one."""
        res = await self._ws_request("openOrders.status", {"symbol": symbol})
        if isinstance(res, list):
            for order in res:
                if order.get('type') == 'STOP_MARKET' and order.get('status') == 'NEW':
                    await self._ws_request("order.cancel", {
                        "symbol": symbol, "orderId": order['orderId']
                    })
                    logger.debug("🗑️ Cancelled old SL order %s for %s", order['orderId'], symbol)

    def _round_to_step(self, value: float, step: float) -> str:
        """Round value to the nearest step size, returning a clean string for Binance."""
        if step <= 0: return str(value)
        import math
        precision = max(0, int(round(-math.log10(step))))
        rounded = round(int(value / step) * step, precision)
        return f"{rounded:.{precision}f}".rstrip('0').rstrip('.') if precision > 0 else str(int(rounded))

    async def open_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> float:
        """Place POST_ONLY limit order(s). If Maker Grid is enabled, place multiple layered bids."""
        info = await self._get_symbol_info(symbol)
        
        orders = []
        if MAKER_GRID_ENABLED and GRID_ORDERS_COUNT > 1:
            chunk_qty = quantity / GRID_ORDERS_COUNT
            for i in range(GRID_ORDERS_COUNT):
                # Spread price further away from current price
                spread_mult = 1.0 - (GRID_SPREAD_PCT / 100.0 * i) if side == "BUY" else 1.0 + (GRID_SPREAD_PCT / 100.0 * i)
                grid_price = price * spread_mult
                qty_str = self._round_to_step(chunk_qty, info['stepSize'])
                if float(qty_str) <= 0:
                    qty_str = str(info['stepSize'])
                orders.append({
                    "qty": qty_str,
                    "price": self._round_to_step(grid_price, info['tickSize'])
                })
        else:
            orders.append({
                "qty": self._round_to_step(quantity, info['stepSize']),
                "price": self._round_to_step(price, info['tickSize'])
            })
            
        order_ids = []
        for o in orders:
            payload = {
                "symbol": symbol, "side": side, "type": "LIMIT", "timeInForce": "GTX",
                "quantity": o["qty"], "price": o["price"]
            }
            res = await self._ws_request("order.place", payload)
            if "orderId" in res:
                order_ids.append(res["orderId"])
            else:
                logger.error("💥 Failed to open %s limit grid order chunk: %s", side, res)
                
        if not order_ids:
            return -1.0
            
        timeout = LIMIT_ORDER_TIMEOUT_SEC
        start_time = time.time()
        logger.info("⏳ Waiting for %s GRID LIMIT %d orders to fill (total size: %s)...", side, len(order_ids), quantity)
        
        filled_qty = 0.0
        total_spent = 0.0
        
        while time.time() - start_time < timeout:
            await asyncio.sleep(1)
            
            # Poll all active orders via WS
            active_ids = list(order_ids) # copy
            for oid in active_ids:
                check = await self._ws_request("order.status", {"symbol": symbol, "orderId": oid})
                status = check.get("status")
                
                if status == "FILLED":
                    exec_qty = float(check.get('executedQty', 0))
                    avg_px = float(check.get('avgPrice', 0))
                    filled_qty += exec_qty
                    total_spent += exec_qty * avg_px
                    order_ids.remove(oid) # Done tracking this chunk
                    logger.info("🚀 GRID CHUNK EXECUTED (WS): %s %s size=%s @ %.4f", side, symbol, exec_qty, avg_px)
                    
            if not order_ids:
                break # All filled!
                
        # Cancel remaining
        for oid in order_ids:
            await self._ws_request("order.cancel", {"symbol": symbol, "orderId": oid})
            
        if filled_qty > 0:
            final_avg_price = total_spent / filled_qty
            logger.info("✅ GRID COMPLETE: Executed total %.3f at Avg Price %.4f", filled_qty, final_avg_price)
            # Override quantity parameter inside the caller requires changing the return signature, 
            # but since V24 only uses this entry price to calculate SL/TP correctly:
            return final_avg_price
            
        logger.warning("⚠️ Limit grid timeout (%ds) with no fills.", timeout)
        return -1.0

    # ── ASYNC OVERRIDES ────────────────────────────────────────────────────────

    async def handle_tick(self, symbol: str, current_price: float, atr: float = 0.0) -> Optional[Dict]:
        """
        V32: Lightweight tick monitor. Checks for SL/TP and updates trailing SL 
        without syncing balances or making unnecessary REST checks. 
        Returns result dict only if a position was closed.
        """
        if symbol not in self.positions:
            return None
            
        result: Dict = {
            "action":    "NONE",
            "signal":    "TICK_UPDATE",
            "symbol":    symbol,
            "price":     current_price,
            "pnl":       0.0,
            "blocked":   False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # 1. Update Trailing SL (Synchronous logic + async exchange sync)
        if atr > 0:
            old_sl = self.positions[symbol].get('stop_loss')
            self._update_trailing_sl(symbol, current_price, atr)
            new_sl = self.positions[symbol].get('stop_loss')
            
            # If trailing SL moved, update the exchange-side STOP_MARKET order
            if new_sl and old_sl and new_sl != old_sl:
                pos = self.positions[symbol]
                close_side = "SELL" if pos['side'] == "BUY" else "BUY"
                qty = pos.get('quantity', 0)
                if qty > 0:
                    asyncio.create_task(self.place_exchange_sl(symbol, close_side, qty, new_sl))

        # 2. Check Software SL/TP hit (e.g. for Take-Profits or if exchange SL failed)
        sl_tp_trigger = self.check_sl_tp(symbol, current_price)
        if sl_tp_trigger is not None:
            logger.info("⚡ TICK-LEVEL TRIGGER: %s hit on %s at $%.4f", sl_tp_trigger, symbol, current_price)
            # Fetch balance right before closing
            await self.sync_balance()
            return await self._close_position(symbol, current_price, result, sl_tp_trigger)
            
        return None

    async def execute_trade(self, signal: str, current_price: float, symbol: str, atr: float = 0.0, provider=None) -> Dict:
        """
        Async orchestration of the PaperTrader cycle logic.
        """
        result: Dict = {
            "action":    "NONE",
            "signal":    signal,
            "symbol":    symbol,
            "price":     current_price,
            "pnl":       0.0,
            "blocked":   False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Sync actual balance before execution
        await self.sync_balance()

        # Gate 1: Global circuit breaker
        if not self.check_circuit_breaker():
            result["action"], result["blocked"] = "BLOCKED_BY_CIRCUIT_BREAKER", True
            return result

        # Gate 2: Trailing SL update + V32: Sync exchange SL when trailing stop moves
        if atr > 0 and symbol in self.positions:
            old_sl = self.positions[symbol].get('stop_loss')
            self._update_trailing_sl(symbol, current_price, atr)
            new_sl = self.positions[symbol].get('stop_loss')
            # If trailing SL moved, update the exchange-side STOP_MARKET order
            if new_sl and old_sl and new_sl != old_sl:
                pos = self.positions[symbol]
                close_side = "SELL" if pos['side'] == "BUY" else "BUY"
                qty = pos.get('quantity', 0)
                if qty > 0:
                    await self.place_exchange_sl(symbol, close_side, qty, new_sl)

        # Gate 3: SL/TP auto-close check (Synchronous check, Async execution)
        sl_tp_trigger = self.check_sl_tp(symbol, current_price)
        if sl_tp_trigger is not None:
            return await self._close_position(symbol, current_price, result, sl_tp_trigger)

        # Gate 4: No-op
        if signal == "HOLD":
            result["action"] = "HOLD"
            return result

        # ── OPEN LONG
        if signal == "BUY" and symbol not in self.positions:
            return await self._open_position(symbol, current_price, "BUY", atr, result, provider)

        # ── OPEN SHORT
        if signal == "SHORT" and symbol not in self.positions:
            if not SHORT_ENABLED:
                result["action"] = "SHORTS_DISABLED"
                return result
            return await self._open_position(symbol, current_price, "SELL", atr, result, provider)

        # ── CLOSE LONG
        if signal == "SELL" and symbol in self.positions:
            if self.positions[symbol]["side"] == "BUY":
                return await self._close_position(symbol, current_price, result, "SIGNAL_SELL")

        # ── CLOSE SHORT
        if signal == "COVER" and symbol in self.positions:
            if self.positions[symbol]["side"] == "SELL":
                return await self._close_position(symbol, current_price, result, "SIGNAL_COVER")

        return result

    async def _open_position(self, symbol: str, current_price: float, side: str, atr: float, result: Dict, provider=None) -> Dict:
        if len(self.positions) >= MAX_OPEN_POSITIONS:
            result["action"] = "MAX_POSITIONS_REACHED"
            return result

        total_exposure = sum(p["size_usd"] for p in self.positions.values())
        
        # V25: Always use Kelly-based sizing (1/4 Kelly conservative model)
        size_usd = self._kelly_position_size()
        
        # Hard cap: never exceed MAX_RISK_PER_TRADE_PCT of balance
        max_risk_usd = self.balance * (MAX_RISK_PER_TRADE_PCT / 100.0)
        size_usd = min(size_usd, max_risk_usd, MAX_EXPOSURE_USD - total_exposure)

        # V32: Sane minimum (Binance min notional is ~$5 for most pairs)
        if size_usd < 5.0:
            result["action"] = "INSUFFICIENT_BALANCE"
            logger.warning("⚠️ Position size $%.2f too small for %s. Skipping.", size_usd, symbol)
            return result

        # V32: Leverage-aware quantity calculation
        # size_usd = margin. Notional = margin * leverage.
        notional = size_usd * LEVERAGE
        quantity = round(notional / current_price, 3)
        if quantity <= 0: quantity = 0.001
        
        if USE_TESTNET and quantity < 0.05:
            quantity = 0.05  # Force minimum testnet order volume
        
        # V23 Phase 1: Institutional Limit Execution for Entries
        avg_price = await self.open_limit_order(symbol, side, quantity, current_price)
        if avg_price < 0:
            result["action"], result["blocked"] = "API_LIMIT_FAILED", True
            return result
        
        # Use API execution price
        entry = avg_price
        
        stop_loss, take_profit = None, None
        if atr > 0:
            tp_mult = ATR_TP_MULTIPLIER
            if DYNAMIC_TP_ENABLED:
                tp_mult = min(DYNAMIC_TP_MAX_MULT, ATR_TP_MULTIPLIER * 1.5)
            
            # V29 Phase 2: Dynamic OrderBook Stop-Loss
            # Check the L2 Cache to see if we can hide our STOP LOSS behind a massive wall
            sl_adjusted = False
            base_sl = entry - atr * ATR_SL_MULTIPLIER if side == "BUY" else entry + atr * ATR_SL_MULTIPLIER
            
            if provider and getattr(provider, 'depth_cache', None) and symbol in provider.depth_cache:
                depth = provider.depth_cache[symbol]
                
                if side == "BUY":
                    # Look for massive BID wall to place stop exactly 1 tick below it
                    bids = sorted(depth.get('bids', []), key=lambda x: x[0], reverse=True)
                    # We want walls below entry but above base_sl
                    valid_bids = [b for b in bids if base_sl < b[0] < entry]
                    if valid_bids:
                        # Find the largest limit wall within the ATR radius
                        largest_wall = max(valid_bids, key=lambda x: x[1])
                        wall_price, wall_size = largest_wall
                        
                        info = await self._get_symbol_info(symbol)
                        tick = info.get('tickSize', 0.01)
                        # Hide the SL one tick behind the wall
                        stop_loss = wall_price - tick
                        sl_adjusted = True
                        logger.info("   🛡️ ORDERBOOK SL: Tucked BUY Stop-Loss behind huge Bid Wall at %.4f (Size: %.2f). Base SL was %.4f", wall_price, wall_size, base_sl)
                        
                else:
                    # Look for massive ASK wall to place stop exactly 1 tick above it
                    asks = sorted(depth.get('asks', []), key=lambda x: x[0])
                    # We want walls above entry but below base_sl
                    valid_asks = [a for a in asks if entry < a[0] < base_sl]
                    if valid_asks:
                        largest_wall = max(valid_asks, key=lambda x: x[1])
                        wall_price, wall_size = largest_wall
                        
                        info = await self._get_symbol_info(symbol)
                        tick = info.get('tickSize', 0.01)
                        # Hide the SL one tick behind the wall
                        stop_loss = wall_price + tick
                        sl_adjusted = True
                        logger.info("   🛡️ ORDERBOOK SL: Tucked SHORT Stop-Loss behind huge Ask Wall at %.4f (Size: %.2f). Base SL was %.4f", wall_price, wall_size, base_sl)

            if not sl_adjusted:
                stop_loss = base_sl
                
            if side == "BUY":
                take_profit = entry + atr * tp_mult
            else:
                take_profit = entry - atr * tp_mult

        pos = {
            "side": side, "entry": entry, "size_usd": size_usd, "quantity": quantity,
            "stop_loss": stop_loss, "take_profit": take_profit,
            "trail_high": entry if side == "BUY" else None,
            "trail_low": entry if side == "SELL" else None,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }

        self.positions[symbol] = pos
        self.balance -= size_usd
        save_position(symbol, pos, self.db_path)

        result["action"] = "OPENED_LONG" if side == "BUY" else "OPENED_SHORT"
        result["size_usd"] = size_usd
        result["stop_loss"] = stop_loss
        result["take_profit"] = take_profit
        
        # V32: Slippage Analytics
        slippage = (entry - current_price) / current_price * 100.0 if side == "BUY" else (current_price - entry) / current_price * 100.0
        result["slippage_pct"] = slippage
        
        direction = "📈" if side == "BUY" else "📉"
        logger.info("%s API: %s %s @ %.4f (Slippage: %.3f%%) | SL=%.4f | balance=$%.2f",
                    direction, result["action"], symbol, entry, slippage, stop_loss or 0, self.balance)
        
        # V32: Place REAL exchange-side Stop-Loss as safety net
        if stop_loss and quantity > 0:
            close_side = "SELL" if side == "BUY" else "BUY"
            await self.place_exchange_sl(symbol, close_side, quantity, stop_loss)
        
        return result

    async def _close_position(self, symbol: str, current_price: float, result: Dict, reason: str) -> Dict:
        pos = self.positions[symbol]
        
        # V32: Cancel exchange-side SL before closing to prevent double execution
        await self._cancel_all_sl_orders(symbol)
        
        # API execution to close
        close_side = "SELL" if pos["side"] == "BUY" else "BUY"
        avg_price = await self.open_market_order(symbol, close_side, pos["quantity"])
        if avg_price < 0:
            result["action"], result["blocked"] = "API_CLOSING_ERROR", True
            return result
            
        exit_price = avg_price if avg_price > 0 else current_price

        # Calculate PnL locally
        pnl = self.simulate_pnl(entry=pos["entry"], exit_price=exit_price, size_usd=pos["size_usd"], side=pos["side"])
        self.balance += pos["size_usd"] + pnl

        trade_id = log_trade(
            side=close_side, price=exit_price, size_usd=pos["size_usd"],
            pnl=pnl, db_path=self.db_path, symbol=symbol,
            slippage=slippage
        )

        del self.positions[symbol]
        delete_position(symbol, self.db_path)

        result["action"] = "CLOSED_LONG" if close_side == "SELL" else "CLOSED_SHORT"
        result["pnl"] = pnl
        result["trade_id"] = trade_id
        result["reason"] = reason

        # V32: Slippage Analytics
        slippage = (current_price - exit_price) / current_price * 100.0 if close_side == "SELL" else (exit_price - current_price) / current_price * 100.0
        result["slippage_pct"] = slippage

        logger.info("💰 API: %s %s @ %.4f (Slippage: %.3f%%) | PnL=$%.4f | reason=%s | balance=$%.2f",
                    result["action"], symbol, exit_price, slippage, pnl, reason, self.balance)
        return result
