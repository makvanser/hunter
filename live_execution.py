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
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from config import (
    API_KEY, API_SECRET, BASE_URL, 
    MAX_EXPOSURE_USD, TRADE_SIZE_USD, MAX_OPEN_POSITIONS, LEVERAGE,
    ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER, LIMIT_ORDER_TIMEOUT_SEC,
    MAKER_GRID_ENABLED, GRID_ORDERS_COUNT, GRID_SPREAD_PCT,
    DYNAMIC_TP_ENABLED, DYNAMIC_TP_MAX_MULT,
    TRAILING_SL_ENABLED, TRAILING_SL_ACTIVATION_PCT, TRAILING_SL_ATR_MULT,
    USE_TESTNET, USE_DYNAMIC_SIZING, RISK_PER_TRADE_PCT
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

    async def _api_post(self, endpoint: str, payload: Dict[str, Any]) -> Dict:
        """Internal helper for signed POST requests."""
        payload['timestamp'] = int(time.time() * 1000)
        payload['signature'] = self._sign_payload(payload)
        url = f"{BASE_URL}{endpoint}"
        
        headers = {"X-MBX-APIKEY": API_KEY, "Content-Type": "application/x-www-form-urlencoded"}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, data=payload) as response:
                    data = await response.json()
                    if response.status != 200:
                        logger.error("❌ Binance API Error [%s]: %s", response.status, data.get('msg', data))
                        return {"error": data}
                    return data
            except Exception as e:
                logger.error("❌ Async HTTP Error on %s: %s", endpoint, e)
                return {"error": str(e)}

    async def _api_get(self, endpoint: str, params: Dict[str, Any]) -> Dict:
        """Internal helper for signed GET requests."""
        params['timestamp'] = int(time.time() * 1000)
        params['signature'] = self._sign_payload(params)
        url = f"{BASE_URL}{endpoint}"
        
        headers = {"X-MBX-APIKEY": API_KEY}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers, params=params) as response:
                    data = await response.json()
                    if response.status != 200:
                        logger.error("❌ Binance API Error [%s]: %s", response.status, data.get('msg', data))
                        return {"error": data}
                    return data
            except Exception as e:
                logger.error("❌ Async HTTP Error on GET %s: %s", endpoint, e)
                return {"error": str(e)}

    async def _api_delete(self, endpoint: str, params: Dict[str, Any]) -> Dict:
        """Internal helper for signed DELETE requests."""
        params['timestamp'] = int(time.time() * 1000)
        params['signature'] = self._sign_payload(params)
        url = f"{BASE_URL}{endpoint}"
        
        headers = {"X-MBX-APIKEY": API_KEY, "Content-Type": "application/x-www-form-urlencoded"}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.delete(url, headers=headers, params=params) as response:
                    data = await response.json()
                    if response.status != 200:
                        logger.error("❌ Binance API Error [%s]: %s", response.status, data.get('msg', data))
                        return {"error": data}
                    return data
            except Exception as e:
                logger.error("❌ Async HTTP Error on DELETE %s: %s", endpoint, e)
                return {"error": str(e)}

    async def sync_balance(self) -> float:
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
        """Place a market order. Return executed price."""
        payload = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": str(quantity)}
        res = await self._api_post("/fapi/v1/order", payload)
        if "error" in res:
            logger.error("💥 Failed to open %s market order: %s", side, res)
            return -1.0
            
        order_id = res.get("orderId")
        avg_price = float(res.get('avgPrice', 0) or 0)
        
        # Binance testnet often returns avgPrice = 0.0 for immediate REST responses. Polling actual fill state:
        if avg_price == 0.0 and order_id:
            await asyncio.sleep(0.5)
            check = await self._api_get("/fapi/v1/order", {"symbol": symbol, "orderId": order_id})
            avg_price = float(check.get('avgPrice', 0) or 0)
            
        logger.info("🚀 LIVE ORDER EXECUTED: %s %s size=%s", side, symbol, quantity)
        return avg_price

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
            res = await self._api_post("/fapi/v1/order", payload)
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
            
            # Poll all active orders
            active_ids = list(order_ids) # copy
            for oid in active_ids:
                check = await self._api_get("/fapi/v1/order", {"symbol": symbol, "orderId": oid})
                status = check.get("status")
                
                if status == "FILLED":
                    exec_qty = float(check.get('executedQty', 0))
                    avg_px = float(check.get('avgPrice', 0))
                    filled_qty += exec_qty
                    total_spent += exec_qty * avg_px
                    order_ids.remove(oid) # Done tracking this chunk
                    logger.info("🚀 GRID CHUNK EXECUTED: %s %s size=%s @ %.4f", side, symbol, exec_qty, avg_px)
                    
            if not order_ids:
                break # All filled!
                
        # Cancel remaining
        for oid in order_ids:
            await self._api_delete("/fapi/v1/order", {"symbol": symbol, "orderId": oid})
            
        if filled_qty > 0:
            final_avg_price = total_spent / filled_qty
            logger.info("✅ GRID COMPLETE: Executed total %.3f at Avg Price %.4f", filled_qty, final_avg_price)
            # Override quantity parameter inside the caller requires changing the return signature, 
            # but since V24 only uses this entry price to calculate SL/TP correctly:
            return final_avg_price
            
        logger.warning("⚠️ Limit grid timeout (%ds) with no fills.", timeout)
        return -1.0

    # ── ASYNC OVERRIDES ────────────────────────────────────────────────────────

    async def execute_trade(self, signal: str, current_price: float, symbol: str, atr: float = 0.0) -> Dict:
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

        # Gate 2: Trailing SL update (Synchronous)
        if atr > 0 and symbol in self.positions:
            self._update_trailing_sl(symbol, current_price, atr)

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
            return await self._open_position(symbol, current_price, "BUY", atr, result)

        # ── OPEN SHORT
        if signal == "SHORT" and symbol not in self.positions:
            if not SHORT_ENABLED:
                result["action"] = "SHORTS_DISABLED"
                return result
            return await self._open_position(symbol, current_price, "SELL", atr, result)

        # ── CLOSE LONG
        if signal == "SELL" and symbol in self.positions:
            if self.positions[symbol]["side"] == "BUY":
                return await self._close_position(symbol, current_price, result, "SIGNAL_SELL")

        # ── CLOSE SHORT
        if signal == "COVER" and symbol in self.positions:
            if self.positions[symbol]["side"] == "SELL":
                return await self._close_position(symbol, current_price, result, "SIGNAL_COVER")

        return result

    async def _open_position(self, symbol: str, current_price: float, side: str, atr: float, result: Dict) -> Dict:
        if len(self.positions) >= MAX_OPEN_POSITIONS:
            result["action"] = "MAX_POSITIONS_REACHED"
            return result

        total_exposure = sum(p["size_usd"] for p in self.positions.values())
        
        if USE_DYNAMIC_SIZING:
            # Dynamic compounding size: Calculate base risk per trade from actual balance and apply leverage
            size_usd = self.balance * (RISK_PER_TRADE_PCT / 100.0) * LEVERAGE
        else:
            size_usd = self._kelly_position_size()
            
        size_usd = min(size_usd, self.balance * LEVERAGE, MAX_EXPOSURE_USD - total_exposure)

        if size_usd < 100:
            size_usd = 3000.0  # Binance testnet minimal limit

        # Submit to Binance API
        quantity = round(size_usd / current_price, 3)
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
                # If market is already explosive, allow larger TP to ride the wave.
                # In V24 Phase 2 we dynamically scale TP up to 10x ATR if conditions are extreme.
                # Currently using a static wide dynamic modifier, to be refined with 'market_state' in Phase 4.
                tp_mult = min(DYNAMIC_TP_MAX_MULT, ATR_TP_MULTIPLIER * 1.5)
            
            if side == "BUY":
                stop_loss = entry - atr * ATR_SL_MULTIPLIER
                take_profit = entry + atr * tp_mult
            else:
                stop_loss = entry + atr * ATR_SL_MULTIPLIER
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

        direction = "📈" if side == "BUY" else "📉"
        logger.info("%s API: %s %s @ %.4f | SL=%.4f | TP=%.4f | balance=$%.2f",
                    direction, result["action"], symbol, entry, stop_loss or 0, take_profit or 0, self.balance)
        return result

    async def _close_position(self, symbol: str, current_price: float, result: Dict, reason: str) -> Dict:
        pos = self.positions[symbol]
        
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
            pnl=pnl, db_path=self.db_path, symbol=symbol
        )

        del self.positions[symbol]
        delete_position(symbol, self.db_path)

        result["action"] = "CLOSED_LONG" if close_side == "SELL" else "CLOSED_SHORT"
        result["pnl"] = pnl
        result["trade_id"] = trade_id
        result["reason"] = reason

        logger.info("💰 API: %s %s @ %.4f | PnL=$%.4f | reason=%s | balance=$%.2f",
                    result["action"], symbol, exit_price, pnl, reason, self.balance)
        return result
