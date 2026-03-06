"""
Hunter V16 — Execution Module
=============================================================
Paper-trading engine with:
  - Multi-asset portfolio (per-symbol positions)
  - Persistent positions (survives restarts via SQLite)   [V16]
  - Long and Short positions                              [V15]
  - ATR-based Stop-Loss / Take-Profit                    [V14]
  - Trailing Stop-Loss                                   [V15]
  - Global circuit breaker                               [V12]
  - Trading fees + slippage in P&L                       [V16]
  - MAX_OPEN_POSITIONS + MAX_EXPOSURE_USD caps           [V16]
  - Kelly Criterion position sizing                      [V16]
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from config import (
    ATR_SL_MULTIPLIER,
    ATR_TP_MULTIPLIER,
    INITIAL_BALANCE_USD,
    KELLY_ENABLED,
    KELLY_FRACTION,
    KELLY_MAX_PCT,
    KELLY_MIN_TRADES,
    MAX_DRAWDOWN_PCT,
    MAX_RISK_PER_TRADE_PCT,
    MAX_EXPOSURE_USD,
    MAX_OPEN_POSITIONS,
    SHORT_ENABLED,
    SLIPPAGE,
    TAKER_FEE,
    TRADE_SIZE_USD,
    TRAILING_SL_ACTIVATION_PCT,
    TRAILING_SL_ATR_MULT,
    TRAILING_SL_ENABLED,
)
from database import (
    delete_position,
    get_consecutive_losses,
    get_cooldown_until,
    get_trade_stats,
    init_db,
    is_on_cooldown,
    load_positions,
    log_trade,
    save_position,
)

logger = logging.getLogger("hunter.execution")


class PaperTrader:
    """
    Simulated order executor with full V16 feature set.

    Position structure per symbol:
    {
      "side":        "BUY" | "SELL",
      "entry":       float,
      "size_usd":    float,
      "stop_loss":   float | None,
      "take_profit": float | None,
      "trail_high":  float | None,   # Highest price seen (longs)
      "trail_low":   float | None,   # Lowest price seen (shorts)
      "opened_at":   str,            # ISO timestamp
    }
    """

    def __init__(self, db_path: str | None = None):
        from config import DB_PATH

        self.db_path = db_path or DB_PATH
        self.balance = INITIAL_BALANCE_USD

        init_db(self.db_path)

        # V16: Restore positions from DB (survives restarts)
        self.positions: Dict[str, Dict] = load_positions(self.db_path)
        if self.positions:
            logger.info(
                "PaperTrader V16 restored %d open position(s): %s",
                len(self.positions),
                list(self.positions.keys()),
            )

        logger.info(
            "PaperTrader V16 initialised | balance=$%.2f | db=%s",
            self.balance,
            self.db_path,
        )

    # ── Safety Gate (GLOBAL) ──────────────────────────────────
    def check_circuit_breaker(self) -> bool:
        """Return True if trading is ALLOWED, False if blocked."""
        # V22: Global drawdown guard
        # Use actual exchange balance as reference for LiveTrader (testnet/mainnet)
        ref_balance = getattr(self, '_initial_balance', INITIAL_BALANCE_USD)
        drawdown_pct = (ref_balance - self.balance) / ref_balance * 100 if ref_balance > 0 else 0
        if drawdown_pct >= MAX_DRAWDOWN_PCT:
            logger.warning(
                "🚨 GLOBAL DRAWDOWN BREAKER — balance $%.2f is %.1f%% below initial (limit %.1f%%)",
                self.balance, drawdown_pct, MAX_DRAWDOWN_PCT,
            )
            return False

        if is_on_cooldown(self.db_path):
            cd = get_cooldown_until(self.db_path)
            logger.warning(
                "⛔ CIRCUIT BREAKER ACTIVE — cooldown until %s",
                cd.isoformat() if cd else "unknown",
            )
            return False

        losses = get_consecutive_losses(self.db_path)
        if losses >= 3:
            logger.warning(
                "⛔ CIRCUIT BREAKER — %d consecutive losses (limit 3)", losses
            )
            return False

        return True

    # ── Portfolio Info ────────────────────────────────────────
    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def get_position(self, symbol: str) -> Optional[Dict]:
        return self.positions.get(symbol)

    def open_positions_count(self) -> int:
        return len(self.positions)

    # ── Kelly Criterion Position Sizing  (V16) ────────────────
    def _kelly_position_size(self) -> float:
        """
        Compute position size using half-Kelly Criterion.

        Falls back to fixed TRADE_SIZE_USD when:
          - KELLY_ENABLED is False
          - Fewer than KELLY_MIN_TRADES in history
          - avg_win or avg_loss is zero
        """
        if not KELLY_ENABLED:
            return min(TRADE_SIZE_USD, self.balance)

        stats = get_trade_stats(self.db_path)
        if stats["n_trades"] < KELLY_MIN_TRADES:
            logger.debug(
                "Kelly: only %d trades (min %d) → using fixed size",
                stats["n_trades"],
                KELLY_MIN_TRADES,
            )
            return min(TRADE_SIZE_USD, self.balance)

        wr = stats["win_rate"]
        avg_win = stats["avg_win"]
        avg_loss = stats["avg_loss"]

        if avg_win <= 0 or avg_loss <= 0:
            return min(TRADE_SIZE_USD, self.balance)
            
        win_loss_ratio = avg_win / avg_loss

        # V25 Modified Kelly: Win Rate - (Loss Rate / Win-Loss Ratio)
        loss_rate = 1.0 - wr
        basic_kelly = wr - (loss_rate / win_loss_ratio)
        basic_kelly = max(0.01, min(basic_kelly, 1.0))  # clamp: never <1% or >100%
        
        # Actual Position = Basic Kelly * 0.25 (1/4 Kelly conservative model)
        actual_position_pct = basic_kelly * KELLY_FRACTION
        
        # Hard Cap: Max 10% risk
        max_allowed_pct = MAX_RISK_PER_TRADE_PCT / 100.0
        final_pct = min(actual_position_pct, max_allowed_pct)

        size = self.balance * final_pct
        size = max(size, 10.0)  # minimum $10

        logger.info(
            "Kelly sizing: wr=%.1f%% avg_win=$%.2f avg_loss=$%.2f → "
            "basic_kelly=%.3f quarter=%.3f cap=%.1f%% → size=$%.2f",
            wr * 100,
            avg_win,
            avg_loss,
            basic_kelly,
            actual_position_pct,
            MAX_RISK_PER_TRADE_PCT,
            size,
        )
        return round(size, 2)

    # ── Trailing SL Update (V15) ──────────────────────────────
    def _update_trailing_sl(self, symbol: str, current_price: float, atr: float) -> None:
        """Update trailing stop-loss for an open position."""
        if not TRAILING_SL_ENABLED or atr <= 0:
            return

        pos = self.positions.get(symbol)
        if pos is None:
            return

        entry = pos["entry"]
        side = pos["side"]
        modified = False

        if side == "BUY":
            trail_high = pos.get("trail_high") or entry
            if current_price > trail_high:
                trail_high = current_price
                pos["trail_high"] = trail_high
                modified = True

            profit_pct = (trail_high - entry) / entry * 100
            if profit_pct < TRAILING_SL_ACTIVATION_PCT:
                return

            new_sl = trail_high - atr * TRAILING_SL_ATR_MULT
            current_sl = pos.get("stop_loss") or 0.0
            if new_sl > current_sl:
                pos["stop_loss"] = new_sl
                modified = True
                logger.debug(
                    "📈 Trailing SL updated LONG %s: SL=%.4f (trail_high=%.4f)",
                    symbol, new_sl, trail_high,
                )

        elif side == "SELL":
            trail_low = pos.get("trail_low") or entry
            if current_price < trail_low:
                trail_low = current_price
                pos["trail_low"] = trail_low
                modified = True

            profit_pct = (entry - trail_low) / entry * 100
            if profit_pct < TRAILING_SL_ACTIVATION_PCT:
                return

            new_sl = trail_low + atr * TRAILING_SL_ATR_MULT
            current_sl = pos.get("stop_loss") or float("inf")
            if new_sl < current_sl:
                pos["stop_loss"] = new_sl
                modified = True
                logger.debug(
                    "📉 Trailing SL updated SHORT %s: SL=%.4f (trail_low=%.4f)",
                    symbol, new_sl, trail_low,
                )

        # V16: persist trailing SL changes immediately
        if modified:
            save_position(symbol, pos, self.db_path)

    # ── SL/TP Check (V14/V15) ────────────────────────────────
    def check_sl_tp(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if current price has hit stop-loss or take-profit.
        Returns "SL_HIT", "TP_HIT", or None.
        """
        pos = self.positions.get(symbol)
        if pos is None:
            return None

        sl   = pos.get("stop_loss")
        tp   = pos.get("take_profit")
        side = pos["side"]

        if side == "BUY":
            if sl is not None and current_price <= sl:
                return "SL_HIT"
            if tp is not None and current_price >= tp:
                return "TP_HIT"

        elif side == "SELL":
            if sl is not None and current_price >= sl:
                return "SL_HIT"
            if tp is not None and current_price <= tp:
                return "TP_HIT"

        return None

    # ── Core Execution ────────────────────────────────────────
    def execute_trade(
        self,
        signal: str,
        current_price: float,
        symbol: str,
        atr: float = 0.0,
    ) -> Dict:
        """
        Process a signal for a specific symbol.

        Signals: BUY, SELL, SHORT, COVER, HOLD

        Flow:
        1. Circuit breaker check.
        2. Trailing SL update (V15).
        3. SL/TP auto-close check.
        4. HOLD → skip.
        5. BUY/SHORT → open (with caps, Kelly sizing).
        6. SELL/COVER → close (with fees/slippage).
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

        # Gate 1: Global circuit breaker
        if not self.check_circuit_breaker():
            result["action"]  = "BLOCKED_BY_CIRCUIT_BREAKER"
            result["blocked"] = True
            return result

        # Gate 2: Trailing SL update
        if atr > 0 and symbol in self.positions:
            self._update_trailing_sl(symbol, current_price, atr)

        # Gate 3: SL/TP auto-close
        sl_tp_trigger = self.check_sl_tp(symbol, current_price)
        if sl_tp_trigger is not None:
            return self._close_position(symbol, current_price, result, sl_tp_trigger)

        # Gate 4: No-op
        if signal == "HOLD":
            result["action"] = "HOLD"
            return result

        # ── OPEN LONG ─────────────────────────────────────────
        if signal == "BUY" and symbol not in self.positions:
            return self._open_position(symbol, current_price, "BUY", atr, result)

        # ── OPEN SHORT (V15) ──────────────────────────────────
        if signal == "SHORT" and symbol not in self.positions:
            if not SHORT_ENABLED:
                result["action"] = "SHORTS_DISABLED"
                return result
            return self._open_position(symbol, current_price, "SELL", atr, result)

        # ── DCA / AVERAGING (V19) ─────────────────────────────
        if signal in ("DCA_BUY", "DCA_SHORT") and symbol in self.positions:
            return self._dca_position(symbol, current_price, atr, result)

        # ── CLOSE LONG ────────────────────────────────────────
        if signal == "SELL" and symbol in self.positions:
            if self.positions[symbol]["side"] == "BUY":
                return self._close_position(symbol, current_price, result, "SIGNAL_SELL")

        # ── CLOSE SHORT (V15) ─────────────────────────────────
        if signal == "COVER" and symbol in self.positions:
            if self.positions[symbol]["side"] == "SELL":
                return self._close_position(symbol, current_price, result, "SIGNAL_COVER")

        result["action"] = "NO_ACTION"
        return result

    # ── Open Position Helper (V15/V16) ───────────────────────
    def _open_position(
        self,
        symbol: str,
        current_price: float,
        side: str,
        atr: float,
        result: Dict,
    ) -> Dict:
        """Open a LONG (BUY) or SHORT (SELL) position."""

        # V16: Max open positions cap
        if len(self.positions) >= MAX_OPEN_POSITIONS:
            result["action"] = "MAX_POSITIONS_REACHED"
            logger.warning(
                "⚠️  Cannot open %s %s — already at MAX_OPEN_POSITIONS=%d",
                side, symbol, MAX_OPEN_POSITIONS,
            )
            return result

        # V16: Max total exposure cap
        total_exposure = sum(p["size_usd"] for p in self.positions.values())

        # V16: Kelly Criterion or fixed size
        size_usd = self._kelly_position_size()
        size_usd = min(size_usd, self.balance)
        size_usd = min(size_usd, MAX_EXPOSURE_USD - total_exposure)

        if size_usd <= 0:
            result["action"] = "INSUFFICIENT_BALANCE"
            return result

        # ATR-based SL/TP
        stop_loss   = None
        take_profit = None
        if atr > 0:
            if side == "BUY":
                stop_loss   = current_price - atr * ATR_SL_MULTIPLIER
                take_profit = current_price + atr * ATR_TP_MULTIPLIER
            else:  # SHORT
                stop_loss   = current_price + atr * ATR_SL_MULTIPLIER
                take_profit = current_price - atr * ATR_TP_MULTIPLIER

        opened_at = datetime.now(timezone.utc).isoformat()
        pos = {
            "side":        side,
            "entry":       current_price,
            "size_usd":    size_usd,
            "stop_loss":   stop_loss,
            "take_profit": take_profit,
            "trail_high":  current_price if side == "BUY"  else None,
            "trail_low":   current_price if side == "SELL" else None,
            "opened_at":   opened_at,
        }

        self.positions[symbol] = pos
        self.balance -= size_usd

        # V16: Persist position to DB
        save_position(symbol, pos, self.db_path)

        action    = "OPENED_LONG" if side == "BUY" else "OPENED_SHORT"
        direction = "📈" if side == "BUY" else "📉"
        result["action"]      = action
        result["size_usd"]    = size_usd
        result["stop_loss"]   = stop_loss
        result["take_profit"] = take_profit

        logger.info(
            "%s %s %s @ %.4f | size=$%.2f | SL=%.4f | TP=%.4f | balance=$%.2f",
            direction, action, symbol, current_price, size_usd,
            stop_loss or 0.0, take_profit or 0.0, self.balance,
        )
        return result

    # ── DCA Position Helper (V19) ──────────────────────────────
    def _dca_position(
        self,
        symbol: str,
        current_price: float,
        atr: float,
        result: Dict,
    ) -> Dict:
        """Average down (or up) an existing position."""
        pos = self.positions[symbol]
        side = pos["side"]

        # 1. Size constraint
        total_exposure = sum(p["size_usd"] for p in self.positions.values())
        size_usd = self._kelly_position_size()
        size_usd = min(size_usd, self.balance)
        size_usd = min(size_usd, MAX_EXPOSURE_USD - total_exposure)

        if size_usd <= 0:
            result["action"] = "INSUFFICIENT_BALANCE_FOR_DCA"
            return result

        # 2. Update Entry Price (Weighted Average)
        old_size = pos["size_usd"]
        old_entry = pos["entry"]
        new_size = old_size + size_usd
        new_entry = ((old_entry * old_size) + (current_price * size_usd)) / new_size

        # 3. Apply changes
        self.balance -= size_usd
        pos["size_usd"] = new_size
        pos["entry"] = new_entry
        pos["dca_count"] = pos.get("dca_count", 0) + 1

        # Calculate new dynamic SL/TP based on new average entry
        if atr > 0:
            if side == "BUY":
                pos["stop_loss"]   = new_entry - atr * ATR_SL_MULTIPLIER
                pos["take_profit"] = new_entry + atr * ATR_TP_MULTIPLIER
                # Reset trailing high
                pos["trail_high"]  = current_price
            else:  # SHORT
                pos["stop_loss"]   = new_entry + atr * ATR_SL_MULTIPLIER
                pos["take_profit"] = new_entry - atr * ATR_TP_MULTIPLIER
                # Reset trailing low
                pos["trail_low"]   = current_price

        save_position(symbol, pos, self.db_path)

        result["action"]     = "DCA_" + side
        result["size_usd"]   = size_usd
        result["new_entry"]  = new_entry
        result["dca_count"]  = pos["dca_count"]

        direction = "📈" if side == "BUY" else "📉"
        logger.info(
            "%s DCA %s %s @ %.4f (added $%.2f) | new_entry=%.4f | total_size=$%.2f | balance=$%.2f",
            direction, side, symbol, current_price, size_usd, new_entry, new_size, self.balance,
        )
        return result

    # ── Close Position Helper (V14/V15/V16) ──────────────────
    def _close_position(
        self,
        symbol: str,
        current_price: float,
        result: Dict,
        reason: str,
    ) -> Dict:
        """Close a position. Calculates PnL with fees/slippage (V16)."""
        pos = self.positions[symbol]
        pnl = self.simulate_pnl(
            entry=pos["entry"],
            exit_price=current_price,
            size_usd=pos["size_usd"],
            side=pos["side"],
        )
        self.balance += pos["size_usd"] + pnl

        trade_id = log_trade(
            side="SELL" if pos["side"] == "BUY" else "BUY",
            price=current_price,
            size_usd=pos["size_usd"],
            pnl=pnl,
            db_path=self.db_path,
            symbol=symbol,
        )

        # V16: Remove from persistent positions
        delete_position(symbol, self.db_path)

        action_map = {
            "SL_HIT":       "CLOSED_SL",
            "TP_HIT":       "CLOSED_TP",
            "SIGNAL_SELL":  "CLOSED_LONG",
            "SIGNAL_COVER": "CLOSED_SHORT",
        }
        action = action_map.get(reason, "CLOSED_LONG")

        logger.info(
            "💰 %s %s @ %.4f | PnL=$%.4f | reason=%s | trade_id=%d | balance=$%.2f",
            action, symbol, current_price, pnl, reason, trade_id, self.balance,
        )

        result["action"]       = action
        result["pnl"]          = pnl
        result["trade_id"]     = trade_id
        result["close_reason"] = reason

        del self.positions[symbol]
        return result

    # ── PnL Helper (V16: includes fees + slippage) ───────────
    @staticmethod
    def simulate_pnl(
        entry: float,
        exit_price: float,
        size_usd: float,
        side: str,
    ) -> float:
        """
        Calculate absolute USD profit/loss.

        V16: Deducts trading fees and slippage on both open and close legs.
          LONG:  pnl = (exit - entry) / entry * size - round_trip_costs
          SHORT: pnl = (entry - exit) / entry * size - round_trip_costs
        """
        if side == "BUY":
            gross = (exit_price - entry) / entry * size_usd
        elif side == "SELL":
            gross = (entry - exit_price) / entry * size_usd
        else:
            return 0.0

        # V16: Round-trip costs = (fee + slippage) × 2 × position size
        costs = (TAKER_FEE + SLIPPAGE) * 2 * size_usd
        return gross - costs
