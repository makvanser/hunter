"""
Hunter V12.1 — Execution Module (Multi-Asset)
================================================
Paper-trading engine with per-symbol portfolio tracking
and a GLOBAL circuit breaker (3 losses on ANY symbol → stop all).
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from config import INITIAL_BALANCE_USD, TRADE_SIZE_USD
from database import (
    get_consecutive_losses,
    is_on_cooldown,
    log_trade,
    init_db,
    get_cooldown_until,
)

logger = logging.getLogger("hunter.execution")


class PaperTrader:
    """Simulated order executor with multi-asset portfolio and circuit-breaker."""

    def __init__(self, db_path: str | None = None):
        from config import DB_PATH

        self.db_path = db_path or DB_PATH
        self.balance = INITIAL_BALANCE_USD

        # ── Multi-asset position tracker ──
        # { "BTCUSDT": {"side": "BUY", "entry": 50000.0, "size_usd": 100.0},
        #   "ETHUSDT": {"side": "BUY", "entry": 3200.0,  "size_usd": 100.0}, ... }
        self.positions: Dict[str, Dict] = {}

        init_db(self.db_path)
        logger.info(
            "PaperTrader initialised | balance=$%.2f | db=%s",
            self.balance,
            self.db_path,
        )

    # ── Safety Gate (GLOBAL) ─────────────────────────────────
    def check_circuit_breaker(self) -> bool:
        """
        Return True if trading is ALLOWED, False if blocked.
        This is GLOBAL — 3 losses on ANY symbol stops EVERYTHING.
        """
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
                "⛔ CIRCUIT BREAKER — %d consecutive losses (limit 3)",
                losses,
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

    # ── Core Execution ───────────────────────────────────────
    def execute_trade(self, signal: str, current_price: float, symbol: str) -> Dict:
        """
        Process a signal for a specific symbol.

        Flow:
        1. Check circuit breaker → refuse if tripped.
        2. If signal is HOLD → skip.
        3. If BUY and no position for this symbol → open long.
        4. If SELL and holding long on this symbol → close and realise PnL.
        """
        result: Dict = {
            "action": "NONE",
            "signal": signal,
            "symbol": symbol,
            "price": current_price,
            "pnl": 0.0,
            "blocked": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Gate 1: Global circuit breaker
        if not self.check_circuit_breaker():
            result["action"] = "BLOCKED_BY_CIRCUIT_BREAKER"
            result["blocked"] = True
            return result

        # Gate 2: No-op signals
        if signal == "HOLD":
            result["action"] = "HOLD"
            return result

        # ── OPEN a position on this symbol ──
        if signal == "BUY" and symbol not in self.positions:
            size_usd = min(TRADE_SIZE_USD, self.balance)
            if size_usd <= 0:
                result["action"] = "INSUFFICIENT_BALANCE"
                return result

            self.positions[symbol] = {
                "side": "BUY",
                "entry": current_price,
                "size_usd": size_usd,
            }
            self.balance -= size_usd
            result["action"] = "OPENED_LONG"
            result["size_usd"] = size_usd
            logger.info(
                "📈 OPENED LONG %s @ %.4f | size=$%.2f | balance=$%.2f | open_positions=%d",
                symbol, current_price, size_usd, self.balance, len(self.positions),
            )
            return result

        # ── CLOSE a position on this symbol ──
        if signal == "SELL" and symbol in self.positions:
            pos = self.positions[symbol]
            pnl = self.simulate_pnl(
                entry=pos["entry"],
                exit_price=current_price,
                size_usd=pos["size_usd"],
                side=pos["side"],
            )
            self.balance += pos["size_usd"] + pnl
            trade_id = log_trade(
                side="SELL",
                price=current_price,
                size_usd=pos["size_usd"],
                pnl=pnl,
                db_path=self.db_path,
            )
            logger.info(
                "📉 CLOSED LONG %s @ %.4f | PnL=$%.4f | trade_id=%d | balance=$%.2f",
                symbol, current_price, pnl, trade_id, self.balance,
            )
            result["action"] = "CLOSED_LONG"
            result["pnl"] = pnl
            result["trade_id"] = trade_id

            # Remove from portfolio
            del self.positions[symbol]
            return result

        # Signal doesn't match current state (e.g. BUY while already long on this symbol)
        result["action"] = "NO_ACTION"
        return result

    # ── PnL Helper ───────────────────────────────────────────
    @staticmethod
    def simulate_pnl(
        entry: float,
        exit_price: float,
        size_usd: float,
        side: str,
    ) -> float:
        """
        Calculate absolute USD profit/loss for a paper trade.

        LONG:  pnl_usd = (exit - entry) / entry * size_usd
        """
        if side == "BUY":
            return (exit_price - entry) / entry * size_usd
        return 0.0
