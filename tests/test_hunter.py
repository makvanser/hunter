"""
Hunter V12.1 — Test Suite
===========================
Covers RSI, ADX regime, contrarian signals, circuit breaker,
PnL USD, multi-asset portfolio, and config blacklist.
"""

import math
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import (
    compute_adx,
    compute_bollinger,
    compute_rsi,
    generate_signal,
    get_market_regime,
)
from config import BLACKLIST
from database import (
    get_consecutive_losses,
    init_db,
    is_on_cooldown,
    log_trade,
    reset_circuit_breaker,
    set_cooldown_until,
)
from execution import PaperTrader


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _make_trending_candles(n: int = 60):
    """Generate synthetic candles with a strong uptrend (high ADX)."""
    closes, highs, lows = [], [], []
    for i in range(n):
        c = 100 + i * 2.0  # monotonic rise
        closes.append(c)
        highs.append(c + 1.0)
        lows.append(c - 1.0)
    return highs, lows, closes


def _make_choppy_candles(n: int = 60):
    """Generate synthetic candles with no clear direction (low ADX)."""
    closes, highs, lows = [], [], []
    for i in range(n):
        c = 100 + math.sin(i) * 0.5  # oscillates within ±0.5
        closes.append(c)
        highs.append(c + 0.2)
        lows.append(c - 0.2)
    return highs, lows, closes


def _temp_db():
    """Return a path to a temporary SQLite database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


# ─────────────────────────────────────────────────────────────
# RSI Tests
# ─────────────────────────────────────────────────────────────
class TestRSI:
    def test_rsi_boundaries(self):
        closes = [45000 + i * 50 for i in range(30)]
        rsi = compute_rsi(closes)
        assert 0 <= rsi <= 100

    def test_rsi_strong_uptrend_is_high(self):
        closes = [100 + i * 10 for i in range(30)]
        rsi = compute_rsi(closes)
        assert rsi > 70, f"Expected overbought RSI, got {rsi:.2f}"

    def test_rsi_strong_downtrend_is_low(self):
        closes = [1000 - i * 10 for i in range(30)]
        rsi = compute_rsi(closes)
        assert rsi < 30, f"Expected oversold RSI, got {rsi:.2f}"


# ─────────────────────────────────────────────────────────────
# ADX / Regime Tests
# ─────────────────────────────────────────────────────────────
class TestADX:
    def test_adx_trending(self):
        highs, lows, closes = _make_trending_candles(60)
        adx = compute_adx(highs, lows, closes)
        regime = get_market_regime(adx)
        assert regime == "TRENDING", f"ADX={adx:.2f}, expected TRENDING"

    def test_adx_choppy(self):
        highs, lows, closes = _make_choppy_candles(60)
        adx = compute_adx(highs, lows, closes)
        regime = get_market_regime(adx)
        assert regime == "CHOPPY", f"ADX={adx:.2f}, expected CHOPPY"


# ─────────────────────────────────────────────────────────────
# Bollinger Tests
# ─────────────────────────────────────────────────────────────
class TestBollinger:
    def test_bollinger_band_order(self):
        closes = [100 + math.sin(i) * 5 for i in range(30)]
        upper, middle, lower = compute_bollinger(closes)
        assert upper > middle > lower


# ─────────────────────────────────────────────────────────────
# Signal Tests
# ─────────────────────────────────────────────────────────────
class TestSignal:
    def test_buy_signal(self):
        sig = generate_signal(rsi=25, ls_ratio=0.6, whale_net_vol=100, regime="TRENDING")
        assert sig == "BUY"

    def test_hold_when_choppy(self):
        sig = generate_signal(rsi=10, ls_ratio=0.3, whale_net_vol=9999, regime="CHOPPY")
        assert sig == "HOLD"

    def test_hold_when_rsi_not_oversold(self):
        sig = generate_signal(rsi=50, ls_ratio=0.6, whale_net_vol=100, regime="TRENDING")
        assert sig == "HOLD"

    def test_hold_when_ls_ratio_high(self):
        sig = generate_signal(rsi=25, ls_ratio=0.9, whale_net_vol=100, regime="TRENDING")
        assert sig == "HOLD"

    def test_hold_when_whale_vol_negative(self):
        sig = generate_signal(rsi=25, ls_ratio=0.6, whale_net_vol=-50, regime="TRENDING")
        assert sig == "HOLD"

    def test_sell_signal(self):
        sig = generate_signal(rsi=80, ls_ratio=1.2, whale_net_vol=0, regime="TRENDING")
        assert sig == "SELL"

    def test_sell_works_even_in_choppy(self):
        """Critical: overbought exit must fire regardless of regime."""
        sig = generate_signal(rsi=80, ls_ratio=1.0, whale_net_vol=0, regime="CHOPPY")
        assert sig == "SELL"


# ─────────────────────────────────────────────────────────────
# PnL Calculation Tests (V12 fix)
# ─────────────────────────────────────────────────────────────
class TestPnL:
    def test_long_profit_usd(self):
        """$100 position, entry 50000, exit 51000 → +$2.00 profit."""
        pnl = PaperTrader.simulate_pnl(
            entry=50_000, exit_price=51_000, size_usd=100, side="BUY"
        )
        assert abs(pnl - 2.0) < 0.01, f"Expected ~$2.00, got ${pnl:.4f}"

    def test_long_loss_usd(self):
        """$100 position, entry 50000, exit 49000 → -$2.00 loss."""
        pnl = PaperTrader.simulate_pnl(
            entry=50_000, exit_price=49_000, size_usd=100, side="BUY"
        )
        assert abs(pnl - (-2.0)) < 0.01, f"Expected ~-$2.00, got ${pnl:.4f}"

    def test_breakeven(self):
        """Same entry/exit → $0 PnL."""
        pnl = PaperTrader.simulate_pnl(
            entry=50_000, exit_price=50_000, size_usd=100, side="BUY"
        )
        assert pnl == 0.0


# ─────────────────────────────────────────────────────────────
# Circuit Breaker Tests
# ─────────────────────────────────────────────────────────────
class TestCircuitBreaker:
    def test_three_losses_activate_cooldown(self):
        db = _temp_db()
        init_db(db)
        for _ in range(3):
            log_trade("SELL", 100.0, 100.0, pnl=-5.0, db_path=db)
        assert is_on_cooldown(db)
        os.unlink(db)

    def test_win_resets_counter(self):
        db = _temp_db()
        init_db(db)
        log_trade("SELL", 100.0, 100.0, pnl=-5.0, db_path=db)
        log_trade("SELL", 100.0, 100.0, pnl=-5.0, db_path=db)
        assert get_consecutive_losses(db) == 2
        log_trade("SELL", 110.0, 100.0, pnl=10.0, db_path=db)  # WIN
        assert get_consecutive_losses(db) == 0
        os.unlink(db)

    def test_cooldown_blocks_trade(self):
        db = _temp_db()
        init_db(db)
        set_cooldown_until(
            datetime.now(timezone.utc) + timedelta(hours=1), db_path=db
        )
        trader = PaperTrader(db_path=db)
        result = trader.execute_trade("BUY", 50000.0, "BTCUSDT")
        assert result["blocked"] is True
        assert result["action"] == "BLOCKED_BY_CIRCUIT_BREAKER"
        os.unlink(db)

    def test_expired_cooldown_allows_trade(self):
        db = _temp_db()
        init_db(db)
        reset_circuit_breaker(db)
        set_cooldown_until(
            datetime.now(timezone.utc) - timedelta(hours=1), db_path=db
        )
        trader = PaperTrader(db_path=db)
        result = trader.execute_trade("BUY", 50000.0, "BTCUSDT")
        assert result["blocked"] is False
        os.unlink(db)


# ─────────────────────────────────────────────────────────────
# Multi-Asset Portfolio Tests (V12.1)
# ─────────────────────────────────────────────────────────────
class TestMultiAsset:
    def test_open_two_symbols_simultaneously(self):
        """BUY BTC and ETH at the same time → two open positions."""
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        r1 = trader.execute_trade("BUY", 50000.0, "BTCUSDT")
        r2 = trader.execute_trade("BUY", 3200.0, "ETHUSDT")
        assert r1["action"] == "OPENED_LONG"
        assert r2["action"] == "OPENED_LONG"
        assert trader.open_positions_count() == 2
        assert trader.has_position("BTCUSDT")
        assert trader.has_position("ETHUSDT")
        os.unlink(db)

    def test_close_one_keeps_other(self):
        """SELL BTC while ETH stays open."""
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        trader.execute_trade("BUY", 50000.0, "BTCUSDT")
        trader.execute_trade("BUY", 3200.0, "ETHUSDT")
        r_sell = trader.execute_trade("SELL", 51000.0, "BTCUSDT")
        assert r_sell["action"] == "CLOSED_LONG"
        assert not trader.has_position("BTCUSDT")
        assert trader.has_position("ETHUSDT")
        assert trader.open_positions_count() == 1
        os.unlink(db)

    def test_buy_duplicate_symbol_is_noop(self):
        """BUY BTC twice → second call is NO_ACTION."""
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        trader.execute_trade("BUY", 50000.0, "BTCUSDT")
        r2 = trader.execute_trade("BUY", 50500.0, "BTCUSDT")
        assert r2["action"] == "NO_ACTION"
        assert trader.open_positions_count() == 1
        os.unlink(db)

    def test_circuit_breaker_blocks_all_symbols(self):
        """3 losses on BTC → ETH is also blocked (global breaker)."""
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        # Open and close BTC 3 times at a loss
        for _ in range(3):
            trader.execute_trade("BUY", 50000.0, "BTCUSDT")
            trader.execute_trade("SELL", 49000.0, "BTCUSDT")  # loss
        # Now try to buy ETH → should be blocked
        r = trader.execute_trade("BUY", 3200.0, "ETHUSDT")
        assert r["blocked"] is True
        assert r["action"] == "BLOCKED_BY_CIRCUIT_BREAKER"
        os.unlink(db)


# ─────────────────────────────────────────────────────────────
# Config / Blacklist Tests
# ─────────────────────────────────────────────────────────────
class TestConfig:
    def test_blacklist_contains_stablecoins(self):
        assert "USDCUSDT" in BLACKLIST
        assert "USDPUSDT" in BLACKLIST

    def test_btcusdt_not_blacklisted(self):
        assert "BTCUSDT" not in BLACKLIST
