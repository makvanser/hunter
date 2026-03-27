"""
Hunter V16 — Database Module
==============================
SQLite-backed trade journal + circuit-breaker state management.
Tracks every trade outcome and enforces the 3-consecutive-loss cooldown.

V14: Added `symbol` column to trades table for per-instrument analytics.
V16: Added `positions` table for persistence across restarts.
     Added get_trade_stats() for Kelly Criterion position sizing.
"""

import sqlite3
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import redis

# Global synchronous Redis Pool for instantaneous in-memory state tracking
# V25: Eliminating SQLite I/O blocks for circuit breakers
try:
    redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
    redis_client.ping() # Quick check
except Exception:
    redis_client = None # Fallback for local tests without redis running

logger = logging.getLogger("hunter.database")


from config import DB_PATH, MAX_CONSECUTIVE_LOSSES, COOLDOWN_HOURS


# ─────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────
def init_db(db_path: str = DB_PATH) -> None:
    """Create tables if they don't exist and seed default state rows."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            symbol      TEXT    NOT NULL DEFAULT 'UNKNOWN',
            side        TEXT    NOT NULL,
            price       REAL    NOT NULL,
            size_usd    REAL    NOT NULL,
            pnl         REAL    DEFAULT 0,
            status      TEXT    DEFAULT 'OPEN',
            slippage_pct REAL   DEFAULT 0.0
        )
    """)

    # V14 migration: add symbol column if it doesn't exist (for older DBs)
    try:
        c.execute("ALTER TABLE trades ADD COLUMN symbol TEXT NOT NULL DEFAULT 'UNKNOWN'")
    except sqlite3.OperationalError:
        pass  # Column already exists
        
    # V32 migration: add slippage_pct for execution analytics
    try:
        c.execute("ALTER TABLE trades ADD COLUMN slippage_pct REAL DEFAULT 0.0")
    except sqlite3.OperationalError:
        pass

    # V16: Persistent positions table — survives bot restarts
    c.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            symbol      TEXT    PRIMARY KEY,
            side        TEXT    NOT NULL,
            entry       REAL    NOT NULL,
            size_usd    REAL    NOT NULL,
            stop_loss   REAL,
            take_profit REAL,
            trail_high  REAL,
            trail_low   REAL,
            opened_at   TEXT,
            dca_count   INTEGER DEFAULT 0,
            quantity    REAL    DEFAULT 0
        )
    """)

    # V19 migration: add dca_count column for older DBs
    try:
        c.execute("ALTER TABLE positions ADD COLUMN dca_count INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    # V34 migration: add quantity column for older DBs
    try:
        c.execute("ALTER TABLE positions ADD COLUMN quantity REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    c.execute("""
        CREATE TABLE IF NOT EXISTS state (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    # V34: Funding PnL Tracker — separate funding payments from trading PnL
    c.execute("""
        CREATE TABLE IF NOT EXISTS funding_payments (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            symbol      TEXT    NOT NULL,
            funding_rate REAL   NOT NULL,
            position_size_usd REAL NOT NULL,
            funding_pnl REAL   NOT NULL
        )
    """)

    # Seed defaults (ignore if rows already exist)
    c.execute(
        "INSERT OR IGNORE INTO state (key, value) VALUES (?, ?)",
        ("consecutive_losses", "0"),
    )
    c.execute(
        "INSERT OR IGNORE INTO state (key, value) VALUES (?, ?)",
        ("cooldown_until", ""),
    )

    conn.commit()
    conn.close()


def log_funding_payment(
    symbol: str,
    funding_rate: float,
    position_size_usd: float,
    funding_pnl: float,
    db_path: str = DB_PATH,
) -> int:
    """V34: Log a funding rate settlement to the funding_payments table."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    c.execute(
        "INSERT INTO funding_payments (timestamp, symbol, funding_rate, position_size_usd, funding_pnl) "
        "VALUES (?, ?, ?, ?, ?)",
        (now, symbol, funding_rate, position_size_usd, funding_pnl),
    )
    row_id = c.lastrowid
    conn.commit()
    conn.close()
    return row_id


# ─────────────────────────────────────────────────────────────
# Trade Logging
# ─────────────────────────────────────────────────────────────
def log_trade(
    side: str,
    price: float,
    size_usd: float,
    pnl: float,
    db_path: str = DB_PATH,
    symbol: str = "UNKNOWN",
    slippage: float = 0.0,
) -> int:
    """
    Record a completed trade and update the consecutive-loss counter.
    Returns the trade row id.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    now = datetime.now(timezone.utc).isoformat()
    c.execute(
        "INSERT INTO trades (timestamp, symbol, side, price, size_usd, pnl, status, slippage_pct) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (now, symbol, side, price, size_usd, pnl, "CLOSED", slippage),
    )
    trade_id = c.lastrowid

    # Update consecutive-loss streak
    if pnl < 0:
        losses = _get_state_int(c, "consecutive_losses") + 1
        _set_state(c, "consecutive_losses", str(losses))
        if redis_client:
            try:
                redis_client.set("hunter:consecutive_losses", str(losses))
            except Exception: pass
            
        if losses >= MAX_CONSECUTIVE_LOSSES:
            cooldown_end = datetime.now(timezone.utc) + timedelta(hours=COOLDOWN_HOURS)
            _set_state(c, "cooldown_until", cooldown_end.isoformat())
            if redis_client:
                try:
                    redis_client.set("hunter:cooldown_until", cooldown_end.isoformat())
                except Exception: pass
    else:
        _set_state(c, "consecutive_losses", "0")
        if redis_client:
            try:
                redis_client.set("hunter:consecutive_losses", "0")
            except Exception: pass

    conn.commit()
    conn.close()
    return trade_id


# ─────────────────────────────────────────────────────────────
# Persistent Positions  (V16)
# ─────────────────────────────────────────────────────────────
def save_position(symbol: str, pos: Dict, db_path: str = DB_PATH) -> None:
    """Upsert an open position so it survives restarts."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """INSERT OR REPLACE INTO positions
           (symbol, side, entry, size_usd, stop_loss, take_profit,
            trail_high, trail_low, opened_at, dca_count, quantity)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            symbol,
            pos["side"],
            pos["entry"],
            pos["size_usd"],
            pos.get("stop_loss"),
            pos.get("take_profit"),
            pos.get("trail_high"),
            pos.get("trail_low"),
            pos.get("opened_at", datetime.now(timezone.utc).isoformat()),
            pos.get("dca_count", 0),
            pos.get("quantity", 0),
        ),
    )
    conn.commit()
    conn.close()


def load_positions(db_path: str = DB_PATH) -> Dict[str, Dict]:
    """Load all open positions from DB on startup."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        "SELECT symbol, side, entry, size_usd, stop_loss, take_profit, "
        "trail_high, trail_low, opened_at, dca_count, quantity FROM positions"
    )
    rows = c.fetchall()
    conn.close()

    positions = {}
    for r in rows:
        positions[r[0]] = {
            "side": r[1],
            "entry": r[2],
            "size_usd": r[3],
            "stop_loss": r[4],
            "take_profit": r[5],
            "trail_high": r[6],
            "trail_low": r[7],
            "opened_at": r[8],
            "dca_count": r[9],
            "quantity": r[10],
        }
    return positions

    result = {}
    for row in rows:
        sym, side, entry, size_usd, sl, tp, th, tl, opened_at = row
        result[sym] = {
            "side": side,
            "entry": entry,
            "size_usd": size_usd,
            "stop_loss": sl,
            "take_profit": tp,
            "trail_high": th,
            "trail_low": tl,
            "opened_at": opened_at,
        }
    return result


def delete_position(symbol: str, db_path: str = DB_PATH) -> None:
    """Remove a closed position from the DB."""
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────
# Trade Statistics — for Kelly Criterion  (V16)
# ─────────────────────────────────────────────────────────────
def get_trade_stats(db_path: str = DB_PATH, n_recent: int = 50) -> Dict:
    """
    Compute win-rate, avg_win, avg_loss over last n_recent closed trades.

    Returns dict with keys: n_trades, win_rate, avg_win, avg_loss.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        "SELECT pnl FROM trades WHERE status='CLOSED' ORDER BY id DESC LIMIT ?",
        (n_recent,),
    )
    rows = c.fetchall()
    conn.close()

    if not rows:
        return {"n_trades": 0, "win_rate": 0.5, "avg_win": 1.0, "avg_loss": 1.0}

    pnls   = [r[0] for r in rows]
    wins   = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]

    return {
        "n_trades": len(pnls),
        "win_rate": len(wins) / len(pnls),
        "avg_win":  sum(wins)   / len(wins)   if wins   else 1.0,
        "avg_loss": sum(losses) / len(losses) if losses else 1.0,
    }


# ─────────────────────────────────────────────────────────────
# Circuit-Breaker Queries
# ─────────────────────────────────────────────────────────────
def get_consecutive_losses(db_path: str = DB_PATH) -> int:
    # Fast path: Redis
    if redis_client:
        try:
            val = redis_client.get("hunter:consecutive_losses")
            if val is not None:
                return int(val)
        except Exception:
            pass
            
    # Fallback to SQLite (IO-bound)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    val = _get_state_int(c, "consecutive_losses")
    conn.close()
    return val


def get_cooldown_until(db_path: str = DB_PATH) -> Optional[datetime]:
    # Fast Path: Redis
    raw = None
    if redis_client:
        try:
            raw = redis_client.get("hunter:cooldown_until")
        except Exception:
            pass
            
    if not raw:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        raw = _get_state_str(c, "cooldown_until")
        conn.close()
        
    if not raw:
        return None
    return datetime.fromisoformat(raw)


def set_cooldown_until(dt: Optional[datetime], db_path: str = DB_PATH) -> None:
    val = dt.isoformat() if dt else ""
    if redis_client:
        try:
            redis_client.set("hunter:cooldown_until", val)
        except Exception:
            pass

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    _set_state(c, "cooldown_until", val)
    conn.commit()
    conn.close()


def is_on_cooldown(db_path: str = DB_PATH) -> bool:
    """Return True if the circuit breaker is currently active."""
    cd = get_cooldown_until(db_path)
    if cd is None:
        return False
    return datetime.now(timezone.utc) < cd


def reset_circuit_breaker(db_path: str = DB_PATH) -> None:
    """Manually reset both the loss counter and the cooldown."""
    if redis_client:
        try:
            redis_client.set("hunter:consecutive_losses", "0")
            redis_client.set("hunter:cooldown_until", "")
        except Exception:
            pass

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    _set_state(c, "consecutive_losses", "0")
    _set_state(c, "cooldown_until", "")
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────
def _get_state_str(cursor: sqlite3.Cursor, key: str) -> str:
    cursor.execute("SELECT value FROM state WHERE key = ?", (key,))
    row = cursor.fetchone()
    return row[0] if row else ""


def _get_state_int(cursor: sqlite3.Cursor, key: str) -> int:
    raw = _get_state_str(cursor, key)
    return int(raw) if raw else 0


def _set_state(cursor: sqlite3.Cursor, key: str, value: str) -> None:
    cursor.execute(
        "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
        (key, value),
    )
