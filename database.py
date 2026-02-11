"""
Hunter V11 — Database Module
==============================
SQLite-backed trade journal + circuit-breaker state management.
Tracks every trade outcome and enforces the 3-consecutive-loss cooldown.
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

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
            side        TEXT    NOT NULL,   -- BUY / SELL
            price       REAL    NOT NULL,
            size_usd    REAL    NOT NULL,
            pnl         REAL    DEFAULT 0,
            status      TEXT    DEFAULT 'OPEN'
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS state (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
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


# ─────────────────────────────────────────────────────────────
# Trade Logging
# ─────────────────────────────────────────────────────────────
def log_trade(
    side: str,
    price: float,
    size_usd: float,
    pnl: float,
    db_path: str = DB_PATH,
) -> int:
    """
    Record a completed trade and update the consecutive-loss counter.

    Returns the trade row id.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    now = datetime.now(timezone.utc).isoformat()
    c.execute(
        "INSERT INTO trades (timestamp, side, price, size_usd, pnl, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (now, side, price, size_usd, pnl, "CLOSED"),
    )
    trade_id = c.lastrowid

    # Update consecutive-loss streak
    if pnl < 0:
        losses = _get_state_int(c, "consecutive_losses") + 1
        _set_state(c, "consecutive_losses", str(losses))

        # Trip the circuit breaker if threshold is reached
        if losses >= MAX_CONSECUTIVE_LOSSES:
            cooldown_end = datetime.now(timezone.utc) + timedelta(hours=COOLDOWN_HOURS)
            _set_state(c, "cooldown_until", cooldown_end.isoformat())
    else:
        # A win (or break-even) resets the streak
        _set_state(c, "consecutive_losses", "0")

    conn.commit()
    conn.close()
    return trade_id


# ─────────────────────────────────────────────────────────────
# Circuit-Breaker Queries
# ─────────────────────────────────────────────────────────────
def get_consecutive_losses(db_path: str = DB_PATH) -> int:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    val = _get_state_int(c, "consecutive_losses")
    conn.close()
    return val


def get_cooldown_until(db_path: str = DB_PATH) -> Optional[datetime]:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    raw = _get_state_str(c, "cooldown_until")
    conn.close()
    if not raw:
        return None
    return datetime.fromisoformat(raw)


def set_cooldown_until(dt: Optional[datetime], db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    _set_state(c, "cooldown_until", dt.isoformat() if dt else "")
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
