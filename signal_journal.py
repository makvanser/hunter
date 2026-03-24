"""
Hunter V26 — Signal Journal
============================
Records EVERY signal (taken AND blocked) with full market context
to SQLite for post-hoc analysis of filter effectiveness.

Every cycle writes one row. The weekly analyzer later fills in
price_after_1h/4h/24h to compute would_have_profited.
"""

import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict

logger = logging.getLogger("hunter.journal")

DB_PATH = "signal_journal.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_journal():
    """Create the signals table if it doesn't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            composite_score REAL,
            original_action TEXT,
            final_action TEXT,
            blocked_by TEXT,
            ml_confidence REAL,
            obi REAL,
            rsi REAL,
            adx REAL,
            atr_pct REAL,
            regime TEXT,
            price_at_signal REAL,
            price_after_1h REAL,
            price_after_4h REAL,
            price_after_24h REAL,
            would_have_profited INTEGER
        )
    """)
    conn.commit()
    conn.close()
    logger.info("📓 Signal Journal initialized (%s)", DB_PATH)


def log_signal(
    symbol: str,
    composite_score: float,
    original_action: str,
    final_action: str,
    blocked_by: Optional[str],
    ml_confidence: float,
    obi: float,
    rsi: float,
    adx: float,
    atr_pct: float,
    regime: str,
    price_at_signal: float,
):
    """Log one signal event to the journal."""
    try:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO signals 
               (timestamp, symbol, composite_score, original_action, final_action,
                blocked_by, ml_confidence, obi, rsi, adx, atr_pct, regime, price_at_signal)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                symbol,
                round(composite_score, 5),
                original_action,
                final_action,
                blocked_by,
                round(ml_confidence, 4),
                round(obi, 4),
                round(rsi, 2),
                round(adx, 2),
                round(atr_pct, 3),
                regime,
                round(price_at_signal, 8),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("Journal write failed: %s", e)


def get_unanalyzed_signals(limit: int = 500) -> List[Dict]:
    """Get signals where outcome hasn't been computed yet."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT * FROM signals 
           WHERE price_after_4h IS NULL 
             AND original_action != 'HOLD'
           ORDER BY timestamp ASC 
           LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_outcome(
    signal_id: int,
    price_after_1h: float,
    price_after_4h: float,
    price_after_24h: float,
    would_have_profited: bool,
):
    """Fill in the outcome columns for a signal."""
    conn = _get_conn()
    conn.execute(
        """UPDATE signals 
           SET price_after_1h = ?, price_after_4h = ?, price_after_24h = ?,
               would_have_profited = ?
           WHERE id = ?""",
        (
            round(price_after_1h, 8),
            round(price_after_4h, 8),
            round(price_after_24h, 8),
            1 if would_have_profited else 0,
            signal_id,
        ),
    )
    conn.commit()
    conn.close()


def get_weekly_stats() -> Dict:
    """Get summary statistics for the last 7 days."""
    conn = _get_conn()
    
    # Total signals by action
    rows = conn.execute(
        """SELECT original_action, final_action, blocked_by,
                  COUNT(*) as cnt,
                  SUM(CASE WHEN would_have_profited = 1 THEN 1 ELSE 0 END) as profitable,
                  AVG(composite_score) as avg_score
           FROM signals
           WHERE timestamp > datetime('now', '-7 days')
             AND original_action != 'HOLD'
           GROUP BY original_action, final_action, blocked_by
           ORDER BY cnt DESC"""
    ).fetchall()
    conn.close()
    
    return [dict(r) for r in rows]
