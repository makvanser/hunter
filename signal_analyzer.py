"""
Hunter V26 — Signal Analyzer (Weekly Self-Diagnosis)
=====================================================
Runs weekly via apscheduler. For each recorded signal:
  1. Fetches what price actually did 1h/4h/24h later
  2. Computes whether the blocked signal would have been profitable
  3. Generates a report showing which filters are destroying alpha

Output: structured log + signal_report.txt
"""

import logging
import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta
from typing import Dict, List

from signal_journal import get_unanalyzed_signals, update_outcome, get_weekly_stats

logger = logging.getLogger("hunter.analyzer")

MIN_PROFIT_PCT = 0.3  # A move > 0.3% in the right direction = "would have profited"


async def _fetch_price_at(symbol: str, timestamp_ms: int) -> float:
    """Fetch the close price of a 1m candle at a specific time from Binance."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": timestamp_ms,
        "limit": 1,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
                if data and len(data) > 0:
                    return float(data[0][4])  # Close price
    except Exception as e:
        logger.error("Price fetch failed for %s at %d: %s", symbol, timestamp_ms, e)
    return 0.0


def _would_have_profited(action: str, entry_price: float, future_price: float) -> bool:
    """Check if the signal direction was correct."""
    if entry_price <= 0 or future_price <= 0:
        return False
    pct_change = (future_price - entry_price) / entry_price * 100
    
    if action in ("BUY", "DCA_BUY"):
        return pct_change > MIN_PROFIT_PCT
    elif action in ("SHORT", "DCA_SHORT"):
        return pct_change < -MIN_PROFIT_PCT
    return False


async def analyze_signals():
    """
    Main analysis loop:
    1. Get all unanalyzed signals
    2. Fetch actual future prices
    3. Update outcomes
    4. Generate report
    """
    signals = get_unanalyzed_signals(limit=300)
    
    if not signals:
        logger.info("📓 No unanalyzed signals found. Skipping analysis.")
        return
    
    logger.info("📊 Analyzing %d signals from journal...", len(signals))
    
    analyzed = 0
    for sig in signals:
        try:
            ts = datetime.fromisoformat(sig["timestamp"])
            ts_ms = int(ts.timestamp() * 1000)
            
            # Only analyze signals older than 24h (so we have price_after_24h)
            if datetime.now(timezone.utc) - ts < timedelta(hours=25):
                continue
            
            symbol = sig["symbol"]
            
            # Fetch prices at 1h, 4h, 24h after signal
            p1h = await _fetch_price_at(symbol, ts_ms + 3_600_000)
            p4h = await _fetch_price_at(symbol, ts_ms + 14_400_000)
            p24h = await _fetch_price_at(symbol, ts_ms + 86_400_000)
            
            action = sig["original_action"]
            profited = _would_have_profited(action, sig["price_at_signal"], p4h)
            
            update_outcome(sig["id"], p1h, p4h, p24h, profited)
            analyzed += 1
            
            # Rate limit: don't spam Binance
            await asyncio.sleep(0.2)
            
        except Exception as e:
            logger.error("Failed to analyze signal %d: %s", sig["id"], e)
    
    logger.info("✅ Analyzed %d/%d signals", analyzed, len(signals))
    
    # Generate report
    _generate_report()


def _generate_report():
    """Generate and log the weekly filter effectiveness report."""
    stats = get_weekly_stats()
    
    if not stats:
        logger.info("📊 No data for weekly report.")
        return
    
    lines = []
    lines.append("=" * 70)
    lines.append("  📊 WEEKLY SIGNAL ANALYSIS REPORT")
    lines.append("=" * 70)
    
    total_signals = 0
    total_blocked = 0
    blocked_would_profit = 0
    taken_would_profit = 0
    taken_total = 0
    filter_stats = {}
    
    for row in stats:
        cnt = row["cnt"]
        profitable = row["profitable"] or 0
        original = row["original_action"]
        final = row["final_action"]
        blocked_by = row["blocked_by"]
        avg_score = row["avg_score"] or 0
        
        total_signals += cnt
        
        was_blocked = (original != final) or (final == "HOLD" and original != "HOLD")
        
        if was_blocked and blocked_by:
            total_blocked += cnt
            blocked_would_profit += profitable
            
            if blocked_by not in filter_stats:
                filter_stats[blocked_by] = {"blocked": 0, "would_profit": 0}
            filter_stats[blocked_by]["blocked"] += cnt
            filter_stats[blocked_by]["would_profit"] += profitable
        elif original != "HOLD":
            taken_total += cnt
            taken_would_profit += profitable
        
        hit_pct = (profitable / cnt * 100) if cnt > 0 else 0
        lines.append(
            f"  {original:>10} → {final:<10} | blocked_by={blocked_by or 'N/A':<12} | "
            f"count={cnt:>4} | hit_rate={hit_pct:>5.1f}% | avg_score={avg_score:>+.3f}"
        )
    
    lines.append("-" * 70)
    
    # Summary
    taken_hit = (taken_would_profit / taken_total * 100) if taken_total > 0 else 0
    blocked_hit = (blocked_would_profit / total_blocked * 100) if total_blocked > 0 else 0
    
    lines.append(f"  TAKEN signals:   {taken_total:>4} | Hit rate: {taken_hit:.1f}%")
    lines.append(f"  BLOCKED signals: {total_blocked:>4} | Would-have-hit rate: {blocked_hit:.1f}%")
    
    if blocked_hit > taken_hit and total_blocked > 10:
        lines.append("")
        lines.append("  ⚠️  WARNING: Blocked signals have HIGHER hit rate than taken signals!")
        lines.append("  ⚠️  Your filters are DESTROYING ALPHA. Consider loosening them.")
    
    # Per-filter breakdown
    if filter_stats:
        lines.append("")
        lines.append("  Per-Filter Breakdown:")
        for fname, fdata in sorted(filter_stats.items(), key=lambda x: x[1]["blocked"], reverse=True):
            fhit = (fdata["would_profit"] / fdata["blocked"] * 100) if fdata["blocked"] > 0 else 0
            verdict = "🟢 GOOD" if fhit < 40 else "🟡 REVIEW" if fhit < 55 else "🔴 REMOVE"
            lines.append(
                f"    {fname:<15} blocked {fdata['blocked']:>4} signals | "
                f"{fhit:.1f}% would have profited | {verdict}"
            )
    
    lines.append("=" * 70)
    
    report = "\n".join(lines)
    logger.info("\n%s", report)
    
    # Also save to file
    try:
        with open("signal_report.txt", "w") as f:
            f.write(report)
        logger.info("📄 Report saved to signal_report.txt")
    except Exception as e:
        logger.error("Failed to save report: %s", e)


async def run_weekly_analysis():
    """Entry point for the scheduler."""
    logger.info("📅 Starting weekly signal analysis...")
    await analyze_signals()
