"""
Hunter V18 — Structured Reporting Module
========================================
Generates human-readable console reports for each trading cycle.
"""

import logging
from typing import Dict, Any

from analysis import MarketState
from config import TIMEFRAME

logger = logging.getLogger("hunter.report")

class ReportGenerator:
    """Generates structured CLI reports for trades and analysis."""
    
    @staticmethod
    def print_cycle_report(symbol: str, state: MarketState, signal_dict: Dict[str, Any], execute_result: Dict[str, Any]):
        """
        Prints a consolidated block with all relevant cycle information.
        """
        action = signal_dict.get("action", "HOLD")
        conf = signal_dict.get("confidence", 0.0)
        score = signal_dict.get("composite_score", 0.0)
        
        # Determine color/emoji for action
        action_color = "🟢" if "BUY" in action else "🔴" if "SELL" in action or "SHORT" in action else "⚪"
        
        # Build the strings
        report = []
        report.append("="*60)
        report.append(f" {action_color} SIGNAL: {action} | {symbol} ({TIMEFRAME})".ljust(59) + "|")
        report.append("-" * 60)
        
        # Core Metrics
        price_str = f"Price: ${state.current_price:,.2f}"
        conf_str = f"Conf: {conf:.1f}%"
        score_str = f"Net Score: {score:+.3f}"
        report.append(f" {price_str:<20} {conf_str:<18} {score_str:<18} |")
        
        # Technical Snapshot
        regime_icon = "📈" if state.regime == "TRENDING" else "🦀"
        rsi_str = f"RSI: {state.rsi:.1f}"
        vwap_str = f"VWAP Diff: {state.vwap_diff_pct:+.2f}%"
        macd_str = f"MACD Hist: {state.macd_histogram:+.2f}"
        report.append(f" {regime_icon} {state.regime:<18} {rsi_str:<18} {macd_str:<18} |")
        
        # Sentiment & Macro Snapshot
        ls_str = f"L/S Ratio: {state.ls_ratio:.2f}"
        soc_str = f"Social: {state.social_score:+.2f}"
        btc_cor_str = f"BTC Corr: {state.btc_correlation:+.2f}"
        report.append(f" 📊 {ls_str:<18} {soc_str:<18} {btc_cor_str:<18} |")
        
        # Overrides & Execution Result
        report.append("-" * 60)
        exec_msg = execute_result.get("msg", "")
        pnl = execute_result.get("pnl", 0.0)
        
        if pnl != 0.0:
            pnl_str = f"PnL: ${pnl:+.2f}"
            report.append(f" 💼 EXECUTION: {exec_msg} ({pnl_str})".ljust(59) + "|")
        else:
            report.append(f" 💼 EXECUTION: {exec_msg}".ljust(59) + "|")
            
        report.append("=" * 60)
        
        # Output as one block to prevent interleaving in async logs
        block = "\n" + "\n".join(report) + "\n"
        print(block)
