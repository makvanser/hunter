"""
Hunter V11 — Analysis Module
==============================
Pure-function Technical Analysis calculations.
No external data-fetching, no side-effects.

Implements RSI, ADX, and Bollinger Bands from scratch
(no TA-Lib C dependency required).
"""

from typing import List, Tuple

from config import (
    ADX_PERIOD,
    ADX_THRESHOLD,
    BB_PERIOD,
    BB_STD,
    LS_RATIO_THRESHOLD,
    RSI_OVERSOLD,
    RSI_PERIOD,
    WHALE_NET_VOL_MIN,
)


# ─────────────────────────────────────────────────────────────
# RSI  (Wilder's smoothing)
# ─────────────────────────────────────────────────────────────
def compute_rsi(closes: List[float], period: int = RSI_PERIOD) -> float:
    """
    Relative Strength Index using Wilder's exponential smoothing.

    Requires at least ``period + 1`` data points.
    Returns a value in [0, 100].
    """
    if len(closes) < period + 1:
        raise ValueError(
            f"Need at least {period + 1} closes, got {len(closes)}"
        )

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    # Seed: simple average of first `period` changes
    gains = [d if d > 0 else 0.0 for d in deltas[:period]]
    losses = [-d if d < 0 else 0.0 for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Wilder smoothing for the rest
    for d in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(d, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0.0)) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ─────────────────────────────────────────────────────────────
# ADX  (Average Directional Index)
# ─────────────────────────────────────────────────────────────
def compute_adx(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = ADX_PERIOD,
) -> float:
    """
    Average Directional Index.

    Requires at least ``2 * period + 1`` bars.
    Returns a value ≥ 0 (typically 0–100).
    """
    n = len(closes)
    if n < 2 * period + 1:
        raise ValueError(
            f"Need at least {2 * period + 1} bars for ADX, got {n}"
        )

    # Step 1: True Range, +DM, -DM
    tr_list: List[float] = []
    plus_dm_list: List[float] = []
    minus_dm_list: List[float] = []

    for i in range(1, n):
        hi = highs[i]
        lo = lows[i]
        prev_close = closes[i - 1]
        prev_hi = highs[i - 1]
        prev_lo = lows[i - 1]

        tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        up_move = hi - prev_hi
        down_move = prev_lo - lo

        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

        tr_list.append(tr)
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

    # Step 2: Wilder-smooth TR, +DM, -DM over `period`
    def wilder_smooth(data: List[float], p: int) -> List[float]:
        smoothed = [sum(data[:p])]
        for val in data[p:]:
            smoothed.append(smoothed[-1] - (smoothed[-1] / p) + val)
        return smoothed

    atr = wilder_smooth(tr_list, period)
    smooth_plus = wilder_smooth(plus_dm_list, period)
    smooth_minus = wilder_smooth(minus_dm_list, period)

    # Step 3: +DI, -DI, DX
    dx_list: List[float] = []
    for i in range(len(atr)):
        if atr[i] == 0:
            dx_list.append(0.0)
            continue
        plus_di = 100.0 * smooth_plus[i] / atr[i]
        minus_di = 100.0 * smooth_minus[i] / atr[i]
        di_sum = plus_di + minus_di
        dx = (abs(plus_di - minus_di) / di_sum * 100.0) if di_sum != 0 else 0.0
        dx_list.append(dx)

    # Step 4: ADX = Wilder-smooth of DX over `period`
    if len(dx_list) < period:
        raise ValueError("Not enough DX values to compute ADX")

    adx = sum(dx_list[:period]) / period
    for dx_val in dx_list[period:]:
        adx = (adx * (period - 1) + dx_val) / period

    return adx


# ─────────────────────────────────────────────────────────────
# Bollinger Bands
# ─────────────────────────────────────────────────────────────
def compute_bollinger(
    closes: List[float],
    period: int = BB_PERIOD,
    num_std: float = BB_STD,
) -> Tuple[float, float, float]:
    """
    Returns (upper_band, middle_band, lower_band) for the latest bar.
    """
    if len(closes) < period:
        raise ValueError(
            f"Need at least {period} closes for Bollinger, got {len(closes)}"
        )

    window = closes[-period:]
    middle = sum(window) / period
    variance = sum((x - middle) ** 2 for x in window) / period
    std = variance ** 0.5
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


# ─────────────────────────────────────────────────────────────
# Market Regime Filter
# ─────────────────────────────────────────────────────────────
def get_market_regime(adx_value: float) -> str:
    """Return 'TRENDING' if ADX ≥ threshold, else 'CHOPPY'."""
    return "TRENDING" if adx_value >= ADX_THRESHOLD else "CHOPPY"


# ─────────────────────────────────────────────────────────────
# Contrarian Signal Generator
# ─────────────────────────────────────────────────────────────
def generate_signal(
    rsi: float,
    ls_ratio: float,
    whale_net_vol: float,
    regime: str,
) -> str:
    """
    Contrarian signal logic (CoinBureau-inspired).

    SELL = RSI > 70 (overbought exit)
           Works in ANY regime — never trap a position in flat.

    BUY  = regime is TRENDING
           AND RSI < 30          (oversold)
           AND Long/Short < 0.8  (crowd is shorting)
           AND Whale net vol > 0 (smart money buying)

    Otherwise → HOLD.
    """
    # ── Sell: EXIT is always allowed (even in CHOPPY) ──
    if rsi > 70:
        return "SELL"

    # ── Buy: ENTRY only in trending market ──
    if regime != "TRENDING":
        return "HOLD"

    if (
        rsi < RSI_OVERSOLD
        and ls_ratio < LS_RATIO_THRESHOLD
        and whale_net_vol > WHALE_NET_VOL_MIN
    ):
        return "BUY"

    return "HOLD"

