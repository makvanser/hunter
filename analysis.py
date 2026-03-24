"""
Hunter V16 — Analysis Module
==============================
Pure-function Technical Analysis calculations.
No external data-fetching, no side-effects.

Implements RSI, ADX, Bollinger Bands, MACD, EMA/SMA, ATR, VWAP,
Divergence Detection, Support/Resistance, Composite Signal Scoring,
Stochastic RSI, and RSI Slope Filter.

V13: Added news_sentiment parameter to generate_signal()
V14: Added MACD, ATR, VWAP, EMA/SMA, divergence, S/R, composite scoring.
V15: Added funding rate, swing-point divergence.
V16: Added StochRSI, RSI slope filter, ADX-guard for divergence.
"""

import logging
import math
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

from config import (
    ADX_PERIOD,
    ADX_STRONG_TREND,
    ADX_THRESHOLD,
    ATR_PERIOD,
    BB_PERIOD,
    BB_STD,
    COMPOSITE_BUY_THRESHOLD,
    COMPOSITE_SELL_THRESHOLD,
    LS_RATIO_THRESHOLD,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    NEWS_OVERRIDE_RSI,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_PERIOD,
    RSI_SLOPE_BARS,
    SR_LOOKBACK,
    STOCH_RSI_PERIOD,
    WEIGHT_BOLLINGER,
    WEIGHT_FUNDING,
    WEIGHT_LS_RATIO,
    WEIGHT_MACD,
    WEIGHT_MACRO,
    WEIGHT_MTF,
    WEIGHT_SOCIAL,
    WEIGHT_RSI,
    WEIGHT_WHALE,
    WHALE_NET_VOL_MIN,
    VOLUME_CONFIRM_ENABLED,
    MAX_DCA_STEPS,
    DCA_PRICE_DROP_PCT,
    DCA_MAX_DRAWDOWN_PCT,
    MAKER_GRID_ENABLED,
)

# Local adjustments for V17 where config doesn't have them
WEIGHT_BB = 0.15
WEIGHT_TREND = 0.10
WEIGHT_VWAP = 0.03
WEAK_WHALE_USD = 100000
WEIGHT_OI = 0.04
WEIGHT_DIV = 0.03
USE_COMPOSITE = True

logger = logging.getLogger("hunter.analysis")


# ─────────────────────────────────────────────────────────────
# Market State (V17 refactoring)
# ─────────────────────────────────────────────────────────────
@dataclass
class MarketState:
    """Encapsulates all computed indicators for a single cycle."""
    current_price: float
    rsi: float
    ls_ratio: float
    whale_net_vol: float
    regime: str
    social_score: float
    macd_histogram: float
    bb_position: float
    vwap_diff_pct: float
    divergence: str
    funding_rate: float
    open_interest_delta: float
    liq_imbalance: float
    atr_pct: float
    rsi_slope: float
    stoch_rsi: float
    mtf_agreement: float
    volume_confirm: bool
    near_resistance: bool
    btc_correlation: float
    btc_dominance: float
    cvd: float = 0.0
    btc_spread_zscore: float = 0.0
    # V29 Phase 3: High-Freq Microstructure
    obi: float = 0.0
    obi_delta: float = 0.0
    cvd_slope: float = 0.0
    bid_ask_spread: float = 0.0
    kalman_zscore: float = 0.0


# ─────────────────────────────────────────────────────────────
# EMA / SMA  (reusable building blocks)
# ─────────────────────────────────────────────────────────────
def compute_sma(data: List[float], period: int) -> float:
    """Simple Moving Average of the last `period` values."""
    if len(data) < period:
        raise ValueError(f"Need at least {period} data points, got {len(data)}")
    return sum(data[-period:]) / period


def compute_ema(data: List[float], period: int) -> float:
    """
    Exponential Moving Average.

    Uses SMA of first `period` values as seed, then applies the
    standard EMA multiplier k = 2 / (period + 1).
    """
    if len(data) < period:
        raise ValueError(f"Need at least {period} data points, got {len(data)}")

    k = 2.0 / (period + 1)
    ema = sum(data[:period]) / period  # seed with SMA

    for price in data[period:]:
        ema = price * k + ema * (1 - k)

    return ema


def _compute_ema_series(data: List[float], period: int) -> List[float]:
    """
    Full EMA series (internal helper).

    Returns a list the same length as `data` (first `period-1` entries
    are forward-filled with the seed SMA).
    """
    if len(data) < period:
        raise ValueError(f"Need at least {period} data points, got {len(data)}")

    k = 2.0 / (period + 1)
    seed = sum(data[:period]) / period
    result = [seed] * period  # fill initial values with seed

    ema = seed
    for price in data[period:]:
        ema = price * k + ema * (1 - k)
        result.append(ema)

    return result


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


def compute_rsi_series(closes: List[float], period: int = RSI_PERIOD) -> List[float]:
    """Full RSI series for divergence detection."""
    if len(closes) < period + 1:
        raise ValueError(f"Need at least {period + 1} closes, got {len(closes)}")

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    rsi_values: List[float] = []

    gains = [d if d > 0 else 0.0 for d in deltas[:period]]
    losses_list = [-d if d < 0 else 0.0 for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses_list) / period

    if avg_loss == 0:
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

    for d in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(d, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0.0)) / period
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

    return rsi_values


# ─────────────────────────────────────────────────────────────
# Stochastic RSI  (V16)
# ─────────────────────────────────────────────────────────────
def compute_stoch_rsi(
    closes: List[float],
    rsi_period: int = RSI_PERIOD,
    stoch_period: int = STOCH_RSI_PERIOD,
) -> float:
    """
    Stochastic RSI: (RSI - min_RSI) / (max_RSI - min_RSI) × 100

    Returns 0–100. < 20 = oversold, > 80 = overbought.
    More sensitive to reversals than plain RSI.
    """
    rsi_vals = compute_rsi_series(closes, rsi_period)
    if len(rsi_vals) < stoch_period:
        return 50.0  # Neutral fallback
    window = rsi_vals[-stoch_period:]
    lo, hi = min(window), max(window)
    if hi == lo:
        return 50.0
    return (rsi_vals[-1] - lo) / (hi - lo) * 100.0


# ─────────────────────────────────────────────────────────────
# RSI Slope Filter  (V16)
# ─────────────────────────────────────────────────────────────
def compute_rsi_slope(
    closes: List[float],
    rsi_period: int = RSI_PERIOD,
    slope_bars: int = RSI_SLOPE_BARS,
) -> float:
    """
    RSI slope over the last `slope_bars` RSI values.

    Positive = RSI rising (momentum building).
    Negative = RSI falling (fading momentum).
    Returns RSI[-1] - RSI[-1-slope_bars].
    """
    need = rsi_period + slope_bars + 1
    if len(closes) < need:
        return 0.0
    rsi_vals = compute_rsi_series(closes, rsi_period)
    if len(rsi_vals) < slope_bars + 1:
        return 0.0
    return rsi_vals[-1] - rsi_vals[-1 - slope_bars]


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
    Compute Bollinger Bands for the last `period` closes.

    Returns (lower_band, middle_band, upper_band) — ascending order.
    lower < middle < upper always.
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
    return lower, middle, upper  # ascending order: lower < middle < upper


# ─────────────────────────────────────────────────────────────
# MACD  (V14)
# ─────────────────────────────────────────────────────────────
def compute_macd(
    closes: List[float],
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal_period: int = MACD_SIGNAL,
) -> Tuple[float, float, float]:
    """
    Moving Average Convergence Divergence.

    Returns (macd_line, signal_line, histogram) for the latest bar.

    - macd_line   = EMA(fast) - EMA(slow)
    - signal_line = EMA(macd_line_series, signal_period)
    - histogram   = macd_line - signal_line
    """
    min_needed = slow + signal_period
    if len(closes) < min_needed:
        raise ValueError(
            f"Need at least {min_needed} closes for MACD, got {len(closes)}"
        )

    fast_ema_series = _compute_ema_series(closes, fast)
    slow_ema_series = _compute_ema_series(closes, slow)

    # MACD line starts at index (slow-1) where slow EMA becomes valid
    macd_series = [
        fast_ema_series[i] - slow_ema_series[i]
        for i in range(slow - 1, len(closes))
    ]

    # Signal line = EMA of MACD series
    if len(macd_series) < signal_period:
        raise ValueError("Not enough MACD values for signal line")

    signal_series = _compute_ema_series(macd_series, signal_period)

    macd_line = macd_series[-1]
    signal_line = signal_series[-1]
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# ─────────────────────────────────────────────────────────────
# ATR  (Average True Range)  (V14)
# ─────────────────────────────────────────────────────────────
def compute_atr(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = ATR_PERIOD,
) -> float:
    """
    Average True Range using Wilder's smoothing.

    Requires at least ``period + 1`` bars.
    Returns the latest ATR value.
    """
    n = len(closes)
    if n < period + 1:
        raise ValueError(f"Need at least {period + 1} bars for ATR, got {n}")

    tr_list: List[float] = []
    for i in range(1, n):
        hi = highs[i]
        lo = lows[i]
        prev_close = closes[i - 1]
        tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        tr_list.append(tr)

    # Wilder smoothing
    atr = sum(tr_list[:period]) / period
    for tr_val in tr_list[period:]:
        atr = (atr * (period - 1) + tr_val) / period

    return atr


# ─────────────────────────────────────────────────────────────
# VWAP  (Volume-Weighted Average Price)  (V14)
# ─────────────────────────────────────────────────────────────
def compute_vwap(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
) -> float:
    """
    Session VWAP = Σ(typical_price × volume) / Σ(volume).

    Typical price = (high + low + close) / 3.
    Returns 0.0 if total volume is zero.
    """
    if not (len(highs) == len(lows) == len(closes) == len(volumes)):
        raise ValueError("All input lists must have the same length")
    if len(closes) == 0:
        raise ValueError("Need at least 1 bar for VWAP")

    cum_tp_vol = 0.0
    cum_vol = 0.0
    for h, l, c, v in zip(highs, lows, closes, volumes):
        tp = (h + l + c) / 3.0
        cum_tp_vol += tp * v
        cum_vol += v

    return cum_tp_vol / cum_vol if cum_vol > 0 else 0.0


# ─────────────────────────────────────────────────────────────
# Divergence Detection  (V15 — Swing-Point based)
# ─────────────────────────────────────────────────────────────
def _find_swing_lows(data: List[float], window: int = 3) -> List[int]:
    """
    Find indices of local minima in `data`.
    A point at index i is a swing low if it is the minimum
    within [i-window, i+window].
    """
    n = len(data)
    lows = []
    for i in range(window, n - window):
        if data[i] == min(data[i - window: i + window + 1]):
            lows.append(i)
    return lows


def _find_swing_highs(data: List[float], window: int = 3) -> List[int]:
    """
    Find indices of local maxima in `data`.
    A point at index i is a swing high if it is the maximum
    within [i-window, i+window].
    """
    n = len(data)
    highs = []
    for i in range(window, n - window):
        if data[i] == max(data[i - window: i + window + 1]):
            highs.append(i)
    return highs


def detect_divergence(
    prices: List[float],
    oscillator: List[float],
    lookback: int = 60,
    swing_window: int = 3,
    adx_value: float = 0.0,
) -> str:
    """
    Detect bullish or bearish divergence using true swing points.

    V16: adx_value parameter — when ADX > ADX_STRONG_TREND (40),
    divergences are often false signals in a strong trend, so NONE is returned.

    Bullish divergence:
      - Price makes a lower swing low (price[swL2] < price[swL1])
      - Oscillator makes a higher swing low (osc[swL2] > osc[swL1])

    Bearish divergence:
      - Price makes a higher swing high (price[swH2] > price[swH1])
      - Oscillator makes a lower swing high (osc[swH2] < osc[swH1])

    Returns "BULLISH_DIV", "BEARISH_DIV", or "NONE".
    """
    # V16: ADX guard — in strong trends divergences are unreliable
    if adx_value > ADX_STRONG_TREND:
        return "NONE"

    if len(prices) < lookback or len(oscillator) < lookback:
        return "NONE"


    # Use the last `lookback` bars
    p = prices[-lookback:]
    o = oscillator[-lookback:]

    # Find swing highs and lows in both series
    price_lows = _find_swing_lows(p, swing_window)
    price_highs = _find_swing_highs(p, swing_window)

    # ── Bullish Divergence ──
    if len(price_lows) >= 2:
        # Take two most recent swing lows
        swL1 = price_lows[-2]
        swL2 = price_lows[-1]
        if p[swL2] < p[swL1] and o[swL2] > o[swL1]:
            return "BULLISH_DIV"

    # ── Bearish Divergence ──
    if len(price_highs) >= 2:
        swH1 = price_highs[-2]
        swH2 = price_highs[-1]
        if p[swH2] > p[swH1] and o[swH2] < o[swH1]:
            return "BEARISH_DIV"

    return "NONE"


# ─────────────────────────────────────────────────────────────
# Support & Resistance  (V14)
# ─────────────────────────────────────────────────────────────
def compute_support_resistance(
    highs: List[float],
    lows: List[float],
    lookback: int = SR_LOOKBACK,
) -> Tuple[List[float], List[float]]:
    """
    Identify support and resistance levels via pivot points.

    A pivot high: bar where high[i] > high[i-1] and high[i] > high[i+1].
    A pivot low:  bar where low[i]  < low[i-1]  and low[i]  < low[i+1].

    Returns (support_levels, resistance_levels) as sorted lists.
    """
    n = min(len(highs), len(lows), lookback)
    if n < 3:
        return [], []

    h = highs[-n:]
    l = lows[-n:]

    supports: List[float] = []
    resistances: List[float] = []

    for i in range(1, n - 1):
        # Pivot high → resistance
        if h[i] > h[i - 1] and h[i] > h[i + 1]:
            resistances.append(h[i])
        # Pivot low → support
        if l[i] < l[i - 1] and l[i] < l[i + 1]:
            supports.append(l[i])

    return sorted(set(supports)), sorted(set(resistances))


# ─────────────────────────────────────────────────────────────
# Market Regime Filter
# ─────────────────────────────────────────────────────────────
def get_market_regime(adx_value: float) -> str:
    """Return 'TRENDING' if ADX ≥ threshold, else 'CHOPPY'."""
    return "TRENDING" if adx_value >= ADX_THRESHOLD else "CHOPPY"


# ─────────────────────────────────────────────────────────────
# Composite Score & Signal Generation
# ─────────────────────────────────────────────────────────────
def compute_composite_score(state: MarketState) -> float:
    """
    Weighted scoring model [-1 to +1] (V16 revised).
    Positive = Bullish, Negative = Bearish.
    """
    score = 0.0

    # 1) RSI Contribution (Normalized with Dynamic Thresholds / V19)
    # Widen neutral zone when ATR % is high to prevent false signals on Volatility
    dyn_oversold = max(10, 30 - (state.atr_pct * 5))
    dyn_overbought = min(90, 70 + (state.atr_pct * 5))
    
    if state.rsi < dyn_oversold:
        rsi_norm = -1.0
    elif state.rsi > dyn_overbought:
        rsi_norm = 1.0
    else:
        rsi_norm = (state.rsi - 50) / 50.0

    
    # V16 StochRSI sub-score
    stoch_norm = (state.stoch_rsi - 50) / 50.0
    combined_rsi = (rsi_norm * 0.5) + (stoch_norm * 0.5)
    score -= combined_rsi * WEIGHT_RSI

    # 2) MACD Histogram
    macd_norm = 1.0 if state.macd_histogram > 0 else -1.0 if state.macd_histogram < 0 else 0.0
    score += macd_norm * WEIGHT_MACD

    # 3) Bollinger Bands (-0.5 to +0.5 distance from middle)
    bb_norm = (state.bb_position - 0.5) * 2.0
    score -= bb_norm * WEIGHT_BB

    # 4) Trend Alignment (ADX & Regime)
    if MAKER_GRID_ENABLED:
        # V25: Grid bots thrive in choppy sideways markets (low ADX < 20). 
        # Penalize strong trends to prevent getting run over by a breakout.
        regime_score = {
            "STRONG_UP": -0.5,
            "TRENDING": -0.2,
            "CHOPPY": 0.8,
            "STRONG_DOWN": -0.5,
        }.get(state.regime, 0.0)
    else:
        # Trend strategy prefers strong momentum
        regime_score = {
            "STRONG_UP": 1.0,
            "TRENDING": 0.5,
            "CHOPPY": 0.0,
            "STRONG_DOWN": -1.0,
        }.get(state.regime, 0.0)
    score += regime_score * WEIGHT_TREND

    # 5) VWAP Reversion (Negative weight -> buy if price < VWAP)
    # E.g. price is 2% below vwap -> -2.0% diff -> * (-WEIGHT) = +score
    score -= state.vwap_diff_pct * WEIGHT_VWAP

    # 6) Volume & Sentiment Features
    score += state.social_score * WEIGHT_SOCIAL
    
    # 7) Macro Correlation
    # Penalize altcoins if BTC dominance is high and rising (simplification: static penalty if > 52%)
    if state.btc_dominance > 52.0:
        # If it's highly correlated to BTC, it's slightly safer
        score -= (0.15 * (1.0 - max(0.0, state.btc_correlation))) * WEIGHT_MACRO

    if state.ls_ratio < 0.9:
        score += 0.1  # Shorts crowded -> bullish
    elif state.ls_ratio > 1.1:
        score -= 0.1

    if state.whale_net_vol > WEAK_WHALE_USD:
        score += WEIGHT_WHALE
    elif state.whale_net_vol < -WEAK_WHALE_USD:
        score -= WEIGHT_WHALE

    # Contrarian funding rate sentiment (V18: increased thresholds for stronger signals)
    if state.funding_rate < -0.0005:
        score += 0.2
    elif state.funding_rate > 0.0005:
        score -= 0.2
        
    # V16 Open Interest delta
    if state.open_interest_delta > 0:
        score += state.open_interest_delta * WEIGHT_OI

    # V18 Liquidation Imbalance
    # > 100k net short liquidations -> potential top -> Bearish
    # < -100k net long liquidations -> potential bottom -> Bullish
    if state.liq_imbalance > 100_000:
        score -= 0.15
    elif state.liq_imbalance < -100_000:
        score += 0.15

    # Multi-Timeframe Agreement
    score += state.mtf_agreement * WEIGHT_MTF

    # Divergences
    if state.divergence == "BULLISH_DIV":
        score += WEIGHT_DIV
    elif state.divergence == "BEARISH_DIV":
        score -= WEIGHT_DIV

    logger.info("   \ud83d\udcca Composite Score = %.4f", score)
    return max(-1.0, min(1.0, score))


def _generate_signal_core(state: MarketState, current_position: Optional[Union[str, Dict]] = None, use_composite: bool = USE_COMPOSITE) -> str:
    """
    Decides LONG, SHORT, COVER, SELL, HOLD, DCA_BUY, or DCA_SHORT based on indicators.
    Respects current position dict to choose between opening, closing, or averaging.
    """
    # Helper extracting position details
    if isinstance(current_position, str):
        pos_side = current_position
        pos_entry = 0.0
        pos_dca_count = 0
    else:
        pos_side = current_position["side"] if current_position else None
        pos_entry = current_position["entry"] if current_position else 0.0
        pos_dca_count = current_position.get("dca_count", 0) if current_position else 0

    # 1. Social overrides (Highest Precedence)
    if NEWS_OVERRIDE_RSI and state.social_score > 0.5 and state.rsi > 70:
        logger.info("🚀 BULLISH SOCIAL override: Ignoring Overbought signal (RSI=%.2f).", state.rsi)
        return "HOLD"
    if NEWS_OVERRIDE_RSI and state.social_score < -0.5 and state.rsi < 30:
        logger.info("⚠️ BEARISH SOCIAL override: Ignoring Oversold signal (RSI=%.2f).", state.rsi)
        return "HOLD"

    # 2. V20 DCA Isolation (Risk Management preempts momentum analysis)
    #    V22: DCA Safety Guard — block DCA if position drawdown exceeds threshold
    if pos_side == "BUY" and pos_dca_count < MAX_DCA_STEPS and pos_entry > 0:
        drop_pct = (pos_entry - state.current_price) / pos_entry * 100
        if drop_pct >= DCA_MAX_DRAWDOWN_PCT:
            logger.info("🛑 DCA BLOCKED: drawdown %.1f%% exceeds DCA limit %.1f%% → forcing SELL", drop_pct, DCA_MAX_DRAWDOWN_PCT)
            return "SELL"
        if drop_pct >= DCA_PRICE_DROP_PCT:
            return "DCA_BUY"
            
    if pos_side == "SELL" and pos_dca_count < MAX_DCA_STEPS and pos_entry > 0:
        rise_pct = (state.current_price - pos_entry) / pos_entry * 100
        if rise_pct >= DCA_MAX_DRAWDOWN_PCT:
            logger.info("🛑 DCA BLOCKED: drawdown %.1f%% exceeds DCA limit %.1f%% → forcing COVER", rise_pct, DCA_MAX_DRAWDOWN_PCT)
            return "COVER"
        if rise_pct >= DCA_PRICE_DROP_PCT:
            return "DCA_SHORT"

    # 3. Strategy: Composite Score (Primary Momentum)
    if use_composite:
        score = compute_composite_score(state)
        
        # Bullish signal
        # V22: Relaxed rsi_slope from >0 to >=-2 to catch early reversals
        # V26: Grid bots WANT to trade near S/R levels (bounce zones)
        if score > COMPOSITE_BUY_THRESHOLD and (MAKER_GRID_ENABLED or not state.near_resistance):
            if not VOLUME_CONFIRM_ENABLED or state.volume_confirm:
                if state.rsi_slope >= -2:
                    if pos_side is None:
                        return "BUY"
                    elif pos_side == "SELL":
                        return "COVER"
        
        # Bearish signal
        # V22: Added rsi_slope symmetry check for SHORT entries
        if score < COMPOSITE_SELL_THRESHOLD:
            if state.rsi_slope <= 2:
                if pos_side is None:
                    return "SHORT"
                elif pos_side == "BUY":
                    if pos_dca_count == 0: 
                        return "SELL"
                    else:
                        return "HOLD"

    # 4. Exit logic based on absolute RSI levels (fallback safeguard)
    # V20: Also panic-sell blocked. We exit via TP/SL in execution.py if averaging.
    if pos_side == "BUY" and state.rsi > 70 + (state.atr_pct * 5) and pos_dca_count == 0:
        return "SELL"
    if pos_side == "SELL" and state.rsi < 30 - (state.atr_pct * 5) and pos_dca_count == 0:
        return "COVER"

    # 5. Entry only in trending/choppy market based on config (fallback basic strategy)
    if pos_side is None:
        valid_regime = "CHOPPY" if MAKER_GRID_ENABLED else "TRENDING"
        if state.regime == valid_regime:
            if state.rsi < RSI_OVERSOLD and state.ls_ratio < LS_RATIO_THRESHOLD and state.whale_net_vol > WHALE_NET_VOL_MIN:
                return "BUY"
            if state.rsi > RSI_OVERBOUGHT and state.ls_ratio > 1.0 and state.whale_net_vol < -WHALE_NET_VOL_MIN:
                return "SHORT"

    return "HOLD"

def generate_signal(
    state: MarketState, 
    current_position: Optional[Union[str, Dict]] = None, 
    use_composite: bool = USE_COMPOSITE,
    detailed: bool = False
) -> Union[str, Dict[str, Any]]:
    """Wrapper that computes score and confidence for structured reporting."""
    sig = _generate_signal_core(state, current_position, use_composite)
    
    if not detailed:
        return sig
        
    score = compute_composite_score(state) if use_composite else 0.0
    
    # Calculate Confidence (50% at threshold, up to 100% at score 1.2)
    abs_s = abs(score)
    thresh = COMPOSITE_BUY_THRESHOLD
    
    if abs_s < thresh:
        conf = 0.0
    else:
        mapped = ((abs_s - thresh) / (1.2 - thresh)) * 50.0 + 50.0
        conf = min(100.0, max(50.0, float(mapped)))
        
    # Overrides are very confident
    if sig in ["DCA_BUY", "DCA_SHORT"]:
        conf = 100.0
    if sig == "HOLD" and abs_s > thresh:
        # If score is high but we hold, it's an override block (e.g. social or chopping)
        conf = 90.0

    return {
        "action": sig,
        "confidence": conf,
        "composite_score": score
    }
