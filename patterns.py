"""
Hunter V36 — Chart Pattern Detection Engine
=============================================
Adapted from HKUDS/Vibe-Trading tools/pattern_tool.py (MIT License).

Pure-numpy implementation (no pandas dependency) for detecting:
  - Peaks/Valleys
  - Candlestick patterns (doji, hammer, engulfing)
  - Support/Resistance levels via clustering
  - Trend line slope (rolling OLS)
  - Head & Shoulders
  - Double Top / Bottom
  - Triangle patterns (ascending/descending)
  - Broadening (megaphone)
  - Ichimoku Cloud system

All functions work with List[float] and return simple int signals.
Signal convention: -1 = bearish, 0 = neutral, 1 = bullish.
"""

import logging
import math
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger("hunter.patterns")


# ─────────────────────────────────────────────────────────────
# Peaks & Valleys
# ─────────────────────────────────────────────────────────────

def find_peaks_valleys(values: List[float], window: int = 5) -> Dict[str, List[int]]:
    """
    Detect peaks and valleys in a price series.

    Args:
        values: Price series.
        window: Half-window size; effective window is 2*window+1.

    Returns:
        Dict with "peaks" and "valleys", each a list of integer indices.
    """
    n = len(values)
    if n < 2 * window + 1:
        return {"peaks": [], "valleys": []}

    arr = np.array(values, dtype=np.float64)
    peaks, valleys = [], []

    for i in range(window, n - window):
        if np.isnan(arr[i]):
            continue
        seg = arr[i - window: i + window + 1]
        seg_valid = seg[~np.isnan(seg)]
        if len(seg_valid) == 0:
            continue
        if arr[i] >= np.max(seg_valid):
            peaks.append(i)
        if arr[i] <= np.min(seg_valid):
            valleys.append(i)

    return {"peaks": peaks, "valleys": valleys}


# ─────────────────────────────────────────────────────────────
# Candlestick Patterns
# ─────────────────────────────────────────────────────────────

def detect_candlestick(
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
) -> int:
    """
    Detect latest candlestick pattern: doji, hammer, engulfing.

    Returns:
        -1 (bearish engulfing), 0 (neutral/doji), 1 (bullish hammer/engulfing).
    """
    if len(closes) < 2:
        return 0

    o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
    po, pc = opens[-2], closes[-2]

    body = abs(c - o)
    total_range = h - l
    if total_range == 0:
        return 0

    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l

    # Doji: body < 10% of range
    if body / total_range < 0.10:
        return 0

    # Hammer: lower shadow > 2x body, upper shadow < body
    if lower_shadow > 2 * body and upper_shadow < body:
        return 1

    # Bullish engulfing: prev bearish, current bullish, current body > prev body
    prev_body = abs(pc - po)
    if pc < po and c > o and o <= pc and c >= po and body > prev_body:
        return 1

    # Bearish engulfing: prev bullish, current bearish
    if pc > po and c < o and o >= pc and c <= po and body > prev_body:
        return -1

    return 0


# ─────────────────────────────────────────────────────────────
# Support & Resistance (Clustering)
# ─────────────────────────────────────────────────────────────

def compute_support_resistance_clustered(
    closes: List[float],
    window: int = 20,
    num_levels: int = 3,
) -> Dict[str, List[float]]:
    """
    Compute support and resistance levels via peak/valley clustering.

    Args:
        closes: Closing price series.
        window: Peak/valley detection window.
        num_levels: Maximum number of levels to return.

    Returns:
        Dict with "support" and "resistance", each a list of price levels.
    """
    pv = find_peaks_valleys(closes, window=window)
    arr = np.array(closes, dtype=np.float64)

    peak_prices = [float(arr[i]) for i in pv["peaks"] if not np.isnan(arr[i])]
    valley_prices = [float(arr[i]) for i in pv["valleys"] if not np.isnan(arr[i])]

    def cluster(prices: List[float], n: int) -> List[float]:
        if not prices:
            return []
        sp = sorted(prices)
        if len(sp) <= n:
            return sp
        clusters: list = [[sp[0]]]
        rng = sp[-1] - sp[0]
        thr = rng * 0.05 if rng > 0 else 1.0
        for p in sp[1:]:
            if abs(p - np.mean(clusters[-1])) <= thr:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        centers = [(len(c), float(np.mean(c))) for c in clusters]
        centers.sort(reverse=True)
        return [round(c, 4) for _, c in centers[:n]]

    return {
        "support": cluster(valley_prices, num_levels),
        "resistance": cluster(peak_prices, num_levels),
    }


# ─────────────────────────────────────────────────────────────
# Trend Line Slope (Rolling OLS)
# ─────────────────────────────────────────────────────────────

def compute_trend_slope(closes: List[float], window: int = 20) -> float:
    """
    Compute the latest rolling linear-fit slope.

    Args:
        closes: Closing price series.
        window: Fitting window size.

    Returns:
        Slope value; positive = uptrend, negative = downtrend.
    """
    if len(closes) < window:
        return 0.0
    seg = np.array(closes[-window:], dtype=np.float64)
    if np.any(np.isnan(seg)):
        return 0.0
    x = np.arange(window, dtype=np.float64)
    coeffs = np.polyfit(x, seg, 1)
    return float(coeffs[0])


# ─────────────────────────────────────────────────────────────
# Head & Shoulders
# ─────────────────────────────────────────────────────────────

def detect_head_shoulders(closes: List[float], window: int = 10) -> int:
    """
    Detect head-and-shoulders top pattern.

    Returns:
        1 if H&S top detected (bearish reversal signal), 0 otherwise.
    """
    pv = find_peaks_valleys(closes, window=window)
    peaks = pv["peaks"]
    arr = np.array(closes, dtype=np.float64)

    if len(peaks) < 3:
        return 0

    # Check last three peaks
    for i in range(len(peaks) - 3, len(peaks) - 2):
        if i < 0:
            continue
        lv = arr[peaks[i]]
        hv = arr[peaks[i + 1]]
        rv = arr[peaks[i + 2]]

        if any(np.isnan(x) for x in (lv, hv, rv)):
            continue

        # Head must be higher than both shoulders
        if hv <= lv or hv <= rv:
            continue

        # Shoulders should be roughly symmetric (within 5%)
        avg = (lv + rv) / 2
        if avg == 0 or abs(lv - rv) / avg > 0.05:
            continue

        return 1

    return 0


# ─────────────────────────────────────────────────────────────
# Double Top / Bottom
# ─────────────────────────────────────────────────────────────

def detect_double_top_bottom(closes: List[float], window: int = 10) -> int:
    """
    Detect double-top and double-bottom patterns.

    Returns:
        1 (double top - bearish), -1 (double bottom - bullish), or 0 (none).
    """
    pv = find_peaks_valleys(closes, window=window)
    arr = np.array(closes, dtype=np.float64)

    # Check for double top (last two peaks at similar level)
    if len(pv["peaks"]) >= 2:
        i = len(pv["peaks"]) - 2
        v1, v2 = arr[pv["peaks"][i]], arr[pv["peaks"][i + 1]]
        if not (np.isnan(v1) or np.isnan(v2)):
            avg = (v1 + v2) / 2
            if avg != 0 and abs(v1 - v2) / avg < 0.03:
                return 1  # double top - bearish

    # Check for double bottom (last two valleys at similar level)
    if len(pv["valleys"]) >= 2:
        i = len(pv["valleys"]) - 2
        v1, v2 = arr[pv["valleys"][i]], arr[pv["valleys"][i + 1]]
        if not (np.isnan(v1) or np.isnan(v2)):
            avg = (v1 + v2) / 2
            if avg != 0 and abs(v1 - v2) / abs(avg) < 0.03:
                return -1  # double bottom - bullish

    return 0


# ─────────────────────────────────────────────────────────────
# Triangle Detection
# ─────────────────────────────────────────────────────────────

def detect_triangle(closes: List[float], window: int = 20) -> int:
    """
    Detect ascending/descending triangle at the end of the series.

    Returns:
        1 (ascending triangle - bullish), -1 (descending - bearish), 0 (none).
    """
    if len(closes) < window + 1:
        return 0

    seg = closes[-window - 1:]
    sub_window = max(2, window // 5)
    pv = find_peaks_valleys(seg, window=sub_window)

    if len(pv["peaks"]) < 2 or len(pv["valleys"]) < 2:
        return 0

    pvals = [seg[p] for p in pv["peaks"]]
    vvals = [seg[v] for v in pv["valleys"]]

    # Fit slopes to peaks and valleys
    ps = np.polyfit(np.arange(len(pvals), dtype=float), pvals, 1)[0] if len(pvals) >= 2 else 0.0
    vs = np.polyfit(np.arange(len(vvals), dtype=float), vvals, 1)[0] if len(vvals) >= 2 else 0.0

    rng = max(pvals) - min(vvals)
    if rng == 0:
        return 0
    flat = rng * 0.02

    if vs > flat and abs(ps) < flat:
        return 1   # ascending triangle
    elif ps < -flat and abs(vs) < flat:
        return -1  # descending triangle

    return 0


# ─────────────────────────────────────────────────────────────
# Broadening (Megaphone)
# ─────────────────────────────────────────────────────────────

def detect_broadening(closes: List[float], window: int = 20) -> int:
    """
    Detect broadening (megaphone) pattern — increasing volatility.

    Returns:
        1 if broadening detected (warns of instability), 0 otherwise.
    """
    if len(closes) < window + 1:
        return 0

    seg = closes[-window - 1:]
    sub_window = max(2, window // 5)
    pv = find_peaks_valleys(seg, window=sub_window)

    if len(pv["peaks"]) < 2 or len(pv["valleys"]) < 2:
        return 0

    pvals = [seg[p] for p in pv["peaks"]]
    vvals = [seg[v] for v in pv["valleys"]]

    peaks_rising = all(pvals[j + 1] > pvals[j] for j in range(len(pvals) - 1))
    valleys_falling = all(vvals[j + 1] < vvals[j] for j in range(len(vvals) - 1))

    if peaks_rising and valleys_falling:
        return 1
    return 0


# ─────────────────────────────────────────────────────────────
# Ichimoku Cloud System
# ─────────────────────────────────────────────────────────────

def compute_ichimoku(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
) -> Dict[str, Optional[float]]:
    """
    Compute Ichimoku Cloud components for the latest bar.

    Warm-up requires senkou_b_period + kijun_period bars (78).

    Returns:
        Dict with tenkan, kijun, senkou_a, senkou_b, chikou, signal.
        signal: 1 = strong buy, -1 = strong sell, 0 = neutral.
    """
    n = len(closes)
    warmup = senkou_b_period + kijun_period
    if n < warmup:
        return {"tenkan": None, "kijun": None, "senkou_a": None,
                "senkou_b": None, "chikou": None, "signal": 0}

    h = np.array(highs, dtype=np.float64)
    l = np.array(lows, dtype=np.float64)
    c = np.array(closes, dtype=np.float64)

    def midpoint(arr, period, idx):
        seg = arr[max(0, idx - period + 1): idx + 1]
        return (np.max(seg) + np.min(seg)) / 2

    i = n - 1
    tenkan = midpoint(h, tenkan_period, i) / 2 + midpoint(l, tenkan_period, i) / 2
    # Correction: tenkan should be (highest high + lowest low) / 2 over the period
    tenkan = (np.max(h[i - tenkan_period + 1:i + 1]) + np.min(l[i - tenkan_period + 1:i + 1])) / 2
    kijun = (np.max(h[i - kijun_period + 1:i + 1]) + np.min(l[i - kijun_period + 1:i + 1])) / 2

    senkou_a = (tenkan + kijun) / 2
    senkou_b = (np.max(h[i - senkou_b_period + 1:i + 1]) + np.min(l[i - senkou_b_period + 1:i + 1])) / 2

    chikou = c[i]  # Will be compared to price 26 bars ago
    price = c[i]

    # Signal logic: TK cross + cloud position + cloud color
    prev_tenkan = (np.max(h[i - tenkan_period:i]) + np.min(l[i - tenkan_period:i])) / 2
    prev_kijun = (np.max(h[i - kijun_period:i]) + np.min(l[i - kijun_period:i])) / 2

    tk_cross_bull = tenkan > kijun and prev_tenkan <= prev_kijun
    tk_cross_bear = tenkan < kijun and prev_tenkan >= prev_kijun

    cloud_top = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)
    above_cloud = price > cloud_top
    below_cloud = price < cloud_bottom
    bullish_cloud = senkou_a > senkou_b

    signal = 0
    if tk_cross_bull and above_cloud and bullish_cloud:
        signal = 1   # Strong buy
    elif tk_cross_bear and below_cloud and not bullish_cloud:
        signal = -1  # Strong sell

    return {
        "tenkan": round(float(tenkan), 4),
        "kijun": round(float(kijun), 4),
        "senkou_a": round(float(senkou_a), 4),
        "senkou_b": round(float(senkou_b), 4),
        "chikou": round(float(chikou), 4),
        "signal": signal,
    }


# ─────────────────────────────────────────────────────────────
# Composite Pattern Signal
# ─────────────────────────────────────────────────────────────

def get_pattern_signal(
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
) -> Dict[str, any]:
    """
    Run all pattern detectors and return a composite signal.

    Returns:
        Dict with individual signals and composite value.
    """
    candlestick = detect_candlestick(opens, highs, lows, closes)
    h_and_s = detect_head_shoulders(closes, window=10)
    double = detect_double_top_bottom(closes, window=10)
    triangle = detect_triangle(closes, window=20)
    broadening = detect_broadening(closes, window=20)
    ichimoku = compute_ichimoku(highs, lows, closes)

    # Weighted composite: structural patterns > candlestick
    composite = 0.0
    composite += candlestick * 0.15       # Candlestick weight
    composite += (-h_and_s) * 0.25        # H&S is bearish reversal
    composite += (-double) * 0.20         # Double top = bearish, double bottom = bullish (negated)
    composite += triangle * 0.20          # Triangle direction
    composite += ichimoku["signal"] * 0.20  # Ichimoku signal

    # Clamp to [-1, 1]
    composite = max(-1.0, min(1.0, composite))

    return {
        "candlestick": candlestick,
        "head_and_shoulders": h_and_s,
        "double_top_bottom": double,
        "triangle": triangle,
        "broadening": broadening,
        "ichimoku": ichimoku["signal"],
        "composite": round(composite, 4),
    }
