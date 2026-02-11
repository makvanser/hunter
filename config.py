"""
Hunter V12 — Configuration Constants
=====================================
Central place for every tunable parameter.
"""

# ── Trading defaults ─────────────────────────────────────────
SYMBOL = "BTCUSDT"           # Default / fallback symbol
TIMEFRAME = "1h"             # Kline interval
KLINE_LIMIT = 100            # Number of candles to fetch

# ── Binance Futures API ─────────────────────────────────────
BASE_URL = "https://fapi.binance.com"

# ── Stablecoin Blacklist (never trade these) ─────────────────
BLACKLIST = [
    "USDCUSDT",
    "USDPUSDT",
    "DAIUSDT",
    "BUSDUSDT",
    "TUSDUSDT",
    "FDUSDUSDT",
    "EURUSDT",
]

# ── Top-pair scanner ─────────────────────────────────────────
TOP_PAIRS_COUNT = 10         # How many symbols to scan in Auto mode

# ── ADX  (Market Regime Filter) ──────────────────────────────
ADX_PERIOD = 14
ADX_THRESHOLD = 25           # < 25 → CHOPPY (no trade), ≥ 25 → TRENDING

# ── RSI ──────────────────────────────────────────────────────
RSI_PERIOD = 14
RSI_OVERSOLD = 30            # Buy zone

# ── Bollinger Bands ──────────────────────────────────────────
BB_PERIOD = 20
BB_STD = 2

# ── Contrarian Signal Thresholds ─────────────────────────────
LS_RATIO_THRESHOLD = 0.8     # Long/Short ratio must be below this to buy
WHALE_NET_VOL_MIN = 0        # Whale net volume must be > 0 to buy

# ── Circuit Breaker ──────────────────────────────────────────
MAX_CONSECUTIVE_LOSSES = 3
COOLDOWN_HOURS = 24

# ── Paper Trading ────────────────────────────────────────────
TRADE_SIZE_USD = 100
INITIAL_BALANCE_USD = 10_000

# ── Infrastructure ───────────────────────────────────────────
DB_PATH = "hunter.db"
CHECK_INTERVAL_SEC = 300     # 5 minutes
