"""
Hunter V22 — Configuration Constants
=====================================
Central place for every tunable parameter.

V22: Increased exposure, lower composite thresholds, drawdown guards,
     .env-based API key loading.
V16: Persistent positions, fees/slippage, position caps, Kelly, StochRSI,
     RSI slope filter, Open Interest, ADX-guard divergence, S/R signal filter.
"""

# ── Trading defaults ─────────────────────────────────────────
SYMBOL = "BTCUSDT"           # Default / fallback symbol
TIMEFRAME = "15m"            # V26: 15m for Grid bots (4× more signals than 1h)
KLINE_LIMIT = 200            # V16: 200 bars (was 100) — enough for all indicators

# ── Binance Futures API ─────────────────────────────────────
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, fall back to hardcoded values

LIVE_TRADING = str(os.getenv("LIVE_TRADING", "True")).strip().lower() in ("true", "1", "t", "yes")
USE_TESTNET = str(os.getenv("USE_TESTNET", "True")).strip().lower() in ("true", "1", "t", "yes")

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

if LIVE_TRADING and USE_TESTNET:
    BASE_URL = "https://testnet.binancefuture.com"
else:
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
TOP_PAIRS_COUNT = 50         # How many symbols to scan in Auto mode

# ── ADX  (Market Regime Filter) ──────────────────────────────
ADX_PERIOD = 14
ADX_THRESHOLD = 20           # < 20 → CHOPPY (no trade), ≥ 20 → TRENDING (lowered from 25 for aggression)
ADX_STRONG_TREND = 40        # V16: ADX > 40 → suppress divergence signals

# ── RSI ──────────────────────────────────────────────────────
RSI_PERIOD = 14
RSI_OVERSOLD = 30            # Buy zone
RSI_OVERBOUGHT = 70          # Sell zone
RSI_SLOPE_BARS = 5           # V16: bars for RSI slope calculation

# ── Stochastic RSI (V16) ────────────────────────────────────
STOCH_RSI_PERIOD = 14        # StochRSI lookback period

# ── Bollinger Bands ──────────────────────────────────────────
BB_PERIOD = 20
BB_STD = 2

# ── MACD (V14) ──────────────────────────────────────────────
MACD_FAST = 12               # Fast EMA period
MACD_SLOW = 26               # Slow EMA period
MACD_SIGNAL = 9              # Signal line EMA period

# ── ATR & Dynamic TP (V14/V24) ────────────────────────────────
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 2.0      # V22: Stop-Loss = entry ∓ ATR × 2.0 (wider for noise immunity)
ATR_TP_MULTIPLIER = 3.5      # V22: Take-Profit = entry ± ATR × 3.5 (R:R = 1:1.75)
DYNAMIC_TP_ENABLED = True    # V24: Adjust Take-Profit automatically based on volatility
DYNAMIC_TP_MAX_MULT = 10.0   # V24: Maximum ATR multiplier for dynamic TP

# ── Trailing Stop-Loss (V15) ─────────────────────────────────
TRAILING_SL_ENABLED = True
TRAILING_SL_ATR_MULT = 1.5
TRAILING_SL_ACTIVATION_PCT = 0.5

# ── DCA / Grid Parameters (V19) ──────────────────────────────
MAX_DCA_STEPS = 3             # Max averaging entries
DCA_PRICE_DROP_PCT = 2.0      # Minimum percent drop/rise for next DCA

# ── Short Positions (V15) ────────────────────────────────────
SHORT_ENABLED = True

# ── Multi-Timeframe (V14) ────────────────────────────────────
MULTI_TF_INTERVALS = ["15m", "1h", "4h"]
MTF_AGREEMENT_MIN = 2

# ── Volume Confirmation (V16) ────────────────────────────────
VOLUME_CONFIRM_ENABLED = False # V26: Disabled for Grid mode (grid doesn't need trend volume)
VOLUME_CONFIRM_BARS = 3        # Compare last N bars vs previous period

# ── Position Sizing & Risk (V16 / V24 Auto-Compounding) ───────
TRADE_SIZE_USD = 100           # Fallback static size
INITIAL_BALANCE_USD = 10_000   # Default balance for PaperTrading
USE_DYNAMIC_SIZING = True      # V24: Use actual balance for trade sizing
MAX_RISK_PER_TRADE_PCT = 10.0  # V25: Hard cap: risk maximum 10% of total balance per trade
MAX_OPEN_POSITIONS = 1         # V22: single position = full focus for $100 capital
MAX_EXPOSURE_USD = 3000        # V22: 3000 (was 300) — realistic exposure
LEVERAGE = 5                   # V22: 5× leverage for small capital amplification

# ── Drawdown Protection (V22) ─────────────────────────────────
MAX_DRAWDOWN_PCT = 5.0         # V22: halt ALL trading if balance drops 5% from initial
DCA_MAX_DRAWDOWN_PCT = 3.0     # V22: block DCA if single position drawdown > 3%

# ── ML Signal Filter (V22 / Phase 6) ─────────────────────────
ML_ENABLED = True              # Enable ML signal filtering
ML_CONFIDENCE_THRESHOLD = 0.51 # V25: Lowered threshold for statistical edge without massive overlap reduction

# ── Institutional Execution (V23/V24 Maker Grid) ─────────────
LIMIT_ORDER_TIMEOUT_SEC = 120  # V25: Wait up to 120s for Limit order fill before cancel (partial fill safety)
MAKER_GRID_ENABLED = True      # V24: Place a grid of Maker bids instead of a single order
GRID_ORDERS_COUNT = 3          # V24: Number of orders in the grid
GRID_SPREAD_PCT = 0.35         # V25: Min step = %0.35 to cover round trip maker fees AND slippage

# ── Trading Costs (V16) ──────────────────────────────────────
TAKER_FEE = 0.0004            # Binance Futures taker fee = 0.04%
MAKER_FEE = 0.001             # Standard fee = 0.1%
MAKER_FEE_BNB = 0.00075       # V25: 25% discount via BNB (=0.075%)
SLIPPAGE = 0.0005             # Estimated slippage = 0.05% per fill
# Total round-trip cost = (TAKER_FEE + SLIPPAGE) × 2 × size_usd

# ── Dynamic Grid Math (V25) ──────────────────────────────────
# Profit per grid slice = Grid Spread % - (Commission % * 2)
_grid_profit = (GRID_SPREAD_PCT / 100.0) - (MAKER_FEE_BNB * 2)
if _grid_profit <= 0:
    raise ValueError(f"FATAL: GRID_SPREAD_PCT ({GRID_SPREAD_PCT}%) doesn't cover Maker fees (Profit: {_grid_profit*100:.3f}%).")
if GRID_SPREAD_PCT <= 0.20:
    raise ValueError(f"FATAL: GRID_SPREAD_PCT must be strictly > 0.20% bounds for slippage. Provided: {GRID_SPREAD_PCT}%")

# ── Kelly Criterion (V16) ────────────────────────────────────
KELLY_ENABLED = True           # Use Kelly to size positions dynamically
KELLY_FRACTION = 0.25          # V25: Conservative 1/4 Kelly model
KELLY_MAX_PCT = 0.10           # V25: Hard cap at 10% of balance per trade
KELLY_MIN_TRADES = 10          # Min trades before Kelly activates

# ── Contrarian Signal Thresholds ─────────────────────────────
LS_RATIO_THRESHOLD = 0.8
WHALE_NET_VOL_MIN = 0

# ── Composite Signal Weights (V16) ───────────────────────────
WEIGHT_RSI       = 0.20
WEIGHT_MACD      = 0.20
WEIGHT_BOLLINGER = 0.15
WEIGHT_FUNDING   = 0.10
WEIGHT_MTF       = 0.10
WEIGHT_SOCIAL    = 0.10
WEIGHT_LS_RATIO  = 0.05
WEIGHT_MACRO     = 0.10
WEIGHT_WHALE     = 0.00

# Composite thresholds
COMPOSITE_BUY_THRESHOLD  =  0.10      # V26: lowered from 0.15 for grid frequency
COMPOSITE_SELL_THRESHOLD = -0.10      # V26: lowered from -0.15

# ── Circuit Breaker ──────────────────────────────────────────
MAX_CONSECUTIVE_LOSSES = 3
COOLDOWN_HOURS = 24

# ── Infrastructure ───────────────────────────────────────────
DB_PATH = "hunter.db"
CHECK_INTERVAL_SEC = 300

# ── News Sentiment Layer (V13) ───────────────────────────────
CRYPTOPANIC_API_KEY = "c12917daf988840356e4bd7f5bacfb060a6e9353"
NEWS_OVERRIDE_RSI = True
NEWS_POLL_INTERVAL_SEC = 900

# ── Support & Resistance (V14) ───────────────────────────────
SR_LOOKBACK = 50
SR_PROXIMITY_PCT = 0.5       # V16: used to block BUY near resistance

# ── VWAP (V16) ───────────────────────────────────────────────
VWAP_BARS = 24               # V16: use last 24 bars (≈1 trading day) for VWAP
