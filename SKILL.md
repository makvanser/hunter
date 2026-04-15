---
name: hunter
description: >
  Institutional-grade crypto futures bot for Binance with SMC analysis,
  chart pattern detection, statistical validation, VPIN microstructure,
  funding regime intelligence, and ML filtering (24 features).
  Optimized for small capital ($100+), maker-only execution.
category: strategy
---

# Hunter Trading Bot (V36)

## Purpose

Autonomous crypto futures trading on Binance with an institutional-grade analytics stack adapted from [HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading).

## Core Capabilities

| Module | Source | Function |
|--------|--------|----------|
| `validation.py` | Vibe-Trading `quant-statistics` | Monte Carlo permutation test, Bootstrap Sharpe CI, Walk-Forward consistency |
| `patterns.py` | Vibe-Trading `harmonic` + `ichimoku` | H&S, double top/bottom, triangles, candlestick, Ichimoku 5-line system |
| `smc.py` | Vibe-Trading `smc` | BOS/ChoCH/FVG/Order Blocks (via smartmoneyconcepts) |
| `provider.py` (VPIN) | Vibe-Trading `market-microstructure` | Volume-Sync Probability of Informed Trading |
| `funding_arb.py` | Vibe-Trading `perp-funding-basis` | Funding regime detection, OI×Funding matrix signal |
| `optimizer.py` | Vibe-Trading `execution-model` | Almgren-Chriss sqrt-impact slippage model |
| `ml.py` | Vibe-Trading `factor-research` | 24-feature ML filter with VPIN, pattern, and momentum features |
| `strategy_router.py` | — | 7-strategy ensemble: Grid, Momentum, MeanRev, FundingArb, StatArb, Pattern, SMC |

## Signal Convention

- `BUY` / `SHORT` = directional entry (LIMIT maker order)
- `HOLD` = no action
- `SELL` / `COVER` = exit existing position

## Dependencies

```bash
pip install aiohttp python-dotenv scikit-learn numpy
# Optional (SMC patterns):
pip install smartmoneyconcepts pandas
# Optional (GARCH volatility):
pip install statsmodels arch
```

## Parameters

All parameters are in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRADE_SIZE_USD` | 10.0 | Position size |
| `ATR_SL_MULTIPLIER` | 1.5 | Stop-loss × ATR |
| `ATR_TP_MULTIPLIER` | 2.5 | Take-profit × ATR |
| `MAX_DRAWDOWN_PCT` | 15.0 | Global drawdown circuit breaker |
| `LIVE_TRADING` | False | Paper vs live trading |
| `USE_TESTNET` | True | Testnet vs mainnet |

## Environment

```env
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
LIVE_TRADING=True
USE_TESTNET=False
```

## Usage

```bash
# 1. Train ML model
python backtest.py --train-ml

# 2. Run optimizer (validates parameters statistically)
python optimizer.py --window 30

# 3. Start bot
nohup python3 main.py > bot_live.log 2>&1 &
tail -f bot_live.log
```

## Architectural Rules

- **MAKER ONLY**: All execution uses `LIMIT` orders (`GTX` / Post Only)
- **No pandas in hot path**: Core analysis is pure numpy
- **Zero-Trust Balance**: All risk uses `sync_balance()` from exchange
- **Drawdown Protection**: Circuit breakers are immutable
- **Microstructure First**: OBI + VPIN gate before every entry
